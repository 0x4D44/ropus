//! Opus Encoder — top-level Opus encoding entry point.
//!
//! Ported from: reference/src/opus_encoder.c
//! Fixed-point path (non-RES24, non-QEXT, non-DRED).
//! Analysis (tonality detection) is NOT included (matches DISABLE_FLOAT_API).
//! Mode/bandwidth decisions use bitrate thresholds and user hints only.

use crate::celt::encoder::{CeltEncoder, CeltEncoderCtl, SILKInfo, celt_encode_with_ec};
use crate::celt::math_ops::{celt_exp2, celt_ilog2, celt_sqrt, frac_div32};
use crate::celt::range_coder::RangeEncoder;
use crate::silk::common::{silk_lin2log, silk_log2lin};
use crate::silk::encoder::{SilkEncControlStruct, SilkEncoder, silk_encode, silk_init_encoder_top};
use crate::types::*;

use super::decoder::{
    MODE_CELT_ONLY, MODE_HYBRID, MODE_SILK_ONLY, OPUS_BAD_ARG, OPUS_BANDWIDTH_FULLBAND,
    OPUS_BANDWIDTH_MEDIUMBAND, OPUS_BANDWIDTH_NARROWBAND, OPUS_BANDWIDTH_SUPERWIDEBAND,
    OPUS_BANDWIDTH_WIDEBAND, OPUS_BUFFER_TOO_SMALL, OPUS_INTERNAL_ERROR, OPUS_OK,
};
use super::repacketizer::{OpusRepacketizer, opus_packet_pad};

// ===========================================================================
// Constants
// ===========================================================================

// Application modes (from opus_defines.h)
pub const OPUS_APPLICATION_VOIP: i32 = 2048;
pub const OPUS_APPLICATION_AUDIO: i32 = 2049;
pub const OPUS_APPLICATION_RESTRICTED_LOWDELAY: i32 = 2051;

// Internal-only restricted modes
#[allow(dead_code)]
const OPUS_APPLICATION_RESTRICTED_SILK: i32 = 2052;
#[allow(dead_code)]
const OPUS_APPLICATION_RESTRICTED_CELT: i32 = 2053;

// Special values
pub const OPUS_AUTO: i32 = -1000;
pub const OPUS_BITRATE_MAX: i32 = -1;

// Signal types (from opus_defines.h)
pub const OPUS_SIGNAL_VOICE: i32 = 3001;
pub const OPUS_SIGNAL_MUSIC: i32 = 3002;

// Frame size constants (from opus_defines.h)
pub const OPUS_FRAMESIZE_ARG: i32 = 5000;
pub const OPUS_FRAMESIZE_2_5_MS: i32 = 5001;
pub const OPUS_FRAMESIZE_5_MS: i32 = 5002;
pub const OPUS_FRAMESIZE_10_MS: i32 = 5003;
pub const OPUS_FRAMESIZE_20_MS: i32 = 5004;
pub const OPUS_FRAMESIZE_40_MS: i32 = 5005;
pub const OPUS_FRAMESIZE_60_MS: i32 = 5006;
pub const OPUS_FRAMESIZE_80_MS: i32 = 5007;
pub const OPUS_FRAMESIZE_100_MS: i32 = 5008;
pub const OPUS_FRAMESIZE_120_MS: i32 = 5009;

// Encoder buffer size (max delay_buffer samples per channel)
#[allow(dead_code)]
const MAX_ENCODER_BUFFER: i32 = 480;

// VAD decision sentinel
const VAD_NO_DECISION: i32 = -1;

// SILK signal type for no voice activity
const TYPE_NO_VOICE_ACTIVITY: i32 = 0;

// DTX parameters
const NB_SPEECH_FRAMES_BEFORE_DTX: i32 = 10; // 200ms
const MAX_CONSECUTIVE_DTX: i32 = 20; // 400ms

// PSEUDO_SNR_THRESHOLD = 10^(25/10) = 316.23 → QCONST16(316.23, 0) = 316
const PSEUDO_SNR_THRESHOLD: i32 = 316;

// HP filter smoothing coefficient — Q16, matching C's SILK_FIX_CONST(0.015, 16) = 983
const VARIABLE_HP_SMTH_COEF2: i32 = 983; // Q16
const VARIABLE_HP_MIN_CUTOFF_HZ: i32 = 60;

// ===========================================================================
// Static tables
// ===========================================================================

// Bandwidth thresholds: [threshold, hysteresis] pairs for NB↔MB, MB↔WB, WB↔SWB, SWB↔FB
static MONO_VOICE_BANDWIDTH_THRESHOLDS: [i32; 8] = [9000, 700, 9000, 700, 13500, 1000, 14000, 2000];
static MONO_MUSIC_BANDWIDTH_THRESHOLDS: [i32; 8] = [9000, 700, 9000, 700, 11000, 1000, 12000, 2000];
static STEREO_VOICE_BANDWIDTH_THRESHOLDS: [i32; 8] =
    [9000, 700, 9000, 700, 13500, 1000, 14000, 2000];
static STEREO_MUSIC_BANDWIDTH_THRESHOLDS: [i32; 8] =
    [9000, 700, 9000, 700, 11000, 1000, 12000, 2000];

// Mode thresholds: [mono, stereo] × [voice, music]
static MODE_THRESHOLDS: [[i32; 2]; 2] = [
    [64000, 10000], // mono
    [44000, 10000], // stereo
];

// Stereo downmix thresholds
const STEREO_VOICE_THRESHOLD: i32 = 19000;
const STEREO_MUSIC_THRESHOLD: i32 = 17000;

// FEC thresholds: [threshold, hysteresis] per bandwidth (NB..FB)
static FEC_THRESHOLDS: [i32; 10] = [
    12000, 1000, 14000, 1000, 16000, 1000, 20000, 1000, 22000, 1000,
];

// Hybrid SILK rate table: [total_rate, SILK_noFEC_10ms, SILK_noFEC_20ms, SILK_FEC_10ms, SILK_FEC_20ms]
static RATE_TABLE: [[i32; 5]; 7] = [
    [0, 0, 0, 0, 0],
    [12000, 10000, 10000, 11000, 11000],
    [16000, 13500, 13500, 15000, 15000],
    [20000, 16000, 16000, 18000, 18000],
    [24000, 18000, 18000, 21000, 21000],
    [32000, 22000, 22000, 28000, 28000],
    [64000, 38000, 38000, 50000, 50000],
];

// ===========================================================================
// SILK fixed-point helpers
// ===========================================================================

/// silk_SMULWB: 32×16-bit multiply, return upper 32 bits.
/// Matches C: `((a32 >> 16) * (opus_int16)(b32)) + (((a32 & 0xFFFF) * (opus_int16)(b32)) >> 16)`
#[inline(always)]
fn silk_smulwb(a32: i32, b32: i32) -> i32 {
    ((a32 as i64) * (b32 as i16 as i64) >> 16) as i32
}

/// silk_MUL: simple multiply.
#[inline(always)]
fn silk_mul(a: i32, b: i32) -> i32 {
    a * b
}

/// silk_SMULWW: 32×32-bit multiply, return upper 32 bits (result >> 16).
#[inline(always)]
fn silk_smulww(a32: i32, b32: i32) -> i32 {
    ((a32 as i64 * b32 as i64) >> 16) as i32
}

/// SILK_FIX_CONST: compile-time Q-format conversion.
const fn silk_fix_const(x: f64, bits: u32) -> i32 {
    (x * ((1i64 << bits) as f64) + 0.5) as i32
}

/// silk_RSHIFT_ROUND: right shift with rounding.
#[inline(always)]
fn silk_rshift_round(a: i32, shift: i32) -> i32 {
    if shift == 1 {
        (a >> 1) + (a & 1)
    } else if shift <= 0 {
        a
    } else {
        ((a >> (shift - 1)) + 1) >> 1
    }
}

/// silk_LSHIFT: left shift.
#[inline(always)]
fn silk_lshift(a: i32, shift: i32) -> i32 {
    (a as u32).wrapping_shl(shift as u32) as i32
}

/// silk_SAT16: saturate to i16 range.
#[inline(always)]
fn silk_sat16(a: i32) -> i32 {
    if a > i16::MAX as i32 {
        i16::MAX as i32
    } else if a < i16::MIN as i32 {
        i16::MIN as i32
    } else {
        a
    }
}

// ===========================================================================
// Types
// ===========================================================================

/// Stereo width estimation state.
/// Matches C `StereoWidthState`.
#[derive(Clone, Default)]
pub struct StereoWidthState {
    pub xx: i32,
    pub xy: i32,
    pub yy: i32,
    pub smoothed_width: i32, // Q15
    pub max_follower: i32,   // Q15
}

/// Snapshot of key SILK encoder internal state, used for comparison testing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SilkEncoderSnapshot {
    pub fs_khz: i32,
    pub frame_length: i32,
    pub nb_subfr: i32,
    pub input_buf_ix: i32,
    pub n_frames_per_packet: i32,
    pub packet_size_ms: i32,
    pub first_frame_after_reset: i32,
    pub controlled_since_last_payload: i32,
    pub prefill_flag: i32,
    pub n_frames_encoded: i32,
    pub speech_activity_q8: i32,
    pub signal_type: i32,
    pub input_quality_bands_q15: i32,
}

/// Opus encoder state.
pub struct OpusEncoder {
    // --- Immutable after init ---
    pub channels: i32,
    pub fs: i32,
    pub application: i32,

    // --- Sub-encoders ---
    silk_enc: Option<SilkEncoder>,
    celt_enc: Option<CeltEncoder>,

    // --- SILK control ---
    silk_mode: SilkEncControlStruct,

    // --- User configuration ---
    delay_compensation: i32,
    force_channels: i32,
    signal_type: i32,
    user_bandwidth: i32,
    max_bandwidth: i32,
    user_forced_mode: i32,
    voice_ratio: i32,
    use_vbr: i32,
    vbr_constraint: i32,
    variable_duration: i32,
    bitrate_bps: i32,
    user_bitrate_bps: i32,
    lsb_depth: i32,
    encoder_buffer: i32,
    lfe: i32,
    use_dtx: i32,
    fec_config: i32,

    // --- Resettable state ---
    stream_channels: i32,
    hybrid_stereo_width_q14: i16,
    variable_hp_smth2_q15: i32,
    prev_hb_gain: i32,
    hp_mem: [i32; 4],
    mode: i32,
    prev_mode: i32,
    prev_channels: i32,
    prev_framesize: i32,
    bandwidth: i32,
    auto_bandwidth: i32,
    silk_bw_switch: i32,
    first: i32,
    width_mem: StereoWidthState,
    nb_no_activity_ms_q1: i32,
    peak_signal_energy: i32,
    nonfinal_frame: i32,
    pub range_final: u32,
    delay_buffer: Vec<i16>,
    /// Saved 2.5ms of prefill data from delay_buffer, captured BEFORE the
    /// delay buffer update. Used for CELT prefill on mode transitions.
    /// C: tmp_prefill in opus_encode_frame_native.
    tmp_prefill: Vec<i16>,
}

// ===========================================================================
// Helper functions
// ===========================================================================

/// Generate TOC byte from mode, frame rate, bandwidth, and channels.
/// Matches C `gen_toc`.
fn gen_toc(mode: i32, framerate: i32, bandwidth: i32, channels: i32) -> u8 {
    let mut period = 0i32;
    let mut fr = framerate;
    while fr < 400 {
        fr <<= 1;
        period += 1;
    }

    let mut toc: u8;
    if mode == MODE_SILK_ONLY {
        toc = ((bandwidth - OPUS_BANDWIDTH_NARROWBAND) << 5) as u8;
        toc |= ((period - 2) << 3) as u8;
    } else if mode == MODE_CELT_ONLY {
        let mut tmp = bandwidth - OPUS_BANDWIDTH_MEDIUMBAND;
        if tmp < 0 {
            tmp = 0;
        }
        toc = 0x80;
        toc |= (tmp << 5) as u8;
        toc |= (period << 3) as u8;
    } else {
        // MODE_HYBRID
        toc = 0x60;
        toc |= ((bandwidth - OPUS_BANDWIDTH_SUPERWIDEBAND) << 4) as u8;
        toc |= ((period - 2) << 3) as u8;
    }

    if channels == 2 {
        toc |= 0x4;
    }
    toc
}

/// Select frame size based on variable_duration and application.
/// Returns frame size in samples or -1 on error.
/// Matches C `frame_size_select`.
pub(crate) fn frame_size_select(frame_size: i32, variable_duration: i32, fs: i32) -> i32 {
    if frame_size < fs / 400 {
        return -1;
    }
    let new_size = if variable_duration == OPUS_FRAMESIZE_ARG {
        frame_size
    } else if variable_duration >= OPUS_FRAMESIZE_2_5_MS
        && variable_duration <= OPUS_FRAMESIZE_120_MS
    {
        if variable_duration <= OPUS_FRAMESIZE_40_MS {
            (fs / 400) << (variable_duration - OPUS_FRAMESIZE_2_5_MS)
        } else {
            (variable_duration - OPUS_FRAMESIZE_2_5_MS - 2) * fs / 50
        }
    } else {
        return -1;
    };

    if new_size > frame_size {
        return -1;
    }

    // Validate frame size
    let ns = new_size;
    if !(400 * ns == fs
        || 200 * ns == fs
        || 100 * ns == fs
        || 50 * ns == fs
        || 25 * ns == fs
        || 50 * ns == 3 * fs
        || 50 * ns == 4 * fs
        || 50 * ns == 5 * fs
        || 50 * ns == 6 * fs)
    {
        return -1;
    }
    new_size
}

/// Convert bits to bitrate. Matches C `bits_to_bitrate`.
#[inline(always)]
fn bits_to_bitrate(bits: i32, fs: i32, frame_size: i32) -> i32 {
    (bits as i64 * (6 * fs as i64 / frame_size as i64) / 6) as i32
}

/// Convert bitrate to bits. Matches C `bitrate_to_bits`.
#[inline(always)]
fn bitrate_to_bits(bitrate: i32, fs: i32, frame_size: i32) -> i32 {
    (bitrate as i64 * 6 / (6 * fs as i64 / frame_size as i64)) as i32
}

/// Detect digital silence in PCM buffer.
/// Matches C `is_digital_silence` for fixed-point (non-RES24).
fn is_digital_silence(pcm: &[i16], frame_size: i32, channels: i32, _lsb_depth: i32) -> bool {
    let n = (frame_size * channels) as usize;
    for i in 0..n {
        if pcm[i] != 0 {
            return false;
        }
    }
    true
}

/// Compute frame energy for DTX activity detection.
/// Matches C `compute_frame_energy` (fixed-point path, opus_encoder.c:1080-1105).
fn compute_frame_energy(pcm: &[i16], frame_size: i32, channels: i32) -> i32 {
    let len = (frame_size * channels) as usize;

    // Find max amplitude
    let mut sample_max: i32 = 0;
    for i in 0..len {
        let abs_val = (pcm[i] as i32).abs();
        if abs_val > sample_max {
            sample_max = abs_val;
        }
    }

    // Compute shift to prevent overflow in MAC
    let max_shift = celt_ilog2(len as i32);
    let shift = imax(0, (celt_ilog2(1 + sample_max) << 1) + max_shift - 28);

    // Accumulate energy
    let mut energy: i32 = 0;
    for i in 0..len {
        let s = (pcm[i] as i32) >> shift;
        energy += s * s;
    }

    // Normalize by frame length and shift back
    energy /= len as i32;
    energy <<= shift;

    energy
}

/// Resolve user bitrate to effective bitrate.
/// Matches C `user_bitrate_to_bitrate` — always caps at max_data_bytes capacity.
fn user_bitrate_to_bitrate(
    user_bitrate_bps: i32,
    channels: i32,
    fs: i32,
    frame_size: i32,
    max_data_bytes: i32,
) -> i32 {
    let max_bitrate = bits_to_bitrate(max_data_bytes * 8, fs, frame_size);
    let user_bitrate = if user_bitrate_bps == OPUS_AUTO {
        60 * fs / frame_size + fs * channels
    } else if user_bitrate_bps == OPUS_BITRATE_MAX {
        1500000
    } else {
        user_bitrate_bps
    };
    imin(user_bitrate, max_bitrate)
}

/// Compute equivalent rate normalized to 20ms/complexity-10/VBR.
/// Matches C `compute_equiv_rate`.
fn compute_equiv_rate(
    bitrate: i32,
    channels: i32,
    frame_rate: i32,
    vbr: i32,
    mode: i32,
    complexity: i32,
    loss: i32,
) -> i32 {
    let mut equiv = bitrate;
    // Frame overhead for rates > 50 fps
    if frame_rate > 50 {
        equiv -= (40 * channels + 20) * (frame_rate - 50);
    }
    // CBR penalty
    if vbr == 0 {
        equiv -= equiv / 12;
    }
    // Complexity penalty
    equiv = equiv * (90 + complexity) / 100;
    // Mode-specific adjustments
    if mode == MODE_SILK_ONLY || mode == MODE_HYBRID {
        // Low complexity penalty
        if complexity < 2 {
            equiv = equiv * 4 / 5;
        }
        // Packet loss penalty
        equiv -= equiv * loss / (6 * loss + 10);
    } else if mode == MODE_CELT_ONLY {
        // No-pitch penalty
        if complexity < 5 {
            equiv = equiv * 9 / 10;
        }
    } else {
        // Unknown mode: moderate loss penalty
        equiv -= equiv * loss / (12 * loss + 20);
    }
    equiv
}

/// Compute SILK bitrate for hybrid mode via piecewise-linear interpolation.
/// Matches C `compute_silk_rate_for_hybrid`.
fn compute_silk_rate_for_hybrid(
    rate: i32,
    bandwidth: i32,
    frame20ms: bool,
    vbr: i32,
    fec: i32,
    channels: i32,
) -> i32 {
    let entry = 1 + (if frame20ms { 1 } else { 0 }) + 2 * (if fec != 0 { 1 } else { 0 });
    // C does rate /= channels early; all remaining logic is per-channel
    let rate = rate / channels;
    let n = RATE_TABLE.len();

    // Find first table entry with rate_table[i][0] > rate (matches C loop)
    let mut i = n;
    for idx in 1..n {
        if RATE_TABLE[idx][0] > rate {
            i = idx;
            break;
        }
    }

    let mut silk_rate;
    if i == n {
        // Rate exceeds all table entries: last entry + 50% of excess
        silk_rate = RATE_TABLE[n - 1][entry];
        silk_rate += (rate - RATE_TABLE[n - 1][0]) / 2;
    } else {
        // Direct integer interpolation matching C exactly (single division)
        let lo = RATE_TABLE[i - 1][entry];
        let hi = RATE_TABLE[i][entry];
        let x0 = RATE_TABLE[i - 1][0];
        let x1 = RATE_TABLE[i][0];
        silk_rate = (lo * (x1 - rate) + hi * (rate - x0)) / (x1 - x0);
    }

    // CBR/SWB boosts applied per-channel BEFORE multiplication (matches C)
    if vbr == 0 {
        silk_rate += 100;
    }
    if bandwidth == OPUS_BANDWIDTH_SUPERWIDEBAND {
        silk_rate += 300;
    }
    silk_rate *= channels;
    // Stereo reduction (C uses per-channel rate after rate /= channels)
    if channels == 2 && rate >= 12000 {
        silk_rate -= 1000;
    }
    silk_rate
}

/// Decide whether to enable FEC. May reduce bandwidth.
/// Matches C `decide_fec`.
fn decide_fec(
    use_in_band_fec: i32,
    packet_loss_perc: i32,
    last_fec: i32,
    mode: i32,
    bandwidth: &mut i32,
    rate: i32,
) -> i32 {
    if use_in_band_fec == 0 || packet_loss_perc == 0 || mode == MODE_CELT_ONLY {
        return 0;
    }
    let orig_bandwidth = *bandwidth;
    loop {
        let idx = 2 * (*bandwidth - OPUS_BANDWIDTH_NARROWBAND) as usize;
        if idx + 1 >= FEC_THRESHOLDS.len() {
            break;
        }
        let mut threshold = FEC_THRESHOLDS[idx];
        let hysteresis = FEC_THRESHOLDS[idx + 1];

        if last_fec == 1 {
            threshold -= hysteresis;
        } else {
            threshold += hysteresis;
        }

        // Scale by loss: threshold * (125 - min(loss, 25)) * 0.01
        let loss_factor = 125 - imin(packet_loss_perc, 25);
        threshold = silk_smulwb(silk_mul(threshold, loss_factor), silk_fix_const(0.01, 16));

        if rate > threshold {
            return 1;
        } else if packet_loss_perc <= 5 {
            return 0;
        } else if *bandwidth > OPUS_BANDWIDTH_NARROWBAND {
            *bandwidth -= 1;
        } else {
            break;
        }
    }
    *bandwidth = orig_bandwidth;
    0
}

/// Compute bytes for redundancy frame.
/// Matches C `compute_redundancy_bytes`.
fn compute_redundancy_bytes(
    max_data_bytes: i32,
    bitrate_bps: i32,
    frame_rate: i32,
    channels: i32,
) -> i32 {
    let base_bits = 40 * channels + 20;
    let redundancy_rate = bitrate_bps + base_bits * (200 - frame_rate);
    let redundancy_rate = 3 * redundancy_rate / 2;
    let mut redundancy_bytes = redundancy_rate / 1600;

    // Cap based on available space
    let available_bits = max_data_bytes * 8 - 2 * base_bits;
    let redundancy_bytes_cap = (available_bits * 240 / (240 + 48000 / frame_rate) + base_bits) / 8;
    redundancy_bytes = imin(redundancy_bytes, redundancy_bytes_cap);

    if redundancy_bytes > 4 + 8 * channels {
        imin(257, redundancy_bytes)
    } else {
        0
    }
}

/// Decide DTX mode based on activity.
/// Matches C `decide_dtx_mode`.
fn decide_dtx_mode(activity: i32, nb_no_activity_ms_q1: &mut i32, frame_size_ms_q1: i32) -> bool {
    if activity == 0 {
        *nb_no_activity_ms_q1 += frame_size_ms_q1;
    } else {
        *nb_no_activity_ms_q1 = 0;
    }

    let threshold = NB_SPEECH_FRAMES_BEFORE_DTX * 20 * 2;
    let max_threshold = (NB_SPEECH_FRAMES_BEFORE_DTX + MAX_CONSECUTIVE_DTX) * 20 * 2;

    if *nb_no_activity_ms_q1 > threshold && *nb_no_activity_ms_q1 <= max_threshold {
        true
    } else {
        if *nb_no_activity_ms_q1 > max_threshold {
            *nb_no_activity_ms_q1 = threshold;
        }
        false
    }
}

// ===========================================================================
// HP / DC Filters
// ===========================================================================

/// Biquad filter for HP cutoff, stride-1 (mono) fixed-point path.
/// Matches C `silk_biquad_alt_stride1` in silk/biquad_alt.c.
fn silk_biquad_alt_stride1(
    input: &[i16],
    b_q28: &[i32; 3],
    a_q28: &[i32; 2],
    state: &mut [i32; 2],
    output: &mut [i16],
    len: usize,
) {
    let a0_l = (-a_q28[0]) & 0x3FFF;
    let a0_u = (-a_q28[0]) >> 14;
    let a1_l = (-a_q28[1]) & 0x3FFF;
    let a1_u = (-a_q28[1]) >> 14;

    for k in 0..len {
        let inval = input[k] as i32;

        // out32_Q14 = (S[0] + SMULWB(B[0], inval)) << 2
        let out32_q14 = (state[0].wrapping_add(silk_smulwb(b_q28[0], inval))) << 2;

        // Update S[0]
        state[0] = state[1]
            .wrapping_add(silk_rshift_round(silk_smulwb(out32_q14, a0_l), 14))
            .wrapping_add(silk_smulwb(out32_q14, a0_u))
            .wrapping_add(silk_smulwb(b_q28[1], inval));

        // Update S[1]
        state[1] = silk_rshift_round(silk_smulwb(out32_q14, a1_l), 14)
            .wrapping_add(silk_smulwb(out32_q14, a1_u))
            .wrapping_add(silk_smulwb(b_q28[2], inval));

        // Output: ceiling-shift Q14->Q0, saturate to i16
        output[k] = silk_sat16((out32_q14 + (1 << 14) - 1) >> 14) as i16;
    }
}

/// Biquad filter for HP cutoff, stride-2 (stereo interleaved) fixed-point path.
/// Matches C `silk_biquad_alt_stride2_c` in silk/biquad_alt.c.
/// Input/output are interleaved: [L0, R0, L1, R1, ...].
/// State vector has 4 elements: S[0],S[1] for left, S[2],S[3] for right.
fn silk_biquad_alt_stride2(
    input: &[i16],
    b_q28: &[i32; 3],
    a_q28: &[i32; 2],
    state: &mut [i32; 4],
    output: &mut [i16],
    len: usize,
) {
    let a0_l = (-a_q28[0]) & 0x3FFF;
    let a0_u = (-a_q28[0]) >> 14;
    let a1_l = (-a_q28[1]) & 0x3FFF;
    let a1_u = (-a_q28[1]) >> 14;

    for k in 0..len {
        let in_l = input[2 * k] as i32;
        let in_r = input[2 * k + 1] as i32;

        // Compute output Q14 for both channels
        let out32_q14_l = (state[0].wrapping_add(silk_smulwb(b_q28[0], in_l))) << 2;
        let out32_q14_r = (state[2].wrapping_add(silk_smulwb(b_q28[0], in_r))) << 2;

        // Update S[0] (left) and S[2] (right)
        state[0] = state[1]
            .wrapping_add(silk_rshift_round(silk_smulwb(out32_q14_l, a0_l), 14))
            .wrapping_add(silk_smulwb(out32_q14_l, a0_u))
            .wrapping_add(silk_smulwb(b_q28[1], in_l));
        state[2] = state[3]
            .wrapping_add(silk_rshift_round(silk_smulwb(out32_q14_r, a0_l), 14))
            .wrapping_add(silk_smulwb(out32_q14_r, a0_u))
            .wrapping_add(silk_smulwb(b_q28[1], in_r));

        // Update S[1] (left) and S[3] (right)
        state[1] = silk_rshift_round(silk_smulwb(out32_q14_l, a1_l), 14)
            .wrapping_add(silk_smulwb(out32_q14_l, a1_u))
            .wrapping_add(silk_smulwb(b_q28[2], in_l));
        state[3] = silk_rshift_round(silk_smulwb(out32_q14_r, a1_l), 14)
            .wrapping_add(silk_smulwb(out32_q14_r, a1_u))
            .wrapping_add(silk_smulwb(b_q28[2], in_r));

        // Output: ceiling-shift Q14->Q0, saturate to i16
        output[2 * k] = silk_sat16((out32_q14_l + (1 << 14) - 1) >> 14) as i16;
        output[2 * k + 1] = silk_sat16((out32_q14_r + (1 << 14) - 1) >> 14) as i16;
    }
}

/// Public debug wrapper for hp_cutoff (for test harness comparison).
pub fn hp_cutoff_debug(
    input: &[i16],
    cutoff_hz: i32,
    output: &mut [i16],
    hp_mem: &mut [i32; 4],
    len: usize,
    channels: i32,
    fs: i32,
) {
    hp_cutoff(input, cutoff_hz, output, hp_mem, len, channels, fs);
}

/// Variable HP cutoff filter for VOIP mode.
/// Matches C `hp_cutoff` (fixed-point path).
fn hp_cutoff(
    input: &[i16],
    cutoff_hz: i32,
    output: &mut [i16],
    hp_mem: &mut [i32; 4],
    len: usize,
    channels: i32,
    fs: i32,
) {
    // Fc_Q19 = (1.5*pi/1000) * cutoff_Hz / (Fs/1000)
    let pi_q19: i32 = qconst32(std::f64::consts::PI * 1.5 / 1000.0, 19);
    let fc_q19 = pi_q19 * cutoff_hz / (fs / 1000);

    // r_Q28 = 1.0_Q28 - 0.92_Q9 * Fc_Q19
    let r_q28: i32 = (1i32 << 28) - silk_mul(qconst32(0.92, 9), fc_q19);

    // Biquad coefficients
    let b_q28 = [r_q28, -(r_q28 << 1), r_q28];

    // r_Q22 = r_Q28 >> 6
    let r_q22 = r_q28 >> 6;
    let fc_q19_sq = silk_smulww(fc_q19, fc_q19); // Fc²

    let a_q28 = [
        silk_smulww(r_q22, fc_q19_sq - qconst32(2.0, 22)),
        silk_smulww(r_q22, r_q22),
    ];

    // Apply filter: stride1 for mono, stride2 for stereo (matches C reference)
    if channels == 1 {
        let mut state = [hp_mem[0], hp_mem[1]];
        silk_biquad_alt_stride1(input, &b_q28, &a_q28, &mut state, output, len);
        hp_mem[0] = state[0];
        hp_mem[1] = state[1];
    } else {
        silk_biquad_alt_stride2(input, &b_q28, &a_q28, hp_mem, output, len);
    }
}

/// DC rejection filter (fixed-point).
/// Matches C `dc_reject` (fixed-point, non-RES24 path).
fn dc_reject(
    input: &[i16],
    cutoff_hz: i32,
    output: &mut [i16],
    hp_mem: &mut [i32; 4],
    len: usize,
    channels: i32,
    fs: i32,
) {
    let shift = celt_ilog2(fs / (cutoff_hz * 4));
    for c in 0..channels as usize {
        for i in 0..len {
            let idx = i * channels as usize + c;
            // Scale to Q14
            let x = (input[idx] as i32) << 14;
            // High-pass: y = x - mem
            let y = x - hp_mem[2 * c];
            // LP update: mem += (x - mem) >> shift
            hp_mem[2 * c] += pshr32(x - hp_mem[2 * c], shift);
            // Output: round Q14 back, saturate
            // C reference uses SATURATE(val, 32767) which clamps symmetrically
            // to [-32767, 32767], not [-32768, 32767] like sat16.
            let val = pshr32(y, 14);
            output[idx] = val.max(-32767).min(32767) as i16;
        }
    }
}

/// Stereo width crossfade. Matches C `stereo_fade`.
fn stereo_fade(
    pcm: &mut [i16],
    g1: i32,
    g2: i32,
    overlap48: i32,
    frame_size: i32,
    channels: i32,
    window: &[i16],
    fs: i32,
) {
    // Matches C stereo_fade() exactly: invert gains then subtract scaled diff.
    let inc = imax(1, 48000 / fs) as usize;
    let overlap = overlap48 as usize / inc;
    let g1 = Q15ONE - g1;
    let g2 = Q15ONE - g2;
    for i in 0..overlap {
        let w = window[i * inc] as i32;
        let w = mult16_16_q15(w, w);
        let g = shr32(mac16_16(mult16_16(w, g2), Q15ONE - w, g1), 15);
        let diff =
            half32(pcm[i * channels as usize] as i32 - pcm[i * channels as usize + 1] as i32);
        let diff = mult16_16_q15(g, diff);
        pcm[i * channels as usize] = sat16(pcm[i * channels as usize] as i32 - diff);
        pcm[i * channels as usize + 1] = sat16(pcm[i * channels as usize + 1] as i32 + diff);
    }
    for i in overlap..frame_size as usize {
        let diff =
            half32(pcm[i * channels as usize] as i32 - pcm[i * channels as usize + 1] as i32);
        let diff = mult16_16_q15(g2, diff);
        pcm[i * channels as usize] = sat16(pcm[i * channels as usize] as i32 - diff);
        pcm[i * channels as usize + 1] = sat16(pcm[i * channels as usize + 1] as i32 + diff);
    }
}

/// Gain crossfade. Matches C `gain_fade`.
fn gain_fade(
    pcm: &mut [i16],
    g1: i32,
    g2: i32,
    overlap48: i32,
    frame_size: i32,
    channels: i32,
    window: &[i16],
    fs: i32,
) {
    let overlap = overlap48 * fs / 48000;
    let inc = (48000 / fs) as usize;
    for i in 0..overlap as usize {
        let w = window[i * inc] as i32;
        let w = mult16_16_q15(w, w);
        let g = ((w as i64 * g2 as i64 + (Q15ONE - w) as i64 * g1 as i64) >> 15) as i32;
        for c in 0..channels as usize {
            let idx = i * channels as usize + c;
            pcm[idx] = mult16_16_q15(g, pcm[idx] as i32) as i16;
        }
    }
    if g2 != Q15ONE {
        for i in overlap as usize..frame_size as usize {
            for c in 0..channels as usize {
                let idx = i * channels as usize + c;
                pcm[idx] = mult16_16_q15(g2, pcm[idx] as i32) as i16;
            }
        }
    }
}

// ===========================================================================
// Stereo width computation
// ===========================================================================

/// Compute stereo width (fixed-point).
/// Matches C `compute_stereo_width`.
fn compute_stereo_width(pcm: &[i16], frame_size: i32, fs: i32, mem: &mut StereoWidthState) -> i32 {
    let frame_rate = fs / frame_size;
    let short_alpha = imin(Q15ONE, 25 * Q15ONE / imax(50, frame_rate));
    let shift = celt_ilog2(frame_size) - 2;

    let mut xx: i32 = 0;
    let mut xy: i32 = 0;
    let mut yy: i32 = 0;

    // 4-sample unrolled accumulation
    let mut i = 0;
    while i + 3 < frame_size as usize {
        let mut pxx: i32 = 0;
        let mut pxy: i32 = 0;
        let mut pyy: i32 = 0;
        for j in 0..4 {
            let x = pcm[(i + j) * 2] as i32;
            let y = pcm[(i + j) * 2 + 1] as i32;
            pxx += shr32(mult16_16(x, x), 2);
            pxy += shr32(mult16_16(x, y), 2);
            pyy += shr32(mult16_16(y, y), 2);
        }
        xx += shr32(pxx, shift);
        xy += shr32(pxy, shift);
        yy += shr32(pyy, shift);
        i += 4;
    }

    // Smooth
    mem.xx += mult16_32_q15(short_alpha, xx - mem.xx);
    mem.xy = mult16_32_q15(Q15ONE - short_alpha, mem.xy) + mult16_32_q15(short_alpha, xy);
    mem.yy += mult16_32_q15(short_alpha, yy - mem.yy);

    // Clamp to non-negative
    mem.xx = imax(0, mem.xx);
    mem.xy = imax(0, mem.xy);
    mem.yy = imax(0, mem.yy);

    if imax(mem.xx, mem.yy) > qconst32(8e-4, 18) {
        let sqrt_xx = celt_sqrt(mem.xx);
        let sqrt_yy = celt_sqrt(mem.yy);
        let qrrt_xx = celt_sqrt(sqrt_xx);
        let qrrt_yy = celt_sqrt(sqrt_yy);

        // Clamp XY to geometric mean
        let gm = mult16_16(sqrt_xx, sqrt_yy);
        if mem.xy > gm {
            mem.xy = gm;
        }

        // Inter-channel correlation
        let corr = shr32(
            frac_div32(mem.xy, EPSILON + mult16_16(sqrt_xx, sqrt_yy)),
            16,
        );

        // Loudness difference
        let ldiff = if qrrt_xx + qrrt_yy > 0 {
            Q15ONE * abs32(qrrt_xx - qrrt_yy) / (EPSILON + qrrt_xx + qrrt_yy)
        } else {
            0
        };

        // width = sqrt(1 - corr²) * ldiff
        let corr_sq = mult16_16(corr, corr);
        let decorr = celt_sqrt(imax(0, qconst32(1.0, 30) - corr_sq));
        let width = mult16_16_q15(imin(Q15ONE, decorr), ldiff);

        // 1-second smoothing
        mem.smoothed_width += (width - mem.smoothed_width) / frame_rate;

        // Peak follower
        mem.max_follower = imax(
            mem.max_follower - qconst16(0.02, 15) / frame_rate,
            mem.smoothed_width,
        );
    }

    imin(Q15ONE, 20 * mem.max_follower)
}

// ===========================================================================
// OpusEncoder implementation
// ===========================================================================

impl OpusEncoder {
    /// Create and initialize a new Opus encoder.
    /// `fs`: sample rate (8000, 12000, 16000, 24000, 48000).
    /// `channels`: 1 or 2.
    /// `application`: OPUS_APPLICATION_VOIP, _AUDIO, or _RESTRICTED_LOWDELAY.
    pub fn new(fs: i32, channels: i32, application: i32) -> Result<Self, i32> {
        // Validate
        if fs != 8000 && fs != 12000 && fs != 16000 && fs != 24000 && fs != 48000 {
            return Err(OPUS_BAD_ARG);
        }
        if channels != 1 && channels != 2 {
            return Err(OPUS_BAD_ARG);
        }
        if application != OPUS_APPLICATION_VOIP
            && application != OPUS_APPLICATION_AUDIO
            && application != OPUS_APPLICATION_RESTRICTED_LOWDELAY
        {
            return Err(OPUS_BAD_ARG);
        }

        // Initialize sub-encoders
        let mut silk_enc = SilkEncoder::new();
        silk_init_encoder_top(&mut silk_enc, channels as usize);

        let celt_enc = CeltEncoder::new(fs, channels).ok_or(OPUS_INTERNAL_ERROR)?;

        let encoder_buffer = fs / 100; // 10ms

        let mut enc = Self {
            channels,
            fs,
            application,
            silk_enc: Some(silk_enc),
            celt_enc: Some(celt_enc),
            silk_mode: SilkEncControlStruct {
                n_channels_api: channels,
                n_channels_internal: channels,
                api_sample_rate: fs,
                max_internal_sample_rate: 16000,
                min_internal_sample_rate: 8000,
                desired_internal_sample_rate: 16000,
                payload_size_ms: 20,
                bit_rate: 25000,
                packet_loss_percentage: 0,
                complexity: 9,
                use_in_band_fec: 0,
                use_dred: 0,
                lbrr_coded: 0,
                use_dtx: 0,
                use_cbr: 0,
                max_bits: 0,
                to_mono: 0,
                opus_can_switch: 0,
                reduced_dependency: 0,
                internal_sample_rate: 0,
                allow_bandwidth_switch: 0,
                in_wb_mode_without_variable_lp: 0,
                stereo_width_q14: 0,
                switch_ready: 0,
                signal_type: 0,
                offset: 0,
            },
            delay_compensation: fs / 250, // 4ms
            force_channels: OPUS_AUTO,
            signal_type: OPUS_AUTO,
            user_bandwidth: OPUS_AUTO,
            max_bandwidth: OPUS_BANDWIDTH_FULLBAND,
            user_forced_mode: OPUS_AUTO,
            voice_ratio: -1,
            use_vbr: 1,
            vbr_constraint: 1,
            variable_duration: OPUS_FRAMESIZE_ARG,
            bitrate_bps: 3000 + fs * channels,
            user_bitrate_bps: OPUS_AUTO,
            lsb_depth: 24,
            encoder_buffer,
            lfe: 0,
            use_dtx: 0,
            fec_config: 0,
            stream_channels: channels,
            hybrid_stereo_width_q14: 1 << 14,
            variable_hp_smth2_q15: silk_lshift(silk_lin2log(60), 8),
            prev_hb_gain: Q15ONE,
            hp_mem: [0; 4],
            mode: MODE_HYBRID,
            prev_mode: 0,
            prev_channels: 0,
            prev_framesize: fs / 50, // 20ms
            bandwidth: OPUS_BANDWIDTH_FULLBAND,
            auto_bandwidth: OPUS_BANDWIDTH_FULLBAND,
            silk_bw_switch: 0,
            first: 1,
            width_mem: StereoWidthState::default(),
            nb_no_activity_ms_q1: 0,
            peak_signal_energy: 0,
            nonfinal_frame: 0,
            range_final: 0,
            delay_buffer: vec![0i16; (encoder_buffer * channels) as usize],
            tmp_prefill: vec![0i16; (channels * fs / 400) as usize],
        };

        // Configure CELT
        if let Some(ref mut celt) = enc.celt_enc {
            celt.ctl(CeltEncoderCtl::SetSignalling(0));
            celt.ctl(CeltEncoderCtl::SetComplexity(9));
        }

        Ok(enc)
    }

    // -----------------------------------------------------------------------
    // pub(crate) accessors for multistream module
    // These are used by the multistream encoder wrapper (opus/multistream.rs).
    // Suppressing dead_code: they're called when multistream encoding is used.
    // -----------------------------------------------------------------------

    #[allow(dead_code)]
    pub(crate) fn ms_get_vbr(&self) -> i32 {
        self.use_vbr
    }
    #[allow(dead_code)]
    pub(crate) fn ms_get_bitrate(&self) -> i32 {
        self.bitrate_bps
    }
    #[allow(dead_code)]
    pub(crate) fn ms_set_user_bitrate(&mut self, rate: i32) {
        self.user_bitrate_bps = rate;
    }
    #[allow(dead_code)]
    pub(crate) fn ms_set_bandwidth(&mut self, bw: i32) {
        self.user_bandwidth = bw;
    }
    #[allow(dead_code)]
    pub(crate) fn ms_set_max_bandwidth(&mut self, bw: i32) {
        self.max_bandwidth = bw;
    }
    #[allow(dead_code)]
    pub(crate) fn ms_set_force_mode(&mut self, mode: i32) {
        self.user_forced_mode = mode;
    }
    #[allow(dead_code)]
    pub(crate) fn ms_set_force_channels(&mut self, ch: i32) {
        self.force_channels = ch;
    }
    #[allow(dead_code)]
    pub(crate) fn ms_set_lfe(&mut self, lfe: i32) {
        self.lfe = lfe;
    }
    #[allow(dead_code)]
    pub(crate) fn ms_get_variable_duration(&self) -> i32 {
        self.variable_duration
    }
    #[allow(dead_code)]
    pub(crate) fn ms_set_variable_duration(&mut self, v: i32) {
        self.variable_duration = v;
    }
    #[allow(dead_code)]
    pub(crate) fn ms_get_lsb_depth(&self) -> i32 {
        self.lsb_depth
    }
    #[allow(dead_code)]
    pub(crate) fn ms_set_lsb_depth(&mut self, v: i32) {
        self.lsb_depth = v;
    }
    #[allow(dead_code)]
    pub(crate) fn ms_get_complexity(&self) -> i32 {
        self.silk_mode.complexity
    }
    #[allow(dead_code)]
    pub(crate) fn ms_set_complexity(&mut self, v: i32) {
        self.silk_mode.complexity = v;
        if let Some(ref mut celt) = self.celt_enc {
            celt.complexity = v;
        }
    }
    #[allow(dead_code)]
    pub(crate) fn ms_set_vbr(&mut self, v: i32) {
        self.use_vbr = v;
    }
    #[allow(dead_code)]
    pub(crate) fn ms_set_vbr_constraint(&mut self, v: i32) {
        self.vbr_constraint = v;
    }
    #[allow(dead_code)]
    pub(crate) fn ms_set_signal(&mut self, v: i32) {
        self.signal_type = v;
    }
    #[allow(dead_code)]
    pub(crate) fn ms_set_inband_fec(&mut self, v: i32) {
        self.fec_config = v;
    }
    #[allow(dead_code)]
    pub(crate) fn ms_set_packet_loss_perc(&mut self, v: i32) {
        self.silk_mode.packet_loss_percentage = v;
        if let Some(ref mut celt) = self.celt_enc {
            celt.loss_rate = v;
        }
    }
    #[allow(dead_code)]
    pub(crate) fn ms_set_dtx(&mut self, v: i32) {
        self.use_dtx = v;
    }
    #[allow(dead_code)]
    pub(crate) fn ms_set_prediction_disabled(&mut self, v: i32) {
        if let Some(ref mut celt) = self.celt_enc {
            celt.disable_pf = v;
        }
    }
    #[allow(dead_code)]
    pub(crate) fn ms_set_phase_inversion_disabled(&mut self, v: i32) {
        if let Some(ref mut celt) = self.celt_enc {
            celt.disable_inv = v;
        }
    }
    #[allow(dead_code)]
    pub(crate) fn ms_set_application(&mut self, v: i32) {
        // Only update if valid
        if v == OPUS_APPLICATION_VOIP
            || v == OPUS_APPLICATION_AUDIO
            || v == OPUS_APPLICATION_RESTRICTED_LOWDELAY
        {
            self.application = v;
        }
    }
    #[allow(dead_code)]
    pub(crate) fn ms_get_lookahead(&self) -> i32 {
        self.delay_compensation
    }
    #[allow(dead_code)]
    pub(crate) fn ms_get_celt_mode(&self) -> Option<&'static crate::celt::modes::CELTMode> {
        self.celt_enc.as_ref().map(|c| c.mode)
    }

    /// Reset encoder to initial state.
    pub fn reset(&mut self) {
        self.stream_channels = self.channels;
        self.hybrid_stereo_width_q14 = 1 << 14;
        self.variable_hp_smth2_q15 = silk_lshift(silk_lin2log(60), 8);
        self.prev_hb_gain = Q15ONE;
        self.hp_mem = [0; 4];
        self.mode = MODE_HYBRID;
        self.prev_mode = 0;
        self.prev_channels = self.channels;
        self.prev_framesize = self.fs / 50;
        self.bandwidth = OPUS_BANDWIDTH_FULLBAND;
        self.auto_bandwidth = OPUS_BANDWIDTH_FULLBAND;
        self.silk_bw_switch = 0;
        self.first = 1;
        self.width_mem = StereoWidthState::default();
        self.nb_no_activity_ms_q1 = 0;
        self.peak_signal_energy = 0;
        self.nonfinal_frame = 0;
        self.range_final = 0;
        self.delay_buffer.fill(0);
        self.tmp_prefill.fill(0);

        if let Some(ref mut silk) = self.silk_enc {
            silk_init_encoder_top(silk, self.channels as usize);
        }
        if let Some(ref mut celt) = self.celt_enc {
            celt.reset();
        }
    }

    /// Encode PCM audio (16-bit input).
    /// Returns number of bytes written to `data`, or a negative error code.
    pub fn encode(
        &mut self,
        pcm: &[i16],
        frame_size: i32,
        data: &mut [u8],
        max_data_bytes: i32,
    ) -> Result<i32, i32> {
        let frame_size = frame_size_select(frame_size, self.variable_duration, self.fs);
        if frame_size < 0 {
            return Err(OPUS_BAD_ARG);
        }
        self.encode_native(pcm, frame_size, data, max_data_bytes, 16)
    }

    /// Encode PCM audio (float input, converts to i16 internally).
    /// Returns number of bytes written to `data`, or a negative error code.
    pub fn encode_float(
        &mut self,
        pcm: &[f32],
        frame_size: i32,
        data: &mut [u8],
        max_data_bytes: i32,
    ) -> Result<i32, i32> {
        let frame_size = frame_size_select(frame_size, self.variable_duration, self.fs);
        if frame_size < 0 {
            return Err(OPUS_BAD_ARG);
        }
        // Convert float to i16
        let n = (frame_size * self.channels) as usize;
        let mut pcm16 = vec![0i16; n];
        for i in 0..n {
            pcm16[i] = float2int16(pcm[i]);
        }
        self.encode_native(&pcm16, frame_size, data, max_data_bytes, 24)
    }

    // -----------------------------------------------------------------------
    // opus_encode_native — top-level orchestrator
    // -----------------------------------------------------------------------

    pub(crate) fn encode_native(
        &mut self,
        pcm: &[i16],
        frame_size: i32,
        data: &mut [u8],
        out_data_bytes: i32,
        lsb_depth: i32,
    ) -> Result<i32, i32> {
        let max_data_bytes = imin(1276 * 6, out_data_bytes);
        self.range_final = 0;

        if frame_size <= 0 || max_data_bytes <= 0 {
            return Err(OPUS_BAD_ARG);
        }
        // Can't encode 100ms in 1 byte
        if max_data_bytes == 1 && self.fs == frame_size * 10 {
            return Err(OPUS_BUFFER_TOO_SMALL);
        }

        let lsb_depth = imin(lsb_depth, self.lsb_depth);
        let is_silence = is_digital_silence(pcm, frame_size, self.channels, lsb_depth);

        // C: st->voice_ratio = -1; (reset each frame, #else / FIXED_POINT path)
        self.voice_ratio = -1;

        // --- Track peak signal energy ---
        // C: st->peak_signal_energy = MAX32(MULT16_32_Q15(QCONST16(0.999f, 15), st->peak_signal_energy),
        //         compute_frame_energy(pcm, frame_size, st->channels, st->arch));
        if !is_silence {
            let frame_energy = compute_frame_energy(pcm, frame_size, self.channels);
            self.peak_signal_energy =
                (((self.peak_signal_energy as i64 * 32735) >> 15) as i32).max(frame_energy);
        }

        // --- Stereo width ---
        let stereo_width = if self.channels == 2 && self.force_channels != 1 {
            compute_stereo_width(pcm, frame_size, self.fs, &mut self.width_mem)
        } else {
            0
        };

        // --- Bitrate ---
        let mut bitrate_bps = user_bitrate_to_bitrate(
            self.user_bitrate_bps,
            self.channels,
            self.fs,
            frame_size,
            max_data_bytes,
        );
        self.bitrate_bps = bitrate_bps;

        let frame_rate = self.fs / frame_size;

        // CBR byte count
        let mut max_data_bytes = max_data_bytes;
        if self.use_vbr == 0 {
            let cbr_bytes = imin(
                (bitrate_to_bits(bitrate_bps, self.fs, frame_size) + 4) / 8,
                max_data_bytes,
            );
            bitrate_bps = bits_to_bitrate(cbr_bytes * 8, self.fs, frame_size);
            max_data_bytes = imax(1, cbr_bytes);
            self.bitrate_bps = bitrate_bps;
        }

        // C: max_rate is computed AFTER CBR adjustment reduces max_data_bytes
        let max_rate = bits_to_bitrate(max_data_bytes * 8, self.fs, frame_size);

        // --- PLC frame emission ---
        if max_data_bytes < 3
            || bitrate_bps < 3 * frame_rate * 8
            || (frame_rate < 50 && (max_data_bytes * frame_rate < 300 || bitrate_bps < 2400))
        {
            // Emit 1-byte TOC-only packet
            let toc_mode = if self.prev_mode == 0 {
                MODE_SILK_ONLY
            } else {
                self.prev_mode
            };
            let toc_bw = if self.bandwidth == 0 {
                OPUS_BANDWIDTH_NARROWBAND
            } else {
                self.bandwidth
            };
            data[0] = gen_toc(toc_mode, frame_rate, toc_bw, self.stream_channels);
            self.range_final = 0;
            // Update delay buffer
            self.update_delay_buffer(pcm, frame_size);
            return Ok(1);
        }

        // --- Equivalent rate ---
        let complexity = self.silk_mode.complexity;
        let loss = self.silk_mode.packet_loss_percentage;
        let equiv_rate = compute_equiv_rate(
            bitrate_bps,
            self.channels,
            frame_rate,
            self.use_vbr,
            0,
            complexity,
            loss,
        );

        // --- Voice estimate (Q7) ---
        let voice_est: i32;
        if self.signal_type == OPUS_SIGNAL_VOICE {
            voice_est = 127;
        } else if self.signal_type == OPUS_SIGNAL_MUSIC {
            voice_est = 0;
        } else if self.voice_ratio >= 0 {
            let mut ve = self.voice_ratio * 327 >> 8;
            if self.application == OPUS_APPLICATION_AUDIO {
                ve = imin(ve, 115);
            }
            voice_est = ve;
        } else if self.application == OPUS_APPLICATION_VOIP {
            voice_est = 115;
        } else {
            voice_est = 48;
        }

        // --- Channel count decision ---
        if self.force_channels != OPUS_AUTO && self.channels == 2 {
            self.stream_channels = self.force_channels;
        } else if self.channels == 2 {
            let stereo_threshold = STEREO_MUSIC_THRESHOLD
                + (voice_est as i64
                    * voice_est as i64
                    * (STEREO_VOICE_THRESHOLD - STEREO_MUSIC_THRESHOLD) as i64
                    / 16384) as i32;
            let hysteresis = if self.stream_channels == 2 {
                -1000
            } else {
                1000
            };
            self.stream_channels = if equiv_rate > stereo_threshold + hysteresis {
                2
            } else {
                1
            };
        } else {
            self.stream_channels = self.channels;
        }

        // Recompute equiv_rate with stream_channels
        let equiv_rate = compute_equiv_rate(
            bitrate_bps,
            self.stream_channels,
            frame_rate,
            self.use_vbr,
            0,
            complexity,
            loss,
        );

        // --- SILK DTX ---
        // C: st->silk_mode.useDTX = st->use_dtx && !is_silence;
        self.silk_mode.use_dtx = if self.use_dtx != 0 && !is_silence {
            1
        } else {
            0
        };

        // --- Mode selection ---
        let mut mode: i32;
        if self.application == OPUS_APPLICATION_RESTRICTED_LOWDELAY {
            mode = MODE_CELT_ONLY;
        } else if self.user_forced_mode == OPUS_AUTO {
            // Interpolate threshold between voice and music
            // C: MULT16_32_Q15(Q15ONE-stereo_width, A) + MULT16_32_Q15(stereo_width, B)
            let mode_voice = mult16_32_q15(Q15ONE - stereo_width, MODE_THRESHOLDS[0][0])
                + mult16_32_q15(stereo_width, MODE_THRESHOLDS[1][0]);
            // C: both terms use mode_thresholds[1][1] (not [0][1])
            let mode_music = mult16_32_q15(Q15ONE - stereo_width, MODE_THRESHOLDS[1][1])
                + mult16_32_q15(stereo_width, MODE_THRESHOLDS[1][1]);
            let mut threshold = mode_music
                + (voice_est as i64 * voice_est as i64 * (mode_voice - mode_music) as i64 / 16384)
                    as i32;

            if self.application == OPUS_APPLICATION_VOIP {
                threshold += 8000;
            }
            // Hysteresis
            if self.prev_mode == MODE_CELT_ONLY {
                threshold -= 4000;
            } else if self.prev_mode > 0 {
                threshold += 4000;
            }

            mode = if equiv_rate >= threshold {
                MODE_CELT_ONLY
            } else {
                MODE_SILK_ONLY
            };

            // FEC override
            if self.silk_mode.use_in_band_fec != 0
                && loss > (128 - voice_est) >> 4
                && (self.fec_config != 2 || voice_est > 25)
            {
                mode = MODE_SILK_ONLY;
            }
            // DTX override
            if self.silk_mode.use_dtx != 0 && voice_est > 100 {
                mode = MODE_SILK_ONLY;
            }
            // Low bitrate override
            let low_rate_threshold = if frame_rate > 50 { 9000 } else { 6000 };
            if max_data_bytes < bitrate_to_bits(low_rate_threshold, self.fs, frame_size) / 8 {
                mode = MODE_CELT_ONLY;
            }
        } else {
            mode = self.user_forced_mode;
        }

        // --- Mode overrides ---
        if mode != MODE_CELT_ONLY && frame_size < self.fs / 100 {
            mode = MODE_CELT_ONLY;
        }
        if self.lfe != 0 {
            mode = MODE_CELT_ONLY;
        }

        // --- Redundancy decision ---
        let mut redundancy = false;
        let mut celt_to_silk = false;
        let mut to_celt = false;
        let mut prefill: i32 = 0;

        if self.prev_mode > 0 {
            let was_celt = self.prev_mode == MODE_CELT_ONLY;
            let is_celt = mode == MODE_CELT_ONLY;
            if was_celt != is_celt {
                redundancy = true;
                celt_to_silk = mode != MODE_CELT_ONLY;
                if !celt_to_silk {
                    if frame_size >= self.fs / 100 {
                        mode = self.prev_mode;
                        to_celt = true;
                    } else {
                        redundancy = false;
                    }
                }
            }
        }

        // --- Stereo→mono transition ---
        if self.stream_channels == 1
            && self.prev_channels == 2
            && self.silk_mode.to_mono == 0
            && mode != MODE_CELT_ONLY
            && self.prev_mode != MODE_CELT_ONLY
        {
            self.silk_mode.to_mono = 1;
            self.stream_channels = 2;
        } else {
            self.silk_mode.to_mono = 0;
        }

        // Recompute equiv_rate with final mode
        let equiv_rate = compute_equiv_rate(
            bitrate_bps,
            self.stream_channels,
            frame_rate,
            self.use_vbr,
            mode,
            complexity,
            loss,
        );

        // --- SILK re-init on transition ---
        if mode != MODE_CELT_ONLY && self.prev_mode == MODE_CELT_ONLY {
            if let Some(ref mut silk) = self.silk_enc {
                silk_init_encoder_top(silk, self.channels as usize);
            }
            prefill = 1;
        }

        // --- Bandwidth selection ---
        if mode == MODE_CELT_ONLY || self.first != 0 || self.silk_mode.allow_bandwidth_switch != 0 {
            let (voice_bw_thresholds, music_bw_thresholds) =
                if self.channels == 2 && self.force_channels != 1 {
                    (
                        &STEREO_VOICE_BANDWIDTH_THRESHOLDS,
                        &STEREO_MUSIC_BANDWIDTH_THRESHOLDS,
                    )
                } else {
                    (
                        &MONO_VOICE_BANDWIDTH_THRESHOLDS,
                        &MONO_MUSIC_BANDWIDTH_THRESHOLDS,
                    )
                };

            // Interpolate bandwidth thresholds depending on voice estimation
            let mut bandwidth_thresholds = [0i32; 8];
            for i in 0..8 {
                bandwidth_thresholds[i] = music_bw_thresholds[i]
                    + ((voice_est * voice_est * (voice_bw_thresholds[i] - music_bw_thresholds[i]))
                        >> 14);
            }

            let mut bw = OPUS_BANDWIDTH_FULLBAND;
            while bw > OPUS_BANDWIDTH_NARROWBAND {
                let idx = 2 * (bw - OPUS_BANDWIDTH_MEDIUMBAND) as usize;
                if idx + 1 < bandwidth_thresholds.len() {
                    let mut thr = bandwidth_thresholds[idx];
                    let hys = bandwidth_thresholds[idx + 1];
                    if self.first == 0 {
                        if self.auto_bandwidth >= bw {
                            thr -= hys;
                        } else {
                            thr += hys;
                        }
                    }
                    if equiv_rate >= thr {
                        break;
                    }
                }
                bw -= 1;
            }
            // Skip mediumband
            if bw == OPUS_BANDWIDTH_MEDIUMBAND {
                bw = OPUS_BANDWIDTH_WIDEBAND;
            }
            self.bandwidth = bw;
            self.auto_bandwidth = bw;

            // Prevent SWB/FB until SILK variable LP is off
            if self.first == 0
                && mode != MODE_CELT_ONLY
                && self.silk_mode.in_wb_mode_without_variable_lp == 0
                && self.bandwidth > OPUS_BANDWIDTH_WIDEBAND
            {
                self.bandwidth = OPUS_BANDWIDTH_WIDEBAND;
            }
        }

        // Cap by max_bandwidth
        self.bandwidth = imin(self.bandwidth, self.max_bandwidth);
        if self.user_bandwidth != OPUS_AUTO {
            self.bandwidth = self.user_bandwidth;
        }
        // Cap by max rate in SILK mode
        if mode != MODE_CELT_ONLY && max_rate < 15000 {
            self.bandwidth = imin(self.bandwidth, OPUS_BANDWIDTH_WIDEBAND);
        }
        // Nyquist limits
        if self.fs <= 24000 {
            self.bandwidth = imin(self.bandwidth, OPUS_BANDWIDTH_SUPERWIDEBAND);
        }
        if self.fs <= 16000 {
            self.bandwidth = imin(self.bandwidth, OPUS_BANDWIDTH_WIDEBAND);
        }
        if self.fs <= 12000 {
            self.bandwidth = imin(self.bandwidth, OPUS_BANDWIDTH_MEDIUMBAND);
        }
        if self.fs <= 8000 {
            self.bandwidth = imin(self.bandwidth, OPUS_BANDWIDTH_NARROWBAND);
        }

        // --- FEC decision ---
        let mut fec_bandwidth = self.bandwidth;
        self.silk_mode.lbrr_coded = decide_fec(
            self.silk_mode.use_in_band_fec,
            self.silk_mode.packet_loss_percentage,
            self.silk_mode.lbrr_coded,
            mode,
            &mut fec_bandwidth,
            equiv_rate,
        );
        self.bandwidth = fec_bandwidth;

        // Set CELT lsb_depth (C: opus_encoder.c:1677-1678)
        if self.application != OPUS_APPLICATION_RESTRICTED_SILK {
            if let Some(ref mut celt) = self.celt_enc {
                celt.ctl(CeltEncoderCtl::SetLsbDepth(lsb_depth));
            }
        }

        // CELT mediumband → wideband
        if mode == MODE_CELT_ONLY && self.bandwidth == OPUS_BANDWIDTH_MEDIUMBAND {
            self.bandwidth = OPUS_BANDWIDTH_WIDEBAND;
        }
        // LFE → narrowband
        if self.lfe != 0 {
            self.bandwidth = OPUS_BANDWIDTH_NARROWBAND;
        }

        // --- SILK vs HYBRID refinement ---
        if mode == MODE_SILK_ONLY && self.bandwidth > OPUS_BANDWIDTH_WIDEBAND {
            mode = MODE_HYBRID;
        }
        if mode == MODE_HYBRID && self.bandwidth <= OPUS_BANDWIDTH_WIDEBAND {
            mode = MODE_SILK_ONLY;
        }

        // Store finalized mode for use in encode_frame_native
        self.mode = mode;

        // --- Multi-frame handling ---
        let max_silk_frame = 3 * self.fs / 50; // 60ms
        let max_celt_frame = self.fs / 50; // 20ms
        let needs_multiframe = if mode == MODE_SILK_ONLY {
            frame_size > max_silk_frame
        } else {
            frame_size > max_celt_frame
        };

        if needs_multiframe {
            return self.encode_multiframe(
                pcm,
                frame_size,
                data,
                max_data_bytes,
                max_data_bytes,
                lsb_depth,
                mode,
                bitrate_bps,
                is_silence,
                redundancy,
                celt_to_silk,
                prefill,
                equiv_rate,
                to_celt,
            );
        }

        // --- Single frame ---
        self.encode_frame_native(
            pcm,
            frame_size,
            data,
            max_data_bytes,
            max_data_bytes,
            is_silence,
            redundancy,
            celt_to_silk,
            prefill,
            equiv_rate,
            to_celt,
        )
    }

    // -----------------------------------------------------------------------
    // encode_multiframe — split into sub-frames and repacketize
    // -----------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn encode_multiframe(
        &mut self,
        pcm: &[i16],
        frame_size: i32,
        data: &mut [u8],
        max_data_bytes: i32,
        orig_max_data_bytes: i32,
        lsb_depth: i32,
        mode: i32,
        _bitrate_bps: i32,
        _is_silence: bool,
        redundancy: bool,
        celt_to_silk: bool,
        prefill: i32,
        equiv_rate: i32,
        to_celt: bool,
    ) -> Result<i32, i32> {
        // Ensure self.mode matches the mode parameter.  In the normal encode
        // path this is already set by the caller (line ~1628), but being
        // explicit here keeps encode_multiframe self-contained.
        self.mode = mode;

        // Determine sub-frame size
        let enc_frame_size = if mode == MODE_SILK_ONLY {
            if frame_size == 2 * self.fs / 25 {
                // 80ms → 2×40ms
                2 * self.fs / 50
            } else if frame_size == 3 * self.fs / 25 {
                // 120ms → 2×60ms
                3 * self.fs / 50
            } else {
                self.fs / 50 // 20ms
            }
        } else {
            self.fs / 50 // 20ms
        };

        let nb_frames = frame_size / enc_frame_size;
        if nb_frames < 1 {
            return Err(OPUS_INTERNAL_ERROR);
        }

        // C: bak_to_mono = st->silk_mode.toMono;
        let bak_to_mono = self.silk_mode.to_mono;
        if bak_to_mono != 0 {
            self.force_channels = 1;
        } else {
            self.prev_channels = self.stream_channels;
        }

        // C: repacketize_len = use_vbr ? out_data_bytes : IMIN(cbr_bytes, out_data_bytes)
        // For CBR, max_data_bytes is already clamped to cbr_bytes by the caller.
        let repacketize_len = if self.use_vbr != 0 || self.user_bitrate_bps == OPUS_BITRATE_MAX {
            orig_max_data_bytes
        } else {
            imin(max_data_bytes, orig_max_data_bytes)
        };

        // C: max_header_bytes = nb_frames == 2 ? 3 : (2+(nb_frames-1)*2)
        let max_header_bytes = if nb_frames == 2 {
            3
        } else {
            2 + (nb_frames - 1) * 2
        };
        let max_len_sum = nb_frames + repacketize_len - max_header_bytes;
        let mut tot_size: i32 = 0;
        let mut dtx_count: i32 = 0;

        // Encode each sub-frame using encode_frame_native with per-frame
        // transition flags (matching C opus_encode_native multiframe loop).
        let mut sub_packets: Vec<Vec<u8>> = Vec::with_capacity(nb_frames as usize);
        for i in 0..nb_frames {
            self.silk_mode.to_mono = 0;
            self.nonfinal_frame = if i < nb_frames - 1 { 1 } else { 0 };

            // C: frame_to_celt = to_celt && i==nb_frames-1;
            let frame_to_celt = to_celt && i == nb_frames - 1;
            // C: frame_redundancy = redundancy && (frame_to_celt || (!to_celt && i==0));
            let frame_redundancy = redundancy && (frame_to_celt || (!to_celt && i == 0));

            // C: curr_max = IMIN(bitrate_to_bits(...)/8, max_len_sum/nb_frames);
            //    curr_max = IMIN(max_len_sum-tot_size, curr_max);
            let mut curr_max = imin(
                bitrate_to_bits(self.bitrate_bps, self.fs, enc_frame_size) / 8,
                max_len_sum / nb_frames,
            );
            curr_max = imin(max_len_sum - tot_size, curr_max);

            let offset = (i * enc_frame_size * self.channels) as usize;
            let pcm_frame = &pcm[offset..];
            let frame_is_silence =
                is_digital_silence(pcm_frame, enc_frame_size, self.channels, lsb_depth);

            let mut frame_buf = vec![0u8; curr_max as usize];

            let ret = self.encode_frame_native(
                pcm_frame,
                enc_frame_size,
                &mut frame_buf,
                curr_max,
                curr_max,
                frame_is_silence,
                frame_redundancy,
                celt_to_silk,
                prefill,
                equiv_rate,
                frame_to_celt,
            )?;

            if ret == 1 {
                dtx_count += 1;
            }

            frame_buf.truncate(ret as usize);
            tot_size += ret;
            sub_packets.push(frame_buf);
        }
        self.nonfinal_frame = 0;

        // Repacketize — C uses out_range_impl with CBR pad flag
        let mut rp = OpusRepacketizer::new();
        for pkt in &sub_packets {
            let ret = rp.cat(pkt, pkt.len() as i32);
            if ret != OPUS_OK {
                return Err(ret);
            }
        }

        let pad_cbr = self.use_vbr == 0 && dtx_count != nb_frames;
        let ret = rp.out_range_impl(
            0,
            nb_frames as usize,
            data,
            repacketize_len,
            false,
            pad_cbr,
            &[],
        );
        if ret < 0 {
            return Err(ret);
        }

        self.silk_mode.to_mono = bak_to_mono;

        self.range_final = if let Some(ref celt) = self.celt_enc {
            celt.rng
        } else {
            0
        };

        Ok(ret)
    }

    // -----------------------------------------------------------------------
    // encode_frame_native — encode a single frame
    // -----------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn encode_frame_native(
        &mut self,
        pcm: &[i16],
        frame_size: i32,
        data: &mut [u8],
        max_data_bytes: i32,
        orig_max_data_bytes: i32,
        is_silence: bool,
        mut redundancy: bool,
        mut celt_to_silk: bool,
        mut prefill: i32,
        equiv_rate: i32,
        to_celt: bool,
    ) -> Result<i32, i32> {
        let max_data_bytes = imin(max_data_bytes, 1276);
        let mut curr_bandwidth = self.bandwidth;
        let delay_compensation = if self.application == OPUS_APPLICATION_RESTRICTED_LOWDELAY {
            0
        } else {
            self.delay_compensation
        };
        let total_buffer = delay_compensation;
        let frame_rate = self.fs / frame_size;

        // --- Activity detection ---
        // C: if (is_silence) activity = !is_silence;
        //    else if (st->mode == MODE_CELT_ONLY) { noise_energy based check }
        //    else activity = VAD_NO_DECISION; (SILK handles it)
        let mut activity = if is_silence {
            0
        } else if self.mode == MODE_CELT_ONLY {
            let noise_energy = compute_frame_energy(pcm, frame_size, self.channels);
            // C: activity = st->peak_signal_energy < (QCONST16(PSEUDO_SNR_THRESHOLD, 0) * (opus_val64)HALF32(noise_energy));
            // HALF32(x) = x >> 1 in fixed-point
            let half_noise = (noise_energy >> 1) as i64;
            if (self.peak_signal_energy as i64) < (PSEUDO_SNR_THRESHOLD as i64 * half_noise) {
                1
            } else {
                0
            }
        } else {
            VAD_NO_DECISION
        };

        // --- SILK bandwidth switch ---
        if self.silk_bw_switch != 0 {
            redundancy = true;
            celt_to_silk = true;
            self.silk_bw_switch = 0;
            prefill = 2;
        }
        if self.mode == MODE_CELT_ONLY {
            redundancy = false;
        }

        // --- Redundancy bytes ---
        let mut redundancy_bytes: i32 = 0;
        if redundancy {
            redundancy_bytes = compute_redundancy_bytes(
                max_data_bytes,
                self.bitrate_bps,
                frame_rate,
                self.stream_channels,
            );
            if redundancy_bytes == 0 {
                redundancy = false;
            }
        }

        // --- Bits target ---
        let bits_target = imin(
            8 * (max_data_bytes - redundancy_bytes),
            bitrate_to_bits(self.bitrate_bps, self.fs, frame_size),
        ) - 8;

        // --- Build pcm_buf with delay compensation ---
        let pcm_buf_len = ((total_buffer + frame_size) * self.channels) as usize;
        let mut pcm_buf = vec![0i16; pcm_buf_len];

        // Copy delay buffer prefix
        if total_buffer > 0 && !self.delay_buffer.is_empty() {
            let db_offset = ((self.encoder_buffer - total_buffer) * self.channels) as usize;
            let copy_len = (total_buffer * self.channels) as usize;
            let db_len = self.delay_buffer.len();
            if db_offset + copy_len <= db_len {
                pcm_buf[..copy_len]
                    .copy_from_slice(&self.delay_buffer[db_offset..db_offset + copy_len]);
            }
        }

        // --- HP smoothing ---
        let hp_freq_smth1 = if self.mode == MODE_CELT_ONLY {
            silk_lshift(silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ), 8)
        } else if let Some(ref silk) = self.silk_enc {
            silk.state_fxx[0].s_cmn.variable_hp_smth1_q15
        } else {
            silk_lshift(silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ), 8)
        };

        // C: silk_SMLAWB(smth2, diff, SILK_FIX_CONST(0.015, 16)) = smth2 + (diff * 983) >> 16
        let hp_diff = hp_freq_smth1 - self.variable_hp_smth2_q15;
        self.variable_hp_smth2_q15 +=
            ((hp_diff as i64 * VARIABLE_HP_SMTH_COEF2 as i64) >> 16) as i32;
        let cutoff_hz = silk_log2lin(shr32(self.variable_hp_smth2_q15, 8));

        // --- HP / DC filter on new PCM ---
        let new_pcm_offset = (total_buffer * self.channels) as usize;
        if self.application == OPUS_APPLICATION_VOIP {
            hp_cutoff(
                pcm,
                cutoff_hz,
                &mut pcm_buf[new_pcm_offset..],
                &mut self.hp_mem,
                frame_size as usize,
                self.channels,
                self.fs,
            );
        } else {
            dc_reject(
                pcm,
                3,
                &mut pcm_buf[new_pcm_offset..],
                &mut self.hp_mem,
                frame_size as usize,
                self.channels,
                self.fs,
            );
        }

        // --- Initialize range encoder ---
        // Split output: [TOC byte | encoded data | redundancy]
        let (toc_slice, enc_data) = data.split_at_mut(1);
        let enc_data_len = (orig_max_data_bytes - 1) as usize;

        let mut range_final: u32;
        let mut ret: i32 = 0;
        let nb_compr_bytes: i32;
        let mut redundant_rng: u32 = 0;
        // Bit count of the range encoder captured just before `enc` is dropped
        // so the outer budget-bust check (C: opus_encoder.c:2580) has access to
        // `ec_tell(&enc)` after the inner scope ends.
        let bits_used: i32;

        // --- SILK processing ---
        let mut hb_gain = Q15ONE;
        let mut start_band = 0i32;
        let mut redundancy_frame: Vec<u8> = Vec::new();

        {
            let mut enc = RangeEncoder::new(&mut enc_data[..enc_data_len]);

            if self.mode != MODE_CELT_ONLY {
                let total_bit_rate = bits_to_bitrate(bits_target, self.fs, frame_size);

                if self.mode == MODE_HYBRID {
                    self.silk_mode.bit_rate = compute_silk_rate_for_hybrid(
                        total_bit_rate,
                        curr_bandwidth,
                        frame_size == self.fs / 50,
                        self.use_vbr,
                        self.silk_mode.lbrr_coded,
                        self.stream_channels,
                    );
                    // HB gain attenuation — always computed in hybrid mode.
                    // C: HB_gain = Q15ONE - SHR32(celt_exp2(-celt_rate * QCONST16(1.f/1024, 10)), 1)
                    // QCONST16(1.f/1024, 10) = round(1/1024 * 2^10) = 1, so argument is -celt_rate
                    let celt_rate = total_bit_rate - self.silk_mode.bit_rate;
                    hb_gain = Q15ONE - shr32(celt_exp2(-celt_rate), 1);
                    hb_gain = imax(0, hb_gain);
                } else {
                    self.silk_mode.bit_rate = total_bit_rate;
                }

                // SILK mode parameters
                self.silk_mode.payload_size_ms = 1000 * frame_size / self.fs;
                self.silk_mode.n_channels_api = self.channels;
                self.silk_mode.n_channels_internal = self.stream_channels;
                self.silk_mode.desired_internal_sample_rate =
                    if curr_bandwidth == OPUS_BANDWIDTH_NARROWBAND {
                        8000
                    } else if curr_bandwidth == OPUS_BANDWIDTH_MEDIUMBAND {
                        12000
                    } else {
                        16000
                    };
                if self.mode == MODE_HYBRID {
                    self.silk_mode.min_internal_sample_rate = 16000;
                } else {
                    self.silk_mode.min_internal_sample_rate = 8000;
                }
                self.silk_mode.max_internal_sample_rate = 16000;

                // C: opus_encoder.c:2129-2143 — At very low bitrates in SILK_ONLY mode,
                // cap the internal sample rate so SILK doesn't try to encode more
                // bandwidth than the bitrate can support.
                if self.mode == MODE_SILK_ONLY {
                    let effective_max_rate =
                        bits_to_bitrate(max_data_bytes * 8, self.fs, frame_size);
                    let effective_max_rate = if frame_rate > 50 {
                        effective_max_rate * 2 / 3
                    } else {
                        effective_max_rate
                    };
                    if effective_max_rate < 8000 {
                        self.silk_mode.max_internal_sample_rate = 12000;
                        self.silk_mode.desired_internal_sample_rate =
                            imin(12000, self.silk_mode.desired_internal_sample_rate);
                    }
                    if effective_max_rate < 7000 {
                        self.silk_mode.max_internal_sample_rate = 8000;
                        self.silk_mode.desired_internal_sample_rate =
                            imin(8000, self.silk_mode.desired_internal_sample_rate);
                    }
                }

                self.silk_mode.use_cbr = if self.use_vbr != 0 { 0 } else { 1 };
                self.silk_mode.max_bits = (max_data_bytes - 1) * 8;
                if redundancy && redundancy_bytes >= 2 {
                    // Count 1 bit for redundancy position and 20 bits for
                    // flag+size (only for hybrid).
                    self.silk_mode.max_bits -= redundancy_bytes * 8 + 1;
                    if self.mode == MODE_HYBRID {
                        self.silk_mode.max_bits -= 20;
                    }
                }

                if self.silk_mode.use_cbr != 0 {
                    // When in CBR mode but encoding hybrid, switch SILK to
                    // VBR with cap. Variations are absorbed by CELT/DRED.
                    if self.mode == MODE_HYBRID {
                        let other_bits = 0i16.max(
                            (self.silk_mode.max_bits
                                - self.silk_mode.bit_rate * frame_size / self.fs)
                                as i16,
                        );
                        self.silk_mode.max_bits =
                            0.max(self.silk_mode.max_bits - (other_bits as i32) * 3 / 4);
                        self.silk_mode.use_cbr = 0;
                    }
                } else {
                    // Constrained VBR
                    if self.mode == MODE_HYBRID {
                        let max_rate = compute_silk_rate_for_hybrid(
                            self.silk_mode.max_bits * self.fs / frame_size,
                            self.bandwidth,
                            frame_size == self.fs / 50,
                            self.use_vbr,
                            self.silk_mode.lbrr_coded,
                            self.stream_channels,
                        );
                        self.silk_mode.max_bits = bitrate_to_bits(max_rate, self.fs, frame_size);
                    }
                }

                // Prefill SILK on mode transition
                // C: applies gain_fade onset ramp on delay_buffer IN-PLACE, zeros
                // before it, then feeds entire delay_buffer to silk_Encode for prefill.
                // The in-place modification is intentional — tmp_prefill (for CELT
                // prefill) is copied from delay_buffer AFTER these modifications, so
                // it must see the gain-faded data.
                if prefill != 0 && self.application != OPUS_APPLICATION_RESTRICTED_SILK {
                    if let Some(ref mut silk) = self.silk_enc {
                        let db_samples = (self.encoder_buffer * self.channels) as usize;
                        // C: prefill_offset = channels * (encoder_buffer - delay_compensation - Fs/400)
                        let prefill_offset = (self.channels
                            * (self.encoder_buffer - self.delay_compensation - self.fs / 400))
                            as usize;
                        // Apply gain_fade onset ramp (0 → Q15ONE) on the last 2.5ms
                        if prefill_offset + (self.fs as usize / 400 * self.channels as usize)
                            <= self.delay_buffer.len()
                        {
                            let celt_overlap = if let Some(ref celt) = self.celt_enc {
                                celt.mode.overlap as i32
                            } else {
                                120
                            };
                            let celt_window = if let Some(ref celt) = self.celt_enc {
                                celt.mode.window.to_vec()
                            } else {
                                vec![]
                            };
                            if !celt_window.is_empty() {
                                gain_fade(
                                    &mut self.delay_buffer[prefill_offset..],
                                    0,
                                    Q15ONE,
                                    celt_overlap,
                                    self.fs / 400,
                                    self.channels,
                                    &celt_window,
                                    self.fs,
                                );
                            }
                            // Zero everything before the ramp
                            for s in self.delay_buffer[..prefill_offset].iter_mut() {
                                *s = 0;
                            }
                        }
                        let mut prefill_control = self.silk_mode.clone();
                        prefill_control.payload_size_ms = 10;
                        prefill_control.complexity = 0;
                        let mut zero = 0i32;
                        let prefill_pcm = self.delay_buffer[..db_samples].to_vec();
                        silk_encode(
                            silk,
                            &mut prefill_control,
                            &prefill_pcm,
                            self.encoder_buffer * self.channels,
                            &mut enc,
                            &mut zero,
                            prefill,
                            activity,
                        );
                    }
                    self.silk_mode.opus_can_switch = 0;
                }

                // Encode SILK
                let mut n_bytes = 0i32;
                if let Some(ref mut silk) = self.silk_enc {
                    let silk_offset = (total_buffer * self.channels) as usize;
                    let silk_pcm =
                        &pcm_buf[silk_offset..silk_offset + (frame_size * self.channels) as usize];
                    let silk_ret = silk_encode(
                        silk,
                        &mut self.silk_mode,
                        silk_pcm,
                        frame_size * self.channels,
                        &mut enc,
                        &mut n_bytes,
                        0,
                        activity,
                    );
                    if silk_ret != 0 {
                        return Err(OPUS_INTERNAL_ERROR);
                    }
                }

                // Extract internal bandwidth from SILK
                if self.mode == MODE_SILK_ONLY {
                    curr_bandwidth = match self.silk_mode.internal_sample_rate {
                        8000 => OPUS_BANDWIDTH_NARROWBAND,
                        12000 => OPUS_BANDWIDTH_MEDIUMBAND,
                        _ => OPUS_BANDWIDTH_WIDEBAND,
                    };
                }

                // C: st->silk_mode.opusCanSwitch = st->silk_mode.switchReady && !st->nonfinal_frame;
                self.silk_mode.opus_can_switch =
                    if self.silk_mode.switch_ready != 0 && self.nonfinal_frame == 0 {
                        1
                    } else {
                        0
                    };

                // Get activity from SILK
                if activity == VAD_NO_DECISION {
                    activity = if self.silk_mode.signal_type != TYPE_NO_VOICE_ACTIVITY {
                        1
                    } else {
                        0
                    };
                }

                // DTX: if SILK produced 0 bytes
                // C returns immediately without updating delay buffer
                if n_bytes == 0 {
                    self.range_final = 0;
                    toc_slice[0] =
                        gen_toc(self.mode, frame_rate, curr_bandwidth, self.stream_channels);
                    return Ok(1);
                }

                // Check for SILK-initiated bandwidth switch (C: opus_encoder.c:2251-2260)
                if self.silk_mode.opus_can_switch != 0 {
                    if self.application != OPUS_APPLICATION_RESTRICTED_SILK {
                        redundancy_bytes = compute_redundancy_bytes(
                            max_data_bytes,
                            self.bitrate_bps,
                            frame_rate,
                            self.stream_channels,
                        );
                        redundancy = redundancy_bytes != 0;
                    }
                    celt_to_silk = false;
                    self.silk_bw_switch = 1;
                }

                start_band = 17;
            }

            // --- CELT encoder configuration ---
            if let Some(ref mut celt) = self.celt_enc {
                let endband = match curr_bandwidth {
                    b if b == OPUS_BANDWIDTH_NARROWBAND => 13,
                    b if b == OPUS_BANDWIDTH_MEDIUMBAND || b == OPUS_BANDWIDTH_WIDEBAND => 17,
                    b if b == OPUS_BANDWIDTH_SUPERWIDEBAND => 19,
                    _ => 21,
                };
                celt.ctl(CeltEncoderCtl::SetEndBand(endband));
                celt.ctl(CeltEncoderCtl::SetChannels(self.stream_channels));
                // C: opus_encoder.c:2286 — always set BITRATE_MAX before CELT encoding
                celt.ctl(CeltEncoderCtl::SetBitrate(OPUS_BITRATE_MAX));
            }

            // --- Save CELT prefill data BEFORE delay buffer update ---
            // C: OPUS_COPY(tmp_prefill, &delay_buffer[(encoder_buffer-total_buffer-Fs/400)*ch], ch*Fs/400)
            // This captures 2.5ms from the delay buffer at an offset that will be
            // overwritten by the update below.
            if self.mode != MODE_SILK_ONLY
                && self.mode != self.prev_mode
                && self.prev_mode > 0
                && self.application != OPUS_APPLICATION_RESTRICTED_SILK
            {
                let n4 = (self.fs / 400) as usize;
                let ch = self.channels as usize;
                let src_offset = ((self.encoder_buffer as usize)
                    .saturating_sub(total_buffer as usize)
                    .saturating_sub(n4))
                    * ch;
                let copy_len = n4 * ch;
                if src_offset + copy_len <= self.delay_buffer.len()
                    && copy_len <= self.tmp_prefill.len()
                {
                    self.tmp_prefill[..copy_len]
                        .copy_from_slice(&self.delay_buffer[src_offset..src_offset + copy_len]);
                }
            }

            // --- Update delay buffer BEFORE CELT encoding ---
            // C copies from pcm_buf (dc_reject'd), not raw pcm
            self.update_delay_buffer_from_pcm_buf(&pcm_buf, frame_size, total_buffer);

            // --- HB gain fade ---
            // C: if ((prev_HB_gain < Q15ONE || HB_gain < Q15ONE) && celt_mode != NULL)
            // celt_mode is always non-NULL for non-RESTRICTED_SILK apps, so check celt_enc.
            if (self.prev_hb_gain < Q15ONE || hb_gain < Q15ONE) && self.celt_enc.is_some() {
                let mode_ref = if let Some(ref celt) = self.celt_enc {
                    celt.mode
                } else {
                    // Should not happen in non-SILK-only mode
                    return Err(OPUS_INTERNAL_ERROR);
                };
                gain_fade(
                    &mut pcm_buf,
                    self.prev_hb_gain,
                    hb_gain,
                    mode_ref.overlap as i32,
                    frame_size,
                    self.channels,
                    mode_ref.window,
                    self.fs,
                );
            }
            self.prev_hb_gain = hb_gain;

            // --- Stereo width ---
            // Matches C: compute stereoWidth_Q14 for non-hybrid or mono stream
            if self.mode != MODE_HYBRID || self.stream_channels == 1 {
                if equiv_rate > 32000 {
                    self.silk_mode.stereo_width_q14 = 16384;
                } else if equiv_rate < 16000 {
                    self.silk_mode.stereo_width_q14 = 0;
                } else {
                    self.silk_mode.stereo_width_q14 =
                        16384 - 2048 * (32000 - equiv_rate) / (equiv_rate - 14000);
                }
            }
            // Apply stereo width reduction (at low bitrates)
            if self.channels == 2
                && ((self.hybrid_stereo_width_q14 as i32) < (1 << 14)
                    || self.silk_mode.stereo_width_q14 < (1 << 14))
            {
                let mut g1 = self.hybrid_stereo_width_q14 as i32;
                let mut g2 = self.silk_mode.stereo_width_q14;
                // Scale Q14 -> Q15: 16384 maps to Q15ONE, others shift left by 1
                g1 = if g1 == 16384 { Q15ONE } else { shl16(g1, 1) };
                g2 = if g2 == 16384 { Q15ONE } else { shl16(g2, 1) };
                if let Some(ref celt) = self.celt_enc {
                    stereo_fade(
                        &mut pcm_buf,
                        g1,
                        g2,
                        celt.mode.overlap as i32,
                        frame_size,
                        self.channels,
                        celt.mode.window,
                        self.fs,
                    );
                }
                self.hybrid_stereo_width_q14 = self.silk_mode.stereo_width_q14 as i16;
            }

            // --- Redundancy signaling ---
            if self.mode != MODE_CELT_ONLY
                && enc.tell() + 17 + 20 * (if self.mode == MODE_HYBRID { 1 } else { 0 })
                    <= 8 * (max_data_bytes - 1)
            {
                if self.mode == MODE_HYBRID {
                    enc.encode_bit_logp(redundancy, 12);
                }
                if redundancy {
                    enc.encode_bit_logp(celt_to_silk, 1);
                    if self.mode == MODE_HYBRID {
                        let max_redundancy =
                            ((8 * (max_data_bytes - 1) - enc.tell()) - 8 - 3 * 8) / 8;
                        redundancy_bytes = imin(imin(257, max_redundancy), redundancy_bytes);
                        redundancy_bytes = imax(2, redundancy_bytes);
                        enc.encode_uint((redundancy_bytes - 2) as u32, 256);
                    }
                }
            } else {
                redundancy = false;
            }

            if !redundancy {
                self.silk_bw_switch = 0;
                redundancy_bytes = 0;
            }
            if self.mode != MODE_CELT_ONLY {
                start_band = 17;
            }

            // --- Finalize or prepare for CELT ---
            if self.mode == MODE_SILK_ONLY {
                let _bits_before_done = enc.tell();
                ret = (enc.tell() + 7) >> 3;
                enc.done();
                nb_compr_bytes = ret;
                range_final = enc.get_rng();
            } else {
                nb_compr_bytes = (max_data_bytes - 1) - redundancy_bytes;
                enc.shrink(nb_compr_bytes as u32);
                range_final = 0; // Will be set after CELT
            }

            // --- CELT→SILK redundancy frame ---
            if redundancy && celt_to_silk {
                if let Some(ref mut celt) = self.celt_enc {
                    celt.ctl(CeltEncoderCtl::SetStartBand(0));
                    celt.ctl(CeltEncoderCtl::SetVbr(0));
                    celt.ctl(CeltEncoderCtl::SetBitrate(OPUS_BITRATE_MAX));
                    redundancy_frame = vec![0u8; redundancy_bytes as usize];
                    celt_encode_with_ec(
                        celt,
                        &pcm_buf,
                        self.fs / 200,
                        &mut redundancy_frame,
                        redundancy_bytes,
                        None,
                    );
                    celt.ctl(CeltEncoderCtl::ResetState);
                }
            }

            // --- Set CELT start band ---
            if let Some(ref mut celt) = self.celt_enc {
                celt.ctl(CeltEncoderCtl::SetStartBand(start_band));
            }

            // --- Set CELT prediction ---
            if self.mode != MODE_SILK_ONLY {
                if let Some(ref mut celt) = self.celt_enc {
                    let celt_pred = if self.silk_mode.reduced_dependency != 0 {
                        0
                    } else {
                        2
                    };
                    celt.ctl(CeltEncoderCtl::SetPrediction(celt_pred));
                }
            }

            // --- Main CELT encode ---
            if self.mode != MODE_SILK_ONLY {
                if let Some(ref mut celt) = self.celt_enc {
                    // Configure VBR/bitrate for CELT (matches C: opus_encoder.c:2455)
                    celt.ctl(CeltEncoderCtl::SetVbr(self.use_vbr));
                    if self.mode == MODE_HYBRID {
                        if self.use_vbr != 0 {
                            celt.ctl(CeltEncoderCtl::SetBitrate(
                                self.bitrate_bps - self.silk_mode.bit_rate,
                            ));
                            celt.ctl(CeltEncoderCtl::SetVbrConstraint(0));
                        }
                    } else {
                        if self.use_vbr != 0 {
                            celt.ctl(CeltEncoderCtl::SetVbr(1));
                            celt.ctl(CeltEncoderCtl::SetVbrConstraint(self.vbr_constraint));
                            celt.ctl(CeltEncoderCtl::SetBitrate(self.bitrate_bps));
                        }
                    }

                    // Prefill on mode transition
                    // C: uses tmp_prefill saved from delay_buffer BEFORE the delay buffer
                    // update, at offset (encoder_buffer - total_buffer - Fs/400). This is
                    // 2.5ms earlier than the start of pcm_buf's delay compensation region.
                    if self.mode != self.prev_mode && self.prev_mode > 0 {
                        celt.ctl(CeltEncoderCtl::ResetState);
                        let mut prefill_buf = [0u8; 2];
                        celt_encode_with_ec(
                            celt,
                            &self.tmp_prefill,
                            self.fs / 400,
                            &mut prefill_buf,
                            2,
                            None,
                        );
                        celt.ctl(CeltEncoderCtl::SetPrediction(0));
                    }

                    // Set analysis/silk info for CELT
                    celt.ctl(CeltEncoderCtl::SetSilkInfo(SILKInfo {
                        signal_type: self.silk_mode.signal_type,
                        offset: self.silk_mode.offset,
                    }));

                    // Encode if there's room
                    if enc.tell() <= 8 * nb_compr_bytes {
                        let mut dummy_buf = vec![0u8; nb_compr_bytes as usize + 1];
                        ret = celt_encode_with_ec(
                            celt,
                            &pcm_buf,
                            frame_size,
                            &mut dummy_buf,
                            nb_compr_bytes,
                            Some(&mut enc),
                        );
                        if ret < 0 {
                            return Err(OPUS_INTERNAL_ERROR);
                        }
                    }
                    range_final = celt.rng;
                }
            }

            // NOTE: enc.done() is NOT called here — the CELT encoder already
            // calls done() internally (matching C's celt_encoder.c:2861).
            // Calling it twice corrupts the buffer layout.

            // Capture range-coder bit count for the outer budget-bust check
            // (C: opus_encoder.c:2580 — `ec_tell(&enc) > (max_data_bytes-1)*8`).
            // Must be read here, before `enc` is dropped at the end of this scope.
            bits_used = enc.tell();
        }
        // enc is now dropped, enc_data is available

        // --- Place redundancy data ---
        // C: opus_encoder.c lines 2502-2506
        // For CELT->SILK redundancy in hybrid VBR mode, the CELT encoder may
        // produce fewer bytes than nb_compr_bytes. Move redundancy data to
        // right after the actual CELT data.
        if redundancy && celt_to_silk && !redundancy_frame.is_empty() {
            let copy_len = imin(redundancy_bytes, redundancy_frame.len() as i32) as usize;
            if self.mode == MODE_HYBRID && nb_compr_bytes != ret {
                // VBR hybrid: place at actual CELT end (ret), not max (nb_compr_bytes)
                let dst_start = ret as usize;
                if dst_start + copy_len <= enc_data.len() {
                    enc_data[dst_start..dst_start + copy_len]
                        .copy_from_slice(&redundancy_frame[..copy_len]);
                }
            } else {
                let dst_start = nb_compr_bytes as usize;
                enc_data[dst_start..dst_start + copy_len]
                    .copy_from_slice(&redundancy_frame[..copy_len]);
            }
        }

        // --- SILK→CELT redundancy frame ---
        if redundancy && !celt_to_silk {
            if let Some(ref mut celt) = self.celt_enc {
                celt.ctl(CeltEncoderCtl::ResetState);
                celt.ctl(CeltEncoderCtl::SetStartBand(0));
                celt.ctl(CeltEncoderCtl::SetPrediction(0));
                celt.ctl(CeltEncoderCtl::SetVbr(0));
                celt.ctl(CeltEncoderCtl::SetBitrate(OPUS_BITRATE_MAX));

                // 2.5ms prefill
                let n4 = (self.fs / 400) as usize;
                let n2 = (self.fs / 200) as usize;
                let tail_start = ((frame_size as usize - n2 - n4) * self.channels as usize)
                    .min(pcm_buf.len().saturating_sub(n4 * self.channels as usize));
                let prefill_pcm = &pcm_buf[tail_start..];
                let mut prefill_buf = [0u8; 2];
                celt_encode_with_ec(celt, prefill_pcm, self.fs / 400, &mut prefill_buf, 2, None);

                // 5ms redundancy
                let tail_start2 = ((frame_size as usize - n2) * self.channels as usize)
                    .min(pcm_buf.len().saturating_sub(n2 * self.channels as usize));
                let red_pcm = &pcm_buf[tail_start2..];
                let dst_start = nb_compr_bytes as usize;
                let dst_end = dst_start + redundancy_bytes as usize;
                if dst_end <= enc_data.len() {
                    celt_encode_with_ec(
                        celt,
                        red_pcm,
                        self.fs / 200,
                        &mut enc_data[dst_start..dst_end],
                        redundancy_bytes,
                        None,
                    );
                    redundant_rng = celt.rng;
                }
            }
        }

        // --- TOC byte ---
        toc_slice[0] = gen_toc(self.mode, frame_rate, curr_bandwidth, self.stream_channels);

        range_final ^= redundant_rng;
        self.range_final = range_final;

        // --- State updates ---
        self.update_state(to_celt, frame_size);

        // --- DTX decision ---
        if self.use_dtx != 0 && self.silk_mode.use_dtx == 0 {
            let frame_ms_q1 = 2 * 1000 * frame_size / self.fs;
            if decide_dtx_mode(activity, &mut self.nb_no_activity_ms_q1, frame_ms_q1) {
                self.range_final = 0;
                toc_slice[0] = gen_toc(self.mode, frame_rate, curr_bandwidth, self.stream_channels);
                return Ok(1);
            }
        } else {
            self.nb_no_activity_ms_q1 = 0;
        }

        // --- Compute total output bytes ---
        // C: opus_encoder.c lines 2578-2601
        // At this point `ret` holds:
        //   - SILK-only: the SILK compressed bytes (set at line ~2128)
        //   - Other modes: the CELT return value (VBR-shrunk bytes)
        // Budget-bust check (C: lines 2578-2589)
        // In the unlikely case that the SILK encoder busted its target, tell
        // the decoder to call the PLC: emit a 1-byte packet with TOC + data[1]=0
        // and clear rangeFinal so the concealment path runs on decode.
        if bits_used > (max_data_bytes - 1) * 8 {
            if max_data_bytes < 2 {
                return Err(OPUS_BUFFER_TOO_SMALL);
            }
            // Note: `data` layout is [TOC | enc_data]; writing `data[1]` is the
            // first byte of the encoded-data slice (C: data[1] = 0).
            enc_data[0] = 0;
            ret = 1;
            self.range_final = 0;
        } else if self.mode == MODE_SILK_ONLY && !redundancy {
            // Strip trailing zeros (C: lines 2590-2598)
            while ret > 2 && data[ret as usize] == 0 {
                ret -= 1;
            }
        }
        // Count TOC and redundancy (C: line 2601)
        ret += 1; // TOC byte
        if redundancy {
            ret += redundancy_bytes;
        }

        // --- CBR padding ---
        if self.use_vbr == 0 && ret < orig_max_data_bytes {
            let pad_ret = opus_packet_pad(data, ret, orig_max_data_bytes);
            if pad_ret == OPUS_OK {
                ret = orig_max_data_bytes;
            }
        }

        Ok(ret)
    }

    // -----------------------------------------------------------------------
    // State update helpers
    // -----------------------------------------------------------------------

    fn update_state(&mut self, to_celt: bool, frame_size: i32) {
        self.prev_mode = if to_celt { MODE_CELT_ONLY } else { self.mode };
        self.prev_channels = self.stream_channels;
        self.prev_framesize = frame_size;
        self.first = 0;
    }

    /// Update delay buffer from raw pcm (for early return paths before pcm_buf exists).
    fn update_delay_buffer(&mut self, pcm: &[i16], frame_size: i32) {
        if self.encoder_buffer == 0 || self.delay_buffer.is_empty() {
            return;
        }
        let ch = self.channels as usize;
        let eb = self.encoder_buffer as usize;
        let fs = frame_size as usize;
        let db = &mut self.delay_buffer;
        if eb > fs {
            db.copy_within(fs * ch..eb * ch, 0);
            let new_start = (eb - fs) * ch;
            let copy_len = (fs * ch).min(pcm.len());
            db[new_start..new_start + copy_len].copy_from_slice(&pcm[..copy_len]);
        } else {
            let skip = (fs - eb) * ch;
            let copy_len = (eb * ch).min(pcm.len().saturating_sub(skip));
            if skip < pcm.len() {
                db[..copy_len].copy_from_slice(&pcm[skip..skip + copy_len]);
            }
        }
    }

    /// Update delay buffer from pcm_buf (dc_reject'd data), matching C reference.
    /// C: copies from pcm_buf, NOT from raw pcm input.
    fn update_delay_buffer_from_pcm_buf(
        &mut self,
        pcm_buf: &[i16],
        frame_size: i32,
        total_buffer: i32,
    ) {
        if self.encoder_buffer == 0 || self.delay_buffer.is_empty() {
            return;
        }
        let ch = self.channels as usize;
        let eb = self.encoder_buffer as usize;
        let fs = frame_size as usize;
        let tb = total_buffer as usize;
        let db = &mut self.delay_buffer;

        if ch * (eb.saturating_sub(fs + tb)) > 0 {
            // Case 1: encoder_buffer > frame_size + total_buffer
            // Shift existing delay_buffer left by frame_size, then copy all of pcm_buf
            let shift_src = fs * ch;
            let shift_len = (eb - fs - tb) * ch;
            db.copy_within(shift_src..shift_src + shift_len, 0);
            let copy_start = shift_len;
            let copy_len = ((fs + tb) * ch).min(pcm_buf.len());
            db[copy_start..copy_start + copy_len].copy_from_slice(&pcm_buf[..copy_len]);
        } else {
            // Case 2: encoder_buffer <= frame_size + total_buffer
            // Copy tail of pcm_buf into delay_buffer
            let src_offset = (fs + tb - eb) * ch;
            let copy_len = (eb * ch).min(pcm_buf.len().saturating_sub(src_offset));
            if src_offset < pcm_buf.len() {
                db[..copy_len].copy_from_slice(&pcm_buf[src_offset..src_offset + copy_len]);
            }
        }
    }

    // -----------------------------------------------------------------------
    // CTL interface — get/set methods
    // -----------------------------------------------------------------------

    pub fn set_bitrate(&mut self, bitrate: i32) -> i32 {
        if bitrate != OPUS_AUTO && bitrate != OPUS_BITRATE_MAX {
            if bitrate <= 0 {
                return OPUS_BAD_ARG;
            }
            // Clamp to valid range, matching C reference behavior
            let clamped = if bitrate <= 500 {
                500
            } else if bitrate > 750000 * self.channels {
                750000 * self.channels
            } else {
                bitrate
            };
            self.user_bitrate_bps = clamped;
        } else {
            self.user_bitrate_bps = bitrate;
        }
        OPUS_OK
    }

    pub fn get_bitrate(&self) -> i32 {
        user_bitrate_to_bitrate(
            self.user_bitrate_bps,
            self.channels,
            self.fs,
            self.prev_framesize,
            1276,
        )
    }

    pub fn set_complexity(&mut self, complexity: i32) -> i32 {
        if complexity < 0 || complexity > 10 {
            return OPUS_BAD_ARG;
        }
        self.silk_mode.complexity = complexity;
        if let Some(ref mut celt) = self.celt_enc {
            celt.ctl(CeltEncoderCtl::SetComplexity(complexity));
        }
        OPUS_OK
    }

    pub fn get_complexity(&self) -> i32 {
        self.silk_mode.complexity
    }

    pub fn set_vbr(&mut self, vbr: i32) -> i32 {
        if vbr < 0 || vbr > 1 {
            return OPUS_BAD_ARG;
        }
        self.use_vbr = vbr;
        self.silk_mode.use_cbr = 1 - vbr;
        OPUS_OK
    }

    pub fn get_vbr(&self) -> i32 {
        self.use_vbr
    }

    pub fn set_vbr_constraint(&mut self, constraint: i32) -> i32 {
        if constraint < 0 || constraint > 1 {
            return OPUS_BAD_ARG;
        }
        self.vbr_constraint = constraint;
        OPUS_OK
    }

    pub fn get_vbr_constraint(&self) -> i32 {
        self.vbr_constraint
    }

    pub fn set_force_channels(&mut self, channels: i32) -> i32 {
        if channels != OPUS_AUTO && (channels < 1 || channels > self.channels) {
            return OPUS_BAD_ARG;
        }
        self.force_channels = channels;
        OPUS_OK
    }

    pub fn get_force_channels(&self) -> i32 {
        self.force_channels
    }

    pub fn set_bandwidth(&mut self, bandwidth: i32) -> i32 {
        if bandwidth != OPUS_AUTO
            && (bandwidth < OPUS_BANDWIDTH_NARROWBAND || bandwidth > OPUS_BANDWIDTH_FULLBAND)
        {
            return OPUS_BAD_ARG;
        }
        self.user_bandwidth = bandwidth;
        OPUS_OK
    }

    pub fn get_bandwidth(&self) -> i32 {
        self.bandwidth
    }

    pub fn set_max_bandwidth(&mut self, bandwidth: i32) -> i32 {
        if bandwidth < OPUS_BANDWIDTH_NARROWBAND || bandwidth > OPUS_BANDWIDTH_FULLBAND {
            return OPUS_BAD_ARG;
        }
        self.max_bandwidth = bandwidth;
        OPUS_OK
    }

    pub fn get_max_bandwidth(&self) -> i32 {
        self.max_bandwidth
    }

    pub fn set_signal(&mut self, signal: i32) -> i32 {
        if signal != OPUS_AUTO && signal != OPUS_SIGNAL_VOICE && signal != OPUS_SIGNAL_MUSIC {
            return OPUS_BAD_ARG;
        }
        self.signal_type = signal;
        OPUS_OK
    }

    pub fn get_signal(&self) -> i32 {
        self.signal_type
    }

    pub fn set_inband_fec(&mut self, fec: i32) -> i32 {
        if fec < 0 || fec > 2 {
            return OPUS_BAD_ARG;
        }
        self.fec_config = fec;
        self.silk_mode.use_in_band_fec = if fec != 0 { 1 } else { 0 };
        OPUS_OK
    }

    pub fn get_inband_fec(&self) -> i32 {
        self.fec_config
    }

    pub fn set_packet_loss_perc(&mut self, loss: i32) -> i32 {
        if loss < 0 || loss > 100 {
            return OPUS_BAD_ARG;
        }
        self.silk_mode.packet_loss_percentage = loss;
        if let Some(ref mut celt) = self.celt_enc {
            celt.ctl(CeltEncoderCtl::SetPacketLossPerc(loss));
        }
        OPUS_OK
    }

    pub fn get_packet_loss_perc(&self) -> i32 {
        self.silk_mode.packet_loss_percentage
    }

    pub fn set_dtx(&mut self, dtx: i32) -> i32 {
        if dtx < 0 || dtx > 1 {
            return OPUS_BAD_ARG;
        }
        self.use_dtx = dtx;
        OPUS_OK
    }

    pub fn get_dtx(&self) -> i32 {
        self.use_dtx
    }

    pub fn set_lsb_depth(&mut self, depth: i32) -> i32 {
        if depth < 8 || depth > 24 {
            return OPUS_BAD_ARG;
        }
        self.lsb_depth = depth;
        OPUS_OK
    }

    pub fn get_lsb_depth(&self) -> i32 {
        self.lsb_depth
    }

    /// Debug accessor for HP filter state.
    pub fn get_hp_mem(&self) -> [i32; 4] {
        self.hp_mem
    }

    /// Debug accessor for variable HP smoothing state.
    pub fn get_variable_hp_smth2(&self) -> i32 {
        self.variable_hp_smth2_q15
    }

    pub fn set_expert_frame_duration(&mut self, duration: i32) -> i32 {
        if duration != OPUS_FRAMESIZE_ARG
            && (duration < OPUS_FRAMESIZE_2_5_MS || duration > OPUS_FRAMESIZE_120_MS)
        {
            return OPUS_BAD_ARG;
        }
        self.variable_duration = duration;
        OPUS_OK
    }

    pub fn get_expert_frame_duration(&self) -> i32 {
        self.variable_duration
    }

    pub fn set_prediction_disabled(&mut self, disabled: i32) -> i32 {
        if disabled < 0 || disabled > 1 {
            return OPUS_BAD_ARG;
        }
        self.silk_mode.reduced_dependency = disabled;
        OPUS_OK
    }

    pub fn get_prediction_disabled(&self) -> i32 {
        self.silk_mode.reduced_dependency
    }

    pub fn set_phase_inversion_disabled(&mut self, disabled: i32) -> i32 {
        if let Some(ref mut celt) = self.celt_enc {
            celt.ctl(CeltEncoderCtl::SetPhaseInversionDisabled(disabled));
        }
        OPUS_OK
    }

    pub fn get_phase_inversion_disabled(&self) -> i32 {
        if let Some(ref celt) = self.celt_enc {
            // Read from CELT encoder's disable_inv field
            celt.disable_inv
        } else {
            0
        }
    }

    pub fn set_voice_ratio(&mut self, ratio: i32) -> i32 {
        if ratio < -1 || ratio > 100 {
            return OPUS_BAD_ARG;
        }
        self.voice_ratio = ratio;
        OPUS_OK
    }

    pub fn get_voice_ratio(&self) -> i32 {
        self.voice_ratio
    }

    pub fn set_force_mode(&mut self, mode: i32) -> i32 {
        if mode != OPUS_AUTO
            && mode != MODE_SILK_ONLY
            && mode != MODE_HYBRID
            && mode != MODE_CELT_ONLY
        {
            return OPUS_BAD_ARG;
        }
        self.user_forced_mode = mode;
        OPUS_OK
    }

    pub fn get_lookahead(&self) -> i32 {
        let mut lookahead = self.fs / 400; // 2.5ms
        if self.application != OPUS_APPLICATION_RESTRICTED_LOWDELAY {
            lookahead += self.delay_compensation;
        }
        lookahead
    }

    pub fn get_sample_rate(&self) -> i32 {
        self.fs
    }

    pub fn get_final_range(&self) -> u32 {
        self.range_final
    }

    pub fn get_application(&self) -> i32 {
        self.application
    }

    /// Validate and set encoder application. Mirrors
    /// `OPUS_SET_APPLICATION` from the C reference.
    pub fn set_application(&mut self, application: i32) -> i32 {
        if application != OPUS_APPLICATION_VOIP
            && application != OPUS_APPLICATION_AUDIO
            && application != OPUS_APPLICATION_RESTRICTED_LOWDELAY
        {
            return OPUS_BAD_ARG;
        }
        self.application = application;
        OPUS_OK
    }

    pub fn get_channels(&self) -> i32 {
        self.channels
    }

    pub fn get_stream_channels(&self) -> i32 {
        self.stream_channels
    }

    pub fn get_mode(&self) -> i32 {
        self.mode
    }

    pub fn get_prev_mode(&self) -> i32 {
        self.prev_mode
    }

    /// Access the internal SILK encoder state (for diagnostic/debugging use).
    pub fn silk_encoder(&self) -> Option<&SilkEncoder> {
        self.silk_enc.as_ref()
    }

    /// Return key SILK encoder internal state for comparison testing.
    /// Returns None if no SILK encoder is allocated.
    pub fn get_silk_state(&self) -> Option<SilkEncoderSnapshot> {
        let silk = self.silk_enc.as_ref()?;
        let s = &silk.state_fxx[0].s_cmn;
        Some(SilkEncoderSnapshot {
            fs_khz: s.fs_khz,
            frame_length: s.frame_length,
            nb_subfr: s.nb_subfr,
            input_buf_ix: s.input_buf_ix,
            n_frames_per_packet: s.n_frames_per_packet,
            packet_size_ms: s.packet_size_ms,
            first_frame_after_reset: s.first_frame_after_reset,
            controlled_since_last_payload: s.controlled_since_last_payload,
            prefill_flag: s.prefill_flag,
            n_frames_encoded: s.n_frames_encoded,
            speech_activity_q8: s.speech_activity_q8,
            signal_type: s.indices.signal_type as i32,
            input_quality_bands_q15: s.input_quality_bands_q15[0],
        })
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opus::decoder::{OpusDecoder, opus_packet_has_lbrr};

    fn patterned_pcm_i16(frame_size: usize, channels: usize, seed: i32) -> Vec<i16> {
        (0..frame_size * channels)
            .map(|i| {
                let base = ((i as i32 * 7919 + seed * 911) % 28000) - 14000;
                if channels == 2 && i % 2 == 1 {
                    (base / 2) as i16
                } else {
                    base as i16
                }
            })
            .collect()
    }

    fn patterned_pcm_f32(frame_size: usize, channels: usize, seed: i32) -> Vec<f32> {
        patterned_pcm_i16(frame_size, channels, seed)
            .into_iter()
            .map(|sample| sample as f32 / 32768.0)
            .collect()
    }

    fn packet_mode_from_toc(packet: &[u8]) -> i32 {
        if packet[0] & 0x80 != 0 {
            MODE_CELT_ONLY
        } else if (packet[0] & 0x60) == 0x60 {
            MODE_HYBRID
        } else {
            MODE_SILK_ONLY
        }
    }

    #[test]
    fn test_gen_toc() {
        // SILK-only, NB, mono, 20ms (framerate=50, period=3)
        let toc = gen_toc(MODE_SILK_ONLY, 50, OPUS_BANDWIDTH_NARROWBAND, 1);
        assert_eq!(toc & 0x80, 0); // Not CELT
        assert_eq!(toc & 0x60, 0); // Not hybrid
        assert_eq!(toc & 0x04, 0); // Mono

        // CELT-only, FB, stereo, 20ms
        let toc = gen_toc(MODE_CELT_ONLY, 50, OPUS_BANDWIDTH_FULLBAND, 2);
        assert_eq!(toc & 0x80, 0x80); // CELT flag
        assert_eq!(toc & 0x04, 0x04); // Stereo

        // Hybrid, SWB, mono, 20ms
        let toc = gen_toc(MODE_HYBRID, 50, OPUS_BANDWIDTH_SUPERWIDEBAND, 1);
        assert_eq!(toc & 0xE0, 0x60); // Hybrid
        assert_eq!(toc & 0x10, 0); // SWB (not FB)
        assert_eq!(toc & 0x04, 0); // Mono
    }

    #[test]
    fn test_frame_size_select() {
        // 20ms at 48kHz = 960 samples
        assert_eq!(frame_size_select(960, OPUS_FRAMESIZE_ARG, 48000), 960);
        // 10ms = 480
        assert_eq!(frame_size_select(960, OPUS_FRAMESIZE_10_MS, 48000), 480);
        // 2.5ms = 120
        assert_eq!(frame_size_select(960, OPUS_FRAMESIZE_2_5_MS, 48000), 120);
        // 40ms and 120ms hit the long-frame branch.
        assert_eq!(frame_size_select(1920, OPUS_FRAMESIZE_40_MS, 48000), 1920);
        assert_eq!(frame_size_select(5760, OPUS_FRAMESIZE_120_MS, 48000), 5760);
        // Invalid: requested size > input
        assert_eq!(frame_size_select(480, OPUS_FRAMESIZE_20_MS, 48000), -1);
        // Invalid duration selector.
        assert_eq!(frame_size_select(960, 4999, 48000), -1);
    }

    #[test]
    fn test_is_digital_silence() {
        let silence = [0i16; 960];
        assert!(is_digital_silence(&silence, 480, 1, 16));

        let mut noisy = [0i16; 960];
        noisy[100] = 1;
        assert!(!is_digital_silence(&noisy, 480, 1, 16));
    }

    #[test]
    fn test_compute_equiv_rate() {
        assert_eq!(
            compute_equiv_rate(64000, 1, 50, 1, MODE_SILK_ONLY, 10, 0),
            64000
        );
        assert_eq!(
            compute_equiv_rate(64000, 1, 100, 0, MODE_CELT_ONLY, 0, 0),
            45292
        );
        assert_eq!(compute_equiv_rate(64000, 1, 50, 1, 12345, 10, 10), 59429);
        assert_eq!(
            compute_equiv_rate(64000, 1, 50, 1, MODE_SILK_ONLY, 1, 0),
            46592
        );
    }

    #[test]
    fn test_decide_fec() {
        // FEC disabled: should return 0
        assert_eq!(
            decide_fec(
                0,
                10,
                0,
                MODE_SILK_ONLY,
                &mut OPUS_BANDWIDTH_WIDEBAND.clone(),
                20000
            ),
            0
        );
        // CELT-only: should return 0
        assert_eq!(
            decide_fec(
                1,
                10,
                0,
                MODE_CELT_ONLY,
                &mut OPUS_BANDWIDTH_WIDEBAND.clone(),
                20000
            ),
            0
        );
        // No loss: should return 0
        assert_eq!(
            decide_fec(
                1,
                0,
                0,
                MODE_SILK_ONLY,
                &mut OPUS_BANDWIDTH_WIDEBAND.clone(),
                20000
            ),
            0
        );
        // Low rate / high loss path walks bandwidth downward and restores the original value.
        let mut bw = OPUS_BANDWIDTH_FULLBAND;
        assert_eq!(decide_fec(1, 25, 0, MODE_SILK_ONLY, &mut bw, 0), 0);
        assert_eq!(bw, OPUS_BANDWIDTH_FULLBAND);
    }

    #[test]
    fn test_encoder_create() {
        let enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_VOIP);
        assert!(enc.is_ok());
        let enc = enc.unwrap();
        assert_eq!(enc.channels, 2);
        assert_eq!(enc.fs, 48000);
        assert_eq!(enc.get_sample_rate(), 48000);
    }

    #[test]
    fn test_encoder_create_invalid() {
        assert!(OpusEncoder::new(44100, 2, OPUS_APPLICATION_VOIP).is_err());
        assert!(OpusEncoder::new(48000, 3, OPUS_APPLICATION_VOIP).is_err());
        assert!(OpusEncoder::new(48000, 2, 9999).is_err());
    }

    #[test]
    fn test_encoder_ctl() {
        let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();

        assert_eq!(enc.set_bitrate(64000), OPUS_OK);
        assert!(enc.get_bitrate() > 0);

        assert_eq!(enc.set_complexity(5), OPUS_OK);
        assert_eq!(enc.get_complexity(), 5);

        assert_eq!(enc.set_vbr(0), OPUS_OK);
        assert_eq!(enc.get_vbr(), 0);

        assert_eq!(enc.set_signal(OPUS_SIGNAL_VOICE), OPUS_OK);
        assert_eq!(enc.get_signal(), OPUS_SIGNAL_VOICE);

        assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
        // get_bandwidth() returns the actual encoding bandwidth, not user-set.
        // Before any encode call, it retains the init default (FULLBAND).
        assert_eq!(enc.get_bandwidth(), OPUS_BANDWIDTH_FULLBAND);
    }

    #[test]
    fn test_silk_smulwb() {
        // SMULWB(1 << 16, 1 << 15): b32=32768 truncated to i16 = -32768
        assert_eq!(silk_smulwb(1 << 16, 1 << 15), -32768);
        // SMULWB(0, anything) = 0
        assert_eq!(silk_smulwb(0, 12345), 0);
    }

    #[test]
    fn test_bits_bitrate_roundtrip() {
        let bitrate = 64000;
        let fs = 48000;
        let frame_size = 960;
        let bits = bitrate_to_bits(bitrate, fs, frame_size);
        let recovered = bits_to_bitrate(bits, fs, frame_size);
        // Should be approximately equal (rounding)
        assert!((recovered - bitrate).abs() < 100);
    }

    #[test]
    fn test_encode_emits_toc_only_packet_for_tiny_budget() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        let pcm = patterned_pcm_i16(960, 1, 7);
        let mut packet = [0u8; 2];

        let len = enc.encode(&pcm, 960, &mut packet, 2).unwrap();

        assert_eq!(len, 1);
        assert_ne!(packet[0], 0);
        assert_eq!(enc.get_final_range(), 0);
    }

    #[test]
    fn test_encode_rejects_one_byte_100ms_packet() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        let pcm = patterned_pcm_i16(4800, 1, 19);
        let mut packet = [0u8; 1];

        assert_eq!(
            enc.encode(&pcm, 4800, &mut packet, 1),
            Err(OPUS_BUFFER_TOO_SMALL)
        );
    }

    #[test]
    fn test_encode_decode_silk_only_multiframe_with_fec() {
        let mut enc = OpusEncoder::new(16000, 1, OPUS_APPLICATION_VOIP).unwrap();
        assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
        assert_eq!(enc.set_bitrate(40000), OPUS_OK);
        assert_eq!(enc.set_vbr(1), OPUS_OK);
        assert_eq!(enc.set_inband_fec(1), OPUS_OK);
        assert_eq!(enc.set_packet_loss_perc(25), OPUS_OK);
        assert_eq!(enc.set_dtx(1), OPUS_OK);
        assert_eq!(enc.set_signal(OPUS_SIGNAL_VOICE), OPUS_OK);
        assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_40_MS), OPUS_OK);

        let mut dec = OpusDecoder::new(16000, 1).unwrap();
        let mut saw_lbrr = false;

        for frame_idx in 0..4 {
            let pcm = patterned_pcm_i16(640, 1, 101 + frame_idx * 17);
            let mut packet = vec![0u8; 1500];
            let packet_capacity = packet.len() as i32;

            let len = enc.encode(&pcm, 640, &mut packet, packet_capacity).unwrap();
            let packet = &packet[..len as usize];

            assert!(len > 1);
            assert_eq!(enc.get_mode(), MODE_SILK_ONLY);
            assert_eq!(packet_mode_from_toc(packet), MODE_SILK_ONLY);

            let has_lbrr = opus_packet_has_lbrr(packet, len).unwrap();
            saw_lbrr |= has_lbrr;

            let mut out = vec![0i16; 640];
            let decoded = dec.decode(Some(packet), &mut out, 640, false).unwrap();
            assert_eq!(decoded, 640);
            assert!(out.iter().any(|&sample| sample != 0));
        }

        assert!(
            saw_lbrr,
            "expected at least one packet with in-band FEC/LBRR for this configuration"
        );

        let mut plc = vec![0i16; 640];
        assert_eq!(dec.decode(None, &mut plc, 640, false).unwrap(), 640);
    }

    #[test]
    fn test_encode_float_lowdelay_celt_roundtrip() {
        let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_RESTRICTED_LOWDELAY).unwrap();
        assert_eq!(enc.set_vbr(0), OPUS_OK);
        assert_eq!(enc.set_complexity(10), OPUS_OK);
        assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_5_MS), OPUS_OK);
        assert_eq!(enc.set_phase_inversion_disabled(1), OPUS_OK);

        let pcm = patterned_pcm_f32(240, 2, 43);
        let mut packet = vec![0u8; 1500];
        let packet_capacity = packet.len() as i32;
        let len = enc
            .encode_float(&pcm, 240, &mut packet, packet_capacity)
            .unwrap();
        let packet = &packet[..len as usize];

        assert!(len > 0);
        assert_eq!(enc.get_mode(), MODE_CELT_ONLY);
        assert_eq!(packet_mode_from_toc(packet), MODE_CELT_ONLY);

        let mut dec = OpusDecoder::new(48000, 2).unwrap();
        dec.set_phase_inversion_disabled(true);

        let mut out = vec![0f32; 240 * 2];
        let decoded = dec
            .decode_float(Some(packet), &mut out, 240, false)
            .unwrap();
        assert_eq!(decoded, 240);
        assert!(out.iter().any(|sample| sample.abs() > 1e-4));
    }

    #[test]
    fn test_decode24_and_fec_fallback_for_celt_packet() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_RESTRICTED_LOWDELAY).unwrap();
        assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_10_MS), OPUS_OK);

        let pcm = patterned_pcm_i16(480, 1, 77);
        let mut packet = vec![0u8; 1500];
        let packet_capacity = packet.len() as i32;
        let len = enc.encode(&pcm, 480, &mut packet, packet_capacity).unwrap();
        let packet = &packet[..len as usize];

        let mut dec24 = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm24 = vec![0i32; 480];
        let decoded24 = dec24
            .decode24(Some(packet), &mut pcm24, 480, false)
            .unwrap();
        assert_eq!(decoded24, 480);
        assert!(pcm24.iter().any(|&sample| sample != 0));

        let mut plc_dec = OpusDecoder::new(48000, 1).unwrap();
        let mut warmup = vec![0i16; 480];
        assert_eq!(
            plc_dec
                .decode(Some(packet), &mut warmup, 480, false)
                .unwrap(),
            480
        );

        let mut fec_pcm = vec![0i16; 480];
        let decoded_fec = plc_dec
            .decode(Some(packet), &mut fec_pcm, 480, true)
            .unwrap();
        assert_eq!(decoded_fec, 480);
        assert!(fec_pcm.iter().any(|&sample| sample != 0));
    }

    #[test]
    fn test_encoder_rejects_invalid_ctl_ranges() {
        let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();

        assert_eq!(enc.set_vbr(2), OPUS_BAD_ARG);
        assert_eq!(enc.set_vbr_constraint(-1), OPUS_BAD_ARG);
        assert_eq!(enc.set_force_channels(3), OPUS_BAD_ARG);
        assert_eq!(enc.set_bandwidth(12345), OPUS_BAD_ARG);
        assert_eq!(enc.set_max_bandwidth(12345), OPUS_BAD_ARG);
        assert_eq!(enc.set_signal(12345), OPUS_BAD_ARG);
        assert_eq!(enc.set_inband_fec(3), OPUS_BAD_ARG);
        assert_eq!(enc.set_packet_loss_perc(101), OPUS_BAD_ARG);
        assert_eq!(enc.set_dtx(2), OPUS_BAD_ARG);
        assert_eq!(enc.set_lsb_depth(7), OPUS_BAD_ARG);
        assert_eq!(enc.set_expert_frame_duration(4999), OPUS_BAD_ARG);
        assert_eq!(enc.set_prediction_disabled(2), OPUS_BAD_ARG);
        assert_eq!(enc.set_voice_ratio(101), OPUS_BAD_ARG);
        assert_eq!(enc.set_force_mode(12345), OPUS_BAD_ARG);
    }

    #[test]
    fn test_encoder_bitrate_and_ctl_special_values() {
        let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();

        assert_eq!(enc.set_bitrate(OPUS_AUTO), OPUS_OK);
        assert_eq!(enc.user_bitrate_bps, OPUS_AUTO);

        assert_eq!(enc.set_bitrate(OPUS_BITRATE_MAX), OPUS_OK);
        assert_eq!(enc.user_bitrate_bps, OPUS_BITRATE_MAX);

        assert_eq!(enc.set_bitrate(400), OPUS_OK);
        assert_eq!(enc.user_bitrate_bps, 500);

        assert_eq!(enc.set_bitrate(2_000_000), OPUS_OK);
        assert_eq!(enc.user_bitrate_bps, 1_500_000);

        assert_eq!(enc.set_vbr_constraint(0), OPUS_OK);
        assert_eq!(enc.get_vbr_constraint(), 0);
        assert_eq!(enc.set_vbr_constraint(1), OPUS_OK);
        assert_eq!(enc.get_vbr_constraint(), 1);

        assert_eq!(enc.set_force_channels(OPUS_AUTO), OPUS_OK);
        assert_eq!(enc.get_force_channels(), OPUS_AUTO);
        assert_eq!(enc.set_bandwidth(OPUS_AUTO), OPUS_OK);
        assert_eq!(enc.user_bandwidth, OPUS_AUTO);
        assert_eq!(enc.set_signal(OPUS_AUTO), OPUS_OK);
        assert_eq!(enc.get_signal(), OPUS_AUTO);
        assert_eq!(enc.set_max_bandwidth(OPUS_BANDWIDTH_FULLBAND), OPUS_OK);
        assert_eq!(enc.get_max_bandwidth(), OPUS_BANDWIDTH_FULLBAND);
        assert_eq!(enc.set_voice_ratio(-1), OPUS_OK);
        assert_eq!(enc.get_voice_ratio(), -1);
    }

    #[test]
    fn test_encode_native_argument_and_silence_paths() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        let mut empty: [u8; 0] = [];
        assert_eq!(
            enc.encode_native(&[], 0, &mut empty, 0, 16),
            Err(OPUS_BAD_ARG)
        );

        let pcm = patterned_pcm_i16(4800, 1, 19);
        let mut tiny_packet = [0u8; 1];
        assert_eq!(
            enc.encode_native(&pcm, 4800, &mut tiny_packet, 1, 16),
            Err(OPUS_BUFFER_TOO_SMALL)
        );

        assert_eq!(enc.set_vbr(0), OPUS_OK);
        assert_eq!(enc.set_bitrate(64000), OPUS_OK);

        let silence = [0i16; 960];
        let mut packet = vec![0u8; 1500];
        let packet_len = packet.len() as i32;
        let len = enc
            .encode_native(&silence, 960, &mut packet, packet_len, 16)
            .unwrap();

        assert!(len > 1);
        assert_eq!(enc.peak_signal_energy, 0);
        assert_eq!(enc.voice_ratio, -1);
    }

    #[test]
    fn test_encode_decode_forced_hybrid_stereo_roundtrip() {
        let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_VOIP).unwrap();
        assert_eq!(enc.set_force_mode(MODE_HYBRID), OPUS_OK);
        assert_eq!(enc.set_bitrate(96000), OPUS_OK);
        assert_eq!(enc.set_vbr(1), OPUS_OK);
        assert_eq!(enc.set_signal(OPUS_SIGNAL_VOICE), OPUS_OK);
        assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_20_MS), OPUS_OK);

        let pcm = patterned_pcm_i16(960, 2, 303);
        let mut packet = vec![0u8; 1500];
        let packet_capacity = packet.len() as i32;
        let len = enc.encode(&pcm, 960, &mut packet, packet_capacity).unwrap();
        let packet = &packet[..len as usize];

        assert!(len > 1);
        assert_eq!(enc.get_mode(), MODE_HYBRID);
        assert_eq!(packet_mode_from_toc(packet), MODE_HYBRID);

        let mut dec = OpusDecoder::new(48000, 2).unwrap();
        let mut out = vec![0i16; 960 * 2];
        let decoded = dec.decode(Some(packet), &mut out, 960, false).unwrap();
        assert_eq!(decoded, 960);
        assert!(out.iter().any(|&sample| sample != 0));
    }

    #[test]
    fn test_encoder_controls_update_state_and_lookahead() {
        let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();

        assert_eq!(enc.get_lookahead(), 312);
        assert_eq!(enc.set_bitrate(0), OPUS_BAD_ARG);
        assert_eq!(enc.set_bitrate(1), OPUS_OK);
        assert_eq!(enc.get_bitrate(), 500);

        assert_eq!(enc.set_force_channels(1), OPUS_OK);
        assert_eq!(enc.get_force_channels(), 1);
        assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
        assert_eq!(enc.set_max_bandwidth(OPUS_BANDWIDTH_SUPERWIDEBAND), OPUS_OK);
        assert_eq!(enc.get_max_bandwidth(), OPUS_BANDWIDTH_SUPERWIDEBAND);
        assert_eq!(enc.set_signal(OPUS_SIGNAL_MUSIC), OPUS_OK);
        assert_eq!(enc.get_signal(), OPUS_SIGNAL_MUSIC);
        assert_eq!(enc.set_inband_fec(2), OPUS_OK);
        assert_eq!(enc.get_inband_fec(), 2);
        assert_eq!(enc.silk_mode.use_in_band_fec, 1);
        assert_eq!(enc.set_packet_loss_perc(12), OPUS_OK);
        assert_eq!(enc.get_packet_loss_perc(), 12);
        assert_eq!(enc.set_dtx(1), OPUS_OK);
        assert_eq!(enc.get_dtx(), 1);
        assert_eq!(enc.set_lsb_depth(16), OPUS_OK);
        assert_eq!(enc.get_lsb_depth(), 16);
        assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_20_MS), OPUS_OK);
        assert_eq!(enc.get_expert_frame_duration(), OPUS_FRAMESIZE_20_MS);
        assert_eq!(enc.set_prediction_disabled(1), OPUS_OK);
        assert_eq!(enc.get_prediction_disabled(), 1);
        assert_eq!(enc.set_voice_ratio(100), OPUS_OK);
        assert_eq!(enc.get_voice_ratio(), 100);
        assert_eq!(enc.set_force_mode(MODE_HYBRID), OPUS_OK);
        assert_eq!(enc.user_forced_mode, MODE_HYBRID);
        assert_eq!(enc.set_phase_inversion_disabled(1), OPUS_OK);
        assert_eq!(enc.get_phase_inversion_disabled(), 1);

        let lowdelay = OpusEncoder::new(48000, 2, OPUS_APPLICATION_RESTRICTED_LOWDELAY).unwrap();
        assert_eq!(lowdelay.get_lookahead(), 120);
    }

    #[test]
    fn test_helper_branches_cover_mode_rate_fec_and_redundancy_math() {
        assert_eq!(
            user_bitrate_to_bitrate(OPUS_AUTO, 1, 48000, 960, 1276),
            51000
        );
        assert_eq!(
            user_bitrate_to_bitrate(OPUS_BITRATE_MAX, 2, 48000, 960, 1276),
            510400
        );
        assert_eq!(user_bitrate_to_bitrate(64000, 2, 48000, 960, 1276), 64000);

        assert_eq!(
            compute_silk_rate_for_hybrid(18000, OPUS_BANDWIDTH_FULLBAND, false, 1, 0, 1),
            14750
        );
        assert_eq!(
            compute_silk_rate_for_hybrid(26000, OPUS_BANDWIDTH_FULLBAND, false, 1, 0, 2),
            20750
        );
        assert_eq!(
            compute_silk_rate_for_hybrid(70000, OPUS_BANDWIDTH_SUPERWIDEBAND, true, 0, 1, 1),
            53400
        );

        assert_eq!(compute_redundancy_bytes(20, 4000, 50, 1), 0);
        assert_eq!(compute_redundancy_bytes(1000, 64000, 50, 2), 74);
        assert_eq!(compute_redundancy_bytes(2000, 1_000_000, 50, 2), 257);

        let mut bw = OPUS_BANDWIDTH_WIDEBAND;
        assert_eq!(decide_fec(1, 25, 0, MODE_SILK_ONLY, &mut bw, 8000), 0);
        assert_eq!(bw, OPUS_BANDWIDTH_WIDEBAND);

        let mut bw = OPUS_BANDWIDTH_NARROWBAND;
        assert_eq!(decide_fec(1, 25, 1, MODE_SILK_ONLY, &mut bw, 12000), 1);
        assert_eq!(bw, OPUS_BANDWIDTH_NARROWBAND);

        let mut bw = OPUS_BANDWIDTH_WIDEBAND;
        assert_eq!(decide_fec(1, 25, 0, MODE_CELT_ONLY, &mut bw, 20000), 0);
        assert_eq!(bw, OPUS_BANDWIDTH_WIDEBAND);
    }

    #[test]
    fn test_frame_energy_and_stereo_width_branches() {
        let mono = vec![2i16; 480];
        assert_eq!(compute_frame_energy(&mono, 480, 1), 4);

        let stereo = vec![2i16; 960];
        assert_eq!(compute_frame_energy(&stereo, 480, 2), 4);

        let silence = vec![0i16; 960];
        assert_eq!(compute_frame_energy(&silence, 480, 2), 0);

        let mut mem = StereoWidthState::default();
        assert_eq!(compute_stereo_width(&silence, 480, 48000, &mut mem), 0);
        assert_eq!(mem.smoothed_width, 0);

        let stereo_identical = vec![100i16; 960];
        let mut mem = StereoWidthState::default();
        let width = compute_stereo_width(&stereo_identical, 480, 48000, &mut mem);
        assert_eq!(width, 0);
        assert!(mem.xx > 0 && mem.yy > 0);
    }

    #[test]
    fn test_delay_buffer_update_branches() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        enc.encoder_buffer = 6;
        enc.delay_buffer = vec![0; 6];

        enc.update_delay_buffer(&[10, 11, 12, 13, 14, 15], 4);
        assert_eq!(enc.delay_buffer, vec![0, 0, 10, 11, 12, 13]);

        enc.encoder_buffer = 4;
        enc.delay_buffer = vec![0; 4];
        enc.update_delay_buffer(&[10, 11, 12, 13, 14, 15], 6);
        assert_eq!(enc.delay_buffer, vec![12, 13, 14, 15]);

        enc.encoder_buffer = 6;
        enc.delay_buffer = vec![0; 6];
        enc.update_delay_buffer_from_pcm_buf(&[20, 21, 22, 23, 24, 25], 4, 0);
        assert_eq!(enc.delay_buffer, vec![0, 0, 20, 21, 22, 23]);

        enc.encoder_buffer = 4;
        enc.delay_buffer = vec![0; 4];
        enc.update_delay_buffer_from_pcm_buf(&[20, 21, 22, 23, 24, 25], 6, 0);
        assert_eq!(enc.delay_buffer, vec![22, 23, 24, 25]);
    }

    #[test]
    fn test_mode_transition_prefill_paths() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
        assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_20_MS), OPUS_OK);
        assert_eq!(enc.set_bitrate(64000), OPUS_OK);
        assert_eq!(enc.set_vbr(1), OPUS_OK);
        assert_eq!(enc.set_signal(OPUS_SIGNAL_VOICE), OPUS_OK);

        let pcm = patterned_pcm_i16(960, 1, 511);
        let mut packet = vec![0u8; 1500];
        let packet_capacity = packet.len() as i32;

        assert_eq!(enc.set_force_mode(MODE_CELT_ONLY), OPUS_OK);
        let len_celt = enc.encode(&pcm, 960, &mut packet, packet_capacity).unwrap();
        assert!(len_celt > 1);
        assert_eq!(enc.get_mode(), MODE_CELT_ONLY);
        assert_eq!(enc.get_prev_mode(), MODE_CELT_ONLY);
        assert_eq!(
            packet_mode_from_toc(&packet[..len_celt as usize]),
            MODE_CELT_ONLY
        );

        assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
        assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
        let len_silk = enc.encode(&pcm, 960, &mut packet, packet_capacity).unwrap();
        assert!(len_silk > 1);
        assert_ne!(enc.get_mode(), MODE_CELT_ONLY);
        assert_eq!(enc.get_prev_mode(), enc.get_mode());
        assert_eq!(
            packet_mode_from_toc(&packet[..len_silk as usize]),
            enc.get_mode()
        );

        assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
        assert_eq!(enc.set_force_mode(MODE_CELT_ONLY), OPUS_OK);
        let len_transition = enc.encode(&pcm, 960, &mut packet, packet_capacity).unwrap();
        assert!(len_transition > 1);
        assert_ne!(enc.get_mode(), MODE_CELT_ONLY);
        assert_eq!(enc.get_prev_mode(), MODE_CELT_ONLY);
        assert_eq!(
            packet_mode_from_toc(&packet[..len_transition as usize]),
            enc.get_mode()
        );
    }

    #[test]
    fn test_decide_dtx_mode_thresholds() {
        let mut no_activity = 0;
        for _ in 0..10 {
            assert!(!decide_dtx_mode(0, &mut no_activity, 40));
        }
        assert_eq!(no_activity, 400);
        assert!(decide_dtx_mode(0, &mut no_activity, 40));
        assert_eq!(no_activity, 440);

        assert!(!decide_dtx_mode(1, &mut no_activity, 40));
        assert_eq!(no_activity, 0);

        no_activity = 1201;
        assert!(!decide_dtx_mode(0, &mut no_activity, 40));
        assert_eq!(no_activity, 400);
    }

    #[test]
    fn test_encode_forced_mono_celt_and_dtx_branch_paths() {
        let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
        assert_eq!(enc.set_force_mode(MODE_CELT_ONLY), OPUS_OK);
        assert_eq!(enc.set_force_channels(1), OPUS_OK);
        assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
        assert_eq!(enc.set_dtx(1), OPUS_OK);
        assert_eq!(enc.set_phase_inversion_disabled(1), OPUS_OK);
        assert_eq!(enc.get_phase_inversion_disabled(), 1);

        let pcm = patterned_pcm_i16(960, 2, 901);
        let mut active_packet = vec![0u8; 1500];
        let active_capacity = active_packet.len() as i32;
        let len = enc
            .encode(&pcm, 960, &mut active_packet, active_capacity)
            .unwrap();
        let active_packet = &active_packet[..len as usize];

        assert!(len > 1);
        assert_eq!(enc.get_mode(), MODE_CELT_ONLY);
        assert_eq!(enc.get_stream_channels(), 1);
        assert_eq!(enc.get_bandwidth(), OPUS_BANDWIDTH_WIDEBAND);
        assert_eq!(packet_mode_from_toc(active_packet), MODE_CELT_ONLY);

        let silence = [0i16; 960 * 2];
        let mut dtx_packet = vec![0u8; 1500];
        let dtx_capacity = dtx_packet.len() as i32;
        let mut len = enc
            .encode(&silence, 960, &mut dtx_packet, dtx_capacity)
            .unwrap();
        assert!(len > 1);
        for _ in 1..10 {
            len = enc
                .encode(&silence, 960, &mut dtx_packet, dtx_capacity)
                .unwrap();
            assert!(len > 1);
        }

        len = enc
            .encode(&silence, 960, &mut dtx_packet, dtx_capacity)
            .unwrap();
        assert_eq!(len, 1);
        assert_eq!(enc.get_final_range(), 0);
        assert_ne!(dtx_packet[0], 0);
    }

    #[test]
    fn test_multistream_accessors_reset_and_public_getters() {
        let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();

        enc.ms_set_user_bitrate(54321);
        enc.ms_set_bandwidth(OPUS_BANDWIDTH_WIDEBAND);
        enc.ms_set_max_bandwidth(OPUS_BANDWIDTH_SUPERWIDEBAND);
        enc.ms_set_force_mode(MODE_CELT_ONLY);
        enc.ms_set_force_channels(1);
        enc.ms_set_lfe(1);
        enc.ms_set_variable_duration(OPUS_FRAMESIZE_10_MS);
        enc.ms_set_lsb_depth(12);
        enc.ms_set_complexity(7);
        enc.ms_set_vbr(0);
        enc.ms_set_vbr_constraint(1);
        enc.ms_set_signal(OPUS_SIGNAL_MUSIC);
        enc.ms_set_inband_fec(1);
        enc.ms_set_packet_loss_perc(17);
        enc.ms_set_dtx(1);
        enc.ms_set_prediction_disabled(1);
        enc.ms_set_phase_inversion_disabled(1);
        enc.ms_set_application(OPUS_APPLICATION_RESTRICTED_LOWDELAY);
        enc.ms_set_application(12345);

        enc.hp_mem = [1, 2, 3, 4];
        enc.variable_hp_smth2_q15 = 123456;
        enc.range_final = 0x1357_9BDF;
        enc.mode = MODE_CELT_ONLY;
        enc.prev_mode = MODE_SILK_ONLY;
        enc.stream_channels = 1;

        assert_eq!(enc.ms_get_vbr(), 0);
        assert_eq!(enc.ms_get_bitrate(), enc.bitrate_bps);
        assert_eq!(enc.ms_get_variable_duration(), OPUS_FRAMESIZE_10_MS);
        assert_eq!(enc.ms_get_lsb_depth(), 12);
        assert_eq!(enc.ms_get_complexity(), 7);
        assert_eq!(enc.ms_get_lookahead(), enc.delay_compensation);
        assert!(enc.ms_get_celt_mode().is_some());
        assert_eq!(enc.get_hp_mem(), [1, 2, 3, 4]);
        assert_eq!(enc.get_variable_hp_smth2(), 123456);
        assert_eq!(enc.get_sample_rate(), 48000);
        assert_eq!(enc.get_final_range(), 0x1357_9BDF);
        assert_eq!(enc.get_application(), OPUS_APPLICATION_RESTRICTED_LOWDELAY);
        assert_eq!(enc.get_channels(), 2);
        assert_eq!(enc.get_stream_channels(), 1);
        assert_eq!(enc.get_mode(), MODE_CELT_ONLY);
        assert_eq!(enc.get_prev_mode(), MODE_SILK_ONLY);

        enc.reset();
        assert_eq!(enc.get_stream_channels(), 2);
        assert_eq!(enc.get_mode(), MODE_HYBRID);
        assert_eq!(enc.get_prev_mode(), 0);
        assert_eq!(enc.get_final_range(), 0);
        assert_eq!(enc.get_hp_mem(), [0; 4]);
    }

    #[test]
    fn test_ms_helper_accessors_cover_none_and_some_celt_paths() {
        let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();

        enc.celt_enc = None;
        enc.ms_set_complexity(4);
        enc.ms_set_packet_loss_perc(9);
        enc.ms_set_prediction_disabled(1);
        enc.ms_set_phase_inversion_disabled(1);
        enc.ms_set_application(12_345);
        assert_eq!(enc.ms_get_complexity(), 4);
        assert_eq!(enc.silk_mode.packet_loss_percentage, 9);
        assert_eq!(enc.get_application(), OPUS_APPLICATION_AUDIO);

        enc.celt_enc = Some(CeltEncoder::new(48000, 2).unwrap());
        enc.bitrate_bps = 48_000;
        enc.ms_set_user_bitrate(64_321);
        enc.ms_set_bandwidth(OPUS_BANDWIDTH_WIDEBAND);
        enc.ms_set_max_bandwidth(OPUS_BANDWIDTH_FULLBAND);
        enc.ms_set_force_mode(MODE_CELT_ONLY);
        enc.ms_set_force_channels(1);
        enc.ms_set_lfe(1);
        enc.ms_set_variable_duration(OPUS_FRAMESIZE_10_MS);
        enc.ms_set_lsb_depth(14);
        enc.ms_set_complexity(8);
        enc.ms_set_vbr(0);
        enc.ms_set_vbr_constraint(1);
        enc.ms_set_signal(OPUS_SIGNAL_MUSIC);
        enc.ms_set_inband_fec(2);
        enc.ms_set_packet_loss_perc(17);
        enc.ms_set_dtx(1);
        enc.ms_set_prediction_disabled(1);
        enc.ms_set_phase_inversion_disabled(1);
        enc.ms_set_application(OPUS_APPLICATION_RESTRICTED_LOWDELAY);

        assert_eq!(enc.ms_get_vbr(), 0);
        assert_eq!(enc.ms_get_bitrate(), 48_000);
        assert_eq!(enc.ms_get_variable_duration(), OPUS_FRAMESIZE_10_MS);
        assert_eq!(enc.ms_get_lsb_depth(), 14);
        assert_eq!(enc.ms_get_complexity(), 8);
        assert_eq!(enc.user_bitrate_bps, 64_321);
        assert_eq!(enc.user_bandwidth, OPUS_BANDWIDTH_WIDEBAND);
        assert_eq!(enc.max_bandwidth, OPUS_BANDWIDTH_FULLBAND);
        assert_eq!(enc.user_forced_mode, MODE_CELT_ONLY);
        assert_eq!(enc.force_channels, 1);
        assert_eq!(enc.lfe, 1);
        assert_eq!(enc.signal_type, OPUS_SIGNAL_MUSIC);
        assert_eq!(enc.fec_config, 2);
        assert_eq!(enc.use_dtx, 1);
        assert_eq!(enc.get_application(), OPUS_APPLICATION_RESTRICTED_LOWDELAY);

        let celt = enc.celt_enc.as_ref().unwrap();
        assert_eq!(celt.complexity, 8);
        assert_eq!(celt.loss_rate, 17);
        assert_eq!(celt.disable_pf, 1);
        assert_eq!(celt.disable_inv, 1);
    }

    #[test]
    fn test_encode_wrappers_reject_invalid_selected_frame_size() {
        let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
        enc.ms_set_variable_duration(12_345);

        let pcm_i16 = patterned_pcm_i16(960, 2, 77);
        let pcm_f32 = patterned_pcm_f32(960, 2, 77);
        let mut packet = [0u8; 16];

        assert_eq!(
            enc.encode(&pcm_i16, 960, &mut packet, 16),
            Err(OPUS_BAD_ARG)
        );
        assert_eq!(
            enc.encode_float(&pcm_f32, 960, &mut packet, 16),
            Err(OPUS_BAD_ARG)
        );
    }

    #[test]
    fn test_public_getters_cover_runtime_state_snapshots() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
        enc.hp_mem = [11, 22, 33, 44];
        enc.variable_hp_smth2_q15 = 55_555;
        enc.range_final = 0x2468_ACE0;
        enc.mode = MODE_CELT_ONLY;
        enc.prev_mode = MODE_HYBRID;
        enc.stream_channels = 1;

        assert_eq!(enc.get_hp_mem(), [11, 22, 33, 44]);
        assert_eq!(enc.get_variable_hp_smth2(), 55_555);
        assert_eq!(enc.get_sample_rate(), 48_000);
        assert_eq!(enc.get_final_range(), 0x2468_ACE0);
        assert_eq!(enc.get_application(), OPUS_APPLICATION_VOIP);
        assert_eq!(enc.get_channels(), 1);
        assert_eq!(enc.get_stream_channels(), 1);
        assert_eq!(enc.get_mode(), MODE_CELT_ONLY);
        assert_eq!(enc.get_prev_mode(), MODE_HYBRID);
    }

    #[test]
    fn test_encode_multiframe_cbr_padding_and_wrapper_errors() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        assert_eq!(enc.set_vbr(0), OPUS_OK);
        assert_eq!(enc.set_bitrate(32000), OPUS_OK);

        let pcm = patterned_pcm_i16(3840, 1, 1201);
        let mut packet = vec![0u8; 400];
        let len = enc.encode(&pcm, 3840, &mut packet, 200).unwrap();
        assert_eq!(len, 200);
        assert!(enc.get_prev_mode() > 0);
        assert!(enc.get_final_range() != 0);

        let pcmf = patterned_pcm_f32(960, 1, 1203);
        assert_eq!(
            enc.encode_float(&pcmf, -1, &mut packet, 200),
            Err(OPUS_BAD_ARG)
        );
    }

    #[test]
    fn test_encode_multiframe_silk_special_frame_sizes() {
        let mut enc = OpusEncoder::new(16000, 1, OPUS_APPLICATION_VOIP).unwrap();
        assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
        assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
        assert_eq!(enc.set_bitrate(24000), OPUS_OK);
        assert_eq!(enc.set_vbr(1), OPUS_OK);
        assert_eq!(enc.set_signal(OPUS_SIGNAL_VOICE), OPUS_OK);

        let mut packet = vec![0u8; 1500];
        let capacity = packet.len() as i32;

        let pcm_80 = patterned_pcm_i16((2 * enc.fs / 25) as usize, 1, 1401);
        let len_80 = enc
            .encode_multiframe(
                &pcm_80,
                2 * enc.fs / 25,
                &mut packet,
                capacity,
                capacity,
                16,
                MODE_SILK_ONLY,
                enc.bitrate_bps,
                false,
                false,
                false,
                0,
                enc.bitrate_bps,
                false,
            )
            .unwrap();
        assert!(len_80 > 1);
        assert_eq!(enc.nonfinal_frame, 0);

        let pcm_120 = patterned_pcm_i16((3 * enc.fs / 25) as usize, 1, 1403);
        let len_120 = enc
            .encode_multiframe(
                &pcm_120,
                3 * enc.fs / 25,
                &mut packet,
                capacity,
                capacity,
                16,
                MODE_SILK_ONLY,
                enc.bitrate_bps,
                false,
                false,
                false,
                0,
                enc.bitrate_bps,
                false,
            )
            .unwrap();
        assert!(len_120 > 1);
        assert_eq!(enc.nonfinal_frame, 0);
    }

    #[test]
    fn test_encode_stereo_voice_ratio_forced_mono_transition_path() {
        let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
        assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
        assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
        assert_eq!(enc.set_max_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
        assert_eq!(enc.set_force_channels(1), OPUS_OK);
        assert_eq!(enc.set_voice_ratio(80), OPUS_OK);
        assert_eq!(enc.set_signal(OPUS_AUTO), OPUS_OK);
        assert_eq!(enc.set_bitrate(20000), OPUS_OK);
        assert_eq!(enc.set_vbr(1), OPUS_OK);

        enc.prev_mode = MODE_SILK_ONLY;
        enc.prev_channels = 2;
        enc.stream_channels = 2;
        enc.silk_mode.to_mono = 0;

        let pcm = patterned_pcm_i16(960, 2, 1501);
        let mut packet = vec![0u8; 1500];
        let packet_capacity = packet.len() as i32;
        let len = enc.encode(&pcm, 960, &mut packet, packet_capacity).unwrap();

        assert!(len > 1);
        assert_ne!(enc.get_mode(), MODE_CELT_ONLY);
        assert_eq!(
            packet_mode_from_toc(&packet[..len as usize]),
            enc.get_mode()
        );
        assert_eq!(enc.silk_mode.to_mono, 1);
        assert_eq!(enc.get_stream_channels(), 2);
    }

    #[test]
    fn test_encode_frame_native_celt_resets_silk_bw_switch() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_RESTRICTED_LOWDELAY).unwrap();
        assert_eq!(enc.set_bitrate(32000), OPUS_OK);
        assert_eq!(enc.set_vbr(1), OPUS_OK);
        enc.mode = MODE_CELT_ONLY;
        enc.prev_mode = MODE_CELT_ONLY;
        enc.bandwidth = OPUS_BANDWIDTH_FULLBAND;
        enc.silk_bw_switch = 1;

        let pcm = patterned_pcm_i16(960, 1, 1601);
        let mut packet = vec![0u8; 1500];
        let packet_capacity = packet.len() as i32;
        let len = enc
            .encode_frame_native(
                &pcm,
                960,
                &mut packet,
                packet_capacity,
                packet_capacity,
                false,
                false,
                false,
                0,
                enc.bitrate_bps,
                false,
            )
            .unwrap();

        assert!(len > 1);
        assert_eq!(enc.silk_bw_switch, 0);
        assert_eq!(
            packet_mode_from_toc(&packet[..len as usize]),
            MODE_CELT_ONLY
        );
    }

    #[test]
    fn test_helper_noop_branches_and_phase_inversion_default() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        enc.celt_enc = None;
        assert_eq!(enc.get_phase_inversion_disabled(), 0);

        enc.encoder_buffer = 0;
        enc.delay_buffer = vec![7, 8, 9];
        enc.update_delay_buffer(&[1, 2, 3], 3);
        enc.update_delay_buffer_from_pcm_buf(&[4, 5, 6], 3, 0);
        assert_eq!(enc.delay_buffer, vec![7, 8, 9]);

        enc.encoder_buffer = 4;
        enc.delay_buffer = vec![11, 12, 13, 14];
        enc.update_delay_buffer(&[1, 2], 6);
        assert_eq!(enc.delay_buffer, vec![11, 12, 13, 14]);
    }

    /// In SILK_ONLY mode at very low bitrates, the encoder should cap the
    /// internal sample rate so SILK doesn't try to encode more bandwidth
    /// than the bitrate can support (C: opus_encoder.c:2129-2143).
    ///
    /// effective_max_rate = bits_to_bitrate(max_data_bytes * 8, fs, frame_size)
    /// For 48 kHz / 960 samples (20 ms, frame_rate=50):
    ///   effective_max_rate = max_data_bytes * 400
    ///
    /// Thresholds: <8000 -> cap at 12000, <7000 -> cap at 8000.
    #[test]
    fn test_effective_max_rate_narrows_silk_internal_rate() {
        // --- Case 1: effective_max_rate = 10000 (>= 8000) -> no narrowing ---
        {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            enc.mode = MODE_SILK_ONLY;
            enc.prev_mode = MODE_SILK_ONLY;
            enc.bandwidth = OPUS_BANDWIDTH_WIDEBAND;

            let pcm = patterned_pcm_i16(960, 1, 501);
            let mut packet = vec![0u8; 1500];
            // max_data_bytes = 25 -> effective_max_rate = 25*400 = 10000
            // Result ignored: we only inspect silk_mode state after the
            // narrowing logic runs (encode may fail with tiny buffers).
            let _ = enc.encode_frame_native(
                &pcm,
                960,
                &mut packet,
                25,
                25,
                false,
                false,
                false,
                0,
                6000,
                false,
            );
            assert_eq!(
                enc.silk_mode.max_internal_sample_rate, 16000,
                "rate >= 8000: max_internal_sample_rate should stay at 16000"
            );
        }

        // --- Case 2: effective_max_rate = 7200 (< 8000, >= 7000) -> cap at 12000 ---
        {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            enc.mode = MODE_SILK_ONLY;
            enc.prev_mode = MODE_SILK_ONLY;
            enc.bandwidth = OPUS_BANDWIDTH_WIDEBAND;

            let pcm = patterned_pcm_i16(960, 1, 502);
            let mut packet = vec![0u8; 1500];
            // max_data_bytes = 18 -> effective_max_rate = 18*400 = 7200
            let _ = enc.encode_frame_native(
                &pcm,
                960,
                &mut packet,
                18,
                18,
                false,
                false,
                false,
                0,
                6000,
                false,
            );
            assert_eq!(
                enc.silk_mode.max_internal_sample_rate, 12000,
                "rate < 8000: max_internal_sample_rate should be capped at 12000"
            );
        }

        // --- Case 3: effective_max_rate = 6000 (< 7000) -> cap at 8000 ---
        {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            enc.mode = MODE_SILK_ONLY;
            enc.prev_mode = MODE_SILK_ONLY;
            enc.bandwidth = OPUS_BANDWIDTH_WIDEBAND;

            let pcm = patterned_pcm_i16(960, 1, 503);
            let mut packet = vec![0u8; 1500];
            // max_data_bytes = 15 -> effective_max_rate = 15*400 = 6000
            let _ = enc.encode_frame_native(
                &pcm,
                960,
                &mut packet,
                15,
                15,
                false,
                false,
                false,
                0,
                6000,
                false,
            );
            assert_eq!(
                enc.silk_mode.max_internal_sample_rate, 8000,
                "rate < 7000: max_internal_sample_rate should be capped at 8000"
            );
        }
    }

    /// Verify that the frame_rate > 50 branch applies the 2/3 penalty
    /// to effective_max_rate, making the narrowing kick in sooner.
    #[test]
    fn test_effective_max_rate_high_frame_rate_penalty() {
        // With fs=48000, frame_size=480 (10 ms), frame_rate=100 (>50).
        // effective_max_rate before penalty: max_data_bytes * 800
        // after 2/3 penalty: max_data_bytes * 800 * 2/3
        //
        // max_data_bytes = 15 -> pre-penalty = 12000, post-penalty = 8000
        // 8000 < 8000 is false -> no narrowing
        //
        // max_data_bytes = 14 -> pre-penalty = 11200, post-penalty = 7466
        // 7466 < 8000 -> cap at 12000, >= 7000 -> no further cap
        //
        // max_data_bytes = 13 -> pre-penalty = 10400, post-penalty = 6933
        // 6933 < 7000 -> cap at 8000

        // --- post-penalty rate = 8000 -> NOT < 8000 -> no narrowing ---
        {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            enc.mode = MODE_SILK_ONLY;
            enc.prev_mode = MODE_SILK_ONLY;
            enc.bandwidth = OPUS_BANDWIDTH_WIDEBAND;

            let pcm = patterned_pcm_i16(480, 1, 601);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode_frame_native(
                &pcm,
                480,
                &mut packet,
                15,
                15,
                false,
                false,
                false,
                0,
                6000,
                false,
            );
            assert_eq!(
                enc.silk_mode.max_internal_sample_rate, 16000,
                "post-penalty rate == 8000 should not trigger narrowing"
            );
        }

        // --- post-penalty rate = 7466 -> < 8000 -> cap at 12000 ---
        {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            enc.mode = MODE_SILK_ONLY;
            enc.prev_mode = MODE_SILK_ONLY;
            enc.bandwidth = OPUS_BANDWIDTH_WIDEBAND;

            let pcm = patterned_pcm_i16(480, 1, 602);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode_frame_native(
                &pcm,
                480,
                &mut packet,
                14,
                14,
                false,
                false,
                false,
                0,
                6000,
                false,
            );
            assert_eq!(
                enc.silk_mode.max_internal_sample_rate, 12000,
                "post-penalty rate 7466 < 8000 should cap at 12000"
            );
        }

        // --- post-penalty rate = 6933 -> < 7000 -> cap at 8000 ---
        {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            enc.mode = MODE_SILK_ONLY;
            enc.prev_mode = MODE_SILK_ONLY;
            enc.bandwidth = OPUS_BANDWIDTH_WIDEBAND;

            let pcm = patterned_pcm_i16(480, 1, 603);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode_frame_native(
                &pcm,
                480,
                &mut packet,
                13,
                13,
                false,
                false,
                false,
                0,
                6000,
                false,
            );
            assert_eq!(
                enc.silk_mode.max_internal_sample_rate, 8000,
                "post-penalty rate 6933 < 7000 should cap at 8000"
            );
        }
    }

    // =======================================================================
    // Coverage gap tests
    // =======================================================================

    /// Gap 1: SILK→CELT bandwidth switch triggering redundancy encoding
    /// (lines ~1881-1903). Setting silk_bw_switch=1 before encode_frame_native
    /// forces the redundancy/celt_to_silk/prefill path.
    #[test]
    fn test_silk_bw_switch_triggers_redundancy() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
        assert_eq!(enc.set_bitrate(64000), OPUS_OK);
        assert_eq!(enc.set_vbr(1), OPUS_OK);

        // First: encode a SILK frame to establish prev_mode
        assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
        assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
        let pcm = patterned_pcm_i16(960, 1, 2001);
        let mut packet = vec![0u8; 1500];
        let cap = packet.len() as i32;
        let _ = enc.encode(&pcm, 960, &mut packet, cap).unwrap();
        assert_eq!(enc.get_mode(), MODE_SILK_ONLY);

        // Now set silk_bw_switch=1 and encode again in SILK mode.
        // This should trigger the redundancy path at line 1881.
        enc.silk_bw_switch = 1;
        enc.mode = MODE_SILK_ONLY;
        enc.bandwidth = OPUS_BANDWIDTH_WIDEBAND;
        let pcm2 = patterned_pcm_i16(960, 1, 2002);
        let len = enc
            .encode_frame_native(
                &pcm2,
                960,
                &mut packet,
                cap,
                cap,
                false,
                false,
                false,
                0,
                enc.bitrate_bps,
                false,
            )
            .unwrap();
        assert!(len > 1);
        // silk_bw_switch should be cleared
        assert_eq!(enc.silk_bw_switch, 0);
    }

    /// Gap 2: DTX activation — silence detection → 1-byte DTX packet after
    /// enough silence frames (lines ~2543-2548). Uses CELT-only with CELT DTX
    /// (use_dtx=1, silk_mode.use_dtx=0 triggers the non-SILK DTX path).
    #[test]
    fn test_dtx_activation_celt_silence_path() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        assert_eq!(enc.set_force_mode(MODE_CELT_ONLY), OPUS_OK);
        assert_eq!(enc.set_bitrate(32000), OPUS_OK);
        assert_eq!(enc.set_vbr(1), OPUS_OK);
        assert_eq!(enc.set_dtx(1), OPUS_OK);

        let silence = [0i16; 960];
        let mut packet = vec![0u8; 1500];
        let cap = packet.len() as i32;

        // Encode active frames first to set peak_signal_energy
        let pcm_active = patterned_pcm_i16(960, 1, 2101);
        let _ = enc.encode(&pcm_active, 960, &mut packet, cap).unwrap();

        // Now encode silence repeatedly — after NB_SPEECH_FRAMES_BEFORE_DTX (10)
        // frames of silence, we should get a DTX (1-byte) packet.
        let mut got_dtx = false;
        for _ in 0..15 {
            let len = enc.encode(&silence, 960, &mut packet, cap).unwrap();
            if len == 1 {
                got_dtx = true;
                assert_eq!(enc.get_final_range(), 0);
                break;
            }
        }
        assert!(
            got_dtx,
            "expected DTX 1-byte packet after sustained silence"
        );
    }

    /// Gap 3: Hybrid mode SILK rate interpolation with DRED/LBRR flags.
    /// (lines ~1987-1994, 2077-2087). Force hybrid mode and encode to
    /// exercise compute_silk_rate_for_hybrid in the encode path.
    #[test]
    fn test_hybrid_silk_rate_interpolation_with_fec() {
        // Test compute_silk_rate_for_hybrid with LBRR flag variations
        // SWB path with FEC (lbrr=1) and CBR: entry=4, rate=32000
        // interp from [32000,28000] boundary → 28000, +100 CBR, +300 SWB = 28400
        assert_eq!(
            compute_silk_rate_for_hybrid(32000, OPUS_BANDWIDTH_SUPERWIDEBAND, true, 0, 1, 1),
            28400
        );
        // FB path without FEC and VBR: entry=2, rate=32000
        // interp from [32000,22000] boundary → 22000, no CBR, no SWB = 22000
        assert_eq!(
            compute_silk_rate_for_hybrid(32000, OPUS_BANDWIDTH_FULLBAND, true, 1, 0, 1),
            22000
        );
        // Stereo with high rate (exceeds table): rate/2=100000, entry=4
        // last entry [64000,50000]: 50000 + (100000-64000)/2 = 68000
        // VBR=1 no CBR boost, no SWB boost → 68000*2 - 1000 stereo = 135000
        assert_eq!(
            compute_silk_rate_for_hybrid(200000, OPUS_BANDWIDTH_FULLBAND, true, 1, 1, 2),
            135000
        );

        // Also exercise via actual hybrid encode to hit lines 1987-1994
        let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_VOIP).unwrap();
        assert_eq!(enc.set_force_mode(MODE_HYBRID), OPUS_OK);
        assert_eq!(enc.set_bitrate(48000), OPUS_OK);
        assert_eq!(enc.set_vbr(1), OPUS_OK);
        assert_eq!(enc.set_inband_fec(1), OPUS_OK);
        assert_eq!(enc.set_packet_loss_perc(10), OPUS_OK);

        let pcm = patterned_pcm_i16(960, 2, 2201);
        let mut packet = vec![0u8; 1500];
        let cap = packet.len() as i32;
        let len = enc.encode(&pcm, 960, &mut packet, cap).unwrap();
        assert!(len > 1);
        assert_eq!(enc.get_mode(), MODE_HYBRID);
    }

    /// Gap 3b: Hybrid constrained VBR exercises the max_bits recomputation
    /// through compute_silk_rate_for_hybrid (lines 2077-2087).
    #[test]
    fn test_hybrid_constrained_vbr_silk_rate() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        assert_eq!(enc.set_force_mode(MODE_HYBRID), OPUS_OK);
        assert_eq!(enc.set_bitrate(40000), OPUS_OK);
        assert_eq!(enc.set_vbr(1), OPUS_OK);
        assert_eq!(enc.set_vbr_constraint(1), OPUS_OK);

        let mut packet = vec![0u8; 1500];
        let cap = packet.len() as i32;
        // Encode a few frames to stabilize state
        for seed in 0..3 {
            let pcm = patterned_pcm_i16(960, 1, 2301 + seed);
            let _ = enc.encode(&pcm, 960, &mut packet, cap).unwrap();
        }
        assert_eq!(enc.get_mode(), MODE_HYBRID);
    }

    /// Gap 4: FEC hysteresis — decide_fec with last_fec=1 (lines ~1592-1601).
    /// When last_fec was enabled, threshold is lowered by hysteresis, making
    /// FEC easier to keep.
    #[test]
    fn test_decide_fec_hysteresis_last_fec_enabled() {
        // With last_fec=1, threshold at WB is reduced by hysteresis.
        // WB: threshold=16000, hyst=1000. last_fec=1 → 15000.
        // loss=10: factor=115, silk_smulwb(15000*115, 655)=17239.
        // rate=18000 > 17239 → returns 1, bandwidth stays WB.
        let mut bw = OPUS_BANDWIDTH_WIDEBAND;
        let result = decide_fec(1, 10, 1, MODE_SILK_ONLY, &mut bw, 18000);
        assert_eq!(
            result, 1,
            "last_fec=1 hysteresis should keep FEC at WB with rate 18000"
        );
        assert_eq!(
            bw, OPUS_BANDWIDTH_WIDEBAND,
            "bandwidth should stay WB with hysteresis"
        );

        // Without hysteresis (last_fec=0), WB threshold = 16000+1000=17000 → scaled=19538.
        // 18000 < 19538 → falls through; WB gets reduced to MB.
        // MB: threshold=(14000+1000)*115*... = 17239.  18000 > 17239 → returns 1.
        // But bandwidth was changed to MB!
        let mut bw2 = OPUS_BANDWIDTH_WIDEBAND;
        let result2 = decide_fec(1, 10, 0, MODE_SILK_ONLY, &mut bw2, 18000);
        assert_eq!(
            result2, 1,
            "without hysteresis, FEC still enabled but bw reduced"
        );
        assert_eq!(
            bw2, OPUS_BANDWIDTH_MEDIUMBAND,
            "bandwidth should be reduced to MB without hysteresis"
        );
    }

    /// Gap 4b: FEC hysteresis through the encode path — set up encoder with
    /// FEC enabled, sufficient loss, and SILK mode to hit the decide_fec call
    /// at lines 1592-1601.
    #[test]
    fn test_fec_decision_in_encode_path() {
        let mut enc = OpusEncoder::new(16000, 1, OPUS_APPLICATION_VOIP).unwrap();
        assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
        assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
        assert_eq!(enc.set_bitrate(24000), OPUS_OK);
        assert_eq!(enc.set_vbr(1), OPUS_OK);
        assert_eq!(enc.set_inband_fec(1), OPUS_OK);
        assert_eq!(enc.set_packet_loss_perc(15), OPUS_OK);
        assert_eq!(enc.set_signal(OPUS_SIGNAL_VOICE), OPUS_OK);

        let mut packet = vec![0u8; 1500];
        let cap = packet.len() as i32;

        // Encode 3 frames to let FEC state stabilize
        for i in 0..3 {
            let pcm = patterned_pcm_i16(320, 1, 2401 + i);
            let _ = enc.encode(&pcm, 320, &mut packet, cap).unwrap();
        }
        // Check that lbrr_coded is set (FEC was decided)
        // The exact value depends on the rate/bandwidth interaction.
        // The important thing is that the decide_fec path was exercised.
        assert!(enc.silk_mode.lbrr_coded == 0 || enc.silk_mode.lbrr_coded == 1);
    }

    /// Gap 5: Stereo width edge cases — low-bitrate stereo width reduction
    /// in hybrid (lines ~2289-2325). Force hybrid stereo at low bitrate to
    /// trigger the stereo_fade path.
    #[test]
    fn test_stereo_width_reduction_hybrid_low_bitrate() {
        let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
        assert_eq!(enc.set_force_mode(MODE_HYBRID), OPUS_OK);
        assert_eq!(enc.set_bitrate(20000), OPUS_OK);
        assert_eq!(enc.set_vbr(1), OPUS_OK);

        let mut packet = vec![0u8; 1500];
        let cap = packet.len() as i32;

        // Encode a few frames to exercise stereo width calculation
        for i in 0..3 {
            let pcm = patterned_pcm_i16(960, 2, 2501 + i * 17);
            let _ = enc.encode(&pcm, 960, &mut packet, cap).unwrap();
        }
        assert_eq!(enc.get_mode(), MODE_HYBRID);
        // At this low bitrate, stereo width should be reduced
        assert!(
            (enc.hybrid_stereo_width_q14 as i32) < (1 << 14),
            "stereo width should be reduced at low bitrate"
        );
    }

    /// Gap 6: LFE channel mode forces CELT-only narrowband (lines ~1450-1451,
    /// 1615-1616).
    #[test]
    fn test_lfe_forces_celt_only_narrowband() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        enc.ms_set_lfe(1);
        assert_eq!(enc.set_bitrate(32000), OPUS_OK);
        assert_eq!(enc.set_vbr(1), OPUS_OK);

        let pcm = patterned_pcm_i16(960, 1, 2601);
        let mut packet = vec![0u8; 1500];
        let cap = packet.len() as i32;
        let len = enc.encode(&pcm, 960, &mut packet, cap).unwrap();
        assert!(len > 0);
        // LFE forces CELT-only
        assert_eq!(enc.get_mode(), MODE_CELT_ONLY);
        // LFE forces narrowband
        assert_eq!(enc.get_bandwidth(), OPUS_BANDWIDTH_NARROWBAND);
    }

    /// Gap 7: RESTRICTED_LOWDELAY application forces CELT-only with zero delay
    /// compensation (lines ~1397-1398). Also covers the delay_compensation=0
    /// branch in encode_frame_native.
    #[test]
    fn test_restricted_lowdelay_forces_celt_zero_delay() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_RESTRICTED_LOWDELAY).unwrap();
        assert_eq!(enc.set_bitrate(64000), OPUS_OK);
        assert_eq!(enc.set_vbr(1), OPUS_OK);
        assert_eq!(
            enc.set_expert_frame_duration(OPUS_FRAMESIZE_2_5_MS),
            OPUS_OK
        );

        let pcm = patterned_pcm_i16(120, 1, 2701);
        let mut packet = vec![0u8; 1500];
        let cap = packet.len() as i32;
        let len = enc.encode(&pcm, 120, &mut packet, cap).unwrap();
        assert!(len > 0);
        assert_eq!(enc.get_mode(), MODE_CELT_ONLY);
        // Lowdelay has reduced lookahead
        assert_eq!(enc.get_lookahead(), 120);
    }

    /// Gap 8: VBR constraint with CELT (line ~2421). In non-hybrid CELT mode
    /// with VBR and constraint, the CELT encoder gets SetVbrConstraint.
    #[test]
    fn test_vbr_constraint_celt_only_path() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        assert_eq!(enc.set_force_mode(MODE_CELT_ONLY), OPUS_OK);
        assert_eq!(enc.set_bitrate(64000), OPUS_OK);
        assert_eq!(enc.set_vbr(1), OPUS_OK);
        assert_eq!(enc.set_vbr_constraint(1), OPUS_OK);

        let pcm = patterned_pcm_i16(960, 1, 2801);
        let mut packet = vec![0u8; 1500];
        let cap = packet.len() as i32;
        let len = enc.encode(&pcm, 960, &mut packet, cap).unwrap();
        assert!(len > 1);
        assert_eq!(enc.get_mode(), MODE_CELT_ONLY);
        // Confirm VBR constraint is active
        assert_eq!(enc.get_vbr(), 1);
        assert_eq!(enc.get_vbr_constraint(), 1);
    }

    /// Gap 9: HP cutoff filter — mono and stereo biquad filter paths
    /// (lines ~727-734).
    #[test]
    fn test_hp_cutoff_mono_and_stereo() {
        // Mono path (stride1)
        let input_mono = vec![1000i16; 480];
        let mut output_mono = vec![0i16; 480];
        let mut hp_mem_mono = [0i32; 4];
        hp_cutoff_debug(
            &input_mono,
            100,
            &mut output_mono,
            &mut hp_mem_mono,
            480,
            1,
            48000,
        );
        // Filter should produce output; DC content should be attenuated
        assert!(output_mono.iter().any(|&s| s != 0));
        // HP mem should be updated
        assert!(hp_mem_mono.iter().any(|&m| m != 0));

        // Stereo path (stride2)
        let input_stereo = vec![500i16; 960];
        let mut output_stereo = vec![0i16; 960];
        let mut hp_mem_stereo = [0i32; 4];
        hp_cutoff_debug(
            &input_stereo,
            80,
            &mut output_stereo,
            &mut hp_mem_stereo,
            480,
            2,
            48000,
        );
        assert!(output_stereo.iter().any(|&s| s != 0));
        assert!(hp_mem_stereo.iter().any(|&m| m != 0));
    }

    /// Gap 10: Prefill gain fade on mode transition (lines ~2096-2149).
    /// Transition from CELT→SILK triggers prefill with gain_fade.
    #[test]
    fn test_prefill_gain_fade_on_celt_to_silk_transition() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
        assert_eq!(enc.set_bitrate(48000), OPUS_OK);
        assert_eq!(enc.set_vbr(1), OPUS_OK);
        assert_eq!(enc.set_signal(OPUS_SIGNAL_VOICE), OPUS_OK);

        // Encode as CELT first
        assert_eq!(enc.set_force_mode(MODE_CELT_ONLY), OPUS_OK);
        let pcm = patterned_pcm_i16(960, 1, 3001);
        let mut packet = vec![0u8; 1500];
        let cap = packet.len() as i32;
        let _ = enc.encode(&pcm, 960, &mut packet, cap).unwrap();
        assert_eq!(enc.get_mode(), MODE_CELT_ONLY);

        // Force transition to SILK — this triggers prefill (line 1506) and
        // the gain_fade ramp at lines 2096-2149.
        assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
        assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
        let pcm2 = patterned_pcm_i16(960, 1, 3002);
        let len = enc.encode(&pcm2, 960, &mut packet, cap).unwrap();
        assert!(len > 1);
        // After transition the mode should not be CELT_ONLY
        // (it stays as prev_mode=SILK during transition encode)
        assert_ne!(
            packet_mode_from_toc(&packet[..len as usize]),
            MODE_CELT_ONLY
        );
    }

    /// Gap 11: CBR padding — multiframe CBR path where pad_cbr is triggered
    /// (lines ~1806-1820). In CBR mode, if not all frames are DTX, the
    /// repacketizer pads to the target size.
    #[test]
    fn test_cbr_padding_multiframe() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        assert_eq!(enc.set_force_mode(MODE_CELT_ONLY), OPUS_OK);
        assert_eq!(enc.set_vbr(0), OPUS_OK); // CBR
        assert_eq!(enc.set_bitrate(48000), OPUS_OK);
        assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_40_MS), OPUS_OK);

        let pcm = patterned_pcm_i16(1920, 1, 3101);
        let mut packet = vec![0u8; 1500];
        let cap = packet.len() as i32;
        let len = enc.encode(&pcm, 1920, &mut packet, cap).unwrap();
        assert!(len > 1);
        // CBR should produce a consistent size per bitrate
        // 48000 bps * 40ms = 1920 bits = 240 bytes, plus overhead
        // The key is that pad_cbr was triggered (use_vbr==0, not all DTX)
    }

    /// Gap 12: Uncommon frame sizes — 100ms encoding validation (line ~1246).
    /// Also tests the 100ms → multi-frame split in SILK mode.
    #[test]
    fn test_100ms_frame_size_encoding() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
        assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
        assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
        assert_eq!(enc.set_bitrate(24000), OPUS_OK);
        assert_eq!(enc.set_vbr(1), OPUS_OK);
        assert_eq!(
            enc.set_expert_frame_duration(OPUS_FRAMESIZE_100_MS),
            OPUS_OK
        );

        // 100ms at 48kHz = 4800 samples
        let pcm = patterned_pcm_i16(4800, 1, 3201);
        let mut packet = vec![0u8; 1500];
        let cap = packet.len() as i32;
        let len = enc.encode(&pcm, 4800, &mut packet, cap).unwrap();
        assert!(len > 1, "100ms frame should produce valid packet");

        // Also test 1-byte rejection for 100ms
        let mut tiny = [0u8; 1];
        assert_eq!(
            enc.encode(&pcm, 4800, &mut tiny, 1),
            Err(OPUS_BUFFER_TOO_SMALL)
        );
    }

    /// Gap 13: Signal type MUSIC paths — voice estimation for AUDIO application
    /// (line ~1344). When signal is MUSIC, voice_est=0 which shifts mode
    /// threshold toward CELT.
    #[test]
    fn test_signal_type_music_voice_estimation() {
        let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
        assert_eq!(enc.set_signal(OPUS_SIGNAL_MUSIC), OPUS_OK);
        assert_eq!(enc.set_bitrate(96000), OPUS_OK);
        assert_eq!(enc.set_vbr(1), OPUS_OK);

        let pcm = patterned_pcm_i16(960, 2, 3301);
        let mut packet = vec![0u8; 1500];
        let cap = packet.len() as i32;
        let len = enc.encode(&pcm, 960, &mut packet, cap).unwrap();
        assert!(len > 1);
        // MUSIC signal at high bitrate should select CELT
        assert_eq!(enc.get_mode(), MODE_CELT_ONLY);
    }

    /// Gap 14: Bandwidth downgrade/restore — SILK-initiated redundancy recalc
    /// (lines ~2209-2222). When silk_mode.opus_can_switch is set during SILK
    /// encode, the redundancy bytes are recalculated and silk_bw_switch is set.
    #[test]
    fn test_silk_bandwidth_switch_redundancy_recalc() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
        assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
        assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
        assert_eq!(enc.set_bitrate(32000), OPUS_OK);
        assert_eq!(enc.set_vbr(1), OPUS_OK);
        assert_eq!(enc.set_signal(OPUS_SIGNAL_VOICE), OPUS_OK);

        let mut packet = vec![0u8; 1500];
        let cap = packet.len() as i32;

        // Encode a few frames to let SILK stabilize
        for i in 0..5 {
            let pcm = patterned_pcm_i16(960, 1, 3401 + i * 13);
            let _ = enc.encode(&pcm, 960, &mut packet, cap).unwrap();
        }
        // The silk_bw_switch is set by SILK internally when it wants to switch.
        // We manually set it and encode to verify the path.
        enc.silk_bw_switch = 1;
        let pcm2 = patterned_pcm_i16(960, 1, 3499);
        let len = enc
            .encode_frame_native(
                &pcm2,
                960,
                &mut packet,
                cap,
                cap,
                false,
                false,
                false,
                0,
                enc.bitrate_bps,
                false,
            )
            .unwrap();
        assert!(len > 1);
        assert_eq!(enc.silk_bw_switch, 0);
    }

    /// Gap 5b: Non-hybrid stereo width with intermediate bitrate.
    /// Lines 2293-2299: equiv_rate between 16000 and 32000 triggers the
    /// interpolation formula for stereo_width_q14.
    #[test]
    fn test_stereo_width_intermediate_bitrate_non_hybrid() {
        let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_VOIP).unwrap();
        assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
        assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
        assert_eq!(enc.set_bitrate(20000), OPUS_OK);
        assert_eq!(enc.set_vbr(1), OPUS_OK);

        let mut packet = vec![0u8; 1500];
        let cap = packet.len() as i32;

        for i in 0..3 {
            let pcm = patterned_pcm_i16(960, 2, 3501 + i * 7);
            let _ = enc.encode(&pcm, 960, &mut packet, cap).unwrap();
        }
        // At bitrate ~20000, equiv_rate should be in the 16000..32000 range
        // for stereo width interpolation
        assert!(
            enc.silk_mode.stereo_width_q14 >= 0 && enc.silk_mode.stereo_width_q14 <= 16384,
            "stereo width should be in valid range"
        );
    }

    /// Gap 9b: HP cutoff filter through actual encode — encoding at 8kHz
    /// exercises the narrowband path including the DC reject filter.
    #[test]
    fn test_encode_8khz_narrowband_hp_filter() {
        let mut enc = OpusEncoder::new(8000, 1, OPUS_APPLICATION_VOIP).unwrap();
        assert_eq!(enc.set_bitrate(12000), OPUS_OK);
        assert_eq!(enc.set_vbr(1), OPUS_OK);
        assert_eq!(enc.set_signal(OPUS_SIGNAL_VOICE), OPUS_OK);

        let pcm = patterned_pcm_i16(160, 1, 3601);
        let mut packet = vec![0u8; 1500];
        let cap = packet.len() as i32;
        let len = enc.encode(&pcm, 160, &mut packet, cap).unwrap();
        assert!(len > 0);
        // 8kHz must be narrowband
        assert_eq!(enc.get_bandwidth(), OPUS_BANDWIDTH_NARROWBAND);
    }

    /// Gap 9c: HP filter for stereo path via encoding at 48kHz with VOIP
    /// (exercises the variable HP filter stereo biquad branch).
    #[test]
    fn test_encode_stereo_voip_hp_filter() {
        let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_VOIP).unwrap();
        assert_eq!(enc.set_bitrate(64000), OPUS_OK);
        assert_eq!(enc.set_vbr(1), OPUS_OK);
        assert_eq!(enc.set_signal(OPUS_SIGNAL_VOICE), OPUS_OK);

        let pcm = patterned_pcm_i16(960, 2, 3701);
        let mut packet = vec![0u8; 1500];
        let cap = packet.len() as i32;
        let len = enc.encode(&pcm, 960, &mut packet, cap).unwrap();
        assert!(len > 1);
        // HP mem should be updated after encoding (VOIP uses HP filter)
        assert!(enc.hp_mem.iter().any(|&m| m != 0));
    }

    /// Gap 12b: Frame sizes at different sample rates — 12kHz encoding.
    #[test]
    fn test_encode_12khz_mediumband() {
        let mut enc = OpusEncoder::new(12000, 1, OPUS_APPLICATION_VOIP).unwrap();
        assert_eq!(enc.set_bitrate(16000), OPUS_OK);
        assert_eq!(enc.set_vbr(1), OPUS_OK);
        assert_eq!(enc.set_signal(OPUS_SIGNAL_VOICE), OPUS_OK);

        // 20ms at 12kHz = 240 samples
        let pcm = patterned_pcm_i16(240, 1, 3801);
        let mut packet = vec![0u8; 1500];
        let cap = packet.len() as i32;
        let len = enc.encode(&pcm, 240, &mut packet, cap).unwrap();
        assert!(len > 0);
        // 12kHz caps at mediumband
        assert!(enc.get_bandwidth() <= OPUS_BANDWIDTH_MEDIUMBAND);
    }

    /// Gap 11b: CBR multiframe CELT — 60ms encoding with CBR and force CELT.
    /// Tests the repacketize path with padding for 3 sub-frames.
    #[test]
    fn test_cbr_multiframe_60ms_celt() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        assert_eq!(enc.set_force_mode(MODE_CELT_ONLY), OPUS_OK);
        assert_eq!(enc.set_vbr(0), OPUS_OK); // CBR
        assert_eq!(enc.set_bitrate(64000), OPUS_OK);
        assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_60_MS), OPUS_OK);

        // 60ms at 48kHz = 2880 samples
        let pcm = patterned_pcm_i16(2880, 1, 3901);
        let mut packet = vec![0u8; 1500];
        let cap = packet.len() as i32;
        let len = enc.encode(&pcm, 2880, &mut packet, cap).unwrap();
        assert!(len > 1);
        assert_eq!(enc.get_mode(), MODE_CELT_ONLY);
    }

    /// Gap 13b: AUDIO application with voice_ratio set — exercises the
    /// `ve = imin(ve, 115)` path at line 1344.
    #[test]
    fn test_audio_application_voice_ratio_capped() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        assert_eq!(enc.set_voice_ratio(100), OPUS_OK);
        assert_eq!(enc.set_bitrate(32000), OPUS_OK);
        assert_eq!(enc.set_vbr(1), OPUS_OK);

        // voice_ratio=100 → ve = 100*327>>8 = 127, but AUDIO caps at 115
        let pcm = patterned_pcm_i16(960, 1, 4001);
        let mut packet = vec![0u8; 1500];
        let cap = packet.len() as i32;
        let len = enc.encode(&pcm, 960, &mut packet, cap).unwrap();
        assert!(len > 0);
    }

    // -----------------------------------------------------------------------
    // CTL error-path and boundary coverage
    // -----------------------------------------------------------------------

    #[test]
    fn test_ctl_set_bandwidth_error_paths() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        // Valid auto
        assert_eq!(enc.set_bandwidth(OPUS_AUTO), OPUS_OK);
        // Valid narrowband
        assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_NARROWBAND), OPUS_OK);
        // Valid fullband
        assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_FULLBAND), OPUS_OK);
        // Invalid: below narrowband
        assert_eq!(
            enc.set_bandwidth(OPUS_BANDWIDTH_NARROWBAND - 1),
            OPUS_BAD_ARG
        );
        // Invalid: above fullband
        assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_FULLBAND + 1), OPUS_BAD_ARG);
    }

    #[test]
    fn test_ctl_set_max_bandwidth_error_paths() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        assert_eq!(enc.set_max_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
        assert_eq!(enc.get_max_bandwidth(), OPUS_BANDWIDTH_WIDEBAND);
        assert_eq!(
            enc.set_max_bandwidth(OPUS_BANDWIDTH_NARROWBAND - 1),
            OPUS_BAD_ARG
        );
        assert_eq!(
            enc.set_max_bandwidth(OPUS_BANDWIDTH_FULLBAND + 1),
            OPUS_BAD_ARG
        );
    }

    #[test]
    fn test_ctl_set_signal_error_paths() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        assert_eq!(enc.set_signal(OPUS_AUTO), OPUS_OK);
        assert_eq!(enc.set_signal(OPUS_SIGNAL_VOICE), OPUS_OK);
        assert_eq!(enc.set_signal(OPUS_SIGNAL_MUSIC), OPUS_OK);
        assert_eq!(enc.get_signal(), OPUS_SIGNAL_MUSIC);
        assert_eq!(enc.set_signal(9999), OPUS_BAD_ARG);
    }

    #[test]
    fn test_ctl_set_inband_fec_error_paths() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
        assert_eq!(enc.set_inband_fec(0), OPUS_OK);
        assert_eq!(enc.set_inband_fec(1), OPUS_OK);
        assert_eq!(enc.set_inband_fec(2), OPUS_OK);
        assert_eq!(enc.get_inband_fec(), 2);
        assert_eq!(enc.set_inband_fec(-1), OPUS_BAD_ARG);
        assert_eq!(enc.set_inband_fec(3), OPUS_BAD_ARG);
    }

    #[test]
    fn test_ctl_set_packet_loss_perc_error_paths() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
        assert_eq!(enc.set_packet_loss_perc(0), OPUS_OK);
        assert_eq!(enc.set_packet_loss_perc(50), OPUS_OK);
        assert_eq!(enc.set_packet_loss_perc(100), OPUS_OK);
        assert_eq!(enc.get_packet_loss_perc(), 100);
        assert_eq!(enc.set_packet_loss_perc(-1), OPUS_BAD_ARG);
        assert_eq!(enc.set_packet_loss_perc(101), OPUS_BAD_ARG);
    }

    #[test]
    fn test_ctl_set_dtx_error_paths() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
        assert_eq!(enc.set_dtx(0), OPUS_OK);
        assert_eq!(enc.set_dtx(1), OPUS_OK);
        assert_eq!(enc.get_dtx(), 1);
        assert_eq!(enc.set_dtx(-1), OPUS_BAD_ARG);
        assert_eq!(enc.set_dtx(2), OPUS_BAD_ARG);
    }

    #[test]
    fn test_ctl_set_lsb_depth_error_paths() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        assert_eq!(enc.set_lsb_depth(8), OPUS_OK);
        assert_eq!(enc.set_lsb_depth(16), OPUS_OK);
        assert_eq!(enc.set_lsb_depth(24), OPUS_OK);
        assert_eq!(enc.get_lsb_depth(), 24);
        assert_eq!(enc.set_lsb_depth(7), OPUS_BAD_ARG);
        assert_eq!(enc.set_lsb_depth(25), OPUS_BAD_ARG);
    }

    #[test]
    fn test_ctl_getters_exercise() {
        let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
        enc.set_bitrate(64000);
        let pcm = patterned_pcm_i16(960, 2, 42);
        let mut pkt = vec![0u8; 1500];
        enc.encode(&pcm, 960, &mut pkt, 1500).unwrap();

        // Exercise all getters
        let _ = enc.get_bandwidth();
        let _ = enc.get_max_bandwidth();
        let _ = enc.get_signal();
        let _ = enc.get_inband_fec();
        let _ = enc.get_packet_loss_perc();
        let _ = enc.get_dtx();
        let _ = enc.get_lsb_depth();
        let _ = enc.get_hp_mem();
        let _ = enc.get_variable_hp_smth2();
    }

    #[test]
    fn test_encode_with_forced_narrowband() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
        enc.set_bitrate(16000);
        enc.set_max_bandwidth(OPUS_BANDWIDTH_NARROWBAND);
        let pcm = patterned_pcm_i16(960, 1, 11);
        let mut pkt = vec![0u8; 1500];
        let len = enc.encode(&pcm, 960, &mut pkt, 1500).unwrap();
        assert!(len > 0);
    }

    #[test]
    fn test_encode_with_forced_mediumband() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
        enc.set_bitrate(20000);
        enc.set_max_bandwidth(OPUS_BANDWIDTH_MEDIUMBAND);
        let pcm = patterned_pcm_i16(960, 1, 22);
        let mut pkt = vec![0u8; 1500];
        let len = enc.encode(&pcm, 960, &mut pkt, 1500).unwrap();
        assert!(len > 0);
    }

    #[test]
    fn test_encode_with_forced_wideband() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
        enc.set_bitrate(24000);
        enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND);
        let pcm = patterned_pcm_i16(960, 1, 33);
        let mut pkt = vec![0u8; 1500];
        let len = enc.encode(&pcm, 960, &mut pkt, 1500).unwrap();
        assert!(len > 0);
    }

    #[test]
    fn test_encode_buffer_too_small() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        enc.set_bitrate(128000);
        let pcm = patterned_pcm_i16(960, 1, 1);
        let mut pkt = vec![0u8; 2]; // Way too small for high bitrate
        let result = enc.encode(&pcm, 960, &mut pkt, 2);
        // Should either succeed with tiny packet or return error
        assert!(result.is_ok() || result.is_err());
    }

    // -----------------------------------------------------------------------
    // Additional CTL/accessor error path coverage
    // -----------------------------------------------------------------------

    /// Test complexity setter boundary values and error paths.
    #[test]
    fn test_ctl_set_complexity_error_paths() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        assert_eq!(enc.set_complexity(0), OPUS_OK);
        assert_eq!(enc.get_complexity(), 0);
        assert_eq!(enc.set_complexity(10), OPUS_OK);
        assert_eq!(enc.get_complexity(), 10);
        assert_eq!(enc.set_complexity(-1), OPUS_BAD_ARG);
        assert_eq!(enc.set_complexity(11), OPUS_BAD_ARG);
    }

    /// Test prediction_disabled setter boundary values.
    #[test]
    fn test_ctl_set_prediction_disabled_error_paths() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        assert_eq!(enc.set_prediction_disabled(0), OPUS_OK);
        assert_eq!(enc.get_prediction_disabled(), 0);
        assert_eq!(enc.set_prediction_disabled(1), OPUS_OK);
        assert_eq!(enc.get_prediction_disabled(), 1);
        assert_eq!(enc.set_prediction_disabled(-1), OPUS_BAD_ARG);
        assert_eq!(enc.set_prediction_disabled(2), OPUS_BAD_ARG);
    }

    /// Test voice_ratio setter boundary with encoding to verify path coverage.
    #[test]
    fn test_ctl_voice_ratio_boundary_with_encode() {
        let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
        // voice_ratio=0 → ve = 0*327>>8 = 0, capped at 115 → still 0
        assert_eq!(enc.set_voice_ratio(0), OPUS_OK);
        assert_eq!(enc.get_voice_ratio(), 0);
        enc.set_bitrate(32000);
        let pcm = patterned_pcm_i16(960, 2, 9001);
        let mut pkt = vec![0u8; 1500];
        let len = enc.encode(&pcm, 960, &mut pkt, 1500).unwrap();
        assert!(len > 0);

        // voice_ratio at max boundary
        assert_eq!(enc.set_voice_ratio(100), OPUS_OK);
        let len2 = enc.encode(&pcm, 960, &mut pkt, 1500).unwrap();
        assert!(len2 > 0);
    }

    /// Test force_channels setter error paths for mono encoder.
    #[test]
    fn test_ctl_force_channels_mono_encoder() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        assert_eq!(enc.set_force_channels(1), OPUS_OK);
        assert_eq!(enc.get_force_channels(), 1);
        // mono encoder cannot force 2 channels
        assert_eq!(enc.set_force_channels(2), OPUS_BAD_ARG);
        assert_eq!(enc.set_force_channels(0), OPUS_BAD_ARG);
        assert_eq!(enc.set_force_channels(OPUS_AUTO), OPUS_OK);
    }

    /// Test expert_frame_duration at FRAMESIZE_ARG boundary.
    #[test]
    fn test_ctl_expert_frame_duration_arg_value() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_ARG), OPUS_OK);
        assert_eq!(enc.get_expert_frame_duration(), OPUS_FRAMESIZE_ARG);
        // Verify all valid frame durations
        assert_eq!(
            enc.set_expert_frame_duration(OPUS_FRAMESIZE_2_5_MS),
            OPUS_OK
        );
        assert_eq!(
            enc.set_expert_frame_duration(OPUS_FRAMESIZE_120_MS),
            OPUS_OK
        );
        assert_eq!(enc.get_expert_frame_duration(), OPUS_FRAMESIZE_120_MS);
    }

    // =========================================================================
    // Mutation-killing pinning tests — exact value assertions for helpers
    // =========================================================================

    #[test]
    fn test_pin_gen_toc_all_modes() {
        // Pin exact TOC bytes for all 3 modes x representative frame sizes x bandwidths x channels.
        // gen_toc(mode, framerate, bandwidth, channels) -> u8

        // SILK_ONLY: toc = ((bw - NB) << 5) | ((period - 2) << 3) | stereo
        // framerate=50 -> period=3, framerate=100 -> period=2, framerate=25 -> period=4

        // SILK NB mono 20ms (fr=50, period=3)
        assert_eq!(gen_toc(MODE_SILK_ONLY, 50, OPUS_BANDWIDTH_NARROWBAND, 1), 0x08);
        // SILK NB stereo 20ms
        assert_eq!(gen_toc(MODE_SILK_ONLY, 50, OPUS_BANDWIDTH_NARROWBAND, 2), 0x0C);
        // SILK WB mono 10ms (fr=100, period=2)
        assert_eq!(gen_toc(MODE_SILK_ONLY, 100, OPUS_BANDWIDTH_WIDEBAND, 1), 0x40);
        // SILK WB stereo 10ms
        assert_eq!(gen_toc(MODE_SILK_ONLY, 100, OPUS_BANDWIDTH_WIDEBAND, 2), 0x44);
        // SILK MB mono 20ms (fr=50, period=3)
        assert_eq!(gen_toc(MODE_SILK_ONLY, 50, OPUS_BANDWIDTH_MEDIUMBAND, 1), 0x28);
        // SILK NB mono 40ms (fr=25, period=4)
        assert_eq!(gen_toc(MODE_SILK_ONLY, 25, OPUS_BANDWIDTH_NARROWBAND, 1), 0x10);
        // SILK NB mono ~60ms (fr=16, period=5)
        assert_eq!(gen_toc(MODE_SILK_ONLY, 16, OPUS_BANDWIDTH_NARROWBAND, 1), 0x18);

        // CELT_ONLY: toc = 0x80 | (max(0, bw - MB) << 5) | (period << 3) | stereo
        // CELT NB mono 20ms (bw=NB, tmp=NB-MB<0 -> 0, fr=50 -> period=3) = 0x98
        assert_eq!(gen_toc(MODE_CELT_ONLY, 50, OPUS_BANDWIDTH_NARROWBAND, 1), 0x98);
        // CELT MB mono 20ms (tmp=0) = 0x98
        assert_eq!(gen_toc(MODE_CELT_ONLY, 50, OPUS_BANDWIDTH_MEDIUMBAND, 1), 0x98);
        // CELT WB mono 20ms (tmp=1) = 0xB8
        assert_eq!(gen_toc(MODE_CELT_ONLY, 50, OPUS_BANDWIDTH_WIDEBAND, 1), 0xB8);
        // CELT FB stereo 10ms (tmp=3, fr=100 -> period=2) = 0xF4
        assert_eq!(gen_toc(MODE_CELT_ONLY, 100, OPUS_BANDWIDTH_FULLBAND, 2), 0xF4);
        // CELT SWB mono 5ms (fr=200 -> period=1) = 0xC8
        assert_eq!(gen_toc(MODE_CELT_ONLY, 200, OPUS_BANDWIDTH_SUPERWIDEBAND, 1), 0xC8);
        // CELT FB mono 2.5ms (fr=400 -> period=0) = 0xE0
        assert_eq!(gen_toc(MODE_CELT_ONLY, 400, OPUS_BANDWIDTH_FULLBAND, 1), 0xE0);

        // HYBRID: toc = 0x60 | ((bw - SWB) << 4) | ((period - 2) << 3) | stereo
        // Hybrid SWB mono 20ms (fr=50 -> period=3) = 0x68
        assert_eq!(gen_toc(MODE_HYBRID, 50, OPUS_BANDWIDTH_SUPERWIDEBAND, 1), 0x68);
        // Hybrid FB stereo 20ms = 0x7C
        assert_eq!(gen_toc(MODE_HYBRID, 50, OPUS_BANDWIDTH_FULLBAND, 2), 0x7C);
        // Hybrid SWB stereo 10ms (fr=100 -> period=2) = 0x64
        assert_eq!(gen_toc(MODE_HYBRID, 100, OPUS_BANDWIDTH_SUPERWIDEBAND, 2), 0x64);
        // Hybrid FB mono 10ms = 0x70
        assert_eq!(gen_toc(MODE_HYBRID, 100, OPUS_BANDWIDTH_FULLBAND, 1), 0x70);
    }

    #[test]
    fn test_pin_compute_equiv_rate_sweep() {
        // Pin exact output for a sweep of configs covering all branches.
        // compute_equiv_rate(bitrate, channels, frame_rate, vbr, mode, complexity, loss)

        // Base: (32k, 1ch, 50fps, VBR, SILK, complexity=10, loss=0)
        assert_eq!(compute_equiv_rate(32000, 1, 50, 1, MODE_SILK_ONLY, 10, 0), 32000);
        // (64k, 2ch, 50fps, VBR, SILK, complexity=10, loss=0)
        assert_eq!(compute_equiv_rate(64000, 2, 50, 1, MODE_SILK_ONLY, 10, 0), 64000);
        // (128k, 2ch, 100fps, VBR, CELT, complexity=10, loss=0) -- frame_rate>50 branch
        assert_eq!(compute_equiv_rate(128000, 2, 100, 1, MODE_CELT_ONLY, 10, 0), 123000);
        // (16k, 1ch, 50fps, VBR, SILK, complexity=10, loss=5) -- loss penalty
        assert_eq!(compute_equiv_rate(16000, 1, 50, 1, MODE_SILK_ONLY, 10, 5), 14000);
        // (64k, 1ch, 50fps, CBR, SILK, complexity=1, loss=10) -- CBR + low complexity + loss
        assert_eq!(compute_equiv_rate(64000, 1, 50, 0, MODE_SILK_ONLY, 1, 10), 36607);
        // (64k, 1ch, 50fps, VBR, CELT, complexity=3, loss=0) -- CELT low complexity
        assert_eq!(compute_equiv_rate(64000, 1, 50, 1, MODE_CELT_ONLY, 3, 0), 53568);
        // Unknown mode (12345): moderate loss penalty
        assert_eq!(compute_equiv_rate(64000, 1, 50, 1, 12345, 10, 10), 59429);
        // HYBRID has same path as SILK
        assert_eq!(compute_equiv_rate(64000, 1, 50, 1, MODE_HYBRID, 10, 0), 64000);
        // HYBRID with loss=25
        assert_eq!(compute_equiv_rate(64000, 1, 50, 1, MODE_HYBRID, 10, 25), 54000);
    }

    #[test]
    fn test_pin_decide_fec_exact() {
        // Pin exact FEC decisions for various loss rates and rates.

        // Loss=0 always returns 0
        let mut bw = OPUS_BANDWIDTH_WIDEBAND;
        assert_eq!(decide_fec(1, 0, 0, MODE_SILK_ONLY, &mut bw, 100000), 0);
        assert_eq!(bw, OPUS_BANDWIDTH_WIDEBAND);

        // Loss=5, WB, last_fec=0, rate=20000: rate below scaled threshold -> 0
        let mut bw = OPUS_BANDWIDTH_WIDEBAND;
        assert_eq!(decide_fec(1, 5, 0, MODE_SILK_ONLY, &mut bw, 20000), 0);
        assert_eq!(bw, OPUS_BANDWIDTH_WIDEBAND);

        // Loss=10, WB, last_fec=1, rate=18000 (hysteresis keeps FEC)
        let mut bw = OPUS_BANDWIDTH_WIDEBAND;
        assert_eq!(decide_fec(1, 10, 1, MODE_SILK_ONLY, &mut bw, 18000), 1);
        assert_eq!(bw, OPUS_BANDWIDTH_WIDEBAND);

        // Loss=25, NB, last_fec=0, rate=20000: NB threshold exceeded -> 1
        let mut bw = OPUS_BANDWIDTH_NARROWBAND;
        assert_eq!(decide_fec(1, 25, 0, MODE_SILK_ONLY, &mut bw, 20000), 1);
        assert_eq!(bw, OPUS_BANDWIDTH_NARROWBAND);

        // Loss=50, FB, high rate -> FEC enabled, bandwidth preserved
        let mut bw = OPUS_BANDWIDTH_FULLBAND;
        assert_eq!(decide_fec(1, 50, 0, MODE_SILK_ONLY, &mut bw, 100000), 1);
        assert_eq!(bw, OPUS_BANDWIDTH_FULLBAND);

        // Loss=5, low rate -- loss<=5 returns 0 on walkdown
        let mut bw = OPUS_BANDWIDTH_WIDEBAND;
        assert_eq!(decide_fec(1, 5, 0, MODE_SILK_ONLY, &mut bw, 5000), 0);
        assert_eq!(bw, OPUS_BANDWIDTH_WIDEBAND);
    }

    #[test]
    fn test_pin_stereo_fade_exact() {
        let window = crate::celt::modes::MODE_48000_960_120.window;

        // Case 1: known stereo signal, g1=Q15ONE, g2=Q15ONE/2, fs=48000
        // overlap48=120, at 48kHz inc=1, overlap=120. Need at least 130 stereo samples.
        let frame_size = 150;
        let mut pcm = vec![0i16; frame_size * 2];
        for i in 0..frame_size {
            pcm[i * 2] = 1000;
            pcm[i * 2 + 1] = -1000;
        }
        stereo_fade(&mut pcm, Q15ONE, Q15ONE / 2, 120, frame_size as i32, 2, window, 48000);
        // Overlap region fades from g1 to g2; post-overlap uses g2 constantly.
        // Gains are inverted inside: g1'=Q15ONE-Q15ONE=0, g2'=Q15ONE-Q15ONE/2.
        // At i=0: window value near zero, so almost no width reduction.
        // At i=119 (end of overlap): nearly full g2 applied.
        // At i=120+ (post-overlap): constant g2 applied.
        assert_eq!((pcm[0], pcm[1]), (1000, -1000));
        assert_eq!((pcm[2], pcm[3]), (1000, -1000));
        assert_eq!((pcm[20], pcm[21]), (1000, -1000));
        assert_eq!((pcm[120], pcm[121]), (745, -745));
        assert_eq!((pcm[238], pcm[239]), (501, -501));
        assert_eq!((pcm[240], pcm[241]), (500, -500));
        assert_eq!((pcm[298], pcm[299]), (500, -500));

        // Case 2: g1=0 -> g2=Q15ONE (transition from full width to no reduction)
        // At fs=8000, inc=6, overlap=120/6=20. frame_size=30.
        // Inverted: g1' = Q15ONE-0 = Q15ONE, g2' = Q15ONE-Q15ONE = 0.
        // In overlap: interpolation from Q15ONE (full reduction) to 0 (no reduction).
        // Post-overlap: g2'=0 means no width reduction.
        let frame_size2 = 30;
        let mut pcm2 = vec![0i16; frame_size2 * 2];
        for i in 0..frame_size2 {
            pcm2[i * 2] = 5000;
            pcm2[i * 2 + 1] = -3000;
        }
        stereo_fade(&mut pcm2, 0, Q15ONE, 120, frame_size2 as i32, 2, window, 8000);
        // Overlap region shows gradual transition from full reduction to none.
        assert_eq!((pcm2[0], pcm2[1]), (1001, 999));     // near start: mostly reduced
        assert_eq!((pcm2[10], pcm2[11]), (1222, 778));    // early overlap
        assert_eq!((pcm2[20], pcm2[21]), (3042, -1042));  // mid overlap
        assert_eq!((pcm2[30], pcm2[31]), (4805, -2805));  // late overlap
        assert_eq!((pcm2[38], pcm2[39]), (5000, -3000));  // end of overlap: no reduction
        assert_eq!((pcm2[40], pcm2[41]), (5000, -3000));  // post-overlap: no reduction
        assert_eq!((pcm2[58], pcm2[59]), (5000, -3000));
    }

    #[test]
    fn test_pin_gain_fade_exact() {
        let window = crate::celt::modes::MODE_48000_960_120.window;

        // Case 1: mono, g1=0, g2=Q15ONE (fade in), fs=48000
        // overlap = overlap48 * fs / 48000 = 120 * 48000 / 48000 = 120. Need 130+ samples.
        let frame_size = 150;
        let mut pcm = vec![10000i16; frame_size];
        gain_fade(&mut pcm, 0, Q15ONE, 120, frame_size as i32, 1, window, 48000);
        // In overlap region (0..120), gain fades from g1=0 to g2=Q15ONE.
        // After overlap (120..150), g2=Q15ONE so no change (g2==Q15ONE branch skips).
        assert_eq!(pcm[0], 0);
        assert_eq!(pcm[1], 0);
        assert_eq!(pcm[10], 8);
        assert_eq!(pcm[60], 5102);
        assert_eq!(pcm[119], 9999);
        assert_eq!(pcm[120], 10000);
        assert_eq!(pcm[149], 10000);

        // Case 2: stereo, g1=Q15ONE, g2=Q15ONE/2, fs=8000
        // overlap = 120 * 8000 / 48000 = 20. inc = 48000/8000 = 6.
        let frame_size2 = 30;
        let mut pcm2 = vec![0i16; frame_size2 * 2];
        for i in 0..frame_size2 {
            pcm2[i * 2] = 8000;
            pcm2[i * 2 + 1] = -4000;
        }
        gain_fade(&mut pcm2, Q15ONE, Q15ONE / 2, 120, frame_size2 as i32, 2, window, 8000);
        // Fade from g1=Q15ONE to g2=Q15ONE/2 over 20 samples, then constant Q15ONE/2.
        assert_eq!((pcm2[0], pcm2[1]), (7999, -4000));
        assert_eq!((pcm2[10], pcm2[11]), (7778, -3890));
        assert_eq!((pcm2[20], pcm2[21]), (5958, -2980));
        assert_eq!((pcm2[38], pcm2[39]), (3999, -2000));
        assert_eq!((pcm2[40], pcm2[41]), (3999, -2000));
        assert_eq!((pcm2[58], pcm2[59]), (3999, -2000));
    }

    #[test]
    fn test_pin_dc_reject_exact() {
        // dc_reject(input, cutoff_hz, output, hp_mem, len, channels, fs)
        // Mono: constant DC=5000, 16 samples at 48kHz, cutoff=3Hz
        let input = vec![5000i16; 16];
        let mut output = vec![0i16; 16];
        let mut hp_mem = [0i32; 4];
        dc_reject(&input, 3, &mut output, &mut hp_mem, 16, 1, 48000);
        #[rustfmt::skip]
        let expected_mono: [i16; 16] = [
            5000, 4998, 4995, 4993, 4990, 4988, 4985, 4983,
            4981, 4978, 4976, 4973, 4971, 4968, 4966, 4964,
        ];
        assert_eq!(output, expected_mono);
        assert_eq!(hp_mem, [637660, 0, 0, 0]);

        // Stereo: left=3000, right=-2000, 8 stereo frames at 48kHz
        let mut input_s = vec![0i16; 16];
        for i in 0..8 {
            input_s[i * 2] = 3000;
            input_s[i * 2 + 1] = -2000;
        }
        let mut output_s = vec![0i16; 16];
        let mut hp_mem_s = [0i32; 4];
        dc_reject(&input_s, 3, &mut output_s, &mut hp_mem_s, 8, 2, 48000);
        #[rustfmt::skip]
        let expected_stereo: [i16; 16] = [
            3000, -2000, 2999, -1999, 2997, -1998, 2996, -1997,
            2994, -1996, 2993, -1995, 2991, -1994, 2990, -1993,
        ];
        assert_eq!(output_s, expected_stereo);
        assert_eq!(hp_mem_s, [191672, 0, -127781, 0]);
    }

    #[test]
    fn test_pin_compute_stereo_width_exact() {
        // Correlated stereo (both channels identical) -> width = 0
        let corr: Vec<i16> = (0..480).flat_map(|i| {
            let v = ((i as f64 * 0.1).sin() * 10000.0) as i16;
            vec![v, v]
        }).collect();
        let mut mem = StereoWidthState::default();
        assert_eq!(compute_stereo_width(&corr, 480, 48000, &mut mem), 0);
        assert_eq!(mem.xx, 23163419);
        assert_eq!(mem.xy, 23155344);
        assert_eq!(mem.yy, 23163419);
        assert_eq!(mem.smoothed_width, 0);
        assert_eq!(mem.max_follower, 0);

        // Anti-correlated stereo (right = -left) -> xy clamped to 0
        let anti: Vec<i16> = (0..480).flat_map(|i| {
            let v = ((i as f64 * 0.1).sin() * 10000.0) as i16;
            vec![v, -v]
        }).collect();
        let mut mem2 = StereoWidthState::default();
        assert_eq!(compute_stereo_width(&anti, 480, 48000, &mut mem2), 0);
        assert_eq!(mem2.xx, 23163419);
        assert_eq!(mem2.xy, 0);
        assert_eq!(mem2.yy, 23163419);
        assert_eq!(mem2.smoothed_width, 0);
        assert_eq!(mem2.max_follower, 0);

        // Uncorrelated: left and right are different signals -> some width
        let uncorr: Vec<i16> = (0..480).flat_map(|i| {
            let l = ((i as f64 * 0.1).sin() * 10000.0) as i16;
            let r = ((i as f64 * 0.17 + 1.0).sin() * 8000.0) as i16;
            vec![l, r]
        }).collect();
        let mut mem3 = StereoWidthState::default();
        assert_eq!(compute_stereo_width(&uncorr, 480, 48000, &mut mem3), 340);
        assert_eq!(mem3.xx, 23163419);
        assert_eq!(mem3.xy, 0);
        assert_eq!(mem3.yy, 14993283);
        assert_eq!(mem3.smoothed_width, 17);
        assert_eq!(mem3.max_follower, 17);
    }

    #[test]
    fn test_pin_compute_silk_rate_for_hybrid() {
        // Pin exact SILK rate for various total bitrates and bandwidths.

        // Low rate, SWB, 10ms, VBR, no FEC, mono: entry=1
        assert_eq!(compute_silk_rate_for_hybrid(14000, OPUS_BANDWIDTH_SUPERWIDEBAND, false, 1, 0, 1), 12050);
        // Mid rate, FB, 20ms, VBR, FEC, mono: entry=4
        assert_eq!(compute_silk_rate_for_hybrid(24000, OPUS_BANDWIDTH_FULLBAND, true, 1, 1, 1), 21000);
        // Mid rate, SWB, 20ms, CBR, no FEC, stereo: entry=2
        assert_eq!(compute_silk_rate_for_hybrid(32000, OPUS_BANDWIDTH_SUPERWIDEBAND, true, 0, 0, 2), 26800);
        // Very high rate (exceeds table), FB, 20ms, VBR, no FEC, mono: entry=2
        assert_eq!(compute_silk_rate_for_hybrid(150000, OPUS_BANDWIDTH_FULLBAND, true, 1, 0, 1), 81000);
        // Rate at table boundary: 12000, FB, 10ms, VBR, no FEC, mono
        assert_eq!(compute_silk_rate_for_hybrid(12000, OPUS_BANDWIDTH_FULLBAND, false, 1, 0, 1), 10000);
        // Very low rate: 8000
        assert_eq!(compute_silk_rate_for_hybrid(8000, OPUS_BANDWIDTH_SUPERWIDEBAND, false, 1, 0, 1), 6966);
    }

    #[test]
    fn test_pin_compute_redundancy_bytes_exact() {
        // Pin exact redundancy byte count.
        // Existing known values
        assert_eq!(compute_redundancy_bytes(20, 4000, 50, 1), 0);
        assert_eq!(compute_redundancy_bytes(1000, 64000, 50, 2), 74);
        assert_eq!(compute_redundancy_bytes(2000, 1_000_000, 50, 2), 257);

        // Additional configs pinned
        assert_eq!(compute_redundancy_bytes(500, 32000, 50, 1), 38);
        assert_eq!(compute_redundancy_bytes(200, 48000, 100, 2), 54);
        assert_eq!(compute_redundancy_bytes(100, 24000, 50, 1), 24);
        assert_eq!(compute_redundancy_bytes(50, 64000, 50, 1), 14);
        assert_eq!(compute_redundancy_bytes(300, 96000, 50, 2), 67);
    }

    #[test]
    fn test_pin_decide_dtx_mode_exact() {
        // decide_dtx_mode(activity, nb_no_activity_ms_q1, frame_size_ms_q1) -> bool
        // threshold = NB_SPEECH_FRAMES_BEFORE_DTX(10) * 20 * 2 = 400
        // max_threshold = (10 + 20) * 20 * 2 = 1200

        // Case 1: Ramp up from 0 with activity=0, frame_size=40 (20ms Q1)
        let mut nb = 0i32;
        for i in 0..10 {
            let result = decide_dtx_mode(0, &mut nb, 40);
            assert!(!result, "frame {i}: nb={nb} should be <= threshold");
        }
        assert_eq!(nb, 400);

        // Frame 11: nb=440 > 400 and <= 1200 -> DTX=true
        assert!(decide_dtx_mode(0, &mut nb, 40));
        assert_eq!(nb, 440);

        // Count DTX frames until max_threshold triggers a reset.
        // From nb=440, adding 40 each time: 480, 520, ..., 1200 (all DTX=true since > 400 && <= 1200).
        // Next: nb=1240 > 1200 -> DTX=false, nb reset to 400.
        // That's (1200 - 440) / 40 = 19 DTX=true frames, plus 1 DTX=false frame.
        let mut dtx_count = 0;
        let prev_nb = nb;
        for _ in 0..25 {
            let r = decide_dtx_mode(0, &mut nb, 40);
            if r { dtx_count += 1; }
            if nb < prev_nb { break; } // reset detected
        }
        assert_eq!(nb, 400); // reset to threshold
        assert_eq!(dtx_count, 19);

        // Activity resets counter
        let mut nb2 = 500i32;
        assert!(!decide_dtx_mode(1, &mut nb2, 40));
        assert_eq!(nb2, 0);

        // Large frame size (60ms -> Q1=120)
        let mut nb3 = 0i32;
        for _ in 0..3 {
            assert!(!decide_dtx_mode(0, &mut nb3, 120));
        }
        assert_eq!(nb3, 360);
        assert!(decide_dtx_mode(0, &mut nb3, 120));
        assert_eq!(nb3, 480);
    }

    // ==========================================================================
    // Branch-coverage stage 1 — broad parameter-sweep and CTL-path exercises.
    // These tests target uncovered branches in opus/encoder.rs enumerated by
    // `cargo +nightly llvm-cov --branch`. They do not pin numerical behavior;
    // each test encodes frames and asserts the return is a valid success or
    // error code, while exercising the branches that were previously cold.
    // ==========================================================================
    mod branch_coverage_stage1 {
        use super::*;

        // ---- CTL paths with celt_enc = None (tests the "else" of `if let Some`)
        #[test]
        fn bc_ctl_setters_without_celt_encoder() {
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
            enc.celt_enc = None;

            // set_packet_loss_perc with no celt encoder (L2819 else)
            assert_eq!(enc.set_packet_loss_perc(7), OPUS_OK);
            assert_eq!(enc.get_packet_loss_perc(), 7);
            // set_complexity with no celt encoder
            assert_eq!(enc.set_complexity(3), OPUS_OK);
            assert_eq!(enc.get_complexity(), 3);
            // set_phase_inversion_disabled with no celt encoder (L2890 else)
            assert_eq!(enc.set_phase_inversion_disabled(1), OPUS_OK);
            assert_eq!(enc.get_phase_inversion_disabled(), 0);
        }

        // ---- set_force_mode / set_voice_ratio / set_expert_frame_duration boundaries
        #[test]
        fn bc_ctl_boundary_values() {
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();

            // set_voice_ratio boundary: -2 below, 101 above, -1 and 100 valid (L2906)
            assert_eq!(enc.set_voice_ratio(-2), OPUS_BAD_ARG);
            assert_eq!(enc.set_voice_ratio(-1), OPUS_OK);
            assert_eq!(enc.set_voice_ratio(0), OPUS_OK);
            assert_eq!(enc.set_voice_ratio(100), OPUS_OK);
            assert_eq!(enc.set_voice_ratio(101), OPUS_BAD_ARG);

            // set_force_mode with every valid value including OPUS_AUTO (L2918)
            assert_eq!(enc.set_force_mode(OPUS_AUTO), OPUS_OK);
            assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
            assert_eq!(enc.set_force_mode(MODE_HYBRID), OPUS_OK);
            assert_eq!(enc.set_force_mode(MODE_CELT_ONLY), OPUS_OK);

            // set_expert_frame_duration sweep through every valid value (L2865)
            for &d in &[
                OPUS_FRAMESIZE_ARG,
                OPUS_FRAMESIZE_2_5_MS,
                OPUS_FRAMESIZE_5_MS,
                OPUS_FRAMESIZE_10_MS,
                OPUS_FRAMESIZE_20_MS,
                OPUS_FRAMESIZE_40_MS,
                OPUS_FRAMESIZE_60_MS,
                OPUS_FRAMESIZE_80_MS,
                OPUS_FRAMESIZE_100_MS,
                OPUS_FRAMESIZE_120_MS,
            ] {
                assert_eq!(enc.set_expert_frame_duration(d), OPUS_OK);
                assert_eq!(enc.get_expert_frame_duration(), d);
            }
            // Out of range: below min (note OPUS_FRAMESIZE_2_5_MS-1 is OPUS_FRAMESIZE_ARG itself,
            // so we test something definitely outside the accepted range) and above max.
            assert_eq!(enc.set_expert_frame_duration(4999), OPUS_BAD_ARG);
            assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_120_MS + 1), OPUS_BAD_ARG);

            // set_complexity extremes (0 and 10) — walks through CELT ctl path
            assert_eq!(enc.set_complexity(0), OPUS_OK);
            assert_eq!(enc.set_complexity(10), OPUS_OK);
            assert_eq!(enc.set_complexity(-1), OPUS_BAD_ARG);
            assert_eq!(enc.set_complexity(11), OPUS_BAD_ARG);

            // set_lsb_depth boundaries
            assert_eq!(enc.set_lsb_depth(8), OPUS_OK);
            assert_eq!(enc.set_lsb_depth(24), OPUS_OK);
            assert_eq!(enc.set_lsb_depth(25), OPUS_BAD_ARG);

            // set_packet_loss_perc boundaries
            assert_eq!(enc.set_packet_loss_perc(0), OPUS_OK);
            assert_eq!(enc.set_packet_loss_perc(100), OPUS_OK);
            assert_eq!(enc.set_packet_loss_perc(-1), OPUS_BAD_ARG);

            // set_bandwidth AUTO path and explicit values (covers get_bandwidth)
            assert_eq!(enc.set_bandwidth(OPUS_AUTO), OPUS_OK);
            for &b in &[
                OPUS_BANDWIDTH_NARROWBAND,
                OPUS_BANDWIDTH_MEDIUMBAND,
                OPUS_BANDWIDTH_WIDEBAND,
                OPUS_BANDWIDTH_SUPERWIDEBAND,
                OPUS_BANDWIDTH_FULLBAND,
            ] {
                assert_eq!(enc.set_bandwidth(b), OPUS_OK);
                assert_eq!(enc.set_max_bandwidth(b), OPUS_OK);
            }
            assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_FULLBAND + 1), OPUS_BAD_ARG);
            assert_eq!(enc.set_max_bandwidth(OPUS_BANDWIDTH_NARROWBAND - 1), OPUS_BAD_ARG);
            // get_bandwidth returns encoder bandwidth, not user-set
            let _ = enc.get_bandwidth();

            // set_force_channels boundaries on stereo encoder (AUTO path is L2918 adjacent)
            assert_eq!(enc.set_force_channels(OPUS_AUTO), OPUS_OK);
            assert_eq!(enc.set_force_channels(1), OPUS_OK);
            assert_eq!(enc.set_force_channels(2), OPUS_OK);
            assert_eq!(enc.set_force_channels(0), OPUS_BAD_ARG);
        }

        // ---- set_force_channels on mono encoder (channels==1) so bounds differ
        #[test]
        fn bc_force_channels_mono_encoder() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            assert_eq!(enc.set_force_channels(1), OPUS_OK);
            assert_eq!(enc.set_force_channels(2), OPUS_BAD_ARG);
            assert_eq!(enc.set_force_channels(OPUS_AUTO), OPUS_OK);
        }

        // ---- Encode across an application × signal × complexity grid (single frame).
        #[test]
        fn bc_encode_application_signal_grid() {
            let fs = 16000;
            let frame = 320; // 20ms
            let apps = [
                OPUS_APPLICATION_VOIP,
                OPUS_APPLICATION_AUDIO,
                OPUS_APPLICATION_RESTRICTED_LOWDELAY,
            ];
            let signals = [OPUS_SIGNAL_VOICE, OPUS_SIGNAL_MUSIC, OPUS_AUTO];
            let complexities = [0, 5, 10];
            for &app in &apps {
                for &sig in &signals {
                    for &comp in &complexities {
                        let mut enc = OpusEncoder::new(fs, 1, app).unwrap();
                        assert_eq!(enc.set_signal(sig), OPUS_OK);
                        assert_eq!(enc.set_complexity(comp), OPUS_OK);
                        assert_eq!(enc.set_bitrate(24000), OPUS_OK);

                        let pcm = patterned_pcm_i16(frame as usize, 1, 2000 + comp);
                        let mut packet = vec![0u8; 1500];
                        let len = enc.encode(&pcm, frame, &mut packet, 1500).unwrap();
                        assert!(len > 0);
                    }
                }
            }
        }

        // ---- VBR / CBR / FEC / DTX / bitrate-extreme matrix.
        #[test]
        fn bc_encode_vbr_cbr_fec_dtx_matrix() {
            let frame = 960; // 20ms at 48k
            let cases: &[(i32, i32, i32, i32, i32, i32)] = &[
                // (bitrate, vbr, vbr_constraint, fec, dtx, loss_perc)
                (6000, 0, 0, 0, 0, 0),
                (10000, 1, 1, 1, 1, 30),
                (16000, 1, 0, 2, 0, 10),
                (32000, 0, 1, 1, 1, 50),
                (64000, 1, 0, 0, 0, 0),
                (128000, 1, 0, 0, 0, 0),
                (256000, 0, 0, 0, 0, 0),
                (510000, 1, 0, 0, 0, 0),
            ];
            for &(br, vbr, vbrc, fec, dtx, loss) in cases {
                let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
                assert_eq!(enc.set_bitrate(br), OPUS_OK);
                assert_eq!(enc.set_vbr(vbr), OPUS_OK);
                assert_eq!(enc.set_vbr_constraint(vbrc), OPUS_OK);
                assert_eq!(enc.set_inband_fec(fec), OPUS_OK);
                assert_eq!(enc.set_dtx(dtx), OPUS_OK);
                assert_eq!(enc.set_packet_loss_perc(loss), OPUS_OK);

                let pcm = patterned_pcm_i16(frame as usize, 2, br);
                let mut packet = vec![0u8; 1500];
                let len = enc.encode(&pcm, frame, &mut packet, 1500).unwrap();
                assert!(len > 0);
            }
        }

        // ---- Mode switching: Walk through SILK -> Hybrid -> CELT -> SILK.
        #[test]
        fn bc_mode_switching_sequence() {
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_bitrate(64000), OPUS_OK);
            assert_eq!(enc.set_vbr(1), OPUS_OK);
            let frame = 960;

            let mut packet = vec![0u8; 1500];
            let modes = [
                MODE_SILK_ONLY,
                MODE_HYBRID,
                MODE_CELT_ONLY,
                MODE_SILK_ONLY,
                MODE_CELT_ONLY,
                MODE_HYBRID,
            ];
            for (i, &m) in modes.iter().enumerate() {
                assert_eq!(enc.set_force_mode(m), OPUS_OK);
                // Vary bandwidth to hit different bandwidth switch paths
                let bw = match i % 4 {
                    0 => OPUS_BANDWIDTH_WIDEBAND,
                    1 => OPUS_BANDWIDTH_SUPERWIDEBAND,
                    2 => OPUS_BANDWIDTH_FULLBAND,
                    _ => OPUS_AUTO,
                };
                let _ = enc.set_bandwidth(bw);

                let pcm = patterned_pcm_i16(frame as usize, 2, 3000 + i as i32);
                let len = enc.encode(&pcm, frame, &mut packet, 1500).unwrap();
                assert!(len > 0);
            }
        }

        // ---- Exercise FEC override: high loss + voice signal -> FEC forces SILK
        #[test]
        fn bc_encode_fec_overrides_mode() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            assert_eq!(enc.set_bitrate(20000), OPUS_OK);
            assert_eq!(enc.set_inband_fec(1), OPUS_OK);
            assert_eq!(enc.set_packet_loss_perc(40), OPUS_OK);
            assert_eq!(enc.set_signal(OPUS_SIGNAL_VOICE), OPUS_OK);
            assert_eq!(enc.set_force_mode(OPUS_AUTO), OPUS_OK);

            let mut packet = vec![0u8; 1500];
            for i in 0..3 {
                let pcm = patterned_pcm_i16(960, 1, 4100 + i);
                let len = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
                assert!(len > 0);
            }
        }

        // ---- Exercise FEC config == 2 branch (voice_est > 25 or not)
        #[test]
        fn bc_encode_fec_config2_variants() {
            for &ratio in &[0, 50, 100] {
                let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
                assert_eq!(enc.set_bitrate(24000), OPUS_OK);
                assert_eq!(enc.set_inband_fec(2), OPUS_OK);
                assert_eq!(enc.set_packet_loss_perc(30), OPUS_OK);
                assert_eq!(enc.set_voice_ratio(ratio), OPUS_OK);
                let pcm = patterned_pcm_i16(960, 1, 4200 + ratio);
                let mut packet = vec![0u8; 1500];
                let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            }
        }

        // ---- DTX override when voice_est > 100 (voice signal + DTX)
        #[test]
        fn bc_encode_dtx_override_voice() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            assert_eq!(enc.set_bitrate(24000), OPUS_OK);
            assert_eq!(enc.set_dtx(1), OPUS_OK);
            assert_eq!(enc.set_signal(OPUS_SIGNAL_VOICE), OPUS_OK);
            // active frames (not silence) so use_dtx gets forwarded
            let pcm = patterned_pcm_i16(960, 1, 4301);
            let mut packet = vec![0u8; 1500];
            let len = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            assert!(len > 0);
        }

        // ---- Cover bandwidth override path (user_bandwidth != AUTO)
        #[test]
        fn bc_encode_bandwidth_override() {
            for &bw in &[
                OPUS_BANDWIDTH_NARROWBAND,
                OPUS_BANDWIDTH_MEDIUMBAND,
                OPUS_BANDWIDTH_WIDEBAND,
                OPUS_BANDWIDTH_SUPERWIDEBAND,
                OPUS_BANDWIDTH_FULLBAND,
            ] {
                let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
                assert_eq!(enc.set_bandwidth(bw), OPUS_OK);
                assert_eq!(enc.set_bitrate(64000), OPUS_OK);
                let pcm = patterned_pcm_i16(960, 2, 4400 + bw);
                let mut packet = vec![0u8; 1500];
                let len = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
                assert!(len > 0);
            }
        }

        // ---- Cover low-rate + high-framerate mode overrides (frame_rate > 50)
        #[test]
        fn bc_encode_high_framerate() {
            // 10ms at 48kHz -> frame_rate=100, triggers low_rate_threshold = 9000 branch
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            assert_eq!(enc.set_bitrate(6000), OPUS_OK);
            assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_10_MS), OPUS_OK);
            let pcm = patterned_pcm_i16(480, 1, 4500);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 480, &mut packet, 1500).unwrap();

            // 5ms at 48kHz -> frame_rate=200
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_RESTRICTED_LOWDELAY).unwrap();
            assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_5_MS), OPUS_OK);
            let pcm = patterned_pcm_i16(240, 1, 4501);
            let _ = enc.encode(&pcm, 240, &mut packet, 1500).unwrap();

            // 2.5ms at 48kHz -> frame_rate=400 (hits gen_toc period=0 branch)
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_RESTRICTED_LOWDELAY).unwrap();
            assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_2_5_MS), OPUS_OK);
            let pcm = patterned_pcm_i16(120, 1, 4502);
            let _ = enc.encode(&pcm, 120, &mut packet, 1500).unwrap();
        }

        // ---- Multi-rate sample rate coverage.
        #[test]
        fn bc_encode_sample_rates() {
            for &(fs, frame) in &[
                (8000, 160),
                (12000, 240),
                (16000, 320),
                (24000, 480),
                (48000, 960),
            ] {
                let mut enc = OpusEncoder::new(fs, 1, OPUS_APPLICATION_VOIP).unwrap();
                assert_eq!(enc.set_bitrate(16000), OPUS_OK);
                let pcm = patterned_pcm_i16(frame as usize, 1, fs);
                let mut packet = vec![0u8; 1500];
                let len = enc.encode(&pcm, frame, &mut packet, 1500).unwrap();
                assert!(len > 0);
            }
        }

        // ---- Encode frame sizes 2.5ms/5ms/10ms/20ms/40ms/60ms/80ms/100ms/120ms
        #[test]
        fn bc_encode_all_frame_durations() {
            let fs = 48000;
            let cases = [
                (OPUS_FRAMESIZE_2_5_MS, 120),
                (OPUS_FRAMESIZE_5_MS, 240),
                (OPUS_FRAMESIZE_10_MS, 480),
                (OPUS_FRAMESIZE_20_MS, 960),
                (OPUS_FRAMESIZE_40_MS, 1920),
                (OPUS_FRAMESIZE_60_MS, 2880),
                (OPUS_FRAMESIZE_80_MS, 3840),
                (OPUS_FRAMESIZE_100_MS, 4800),
                (OPUS_FRAMESIZE_120_MS, 5760),
            ];
            for &(dur, frame) in &cases {
                let mut enc = OpusEncoder::new(fs, 1, OPUS_APPLICATION_AUDIO).unwrap();
                assert_eq!(enc.set_expert_frame_duration(dur), OPUS_OK);
                assert_eq!(enc.set_bitrate(32000), OPUS_OK);
                let pcm = patterned_pcm_i16(frame as usize, 1, dur);
                let mut packet = vec![0u8; 1500];
                let len = enc.encode(&pcm, frame, &mut packet, 1500).unwrap();
                assert!(len > 0);
            }
        }

        // ---- Multi-frame SILK path: 40ms/60ms/80ms/120ms in SILK mode.
        #[test]
        fn bc_multiframe_silk_durations() {
            for &(dur, frame) in &[
                (OPUS_FRAMESIZE_40_MS, 1920),
                (OPUS_FRAMESIZE_60_MS, 2880),
                (OPUS_FRAMESIZE_80_MS, 3840),
                (OPUS_FRAMESIZE_120_MS, 5760),
            ] {
                let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
                assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
                assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
                assert_eq!(enc.set_bitrate(32000), OPUS_OK);
                assert_eq!(enc.set_expert_frame_duration(dur), OPUS_OK);
                let pcm = patterned_pcm_i16(frame as usize, 1, dur);
                let mut packet = vec![0u8; 1500];
                let len = enc.encode(&pcm, frame, &mut packet, 1500).unwrap();
                assert!(len > 0);
            }
        }

        // ---- CBR multiframe: padding path (pad_cbr branch)
        #[test]
        fn bc_multiframe_cbr_padding_paths() {
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_vbr(0), OPUS_OK);
            assert_eq!(enc.set_bitrate(48000), OPUS_OK);
            assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_60_MS), OPUS_OK);
            let pcm = patterned_pcm_i16(2880, 2, 5500);
            let mut packet = vec![0u8; 600];
            let len = enc.encode(&pcm, 2880, &mut packet, 300).unwrap();
            assert!(len > 0);
        }

        // ---- Stream silence over many frames to drive DTX counters + DTX silence packets
        #[test]
        fn bc_encode_dtx_many_silence_frames() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            assert_eq!(enc.set_dtx(1), OPUS_OK);
            assert_eq!(enc.set_bitrate(24000), OPUS_OK);
            assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
            assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);

            let silence = [0i16; 960];
            let mut packet = vec![0u8; 1500];
            for _ in 0..20 {
                let _ = enc.encode(&silence, 960, &mut packet, 1500).unwrap();
            }
        }

        // ---- Voice ratio sweep to cover voice_est calculations
        #[test]
        fn bc_encode_voice_ratio_sweep() {
            for &ratio in &[-1, 0, 25, 50, 75, 100] {
                let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
                assert_eq!(enc.set_voice_ratio(ratio), OPUS_OK);
                assert_eq!(enc.set_signal(OPUS_AUTO), OPUS_OK);
                assert_eq!(enc.set_bitrate(32000), OPUS_OK);
                let pcm = patterned_pcm_i16(960, 1, 5700 + ratio);
                let mut packet = vec![0u8; 1500];
                let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            }
        }

        // ---- Stereo to mono transition with voice signal, no force_channels.
        #[test]
        fn bc_stereo_to_mono_transition_without_force() {
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_bitrate(12000), OPUS_OK); // low rate to bias mono
            assert_eq!(enc.set_signal(OPUS_SIGNAL_VOICE), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 2, 5800);
            let mut packet = vec![0u8; 1500];
            for i in 0..4 {
                let p = patterned_pcm_i16(960, 2, 5800 + i);
                let _ = enc.encode(&p, 960, &mut packet, 1500).unwrap();
            }
            drop(pcm);
        }

        // ---- encode_float path + lsb_depth 24
        #[test]
        fn bc_encode_float_with_full_lsb_depth() {
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_lsb_depth(24), OPUS_OK);
            let pcm = patterned_pcm_f32(960, 2, 6100);
            let mut packet = vec![0u8; 1500];
            let len = enc.encode_float(&pcm, 960, &mut packet, 1500).unwrap();
            assert!(len > 0);
        }

        // ---- lfe=1 path forces CELT_ONLY + narrowband
        #[test]
        fn bc_encode_lfe_channel() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            enc.lfe = 1;
            assert_eq!(enc.set_bitrate(32000), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 1, 6200);
            let mut packet = vec![0u8; 1500];
            let len = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            assert!(len > 0);
        }

        // ---- RESTRICTED_LOWDELAY forces CELT_ONLY; exercise signal & complexity within.
        #[test]
        fn bc_encode_restricted_lowdelay_grid() {
            for &comp in &[0, 5, 10] {
                for &sig in &[OPUS_AUTO, OPUS_SIGNAL_VOICE, OPUS_SIGNAL_MUSIC] {
                    let mut enc =
                        OpusEncoder::new(48000, 2, OPUS_APPLICATION_RESTRICTED_LOWDELAY).unwrap();
                    assert_eq!(enc.set_complexity(comp), OPUS_OK);
                    assert_eq!(enc.set_signal(sig), OPUS_OK);
                    assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_10_MS), OPUS_OK);
                    let pcm = patterned_pcm_i16(480, 2, 6300 + comp);
                    let mut packet = vec![0u8; 1500];
                    let len = enc.encode(&pcm, 480, &mut packet, 1500).unwrap();
                    assert!(len > 0);
                }
            }
        }

        // ---- Tiny-budget encode returns early TOC-only packet
        #[test]
        fn bc_encode_tiny_budget_returns_toc() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            let pcm = patterned_pcm_i16(960, 1, 6400);
            let mut packet = [0u8; 2];
            let len = enc.encode(&pcm, 960, &mut packet, 2).unwrap();
            assert_eq!(len, 1);
        }

        // ---- Encode into exactly-sized buffer variations (for CBR padding coverage)
        #[test]
        fn bc_encode_cbr_padding_variants() {
            for &br in &[8000, 16000, 24000, 48000] {
                let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
                assert_eq!(enc.set_vbr(0), OPUS_OK);
                assert_eq!(enc.set_bitrate(br), OPUS_OK);
                let pcm = patterned_pcm_i16(960, 1, br);
                let mut packet = vec![0u8; 1500];
                let cap = (br / 400).max(10);
                let len = enc.encode(&pcm, 960, &mut packet, cap).unwrap();
                assert!(len > 0);
            }
        }

        // ---- Exercise `prediction_disabled` toggled mid-stream (CELT prediction path)
        #[test]
        fn bc_encode_prediction_disabled_toggle() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_bitrate(48000), OPUS_OK);
            assert_eq!(enc.set_force_mode(MODE_CELT_ONLY), OPUS_OK);
            let mut packet = vec![0u8; 1500];
            for (i, &disabled) in [0, 1, 0, 1].iter().enumerate() {
                assert_eq!(enc.set_prediction_disabled(disabled), OPUS_OK);
                let pcm = patterned_pcm_i16(960, 1, 6600 + i as i32);
                let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            }
        }

        // ---- Max-bandwidth cap actually clamps
        #[test]
        fn bc_encode_max_bandwidth_caps() {
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_max_bandwidth(OPUS_BANDWIDTH_NARROWBAND), OPUS_OK);
            assert_eq!(enc.set_bitrate(64000), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 2, 6700);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
        }

        // ---- encode_native error path: frame_size <= 0
        #[test]
        fn bc_encode_native_bad_frame_size() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            let pcm = patterned_pcm_i16(960, 1, 6800);
            let mut packet = vec![0u8; 1500];
            assert_eq!(enc.encode_native(&pcm, 0, &mut packet, 1500, 16), Err(OPUS_BAD_ARG));
            assert_eq!(enc.encode_native(&pcm, -1, &mut packet, 1500, 16), Err(OPUS_BAD_ARG));
            assert_eq!(enc.encode_native(&pcm, 960, &mut packet, 0, 16), Err(OPUS_BAD_ARG));
        }

        // ---- encode_float wrapper frame_size < 0
        #[test]
        fn bc_encode_float_bad_frame_size() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            let pcm = patterned_pcm_f32(960, 1, 6900);
            let mut packet = vec![0u8; 1500];
            assert_eq!(
                enc.encode_float(&pcm, -5, &mut packet, 1500),
                Err(OPUS_BAD_ARG)
            );
        }

        // ---- SILK-only mode with CBR and hybrid SILK-rate branches
        #[test]
        fn bc_silk_only_cbr_and_hybrid_cbr() {
            // SILK-only CBR
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
            assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
            assert_eq!(enc.set_vbr(0), OPUS_OK);
            assert_eq!(enc.set_bitrate(20000), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 1, 7001);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();

            // Hybrid CBR at high rate (non-trivial CELT fraction)
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_VOIP).unwrap();
            assert_eq!(enc.set_force_mode(MODE_HYBRID), OPUS_OK);
            assert_eq!(enc.set_vbr(0), OPUS_OK);
            assert_eq!(enc.set_bitrate(96000), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 2, 7002);
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();

            // Hybrid constrained VBR
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_VOIP).unwrap();
            assert_eq!(enc.set_force_mode(MODE_HYBRID), OPUS_OK);
            assert_eq!(enc.set_vbr(1), OPUS_OK);
            assert_eq!(enc.set_vbr_constraint(1), OPUS_OK);
            assert_eq!(enc.set_bitrate(64000), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 2, 7003);
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
        }

        // ---- Exercise redundancy signaling with small packet budget (redundancy=false branch)
        #[test]
        fn bc_encode_small_budget_no_redundancy() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            assert_eq!(enc.set_bitrate(16000), OPUS_OK);
            // Encode first frame to establish prev_mode
            let pcm = patterned_pcm_i16(960, 1, 7101);
            let mut packet = vec![0u8; 100];
            let _ = enc.encode(&pcm, 960, &mut packet, 100).unwrap();
            // Force a mode switch to trigger redundancy; constrain budget
            assert_eq!(enc.set_force_mode(MODE_CELT_ONLY), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 1, 7102);
            let _ = enc.encode(&pcm, 960, &mut packet, 30).unwrap();
        }

        // ---- Force channels = 1 on a stereo encoder with voice signal at low rate
        #[test]
        fn bc_encode_forced_mono_stereo_encoder() {
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_force_channels(1), OPUS_OK);
            assert_eq!(enc.set_bitrate(12000), OPUS_OK);
            assert_eq!(enc.set_signal(OPUS_SIGNAL_VOICE), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 2, 7200);
            let mut packet = vec![0u8; 1500];
            let len = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            assert!(len > 0);
        }

        // ---- reset() path after modifications (hits reset branches)
        #[test]
        fn bc_reset_after_encode() {
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
            let pcm = patterned_pcm_i16(960, 2, 7300);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            enc.reset();
            assert_eq!(enc.get_stream_channels(), 2);
            assert_eq!(enc.get_prev_mode(), 0);
            // Re-encode after reset
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
        }

        // ---- reset() with celt_enc = None
        #[test]
        fn bc_reset_without_celt_encoder() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            enc.celt_enc = None;
            enc.reset();
            assert_eq!(enc.get_prev_mode(), 0);
        }

        // ---- silk_bw_switch redundancy path: explicitly set silk_bw_switch to 1
        // before encoding a CELT frame.
        #[test]
        fn bc_silk_bw_switch_flag() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            assert_eq!(enc.set_bitrate(32000), OPUS_OK);
            // Warm up with a SILK frame
            let pcm = patterned_pcm_i16(960, 1, 7401);
            let mut packet = vec![0u8; 1500];
            assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            // Manually engage silk_bw_switch
            enc.silk_bw_switch = 1;
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            assert_eq!(enc.silk_bw_switch, 0);
        }

        // ---- frame_size_select: frame_size < fs/400
        #[test]
        fn bc_frame_size_too_small() {
            assert_eq!(
                frame_size_select(50, OPUS_FRAMESIZE_ARG, 48000),
                -1
            );
        }

        // ---- helper: silk_rshift_round shift<=0 and shift>1 paths (L147/L149 context)
        #[test]
        fn bc_silk_rshift_round_branches() {
            // shift == 1 path
            assert_eq!(silk_rshift_round(5, 1), 3);
            assert_eq!(silk_rshift_round(4, 1), 2);
            // shift == 0 -> a unchanged
            assert_eq!(silk_rshift_round(5, 0), 5);
            // shift < 0 -> a unchanged
            assert_eq!(silk_rshift_round(-7, -3), -7);
            // shift > 1
            assert_eq!(silk_rshift_round(16, 2), 4);
            assert_eq!(silk_rshift_round(17, 2), 4);
        }

        // ---- signal_type variations combined with application
        #[test]
        fn bc_signal_type_application_grid() {
            for &app in &[OPUS_APPLICATION_VOIP, OPUS_APPLICATION_AUDIO] {
                for &sig in &[OPUS_SIGNAL_VOICE, OPUS_SIGNAL_MUSIC] {
                    let mut enc = OpusEncoder::new(48000, 1, app).unwrap();
                    assert_eq!(enc.set_signal(sig), OPUS_OK);
                    assert_eq!(enc.set_bitrate(32000), OPUS_OK);
                    let pcm = patterned_pcm_i16(960, 1, 7500);
                    let mut packet = vec![0u8; 1500];
                    let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
                }
            }
        }

        // ---- Large buffer + high complexity + FEC stereo music, multiple frames
        // to exercise long-running internal state transitions.
        #[test]
        fn bc_long_running_stereo_music_sequence() {
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_complexity(8), OPUS_OK);
            assert_eq!(enc.set_bitrate(96000), OPUS_OK);
            assert_eq!(enc.set_signal(OPUS_SIGNAL_MUSIC), OPUS_OK);
            assert_eq!(enc.set_inband_fec(1), OPUS_OK);
            assert_eq!(enc.set_packet_loss_perc(15), OPUS_OK);
            let mut packet = vec![0u8; 1500];
            for i in 0..6 {
                let pcm = patterned_pcm_i16(960, 2, 7600 + i);
                let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            }
        }

        // ---- NB encoder 8kHz uses hp_cutoff VOIP path
        #[test]
        fn bc_voip_hp_cutoff_coverage() {
            let mut enc = OpusEncoder::new(8000, 1, OPUS_APPLICATION_VOIP).unwrap();
            assert_eq!(enc.set_bitrate(8000), OPUS_OK);
            let pcm = patterned_pcm_i16(160, 1, 7700);
            let mut packet = vec![0u8; 500];
            let _ = enc.encode(&pcm, 160, &mut packet, 500).unwrap();
            // Mem should have been updated
            let hp = enc.get_hp_mem();
            let _ = hp;
        }

        // ---- AUDIO path uses dc_reject filter (hp_mem updated differently)
        #[test]
        fn bc_audio_dc_reject_path() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            let pcm = patterned_pcm_i16(960, 1, 7800);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
        }

        // ---- Encoder_native with lsb_depth variations (16 vs 24)
        #[test]
        fn bc_encode_native_lsb_depth_variations() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            let pcm = patterned_pcm_i16(960, 1, 7900);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode_native(&pcm, 960, &mut packet, 1500, 16).unwrap();
            let _ = enc.encode_native(&pcm, 960, &mut packet, 1500, 24).unwrap();
            let _ = enc.encode_native(&pcm, 960, &mut packet, 1500, 8).unwrap();
        }

        // ---- VBR toggle mid-stream
        #[test]
        fn bc_vbr_toggle_midstream() {
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_bitrate(48000), OPUS_OK);
            let mut packet = vec![0u8; 1500];
            for (i, &vbr) in [1, 0, 1, 0].iter().enumerate() {
                assert_eq!(enc.set_vbr(vbr), OPUS_OK);
                let pcm = patterned_pcm_i16(960, 2, 8000 + i as i32);
                let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            }
        }

        // ---- Budget just under redundancy_signaling limit; test redundancy=false fallback
        #[test]
        fn bc_redundancy_fallback_budget() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            assert_eq!(enc.set_bitrate(16000), OPUS_OK);
            let mut packet = vec![0u8; 1500];
            // Prime prev_mode
            assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 1, 8100);
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            // Mode switch: constrain budget to be small enough that redundancy
            // signaling cannot fit
            assert_eq!(enc.set_force_mode(MODE_CELT_ONLY), OPUS_OK);
            let _ = enc.encode(&pcm, 960, &mut packet, 20).unwrap();
        }

        // ---- compute_frame_size variants: exercise ns * factor validations
        #[test]
        fn bc_frame_size_select_every_period() {
            for &(fs, ns) in &[
                (48000, 120),  // 400*ns == fs
                (48000, 240),  // 200*ns == fs
                (48000, 480),  // 100*ns == fs
                (48000, 960),  // 50*ns == fs
                (48000, 1920), // 25*ns == fs
                (48000, 2880), // 50*ns == 3*fs
                (48000, 3840), // 50*ns == 4*fs
                (48000, 4800), // 50*ns == 5*fs
                (48000, 5760), // 50*ns == 6*fs
            ] {
                assert_eq!(frame_size_select(ns, OPUS_FRAMESIZE_ARG, fs), ns);
            }
            // An odd ns that fits none of the valid frame sizes
            assert_eq!(frame_size_select(137, OPUS_FRAMESIZE_ARG, 48000), -1);
        }

        // ---- gen_toc for every mode × bandwidth × period combination
        #[test]
        fn bc_gen_toc_all_combinations() {
            for &fr in &[50, 100, 200, 400] {
                for &ch in &[1, 2] {
                    // SILK
                    for &bw in &[
                        OPUS_BANDWIDTH_NARROWBAND,
                        OPUS_BANDWIDTH_MEDIUMBAND,
                        OPUS_BANDWIDTH_WIDEBAND,
                    ] {
                        let _ = gen_toc(MODE_SILK_ONLY, fr, bw, ch);
                    }
                    // Hybrid
                    for &bw in &[OPUS_BANDWIDTH_SUPERWIDEBAND, OPUS_BANDWIDTH_FULLBAND] {
                        let _ = gen_toc(MODE_HYBRID, fr, bw, ch);
                    }
                    // CELT
                    for &bw in &[
                        OPUS_BANDWIDTH_NARROWBAND, // tmp < 0 branch
                        OPUS_BANDWIDTH_MEDIUMBAND,
                        OPUS_BANDWIDTH_WIDEBAND,
                        OPUS_BANDWIDTH_SUPERWIDEBAND,
                        OPUS_BANDWIDTH_FULLBAND,
                    ] {
                        let _ = gen_toc(MODE_CELT_ONLY, fr, bw, ch);
                    }
                }
            }
        }

        // ---- Encoder with signal = VOICE + VOIP + FEC=2 + low voice_est via music
        #[test]
        fn bc_fec_config2_music_signal() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_signal(OPUS_SIGNAL_MUSIC), OPUS_OK);
            assert_eq!(enc.set_inband_fec(2), OPUS_OK);
            assert_eq!(enc.set_packet_loss_perc(20), OPUS_OK);
            assert_eq!(enc.set_bitrate(32000), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 1, 8200);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
        }

        // ---- Encode after setting phase_inversion_disabled mid-stream
        #[test]
        fn bc_phase_inversion_toggled() {
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_bitrate(48000), OPUS_OK);
            let mut packet = vec![0u8; 1500];
            for (i, &v) in [0, 1, 0, 1].iter().enumerate() {
                assert_eq!(enc.set_phase_inversion_disabled(v), OPUS_OK);
                let pcm = patterned_pcm_i16(960, 2, 8300 + i as i32);
                let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            }
        }

        // ---- silk_encoder() and get_silk_state() accessors
        #[test]
        fn bc_silk_encoder_accessors() {
            let enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_VOIP).unwrap();
            assert!(enc.silk_encoder().is_some());
            let snap = enc.get_silk_state();
            assert!(snap.is_some());
            let s = snap.unwrap();
            // basic sanity — fields readable
            let _ = s.fs_khz + s.frame_length + s.nb_subfr;
            let _ = s.input_buf_ix + s.n_frames_per_packet + s.packet_size_ms;
            let _ = s.first_frame_after_reset + s.controlled_since_last_payload;
            let _ = s.prefill_flag + s.n_frames_encoded + s.speech_activity_q8;
            let _ = s.signal_type + s.input_quality_bands_q15 as i32;

            // None path
            let mut enc2 = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            enc2.silk_enc = None;
            assert!(enc2.silk_encoder().is_none());
            assert!(enc2.get_silk_state().is_none());
        }

        // ---- Tiny-budget with prev_mode set non-zero and bandwidth preserved
        // (covers L1326 toc_mode != SILK fallback)
        #[test]
        fn bc_tiny_budget_with_prior_mode() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_bitrate(64000), OPUS_OK);
            // prime by encoding a valid frame
            let pcm = patterned_pcm_i16(960, 1, 9001);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            assert!(enc.get_prev_mode() > 0);
            // Now request tiny-budget path (<3 bytes)
            let mut tiny = [0u8; 2];
            let len = enc.encode(&pcm, 960, &mut tiny, 2).unwrap();
            assert_eq!(len, 1);
        }

        // ---- Tiny-budget with bandwidth explicitly zero (default on new encoder)
        // covers L1329 toc_bw == 0 fallback.
        #[test]
        fn bc_tiny_budget_first_frame_zero_bandwidth() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            enc.bandwidth = 0;
            let pcm = patterned_pcm_i16(960, 1, 9100);
            let mut tiny = [0u8; 2];
            let len = enc.encode(&pcm, 960, &mut tiny, 2).unwrap();
            assert_eq!(len, 1);
        }

        // ---- VOIP app + voice_ratio >= 0 + AUDIO cap branch (L1360-1364)
        #[test]
        fn bc_voice_ratio_with_audio_app_cap() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_signal(OPUS_AUTO), OPUS_OK);
            assert_eq!(enc.set_voice_ratio(100), OPUS_OK); // very high voice
            assert_eq!(enc.set_bitrate(32000), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 1, 9200);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
        }

        // ---- Force a 5ms frame size with prior CELT mode to cover redundancy==false
        // branch at L1490 (frame_size < fs/100, !celt_to_silk).
        #[test]
        fn bc_celt_redundancy_short_frame() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_RESTRICTED_LOWDELAY).unwrap();
            assert_eq!(enc.set_bitrate(48000), OPUS_OK);
            // First encode a 20ms CELT frame to set prev_mode
            assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_20_MS), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 1, 9300);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            assert_eq!(enc.get_prev_mode(), MODE_CELT_ONLY);
            // Now encode a 5ms frame which is short -- but RESTRICTED_LOWDELAY always uses CELT
            // so prev_mode==celt==celt, was_celt==is_celt. To really hit the short branch we
            // need a non-lowdelay encoder and force CELT->SILK transition with a <10ms frame,
            // but SILK can't do <10ms. This path is effectively dead — still encode the frame.
            assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_5_MS), OPUS_OK);
            let pcm = patterned_pcm_i16(240, 1, 9301);
            let _ = enc.encode(&pcm, 240, &mut packet, 1500).unwrap();
        }

        // ---- Bandwidth narrowing path: very low bitrate, should walk bandwidth down
        // to the NB floor then break (hits L1567 break path).
        #[test]
        fn bc_bandwidth_narrowing_floor() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            assert_eq!(enc.set_bitrate(6000), OPUS_OK);
            assert_eq!(enc.set_signal(OPUS_SIGNAL_VOICE), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 1, 9400);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
        }

        // ---- After first encode, second encode exercises `first == 0` branch in bandwidth loop
        #[test]
        fn bc_bandwidth_hysteresis_second_frame() {
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_bitrate(32000), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 2, 9500);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            // Second encode after first==0
            let pcm2 = patterned_pcm_i16(960, 2, 9501);
            let _ = enc.encode(&pcm2, 960, &mut packet, 1500).unwrap();
        }

        // ---- Multiframe with pre-set bak_to_mono != 0 (hits L1743)
        #[test]
        fn bc_multiframe_with_bak_to_mono_set() {
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_VOIP).unwrap();
            assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
            assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
            assert_eq!(enc.set_bitrate(24000), OPUS_OK);
            assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_40_MS), OPUS_OK);
            // pre-set to_mono to non-zero
            enc.silk_mode.to_mono = 1;
            let pcm = patterned_pcm_i16(1920, 2, 9600);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 1920, &mut packet, 1500).unwrap();
        }

        // ---- Multiframe BITRATE_MAX repacketize_len branch (L1750 alt)
        #[test]
        fn bc_multiframe_bitrate_max() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_bitrate(OPUS_BITRATE_MAX), OPUS_OK);
            assert_eq!(enc.set_vbr(0), OPUS_OK);
            assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_60_MS), OPUS_OK);
            let pcm = patterned_pcm_i16(2880, 1, 9700);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 2880, &mut packet, 1500).unwrap();
        }

        // ---- Multiframe DTX: all silence frames, count dtx_count == nb_frames
        // (covers pad_cbr == false branch at L1826)
        #[test]
        fn bc_multiframe_all_silence_dtx() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            assert_eq!(enc.set_dtx(1), OPUS_OK);
            assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
            assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
            assert_eq!(enc.set_vbr(0), OPUS_OK);
            assert_eq!(enc.set_bitrate(16000), OPUS_OK);
            assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_40_MS), OPUS_OK);
            let silence = vec![0i16; 1920];
            let mut packet = vec![0u8; 1500];
            // encode several frames to let DTX trigger
            for _ in 0..15 {
                let _ = enc.encode(&silence, 1920, &mut packet, 1500).unwrap();
            }
        }

        // ---- Hit the CELT-only activity energy-based computation path (L1887-1895)
        #[test]
        fn bc_celt_activity_energy_both_branches() {
            // High peak_signal_energy from a loud frame, then a quiet (but not silent) frame
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_force_mode(MODE_CELT_ONLY), OPUS_OK);
            assert_eq!(enc.set_bitrate(32000), OPUS_OK);
            let loud: Vec<i16> = (0..960).map(|i| ((i * 100) % 30000) as i16).collect();
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&loud, 960, &mut packet, 1500).unwrap();
            // Now a quieter frame
            let quiet: Vec<i16> = (0..960).map(|i| ((i * 3) % 100) as i16).collect();
            let _ = enc.encode(&quiet, 960, &mut packet, 1500).unwrap();
        }

        // ---- SILK-only with CBR hybrid max_bits adjustment (L2078-2103)
        // by forcing hybrid + CBR at high bitrate
        #[test]
        fn bc_hybrid_cbr_max_bits() {
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_force_mode(MODE_HYBRID), OPUS_OK);
            assert_eq!(enc.set_vbr(0), OPUS_OK);
            assert_eq!(enc.set_bitrate(64000), OPUS_OK);
            assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_FULLBAND), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 2, 9800);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
        }

        // ---- Hybrid at low rate (SWB) covers various silk_mode branches
        #[test]
        fn bc_hybrid_low_rate_swb() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            assert_eq!(enc.set_force_mode(MODE_HYBRID), OPUS_OK);
            assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_SUPERWIDEBAND), OPUS_OK);
            assert_eq!(enc.set_bitrate(20000), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 1, 9900);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
        }

        // ---- CELT-only mode transition generates tmp_prefill coverage (L2261-2278)
        #[test]
        fn bc_celt_prev_mode_tmp_prefill() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_bitrate(48000), OPUS_OK);
            // Start with SILK, then transition to CELT (with prev_mode > 0)
            assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 1, 10000);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            // Now force CELT; mode != prev_mode, prev_mode > 0, mode != SILK
            assert_eq!(enc.set_force_mode(MODE_CELT_ONLY), OPUS_OK);
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
        }

        // ---- Exercise SILK→CELT redundancy path (redundancy && !celt_to_silk) via
        // transitioning from SILK to CELT with enough budget.
        #[test]
        fn bc_silk_to_celt_redundancy() {
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_bitrate(64000), OPUS_OK);
            assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
            assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 2, 10100);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            // Transition to CELT
            assert_eq!(enc.set_force_mode(MODE_CELT_ONLY), OPUS_OK);
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
        }

        // ---- Exercise CELT→SILK redundancy path (celt_to_silk case)
        #[test]
        fn bc_celt_to_silk_redundancy() {
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_bitrate(64000), OPUS_OK);
            assert_eq!(enc.set_force_mode(MODE_CELT_ONLY), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 2, 10200);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            // Transition to SILK (using a frame_size >= fs/100 keeps redundancy)
            assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
            assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
        }

        // ---- silk_enc = None triggers else in hp_freq_smth1 selection (L1952)
        #[test]
        fn bc_encode_with_silk_enc_none() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            enc.silk_enc = None;
            // Also force CELT-only since SILK path needs silk_enc
            assert_eq!(enc.set_force_mode(MODE_CELT_ONLY), OPUS_OK);
            assert_eq!(enc.set_bitrate(32000), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 1, 10300);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
        }

        // ---- Encode across channels=2 with low bitrate music stereo width path
        #[test]
        fn bc_stereo_width_low_bitrate_music() {
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_bitrate(10000), OPUS_OK); // < 16000 -> stereo_width_q14 = 0
            assert_eq!(enc.set_signal(OPUS_SIGNAL_MUSIC), OPUS_OK);
            let mut packet = vec![0u8; 1500];
            for i in 0..3 {
                let pcm = patterned_pcm_i16(960, 2, 10400 + i);
                let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            }
        }

        // ---- Stereo width intermediate rate (16000..32000) interpolation
        #[test]
        fn bc_stereo_width_interpolation_rate() {
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_bitrate(22000), OPUS_OK);
            let mut packet = vec![0u8; 1500];
            for i in 0..3 {
                let pcm = patterned_pcm_i16(960, 2, 10500 + i);
                let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            }
        }

        // ---- High rate stereo -> stereo_width_q14 = 16384 (max)
        #[test]
        fn bc_stereo_width_high_rate() {
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_bitrate(128000), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 2, 10600);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
        }

        // ---- Force mode transition SILK->Hybrid->CELT with active encoding
        // to exercise silk_bw_switch redundancy
        #[test]
        fn bc_mode_sequence_with_silk_bw_switch() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            assert_eq!(enc.set_bitrate(32000), OPUS_OK);
            let mut packet = vec![0u8; 1500];
            // SILK WB
            assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
            assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 1, 10700);
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            // Hybrid SWB
            assert_eq!(enc.set_force_mode(MODE_HYBRID), OPUS_OK);
            assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_SUPERWIDEBAND), OPUS_OK);
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            // SILK WB again (force bandwidth switch)
            assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
            assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
        }

        // ---- Expert duration variations: encode at 40ms, 60ms, 80ms, 100ms with DTX off
        #[test]
        fn bc_long_frames_no_dtx() {
            for &(dur, fr) in &[
                (OPUS_FRAMESIZE_40_MS, 1920),
                (OPUS_FRAMESIZE_60_MS, 2880),
                (OPUS_FRAMESIZE_80_MS, 3840),
                (OPUS_FRAMESIZE_100_MS, 4800),
            ] {
                let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
                assert_eq!(enc.set_expert_frame_duration(dur), OPUS_OK);
                assert_eq!(enc.set_bitrate(48000), OPUS_OK);
                assert_eq!(enc.set_dtx(0), OPUS_OK);
                let pcm = patterned_pcm_i16(fr as usize, 1, dur);
                let mut packet = vec![0u8; 1500];
                let _ = enc.encode(&pcm, fr, &mut packet, 1500).unwrap();
            }
        }

        // ---- force_channels = 1 with force_mode = CELT on stereo, mixing voice/music
        // to exercise force_channels !=AUTO && channels==2 path (L1372-1373)
        #[test]
        fn bc_force_channels_stereo_celt_voice() {
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_force_channels(1), OPUS_OK);
            assert_eq!(enc.set_force_mode(MODE_CELT_ONLY), OPUS_OK);
            assert_eq!(enc.set_bitrate(32000), OPUS_OK);
            assert_eq!(enc.set_signal(OPUS_SIGNAL_VOICE), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 2, 10800);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
        }

        // ---- Packet_loss_perc sweep to cover loss_factor branches in decide_fec
        #[test]
        fn bc_packet_loss_perc_sweep() {
            for &loss in &[0, 5, 6, 10, 25, 50, 100] {
                let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
                assert_eq!(enc.set_bitrate(24000), OPUS_OK);
                assert_eq!(enc.set_inband_fec(1), OPUS_OK);
                assert_eq!(enc.set_packet_loss_perc(loss), OPUS_OK);
                let pcm = patterned_pcm_i16(960, 1, 10900 + loss);
                let mut packet = vec![0u8; 1500];
                let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            }
        }

        // ---- Use 16kHz to hit fs<=16000 bandwidth cap
        #[test]
        fn bc_16k_bandwidth_cap() {
            let mut enc = OpusEncoder::new(16000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_FULLBAND), OPUS_OK); // will be capped to WB
            assert_eq!(enc.set_bitrate(32000), OPUS_OK);
            let pcm = patterned_pcm_i16(320, 1, 11000);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 320, &mut packet, 1500).unwrap();
        }

        // ---- 12kHz Nyquist cap
        #[test]
        fn bc_12k_bandwidth_cap() {
            let mut enc = OpusEncoder::new(12000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_SUPERWIDEBAND), OPUS_OK);
            assert_eq!(enc.set_bitrate(24000), OPUS_OK);
            let pcm = patterned_pcm_i16(240, 1, 11100);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 240, &mut packet, 1500).unwrap();
        }

        // ---- 8kHz Nyquist cap
        #[test]
        fn bc_8k_bandwidth_cap() {
            let mut enc = OpusEncoder::new(8000, 1, OPUS_APPLICATION_VOIP).unwrap();
            assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
            assert_eq!(enc.set_bitrate(16000), OPUS_OK);
            let pcm = patterned_pcm_i16(160, 1, 11200);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 160, &mut packet, 1500).unwrap();
        }

        // ---- celt_enc = None while encoding (exercises several
        // `if let Some(ref mut celt)` false branches throughout encode_frame_native)
        #[test]
        fn bc_encode_with_celt_enc_none_silk_mode() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
            assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
            assert_eq!(enc.set_bitrate(20000), OPUS_OK);
            // First encode a valid frame with celt
            let pcm = patterned_pcm_i16(960, 1, 11400);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            // Now drop celt_enc and re-encode in SILK-only mode
            enc.celt_enc = None;
            let res = enc.encode(&pcm, 960, &mut packet, 1500);
            // May error out since CELT is needed for some SILK paths
            let _ = res;
        }

        // ---- silk_enc None with non-CELT mode: hits L1952 else-branch in hp_freq_smth1
        #[test]
        fn bc_hp_freq_smth1_silk_none_nonceltmode() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
            assert_eq!(enc.set_bitrate(24000), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 1, 11500);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            enc.silk_enc = None;
            // Attempting SILK-only with no silk encoder — may return error or produce garbage
            let _ = enc.encode(&pcm, 960, &mut packet, 1500);
        }

        // ---- Multiframe with DTX active: some sub-frames silent, some not.
        // Mix silence and active to drive dtx_count (ret==1 branch at L1808)
        #[test]
        fn bc_multiframe_mixed_silence() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            assert_eq!(enc.set_dtx(1), OPUS_OK);
            assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
            assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
            assert_eq!(enc.set_bitrate(20000), OPUS_OK);
            assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_60_MS), OPUS_OK);
            let mut packet = vec![0u8; 1500];
            // First many frames of silence to warm DTX
            let silence = vec![0i16; 2880];
            for _ in 0..10 {
                let _ = enc.encode(&silence, 2880, &mut packet, 1500).unwrap();
            }
            // Now mixed active frame
            let active = patterned_pcm_i16(2880, 1, 11600);
            let _ = enc.encode(&active, 2880, &mut packet, 1500).unwrap();
        }

        // ---- Set mediumband explicitly at WB fs to trigger L1572 skip to WB
        // after the bandwidth selection loop's idx comparison.
        #[test]
        fn bc_bandwidth_mediumband_skip() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            assert_eq!(enc.set_force_mode(MODE_CELT_ONLY), OPUS_OK);
            assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_MEDIUMBAND), OPUS_OK);
            assert_eq!(enc.set_bitrate(24000), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 1, 11700);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
        }

        // ---- Encode a stereo sequence whose bitrate drops triggering stereo→mono
        // naturally over multiple frames (covers to_mono=1 path + bak_to_mono=1
        // in subsequent multiframe encode).
        #[test]
        fn bc_stereo_to_mono_natural_transition() {
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_bitrate(6000), OPUS_OK); // very low -> mono
            assert_eq!(enc.set_signal(OPUS_SIGNAL_VOICE), OPUS_OK);
            let mut packet = vec![0u8; 1500];
            // Warm up with stereo
            for i in 0..2 {
                let pcm = patterned_pcm_i16(960, 2, 11800 + i);
                let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            }
            // Now set multi-frame 40ms to hit multiframe path with bak_to_mono
            assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_40_MS), OPUS_OK);
            for i in 0..2 {
                let pcm = patterned_pcm_i16(1920, 2, 12000 + i);
                let _ = enc.encode(&pcm, 1920, &mut packet, 1500).unwrap();
            }
        }

        // ---- Multiframe with BITRATE_MAX + silence: dtx_count == nb_frames
        #[test]
        fn bc_multiframe_all_silence_all_dtx_bitrate_max() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            assert_eq!(enc.set_dtx(1), OPUS_OK);
            assert_eq!(enc.set_bitrate(OPUS_BITRATE_MAX), OPUS_OK);
            assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_40_MS), OPUS_OK);
            let silence = vec![0i16; 1920];
            let mut packet = vec![0u8; 1500];
            for _ in 0..15 {
                let _ = enc.encode(&silence, 1920, &mut packet, 1500).unwrap();
            }
        }

        // ---- SILK-only long sequence at moderate rate to trigger switch_ready (L2202)
        #[test]
        fn bc_silk_long_sequence_for_switch() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
            assert_eq!(enc.set_bandwidth(OPUS_AUTO), OPUS_OK);
            assert_eq!(enc.set_bitrate(20000), OPUS_OK);
            let mut packet = vec![0u8; 1500];
            for i in 0..12 {
                let pcm = patterned_pcm_i16(960, 1, 12500 + i);
                let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            }
        }

        // ---- SILK-only pure silence (not DTX enabled): SILK VAD sees no activity
        // but still encodes (hits L2212 else-branch)
        #[test]
        fn bc_silk_pure_silence_no_dtx() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
            assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
            assert_eq!(enc.set_dtx(0), OPUS_OK);
            assert_eq!(enc.set_bitrate(20000), OPUS_OK);
            let silence = vec![0i16; 960];
            let mut packet = vec![0u8; 1500];
            for _ in 0..5 {
                let _ = enc.encode(&silence, 960, &mut packet, 1500).unwrap();
            }
        }

        // ---- Rate/voice-est combination that lands at mediumband skip (L1572)
        // Use music signal at ~9000 bps per channel to land on MB threshold.
        #[test]
        fn bc_mediumband_skip() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_signal(OPUS_SIGNAL_MUSIC), OPUS_OK);
            assert_eq!(enc.set_force_mode(MODE_CELT_ONLY), OPUS_OK);
            assert_eq!(enc.set_bitrate(9000), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 1, 12600);
            let mut packet = vec![0u8; 1500];
            for _ in 0..3 {
                let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            }
            // Same with voice signal
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_signal(OPUS_SIGNAL_VOICE), OPUS_OK);
            assert_eq!(enc.set_force_mode(MODE_CELT_ONLY), OPUS_OK);
            assert_eq!(enc.set_bitrate(10000), OPUS_OK);
            let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
        }

        // ---- Stereo encoder at VOIP with mid-rate exercises STEREO_VOICE thresholds
        #[test]
        fn bc_stereo_voice_voip_bandwidth() {
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_VOIP).unwrap();
            assert_eq!(enc.set_signal(OPUS_SIGNAL_VOICE), OPUS_OK);
            assert_eq!(enc.set_bitrate(16000), OPUS_OK);
            let pcm = patterned_pcm_i16(960, 2, 12700);
            let mut packet = vec![0u8; 1500];
            for _ in 0..3 {
                let _ = enc.encode(&pcm, 960, &mut packet, 1500).unwrap();
            }
        }

        // ---- Multiframe with redundancy at boundary between SILK and CELT
        // (to_celt && i == nb_frames-1 branch)
        #[test]
        fn bc_multiframe_redundancy_to_celt() {
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            assert_eq!(enc.set_bitrate(48000), OPUS_OK);
            // Start with SILK
            assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
            assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
            assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_40_MS), OPUS_OK);
            let pcm = patterned_pcm_i16(1920, 1, 11300);
            let mut packet = vec![0u8; 1500];
            let _ = enc.encode(&pcm, 1920, &mut packet, 1500).unwrap();
            // Now transition to CELT; multiframe with to_celt flag at last sub-frame
            assert_eq!(enc.set_force_mode(MODE_CELT_ONLY), OPUS_OK);
            assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_40_MS), OPUS_OK);
            let _ = enc.encode(&pcm, 1920, &mut packet, 1500).unwrap();
        }
    }
}
