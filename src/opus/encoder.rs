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
/// Matches C `user_bitrate_to_bitrate`.
fn user_bitrate_to_bitrate(
    user_bitrate_bps: i32,
    channels: i32,
    fs: i32,
    frame_size: i32,
    max_data_bytes: i32,
) -> i32 {
    if user_bitrate_bps == OPUS_AUTO {
        60 * fs / frame_size + fs * channels
    } else if user_bitrate_bps == OPUS_BITRATE_MAX {
        max_data_bytes * 8 * fs / frame_size
    } else {
        user_bitrate_bps
    }
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
        // Cap by max_data_bytes
        let max_rate = bits_to_bitrate(max_data_bytes * 8, self.fs, frame_size);
        bitrate_bps = imin(bitrate_bps, max_rate);
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
        self.silk_mode.use_dtx = if self.use_dtx != 0 && !is_silence { 1 } else { 0 };

        // --- Mode selection ---
        let mut mode: i32;
        if self.application == OPUS_APPLICATION_RESTRICTED_LOWDELAY {
            mode = MODE_CELT_ONLY;
        } else if self.user_forced_mode == OPUS_AUTO {
            // Interpolate threshold between voice and music
            let mode_voice = MODE_THRESHOLDS[0][0]
                + (stereo_width as i64 * (MODE_THRESHOLDS[1][0] - MODE_THRESHOLDS[0][0]) as i64
                    / Q15ONE as i64) as i32;
            // C: both MULT16_32_Q15 terms use mode_thresholds[1][1]
            let mode_music = MODE_THRESHOLDS[1][1];
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
        _redundancy: bool,
        _celt_to_silk: bool,
        _prefill: i32,
        _equiv_rate: i32,
        _to_celt: bool,
    ) -> Result<i32, i32> {
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

        let bytes_per_frame = imin(1276, imax(1, max_data_bytes / nb_frames));

        // Encode each sub-frame
        let mut sub_packets: Vec<Vec<u8>> = Vec::with_capacity(nb_frames as usize);
        for i in 0..nb_frames {
            let offset = (i * enc_frame_size * self.channels) as usize;
            let pcm_frame = &pcm[offset..];
            let mut frame_buf = vec![0u8; bytes_per_frame as usize];

            self.nonfinal_frame = if i < nb_frames - 1 { 1 } else { 0 };

            let ret = self.encode_native(
                pcm_frame,
                enc_frame_size,
                &mut frame_buf,
                bytes_per_frame,
                lsb_depth,
            )?;
            frame_buf.truncate(ret as usize);
            sub_packets.push(frame_buf);
        }
        self.nonfinal_frame = 0;

        // Repacketize
        let mut rp = OpusRepacketizer::new();
        for pkt in &sub_packets {
            let ret = rp.cat(pkt, pkt.len() as i32);
            if ret != OPUS_OK {
                return Err(ret);
            }
        }

        let ret = rp.out(data, orig_max_data_bytes);
        if ret < 0 {
            return Err(ret);
        }

        // CBR padding
        if self.use_vbr == 0 && ret < orig_max_data_bytes {
            let pad_ret = opus_packet_pad(data, ret, orig_max_data_bytes);
            if pad_ret == OPUS_OK {
                return Ok(orig_max_data_bytes);
            }
        }

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
        self.variable_hp_smth2_q15 += ((hp_diff as i64 * VARIABLE_HP_SMTH_COEF2 as i64) >> 16) as i32;
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
                    // HB gain attenuation
                    // C: HB_gain = Q15ONE - SHR32(celt_exp2(-celt_rate * QCONST16(1.f/1024, 10)), 1)
                    // QCONST16(1.f/1024, 10) = round(1/1024 * 2^10) = 1, so argument is -celt_rate (Q10)
                    let celt_rate = total_bit_rate - self.silk_mode.bit_rate;
                    if celt_rate > 0 {
                        hb_gain = Q15ONE - shr32(celt_exp2(-celt_rate), 1);
                        hb_gain = imax(0, hb_gain);
                    }
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
                // C: resets encoder, sets payloadSize_ms=10, complexity=0
                if prefill != 0 {
                    if let Some(ref mut silk) = self.silk_enc {
                        let db = self.delay_buffer.clone();
                        let db_samples = (self.encoder_buffer * self.channels) as usize;
                        let mut prefill_control = self.silk_mode.clone();
                        // C: encControl->payloadSize_ms = 10; encControl->complexity = 0;
                        prefill_control.payload_size_ms = 10;
                        prefill_control.complexity = 0;
                        let mut zero = 0i32;
                        let prefill_pcm = if db_samples > 0 {
                            &db[..db_samples]
                        } else {
                            &[]
                        };
                        silk_encode(
                            silk,
                            &mut prefill_control,
                            prefill_pcm,
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
                    if self.silk_mode.switch_ready != 0 && self.nonfinal_frame == 0 { 1 } else { 0 };

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
            }

            // --- Update delay buffer BEFORE CELT encoding ---
            // C copies from pcm_buf (dc_reject'd), not raw pcm
            self.update_delay_buffer_from_pcm_buf(&pcm_buf, frame_size, total_buffer);

            // --- HB gain fade ---
            if (self.prev_hb_gain < Q15ONE || hb_gain < Q15ONE) && self.mode != MODE_SILK_ONLY {
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
                    if self.mode != self.prev_mode && self.prev_mode > 0 {
                        celt.ctl(CeltEncoderCtl::ResetState);
                        let mut prefill_buf = [0u8; 2];
                        let n4 = (self.fs / 400) as usize;
                        let prefill_pcm = if pcm_buf.len() >= n4 * self.channels as usize {
                            &pcm_buf[..n4 * self.channels as usize]
                        } else {
                            &pcm_buf
                        };
                        celt_encode_with_ec(
                            celt,
                            prefill_pcm,
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
        if self.mode != MODE_SILK_ONLY {
            // For CELT/hybrid modes, check if encoder busted the budget
            // (shouldn't happen normally, but matches C safety check)
        }
        // Strip trailing zeros (SILK-only, no redundancy) (C: lines 2590-2598)
        if self.mode == MODE_SILK_ONLY && !redundancy {
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
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
        // Invalid: requested size > input
        assert_eq!(frame_size_select(480, OPUS_FRAMESIZE_20_MS, 48000), -1);
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
        // Basic: 64kbps, mono, 50fps, VBR, complexity 10
        let equiv = compute_equiv_rate(64000, 1, 50, 1, 0, 10, 0);
        assert!(equiv > 0);
        assert!(equiv <= 64000);
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
}
