//! CELT encoder — transform-domain codec within Opus.
//!
//! Port of `celt_encoder.c` from xiph/opus. Fixed-point path only (non-QEXT).
//! Produces bit-exact output matching the C reference when compiled with
//! `FIXED_POINT` and `OPUS_FAST_INT64`.
//!
//! The encoder converts PCM audio into a compressed bitstream using an
//! MDCT-based approach with perceptual optimization. Key stages:
//! pre-emphasis → pitch pre-filter → MDCT → energy quantization →
//! spectral quantization (PVQ) → VBR rate control → bitstream packing.

use super::bands::{
    SPREAD_AGGRESSIVE, SPREAD_NONE, SPREAD_NORMAL, compute_band_energies, haar1,
    hysteresis_decision, normalise_bands, quant_all_bands, spreading_decision,
};
use super::fft;
use super::fft::{KissFftCpx, KissFftState, opus_fft_impl};
use super::math_ops::*;
use super::modes::{
    CELTMode, MODE_48000_960_120, NB_EBANDS, bitrate_to_bits, init_caps, resampling_factor,
};
use super::pitch::{pitch_downsample, pitch_search, remove_doubling};
use super::quant_bands::EMEANS;
use super::quant_bands::{amp2log2, quant_coarse_energy, quant_energy_finalise, quant_fine_energy};
use super::range_coder::RangeEncoder;
use super::rate::{
    BITRES, SPREAD_ICDF, TAPSET_ICDF, TF_SELECT_TABLE, TRIM_ICDF, clt_compute_allocation,
};
use super::vq::celt_inner_prod_norm_shift;
use crate::types::*;

// ===========================================================================
// Constants
// ===========================================================================

/// Maximum pitch period for the comb filter.
pub const COMBFILTER_MAXPERIOD: i32 = 1024;

/// Minimum pitch period for the comb filter.
pub const COMBFILTER_MINPERIOD: i32 = 15;

/// Maximum standard Opus packet bytes.
const MAX_PACKET_BYTES: i32 = 1275;

/// Silence/reset energy level in log2 domain (≈ -168 dB).
const GCONST_NEG28: i32 = -28 << DB_SHIFT;

/// Convert floating-point constant to Q(DB_SHIFT) fixed-point.
/// Matches C `GCONST(x)` = `(celt_glog)(0.5 + x * (1 << DB_SHIFT))`.
const fn gconst(x: f64) -> i32 {
    qconst32(x, DB_SHIFT as u32)
}

/// Opus error codes matching the C reference.
pub const OPUS_OK: i32 = 0;
pub const OPUS_BAD_ARG: i32 = -1;
pub const OPUS_INTERNAL_ERROR: i32 = -3;
/// Sentinel "use the maximum bitrate" for `SetBitrate`. Matches C
/// `OPUS_BITRATE_MAX` in `opus_defines.h`. Numerically coincides with
/// `OPUS_BAD_ARG` because both live in disjoint value spaces in the C
/// reference: `OPUS_BITRATE_MAX` is a parameter value (compared against
/// `bitrate`), `OPUS_BAD_ARG` is a return code. A previous definition of
/// `-1000` silently rejected `SetBitrate(OPUS_BITRATE_MAX)` from
/// `OpusEncoder`, leaving a stale bitrate on the CELT encoder during the
/// 5 ms SILK→CELT redundancy encode and causing `nb_compressed_bytes` to
/// be CBR-shrunk from `redundancy_bytes` down to a much smaller value,
/// which diverged `redundant_rng` from the decoder.
pub const OPUS_BITRATE_MAX: i32 = -1;

/// Comb filter tapset gains: `gains[tapset][tap]` in Q15.
/// Matches C `gains[3][3]` in `comb_filter()`.
static COMB_GAINS: [[i32; 3]; 3] = [
    [
        qconst16(0.3066406250, 15),
        qconst16(0.2170410156, 15),
        qconst16(0.1296386719, 15),
    ],
    [qconst16(0.4638671875, 15), qconst16(0.2680664062, 15), 0],
    [qconst16(0.7998046875, 15), qconst16(0.1000976562, 15), 0],
];

/// Precomputed `6*64/x` table for transient analysis (128 entries).
/// Trained on real data to minimize average error.
/// Must match C reference exactly (celt_encoder.c:287-296).
static INV_TABLE: [u8; 128] = [
    255, 255, 156, 110, 86, 70, 59, 51, 45, 40, 37, 33, 31, 28, 26, 25, 23, 22, 21, 20, 19, 18, 17,
    16, 16, 15, 15, 14, 13, 13, 12, 12, 12, 12, 11, 11, 11, 10, 10, 10, 9, 9, 9, 9, 9, 9, 8, 8, 8,
    8, 8, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2,
];

/// Intensity stereo thresholds (in kbps), 21 entries.
static INTENSITY_THRESHOLDS: [i32; 21] = [
    1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 36, 44, 50, 56, 62, 67, 72, 79, 88, 106, 134,
];

/// Intensity stereo hysteresis values, 21 entries.
static INTENSITY_HISTERESIS: [i32; 21] = [
    1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 5, 6, 8, 8,
];

// ===========================================================================
// Types
// ===========================================================================

/// Number of leak-boost bands carried in `AnalysisInfo`. Matches the C
/// `#define LEAK_BANDS 19` in `celt.h:63`.
pub const LEAK_BANDS: usize = 19;

/// External audio analysis data from the Opus layer.
///
/// In the full Opus encoder, this carries machine-learning analysis results
/// (tonality, activity, bandwidth, etc.). For standalone CELT encoding,
/// this is zeroed / invalid.
///
/// Layout mirrors C `AnalysisInfo` in `celt.h:65-79` — including the
/// `leak_boost[19]` Q6 bytes consumed by `dynalloc_analysis`.
#[derive(Clone, Debug)]
pub struct AnalysisInfo {
    pub valid: i32,
    pub tonality: f32,
    pub tonality_slope: f32,
    pub noisiness: f32,
    pub activity: f32,
    pub music_prob: f32,
    pub music_prob_min: f32,
    pub music_prob_max: f32,
    pub bandwidth: i32,
    pub activity_probability: f32,
    pub max_pitch_ratio: f32,
    pub leak_boost: [u8; LEAK_BANDS],
}

impl Default for AnalysisInfo {
    fn default() -> Self {
        Self {
            valid: 0,
            tonality: 0.0,
            tonality_slope: 0.0,
            noisiness: 0.0,
            activity: 0.0,
            music_prob: 0.0,
            music_prob_min: 0.0,
            music_prob_max: 0.0,
            bandwidth: 0,
            activity_probability: 0.0,
            max_pitch_ratio: 0.0,
            leak_boost: [0; LEAK_BANDS],
        }
    }
}

/// SILK codec info for hybrid mode.
///
/// When Opus runs in hybrid mode, the SILK layer passes information
/// to the CELT encoder about its internal state.
#[derive(Clone, Debug, Default)]
pub struct SILKInfo {
    pub signal_type: i32,
    pub offset: i32,
}

/// CELT encoder control requests.
///
/// Replaces the C variadic `opus_custom_encoder_ctl()` with a type-safe enum.
pub enum CeltEncoderCtl {
    SetComplexity(i32),
    SetStartBand(i32),
    SetEndBand(i32),
    SetPrediction(i32),
    SetPacketLossPerc(i32),
    SetVbrConstraint(i32),
    SetVbr(i32),
    SetBitrate(i32),
    SetChannels(i32),
    SetLsbDepth(i32),
    GetLsbDepth,
    SetPhaseInversionDisabled(i32),
    GetPhaseInversionDisabled,
    ResetState,
    SetAnalysis(AnalysisInfo),
    SetSilkInfo(SILKInfo),
    SetSignalling(i32),
    SetLfe(i32),
    SetEnergyMask, // pointer-based in C, handled differently in Rust
    GetFinalRange,
    SetInputClipping(i32),
}

/// CELT encoder state.
///
/// Matches the C `OpusCustomEncoder` struct. Variable-length trailing arrays
/// are replaced with explicit `Vec` fields.
pub struct CeltEncoder {
    // -- Configuration fields (persist across reset) --
    pub mode: &'static CELTMode,
    pub channels: i32,
    pub stream_channels: i32,
    pub force_intra: i32,
    pub clip: i32,
    pub disable_pf: i32,
    pub complexity: i32,
    pub upsample: i32,
    pub start: i32,
    pub end: i32,
    pub bitrate: i32,
    pub vbr: i32,
    pub signalling: i32,
    pub constrained_vbr: i32,
    pub loss_rate: i32,
    pub lsb_depth: i32,
    pub lfe: i32,
    pub disable_inv: i32,

    // -- Running state fields (cleared on reset) --
    // TODO(stage-6-tier-1): `delayed_intra`, `tonal_average`, and `spec_avg`
    // below are the three CELT state fields flagged by Stage 6.4c as
    // diverging simultaneously at frame 90 on music vectors (downstream of
    // bandLogE/MDCT). A future tier-1 campaign should dump `bandE`/
    // `bandLogE`/MDCT output `X` at frame 89 exit and frame 90 entry; see
    // `wrk_journals/2026.04.19 - JRN - stage10-audit-stream-B.md` (G-6.1).
    pub rng: u32,
    pub spread_decision: i32,
    pub delayed_intra: i32,
    pub tonal_average: i32,
    pub last_coded_bands: i32,
    pub hf_average: i32,
    pub tapset_decision: i32,
    pub prefilter_period: i32,
    pub prefilter_gain: i32,
    pub prefilter_tapset: i32,
    pub consec_transient: i32,
    pub analysis: AnalysisInfo,
    pub silk_info: SILKInfo,
    pub preemph_mem_e: [i32; 2],
    pub preemph_mem_d: [i32; 2],
    pub vbr_reservoir: i32,
    pub vbr_drift: i32,
    pub vbr_offset: i32,
    pub vbr_count: i32,
    pub overlap_max: i32,
    pub stereo_saving: i32,
    pub intensity: i32,
    pub energy_mask: Option<Vec<i32>>,
    pub spec_avg: i32,

    // -- Variable-length arrays --
    pub in_mem: Vec<i32>,
    pub prefilter_mem: Vec<i32>,
    pub old_band_e: Vec<i32>,
    pub old_log_e: Vec<i32>,
    pub old_log_e2: Vec<i32>,
    pub energy_error: Vec<i32>,
}

// ===========================================================================
// Comb filter (from celt.c)
// ===========================================================================

/// Apply the pitch comb filter for pre-filtering.
///
/// Matches C `comb_filter()` from `celt.c`. Transitions from (T0,g0,tapset0)
/// to (T1,g1,tapset1) over the overlap region using the MDCT window.
///
/// `y` and `x` can alias (same buffer). The caller must ensure `x` has
/// enough history before index 0 (at least `max(T0,T1)+2` samples).
fn comb_filter(
    y: &mut [i32],
    x: &[i32],
    x_offset: usize, // where index 0 of x is in the backing buffer
    t0: i32,
    t1: i32,
    n: i32,
    g0: i32,
    g1: i32,
    tapset0: i32,
    tapset1: i32,
    window: &[i16],
    overlap: i32,
) {
    let t0 = t0.max(COMBFILTER_MINPERIOD);
    let t1 = t1.max(COMBFILTER_MINPERIOD);

    if g0 == 0 && g1 == 0 {
        // No filtering needed - just copy if x != y
        for i in 0..n as usize {
            y[i] = x[x_offset + i];
        }
        return;
    }

    let g00 = mult16_16_p15(g0, COMB_GAINS[tapset0 as usize][0]);
    let g01 = mult16_16_p15(g0, COMB_GAINS[tapset0 as usize][1]);
    let g02 = mult16_16_p15(g0, COMB_GAINS[tapset0 as usize][2]);
    let g10 = mult16_16_p15(g1, COMB_GAINS[tapset1 as usize][0]);
    let g11 = mult16_16_p15(g1, COMB_GAINS[tapset1 as usize][1]);
    let g12 = mult16_16_p15(g1, COMB_GAINS[tapset1 as usize][2]);

    // Check if the filter changed; if not, skip overlap
    let ov = if g0 == g1 && t0 == t1 && tapset0 == tapset1 {
        0
    } else {
        overlap
    };

    let t1u = t1 as usize;

    // Overlap region: crossfade between old and new filter
    for i in 0..ov as usize {
        let xi = x_offset + i;
        let w = window[i] as i32;
        let f = mult16_16_q15(w, w);
        let one_minus_f = Q15_ONE - f;

        let mut val = x[xi];
        // Old filter contribution (weighted by 1-f)
        val += mult16_32_q15(mult16_16_q15(one_minus_f, g00), x[xi - t0 as usize]);
        val += mult16_32_q15(
            mult16_16_q15(one_minus_f, g01),
            x[xi - t0 as usize + 1] + x[xi - t0 as usize - 1],
        );
        val += mult16_32_q15(
            mult16_16_q15(one_minus_f, g02),
            x[xi - t0 as usize + 2] + x[xi - t0 as usize - 2],
        );
        // New filter contribution (weighted by f)
        val += mult16_32_q15(mult16_16_q15(f, g10), x[xi - t1u]);
        val += mult16_32_q15(mult16_16_q15(f, g11), x[xi - t1u + 1] + x[xi - t1u - 1]);
        val += mult16_32_q15(mult16_16_q15(f, g12), x[xi - t1u + 2] + x[xi - t1u - 2]);
        // Fixed-point bias
        val -= 3;
        y[i] = saturate(val, SIG_SAT);
    }

    if g1 == 0 {
        for i in ov as usize..n as usize {
            y[i] = x[x_offset + i];
        }
        return;
    }

    // Constant filter region
    for i in ov as usize..n as usize {
        let xi = x_offset + i;
        let mut val = x[xi];
        val += mult16_32_q15(g10, x[xi - t1u]);
        val += mult16_32_q15(g11, x[xi - t1u + 1] + x[xi - t1u - 1]);
        val += mult16_32_q15(g12, x[xi - t1u + 2] + x[xi - t1u - 2]);
        // Fixed-point bias
        val -= 1;
        y[i] = saturate(val, SIG_SAT);
    }
}

// ===========================================================================
// MDCT forward (from mdct.c)
// ===========================================================================

/// Compute forward MDCT using the FFT.
///
/// Matches C `clt_mdct_forward_c()` from `mdct.c`. Implements the MDCT as:
/// 1. Window, shuffle, and fold the input into N/2 real values
/// 2. Pre-rotate by twiddle factors to form N/4 complex values
/// 3. N/4-point complex FFT
/// 4. Post-rotate to produce the final MDCT output
///
/// The twiddle factors come from a pre-computed lookup table (`trig`).
pub(crate) fn clt_mdct_forward(
    input: &[i32],
    output: &mut [i32],
    window: &[i16],
    overlap: i32,
    _shift: i32,
    stride: i32,
    fft_state: &KissFftState,
    trig: &[i16],
) {
    let n = fft_state.nfft as usize * 4;
    let n2 = n >> 1;
    let n4 = n >> 2;
    let ov = overlap as usize;

    let scale = fft_state.scale;
    let scale_shift = fft_state.scale_shift as i32 - 1;

    // Step 1: Window, shuffle, fold into f[0..N/2-1]
    // Matches C clt_mdct_forward_c() using signed index arithmetic.
    let mut f = vec![0i32; n2];
    {
        let half_ov = (ov >> 1) as i32;
        let n2i = n2 as i32;
        let n4i = n4 as i32;
        let ov_quarter = ((ov as i32) + 3) >> 2;

        let mut yp = 0usize;
        let mut xp1 = half_ov;
        let mut xp2 = n2i - 1 + half_ov;
        let mut wp1 = half_ov;
        let mut wp2 = half_ov - 1;

        // First region: overlap windowed
        for _i in 0..ov_quarter {
            let w1 = window[wp1 as usize] as i32;
            let w2 = window[wp2 as usize] as i32;
            f[yp] = mult16_32_q15(w2, input[(xp1 + n2i) as usize])
                + mult16_32_q15(w1, input[xp2 as usize]);
            f[yp + 1] = mult16_32_q15(w1, input[xp1 as usize])
                - mult16_32_q15(w2, input[(xp2 - n2i) as usize]);
            xp1 += 2;
            xp2 -= 2;
            wp1 += 2;
            wp2 -= 2;
            yp += 2;
        }

        // Middle region: no windowing
        let mut i = ov_quarter;
        while i < n4i - ov_quarter {
            f[yp] = input[xp2 as usize];
            f[yp + 1] = input[xp1 as usize];
            xp1 += 2;
            xp2 -= 2;
            yp += 2;
            i += 1;
        }

        // Last region: overlap windowed (tail)
        wp1 = 0;
        wp2 = ov as i32 - 1;
        while i < n4i {
            let w1 = window[wp1 as usize] as i32;
            let w2 = window[wp2 as usize] as i32;
            f[yp] = -mult16_32_q15(w1, input[(xp1 - n2i) as usize])
                + mult16_32_q15(w2, input[xp2 as usize]);
            f[yp + 1] = mult16_32_q15(w2, input[xp1 as usize])
                + mult16_32_q15(w1, input[(xp2 + n2i) as usize]);
            xp1 += 2;
            xp2 -= 2;
            wp1 += 2;
            wp2 -= 2;
            yp += 2;
            i += 1;
        }
    }

    // Step 2: Pre-rotation using pre-computed trig table
    // C: t0 = t[i], t1 = t[N4+i] (from mdct_lookup.trig)
    let mut f2 = vec![KissFftCpx::default(); n4];
    let mut maxval = 1i32;
    for i in 0..n4 {
        let t0 = trig[i] as i32;
        let t1 = trig[n4 + i] as i32;

        let re = f[2 * i];
        let im = f[2 * i + 1];
        let yr = mult16_32_q15(t0, re).wrapping_sub(mult16_32_q15(t1, im));
        let yi = mult16_32_q15(t0, im).wrapping_add(mult16_32_q15(t1, re));

        // S_MUL2 = MULT16_32_Q16 (non-QEXT)
        let yc_r = mult16_32_q16(scale, yr);
        let yc_i = mult16_32_q16(scale, yi);

        maxval = maxval.max(yc_r.abs()).max(yc_i.abs());

        let br = fft_state.bitrev[i] as usize;
        f2[br] = KissFftCpx { r: yc_r, i: yc_i };
    }

    // Headroom calculation for fixed-point FFT
    let headroom = 0i32.max(scale_shift.min(28 - celt_ilog2(maxval)));

    // Step 3: N/4-point complex FFT
    opus_fft_impl(fft_state, &mut f2, scale_shift - headroom);

    // Step 4: Post-rotation using same trig table
    // C: t0 = t[i], t1 = t[N4+i] (non-QEXT path)
    let st = stride as usize;
    let mut yp1 = 0usize;
    let mut yp2 = st * (n2 - 1);
    for i in 0..n4 {
        let t0 = trig[i] as i32;
        let t1 = trig[n4 + i] as i32;

        let yr = pshr32(
            mult16_32_q15(t1, f2[i].i).wrapping_sub(mult16_32_q15(t0, f2[i].r)),
            headroom,
        );
        let yi = pshr32(
            mult16_32_q15(t1, f2[i].r).wrapping_add(mult16_32_q15(t0, f2[i].i)),
            headroom,
        );

        output[yp1] = yr;
        if yp2 < output.len() {
            output[yp2] = yi;
        }
        yp1 += 2 * st;
        yp2 = yp2.wrapping_sub(2 * st);
    }
}

/// Select the correct FFT state for a given shift level.
pub(crate) fn get_fft_state(shift: i32) -> &'static KissFftState {
    match shift {
        0 => &fft::FFT_STATE_48000_960_0,
        1 => &fft::FFT_STATE_48000_960_1,
        2 => &fft::FFT_STATE_48000_960_2,
        3 => &fft::FFT_STATE_48000_960_3,
        _ => &fft::FFT_STATE_48000_960_0,
    }
}

// ===========================================================================
// Encoder helper: compute_mdcts
// ===========================================================================

/// Compute MDCT of the input signal, with short-block or long-block mode.
///
/// Matches C `compute_mdcts()` from `celt_encoder.c`.
fn compute_mdcts(
    mode: &CELTMode,
    short_blocks: i32,
    input: &[i32],
    output: &mut [i32],
    c: i32,
    cc: i32,
    lm: i32,
    upsample: i32,
) {
    let overlap = mode.overlap;
    let (b, n, shift) = if short_blocks != 0 {
        (short_blocks, mode.short_mdct_size, mode.max_lm)
    } else {
        (1, mode.short_mdct_size << lm, mode.max_lm - lm)
    };

    let fft_st = get_fft_state(shift);

    // Compute trig table offset, matching C: trig += N for each shift level
    let mdct = &super::mdct::MDCT_48000_960;
    let mut trig_offset = 0usize;
    let mut trig_n = mdct.n;
    for _ in 0..shift {
        trig_n >>= 1;
        trig_offset += trig_n as usize;
    }
    let trig = &mdct.trig[trig_offset..];

    let stride = if short_blocks != 0 { b } else { 1 };
    for ch in 0..cc as usize {
        let in_base = ch * (b as usize * n as usize + overlap as usize);
        for blk in 0..b as usize {
            let in_start = in_base + blk * n as usize;
            let out_base = blk + ch * (n as usize * b as usize);
            // C calls clt_mdct_forward with stride directly — the post-rotation
            // writes output at positions [0, 2*stride, 4*stride, ...] and
            // [stride*(N2-1), stride*(N2-3), ...]. We must pass stride directly
            // to match C's interleaved output layout.
            clt_mdct_forward(
                &input[in_start..],
                &mut output[out_base..],
                mode.window,
                overlap,
                shift,
                stride,
                fft_st,
                trig,
            );
        }
    }

    // If CC==2 and C==1, mix stereo to mono
    if cc == 2 && c == 1 {
        let bn = (b * n) as usize;
        for i in 0..bn {
            output[i] = half32(output[i]) + half32(output[bn + i]);
        }
    }

    // Handle upsampling: scale and zero upper bins
    if upsample != 1 {
        let bn = (b * n) as usize;
        let bound = bn / upsample as usize;
        for ch in 0..c as usize {
            let base = ch * bn;
            for i in 0..bound {
                output[base + i] *= upsample;
            }
            for i in bound..bn {
                output[base + i] = 0;
            }
        }
    }
}

// ===========================================================================
// Encoder helper: celt_preemphasis
// ===========================================================================

/// Apply pre-emphasis filter to PCM input.
///
/// Matches C `celt_preemphasis()` from `celt_encoder.c` (fast path). The
/// state representation must match C exactly: `mem` holds
/// `MULT16_32_Q15(coef0, x_prev)`, NOT `x_prev` itself. Both forms produce
/// identical output samples, but downstream diagnostic tooling (and a
/// belt-and-braces match against the C encoder state) assumes the C form.
pub fn celt_preemphasis(
    pcm: &[i16],
    pcm_offset: usize,
    inp: &mut [i32],
    inp_offset: usize,
    n: i32,
    cc: i32,
    upsample: i32,
    coef: &[i32; 4],
    mem: &mut i32,
    _clip: i32,
) {
    let coef0 = extract16(coef[0]);
    let nu = n as usize;
    let ccu = cc as usize;

    // Fast path: no upsampling, simple first-order pre-emphasis.
    // Matches C celt_encoder.c:570-580 — keeps mem = MULT16_32_Q15(coef0, x_prev).
    if upsample == 1 {
        let mut m = *mem;
        for i in 0..nu {
            let x = shl32(pcm[(pcm_offset + i * ccu) as usize] as i32, SIG_SHIFT);
            inp[inp_offset + i] = x - m;
            m = mult16_32_q15(coef0, x);
        }
        *mem = m;
    } else {
        // Upsampling path: insert zeros between samples.
        // Mirrors the C upsample loop at celt_encoder.c:636-644.
        let mut m = *mem;
        let mut j = 0usize;
        for i in 0..nu {
            let x = if (i % upsample as usize) == 0 && j < (nu / upsample as usize) {
                let val = shl32(pcm[(pcm_offset + j * ccu) as usize] as i32, SIG_SHIFT);
                j += 1;
                val
            } else {
                0i32
            };
            inp[inp_offset + i] = x - m;
            m = mult16_32_q15(coef0, x);
        }
        *mem = m;
    }
}

// ===========================================================================
// Encoder helper: transient_analysis
// ===========================================================================

/// Detect transients in the input signal.
///
/// Matches C `transient_analysis()` from `celt_encoder.c` (FIXED_POINT path).
/// Returns true if a transient is detected (`mask_metric > 200`).
fn transient_analysis(
    input: &[i32],
    len: i32,
    cc: i32,
    tf_estimate: &mut i32,
    tf_chan: &mut i32,
    allow_weak_transients: bool,
    weak_transient: &mut bool,
    tone_freq: i32,
    toneishness: i32,
) -> bool {
    let len = len as usize;
    let len2 = len / 2;
    #[allow(unused_assignments)]
    let mut is_transient = false;
    let mut mask_metric: i32 = 0;
    let tf_max: i32;

    // Forward masking decay: 6.7 dB/ms normal, 3.3 dB/ms for weak transients
    let forward_shift: i32 = if allow_weak_transients { 5 } else { 4 };

    *weak_transient = false;
    *tf_chan = 0;

    // Normalize input to avoid overflow in filter (C: celt_encoder.c:299)
    let in_shift = 0i32.max(celt_ilog2(1 + celt_maxabs32(&input[..cc as usize * len])) - 14);

    let mut tmp = vec![0i32; len];

    for c in 0..cc as usize {
        let mut mem0: i32 = 0;
        let mut mem1: i32 = 0;
        let mut unmask: i32 = 0;

        // High-pass filter: (1 - 2*z^-1 + z^-2) / (1 - z^-1 + .5*z^-2)
        // (C: celt_encoder.c:326-348, FIXED_POINT path)
        for i in 0..len {
            let x = shr32(input[i + c * len], in_shift);
            let y = mem0 + x;
            mem0 = mem1 + y - shl32(x, 1);
            mem1 = x - shr32(y, 1);
            tmp[i] = sround16(y, 2);
        }

        // First few samples are bad because we don't propagate the memory
        for i in 0..12.min(len) {
            tmp[i] = 0;
        }

        // Normalize tmp to max range (C: celt_encoder.c:353-364)
        {
            let shift = 14 - celt_ilog2(max16(1, celt_maxabs16(&tmp[..len])));
            if shift != 0 {
                for i in 0..len {
                    tmp[i] = shl16(tmp[i], shift);
                }
            }
        }

        // Forward pass: grouping by two (C: celt_encoder.c:370-382)
        let mut mean: i32 = 0;
        mem0 = 0;
        for i in 0..len2 {
            let x2 = pshr32(
                mult16_16(tmp[2 * i], tmp[2 * i]) + mult16_16(tmp[2 * i + 1], tmp[2 * i + 1]),
                4,
            );
            mean += pshr32(x2, 12);
            mem0 = mem0 + pshr32(x2 - mem0, forward_shift);
            tmp[i] = pshr32(mem0, 12);
        }

        // Backward pass (C: celt_encoder.c:387-400)
        mem0 = 0;
        let mut max_e: i32 = 0;
        for i in (0..len2).rev() {
            mem0 = mem0 + pshr32(shl32(tmp[i], 4) - mem0, 3);
            tmp[i] = pshr32(mem0, 4);
            max_e = max16(max_e, tmp[i]);
        }

        // Frame energy: geometric mean of energy and half the max
        // (C: celt_encoder.c:410-411)
        mean = mult16_16(
            celt_sqrt(mean),
            celt_sqrt(mult16_16(max_e, (len2 >> 1) as i32)),
        );

        // Inverse of the mean energy in Q15+6 (C: celt_encoder.c:416)
        let norm = shl32(extend32(len2 as i32), 6 + 14) / (EPSILON + shr32(mean, 1));

        // Harmonic mean, 1/4th of samples (C: celt_encoder.c:426-435)
        let mut i = 12;
        while i < len2.saturating_sub(5) {
            let id = max32(0, min32(127, mult16_32_q15(tmp[i] + EPSILON, norm))) as usize;
            unmask += INV_TABLE[id] as i32;
            i += 4;
        }

        // Normalize (C: celt_encoder.c:438)
        unmask = 64 * unmask * 4 / (6 * (len2 as i32 - 17));

        if unmask > mask_metric {
            *tf_chan = c as i32;
            mask_metric = unmask;
        }
    }

    is_transient = mask_metric > 200;

    // Tone protection (C: celt_encoder.c:448-452)
    // C: QCONST32(.98f, 29) = 526133504 (f32 precision; f64 gives 526133494)
    if toneishness > 526133504 && tone_freq < qconst16(0.026, 13) {
        is_transient = false;
        mask_metric = 0;
    }

    // Weak transient handling (C: celt_encoder.c:455-458)
    if allow_weak_transients && is_transient && mask_metric < 600 {
        *weak_transient = true;
        is_transient = false;
    }

    // VBR boost metric (C: celt_encoder.c:460-462)
    tf_max = max16(0, celt_sqrt(27 * mask_metric.max(0)) - 42);
    *tf_estimate = celt_sqrt(max32(
        0,
        shl32(mult16_16(qconst16(0.0069, 14), min16(163, tf_max)), 14) - qconst32(0.139, 28),
    ));

    is_transient
}

// ===========================================================================
// Encoder helper: patch_transient_decision
// ===========================================================================

/// Check if a transient was missed by comparing current and previous frame energies.
///
/// Matches C `patch_transient_decision()` from `celt_encoder.c`.
fn patch_transient_decision(
    band_log_e: &[i32],
    old_band_e: &[i32],
    nb_ebands: i32,
    start: i32,
    end: i32,
    c: i32,
) -> bool {
    let mut mean_diff: i32 = 0;
    let mut spread_old = [0i32; 26];
    let nb = nb_ebands as usize;
    let s = start as usize;
    let e = end as usize;

    // Compute spread_old ONCE, outside the channel loop (Bug 1 fix).
    // Apply an aggressive (-6 dB/Bark) spreading function to the old frame
    // to avoid false detection caused by irrelevant bands.
    if c == 1 {
        // Bug 3 fix: only initialize spread_old[start], not entire array.
        spread_old[s] = old_band_e[s];
        // Bug 2 fix: forward loop starts from start+1, not 1.
        for i in (s + 1)..e {
            spread_old[i] = spread_old[i - 1]
                .wrapping_sub(gconst(1.0))
                .max(old_band_e[i]);
        }
    } else {
        spread_old[s] = old_band_e[s].max(old_band_e[s + nb]);
        for i in (s + 1)..e {
            spread_old[i] = spread_old[i - 1]
                .wrapping_sub(gconst(1.0))
                .max(old_band_e[i].max(old_band_e[i + nb]));
        }
    }
    // Backward spreading
    for i in (s..=(e.wrapping_sub(2))).rev() {
        spread_old[i] = spread_old[i].max(spread_old[i + 1].wrapping_sub(gconst(1.0)));
    }

    // Compute mean increase (channel loop)
    // Bug 5 fix: use per-channel newE[i + c*nbEBands], not max of both channels.
    // Bug 4 fix: upper bound is end-1, not end.
    // Bug 6 fix: C declares x1, x2 as opus_val16 (int16), so assigning the
    // int32 celt_glog values from MAXG(0, ...) truncates the high bits.
    // This silent truncation is required for bit-exact parity with the C
    // reference. See wrk_docs/2026.04.08 - HLD - celt-redundancy-fix.md.
    let mut ch = 0;
    loop {
        for i in (2.max(s))..(e - 1) {
            let x1 = (0i32.max(band_log_e[i + ch * nb]) as i16) as i32;
            let x2 = (0i32.max(spread_old[i]) as i16) as i32;
            mean_diff = mean_diff.wrapping_add(0i32.max(x1.wrapping_sub(x2)));
        }
        ch += 1;
        if ch >= c as usize {
            break;
        }
    }
    mean_diff /= c * (end - 1 - start.max(2));

    mean_diff > gconst(1.0)
}

// ===========================================================================
// Encoder helper: tf_analysis and tf_encode
// ===========================================================================

/// L1 metric with temporal bias for TF analysis.
///
/// Matches C `l1_metric()` from `celt_encoder.c`.
fn l1_metric(tmp: &[i32], n: i32, lm: i32, bias: i32) -> i32 {
    let mut l1: i32 = 0;
    for i in 0..n as usize {
        l1 += abs16(shr32(tmp[i], NORM_SHIFT - 14));
    }
    // Frequency bias: longer windows get a bonus
    l1 = mac16_32_q15(l1, lm * bias, l1);
    l1
}

/// Time-frequency resolution analysis using Viterbi DP.
///
/// Matches C `tf_analysis()` from `celt_encoder.c`.
fn tf_analysis(
    m: &CELTMode,
    len: i32,
    is_transient: bool,
    tf_res: &mut [i32],
    lambda: i32,
    x: &[i32],
    n0: i32,
    lm: i32,
    tf_estimate: i32,
    tf_chan: i32,
    importance: &[i32],
) -> i32 {
    let is_trans_i = is_transient as usize;

    let bias = mult16_16_q14(
        qconst16(0.04, 15),
        max16(-qconst16(0.25, 14), qconst16(0.5, 14) - tf_estimate),
    );

    let mut metric = vec![0i32; len as usize];
    let mut path0 = vec![0i32; len as usize];
    let mut path1 = vec![0i32; len as usize];

    // Allocate tmp buffers sized for the largest band
    let max_band_size = ((m.ebands[len as usize] - m.ebands[len as usize - 1]) << lm) as usize;
    let mut tmp = vec![0i32; max_band_size];
    let mut tmp_1 = vec![0i32; max_band_size];

    // Compute metric per band
    for i in 0..len as usize {
        let n = ((m.ebands[i + 1] - m.ebands[i]) << lm) as usize;
        let narrow = (m.ebands[i + 1] - m.ebands[i]) == 1;

        // Copy band data
        let x_off = tf_chan as usize * n0 as usize + (m.ebands[i] << lm) as usize;
        tmp[..n].copy_from_slice(&x[x_off..x_off + n]);

        let mut best_l1 = l1_metric(&tmp[..n], n as i32, if is_transient { lm } else { 0 }, bias);
        let mut best_level = 0i32;

        // Check the -1 case for transients
        if is_transient && !narrow {
            tmp_1[..n].copy_from_slice(&tmp[..n]);
            haar1(&mut tmp_1[..n], (n >> lm as usize) as i32, 1 << lm);
            let l1 = l1_metric(&tmp_1[..n], n as i32, lm + 1, bias);
            if l1 < best_l1 {
                best_l1 = l1;
                best_level = -1;
            }
        }

        let k_limit = if is_transient || narrow { lm } else { lm + 1 };
        for k in 0..k_limit as usize {
            let b = if is_transient {
                lm - k as i32 - 1
            } else {
                k as i32 + 1
            };
            haar1(&mut tmp[..n], (n >> k) as i32, 1 << k);
            let l1 = l1_metric(&tmp[..n], n as i32, b, bias);
            if l1 < best_l1 {
                best_l1 = l1;
                best_level = k as i32 + 1;
            }
        }

        metric[i] = if is_transient {
            2 * best_level
        } else {
            -2 * best_level
        };
        if narrow && (metric[i] == 0 || metric[i] == -2 * lm) {
            metric[i] -= 1;
        }
    }

    // Search for optimal tf_select
    let mut tf_select = 0i32;
    let mut selcost = [0i32; 2];
    for sel in 0..2 {
        let mut cost0 = importance[0]
            * (metric[0] - 2 * TF_SELECT_TABLE[lm as usize][4 * is_trans_i + 2 * sel + 0] as i32)
                .abs();
        let mut cost1 = importance[0]
            * (metric[0] - 2 * TF_SELECT_TABLE[lm as usize][4 * is_trans_i + 2 * sel + 1] as i32)
                .abs()
            + if is_transient { 0 } else { lambda };
        for i in 1..len as usize {
            let curr0 = cost0.min(cost1 + lambda);
            let curr1 = (cost0 + lambda).min(cost1);
            cost0 = curr0
                + importance[i]
                    * (metric[i]
                        - 2 * TF_SELECT_TABLE[lm as usize][4 * is_trans_i + 2 * sel + 0] as i32)
                        .abs();
            cost1 = curr1
                + importance[i]
                    * (metric[i]
                        - 2 * TF_SELECT_TABLE[lm as usize][4 * is_trans_i + 2 * sel + 1] as i32)
                        .abs();
        }
        selcost[sel] = cost0.min(cost1);
    }
    if selcost[1] < selcost[0] && is_transient {
        tf_select = 1;
    }

    // Final Viterbi forward pass with chosen tf_select
    let mut cost0 = importance[0]
        * (metric[0]
            - 2 * TF_SELECT_TABLE[lm as usize][4 * is_trans_i + 2 * tf_select as usize + 0] as i32)
            .abs();
    let mut cost1 = importance[0]
        * (metric[0]
            - 2 * TF_SELECT_TABLE[lm as usize][4 * is_trans_i + 2 * tf_select as usize + 1] as i32)
            .abs()
        + if is_transient { 0 } else { lambda };
    for i in 1..len as usize {
        let from0 = cost0;
        let from1 = cost1 + lambda;
        let curr0 = if from0 < from1 {
            path0[i] = 0;
            from0
        } else {
            path0[i] = 1;
            from1
        };
        let from0 = cost0 + lambda;
        let from1 = cost1;
        let curr1 = if from0 < from1 {
            path1[i] = 0;
            from0
        } else {
            path1[i] = 1;
            from1
        };
        cost0 = curr0
            + importance[i]
                * (metric[i]
                    - 2 * TF_SELECT_TABLE[lm as usize][4 * is_trans_i + 2 * tf_select as usize + 0]
                        as i32)
                    .abs();
        cost1 = curr1
            + importance[i]
                * (metric[i]
                    - 2 * TF_SELECT_TABLE[lm as usize][4 * is_trans_i + 2 * tf_select as usize + 1]
                        as i32)
                    .abs();
    }

    // Backward trace
    tf_res[len as usize - 1] = if cost0 < cost1 { 0 } else { 1 };
    for i in (0..len as usize - 1).rev() {
        tf_res[i] = if tf_res[i + 1] == 1 {
            path1[i + 1]
        } else {
            path0[i + 1]
        };
    }

    tf_select
}

/// Encode TF resolution flags into the bitstream.
///
/// Matches C `tf_encode()` from `celt_encoder.c`.
fn tf_encode(
    start: i32,
    end: i32,
    is_transient: bool,
    tf_res: &mut [i32],
    lm: i32,
    mut tf_select: i32,
    ec: &mut RangeEncoder,
) {
    let mut budget = ec.storage() * 8;
    let mut tell = ec.tell() as u32;
    let mut logp: u32 = if is_transient { 2 } else { 4 };
    // Reserve space to code the tf_select decision.
    let tf_select_rsv = lm > 0 && tell + logp + 1 <= budget;
    budget -= tf_select_rsv as u32;
    let mut curr = 0i32;
    let mut tf_changed = 0i32;

    for i in start..end {
        if tell + logp <= budget {
            ec.encode_bit_logp(tf_res[i as usize] ^ curr != 0, logp);
            tell = ec.tell() as u32;
            curr = tf_res[i as usize];
            tf_changed |= curr;
        } else {
            tf_res[i as usize] = curr;
        }
        logp = if is_transient { 4 } else { 5 };
    }

    // Only code tf_select if it would actually make a difference.
    let is_trans_i = is_transient as usize;
    if tf_select_rsv
        && TF_SELECT_TABLE[lm as usize][4 * is_trans_i + 0 + tf_changed as usize]
            != TF_SELECT_TABLE[lm as usize][4 * is_trans_i + 2 + tf_changed as usize]
    {
        ec.encode_bit_logp(tf_select != 0, 1);
    } else {
        tf_select = 0;
    }

    for i in start..end {
        tf_res[i as usize] = TF_SELECT_TABLE[lm as usize]
            [4 * is_trans_i + 2 * tf_select as usize + tf_res[i as usize] as usize]
            as i32;
    }
}

// ===========================================================================
// Encoder helper: alloc_trim_analysis
// ===========================================================================

/// Compute the allocation trim value (0–10) that adjusts bit distribution.
///
/// Matches C `alloc_trim_analysis()` from `celt_encoder.c`.
fn alloc_trim_analysis(
    m: &CELTMode,
    x: &[i32],
    band_log_e: &[i32],
    end: i32,
    lm: i32,
    c: i32,
    n0: i32,
    analysis: &AnalysisInfo,
    stereo_saving: &mut i32,
    tf_estimate: i32,
    intensity: i32,
    surround_trim: i32,
    equiv_rate: i32,
) -> i32 {
    let nb_ebands = m.nb_ebands;
    let n0u = n0 as usize;

    // Base trim (Q8)
    // C: trim = QCONST16(5.f, 8);
    let mut trim: i32 = qconst16(5.0, 8);
    if equiv_rate < 64000 {
        trim = qconst16(4.0, 8);
    } else if equiv_rate < 80000 {
        // C: frac = (equiv_rate-64000) >> 10;
        //    trim = QCONST16(4.f, 8) + QCONST16(1.f/16.f, 8)*frac;
        let frac = (equiv_rate - 64000) >> 10;
        trim = qconst16(4.0, 8) + qconst16(1.0 / 16.0, 8) * frac;
    }

    // Stereo analysis: compute inter-channel correlation
    if c == 2 {
        let mut sum: i32 = 0; // Q10
        let mut min_xc: i32;

        // Bands 0-7: accumulate correlation into sum (Q10)
        // C: for (i=0;i<8;i++) {
        //      partial = celt_inner_prod_norm_shift(...);
        //      sum = ADD16(sum, EXTRACT16(SHR32(partial, 18)));
        //    }
        for i in 0..8usize {
            let partial = celt_inner_prod_norm_shift(
                &x[(m.ebands[i] as usize) << lm as usize..],
                &x[n0u + ((m.ebands[i] as usize) << lm as usize)..],
                ((m.ebands[i + 1] - m.ebands[i]) as usize) << lm as usize,
            );
            sum = add16(sum, extract16(shr32(partial, 18)));
        }
        // C: sum = MULT16_16_Q15(QCONST16(1.f/8, 15), sum);
        sum = mult16_16_q15(qconst16(1.0 / 8.0, 15), sum);
        // C: sum = MIN16(QCONST16(1.f, 10), ABS16(sum));
        sum = min16(qconst16(1.0, 10), abs16(sum));
        // C: minXC = sum;
        min_xc = sum;

        // Bands 8..intensity: update minXC
        // C: for (i=8;i<intensity;i++) {
        //      partial = celt_inner_prod_norm_shift(...);
        //      minXC = MIN16(minXC, ABS16(EXTRACT16(SHR32(partial, 18))));
        //    }
        for i in 8..(intensity as usize) {
            let partial = celt_inner_prod_norm_shift(
                &x[(m.ebands[i] as usize) << lm as usize..],
                &x[n0u + ((m.ebands[i] as usize) << lm as usize)..],
                ((m.ebands[i + 1] - m.ebands[i]) as usize) << lm as usize,
            );
            min_xc = min16(min_xc, abs16(extract16(shr32(partial, 18))));
        }
        // C: minXC = MIN16(QCONST16(1.f, 10), ABS16(minXC));
        min_xc = min16(qconst16(1.0, 10), abs16(min_xc));

        // Mid-side savings based on LF average
        // C: logXC = celt_log2(QCONST32(1.001f, 20)-MULT16_16(sum, sum));
        let mut log_xc = celt_log2(qconst32(1.001, 20) - mult16_16(sum, sum));
        // Mid-side savings based on min correlation
        // C: logXC2 = MAX16(HALF16(logXC), celt_log2(QCONST32(1.001f, 20)-MULT16_16(minXC, minXC)));
        let mut log_xc2 = max16(
            half16(log_xc),
            celt_log2(qconst32(1.001, 20) - mult16_16(min_xc, min_xc)),
        );

        // FIXED_POINT: compensate for Q20 vs Q14 input and convert to Q8
        // C: logXC = PSHR32(logXC-QCONST16(6.f, 10),10-8);
        //    logXC2 = PSHR32(logXC2-QCONST16(6.f, 10),10-8);
        log_xc = pshr32(log_xc - qconst16(6.0, 10), 10 - 8);
        log_xc2 = pshr32(log_xc2 - qconst16(6.0, 10), 10 - 8);

        // C: trim += MAX16(-QCONST16(4.f, 8), MULT16_16_Q15(QCONST16(.75f,15),logXC));
        trim += max16(-qconst16(4.0, 8), mult16_16_q15(qconst16(0.75, 15), log_xc));
        // C: *stereo_saving = MIN16(*stereo_saving + QCONST16(0.25f, 8), -HALF16(logXC2));
        *stereo_saving = min16(*stereo_saving + qconst16(0.25, 8), -half16(log_xc2));
    }

    // Estimate spectral tilt
    // C: c=0; do {
    //      for (i=0;i<end-1;i++)
    //         diff += SHR32(bandLogE[i+c*m->nbEBands], 5)*(opus_int32)(2+2*i-end);
    //    } while (++c<C);
    //    diff /= C*(end-1);
    let mut diff: i32 = 0;
    let mut ch = 0;
    loop {
        for i in 0..(end - 1) as usize {
            diff += shr32(band_log_e[i + ch * nb_ebands as usize], 5) * (2 + 2 * i as i32 - end);
        }
        ch += 1;
        if ch >= c as usize {
            break;
        }
    }
    diff /= c * (end - 1);

    // C: trim -= MAX32(-QCONST16(2.f, 8), MIN32(QCONST16(2.f, 8),
    //          SHR32(diff+QCONST32(1.f, DB_SHIFT-5),DB_SHIFT-13)/6 ));
    trim -= (-qconst16(2.0, 8)).max(
        qconst16(2.0, 8).min(shr32(diff + qconst32(1.0, DB_SHIFT as u32 - 5), DB_SHIFT - 13) / 6),
    );

    // Surround masking adjustment
    // C: trim -= SHR16(surround_trim, DB_SHIFT-8);
    trim -= shr16(surround_trim, DB_SHIFT - 8);

    // TF estimate adjustment
    // C: trim -= 2*SHR16(tf_estimate, 14-8);
    trim -= 2 * shr16(tf_estimate, 14 - 8);

    // Analysis-based tonality-slope adjustment (C celt_encoder.c:934-939).
    if analysis.valid != 0 {
        let slope_f = 512.0_f32 * (analysis.tonality_slope + 0.05_f32);
        let slope_i16 = slope_f as i16;
        let clamped = (slope_i16 as i32).clamp(-512, 512);
        trim -= clamped;
    }

    // Quantize to integer 0–10
    // C: trim_index = PSHR32(trim, 8);
    //    trim_index = IMAX(0, IMIN(10, trim_index));
    let trim_index = pshr32(trim, 8).max(0).min(10);
    trim_index
}

// ===========================================================================
// Encoder helper: stereo_analysis
// ===========================================================================

/// Decide whether dual stereo or M/S is better based on L1 norms.
///
/// Matches C `stereo_analysis()` from `celt_encoder.c`.
fn stereo_analysis(m: &CELTMode, x: &[i32], lm: i32, n0: i32) -> i32 {
    let n0u = n0 as usize;
    let mut sum_lr: i32 = EPSILON;
    let mut sum_ms: i32 = EPSILON;

    // Use the L1 norm to model the entropy of the L/R signal vs the M/S signal.
    // Always loop 13 bands (the theta adjustment handles LM<=1 separately).
    for i in 0..13 {
        let j_start = (m.ebands[i] as usize) << (lm as usize);
        let j_end = (m.ebands[i + 1] as usize) << (lm as usize);
        for j in j_start..j_end {
            // SHR32 normalization: shift from Q(NORM_SHIFT) to Q14
            let l = shr32(x[j], NORM_SHIFT - 14);
            let r = shr32(x[n0u + j], NORM_SHIFT - 14);
            let ms_m = l + r;
            let s = l - r;
            sum_lr += abs32(l) + abs32(r);
            sum_ms += abs32(ms_m) + abs32(s);
        }
    }

    let sum_ms = mult16_32_q15(qconst16(0.707107, 15), sum_ms);
    // We don't need thetas for lower bands with LM<=1
    let mut thetas: i32 = 13;
    if lm <= 1 {
        thetas -= 8;
    }
    let ebands13 = m.ebands[13] as i32;
    (mult16_32_q15((ebands13 << (lm + 1)) + thetas, sum_ms)
        > mult16_32_q15(ebands13 << (lm + 1), sum_lr)) as i32
}

// ===========================================================================
// Encoder helpers: median_of_5 / median_of_3
// ===========================================================================

/// Median of 5 values using a sorting network.
fn median_of_5(d: &[i32]) -> i32 {
    // Matches C reference median_of_5() from celt_encoder.c
    let t2 = d[2];
    let (t0, t1) = if d[0] > d[1] {
        (d[1], d[0])
    } else {
        (d[0], d[1])
    };
    let (t3, t4) = if d[3] > d[4] {
        (d[4], d[3])
    } else {
        (d[3], d[4])
    };
    let (t1, t3, t4) = if t0 > t3 { (t4, t0, t1) } else { (t1, t3, t4) };
    if t2 > t1 {
        if t1 < t3 { t2.min(t3) } else { t4.min(t1) }
    } else {
        if t2 < t3 { t1.min(t3) } else { t2.min(t4) }
    }
}

/// Median of 3 values.
fn median_of_3(a: i32, b: i32, c: i32) -> i32 {
    if a < b {
        if b < c {
            b
        } else if a < c {
            c
        } else {
            a
        }
    } else {
        if a < c {
            a
        } else if b < c {
            c
        } else {
            b
        }
    }
}

// ===========================================================================
// Encoder helper: dynalloc_analysis
// ===========================================================================

/// Dynamic bit allocation analysis.
///
/// Matches C `dynalloc_analysis()` from `celt_encoder.c`.
/// Computes per-band boost offsets based on spectral envelope analysis.
fn dynalloc_analysis(
    band_log_e: &[i32],
    band_log_e2: &[i32],
    old_band_e: &[i32],
    nb_ebands: i32,
    start: i32,
    end: i32,
    c: i32,
    offsets: &mut [i32],
    lsb_depth: i32,
    log_n: &[i16],
    is_transient: bool,
    vbr: i32,
    constrained_vbr: i32,
    e_bands: &[i16],
    lm: i32,
    effective_bytes: i32,
    tot_boost: &mut i32,
    lfe: i32,
    surround_dynalloc: &[i32],
    analysis: &AnalysisInfo,
    importance: &mut [i32],
    spread_weight: &mut [i32],
    tone_freq: i32,
    toneishness: i32,
) -> i32 {
    let nbu = nb_ebands as usize;
    // C: maxDepth=-GCONST(31.9f) = -535193184 (f32 precision; f64 gives -535193190)
    let mut max_depth: i32 = -535193184;
    *tot_boost = 0;
    for i in 0..nbu {
        offsets[i] = 0;
    }

    let mut follower = vec![0i32; c as usize * nbu];
    let mut noise_floor = vec![0i32; nbu];

    // Compute noise floor per band (C: lines 1076-1083)
    // noise_floor[i] = GCONST(0.0625)*logN[i] + GCONST(0.5) + SHL32(9-lsb_depth,DB_SHIFT)
    //                - SHL32(eMeans[i],DB_SHIFT-4) + GCONST(0.0062)*(i+5)*(i+5)
    for i in 0..end as usize {
        noise_floor[i] =
            gconst(0.0625) * (log_n[i] as i32) + gconst(0.5) + shl32(9 - lsb_depth, DB_SHIFT)
                - shl32(EMEANS[i] as i32, DB_SHIFT - 4)
                + gconst(0.0062) * ((i as i32 + 5) * (i as i32 + 5));
    }

    // Compute maxDepth from bandLogE (C: lines 1084-1088)
    for ch in 0..c as usize {
        for i in 0..end as usize {
            max_depth = max_depth.max(band_log_e[ch * nbu + i] - noise_floor[i]);
        }
    }

    // Masking model for spread_weight (C: lines 1089-1124)
    {
        let mut mask = vec![0i32; nbu];
        let mut sig = vec![0i32; nbu];
        for i in 0..end as usize {
            mask[i] = band_log_e[i] - noise_floor[i];
        }
        if c == 2 {
            for i in 0..end as usize {
                mask[i] = mask[i].max(band_log_e[nbu + i] - noise_floor[i]);
            }
        }
        sig[..end as usize].copy_from_slice(&mask[..end as usize]);
        for i in 1..end as usize {
            mask[i] = mask[i].max(mask[i - 1] - gconst(2.0));
        }
        for i in (0..end as usize - 1).rev() {
            mask[i] = mask[i].max(mask[i + 1] - gconst(3.0));
        }
        for i in 0..end as usize {
            // SMR: mask is never more than 72 dB below peak, never below noise floor
            let smr = sig[i] - 0i32.max(max_depth - gconst(12.0)).max(mask[i]);
            // Clamp SMR to compute shift (C uses PSHR32 for fixed-point)
            let shift = -pshr32((-gconst(5.0)).max(smr.min(0)), DB_SHIFT);
            spread_weight[i] = 32 >> shift;
        }
    }

    // Main dynalloc computation (C: lines 1128-1276)
    // Guard: only compute when we have enough bytes (C: effectiveBytes >= 30+5*LM)
    if effective_bytes >= (30 + 5 * lm) && lfe == 0 {
        let mut band_log_e3 = vec![0i32; nbu];
        let mut last = 0i32;

        // Per-channel follower computation (C: lines 1131-1173)
        for ch in 0..c as usize {
            // Copy bandLogE2 for this channel
            band_log_e3[..end as usize]
                .copy_from_slice(&band_log_e2[ch * nbu..ch * nbu + end as usize]);

            // For LM==0, take max with oldBandE (C: lines 1137-1142)
            if lm == 0 {
                for i in 0..8.min(end as usize) {
                    band_log_e3[i] = band_log_e3[i].max(old_band_e[ch * nbu + i]);
                }
            }

            let f = &mut follower[ch * nbu..];

            // Forward smoothing (C: lines 1145-1153)
            f[0] = band_log_e3[0];
            for i in 1..end as usize {
                if band_log_e3[i] > band_log_e3[i - 1] + gconst(0.5) {
                    last = i as i32;
                }
                f[i] = (f[i - 1] + gconst(1.5)).min(band_log_e3[i]);
            }

            // Backward smoothing from last (C: lines 1155-1156)
            for i in (0..last as usize).rev() {
                f[i] = f[i].min((f[i + 1] + gconst(2.0)).min(band_log_e3[i]));
            }

            // Median filter overlay (C: lines 1161-1169)
            let offset = gconst(1.0);
            for i in 2..end as usize - 2 {
                f[i] = f[i].max(median_of_5(&band_log_e3[i - 2..]) - offset);
            }
            let tmp = median_of_3(band_log_e3[0], band_log_e3[1], band_log_e3[2]) - offset;
            f[0] = f[0].max(tmp);
            f[1] = f[1].max(tmp);
            let e = end as usize;
            let tmp =
                median_of_3(band_log_e3[e - 3], band_log_e3[e - 2], band_log_e3[e - 1]) - offset;
            f[e - 2] = f[e - 2].max(tmp);
            f[e - 1] = f[e - 1].max(tmp);

            // Noise floor enforcement (C: lines 1171-1172)
            for i in 0..end as usize {
                f[i] = f[i].max(noise_floor[i]);
            }
        }

        // Transform follower to excess energy (C: lines 1174-1188)
        if c == 2 {
            for i in start as usize..end as usize {
                // Cross-channel masking (24 dB)
                follower[nbu + i] = follower[nbu + i].max(follower[i] - gconst(4.0));
                follower[i] = follower[i].max(follower[nbu + i] - gconst(4.0));
                follower[i] = half32(
                    (band_log_e[i] - follower[i]).max(0)
                        + (band_log_e[nbu + i] - follower[nbu + i]).max(0),
                );
            }
        } else {
            for i in start as usize..end as usize {
                follower[i] = (band_log_e[i] - follower[i]).max(0);
            }
        }

        // Add surround dynalloc (C: lines 1189-1190)
        for i in start as usize..end as usize {
            if i < surround_dynalloc.len() {
                follower[i] = follower[i].max(surround_dynalloc[i]);
            }
        }

        // Compute importance (C: lines 1191-1198)
        for i in start as usize..end as usize {
            let follow_clamped = follower[i].min(gconst(4.0));
            let exp_val = celt_exp2(pshr32(follow_clamped, DB_SHIFT - 10));
            importance[i] = pshr32(13 * exp_val, 16);
        }

        // CBR/CVBR halving: C: `(!vbr || constrained_vbr) && !isTransient`
        if (vbr == 0 || constrained_vbr != 0) && !is_transient {
            for i in start as usize..end as usize {
                follower[i] = half32(follower[i]);
            }
        }

        // Band scaling (C: lines 1205-1211)
        for i in start as usize..end as usize {
            if i < 8 {
                follower[i] *= 2;
            }
            if i >= 12 {
                follower[i] = half32(follower[i]);
            }
        }

        // Tone compensation (C: lines 1212-1229)
        // C: QCONST32(.98f, 29) = 526133504 (f32 precision; f64 gives 526133494)
        if toneishness > 526133504 {
            let freq_bin = pshr32(
                (tone_freq as i64 * qconst16(120.0 / std::f64::consts::PI, 9) as i64) as i32,
                13 + 9,
            );
            for i in start as usize..end as usize {
                if freq_bin >= e_bands[i] as i32 && freq_bin <= e_bands[i + 1] as i32 {
                    follower[i] += gconst(2.0);
                }
                if freq_bin >= e_bands[i] as i32 - 1 && freq_bin <= e_bands[i + 1] as i32 + 1 {
                    follower[i] += gconst(1.0);
                }
                if freq_bin >= e_bands[i] as i32 - 2 && freq_bin <= e_bands[i + 1] as i32 + 2 {
                    follower[i] += gconst(1.0);
                }
                if freq_bin >= e_bands[i] as i32 - 3 && freq_bin <= e_bands[i + 1] as i32 + 3 {
                    follower[i] += gconst(0.5);
                }
            }
            if freq_bin >= e_bands[end as usize] as i32 {
                follower[end as usize - 1] += gconst(2.0);
                follower[end as usize - 2] += gconst(1.0);
            }
        }

        // Analysis-driven leak boost (C celt_encoder.c:1226-1230).
        if analysis.valid != 0 {
            let leak_end = (LEAK_BANDS as i32).min(end) as usize;
            for i in start as usize..leak_end {
                follower[i] += gconst(1.0 / 64.0) * analysis.leak_boost[i] as i32;
            }
        }

        // Compute offsets from follower (C: lines 1239-1272)
        for i in start as usize..end as usize {
            let width = c * (e_bands[i + 1] - e_bands[i]) as i32 * (1 << lm);

            follower[i] = follower[i].min(gconst(4.0));
            follower[i] = shr32(follower[i], 8);

            let (boost, boost_bits) = if width < 6 {
                let b = shr32(follower[i], DB_SHIFT - 8) as i32;
                (b, b * width << BITRES)
            } else if width > 48 {
                let b = shr32(follower[i] * 8, DB_SHIFT - 8) as i32;
                (b, (b * width << BITRES) / 8)
            } else {
                let b = shr32(follower[i] * width / 6, DB_SHIFT - 8) as i32;
                (b, b * 6 << BITRES)
            };

            // CBR cap: C: `(!vbr || (cvbr && !trans)) && budget_exceeded`
            // Note: !trans only qualifies cvbr, not pure CBR
            if (vbr == 0 || (constrained_vbr != 0 && !is_transient))
                && (*tot_boost + boost_bits) >> BITRES >> 3 > 2 * effective_bytes / 3
            {
                let cap = (2 * effective_bytes / 3) << BITRES << 3;
                offsets[i] = cap - *tot_boost;
                *tot_boost = cap;
                break;
            } else {
                offsets[i] = boost;
                *tot_boost += boost_bits;
            }
        }
    } else {
        // Not enough bytes for dynalloc — set importance to 13 (C: lines 1273-1276)
        for i in start as usize..end as usize {
            importance[i] = 13;
        }
    }

    max_depth
}

// ===========================================================================
// Encoder helper: tone_detect (simplified)
// ===========================================================================

/// Detect pure tones in the input signal.
/// Normalize tone input levels (fixed-point only).
/// Matches C `normalize_tone_input()` from `celt_encoder.c`.
fn normalize_tone_input(x: &mut [i16], len: usize) {
    let mut ac0: i32 = len as i32;
    for i in 0..len {
        ac0 += shr32(mult16_16(x[i] as i32, x[i] as i32), 10);
    }
    let shift = 5 - (28 - celt_ilog2(ac0)) / 2;
    if shift > 0 {
        for i in 0..len {
            x[i] = pshr32(x[i] as i32, shift) as i16;
        }
    }
}

/// Fixed-point acos approximation. Input in Q29, output in Q13 (radians).
/// Matches C `acos_approx()` from `celt_encoder.c`.
fn acos_approx(x: i32) -> i32 {
    let flip = x < 0;
    let x = x.abs();
    let x14: i32 = x >> 15; // Q14
    let mut tmp: i32 = (762i32 * x14 >> 14) - 3308;
    tmp = (tmp * x14 >> 14) + 25726;
    tmp = tmp * celt_sqrt(imax(0, (1 << 30) - (x << 1))) >> 16;
    if flip { 25736 - tmp } else { tmp }
}

/// Compute 2nd-order LPC via forward+backward covariance method.
/// Matches C `tone_lpc()` from `celt_encoder.c`.
/// Returns 1 on failure (degenerate), 0 on success.
fn tone_lpc(x: &[i16], len: usize, delay: usize, lpc: &mut [i32; 2]) -> bool {
    #[allow(unused_assignments)]
    let (mut r00, mut r01, mut r11, mut r02, mut r12, mut r22): (i32, i32, i32, i32, i32, i32) =
        (0, 0, 0, 0, 0, 0);
    let d2 = 2 * delay;
    // Forward prediction correlations
    for i in 0..len - d2 {
        r00 += mult16_16(x[i] as i32, x[i] as i32);
        r01 += mult16_16(x[i] as i32, x[i + delay] as i32);
        r02 += mult16_16(x[i] as i32, x[i + d2] as i32);
    }
    // Edge corrections for r11, r22, r12
    let mut edges: i32 = 0;
    for i in 0..delay {
        edges += mult16_16(x[len + i - d2] as i32, x[len + i - d2] as i32)
            - mult16_16(x[i] as i32, x[i] as i32);
    }
    r11 = r00 + edges;
    edges = 0;
    for i in 0..delay {
        edges += mult16_16(x[len + i - delay] as i32, x[len + i - delay] as i32)
            - mult16_16(x[i + delay] as i32, x[i + delay] as i32);
    }
    r22 = r11 + edges;
    edges = 0;
    for i in 0..delay {
        edges += mult16_16(x[len + i - d2] as i32, x[len + i - delay] as i32)
            - mult16_16(x[i] as i32, x[i + delay] as i32);
    }
    r12 = r01 + edges;
    // Combine forward and backward
    let r00_save = r00;
    let r01_save = r01;
    r00 = r00_save + r22;
    r01 = r01_save + r12;
    r11 = 2 * r11;
    r02 = 2 * r02;
    r12 = r12 + r01_save;
    r22 = r00_save + r22;
    let _ = r22; // r22 not used after this

    // Solve 2x2 system
    let den = mult32_32_q31(r00, r11) - mult32_32_q31(r01, r01);
    if den <= shr32(mult32_32_q31(r00, r11), 10) {
        return true; // fail
    }
    let num1 = mult32_32_q31(r02, r11) - mult32_32_q31(r01, r12);
    if num1 >= den {
        lpc[1] = qconst32(1.0, 29);
    } else if num1 <= -den {
        lpc[1] = -qconst32(1.0, 29);
    } else {
        lpc[1] = frac_div32_q29(num1, den);
    }
    let num0 = mult32_32_q31(r00, r12) - mult32_32_q31(r02, r01);
    if half32(num0) >= den {
        // C: QCONST32(1.999999f, 29) = 1073741312 (f32 precision; f64 gives 1073741287)
        lpc[0] = 1073741312;
    } else if half32(num0) <= -den {
        lpc[0] = -1073741312;
    } else {
        lpc[0] = frac_div32_q29(num0, den);
    }
    false // success
}

/// Detect pure or nearly-pure tones.
/// Matches C `tone_detect()` from `celt_encoder.c`.
/// Returns detected tone frequency (Q13) or -1 if no tone.
fn tone_detect(input: &[i32], cc: i32, n: i32, toneishness: &mut i32, fs: i32) -> i32 {
    let nu = n as usize;
    let mut x = vec![0i16; nu];

    // Downscale input
    if cc == 2 {
        for i in 0..nu {
            x[i] = pshr32(shr32(input[i], 1) + shr32(input[i + nu], 1), SIG_SHIFT + 2) as i16;
        }
    } else {
        for i in 0..nu {
            x[i] = pshr32(input[i], SIG_SHIFT + 2) as i16;
        }
    }

    normalize_tone_input(&mut x, nu);

    let mut delay: usize = 1;
    let mut lpc = [0i32; 2];
    let mut fail = tone_lpc(&x, nu, delay, &mut lpc);

    // If LPC resonates too close to DC, retry with downsampling
    while delay as i32 <= fs / 3000 && (fail || (lpc[0] > qconst32(1.0, 29) && lpc[1] < 0)) {
        delay *= 2;
        fail = tone_lpc(&x, nu, delay, &mut lpc);
    }

    // Check for complex roots
    if !fail && mult32_32_q31(lpc[0], lpc[0]) + mult32_32_q31(qconst32(3.999999, 29), lpc[1]) < 0 {
        *toneishness = -lpc[1];
        (acos_approx(lpc[0] >> 1) + delay as i32 / 2) / delay as i32
    } else {
        *toneishness = 0;
        -1
    }
}

// ===========================================================================
// Encoder helper: run_prefilter
// ===========================================================================

/// Run the pitch pre-filter (comb filter) analysis and application.
///
/// Matches C `run_prefilter()` from `celt_encoder.c`.
/// Returns whether the pre-filter was applied (pf_on).
fn run_prefilter(
    st: &mut CeltEncoder,
    input: &mut [i32],
    cc: i32,
    n: i32,
    prefilter_tapset: &mut i32,
    pitch_index: &mut i32,
    gain: &mut i32,
    qgain: &mut i32,
    enabled: bool,
    complexity: i32,
    tf_estimate: i32,
    nb_available_bytes: i32,
    tone_freq: i32,
    toneishness: i32,
) -> i32 {
    let mode = st.mode;
    let overlap = mode.overlap;
    let nu = n as usize;
    let ovu = overlap as usize;
    let max_period = COMBFILTER_MAXPERIOD;
    let min_period = COMBFILTER_MINPERIOD;
    let mpu = max_period as usize;

    // --- Build work buffer: pre[c] = prefilter_mem[c] ++ input[c][overlap..] ---
    // Allocate per-channel work buffers (unfiltered signal with history prepended)
    let mut pre: Vec<Vec<i32>> = Vec::with_capacity(cc as usize);
    for c in 0..cc as usize {
        let mut buf = vec![0i32; nu + mpu];
        // Copy history from prefilter_mem
        buf[..mpu].copy_from_slice(&st.prefilter_mem[c * mpu..(c + 1) * mpu]);
        // Copy current frame (after overlap region)
        let in_off = c * (nu + ovu) + ovu;
        buf[mpu..mpu + nu].copy_from_slice(&input[in_off..in_off + nu]);
        pre.push(buf);
    }

    // =========================================================================
    // Phase 1: Pitch estimation (C lines 1447-1546)
    // =========================================================================

    let mut pitch_idx: i32;
    let mut gain1: i32;
    let mut pf_on: i32;
    let mut qg: i32;

    // Path 1: Tone detection (toneishness > 0.99)
    // C: QCONST32(.99f, 29) = 531502208 (f32 precision; f64 gives 531502203)
    if enabled && toneishness > 531502208 {
        let mut tf = tone_freq;
        // Alias correction: mirror frequencies above pi
        if tf >= qconst16(3.1416, 13) {
            tf = qconst16(3.141593, 13) - tone_freq;
        }
        // Find pitch doubling multiple
        let mut multiple: i32 = 1;
        while tf >= multiple * qconst16(0.39, 13) {
            multiple += 1;
        }
        if tf > qconst16(0.006148, 13) {
            pitch_idx = ((51472 * multiple + tf / 2) / tf).min(COMBFILTER_MAXPERIOD - 2);
        } else {
            pitch_idx = COMBFILTER_MINPERIOD;
        }
        gain1 = qconst16(0.75, 15);
    }
    // Path 2: Standard pitch search (enabled && complexity >= 5)
    else if enabled && complexity >= 5 {
        let pb_len = ((max_period + n) >> 1) as usize;
        let mut pitch_buf = vec![0i32; pb_len];

        // pitch_downsample takes full-resolution pre[] refs, outputs half-res pitch_buf
        let pre_refs: Vec<&[i32]> = pre.iter().map(|v| v.as_slice()).collect();
        pitch_downsample(&pre_refs, &mut pitch_buf, pb_len, cc as usize, 2);

        // Coarse pitch search: pass full-resolution N and max_period-3*min_period
        let search_result = pitch_search(
            &pitch_buf[(max_period >> 1) as usize..],
            &pitch_buf,
            n as usize,
            (max_period - 3 * min_period) as usize,
        );
        pitch_idx = max_period - search_result;

        // Fine pitch search via remove_doubling
        gain1 = remove_doubling(
            &pitch_buf,
            max_period,
            min_period,
            n,
            &mut pitch_idx,
            st.prefilter_period,
            st.prefilter_gain,
        );
        if pitch_idx > max_period - 2 {
            pitch_idx = max_period - 2;
        }

        // Scale gain by 0.7
        gain1 = mult16_16_q15(qconst16(0.7, 15), gain1);

        // Loss rate attenuation
        if st.loss_rate > 2 {
            gain1 = half32(gain1);
        }
        if st.loss_rate > 4 {
            gain1 = half32(gain1);
        }
        if st.loss_rate > 8 {
            gain1 = 0;
        }
    }
    // Path 3: Disabled
    else {
        gain1 = 0;
        pitch_idx = COMBFILTER_MINPERIOD;
    }

    // Analysis-driven pitch gain attenuation (C celt_encoder.c:1492-1497).
    if st.analysis.valid != 0 {
        let scaled = gain1 as f32 * st.analysis.max_pitch_ratio;
        gain1 = (scaled as i16) as i32;
    }

    // --- Adaptive threshold and quantization (C lines 1506-1546) ---

    let mut pf_threshold = qconst16(0.2, 15);

    // Pitch continuity check
    if (pitch_idx - st.prefilter_period).abs() * 10 > pitch_idx {
        pf_threshold += qconst16(0.2, 15);
        // Disable on strong transients without continuity
        if tf_estimate > qconst16(0.98, 14) {
            gain1 = 0;
        }
    }
    // Byte budget adjustments
    if nb_available_bytes < 25 {
        pf_threshold += qconst16(0.1, 15);
    }
    if nb_available_bytes < 35 {
        pf_threshold += qconst16(0.1, 15);
    }
    // Previous gain adjustments
    if st.prefilter_gain > qconst16(0.4, 15) {
        pf_threshold -= qconst16(0.1, 15);
    }
    if st.prefilter_gain > qconst16(0.55, 15) {
        pf_threshold -= qconst16(0.1, 15);
    }

    // Hard floor at 0.2
    pf_threshold = pf_threshold.max(qconst16(0.2, 15));

    if gain1 < pf_threshold {
        gain1 = 0;
        pf_on = 0;
        qg = 0;
    } else {
        // Gain hysteresis: snap to previous if within 0.1
        if (gain1 - st.prefilter_gain).abs() < qconst16(0.1, 15) {
            gain1 = st.prefilter_gain;
        }
        // Quantize to 3-bit index (0..7)
        qg = ((gain1 + 1536) >> 10) / 3 - 1;
        qg = qg.max(0).min(7);
        // Dequantize: gain1 = QCONST16(0.09375, 15) * (qg + 1) = 3072 * (qg + 1)
        gain1 = qconst16(0.09375, 15) * (qg + 1);
        pf_on = 1;
    }

    // =========================================================================
    // Phase 2: Comb filter application (C lines 1549-1590)
    // ALWAYS runs — even when gain1==0 — to crossfade from old_gain to 0.
    // =========================================================================

    let mut before = [0i32; 2];
    let mut after = [0i32; 2];

    // Clamp old period
    st.prefilter_period = st.prefilter_period.max(COMBFILTER_MINPERIOD);

    for c in 0..cc as usize {
        let offset = (mode.short_mdct_size - overlap) as usize;
        let in_base = c * (nu + ovu);

        // 1. Copy filtered overlap from in_mem (NOT prefilter_mem)
        input[in_base..in_base + ovu].copy_from_slice(&st.in_mem[c * ovu..(c + 1) * ovu]);

        // 2. Compute before-energy: sum of abs(input[overlap..overlap+N] >> 12)
        for i in 0..nu {
            before[c] += shr32(input[in_base + ovu + i], 12).abs();
        }

        // 3. Apply comb filter with NEGATIVE gains (analysis filter)
        //    Two segments: constant old filter, then crossfade old→new.

        // Segment 1: constant old filter (offset samples, 0 for 48kHz)
        if offset > 0 {
            let mut out_seg = vec![0i32; offset];
            comb_filter(
                &mut out_seg,
                &pre[c],
                mpu,
                st.prefilter_period,
                st.prefilter_period,
                offset as i32,
                -st.prefilter_gain,
                -st.prefilter_gain,
                st.prefilter_tapset,
                st.prefilter_tapset,
                &[], // window=NULL, overlap=0
                0,
            );
            input[in_base + ovu..in_base + ovu + offset].copy_from_slice(&out_seg);
        }

        // Segment 2: crossfade old→new (N-offset samples)
        let seg2_len = nu - offset;
        let mut out_seg2 = vec![0i32; seg2_len];
        comb_filter(
            &mut out_seg2,
            &pre[c],
            mpu + offset,
            st.prefilter_period,
            pitch_idx,
            seg2_len as i32,
            -st.prefilter_gain,
            -gain1,
            st.prefilter_tapset,
            *prefilter_tapset,
            mode.window,
            overlap,
        );
        input[in_base + ovu + offset..in_base + ovu + nu].copy_from_slice(&out_seg2);

        // 4. Compute after-energy
        for i in 0..nu {
            after[c] += shr32(input[in_base + ovu + i], 12).abs();
        }
    }

    // 5. Cancel check
    let mut cancel_pitch = false;
    if cc == 2 {
        // Stereo: threshold with cross-channel coupling
        // C declares thresh as opus_val16 (i16), so truncation is intentional
        let thresh0 = (mult16_32_q15(mult16_16_q15(qconst16(0.25, 15), gain1), before[0])
            + mult16_32_q15(qconst16(0.01, 15), before[1])) as i16 as i32;
        let thresh1 = (mult16_32_q15(mult16_16_q15(qconst16(0.25, 15), gain1), before[1])
            + mult16_32_q15(qconst16(0.01, 15), before[0])) as i16 as i32;
        // Don't use the filter if one channel gets significantly worse
        if after[0] - before[0] > thresh0 || after[1] - before[1] > thresh1 {
            cancel_pitch = true;
        }
        // Use the filter only if at least one channel gets significantly better
        if before[0] - after[0] < thresh0 && before[1] - after[1] < thresh1 {
            cancel_pitch = true;
        }
    } else {
        // Mono: check that the channel actually got better
        if after[0] > before[0] {
            cancel_pitch = true;
        }
    }

    // If cancelled, revert to unfiltered and crossfade old_gain → 0
    if cancel_pitch {
        for c in 0..cc as usize {
            let offset = (mode.short_mdct_size - overlap) as usize;
            let in_base = c * (nu + ovu);

            // Revert: copy unfiltered pre[] back
            input[in_base + ovu..in_base + ovu + nu].copy_from_slice(&pre[c][mpu..mpu + nu]);

            // Apply one final crossfade: old_gain → 0 (only overlap samples)
            let mut out_seg = vec![0i32; ovu];
            comb_filter(
                &mut out_seg,
                &pre[c],
                mpu + offset,
                st.prefilter_period,
                pitch_idx,
                overlap,
                -st.prefilter_gain,
                0, // -0 = 0
                st.prefilter_tapset,
                *prefilter_tapset,
                mode.window,
                overlap,
            );
            input[in_base + ovu + offset..in_base + ovu + offset + ovu].copy_from_slice(&out_seg);
        }
        gain1 = 0;
        pf_on = 0;
        qg = 0;
    }

    // =========================================================================
    // Phase 3: Memory update (C lines 1592-1602)
    // =========================================================================

    for c in 0..cc as usize {
        let in_base = c * (nu + ovu);

        // Save filtered tail for next frame's overlap (from in[])
        st.in_mem[c * ovu..(c + 1) * ovu].copy_from_slice(&input[in_base + nu..in_base + nu + ovu]);

        // Save unfiltered history for next frame's pitch search (from pre[])
        let pm_base = c * mpu;
        if nu > mpu {
            st.prefilter_mem[pm_base..pm_base + mpu].copy_from_slice(&pre[c][nu..nu + mpu]);
        } else {
            // Shift old memory left by N, append N new samples from pre[]
            // OPUS_MOVE: overlapping copy (src > dst, so forward copy is safe)
            let keep = mpu - nu;
            for i in 0..keep {
                st.prefilter_mem[pm_base + i] = st.prefilter_mem[pm_base + nu + i];
            }
            st.prefilter_mem[pm_base + keep..pm_base + mpu].copy_from_slice(&pre[c][mpu..mpu + nu]);
        }
    }

    *gain = gain1;
    *pitch_index = pitch_idx;
    *qgain = qg;
    pf_on
}

// ===========================================================================
// Encoder helper: compute_vbr
// ===========================================================================

/// Compute VBR target bits for the current frame.
///
/// Matches C `compute_vbr()` from `celt_encoder.c`.
fn compute_vbr(
    mode: &CELTMode,
    analysis: &AnalysisInfo,
    base_target: i32,
    lm: i32,
    bitrate: i32,
    last_coded_bands: i32,
    c: i32,
    intensity: i32,
    constrained_vbr: i32,
    stereo_saving: i32,
    tot_boost: i32,
    tf_estimate: i32,
    pitch_change: i32,
    max_depth: i32,
    lfe: i32,
    has_surround_mask: bool,
    _surround_masking: i32,
    temporal_vbr: i32,
) -> i32 {
    let nb_ebands = mode.nb_ebands;
    let e_bands = mode.ebands;

    let coded_bands = if last_coded_bands != 0 {
        last_coded_bands
    } else {
        nb_ebands
    };
    let mut coded_bins = (e_bands[coded_bands as usize] as i32) << lm;
    if c == 2 {
        coded_bins += (e_bands[imin(intensity, coded_bands) as usize] as i32) << lm;
    }

    let mut target = base_target;

    // Analysis-driven activity reduction (C celt_encoder.c:1630-1633).
    //   if (analysis->valid && analysis->activity<.4)
    //      target -= (opus_int32)((coded_bins<<BITRES)*(.4f-analysis->activity));
    // The f32 product casts to i32 with truncation toward zero, then the
    // `target -= …` is an integer subtraction.
    // Note: `.4` in C is a bare double literal; `float < double` promotes
    // to f64 for the comparison. The inner `(.4f-analysis->activity)` uses
    // a single-precision `.4f` literal — kept in f32 for that arithmetic.
    if analysis.valid != 0 && (analysis.activity as f64) < 0.4_f64 {
        let reduction = ((coded_bins << BITRES) as f32 * (0.4_f32 - analysis.activity)) as i32;
        target -= reduction;
    }

    // Stereo savings
    if c == 2 {
        let coded_stereo_bands = imin(intensity, coded_bands);
        let coded_stereo_dof =
            ((e_bands[coded_stereo_bands as usize] as i32) << lm) - coded_stereo_bands;
        // Maximum fraction of the bits we can save if the signal is mono.
        let max_frac = div32_16(mult16_16(qconst16(0.8, 15), coded_stereo_dof), coded_bins);
        let stereo_saving = min16(stereo_saving, qconst16(1.0, 8));
        target -= min32(
            mult16_32_q15(max_frac, target),
            shr32(
                mult16_16(stereo_saving - qconst16(0.1, 8), coded_stereo_dof << BITRES),
                8,
            ),
        );
    }

    // Boost the rate according to dynalloc (minus the dynalloc average for calibration).
    target += tot_boost - (19 << lm);

    // Apply transient boost, compensating for average boost.
    let tf_calibration = qconst16(0.044, 14);
    target += shl32(mult16_32_q15(tf_estimate - tf_calibration, target), 1);

    // Analysis-driven tonality boost (C celt_encoder.c:1655-1669).
    // Mirrors C exactly:
    //   tonal = MAX(0, analysis->tonality-.15f) - 0.12f;     // f32
    //   tonal_target = target + (opus_int32)((coded_bins<<BITRES)*1.2f*tonal);
    //   if (pitch_change) tonal_target += (opus_int32)((coded_bins<<BITRES)*.8f);
    // Each `(opus_int32)` cast truncates toward zero from f32, and each
    // addition is i32 + i32.
    if analysis.valid != 0 && lfe == 0 {
        let tonal = (analysis.tonality - 0.15_f32).max(0.0_f32) - 0.12_f32;
        let bits = (coded_bins << BITRES) as f32;
        let mut tonal_target = target + (bits * 1.2_f32 * tonal) as i32;
        if pitch_change != 0 {
            tonal_target += (bits * 0.8_f32) as i32;
        }
        target = tonal_target;
    }

    // Surround masking (not applicable for non-surround, but keep structure)
    // (skipped: has_surround_mask is false for standard encoding)

    // Floor depth: cap target based on signal energy
    {
        let bins = (e_bands[(nb_ebands - 2) as usize] as i32) << lm;
        let floor_depth = shr32(
            mult16_32_q15((c * bins) << BITRES, max_depth),
            DB_SHIFT - 15,
        );
        let floor_depth = imax(floor_depth, target >> 2);
        target = imin(target, floor_depth);
    }

    // Make VBR less aggressive for constrained VBR
    if (!has_surround_mask || lfe != 0) && constrained_vbr != 0 {
        target = base_target + mult16_32_q15(qconst16(0.67, 15), target - base_target);
    }

    // Temporal VBR adjustment
    if !has_surround_mask && tf_estimate < qconst16(0.2, 14) {
        let amount = mult16_16_q15(
            qconst16(0.0000031, 30),
            imax(0, imin(32000, 96000 - bitrate)),
        );
        let tvbr_factor = shr32(mult16_16(shr32(temporal_vbr, DB_SHIFT - 10), amount), 10);
        target += mult16_32_q15(tvbr_factor, target);
    }

    // Don't allow more than doubling the rate
    target = imin(2 * base_target, target);

    target
}

// ===========================================================================
// Main encode function
// ===========================================================================

/// Encode one frame of audio.
///
/// Matches C `celt_encode_with_ec()` from `celt_encoder.c`.
///
/// # Parameters
/// - `st`: Encoder state
/// - `pcm`: Input PCM samples (interleaved if stereo), i16
/// - `frame_size`: Samples per channel (before upsampling)
/// - `compressed`: Output buffer for encoded bytes
/// - `nb_compressed_bytes`: Maximum output size
/// - `enc`: Optional pre-initialized entropy encoder (for hybrid mode)
///
/// # Returns
/// Number of bytes written, or negative error code.
pub fn celt_encode_with_ec(
    st: &mut CeltEncoder,
    pcm: &[i16],
    frame_size: i32,
    compressed: &mut [u8],
    nb_compressed_bytes: i32,
    enc: Option<&mut RangeEncoder>,
) -> i32 {
    if let Some(e) = enc {
        celt_encode_core(st, pcm, frame_size, nb_compressed_bytes, e, false)
    } else {
        let mut local_enc = RangeEncoder::new(compressed);
        celt_encode_core(
            st,
            pcm,
            frame_size,
            nb_compressed_bytes,
            &mut local_enc,
            true,
        )
    }
}

fn celt_encode_core(
    st: &mut CeltEncoder,
    pcm: &[i16],
    frame_size: i32,
    nb_compressed_bytes: i32,
    enc_ref: &mut RangeEncoder,
    _enc_inited: bool,
) -> i32 {
    let mode = st.mode;
    let cc = st.channels;
    let c = st.stream_channels;
    let nb_ebands = mode.nb_ebands;
    let overlap = mode.overlap;
    let e_bands = mode.ebands;
    let start = st.start;
    let end = st.end;
    let hybrid = start != 0;

    // Validate inputs
    if nb_compressed_bytes < 2 || pcm.is_empty() {
        return OPUS_BAD_ARG;
    }

    let mut nb_compressed_bytes = nb_compressed_bytes.min(MAX_PACKET_BYTES);

    // Determine frame size and LM
    let frame_size = frame_size * st.upsample;
    let mut lm = -1i32;
    for l in 0..=mode.max_lm {
        if mode.short_mdct_size << l == frame_size {
            lm = l;
            break;
        }
    }
    if lm < 0 {
        return OPUS_BAD_ARG;
    }

    let m = 1 << lm;
    let n = m * mode.short_mdct_size;
    let eff_end = end.min(mode.eff_ebands);

    let tell0_frac = enc_ref.tell_frac() as i32;
    let mut tell = enc_ref.tell() as i32;
    let nb_filled_bytes = (tell + 4) >> 3;

    // VBR rate computation
    let mut vbr_rate = 0i32;
    let effective_bytes;
    if st.vbr != 0 && st.bitrate != OPUS_BITRATE_MAX {
        vbr_rate = bitrate_to_bits(st.bitrate, mode.fs, frame_size) << BITRES;
        effective_bytes = vbr_rate >> (3 + BITRES);
    } else {
        // Bug 12: CBR rate clamping (C lines 1910-1921)
        let mut tmp = st.bitrate * frame_size;
        if tell > 1 {
            tmp += tell * mode.fs;
        }
        if st.bitrate != OPUS_BITRATE_MAX {
            nb_compressed_bytes = nb_compressed_bytes
                .min((tmp + 4 * mode.fs) / (8 * mode.fs) - (st.signalling != 0) as i32)
                .max(2);
            enc_ref.shrink(nb_compressed_bytes as u32);
        }
        effective_bytes = nb_compressed_bytes - nb_filled_bytes;
    }
    let mut nb_available_bytes = nb_compressed_bytes - nb_filled_bytes;

    // Equivalent rate for analysis decisions (C lines 1925-1927, before CVBR clamping)
    let mut equiv_rate = (nb_compressed_bytes as i64 * 8 * 50 << (3 - lm)) as i32
        - (40 * c + 20) * ((400 >> lm) - 50);
    // Bug 4: equiv_rate clamping (C lines 1926-1927)
    if st.bitrate != OPUS_BITRATE_MAX {
        equiv_rate = equiv_rate.min(st.bitrate - (40 * c + 20) * ((400 >> lm) - 50));
    }

    // Bug 1: Constrained VBR early clamping (C lines 1935-1960)
    if vbr_rate > 0 && st.constrained_vbr != 0 {
        let vbr_bound = vbr_rate;
        let max_allowed = ((vbr_rate + vbr_bound - st.vbr_reservoir) >> (BITRES + 3))
            .max(if tell == 1 { 2 } else { 0 })
            .min(nb_available_bytes);
        if max_allowed < nb_available_bytes {
            nb_compressed_bytes = nb_filled_bytes + max_allowed;
            nb_available_bytes = max_allowed;
            enc_ref.shrink(nb_compressed_bytes as u32);
        }
    }
    // total_bits must be set after CVBR clamping (C line 1961)
    let mut total_bits = nb_compressed_bytes * 8;

    // Silence detection (Bug 11: overlap_max tracking, C lines 1969-1973)
    let mut silence = false;
    let pcm_as_i32: Vec<i32> = pcm.iter().map(|&x| x as i32).collect();
    let mut sample_max = st.overlap_max.max(celt_maxabs16(
        &pcm_as_i32[..(cc * (n - overlap) / st.upsample) as usize],
    ));
    st.overlap_max = celt_maxabs16(
        &pcm_as_i32[(cc * (n - overlap) / st.upsample) as usize..(cc * n / st.upsample) as usize],
    );
    sample_max = sample_max.max(st.overlap_max);
    if sample_max == 0 {
        silence = true;
    }

    // Encode silence flag (C: celt_encoder.c:1981-1984)
    if tell == 1 {
        enc_ref.encode_bit_logp(silence, 15);
        tell = enc_ref.tell() as i32;
    } else {
        silence = false;
    }

    // Handle silence (matches C: celt_encoder.c:1986-2008)
    if silence {
        if vbr_rate > 0 {
            // VBR: shrink to minimum packet
            nb_compressed_bytes = (nb_filled_bytes + 2).min(nb_compressed_bytes);
            total_bits = nb_compressed_bytes * 8;
            enc_ref.shrink(nb_compressed_bytes as u32);
            st.vbr_reservoir += nb_compressed_bytes * (8 << BITRES) - vbr_rate;
        }
        // Pretend we've filled all remaining bits with zeros
        // (that's what the initializer did anyway).
        // This prevents encoding actual data in the silence frame.
        tell = nb_compressed_bytes * 8;
        enc_ref.add_nbits_total(tell - enc_ref.tell());
    }

    // Pre-emphasis — max: 2 * (960 + 120) = 2160
    const MAX_INP: usize = 2 * (960 + 120);
    let mut input = [0i32; MAX_INP];
    for ch in 0..cc as usize {
        let inp_start = ch * (n as usize + overlap as usize) + overlap as usize;
        celt_preemphasis(
            pcm,
            ch,
            &mut input,
            inp_start,
            n,
            cc,
            st.upsample,
            &mode.preemph,
            &mut st.preemph_mem_e[ch],
            st.clip,
        );
        // Copy overlap from prefilter memory
        let mem_end = (ch + 1) * COMBFILTER_MAXPERIOD as usize;
        let ov_start = mem_end - overlap as usize;
        let in_start = ch * (n as usize + overlap as usize);
        for i in 0..overlap as usize {
            input[in_start + i] = st.prefilter_mem[ov_start + i];
        }
    }

    // Tone detection
    let mut toneishness: i32 = 0;
    let tone_freq = tone_detect(&input, cc, n + overlap, &mut toneishness, mode.fs);

    // Transient analysis
    let mut is_transient = false;
    let mut short_blocks = 0i32;
    let mut tf_estimate: i32 = 0;
    let mut tf_chan: i32 = 0;
    let mut weak_transient = false;
    let allow_weak_transients = hybrid && effective_bytes < 15 && st.silk_info.signal_type != 2;

    if st.complexity >= 1 && st.lfe == 0 {
        is_transient = transient_analysis(
            &input,
            n + overlap,
            cc,
            &mut tf_estimate,
            &mut tf_chan,
            allow_weak_transients,
            &mut weak_transient,
            tone_freq,
            toneishness,
        );
    }

    // Clamp toneishness (C: MIN32(toneishness, QCONST32(1.f,29)-SHL32(tf_estimate,15)))
    toneishness = toneishness.min(qconst32(1.0, 29) - shl32(tf_estimate, 15));

    // Pitch pre-filter
    let prefilter_enabled = ((st.lfe != 0 && nb_available_bytes > 3)
        || nb_available_bytes > 12 * c)
        && !hybrid
        && !silence
        && tell + 16 <= total_bits
        && st.disable_pf == 0;

    let mut pitch_index = COMBFILTER_MINPERIOD;
    let mut gain1: i32 = 0;
    let mut qg: i32 = 0;
    let mut prefilter_tapset = st.tapset_decision;
    let pitch_change;

    let pf_on = run_prefilter(
        st,
        &mut input,
        cc,
        n,
        &mut prefilter_tapset,
        &mut pitch_index,
        &mut gain1,
        &mut qg,
        prefilter_enabled,
        st.complexity,
        tf_estimate,
        nb_available_bytes,
        tone_freq,
        toneishness,
    );

    // C celt_encoder.c:2042-2044:
    //   pitch_change = (gain1>.4 || prefilter_gain>.4)
    //               && (!analysis.valid || analysis.tonality>.3)
    //               && (pitch_index > 1.26*prev || pitch_index < .79*prev);
    // The analysis gate prevents spurious pitch-change decisions when the
    // tonality analyzer says the current frame isn't actually tonal.
    // Note: `.3` in C is a bare double literal, so `tonality > .3` promotes
    // the f32 to f64 before comparison. Replicate that precisely — an f32
    // comparison against `0.3_f32` uses a slightly higher threshold (since
    // `0.3` rounds up in f32) and can flip the branch on borderline values.
    pitch_change = if (gain1 > qconst16(0.4, 15) || st.prefilter_gain > qconst16(0.4, 15))
        && (st.analysis.valid == 0 || (st.analysis.tonality as f64) > 0.3_f64)
        && (pitch_index as f64 > 1.26 * st.prefilter_period as f64
            || (pitch_index as f64) < 0.79 * st.prefilter_period as f64)
    {
        1
    } else {
        0
    };

    // Encode pre-filter parameters
    if pf_on == 0 {
        if !hybrid && tell + 16 <= total_bits {
            enc_ref.encode_bit_logp(false, 1);
        }
    } else {
        // Encode pitch pre-filter on
        enc_ref.encode_bit_logp(true, 1);
        // Encode pitch octave + fine period (C adds 1 before encoding, subtracts after)
        let pi_enc = pitch_index + 1;
        let octave = ec_ilog(pi_enc as u32) as i32 - 5;
        enc_ref.encode_uint(octave as u32, 6);
        enc_ref.encode_bits((pi_enc - (16 << octave)) as u32, (4 + octave) as u32);
        // Encode gain (3 bits)
        enc_ref.encode_bits(qg as u32, 3);
        // Encode tapset
        enc_ref.encode_icdf(prefilter_tapset as u32, &TAPSET_ICDF, 2);
    }

    // Transient flag
    let mut transient_got_disabled = false;
    if lm > 0 && enc_ref.tell() as i32 + 3 <= total_bits {
        if is_transient {
            short_blocks = m;
        }
    } else {
        is_transient = false;
        transient_got_disabled = true;
    }

    // MDCT computation — stack arrays with known max sizes
    let _freq_size = cc as usize * n as usize;
    let mut freq = [0i32; 2 * 960]; // max 2 channels * 960 samples
    let mut band_e = [0i32; 2 * NB_EBANDS]; // max 2 channels * 21 bands
    let mut band_log_e = [0i32; 2 * NB_EBANDS];
    let mut band_log_e2 = [0i32; 2 * NB_EBANDS];

    // Secondary MDCT for bandLogE2 (long block when transient)
    let second_mdct = short_blocks != 0 && st.complexity >= 8;
    if second_mdct {
        compute_mdcts(mode, 0, &input, &mut freq, c, cc, lm, st.upsample);
        compute_band_energies(mode, &freq, &mut band_e, eff_end, c, lm);
        amp2log2(mode, eff_end, end, &band_e, &mut band_log_e2, c);
        // LM compensation
        for ch in 0..c as usize {
            for i in 0..nb_ebands as usize {
                band_log_e2[ch * nb_ebands as usize + i] += half32(shl32(lm, DB_SHIFT));
            }
        }
    }

    // Main MDCT
    compute_mdcts(
        mode,
        short_blocks,
        &input,
        &mut freq,
        c,
        cc,
        lm,
        st.upsample,
    );
    compute_band_energies(mode, &freq, &mut band_e, eff_end, c, lm);

    // Reset tf_chan when stereo is downmixed to mono (C: celt_encoder.c)
    if cc == 2 && c == 1 {
        tf_chan = 0;
    }

    // LFE band limiting
    if st.lfe != 0 {
        for i in 2..end as usize {
            band_e[i] = band_e[i].min(mult16_32_q15(qconst16(1e-4, 15), band_e[0]));
            band_e[i] = band_e[i].max(EPSILON);
        }
    }

    amp2log2(mode, eff_end, end, &band_e, &mut band_log_e, c);

    if !second_mdct {
        band_log_e2[..c as usize * nb_ebands as usize]
            .copy_from_slice(&band_log_e[..c as usize * nb_ebands as usize]);
    }

    // Temporal VBR
    let mut temporal_vbr: i32 = 0;
    if st.lfe == 0 {
        let mut follow: i32 = -qconst32(10.0, (DB_SHIFT - 5) as u32);
        let mut frame_avg: i32 = 0;
        let offset = if short_blocks != 0 {
            half32(shl32(lm, DB_SHIFT - 5))
        } else {
            0
        };
        for i in start as usize..end as usize {
            // C: follow = MAX(follow - QCONST32(1.0, DB_SHIFT-5), SHR32(bandLogE[i],5) - offset)
            follow = (follow - (1 << (DB_SHIFT - 5))).max(shr32(band_log_e[i], 5) - offset);
            if c == 2 {
                follow = follow.max(shr32(band_log_e[i + nb_ebands as usize], 5) - offset);
            }
            frame_avg += follow;
        }
        if end > start {
            frame_avg /= (end - start) as i32;
        }
        temporal_vbr = shl32(frame_avg, 5) - st.spec_avg;
        temporal_vbr = temporal_vbr.max(-(3 << DB_SHIFT) / 2).min(3 << DB_SHIFT);
        st.spec_avg += mult16_32_q15(qconst16(0.02, 15), temporal_vbr);
    }

    // Patch transient decision
    if lm > 0
        && enc_ref.tell() as i32 + 3 <= total_bits
        && !is_transient
        && st.complexity >= 5
        && st.lfe == 0
        && !hybrid
    {
        if patch_transient_decision(&band_log_e, &st.old_band_e, nb_ebands, start, end, c) {
            is_transient = true;
            short_blocks = m;
            compute_mdcts(
                mode,
                short_blocks,
                &input,
                &mut freq,
                c,
                cc,
                lm,
                st.upsample,
            );
            compute_band_energies(mode, &freq, &mut band_e, eff_end, c, lm);
            amp2log2(mode, eff_end, end, &band_e, &mut band_log_e, c);
            for ch in 0..c as usize {
                for i in 0..end as usize {
                    band_log_e2[nb_ebands as usize * ch + i] += half32(shl32(lm, DB_SHIFT));
                }
            }
            tf_estimate = qconst16(0.2, 14);
        }
    }

    // Encode transient flag
    if lm > 0 && enc_ref.tell() as i32 + 3 <= total_bits {
        enc_ref.encode_bit_logp(is_transient, 3);
    }

    // Band normalization
    let _x_size = c as usize * n as usize;
    let mut x_norm = [0i32; 2 * 960];
    normalise_bands(mode, &freq, &mut x_norm, &band_e, eff_end, c, m);

    // TF analysis
    let enable_tf_analysis = effective_bytes >= 15 * c
        && !hybrid
        && st.complexity >= 2
        && st.lfe == 0
        // C: QCONST32(.98f, 29) = 526133504 (f32 precision; f64 gives 526133494)
        && toneishness < 526133504;

    let mut offsets = [0i32; NB_EBANDS];
    let mut importance = [13i32; NB_EBANDS];
    let mut spread_weight = [1i32; NB_EBANDS];
    let mut tot_boost = 0i32;

    let surround_dynalloc = [0i32; 2 * NB_EBANDS];

    let max_depth = dynalloc_analysis(
        &band_log_e,
        &band_log_e2,
        &st.old_band_e,
        nb_ebands,
        start,
        end,
        c,
        &mut offsets,
        st.lsb_depth,
        mode.log_n,
        is_transient,
        st.vbr,
        st.constrained_vbr,
        e_bands,
        lm,
        effective_bytes,
        &mut tot_boost,
        st.lfe,
        &surround_dynalloc,
        &st.analysis,
        &mut importance,
        &mut spread_weight,
        tone_freq,
        toneishness,
    );

    let mut tf_res = [0i32; NB_EBANDS];
    let tf_select;
    if enable_tf_analysis {
        let lambda = 80i32.max(20480 / effective_bytes.max(1) + 2);
        tf_select = tf_analysis(
            mode,
            eff_end,
            is_transient,
            &mut tf_res,
            lambda,
            &x_norm,
            n,
            lm,
            tf_estimate,
            tf_chan,
            &importance,
        );
        for i in eff_end as usize..end as usize {
            tf_res[i] = if eff_end > 0 {
                tf_res[eff_end as usize - 1]
            } else {
                0
            };
        }
    } else if hybrid && weak_transient {
        for i in start as usize..end as usize {
            tf_res[i] = 1;
        }
        tf_select = 0;
    } else if hybrid && effective_bytes < 15 && st.silk_info.signal_type != 2 {
        // For low bitrate hybrid, force temporal resolution to 5 ms rather than 2.5 ms.
        for i in start as usize..end as usize {
            tf_res[i] = 0;
        }
        tf_select = is_transient as i32;
    } else {
        for i in start as usize..end as usize {
            tf_res[i] = is_transient as i32;
        }
        tf_select = 0;
    }

    // Energy quantization: error biasing (C: celt_encoder.c:2289-2292)
    let mut error = [0i32; 2 * NB_EBANDS];
    for ch in 0..c as usize {
        for i in start as usize..end as usize {
            let idx = i + ch * nb_ebands as usize;
            if abs32(band_log_e[idx] - st.old_band_e[idx]) < (2 << DB_SHIFT) {
                band_log_e[idx] -= mult16_32_q15(qconst16(0.25, 15), st.energy_error[idx]);
            }
        }
    }

    // Coarse energy quantization
    quant_coarse_energy(
        mode,
        start,
        end,
        eff_end,
        &band_log_e,
        &mut st.old_band_e,
        total_bits as u32,
        &mut error,
        enc_ref,
        c,
        lm,
        nb_available_bytes,
        st.force_intra,
        &mut st.delayed_intra,
        (st.complexity >= 4) as i32,
        st.loss_rate,
        st.lfe,
    );

    // TF encoding
    tf_encode(
        start,
        end,
        is_transient,
        &mut tf_res,
        lm,
        tf_select,
        enc_ref,
    );
    tell = enc_ref.tell() as i32;

    // Spread decision
    if tell + 4 <= total_bits {
        if st.lfe != 0 {
            st.tapset_decision = 0;
            st.spread_decision = SPREAD_NORMAL;
        } else if hybrid {
            if st.complexity == 0 {
                st.spread_decision = SPREAD_NONE;
            } else if is_transient {
                st.spread_decision = SPREAD_NORMAL;
            } else {
                st.spread_decision = SPREAD_AGGRESSIVE;
            }
        } else if short_blocks != 0 || st.complexity < 3 || nb_available_bytes < 10 * c {
            if st.complexity == 0 {
                st.spread_decision = SPREAD_NONE;
            } else {
                st.spread_decision = SPREAD_NORMAL;
            }
        } else {
            st.spread_decision = spreading_decision(
                mode,
                &x_norm,
                &mut st.tonal_average,
                st.spread_decision,
                &mut st.hf_average,
                &mut st.tapset_decision,
                pf_on != 0 && short_blocks == 0,
                eff_end,
                c,
                m,
                &spread_weight,
            );
        }
        enc_ref.encode_icdf(st.spread_decision as u32, &SPREAD_ICDF, 5);
    } else {
        st.spread_decision = SPREAD_NORMAL;
    }

    // Caps initialization
    let mut cap = [0i32; NB_EBANDS];
    init_caps(mode, &mut cap, lm, c);

    // Dynamic allocation encoding
    let mut dynalloc_logp = 6i32;
    let total_bits_bitres = nb_compressed_bytes * 8 << BITRES;
    let mut total_boost_enc = 0i32;

    for i in start as usize..end as usize {
        let width = c * (e_bands[i + 1] - e_bands[i]) as i32 * m;
        let quanta = (width << BITRES).min((6 << BITRES).max(width));
        let mut dynalloc_loop_logp = dynalloc_logp;
        let mut boost = 0i32;

        let mut j = 0;
        while (enc_ref.tell_frac() as i32 + (dynalloc_loop_logp << BITRES))
            < total_bits_bitres - total_boost_enc
            && boost < cap[i]
        {
            let flag = j < offsets[i];
            enc_ref.encode_bit_logp(flag, dynalloc_loop_logp as u32);
            if !flag {
                break;
            }
            boost += quanta;
            total_boost_enc += quanta;
            dynalloc_loop_logp = 1;
            j += 1;
        }
        if j > 0 {
            dynalloc_logp = (dynalloc_logp - 1).max(2);
        }
        offsets[i] = boost;
    }

    // Stereo analysis
    let mut dual_stereo = 0i32;
    if c == 2 {
        if lm != 0 {
            dual_stereo = stereo_analysis(mode, &x_norm, lm, n);
        }
        st.intensity = hysteresis_decision(
            equiv_rate / 1000,
            &INTENSITY_THRESHOLDS,
            &INTENSITY_HISTERESIS,
            21,
            st.intensity,
        );
        st.intensity = st.intensity.max(start).min(end);
    }

    // Allocation trim
    let mut alloc_trim = 5i32;
    tell = enc_ref.tell_frac() as i32;
    if tell + (6 << BITRES) <= total_bits_bitres - total_boost_enc {
        if start > 0 || st.lfe != 0 {
            st.stereo_saving = 0;
            alloc_trim = 5;
        } else {
            alloc_trim = alloc_trim_analysis(
                mode,
                &x_norm,
                &band_log_e,
                end,
                lm,
                c,
                n,
                &st.analysis,
                &mut st.stereo_saving,
                tf_estimate,
                st.intensity,
                0,
                equiv_rate,
            );
        }
        enc_ref.encode_icdf(alloc_trim as u32, &TRIM_ICDF, 7);
    }

    // Minimum allowed bytes
    let mut min_allowed =
        ((enc_ref.tell_frac() as i32 + total_boost_enc + (1 << (BITRES + 3)) - 1) >> (BITRES + 3))
            + 2;
    // Take into account the 37 bits we need to have left in the packet to
    // signal a redundant frame in hybrid mode. Creating a shorter packet
    // would create an entropy coder desync between the encoder (which wrote
    // the redundancy bit under its `8*(max_data_bytes-1)` budget) and the
    // decoder (whose same check uses `8*len` against the final packet size).
    // Matches C `celt_encoder.c:2432-2433`.
    if hybrid {
        min_allowed = min_allowed.max(
            (tell0_frac + (37 << BITRES) + total_boost_enc + (1 << (BITRES + 3)) - 1)
                >> (BITRES + 3),
        );
    }

    // VBR rate control
    if vbr_rate > 0 {
        // Bug 5: packet_size_cap in VBR (C line 2445)
        // Don't attempt to use more than 510 kb/s, even for frames smaller than 20 ms.
        let packet_size_cap: i32 = 1275;
        nb_compressed_bytes = nb_compressed_bytes.min(packet_size_cap >> (3 - lm));

        let lm_diff = mode.max_lm - lm;
        let mut base_target = if !hybrid {
            vbr_rate - ((40 * c + 20) << BITRES)
        } else {
            (vbr_rate - ((9 * c + 4) << BITRES)).max(0)
        };

        // C: if (st->constrained_vbr) base_target += (st->vbr_offset>>lm_diff);
        if st.constrained_vbr != 0 {
            base_target += st.vbr_offset >> lm_diff;
        }

        // Bug 6: Hybrid VBR target (C lines 2456-2475)
        let mut target = if !hybrid {
            compute_vbr(
                mode,
                &st.analysis,
                base_target,
                lm,
                equiv_rate,
                st.last_coded_bands,
                c,
                st.intensity,
                st.constrained_vbr,
                st.stereo_saving,
                tot_boost,
                tf_estimate,
                pitch_change,
                max_depth,
                st.lfe,
                st.energy_mask.is_some(),
                0,
                temporal_vbr,
            )
        } else {
            let mut t = base_target;
            // Tonal frames need more bits than noisy ones
            if st.silk_info.offset < 100 {
                t += 12 << BITRES >> (3 - lm);
            }
            if st.silk_info.offset > 100 {
                t -= 18 << BITRES >> (3 - lm);
            }
            // Boosting bitrate on transients and vowels
            t += mult16_16_q14(tf_estimate - qconst16(0.25, 14), 50 << BITRES);
            // If strong transient, ensure enough bits for first two bands
            if tf_estimate > qconst16(0.7, 14) {
                t = t.max(50 << BITRES);
            }
            t
        };

        target += enc_ref.tell_frac() as i32;

        let mut nb_avail = (target + (1 << (BITRES + 2))) >> (BITRES + 3);
        nb_avail = nb_avail.max(min_allowed).min(nb_compressed_bytes);

        let mut delta = target - vbr_rate;

        // Bug 2: VBR silence override (C lines 2487-2499)
        target = nb_avail << (BITRES + 3);
        if silence {
            nb_avail = 2;
            target = 2 * 8 << BITRES;
            delta = 0;
        }

        if st.vbr_count < 970 {
            st.vbr_count += 1;
        }
        let alpha = if st.vbr_count < 970 {
            celt_rcp(shl32(st.vbr_count + 20, 16))
        } else {
            qconst16(0.001, 15)
        };

        // Update VBR state — uses target (possibly overridden for silence)
        if st.constrained_vbr != 0 {
            st.vbr_reservoir += target - vbr_rate;
        }

        if st.constrained_vbr != 0 {
            st.vbr_drift += mult16_32_q15(
                alpha,
                (delta * (1 << lm_diff)) - st.vbr_offset - st.vbr_drift,
            );
            st.vbr_offset = -st.vbr_drift;
        }

        if st.constrained_vbr != 0 && st.vbr_reservoir < 0 {
            let adjust = (-st.vbr_reservoir) / (8 << BITRES);
            nb_avail += if silence { 0 } else { adjust };
            st.vbr_reservoir = 0;
        }

        nb_compressed_bytes = nb_compressed_bytes.min(nb_avail);
        // Bug 3: ec_enc_shrink after VBR (C line 2532)
        enc_ref.shrink(nb_compressed_bytes as u32);
    }

    // Bit allocation
    let mut fine_quant = [0i32; NB_EBANDS];
    let mut pulses = [0i32; NB_EBANDS];
    let mut fine_priority = [0i32; NB_EBANDS];
    let mut balance = 0i32;

    let bits = ((nb_compressed_bytes as i64 * 8) << BITRES) as i32 - enc_ref.tell_frac() as i32 - 1;

    let anti_collapse_rsv = if is_transient && lm >= 2 && bits >= ((lm + 2) << BITRES) {
        1 << BITRES
    } else {
        0
    };
    let bits = bits - anti_collapse_rsv;

    let mut signal_bandwidth = end - 1;

    // Analysis-driven signal bandwidth floor (C celt_encoder.c:2607-2622).
    // When the tonality analyzer is confident, we pick `max(analysis.bandwidth,
    // min_bandwidth_for_rate)` as the last coded band for CELT allocation.
    // Rate thresholds are in bit-rate per channel.
    if st.analysis.valid != 0 {
        let min_bandwidth = if equiv_rate < 32000 * c {
            13
        } else if equiv_rate < 48000 * c {
            16
        } else if equiv_rate < 60000 * c {
            18
        } else if equiv_rate < 80000 * c {
            19
        } else {
            20
        };
        signal_bandwidth = imax(st.analysis.bandwidth, min_bandwidth);
    }
    if st.lfe != 0 {
        signal_bandwidth = 1;
    }

    let coded_bands = clt_compute_allocation(
        mode,
        start,
        end,
        &offsets,
        &cap,
        alloc_trim,
        &mut st.intensity,
        &mut dual_stereo,
        bits,
        &mut balance,
        &mut pulses,
        &mut fine_quant,
        &mut fine_priority,
        c,
        lm,
        enc_ref,
        true,
        st.last_coded_bands,
        signal_bandwidth,
    );

    // Update lastCodedBands with hysteresis (C: celt_encoder.c ~2631-2634)
    if st.last_coded_bands != 0 {
        st.last_coded_bands =
            (st.last_coded_bands + 1).min((st.last_coded_bands - 1).max(coded_bands));
    } else {
        st.last_coded_bands = coded_bands;
    }

    // Fine energy quantization
    quant_fine_energy(
        mode,
        start,
        end,
        &mut st.old_band_e,
        &mut error,
        None,
        &fine_quant,
        enc_ref,
        c,
    );

    // Clear energy error
    for i in 0..nb_ebands as usize * cc as usize {
        st.energy_error[i] = 0;
    }

    // Residual quantization
    let mut collapse_masks = [0u8; 2 * NB_EBANDS];
    let (x_norm_x, x_norm_y) = x_norm.split_at_mut(n as usize);
    let y_norm_opt = if c == 2 {
        Some(&mut x_norm_y[..])
    } else {
        None
    };
    quant_all_bands(
        true,
        mode,
        start,
        end,
        x_norm_x,
        y_norm_opt,
        &mut collapse_masks,
        &band_e,
        &mut pulses,
        short_blocks != 0,
        st.spread_decision,
        dual_stereo != 0,
        st.intensity,
        &tf_res,
        nb_compressed_bytes * (8 << BITRES) - anti_collapse_rsv,
        balance,
        enc_ref,
        lm,
        coded_bands,
        &mut st.rng,
        st.complexity,
        st.disable_inv != 0,
    );

    // Anti-collapse
    let anti_collapse_on;
    if anti_collapse_rsv > 0 {
        anti_collapse_on = (st.consec_transient < 2) as i32;
        enc_ref.encode_bits(anti_collapse_on as u32, 1);
    }

    // Energy finalization
    quant_energy_finalise(
        mode,
        start,
        end,
        Some(&mut st.old_band_e),
        &mut error,
        &fine_quant,
        &fine_priority,
        nb_compressed_bytes * 8 - enc_ref.tell() as i32,
        enc_ref,
        c,
    );

    // Store energy error for next frame
    for i in 0..nb_ebands as usize * cc as usize {
        st.energy_error[i] = error
            .get(i)
            .copied()
            .unwrap_or(0)
            .max(-(1 << (DB_SHIFT - 1)))
            .min(1 << (DB_SHIFT - 1));
    }

    // Update state
    st.prefilter_period = pitch_index;
    st.prefilter_gain = gain1;
    st.prefilter_tapset = prefilter_tapset;

    // Copy mono band energy to second channel if needed
    if cc == 2 && c == 1 {
        let nbu = nb_ebands as usize;
        for i in 0..nbu {
            st.old_band_e[nbu + i] = st.old_band_e[i];
        }
    }

    // Update log energy history
    let nbu = nb_ebands as usize;
    if !is_transient {
        for i in 0..cc as usize * nbu {
            st.old_log_e2[i] = st.old_log_e[i];
        }
        for i in 0..cc as usize * nbu {
            st.old_log_e[i] = st.old_band_e[i];
        }
    } else {
        for i in 0..cc as usize * nbu {
            st.old_log_e[i] = st.old_log_e[i].min(st.old_band_e[i]);
        }
    }

    // Clear out-of-range bands
    for ch in 0..cc as usize {
        for i in 0..start as usize {
            st.old_band_e[ch * nbu + i] = 0;
            st.old_log_e[ch * nbu + i] = GCONST_NEG28;
            st.old_log_e2[ch * nbu + i] = GCONST_NEG28;
        }
        for i in end as usize..nbu {
            st.old_band_e[ch * nbu + i] = 0;
            st.old_log_e[ch * nbu + i] = GCONST_NEG28;
            st.old_log_e2[ch * nbu + i] = GCONST_NEG28;
        }
    }

    // Reset old band energies for silence frames (C: celt_encoder.c)
    if silence {
        for i in 0..c as usize * nbu {
            st.old_band_e[i] = gconst(-28.0);
        }
    }

    // Update transient counter
    if is_transient || transient_got_disabled {
        st.consec_transient += 1;
    } else {
        st.consec_transient = 0;
    }

    // Save entropy coder state
    st.rng = enc_ref.get_rng();

    // Finalize bitstream
    enc_ref.done();
    if enc_ref.error() {
        return OPUS_INTERNAL_ERROR;
    }

    nb_compressed_bytes
}

// ===========================================================================
// Public API
// ===========================================================================

impl CeltEncoder {
    /// Create and initialize a new CELT encoder.
    ///
    /// # Parameters
    /// - `sampling_rate`: Sample rate (8000, 12000, 16000, 24000, or 48000)
    /// - `channels`: Number of channels (1 or 2)
    ///
    /// # Returns
    /// Initialized encoder, or `None` if parameters are invalid.
    pub fn new(sampling_rate: i32, channels: i32) -> Option<Self> {
        if channels < 1 || channels > 2 {
            return None;
        }

        let mode = &MODE_48000_960_120;
        let upsample = resampling_factor(sampling_rate);
        if upsample == 0 {
            return None;
        }

        let nb_ebands = mode.nb_ebands as usize;
        let overlap = mode.overlap as usize;
        let ch = channels as usize;

        let enc = CeltEncoder {
            mode,
            channels,
            stream_channels: channels,
            force_intra: 0,
            clip: 1,
            disable_pf: 0,
            complexity: 5,
            upsample,
            start: 0,
            end: mode.eff_ebands,
            bitrate: OPUS_BITRATE_MAX,
            vbr: 0,
            signalling: 1,
            constrained_vbr: 1,
            loss_rate: 0,
            lsb_depth: 24,
            lfe: 0,
            disable_inv: 0,

            rng: 0,
            spread_decision: SPREAD_NORMAL,
            delayed_intra: 1,
            tonal_average: 256,
            last_coded_bands: 0,
            hf_average: 0,
            tapset_decision: 0,
            prefilter_period: 0,
            prefilter_gain: 0,
            prefilter_tapset: 0,
            consec_transient: 0,
            analysis: AnalysisInfo::default(),
            silk_info: SILKInfo::default(),
            preemph_mem_e: [0; 2],
            preemph_mem_d: [0; 2],
            vbr_reservoir: 0,
            vbr_drift: 0,
            vbr_offset: 0,
            vbr_count: 0,
            overlap_max: 0,
            stereo_saving: 0,
            intensity: 0,
            energy_mask: None,
            spec_avg: 0,

            in_mem: vec![0; ch * overlap],
            prefilter_mem: vec![0; ch * COMBFILTER_MAXPERIOD as usize],
            old_band_e: vec![0; ch * nb_ebands],
            old_log_e: vec![GCONST_NEG28; ch * nb_ebands],
            old_log_e2: vec![GCONST_NEG28; ch * nb_ebands],
            energy_error: vec![0; ch * nb_ebands],
        };

        Some(enc)
    }

    /// Reset all encoder state to initial values.
    pub fn reset(&mut self) {
        let nb_ebands = self.mode.nb_ebands as usize;
        let ch = self.channels as usize;

        self.rng = 0;
        self.spread_decision = SPREAD_NORMAL;
        self.delayed_intra = 1;
        self.tonal_average = 256;
        self.last_coded_bands = 0;
        self.hf_average = 0;
        self.tapset_decision = 0;
        self.prefilter_period = 0;
        self.prefilter_gain = 0;
        self.prefilter_tapset = 0;
        self.consec_transient = 0;
        self.analysis = AnalysisInfo::default();
        self.silk_info = SILKInfo::default();
        self.preemph_mem_e = [0; 2];
        self.preemph_mem_d = [0; 2];
        self.vbr_reservoir = 0;
        self.vbr_drift = 0;
        self.vbr_offset = 0;
        self.vbr_count = 0;
        self.overlap_max = 0;
        self.stereo_saving = 0;
        self.intensity = 0;
        self.spec_avg = 0;

        self.in_mem.fill(0);
        self.prefilter_mem.fill(0);
        self.old_band_e.fill(0);
        for i in 0..ch * nb_ebands {
            self.old_log_e[i] = GCONST_NEG28;
            self.old_log_e2[i] = GCONST_NEG28;
        }
        self.energy_error.fill(0);
    }

    /// Apply a control request to the encoder.
    ///
    /// Returns `OPUS_OK` on success, or a negative error code.
    pub fn ctl(&mut self, request: CeltEncoderCtl) -> i32 {
        match request {
            CeltEncoderCtl::SetComplexity(v) => {
                if v < 0 || v > 10 {
                    return OPUS_BAD_ARG;
                }
                self.complexity = v;
            }
            CeltEncoderCtl::SetStartBand(v) => {
                if v < 0 || v >= self.mode.nb_ebands {
                    return OPUS_BAD_ARG;
                }
                self.start = v;
            }
            CeltEncoderCtl::SetEndBand(v) => {
                if v < 1 || v > self.mode.nb_ebands {
                    return OPUS_BAD_ARG;
                }
                self.end = v;
            }
            CeltEncoderCtl::SetPrediction(v) => {
                if v < 0 || v > 2 {
                    return OPUS_BAD_ARG;
                }
                self.disable_pf = (v <= 1) as i32;
                self.force_intra = (v == 0) as i32;
            }
            CeltEncoderCtl::SetPacketLossPerc(v) => {
                if v < 0 || v > 100 {
                    return OPUS_BAD_ARG;
                }
                self.loss_rate = v;
            }
            CeltEncoderCtl::SetVbrConstraint(v) => {
                self.constrained_vbr = v;
            }
            CeltEncoderCtl::SetVbr(v) => {
                self.vbr = v;
            }
            CeltEncoderCtl::SetBitrate(v) => {
                if v <= 500 && v != OPUS_BITRATE_MAX {
                    return OPUS_BAD_ARG;
                }
                self.bitrate = v.min(750000 * self.channels);
            }
            CeltEncoderCtl::SetChannels(v) => {
                if v < 1 || v > 2 {
                    return OPUS_BAD_ARG;
                }
                self.stream_channels = v;
            }
            CeltEncoderCtl::SetLsbDepth(v) => {
                if v < 8 || v > 24 {
                    return OPUS_BAD_ARG;
                }
                self.lsb_depth = v;
            }
            CeltEncoderCtl::GetLsbDepth => { /* caller reads st.lsb_depth */ }
            CeltEncoderCtl::SetPhaseInversionDisabled(v) => {
                if v < 0 || v > 1 {
                    return OPUS_BAD_ARG;
                }
                self.disable_inv = v;
            }
            CeltEncoderCtl::GetPhaseInversionDisabled => { /* caller reads st.disable_inv */ }
            CeltEncoderCtl::ResetState => {
                self.reset();
            }
            CeltEncoderCtl::SetAnalysis(info) => {
                self.analysis = info;
            }
            CeltEncoderCtl::SetSilkInfo(info) => {
                self.silk_info = info;
            }
            CeltEncoderCtl::SetSignalling(v) => {
                self.signalling = v;
            }
            CeltEncoderCtl::SetLfe(v) => {
                self.lfe = v;
            }
            CeltEncoderCtl::SetEnergyMask => { /* handled separately */ }
            CeltEncoderCtl::GetFinalRange => { /* caller reads st.rng */ }
            CeltEncoderCtl::SetInputClipping(v) => {
                self.clip = v;
            }
        }
        OPUS_OK
    }

    /// Get the final range of the entropy coder (for verification).
    pub fn final_range(&self) -> u32 {
        self.rng
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let enc = CeltEncoder::new(48000, 1);
        assert!(enc.is_some());
        let enc = enc.unwrap();
        assert_eq!(enc.channels, 1);
        assert_eq!(enc.upsample, 1);
        assert_eq!(enc.complexity, 5);
        assert_eq!(enc.lsb_depth, 24);
        assert_eq!(enc.spread_decision, SPREAD_NORMAL);
    }

    #[test]
    fn test_encoder_creation_stereo() {
        let enc = CeltEncoder::new(48000, 2);
        assert!(enc.is_some());
        let enc = enc.unwrap();
        assert_eq!(enc.channels, 2);
        assert_eq!(enc.stream_channels, 2);
    }

    #[test]
    fn test_encoder_creation_resampled() {
        let enc = CeltEncoder::new(16000, 1);
        assert!(enc.is_some());
        let enc = enc.unwrap();
        assert_eq!(enc.upsample, 3);
    }

    #[test]
    fn test_encoder_bad_params() {
        assert!(CeltEncoder::new(48000, 0).is_none());
        assert!(CeltEncoder::new(48000, 3).is_none());
        assert!(CeltEncoder::new(44100, 1).is_none());
    }

    #[test]
    fn test_encoder_reset() {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        enc.spread_decision = SPREAD_AGGRESSIVE;
        enc.vbr_count = 500;
        enc.reset();
        assert_eq!(enc.spread_decision, SPREAD_NORMAL);
        assert_eq!(enc.vbr_count, 0);
        assert_eq!(enc.delayed_intra, 1);
        assert_eq!(enc.tonal_average, 256);
        // Check log energies reset to -28 dB
        for &v in &enc.old_log_e {
            assert_eq!(v, GCONST_NEG28);
        }
    }

    #[test]
    fn test_encoder_ctl() {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        assert_eq!(enc.ctl(CeltEncoderCtl::SetComplexity(10)), OPUS_OK);
        assert_eq!(enc.complexity, 10);
        assert_eq!(enc.ctl(CeltEncoderCtl::SetComplexity(11)), OPUS_BAD_ARG);

        assert_eq!(enc.ctl(CeltEncoderCtl::SetBitrate(64000)), OPUS_OK);
        assert_eq!(enc.bitrate, 64000);

        assert_eq!(enc.ctl(CeltEncoderCtl::SetLsbDepth(16)), OPUS_OK);
        assert_eq!(enc.lsb_depth, 16);
        assert_eq!(enc.ctl(CeltEncoderCtl::SetLsbDepth(7)), OPUS_BAD_ARG);
    }

    #[test]
    fn test_encoder_ctl_branch_matrix_for_band_and_flag_settings() {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();

        assert_eq!(enc.ctl(CeltEncoderCtl::SetStartBand(-1)), OPUS_BAD_ARG);
        assert_eq!(
            enc.ctl(CeltEncoderCtl::SetStartBand(enc.mode.nb_ebands - 1)),
            OPUS_OK
        );
        assert_eq!(enc.start, enc.mode.nb_ebands - 1);

        assert_eq!(enc.ctl(CeltEncoderCtl::SetEndBand(0)), OPUS_BAD_ARG);
        assert_eq!(
            enc.ctl(CeltEncoderCtl::SetEndBand(enc.mode.nb_ebands)),
            OPUS_OK
        );
        assert_eq!(enc.end, enc.mode.nb_ebands);

        assert_eq!(enc.ctl(CeltEncoderCtl::SetPrediction(3)), OPUS_BAD_ARG);
        assert_eq!(enc.ctl(CeltEncoderCtl::SetPrediction(0)), OPUS_OK);
        assert_eq!(enc.disable_pf, 1);
        assert_eq!(enc.force_intra, 1);
        assert_eq!(enc.ctl(CeltEncoderCtl::SetPrediction(2)), OPUS_OK);
        assert_eq!(enc.disable_pf, 0);
        assert_eq!(enc.force_intra, 0);

        assert_eq!(enc.ctl(CeltEncoderCtl::SetPacketLossPerc(-1)), OPUS_BAD_ARG);
        assert_eq!(enc.ctl(CeltEncoderCtl::SetPacketLossPerc(15)), OPUS_OK);
        assert_eq!(enc.loss_rate, 15);

        assert_eq!(enc.ctl(CeltEncoderCtl::SetBitrate(500)), OPUS_BAD_ARG);
        assert_eq!(
            enc.ctl(CeltEncoderCtl::SetBitrate(OPUS_BITRATE_MAX)),
            OPUS_OK
        );
        assert_eq!(enc.bitrate, OPUS_BITRATE_MAX);

        assert_eq!(enc.ctl(CeltEncoderCtl::SetChannels(0)), OPUS_BAD_ARG);
        assert_eq!(enc.ctl(CeltEncoderCtl::SetChannels(2)), OPUS_OK);
        assert_eq!(enc.stream_channels, 2);

        assert_eq!(enc.ctl(CeltEncoderCtl::SetLsbDepth(25)), OPUS_BAD_ARG);
        assert_eq!(
            enc.ctl(CeltEncoderCtl::SetPhaseInversionDisabled(2)),
            OPUS_BAD_ARG
        );
        assert_eq!(
            enc.ctl(CeltEncoderCtl::SetPhaseInversionDisabled(1)),
            OPUS_OK
        );
        assert_eq!(enc.disable_inv, 1);
    }

    #[test]
    fn test_encoder_ctl_auxiliary_setters_and_noop_getters() {
        let mut enc = CeltEncoder::new(48000, 2).unwrap();
        let analysis = AnalysisInfo {
            valid: 1,
            tonality: 0.5,
            ..AnalysisInfo::default()
        };
        let silk_info = SILKInfo {
            signal_type: 2,
            ..SILKInfo::default()
        };

        enc.rng = 0x1234_5678;
        enc.vbr_count = 321;

        assert_eq!(enc.ctl(CeltEncoderCtl::GetLsbDepth), OPUS_OK);
        assert_eq!(enc.ctl(CeltEncoderCtl::GetPhaseInversionDisabled), OPUS_OK);
        assert_eq!(
            enc.ctl(CeltEncoderCtl::SetAnalysis(analysis.clone())),
            OPUS_OK
        );
        assert_eq!(enc.analysis.valid, analysis.valid);
        assert_eq!(enc.analysis.tonality, analysis.tonality);
        assert_eq!(
            enc.ctl(CeltEncoderCtl::SetSilkInfo(silk_info.clone())),
            OPUS_OK
        );
        assert_eq!(enc.silk_info.signal_type, silk_info.signal_type);
        assert_eq!(enc.ctl(CeltEncoderCtl::SetSignalling(0)), OPUS_OK);
        assert_eq!(enc.signalling, 0);
        assert_eq!(enc.ctl(CeltEncoderCtl::SetLfe(1)), OPUS_OK);
        assert_eq!(enc.lfe, 1);
        assert_eq!(enc.ctl(CeltEncoderCtl::SetEnergyMask), OPUS_OK);
        assert_eq!(enc.ctl(CeltEncoderCtl::GetFinalRange), OPUS_OK);
        assert_eq!(enc.final_range(), 0x1234_5678);
        assert_eq!(enc.ctl(CeltEncoderCtl::SetInputClipping(0)), OPUS_OK);
        assert_eq!(enc.clip, 0);

        assert_eq!(enc.ctl(CeltEncoderCtl::ResetState), OPUS_OK);
        assert_eq!(enc.vbr_count, 0);
        assert_eq!(enc.final_range(), 0);
    }

    #[test]
    fn test_median_of_5() {
        assert_eq!(median_of_5(&[1, 2, 3, 4, 5]), 3);
        assert_eq!(median_of_5(&[5, 4, 3, 2, 1]), 3);
        assert_eq!(median_of_5(&[1, 1, 1, 1, 1]), 1);
        assert_eq!(median_of_5(&[10, 20, 15, 5, 25]), 15);
    }

    #[test]
    fn test_median_of_3() {
        assert_eq!(median_of_3(1, 2, 3), 2);
        assert_eq!(median_of_3(3, 2, 1), 2);
        assert_eq!(median_of_3(1, 3, 2), 2);
        assert_eq!(median_of_3(5, 5, 5), 5);
    }

    #[test]
    fn test_inv_table() {
        assert_eq!(INV_TABLE.len(), 128);
        assert_eq!(INV_TABLE[0], 255);
        assert_eq!(INV_TABLE[127], 2);
    }

    #[test]
    fn test_comb_gains() {
        // Verify tapset 0 gain sum is approximately 1.0 in Q15
        let sum = COMB_GAINS[0][0] + 2 * COMB_GAINS[0][1] + 2 * COMB_GAINS[0][2];
        // Should be close to Q15ONE (32767)
        assert!(sum > 30000 && sum < 34000, "tapset 0 gain sum = {}", sum);
    }

    #[test]
    fn test_silence_encode() {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        enc.vbr = 1;
        enc.bitrate = 64000;

        // Encode silence (all zeros)
        let pcm = vec![0i16; 960];
        let mut compressed = vec![0u8; 1275];
        let result = celt_encode_with_ec(&mut enc, &pcm, 960, &mut compressed, 128, None);
        // Should succeed with a small number of bytes
        assert!(result > 0, "encode returned {}", result);
        assert!(result <= 128);
    }

    // -----------------------------------------------------------------------
    // patch_transient_decision tests
    // -----------------------------------------------------------------------

    /// Test with band_log_e values that exceed i16::MAX (the C reference bug).
    ///
    /// In the C reference, `x1` and `x2` are declared as `opus_val16` (int16_t).
    /// When Q24 log-energy values like 52363264 (0x031F0000) are assigned to
    /// int16_t, they truncate to 0x0000 = 0. After max(0, ..) this stays 0.
    /// This makes mean_diff = 0, so C returns false (no transient).
    ///
    /// Our code reproduces this truncation via `as i16`, so the test encodes
    /// the C behavior (false).
    #[test]
    fn test_patch_transient_large_values_should_match_c_reference() {
        // Real value from the failing packet: 52363264 = 0x031F0000 (Q24)
        // When truncated to i16: (int16_t)0x0000 = 0 -> stays 0
        let large_energy: i32 = 52363264;

        let nb_ebands: i32 = 21;
        let start: i32 = 0;
        let end: i32 = 17;
        let c: i32 = 1; // mono

        // band_log_e: all bands at the large energy value
        let band_log_e = vec![large_energy; nb_ebands as usize];
        // old_band_e: all bands at 0 (creating a large difference)
        let old_band_e = vec![0i32; nb_ebands as usize];

        let result = patch_transient_decision(&band_log_e, &old_band_e, nb_ebands, start, end, c);

        // C reference returns false because i16 truncation clamps x1 to 0
        assert!(
            !result,
            "patch_transient_decision returned true for large Q24 values; \
             should return false to match C reference i16 truncation behavior"
        );
    }

    /// Test where truncation changes the result: false WITH truncation, true WITHOUT.
    ///
    /// band_log_e = 50000000 (0x02FAF080), truncates to i16: 0xF080 = -3968
    /// old_band_e = 10000000 (0x00989680), truncates to i16: 0x9680 = -27008
    /// After truncation: x1 = -3968, x2 = -27008, diff = 23040 per band.
    /// mean_diff = 23040, well below threshold 16777216 → returns false.
    /// Without truncation: x1 = 50000000, x2 = 10000000, diff = 40000000
    /// → returns true. So truncation flips the result.
    #[test]
    fn test_patch_transient_truncation_flips_result() {
        let nb_ebands: i32 = 21;
        let start: i32 = 0;
        let end: i32 = 17;
        let c: i32 = 1; // mono

        // Both values exceed i16 range but have different low 16 bits
        let band_log_e = vec![50_000_000i32; nb_ebands as usize]; // 0x02FAF080
        let old_band_e = vec![10_000_000i32; nb_ebands as usize]; // 0x00989680

        let result = patch_transient_decision(&band_log_e, &old_band_e, nb_ebands, start, end, c);

        // With i16 truncation: mean_diff = 23040 < 16777216 → false
        // Without truncation: mean_diff = 40000000 > 16777216 → true
        assert!(
            !result,
            "patch_transient_decision returned true; i16 truncation should \
             suppress the large Q24 difference to below threshold"
        );
    }

    /// Test the i16 truncation boundary: 32767 survives, 32768 truncates to -32768.
    ///
    /// 32767 (i16::MAX) passes through truncation unchanged: x1 = 32767, x2 = 0.
    /// mean_diff = 32767, far below threshold 16777216 → false.
    /// 32768 (i16::MAX + 1) truncates to -32768, then max(0, x1-x2) = 0.
    /// mean_diff = 0 → false. A single increment causes a dramatic cliff.
    #[test]
    fn test_patch_transient_boundary_value_at_i16_max() {
        let nb_ebands: i32 = 21;
        let start: i32 = 0;
        let end: i32 = 17;
        let c: i32 = 1;

        let old_band_e = vec![0i32; nb_ebands as usize];

        // 32767: largest value that survives i16 truncation intact
        let band_log_e_max = vec![i16::MAX as i32; nb_ebands as usize];
        let result_max =
            patch_transient_decision(&band_log_e_max, &old_band_e, nb_ebands, start, end, c);
        // mean_diff = 32767, threshold = 16777216 → false
        assert!(!result_max, "32767 should not exceed threshold");

        // 32768: truncates to -32768, clamped to 0 by max(0, x1-x2)
        let band_log_e_overflow = vec![i16::MAX as i32 + 1; nb_ebands as usize];
        let result_overflow =
            patch_transient_decision(&band_log_e_overflow, &old_band_e, nb_ebands, start, end, c);
        // mean_diff = 0 → false (truncation cliff)
        assert!(
            !result_overflow,
            "32768 truncates to -32768; mean_diff should be 0"
        );
    }

    /// Test stereo (C=2) with large energy values — truncation applies per-channel.
    ///
    /// With C=2, band_log_e has 2*nbEBands entries and the channel loop runs
    /// twice. The same i16 truncation applies, so large values still produce
    /// mean_diff = 0.
    #[test]
    fn test_patch_transient_stereo_truncation() {
        let nb_ebands: i32 = 21;
        let start: i32 = 0;
        let end: i32 = 17;
        let c: i32 = 2; // stereo

        let large_energy: i32 = 52_363_264; // 0x031F0000, truncates to 0

        // Stereo: band_log_e needs 2 * nbEBands entries (ch0 then ch1)
        let band_log_e = vec![large_energy; 2 * nb_ebands as usize];
        let old_band_e = vec![0i32; 2 * nb_ebands as usize];

        let result = patch_transient_decision(&band_log_e, &old_band_e, nb_ebands, start, end, c);

        // Both channels: x1 truncates to 0, x2 = 0, mean_diff = 0 → false
        assert!(
            !result,
            "stereo patch_transient_decision should return false; \
             i16 truncation zeroes both channels"
        );
    }

    // -----------------------------------------------------------------------
    // Coverage improvement: encode-path tests
    // -----------------------------------------------------------------------

    /// Helper: encode a frame and return compressed size.
    fn encode_frame(enc: &mut CeltEncoder, pcm: &[i16], frame_size: i32, max_bytes: i32) -> i32 {
        let mut compressed = vec![0u8; max_bytes as usize];
        celt_encode_with_ec(enc, pcm, frame_size, &mut compressed, max_bytes, None)
    }

    /// Helper: generate patterned PCM for a given frame size and channels.
    fn gen_pcm(frame_size: usize, channels: usize, seed: i32) -> Vec<i16> {
        (0..frame_size * channels)
            .map(|i| (((i as i32 * 7919 + seed * 911) % 28000) - 14000) as i16)
            .collect()
    }

    #[test]
    fn test_encode_cvbr_clamping_low_bitrate() {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        enc.vbr = 1;
        enc.constrained_vbr = 1;
        enc.bitrate = 12000; // Very low → CVBR clamping kicks in
        enc.complexity = 5;
        // Drain reservoir by encoding several frames
        for i in 0..8 {
            let pcm = gen_pcm(960, 1, i);
            let ret = encode_frame(&mut enc, &pcm, 960, 128);
            assert!(ret > 0, "frame {} encode failed: {}", i, ret);
        }
    }

    #[test]
    fn test_encode_cvbr_reservoir_depleted() {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        enc.vbr = 1;
        enc.constrained_vbr = 1;
        enc.bitrate = 8000; // Very tight
        enc.vbr_reservoir = -5000; // Force negative reservoir
        let pcm = gen_pcm(960, 1, 42);
        let ret = encode_frame(&mut enc, &pcm, 960, 128);
        assert!(ret > 0);
    }

    #[test]
    fn test_encode_silence_vbr_shrinks_packet() {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        enc.vbr = 1;
        enc.bitrate = 64000;
        let pcm = vec![0i16; 960]; // Pure silence
        let ret = encode_frame(&mut enc, &pcm, 960, 128);
        assert!(
            ret > 0 && ret <= 4,
            "silence VBR should produce tiny packet, got {}",
            ret
        );
    }

    #[test]
    fn test_encode_cbr_with_signalling() {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        enc.vbr = 0;
        enc.signalling = 1;
        enc.bitrate = 64000;
        let pcm = gen_pcm(960, 1, 7);
        let ret = encode_frame(&mut enc, &pcm, 960, 128);
        assert!(ret > 0);
    }

    #[test]
    fn test_encode_cbr_without_signalling() {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        enc.vbr = 0;
        enc.signalling = 0;
        enc.bitrate = 64000;
        let pcm = gen_pcm(960, 1, 13);
        let ret = encode_frame(&mut enc, &pcm, 960, 128);
        assert!(ret > 0);
    }

    #[test]
    fn test_encode_cbr_max_bitrate() {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        enc.vbr = 0;
        enc.bitrate = OPUS_BITRATE_MAX;
        let pcm = gen_pcm(960, 1, 99);
        let ret = encode_frame(&mut enc, &pcm, 960, 1275);
        assert!(ret > 0);
    }

    #[test]
    fn test_encode_transient_impulse() {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        enc.vbr = 1;
        enc.bitrate = 128000;
        enc.complexity = 5;
        // Create impulse: loud burst then silence
        let mut pcm = vec![0i16; 960];
        for i in 0..48 {
            pcm[i] = 30000;
        }
        let ret = encode_frame(&mut enc, &pcm, 960, 256);
        assert!(ret > 0);
    }

    #[test]
    fn test_encode_short_blocks_consecutive_transients() {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        enc.vbr = 1;
        enc.bitrate = 128000;
        enc.complexity = 10;
        // Encode multiple frames with sharp transients
        for frame_idx in 0..5 {
            let mut pcm = vec![0i16; 960];
            let start = (frame_idx * 37) % 900;
            for i in start..start.min(959) + 30 {
                pcm[i] = if i % 2 == 0 { 32000 } else { -32000 };
            }
            let ret = encode_frame(&mut enc, &pcm, 960, 256);
            assert!(ret > 0, "frame {} failed: {}", frame_idx, ret);
        }
    }

    #[test]
    fn test_encode_stereo_low_bitrate_intensity() {
        let mut enc = CeltEncoder::new(48000, 2).unwrap();
        enc.vbr = 1;
        enc.bitrate = 24000; // Low enough to trigger intensity stereo
        enc.complexity = 5;
        let pcm = gen_pcm(960, 2, 21);
        let ret = encode_frame(&mut enc, &pcm, 960, 64);
        assert!(ret > 0);
    }

    #[test]
    fn test_encode_stereo_high_bitrate_dual() {
        let mut enc = CeltEncoder::new(48000, 2).unwrap();
        enc.vbr = 1;
        enc.bitrate = 256000; // High enough for dual stereo
        enc.complexity = 10;
        let pcm = gen_pcm(960, 2, 33);
        let ret = encode_frame(&mut enc, &pcm, 960, 512);
        assert!(ret > 0);
    }

    #[test]
    fn test_encode_very_low_bitrate() {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        enc.vbr = 0;
        enc.bitrate = 6000;
        let pcm = gen_pcm(960, 1, 55);
        let ret = encode_frame(&mut enc, &pcm, 960, 16);
        assert!(ret > 0);
    }

    #[test]
    fn test_encode_minimum_packet_two_bytes() {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        enc.vbr = 0;
        enc.bitrate = 6000;
        let pcm = gen_pcm(960, 1, 66);
        let ret = encode_frame(&mut enc, &pcm, 960, 2);
        assert!(ret >= 2);
    }

    #[test]
    fn test_encode_lfe_mode_via_opus() {
        // LFE mode is set by the multistream encoder, not directly on CeltEncoder.
        // Test that setting lfe=1 + end=1 (narrowband) works:
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        enc.vbr = 0;
        enc.bitrate = 32000;
        enc.lfe = 1;
        enc.end = 1; // Restrict to LFE band
        enc.complexity = 0;
        let pcm: Vec<i16> = (0..960)
            .map(|i| ((i as f64 * 0.02).sin() * 5000.0) as i16)
            .collect();
        let ret = encode_frame(&mut enc, &pcm, 960, 64);
        assert!(ret > 0);
    }

    #[test]
    fn test_encode_complexity_zero() {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        enc.vbr = 1;
        enc.bitrate = 64000;
        enc.complexity = 0;
        let pcm = gen_pcm(960, 1, 88);
        let ret = encode_frame(&mut enc, &pcm, 960, 128);
        assert!(ret > 0);
    }

    #[test]
    fn test_encode_complexity_ten() {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        enc.vbr = 1;
        enc.bitrate = 64000;
        enc.complexity = 10;
        let pcm = gen_pcm(960, 1, 99);
        let ret = encode_frame(&mut enc, &pcm, 960, 128);
        assert!(ret > 0);
    }

    #[test]
    fn test_encode_bad_frame_size() {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        enc.bitrate = 64000;
        let pcm = gen_pcm(960, 1, 1);
        // frame_size=100 doesn't match any valid LM
        let ret = encode_frame(&mut enc, &pcm, 100, 128);
        assert_eq!(ret, OPUS_BAD_ARG);
    }

    #[test]
    fn test_encode_too_small_buffer() {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        enc.bitrate = 64000;
        let pcm = gen_pcm(960, 1, 1);
        let ret = encode_frame(&mut enc, &pcm, 960, 1);
        assert_eq!(ret, OPUS_BAD_ARG);
    }

    #[test]
    fn test_encode_phase_inversion_disabled() {
        let mut enc = CeltEncoder::new(48000, 2).unwrap();
        enc.vbr = 1;
        enc.bitrate = 128000;
        enc.disable_inv = 1;
        let pcm = gen_pcm(960, 2, 44);
        let ret = encode_frame(&mut enc, &pcm, 960, 256);
        assert!(ret > 0);
    }

    #[test]
    fn test_encode_240_frame_size() {
        // 5ms frame (LM=0 at 48kHz)
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        enc.vbr = 1;
        enc.bitrate = 64000;
        let pcm = gen_pcm(240, 1, 11);
        let ret = encode_frame(&mut enc, &pcm, 240, 128);
        assert!(ret > 0);
    }

    #[test]
    fn test_encode_480_frame_size() {
        // 10ms frame
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        enc.vbr = 1;
        enc.bitrate = 64000;
        let pcm = gen_pcm(480, 1, 22);
        let ret = encode_frame(&mut enc, &pcm, 480, 128);
        assert!(ret > 0);
    }

    #[test]
    fn test_encode_multiple_frames_vbr_reservoir() {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        enc.vbr = 1;
        enc.constrained_vbr = 1;
        enc.bitrate = 48000;
        // Encode enough frames to exercise the reservoir dynamics
        for i in 0..20 {
            let pcm = gen_pcm(960, 1, i * 7);
            let ret = encode_frame(&mut enc, &pcm, 960, 128);
            assert!(ret > 0, "frame {} failed: {}", i, ret);
        }
    }

    #[test]
    fn test_encode_prediction_disabled() {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        enc.vbr = 1;
        enc.bitrate = 64000;
        enc.disable_pf = 1;
        enc.force_intra = 1;
        let pcm = gen_pcm(960, 1, 33);
        let ret = encode_frame(&mut enc, &pcm, 960, 128);
        assert!(ret > 0);
    }

    #[test]
    fn test_encode_with_loss_rate() {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        enc.vbr = 1;
        enc.bitrate = 64000;
        enc.loss_rate = 25;
        let pcm = gen_pcm(960, 1, 44);
        let ret = encode_frame(&mut enc, &pcm, 960, 128);
        assert!(ret > 0);
    }

    #[test]
    fn test_encode_16khz_upsampled() {
        let mut enc = CeltEncoder::new(16000, 1).unwrap();
        enc.vbr = 1;
        enc.bitrate = 32000;
        let pcm = gen_pcm(320, 1, 55);
        let ret = encode_frame(&mut enc, &pcm, 320, 64);
        assert!(ret > 0);
    }

    #[test]
    fn test_encode_stereo_silence_vbr() {
        let mut enc = CeltEncoder::new(48000, 2).unwrap();
        enc.vbr = 1;
        enc.bitrate = 64000;
        let pcm = vec![0i16; 960 * 2];
        let ret = encode_frame(&mut enc, &pcm, 960, 128);
        assert!(ret > 0);
    }

    #[test]
    fn test_encode_hybrid_mode_start_band() {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        enc.vbr = 1;
        enc.bitrate = 64000;
        enc.start = 17; // Non-zero start → hybrid mode
        let pcm = gen_pcm(960, 1, 77);
        let ret = encode_frame(&mut enc, &pcm, 960, 128);
        assert!(ret > 0);
    }

    // ===================================================================
    // Mutation-killing pinning tests for pure helper functions
    // ===================================================================

    #[test]
    fn test_pin_acos_approx() {
        // acos_approx: Q29 input -> Q13 output (radians)
        // x=0 -> pi/2 in Q13
        assert_eq!(acos_approx(0), 12862);
        // x near +1 -> acos near 0
        assert_eq!(acos_approx((1 << 29) - 1), 0);
        // x near -1 -> acos near pi in Q13
        assert_eq!(acos_approx(-((1 << 29) - 1)), 25736);
        // x=0.5 (Q29) -> acos(0.5) near pi/3
        assert_eq!(acos_approx(1 << 28), 8578);
        // x=-0.5 -> acos(-0.5) near 2*pi/3
        assert_eq!(acos_approx(-(1 << 28)), 17158);
        // x=0.25 (Q29)
        assert_eq!(acos_approx(1 << 27), 10802);
        // x=-0.25
        assert_eq!(acos_approx(-(1 << 27)), 14934);
    }

    #[test]
    fn test_pin_celt_preemphasis_fast_path() {
        // celt_preemphasis: upsample=1 (fast path), mono, 8 samples.
        // Standard Opus pre-emphasis coef: 0.85 in Q15 = 27853.
        // Filter: inp[i] = x[i] - m; m = MULT16_32_Q15(coef0, x[i]) after each
        // iter, so saved mem = coef0 * x_last / 2^15 (NOT x_last itself).
        let coef: [i32; 4] = [qconst16(0.85, 15), 0, 0, 0];
        let pcm: Vec<i16> = vec![1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000];
        let mut inp = vec![0i32; 8];
        let mut mem: i32 = 0;

        celt_preemphasis(&pcm, 0, &mut inp, 0, 8, 1, 1, &coef, &mut mem, 0);

        // Sample-level outputs unchanged; only `mem` scaling differs.
        assert_eq!(inp[0], 4096000); // 1000 << 12 = 4096000, mem was 0
        assert_eq!(inp[1], 4710375);
        assert_eq!(inp[2], 5324750);
        assert_eq!(inp[3], 5939125);
        assert_eq!(inp[4], 6553500);
        assert_eq!(inp[5], 7167875);
        assert_eq!(inp[6], 7782250);
        assert_eq!(inp[7], 8396625);
        // mem = MULT16_32_Q15(27853, 8000<<12) = 27853 * 1000 = 27_853_000
        assert_eq!(mem, 27_853_000);
    }

    #[test]
    fn test_pin_celt_preemphasis_upsample() {
        // celt_preemphasis: upsample=2 path, mono, 8 output samples (4 PCM samples).
        // Every other output gets a real sample; the rest are zero-inserted.
        // Filter: inp[i] = x - m; m = MULT16_32_Q15(coef0, x) — applies even
        // when the current x is a zero-insert, so mem ends up scaled by coef0.
        let coef: [i32; 4] = [qconst16(0.85, 15), 0, 0, 0];
        let pcm: Vec<i16> = vec![1000, 2000, 3000, 4000];
        let mut inp = vec![0i32; 8];
        let mut mem: i32 = 0;

        celt_preemphasis(&pcm, 0, &mut inp, 0, 8, 1, 2, &coef, &mut mem, 0);

        // Even indices get real samples, odd indices get zero-inserted
        assert_eq!(inp[0], 4096000); // 1000 << 12, mem=0
        assert_eq!(inp[1], -3481625); // 0 - coef * 4096000 (old mem after iter 0)
        assert_eq!(inp[2], 8192000); // 2000 << 12, mem was 0 (post zero-insert)
        assert_eq!(inp[3], -6963250); // 0 - coef * 8192000
        assert_eq!(inp[4], 12288000); // 3000 << 12
        assert_eq!(inp[5], -10444875); // 0 - coef * 12288000
        assert_eq!(inp[6], 16384000); // 4000 << 12
        assert_eq!(inp[7], -13926500); // 0 - coef * 16384000
        // Final m = MULT16_32_Q15(coef0, 0) = 0 (last iter's x is a zero-insert)
        assert_eq!(mem, 0);
    }

    #[test]
    fn test_pin_normalize_tone_input() {
        // Test 1: large-amplitude signal -> shifted down by 1 (10000 >> 1 = 5000)
        let mut large: Vec<i16> = vec![10000, -10000, 10000, -10000, 10000, -10000, 10000, -10000];
        let len = large.len();
        normalize_tone_input(&mut large, len);
        assert_eq!(large, [5000, -5000, 5000, -5000, 5000, -5000, 5000, -5000]);

        // Test 2: small-amplitude signal -> shift=0, unchanged
        let mut small: Vec<i16> = vec![1, -1, 1, -1, 1, -1, 1, -1];
        let len = small.len();
        normalize_tone_input(&mut small, len);
        assert_eq!(small, [1, -1, 1, -1, 1, -1, 1, -1]);
    }

    #[test]
    fn test_pin_l1_metric() {
        // l1_metric: sums |tmp[i] >> (NORM_SHIFT-14)| then applies bias
        // NORM_SHIFT = 24, so each element is shifted right by 10
        let tmp: Vec<i32> = vec![1 << 20, -(1 << 20), 1 << 19, -(1 << 19)];

        // lm=0, bias=0: pure L1 sum, no frequency bonus
        assert_eq!(l1_metric(&tmp, 4, 0, 0), 3072);
        // lm=2, bias=qconst16(0.04,14): frequency bias applied (lm*bias * l1 >> 15)
        assert_eq!(l1_metric(&tmp, 4, 2, qconst16(0.04, 14)), 3194);
        // lm=0 with nonzero bias: bias term is lm*bias=0, so same as no bias
        assert_eq!(l1_metric(&tmp, 4, 0, qconst16(0.04, 14)), 3072);
    }

    #[test]
    fn test_pin_compute_vbr_mono() {
        let mode = &MODE_48000_960_120;
        let analysis = AnalysisInfo::default();
        let result = compute_vbr(
            mode,
            &analysis,
            6400,              // base_target
            3,                 // lm
            64000,             // bitrate
            0,                 // last_coded_bands
            1,                 // c (mono)
            0,                 // intensity
            0,                 // constrained_vbr
            0,                 // stereo_saving
            0,                 // tot_boost
            0,                 // tf_estimate
            0,                 // pitch_change
            qconst16(0.5, 15), // max_depth
            0,                 // lfe
            false,             // has_surround_mask
            0,                 // surround_masking
            0,                 // temporal_vbr
        );
        // Mono VBR target with default parameters
        assert_eq!(result, 1493);
    }

    #[test]
    fn test_pin_compute_vbr_stereo() {
        let mode = &MODE_48000_960_120;
        let analysis = AnalysisInfo::default();
        let result = compute_vbr(
            mode,
            &analysis,
            6400,              // base_target
            3,                 // lm
            64000,             // bitrate
            0,                 // last_coded_bands
            2,                 // c (stereo)
            10,                // intensity
            1,                 // constrained_vbr
            128,               // stereo_saving (Q8)
            0,                 // tot_boost
            0,                 // tf_estimate
            0,                 // pitch_change
            qconst16(0.5, 15), // max_depth
            0,                 // lfe
            false,             // has_surround_mask
            0,                 // surround_masking
            0,                 // temporal_vbr
        );
        // Stereo with intensity=10, constrained VBR, stereo savings
        assert_eq!(result, 3068);
    }

    #[test]
    fn test_pin_transient_analysis_silence() {
        // Silence input: all zeros, len=960, mono
        let input = vec![0i32; 960];
        let mut tf_estimate: i32 = 0;
        let mut tf_chan: i32 = 0;
        let mut weak_transient = false;

        let is_transient = transient_analysis(
            &input,
            960,
            1,
            &mut tf_estimate,
            &mut tf_chan,
            false,
            &mut weak_transient,
            0,
            0,
        );
        // Silence: no transient, all outputs zero
        assert!(!is_transient);
        assert_eq!(tf_estimate, 0);
        assert_eq!(tf_chan, 0);
        assert!(!weak_transient);
    }

    #[test]
    fn test_pin_transient_analysis_impulse() {
        // Sharp impulse: first 48 samples = 1 << SIG_SHIFT, rest = 0
        let mut input = vec![0i32; 960];
        for i in 0..48 {
            input[i] = 1 << SIG_SHIFT;
        }
        let mut tf_estimate: i32 = 0;
        let mut tf_chan: i32 = 0;
        let mut weak_transient = false;

        let is_transient = transient_analysis(
            &input,
            960,
            1,
            &mut tf_estimate,
            &mut tf_chan,
            false,
            &mut weak_transient,
            0,
            0,
        );
        // Sharp impulse: detected as transient
        assert!(is_transient);
        assert_eq!(tf_estimate, 16262);
        assert_eq!(tf_chan, 0);
        assert!(!weak_transient);
    }

    #[test]
    fn test_pin_tone_lpc_degenerate() {
        // Test 1: constant signal -> degenerate (returns true = fail)
        let constant_signal: Vec<i16> = vec![1000; 200];
        let mut lpc = [0i32; 2];
        let fail = tone_lpc(&constant_signal, 200, 10, &mut lpc);
        assert!(fail); // degenerate: den <= shr32(r00*r11, 10)
        assert_eq!(lpc[0], 0); // LPC untouched on failure
        assert_eq!(lpc[1], 0);

        // Test 2: cosine signal (period=30, delay=5) -> non-degenerate
        let mut cosine_signal: Vec<i16> = vec![0; 200];
        let period = 30.0f64;
        for i in 0..200 {
            cosine_signal[i] =
                (1000.0 * (2.0 * std::f64::consts::PI * i as f64 / period).cos()) as i16;
        }
        let mut lpc2 = [0i32; 2];
        let fail2 = tone_lpc(&cosine_signal, 200, 5, &mut lpc2);
        assert!(!fail2); // success
        assert_eq!(lpc2[0], 536870782); // near Q29(1.0) = 536870912
        assert_eq!(lpc2[1], -536870783); // near -Q29(1.0)

        // Test 3: cosine (period=40, delay=7) -> non-degenerate, different LPC
        let mut cosine3: Vec<i16> = vec![0; 200];
        let period3 = 40.0f64;
        for i in 0..200 {
            cosine3[i] = (1000.0 * (2.0 * std::f64::consts::PI * i as f64 / period3).cos()) as i16;
        }
        let mut lpc3 = [0i32; 2];
        let fail3 = tone_lpc(&cosine3, 200, 7, &mut lpc3);
        assert!(!fail3);
        assert_eq!(lpc3[0], 487471184);
        assert_eq!(lpc3[1], -536870452);
    }

    // -----------------------------------------------------------------------
    // Stage 3 branch-coverage additions
    // -----------------------------------------------------------------------
    mod branch_coverage_stage3 {
        use super::*;

        /// Helper to generate a sinusoidal signal that exercises tonal paths.
        fn gen_sine(frame_size: usize, channels: usize, freq_hz: f64) -> Vec<i16> {
            let mut out = vec![0i16; frame_size * channels];
            let sr = 48000.0;
            for i in 0..frame_size {
                let s =
                    (8000.0 * (2.0 * std::f64::consts::PI * freq_hz * i as f64 / sr).sin()) as i16;
                for c in 0..channels {
                    out[i * channels + c] = s;
                }
            }
            out
        }

        /// gen_pcm variant matching the outer helper (copied since we can't
        /// access the parent fn from a nested mod).
        fn gen_pcm(frame_size: usize, channels: usize, seed: i32) -> Vec<i16> {
            (0..frame_size * channels)
                .map(|i| (((i as i32 * 7919 + seed * 911) % 28000) - 14000) as i16)
                .collect()
        }

        fn encode_frame(
            enc: &mut CeltEncoder,
            pcm: &[i16],
            frame_size: i32,
            max_bytes: i32,
        ) -> i32 {
            let mut compressed = vec![0u8; max_bytes as usize];
            celt_encode_with_ec(enc, pcm, frame_size, &mut compressed, max_bytes, None)
        }

        // ---------------------------------------------------------------
        // CTL setters — explicit coverage of valid / invalid branches
        // ---------------------------------------------------------------

        #[test]
        fn ctl_set_prediction_all_values() {
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            // prediction=0: disable_pf=1, force_intra=1
            assert_eq!(enc.ctl(CeltEncoderCtl::SetPrediction(0)), OPUS_OK);
            assert_eq!(enc.disable_pf, 1);
            assert_eq!(enc.force_intra, 1);
            // prediction=1: disable_pf=1, force_intra=0
            assert_eq!(enc.ctl(CeltEncoderCtl::SetPrediction(1)), OPUS_OK);
            assert_eq!(enc.disable_pf, 1);
            assert_eq!(enc.force_intra, 0);
            // prediction=2: disable_pf=0, force_intra=0
            assert_eq!(enc.ctl(CeltEncoderCtl::SetPrediction(2)), OPUS_OK);
            assert_eq!(enc.disable_pf, 0);
            assert_eq!(enc.force_intra, 0);
            // Out-of-range values
            assert_eq!(enc.ctl(CeltEncoderCtl::SetPrediction(-1)), OPUS_BAD_ARG);
            assert_eq!(enc.ctl(CeltEncoderCtl::SetPrediction(3)), OPUS_BAD_ARG);
        }

        #[test]
        fn ctl_set_complexity_edges() {
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            assert_eq!(enc.ctl(CeltEncoderCtl::SetComplexity(0)), OPUS_OK);
            assert_eq!(enc.complexity, 0);
            assert_eq!(enc.ctl(CeltEncoderCtl::SetComplexity(10)), OPUS_OK);
            assert_eq!(enc.complexity, 10);
            assert_eq!(enc.ctl(CeltEncoderCtl::SetComplexity(-1)), OPUS_BAD_ARG);
            assert_eq!(enc.ctl(CeltEncoderCtl::SetComplexity(11)), OPUS_BAD_ARG);
        }

        #[test]
        fn ctl_set_packet_loss_edges() {
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            assert_eq!(enc.ctl(CeltEncoderCtl::SetPacketLossPerc(0)), OPUS_OK);
            assert_eq!(enc.ctl(CeltEncoderCtl::SetPacketLossPerc(100)), OPUS_OK);
            assert_eq!(enc.loss_rate, 100);
            assert_eq!(
                enc.ctl(CeltEncoderCtl::SetPacketLossPerc(101)),
                OPUS_BAD_ARG
            );
        }

        #[test]
        fn ctl_vbr_toggles_and_constraint() {
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            assert_eq!(enc.ctl(CeltEncoderCtl::SetVbr(1)), OPUS_OK);
            assert_eq!(enc.vbr, 1);
            assert_eq!(enc.ctl(CeltEncoderCtl::SetVbr(0)), OPUS_OK);
            assert_eq!(enc.vbr, 0);
            assert_eq!(enc.ctl(CeltEncoderCtl::SetVbrConstraint(1)), OPUS_OK);
            assert_eq!(enc.constrained_vbr, 1);
            assert_eq!(enc.ctl(CeltEncoderCtl::SetVbrConstraint(0)), OPUS_OK);
            assert_eq!(enc.constrained_vbr, 0);
        }

        #[test]
        fn ctl_bitrate_bounds_and_channel_cap() {
            let mut enc = CeltEncoder::new(48000, 2).unwrap();
            // v <= 500 and != OPUS_BITRATE_MAX is invalid
            assert_eq!(enc.ctl(CeltEncoderCtl::SetBitrate(500)), OPUS_BAD_ARG);
            assert_eq!(enc.ctl(CeltEncoderCtl::SetBitrate(0)), OPUS_BAD_ARG);
            // Valid value capped to 750000 * channels (here 2)
            assert_eq!(enc.ctl(CeltEncoderCtl::SetBitrate(10_000_000)), OPUS_OK);
            assert_eq!(enc.bitrate, 1_500_000);
            assert_eq!(enc.ctl(CeltEncoderCtl::SetBitrate(64000)), OPUS_OK);
            assert_eq!(enc.bitrate, 64000);
        }

        #[test]
        fn ctl_signalling_and_lfe_input_clipping() {
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            assert_eq!(enc.ctl(CeltEncoderCtl::SetSignalling(1)), OPUS_OK);
            assert_eq!(enc.signalling, 1);
            assert_eq!(enc.ctl(CeltEncoderCtl::SetSignalling(0)), OPUS_OK);
            assert_eq!(enc.signalling, 0);
            assert_eq!(enc.ctl(CeltEncoderCtl::SetLfe(1)), OPUS_OK);
            assert_eq!(enc.lfe, 1);
            assert_eq!(enc.ctl(CeltEncoderCtl::SetLfe(0)), OPUS_OK);
            assert_eq!(enc.lfe, 0);
            assert_eq!(enc.ctl(CeltEncoderCtl::SetInputClipping(1)), OPUS_OK);
            assert_eq!(enc.clip, 1);
            assert_eq!(enc.ctl(CeltEncoderCtl::SetInputClipping(0)), OPUS_OK);
            assert_eq!(enc.clip, 0);
        }

        #[test]
        fn ctl_set_channels_branches() {
            let mut enc = CeltEncoder::new(48000, 2).unwrap();
            assert_eq!(enc.ctl(CeltEncoderCtl::SetChannels(1)), OPUS_OK);
            assert_eq!(enc.stream_channels, 1);
            assert_eq!(enc.ctl(CeltEncoderCtl::SetChannels(2)), OPUS_OK);
            assert_eq!(enc.stream_channels, 2);
            assert_eq!(enc.ctl(CeltEncoderCtl::SetChannels(3)), OPUS_BAD_ARG);
        }

        #[test]
        fn ctl_lsb_depth_and_phase_inv_edges() {
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            assert_eq!(enc.ctl(CeltEncoderCtl::SetLsbDepth(8)), OPUS_OK);
            assert_eq!(enc.lsb_depth, 8);
            assert_eq!(enc.ctl(CeltEncoderCtl::SetLsbDepth(24)), OPUS_OK);
            assert_eq!(enc.lsb_depth, 24);
            assert_eq!(enc.ctl(CeltEncoderCtl::SetLsbDepth(0)), OPUS_BAD_ARG);
            assert_eq!(
                enc.ctl(CeltEncoderCtl::SetPhaseInversionDisabled(0)),
                OPUS_OK
            );
            assert_eq!(enc.disable_inv, 0);
            assert_eq!(
                enc.ctl(CeltEncoderCtl::SetPhaseInversionDisabled(-1)),
                OPUS_BAD_ARG
            );
        }

        #[test]
        fn ctl_start_end_band_boundaries() {
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            // Start band out of range
            assert_eq!(
                enc.ctl(CeltEncoderCtl::SetStartBand(enc.mode.nb_ebands)),
                OPUS_BAD_ARG
            );
            // End band out of range
            assert_eq!(enc.ctl(CeltEncoderCtl::SetEndBand(0)), OPUS_BAD_ARG);
            assert_eq!(
                enc.ctl(CeltEncoderCtl::SetEndBand(enc.mode.nb_ebands + 1)),
                OPUS_BAD_ARG
            );
            // Valid: 0, 1 (narrow)
            assert_eq!(enc.ctl(CeltEncoderCtl::SetStartBand(0)), OPUS_OK);
            assert_eq!(enc.ctl(CeltEncoderCtl::SetEndBand(1)), OPUS_OK);
        }

        #[test]
        fn ctl_reset_and_get_endpoints() {
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            enc.vbr_count = 42;
            assert_eq!(enc.ctl(CeltEncoderCtl::ResetState), OPUS_OK);
            assert_eq!(enc.vbr_count, 0);
            assert_eq!(enc.ctl(CeltEncoderCtl::GetLsbDepth), OPUS_OK);
            assert_eq!(enc.ctl(CeltEncoderCtl::GetPhaseInversionDisabled), OPUS_OK);
            assert_eq!(enc.ctl(CeltEncoderCtl::GetFinalRange), OPUS_OK);
            assert_eq!(enc.ctl(CeltEncoderCtl::SetEnergyMask), OPUS_OK);
        }

        // ---------------------------------------------------------------
        // transient_analysis direct branches (weak transient, tone protect)
        // ---------------------------------------------------------------

        #[test]
        fn transient_analysis_weak_transient_flag() {
            // Medium impulse: mask_metric between 200 and 600 -> weak transient
            // when allow_weak_transients is true.
            let mut input = vec![0i32; 960];
            for i in 100..108 {
                input[i] = 1 << (SIG_SHIFT - 1);
            }
            let mut tf_estimate = 0;
            let mut tf_chan = 0;
            let mut weak = false;
            let _ = transient_analysis(
                &input,
                960,
                1,
                &mut tf_estimate,
                &mut tf_chan,
                true, // allow weak
                &mut weak,
                0,
                0,
            );
            // Just make sure no panic and branch gets exercised
            let _ = weak;
        }

        #[test]
        fn transient_analysis_tone_protection() {
            // Strong toneishness + very low tone_freq triggers tone protection path
            let mut input = vec![0i32; 960];
            for i in 0..48 {
                input[i] = 1 << SIG_SHIFT;
            }
            let mut tf_estimate = 0;
            let mut tf_chan = 0;
            let mut weak = false;
            let is_trans = transient_analysis(
                &input,
                960,
                1,
                &mut tf_estimate,
                &mut tf_chan,
                false,
                &mut weak,
                qconst16(0.001, 13), // very low tone_freq
                qconst32(0.999, 29), // strong toneishness
            );
            // Tone protection forces is_transient=false regardless of mask
            assert!(!is_trans);
        }

        // ---------------------------------------------------------------
        // Encode with narrowed start/end bands, lfe, various complexities
        // ---------------------------------------------------------------

        #[test]
        fn encode_narrow_end_band() {
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            enc.vbr = 0;
            enc.bitrate = 32000;
            enc.end = 10; // truncate upper bands
            let pcm = gen_pcm(960, 1, 3);
            let ret = encode_frame(&mut enc, &pcm, 960, 128);
            assert!(ret > 0);
        }

        #[test]
        fn encode_hybrid_start_end_narrow() {
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            enc.vbr = 1;
            enc.bitrate = 48000;
            enc.start = 17;
            enc.end = 19;
            let pcm = gen_pcm(960, 1, 13);
            let ret = encode_frame(&mut enc, &pcm, 960, 128);
            assert!(ret > 0);
        }

        #[test]
        fn encode_lfe_complexity0_impulse() {
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            enc.vbr = 0;
            enc.bitrate = 16000;
            enc.lfe = 1;
            enc.complexity = 0;
            enc.end = 2;
            let mut pcm = vec![0i16; 960];
            for i in 0..20 {
                pcm[i] = 25000;
            }
            let ret = encode_frame(&mut enc, &pcm, 960, 64);
            assert!(ret > 0);
        }

        #[test]
        fn encode_dc_step_input() {
            // DC step: half silence, half constant -> interesting transient analysis
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            enc.vbr = 1;
            enc.bitrate = 64000;
            enc.complexity = 10;
            let mut pcm = vec![0i16; 960];
            for v in pcm.iter_mut().skip(480) {
                *v = 8000;
            }
            let ret = encode_frame(&mut enc, &pcm, 960, 128);
            assert!(ret > 0);
        }

        #[test]
        fn encode_sine_tonal_input() {
            // Pure sine exercises tone_detect path and toneishness-based branches
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            enc.vbr = 1;
            enc.bitrate = 48000;
            enc.complexity = 10;
            let pcm = gen_sine(960, 1, 440.0);
            for _ in 0..3 {
                let ret = encode_frame(&mut enc, &pcm, 960, 128);
                assert!(ret > 0);
            }
        }

        #[test]
        fn encode_stereo_sine_intensity_sweep() {
            // Stereo tonal signal at varied low bitrates drives intensity stereo
            for br in [16_000, 20_000, 24_000, 32_000] {
                let mut enc = CeltEncoder::new(48000, 2).unwrap();
                enc.vbr = 1;
                enc.bitrate = br;
                enc.complexity = 5;
                let pcm = gen_sine(960, 2, 880.0);
                let ret = encode_frame(&mut enc, &pcm, 960, 128);
                assert!(ret > 0, "bitrate {} failed: {}", br, ret);
            }
        }

        #[test]
        fn encode_cbr_various_rates_no_signalling() {
            for br in [8_000, 16_000, 32_000, 64_000, 128_000] {
                let mut enc = CeltEncoder::new(48000, 1).unwrap();
                enc.vbr = 0;
                enc.signalling = 0;
                enc.bitrate = br;
                let pcm = gen_pcm(960, 1, br / 1000);
                let ret = encode_frame(&mut enc, &pcm, 960, 256);
                assert!(ret > 0);
            }
        }

        #[test]
        fn encode_weak_transient_hybrid_path() {
            // Hybrid + very low effective_bytes and silk_info.signal_type != 2
            // enables allow_weak_transients path in transient_analysis.
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            enc.vbr = 0;
            enc.bitrate = 12000;
            enc.start = 17;
            enc.silk_info.signal_type = 1;
            let mut pcm = vec![0i16; 960];
            // Medium impulse
            for i in 200..210 {
                pcm[i] = 15000;
            }
            let ret = encode_frame(&mut enc, &pcm, 960, 14);
            assert!(ret > 0);
        }

        #[test]
        fn encode_multiple_frame_sizes() {
            // LM=0,1,2,3 sweep for mono at 48kHz
            for &(fs, bytes) in &[(120, 16), (240, 32), (480, 64), (960, 128)] {
                let mut enc = CeltEncoder::new(48000, 1).unwrap();
                enc.vbr = 1;
                enc.bitrate = 64000;
                enc.complexity = 10;
                let pcm = gen_pcm(fs as usize, 1, fs);
                let ret = encode_frame(&mut enc, &pcm, fs, bytes);
                assert!(ret > 0, "LM for frame_size={} failed: {}", fs, ret);
            }
        }

        #[test]
        fn encode_min_complexity_path() {
            // Complexity < 5 skips the standard pitch search (path 2) entirely
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            enc.vbr = 1;
            enc.bitrate = 48000;
            enc.complexity = 3;
            let pcm = gen_pcm(960, 1, 42);
            let ret = encode_frame(&mut enc, &pcm, 960, 128);
            assert!(ret > 0);
        }

        #[test]
        fn encode_complexity_lt2_skips_tf_analysis() {
            // enable_tf_analysis requires complexity >= 2. complexity=1 disables it.
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            enc.vbr = 1;
            enc.bitrate = 64000;
            enc.complexity = 1;
            let pcm = gen_pcm(960, 1, 77);
            let ret = encode_frame(&mut enc, &pcm, 960, 128);
            assert!(ret > 0);
        }

        #[test]
        fn encode_force_intra_repeated() {
            // force_intra=1 exercises the coarse-energy intra path
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            enc.vbr = 1;
            enc.bitrate = 64000;
            enc.force_intra = 1;
            for i in 0..3 {
                let pcm = gen_pcm(960, 1, i);
                let ret = encode_frame(&mut enc, &pcm, 960, 128);
                assert!(ret > 0);
            }
        }

        #[test]
        fn encode_stereo_varied_complexity() {
            for cpx in [0, 3, 5, 8, 10] {
                let mut enc = CeltEncoder::new(48000, 2).unwrap();
                enc.vbr = 1;
                enc.bitrate = 96_000;
                enc.complexity = cpx;
                let pcm = gen_pcm(960, 2, cpx);
                let ret = encode_frame(&mut enc, &pcm, 960, 256);
                assert!(ret > 0, "complexity {} failed: {}", cpx, ret);
            }
        }

        #[test]
        fn encode_high_loss_rate_kills_prefilter_gain() {
            // loss_rate > 8 zeroes gain1 in run_prefilter
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            enc.vbr = 1;
            enc.bitrate = 64000;
            enc.loss_rate = 20;
            let pcm = gen_sine(960, 1, 300.0);
            for _ in 0..2 {
                let ret = encode_frame(&mut enc, &pcm, 960, 128);
                assert!(ret > 0);
            }
        }

        #[test]
        fn encode_with_analysis_valid() {
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            enc.vbr = 1;
            enc.bitrate = 64000;
            enc.analysis = AnalysisInfo {
                valid: 1,
                tonality: 0.8,
                tonality_slope: 0.1,
                ..AnalysisInfo::default()
            };
            let pcm = gen_sine(960, 1, 220.0);
            let ret = encode_frame(&mut enc, &pcm, 960, 128);
            assert!(ret > 0);
        }

        #[test]
        fn encode_silk_info_signal_type_variants() {
            for sig in [0, 1, 2, 3] {
                let mut enc = CeltEncoder::new(48000, 1).unwrap();
                enc.vbr = 0;
                enc.bitrate = 12000;
                enc.start = 17;
                enc.silk_info.signal_type = sig;
                let pcm = gen_pcm(960, 1, sig);
                let ret = encode_frame(&mut enc, &pcm, 960, 20);
                assert!(ret > 0);
            }
        }

        // ---------------------------------------------------------------
        // Added targeted tests for uncovered encoder branches
        // ---------------------------------------------------------------

        #[test]
        fn encode_lfe_narrow_band_loop() {
            // LFE with end == 3 still has the "for i in 2..end" loop running
            // (one iter) and avoids other asserts triggered by wide-band LFE.
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            enc.vbr = 1;
            enc.bitrate = 16_000;
            enc.lfe = 1;
            enc.end = 3;
            let pcm = gen_sine(960, 1, 80.0);
            let ret = encode_frame(&mut enc, &pcm, 960, 64);
            assert!(ret > 0);
        }

        #[test]
        fn encode_pure_tone_strong_toneishness() {
            // Pure sine at low frequency hits the pre-filter Path 1
            // (toneishness > 0.99) and tone-compensation branches in dynalloc.
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            enc.vbr = 1;
            enc.bitrate = 96_000;
            enc.complexity = 10;
            // Large-amplitude, repeatable tone — drives toneishness high
            let mut pcm = vec![0i16; 960];
            for i in 0..960 {
                let angle = 2.0 * std::f64::consts::PI * 440.0 * i as f64 / 48000.0;
                pcm[i] = (20000.0 * angle.sin()) as i16;
            }
            // Encode multiple frames so the LPC converges on the sine model
            for _ in 0..4 {
                let ret = encode_frame(&mut enc, &pcm, 960, 256);
                assert!(ret > 0);
            }
        }

        #[test]
        fn encode_very_high_freq_tone() {
            // High frequency tone near Nyquist: exercises frequency aliasing
            // correction in run_prefilter (tf >= pi branch at L1760).
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            enc.vbr = 1;
            enc.bitrate = 96_000;
            enc.complexity = 10;
            let mut pcm = vec![0i16; 960];
            for i in 0..960 {
                let angle = 2.0 * std::f64::consts::PI * 14000.0 * i as f64 / 48000.0;
                pcm[i] = (18000.0 * angle.sin()) as i16;
            }
            for _ in 0..4 {
                let ret = encode_frame(&mut enc, &pcm, 960, 256);
                assert!(ret > 0);
            }
        }

        #[test]
        fn encode_cbr_heavy_dynalloc_budget_cap() {
            // CBR with very tight budget + high-energy impulse bands
            // triggers the budget-cap break at L1538-1544.
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            enc.vbr = 0;
            enc.bitrate = 10_000;
            enc.complexity = 5;
            let mut pcm = vec![0i16; 960];
            // Load high-energy content across the band
            for (i, v) in pcm.iter_mut().enumerate() {
                *v = if i % 7 < 4 { 18000 } else { -18000 };
            }
            for _ in 0..3 {
                let ret = encode_frame(&mut enc, &pcm, 960, 20);
                assert!(ret > 0);
            }
        }

        #[test]
        fn encode_large_transient_jump() {
            // Large energy jump between consecutive frames forces
            // patch_transient_decision to return true (L2520 inner path).
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            enc.vbr = 1;
            enc.bitrate = 64_000;
            enc.complexity = 8;
            // Frame 1: silence-ish; Frame 2: loud
            let pcm_quiet = vec![100i16; 960];
            let mut pcm_loud = vec![0i16; 960];
            for (i, v) in pcm_loud.iter_mut().enumerate() {
                *v = if i % 2 == 0 { 25000 } else { -25000 };
            }
            let _ = encode_frame(&mut enc, &pcm_quiet, 960, 128);
            let _ = encode_frame(&mut enc, &pcm_quiet, 960, 128);
            let ret = encode_frame(&mut enc, &pcm_loud, 960, 128);
            assert!(ret > 0);
        }

        #[test]
        fn tone_lpc_clamp_branches() {
            // Craft signals that force lpc clamping (lines 1642, 1644, 1650, 1653)
            let mut lpc = [0i32; 2];
            // A pure cosine at a sharp resonance forces lpc[0] clamp
            let mut s: Vec<i16> = Vec::with_capacity(400);
            for i in 0..400 {
                let v = (25000.0 * (i as f64 * 0.02).cos()) as i16;
                s.push(v);
            }
            for delay in [2, 4, 8, 16] {
                let _ = tone_lpc(&s, 400, delay, &mut lpc);
            }
            // Negative cosine — flips sign direction
            let s2: Vec<i16> = (0..400)
                .map(|i| -((25000.0 * (i as f64 * 0.04).cos()) as i16))
                .collect();
            for delay in [3, 6, 12] {
                let _ = tone_lpc(&s2, 400, delay, &mut lpc);
            }
        }

        #[test]
        fn encode_hybrid_eff_end_less_than_end() {
            // Push end beyond eff_ebands with start=17 (hybrid) to hit the
            // eff_end..end tail copy loop (L2613).
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            enc.vbr = 1;
            enc.bitrate = 64_000;
            enc.complexity = 5;
            enc.start = 17;
            enc.end = enc.mode.nb_ebands;
            let pcm = gen_pcm(960, 1, 17);
            let ret = encode_frame(&mut enc, &pcm, 960, 128);
            assert!(ret > 0);
        }

        #[test]
        fn encode_low_complexity_short_block() {
            // complexity < 3 forces spread decision to SPREAD_NORMAL (no
            // spreading_decision call) and skips some analysis paths.
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            enc.vbr = 1;
            enc.bitrate = 32_000;
            enc.complexity = 2;
            let mut pcm = vec![0i16; 960];
            // Short sharp impulse in middle
            for i in 480..520 {
                pcm[i] = 20000;
            }
            let ret = encode_frame(&mut enc, &pcm, 960, 128);
            assert!(ret > 0);
        }

        #[test]
        fn encode_16k_complex_signal() {
            // 16k upsampled, stereo, low complexity, low bitrate — narrow paths
            let mut enc = CeltEncoder::new(16000, 2).unwrap();
            enc.vbr = 1;
            enc.bitrate = 16_000;
            enc.complexity = 3;
            let mut pcm = vec![0i16; 320 * 2];
            for (i, v) in pcm.iter_mut().enumerate() {
                *v = (((i as i32 * 31) % 20000) - 10000) as i16;
            }
            let ret = encode_frame(&mut enc, &pcm, 320, 64);
            assert!(ret > 0);
        }

        #[test]
        fn encode_stereo_with_high_complexity_second_mdct() {
            // complexity >= 8 enables the "second_mdct" code path inside
            // celt_encode_core when short_blocks is set (after transient).
            let mut enc = CeltEncoder::new(48000, 2).unwrap();
            enc.vbr = 1;
            enc.bitrate = 128_000;
            enc.complexity = 9;
            let mut pcm = vec![0i16; 960 * 2];
            for i in 0..40 {
                pcm[2 * i] = 28000;
                pcm[2 * i + 1] = -28000;
            }
            let ret = encode_frame(&mut enc, &pcm, 960, 256);
            assert!(ret > 0);
        }

        #[test]
        fn encode_stereo_cbr_low_bitrate_many_frames() {
            // Multi-frame stereo CBR at low bitrate drives intensity stereo
            // transitions and hysteresis paths.
            let mut enc = CeltEncoder::new(48000, 2).unwrap();
            enc.vbr = 0;
            enc.bitrate = 16_000;
            enc.complexity = 5;
            for f in 0..6 {
                let pcm = gen_pcm(960, 2, f * 3);
                let ret = encode_frame(&mut enc, &pcm, 960, 40);
                assert!(ret > 0, "frame {} failed: {}", f, ret);
            }
        }

        #[test]
        fn encode_narrow_end_complexity_range() {
            // Narrow-band encode at various complexities covers spreading path
            // variants.
            for cpx in [0, 1, 2, 3, 4] {
                let mut enc = CeltEncoder::new(48000, 1).unwrap();
                enc.vbr = 1;
                enc.bitrate = 16_000;
                enc.complexity = cpx;
                enc.end = 8;
                let pcm = gen_pcm(960, 1, cpx);
                let ret = encode_frame(&mut enc, &pcm, 960, 64);
                assert!(ret > 0);
            }
        }

        // ---------------------------------------------------------------
        // Additional targeted tests — misc paths
        // ---------------------------------------------------------------

        #[test]
        fn encode_extreme_silence_between_loud_frames() {
            // Wake up / sleep transient triggers patch_transient_decision true.
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            enc.vbr = 1;
            enc.bitrate = 96_000;
            enc.complexity = 10;
            // First 3 frames: silence to establish baseline
            for _ in 0..3 {
                let silent = vec![0i16; 960];
                let _ = encode_frame(&mut enc, &silent, 960, 128);
            }
            // Loud square wave — big energy jump
            let loud: Vec<i16> = (0..960)
                .map(|i| if i % 8 < 4 { 30000 } else { -30000 })
                .collect();
            let ret = encode_frame(&mut enc, &loud, 960, 256);
            assert!(ret > 0);
        }

        #[test]
        fn encode_many_bitrate_hysteresis() {
            // Drive stereo intensity hysteresis via bitrate steps
            let mut enc = CeltEncoder::new(48000, 2).unwrap();
            enc.vbr = 1;
            enc.complexity = 8;
            let rates = [160_000i32, 96_000, 48_000, 24_000, 16_000, 24_000, 96_000];
            for (i, &br) in rates.iter().enumerate() {
                enc.bitrate = br;
                let pcm = gen_pcm(960, 2, i as i32);
                let ret = encode_frame(&mut enc, &pcm, 960, 256);
                assert!(ret > 0);
            }
        }

        #[test]
        fn encode_cvbr_many_frames_drain_reservoir() {
            // Stream of frames forces vbr_reservoir to go negative then positive
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            enc.vbr = 1;
            enc.constrained_vbr = 1;
            enc.bitrate = 20_000;
            enc.complexity = 5;
            for i in 0..40 {
                let pcm = if i % 4 == 0 {
                    gen_sine(960, 1, 440.0)
                } else {
                    gen_pcm(960, 1, i)
                };
                let ret = encode_frame(&mut enc, &pcm, 960, 64);
                assert!(ret > 0);
            }
        }

        #[test]
        fn encode_forced_intra_cvbr() {
            // Combine force_intra=1 with CVBR to cover the intra + reservoir path
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            enc.vbr = 1;
            enc.constrained_vbr = 1;
            enc.bitrate = 32_000;
            enc.force_intra = 1;
            for i in 0..3 {
                let pcm = gen_pcm(960, 1, i);
                let ret = encode_frame(&mut enc, &pcm, 960, 128);
                assert!(ret > 0);
            }
        }

        #[test]
        fn encode_240_lm1_stereo_cvbr() {
            // LM=1 (5ms) stereo with CVBR exercises narrow frame code paths
            let mut enc = CeltEncoder::new(48000, 2).unwrap();
            enc.vbr = 1;
            enc.constrained_vbr = 1;
            enc.bitrate = 48_000;
            enc.complexity = 8;
            for i in 0..3 {
                let pcm = gen_pcm(240, 2, i);
                let ret = encode_frame(&mut enc, &pcm, 240, 64);
                assert!(ret > 0);
            }
        }

        #[test]
        fn encode_120_lm0_smallest_frame() {
            // LM=0 (2.5ms) — smallest CELT frame
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            enc.vbr = 1;
            enc.bitrate = 48_000;
            enc.complexity = 5;
            for i in 0..4 {
                let pcm = gen_pcm(120, 1, i);
                let ret = encode_frame(&mut enc, &pcm, 120, 64);
                assert!(ret > 0);
            }
        }

        #[test]
        fn encode_clip_enabled_loud_input() {
            // input_clipping=1 with loud input exercises clip path in preemphasis
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            enc.vbr = 1;
            enc.bitrate = 64_000;
            enc.clip = 1;
            // Clip-inducing input (values near i16 limits)
            let pcm: Vec<i16> = (0..960)
                .map(|i| if i & 1 == 0 { 32767 } else { -32768 })
                .collect();
            let ret = encode_frame(&mut enc, &pcm, 960, 128);
            assert!(ret > 0);
        }

        #[test]
        fn encode_clip_disabled_loud_input() {
            let mut enc = CeltEncoder::new(48000, 1).unwrap();
            enc.vbr = 1;
            enc.bitrate = 64_000;
            enc.clip = 0;
            let pcm: Vec<i16> = (0..960)
                .map(|i| if i & 1 == 0 { 32767 } else { -32768 })
                .collect();
            let ret = encode_frame(&mut enc, &pcm, 960, 128);
            assert!(ret > 0);
        }

        #[test]
        fn encode_stereo_clip_loud() {
            let mut enc = CeltEncoder::new(48000, 2).unwrap();
            enc.vbr = 1;
            enc.bitrate = 128_000;
            enc.clip = 1;
            let pcm: Vec<i16> = (0..960 * 2)
                .map(|i| if i % 5 < 3 { 32767 } else { -30000 })
                .collect();
            let ret = encode_frame(&mut enc, &pcm, 960, 256);
            assert!(ret > 0);
        }
    }
}
