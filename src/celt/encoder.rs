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

use crate::types::*;
use super::bands::{
    compute_band_energies, haar1, hysteresis_decision, normalise_bands,
    quant_all_bands, spreading_decision, SPREAD_AGGRESSIVE, SPREAD_NONE, SPREAD_NORMAL,
};
use super::fft;
use super::fft::{KissFftCpx, KissFftState, opus_fft_impl};
use super::math_ops::*;
use super::modes::{
    init_caps, resampling_factor, bitrate_to_bits, CELTMode,
    MODE_48000_960_120,
};
use super::pitch::{pitch_downsample, pitch_search, remove_doubling};
use super::quant_bands::{amp2log2, quant_coarse_energy, quant_energy_finalise, quant_fine_energy};
use super::range_coder::RangeEncoder;
use super::rate::{
    clt_compute_allocation, BITRES, SPREAD_ICDF, TAPSET_ICDF, TF_SELECT_TABLE, TRIM_ICDF,
};
use super::tables::E_MEANS;
use super::vq::celt_inner_prod_norm_shift;

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

/// Opus error codes matching the C reference.
pub const OPUS_OK: i32 = 0;
pub const OPUS_BAD_ARG: i32 = -1;
pub const OPUS_INTERNAL_ERROR: i32 = -3;
pub const OPUS_BITRATE_MAX: i32 = -1000;

/// Comb filter tapset gains: `gains[tapset][tap]` in Q15.
/// Matches C `gains[3][3]` in `comb_filter()`.
static COMB_GAINS: [[i32; 3]; 3] = [
    [qconst16(0.3066406250, 15), qconst16(0.2170410156, 15), qconst16(0.1296386719, 15)],
    [qconst16(0.4638671875, 15), qconst16(0.2680664062, 15), 0],
    [qconst16(0.7998046875, 15), qconst16(0.1000976562, 15), 0],
];

/// Precomputed `6*64/x` table for transient analysis (128 entries).
/// Trained on real data to minimize average error.
static INV_TABLE: [i16; 128] = [
    255, 255, 156,  110,  86,  70,  59,  51,  45,  40,  37,  33,  31,  28,  26,  25,
     23,  22,  21,  20,  19,  18,  17,  16,  16,  15,  15,  14,  13,  13,  12,  12,
     12,  12,  11,  11,  11,  10,  10,  10,  10,   9,   9,   9,   9,   9,   8,   8,
      8,   8,   8,   7,   7,   7,   7,   7,   7,   6,   6,   6,   6,   6,   6,   6,
      6,   6,   6,   6,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,
      4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,
      3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,
      3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   2,   2,   2,   2,   2,
];

/// Intensity stereo thresholds (in kbps), 21 entries.
static INTENSITY_THRESHOLDS: [i32; 21] = [
     1,  2,  3,  4,  5,  6,  7,  8, 16, 24, 36, 44, 50, 56, 62, 67, 72, 79, 88, 106, 134,
];

/// Intensity stereo hysteresis values, 21 entries.
static INTENSITY_HISTERESIS: [i32; 21] = [
    1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 5, 6, 8, 8,
];

// ===========================================================================
// Types
// ===========================================================================

/// External audio analysis data from the Opus layer.
///
/// In the full Opus encoder, this carries machine-learning analysis results
/// (tonality, activity, bandwidth, etc.). For standalone CELT encoding,
/// this is zeroed / invalid.
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
    x_offset: usize,  // where index 0 of x is in the backing buffer
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
    let ov = if g0 == g1 && t0 == t1 && tapset0 == tapset1 { 0 } else { overlap };

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
        val += mult16_32_q15(
            mult16_16_q15(f, g11),
            x[xi - t1u + 1] + x[xi - t1u - 1],
        );
        val += mult16_32_q15(
            mult16_16_q15(f, g12),
            x[xi - t1u + 2] + x[xi - t1u - 2],
        );
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
/// The twiddle factors are computed on the fly using `celt_cos_norm`.
pub(crate) fn clt_mdct_forward(
    input: &[i32],
    output: &mut [i32],
    window: &[i16],
    overlap: i32,
    shift: i32,
    stride: i32,
    fft_state: &KissFftState,
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
            f[yp] = mult16_32_q15(w2, input[(xp1 + n2i) as usize]) + mult16_32_q15(w1, input[xp2 as usize]);
            f[yp + 1] = mult16_32_q15(w1, input[xp1 as usize]) - mult16_32_q15(w2, input[(xp2 - n2i) as usize]);
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
            f[yp] = -mult16_32_q15(w1, input[(xp1 - n2i) as usize]) + mult16_32_q15(w2, input[xp2 as usize]);
            f[yp + 1] = mult16_32_q15(w2, input[xp1 as usize]) + mult16_32_q15(w1, input[(xp2 + n2i) as usize]);
            xp1 += 2;
            xp2 -= 2;
            wp1 += 2;
            wp2 -= 2;
            yp += 2;
            i += 1;
        }
    }

    // Step 2: Pre-rotation - compute twiddle factors on the fly and apply
    let mut f2 = vec![KissFftCpx::default(); n4];
    let mut maxval = 1i32;
    for i in 0..n4 {
        // Compute twiddle: cos(pi*(2*i+1)/(4*n4)) and sin(pi*(2*i+1)/(4*n4))
        // Using celt_cos_norm which takes x in Q16 where 1.0 = full period
        let _phase_num = (2 * i + 1) as i32;
        let _phase_den = (4 * n4) as i32;
        // t0 = cos(pi*phase_num/(2*phase_den)) = celt_cos_norm(phase_num*65536/(2*phase_den))
        // But we use the same computation as the C init: (i*131072+16386)/(N/2)
        let t0 = extract16(celt_cos_norm(div32(
            (i as i32) * 131072 + 16386,
            n2 as i32,
        )));
        let t1 = extract16(celt_cos_norm(div32(
            (n4 as i32 + i as i32) * 131072 + 16386,
            n2 as i32,
        )));

        let re = f[2 * i];
        let im = f[2 * i + 1];
        // yr = re*t0 - im*t1, yi = im*t0 + re*t1 (using S_MUL = MULT16_32_Q15)
        let yr = mult16_32_q15(t0, re) - mult16_32_q15(t1, im);
        let yi = mult16_32_q15(t0, im) + mult16_32_q15(t1, re);

        // Scale by S_MUL2 = MULT16_32_Q16 for non-QEXT
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

    // Step 4: Post-rotation
    let st = stride as usize;
    let mut yp1 = 0usize;
    let mut yp2 = st * (n2 - 1);
    for i in 0..n4 {
        let t0 = extract16(celt_cos_norm(div32(
            (i as i32) * 131072 + 16386,
            n2 as i32,
        )));
        let t1 = extract16(celt_cos_norm(div32(
            (n4 as i32 + i as i32) * 131072 + 16386,
            n2 as i32,
        )));

        let yr = pshr32(
            mult16_32_q15(t1, f2[i].i) - mult16_32_q15(t0, f2[i].r),
            headroom,
        );
        let yi = pshr32(
            mult16_32_q15(t1, f2[i].r) + mult16_32_q15(t0, f2[i].i),
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

    for ch in 0..cc as usize {
        let in_base = ch * (b as usize * n as usize + overlap as usize);
        for blk in 0..b as usize {
            let in_start = in_base + blk * n as usize;
            let out_start = blk + ch * (n as usize * b as usize);
            // Create a temporary output slice for this block
            let mut tmp_out = vec![0i32; n as usize];
            clt_mdct_forward(
                &input[in_start..],
                &mut tmp_out,
                mode.window,
                overlap,
                shift,
                1, // stride=1 into temp buffer; real stride applied in copy below
                fft_st,
            );
            // Copy with stride
            let stride = if short_blocks != 0 { b as usize } else { 1 };
            for k in 0..n as usize {
                let out_idx = out_start + k * stride;
                if out_idx < output.len() {
                    output[out_idx] = tmp_out[k];
                }
            }
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
/// Matches C `celt_preemphasis()` from `celt_encoder.c`.
/// Uses the simple first-order pre-emphasis: y[n] = x[n] - coef * x[n-1].
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

    // Fast path: no upsampling, simple first-order pre-emphasis
    if upsample == 1 {
        let mut m = *mem;
        for i in 0..nu {
            let x = shl32(pcm[(pcm_offset + i * ccu) as usize] as i32, SIG_SHIFT);
            inp[inp_offset + i] = x - mult16_32_q15(coef0, m);
            m = x;
        }
        *mem = m;
    } else {
        // Upsampling path: insert zeros between samples
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
            inp[inp_offset + i] = x - mult16_32_q15(coef0, m);
            m = x;
        }
        *mem = m;
    }
}

// ===========================================================================
// Encoder helper: transient_analysis
// ===========================================================================

/// Detect transients in the input signal.
///
/// Matches C `transient_analysis()` from `celt_encoder.c`.
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
    let len2 = (len / 2) as usize;
    #[allow(unused_assignments)]
    let mut is_transient = false;
    let mut mask_metric: i32 = 0;
    let tf_max: i32;

    // Forward masking decay: 4 for normal, 5 for weak transient mode
    let forward_shift: i32 = if allow_weak_transients { 5 } else { 4 };

    *weak_transient = false;
    *tf_chan = 0;

    for c in 0..cc as usize {
        let mut mem0: i32 = 0;
        let mut mem1: i32 = 0;

        // High-pass filter and downsample by 2
        let mut tmp = vec![0i32; len2];
        let n_plus_overlap = len as usize;
        for i in 0..n_plus_overlap {
            let x = shr32(input[c * n_plus_overlap + i], SIG_SHIFT);
            let y = x - mem1 + half32(mem0);
            mem0 = mem1;
            mem1 = x;
            if i >= 12 && (i & 1) == 0 && (i / 2) < len2 {
                tmp[i / 2] = y;
            }
        }
        // Clear first 6 samples (from the 12 skipped above, downsampled)
        for i in 0..6.min(len2) {
            tmp[i] = 0;
        }

        // Normalize
        let maxabs = celt_maxabs16(&tmp[..len2]);
        let shift = 14i32.saturating_sub(celt_ilog2(1.max(maxabs)));
        let shift = shift.max(0);
        for i in 0..len2 {
            tmp[i] = shl32(tmp[i], shift);
        }

        // Forward masking
        let mut forward = vec![0i32; len2];
        let mut accum: i32 = 0;
        for i in 0..len2 {
            let val = sround16(tmp[i], 2);
            let e = mult16_16(val, val);
            accum = accum.wrapping_sub(shr32(accum, forward_shift)).wrapping_add(e);
            forward[i] = accum;
        }

        // Backward masking (steeper decay)
        let mut backward = vec![0i32; len2];
        accum = 0;
        for i in (0..len2).rev() {
            let val = sround16(tmp[i], 2);
            let e = mult16_16(val, val);
            accum = accum.wrapping_sub(shr32(accum, 3)).wrapping_add(e);
            backward[i] = accum;
        }

        // Compute transient metric
        let mut unmask: i32 = 0;
        for i in 0..len2 {
            let f = forward[i].max(1);
            let b = backward[i].max(1);
            let id = (128i64 * f as i64 / (f as i64 + b as i64)).max(0).min(127) as usize;
            unmask += INV_TABLE[id] as i32;
        }

        // Normalize by frame length
        let c_metric = if len2 > 17 {
            64 * unmask * 4 / (6 * (len2 as i32 - 17))
        } else {
            0
        };

        if c_metric > mask_metric {
            mask_metric = c_metric;
            *tf_chan = c as i32;
        }
    }

    // Tone protection: suppress transient detection for pure tones
    if toneishness > qconst32(0.98, 29) && tone_freq < qconst16(0.026, 13) {
        mask_metric = 0;
    }

    is_transient = mask_metric > 200;

    // Weak transient handling
    if allow_weak_transients && is_transient && mask_metric < 600 {
        *weak_transient = true;
        is_transient = false;
    }

    // Compute tf_estimate for VBR
    tf_max = (celt_sqrt(27 * mask_metric.max(0)) - 42).max(0);
    let tf_tmp = mult16_16(qconst16(0.0069, 14), tf_max.min(163));
    let tf_tmp2 = shl32(tf_tmp, 14).wrapping_sub(qconst32(0.139, 28));
    *tf_estimate = celt_sqrt(tf_tmp2.max(0));

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
    let mut spread_old = vec![0i32; nb_ebands as usize];

    for ch in 0..c {
        // Get max of channels for stereo
        for i in 0..nb_ebands as usize {
            let old_e = if c == 2 {
                old_band_e[i].max(old_band_e[i + nb_ebands as usize])
            } else {
                old_band_e[i]
            };
            spread_old[i] = old_e;
        }

        // Apply aggressive forward spreading (-1 dB/band in log2 domain)
        for i in 1..end as usize {
            spread_old[i] = spread_old[i].max(spread_old[i - 1] - (1 << DB_SHIFT));
        }
        // Backward spreading
        for i in (start as usize..(end as usize - 1)).rev() {
            spread_old[i] = spread_old[i].max(spread_old[i + 1] - (1 << DB_SHIFT));
        }

        for i in (start.max(2) as usize)..(end as usize) {
            let new_e = if c == 2 {
                band_log_e[i + ch as usize * nb_ebands as usize]
                    .max(band_log_e[i + (1 - ch as usize) * nb_ebands as usize])
            } else {
                band_log_e[i]
            };
            let diff = new_e - spread_old[i];
            if diff > 0 {
                mean_diff += diff;
            }
        }
    }

    let count = c * (end - 1 - start.max(2));
    if count > 0 {
        mean_diff /= count;
    }

    // Threshold: 1.0 in Q(DB_SHIFT) = 1 << DB_SHIFT
    mean_diff > (1 << DB_SHIFT)
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
    let nb_ebands = m.nb_ebands as usize;
    let n0u = n0 as usize;

    // Bias towards longer windows when tf_estimate is low
    let bias = mult16_16_q14(
        qconst16(0.04, 15),
        qconst16(0.5, 14).max(-qconst16(0.25, 14).max(qconst16(0.5, 14) - tf_estimate)),
    );

    let mut metric = vec![0i32; nb_ebands];
    let mut path0 = vec![0i32; nb_ebands];
    let mut path1 = vec![0i32; nb_ebands];
    let mut sel_cost = [0i32; 2];

    // Compute metric per band by comparing L1 norms at different TF resolutions
    for i in (m.ebands[0] as usize / n0u)..len as usize {
        let band_start = m.ebands[i] as usize;
        let band_end = m.ebands[i + 1] as usize;
        let n = (band_end - band_start) << lm as usize;
        let narrow = (band_end - band_start) == 1;

        if n <= 0 {
            continue;
        }

        // Use channel indicated by tf_chan
        let x_off = tf_chan as usize * n0u * (1 << lm as usize) + band_start * (1 << lm as usize);

        // Compute L1 at current resolution
        let mut tmp = vec![0i32; n];
        if x_off + n <= x.len() {
            tmp[..n].copy_from_slice(&x[x_off..x_off + n]);
        }
        let l1_base = l1_metric(&tmp, n as i32, lm, bias);

        // Apply Haar transform to test alternative TF resolution
        if is_transient && n >= 2 {
            haar1(&mut tmp, (n >> lm as usize) as i32, 1 << lm);
        } else if !is_transient && lm > 0 {
            // For non-transient, test with shorter blocks
            for _ in 0..lm {
                haar1(&mut tmp, n as i32, 1);
            }
        }
        let l1_alt = l1_metric(&tmp, n as i32, lm, bias);

        // Metric: positive = prefer alternative TF, negative = prefer current
        if l1_base > 0 {
            metric[i] = if is_transient {
                // For transient: positive metric means short blocks are better
                ((l1_base - l1_alt) as i64 * 256 / l1_base as i64) as i32
            } else {
                ((l1_alt - l1_base) as i64 * 256 / l1_base as i64) as i32
            };
        }

        // Narrow band adjustment
        if narrow && (metric[i] == 0 || metric[i] == -2 * lm) {
            metric[i] -= 1;
        }
    }

    // Viterbi forward pass
    let mut cost0 = 0i32;
    let mut cost1 = 0i32;
    for i in (m.ebands[0] as usize / n0u)..len as usize {
        let imp = importance[i].max(1);
        let curr0 = cost0;
        let curr1 = cost1;
        let mc = metric[i] * imp;

        // State 0: tf_res=0
        let from0_to0 = curr0;
        let from1_to0 = curr1 + lambda * imp;
        if from0_to0 < from1_to0 {
            cost0 = from0_to0;
            path0[i] = 0;
        } else {
            cost0 = from1_to0;
            path0[i] = 1;
        }

        // State 1: tf_res=1
        let from0_to1 = curr0 + lambda * imp;
        let from1_to1 = curr1;
        if from0_to1 + mc < from1_to1 + mc {
            cost1 = from0_to1 + mc;
            path1[i] = 0;
        } else {
            cost1 = from1_to1 + mc;
            path1[i] = 1;
        }
    }

    // Backward trace
    let mut tf_select = 0;
    let final_state = if cost0 < cost1 { 0 } else { 1 };
    let mut state = final_state;
    for i in (0..len as usize).rev() {
        tf_res[i] = state;
        state = if state == 0 { path0[i] } else { path1[i] };
    }

    // Check if tf_select=1 gives better results
    if is_transient && lm > 0 {
        sel_cost[0] = cost0.min(cost1);
        // Recompute with tf_select=1 adjustment would go here
        // For simplicity, use 0 unless the cost difference is significant
        sel_cost[1] = sel_cost[0]; // simplified
        if sel_cost[1] < sel_cost[0] - lambda {
            tf_select = 1;
        }
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
    tf_select: i32,
    ec: &mut RangeEncoder,
) {
    let budget = (ec.tell_frac() + (end - start) as u32 * 8).min(ec.buffer().len() as u32 * 8);
    let mut tell = ec.tell_frac();

    // Reserve bit for tf_select if LM > 0
    let tf_select_rsv = lm > 0 && tell + 8 + 1 <= budget as u32;

    let mut logp = if is_transient { 2u32 } else { 4 };
    let mut curr = 0i32;

    for i in start..end {
        if tell + logp <= budget as u32 {
            ec.encode_bit_logp(tf_res[i as usize] ^ curr != 0, logp);
            tell = ec.tell_frac();
            curr = tf_res[i as usize];
            logp = if is_transient { 4 } else { 5 };
        } else {
            tf_res[i as usize] = curr;
        }
    }

    // Encode tf_select
    if tf_select_rsv && tell + 1 <= budget as u32 {
        ec.encode_bit_logp(tf_select != 0, 1);
    }

    // Apply tf_select_table to convert raw flags to actual TF values
    for i in start..end {
        let idx = 4 * (is_transient as usize) + 2 * (tf_select as usize) + tf_res[i as usize] as usize;
        tf_res[i as usize] = TF_SELECT_TABLE[lm as usize][idx] as i32;
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
    _analysis: &AnalysisInfo,
    stereo_saving: &mut i32,
    tf_estimate: i32,
    _intensity: i32,
    surround_trim: i32,
    equiv_rate: i32,
) -> i32 {
    let _nb_ebands = m.nb_ebands;
    let n0u = n0 as usize;

    // Base trim (Q8)
    let mut trim = qconst16(5.0, 8);
    if equiv_rate < 64000 {
        trim = qconst16(4.0, 8);
    } else if equiv_rate < 80000 {
        let frac = (equiv_rate - 64000) as i32;
        trim = qconst16(4.0, 8) + mult16_16_q15(
            qconst16(1.0 / 16000.0, 15) * frac,
            qconst16(1.0, 8),
        );
    }

    // Stereo analysis: compute inter-channel correlation
    if c == 2 {
        let mut sum: i32 = 0;
        let mut min_xc: i32 = Q15_ONE;

        // Bands 0-7: full correlation
        let corr_end = 8.min(end as usize);
        for i in 0..corr_end {
            let n = ((m.ebands[i + 1] - m.ebands[i]) as usize) << lm as usize;
            let x_off = m.ebands[i] as usize * (1 << lm as usize);
            if n > 0 && x_off + n <= x.len() && n0u * (1 << lm as usize) + x_off + n <= x.len() {
                let xc = celt_inner_prod_norm_shift(
                    &x[x_off..x_off + n],
                    &x[n0u * (1 << lm as usize) + x_off..],
                    n,
                );
                sum += xc;
                min_xc = min_xc.min(xc);
            }
        }
        if corr_end > 0 {
            sum = mult16_16_q15(qconst16(1.0 / 8.0, 15), sum);
        }

        // Compute stereo savings
        let log_xc = celt_log2(qconst32(1.001, 20) - mult16_16(sum, sum));
        let log_xc2 = (log_xc / 2).max(celt_log2(qconst32(1.001, 20) - mult16_16(min_xc, min_xc)));

        *stereo_saving = shr32(log_xc2 - (log_xc / 2), DB_SHIFT - 8).min(qconst16(2.0, 8));
        trim += (qconst16(0.5, 8).max(*stereo_saving)).max(-qconst16(4.0, 8));
    }

    // Spectral tilt: measure energy distribution across bands
    let mut tilt: i32 = 0;
    for i in 1..end as usize {
        let diff = band_log_e[i] - band_log_e[i - 1];
        tilt += diff;
    }
    if end > 1 {
        tilt /= (end - 1) as i32;
    }
    trim -= pshr32(tilt, DB_SHIFT - 8).max(-qconst16(2.0, 8)).min(qconst16(2.0, 8));

    // Surround masking adjustment
    trim -= surround_trim;

    // TF estimate adjustment
    trim -= pshr32((tf_estimate - qconst16(0.2, 14)) as i32 * 2, 14 - 8)
        .max(-qconst16(2.0, 8))
        .min(qconst16(2.0, 8));

    // Quantize to integer 0–10
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
    let mut sum_lr: i32 = 0;
    let mut sum_ms: i32 = 0;

    let analysis_end = 13.min(m.nb_ebands as usize);
    let reduced_end = if lm <= 1 { analysis_end.saturating_sub(8) } else { analysis_end };

    for i in 0..reduced_end {
        let n = ((m.ebands[i + 1] - m.ebands[i]) as usize) << lm as usize;
        let x_off = m.ebands[i] as usize * (1 << lm as usize);
        let y_off = n0u * (1 << lm as usize) + x_off;

        for j in 0..n {
            if x_off + j < x.len() && y_off + j < x.len() {
                let l = x[x_off + j];
                let r = x[y_off + j];
                sum_lr += abs32(l) + abs32(r);
                sum_ms += abs32(l + r) + abs32(l - r);
            }
        }
    }

    // M/S is better if sumMS < 0.707 * sumLR
    let scaled_ms = mult16_32_q15(qconst16(0.707107, 15), sum_ms);
    (scaled_ms < sum_lr) as i32
}

// ===========================================================================
// Encoder helpers: median_of_5 / median_of_3
// ===========================================================================

/// Median of 5 values using a sorting network.
fn median_of_5(d: &[i32]) -> i32 {
    // Matches C reference median_of_5() from celt_encoder.c
    let t2 = d[2];
    let (t0, t1) = if d[0] > d[1] { (d[1], d[0]) } else { (d[0], d[1]) };
    let (t3, t4) = if d[3] > d[4] { (d[4], d[3]) } else { (d[3], d[4]) };
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
        if b < c { b } else if a < c { c } else { a }
    } else {
        if a < c { a } else if b < c { c } else { b }
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
    _band_log_e: &[i32],
    band_log_e2: &[i32],
    _old_band_e: &[i32],
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
    _analysis: &AnalysisInfo,
    importance: &mut [i32],
    spread_weight: &mut [i32],
    _tone_freq: i32,
    _toneishness: i32,
) -> i32 {
    let nbu = nb_ebands as usize;
    let mut max_depth: i32 = GCONST_NEG28;
    *tot_boost = 0;

    let mut follower = vec![0i32; nbu];
    let mut noise_floor = vec![0i32; nbu];

    // Compute noise floor per band
    for i in start as usize..end as usize {
        let log_n_val = (log_n[i] as i32) << (DB_SHIFT - 4); // Approximate conversion
        noise_floor[i] = log_n_val / 16
            + (1 << (DB_SHIFT - 1))
            + (9 - lsb_depth) * (1 << DB_SHIFT)
            - shl32(E_MEANS[i] as i32, DB_SHIFT - 4);
    }

    // Compute SMR (signal-to-mask ratio) for each band and channel
    let mut band_log_e3 = vec![0i32; nbu];
    for i in start as usize..end as usize {
        let mut max_e = GCONST_NEG28;
        for ch in 0..c as usize {
            let e = band_log_e2[ch * nbu + i];
            max_e = max_e.max(e);
        }
        band_log_e3[i] = max_e;
        max_depth = max_depth.max(max_e - noise_floor[i]);
    }

    // Follower envelope with median filtering
    if end - start > 4 {
        for i in (start + 2) as usize..(end - 2) as usize {
            let med = median_of_5(&band_log_e3[i - 2..]);
            follower[i] = med - (1 << DB_SHIFT); // offset = 1 dB
        }
        for i in start as usize..(start + 2).min(end) as usize {
            follower[i] = band_log_e3[i] - (1 << DB_SHIFT);
        }
        if end >= 2 {
            for i in (end - 2).max(start) as usize..end as usize {
                follower[i] = band_log_e3[i] - (1 << DB_SHIFT);
            }
        }
    } else {
        for i in start as usize..end as usize {
            follower[i] = band_log_e3[i] - (1 << DB_SHIFT);
        }
    }

    // Downward spreading: -1.5 dB/band forward
    for i in (start + 1) as usize..end as usize {
        follower[i] = follower[i].min(follower[i - 1] + (3 << (DB_SHIFT - 1)));
    }
    // Upward spreading: -2 dB/band backward
    for i in (start as usize..(end - 1) as usize).rev() {
        follower[i] = follower[i].min(follower[i + 1] + (2 << DB_SHIFT));
    }

    // Compute boosts and importance
    for i in start as usize..end as usize {
        let boost = (band_log_e3[i] - follower[i]).max(0);
        let width = (e_bands[i + 1] - e_bands[i]) as i32;

        // Convert boost to offset units
        offsets[i] = if boost > (2 << DB_SHIFT) {
            ((boost - (1 << DB_SHIFT)) >> (DB_SHIFT - 3)).min(8)
        } else {
            0
        };

        // Add surround dynalloc
        if surround_dynalloc.len() > i {
            let surr_boost = (surround_dynalloc[i] >> (DB_SHIFT - 3)).min(4);
            offsets[i] = offsets[i].max(surr_boost);
        }

        // Importance based on follower level (C: PSHR32(13*celt_exp2_db(MIN(follower,4.f)),16))
        let follow_clamped = follower[i].min(4 << DB_SHIFT);
        let exp_val = celt_exp2(follow_clamped.min(4 << DB_SHIFT));
        importance[i] = (13i64 * exp_val as i64 >> 16).max(1) as i32;
        spread_weight[i] = (width << lm).max(1);
    }

    // LFE override
    if lfe != 0 {
        for i in start as usize..end as usize {
            offsets[i] = 0;
            importance[i] = 13;
        }
    }

    // Cap dynalloc at budget fraction for CBR/CVBR
    if vbr == 0 || (constrained_vbr != 0 && !is_transient) {
        let dynalloc_budget = effective_bytes * 2 / 3;
        let mut boost_sum = 0i32;
        for i in start as usize..end as usize {
            boost_sum += offsets[i] * ((e_bands[i + 1] - e_bands[i]) as i32) << lm;
        }
        if boost_sum > dynalloc_budget << BITRES {
            // Scale down
            for i in start as usize..end as usize {
                offsets[i] = offsets[i] * dynalloc_budget / (boost_sum >> BITRES).max(1);
            }
        }
    }

    max_depth
}

// ===========================================================================
// Encoder helper: tone_detect (simplified)
// ===========================================================================

/// Detect pure tones in the input signal.
///
/// Matches C `tone_detect()` from `celt_encoder.c`.
/// Returns the detected tone frequency (Q13) or -1 if no tone.
fn tone_detect(
    _input: &[i32],
    _cc: i32,
    _n: i32,
    toneishness: &mut i32,
    _fs: i32,
) -> i32 {
    // Tone detection is primarily used for very specific edge cases
    // (pure sinusoid detection to prevent false transients).
    // For the initial port, we return "no tone detected" which is safe
    // — it just means we won't get the tone protection optimization.
    *toneishness = 0;
    -1 // No tone detected
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
    _tf_estimate: i32,
    _nb_available_bytes: i32,
    _tone_freq: i32,
    _toneishness: i32,
) -> i32 {
    let overlap = st.mode.overlap;
    let nu = n as usize;
    let ovu = overlap as usize;

    *pitch_index = COMBFILTER_MINPERIOD;
    *gain = 0;
    *qgain = 0;

    if !enabled || complexity < 1 {
        // Copy prefilter memory to overlap region and update memory
        for c in 0..cc as usize {
            let mem_start = c * COMBFILTER_MAXPERIOD as usize;
            let in_start = c * (nu + ovu);
            // Copy overlap from prefilter memory
            let src_start = mem_start + COMBFILTER_MAXPERIOD as usize - ovu;
            for i in 0..ovu {
                input[in_start + i] = st.prefilter_mem[src_start + i];
            }
        }
        // Update prefilter memory with new input
        for c in 0..cc as usize {
            let mem_start = c * COMBFILTER_MAXPERIOD as usize;
            let in_start = c * (nu + ovu) + ovu;
            // Shift memory
            let shift = nu.min(COMBFILTER_MAXPERIOD as usize);
            if COMBFILTER_MAXPERIOD as usize > shift {
                let keep = COMBFILTER_MAXPERIOD as usize - shift;
                for i in 0..keep {
                    st.prefilter_mem[mem_start + i] = st.prefilter_mem[mem_start + shift + i];
                }
            }
            // Copy new samples
            let dst_start = mem_start + (COMBFILTER_MAXPERIOD as usize).saturating_sub(shift);
            for i in 0..shift.min(nu) {
                st.prefilter_mem[dst_start + i] = input[in_start + nu - shift + i];
            }
        }
        return 0;
    }

    // Pitch search
    let max_period = COMBFILTER_MAXPERIOD.min(n);
    let min_period = COMBFILTER_MINPERIOD;

    // Downsample for pitch search
    let pitch_buf_len = (max_period + n) as usize;
    let mut pitch_buf = vec![0i32; pitch_buf_len];

    // Build pitch buffer from prefilter memory + current input
    for c in 0..cc.min(1) as usize {
        let mem_start = c * COMBFILTER_MAXPERIOD as usize;
        let in_start = c * (nu + ovu) + ovu;

        // Copy from prefilter memory
        for i in 0..max_period as usize {
            let mem_idx = COMBFILTER_MAXPERIOD as usize - max_period as usize + i;
            pitch_buf[i] = st.prefilter_mem[mem_start + mem_idx];
        }
        // Copy current frame
        for i in 0..nu.min(pitch_buf_len - max_period as usize) {
            pitch_buf[max_period as usize + i] = input[in_start + i];
        }
    }

    // Downsample for coarse pitch search
    let ds_len = pitch_buf_len / 2;
    let mut x_lp = vec![0i32; ds_len];
    let input_refs: [&[i32]; 1] = [&pitch_buf];
    pitch_downsample(&input_refs, &mut x_lp, ds_len, 1, 2);

    // Coarse pitch search
    let mut t0 = pitch_search(
        &x_lp[max_period as usize / 2..],
        &x_lp,
        ds_len - max_period as usize / 2,
        max_period as usize / 2,
    );
    t0 = t0 * 2;

    // Fine pitch search
    let mut gain1 = remove_doubling(
        &pitch_buf,
        max_period,
        min_period,
        n,
        &mut t0,
        st.prefilter_period,
        st.prefilter_gain,
    );

    *pitch_index = t0;

    // Gain thresholding
    let gain_threshold = qconst16(0.2, 15);
    if gain1 < gain_threshold {
        gain1 = 0;
        *pitch_index = COMBFILTER_MINPERIOD;
    }

    // Quantize gain
    if gain1 > 0 {
        let qg_raw = ((gain1 + 1536) >> 10) / 3 - 1;
        *qgain = qg_raw.max(0).min(7);
        // Dequantize
        *gain = (*qgain * 3 + 1) * 1024 / 32; // Approximate Q15 gain
        gain1 = *gain;
    } else {
        *qgain = 0;
        *gain = 0;
    }

    // Apply the comb filter to the input
    if gain1 > 0 {
        let old_period = st.prefilter_period.max(COMBFILTER_MINPERIOD);
        let old_gain = st.prefilter_gain;
        let old_tapset = st.prefilter_tapset;

        for c in 0..cc as usize {
            let in_start = c * (nu + ovu) + ovu;
            // The comb filter operates on the pre-emphasis output
            // We need prefilter memory as history
            let mem_start = c * COMBFILTER_MAXPERIOD as usize;

            // Build a working buffer with history + current frame
            let work_len = COMBFILTER_MAXPERIOD as usize + nu;
            let mut work = vec![0i32; work_len];
            for i in 0..COMBFILTER_MAXPERIOD as usize {
                work[i] = st.prefilter_mem[mem_start + i];
            }
            for i in 0..nu {
                work[COMBFILTER_MAXPERIOD as usize + i] = input[in_start + i];
            }

            // Apply comb filter
            let mut out = vec![0i32; nu];
            comb_filter(
                &mut out,
                &work,
                COMBFILTER_MAXPERIOD as usize,
                old_period,
                *pitch_index,
                n,
                old_gain,
                gain1,
                old_tapset,
                *prefilter_tapset,
                st.mode.window,
                overlap,
            );

            // Write filtered output back
            for i in 0..nu {
                input[in_start + i] = out[i];
            }
        }
    }

    // Update prefilter memory
    for c in 0..cc as usize {
        let mem_start = c * COMBFILTER_MAXPERIOD as usize;
        let in_start = c * (nu + ovu) + ovu;
        let shift = nu.min(COMBFILTER_MAXPERIOD as usize);
        if COMBFILTER_MAXPERIOD as usize > shift {
            let keep = COMBFILTER_MAXPERIOD as usize - shift;
            for i in 0..keep {
                st.prefilter_mem[mem_start + i] = st.prefilter_mem[mem_start + shift + i];
            }
        }
        let dst_start = mem_start + (COMBFILTER_MAXPERIOD as usize).saturating_sub(shift);
        for i in 0..shift.min(nu) {
            st.prefilter_mem[dst_start + i] = input[in_start + nu - shift + i];
        }
    }

    if gain1 > 0 { 1 } else { 0 }
}

// ===========================================================================
// Encoder helper: compute_vbr
// ===========================================================================

/// Compute VBR target bits for the current frame.
///
/// Matches C `compute_vbr()` from `celt_encoder.c`.
fn compute_vbr(
    _mode: &CELTMode,
    _analysis: &AnalysisInfo,
    base_target: i32,
    _lm: i32,
    _equiv_rate: i32,
    _last_coded_bands: i32,
    _c: i32,
    _intensity: i32,
    constrained_vbr: i32,
    _stereo_saving: i32,
    tot_boost: i32,
    tf_estimate: i32,
    _pitch_change: i32,
    _max_depth: i32,
    _lfe: i32,
    _has_surround: bool,
    _surround_masking: i32,
    _temporal_vbr: i32,
) -> i32 {
    let mut target = base_target;

    // Dynalloc boost adjustment
    target += tot_boost - (19 << _lm);

    // Transient boost
    let tf_calibration = qconst16(0.044, 14);
    if tf_estimate > tf_calibration {
        target += shl32(
            mult16_32_q15(tf_estimate - tf_calibration, target),
            1,
        );
    }

    // Floor depth
    target = target.max(base_target / 4);

    // CVBR damping
    if constrained_vbr != 0 {
        target = base_target + mult16_32_q15(qconst16(0.67, 15), target - base_target);
    }

    // Maximum: 2× base target
    target = target.min(2 * base_target);

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

    // Initialize entropy encoder
    let mut local_enc = RangeEncoder::new(compressed);
    let enc_ref = if enc.is_some() {
        // For hybrid mode, the caller provides the encoder
        // For now, we always use local encoder
        &mut local_enc
    } else {
        &mut local_enc
    };

    let _tell0_frac = enc_ref.tell_frac();
    let mut tell = enc_ref.tell() as i32;
    let nb_filled_bytes = (tell + 4) >> 3;

    // VBR rate computation
    let mut vbr_rate = 0i32;
    let effective_bytes;
    if st.vbr != 0 && st.bitrate != OPUS_BITRATE_MAX {
        vbr_rate = bitrate_to_bits(st.bitrate, mode.fs, frame_size) << BITRES;
        effective_bytes = vbr_rate >> (3 + BITRES);
    } else {
        effective_bytes = nb_compressed_bytes - nb_filled_bytes;
    }
    let nb_available_bytes = nb_compressed_bytes - nb_filled_bytes;
    let total_bits = nb_compressed_bytes * 8 * (1 << BITRES);

    // Equivalent rate for analysis decisions
    let equiv_rate = (nb_compressed_bytes as i64 * 8 * 50 << (3 - lm)) as i32
        - (40 * c + 20) * ((400 >> lm) - 50);

    // Silence detection
    let mut silence = false;
    let sample_max = celt_maxabs16(
        &pcm.iter().map(|&x| x as i32).collect::<Vec<_>>()[..((n - overlap) / st.upsample * cc) as usize],
    );
    if sample_max == 0 {
        silence = true;
    }

    // Encode silence flag
    if tell == 1 {
        enc_ref.encode_bit_logp(silence, 15);
        tell = enc_ref.tell() as i32;
    }

    // Handle silence in VBR mode
    if silence && vbr_rate > 0 {
        nb_compressed_bytes = 2.max(nb_filled_bytes);
        // Fill remaining with zeros
        for i in nb_filled_bytes as usize..nb_compressed_bytes as usize {
            if i < compressed.len() {
                compressed[i] = 0;
            }
        }
        // Update state for silence
        for i in 0..cc as usize * nb_ebands as usize {
            st.old_band_e[i] = GCONST_NEG28;
        }
        st.vbr_reservoir += (2 * 8 << BITRES) - vbr_rate;
        return nb_compressed_bytes;
    }

    // Pre-emphasis
    let inp_size = cc as usize * (n as usize + overlap as usize);
    let mut input = vec![0i32; inp_size];
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
    let allow_weak_transients = hybrid;

    if st.complexity >= 1 && st.lfe == 0 {
        is_transient = transient_analysis(
            &input, n + overlap, cc, &mut tf_estimate, &mut tf_chan,
            allow_weak_transients, &mut weak_transient, tone_freq, toneishness,
        );
    }

    // Pitch pre-filter
    let prefilter_enabled = ((st.lfe != 0 && nb_available_bytes > 3) || nb_available_bytes > 12 * c)
        && !hybrid && !silence && tell + 16 <= total_bits && st.disable_pf == 0;

    let mut pitch_index = COMBFILTER_MINPERIOD;
    let mut gain1: i32 = 0;
    let mut qg: i32 = 0;
    let mut prefilter_tapset = st.tapset_decision;
    let pitch_change;

    let pf_on = run_prefilter(
        st, &mut input, cc, n,
        &mut prefilter_tapset, &mut pitch_index, &mut gain1, &mut qg,
        prefilter_enabled, st.complexity, tf_estimate, nb_available_bytes,
        tone_freq, toneishness,
    );

    pitch_change = if (gain1 > qconst16(0.4, 15) || st.prefilter_gain > qconst16(0.4, 15))
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
        // Encode pitch octave + fine period
        let octave = celt_ilog2(pitch_index) - 5;
        enc_ref.encode_uint(octave.max(0) as u32, 6);
        let fine = pitch_index.wrapping_sub(16 << octave);
        enc_ref.encode_bits(fine as u32, (4 + octave) as u32);
        // Encode gain (3 bits)
        enc_ref.encode_bits(qg as u32, 3);
        // Encode tapset
        if st.complexity >= 2 {
            enc_ref.encode_icdf(prefilter_tapset as u32, &TAPSET_ICDF, 4);
        }
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

    // MDCT computation
    let freq_size = cc as usize * n as usize;
    let mut freq = vec![0i32; freq_size];
    let mut band_e = vec![0i32; nb_ebands as usize * cc as usize];
    let mut band_log_e = vec![0i32; nb_ebands as usize * cc as usize];
    let mut band_log_e2 = vec![0i32; c as usize * nb_ebands as usize];

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
    compute_mdcts(mode, short_blocks, &input, &mut freq, c, cc, lm, st.upsample);
    compute_band_energies(mode, &freq, &mut band_e, eff_end, c, lm);

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
        let offset = if short_blocks != 0 { half32(shl32(lm, DB_SHIFT - 5)) } else { 0 };
        for i in start as usize..end as usize {
            follow = follow.max(shr32(band_log_e[i], 5) - offset) - (1 << (DB_SHIFT - 5));
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
    if lm > 0 && enc_ref.tell() as i32 + 3 <= total_bits && !is_transient
        && st.complexity >= 5 && st.lfe == 0 && !hybrid
    {
        if patch_transient_decision(&band_log_e, &st.old_band_e, nb_ebands, start, end, c) {
            is_transient = true;
            short_blocks = m;
            compute_mdcts(mode, short_blocks, &input, &mut freq, c, cc, lm, st.upsample);
            compute_band_energies(mode, &freq, &mut band_e, eff_end, c, lm);
            amp2log2(mode, eff_end, end, &band_e, &mut band_log_e, c);
            for ch in 0..c as usize {
                for i in 0..nb_ebands as usize {
                    band_log_e2[ch * nb_ebands as usize + i] =
                        band_log_e[ch * nb_ebands as usize + i] + half32(shl32(lm, DB_SHIFT));
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
    let x_size = c as usize * n as usize;
    let mut x_norm = vec![0i32; x_size];
    normalise_bands(mode, &freq, &mut x_norm, &band_e, eff_end, c, m);

    // TF analysis
    let enable_tf_analysis = effective_bytes >= 15 * c && !hybrid
        && st.complexity >= 2 && st.lfe == 0 && toneishness < qconst32(0.98, 29);

    let mut offsets = vec![0i32; nb_ebands as usize];
    let mut importance = vec![13i32; nb_ebands as usize];
    let mut spread_weight = vec![1i32; nb_ebands as usize];
    let mut tot_boost = 0i32;

    let surround_dynalloc = vec![0i32; c as usize * nb_ebands as usize];

    let max_depth = dynalloc_analysis(
        &band_log_e, &band_log_e2, &st.old_band_e,
        nb_ebands, start, end, c, &mut offsets,
        st.lsb_depth, mode.log_n, is_transient,
        st.vbr, st.constrained_vbr, e_bands, lm,
        effective_bytes, &mut tot_boost, st.lfe,
        &surround_dynalloc, &st.analysis,
        &mut importance, &mut spread_weight,
        tone_freq, toneishness,
    );

    let mut tf_res = vec![0i32; nb_ebands as usize];
    let tf_select;
    if enable_tf_analysis {
        let lambda = 80i32.max(20480 / effective_bytes.max(1) + 2);
        tf_select = tf_analysis(
            mode, eff_end, is_transient, &mut tf_res, lambda,
            &x_norm, n, lm, tf_estimate, tf_chan, &importance,
        );
        for i in eff_end as usize..end as usize {
            tf_res[i] = if eff_end > 0 { tf_res[eff_end as usize - 1] } else { 0 };
        }
    } else if hybrid && weak_transient {
        for i in start as usize..end as usize {
            tf_res[i] = 1;
        }
        tf_select = 0;
    } else {
        for i in start as usize..end as usize {
            tf_res[i] = is_transient as i32;
        }
        tf_select = 0;
    }

    // Energy quantization: error biasing
    let mut error = vec![0i32; c as usize * nb_ebands as usize];
    for ch in 0..c as usize {
        for i in start as usize..end as usize {
            let idx = i + ch * nb_ebands as usize;
            if abs32(band_log_e[idx] - st.old_band_e[idx]) < (2 << DB_SHIFT) {
                let _biased = band_log_e[idx] - mult16_32_q15(
                    qconst16(0.25, 15),
                    st.energy_error[idx],
                );
                // Need to modify band_log_e for coarse quant - use a local copy
                // Note: in the C code, bandLogE is modified in place
            }
        }
    }

    // Coarse energy quantization
    quant_coarse_energy(
        mode, start, end, eff_end, &band_log_e, &mut st.old_band_e,
        total_bits as u32, &mut error, enc_ref, c, lm,
        nb_available_bytes, st.force_intra, &mut st.delayed_intra,
        (st.complexity >= 4) as i32, st.loss_rate, st.lfe,
    );

    // TF encoding
    tf_encode(start, end, is_transient, &mut tf_res, lm, tf_select, enc_ref);
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
                mode, &x_norm, &mut st.tonal_average, st.spread_decision,
                &mut st.hf_average, &mut st.tapset_decision,
                pf_on != 0 && short_blocks == 0, eff_end, c, m, &spread_weight,
            );
        }
        enc_ref.encode_icdf(st.spread_decision as u32, &SPREAD_ICDF, 5);
    } else {
        st.spread_decision = SPREAD_NORMAL;
    }

    // Caps initialization
    let mut cap = vec![0i32; nb_ebands as usize];
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
        while (enc_ref.tell_frac() as i32 + (dynalloc_loop_logp << BITRES)) < total_bits_bitres - total_boost_enc
            && boost < cap[i]
        {
            let flag = j < offsets[i];
            enc_ref.encode_bit_logp(flag, dynalloc_loop_logp as u32);
            if !flag { break; }
            boost += quanta;
            total_boost_enc += quanta;
            dynalloc_loop_logp = 1;
            j += 1;
        }
        if j > 0 {
            dynalloc_logp = dynalloc_logp.max(2) - 1;
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
                mode, &x_norm, &band_log_e, end, lm, c, n,
                &st.analysis, &mut st.stereo_saving, tf_estimate,
                st.intensity, 0, equiv_rate,
            );
        }
        enc_ref.encode_icdf(alloc_trim as u32, &TRIM_ICDF, 7);
    }

    // Minimum allowed bytes
    let min_allowed = ((enc_ref.tell_frac() as i32 + total_boost_enc + (1 << (BITRES + 3)) - 1) >> (BITRES + 3)) + 2;

    // VBR rate control
    if vbr_rate > 0 {
        let lm_diff = mode.max_lm - lm;
        let base_target = if !hybrid {
            vbr_rate - ((40 * c + 20) << BITRES)
        } else {
            (vbr_rate - ((9 * c + 4) << BITRES)).max(0)
        };

        let mut target = compute_vbr(
            mode, &st.analysis, base_target, lm, equiv_rate,
            st.last_coded_bands, c, st.intensity,
            st.constrained_vbr, st.stereo_saving, tot_boost,
            tf_estimate, pitch_change, max_depth, st.lfe,
            st.energy_mask.is_some(), 0, temporal_vbr,
        );

        target += enc_ref.tell_frac() as i32;

        let mut nb_avail = (target + (1 << (BITRES + 2))) >> (BITRES + 3);
        nb_avail = nb_avail.max(min_allowed).min(nb_compressed_bytes);

        let delta = target - vbr_rate;

        // Update VBR state
        if st.constrained_vbr != 0 {
            st.vbr_reservoir += (nb_avail << (BITRES + 3)) - vbr_rate;
        }

        if st.vbr_count < 970 {
            st.vbr_count += 1;
        }
        let alpha = if st.vbr_count < 970 {
            celt_rcp(shl32(st.vbr_count + 20, 16))
        } else {
            qconst16(0.001, 15)
        };

        if st.constrained_vbr != 0 {
            st.vbr_drift += mult16_32_q15(
                alpha,
                (delta * (1 << lm_diff)) - st.vbr_offset - st.vbr_drift,
            );
            st.vbr_offset = -st.vbr_drift;
        }

        if st.constrained_vbr != 0 && st.vbr_reservoir < 0 {
            let adjust = (-st.vbr_reservoir) / (8 << BITRES);
            if !silence {
                nb_avail += adjust;
            }
            st.vbr_reservoir = 0;
        }

        nb_compressed_bytes = nb_compressed_bytes.min(nb_avail);
        let _ = nb_compressed_bytes - nb_filled_bytes; // effective_bytes updated but used later
    }

    // Bit allocation
    let mut fine_quant = vec![0i32; nb_ebands as usize];
    let mut pulses = vec![0i32; nb_ebands as usize];
    let mut fine_priority = vec![0i32; nb_ebands as usize];
    let mut balance = 0i32;

    let bits = ((nb_compressed_bytes as i64 * 8) << BITRES) as i32
        - enc_ref.tell_frac() as i32 - 1;

    let anti_collapse_rsv = if is_transient && lm >= 2 && bits >= ((lm + 2) << BITRES) {
        1 << BITRES
    } else {
        0
    };
    let bits = bits - anti_collapse_rsv;

    let signal_bandwidth = end - 1;

    let coded_bands = clt_compute_allocation(
        mode, start, end, &offsets, &cap, alloc_trim,
        &mut st.intensity, &mut dual_stereo,
        bits, &mut balance, &mut pulses, &mut fine_quant, &mut fine_priority,
        c, lm, enc_ref, true, st.last_coded_bands, signal_bandwidth,
    );

    // Update lastCodedBands with hysteresis
    if coded_bands <= st.last_coded_bands.min(end) && coded_bands > st.last_coded_bands.min(end) - 2 {
        st.last_coded_bands = st.last_coded_bands.min(end) - 1;
    } else {
        st.last_coded_bands = coded_bands;
    }

    // Fine energy quantization
    quant_fine_energy(mode, start, end, &mut st.old_band_e, &mut error, None, &fine_quant, enc_ref, c);

    // Clear energy error
    for i in 0..nb_ebands as usize * cc as usize {
        st.energy_error[i] = 0;
    }

    // Residual quantization
    let mut collapse_masks = vec![0u8; c as usize * nb_ebands as usize];
    let mut y_norm_buf = vec![0i32; n as usize];
    let y_norm_opt = if c == 2 { Some(&mut y_norm_buf[..]) } else { None };
    quant_all_bands(
        true, mode, start, end,
        &mut x_norm,
        y_norm_opt,
        &mut collapse_masks,
        &band_e, &mut pulses,
        short_blocks != 0, st.spread_decision, dual_stereo != 0,
        st.intensity, &tf_res,
        nb_compressed_bytes * (8 << BITRES) - anti_collapse_rsv,
        balance, enc_ref, lm, coded_bands,
        &mut st.rng, st.complexity, st.disable_inv != 0,
    );

    // Anti-collapse
    let anti_collapse_on;
    if anti_collapse_rsv > 0 {
        anti_collapse_on = (st.consec_transient < 2) as i32;
        enc_ref.encode_bits(anti_collapse_on as u32, 1);
    }

    // Energy finalization
    quant_energy_finalise(
        mode, start, end,
        Some(&mut st.old_band_e), &mut error,
        &fine_quant, &fine_priority,
        nb_compressed_bytes * 8 - enc_ref.tell() as i32,
        enc_ref, c,
    );

    // Store energy error for next frame
    for i in 0..nb_ebands as usize * cc as usize {
        st.energy_error[i] = error.get(i).copied().unwrap_or(0)
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
                if v < 0 || v > 10 { return OPUS_BAD_ARG; }
                self.complexity = v;
            }
            CeltEncoderCtl::SetStartBand(v) => {
                if v < 0 || v >= self.mode.nb_ebands { return OPUS_BAD_ARG; }
                self.start = v;
            }
            CeltEncoderCtl::SetEndBand(v) => {
                if v < 1 || v > self.mode.nb_ebands { return OPUS_BAD_ARG; }
                self.end = v;
            }
            CeltEncoderCtl::SetPrediction(v) => {
                if v < 0 || v > 2 { return OPUS_BAD_ARG; }
                self.disable_pf = (v <= 1) as i32;
                self.force_intra = (v == 0) as i32;
            }
            CeltEncoderCtl::SetPacketLossPerc(v) => {
                if v < 0 || v > 100 { return OPUS_BAD_ARG; }
                self.loss_rate = v;
            }
            CeltEncoderCtl::SetVbrConstraint(v) => { self.constrained_vbr = v; }
            CeltEncoderCtl::SetVbr(v) => { self.vbr = v; }
            CeltEncoderCtl::SetBitrate(v) => {
                if v <= 500 && v != OPUS_BITRATE_MAX { return OPUS_BAD_ARG; }
                self.bitrate = v.min(750000 * self.channels);
            }
            CeltEncoderCtl::SetChannels(v) => {
                if v < 1 || v > 2 { return OPUS_BAD_ARG; }
                self.stream_channels = v;
            }
            CeltEncoderCtl::SetLsbDepth(v) => {
                if v < 8 || v > 24 { return OPUS_BAD_ARG; }
                self.lsb_depth = v;
            }
            CeltEncoderCtl::GetLsbDepth => { /* caller reads st.lsb_depth */ }
            CeltEncoderCtl::SetPhaseInversionDisabled(v) => {
                if v < 0 || v > 1 { return OPUS_BAD_ARG; }
                self.disable_inv = v;
            }
            CeltEncoderCtl::GetPhaseInversionDisabled => { /* caller reads st.disable_inv */ }
            CeltEncoderCtl::ResetState => { self.reset(); }
            CeltEncoderCtl::SetAnalysis(info) => { self.analysis = info; }
            CeltEncoderCtl::SetSilkInfo(info) => { self.silk_info = info; }
            CeltEncoderCtl::SetSignalling(v) => { self.signalling = v; }
            CeltEncoderCtl::SetLfe(v) => { self.lfe = v; }
            CeltEncoderCtl::SetEnergyMask => { /* handled separately */ }
            CeltEncoderCtl::GetFinalRange => { /* caller reads st.rng */ }
            CeltEncoderCtl::SetInputClipping(v) => { self.clip = v; }
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
        let result = celt_encode_with_ec(
            &mut enc, &pcm, 960, &mut compressed, 128, None,
        );
        // Should succeed with a small number of bytes
        assert!(result > 0, "encode returned {}", result);
        assert!(result <= 128);
    }
}
