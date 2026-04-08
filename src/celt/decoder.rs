//! CELT decoder — frequency-domain audio decoder within the Opus codec.
//!
//! Matches `celt_decoder.c` and parts of `celt.c` from the C reference
//! (FIXED_POINT path, non-QEXT, non-DEEP_PLC).

use super::bands::{anti_collapse, celt_lcg_rand, denormalise_bands, quant_all_bands};
use super::lpc::{celt_autocorr, celt_fir, celt_iir, celt_lpc};
use super::math_ops::{celt_maxabs16, celt_sqrt, celt_zlog2, frac_div32};
use super::mdct::{MDCT_48000_960, clt_mdct_backward};
use super::modes::{
    CELTMode, DEC_PITCH_BUF_SIZE, MAX_PERIOD, MODE_48000_960_120, NB_EBANDS, init_caps,
    resampling_factor,
};
use super::pitch::{pitch_downsample, pitch_search};
use super::quant_bands::{unquant_coarse_energy, unquant_energy_finalise, unquant_fine_energy};
use super::range_coder::RangeDecoder;
use super::rate::{BITRES, clt_compute_allocation};
use super::vq::renormalise_vector;
use crate::types::*;

// ===========================================================================
// DNN PLC argument type (conditional on feature flag)
// ===========================================================================

#[cfg(feature = "dnn")]
pub type DnnPlcArg<'a> = Option<&'a mut crate::dnn::lpcnet::LPCNetPLCState>;
#[cfg(not(feature = "dnn"))]
pub type DnnPlcArg<'a> = ();

// ===========================================================================
// Constants
// ===========================================================================

/// Decode buffer size per channel in samples. Same as DEC_PITCH_BUF_SIZE.
const DECODE_BUFFER_SIZE: i32 = DEC_PITCH_BUF_SIZE;

/// Maximum pitch period for comb filter.
#[allow(dead_code)]
const COMBFILTER_MAXPERIOD: i32 = 1024;
/// Minimum pitch period for comb filter.
const COMBFILTER_MINPERIOD: i32 = 15;

/// Maximum pitch lag for PLC search (66.67 Hz at 48 kHz).
const PLC_PITCH_LAG_MAX: i32 = 720;
/// Minimum pitch lag for PLC search (480 Hz at 48 kHz).
const PLC_PITCH_LAG_MIN: i32 = 100;

// --- Frame type constants ---
const FRAME_NONE: i32 = 0;
const FRAME_NORMAL: i32 = 1;
const FRAME_PLC_NOISE: i32 = 2;
const FRAME_PLC_PERIODIC: i32 = 3;
#[cfg(feature = "dnn")]
const FRAME_PLC_NEURAL: i32 = 5;
#[cfg(feature = "dnn")]
const FRAME_DRED: i32 = 6;

// ===========================================================================
// Static tables
// ===========================================================================

/// iCDF for allocation trim (11 values: 0–10).
static TRIM_ICDF: [u8; 11] = [126, 124, 119, 109, 87, 41, 19, 9, 4, 2, 0];

/// iCDF for spread decision (4 values: NONE/LIGHT/NORMAL/AGGRESSIVE).
static SPREAD_ICDF: [u8; 4] = [25, 23, 2, 0];

/// iCDF for postfilter tapset (3 values: 0/1/2).
static TAPSET_ICDF: [u8; 3] = [2, 1, 0];

/// Time-frequency resolution select table [LM][4*isTransient + 2*tf_select + tf_changed].
static TF_SELECT_TABLE: [[i8; 8]; 4] = [
    [0, -1, 0, -1, 0, -1, 0, -1],
    [0, -1, 0, -2, 1, 0, 1, -1],
    [0, -2, 0, -3, 2, 0, 1, -1],
    [0, -2, 0, -3, 3, 0, 1, -1],
];

/// Comb filter tap gains for 3-tap post-filter, Q15.
/// Indexed by tapset (0, 1, 2) × tap (center, ±1, ±2).
static COMB_GAINS: [[i32; 3]; 3] = [
    [
        qconst16(0.3066406250, 15),
        qconst16(0.2170410156, 15),
        qconst16(0.1296386719, 15),
    ],
    [qconst16(0.4638671875, 15), qconst16(0.2680664062, 15), 0],
    [qconst16(0.7998046875, 15), qconst16(0.1000976562, 15), 0],
];

/// Q24 constant helper for celt_glog values.
#[inline(always)]
const fn gconst(x: f64) -> i32 {
    qconst32(x, DB_SHIFT as u32)
}

// ===========================================================================
// CeltDecoder struct
// ===========================================================================

/// CELT decoder state.
///
/// Matches `OpusCustomDecoder` in the C reference. Variable-length trailing
/// arrays are modeled as separate `Vec` fields instead of a single flexible
/// array member with pointer arithmetic.
pub struct CeltDecoder {
    // --- Configuration (persist across reset) ---
    pub mode: &'static CELTMode,
    pub overlap: i32,
    pub channels: i32,
    pub stream_channels: i32,
    pub downsample: i32,
    pub start: i32,
    pub end: i32,
    pub signalling: bool,
    pub disable_inv: bool,
    pub complexity: i32,

    // --- Dynamic state (cleared on reset at DECODER_RESET_START) ---
    pub rng: u32,
    pub error: i32,
    pub last_pitch_index: i32,
    pub loss_duration: i32,
    pub plc_duration: i32,
    pub last_frame_type: i32,
    pub skip_plc: i32,
    pub postfilter_period: i32,
    pub postfilter_period_old: i32,
    pub postfilter_gain: i32,
    pub postfilter_gain_old: i32,
    pub postfilter_tapset: i32,
    pub postfilter_tapset_old: i32,
    pub prefilter_and_fold: bool,
    pub preemph_mem_d: [i32; 2],

    // --- Variable-length arrays ---
    /// Per-channel decode buffer: channels × (DECODE_BUFFER_SIZE + overlap) samples.
    decode_mem: Vec<i32>,
    /// Current band energies (log domain, Q24). Length: 2 × nb_ebands.
    old_band_e: Vec<i32>,
    /// Previous frame band energies. Length: 2 × nb_ebands.
    old_log_e: Vec<i32>,
    /// Two frames ago band energies. Length: 2 × nb_ebands.
    old_log_e2: Vec<i32>,
    /// Background noise estimate per band. Length: 2 × nb_ebands.
    background_log_e: Vec<i32>,
    /// LPC coefficients for PLC. Length: channels × CELT_LPC_ORDER.
    lpc_coef: Vec<i32>,

    // --- DNN neural PLC state (behind feature flag) ---
    /// 16kHz PLC buffer for neural concealment (LPCNet operates at 16kHz).
    #[cfg(feature = "dnn")]
    plc_pcm: Vec<i16>,
    /// Number of samples filled in plc_pcm.
    #[cfg(feature = "dnn")]
    plc_fill: usize,
    /// Preemphasis filter memory for 48kHz→16kHz downsampling.
    #[cfg(feature = "dnn")]
    plc_preemphasis_mem: f32,
}

// ===========================================================================
// Comb filter (from celt.c)
// ===========================================================================

/// Constant-parameter comb filter (after the overlap transition region).
/// Matches C `comb_filter_const` (non-ARM path, FIXED_POINT).
#[allow(dead_code)]
fn comb_filter_const(y: &mut [i32], x: &[i32], t: i32, n: i32, g10: i32, g11: i32, g12: i32) {
    let t = t as usize;
    debug_assert!(t >= COMBFILTER_MINPERIOD as usize);
    // Preload sliding window: x2=x[-T], x1=x[-T+1], x0=x[-T+2]
    let mut x4 = x[0_usize.wrapping_sub(t).wrapping_sub(2)]; // x[i-T-2] at i=0
    let mut x3 = x[0_usize.wrapping_sub(t).wrapping_sub(1)]; // x[i-T-1]
    let mut x2 = x[0_usize.wrapping_sub(t)]; // x[i-T]
    let mut x1 = x[1_usize.wrapping_sub(t)]; // x[i-T+1]
    let mut x0: i32;
    for i in 0..n as usize {
        x0 = x[i + 2 - t]; // x[i-T+2] — this wraps correctly because x points into decode_mem
        let val = x[i]
            + mult16_32_q15(g10, x2)
            + mult16_32_q15(g11, add32(x1, x3))
            + mult16_32_q15(g12, add32(x0, x4));
        // Fixed-point bias correction: subtract 1 to avoid systematic rounding bias
        y[i] = saturate(val - 1, SIG_SAT);
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
    }
}

/// Apply the post-filter comb filter with overlap crossfade.
/// Matches C `comb_filter()` from celt.c (FIXED_POINT path).
///
/// `buf` is the signal buffer. `buf_off` is the offset into buf where
/// processing starts. The function reads samples at negative offsets
/// relative to `buf_off` (the pitch history).
fn comb_filter(
    buf: &mut [i32],
    buf_off: usize,
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
    if g0 == 0 && g1 == 0 {
        // No filtering needed
        return;
    }
    let t0 = imax(t0, COMBFILTER_MINPERIOD) as usize;
    let t1 = imax(t1, COMBFILTER_MINPERIOD) as usize;

    // Compute gain × tap products (rounded multiply, matching C MULT_COEF_TAPS)
    let g00 = mult16_16_p15(g0, COMB_GAINS[tapset0 as usize][0]);
    let g01 = mult16_16_p15(g0, COMB_GAINS[tapset0 as usize][1]);
    let g02 = mult16_16_p15(g0, COMB_GAINS[tapset0 as usize][2]);
    let g10 = mult16_16_p15(g1, COMB_GAINS[tapset1 as usize][0]);
    let g11 = mult16_16_p15(g1, COMB_GAINS[tapset1 as usize][1]);
    let g12 = mult16_16_p15(g1, COMB_GAINS[tapset1 as usize][2]);

    let overlap = overlap as usize;

    // Check if parameters changed — skip overlap if identical
    let params_changed = g0 != g1 || t0 != t1 || tapset0 != tapset1;

    // Overlap region: crossfade from old to new parameters
    // Matches C: per-tap windowed gains, then applied to signal
    if params_changed {
        // Preload x values for new filter (T1)
        let mut x1 = buf[buf_off.wrapping_sub(t1) + 1];
        let mut x2 = buf[buf_off.wrapping_sub(t1)];
        let mut x3 = buf[buf_off.wrapping_sub(t1).wrapping_sub(1)];
        let mut x4 = buf[buf_off.wrapping_sub(t1).wrapping_sub(2)];

        for i in 0..overlap {
            // Squared window: f = MULT_COEF(window[i], window[i]) = MULT16_16_Q15
            let f = mult16_16_q15(window[i] as i32, window[i] as i32);
            let one_minus_f = Q15ONE - f;

            let x0 = buf[buf_off + i + 2 - t1];

            // Per-tap windowed gains applied to signal (matching C structure)
            let mut val = buf[buf_off + i]
                + mult16_32_q15(mult16_16_q15(one_minus_f, g00), buf[buf_off + i - t0])
                + mult16_32_q15(
                    mult16_16_q15(one_minus_f, g01),
                    add32(buf[buf_off + i - t0 + 1], buf[buf_off + i - t0 - 1]),
                )
                + mult16_32_q15(
                    mult16_16_q15(one_minus_f, g02),
                    add32(buf[buf_off + i - t0 + 2], buf[buf_off + i - t0 - 2]),
                )
                + mult16_32_q15(mult16_16_q15(f, g10), x2)
                + mult16_32_q15(mult16_16_q15(f, g11), add32(x1, x3))
                + mult16_32_q15(mult16_16_q15(f, g12), add32(x0, x4));
            // Fixed-point bias (matching C: SUB32(y[i], 3))
            val -= 3;
            buf[buf_off + i] = saturate(val, SIG_SAT);

            x4 = x3;
            x3 = x2;
            x2 = x1;
            x1 = x0;
        }
    }

    let start = if params_changed { overlap } else { 0 };

    if g1 == 0 {
        return;
    }

    // Constant region after overlap
    if n as usize > start {
        // Use the const filter path for the remaining samples
        let off = buf_off + start;
        let count = n as usize - start;
        // We need to work with slices that allow negative-offset reads relative to off.
        // Since buf[off - T1] must be valid, we work directly.
        for i in 0..count {
            let idx = off + i;
            let x0 = buf[idx + 2 - t1];
            let x2 = buf[idx - t1];
            let x1 = buf[idx + 1 - t1];
            let x3 = buf[idx - 1 - t1];
            let x4 = buf[idx - 2 - t1];
            let val = buf[idx]
                + mult16_32_q15(g10, x2)
                + mult16_32_q15(g11, add32(x1, x3))
                + mult16_32_q15(g12, add32(x0, x4));
            buf[idx] = saturate(val - 1, SIG_SAT);
        }
    }
}

// ===========================================================================
// De-emphasis filter (from celt_decoder.c)
// ===========================================================================

/// De-emphasis filter: first-order IIR y[n] = x[n] + coef0 * y[n-1].
/// Then convert from signal domain (SIG_SHIFT=12) to i16 output.
///
/// Handles downsampling and accumulation modes.
/// Matches C `deemphasis()` (FIXED_POINT, non-custom-mode path).
fn deemphasis(
    inp: &[&[i32]],  // per-channel input signal slices
    pcm: &mut [i16], // interleaved output
    n: i32,
    cc: i32, // output channels
    downsample: i32,
    coef: &[i32; 4],    // preemph coefficients
    mem: &mut [i32; 2], // per-channel IIR state
    accum: bool,
) {
    let coef0 = coef[0];
    let nd = n / downsample;

    for c in 0..cc as usize {
        let x = inp[c];
        let mut m = mem[c];

        if downsample == 1 {
            // Fast path: no downsampling
            for j in 0..n as usize {
                let tmp = saturate(x[j] + VERY_SMALL + m, SIG_SAT);
                m = mult16_32_q15(coef0, tmp);
                let out = sig2word16(tmp);
                if accum {
                    // Saturated add
                    let prev = pcm[j * cc as usize + c] as i32;
                    pcm[j * cc as usize + c] = sat16(prev + out);
                } else {
                    pcm[j * cc as usize + c] = out as i16;
                }
            }
        } else {
            // Downsampling path: filter all samples, pick every downsample-th
            let mut scratch = vec![0i32; n as usize];
            for j in 0..n as usize {
                let tmp = saturate(x[j] + VERY_SMALL + m, SIG_SAT);
                m = mult16_32_q15(coef0, tmp);
                scratch[j] = tmp;
            }
            for j in 0..nd as usize {
                let out = sig2word16(scratch[j * downsample as usize]);
                if accum {
                    let prev = pcm[j * cc as usize + c] as i32;
                    pcm[j * cc as usize + c] = sat16(prev + out);
                } else {
                    pcm[j * cc as usize + c] = out as i16;
                }
            }
        }
        mem[c] = m;
    }
}

// ===========================================================================
// TF (time-frequency) resolution decode (from celt.c)
// ===========================================================================

/// Decode time-frequency resolution flags per band.
/// Matches C `tf_decode()`.
fn tf_decode(
    start: i32,
    end: i32,
    is_transient: bool,
    tf_res: &mut [i32],
    lm: i32,
    dec: &mut RangeDecoder,
) -> i32 {
    let budget = dec.storage() * 8;
    let mut tell = dec.tell() as u32;
    let logp: u32 = if is_transient { 2 } else { 4 };
    let tf_select_rsv = if lm > 0 && tell + logp + 1 <= budget {
        1
    } else {
        0
    };
    let effective_budget = budget - tf_select_rsv;
    let mut tf_changed: i32 = 0;
    let mut curr: i32 = 0;
    let mut logp = logp;

    for i in start..end {
        if tell + logp <= effective_budget as u32 {
            curr ^= if dec.decode_bit_logp(logp as u32) {
                1
            } else {
                0
            };
            tell = dec.tell() as u32;
            tf_changed |= curr;
        }
        tf_res[i as usize] = curr;
        logp = if is_transient { 4 } else { 5 };
    }

    let is_trans_i = if is_transient { 1i32 } else { 0 };
    let mut tf_select = 0i32;
    if tf_select_rsv != 0
        && TF_SELECT_TABLE[lm as usize][(4 * is_trans_i + 0 + tf_changed) as usize]
            != TF_SELECT_TABLE[lm as usize][(4 * is_trans_i + 2 + tf_changed) as usize]
    {
        tf_select = if dec.decode_bit_logp(1) { 1 } else { 0 };
    }
    for i in start..end {
        tf_res[i as usize] = TF_SELECT_TABLE[lm as usize]
            [(4 * is_trans_i + 2 * tf_select + tf_res[i as usize]) as usize]
            as i32;
    }
    tf_select
}

// ===========================================================================
// CELT synthesis (from celt_decoder.c)
// ===========================================================================

/// Synthesize time-domain audio from normalized MDCT coefficients.
/// Applies band denormalization, inverse MDCT, and output saturation.
/// Matches C `celt_synthesis()` (FIXED_POINT, non-QEXT path).
fn celt_synthesis(
    mode: &CELTMode,
    x: &mut [i32],
    out_syn: &mut [i32],       // flat buffer: CC * (DECODE_BUFFER_SIZE + overlap)
    out_syn_offsets: &[usize], // per-channel offset where out_syn[c] starts
    old_band_e: &[i32],
    start: i32,
    eff_end: i32,
    c: i32,  // stream channels
    cc: i32, // output channels
    is_transient: bool,
    lm: i32,
    downsample: i32,
    silence: bool,
) {
    let overlap = mode.overlap;
    let nb_ebands = mode.nb_ebands;
    let n = mode.short_mdct_size << lm;
    let big_m = 1 << lm;
    let (b, nb, shift) = if is_transient {
        (big_m, mode.short_mdct_size, mode.max_lm)
    } else {
        (1, mode.short_mdct_size << lm, mode.max_lm - lm)
    };

    let mut freq = [0i32; 960]; // max N = short_mdct_size * nb_short_mdcts = 120 * 8

    if cc == 2 && c == 1 {
        // Mono stream → stereo output: decode once, IMDCT to both channels
        denormalise_bands(
            mode, x, &mut freq, old_band_e, start, eff_end, big_m, downsample, silence,
        );

        // IMDCT for channel 0
        for b_idx in 0..b {
            let out_off = out_syn_offsets[0] + (nb * b_idx) as usize;
            clt_mdct_backward(
                &MDCT_48000_960,
                &freq[b_idx as usize..],
                &mut out_syn[out_off..],
                mode.window,
                overlap,
                shift,
                b,
            );
        }
        // IMDCT for channel 1 (same freq data)
        for b_idx in 0..b {
            let out_off = out_syn_offsets[1] + (nb * b_idx) as usize;
            clt_mdct_backward(
                &MDCT_48000_960,
                &freq[b_idx as usize..],
                &mut out_syn[out_off..],
                mode.window,
                overlap,
                shift,
                b,
            );
        }
    } else if cc == 1 && c == 2 {
        // Stereo stream → mono output: decode both, average, IMDCT once
        denormalise_bands(
            mode, x, &mut freq, old_band_e, start, eff_end, big_m, downsample, silence,
        );
        let mut freq2 = [0i32; 960];
        denormalise_bands(
            mode,
            &x[n as usize..],
            &mut freq2,
            &old_band_e[nb_ebands as usize..],
            start,
            eff_end,
            big_m,
            downsample,
            silence,
        );
        // Downmix: average
        for i in 0..n as usize {
            freq[i] = half32(freq[i]) + half32(freq2[i]);
        }
        for b_idx in 0..b {
            let out_off = out_syn_offsets[0] + (nb * b_idx) as usize;
            clt_mdct_backward(
                &MDCT_48000_960,
                &freq[b_idx as usize..],
                &mut out_syn[out_off..],
                mode.window,
                overlap,
                shift,
                b,
            );
        }
    } else {
        // Normal case: CC == C
        for ch in 0..cc as usize {
            denormalise_bands(
                mode,
                &x[ch * n as usize..],
                &mut freq,
                &old_band_e[ch * nb_ebands as usize..],
                start,
                eff_end,
                big_m,
                downsample,
                silence,
            );

            for b_idx in 0..b {
                let out_off = out_syn_offsets[ch] + (nb * b_idx) as usize;
                clt_mdct_backward(
                    &MDCT_48000_960,
                    &freq[b_idx as usize..],
                    &mut out_syn[out_off..],
                    mode.window,
                    overlap,
                    shift,
                    b,
                );
            }
        }
    }

    // Saturate output to SIG_SAT
    for ch in 0..cc as usize {
        let off = out_syn_offsets[ch];
        for i in 0..n as usize {
            out_syn[off + i] = saturate(out_syn[off + i], SIG_SAT);
        }
    }
}

// ===========================================================================
// Prefilter and fold (from celt_decoder.c)
// ===========================================================================

/// Apply inverse postfilter to MDCT overlap region and TDAC simulation.
/// Matches C `prefilter_and_fold()`.
fn prefilter_and_fold(
    buf: &mut [i32],       // flat decode_mem buffer
    buf_offsets: &[usize], // per-channel offsets
    cc: i32,
    overlap: i32,
    n: i32,
    postfilter_period_old: i32,
    postfilter_period: i32,
    postfilter_gain_old: i32,
    postfilter_gain: i32,
    postfilter_tapset_old: i32,
    postfilter_tapset: i32,
    window: &[i16],
) {
    let overlap = overlap as usize;
    let decode_buffer_size = DECODE_BUFFER_SIZE as usize;

    for c in 0..cc as usize {
        let off = buf_offsets[c];
        let mut etmp = [0i32; 120]; // max overlap = 120

        // Apply inverse comb filter (negative gains!) to overlap region
        // Copy the overlap samples from decode_mem
        let src_off = off + decode_buffer_size - n as usize;
        for i in 0..overlap {
            etmp[i] = buf[src_off + i];
        }

        // Apply inverse filter with negative gains
        {
            let _t0 = imax(postfilter_period_old, COMBFILTER_MINPERIOD) as usize;
            let t1 = imax(postfilter_period, COMBFILTER_MINPERIOD) as usize;
            let neg_g0 = -postfilter_gain_old;
            let neg_g1 = -postfilter_gain;

            if neg_g0 != 0 || neg_g1 != 0 {
                let _g00 = mult16_16_p15(neg_g0, COMB_GAINS[postfilter_tapset_old as usize][0]);
                let _g01 = mult16_16_p15(neg_g0, COMB_GAINS[postfilter_tapset_old as usize][1]);
                let _g02 = mult16_16_p15(neg_g0, COMB_GAINS[postfilter_tapset_old as usize][2]);
                let g10 = mult16_16_p15(neg_g1, COMB_GAINS[postfilter_tapset as usize][0]);
                let g11 = mult16_16_p15(neg_g1, COMB_GAINS[postfilter_tapset as usize][1]);
                let g12 = mult16_16_p15(neg_g1, COMB_GAINS[postfilter_tapset as usize][2]);

                // No window crossfade for prefilter_and_fold (window=NULL, overlap=0 in C call)
                // Just apply the new filter directly
                for i in 0..overlap {
                    let idx = src_off + i;
                    let val = buf[idx]
                        + mult16_32_q15(g10, buf[idx - t1])
                        + mult16_32_q15(g11, add32(buf[idx + 1 - t1], buf[idx - 1 - t1]))
                        + mult16_32_q15(g12, add32(buf[idx + 2 - t1], buf[idx - 2 - t1]));
                    // Fixed-point bias correction matching C comb_filter_const_c: SUB32(y[i], 1)
                    etmp[i] = saturate(val - 1, SIG_SAT);
                }
            }
        }

        // Apply TDAC windowing: fold the overlap region
        for i in 0..overlap / 2 {
            let w_rise = window[i] as i32;
            let w_fall = window[overlap - 1 - i] as i32;
            buf[off + decode_buffer_size - n as usize + i] =
                mult16_32_q15(w_fall, etmp[overlap - 1 - i]) + mult16_32_q15(w_rise, etmp[i]);
        }
    }
}

// ===========================================================================
// CeltDecoder implementation
// ===========================================================================

impl CeltDecoder {
    /// Allocate and initialize a new CELT decoder.
    ///
    /// `sampling_rate` — output rate (8000/12000/16000/24000/48000).
    /// `channels` — output channels (1 or 2).
    pub fn new(sampling_rate: i32, channels: i32) -> Result<Self, i32> {
        if channels < 1 || channels > 2 {
            return Err(-1); // OPUS_BAD_ARG
        }
        let ds = resampling_factor(sampling_rate);
        if ds == 0 {
            return Err(-1); // OPUS_BAD_ARG
        }

        let mode = &MODE_48000_960_120;
        let nb_ebands = mode.nb_ebands as usize;
        let overlap = mode.overlap;
        let buf_size = (DECODE_BUFFER_SIZE + overlap) as usize;

        let dec = CeltDecoder {
            mode,
            overlap,
            channels,
            stream_channels: channels,
            downsample: ds,
            start: 0,
            end: mode.eff_ebands,
            signalling: true,
            disable_inv: channels == 1,
            complexity: 5,

            rng: 0,
            error: 0,
            last_pitch_index: 0,
            loss_duration: 0,
            plc_duration: 0,
            last_frame_type: FRAME_NONE,
            skip_plc: 1,
            postfilter_period: 0,
            postfilter_period_old: 0,
            postfilter_gain: 0,
            postfilter_gain_old: 0,
            postfilter_tapset: 0,
            postfilter_tapset_old: 0,
            prefilter_and_fold: false,
            preemph_mem_d: [0; 2],

            decode_mem: vec![0i32; channels as usize * buf_size],
            old_band_e: vec![0i32; 2 * nb_ebands],
            old_log_e: vec![-gconst(28.0); 2 * nb_ebands],
            old_log_e2: vec![-gconst(28.0); 2 * nb_ebands],
            background_log_e: vec![0i32; 2 * nb_ebands],
            lpc_coef: vec![0i32; channels as usize * CELT_LPC_ORDER],

            #[cfg(feature = "dnn")]
            plc_pcm: vec![0i16; 560], // Worst case: (960+48+120)/3 + 160 = ~536
            #[cfg(feature = "dnn")]
            plc_fill: 0,
            #[cfg(feature = "dnn")]
            plc_preemphasis_mem: 0.0,
        };

        Ok(dec)
    }

    /// Reset all dynamic decoder state. Configuration is preserved.
    pub fn reset(&mut self) {
        self.rng = 0;
        self.error = 0;
        self.last_pitch_index = 0;
        self.loss_duration = 0;
        self.plc_duration = 0;
        self.last_frame_type = FRAME_NONE;
        self.skip_plc = 1;
        self.postfilter_period = 0;
        self.postfilter_period_old = 0;
        self.postfilter_gain = 0;
        self.postfilter_gain_old = 0;
        self.postfilter_tapset = 0;
        self.postfilter_tapset_old = 0;
        self.prefilter_and_fold = false;
        self.preemph_mem_d = [0; 2];
        self.decode_mem.fill(0);
        self.old_band_e.fill(0);
        self.old_log_e.fill(-gconst(28.0));
        self.old_log_e2.fill(-gconst(28.0));
        self.background_log_e.fill(0);
        self.lpc_coef.fill(0);
        #[cfg(feature = "dnn")]
        {
            self.plc_pcm.fill(0);
            self.plc_fill = 0;
            self.plc_preemphasis_mem = 0.0;
        }
    }

    /// Per-channel decode buffer size including overlap.
    #[inline]
    fn buf_size(&self) -> usize {
        (DECODE_BUFFER_SIZE + self.overlap) as usize
    }

    /// Offset into decode_mem where channel `c` starts.
    #[inline]
    fn ch_off(&self, c: usize) -> usize {
        c * self.buf_size()
    }

    /// Offset where out_syn starts for channel `c` given frame size `n`.
    #[inline]
    fn out_syn_off(&self, c: usize, n: i32) -> usize {
        self.ch_off(c) + (DECODE_BUFFER_SIZE - n) as usize
    }

    // -----------------------------------------------------------------------
    // PLC pitch search
    // -----------------------------------------------------------------------

    /// Search for the best pitch period for PLC.
    /// Matches C `celt_plc_pitch_search()`.
    fn plc_pitch_search(&self) -> i32 {
        let buf_size = DECODE_BUFFER_SIZE as usize;
        let lp_len = buf_size >> 1;
        let mut lp_pitch_buf = vec![0i32; lp_len];
        let cc = self.channels as usize;

        // Collect per-channel decode_mem references for pitch_downsample
        let mut ch_bufs: Vec<&[i32]> = Vec::with_capacity(cc);
        for c in 0..cc {
            let off = self.ch_off(c);
            ch_bufs.push(&self.decode_mem[off..off + buf_size]);
        }
        pitch_downsample(&ch_bufs, &mut lp_pitch_buf, lp_len, cc, 2);

        let search_offset = (PLC_PITCH_LAG_MAX >> 1) as usize;
        let search_len = buf_size - PLC_PITCH_LAG_MAX as usize;
        let pitch_index = pitch_search(
            &lp_pitch_buf[search_offset..],
            &lp_pitch_buf,
            search_len,
            (PLC_PITCH_LAG_MAX - PLC_PITCH_LAG_MIN) as usize,
        );
        PLC_PITCH_LAG_MAX - pitch_index
    }

    // -----------------------------------------------------------------------
    // Neural PLC helpers (behind feature flag)
    // -----------------------------------------------------------------------

    /// Sinc low-pass filter coefficients for 16kHz <-> 48kHz resampling.
    /// 49 taps (SINC_ORDER=48), matching C reference `celt_decoder.c:629`.
    #[cfg(feature = "dnn")]
    const SINC_FILTER: [f32; 49] = [
        4.2931e-05, -0.000190293, -0.000816132, -0.000637162, 0.00141662,
        0.00354764, 0.00184368, -0.00428274, -0.00856105, -0.0034003,
        0.00930201, 0.0159616, 0.00489785, -0.0169649, -0.0259484, -0.00596856,
        0.0286551, 0.0405872, 0.00649994, -0.0509284, -0.0716655, -0.00665212,
        0.134336, 0.278927, 0.339995, 0.278927, 0.134336, -0.00665212,
        -0.0716655, -0.0509284, 0.00649994, 0.0405872, 0.0286551, -0.00596856,
        -0.0259484, -0.0169649, 0.00489785, 0.0159616, 0.00930201, -0.0034003,
        -0.00856105, -0.00428274, 0.00184368, 0.00354764, 0.00141662,
        -0.000637162, -0.000816132, -0.000190293, 4.2931e-05,
    ];

    /// CELT preemphasis coefficient (0.85), matching C `dnn/freq.h`.
    #[cfg(feature = "dnn")]
    const PREEMPHASIS: f32 = 0.85;

    /// SINC_ORDER for the polyphase resampler.
    #[cfg(feature = "dnn")]
    const SINC_ORDER: usize = 48;

    /// LPCNet frame size at 16kHz (10ms).
    #[cfg(feature = "dnn")]
    const LPCNET_FRAME_SIZE: usize = 160;

    /// Number of 10ms frames used for PLC state update.
    #[cfg(feature = "dnn")]
    const PLC_UPDATE_FRAMES: usize = 4;

    /// Downsample 48kHz decode memory to 16kHz and feed to LPCNet.
    /// Matches C `update_plc_state()` in `celt_decoder.c:639`.
    #[cfg(feature = "dnn")]
    fn update_plc_state(&mut self, lpcnet: &mut crate::dnn::lpcnet::LPCNetPLCState) {
        let decode_buffer_size = DECODE_BUFFER_SIZE as usize;
        let cc = self.channels as usize;
        let plc_update_samples = Self::PLC_UPDATE_FRAMES * Self::LPCNET_FRAME_SIZE;

        // Mix to mono in f32
        let mut buf48k = vec![0.0f32; decode_buffer_size];
        if cc == 1 {
            let off = self.ch_off(0);
            for i in 0..decode_buffer_size {
                buf48k[i] = self.decode_mem[off + i] as f32;
            }
        } else {
            let off0 = self.ch_off(0);
            let off1 = self.ch_off(1);
            for i in 0..decode_buffer_size {
                buf48k[i] =
                    0.5 * (self.decode_mem[off0 + i] as f32 + self.decode_mem[off1 + i] as f32);
            }
        }

        // Apply preemphasis: buf48k[i] += PREEMPHASIS * buf48k[i-1]
        for i in 1..decode_buffer_size {
            buf48k[i] += Self::PREEMPHASIS * buf48k[i - 1];
        }
        self.plc_preemphasis_mem = buf48k[decode_buffer_size - 1];

        // Downsample 48kHz -> 16kHz using the sinc filter (factor 3)
        let offset =
            decode_buffer_size - Self::SINC_ORDER - 1 - 3 * (plc_update_samples - 1);
        let mut buf16k = vec![0i16; plc_update_samples];
        for i in 0..plc_update_samples {
            let mut sum = 0.0f32;
            for j in 0..Self::SINC_ORDER + 1 {
                sum += buf48k[3 * i + j + offset] * Self::SINC_FILTER[j];
            }
            buf16k[i] = sum.round().max(-32767.0).min(32767.0) as i16;
        }

        // Feed to LPCNet update (preserve fec state across update)
        let tmp_read_pos = lpcnet.fec_read_pos;
        let tmp_fec_skip = lpcnet.fec_skip;
        for i in 0..Self::PLC_UPDATE_FRAMES {
            let start = Self::LPCNET_FRAME_SIZE * i;
            let end = start + Self::LPCNET_FRAME_SIZE;
            lpcnet.update(&buf16k[start..end]);
        }
        lpcnet.fec_read_pos = tmp_read_pos;
        lpcnet.fec_skip = tmp_fec_skip;
    }

    // -----------------------------------------------------------------------
    // Packet loss concealment
    // -----------------------------------------------------------------------

    /// Handle a lost frame by generating concealment audio.
    /// Matches C `celt_decode_lost()`.
    fn decode_lost(&mut self, n: i32, lm: i32, mut lpcnet: DnnPlcArg<'_>) {
        let mode = self.mode;
        let cc = self.channels;
        let c_stream = cc; // In non-hybrid mode, C == CC
        let nb_ebands = mode.nb_ebands;
        let overlap = mode.overlap;
        let start = self.start;
        let loss_duration = self.loss_duration;
        let max_period = MAX_PERIOD;
        let _buf_size_per_ch = self.buf_size();

        // Determine PLC strategy
        let mut curr_frame_type = if self.plc_duration >= 40 || start != 0 || self.skip_plc != 0 {
            FRAME_PLC_NOISE
        } else {
            FRAME_PLC_PERIODIC
        };

        // Neural PLC override when DNN feature is enabled and model is loaded.
        #[cfg(feature = "dnn")]
        if let Some(ref lpcnet) = lpcnet {
            if start == 0 && lpcnet.loaded {
                if self.complexity >= 5 && self.plc_duration < 80 && self.skip_plc == 0 {
                    curr_frame_type = FRAME_PLC_NEURAL;
                }
                if lpcnet.fec_fill_pos > lpcnet.fec_read_pos {
                    curr_frame_type = FRAME_DRED;
                }
            }
        }
        #[cfg(not(feature = "dnn"))]
        let _ = &lpcnet;

        if curr_frame_type == FRAME_PLC_NOISE {
            // --- Noise-based PLC ---
            let end = self.end;
            let eff_end = imax(start, imin(end, mode.eff_ebands));
            let ebands = mode.ebands;

            let mut x = vec![0i32; cc as usize * n as usize];

            // Shift decode memory left by N samples
            for c in 0..cc as usize {
                let off = self.ch_off(c);
                let copy_len = (DECODE_BUFFER_SIZE - n + overlap) as usize;
                self.decode_mem
                    .copy_within(off + n as usize..off + n as usize + copy_len, off);
            }

            if self.prefilter_and_fold {
                let offsets: Vec<usize> = (0..cc as usize).map(|c| self.ch_off(c)).collect();
                prefilter_and_fold(
                    &mut self.decode_mem,
                    &offsets,
                    cc,
                    overlap,
                    n,
                    self.postfilter_period_old,
                    self.postfilter_period,
                    self.postfilter_gain_old,
                    self.postfilter_gain,
                    self.postfilter_tapset_old,
                    self.postfilter_tapset,
                    mode.window,
                );
            }

            // Energy decay
            let decay = if loss_duration == 0 {
                gconst(1.5)
            } else {
                gconst(0.5)
            };
            for c in 0..cc as usize {
                for i in start as usize..end as usize {
                    let idx = c * nb_ebands as usize + i;
                    self.old_band_e[idx] =
                        imax(self.background_log_e[idx], self.old_band_e[idx] - decay);
                }
            }

            // Generate random excitation
            let mut seed = self.rng;
            for c in 0..cc as usize {
                for i in start as usize..eff_end as usize {
                    let boffs = n as usize * c + ((ebands[i] as i32) << lm) as usize;
                    let blen = ((ebands[i + 1] - ebands[i]) as i32) << lm;
                    for j in 0..blen as usize {
                        seed = celt_lcg_rand(seed);
                        // SHL32((celt_norm)((opus_int32)seed>>20), NORM_SHIFT-14)
                        x[boffs + j] = shl32((seed as i32) >> 20, NORM_SHIFT - 14);
                    }
                    renormalise_vector(&mut x[boffs..boffs + blen as usize], blen as usize, Q31ONE);
                }
            }
            self.rng = seed;

            // Synthesis
            let out_syn_offsets: Vec<usize> =
                (0..cc as usize).map(|c| self.out_syn_off(c, n)).collect();
            celt_synthesis(
                mode,
                &mut x,
                &mut self.decode_mem,
                &out_syn_offsets,
                &self.old_band_e,
                start,
                eff_end,
                c_stream,
                cc,
                false,
                lm,
                self.downsample,
                false,
            );

            // Apply postfilter with last parameters
            for c in 0..cc as usize {
                let syn_off = self.out_syn_off(c, n);
                self.postfilter_period = imax(self.postfilter_period, COMBFILTER_MINPERIOD);
                self.postfilter_period_old = imax(self.postfilter_period_old, COMBFILTER_MINPERIOD);
                comb_filter(
                    &mut self.decode_mem,
                    syn_off,
                    self.postfilter_period_old,
                    self.postfilter_period,
                    mode.short_mdct_size,
                    self.postfilter_gain_old,
                    self.postfilter_gain,
                    self.postfilter_tapset_old,
                    self.postfilter_tapset,
                    mode.window,
                    overlap,
                );
                if lm != 0 {
                    let off2 = syn_off + mode.short_mdct_size as usize;
                    comb_filter(
                        &mut self.decode_mem,
                        off2,
                        self.postfilter_period,
                        self.postfilter_period,
                        n - mode.short_mdct_size,
                        self.postfilter_gain,
                        self.postfilter_gain,
                        self.postfilter_tapset,
                        self.postfilter_tapset,
                        mode.window,
                        overlap,
                    );
                }
            }
            self.postfilter_period_old = self.postfilter_period;
            self.postfilter_gain_old = self.postfilter_gain;
            self.postfilter_tapset_old = self.postfilter_tapset;
            self.prefilter_and_fold = false;
            self.skip_plc = 1;
        } else {
            // --- Pitch-based PLC (and neural PLC) ---
            let mut fade: i32 = Q15ONE;
            let pitch_index;

            // Neural PLC flags for frame type transitions
            #[cfg(feature = "dnn")]
            let curr_neural = curr_frame_type == FRAME_PLC_NEURAL
                || curr_frame_type == FRAME_DRED;
            #[cfg(feature = "dnn")]
            let last_neural = self.last_frame_type == FRAME_PLC_NEURAL
                || self.last_frame_type == FRAME_DRED;

            // Pitch search: skip if continuing periodic or continuing neural PLC
            #[cfg(feature = "dnn")]
            let skip_pitch_search = self.last_frame_type == FRAME_PLC_PERIODIC
                || (last_neural && curr_neural);
            #[cfg(not(feature = "dnn"))]
            let skip_pitch_search = self.last_frame_type == FRAME_PLC_PERIODIC;

            if !skip_pitch_search {
                pitch_index = self.plc_pitch_search();
                self.last_pitch_index = pitch_index;
            } else {
                pitch_index = self.last_pitch_index;
                fade = qconst16(0.8, 15);
            }

            // Initialize neural PLC state on transition to neural
            #[cfg(feature = "dnn")]
            if curr_neural && !last_neural {
                if let Some(ref mut lpcnet) = lpcnet {
                    self.update_plc_state(lpcnet);
                }
            }

            let exc_length = imin(2 * pitch_index, max_period);

            // Per-channel extrapolation
            for c in 0..cc as usize {
                let mut s1: i32 = 0;
                let buf_off = self.ch_off(c);
                let decode_buffer_size = DECODE_BUFFER_SIZE as usize;

                // Extract excitation samples from decode_mem
                let mut _exc = vec![0i32; (max_period + CELT_LPC_ORDER as i32) as usize];
                let exc_base = CELT_LPC_ORDER; // exc[0] corresponds to _exc[CELT_LPC_ORDER]
                for i in 0..(max_period + CELT_LPC_ORDER as i32) as usize {
                    let src_idx =
                        buf_off + decode_buffer_size - max_period as usize - CELT_LPC_ORDER + i;
                    _exc[i] = sround16(self.decode_mem[src_idx], SIG_SHIFT);
                }

                if self.last_frame_type != FRAME_PLC_PERIODIC {
                    // Compute LPC coefficients
                    let mut ac = vec![0i32; CELT_LPC_ORDER + 1];
                    let exc_ref = &_exc[exc_base..exc_base + max_period as usize];

                    // Build window reference for autocorrelation
                    let win_slice: Vec<i32> = mode.window.iter().map(|&w| w as i32).collect();

                    celt_autocorr(
                        exc_ref,
                        &mut ac,
                        Some(&win_slice),
                        overlap as usize,
                        CELT_LPC_ORDER,
                        max_period as usize,
                    );

                    // Add noise floor of -40 dB (fixed-point: ac[0] += ac[0] >> 13)
                    ac[0] += shr32(ac[0], 13);

                    // Lag windowing for stability
                    for i in 1..=CELT_LPC_ORDER {
                        ac[i] -= mult16_32_q15(2 * (i as i32) * (i as i32), ac[i]);
                    }

                    let lpc_off = c * CELT_LPC_ORDER;
                    celt_lpc(
                        &mut self.lpc_coef[lpc_off..lpc_off + CELT_LPC_ORDER],
                        &ac,
                        CELT_LPC_ORDER,
                    );

                    // Bandwidth expansion until sum(|coefs|) < 65535
                    loop {
                        let mut tmp = Q15ONE;
                        let mut sum: i32 = qconst16(1.0, SIG_SHIFT as u32);
                        for i in 0..CELT_LPC_ORDER {
                            sum += abs16(self.lpc_coef[lpc_off + i]);
                        }
                        if sum < 65535 {
                            break;
                        }
                        for i in 0..CELT_LPC_ORDER {
                            tmp = mult16_16_q15(qconst16(0.99, 15), tmp);
                            self.lpc_coef[lpc_off + i] =
                                mult16_16_q15(self.lpc_coef[lpc_off + i], tmp);
                        }
                    }
                }

                // Compute excitation via FIR filter
                // C calls: celt_fir(exc, lpc+1, exc, exc_length, LPC_ORDER, mem)
                // where exc = _exc + CELT_LPC_ORDER + MAX_PERIOD - exc_length.
                // Our celt_fir reads x[0..ord] as history and x[ord..ord+n] as input.
                // So we pass starting from (fir_start - ord) to include the LPC history.
                {
                    let lpc_off = c * CELT_LPC_ORDER;
                    let mut fir_tmp = vec![0i32; exc_length as usize];
                    let fir_start = exc_base + max_period as usize - exc_length as usize;
                    let fir_input_start = fir_start - CELT_LPC_ORDER;
                    celt_fir(
                        &_exc[fir_input_start..],
                        &self.lpc_coef[lpc_off..lpc_off + CELT_LPC_ORDER],
                        &mut fir_tmp,
                        exc_length as usize,
                        CELT_LPC_ORDER,
                    );
                    _exc[fir_start..fir_start + exc_length as usize].copy_from_slice(&fir_tmp);
                }

                // Compute decay factor from energy ratio of two half-periods
                let decay: i32;
                {
                    let mut e1: i32 = 1;
                    let mut e2: i32 = 1;
                    let _exc_slice = &_exc[exc_base..exc_base + max_period as usize];
                    let shift_val = imax(
                        0,
                        2 * celt_zlog2(celt_maxabs16(
                            &_exc[exc_base + max_period as usize - exc_length as usize
                                ..exc_base + max_period as usize],
                        )) - 20,
                    );
                    let decay_length = (exc_length >> 1) as usize;
                    for i in 0..decay_length {
                        let e = _exc[exc_base + max_period as usize - decay_length + i];
                        e1 += shr32(mult16_16(e, e), shift_val);
                        let e = _exc[exc_base + max_period as usize - 2 * decay_length + i];
                        e2 += shr32(mult16_16(e, e), shift_val);
                    }
                    e1 = min32(e1, e2);
                    decay = celt_sqrt(frac_div32(shr32(e1, 1), e2));
                }

                // Move decode memory left by N
                {
                    let copy_len = (DECODE_BUFFER_SIZE - n) as usize;
                    self.decode_mem.copy_within(
                        buf_off + n as usize..buf_off + n as usize + copy_len,
                        buf_off,
                    );
                }

                // Extrapolate excitation with pitch period and decay
                let extrapolation_offset = (max_period - pitch_index) as usize;
                let extrapolation_len = (n + overlap) as usize;
                let mut attenuation = mult16_16_q15(fade, decay);
                let mut j: usize = 0;

                for i in 0..extrapolation_len {
                    if j >= pitch_index as usize {
                        j -= pitch_index as usize;
                        attenuation = mult16_16_q15(attenuation, decay);
                    }
                    let exc_val = _exc[exc_base + extrapolation_offset + j];
                    let dest = buf_off + decode_buffer_size - n as usize + i;
                    self.decode_mem[dest] =
                        shl32(extend32(mult16_16_q15(attenuation, exc_val)), SIG_SHIFT);

                    // Accumulate energy of the source signal for comparison
                    let src = buf_off + decode_buffer_size - max_period as usize - n as usize
                        + extrapolation_offset
                        + j;
                    let tmp = sround16(self.decode_mem[src], SIG_SHIFT);
                    s1 += shr32(mult16_16(tmp, tmp), 11);

                    j += 1;
                }

                // Apply IIR synthesis filter
                {
                    let lpc_off = c * CELT_LPC_ORDER;
                    let mut lpc_mem = [0i32; CELT_LPC_ORDER];
                    for i in 0..CELT_LPC_ORDER {
                        let idx = buf_off + decode_buffer_size - n as usize - 1 - i;
                        lpc_mem[i] = sround16(self.decode_mem[idx], SIG_SHIFT);
                    }

                    let iir_start = buf_off + decode_buffer_size - n as usize;
                    // celt_iir needs separate input/output. Copy input first.
                    let iir_input: Vec<i32> =
                        self.decode_mem[iir_start..iir_start + extrapolation_len].to_vec();
                    celt_iir(
                        &iir_input,
                        &self.lpc_coef[lpc_off..lpc_off + CELT_LPC_ORDER],
                        &mut self.decode_mem[iir_start..iir_start + extrapolation_len],
                        extrapolation_len,
                        CELT_LPC_ORDER,
                        &mut lpc_mem,
                    );

                    // Saturate
                    for i in 0..extrapolation_len {
                        self.decode_mem[iir_start + i] =
                            saturate(self.decode_mem[iir_start + i], SIG_SAT);
                    }
                }

                // Energy guard: check if synthesis exploded
                {
                    let iir_start = buf_off + decode_buffer_size - n as usize;
                    let mut s2: i32 = 0;
                    for i in 0..extrapolation_len {
                        let tmp = sround16(self.decode_mem[iir_start + i], SIG_SHIFT);
                        s2 += shr32(mult16_16(tmp, tmp), 11);
                    }

                    // Fixed-point check: !(S1 > S2 >> 2)
                    if !(s1 > shr32(s2, 2)) {
                        // Explosion: zero out
                        for i in 0..extrapolation_len {
                            self.decode_mem[iir_start + i] = 0;
                        }
                    } else if s1 < s2 {
                        // Attenuate smoothly
                        let ratio = celt_sqrt(frac_div32(shr32(s1, 1) + 1, s2 + 1));
                        for i in 0..overlap as usize {
                            let w = mode.window[i] as i32;
                            let tmp_g = Q15ONE - mult16_16_q15(w, Q15ONE - ratio);
                            self.decode_mem[iir_start + i] =
                                mult16_32_q15(tmp_g, self.decode_mem[iir_start + i]);
                        }
                        for i in overlap as usize..extrapolation_len {
                            self.decode_mem[iir_start + i] =
                                mult16_32_q15(ratio, self.decode_mem[iir_start + i]);
                        }
                    }
                }
            } // end per-channel loop

            // Neural PLC: synthesize at 16kHz, upsample to 48kHz, cross-fade.
            // Uses curr_neural/last_neural computed at the top of this branch.
            #[cfg(feature = "dnn")]
            {
                if curr_neural {
                    if let Some(ref mut lpcnet) = lpcnet {
                        let decode_buffer_size = DECODE_BUFFER_SIZE as usize;
                        let overlap_usize = overlap as usize;

                        // Save overlap region from classical PLC for cross-fade
                        let mut buf_copy =
                            vec![vec![0.0f32; overlap_usize]; cc as usize];
                        for c in 0..cc as usize {
                            let off = self.ch_off(c) + decode_buffer_size - n as usize;
                            for i in 0..overlap_usize {
                                buf_copy[c][i] = self.decode_mem[off + i] as f32;
                            }
                        }

                        // Fill plc_pcm with enough 16kHz samples.
                        // Need (N + SINC_ORDER + overlap) / 3 samples at 16kHz.
                        let samples_needed_16k =
                            ((n + Self::SINC_ORDER as i32 + overlap) / 3) as usize;
                        if !last_neural {
                            self.plc_fill = 0;
                        }
                        while self.plc_fill < samples_needed_16k {
                            // Grow plc_pcm if needed
                            if self.plc_fill + Self::LPCNET_FRAME_SIZE > self.plc_pcm.len() {
                                self.plc_pcm
                                    .resize(self.plc_fill + Self::LPCNET_FRAME_SIZE, 0);
                            }
                            lpcnet
                                .conceal(&mut self.plc_pcm[self.plc_fill..self.plc_fill + Self::LPCNET_FRAME_SIZE]);
                            self.plc_fill += Self::LPCNET_FRAME_SIZE;
                        }

                        // Upsample 16kHz -> 48kHz using polyphase sinc filter.
                        // Writes to channel 0's decode_mem.
                        let buf_off = self.ch_off(0);
                        let resamp_len = ((n + overlap) / 3) as usize;
                        for i in 0..resamp_len {
                            let mut sum0 = 0.0f32;
                            for j in 0..17 {
                                sum0 += 3.0
                                    * self.plc_pcm[i + j] as f32
                                    * Self::SINC_FILTER[3 * j];
                            }
                            self.decode_mem
                                [buf_off + decode_buffer_size - n as usize + 3 * i] =
                                sum0.round() as i32;

                            let mut sum1 = 0.0f32;
                            for j in 0..16 {
                                sum1 += 3.0
                                    * self.plc_pcm[i + j + 1] as f32
                                    * Self::SINC_FILTER[3 * j + 2];
                            }
                            self.decode_mem
                                [buf_off + decode_buffer_size - n as usize + 3 * i + 1] =
                                sum1.round() as i32;

                            let mut sum2 = 0.0f32;
                            for j in 0..16 {
                                sum2 += 3.0
                                    * self.plc_pcm[i + j + 1] as f32
                                    * Self::SINC_FILTER[3 * j + 1];
                            }
                            self.decode_mem
                                [buf_off + decode_buffer_size - n as usize + 3 * i + 2] =
                                sum2.round() as i32;
                        }

                        // Shift plc_pcm: consumed N/3 samples
                        let consumed = (n / 3) as usize;
                        self.plc_pcm.copy_within(consumed..self.plc_fill, 0);
                        self.plc_fill -= consumed;

                        // Remove preemphasis from the N samples
                        let buf_start = buf_off + decode_buffer_size - n as usize;
                        for i in 0..n as usize {
                            let tmp = self.decode_mem[buf_start + i] as f32;
                            self.decode_mem[buf_start + i] = (tmp
                                - Self::PREEMPHASIS * self.plc_preemphasis_mem)
                                .round() as i32;
                            self.plc_preemphasis_mem = tmp;
                        }

                        // Remove preemphasis from the overlap region
                        let mut overlap_mem = self.plc_preemphasis_mem;
                        let overlap_start = buf_off + decode_buffer_size;
                        for i in 0..overlap_usize {
                            let tmp = self.decode_mem[overlap_start + i] as f32;
                            self.decode_mem[overlap_start + i] =
                                (tmp - Self::PREEMPHASIS * overlap_mem).round() as i32;
                            overlap_mem = tmp;
                        }

                        // Mono neural PLC: copy ch0 to ch1 for stereo
                        if cc == 2 {
                            let ch0_off = self.ch_off(0);
                            let ch1_off = self.ch_off(1);
                            let copy_len = decode_buffer_size + overlap_usize;
                            debug_assert!(ch1_off >= ch0_off + copy_len);
                            self.decode_mem
                                .copy_within(ch0_off..ch0_off + copy_len, ch1_off);
                        }

                        // Cross-fade: blend classical PLC overlap with neural output
                        // on the first neural frame.
                        if !last_neural {
                            for c in 0..cc as usize {
                                let off =
                                    self.ch_off(c) + decode_buffer_size - n as usize;
                                for i in 0..overlap_usize {
                                    let w = mode.window[i] as f32 / 32768.0;
                                    let classical = buf_copy[c][i];
                                    let neural = self.decode_mem[off + i] as f32;
                                    self.decode_mem[off + i] =
                                        ((1.0 - w) * classical + w * neural).round()
                                            as i32;
                                }
                            }
                        }
                    }
                }
            }

            self.prefilter_and_fold = true;
        }

        // Update loss counters
        let big_m = 1 << lm;
        self.loss_duration = imin(10000, self.loss_duration + big_m);
        self.plc_duration = imin(10000, self.plc_duration + big_m);
        // DRED resets plc_duration and skip_plc (it provides real FEC data)
        #[cfg(feature = "dnn")]
        if curr_frame_type == FRAME_DRED {
            self.plc_duration = 0;
            self.skip_plc = 0;
        }
        self.last_frame_type = curr_frame_type;
    }

    // -----------------------------------------------------------------------
    // Main decode function
    // -----------------------------------------------------------------------

    /// Decode one CELT frame from the given bitstream data.
    ///
    /// - `data`: compressed bytes, or `None` to trigger PLC
    /// - `pcm`: output PCM buffer (interleaved for stereo)
    /// - `frame_size`: samples per channel at the OUTPUT rate
    /// - `dec`: optional pre-initialized range decoder
    /// - `accum`: if true, add to pcm instead of overwriting (hybrid mode)
    ///
    /// Returns the number of decoded samples per channel, or a negative error code.
    pub fn decode_with_ec<'a>(
        &mut self,
        data: Option<&'a [u8]>,
        pcm: &mut [i16],
        frame_size: i32,
        dec: Option<&mut RangeDecoder<'a>>,
        accum: bool,
        lpcnet: DnnPlcArg<'_>,
    ) -> Result<i32, i32> {
        let mode = self.mode;
        let nb_ebands = mode.nb_ebands;
        let overlap = mode.overlap;
        let ebands = mode.ebands;
        let start = self.start;
        let end = self.end;
        let cc = self.channels;
        let c_stream = self.stream_channels;

        // Scale frame_size by downsample (internal decode always at native rate)
        let frame_size_native = frame_size * self.downsample;

        // Determine LM from frame_size_native
        let mut lm = -1i32;
        for i in 0..=mode.max_lm {
            if mode.short_mdct_size << i == frame_size_native {
                lm = i;
                break;
            }
        }
        if lm < 0 {
            return Err(-1); // OPUS_BAD_ARG
        }
        let big_m = 1 << lm;
        let n = big_m * mode.short_mdct_size;

        // Prepare per-channel output offsets (max 2 channels)
        let out_syn_offsets: [usize; 2] = [
            self.out_syn_off(0, n),
            if cc >= 2 { self.out_syn_off(1, n) } else { 0 },
        ];

        // --- Packet loss detection ---
        let data_len = data.map_or(0, |d| d.len());
        if data.is_none() || data_len <= 1 {
            self.decode_lost(n, lm, lpcnet);

            // Apply de-emphasis and return
            let ds = self.downsample;
            let off0 = self.out_syn_off(0, n);
            let off1 = if cc >= 2 { self.out_syn_off(1, n) } else { 0 };
            let inp0 = &self.decode_mem[off0..off0 + n as usize];
            let inp1 = &self.decode_mem[off1..off1 + n as usize];
            let inp: [&[i32]; 2] = [inp0, inp1];
            deemphasis(
                &inp[..cc as usize],
                pcm,
                n,
                cc,
                ds,
                &mode.preemph,
                &mut self.preemph_mem_d,
                accum,
            );
            return Ok(frame_size);
        }

        // Normal decode path: lpcnet is not used (only for PLC).
        let _ = lpcnet;

        let data = data.unwrap();

        // Initialize range decoder
        let mut owned_dec;
        let dec = if let Some(d) = dec {
            d
        } else {
            owned_dec = RangeDecoder::new(data);
            &mut owned_dec
        };

        let mut total_bits = (data_len as i32) * 8;
        let mut tell = dec.tell();

        // --- For mono, ensure consistency across channel energy slots ---
        if c_stream == 1 {
            for i in 0..nb_ebands as usize {
                self.old_band_e[i] =
                    imax(self.old_band_e[i], self.old_band_e[nb_ebands as usize + i]);
            }
        }

        // --- Silence detection ---
        let mut silence = false;
        if tell >= total_bits {
            silence = true;
        } else if tell == 1 {
            silence = dec.decode_bit_logp(15);
        }
        if silence {
            // Pretend all bits consumed (matches C: dec->nbits_total += tell - ec_tell(dec))
            tell = total_bits;
            dec.add_nbits_total(tell - dec.tell());
        }

        // --- Postfilter parameters ---
        let mut postfilter_pitch = 0i32;
        let mut postfilter_gain = 0i32;
        let mut postfilter_tapset = 0i32;

        if start == 0 && tell + 16 <= total_bits {
            let has_postfilter = dec.decode_bit_logp(1);
            if has_postfilter {
                let octave = dec.decode_uint(6);
                postfilter_pitch = (16 << octave) as i32 + dec.decode_bits(4 + octave) as i32 - 1;
                let qg = dec.decode_bits(3) as i32;
                postfilter_gain = qconst16(0.09375, 15) * (qg + 1);
                tell = dec.tell();
                if tell + 2 <= total_bits {
                    postfilter_tapset = dec.decode_icdf(&TAPSET_ICDF, 2);
                }
            }
            tell = dec.tell();
        }

        // --- Transient detection ---
        let mut is_transient = false;
        let mut short_blocks = 0i32;
        if lm > 0 && tell + 3 <= total_bits {
            is_transient = dec.decode_bit_logp(3);
            tell = dec.tell();
        }
        if is_transient {
            short_blocks = big_m;
        }

        // --- Energy decoding ---
        let intra_ener = if tell + 3 <= total_bits {
            if dec.decode_bit_logp(3) { 1i32 } else { 0 }
        } else {
            0
        };

        // Loss recovery energy clamping
        if intra_ener == 0 && self.loss_duration != 0 {
            for c in 0..2i32 {
                let missing = imin(10, self.loss_duration >> lm);
                let safety = if lm == 0 {
                    gconst(1.5)
                } else if lm == 1 {
                    gconst(0.5)
                } else {
                    0
                };
                for i in start as usize..end as usize {
                    let idx = c as usize * nb_ebands as usize + i;
                    let be = self.old_band_e[idx];
                    let le = self.old_log_e[idx];
                    let le2 = self.old_log_e2[idx];
                    if be < imax(le, le2) {
                        // Energy going down — continue the trend
                        let e0 = be;
                        let e1 = le;
                        let e2 = le2;
                        let slope = max32(e1 - e0, half32(e2 - e0));
                        let slope = imin(slope, gconst(2.0));
                        let e0 = e0 - max32(0, (1 + missing) * slope);
                        self.old_band_e[idx] = max32(-gconst(20.0), e0);
                    } else {
                        // Take min of last frames
                        self.old_band_e[idx] = imin(imin(be, le), le2);
                    }
                    self.old_band_e[idx] -= safety;
                }
            }
        }

        unquant_coarse_energy(
            mode,
            start,
            end,
            &mut self.old_band_e,
            intra_ener,
            dec,
            c_stream,
            lm,
        );

        // --- TF resolution ---
        let mut tf_res = [0i32; NB_EBANDS];
        tf_decode(start, end, is_transient, &mut tf_res, lm, dec);

        tell = dec.tell();

        // --- Spread decision ---
        let spread_decision = if tell + 4 <= total_bits {
            dec.decode_icdf(&SPREAD_ICDF, 5)
        } else {
            2 // SPREAD_NORMAL
        };

        // --- Dynamic bit allocation ---
        let mut cap = [0i32; NB_EBANDS];
        init_caps(mode, &mut cap, lm, c_stream);

        let mut offsets = [0i32; NB_EBANDS];
        let mut dynalloc_logp = 6i32;
        total_bits <<= BITRES;
        tell = dec.tell_frac() as i32;

        for i in start as usize..end as usize {
            let width = c_stream * ((ebands[i + 1] - ebands[i]) as i32) << lm;
            let quanta = imin(width << BITRES, imax(6 << BITRES, width));
            let mut dynalloc_loop_logp = dynalloc_logp;
            let mut boost = 0i32;
            while tell + (dynalloc_loop_logp << BITRES) < total_bits && boost < cap[i] {
                let flag = dec.decode_bit_logp(dynalloc_loop_logp as u32);
                tell = dec.tell_frac() as i32;
                if !flag {
                    break;
                }
                boost += quanta;
                total_bits -= quanta;
                dynalloc_loop_logp = 1;
            }
            offsets[i] = boost;
            // After first band gets a boost, tighten the probability
            if boost > 0 {
                dynalloc_logp = imax(2, dynalloc_logp - 1);
            }
        }

        // Alloc trim
        let alloc_trim = if tell + (6 << BITRES) <= total_bits {
            dec.decode_icdf(&TRIM_ICDF, 7)
        } else {
            5
        };

        // Anti-collapse reservation
        let tell_frac_here = dec.tell_frac() as i32;
        let bits_pre_acr = ((data_len as i32) << 3 << BITRES) - tell_frac_here - 1;
        let anti_collapse_rsv = if is_transient && lm >= 2 && bits_pre_acr >= (lm + 2) << BITRES {
            1 << BITRES
        } else {
            0
        };
        let bits = bits_pre_acr - anti_collapse_rsv;

        // Compute allocation
        let eff_end = imax(start, imin(end, mode.eff_ebands));
        let mut pulses = [0i32; NB_EBANDS];
        let mut fine_quant = [0i32; NB_EBANDS];
        let mut fine_priority = [0i32; NB_EBANDS];
        let mut intensity = 0i32;
        let mut dual_stereo = 0i32;
        let mut balance = 0i32;

        let coded_bands = clt_compute_allocation(
            mode,
            start,
            end,
            &offsets,
            &cap,
            alloc_trim,
            &mut intensity,
            &mut dual_stereo,
            bits,
            &mut balance,
            &mut pulses,
            &mut fine_quant,
            &mut fine_priority,
            c_stream,
            lm,
            dec,
            false, // encode=false
            0,     // prev (not used in decode)
            -1,    // signal_bandwidth (not used in decode)
        );

        // --- Fine energy + coefficient decoding ---
        unquant_fine_energy(
            mode,
            start,
            end,
            &mut self.old_band_e,
            None,
            &fine_quant,
            dec,
            c_stream,
        );

        // Shift decode memory left by N
        for c in 0..cc as usize {
            let off = self.ch_off(c);
            let copy_len = (DECODE_BUFFER_SIZE - n + overlap) as usize;
            self.decode_mem
                .copy_within(off + n as usize..off + n as usize + copy_len, off);
        }

        // Decode PVQ coefficients — stack arrays (max 2*960=1920 for x, 2*21=42 for masks)
        const MAX_X: usize = 2 * 960;
        const MAX_COLLAPSE: usize = 2 * NB_EBANDS;
        let x_len = c_stream as usize * n as usize;
        let cm_len = c_stream as usize * nb_ebands as usize;
        let mut x = [0i32; MAX_X];
        let mut collapse_masks = [0u8; MAX_COLLAPSE];
        let total_bits_alloc = (data_len as i32) * (8 << BITRES) - anti_collapse_rsv;

        // For stereo, quant_all_bands needs X and Y as separate mutable slices.
        // We use split_at_mut to avoid double-borrow, and pass the parts directly.
        if c_stream == 2 {
            let (x_part, y_part) = x[..x_len].split_at_mut(n as usize);
            quant_all_bands(
                false, // decode
                mode,
                start,
                end,
                x_part,
                Some(y_part),
                &mut collapse_masks[..cm_len],
                &self.old_band_e,
                &mut pulses,
                short_blocks != 0,
                spread_decision,
                dual_stereo != 0,
                intensity,
                &tf_res,
                total_bits_alloc,
                balance,
                dec,
                lm,
                coded_bands,
                &mut self.rng,
                0, // complexity
                self.disable_inv,
            );
        } else {
            quant_all_bands(
                false, // decode
                mode,
                start,
                end,
                &mut x[..x_len],
                None,
                &mut collapse_masks[..cm_len],
                &self.old_band_e,
                &mut pulses,
                short_blocks != 0,
                spread_decision,
                dual_stereo != 0,
                intensity,
                &tf_res,
                total_bits_alloc,
                balance,
                dec,
                lm,
                coded_bands,
                &mut self.rng,
                0, // complexity
                self.disable_inv,
            );
        }

        // Anti-collapse bit: read as raw bit from end (ec_dec_bits), NOT range coder!
        let mut anti_collapse_on = 0i32;
        if anti_collapse_rsv > 0 {
            anti_collapse_on = dec.decode_bits(1) as i32;
        }

        // Finalize energy (MUST come before anti_collapse, matching C order)
        unquant_energy_finalise(
            mode,
            start,
            end,
            Some(&mut self.old_band_e),
            &fine_quant,
            &fine_priority,
            (data_len as i32) * 8 - dec.tell(),
            dec,
            c_stream,
        );

        // Anti-collapse (after finalise, using updated old_band_e)
        if anti_collapse_on != 0 {
            anti_collapse(
                mode,
                &mut x[..x_len],
                &mut collapse_masks[..cm_len],
                lm,
                c_stream,
                n,
                start,
                end,
                &self.old_band_e,
                &self.old_log_e,
                &self.old_log_e2,
                &pulses,
                self.rng,
                false,
            );
        }

        // Silence: set all energies to -28 dB
        if silence {
            for i in 0..(2 * nb_ebands) as usize {
                self.old_band_e[i] = -gconst(28.0);
            }
        }

        // Prefilter and fold if needed
        if self.prefilter_and_fold {
            let pf_offsets: [usize; 2] = [
                self.ch_off(0),
                if cc >= 2 { self.ch_off(1) } else { 0 },
            ];
            prefilter_and_fold(
                &mut self.decode_mem,
                &pf_offsets[..cc as usize],
                cc,
                overlap,
                n,
                self.postfilter_period_old,
                self.postfilter_period,
                self.postfilter_gain_old,
                self.postfilter_gain,
                self.postfilter_tapset_old,
                self.postfilter_tapset,
                mode.window,
            );
        }

        // --- Synthesis ---
        celt_synthesis(
            mode,
            &mut x[..x_len],
            &mut self.decode_mem,
            &out_syn_offsets,
            &self.old_band_e,
            start,
            eff_end,
            c_stream,
            cc,
            is_transient,
            lm,
            self.downsample,
            silence,
        );

        // --- Postfilter application ---
        for c in 0..cc as usize {
            let syn_off = self.out_syn_off(c, n);
            self.postfilter_period = imax(self.postfilter_period, COMBFILTER_MINPERIOD);
            self.postfilter_period_old = imax(self.postfilter_period_old, COMBFILTER_MINPERIOD);
            comb_filter(
                &mut self.decode_mem,
                syn_off,
                self.postfilter_period_old,
                self.postfilter_period,
                mode.short_mdct_size,
                self.postfilter_gain_old,
                self.postfilter_gain,
                self.postfilter_tapset_old,
                self.postfilter_tapset,
                mode.window,
                overlap,
            );
            if lm != 0 {
                let off2 = syn_off + mode.short_mdct_size as usize;
                comb_filter(
                    &mut self.decode_mem,
                    off2,
                    self.postfilter_period,
                    postfilter_pitch,
                    n - mode.short_mdct_size,
                    self.postfilter_gain,
                    postfilter_gain,
                    self.postfilter_tapset,
                    postfilter_tapset,
                    mode.window,
                    overlap,
                );
            }
        }

        // --- State update ---
        self.postfilter_period_old = self.postfilter_period;
        self.postfilter_gain_old = self.postfilter_gain;
        self.postfilter_tapset_old = self.postfilter_tapset;
        self.postfilter_period = postfilter_pitch;
        self.postfilter_gain = postfilter_gain;
        self.postfilter_tapset = postfilter_tapset;
        if lm != 0 {
            self.postfilter_period_old = self.postfilter_period;
            self.postfilter_gain_old = self.postfilter_gain;
            self.postfilter_tapset_old = self.postfilter_tapset;
        }

        // Copy energies for mono stream
        if c_stream == 1 {
            let nb = nb_ebands as usize;
            self.old_band_e.copy_within(..nb, nb);
        }

        // Update energy history
        if !is_transient {
            self.old_log_e2.copy_from_slice(&self.old_log_e);
            self.old_log_e.copy_from_slice(&self.old_band_e);
        } else {
            for i in 0..(2 * nb_ebands) as usize {
                self.old_log_e[i] = imin(self.old_log_e[i], self.old_band_e[i]);
            }
        }

        // Background noise estimate: allow max +2.4 dB/second increase
        let max_bg_increase = imin(160, self.loss_duration + big_m) as i64 * gconst(0.001) as i64;
        let max_bg_increase = max_bg_increase as i32;
        for i in 0..(2 * nb_ebands) as usize {
            self.background_log_e[i] = imin(
                self.background_log_e[i] + max_bg_increase,
                self.old_band_e[i],
            );
        }

        // Zero out-of-band energies
        for c in 0..2usize {
            for i in 0..start as usize {
                let idx = c * nb_ebands as usize + i;
                self.old_band_e[idx] = 0;
                self.old_log_e[idx] = -gconst(28.0);
                self.old_log_e2[idx] = -gconst(28.0);
            }
            for i in end as usize..nb_ebands as usize {
                let idx = c * nb_ebands as usize + i;
                self.old_band_e[idx] = 0;
                self.old_log_e[idx] = -gconst(28.0);
                self.old_log_e2[idx] = -gconst(28.0);
            }
        }

        // Store range coder state
        self.rng = dec.get_rng();

        // --- De-emphasis ---
        let ds = self.downsample;
        let de0 = self.out_syn_off(0, n);
        let de1 = if cc >= 2 { self.out_syn_off(1, n) } else { 0 };
        let di0 = &self.decode_mem[de0..de0 + n as usize];
        let di1 = &self.decode_mem[de1..de1 + n as usize];
        let de_inp: [&[i32]; 2] = [di0, di1];
        deemphasis(
            &de_inp[..cc as usize],
            pcm,
            n,
            cc,
            ds,
            &mode.preemph,
            &mut self.preemph_mem_d,
            accum,
        );

        // Reset PLC state
        self.loss_duration = 0;
        self.plc_duration = 0;
        self.last_frame_type = FRAME_NORMAL;
        self.prefilter_and_fold = false;
        self.skip_plc = 0;

        // Verify bitstream consistency
        if dec.tell() > (data_len as i32) * 8 {
            self.error = 1;
        }

        Ok(frame_size)
    }

    // -----------------------------------------------------------------------
    // CTL (control) interface
    // -----------------------------------------------------------------------

    /// Set decoder complexity (0–10). Affects PLC quality.
    pub fn set_complexity(&mut self, complexity: i32) -> Result<(), i32> {
        if complexity < 0 || complexity > 10 {
            return Err(-1);
        }
        self.complexity = complexity;
        Ok(())
    }

    /// Get current complexity setting.
    pub fn get_complexity(&self) -> i32 {
        self.complexity
    }

    /// Set the first band to decode (0 for full, 17 for hybrid high-band).
    pub fn set_start_band(&mut self, band: i32) -> Result<(), i32> {
        if band < 0 || band > self.mode.nb_ebands {
            return Err(-1);
        }
        self.start = band;
        Ok(())
    }

    /// Set the last band (exclusive) to decode.
    pub fn set_end_band(&mut self, band: i32) -> Result<(), i32> {
        if band < 1 || band > self.mode.nb_ebands {
            return Err(-1);
        }
        self.end = band;
        Ok(())
    }

    /// Set the number of channels encoded in the stream.
    pub fn set_channels(&mut self, channels: i32) -> Result<(), i32> {
        if channels < 1 || channels > 2 {
            return Err(-1);
        }
        self.stream_channels = channels;
        Ok(())
    }

    /// Get and clear the sticky error flag.
    pub fn get_and_clear_error(&mut self) -> i32 {
        let err = self.error;
        self.error = 0;
        err
    }

    /// Get the decoder lookahead in output samples.
    pub fn get_lookahead(&self) -> i32 {
        self.overlap / self.downsample
    }

    /// Get the last detected pitch period.
    pub fn get_pitch(&self) -> i32 {
        self.postfilter_period
    }

    /// Get a reference to the mode configuration.
    pub fn get_mode(&self) -> &'static CELTMode {
        self.mode
    }

    /// Set the signalling flag (for custom modes).
    pub fn set_signalling(&mut self, signalling: bool) {
        self.signalling = signalling;
    }

    /// Set phase inversion disabled flag.
    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) {
        self.disable_inv = disabled;
    }

    /// Get phase inversion disabled flag.
    pub fn get_phase_inversion_disabled(&self) -> bool {
        self.disable_inv
    }

    /// Get old_band_e (debug accessor).
    pub fn debug_old_band_e(&self) -> &[i32] {
        &self.old_band_e
    }

    /// Get old_log_e (debug accessor).
    pub fn debug_old_log_e(&self) -> &[i32] {
        &self.old_log_e
    }

    /// Get old_log_e2 (debug accessor).
    pub fn debug_old_log_e2(&self) -> &[i32] {
        &self.old_log_e2
    }

    /// Get a slice of decode_mem (debug accessor).
    pub fn debug_get_decode_mem(&self, offset: usize, count: usize) -> &[i32] {
        &self.decode_mem[offset..offset + count]
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_new_mono() {
        let dec = CeltDecoder::new(48000, 1).unwrap();
        assert_eq!(dec.channels, 1);
        assert_eq!(dec.downsample, 1);
        assert_eq!(dec.overlap, 120);
        assert_eq!(dec.start, 0);
        assert_eq!(dec.end, 21);
        assert_eq!(dec.old_log_e.len(), 42);
    }

    #[test]
    fn decoder_new_stereo() {
        let dec = CeltDecoder::new(48000, 2).unwrap();
        assert_eq!(dec.channels, 2);
        assert_eq!(dec.decode_mem.len(), 2 * (2048 + 120));
    }

    #[test]
    fn decoder_new_16khz() {
        let dec = CeltDecoder::new(16000, 1).unwrap();
        assert_eq!(dec.downsample, 3);
    }

    #[test]
    fn decoder_new_8khz() {
        let dec = CeltDecoder::new(8000, 1).unwrap();
        assert_eq!(dec.downsample, 6);
    }

    #[test]
    fn decoder_new_invalid_rate() {
        // Invalid rate: resampling_factor returns 0, new() returns Err.
        assert!(CeltDecoder::new(44100, 1).is_err());
    }

    #[test]
    fn decoder_new_invalid_channels() {
        assert!(CeltDecoder::new(48000, 0).is_err());
        assert!(CeltDecoder::new(48000, 3).is_err());
    }

    #[test]
    fn decoder_reset() {
        let mut dec = CeltDecoder::new(48000, 1).unwrap();
        dec.rng = 12345;
        dec.loss_duration = 100;
        dec.last_frame_type = FRAME_NORMAL;
        dec.reset();
        assert_eq!(dec.rng, 0);
        assert_eq!(dec.loss_duration, 0);
        assert_eq!(dec.last_frame_type, FRAME_NONE);
        // Energies should be reset to -GCONST(28)
        for &e in &dec.old_log_e {
            assert_eq!(e, -gconst(28.0));
        }
    }

    #[test]
    fn ctl_complexity() {
        let mut dec = CeltDecoder::new(48000, 1).unwrap();
        assert!(dec.set_complexity(5).is_ok());
        assert_eq!(dec.get_complexity(), 5);
        assert!(dec.set_complexity(-1).is_err());
        assert!(dec.set_complexity(11).is_err());
    }

    #[test]
    fn ctl_bands() {
        let mut dec = CeltDecoder::new(48000, 1).unwrap();
        assert!(dec.set_start_band(0).is_ok());
        assert!(dec.set_start_band(17).is_ok());
        assert!(dec.set_end_band(21).is_ok());
        assert!(dec.set_end_band(0).is_err());
    }

    #[test]
    fn ctl_lookahead() {
        let dec = CeltDecoder::new(48000, 1).unwrap();
        assert_eq!(dec.get_lookahead(), 120); // overlap / downsample = 120/1
        let dec = CeltDecoder::new(16000, 1).unwrap();
        assert_eq!(dec.get_lookahead(), 40); // 120/3 = 40
    }

    #[test]
    fn plc_null_data_mono() {
        // Decode with NULL data should trigger PLC and produce silence-ish output
        let mut dec = CeltDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0i16; 960];
        #[cfg(feature = "dnn")]
        let lpcnet_arg: DnnPlcArg<'_> = None;
        #[cfg(not(feature = "dnn"))]
        let lpcnet_arg: DnnPlcArg<'_> = ();
        let result = dec.decode_with_ec(None, &mut pcm, 960, None, false, lpcnet_arg);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 960);
    }

    #[test]
    fn gconst_values() {
        // Verify key GCONST values match expected Q24 format
        assert_eq!(gconst(28.0), qconst32(28.0, DB_SHIFT as u32));
        assert_eq!(gconst(1.5), qconst32(1.5, DB_SHIFT as u32));
        assert_eq!(gconst(0.5), qconst32(0.5, DB_SHIFT as u32));
    }

    #[test]
    fn tf_select_table_shape() {
        assert_eq!(TF_SELECT_TABLE.len(), 4);
        for row in &TF_SELECT_TABLE {
            assert_eq!(row.len(), 8);
        }
    }
}
