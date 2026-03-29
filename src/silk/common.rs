//! SILK common types and utilities.
//!
//! Ported from: SigProc_FIX.h, sort.c, bwexpander.c, bwexpander_32.c,
//! lin2log.c, log2lin.c, LPC_analysis_filter_FIX.c, LPC_inv_pred_gain.c,
//! LPC_fit.c, sum_sqr_shift.c, inner_prod_aligned.c, sigm_Q15.c

use crate::types::*;
use crate::silk::tables::*;

// ===========================================================================
// Constants from define.h / SigProc_FIX.h
// ===========================================================================

pub const MAX_NB_SUBFR: usize = 4;
pub const MAX_FRAME_LENGTH: usize = 320;
pub const MAX_SUB_FRAME_LENGTH: usize = 80;
pub const SUB_FRAME_LENGTH_MS: usize = 5;
pub const MAX_FRAME_LENGTH_MS: usize = 20;
pub const LTP_MEM_LENGTH_MS: usize = 20;
pub const DECODER_NUM_CHANNELS: usize = 2;
pub const MAX_FRAMES_PER_PACKET: usize = 3;
pub const MAX_API_FS_KHZ: usize = 48;

pub const SHELL_CODEC_FRAME_LENGTH: usize = 16;
pub const LOG2_SHELL_CODEC_FRAME_LENGTH: usize = 4;
pub const MAX_NB_SHELL_BLOCKS: usize = 20; // MAX_FRAME_LENGTH / SHELL_CODEC_FRAME_LENGTH

pub const SILK_MAX_PULSES: i32 = 16;
pub const N_RATE_LEVELS: usize = 10;

pub const NLSF_QUANT_MAX_AMPLITUDE: i32 = 4;
pub const NLSF_QUANT_LEVEL_ADJ: f64 = 0.1;
pub const MAX_LPC_STABILIZE_ITERATIONS: usize = 16;

pub const TYPE_NO_VOICE_ACTIVITY: i32 = 0;
pub const TYPE_UNVOICED: i32 = 1;
pub const TYPE_VOICED: i32 = 2;

pub const CODE_INDEPENDENTLY: i32 = 0;
pub const CODE_CONDITIONALLY: i32 = 1;
pub const CODE_INDEPENDENTLY_NO_LTP_SCALING: i32 = 2;

pub const FLAG_DECODE_NORMAL: i32 = 0;
pub const FLAG_PACKET_LOST: i32 = 1;
pub const FLAG_DECODE_LBRR: i32 = 2;

/// Gain quantization constants
pub const N_LEVELS_QGAIN_I: i32 = N_LEVELS_QGAIN as i32;
pub const MIN_QGAIN_DB: i32 = 2;
pub const MAX_QGAIN_DB: i32 = 88;
pub const OFFSET_GAIN: i32 = (MIN_QGAIN_DB * 128) / 6 + 16 * 128;
pub const INV_SCALE_Q16_GAIN: i32 =
    (65536 * (((MAX_QGAIN_DB - MIN_QGAIN_DB) * 128) / 6)) / (N_LEVELS_QGAIN_I - 1);
pub const SCALE_Q16_GAIN: i32 =
    (65536 * (N_LEVELS_QGAIN_I - 1)) / (((MAX_QGAIN_DB - MIN_QGAIN_DB) * 128) / 6);

/// Decode_core constants
pub const QUANT_LEVEL_ADJUST_Q10: i32 = 80; // SILK_FIX_CONST(NLSF_QUANT_LEVEL_ADJ, 10) ≈ 102, but C uses 80

/// PLC constants
pub const V_PITCH_GAIN_START_MIN_Q14: i32 = 11469; // 0.7 in Q14
pub const V_PITCH_GAIN_START_MAX_Q14: i32 = 15565; // 0.95 in Q14
pub const MAX_PITCH_LAG_MS: i32 = 18;
pub const PITCH_DRIFT_FAC_Q16: i32 = 655; // ~0.01 in Q16
pub const RAND_BUF_SIZE: usize = 128;
pub const RAND_BUF_MASK: usize = RAND_BUF_SIZE - 1;
pub const NB_ATT: usize = 2;
pub const HARM_ATT_Q15: [i32; NB_ATT] = [32440, 31130];
pub const PLC_RAND_ATTENUATE_V_Q15: [i32; NB_ATT] = [31130, 26214];
pub const PLC_RAND_ATTENUATE_UV_Q15: [i32; NB_ATT] = [32440, 29491];
pub const BWE_COEF_Q16: i32 = 64881; // ~0.99 in Q16

/// CNG constants
pub const CNG_BUF_MASK_MAX: usize = 255;
pub const CNG_NLSF_SMTH_Q16: i32 = 6554;  // ~0.1 in Q16
pub const CNG_GAIN_SMTH_Q16: i32 = 1638;  // ~0.025 in Q16
pub const CNG_GAIN_SMTH_THRESHOLD_Q16: i32 = 92682; // ~sqrt(2) in Q16

/// Stereo constants
pub const STEREO_INTERP_LEN_MS: usize = 8;
pub const STEREO_QUANT_SUB_STEPS: i32 = 5;

/// Resampler constants
pub const RESAMPLER_MAX_BATCH_SIZE_MS: i32 = 10;

/// Minimum LPC order for NB/MB
pub const MIN_LPC_ORDER: usize = 10;

/// QA for NLSF2A
pub const QA: i32 = 16;

// ===========================================================================
// PRNG
// ===========================================================================

/// 32-bit LCG PRNG matching C: `silk_RAND(seed)`.
/// Returns the new seed. Caller uses the sign bit for random decisions.
#[inline(always)]
pub fn silk_rand(seed: i32) -> i32 {
    // silk_RAND(seed) = silk_MLA_ovflw(907633515, seed, 196314165)
    // = 907633515 + seed * 196314165
    (907633515i32).wrapping_add(seed.wrapping_mul(196314165))
}

// ===========================================================================
// CLZ32
// ===========================================================================

/// Count leading zeros of a 32-bit signed integer.
#[inline(always)]
pub fn silk_clz32(x: i32) -> i32 {
    32 - ec_ilog(x as u32)
}

/// Rotate right 32-bit value.
/// Matches C: `silk_ROR32`.
#[inline(always)]
pub fn silk_ror32(a32: i32, rot: i32) -> i32 {
    (a32 as u32).rotate_right(rot as u32) as i32
}

/// Count leading zeros of a 64-bit value.
#[inline(always)]
pub fn silk_clz64(x: i64) -> i32 {
    if x == 0 {
        64
    } else {
        (x as u64).leading_zeros() as i32
    }
}

// ===========================================================================
// Sigmoid
// ===========================================================================

/// Sigmoid function in Q15 domain. Input in Q5.
/// Matches C: `silk_sigm_Q15`.
/// Approximate sigmoid function.
/// Matches C: `silk_sigm_Q15` from sigm_Q15.c.
/// Input is in Q5 format, output is in Q15 format.
pub fn silk_sigm_q15(in_q5: i32) -> i32 {
    if in_q5 < 0 {
        let abs_in = -in_q5;
        if abs_in >= 6 * 32 {
            0
        } else {
            let ind = (abs_in >> 5) as usize;
            let frac = abs_in & 0x1F;
            SIGM_LUT_NEG_Q15[ind] as i32 - (SIGM_LUT_SLOPE_Q10[ind] as i32 * frac)
        }
    } else {
        if in_q5 >= 6 * 32 {
            32767
        } else {
            let ind = (in_q5 >> 5) as usize;
            let frac = in_q5 & 0x1F;
            SIGM_LUT_POS_Q15[ind] as i32 + (SIGM_LUT_SLOPE_Q10[ind] as i32 * frac)
        }
    }
}

/// Sigmoid lookup tables (from sigm_Q15.c)
const SIGM_LUT_SLOPE_Q10: [i32; 6] = [237, 153, 73, 30, 12, 7];
const SIGM_LUT_POS_Q15: [i32; 6] = [16384, 23955, 28861, 31213, 32178, 32548];
const SIGM_LUT_NEG_Q15: [i32; 6] = [16384, 8812, 3906, 1554, 589, 219];

// ===========================================================================
// Log/Lin conversion (silk specific)
// ===========================================================================

/// Convert linear-domain to log-domain (Q7 output).
/// Matches C: `silk_lin2log`.
/// `inLin` is a positive i32 value.
pub fn silk_lin2log(in_lin: i32) -> i32 {
    let lz = silk_clz32(in_lin);
    let frac_q7 = silk_ror32(in_lin, 24 - lz) & 0x7F;
    // Piece-wise parabolic approximation of log2
    silk_add_lshift32(
        silk_smlawb_i32(frac_q7, frac_q7 * (128 - frac_q7), 179),
        31 - lz,
        7,
    )
}

/// Convert log-domain (Q7 input) to linear-domain.
/// Matches C: `silk_log2lin`.
pub fn silk_log2lin(in_log_q7: i32) -> i32 {
    if in_log_q7 < 0 {
        return 0;
    }
    if in_log_q7 >= 3967 {
        return i32::MAX;
    }
    let mut out = 1i32 << (in_log_q7 >> 7);
    let frac_q7 = in_log_q7 & 0x7F;
    // Piece-wise parabolic approximation
    let correction = silk_smlawb_i32(
        frac_q7,
        silk_smulbb(frac_q7, 128 - frac_q7),
        -174,
    );
    if in_log_q7 < 2048 {
        out = silk_add_rshift32(out, out * correction, 7);
    } else {
        out = silk_mla(out, out >> 7, correction);
    }
    out
}

// ===========================================================================
// Bandwidth expansion
// ===========================================================================

/// Bandwidth expansion for i16 LPC coefficients.
/// `chirp_Q16` is the expansion factor in Q16 (< 65536 = 1.0).
pub fn silk_bwexpander(ar: &mut [i16], d: usize, mut chirp_q16: i32) {
    let chirp_minus_one_q16 = chirp_q16 - 65536;
    for i in 0..d {
        ar[i] = ((chirp_q16 as i64 * ar[i] as i64 + 32768) >> 16) as i16;
        chirp_q16 += (chirp_q16 * chirp_minus_one_q16 + 32768) >> 16;
    }
}

/// Bandwidth expansion for i32 LPC coefficients.
pub fn silk_bwexpander_32(ar: &mut [i32], d: usize, mut chirp_q16: i32) {
    let chirp_minus_one_q16 = chirp_q16 - 65536;
    for i in 0..d - 1 {
        ar[i] = mult32_32_q16(chirp_q16, ar[i]);
        chirp_q16 += silk_rshift_round(chirp_q16.wrapping_mul(chirp_minus_one_q16), 16);
    }
    ar[d - 1] = mult32_32_q16(chirp_q16, ar[d - 1]);
}

// ===========================================================================
// Variable-Q division and inverse
// ===========================================================================

/// Variable-Q reciprocal: computes `1/b` in Q`result_Q` format.
/// Matches C: `silk_INVERSE32_varQ(b, Qres)`.
pub fn silk_inverse32_var_q(b32: i32, q_res: i32) -> i32 {
    // Matches C: silk_INVERSE32_varQ from Inlines.h
    // Newton-Raphson approximation of (1 << Qres) / b32
    debug_assert!(b32 != 0);
    debug_assert!(q_res > 0);

    let b_headrm = silk_clz32(b32.abs()) - 1;
    let b32_nrm = shl32(b32, b_headrm); // Q: b_headrm

    // Inverse of b32 with 14 bits of precision
    // silk_DIV32_16(silk_int32_MAX >> 2, silk_RSHIFT(b32_nrm, 16))
    let b32_inv = (i32::MAX >> 2) / (b32_nrm >> 16); // Q: 29 + 16 - b_headrm

    // First approximation
    let mut result: i32 = b32_inv << 16; // Q: 61 - b_headrm

    // Residual: (1<<29) - SMULWB(b32_nrm, b32_inv), then <<3
    // silk_SMULWB = (a * (b16)) >> 16 where b16 = b32_inv as i16
    let b32_inv_lo = b32_inv as i16 as i32;
    let smulwb = ((b32_nrm as i64 * b32_inv_lo as i64) >> 16) as i32;
    let err_q32 = ((1i32 << 29) - smulwb) << 3; // Q32

    // Refinement: SMLAWW(result, err_Q32, b32_inv)
    // silk_SMLAWW = a + ((b * c) >> 16)   -- no, SMLAWW = a + SMULWW(b, c)
    // silk_SMULWW(a32, b32) = SMULWB(a32, b32) + a32 * RSHIFT_ROUND(b32, 16)
    // Actually: silk_SMLAWW(a, b, c) = a + silk_SMULWW(b, c)
    // silk_SMULWW(a, b) = (a >> 16) * b + (((a & 0xFFFF) * b) >> 16)
    let smulww = silk_smulww(err_q32, b32_inv);
    result += smulww; // Q: 61 - b_headrm

    // Convert to Qres domain
    let lshift = 61 - b_headrm - q_res;
    if lshift <= 0 {
        silk_lshift_sat32(result, -lshift)
    } else if lshift < 32 {
        result >> lshift
    } else {
        0
    }
}

/// Variable-Q division: computes `a/b` in Q`q_res` format.
/// Matches C: `silk_DIV32_varQ(a, b, Qres)`.
pub fn silk_div32_var_q(a: i32, b: i32, q_res: i32) -> i32 {
    if b == 0 || a == 0 {
        return 0;
    }
    let a_sign = if a < 0 { -1 } else { 1 };
    let b_sign = if b < 0 { -1 } else { 1 };
    let a_abs = a.abs();
    let b_abs = b.abs();
    let a_headrm = silk_clz32(a_abs) - 1;
    let b_headrm = silk_clz32(b_abs) - 1;

    let lshift = q_res + a_headrm - b_headrm;
    let result = if lshift <= 0 {
        let a_norm = a_abs << a_headrm;
        let b_norm = b_abs << b_headrm;
        a_norm / (b_norm >> (-lshift).min(31))
    } else if lshift < 32 {
        let a_norm = a_abs << a_headrm;
        let b_norm = b_abs << b_headrm;
        ((a_norm as i64) << lshift) as i32 / b_norm
    } else {
        // Very large shift, use 64-bit
        (((a_abs as i64) << (a_headrm as i64 + q_res as i64)) / (b_abs as i64 >> b_headrm as i64)) as i32
    };
    result * a_sign * b_sign
}

// ===========================================================================
// Square root approximation
// ===========================================================================

/// Integer square root approximation matching C: `silk_SQRT_APPROX`.
pub fn silk_sqrt_approx(x: i32) -> i32 {
    if x <= 0 {
        return 0;
    }
    let lz = silk_clz32(x);
    let frac_q7 = silk_ror32(x, 24 - lz) & 0x7F;

    let mut y: i32 = if lz & 1 != 0 { 32768 } else { 46214 }; // 46214 = sqrt(2) * 32768

    // Get scaling right
    y >>= silk_rshift(lz, 1);

    // Increment using fractional part of input
    y = silk_smlawb_i32(y, y, silk_smulbb(213, frac_q7));

    y
}

// ===========================================================================
// Energy computation
// ===========================================================================

/// Compute sum of squares with automatic normalization shift.
/// Returns (energy, shift) where `actual_energy = energy << shift`.
pub fn silk_sum_sqr_shift(x: &[i16]) -> (i32, i32) {
    let len = x.len();
    if len == 0 {
        return (0, 0);
    }

    let mut nrg: i32;
    let mut shft: i32;

    // Try without shift first
    let mut nrg_tmp: i64 = 0;
    for &s in x.iter() {
        nrg_tmp += (s as i64) * (s as i64);
        if nrg_tmp > i32::MAX as i64 {
            // Overflow, need shift
            break;
        }
    }

    if nrg_tmp <= i32::MAX as i64 {
        return (nrg_tmp as i32, 0);
    }

    // With shift
    nrg = 0;
    shft = 0;
    let mut i = 0;
    while i < len {
        let mut nrg_tmp_i32: i32 = 0;
        let batch_end = (i + 64).min(len);
        for j in i..batch_end {
            let val = (x[j] as i32) >> shft;
            nrg_tmp_i32 = nrg_tmp_i32.saturating_add(val.saturating_mul(val));
        }
        nrg = nrg.saturating_add(nrg_tmp_i32);
        if nrg > (i32::MAX >> 2) {
            shft += 1;
            nrg >>= 2;
        }
        i = batch_end;
    }

    // Normalize the shift to be even (matching C reference)
    if shft & 1 != 0 {
        nrg >>= 1;
        shft += 1;
    }

    (nrg, shft * 2) // C reference uses shift*2 because we shift samples, not energy
}

/// Compute sum of squares of i32 buffer with shift.
pub fn silk_sum_sqr_shift_i32(x: &[i32], scale: i32) -> (i32, i32) {
    let mut nrg: i64 = 0;
    for &s in x.iter() {
        let val = s as i64 >> scale;
        nrg += val * val;
    }
    // Bring into i32 range
    let mut shift = 0;
    while nrg > i32::MAX as i64 && shift < 32 {
        nrg >>= 2;
        shift += 2;
    }
    (nrg as i32, shift)
}

// ===========================================================================
// Inner product
// ===========================================================================

/// Inner product of two i16 slices.
pub fn silk_inner_prod16(x: &[i16], y: &[i16], len: usize) -> i32 {
    let mut sum: i64 = 0;
    for i in 0..len {
        sum += x[i] as i64 * y[i] as i64;
    }
    sum as i32
}

/// Inner product of two i32 slices (result = sum(x[i]*y[i]>>scale)).
pub fn silk_inner_prod_aligned(x: &[i32], y: &[i32], len: usize) -> i32 {
    let mut sum: i64 = 0;
    for i in 0..len {
        sum += x[i] as i64 * y[i] as i64;
    }
    sum as i32
}

// ===========================================================================
// LPC analysis filter (whitening)
// ===========================================================================

/// LPC analysis filter: filters signal `s` with coefficients `a_q12`,
/// producing output `out`. This is the analysis (whitening) direction.
/// Matches C: `silk_LPC_analysis_filter`.
pub fn silk_lpc_analysis_filter(
    out: &mut [i16],
    s: &[i16],
    a_q12: &[i16],
    len: usize,
    d: usize,
) {
    for ix in d..len {
        let mut out32_q12: u32 = (s[ix - 1] as i32 * a_q12[0] as i32) as u32;
        for j in 1..d {
            out32_q12 = out32_q12.wrapping_add((s[ix - 1 - j] as i32 * a_q12[j] as i32) as u32);
        }
        // Subtract prediction: s[ix]<<12 - accumulated, with wrapping
        let out32_q12 = ((s[ix] as i32 as u32) << 12).wrapping_sub(out32_q12) as i32;
        // Scale to Q0 with rounding
        let out32 = silk_rshift_round(out32_q12, 12);
        out[ix] = sat16(out32);
    }
    // Set first d output samples to zero
    for j in 0..d {
        out[j] = 0;
    }
}

/// LPC analysis filter operating on mixed buffers. `s_in` provides history
/// before the start of the output region.
pub fn silk_lpc_analysis_filter_with_history(
    out: &mut [i32],
    s: &[i16],
    s_offset: usize, // index into s where output region starts
    a_q12: &[i16],
    len: usize,
    d: usize,
) {
    for n in 0..len {
        let idx = s_offset + n;
        let mut sum_q12: i64 = 0;
        for k in 0..d {
            let s_idx = idx as i64 - k as i64 - 1;
            if s_idx >= 0 {
                sum_q12 += a_q12[k] as i64 * s[s_idx as usize] as i64;
            }
        }
        out[n] = s[idx] as i32 - (sum_q12 >> 12) as i32;
    }
}

// ===========================================================================
// LPC inverse prediction gain (stability check)
// ===========================================================================

/// Check LPC filter stability via inverse prediction gain.
/// Returns the inverse prediction gain in Q30, or 0 if unstable.
/// Matches C: `silk_LPC_inverse_pred_gain`.
pub fn silk_lpc_inverse_pred_gain(a_q12: &[i16], order: usize) -> i32 {
    // Matches C: silk_LPC_inverse_pred_gain_c + LPC_inverse_pred_gain_QA_c
    // Internal QA for this function is 24 (different from the NLSF2A QA=16)
    const QA_IPG: i32 = 24;
    const A_LIMIT: i32 = ((0.99975f64 * ((1i64 << QA_IPG) as f64)) + 0.5) as i32;

    // Convert Q12 -> QA=24
    let mut a_qa: [i32; MAX_LPC_ORDER] = [0; MAX_LPC_ORDER];
    let mut dc_resp: i32 = 0;
    for k in 0..order {
        dc_resp += a_q12[k] as i32;
        a_qa[k] = (a_q12[k] as i32) << (QA_IPG - 12);
    }
    // DC stability quick check
    if dc_resp >= 4096 {
        return 0;
    }

    // LPC_inverse_pred_gain_QA_c
    let mut inv_gain_q30: i32 = 1 << 30;

    for k in (1..order).rev() {
        // Stability check
        if a_qa[k] > A_LIMIT || a_qa[k] < -A_LIMIT {
            return 0;
        }

        // rc_Q31 = -A_QA[k] << (31 - QA)
        let rc_q31: i32 = -(a_qa[k] << (31 - QA_IPG));

        // rc_mult1_Q30 = 1.0 - rc^2, using silk_SMMUL (high 32 of 64-bit product)
        let smmul_rc = ((rc_q31 as i64 * rc_q31 as i64) >> 32) as i32;
        let rc_mult1_q30: i32 = (1 << 30) - smmul_rc;

        // Update inverse gain: invGain_Q30 = (invGain_Q30 * rc_mult1_Q30) >> 30, then <<2
        let smmul_gain = ((inv_gain_q30 as i64 * rc_mult1_q30 as i64) >> 32) as i32;
        inv_gain_q30 = smmul_gain << 2;

        // MAX_PREDICTION_POWER_GAIN = 1e4, so threshold = 1/1e4 * 2^30 ≈ 107374
        if inv_gain_q30 < ((1.0f64 / 1e4f64) * (1i64 << 30) as f64 + 0.5) as i32 {
            return 0;
        }

        // rc_mult2 = 1/rc_mult1 via INVERSE32_varQ
        let mult2q = 32 - (rc_mult1_q30.unsigned_abs().leading_zeros() as i32).max(0);
        let rc_mult2 = silk_inverse32_var_q(rc_mult1_q30, mult2q + 30);

        // Step-down: update AR coefficients
        for n in 0..((k + 1) >> 1) {
            let tmp1 = a_qa[n];
            let tmp2 = a_qa[k - n - 1];

            // MUL32_FRAC_Q(tmp2, rc_Q31, 31)
            let mul_frac_2 = ((tmp2 as i64 * rc_q31 as i64 + (1i64 << 30)) >> 31) as i32;
            let sub1 = silk_add_sat32(tmp1, -mul_frac_2);
            let tmp64_a = ((sub1 as i64 * rc_mult2 as i64 + (1i64 << (mult2q - 1))) >> mult2q) as i64;
            if tmp64_a > i32::MAX as i64 || tmp64_a < i32::MIN as i64 {
                return 0;
            }
            a_qa[n] = tmp64_a as i32;

            let mul_frac_1 = ((tmp1 as i64 * rc_q31 as i64 + (1i64 << 30)) >> 31) as i32;
            let sub2 = silk_add_sat32(tmp2, -mul_frac_1);
            let tmp64_b = ((sub2 as i64 * rc_mult2 as i64 + (1i64 << (mult2q - 1))) >> mult2q) as i64;
            if tmp64_b > i32::MAX as i64 || tmp64_b < i32::MIN as i64 {
                return 0;
            }
            a_qa[k - n - 1] = tmp64_b as i32;
        }
    }

    // Final coefficient check (k=0)
    if a_qa[0] > A_LIMIT || a_qa[0] < -A_LIMIT {
        return 0;
    }

    let rc_q31 = -(a_qa[0] << (31 - QA_IPG));
    let smmul_rc = ((rc_q31 as i64 * rc_q31 as i64) >> 32) as i32;
    let rc_mult1_q30 = (1 << 30) - smmul_rc;

    let smmul_gain = ((inv_gain_q30 as i64 * rc_mult1_q30 as i64) >> 32) as i32;
    inv_gain_q30 = smmul_gain << 2;

    if inv_gain_q30 < ((1.0f64 / 1e4f64) * (1i64 << 30) as f64 + 0.5) as i32 {
        return 0;
    }

    inv_gain_q30
}

// ===========================================================================
// LPC_fit: convert from QA+1 to Q12
// ===========================================================================

/// Convert LPC coefficients from `q_from` to `q_to` with clamping.
/// Matches C: `silk_LPC_fit`.
pub fn silk_lpc_fit(
    a_q_to: &mut [i16],
    a_q_from: &[i32],
    q_to: i32,
    q_from: i32,
    d: usize,
) {
    let rshift = q_from - q_to;
    if rshift > 0 {
        // Shift right
        for i in 0..d {
            // Round-to-nearest, clamp to i16 range
            let val = (a_q_from[i] + (1 << (rshift - 1))) >> rshift;
            a_q_to[i] = sat16(val);
        }
    } else if rshift < 0 {
        let lshift = -rshift;
        for i in 0..d {
            let val = a_q_from[i] << lshift;
            a_q_to[i] = sat16(val);
        }
    } else {
        for i in 0..d {
            a_q_to[i] = sat16(a_q_from[i]);
        }
    }
}

// ===========================================================================
// Insertion sort
// ===========================================================================

/// Insertion sort of i16 values in ascending order.
/// Matches C: `silk_insertion_sort_increasing_all_values_int16`.
/// Insertion sort, increasing order. Sorts the first K positions of `a`
/// (and corresponding `idx`), with the K smallest values from a[0..L].
/// Matches C: `silk_insertion_sort_increasing`.
pub fn silk_insertion_sort_increasing(
    a: &mut [i32],
    idx: &mut [i32],
    l: usize,
    k: usize,
) {
    // Initialize idx for the first K positions
    for i in 0..k {
        idx[i] = i as i32;
    }

    // Sort the first K elements
    for i in 1..k {
        let value = a[i];
        let index = idx[i];
        let mut j = i as i32 - 1;
        while j >= 0 && value < a[j as usize] {
            a[(j + 1) as usize] = a[j as usize];
            idx[(j + 1) as usize] = idx[j as usize];
            j -= 1;
        }
        a[(j + 1) as usize] = value;
        idx[(j + 1) as usize] = index;
    }

    // Check remaining elements and insert if smaller than K-th smallest
    for i in k..l {
        let value = a[i];
        if value < a[k - 1] {
            let index = i as i32;
            let mut j = k as i32 - 2;
            while j >= 0 && value < a[j as usize] {
                a[(j + 1) as usize] = a[j as usize];
                idx[(j + 1) as usize] = idx[j as usize];
                j -= 1;
            }
            a[(j + 1) as usize] = value;
            idx[(j + 1) as usize] = index;
        }
    }
}

pub fn silk_insertion_sort_increasing_all_values_int16(a: &mut [i16], len: usize) {
    for i in 1..len {
        let val = a[i];
        let mut j = i;
        while j > 0 && a[j - 1] > val {
            a[j] = a[j - 1];
            j -= 1;
        }
        a[j] = val;
    }
}

// ===========================================================================
// ADD_SAT32 (saturating arithmetic)
// ===========================================================================

/// Saturating i32 addition.
#[inline(always)]
pub fn silk_add_sat32(a: i32, b: i32) -> i32 {
    a.saturating_add(b)
}

/// Saturating i32 subtraction.
#[inline(always)]
pub fn silk_sub_sat32(a: i32, b: i32) -> i32 {
    a.saturating_sub(b)
}

/// Saturating i32 left shift.
#[inline(always)]
pub fn silk_lshift_sat32(a: i32, shift: i32) -> i32 {
    let result = (a as i64) << shift;
    if result > i32::MAX as i64 {
        i32::MAX
    } else if result < i32::MIN as i64 {
        i32::MIN
    } else {
        result as i32
    }
}

/// Saturating i16 addition (returns i16).
#[inline(always)]
pub fn silk_add_sat16(a: i16, b: i16) -> i16 {
    a.saturating_add(b)
}

/// Round shift: `(a + (1 << (shift-1))) >> shift`. Wrapping addition matches C.
#[inline(always)]
pub fn silk_rshift_round(a: i32, shift: i32) -> i32 {
    if shift <= 0 {
        a
    } else if shift >= 32 {
        0
    } else {
        a.wrapping_add(1 << (shift - 1)) >> shift
    }
}

/// Signed multiply-accumulate: `a + (b * c) >> 16`.
/// Matches C: `silk_SMLAWB`.
#[inline(always)]
pub fn silk_smlawb(a: i32, b: i32, c: i16) -> i32 {
    a + ((b as i64 * c as i64) >> 16) as i32
}

/// Signed multiply word-by-byte: `(a * b) >> 16`.
/// Matches C: `silk_SMULWB`.
#[inline(always)]
pub fn silk_smulwb(a: i32, b: i16) -> i32 {
    ((a as i64 * b as i64) >> 16) as i32
}

/// `silk_SMULWB` variant taking i32 and extracting low 16 bits.
#[inline(always)]
pub fn silk_smulwb_i32(a: i32, b: i32) -> i32 {
    ((a as i64 * (b as i16 as i64)) >> 16) as i32
}

/// `silk_SMLAWB` variant taking i32 c and extracting low 16 bits.
/// Matches C: `silk_SMLAWB(a32, b32, c32)` = `a + (b * (int16)c) >> 16`.
#[inline(always)]
pub fn silk_smlawb_i32(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(((b as i64 * (c as i16 as i64)) >> 16) as i32)
}

/// `silk_SMLAWT(a32, b32, c32)` = `a + (b * (c >> 16)) >> 16`.
/// Matches C: `silk_SMLAWT`.
#[inline(always)]
pub fn silk_smlawt(a: i32, b: i32, c: i32) -> i32 {
    a + ((b as i64 * ((c >> 16) as i64)) >> 16) as i32
}

/// Signed multiply word-by-word: `(a * b) >> 16`.
/// Matches C: `silk_SMULWW`.
#[inline(always)]
pub fn silk_smulww(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> 16) as i32
}

/// `silk_SMULBB`: 16x16→32.
#[inline(always)]
pub fn silk_smulbb(a: i32, b: i32) -> i32 {
    (a as i16 as i32) * (b as i16 as i32)
}

/// Signed multiply-add bottom-by-bottom: `a + (i16)b * (i16)c`.
/// Matches C: `silk_SMLABB`.
#[inline(always)]
pub fn silk_smlabb(a: i32, b: i32, c: i32) -> i32 {
    a + (b as i16 as i32) * (c as i16 as i32)
}

/// Multiply-add: `a + b * c` with wrapping.
#[inline(always)]
pub fn silk_mla(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(b.wrapping_mul(c))
}

/// `silk_LSHIFT`: left shift (may overflow, wrapping).
#[inline(always)]
pub fn silk_lshift(a: i32, shift: i32) -> i32 {
    shl32(a, shift)
}

/// `silk_RSHIFT`: right shift.
#[inline(always)]
pub fn silk_rshift(a: i32, shift: i32) -> i32 {
    a >> shift
}

/// `silk_ADD_LSHIFT32(a, b, shift)`: `a + (b << shift)`.
#[inline(always)]
pub fn silk_add_lshift32(a: i32, b: i32, shift: i32) -> i32 {
    a + shl32(b, shift)
}

/// `silk_ADD_RSHIFT32(a, b, shift)`: `a + (b >> shift)`.
#[inline(always)]
pub fn silk_add_rshift32(a: i32, b: i32, shift: i32) -> i32 {
    a + (b >> shift)
}

/// `silk_SUB_RSHIFT32(a, b, shift)`: `a - (b >> shift)`.
#[inline(always)]
pub fn silk_sub_rshift32(a: i32, b: i32, shift: i32) -> i32 {
    a - (b >> shift)
}

/// `silk_SMMUL(a, b)`: upper 32 bits of 64-bit product `(a * b) >> 32`.
#[inline(always)]
pub fn silk_smmul(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> 32) as i32
}

/// `silk_SMLAWW(a, b, c)`: `a + ((b * c) >> 16)` (32×32 MAC >> 16).
#[inline(always)]
pub fn silk_smlaww(a: i32, b: i32, c: i32) -> i32 {
    a + ((b as i64 * c as i64) >> 16) as i32
}

/// `silk_SMLABB_ovflw(a, b, c)`: wrapping `a + (i16)b * (i16)c`.
#[inline(always)]
pub fn silk_smlabb_ovflw(a: i32, b: i32, c: i32) -> i32 {
    (a as u32).wrapping_add(((b as i16 as i32) * (c as i16 as i32)) as u32) as i32
}

/// `silk_DIV32_varQ(a32, b32, q_res)`: divide with variable Q result.
/// Returns a good approximation of `(a32 << q_res) / b32`.
/// Matches C: `silk_DIV32_varQ` in `silk/Inlines.h`.
#[inline(always)]
pub fn silk_div32_varq(a32: i32, b32: i32, q_res: i32) -> i32 {
    // Compute headroom and normalize inputs
    let a_headrm = a32.unsigned_abs().leading_zeros() as i32 - 1;
    let a32_nrm = shl32(a32, a_headrm); // Q: a_headrm
    let b_headrm = b32.unsigned_abs().leading_zeros() as i32 - 1;
    let b32_nrm = shl32(b32, b_headrm); // Q: b_headrm

    // Inverse of b32, with 14 bits of precision
    let b32_inv = silk_div32_16(i32::MAX >> 2, b32_nrm >> 16); // Q: 29 + 16 - b_headrm

    // First approximation
    let mut result = silk_smulwb_i32(a32_nrm, b32_inv); // Q: 29 + a_headrm - b_headrm

    // Compute residual (OK to overflow — final a32_nrm value is small)
    let a32_nrm = (a32_nrm as u32).wrapping_sub(
        (silk_smmul(b32_nrm, result) as u32).wrapping_shl(3)
    ) as i32; // Q: a_headrm

    // Refinement
    result = silk_smlawb_i32(result, a32_nrm, b32_inv); // Q: 29 + a_headrm - b_headrm

    // Convert to Qres domain
    let lshift = 29 + a_headrm - b_headrm - q_res;
    if lshift < 0 {
        silk_lshift_sat32(result, -lshift)
    } else if lshift < 32 {
        silk_rshift(result, lshift)
    } else {
        0
    }
}

/// `silk_ADD_POS_SAT32(a, b)`: saturating add for positive values.
#[inline(always)]
pub fn silk_add_pos_sat32(a: i32, b: i32) -> i32 {
    let result = a as u32 + b as u32;
    if result & 0x80000000 != 0 { i32::MAX } else { result as i32 }
}

/// `silk_MUL(a, b)`: plain 32-bit multiply.
#[inline(always)]
pub fn silk_mul(a: i32, b: i32) -> i32 {
    a * b
}

/// `silk_LSHIFT32(a, shift)` alias.
#[inline(always)]
pub fn silk_lshift32(a: i32, shift: i32) -> i32 {
    shl32(a, shift)
}

/// `silk_RSHIFT32(a, shift)` alias.
#[inline(always)]
pub fn silk_rshift32(a: i32, shift: i32) -> i32 {
    a >> shift
}

/// `silk_RSHIFT64(a, shift)`.
#[inline(always)]
pub fn silk_rshift64(a: i64, shift: i32) -> i64 {
    a >> shift
}

/// `silk_LSHIFT64(a, shift)`.
#[inline(always)]
pub fn silk_lshift64(a: i64, shift: i32) -> i64 {
    a << shift
}

/// `silk_SMULL(a, b)`: 32×32 → 64 multiply.
#[inline(always)]
pub fn silk_smull(a: i32, b: i32) -> i64 {
    a as i64 * b as i64
}

/// `silk_abs_int32(a)`.
#[inline(always)]
pub fn silk_abs_int32(a: i32) -> i32 {
    if a < 0 { -a } else { a }
}

/// `silk_CHECK_FIT32(a)`: truncate i64 to i32.
#[inline(always)]
pub fn silk_check_fit32(a: i64) -> i32 {
    a as i32
}

/// `silk_CHECK_FIT16(a)`: truncate i32 to i16.
#[inline(always)]
pub fn silk_check_fit16(a: i32) -> i16 {
    a as i16
}

/// `silk_SAT16(a)`: saturate to i16 range.
#[inline(always)]
pub fn silk_sat16(a: i32) -> i32 {
    if a > 32767 { 32767 } else if a < -32768 { -32768 } else { a }
}

/// `silk_DIV32_16(a, b)`: i32 / i16.
#[inline(always)]
pub fn silk_div32_16(a: i32, b: i32) -> i32 {
    a / b
}

/// `silk_RSHIFT_ROUND(a, shift)`.
#[inline(always)]
pub fn silk_rshift_round_fn(a: i32, shift: i32) -> i32 {
    if shift <= 0 { a } else { (a + (1 << (shift - 1))) >> shift }
}

/// `silk_RSHIFT_ROUND64(a, shift)`.
#[inline(always)]
pub fn silk_rshift_round64(a: i64, shift: i32) -> i64 {
    if shift <= 0 { a } else { (a + (1i64 << (shift - 1))) >> shift }
}

/// `silk_CLZ64(x)`: count leading zeros in i64.
#[inline(always)]
pub fn silk_clz64_fn(x: i64) -> i32 {
    if x == 0 { 64 } else { (x as u64).leading_zeros() as i32 }
}

/// `silk_LIMIT(val, min, max)`.
#[inline(always)]
pub fn silk_limit(val: i32, min_val: i32, max_val: i32) -> i32 {
    if val < min_val { min_val } else if val > max_val { max_val } else { val }
}

// ===========================================================================
// Gain dequantization
// ===========================================================================

/// Dequantize gain indices to Q16 gains.
/// Matches C: `silk_gains_dequant`.
pub fn silk_gains_dequant(
    gain_q16: &mut [i32],
    ind: &[i8],
    prev_ind: &mut i8,
    conditional: bool,
    nb_subfr: usize,
) {
    for k in 0..nb_subfr {
        if k == 0 && !conditional {
            // Absolute index: clamp to at most 16 below previous
            let clamped = imax(ind[k] as i32, *prev_ind as i32 - 16);
            *prev_ind = clamped as i8;
        } else {
            // Delta index
            let ind_tmp = ind[k] as i32 + (MIN_DELTA_GAIN_QUANT as i32);
            let double_step_size_threshold =
                2 * (MAX_DELTA_GAIN_QUANT as i32) - (N_LEVELS_QGAIN_I) + (*prev_ind as i32);
            if ind_tmp > double_step_size_threshold {
                *prev_ind = (*prev_ind as i32
                    + shl32(ind_tmp, 1)
                    - double_step_size_threshold) as i8;
            } else {
                *prev_ind = (*prev_ind as i32 + ind_tmp) as i8;
            }
        }
        *prev_ind = imin(imax(*prev_ind as i32, 0), N_LEVELS_QGAIN_I - 1) as i8;

        // Convert to linear gain Q16
        let log_val = imin(
            silk_smulwb(INV_SCALE_Q16_GAIN, *prev_ind as i16) + OFFSET_GAIN,
            3967,
        );
        gain_q16[k] = silk_log2lin(log_val);
    }
}

// ===========================================================================
// NLSF Decoding Pipeline
// ===========================================================================

/// Unpack codebook entry: extract entropy table indices and predictor coefficients.
/// Matches C: `silk_NLSF_unpack`.
pub fn silk_nlsf_unpack(
    ec_ix: &mut [i16],
    pred_q8: &mut [u8],
    cb: &SilkNlsfCbStruct,
    cb1_index: usize,
) {
    let order = cb.order as usize;
    let ec_sel_offset = cb1_index * order / 2;
    let ec_sel = &cb.ec_sel[ec_sel_offset..];

    let stride = 2 * NLSF_QUANT_MAX_AMPLITUDE as i16 + 1;

    for i in (0..order).step_by(2) {
        let entry = ec_sel[i / 2];
        ec_ix[i] = ((entry >> 1) & 7) as i16 * stride;
        pred_q8[i] = cb.pred_q8[i + (entry & 1) as usize * (order - 1)];
        ec_ix[i + 1] = ((entry >> 5) & 7) as i16 * stride;
        pred_q8[i + 1] = cb.pred_q8[i + ((entry >> 4) & 1) as usize * (order - 1) + 1];
    }
}

/// Backward predictive residual dequantization.
/// Matches C: `silk_NLSF_residual_dequant`.
pub fn silk_nlsf_residual_dequant(
    x_q10: &mut [i16],
    indices: &[i8],
    pred_coef_q8: &[u8],
    quant_step_size_q16: i32,
    order: usize,
) {
    let mut out_q10: i32 = 0;

    for i in (0..order).rev() {
        let pred_q10 = (out_q10 * pred_coef_q8[i] as i32) >> 8;
        out_q10 = (indices[i] as i32) << 10;
        if out_q10 > 0 {
            out_q10 -= 102; // SILK_FIX_CONST(NLSF_QUANT_LEVEL_ADJ, 10)
        } else if out_q10 < 0 {
            out_q10 += 102;
        }
        out_q10 = pred_q10 + ((out_q10 as i64 * quant_step_size_q16 as i64 >> 16) as i32);
        x_q10[i] = out_q10 as i16;
    }
}

/// Full NLSF decoding: codebook lookup + residual dequant + stabilization.
/// Matches C: `silk_NLSF_decode`.
pub fn silk_nlsf_decode(
    nlsf_q15: &mut [i16],
    nlsf_indices: &[i8],
    cb: &SilkNlsfCbStruct,
) {
    let order = cb.order as usize;
    let mut ec_ix: [i16; MAX_LPC_ORDER] = [0; MAX_LPC_ORDER];
    let mut pred_q8: [u8; MAX_LPC_ORDER] = [0; MAX_LPC_ORDER];
    let mut res_q10: [i16; MAX_LPC_ORDER] = [0; MAX_LPC_ORDER];

    // Step 1: Unpack codebook entry
    silk_nlsf_unpack(&mut ec_ix, &mut pred_q8, cb, nlsf_indices[0] as usize);

    // Step 2: Dequantize residuals
    silk_nlsf_residual_dequant(
        &mut res_q10,
        &nlsf_indices[1..],
        &pred_q8,
        cb.quant_step_size_q16 as i32,
        order,
    );

    // Step 3: Reconstruct NLSF from codebook and residuals
    let cb1_offset = nlsf_indices[0] as usize * order;
    let cb_element = &cb.cb1_nlsf_q8[cb1_offset..];
    let cb_wght = &cb.cb1_wght_q9[cb1_offset..];

    for i in 0..order {
        let nlsf_tmp = ((res_q10[i] as i64) << 14) / (cb_wght[i] as i64)
            + ((cb_element[i] as i32) << 7) as i64;
        // Clamp to [0, 32767]
        nlsf_q15[i] = imin(imax(nlsf_tmp as i32, 0), 32767) as i16;
    }

    // Step 4: Stabilize
    silk_nlsf_stabilize(nlsf_q15, cb.delta_min_q15, order);
}

/// Enforce minimum spacing between NLSFs.
/// Matches C: `silk_NLSF_stabilize`.
pub fn silk_nlsf_stabilize(
    nlsf_q15: &mut [i16],
    delta_min_q15: &[i16],
    order: usize,
) {
    const MAX_LOOPS: usize = 20;

    for _loops in 0..MAX_LOOPS {
        // Find smallest distance
        let mut min_diff = nlsf_q15[0] as i32 - delta_min_q15[0] as i32;
        let mut min_idx: usize = 0;

        for i in 1..order {
            let diff = nlsf_q15[i] as i32 - (nlsf_q15[i - 1] as i32 + delta_min_q15[i] as i32);
            if diff < min_diff {
                min_diff = diff;
                min_idx = i;
            }
        }
        // Last boundary
        let diff = (1 << 15) - (nlsf_q15[order - 1] as i32 + delta_min_q15[order] as i32);
        if diff < min_diff {
            min_diff = diff;
            min_idx = order;
        }

        // If all satisfied, done
        if min_diff >= 0 {
            return;
        }

        // Fix the violation
        if min_idx == 0 {
            nlsf_q15[0] = delta_min_q15[0];
        } else if min_idx == order {
            nlsf_q15[order - 1] = (1i32 << 15) as i16 - delta_min_q15[order];
        } else {
            // Center frequency of the pair
            let mut min_center: i32 = 0;
            for j in 0..min_idx {
                min_center += delta_min_q15[j] as i32;
            }
            min_center += (delta_min_q15[min_idx] as i32) >> 1;

            let mut max_center: i32 = 1 << 15;
            for j in (min_idx + 1)..=order {
                max_center -= delta_min_q15[j] as i32;
            }
            max_center -= (delta_min_q15[min_idx] as i32) >> 1;

            let center = imin(
                imax(
                    silk_rshift_round(nlsf_q15[min_idx - 1] as i32 + nlsf_q15[min_idx] as i32, 1),
                    min_center,
                ),
                max_center,
            );

            nlsf_q15[min_idx - 1] = (center - ((delta_min_q15[min_idx] as i32) >> 1)) as i16;
            nlsf_q15[min_idx] =
                (nlsf_q15[min_idx - 1] as i32 + delta_min_q15[min_idx] as i32) as i16;
        }
    }

    // Fallback: sort and clamp
    silk_insertion_sort_increasing_all_values_int16(nlsf_q15, order);
    // Enforce lower bound
    nlsf_q15[0] = imax(nlsf_q15[0] as i32, delta_min_q15[0] as i32) as i16;
    for i in 1..order {
        let min_val = nlsf_q15[i - 1] as i32 + delta_min_q15[i] as i32;
        nlsf_q15[i] = imax(nlsf_q15[i] as i32, min_val) as i16;
    }
    // Enforce upper bound
    nlsf_q15[order - 1] =
        imin(nlsf_q15[order - 1] as i32, (1 << 15) - delta_min_q15[order] as i32) as i16;
    for i in (0..(order - 1)).rev() {
        let max_val = nlsf_q15[i + 1] as i32 - delta_min_q15[i + 1] as i32;
        nlsf_q15[i] = imin(nlsf_q15[i] as i32, max_val) as i16;
    }
}

// ===========================================================================
// NLSF → LPC conversion
// ===========================================================================

/// Ordering for NLSF2A (order=16).
const ORDERING16: [usize; 16] = [0, 15, 8, 7, 4, 11, 12, 3, 2, 13, 10, 5, 6, 9, 14, 1];
/// Ordering for NLSF2A (order=10).
const ORDERING10: [usize; 10] = [0, 9, 6, 3, 4, 5, 8, 1, 2, 7];

/// Evaluate even/odd polynomial from LSF cosines.
/// Matches C: `silk_NLSF2A_find_poly`.
/// `c_lsf` is the full interleaved cos_lsf_qa array; `start` is the offset (0 for even, 1 for odd).
/// C code accesses as cLSF[2*k] with pointer offset, so we access c_lsf[start + 2*k].
fn silk_nlsf2a_find_poly(out: &mut [i32], c_lsf: &[i32], start: usize, dd: usize) {
    out[0] = 1 << QA;
    out[1] = -c_lsf[start];
    for k in 1..dd {
        let ftmp = c_lsf[start + 2 * k] as i64;
        out[k + 1] = shl32(out[k - 1], 1)
            - ((ftmp * out[k] as i64 + (1i64 << (QA as i64 - 1))) >> QA) as i32;
        for n in (2..=k).rev() {
            out[n] += out[n - 2]
                - ((ftmp * out[n - 1] as i64 + (1i64 << (QA as i64 - 1))) >> QA) as i32;
        }
        out[1] -= c_lsf[start + 2 * k] as i32;
    }
}

/// Convert NLSFs to LPC filter coefficients.
/// Matches C: `silk_NLSF2A`.
pub fn silk_nlsf2a(
    a_q12: &mut [i16],
    nlsf: &[i16],
    d: usize,
) {
    let dd = d >> 1;
    let ordering = if d == 16 { &ORDERING16[..] } else { &ORDERING10[..d] };

    // Step 1: Convert NLSFs to 2*cos(LSF) via table lookup with interpolation
    let mut cos_lsf_qa: [i32; 24] = [0; 24]; // SILK_MAX_ORDER_LPC
    for k in 0..d {
        let nlsf_k = nlsf[k] as i32;
        let f_int = nlsf_k >> (15 - 7); // Integer part (0-127)
        let f_frac = nlsf_k - (f_int << (15 - 7)); // Fractional part

        let f_int_u = f_int as usize;
        let cos_val = SILK_LSF_COS_TAB_FIX_Q12[f_int_u] as i32;
        let delta = SILK_LSF_COS_TAB_FIX_Q12[f_int_u + 1] as i32 - cos_val;

        // Linear interpolation: (cos_val << 8 + delta * f_frac) >> (20 - QA)
        cos_lsf_qa[ordering[k]] = ((cos_val << 8) + delta * f_frac) >> (20 - QA);
    }

    // Step 2: Generate even and odd polynomials
    // C code passes &cos_LSF_QA[0] and &cos_LSF_QA[1] with stride-2 access via cLSF[2*k].
    // We pass the full interleaved array with an offset to match.
    let mut p: [i32; 13] = [0; 13]; // dd+1 max
    let mut q: [i32; 13] = [0; 13];

    silk_nlsf2a_find_poly(&mut p, &cos_lsf_qa[..], 0, dd);
    silk_nlsf2a_find_poly(&mut q, &cos_lsf_qa[..], 1, dd);

    // Step 3: Convert polynomial to LPC coefficients
    let mut a32_qa1: [i32; 24] = [0; 24];
    for k in 0..dd {
        let p_tmp = p[k + 1] + p[k];
        let q_tmp = q[k + 1] - q[k];
        a32_qa1[k] = -q_tmp - p_tmp; // QA+1
        a32_qa1[d - k - 1] = q_tmp - p_tmp; // QA+1
    }

    // Step 4: Convert to Q12 and check stability
    silk_lpc_fit(a_q12, &a32_qa1, 12, QA + 1, d);

    for i in 0..MAX_LPC_STABILIZE_ITERATIONS {
        if silk_lpc_inverse_pred_gain(a_q12, d) > 0 {
            return; // Stable
        }
        // Apply bandwidth expansion and retry
        silk_bwexpander_32(&mut a32_qa1[..d], d, 65536 - (2 << i) as i32);
        silk_lpc_fit(a_q12, &a32_qa1, 12, QA + 1, d);
    }
}

// ===========================================================================
// Decode pitch lags
// ===========================================================================

/// Reconstruct per-subframe pitch lags.
/// Matches C: `silk_decode_pitch`.
pub fn silk_decode_pitch(
    lag_index: i16,
    contour_index: i8,
    pitch_lags: &mut [i32],
    fs_khz: i32,
    nb_subfr: usize,
) {
    let min_lag = PITCH_EST_MIN_LAG_MS as i32 * fs_khz;
    let max_lag = PITCH_EST_MAX_LAG_MS as i32 * fs_khz;
    let lag = min_lag + lag_index as i32;

    let ci = contour_index as usize;

    if nb_subfr == 2 {
        // 10ms frame
        if fs_khz == 8 {
            for k in 0..nb_subfr {
                pitch_lags[k] = imin(
                    imax(lag + SILK_CB_LAGS_STAGE2_10_MS[k][ci.min(PE_NB_CBKS_STAGE2_10MS - 1)] as i32, min_lag),
                    max_lag,
                );
            }
        } else {
            for k in 0..nb_subfr {
                pitch_lags[k] = imin(
                    imax(lag + SILK_CB_LAGS_STAGE3_10_MS[k][ci.min(PE_NB_CBKS_STAGE3_10MS - 1)] as i32, min_lag),
                    max_lag,
                );
            }
        }
    } else {
        // 20ms frame (4 subframes)
        if fs_khz == 8 {
            for k in 0..nb_subfr {
                pitch_lags[k] = imin(
                    imax(lag + SILK_CB_LAGS_STAGE2[k][ci.min(PE_NB_CBKS_STAGE2_EXT - 1)] as i32, min_lag),
                    max_lag,
                );
            }
        } else {
            for k in 0..nb_subfr {
                pitch_lags[k] = imin(
                    imax(lag + SILK_CB_LAGS_STAGE3[k][ci.min(PE_NB_CBKS_STAGE3_MAX - 1)] as i32, min_lag),
                    max_lag,
                );
            }
        }
    }
}
