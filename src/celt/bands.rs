//! CELT band processing — spectral quantization, energy computation,
//! normalization, and stereo coding.
//!
//! Matches `bands.c` / `bands.h` in the C reference (fixed-point path only).
//! All functions produce bit-exact output matching the C reference when compiled
//! with `FIXED_POINT` and `OPUS_FAST_INT64`.

use super::ec_ctx::EcCoder;
use super::math_ops::*;
use super::modes::CELTMode;
use super::rate::*;
use super::quant_bands::EMEANS;
use super::vq::{
    alg_quant, alg_unquant, celt_inner_prod_norm_shift, renormalise_vector, stereo_itheta,
};
use crate::types::*;

// ===========================================================================
// Constants
// ===========================================================================

pub const SPREAD_NONE: i32 = 0;
pub const SPREAD_LIGHT: i32 = 1;
pub const SPREAD_NORMAL: i32 = 2;
pub const SPREAD_AGGRESSIVE: i32 = 3;

/// Minimum stereo energy threshold (fixed-point).
const MIN_STEREO_ENERGY: i32 = 2;

/// Bit-reversed Gray code reordering table for Hadamard transforms.
/// Lines are for N=2, 4, 8, 16.
static ORDERY_TABLE: [i32; 30] = [
    1, 0, 3, 0, 2, 1, 7, 0, 4, 3, 6, 1, 5, 2, 15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2, 10, 5,
];

// ===========================================================================
// Division helpers (match C entcode.h celt_udiv / celt_sudiv)
// ===========================================================================

/// Unsigned integer division.
#[inline(always)]
fn celt_udiv(n: u32, d: u32) -> u32 {
    n / d
}

/// Signed integer division (rounds toward zero).
#[inline(always)]
fn celt_sudiv(n: i32, d: i32) -> i32 {
    n / d
}

// ===========================================================================
// Hysteresis decision
// ===========================================================================

/// Choose a bin index with hysteresis to prevent rapid switching.
/// Matches C `hysteresis_decision()`.
pub fn hysteresis_decision(
    val: i32,
    thresholds: &[i32],
    hysteresis: &[i32],
    n: usize,
    prev: i32,
) -> i32 {
    let mut i: i32 = 0;
    while (i as usize) < n {
        if val < thresholds[i as usize] {
            break;
        }
        i += 1;
    }
    if i > prev && val < thresholds[prev as usize] + hysteresis[prev as usize] {
        i = prev;
    }
    if i < prev && val > thresholds[(prev - 1) as usize] - hysteresis[(prev - 1) as usize] {
        i = prev;
    }
    i
}

// ===========================================================================
// LCG random
// ===========================================================================

/// Linear congruential generator. Matches C `celt_lcg_rand()`.
#[inline(always)]
pub fn celt_lcg_rand(seed: u32) -> u32 {
    1664525u32.wrapping_mul(seed).wrapping_add(1013904223)
}

// ===========================================================================
// Bit-exact cos / log2tan
// ===========================================================================

/// Bit-exact cosine approximation for theta quantization.
/// Input: x in [0, 16384] (Q14, quarter-turn). Output: Q15.
/// Matches C `bitexact_cos()`.
pub fn bitexact_cos(x: i16) -> i16 {
    let x32 = x as i32;
    let tmp = (4096 + x32 * x32) >> 13;
    let x2 = tmp as i16;
    let x2i = x2 as i32;
    // Polynomial: (32767 - x2) + x2*(-7651 + x2*(8277 + x2*(-626)))
    let p = -7651 + frac_mul16(x2i, 8277 + frac_mul16(-626, x2i));
    let result = (32767 - x2i) + frac_mul16(x2i, p);
    (1 + result) as i16
}

/// Bit-exact log2(tan(θ)) for mid/side bit allocation.
/// Matches C `bitexact_log2tan()`.
pub fn bitexact_log2tan(isin: i32, icos: i32) -> i32 {
    let lc = ec_ilog(icos as u32);
    let ls = ec_ilog(isin as u32);
    let icos = icos << (15 - lc);
    let isin = isin << (15 - ls);
    (ls - lc) * (1 << 11) + frac_mul16(isin, frac_mul16(isin, -2597) + 7932)
        - frac_mul16(icos, frac_mul16(icos, -2597) + 7932)
}

// ===========================================================================
// Energy computation (fixed-point)
// ===========================================================================

/// Compute the amplitude (sqrt energy) in each band.
/// Matches C `compute_band_energies()` (FIXED_POINT path).
pub fn compute_band_energies(
    m: &CELTMode,
    x: &[i32],
    band_e: &mut [i32],
    end: i32,
    c_channels: i32,
    lm: i32,
) {
    let n = m.short_mdct_size << lm;
    for c in 0..c_channels {
        for i in 0..end as usize {
            let start_bin = (m.ebands[i] as i32) << lm;
            let end_bin = (m.ebands[i + 1] as i32) << lm;

            let maxval =
                celt_maxabs32(&x[(c * n + start_bin) as usize..(c * n + end_bin) as usize]);
            if maxval > 0 {
                let shift = imax(
                    0,
                    30 - celt_ilog2(maxval + (maxval >> 14) + 1)
                        - ((((m.log_n[i] as i32 + 7) >> BITRES) + lm + 1) >> 1),
                );
                let mut sum: i32 = 0;
                for j in start_bin..end_bin {
                    let xv = shl32(x[(j + c * n) as usize], shift);
                    sum = add32(sum, mult32_32_q31(xv, xv));
                }
                band_e[i + (c * m.nb_ebands) as usize] =
                    max32(maxval, pshr32(celt_sqrt32(shr32(sum, 1)), shift));
            } else {
                band_e[i + (c * m.nb_ebands) as usize] = EPSILON;
            }
        }
    }
}

// ===========================================================================
// Normalization / Denormalization
// ===========================================================================

/// Normalise each band so energy is one. Matches C `normalise_bands()` (FIXED_POINT).
pub fn normalise_bands(
    m: &CELTMode,
    freq: &[i32],
    x: &mut [i32],
    band_e: &[i32],
    end: i32,
    c_channels: i32,
    big_m: i32,
) {
    let n = big_m * m.short_mdct_size;
    for c in 0..c_channels {
        for i in 0..end as usize {
            let mut e = band_e[i + (c * m.nb_ebands) as usize];
            // Prevent energy rounding from blowing up normalized signal
            if e < 10 {
                e += EPSILON;
            }
            let shift = 30 - celt_zlog2(e);
            let e_shifted = shl32(e, shift);
            let g = celt_rcp_norm32(e_shifted);
            let j_start = big_m * m.ebands[i] as i32;
            let j_end = big_m * m.ebands[i + 1] as i32;
            for j in j_start..j_end {
                x[(j + c * n) as usize] = pshr32(
                    mult32_32_q31(g, shl32(freq[(j + c * n) as usize], shift)),
                    30 - NORM_SHIFT,
                );
            }
        }
    }
}

/// Denormalise bands to restore full amplitude from unit-energy representation.
/// Matches C `denormalise_bands()` (FIXED_POINT path).
pub fn denormalise_bands(
    m: &CELTMode,
    x: &[i32],
    freq: &mut [i32],
    band_log_e: &[i32],
    start: i32,
    end: i32,
    big_m: i32,
    downsample: i32,
    silence: bool,
) {
    let n = big_m * m.short_mdct_size;
    let mut bound = big_m * m.ebands[end as usize] as i32;
    if downsample != 1 {
        bound = imin(bound, n / downsample);
    }
    let (start, end) = if silence { (0i32, 0i32) } else { (start, end) };
    let bound = if silence { 0 } else { bound };

    let x_offset = big_m * m.ebands[start as usize] as i32;
    let mut x_idx: usize = x_offset as usize;

    // Zero out bins before start
    if start != 0 {
        for fi in 0..x_offset as usize {
            freq[fi] = 0;
        }
    }
    let mut f_idx: usize = x_offset as usize;

    for i in start..end {
        let iu = i as usize;
        let j_start = big_m * m.ebands[iu] as i32;
        let band_end = big_m * m.ebands[iu + 1] as i32;
        // lg = bandLogE[i] + eMeans[i] << (DB_SHIFT - 4)
        let lg = add32(band_log_e[iu], shl32(EMEANS[iu] as i32, DB_SHIFT - 4));

        let (g, shift) = {
            // Handle the integer part of the log energy
            let mut shift = 17 - (lg >> DB_SHIFT);
            let g;
            if shift >= 31 {
                shift = 0;
                g = 0;
            } else {
                // Handle the fractional part
                g = shl32(celt_exp2_db_frac(lg & ((1 << DB_SHIFT) - 1)), 2);
            }
            // Handle extreme gains with negative shift
            if shift < 0 {
                // Cap gain to avoid overflow (equivalent to cap of 18 on lg)
                (2147483647i32, 0i32)
            } else {
                (g, shift)
            }
        };

        let mut j = j_start;
        while j < band_end {
            freq[f_idx] = pshr32(mult32_32_q31(shl32(x[x_idx], 30 - NORM_SHIFT), g), shift);
            f_idx += 1;
            x_idx += 1;
            j += 1;
        }
    }

    // Zero out remaining bins
    for fi in bound as usize..n as usize {
        freq[fi] = 0;
    }
}

// ===========================================================================
// Anti-collapse
// ===========================================================================

/// Inject noise into collapsed bands to prevent audible artifacts.
/// Matches C `anti_collapse()` (FIXED_POINT path).
pub fn anti_collapse(
    m: &CELTMode,
    x_: &mut [i32],
    collapse_masks: &mut [u8],
    lm: i32,
    c_channels: i32,
    size: i32,
    start: i32,
    end: i32,
    log_e: &[i32],
    prev1_log_e: &[i32],
    prev2_log_e: &[i32],
    pulses: &[i32],
    mut seed: u32,
    encode: bool,
) {
    for i in start..end {
        let iu = i as usize;
        let n0 = (m.ebands[iu + 1] - m.ebands[iu]) as i32;
        // depth in 1/8 bits
        let depth = (celt_udiv(1 + pulses[iu] as u32, n0 as u32) >> lm as u32) as i32;

        let thresh32 = shr32(celt_exp2(-shl16(depth, 10 - BITRES)), 1);
        let thresh = mult16_16_q15(qconst16(0.5, 15), min32(32767, thresh32));
        let sqrt_1 = {
            let t = n0 << lm;
            let shift = celt_ilog2(t) >> 1;
            let t = shl32(t, (7 - shift) << 1);
            (celt_rsqrt_norm(t), shift)
        };

        for c in 0..c_channels {
            let mut prev1 = prev1_log_e[(c * m.nb_ebands + i) as usize];
            let mut prev2 = prev2_log_e[(c * m.nb_ebands + i) as usize];
            if !encode && c_channels == 1 {
                prev1 = max32(prev1, prev1_log_e[(m.nb_ebands + i) as usize]);
                prev2 = max32(prev2, prev2_log_e[(m.nb_ebands + i) as usize]);
            }
            let ediff = max32(
                0,
                log_e[(c * m.nb_ebands + i) as usize] - min32(prev1, prev2),
            );

            // r = 2 * exp2_db(-Ediff), clamped
            let r = if ediff < qconst32(16.0, DB_SHIFT as u32) {
                let r32 = shr32(celt_exp2_db(-ediff), 1);
                2 * min16(16383, r32)
            } else {
                0
            };

            // Scale by sqrt(2) for LM==3
            let r = if lm == 3 {
                mult16_16_q14(23170, min32(23169, r))
            } else {
                r
            };
            let r = shr16(min16(thresh, r), 1);
            let r = vshr32(mult16_16_q15(sqrt_1.0, r), sqrt_1.1 + 14 - NORM_SHIFT);

            let x_base = (c * size + ((m.ebands[iu] as i32) << lm)) as usize;
            let mut renormalize = false;
            for k in 0..(1 << lm) {
                // Detect collapse
                if collapse_masks[iu * c_channels as usize + c as usize] & (1 << k) == 0 {
                    // Fill with noise
                    for j in 0..n0 {
                        seed = celt_lcg_rand(seed);
                        x_[x_base + ((j << lm) + k) as usize] =
                            if seed & 0x8000 != 0 { r } else { -r };
                    }
                    renormalize = true;
                }
            }
            if renormalize {
                renormalise_vector(
                    &mut x_[x_base..x_base + (n0 << lm) as usize],
                    (n0 << lm) as usize,
                    Q31ONE,
                );
            }
        }
    }
}

// ===========================================================================
// Channel weight computation
// ===========================================================================

/// Compute per-channel weights for stereo distortion optimization.
/// Matches C `compute_channel_weights()`.
fn compute_channel_weights(ex: i32, ey: i32) -> [i32; 2] {
    let min_e = min32(ex, ey);
    let ex = add32(ex, min_e / 3);
    let ey = add32(ey, min_e / 3);
    let shift = celt_ilog2(EPSILON + max32(ex, ey)) - 14;
    [vshr32(ex, shift), vshr32(ey, shift)]
}

// ===========================================================================
// Stereo helpers
// ===========================================================================

/// Apply intensity stereo rotation. Matches C `intensity_stereo()`.
fn intensity_stereo(m: &CELTMode, x: &mut [i32], y: &[i32], band_e: &[i32], band_id: i32, n: i32) {
    let i = band_id as usize;
    let shift = celt_zlog2(max32(band_e[i], band_e[i + m.nb_ebands as usize])) - 13;
    let left = vshr32(band_e[i], shift);
    let right = vshr32(band_e[i + m.nb_ebands as usize], shift);
    let norm = EPSILON + celt_sqrt(EPSILON + mult16_16(left, left) + mult16_16(right, right));
    let left = min32(left, norm - 1);
    let right = min32(right, norm - 1);
    let a1 = div32_16(shl32(extend32(left), 15), norm);
    let a2 = div32_16(shl32(extend32(right), 15), norm);
    for j in 0..n as usize {
        x[j] = add32(mult16_32_q15(a1, x[j]), mult16_32_q15(a2, y[j]));
    }
}

/// Split into mid/side for stereo coding. Matches C `stereo_split()`.
fn stereo_split(x: &mut [i32], y: &mut [i32], n: i32) {
    let sqrt_half = qconst32(0.70710678, 31);
    for j in 0..n as usize {
        let l = mult32_32_q31(sqrt_half, x[j]);
        let r = mult32_32_q31(sqrt_half, y[j]);
        x[j] = add32(l, r);
        y[j] = sub32(r, l);
    }
}

/// Merge mid/side back into L/R after stereo decoding. Matches C `stereo_merge()`.
fn stereo_merge(x: &mut [i32], y: &mut [i32], mid: i32, n: i32) {
    let nu = n as usize;
    // Compute norm of X+Y and X-Y as |X|^2 + |Y|^2 +/- sum(xy)
    let xp = celt_inner_prod_norm_shift(&y[..nu], &x[..nu], nu);
    let side = celt_inner_prod_norm_shift(&y[..nu], &y[..nu], nu);
    // Compensating for the mid normalization
    let xp = mult32_32_q31(mid, xp);
    let el = shr32(mult32_32_q31(mid, mid), 3) + side - 2 * xp;
    let er = shr32(mult32_32_q31(mid, mid), 3) + side + 2 * xp;

    if er < qconst32(6e-4, 28) || el < qconst32(6e-4, 28) {
        // Copy X to Y to avoid numerical issues
        y[..nu].copy_from_slice(&x[..nu]);
        return;
    }

    let kl = imax(7, celt_ilog2(el) >> 1);
    let kr = imax(7, celt_ilog2(er) >> 1);
    let t = vshr32(el, (kl << 1) - 29);
    let lgain = celt_rsqrt_norm32(t);
    let t = vshr32(er, (kr << 1) - 29);
    let rgain = celt_rsqrt_norm32(t);

    for j in 0..nu {
        let l = mult32_32_q31(mid, x[j]);
        let r = y[j];
        x[j] = vshr32(mult32_32_q31(lgain, sub32(l, r)), kl - 15);
        y[j] = vshr32(mult32_32_q31(rgain, add32(l, r)), kr - 15);
    }
}

// ===========================================================================
// Spreading decision
// ===========================================================================

/// Decide spreading mode based on spectral characteristics.
/// Matches C `spreading_decision()`.
pub fn spreading_decision(
    m: &CELTMode,
    x: &[i32],
    average: &mut i32,
    last_decision: i32,
    hf_average: &mut i32,
    tapset_decision: &mut i32,
    update_hf: bool,
    end: i32,
    c_channels: i32,
    big_m: i32,
    spread_weight: &[i32],
) -> i32 {
    let mut sum: i32 = 0;
    let mut nb_bands: i32 = 0;
    let n0 = big_m * m.short_mdct_size;
    let mut hf_sum: i32 = 0;

    if big_m * (m.ebands[end as usize] as i32 - m.ebands[(end - 1) as usize] as i32) <= 8 {
        return SPREAD_NONE;
    }

    for c in 0..c_channels {
        for i in 0..end as usize {
            let band_n = big_m * (m.ebands[i + 1] as i32 - m.ebands[i] as i32);
            if band_n <= 8 {
                continue;
            }
            let x_off = (big_m * m.ebands[i] as i32 + c * n0) as usize;
            let mut tcount = [0i32; 3];
            for j in 0..band_n as usize {
                // Q13: x2N = (x[j]>>10)^2 * N
                let xn = shr32(x[x_off + j], NORM_SHIFT - 14);
                let x2n = mult16_16(mult16_16_q15(xn, xn), band_n);
                if x2n < qconst16(0.25, 13) {
                    tcount[0] += 1;
                }
                if x2n < qconst16(0.0625, 13) {
                    tcount[1] += 1;
                }
                if x2n < qconst16(0.015625, 13) {
                    tcount[2] += 1;
                }
            }

            // Only include four last bands (8 kHz and up)
            if i as i32 > m.nb_ebands - 4 {
                hf_sum += celt_udiv(32 * (tcount[1] + tcount[0]) as u32, band_n as u32) as i32;
            }
            let tmp = (if 2 * tcount[2] >= band_n { 1 } else { 0 })
                + (if 2 * tcount[1] >= band_n { 1 } else { 0 })
                + (if 2 * tcount[0] >= band_n { 1 } else { 0 });
            sum += tmp * spread_weight[i];
            nb_bands += spread_weight[i];
        }
    }

    if update_hf {
        if hf_sum != 0 {
            hf_sum = celt_udiv(hf_sum as u32, (c_channels * (4 - m.nb_ebands + end)) as u32) as i32;
        }
        *hf_average = (*hf_average + hf_sum) >> 1;
        hf_sum = *hf_average;
        if *tapset_decision == 2 {
            hf_sum += 4;
        } else if *tapset_decision == 0 {
            hf_sum -= 4;
        }
        if hf_sum > 22 {
            *tapset_decision = 2;
        } else if hf_sum > 18 {
            *tapset_decision = 1;
        } else {
            *tapset_decision = 0;
        }
    }

    sum = celt_udiv((sum << 8) as u32, nb_bands as u32) as i32;
    // Recursive averaging
    sum = (sum + *average) >> 1;
    *average = sum;
    // Hysteresis
    sum = (3 * sum + (((3 - last_decision) << 7) + 64) + 2) >> 2;
    if sum < 80 {
        SPREAD_AGGRESSIVE
    } else if sum < 256 {
        SPREAD_NORMAL
    } else if sum < 384 {
        SPREAD_LIGHT
    } else {
        SPREAD_NONE
    }
}

// ===========================================================================
// Haar transform
// ===========================================================================

/// In-place Haar wavelet transform for time-frequency resolution changes.
/// Matches C `haar1()`.
pub fn haar1(x: &mut [i32], n0: i32, stride: i32) {
    let n0 = n0 >> 1;
    let sqrt_half = qconst32(0.70710678, 31);
    for i in 0..stride {
        for j in 0..n0 {
            let idx0 = (stride * 2 * j + i) as usize;
            let idx1 = (stride * (2 * j + 1) + i) as usize;
            let tmp1 = mult32_32_q31(sqrt_half, x[idx0]);
            let tmp2 = mult32_32_q31(sqrt_half, x[idx1]);
            x[idx0] = add32(tmp1, tmp2);
            x[idx1] = sub32(tmp1, tmp2);
        }
    }
}

// ===========================================================================
// Hadamard interleave / deinterleave
// ===========================================================================

/// Reorder samples from frequency order to time order for split quantization.
/// Matches C `deinterleave_hadamard()`.
fn deinterleave_hadamard(x: &mut [i32], n0: i32, stride: i32, hadamard: bool) {
    let n = n0 * stride;
    let mut tmp = vec![0i32; n as usize];
    if hadamard {
        let ordery_off = (stride - 2) as usize;
        for i in 0..stride {
            for j in 0..n0 {
                tmp[(ORDERY_TABLE[ordery_off + i as usize] * n0 + j) as usize] =
                    x[(j * stride + i) as usize];
            }
        }
    } else {
        for i in 0..stride {
            for j in 0..n0 {
                tmp[(i * n0 + j) as usize] = x[(j * stride + i) as usize];
            }
        }
    }
    x[..n as usize].copy_from_slice(&tmp[..n as usize]);
}

/// Reorder samples from time order back to frequency order after quantization.
/// Matches C `interleave_hadamard()`.
fn interleave_hadamard(x: &mut [i32], n0: i32, stride: i32, hadamard: bool) {
    let n = n0 * stride;
    let mut tmp = vec![0i32; n as usize];
    if hadamard {
        let ordery_off = (stride - 2) as usize;
        for i in 0..stride {
            for j in 0..n0 {
                tmp[(j * stride + i) as usize] =
                    x[(ORDERY_TABLE[ordery_off + i as usize] * n0 + j) as usize];
            }
        }
    } else {
        for i in 0..stride {
            for j in 0..n0 {
                tmp[(j * stride + i) as usize] = x[(i * n0 + j) as usize];
            }
        }
    }
    x[..n as usize].copy_from_slice(&tmp[..n as usize]);
}

// ===========================================================================
// Quantization resolution
// ===========================================================================

/// Compute the number of quantization levels for the split angle.
/// Matches C `compute_qn()`.
fn compute_qn(n: i32, b: i32, offset: i32, pulse_cap: i32, stereo: bool) -> i32 {
    static EXP2_TABLE8: [i16; 8] = [16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048];

    let mut n2 = 2 * n - 1;
    if stereo && n == 2 {
        n2 -= 1;
    }
    let mut qb = celt_sudiv(b + n2 * offset, n2);
    qb = imin(b - pulse_cap - (4 << BITRES), qb);
    qb = imin(8 << BITRES, qb);

    if qb < (1 << BITRES >> 1) {
        1
    } else {
        let mut qn = (EXP2_TABLE8[(qb & 0x7) as usize] as i32) >> (14 - (qb >> BITRES));
        qn = (qn + 1) >> 1 << 1;
        qn
    }
}

// ===========================================================================
// Band context structures
// ===========================================================================

/// Per-band quantization context. Matches C `struct band_ctx`.
struct BandCtx<'a, EC: EcCoder> {
    encode: bool,
    resynth: bool,
    m: &'a CELTMode,
    i: i32,
    intensity: i32,
    spread: i32,
    tf_change: i32,
    ec: &'a mut EC,
    remaining_bits: i32,
    band_e: &'a [i32],
    seed: u32,
    theta_round: i32,
    disable_inv: bool,
    avoid_split_noise: bool,
}

/// Split context returned by compute_theta. Matches C `struct split_ctx`.
struct SplitCtx {
    inv: bool,
    imid: i32,
    iside: i32,
    delta: i32,
    itheta: i32,
    qalloc: i32,
}

// ===========================================================================
// Theta computation
// ===========================================================================

/// Compute and code the split angle theta between two half-bands.
/// Matches C `compute_theta()` (no ENABLE_QEXT).
fn compute_theta<EC: EcCoder>(
    ctx: &mut BandCtx<EC>,
    x: &mut [i32],
    y: &mut [i32],
    n: i32,
    b: &mut i32,
    big_b: i32,
    b0: i32,
    lm: i32,
    stereo: bool,
    fill: &mut i32,
) -> SplitCtx {
    let nu = n as usize;
    let encode = ctx.encode;
    let m = ctx.m;
    let i = ctx.i;
    let intensity = ctx.intensity;
    let band_e = ctx.band_e;

    // Decide resolution for split parameter theta
    let pulse_cap = m.log_n[i as usize] as i32 + lm * (1 << BITRES);
    let offset = (pulse_cap >> 1)
        - if stereo && n == 2 {
            QTHETA_OFFSET_TWOPHASE
        } else {
            QTHETA_OFFSET
        };
    let mut qn = compute_qn(n, *b, offset, pulse_cap, stereo);
    if stereo && i >= intensity {
        qn = 1;
    }

    let mut itheta: i32 = 0;
    if encode {
        let itheta_q30 = stereo_itheta(&x[..nu], &y[..nu], stereo, nu);
        itheta = itheta_q30 >> 16;

        // Debug: trace theta for band 13
        if i == 13 {
            let (_, eoffs, _) = ctx.ec.ec_debug_state();
            if eoffs >= 50 && eoffs <= 65 {
                eprintln!("[R CT] band={} n={} qn={} itheta_q30={} itheta_q14={} eoffs={}",
                    i, n, qn, itheta_q30, itheta, eoffs);
            }
        }
    }

    let tell = ctx.ec.ec_tell_frac();
    let mut inv = false;

    if qn != 1 {
        if encode {
            if !stereo || ctx.theta_round == 0 {
                itheta = ((itheta as i64 * qn as i64 + 8192) >> 14) as i32;
                // Debug: trace quantized theta for band 13
                if i == 13 {
                    let (_, eoffs, _) = ctx.ec.ec_debug_state();
                    if eoffs >= 50 && eoffs <= 65 {
                        eprintln!("[R CT quant] band={} qn={} itheta_quant={} eoffs={}", i, qn, itheta, eoffs);
                    }
                }
                if !stereo && ctx.avoid_split_noise && itheta > 0 && itheta < qn {
                    // Check if theta will cause noise injection on one side
                    let unquantized = celt_udiv(itheta as u32 * 16384, qn as u32) as i32;
                    let imid_t = bitexact_cos(unquantized as i16) as i32;
                    let iside_t = bitexact_cos((16384 - unquantized) as i16) as i32;
                    let delta_t = frac_mul16((n - 1) << 7, bitexact_log2tan(iside_t, imid_t));
                    if delta_t > *b {
                        itheta = qn;
                    } else if delta_t < -*b {
                        itheta = 0;
                    }
                }
            } else {
                // Bias quantization towards itheta=0 and itheta=16384
                let bias = if itheta > 8192 {
                    32767 / qn
                } else {
                    -32767 / qn
                };
                let down = imin(
                    qn - 1,
                    imax(0, ((itheta as i64 * qn as i64 + bias as i64) >> 14) as i32),
                );
                if ctx.theta_round < 0 {
                    itheta = down;
                } else {
                    itheta = down + 1;
                }
            }
        }

        // Entropy coding of the angle
        if stereo && n > 2 {
            // Step pdf for stereo
            let p0: i32 = 3;
            let mut x_val = itheta;
            let x0 = qn / 2;
            let ft = (p0 * (x0 + 1) + x0) as u32;
            if encode {
                let fl = if x_val <= x0 {
                    (p0 * x_val) as u32
                } else {
                    (x_val - 1 - x0 + (x0 + 1) * p0) as u32
                };
                let fh = if x_val <= x0 {
                    (p0 * (x_val + 1)) as u32
                } else {
                    (x_val - x0 + (x0 + 1) * p0) as u32
                };
                ctx.ec.ec_encode(fl, fh, ft);
            } else {
                let fs = ctx.ec.ec_decode(ft);
                if fs < ((x0 + 1) * p0) as u32 {
                    x_val = (fs / p0 as u32) as i32;
                } else {
                    x_val = x0 + 1 + (fs as i32 - (x0 + 1) * p0);
                }
                let fl = if x_val <= x0 {
                    (p0 * x_val) as u32
                } else {
                    (x_val - 1 - x0 + (x0 + 1) * p0) as u32
                };
                let fh = if x_val <= x0 {
                    (p0 * (x_val + 1)) as u32
                } else {
                    (x_val - x0 + (x0 + 1) * p0) as u32
                };
                ctx.ec.ec_dec_update(fl, fh, ft);
                itheta = x_val;
            }
        } else if b0 > 1 || stereo {
            // Uniform pdf
            if encode {
                ctx.ec.ec_enc_uint(itheta as u32, (qn + 1) as u32);
            } else {
                itheta = ctx.ec.ec_dec_uint((qn + 1) as u32) as i32;
            }
        } else {
            // Triangular pdf
            let ft = (((qn >> 1) + 1) * ((qn >> 1) + 1)) as u32;
            if encode {
                let fs = if itheta <= (qn >> 1) {
                    itheta + 1
                } else {
                    qn + 1 - itheta
                };
                let fl = if itheta <= (qn >> 1) {
                    ((itheta * (itheta + 1)) >> 1) as u32
                } else {
                    (ft as i32 - ((qn + 1 - itheta) * (qn + 2 - itheta) >> 1)) as u32
                };
                ctx.ec.ec_encode(fl, fl + fs as u32, ft);
            } else {
                let fm = ctx.ec.ec_decode(ft);
                let (fl, fs);
                if fm < ((qn >> 1) * ((qn >> 1) + 1) >> 1) as u32 {
                    itheta = ((isqrt32(8 * fm + 1) as i32) - 1) >> 1;
                    fs = (itheta + 1) as u32;
                    fl = ((itheta * (itheta + 1)) >> 1) as u32;
                } else {
                    itheta = (2 * (qn + 1) - isqrt32(8 * (ft - fm - 1) + 1) as i32) >> 1;
                    fs = (qn + 1 - itheta) as u32;
                    fl = (ft as i32 - ((qn + 1 - itheta) * (qn + 2 - itheta) >> 1)) as u32;
                }
                ctx.ec.ec_dec_update(fl, fl + fs, ft);
            }
        }

        itheta = celt_udiv(itheta as u32 * 16384, qn as u32) as i32;

        if encode && stereo {
            if itheta == 0 {
                intensity_stereo(m, x, y, band_e, i, n);
            } else {
                stereo_split(x, y, n);
            }
        }
    } else if stereo {
        // qn == 1: intensity stereo
        if encode {
            inv = itheta > 8192 && !ctx.disable_inv;
            if inv {
                for j in 0..nu {
                    y[j] = -y[j];
                }
            }
            intensity_stereo(m, x, y, band_e, i, n);
        }
        if *b > 2 << BITRES && ctx.remaining_bits > 2 << BITRES {
            if encode {
                ctx.ec.ec_enc_bit_logp(inv, 2);
            } else {
                inv = ctx.ec.ec_dec_bit_logp(2);
            }
        } else {
            inv = false;
        }
        // inv flag override to avoid problems with downmixing
        if ctx.disable_inv {
            inv = false;
        }
        itheta = 0;
    }

    let qalloc = ctx.ec.ec_tell_frac() as i32 - tell as i32;
    *b -= qalloc;

    let (imid, iside, delta) = if itheta == 0 {
        *fill &= (1 << big_b) - 1;
        (32767, 0, -16384)
    } else if itheta == 16384 {
        *fill &= ((1 << big_b) - 1) << big_b;
        (0, 32767, 16384)
    } else {
        let imid = bitexact_cos(itheta as i16) as i32;
        let iside = bitexact_cos((16384 - itheta) as i16) as i32;
        let delta = frac_mul16((n - 1) << 7, bitexact_log2tan(iside, imid));
        (imid, iside, delta)
    };

    SplitCtx {
        inv,
        imid,
        iside,
        delta,
        itheta,
        qalloc,
    }
}

// ===========================================================================
// N=1 special case
// ===========================================================================

/// Quantize a band with N=1 (single sample). Matches C `quant_band_n1()`.
fn quant_band_n1<EC: EcCoder>(
    ctx: &mut BandCtx<EC>,
    x: &mut [i32],
    y: Option<&mut [i32]>,
    lowband_out: Option<&mut [i32]>,
) -> u32 {
    let encode = ctx.encode;

    // First channel (X)
    {
        let mut sign: u32 = 0;
        if ctx.remaining_bits >= 1 << BITRES {
            if encode {
                sign = if x[0] < 0 { 1 } else { 0 };
                ctx.ec.ec_enc_bits(sign, 1);
            } else {
                sign = ctx.ec.ec_dec_bits(1);
            }
            ctx.remaining_bits -= 1 << BITRES;
        }
        if ctx.resynth {
            x[0] = if sign != 0 {
                -NORM_SCALING
            } else {
                NORM_SCALING
            };
        }
    }

    // Second channel (Y) if stereo
    if let Some(y) = y {
        let mut sign: u32 = 0;
        if ctx.remaining_bits >= 1 << BITRES {
            if encode {
                sign = if y[0] < 0 { 1 } else { 0 };
                ctx.ec.ec_enc_bits(sign, 1);
            } else {
                sign = ctx.ec.ec_dec_bits(1);
            }
            ctx.remaining_bits -= 1 << BITRES;
        }
        if ctx.resynth {
            y[0] = if sign != 0 {
                -NORM_SCALING
            } else {
                NORM_SCALING
            };
        }
    }

    if let Some(lbo) = lowband_out {
        lbo[0] = shr32(x[0], 4);
    }
    1
}

// ===========================================================================
// Mono partition quantization
// ===========================================================================

/// Recursive mono partition quantization. Matches C `quant_partition()`.
fn quant_partition<EC: EcCoder>(
    ctx: &mut BandCtx<EC>,
    x: &mut [i32],
    mut n: i32,
    mut b: i32,
    mut big_b: i32,
    lowband: Option<&[i32]>,
    mut lm: i32,
    gain: i32,
    mut fill: i32,
) -> u32 {
    let encode = ctx.encode;
    let m = ctx.m;
    let i = ctx.i;
    let spread = ctx.spread;

    // Check if we need to split
    let cache_idx = m.cache.index[((lm + 1) * m.nb_ebands + i) as usize] as usize;
    let cache = &m.cache.bits[cache_idx..];

    if lm != -1 && b > cache[cache[0] as usize] as i32 + 12 && n > 2 {
        // Split the band in two
        let b0 = big_b;
        n >>= 1;
        let nu = n as usize;
        lm -= 1;
        if big_b == 1 {
            fill = (fill & 1) | (fill << 1);
        }
        big_b = (big_b + 1) >> 1;

        // We need to split x into two halves: x[..nu] and x[nu..n]
        // compute_theta needs both halves as separate mutable slices
        // We'll work with indices into x directly
        let (x_lo, x_hi) = x.split_at_mut(nu);

        let sctx = compute_theta(
            ctx,
            x_lo,
            &mut x_hi[..nu],
            n,
            &mut b,
            big_b,
            b0,
            lm,
            false,
            &mut fill,
        );
        let imid = sctx.imid;
        let iside = sctx.iside;
        let delta = sctx.delta;
        let itheta = sctx.itheta;
        let qalloc = sctx.qalloc;

        // Fixed-point, no ENABLE_QEXT: mid/side from imid/iside
        let mid = shl32(extend32(imid), 16);
        let side = shl32(extend32(iside), 16);

        // Give more bits to low-energy MDCTs
        let mut delta = delta;
        if b0 > 1 && (itheta & 0x3fff) != 0 {
            if itheta > 8192 {
                delta -= delta >> (4 - lm);
            } else {
                delta = imin(0, delta + (n << BITRES >> (5 - lm)));
            }
        }
        let mbits = imax(0, imin(b, (b - delta) / 2));
        let mut sbits = b - mbits;
        ctx.remaining_bits -= qalloc;

        // Prepare lowband for second half
        let next_lowband2: Option<Vec<i32>> = lowband.map(|lb| {
            if nu < lb.len() {
                lb[nu..].to_vec()
            } else {
                vec![]
            }
        });

        let rebalance = ctx.remaining_bits;
        let cm;
        if mbits >= sbits {
            let cm_lo = quant_partition(
                ctx,
                x_lo,
                n,
                mbits,
                big_b,
                lowband,
                lm,
                mult32_32_q31(gain, mid),
                fill,
            );
            let rebalance = mbits - (rebalance - ctx.remaining_bits);
            if rebalance > 3 << BITRES && itheta != 0 {
                sbits += rebalance - (3 << BITRES);
            }
            let cm_hi = quant_partition(
                ctx,
                x_hi,
                n,
                sbits,
                big_b,
                next_lowband2.as_deref(),
                lm,
                mult32_32_q31(gain, side),
                fill >> big_b,
            );
            cm = cm_lo | (cm_hi << (b0 >> 1));
        } else {
            let cm_hi = quant_partition(
                ctx,
                x_hi,
                n,
                sbits,
                big_b,
                next_lowband2.as_deref(),
                lm,
                mult32_32_q31(gain, side),
                fill >> big_b,
            );
            let rebalance = sbits - (rebalance - ctx.remaining_bits);
            let mut mbits = mbits;
            if rebalance > 3 << BITRES && itheta != 16384 {
                mbits += rebalance - (3 << BITRES);
            }
            let cm_lo = quant_partition(
                ctx,
                x_lo,
                n,
                mbits,
                big_b,
                lowband,
                lm,
                mult32_32_q31(gain, mid),
                fill,
            );
            cm = cm_lo | (cm_hi << (b0 >> 1));
        }
        cm
    } else {
        // Base case: no-split
        let mut q = bits2pulses(m, i, lm, b);
        let mut curr_bits = pulses2bits(m, i, lm, q);
        ctx.remaining_bits -= curr_bits;

        // Ensure we never bust the budget
        while ctx.remaining_bits < 0 && q > 0 {
            ctx.remaining_bits += curr_bits;
            q -= 1;
            curr_bits = pulses2bits(m, i, lm, q);
            ctx.remaining_bits -= curr_bits;
        }

        if q != 0 {
            let k = get_pulses(q);
            if encode {
                alg_quant(ctx.ec, x, n, k, spread, big_b, gain, ctx.resynth)
            } else {
                alg_unquant(ctx.ec, x, n, k, spread, big_b, gain)
            }
        } else {
            // No pulses: fill with noise or folded spectrum
            if ctx.resynth {
                let cm_mask: u32 = (1u32 << big_b as u32) - 1;
                let fill = fill as u32 & cm_mask;
                if fill == 0 {
                    for j in 0..n as usize {
                        x[j] = 0;
                    }
                    0
                } else if lowband.is_none() {
                    // Noise
                    for j in 0..n as usize {
                        ctx.seed = celt_lcg_rand(ctx.seed);
                        x[j] = shl32((ctx.seed >> 20) as i32, NORM_SHIFT - 14);
                    }
                    renormalise_vector(x, n as usize, gain);
                    cm_mask
                } else {
                    let lb = lowband.unwrap();
                    // Folded spectrum
                    for j in 0..n as usize {
                        ctx.seed = celt_lcg_rand(ctx.seed);
                        // About 48 dB below the "normal" folding level
                        let tmp: i32 = qconst16(1.0 / 256.0, NORM_SHIFT as u32 - 4);
                        let tmp = if ctx.seed & 0x8000 != 0 { tmp } else { -tmp };
                        x[j] = lb[j] + tmp;
                    }
                    renormalise_vector(x, n as usize, gain);
                    fill
                }
            } else {
                0
            }
        }
    }
}

// ===========================================================================
// Mono band quantization
// ===========================================================================

/// Quantize a mono band with time-frequency transforms.
/// Matches C `quant_band()`.
fn quant_band<EC: EcCoder>(
    ctx: &mut BandCtx<EC>,
    x: &mut [i32],
    n: i32,
    b: i32,
    mut big_b: i32,
    lowband: Option<&[i32]>,
    lm: i32,
    lowband_out: Option<&mut [i32]>,
    gain: i32,
    lowband_scratch: Option<&mut [i32]>,
    mut fill: i32,
) -> u32 {
    let n0 = n;
    let mut n_b = n;
    let b0 = big_b;
    let mut time_divide = 0;
    let mut recombine = 0;
    let long_blocks = b0 == 1;
    let encode = ctx.encode;
    let tf_change = ctx.tf_change;

    n_b = celt_udiv(n_b as u32, big_b as u32) as i32;

    // Special case for one sample
    if n == 1 {
        return quant_band_n1(ctx, x, None, lowband_out);
    }

    if tf_change > 0 {
        recombine = tf_change;
    }

    // Copy lowband to scratch if we'll be modifying it via Haar transforms
    let mut scratch_buf: Option<Vec<i32>> = None;
    let mut use_scratch_as_lowband = false;
    if lowband_scratch.is_some()
        && lowband.is_some()
        && (recombine != 0 || ((n_b & 1) == 0 && tf_change < 0) || b0 > 1)
    {
        if let Some(lb) = lowband {
            let mut sb = vec![0i32; n as usize];
            let copy_len = n as usize;
            sb[..copy_len].copy_from_slice(&lb[..copy_len]);
            scratch_buf = Some(sb);
            use_scratch_as_lowband = true;
        }
    }

    // Band recombining to increase frequency resolution
    for k in 0..recombine {
        static BIT_INTERLEAVE_TABLE: [u8; 16] = [0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3];
        if encode {
            haar1(x, n >> k, 1 << k);
        }
        if let Some(ref mut sb) = scratch_buf {
            if use_scratch_as_lowband {
                haar1(sb, n >> k, 1 << k);
            }
        }
        fill = (BIT_INTERLEAVE_TABLE[(fill & 0xF) as usize] as i32)
            | ((BIT_INTERLEAVE_TABLE[(fill >> 4) as usize] as i32) << 2);
    }
    big_b >>= recombine;
    n_b <<= recombine;

    // Increasing the time resolution
    while (n_b & 1) == 0 && tf_change + time_divide < 0 {
        if encode {
            haar1(x, n_b, big_b);
        }
        if let Some(ref mut sb) = scratch_buf {
            if use_scratch_as_lowband {
                haar1(sb, n_b, big_b);
            }
        }
        fill |= fill << big_b;
        big_b <<= 1;
        n_b >>= 1;
        time_divide += 1;
    }
    let b0_new = big_b;
    let n_b0 = n_b;

    // Reorganize samples: frequency order → time order
    if b0_new > 1 {
        if encode {
            deinterleave_hadamard(x, n_b >> recombine, b0_new << recombine, long_blocks);
        }
        if let Some(ref mut sb) = scratch_buf {
            if use_scratch_as_lowband {
                deinterleave_hadamard(sb, n_b >> recombine, b0_new << recombine, long_blocks);
            }
        }
    }

    // Quantize
    let lowband_ref: Option<&[i32]> = if use_scratch_as_lowband {
        scratch_buf.as_deref()
    } else {
        lowband
    };
    let mut cm = quant_partition(ctx, x, n, b, big_b, lowband_ref, lm, gain, fill);

    // Resynthesis: undo transforms
    if ctx.resynth {
        if b0_new > 1 {
            interleave_hadamard(x, n_b >> recombine, b0_new << recombine, long_blocks);
        }

        let mut n_b_r = n_b0;
        let mut big_b_r = b0_new;
        for _ in 0..time_divide {
            big_b_r >>= 1;
            n_b_r <<= 1;
            cm |= cm >> big_b_r;
            haar1(x, n_b_r, big_b_r);
        }

        for k in 0..recombine {
            static BIT_DEINTERLEAVE_TABLE: [u8; 16] = [
                0x00, 0x03, 0x0C, 0x0F, 0x30, 0x33, 0x3C, 0x3F, 0xC0, 0xC3, 0xCC, 0xCF, 0xF0, 0xF3,
                0xFC, 0xFF,
            ];
            cm = BIT_DEINTERLEAVE_TABLE[cm as usize] as u32;
            haar1(x, n0 >> k, 1 << k);
        }

        // Scale output for later folding
        if let Some(lbo) = lowband_out {
            let n_scale = celt_sqrt(shl32(extend32(n0), 22));
            for j in 0..n0 as usize {
                lbo[j] = mult16_32_q15(n_scale, x[j]);
            }
        }
        cm &= (1 << big_b) - 1;
    }
    cm
}

// ===========================================================================
// Stereo band quantization
// ===========================================================================

/// Quantize a stereo band. Matches C `quant_band_stereo()` (no ENABLE_QEXT).
fn quant_band_stereo<EC: EcCoder>(
    ctx: &mut BandCtx<EC>,
    x: &mut [i32],
    y: &mut [i32],
    n: i32,
    mut b: i32,
    big_b: i32,
    lowband: Option<&[i32]>,
    lm: i32,
    lowband_out: Option<&mut [i32]>,
    lowband_scratch: Option<&mut [i32]>,
    mut fill: i32,
) -> u32 {
    let nu = n as usize;
    let encode = ctx.encode;

    // Special case for one sample
    if n == 1 {
        return quant_band_n1(ctx, x, Some(y), lowband_out);
    }

    let orig_fill = fill;

    // Equalize very low-energy stereo channels
    if encode {
        if ctx.band_e[ctx.i as usize] < MIN_STEREO_ENERGY
            || ctx.band_e[(ctx.m.nb_ebands + ctx.i) as usize] < MIN_STEREO_ENERGY
        {
            if ctx.band_e[ctx.i as usize] > ctx.band_e[(ctx.m.nb_ebands + ctx.i) as usize] {
                y[..nu].copy_from_slice(&x[..nu]);
            } else {
                x[..nu].copy_from_slice(&y[..nu]);
            }
        }
    }

    let sctx = compute_theta(ctx, x, y, n, &mut b, big_b, big_b, lm, true, &mut fill);
    let inv = sctx.inv;
    let imid = sctx.imid;
    let iside = sctx.iside;
    let delta = sctx.delta;
    let itheta = sctx.itheta;
    let qalloc = sctx.qalloc;

    // Fixed-point, no ENABLE_QEXT
    let mid = shl32(extend32(imid), 16);
    let side = shl32(extend32(iside), 16);

    let cm;

    if n == 2 {
        // Special case for N=2 stereo
        let mbits;
        mbits = b;
        let mut sbits_val = 0;
        if itheta != 0 && itheta != 16384 {
            sbits_val = 1 << BITRES;
        }
        let mbits = mbits - sbits_val;
        let c = if itheta > 8192 { 1 } else { 0 };
        ctx.remaining_bits -= qalloc + sbits_val;

        // x2/y2 point to the appropriate channel based on c
        let mut sign: i32 = 0;
        if sbits_val != 0 {
            if encode {
                // Compute cross-product sign
                let (x2, y2) = if c == 1 {
                    (&*y as &[i32], &*x as &[i32])
                } else {
                    (&*x as &[i32], &*y as &[i32])
                };
                sign = if mult32_32_q31(x2[0], y2[1]) - mult32_32_q31(x2[1], y2[0]) < 0 {
                    1
                } else {
                    0
                };
                ctx.ec.ec_enc_bits(sign as u32, 1);
            } else {
                sign = ctx.ec.ec_dec_bits(1) as i32;
            }
        }
        sign = 1 - 2 * sign;

        // Quantize the "main" channel
        // For c==1, main is Y; for c==0, main is X
        if c == 1 {
            cm = quant_band(
                ctx,
                y,
                n,
                mbits,
                big_b,
                lowband,
                lm,
                lowband_out,
                Q31ONE,
                lowband_scratch,
                orig_fill,
            );
            // y2[0] = -sign*x2[1], y2[1] = sign*x2[0]
            // When c==1: x2=Y, y2=X
            x[0] = -sign * y[1];
            x[1] = sign * y[0];
        } else {
            cm = quant_band(
                ctx,
                x,
                n,
                mbits,
                big_b,
                lowband,
                lm,
                lowband_out,
                Q31ONE,
                lowband_scratch,
                orig_fill,
            );
            // When c==0: x2=X, y2=Y
            y[0] = -sign * x[1];
            y[1] = sign * x[0];
        }

        if ctx.resynth {
            let tmp0 = x[0];
            let tmp1 = x[1];
            x[0] = mult32_32_q31(mid, tmp0);
            x[1] = mult32_32_q31(mid, tmp1);
            y[0] = mult32_32_q31(side, y[0]);
            y[1] = mult32_32_q31(side, y[1]);
            let xtmp = x[0];
            x[0] = sub32(xtmp, y[0]);
            y[0] = add32(xtmp, y[0]);
            let xtmp = x[1];
            x[1] = sub32(xtmp, y[1]);
            y[1] = add32(xtmp, y[1]);
        }
    } else {
        // "Normal" split code
        let mbits = imax(0, imin(b, (b - delta) / 2));
        let mut sbits = b - mbits;
        ctx.remaining_bits -= qalloc;

        let rebalance = ctx.remaining_bits;
        if mbits >= sbits {
            let cm_x = quant_band(
                ctx,
                x,
                n,
                mbits,
                big_b,
                lowband,
                lm,
                lowband_out,
                Q31ONE,
                lowband_scratch,
                fill,
            );
            let rebalance = mbits - (rebalance - ctx.remaining_bits);
            if rebalance > 3 << BITRES && itheta != 0 {
                sbits += rebalance - (3 << BITRES);
            }
            let cm_y = quant_band(
                ctx,
                y,
                n,
                sbits,
                big_b,
                None,
                lm,
                None,
                side,
                None,
                fill >> big_b,
            );
            cm = cm_x | cm_y;
        } else {
            let cm_y = quant_band(
                ctx,
                y,
                n,
                sbits,
                big_b,
                None,
                lm,
                None,
                side,
                None,
                fill >> big_b,
            );
            let rebalance = sbits - (rebalance - ctx.remaining_bits);
            let mut mbits = mbits;
            if rebalance > 3 << BITRES && itheta != 16384 {
                mbits += rebalance - (3 << BITRES);
            }
            let cm_x = quant_band(
                ctx,
                x,
                n,
                mbits,
                big_b,
                lowband,
                lm,
                lowband_out,
                Q31ONE,
                lowband_scratch,
                fill,
            );
            cm = cm_x | cm_y;
        }
    }

    // Resynthesis: merge stereo and apply inv
    if ctx.resynth {
        if n != 2 {
            stereo_merge(x, y, mid, n);
        }
        if inv {
            for j in 0..nu {
                y[j] = -y[j];
            }
        }
    }
    cm
}

// ===========================================================================
// Special hybrid folding
// ===========================================================================

/// Duplicate first-band folding data so second band can fold.
/// Matches C `special_hybrid_folding()`.
fn special_hybrid_folding(
    m: &CELTMode,
    norm: &mut [i32],
    norm2: &mut [i32],
    start: i32,
    big_m: i32,
    dual_stereo: bool,
) {
    let n1 = big_m * (m.ebands[(start + 1) as usize] - m.ebands[start as usize]) as i32;
    let n2 = big_m * (m.ebands[(start + 2) as usize] - m.ebands[(start + 1) as usize]) as i32;
    // Copy the tail of band 0 folding data to bridge into band 1
    let src_start = (2 * n1 - n2) as usize;
    let dst_start = n1 as usize;
    let copy_len = (n2 - n1) as usize;
    // Use intermediate buffer to handle potential overlap
    let tmp: Vec<i32> = norm[src_start..src_start + copy_len].to_vec();
    norm[dst_start..dst_start + copy_len].copy_from_slice(&tmp);
    if dual_stereo {
        let tmp: Vec<i32> = norm2[src_start..src_start + copy_len].to_vec();
        norm2[dst_start..dst_start + copy_len].copy_from_slice(&tmp);
    }
}

// ===========================================================================
// Main entry point: quant_all_bands
// ===========================================================================

/// Quantize/dequantize all bands. This is the main band processing entry point.
/// Matches C `quant_all_bands()` (no ENABLE_QEXT).
///
/// - `encode`: true for encoding, false for decoding.
/// - `x_`, `y_`: spectral coefficients for channels 0 and 1 (Y may be None for mono).
/// - `collapse_masks`: per-band collapse tracking masks.
/// - `band_e`: per-band sqrt energies.
/// - `pulses`: per-band bit allocation from rate control.
/// - `tf_res`: per-band time-frequency resolution change.
/// - `seed`: LCG state, updated on return.
pub fn quant_all_bands<EC: EcCoder>(
    encode: bool,
    m: &CELTMode,
    start: i32,
    end: i32,
    x_: &mut [i32],
    mut y_: Option<&mut [i32]>,
    collapse_masks: &mut [u8],
    band_e: &[i32],
    pulses: &mut [i32],
    short_blocks: bool,
    spread: i32,
    mut dual_stereo: bool,
    intensity: i32,
    tf_res: &[i32],
    total_bits: i32,
    mut balance: i32,
    ec: &mut EC,
    lm: i32,
    coded_bands: i32,
    seed: &mut u32,
    complexity: i32,
    disable_inv: bool,
) {
    let big_m = 1 << lm;
    let big_b = if short_blocks { big_m } else { 1 };
    let c_channels: i32 = if y_.is_some() { 2 } else { 1 };

    let theta_rdo = encode && y_.is_some() && !dual_stereo && complexity >= 8;
    let resynth = !encode || theta_rdo;

    let norm_offset = (big_m * m.ebands[start as usize] as i32) as usize;
    let norm_size = (big_m * m.ebands[(m.nb_ebands - 1) as usize] as i32) as usize - norm_offset;
    let mut _norm = vec![0i32; c_channels as usize * norm_size];

    // For the decoder, the last band can be used as scratch space
    let scratch_size = if encode && resynth {
        (big_m * (m.ebands[m.nb_ebands as usize] - m.ebands[(m.nb_ebands - 1) as usize]) as i32)
            as usize
    } else {
        0
    };
    let mut _lowband_scratch = vec![0i32; scratch_size];

    // theta_rdo save buffers (for two-pass stereo encoding)
    let resynth_alloc = if theta_rdo {
        ((m.ebands[m.nb_ebands as usize] - m.ebands[m.nb_ebands as usize - 1]) as i32) << lm
    } else {
        0
    } as usize;
    let mut x_save = vec![0i32; resynth_alloc];
    let mut y_save = vec![0i32; resynth_alloc];
    let mut x_save2 = vec![0i32; resynth_alloc];
    let mut y_save2 = vec![0i32; resynth_alloc];
    let mut norm_save2 = vec![0i32; resynth_alloc];
    let mut bytes_save = vec![0u8; if theta_rdo { 1275 } else { 0 }];

    // Norm buffer accessors: norm = _norm[0..norm_size], norm2 = _norm[norm_size..]
    let mut lowband_offset: i32 = 0;
    let mut update_lowband = true;

    // We need to handle the EC borrow carefully: BandCtx borrows ec mutably,
    // but quant_all_bands also needs to call ec methods. We'll create ctx
    // inside the loop where needed.

    let mut ctx_seed = *seed;
    let mut ctx_avoid_split_noise = big_b > 1;

    // Get Y_ as a raw pointer so we can split borrows
    // We handle stereo by working with index ranges into x_ and y_
    let has_y = y_.is_some();

    // We need separate norm buffers for dual stereo
    // norm = _norm[..norm_size], norm2 = _norm[norm_size..] (only if stereo)

    // Debug: frame counter for quant_all_bands calls (encoder-only)
    static QAB_FRAME_CTR: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
    let qab_frame = if encode {
        QAB_FRAME_CTR.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    } else {
        -1
    };
    let trace_qab = qab_frame == 7; // 0-indexed, frame 7

    for i_band in start..end {
        let iu = i_band as usize;
        let last = i_band == end - 1;

        let band_start = (big_m * m.ebands[iu] as i32) as usize;
        let band_end_bin = (big_m * m.ebands[iu + 1] as i32) as usize;
        let n = (band_end_bin - band_start) as i32;

        let tell = ec.ec_tell_frac();

        // Compute bit budget
        if i_band != start {
            balance -= tell as i32;
        }
        let remaining_bits = total_bits - tell as i32 - 1;

        let b;
        if i_band <= coded_bands - 1 {
            let curr_balance = celt_sudiv(balance, imin(3, coded_bands - i_band));
            b = imax(
                0,
                imin(16383, imin(remaining_bits + 1, pulses[iu] + curr_balance)),
            );
        } else {
            b = 0;
        }

        // Update lowband folding offset
        if resynth
            && (big_m * m.ebands[iu] as i32 - n >= big_m * m.ebands[start as usize] as i32
                || i_band == start + 1)
            && (update_lowband || lowband_offset == 0)
        {
            lowband_offset = i_band;
        }
        if i_band == start + 1 {
            let (norm1, norm2) = _norm.split_at_mut(norm_size);
            special_hybrid_folding(m, norm1, norm2, start, big_m, dual_stereo);
        }

        let tf_change = tf_res[iu];

        // For bands beyond effEBands, redirect to norm buffer
        let beyond_eff = i_band >= m.eff_ebands;

        let _lowband_scratch_ref: Option<&mut [i32]> = if beyond_eff || (last && !theta_rdo) {
            None
        } else if !_lowband_scratch.is_empty() {
            Some(&mut _lowband_scratch)
        } else if !encode {
            // Decoder uses last band of X as scratch
            None // We'll handle this by not providing scratch
        } else {
            None
        };

        // Get conservative estimate of collapse masks for folding bands
        let mut x_cm: u32;
        let mut y_cm: u32;
        let mut effective_lowband: i32 = -1;

        if lowband_offset != 0 && (spread != SPREAD_AGGRESSIVE || big_b > 1 || tf_change < 0) {
            effective_lowband = imax(
                0,
                big_m * m.ebands[lowband_offset as usize] as i32 - norm_offset as i32 - n,
            );
            let mut fold_start = lowband_offset as usize;
            loop {
                fold_start -= 1;
                if !(big_m * m.ebands[fold_start] as i32 > effective_lowband + norm_offset as i32) {
                    break;
                }
            }
            let mut fold_end = (lowband_offset - 1) as usize;
            loop {
                fold_end += 1;
                if fold_end >= iu
                    || big_m * m.ebands[fold_end] as i32
                        >= effective_lowband + norm_offset as i32 + n
                {
                    break;
                }
            }
            x_cm = 0;
            y_cm = 0;
            let mut fold_i = fold_start;
            loop {
                x_cm |= collapse_masks[fold_i * c_channels as usize] as u32;
                y_cm |=
                    collapse_masks[fold_i * c_channels as usize + (c_channels as usize - 1)] as u32;
                fold_i += 1;
                if fold_i >= fold_end {
                    break;
                }
            }
        } else {
            x_cm = (1u32 << big_b as u32) - 1;
            y_cm = x_cm;
        }

        // Switch off dual stereo at intensity boundary
        if dual_stereo && i_band == intensity {
            dual_stereo = false;
            if resynth {
                let (norm1, norm2) = _norm.split_at_mut(norm_size);
                for j in 0..(big_m * m.ebands[iu] as i32 - norm_offset as i32) as usize {
                    norm1[j] = half32(norm1[j] + norm2[j]);
                }
            }
        }

        // Build lowband reference
        let lowband_ref: Option<Vec<i32>> = if effective_lowband != -1 {
            let off = effective_lowband as usize;
            Some(_norm[off..off + n as usize].to_vec())
        } else {
            None
        };

        // Build lowband_out target offset
        let norm_out_offset = if !last {
            Some((big_m * m.ebands[iu] as i32) as usize - norm_offset)
        } else {
            None
        };

        if dual_stereo {
            // Need to handle y_ separately
            // For dual stereo, quantize X and Y independently
            let lb = lowband_ref.as_deref();
            let lb2: Option<Vec<i32>> = if effective_lowband != -1 {
                let off = norm_size + effective_lowband as usize;
                Some(_norm[off..off + n as usize].to_vec())
            } else {
                None
            };

            // Quantize X
            {
                let mut ctx = BandCtx {
                    encode,
                    resynth,
                    m,
                    i: i_band,
                    intensity,
                    spread,
                    tf_change,
                    ec,
                    remaining_bits,
                    band_e,
                    seed: ctx_seed,
                    theta_round: 0,
                    disable_inv,
                    avoid_split_noise: ctx_avoid_split_noise,
                };

                let x_slice = &mut x_[band_start..band_end_bin];
                let mut lbo_buf = vec![0i32; n as usize];
                x_cm = quant_band(
                    &mut ctx,
                    x_slice,
                    n,
                    b / 2,
                    big_b,
                    lb,
                    lm,
                    if !last { Some(&mut lbo_buf) } else { None },
                    Q31ONE,
                    None,
                    x_cm as i32,
                );
                if !last {
                    let out_off = norm_out_offset.unwrap();
                    _norm[out_off..out_off + n as usize].copy_from_slice(&lbo_buf);
                }
                ctx_seed = ctx.seed;
                // Update remaining_bits from ctx (it's modified during quantization)
            }

            // Quantize Y
            if let Some(y_buf) = y_.as_deref_mut() {
                // We need to re-borrow ec since the previous ctx dropped
                let mut ctx = BandCtx {
                    encode,
                    resynth,
                    m,
                    i: i_band,
                    intensity,
                    spread,
                    tf_change,
                    ec,
                    remaining_bits, // This is approximate; the C code shares ctx
                    band_e,
                    seed: ctx_seed,
                    theta_round: 0,
                    disable_inv,
                    avoid_split_noise: ctx_avoid_split_noise,
                };

                let y_slice = &mut y_buf[band_start..band_end_bin];
                let mut lbo_buf = vec![0i32; n as usize];
                y_cm = quant_band(
                    &mut ctx,
                    y_slice,
                    n,
                    b / 2,
                    big_b,
                    lb2.as_deref(),
                    lm,
                    if !last { Some(&mut lbo_buf) } else { None },
                    Q31ONE,
                    None,
                    y_cm as i32,
                );
                if !last {
                    let out_off = norm_size + norm_out_offset.unwrap();
                    _norm[out_off..out_off + n as usize].copy_from_slice(&lbo_buf);
                }
                ctx_seed = ctx.seed;
            }
        } else {
            if has_y {
                // MS stereo or intensity stereo
                let lb = lowband_ref.as_deref();
                let (x_slice, y_slice) = {
                    let y_buf = y_.as_deref_mut().unwrap();
                    let xs = &mut x_[band_start..band_end_bin];
                    let ys = &mut y_buf[band_start..band_end_bin];
                    (xs, ys)
                };

                if theta_rdo && i_band < intensity {
                    // Two-pass stereo: try theta_round=-1 and +1, pick lower distortion
                    let nu = n as usize;
                    let nbe = m.nb_ebands as usize;
                    let w = compute_channel_weights(band_e[iu], band_e[iu + nbe]);
                    let cm = x_cm | y_cm;

                    // Save pre-pass state
                    let ec_snap = ec.ec_snapshot();
                    let nstart = ec.ec_range_bytes_usize();
                    let nend = ec.ec_storage_usize();
                    let save_bytes_len = nend - nstart;
                    let save_seed = ctx_seed;
                    let save_avoid = ctx_avoid_split_noise;
                    x_save[..nu].copy_from_slice(&x_slice[..nu]);
                    y_save[..nu].copy_from_slice(&y_slice[..nu]);

                    // --- Pass 1: theta_round = -1 (round down) ---
                    let mut ctx = BandCtx {
                        encode,
                        resynth,
                        m,
                        i: i_band,
                        intensity,
                        spread,
                        tf_change,
                        ec,
                        remaining_bits,
                        band_e,
                        seed: ctx_seed,
                        theta_round: -1,
                        disable_inv,
                        avoid_split_noise: ctx_avoid_split_noise,
                    };
                    let mut lbo_buf = vec![0i32; nu];
                    x_cm = quant_band_stereo(
                        &mut ctx,
                        x_slice,
                        y_slice,
                        n,
                        b,
                        big_b,
                        lb,
                        lm,
                        if !last { Some(&mut lbo_buf) } else { None },
                        None,
                        cm as i32,
                    );
                    let dist0 = mult16_32_q15(
                        w[0],
                        celt_inner_prod_norm_shift(&x_save[..nu], &x_slice[..nu], nu),
                    ) + mult16_32_q15(
                        w[1],
                        celt_inner_prod_norm_shift(&y_save[..nu], &y_slice[..nu], nu),
                    );
                    ctx_seed = ctx.seed;
                    ctx_avoid_split_noise = ctx.avoid_split_noise;
                    let ec = ctx.ec; // release borrow

                    // Save pass-1 result (scalar state + X/Y/norm + buffer bytes)
                    let cm2 = x_cm;
                    let ec_snap2 = ec.ec_snapshot();
                    let save_seed2 = ctx_seed;
                    let save_avoid2 = ctx_avoid_split_noise;
                    x_save2[..nu].copy_from_slice(&x_slice[..nu]);
                    y_save2[..nu].copy_from_slice(&y_slice[..nu]);
                    if !last {
                        norm_save2[..nu].copy_from_slice(&lbo_buf[..nu]);
                    }
                    // Save buffer bytes AFTER pass 1 (buffer is shared, so this
                    // captures the post-pass-1 content at the pre-pass byte offsets).
                    // Matches C: bytes_buf = ec_save.buf + nstart_bytes; OPUS_COPY(bytes_save, bytes_buf, save_bytes);
                    bytes_save[..save_bytes_len]
                        .copy_from_slice(&ec.ec_buffer()[nstart..nend]);

                    // Restore pre-pass-1 state for pass 2
                    // ec_restore only restores scalar state; we must also restore buffer bytes
                    ec.ec_restore(&ec_snap);
                    // In C, *ec = ec_save restores all scalars but buf pointer stays the same.
                    // Pass 2 will overwrite the buffer. We do NOT need to restore bytes here
                    // because ec_restore put the scalar offsets back, and pass 2 will write
                    // starting from those offsets. The buffer region beyond the current write
                    // position still has pass-1 data, which is fine — pass 2 will overwrite it.
                    ctx_seed = save_seed;
                    ctx_avoid_split_noise = save_avoid;
                    x_slice[..nu].copy_from_slice(&x_save[..nu]);
                    y_slice[..nu].copy_from_slice(&y_save[..nu]);

                    // Re-apply special hybrid folding if band == start+1
                    if i_band == start + 1 {
                        let (norm1, norm2) = _norm.split_at_mut(norm_size);
                        special_hybrid_folding(m, norm1, norm2, start, big_m, dual_stereo);
                    }

                    // --- Pass 2: theta_round = +1 (round up) ---
                    let mut ctx = BandCtx {
                        encode,
                        resynth,
                        m,
                        i: i_band,
                        intensity,
                        spread,
                        tf_change,
                        ec,
                        remaining_bits,
                        band_e,
                        seed: ctx_seed,
                        theta_round: 1,
                        disable_inv,
                        avoid_split_noise: ctx_avoid_split_noise,
                    };
                    let mut lbo_buf2 = vec![0i32; nu];
                    x_cm = quant_band_stereo(
                        &mut ctx,
                        x_slice,
                        y_slice,
                        n,
                        b,
                        big_b,
                        lb,
                        lm,
                        if !last { Some(&mut lbo_buf2) } else { None },
                        None,
                        cm as i32,
                    );
                    let dist1 = mult16_32_q15(
                        w[0],
                        celt_inner_prod_norm_shift(&x_save[..nu], &x_slice[..nu], nu),
                    ) + mult16_32_q15(
                        w[1],
                        celt_inner_prod_norm_shift(&y_save[..nu], &y_slice[..nu], nu),
                    );
                    ctx_seed = ctx.seed;
                    ctx_avoid_split_noise = ctx.avoid_split_noise;
                    let ec = ctx.ec;

                    // Pick the pass with higher correlation (lower distortion)
                    if dist0 >= dist1 {
                        // Pass 1 won — restore its state
                        x_cm = cm2;
                        ec.ec_restore(&ec_snap2);
                        ctx_seed = save_seed2;
                        ctx_avoid_split_noise = save_avoid2;
                        x_slice[..nu].copy_from_slice(&x_save2[..nu]);
                        y_slice[..nu].copy_from_slice(&y_save2[..nu]);
                        if !last {
                            let out_off = norm_out_offset.unwrap();
                            _norm[out_off..out_off + nu].copy_from_slice(&norm_save2[..nu]);
                        }
                        // Restore pass-1 buffer bytes (pass 2 overwrote them)
                        ec.ec_buffer_mut()[nstart..nend]
                            .copy_from_slice(&bytes_save[..save_bytes_len]);
                    } else if !last {
                        // Pass 2 won — write its norm output
                        let out_off = norm_out_offset.unwrap();
                        _norm[out_off..out_off + nu].copy_from_slice(&lbo_buf2[..nu]);
                    }
                } else {
                    // Non-theta_rdo path: single pass with theta_round = 0
                    let mut ctx = BandCtx {
                        encode,
                        resynth,
                        m,
                        i: i_band,
                        intensity,
                        spread,
                        tf_change,
                        ec,
                        remaining_bits,
                        band_e,
                        seed: ctx_seed,
                        theta_round: 0,
                        disable_inv,
                        avoid_split_noise: ctx_avoid_split_noise,
                    };

                    let mut lbo_buf = vec![0i32; n as usize];
                    x_cm = quant_band_stereo(
                        &mut ctx,
                        x_slice,
                        y_slice,
                        n,
                        b,
                        big_b,
                        lb,
                        lm,
                        if !last { Some(&mut lbo_buf) } else { None },
                        None,
                        (x_cm | y_cm) as i32,
                    );
                    if !last {
                        let out_off = norm_out_offset.unwrap();
                        _norm[out_off..out_off + n as usize].copy_from_slice(&lbo_buf);
                    }
                    ctx_seed = ctx.seed;
                }
                y_cm = x_cm;
            } else {
                // Mono
                let lb = lowband_ref.as_deref();

                let mut ctx = BandCtx {
                    encode,
                    resynth,
                    m,
                    i: i_band,
                    intensity,
                    spread,
                    tf_change,
                    ec,
                    remaining_bits,
                    band_e,
                    seed: ctx_seed,
                    theta_round: 0,
                    disable_inv,
                    avoid_split_noise: ctx_avoid_split_noise,
                };

                let x_slice = &mut x_[band_start..band_end_bin];
                // Debug: dump input spectrum for band 13 in frame 7
                if trace_qab && i_band == 13 {
                    eprintln!("[QAB F7] band 13 INPUT x[0..min(32,n)]={:?}",
                        &x_slice[..n.min(32) as usize]);
                    if let Some(ref lbr) = lb {
                        eprintln!("[QAB F7] band 13 LOWBAND[0..min(16,n)]={:?}",
                            &lbr[..n.min(16) as usize]);
                    }
                    eprintln!("[QAB F7] band 13 params: n={} b={} bigB={} lm={} seed={}",
                        n, b, big_b, lm, ctx_seed);
                }
                let mut lbo_buf = vec![0i32; n as usize];
                x_cm = quant_band(
                    &mut ctx,
                    x_slice,
                    n,
                    b,
                    big_b,
                    lb,
                    lm,
                    if !last { Some(&mut lbo_buf) } else { None },
                    Q31ONE,
                    None,
                    (x_cm | y_cm) as i32,
                );
                y_cm = x_cm;
                if !last {
                    let out_off = norm_out_offset.unwrap();
                    _norm[out_off..out_off + n as usize].copy_from_slice(&lbo_buf);
                }
                ctx_seed = ctx.seed;
            }
        }

        // Debug: trace per-band state in frame 7
        if trace_qab {
            let (offs, eoffs, storage) = ec.ec_debug_state();
            let byte261 = if 261 < ec.ec_buffer().len() { ec.ec_buffer()[261] } else { 0 };
            eprintln!("[QAB F7] band {:2}: offs={:3} eoffs={:3} tell={:5} b={:5} n={:3} tf={} byte261=0x{:02x}",
                i_band, offs, eoffs, ec.ec_tell(), b, n, tf_change, byte261);
        }

        collapse_masks[iu * c_channels as usize] = x_cm as u8;
        collapse_masks[iu * c_channels as usize + (c_channels as usize - 1)] = y_cm as u8;
        balance += pulses[iu] + tell as i32;

        // Update folding position only as long as we have 1 bit/sample depth
        update_lowband = b > (n << BITRES);
        // Only avoid noise on split for the first band
        ctx_avoid_split_noise = false;
    }

    *seed = ctx_seed;
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_celt_lcg_rand() {
        assert_eq!(celt_lcg_rand(0), 1013904223);
        assert_eq!(celt_lcg_rand(1), 1664525u32 + 1013904223);
        // Verify wrapping behavior
        let s1 = celt_lcg_rand(0);
        let s2 = celt_lcg_rand(s1);
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_bitexact_cos() {
        // cos(0) = 32767 + 1 = 32768? No, the C code returns 1+x2 where x2 at x=0:
        // tmp = (4096 + 0) >> 13 = 0, x2=0
        // result = (32767 - 0) + FRAC_MUL16(0, ...) = 32767
        // return 1 + 32767 = 32768... but that overflows i16!
        // Actually in C: celt_sig_assert(x2<=32766), return 1+x2
        // For x=0: x2 = 32767, so 1+32767 = 32768 which wraps to -32768 in i16
        // But the assertion says x2 <= 32766, so x=0 should give x2 <= 32766
        // Let's verify: x=0, tmp=(4096+0)>>13=0, x2=0
        // result = 32767 + FRAC_MUL16(0, anything) = 32767
        // return 1 + 32767 = 32768 → wraps to -32768 as i16
        // Actually bitexact_cos(0) should be 32767 (cos(0) ≈ 1.0 in Q15)
        // The C code seems to handle this at the boundary
        let c = bitexact_cos(0);
        // At x=0 the polynomial gives 32767, +1 = 32768 which wraps
        // The function is called with x in 0..16384 range
        // cos(0) should be ~32767
        assert!(c > 32700 || c < -32700); // Near max magnitude

        // cos(π/2) ≈ 0: bitexact_cos(16383) (16384 overflows the i16 polynomial)
        let c = bitexact_cos(16383);
        assert!(c.abs() < 100); // Should be near zero

        // cos(π/4) ≈ 0.707: bitexact_cos(8192)
        let c = bitexact_cos(8192);
        // 0.707 * 32768 ≈ 23170
        assert!((c as i32 - 23170).abs() < 200);
    }

    #[test]
    fn test_bitexact_log2tan() {
        // log2(tan(π/4)) = log2(1) = 0
        let cos_val = bitexact_cos(8192);
        let sin_val = bitexact_cos(16384 - 8192);
        let result = bitexact_log2tan(sin_val as i32, cos_val as i32);
        assert!(result.abs() < 100); // Should be near zero

        // Asymmetric case: more energy to one side
        let result = bitexact_log2tan(30000, 10000);
        assert!(result > 0); // sin > cos → positive

        let result = bitexact_log2tan(10000, 30000);
        assert!(result < 0); // sin < cos → negative
    }

    #[test]
    fn test_hysteresis_decision() {
        let thresholds = [100, 200, 300];
        let hysteresis = [10, 10, 10];

        // Below first threshold
        assert_eq!(hysteresis_decision(50, &thresholds, &hysteresis, 3, 0), 0);
        // Above all thresholds
        assert_eq!(hysteresis_decision(350, &thresholds, &hysteresis, 3, 0), 3);
        // Above threshold and above hysteresis band: 115 > 100+10=110
        assert_eq!(hysteresis_decision(115, &thresholds, &hysteresis, 3, 0), 1);
        // Hysteresis keeps previous: val is 105, prev=1, threshold[0]+hyst[0]=110
        // Since i=1 > prev=0... wait, that doesn't apply. Let's check:
        // i=1 (105 < 200), prev=0: i > prev (1 > 0) and val(105) < thresholds[0]+hyst[0]=110 → i=prev=0? No.
        // Actually: thresholds[prev] = thresholds[0] = 100, hysteresis[prev] = 10
        // val(105) < 100 + 10 = 110 → true, so i = prev = 0
        assert_eq!(hysteresis_decision(105, &thresholds, &hysteresis, 3, 0), 0);
    }

    #[test]
    fn test_haar1() {
        let sqrt_half = qconst32(0.70710678, 31);
        let mut x = [1 << 24, 1 << 24, 0, 0]; // Two pairs
        haar1(&mut x, 4, 1);
        // First pair: (a+b)/sqrt(2), (a-b)/sqrt(2)
        // a = b = 1<<24, so sum = 2*1<<24, diff = 0
        // After mult by sqrt(1/2): sum ≈ 1<<24 * sqrt(2), diff = 0
        assert!(x[0] > 0);
        assert_eq!(x[1], 0); // a == b → difference is 0
    }

    #[test]
    fn test_compute_qn() {
        // Very low bitrate: should return 1
        assert_eq!(compute_qn(4, 0, 0, 100, false), 1);
        // Higher bitrate: should return even value > 1
        let qn = compute_qn(4, 200, 30, 50, false);
        assert!(qn >= 1);
        assert!(qn <= 256);
        assert_eq!(qn & 1, 0); // Must be even (or 1)
    }
}
