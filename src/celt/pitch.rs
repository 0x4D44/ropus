//! CELT pitch analysis — pitch period estimation, correlation kernels,
//! and octave error correction.
//!
//! Matches the C reference: `pitch.c`, `pitch.h` (fixed-point path only).
//! All functions produce bit-exact output matching the C reference when
//! compiled with `FIXED_POINT` and `OPUS_FAST_INT64`.

use super::lpc::{celt_autocorr, celt_lpc};
use super::math_ops::{celt_ilog2, celt_maxabs16, celt_maxabs32, celt_rsqrt_norm, frac_div32};
use crate::types::*;

// ===========================================================================
// Constants
// ===========================================================================

/// Second-check table for remove_doubling: selects which harmonic to verify
/// when checking subharmonic divisor k.
const SECOND_CHECK: [i32; 16] = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2];

// ===========================================================================
// Inline correlation kernels (from pitch.h)
// ===========================================================================

/// Scalar implementation of xcorr_kernel (used in the non-SIMD path and tests).
///
/// The inner loop is unrolled 4x, maintaining a sliding window of y values.
/// Preconditions: `len >= 3`, `x` has >= `len` elements, `y` has >= `len + 3` elements.
#[inline(always)]
#[cfg(any(test, not(feature = "simd")))]
pub(crate) fn xcorr_kernel_scalar(x: &[i32], y: &[i32], sum: &mut [i32; 4], len: usize) {
    debug_assert!(len >= 3);

    let mut xi: usize = 0;
    let mut yi: usize = 0;

    let mut y_0 = y[yi];
    yi += 1;
    let mut y_1 = y[yi];
    yi += 1;
    let mut y_2 = y[yi];
    yi += 1;
    let mut y_3: i32 = 0; // Always assigned before use; init silences compiler

    // Main loop: process 4 x-values per iteration
    let mut j: usize = 0;
    while j + 3 < len {
        let tmp = x[xi];
        xi += 1;
        y_3 = y[yi];
        yi += 1;
        sum[0] = mac16_16(sum[0], tmp, y_0);
        sum[1] = mac16_16(sum[1], tmp, y_1);
        sum[2] = mac16_16(sum[2], tmp, y_2);
        sum[3] = mac16_16(sum[3], tmp, y_3);

        let tmp = x[xi];
        xi += 1;
        y_0 = y[yi];
        yi += 1;
        sum[0] = mac16_16(sum[0], tmp, y_1);
        sum[1] = mac16_16(sum[1], tmp, y_2);
        sum[2] = mac16_16(sum[2], tmp, y_3);
        sum[3] = mac16_16(sum[3], tmp, y_0);

        let tmp = x[xi];
        xi += 1;
        y_1 = y[yi];
        yi += 1;
        sum[0] = mac16_16(sum[0], tmp, y_2);
        sum[1] = mac16_16(sum[1], tmp, y_3);
        sum[2] = mac16_16(sum[2], tmp, y_0);
        sum[3] = mac16_16(sum[3], tmp, y_1);

        let tmp = x[xi];
        xi += 1;
        y_2 = y[yi];
        yi += 1;
        sum[0] = mac16_16(sum[0], tmp, y_3);
        sum[1] = mac16_16(sum[1], tmp, y_0);
        sum[2] = mac16_16(sum[2], tmp, y_1);
        sum[3] = mac16_16(sum[3], tmp, y_2);

        j += 4;
    }

    // Remainder: up to 3 more elements
    if j < len {
        j += 1;
        let tmp = x[xi];
        xi += 1;
        y_3 = y[yi];
        yi += 1;
        sum[0] = mac16_16(sum[0], tmp, y_0);
        sum[1] = mac16_16(sum[1], tmp, y_1);
        sum[2] = mac16_16(sum[2], tmp, y_2);
        sum[3] = mac16_16(sum[3], tmp, y_3);
    }
    if j < len {
        j += 1;
        let tmp = x[xi];
        xi += 1;
        y_0 = y[yi];
        yi += 1;
        sum[0] = mac16_16(sum[0], tmp, y_1);
        sum[1] = mac16_16(sum[1], tmp, y_2);
        sum[2] = mac16_16(sum[2], tmp, y_3);
        sum[3] = mac16_16(sum[3], tmp, y_0);
    }
    if j < len {
        let tmp = x[xi];
        y_1 = y[yi];
        sum[0] = mac16_16(sum[0], tmp, y_2);
        sum[1] = mac16_16(sum[1], tmp, y_3);
        sum[2] = mac16_16(sum[2], tmp, y_0);
        sum[3] = mac16_16(sum[3], tmp, y_1);
    }
}

/// Compute 4 cross-correlations simultaneously:
///   sum[i] += Sigma_{j=0}^{len-1} x[j] * y[j+i]  for i = 0..3
///
/// With the `simd` feature enabled, dispatches to a SIMD implementation
/// using `wide::i32x4`. Without the feature, uses the scalar implementation.
#[inline(always)]
pub fn xcorr_kernel(x: &[i32], y: &[i32], sum: &mut [i32; 4], len: usize) {
    #[cfg(feature = "simd")]
    {
        super::simd::xcorr_kernel_simd(x, y, sum, len);
    }
    #[cfg(not(feature = "simd"))]
    {
        xcorr_kernel_scalar(x, y, sum, len);
    }
}

/// Compute two inner products in a single pass.
/// Returns `(dot(x, y01), dot(x, y02))` over `n` elements.
#[inline(always)]
pub fn dual_inner_prod(x: &[i32], y01: &[i32], y02: &[i32], n: usize) -> (i32, i32) {
    let mut xy01: i32 = 0;
    let mut xy02: i32 = 0;
    for i in 0..n {
        xy01 = mac16_16(xy01, x[i], y01[i]);
        xy02 = mac16_16(xy02, x[i], y02[i]);
    }
    (xy01, xy02)
}

/// Single inner product: returns `Σ x[i] * y[i]` for `i = 0..n-1`.
#[inline(always)]
pub fn celt_inner_prod(x: &[i32], y: &[i32], n: usize) -> i32 {
    let mut xy: i32 = 0;
    for i in 0..n {
        xy = mac16_16(xy, x[i], y[i]);
    }
    xy
}

// ===========================================================================
// Cross-correlation
// ===========================================================================

/// Compute cross-correlation of `x` against `max_pitch` shifted copies of `y`.
///   xcorr[i] = dot(x[0..len-1], y[i..i+len-1])  for i = 0..max_pitch-1
///
/// Uses xcorr_kernel for 4-at-a-time processing in the main loop.
/// Returns maximum correlation value (used for shift normalization).
///
/// Preconditions: `max_pitch > 0`, `x` has >= `len` elements,
/// `y` has >= `len + max_pitch` elements.
pub fn celt_pitch_xcorr(
    x: &[i32],
    y: &[i32],
    xcorr: &mut [i32],
    len: usize,
    max_pitch: usize,
) -> i32 {
    debug_assert!(max_pitch > 0);
    let mut maxcorr: i32 = 1;
    let mut i: usize = 0;

    // Main loop: 4 lags at a time via xcorr_kernel (requires len >= 3)
    if len >= 3 {
        while i + 3 < max_pitch {
            let mut sum = [0i32; 4];
            xcorr_kernel(x, &y[i..], &mut sum, len);
            xcorr[i] = sum[0];
            xcorr[i + 1] = sum[1];
            xcorr[i + 2] = sum[2];
            xcorr[i + 3] = sum[3];
            // Track maximum across all 4 results
            sum[0] = max32(sum[0], sum[1]);
            sum[2] = max32(sum[2], sum[3]);
            sum[0] = max32(sum[0], sum[2]);
            maxcorr = max32(maxcorr, sum[0]);
            i += 4;
        }
    }

    // Remainder: individual lags
    while i < max_pitch {
        let sum = celt_inner_prod(x, &y[i..], len);
        xcorr[i] = sum;
        maxcorr = max32(maxcorr, sum);
        i += 1;
    }

    maxcorr
}

// ===========================================================================
// Private helpers
// ===========================================================================

/// 5-tap FIR filter applied in-place. Direct-form structure with internal
/// SIG_SHIFT precision. Used for spectral whitening in pitch_downsample.
///
/// - `x`: signal buffer (modified in-place), length >= `n`
/// - `num`: 5 filter coefficients in Q(SIG_SHIFT=12)
/// - `n`: number of samples to filter
fn celt_fir5(x: &mut [i32], num: &[i32; 5], n: usize) {
    let num0 = num[0];
    let num1 = num[1];
    let num2 = num[2];
    let num3 = num[3];
    let num4 = num[4];
    let mut mem0: i32 = 0;
    let mut mem1: i32 = 0;
    let mut mem2: i32 = 0;
    let mut mem3: i32 = 0;
    let mut mem4: i32 = 0;
    for i in 0..n {
        // x[i] is read before being overwritten — safe for in-place operation
        let mut sum = shl32(extend32(x[i]), SIG_SHIFT);
        sum = mac16_16(sum, num0, mem0);
        sum = mac16_16(sum, num1, mem1);
        sum = mac16_16(sum, num2, mem2);
        sum = mac16_16(sum, num3, mem3);
        sum = mac16_16(sum, num4, mem4);
        mem4 = mem3;
        mem3 = mem2;
        mem2 = mem1;
        mem1 = mem0;
        mem0 = x[i];
        x[i] = round16(sum, SIG_SHIFT);
    }
}

/// Find the top-2 lags with highest normalized correlation.
/// Compares xcorr[i]² / Syy(i) via cross-multiplication to avoid division.
///
/// Fixed-point parameters:
/// - `yshift`: right-shift applied when accumulating Syy (energy normalization)
/// - `maxcorr`: maximum correlation value from celt_pitch_xcorr (for shift calc)
fn find_best_pitch(
    xcorr: &[i32],
    y: &[i32],
    len: usize,
    max_pitch: usize,
    best_pitch: &mut [i32; 2],
    yshift: i32,
    maxcorr: i32,
) {
    let mut syy: i32 = 1;
    let mut best_num = [-1i32; 2];
    let mut best_den = [0i32; 2];

    // Normalize correlation values to ~14-bit before squaring
    let xshift = celt_ilog2(maxcorr) - 14;

    best_pitch[0] = 0;
    best_pitch[1] = 1;

    // Initial energy of y[0..len-1]
    for j in 0..len {
        syy = add32(syy, shr32(mult16_16(y[j], y[j]), yshift));
    }

    for i in 0..max_pitch {
        if xcorr[i] > 0 {
            // Shift correlation to ~14-bit, then square for comparison
            let xcorr16 = extract16(vshr32(xcorr[i], xshift));
            let num = mult16_16_q15(xcorr16, xcorr16);

            // Cross-multiply comparison: num/syy > best_num/best_den
            if mult16_32_q15(num, best_den[1]) > mult16_32_q15(best_num[1], syy) {
                if mult16_32_q15(num, best_den[0]) > mult16_32_q15(best_num[0], syy) {
                    // New best — push old best to second place
                    best_num[1] = best_num[0];
                    best_den[1] = best_den[0];
                    best_pitch[1] = best_pitch[0];
                    best_num[0] = num;
                    best_den[0] = syy;
                    best_pitch[0] = i as i32;
                } else {
                    // New second-best
                    best_num[1] = num;
                    best_den[1] = syy;
                    best_pitch[1] = i as i32;
                }
            }
        }
        // Slide energy window by 1: add y[i+len]², subtract y[i]²
        syy +=
            shr32(mult16_16(y[i + len], y[i + len]), yshift) - shr32(mult16_16(y[i], y[i]), yshift);
        syy = max32(1, syy);
    }
}

/// Compute normalized pitch gain: xy / sqrt(xx * yy).
/// Returns Q15 value clamped to [-Q15ONE, Q15ONE].
///
/// Fixed-point implementation uses careful shift management to avoid overflow:
/// 1. Normalize xx and yy to ~14-bit
/// 2. Compute product, adjust to even total shift
/// 3. Use reciprocal sqrt for division
fn compute_pitch_gain(xy: i32, xx: i32, yy: i32) -> i32 {
    if xy == 0 || xx == 0 || yy == 0 {
        return 0;
    }
    let sx = celt_ilog2(xx) - 14;
    let sy = celt_ilog2(yy) - 14;
    let mut shift = sx + sy;

    // Product of normalized xx and yy, shifted to ~14 bits
    let mut x2y2 = shr32(mult16_16(vshr32(xx, sx), vshr32(yy, sy)), 14);

    // celt_rsqrt_norm requires even total shift for correct Q-format
    if (shift & 1) != 0 {
        if x2y2 < 32768 {
            x2y2 <<= 1;
            shift -= 1;
        } else {
            x2y2 >>= 1;
            shift += 1;
        }
    }

    let den = celt_rsqrt_norm(x2y2);
    let g = mult16_32_q15(den, xy);
    let g = vshr32(g, (shift >> 1) - 1);
    extract16(max32(-Q15ONE, min32(g, Q15ONE)))
}

/// Unsigned integer division. Matches C `celt_udiv` (no lookup table optimization
/// in initial port — divisors are always small: 2*k where k ≤ 15).
#[inline(always)]
fn celt_udiv(n: i32, d: i32) -> i32 {
    // Both n and d are always positive in the calling context
    ((n as u32) / (d as u32)) as i32
}

// ===========================================================================
// Public API
// ===========================================================================

/// Downsample and whiten the input signal for pitch analysis.
///
/// Converts from `celt_sig` (Q27 fixed-point) to `opus_val16` (Q15) at
/// reduced sample rate, then applies an LPC-derived whitening filter.
///
/// - `x`: array of `c` channel slices, each with `len * factor` samples
/// - `x_lp`: output buffer, `len` samples (downsampled + whitened)
/// - `len`: output length (number of downsampled samples)
/// - `c`: number of channels (1 or 2)
/// - `factor`: downsampling factor (typically 2)
pub fn pitch_downsample(x: &[&[i32]], x_lp: &mut [i32], len: usize, c: usize, factor: usize) {
    let offset = factor / 2;

    // Fixed-point: compute dynamic shift to keep downsampled values in 16-bit range
    let mut maxabs = celt_maxabs32(&x[0][..len * factor]);
    if c == 2 {
        let maxabs_1 = celt_maxabs32(&x[1][..len * factor]);
        maxabs = max32(maxabs, maxabs_1);
    }
    if maxabs < 1 {
        maxabs = 1;
    }
    let mut shift = celt_ilog2(maxabs) - 10;
    if shift < 0 {
        shift = 0;
    }
    if c == 2 {
        shift += 1; // Extra bit headroom for channel summation
    }

    // Downsample with 3-tap FIR [0.25, 0.5, 0.25]
    for i in 1..len {
        x_lp[i] = shr32(x[0][factor * i - offset], shift + 2)
            + shr32(x[0][factor * i + offset], shift + 2)
            + shr32(x[0][factor * i], shift + 1);
    }
    // Boundary case: i=0 omits the (factor*i - offset) term
    x_lp[0] = shr32(x[0][offset], shift + 2) + shr32(x[0][0], shift + 1);

    if c == 2 {
        for i in 1..len {
            x_lp[i] += shr32(x[1][factor * i - offset], shift + 2)
                + shr32(x[1][factor * i + offset], shift + 2)
                + shr32(x[1][factor * i], shift + 1);
        }
        x_lp[0] += shr32(x[1][offset], shift + 2) + shr32(x[1][0], shift + 1);
    }

    // 4th-order autocorrelation of the downsampled signal
    let mut ac = [0i32; 5];
    celt_autocorr(x_lp, &mut ac, None, 0, 4, len);

    // Noise floor -40 dB
    ac[0] += shr32(ac[0], 13);

    // Lag windowing: ac[i] -= ac[i] * (0.008*i)²
    // Approximation of exp(-0.5*(2π*0.002*i)²)
    for i in 1..=4 {
        ac[i] -= mult16_32_q15(2 * (i as i32) * (i as i32), ac[i]);
    }

    // Levinson-Durbin LPC, order 4
    let mut lpc = [0i32; 4];
    celt_lpc(&mut lpc, &ac, 4);

    // Bandwidth expansion: scale by 0.9^i to stabilize the filter
    let mut tmp = Q15ONE;
    for i in 0..4 {
        tmp = mult16_16_q15(qconst16(0.9, 15), tmp);
        lpc[i] = mult16_16_q15(lpc[i], tmp);
    }

    // Add a zero at z = -0.8 to create 5-tap whitening FIR.
    // Convolves [1, lpc[0], lpc[1], lpc[2], lpc[3]] with [1, 0.8].
    let c1 = qconst16(0.8, 15);
    let lpc2: [i32; 5] = [
        lpc[0] + qconst16(0.8, SIG_SHIFT as u32),
        lpc[1] + mult16_16_q15(c1, lpc[0]),
        lpc[2] + mult16_16_q15(c1, lpc[1]),
        lpc[3] + mult16_16_q15(c1, lpc[2]),
        mult16_16_q15(c1, lpc[3]),
    ];

    // Apply whitening filter in-place
    celt_fir5(x_lp, &lpc2, len);
}

/// Multi-resolution pitch search. Finds the best pitch period using a
/// coarse-to-fine strategy with 4x and 2x decimation.
///
/// - `x_lp`: whitened signal from pitch_downsample, `len` samples
/// - `y`: past signal buffer, `len + max_pitch` samples
/// - `len`: analysis window length
/// - `max_pitch`: maximum pitch period (at 2x decimated rate)
///
/// Returns the best pitch period (at 2x decimated rate).
pub fn pitch_search(x_lp: &[i32], y: &[i32], len: usize, max_pitch: usize) -> i32 {
    debug_assert!(len > 0);
    debug_assert!(max_pitch > 0);

    let lag = len + max_pitch;
    let len4 = len >> 2;
    let lag4 = lag >> 2;
    let max_pitch4 = max_pitch >> 2;
    let max_pitch2 = max_pitch >> 1;
    let len2 = len >> 1;

    // Stage 1: Decimate to 4x (simple subsampling, every other sample)
    let mut x_lp4 = vec![0i32; len4];
    let mut y_lp4 = vec![0i32; lag4];
    let mut xcorr = vec![0i32; max_pitch2];

    for j in 0..len4 {
        x_lp4[j] = x_lp[2 * j];
    }
    for j in 0..lag4 {
        y_lp4[j] = y[2 * j];
    }

    // Fixed-point: normalize to prevent overflow in correlation
    let xmax = celt_maxabs16(&x_lp4);
    let ymax = celt_maxabs16(&y_lp4);
    let mut shift = celt_ilog2(max32(1, max32(xmax, ymax))) - 14 + celt_ilog2(len as i32) / 2;
    if shift > 0 {
        for j in 0..len4 {
            x_lp4[j] = shr16(x_lp4[j], shift);
        }
        for j in 0..lag4 {
            y_lp4[j] = shr16(y_lp4[j], shift);
        }
        // Use double the shift for a MAC (product of two shifted values)
        shift *= 2;
    } else {
        shift = 0;
    }

    // Stage 2: Coarse correlation at 4x decimation
    let maxcorr = celt_pitch_xcorr(&x_lp4, &y_lp4, &mut xcorr, len4, max_pitch4);

    let mut best_pitch = [0i32; 2];
    find_best_pitch(
        &xcorr,
        &y_lp4,
        len4,
        max_pitch4,
        &mut best_pitch,
        0,
        maxcorr,
    );

    // Stage 3: Fine correlation at 2x decimation
    // Only evaluate lags within ±2 of the two coarse candidates (scaled to 2x)
    let mut maxcorr: i32 = 1;
    for i in 0..max_pitch2 {
        xcorr[i] = 0;
        if (i as i32 - 2 * best_pitch[0]).abs() > 2 && (i as i32 - 2 * best_pitch[1]).abs() > 2 {
            continue;
        }
        // Fixed-point inner product with shift applied
        let mut sum: i32 = 0;
        for j in 0..len2 {
            sum += shr32(mult16_16(x_lp[j], y[i + j]), shift);
        }
        xcorr[i] = max32(-1, sum);
        maxcorr = max32(maxcorr, sum);
    }

    find_best_pitch(
        &xcorr,
        y,
        len2,
        max_pitch2,
        &mut best_pitch,
        shift + 1,
        maxcorr,
    );

    // Stage 4: Pseudo-interpolation refinement
    let offset;
    if best_pitch[0] > 0 && best_pitch[0] < (max_pitch2 as i32) - 1 {
        let bp = best_pitch[0] as usize;
        let a = xcorr[bp - 1];
        let b = xcorr[bp];
        let c = xcorr[bp + 1];
        if (c - a) > mult16_32_q15(qconst16(0.7, 15), b - a) {
            offset = 1;
        } else if (a - c) > mult16_32_q15(qconst16(0.7, 15), b - c) {
            offset = -1;
        } else {
            offset = 0;
        }
    } else {
        offset = 0;
    }

    2 * best_pitch[0] - offset
}

/// Correct pitch octave errors by checking subharmonics of the detected pitch.
/// Returns the refined pitch gain (Q15).
///
/// - `x`: signal buffer, at least `maxperiod + n` samples
/// - `maxperiod`: maximum allowed pitch period
/// - `minperiod`: minimum allowed pitch period
/// - `n`: analysis window length
/// - `t0`: in/out — initial pitch estimate → refined pitch period
/// - `prev_period`: pitch period from previous frame (for continuity)
/// - `prev_gain`: pitch gain from previous frame (Q15)
pub fn remove_doubling(
    x: &[i32],
    maxperiod: i32,
    minperiod: i32,
    n: i32,
    t0: &mut i32,
    prev_period: i32,
    prev_gain: i32,
) -> i32 {
    let minperiod0 = minperiod;

    // Work at half resolution
    let maxperiod = maxperiod / 2;
    let minperiod = minperiod / 2;
    *t0 /= 2;
    let prev_period = prev_period / 2;
    let n = (n / 2) as usize;

    // x[base] corresponds to C's x[0] after `x += maxperiod`
    let base = maxperiod as usize;

    if *t0 >= maxperiod {
        *t0 = maxperiod - 1;
    }
    let t0_val = *t0;
    let mut t = t0_val;

    // Precompute energy lookup table via sliding window
    let mut yy_lookup = vec![0i32; (maxperiod as usize) + 1];

    let (xx, xy) = dual_inner_prod(&x[base..], &x[base..], &x[base - t0_val as usize..], n);

    yy_lookup[0] = xx;
    let mut yy = xx;
    for i in 1..=(maxperiod as usize) {
        // Slide window: add x[-i]², subtract x[N-i]² (wrapping matches C)
        yy = yy
            .wrapping_add(mult16_16(x[base - i], x[base - i]))
            .wrapping_sub(mult16_16(x[base + n - i], x[base + n - i]));
        yy_lookup[i] = max32(0, yy);
    }

    yy = yy_lookup[t0_val as usize];
    let mut best_xy = xy;
    let mut best_yy = yy;
    let g0 = compute_pitch_gain(xy, xx, yy);
    let mut g = g0;

    // Check subharmonics: for each divisor k = 2..15, test period T0/k
    for k in 2..=15i32 {
        let t1 = celt_udiv(2 * t0_val + k, 2 * k);
        if t1 < minperiod {
            break;
        }

        // Secondary check period for confidence
        let t1b;
        if k == 2 {
            if t1 + t0_val > maxperiod {
                t1b = t0_val;
            } else {
                t1b = t0_val + t1;
            }
        } else {
            t1b = celt_udiv(2 * SECOND_CHECK[k as usize] * t0_val + k, 2 * k);
        }

        // Average correlation at T1 and T1b for robustness
        let (xy_new, xy2) = dual_inner_prod(
            &x[base..],
            &x[base - t1 as usize..],
            &x[base - t1b as usize..],
            n,
        );
        let xy_avg = half32(xy_new.wrapping_add(xy2));
        let yy_avg = half32(yy_lookup[t1 as usize].wrapping_add(yy_lookup[t1b as usize]));
        let g1 = compute_pitch_gain(xy_avg, xx, yy_avg);

        // Continuity bias: favor pitches close to the previous frame's pitch
        let cont;
        if (t1 - prev_period).abs() <= 1 {
            cont = prev_gain;
        } else if (t1 - prev_period).abs() <= 2 && 5 * k * k < t0_val {
            cont = half16(prev_gain);
        } else {
            cont = 0;
        }

        // Threshold with bias against very short periods (false-positive prevention)
        let thresh;
        if t1 < 3 * minperiod {
            thresh = max16(
                qconst16(0.4, 15),
                mult16_16_q15(qconst16(0.85, 15), g0) - cont,
            );
        } else if t1 < 2 * minperiod {
            // Note: this branch is unreachable (2*min < 3*min), but matches C exactly
            thresh = max16(
                qconst16(0.5, 15),
                mult16_16_q15(qconst16(0.9, 15), g0) - cont,
            );
        } else {
            thresh = max16(
                qconst16(0.3, 15),
                mult16_16_q15(qconst16(0.7, 15), g0) - cont,
            );
        }

        if g1 > thresh {
            best_xy = xy_avg;
            best_yy = yy_avg;
            t = t1;
            g = g1;
        }
    }

    // Compute final pitch gain as best_xy / (best_yy + 1)
    best_xy = max32(0, best_xy);
    let mut pg;
    if best_yy <= best_xy {
        pg = Q15ONE;
    } else {
        pg = shr32(frac_div32(best_xy, best_yy + 1), 16);
    }

    // Sub-sample refinement: 3-point correlation around the best period
    let mut xcorr_arr = [0i32; 3];
    for k in 0..3i32 {
        let lag = t + k - 1;
        let start = (base as i32 - lag) as usize;
        xcorr_arr[k as usize] = celt_inner_prod(&x[base..], &x[start..], n);
    }

    let offset;
    if xcorr_arr[2].wrapping_sub(xcorr_arr[0])
        > mult16_32_q15(qconst16(0.7, 15), xcorr_arr[1].wrapping_sub(xcorr_arr[0]))
    {
        offset = 1;
    } else if xcorr_arr[0].wrapping_sub(xcorr_arr[2])
        > mult16_32_q15(qconst16(0.7, 15), xcorr_arr[1].wrapping_sub(xcorr_arr[2]))
    {
        offset = -1;
    } else {
        offset = 0;
    }

    if pg > g {
        pg = g;
    }

    // Scale back to original resolution and clamp
    *t0 = 2 * t + offset;
    if *t0 < minperiod0 {
        *t0 = minperiod0;
    }

    pg
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_celt_inner_prod_basic() {
        let x = [1, 2, 3, 4, 5];
        let y = [5, 4, 3, 2, 1];
        // 1*5 + 2*4 + 3*3 + 4*2 + 5*1 = 5 + 8 + 9 + 8 + 5 = 35
        assert_eq!(celt_inner_prod(&x, &y, 5), 35);
    }

    #[test]
    fn test_celt_inner_prod_zero_length() {
        let x = [1, 2, 3];
        let y = [4, 5, 6];
        assert_eq!(celt_inner_prod(&x, &y, 0), 0);
    }

    #[test]
    fn test_dual_inner_prod_basic() {
        let x = [1, 2, 3];
        let y01 = [4, 5, 6];
        let y02 = [7, 8, 9];
        // dot(x, y01) = 1*4 + 2*5 + 3*6 = 32
        // dot(x, y02) = 1*7 + 2*8 + 3*9 = 50
        let (xy1, xy2) = dual_inner_prod(&x, &y01, &y02, 3);
        assert_eq!(xy1, 32);
        assert_eq!(xy2, 50);
    }

    #[test]
    fn test_xcorr_kernel_basic() {
        // x = [1, 1, 1], y = [1, 1, 1, 0, 0, 0]
        // sum[0] = dot(x, y[0..3]) = 1+1+1 = 3
        // sum[1] = dot(x, y[1..4]) = 1+1+0 = 2
        // sum[2] = dot(x, y[2..5]) = 1+0+0 = 1
        // sum[3] = dot(x, y[3..6]) = 0+0+0 = 0
        let x = [1, 1, 1];
        let y = [1, 1, 1, 0, 0, 0];
        let mut sum = [0i32; 4];
        xcorr_kernel(&x, &y, &mut sum, 3);
        assert_eq!(sum, [3, 2, 1, 0]);
    }

    #[test]
    fn test_xcorr_kernel_len_4() {
        // len=4 exercises the main loop (1 iteration) with no remainder
        let x = [1, 2, 3, 4];
        let y = [1, 2, 3, 4, 5, 6, 7];
        let mut sum = [0i32; 4];
        xcorr_kernel(&x, &y, &mut sum, 4);
        // sum[0] = 1*1 + 2*2 + 3*3 + 4*4 = 30
        // sum[1] = 1*2 + 2*3 + 3*4 + 4*5 = 40
        // sum[2] = 1*3 + 2*4 + 3*5 + 4*6 = 50
        // sum[3] = 1*4 + 2*5 + 3*6 + 4*7 = 60
        assert_eq!(sum, [30, 40, 50, 60]);
    }

    #[test]
    fn test_xcorr_kernel_len_7() {
        // len=7: main loop runs once (4 elements), remainder handles 3
        let x = [1; 7];
        let y = [1; 10];
        let mut sum = [0i32; 4];
        xcorr_kernel(&x, &y, &mut sum, 7);
        assert_eq!(sum, [7, 7, 7, 7]);
    }

    #[test]
    fn test_xcorr_kernel_accumulates() {
        // Verify that sum is accumulated, not overwritten
        let x = [1, 1, 1];
        let y = [1, 1, 1, 1, 1, 1];
        let mut sum = [10, 20, 30, 40];
        xcorr_kernel(&x, &y, &mut sum, 3);
        assert_eq!(sum, [13, 23, 33, 43]);
    }

    #[test]
    fn test_celt_pitch_xcorr_basic() {
        let x = [1, 1, 1, 1];
        let y = [0, 0, 1, 1, 1, 1, 0, 0];
        let mut xcorr = [0i32; 4];
        let maxcorr = celt_pitch_xcorr(&x, &y, &mut xcorr, 4, 4);
        // xcorr[0] = dot(x, y[0..4]) = 0+0+1+1 = 2
        // xcorr[1] = dot(x, y[1..5]) = 0+1+1+1 = 3
        // xcorr[2] = dot(x, y[2..6]) = 1+1+1+1 = 4
        // xcorr[3] = dot(x, y[3..7]) = 1+1+1+0 = 3
        assert_eq!(xcorr, [2, 3, 4, 3]);
        assert_eq!(maxcorr, 4);
    }

    #[test]
    fn test_compute_pitch_gain_zero() {
        assert_eq!(compute_pitch_gain(0, 100, 100), 0);
        assert_eq!(compute_pitch_gain(100, 0, 100), 0);
        assert_eq!(compute_pitch_gain(100, 100, 0), 0);
    }

    #[test]
    fn test_compute_pitch_gain_perfect_correlation() {
        // When xy = xx = yy, gain should be close to Q15ONE
        let val = 1_000_000;
        let g = compute_pitch_gain(val, val, val);
        // Should be close to 32767 (Q15ONE)
        assert!(g > 30000, "Expected near-perfect gain, got {}", g);
    }

    #[test]
    fn test_celt_fir5_passthrough() {
        // Zero filter coefficients → output = input (since sum = x[i] << SIG_SHIFT,
        // then ROUND16(sum, SIG_SHIFT) ≈ x[i])
        let mut x = [1000, -500, 250, -125, 63];
        let num = [0i32; 5];
        celt_fir5(&mut x, &num, 5);
        // With zero coefficients, output should match input
        assert_eq!(x, [1000, -500, 250, -125, 63]);
    }

    #[test]
    fn test_find_best_pitch_simple() {
        // Construct a scenario where lag 2 has the strongest correlation
        let mut xcorr = [0i32; 5];
        xcorr[0] = 10;
        xcorr[1] = 20;
        xcorr[2] = 100; // Best
        xcorr[3] = 50;
        xcorr[4] = 5;

        // Uniform energy so Syy doesn't matter much
        let y = [100i32; 10];
        let mut best_pitch = [0i32; 2];
        find_best_pitch(&xcorr, &y, 5, 5, &mut best_pitch, 0, 100);
        assert_eq!(best_pitch[0], 2); // Strongest
    }
}
