//! CELT LPC analysis — Levinson-Durbin and autocorrelation.
//!
//! Matches the C reference: `celt_lpc.c`, `celt_lpc.h` (fixed-point path only,
//! OPUS_FAST_INT64 enabled). All functions produce bit-exact output.

use super::math_ops::{celt_ilog2, frac_div32};
use super::pitch::{celt_pitch_xcorr, xcorr_kernel};
use crate::types::*;

// ===========================================================================
// Levinson-Durbin LPC analysis
// ===========================================================================

/// Compute LPC coefficients from autocorrelation via Levinson-Durbin recursion.
/// Matches C `_celt_lpc` (FIXED_POINT + OPUS_FAST_INT64 path).
///
/// - `lpc_out`: output LPC coefficients in Q12, length >= `p`
/// - `ac`: autocorrelation values, length >= `p + 1`
/// - `p`: LPC order (must be <= CELT_LPC_ORDER = 24)
pub fn celt_lpc(lpc_out: &mut [i32], ac: &[i32], p: usize) {
    debug_assert!(p <= CELT_LPC_ORDER);

    // Internal Q25 LPC coefficients
    let mut lpc = [0i32; CELT_LPC_ORDER];
    let mut error = ac[0];

    if ac[0] != 0 {
        for i in 0..p {
            // Sum up this iteration's reflection coefficient (OPUS_FAST_INT64 path)
            let mut acc: i64 = 0;
            for j in 0..i {
                acc += lpc[j] as i64 * ac[i - j] as i64;
            }
            let mut rr = (acc >> 31) as i32;
            rr += shr32(ac[i + 1], 6);
            let r = -frac_div32(shl32(rr, 6), error);

            // Update LPC coefficients and total error
            lpc[i] = shr32(r, 6);
            for j in 0..((i + 1) >> 1) {
                let tmp1 = lpc[j];
                let tmp2 = lpc[i - 1 - j];
                lpc[j] = tmp1 + mult32_32_q31(r, tmp2);
                lpc[i - 1 - j] = tmp2 + mult32_32_q31(r, tmp1);
            }

            error -= mult32_32_q31(mult32_32_q31(r, r), error);

            // Bail out once we get 30 dB gain
            if error <= shr32(ac[0], 10) {
                break;
            }
        }
    }

    // Convert Q25 LPC to Q12 output with bandwidth expansion if needed.
    // This reuses the logic in silk_LPC_fit() and silk_bwexpander_32().
    let mut idx: usize = 0;
    let mut converged = false;

    for _iter in 0..10 {
        let mut maxabs_val: i32 = 0;
        for i in 0..p {
            let absval = abs32(lpc[i]);
            if absval > maxabs_val {
                maxabs_val = absval;
                idx = i;
            }
        }
        maxabs_val = pshr32(maxabs_val, 13); // Q25 → Q12

        if maxabs_val > 32767 {
            maxabs_val = min32(maxabs_val, 163838);
            let mut chirp_q16 = qconst32(0.999, 16)
                - div32(
                    shl32(maxabs_val - 32767, 14),
                    shr32(mult32_32_32(maxabs_val, idx as i32 + 1), 2),
                );
            let chirp_minus_one_q16 = chirp_q16 - 65536;

            // Apply bandwidth expansion
            for i in 0..(p - 1) {
                lpc[i] = mult32_32_q16(chirp_q16, lpc[i]);
                chirp_q16 += pshr32(mult32_32_32(chirp_q16, chirp_minus_one_q16), 16);
            }
            lpc[p - 1] = mult32_32_q16(chirp_q16, lpc[p - 1]);
        } else {
            converged = true;
            break;
        }
    }

    if !converged {
        // Coefficients don't fit in 16-bit after 10 iterations — fall back to A(z)=1
        for i in 0..p {
            lpc_out[i] = 0;
        }
        lpc_out[0] = 4096; // 1.0 in Q12
    } else {
        for i in 0..p {
            lpc_out[i] = extract16(pshr32(lpc[i], 13)); // Q25 → Q12
        }
    }
}

// ===========================================================================
// Autocorrelation
// ===========================================================================

/// Compute autocorrelation of `x` with optional windowing.
/// Matches C `_celt_autocorr` (FIXED_POINT path).
///
/// - `x`: input signal, length >= `n`
/// - `ac`: output autocorrelation, length >= `lag + 1`
/// - `window`: optional symmetric window, length >= `overlap`
/// - `overlap`: number of windowed samples at each end (0 = no windowing)
/// - `lag`: maximum autocorrelation lag
/// - `n`: signal length
///
/// Returns the cumulative shift applied for normalization.
pub fn celt_autocorr(
    x: &[i32],
    ac: &mut [i32],
    window: Option<&[i32]>,
    overlap: usize,
    lag: usize,
    n: usize,
) -> i32 {
    debug_assert!(n > 0);
    let fast_n = n - lag;

    // Working buffer — always copy so we can apply windowing and/or shift in-place
    let mut xx: Vec<i32> = x[..n].to_vec();

    if overlap > 0 {
        let window = window.expect("window required when overlap > 0");
        for i in 0..overlap {
            let w = extract16(window[i]); // COEF2VAL16
            xx[i] = mult16_16_q15(x[i], w);
            xx[n - i - 1] = mult16_16_q15(x[n - i - 1], w);
        }
    }

    // Fixed-point: compute shift to prevent overflow in correlation
    let ac0_shift = celt_ilog2(n as i32 + ((n as i32) >> 4));
    let mut ac0: i32 = 1 + ((n as i32) << 7);
    if (n & 1) != 0 {
        ac0 += shr32(mult16_16(xx[0], xx[0]), ac0_shift);
    }
    let mut i = n & 1;
    while i < n {
        ac0 += shr32(mult16_16(xx[i], xx[i]), ac0_shift);
        ac0 += shr32(mult16_16(xx[i + 1], xx[i + 1]), ac0_shift);
        i += 2;
    }
    // Consider the effect of rounding-to-nearest when scaling the signal
    ac0 += shr32(ac0, 7);

    let mut shift = (celt_ilog2(ac0) - 30 + ac0_shift + 1) / 2;
    if shift > 0 {
        for i in 0..n {
            xx[i] = pshr32(xx[i], shift);
        }
    } else {
        shift = 0;
    }

    // Main correlation via celt_pitch_xcorr
    let _maxcorr = celt_pitch_xcorr(&xx, &xx, ac, fast_n, lag + 1);

    // Handle the tail: partial overlap that celt_pitch_xcorr doesn't cover
    for k in 0..=lag {
        let mut d: i32 = 0;
        for i in (k + fast_n)..n {
            d = mac16_16(d, xx[i], xx[i - k]);
        }
        ac[k] += d;
    }

    // Normalize ac[0] to [2^28, 2^30) range
    shift = 2 * shift;
    if shift <= 0 {
        ac[0] += shl32(1, -shift);
    }
    if ac[0] < 268435456 {
        // < 2^28: shift left to fill range
        let shift2 = 29 - ec_ilog(ac[0] as u32);
        for i in 0..=lag {
            ac[i] = shl32(ac[i], shift2);
        }
        shift -= shift2;
    } else if ac[0] >= 536870912 {
        // >= 2^29: shift right
        let mut shift2 = 1;
        if ac[0] >= 1073741824 {
            shift2 += 1;
        }
        for i in 0..=lag {
            ac[i] = shr32(ac[i], shift2);
        }
        shift += shift2;
    }

    shift
}

// ===========================================================================
// FIR filter
// ===========================================================================

/// Apply an FIR filter to the input signal.
/// Matches C `celt_fir_c` (FIXED_POINT path, non-SMALL_FOOTPRINT).
///
/// - `x`: input signal, length >= `n + ord` (the `ord` samples before index 0
///   are the filter memory, i.e. `x[i - ord..i]` must be valid for all i in 0..n).
///   In practice the caller provides a pointer into a larger buffer so that
///   `x[-ord..-1]` are the previous samples.  We model this by taking `x` as a
///   slice starting `ord` positions before the first "real" sample — the first
///   real sample is at `x[ord]`.
/// - `num`: FIR coefficients, length >= `ord`
/// - `y`: output buffer, length >= `n` (must NOT alias `x`)
/// - `n`: number of output samples
/// - `ord`: filter order
///
/// The function processes 4 samples at a time using `xcorr_kernel`, with a
/// scalar tail for the remaining 0–3 samples.
pub fn celt_fir(x: &[i32], num: &[i32], y: &mut [i32], n: usize, ord: usize) {
    // Reverse the coefficients for correlation-style processing
    let mut rnum = vec![0i32; ord];
    for i in 0..ord {
        rnum[i] = num[ord - i - 1];
    }

    // Unrolled 4-at-a-time loop
    let mut i = 0;
    while i + 3 < n {
        let mut sum = [0i32; 4];
        sum[0] = shl32(extend32(x[i + ord]), SIG_SHIFT);
        sum[1] = shl32(extend32(x[i + ord + 1]), SIG_SHIFT);
        sum[2] = shl32(extend32(x[i + ord + 2]), SIG_SHIFT);
        sum[3] = shl32(extend32(x[i + ord + 3]), SIG_SHIFT);

        // xcorr_kernel(rnum, x + i + ord - ord, sum, ord)
        // = xcorr_kernel(rnum, x + i, sum, ord)
        xcorr_kernel(&rnum, &x[i..], &mut sum, ord);

        y[i] = sround16(sum[0], SIG_SHIFT);
        y[i + 1] = sround16(sum[1], SIG_SHIFT);
        y[i + 2] = sround16(sum[2], SIG_SHIFT);
        y[i + 3] = sround16(sum[3], SIG_SHIFT);
        i += 4;
    }

    // Scalar tail
    while i < n {
        let mut sum = shl32(extend32(x[i + ord]), SIG_SHIFT);
        for j in 0..ord {
            sum = mac16_16(sum, rnum[j], x[i + j]);
        }
        y[i] = sround16(sum, SIG_SHIFT);
        i += 1;
    }
}

// ===========================================================================
// IIR filter
// ===========================================================================

/// Apply an IIR filter to the input signal.
/// Matches C `celt_iir` (FIXED_POINT path, non-SMALL_FOOTPRINT).
///
/// - `x`: input signal in Q(SIG_SHIFT), length >= `n`
/// - `den`: denominator (feedback) coefficients, length >= `ord`
///   (order must be a multiple of 4)
/// - `y_out`: output buffer, length >= `n`
/// - `n`: number of samples
/// - `ord`: filter order (must be divisible by 4)
/// - `mem`: filter memory / state buffer, length >= `ord`.
///   On entry, contains the previous `ord` output samples (most recent first).
///   On exit, updated with the last `ord` output samples (most recent first).
///
/// Note: in the C reference, `_x` and `_y` may alias (in-place operation).
/// Here, `x` and `y_out` may refer to the same data — the caller can copy
/// the input beforehand if needed, or the function handles it by reading
/// `x[i]` before writing `y_out[i]`.
pub fn celt_iir(x: &[i32], den: &[i32], y_out: &mut [i32], n: usize, ord: usize, mem: &mut [i32]) {
    debug_assert!(ord & 3 == 0, "celt_iir: order must be a multiple of 4");

    // Reverse denominator coefficients
    let mut rden = vec![0i32; ord];
    for i in 0..ord {
        rden[i] = den[ord - i - 1];
    }

    // Internal y buffer: first `ord` entries are negated memory (for xcorr_kernel
    // which adds rather than subtracts), followed by `n` entries for new output.
    let mut y = vec![0i32; n + ord];
    for i in 0..ord {
        y[i] = -mem[ord - i - 1];
    }
    // y[ord..n+ord] is already zeroed by vec!

    // Unrolled 4-at-a-time loop
    let mut i = 0;
    while i + 3 < n {
        let mut sum = [0i32; 4];
        sum[0] = x[i];
        sum[1] = x[i + 1];
        sum[2] = x[i + 2];
        sum[3] = x[i + 3];

        // xcorr_kernel adds rden[j]*y[i+j] into sum — since y[] is negated,
        // this effectively subtracts the feedback.
        xcorr_kernel(&rden, &y[i..], &mut sum, ord);

        // Patch up: account for the IIR feedback between the 4 outputs that
        // xcorr_kernel couldn't know about (it treated them as FIR).
        y[i + ord] = -sround16(sum[0], SIG_SHIFT);
        y_out[i] = sum[0];

        sum[1] = mac16_16(sum[1], y[i + ord], den[0]);
        y[i + ord + 1] = -sround16(sum[1], SIG_SHIFT);
        y_out[i + 1] = sum[1];

        sum[2] = mac16_16(sum[2], y[i + ord + 1], den[0]);
        sum[2] = mac16_16(sum[2], y[i + ord], den[1]);
        y[i + ord + 2] = -sround16(sum[2], SIG_SHIFT);
        y_out[i + 2] = sum[2];

        sum[3] = mac16_16(sum[3], y[i + ord + 2], den[0]);
        sum[3] = mac16_16(sum[3], y[i + ord + 1], den[1]);
        sum[3] = mac16_16(sum[3], y[i + ord], den[2]);
        y[i + ord + 3] = -sround16(sum[3], SIG_SHIFT);
        y_out[i + 3] = sum[3];

        i += 4;
    }

    // Scalar tail
    while i < n {
        let mut sum = x[i];
        for j in 0..ord {
            sum -= mult16_16(rden[j], y[i + j]);
        }
        y[i + ord] = sround16(sum, SIG_SHIFT);
        y_out[i] = sum;
        i += 1;
    }

    // Update memory: last `ord` outputs, most recent first
    for i in 0..ord {
        mem[i] = y_out[n - i - 1];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn celt_lpc_zero_autocorr_falls_back_to_unit_filter() {
        let mut lpc_out = [1234; 4];
        let ac = [0; 5];

        celt_lpc(&mut lpc_out, &ac, 4);

        assert_eq!(lpc_out, [0, 0, 0, 0]);
    }

    #[test]
    fn celt_lpc_nonzero_autocorr_emits_prediction_coefficients() {
        let mut lpc_out = [0; 2];
        let ac = [1 << 20, 1 << 18, 0];

        celt_lpc(&mut lpc_out, &ac, 2);

        assert!(lpc_out[0] < 0);
        assert_ne!(lpc_out, [4096, 0]);
    }

    #[test]
    fn celt_autocorr_zero_signal_odd_length_normalizes_and_hits_tail() {
        let x = [0; 5];
        let mut ac = [1; 3];

        let shift = celt_autocorr(&x, &mut ac, None, 0, 2, 5);

        assert_eq!(shift, -28);
        assert_eq!(ac, [268_435_456, 0, 0]);
    }

    #[test]
    fn celt_autocorr_zero_signal_with_window_hits_window_branch() {
        let x = [0; 6];
        let window = [32_767, 16_384];
        let mut ac = [1; 4];

        let shift = celt_autocorr(&x, &mut ac, Some(&window), 2, 3, 6);

        assert_eq!(shift, -28);
        assert_eq!(ac, [268_435_456, 0, 0, 0]);
    }

    #[test]
    #[should_panic(expected = "window required when overlap > 0")]
    fn celt_autocorr_requires_window_for_overlap() {
        let x = [0; 4];
        let mut ac = [0; 2];

        let _ = celt_autocorr(&x, &mut ac, None, 1, 1, 4);
    }

    #[test]
    fn celt_fir_zero_coefficients_copy_input_and_use_scalar_tail() {
        let x = [10, 11, 12, 13, 14, 15, 16, 17];
        let num = [0, 0, 0];
        let mut y = [-1; 5];

        celt_fir(&x, &num, &mut y, 5, 3);

        assert_eq!(y, [13, 14, 15, 16, 17]);
    }

    #[test]
    fn celt_iir_zero_feedback_copy_input_and_update_memory() {
        let x = [7, 8, 9, 10, 11];
        let den = [0, 0, 0, 0];
        let mut y = [-1; 5];
        let mut mem = [1, 2, 3, 4];

        celt_iir(&x, &den, &mut y, 5, 4, &mut mem);

        assert_eq!(y, x);
        assert_eq!(mem, [11, 10, 9, 8]);
    }
}
