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
    //
    // Saturation note: the C reference's x86 SSE4.1 path (celt_fir_sse4_1)
    // uses _mm_packs_epi32 which saturates to the full int16_t range
    // [-32768, 32767].  The C scalar path uses SROUND16 which saturates to
    // [-32767, 32767] (via SATURATE(x, 32767)).  On x86, the SSE4.1 path
    // is always taken.  We match the SSE4.1 behaviour by clamping to
    // [-32768, 32767] so that bit-exact output is achieved on the x86
    // platform where the C reference is built and tested.
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

        y[i] = pshr32(sum[0], SIG_SHIFT).clamp(-32768, 32767);
        y[i + 1] = pshr32(sum[1], SIG_SHIFT).clamp(-32768, 32767);
        y[i + 2] = pshr32(sum[2], SIG_SHIFT).clamp(-32768, 32767);
        y[i + 3] = pshr32(sum[3], SIG_SHIFT).clamp(-32768, 32767);
        i += 4;
    }

    // Scalar tail — also matches SSE4.1's SATURATE16 [-32768, 32767]
    while i < n {
        let mut sum = shl32(extend32(x[i + ord]), SIG_SHIFT);
        for j in 0..ord {
            sum = mac16_16(sum, rnum[j], x[i + j]);
        }
        y[i] = pshr32(sum, SIG_SHIFT).clamp(-32768, 32767);
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

    // ===================================================================
    // celt_lpc tests (P0)
    // ===================================================================

    #[test]
    fn test_lpc_zero_ac0() {
        // ac[0] = 0 => all zeros output (guard at line 27)
        let mut lpc_out = [1234; 4];
        let ac = [0; 5];
        celt_lpc(&mut lpc_out, &ac, 4);
        assert_eq!(lpc_out, [0, 0, 0, 0]);
    }

    #[test]
    fn test_lpc_white_noise() {
        // Delta autocorrelation: signal is uncorrelated, no prediction possible
        let mut lpc_out = [9999; 4];
        let ac = [1 << 20, 0, 0, 0, 0];
        celt_lpc(&mut lpc_out, &ac, 4);
        for (i, &coeff) in lpc_out.iter().enumerate() {
            assert_eq!(
                coeff, 0,
                "lpc_out[{i}] should be 0 for white noise, got {coeff}"
            );
        }
    }

    #[test]
    fn test_lpc_order_1_ar() {
        // First-order Levinson: r = -ac[1]/ac[0]
        // ac = [100<<6, 50<<6] => r = -0.5 => Q12 = -2048
        let mut lpc_out = [0; 1];
        let ac = [100 << 6, 50 << 6];
        celt_lpc(&mut lpc_out, &ac, 1);
        assert!(
            (lpc_out[0] - (-2048)).abs() <= 2,
            "order-1 LPC: expected ~-2048, got {}",
            lpc_out[0]
        );
    }

    #[test]
    fn test_lpc_dc_signal() {
        // Flat autocorrelation (all equal) represents DC.
        // The LPC analysis filter uses sign convention where prediction is subtracted,
        // so first coefficient should have magnitude ~4096 (Q12 for 1.0).
        let n = 1 << 20;
        let ac = [n, n, n, n, n];
        let mut lpc_out = [0; 4];
        celt_lpc(&mut lpc_out, &ac, 4);
        assert!(
            lpc_out[0].abs() >= 3900,
            "DC signal: |lpc[0]| expected near 4096, got {}",
            lpc_out[0]
        );
    }

    #[test]
    fn test_lpc_convergence_bailout() {
        // Strongly correlated: error drops below ac[0]>>10 quickly.
        // After bailout, later coefficients should stay near zero.
        let mut lpc_out = [0; 4];
        let ac = [1 << 24, (1 << 24) - 1, 0, 0, 0];
        celt_lpc(&mut lpc_out, &ac, 4);
        assert!(
            lpc_out[0].abs() > 0,
            "first coefficient should be non-zero after bailout"
        );
        // Coefficients beyond the first should be near zero since the bailout
        // stops iteration once 30dB gain is reached
        for i in 2..4 {
            assert!(
                lpc_out[i].abs() <= 10,
                "lpc_out[{i}]={} should be near zero after early bailout",
                lpc_out[i]
            );
        }
    }

    #[test]
    fn test_lpc_max_order() {
        // Full order = CELT_LPC_ORDER = 24
        let mut ac = [0i32; CELT_LPC_ORDER + 1];
        ac[0] = 1 << 24;
        for k in 1..=CELT_LPC_ORDER {
            ac[k] = (ac[k - 1] as i64 * 900 / 1000) as i32;
        }
        let mut lpc_out = [0i32; CELT_LPC_ORDER];
        celt_lpc(&mut lpc_out, &ac, CELT_LPC_ORDER);
        assert!(
            lpc_out[0] != 0,
            "max order should produce non-zero coefficients"
        );
        for (i, &c) in lpc_out.iter().enumerate() {
            assert!(c.abs() <= 32767, "lpc_out[{i}] = {c} exceeds Q12 i16 range");
        }
    }

    #[test]
    fn test_lpc_bw_expansion_fallback() {
        // Pathological ac designed to produce huge coefficients,
        // forcing the 10-iteration BW expansion fallback to A(z)=1
        let mut ac = [0i32; 5];
        ac[0] = 1 << 24;
        ac[1] = ac[0] - 1;
        ac[2] = ac[0] - 2;
        ac[3] = ac[0] - 3;
        ac[4] = ac[0] - 4;
        let mut lpc_out = [0i32; 4];
        celt_lpc(&mut lpc_out, &ac, 4);
        let all_fit = lpc_out.iter().all(|&c| c.abs() <= 32767);
        let is_fallback = lpc_out[0] == 4096 && lpc_out[1..].iter().all(|&c| c == 0);
        assert!(
            all_fit || is_fallback,
            "coefficients should fit in Q12 or be A(z)=1 fallback"
        );
    }

    #[test]
    fn test_lpc_nonzero_autocorr_emits_prediction() {
        let mut lpc_out = [0; 2];
        let ac = [1 << 20, 1 << 18, 0];
        celt_lpc(&mut lpc_out, &ac, 2);
        assert!(lpc_out[0] < 0);
        assert_ne!(lpc_out, [4096, 0]);
    }

    // ===================================================================
    // celt_autocorr tests (P1)
    // ===================================================================

    #[test]
    fn test_autocorr_zero_signal_odd_length() {
        let x = [0; 5];
        let mut ac = [1; 3];
        let shift = celt_autocorr(&x, &mut ac, None, 0, 2, 5);
        assert_eq!(shift, -28);
        assert_eq!(ac, [268_435_456, 0, 0]);
    }

    #[test]
    fn test_autocorr_zero_signal_with_window() {
        let x = [0; 6];
        let window = [32_767, 16_384];
        let mut ac = [1; 4];
        let shift = celt_autocorr(&x, &mut ac, Some(&window), 2, 3, 6);
        assert_eq!(shift, -28);
        assert_eq!(ac, [268_435_456, 0, 0, 0]);
    }

    #[test]
    #[should_panic(expected = "window required when overlap > 0")]
    fn test_autocorr_requires_window_for_overlap() {
        let x = [0; 4];
        let mut ac = [0; 2];
        let _ = celt_autocorr(&x, &mut ac, None, 1, 1, 4);
    }

    #[test]
    fn test_autocorr_dc() {
        let x = [1000i32; 64];
        let mut ac = [0i32; 5];
        let _shift = celt_autocorr(&x, &mut ac, None, 0, 4, 64);
        // R(0) >= R(k) for all k
        for k in 1..5 {
            assert!(
                ac[0] >= ac[k],
                "ac[0]={} should be >= ac[{k}]={}",
                ac[0],
                ac[k]
            );
        }
        // For DC, all lags nearly equal
        for k in 1..5 {
            let ratio = (ac[k] as f64) / (ac[0] as f64);
            assert!(
                ratio > 0.9,
                "DC: ac[{k}]/ac[0] = {ratio:.4}, expected close to 1.0"
            );
        }
    }

    #[test]
    fn test_autocorr_impulse() {
        let mut x = [0i32; 32];
        x[0] = 10000;
        let mut ac = [0i32; 5];
        let _shift = celt_autocorr(&x, &mut ac, None, 0, 4, 32);
        assert!(ac[0] > 0, "ac[0] should be positive for impulse");
        for k in 1..5 {
            assert_eq!(ac[k], 0, "ac[{k}] should be 0 for impulse, got {}", ac[k]);
        }
    }

    #[test]
    fn test_autocorr_symmetry_property() {
        // R(0) >= |R(k)| for all k
        let x: Vec<i32> = (0..64).map(|i| ((i * 137 + 42) % 500) - 250).collect();
        let mut ac = [0i32; 9];
        let _shift = celt_autocorr(&x, &mut ac, None, 0, 8, 64);
        for k in 1..9 {
            assert!(
                ac[0] >= ac[k].abs(),
                "R(0)={} should be >= |R({k})|={}",
                ac[0],
                ac[k].abs()
            );
        }
    }

    #[test]
    fn test_autocorr_normalization() {
        let x = [30000i32; 64];
        let mut ac = [0i32; 5];
        let _shift = celt_autocorr(&x, &mut ac, None, 0, 4, 64);
        // ac[0] should be normalized to [2^28, 2^30)
        assert!(ac[0] >= 268_435_456, "ac[0]={} should be >= 2^28", ac[0]);
        assert!(ac[0] < 1_073_741_824, "ac[0]={} should be < 2^30", ac[0]);
    }

    #[test]
    fn test_autocorr_shift_consistency() {
        // Test that normalization works across different amplitude levels.
        // Very small amplitudes (e.g. 1) are dominated by the internal bias term
        // and don't meaningfully exercise the shift logic.
        for &amplitude in &[100i32, 1000, 10000, 30000] {
            let x = vec![amplitude; 64];
            let mut ac = [0i32; 5];
            let shift = celt_autocorr(&x, &mut ac, None, 0, 4, 64);
            assert!(
                ac[0] >= 268_435_456,
                "amp={amplitude}: ac[0]={} too small",
                ac[0]
            );
            assert!(
                ac[0] < 1_073_741_824,
                "amp={amplitude}: ac[0]={} too large",
                ac[0]
            );
            assert!(
                shift.abs() < 64,
                "amp={amplitude}: shift={shift} unreasonable"
            );
        }
    }

    // ===================================================================
    // celt_fir tests (P1)
    // ===================================================================

    #[test]
    fn test_fir_zero_coefficients_passthrough() {
        let x = [10, 11, 12, 13, 14, 15, 16, 17];
        let num = [0, 0, 0];
        let mut y = [-1; 5];
        celt_fir(&x, &num, &mut y, 5, 3);
        assert_eq!(y, [13, 14, 15, 16, 17]);
    }

    #[test]
    fn test_fir_n_not_multiple_of_4() {
        for n in 5..=7 {
            let ord = 4;
            let mut x = vec![0i32; n + ord];
            x[ord] = 1;
            let num = vec![0i32; ord];
            let mut y = vec![0i32; n];
            celt_fir(&x, &num, &mut y, n, ord);
            assert_eq!(
                y[0],
                sround16(shl32(extend32(1), SIG_SHIFT), SIG_SHIFT),
                "n={n}: first output sample mismatch"
            );
        }
    }

    #[test]
    fn test_fir_matches_direct_convolution() {
        let ord = 5;
        let n = 12;
        let mut x = vec![0i32; n + ord];
        for i in 0..x.len() {
            x[i] = ((i as i32 * 73 + 17) % 200) - 100;
        }
        let num: Vec<i32> = (0..ord).map(|i| (i as i32 + 1) * 100).collect();

        let mut y_fast = vec![0i32; n];
        celt_fir(&x, &num, &mut y_fast, n, ord);

        // Naive convolution
        let mut rnum = vec![0i32; ord];
        for i in 0..ord {
            rnum[i] = num[ord - i - 1];
        }
        let mut y_naive = vec![0i32; n];
        for i in 0..n {
            let mut sum = shl32(extend32(x[i + ord]), SIG_SHIFT);
            for j in 0..ord {
                sum = mac16_16(sum, rnum[j], x[i + j]);
            }
            y_naive[i] = sround16(sum, SIG_SHIFT);
        }
        assert_eq!(
            y_fast, y_naive,
            "FIR fast path should match naive convolution"
        );
    }

    #[test]
    fn test_fir_delay() {
        let ord = 4;
        let n = 8;
        let mut x = vec![0i32; n + ord];
        x[ord] = 1;
        let mut num = vec![0i32; ord];
        num[0] = 4096;
        let mut y = vec![0i32; n];
        celt_fir(&x, &num, &mut y, n, ord);
        assert!(
            y.iter().any(|&v| v != 0),
            "impulse through non-zero filter should produce output"
        );
    }

    // ===================================================================
    // celt_iir tests (P0)
    // ===================================================================

    #[test]
    fn test_iir_zero_feedback_passthrough() {
        let x = [7, 8, 9, 10, 11];
        let den = [0, 0, 0, 0];
        let mut y = [-1; 5];
        let mut mem = [1, 2, 3, 4];
        celt_iir(&x, &den, &mut y, 5, 4, &mut mem);
        assert_eq!(y, x);
        assert_eq!(mem, [11, 10, 9, 8]);
    }

    #[test]
    fn test_iir_memory_update() {
        let x = [100, 200, 300, 400];
        let den = [0, 0, 0, 0];
        let mut y = [0i32; 4];
        let mut mem = [0i32; 4];
        celt_iir(&x, &den, &mut y, 4, 4, &mut mem);
        assert_eq!(mem[0], y[3], "mem[0] should be last output");
        assert_eq!(mem[1], y[2], "mem[1] should be second-to-last");
        assert_eq!(mem[2], y[1], "mem[2] should be third-to-last");
        assert_eq!(mem[3], y[0], "mem[3] should be fourth-to-last");
    }

    #[test]
    fn test_iir_n_not_multiple_of_4() {
        for n in [5, 7] {
            let den = [0, 0, 0, 0];
            let x: Vec<i32> = (0..n).map(|i| (i as i32 + 1) * 100).collect();
            let mut y = vec![0i32; n];
            let mut mem = [0i32; 4];
            celt_iir(&x, &den, &mut y, n, 4, &mut mem);
            assert_eq!(&y, &x[..n], "n={n}: zero-feedback IIR should pass through");
        }
    }

    #[test]
    fn test_iir_inter_sample_correction() {
        // Critical: 4-sample unrolled path with non-zero feedback must match naive IIR
        let ord = 4;
        let n = 8;
        let den = [500i32, -300, 200, -100];
        let x: Vec<i32> = (0..n).map(|i| (i as i32 + 1) << SIG_SHIFT).collect();

        let mut y_fast = vec![0i32; n];
        let mut mem_fast = [0i32; 4];
        celt_iir(&x, &den, &mut y_fast, n, ord, &mut mem_fast);

        // Naive sample-by-sample IIR
        let mut rden = vec![0i32; ord];
        for i in 0..ord {
            rden[i] = den[ord - i - 1];
        }
        let mut y_internal = vec![0i32; n + ord];
        let mut y_out_naive = vec![0i32; n];
        for i in 0..n {
            let mut sum = x[i];
            for j in 0..ord {
                sum -= mult16_16(rden[j], y_internal[i + j]);
            }
            y_internal[i + ord] = sround16(sum, SIG_SHIFT);
            y_out_naive[i] = sum;
        }
        assert_eq!(
            y_fast, y_out_naive,
            "IIR unrolled must match naive sample-by-sample"
        );
    }

    #[test]
    fn test_iir_simple_first_order() {
        let den = [1000i32, 0, 0, 0];
        let n = 8;
        let mut x = vec![0i32; n];
        x[0] = 1 << SIG_SHIFT;
        let mut y = vec![0i32; n];
        let mut mem = [0i32; 4];
        celt_iir(&x, &den, &mut y, n, 4, &mut mem);
        assert_eq!(y[0], x[0], "first sample should be the impulse");
        assert!(
            y[1..].iter().any(|&v| v != 0),
            "feedback should produce non-zero tail"
        );
    }

    // ===================================================================
    // Stage 4 branch coverage
    // ===================================================================
    mod branch_coverage_stage4 {
        use super::*;

        // celt_lpc: drive the bandwidth-expansion path by crafting an
        // autocorrelation that produces a huge coefficient on the first pass
        // (line 72: maxabs_val > 32767 branch taken; line 93: !converged edge).
        #[test]
        fn test_bc_lpc_bw_expansion_iterations() {
            // Sweep a range of near-singular autocorrelations; each drives
            // the inner bandwidth-expansion loop at least once.
            for gap in 0..8i32 {
                let mut ac = [0i32; 5];
                ac[0] = 1 << 24;
                for i in 1..5 {
                    ac[i] = ac[0] - (gap * (i as i32));
                }
                let mut lpc_out = [0i32; 4];
                celt_lpc(&mut lpc_out, &ac, 4);
                // Non-crash is the assertion.
                let _ = lpc_out;
            }
        }

        // celt_fir: small block (n small), fixed valid ord, zero coefficients.
        #[test]
        fn test_bc_fir_small_block_sweep() {
            for n in 1..=8 {
                let ord = 4;
                let mut x = vec![0i32; n + ord];
                for i in 0..x.len() {
                    x[i] = (i as i32) - (ord as i32);
                }
                let num = vec![0i32; ord];
                let mut y = vec![0i32; n];
                celt_fir(&x, &num, &mut y, n, ord);
                // Zero coefficients: output == x[ord..ord+n]
                for i in 0..n {
                    assert_eq!(y[i], x[i + ord], "n={n} i={i}");
                }
            }
        }

        // celt_iir sweep: zero-feedback, fixed valid ord, varying n (>= ord).
        #[test]
        fn test_bc_iir_small_block_sweep() {
            let ord = 4;
            for n in [ord, ord + 1, ord + 2, ord + 3, ord + 4, 12, 16] {
                let den = vec![0i32; ord];
                let x: Vec<i32> = (0..n).map(|i| (i as i32 + 1) * 10).collect();
                let mut y = vec![0i32; n];
                let mut mem = vec![0i32; ord];
                celt_iir(&x, &den, &mut y, n, ord, &mut mem);
                assert_eq!(&y, &x, "n={n}");
            }
        }

        // celt_fir / celt_iir: zero-coefficient + zero-input sanity.
        #[test]
        fn test_bc_fir_iir_all_zero_input() {
            let n = 8;
            let ord = 4;
            let x = vec![0i32; n + ord];
            let num = vec![0i32; ord];
            let mut y = vec![0i32; n];
            celt_fir(&x, &num, &mut y, n, ord);
            assert!(y.iter().all(|&v| v == 0));

            let den = vec![0i32; ord];
            let xi = vec![0i32; n];
            let mut yi = vec![0i32; n];
            let mut mem = vec![0i32; ord];
            celt_iir(&xi, &den, &mut yi, n, ord, &mut mem);
            assert!(yi.iter().all(|&v| v == 0));
        }

        // celt_lpc: trivially stable across a range of orders
        #[test]
        fn test_bc_lpc_order_sweep_stable() {
            for order in [1usize, 2, 4, 8, 12, 16, CELT_LPC_ORDER] {
                let mut ac = vec![0i32; order + 1];
                ac[0] = 1 << 20;
                ac[1] = 1 << 18;
                let mut lpc_out = vec![0i32; order];
                celt_lpc(&mut lpc_out, &ac, order);
                for (i, &c) in lpc_out.iter().enumerate() {
                    assert!(c.abs() <= 32767, "order={order} i={i}: {c}");
                }
            }
        }
    }
}
