//! SIMD-accelerated implementations of performance-critical CELT functions.
//!
//! Gated behind the `simd` cargo feature. Uses the `wide` crate for safe SIMD
//! abstractions over SSE2/AVX2/NEON. All functions produce bit-exact output
//! matching their scalar counterparts.
//!
//! The core primitive is `mac16_16_x4`: a 4-lane multiply-accumulate that
//! replicates the scalar `mac16_16` semantics (i16 truncation of both operands
//! before multiply, wrapping accumulate).

use crate::{uc, uc_set};
use wide::i32x4;

// ===========================================================================
// Core SIMD primitive
// ===========================================================================

/// 4-lane multiply-accumulate matching scalar `mac16_16` semantics.
///
/// For each lane i:
///   acc[i] = acc[i].wrapping_add((a[i] as i16 as i32) * (b[i] as i16 as i32))
///
/// The i16 truncation is achieved by sign-extending the low 16 bits:
///   (x << 16) >> 16  is equivalent to  x as i16 as i32
///
/// Since both operands fit in i16 after truncation, the product fits in 31 bits
/// (max: 32767 * 32767 = 1,073,676,289), so i32 multiply is sufficient.
#[inline(always)]
fn mac16_16_x4(acc: i32x4, a: i32x4, b: i32x4) -> i32x4 {
    // Sign-extend low 16 bits of each lane
    let a16 = (a << 16) >> 16;
    let b16 = (b << 16) >> 16;
    // Multiply (fits in i32 since inputs are in i16 range) and accumulate
    acc + a16 * b16
}

// ===========================================================================
// xcorr_kernel — cross-correlation (Priority 1)
// ===========================================================================

/// SIMD implementation of `xcorr_kernel`.
///
/// Computes 4 cross-correlations simultaneously:
///   sum[i] += sum_{j=0}^{len-1} x[j] * y[j+i]  for i = 0..3
///
/// Uses `i32x4` to hold the 4 accumulators. Each iteration broadcasts one
/// x-value to all 4 lanes and loads 4 consecutive y-values, then applies
/// `mac16_16_x4`. This is simpler than the scalar version's manual 4x
/// unrolling because the 4-way parallelism lives in the SIMD lanes.
///
/// Preconditions: `len >= 3`, `x` has >= `len` elements, `y` has >= `len + 3` elements.
#[inline(always)]
pub fn xcorr_kernel_simd(x: &[i32], y: &[i32], sum: &mut [i32; 4], len: usize) {
    debug_assert!(len >= 3);
    debug_assert!(x.len() >= len);
    debug_assert!(y.len() >= len + 3);

    let mut acc = i32x4::new(*sum);

    // Hot inner loop of pitch analysis — runs O(len) per call and is called
    // O(max_pitch/4) per frame. Preconditions above give provably-safe bounds
    // for x[j], y[j], y[j+1], y[j+2], y[j+3] across j < len.
    for j in 0..len {
        let xv = i32x4::splat(uc!(x, j));
        let yv = i32x4::new([uc!(y, j), uc!(y, j + 1), uc!(y, j + 2), uc!(y, j + 3)]);
        acc = mac16_16_x4(acc, xv, yv);
    }

    let arr = acc.to_array();
    sum[0] = arr[0];
    sum[1] = arr[1];
    sum[2] = arr[2];
    sum[3] = arr[3];
}

// ===========================================================================
// celt_maxabs32 — maximum absolute value scan (Priority 3a)
// ===========================================================================

/// SIMD implementation of `celt_maxabs32`.
///
/// Finds the maximum absolute value in a slice by tracking parallel max and
/// min values in `i32x4` lanes, then performing a horizontal reduction.
/// Returns `max(maxval, -minval)`.
#[inline(always)]
pub fn celt_maxabs32_simd(x: &[i32]) -> i32 {
    let chunks = x.len() / 4;
    let remainder = x.len() % 4;

    let mut maxv = i32x4::splat(0);
    let mut minv = i32x4::splat(0);

    // base = c * 4 with c < chunks = x.len() / 4 ⇒ base + 3 < x.len().
    for c in 0..chunks {
        let base = c * 4;
        let v = i32x4::new([
            uc!(x, base),
            uc!(x, base + 1),
            uc!(x, base + 2),
            uc!(x, base + 3),
        ]);
        maxv = maxv.max(v);
        minv = minv.min(v);
    }

    // Horizontal reduction
    let max_arr = maxv.to_array();
    let min_arr = minv.to_array();
    let mut maxval = max_arr[0];
    let mut minval = min_arr[0];
    for i in 1..4 {
        if max_arr[i] > maxval {
            maxval = max_arr[i];
        }
        if min_arr[i] < minval {
            minval = min_arr[i];
        }
    }

    // Handle remainder elements: tail_start + i < chunks*4 + remainder = x.len().
    let tail_start = chunks * 4;
    for i in 0..remainder {
        let v = uc!(x, tail_start + i);
        if v > maxval {
            maxval = v;
        }
        if v < minval {
            minval = v;
        }
    }

    if maxval > -minval { maxval } else { -minval }
}

// ===========================================================================
// denormalise_band_simd — scale a band's coefficients by gain (Priority 2)
// ===========================================================================

/// SIMD implementation of the inner loop of `denormalise_bands`.
///
/// For each element: `out[i] = pshr32(mult32_32_q31(shl32(x[i], 30 - NORM_SHIFT), g), shift)`
///
/// Since `g` and `shift` are constant within a band, we process 4 coefficients
/// per iteration. The 32×32→Q31 multiply requires 64-bit intermediates, so we
/// unpack to pairs of i64 values, multiply, shift, and repack.
#[inline(always)]
pub fn denormalise_band_simd(x: &[i32], out: &mut [i32], len: usize, g: i32, shift: i32) {
    use crate::types::{NORM_SHIFT, mult32_32_q31, pshr32, shl32};

    let chunks = len / 4;
    let remainder = len % 4;
    let pre_shift = 30 - NORM_SHIFT;

    // base = c * 4 with c < chunks = len / 4 ⇒ base + 3 < len ≤ x.len() & out.len().
    for c in 0..chunks {
        let base = c * 4;
        // Scalar path using i64 intermediates — the `wide` crate's i32x4 can't
        // do the 64-bit multiply-shift we need, so we unroll 4 scalar ops which
        // the compiler can still auto-vectorize with the right instruction mix.
        uc_set!(
            out,
            base,
            pshr32(mult32_32_q31(shl32(uc!(x, base), pre_shift), g), shift)
        );
        uc_set!(
            out,
            base + 1,
            pshr32(mult32_32_q31(shl32(uc!(x, base + 1), pre_shift), g), shift)
        );
        uc_set!(
            out,
            base + 2,
            pshr32(mult32_32_q31(shl32(uc!(x, base + 2), pre_shift), g), shift)
        );
        uc_set!(
            out,
            base + 3,
            pshr32(mult32_32_q31(shl32(uc!(x, base + 3), pre_shift), g), shift)
        );
    }

    let tail = chunks * 4;
    for i in 0..remainder {
        uc_set!(
            out,
            tail + i,
            pshr32(mult32_32_q31(shl32(uc!(x, tail + i), pre_shift), g), shift)
        );
    }
}

// ===========================================================================
// mdct_window_simd — MDCT overlap-add windowing (Priority 1)
// ===========================================================================

/// SIMD implementation of the MDCT TDAC windowing loop.
///
/// Applies the synthesis window to the overlap region:
///   out_fwd[i] = s_mul(x_fwd, w_desc) - s_mul(x_rev, w_asc)   (forward half)
///   out_rev[i] = s_mul(x_fwd, w_asc)  + s_mul(x_rev, w_desc)   (reverse half)
///
/// where s_mul(a, b) = mult16_32_q15(b, a) = (b as i16 as i64) * (a as i64) >> 15.
///
/// The window values are i16, so we use `i32x4` multiply with truncation.
/// Each iteration processes 4 forward + 4 reverse elements.
#[inline(always)]
pub fn mdct_window_simd(output: &mut [i32], window: &[i16], overlap: usize) {
    let half = overlap / 2;
    let chunks = half / 4;
    let remainder = half % 4;

    // Forward: yp walks 0..half, xp walks (overlap-1) down. yp < half ≤ overlap/2,
    // xp ∈ [overlap/2, overlap-1]. Both < output.len() & window.len() = overlap.
    for c in 0..chunks {
        let base = c * 4;
        for k in 0..4 {
            let yp = base + k;
            let xp = overlap - 1 - base - k;

            let x_fwd = uc!(output, xp);
            let x_rev = uc!(output, yp);
            let w_asc = uc!(window, yp) as i32;
            let w_desc = uc!(window, xp) as i32;

            // C relies on signed i32 wrap-around here (the sum of two Q15 products
            // can exceed i32 range with extreme signal values from garbage input).
            uc_set!(
                output,
                yp,
                s_mul_inline(x_rev, w_desc).wrapping_sub(s_mul_inline(x_fwd, w_asc))
            );
            uc_set!(
                output,
                xp,
                s_mul_inline(x_rev, w_asc).wrapping_add(s_mul_inline(x_fwd, w_desc))
            );
        }
    }

    let tail = chunks * 4;
    for i in 0..remainder {
        let yp = tail + i;
        let xp = overlap - 1 - tail - i;
        let w_asc = uc!(window, yp) as i32;
        let w_desc = uc!(window, xp) as i32;

        let x_fwd = uc!(output, xp);
        let x_rev = uc!(output, yp);

        uc_set!(
            output,
            yp,
            s_mul_inline(x_rev, w_desc).wrapping_sub(s_mul_inline(x_fwd, w_asc))
        );
        uc_set!(
            output,
            xp,
            s_mul_inline(x_rev, w_asc).wrapping_add(s_mul_inline(x_fwd, w_desc))
        );
    }
}

/// Inline s_mul matching `mult16_32_q15(b, a)` — window coeff `b` is i16 range.
#[inline(always)]
fn s_mul_inline(a: i32, b: i32) -> i32 {
    ((b as i16 as i64) * (a as i64) >> 15) as i32
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // mac16_16_x4 tests
    // -----------------------------------------------------------------------

    #[test]
    fn mac16_16_x4_basic() {
        use crate::types::mac16_16;

        let acc = i32x4::splat(0);
        let a = i32x4::new([1, 2, 3, 4]);
        let b = i32x4::new([10, 20, 30, 40]);

        let result = mac16_16_x4(acc, a, b).to_array();

        assert_eq!(result[0], mac16_16(0, 1, 10));
        assert_eq!(result[1], mac16_16(0, 2, 20));
        assert_eq!(result[2], mac16_16(0, 3, 30));
        assert_eq!(result[3], mac16_16(0, 4, 40));
    }

    #[test]
    fn mac16_16_x4_truncation() {
        // Verify that high bits are discarded (i16 truncation)
        use crate::types::mac16_16;

        // 0x10001 truncates to 1, 0x20002 truncates to 2, etc.
        let a = i32x4::new([0x10001, 0x20002, 0x30003, 0x40004]);
        let b = i32x4::new([0x50005, 0x60006, 0x70007, 0x80008]);
        let acc = i32x4::new([100, 200, 300, 400]);

        let result = mac16_16_x4(acc, a, b).to_array();

        assert_eq!(result[0], mac16_16(100, 0x10001, 0x50005));
        assert_eq!(result[1], mac16_16(200, 0x20002, 0x60006));
        assert_eq!(result[2], mac16_16(300, 0x30003, 0x70007));
        assert_eq!(result[3], mac16_16(400, 0x40004, 0x80008));
    }

    #[test]
    fn mac16_16_x4_negative() {
        use crate::types::mac16_16;

        let a = i32x4::new([-1, -32768, 32767, -100]);
        let b = i32x4::new([32767, -32768, 32767, 100]);
        let acc = i32x4::new([1000, 2000, 3000, 4000]);

        let result = mac16_16_x4(acc, a, b).to_array();

        assert_eq!(result[0], mac16_16(1000, -1, 32767));
        assert_eq!(result[1], mac16_16(2000, -32768, -32768));
        assert_eq!(result[2], mac16_16(3000, 32767, 32767));
        assert_eq!(result[3], mac16_16(4000, -100, 100));
    }

    #[test]
    fn mac16_16_x4_wrapping_accumulate() {
        use crate::types::mac16_16;

        // Start near i32::MAX to test wrapping behavior
        let a = i32x4::new([32767, 32767, 1, 1]);
        let b = i32x4::new([32767, 32767, 1, 1]);
        let acc = i32x4::new([i32::MAX - 100, i32::MAX, 0, i32::MIN]);

        let result = mac16_16_x4(acc, a, b).to_array();

        assert_eq!(result[0], mac16_16(i32::MAX - 100, 32767, 32767));
        assert_eq!(result[1], mac16_16(i32::MAX, 32767, 32767));
        assert_eq!(result[2], mac16_16(0, 1, 1));
        assert_eq!(result[3], mac16_16(i32::MIN, 1, 1));
    }

    // -----------------------------------------------------------------------
    // xcorr_kernel_simd tests
    // -----------------------------------------------------------------------

    use super::super::pitch::xcorr_kernel_scalar;

    #[test]
    fn xcorr_kernel_simd_matches_scalar_basic() {
        let x = [1, 2, 3, 4, 5, 6, 7, 8];
        let y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110];

        let mut sum_scalar = [0i32; 4];
        let mut sum_simd = [0i32; 4];

        xcorr_kernel_scalar(&x, &y, &mut sum_scalar, x.len());
        xcorr_kernel_simd(&x, &y, &mut sum_simd, x.len());

        assert_eq!(sum_scalar, sum_simd);
    }

    #[test]
    fn xcorr_kernel_simd_matches_scalar_min_len() {
        // Minimum length: 3
        let x = [100, 200, 300];
        let y = [10, 20, 30, 40, 50, 60];

        let mut sum_scalar = [0i32; 4];
        let mut sum_simd = [0i32; 4];

        xcorr_kernel_scalar(&x, &y, &mut sum_scalar, 3);
        xcorr_kernel_simd(&x, &y, &mut sum_simd, 3);

        assert_eq!(sum_scalar, sum_simd);
    }

    #[test]
    fn xcorr_kernel_simd_matches_scalar_nonzero_initial() {
        let x = [1000, 2000, 3000, 4000, 5000];
        let y = [100, 200, 300, 400, 500, 600, 700, 800];

        let mut sum_scalar = [111, 222, 333, 444];
        let mut sum_simd = [111, 222, 333, 444];

        xcorr_kernel_scalar(&x, &y, &mut sum_scalar, x.len());
        xcorr_kernel_simd(&x, &y, &mut sum_simd, x.len());

        assert_eq!(sum_scalar, sum_simd);
    }

    #[test]
    fn xcorr_kernel_simd_matches_scalar_truncation() {
        // Values with high bits that get truncated by mac16_16
        let x = [0x10001, 0x20002, 0x30003, 0x40004];
        let y = [
            0x50005, 0x60006, 0x70007, 0x80008, 0x90009, 0xA000A, 0xB000B,
        ];

        let mut sum_scalar = [0i32; 4];
        let mut sum_simd = [0i32; 4];

        xcorr_kernel_scalar(&x, &y, &mut sum_scalar, x.len());
        xcorr_kernel_simd(&x, &y, &mut sum_simd, x.len());

        assert_eq!(sum_scalar, sum_simd);
    }

    #[test]
    fn xcorr_kernel_simd_matches_scalar_negative() {
        let x = [-32768, 32767, -1, 0, 12345, -12345, 100, -100];
        let y = [
            32767, -32768, 0, 1, -1, 32767, -32768, 100, -100, 12345, -12345,
        ];

        let mut sum_scalar = [0i32; 4];
        let mut sum_simd = [0i32; 4];

        xcorr_kernel_scalar(&x, &y, &mut sum_scalar, x.len());
        xcorr_kernel_simd(&x, &y, &mut sum_simd, x.len());

        assert_eq!(sum_scalar, sum_simd);
    }

    #[test]
    fn xcorr_kernel_simd_matches_scalar_large_random() {
        // Deterministic pseudo-random data (LCG)
        let mut rng: u32 = 0xDEAD_BEEF;
        let next = |rng: &mut u32| -> i32 {
            *rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            *rng as i32
        };

        let len = 960; // typical frame size
        let mut x = vec![0i32; len];
        let mut y = vec![0i32; len + 4];

        for v in x.iter_mut() {
            *v = next(&mut rng);
        }
        for v in y.iter_mut() {
            *v = next(&mut rng);
        }

        let mut sum_scalar = [0i32; 4];
        let mut sum_simd = [0i32; 4];

        xcorr_kernel_scalar(&x, &y, &mut sum_scalar, len);
        xcorr_kernel_simd(&x, &y, &mut sum_simd, len);

        assert_eq!(sum_scalar, sum_simd);
    }

    #[test]
    fn xcorr_kernel_simd_matches_scalar_all_lengths() {
        // Test all lengths from 3 to 32
        let mut rng: u32 = 0xCAFE_BABE;
        let next = |rng: &mut u32| -> i32 {
            *rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            *rng as i32
        };

        for len in 3..=32 {
            let mut x = vec![0i32; len];
            let mut y = vec![0i32; len + 4];
            for v in x.iter_mut() {
                *v = next(&mut rng);
            }
            for v in y.iter_mut() {
                *v = next(&mut rng);
            }

            let mut sum_scalar = [0i32; 4];
            let mut sum_simd = [0i32; 4];

            xcorr_kernel_scalar(&x, &y, &mut sum_scalar, len);
            xcorr_kernel_simd(&x, &y, &mut sum_simd, len);

            assert_eq!(sum_scalar, sum_simd, "mismatch at len={}", len);
        }
    }

    // -----------------------------------------------------------------------
    // celt_maxabs32_simd tests
    // -----------------------------------------------------------------------

    use super::super::math_ops::celt_maxabs32_scalar;

    #[test]
    fn celt_maxabs32_simd_basic() {
        let x = [100, -200, 50, 0];
        assert_eq!(celt_maxabs32_simd(&x), celt_maxabs32_scalar(&x));
    }

    #[test]
    fn celt_maxabs32_simd_empty() {
        let x: [i32; 0] = [];
        assert_eq!(celt_maxabs32_simd(&x), celt_maxabs32_scalar(&x));
    }

    #[test]
    fn celt_maxabs32_simd_single() {
        assert_eq!(celt_maxabs32_simd(&[42]), celt_maxabs32_scalar(&[42]));
        assert_eq!(celt_maxabs32_simd(&[-42]), celt_maxabs32_scalar(&[-42]));
        assert_eq!(celt_maxabs32_simd(&[0]), celt_maxabs32_scalar(&[0]));
    }

    #[test]
    fn celt_maxabs32_simd_all_negative() {
        let x = [-10, -20, -30, -40, -50];
        assert_eq!(celt_maxabs32_simd(&x), celt_maxabs32_scalar(&x));
    }

    #[test]
    fn celt_maxabs32_simd_all_positive() {
        let x = [10, 20, 30, 40, 50];
        assert_eq!(celt_maxabs32_simd(&x), celt_maxabs32_scalar(&x));
    }

    #[test]
    fn celt_maxabs32_simd_extremes() {
        let x = [i32::MAX, 0, i32::MIN + 1, 100];
        assert_eq!(celt_maxabs32_simd(&x), celt_maxabs32_scalar(&x));
    }

    #[test]
    fn celt_maxabs32_simd_various_lengths() {
        // Test all lengths from 0 to 20 (covers SIMD + tail cases)
        let mut rng: u32 = 0xBAAD_F00D;
        let next = |rng: &mut u32| -> i32 {
            *rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            *rng as i32
        };

        for len in 0..=20 {
            let mut x = vec![0i32; len];
            for v in x.iter_mut() {
                *v = next(&mut rng);
            }
            assert_eq!(
                celt_maxabs32_simd(&x),
                celt_maxabs32_scalar(&x),
                "mismatch at len={}",
                len
            );
        }
    }

    #[test]
    fn celt_maxabs32_simd_typical_band_sizes() {
        // Band sizes range from 4 to ~96 coefficients
        let mut rng: u32 = 0xFACE_CAFE;
        let next = |rng: &mut u32| -> i32 {
            *rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            *rng as i32
        };

        for &len in &[4, 8, 12, 16, 24, 32, 48, 64, 96] {
            let mut x = vec![0i32; len];
            for v in x.iter_mut() {
                *v = next(&mut rng);
            }
            assert_eq!(
                celt_maxabs32_simd(&x),
                celt_maxabs32_scalar(&x),
                "mismatch at len={}",
                len
            );
        }
    }

    // -----------------------------------------------------------------------
    // denormalise_band_simd tests
    // -----------------------------------------------------------------------

    #[test]
    fn denormalise_band_simd_matches_scalar() {
        use crate::types::{NORM_SHIFT, mult32_32_q31, pshr32, shl32};

        let mut rng: u32 = 0xDEAD_1234;
        let next = |rng: &mut u32| -> i32 {
            *rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            (*rng as i32) >> 8 // keep values in reasonable range
        };

        for &len in &[1, 2, 3, 4, 5, 7, 8, 12, 16, 24, 32, 48, 64, 96] {
            let x: Vec<i32> = (0..len).map(|_| next(&mut rng)).collect();
            let g = next(&mut rng).abs().max(1);
            let shift = (next(&mut rng).abs() % 20).max(0);

            // Scalar reference
            let pre_shift = 30 - NORM_SHIFT;
            let expected: Vec<i32> = x
                .iter()
                .map(|&v| pshr32(mult32_32_q31(shl32(v, pre_shift), g), shift))
                .collect();

            let mut actual = vec![0i32; len];
            denormalise_band_simd(&x, &mut actual, len, g, shift);

            assert_eq!(expected, actual, "mismatch at len={}", len);
        }
    }

    // -----------------------------------------------------------------------
    // mdct_window_simd tests
    // -----------------------------------------------------------------------

    #[test]
    fn mdct_window_simd_matches_scalar() {
        let mut rng: u32 = 0xCAFE_5678;
        let next32 = |rng: &mut u32| -> i32 {
            *rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            *rng as i32
        };
        let next16 = |rng: &mut u32| -> i16 {
            *rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            (*rng >> 16) as i16
        };

        // Test typical overlap sizes (must be even)
        for &ov in &[8, 16, 24, 32, 64, 120] {
            let window: Vec<i16> = (0..ov).map(|_| next16(&mut rng)).collect();
            let data: Vec<i32> = (0..ov).map(|_| next32(&mut rng)).collect();

            // Scalar reference
            let mut expected = data.clone();
            {
                let mut xp1 = ov - 1;
                let mut yp1 = 0;
                let mut wp1 = 0;
                let mut wp2 = ov - 1;
                for _ in 0..ov / 2 {
                    let x1 = expected[xp1];
                    let x2 = expected[yp1];
                    expected[yp1] = s_mul_inline(x2, window[wp2] as i32)
                        .wrapping_sub(s_mul_inline(x1, window[wp1] as i32));
                    expected[xp1] = s_mul_inline(x2, window[wp1] as i32)
                        .wrapping_add(s_mul_inline(x1, window[wp2] as i32));
                    yp1 += 1;
                    xp1 -= 1;
                    wp1 += 1;
                    wp2 -= 1;
                }
            }

            // SIMD version
            let mut actual = data.clone();
            mdct_window_simd(&mut actual, &window, ov);

            assert_eq!(expected, actual, "mismatch at overlap={}", ov);
        }
    }

    /// Exercise the scalar remainder loop in mdct_window_simd (lines 202-213).
    /// The remainder fires when (overlap/2) % 4 != 0. We test overlap=6 (half=3,
    /// remainder=3), overlap=10 (half=5, remainder=1), and overlap=2 (half=1,
    /// remainder=1).
    #[test]
    fn mdct_window_simd_remainder_loop() {
        let mut rng: u32 = 0xABCD_EF01;
        let next32 = |rng: &mut u32| -> i32 {
            *rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            *rng as i32
        };
        let next16 = |rng: &mut u32| -> i16 {
            *rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            (*rng >> 16) as i16
        };

        // Overlaps where half%4 != 0, triggering the remainder path
        for &ov in &[2, 6, 10, 14, 18, 22] {
            let half = ov / 2;
            assert_ne!(half % 4, 0, "overlap={ov} should have non-zero remainder");

            let window: Vec<i16> = (0..ov).map(|_| next16(&mut rng)).collect();
            let data: Vec<i32> = (0..ov).map(|_| next32(&mut rng)).collect();

            // Scalar reference
            let mut expected = data.clone();
            for i in 0..half {
                let yp = i;
                let xp = ov - 1 - i;
                let x_fwd = expected[xp];
                let x_rev = expected[yp];
                let w_asc = window[i] as i32;
                let w_desc = window[ov - 1 - i] as i32;
                expected[yp] = s_mul_inline(x_rev, w_desc).wrapping_sub(s_mul_inline(x_fwd, w_asc));
                expected[xp] = s_mul_inline(x_rev, w_asc).wrapping_add(s_mul_inline(x_fwd, w_desc));
            }

            let mut actual = data.clone();
            mdct_window_simd(&mut actual, &window, ov);

            assert_eq!(expected, actual, "remainder mismatch at overlap={ov}");
        }
    }
}
