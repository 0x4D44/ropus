//! CELT math operations — fixed-point transcendental approximations.
//!
//! Matches the C reference: `mathops.c`, `mathops.h`, `float_cast.h`.
//! All functions produce bit-exact output matching the C reference when
//! compiled with `FIXED_POINT` and `OPUS_FAST_INT64`.

use crate::types::*;

// ===========================================================================
// Integer square root
// ===========================================================================

/// Compute `floor(sqrt(val))` with exact arithmetic for all u32 > 0.
/// Uses binary digit-by-digit method from azillionmonkeys.com/qed/sqroot.html.
pub fn isqrt32(mut val: u32) -> u32 {
    debug_assert!(val > 0, "isqrt32: val must be > 0");
    let mut g: u32 = 0;
    let mut bshift: i32 = ((ec_ilog(val) - 1) >> 1) as i32;
    let mut b: u32 = 1u32 << bshift;
    loop {
        // t = (2*g + b) << bshift
        let t: u32 = ((g << 1) + b) << (bshift as u32);
        if t <= val {
            g += b;
            val -= t;
        }
        b >>= 1;
        bshift -= 1;
        if bshift < 0 {
            break;
        }
    }
    g
}

// ===========================================================================
// Integer log2
// ===========================================================================

/// Integer log2. Returns `EC_ILOG(x) - 1`. Safe for x <= 0 (returns 0).
#[inline(always)]
pub fn celt_ilog2(x: i32) -> i32 {
    if x <= 0 { return 0; }
    ec_ilog(x as u32) - 1
}

/// Integer log2, safe for zero (returns 0 for x <= 0).
#[inline(always)]
pub fn celt_zlog2(x: i32) -> i32 {
    if x <= 0 {
        0
    } else {
        celt_ilog2(x)
    }
}

// ===========================================================================
// Reciprocal square root
// ===========================================================================

/// Reciprocal sqrt approximation for range [0.25, 1.0) (Q16 in, Q14 out).
/// Max relative error: 1.05e-4.
pub fn celt_rsqrt_norm(x: i32) -> i32 {
    // n in [-16384, 32767] (Q15 [-0.5, 1.0))
    let n: i32 = x - 32768;

    // Minimax quadratic seed (Q14):
    //   r = 1.4378 + n*(-0.8234 + n*0.4096)
    //   Coefficients: 23557, -13490, 6713
    let r: i32 = add16(23557, mult16_16_q15(n, add16(-13490, mult16_16_q15(n, 6713))));

    // y = x*r*r - 1 in Q15, range [-1564, 1594]
    let r2: i32 = mult16_16_q15(r, r);
    let y: i32 = shl16(sub16(add16(mult16_16_q15(r2, n), r2), 16384), 1);

    // 2nd-order Householder iteration: r += r * y * (0.375*y - 0.5)
    add16(
        r,
        mult16_16_q15(r, mult16_16_q15(y, sub16(mult16_16_q15(y, 12288), 16384))),
    )
}

/// Reciprocal sqrt approximation for range [0.25, 1.0) (Q31 in, Q29 out).
/// Refines 16-bit seed via Newton-Raphson.
pub fn celt_rsqrt_norm32(x: i32) -> i32 {
    // Get 16-bit seed, convert to Q29
    let r_q29: i32 = shl32(celt_rsqrt_norm(shr32(x, 31 - 16)), 15);

    // Newton-Raphson: r = r * (1.5 - 0.5*x*r*r)
    // Split to avoid macro explosion in C; we keep the same structure.
    let mut tmp: i32 = mult32_32_q31(r_q29, r_q29);
    tmp = mult32_32_q31(1_073_741_824 /* 0.5 in Q31 */, tmp);
    tmp = mult32_32_q31(x, tmp);
    shl32(
        mult32_32_q31(r_q29, sub32(201_326_592 /* 1.5 in Q27 */, tmp)),
        4,
    )
}

// ===========================================================================
// Reciprocal
// ===========================================================================

/// 16-bit reciprocal for normalized Q15 input in [0.5, 1.0).
/// Output is Q15. Uses linear seed + 2 Newton iterations.
pub fn celt_rcp_norm16(x: i32) -> i32 {
    // Linear approximation: r = 1.8824 - 0.9412*n in Q14 range [15420, 30840]
    let mut r: i32 = add16(30840, mult16_16_q15(-15420, x));

    // Newton iteration 1: r -= r * (r*x + r - 1.Q15)
    r = sub16(
        r,
        mult16_16_q15(r, add16(mult16_16_q15(r, x), add16(r, -32768))),
    );

    // Newton iteration 2: same form, subtract extra 1 to avoid overflow
    sub16(
        r,
        add16(
            1,
            mult16_16_q15(r, add16(mult16_16_q15(r, x), add16(r, -32768))),
        ),
    )
}

/// 32-bit reciprocal for normalized Q31 input in [0.5, 1.0).
/// Output is Q30 in [1.0, 2.0).
pub fn celt_rcp_norm32(x: i32) -> i32 {
    debug_assert!(x >= 1_073_741_824, "celt_rcp_norm32: x must be >= 2^30");

    // Seed from 16-bit reciprocal, extended to Q30
    let r_q30: i32 = shl32(extend32(celt_rcp_norm16(shr32(x, 15) - 32768)), 16);

    // Newton: r = r - r*(r*x - 1)
    // Add 1 to avoid overflow (matching C exactly)
    sub32(
        r_q30,
        add32(
            shl32(
                mult32_32_q31(
                    add32(mult32_32_q31(r_q30, x), -1_073_741_824 /* -1.0 Q30 */),
                    r_q30,
                ),
                1,
            ),
            1,
        ),
    )
}

/// General reciprocal approximation (Q15 input, Q16 output).
/// Max relative error: 7.05e-5.
pub fn celt_rcp(x: i32) -> i32 {
    debug_assert!(x > 0, "celt_rcp: x must be > 0");
    let i = celt_ilog2(x);

    // Normalize to [0.5, 1.0) in Q15, compute reciprocal
    let r: i32 = celt_rcp_norm16(vshr32(x, i - 15) - 32768);

    // Denormalize: shift result to Q16
    vshr32(extend32(r), i - 16)
}

// ===========================================================================
// Fractional division
// ===========================================================================

/// Fractional division a/b → Q29 result.
pub fn frac_div32_q29(a: i32, b: i32) -> i32 {
    let shift = celt_ilog2(b) - 29;
    let a = vshr32(a, shift);
    let b = vshr32(b, shift);

    // 16-bit reciprocal of b
    let rcp: i32 = round16(celt_rcp(round16(b, 16)), 3);

    // First estimate
    let mut result: i32 = mult16_32_q15(rcp, a);

    // Remainder and refinement
    let rem: i32 = pshr32(a, 2) - mult32_32_q31(result, b);
    result = add32(result, shl32(mult16_32_q15(rcp, rem), 2));

    result
}

/// Fractional division a/b → Q31 result (saturated).
pub fn frac_div32(a: i32, b: i32) -> i32 {
    let result = frac_div32_q29(a, b);
    if result >= 536_870_912 {
        // >= 2^29
        2_147_483_647 // 2^31 - 1
    } else if result <= -536_870_912 {
        // <= -2^29
        -2_147_483_647 // -(2^31 - 1)
    } else {
        shl32(result, 2)
    }
}

/// Division via reciprocal: `a / b` using `MULT32_32_Q31(a, celt_rcp(b))`.
#[inline(always)]
pub fn celt_div(a: i32, b: i32) -> i32 {
    mult32_32_q31(a, celt_rcp(b))
}

// ===========================================================================
// Square root
// ===========================================================================

/// Square root approximation (QX input, QX/2 output, 16-bit precision).
/// Uses 5th-order minimax polynomial over [0.25, 1.0).
pub fn celt_sqrt(x: i32) -> i32 {
    // Minimax coefficients for sqrt(x) over [0.25, 1.0) in Q15.
    // RMS error 3.4e-5, max error 8.2e-5.
    const C: [i32; 6] = [23171, 11574, -2901, 1592, -1002, 336];

    if x == 0 {
        return 0;
    } else if x >= 1_073_741_824 {
        // >= 2^30
        return 32767;
    }

    let k = (celt_ilog2(x) >> 1) - 7;
    let x = vshr32(x, 2 * k);
    let n = x - 32768;

    // Horner evaluation: C[0] + n*(C[1] + n*(C[2] + n*(C[3] + n*(C[4] + n*C[5]))))
    let rt = add32(
        C[0],
        mult16_16_q15(
            n,
            add16(
                C[1],
                mult16_16_q15(
                    n,
                    add16(
                        C[2],
                        mult16_16_q15(
                            n,
                            add16(
                                C[3],
                                mult16_16_q15(n, add16(C[4], mult16_16_q15(n, C[5]))),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    );

    vshr32(rt, 7 - k)
}

/// Square root with 32-bit precision (QX in, Q(X/2+16) out).
/// Uses rsqrt_norm32 for higher precision.
pub fn celt_sqrt32(x: i32) -> i32 {
    if x == 0 {
        return 0;
    } else if x >= 1_073_741_824 {
        return 2_147_483_647; // 2^31 - 1
    }

    let k = celt_ilog2(x) >> 1;
    let x_frac = vshr32(x, 2 * (k - 14) - 1);
    let x_frac = mult32_32_q31(celt_rsqrt_norm32(x_frac), x_frac);

    if k < 12 {
        pshr32(x_frac, 12 - k)
    } else {
        shl32(x_frac, k - 12)
    }
}

// ===========================================================================
// Cosine
// ===========================================================================

/// Inner cosine approximation for cos(π/2 · x) where x is Q15 in [0, 32767].
/// Uses 4th-order Chebyshev-like polynomial.
fn celt_cos_pi_2(x: i32) -> i32 {
    const L1: i32 = 32767;
    const L2: i32 = -7651;
    const L3: i32 = 8277;
    const L4: i32 = -626;

    let x2: i32 = mult16_16_p15(x, x);
    add16(
        1,
        min16(
            32766,
            add32(
                sub16(L1, x2),
                mult16_16_p15(
                    x2,
                    add32(L2, mult16_16_p15(x2, add32(L3, mult16_16_p15(L4, x2)))),
                ),
            ),
        ),
    )
}

/// Cosine with Q16 phase input → Q15 output.
/// Computes `cos(π/2 · x)` where x is in [0, 2^17) representing [0, 2π).
pub fn celt_cos_norm(x: i32) -> i32 {
    // Fold to [0, 2π): mask to 17 bits
    let mut x = x & 0x0001ffff;

    // Mirror to [0, π]: if x > 2^16, x = 2^17 - x
    if x > shl32(extend32(1), 16) {
        x = sub32(shl32(extend32(1), 17), x);
    }

    // Quadrant dispatch
    if x & 0x00007fff != 0 {
        // Has fractional bits — use polynomial
        if x < shl32(extend32(1), 15) {
            celt_cos_pi_2(extract16(x))
        } else {
            neg16(celt_cos_pi_2(extract16(65536 - x)))
        }
    } else {
        // Exact boundary values
        if x & 0x0000ffff != 0 {
            0 // x = π/2 → cos = 0
        } else if x & 0x0001ffff != 0 {
            -32767 // x = π → cos = -1
        } else {
            32767 // x = 0 → cos = 1
        }
    }
}

/// Cosine with Q30 phase input → Q31 output.
/// Computes `cos(π/2 · x)` where x is Q30 in [-1.0, 1.0].
pub fn celt_cos_norm32(x: i32) -> i32 {
    const COS_NORM_COEFF_A0: i32 = 134_217_720; // Q27
    const COS_NORM_COEFF_A1: i32 = -662_336_704; // Q29
    const COS_NORM_COEFF_A2: i32 = 544_710_848; // Q31
    const COS_NORM_COEFF_A3: i32 = -178_761_936; // Q33
    const COS_NORM_COEFF_A4: i32 = 29_487_206; // Q35

    debug_assert!(
        x >= -1_073_741_824 && x <= 1_073_741_824,
        "celt_cos_norm32: x out of range"
    );

    // Make cos(±π/2) exactly zero
    if abs32(x) == (1 << 30) {
        return 0;
    }

    let x_sq_q29 = mult32_32_q31(x, x);

    // Horner evaluation with step-by-step expansion
    let mut tmp = add32(COS_NORM_COEFF_A3, mult32_32_q31(x_sq_q29, COS_NORM_COEFF_A4));
    tmp = add32(COS_NORM_COEFF_A2, mult32_32_q31(x_sq_q29, tmp));
    tmp = add32(COS_NORM_COEFF_A1, mult32_32_q31(x_sq_q29, tmp));

    shl32(add32(COS_NORM_COEFF_A0, mult32_32_q31(x_sq_q29, tmp)), 4)
}

// ===========================================================================
// Logarithm (base-2)
// ===========================================================================

/// Base-2 logarithm approximation (Q14 input, Q10 output).
/// Returns -32767 for x == 0.
pub fn celt_log2(x: i32) -> i32 {
    // Polynomial coefficients (Q15, with rounding offset baked into C[0])
    const C: [i32; 5] = [-6801 + (1 << (13 - 10)), 15746, -5217, 2545, -1401];

    if x == 0 {
        return -32767;
    }

    let i = celt_ilog2(x);
    // Normalize to [-16384, 16383] (Q14 mantissa in [-0.5, 0.5))
    let n = vshr32(x, i - 15) - 32768 - 16384;

    // 4th-order polynomial (Horner form)
    let frac = add16(
        C[0],
        mult16_16_q15(
            n,
            add16(
                C[1],
                mult16_16_q15(
                    n,
                    add16(C[2], mult16_16_q15(n, add16(C[3], mult16_16_q15(n, C[4])))),
                ),
            ),
        ),
    );

    // Combine integer and fractional parts → Q10
    shl32(i - 13, 10) + shr32(frac, 14 - 10)
}

/// Wrapper: celt_log2 output shifted to Q(DB_SHIFT=24).
#[inline(always)]
pub fn celt_log2_db(x: i32) -> i32 {
    shl32(extend32(celt_log2(x)), DB_SHIFT - 10)
}

// ===========================================================================
// Exponential (base-2)
// ===========================================================================

/// Fractional part of exp2: evaluates 2^(x/32768) for Q10 fractional input.
/// Output is Q15.
fn celt_exp2_frac(x: i32) -> i32 {
    // Coefficients: K0=1, K1=ln(2), K2=3-4*ln(2), K3=3*ln(2)-2
    const D0: i32 = 16383;
    const D1: i32 = 22804;
    const D2: i32 = 14819;
    const D3: i32 = 10204;

    let frac = shl16(x, 4);
    add16(
        D0,
        mult16_16_q15(frac, add16(D1, mult16_16_q15(frac, add16(D2, mult16_16_q15(D3, frac))))),
    )
}

/// Base-2 exponential approximation (Q10 input, Q16 output).
/// Saturates to 0x7F000000 for large positive, 0 for large negative.
pub fn celt_exp2(x: i32) -> i32 {
    let integer = shr16(x, 10);
    if integer > 14 {
        return 0x7f000000;
    } else if integer < -15 {
        return 0;
    }
    let frac = celt_exp2_frac(x - shl16(integer, 10));
    vshr32(extend32(frac), -integer - 2)
}

/// Wrapper: exp2 for Q(DB_SHIFT) fractional input → Q(DB_SHIFT) scale.
#[inline(always)]
pub fn celt_exp2_db_frac(x: i32) -> i32 {
    shl32(celt_exp2_frac(pshr32(x, DB_SHIFT - 10)), 14)
}

/// Wrapper: exp2 for Q(DB_SHIFT) input → Q16 output.
#[inline(always)]
pub fn celt_exp2_db(x: i32) -> i32 {
    celt_exp2(pshr32(x, DB_SHIFT - 10))
}

// ===========================================================================
// Arctangent
// ===========================================================================

/// Computes `atan(x) * 2/π` for Q30 input in [-1.0, 1.0], Q30 output.
pub fn celt_atan_norm(x: i32) -> i32 {
    const ATAN_2_OVER_PI: i32 = 1_367_130_551; // Q31
    const ATAN_COEFF_A03: i32 = -715_791_936; // Q31
    const ATAN_COEFF_A05: i32 = 857_391_616; // Q32
    const ATAN_COEFF_A07: i32 = -1_200_579_328; // Q33
    const ATAN_COEFF_A09: i32 = 1_682_636_672; // Q34
    const ATAN_COEFF_A11: i32 = -1_985_085_440; // Q35
    const ATAN_COEFF_A13: i32 = 1_583_306_112; // Q36
    const ATAN_COEFF_A15: i32 = -598_602_432; // Q37

    debug_assert!(
        x <= 1_073_741_824 && x >= -1_073_741_824,
        "celt_atan_norm: x out of range"
    );

    // Exact boundary values to avoid polynomial imprecision
    if x == 1_073_741_824 {
        return 536_870_912; // 0.5 in Q30
    }
    if x == -1_073_741_824 {
        return -536_870_912; // -0.5 in Q30
    }

    let x_q31 = shl32(x, 1);
    let x_sq_q30 = mult32_32_q31(x_q31, x);

    // Horner evaluation of odd-power polynomial
    let mut tmp = mult32_32_q31(x_sq_q30, ATAN_COEFF_A15);
    tmp = mult32_32_q31(x_sq_q30, add32(ATAN_COEFF_A13, tmp));
    tmp = mult32_32_q31(x_sq_q30, add32(ATAN_COEFF_A11, tmp));
    tmp = mult32_32_q31(x_sq_q30, add32(ATAN_COEFF_A09, tmp));
    tmp = mult32_32_q31(x_sq_q30, add32(ATAN_COEFF_A07, tmp));
    tmp = mult32_32_q31(x_sq_q30, add32(ATAN_COEFF_A05, tmp));
    tmp = mult32_32_q31(x_sq_q30, add32(ATAN_COEFF_A03, tmp));
    tmp = add32(x, mult32_32_q31(x_q31, tmp));

    mult32_32_q31(ATAN_2_OVER_PI, tmp)
}

/// Computes `atan2(y, x) * 2/π` for positive Q30 inputs. Output is Q30.
pub fn celt_atan2p_norm(y: i32, x: i32) -> i32 {
    debug_assert!(x >= 0 && y >= 0, "celt_atan2p_norm: inputs must be >= 0");
    if y == 0 && x == 0 {
        0
    } else if y < x {
        celt_atan_norm(shr32(frac_div32(y, x), 1))
    } else {
        debug_assert!(y > 0);
        1_073_741_824 /* 1.0 Q30 */ - celt_atan_norm(shr32(frac_div32(x, y), 1))
    }
}

/// Atan approximation using 4th-order polynomial (Q15 in, Q15 out).
/// Input normalized by π/4.
pub fn celt_atan01(x: i32) -> i32 {
    const M1: i32 = 32767;
    const M2: i32 = -21;
    const M3: i32 = -11943;
    const M4: i32 = 4936;

    mult16_16_p15(
        x,
        add32(
            M1,
            mult16_16_p15(x, add32(M2, mult16_16_p15(x, add32(M3, mult16_16_p15(M4, x))))),
        ),
    )
}

/// atan2 approximation for positive inputs (Q15 in, Q15 out).
pub fn celt_atan2p(y: i32, x: i32) -> i32 {
    if x == 0 && y == 0 {
        0
    } else if y < x {
        let arg = celt_div(shl32(extend32(y), 15), x);
        let arg = if arg >= 32767 { 32767 } else { arg };
        shr16(celt_atan01(extract16(arg)), 1)
    } else {
        let arg = celt_div(shl32(extend32(x), 15), y);
        let arg = if arg >= 32767 { 32767 } else { arg };
        25736 - shr16(celt_atan01(extract16(arg)), 1)
    }
}

// ===========================================================================
// Max absolute value scanning
// ===========================================================================

/// Find maximum absolute value in a slice of 16-bit values (stored as i32).
pub fn celt_maxabs16(x: &[i32]) -> i32 {
    let mut maxval: i32 = 0;
    let mut minval: i32 = 0;
    for &v in x {
        maxval = max16(maxval, v);
        minval = min16(minval, v);
    }
    max32(extend32(maxval), -extend32(minval))
}

/// Find maximum absolute value in a slice of 32-bit values.
pub fn celt_maxabs32(x: &[i32]) -> i32 {
    let mut maxval: i32 = 0;
    let mut minval: i32 = 0;
    for &v in x {
        maxval = max32(maxval, v);
        minval = min32(minval, v);
    }
    max32(maxval, -minval)
}

// ===========================================================================
// Float API utilities (from mathops.c, not gated by DISABLE_FLOAT_API)
// ===========================================================================

/// Convert float samples to i16, with rounding and clamping.
pub fn celt_float2int16(input: &[f32], output: &mut [i16]) {
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = float2int16(*inp);
    }
}

/// Clamp samples to [-2.0, 2.0]. Returns 0 (C implementation can't provide
/// quick hint about whether all samples are within [-1, 1]).
pub fn opus_limit2_checkwithin1(samples: &mut [f32]) -> i32 {
    if samples.is_empty() {
        return 1;
    }
    for s in samples.iter_mut() {
        let mut v = *s;
        if v < -2.0 {
            v = -2.0;
        }
        if v > 2.0 {
            v = 2.0;
        }
        *s = v;
    }
    0
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- isqrt32 ---

    #[test]
    fn test_isqrt32_perfect_squares() {
        assert_eq!(isqrt32(1), 1);
        assert_eq!(isqrt32(4), 2);
        assert_eq!(isqrt32(9), 3);
        assert_eq!(isqrt32(16), 4);
        assert_eq!(isqrt32(100), 10);
        assert_eq!(isqrt32(10000), 100);
        assert_eq!(isqrt32(1_000_000), 1000);
    }

    #[test]
    fn test_isqrt32_non_perfect() {
        assert_eq!(isqrt32(2), 1); // floor(sqrt(2)) = 1
        assert_eq!(isqrt32(3), 1);
        assert_eq!(isqrt32(5), 2);
        assert_eq!(isqrt32(8), 2);
        assert_eq!(isqrt32(99), 9);
        assert_eq!(isqrt32(101), 10);
    }

    #[test]
    fn test_isqrt32_large_values() {
        assert_eq!(isqrt32(0xFFFFFFFF), 65535);
        assert_eq!(isqrt32(0x80000000), 46340);
        // Verify: 46340^2 = 2,147,395,600 <= 2,147,483,648
        // 46341^2 = 2,147,488,281 > 2,147,483,648
    }

    #[test]
    fn test_isqrt32_property() {
        // For a range of values, verify floor(sqrt(n))^2 <= n < (floor(sqrt(n))+1)^2
        for val in [1u32, 2, 7, 15, 255, 1023, 65535, 100000, 0x7FFFFFFF, 0xFFFFFFFE] {
            let s = isqrt32(val);
            assert!(
                (s as u64) * (s as u64) <= val as u64,
                "isqrt32({val}): {s}^2 > {val}"
            );
            assert!(
                ((s + 1) as u64) * ((s + 1) as u64) > val as u64,
                "isqrt32({val}): ({})^2 <= {val}",
                s + 1
            );
        }
    }

    // --- celt_ilog2 / celt_zlog2 ---

    #[test]
    fn test_celt_ilog2() {
        assert_eq!(celt_ilog2(1), 0);
        assert_eq!(celt_ilog2(2), 1);
        assert_eq!(celt_ilog2(4), 2);
        assert_eq!(celt_ilog2(255), 7);
        assert_eq!(celt_ilog2(256), 8);
    }

    #[test]
    fn test_celt_zlog2() {
        assert_eq!(celt_zlog2(0), 0);
        assert_eq!(celt_zlog2(-1), 0);
        assert_eq!(celt_zlog2(1), 0);
        assert_eq!(celt_zlog2(256), 8);
    }

    // --- celt_sqrt ---

    #[test]
    fn test_celt_sqrt_edge_cases() {
        assert_eq!(celt_sqrt(0), 0);
        assert_eq!(celt_sqrt(1_073_741_824), 32767);
        assert_eq!(celt_sqrt(2_000_000_000), 32767); // saturated
    }

    #[test]
    fn test_celt_sqrt_known_values() {
        // For Q14 input, sqrt should give Q7 output (QX/2)
        // Input 16384 (= 1.0 in Q14) → output should be ~128 (= 1.0 in Q7)
        let result = celt_sqrt(16384);
        assert!(
            (result - 128).abs() <= 2,
            "celt_sqrt(16384) = {result}, expected ~128"
        );
    }

    // --- celt_cos_norm ---

    #[test]
    fn test_celt_cos_norm_boundaries() {
        // x=0 → cos(0) = 1.0 → 32767 in Q15
        assert_eq!(celt_cos_norm(0), 32767);
        // x=32768 (= 2^15) → cos(π/2) = 0
        assert_eq!(celt_cos_norm(32768), 0);
        // x=65536 (= 2^16) → cos(π) = -1.0 → -32767
        assert_eq!(celt_cos_norm(65536), -32767);
        // x=131072 (= 2^17) → cos(2π) = 1.0 → wraps to cos(0)
        assert_eq!(celt_cos_norm(131072), 32767);
    }

    #[test]
    fn test_celt_cos_norm_symmetry() {
        // cos is even: cos(-x) = cos(x). Test via wrapping (2^17 - x).
        for x in [100, 1000, 10000, 30000] {
            let pos = celt_cos_norm(x);
            // cos(2π - x) = cos(x) when x is in phase units
            let neg = celt_cos_norm(131072 - x);
            assert_eq!(
                pos, neg,
                "Symmetry failed for x={x}: cos(x)={pos}, cos(2π-x)={neg}"
            );
        }
    }

    // --- celt_cos_norm32 ---

    #[test]
    fn test_celt_cos_norm32_boundaries() {
        // cos(0) = max positive
        assert!(celt_cos_norm32(0) > 2_000_000_000);
        // cos(±π/2) = 0
        assert_eq!(celt_cos_norm32(1 << 30), 0);
        assert_eq!(celt_cos_norm32(-(1 << 30)), 0);
    }

    // --- celt_log2 / celt_exp2 ---

    #[test]
    fn test_celt_log2_zero() {
        assert_eq!(celt_log2(0), -32767);
    }

    #[test]
    fn test_celt_log2_power_of_two() {
        // Input is Q14. 16384 = 1.0 in Q14. log2(1.0) = 0.0 → Q10 = 0.
        assert!(
            celt_log2(16384).abs() <= 1,
            "celt_log2(16384) = {}, expected ~0",
            celt_log2(16384)
        );
        // 32768 = 2.0 in Q14. log2(2.0) = 1.0 → Q10 = 1024.
        let result = celt_log2(32768);
        assert!(
            (result - 1024).abs() <= 2,
            "celt_log2(32768) = {result}, expected ~1024"
        );
    }

    #[test]
    fn test_celt_exp2_saturation() {
        // Large positive → saturate
        assert_eq!(celt_exp2(15 * 1024), 0x7f000000);
        // Large negative → 0
        assert_eq!(celt_exp2(-16 * 1024), 0);
    }

    #[test]
    fn test_log2_exp2_roundtrip() {
        // For a range of values, log2 then exp2 should approximate identity.
        for &x in &[1000i32, 4096, 8192, 16384, 30000] {
            let log_val = celt_log2(x);
            let recovered = celt_exp2(log_val);
            // The roundtrip loses precision, so allow tolerance.
            // exp2 output is Q16, input was Q14 → shifted by 2.
            let expected = shl32(x, 2);
            let diff = (recovered - expected).abs();
            assert!(
                diff < expected / 32 + 16,
                "Roundtrip failed for x={x}: log2={log_val}, exp2={recovered}, expected~{expected}, diff={diff}"
            );
        }
    }

    // --- celt_rcp ---

    #[test]
    fn test_celt_rcp_basic() {
        // rcp(16384) should approximate 1/0.5 = 2.0 in Q16 = 131072
        // Input 16384 in Q15 = 0.5
        let result = celt_rcp(16384);
        let diff = (result - 131072).abs();
        assert!(
            diff < 32,
            "celt_rcp(16384) = {result}, expected ~131072, diff={diff}"
        );
    }

    #[test]
    fn test_celt_rcp_one() {
        // rcp(32767) ≈ 1/1.0 = 1.0 in Q16 = 65536
        let result = celt_rcp(32767);
        let diff = (result - 65536).abs();
        assert!(
            diff < 16,
            "celt_rcp(32767) = {result}, expected ~65536, diff={diff}"
        );
    }

    // --- frac_div32 ---

    #[test]
    fn test_frac_div32_saturation() {
        // When a/b > 1, Q31 result saturates to MAX.
        // a=10000, b=5000 → ratio=2.0 → Q29 result > 2^29 → saturates.
        let result = frac_div32(10000, 5000);
        assert_eq!(result, 2_147_483_647);
    }

    #[test]
    fn test_frac_div32_half() {
        // a/b = 0.5 → Q31 result should be ~2^30 = 1073741824
        // Use a=500_000, b=1_000_000
        let result = frac_div32(500_000, 1_000_000);
        let diff = (result - 1_073_741_824).abs();
        assert!(
            diff < 4096,
            "frac_div32(500K, 1M) = {result}, expected ~1073741824, diff={diff}"
        );
    }

    // --- celt_atan_norm ---

    #[test]
    fn test_celt_atan_norm_boundaries() {
        // atan(0) = 0
        assert_eq!(celt_atan_norm(0), 0);
        // atan(1.0) * 2/π = 0.5 → Q30 = 536870912
        assert_eq!(celt_atan_norm(1_073_741_824), 536_870_912);
        // atan(-1.0) * 2/π = -0.5
        assert_eq!(celt_atan_norm(-1_073_741_824), -536_870_912);
    }

    #[test]
    fn test_celt_atan_norm_small() {
        // For small x, atan(x) ≈ x, so atan(x)*2/π ≈ x*2/π ≈ 0.6366*x
        let x = 100_000; // small in Q30
        let result = celt_atan_norm(x);
        let expected = (x as f64 * 0.6366197) as i32;
        assert!(
            (result - expected).abs() < 100,
            "celt_atan_norm({x}) = {result}, expected ~{expected}"
        );
    }

    // --- celt_atan01 ---

    #[test]
    fn test_celt_atan01_zero() {
        assert_eq!(celt_atan01(0), 0);
    }

    #[test]
    fn test_celt_atan01_one() {
        // atan(1.0) = π/4 ≈ 0.7854. In Q15: 0.7854 * 32768 ≈ 25736.
        let result = celt_atan01(32767);
        assert!(
            (result - 25736).abs() <= 4,
            "celt_atan01(32767) = {result}, expected ~25736"
        );
    }

    // --- celt_maxabs ---

    #[test]
    fn test_celt_maxabs16_basic() {
        assert_eq!(celt_maxabs16(&[0, 0, 0]), 0);
        assert_eq!(celt_maxabs16(&[10, -20, 5]), 20);
        assert_eq!(celt_maxabs16(&[-32767, 100, 32767]), 32767);
    }

    #[test]
    fn test_celt_maxabs16_empty() {
        assert_eq!(celt_maxabs16(&[]), 0);
    }

    #[test]
    fn test_celt_maxabs32_basic() {
        assert_eq!(celt_maxabs32(&[100, -200, 50]), 200);
    }

    // --- celt_rsqrt_norm ---

    #[test]
    fn test_celt_rsqrt_norm_quarter() {
        // Input 0.25 in Q16 = 16384. rsqrt(0.25) = 2.0. Output Q14 = 32768.
        // But Q14 max is 16383... hmm, 2.0 in Q14 = 32768 which wraps.
        // Actually, the range is [0.25, 1.0), and rsqrt(0.25) = 2.0 is the boundary.
        // For x slightly above 0.25, the result should be slightly below 2.0.
        let result = celt_rsqrt_norm(16500); // slightly above 0.25
        // rsqrt(16500/65536) = rsqrt(0.2518) ≈ 1.993 → Q14 ≈ 32649
        assert!(
            result > 30000 && result < 33000,
            "celt_rsqrt_norm(16500) = {result}"
        );
    }

    #[test]
    fn test_celt_rsqrt_norm_one() {
        // Input ~1.0 in Q16 = 65535 (just below 1.0). rsqrt(1.0) = 1.0 → Q14 = 16384
        let result = celt_rsqrt_norm(65535);
        let diff = (result - 16384).abs();
        assert!(
            diff < 32,
            "celt_rsqrt_norm(65535) = {result}, expected ~16384"
        );
    }

    // --- Float API ---

    #[test]
    fn test_celt_float2int16() {
        let input = [0.0f32, 0.5, -0.5, 1.0, -1.0];
        let mut output = [0i16; 5];
        celt_float2int16(&input, &mut output);
        assert_eq!(output[0], 0);
        assert_eq!(output[1], 16384);
        assert_eq!(output[2], -16384);
        assert_eq!(output[3], 32767);
        assert_eq!(output[4], -32768);
    }

    #[test]
    fn test_opus_limit2_checkwithin1() {
        let mut samples = [0.0f32, 1.5, -1.5, 3.0, -3.0];
        let result = opus_limit2_checkwithin1(&mut samples);
        assert_eq!(result, 0);
        assert_eq!(samples[0], 0.0);
        assert_eq!(samples[1], 1.5);
        assert_eq!(samples[2], -1.5);
        assert_eq!(samples[3], 2.0); // clamped
        assert_eq!(samples[4], -2.0); // clamped
    }

    #[test]
    fn test_opus_limit2_checkwithin1_empty() {
        let mut samples: [f32; 0] = [];
        assert_eq!(opus_limit2_checkwithin1(&mut samples), 1);
    }
}
