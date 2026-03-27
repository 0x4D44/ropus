//! Core type aliases and fixed-point arithmetic primitives.
//!
//! Matches the C reference: `arch.h`, `fixed_generic.h`, `ecintrin.h`.
//! All functions are `#[inline(always)]` to match C macro performance.
//! Uses the `OPUS_FAST_INT64` path (64-bit intermediates) for all 32×32
//! multiplies, since Rust has native i64.

// ---------------------------------------------------------------------------
// Type aliases (matching opus_types.h)
// ---------------------------------------------------------------------------

pub type OpusInt16 = i16;
pub type OpusInt32 = i32;
pub type OpusUint32 = u32;
pub type OpusInt64 = i64;

// ---------------------------------------------------------------------------
// Constants (matching arch.h)
// ---------------------------------------------------------------------------

pub const Q15ONE: i32 = 32767;
pub const Q31ONE: i32 = 2147483647;
pub const SIG_SHIFT: i32 = 12;
pub const SIG_SAT: i32 = 536870911; // 2^29 - 1
pub const DB_SHIFT: i32 = 24;
pub const NORM_SHIFT: i32 = 24;
pub const NORM_SCALING: i32 = 1 << NORM_SHIFT;
pub const EPSILON: i32 = 1;
pub const VERY_SMALL: i32 = 0;
pub const VERY_LARGE16: i32 = 32767;
pub const Q15_ONE: i32 = 32767;
pub const CELT_SIG_SCALE: f32 = 32768.0;
pub const CELT_LPC_ORDER: usize = 24;

// ---------------------------------------------------------------------------
// EC_ILOG (from ecintrin.h)
// ---------------------------------------------------------------------------

/// Number of bits needed to represent the value (position of highest set bit + 1).
/// For input 0, returns 0 (undefined in C reference, but safe here).
#[inline(always)]
pub fn ec_ilog(x: u32) -> i32 {
    32 - x.leading_zeros() as i32
}

// ---------------------------------------------------------------------------
// Min / Max / Abs
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn min16(a: i32, b: i32) -> i32 {
    if a < b { a } else { b }
}
#[inline(always)]
pub fn max16(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}
#[inline(always)]
pub fn min32(a: i32, b: i32) -> i32 {
    if a < b { a } else { b }
}
#[inline(always)]
pub fn max32(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}
#[inline(always)]
pub fn imin(a: i32, b: i32) -> i32 {
    if a < b { a } else { b }
}
#[inline(always)]
pub fn imax(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}
#[inline(always)]
pub fn abs16(x: i32) -> i32 {
    if x < 0 { -x } else { x }
}
#[inline(always)]
pub fn abs32(x: i32) -> i32 {
    if x < 0 { -x } else { x }
}

// ---------------------------------------------------------------------------
// Saturation
// ---------------------------------------------------------------------------

/// Saturate i32 to i16 range [-32768, 32767].
#[inline(always)]
pub fn sat16(x: i32) -> i16 {
    if x > 32767 {
        32767
    } else if x < -32768 {
        -32768
    } else {
        x as i16
    }
}

/// Clamp to [-a, a].
#[inline(always)]
pub fn saturate(x: i32, a: i32) -> i32 {
    if x > a {
        a
    } else if x < -a {
        -a
    } else {
        x
    }
}

/// Saturate to i16 range, returned as i32.
#[inline(always)]
pub fn saturate16(x: i32) -> i32 {
    if x > 32767 {
        32767
    } else if x < -32768 {
        -32768
    } else {
        x
    }
}

// ---------------------------------------------------------------------------
// Negate
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn neg16(x: i32) -> i32 {
    -x
}
#[inline(always)]
pub fn neg32(x: i32) -> i32 {
    -x
}

// ---------------------------------------------------------------------------
// Width conversions
// ---------------------------------------------------------------------------

/// Truncate to 16-bit, sign-extend back to i32. Matches C: `(opus_val16)(x)`.
#[inline(always)]
pub fn extract16(x: i32) -> i32 {
    x as i16 as i32
}

/// Identity cast to i32. Matches C: `(opus_val32)(x)`.
#[inline(always)]
pub fn extend32(x: i32) -> i32 {
    x
}

// ---------------------------------------------------------------------------
// Shift operations
// ---------------------------------------------------------------------------

/// Arithmetic right shift of a 16-bit value (operates on i32).
#[inline(always)]
pub fn shr16(a: i32, shift: i32) -> i32 {
    a >> shift
}

/// Left shift of 16-bit value via unsigned cast.
/// Matches C: `(opus_int16)((opus_uint16)(a) << shift)`.
#[inline(always)]
pub fn shl16(a: i32, shift: i32) -> i32 {
    (a as u16).wrapping_shl(shift as u32) as i16 as i32
}

/// Arithmetic right shift of 32-bit value.
#[inline(always)]
pub fn shr32(a: i32, shift: i32) -> i32 {
    a >> shift
}

/// Left shift via unsigned cast (avoids signed overflow UB).
/// Matches C: `(opus_int32)((opus_uint32)(a) << shift)`.
#[inline(always)]
pub fn shl32(a: i32, shift: i32) -> i32 {
    (a as u32).wrapping_shl(shift as u32) as i32
}

/// Round-to-nearest right shift.
/// Matches C: `(a + ((1 << shift) >> 1)) >> shift`.
#[inline(always)]
pub fn pshr32(a: i32, shift: i32) -> i32 {
    // Bias = (1 << shift) >> 1 = 1 << (shift-1) for shift>0, 0 for shift=0
    let bias = shl32(1, shift) >> 1;
    (a + bias) >> shift
}

/// Variable-direction shift: positive = right, negative = left.
#[inline(always)]
pub fn vshr32(a: i32, shift: i32) -> i32 {
    if shift > 0 {
        shr32(a, shift)
    } else {
        shl32(a, -shift)
    }
}

/// Arithmetic right shift of 64-bit value.
#[inline(always)]
pub fn shr64(a: i64, shift: i32) -> i64 {
    a >> shift
}

/// Raw shift right (matches C `SHR` macro).
#[inline(always)]
pub fn shr(a: i32, shift: i32) -> i32 {
    a >> shift
}

/// Raw shift left (same as shl32, matches C `SHL` macro).
#[inline(always)]
pub fn shl(a: i32, shift: i32) -> i32 {
    shl32(a, shift)
}

/// Raw round-to-nearest shift (matches C `PSHR` macro).
#[inline(always)]
pub fn pshr(a: i32, shift: i32) -> i32 {
    pshr32(a, shift)
}

// ---------------------------------------------------------------------------
// Addition / Subtraction
// ---------------------------------------------------------------------------

/// Add two 16-bit values. Truncates operands and result to 16-bit range.
/// Matches C: `(opus_val16)((opus_val16)(a) + (opus_val16)(b))`.
#[inline(always)]
pub fn add16(a: i32, b: i32) -> i32 {
    ((a as i16 as i32) + (b as i16 as i32)) as i16 as i32
}

/// Subtract two 16-bit values.
#[inline(always)]
pub fn sub16(a: i32, b: i32) -> i32 {
    ((a as i16 as i32) - (b as i16 as i32)) as i16 as i32
}

/// Add two 32-bit values (panics on overflow in debug, wraps in release).
#[inline(always)]
pub fn add32(a: i32, b: i32) -> i32 {
    a + b
}

/// Subtract two 32-bit values.
#[inline(always)]
pub fn sub32(a: i32, b: i32) -> i32 {
    a - b
}

/// Wrapping add (overflow-tolerant). Matches C `ADD32_ovflw`.
#[inline(always)]
pub fn add32_ovflw(a: i32, b: i32) -> i32 {
    (a as u32).wrapping_add(b as u32) as i32
}

/// Wrapping subtract (overflow-tolerant). Matches C `SUB32_ovflw`.
#[inline(always)]
pub fn sub32_ovflw(a: i32, b: i32) -> i32 {
    (a as u32).wrapping_sub(b as u32) as i32
}

/// Wrapping negate (overflow-tolerant). Matches C `NEG32_ovflw`.
#[inline(always)]
pub fn neg32_ovflw(a: i32) -> i32 {
    (0u32).wrapping_sub(a as u32) as i32
}

/// Overflow-tolerant left shift.
#[inline(always)]
pub fn shl32_ovflw(a: i32, shift: i32) -> i32 {
    shl32(a, shift)
}

/// Overflow-tolerant round-to-nearest right shift.
#[inline(always)]
pub fn pshr32_ovflw(a: i32, shift: i32) -> i32 {
    shr32(add32_ovflw(a, shl32(1, shift) >> 1), shift)
}

// ---------------------------------------------------------------------------
// Rounding and compound shifts
// ---------------------------------------------------------------------------

/// Shift right with rounding, truncate to 16-bit.
/// Matches C: `EXTRACT16(PSHR32(x, a))`.
#[inline(always)]
pub fn round16(x: i32, a: i32) -> i32 {
    extract16(pshr32(x, a))
}

/// Shift right with rounding, saturate to [-32767, 32767].
/// Matches C: `EXTRACT16(SATURATE(PSHR32(x, a), 32767))`.
#[inline(always)]
pub fn sround16(x: i32, a: i32) -> i32 {
    extract16(saturate(pshr32(x, a), 32767))
}

/// Divide by 2 via right shift (16-bit).
#[inline(always)]
pub fn half16(x: i32) -> i32 {
    shr16(x, 1)
}

/// Divide by 2 via right shift (32-bit).
#[inline(always)]
pub fn half32(x: i32) -> i32 {
    shr32(x, 1)
}

// ---------------------------------------------------------------------------
// Multiply operations
// ---------------------------------------------------------------------------

/// 32×32 → 32 multiply (matches C `IMUL32`).
#[inline(always)]
pub fn imul32(a: i32, b: i32) -> i32 {
    a * b
}

/// 16×16 → 16 multiply (result expected to fit in 16 bits).
/// Matches C: `(opus_val16)(a) * (opus_val16)(b)`.
#[inline(always)]
pub fn mult16_16_16(a: i32, b: i32) -> i32 {
    (a as i16 as i32) * (b as i16 as i32)
}

/// 32×32 → 32 multiply (lower 32 bits, wrapping).
#[inline(always)]
pub fn mult32_32_32(a: i32, b: i32) -> i32 {
    a.wrapping_mul(b)
}

/// 16×16 → 32 multiply.
/// Matches C: `(opus_val32)(opus_val16)(a) * (opus_val32)(opus_val16)(b)`.
#[inline(always)]
pub fn mult16_16(a: i32, b: i32) -> i32 {
    (a as i16 as i32) * (b as i16 as i32)
}

/// 16-bit signed × 16-bit unsigned → 32 multiply.
/// Matches C: `(opus_val16)(a) * (opus_uint16)(b)`.
#[inline(always)]
pub fn mult16_16su(a: i32, b: i32) -> i32 {
    (a as i16 as i32) * (b as u16 as i32)
}

/// 16×16 multiply-accumulate.
#[inline(always)]
pub fn mac16_16(c: i32, a: i32, b: i32) -> i32 {
    c + mult16_16(a, b)
}

// --- Shifted 16×16 multiplies ---

/// 16×16 multiply, >> 11 (truncate).
#[inline(always)]
pub fn mult16_16_q11_32(a: i32, b: i32) -> i32 {
    shr(mult16_16(a, b), 11)
}

/// 16×16 multiply, >> 11.
#[inline(always)]
pub fn mult16_16_q11(a: i32, b: i32) -> i32 {
    shr(mult16_16(a, b), 11)
}

/// 16×16 multiply, >> 13.
#[inline(always)]
pub fn mult16_16_q13(a: i32, b: i32) -> i32 {
    shr(mult16_16(a, b), 13)
}

/// 16×16 multiply, >> 14.
#[inline(always)]
pub fn mult16_16_q14(a: i32, b: i32) -> i32 {
    shr(mult16_16(a, b), 14)
}

/// 16×16 multiply, >> 15.
#[inline(always)]
pub fn mult16_16_q15(a: i32, b: i32) -> i32 {
    shr(mult16_16(a, b), 15)
}

/// 16×16 multiply, + 4096 then >> 13 (round-to-nearest).
#[inline(always)]
pub fn mult16_16_p13(a: i32, b: i32) -> i32 {
    shr(4096 + mult16_16(a, b), 13)
}

/// 16×16 multiply, + 8192 then >> 14 (round-to-nearest).
#[inline(always)]
pub fn mult16_16_p14(a: i32, b: i32) -> i32 {
    shr(8192 + mult16_16(a, b), 14)
}

/// 16×16 multiply, + 16384 then >> 15 (round-to-nearest).
#[inline(always)]
pub fn mult16_16_p15(a: i32, b: i32) -> i32 {
    shr(16384 + mult16_16(a, b), 15)
}

// --- 16×32 multiplies (OPUS_FAST_INT64 path) ---

/// 16×32 multiply, >> 15.
#[inline(always)]
pub fn mult16_32_q15(a: i32, b: i32) -> i32 {
    ((a as i16 as i64) * (b as i64) >> 15) as i32
}

/// 16×32 multiply, >> 16.
#[inline(always)]
pub fn mult16_32_q16(a: i32, b: i32) -> i32 {
    ((a as i16 as i64) * (b as i64) >> 16) as i32
}

/// 16×32 multiply, round-to-nearest >> 16.
#[inline(always)]
pub fn mult16_32_p16(a: i32, b: i32) -> i32 {
    ((a as i16 as i64 * b as i64 + 32768) >> 16) as i32
}

// --- 32×32 multiplies (OPUS_FAST_INT64 path) ---

/// 32×32 multiply, >> 16.
#[inline(always)]
pub fn mult32_32_q16(a: i32, b: i32) -> i32 {
    ((a as i64) * (b as i64) >> 16) as i32
}

/// 32×32 multiply, >> 31.
#[inline(always)]
pub fn mult32_32_q31(a: i32, b: i32) -> i32 {
    ((a as i64) * (b as i64) >> 31) as i32
}

/// 32×32 multiply, + 2^30 then >> 31 (round-to-nearest).
#[inline(always)]
pub fn mult32_32_p31(a: i32, b: i32) -> i32 {
    ((1_073_741_824i64 + (a as i64) * (b as i64)) >> 31) as i32
}

/// Overflow-tolerant version of mult32_32_p31.
#[inline(always)]
pub fn mult32_32_p31_ovflw(a: i32, b: i32) -> i32 {
    mult32_32_p31(a, b)
}

/// 32×32 multiply, >> 32.
#[inline(always)]
pub fn mult32_32_q32(a: i32, b: i32) -> i32 {
    ((a as i64) * (b as i64) >> 32) as i32
}

// --- Multiply-accumulate (16×32) ---

/// 16×32 MAC, >> 15. Uses decomposed form matching C `MAC16_32_Q15`.
/// Constraint: b must fit in 31 bits.
#[inline(always)]
pub fn mac16_32_q15(c: i32, a: i32, b: i32) -> i32 {
    c + mult16_16(a, shr(b, 15)) + shr(mult16_16(a, b & 0x00007fff), 15)
}

/// 16×32 MAC, >> 16.
#[inline(always)]
pub fn mac16_32_q16(c: i32, a: i32, b: i32) -> i32 {
    c + mult16_16(a, shr(b, 16)) + shr(mult16_16su(a, b & 0x0000ffff), 16)
}

// --- Division ---

/// Divide 32-bit by 16-bit.
#[inline(always)]
pub fn div32_16(a: i32, b: i32) -> i32 {
    a / (b as i16 as i32)
}

/// Divide 32-bit by 32-bit.
#[inline(always)]
pub fn div32(a: i32, b: i32) -> i32 {
    a / b
}

// ---------------------------------------------------------------------------
// Compile-time Q-format conversion
// ---------------------------------------------------------------------------

/// Convert float constant to Q-format fixed-point (compile-time).
/// Matches C: `((opus_val16)(0.5 + x * (1 << bits)))`.
pub const fn qconst16(x: f64, bits: u32) -> i32 {
    (0.5 + x * ((1i64 << bits) as f64)) as i32
}

/// Convert float constant to Q-format fixed-point (compile-time).
/// Matches C: `((opus_val32)(0.5 + x * (1 << bits)))`.
pub const fn qconst32(x: f64, bits: u32) -> i32 {
    (0.5 + x * ((1i64 << bits) as f64)) as i32
}

// ---------------------------------------------------------------------------
// Signal conversion
// ---------------------------------------------------------------------------

/// Convert signal (Q(SIG_SHIFT=12)) to 16-bit word.
/// Matches C `SIG2WORD16_generic`.
#[inline(always)]
pub fn sig2word16(x: i32) -> i32 {
    let x = pshr32(x, SIG_SHIFT);
    let x = max32(x, -32768);
    let x = min32(x, 32767);
    extract16(x)
}

// ---------------------------------------------------------------------------
// Float conversion (from float_cast.h)
// ---------------------------------------------------------------------------

/// Round float to nearest integer (ties to even, matching SSE `cvtss2si`).
#[inline(always)]
pub fn float2int(x: f32) -> i32 {
    x.round_ties_even() as i32
}

/// Convert float [-1, 1] to i16 [-32768, 32767].
/// Matches C `FLOAT2INT16`.
#[inline(always)]
pub fn float2int16(x: f32) -> i16 {
    let x = x * CELT_SIG_SCALE;
    let x = if x > 32767.0 { 32767.0 } else { x };
    let x = if x < -32768.0 { -32768.0 } else { x };
    float2int(x) as i16
}

/// Convert float to 24-bit integer.
/// Matches C `FLOAT2INT24`.
#[inline(always)]
pub fn float2int24(x: f32) -> i32 {
    let x = x * (CELT_SIG_SCALE * 256.0);
    let x = if x > 16777216.0 { 16777216.0 } else { x };
    let x = if x < -16777216.0 { -16777216.0 } else { x };
    float2int(x)
}

// ---------------------------------------------------------------------------
// FRAC_MUL16 (from mathops.h)
// ---------------------------------------------------------------------------

/// Fractional 16×16 multiply with rounding. Bit-exactness critical.
/// Matches C: `(16384 + (opus_int16)(a) * (opus_int16)(b)) >> 15`.
#[inline(always)]
pub fn frac_mul16(a: i32, b: i32) -> i32 {
    (16384 + (a as i16 as i32) * (b as i16 as i32)) >> 15
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ec_ilog() {
        assert_eq!(ec_ilog(1), 1);
        assert_eq!(ec_ilog(2), 2);
        assert_eq!(ec_ilog(3), 2);
        assert_eq!(ec_ilog(4), 3);
        assert_eq!(ec_ilog(255), 8);
        assert_eq!(ec_ilog(256), 9);
        assert_eq!(ec_ilog(0x80000000), 32);
        assert_eq!(ec_ilog(0xFFFFFFFF), 32);
    }

    #[test]
    fn test_shl32_unsigned_wrap() {
        // Shifting a negative value left should use unsigned semantics
        assert_eq!(shl32(1, 31), i32::MIN); // 0x80000000
        assert_eq!(shl32(-1, 1), -2);
    }

    #[test]
    fn test_pshr32() {
        assert_eq!(pshr32(100, 0), 100);
        assert_eq!(pshr32(100, 1), 50);
        assert_eq!(pshr32(3, 1), 2); // (3+1)>>1 = 2
        assert_eq!(pshr32(5, 1), 3); // (5+1)>>1 = 3
        assert_eq!(pshr32(16384, 15), 1); // (16384+16384)>>15 = 1
    }

    #[test]
    fn test_vshr32() {
        assert_eq!(vshr32(256, 4), 16);
        assert_eq!(vshr32(16, -4), 256);
        assert_eq!(vshr32(100, 0), 100);
    }

    #[test]
    fn test_mult16_16_q15() {
        assert_eq!(mult16_16_q15(32767, 32767), 32766); // ~1.0 * ~1.0
        assert_eq!(mult16_16_q15(16384, 16384), 8192); // 0.5 * 0.5 = 0.25
        assert_eq!(mult16_16_q15(0, 32767), 0);
    }

    #[test]
    fn test_mult32_32_q31() {
        // ~1.0 * ~1.0 in Q31
        let a = 2_147_483_647i32; // Q31ONE
        let b = 2_147_483_647i32;
        let result = mult32_32_q31(a, b);
        // (2^31-1)^2 >> 31 ≈ 2^31 - 2 = 2147483646
        assert_eq!(result, 2_147_483_646);
    }

    #[test]
    fn test_sat16() {
        assert_eq!(sat16(0), 0);
        assert_eq!(sat16(32767), 32767);
        assert_eq!(sat16(32768), 32767);
        assert_eq!(sat16(-32768), -32768);
        assert_eq!(sat16(-32769), -32768);
    }

    #[test]
    fn test_add16_overflow() {
        // 30000 + 30000 = 60000, truncated to i16 = -5536
        assert_eq!(add16(30000, 30000), -5536);
    }

    #[test]
    fn test_extract16() {
        assert_eq!(extract16(0), 0);
        assert_eq!(extract16(32767), 32767);
        assert_eq!(extract16(32768), -32768); // wrap
        assert_eq!(extract16(65535), -1); // 0xFFFF as i16
    }

    #[test]
    fn test_frac_mul16() {
        // (16384 + 32767*32767) >> 15 = (16384 + 1073676289) >> 15 = 32766
        assert_eq!(frac_mul16(32767, 32767), 32766);
        assert_eq!(frac_mul16(0, 32767), 0);
        // 0.5 * 0.5 = 0.25
        assert_eq!(frac_mul16(16384, 16384), 8192);
    }

    #[test]
    fn test_qconst16() {
        assert_eq!(qconst16(1.0, 15), 32768); // slightly over Q15ONE
        assert_eq!(qconst16(0.5, 15), 16384);
        assert_eq!(qconst16(0.0, 15), 0);
    }

    #[test]
    fn test_round16() {
        assert_eq!(round16(32768, 1), 16384);
    }

    #[test]
    fn test_sround16() {
        assert_eq!(sround16(100000, 1), 32767); // saturates to max
        assert_eq!(sround16(-100000, 1), -32767); // saturates to min
    }
}
