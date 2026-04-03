//! CELT vector quantization (PVQ).
//!
//! Implements PVQ search, encode, decode, exp_rotation, vector normalization,
//! and stereo angle computation. Matches `vq.c`/`vq.h` in the C reference
//! (FIXED_POINT path, no ENABLE_QEXT, no SIMD overrides).

use super::cwrs::{celt_pvq_v, cwrsi, icwrs};
use super::ec_ctx::EcCoder;
use super::math_ops::*;
use crate::types::*;

// ===========================================================================
// Constants
// ===========================================================================

/// Spread mode: no spreading rotation.
const SPREAD_NONE: i32 = 0;

/// Spread factor table indexed by `spread - 1` for SPREAD_LIGHT/NORMAL/AGGRESSIVE.
const SPREAD_FACTOR: [i32; 3] = [15, 10, 5];

// ===========================================================================
// Unsigned integer division (matches C `celt_udiv`)
// ===========================================================================

#[inline(always)]
fn celt_udiv(n: u32, d: u32) -> u32 {
    n / d
}

// ===========================================================================
// Inner products
// ===========================================================================

/// Inner product with 64-bit accumulator and norm-shift.
/// Accumulates `x[i] * y[i]` in i64, then shifts right by `2*(NORM_SHIFT-14) = 20`.
/// Matches C `celt_inner_prod_norm_shift()`.
pub fn celt_inner_prod_norm_shift(x: &[i32], y: &[i32], n: usize) -> i32 {
    let mut sum: i64 = 0;
    for i in 0..n {
        sum += x[i] as i64 * y[i] as i64;
    }
    (sum >> (2 * (NORM_SHIFT - 14))) as i32
}

/// Inner product on Q14-scaled norm values (after scaledown).
/// Matches C `celt_inner_prod_norm()`.
fn celt_inner_prod_norm(x: &[i32], y: &[i32], n: usize) -> i32 {
    let mut sum: i32 = 0;
    for i in 0..n {
        sum += x[i] * y[i];
    }
    sum
}

// ===========================================================================
// Norm scaling
// ===========================================================================

/// Scale down norm values by `shift` bits with rounding.
/// Matches C `norm_scaledown()` which uses `PSHR32` (round-to-nearest).
fn norm_scaledown(x: &mut [i32], n: usize, shift: i32) {
    debug_assert!(shift >= 0);
    if shift <= 0 {
        return;
    }
    for i in 0..n {
        x[i] = pshr32(x[i], shift);
    }
}

/// Scale up norm values by `shift` bits.
/// Matches C `norm_scaleup()` which uses `SHL32`.
fn norm_scaleup(x: &mut [i32], n: usize, shift: i32) {
    debug_assert!(shift >= 0);
    if shift <= 0 {
        return;
    }
    for i in 0..n {
        x[i] = shl32(x[i], shift);
    }
}

// ===========================================================================
// Exp rotation (spreading)
// ===========================================================================

/// Single-pass Givens rotation butterfly.
/// Matches C `exp_rotation1()`.
///
/// Applies a forward sweep (left-to-right) then backward sweep (right-to-left)
/// between elements separated by `stride`, using rotation matrix [c, -s; s, c].
fn exp_rotation1(x: &mut [i32], len: usize, stride: usize, c: i32, s: i32) {
    let ms = neg16(s);
    norm_scaledown(x, len, NORM_SHIFT - 14);

    // Forward sweep: left to right
    if len > stride {
        for i in 0..len - stride {
            let x1 = x[i];
            let x2 = x[i + stride];
            // new x2 = c*x2 + s*x1
            x[i + stride] = extract16(pshr32(mac16_16(mult16_16(c, x2), s, x1), 15));
            // new x1 = c*x1 - s*x2
            x[i] = extract16(pshr32(mac16_16(mult16_16(c, x1), ms, x2), 15));
        }
    }

    // Backward sweep: right to left
    if len >= 2 * stride + 1 {
        for i in (0..len - 2 * stride).rev() {
            let x1 = x[i];
            let x2 = x[i + stride];
            x[i + stride] = extract16(pshr32(mac16_16(mult16_16(c, x2), s, x1), 15));
            x[i] = extract16(pshr32(mac16_16(mult16_16(c, x1), ms, x2), 15));
        }
    }

    norm_scaleup(x, len, NORM_SHIFT - 14);
}

/// Spreading rotation applied before/after quantization.
/// Matches C `exp_rotation()`.
///
/// - `dir > 0`: forward rotation (before quantization)
/// - `dir < 0`: inverse rotation (after quantization / during decode)
pub fn exp_rotation(x: &mut [i32], len: i32, dir: i32, stride: i32, k: i32, spread: i32) {
    if 2 * k >= len || spread == SPREAD_NONE {
        return;
    }

    let factor = SPREAD_FACTOR[(spread - 1) as usize];

    // gain = len / (len + factor*K) in Q15-ish via celt_div
    let gain = celt_div(mult16_16(Q15_ONE, len), len + factor * k);
    // theta = gain^2 / 2
    let theta = half16(mult16_16_q15(gain, gain));

    let c = celt_cos_norm(extend32(theta));
    // sin(theta) = cos(pi/2 - theta) via quarter-period identity
    let s = celt_cos_norm(extend32(sub16(Q15ONE, theta)));

    // Compute secondary stride (approx sqrt(len/stride)) when len >= 8*stride
    let mut stride2: i32 = 0;
    if len >= 8 * stride {
        stride2 = 1;
        while (stride2 * stride2 + stride2) * stride + (stride >> 2) < len {
            stride2 += 1;
        }
    }

    // Process each sub-block independently
    let block_len = celt_udiv(len as u32, stride as u32) as usize;

    for i in 0..stride as usize {
        let offset = i * block_len;
        let block = &mut x[offset..offset + block_len];
        if dir < 0 {
            // Inverse rotation: stride2 first (if any), then stride 1
            if stride2 != 0 {
                exp_rotation1(block, block_len, stride2 as usize, s, c);
            }
            exp_rotation1(block, block_len, 1, c, s);
        } else {
            // Forward rotation: stride 1 first, then stride2 (if any)
            exp_rotation1(block, block_len, 1, c, -s);
            if stride2 != 0 {
                exp_rotation1(block, block_len, stride2 as usize, s, -c);
            }
        }
    }
}

// ===========================================================================
// Normalise residual (internal)
// ===========================================================================

/// Convert integer pulse vector to normalized coefficients.
/// Matches C `normalise_residual()` (non-QEXT path, shift=0).
///
/// Uses 32-bit precision rsqrt (`celt_rsqrt_norm32`) for accuracy.
/// Output is in Q(NORM_SHIFT) format.
fn normalise_residual(iy: &[i32], x: &mut [i32], n: usize, ryy: i32, gain: i32) {
    let k = celt_ilog2(ryy) >> 1;
    // Normalize Ryy to approximately [2^29, 2^31) for rsqrt_norm32
    let t = vshr32(ryy, 2 * (k - 7) - 15);
    let g = mult32_32_q31(celt_rsqrt_norm32(t), gain);
    for i in 0..n {
        x[i] = vshr32(mult16_32_q15(iy[i], g), k + 15 - NORM_SHIFT);
    }
}

// ===========================================================================
// Collapse mask (internal)
// ===========================================================================

/// Compute bitmask of short blocks that received at least one pulse.
/// Bit `i` is set if any element in block `i` of the pulse vector is nonzero.
/// Matches C `extract_collapse_mask()`.
fn extract_collapse_mask(iy: &[i32], n: i32, b: i32) -> u32 {
    if b <= 1 {
        return 1;
    }
    let n0 = celt_udiv(n as u32, b as u32) as usize;
    let mut collapse_mask: u32 = 0;
    for i in 0..b as usize {
        let mut tmp: u32 = 0;
        for j in 0..n0 {
            tmp |= iy[i * n0 + j] as u32;
        }
        if tmp != 0 {
            collapse_mask |= 1 << i;
        }
    }
    collapse_mask
}

// ===========================================================================
// PVQ search
// ===========================================================================

/// Find the K-pulse integer vector closest to X on the L1 sphere.
///
/// On entry, `x` contains the (normalized) band coefficients.
/// On exit, `x` contains absolute values (signs stripped) and `iy` contains
/// the signed integer pulse vector with `sum(|iy[j]|) == k`.
///
/// Returns `yy` -- the squared norm of the pulse vector.
///
/// Matches C `op_pvq_search_c()`.
fn op_pvq_search(x: &mut [i32], iy: &mut [i32], k: i32, n: usize) -> i32 {
    let mut y: Vec<i32> = vec![0i32; n];
    let mut signx: Vec<i32> = vec![0i32; n];

    // Fixed-point prescaling: compute shift so X values fit in Q14 for
    // safe 16x16 multiply-accumulate in the greedy loop
    {
        let shift_raw = (celt_ilog2(1 + celt_inner_prod_norm_shift(x, x, n)) + 1) / 2;
        let shift = imax(0, shift_raw + (NORM_SHIFT - 14) - 14);
        norm_scaledown(x, n, shift);
    }

    // Strip signs: store sign flags, replace X with absolute values
    for j in 0..n {
        signx[j] = if x[j] < 0 { 1 } else { 0 };
        x[j] = abs16(x[j]);
        iy[j] = 0;
        y[j] = 0;
    }

    let mut xy: i32 = 0;
    let mut yy: i32 = 0;
    let mut pulses_left: i32 = k;

    // Pre-search: project X onto the L1 sphere of radius K when K > N/2.
    // This gives a good initial approximation, reducing the number of
    // greedy refinement iterations.
    if k > (n as i32 >> 1) {
        let mut sum: i32 = 0;
        for j in 0..n {
            sum += x[j];
        }

        // Degenerate input (near-zero or too small): replace with single pulse at [0]
        if sum <= k {
            x[0] = qconst16(1.0, 14); // 16384
            for j in 1..n {
                x[j] = 0;
            }
            sum = qconst16(1.0, 14);
        }

        // rcp = K / sum in Q15 (via celt_rcp which returns ~2^31/sum)
        let rcp = extract16(mult16_32_q16(k, celt_rcp(sum)));

        for j in 0..n {
            // Round toward zero -- critical to not exceed K total pulses
            iy[j] = mult16_16_q15(x[j], rcp);
            y[j] = iy[j];
            yy = mac16_16(yy, y[j], y[j]);
            xy = mac16_16(xy, x[j], y[j]);
            // Store 2*iy[j] to avoid multiply by 2 in the greedy inner loop
            y[j] *= 2;
            pulses_left -= iy[j];
        }
    }
    debug_assert!(pulses_left >= 0);

    // Safety check: if too many pulses remain (shouldn't happen except on
    // degenerate input like silence), dump them all into bin 0
    if pulses_left > n as i32 + 3 {
        let tmp = pulses_left;
        yy = mac16_16(yy, tmp, tmp);
        yy = mac16_16(yy, tmp, y[0]);
        iy[0] += pulses_left;
        pulses_left = 0;
    }

    // Greedy refinement: place remaining pulses one at a time.
    // For each pulse, find the dimension where adding it maximizes Rxy/sqrt(Ryy).
    // Uses division-free comparison: best_den * Rxy^2 > Ryy * best_num.
    for i in 0..pulses_left {
        // Right-shift to keep Rxy in 16-bit range after accumulation
        let rshift = 1 + celt_ilog2(k - pulses_left + i + 1);

        let mut best_id: usize = 0;

        // The squared magnitude term (+1 per pulse) is always added
        yy = add16(yy, 1);

        // Score position 0 (outside loop to reduce branch mispredictions)
        let rxy_0 = extract16(shr32(add32(xy, extend32(x[0])), rshift));
        let ryy_0 = add16(yy, y[0]);
        // Approximate score: Rxy^2 (we maximize Rxy^2 / Ryy)
        let rxy_0_sq = mult16_16_q15(rxy_0, rxy_0);
        let mut best_den = ryy_0;
        let mut best_num: i32 = rxy_0_sq;

        // Score positions 1..N-1
        for j in 1..n {
            let rxy = extract16(shr32(add32(xy, extend32(x[j])), rshift));
            // y[j] stores 2*iy[j], so this adds the cross-term without multiply
            let ryy = add16(yy, y[j]);
            let rxy_sq = mult16_16_q15(rxy, rxy);

            // Division-free comparison: best_den * Rxy^2 > Ryy * best_num
            if mult16_16(best_den, rxy_sq) > mult16_16(ryy, best_num) {
                best_den = ryy;
                best_num = rxy_sq;
                best_id = j;
            }
        }

        // Commit the pulse to the best position
        xy = add32(xy, extend32(x[best_id]));
        yy = add16(yy, y[best_id]);

        // y stores 2*iy, so increment by 2
        y[best_id] += 2;
        iy[best_id] += 1;
    }

    // Restore original signs using branchless flip:
    // signx=0 -> (iy ^ 0) + 0 = iy  (unchanged)
    // signx=1 -> (iy ^ -1) + 1 = -iy (negated)
    for j in 0..n {
        iy[j] = (iy[j] ^ (-signx[j])) + signx[j];
    }

    yy
}

// ===========================================================================
// Top-level encode / decode
// ===========================================================================

/// PVQ encoder: search for best K-pulse approximation, encode via CWRS,
/// optionally resynthesize the quantized signal.
///
/// Returns the collapse mask (bitmask of short blocks that received pulses).
///
/// Matches C `alg_quant()` (non-QEXT path).
pub fn alg_quant<EC: EcCoder>(
    ec: &mut EC,
    x: &mut [i32],
    n: i32,
    k: i32,
    spread: i32,
    b: i32,
    gain: i32,
    resynth: bool,
) -> u32 {
    debug_assert!(k > 0, "alg_quant() needs at least one pulse");
    debug_assert!(n > 1, "alg_quant() needs at least two dimensions");

    let nu = n as usize;
    // Extra 3 elements for SIMD headroom (matching C: ALLOC(iy, N+3, int))
    let mut iy = vec![0i32; nu + 3];

    // Apply forward spreading rotation
    exp_rotation(x, n, 1, b, k, spread);

    // Find the best K-pulse approximation
    let yy = op_pvq_search(&mut x[..nu], &mut iy, k, nu);
    let collapse_mask = extract_collapse_mask(&iy, n, b);

    // Encode pulse vector via CWRS combinatorial coding
    let index = icwrs(n, &iy[..nu]);
    let total = celt_pvq_v(n, k);
    ec.ec_enc_uint(index, total);

    if resynth {
        // Reconstruct the quantized signal from the pulse vector
        normalise_residual(&iy, x, nu, yy, gain);
    }

    if resynth {
        // Undo the spreading rotation for the reconstructed signal
        exp_rotation(x, n, -1, b, k, spread);
    }

    collapse_mask
}

/// PVQ decoder: read pulse vector from entropy coder, reconstruct the signal.
///
/// Returns the collapse mask.
///
/// Matches C `alg_unquant()` (non-QEXT path).
pub fn alg_unquant<EC: EcCoder>(
    ec: &mut EC,
    x: &mut [i32],
    n: i32,
    k: i32,
    spread: i32,
    b: i32,
    gain: i32,
) -> u32 {
    debug_assert!(k > 0, "alg_unquant() needs at least one pulse");
    debug_assert!(n > 1, "alg_unquant() needs at least two dimensions");

    let nu = n as usize;
    let mut iy = vec![0i32; nu];

    // Decode pulse vector via CWRS combinatorial coding
    let total = celt_pvq_v(n, k);
    let index = ec.ec_dec_uint(total);
    let ryy = cwrsi(n, k, index, &mut iy);

    // Reconstruct signal from pulse vector and apply inverse rotation
    normalise_residual(&iy, x, nu, ryy, gain);
    exp_rotation(x, n, -1, b, k, spread);
    let collapse_mask = extract_collapse_mask(&iy, n, b);

    collapse_mask
}

// ===========================================================================
// Renormalize vector
// ===========================================================================

/// Rescale X so its L2 norm equals `gain`.
/// Uses 16-bit precision rsqrt (sufficient for correcting norm drift
/// during band processing).
///
/// Matches C `renormalise_vector()`.
pub fn renormalise_vector(x: &mut [i32], n: usize, gain: i32) {
    norm_scaledown(x, n, NORM_SHIFT - 14);
    let e = EPSILON + celt_inner_prod_norm(x, x, n);
    let k = celt_ilog2(e) >> 1;
    let t = vshr32(e, 2 * (k - 7));
    // g = rsqrt(E) * gain
    let g = mult32_32_q31(celt_rsqrt_norm(t), gain);
    for i in 0..n {
        x[i] = extract16(pshr32(mult16_16(g, x[i]), k + 15 - 14));
    }
    norm_scaleup(x, n, NORM_SHIFT - 14);
}

// ===========================================================================
// Stereo angle computation
// ===========================================================================

/// Compute the quantized stereo angle between X and Y.
///
/// For `stereo=true` (MS stereo): computes mid=(X+Y)/2 and side=(X-Y)/2 energies.
/// For `stereo=false` (mono split): computes energy of X and Y directly.
///
/// Returns `itheta = atan2(side_energy, mid_energy)` in fixed-point.
/// Range is [0, 16384] in Q14 where 16384 represents pi/2.
///
/// Matches C `stereo_itheta()`.
pub fn stereo_itheta(x: &[i32], y: &[i32], stereo: bool, n: usize) -> i32 {
    let mut emid: i32 = 0;
    let mut eside: i32 = 0;

    if stereo {
        for i in 0..n {
            // Shift down to Q13 before squaring to prevent overflow
            let m = pshr32(add32(x[i], y[i]), NORM_SHIFT - 13);
            let s = pshr32(sub32(x[i], y[i]), NORM_SHIFT - 13);
            emid = mac16_16(emid, m, m);
            eside = mac16_16(eside, s, s);
        }
    } else {
        emid += celt_inner_prod_norm_shift(x, x, n);
        eside += celt_inner_prod_norm_shift(y, y, n);
    }

    let mid = celt_sqrt32(emid);
    let side = celt_sqrt32(eside);

    // atan2(side, mid) in Q14 fixed-point
    celt_atan2p_norm(side, mid)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_celt_inner_prod_norm_shift_basic() {
        // Zero vectors
        let z = [0i32; 4];
        assert_eq!(celt_inner_prod_norm_shift(&z, &z, 4), 0);

        // Unit vector in Q24: 2^24 in one dimension
        let a = [1 << NORM_SHIFT, 0, 0, 0];
        // Expected: (2^24)^2 >> 20 = 2^48 >> 20 = 2^28
        let result = celt_inner_prod_norm_shift(&a, &a, 4);
        assert_eq!(result, 1 << 28);
    }

    #[test]
    fn test_norm_scaledown_scaleup_roundtrip() {
        let original = [1000i32, -2000, 3000, -4000];
        let mut x = original;
        let shift = NORM_SHIFT - 14; // = 10

        norm_scaledown(&mut x, 4, shift);
        // After scaledown by 10, values should be approximately original >> 10
        // with rounding: 1000 + 512 >> 10 = 1512 >> 10 = 1
        assert_eq!(x[0], 1);
        assert_eq!(x[1], -2); // (-2000 + 512) >> 10 = -1488 >> 10 = -2 (arithmetic shift)

        norm_scaleup(&mut x, 4, shift);
        // After scaleup, values are scaled back but with rounding loss
        // The values won't exactly match original due to quantization
        assert_ne!(x[0], original[0]); // rounding loss expected
    }

    #[test]
    fn test_extract_collapse_mask_single_block() {
        let iy = [1, 0, -1, 0];
        assert_eq!(extract_collapse_mask(&iy, 4, 1), 1);
    }

    #[test]
    fn test_extract_collapse_mask_multiple_blocks() {
        // 4 elements, 2 blocks of 2
        let iy = [0, 0, 1, 0]; // block 0: [0,0] = no pulses, block 1: [1,0] = has pulses
        assert_eq!(extract_collapse_mask(&iy, 4, 2), 0b10);

        let iy = [1, 0, 0, 0]; // block 0: [1,0] = has pulses, block 1: [0,0] = no pulses
        assert_eq!(extract_collapse_mask(&iy, 4, 2), 0b01);

        let iy = [1, 0, 0, -1]; // both blocks have pulses
        assert_eq!(extract_collapse_mask(&iy, 4, 2), 0b11);
    }

    #[test]
    fn test_op_pvq_search_single_pulse() {
        // N=2, K=1: one pulse in two dimensions
        // X = [1.0, 0.0] in Q24 -> should place pulse at position 0
        let mut x = [1 << NORM_SHIFT, 0];
        let mut iy = [0i32; 2];

        let yy = op_pvq_search(&mut x, &mut iy, 1, 2);

        // One pulse total: |iy[0]| + |iy[1]| = 1
        let sum: i32 = iy.iter().map(|v| v.abs()).sum();
        assert_eq!(sum, 1);
        // Squared norm = 1
        assert!(yy >= 1);
    }

    #[test]
    fn test_op_pvq_search_preserves_pulse_count() {
        // N=4, K=8: pulse count must be preserved
        let mut x = [
            (0.5 * (1 << NORM_SHIFT) as f64) as i32,
            (0.5 * (1 << NORM_SHIFT) as f64) as i32,
            (0.5 * (1 << NORM_SHIFT) as f64) as i32,
            (0.5 * (1 << NORM_SHIFT) as f64) as i32,
        ];
        let mut iy = [0i32; 4];

        let _yy = op_pvq_search(&mut x, &mut iy, 8, 4);

        let sum: i32 = iy.iter().map(|v| v.abs()).sum();
        assert_eq!(sum, 8, "total pulse count must equal K=8");
    }

    #[test]
    fn test_op_pvq_search_negative_input() {
        // X = [-1.0, 0.0] in Q24 -> pulse should be negative at position 0
        let mut x = [-(1 << NORM_SHIFT), 0];
        let mut iy = [0i32; 2];

        let _yy = op_pvq_search(&mut x, &mut iy, 1, 2);

        assert_eq!(iy[0], -1, "pulse should be negative matching input sign");
        assert_eq!(iy[1], 0);
    }

    #[test]
    fn test_stereo_itheta_mono_equal_energy() {
        // Two vectors with equal energy -> itheta should be near pi/4
        // celt_atan2p_norm returns Q30 where 2^30 = pi/2, so pi/4 = 2^29
        let scale = 1 << (NORM_SHIFT - 2); // moderate Q24 value
        let x = [scale, scale, 0, 0];
        let y = [0, 0, scale, scale];

        let itheta = stereo_itheta(&x, &y, false, 4);

        let expected = 1 << 29; // pi/4 in Q30
        assert!(
            (itheta - expected).abs() < (1 << 20),
            "itheta={} should be near {} for equal energies",
            itheta,
            expected,
        );
    }

    #[test]
    fn test_stereo_itheta_mono_zero_side() {
        // Y is zero -> eside = 0 -> itheta should be 0
        let scale = 1 << (NORM_SHIFT - 2);
        let x = [scale, scale, scale, scale];
        let y = [0, 0, 0, 0];

        let itheta = stereo_itheta(&x, &y, false, 4);
        assert_eq!(itheta, 0, "itheta should be 0 when side energy is 0");
    }

    #[test]
    fn test_renormalise_vector_preserves_direction() {
        // Create a vector in Q(NORM_SHIFT=24)
        // Gain must be large enough for the internal rsqrt * gain product to survive
        // the Q31 right-shift. Using NORM_SCALING as gain.
        let gain = NORM_SCALING; // 2^24
        let mut x = [
            (0.6 * (1 << NORM_SHIFT) as f64) as i32,
            (0.8 * (1 << NORM_SHIFT) as f64) as i32,
        ];
        let original_sign_0 = x[0] > 0;
        let original_sign_1 = x[1] > 0;

        renormalise_vector(&mut x, 2, gain);

        // Signs should be preserved
        assert_eq!(x[0] > 0, original_sign_0);
        assert_eq!(x[1] > 0, original_sign_1);

        // Vector should be non-zero
        assert!(x[0] != 0 || x[1] != 0);
    }

    #[test]
    fn test_exp_rotation_skip_conditions() {
        // Should skip when 2*K >= len
        let mut x = [100i32, 200, 300, 400];
        let original = x;
        exp_rotation(&mut x, 4, 1, 1, 3, 2); // 2*3 = 6 >= 4 -> skip
        assert_eq!(x, original);

        // Should skip for SPREAD_NONE
        exp_rotation(&mut x, 4, 1, 1, 1, SPREAD_NONE);
        assert_eq!(x, original);
    }
}
