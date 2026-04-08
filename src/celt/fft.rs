//! Complex FFT implementation for the CELT layer of Opus.
//!
//! Port of `kiss_fft.c` from xiph/opus. Fixed-point (non-QEXT) path.
//! Uses an iterative mixed-radix decimation-in-time algorithm with
//! radix-2, 3, 4, and 5 butterfly stages.

use crate::types::*;
use crate::{uc, uc_mut, uc_set};

// =========================================================================
// Constants
// =========================================================================

pub const MAXFACTORS: usize = 8;

// =========================================================================
// Types
// =========================================================================

/// Complex data sample (i32 components). Matches C `kiss_fft_cpx`.
#[derive(Clone, Copy, Debug, Default)]
pub struct KissFftCpx {
    pub r: i32,
    pub i: i32,
}

/// Complex twiddle factor (i16 components). Matches C `kiss_twiddle_cpx`.
#[derive(Clone, Copy, Debug, Default)]
pub struct KissTwiddleCpx {
    pub r: i16,
    pub i: i16,
}

/// FFT configuration state. Matches C `kiss_fft_state`.
pub struct KissFftState {
    pub nfft: i32,
    /// Scale factor: 1/nfft in Q15. Stored as i32 but value fits in i16.
    pub scale: i32,
    /// celt_ilog2(nfft), used for fixed-point downshift budget.
    pub scale_shift: i32,
    /// Twiddle sharing: -1 = owns twiddles, >=0 = shared at stride 1<<shift.
    pub shift: i32,
    /// Factorization: [p0,m0, p1,m1, ...] pairs.
    pub factors: [i16; 2 * MAXFACTORS],
    /// Bit-reversal permutation table.
    pub bitrev: &'static [i16],
    /// Twiddle factors (shared across sub-sampled FFT sizes).
    pub twiddles: &'static [KissTwiddleCpx],
}

// =========================================================================
// Complex arithmetic helpers — match _kiss_fft_guts.h macros
// =========================================================================

/// `S_MUL(a,b)` = `MULT16_32_Q15(b, a)`.
/// `a` is data (i32), `b` is twiddle coefficient (i16 range).
#[inline(always)]
fn s_mul(a: i32, b: i32) -> i32 {
    mult16_32_q15(b, a)
}

/// `S_MUL2(a,b)` = `MULT16_32_Q16(b, a)`. Only used for initial FFT scaling.
#[inline(always)]
fn s_mul2(a: i32, b: i32) -> i32 {
    mult16_32_q16(b, a)
}

/// Complex multiply: data (i32) × twiddle (i16).
#[inline(always)]
fn c_mul(a: KissFftCpx, b: KissTwiddleCpx) -> KissFftCpx {
    KissFftCpx {
        r: sub32_ovflw(s_mul(a.r, b.r as i32), s_mul(a.i, b.i as i32)),
        i: add32_ovflw(s_mul(a.r, b.i as i32), s_mul(a.i, b.r as i32)),
    }
}

/// `C_ADD(a, b)` — wrapping complex addition.
#[inline(always)]
fn c_add(a: KissFftCpx, b: KissFftCpx) -> KissFftCpx {
    KissFftCpx {
        r: add32_ovflw(a.r, b.r),
        i: add32_ovflw(a.i, b.i),
    }
}

/// `C_SUB(a, b)` — wrapping complex subtraction.
#[inline(always)]
fn c_sub(a: KissFftCpx, b: KissFftCpx) -> KissFftCpx {
    KissFftCpx {
        r: sub32_ovflw(a.r, b.r),
        i: sub32_ovflw(a.i, b.i),
    }
}

/// `C_ADDTO(res, a)` — wrapping in-place complex addition.
#[inline(always)]
fn c_addto(res: &mut KissFftCpx, a: KissFftCpx) {
    res.r = add32_ovflw(res.r, a.r);
    res.i = add32_ovflw(res.i, a.i);
}

/// `C_MULBYSCALAR(c, s)` — multiply complex by real scalar.
#[inline(always)]
fn c_mulbyscalar(c: &mut KissFftCpx, s: i32) {
    c.r = s_mul(c.r, s);
    c.i = s_mul(c.i, s);
}

/// `HALF_OF(x)` = `x >> 1`.
#[inline(always)]
fn half_of(x: i32) -> i32 {
    x >> 1
}

// =========================================================================
// Fixed-point downshift between butterfly stages
// =========================================================================

/// Reduce data magnitude between butterfly stages to prevent overflow.
/// `total` tracks remaining shift budget; `step` is the request for this stage.
fn fft_downshift(x: &mut [KissFftCpx], n: usize, total: &mut i32, step: i32) {
    let shift = imin(step, *total);
    *total -= shift;
    if shift == 1 {
        for j in 0..n {
            uc_mut!(x, j).r = shr32(uc!(x, j).r, 1);
            uc_mut!(x, j).i = shr32(uc!(x, j).i, 1);
        }
    } else if shift > 0 {
        for j in 0..n {
            uc_mut!(x, j).r = pshr32(uc!(x, j).r, shift);
            uc_mut!(x, j).i = pshr32(uc!(x, j).i, shift);
        }
    }
}

// =========================================================================
// Butterfly functions
// =========================================================================

/// Radix-2 butterfly, specialized for m=4 (follows a radix-4 stage).
/// Uses hardcoded W8 = cos(π/4) twiddle for the merged radix-2×4 optimization.
fn kf_bfly2(fout: &mut [KissFftCpx], _m: usize, n: usize) {
    // W8 = cos(π/4) in Q15
    let tw: i32 = qconst32(0.7071067812, 15);
    debug_assert!(_m == 4);
    let mut idx = 0;
    for _ in 0..n {
        // Element 0: trivial butterfly (twiddle = 1)
        let t = uc!(fout, idx + 4);
        uc_set!(fout, idx + 4, c_sub(uc!(fout, idx), t));
        c_addto(uc_mut!(fout, idx), t);

        // Element 1: twiddle = W8*(1+j)/√2
        let t = KissFftCpx {
            r: s_mul(add32_ovflw(uc!(fout, idx + 5).r, uc!(fout, idx + 5).i), tw),
            i: s_mul(sub32_ovflw(uc!(fout, idx + 5).i, uc!(fout, idx + 5).r), tw),
        };
        uc_set!(fout, idx + 5, c_sub(uc!(fout, idx + 1), t));
        c_addto(uc_mut!(fout, idx + 1), t);

        // Element 2: twiddle = -j
        let t = KissFftCpx {
            r: uc!(fout, idx + 6).i,
            i: neg32_ovflw(uc!(fout, idx + 6).r),
        };
        uc_set!(fout, idx + 6, c_sub(uc!(fout, idx + 2), t));
        c_addto(uc_mut!(fout, idx + 2), t);

        // Element 3: twiddle = W8 conjugate
        let t = KissFftCpx {
            r: s_mul(sub32_ovflw(uc!(fout, idx + 7).i, uc!(fout, idx + 7).r), tw),
            i: s_mul(
                neg32_ovflw(add32_ovflw(uc!(fout, idx + 7).i, uc!(fout, idx + 7).r)),
                tw,
            ),
        };
        uc_set!(fout, idx + 7, c_sub(uc!(fout, idx + 3), t));
        c_addto(uc_mut!(fout, idx + 3), t);

        idx += 8;
    }
}

/// Radix-4 butterfly. Degenerate case (m==1) uses no twiddle multiplications.
fn kf_bfly4(
    fout: &mut [KissFftCpx],
    fstride: usize,
    st: &KissFftState,
    m: usize,
    n: usize,
    mm: usize,
) {
    if m == 1 {
        // Degenerate case: all twiddles are 1
        let mut idx = 0;
        for _ in 0..n {
            let scratch0 = c_sub(uc!(fout, idx), uc!(fout, idx + 2));
            let tmp = uc!(fout, idx + 2);
            c_addto(uc_mut!(fout, idx), tmp);
            let scratch1 = c_add(uc!(fout, idx + 1), uc!(fout, idx + 3));
            uc_set!(fout, idx + 2, c_sub(uc!(fout, idx), scratch1));
            c_addto(uc_mut!(fout, idx), scratch1);
            let scratch1 = c_sub(uc!(fout, idx + 1), uc!(fout, idx + 3));

            uc_mut!(fout, idx + 1).r = add32_ovflw(scratch0.r, scratch1.i);
            uc_mut!(fout, idx + 1).i = sub32_ovflw(scratch0.i, scratch1.r);
            uc_mut!(fout, idx + 3).r = sub32_ovflw(scratch0.r, scratch1.i);
            uc_mut!(fout, idx + 3).i = add32_ovflw(scratch0.i, scratch1.r);
            idx += 4;
        }
    } else {
        let m2 = 2 * m;
        let m3 = 3 * m;
        let twiddles = st.twiddles;
        for i in 0..n {
            let base = i * mm;
            let mut tw1_idx = 0usize;
            let mut tw2_idx = 0usize;
            let mut tw3_idx = 0usize;
            for j in 0..m {
                let f = base + j;
                let scratch0 = c_mul(uc!(fout, f + m), uc!(twiddles, tw1_idx));
                let scratch1 = c_mul(uc!(fout, f + m2), uc!(twiddles, tw2_idx));
                let scratch2 = c_mul(uc!(fout, f + m3), uc!(twiddles, tw3_idx));

                let scratch5 = c_sub(uc!(fout, f), scratch1);
                c_addto(uc_mut!(fout, f), scratch1);
                let scratch3 = c_add(scratch0, scratch2);
                let scratch4 = c_sub(scratch0, scratch2);
                uc_set!(fout, f + m2, c_sub(uc!(fout, f), scratch3));
                tw1_idx += fstride;
                tw2_idx += fstride * 2;
                tw3_idx += fstride * 3;
                c_addto(uc_mut!(fout, f), scratch3);

                uc_mut!(fout, f + m).r = add32_ovflw(scratch5.r, scratch4.i);
                uc_mut!(fout, f + m).i = sub32_ovflw(scratch5.i, scratch4.r);
                uc_mut!(fout, f + m3).r = sub32_ovflw(scratch5.r, scratch4.i);
                uc_mut!(fout, f + m3).i = add32_ovflw(scratch5.i, scratch4.r);
            }
        }
    }
}

/// Radix-3 butterfly. Uses hardcoded sin(2π/3) constant.
fn kf_bfly3(
    fout: &mut [KissFftCpx],
    fstride: usize,
    st: &KissFftState,
    m: usize,
    n: usize,
    mm: usize,
) {
    let m2 = 2 * m;
    // epi3.i = -sin(2π/3) in Q15
    let epi3_i: i32 = -qconst32(0.86602540, 15);

    for i in 0..n {
        let base = i * mm;
        let mut tw1_idx = 0usize;
        let mut tw2_idx = 0usize;
        let twiddles = st.twiddles;
        for j in 0..m {
            let f = base + j;
            let scratch1 = c_mul(uc!(fout, f + m), uc!(twiddles, tw1_idx));
            let scratch2 = c_mul(uc!(fout, f + m2), uc!(twiddles, tw2_idx));

            let scratch3 = c_add(scratch1, scratch2);
            let mut scratch0 = c_sub(scratch1, scratch2);
            tw1_idx += fstride;
            tw2_idx += fstride * 2;

            uc_mut!(fout, f + m).r = sub32_ovflw(uc!(fout, f).r, half_of(scratch3.r));
            uc_mut!(fout, f + m).i = sub32_ovflw(uc!(fout, f).i, half_of(scratch3.i));

            c_mulbyscalar(&mut scratch0, epi3_i);

            c_addto(uc_mut!(fout, f), scratch3);

            uc_mut!(fout, f + m2).r = add32_ovflw(uc!(fout, f + m).r, scratch0.i);
            uc_mut!(fout, f + m2).i = sub32_ovflw(uc!(fout, f + m).i, scratch0.r);

            uc_mut!(fout, f + m).r = sub32_ovflw(uc!(fout, f + m).r, scratch0.i);
            uc_mut!(fout, f + m).i = add32_ovflw(uc!(fout, f + m).i, scratch0.r);
        }
    }
}

/// Radix-5 butterfly. Uses hardcoded DFT-5 rotation constants.
fn kf_bfly5(
    fout: &mut [KissFftCpx],
    fstride: usize,
    st: &KissFftState,
    m: usize,
    n: usize,
    mm: usize,
) {
    // ya = e^{-j2π/5}, yb = e^{-j4π/5} in Q15
    let ya_r: i32 = qconst32(0.30901699, 15);
    let ya_i: i32 = -qconst32(0.95105652, 15);
    let yb_r: i32 = -qconst32(0.80901699, 15);
    let yb_i: i32 = -qconst32(0.58778525, 15);
    let twiddles = st.twiddles;

    for i in 0..n {
        let base = i * mm;
        for u in 0..m {
            let f0 = base + u;
            let f1 = f0 + m;
            let f2 = f0 + 2 * m;
            let f3 = f0 + 3 * m;
            let f4 = f0 + 4 * m;

            let scratch0 = uc!(fout, f0);
            let scratch1 = c_mul(uc!(fout, f1), uc!(twiddles, u * fstride));
            let scratch2 = c_mul(uc!(fout, f2), uc!(twiddles, 2 * u * fstride));
            let scratch3 = c_mul(uc!(fout, f3), uc!(twiddles, 3 * u * fstride));
            let scratch4 = c_mul(uc!(fout, f4), uc!(twiddles, 4 * u * fstride));

            let scratch7 = c_add(scratch1, scratch4);
            let scratch10 = c_sub(scratch1, scratch4);
            let scratch8 = c_add(scratch2, scratch3);
            let scratch9 = c_sub(scratch2, scratch3);

            uc_mut!(fout, f0).r = add32_ovflw(uc!(fout, f0).r, add32_ovflw(scratch7.r, scratch8.r));
            uc_mut!(fout, f0).i = add32_ovflw(uc!(fout, f0).i, add32_ovflw(scratch7.i, scratch8.i));

            let scratch5 = KissFftCpx {
                r: add32_ovflw(
                    scratch0.r,
                    add32_ovflw(s_mul(scratch7.r, ya_r), s_mul(scratch8.r, yb_r)),
                ),
                i: add32_ovflw(
                    scratch0.i,
                    add32_ovflw(s_mul(scratch7.i, ya_r), s_mul(scratch8.i, yb_r)),
                ),
            };
            let scratch6 = KissFftCpx {
                r: add32_ovflw(s_mul(scratch10.i, ya_i), s_mul(scratch9.i, yb_i)),
                i: neg32_ovflw(add32_ovflw(
                    s_mul(scratch10.r, ya_i),
                    s_mul(scratch9.r, yb_i),
                )),
            };

            uc_set!(fout, f1, c_sub(scratch5, scratch6));
            uc_set!(fout, f4, c_add(scratch5, scratch6));

            let scratch11 = KissFftCpx {
                r: add32_ovflw(
                    scratch0.r,
                    add32_ovflw(s_mul(scratch7.r, yb_r), s_mul(scratch8.r, ya_r)),
                ),
                i: add32_ovflw(
                    scratch0.i,
                    add32_ovflw(s_mul(scratch7.i, yb_r), s_mul(scratch8.i, ya_r)),
                ),
            };
            let scratch12 = KissFftCpx {
                r: sub32_ovflw(s_mul(scratch9.i, ya_i), s_mul(scratch10.i, yb_i)),
                i: sub32_ovflw(s_mul(scratch10.r, yb_i), s_mul(scratch9.r, ya_i)),
            };

            uc_set!(fout, f2, c_add(scratch11, scratch12));
            uc_set!(fout, f3, c_sub(scratch11, scratch12));
        }
    }
}

// =========================================================================
// FFT public API
// =========================================================================

/// Internal FFT implementation. Operates on already bit-reversed data in-place.
/// `downshift` is the remaining right-shift budget (fixed-point only).
pub fn opus_fft_impl(st: &KissFftState, fout: &mut [KissFftCpx], mut downshift: i32) {
    let shift = if st.shift > 0 { st.shift as u32 } else { 0 };

    // Build fstride table
    let mut fstride_arr = [0usize; MAXFACTORS];
    fstride_arr[0] = 1;
    let mut l = 0usize;
    loop {
        let p = st.factors[2 * l] as usize;
        let m = st.factors[2 * l + 1];
        fstride_arr[l + 1] = fstride_arr[l] * p;
        l += 1;
        if m == 1 {
            break;
        }
    }

    let mut m = st.factors[2 * l - 1] as usize;
    for i in (0..l).rev() {
        let m2 = if i != 0 {
            st.factors[2 * i - 1] as usize
        } else {
            1
        };
        let fs = fstride_arr[i];
        match st.factors[2 * i] {
            2 => {
                fft_downshift(fout, st.nfft as usize, &mut downshift, 1);
                kf_bfly2(fout, m, fs);
            }
            4 => {
                fft_downshift(fout, st.nfft as usize, &mut downshift, 2);
                kf_bfly4(fout, fs << shift, st, m, fs, m2);
            }
            3 => {
                fft_downshift(fout, st.nfft as usize, &mut downshift, 2);
                kf_bfly3(fout, fs << shift, st, m, fs, m2);
            }
            5 => {
                fft_downshift(fout, st.nfft as usize, &mut downshift, 3);
                kf_bfly5(fout, fs << shift, st, m, fs, m2);
            }
            _ => panic!("unsupported FFT radix"),
        }
        m = m2;
    }
    // Apply any remaining downshift
    let remaining = downshift;
    fft_downshift(fout, st.nfft as usize, &mut downshift, remaining);
}

/// Forward FFT. Out-of-place (fin must not alias fout).
/// Scales input by 1/nfft during bit-reversal permutation.
pub fn opus_fft(st: &KissFftState, fin: &[KissFftCpx], fout: &mut [KissFftCpx]) {
    let scale = st.scale;
    let scale_shift = st.scale_shift - 1;
    let bitrev = st.bitrev;

    for i in 0..st.nfft as usize {
        let x = uc!(fin, i);
        let rev = uc!(bitrev, i) as usize;
        uc_mut!(fout, rev).r = s_mul2(x.r, scale);
        uc_mut!(fout, rev).i = s_mul2(x.i, scale);
    }
    opus_fft_impl(st, fout, scale_shift);
}

/// Inverse FFT. Out-of-place (fin must not alias fout).
/// Uses the conjugate-FFT-conjugate trick: no 1/N scaling.
pub fn opus_ifft(st: &KissFftState, fin: &[KissFftCpx], fout: &mut [KissFftCpx]) {
    let bitrev = st.bitrev;
    for i in 0..st.nfft as usize {
        let rev = uc!(bitrev, i) as usize;
        uc_set!(fout, rev, uc!(fin, i));
    }
    for i in 0..st.nfft as usize {
        uc_mut!(fout, i).i = -uc!(fout, i).i;
    }
    opus_fft_impl(st, fout, 0);
    for i in 0..st.nfft as usize {
        uc_mut!(fout, i).i = -uc!(fout, i).i;
    }
}

// =========================================================================
// Static tables — 48kHz mode (N=1920, FFT sizes 480/240/120/60)
// =========================================================================

const fn tw(r: i16, i: i16) -> KissTwiddleCpx {
    KissTwiddleCpx { r, i }
}

/// Shared twiddle table for all 48kHz FFT sizes. 480 entries (Q15).
/// Sub-sampled FFTs access at stride 1<<shift.
#[rustfmt::skip]
pub static FFT_TWIDDLES_48000_960: [KissTwiddleCpx; 480] = [
tw(32767, 0), tw(32765, -429), tw(32757, -858), tw(32743, -1286),
tw(32723, -1715), tw(32698, -2143), tw(32667, -2571), tw(32631, -2998),
tw(32588, -3425), tw(32541, -3851), tw(32488, -4277), tw(32429, -4702),
tw(32365, -5126), tw(32295, -5549), tw(32219, -5971), tw(32138, -6393),
tw(32052, -6813), tw(31960, -7232), tw(31863, -7650), tw(31760, -8066),
tw(31651, -8481), tw(31538, -8895), tw(31419, -9307), tw(31294, -9717),
tw(31164, -10126), tw(31029, -10533), tw(30888, -10938), tw(30743, -11342),
tw(30592, -11743), tw(30435, -12142), tw(30274, -12540), tw(30107, -12935),
tw(29935, -13328), tw(29758, -13719), tw(29576, -14107), tw(29389, -14493),
tw(29197, -14876), tw(28999, -15257), tw(28797, -15636), tw(28590, -16011),
tw(28378, -16384), tw(28161, -16754), tw(27939, -17121), tw(27713, -17485),
tw(27482, -17847), tw(27246, -18205), tw(27005, -18560), tw(26760, -18912),
tw(26510, -19261), tw(26255, -19606), tw(25997, -19948), tw(25733, -20286),
tw(25466, -20622), tw(25193, -20953), tw(24917, -21281), tw(24636, -21605),
tw(24351, -21926), tw(24062, -22243), tw(23769, -22556), tw(23472, -22865),
tw(23170, -23170), tw(22865, -23472), tw(22556, -23769), tw(22243, -24062),
tw(21926, -24351), tw(21605, -24636), tw(21281, -24917), tw(20953, -25193),
tw(20622, -25466), tw(20286, -25733), tw(19948, -25997), tw(19606, -26255),
tw(19261, -26510), tw(18912, -26760), tw(18560, -27005), tw(18205, -27246),
tw(17847, -27482), tw(17485, -27713), tw(17121, -27939), tw(16754, -28161),
tw(16384, -28378), tw(16011, -28590), tw(15636, -28797), tw(15257, -28999),
tw(14876, -29197), tw(14493, -29389), tw(14107, -29576), tw(13719, -29758),
tw(13328, -29935), tw(12935, -30107), tw(12540, -30274), tw(12142, -30435),
tw(11743, -30592), tw(11342, -30743), tw(10938, -30888), tw(10533, -31029),
tw(10126, -31164), tw(9717, -31294), tw(9307, -31419), tw(8895, -31538),
tw(8481, -31651), tw(8066, -31760), tw(7650, -31863), tw(7232, -31960),
tw(6813, -32052), tw(6393, -32138), tw(5971, -32219), tw(5549, -32295),
tw(5126, -32365), tw(4702, -32429), tw(4277, -32488), tw(3851, -32541),
tw(3425, -32588), tw(2998, -32631), tw(2571, -32667), tw(2143, -32698),
tw(1715, -32723), tw(1286, -32743), tw(858, -32757), tw(429, -32765),
tw(0, -32767), tw(-429, -32765), tw(-858, -32757), tw(-1286, -32743),
tw(-1715, -32723), tw(-2143, -32698), tw(-2571, -32667), tw(-2998, -32631),
tw(-3425, -32588), tw(-3851, -32541), tw(-4277, -32488), tw(-4702, -32429),
tw(-5126, -32365), tw(-5549, -32295), tw(-5971, -32219), tw(-6393, -32138),
tw(-6813, -32052), tw(-7232, -31960), tw(-7650, -31863), tw(-8066, -31760),
tw(-8481, -31651), tw(-8895, -31538), tw(-9307, -31419), tw(-9717, -31294),
tw(-10126, -31164), tw(-10533, -31029), tw(-10938, -30888), tw(-11342, -30743),
tw(-11743, -30592), tw(-12142, -30435), tw(-12540, -30274), tw(-12935, -30107),
tw(-13328, -29935), tw(-13719, -29758), tw(-14107, -29576), tw(-14493, -29389),
tw(-14876, -29197), tw(-15257, -28999), tw(-15636, -28797), tw(-16011, -28590),
tw(-16384, -28378), tw(-16754, -28161), tw(-17121, -27939), tw(-17485, -27713),
tw(-17847, -27482), tw(-18205, -27246), tw(-18560, -27005), tw(-18912, -26760),
tw(-19261, -26510), tw(-19606, -26255), tw(-19948, -25997), tw(-20286, -25733),
tw(-20622, -25466), tw(-20953, -25193), tw(-21281, -24917), tw(-21605, -24636),
tw(-21926, -24351), tw(-22243, -24062), tw(-22556, -23769), tw(-22865, -23472),
tw(-23170, -23170), tw(-23472, -22865), tw(-23769, -22556), tw(-24062, -22243),
tw(-24351, -21926), tw(-24636, -21605), tw(-24917, -21281), tw(-25193, -20953),
tw(-25466, -20622), tw(-25733, -20286), tw(-25997, -19948), tw(-26255, -19606),
tw(-26510, -19261), tw(-26760, -18912), tw(-27005, -18560), tw(-27246, -18205),
tw(-27482, -17847), tw(-27713, -17485), tw(-27939, -17121), tw(-28161, -16754),
tw(-28378, -16384), tw(-28590, -16011), tw(-28797, -15636), tw(-28999, -15257),
tw(-29197, -14876), tw(-29389, -14493), tw(-29576, -14107), tw(-29758, -13719),
tw(-29935, -13328), tw(-30107, -12935), tw(-30274, -12540), tw(-30435, -12142),
tw(-30592, -11743), tw(-30743, -11342), tw(-30888, -10938), tw(-31029, -10533),
tw(-31164, -10126), tw(-31294, -9717), tw(-31419, -9307), tw(-31538, -8895),
tw(-31651, -8481), tw(-31760, -8066), tw(-31863, -7650), tw(-31960, -7232),
tw(-32052, -6813), tw(-32138, -6393), tw(-32219, -5971), tw(-32295, -5549),
tw(-32365, -5126), tw(-32429, -4702), tw(-32488, -4277), tw(-32541, -3851),
tw(-32588, -3425), tw(-32631, -2998), tw(-32667, -2571), tw(-32698, -2143),
tw(-32723, -1715), tw(-32743, -1286), tw(-32757, -858), tw(-32765, -429),
tw(-32767, 0), tw(-32765, 429), tw(-32757, 858), tw(-32743, 1286),
tw(-32723, 1715), tw(-32698, 2143), tw(-32667, 2571), tw(-32631, 2998),
tw(-32588, 3425), tw(-32541, 3851), tw(-32488, 4277), tw(-32429, 4702),
tw(-32365, 5126), tw(-32295, 5549), tw(-32219, 5971), tw(-32138, 6393),
tw(-32052, 6813), tw(-31960, 7232), tw(-31863, 7650), tw(-31760, 8066),
tw(-31651, 8481), tw(-31538, 8895), tw(-31419, 9307), tw(-31294, 9717),
tw(-31164, 10126), tw(-31029, 10533), tw(-30888, 10938), tw(-30743, 11342),
tw(-30592, 11743), tw(-30435, 12142), tw(-30274, 12540), tw(-30107, 12935),
tw(-29935, 13328), tw(-29758, 13719), tw(-29576, 14107), tw(-29389, 14493),
tw(-29197, 14876), tw(-28999, 15257), tw(-28797, 15636), tw(-28590, 16011),
tw(-28378, 16384), tw(-28161, 16754), tw(-27939, 17121), tw(-27713, 17485),
tw(-27482, 17847), tw(-27246, 18205), tw(-27005, 18560), tw(-26760, 18912),
tw(-26510, 19261), tw(-26255, 19606), tw(-25997, 19948), tw(-25733, 20286),
tw(-25466, 20622), tw(-25193, 20953), tw(-24917, 21281), tw(-24636, 21605),
tw(-24351, 21926), tw(-24062, 22243), tw(-23769, 22556), tw(-23472, 22865),
tw(-23170, 23170), tw(-22865, 23472), tw(-22556, 23769), tw(-22243, 24062),
tw(-21926, 24351), tw(-21605, 24636), tw(-21281, 24917), tw(-20953, 25193),
tw(-20622, 25466), tw(-20286, 25733), tw(-19948, 25997), tw(-19606, 26255),
tw(-19261, 26510), tw(-18912, 26760), tw(-18560, 27005), tw(-18205, 27246),
tw(-17847, 27482), tw(-17485, 27713), tw(-17121, 27939), tw(-16754, 28161),
tw(-16384, 28378), tw(-16011, 28590), tw(-15636, 28797), tw(-15257, 28999),
tw(-14876, 29197), tw(-14493, 29389), tw(-14107, 29576), tw(-13719, 29758),
tw(-13328, 29935), tw(-12935, 30107), tw(-12540, 30274), tw(-12142, 30435),
tw(-11743, 30592), tw(-11342, 30743), tw(-10938, 30888), tw(-10533, 31029),
tw(-10126, 31164), tw(-9717, 31294), tw(-9307, 31419), tw(-8895, 31538),
tw(-8481, 31651), tw(-8066, 31760), tw(-7650, 31863), tw(-7232, 31960),
tw(-6813, 32052), tw(-6393, 32138), tw(-5971, 32219), tw(-5549, 32295),
tw(-5126, 32365), tw(-4702, 32429), tw(-4277, 32488), tw(-3851, 32541),
tw(-3425, 32588), tw(-2998, 32631), tw(-2571, 32667), tw(-2143, 32698),
tw(-1715, 32723), tw(-1286, 32743), tw(-858, 32757), tw(-429, 32765),
tw(0, 32767), tw(429, 32765), tw(858, 32757), tw(1286, 32743),
tw(1715, 32723), tw(2143, 32698), tw(2571, 32667), tw(2998, 32631),
tw(3425, 32588), tw(3851, 32541), tw(4277, 32488), tw(4702, 32429),
tw(5126, 32365), tw(5549, 32295), tw(5971, 32219), tw(6393, 32138),
tw(6813, 32052), tw(7232, 31960), tw(7650, 31863), tw(8066, 31760),
tw(8481, 31651), tw(8895, 31538), tw(9307, 31419), tw(9717, 31294),
tw(10126, 31164), tw(10533, 31029), tw(10938, 30888), tw(11342, 30743),
tw(11743, 30592), tw(12142, 30435), tw(12540, 30274), tw(12935, 30107),
tw(13328, 29935), tw(13719, 29758), tw(14107, 29576), tw(14493, 29389),
tw(14876, 29197), tw(15257, 28999), tw(15636, 28797), tw(16011, 28590),
tw(16384, 28378), tw(16754, 28161), tw(17121, 27939), tw(17485, 27713),
tw(17847, 27482), tw(18205, 27246), tw(18560, 27005), tw(18912, 26760),
tw(19261, 26510), tw(19606, 26255), tw(19948, 25997), tw(20286, 25733),
tw(20622, 25466), tw(20953, 25193), tw(21281, 24917), tw(21605, 24636),
tw(21926, 24351), tw(22243, 24062), tw(22556, 23769), tw(22865, 23472),
tw(23170, 23170), tw(23472, 22865), tw(23769, 22556), tw(24062, 22243),
tw(24351, 21926), tw(24636, 21605), tw(24917, 21281), tw(25193, 20953),
tw(25466, 20622), tw(25733, 20286), tw(25997, 19948), tw(26255, 19606),
tw(26510, 19261), tw(26760, 18912), tw(27005, 18560), tw(27246, 18205),
tw(27482, 17847), tw(27713, 17485), tw(27939, 17121), tw(28161, 16754),
tw(28378, 16384), tw(28590, 16011), tw(28797, 15636), tw(28999, 15257),
tw(29197, 14876), tw(29389, 14493), tw(29576, 14107), tw(29758, 13719),
tw(29935, 13328), tw(30107, 12935), tw(30274, 12540), tw(30435, 12142),
tw(30592, 11743), tw(30743, 11342), tw(30888, 10938), tw(31029, 10533),
tw(31164, 10126), tw(31294, 9717), tw(31419, 9307), tw(31538, 8895),
tw(31651, 8481), tw(31760, 8066), tw(31863, 7650), tw(31960, 7232),
tw(32052, 6813), tw(32138, 6393), tw(32219, 5971), tw(32295, 5549),
tw(32365, 5126), tw(32429, 4702), tw(32488, 4277), tw(32541, 3851),
tw(32588, 3425), tw(32631, 2998), tw(32667, 2571), tw(32698, 2143),
tw(32723, 1715), tw(32743, 1286), tw(32757, 858), tw(32765, 429),
];

// Bit-reversal permutation tables

#[rustfmt::skip]
pub static FFT_BITREV480: [i16; 480] = [
0, 96, 192, 288, 384, 32, 128, 224, 320, 416, 64, 160, 256, 352, 448,
8, 104, 200, 296, 392, 40, 136, 232, 328, 424, 72, 168, 264, 360, 456,
16, 112, 208, 304, 400, 48, 144, 240, 336, 432, 80, 176, 272, 368, 464,
24, 120, 216, 312, 408, 56, 152, 248, 344, 440, 88, 184, 280, 376, 472,
4, 100, 196, 292, 388, 36, 132, 228, 324, 420, 68, 164, 260, 356, 452,
12, 108, 204, 300, 396, 44, 140, 236, 332, 428, 76, 172, 268, 364, 460,
20, 116, 212, 308, 404, 52, 148, 244, 340, 436, 84, 180, 276, 372, 468,
28, 124, 220, 316, 412, 60, 156, 252, 348, 444, 92, 188, 284, 380, 476,
1, 97, 193, 289, 385, 33, 129, 225, 321, 417, 65, 161, 257, 353, 449,
9, 105, 201, 297, 393, 41, 137, 233, 329, 425, 73, 169, 265, 361, 457,
17, 113, 209, 305, 401, 49, 145, 241, 337, 433, 81, 177, 273, 369, 465,
25, 121, 217, 313, 409, 57, 153, 249, 345, 441, 89, 185, 281, 377, 473,
5, 101, 197, 293, 389, 37, 133, 229, 325, 421, 69, 165, 261, 357, 453,
13, 109, 205, 301, 397, 45, 141, 237, 333, 429, 77, 173, 269, 365, 461,
21, 117, 213, 309, 405, 53, 149, 245, 341, 437, 85, 181, 277, 373, 469,
29, 125, 221, 317, 413, 61, 157, 253, 349, 445, 93, 189, 285, 381, 477,
2, 98, 194, 290, 386, 34, 130, 226, 322, 418, 66, 162, 258, 354, 450,
10, 106, 202, 298, 394, 42, 138, 234, 330, 426, 74, 170, 266, 362, 458,
18, 114, 210, 306, 402, 50, 146, 242, 338, 434, 82, 178, 274, 370, 466,
26, 122, 218, 314, 410, 58, 154, 250, 346, 442, 90, 186, 282, 378, 474,
6, 102, 198, 294, 390, 38, 134, 230, 326, 422, 70, 166, 262, 358, 454,
14, 110, 206, 302, 398, 46, 142, 238, 334, 430, 78, 174, 270, 366, 462,
22, 118, 214, 310, 406, 54, 150, 246, 342, 438, 86, 182, 278, 374, 470,
30, 126, 222, 318, 414, 62, 158, 254, 350, 446, 94, 190, 286, 382, 478,
3, 99, 195, 291, 387, 35, 131, 227, 323, 419, 67, 163, 259, 355, 451,
11, 107, 203, 299, 395, 43, 139, 235, 331, 427, 75, 171, 267, 363, 459,
19, 115, 211, 307, 403, 51, 147, 243, 339, 435, 83, 179, 275, 371, 467,
27, 123, 219, 315, 411, 59, 155, 251, 347, 443, 91, 187, 283, 379, 475,
7, 103, 199, 295, 391, 39, 135, 231, 327, 423, 71, 167, 263, 359, 455,
15, 111, 207, 303, 399, 47, 143, 239, 335, 431, 79, 175, 271, 367, 463,
23, 119, 215, 311, 407, 55, 151, 247, 343, 439, 87, 183, 279, 375, 471,
31, 127, 223, 319, 415, 63, 159, 255, 351, 447, 95, 191, 287, 383, 479,
];

#[rustfmt::skip]
pub static FFT_BITREV240: [i16; 240] = [
0, 48, 96, 144, 192, 16, 64, 112, 160, 208, 32, 80, 128, 176, 224,
4, 52, 100, 148, 196, 20, 68, 116, 164, 212, 36, 84, 132, 180, 228,
8, 56, 104, 152, 200, 24, 72, 120, 168, 216, 40, 88, 136, 184, 232,
12, 60, 108, 156, 204, 28, 76, 124, 172, 220, 44, 92, 140, 188, 236,
1, 49, 97, 145, 193, 17, 65, 113, 161, 209, 33, 81, 129, 177, 225,
5, 53, 101, 149, 197, 21, 69, 117, 165, 213, 37, 85, 133, 181, 229,
9, 57, 105, 153, 201, 25, 73, 121, 169, 217, 41, 89, 137, 185, 233,
13, 61, 109, 157, 205, 29, 77, 125, 173, 221, 45, 93, 141, 189, 237,
2, 50, 98, 146, 194, 18, 66, 114, 162, 210, 34, 82, 130, 178, 226,
6, 54, 102, 150, 198, 22, 70, 118, 166, 214, 38, 86, 134, 182, 230,
10, 58, 106, 154, 202, 26, 74, 122, 170, 218, 42, 90, 138, 186, 234,
14, 62, 110, 158, 206, 30, 78, 126, 174, 222, 46, 94, 142, 190, 238,
3, 51, 99, 147, 195, 19, 67, 115, 163, 211, 35, 83, 131, 179, 227,
7, 55, 103, 151, 199, 23, 71, 119, 167, 215, 39, 87, 135, 183, 231,
11, 59, 107, 155, 203, 27, 75, 123, 171, 219, 43, 91, 139, 187, 235,
15, 63, 111, 159, 207, 31, 79, 127, 175, 223, 47, 95, 143, 191, 239,
];

#[rustfmt::skip]
pub static FFT_BITREV120: [i16; 120] = [
0, 24, 48, 72, 96, 8, 32, 56, 80, 104, 16, 40, 64, 88, 112,
4, 28, 52, 76, 100, 12, 36, 60, 84, 108, 20, 44, 68, 92, 116,
1, 25, 49, 73, 97, 9, 33, 57, 81, 105, 17, 41, 65, 89, 113,
5, 29, 53, 77, 101, 13, 37, 61, 85, 109, 21, 45, 69, 93, 117,
2, 26, 50, 74, 98, 10, 34, 58, 82, 106, 18, 42, 66, 90, 114,
6, 30, 54, 78, 102, 14, 38, 62, 86, 110, 22, 46, 70, 94, 118,
3, 27, 51, 75, 99, 11, 35, 59, 83, 107, 19, 43, 67, 91, 115,
7, 31, 55, 79, 103, 15, 39, 63, 87, 111, 23, 47, 71, 95, 119,
];

#[rustfmt::skip]
pub static FFT_BITREV60: [i16; 60] = [
0, 12, 24, 36, 48, 4, 16, 28, 40, 52, 8, 20, 32, 44, 56,
1, 13, 25, 37, 49, 5, 17, 29, 41, 53, 9, 21, 33, 45, 57,
2, 14, 26, 38, 50, 6, 18, 30, 42, 54, 10, 22, 34, 46, 58,
3, 15, 27, 39, 51, 7, 19, 31, 43, 55, 11, 23, 35, 47, 59,
];

// =========================================================================
// Static FFT state definitions — 48kHz mode
// =========================================================================

/// 480-point FFT state (shift=0, base FFT, owns twiddles).
pub static FFT_STATE_48000_960_0: KissFftState = KissFftState {
    nfft: 480,
    scale: 17476,
    scale_shift: 8,
    shift: -1,
    factors: [5, 96, 3, 32, 4, 8, 2, 4, 4, 1, 0, 0, 0, 0, 0, 0],
    bitrev: &FFT_BITREV480,
    twiddles: &FFT_TWIDDLES_48000_960,
};

/// 240-point FFT state (shift=1, shares twiddles from base).
pub static FFT_STATE_48000_960_1: KissFftState = KissFftState {
    nfft: 240,
    scale: 17476,
    scale_shift: 7,
    shift: 1,
    factors: [5, 48, 3, 16, 4, 4, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    bitrev: &FFT_BITREV240,
    twiddles: &FFT_TWIDDLES_48000_960,
};

/// 120-point FFT state (shift=2).
pub static FFT_STATE_48000_960_2: KissFftState = KissFftState {
    nfft: 120,
    scale: 17476,
    scale_shift: 6,
    shift: 2,
    factors: [5, 24, 3, 8, 2, 4, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    bitrev: &FFT_BITREV120,
    twiddles: &FFT_TWIDDLES_48000_960,
};

/// 60-point FFT state (shift=3).
pub static FFT_STATE_48000_960_3: KissFftState = KissFftState {
    nfft: 60,
    scale: 17476,
    scale_shift: 5,
    shift: 3,
    factors: [5, 12, 3, 4, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    bitrev: &FFT_BITREV60,
    twiddles: &FFT_TWIDDLES_48000_960,
};

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_state_consistency() {
        // Verify nfft matches bitrev table length
        assert_eq!(FFT_STATE_48000_960_0.nfft as usize, FFT_BITREV480.len());
        assert_eq!(FFT_STATE_48000_960_1.nfft as usize, FFT_BITREV240.len());
        assert_eq!(FFT_STATE_48000_960_2.nfft as usize, FFT_BITREV120.len());
        assert_eq!(FFT_STATE_48000_960_3.nfft as usize, FFT_BITREV60.len());
    }

    #[test]
    fn test_twiddle_table_symmetry() {
        // First entry should be (1, 0) in Q15
        assert_eq!(FFT_TWIDDLES_48000_960[0].r, 32767);
        assert_eq!(FFT_TWIDDLES_48000_960[0].i, 0);
        // Entry at N/4 should be (0, -1) in Q15
        assert_eq!(FFT_TWIDDLES_48000_960[120].r, 0);
        assert_eq!(FFT_TWIDDLES_48000_960[120].i, -32767);
        // Entry at N/2 should be (-1, 0) in Q15
        assert_eq!(FFT_TWIDDLES_48000_960[240].r, -32767);
        assert_eq!(FFT_TWIDDLES_48000_960[240].i, 0);
    }

    #[test]
    fn test_bitrev_permutation_valid() {
        // All entries should be in [0, nfft) and be a permutation
        for (table, n) in [
            (&FFT_BITREV480[..], 480),
            (&FFT_BITREV240[..], 240),
            (&FFT_BITREV120[..], 120),
            (&FFT_BITREV60[..], 60),
        ] {
            let mut seen = vec![false; n];
            for &v in table {
                let v = v as usize;
                assert!(v < n, "bitrev entry {} out of range for N={}", v, n);
                assert!(!seen[v], "duplicate bitrev entry {} for N={}", v, n);
                seen[v] = true;
            }
        }
    }

    #[test]
    fn test_fft_dc_impulse() {
        // FFT of a DC signal: all samples = 1
        // Expected: bin[0] = N (scaled by 1/N → ~1), rest ≈ 0
        let st = &FFT_STATE_48000_960_3; // 60-point FFT
        let n = st.nfft as usize;
        let fin: Vec<KissFftCpx> = (0..n).map(|_| KissFftCpx { r: 1 << 15, i: 0 }).collect();
        let mut fout = vec![KissFftCpx::default(); n];
        opus_fft(st, &fin, &mut fout);

        // After 1/N scaling, DC bin should be approximately 1<<15
        // Non-DC bins should be near zero
        for k in 1..n {
            assert!(
                fout[k].r.abs() < 8 && fout[k].i.abs() < 8,
                "bin {} non-zero: ({}, {})",
                k,
                fout[k].r,
                fout[k].i
            );
        }
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        // Forward then inverse FFT should recover the original signal
        let st = &FFT_STATE_48000_960_3; // 60-point FFT
        let n = st.nfft as usize;

        // Create test signal: simple ramp
        let input: Vec<KissFftCpx> = (0..n)
            .map(|i| KissFftCpx {
                r: (i as i32) * 100,
                i: -(i as i32) * 50,
            })
            .collect();

        let mut freq = vec![KissFftCpx::default(); n];
        let mut output = vec![KissFftCpx::default(); n];

        opus_fft(st, &input, &mut freq);
        opus_ifft(st, &freq, &mut output);

        // IFFT doesn't scale by 1/N, so output = input * N (but FFT already
        // scaled by 1/N, so output ≈ input). Fixed-point rounding noise
        // grows with signal magnitude; use a proportional tolerance.
        for i in 0..n {
            let err_r = (output[i].r - input[i].r).abs();
            let err_i = (output[i].i - input[i].i).abs();
            let mag = input[i].r.abs() + input[i].i.abs();
            let tol = 64 + (mag >> 5);
            assert!(
                err_r <= tol && err_i <= tol,
                "roundtrip error at {}: expected ({}, {}), got ({}, {}), tol={}",
                i,
                input[i].r,
                input[i].i,
                output[i].r,
                output[i].i,
                tol
            );
        }
    }
}
