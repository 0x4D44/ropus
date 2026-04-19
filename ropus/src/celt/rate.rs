//! CELT bit allocation and rate control.
//!
//! Provides constants and functions for converting between pulse counts
//! and bit costs. Matches `rate.h` / `rate.c` in the C reference.

use super::ec_ctx::EcCoder;
use super::modes::{CELTMode, NB_EBANDS};

/// Bit resolution: all bit counts are in 1/8-bit units.
pub const BITRES: i32 = 3;

/// Offset for theta quantization (mono/general case).
pub const QTHETA_OFFSET: i32 = 4;

/// Offset for theta quantization (stereo N=2 case).
pub const QTHETA_OFFSET_TWOPHASE: i32 = 16;

/// log2(MAX_PSEUDO) rounded up -- number of binary search iterations.
const LOG_MAX_PSEUDO: i32 = 6;

/// Convert cache index to actual pulse count.
/// Matches C `get_pulses(i)`.
#[inline(always)]
pub fn get_pulses(i: i32) -> i32 {
    if i < 8 {
        i
    } else {
        (8 + (i & 7)) << ((i >> 3) - 1)
    }
}

/// Find the number of pulses that fit within a given bit budget.
/// Uses binary search over the pulse cache.
/// Matches C `bits2pulses()`.
pub fn bits2pulses(m: &CELTMode, band: i32, lm: i32, bits: i32) -> i32 {
    let lm1 = lm + 1;
    let idx = m.cache.index[(lm1 * m.nb_ebands + band) as usize] as usize;
    let cache = &m.cache.bits[idx..];
    let mut lo = 0i32;
    let mut hi = cache[0] as i32;
    let bits = bits - 1;
    for _ in 0..LOG_MAX_PSEUDO {
        let mid = (lo + hi + 1) >> 1;
        if cache[mid as usize] as i32 >= bits {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let lo_bits = if lo == 0 {
        -1
    } else {
        cache[lo as usize] as i32
    };
    if bits - lo_bits <= cache[hi as usize] as i32 - bits {
        lo
    } else {
        hi
    }
}

/// Look up the bit cost for a given pulse count.
/// Matches C `pulses2bits()`.
pub fn pulses2bits(m: &CELTMode, band: i32, lm: i32, q: i32) -> i32 {
    let lm1 = lm + 1;
    let idx = m.cache.index[(lm1 * m.nb_ebands + band) as usize] as usize;
    let cache = &m.cache.bits[idx..];
    if q == 0 {
        0
    } else {
        cache[q as usize] as i32 + 1
    }
}

// ===========================================================================
// Additional constants (from rate.h / rate.c)
// ===========================================================================

/// Maximum number of fine energy bits per band.
pub const MAX_FINE_BITS: i32 = 8;

/// Offset for allocating fine energy bits (relative to their "fair share").
pub const FINE_OFFSET: i32 = 21;

/// Number of binary-search steps when interpolating between allocation vectors.
const ALLOC_STEPS: i32 = 6;

/// Table of ceil(log2(i)) in Q(BITRES) for small i, used for intensity
/// and skip signalling bit reservations.
/// Matches the C static `LOG2_FRAC_TABLE[24]` in rate.c.
static LOG2_FRAC_TABLE: [u8; 24] = [
    0, 8, 13, 16, 19, 21, 23, 24, 26, 27, 28, 29, 30, 31, 32, 32, 33, 34, 34, 35, 36, 36, 37, 37,
];

// ===========================================================================
// Helpers
// ===========================================================================

/// Unsigned integer division (matches C `celt_udiv`).
#[inline(always)]
fn celt_udiv(n: u32, d: u32) -> u32 {
    debug_assert!(d > 0);
    n / d
}

// ===========================================================================
// interp_bits2pulses (from rate.c)
// ===========================================================================

/// Interpolate between two bit allocation vectors and convert to pulse counts.
///
/// Matches the C `interp_bits2pulses()` in `rate.c`. This is the core allocation
/// loop: it decides how many bands to code, which bands to skip, how many fine
/// energy bits each band gets, and how many PVQ bits remain.
///
/// `bits1`/`bits2` are the two allocation vectors to interpolate between.
/// `thresh` is the per-band minimum to allocate PVQ bits.
/// `cap` is the per-band maximum reliable bit capacity.
///
/// Returns the number of coded bands (`codedBands`).
fn interp_bits2pulses<EC: EcCoder>(
    m: &CELTMode,
    start: i32,
    end: i32,
    skip_start: i32,
    bits1: &[i32],
    bits2: &[i32],
    thresh: &[i32],
    cap: &[i32],
    total: i32,
    balance: &mut i32,
    skip_rsv: i32,
    intensity: &mut i32,
    intensity_rsv: i32,
    dual_stereo: &mut i32,
    dual_stereo_rsv: i32,
    bits: &mut [i32],
    ebits: &mut [i32],
    fine_priority: &mut [i32],
    c: i32,
    lm: i32,
    ec: &mut EC,
    encode: bool,
    prev: i32,
    signal_bandwidth: i32,
) -> i32 {
    let alloc_floor = c << BITRES;
    let stereo = if c > 1 { 1i32 } else { 0i32 };
    let log_m = lm << BITRES;

    let mut intensity_rsv = intensity_rsv;
    let mut dual_stereo_rsv = dual_stereo_rsv;
    let mut total = total;
    let mut psum: i32;
    let coded_bands: i32;

    // Binary search to find the interpolation factor
    let mut lo = 0i32;
    let mut hi = 1i32 << ALLOC_STEPS;
    for _ in 0..ALLOC_STEPS {
        let mid = (lo + hi) >> 1;
        psum = 0;
        let mut done = 0i32;
        let mut j = end;
        while {
            j -= 1;
            j >= start
        } {
            let tmp = bits1[j as usize] + (mid * bits2[j as usize] >> ALLOC_STEPS);
            if tmp >= thresh[j as usize] || done != 0 {
                done = 1;
                // Don't allocate more than we can actually use
                psum += tmp.min(cap[j as usize]);
            } else if tmp >= alloc_floor {
                psum += alloc_floor;
            }
        }
        if psum > total {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    // Compute the actual allocation with the chosen interpolation factor
    psum = 0;
    let mut done = 0i32;
    {
        let mut j = end;
        while {
            j -= 1;
            j >= start
        } {
            let mut tmp = bits1[j as usize] + (lo * bits2[j as usize] >> ALLOC_STEPS);
            if tmp < thresh[j as usize] && done == 0 {
                if tmp >= alloc_floor {
                    tmp = alloc_floor;
                } else {
                    tmp = 0;
                }
            } else {
                done = 1;
            }
            // Don't allocate more than we can actually use
            tmp = tmp.min(cap[j as usize]);
            bits[j as usize] = tmp;
            psum += tmp;
        }
    }

    // Decide which bands to skip, working backwards from the end.
    coded_bands = 'skip: {
        let mut cb = end;
        loop {
            let j = cb - 1;
            // Never skip the first band, nor a band that has been boosted by dynalloc.
            if j <= skip_start {
                // Give the bit we reserved to end skipping back.
                total += skip_rsv;
                break 'skip cb;
            }
            // Figure out how many left-over bits we would be adding to this band.
            let left = total - psum;
            let percoeff = celt_udiv(
                left as u32,
                (m.ebands[cb as usize] - m.ebands[start as usize]) as u32,
            ) as i32;
            let left2 = left - (m.ebands[cb as usize] - m.ebands[start as usize]) as i32 * percoeff;
            let rem = (left2 - (m.ebands[j as usize] - m.ebands[start as usize]) as i32).max(0);
            let band_width = (m.ebands[cb as usize] - m.ebands[j as usize]) as i32;
            let mut band_bits = bits[j as usize] + percoeff * band_width + rem;

            // Only code a skip decision if we're above the threshold for this band.
            if band_bits >= thresh[j as usize].max(alloc_floor + (1 << BITRES)) {
                if encode {
                    // Encoder skip decision with hysteresis
                    let depth_threshold = if cb > 17 {
                        if j < prev { 7 } else { 9 }
                    } else {
                        0
                    };
                    if cb <= start + 2
                        || (band_bits > ((depth_threshold * band_width << lm << BITRES) >> 4)
                            && j <= signal_bandwidth)
                    {
                        ec.ec_enc_bit_logp(true, 1);
                        break 'skip cb;
                    }
                    ec.ec_enc_bit_logp(false, 1);
                } else if ec.ec_dec_bit_logp(1) {
                    break 'skip cb;
                }
                // We used a bit to skip this band.
                psum += 1 << BITRES;
                band_bits -= 1 << BITRES;
            }
            // Reclaim the bits originally allocated to this band.
            psum -= bits[j as usize] + intensity_rsv;
            if intensity_rsv > 0 {
                intensity_rsv = LOG2_FRAC_TABLE[(j - start) as usize] as i32;
            }
            psum += intensity_rsv;
            if band_bits >= alloc_floor {
                // If we have enough for a fine energy bit per channel, use it.
                psum += alloc_floor;
                bits[j as usize] = alloc_floor;
            } else {
                // Otherwise this band gets nothing at all.
                bits[j as usize] = 0;
            }
            cb -= 1;
        }
    };

    debug_assert!(coded_bands > start);

    // Code the intensity and dual stereo parameters.
    if intensity_rsv > 0 {
        if encode {
            *intensity = (*intensity).min(coded_bands);
            ec.ec_enc_uint(
                (*intensity - start) as u32,
                (coded_bands + 1 - start) as u32,
            );
        } else {
            *intensity = start + ec.ec_dec_uint((coded_bands + 1 - start) as u32) as i32;
        }
    } else {
        *intensity = 0;
    }
    if *intensity <= start {
        total += dual_stereo_rsv;
        dual_stereo_rsv = 0;
    }
    if dual_stereo_rsv > 0 {
        if encode {
            ec.ec_enc_bit_logp(*dual_stereo != 0, 1);
        } else {
            *dual_stereo = ec.ec_dec_bit_logp(1) as i32;
        }
    } else {
        *dual_stereo = 0;
    }

    // Allocate the remaining bits
    let mut left = total - psum;
    let percoeff = celt_udiv(
        left as u32,
        (m.ebands[coded_bands as usize] - m.ebands[start as usize]) as u32,
    ) as i32;
    left -= (m.ebands[coded_bands as usize] - m.ebands[start as usize]) as i32 * percoeff;
    for j in start..coded_bands {
        bits[j as usize] += percoeff * (m.ebands[(j + 1) as usize] - m.ebands[j as usize]) as i32;
    }
    for j in start..coded_bands {
        let tmp = left.min((m.ebands[(j + 1) as usize] - m.ebands[j as usize]) as i32);
        bits[j as usize] += tmp;
        left -= tmp;
    }

    // Fine energy allocation
    let mut bal: i32 = 0;
    let mut j = start;
    while j < coded_bands {
        let ju = j as usize;
        let n0 = (m.ebands[(j + 1) as usize] - m.ebands[ju]) as i32;
        let n = n0 << lm;
        let bit: i32 = bits[ju] + bal;
        let excess: i32;

        debug_assert!(bits[ju] >= 0);

        if n > 1 {
            excess = (bit - cap[ju]).max(0);
            bits[ju] = bit - excess;

            // Compensate for the extra DoF in stereo
            let den = c * n
                + if c == 2 && n > 2 && *dual_stereo == 0 && j < *intensity {
                    1
                } else {
                    0
                };

            let nc_log_n = den * (m.log_n[ju] as i32 + log_m);

            // Offset for the number of fine bits by log2(N)/2 + FINE_OFFSET
            // compared to their "fair share" of total/N
            let mut offset = (nc_log_n >> 1) - den * FINE_OFFSET;

            // N=2 is the only point that doesn't match the curve
            if n == 2 {
                offset += den << BITRES >> 2;
            }

            // Changing the offset for allocating the second and third fine energy bit
            if bits[ju] + offset < den * 2 << BITRES {
                offset += nc_log_n >> 2;
            } else if bits[ju] + offset < den * 3 << BITRES {
                offset += nc_log_n >> 3;
            }

            // Divide with rounding
            ebits[ju] = (bits[ju] + offset + (den << (BITRES - 1))).max(0);
            ebits[ju] = (celt_udiv(ebits[ju] as u32, den as u32) as i32) >> BITRES;

            // Make sure not to bust
            if c * ebits[ju] > (bits[ju] >> BITRES) {
                ebits[ju] = bits[ju] >> stereo >> BITRES;
            }

            // More than that is useless because that's about as far as PVQ can go
            ebits[ju] = ebits[ju].min(MAX_FINE_BITS);

            // If we rounded down or capped this band, make it a candidate for
            // the final fine energy pass
            fine_priority[ju] = (ebits[ju] * (den << BITRES) >= bits[ju] + offset) as i32;

            // Remove the allocated fine bits; the rest are assigned to PVQ
            bits[ju] -= c * ebits[ju] << BITRES;
        } else {
            // For N=1, all bits go to fine energy except for a single sign bit
            excess = 0i32.max(bit - (c << BITRES));
            bits[ju] = bit - excess;
            ebits[ju] = 0;
            fine_priority[ju] = 1;
        }

        // Fine energy can't take advantage of the re-balancing in
        // quant_all_bands(). Instead, do the re-balancing here.
        if excess > 0 {
            let extra_fine = (excess >> (stereo + BITRES)).min(MAX_FINE_BITS - ebits[ju]);
            ebits[ju] += extra_fine;
            let extra_bits = extra_fine * c << BITRES;
            fine_priority[ju] = (extra_bits >= excess - bal) as i32;
            bal = excess - extra_bits;
        } else {
            bal = excess;
        }

        debug_assert!(bits[ju] >= 0);
        debug_assert!(ebits[ju] >= 0);
        j += 1;
    }

    // Save any remaining bits over the cap for the rebalancing in quant_all_bands().
    *balance = bal;

    // The skipped bands use all their bits for fine energy.
    while j < end {
        let ju = j as usize;
        ebits[ju] = bits[ju] >> stereo >> BITRES;
        debug_assert!(c * ebits[ju] << BITRES == bits[ju]);
        bits[ju] = 0;
        fine_priority[ju] = (ebits[ju] < 1) as i32;
        j += 1;
    }

    coded_bands
}

// ===========================================================================
// ICDF tables (from celt.h / rate.c)
// ===========================================================================

/// ICDF table for spread mode (4 symbols: NONE, LIGHT, NORMAL, AGGRESSIVE).
/// Encoded with ftb=5.
pub static SPREAD_ICDF: [u8; 4] = [25, 23, 2, 0];

/// ICDF table for alloc trim (11 symbols: trim values 0–10).
/// Encoded with ftb=7.
pub static TRIM_ICDF: [u8; 11] = [126, 124, 119, 109, 87, 41, 19, 9, 4, 2, 0];

/// ICDF table for prefilter tapset (3 symbols: tapset 0, 1, 2).
/// Encoded with ftb=4.
pub static TAPSET_ICDF: [u8; 3] = [2, 1, 0];

/// TF resolution selection table.
///
/// Indexed as `TF_SELECT_TABLE[LM][4*isTransient + 2*tf_select + tf_res]`.
/// Maps the raw tf_res flags and tf_select bit to actual TF resolution values.
pub static TF_SELECT_TABLE: [[i8; 8]; 4] = [
    // isTransient=0          isTransient=1
    [0, -1, 0, -1, 0, -1, 0, -1], // LM=0 (2.5 ms)
    [0, -1, 0, -2, 1, 0, 1, -1],  // LM=1 (5 ms)
    [0, -2, 0, -3, 2, 0, 1, -1],  // LM=2 (10 ms)
    [0, -2, 0, -3, 3, 0, 1, -1],  // LM=3 (20 ms)
];

// ===========================================================================
// clt_compute_allocation (from rate.c)
// ===========================================================================

/// Compute the pulse allocation — how many pulses (and fine energy bits)
/// each band gets.
///
/// Matches the C `clt_compute_allocation()` in `rate.c`. This is the top-level
/// allocation function called once per frame to decide the bit distribution
/// across all energy bands.
///
/// Returns the number of coded bands.
pub fn clt_compute_allocation<EC: EcCoder>(
    m: &CELTMode,
    start: i32,
    end: i32,
    offsets: &[i32],
    cap: &[i32],
    alloc_trim: i32,
    intensity: &mut i32,
    dual_stereo: &mut i32,
    total: i32,
    balance: &mut i32,
    pulses: &mut [i32],
    ebits: &mut [i32],
    fine_priority: &mut [i32],
    c: i32,
    lm: i32,
    ec: &mut EC,
    encode: bool,
    prev: i32,
    signal_bandwidth: i32,
) -> i32 {
    let total = total.max(0);
    let len = m.nb_ebands;
    let mut skip_start = start;

    // Reserve a bit to signal the end of manually skipped bands.
    let skip_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
    let mut total = total - skip_rsv;

    // Reserve bits for the intensity and dual stereo parameters.
    let mut intensity_rsv = 0i32;
    let mut dual_stereo_rsv = 0i32;
    if c == 2 {
        intensity_rsv = LOG2_FRAC_TABLE[(end - start) as usize] as i32;
        if intensity_rsv > total {
            intensity_rsv = 0;
        } else {
            total -= intensity_rsv;
            dual_stereo_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
            total -= dual_stereo_rsv;
        }
    }

    let mut bits1 = [0i32; NB_EBANDS];
    let mut bits2 = [0i32; NB_EBANDS];
    let mut thresh = [0i32; NB_EBANDS];
    let mut trim_offset = [0i32; NB_EBANDS];

    for j in start..end {
        let ju = j as usize;
        let band_width = (m.ebands[(j + 1) as usize] - m.ebands[j as usize]) as i32;
        // Below this threshold, we're sure not to allocate any PVQ bits
        thresh[ju] = (c << BITRES).max(3 * band_width << lm << BITRES >> 4);
        // Tilt of the allocation curve
        trim_offset[ju] =
            c * band_width * (alloc_trim - 5 - lm) * (end - j - 1) * (1 << (lm + BITRES)) >> 6;
        // Giving less resolution to single-coefficient bands
        if band_width << lm == 1 {
            trim_offset[ju] -= c << BITRES;
        }
    }

    // Binary search to find which two allocation vectors bracket the budget.
    // C uses do-while(lo <= hi), so the body always runs at least once.
    let mut lo = 1i32;
    let mut hi = m.nb_alloc_vectors - 1;
    loop {
        let mut done = 0i32;
        let mut psum = 0i32;
        let mid = (lo + hi) >> 1;
        let mut j = end;
        while {
            j -= 1;
            j >= start
        } {
            let ju = j as usize;
            let n = (m.ebands[(j + 1) as usize] - m.ebands[j as usize]) as i32;
            let mut bitsj = c * n * (m.alloc_vectors[(mid * len + j) as usize] as i32) << lm >> 2;
            if bitsj > 0 {
                bitsj = (bitsj + trim_offset[ju]).max(0);
            }
            bitsj += offsets[ju];
            if bitsj >= thresh[ju] || done != 0 {
                done = 1;
                // Don't allocate more than we can actually use
                psum += bitsj.min(cap[ju]);
            } else if bitsj >= c << BITRES {
                psum += c << BITRES;
            }
        }
        if psum > total {
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
        if lo > hi {
            break;
        }
    }
    hi = lo;
    lo -= 1;

    // Compute the two allocation vectors to interpolate between
    for j in start..end {
        let ju = j as usize;
        let n = (m.ebands[(j + 1) as usize] - m.ebands[j as usize]) as i32;
        let mut bits1j = (c * n * (m.alloc_vectors[(lo * len + j) as usize] as i32)) << lm >> 2;
        let mut bits2j = if hi >= m.nb_alloc_vectors {
            cap[ju]
        } else {
            (c * n * (m.alloc_vectors[(hi * len + j) as usize] as i32)) << lm >> 2
        };
        if bits1j > 0 {
            bits1j = (bits1j + trim_offset[ju]).max(0);
        }
        if bits2j > 0 {
            bits2j = (bits2j + trim_offset[ju]).max(0);
        }
        if lo > 0 {
            bits1j += offsets[ju];
        }
        bits2j += offsets[ju];
        if offsets[ju] > 0 {
            skip_start = j;
        }
        bits2j = (bits2j - bits1j).max(0);
        bits1[ju] = bits1j;
        bits2[ju] = bits2j;
    }

    let coded_bands = interp_bits2pulses(
        m,
        start,
        end,
        skip_start,
        &bits1,
        &bits2,
        &thresh,
        cap,
        total,
        balance,
        skip_rsv,
        intensity,
        intensity_rsv,
        dual_stereo,
        dual_stereo_rsv,
        pulses,
        ebits,
        fine_priority,
        c,
        lm,
        ec,
        encode,
        prev,
        signal_bandwidth,
    );

    coded_bands
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celt::modes::mode_create;
    use crate::celt::range_coder::{RangeDecoder, RangeEncoder};

    // ===================================================================
    // P0: get_pulses
    // ===================================================================

    #[test]
    fn test_get_pulses_linear_range() {
        for i in 0..8 {
            assert_eq!(get_pulses(i), i, "get_pulses({i}) should be {i}");
        }
    }

    #[test]
    fn test_get_pulses_exponential_range() {
        assert_eq!(get_pulses(8), 8); // (8+0)<<0
        assert_eq!(get_pulses(9), 9); // (8+1)<<0
        assert_eq!(get_pulses(16), 16); // (8+0)<<1
        assert_eq!(get_pulses(17), 18); // (8+1)<<1
        assert_eq!(get_pulses(24), 32); // (8+0)<<2
    }

    #[test]
    fn test_get_pulses_monotonic() {
        let mut prev = get_pulses(0);
        for i in 1..64 {
            let val = get_pulses(i);
            assert!(
                val >= prev,
                "get_pulses({i})={val} < get_pulses({})={prev}",
                i - 1
            );
            prev = val;
        }
    }

    // ===================================================================
    // P0: bits2pulses / pulses2bits
    // ===================================================================

    #[test]
    fn test_bits2pulses_zero_budget() {
        let mode = mode_create(48000, 960).expect("static mode");
        for band in [0, 5, 10, 15] {
            let q = bits2pulses(mode, band, 0, 0);
            assert_eq!(q, 0, "zero budget should give 0 pulses for band {band}");
        }
    }

    #[test]
    fn test_pulses2bits_zero() {
        let mode = mode_create(48000, 960).expect("static mode");
        for band in [0, 5, 10, 15] {
            assert_eq!(
                pulses2bits(mode, band, 0, 0),
                0,
                "0 pulses should cost 0 bits for band {band}"
            );
        }
    }

    #[test]
    fn test_bits2pulses_monotonic() {
        let mode = mode_create(48000, 960).expect("static mode");
        let band = 8;
        let lm = 0;
        let mut prev_q = 0;
        for bits in [0, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512] {
            let q = bits2pulses(mode, band, lm, bits);
            assert!(
                q >= prev_q,
                "band={band} bits={bits}: q={q} < prev_q={prev_q}"
            );
            prev_q = q;
        }
    }

    #[test]
    fn test_pulses2bits_monotonic() {
        let mode = mode_create(48000, 960).expect("static mode");
        let band = 5;
        let lm = 2;
        let idx = mode.cache.index[((lm + 1) * mode.nb_ebands + band) as usize] as usize;
        let max_q = mode.cache.bits[idx] as i32;
        let mut prev_bits = 0;
        for q in 0..=max_q.min(30) {
            let b = pulses2bits(mode, band, lm, q);
            assert!(
                b >= prev_bits,
                "band={band} q={q}: bits={b} < prev_bits={prev_bits}"
            );
            prev_bits = b;
        }
    }

    #[test]
    fn test_bits2pulses_roundtrip() {
        // bits2pulses finds the *nearest* pulse count, so when two pulse counts
        // have very similar bit costs, the roundtrip may return q-1.
        // The invariant is: bits2pulses(pulses2bits(q)) is either q or q-1.
        let mode = mode_create(48000, 960).expect("static mode");
        let band = 8;
        let lm = 0;
        let idx = mode.cache.index[((lm + 1) * mode.nb_ebands + band) as usize] as usize;
        let max_q = mode.cache.bits[idx] as i32;
        for q in 0..=max_q.min(20) {
            let bits = pulses2bits(mode, band, lm, q);
            let q_back = bits2pulses(mode, band, lm, bits);
            assert!(
                q_back == q || (q > 0 && q_back == q - 1),
                "roundtrip: q={q} -> bits={bits} -> q_back={q_back} (expected {q}{})",
                if q > 0 {
                    format!(" or {}", q - 1)
                } else {
                    String::new()
                }
            );
        }
    }

    #[test]
    fn test_bits2pulses_large_budget() {
        let mode = mode_create(48000, 960).expect("static mode");
        let band = 0;
        let lm = 0;
        let q = bits2pulses(mode, band, lm, 100000);
        let idx = mode.cache.index[((lm + 1) * mode.nb_ebands + band) as usize] as usize;
        let max_q = mode.cache.bits[idx] as i32;
        assert_eq!(q, max_q, "large budget should give max pulses from cache");
    }

    // ===================================================================
    // P1: clt_compute_allocation encode/decode symmetry
    // ===================================================================

    #[test]
    fn test_allocation_roundtrips_control_decisions() {
        let mode = mode_create(48000, 960).expect("static mode");
        let mut buf = [0u8; 256];
        let offsets = [0i32; NB_EBANDS];
        let cap = [4096i32; NB_EBANDS];

        let mut enc_pulses = [0i32; NB_EBANDS];
        let mut enc_ebits = [0i32; NB_EBANDS];
        let mut enc_priority = [0i32; NB_EBANDS];
        let mut enc_intensity = NB_EBANDS as i32;
        let mut enc_dual_stereo = 1;
        let mut enc_balance = 0;

        let enc_bands = {
            let mut enc = RangeEncoder::new(&mut buf);
            let coded = clt_compute_allocation(
                mode,
                0,
                NB_EBANDS as i32,
                &offsets,
                &cap,
                7,
                &mut enc_intensity,
                &mut enc_dual_stereo,
                700 << BITRES,
                &mut enc_balance,
                &mut enc_pulses,
                &mut enc_ebits,
                &mut enc_priority,
                2,
                0,
                &mut enc,
                true,
                NB_EBANDS as i32,
                NB_EBANDS as i32,
            );
            enc.done();
            coded
        };

        let mut dec_pulses = [0i32; NB_EBANDS];
        let mut dec_ebits = [0i32; NB_EBANDS];
        let mut dec_priority = [0i32; NB_EBANDS];
        let mut dec_intensity = 0;
        let mut dec_dual_stereo = 0;
        let mut dec_balance = 0;

        let mut dec = RangeDecoder::new(&buf);
        let dec_bands = clt_compute_allocation(
            mode,
            0,
            NB_EBANDS as i32,
            &offsets,
            &cap,
            7,
            &mut dec_intensity,
            &mut dec_dual_stereo,
            700 << BITRES,
            &mut dec_balance,
            &mut dec_pulses,
            &mut dec_ebits,
            &mut dec_priority,
            2,
            0,
            &mut dec,
            false,
            NB_EBANDS as i32,
            NB_EBANDS as i32,
        );

        assert_eq!(dec_bands, enc_bands, "coded bands mismatch");
        assert_eq!(dec_intensity, enc_intensity, "intensity mismatch");
        assert_eq!(dec_dual_stereo, enc_dual_stereo, "dual_stereo mismatch");
        assert_eq!(dec_balance, enc_balance, "balance mismatch");
        assert_eq!(dec_pulses, enc_pulses, "pulses mismatch");
        assert_eq!(dec_ebits, enc_ebits, "ebits mismatch");
        assert_eq!(dec_priority, enc_priority, "priority mismatch");
    }

    #[test]
    fn test_allocation_mono_no_stereo_params() {
        let mode = mode_create(48000, 960).expect("static mode");
        let mut buf = [0u8; 256];
        let offsets = [0i32; NB_EBANDS];
        let cap = [4096i32; NB_EBANDS];

        let mut pulses = [0i32; NB_EBANDS];
        let mut ebits = [0i32; NB_EBANDS];
        let mut priority = [0i32; NB_EBANDS];
        let mut intensity = 0;
        let mut dual_stereo = 0;
        let mut balance = 0;

        let mut enc = RangeEncoder::new(&mut buf);
        let _coded = clt_compute_allocation(
            mode,
            0,
            NB_EBANDS as i32,
            &offsets,
            &cap,
            5,
            &mut intensity,
            &mut dual_stereo,
            500 << BITRES,
            &mut balance,
            &mut pulses,
            &mut ebits,
            &mut priority,
            1,
            0,
            &mut enc,
            true,
            NB_EBANDS as i32,
            NB_EBANDS as i32,
        );

        assert_eq!(intensity, 0, "mono should have intensity=0");
        assert_eq!(dual_stereo, 0, "mono should have dual_stereo=0");
    }

    #[test]
    fn test_allocation_total_bits_bounded() {
        let mode = mode_create(48000, 960).expect("static mode");
        let mut buf = [0u8; 256];
        let offsets = [0i32; NB_EBANDS];
        let cap = [4096i32; NB_EBANDS];

        for total_bits in [100 << BITRES, 300 << BITRES, 700 << BITRES] {
            let mut pulses = [0i32; NB_EBANDS];
            let mut ebits = [0i32; NB_EBANDS];
            let mut priority = [0i32; NB_EBANDS];
            let mut intensity = NB_EBANDS as i32;
            let mut dual_stereo = 1;
            let mut balance = 0;

            let mut enc = RangeEncoder::new(&mut buf);
            let _coded = clt_compute_allocation(
                mode,
                0,
                NB_EBANDS as i32,
                &offsets,
                &cap,
                5,
                &mut intensity,
                &mut dual_stereo,
                total_bits,
                &mut balance,
                &mut pulses,
                &mut ebits,
                &mut priority,
                2,
                0,
                &mut enc,
                true,
                NB_EBANDS as i32,
                NB_EBANDS as i32,
            );

            let allocated: i32 =
                pulses.iter().sum::<i32>() + ebits.iter().map(|&e| e * 2 << BITRES).sum::<i32>();
            assert!(
                allocated >= 0,
                "total={total_bits}: allocated bits should be non-negative"
            );
        }
    }

    #[test]
    fn test_bits2pulses_various_lm() {
        let mode = mode_create(48000, 960).expect("static mode");
        let band = 8;
        for lm in 0..=3 {
            let q = bits2pulses(mode, band, lm, 64);
            assert!(
                q >= 0,
                "lm={lm}: bits2pulses should return non-negative, got {q}"
            );
            let b = pulses2bits(mode, band, lm, q);
            assert!(
                b >= 0,
                "lm={lm}: pulses2bits should return non-negative, got {b}"
            );
        }
    }

    // ===================================================================
    // Pinning tests: assert exact output values to catch arithmetic
    // mutations during mutation testing.
    // ===================================================================

    // --- get_pulses ---

    #[test]
    fn test_pin_rate_get_pulses_exact() {
        // Pin the full mapping from cache index to pulse count (0..48).
        let vals: Vec<i32> = (0..48).map(get_pulses).collect();
        assert_eq!(
            vals,
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28,
                30, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144,
                160, 176, 192, 208, 224, 240,
            ]
        );
    }

    #[test]
    fn test_pin_rate_get_pulses_boundary() {
        // Pin specific boundary values between linear and exponential ranges.
        assert_eq!(get_pulses(7), 7); // last linear
        assert_eq!(get_pulses(8), 8); // first exponential: (8+0)<<0
        assert_eq!(get_pulses(15), 15); // (8+7)<<0
        assert_eq!(get_pulses(16), 16); // (8+0)<<1
        assert_eq!(get_pulses(23), 30); // (8+7)<<1
        assert_eq!(get_pulses(24), 32); // (8+0)<<2
        assert_eq!(get_pulses(32), 64); // (8+0)<<3
        assert_eq!(get_pulses(40), 128); // (8+0)<<4
        assert_eq!(get_pulses(47), 240); // (8+7)<<4
    }

    // --- bits2pulses ---

    #[test]
    fn test_pin_rate_bits2pulses_lm0() {
        // Pin bits2pulses for lm=0 across all tested bands and bit budgets.
        let mode = mode_create(48000, 960).expect("static mode");
        let budgets = [0, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];

        let b0: Vec<i32> = budgets
            .iter()
            .map(|&b| bits2pulses(mode, 0, 0, b))
            .collect();
        assert_eq!(b0, [0, 1, 40, 40, 40, 40, 40, 40, 40, 40]);

        let b10: Vec<i32> = budgets
            .iter()
            .map(|&b| bits2pulses(mode, 10, 0, b))
            .collect();
        assert_eq!(b10, [0, 0, 1, 4, 31, 40, 40, 40, 40, 40]);

        let b15: Vec<i32> = budgets
            .iter()
            .map(|&b| bits2pulses(mode, 15, 0, b))
            .collect();
        assert_eq!(b15, [0, 0, 1, 1, 3, 10, 35, 35, 35, 35]);

        let b20: Vec<i32> = budgets
            .iter()
            .map(|&b| bits2pulses(mode, 20, 0, b))
            .collect();
        assert_eq!(b20, [0, 0, 0, 1, 2, 4, 9, 9, 9, 9]);
    }

    #[test]
    fn test_pin_rate_bits2pulses_lm2() {
        // Pin bits2pulses for lm=2 (10ms frames, most common).
        let mode = mode_create(48000, 960).expect("static mode");
        let budgets = [0, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];

        let b0: Vec<i32> = budgets
            .iter()
            .map(|&b| bits2pulses(mode, 0, 2, b))
            .collect();
        assert_eq!(b0, [0, 0, 1, 1, 4, 22, 40, 40, 40, 40]);

        let b10: Vec<i32> = budgets
            .iter()
            .map(|&b| bits2pulses(mode, 10, 2, b))
            .collect();
        assert_eq!(b10, [0, 0, 0, 1, 2, 7, 25, 25, 25, 25]);

        let b15: Vec<i32> = budgets
            .iter()
            .map(|&b| bits2pulses(mode, 15, 2, b))
            .collect();
        assert_eq!(b15, [0, 0, 0, 1, 2, 3, 9, 9, 9, 9]);

        let b20: Vec<i32> = budgets
            .iter()
            .map(|&b| bits2pulses(mode, 20, 2, b))
            .collect();
        assert_eq!(b20, [0, 0, 0, 1, 1, 2, 5, 5, 5, 5]);
    }

    #[test]
    fn test_pin_rate_bits2pulses_lm3() {
        // Pin bits2pulses for lm=3 (20ms frames).
        let mode = mode_create(48000, 960).expect("static mode");
        let budgets = [0, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];

        let b0: Vec<i32> = budgets
            .iter()
            .map(|&b| bits2pulses(mode, 0, 3, b))
            .collect();
        assert_eq!(b0, [0, 0, 0, 1, 2, 7, 25, 25, 25, 25]);

        let b10: Vec<i32> = budgets
            .iter()
            .map(|&b| bits2pulses(mode, 10, 3, b))
            .collect();
        assert_eq!(b10, [0, 0, 0, 1, 2, 4, 12, 12, 12, 12]);

        let b20: Vec<i32> = budgets
            .iter()
            .map(|&b| bits2pulses(mode, 20, 3, b))
            .collect();
        assert_eq!(b20, [0, 0, 0, 0, 1, 2, 4, 4, 4, 4]);
    }

    // --- pulses2bits ---

    #[test]
    fn test_pin_rate_pulses2bits_lm0() {
        // Pin pulses2bits for lm=0 across representative bands.
        let mode = mode_create(48000, 960).expect("static mode");

        // band=0, lm=0: single-coeff band, max_q=40
        let b0: Vec<i32> = (0..=15).map(|q| pulses2bits(mode, 0, 0, q)).collect();
        assert_eq!(b0, [0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]);

        // band=10, lm=0: wider band, max_q=40
        let b10: Vec<i32> = (0..=15).map(|q| pulses2bits(mode, 10, 0, q)).collect();
        assert_eq!(
            b10,
            [
                0, 16, 24, 29, 32, 35, 37, 39, 40, 42, 43, 44, 45, 46, 47, 48
            ]
        );

        // band=15, lm=0: wider still, max_q=35
        let b15: Vec<i32> = (0..=15).map(|q| pulses2bits(mode, 15, 0, q)).collect();
        assert_eq!(
            b15,
            [
                0, 29, 50, 66, 79, 90, 100, 108, 115, 121, 127, 133, 137, 142, 146, 150
            ]
        );

        // band=20, lm=0: widest band, max_q=9
        let b20: Vec<i32> = (0..=9).map(|q| pulses2bits(mode, 20, 0, q)).collect();
        assert_eq!(b20, [0, 44, 80, 111, 139, 164, 187, 208, 228, 247]);
    }

    #[test]
    fn test_pin_rate_pulses2bits_lm2() {
        // Pin pulses2bits for lm=2 across representative bands.
        let mode = mode_create(48000, 960).expect("static mode");

        let b0: Vec<i32> = (0..=15).map(|q| pulses2bits(mode, 0, 2, q)).collect();
        assert_eq!(
            b0,
            [
                0, 24, 40, 52, 61, 68, 74, 80, 84, 88, 92, 95, 98, 101, 103, 106
            ]
        );

        let b10: Vec<i32> = (0..=15).map(|q| pulses2bits(mode, 10, 2, q)).collect();
        assert_eq!(
            b10,
            [
                0, 32, 56, 76, 92, 106, 118, 129, 139, 147, 155, 162, 169, 175, 181, 186
            ]
        );

        let b15: Vec<i32> = (0..=9).map(|q| pulses2bits(mode, 15, 2, q)).collect();
        assert_eq!(b15, [0, 45, 82, 114, 143, 169, 193, 215, 236, 256]);

        let b20: Vec<i32> = (0..=5).map(|q| pulses2bits(mode, 20, 2, q)).collect();
        assert_eq!(b20, [0, 60, 112, 159, 203, 244]);
    }

    #[test]
    fn test_pin_rate_pulses2bits_lm3() {
        // Pin pulses2bits for lm=3.
        let mode = mode_create(48000, 960).expect("static mode");

        let b10: Vec<i32> = (0..=12).map(|q| pulses2bits(mode, 10, 3, q)).collect();
        assert_eq!(
            b10,
            [0, 40, 72, 100, 124, 145, 165, 183, 199, 215, 229, 242, 254]
        );

        let b15: Vec<i32> = (0..=6).map(|q| pulses2bits(mode, 15, 3, q)).collect();
        assert_eq!(b15, [0, 53, 98, 138, 175, 209, 241]);

        let b20: Vec<i32> = (0..=4).map(|q| pulses2bits(mode, 20, 3, q)).collect();
        assert_eq!(b20, [0, 68, 128, 183, 235]);
    }

    // --- clt_compute_allocation: mono ---

    /// Helper to run a mono allocation and return (coded_bands, balance, pulses, ebits, priority).
    fn run_mono_alloc(
        total: i32,
        alloc_trim: i32,
        lm: i32,
        end: i32,
        offsets: &[i32; NB_EBANDS],
    ) -> (
        i32,
        i32,
        [i32; NB_EBANDS],
        [i32; NB_EBANDS],
        [i32; NB_EBANDS],
    ) {
        let mode = mode_create(48000, 960).expect("static mode");
        let cap = [4096i32; NB_EBANDS];
        let mut buf = [0u8; 256];
        let mut pulses = [0i32; NB_EBANDS];
        let mut ebits = [0i32; NB_EBANDS];
        let mut priority = [0i32; NB_EBANDS];
        let mut intensity = 0;
        let mut dual_stereo = 0;
        let mut balance = 0;

        let mut enc = RangeEncoder::new(&mut buf);
        let coded = clt_compute_allocation(
            mode,
            0,
            end,
            offsets,
            &cap,
            alloc_trim,
            &mut intensity,
            &mut dual_stereo,
            total,
            &mut balance,
            &mut pulses,
            &mut ebits,
            &mut priority,
            1,
            lm,
            &mut enc,
            true,
            end,
            end,
        );
        (coded, balance, pulses, ebits, priority)
    }

    /// Helper to run a stereo allocation.
    fn run_stereo_alloc(
        total: i32,
        alloc_trim: i32,
        lm: i32,
    ) -> (
        i32,
        i32,
        i32,
        i32,
        [i32; NB_EBANDS],
        [i32; NB_EBANDS],
        [i32; NB_EBANDS],
    ) {
        let mode = mode_create(48000, 960).expect("static mode");
        let offsets = [0i32; NB_EBANDS];
        let cap = [4096i32; NB_EBANDS];
        let mut buf = [0u8; 256];
        let mut pulses = [0i32; NB_EBANDS];
        let mut ebits = [0i32; NB_EBANDS];
        let mut priority = [0i32; NB_EBANDS];
        let mut intensity = NB_EBANDS as i32;
        let mut dual_stereo = 1;
        let mut balance = 0;
        let end = NB_EBANDS as i32;

        let mut enc = RangeEncoder::new(&mut buf);
        let coded = clt_compute_allocation(
            mode,
            0,
            end,
            &offsets,
            &cap,
            alloc_trim,
            &mut intensity,
            &mut dual_stereo,
            total,
            &mut balance,
            &mut pulses,
            &mut ebits,
            &mut priority,
            2,
            lm,
            &mut enc,
            true,
            end,
            end,
        );
        (
            coded,
            intensity,
            dual_stereo,
            balance,
            pulses,
            ebits,
            priority,
        )
    }

    #[test]
    fn test_pin_rate_alloc_mono_low_budget() {
        // 200 eighth-bits, mono, lm=0, neutral trim
        let offsets = [0i32; NB_EBANDS];
        let (coded, balance, pulses, ebits, priority) =
            run_mono_alloc(200, 5, 0, NB_EBANDS as i32, &offsets);

        assert_eq!(coded, 13);
        assert_eq!(balance, 0);
        assert_eq!(
            pulses,
            [
                8, 8, 8, 8, 8, 8, 8, 8, 25, 21, 17, 13, 20, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        );
        assert_eq!(
            ebits,
            [
                1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
            ]
        );
        assert_eq!(
            priority,
            [
                0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1
            ]
        );
    }

    #[test]
    fn test_pin_rate_alloc_mono_medium_budget() {
        // 1000 eighth-bits (~125 bytes), mono, lm=0, neutral trim
        let offsets = [0i32; NB_EBANDS];
        let (coded, balance, pulses, ebits, priority) =
            run_mono_alloc(1000, 5, 0, NB_EBANDS as i32, &offsets);

        assert_eq!(coded, 20);
        assert_eq!(balance, 0);
        assert_eq!(
            pulses,
            [
                8, 8, 8, 8, 8, 8, 8, 8, 41, 37, 34, 31, 65, 58, 52, 73, 64, 77, 99, 81, 0
            ]
        );
        assert_eq!(
            ebits,
            [
                2, 3, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0
            ]
        );
        assert_eq!(
            priority,
            [
                0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1
            ]
        );
    }

    #[test]
    fn test_pin_rate_alloc_mono_high_budget() {
        // 4000 eighth-bits (~500 bytes), mono, lm=0 -- all 21 bands coded
        let offsets = [0i32; NB_EBANDS];
        let (coded, balance, pulses, ebits, priority) =
            run_mono_alloc(4000, 5, 0, NB_EBANDS as i32, &offsets);

        assert_eq!(coded, 21);
        assert_eq!(balance, 0);
        assert_eq!(
            pulses,
            [
                8, 8, 8, 8, 8, 8, 8, 8, 67, 64, 62, 67, 162, 157, 152, 236, 229, 306, 460, 616, 614
            ]
        );
        assert_eq!(
            ebits,
            [
                4, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3
            ]
        );
        assert_eq!(
            priority,
            [
                0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0
            ]
        );
    }

    #[test]
    fn test_pin_rate_alloc_mono_2000_all_bands() {
        // 2000 eighth-bits, mono, lm=0 -- all 21 bands coded
        let offsets = [0i32; NB_EBANDS];
        let (coded, balance, pulses, ebits, _) =
            run_mono_alloc(2000, 5, 0, NB_EBANDS as i32, &offsets);

        assert_eq!(coded, 21);
        assert_eq!(balance, 0);
        assert_eq!(
            pulses,
            [
                8, 8, 8, 8, 8, 8, 8, 8, 45, 50, 47, 44, 97, 89, 84, 124, 116, 149, 212, 260, 195
            ]
        );
        assert_eq!(
            ebits,
            [
                3, 4, 3, 3, 3, 3, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2
            ]
        );
    }

    // --- clt_compute_allocation: stereo ---

    #[test]
    fn test_pin_rate_alloc_stereo_low_budget() {
        // Stereo, 500 eighth-bits, lm=0
        let (coded, intensity, dual_stereo, balance, pulses, ebits, priority) =
            run_stereo_alloc(500, 5, 0);

        assert_eq!(coded, 14);
        assert_eq!(intensity, 14);
        assert_eq!(dual_stereo, 1);
        assert_eq!(balance, 0);
        assert_eq!(
            pulses,
            [
                16, 16, 16, 16, 16, 16, 16, 16, 50, 44, 38, 31, 47, 34, 0, 0, 0, 0, 0, 0, 0
            ]
        );
        assert_eq!(
            ebits,
            [
                1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0
            ]
        );
        assert_eq!(
            priority,
            [
                0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1
            ]
        );
    }

    #[test]
    fn test_pin_rate_alloc_stereo_high_budget() {
        // Stereo, 4000 eighth-bits, lm=0
        let (coded, intensity, dual_stereo, balance, pulses, ebits, _) =
            run_stereo_alloc(4000, 5, 0);

        assert_eq!(coded, 21);
        assert_eq!(intensity, 21);
        assert_eq!(dual_stereo, 1);
        assert_eq!(balance, 0);
        assert_eq!(
            pulses,
            [
                16, 16, 16, 16, 16, 16, 16, 16, 99, 98, 92, 86, 194, 180, 169, 252, 235, 298, 416,
                510, 375
            ]
        );
        assert_eq!(
            ebits,
            [
                3, 3, 4, 3, 3, 2, 3, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2
            ]
        );
    }

    // --- clt_compute_allocation: alloc_trim sweep ---

    #[test]
    fn test_pin_rate_alloc_trim_extremes() {
        // Pin trim=0 (bass-heavy) and trim=10 (treble-heavy) at 1000 eighth-bits, mono
        let offsets = [0i32; NB_EBANDS];

        let (coded0, _, pulses0, ebits0, _) =
            run_mono_alloc(1000, 0, 0, NB_EBANDS as i32, &offsets);
        assert_eq!(coded0, 20);
        assert_eq!(
            pulses0,
            [
                8, 8, 8, 8, 8, 8, 8, 8, 32, 30, 28, 26, 58, 53, 50, 73, 68, 88, 122, 124, 0
            ]
        );
        assert_eq!(
            ebits0,
            [
                1, 1, 2, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1
            ]
        );

        let (coded10, _, pulses10, ebits10, _) =
            run_mono_alloc(1000, 10, 0, NB_EBANDS as i32, &offsets);
        assert_eq!(coded10, 20);
        assert_eq!(
            pulses10,
            [
                8, 8, 8, 8, 8, 8, 8, 8, 42, 35, 39, 35, 70, 60, 52, 69, 57, 62, 69, 58, 0
            ]
        );
        assert_eq!(
            ebits10,
            [
                3, 4, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0
            ]
        );
    }

    // --- clt_compute_allocation: LM sweep ---

    #[test]
    fn test_pin_rate_alloc_lm_sweep() {
        // Pin allocation across frame sizes: lm=0 (2.5ms), lm=1 (5ms), lm=2 (10ms), lm=3 (20ms)
        let offsets = [0i32; NB_EBANDS];

        let (coded0, _, pulses0, ebits0, _) =
            run_mono_alloc(2000, 5, 0, NB_EBANDS as i32, &offsets);
        assert_eq!(coded0, 21);
        assert_eq!(
            pulses0,
            [
                8, 8, 8, 8, 8, 8, 8, 8, 45, 50, 47, 44, 97, 89, 84, 124, 116, 149, 212, 260, 195
            ]
        );
        assert_eq!(
            ebits0,
            [
                3, 4, 3, 3, 3, 3, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2
            ]
        );

        let (coded1, _, pulses1, ebits1, _) =
            run_mono_alloc(2000, 5, 1, NB_EBANDS as i32, &offsets);
        assert_eq!(coded1, 20);
        assert_eq!(
            pulses1,
            [
                47, 51, 47, 44, 40, 37, 34, 39, 74, 68, 71, 65, 121, 108, 105, 136, 120, 144, 185,
                152, 0
            ]
        );
        assert_eq!(
            ebits1,
            [
                3, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1
            ]
        );

        let (coded2, _, pulses2, _, _) = run_mono_alloc(2000, 5, 2, NB_EBANDS as i32, &offsets);
        assert_eq!(coded2, 17);
        assert_eq!(
            pulses2,
            [
                97, 91, 85, 79, 70, 73, 68, 63, 117, 107, 105, 93, 163, 137, 123, 147, 110, 0, 0,
                0, 0
            ]
        );

        let (coded3, _, pulses3, _, _) = run_mono_alloc(2000, 5, 3, NB_EBANDS as i32, &offsets);
        assert_eq!(coded3, 15);
        assert_eq!(
            pulses3,
            [
                155, 141, 128, 113, 111, 101, 93, 84, 152, 134, 124, 105, 169, 117, 73, 0, 0, 0, 0,
                0, 0
            ]
        );
    }

    // --- clt_compute_allocation: narrowband ---

    #[test]
    fn test_pin_rate_alloc_narrowband() {
        // Narrowband: end=13, so only the lower 13 bands are considered.
        let offsets = [0i32; NB_EBANDS];

        let (coded, _, pulses, ebits, _) = run_mono_alloc(500, 5, 0, 13, &offsets);
        assert_eq!(coded, 13);
        assert_eq!(
            pulses,
            [
                8, 8, 8, 8, 8, 8, 8, 8, 39, 35, 40, 37, 69, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        );
        assert_eq!(
            ebits,
            [
                3, 2, 3, 2, 3, 1, 2, 2, 2, 2, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        );

        let (coded2, _, pulses2, ebits2, _) = run_mono_alloc(1000, 5, 0, 13, &offsets);
        assert_eq!(coded2, 13);
        assert_eq!(
            pulses2,
            [
                8, 8, 8, 8, 8, 8, 8, 8, 69, 66, 64, 69, 156, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        );
        assert_eq!(
            ebits2,
            [
                4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        );
    }

    // --- clt_compute_allocation: dynalloc offsets ---

    #[test]
    fn test_pin_rate_alloc_with_offsets() {
        // Dynalloc offsets boost specific bands.
        let mut offsets = [0i32; NB_EBANDS];
        offsets[3] = 16;
        offsets[5] = 32;
        offsets[8] = 64;

        let (coded, balance, pulses, ebits, priority) =
            run_mono_alloc(2000, 5, 0, NB_EBANDS as i32, &offsets);

        assert_eq!(coded, 21);
        assert_eq!(balance, 0);
        assert_eq!(
            pulses,
            [
                8, 8, 8, 8, 8, 8, 8, 8, 79, 46, 44, 41, 92, 85, 79, 118, 110, 140, 199, 250, 173
            ]
        );
        assert_eq!(
            ebits,
            [
                3, 4, 3, 5, 2, 7, 2, 2, 7, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
            ]
        );
        assert_eq!(
            priority,
            [
                0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1
            ]
        );
    }

    // --- clt_compute_allocation: minimum budget ---

    #[test]
    fn test_pin_rate_alloc_minimum_budget() {
        // Very small budgets -- mostly just 1 coded band.
        let offsets = [0i32; NB_EBANDS];

        let (coded0, _, pulses0, ebits0, _) = run_mono_alloc(0, 5, 0, NB_EBANDS as i32, &offsets);
        assert_eq!(coded0, 1);
        assert_eq!(pulses0, [0; NB_EBANDS]);
        assert_eq!(ebits0, [0; NB_EBANDS]);

        let (coded8, _, pulses8, _, _) = run_mono_alloc(8, 5, 0, NB_EBANDS as i32, &offsets);
        assert_eq!(coded8, 1);
        assert_eq!(
            pulses8,
            [
                8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        );

        let (coded16, _, pulses16, ebits16, _) =
            run_mono_alloc(16, 5, 0, NB_EBANDS as i32, &offsets);
        assert_eq!(coded16, 1);
        assert_eq!(
            pulses16,
            [
                8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        );
        assert_eq!(
            ebits16,
            [
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        );

        let (coded64, _, pulses64, ebits64, _) =
            run_mono_alloc(64, 5, 0, NB_EBANDS as i32, &offsets);
        assert_eq!(coded64, 1);
        assert_eq!(
            pulses64,
            [
                8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        );
        assert_eq!(
            ebits64,
            [
                2, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        );
    }

    // --- clt_compute_allocation: encode/decode pinning ---

    #[test]
    fn test_pin_rate_alloc_encode_decode_exact() {
        // Verify that encode and decode produce identical allocation vectors.
        // This pins both directions at 2000 eighth-bits, stereo.
        let mode = mode_create(48000, 960).expect("static mode");
        let offsets = [0i32; NB_EBANDS];
        let cap = [4096i32; NB_EBANDS];
        let end = NB_EBANDS as i32;
        let mut buf = [0u8; 256];

        // Encode
        let mut enc_pulses = [0i32; NB_EBANDS];
        let mut enc_ebits = [0i32; NB_EBANDS];
        let mut enc_priority = [0i32; NB_EBANDS];
        let mut enc_intensity = NB_EBANDS as i32;
        let mut enc_dual_stereo = 1;
        let mut enc_balance = 0;
        let enc_coded = {
            let mut enc = RangeEncoder::new(&mut buf);
            let c = clt_compute_allocation(
                mode,
                0,
                end,
                &offsets,
                &cap,
                5,
                &mut enc_intensity,
                &mut enc_dual_stereo,
                2000,
                &mut enc_balance,
                &mut enc_pulses,
                &mut enc_ebits,
                &mut enc_priority,
                2,
                0,
                &mut enc,
                true,
                end,
                end,
            );
            enc.done();
            c
        };

        // Pin encode outputs
        assert_eq!(enc_coded, 20);
        assert_eq!(enc_intensity, 20);
        assert_eq!(enc_dual_stereo, 1);

        // Decode
        let mut dec_pulses = [0i32; NB_EBANDS];
        let mut dec_ebits = [0i32; NB_EBANDS];
        let mut dec_priority = [0i32; NB_EBANDS];
        let mut dec_intensity = 0;
        let mut dec_dual_stereo = 0;
        let mut dec_balance = 0;
        let mut dec = RangeDecoder::new(&buf);
        let dec_coded = clt_compute_allocation(
            mode,
            0,
            end,
            &offsets,
            &cap,
            5,
            &mut dec_intensity,
            &mut dec_dual_stereo,
            2000,
            &mut dec_balance,
            &mut dec_pulses,
            &mut dec_ebits,
            &mut dec_priority,
            2,
            0,
            &mut dec,
            false,
            end,
            end,
        );

        // Both must match exactly
        assert_eq!(dec_coded, enc_coded);
        assert_eq!(dec_intensity, enc_intensity);
        assert_eq!(dec_dual_stereo, enc_dual_stereo);
        assert_eq!(dec_balance, enc_balance);
        assert_eq!(dec_pulses, enc_pulses);
        assert_eq!(dec_ebits, enc_ebits);
        assert_eq!(dec_priority, enc_priority);
    }

    // =======================================================================
    // Stage 4 branch coverage
    // =======================================================================
    mod branch_coverage_stage4 {
        use super::*;

        /// Run an encode + decode allocation pair; returns (enc_coded, dec_coded).
        fn roundtrip_alloc(
            total: i32,
            alloc_trim: i32,
            lm: i32,
            c: i32,
            start: i32,
            end: i32,
            signal_bandwidth: i32,
            prev: i32,
            offsets: &[i32; NB_EBANDS],
            caps: i32,
        ) -> (i32, i32) {
            let mode = mode_create(48000, 960).expect("static mode");
            let cap = [caps; NB_EBANDS];
            let mut buf = [0u8; 512];

            let mut enc_pulses = [0i32; NB_EBANDS];
            let mut enc_ebits = [0i32; NB_EBANDS];
            let mut enc_priority = [0i32; NB_EBANDS];
            let mut enc_intensity = if c == 2 { end } else { 0 };
            let mut enc_dual_stereo = if c == 2 { 1 } else { 0 };
            let mut enc_balance = 0;

            let enc_coded = {
                let mut enc = RangeEncoder::new(&mut buf);
                let c_ = clt_compute_allocation(
                    mode,
                    start,
                    end,
                    offsets,
                    &cap,
                    alloc_trim,
                    &mut enc_intensity,
                    &mut enc_dual_stereo,
                    total,
                    &mut enc_balance,
                    &mut enc_pulses,
                    &mut enc_ebits,
                    &mut enc_priority,
                    c,
                    lm,
                    &mut enc,
                    true,
                    prev,
                    signal_bandwidth,
                );
                enc.done();
                c_
            };

            let mut dec_pulses = [0i32; NB_EBANDS];
            let mut dec_ebits = [0i32; NB_EBANDS];
            let mut dec_priority = [0i32; NB_EBANDS];
            let mut dec_intensity = 0;
            let mut dec_dual_stereo = 0;
            let mut dec_balance = 0;

            let mut dec = RangeDecoder::new(&buf);
            let dec_coded = clt_compute_allocation(
                mode,
                start,
                end,
                offsets,
                &cap,
                alloc_trim,
                &mut dec_intensity,
                &mut dec_dual_stereo,
                total,
                &mut dec_balance,
                &mut dec_pulses,
                &mut dec_ebits,
                &mut dec_priority,
                c,
                lm,
                &mut dec,
                false,
                prev,
                signal_bandwidth,
            );
            (enc_coded, dec_coded)
        }

        // Sweep total-bit budgets across the full range to hit the
        // skip-decision encoder paths (lines 235, 237, 238, 242-244, 250).
        #[test]
        fn test_bc_alloc_total_bits_sweep_stereo() {
            let offsets = [0i32; NB_EBANDS];
            let end = NB_EBANDS as i32;
            for &total in &[0i32, 16, 64, 256, 1000, 2500, 8000, 20000] {
                for trim in &[0i32, 5, 10] {
                    for lm in &[0i32, 1, 2, 3] {
                        let (enc_c, dec_c) =
                            roundtrip_alloc(total, *trim, *lm, 2, 0, end, end, end, &offsets, 4096);
                        assert_eq!(enc_c, dec_c, "total={total} trim={trim} lm={lm}");
                    }
                }
            }
        }

        // Cover signal_bandwidth < j and prev < j hysteresis branches
        // (lines 237-244) by sweeping both.
        #[test]
        fn test_bc_alloc_signal_bandwidth_and_prev_sweep() {
            let offsets = [0i32; NB_EBANDS];
            let end = NB_EBANDS as i32;
            for sig_bw in &[0i32, 5, 10, 15, 20, end] {
                for prev in &[0i32, 5, 10, 17, end] {
                    let (enc_c, dec_c) =
                        roundtrip_alloc(2000, 5, 0, 2, 0, end, *sig_bw, *prev, &offsets, 4096);
                    assert_eq!(enc_c, dec_c, "sig_bw={sig_bw} prev={prev}");
                }
            }
        }

        // Low caps force band_bits to be saturated and change the
        // skip-decision threshold path.
        #[test]
        fn test_bc_alloc_small_caps() {
            let offsets = [0i32; NB_EBANDS];
            let end = NB_EBANDS as i32;
            for caps in &[8i32, 32, 128, 512, 4096] {
                let (enc_c, dec_c) =
                    roundtrip_alloc(1500, 5, 0, 2, 0, end, end, end, &offsets, *caps);
                assert_eq!(enc_c, dec_c, "caps={caps}");
            }
        }

        // Tiny total below intensity_rsv forces the "intensity_rsv > total"
        // branch (line 496 / pulses falsed at 500). Also target the narrow
        // window where intensity fits but dual_stereo_rsv does not (line 500
        // else: total < 1<<BITRES after intensity subtraction).
        #[test]
        fn test_bc_alloc_tiny_stereo_budget() {
            let offsets = [0i32; NB_EBANDS];
            let end = NB_EBANDS as i32;
            // Sweep around the LOG2_FRAC_TABLE[21]=36 intensity_rsv boundary.
            for total in 0..=60i32 {
                let (_e, _d) = roundtrip_alloc(total, 5, 0, 2, 0, end, end, end, &offsets, 4096);
            }
            // Also sweep narrower stereo windows: intensity_rsv scales with end-start.
            for end in 2..=10i32 {
                for total in 0..=30i32 {
                    let (_e, _d) =
                        roundtrip_alloc(total, 5, 0, 2, 0, end, end, end, &offsets, 4096);
                }
            }
        }

        // Narrow band end to force intensity_rsv coding with a small
        // coded_bands window (hits lines 279, 281-286).
        #[test]
        fn test_bc_alloc_narrow_stereo_ranges() {
            let offsets = [0i32; NB_EBANDS];
            for end in 3..=8i32 {
                let (enc_c, dec_c) =
                    roundtrip_alloc(600, 5, 0, 2, 0, end, end, end, &offsets, 4096);
                assert_eq!(enc_c, dec_c, "end={end}");
            }
        }

        // skip_start offset trick: setting offsets boosts skip_start to keep
        // coded_bands bumped up against the boost point.
        #[test]
        fn test_bc_alloc_boost_offsets_sweep() {
            let end = NB_EBANDS as i32;
            for boost_band in [3, 7, 10, 15].iter() {
                let mut offsets = [0i32; NB_EBANDS];
                offsets[*boost_band] = 80;
                let (enc_c, dec_c) =
                    roundtrip_alloc(1200, 5, 0, 2, 0, end, end, end, &offsets, 4096);
                assert_eq!(enc_c, dec_c, "boost_band={boost_band}");
            }
        }

        // bits2pulses roundtrip: exercise q=0 path.
        #[test]
        fn test_bc_bits2pulses_q0_roundtrip() {
            let mode = mode_create(48000, 960).expect("static mode");
            // For small budgets, q will be 0.
            for band in 0..mode.nb_ebands {
                for lm in 0..=3 {
                    let q0 = bits2pulses(mode, band, lm, 0);
                    let b = pulses2bits(mode, band, lm, q0);
                    let _ = bits2pulses(mode, band, lm, b);
                }
            }
        }

        // Mono allocation sweep at various trim/lm/end values.
        #[test]
        fn test_bc_alloc_mono_sweep() {
            let offsets = [0i32; NB_EBANDS];
            for &total in &[0i32, 64, 500, 1000, 3000, 10000] {
                for trim in [0i32, 3, 5, 7, 10].iter() {
                    for lm in [0i32, 1, 2, 3].iter() {
                        let (enc_c, dec_c) = roundtrip_alloc(
                            total,
                            *trim,
                            *lm,
                            1,
                            0,
                            NB_EBANDS as i32,
                            NB_EBANDS as i32,
                            NB_EBANDS as i32,
                            &offsets,
                            4096,
                        );
                        assert_eq!(enc_c, dec_c, "total={total} trim={trim} lm={lm}");
                    }
                }
            }
        }
    }
}
