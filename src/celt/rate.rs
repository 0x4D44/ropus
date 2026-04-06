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
