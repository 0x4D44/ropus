//! Band energy quantization for the CELT codec.
//!
//! Matches `celt/quant_bands.c`, `celt/quant_bands.h`, `celt/laplace.c`,
//! `celt/laplace.h` from the xiph/opus C reference (fixed-point path only).
//!
//! Three-pass hierarchical energy quantization:
//! 1. **Coarse**: integer-resolution energy deltas via Laplace entropy coding
//! 2. **Fine**: sub-integer refinement bits (raw bits)
//! 3. **Finalise**: leftover bits for one more bit of precision per band

use super::math_ops::celt_log2_db;
use super::modes::CELTMode;
use super::range_coder::{RangeDecoder, RangeEncoder};
use crate::types::*;

// ============================================================================
// Constants
// ============================================================================

/// Maximum fine quantization bits per band (from rate.h: MAX_FINE_BITS).
pub const MAX_FINE_BITS: i32 = 8;

/// Mean energy per band in Q4 (signed char).
/// Subtracted before quantization to center residuals near zero for
/// better Laplace coding efficiency. Matches C `eMeans[25]` in quant_bands.c.
pub static EMEANS: [i8; 25] = [
    103, 100, 92, 85, 81, 77, 72, 70, 78, 75, 73, 71, 78, 74, 69, 72, 70, 74, 76, 71, 60, 60, 60,
    60, 60,
];

/// Prediction coefficients indexed by LM, Q15.
/// Longer frames → lower prediction (less inter-frame correlation).
static PRED_COEF: [i32; 4] = [29440, 26112, 21248, 16384];

/// Inter-band smoothing coefficients indexed by LM, Q15.
/// Controls how much of the quantized energy feeds back into `prev[c]`.
static BETA_COEF: [i32; 4] = [30147, 22282, 12124, 6554];

/// Inter-band smoothing for intra mode, Q15. ~0.15.
const BETA_INTRA: i32 = 4915;

/// Laplace probability model parameters.
/// Layout: `E_PROB_MODEL[LM][intra][2*min(i,20)..]`
/// Each pair: (P(0) in Q8, decay rate in Q8).
/// Converted to Laplace codec parameters: `fs = p0 << 7`, `decay = d << 6`.
#[rustfmt::skip]
static E_PROB_MODEL: [[[u8; 42]; 2]; 4] = [
    // 120 sample frames
    [
        // Inter
        [
             72, 127,  65, 129,  66, 128,  65, 128,  64, 128,  62, 128,  64, 128,
             64, 128,  92,  78,  92,  79,  92,  78,  90,  79, 116,  41, 115,  40,
            114,  40, 132,  26, 132,  26, 145,  17, 161,  12, 176,  10, 177,  11,
        ],
        // Intra
        [
             24, 179,  48, 138,  54, 135,  54, 132,  53, 134,  56, 133,  55, 132,
             55, 132,  61, 114,  70,  96,  74,  88,  75,  88,  87,  74,  89,  66,
             91,  67, 100,  59, 108,  50, 120,  40, 122,  37,  97,  43,  78,  50,
        ],
    ],
    // 240 sample frames
    [
        // Inter
        [
             83,  78,  84,  81,  88,  75,  86,  74,  87,  71,  90,  73,  93,  74,
             93,  74, 109,  40, 114,  36, 117,  34, 117,  34, 143,  17, 145,  18,
            146,  19, 162,  12, 165,  10, 178,   7, 189,   6, 190,   8, 177,   9,
        ],
        // Intra
        [
             23, 178,  54, 115,  63, 102,  66,  98,  69,  99,  74,  89,  71,  91,
             73,  91,  78,  89,  86,  80,  92,  66,  93,  64, 102,  59, 103,  60,
            104,  60, 117,  52, 123,  44, 138,  35, 133,  31,  97,  38,  77,  45,
        ],
    ],
    // 480 sample frames
    [
        // Inter
        [
             61,  90,  93,  60, 105,  42, 107,  41, 110,  45, 116,  38, 113,  38,
            112,  38, 124,  26, 132,  27, 136,  19, 140,  20, 155,  14, 159,  16,
            158,  18, 170,  13, 177,  10, 187,   8, 192,   6, 175,   9, 159,  10,
        ],
        // Intra
        [
             21, 178,  59, 110,  71,  86,  75,  85,  84,  83,  91,  66,  88,  73,
             87,  72,  92,  75,  98,  72, 105,  58, 107,  54, 115,  52, 114,  55,
            112,  56, 129,  51, 132,  40, 150,  33, 140,  29,  98,  35,  77,  42,
        ],
    ],
    // 960 sample frames
    [
        // Inter
        [
             42, 121,  96,  66, 108,  43, 111,  40, 117,  44, 123,  32, 120,  36,
            119,  33, 127,  33, 134,  34, 139,  21, 147,  23, 152,  20, 158,  25,
            154,  26, 166,  21, 173,  16, 184,  13, 184,  10, 150,  13, 139,  15,
        ],
        // Intra
        [
             22, 178,  63, 114,  74,  82,  84,  83,  92,  82, 103,  62,  96,  72,
             96,  67, 101,  73, 107,  72, 113,  55, 118,  52, 125,  52, 118,  52,
            117,  55, 135,  49, 137,  39, 157,  32, 145,  29,  97,  33,  77,  40,
        ],
    ],
];

/// ICDF for 3-symbol fallback coding of coarse energy (qi in {-1, 0, 1}).
/// Symbol mapping: qi=-1 → 1, qi=0 → 0, qi=1 → 2.
static SMALL_ENERGY_ICDF: [u8; 3] = [2, 1, 0];

// ============================================================================
// Laplace entropy coding (from celt/laplace.c)
// ============================================================================

/// Minimum probability per symbol in the Laplace distribution.
const LAPLACE_LOG_MINP: u32 = 0;
const LAPLACE_MINP: u32 = 1 << LAPLACE_LOG_MINP;

/// Minimum guaranteed representable energy deltas per direction.
const LAPLACE_NMIN: u32 = 16;

/// Compute probability of value ±1 from P(0) and decay rate.
/// When called, decay is positive and at most 11456.
#[inline]
fn ec_laplace_get_freq1(fs0: u32, decay: i32) -> u32 {
    let ft: u32 = 32768 - LAPLACE_MINP * (2 * LAPLACE_NMIN) - fs0;
    ((ft as i64 * (16384 - decay) as i64) >> 15) as u32
}

/// Encode a Laplace-distributed value into the range coder.
///
/// `value` may be clamped if the distribution tail is exhausted —
/// the caller must use the modified value for reconstruction.
fn ec_laplace_encode(enc: &mut RangeEncoder, value: &mut i32, mut fs: u32, decay: i32) {
    let mut fl: u32 = 0;
    let val = *value;

    if val != 0 {
        // s = -1 for negative, 0 for positive
        let s: i32 = if val < 0 { -1 } else { 0 };
        let abs_val = (val + s) ^ s;
        fl = fs;
        fs = ec_laplace_get_freq1(fs, decay);

        // Search the decaying part of the PDF
        let mut i = 1;
        while fs > 0 && i < abs_val {
            fs *= 2;
            fl += fs + 2 * LAPLACE_MINP;
            fs = ((fs as i64 * decay as i64) >> 15) as u32;
            i += 1;
        }

        // Everything beyond the geometric part has probability LAPLACE_MINP
        if fs == 0 {
            let ndi_max = (32768u32
                .wrapping_sub(fl)
                .wrapping_add(LAPLACE_MINP)
                .wrapping_sub(1)
                >> LAPLACE_LOG_MINP) as i32;
            let ndi_max = (ndi_max - s) >> 1;
            let di = imin(abs_val - i, ndi_max - 1);
            fl += (2 * di + 1 + s) as u32 * LAPLACE_MINP;
            fs = imin(LAPLACE_MINP as i32, 32768u32.wrapping_sub(fl) as i32) as u32;
            *value = (i + di + s) ^ s;
        } else {
            fs += LAPLACE_MINP;
            // For positive: fl += fs. For negative: fl += 0.
            fl += fs & !(s as u32);
        }
        debug_assert!(fl + fs <= 32768);
        debug_assert!(fs > 0);
    }
    enc.encode_bin(fl, fl + fs, 15);
}

/// Decode a Laplace-distributed value from the range coder.
fn ec_laplace_decode(dec: &mut RangeDecoder, mut fs: u32, decay: i32) -> i32 {
    let mut val: i32 = 0;
    let mut fl: u32;

    let fm = dec.decode_bin(15);
    fl = 0;

    if fm >= fs {
        val += 1;
        fl = fs;
        fs = ec_laplace_get_freq1(fs, decay) + LAPLACE_MINP;

        // Search the decaying part of the PDF
        while fs > LAPLACE_MINP && fm >= fl + 2 * fs {
            fs *= 2;
            fl += fs;
            fs = ((fs - 2 * LAPLACE_MINP) as i64 * decay as i64 >> 15) as u32;
            fs += LAPLACE_MINP;
            val += 1;
        }

        // Everything beyond the geometric part has probability LAPLACE_MINP
        if fs <= LAPLACE_MINP {
            let di = ((fm - fl) >> (LAPLACE_LOG_MINP + 1)) as i32;
            val += di;
            fl += 2 * di as u32 * LAPLACE_MINP;
        }

        if fm < fl + fs {
            val = -val;
        } else {
            fl += fs;
        }
    }

    debug_assert!(fl < 32768);
    debug_assert!(fs > 0);
    debug_assert!(fl <= fm);
    debug_assert!(fm < core::cmp::min(fl + fs, 32768));
    let fh_final = core::cmp::min(fl + fs, 32768);
    dec.update(fl, fh_final, 32768);
    val
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Compute distortion between current and previous band energies.
/// Returns min(200, sum_of_squared_diffs >> 14).
fn loss_distortion(
    e_bands: &[i32],
    old_e_bands: &[i32],
    start: i32,
    end: i32,
    len: i32,
    cc: i32,
) -> i32 {
    let mut dist: i32 = 0;
    let mut c = 0;
    loop {
        for i in start..end {
            let idx = (i + c * len) as usize;
            // Shift Q24 difference right by 17 → Q7, then square and accumulate
            let d = pshr32(sub32(e_bands[idx], old_e_bands[idx]), DB_SHIFT - 7);
            dist = mac16_16(dist, d, d);
        }
        c += 1;
        if c >= cc {
            break;
        }
    }
    min32(200, shr32(dist, 14))
}

/// Core coarse energy quantization (one pass — either intra or inter).
/// Returns badness metric (sum of |qi_original - qi_clipped|).
fn quant_coarse_energy_impl(
    m: &CELTMode,
    start: i32,
    end: i32,
    e_bands: &[i32],
    old_e_bands: &mut [i32],
    budget: i32,
    tell: i32,
    prob_model: &[u8],
    error: &mut [i32],
    enc: &mut RangeEncoder,
    cc: i32,
    lm: i32,
    intra: i32,
    max_decay: i32,
    lfe: i32,
) -> i32 {
    let nb_ebands = m.nb_ebands;
    let mut badness: i32 = 0;
    let mut prev: [i32; 2] = [0, 0];

    let coef: i32;
    let beta: i32;
    if intra != 0 {
        coef = 0;
        beta = BETA_INTRA;
    } else {
        beta = BETA_COEF[lm as usize];
        coef = PRED_COEF[lm as usize];
    }

    // Write intra flag if budget allows
    let mut tell = tell;
    if tell + 3 <= budget {
        enc.encode_bit_logp(intra != 0, 3);
    }

    // Encode at fixed coarse resolution
    for i in start..end {
        let mut c = 0;
        loop {
            let idx = (i + c * nb_ebands) as usize;

            let x = e_bands[idx];
            let old_e = max32(-qconst32(9.0, DB_SHIFT as u32), old_e_bands[idx]);

            // Prediction residual (Q24)
            let f = x - mult16_32_q15(coef, old_e) - prev[c as usize];

            // Round to nearest integer (critically important for bit-exactness)
            let mut qi = (f + qconst32(0.5, DB_SHIFT as u32)) >> DB_SHIFT;

            // Prevent energy from dropping too fast
            let decay_bound = max32(
                -qconst32(28.0, DB_SHIFT as u32),
                sub32(old_e_bands[idx], max_decay),
            );
            if qi < 0 && x < decay_bound {
                qi += shr32(sub32(decay_bound, x), DB_SHIFT);
                if qi > 0 {
                    qi = 0;
                }
            }

            let qi0 = qi;

            // Clip qi when running low on bits
            tell = enc.tell();
            let bits_left = budget - tell - 3 * cc * (end - i);
            if i != start && bits_left < 30 {
                if bits_left < 24 {
                    qi = imin(1, qi);
                }
                if bits_left < 16 {
                    qi = imax(-1, qi);
                }
            }
            if lfe != 0 && i >= 2 {
                qi = imin(qi, 0);
            }

            // Entropy code the quantized value using the best available method
            if budget - tell >= 15 {
                let pi = 2 * imin(i, 20) as usize;
                ec_laplace_encode(
                    enc,
                    &mut qi,
                    (prob_model[pi] as u32) << 7,
                    (prob_model[pi + 1] as i32) << 6,
                );
            } else if budget - tell >= 2 {
                qi = imax(-1, imin(qi, 1));
                // Map qi to symbol: qi=-1→1, qi=0→0, qi=1→2
                let s = (2 * qi ^ (if qi < 0 { -1 } else { 0 })) as u32;
                enc.encode_icdf(s, &SMALL_ENERGY_ICDF, 2);
            } else if budget - tell >= 1 {
                qi = imin(0, qi);
                enc.encode_bit_logp(-qi != 0, 1);
            } else {
                qi = -1;
            }

            error[idx] = f - shl32(qi, DB_SHIFT);
            badness += (qi0 - qi).abs();
            let q = shl32(extend32(qi), DB_SHIFT);

            let tmp = mult16_32_q15(coef, old_e) + prev[c as usize] + q;
            let tmp = max32(-qconst32(28.0, DB_SHIFT as u32), tmp);
            old_e_bands[idx] = tmp;
            prev[c as usize] = prev[c as usize] + q - mult16_32_q15(beta, q);

            c += 1;
            if c >= cc {
                break;
            }
        }
    }
    if lfe != 0 { 0 } else { badness }
}

// ============================================================================
// Public encoder API
// ============================================================================

/// Top-level coarse energy quantization.
///
/// Decides between intra/inter coding, optionally runs both and picks the
/// better one (two-pass mode). Updates `old_e_bands` and `error` in place.
pub fn quant_coarse_energy(
    m: &CELTMode,
    start: i32,
    end: i32,
    eff_end: i32,
    e_bands: &[i32],
    old_e_bands: &mut [i32],
    budget: u32,
    error: &mut [i32],
    enc: &mut RangeEncoder,
    cc: i32,
    lm: i32,
    nb_available_bytes: i32,
    force_intra: i32,
    delayed_intra: &mut i32,
    mut two_pass: i32,
    loss_rate: i32,
    lfe: i32,
) {
    let nb_ebands = m.nb_ebands;
    let total_bands = (cc * nb_ebands) as usize;

    let mut intra = if force_intra != 0
        || (two_pass == 0
            && *delayed_intra > 2 * cc * (end - start)
            && nb_available_bytes > (end - start) * cc)
    {
        1i32
    } else {
        0i32
    };

    let intra_bias =
        ((budget as i64 * *delayed_intra as i64 * loss_rate as i64) / (cc as i64 * 512)) as i32;
    let new_distortion = loss_distortion(e_bands, old_e_bands, start, eff_end, nb_ebands, cc);

    let tell = enc.tell();
    if tell + 3 > budget as i32 {
        two_pass = 0;
        intra = 0;
    }

    // Compute max allowed energy decay per frame
    let mut max_decay = qconst32(16.0, DB_SHIFT as u32);
    if end - start > 10 {
        // max_decay = min(16.0, 0.125 * nbAvailableBytes) in Q24
        max_decay = shl32(
            min32(shr32(max_decay, DB_SHIFT - 3), extend32(nb_available_bytes)),
            DB_SHIFT - 3,
        );
    }
    if lfe != 0 {
        max_decay = qconst32(3.0, DB_SHIFT as u32);
    }

    let enc_start_snapshot = enc.snapshot();

    // Temporary buffers for the intra pass
    let mut old_e_bands_intra = vec![0i32; total_bands];
    let mut error_intra = vec![0i32; total_bands];
    old_e_bands_intra[..total_bands].copy_from_slice(&old_e_bands[..total_bands]);

    let mut badness1: i32 = 0;

    if two_pass != 0 || intra != 0 {
        badness1 = quant_coarse_energy_impl(
            m,
            start,
            end,
            e_bands,
            &mut old_e_bands_intra,
            budget as i32,
            tell,
            &E_PROB_MODEL[lm as usize][1],
            &mut error_intra,
            enc,
            cc,
            lm,
            1,
            max_decay,
            lfe,
        );
    }

    if intra == 0 {
        let tell_intra = enc.tell_frac();
        let enc_intra_snapshot = enc.snapshot();

        let nstart_bytes = enc_start_snapshot.range_bytes() as usize;
        let nintra_bytes = enc.range_bytes() as usize;
        let intra_byte_count = nintra_bytes - nstart_bytes;

        // Save the intra-coded bytes from the buffer
        let mut intra_bits = vec![0u8; intra_byte_count.max(1)];
        if intra_byte_count > 0 {
            intra_bits[..intra_byte_count]
                .copy_from_slice(&enc.buffer()[nstart_bytes..nintra_bytes]);
        }

        // Restore encoder to pre-coarse state and run inter pass
        enc.restore(&enc_start_snapshot);

        let badness2 = quant_coarse_energy_impl(
            m,
            start,
            end,
            e_bands,
            old_e_bands,
            budget as i32,
            tell,
            &E_PROB_MODEL[lm as usize][intra as usize],
            error,
            enc,
            cc,
            lm,
            0,
            max_decay,
            lfe,
        );

        if two_pass != 0
            && (badness1 < badness2
                || (badness1 == badness2
                    && (enc.tell_frac() as i32) + intra_bias > tell_intra as i32))
        {
            // Intra was better — restore intra state and bytes
            enc.restore(&enc_intra_snapshot);
            if intra_byte_count > 0 {
                enc.buffer_mut()[nstart_bytes..nintra_bytes]
                    .copy_from_slice(&intra_bits[..intra_byte_count]);
            }
            old_e_bands[..total_bands].copy_from_slice(&old_e_bands_intra[..total_bands]);
            error[..total_bands].copy_from_slice(&error_intra[..total_bands]);
            intra = 1;
        }
    } else {
        // Intra-only path
        old_e_bands[..total_bands].copy_from_slice(&old_e_bands_intra[..total_bands]);
        error[..total_bands].copy_from_slice(&error_intra[..total_bands]);
    }

    // Update delayed intra metric
    if intra != 0 {
        *delayed_intra = new_distortion;
    } else {
        *delayed_intra = add32(
            mult16_32_q15(
                mult16_16_q15(PRED_COEF[lm as usize], PRED_COEF[lm as usize]),
                *delayed_intra,
            ),
            new_distortion,
        );
    }
}

/// Encode fine energy refinement bits.
///
/// For each band with `extra_quant[i] > 0` bits allocated, encodes a
/// sub-integer correction to the coarse energy. Updates `old_e_bands`
/// and `error` in place.
pub fn quant_fine_energy(
    m: &CELTMode,
    start: i32,
    end: i32,
    old_e_bands: &mut [i32],
    error: &mut [i32],
    prev_quant: Option<&[i32]>,
    extra_quant: &[i32],
    enc: &mut RangeEncoder,
    cc: i32,
) {
    let nb_ebands = m.nb_ebands;

    for i in start..end {
        let iu = i as usize;
        let extra: i32 = 1 << extra_quant[iu];
        if extra_quant[iu] <= 0 {
            continue;
        }
        if enc.tell() + cc * extra_quant[iu] > enc.storage() as i32 * 8 {
            continue;
        }
        let prev: i32 = match prev_quant {
            Some(pq) => pq[iu],
            None => 0,
        };

        let mut c = 0;
        loop {
            let idx = (i + c * nb_ebands) as usize;

            // Floor quantization (no rounding) via variable-direction shift
            let mut q2 = vshr32(
                add32(error[idx], shr32(qconst32(0.5, DB_SHIFT as u32), prev)),
                DB_SHIFT - extra_quant[iu] - prev,
            );
            if q2 > extra - 1 {
                q2 = extra - 1;
            }
            if q2 < 0 {
                q2 = 0;
            }
            enc.encode_bits(q2 as u32, extra_quant[iu] as u32);

            // Compute offset: center of quantization bin in Q24
            let mut offset = sub32(
                vshr32(2 * q2 + 1, extra_quant[iu] - DB_SHIFT + 1),
                qconst32(0.5, DB_SHIFT as u32),
            );
            offset = shr32(offset, prev);

            old_e_bands[idx] += offset;
            error[idx] -= offset;

            c += 1;
            if c >= cc {
                break;
            }
        }
    }
}

/// Use up remaining bits for final energy refinement.
///
/// Two priority passes (prio=0 then prio=1), spending one bit per band
/// per channel. Bands with `fine_quant[i] >= MAX_FINE_BITS` are skipped.
pub fn quant_energy_finalise(
    m: &CELTMode,
    start: i32,
    end: i32,
    mut old_e_bands: Option<&mut [i32]>,
    error: &mut [i32],
    fine_quant: &[i32],
    fine_priority: &[i32],
    mut bits_left: i32,
    enc: &mut RangeEncoder,
    cc: i32,
) {
    let nb_ebands = m.nb_ebands;

    // Two priority passes: prio=0 bands first, then prio=1
    for prio in 0..2 {
        let mut i = start;
        while i < end && bits_left >= cc {
            let iu = i as usize;
            if fine_quant[iu] >= MAX_FINE_BITS || fine_priority[iu] != prio {
                i += 1;
                continue;
            }
            let mut c = 0;
            loop {
                let idx = (i + c * nb_ebands) as usize;
                let q2 = if error[idx] < 0 { 0i32 } else { 1i32 };
                enc.encode_bits(q2 as u32, 1);

                // Offset: (q2 - 0.5) scaled by fine_quant[i]+1
                let offset = shr32(
                    shl32(q2, DB_SHIFT) - qconst32(0.5, DB_SHIFT as u32),
                    fine_quant[iu] + 1,
                );
                if let Some(ref mut oeb) = old_e_bands {
                    oeb[idx] += offset;
                }
                error[idx] -= offset;
                bits_left -= 1;

                c += 1;
                if c >= cc {
                    break;
                }
            }
            i += 1;
        }
    }
}

// ============================================================================
// Public decoder API
// ============================================================================

/// Decode coarse energy from the bitstream.
///
/// Mirrors `quant_coarse_energy_impl` on the encoder side.
/// `intra` flag is determined by the caller (read from bitstream outside this function).
pub fn unquant_coarse_energy(
    m: &CELTMode,
    start: i32,
    end: i32,
    old_e_bands: &mut [i32],
    intra: i32,
    dec: &mut RangeDecoder,
    cc: i32,
    lm: i32,
) {
    let nb_ebands = m.nb_ebands;
    let prob_model = &E_PROB_MODEL[lm as usize][if intra != 0 { 1 } else { 0 }];

    // Decoder uses i64 for prev to prevent accumulation drift
    let mut prev: [i64; 2] = [0, 0];

    let coef: i32;
    let beta: i32;
    if intra != 0 {
        coef = 0;
        beta = BETA_INTRA;
    } else {
        beta = BETA_COEF[lm as usize];
        coef = PRED_COEF[lm as usize];
    }

    let budget = dec.storage() as i32 * 8;

    // Decode at fixed coarse resolution
    for i in start..end {
        let mut c = 0;
        loop {
            debug_assert!(c < 2);
            let idx = (i + c * nb_ebands) as usize;
            let c_idx = c as usize;

            let tell = dec.tell();
            let qi: i32;
            if budget - tell >= 15 {
                let pi = 2 * imin(i, 20) as usize;
                qi = ec_laplace_decode(
                    dec,
                    (prob_model[pi] as u32) << 7,
                    (prob_model[pi + 1] as i32) << 6,
                );
            } else if budget - tell >= 2 {
                let raw = dec.decode_icdf(&SMALL_ENERGY_ICDF, 2);
                // Reverse mapping: symbol → qi
                qi = (raw >> 1) ^ -(raw & 1);
            } else if budget - tell >= 1 {
                qi = -(dec.decode_bit_logp(1) as i32);
            } else {
                qi = -1;
            }

            let q = shl32(extend32(qi), DB_SHIFT);

            old_e_bands[idx] = max32(-qconst32(9.0, DB_SHIFT as u32), old_e_bands[idx]);
            // Compute with i64 prev to avoid accumulation drift
            let tmp_64 = (mult16_32_q15(coef, old_e_bands[idx]) as i64) + prev[c_idx] + (q as i64);
            // Truncate to i32, then clamp to ±28.0 in Q24
            let tmp = min32(
                qconst32(28.0, DB_SHIFT as u32),
                max32(-qconst32(28.0, DB_SHIFT as u32), tmp_64 as i32),
            );
            old_e_bands[idx] = tmp;
            prev[c_idx] = prev[c_idx] + (q as i64) - (mult16_32_q15(beta, q) as i64);

            c += 1;
            if c >= cc {
                break;
            }
        }
    }
}

/// Decode fine energy refinement bits.
pub fn unquant_fine_energy(
    m: &CELTMode,
    start: i32,
    end: i32,
    old_e_bands: &mut [i32],
    prev_quant: Option<&[i32]>,
    extra_quant: &[i32],
    dec: &mut RangeDecoder,
    cc: i32,
) {
    let nb_ebands = m.nb_ebands;

    for i in start..end {
        let iu = i as usize;
        let extra = extra_quant[iu];
        if extra_quant[iu] <= 0 {
            continue;
        }
        if dec.tell() + cc * extra_quant[iu] > dec.storage() as i32 * 8 {
            continue;
        }
        let prev: i32 = match prev_quant {
            Some(pq) => pq[iu],
            None => 0,
        };

        let mut c = 0;
        loop {
            let idx = (i + c * nb_ebands) as usize;
            let q2 = dec.decode_bits(extra as u32) as i32;

            // Compute offset: center of quantization bin in Q24
            let mut offset = sub32(
                vshr32(2 * q2 + 1, extra - DB_SHIFT + 1),
                qconst32(0.5, DB_SHIFT as u32),
            );
            offset = shr32(offset, prev);

            old_e_bands[idx] += offset;

            c += 1;
            if c >= cc {
                break;
            }
        }
    }
}

/// Decode final energy refinement bits.
pub fn unquant_energy_finalise(
    m: &CELTMode,
    start: i32,
    end: i32,
    mut old_e_bands: Option<&mut [i32]>,
    fine_quant: &[i32],
    fine_priority: &[i32],
    mut bits_left: i32,
    dec: &mut RangeDecoder,
    cc: i32,
) {
    let nb_ebands = m.nb_ebands;

    for prio in 0..2 {
        let mut i = start;
        while i < end && bits_left >= cc {
            let iu = i as usize;
            if fine_quant[iu] >= MAX_FINE_BITS || fine_priority[iu] != prio {
                i += 1;
                continue;
            }
            let mut c = 0;
            loop {
                let idx = (i + c * nb_ebands) as usize;
                let q2 = dec.decode_bits(1) as i32;

                let offset = shr32(
                    shl32(q2, DB_SHIFT) - qconst32(0.5, DB_SHIFT as u32),
                    fine_quant[iu] + 1,
                );
                if let Some(ref mut oeb) = old_e_bands {
                    oeb[idx] += offset;
                }
                bits_left -= 1;

                c += 1;
                if c >= cc {
                    break;
                }
            }
            i += 1;
        }
    }
}

// ============================================================================
// Domain conversion
// ============================================================================

/// Convert per-band amplitude energies to mean-removed log2 domain.
///
/// For bands `[0, eff_end)`: computes `celt_log2_db(bandE) - eMeans[i] * 2^(DB_SHIFT-4) + 2.0`.
/// For bands `[eff_end, end)`: sets to `-14.0` (silence floor).
pub fn amp2log2(
    m: &CELTMode,
    eff_end: i32,
    end: i32,
    band_e: &[i32],
    band_log_e: &mut [i32],
    cc: i32,
) {
    let nb_ebands = m.nb_ebands;
    let mut c = 0;
    loop {
        for i in 0..eff_end {
            let idx = (i + c * nb_ebands) as usize;
            band_log_e[idx] = celt_log2_db(band_e[idx])
                - shl32(EMEANS[i as usize] as i32, DB_SHIFT - 4)
                // Compensate for bandE[] being Q12 but celt_log2() expecting Q14 input
                + qconst32(2.0, DB_SHIFT as u32);
        }
        for i in eff_end..end {
            let idx = (c * nb_ebands + i) as usize;
            band_log_e[idx] = -qconst32(14.0, DB_SHIFT as u32);
        }
        c += 1;
        if c >= cc {
            break;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celt::range_coder::{RangeDecoder, RangeEncoder};

    // Minimal CELTMode stub for testing
    fn test_mode() -> CELTMode {
        CELTMode {
            fs: 48000,
            overlap: 120,
            nb_ebands: 21,
            eff_ebands: 21,
            preemph: [0; 4],
            ebands: &[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100,
            ],
            max_lm: 3,
            nb_short_mdcts: 1,
            short_mdct_size: 960,
            nb_alloc_vectors: 0,
            alloc_vectors: &[],
            log_n: &[],
            window: &[],
            cache: crate::celt::modes::PulseCache {
                size: 0,
                index: &[],
                bits: &[],
                caps: &[],
            },
        }
    }

    #[test]
    fn laplace_encode_decode_roundtrip() {
        let mut buf = vec![0u8; 256];
        let values = [0, 1, -1, 5, -5, 10, -10, 0, 3, -3];
        let fs = 15000u32;
        let decay = 6000;

        let mut encoded_values = values;
        {
            let mut enc = RangeEncoder::new(&mut buf);
            for v in encoded_values.iter_mut() {
                ec_laplace_encode(&mut enc, v, fs, decay);
            }
            enc.done();
            assert!(!enc.error());
        }
        {
            let mut dec = RangeDecoder::new(&buf);
            for &expected in encoded_values.iter() {
                let got = ec_laplace_decode(&mut dec, fs, decay);
                assert_eq!(got, expected);
            }
        }
    }

    #[test]
    fn laplace_zero_value() {
        let mut buf = vec![0u8; 64];
        let fs = 20000u32;
        let decay = 8000;

        let mut val = 0;
        {
            let mut enc = RangeEncoder::new(&mut buf);
            ec_laplace_encode(&mut enc, &mut val, fs, decay);
            enc.done();
        }
        {
            let mut dec = RangeDecoder::new(&buf);
            assert_eq!(ec_laplace_decode(&mut dec, fs, decay), 0);
        }
    }

    #[test]
    fn coarse_energy_encode_decode_roundtrip() {
        let m = test_mode();
        let nb_ebands = m.nb_ebands as usize;
        let cc = 1;
        let lm = 0;
        let start = 0;
        let end = 21;
        let total = cc as usize * nb_ebands;
        let budget_bytes = 200;

        // Synthesize some band energies
        let mut e_bands = vec![0i32; total];
        for i in 0..end as usize {
            e_bands[i] =
                qconst32(3.0, DB_SHIFT as u32) + (i as i32) * qconst32(0.5, DB_SHIFT as u32);
        }

        let mut old_e_bands_enc = vec![0i32; total];
        let mut error = vec![0i32; total];
        let mut delayed_intra = 0i32;

        let mut buf = vec![0u8; budget_bytes];

        // Encode
        {
            let mut enc = RangeEncoder::new(&mut buf);
            quant_coarse_energy(
                &m,
                start,
                end,
                end,
                &e_bands,
                &mut old_e_bands_enc,
                (budget_bytes as u32) * 8,
                &mut error,
                &mut enc,
                cc,
                lm,
                budget_bytes as i32,
                0,
                &mut delayed_intra,
                1,
                0,
                0,
            );
            enc.done();
        }

        // Decode: read the intra flag, then decode coarse energy
        let mut old_e_bands_dec = vec![0i32; total];
        {
            let mut dec = RangeDecoder::new(&buf);
            // The first 3-bit symbol is the intra flag; we need to skip it.
            let intra = if dec.decode_bit_logp(3) { 1 } else { 0 };
            unquant_coarse_energy(
                &m,
                start,
                end,
                &mut old_e_bands_dec,
                intra,
                &mut dec,
                cc,
                lm,
            );
        }

        // After encoding and decoding, old_e_bands should match
        assert_eq!(old_e_bands_enc, old_e_bands_dec);
    }

    #[test]
    fn fine_energy_encode_decode_roundtrip() {
        let m = test_mode();
        let nb_ebands = m.nb_ebands as usize;
        let cc = 1;
        let start = 0;
        let end = 10;
        let total = cc as usize * nb_ebands;

        let mut old_e_bands_enc = vec![qconst32(1.0, DB_SHIFT as u32); total];
        let mut old_e_bands_dec = old_e_bands_enc.clone();
        let mut error = vec![qconst32(0.3, DB_SHIFT as u32); total];

        // 3 extra bits per band, no previous fine quant
        let extra_quant: Vec<i32> = (0..nb_ebands)
            .map(|i| if (i as i32) < end { 3 } else { 0 })
            .collect();

        let mut buf = vec![0u8; 256];
        {
            let mut enc = RangeEncoder::new(&mut buf);
            quant_fine_energy(
                &m,
                start,
                end,
                &mut old_e_bands_enc,
                &mut error,
                None,
                &extra_quant,
                &mut enc,
                cc,
            );
            enc.done();
        }
        {
            let mut dec = RangeDecoder::new(&buf);
            unquant_fine_energy(
                &m,
                start,
                end,
                &mut old_e_bands_dec,
                None,
                &extra_quant,
                &mut dec,
                cc,
            );
        }

        assert_eq!(old_e_bands_enc, old_e_bands_dec);
    }

    #[test]
    fn energy_finalise_encode_decode_roundtrip() {
        let m = test_mode();
        let nb_ebands = m.nb_ebands as usize;
        let cc = 1;
        let start = 0;
        let end = 5;
        let total = cc as usize * nb_ebands;

        let mut old_e_bands_enc = vec![qconst32(2.0, DB_SHIFT as u32); total];
        let mut old_e_bands_dec = old_e_bands_enc.clone();
        let mut error_enc = vec![qconst32(0.1, DB_SHIFT as u32); total];
        let _error_dec = error_enc.clone();

        let fine_quant: Vec<i32> = (0..nb_ebands)
            .map(|i| if (i as i32) < end { 3 } else { 0 })
            .collect();
        let fine_priority: Vec<i32> = (0..nb_ebands).map(|i| (i % 2) as i32).collect();
        let bits_left = end * cc; // enough for one bit per band per channel

        let mut buf = vec![0u8; 256];
        {
            let mut enc = RangeEncoder::new(&mut buf);
            quant_energy_finalise(
                &m,
                start,
                end,
                Some(&mut old_e_bands_enc),
                &mut error_enc,
                &fine_quant,
                &fine_priority,
                bits_left,
                &mut enc,
                cc,
            );
            enc.done();
        }
        {
            let mut dec = RangeDecoder::new(&buf);
            unquant_energy_finalise(
                &m,
                start,
                end,
                Some(&mut old_e_bands_dec),
                &fine_quant,
                &fine_priority,
                bits_left,
                &mut dec,
                cc,
            );
        }

        assert_eq!(old_e_bands_enc, old_e_bands_dec);
    }

    #[test]
    fn emeans_matches_c_reference() {
        // Verify Q4 to float conversion matches C reference float table
        let expected_float: [f32; 25] = [
            6.437500, 6.250000, 5.750000, 5.312500, 5.062500, 4.812500, 4.500000, 4.375000,
            4.875000, 4.687500, 4.562500, 4.437500, 4.875000, 4.625000, 4.312500, 4.500000,
            4.375000, 4.625000, 4.750000, 4.437500, 3.750000, 3.750000, 3.750000, 3.750000,
            3.750000,
        ];
        for i in 0..25 {
            let float_val = EMEANS[i] as f32 / 16.0;
            assert!(
                (float_val - expected_float[i]).abs() < 1e-6,
                "eMeans[{i}]: got {float_val}, expected {}",
                expected_float[i]
            );
        }
    }

    #[test]
    fn laplace_get_freq1_basic() {
        // With fs0=16384 (P(0)=0.5) and decay=8192 (0.5):
        // ft = 32768 - 1*32 - 16384 = 16352
        // freq1 = 16352 * (16384 - 8192) >> 15 = 16352 * 8192 / 32768 = 4088
        let freq1 = ec_laplace_get_freq1(16384, 8192);
        assert_eq!(freq1, 4088);
    }

    #[test]
    fn small_energy_icdf_symbol_mapping() {
        // Verify the ICDF symbol mapping for coarse energy fallback
        // qi=-1 → symbol 1, qi=0 → symbol 0, qi=1 → symbol 2
        for qi in [-1i32, 0, 1] {
            let s = 2 * qi ^ (if qi < 0 { -1 } else { 0 });
            let recovered = (s >> 1) ^ -(s & 1);
            assert_eq!(recovered, qi, "roundtrip failed for qi={qi}");
        }
    }

    #[test]
    fn amp2log2_silence_floor() {
        let m = test_mode();
        let nb_ebands = m.nb_ebands as usize;
        let cc = 1;
        let eff_end = 10;
        let end = 21;

        let band_e = vec![qconst32(1.0, 12); nb_ebands]; // Q12 energies
        let mut band_log_e = vec![0i32; nb_ebands];

        amp2log2(&m, eff_end, end, &band_e, &mut band_log_e, cc);

        // Bands [eff_end, end) should be at silence floor = -14.0 in Q24
        let silence = -qconst32(14.0, DB_SHIFT as u32);
        for i in eff_end..end {
            assert_eq!(
                band_log_e[i as usize], silence,
                "band {i} should be at silence floor"
            );
        }
    }

    #[test]
    fn encoder_snapshot_roundtrip() {
        let mut buf = vec![0u8; 256];
        let mut enc = RangeEncoder::new(&mut buf);

        // Encode some data
        enc.encode_bit_logp(true, 3);
        enc.encode_bit_logp(false, 3);

        // Save state
        let snap = enc.snapshot();
        let bytes_at_snap = enc.range_bytes();

        // Encode more data
        enc.encode_bit_logp(true, 1);
        enc.encode_bit_logp(true, 1);
        assert!(enc.range_bytes() >= bytes_at_snap);

        // Restore state
        enc.restore(&snap);
        assert_eq!(enc.range_bytes(), bytes_at_snap);

        // Encode different data from the same point
        enc.encode_bit_logp(false, 1);
        enc.done();
        assert!(!enc.error());
    }
}
