//! CWRS — Combinatorial Waveform Representation Sequence (PVQ codebook enumeration).
//!
//! Implements the bijective mapping between pulse vectors (integer vectors where
//! `sum(|y[i]|) == K`) and codebook indices in `[0, V(N,K))`.
//!
//! This is the core entropy coding primitive for CELT band quantization:
//! - Encoder: `vq::alg_quant()` → `encode_pulses()` → `RangeEncoder::encode_uint()`
//! - Decoder: `RangeDecoder::decode_uint()` → `decode_pulses()` → `vq::alg_unquant()`
//!
//! Port of `celt/cwrs.c` (table-lookup path) from the xiph/opus C reference.

use crate::celt::range_coder::{RangeDecoder, RangeEncoder};
use crate::types::{ec_ilog, mac16_16};

// ============================================================================
// Precomputed U(N,K) table
// ============================================================================
//
// U(N,K) = number of N-dimensional signed pulse vectors with at most K-1
// pulses in the first N-1 dimensions. Related to V(N,K) by:
//   V(N,K) = U(N,K) + U(N,K+1)
//
// Rows are stored contiguously with overlapping offsets. The offset for row N
// is chosen so that CELT_PVQ_U_DATA[offset[N] + k] = U(N, k). The overlap
// is safe because out-of-range columns are never accessed via celt_pvq_u()
// (which uses min(N,K) as the row index to exploit symmetry U(N,K) = U(K,N)).

/// Row offsets into `CELT_PVQ_U_DATA` for direct table access.
const CELT_PVQ_U_ROW_OFFSETS: [usize; 15] = [
    0, 176, 351, 525, 698, 870, 1041, 1131, 1178, 1207, 1226, 1240, 1248, 1254, 1257,
];

/// Flat precomputed U(N,K) values for N=0..14.
#[rustfmt::skip]
static CELT_PVQ_U_DATA: [u32; 1272] = [
    // N=0, K=0..176 (177 entries): U(0,0)=1, U(0,K>0)=0
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // N=1, K=1..176 (176 entries): U(1,K)=1 for K>=1
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // N=2, K=2..176 (175 entries): U(2,K) = 2K-1
    3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41,
    43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79,
    81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113,
    115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137, 139, 141, 143,
    145, 147, 149, 151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 173,
    175, 177, 179, 181, 183, 185, 187, 189, 191, 193, 195, 197, 199, 201, 203,
    205, 207, 209, 211, 213, 215, 217, 219, 221, 223, 225, 227, 229, 231, 233,
    235, 237, 239, 241, 243, 245, 247, 249, 251, 253, 255, 257, 259, 261, 263,
    265, 267, 269, 271, 273, 275, 277, 279, 281, 283, 285, 287, 289, 291, 293,
    295, 297, 299, 301, 303, 305, 307, 309, 311, 313, 315, 317, 319, 321, 323,
    325, 327, 329, 331, 333, 335, 337, 339, 341, 343, 345, 347, 349, 351,
    // N=3, K=3..176 (174 entries)
    13, 25, 41, 61, 85, 113, 145, 181, 221, 265, 313, 365, 421, 481, 545, 613,
    685, 761, 841, 925, 1013, 1105, 1201, 1301, 1405, 1513, 1625, 1741, 1861,
    1985, 2113, 2245, 2381, 2521, 2665, 2813, 2965, 3121, 3281, 3445, 3613, 3785,
    3961, 4141, 4325, 4513, 4705, 4901, 5101, 5305, 5513, 5725, 5941, 6161, 6385,
    6613, 6845, 7081, 7321, 7565, 7813, 8065, 8321, 8581, 8845, 9113, 9385, 9661,
    9941, 10225, 10513, 10805, 11101, 11401, 11705, 12013, 12325, 12641, 12961,
    13285, 13613, 13945, 14281, 14621, 14965, 15313, 15665, 16021, 16381, 16745,
    17113, 17485, 17861, 18241, 18625, 19013, 19405, 19801, 20201, 20605, 21013,
    21425, 21841, 22261, 22685, 23113, 23545, 23981, 24421, 24865, 25313, 25765,
    26221, 26681, 27145, 27613, 28085, 28561, 29041, 29525, 30013, 30505, 31001,
    31501, 32005, 32513, 33025, 33541, 34061, 34585, 35113, 35645, 36181, 36721,
    37265, 37813, 38365, 38921, 39481, 40045, 40613, 41185, 41761, 42341, 42925,
    43513, 44105, 44701, 45301, 45905, 46513, 47125, 47741, 48361, 48985, 49613,
    50245, 50881, 51521, 52165, 52813, 53465, 54121, 54781, 55445, 56113, 56785,
    57461, 58141, 58825, 59513, 60205, 60901, 61601,
    // N=4, K=4..176 (173 entries)
    63, 129, 231, 377, 575, 833, 1159, 1561, 2047, 2625, 3303, 4089, 4991, 6017,
    7175, 8473, 9919, 11521, 13287, 15225, 17343, 19649, 22151, 24857, 27775,
    30913, 34279, 37881, 41727, 45825, 50183, 54809, 59711, 64897, 70375, 76153,
    82239, 88641, 95367, 102425, 109823, 117569, 125671, 134137, 142975, 152193,
    161799, 171801, 182207, 193025, 204263, 215929, 228031, 240577, 253575,
    267033, 280959, 295361, 310247, 325625, 341503, 357889, 374791, 392217,
    410175, 428673, 447719, 467321, 487487, 508225, 529543, 551449, 573951,
    597057, 620775, 645113, 670079, 695681, 721927, 748825, 776383, 804609,
    833511, 863097, 893375, 924353, 956039, 988441, 1021567, 1055425, 1090023,
    1125369, 1161471, 1198337, 1235975, 1274393, 1313599, 1353601, 1394407,
    1436025, 1478463, 1521729, 1565831, 1610777, 1656575, 1703233, 1750759,
    1799161, 1848447, 1898625, 1949703, 2001689, 2054591, 2108417, 2163175,
    2218873, 2275519, 2333121, 2391687, 2451225, 2511743, 2573249, 2635751,
    2699257, 2763775, 2829313, 2895879, 2963481, 3032127, 3101825, 3172583,
    3244409, 3317311, 3391297, 3466375, 3542553, 3619839, 3698241, 3777767,
    3858425, 3940223, 4023169, 4107271, 4192537, 4278975, 4366593, 4455399,
    4545401, 4636607, 4729025, 4822663, 4917529, 5013631, 5110977, 5209575,
    5309433, 5410559, 5512961, 5616647, 5721625, 5827903, 5935489, 6044391,
    6154617, 6266175, 6379073, 6493319, 6608921, 6725887, 6844225, 6963943,
    7085049, 7207551,
    // N=5, K=5..176 (172 entries)
    321, 681, 1289, 2241, 3649, 5641, 8361, 11969, 16641, 22569, 29961, 39041,
    50049, 63241, 78889, 97281, 118721, 143529, 172041, 204609, 241601, 283401,
    330409, 383041, 441729, 506921, 579081, 658689, 746241, 842249, 947241,
    1061761, 1186369, 1321641, 1468169, 1626561, 1797441, 1981449, 2179241,
    2391489, 2618881, 2862121, 3121929, 3399041, 3694209, 4008201, 4341801,
    4695809, 5071041, 5468329, 5888521, 6332481, 6801089, 7295241, 7815849,
    8363841, 8940161, 9545769, 10181641, 10848769, 11548161, 12280841, 13047849,
    13850241, 14689089, 15565481, 16480521, 17435329, 18431041, 19468809,
    20549801, 21675201, 22846209, 24064041, 25329929, 26645121, 28010881,
    29428489, 30899241, 32424449, 34005441, 35643561, 37340169, 39096641,
    40914369, 42794761, 44739241, 46749249, 48826241, 50971689, 53187081,
    55473921, 57833729, 60268041, 62778409, 65366401, 68033601, 70781609,
    73612041, 76526529, 79526721, 82614281, 85790889, 89058241, 92418049,
    95872041, 99421961, 103069569, 106816641, 110664969, 114616361, 118672641,
    122835649, 127107241, 131489289, 135983681, 140592321, 145317129, 150160041,
    155123009, 160208001, 165417001, 170752009, 176215041, 181808129, 187533321,
    193392681, 199388289, 205522241, 211796649, 218213641, 224775361, 231483969,
    238341641, 245350569, 252512961, 259831041, 267307049, 274943241, 282741889,
    290705281, 298835721, 307135529, 315607041, 324252609, 333074601, 342075401,
    351257409, 360623041, 370174729, 379914921, 389846081, 399970689, 410291241,
    420810249, 431530241, 442453761, 453583369, 464921641, 476471169, 488234561,
    500214441, 512413449, 524834241, 537479489, 550351881, 563454121, 576788929,
    590359041, 604167209, 618216201, 632508801,
    // N=6, K=6..96 (91 entries)
    1683, 3653, 7183, 13073, 22363, 36365, 56695, 85305, 124515, 177045, 246047,
    335137, 448427, 590557, 766727, 982729, 1244979, 1560549, 1937199, 2383409,
    2908411, 3522221, 4235671, 5060441, 6009091, 7095093, 8332863, 9737793,
    11326283, 13115773, 15124775, 17372905, 19880915, 22670725, 25765455,
    29189457, 32968347, 37129037, 41699767, 46710137, 52191139, 58175189,
    64696159, 71789409, 79491819, 87841821, 96879431, 106646281, 117185651,
    128542501, 140763503, 153897073, 167993403, 183104493, 199284183, 216588185,
    235074115, 254801525, 275831935, 298228865, 322057867, 347386557, 374284647,
    402823977, 433078547, 465124549, 499040399, 534906769, 572806619, 612825229,
    655050231, 699571641, 746481891, 795875861, 847850911, 902506913, 959946283,
    1020274013, 1083597703, 1150027593, 1219676595, 1292660325, 1369097135,
    1449108145, 1532817275, 1620351277, 1711839767, 1807415257, 1907213187,
    2011371957, 2120032959,
    // N=7, K=7..54 (48 entries)
    8989, 19825, 40081, 75517, 134245, 227305, 369305, 579125, 880685, 1303777,
    1884961, 2668525, 3707509, 5064793, 6814249, 9041957, 11847485, 15345233,
    19665841, 24957661, 31388293, 39146185, 48442297, 59511829, 72616013,
    88043969, 106114625, 127178701, 151620757, 179861305, 212358985, 249612805,
    292164445, 340600625, 395555537, 457713341, 527810725, 606639529, 695049433,
    793950709, 904317037, 1027188385, 1163673953, 1314955181, 1482288821,
    1667010073, 1870535785, 2094367717,
    // N=8, K=8..37 (30 entries)
    48639, 108545, 224143, 433905, 795455, 1392065, 2340495, 3800305, 5984767,
    9173505, 13726991, 20103025, 28875327, 40754369, 56610575, 77500017,
    104692735, 139703809, 184327311, 240673265, 311207743, 398796225, 506750351,
    638878193, 799538175, 993696769, 1226990095, 1505789553, 1837271615,
    2229491905,
    // N=9, K=9..28 (20 entries)
    265729, 598417, 1256465, 2485825, 4673345, 8405905, 14546705, 24331777,
    39490049, 62390545, 96220561, 145198913, 214828609, 312193553, 446304145,
    628496897, 872893441, 1196924561, 1621925137, 2173806145,
    // N=10, K=10..24 (15 entries)
    1462563, 3317445, 7059735, 14218905, 27298155, 50250765, 89129247, 152951073,
    254831667, 413442773, 654862247, 1014889769, 1541911931, 2300409629,
    3375210671,
    // N=11, K=11..19 (9 entries)
    8097453, 18474633, 39753273, 81270333, 158819253, 298199265, 540279585,
    948062325, 1616336765,
    // N=12, K=12..18 (7 entries)
    45046719, 103274625, 224298231, 464387817, 921406335, 1759885185,
    3248227095,
    // N=13, K=13..16 (4 entries)
    251595969, 579168825, 1267854873, 2653649025,
    // N=14, K=14 (1 entry)
    1409933619,
];

// ============================================================================
// Table lookup helpers
// ============================================================================

/// Direct row/column access into the precomputed table.
/// Returns U(row, col). Caller must ensure row ∈ [0,14] and col is in range.
#[inline(always)]
fn pvq_u_row(row: usize, col: usize) -> u32 {
    let idx = CELT_PVQ_U_ROW_OFFSETS[row] + col;
    if idx < CELT_PVQ_U_DATA.len() { CELT_PVQ_U_DATA[idx] } else { 0 }
}

/// U(N,K) with symmetry: uses min(N,K) as row to stay within the 15-row table.
/// This matches the C macro `CELT_PVQ_U(_n, _k)`.
#[inline(always)]
pub fn celt_pvq_u(n: i32, k: i32) -> u32 {
    pvq_u_row(n.min(k) as usize, n.max(k) as usize)
}

/// V(N,K) = U(N,K) + U(N,K+1) — total PVQ codeword count.
/// This matches the C macro `CELT_PVQ_V(_n, _k)`.
#[inline(always)]
pub fn celt_pvq_v(n: i32, k: i32) -> u32 {
    celt_pvq_u(n, k).wrapping_add(celt_pvq_u(n, k + 1))
}

// ============================================================================
// Encoding: pulse vector → codebook index
// ============================================================================

/// Compute the codebook index of pulse vector `y[0..n]`.
///
/// Maps a signed integer vector with `sum(|y[i]|) == K` to an index in `[0, V(N,K))`.
/// Processes dimensions from last to first, accumulating the combinatorial index.
pub(crate) fn icwrs(n: i32, y: &[i32]) -> u32 {
    debug_assert!(n >= 2);
    debug_assert!(y.len() >= n as usize);

    let mut j = (n - 1) as usize;
    // Start with sign bit of last element
    let mut i: u32 = if y[j] < 0 { 1 } else { 0 };
    let mut k: i32 = y[j].abs();

    // Process remaining dimensions from second-to-last down to first
    loop {
        if j == 0 {
            break;
        }
        j -= 1;
        // Skip count: all vectors with fewer pulses in remaining dimensions
        i = i.wrapping_add(celt_pvq_u((n - j as i32) as i32, k));
        k += y[j].abs();
        // Sign offset for negative elements
        if y[j] < 0 {
            i = i.wrapping_add(celt_pvq_u((n - j as i32) as i32, k + 1));
        }
    }

    i
}

/// Encode a pulse vector into the bitstream.
///
/// Computes the combinatorial index of `y` within the PVQ codebook of size
/// `V(n, k)`, then writes it as a uniform integer via the range encoder.
///
/// # Panics (debug)
/// - `k` must be > 0
/// - `y.len()` must be >= `n`
/// - `sum(|y[i]|)` must equal `k`
pub fn encode_pulses(y: &[i32], n: i32, k: i32, enc: &mut RangeEncoder) {
    debug_assert!(k > 0);
    let index = icwrs(n, y);
    let total = celt_pvq_v(n, k);
    enc.encode_uint(index, total);
}

// ============================================================================
// Decoding: codebook index → pulse vector
// ============================================================================

/// Decode codebook index `i` into pulse vector `y[0..n]`, returning `sum(y[j]^2)`.
///
/// Reconstructs the signed pulse vector from its combinatorial index using the
/// precomputed U(N,K) table. The squared norm is accumulated as a side-effect
/// using 16×16 multiply-accumulate (matching the C reference's MAC16_16).
pub(crate) fn cwrsi(mut n: i32, mut k: i32, mut i: u32, y: &mut [i32]) -> i32 {
    debug_assert!(k > 0);
    debug_assert!(n > 1);

    let mut yy: i32 = 0;
    let mut j: usize = 0; // output index into y[]

    while n > 2 {
        if k >= n {
            // --- Lots of pulses: search within row n ---
            let mut p = pvq_u_row(n as usize, (k + 1) as usize);
            // Extract sign: s = 0 (positive/zero) or -1 (negative)
            let s: i32 = if i >= p { -1 } else { 0 };
            // Branchless conditional subtract: i -= p if negative
            i = i.wrapping_sub(p & (s as u32));
            let k0 = k;
            let q = pvq_u_row(n as usize, n as usize);
            if q > i {
                // Index falls below U(n,n): search rows via symmetry U(k,n)
                k = n;
                loop {
                    k -= 1;
                    p = pvq_u_row(k as usize, n as usize);
                    if p <= i {
                        break;
                    }
                }
            } else {
                // Normal search: scan columns of row n downward
                p = pvq_u_row(n as usize, k as usize);
                while p > i {
                    k -= 1;
                    p = pvq_u_row(n as usize, k as usize);
                }
            }
            i = i.wrapping_sub(p);
            // Reconstruct signed value: (yj + s) ^ s is branchless conditional negate
            let val = (k0 - k + s) ^ s;
            y[j] = val;
            j += 1;
            yy = mac16_16(yy, val, val);
        } else {
            // --- Lots of dimensions: check for zero pulses ---
            let p_zero = pvq_u_row(k as usize, n as usize);
            let q = pvq_u_row((k + 1) as usize, n as usize);
            if p_zero <= i && i < q {
                // Zero pulses in this dimension
                i = i.wrapping_sub(p_zero);
                y[j] = 0;
                j += 1;
            } else {
                // Non-zero: extract sign using q threshold
                let s: i32 = if i >= q { -1 } else { 0 };
                i = i.wrapping_sub(q & (s as u32));
                let k0 = k;
                // Search for pulse count via row descent
                let mut p: u32;
                loop {
                    k -= 1;
                    p = pvq_u_row(k as usize, n as usize);
                    if p <= i {
                        break;
                    }
                }
                i = i.wrapping_sub(p);
                let val = (k0 - k + s) ^ s;
                y[j] = val;
                j += 1;
                yy = mac16_16(yy, val, val);
            }
        }
        n -= 1;
    }

    // --- N == 2: closed-form decode using U(2,K) = 2K-1 ---
    {
        let p = (2u32).wrapping_mul(k as u32).wrapping_add(1); // U(2, k+1)
        let s: i32 = if i >= p { -1 } else { 0 };
        i = i.wrapping_sub(p & (s as u32));
        let k0 = k;
        // Closed-form inverse of U(2, k) = 2k-1: k = (i+1) >> 1
        k = ((i.wrapping_add(1)) >> 1) as i32;
        if k > 0 {
            i = i.wrapping_sub((2u32).wrapping_mul(k as u32).wrapping_sub(1)); // subtract U(2, k)
        }
        let val = (k0 - k + s) ^ s;
        y[j] = val;
        j += 1;
        yy = mac16_16(yy, val, val);
    }

    // --- N == 1: sign bit only ---
    {
        // At this point i is 0 (positive) or 1 (negative)
        let s = -(i as i32);
        let val = (k + s) ^ s;
        y[j] = val;
        yy = mac16_16(yy, val, val);
    }

    yy
}

/// Decode a pulse vector from the bitstream, returning `sum(y[j]^2)`.
///
/// Reads a uniform integer from the range decoder, then reconstructs the
/// pulse vector via combinatorial decoding.
///
/// # Panics (debug)
/// - `k` must be > 0
/// - `n` must be > 1
/// - `y.len()` must be >= `n`
pub fn decode_pulses(y: &mut [i32], n: i32, k: i32, dec: &mut RangeDecoder) -> i32 {
    debug_assert!(k > 0);
    let total = celt_pvq_v(n, k);
    let index = dec.decode_uint(total);
    cwrsi(n, k, index, y)
}

// ============================================================================
// Bit allocation helpers (corresponds to CUSTOM_MODES in C reference)
// ============================================================================

/// Conservative (always rounds up) binary logarithm with `frac` fractional bits.
///
/// Maximum overestimation is 0.06254 bits at `frac=4` (tested for all u32 inputs).
/// Uses an iterative squaring method that normalizes to 16-bit precision then
/// extracts fractional bits one at a time from MSB to LSB.
pub fn log2_frac(mut val: u32, mut frac: i32) -> i32 {
    let mut l = ec_ilog(val);
    // Check if val is an exact power of 2
    if val & (val - 1) != 0 {
        // Not a power of 2: normalize to [0x8000, 0xFFFF] range
        if l > 16 {
            // Round up to avoid losing precision. This is (val >> (l-16)) rounded up,
            // but avoids overflow that adding a bias before shifting could cause.
            val = ((val - 1) >> (l - 16)) + 1;
        } else {
            val <<= 16 - l;
        }
        l = (l - 1) << frac;
        // Extract fractional bits via iterative squaring.
        // Each iteration determines one fractional bit (MSB to LSB).
        loop {
            // b = 1 if val overflows 16 bits, 0 otherwise
            let b = (val >> 16) as i32;
            l += b << frac;
            // Normalize: divide by 2 if overflow occurred
            val = (val + b as u32) >> (b as u32);
            // Square and scale back to 16-bit range
            val = (val * val + 0x7FFF) >> 15;
            if frac <= 0 {
                break;
            }
            frac -= 1;
        }
        // If val is not exactly 0x8000, round up the remainder
        l + if val > 0x8000 { 1 } else { 0 }
    } else {
        // Exact power of 2: no fractional bits needed
        (l - 1) << frac
    }
}

/// Compute the number of bits needed to encode V(N,K) codewords for K in [0, maxk].
///
/// Sets `bits[k] = ceil(log2(V(n, k)))` in Q`frac` format. `bits[0]` is always 0.
/// Used by rate.c for bit allocation tables.
pub fn get_required_bits(bits: &mut [i16], n: i32, maxk: i32, frac: i32) {
    debug_assert!(maxk > 0);
    debug_assert!(bits.len() > maxk as usize);
    bits[0] = 0;
    for k in 1..=maxk {
        bits[k as usize] = log2_frac(celt_pvq_v(n, k), frac) as i16;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celt::range_coder::{RangeDecoder, RangeEncoder};

    // --- Table consistency ---

    #[test]
    fn test_u_boundary_values() {
        // U(0,0) = 1 (base case)
        assert_eq!(celt_pvq_u(0, 0), 1);
        // U(0, K>0) = 0
        assert_eq!(celt_pvq_u(0, 1), 0);
        assert_eq!(celt_pvq_u(0, 100), 0);
        // U(1, K>=1) = 1
        assert_eq!(celt_pvq_u(1, 1), 1);
        assert_eq!(celt_pvq_u(1, 50), 1);
        assert_eq!(celt_pvq_u(1, 176), 1);
        // U(2, K) = 2K-1 for K>=2
        assert_eq!(celt_pvq_u(2, 2), 3);
        assert_eq!(celt_pvq_u(2, 10), 19);
        assert_eq!(celt_pvq_u(2, 100), 199);
        assert_eq!(celt_pvq_u(2, 176), 351);
    }

    #[test]
    fn test_u_symmetry() {
        // U(N,K) = U(K,N) for several pairs
        let pairs = [(3, 5), (4, 7), (5, 10), (6, 12), (7, 9), (2, 100)];
        for (n, k) in pairs {
            assert_eq!(
                celt_pvq_u(n, k),
                celt_pvq_u(k, n),
                "U({},{}) != U({},{})",
                n,
                k,
                k,
                n
            );
        }
    }

    #[test]
    fn test_u_known_from_table() {
        // Spot-check values from the U[10][10] table in the C source comments
        assert_eq!(celt_pvq_u(3, 3), 13);
        assert_eq!(celt_pvq_u(3, 4), 25);
        assert_eq!(celt_pvq_u(4, 4), 63);
        assert_eq!(celt_pvq_u(5, 5), 321);
        assert_eq!(celt_pvq_u(6, 6), 1683);
        assert_eq!(celt_pvq_u(7, 7), 8989);
        assert_eq!(celt_pvq_u(8, 8), 48639);
        assert_eq!(celt_pvq_u(9, 9), 265729);
        assert_eq!(celt_pvq_u(14, 14), 1409933619);
    }

    #[test]
    fn test_v_known_values() {
        // V(1, K) = 2 for K > 0
        for k in 1..=10 {
            assert_eq!(celt_pvq_v(1, k), 2, "V(1,{}) should be 2", k);
        }
        // V(2, K) = 4K
        for k in 1..=20 {
            assert_eq!(celt_pvq_v(2, k), 4 * k as u32, "V(2,{}) should be {}", k, 4 * k);
        }
        // V(3, K) = 4K^2 + 2 for K > 0 (from the polynomial in the C comments)
        for k in 1..=10 {
            let expected = 4 * (k as u32) * (k as u32) + 2;
            assert_eq!(celt_pvq_v(3, k), expected, "V(3,{}) should be {}", k, expected);
        }
    }

    // --- Encode/decode roundtrip ---

    #[test]
    fn test_icwrs_cwrsi_roundtrip_exhaustive_small() {
        // Exhaustively test all codewords for small N and K
        for n in 2..=5i32 {
            for k in 1..=4i32 {
                let v = celt_pvq_v(n, k);
                for idx in 0..v {
                    let mut y = vec![0i32; n as usize];
                    let yy = cwrsi(n, k, idx, &mut y);

                    // Verify L1 norm equals K
                    let l1: i32 = y.iter().map(|&yi| yi.abs()).sum();
                    assert_eq!(l1, k, "L1 norm mismatch: n={}, k={}, idx={}, y={:?}", n, k, idx, y);

                    // Verify squared norm
                    let expected_yy: i32 = y.iter().map(|&yi| (yi as i16 as i32) * (yi as i16 as i32)).sum();
                    assert_eq!(yy, expected_yy, "yy mismatch: n={}, k={}, idx={}", n, k, idx);

                    // Verify roundtrip
                    let encoded = icwrs(n, &y);
                    assert_eq!(
                        encoded, idx,
                        "roundtrip failed: n={}, k={}, idx={}, y={:?}",
                        n, k, idx, y
                    );
                }
            }
        }
    }

    #[test]
    fn test_known_n2_k1_mappings() {
        // V(2,1) = 4 codewords. Verify the exact mapping.
        let expected: [(u32, [i32; 2]); 4] = [
            (0, [1, 0]),
            (1, [0, 1]),
            (2, [0, -1]),
            (3, [-1, 0]),
        ];
        for &(idx, ref expected_y) in &expected {
            let mut y = [0i32; 2];
            cwrsi(2, 1, idx, &mut y);
            assert_eq!(&y, expected_y, "cwrsi(2, 1, {}) = {:?}, expected {:?}", idx, y, expected_y);
            assert_eq!(icwrs(2, &y), idx, "icwrs(2, {:?}) != {}", y, idx);
        }
    }

    #[test]
    fn test_larger_roundtrip() {
        // Test with larger N and K (but not exhaustive)
        let cases: &[(i32, i32)] = &[(8, 3), (10, 2), (4, 8), (6, 6), (14, 1)];
        for &(n, k) in cases {
            let v = celt_pvq_v(n, k);
            // Test first, last, and middle indices
            let test_indices = [0u32, 1, v / 4, v / 2, 3 * v / 4, v - 2, v - 1];
            for &idx in &test_indices {
                if idx >= v {
                    continue;
                }
                let mut y = vec![0i32; n as usize];
                cwrsi(n, k, idx, &mut y);
                let l1: i32 = y.iter().map(|&yi| yi.abs()).sum();
                assert_eq!(l1, k, "L1 norm: n={}, k={}, idx={}", n, k, idx);
                let encoded = icwrs(n, &y);
                assert_eq!(encoded, idx, "roundtrip: n={}, k={}, idx={}", n, k, idx);
            }
        }
    }

    // --- Range coder integration ---

    #[test]
    fn test_encode_decode_pulses_roundtrip() {
        let test_vectors: &[(&[i32], i32)] = &[
            (&[1, 0], 1),
            (&[0, -1], 1),
            (&[1, -1, 0, 1], 3),
            (&[2, 0, -1], 3),
            (&[0, 0, 0, 0, 1], 1),
            (&[-3, 0, 2, 0, 1, 0], 6),
        ];

        for &(y_orig, k) in test_vectors {
            let n = y_orig.len() as i32;
            // Verify L1 norm matches k
            let l1: i32 = y_orig.iter().map(|&yi| yi.abs()).sum();
            assert_eq!(l1, k, "bad test vector: {:?}", y_orig);

            let mut buf = vec![0u8; 256];
            {
                let mut enc = RangeEncoder::new(&mut buf);
                encode_pulses(y_orig, n, k, &mut enc);
                enc.done();
            }

            let mut y_dec = vec![0i32; n as usize];
            {
                let mut dec = RangeDecoder::new(&buf);
                let yy = decode_pulses(&mut y_dec, n, k, &mut dec);
                let expected_yy: i32 = y_orig.iter().map(|&yi| {
                    (yi as i16 as i32) * (yi as i16 as i32)
                }).sum();
                assert_eq!(yy, expected_yy, "yy mismatch for {:?}", y_orig);
            }

            assert_eq!(&y_dec[..], y_orig, "decode mismatch for {:?}", y_orig);
        }
    }

    // --- log2_frac ---

    #[test]
    fn test_log2_frac_powers_of_two() {
        // Exact powers of 2: log2(2^k) = k, no fractional bits needed
        for k in 0..31i32 {
            let val = 1u32 << k;
            assert_eq!(log2_frac(val, 4), k << 4, "log2_frac(2^{}, 4) should be {}", k, k << 4);
        }
    }

    #[test]
    fn test_log2_frac_conservative() {
        // log2_frac always rounds up (conservative)
        // log2(3) ≈ 1.585, so with frac=4: 1.585 * 16 ≈ 25.36, rounds up to 26
        let result = log2_frac(3, 4);
        assert!(result >= 25, "log2_frac(3, 4) = {} should be >= 25", result);
        assert!(result <= 26, "log2_frac(3, 4) = {} should be <= 26", result);

        // log2(5) ≈ 2.322, so with frac=4: 2.322 * 16 ≈ 37.15, rounds up to 38
        let result = log2_frac(5, 4);
        assert!(result >= 37, "log2_frac(5, 4) = {} should be >= 37", result);
        assert!(result <= 38, "log2_frac(5, 4) = {} should be <= 38", result);
    }

    #[test]
    fn test_log2_frac_large_values() {
        // log2(0xFFFFFFFF) ≈ 32.0 (just under)
        let result = log2_frac(0xFFFFFFFF, 4);
        // Should be close to 32 << 4 = 512 but slightly less
        assert!(result >= 511 && result <= 512, "log2_frac(MAX, 4) = {}", result);
    }

    // --- get_required_bits ---

    #[test]
    fn test_get_required_bits_basic() {
        let mut bits = vec![0i16; 11];
        get_required_bits(&mut bits, 2, 10, 4);
        // bits[0] = 0 always
        assert_eq!(bits[0], 0);
        // V(2, k) = 4k, so bits[k] = log2_frac(4k, 4)
        for k in 1..=10 {
            let expected = log2_frac(4 * k as u32, 4) as i16;
            assert_eq!(bits[k], expected, "bits[{}] mismatch", k);
        }
    }

    #[test]
    fn test_get_required_bits_monotonic() {
        // Required bits should be non-decreasing with K
        let mut bits = vec![0i16; 21];
        get_required_bits(&mut bits, 4, 20, 4);
        for k in 1..20 {
            assert!(
                bits[k + 1] >= bits[k],
                "bits not monotonic at k={}: {} < {}",
                k,
                bits[k + 1],
                bits[k]
            );
        }
    }

    // --- Edge cases ---

    #[test]
    fn test_single_pulse_all_dimensions() {
        // K=1 with various N: only one dimension gets a ±1
        for n in 2..=10i32 {
            let v = celt_pvq_v(n, 1);
            assert_eq!(v, 2 * n as u32, "V({}, 1) should be {}", n, 2 * n);
            for idx in 0..v {
                let mut y = vec![0i32; n as usize];
                cwrsi(n, 1, idx, &mut y);
                let l1: i32 = y.iter().map(|&yi| yi.abs()).sum();
                assert_eq!(l1, 1);
                // Exactly one non-zero element
                let nonzero = y.iter().filter(|&&yi| yi != 0).count();
                assert_eq!(nonzero, 1, "n={}, idx={}, y={:?}", n, idx, y);
            }
        }
    }

    #[test]
    fn test_n2_large_k() {
        // N=2 with larger K values
        for k in &[5, 10, 50, 100] {
            let v = celt_pvq_v(2, *k);
            assert_eq!(v, 4 * *k as u32);
            // Test a few indices
            for idx in [0, v / 2, v - 1] {
                let mut y = [0i32; 2];
                cwrsi(2, *k, idx, &mut y);
                let l1: i32 = y.iter().map(|&yi| yi.abs()).sum();
                assert_eq!(l1, *k);
                assert_eq!(icwrs(2, &y), idx);
            }
        }
    }

    // --- Table size verification ---

    #[test]
    fn test_table_last_entry() {
        // The last entry should be U(14, 14) = 1409933619
        assert_eq!(CELT_PVQ_U_DATA[1271], 1409933619);
    }

    #[test]
    fn test_row_offsets_consistency() {
        // Verify that each row offset + the row's minimum K gives a valid index
        // and the value matches U(n, n) for diagonal entries
        let diagonal_values: [u32; 15] = [
            1,         // U(0,0)
            1,         // U(1,1)
            3,         // U(2,2)
            13,        // U(3,3)
            63,        // U(4,4)
            321,       // U(5,5)
            1683,      // U(6,6)
            8989,      // U(7,7)
            48639,     // U(8,8)
            265729,    // U(9,9)
            1462563,   // U(10,10)
            8097453,   // U(11,11)
            45046719,  // U(12,12)
            251595969, // U(13,13)
            1409933619, // U(14,14)
        ];
        for n in 0..15 {
            assert_eq!(
                pvq_u_row(n, n),
                diagonal_values[n],
                "U({0},{0}) mismatch",
                n
            );
        }
    }
}
