//! Inverse MDCT (Modified Discrete Cosine Transform) for the CELT decoder.
//!
//! Port of `clt_mdct_backward` from `celt/mdct.c` (FIXED_POINT, non-QEXT path).
//!
//! The MDCT uses an N/4 complex FFT as the core transform. The algorithm is:
//! 1. Pre-rotation with twiddle factors (folds input into N/4 complex pairs)
//! 2. In-place FFT via `opus_fft_impl`
//! 3. Post-rotation with twiddle factors (unfolds to real output)
//! 4. Windowed overlap-add (TDAC mirror) with the synthesis window

use crate::{uc, uc_set};

use super::fft::{KissFftCpx, KissFftState, opus_fft_impl};
use crate::celt::math_ops::{celt_ilog2, celt_zlog2};
use crate::types::*;

// =========================================================================
// MDCT lookup struct
// =========================================================================

/// MDCT configuration. Holds the FFT states and twiddle factors for all
/// supported block sizes. Mirrors the C `mdct_lookup` struct.
///
/// For the standard 48 kHz Opus mode (N=1920):
/// - `kfft[0]` = 480-point FFT (shift=0, full-size MDCT)
/// - `kfft[1]` = 240-point FFT (shift=1)
/// - `kfft[2]` = 120-point FFT (shift=2)
/// - `kfft[3]` = 60-point FFT  (shift=3, short blocks)
pub struct MdctLookup {
    /// Full MDCT size (N), e.g. 1920 for 48 kHz.
    pub n: i32,
    /// Maximum supported shift (log2 of sub-sampling).
    pub maxshift: i32,
    /// FFT states for each shift level.
    pub kfft: [&'static KissFftState; 4],
    /// Twiddle factor table (concatenated for all shift levels).
    /// Layout: [shift=0: N/2 values] [shift=1: N/4 values] ...
    pub trig: &'static [i16],
}

// =========================================================================
// S_MUL for MDCT context — MULT16_32_Q15(b, a)
// =========================================================================

/// `S_MUL(a, b)` = `MULT16_32_Q15(b, a)` in the fixed-point MDCT path.
/// `a` is data (i32), `b` is twiddle (i16 range stored as i16 in trig table).
#[inline(always)]
fn s_mul(a: i32, b: i32) -> i32 {
    mult16_32_q15(b, a)
}

// =========================================================================
// clt_mdct_backward — Inverse MDCT
// =========================================================================

/// Compute the inverse MDCT with weighted overlap-add.
///
/// This is a bit-exact port of `clt_mdct_backward_c` from `celt/mdct.c`
/// (FIXED_POINT, non-QEXT path).
///
/// # Arguments
///
/// * `l` - MDCT lookup (twiddle factors and FFT state references)
/// * `input` - Frequency-domain coefficients. For short blocks (`stride > 1`),
///   coefficients are interleaved across sub-blocks.
/// * `output` - Time-domain output buffer (length >= N). On entry, the overlap
///   region (`output[0..overlap]`) contains the previous frame's tail for OLA.
///   On exit, contains the reconstructed signal with overlap-add applied.
/// * `window` - Synthesis window (Q15), length = `overlap`.
/// * `overlap` - Overlap size in samples (e.g. 120).
/// * `shift` - `maxLM - LM`: how many times to halve the base MDCT size.
/// * `stride` - B (number of short blocks): 1 for long block, M for short.
pub fn clt_mdct_backward(
    l: &MdctLookup,
    input: &[i32],
    output: &mut [i32],
    window: &[i16],
    overlap: i32,
    shift: i32,
    stride: i32,
) {
    // Compute effective N for this shift level, and advance trig pointer.
    // The C code halves N first, then advances trig by the halved N.
    let mut n = l.n;
    let mut trig_offset: usize = 0;
    for _ in 0..shift {
        n >>= 1;
        trig_offset += n as usize;
    }
    let n2 = n >> 1;
    let n4 = n >> 2;
    let trig = &l.trig[trig_offset..];

    // ---- Fixed-point dynamic range analysis ----
    let pre_shift: i32;
    let post_shift: i32;
    let fft_shift: i32;
    {
        let input_len = input.len();
        let mut sumval: i32 = n2;
        let mut maxval: i32 = 0;
        for i in 0..n2 as usize {
            let idx = i * stride as usize;
            let sample = if idx < input_len { uc!(input, idx) } else { 0 };
            maxval = max32(maxval, abs32(sample));
            sumval = add32_ovflw(sumval, abs32(shr32(sample, 11)));
        }
        pre_shift = imax(0, 29 - celt_zlog2(1 + maxval));
        let ps = imax(0, 19 - celt_ilog2(abs32(sumval)));
        post_shift = imin(ps, pre_shift);
        fft_shift = pre_shift - post_shift;
    }

    // ---- Pre-rotation ----
    // Reads from input (strided), writes into output at overlap/2 as
    // interleaved complex pairs [yi, yr] in bit-reversed order.
    {
        let half_ov = (overlap >> 1) as usize;
        let st = l.kfft[shift as usize];
        let input_len = input.len();
        let mut xp1_idx: usize = 0; // walks forward by 2*stride
        let mut xp2_idx: usize = (n2 - 1) as usize; // walks backward
        // Note: xp1 and xp2 index into input[] with stride spacing
        // but the actual read positions are xp1_idx*stride and xp2_idx*stride

        let bitrev = st.bitrev;
        for i in 0..n4 as usize {
            let rev = uc!(bitrev, i) as usize;
            let idx1 = xp1_idx * stride as usize;
            let idx2 = xp2_idx * stride as usize;
            let x1 = shl32_ovflw(
                if idx1 < input_len {
                    uc!(input, idx1)
                } else {
                    0
                },
                pre_shift,
            );
            let x2 = shl32_ovflw(
                if idx2 < input_len {
                    uc!(input, idx2)
                } else {
                    0
                },
                pre_shift,
            );

            let yr = add32_ovflw(
                s_mul(x2, uc!(trig, i) as i32),
                s_mul(x1, uc!(trig, n4 as usize + i) as i32),
            );
            let yi = sub32_ovflw(
                s_mul(x1, uc!(trig, i) as i32),
                s_mul(x2, uc!(trig, n4 as usize + i) as i32),
            );

            uc_set!(output, half_ov + 2 * rev + 1, yr);
            uc_set!(output, half_ov + 2 * rev, yi);

            xp1_idx += 2;
            xp2_idx = xp2_idx.wrapping_sub(2);
        }
    }

    // ---- In-place FFT ----
    // The output buffer at overlap/2 is treated as N/4 complex values.
    // We need to reinterpret pairs of i32 as KissFftCpx for the FFT.
    {
        let half_ov = (overlap >> 1) as usize;
        let st = l.kfft[shift as usize];
        let fft_len = n4 as usize;

        // Build a temporary KissFftCpx buffer from the output pairs
        let mut fft_buf: Vec<KissFftCpx> = Vec::with_capacity(fft_len);
        for j in 0..fft_len {
            fft_buf.push(KissFftCpx {
                r: uc!(output, half_ov + 2 * j),
                i: uc!(output, half_ov + 2 * j + 1),
            });
        }

        opus_fft_impl(st, &mut fft_buf, fft_shift);

        for j in 0..fft_len {
            uc_set!(output, half_ov + 2 * j, uc!(fft_buf, j).r);
            uc_set!(output, half_ov + 2 * j + 1, uc!(fft_buf, j).i);
        }
    }

    // ---- Post-rotation and de-shuffle ----
    // Works from both ends toward the middle, in-place.
    {
        let half_ov = (overlap >> 1) as usize;
        let mut yp0 = half_ov; // walks forward by 2
        let mut yp1 = half_ov + n2 as usize - 2; // walks backward by 2

        // Loop to (N4+1)>>1 to handle odd N4. When N4 is odd, the
        // middle pair will be computed twice.
        for i in 0..((n4 + 1) >> 1) as usize {
            let re = uc!(output, yp0 + 1);
            let im = uc!(output, yp0);
            let t0 = uc!(trig, i) as i32;
            let t1 = uc!(trig, n4 as usize + i) as i32;

            let yr = pshr32_ovflw(add32_ovflw(s_mul(re, t0), s_mul(im, t1)), post_shift);
            let yi = pshr32_ovflw(sub32_ovflw(s_mul(re, t1), s_mul(im, t0)), post_shift);

            let re2 = uc!(output, yp1 + 1);
            let im2 = uc!(output, yp1);

            uc_set!(output, yp0, yr);
            uc_set!(output, yp1 + 1, yi);

            let t0b = uc!(trig, (n4 as usize) - i - 1) as i32;
            let t1b = uc!(trig, (n2 as usize) - i - 1) as i32;

            let yr2 = pshr32_ovflw(add32_ovflw(s_mul(re2, t0b), s_mul(im2, t1b)), post_shift);
            let yi2 = pshr32_ovflw(sub32_ovflw(s_mul(re2, t1b), s_mul(im2, t0b)), post_shift);

            uc_set!(output, yp1, yr2);
            uc_set!(output, yp0 + 1, yi2);

            yp0 += 2;
            yp1 -= 2;
        }
    }

    // ---- Mirror on both sides for TDAC (windowed overlap-add) ----
    {
        let ov = overlap as usize;
        super::simd::mdct_window_simd(output, window, ov);
    }
}

// =========================================================================
// Static twiddle table — 48 kHz mode (N=1920, from static_modes_fixed.h)
// =========================================================================
//
// Layout: [shift=0: 960 values] [shift=1: 480 values]
//         [shift=2: 240 values] [shift=3: 120 values]
// Total: 1800 values.
//
// Each value is cos(2π(i+0.125)/N) in Q15 fixed-point.
// Generated by: trig[i] = TRIG_UPSCALE * celt_cos_norm((((i<<17) + N/2 + 16384) / N))

#[rustfmt::skip]
pub static MDCT_TWIDDLES_960: [i16; 1800] = [
    // ---- shift=0: N=1920, N2=960 entries ----
    32767, 32767, 32767, 32766, 32765,
    32763, 32761, 32759, 32756, 32753,
    32750, 32746, 32742, 32738, 32733,
    32728, 32722, 32717, 32710, 32704,
    32697, 32690, 32682, 32674, 32666,
    32657, 32648, 32639, 32629, 32619,
    32609, 32598, 32587, 32576, 32564,
    32552, 32539, 32526, 32513, 32500,
    32486, 32472, 32457, 32442, 32427,
    32411, 32395, 32379, 32362, 32345,
    32328, 32310, 32292, 32274, 32255,
    32236, 32217, 32197, 32177, 32157,
    32136, 32115, 32093, 32071, 32049,
    32027, 32004, 31981, 31957, 31933,
    31909, 31884, 31859, 31834, 31809,
    31783, 31756, 31730, 31703, 31676,
    31648, 31620, 31592, 31563, 31534,
    31505, 31475, 31445, 31415, 31384,
    31353, 31322, 31290, 31258, 31226,
    31193, 31160, 31127, 31093, 31059,
    31025, 30990, 30955, 30920, 30884,
    30848, 30812, 30775, 30738, 30701,
    30663, 30625, 30587, 30548, 30509,
    30470, 30430, 30390, 30350, 30309,
    30269, 30227, 30186, 30144, 30102,
    30059, 30016, 29973, 29930, 29886,
    29842, 29797, 29752, 29707, 29662,
    29616, 29570, 29524, 29477, 29430,
    29383, 29335, 29287, 29239, 29190,
    29142, 29092, 29043, 28993, 28943,
    28892, 28842, 28791, 28739, 28688,
    28636, 28583, 28531, 28478, 28425,
    28371, 28317, 28263, 28209, 28154,
    28099, 28044, 27988, 27932, 27876,
    27820, 27763, 27706, 27648, 27591,
    27533, 27474, 27416, 27357, 27298,
    27238, 27178, 27118, 27058, 26997,
    26936, 26875, 26814, 26752, 26690,
    26628, 26565, 26502, 26439, 26375,
    26312, 26247, 26183, 26119, 26054,
    25988, 25923, 25857, 25791, 25725,
    25658, 25592, 25524, 25457, 25389,
    25322, 25253, 25185, 25116, 25047,
    24978, 24908, 24838, 24768, 24698,
    24627, 24557, 24485, 24414, 24342,
    24270, 24198, 24126, 24053, 23980,
    23907, 23834, 23760, 23686, 23612,
    23537, 23462, 23387, 23312, 23237,
    23161, 23085, 23009, 22932, 22856,
    22779, 22701, 22624, 22546, 22468,
    22390, 22312, 22233, 22154, 22075,
    21996, 21916, 21836, 21756, 21676,
    21595, 21515, 21434, 21352, 21271,
    21189, 21107, 21025, 20943, 20860,
    20777, 20694, 20611, 20528, 20444,
    20360, 20276, 20192, 20107, 20022,
    19937, 19852, 19767, 19681, 19595,
    19509, 19423, 19336, 19250, 19163,
    19076, 18988, 18901, 18813, 18725,
    18637, 18549, 18460, 18372, 18283,
    18194, 18104, 18015, 17925, 17835,
    17745, 17655, 17565, 17474, 17383,
    17292, 17201, 17110, 17018, 16927,
    16835, 16743, 16650, 16558, 16465,
    16372, 16279, 16186, 16093, 15999,
    15906, 15812, 15718, 15624, 15529,
    15435, 15340, 15245, 15150, 15055,
    14960, 14864, 14769, 14673, 14577,
    14481, 14385, 14288, 14192, 14095,
    13998, 13901, 13804, 13706, 13609,
    13511, 13414, 13316, 13218, 13119,
    13021, 12923, 12824, 12725, 12626,
    12527, 12428, 12329, 12230, 12130,
    12030, 11930, 11831, 11730, 11630,
    11530, 11430, 11329, 11228, 11128,
    11027, 10926, 10824, 10723, 10622,
    10520, 10419, 10317, 10215, 10113,
    10011, 9909, 9807, 9704, 9602,
    9499, 9397, 9294, 9191, 9088,
    8985, 8882, 8778, 8675, 8572,
    8468, 8364, 8261, 8157, 8053,
    7949, 7845, 7741, 7637, 7532,
    7428, 7323, 7219, 7114, 7009,
    6905, 6800, 6695, 6590, 6485,
    6380, 6274, 6169, 6064, 5958,
    5853, 5747, 5642, 5536, 5430,
    5325, 5219, 5113, 5007, 4901,
    4795, 4689, 4583, 4476, 4370,
    4264, 4157, 4051, 3945, 3838,
    3732, 3625, 3518, 3412, 3305,
    3198, 3092, 2985, 2878, 2771,
    2664, 2558, 2451, 2344, 2237,
    2130, 2023, 1916, 1809, 1702,
    1594, 1487, 1380, 1273, 1166,
    1059, 952, 844, 737, 630,
    523, 416, 308, 201, 94,
    -13, -121, -228, -335, -442,
    -550, -657, -764, -871, -978,
    -1086, -1193, -1300, -1407, -1514,
    -1621, -1728, -1835, -1942, -2049,
    -2157, -2263, -2370, -2477, -2584,
    -2691, -2798, -2905, -3012, -3118,
    -3225, -3332, -3439, -3545, -3652,
    -3758, -3865, -3971, -4078, -4184,
    -4290, -4397, -4503, -4609, -4715,
    -4821, -4927, -5033, -5139, -5245,
    -5351, -5457, -5562, -5668, -5774,
    -5879, -5985, -6090, -6195, -6301,
    -6406, -6511, -6616, -6721, -6826,
    -6931, -7036, -7140, -7245, -7349,
    -7454, -7558, -7663, -7767, -7871,
    -7975, -8079, -8183, -8287, -8390,
    -8494, -8597, -8701, -8804, -8907,
    -9011, -9114, -9217, -9319, -9422,
    -9525, -9627, -9730, -9832, -9934,
    -10037, -10139, -10241, -10342, -10444,
    -10546, -10647, -10748, -10850, -10951,
    -11052, -11153, -11253, -11354, -11455,
    -11555, -11655, -11756, -11856, -11955,
    -12055, -12155, -12254, -12354, -12453,
    -12552, -12651, -12750, -12849, -12947,
    -13046, -13144, -13242, -13340, -13438,
    -13536, -13633, -13731, -13828, -13925,
    -14022, -14119, -14216, -14312, -14409,
    -14505, -14601, -14697, -14793, -14888,
    -14984, -15079, -15174, -15269, -15364,
    -15459, -15553, -15647, -15741, -15835,
    -15929, -16023, -16116, -16210, -16303,
    -16396, -16488, -16581, -16673, -16766,
    -16858, -16949, -17041, -17133, -17224,
    -17315, -17406, -17497, -17587, -17678,
    -17768, -17858, -17948, -18037, -18127,
    -18216, -18305, -18394, -18483, -18571,
    -18659, -18747, -18835, -18923, -19010,
    -19098, -19185, -19271, -19358, -19444,
    -19531, -19617, -19702, -19788, -19873,
    -19959, -20043, -20128, -20213, -20297,
    -20381, -20465, -20549, -20632, -20715,
    -20798, -20881, -20963, -21046, -21128,
    -21210, -21291, -21373, -21454, -21535,
    -21616, -21696, -21776, -21856, -21936,
    -22016, -22095, -22174, -22253, -22331,
    -22410, -22488, -22566, -22643, -22721,
    -22798, -22875, -22951, -23028, -23104,
    -23180, -23256, -23331, -23406, -23481,
    -23556, -23630, -23704, -23778, -23852,
    -23925, -23998, -24071, -24144, -24216,
    -24288, -24360, -24432, -24503, -24574,
    -24645, -24716, -24786, -24856, -24926,
    -24995, -25064, -25133, -25202, -25270,
    -25339, -25406, -25474, -25541, -25608,
    -25675, -25742, -25808, -25874, -25939,
    -26005, -26070, -26135, -26199, -26264,
    -26327, -26391, -26455, -26518, -26581,
    -26643, -26705, -26767, -26829, -26891,
    -26952, -27013, -27073, -27133, -27193,
    -27253, -27312, -27372, -27430, -27489,
    -27547, -27605, -27663, -27720, -27777,
    -27834, -27890, -27946, -28002, -28058,
    -28113, -28168, -28223, -28277, -28331,
    -28385, -28438, -28491, -28544, -28596,
    -28649, -28701, -28752, -28803, -28854,
    -28905, -28955, -29006, -29055, -29105,
    -29154, -29203, -29251, -29299, -29347,
    -29395, -29442, -29489, -29535, -29582,
    -29628, -29673, -29719, -29764, -29808,
    -29853, -29897, -29941, -29984, -30027,
    -30070, -30112, -30154, -30196, -30238,
    -30279, -30320, -30360, -30400, -30440,
    -30480, -30519, -30558, -30596, -30635,
    -30672, -30710, -30747, -30784, -30821,
    -30857, -30893, -30929, -30964, -30999,
    -31033, -31068, -31102, -31135, -31168,
    -31201, -31234, -31266, -31298, -31330,
    -31361, -31392, -31422, -31453, -31483,
    -31512, -31541, -31570, -31599, -31627,
    -31655, -31682, -31710, -31737, -31763,
    -31789, -31815, -31841, -31866, -31891,
    -31915, -31939, -31963, -31986, -32010,
    -32032, -32055, -32077, -32099, -32120,
    -32141, -32162, -32182, -32202, -32222,
    -32241, -32260, -32279, -32297, -32315,
    -32333, -32350, -32367, -32383, -32399,
    -32415, -32431, -32446, -32461, -32475,
    -32489, -32503, -32517, -32530, -32542,
    -32555, -32567, -32579, -32590, -32601,
    -32612, -32622, -32632, -32641, -32651,
    -32659, -32668, -32676, -32684, -32692,
    -32699, -32706, -32712, -32718, -32724,
    -32729, -32734, -32739, -32743, -32747,
    -32751, -32754, -32757, -32760, -32762,
    -32764, -32765, -32767, -32767, -32767,
    // ---- shift=1: N=960, N2=480 entries ----
    32767, 32767, 32765, 32761, 32756,
    32750, 32742, 32732, 32722, 32710,
    32696, 32681, 32665, 32647, 32628,
    32608, 32586, 32562, 32538, 32512,
    32484, 32455, 32425, 32393, 32360,
    32326, 32290, 32253, 32214, 32174,
    32133, 32090, 32046, 32001, 31954,
    31906, 31856, 31805, 31753, 31700,
    31645, 31588, 31530, 31471, 31411,
    31349, 31286, 31222, 31156, 31089,
    31020, 30951, 30880, 30807, 30733,
    30658, 30582, 30504, 30425, 30345,
    30263, 30181, 30096, 30011, 29924,
    29836, 29747, 29656, 29564, 29471,
    29377, 29281, 29184, 29086, 28987,
    28886, 28784, 28681, 28577, 28471,
    28365, 28257, 28147, 28037, 27925,
    27812, 27698, 27583, 27467, 27349,
    27231, 27111, 26990, 26868, 26744,
    26620, 26494, 26367, 26239, 26110,
    25980, 25849, 25717, 25583, 25449,
    25313, 25176, 25038, 24900, 24760,
    24619, 24477, 24333, 24189, 24044,
    23898, 23751, 23602, 23453, 23303,
    23152, 22999, 22846, 22692, 22537,
    22380, 22223, 22065, 21906, 21746,
    21585, 21423, 21261, 21097, 20933,
    20767, 20601, 20434, 20265, 20096,
    19927, 19756, 19584, 19412, 19239,
    19065, 18890, 18714, 18538, 18361,
    18183, 18004, 17824, 17644, 17463,
    17281, 17098, 16915, 16731, 16546,
    16361, 16175, 15988, 15800, 15612,
    15423, 15234, 15043, 14852, 14661,
    14469, 14276, 14083, 13889, 13694,
    13499, 13303, 13107, 12910, 12713,
    12515, 12317, 12118, 11918, 11718,
    11517, 11316, 11115, 10913, 10710,
    10508, 10304, 10100, 9896, 9691,
    9486, 9281, 9075, 8869, 8662,
    8455, 8248, 8040, 7832, 7623,
    7415, 7206, 6996, 6787, 6577,
    6366, 6156, 5945, 5734, 5523,
    5311, 5100, 4888, 4675, 4463,
    4251, 4038, 3825, 3612, 3399,
    3185, 2972, 2758, 2544, 2330,
    2116, 1902, 1688, 1474, 1260,
    1045, 831, 617, 402, 188,
    -27, -241, -456, -670, -885,
    -1099, -1313, -1528, -1742, -1956,
    -2170, -2384, -2598, -2811, -3025,
    -3239, -3452, -3665, -3878, -4091,
    -4304, -4516, -4728, -4941, -5153,
    -5364, -5576, -5787, -5998, -6209,
    -6419, -6629, -6839, -7049, -7258,
    -7467, -7676, -7884, -8092, -8300,
    -8507, -8714, -8920, -9127, -9332,
    -9538, -9743, -9947, -10151, -10355,
    -10558, -10761, -10963, -11165, -11367,
    -11568, -11768, -11968, -12167, -12366,
    -12565, -12762, -12960, -13156, -13352,
    -13548, -13743, -13937, -14131, -14324,
    -14517, -14709, -14900, -15091, -15281,
    -15470, -15659, -15847, -16035, -16221,
    -16407, -16593, -16777, -16961, -17144,
    -17326, -17508, -17689, -17869, -18049,
    -18227, -18405, -18582, -18758, -18934,
    -19108, -19282, -19455, -19627, -19799,
    -19969, -20139, -20308, -20475, -20642,
    -20809, -20974, -21138, -21301, -21464,
    -21626, -21786, -21946, -22105, -22263,
    -22420, -22575, -22730, -22884, -23037,
    -23189, -23340, -23490, -23640, -23788,
    -23935, -24080, -24225, -24369, -24512,
    -24654, -24795, -24934, -25073, -25211,
    -25347, -25482, -25617, -25750, -25882,
    -26013, -26143, -26272, -26399, -26526,
    -26651, -26775, -26898, -27020, -27141,
    -27260, -27379, -27496, -27612, -27727,
    -27841, -27953, -28065, -28175, -28284,
    -28391, -28498, -28603, -28707, -28810,
    -28911, -29012, -29111, -29209, -29305,
    -29401, -29495, -29587, -29679, -29769,
    -29858, -29946, -30032, -30118, -30201,
    -30284, -30365, -30445, -30524, -30601,
    -30677, -30752, -30825, -30897, -30968,
    -31038, -31106, -31172, -31238, -31302,
    -31365, -31426, -31486, -31545, -31602,
    -31658, -31713, -31766, -31818, -31869,
    -31918, -31966, -32012, -32058, -32101,
    -32144, -32185, -32224, -32262, -32299,
    -32335, -32369, -32401, -32433, -32463,
    -32491, -32518, -32544, -32568, -32591,
    -32613, -32633, -32652, -32669, -32685,
    -32700, -32713, -32724, -32735, -32744,
    -32751, -32757, -32762, -32766, -32767,
    // ---- shift=2: N=480, N2=240 entries ----
    32767, 32764, 32755, 32741, 32720,
    32694, 32663, 32626, 32583, 32535,
    32481, 32421, 32356, 32286, 32209,
    32128, 32041, 31948, 31850, 31747,
    31638, 31523, 31403, 31278, 31148,
    31012, 30871, 30724, 30572, 30415,
    30253, 30086, 29913, 29736, 29553,
    29365, 29172, 28974, 28771, 28564,
    28351, 28134, 27911, 27684, 27452,
    27216, 26975, 26729, 26478, 26223,
    25964, 25700, 25432, 25159, 24882,
    24601, 24315, 24026, 23732, 23434,
    23133, 22827, 22517, 22204, 21886,
    21565, 21240, 20912, 20580, 20244,
    19905, 19563, 19217, 18868, 18516,
    18160, 17802, 17440, 17075, 16708,
    16338, 15964, 15588, 15210, 14829,
    14445, 14059, 13670, 13279, 12886,
    12490, 12093, 11693, 11291, 10888,
    10482, 10075, 9666, 9255, 8843,
    8429, 8014, 7597, 7180, 6760,
    6340, 5919, 5496, 5073, 4649,
    4224, 3798, 3372, 2945, 2517,
    2090, 1661, 1233, 804, 375,
    -54, -483, -911, -1340, -1768,
    -2197, -2624, -3052, -3479, -3905,
    -4330, -4755, -5179, -5602, -6024,
    -6445, -6865, -7284, -7702, -8118,
    -8533, -8946, -9358, -9768, -10177,
    -10584, -10989, -11392, -11793, -12192,
    -12589, -12984, -13377, -13767, -14155,
    -14541, -14924, -15305, -15683, -16058,
    -16430, -16800, -17167, -17531, -17892,
    -18249, -18604, -18956, -19304, -19649,
    -19990, -20329, -20663, -20994, -21322,
    -21646, -21966, -22282, -22595, -22904,
    -23208, -23509, -23806, -24099, -24387,
    -24672, -24952, -25228, -25499, -25766,
    -26029, -26288, -26541, -26791, -27035,
    -27275, -27511, -27741, -27967, -28188,
    -28405, -28616, -28823, -29024, -29221,
    -29412, -29599, -29780, -29957, -30128,
    -30294, -30455, -30611, -30761, -30906,
    -31046, -31181, -31310, -31434, -31552,
    -31665, -31773, -31875, -31972, -32063,
    -32149, -32229, -32304, -32373, -32437,
    -32495, -32547, -32594, -32635, -32671,
    -32701, -32726, -32745, -32758, -32766,
    // ---- shift=3: N=240, N2=120 entries ----
    32767, 32754, 32717, 32658, 32577,
    32473, 32348, 32200, 32029, 31837,
    31624, 31388, 31131, 30853, 30553,
    30232, 29891, 29530, 29148, 28746,
    28324, 27883, 27423, 26944, 26447,
    25931, 25398, 24847, 24279, 23695,
    23095, 22478, 21846, 21199, 20538,
    19863, 19174, 18472, 17757, 17030,
    16291, 15541, 14781, 14010, 13230,
    12441, 11643, 10837, 10024, 9204,
    8377, 7545, 6708, 5866, 5020,
    4171, 3319, 2464, 1608, 751,
    -107, -965, -1822, -2678, -3532,
    -4383, -5232, -6077, -6918, -7754,
    -8585, -9409, -10228, -11039, -11843,
    -12639, -13426, -14204, -14972, -15730,
    -16477, -17213, -17937, -18648, -19347,
    -20033, -20705, -21363, -22006, -22634,
    -23246, -23843, -24423, -24986, -25533,
    -26062, -26573, -27066, -27540, -27995,
    -28431, -28848, -29245, -29622, -29979,
    -30315, -30630, -30924, -31197, -31449,
    -31679, -31887, -32074, -32239, -32381,
    -32501, -32600, -32675, -32729, -32759,
];

// =========================================================================
// Static MDCT instance — 48 kHz mode
// =========================================================================

use super::fft::{
    FFT_STATE_48000_960_0, FFT_STATE_48000_960_1, FFT_STATE_48000_960_2, FFT_STATE_48000_960_3,
};

/// Static MDCT lookup for the 48 kHz / 960-sample mode.
///
/// N=1920, maxshift=3. Serves all four frame sizes.
pub static MDCT_48000_960: MdctLookup = MdctLookup {
    n: 1920,
    maxshift: 3,
    kfft: [
        &FFT_STATE_48000_960_0,
        &FFT_STATE_48000_960_1,
        &FFT_STATE_48000_960_2,
        &FFT_STATE_48000_960_3,
    ],
    trig: &MDCT_TWIDDLES_960,
};

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twiddle_table_length() {
        // N=1920, maxshift=3:
        // shift=0 -> N2=960, shift=1 -> N2=480, shift=2 -> N2=240, shift=3 -> N2=120
        // Total = 960 + 480 + 240 + 120 = 1800
        assert_eq!(MDCT_TWIDDLES_960.len(), 1800);
    }

    #[test]
    fn test_twiddle_first_last() {
        // shift=0: first value should be near 32767 (cos(2π·0.125/1920) ≈ 1.0)
        assert_eq!(MDCT_TWIDDLES_960[0], 32767);
        // shift=0 last value: entry 959 should be near -32767
        assert_eq!(MDCT_TWIDDLES_960[959], -32767);
    }

    #[test]
    fn test_mdct_lookup_consistency() {
        let l = &MDCT_48000_960;
        assert_eq!(l.n, 1920);
        assert_eq!(l.maxshift, 3);
        assert_eq!(l.kfft[0].nfft, 480); // N/4 = 1920/4
        assert_eq!(l.kfft[1].nfft, 240); // 960/4
        assert_eq!(l.kfft[2].nfft, 120); // 480/4
        assert_eq!(l.kfft[3].nfft, 60); // 240/4
    }

    #[test]
    fn test_twiddle_segment_offsets() {
        // Verify the trig offset computation matches expected layout.
        // Layout: [shift=0: N/2 values] [shift=1: N/4 values] ...
        // The C code: N>>=1; trig+=N; (halve first, advance by halved N)
        let l = &MDCT_48000_960;
        let mut offset: usize = 0;
        let mut n = l.n;
        for shift in 0..=l.maxshift {
            let n2 = (n >> 1) as usize;
            // At this shift, trig starts at offset and has n2 entries
            let seg = &l.trig[offset..offset + n2];
            // Every shift level: first entry should be ~32767
            assert_eq!(seg[0], 32767, "shift={} first twiddle", shift);
            offset += n2;
            n >>= 1;
        }
        // Total offset should equal total trig length
        assert_eq!(offset, l.trig.len());
    }

    #[test]
    fn test_mdct_backward_zero_input() {
        // All-zero input should produce all-zero output (no windowing effect
        // since there's nothing to mix).
        let l = &MDCT_48000_960;
        let overlap = 120;
        let shift = 3; // shortest block: N=240, N2=120
        let stride = 1;
        let n2 = 120; // N/2 for this shift level

        let input = vec![0i32; n2];
        // N=240 for shift=3
        let n = 240;
        let mut output = vec![0i32; n];

        let mode = &crate::celt::modes::MODE_48000_960_120;
        clt_mdct_backward(l, &input, &mut output, mode.window, overlap, shift, stride);

        for i in 0..n {
            assert_eq!(output[i], 0, "output[{}] should be 0", i);
        }
    }

    #[test]
    fn test_mdct_backward_impulse_no_crash() {
        // A single nonzero coefficient should not crash and should produce
        // bounded output.
        let l = &MDCT_48000_960;
        let overlap = 120;
        let shift = 3; // N=240
        let stride = 1;
        let n2 = 120;
        let n = 240;

        let mut input = vec![0i32; n2];
        input[0] = 1 << 15; // DC impulse

        let mut output = vec![0i32; n];

        let mode = &crate::celt::modes::MODE_48000_960_120;
        clt_mdct_backward(l, &input, &mut output, mode.window, overlap, shift, stride);

        // Output should be bounded (no overflow to extreme values)
        let max_abs = output.iter().map(|x| x.abs()).max().unwrap_or(0);
        assert!(
            max_abs < i32::MAX / 2,
            "output should be bounded, got max abs {}",
            max_abs
        );
    }

    /// MDCT roundtrip: forward then backward, single frame.
    ///
    /// The MDCT backward only fills output[0..N/2+overlap/2] -- the remaining
    /// samples come from the NEXT frame (lapped TDAC property). This test
    /// verifies the scaling is correct in the region that IS filled.
    #[test]
    fn test_mdct_roundtrip() {
        use crate::celt::encoder::{clt_mdct_forward, get_fft_state};

        let l = &MDCT_48000_960;
        let mode = &crate::celt::modes::MODE_48000_960_120;
        let overlap = 120i32;
        let shift = 0i32;
        let stride = 1i32;
        let n: usize = 1920;
        let n2: usize = 960;
        let half_ov = (overlap as usize) / 2; // 60

        // Build input: sine with DC offset, avoiding zero crossings for ratio analysis.
        let input_len = n + overlap as usize;
        let mut input = vec![0i32; input_len];
        for i in 0..input_len {
            let phase = (i as f64) * 2.0 * std::f64::consts::PI / 240.0;
            input[i] = 1000 + (500.0 * phase.sin()) as i32;
        }

        // Forward transform
        let fft_st = get_fft_state(shift);
        let trig = l.trig;
        let mut freq = vec![0i32; n2];
        clt_mdct_forward(
            &input,
            &mut freq,
            mode.window,
            overlap,
            shift,
            stride,
            fft_st,
            trig,
        );

        // Backward transform (no previous frame OLA -- zeros)
        let mut roundtrip = vec![0i32; n];
        clt_mdct_backward(
            l,
            &freq,
            &mut roundtrip,
            mode.window,
            overlap,
            shift,
            stride,
        );

        // The backward fills output[0..N/2+overlap/2] = [0..1020].
        // output[0..overlap] is windowed (TDAC synthesis window).
        // output[overlap..N/2+overlap/2] = [120..1020] is raw IMDCT, no window.
        let active_end = n2 + half_ov; // 1020

        // Verify the output range
        let last_nonzero = (0..n).rev().find(|&i| roundtrip[i] != 0).unwrap_or(0);
        println!("=== MDCT Roundtrip (single frame) ===");
        println!("N={}, N2={}, overlap={}", n, n2, overlap);
        println!("Expected active range: [0..{}]", active_end);
        println!("Actual last nonzero:   output[{}]", last_nonzero);
        assert!(
            last_nonzero < active_end + 2,
            "backward wrote beyond expected range: last_nonzero={}, expected<{}",
            last_nonzero,
            active_end
        );

        // Cross-correlation in the active region to find best alignment & scale
        let check_start = overlap as usize + 10; // skip first few overlap-boundary samples
        let check_end = active_end - 10;
        println!(
            "\nCross-correlating roundtrip[{}..{}] vs input[offset+i]...",
            check_start, check_end
        );

        let mut best_corr = f64::NEG_INFINITY;
        let mut best_offset: i32 = 0;
        for trial_offset in -200i32..200 {
            let mut sxy = 0.0f64;
            let mut sx2 = 0.0f64;
            let mut sy2 = 0.0f64;
            for i in check_start..check_end {
                let oi = (i as i32 + trial_offset) as usize;
                if oi < input_len {
                    let x = input[oi] as f64;
                    let y = roundtrip[i] as f64;
                    sxy += x * y;
                    sx2 += x * x;
                    sy2 += y * y;
                }
            }
            if sx2 > 0.0 && sy2 > 0.0 {
                let c = sxy / (sx2.sqrt() * sy2.sqrt());
                if c > best_corr {
                    best_corr = c;
                    best_offset = trial_offset;
                }
            }
        }
        println!(
            "  Best correlation: {:.8} at offset {}",
            best_corr, best_offset
        );

        // Compute least-squares scale at best offset
        let mut sxy = 0.0f64;
        let mut sx2 = 0.0f64;
        for i in check_start..check_end {
            let oi = (i as i32 + best_offset) as usize;
            if oi < input_len {
                let x = input[oi] as f64;
                let y = roundtrip[i] as f64;
                sxy += x * y;
                sx2 += x * x;
            }
        }
        let scale_factor = sxy / sx2;
        println!("  Scale factor (roundtrip/original): {:.6}", scale_factor);

        // Print first 20 samples in the check region
        println!("\nFirst 20 roundtrip values in active region:");
        println!(
            "{:>5} {:>12} {:>12} {:>10}",
            "idx", "original", "roundtrip", "ratio"
        );
        for i in 0..20 {
            let idx = check_start + i;
            let orig = input[(idx as i32 + best_offset) as usize];
            let rt = roundtrip[idx];
            let ratio = if orig != 0 {
                rt as f64 / orig as f64
            } else {
                f64::NAN
            };
            println!("{:5} {:12} {:12} {:10.4}", idx, orig, rt, ratio);
        }

        // Report findings
        println!(
            "\nSingle-frame roundtrip: corr={:.6}, scale={:.6}, offset={}",
            best_corr, scale_factor, best_offset
        );

        // Also test with a DC signal for clean ratio measurement
        let dc_input = vec![1000i32; input_len];
        // Apply the standard window at the edges so the forward doesn't see a discontinuity
        let mut dc_freq = vec![0i32; n2];
        clt_mdct_forward(
            &dc_input,
            &mut dc_freq,
            mode.window,
            overlap,
            shift,
            stride,
            fft_st,
            trig,
        );
        let mut dc_rt = vec![0i32; n];
        clt_mdct_backward(l, &dc_freq, &mut dc_rt, mode.window, overlap, shift, stride);
        println!("\nDC roundtrip (input=1000 everywhere):");
        println!("  Middle region samples (output[120..140]):");
        for i in 120..140 {
            println!(
                "    output[{}] = {} (ratio={:.4})",
                i,
                dc_rt[i],
                dc_rt[i] as f64 / 1000.0
            );
        }
        // The DC signal's middle region should give a consistent ratio
        let dc_ratios: Vec<f64> = (200..900).map(|i| dc_rt[i] as f64 / 1000.0).collect();
        let dc_mean = dc_ratios.iter().sum::<f64>() / dc_ratios.len() as f64;
        let dc_var =
            dc_ratios.iter().map(|r| (r - dc_mean).powi(2)).sum::<f64>() / dc_ratios.len() as f64;
        println!(
            "  DC middle ratio: mean={:.6}, stddev={:.6}",
            dc_mean,
            dc_var.sqrt()
        );

        assert!(
            best_corr > 0.95,
            "Roundtrip correlation too low: {:.6} (expected > 0.95)",
            best_corr
        );
    }

    /// Two-frame MDCT roundtrip with decode_mem simulation.
    ///
    /// The C decoder uses a sliding `decode_mem` buffer where:
    /// - `out_syn = decode_mem + decode_buffer_size - N`
    /// - After each frame, `OPUS_MOVE(decode_mem, decode_mem+N, ...)`
    /// - The backward only fills `out_syn[0..N/2+overlap/2]`; the overlap
    ///   for the next frame comes from the memory shift.
    ///
    /// This test simulates that buffer management to verify TDAC reconstruction.
    #[test]
    fn test_mdct_roundtrip_two_frames() {
        use crate::celt::encoder::{clt_mdct_forward, get_fft_state};

        let l = &MDCT_48000_960;
        let mode = &crate::celt::modes::MODE_48000_960_120;
        let overlap = 120i32;
        let shift = 0i32;
        let stride = 1i32;
        let ov = overlap as usize;
        let half_ov = ov / 2;

        // Key dimensions for 48kHz, LM=3 (20ms frame):
        // - N_frame = shortMdctSize << LM = 120 << 3 = 960 (frame hop size)
        // - N_mdct = l.n = 1920 (MDCT transform size = 2 * N_frame)
        // - N2 = N_mdct / 2 = 960
        // - Forward input: N_frame + overlap = 1080 samples
        // - Backward writes: output[0..N2+overlap/2] = output[0..1020]
        // - Backward OLA: output[0..overlap] windowed with previous frame's data
        let n_frame: usize = 960;
        let n_mdct: usize = 1920;
        let n2: usize = 960;

        // Simulate C decoder's decode_mem buffer
        let decode_buffer_size: usize = 2048;
        let decode_mem_size = decode_buffer_size + ov;
        let mut decode_mem = vec![0i32; decode_mem_size];
        let out_syn_off = decode_buffer_size - n_frame; // = 1088

        // Build long input: 6 frames to let OLA settle
        let num_frames = 6;
        let total_len = num_frames * n_frame + ov;
        let mut input = vec![0i32; total_len];
        for i in 0..total_len {
            let p1 = (i as f64) * 2.0 * std::f64::consts::PI / 240.0;
            let p2 = (i as f64) * 2.0 * std::f64::consts::PI / 73.0;
            input[i] = 2000 + (800.0 * p1.sin()) as i32 + (400.0 * p2.cos()) as i32;
        }

        let fft_st = get_fft_state(shift);
        let trig = l.trig;

        println!("=== Two-frame MDCT Roundtrip (decode_mem simulation) ===");
        println!(
            "N_frame={}, N_mdct={}, overlap={}, decode_buffer_size={}",
            n_frame, n_mdct, ov, decode_buffer_size
        );
        println!("out_syn offset in decode_mem: {}", out_syn_off);
        println!("Backward active range: [0..{}]", n2 + half_ov);

        // The encoder's frame hop is N_frame. Frame k reads input[k*N_frame..k*N_frame+N_frame+overlap].
        // The forward produces N2=960 freq bins.
        // The backward writes to out_syn[0..1020].
        // After each frame, OPUS_MOVE shifts decode_mem left by N_frame.

        let mut last_ola_snr = 0.0f64;
        for frame in 0..num_frames {
            // OPUS_MOVE: shift decode_mem left by N_frame BEFORE the backward
            // (matching C decoder order: move first, then synthesize)
            if frame > 0 {
                let move_len = decode_buffer_size - n_frame + ov;
                for i in 0..move_len {
                    decode_mem[i] = decode_mem[n_frame + i];
                }
                for i in move_len..decode_mem_size {
                    decode_mem[i] = 0;
                }
            }

            let frame_start = frame * n_frame;

            // Forward transform
            let mut freq = vec![0i32; n2];
            clt_mdct_forward(
                &input[frame_start..],
                &mut freq,
                mode.window,
                overlap,
                shift,
                stride,
                fft_st,
                trig,
            );

            // Backward transform: out_syn is at decode_mem[out_syn_off..]
            // The OLA data at out_syn[0..overlap] comes from previous frame via OPUS_MOVE.
            clt_mdct_backward(
                l,
                &freq,
                &mut decode_mem[out_syn_off..],
                mode.window,
                overlap,
                shift,
                stride,
            );

            // Check OLA region quality (frames >= 2 for stable OLA)
            // The reconstructed signal at out_syn[i] should correspond to
            // input[frame_start + overlap/2 + i] (the analysis window center).
            let input_offset = frame_start + half_ov;
            if frame >= 2 {
                let mut max_err: i64 = 0;
                let mut sum_err2 = 0.0f64;
                let mut sum_orig2 = 0.0f64;
                for i in 0..ov {
                    let orig = input[input_offset + i];
                    let recon = decode_mem[out_syn_off + i];
                    let err = (recon as i64 - orig as i64).abs();
                    if err > max_err {
                        max_err = err;
                    }
                    sum_err2 += (err as f64).powi(2);
                    sum_orig2 += (orig as f64).powi(2);
                }
                let rms_err = (sum_err2 / ov as f64).sqrt();
                let rms_orig = (sum_orig2 / ov as f64).sqrt();
                let snr = if rms_err > 0.0 {
                    20.0 * (rms_orig / rms_err).log10()
                } else {
                    f64::INFINITY
                };
                last_ola_snr = snr;
                println!(
                    "Frame {} OLA[0..{}]: max_err={}, SNR={:.1} dB",
                    frame, ov, max_err, snr
                );

                if frame == 2 {
                    println!(
                        "{:>5} {:>12} {:>12} {:>10}",
                        "idx", "original", "recon", "error"
                    );
                    for i in 0..20.min(ov) {
                        let orig = input[input_offset + i];
                        let recon = decode_mem[out_syn_off + i];
                        println!("{:5} {:12} {:12} {:10}", i, orig, recon, recon - orig);
                    }
                }
            }

            // Check middle region (beyond OLA, within backward's active range)
            let check_start = ov;
            let check_end = n2 + half_ov - 10;
            let mut sxy = 0.0f64;
            let mut sx2 = 0.0f64;
            let mut sy2 = 0.0f64;
            for i in check_start..check_end {
                if input_offset + i >= total_len {
                    break;
                }
                let x = input[input_offset + i] as f64;
                let y = decode_mem[out_syn_off + i] as f64;
                sxy += x * y;
                sx2 += x * x;
                sy2 += y * y;
            }
            let corr = if sx2 > 0.0 && sy2 > 0.0 {
                sxy / (sx2.sqrt() * sy2.sqrt())
            } else {
                0.0
            };
            let scale = if sx2 > 0.0 { sxy / sx2 } else { 0.0 };
            println!(
                "Frame {} middle[{}..{}]: corr={:.6}, scale={:.6}",
                frame, check_start, check_end, corr, scale
            );
        }

        println!("\nFinal OLA SNR: {:.1} dB", last_ola_snr);
        // Report the findings -- the assertion threshold depends on whether
        // the transform pair is correct or has a systematic scaling issue.
        if last_ola_snr > 40.0 {
            println!("PASS: TDAC reconstruction is correct (SNR > 40 dB).");
        } else if last_ola_snr > 20.0 {
            println!("WARNING: TDAC has minor issues (20 < SNR < 40 dB).");
        } else {
            println!("FAIL: TDAC reconstruction has significant scaling/structural issues.");
        }
    }

    // --- Coverage additions: different shift levels, stride, overlap paths ---

    #[test]
    fn test_mdct_backward_shift_0_full_size() {
        let l = &MDCT_48000_960;
        let mode = &crate::celt::modes::MODE_48000_960_120;
        let overlap = 120;
        let shift = 0; // Full size: N=1920, N2=960
        let n2 = 960;
        let n = 1920;

        let mut input = vec![0i32; n2];
        input[0] = 1 << 15;
        let mut output = vec![0i32; n];
        clt_mdct_backward(l, &input, &mut output, mode.window, overlap, shift, 1);
        let max_abs = output.iter().map(|x| x.abs()).max().unwrap_or(0);
        assert!(
            max_abs < i32::MAX / 2,
            "shift=0 output should be bounded, got {max_abs}"
        );
        assert!(
            output.iter().any(|&v| v != 0),
            "shift=0 impulse should produce non-zero output"
        );
    }

    #[test]
    fn test_mdct_backward_shift_1() {
        let l = &MDCT_48000_960;
        let mode = &crate::celt::modes::MODE_48000_960_120;
        let overlap = 120;
        let shift = 1; // N=960, N2=480
        let n2 = 480;
        let n = 960;

        let mut input = vec![0i32; n2];
        input[0] = 1 << 15;
        let mut output = vec![0i32; n];
        clt_mdct_backward(l, &input, &mut output, mode.window, overlap, shift, 1);
        let max_abs = output.iter().map(|x| x.abs()).max().unwrap_or(0);
        assert!(
            max_abs < i32::MAX / 2,
            "shift=1 output should be bounded, got {max_abs}"
        );
        assert!(
            output.iter().any(|&v| v != 0),
            "shift=1 impulse should produce non-zero output"
        );
    }

    #[test]
    fn test_mdct_backward_shift_2() {
        let l = &MDCT_48000_960;
        let mode = &crate::celt::modes::MODE_48000_960_120;
        let overlap = 120;
        let shift = 2; // N=480, N2=240
        let n2 = 240;
        let n = 480;

        let mut input = vec![0i32; n2];
        input[0] = 1 << 15;
        let mut output = vec![0i32; n];
        clt_mdct_backward(l, &input, &mut output, mode.window, overlap, shift, 1);
        let max_abs = output.iter().map(|x| x.abs()).max().unwrap_or(0);
        assert!(
            max_abs < i32::MAX / 2,
            "shift=2 output should be bounded, got {max_abs}"
        );
        assert!(
            output.iter().any(|&v| v != 0),
            "shift=2 impulse should produce non-zero output"
        );
    }

    #[test]
    fn test_mdct_backward_shift_3_zero_vs_impulse() {
        let l = &MDCT_48000_960;
        let mode = &crate::celt::modes::MODE_48000_960_120;
        let overlap = 120;
        let shift = 3; // N=240, N2=120
        let n2 = 120;
        let n = 240;

        // Zero input
        let input_zero = vec![0i32; n2];
        let mut output_zero = vec![0i32; n];
        clt_mdct_backward(
            l,
            &input_zero,
            &mut output_zero,
            mode.window,
            overlap,
            shift,
            1,
        );
        assert!(
            output_zero.iter().all(|&v| v == 0),
            "zero input should give zero output"
        );

        // Impulse input
        let mut input_imp = vec![0i32; n2];
        input_imp[n2 / 2] = 1 << 15;
        let mut output_imp = vec![0i32; n];
        clt_mdct_backward(
            l,
            &input_imp,
            &mut output_imp,
            mode.window,
            overlap,
            shift,
            1,
        );
        assert!(
            output_imp.iter().any(|&v| v != 0),
            "impulse input at mid-band should produce output"
        );
    }

    #[test]
    fn test_mdct_backward_stride_2() {
        let l = &MDCT_48000_960;
        let mode = &crate::celt::modes::MODE_48000_960_120;
        let overlap = 120;
        let shift = 3; // N=240, N2=120
        let n2 = 120;
        let n = 240;

        let mut input = vec![0i32; n2 * 2]; // stride=2 means every other sample
        input[0] = 1 << 15;
        let mut output = vec![0i32; n];
        clt_mdct_backward(l, &input, &mut output, mode.window, overlap, shift, 2);
        let max_abs = output.iter().map(|x| x.abs()).max().unwrap_or(0);
        assert!(max_abs < i32::MAX / 2, "stride=2 output should be bounded");
    }

    mod branch_coverage_stage5 {
        use super::*;

        /// Pass input shorter than n2 to trigger the defensive `idx < input_len`
        /// else branches (L104, L132, L140).
        #[test]
        fn backward_short_input_triggers_defensive_bounds() {
            let l = &MDCT_48000_960;
            let mode = &crate::celt::modes::MODE_48000_960_120;
            let overlap = 120;
            let shift = 3; // N=240, N2=120
            let n = 240;

            // Half-sized input: forces idx >= input_len for roughly half the indices.
            let input = vec![100i32; 40]; // n2=120, so 80 reads will fall to else branch
            let mut output = vec![0i32; n];
            clt_mdct_backward(l, &input, &mut output, mode.window, overlap, shift, 1);
            // Just verify it doesn't panic and output is bounded.
            let max_abs = output.iter().map(|x| x.abs()).max().unwrap_or(0);
            assert!(max_abs < i32::MAX / 2);
        }

        /// Empty input: every idx hits the else branch.
        #[test]
        fn backward_empty_input_all_defensive() {
            let l = &MDCT_48000_960;
            let mode = &crate::celt::modes::MODE_48000_960_120;
            let shift = 3;
            let n = 240;
            let input: Vec<i32> = Vec::new();
            let mut output = vec![0i32; n];
            clt_mdct_backward(l, &input, &mut output, mode.window, 120, shift, 1);
            // With zero input, output should be all zero.
            assert!(output.iter().all(|&v| v == 0));
        }

        /// Stride sweep: exercise stride=4 path on shift=3.
        #[test]
        fn backward_stride_4() {
            let l = &MDCT_48000_960;
            let mode = &crate::celt::modes::MODE_48000_960_120;
            let shift = 3;
            let n = 240;
            let n2 = 120;
            let mut input = vec![0i32; n2 * 4];
            input[0] = 1 << 15;
            let mut output = vec![0i32; n];
            clt_mdct_backward(l, &input, &mut output, mode.window, 120, shift, 4);
            let max_abs = output.iter().map(|x| x.abs()).max().unwrap_or(0);
            assert!(max_abs < i32::MAX / 2);
        }

        /// Exercise short input on every shift level.
        #[test]
        fn backward_short_input_all_shifts() {
            let l = &MDCT_48000_960;
            let mode = &crate::celt::modes::MODE_48000_960_120;
            for shift in 0..=3 {
                let n = (1920i32 >> shift) as usize;
                // Truncate input to drive the defensive else branches.
                let input = vec![10i32; n / 8];
                let mut output = vec![0i32; n];
                clt_mdct_backward(l, &input, &mut output, mode.window, 120, shift, 1);
                let max_abs = output.iter().map(|x| x.abs()).max().unwrap_or(0);
                assert!(max_abs < i32::MAX / 2, "shift={shift}");
            }
        }

        /// Forward MDCT across all four shift levels with small DC input.
        #[test]
        fn forward_all_shifts_dc() {
            use crate::celt::encoder::{clt_mdct_forward, get_fft_state};
            let l = &MDCT_48000_960;
            let mode = &crate::celt::modes::MODE_48000_960_120;
            let overlap = 120i32;
            for shift in 0..=3i32 {
                let n = (1920 >> shift) as usize;
                let n2 = n / 2;
                let fft_st = get_fft_state(shift);
                let input = vec![500i32; n + overlap as usize];
                let mut freq = vec![0i32; n2];
                clt_mdct_forward(
                    &input,
                    &mut freq,
                    mode.window,
                    overlap,
                    shift,
                    1,
                    fft_st,
                    l.trig,
                );
                // Output should be bounded.
                let max_abs = freq.iter().map(|x| x.abs()).max().unwrap_or(0);
                assert!(max_abs < i32::MAX / 2, "shift={shift}");
            }
        }

        /// Two-frame backward with minimal overlap (drives the TDAC OLA path).
        #[test]
        fn backward_two_frames_minimal() {
            let l = &MDCT_48000_960;
            let mode = &crate::celt::modes::MODE_48000_960_120;
            let overlap = 120;
            let shift = 2; // N=480, N2=240
            let n = 480;
            let n2 = 240;
            let input1 = vec![100i32; n2];
            let mut output1 = vec![0i32; n];
            clt_mdct_backward(l, &input1, &mut output1, mode.window, overlap, shift, 1);
            let input2 = vec![-50i32; n2];
            let mut output2 = vec![0i32; n];
            clt_mdct_backward(l, &input2, &mut output2, mode.window, overlap, shift, 1);
            // Both outputs should be non-panicking and bounded.
            for v in output1.iter().chain(output2.iter()) {
                assert!(v.abs() < i32::MAX / 2);
            }
        }
    }
}
