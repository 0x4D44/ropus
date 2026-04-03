//! CELT mode configuration — codec parameter tables and static mode instances.
//!
//! Corresponds to `modes.h` / `modes.c` in the C reference (static mode path).
//! Also includes `resampling_factor()` and `init_caps()` from `celt.c`.
//!
//! Standard Opus only uses pre-computed static modes (no dynamic allocation).
//! A single static mode at 48 kHz / 960 samples serves all four frame sizes
//! (120, 240, 480, 960) via the LM mechanism.

// ===========================================================================
// Constants (from modes.h)
// ===========================================================================

/// Maximum pitch period in samples.
pub const MAX_PERIOD: i32 = 1024;

/// Decoder pitch buffer size (2 × MAX_PERIOD).
pub const DEC_PITCH_BUF_SIZE: i32 = 2048;

/// Number of bit allocation vectors (rows in the allocation matrix).
pub const BITALLOC_SIZE: usize = 11;

/// Number of energy bands in the standard 48 kHz mode.
pub const NB_EBANDS: usize = 21;

/// Number of entries in the band edge table (NB_EBANDS + 1).
pub const NB_EBAND_ENTRIES: usize = 22;

// ===========================================================================
// Static tables — band edges (from modes.c)
// ===========================================================================

/// Standard band edge table for 5 ms short blocks.
///
/// 22 entries defining 21 bands. Each value is an MDCT bin index for one
/// short block (120 bins at 48 kHz). Bandwidth increases from 200 Hz (1 bin)
/// at low frequencies to 4.4 kHz (22 bins) at the top, roughly following
/// the Bark frequency scale.
///
/// ```text
/// Band:  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20
/// Width: 1  1  1  1  1  1  1  1  2  2  2  2  4  4  4  6  6  8  12 18 22
/// ```
pub static EBAND5MS: [i16; NB_EBAND_ENTRIES] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100,
];

// ===========================================================================
// Static tables — bit allocation matrix (from modes.c)
// ===========================================================================

/// Bit allocation matrix: 11 rows × 21 bands.
///
/// Values are in units of 1/32 bit per MDCT coefficient. Row 0 is silence
/// (all zeros), row 10 is maximum (all 200 = 6.25 bits/sample). Intermediate
/// rows are interpolated by the rate allocator based on actual bit budget.
pub static BAND_ALLOCATION: [u8; BITALLOC_SIZE * NB_EBANDS] = [
    // Row 0: silence
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // Row 1: very low bitrate
    90, 80, 75, 69, 63, 56, 49, 40, 34, 29, 20, 18, 10, 0, 0, 0, 0, 0, 0, 0, 0, // Row 2
    110, 100, 90, 84, 78, 71, 65, 58, 51, 45, 39, 32, 26, 20, 12, 0, 0, 0, 0, 0, 0,
    // Row 3
    118, 110, 103, 93, 86, 80, 75, 70, 65, 59, 53, 47, 40, 31, 23, 15, 4, 0, 0, 0, 0,
    // Row 4
    126, 119, 112, 104, 95, 89, 83, 78, 72, 66, 60, 54, 47, 39, 32, 25, 17, 12, 1, 0, 0,
    // Row 5
    134, 127, 120, 114, 103, 97, 91, 85, 78, 72, 66, 60, 54, 47, 41, 35, 29, 23, 16, 10, 1,
    // Row 6
    144, 137, 130, 124, 113, 107, 101, 95, 88, 82, 76, 70, 64, 57, 51, 45, 39, 33, 26, 15, 1,
    // Row 7
    152, 145, 138, 132, 123, 117, 111, 105, 98, 92, 86, 80, 74, 67, 61, 55, 49, 43, 36, 20, 1,
    // Row 8
    162, 155, 148, 142, 133, 127, 121, 115, 108, 102, 96, 90, 84, 77, 71, 65, 59, 53, 46, 30, 1,
    // Row 9
    172, 165, 158, 152, 143, 137, 131, 125, 118, 112, 106, 100, 94, 87, 81, 75, 69, 63, 56, 45, 20,
    // Row 10: maximum
    200, 200, 200, 200, 200, 200, 200, 200, 198, 193, 188, 183, 178, 173, 168, 163, 158, 153, 148,
    129, 104,
];

// ===========================================================================
// Static tables — MDCT window (from static_modes_fixed.h)
// ===========================================================================

/// MDCT analysis/synthesis window, 120 samples, Q15 fixed-point.
///
/// Window function: `w(i) = sin(π/2 · sin²(π/2 · (i+0.5)/120))`.
/// Quantized as `floor(0.5 + 32768.0 * w(i))`, clamped to 32767.
/// Symmetric: rising edge uses `w[i]`, falling edge uses `w[overlap-1-i]`.
static WINDOW120: [i16; 120] = [
    2, 20, 55, 108, 178, 266, 372, 494, 635, 792, 966, 1157, 1365, 1590, 1831, 2089, 2362, 2651,
    2956, 3276, 3611, 3961, 4325, 4703, 5094, 5499, 5916, 6346, 6788, 7241, 7705, 8179, 8663, 9156,
    9657, 10167, 10684, 11207, 11736, 12271, 12810, 13353, 13899, 14447, 14997, 15547, 16098,
    16648, 17197, 17744, 18287, 18827, 19363, 19893, 20418, 20936, 21447, 21950, 22445, 22931,
    23407, 23874, 24330, 24774, 25208, 25629, 26039, 26435, 26819, 27190, 27548, 27893, 28224,
    28541, 28845, 29135, 29411, 29674, 29924, 30160, 30384, 30594, 30792, 30977, 31151, 31313,
    31463, 31602, 31731, 31849, 31958, 32057, 32148, 32229, 32303, 32370, 32429, 32481, 32528,
    32568, 32604, 32634, 32661, 32683, 32701, 32717, 32729, 32740, 32748, 32754, 32758, 32762,
    32764, 32766, 32767, 32767, 32767, 32767, 32767, 32767,
];

// ===========================================================================
// Static tables — logN (from static_modes_fixed.h)
// ===========================================================================

/// log2(band_width) per band in Q3 (BITRES=3), for the standard 48 kHz mode.
///
/// Used by the rate allocator to scale bit budgets by band size.
/// `logN[i] = log2_frac(eBands[i+1] - eBands[i], 3)`.
static LOG_N400: [i16; NB_EBANDS] = [
    0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 16, 16, 16, 21, 21, 24, 29, 34, 36,
];

// ===========================================================================
// Static tables — pulse cache (from static_modes_fixed.h)
// ===========================================================================

/// Pulse cache index table.
///
/// Maps `(LM, band)` to an offset into `CACHE_BITS50`. Indexed as
/// `cache_index[(LM+1) * nbEBands + band]`. Value -1 means the band
/// has no pulses at that LM (band is too narrow).
///
/// 105 entries = 5 LM-groups × 21 bands.
static CACHE_INDEX50: [i16; 105] = [
    -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 41, 41, 41, 82, 82, 123, 164, 200, 222, 0, 0, 0, 0,
    0, 0, 0, 0, 41, 41, 41, 41, 123, 123, 123, 164, 164, 240, 266, 283, 295, 41, 41, 41, 41, 41,
    41, 41, 41, 123, 123, 123, 123, 240, 240, 240, 266, 266, 305, 318, 328, 336, 123, 123, 123,
    123, 123, 123, 123, 123, 240, 240, 240, 240, 305, 305, 305, 318, 318, 343, 351, 358, 364, 240,
    240, 240, 240, 240, 240, 240, 240, 305, 305, 305, 305, 343, 343, 343, 351, 351, 370, 376, 382,
    387,
];

/// Pulse cache bit cost table.
///
/// At each offset from `CACHE_INDEX50`, `bits[0]` is the maximum pseudo-pulse
/// count for that band size, and `bits[k]` for `k = 1..bits[0]` gives the
/// bit cost (in Q(BITRES) minus 1) for `k` pseudo-pulses.
///
/// 392 entries total.
static CACHE_BITS50: [u8; 392] = [
    40, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 40, 15, 23, 28, 31, 34, 36, 38, 39, 41, 42, 43, 44, 45, 46, 47,
    47, 49, 50, 51, 52, 53, 54, 55, 55, 57, 58, 59, 60, 61, 62, 63, 63, 65, 66, 67, 68, 69, 70, 71,
    71, 40, 20, 33, 41, 48, 53, 57, 61, 64, 66, 69, 71, 73, 75, 76, 78, 80, 82, 85, 87, 89, 91, 92,
    94, 96, 98, 101, 103, 105, 107, 108, 110, 112, 114, 117, 119, 121, 123, 124, 126, 128, 40, 23,
    39, 51, 60, 67, 73, 79, 83, 87, 91, 94, 97, 100, 102, 105, 107, 111, 115, 118, 121, 124, 126,
    129, 131, 135, 139, 142, 145, 148, 150, 153, 155, 159, 163, 166, 169, 172, 174, 177, 179, 35,
    28, 49, 65, 78, 89, 99, 107, 114, 120, 126, 132, 136, 141, 145, 149, 153, 159, 165, 171, 176,
    180, 185, 189, 192, 199, 205, 211, 216, 220, 225, 229, 232, 239, 245, 251, 21, 33, 58, 79, 97,
    112, 125, 137, 148, 157, 166, 174, 182, 189, 195, 201, 207, 217, 227, 235, 243, 251, 17, 35,
    63, 86, 106, 123, 139, 152, 165, 177, 187, 197, 206, 214, 222, 230, 237, 250, 25, 31, 55, 75,
    91, 105, 117, 128, 138, 146, 154, 161, 168, 174, 180, 185, 190, 200, 208, 215, 222, 229, 235,
    240, 245, 255, 16, 36, 65, 89, 110, 128, 144, 159, 173, 185, 196, 207, 217, 226, 234, 242, 250,
    11, 41, 74, 103, 128, 151, 172, 191, 209, 225, 241, 255, 9, 43, 79, 110, 138, 163, 186, 207,
    227, 246, 12, 39, 71, 99, 123, 144, 164, 182, 198, 214, 228, 241, 253, 9, 44, 81, 113, 142,
    168, 192, 214, 235, 255, 7, 49, 90, 127, 160, 191, 220, 247, 6, 51, 95, 134, 170, 203, 234, 7,
    47, 87, 123, 155, 184, 212, 237, 6, 52, 97, 137, 174, 208, 240, 5, 57, 106, 151, 192, 231, 5,
    59, 111, 158, 202, 243, 5, 55, 103, 147, 187, 224, 5, 60, 113, 161, 206, 248, 4, 65, 122, 175,
    224, 4, 67, 127, 182, 234,
];

/// Pulse cache capacity table.
///
/// Per-band maximum reliable bit capacity. Indexed as
/// `caps[nbEBands * (2*LM + C - 1) + band]` where C is the channel count.
///
/// 168 entries = 8 configurations × 21 bands (LM=0..3, C=1..2).
static CACHE_CAPS50: [u8; 168] = [
    224, 224, 224, 224, 224, 224, 224, 224, 160, 160, 160, 160, 185, 185, 185, 178, 178, 168, 134,
    61, 37, 224, 224, 224, 224, 224, 224, 224, 224, 240, 240, 240, 240, 207, 207, 207, 198, 198,
    183, 144, 66, 40, 160, 160, 160, 160, 160, 160, 160, 160, 185, 185, 185, 185, 193, 193, 193,
    183, 183, 172, 138, 64, 38, 240, 240, 240, 240, 240, 240, 240, 240, 207, 207, 207, 207, 204,
    204, 204, 193, 193, 180, 143, 66, 40, 185, 185, 185, 185, 185, 185, 185, 185, 193, 193, 193,
    193, 193, 193, 193, 183, 183, 172, 138, 65, 39, 207, 207, 207, 207, 207, 207, 207, 207, 204,
    204, 204, 204, 201, 201, 201, 188, 188, 176, 141, 66, 40, 193, 193, 193, 193, 193, 193, 193,
    193, 193, 193, 193, 193, 194, 194, 194, 184, 184, 173, 139, 65, 39, 204, 204, 204, 204, 204,
    204, 204, 204, 201, 201, 201, 201, 198, 198, 198, 187, 187, 175, 140, 66, 40,
];

// ===========================================================================
// PulseCache struct
// ===========================================================================

/// Pulse cache for PVQ bit allocation lookups.
///
/// Pre-computed bit costs for encoding each band at each pulse count,
/// for all `(LM, band)` combinations.
pub struct PulseCache {
    /// Total number of entries in the `bits` table.
    pub size: i32,
    /// Index table mapping (LM, band) to offset in `bits`.
    /// Length: `nbEBands * (maxLM + 2)`.
    pub index: &'static [i16],
    /// Cached bit costs per pulse count.
    pub bits: &'static [u8],
    /// Maximum reliable bit capacity per band.
    /// Length: `(maxLM + 1) * 2 * nbEBands`.
    pub caps: &'static [u8],
}

// ===========================================================================
// CELTMode struct
// ===========================================================================

/// CELT codec mode configuration.
///
/// Holds all static tables and parameters needed by the CELT encoder/decoder.
/// The C reference uses `const CELTMode *m` throughout; in Rust we use `&CELTMode`.
///
/// A single mode serves multiple frame sizes via the LM mechanism:
/// the actual frame size is `shortMdctSize * nbShortMdcts >> (maxLM - LM)`.
pub struct CELTMode {
    /// Sample rate in Hz.
    pub fs: i32,
    /// Overlap size for MDCT windowing (in samples).
    pub overlap: i32,
    /// Total number of energy bands.
    pub nb_ebands: i32,
    /// Effective bands (those below Nyquist for short MDCT).
    pub eff_ebands: i32,
    /// Pre-emphasis filter coefficients [a1(Q15), a2(Q15), 1/gain(Q12), gain(Q13)].
    pub preemph: [i32; 4],
    /// Band boundary table: `ebands[i]` gives the start bin of band `i`.
    /// Length: `nb_ebands + 1`.
    pub ebands: &'static [i16],
    /// Maximum LM value (log2 of max number of short blocks per frame).
    pub max_lm: i32,
    /// Number of short MDCTs in a frame at maximum LM: `1 << max_lm`.
    pub nb_short_mdcts: i32,
    /// Size of one short MDCT in samples (frequency bins per short block).
    pub short_mdct_size: i32,
    /// Number of rows in the allocation matrix (always BITALLOC_SIZE = 11).
    pub nb_alloc_vectors: i32,
    /// Bit allocation matrix, flat `[nb_alloc_vectors × nb_ebands]`.
    pub alloc_vectors: &'static [u8],
    /// log2(band_width) per band in Q3 (BITRES=3). Length: `nb_ebands`.
    pub log_n: &'static [i16],
    /// MDCT window coefficients (Q15). Length: `overlap`.
    pub window: &'static [i16],
    /// Pulse cache for bit allocation.
    pub cache: PulseCache,
}

// ===========================================================================
// Static mode instance — 48 kHz (from static_modes_fixed.h)
// ===========================================================================

/// Static mode for 48 kHz, frame_size=960, overlap=120.
///
/// Serves frame sizes 960 (20ms), 480 (10ms), 240 (5ms), and 120 (2.5ms)
/// via LM values 3, 2, 1, 0 respectively.
///
/// Pre-emphasis coefficients (48 kHz): A(z) = 1 - 0.85·z⁻¹
/// - preemph[0] = QCONST16(0.8500061035, 15) = 27853
/// - preemph[1] = QCONST16(0.0, 15) = 0
/// - preemph[2] = QCONST16(1.0, SIG_SHIFT=12) = 4096
/// - preemph[3] = QCONST16(1.0, 13) = 8192
pub static MODE_48000_960_120: CELTMode = CELTMode {
    fs: 48000,
    overlap: 120,
    nb_ebands: 21,
    eff_ebands: 21,
    preemph: [27853, 0, 4096, 8192],
    ebands: &EBAND5MS,
    max_lm: 3,
    nb_short_mdcts: 8,
    short_mdct_size: 120,
    nb_alloc_vectors: 11,
    alloc_vectors: &BAND_ALLOCATION,
    log_n: &LOG_N400,
    window: &WINDOW120,
    cache: PulseCache {
        size: 392,
        index: &CACHE_INDEX50,
        bits: &CACHE_BITS50,
        caps: &CACHE_CAPS50,
    },
};

/// List of all compiled-in static modes.
static STATIC_MODE_LIST: [&CELTMode; TOTAL_MODES] = [&MODE_48000_960_120];

/// Number of static modes available.
const TOTAL_MODES: usize = 1;

// ===========================================================================
// Mode creation — static lookup (from modes.c)
// ===========================================================================

/// Look up a pre-computed static mode for the given sample rate and frame size.
///
/// Matches the C `opus_custom_mode_create()` static path. Returns `None` if
/// no matching mode is found (equivalent to `OPUS_BAD_ARG` in the C reference).
///
/// The `j` loop allows one static mode to serve four frame sizes: for 48 kHz
/// with `shortMdctSize=120` and `nbShortMdcts=8`:
/// - j=0: frame_size=960 (20 ms, LM=3)
/// - j=1: frame_size=480 (10 ms, LM=2)
/// - j=2: frame_size=240 (5 ms, LM=1)
/// - j=3: frame_size=120 (2.5 ms, LM=0)
pub fn mode_create(fs: i32, frame_size: i32) -> Option<&'static CELTMode> {
    for mode in STATIC_MODE_LIST.iter() {
        for j in 0..4i32 {
            if fs == mode.fs && (frame_size << j) == mode.short_mdct_size * mode.nb_short_mdcts {
                return Some(mode);
            }
        }
    }
    None
}

// ===========================================================================
// Resampling factor (from celt.c)
// ===========================================================================

/// Return the integer downsampling factor from the mode's internal rate.
///
/// Matches the C `resampling_factor()` from `celt.c`.
///
/// | Rate   | Factor |
/// |--------|--------|
/// | 48000  | 1      |
/// | 24000  | 2      |
/// | 16000  | 3      |
/// | 12000  | 4      |
/// | 8000   | 6      |
pub fn resampling_factor(rate: i32) -> i32 {
    match rate {
        48000 => 1,
        24000 => 2,
        16000 => 3,
        12000 => 4,
        8000 => 6,
        _ => 0,
    }
}

// ===========================================================================
// Init caps (from celt.c)
// ===========================================================================

/// Compute per-band bit capacity limits from the mode's cache.
///
/// Matches the C `init_caps()` from `celt.c`.
///
/// ```text
/// N = (eBands[i+1] - eBands[i]) << LM
/// cap[i] = (cache.caps[nbEBands*(2*LM+C-1) + i] + 64) * C * N >> 2
/// ```
///
/// # Parameters
/// - `m`: Mode configuration
/// - `cap`: Output buffer, length >= `m.nb_ebands`
/// - `lm`: Log2 of the number of short blocks (0..3)
/// - `c`: Number of channels (1 or 2)
pub fn init_caps(m: &CELTMode, cap: &mut [i32], lm: i32, c: i32) {
    for i in 0..m.nb_ebands as usize {
        // Number of MDCT bins in this band at the given LM
        let n = ((m.ebands[i + 1] - m.ebands[i]) as i32) << lm;
        // Capacity from cached caps table, scaled by channels and band size
        cap[i] =
            ((m.cache.caps[(m.nb_ebands * (2 * lm + c - 1)) as usize + i] as i32) + 64) * c * n
                >> 2;
    }
}

// ===========================================================================
// Utility functions (from celt.h)
// ===========================================================================

/// Convert a bit count to a bitrate in bits/second.
///
/// Matches the C `bits_to_bitrate()` inline from `celt.h`.
#[inline(always)]
pub fn bits_to_bitrate(bits: i32, fs: i32, frame_size: i32) -> i32 {
    bits * (6 * fs / frame_size) / 6
}

/// Convert a bitrate in bits/second to a bit count per frame.
///
/// Matches the C `bitrate_to_bits()` inline from `celt.h`.
#[inline(always)]
pub fn bitrate_to_bits(bitrate: i32, fs: i32, frame_size: i32) -> i32 {
    bitrate * 6 / (6 * fs / frame_size)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Table dimensions ---

    #[test]
    fn table_sizes_are_consistent() {
        assert_eq!(EBAND5MS.len(), NB_EBAND_ENTRIES);
        assert_eq!(BAND_ALLOCATION.len(), BITALLOC_SIZE * NB_EBANDS);
        assert_eq!(WINDOW120.len(), 120);
        assert_eq!(LOG_N400.len(), NB_EBANDS);
        // cache_index: 5 LM-groups × 21 bands
        assert_eq!(CACHE_INDEX50.len(), 5 * NB_EBANDS);
        // cache_caps: 8 configs × 21 bands (LM=0..3, C=1..2)
        assert_eq!(CACHE_CAPS50.len(), 8 * NB_EBANDS);
        assert_eq!(CACHE_BITS50.len(), 392);
    }

    // --- Static mode consistency ---

    #[test]
    fn static_mode_field_consistency() {
        let m = &MODE_48000_960_120;
        assert_eq!(m.fs, 48000);
        assert_eq!(m.overlap, 120);
        assert_eq!(m.nb_ebands, NB_EBANDS as i32);
        assert_eq!(m.eff_ebands, NB_EBANDS as i32);
        assert_eq!(m.max_lm, 3);
        assert_eq!(m.nb_short_mdcts, 8);
        assert_eq!(m.short_mdct_size, 120);
        assert_eq!(m.nb_alloc_vectors, BITALLOC_SIZE as i32);
        // Frame size at max LM
        assert_eq!(m.short_mdct_size * m.nb_short_mdcts, 960);
        // Overlap is shortMdctSize rounded down to multiple of 4
        assert_eq!(m.overlap, (m.short_mdct_size >> 2) << 2);
        // nbShortMdcts = 1 << maxLM
        assert_eq!(m.nb_short_mdcts, 1 << m.max_lm);
    }

    #[test]
    fn static_mode_preemph_values() {
        let m = &MODE_48000_960_120;
        // A(z) = 1 - 0.85z^-1 at 48 kHz
        // preemph[0] = QCONST16(0.8500061035, 15)
        assert_eq!(m.preemph[0], 27853);
        // preemph[1] = QCONST16(0.0, 15)
        assert_eq!(m.preemph[1], 0);
        // preemph[2] = QCONST16(1.0, SIG_SHIFT=12) -- inverse of deemphasis gain
        assert_eq!(m.preemph[2], 4096);
        // preemph[3] = QCONST16(1.0, 13) -- deemphasis gain
        assert_eq!(m.preemph[3], 8192);
    }

    // --- Band edge table ---

    #[test]
    fn eband5ms_starts_at_zero() {
        assert_eq!(EBAND5MS[0], 0);
    }

    #[test]
    fn eband5ms_is_monotonically_increasing() {
        for i in 1..EBAND5MS.len() {
            assert!(
                EBAND5MS[i] > EBAND5MS[i - 1],
                "eBands not increasing at index {}: {} <= {}",
                i,
                EBAND5MS[i],
                EBAND5MS[i - 1]
            );
        }
    }

    #[test]
    fn eband5ms_last_entry_is_100() {
        // 100 bins × 200 Hz/bin = 20 kHz Nyquist for 48 kHz mode
        assert_eq!(EBAND5MS[NB_EBANDS], 100);
    }

    #[test]
    fn eband5ms_all_within_short_mdct() {
        let m = &MODE_48000_960_120;
        // All band edges must be <= shortMdctSize for effEBands == nbEBands
        assert!(EBAND5MS[m.nb_ebands as usize] <= m.short_mdct_size as i16);
    }

    #[test]
    fn eband5ms_widths_satisfy_constraints() {
        // From C reference: each band <= last band width,
        // and each band <= 2× the previous band width.
        let last_width = EBAND5MS[NB_EBANDS] - EBAND5MS[NB_EBANDS - 1];
        for i in 1..NB_EBANDS {
            let width = EBAND5MS[i + 1] - EBAND5MS[i];
            let prev_width = EBAND5MS[i] - EBAND5MS[i - 1];
            assert!(
                width <= last_width,
                "band {} width {} > last band width {}",
                i,
                width,
                last_width
            );
            assert!(
                width <= 2 * prev_width,
                "band {} width {} > 2× previous width {}",
                i,
                width,
                prev_width
            );
        }
    }

    // --- Window ---

    #[test]
    fn window_values_are_non_negative() {
        for &w in WINDOW120.iter() {
            assert!(w >= 0, "window value {} is negative", w);
        }
    }

    #[test]
    fn window_is_monotonically_non_decreasing() {
        for i in 1..WINDOW120.len() {
            assert!(
                WINDOW120[i] >= WINDOW120[i - 1],
                "window not non-decreasing at {}: {} < {}",
                i,
                WINDOW120[i],
                WINDOW120[i - 1]
            );
        }
    }

    #[test]
    fn window_ends_at_q15_max() {
        // Last values should be Q15ONE = 32767
        assert_eq!(WINDOW120[119], 32767);
        assert_eq!(WINDOW120[118], 32767);
    }

    #[test]
    fn window_starts_small() {
        // First value should be near zero
        assert_eq!(WINDOW120[0], 2);
    }

    // --- Band allocation matrix ---

    #[test]
    fn band_allocation_row0_is_silence() {
        for j in 0..NB_EBANDS {
            assert_eq!(BAND_ALLOCATION[j], 0);
        }
    }

    #[test]
    fn band_allocation_row10_is_maximum() {
        let row10 = &BAND_ALLOCATION[10 * NB_EBANDS..11 * NB_EBANDS];
        // First 8 bands should be 200 (maximum)
        for &v in &row10[..8] {
            assert_eq!(v, 200);
        }
        // All values should be > 0
        for &v in row10.iter() {
            assert!(v > 0, "row 10 has zero allocation");
        }
    }

    #[test]
    fn band_allocation_values_in_range() {
        for &v in BAND_ALLOCATION.iter() {
            assert!(v <= 200, "allocation value {} exceeds maximum 200", v);
        }
    }

    // --- logN ---

    #[test]
    fn log_n_matches_band_widths() {
        // Verify a few known values:
        // Band 0: width=1, log2(1) in Q3 = 0
        assert_eq!(LOG_N400[0], 0);
        // Band 8: width=2, log2(2) in Q3 = 8
        assert_eq!(LOG_N400[8], 8);
        // Band 12: width=4, log2(4) in Q3 = 16
        assert_eq!(LOG_N400[12], 16);
        // Band 20: width=22, log2(22) in Q3 ≈ 36
        assert_eq!(LOG_N400[20], 36);
    }

    // --- mode_create ---

    #[test]
    fn mode_create_48k_960() {
        let m = mode_create(48000, 960).expect("should match 48kHz/960");
        assert_eq!(m.fs, 48000);
        assert_eq!(m.short_mdct_size * m.nb_short_mdcts, 960);
    }

    #[test]
    fn mode_create_48k_480() {
        let m = mode_create(48000, 480).expect("should match 48kHz/480");
        assert_eq!(m.fs, 48000);
        // 480 << 1 = 960 = shortMdctSize * nbShortMdcts
    }

    #[test]
    fn mode_create_48k_240() {
        let m = mode_create(48000, 240).expect("should match 48kHz/240");
        assert_eq!(m.fs, 48000);
        // 240 << 2 = 960
    }

    #[test]
    fn mode_create_48k_120() {
        let m = mode_create(48000, 120).expect("should match 48kHz/120");
        assert_eq!(m.fs, 48000);
        // 120 << 3 = 960
    }

    #[test]
    fn mode_create_returns_same_static_instance() {
        let m1 = mode_create(48000, 960).unwrap();
        let m2 = mode_create(48000, 480).unwrap();
        let m3 = mode_create(48000, 240).unwrap();
        let m4 = mode_create(48000, 120).unwrap();
        // All should point to the same static mode
        assert!(std::ptr::eq(m1, m2));
        assert!(std::ptr::eq(m2, m3));
        assert!(std::ptr::eq(m3, m4));
    }

    #[test]
    fn mode_create_invalid_rate() {
        assert!(mode_create(44100, 960).is_none());
        assert!(mode_create(22050, 480).is_none());
        assert!(mode_create(0, 960).is_none());
    }

    #[test]
    fn mode_create_invalid_frame_size() {
        assert!(mode_create(48000, 100).is_none());
        assert!(mode_create(48000, 0).is_none());
        assert!(mode_create(48000, 1920).is_none());
    }

    // --- resampling_factor ---

    #[test]
    fn resampling_factor_all_rates() {
        assert_eq!(resampling_factor(48000), 1);
        assert_eq!(resampling_factor(24000), 2);
        assert_eq!(resampling_factor(16000), 3);
        assert_eq!(resampling_factor(12000), 4);
        assert_eq!(resampling_factor(8000), 6);
    }

    #[test]
    fn resampling_factor_invalid_returns_zero() {
        // Invalid rates return 0, matching the C reference.
        assert_eq!(resampling_factor(44100), 0);
    }

    // --- init_caps ---

    #[test]
    fn init_caps_mono_lm0() {
        let m = &MODE_48000_960_120;
        let mut cap = vec![0i32; m.nb_ebands as usize];
        init_caps(m, &mut cap, 0, 1);

        // All caps should be positive
        for (i, &c) in cap.iter().enumerate() {
            assert!(c >= 0, "cap[{}] = {} is negative", i, c);
        }

        // Verify formula for band 0: N = (1-0) << 0 = 1
        // cap[0] = (caps[21*(2*0+1-1) + 0] + 64) * 1 * 1 >> 2
        //        = (caps[0] + 64) * 1 >> 2
        //        = (224 + 64) * 1 >> 2 = 288 >> 2 = 72
        assert_eq!(cap[0], 72);
    }

    #[test]
    fn init_caps_stereo_lm3() {
        let m = &MODE_48000_960_120;
        let mut cap = vec![0i32; m.nb_ebands as usize];
        init_caps(m, &mut cap, 3, 2);

        // All caps should be positive
        for (i, &c) in cap.iter().enumerate() {
            assert!(c >= 0, "cap[{}] = {} is negative", i, c);
        }

        // Stereo caps should generally be larger than mono caps
        let mut cap_mono = vec![0i32; m.nb_ebands as usize];
        init_caps(m, &mut cap_mono, 3, 1);
        // C=2 multiplier: each cap is scaled by C=2 and also uses a different
        // row in the caps table, so stereo caps >= mono caps for most bands
        for i in 0..m.nb_ebands as usize {
            assert!(
                cap[i] >= cap_mono[i],
                "stereo cap[{}]={} < mono cap[{}]={}",
                i,
                cap[i],
                i,
                cap_mono[i]
            );
        }
    }

    #[test]
    fn init_caps_all_lm_values() {
        let m = &MODE_48000_960_120;
        for lm in 0..=3 {
            for c in 1..=2 {
                let mut cap = vec![0i32; m.nb_ebands as usize];
                init_caps(m, &mut cap, lm, c);
                // Sanity: no negative caps
                for (i, &v) in cap.iter().enumerate() {
                    assert!(v >= 0, "cap[{}] = {} negative at LM={}, C={}", i, v, lm, c);
                }
            }
        }
    }

    // --- bits_to_bitrate / bitrate_to_bits ---

    #[test]
    fn bits_bitrate_roundtrip() {
        // At 48 kHz, 960 samples = 20 ms frame
        // 1000 bits/frame at 48kHz/960 = 50000 bps
        let bits = 1000;
        let br = bits_to_bitrate(bits, 48000, 960);
        assert_eq!(br, 50000);

        let bits_back = bitrate_to_bits(br, 48000, 960);
        assert_eq!(bits_back, bits);
    }

    #[test]
    fn bits_to_bitrate_various_frame_sizes() {
        // 480 bits at 48kHz/480 samples (10ms) = 48000 bps
        assert_eq!(bits_to_bitrate(480, 48000, 480), 48000);
        // 240 bits at 48kHz/240 samples (5ms) = 48000 bps
        assert_eq!(bits_to_bitrate(240, 48000, 240), 48000);
    }

    // --- Cache table structure ---

    #[test]
    fn cache_index_valid_offsets() {
        // All non-negative index values should be valid offsets into cache_bits
        for &idx in CACHE_INDEX50.iter() {
            if idx >= 0 {
                assert!(
                    (idx as usize) < CACHE_BITS50.len(),
                    "cache index {} out of bounds (bits len = {})",
                    idx,
                    CACHE_BITS50.len()
                );
            }
        }
    }

    #[test]
    fn cache_first_group_bands_0_to_7_are_minus_one() {
        // At LM+1=0 (the first group), bands 0-7 have -1 (no pulses for 1-bin bands)
        for j in 0..8 {
            assert_eq!(CACHE_INDEX50[j], -1, "expected -1 for band {} at LM+1=0", j);
        }
    }

    #[test]
    fn cache_bits_max_pulses_reasonable() {
        // At each valid cache offset, bits[0] is the max pseudo-pulse count.
        // It should be <= 40 (MAX_PSEUDO).
        let mut checked = std::collections::HashSet::new();
        for &idx in CACHE_INDEX50.iter() {
            if idx >= 0 && checked.insert(idx) {
                let max_pulses = CACHE_BITS50[idx as usize];
                assert!(
                    max_pulses <= 40,
                    "max pulses {} at offset {} exceeds MAX_PSEUDO=40",
                    max_pulses,
                    idx
                );
            }
        }
    }
}
