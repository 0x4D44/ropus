//! Tonality / speech-vs-music analyzer (port of `reference/src/analysis.c`).
//!
//! # Bit-exactness contract
//!
//! This module is validated against reference/src/analysis.c under
//! scalar-C harness config (-ffp-contract=off on the C side, no
//! fast-math on either side). Do not introduce f32::mul_add, do not
//! reorder accumulator chains, do not horizontally-sum into vectors.
//! Tier-1 encoder byte-exactness depends on this.
//!
//! # Layout
//!
//! The tonality analyzer runs at a 24 kHz internal rate and produces a
//! frame-by-frame [`AnalysisInfo`] describing tonality, noisiness,
//! bandwidth, music-probability and a few related metrics. The encoder
//! later consults these to pick mode, bitrate, and SILK bandwidth.
//!
//! Pipeline per frame:
//! 1. [`downmix_and_resample`] collapses multichannel PCM to mono at
//!    24 kHz and accumulates a high-pass energy tail for bandwidth
//!    detection.
//! 2. [`tonality_analysis`] windows 480 mono samples, runs a 480-pt
//!    complex FFT, and derives per-bin tonality + noisiness from the
//!    instantaneous phase structure.
//! 3. Band energies feed a BFCC/tonality feature vector.
//! 4. The vector is classified by a small MLP + GRU + MLP stack
//!    defined in `mlp.rs` / `mlp_data.rs`.
//! 5. The resulting [`AnalysisInfo`] is ring-buffered with a
//!    lookahead to compensate for detector delay.
//!
//! # Fixed-point vs float
//!
//! The reference analyzer is authored in float but threads integer
//! PCM (`opus_val32 = i32` under FIXED_POINT) through the FFT and
//! energy accumulation. Intermediate float scalars (`angle`,
//! `tonality`, features, MLP activations) are all `f32`. This port
//! mirrors that split: `inmem` is `i32`, the MLP math is `f32`, and
//! conversions happen at the same points as in C.
//!
//! # Reset discipline
//!
//! `TonalityAnalysisState` has a `TONALITY_ANALYSIS_RESET_START`
//! delimiter (the `angle` field, per `analysis.h:51`). Fields before
//! that point (`arch`, `application`, `Fs`) survive a reset; everything
//! at or after `angle` is zeroed. See [`tonality_analysis_reset`].

// Stage 6.3 lands this module ahead of its Stage 6.4 encoder wiring.
#![allow(dead_code)]

use crate::celt::fft::{KissFftCpx, KissFftState, opus_fft};
use crate::celt::mdct::MDCT_48000_960;
use crate::celt::modes::CELTMode;
use crate::types::{
    SIG_SHIFT, add32, half32, mult16_32_q15, qconst16, sub32,
};

use super::mlp::{
    MAX_NEURONS, analysis_compute_dense, analysis_compute_gru,
};
use super::mlp_data::{LAYER0, LAYER1, LAYER2};

/// Matches C's fallback `#define M_PI 3.141592653` in reference/src/analysis.c:49-50.
/// MSVC's <math.h> does not define M_PI without _USE_MATH_DEFINES, which the
/// harness config doesn't set — so the reference compiles with this truncated
/// 10-digit value.
///
/// Note: when rounded to f32 this happens to produce the same bit pattern as the
/// full-precision pi (both land on `0x40490fdb`), so the f32 value itself is not
/// the source of drift. What matters is the **computation path**: in C, `M_PI`
/// is a bare double literal, so expressions like `M_PI*M_PI*M_PI*M_PI` and
/// `.5f/M_PI` run in double precision before the single cast to float. See
/// [`M_PI_F64`] for the double-precision value used to replicate that chain.
const M_PI: f32 = 3.141592653_f32;

/// Double-precision form of the truncated M_PI fallback, used to mirror C's
/// `(float)(M_PI expr)` chains bit-exactly. Any expression involving `M_PI` in
/// `analysis.c` is evaluated in `double` (because `M_PI` expands to a bare
/// double literal) and cast to float at the outermost `(float)`. Computing
/// with this f64 constant and casting once matches that path.
const M_PI_F64: f64 = 3.141592653_f64;

// ===========================================================================
// Public constants (ported from analysis.h)
// ===========================================================================

/// Number of frames retained in the energy history for stationarity.
/// Matches `analysis.h:35`.
pub(crate) const NB_FRAMES: usize = 8;

/// Number of tonality bands. Matches `analysis.h:36`.
pub(crate) const NB_TBANDS: usize = 18;

/// Internal sample buffer size (30 ms at 24 kHz). Matches `analysis.h:37`.
pub(crate) const ANALYSIS_BUF_SIZE: usize = 720;

/// Saturation point for the `count` field. Matches `analysis.h:40`.
pub(crate) const ANALYSIS_COUNT_MAX: i32 = 10_000;

/// Ring-buffer capacity for per-frame `AnalysisInfo`. Matches `analysis.h:42`.
pub(crate) const DETECT_SIZE: usize = 100;

/// Number of analysis bands to skip when scoring tonality. Matches
/// `analysis.c:113`.
const NB_TONAL_SKIP_BANDS: usize = 9;

/// Switching penalty for the music/speech transition cost function.
/// Matches `analysis.c:55`.
const TRANSITION_PENALTY: f32 = 10.0_f32;

/// Leakage-band target offset in log2 energy. Matches `analysis.c:415`.
const LEAKAGE_OFFSET: f32 = 2.5_f32;

/// Leakage-band slope per Bark. Matches `analysis.c:416`.
const LEAKAGE_SLOPE: f32 = 2.0_f32;

/// Max number of leak-boost entries stored in `AnalysisInfo`.
/// Matches `celt.h:63` (`#define LEAK_BANDS 19`).
pub(crate) const LEAK_BANDS: usize = 19;

// ===========================================================================
// AnalysisInfo — ported from celt.h:65-79
// ===========================================================================

/// Per-frame analysis result consumed by the Opus encoder.
///
/// Port of the C `AnalysisInfo` struct in `reference/celt/celt.h:65-79`.
/// `leak_boost` is Q6 packed into `u8`; the other fields are native floats
/// or ints. `#[repr(C)]` mirrors the C layout so future FFI parity tests
/// (and the sibling `TonalityAnalysisState`, which also uses `#[repr(C)]`)
/// see byte-identical structs.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AnalysisInfo {
    pub valid: i32,
    pub tonality: f32,
    pub tonality_slope: f32,
    pub noisiness: f32,
    pub activity: f32,
    pub music_prob: f32,
    pub music_prob_min: f32,
    pub music_prob_max: f32,
    pub bandwidth: i32,
    pub activity_probability: f32,
    pub max_pitch_ratio: f32,
    /// Per-band leakage boost, Q6 packed as u8.
    pub leak_boost: [u8; LEAK_BANDS],
}

impl Default for AnalysisInfo {
    fn default() -> Self {
        Self::zeroed()
    }
}

impl AnalysisInfo {
    /// All-zero `AnalysisInfo`, matching a freshly memset C struct.
    #[inline]
    pub const fn zeroed() -> Self {
        Self {
            valid: 0,
            tonality: 0.0,
            tonality_slope: 0.0,
            noisiness: 0.0,
            activity: 0.0,
            music_prob: 0.0,
            music_prob_min: 0.0,
            music_prob_max: 0.0,
            bandwidth: 0,
            activity_probability: 0.0,
            max_pitch_ratio: 0.0,
            leak_boost: [0; LEAK_BANDS],
        }
    }
}

// ===========================================================================
// Downmix function signature (ported from opus_private.h:175)
// ===========================================================================

/// Signature of a downmix callback.
///
/// `downmix(input_pcm, output_sub, subframe, offset, c1, c2, channels)`:
/// writes `subframe` samples into `output_sub`, reading `input_pcm` at
/// byte offset `offset` in the caller-chosen PCM format (float, i16, i24).
/// `c1` selects the primary channel, `c2` is a secondary channel
/// (`-1` = ignore, `-2` = sum remaining channels).
///
/// Matches the C `downmix_func` typedef in `opus_private.h:175`.
pub type DownmixFunc =
    fn(input: &[u8], output: &mut [i32], subframe: i32, offset: i32, c1: i32, c2: i32, c: i32);

// ===========================================================================
// Static tables (ported verbatim from analysis.c)
// ===========================================================================

/// 8x16 type-II DCT matrix used for BFCC extraction.
/// Matches `analysis.c:57-74` (128 float constants, row-major 8 rows x 16 cols).
static DCT_TABLE: [f32; 128] = [
    0.250000_f32, 0.250000_f32, 0.250000_f32, 0.250000_f32, 0.250000_f32, 0.250000_f32,
    0.250000_f32, 0.250000_f32, 0.250000_f32, 0.250000_f32, 0.250000_f32, 0.250000_f32,
    0.250000_f32, 0.250000_f32, 0.250000_f32, 0.250000_f32, 0.351851_f32, 0.338330_f32,
    0.311806_f32, 0.273300_f32, 0.224292_f32, 0.166664_f32, 0.102631_f32, 0.034654_f32,
    -0.034654_f32, -0.102631_f32, -0.166664_f32, -0.224292_f32, -0.273300_f32, -0.311806_f32,
    -0.338330_f32, -0.351851_f32, 0.346760_f32, 0.293969_f32, 0.196424_f32, 0.068975_f32,
    -0.068975_f32, -0.196424_f32, -0.293969_f32, -0.346760_f32, -0.346760_f32, -0.293969_f32,
    -0.196424_f32, -0.068975_f32, 0.068975_f32, 0.196424_f32, 0.293969_f32, 0.346760_f32,
    0.338330_f32, 0.224292_f32, 0.034654_f32, -0.166664_f32, -0.311806_f32, -0.351851_f32,
    -0.273300_f32, -0.102631_f32, 0.102631_f32, 0.273300_f32, 0.351851_f32, 0.311806_f32,
    0.166664_f32, -0.034654_f32, -0.224292_f32, -0.338330_f32, 0.326641_f32, 0.135299_f32,
    -0.135299_f32, -0.326641_f32, -0.326641_f32, -0.135299_f32, 0.135299_f32, 0.326641_f32,
    0.326641_f32, 0.135299_f32, -0.135299_f32, -0.326641_f32, -0.326641_f32, -0.135299_f32,
    0.135299_f32, 0.326641_f32, 0.311806_f32, 0.034654_f32, -0.273300_f32, -0.338330_f32,
    -0.102631_f32, 0.224292_f32, 0.351851_f32, 0.166664_f32, -0.166664_f32, -0.351851_f32,
    -0.224292_f32, 0.102631_f32, 0.338330_f32, 0.273300_f32, -0.034654_f32, -0.311806_f32,
    0.293969_f32, -0.068975_f32, -0.346760_f32, -0.196424_f32, 0.196424_f32, 0.346760_f32,
    0.068975_f32, -0.293969_f32, -0.293969_f32, 0.068975_f32, 0.346760_f32, 0.196424_f32,
    -0.196424_f32, -0.346760_f32, -0.068975_f32, 0.293969_f32, 0.273300_f32, -0.166664_f32,
    -0.338330_f32, 0.034654_f32, 0.351851_f32, 0.102631_f32, -0.311806_f32, -0.224292_f32,
    0.224292_f32, 0.311806_f32, -0.102631_f32, -0.351851_f32, -0.034654_f32, 0.338330_f32,
    0.166664_f32, -0.273300_f32,
];

/// 240-sample analysis window (sine-shaped). Matches `analysis.c:76-107`.
static ANALYSIS_WINDOW: [f32; 240] = [
    0.000043_f32, 0.000171_f32, 0.000385_f32, 0.000685_f32, 0.001071_f32, 0.001541_f32,
    0.002098_f32, 0.002739_f32, 0.003466_f32, 0.004278_f32, 0.005174_f32, 0.006156_f32,
    0.007222_f32, 0.008373_f32, 0.009607_f32, 0.010926_f32, 0.012329_f32, 0.013815_f32,
    0.015385_f32, 0.017037_f32, 0.018772_f32, 0.020590_f32, 0.022490_f32, 0.024472_f32,
    0.026535_f32, 0.028679_f32, 0.030904_f32, 0.033210_f32, 0.035595_f32, 0.038060_f32,
    0.040604_f32, 0.043227_f32, 0.045928_f32, 0.048707_f32, 0.051564_f32, 0.054497_f32,
    0.057506_f32, 0.060591_f32, 0.063752_f32, 0.066987_f32, 0.070297_f32, 0.073680_f32,
    0.077136_f32, 0.080665_f32, 0.084265_f32, 0.087937_f32, 0.091679_f32, 0.095492_f32,
    0.099373_f32, 0.103323_f32, 0.107342_f32, 0.111427_f32, 0.115579_f32, 0.119797_f32,
    0.124080_f32, 0.128428_f32, 0.132839_f32, 0.137313_f32, 0.141849_f32, 0.146447_f32,
    0.151105_f32, 0.155823_f32, 0.160600_f32, 0.165435_f32, 0.170327_f32, 0.175276_f32,
    0.180280_f32, 0.185340_f32, 0.190453_f32, 0.195619_f32, 0.200838_f32, 0.206107_f32,
    0.211427_f32, 0.216797_f32, 0.222215_f32, 0.227680_f32, 0.233193_f32, 0.238751_f32,
    0.244353_f32, 0.250000_f32, 0.255689_f32, 0.261421_f32, 0.267193_f32, 0.273005_f32,
    0.278856_f32, 0.284744_f32, 0.290670_f32, 0.296632_f32, 0.302628_f32, 0.308658_f32,
    0.314721_f32, 0.320816_f32, 0.326941_f32, 0.333097_f32, 0.339280_f32, 0.345492_f32,
    0.351729_f32, 0.357992_f32, 0.364280_f32, 0.370590_f32, 0.376923_f32, 0.383277_f32,
    0.389651_f32, 0.396044_f32, 0.402455_f32, 0.408882_f32, 0.415325_f32, 0.421783_f32,
    0.428254_f32, 0.434737_f32, 0.441231_f32, 0.447736_f32, 0.454249_f32, 0.460770_f32,
    0.467298_f32, 0.473832_f32, 0.480370_f32, 0.486912_f32, 0.493455_f32, 0.500000_f32,
    0.506545_f32, 0.513088_f32, 0.519630_f32, 0.526168_f32, 0.532702_f32, 0.539230_f32,
    0.545751_f32, 0.552264_f32, 0.558769_f32, 0.565263_f32, 0.571746_f32, 0.578217_f32,
    0.584675_f32, 0.591118_f32, 0.597545_f32, 0.603956_f32, 0.610349_f32, 0.616723_f32,
    0.623077_f32, 0.629410_f32, 0.635720_f32, 0.642008_f32, 0.648271_f32, 0.654508_f32,
    0.660720_f32, 0.666903_f32, 0.673059_f32, 0.679184_f32, 0.685279_f32, 0.691342_f32,
    0.697372_f32, 0.703368_f32, 0.709330_f32, 0.715256_f32, 0.721144_f32, 0.726995_f32,
    0.732807_f32, 0.738579_f32, 0.744311_f32, 0.750000_f32, 0.755647_f32, 0.761249_f32,
    0.766807_f32, 0.772320_f32, 0.777785_f32, 0.783203_f32, 0.788573_f32, 0.793893_f32,
    0.799162_f32, 0.804381_f32, 0.809547_f32, 0.814660_f32, 0.819720_f32, 0.824724_f32,
    0.829673_f32, 0.834565_f32, 0.839400_f32, 0.844177_f32, 0.848895_f32, 0.853553_f32,
    0.858151_f32, 0.862687_f32, 0.867161_f32, 0.871572_f32, 0.875920_f32, 0.880203_f32,
    0.884421_f32, 0.888573_f32, 0.892658_f32, 0.896677_f32, 0.900627_f32, 0.904508_f32,
    0.908321_f32, 0.912063_f32, 0.915735_f32, 0.919335_f32, 0.922864_f32, 0.926320_f32,
    0.929703_f32, 0.933013_f32, 0.936248_f32, 0.939409_f32, 0.942494_f32, 0.945503_f32,
    0.948436_f32, 0.951293_f32, 0.954072_f32, 0.956773_f32, 0.959396_f32, 0.961940_f32,
    0.964405_f32, 0.966790_f32, 0.969096_f32, 0.971321_f32, 0.973465_f32, 0.975528_f32,
    0.977510_f32, 0.979410_f32, 0.981228_f32, 0.982963_f32, 0.984615_f32, 0.986185_f32,
    0.987671_f32, 0.989074_f32, 0.990393_f32, 0.991627_f32, 0.992778_f32, 0.993844_f32,
    0.994826_f32, 0.995722_f32, 0.996534_f32, 0.997261_f32, 0.997902_f32, 0.998459_f32,
    0.998929_f32, 0.999315_f32, 0.999615_f32, 0.999829_f32, 0.999957_f32, 1.000000_f32,
];

/// Tonality-band edges (bin indices into the 480-pt FFT output).
/// Matches `analysis.c:109-111`. Length `NB_TBANDS + 1 = 19`.
static TBANDS: [i32; NB_TBANDS + 1] = [
    4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 136, 160, 192, 240,
];

/// Feature-stddev bias. Matches `analysis.c:410-413`.
static STD_FEATURE_BIAS: [f32; 9] = [
    5.684947_f32, 3.475288_f32, 1.770634_f32, 1.599784_f32, 3.773215_f32,
    2.163313_f32, 1.260756_f32, 1.116868_f32, 1.918795_f32,
];

// ===========================================================================
// TonalityAnalysisState
// ===========================================================================

/// Stateful tonality / speech-vs-music analyzer.
///
/// Port of the C `TonalityAnalysisState` struct in `analysis.h:47-81`.
/// The field layout matches C one-to-one so the
/// `TONALITY_ANALYSIS_RESET_START` contract (all fields from `angle`
/// onward are zeroed on reset) can be implemented as a straightforward
/// field-by-field wipe.
#[repr(C)]
pub struct TonalityAnalysisState {
    // ---- Fields before TONALITY_ANALYSIS_RESET_START (preserved on reset) ----
    pub arch: i32,
    pub application: i32,
    pub fs: i32,

    // ---- TONALITY_ANALYSIS_RESET_START ----
    pub angle: [f32; 240],
    pub d_angle: [f32; 240],
    pub d2_angle: [f32; 240],
    pub inmem: [i32; ANALYSIS_BUF_SIZE],
    pub mem_fill: i32,
    pub prev_band_tonality: [f32; NB_TBANDS],
    pub prev_tonality: f32,
    pub prev_bandwidth: i32,
    pub e_frames: [[f32; NB_TBANDS]; NB_FRAMES],
    pub log_e_frames: [[f32; NB_TBANDS]; NB_FRAMES],
    pub low_e: [f32; NB_TBANDS],
    pub high_e: [f32; NB_TBANDS],
    pub mean_e: [f32; NB_TBANDS + 1],
    pub mem: [f32; 32],
    pub cmean: [f32; 8],
    pub std: [f32; 9],
    pub e_tracker: f32,
    pub low_e_count: f32,
    pub e_count: i32,
    pub count: i32,
    pub analysis_offset: i32,
    pub write_pos: i32,
    pub read_pos: i32,
    pub read_subframe: i32,
    pub hp_ener_accum: f32,
    pub initialized: i32,
    pub rnn_state: [f32; MAX_NEURONS],
    pub downmix_state: [i32; 3],
    pub info: [AnalysisInfo; DETECT_SIZE],
}

impl TonalityAnalysisState {
    /// All-zero state — every field, including the pre-reset ones. Convenience
    /// constructor for test code and storage allocation; production code should
    /// go through [`tonality_analysis_init`].
    pub fn new() -> Self {
        Self {
            arch: 0,
            application: 0,
            fs: 0,
            angle: [0.0; 240],
            d_angle: [0.0; 240],
            d2_angle: [0.0; 240],
            inmem: [0; ANALYSIS_BUF_SIZE],
            mem_fill: 0,
            prev_band_tonality: [0.0; NB_TBANDS],
            prev_tonality: 0.0,
            prev_bandwidth: 0,
            e_frames: [[0.0; NB_TBANDS]; NB_FRAMES],
            log_e_frames: [[0.0; NB_TBANDS]; NB_FRAMES],
            low_e: [0.0; NB_TBANDS],
            high_e: [0.0; NB_TBANDS],
            mean_e: [0.0; NB_TBANDS + 1],
            mem: [0.0; 32],
            cmean: [0.0; 8],
            std: [0.0; 9],
            e_tracker: 0.0,
            low_e_count: 0.0,
            e_count: 0,
            count: 0,
            analysis_offset: 0,
            write_pos: 0,
            read_pos: 0,
            read_subframe: 0,
            hp_ener_accum: 0.0,
            initialized: 0,
            rnn_state: [0.0; MAX_NEURONS],
            downmix_state: [0; 3],
            info: [AnalysisInfo::zeroed(); DETECT_SIZE],
        }
    }

    /// Allocate a zeroed `TonalityAnalysisState` on the heap.
    ///
    /// The state is ~75 KB; Stage 6.4 will embed it in the Opus encoder
    /// struct, where sticking to `Self::new()` risks materializing the
    /// whole thing on the stack first. This constructor zeros the heap
    /// slot in place with `write_bytes`, avoiding any stack temporary.
    ///
    /// Every field of `Self` (i32, f32, arrays thereof, arrays of the
    /// `#[repr(C)]` POD `AnalysisInfo`) has an all-zero-byte-pattern for
    /// its logical zero value, so `write_bytes(..., 0, ...)` produces the
    /// same observable state as [`Self::new()`].
    pub fn new_boxed() -> Box<Self> {
        let mut slot = Box::<core::mem::MaybeUninit<Self>>::new_uninit();
        // Safety: `slot` is a Box of exactly `size_of::<Self>()` bytes of
        // uninitialised but valid-for-writes heap memory. We zero every
        // byte, which is a valid bit-pattern for every field of `Self`
        // (all POD: i32 / f32 / u8 arrays + `#[repr(C)]` AnalysisInfo).
        // Then cast the Box type since `Box::<MaybeUninit<T>>::assume_init`
        // is still unstable.
        unsafe {
            core::ptr::write_bytes(
                slot.as_mut_ptr() as *mut u8,
                0,
                core::mem::size_of::<Self>(),
            );
            let raw = Box::into_raw(slot);
            Box::from_raw(raw as *mut Self)
        }
    }
}

// ===========================================================================
// Public API — init / reset / get_info / run
// ===========================================================================

/// Initialize a [`TonalityAnalysisState`].
///
/// Sets the reusable fields (`arch`, `fs`) and then delegates to
/// [`tonality_analysis_reset`] to zero everything from
/// `TONALITY_ANALYSIS_RESET_START` onward. Port of `tonality_analysis_init`
/// in `analysis.c:216-223`.
///
/// `arch` is hard-coded to `0` (the scalar path). The C reference calls
/// `opus_select_arch()` here, but the Rust port only implements the scalar
/// DSP path — there is no runtime SIMD dispatch to select.
pub fn tonality_analysis_init(tonal: &mut TonalityAnalysisState, fs: i32) {
    // Initialize reusable fields.
    tonal.arch = 0;
    tonal.fs = fs;
    // Clear remaining fields.
    tonality_analysis_reset(tonal);
}

/// Reset a [`TonalityAnalysisState`] at a signal discontinuity.
///
/// Clears every field from `TONALITY_ANALYSIS_RESET_START` (`angle`)
/// onward to zero; fields before that point (`arch`, `application`,
/// `fs`) are retained. Port of `tonality_analysis_reset` in
/// `analysis.c:225-230`.
pub(crate) fn tonality_analysis_reset(tonal: &mut TonalityAnalysisState) {
    tonal.angle = [0.0; 240];
    tonal.d_angle = [0.0; 240];
    tonal.d2_angle = [0.0; 240];
    tonal.inmem = [0; ANALYSIS_BUF_SIZE];
    tonal.mem_fill = 0;
    tonal.prev_band_tonality = [0.0; NB_TBANDS];
    tonal.prev_tonality = 0.0;
    tonal.prev_bandwidth = 0;
    tonal.e_frames = [[0.0; NB_TBANDS]; NB_FRAMES];
    tonal.log_e_frames = [[0.0; NB_TBANDS]; NB_FRAMES];
    tonal.low_e = [0.0; NB_TBANDS];
    tonal.high_e = [0.0; NB_TBANDS];
    tonal.mean_e = [0.0; NB_TBANDS + 1];
    tonal.mem = [0.0; 32];
    tonal.cmean = [0.0; 8];
    tonal.std = [0.0; 9];
    tonal.e_tracker = 0.0;
    tonal.low_e_count = 0.0;
    tonal.e_count = 0;
    tonal.count = 0;
    tonal.analysis_offset = 0;
    tonal.write_pos = 0;
    tonal.read_pos = 0;
    tonal.read_subframe = 0;
    tonal.hp_ener_accum = 0.0;
    tonal.initialized = 0;
    tonal.rnn_state = [0.0; MAX_NEURONS];
    tonal.downmix_state = [0; 3];
    tonal.info = [AnalysisInfo::zeroed(); DETECT_SIZE];
}

// ===========================================================================
// Internal helpers — small scalar utilities matching C macros
// ===========================================================================

/// `MAX32(a, b)` for f32. Matches the C ternary (`a < b ? b : a`) for
/// non-NaN inputs. Matches `arch.h:293`.
#[inline(always)]
fn fmax32(a: f32, b: f32) -> f32 {
    if a < b { b } else { a }
}

/// `MIN32(a, b)` for f32. Matches `arch.h:292`.
#[inline(always)]
fn fmin32(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// `MAX16(a, b)` for f32 — same as `fmax32` in float mode.
#[inline(always)]
fn fmax16(a: f32, b: f32) -> f32 {
    fmax32(a, b)
}

/// `MIN16(a, b)` for f32 — same as `fmin32` in float mode.
#[inline(always)]
fn fmin16(a: f32, b: f32) -> f32 {
    fmin32(a, b)
}

/// `ABS16(x)` for f32 — fabsf in float mode.
#[inline(always)]
fn fabs16(x: f32) -> f32 {
    x.abs()
}

/// `IMIN(a, b)` for i32.
#[inline(always)]
fn imin(a: i32, b: i32) -> i32 {
    if a < b { a } else { b }
}

/// `IMAX(a, b)` for i32.
#[inline(always)]
fn imax(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}

/// Float-to-int rounding matching the C `float2int()` helper: round-half-even.
/// See `float_cast.h:68,125,136`. Rust's `round_ties_even` matches
/// `_mm_cvt_ss2si` (SSE default rounding) and `lrintf` (C99 default).
#[inline(always)]
fn float2int(x: f32) -> i32 {
    x.round_ties_even() as i32
}

/// `celt_isnan(x)` — matches the non-fixed branch of `arch.h:273-283`.
#[inline(always)]
fn celt_isnan(x: f32) -> bool {
    x.is_nan()
}

/// `fast_atan2f` — matches `mathops.h:60-77` (analysis.c is
/// `ANALYSIS_C`, so this path is always compiled in even for fixed-point
/// builds).
#[inline]
fn fast_atan2f(y: f32, x: f32) -> f32 {
    const CA: f32 = 0.43157974_f32;
    const CB: f32 = 0.67848403_f32;
    const CC: f32 = 0.08595542_f32;
    // M_PI / 2 as a single-precision constant — C computes this as
    // `(float)PI/2` where `PI = 3.1415926535897931`. Matches the bit
    // pattern of `core::f32::consts::FRAC_PI_2`.
    const CE: f32 = core::f32::consts::FRAC_PI_2;

    let x2 = x * x;
    let y2 = y * y;
    // For very small values, return 0. Matches the C guard.
    if x2 + y2 < 1e-18_f32 {
        return 0.0_f32;
    }
    if x2 < y2 {
        let den = (y2 + CB * x2) * (y2 + CC * x2);
        -x * y * (y2 + CA * x2) / den + (if y < 0.0 { -CE } else { CE })
    } else {
        let den = (x2 + CB * y2) * (x2 + CC * y2);
        x * y * (x2 + CA * y2) / den
            + (if y < 0.0 { -CE } else { CE })
            - (if x * y < 0.0 { -CE } else { CE })
    }
}

/// Detect all-zero PCM buffer.
///
/// Matches the FIXED_POINT branch of `is_digital_silence32` in
/// `analysis.c:428-440`: a frame is silent iff every sample is zero.
/// `lsb_depth` is unused in fixed-point, just like the C reference.
#[inline]
fn is_digital_silence32(pcm: &[i32]) -> bool {
    for &s in pcm.iter() {
        if s != 0 {
            return false;
        }
    }
    true
}

// ===========================================================================
// silk_resampler_down2_hp — 48 kHz -> 24 kHz downsampler with HP tap
// ===========================================================================

/// Two-allpass-section IIR downsampler with a high-pass energy output.
///
/// Port of `silk_resampler_down2_hp` in `analysis.c:115-163`. Consumes
/// `inp` (length `2 * len2` where `len2 = inp.len()/2`), writes `len2`
/// samples to `out`, updates `state[0..=2]`, and returns the accumulated
/// high-pass energy in Q(2*SIG_SHIFT+8) for use in bandwidth detection.
///
/// The algorithm mirrors `silk/resampler_down2.c` but with an extra
/// third-order state that captures the high-pass residual. The fixed-point
/// math uses `MULT16_32_Q15(QCONST16(c, 15), Y)` which in Rust is
/// `mult16_32_q15(qconst16(c, 15), y)`.
fn silk_resampler_down2_hp(
    state: &mut [i32; 3],
    out: &mut [i32],
    inp: &[i32],
    in_len: i32,
) -> i32 {
    let len2 = in_len / 2;
    let mut hp_ener: i64 = 0;

    // Q15 filter coefficients (hoisted so the loop body matches the C
    // `QCONST16(...)` intent without constructing the qconst on every
    // iteration).
    let c0: i32 = qconst16(0.6074371_f64, 15); // 19906
    let c1: i32 = qconst16(0.15063_f64, 15);   // 4936

    for k in 0..len2 as usize {
        // Even input sample
        let mut in32 = inp[2 * k];

        // All-pass section for even input
        let y = sub32(in32, state[0]);
        let x = mult16_32_q15(c0, y);
        let mut out32 = add32(state[0], x);
        state[0] = add32(in32, x);
        let mut out32_hp = out32;

        // Odd input sample
        in32 = inp[2 * k + 1];

        // All-pass section for odd input, summed into out32
        let y = sub32(in32, state[1]);
        let x = mult16_32_q15(c1, y);
        out32 = add32(out32, state[1]);
        out32 = add32(out32, x);
        state[1] = add32(in32, x);

        // Third-order HP tap (negated input into the state).
        let neg_in = -in32;
        let y = sub32(neg_in, state[2]);
        let x = mult16_32_q15(c1, y);
        out32_hp = add32(out32_hp, state[2]);
        out32_hp = add32(out32_hp, x);
        state[2] = add32(neg_in, x);

        // len2 can be up to 480, so shift by 8 to avoid overflow in the
        // i64 accumulator.
        hp_ener += (out32_hp as i64 * out32_hp as i64) >> 8;
        // Store the downsampled output (Q format preserved).
        out[k] = half32(out32);
    }

    // Fixed-point: clamp to the 32-bit range after the shift from Q(2*SIG_SHIFT).
    hp_ener >>= 2 * SIG_SHIFT as i64;
    if hp_ener > 2_147_483_647 {
        hp_ener = 2_147_483_647;
    }
    hp_ener as i32
}

// ===========================================================================
// downmix_and_resample
// ===========================================================================

/// Downmix + optional 48 kHz -> 24 kHz / 16 kHz -> 24 kHz resample.
///
/// Port of `downmix_and_resample` in `analysis.c:165-214`. The Rust port
/// takes the raw pcm bytes as `&[u8]` and relies on the caller's
/// [`DownmixFunc`] to interpret them (float, i16, i24). Returns the
/// accumulated high-pass energy (from `silk_resampler_down2_hp` when
/// Fs=48000; otherwise zero).
///
/// `subframe` is in input samples at the caller rate; `offset` likewise.
/// On entry, the 48 kHz and 16 kHz branches rescale both to the 24 kHz
/// domain. An assertion guards against unsupported sample rates.
fn downmix_and_resample(
    downmix: DownmixFunc,
    x: &[u8],
    y: &mut [i32],
    state: &mut [i32; 3],
    mut subframe: i32,
    mut offset: i32,
    c1: i32,
    c2: i32,
    c: i32,
    fs: i32,
) -> i32 {
    if subframe == 0 {
        return 0;
    }

    // Bring `subframe`/`offset` into the caller's native rate so the
    // downmix callback can pull from the original PCM array.
    if fs == 48_000 {
        subframe *= 2;
        offset *= 2;
    } else if fs == 16_000 {
        subframe = subframe * 2 / 3;
        offset = offset * 2 / 3;
    } else if fs != 24_000 {
        // C uses celt_assert(0) here. Rust: panic explicitly so the
        // upstream bug is loud.
        panic!("downmix_and_resample: unsupported Fs={}", fs);
    }

    // Scratch buffer at the caller rate.
    let mut tmp: Vec<i32> = vec![0; subframe as usize];
    downmix(x, &mut tmp, subframe, offset, c1, c2, c);

    // Halve the downmix when it summed two channels (c2 in [0, C-1] or
    // the "sum-all" sentinel -2 with stereo input).
    if (c2 == -2 && c == 2) || c2 > -1 {
        for j in 0..subframe as usize {
            tmp[j] = half32(tmp[j]);
        }
    }

    let ret: i32;
    if fs == 48_000 {
        ret = silk_resampler_down2_hp(state, y, &tmp, subframe);
    } else if fs == 24_000 {
        // OPUS_COPY(y, tmp, subframe)
        for j in 0..subframe as usize {
            y[j] = tmp[j];
        }
        ret = 0;
    } else {
        // Fs == 16000. Zero-order hold upsample 3x, then run the 2:1 HP
        // downsampler to hit 24 kHz. See the "Don't do this at home"
        // comment in the C reference (`analysis.c:198-200`).
        debug_assert_eq!(fs, 16_000);
        let mut tmp3x: Vec<i32> = vec![0; (3 * subframe) as usize];
        for j in 0..subframe as usize {
            tmp3x[3 * j] = tmp[j];
            tmp3x[3 * j + 1] = tmp[j];
            tmp3x[3 * j + 2] = tmp[j];
        }
        ret = silk_resampler_down2_hp(state, y, &tmp3x, 3 * subframe);
    }
    // The float branch of the C reference scales `ret` by 1/(32768^2)
    // here; the fixed-point branch does not. We're in FIXED_POINT mode.
    ret
}

// ===========================================================================
// tonality_get_info
// ===========================================================================

/// Fetch the most relevant [`AnalysisInfo`] from the ring buffer,
/// applying the cross-frame stabilization + switching-cost logic.
///
/// Port of `tonality_get_info` in `analysis.c:232-408`. `len` is the
/// output frame length in samples at the original (external) rate; it's
/// divided down by `Fs/400` to advance the read position in
/// subframe-of-2.5ms increments.
pub fn tonality_get_info(
    tonal: &mut TonalityAnalysisState,
    info_out: &mut AnalysisInfo,
    len: i32,
) {
    let mut pos = tonal.read_pos;
    let mut curr_lookahead = tonal.write_pos - tonal.read_pos;
    if curr_lookahead < 0 {
        curr_lookahead += DETECT_SIZE as i32;
    }

    tonal.read_subframe += len / (tonal.fs / 400);
    while tonal.read_subframe >= 8 {
        tonal.read_subframe -= 8;
        tonal.read_pos += 1;
    }
    if tonal.read_pos >= DETECT_SIZE as i32 {
        tonal.read_pos -= DETECT_SIZE as i32;
    }

    // On long frames, look at the second analysis window rather than the first.
    if len > tonal.fs / 50 && pos != tonal.write_pos {
        pos += 1;
        if pos == DETECT_SIZE as i32 {
            pos = 0;
        }
    }
    if pos == tonal.write_pos {
        pos -= 1;
    }
    if pos < 0 {
        pos = DETECT_SIZE as i32 - 1;
    }
    let pos0 = pos;
    *info_out = tonal.info[pos as usize];
    if info_out.valid == 0 {
        return;
    }
    let mut tonality_max = info_out.tonality;
    let mut tonality_avg = info_out.tonality;
    let mut tonality_count: i32 = 1;
    // Look at the neighbouring frames and pick largest bandwidth found (to be safe).
    let mut bandwidth_span: i32 = 6;
    // If possible, look ahead for a tone to compensate for the delay in the tone detector.
    for _ in 0..3 {
        pos += 1;
        if pos == DETECT_SIZE as i32 {
            pos = 0;
        }
        if pos == tonal.write_pos {
            break;
        }
        tonality_max = fmax32(tonality_max, tonal.info[pos as usize].tonality);
        tonality_avg += tonal.info[pos as usize].tonality;
        tonality_count += 1;
        info_out.bandwidth = imax(info_out.bandwidth, tonal.info[pos as usize].bandwidth);
        bandwidth_span -= 1;
    }
    pos = pos0;
    // Look back in time to see if any has a wider bandwidth than the current frame.
    for _ in 0..bandwidth_span {
        pos -= 1;
        if pos < 0 {
            pos = DETECT_SIZE as i32 - 1;
        }
        if pos == tonal.write_pos {
            break;
        }
        info_out.bandwidth = imax(info_out.bandwidth, tonal.info[pos as usize].bandwidth);
    }
    info_out.tonality = fmax32(
        tonality_avg / tonality_count as f32,
        tonality_max - 0.2_f32,
    );

    let mut mpos = pos0;
    let mut vpos = pos0;
    // If we have enough look-ahead, compensate for the ~5-frame delay in the music prob and
    // ~1 frame delay in the VAD prob.
    if curr_lookahead > 15 {
        mpos += 5;
        if mpos >= DETECT_SIZE as i32 {
            mpos -= DETECT_SIZE as i32;
        }
        vpos += 1;
        if vpos >= DETECT_SIZE as i32 {
            vpos -= DETECT_SIZE as i32;
        }
    }

    // Transition-cost analysis. See the C comment (`analysis.c:321-351`) for the derivation.
    let mut prob_min: f32 = 1.0_f32;
    let mut prob_max: f32 = 0.0_f32;
    let vad_prob = tonal.info[vpos as usize].activity_probability;
    let mut prob_count = fmax16(0.1_f32, vad_prob);
    let mut prob_avg = fmax16(0.1_f32, vad_prob) * tonal.info[mpos as usize].music_prob;
    loop {
        mpos += 1;
        if mpos == DETECT_SIZE as i32 {
            mpos = 0;
        }
        if mpos == tonal.write_pos {
            break;
        }
        vpos += 1;
        if vpos == DETECT_SIZE as i32 {
            vpos = 0;
        }
        if vpos == tonal.write_pos {
            break;
        }
        let pos_vad = tonal.info[vpos as usize].activity_probability;
        prob_min = fmin16(
            (prob_avg - TRANSITION_PENALTY * (vad_prob - pos_vad)) / prob_count,
            prob_min,
        );
        prob_max = fmax16(
            (prob_avg + TRANSITION_PENALTY * (vad_prob - pos_vad)) / prob_count,
            prob_max,
        );
        prob_count += fmax16(0.1_f32, pos_vad);
        prob_avg += fmax16(0.1_f32, pos_vad) * tonal.info[mpos as usize].music_prob;
    }
    info_out.music_prob = prob_avg / prob_count;
    prob_min = fmin16(prob_avg / prob_count, prob_min);
    prob_max = fmax16(prob_avg / prob_count, prob_max);
    prob_min = fmax16(prob_min, 0.0_f32);
    prob_max = fmin16(prob_max, 1.0_f32);

    // If we don't have enough look-ahead, do our best to make a decent decision.
    if curr_lookahead < 10 {
        let mut pmin = prob_min;
        let mut pmax = prob_max;
        pos = pos0;
        // Look for min/max in the past.
        let back = imin(tonal.count - 1, 15);
        for _ in 0..back {
            pos -= 1;
            if pos < 0 {
                pos = DETECT_SIZE as i32 - 1;
            }
            pmin = fmin16(pmin, tonal.info[pos as usize].music_prob);
            pmax = fmax16(pmax, tonal.info[pos as usize].music_prob);
        }
        // Bias against switching on active audio.
        pmin = fmax16(0.0_f32, pmin - 0.1_f32 * vad_prob);
        pmax = fmin16(1.0_f32, pmax + 0.1_f32 * vad_prob);
        prob_min += (1.0_f32 - 0.1_f32 * curr_lookahead as f32) * (pmin - prob_min);
        prob_max += (1.0_f32 - 0.1_f32 * curr_lookahead as f32) * (pmax - prob_max);
    }
    info_out.music_prob_min = prob_min;
    info_out.music_prob_max = prob_max;
}

// ===========================================================================
// tonality_analysis — the big per-frame kernel
// ===========================================================================

/// Fixed-point energy scale factor.
///
/// Matches the C `#define SCALE_COMPENS (1.f/((opus_int32)1<<(15+SIG_SHIFT)))`
/// from `analysis.c:421`, with SIG_SHIFT=12 giving 1/(1<<27).
const SCALE_COMPENS: f32 = 1.0_f32 / (1_i64 << (15 + SIG_SHIFT)) as f32;

/// `SCALE_ENER(e)` macro — `(SCALE_COMPENS^2) * e` in FIXED_POINT mode.
/// Matches `analysis.c:421-422`.
#[inline(always)]
fn scale_ener(e: f32) -> f32 {
    (SCALE_COMPENS * SCALE_COMPENS) * e
}

/// Internal per-frame kernel. Called from [`run_analysis`] once the input
/// has accumulated enough samples.
///
/// Port of `tonality_analysis` in `analysis.c:445-952` (the ~500-line
/// big stateful routine). Walks through:
/// 1. Input filling + resampling into `inmem`.
/// 2. 480-pt FFT of the windowed buffer.
/// 3. Per-bin tonality / noisiness from instantaneous phase.
/// 4. Per-band energy → band_log2, band_tonality, band stationarity.
/// 5. Leakage per band for the CELT leak-boost.
/// 6. Bandwidth + noise-floor gating.
/// 7. BFCC + feature vector + std-dev tracking.
/// 8. MLP/GRU/MLP classifier invocation.
/// 9. Info fields: `bandwidth`, `tonality`, `activity`, `music_prob`,
///    `activity_probability`, `noisiness`, `valid`.
fn tonality_analysis(
    tonal: &mut TonalityAnalysisState,
    celt_mode: &CELTMode,
    x: &[u8],
    mut len: i32,
    mut offset: i32,
    c1: i32,
    c2: i32,
    c: i32,
    lsb_depth: i32,
    downmix: DownmixFunc,
) {
    // 480-pt FFT. `N = 480`, `N2 = 240` match the C reference.
    const N: usize = 480;
    const N2: usize = 240;

    // We only use the first (largest) FFT state of the celt mode's mdct
    // lookup. For the 48 kHz mode this is the 480-pt kfft.
    // celt_mode is accepted for API parity with the C call site but unused
    // until ropus supports multiple CELT modes (currently only MODE_48000_960_120
    // exists). See modes.rs:283.
    let _ = celt_mode;
    let kfft: &KissFftState = MDCT_48000_960.kfft[0];

    if tonal.initialized == 0 {
        tonal.mem_fill = 240;
        tonal.initialized = 1;
    }
    let alpha = 1.0_f32 / imin(10, 1 + tonal.count) as f32;
    let alpha_e = 1.0_f32 / imin(25, 1 + tonal.count) as f32;
    // Noise floor related decay for bandwidth detection: -2.2 dB/second
    let mut alpha_e2 = 1.0_f32 / imin(100, 1 + tonal.count) as f32;
    if tonal.count <= 1 {
        alpha_e2 = 1.0_f32;
    }

    if tonal.fs == 48_000 {
        // len and offset are now at 24 kHz.
        len /= 2;
        offset /= 2;
    } else if tonal.fs == 16_000 {
        len = 3 * len / 2;
        offset = 3 * offset / 2;
    }

    // --- First fill phase: top up inmem up to ANALYSIS_BUF_SIZE ---
    let first_fill = imin(len, ANALYSIS_BUF_SIZE as i32 - tonal.mem_fill);
    {
        let dst_start = tonal.mem_fill as usize;
        let (_, inmem_tail) = tonal.inmem.split_at_mut(dst_start);
        tonal.hp_ener_accum += downmix_and_resample(
            downmix,
            x,
            inmem_tail,
            &mut tonal.downmix_state,
            first_fill,
            offset,
            c1,
            c2,
            c,
            tonal.fs,
        ) as f32;
    }
    if tonal.mem_fill + len < ANALYSIS_BUF_SIZE as i32 {
        tonal.mem_fill += len;
        // Don't have enough to update the analysis.
        return;
    }
    let hp_ener = tonal.hp_ener_accum;

    // Claim the next slot in the ring buffer before we overwrite inmem.
    let write_idx = tonal.write_pos as usize;
    tonal.write_pos += 1;
    if tonal.write_pos >= DETECT_SIZE as i32 {
        tonal.write_pos -= DETECT_SIZE as i32;
    }

    let is_silence = is_digital_silence32(&tonal.inmem);

    // --- FFT input assembly ---
    // Real-valued 480 samples folded into 240 complex pairs. The C code
    // stores two half-window products per i into slots `i` and `N-i-1`.
    let mut fft_in = [KissFftCpx { r: 0, i: 0 }; N];
    let mut fft_out = [KissFftCpx { r: 0, i: 0 }; N];
    for i in 0..N2 {
        let w = ANALYSIS_WINDOW[i];
        // C casts are: (kiss_fft_scalar)(w * inmem[...]). kiss_fft_scalar
        // is i32 under FIXED_POINT. `float * i32 -> float -> i32` via C
        // rules. Match that exactly: the final cast is `as i32` in Rust
        // using the `round-toward-zero` truncation that matches the C
        // default (float-to-int conversion is truncation in C).
        fft_in[i].r = (w * tonal.inmem[i] as f32) as i32;
        fft_in[i].i = (w * tonal.inmem[N2 + i] as f32) as i32;
        fft_in[N - i - 1].r = (w * tonal.inmem[N - i - 1] as f32) as i32;
        fft_in[N - i - 1].i = (w * tonal.inmem[N + N2 - i - 1] as f32) as i32;
    }

    // --- OPUS_MOVE(inmem, inmem + ANALYSIS_BUF_SIZE - 240, 240);
    //     and refill the tail. ---
    {
        let src_off = ANALYSIS_BUF_SIZE - 240;
        // copy_within handles overlap correctly (memmove semantics).
        tonal.inmem.copy_within(src_off..src_off + 240, 0);
    }
    let remaining = len - (ANALYSIS_BUF_SIZE as i32 - tonal.mem_fill);
    {
        let (_, inmem_tail) = tonal.inmem.split_at_mut(240);
        tonal.hp_ener_accum = downmix_and_resample(
            downmix,
            x,
            inmem_tail,
            &mut tonal.downmix_state,
            remaining,
            offset + ANALYSIS_BUF_SIZE as i32 - tonal.mem_fill,
            c1,
            c2,
            c,
            tonal.fs,
        ) as f32;
    }
    tonal.mem_fill = 240 + remaining;

    if is_silence {
        // On silence, copy the previous analysis.
        let mut prev_pos = tonal.write_pos - 2;
        if prev_pos < 0 {
            prev_pos += DETECT_SIZE as i32;
        }
        tonal.info[write_idx] = tonal.info[prev_pos as usize];
        return;
    }

    opus_fft(kfft, &fft_in, &mut fft_out);
    // Under FIXED_POINT, kiss_fft_scalar is i32 — so no NaN check is
    // performed in C. The `celt_isnan(out[0].r)` path is only compiled
    // under the float FFT. Skip here to match.

    // Scratch arrays per-frame.
    let mut tonality = [0.0_f32; N2];
    let mut noisiness = [0.0_f32; N2];
    let mut tonality2 = [0.0_f32; N2];
    let mut band_tonality = [0.0_f32; NB_TBANDS];
    let mut log_e_band = [0.0_f32; NB_TBANDS];
    let mut bfcc = [0.0_f32; 8];
    let mut features = [0.0_f32; 25];
    let mut mid_e = [0.0_f32; 8];
    let mut band_log2 = [0.0_f32; NB_TBANDS + 1];
    let mut leakage_from = [0.0_f32; NB_TBANDS + 1];
    let mut leakage_to = [0.0_f32; NB_TBANDS + 1];
    let mut layer_out = [0.0_f32; MAX_NEURONS];
    let mut is_masked = [0_i32; NB_TBANDS + 1];

    // Local mutable refs to the state's angle trackers.
    // The C code uses OPUS_RESTRICT pointers; here we mutate tonal.angle
    // etc. directly inside the loop.

    // C: `const float pi4 = (float)(M_PI*M_PI*M_PI*M_PI);` (analysis.c:465).
    // M_PI is a bare double literal in C, so the multiplication chain runs in
    // double precision before the single cast to float. Replicate exactly.
    const PI4: f32 = (M_PI_F64 * M_PI_F64 * M_PI_F64 * M_PI_F64) as f32;

    for i in 1..N2 {
        // out[i] and out[N-i] conjugate pair. Cast each field to f32 in
        // the same order as C: `(float)out[i].r + out[N-i].r` promotes
        // the second operand via the implicit integer-to-float conv.
        let x1r = fft_out[i].r as f32 + fft_out[N - i].r as f32;
        let x1i = fft_out[i].i as f32 - fft_out[N - i].i as f32;
        let x2r = fft_out[i].i as f32 + fft_out[N - i].i as f32;
        let x2i = fft_out[N - i].r as f32 - fft_out[i].r as f32;

        // C: `angle = (float)(.5f/M_PI)*fast_atan2f(X1i, X1r);` (analysis.c:581).
        // `.5f` is float, `M_PI` is a bare double literal, so the division
        // promotes to double and the `(float)` cast rounds at the end.
        // Replicate exactly to match the bit pattern.
        const INV_TWO_PI: f32 = (0.5_f64 / M_PI_F64) as f32;
        let angle = INV_TWO_PI * fast_atan2f(x1i, x1r);
        let d_angle = angle - tonal.angle[i];
        let d2_angle = d_angle - tonal.d_angle[i];

        let angle2 = INV_TWO_PI * fast_atan2f(x2i, x2r);
        let d_angle2 = angle2 - angle;
        let d2_angle2 = d_angle2 - d_angle;

        let mut mod1 = d2_angle - float2int(d2_angle) as f32;
        noisiness[i] = fabs16(mod1);
        mod1 *= mod1;
        mod1 *= mod1;

        let mut mod2 = d2_angle2 - float2int(d2_angle2) as f32;
        noisiness[i] += fabs16(mod2);
        mod2 *= mod2;
        mod2 *= mod2;

        let avg_mod = 0.25_f32 * (tonal.d2_angle[i] + mod1 + 2.0_f32 * mod2);
        // This introduces an extra delay of 2 frames in the detection.
        tonality[i] = 1.0_f32 / (1.0_f32 + 40.0_f32 * 16.0_f32 * PI4 * avg_mod) - 0.015_f32;
        // No delay on this detection, but it's less reliable.
        tonality2[i] = 1.0_f32 / (1.0_f32 + 40.0_f32 * 16.0_f32 * PI4 * mod2) - 0.015_f32;

        tonal.angle[i] = angle2;
        tonal.d_angle[i] = d_angle2;
        tonal.d2_angle[i] = mod2;
    }
    for i in 2..(N2 - 1) {
        let tt = fmin32(tonality2[i], fmax32(tonality2[i - 1], tonality2[i + 1]));
        tonality[i] = 0.9_f32 * fmax32(tonality[i], tt - 0.1_f32);
    }

    let mut frame_tonality: f32 = 0.0;
    let mut max_frame_tonality: f32 = 0.0;
    let info_slot = write_idx;
    tonal.info[info_slot].activity = 0.0;
    let mut frame_noisiness: f32 = 0.0;
    let mut frame_stationarity: f32 = 0.0;
    if tonal.count == 0 {
        for b in 0..NB_TBANDS {
            tonal.low_e[b] = 1e10_f32;
            tonal.high_e[b] = -1e10_f32;
        }
    }
    let mut relative_e: f32 = 0.0;
    let mut frame_loudness: f32 = 0.0;

    // The energy of the very first band is special because of DC.
    {
        let x1r = 2.0_f32 * fft_out[0].r as f32;
        let x2r = 2.0_f32 * fft_out[0].i as f32;
        let mut e = x1r * x1r + x2r * x2r;
        for i in 1..4 {
            let bin_e = fft_out[i].r as f32 * fft_out[i].r as f32
                + fft_out[N - i].r as f32 * fft_out[N - i].r as f32
                + fft_out[i].i as f32 * fft_out[i].i as f32
                + fft_out[N - i].i as f32 * fft_out[N - i].i as f32;
            e += bin_e;
        }
        e = scale_ener(e);
        // C: `.5f * 1.442695f * (float)log(E + 1e-10f)`. `log` is a
        // double function; the argument promotes to double, log is
        // computed in double precision, then cast back to float.
        // Replicate the same chain so the final float bit-pattern
        // matches.
        band_log2[0] = 0.5_f32 * 1.442695_f32 * ((e + 1e-10_f32) as f64).ln() as f32;
    }

    for b in 0..NB_TBANDS {
        let mut e: f32 = 0.0;
        let mut t_e: f32 = 0.0;
        let mut n_e: f32 = 0.0;
        let start = TBANDS[b] as usize;
        let end = TBANDS[b + 1] as usize;
        for i in start..end {
            let bin_e = fft_out[i].r as f32 * fft_out[i].r as f32
                + fft_out[N - i].r as f32 * fft_out[N - i].r as f32
                + fft_out[i].i as f32 * fft_out[i].i as f32
                + fft_out[N - i].i as f32 * fft_out[N - i].i as f32;
            let bin_e = scale_ener(bin_e);
            e += bin_e;
            t_e += bin_e * fmax32(0.0_f32, tonality[i]);
            n_e += bin_e * 2.0_f32 * (0.5_f32 - noisiness[i]);
        }
        // NOTE: Under FIXED_POINT the C reference does NOT perform the
        // `!(E<1e9f)||isnan(E)` NaN guard — it's float-mode only. Match
        // that and skip the guard here.

        tonal.e_frames[tonal.e_count as usize][b] = e;
        frame_noisiness += n_e / (1e-15_f32 + e);

        // C: `(float)sqrt(E + 1e-10f)`. `sqrt` is a double function in
        // C, but IEEE-754 guarantees `(float)sqrt((double)x) ==
        // sqrtf(x)` exactly, so `.sqrt()` on f32 is bit-exact.
        frame_loudness += (e + 1e-10_f32).sqrt();
        // C: `(float)log(E + 1e-10f)` — promote to double, log in
        // double, cast back. Rust `f32::ln` is single-precision and
        // can differ by ULPs from this chain, so go via f64 explicitly.
        log_e_band[b] = ((e + 1e-10_f32) as f64).ln() as f32;
        band_log2[b + 1] =
            0.5_f32 * 1.442695_f32 * ((e + 1e-10_f32) as f64).ln() as f32;
        tonal.log_e_frames[tonal.e_count as usize][b] = log_e_band[b];
        if tonal.count == 0 {
            tonal.high_e[b] = log_e_band[b];
            tonal.low_e[b] = log_e_band[b];
        }
        // C: `tonal->highE[b] > tonal->lowE[b] + 7.5`. `7.5` is a
        // double literal (no `f` suffix), so the `+` promotes lowE to
        // double, and the `>` promotes highE to double. Match that to
        // avoid decision flips on edge cases.
        if (tonal.high_e[b] as f64) > (tonal.low_e[b] as f64 + 7.5_f64) {
            if tonal.high_e[b] - log_e_band[b] > log_e_band[b] - tonal.low_e[b] {
                tonal.high_e[b] -= 0.01_f32;
            } else {
                tonal.low_e[b] += 0.01_f32;
            }
        }
        if log_e_band[b] > tonal.high_e[b] {
            tonal.high_e[b] = log_e_band[b];
            tonal.low_e[b] = fmax32(tonal.high_e[b] - 15.0_f32, tonal.low_e[b]);
        } else if log_e_band[b] < tonal.low_e[b] {
            tonal.low_e[b] = log_e_band[b];
            tonal.high_e[b] = fmin32(tonal.low_e[b] + 15.0_f32, tonal.high_e[b]);
        }
        relative_e += (log_e_band[b] - tonal.low_e[b])
            / (1e-5_f32 + (tonal.high_e[b] - tonal.low_e[b]));

        let mut l1: f32 = 0.0;
        let mut l2: f32 = 0.0;
        for i in 0..NB_FRAMES {
            l1 += tonal.e_frames[i][b].sqrt();
            l2 += tonal.e_frames[i][b];
        }

        // C: `MIN16(0.99f, L1/(float)sqrt(1e-15+NB_FRAMES*L2))`.
        // `1e-15` has no `f` suffix — it's a double literal. The sum
        // promotes to double, sqrt is computed in double, cast back
        // to float. Mirror that chain exactly.
        let denom = ((1e-15_f64 + (NB_FRAMES as f32 * l2) as f64).sqrt()) as f32;
        let mut stationarity = fmin16(0.99_f32, l1 / denom);
        stationarity *= stationarity;
        stationarity *= stationarity;
        frame_stationarity += stationarity;
        band_tonality[b] = fmax16(
            t_e / (1e-15_f32 + e),
            stationarity * tonal.prev_band_tonality[b],
        );
        frame_tonality += band_tonality[b];
        if b >= NB_TBANDS - NB_TONAL_SKIP_BANDS {
            frame_tonality -= band_tonality[b - NB_TBANDS + NB_TONAL_SKIP_BANDS];
        }
        max_frame_tonality = fmax16(
            max_frame_tonality,
            (1.0_f32 + 0.03_f32 * (b as f32 - NB_TBANDS as f32)) * frame_tonality,
        );
        // C also does `slope += band_tonality[b]*(b-8);` here. We pull
        // that out into a dedicated loop below so the in-loop body
        // stays manageable; see the note there for why that's safe.
        tonal.prev_band_tonality[b] = band_tonality[b];
    }

    // Second pass to accumulate `slope`. In C this is folded into the
    // main `for b` loop as `slope += band_tonality[b]*(b-8)`. `slope`
    // has no in-loop data dependency with any other accumulator, so
    // splitting it out is bit-exact: both the summation order and the
    // `band_tonality` values are identical to the inlined C version.
    let mut slope: f32 = 0.0;
    for b in 0..NB_TBANDS {
        slope += band_tonality[b] * (b as f32 - 8.0_f32);
    }

    // Leakage computation.
    leakage_from[0] = band_log2[0];
    leakage_to[0] = band_log2[0] - LEAKAGE_OFFSET;
    for b in 1..(NB_TBANDS + 1) {
        let leak_slope =
            LEAKAGE_SLOPE * (TBANDS[b] - TBANDS[b - 1]) as f32 / 4.0_f32;
        leakage_from[b] = fmin16(leakage_from[b - 1] + leak_slope, band_log2[b]);
        leakage_to[b] = fmax16(leakage_to[b - 1] - leak_slope, band_log2[b] - LEAKAGE_OFFSET);
    }
    // C loops `for (b=NB_TBANDS-2; b>=0; b--)` — reverse Rust range.
    for b in (0..=(NB_TBANDS - 2)).rev() {
        let leak_slope =
            LEAKAGE_SLOPE * (TBANDS[b + 1] - TBANDS[b]) as f32 / 4.0_f32;
        leakage_from[b] = fmin16(leakage_from[b + 1] + leak_slope, leakage_from[b]);
        leakage_to[b] = fmax16(leakage_to[b + 1] - leak_slope, leakage_to[b]);
    }
    // Structural invariant: leak_boost is sized LEAK_BANDS (=19) and
    // we must not write past NB_TBANDS+1 (=19) entries. The assertion
    // lives as a compile-time check.
    const _: () = assert!(NB_TBANDS + 1 <= LEAK_BANDS);
    for b in 0..(NB_TBANDS + 1) {
        let boost = fmax16(0.0_f32, leakage_to[b] - band_log2[b])
            + fmax16(0.0_f32, band_log2[b] - (leakage_from[b] + LEAKAGE_OFFSET));
        let packed = (0.5_f64 + 64.0_f64 * boost as f64).floor() as i32;
        let clamped = if packed < 0 { 0 } else if packed > 255 { 255 } else { packed };
        tonal.info[info_slot].leak_boost[b] = clamped as u8;
    }
    // C: `for (;b<LEAK_BANDS;b++) info->leak_boost[b] = 0;`. With the
    // current constants (NB_TBANDS=18, LEAK_BANDS=19) this range is
    // empty, but the loop is kept for structural equivalence in case
    // the tables are extended upstream.
    #[allow(clippy::reversed_empty_ranges)]
    for b in (NB_TBANDS + 1)..LEAK_BANDS {
        tonal.info[info_slot].leak_boost[b] = 0;
    }

    // Spectral variability: nearest-neighbour distance in log-energy space.
    let mut spec_variability: f32 = 0.0;
    for i in 0..NB_FRAMES {
        let mut min_dist: f32 = 1e15_f32;
        for j in 0..NB_FRAMES {
            let mut dist: f32 = 0.0;
            for k in 0..NB_TBANDS {
                let tmp = tonal.log_e_frames[i][k] - tonal.log_e_frames[j][k];
                dist += tmp * tmp;
            }
            if j != i {
                min_dist = fmin32(min_dist, dist);
            }
        }
        spec_variability += min_dist;
    }
    spec_variability = (spec_variability / NB_FRAMES as f32 / NB_TBANDS as f32).sqrt();

    let mut bandwidth_mask: f32 = 0.0;
    let mut bandwidth: i32 = 0;
    let mut max_e: f32 = 0.0;
    // 5.7e-4 / (1 << max(0, lsb_depth-8)), squared.
    let mut noise_floor = 5.7e-4_f32 / (1_u32 << imax(0, lsb_depth - 8)) as f32;
    noise_floor *= noise_floor;
    let mut below_max_pitch: f32 = 0.0;
    let mut above_max_pitch: f32 = 0.0;
    for b in 0..NB_TBANDS {
        let band_start = TBANDS[b] as usize;
        let band_end = TBANDS[b + 1] as usize;
        let mut e: f32 = 0.0;
        for i in band_start..band_end {
            let bin_e = fft_out[i].r as f32 * fft_out[i].r as f32
                + fft_out[N - i].r as f32 * fft_out[N - i].r as f32
                + fft_out[i].i as f32 * fft_out[i].i as f32
                + fft_out[N - i].i as f32 * fft_out[N - i].i as f32;
            e += bin_e;
        }
        e = scale_ener(e);
        max_e = fmax32(max_e, e);
        if band_start < 64 {
            below_max_pitch += e;
        } else {
            above_max_pitch += e;
        }
        tonal.mean_e[b] = fmax32((1.0_f32 - alpha_e2) * tonal.mean_e[b], e);
        let em = fmax32(e, tonal.mean_e[b]);
        let width = (band_end - band_start) as f32;
        if e * 1e9_f32 > max_e
            && (em > 3.0_f32 * noise_floor * width || e > noise_floor * width)
        {
            bandwidth = b as i32 + 1;
        }
        let masked_thresh = if tonal.prev_bandwidth >= (b as i32 + 1) {
            0.01_f32
        } else {
            0.05_f32
        };
        is_masked[b] = (e < masked_thresh * bandwidth_mask) as i32;
        bandwidth_mask = fmax32(0.05_f32 * bandwidth_mask, e);
    }

    // Special case for the last two bands (above 12 kHz) at Fs=48000.
    //
    // In the C reference, the loop index `b` post-loop equals `NB_TBANDS`
    // (the NB_TBANDS-iteration `for` completes with `b == NB_TBANDS`).
    // The block below reuses that `b`, writing `mean_e[NB_TBANDS]` and
    // `is_masked[NB_TBANDS]`. We mirror those indices explicitly.
    if tonal.fs == 48_000 {
        let noise_ratio: f32 = if tonal.prev_bandwidth == 20 { 10.0_f32 } else { 30.0_f32 };
        let mut e = hp_ener * (1.0_f32 / (60.0_f32 * 60.0_f32));
        // silk_resampler_down2_hp() shifted right by an extra 8 bits.
        e *= 256.0_f32 * (1.0_f32 / Q15ONE_F32) * (1.0_f32 / Q15ONE_F32);
        above_max_pitch += e;
        tonal.mean_e[NB_TBANDS] = fmax32((1.0_f32 - alpha_e2) * tonal.mean_e[NB_TBANDS], e);
        let em = fmax32(e, tonal.mean_e[NB_TBANDS]);
        if em > 3.0_f32 * noise_ratio * noise_floor * 160.0_f32
            || e > noise_ratio * noise_floor * 160.0_f32
        {
            bandwidth = 20;
        }
        let masked_thresh = if tonal.prev_bandwidth == 20 { 0.01_f32 } else { 0.05_f32 };
        is_masked[NB_TBANDS] = (e < masked_thresh * bandwidth_mask) as i32;
    }

    if above_max_pitch > below_max_pitch {
        tonal.info[info_slot].max_pitch_ratio = below_max_pitch / above_max_pitch;
    } else {
        tonal.info[info_slot].max_pitch_ratio = 1.0_f32;
    }
    // Final masking cleanup on the candidate bandwidth.
    if bandwidth == 20 && is_masked[NB_TBANDS] != 0 {
        bandwidth -= 2;
    } else if bandwidth > 0 && bandwidth <= NB_TBANDS as i32 && is_masked[(bandwidth - 1) as usize] != 0 {
        bandwidth -= 1;
    }
    if tonal.count <= 2 {
        bandwidth = 20;
    }
    // C: `20*(float)log10(frame_loudness)`. `log10` is a double
    // function; promote to double, log10 in double, cast back to
    // float. Rust's f32 log10 uses single-precision and can differ by
    // ULPs.
    frame_loudness = 20.0_f32 * (frame_loudness as f64).log10() as f32;
    tonal.e_tracker = fmax32(tonal.e_tracker - 0.003_f32, frame_loudness);
    tonal.low_e_count *= 1.0_f32 - alpha_e;
    if frame_loudness < tonal.e_tracker - 30.0_f32 {
        tonal.low_e_count += alpha_e;
    }

    // BFCCs from logE via the first 8 rows of DCT_TABLE.
    for i in 0..8 {
        let mut sum: f32 = 0.0;
        for b in 0..16 {
            sum += DCT_TABLE[i * 16 + b] * log_e_band[b];
        }
        bfcc[i] = sum;
    }
    for i in 0..8 {
        let mut sum: f32 = 0.0;
        for b in 0..16 {
            sum += DCT_TABLE[i * 16 + b] * 0.5_f32 * (tonal.high_e[b] + tonal.low_e[b]);
        }
        mid_e[i] = sum;
    }

    frame_stationarity /= NB_TBANDS as f32;
    relative_e /= NB_TBANDS as f32;
    if tonal.count < 10 {
        relative_e = 0.5_f32;
    }
    frame_noisiness /= NB_TBANDS as f32;
    tonal.info[info_slot].activity =
        frame_noisiness + (1.0_f32 - frame_noisiness) * relative_e;
    frame_tonality = max_frame_tonality / (NB_TBANDS - NB_TONAL_SKIP_BANDS) as f32;
    frame_tonality = fmax16(frame_tonality, tonal.prev_tonality * 0.8_f32);
    tonal.prev_tonality = frame_tonality;

    let slope = slope / (8.0_f32 * 8.0_f32);
    tonal.info[info_slot].tonality_slope = slope;

    tonal.e_count = (tonal.e_count + 1) % NB_FRAMES as i32;
    tonal.count = imin(tonal.count + 1, ANALYSIS_COUNT_MAX);
    tonal.info[info_slot].tonality = frame_tonality;

    // Feature[0..4]: current BFCC delta vs mem history + cmean bias.
    for i in 0..4 {
        features[i] = -0.12299_f32 * (bfcc[i] + tonal.mem[i + 24])
            + 0.49195_f32 * (tonal.mem[i] + tonal.mem[i + 16])
            + 0.69693_f32 * tonal.mem[i + 8]
            - 1.4349_f32 * tonal.cmean[i];
    }
    for i in 0..4 {
        tonal.cmean[i] = (1.0_f32 - alpha) * tonal.cmean[i] + alpha * bfcc[i];
    }

    for i in 0..4 {
        features[4 + i] =
            0.63246_f32 * (bfcc[i] - tonal.mem[i + 24]) + 0.31623_f32 * (tonal.mem[i] - tonal.mem[i + 16]);
    }
    for i in 0..3 {
        features[8 + i] = 0.53452_f32 * (bfcc[i] + tonal.mem[i + 24])
            - 0.26726_f32 * (tonal.mem[i] + tonal.mem[i + 16])
            - 0.53452_f32 * tonal.mem[i + 8];
    }

    if tonal.count > 5 {
        for i in 0..9 {
            tonal.std[i] = (1.0_f32 - alpha) * tonal.std[i] + alpha * features[i] * features[i];
        }
    }
    for i in 0..4 {
        features[i] = bfcc[i] - mid_e[i];
    }

    // Shift `mem` one "column" so next frame sees the history. The C
    // reference writes index 24 with 16, 16 with 8, 8 with 0, and 0 with
    // the current BFCC. Mirror that order so the aliasing is identical.
    for i in 0..8 {
        tonal.mem[i + 24] = tonal.mem[i + 16];
        tonal.mem[i + 16] = tonal.mem[i + 8];
        tonal.mem[i + 8] = tonal.mem[i];
        tonal.mem[i] = bfcc[i];
    }
    for i in 0..9 {
        features[11 + i] = tonal.std[i].sqrt() - STD_FEATURE_BIAS[i];
    }
    features[18] = spec_variability - 0.78_f32;
    features[20] = tonal.info[info_slot].tonality - 0.154723_f32;
    features[21] = tonal.info[info_slot].activity - 0.724643_f32;
    features[22] = frame_stationarity - 0.743717_f32;
    features[23] = tonal.info[info_slot].tonality_slope + 0.069216_f32;
    features[24] = tonal.low_e_count - 0.067930_f32;

    analysis_compute_dense(&LAYER0, &mut layer_out, &features);
    analysis_compute_gru(&LAYER1, &mut tonal.rnn_state, &layer_out);
    let mut frame_probs = [0.0_f32; 2];
    analysis_compute_dense(&LAYER2, &mut frame_probs, &tonal.rnn_state);

    // Probability of speech or music vs noise.
    tonal.info[info_slot].activity_probability = frame_probs[1];
    tonal.info[info_slot].music_prob = frame_probs[0];

    tonal.info[info_slot].bandwidth = bandwidth;
    tonal.prev_bandwidth = bandwidth;
    tonal.info[info_slot].noisiness = frame_noisiness;
    tonal.info[info_slot].valid = 1;
}

/// Q15ONE as a float, used in the Fs=48000 HP-energy normalization.
/// Matches the Q15ONE fixed-point value of 32767.
const Q15ONE_F32: f32 = 32_767.0_f32;

// ===========================================================================
// run_analysis — public entry point
// ===========================================================================

/// Drive the analyzer over a frame of PCM.
///
/// Port of `run_analysis` in `analysis.c:954-980`. Iterates
/// `tonality_analysis` over consecutive 2.5ms subframes, updates
/// `analysis_offset`, then reads out the current `AnalysisInfo` via
/// [`tonality_get_info`].
///
/// `analysis_pcm` is the raw PCM buffer as bytes; the `downmix` callback
/// is responsible for parsing it (float, i16, i24). `fs` is the external
/// (caller) sample rate, `frame_size` is the encoder's frame in samples
/// at that rate.
pub fn run_analysis(
    analysis: &mut TonalityAnalysisState,
    celt_mode: &CELTMode,
    analysis_pcm: Option<&[u8]>,
    mut analysis_frame_size: i32,
    frame_size: i32,
    c1: i32,
    c2: i32,
    c: i32,
    fs: i32,
    lsb_depth: i32,
    downmix: DownmixFunc,
    analysis_info: &mut AnalysisInfo,
) {
    // Even-align.
    analysis_frame_size -= analysis_frame_size & 1;
    if let Some(pcm) = analysis_pcm {
        // Avoid overflow/wrap-around of the analysis buffer.
        analysis_frame_size = imin((DETECT_SIZE as i32 - 5) * fs / 50, analysis_frame_size);

        let mut pcm_len = analysis_frame_size - analysis.analysis_offset;
        let mut offset = analysis.analysis_offset;
        while pcm_len > 0 {
            tonality_analysis(
                analysis,
                celt_mode,
                pcm,
                imin(fs / 50, pcm_len),
                offset,
                c1,
                c2,
                c,
                lsb_depth,
                downmix,
            );
            offset += fs / 50;
            pcm_len -= fs / 50;
        }
        analysis.analysis_offset = analysis_frame_size;
        analysis.analysis_offset -= frame_size;
    }

    tonality_get_info(analysis, analysis_info, frame_size);
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celt::modes::MODE_48000_960_120;

    /// A trivial downmix callback that interprets the PCM bytes as
    /// `f32` little-endian (matching the default build on x86), applies
    /// `FLOAT2SIG` (multiply by 32768), and sums the selected channels.
    fn test_downmix_float(
        input: &[u8],
        output: &mut [i32],
        subframe: i32,
        offset: i32,
        c1: i32,
        c2: i32,
        c: i32,
    ) {
        // Interpret bytes as f32 samples, interleaved by `c` channels.
        let samples: &[f32] = unsafe {
            core::slice::from_raw_parts(
                input.as_ptr() as *const f32,
                input.len() / core::mem::size_of::<f32>(),
            )
        };
        const CELT_SIG_SCALE: f32 = 32_768.0;
        for j in 0..subframe as usize {
            output[j] = (samples[((j as i32 + offset) * c + c1) as usize] * CELT_SIG_SCALE) as i32;
        }
        if c2 > -1 {
            for j in 0..subframe as usize {
                output[j] += (samples[((j as i32 + offset) * c + c2) as usize] * CELT_SIG_SCALE)
                    as i32;
            }
        } else if c2 == -2 {
            for ch in 1..c {
                for j in 0..subframe as usize {
                    output[j] += (samples[((j as i32 + offset) * c + ch) as usize] * CELT_SIG_SCALE)
                        as i32;
                }
            }
        }
        // Fixed-point mode: skip the +/-65536 clamp from the float path.
    }

    fn f32_to_bytes(samples: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(samples.len() * 4);
        for &s in samples {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }

    #[test]
    fn test_new_boxed_is_equivalent_to_new() {
        let stack = TonalityAnalysisState::new();
        let heap = TonalityAnalysisState::new_boxed();
        // Bit-for-bit compare — both should be all-zero.
        let stack_bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(
                (&stack as *const TonalityAnalysisState) as *const u8,
                core::mem::size_of::<TonalityAnalysisState>(),
            )
        };
        let heap_bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(
                (heap.as_ref() as *const TonalityAnalysisState) as *const u8,
                core::mem::size_of::<TonalityAnalysisState>(),
            )
        };
        assert_eq!(stack_bytes, heap_bytes, "new_boxed must byte-match new()");
        // Sanity: every byte is zero (both constructors promise this).
        assert!(heap_bytes.iter().all(|&b| b == 0), "state must be zeroed");
    }

    #[test]
    fn test_tonality_analysis_init_populates_state() {
        let mut state = TonalityAnalysisState::new();
        tonality_analysis_init(&mut state, 48_000);

        // Preserved (pre-reset-start) fields.
        assert_eq!(state.fs, 48_000, "Fs should be set by init");
        assert_eq!(state.arch, 0, "arch should be scalar (0)");

        // Post-reset-start: everything should be zero / default.
        assert_eq!(state.count, 0);
        assert_eq!(state.mem_fill, 0);
        assert_eq!(state.write_pos, 0);
        assert_eq!(state.read_pos, 0);
        assert_eq!(state.read_subframe, 0);
        assert_eq!(state.initialized, 0);
        assert_eq!(state.e_count, 0);
        assert_eq!(state.analysis_offset, 0);
        assert_eq!(state.prev_bandwidth, 0);
        assert_eq!(state.e_tracker, 0.0);
        assert_eq!(state.low_e_count, 0.0);
        assert_eq!(state.prev_tonality, 0.0);
        assert_eq!(state.hp_ener_accum, 0.0);
        assert!(state.angle.iter().all(|&v| v == 0.0));
        assert!(state.d_angle.iter().all(|&v| v == 0.0));
        assert!(state.d2_angle.iter().all(|&v| v == 0.0));
        assert!(state.inmem.iter().all(|&v| v == 0));
        assert!(state.rnn_state.iter().all(|&v| v == 0.0));
        assert!(state.downmix_state.iter().all(|&v| v == 0));
        assert!(state.info.iter().all(|info| info.valid == 0));
    }

    #[test]
    fn test_tonality_analysis_reset_clears_running_fields() {
        let mut state = TonalityAnalysisState::new();
        tonality_analysis_init(&mut state, 48_000);

        // Manually poke a handful of post-reset fields.
        state.count = 42;
        state.mem_fill = 123;
        state.write_pos = 7;
        state.read_pos = 3;
        state.initialized = 1;
        state.prev_tonality = 0.5;
        state.prev_bandwidth = 4;
        state.e_tracker = 12.0;
        state.angle[0] = 1.25;
        state.d_angle[42] = -0.5;
        state.inmem[100] = 0x1234;
        state.rnn_state[5] = 7.5;
        state.downmix_state[1] = 4242;
        state.info[0].valid = 1;
        state.info[0].music_prob = 0.77;
        state.info[50].bandwidth = 20;

        // Also poke a pre-reset field so we can confirm it survives.
        // (arch is normally written by init, but we're simulating an
        // already-initialized state where it's a real nonzero value.)
        state.arch = 99;
        state.application = 2049;

        tonality_analysis_reset(&mut state);

        // Preserved (pre-reset-start).
        assert_eq!(state.arch, 99, "arch must survive reset");
        assert_eq!(state.application, 2049, "application must survive reset");
        assert_eq!(state.fs, 48_000, "Fs must survive reset");

        // Everything else zero'd.
        assert_eq!(state.count, 0);
        assert_eq!(state.mem_fill, 0);
        assert_eq!(state.write_pos, 0);
        assert_eq!(state.read_pos, 0);
        assert_eq!(state.initialized, 0);
        assert_eq!(state.prev_tonality, 0.0);
        assert_eq!(state.prev_bandwidth, 0);
        assert_eq!(state.e_tracker, 0.0);
        assert_eq!(state.angle[0], 0.0);
        assert_eq!(state.d_angle[42], 0.0);
        assert_eq!(state.inmem[100], 0);
        assert_eq!(state.rnn_state[5], 0.0);
        assert_eq!(state.downmix_state[1], 0);
        assert!(state.info.iter().all(|info| info.valid == 0));
    }

    #[test]
    fn test_run_analysis_smoke_no_panic() {
        // 480 samples of silence at 48 kHz mono. Caller rate = Fs, so
        // `run_analysis` should fire tonality_analysis at 2.5ms subframe
        // granularity (Fs/400 = 120 samples per subframe at 48 kHz, but
        // the internal rate after downsample is 24 kHz = 60 samples / sf).
        let mut state = TonalityAnalysisState::new();
        tonality_analysis_init(&mut state, 48_000);

        let pcm: Vec<f32> = vec![0.0; 480];
        let pcm_bytes = f32_to_bytes(&pcm);

        let mut info = AnalysisInfo::zeroed();
        run_analysis(
            &mut state,
            &MODE_48000_960_120,
            Some(&pcm_bytes),
            480, // analysis_frame_size (whole 10 ms of silence)
            480, // frame_size
            0,   // c1
            -2,  // c2 (sum-remaining, sentinel)
            1,   // C (mono)
            48_000,
            16, // lsb_depth
            test_downmix_float,
            &mut info,
        );

        // First call: the analyzer has just enough input to start the
        // ring buffer (720 samples at 24 kHz is 30 ms, we fed 10 ms at
        // 48 kHz = 10 ms of internal-rate samples). Either way, info is
        // either invalid (not enough data yet) or marked as silence-fill.
        // We specifically check no panic + sane field ranges.
        assert!(
            info.valid == 0 || info.valid == 1,
            "valid flag should be 0 or 1, got {}",
            info.valid
        );
        if info.valid == 1 {
            assert!(
                (0.0..=1.0).contains(&info.music_prob),
                "music_prob {} out of range",
                info.music_prob
            );
            assert!(
                (0.0..=1.0).contains(&info.activity_probability),
                "activity_probability {} out of range",
                info.activity_probability
            );
        }
    }

    #[test]
    fn test_run_analysis_smoke_with_signal() {
        // 480 samples of a 1 kHz sine at 48 kHz mono. Not enough to fill
        // the 30 ms inmem buffer by itself, but exercises the FFT path
        // once we push multiple frames in.
        let mut state = TonalityAnalysisState::new();
        tonality_analysis_init(&mut state, 48_000);

        let freq: f32 = 1_000.0;
        let fs: f32 = 48_000.0;
        // Feed several frames so we're guaranteed to trigger the FFT
        // branch at least once (inmem fills when mem_fill + len >= 720,
        // i.e. after ~3 x 480-sample frames at 48 kHz since each drops
        // to 240 internal samples).
        for frame in 0..4 {
            let start = (frame * 480) as f32;
            let pcm: Vec<f32> = (0..480)
                .map(|i| 0.5 * (2.0 * core::f32::consts::PI * freq * (start + i as f32) / fs).sin())
                .collect();
            let pcm_bytes = f32_to_bytes(&pcm);

            let mut info = AnalysisInfo::zeroed();
            run_analysis(
                &mut state,
                &MODE_48000_960_120,
                Some(&pcm_bytes),
                480,
                480,
                0,
                -2,
                1,
                48_000,
                16,
                test_downmix_float,
                &mut info,
            );
            // We don't assert on specific values — we just want no panic
            // and fields in sane ranges whenever valid is set.
            if info.valid == 1 {
                assert!(info.music_prob.is_finite());
                assert!(info.activity_probability.is_finite());
                assert!(info.tonality.is_finite());
                assert!(info.bandwidth >= 0 && info.bandwidth <= 20);
                assert!(info.max_pitch_ratio.is_finite());
            }
        }
    }
}
