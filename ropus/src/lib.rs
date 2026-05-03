//! A Rust port of the xiph Opus audio codec (fixed-point), bit-exact against the
//! reference implementation.
//!
//! # Stability
//!
//! The stable public API consists of the items re-exported at the crate root
//! (e.g. [`OpusEncoder`], [`OpusDecoder`], [`OpusMSEncoder`], [`OpusMSDecoder`],
//! [`OpusRepacketizer`], and the `OPUS_*` constants). Items reachable via module
//! paths (`ropus::opus::*`, `ropus::celt::*`, `ropus::silk::*`, `ropus::dnn::*`)
//! are accessible but are **not** subject to semver stability guarantees pre-1.0;
//! a future 1.0 release will tighten this surface.
//!
//! # Example
//!
//! Encode 20 ms of silence at 48 kHz mono in VOIP mode and decode it back:
//!
//! ```
//! use ropus::{Application, Channels, DecodeMode, Decoder, Encoder};
//!
//! let mut encoder = Encoder::builder(48_000, Channels::Mono, Application::Voip)
//!     .build()
//!     .unwrap();
//! let pcm_in = [0i16; 960]; // 20 ms at 48 kHz mono
//! let mut packet = [0u8; 4000];
//! let len = encoder.encode(&pcm_in, &mut packet).unwrap();
//!
//! let mut decoder = Decoder::new(48_000, Channels::Mono).unwrap();
//! let mut pcm_out = [0i16; 960];
//! let samples = decoder.decode(&packet[..len], &mut pcm_out, DecodeMode::Normal).unwrap();
//! assert_eq!(samples, 960);
//! ```
//!
//! # API layering
//!
//! `ropus` exposes two API tiers:
//!
//! - **High-level** ([`Encoder`], [`Decoder`]): typed enums ([`DecodeMode`],
//!   [`Channels`], [`Bitrate`], ...), `Result<_, …Error>` error types, hidden
//!   `i32` codes. Use this for new Rust code.
//! - **Low-level** (`ropus::opus::*`, `ropus::celt::*`, `ropus::silk::*`,
//!   including [`OpusDecoder`], [`OpusMSDecoder`]): mirrors the libopus C ABI
//!   verbatim — `bool decode_fec`, `Result<i32, i32>`, raw `i32` channel
//!   counts. Stable so the `capi/` crate can present a byte-identical FFI.
//!
//! The low-level layer intentionally retains `bool` for `decode_fec` and bare
//! `i32` errors. There is no high-level multistream facade today; if one is
//! added later it will live alongside [`Encoder`] / [`Decoder`] and use
//! [`DecodeMode`] like [`Decoder`] does.

// Clippy allows for C-to-Rust codec port patterns.
// This is a bit-exact port of xiph/opus; these patterns are intentional.
#![allow(
    // C reference uses exact float literals (approximation coefficients, fixed-point
    // conversion inputs) that must be preserved for bit-exactness.
    clippy::excessive_precision,
    clippy::approx_constant,
    // C-style explicit casts kept for clarity and 1:1 correspondence with reference.
    clippy::unnecessary_cast,
    // Unrolled loops and table indexing use `+ 0`, `<< 0`, `0 * stride` for pattern
    // clarity with subsequent iterations.
    clippy::identity_op,
    clippy::erasing_op,
    // C codec functions have many parameters; matching the reference API exactly.
    clippy::too_many_arguments,
    // C-style operator precedence preserved for readability against reference.
    clippy::precedence,
    // C-style range checks and control flow preserved for 1:1 correspondence.
    clippy::manual_range_contains,
    clippy::collapsible_if,
    clippy::collapsible_else_if,
    // C-style manual slice copies with computed indices.
    clippy::manual_memcpy,
    // Port uses Result<_, ()> for C-style error returns.
    clippy::result_unit_err,
    // C-style indexed loops preserved for 1:1 reference correspondence.
    clippy::needless_range_loop,
    // C-style min/max chains preserved for bit-exactness verification.
    clippy::manual_clamp,
    // C-style variable declarations (declare then assign in branches).
    clippy::needless_late_init,
    // C-style assign patterns (x = x + y) preserved for reference readability.
    clippy::assign_op_pattern,
    // C-style boolean and comparison patterns.
    clippy::int_plus_one,
    clippy::nonminimal_bool,
    // C-style let-then-return for clarity in complex expressions.
    clippy::let_and_return,
    // Codec types have complex initialization; new() is preferred over Default.
    clippy::new_without_default,
    // C-style manual loop counters preserved for reference correspondence.
    clippy::explicit_counter_loop,
)]

pub mod celt;
pub mod dnn;
pub mod opus;
pub mod silk;
pub mod types;

#[cfg(feature = "trace-silk-encode")]
pub mod silk_trace {
    //! Phase B encode-side trace tuples (Cluster A stage 2b). Gated
    //! entirely behind the `trace-silk-encode` Cargo feature; with the
    //! feature off (the default), this module compiles to nothing and
    //! the SILK encoder emits no trace calls.
    //!
    //! See `wrk_docs/2026.05.02 - HLD - encoder-state-accumulation-fix.md`
    //! §2.3 for the boundary semantics. The diagnostic binary
    //! `harness/src/bin/fuzz_repro_diff.rs` drains this buffer after
    //! each encode and diffs against the C-side FIFO populated by
    //! `harness/silk_enc_api_traced.c`.

    use std::sync::Mutex;

    /// One trace record emitted at a boundary inside the SILK encoder.
    ///
    /// V1 (`boundary_id` 1..=7) carries only the scalar header fields and
    /// leaves `iter = -1` / `payload = []`. V2 (Phase C, `boundary_id`
    /// 100..=109 — see HLD V2 §4.1) reuses the same record, populating
    /// `iter` for per-rate-control-loop boundaries and `payload` with the
    /// full state vector for the boundary. The `boundary_id < 100` vs
    /// `>= 100` split discriminates V1 vs V2 records; the `iter = -1`
    /// sentinel ("not in rate-control loop") covers both V1 records and
    /// V2 setup boundaries (100..=105) which fire once per encode and
    /// must match the C side's `iter = -1` emission.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct Tuple {
        pub boundary_id: i32,
        pub channel: i32,
        pub ec_tell: i32,
        pub rng: u32,
        pub target_rate_bps: i32,
        pub n_bits_exceeded: i32,
        pub curr_n_bits_used_lbrr: i32,
        pub n_bits_used_lbrr: i32,
        pub mid_only_flag: i32,
        pub prev_decode_only_middle: i32,
        /// Rate-control loop iteration index for V2 per-iter boundaries
        /// (106/107/108/109). `-1` is the "not in rate-control loop"
        /// sentinel, used by V1 boundaries and V2 setup boundaries
        /// (100..=105) which fire once per encode. Mirrors the C side's
        /// `DBG_TRACE_F( … , -1, … )` emission in
        /// `harness/silk_encode_frame_FIX_traced.c`.
        pub iter: i32,
        /// State-vector payload for V2 boundaries. Empty for V1. The
        /// boundary_id determines the layout (see HLD V2 §4.1).
        pub payload: Vec<i32>,
    }

    impl Default for Tuple {
        fn default() -> Self {
            Self {
                boundary_id: 0,
                channel: 0,
                ec_tell: 0,
                rng: 0,
                target_rate_bps: 0,
                n_bits_exceeded: 0,
                curr_n_bits_used_lbrr: 0,
                n_bits_used_lbrr: 0,
                mid_only_flag: 0,
                prev_decode_only_middle: 0,
                // -1 is the "not in rate-control loop" sentinel; the C
                // side emits -1 for V1 boundaries (1..=7) and V2 setup
                // boundaries (100..=105). V2 per-iter boundaries
                // (106..=109) overwrite this with the actual iter index.
                iter: -1,
                payload: Vec::new(),
            }
        }
    }

    static BUF: Mutex<Vec<Tuple>> = Mutex::new(Vec::new());

    pub fn clear() {
        if let Ok(mut g) = BUF.lock() {
            g.clear();
        }
    }

    pub fn push(t: Tuple) {
        if let Ok(mut g) = BUF.lock() {
            g.push(t);
        }
    }

    pub fn snapshot() -> Vec<Tuple> {
        BUF.lock().map(|g| g.clone()).unwrap_or_default()
    }
}

mod api;
pub use api::{
    Application, Bandwidth, Bitrate, Channels, DecodeError, DecodeMode, Decoder, DecoderInitError,
    EncodeError, Encoder, EncoderBuildError, EncoderBuilder, ForceChannels, FrameDuration,
    InbandFec, Signal,
};

// Low-level libopus types and integer constants. These mirror the C API and are
// re-exported for advanced users who need finer control than the typed facade
// (`Encoder` / `Decoder` / `Application` / `Bitrate` / `Bandwidth` / `Signal`)
// at the crate root provides; for most callers, prefer those typed wrappers.
pub use opus::decoder::{
    OPUS_BAD_ARG, OPUS_BANDWIDTH_FULLBAND, OPUS_BANDWIDTH_MEDIUMBAND, OPUS_BANDWIDTH_NARROWBAND,
    OPUS_BANDWIDTH_SUPERWIDEBAND, OPUS_BANDWIDTH_WIDEBAND, OPUS_BUFFER_TOO_SMALL,
    OPUS_INTERNAL_ERROR, OPUS_INVALID_PACKET, OPUS_OK, OPUS_UNIMPLEMENTED, OpusDecoder,
};
pub use opus::encoder::{
    OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_APPLICATION_VOIP, OPUS_AUTO,
    OPUS_BITRATE_MAX, OPUS_SIGNAL_MUSIC, OPUS_SIGNAL_VOICE, OpusEncoder,
};
pub use opus::multistream::{OpusMSDecoder, OpusMSEncoder};
pub use opus::repacketizer::OpusRepacketizer;

// Unchecked-indexing hot-path macros.
//
// Used in provably-safe tight inner loops that the C reference has hand-tuned
// with SSE intrinsics:
//   - celt/fft.rs, celt/mdct.rs — FFT butterflies + MDCT rotations
//   - silk/encoder.rs, silk/common.rs — SILK inner loops (NSQ LPC, Burg,
//     autocorrelation, LPC analysis filter) for encoder performance
// Every call site operates under statically-known iteration bounds against
// a slice whose length is at least that bound, and has been fuzzed +
// differential-tested against the C reference with no OOB findings.
// See "2026.04.18 - JRN - restore-unchecked-indexing.md" and the
// encoder-perf follow-up journal.
macro_rules! uc {
    ($slice:expr, $idx:expr) => {
        unsafe { *$slice.get_unchecked($idx) }
    };
}

macro_rules! uc_set {
    ($slice:expr, $idx:expr, $val:expr) => {{
        let uc_idx = $idx;
        let uc_val = $val;
        unsafe { *$slice.get_unchecked_mut(uc_idx) = uc_val }
    }};
}

macro_rules! uc_mut {
    ($slice:expr, $idx:expr) => {
        unsafe { $slice.get_unchecked_mut($idx) }
    };
}

pub(crate) use uc;
pub(crate) use uc_mut;
pub(crate) use uc_set;

#[cfg(test)]
mod coverage_tests;
#[cfg(test)]
mod property_tests_codec;
#[cfg(test)]
mod property_tests_packet;
