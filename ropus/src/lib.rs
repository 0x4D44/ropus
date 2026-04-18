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
//! use ropus::{OpusEncoder, OpusDecoder, OPUS_APPLICATION_VOIP};
//!
//! let mut encoder = OpusEncoder::new(48_000, 1, OPUS_APPLICATION_VOIP).unwrap();
//! let pcm_in = [0i16; 960]; // 20 ms at 48 kHz mono
//! let mut packet = [0u8; 4000];
//! let max_bytes = packet.len() as i32;
//! let len = encoder.encode(&pcm_in, 960, &mut packet, max_bytes).unwrap();
//!
//! let mut decoder = OpusDecoder::new(48_000, 1).unwrap();
//! let mut pcm_out = [0i16; 960];
//! let samples = decoder.decode(Some(&packet[..len as usize]), &mut pcm_out, 960, false).unwrap();
//! assert_eq!(samples, 960);
//! ```

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

mod api;
pub use api::{
    Application, Bandwidth, Bitrate, Channels, DecodeError, Decoder, DecoderInitError,
    EncodeError, Encoder, EncoderBuildError, EncoderBuilder, ForceChannels, FrameDuration, Signal,
};

// Low-level libopus types and integer constants. These mirror the C API and are
// re-exported for advanced users who need finer control than the typed facade
// (`Encoder` / `Decoder` / `Application` / `Bitrate` / `Bandwidth` / `Signal`)
// at the crate root provides; for most callers, prefer those typed wrappers.
pub use opus::encoder::{
    OpusEncoder,
    OPUS_APPLICATION_VOIP, OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY,
    OPUS_SIGNAL_VOICE, OPUS_SIGNAL_MUSIC,
    OPUS_AUTO, OPUS_BITRATE_MAX,
};
pub use opus::decoder::{
    OpusDecoder,
    OPUS_OK, OPUS_BAD_ARG, OPUS_BUFFER_TOO_SMALL, OPUS_INTERNAL_ERROR,
    OPUS_INVALID_PACKET, OPUS_UNIMPLEMENTED,
    OPUS_BANDWIDTH_NARROWBAND, OPUS_BANDWIDTH_MEDIUMBAND, OPUS_BANDWIDTH_WIDEBAND,
    OPUS_BANDWIDTH_SUPERWIDEBAND, OPUS_BANDWIDTH_FULLBAND,
};
pub use opus::multistream::{OpusMSEncoder, OpusMSDecoder};
pub use opus::repacketizer::OpusRepacketizer;

// Unchecked-indexing hot-path macros for CELT FFT / MDCT inner loops.
//
// Every call site lives in `celt/fft.rs` or `celt/mdct.rs` and has been
// fuzzed + differential-tested against the C reference with no OOB
// findings. Bounds-checked indexing cost ~9% on real-content decode
// (see "2026.04.18 - JRN - restore-unchecked-indexing.md").
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
mod property_tests_packet;
#[cfg(test)]
mod property_tests_codec;
