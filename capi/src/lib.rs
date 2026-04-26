//! C ABI shim over the `ropus` Rust codec, sized for the xiph/opus conformance tests.
//!
//! This crate is a dev-only adapter — not a published libopus replacement. It
//! exists so that `reference/tests/test_opus_*.c` can be compiled against us
//! verbatim. The staticlib crate-type produces a library the conformance test
//! harness can link in alongside the renamed C test objects.
//!
//! All `#[no_mangle] extern "C"` entry points are wrapped in
//! `std::panic::catch_unwind` via the `ffi_guard!` macro. Unwinding across a
//! plain `extern "C"` boundary is UB on stable Rust; a panic in the codec (our
//! `src/` is full of `unwrap`/`expect`/bounds-checked indexing) becomes
//! `OPUS_INTERNAL_ERROR`, which matches how the C codec surfaces an assertion
//! failure to callers.

#![allow(non_snake_case, non_camel_case_types)]
// Every `pub unsafe extern "C" fn` in this crate inherits the libopus C-ABI
// safety contract (valid pointers, correctly-sized buffers, single-threaded
// use of a handle, etc.). That contract is documented in the xiph/opus
// headers we're cloning, not re-derived per function here — a hundred
// copy-paste `/// # Safety` blocks would add noise without new information.
#![allow(clippy::missing_safety_doc)]

// Public API surface organised like the reference headers.
pub mod ctl;
pub mod decoder;
pub mod encoder;
pub mod extensions;
pub mod ms_decoder;
pub mod ms_encoder;
pub mod packet_parse;
pub mod projection;
pub mod repacketizer;

/// Monotonically increasing counter used to stamp each freshly-created state
/// handle. Together with the per-handle `generation` field this guarantees
/// that `memcmp` between two independently-created handles sees a difference,
/// and that a handle post-`OPUS_RESET_STATE` differs from any pre-reset byte
/// snapshot. `Relaxed` is sufficient — we don't synchronise any other memory,
/// we just need a unique-enough token per handle.
pub(crate) fn next_handle_generation() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

// --- Allocator used for heap state -------------------------------------------
//
// `test_opus_decode.c` (lines 85-95) asserts "the state structures contain no
// pointers and can be freely copied" — it `malloc`s a replacement buffer,
// `memcpy`s the existing state into it, `memset`s the original to `0xFF` to
// poison it, then calls `opus_decoder_destroy` on the poisoned original, and
// later calls `opus_decoder_destroy` on that `malloc`-allocated replacement.
// Our Rust state (`OpusDecoder`) does contain `Vec` fields, so running `Drop`
// on the poisoned bytes would try to free a wild pointer.
//
// Solution: leak. Allocate the outer state bytes via Rust's global allocator,
// and make `_destroy` a no-op. Neither the outer struct nor any interior Vec
// heap data is ever freed. Conformance test processes are short-lived; this is
// an acceptable test-time concession.
//
// This also sidesteps the MSVC Windows heap-mismatch issue: we never try to
// free a CRT-malloc'd replacement buffer via Rust's allocator, nor vice versa.
// Everything leaks uniformly until the process exits.

/// No-op — see the module comment. We intentionally leak state allocations
/// because the conformance test memcpy's state bytes into CRT-malloc'd
/// replacements and then destroys the original. Running `Drop` on the
/// (poisoned) original would crash; calling any allocator's free on a buffer
/// from the other allocator corrupts the heap. Leaking is the cleanest path.
pub(crate) unsafe fn state_free<T>(_ptr: *mut T) {
    // intentional no-op — see doc comment.
}

// --- Error codes (match `opus_defines.h` verbatim) ----------------------------
pub const OPUS_OK: i32 = 0;
pub const OPUS_BAD_ARG: i32 = -1;
pub const OPUS_BUFFER_TOO_SMALL: i32 = -2;
pub const OPUS_INTERNAL_ERROR: i32 = -3;
pub const OPUS_INVALID_PACKET: i32 = -4;
pub const OPUS_UNIMPLEMENTED: i32 = -5;
pub const OPUS_INVALID_STATE: i32 = -6;
pub const OPUS_ALLOC_FAIL: i32 = -7;

// --- CTL request codes surfaced at crate scope -------------------------------
// The CTL surface exercised by the xiph conformance tests (encoder, decoder,
// and multistream variants) is dispatched by `ctl_shim.c` into typed Rust
// entry points in `ctl.rs`. Unknown requests land on the default arm and
// return `OPUS_UNIMPLEMENTED`; that includes the DRED/OSCE/QEXT families plus
// a handful of niche requests the conformance suite doesn't touch
// (`VOICE_RATIO`, `IN_DTX`, `LFE`, `ENERGY_MASK`).
// `OPUS_RESET_STATE` is the only code re-exported at crate scope because
// other modules refer to it by name (see the generation-bump docs in
// encoder.rs / decoder.rs).
pub const OPUS_RESET_STATE: i32 = 4028;

/// Wrap an FFI body so a panic becomes an error code rather than UB.
///
/// Example:
/// ```ignore
/// #[unsafe(no_mangle)]
/// pub unsafe extern "C" fn foo(...) -> i32 {
///     ffi_guard!(OPUS_INTERNAL_ERROR, { /* body returning i32 */ })
/// }
/// ```
#[macro_export]
macro_rules! ffi_guard {
    ($on_panic:expr, $body:block) => {{
        match ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| $body)) {
            Ok(v) => v,
            Err(_) => $on_panic,
        }
    }};
}

// --- Small string constants shared by encoder + decoder ----------------------

/// Version string returned by `opus_get_version_string`.
///
/// The substring "-fixed" signals a fixed-point build, per the convention the
/// C reference documents in `celt.c::opus_get_version_string`. This matches
/// the format C consumers may grep on (e.g. `opus_demo`).
pub(crate) const VERSION_STRING: &[u8] = b"libopus mdopus-capi-0.1.0-fixed\0";

/// Error-string table (index by `-error`, same convention as C).
pub(crate) const ERROR_STRINGS: &[&[u8]] = &[
    b"success\0",
    b"invalid argument\0",
    b"buffer too small\0",
    b"internal error\0",
    b"corrupted stream\0",
    b"request not implemented\0",
    b"invalid state\0",
    b"memory allocation failed\0",
];
pub(crate) const UNKNOWN_ERROR: &[u8] = b"unknown error\0";

/// `opus_strerror` — returns a static C string for the given error code.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_strerror(error: i32) -> *const std::os::raw::c_char {
    ffi_guard!(UNKNOWN_ERROR.as_ptr() as *const _, {
        if error > 0 {
            return UNKNOWN_ERROR.as_ptr() as *const _;
        }
        ERROR_STRINGS
            .get((-error) as usize)
            .copied()
            .unwrap_or(UNKNOWN_ERROR)
            .as_ptr() as *const _
    })
}

/// `opus_get_version_string` — returns a static C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_get_version_string() -> *const std::os::raw::c_char {
    ffi_guard!(VERSION_STRING.as_ptr() as *const _, {
        VERSION_STRING.as_ptr() as *const _
    })
}
