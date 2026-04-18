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

// Public API surface organised like the reference headers.
pub mod ctl;
pub mod decoder;
pub mod encoder;

// --- Error codes (match `opus_defines.h` verbatim) ----------------------------
pub const OPUS_OK: i32 = 0;
pub const OPUS_BAD_ARG: i32 = -1;
pub const OPUS_BUFFER_TOO_SMALL: i32 = -2;
pub const OPUS_INTERNAL_ERROR: i32 = -3;
pub const OPUS_INVALID_PACKET: i32 = -4;
pub const OPUS_UNIMPLEMENTED: i32 = -5;
pub const OPUS_INVALID_STATE: i32 = -6;
pub const OPUS_ALLOC_FAIL: i32 = -7;

// --- CTL request codes we dispatch on from the C shim -------------------------
// Phase 1 only needs OPUS_RESET_STATE; everything else lands on the
// unimplemented default arm in ctl_shim.c.
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
