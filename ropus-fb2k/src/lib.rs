//! Rust backend for the `foo_input_ropus` foobar2000 component.
//!
//! This crate is the Rust half of the two-DLL design in the HLD: a `cdylib`
//! exposing a small, stable C ABI (see `include/ropus_fb2k.h`). An `rlib`
//! target is kept so `cargo test` can link integration tests against the
//! same crate that ships as the cdylib; integration tests drive the C
//! entry points directly (no Rust-only convenience surface).
//!
//! C ABI surface:
//!
//! * `ropus_fb2k_open` — header + tags parse over a caller-supplied IO
//!   callback struct, producing a `RopusFb2kReader` handle. Reverse-scans
//!   the last Ogg page to populate duration and bitrate when the IO is
//!   seekable.
//! * `ropus_fb2k_close` — destroy the handle.
//! * `ropus_fb2k_get_info` — fill `RopusFb2kInfo` with `channels`,
//!   `pre_skip`, `sample_rate` (always 48 kHz), `total_samples`,
//!   `nominal_bitrate`, and the four ReplayGain fields (R128 + legacy
//!   `REPLAYGAIN_*` tags per HLD §5.5; absent/malformed → `NaN`).
//! * `ropus_fb2k_read_tags` — invoke a caller callback for every parsed
//!   vorbis_comment, plus a synthetic `VENDOR` entry so the caller sees the
//!   encoder string for free. `METADATA_BLOCK_PICTURE` is filtered here.
//! * `ropus_fb2k_decode_next` — decode the next Opus packet into the
//!   caller's interleaved float buffer, transparently trimming the leading
//!   `pre_skip` samples on the first call.
//! * `ropus_fb2k_seek` — seek to a per-channel 48 kHz sample position with
//!   80 ms pre-roll per RFC 7845 §4.2 (page index built lazily on first
//!   call). Unseekable / unknown-duration streams accept seek-to-zero only.
//! * `ropus_fb2k_last_error` / `ropus_fb2k_last_error_code` — thread-local
//!   error slots, paired so the C++ shim can branch on the status code
//!   (never on message text).
//!
//! All `#[unsafe(no_mangle)]` extern "C" entry points wrap their bodies in
//! the local `ffi_guard!` macro (mirrors `capi::ffi_guard!` but with the
//! extra hook of writing a panic notice into the last-error slot), so a
//! panic deep in the parser surfaces as a typed negative status +
//! human-readable last-error string rather than unwinding across the C
//! boundary (UB on stable Rust).

#![allow(clippy::missing_safety_doc)]

// Safety guard: the `test-panic` feature exposes a stable-named
// `extern "C"` hook (`ropus_fb2k_test_set_panic_flag`) that, once armed,
// makes the next `decode_next` call panic deep in the stack. That's
// deliberate for the `ffi_guard!` integration test in `tests/roundtrip.rs`,
// but shipping a release cdylib with this feature on would hand a trivial
// DoS primitive to any caller that can reach the symbol. Fail the build
// if anyone tries to combine the two.
#[cfg(all(feature = "test-panic", not(debug_assertions)))]
compile_error!(
    "the `test-panic` feature must not be enabled in release builds — \
     it exposes a stable extern \"C\" hook (`ropus_fb2k_test_set_panic_flag`) \
     that triggers a panic inside `decode_next`, which a caller could \
     weaponise. enable only with default cargo profiles (dev/test)."
);

use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

mod error;
mod io;
mod reader;
mod tags;

use crate::error::{clear_last_error, last_error_ptr, set_last_error_with_code};
use crate::io::CallbackReader;
use crate::reader::{OggOpusReader, ReaderError};

// ---------------------------------------------------------------------------
// Error codes (kept in sync with `include/ropus_fb2k.h`)
// ---------------------------------------------------------------------------

pub const ROPUS_FB2K_OK: c_int = 0;
pub const ROPUS_FB2K_BAD_ARG: c_int = -1;
pub const ROPUS_FB2K_IO: c_int = -2;
pub const ROPUS_FB2K_ABORTED: c_int = -3;
pub const ROPUS_FB2K_INVALID_STREAM: c_int = -4;
pub const ROPUS_FB2K_UNSUPPORTED: c_int = -5;
/// Internal error: panic or otherwise-impossible state reached inside the
/// Rust backend. The associated FFI call's return value may still be the
/// per-entry-point sentinel (e.g. `BAD_ARG` for `get_info`, `ABORTED` for
/// `decode_next`) because each entry has its own failure shape — callers
/// that want the unified class should read `ropus_fb2k_last_error_code()`
/// and prefer this code over the return value when they disagree.
pub const ROPUS_FB2K_INTERNAL: c_int = -6;

/// Info-only open hint — read the header + tags + last-granule reverse-scan
/// but skip full page-walk. Today both paths reverse-scan identically, so
/// the flag is accepted but not yet differentiated; a future release will
/// replace the full path with a streaming page-walk that also populates
/// the seek index, and the info-only path will keep the reverse-scan.
pub const ROPUS_FB2K_OPEN_INFO_ONLY: u32 = 1 << 0;

/// Tag keys that `ropus_fb2k_read_tags` silently drops before calling the
/// caller's tag callback. The parser still surfaces them on the raw
/// `ParsedTags::iter()` path — filtering happens at the reporting boundary
/// (HLD sec. 2 non-goals: fb2k has its own cover-art pipeline).
///
/// Matched case-insensitively.
const FILTERED_TAG_KEYS: &[&str] = &["METADATA_BLOCK_PICTURE"];

// ---------------------------------------------------------------------------
// C-visible structs
// ---------------------------------------------------------------------------

/// Function-pointer type aliases mirroring `ropus_fb2k.h`. Kept as type
/// aliases (not wrapper structs) so the `#[repr(C)]` layout of
/// `RopusFb2kIo` matches the header byte-for-byte.
pub type RopusFb2kReadFn = extern "C" fn(ctx: *mut c_void, buf: *mut u8, n: usize) -> i64;
pub type RopusFb2kSeekFn = extern "C" fn(ctx: *mut c_void, abs_offset: u64) -> c_int;
pub type RopusFb2kSizeFn = extern "C" fn(ctx: *mut c_void, out_size: *mut u64) -> c_int;
pub type RopusFb2kAbortFn = extern "C" fn(ctx: *mut c_void) -> c_int;

/// IO callback struct — the only thing that crosses the C boundary on
/// open. Copied by value into the reader on open, so the caller may drop
/// this struct as soon as `ropus_fb2k_open` returns.
///
/// `read` is declared as `Option<RopusFb2kReadFn>` so a zero-initialised
/// C struct (fn-pointer = NULL) decodes as `None` rather than triggering
/// UB on dereference. `Option<extern "C" fn>` has the same layout as a
/// bare function pointer thanks to the null-pointer optimization
/// (documented in the Rustonomicon), so the `#[repr(C)]` wire format is
/// unchanged — the header still declares the field as a bare
/// `RopusFb2kReadFn`. A null `read` is rejected with `BAD_ARG` in
/// `ropus_fb2k_open`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RopusFb2kIo {
    pub ctx: *mut c_void,
    pub read: Option<RopusFb2kReadFn>,
    pub seek: Option<RopusFb2kSeekFn>,
    pub size: Option<RopusFb2kSizeFn>,
    pub abort: Option<RopusFb2kAbortFn>,
}

/// Output struct populated by `ropus_fb2k_get_info`. Layout matches the
/// header; absent / malformed ReplayGain tags surface as `NaN`.
#[repr(C)]
pub struct RopusFb2kInfo {
    pub sample_rate: u32,
    pub channels: u8,
    pub pre_skip: u16,
    pub total_samples: u64,
    pub nominal_bitrate: i32,
    pub rg_track_gain: f32,
    pub rg_album_gain: f32,
    pub rg_track_peak: f32,
    pub rg_album_peak: f32,
}

/// Tag-iteration callback type.
pub type TagCb = extern "C" fn(ctx: *mut c_void, key: *const c_char, value: *const c_char);

// ---------------------------------------------------------------------------
// Handle
// ---------------------------------------------------------------------------

/// Opaque handle returned by `ropus_fb2k_open`. Wraps the inner reader.
/// The inner type is parametrised on `CallbackReader` because the C-facing
/// API only ever sees a `RopusFb2kIo`; tests drive the same path by
/// constructing an `RopusFb2kIo` over an in-memory buffer.
pub struct RopusFb2kReader {
    inner: OggOpusReader<CallbackReader>,
}

// ---------------------------------------------------------------------------
// ffi_guard — mirrors capi/src/lib.rs
// ---------------------------------------------------------------------------

/// Wrap an FFI body so a panic becomes a typed status + a stored last-error
/// message rather than UB.
///
/// Divergence from `capi/src/lib.rs::ffi_guard!`: the capi version only
/// catches-and-returns the sentinel; it does not write the last-error slot
/// (capi has no such slot). We deliberately *do* set both the message slot
/// AND the error-code slot on panic so a C caller who sees an unexpected
/// negative status can surface a message instead of a bare error code and
/// can branch on the unified `ROPUS_FB2K_INTERNAL` class rather than the
/// per-entry sentinel (which varies: `BAD_ARG` for `get_info`, `ABORTED`
/// for `decode_next`, NULL pointer for `open`, etc.). The duplication is
/// intentional: making `ropus-fb2k` depend on `capi` would couple two
/// otherwise-independent adapters for the sake of ten lines of macro.
macro_rules! ffi_guard {
    ($on_panic:expr, $body:block) => {{
        match ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| $body)) {
            Ok(v) => v,
            Err(_) => {
                $crate::error::set_last_error_with_code(
                    "internal panic inside ropus-fb2k",
                    $crate::ROPUS_FB2K_INTERNAL,
                );
                $on_panic
            }
        }
    }};
}

// ---------------------------------------------------------------------------
// C entry points
// ---------------------------------------------------------------------------

/// Open an Ogg Opus stream via caller-supplied IO callbacks.
///
/// Returns a non-null handle on success. On failure returns NULL and stores
/// a human-readable message in the thread-local slot retrievable via
/// `ropus_fb2k_last_error`. The negative status is not returned through
/// this entry point (since the return type is a pointer) — callers that
/// want the code should wrap the failure case or call `get_info` on a
/// handle that is guaranteed non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ropus_fb2k_open(
    io: *const RopusFb2kIo,
    flags: u32,
) -> *mut RopusFb2kReader {
    ffi_guard!(ptr::null_mut(), {
        if io.is_null() {
            set_last_error_with_code("io pointer is null", ROPUS_FB2K_BAD_ARG);
            return ptr::null_mut();
        }
        // SAFETY: caller asserts non-null `io` points to a fully-initialised
        // `RopusFb2kIo`. We copy it by value so subsequent mutations to the
        // caller's struct don't affect the reader.
        let io_copy = unsafe { *io };

        let Some(reader) = CallbackReader::new(io_copy) else {
            // `read` is the only non-optional callback; null here is a
            // programmer error on the C side (the header says so).
            set_last_error_with_code("io.read callback is null", ROPUS_FB2K_BAD_ARG);
            return ptr::null_mut();
        };
        // Capture the reverse-scan prerequisite once up front. An unseekable
        // stream (no `seek` callback) or unknown size yields `None`, which
        // tells the reader to skip the scan and publish zeroed duration.
        let size_hint = reader.can_seek().then(|| reader.size()).flatten();
        match OggOpusReader::open(reader, flags, size_hint) {
            Ok(inner) => {
                clear_last_error();
                Box::into_raw(Box::new(RopusFb2kReader { inner }))
            }
            Err(e) => {
                let code = reader_error_code(&e);
                set_last_error_with_code(e.to_string(), code);
                ptr::null_mut()
            }
        }
    })
}

/// Destroy a reader handle. Safe to call on NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ropus_fb2k_close(ptr: *mut RopusFb2kReader) {
    let _: () = ffi_guard!((), {
        if !ptr.is_null() {
            // SAFETY: `ptr` was returned by `ropus_fb2k_open`, which boxes
            // `RopusFb2kReader` and yields the raw pointer via
            // `Box::into_raw`. Callers promise to pass it here exactly once.
            let _ = unsafe { Box::from_raw(ptr) };
        }
    });
}

/// Retrieve the thread-local last-error string. Never NULL (always points
/// to at least an empty C string).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ropus_fb2k_last_error() -> *const c_char {
    ffi_guard!(ptr::null(), { last_error_ptr() })
}

/// Retrieve the negative status code associated with the most recent failed
/// call on this thread. Returns 0 if the last relevant call succeeded (or
/// had no code). Paired with `ropus_fb2k_last_error()` for message text —
/// the C++ shim branches on the code to pick the correct fb2k exception.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ropus_fb2k_last_error_code() -> c_int {
    ffi_guard!(0, { error::last_error_code() })
}

/// Populate `RopusFb2kInfo` from a successfully-opened reader.
///
/// Populates every field per the C header contract: `channels`, `pre_skip`,
/// `sample_rate` (always 48 000 per HLD §2), `total_samples`,
/// `nominal_bitrate`, and the four ReplayGain fields (extracted from R128
/// / legacy `REPLAYGAIN_*` tags at open time per HLD §5.5; absent or
/// malformed tags surface as `NaN`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ropus_fb2k_get_info(
    r: *mut RopusFb2kReader,
    out: *mut RopusFb2kInfo,
) -> c_int {
    ffi_guard!(ROPUS_FB2K_BAD_ARG, {
        if r.is_null() || out.is_null() {
            set_last_error_with_code("get_info: null pointer", ROPUS_FB2K_BAD_ARG);
            return ROPUS_FB2K_BAD_ARG;
        }
        // SAFETY: caller asserts `r` came from `ropus_fb2k_open` (non-null
        // handles only) and `out` points to a writable `RopusFb2kInfo`.
        let reader = unsafe { &*r };
        let rg = reader.inner.replaygain();
        let info = RopusFb2kInfo {
            // We always decode at 48 kHz regardless of `input_sample_rate` —
            // Opus runs internally at 48 kHz for family 0. The header's
            // `input_sample_rate` field is informational only (RFC 7845).
            sample_rate: 48_000,
            channels: reader.inner.channels(),
            pre_skip: reader.inner.pre_skip(),
            total_samples: reader.inner.total_samples(),
            nominal_bitrate: reader.inner.nominal_bitrate(),
            rg_track_gain: rg.track_gain,
            rg_album_gain: rg.album_gain,
            rg_track_peak: rg.track_peak,
            rg_album_peak: rg.album_peak,
        };
        unsafe { ptr::write(out, info) };
        clear_last_error();
        ROPUS_FB2K_OK
    })
}

/// Invoke `cb` for every parsed vorbis_comment. Each `(key, value)` pair is
/// passed as null-terminated UTF-8 strings whose lifetime ends when the
/// callback returns — the callee must copy if it wants to keep them.
///
/// Also emits a synthetic `("VENDOR", vendor_string)` entry first so the
/// caller doesn't need two separate APIs to learn who encoded the file.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ropus_fb2k_read_tags(
    r: *mut RopusFb2kReader,
    cb: Option<TagCb>,
    ctx: *mut c_void,
) -> c_int {
    ffi_guard!(ROPUS_FB2K_BAD_ARG, {
        if r.is_null() || cb.is_none() {
            set_last_error_with_code("read_tags: null pointer", ROPUS_FB2K_BAD_ARG);
            return ROPUS_FB2K_BAD_ARG;
        }
        let cb = cb.unwrap();
        // SAFETY: `r` came from `ropus_fb2k_open`, still valid per caller.
        let reader = unsafe { &*r };

        // Emit the vendor string as a synthetic tag first.
        if !emit_tag(cb, ctx, "VENDOR", reader.inner.vendor()) {
            set_last_error_with_code(
                "read_tags: tag contained interior NUL",
                ROPUS_FB2K_INVALID_STREAM,
            );
            return ROPUS_FB2K_INVALID_STREAM;
        }

        for (k, v) in reader.inner.tags().iter() {
            // Filter cover-art blobs at the reporting boundary (not in the
            // parser), so the raw comment is still available to anyone
            // pulling `ParsedTags::iter()` directly.
            if FILTERED_TAG_KEYS
                .iter()
                .any(|f| k.eq_ignore_ascii_case(f))
            {
                continue;
            }
            if !emit_tag(cb, ctx, k, v) {
                set_last_error_with_code(
                    "read_tags: tag contained interior NUL",
                    ROPUS_FB2K_INVALID_STREAM,
                );
                return ROPUS_FB2K_INVALID_STREAM;
            }
        }
        clear_last_error();
        ROPUS_FB2K_OK
    })
}

/// Invoke a tag callback with key/value, converting to C strings. Returns
/// false if either key or value contains an interior NUL (in which case the
/// caller bails out with INVALID_STREAM rather than silently mangling).
fn emit_tag(cb: TagCb, ctx: *mut c_void, key: &str, value: &str) -> bool {
    let Ok(key_c) = std::ffi::CString::new(key) else {
        return false;
    };
    let Ok(val_c) = std::ffi::CString::new(value) else {
        return false;
    };
    cb(ctx, key_c.as_ptr(), val_c.as_ptr());
    true
}

/// Minimum per-channel sample count the caller's buffer must accept, matching
/// the 120 ms longest Opus frame at 48 kHz. Advertised in `ropus_fb2k.h`; a
/// smaller buffer is rejected with `BAD_ARG` rather than silently truncating.
/// Re-exports the single source-of-truth in `reader.rs` so the two values
/// can never drift.
use crate::reader::MAX_FRAME_SAMPLES_PER_CH as MIN_OUT_SAMPLES_PER_CH;

/// Decode the next Opus packet into the caller's interleaved float buffer.
///
/// Returns samples-per-channel written (`> 0`), `0` at clean end-of-stream,
/// or a negative status. On the very first successful call after open, the
/// leading `pre_skip` samples (typically 312) are trimmed transparently —
/// callers never see the encoder warm-up.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ropus_fb2k_decode_next(
    r: *mut RopusFb2kReader,
    out_interleaved: *mut f32,
    max_samples_per_ch: usize,
) -> c_int {
    ffi_guard!(ROPUS_FB2K_BAD_ARG, {
        if r.is_null() || out_interleaved.is_null() {
            set_last_error_with_code("decode_next: null pointer", ROPUS_FB2K_BAD_ARG);
            return ROPUS_FB2K_BAD_ARG;
        }
        if max_samples_per_ch < MIN_OUT_SAMPLES_PER_CH {
            set_last_error_with_code(
                "out buffer must fit 120 ms (5760 samples/ch)",
                ROPUS_FB2K_BAD_ARG,
            );
            return ROPUS_FB2K_BAD_ARG;
        }

        // SAFETY: `r` came from `ropus_fb2k_open`, still valid per caller.
        let reader = unsafe { &mut *r };
        let channels = reader.inner.channels() as usize;
        // SAFETY: header contract says `out_interleaved` has room for
        // `max_samples_per_ch * channels` f32 values.
        let out_slice = unsafe {
            std::slice::from_raw_parts_mut(out_interleaved, max_samples_per_ch * channels)
        };

        match reader.inner.decode_next(out_slice, max_samples_per_ch) {
            Ok(n) => {
                clear_last_error();
                n
            }
            Err(e) => {
                let code = reader_error_code(&e);
                set_last_error_with_code(e.to_string(), code);
                code
            }
        }
    })
}

/// Seek to a per-channel sample position (48 kHz, post-pre-skip).
///
/// Clamps `sample_pos` to `[0, total_samples]`. Rewinds an extra 80 ms for
/// decoder convergence per RFC 7845 §4.2; the pre-roll is silently discarded
/// on subsequent `decode_next` calls so the caller never sees it.
///
/// For unseekable streams (no `seek` IO callback) or streams of unknown
/// duration, only `sample_pos == 0` is accepted; anything else returns
/// `ROPUS_FB2K_INVALID_STREAM` (→ `exception_io_data` at the C++ shim).
/// We don't use `UNSUPPORTED` here because fb2k's
/// `exception_io_unsupported_format` contract means "fall through to next
/// input decoder", which is the wrong signal once we already own the
/// stream. Aborts during the first-time index build surface as
/// `ROPUS_FB2K_ABORTED`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ropus_fb2k_seek(r: *mut RopusFb2kReader, sample_pos: u64) -> c_int {
    ffi_guard!(ROPUS_FB2K_INTERNAL, {
        if r.is_null() {
            set_last_error_with_code("seek: null pointer", ROPUS_FB2K_BAD_ARG);
            return ROPUS_FB2K_BAD_ARG;
        }
        // SAFETY: `r` came from `ropus_fb2k_open`, still valid per caller.
        let reader = unsafe { &mut *r };
        match reader.inner.seek(sample_pos) {
            Ok(()) => {
                clear_last_error();
                ROPUS_FB2K_OK
            }
            Err(e) => {
                let code = reader_error_code(&e);
                set_last_error_with_code(e.to_string(), code);
                code
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Crate-private helpers
// ---------------------------------------------------------------------------

/// Map a `ReaderError` variant to its C status code. Used by the FFI entry
/// points to populate `last_error_code` consistently with the variant.
fn reader_error_code(e: &ReaderError) -> c_int {
    match e {
        ReaderError::Io(_) => ROPUS_FB2K_IO,
        ReaderError::InvalidStream(_) => ROPUS_FB2K_INVALID_STREAM,
        ReaderError::Unsupported(_) => ROPUS_FB2K_UNSUPPORTED,
        ReaderError::Aborted => ROPUS_FB2K_ABORTED,
    }
}

// ---------------------------------------------------------------------------
// Test-only panic injection (HLD §7 "Panic safety" tier)
// ---------------------------------------------------------------------------
//
// Behind the `test-panic` feature flag we expose a hidden FFI hook
// (`ropus_fb2k_test_set_panic_flag`) and a thread-local flag that the
// `decode_next` path consults at entry. With the flag set, `decode_next`
// `panic!`s deep in the stack so the integration test in
// `tests/roundtrip.rs` can prove `ffi_guard!` catches the unwind, sets
// `LAST_ERROR_CODE = -6 INTERNAL`, populates a human-readable message, and
// returns the per-entry sentinel (`-1 BAD_ARG` for `decode_next`).
//
// Gated entirely behind `cfg(feature = "test-panic")` so the symbol is not
// part of the released cdylib's exported surface and the runtime cost in
// production is exactly zero (no branch, no thread-local, no extra code).

#[cfg(feature = "test-panic")]
thread_local! {
    static PANIC_FLAG: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Returns `true` iff the test harness has armed the panic flag for this
/// thread. Called from `reader.rs::decode_next` (only when the feature
/// is enabled).
#[cfg(feature = "test-panic")]
pub(crate) fn test_panic_should_fire() -> bool {
    PANIC_FLAG.with(|c| c.get())
}

/// Test-only FFI hook to arm the panic-injection flag for the current
/// thread. Set to `true` to make the next `decode_next` call panic; set
/// back to `false` to disarm. The flag is per-thread so parallel test runs
/// can't trip each other.
///
/// **Not** part of the public ABI — the symbol is only exported when the
/// `test-panic` feature is on. Don't ship a `cdylib` built with this flag.
#[cfg(feature = "test-panic")]
#[unsafe(no_mangle)]
pub extern "C" fn ropus_fb2k_test_set_panic_flag(on: bool) {
    PANIC_FLAG.with(|c| c.set(on));
}
