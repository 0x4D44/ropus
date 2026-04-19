//! Thread-local last-error slot for the C ABI.
//!
//! Each calling thread gets its own buffer. `set_last_error` stores a
//! freshly-allocated `CString`; `last_error_ptr` returns a pointer that
//! stays valid until the next `set_last_error` call on the same thread.
//!
//! The empty-string default means `ropus_fb2k_last_error()` never returns a
//! null pointer, per the header contract.
//!
//! Matches the spirit of `capi/src/lib.rs` in keeping the FFI surface panic-
//! and null-proof while leaving heap ownership inside the Rust side.

use std::cell::{Cell, RefCell};
use std::ffi::CString;
use std::os::raw::{c_char, c_int};

thread_local! {
    /// Per-thread last-error slot. The `CString` owns the bytes the returned
    /// `*const c_char` points into; replacing it invalidates any prior pointer,
    /// which is the same rule as C `errno` / `strerror` per thread.
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").expect("empty CString"));

    /// Per-thread last-error status code. Paired with `LAST_ERROR`: each
    /// `set_last_error_with_code` writes both; `clear_last_error` resets to 0
    /// (for "no error"). C callers read it via `ropus_fb2k_last_error_code()`
    /// so the fb2k shim can branch on the error class without string-matching
    /// the message.
    static LAST_ERROR_CODE: Cell<c_int> = const { Cell::new(0) };
}

/// Store a last-error message for the current thread without a status code.
/// Prefer `set_last_error_with_code` when a negative code is available — this
/// wrapper exists for code paths where the code would otherwise be
/// duplicated at every call site (e.g. inside error formatting helpers).
///
/// Currently only used by this module's unit tests, which verify that the
/// code-less path stores 0. If a future caller needs the no-code shape, drop
/// the `#[cfg(test)]` gate.
#[cfg(test)]
pub(crate) fn set_last_error(msg: impl Into<String>) {
    set_last_error_with_code(msg, 0);
}

/// Store a last-error message and its associated negative status code. Interior
/// NULs are replaced with `?` so the raw `String` is always convertible to a
/// `CString` without allocating a second time on error.
pub(crate) fn set_last_error_with_code(msg: impl Into<String>, code: c_int) {
    let mut s = msg.into();
    if s.as_bytes().contains(&0) {
        s = s.replace('\0', "?");
    }
    // Safe to unwrap: we stripped interior NULs above.
    let c = CString::new(s).expect("interior NULs already stripped");
    LAST_ERROR.with(|slot| *slot.borrow_mut() = c);
    LAST_ERROR_CODE.with(|slot| slot.set(code));
}

/// Clear the last-error slot. Used on successful entry-point returns so a
/// stale message from a previous call isn't surfaced to a user asking "why
/// did this open succeed but `last_error()` still says 'aborted'?".
pub(crate) fn clear_last_error() {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = CString::new("").expect("empty CString");
    });
    LAST_ERROR_CODE.with(|slot| slot.set(0));
}

/// Return a pointer to the current thread's last-error C string. The pointer
/// is valid until the next `set_last_error` on this thread. Never NULL.
pub(crate) fn last_error_ptr() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}

/// Return the last-error code stored on this thread. 0 when the last
/// relevant call succeeded (or no code was associated with the message).
pub(crate) fn last_error_code() -> c_int {
    LAST_ERROR_CODE.with(|slot| slot.get())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn default_is_empty_not_null() {
        // Use a fresh thread so we don't depend on what a prior test wrote.
        std::thread::spawn(|| {
            let p = last_error_ptr();
            assert!(!p.is_null());
            let s = unsafe { CStr::from_ptr(p) };
            assert_eq!(s.to_bytes(), b"");
        })
        .join()
        .unwrap();
    }

    #[test]
    fn set_then_read_round_trip() {
        set_last_error("aborted");
        let p = last_error_ptr();
        let s = unsafe { CStr::from_ptr(p) };
        assert_eq!(s.to_str().unwrap(), "aborted");
    }

    #[test]
    fn interior_nuls_are_scrubbed() {
        set_last_error("foo\0bar");
        let p = last_error_ptr();
        let s = unsafe { CStr::from_ptr(p) };
        assert_eq!(s.to_str().unwrap(), "foo?bar");
    }

    #[test]
    fn clear_resets_to_empty() {
        set_last_error("boom");
        clear_last_error();
        let p = last_error_ptr();
        let s = unsafe { CStr::from_ptr(p) };
        assert_eq!(s.to_bytes(), b"");
    }

    #[test]
    fn set_with_code_round_trips() {
        // Fresh thread so other tests on this runner can't race the slot.
        std::thread::spawn(|| {
            set_last_error_with_code("bad thing", -5);
            assert_eq!(last_error_code(), -5);
            let s = unsafe { CStr::from_ptr(last_error_ptr()) };
            assert_eq!(s.to_str().unwrap(), "bad thing");
        })
        .join()
        .unwrap();
    }

    #[test]
    fn clear_resets_code_to_zero() {
        std::thread::spawn(|| {
            set_last_error_with_code("boom", -4);
            assert_eq!(last_error_code(), -4);
            clear_last_error();
            assert_eq!(last_error_code(), 0);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn set_without_code_stores_zero() {
        std::thread::spawn(|| {
            set_last_error("legacy");
            assert_eq!(last_error_code(), 0);
        })
        .join()
        .unwrap();
    }
}
