//! `opus_repacketizer_*` C ABI wrappers.
//!
//! The ropus `OpusRepacketizer<'a>` carries a `'a` lifetime tied to the input
//! slices it borrows. C has no lifetimes, and the repacketizer contract places
//! the onus on the caller to keep input buffers alive between `cat()` and
//! `out()` — exactly what the Rust lifetime tracks.
//!
//! The capi layer erases the lifetime behind a handle: the POD handle points
//! at a `Box<OpusRepacketizer<'static>>` where `'static` is a lie backed by
//! the documented external invariant. All access goes through this crate so
//! the ropus public API (which keeps its honest `'a`) is untouched.
//!
//! # Caller invariant on input buffer lifetime
//!
//! `opus_repacketizer_cat` stores `*const u8` pointers into the caller's
//! packet buffers inside the handle via the erased `'static` lifetime. The
//! caller MUST keep every buffer passed to `_cat` live and unmodified across
//! the entire `opus_repacketizer_cat` → `opus_repacketizer_out` /
//! `opus_repacketizer_out_range` sequence (up to and including the final
//! `_out*` call). Releasing, reusing, or overwriting an input buffer before
//! the terminal `_out*` is UB.
//!
//! This matches the C reference contract. `test_opus_api.c` satisfies it by
//! keeping all packets on the stack within the function that drives `_cat`
//! and `_out*`, so the pointers remain valid for the lifetime of the
//! sequence. Any new caller must preserve this discipline.
//!
//! `test_opus_api.c` does NOT memcpy repacketizer state, so the handle's
//! padded size does not need to match any specific target; we just need
//! `_get_size()` to return a value `>= size_of::<OpusRepacketizerHandle>()`
//! for the test's `malloc(...)` call.

use std::os::raw::{c_int, c_uchar};
use std::ptr;

use ropus::opus::repacketizer::OpusRepacketizer;

use crate::{OPUS_BAD_ARG, OPUS_INTERNAL_ERROR, OPUS_OK, ffi_guard, state_free};

/// C-facing opaque repacketizer type. Named to match `OpusRepacketizer` in the
/// reference public headers; the test stack-allocates it by size from
/// `opus_repacketizer_get_size`.
#[allow(non_camel_case_types)]
pub struct OpusRepacketizerC {
    _private: [u8; 0],
}

const REPACKETIZER_HANDLE_MAGIC: u64 = 0x4D44_4F50_5553_5243; // "MDOPUSRC"
/// Fixed size returned by `opus_repacketizer_get_size`. Comfortably larger
/// than the handle prefix so caller-allocated storage always fits.
const REPACKETIZER_STORAGE_SIZE: usize = 64;

#[repr(C)]
struct OpusRepacketizerHandle {
    magic: u64,
    inner: *mut OpusRepacketizer<'static>,
    generation: u64,
    /// Mirror of `inner.get_nb_frames()`. Kept in sync with `cat` / `init` /
    /// `out*` so `test_opus_extensions.c` can read `rp.nb_frames` directly
    /// off the struct (matches the C reference layout exposed via
    /// `opus_private.h`).
    nb_frames: i32,
    _pad: [u8; REPACKETIZER_STORAGE_SIZE - 28],
}

const _: () = assert!(std::mem::size_of::<OpusRepacketizerHandle>() == REPACKETIZER_STORAGE_SIZE);
/// `nb_frames` must live at byte offset 24 so the C-visible struct layout in
/// `opus_private.h` (`unsigned char _prefix[24]; int nb_frames; ...`) aligns
/// with our handle's mirror. `test_opus_extensions.c` reads `rp.nb_frames`
/// directly; if this offset drifts the test fails.
const _: () = assert!(std::mem::offset_of!(OpusRepacketizerHandle, nb_frames) == 24);

fn alloc_handle_storage() -> *mut OpusRepacketizerHandle {
    let layout = std::alloc::Layout::new::<OpusRepacketizerHandle>();
    // SAFETY: layout is for a non-zero-sized type.
    unsafe {
        let p = std::alloc::alloc_zeroed(layout);
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        p as *mut OpusRepacketizerHandle
    }
}

unsafe fn resolve_handle<'a>(
    st: *mut OpusRepacketizerC,
) -> Option<&'a mut OpusRepacketizer<'static>> {
    if st.is_null() {
        return None;
    }
    let h = st as *const OpusRepacketizerHandle;
    // SAFETY: caller gave us storage >= size_of<OpusRepacketizerHandle>.
    let magic = unsafe { (*h).magic };
    if magic != REPACKETIZER_HANDLE_MAGIC {
        return None;
    }
    let inner = unsafe { (*h).inner };
    if inner.is_null() {
        return None;
    }
    Some(unsafe { &mut *inner })
}

unsafe fn resolve_handle_ref<'a>(
    st: *const OpusRepacketizerC,
) -> Option<&'a OpusRepacketizer<'static>> {
    if st.is_null() {
        return None;
    }
    let h = st as *const OpusRepacketizerHandle;
    let magic = unsafe { (*h).magic };
    if magic != REPACKETIZER_HANDLE_MAGIC {
        return None;
    }
    let inner = unsafe { (*h).inner };
    if inner.is_null() {
        return None;
    }
    Some(unsafe { &*inner })
}

unsafe fn install_handle(dst: *mut OpusRepacketizerHandle, inner: *mut OpusRepacketizer<'static>) {
    // SAFETY: dst points to zeroed storage of size REPACKETIZER_STORAGE_SIZE.
    unsafe {
        (*dst).magic = REPACKETIZER_HANDLE_MAGIC;
        (*dst).inner = inner;
        (*dst).generation = crate::next_handle_generation();
        (*dst).nb_frames = 0;
    }
}

/// Refresh the `nb_frames` mirror from `inner.get_nb_frames()`. Call after any
/// mutation on `inner` so C-side reads of `rp.nb_frames` stay accurate.
unsafe fn sync_nb_frames(st: *mut OpusRepacketizerC) {
    if st.is_null() {
        return;
    }
    let h = st as *mut OpusRepacketizerHandle;
    let inner = unsafe { (*h).inner };
    if inner.is_null() {
        return;
    }
    unsafe {
        (*h).nb_frames = (*inner).get_nb_frames();
    }
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_get_size() -> c_int {
    ffi_guard!(0, { REPACKETIZER_STORAGE_SIZE as c_int })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_init(
    rp: *mut OpusRepacketizerC,
) -> *mut OpusRepacketizerC {
    ffi_guard!(ptr::null_mut(), {
        if rp.is_null() {
            return ptr::null_mut();
        }
        // Unconditionally wipe and install a fresh inner. We cannot peek at
        // `(*h).magic` to branch on "re-init vs fresh" because the caller
        // typically hands us uninitialised malloc'd storage and reading from
        // it would be UB. Any previous inner Box is leaked — the encoder /
        // decoder `_init` entry points use the same leak-on-init policy, and
        // it mirrors `_destroy`'s leak (see `crate::state_free`).
        let inner = Box::into_raw(Box::new(OpusRepacketizer::new()));
        unsafe {
            ptr::write_bytes(rp as *mut u8, 0, REPACKETIZER_STORAGE_SIZE);
            install_handle(rp as *mut OpusRepacketizerHandle, inner);
        }
        rp
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_create() -> *mut OpusRepacketizerC {
    ffi_guard!(ptr::null_mut(), {
        let handle = alloc_handle_storage();
        let inner = Box::into_raw(Box::new(OpusRepacketizer::new()));
        unsafe { install_handle(handle, inner) };
        handle as *mut OpusRepacketizerC
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_destroy(rp: *mut OpusRepacketizerC) {
    let _: () = ffi_guard!((), {
        unsafe { state_free(rp) };
    });
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_cat(
    rp: *mut OpusRepacketizerC,
    data: *const c_uchar,
    len: i32,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if data.is_null() || len < 0 {
            return OPUS_BAD_ARG;
        }
        let Some(inner) = (unsafe { resolve_handle(rp) }) else {
            return OPUS_BAD_ARG;
        };
        let slice = unsafe { std::slice::from_raw_parts(data, len as usize) };
        // Reborrow with an unbounded lifetime; the caller's documented contract
        // is that `data` outlives the repacketizer (as long as it holds the frame).
        let static_slice: &'static [u8] = unsafe { std::mem::transmute(slice) };
        let ret = inner.cat(static_slice, len);
        unsafe { sync_nb_frames(rp) };
        ret
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_out(
    rp: *mut OpusRepacketizerC,
    data: *mut c_uchar,
    maxlen: i32,
) -> i32 {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if data.is_null() || maxlen < 0 {
            return OPUS_BAD_ARG;
        }
        let Some(inner) = (unsafe { resolve_handle_ref(rp) }) else {
            return OPUS_BAD_ARG;
        };
        let out = unsafe { std::slice::from_raw_parts_mut(data, maxlen as usize) };
        inner.out(out, maxlen)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_out_range(
    rp: *mut OpusRepacketizerC,
    begin: c_int,
    end: c_int,
    data: *mut c_uchar,
    maxlen: i32,
) -> i32 {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if data.is_null() || maxlen < 0 || begin < 0 || end < 0 {
            return OPUS_BAD_ARG;
        }
        let Some(inner) = (unsafe { resolve_handle_ref(rp) }) else {
            return OPUS_BAD_ARG;
        };
        let out = unsafe { std::slice::from_raw_parts_mut(data, maxlen as usize) };
        inner.out_range(begin as usize, end as usize, out, maxlen)
    })
}

/// `opus_repacketizer_out_range_impl` — full control over the emit pipeline
/// with self-delimited framing, padding-to-max, and caller-supplied extension
/// injection. Backs `test_opus_extensions.c`'s
/// `test_opus_repacketizer_out_range_impl`.
///
/// Matches the C signature in `reference/src/opus_private.h:214-216`.
///
/// Note: this wrapper does NOT call `sync_nb_frames` because the underlying
/// ropus method takes `&self` and is non-mutating — emitting a packet does
/// not change `nb_frames`. If a future mutating out variant is added, it
/// must call `sync_nb_frames(rp)` after the inner call to keep the
/// C-visible `rp.nb_frames` mirror in step with the inner state (see
/// `opus_repacketizer_cat` for the pattern).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_out_range_impl(
    rp: *mut OpusRepacketizerC,
    begin: c_int,
    end: c_int,
    data: *mut c_uchar,
    maxlen: i32,
    self_delimited: c_int,
    pad: c_int,
    extensions: *const crate::extensions::OpusExtensionDataC,
    nb_extensions: c_int,
) -> i32 {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if data.is_null() || maxlen < 0 || begin < 0 || end < 0 {
            return OPUS_BAD_ARG;
        }
        if nb_extensions < 0 {
            return OPUS_BAD_ARG;
        }
        if nb_extensions > 0 && extensions.is_null() {
            return OPUS_BAD_ARG;
        }
        let Some(inner) = (unsafe { resolve_handle_ref(rp) }) else {
            return OPUS_BAD_ARG;
        };
        let out = unsafe { std::slice::from_raw_parts_mut(data, maxlen as usize) };

        // Translate C extensions into borrowed Rust slices. The caller
        // guarantees the payload memory outlives this call (stack-allocated
        // in the test).
        use ropus::opus::extensions::OpusExtensionData as RExt;
        let mut owned: Vec<RExt<'static>> = Vec::with_capacity(nb_extensions as usize);
        for i in 0..nb_extensions as usize {
            let raw = unsafe { &*extensions.add(i) };
            let slice: &'static [u8] = if raw.len > 0 && !raw.data.is_null() {
                unsafe {
                    std::mem::transmute::<&[u8], &'static [u8]>(std::slice::from_raw_parts(
                        raw.data,
                        raw.len as usize,
                    ))
                }
            } else {
                &[]
            };
            owned.push(RExt {
                id: raw.id,
                frame: raw.frame,
                data: slice,
                len: raw.len,
            });
        }

        inner.out_range_impl(
            begin as usize,
            end as usize,
            out,
            maxlen,
            self_delimited != 0,
            pad != 0,
            &owned,
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_get_nb_frames(rp: *mut OpusRepacketizerC) -> c_int {
    ffi_guard!(0, {
        let Some(inner) = (unsafe { resolve_handle_ref(rp) }) else {
            return 0;
        };
        inner.get_nb_frames()
    })
}

// ---------------------------------------------------------------------------
// opus_multistream_packet_pad / unpad
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_packet_pad(
    data: *mut c_uchar,
    len: i32,
    new_len: i32,
    nb_streams: c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if data.is_null() || len < 0 || new_len < 0 || nb_streams < 1 {
            return OPUS_BAD_ARG;
        }
        if new_len < len {
            return OPUS_BAD_ARG;
        }
        let buf = unsafe { std::slice::from_raw_parts_mut(data, new_len as usize) };
        ropus::opus::repacketizer::opus_multistream_packet_pad(buf, len, new_len, nb_streams)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_packet_unpad(
    data: *mut c_uchar,
    len: i32,
    nb_streams: c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if data.is_null() || len < 1 || nb_streams < 1 {
            return OPUS_BAD_ARG;
        }
        let buf = unsafe { std::slice::from_raw_parts_mut(data, len as usize) };
        ropus::opus::repacketizer::opus_multistream_packet_unpad(buf, len, nb_streams)
    })
}

// Hush unused if ropus APIs get inlined elsewhere.
#[allow(dead_code)]
fn _link_sink() -> i32 {
    OPUS_OK
}
