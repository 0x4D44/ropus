//! `opus_encoder_*` C ABI wrappers.
//!
//! Uses the same handle-indirection pattern as [`crate::decoder`]: the state
//! pointer returned to C is an `OpusEncoderHandle` POD whose first bytes carry
//! a magic + a pointer to the real Rust-heap encoder. The handle is sized to
//! match `size_of::<OpusEncoder>()` so tests that do
//! `enc = malloc(opus_encoder_get_size(c)); opus_encoder_init(enc, ...)` see
//! what they expect. `_destroy` is a no-op (leak); see [`crate::state_free`]
//! for the rationale.
//!
//! Encoder and decoder handles use distinct magic values so that memcpying a
//! decoder's bytes on top of an encoder (or vice versa) is detected.

use std::os::raw::{c_int, c_uchar};
use std::ptr;

use ropus::opus::encoder::OpusEncoder;

use crate::{OPUS_BAD_ARG, OPUS_INTERNAL_ERROR, ffi_guard, state_free};

// ---------------------------------------------------------------------------
// Handle layout
// ---------------------------------------------------------------------------

const ENCODER_HANDLE_MAGIC: u64 = 0x4D44_4F50_5553_4543; // "MDOPUSEC"

/// Opaque POD wrapper visible to C. First 24 bytes are our prefix (magic,
/// inner pointer, generation); the rest is padding sized to the real
/// `OpusEncoder` so caller-allocated buffers (`malloc(opus_encoder_get_size)`)
/// always have room.
///
/// `generation` is bumped on `OPUS_RESET_STATE` and stamped with a globally
/// unique value on create/init, so `memcmp` between two handles or between a
/// pre-reset snapshot and the post-reset handle always observes a difference.
#[repr(C)]
struct OpusEncoderHandle {
    magic: u64,
    inner: *mut OpusEncoder,
    generation: u64,
}

const _: () = {
    assert!(
        core::mem::size_of::<OpusEncoderHandle>() <= core::mem::size_of::<OpusEncoder>(),
        "OpusEncoderHandle must fit within OpusEncoder bytes"
    );
};

fn alloc_handle_storage() -> *mut OpusEncoderHandle {
    let size = std::mem::size_of::<OpusEncoder>();
    let align = std::mem::align_of::<OpusEncoder>();
    // SAFETY: size and align are both non-zero, derived from a valid type.
    unsafe {
        let layout = std::alloc::Layout::from_size_align_unchecked(size, align);
        let p = std::alloc::alloc_zeroed(layout);
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        p as *mut OpusEncoderHandle
    }
}

unsafe fn resolve_handle<'a>(st: *mut OpusEncoder) -> Option<&'a mut OpusEncoder> {
    if st.is_null() {
        return None;
    }
    let h = st as *const OpusEncoderHandle;
    // SAFETY: caller promised `st` points to at least `size_of::<OpusEncoder>()` bytes.
    let magic = unsafe { (*h).magic };
    if magic != ENCODER_HANDLE_MAGIC {
        return None;
    }
    let inner = unsafe { (*h).inner };
    if inner.is_null() {
        return None;
    }
    Some(unsafe { &mut *inner })
}

unsafe fn resolve_handle_ref<'a>(st: *const OpusEncoder) -> Option<&'a OpusEncoder> {
    if st.is_null() {
        return None;
    }
    let h = st as *const OpusEncoderHandle;
    let magic = unsafe { (*h).magic };
    if magic != ENCODER_HANDLE_MAGIC {
        return None;
    }
    let inner = unsafe { (*h).inner };
    if inner.is_null() {
        return None;
    }
    Some(unsafe { &*inner })
}

unsafe fn install_handle(dst: *mut OpusEncoderHandle, inner: *mut OpusEncoder) {
    // SAFETY: `dst` points to zeroed storage of at least size_of::<OpusEncoder>.
    unsafe {
        (*dst).magic = ENCODER_HANDLE_MAGIC;
        (*dst).inner = inner;
        (*dst).generation = crate::next_handle_generation();
    }
}

/// Bump the handle's generation counter. Called on `OPUS_RESET_STATE`.
pub(crate) unsafe fn bump_generation(st: *mut OpusEncoder) {
    if st.is_null() {
        return;
    }
    let h = st as *mut OpusEncoderHandle;
    // SAFETY: caller holds a valid handle pointer.
    unsafe {
        if (*h).magic != ENCODER_HANDLE_MAGIC {
            return;
        }
        (*h).generation = (*h).generation.wrapping_add(1);
    }
}

// ---------------------------------------------------------------------------
// opus_encoder_get_size / create / init / destroy
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_encoder_get_size(channels: c_int) -> c_int {
    ffi_guard!(0, {
        if channels != 1 && channels != 2 {
            return 0;
        }
        std::mem::size_of::<OpusEncoder>() as c_int
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_encoder_init(
    st: *mut OpusEncoder,
    fs: i32,
    channels: c_int,
    application: c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if st.is_null() {
            return OPUS_BAD_ARG;
        }
        match OpusEncoder::new(fs, channels, application) {
            Ok(enc) => {
                let inner = Box::into_raw(Box::new(enc));
                // SAFETY: caller provided at least size_of::<OpusEncoder>() bytes.
                unsafe {
                    ptr::write_bytes(st as *mut u8, 0, std::mem::size_of::<OpusEncoder>());
                    install_handle(st as *mut OpusEncoderHandle, inner);
                }
                0 // OPUS_OK
            }
            Err(e) => e,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_encoder_create(
    fs: i32,
    channels: c_int,
    application: c_int,
    error: *mut c_int,
) -> *mut OpusEncoder {
    ffi_guard!(ptr::null_mut(), {
        match OpusEncoder::new(fs, channels, application) {
            Ok(enc) => {
                if !error.is_null() {
                    unsafe { *error = 0 };
                }
                let inner = Box::into_raw(Box::new(enc));
                let handle = alloc_handle_storage();
                unsafe { install_handle(handle, inner) };
                handle as *mut OpusEncoder
            }
            Err(e) => {
                if !error.is_null() {
                    unsafe { *error = e };
                }
                ptr::null_mut()
            }
        }
    })
}

/// No-op: state is leaked, matching decoder behaviour.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_encoder_destroy(st: *mut OpusEncoder) {
    let _: () = ffi_guard!((), {
        unsafe { state_free(st) };
    });
}

// ---------------------------------------------------------------------------
// opus_encode / opus_encode_float
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_encode(
    st: *mut OpusEncoder,
    pcm: *const i16,
    frame_size: c_int,
    data: *mut c_uchar,
    max_data_bytes: i32,
) -> i32 {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if pcm.is_null() || data.is_null() || frame_size <= 0 || max_data_bytes <= 0 {
            return OPUS_BAD_ARG;
        }
        let Some(enc) = (unsafe { resolve_handle(st) }) else {
            return OPUS_BAD_ARG;
        };
        let channels = enc.get_channels() as usize;
        let Some(n_samples) = (frame_size as usize).checked_mul(channels) else {
            return OPUS_BAD_ARG;
        };
        let pcm_slice = unsafe { std::slice::from_raw_parts(pcm, n_samples) };
        let out_slice =
            unsafe { std::slice::from_raw_parts_mut(data, max_data_bytes as usize) };
        match enc.encode(pcm_slice, frame_size, out_slice, max_data_bytes) {
            Ok(n) => n,
            Err(e) => e,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_encode_float(
    st: *mut OpusEncoder,
    pcm: *const f32,
    frame_size: c_int,
    data: *mut c_uchar,
    max_data_bytes: i32,
) -> i32 {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if pcm.is_null() || data.is_null() || frame_size <= 0 || max_data_bytes <= 0 {
            return OPUS_BAD_ARG;
        }
        let Some(enc) = (unsafe { resolve_handle(st) }) else {
            return OPUS_BAD_ARG;
        };
        let channels = enc.get_channels() as usize;
        let Some(n_samples) = (frame_size as usize).checked_mul(channels) else {
            return OPUS_BAD_ARG;
        };
        let pcm_slice = unsafe { std::slice::from_raw_parts(pcm, n_samples) };
        let out_slice =
            unsafe { std::slice::from_raw_parts_mut(data, max_data_bytes as usize) };
        match enc.encode_float(pcm_slice, frame_size, out_slice, max_data_bytes) {
            Ok(n) => n,
            Err(e) => e,
        }
    })
}

// ---------------------------------------------------------------------------
// Accessors for CTL dispatch and per-stream MS handles
// ---------------------------------------------------------------------------

pub(crate) unsafe fn handle_to_encoder<'a>(st: *mut OpusEncoder) -> Option<&'a mut OpusEncoder> {
    unsafe { resolve_handle(st) }
}

pub(crate) unsafe fn handle_to_encoder_ref<'a>(st: *const OpusEncoder) -> Option<&'a OpusEncoder> {
    unsafe { resolve_handle_ref(st) }
}

/// Build a per-stream encoder handle block whose `inner` points at an existing
/// `OpusEncoder` (typically one inside `OpusMSEncoder`'s `Vec<OpusEncoder>`).
/// The block is `size_of::<OpusEncoder>()` bytes so it is acceptable to every
/// `opus_encoder_*` entry point. Returned pointer is leaked (owned by the MS
/// encoder handle's inner struct).
pub(crate) fn alloc_sub_handle_for(target: *mut OpusEncoder) -> *mut OpusEncoder {
    let handle = alloc_handle_storage();
    // SAFETY: `handle` points to zero-initialised storage of size_of<OpusEncoder>.
    unsafe { install_handle(handle, target) };
    handle as *mut OpusEncoder
}
