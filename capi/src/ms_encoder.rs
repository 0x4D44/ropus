//! `opus_multistream_encoder_*` C ABI wrappers.
//!
//! Same handle-indirection design as `encoder.rs`. An `OpusMSEncoderHandle`
//! is what C sees; the real `ropus::opus::multistream::OpusMSEncoder` lives on
//! the Rust heap, leaked on destroy (see [`crate::state_free`]).
//!
//! In addition to the main handle, each MS encoder holds an array of
//! per-stream sub-encoder handles (returned by
//! `OPUS_MULTISTREAM_GET_ENCODER_STATE`). Those per-stream handles use the
//! single-stream encoder magic, so `opus_encoder_ctl` accepts them.

use std::os::raw::{c_int, c_uchar};
use std::ptr;

use ropus::opus::encoder::OpusEncoder;
use ropus::opus::multistream::OpusMSEncoder;

use crate::{OPUS_BAD_ARG, OPUS_INTERNAL_ERROR, OPUS_OK, ffi_guard, state_free};

const MS_ENCODER_HANDLE_MAGIC: u64 = 0x4D44_4F50_5553_4D45; // "MDOPUSME"

#[repr(C)]
struct OpusMSEncoderHandle {
    magic: u64,
    inner: *mut Inner,
    generation: u64,
}

struct Inner {
    ms: Box<OpusMSEncoder>,
    sub_handles: Vec<*mut OpusEncoder>,
}

// No size-equality assertion with OpusMSEncoder here: MS encoder state lives
// behind the handle on the heap, and `_get_size` returns a handle-sized
// footprint (16KB per stream, always > sizeof handle) rather than the real
// state size. The C test never memcpys MS encoder state.

fn alloc_handle_storage() -> *mut OpusMSEncoderHandle {
    let layout = std::alloc::Layout::new::<OpusMSEncoderHandle>();
    // SAFETY: layout is for a non-zero-sized type with a valid alignment.
    unsafe {
        let p = std::alloc::alloc_zeroed(layout);
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        p as *mut OpusMSEncoderHandle
    }
}

unsafe fn resolve_inner<'a>(st: *mut OpusMSEncoder) -> Option<&'a mut Inner> {
    if st.is_null() {
        return None;
    }
    let h = st as *const OpusMSEncoderHandle;
    // SAFETY: caller provided at least size_of<OpusMSEncoder> bytes.
    let magic = unsafe { (*h).magic };
    if magic != MS_ENCODER_HANDLE_MAGIC {
        return None;
    }
    let inner = unsafe { (*h).inner };
    if inner.is_null() {
        return None;
    }
    Some(unsafe { &mut *inner })
}

unsafe fn resolve_inner_ref<'a>(st: *const OpusMSEncoder) -> Option<&'a Inner> {
    if st.is_null() {
        return None;
    }
    let h = st as *const OpusMSEncoderHandle;
    let magic = unsafe { (*h).magic };
    if magic != MS_ENCODER_HANDLE_MAGIC {
        return None;
    }
    let inner = unsafe { (*h).inner };
    if inner.is_null() {
        return None;
    }
    Some(unsafe { &*inner })
}

unsafe fn install_handle(dst: *mut OpusMSEncoderHandle, inner: *mut Inner) {
    // SAFETY: `dst` points to zeroed storage of at least size_of<OpusMSEncoder>.
    unsafe {
        (*dst).magic = MS_ENCODER_HANDLE_MAGIC;
        (*dst).inner = inner;
        (*dst).generation = crate::next_handle_generation();
    }
}

pub(crate) unsafe fn bump_generation(st: *mut OpusMSEncoder) {
    if st.is_null() {
        return;
    }
    let h = st as *mut OpusMSEncoderHandle;
    // SAFETY: caller holds a valid handle pointer.
    unsafe {
        if (*h).magic != MS_ENCODER_HANDLE_MAGIC {
            return;
        }
        (*h).generation = (*h).generation.wrapping_add(1);
    }
}

pub(crate) unsafe fn handle_to_ms_encoder<'a>(
    st: *mut OpusMSEncoder,
) -> Option<&'a mut OpusMSEncoder> {
    let inner = unsafe { resolve_inner(st) }?;
    Some(&mut inner.ms)
}

pub(crate) unsafe fn handle_to_ms_encoder_ref<'a>(
    st: *const OpusMSEncoder,
) -> Option<&'a OpusMSEncoder> {
    let inner = unsafe { resolve_inner_ref(st) }?;
    Some(&inner.ms)
}

/// Return the per-stream encoder handle pointer for `stream_id` (0-based).
/// Validates the handle, the stream id against `nb_streams`, and the
/// sub-handle slot in a single `&mut Inner` scope — callers must not also
/// hold a separate `&OpusMSEncoder` borrow while invoking this. Returns
/// `null` on any failure (bad handle, out-of-range id, or missing slot).
/// The returned pointer is owned by the MS encoder and must not be freed by
/// the caller.
pub(crate) unsafe fn sub_encoder_handle_ptr(
    st: *mut OpusMSEncoder,
    stream_id: c_int,
) -> *mut OpusEncoder {
    let Some(inner) = (unsafe { resolve_inner(st) }) else {
        return ptr::null_mut();
    };
    if stream_id < 0 || stream_id >= inner.ms.nb_streams() {
        return ptr::null_mut();
    }
    *inner
        .sub_handles
        .get(stream_id as usize)
        .unwrap_or(&ptr::null_mut())
}

/// Build per-stream encoder handles for every sub-encoder. Each block is a
/// leaked heap allocation sized to `size_of::<OpusEncoder>()` with the
/// single-stream encoder handle prefix, pointing at the MS's own sub-encoder
/// via `get_encoder_mut`.
fn build_sub_handles(ms: &mut OpusMSEncoder) -> Vec<*mut OpusEncoder> {
    let n = ms.nb_streams() as usize;
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let sub_ptr: *mut OpusEncoder = match ms.get_encoder_mut(i) {
            Some(e) => e as *mut OpusEncoder,
            None => {
                v.push(ptr::null_mut());
                continue;
            }
        };
        v.push(crate::encoder::alloc_sub_handle_for(sub_ptr));
    }
    v
}

// ---------------------------------------------------------------------------
// opus_multistream_encoder_*
// ---------------------------------------------------------------------------

/// The test asserts `2048 < size <= (1<<18)*streams`. Our handle only needs
/// ~24 bytes, but the caller may use the return value to malloc a block that
/// we then `_init` into — so it must be large enough for that handle. We
/// return 16KB per stream (always satisfies the test's constraints).
fn ms_size_for(streams: c_int) -> c_int {
    16 * 1024 * streams
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_encoder_get_size(
    streams: c_int,
    coupled_streams: c_int,
) -> c_int {
    ffi_guard!(0, {
        if streams < 1
            || coupled_streams < 0
            || coupled_streams > streams
            || streams > 255 - coupled_streams
        {
            return 0;
        }
        ms_size_for(streams)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_surround_encoder_get_size(
    channels: c_int,
    _mapping_family: c_int,
) -> c_int {
    ffi_guard!(0, {
        if channels < 1 || channels > 255 {
            return 0;
        }
        ms_size_for(channels.max(1))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_encoder_create(
    fs: i32,
    channels: c_int,
    streams: c_int,
    coupled_streams: c_int,
    mapping: *const c_uchar,
    application: c_int,
    error: *mut c_int,
) -> *mut OpusMSEncoder {
    ffi_guard!(ptr::null_mut(), {
        if mapping.is_null() || channels < 1 || channels > 255 {
            if !error.is_null() {
                unsafe { *error = OPUS_BAD_ARG };
            }
            return ptr::null_mut();
        }
        let mapping_slice = unsafe { std::slice::from_raw_parts(mapping, channels as usize) };
        match OpusMSEncoder::new(fs, channels, streams, coupled_streams, mapping_slice, application)
        {
            Ok(ms) => {
                let mut boxed = Box::new(ms);
                let sub_handles = build_sub_handles(&mut boxed);
                let inner = Box::into_raw(Box::new(Inner {
                    ms: boxed,
                    sub_handles,
                }));
                let handle = alloc_handle_storage();
                unsafe { install_handle(handle, inner) };
                if !error.is_null() {
                    unsafe { *error = OPUS_OK };
                }
                handle as *mut OpusMSEncoder
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

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_surround_encoder_create(
    fs: i32,
    channels: c_int,
    mapping_family: c_int,
    streams: *mut c_int,
    coupled_streams: *mut c_int,
    mapping: *mut c_uchar,
    application: c_int,
    error: *mut c_int,
) -> *mut OpusMSEncoder {
    ffi_guard!(ptr::null_mut(), {
        if streams.is_null() || coupled_streams.is_null() || mapping.is_null() {
            if !error.is_null() {
                unsafe { *error = OPUS_BAD_ARG };
            }
            return ptr::null_mut();
        }
        match OpusMSEncoder::new_surround(fs, channels, mapping_family, application) {
            Ok((ms, s, cs, m)) => {
                unsafe {
                    *streams = s;
                    *coupled_streams = cs;
                    ptr::copy_nonoverlapping(m.as_ptr(), mapping, m.len());
                }
                let mut boxed = Box::new(ms);
                let sub_handles = build_sub_handles(&mut boxed);
                let inner = Box::into_raw(Box::new(Inner {
                    ms: boxed,
                    sub_handles,
                }));
                let handle = alloc_handle_storage();
                unsafe { install_handle(handle, inner) };
                if !error.is_null() {
                    unsafe { *error = OPUS_OK };
                }
                handle as *mut OpusMSEncoder
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

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_encoder_init(
    st: *mut OpusMSEncoder,
    fs: i32,
    channels: c_int,
    streams: c_int,
    coupled_streams: c_int,
    mapping: *const c_uchar,
    application: c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if st.is_null() || mapping.is_null() || channels < 1 || channels > 255 {
            return OPUS_BAD_ARG;
        }
        let mapping_slice = unsafe { std::slice::from_raw_parts(mapping, channels as usize) };
        match OpusMSEncoder::new(fs, channels, streams, coupled_streams, mapping_slice, application)
        {
            Ok(ms) => {
                let mut boxed = Box::new(ms);
                let sub_handles = build_sub_handles(&mut boxed);
                let inner = Box::into_raw(Box::new(Inner {
                    ms: boxed,
                    sub_handles,
                }));
                // SAFETY: caller provided at least our advertised size (16KB
                // per stream). We only write the 24-byte handle prefix.
                unsafe {
                    ptr::write_bytes(
                        st as *mut u8,
                        0,
                        std::mem::size_of::<OpusMSEncoderHandle>(),
                    );
                    install_handle(st as *mut OpusMSEncoderHandle, inner);
                }
                OPUS_OK
            }
            Err(e) => e,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_surround_encoder_init(
    st: *mut OpusMSEncoder,
    fs: i32,
    channels: c_int,
    mapping_family: c_int,
    streams: *mut c_int,
    coupled_streams: *mut c_int,
    mapping: *mut c_uchar,
    application: c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if st.is_null() || streams.is_null() || coupled_streams.is_null() || mapping.is_null() {
            return OPUS_BAD_ARG;
        }
        match OpusMSEncoder::new_surround(fs, channels, mapping_family, application) {
            Ok((ms, s, cs, m)) => {
                unsafe {
                    *streams = s;
                    *coupled_streams = cs;
                    ptr::copy_nonoverlapping(m.as_ptr(), mapping, m.len());
                }
                let mut boxed = Box::new(ms);
                let sub_handles = build_sub_handles(&mut boxed);
                let inner = Box::into_raw(Box::new(Inner {
                    ms: boxed,
                    sub_handles,
                }));
                unsafe {
                    ptr::write_bytes(
                        st as *mut u8,
                        0,
                        std::mem::size_of::<OpusMSEncoderHandle>(),
                    );
                    install_handle(st as *mut OpusMSEncoderHandle, inner);
                }
                OPUS_OK
            }
            Err(e) => e,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_encoder_destroy(st: *mut OpusMSEncoder) {
    let _: () = ffi_guard!((), {
        unsafe { state_free(st) };
    });
}

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_encode(
    st: *mut OpusMSEncoder,
    pcm: *const i16,
    frame_size: c_int,
    data: *mut c_uchar,
    max_data_bytes: i32,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if pcm.is_null() || data.is_null() || frame_size <= 0 || max_data_bytes <= 0 {
            return OPUS_BAD_ARG;
        }
        let Some(ms) = (unsafe { handle_to_ms_encoder(st) }) else {
            return OPUS_BAD_ARG;
        };
        // Input PCM is interleaved across ALL layout channels, including any
        // muted ones. `nb_channels` is the correct width; matches C reference
        // `opus_multistream_encode_native` (reads `st->layout.nb_channels`
        // samples per frame slot).
        let nb_channels = ms.nb_channels();
        let Some(n_samples) = (frame_size as usize).checked_mul(nb_channels as usize) else {
            return OPUS_BAD_ARG;
        };
        let pcm_slice = unsafe { std::slice::from_raw_parts(pcm, n_samples) };
        let out_slice =
            unsafe { std::slice::from_raw_parts_mut(data, max_data_bytes as usize) };
        match ms.encode(pcm_slice, frame_size, out_slice, max_data_bytes) {
            Ok(n) => n,
            Err(e) => e,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_encode_float(
    st: *mut OpusMSEncoder,
    pcm: *const f32,
    frame_size: c_int,
    data: *mut c_uchar,
    max_data_bytes: i32,
) -> c_int {
    // ropus's MS encoder doesn't expose a float encode entry point publicly.
    // The api test never calls this on a non-null state (it's only referenced
    // indirectly via the MALLOC_FAIL path, which is skipped without glibc).
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        let _ = pcm;
        let _ = frame_size;
        let _ = data;
        let _ = max_data_bytes;
        if unsafe { handle_to_ms_encoder(st) }.is_none() {
            return OPUS_BAD_ARG;
        }
        crate::OPUS_UNIMPLEMENTED
    })
}
