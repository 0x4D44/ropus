//! `opus_projection_*` C ABI wrappers.
//!
//! Mirrors the handle-indirection design of [`crate::ms_encoder`] /
//! [`crate::ms_decoder`]. The real state lives on the Rust heap as
//! `OpusProjectionEncoder` / `OpusProjectionDecoder`; the pointer returned
//! to C is a small POD with magic, inner pointer, and generation counter.
//!
//! Scope: the surface `reference/tests/test_opus_projection.c` needs to
//! compile and link, matching the HLD Piece B entry-point list. Float
//! encode/decode forward to the underlying Rust state which currently lacks
//! a float path, so those wrappers return `OPUS_UNIMPLEMENTED` (same pattern
//! as `opus_multistream_encode_float`).

use std::os::raw::{c_int, c_uchar};
use std::ptr;

use ropus::opus::multistream::{OpusProjectionDecoder, OpusProjectionEncoder};

use crate::{OPUS_BAD_ARG, OPUS_INTERNAL_ERROR, OPUS_OK, OPUS_UNIMPLEMENTED, ffi_guard, state_free};

const PROJ_ENCODER_HANDLE_MAGIC: u64 = 0x4D44_4F50_5553_5045; // "MDOPUSPE"
const PROJ_DECODER_HANDLE_MAGIC: u64 = 0x4D44_4F50_5553_5044; // "MDOPUSPD"

#[repr(C)]
struct OpusProjectionEncoderHandle {
    magic: u64,
    inner: *mut OpusProjectionEncoder,
    generation: u64,
}

#[repr(C)]
struct OpusProjectionDecoderHandle {
    magic: u64,
    inner: *mut OpusProjectionDecoder,
    generation: u64,
}

// Storage footprint returned to C. The handle prefix is 24 bytes; we return
// a comfortably larger value so
// `malloc(opus_projection_ambisonics_encoder_get_size(...)) + _init(...)`
// always has room. The C test never memcpys projection state.
fn proj_enc_size_for(channels: c_int) -> c_int {
    16 * 1024 * channels.max(1)
}

fn proj_dec_size_for(streams: c_int) -> c_int {
    16 * 1024 * streams.max(1)
}

fn alloc_enc_handle_storage() -> *mut OpusProjectionEncoderHandle {
    let layout = std::alloc::Layout::new::<OpusProjectionEncoderHandle>();
    // SAFETY: layout is non-zero-sized with valid alignment.
    unsafe {
        let p = std::alloc::alloc_zeroed(layout);
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        p as *mut OpusProjectionEncoderHandle
    }
}

fn alloc_dec_handle_storage() -> *mut OpusProjectionDecoderHandle {
    let layout = std::alloc::Layout::new::<OpusProjectionDecoderHandle>();
    // SAFETY: layout is non-zero-sized with valid alignment.
    unsafe {
        let p = std::alloc::alloc_zeroed(layout);
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        p as *mut OpusProjectionDecoderHandle
    }
}

unsafe fn resolve_enc<'a>(st: *mut OpusProjectionEncoder) -> Option<&'a mut OpusProjectionEncoder> {
    if st.is_null() {
        return None;
    }
    let h = st as *const OpusProjectionEncoderHandle;
    // SAFETY: caller provided a handle pointer with at least the 24-byte
    // prefix populated by install_enc_handle.
    let magic = unsafe { (*h).magic };
    if magic != PROJ_ENCODER_HANDLE_MAGIC {
        return None;
    }
    let inner = unsafe { (*h).inner };
    if inner.is_null() {
        return None;
    }
    Some(unsafe { &mut *inner })
}

unsafe fn resolve_enc_ref<'a>(
    st: *const OpusProjectionEncoder,
) -> Option<&'a OpusProjectionEncoder> {
    if st.is_null() {
        return None;
    }
    let h = st as *const OpusProjectionEncoderHandle;
    let magic = unsafe { (*h).magic };
    if magic != PROJ_ENCODER_HANDLE_MAGIC {
        return None;
    }
    let inner = unsafe { (*h).inner };
    if inner.is_null() {
        return None;
    }
    Some(unsafe { &*inner })
}

unsafe fn resolve_dec<'a>(st: *mut OpusProjectionDecoder) -> Option<&'a mut OpusProjectionDecoder> {
    if st.is_null() {
        return None;
    }
    let h = st as *const OpusProjectionDecoderHandle;
    let magic = unsafe { (*h).magic };
    if magic != PROJ_DECODER_HANDLE_MAGIC {
        return None;
    }
    let inner = unsafe { (*h).inner };
    if inner.is_null() {
        return None;
    }
    Some(unsafe { &mut *inner })
}

unsafe fn resolve_dec_ref<'a>(
    st: *const OpusProjectionDecoder,
) -> Option<&'a OpusProjectionDecoder> {
    if st.is_null() {
        return None;
    }
    let h = st as *const OpusProjectionDecoderHandle;
    let magic = unsafe { (*h).magic };
    if magic != PROJ_DECODER_HANDLE_MAGIC {
        return None;
    }
    let inner = unsafe { (*h).inner };
    if inner.is_null() {
        return None;
    }
    Some(unsafe { &*inner })
}

unsafe fn install_enc_handle(
    dst: *mut OpusProjectionEncoderHandle,
    inner: *mut OpusProjectionEncoder,
) {
    // SAFETY: caller provided zero-initialised storage of at least
    // size_of::<OpusProjectionEncoderHandle>.
    unsafe {
        (*dst).magic = PROJ_ENCODER_HANDLE_MAGIC;
        (*dst).inner = inner;
        (*dst).generation = crate::next_handle_generation();
    }
}

unsafe fn install_dec_handle(
    dst: *mut OpusProjectionDecoderHandle,
    inner: *mut OpusProjectionDecoder,
) {
    // SAFETY: caller provided zero-initialised storage.
    unsafe {
        (*dst).magic = PROJ_DECODER_HANDLE_MAGIC;
        (*dst).inner = inner;
        (*dst).generation = crate::next_handle_generation();
    }
}

pub(crate) unsafe fn bump_enc_generation(st: *mut OpusProjectionEncoder) {
    if st.is_null() {
        return;
    }
    let h = st as *mut OpusProjectionEncoderHandle;
    // SAFETY: caller holds a valid handle pointer.
    unsafe {
        if (*h).magic != PROJ_ENCODER_HANDLE_MAGIC {
            return;
        }
        (*h).generation = (*h).generation.wrapping_add(1);
    }
}

pub(crate) unsafe fn bump_dec_generation(st: *mut OpusProjectionDecoder) {
    if st.is_null() {
        return;
    }
    let h = st as *mut OpusProjectionDecoderHandle;
    // SAFETY: caller holds a valid handle pointer.
    unsafe {
        if (*h).magic != PROJ_DECODER_HANDLE_MAGIC {
            return;
        }
        (*h).generation = (*h).generation.wrapping_add(1);
    }
}

pub(crate) unsafe fn handle_to_proj_encoder<'a>(
    st: *mut OpusProjectionEncoder,
) -> Option<&'a mut OpusProjectionEncoder> {
    unsafe { resolve_enc(st) }
}

pub(crate) unsafe fn handle_to_proj_encoder_ref<'a>(
    st: *const OpusProjectionEncoder,
) -> Option<&'a OpusProjectionEncoder> {
    unsafe { resolve_enc_ref(st) }
}

pub(crate) unsafe fn handle_to_proj_decoder<'a>(
    st: *mut OpusProjectionDecoder,
) -> Option<&'a mut OpusProjectionDecoder> {
    unsafe { resolve_dec(st) }
}

#[allow(dead_code)]
pub(crate) unsafe fn handle_to_proj_decoder_ref<'a>(
    st: *const OpusProjectionDecoder,
) -> Option<&'a OpusProjectionDecoder> {
    unsafe { resolve_dec_ref(st) }
}

// ---------------------------------------------------------------------------
// opus_projection_ambisonics_encoder_*
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_ambisonics_encoder_get_size(
    channels: c_int,
    _mapping_family: c_int,
) -> c_int {
    ffi_guard!(0, {
        if channels < 1 || channels > 255 {
            return 0;
        }
        proj_enc_size_for(channels)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_ambisonics_encoder_create(
    fs: i32,
    channels: c_int,
    mapping_family: c_int,
    streams: *mut c_int,
    coupled_streams: *mut c_int,
    application: c_int,
    error: *mut c_int,
) -> *mut OpusProjectionEncoder {
    ffi_guard!(ptr::null_mut(), {
        if streams.is_null() || coupled_streams.is_null() {
            if !error.is_null() {
                unsafe { *error = OPUS_BAD_ARG };
            }
            return ptr::null_mut();
        }
        match OpusProjectionEncoder::new(fs, channels, mapping_family, application) {
            Ok((enc, s, cs)) => {
                unsafe {
                    *streams = s;
                    *coupled_streams = cs;
                }
                let inner = Box::into_raw(Box::new(enc));
                let handle = alloc_enc_handle_storage();
                // SAFETY: freshly zero-allocated storage.
                unsafe { install_enc_handle(handle, inner) };
                if !error.is_null() {
                    unsafe { *error = OPUS_OK };
                }
                handle as *mut OpusProjectionEncoder
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
pub unsafe extern "C" fn opus_projection_ambisonics_encoder_init(
    st: *mut OpusProjectionEncoder,
    fs: i32,
    channels: c_int,
    mapping_family: c_int,
    streams: *mut c_int,
    coupled_streams: *mut c_int,
    application: c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if st.is_null() || streams.is_null() || coupled_streams.is_null() {
            return OPUS_BAD_ARG;
        }
        match OpusProjectionEncoder::new(fs, channels, mapping_family, application) {
            Ok((enc, s, cs)) => {
                unsafe {
                    *streams = s;
                    *coupled_streams = cs;
                }
                let inner = Box::into_raw(Box::new(enc));
                // SAFETY: caller provided at least our advertised size.
                unsafe {
                    ptr::write_bytes(
                        st as *mut u8,
                        0,
                        std::mem::size_of::<OpusProjectionEncoderHandle>(),
                    );
                    install_enc_handle(st as *mut OpusProjectionEncoderHandle, inner);
                }
                OPUS_OK
            }
            Err(e) => e,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_encoder_destroy(st: *mut OpusProjectionEncoder) {
    let _: () = ffi_guard!((), {
        unsafe { state_free(st) };
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_encode(
    st: *mut OpusProjectionEncoder,
    pcm: *const i16,
    frame_size: c_int,
    data: *mut c_uchar,
    max_data_bytes: i32,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if pcm.is_null() || data.is_null() || frame_size <= 0 || max_data_bytes <= 0 {
            return OPUS_BAD_ARG;
        }
        let Some(enc) = (unsafe { resolve_enc(st) }) else {
            return OPUS_BAD_ARG;
        };
        let channels = enc.nb_channels();
        let Some(n_samples) = (frame_size as usize).checked_mul(channels as usize) else {
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

/// Float-PCM projection encode. The underlying Rust
/// [`OpusProjectionEncoder`] does not currently expose a float path — the
/// conformance projection test only exercises the i16 path — so we report
/// `OPUS_UNIMPLEMENTED` when called with a valid handle. Mirrors
/// [`crate::ms_encoder::opus_multistream_encode_float`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_encode_float(
    st: *mut OpusProjectionEncoder,
    pcm: *const f32,
    frame_size: c_int,
    data: *mut c_uchar,
    max_data_bytes: i32,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        let _ = pcm;
        let _ = frame_size;
        let _ = data;
        let _ = max_data_bytes;
        if unsafe { resolve_enc(st) }.is_none() {
            return OPUS_BAD_ARG;
        }
        OPUS_UNIMPLEMENTED
    })
}

// ---------------------------------------------------------------------------
// opus_projection_decoder_*
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_decoder_get_size(
    channels: c_int,
    streams: c_int,
    coupled_streams: c_int,
) -> c_int {
    ffi_guard!(0, {
        if streams < 1
            || coupled_streams < 0
            || coupled_streams > streams
            || channels < 1
            || channels > 255
        {
            return 0;
        }
        proj_dec_size_for(streams)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_decoder_create(
    fs: i32,
    channels: c_int,
    streams: c_int,
    coupled_streams: c_int,
    demixing_matrix: *const c_uchar,
    demixing_matrix_size: i32,
    error: *mut c_int,
) -> *mut OpusProjectionDecoder {
    ffi_guard!(ptr::null_mut(), {
        if demixing_matrix.is_null() || demixing_matrix_size < 0 {
            if !error.is_null() {
                unsafe { *error = OPUS_BAD_ARG };
            }
            return ptr::null_mut();
        }
        let bytes = unsafe {
            std::slice::from_raw_parts(demixing_matrix, demixing_matrix_size as usize)
        };
        match OpusProjectionDecoder::new(
            fs,
            channels,
            streams,
            coupled_streams,
            bytes,
            demixing_matrix_size,
        ) {
            Ok(dec) => {
                let inner = Box::into_raw(Box::new(dec));
                let handle = alloc_dec_handle_storage();
                // SAFETY: freshly zero-allocated storage.
                unsafe { install_dec_handle(handle, inner) };
                if !error.is_null() {
                    unsafe { *error = OPUS_OK };
                }
                handle as *mut OpusProjectionDecoder
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
pub unsafe extern "C" fn opus_projection_decoder_init(
    st: *mut OpusProjectionDecoder,
    fs: i32,
    channels: c_int,
    streams: c_int,
    coupled_streams: c_int,
    demixing_matrix: *const c_uchar,
    demixing_matrix_size: i32,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if st.is_null() || demixing_matrix.is_null() || demixing_matrix_size < 0 {
            return OPUS_BAD_ARG;
        }
        let bytes = unsafe {
            std::slice::from_raw_parts(demixing_matrix, demixing_matrix_size as usize)
        };
        match OpusProjectionDecoder::new(
            fs,
            channels,
            streams,
            coupled_streams,
            bytes,
            demixing_matrix_size,
        ) {
            Ok(dec) => {
                let inner = Box::into_raw(Box::new(dec));
                // SAFETY: caller provided at least our advertised size.
                unsafe {
                    ptr::write_bytes(
                        st as *mut u8,
                        0,
                        std::mem::size_of::<OpusProjectionDecoderHandle>(),
                    );
                    install_dec_handle(st as *mut OpusProjectionDecoderHandle, inner);
                }
                OPUS_OK
            }
            Err(e) => e,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_decoder_destroy(st: *mut OpusProjectionDecoder) {
    let _: () = ffi_guard!((), {
        unsafe { state_free(st) };
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_decode(
    st: *mut OpusProjectionDecoder,
    data: *const c_uchar,
    len: i32,
    pcm: *mut i16,
    frame_size: c_int,
    decode_fec: c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if pcm.is_null() || frame_size <= 0 {
            return OPUS_BAD_ARG;
        }
        if decode_fec != 0 && decode_fec != 1 {
            return OPUS_BAD_ARG;
        }
        let plc = data.is_null() || len == 0;
        if !plc && len < 0 {
            return OPUS_BAD_ARG;
        }
        let Some(dec) = (unsafe { resolve_dec(st) }) else {
            return OPUS_BAD_ARG;
        };
        // Output slice is sized by the output channel count (layout
        // nb_channels), matching the C reference.
        let nb_channels = dec.nb_channels();
        let Some(n_samples) = (frame_size as usize).checked_mul(nb_channels as usize) else {
            return OPUS_BAD_ARG;
        };
        let pcm_slice = unsafe { std::slice::from_raw_parts_mut(pcm, n_samples) };
        let packet: Option<&[u8]> = if plc {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(data, len as usize) })
        };
        match dec.decode(packet, len, pcm_slice, frame_size, decode_fec != 0) {
            Ok(n) => n,
            Err(e) => e,
        }
    })
}

/// Float-PCM projection decode. Not implemented — see
/// [`opus_projection_encode_float`] for rationale.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_decode_float(
    st: *mut OpusProjectionDecoder,
    data: *const c_uchar,
    len: i32,
    pcm: *mut f32,
    frame_size: c_int,
    decode_fec: c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        let _ = data;
        let _ = len;
        let _ = pcm;
        let _ = frame_size;
        let _ = decode_fec;
        if unsafe { resolve_dec(st) }.is_none() {
            return OPUS_BAD_ARG;
        }
        OPUS_UNIMPLEMENTED
    })
}
