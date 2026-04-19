//! `opus_multistream_decoder_*` C ABI wrappers.
//!
//! Mirror of `ms_encoder.rs`. Per-stream sub-decoder handles use the same
//! magic as single-stream decoder handles so the test can call
//! `opus_decoder_ctl(od, ...)` on them directly.

use std::os::raw::{c_int, c_uchar};
use std::ptr;

use ropus::opus::decoder::OpusDecoder;
use ropus::opus::multistream::OpusMSDecoder;

use crate::{OPUS_BAD_ARG, OPUS_INTERNAL_ERROR, OPUS_OK, ffi_guard, state_free};

const MS_DECODER_HANDLE_MAGIC: u64 = 0x4D44_4F50_5553_4D44; // "MDOPUSMD"

#[repr(C)]
struct OpusMSDecoderHandle {
    magic: u64,
    inner: *mut Inner,
    generation: u64,
}

struct Inner {
    ms: Box<OpusMSDecoder>,
    sub_handles: Vec<*mut OpusDecoder>,
}

// No size-equality assertion with OpusMSDecoder here: MS decoder state lives
// behind the handle on the heap, and `_get_size` returns a handle-sized
// footprint (16KB per stream) rather than the real state size. The C test
// never memcpys MS decoder state.

fn alloc_handle_storage() -> *mut OpusMSDecoderHandle {
    let layout = std::alloc::Layout::new::<OpusMSDecoderHandle>();
    // SAFETY: layout is for a non-zero-sized type with a valid alignment.
    unsafe {
        let p = std::alloc::alloc_zeroed(layout);
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        p as *mut OpusMSDecoderHandle
    }
}

unsafe fn resolve_inner<'a>(st: *mut OpusMSDecoder) -> Option<&'a mut Inner> {
    if st.is_null() {
        return None;
    }
    let h = st as *const OpusMSDecoderHandle;
    // SAFETY: caller provided at least size_of<OpusMSDecoder> bytes.
    let magic = unsafe { (*h).magic };
    if magic != MS_DECODER_HANDLE_MAGIC {
        return None;
    }
    let inner = unsafe { (*h).inner };
    if inner.is_null() {
        return None;
    }
    Some(unsafe { &mut *inner })
}

unsafe fn resolve_inner_ref<'a>(st: *const OpusMSDecoder) -> Option<&'a Inner> {
    if st.is_null() {
        return None;
    }
    let h = st as *const OpusMSDecoderHandle;
    let magic = unsafe { (*h).magic };
    if magic != MS_DECODER_HANDLE_MAGIC {
        return None;
    }
    let inner = unsafe { (*h).inner };
    if inner.is_null() {
        return None;
    }
    Some(unsafe { &*inner })
}

unsafe fn install_handle(dst: *mut OpusMSDecoderHandle, inner: *mut Inner) {
    // SAFETY: `dst` points to zeroed storage of at least size_of<OpusMSDecoder>.
    unsafe {
        (*dst).magic = MS_DECODER_HANDLE_MAGIC;
        (*dst).inner = inner;
        (*dst).generation = crate::next_handle_generation();
    }
}

pub(crate) unsafe fn bump_generation(st: *mut OpusMSDecoder) {
    if st.is_null() {
        return;
    }
    let h = st as *mut OpusMSDecoderHandle;
    // SAFETY: caller holds a valid handle pointer.
    unsafe {
        if (*h).magic != MS_DECODER_HANDLE_MAGIC {
            return;
        }
        (*h).generation = (*h).generation.wrapping_add(1);
    }
}

pub(crate) unsafe fn handle_to_ms_decoder<'a>(
    st: *mut OpusMSDecoder,
) -> Option<&'a mut OpusMSDecoder> {
    let inner = unsafe { resolve_inner(st) }?;
    Some(&mut inner.ms)
}

pub(crate) unsafe fn handle_to_ms_decoder_ref<'a>(
    st: *const OpusMSDecoder,
) -> Option<&'a OpusMSDecoder> {
    let inner = unsafe { resolve_inner_ref(st) }?;
    Some(&inner.ms)
}

/// Return the per-stream decoder handle pointer for `stream_id` (0-based).
/// Validates the handle, the stream id against `nb_streams`, and the
/// sub-handle slot in a single `&mut Inner` scope — callers must not also
/// hold a separate `&OpusMSDecoder` borrow while invoking this. Returns
/// `null` on any failure.
pub(crate) unsafe fn sub_decoder_handle_ptr(
    st: *mut OpusMSDecoder,
    stream_id: c_int,
) -> *mut OpusDecoder {
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

fn build_sub_handles(ms: &mut OpusMSDecoder) -> Vec<*mut OpusDecoder> {
    let n = ms.nb_streams() as usize;
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let sub_ptr: *mut OpusDecoder = match ms.get_decoder_mut(i) {
            Some(d) => d as *mut OpusDecoder,
            None => {
                v.push(ptr::null_mut());
                continue;
            }
        };
        v.push(crate::decoder::alloc_sub_handle_for(sub_ptr));
    }
    v
}

// ---------------------------------------------------------------------------
// opus_multistream_decoder_*
// ---------------------------------------------------------------------------

/// Size returned to the test, in bytes. Must satisfy the api-test bound
/// `2048 < size <= (1<<18)*streams`. Our handle prefix is 24 bytes, so any
/// value > 2048 is enough in practice; we pick 16KB per stream to stay
/// comfortably inside the per-streams cap for all valid configurations.
fn ms_size_for(streams: c_int) -> c_int {
    16 * 1024 * streams
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_decoder_get_size(
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
pub unsafe extern "C" fn opus_multistream_decoder_create(
    fs: i32,
    channels: c_int,
    streams: c_int,
    coupled_streams: c_int,
    mapping: *const c_uchar,
    error: *mut c_int,
) -> *mut OpusMSDecoder {
    ffi_guard!(ptr::null_mut(), {
        if mapping.is_null() || !(1..=255).contains(&channels) {
            if !error.is_null() {
                unsafe { *error = OPUS_BAD_ARG };
            }
            return ptr::null_mut();
        }
        let mapping_slice = unsafe { std::slice::from_raw_parts(mapping, channels as usize) };
        match OpusMSDecoder::new(fs, channels, streams, coupled_streams, mapping_slice) {
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
                handle as *mut OpusMSDecoder
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
pub unsafe extern "C" fn opus_multistream_decoder_init(
    st: *mut OpusMSDecoder,
    fs: i32,
    channels: c_int,
    streams: c_int,
    coupled_streams: c_int,
    mapping: *const c_uchar,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if st.is_null() || mapping.is_null() || !(1..=255).contains(&channels) {
            return OPUS_BAD_ARG;
        }
        let mapping_slice = unsafe { std::slice::from_raw_parts(mapping, channels as usize) };
        match OpusMSDecoder::new(fs, channels, streams, coupled_streams, mapping_slice) {
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
                    ptr::write_bytes(st as *mut u8, 0, std::mem::size_of::<OpusMSDecoderHandle>());
                    install_handle(st as *mut OpusMSDecoderHandle, inner);
                }
                OPUS_OK
            }
            Err(e) => e,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_decoder_destroy(st: *mut OpusMSDecoder) {
    let _: () = ffi_guard!((), {
        unsafe { state_free(st) };
    });
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_decode(
    st: *mut OpusMSDecoder,
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
        let Some(ms) = (unsafe { handle_to_ms_decoder(st) }) else {
            return OPUS_BAD_ARG;
        };
        // Output PCM is interleaved across ALL layout channels, including any
        // muted channels (mapping == 255). Using `nb_streams + nb_coupled_streams`
        // undercounts by the number of muted channels (e.g. the test-suite
        // `MSdec_err` with channels=3, streams=2, coupled=0, mapping=[0,1,255]).
        // Matches C reference `opus_multistream_decode_native` which sizes
        // `pcm` as `frame_size * st->layout.nb_channels`.
        let nb_channels = ms.nb_channels();
        let Some(n_samples) = (frame_size as usize).checked_mul(nb_channels as usize) else {
            return OPUS_BAD_ARG;
        };
        let pcm_slice = unsafe { std::slice::from_raw_parts_mut(pcm, n_samples) };
        let packet: Option<&[u8]> = if plc {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(data, len as usize) })
        };
        match ms.decode(packet, len, pcm_slice, frame_size, decode_fec != 0) {
            Ok(n) => n,
            Err(e) => e,
        }
    })
}

/// Decode a multistream Opus packet to 24-bit-in-i32 interleaved PCM.
///
/// Matches C `opus_multistream_decode24` (reference
/// `opus_multistream_decoder.c:408`). Same PLC / `decode_fec` validation as
/// [`opus_multistream_decode`]; only the output type differs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_decode24(
    st: *mut OpusMSDecoder,
    data: *const c_uchar,
    len: i32,
    pcm: *mut i32,
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
        let Some(ms) = (unsafe { handle_to_ms_decoder(st) }) else {
            return OPUS_BAD_ARG;
        };
        // Output PCM is interleaved across ALL layout channels, including any
        // muted channels (mapping == 255). See opus_multistream_decode for
        // the rationale.
        let nb_channels = ms.nb_channels();
        let Some(n_samples) = (frame_size as usize).checked_mul(nb_channels as usize) else {
            return OPUS_BAD_ARG;
        };
        let pcm_slice = unsafe { std::slice::from_raw_parts_mut(pcm, n_samples) };
        let packet: Option<&[u8]> = if plc {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(data, len as usize) })
        };
        match ms.decode24(packet, len, pcm_slice, frame_size, decode_fec != 0) {
            Ok(n) => n,
            Err(e) => e,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_decode_float(
    st: *mut OpusMSDecoder,
    data: *const c_uchar,
    len: i32,
    pcm: *mut f32,
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
        let Some(ms) = (unsafe { handle_to_ms_decoder(st) }) else {
            return OPUS_BAD_ARG;
        };
        // Same rationale as the i16 path above: size by total layout
        // channels (including muted), not the physically-routed count.
        let nb_channels = ms.nb_channels();
        let Some(n_samples) = (frame_size as usize).checked_mul(nb_channels as usize) else {
            return OPUS_BAD_ARG;
        };
        let mut tmp = vec![0i16; n_samples];
        let packet: Option<&[u8]> = if plc {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(data, len as usize) })
        };
        let ret = match ms.decode(packet, len, &mut tmp, frame_size, decode_fec != 0) {
            Ok(n) => n,
            Err(e) => return e,
        };
        let out_n = (ret as usize).saturating_mul(nb_channels as usize);
        let out_slice = unsafe { std::slice::from_raw_parts_mut(pcm, out_n) };
        let scale = 1.0f32 / 32768.0f32;
        for i in 0..out_n {
            out_slice[i] = (tmp[i] as f32) * scale;
        }
        ret
    })
}
