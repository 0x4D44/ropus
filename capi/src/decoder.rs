//! `opus_decoder_*` C ABI wrappers.
//!
//! State lifecycle:
//! - `_create` heap-allocates a `Box<OpusDecoder>` and leaks it into a raw
//!   pointer; `_destroy` reclaims it with `Box::from_raw`.
//! - `_get_size` reports the Rust struct size; `_init` uses `ptr::write` to
//!   fresh-initialise into caller-allocated storage of at least that size.
//! - Our `OpusDecoder` has no `Drop`, but holds `Option<Vec<_>>` sub-state
//!   which drops naturally when the box is destroyed.

use std::os::raw::{c_int, c_uchar};
use std::ptr;

use ropus::opus::decoder::OpusDecoder;

use crate::{OPUS_BAD_ARG, OPUS_INTERNAL_ERROR, ffi_guard};

// ---------------------------------------------------------------------------
// opus_decoder_get_size / create / init / destroy
// ---------------------------------------------------------------------------

/// Size of the decoder state in bytes; `0` for an unsupported channel count.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decoder_get_size(channels: c_int) -> c_int {
    ffi_guard!(0, {
        if channels != 1 && channels != 2 {
            return 0;
        }
        std::mem::size_of::<OpusDecoder>() as c_int
    })
}

/// Initialise a decoder into caller-allocated storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decoder_init(
    st: *mut OpusDecoder,
    fs: i32,
    channels: c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if st.is_null() {
            return OPUS_BAD_ARG;
        }
        match OpusDecoder::new(fs, channels) {
            Ok(dec) => {
                debug_assert_eq!(
                    std::mem::size_of::<OpusDecoder>(),
                    unsafe { opus_decoder_get_size(channels) } as usize,
                    "OpusDecoder size mismatch between capi and consumer view"
                );
                unsafe { ptr::write(st, dec) };
                0 // OPUS_OK
            }
            Err(e) => e,
        }
    })
}

/// Allocate + initialise a decoder; returns NULL and writes `*error` on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decoder_create(
    fs: i32,
    channels: c_int,
    error: *mut c_int,
) -> *mut OpusDecoder {
    ffi_guard!(ptr::null_mut(), {
        match OpusDecoder::new(fs, channels) {
            Ok(dec) => {
                if !error.is_null() {
                    unsafe { *error = 0 };
                }
                Box::into_raw(Box::new(dec))
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

/// Reclaim a heap-allocated decoder. No-op on NULL (matches C reference).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decoder_destroy(st: *mut OpusDecoder) {
    let _: () = ffi_guard!((), {
        if !st.is_null() {
            drop(unsafe { Box::from_raw(st) });
        }
    });
}

// ---------------------------------------------------------------------------
// opus_decode / opus_decode_float
// ---------------------------------------------------------------------------

/// Decode an Opus packet to 16-bit PCM. `data=NULL,len=0` requests PLC.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decode(
    st: *mut OpusDecoder,
    data: *const c_uchar,
    len: i32,
    pcm: *mut i16,
    frame_size: c_int,
    decode_fec: c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if st.is_null() || pcm.is_null() || frame_size <= 0 {
            return OPUS_BAD_ARG;
        }
        if len < 0 {
            return OPUS_BAD_ARG;
        }
        // data pointer may be NULL only when len==0 (PLC / lost packet).
        if data.is_null() && len != 0 {
            return OPUS_BAD_ARG;
        }

        let packet: Option<&[u8]> = if data.is_null() || len == 0 {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(data, len as usize) })
        };

        let dec = unsafe { &mut *st };
        let channels = dec.get_channels() as usize;
        let Some(n_samples) = (frame_size as usize).checked_mul(channels) else {
            return OPUS_BAD_ARG;
        };
        let pcm_slice = unsafe { std::slice::from_raw_parts_mut(pcm, n_samples) };

        match dec.decode(packet, pcm_slice, frame_size, decode_fec != 0) {
            Ok(n) => n as c_int,
            Err(e) => e,
        }
    })
}

/// Decode an Opus packet to 32-bit float PCM.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decode_float(
    st: *mut OpusDecoder,
    data: *const c_uchar,
    len: i32,
    pcm: *mut f32,
    frame_size: c_int,
    decode_fec: c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if st.is_null() || pcm.is_null() || frame_size <= 0 {
            return OPUS_BAD_ARG;
        }
        if len < 0 {
            return OPUS_BAD_ARG;
        }
        if data.is_null() && len != 0 {
            return OPUS_BAD_ARG;
        }

        let packet: Option<&[u8]> = if data.is_null() || len == 0 {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(data, len as usize) })
        };

        let dec = unsafe { &mut *st };
        let channels = dec.get_channels() as usize;
        let Some(n_samples) = (frame_size as usize).checked_mul(channels) else {
            return OPUS_BAD_ARG;
        };
        let pcm_slice = unsafe { std::slice::from_raw_parts_mut(pcm, n_samples) };

        match dec.decode_float(packet, pcm_slice, frame_size, decode_fec != 0) {
            Ok(n) => n as c_int,
            Err(e) => e,
        }
    })
}

// ---------------------------------------------------------------------------
// opus_packet_pad / opus_packet_unpad (thin wrappers; functions already exist
// in ropus::opus::repacketizer). Phase-1-listed deliverable.
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_pad(
    data: *mut c_uchar,
    len: i32,
    new_len: i32,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if data.is_null() || len < 0 || new_len < 0 {
            return OPUS_BAD_ARG;
        }
        if new_len < len {
            return OPUS_BAD_ARG;
        }
        // We need writable storage up to `new_len`, but the C ABI doesn't
        // carry that capacity — the caller must have allocated at least
        // `new_len` bytes, matching reference behaviour.
        let buf = unsafe { std::slice::from_raw_parts_mut(data, new_len as usize) };
        ropus::opus::repacketizer::opus_packet_pad(buf, len, new_len)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_unpad(data: *mut c_uchar, len: i32) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if data.is_null() || len < 1 {
            return OPUS_BAD_ARG;
        }
        let buf = unsafe { std::slice::from_raw_parts_mut(data, len as usize) };
        ropus::opus::repacketizer::opus_packet_unpad(buf, len)
    })
}
