//! `opus_encoder_*` C ABI wrappers.
//!
//! State lifecycle mirrors `decoder.rs`: `_create`/`_destroy` use `Box::into_raw`
//! / `Box::from_raw`, while `_init` writes into caller storage sized by `_get_size`.

use std::os::raw::{c_int, c_uchar};
use std::ptr;

use ropus::opus::encoder::OpusEncoder;

use crate::{OPUS_BAD_ARG, OPUS_INTERNAL_ERROR, ffi_guard};

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
                debug_assert_eq!(
                    std::mem::size_of::<OpusEncoder>(),
                    unsafe { opus_encoder_get_size(channels) } as usize,
                    "OpusEncoder size mismatch between capi and consumer view"
                );
                unsafe { ptr::write(st, enc) };
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
                Box::into_raw(Box::new(enc))
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
pub unsafe extern "C" fn opus_encoder_destroy(st: *mut OpusEncoder) {
    let _: () = ffi_guard!((), {
        if !st.is_null() {
            drop(unsafe { Box::from_raw(st) });
        }
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
        if st.is_null() || pcm.is_null() || data.is_null() || frame_size <= 0 || max_data_bytes <= 0
        {
            return OPUS_BAD_ARG;
        }
        let enc = unsafe { &mut *st };
        let channels = enc.channels as usize;
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
        if st.is_null() || pcm.is_null() || data.is_null() || frame_size <= 0 || max_data_bytes <= 0
        {
            return OPUS_BAD_ARG;
        }
        let enc = unsafe { &mut *st };
        let channels = enc.channels as usize;
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
