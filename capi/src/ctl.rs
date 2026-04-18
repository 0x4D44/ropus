//! Typed CTL dispatchers called by `ctl_shim.c`.
//!
//! The C varargs `opus_*_ctl(st, request, ...)` surface is unpacked in
//! `ctl_shim.c`, which then calls one of these typed entry points with a
//! fully-typed scalar argument. Each function does its own panic-guarding so
//! the shim can stay a trivial switch statement.
//!
//! Phase 1 scope: only `OPUS_RESET_STATE` is routed; everything else lands
//! on the `default: OPUS_UNIMPLEMENTED` arm in the shim and never reaches
//! Rust. Later phases add setters/getters one CTL at a time.

use std::os::raw::c_int;

use ropus::opus::decoder::OpusDecoder;
use ropus::opus::encoder::OpusEncoder;

use crate::{OPUS_BAD_ARG, OPUS_INTERNAL_ERROR, OPUS_OK, ffi_guard};

// ---------------------------------------------------------------------------
// Encoder CTLs
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_encoder_ctl_reset(st: *mut OpusEncoder) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if st.is_null() {
            return OPUS_BAD_ARG;
        }
        unsafe { (*st).reset() };
        OPUS_OK
    })
}

// ---------------------------------------------------------------------------
// Decoder CTLs
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_decoder_ctl_reset(st: *mut OpusDecoder) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if st.is_null() {
            return OPUS_BAD_ARG;
        }
        unsafe { (*st).reset() };
        OPUS_OK
    })
}
