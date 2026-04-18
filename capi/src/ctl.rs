//! Typed CTL dispatchers called by `ctl_shim.c`.
//!
//! The C varargs `opus_*_ctl(st, request, ...)` surface is unpacked in
//! `ctl_shim.c`, which then calls one of these typed entry points with a
//! fully-typed scalar argument. Each function does its own panic-guarding so
//! the shim can stay a trivial switch statement.
//!
//! Phase 2 scope: decoder `OPUS_GET_FINAL_RANGE`, `OPUS_GET_LAST_PACKET_DURATION`,
//! plus `OPUS_RESET_STATE` carried over from phase 1. Unknown CTLs still fall
//! through the default arm in `ctl_shim.c` and return `OPUS_UNIMPLEMENTED`,
//! which keeps drift visible in test failures.

use std::os::raw::c_int;

use ropus::opus::decoder::OpusDecoder;
use ropus::opus::encoder::OpusEncoder;

use crate::{OPUS_BAD_ARG, OPUS_INTERNAL_ERROR, OPUS_OK, OPUS_UNIMPLEMENTED, ffi_guard};

// CTL request codes we route inside Rust (mirror `opus_defines.h`).
const OPUS_GET_BANDWIDTH_REQUEST: c_int = 4009;
const OPUS_GET_SAMPLE_RATE_REQUEST: c_int = 4029;
const OPUS_GET_FINAL_RANGE_REQUEST: c_int = 4031;
const OPUS_GET_PITCH_REQUEST: c_int = 4033;
const OPUS_SET_GAIN_REQUEST: c_int = 4034;
const OPUS_GET_GAIN_REQUEST: c_int = 4045; // intentional — see opus_defines.h
const OPUS_GET_LAST_PACKET_DURATION_REQUEST: c_int = 4039;
const OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST: c_int = 4047;
const OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST: c_int = 4046;

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
        let Some(dec) = (unsafe { crate::decoder::handle_to_decoder(st) }) else {
            return OPUS_BAD_ARG;
        };
        dec.reset();
        OPUS_OK
    })
}

/// Decoder set-int CTL dispatcher.  Dispatches by request code; unknown
/// codes return `OPUS_UNIMPLEMENTED` (same contract as the shim default arm).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_decoder_ctl_set_int(
    st: *mut OpusDecoder,
    request: c_int,
    value: c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        let Some(dec) = (unsafe { crate::decoder::handle_to_decoder(st) }) else {
            return OPUS_BAD_ARG;
        };
        match request {
            OPUS_SET_GAIN_REQUEST => match dec.set_gain(value) {
                Ok(()) => OPUS_OK,
                Err(e) => e,
            },
            OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST => {
                if !(0..=1).contains(&value) {
                    return OPUS_BAD_ARG;
                }
                dec.set_phase_inversion_disabled(value != 0);
                OPUS_OK
            }
            _ => OPUS_UNIMPLEMENTED,
        }
    })
}

/// Decoder get-int CTL dispatcher. Writes the result through `out`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_decoder_ctl_get_int(
    st: *mut OpusDecoder,
    request: c_int,
    out: *mut c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if out.is_null() {
            return OPUS_BAD_ARG;
        }
        let Some(dec) = (unsafe { crate::decoder::handle_to_decoder_ref(st) }) else {
            return OPUS_BAD_ARG;
        };
        let value: c_int = match request {
            OPUS_GET_BANDWIDTH_REQUEST => dec.get_bandwidth(),
            OPUS_GET_SAMPLE_RATE_REQUEST => dec.get_sample_rate(),
            OPUS_GET_PITCH_REQUEST => dec.get_pitch(),
            OPUS_GET_GAIN_REQUEST => dec.get_gain(),
            OPUS_GET_LAST_PACKET_DURATION_REQUEST => dec.get_last_packet_duration(),
            OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST => {
                if dec.get_phase_inversion_disabled() { 1 } else { 0 }
            }
            _ => return OPUS_UNIMPLEMENTED,
        };
        unsafe { *out = value };
        OPUS_OK
    })
}

/// Decoder get-uint32 CTL dispatcher (only `OPUS_GET_FINAL_RANGE` for now).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_decoder_ctl_get_uint32(
    st: *mut OpusDecoder,
    request: c_int,
    out: *mut u32,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if out.is_null() {
            return OPUS_BAD_ARG;
        }
        let Some(dec) = (unsafe { crate::decoder::handle_to_decoder_ref(st) }) else {
            return OPUS_BAD_ARG;
        };
        match request {
            OPUS_GET_FINAL_RANGE_REQUEST => {
                unsafe { *out = dec.get_final_range() };
                OPUS_OK
            }
            _ => OPUS_UNIMPLEMENTED,
        }
    })
}
