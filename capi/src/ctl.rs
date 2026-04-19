//! Typed CTL dispatchers called by `ctl_shim.c`.
//!
//! The C varargs `opus_*_ctl(st, request, ...)` surface is unpacked in
//! `ctl_shim.c`, which then calls one of these typed entry points with a
//! fully-typed scalar argument. Each function does its own panic-guarding so
//! the shim can stay a trivial switch statement.
//!
//! Unknown CTLs fall through the default arm in `ctl_shim.c` and return
//! `OPUS_UNIMPLEMENTED`, which keeps drift visible in test failures.

use std::os::raw::{c_int, c_uchar};

use ropus::opus::decoder::OpusDecoder;
use ropus::opus::encoder::OpusEncoder;
use ropus::opus::multistream::{
    OpusMSDecoder, OpusMSEncoder, OpusProjectionDecoder, OpusProjectionEncoder,
};

use crate::{OPUS_BAD_ARG, OPUS_INTERNAL_ERROR, OPUS_OK, OPUS_UNIMPLEMENTED, ffi_guard};

// CTL request codes we route inside Rust (mirror `opus_defines.h`).
const OPUS_SET_APPLICATION_REQUEST: c_int = 4000;
const OPUS_GET_APPLICATION_REQUEST: c_int = 4001;
const OPUS_SET_BITRATE_REQUEST: c_int = 4002;
const OPUS_GET_BITRATE_REQUEST: c_int = 4003;
const OPUS_SET_MAX_BANDWIDTH_REQUEST: c_int = 4004;
const OPUS_GET_MAX_BANDWIDTH_REQUEST: c_int = 4005;
const OPUS_SET_VBR_REQUEST: c_int = 4006;
const OPUS_GET_VBR_REQUEST: c_int = 4007;
const OPUS_SET_BANDWIDTH_REQUEST: c_int = 4008;
const OPUS_GET_BANDWIDTH_REQUEST: c_int = 4009;
const OPUS_SET_COMPLEXITY_REQUEST: c_int = 4010;
const OPUS_GET_COMPLEXITY_REQUEST: c_int = 4011;
const OPUS_SET_INBAND_FEC_REQUEST: c_int = 4012;
const OPUS_GET_INBAND_FEC_REQUEST: c_int = 4013;
const OPUS_SET_PACKET_LOSS_PERC_REQUEST: c_int = 4014;
const OPUS_GET_PACKET_LOSS_PERC_REQUEST: c_int = 4015;
const OPUS_SET_DTX_REQUEST: c_int = 4016;
const OPUS_GET_DTX_REQUEST: c_int = 4017;
const OPUS_SET_VBR_CONSTRAINT_REQUEST: c_int = 4020;
const OPUS_GET_VBR_CONSTRAINT_REQUEST: c_int = 4021;
const OPUS_SET_FORCE_CHANNELS_REQUEST: c_int = 4022;
const OPUS_GET_FORCE_CHANNELS_REQUEST: c_int = 4023;
const OPUS_SET_SIGNAL_REQUEST: c_int = 4024;
const OPUS_GET_SIGNAL_REQUEST: c_int = 4025;
const OPUS_GET_LOOKAHEAD_REQUEST: c_int = 4027;
const OPUS_GET_SAMPLE_RATE_REQUEST: c_int = 4029;
const OPUS_GET_FINAL_RANGE_REQUEST: c_int = 4031;
const OPUS_GET_PITCH_REQUEST: c_int = 4033;
const OPUS_SET_GAIN_REQUEST: c_int = 4034;
const OPUS_GET_GAIN_REQUEST: c_int = 4045; // intentional — see opus_defines.h
const OPUS_SET_LSB_DEPTH_REQUEST: c_int = 4036;
const OPUS_GET_LSB_DEPTH_REQUEST: c_int = 4037;
const OPUS_GET_LAST_PACKET_DURATION_REQUEST: c_int = 4039;
const OPUS_SET_EXPERT_FRAME_DURATION_REQUEST: c_int = 4040;
const OPUS_GET_EXPERT_FRAME_DURATION_REQUEST: c_int = 4041;
const OPUS_SET_PREDICTION_DISABLED_REQUEST: c_int = 4042;
const OPUS_GET_PREDICTION_DISABLED_REQUEST: c_int = 4043;
const OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST: c_int = 4046;
const OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST: c_int = 4047;
const OPUS_GET_IN_DTX_REQUEST: c_int = 4049;
const OPUS_SET_IGNORE_EXTENSIONS_REQUEST: c_int = 4058;
const OPUS_GET_IGNORE_EXTENSIONS_REQUEST: c_int = 4059;

// DNN weight loading CTL (xiph: `reference/include/opus_defines.h:174`).
// Public in the xiph header. Writes-only (no paired GET — xiph's 4053 is
// commented out at `opus_defines.h:175`). Dispatched through a dedicated
// pointer-argument entry point (`mdopus_decoder_ctl_set_dnn_blob`) from
// `ctl_shim.c` because the CTL takes `const unsigned char *, opus_int32`
// rather than a scalar; this constant is for documentation symmetry
// (the shim has its own `#define` of the same value).
#[allow(dead_code)]
const OPUS_SET_DNN_BLOB_REQUEST: c_int = 4052;

// From `reference/src/opus_private.h:172` — private CTL the conformance
// test_opus_encode.c exercises directly. Routed into `OpusEncoder::set_force_mode`
// (validation lives there; maps `OPUS_AUTO | MODE_SILK_ONLY | MODE_HYBRID | MODE_CELT_ONLY`).
const OPUS_SET_FORCE_MODE_REQUEST: c_int = 11002;

// From `reference/src/opus_private.h` — semi-obsolete voice/music ratio
// hint (`-1..=100`, `-1` = auto). Just stored, no behavioural coupling.
const OPUS_SET_VOICE_RATIO_REQUEST: c_int = 11018;
const OPUS_GET_VOICE_RATIO_REQUEST: c_int = 11019;

// From `reference/celt/celt.h` — multistream-internal CTLs forwarded into
// CELT. `OPUS_SET_LFE` flags a channel as low-frequency-effects (int value);
// `OPUS_SET_ENERGY_MASK` passes a per-band mask as a pointer (dispatched
// directly from `ctl_shim.c` via `mdopus_encoder_ctl_set_energy_mask`, not
// via `mdopus_encoder_ctl_set_int`).
const OPUS_SET_LFE_REQUEST: c_int = 10024;
#[allow(dead_code)]
const OPUS_SET_ENERGY_MASK_REQUEST: c_int = 10026;

// From `reference/include/opus_projection.h` — projection-specific CTLs
// exposed on the projection encoder (not decoder: the test and reference
// source confirm these live on `opus_projection_encoder_ctl`). The `_REQUEST`
// constant for OPUS_PROJECTION_GET_DEMIXING_MATRIX is only referenced from
// `ctl_shim.c`; the Rust side dispatches it through a dedicated pointer-argument
// entry point (`mdopus_proj_encoder_ctl_get_demixing_matrix`) rather than the
// int-out getter below.
const OPUS_PROJECTION_GET_DEMIXING_MATRIX_GAIN_REQUEST: c_int = 6001;
const OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST: c_int = 6003;
#[allow(dead_code)]
const OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST: c_int = 6005;

const OPUS_AUTO: c_int = -1000;

// OPUS_APPLICATION_* constants (mirror `opus_defines.h`). Used by the MS
// encoder dispatcher to validate up-front before delegating to ropus.
const OPUS_APPLICATION_VOIP: c_int = 2048;
const OPUS_APPLICATION_AUDIO: c_int = 2049;
const OPUS_APPLICATION_RESTRICTED_LOWDELAY: c_int = 2051;

// ---------------------------------------------------------------------------
// Encoder CTLs
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_encoder_ctl_reset(st: *mut OpusEncoder) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        let Some(enc) = (unsafe { crate::encoder::handle_to_encoder(st) }) else {
            return OPUS_BAD_ARG;
        };
        enc.reset();
        unsafe { crate::encoder::bump_generation(st) };
        OPUS_OK
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_encoder_ctl_set_int(
    st: *mut OpusEncoder,
    request: c_int,
    value: c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        let Some(enc) = (unsafe { crate::encoder::handle_to_encoder(st) }) else {
            return OPUS_BAD_ARG;
        };
        match request {
            OPUS_SET_APPLICATION_REQUEST => enc.set_application(value),
            OPUS_SET_BITRATE_REQUEST => enc.set_bitrate(value),
            OPUS_SET_MAX_BANDWIDTH_REQUEST => enc.set_max_bandwidth(value),
            OPUS_SET_VBR_REQUEST => enc.set_vbr(value),
            OPUS_SET_BANDWIDTH_REQUEST => enc.set_bandwidth(value),
            OPUS_SET_COMPLEXITY_REQUEST => enc.set_complexity(value),
            OPUS_SET_INBAND_FEC_REQUEST => enc.set_inband_fec(value),
            OPUS_SET_PACKET_LOSS_PERC_REQUEST => enc.set_packet_loss_perc(value),
            OPUS_SET_DTX_REQUEST => enc.set_dtx(value),
            OPUS_SET_VBR_CONSTRAINT_REQUEST => enc.set_vbr_constraint(value),
            OPUS_SET_FORCE_CHANNELS_REQUEST => enc.set_force_channels(value),
            OPUS_SET_SIGNAL_REQUEST => enc.set_signal(value),
            OPUS_SET_LSB_DEPTH_REQUEST => enc.set_lsb_depth(value),
            OPUS_SET_EXPERT_FRAME_DURATION_REQUEST => enc.set_expert_frame_duration(value),
            OPUS_SET_PREDICTION_DISABLED_REQUEST => enc.set_prediction_disabled(value),
            OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST => {
                if !(0..=1).contains(&value) {
                    return OPUS_BAD_ARG;
                }
                enc.set_phase_inversion_disabled(value)
            }
            // Validation lives inside `set_voice_ratio` (rejects outside
            // `-1..=100`, matching `opus_encoder.c`).
            OPUS_SET_VOICE_RATIO_REQUEST => enc.set_voice_ratio(value),
            // Multistream-internal: flag a channel as LFE. `set_lfe` stores
            // the flag and forwards to CELT unless the app is RESTRICTED_SILK.
            OPUS_SET_LFE_REQUEST => enc.set_lfe(value),
            // Private CTL from `opus_private.h` — validation happens inside
            // ropus's `set_force_mode` (returns `OPUS_BAD_ARG` for anything
            // outside `OPUS_AUTO | MODE_SILK_ONLY | MODE_HYBRID | MODE_CELT_ONLY`).
            OPUS_SET_FORCE_MODE_REQUEST => enc.set_force_mode(value),
            _ => OPUS_UNIMPLEMENTED,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_encoder_ctl_get_int(
    st: *mut OpusEncoder,
    request: c_int,
    out: *mut c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if out.is_null() {
            return OPUS_BAD_ARG;
        }
        let Some(enc) = (unsafe { crate::encoder::handle_to_encoder_ref(st) }) else {
            return OPUS_BAD_ARG;
        };
        let value: c_int = match request {
            OPUS_GET_APPLICATION_REQUEST => enc.get_application(),
            OPUS_GET_BITRATE_REQUEST => enc.get_bitrate(),
            OPUS_GET_MAX_BANDWIDTH_REQUEST => enc.get_max_bandwidth(),
            OPUS_GET_VBR_REQUEST => enc.get_vbr(),
            OPUS_GET_BANDWIDTH_REQUEST => enc.get_bandwidth(),
            OPUS_GET_COMPLEXITY_REQUEST => enc.get_complexity(),
            OPUS_GET_INBAND_FEC_REQUEST => enc.get_inband_fec(),
            OPUS_GET_PACKET_LOSS_PERC_REQUEST => enc.get_packet_loss_perc(),
            OPUS_GET_DTX_REQUEST => enc.get_dtx(),
            OPUS_GET_VBR_CONSTRAINT_REQUEST => enc.get_vbr_constraint(),
            OPUS_GET_FORCE_CHANNELS_REQUEST => enc.get_force_channels(),
            OPUS_GET_SIGNAL_REQUEST => enc.get_signal(),
            OPUS_GET_LOOKAHEAD_REQUEST => enc.get_lookahead(),
            OPUS_GET_SAMPLE_RATE_REQUEST => enc.get_sample_rate(),
            OPUS_GET_LSB_DEPTH_REQUEST => enc.get_lsb_depth(),
            OPUS_GET_EXPERT_FRAME_DURATION_REQUEST => enc.get_expert_frame_duration(),
            OPUS_GET_PREDICTION_DISABLED_REQUEST => enc.get_prediction_disabled(),
            OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST => enc.get_phase_inversion_disabled(),
            OPUS_GET_VOICE_RATIO_REQUEST => enc.get_voice_ratio(),
            OPUS_GET_IN_DTX_REQUEST => enc.get_in_dtx(),
            _ => return OPUS_UNIMPLEMENTED,
        };
        unsafe { *out = value };
        OPUS_OK
    })
}

/// Typed pointer-argument dispatcher for encoder CTLs.
///
/// Currently handles `OPUS_SET_ENERGY_MASK` only (a `celt_glog *` / `int *`
/// pointer in the reference ABI — multistream-internal, passes a per-band
/// energy mask through to the CELT encoder). The reference C API does not
/// encode a length: callers in `opus_multistream_encoder.c` always size the
/// buffer as `21 * channels` (L993/L1008). We replicate that here by reading
/// the channel count from the encoder state — the single source of truth.
/// A null `ptr` clears the mask (matches `st->energy_masking = NULL`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_encoder_ctl_set_energy_mask(
    st: *mut OpusEncoder,
    ptr: *const c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        let Some(enc) = (unsafe { crate::encoder::handle_to_encoder(st) }) else {
            return OPUS_BAD_ARG;
        };
        if ptr.is_null() {
            return enc.set_energy_mask(None);
        }
        let len = (21 * enc.channels) as usize;
        let slice: &[c_int] = unsafe { std::slice::from_raw_parts(ptr, len) };
        enc.set_energy_mask(Some(slice))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_encoder_ctl_get_uint32(
    st: *mut OpusEncoder,
    request: c_int,
    out: *mut u32,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if out.is_null() {
            return OPUS_BAD_ARG;
        }
        let Some(enc) = (unsafe { crate::encoder::handle_to_encoder_ref(st) }) else {
            return OPUS_BAD_ARG;
        };
        match request {
            OPUS_GET_FINAL_RANGE_REQUEST => {
                unsafe { *out = enc.get_final_range() };
                OPUS_OK
            }
            _ => OPUS_UNIMPLEMENTED,
        }
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
        unsafe { crate::decoder::bump_generation(st) };
        OPUS_OK
    })
}

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
            OPUS_SET_IGNORE_EXTENSIONS_REQUEST => {
                // Mirrors `opus_decoder.c`: value must be 0 or 1.
                if !(0..=1).contains(&value) {
                    return OPUS_BAD_ARG;
                }
                dec.set_ignore_extensions(value != 0);
                OPUS_OK
            }
            // Decoder-side complexity: validation (`0..=10`) lives inside
            // `OpusDecoder::set_complexity`; it forwards the value into the
            // embedded CELT decoder as well, matching the encoder side's
            // CTL semantics.
            OPUS_SET_COMPLEXITY_REQUEST => match dec.set_complexity(value) {
                Ok(()) => OPUS_OK,
                Err(e) => e,
            },
            _ => OPUS_UNIMPLEMENTED,
        }
    })
}

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
                if dec.get_phase_inversion_disabled() {
                    1
                } else {
                    0
                }
            }
            OPUS_GET_IGNORE_EXTENSIONS_REQUEST => {
                if dec.get_ignore_extensions() {
                    1
                } else {
                    0
                }
            }
            OPUS_GET_COMPLEXITY_REQUEST => dec.get_complexity(),
            _ => return OPUS_UNIMPLEMENTED,
        };
        unsafe { *out = value };
        OPUS_OK
    })
}

/// Typed pointer-argument dispatcher for decoder CTLs. Currently handles
/// `OPUS_SET_DNN_BLOB_REQUEST(const unsigned char *data, opus_int32 len)`
/// — the reference validates `len >= 0 && data != NULL` and then delegates
/// to `lpcnet_plc_load_model` + `silk_LoadOSCEModels`
/// (`reference/src/opus_decoder.c:1218`). We fold both checks into a
/// `from_raw_parts` of the slice and forward to `OpusDecoder::set_dnn_blob`,
/// which parses the weight blob and hands each named section off to
/// `LPCNetPLCState::load_model` (ported in Stage 7b.1.5).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_decoder_ctl_set_dnn_blob(
    st: *mut OpusDecoder,
    data: *const c_uchar,
    len: c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if data.is_null() || len < 0 {
            return OPUS_BAD_ARG;
        }
        let Some(dec) = (unsafe { crate::decoder::handle_to_decoder(st) }) else {
            return OPUS_BAD_ARG;
        };
        // SAFETY: `data` is a caller-provided buffer of at least `len`
        // bytes (the xiph CTL contract). Zero-length is permitted — the
        // raw pointer must be non-null per the check above.
        let slice: &[u8] = unsafe { std::slice::from_raw_parts(data, len as usize) };
        match dec.set_dnn_blob(slice) {
            Ok(()) => OPUS_OK,
            Err(e) => e,
        }
    })
}

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

// ---------------------------------------------------------------------------
// Multistream encoder CTLs
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_ms_encoder_ctl_reset(st: *mut OpusMSEncoder) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        let Some(ms) = (unsafe { crate::ms_encoder::handle_to_ms_encoder(st) }) else {
            return OPUS_BAD_ARG;
        };
        // `OpusMSEncoder::reset` already iterates sub-encoders (and clears
        // surround-mode preemph/window buffers); doing it again here would be
        // redundant and drop the surround-specific cleanup.
        ms.reset();
        unsafe { crate::ms_encoder::bump_generation(st) };
        OPUS_OK
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_ms_encoder_ctl_set_int(
    st: *mut OpusMSEncoder,
    request: c_int,
    value: c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        let Some(ms) = (unsafe { crate::ms_encoder::handle_to_ms_encoder(st) }) else {
            return OPUS_BAD_ARG;
        };
        match request {
            OPUS_SET_APPLICATION_REQUEST => {
                // Validate up-front so an invalid value never mutates any
                // sub-encoder; ropus `ms.set_application` fans out
                // atomically (and `ms_set_application` silently ignores
                // invalid values, so a bad value would otherwise leak into
                // `ms.application` without taking effect per stream).
                if value != OPUS_APPLICATION_VOIP
                    && value != OPUS_APPLICATION_AUDIO
                    && value != OPUS_APPLICATION_RESTRICTED_LOWDELAY
                {
                    return OPUS_BAD_ARG;
                }
                ms.set_application(value)
            }
            OPUS_SET_BITRATE_REQUEST => ms.set_bitrate(value),
            OPUS_SET_MAX_BANDWIDTH_REQUEST => ms.set_max_bandwidth(value),
            OPUS_SET_VBR_REQUEST => ms.set_vbr(value),
            OPUS_SET_BANDWIDTH_REQUEST => ms.set_bandwidth(value),
            OPUS_SET_COMPLEXITY_REQUEST => ms.set_complexity(value),
            OPUS_SET_INBAND_FEC_REQUEST => ms.set_inband_fec(value),
            OPUS_SET_PACKET_LOSS_PERC_REQUEST => ms.set_packet_loss_perc(value),
            OPUS_SET_DTX_REQUEST => ms.set_dtx(value),
            OPUS_SET_VBR_CONSTRAINT_REQUEST => ms.set_vbr_constraint(value),
            OPUS_SET_FORCE_CHANNELS_REQUEST => {
                // Validate up-front before delegating; ropus
                // `ms.set_force_channels` does no validation (it just
                // stores the value into each sub-encoder), so without
                // this guard an out-of-range value would partially
                // apply. Valid values: OPUS_AUTO or 1..=2.
                if value != OPUS_AUTO && !(1..=2).contains(&value) {
                    return OPUS_BAD_ARG;
                }
                ms.set_force_channels(value)
            }
            OPUS_SET_SIGNAL_REQUEST => ms.set_signal(value),
            OPUS_SET_LSB_DEPTH_REQUEST => ms.set_lsb_depth(value),
            OPUS_SET_EXPERT_FRAME_DURATION_REQUEST => {
                ms.set_expert_frame_duration(value);
                OPUS_OK
            }
            OPUS_SET_PREDICTION_DISABLED_REQUEST => ms.set_prediction_disabled(value),
            OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST => ms.set_phase_inversion_disabled(value),
            // `OpusMSEncoder::set_force_mode` fans out to every sub-encoder's
            // `ms_set_force_mode` (which skips validation — it only stores the
            // value). Validate up-front here so an invalid mode can't silently
            // stick; mirror the validation in `OpusEncoder::set_force_mode`.
            OPUS_SET_FORCE_MODE_REQUEST => {
                const MODE_SILK_ONLY: c_int = 1000;
                const MODE_CELT_ONLY: c_int = 1002;
                if value != OPUS_AUTO && !(MODE_SILK_ONLY..=MODE_CELT_ONLY).contains(&value) {
                    return OPUS_BAD_ARG;
                }
                ms.set_force_mode(value)
            }
            // Multistream-internal: LFE is routed to the designated LFE
            // sub-encoder only (matches `opus_multistream_encoder.c`
            // init-time pattern L498-507). Returns `OPUS_BAD_ARG` when no
            // LFE stream exists (mapping family != 1 or channels < 6).
            OPUS_SET_LFE_REQUEST => ms.set_lfe(value),
            _ => OPUS_UNIMPLEMENTED,
        }
    })
}

/// Typed pointer-argument dispatcher for MS encoder CTLs. Currently handles
/// `OPUS_SET_ENERGY_MASK` (sized `21 * layout.nb_channels`, fanned out per
/// sub-encoder via layout-aware slicing — matches encode-time behaviour at
/// `opus_multistream_encoder.c` L989-1014). A null `ptr` clears the mask
/// on every sub-encoder.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_ms_encoder_ctl_set_energy_mask(
    st: *mut OpusMSEncoder,
    ptr: *const c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        let Some(ms) = (unsafe { crate::ms_encoder::handle_to_ms_encoder(st) }) else {
            return OPUS_BAD_ARG;
        };
        if ptr.is_null() {
            return ms.set_energy_mask(None);
        }
        // `OpusMSEncoder::set_energy_mask` validates slice length against
        // `21 * nb_channels`. Size the slice accordingly here.
        let len = (21 * ms.nb_channels()) as usize;
        let slice: &[c_int] = unsafe { std::slice::from_raw_parts(ptr, len) };
        ms.set_energy_mask(Some(slice))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_ms_encoder_ctl_get_int(
    st: *mut OpusMSEncoder,
    request: c_int,
    out: *mut c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if out.is_null() {
            return OPUS_BAD_ARG;
        }
        let Some(ms) = (unsafe { crate::ms_encoder::handle_to_ms_encoder_ref(st) }) else {
            return OPUS_BAD_ARG;
        };
        let value: c_int = match request {
            OPUS_GET_APPLICATION_REQUEST => ms.get_application(),
            OPUS_GET_BITRATE_REQUEST => ms.get_bitrate(),
            OPUS_GET_SAMPLE_RATE_REQUEST => ms.get_sample_rate(),
            OPUS_GET_LOOKAHEAD_REQUEST => ms.get_lookahead(),
            OPUS_GET_EXPERT_FRAME_DURATION_REQUEST => ms.get_expert_frame_duration(),
            OPUS_GET_COMPLEXITY_REQUEST => ms
                .get_encoder(0)
                .map(|e| e.get_complexity())
                .unwrap_or(OPUS_AUTO),
            OPUS_GET_VBR_REQUEST => ms.get_encoder(0).map(|e| e.get_vbr()).unwrap_or(0),
            OPUS_GET_VBR_CONSTRAINT_REQUEST => ms
                .get_encoder(0)
                .map(|e| e.get_vbr_constraint())
                .unwrap_or(0),
            OPUS_GET_BANDWIDTH_REQUEST => ms.get_encoder(0).map(|e| e.get_bandwidth()).unwrap_or(0),
            OPUS_GET_MAX_BANDWIDTH_REQUEST => ms
                .get_encoder(0)
                .map(|e| e.get_max_bandwidth())
                .unwrap_or(0),
            OPUS_GET_INBAND_FEC_REQUEST => {
                ms.get_encoder(0).map(|e| e.get_inband_fec()).unwrap_or(0)
            }
            OPUS_GET_PACKET_LOSS_PERC_REQUEST => ms
                .get_encoder(0)
                .map(|e| e.get_packet_loss_perc())
                .unwrap_or(0),
            OPUS_GET_DTX_REQUEST => ms.get_encoder(0).map(|e| e.get_dtx()).unwrap_or(0),
            OPUS_GET_FORCE_CHANNELS_REQUEST => ms
                .get_encoder(0)
                .map(|e| e.get_force_channels())
                .unwrap_or(OPUS_AUTO),
            OPUS_GET_SIGNAL_REQUEST => ms.get_encoder(0).map(|e| e.get_signal()).unwrap_or(0),
            OPUS_GET_LSB_DEPTH_REQUEST => ms.get_encoder(0).map(|e| e.get_lsb_depth()).unwrap_or(0),
            OPUS_GET_PREDICTION_DISABLED_REQUEST => ms
                .get_encoder(0)
                .map(|e| e.get_prediction_disabled())
                .unwrap_or(0),
            OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST => ms
                .get_encoder(0)
                .map(|e| e.get_phase_inversion_disabled())
                .unwrap_or(0),
            _ => return OPUS_UNIMPLEMENTED,
        };
        unsafe { *out = value };
        OPUS_OK
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_ms_encoder_ctl_get_uint32(
    st: *mut OpusMSEncoder,
    request: c_int,
    out: *mut u32,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if out.is_null() {
            return OPUS_BAD_ARG;
        }
        let Some(ms) = (unsafe { crate::ms_encoder::handle_to_ms_encoder_ref(st) }) else {
            return OPUS_BAD_ARG;
        };
        match request {
            OPUS_GET_FINAL_RANGE_REQUEST => {
                unsafe { *out = ms.get_final_range() };
                OPUS_OK
            }
            _ => OPUS_UNIMPLEMENTED,
        }
    })
}

/// `OPUS_MULTISTREAM_GET_ENCODER_STATE(stream_id, OpusEncoder **state)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_ms_encoder_ctl_get_encoder_state(
    st: *mut OpusMSEncoder,
    stream_id: c_int,
    out: *mut *mut OpusEncoder,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if out.is_null() {
            return OPUS_BAD_ARG;
        }
        // Route through a single `&mut Inner` path — `sub_encoder_handle_ptr`
        // validates the handle, `stream_id`, and the slot internally. Do NOT
        // also call `handle_to_ms_encoder_ref` here: the resulting `&` would
        // alias the `&mut Inner` synthesised inside the helper (noalias UB).
        let h = unsafe { crate::ms_encoder::sub_encoder_handle_ptr(st, stream_id) };
        if h.is_null() {
            return OPUS_BAD_ARG;
        }
        unsafe { *out = h };
        OPUS_OK
    })
}

// ---------------------------------------------------------------------------
// Multistream decoder CTLs
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_ms_decoder_ctl_reset(st: *mut OpusMSDecoder) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        let Some(ms) = (unsafe { crate::ms_decoder::handle_to_ms_decoder(st) }) else {
            return OPUS_BAD_ARG;
        };
        ms.reset();
        unsafe { crate::ms_decoder::bump_generation(st) };
        OPUS_OK
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_ms_decoder_ctl_set_int(
    st: *mut OpusMSDecoder,
    request: c_int,
    value: c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        let Some(ms) = (unsafe { crate::ms_decoder::handle_to_ms_decoder(st) }) else {
            return OPUS_BAD_ARG;
        };
        match request {
            OPUS_SET_GAIN_REQUEST => {
                if !(-32768..=32767).contains(&value) {
                    return OPUS_BAD_ARG;
                }
                ms.set_gain(value)
            }
            OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST => {
                if !(0..=1).contains(&value) {
                    return OPUS_BAD_ARG;
                }
                ms.set_phase_inversion_disabled(value)
            }
            _ => OPUS_UNIMPLEMENTED,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_ms_decoder_ctl_get_int(
    st: *mut OpusMSDecoder,
    request: c_int,
    out: *mut c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if out.is_null() {
            return OPUS_BAD_ARG;
        }
        let Some(ms) = (unsafe { crate::ms_decoder::handle_to_ms_decoder_ref(st) }) else {
            return OPUS_BAD_ARG;
        };
        let value: c_int = match request {
            OPUS_GET_SAMPLE_RATE_REQUEST => ms.get_sample_rate(),
            OPUS_GET_BANDWIDTH_REQUEST => ms.get_bandwidth(),
            OPUS_GET_LAST_PACKET_DURATION_REQUEST => ms.get_last_packet_duration(),
            OPUS_GET_GAIN_REQUEST => ms.get_gain(),
            OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST => ms.get_phase_inversion_disabled(),
            _ => return OPUS_UNIMPLEMENTED,
        };
        unsafe { *out = value };
        OPUS_OK
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_ms_decoder_ctl_get_uint32(
    st: *mut OpusMSDecoder,
    request: c_int,
    out: *mut u32,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if out.is_null() {
            return OPUS_BAD_ARG;
        }
        let Some(ms) = (unsafe { crate::ms_decoder::handle_to_ms_decoder_ref(st) }) else {
            return OPUS_BAD_ARG;
        };
        match request {
            OPUS_GET_FINAL_RANGE_REQUEST => {
                unsafe { *out = ms.get_final_range() };
                OPUS_OK
            }
            _ => OPUS_UNIMPLEMENTED,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_ms_decoder_ctl_get_decoder_state(
    st: *mut OpusMSDecoder,
    stream_id: c_int,
    out: *mut *mut OpusDecoder,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if out.is_null() {
            return OPUS_BAD_ARG;
        }
        // Single `&mut Inner` path — see the encoder counterpart for the
        // aliasing rationale.
        let h = unsafe { crate::ms_decoder::sub_decoder_handle_ptr(st, stream_id) };
        if h.is_null() {
            return OPUS_BAD_ARG;
        }
        unsafe { *out = h };
        OPUS_OK
    })
}

// ---------------------------------------------------------------------------
// Projection encoder CTLs
// ---------------------------------------------------------------------------
//
// Surface deliberately minimal: the conformance test exercises
// `OPUS_SET_BITRATE` plus the three `OPUS_PROJECTION_GET_DEMIXING_MATRIX*`
// requests. Other encoder CTLs that the generic conformance work already
// validated on `OpusMSEncoder` fall through to `OPUS_UNIMPLEMENTED` here —
// the projection encoder state is a thin wrapper, and nothing in the
// Piece B scope touches it.

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_proj_encoder_ctl_reset(st: *mut OpusProjectionEncoder) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        let Some(enc) = (unsafe { crate::projection::handle_to_proj_encoder(st) }) else {
            return OPUS_BAD_ARG;
        };
        enc.reset();
        unsafe { crate::projection::bump_enc_generation(st) };
        OPUS_OK
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_proj_encoder_ctl_set_int(
    st: *mut OpusProjectionEncoder,
    request: c_int,
    value: c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        let Some(enc) = (unsafe { crate::projection::handle_to_proj_encoder(st) }) else {
            return OPUS_BAD_ARG;
        };
        match request {
            OPUS_SET_BITRATE_REQUEST => enc.set_bitrate(value),
            OPUS_SET_COMPLEXITY_REQUEST => enc.set_complexity(value),
            OPUS_SET_VBR_REQUEST => enc.set_vbr(value),
            _ => OPUS_UNIMPLEMENTED,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_proj_encoder_ctl_get_int(
    st: *mut OpusProjectionEncoder,
    request: c_int,
    out: *mut c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if out.is_null() {
            return OPUS_BAD_ARG;
        }
        let Some(enc) = (unsafe { crate::projection::handle_to_proj_encoder_ref(st) }) else {
            return OPUS_BAD_ARG;
        };
        let value: c_int = match request {
            OPUS_GET_BITRATE_REQUEST => enc.get_bitrate(),
            OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST => enc.get_demixing_matrix_size(),
            OPUS_PROJECTION_GET_DEMIXING_MATRIX_GAIN_REQUEST => enc.get_demixing_matrix_gain(),
            _ => return OPUS_UNIMPLEMENTED,
        };
        unsafe { *out = value };
        OPUS_OK
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_proj_encoder_ctl_get_uint32(
    st: *mut OpusProjectionEncoder,
    request: c_int,
    out: *mut u32,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if out.is_null() {
            return OPUS_BAD_ARG;
        }
        let Some(enc) = (unsafe { crate::projection::handle_to_proj_encoder_ref(st) }) else {
            return OPUS_BAD_ARG;
        };
        match request {
            OPUS_GET_FINAL_RANGE_REQUEST => {
                unsafe { *out = enc.get_final_range() };
                OPUS_OK
            }
            _ => OPUS_UNIMPLEMENTED,
        }
    })
}

/// `OPUS_PROJECTION_GET_DEMIXING_MATRIX(ptr, size)` — copies the demixing
/// matrix bytes into the caller-provided buffer of exactly `size` bytes.
/// The size must equal `get_demixing_matrix_size()`; other sizes fail with
/// `OPUS_BAD_ARG`, mirroring the reference's same-size check.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_proj_encoder_ctl_get_demixing_matrix(
    st: *mut OpusProjectionEncoder,
    out: *mut c_uchar,
    size: c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if out.is_null() || size < 0 {
            return OPUS_BAD_ARG;
        }
        let Some(enc) = (unsafe { crate::projection::handle_to_proj_encoder_ref(st) }) else {
            return OPUS_BAD_ARG;
        };
        let expected = enc.get_demixing_matrix_size();
        if size != expected {
            return OPUS_BAD_ARG;
        }
        let bytes = enc.get_demixing_matrix();
        debug_assert_eq!(bytes.len(), expected as usize);
        // SAFETY: caller provided `size` bytes of writable storage.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len());
        }
        OPUS_OK
    })
}

// ---------------------------------------------------------------------------
// Projection decoder CTLs
// ---------------------------------------------------------------------------
//
// The projection test never calls `opus_projection_decoder_ctl`. We still
// expose a minimal dispatcher so `opus_projection_decoder_ctl` resolves to
// something reasonable if a future consumer exercises it. All known
// requests route through the underlying multistream decoder's behaviour.

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_proj_decoder_ctl_reset(st: *mut OpusProjectionDecoder) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        let Some(dec) = (unsafe { crate::projection::handle_to_proj_decoder(st) }) else {
            return OPUS_BAD_ARG;
        };
        dec.reset();
        unsafe { crate::projection::bump_dec_generation(st) };
        OPUS_OK
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_proj_decoder_ctl_set_int(
    st: *mut OpusProjectionDecoder,
    request: c_int,
    value: c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        let Some(dec) = (unsafe { crate::projection::handle_to_proj_decoder(st) }) else {
            return OPUS_BAD_ARG;
        };
        match request {
            OPUS_SET_GAIN_REQUEST => {
                if !(-32768..=32767).contains(&value) {
                    return OPUS_BAD_ARG;
                }
                dec.set_gain(value)
            }
            _ => OPUS_UNIMPLEMENTED,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_proj_decoder_ctl_get_int(
    st: *mut OpusProjectionDecoder,
    request: c_int,
    out: *mut c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if out.is_null() {
            return OPUS_BAD_ARG;
        }
        let Some(dec) = (unsafe { crate::projection::handle_to_proj_decoder(st) }) else {
            return OPUS_BAD_ARG;
        };
        let value: c_int = match request {
            OPUS_GET_SAMPLE_RATE_REQUEST => dec.get_sample_rate(),
            OPUS_GET_GAIN_REQUEST => dec.get_gain(),
            _ => return OPUS_UNIMPLEMENTED,
        };
        unsafe { *out = value };
        OPUS_OK
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mdopus_proj_decoder_ctl_get_uint32(
    st: *mut OpusProjectionDecoder,
    request: c_int,
    out: *mut u32,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if out.is_null() {
            return OPUS_BAD_ARG;
        }
        let Some(dec) = (unsafe { crate::projection::handle_to_proj_decoder(st) }) else {
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

// ---------------------------------------------------------------------------
// CTL-constant pinning tests
// ---------------------------------------------------------------------------
//
// The CTL request codes are part of the stable libopus ABI and are also
// hard-coded in `ctl_shim.c` as `#define`s. We cannot cross-check the
// Rust constant against the C `#define` from Rust without FFI, so we
// pin the Rust side against the canonical numeric literal here —
// catches hand-edit drift on the Rust constant. If a drift happens in
// `ctl_shim.c`, the conformance tests (which go end-to-end via the C
// shim) will surface it as an `OPUS_UNIMPLEMENTED` at runtime.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opus_set_dnn_blob_request_has_canonical_value() {
        // Matches `reference/include/opus_defines.h:174` verbatim.
        assert_eq!(OPUS_SET_DNN_BLOB_REQUEST, 4052);
    }
}
