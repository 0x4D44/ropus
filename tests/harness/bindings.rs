//! Manual FFI bindings to the C reference opus library.
//!
//! These map directly to the public opus API and selected internal functions
//! for module-level comparison testing.

#![allow(non_camel_case_types, dead_code)]

use std::os::raw::{c_int, c_uchar, c_uint};

// ---------------------------------------------------------------------------
// Opus types (matching opus_types.h)
// ---------------------------------------------------------------------------
pub type opus_int16 = i16;
pub type opus_int32 = i32;
pub type opus_uint32 = u32;

// ---------------------------------------------------------------------------
// Opus constants (matching opus_defines.h)
// ---------------------------------------------------------------------------
pub const OPUS_OK: c_int = 0;
pub const OPUS_BAD_ARG: c_int = -1;
pub const OPUS_BUFFER_TOO_SMALL: c_int = -2;
pub const OPUS_INTERNAL_ERROR: c_int = -3;
pub const OPUS_INVALID_PACKET: c_int = -4;
pub const OPUS_UNIMPLEMENTED: c_int = -5;
pub const OPUS_INVALID_STATE: c_int = -6;
pub const OPUS_ALLOC_FAIL: c_int = -7;

pub const OPUS_APPLICATION_VOIP: c_int = 2048;
pub const OPUS_APPLICATION_AUDIO: c_int = 2049;
pub const OPUS_APPLICATION_RESTRICTED_LOWDELAY: c_int = 2051;

pub const OPUS_AUTO: c_int = -1000;

// Bandwidth values
pub const OPUS_BANDWIDTH_NARROWBAND: c_int = 1101;
pub const OPUS_BANDWIDTH_MEDIUMBAND: c_int = 1102;
pub const OPUS_BANDWIDTH_WIDEBAND: c_int = 1103;
pub const OPUS_BANDWIDTH_SUPERWIDEBAND: c_int = 1104;
pub const OPUS_BANDWIDTH_FULLBAND: c_int = 1105;

// Signal types
pub const OPUS_SIGNAL_VOICE: c_int = 3001;
pub const OPUS_SIGNAL_MUSIC: c_int = 3002;

// CTL request codes
pub const OPUS_SET_BITRATE_REQUEST: c_int = 4002;
pub const OPUS_SET_MAX_BANDWIDTH_REQUEST: c_int = 4004;
pub const OPUS_SET_VBR_REQUEST: c_int = 4006;
pub const OPUS_SET_BANDWIDTH_REQUEST: c_int = 4008;
pub const OPUS_SET_COMPLEXITY_REQUEST: c_int = 4010;
pub const OPUS_SET_INBAND_FEC_REQUEST: c_int = 4012;
pub const OPUS_SET_PACKET_LOSS_PERC_REQUEST: c_int = 4014;
pub const OPUS_SET_DTX_REQUEST: c_int = 4016;
pub const OPUS_SET_VBR_CONSTRAINT_REQUEST: c_int = 4020;
pub const OPUS_SET_FORCE_CHANNELS_REQUEST: c_int = 4022;
pub const OPUS_SET_SIGNAL_REQUEST: c_int = 4024;
pub const OPUS_SET_LSB_DEPTH_REQUEST: c_int = 4036;
pub const OPUS_SET_PREDICTION_DISABLED_REQUEST: c_int = 4042;
pub const OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST: c_int = 4046;

// Getter CTL request codes
pub const OPUS_GET_FINAL_RANGE_REQUEST: c_int = 4031;
pub const OPUS_GET_BANDWIDTH_REQUEST: c_int = 4009;

// Internal CTL (from opus_private.h)
pub const OPUS_SET_FORCE_MODE_REQUEST: c_int = 11002;

// ---------------------------------------------------------------------------
// Opaque encoder/decoder handles
// ---------------------------------------------------------------------------
#[repr(C)]
pub struct OpusEncoder {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct OpusDecoder {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct OpusRepacketizer {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct OpusMSEncoder {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct OpusMSDecoder {
    _opaque: [u8; 0],
}

// ---------------------------------------------------------------------------
// Range coder context (matches struct ec_ctx in entcode.h)
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ec_ctx {
    pub buf: *mut c_uchar,
    pub storage: opus_uint32,
    pub end_offs: opus_uint32,
    pub end_window: opus_uint32, // ec_window = opus_uint32
    pub nend_bits: c_int,
    pub nbits_total: c_int,
    pub offs: opus_uint32,
    pub rng: opus_uint32,
    pub val: opus_uint32,
    pub ext: opus_uint32,
    pub rem: c_int,
    pub error: c_int,
}

pub type ec_enc = ec_ctx;
pub type ec_dec = ec_ctx;

// ---------------------------------------------------------------------------
// Public Opus API
// ---------------------------------------------------------------------------
#[link(name = "opus_ref", kind = "static")]
unsafe extern "C" {
    // Encoder
    pub fn opus_encoder_get_size(channels: c_int) -> c_int;
    pub fn opus_encoder_create(
        Fs: opus_int32,
        channels: c_int,
        application: c_int,
        error: *mut c_int,
    ) -> *mut OpusEncoder;
    pub fn opus_encode(
        st: *mut OpusEncoder,
        pcm: *const opus_int16,
        frame_size: c_int,
        data: *mut c_uchar,
        max_data_bytes: opus_int32,
    ) -> opus_int32;
    pub fn opus_encoder_destroy(st: *mut OpusEncoder);
    pub fn opus_encoder_ctl(st: *mut OpusEncoder, request: c_int, ...) -> c_int;

    // Decoder
    pub fn opus_decoder_get_size(channels: c_int) -> c_int;
    pub fn opus_decoder_create(
        Fs: opus_int32,
        channels: c_int,
        error: *mut c_int,
    ) -> *mut OpusDecoder;
    pub fn opus_decode(
        st: *mut OpusDecoder,
        data: *const c_uchar,
        len: opus_int32,
        pcm: *mut opus_int16,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
    pub fn opus_decoder_destroy(st: *mut OpusDecoder);
    pub fn opus_decoder_ctl(st: *mut OpusDecoder, request: c_int, ...) -> c_int;

    // Packet introspection
    pub fn opus_packet_get_bandwidth(data: *const c_uchar) -> c_int;
    pub fn opus_packet_get_nb_channels(data: *const c_uchar) -> c_int;
    pub fn opus_packet_get_nb_frames(data: *const c_uchar, len: opus_int32) -> c_int;
    pub fn opus_packet_get_samples_per_frame(data: *const c_uchar, Fs: opus_int32) -> c_int;
    pub fn opus_packet_get_nb_samples(
        data: *const c_uchar,
        len: opus_int32,
        Fs: opus_int32,
    ) -> c_int;

    // Info
    pub fn opus_strerror(error: c_int) -> *const std::os::raw::c_char;
    pub fn opus_get_version_string() -> *const std::os::raw::c_char;

    // Range coder (internal CELT API)
    pub fn ec_enc_init(this: *mut ec_enc, buf: *mut c_uchar, size: opus_uint32);
    pub fn ec_enc_done(this: *mut ec_enc);
    pub fn ec_enc_uint(this: *mut ec_enc, fl: opus_uint32, ft: opus_uint32);
    pub fn ec_enc_bit_logp(this: *mut ec_enc, val: c_int, logp: c_uint);
    pub fn ec_enc_bits(this: *mut ec_enc, fl: opus_uint32, ftb: c_uint);

    pub fn ec_dec_init(this: *mut ec_dec, buf: *mut c_uchar, storage: opus_uint32);
    pub fn ec_dec_uint(this: *mut ec_dec, ft: opus_uint32) -> opus_uint32;
    pub fn ec_dec_bit_logp(this: *mut ec_dec, logp: c_uint) -> c_int;
    pub fn ec_dec_bits(this: *mut ec_dec, ftb: c_uint) -> opus_uint32;
    pub fn ec_dec_icdf(this: *mut ec_dec, icdf: *const c_uchar, ftb: c_uint) -> c_int;

    // Repacketizer
    pub fn opus_repacketizer_create() -> *mut OpusRepacketizer;
    pub fn opus_repacketizer_destroy(rp: *mut OpusRepacketizer);
    pub fn opus_repacketizer_init(rp: *mut OpusRepacketizer) -> *mut OpusRepacketizer;
    pub fn opus_repacketizer_cat(
        rp: *mut OpusRepacketizer,
        data: *const c_uchar,
        len: opus_int32,
    ) -> c_int;
    pub fn opus_repacketizer_out(
        rp: *mut OpusRepacketizer,
        data: *mut c_uchar,
        maxlen: opus_int32,
    ) -> opus_int32;
    pub fn opus_repacketizer_out_range(
        rp: *mut OpusRepacketizer,
        begin: c_int,
        end: c_int,
        data: *mut c_uchar,
        maxlen: opus_int32,
    ) -> opus_int32;
    pub fn opus_repacketizer_get_nb_frames(rp: *mut OpusRepacketizer) -> c_int;
    pub fn opus_packet_pad(
        data: *mut c_uchar,
        len: opus_int32,
        new_len: opus_int32,
    ) -> c_int;
    pub fn opus_packet_unpad(
        data: *mut c_uchar,
        len: opus_int32,
    ) -> opus_int32;

    // Multistream encoder
    pub fn opus_multistream_encoder_create(
        Fs: opus_int32,
        channels: c_int,
        streams: c_int,
        coupled_streams: c_int,
        mapping: *const c_uchar,
        application: c_int,
        error: *mut c_int,
    ) -> *mut OpusMSEncoder;
    pub fn opus_multistream_encode(
        st: *mut OpusMSEncoder,
        pcm: *const opus_int16,
        frame_size: c_int,
        data: *mut c_uchar,
        max_data_bytes: opus_int32,
    ) -> opus_int32;
    pub fn opus_multistream_encoder_destroy(st: *mut OpusMSEncoder);
    pub fn opus_multistream_encoder_ctl(st: *mut OpusMSEncoder, request: c_int, ...) -> c_int;

    // Multistream decoder
    pub fn opus_multistream_decoder_create(
        Fs: opus_int32,
        channels: c_int,
        streams: c_int,
        coupled_streams: c_int,
        mapping: *const c_uchar,
        error: *mut c_int,
    ) -> *mut OpusMSDecoder;
    pub fn opus_multistream_decode(
        st: *mut OpusMSDecoder,
        data: *const c_uchar,
        len: opus_int32,
        pcm: *mut opus_int16,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
    pub fn opus_multistream_decoder_destroy(st: *mut OpusMSDecoder);

    // Debug math comparison helpers
    pub fn debug_c_celt_sqrt32(x: opus_int32) -> opus_int32;
    pub fn debug_c_celt_atan2p_norm(y: opus_int32, x: opus_int32) -> opus_int32;
    pub fn debug_c_celt_atan_norm(x: opus_int32) -> opus_int32;
    pub fn debug_c_frac_div32(a: opus_int32, b: opus_int32) -> opus_int32;
    pub fn debug_c_stereo_itheta(
        x: *const opus_int32,
        y: *const opus_int32,
        stereo: c_int,
        n: c_int,
    ) -> opus_int32;
    pub fn debug_c_celt_inner_prod_norm_shift(
        x: *const opus_int32,
        y: *const opus_int32,
        len: c_int,
    ) -> opus_int32;
    pub fn debug_c_celt_cos_norm32(x: opus_int32) -> opus_int32;
    pub fn debug_c_celt_rsqrt_norm32(x: opus_int32) -> opus_int32;
    pub fn debug_c_normalise_residual_g(ryy: opus_int32, gain: opus_int32) -> opus_int32;
    pub fn debug_c_opus_fast_int64() -> c_int;
    pub fn debug_get_celt_old_band_e(
        dec: *mut OpusDecoder,
        out_buf: *mut opus_int32,
        max_len: c_int,
    ) -> c_int;
    pub fn debug_get_celt_postfilter(
        dec: *mut OpusDecoder,
        period: *mut opus_int32,
        period_old: *mut opus_int32,
        gain: *mut opus_int32,
        gain_old: *mut opus_int32,
        tapset: *mut opus_int32,
        tapset_old: *mut opus_int32,
    );
    pub fn debug_get_celt_old_log_e(
        dec: *mut OpusDecoder,
        out_log_e: *mut opus_int32,
        out_log_e2: *mut opus_int32,
        max_len: c_int,
    ) -> c_int;
    pub fn debug_c_decode_energy(
        data: *const c_uchar,
        len: c_int,
        old_bands_inout: *mut opus_int32,
        fine_quant_out: *mut opus_int32,
        cc: c_int,
        lm: c_int,
    );
    pub fn debug_c_hp_cutoff_stereo(
        input: *const opus_int16,
        cutoff_hz: opus_int32,
        output: *mut opus_int16,
        hp_mem: *mut opus_int32,
        len: c_int,
        fs: opus_int32,
    );
    pub fn debug_get_encoder_hp_state(
        enc: *mut OpusEncoder,
        hp_mem_out: *mut opus_int32,
        variable_hp_smth2: *mut opus_int32,
        mode_out: *mut opus_int32,
        stream_channels_out: *mut opus_int32,
        bandwidth_out: *mut opus_int32,
    );
}

// ---------------------------------------------------------------------------
// Safe wrappers
// ---------------------------------------------------------------------------

/// Decode an opus error code into a human-readable string.
pub fn error_string(code: c_int) -> &'static str {
    unsafe {
        let ptr = opus_strerror(code);
        if ptr.is_null() {
            return "unknown error";
        }
        std::ffi::CStr::from_ptr(ptr)
            .to_str()
            .unwrap_or("unknown error")
    }
}

/// Get the opus library version string.
pub fn version_string() -> &'static str {
    unsafe {
        let ptr = opus_get_version_string();
        if ptr.is_null() {
            return "unknown";
        }
        std::ffi::CStr::from_ptr(ptr).to_str().unwrap_or("unknown")
    }
}
