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

// CTL request codes
pub const OPUS_SET_BITRATE_REQUEST: c_int = 4002;
pub const OPUS_SET_VBR_REQUEST: c_int = 4006;
pub const OPUS_SET_COMPLEXITY_REQUEST: c_int = 4010;

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

    // Debug helper
    pub fn debug_dump_silk_indices(enc: *mut OpusEncoder);
    pub fn debug_test_gains_quant();
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
        std::ffi::CStr::from_ptr(ptr)
            .to_str()
            .unwrap_or("unknown")
    }
}
