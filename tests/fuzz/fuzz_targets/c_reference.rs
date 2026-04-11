//! FFI bindings to the C reference opus library for differential fuzz testing.
//!
//! Only the public Opus API is exposed here — no debug helpers, no internal
//! functions. Safe wrappers manage the full lifecycle (create/configure/use/destroy)
//! to prevent leaks during fuzzing.

#![allow(non_camel_case_types, dead_code)]

use std::os::raw::{c_int, c_uchar};

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

pub const OPUS_APPLICATION_VOIP: c_int = 2048;
pub const OPUS_APPLICATION_AUDIO: c_int = 2049;
pub const OPUS_APPLICATION_RESTRICTED_LOWDELAY: c_int = 2051;

// CTL request codes
pub const OPUS_SET_BITRATE_REQUEST: c_int = 4002;
pub const OPUS_SET_VBR_REQUEST: c_int = 4006;
pub const OPUS_SET_COMPLEXITY_REQUEST: c_int = 4010;
pub const OPUS_SET_BANDWIDTH_REQUEST: c_int = 4008;
pub const OPUS_SET_FORCE_CHANNELS_REQUEST: c_int = 4022;

// Getter CTL request codes
pub const OPUS_GET_FINAL_RANGE_REQUEST: c_int = 4031;

// Bandwidth values
pub const OPUS_BANDWIDTH_NARROWBAND: c_int = 1101;
pub const OPUS_BANDWIDTH_MEDIUMBAND: c_int = 1102;
pub const OPUS_BANDWIDTH_WIDEBAND: c_int = 1103;
pub const OPUS_BANDWIDTH_SUPERWIDEBAND: c_int = 1104;
pub const OPUS_BANDWIDTH_FULLBAND: c_int = 1105;

// ---------------------------------------------------------------------------
// Opaque encoder/decoder/repacketizer handles
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

// ---------------------------------------------------------------------------
// Public Opus C API
// ---------------------------------------------------------------------------
// The C reference library (opus_ref) is compiled and linked by the main mdopus
// crate's build.rs. No #[link] attribute needed here — the symbols are already
// available through the mdopus dependency's link directives.
unsafe extern "C" {
    // Encoder
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
}

// ---------------------------------------------------------------------------
// Safe wrappers for differential fuzz testing
// ---------------------------------------------------------------------------

/// Decode an Opus packet using the C reference. Creates a decoder, decodes one
/// frame, destroys the decoder. Returns decoded PCM or an error code.
pub fn c_decode(data: &[u8], sample_rate: i32, channels: i32) -> Result<Vec<i16>, i32> {
    unsafe {
        let mut error: c_int = 0;
        let dec = opus_decoder_create(sample_rate, channels, &mut error);
        if dec.is_null() || error != OPUS_OK {
            if !dec.is_null() {
                opus_decoder_destroy(dec);
            }
            return Err(error);
        }

        // 5760 = max frame size at 48kHz (120ms)
        let max_frame = 5760 * channels as usize;
        let mut pcm = vec![0i16; max_frame];

        let ret = opus_decode(
            dec,
            data.as_ptr(),
            data.len() as opus_int32,
            pcm.as_mut_ptr(),
            5760,
            0,
        );

        opus_decoder_destroy(dec);

        if ret < 0 {
            Err(ret as i32)
        } else {
            pcm.truncate(ret as usize * channels as usize);
            Ok(pcm)
        }
    }
}

/// Encode PCM samples using the C reference. Creates an encoder, configures it,
/// encodes one frame, destroys the encoder. Returns compressed bytes or an error code.
pub fn c_encode(
    pcm: &[i16],
    frame_size: i32,
    sample_rate: i32,
    channels: i32,
    bitrate: i32,
    complexity: i32,
    application: i32,
) -> Result<Vec<u8>, i32> {
    unsafe {
        let mut error: c_int = 0;
        let enc = opus_encoder_create(sample_rate, channels, application, &mut error);
        if enc.is_null() || error != OPUS_OK {
            if !enc.is_null() {
                opus_encoder_destroy(enc);
            }
            return Err(error);
        }

        // Configure encoder
        opus_encoder_ctl(enc, OPUS_SET_BITRATE_REQUEST, bitrate);
        opus_encoder_ctl(enc, OPUS_SET_VBR_REQUEST, 0 as c_int);
        opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY_REQUEST, complexity);

        let max_data_bytes: opus_int32 = 4000;
        let mut out = vec![0u8; max_data_bytes as usize];

        let ret = opus_encode(
            enc,
            pcm.as_ptr(),
            frame_size,
            out.as_mut_ptr(),
            max_data_bytes,
        );

        opus_encoder_destroy(enc);

        if ret < 0 {
            Err(ret as i32)
        } else {
            out.truncate(ret as usize);
            Ok(out)
        }
    }
}

/// Encode multiple sequential frames using the C reference, keeping encoder state
/// across frames. Returns a Vec of compressed packets (one per frame) or an error.
pub fn c_encode_multiframe(
    pcm_frames: &[&[i16]],
    frame_size: i32,
    sample_rate: i32,
    channels: i32,
    bitrate: i32,
    complexity: i32,
    application: i32,
) -> Result<Vec<Vec<u8>>, i32> {
    unsafe {
        let mut error: c_int = 0;
        let enc = opus_encoder_create(sample_rate, channels, application, &mut error);
        if enc.is_null() || error != OPUS_OK {
            if !enc.is_null() {
                opus_encoder_destroy(enc);
            }
            return Err(error);
        }

        opus_encoder_ctl(enc, OPUS_SET_BITRATE_REQUEST, bitrate);
        opus_encoder_ctl(enc, OPUS_SET_VBR_REQUEST, 0 as c_int);
        opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY_REQUEST, complexity);

        let max_data_bytes: opus_int32 = 4000;
        let mut results = Vec::with_capacity(pcm_frames.len());

        for pcm in pcm_frames {
            let mut out = vec![0u8; max_data_bytes as usize];
            let ret = opus_encode(
                enc,
                pcm.as_ptr(),
                frame_size,
                out.as_mut_ptr(),
                max_data_bytes,
            );

            if ret < 0 {
                opus_encoder_destroy(enc);
                return Err(ret as i32);
            }
            out.truncate(ret as usize);
            results.push(out);
        }

        opus_encoder_destroy(enc);
        Ok(results)
    }
}

/// Decode multiple sequential packets using the C reference, keeping decoder state
/// across frames. Returns a Vec of decoded PCM buffers (one per packet) or an error.
pub fn c_decode_multiframe(
    packets: &[&[u8]],
    sample_rate: i32,
    channels: i32,
    frame_size: i32,
) -> Result<Vec<Vec<i16>>, i32> {
    unsafe {
        let mut error: c_int = 0;
        let dec = opus_decoder_create(sample_rate, channels, &mut error);
        if dec.is_null() || error != OPUS_OK {
            if !dec.is_null() {
                opus_decoder_destroy(dec);
            }
            return Err(error);
        }

        let max_pcm = frame_size as usize * channels as usize;
        let mut results = Vec::with_capacity(packets.len());

        for pkt in packets {
            let mut pcm = vec![0i16; max_pcm];
            let ret = opus_decode(
                dec,
                pkt.as_ptr(),
                pkt.len() as opus_int32,
                pcm.as_mut_ptr(),
                frame_size,
                0,
            );

            if ret < 0 {
                opus_decoder_destroy(dec);
                return Err(ret as i32);
            }
            pcm.truncate(ret as usize * channels as usize);
            results.push(pcm);
        }

        opus_decoder_destroy(dec);
        Ok(results)
    }
}

/// Get bandwidth from first byte of a packet using the C reference.
pub fn c_packet_get_bandwidth(data: &[u8]) -> i32 {
    if data.is_empty() {
        return -1;
    }
    unsafe { opus_packet_get_bandwidth(data.as_ptr()) }
}

/// Get number of channels from first byte of a packet using the C reference.
pub fn c_packet_get_nb_channels(data: &[u8]) -> i32 {
    if data.is_empty() {
        return -1;
    }
    unsafe { opus_packet_get_nb_channels(data.as_ptr()) }
}

/// Get number of frames in a packet using the C reference.
pub fn c_packet_get_nb_frames(data: &[u8]) -> i32 {
    if data.is_empty() {
        return -1;
    }
    unsafe { opus_packet_get_nb_frames(data.as_ptr(), data.len() as opus_int32) }
}

/// Get samples per frame for a packet at a given sample rate using the C reference.
pub fn c_packet_get_samples_per_frame(data: &[u8], fs: i32) -> i32 {
    if data.is_empty() {
        return -1;
    }
    unsafe { opus_packet_get_samples_per_frame(data.as_ptr(), fs) }
}

/// Get total number of samples in a packet using the C reference.
pub fn c_packet_get_nb_samples(data: &[u8], fs: i32) -> i32 {
    if data.is_empty() {
        return -1;
    }
    unsafe { opus_packet_get_nb_samples(data.as_ptr(), data.len() as opus_int32, fs) }
}
