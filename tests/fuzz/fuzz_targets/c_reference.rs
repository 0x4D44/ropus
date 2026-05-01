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
pub const OPUS_SET_BANDWIDTH_REQUEST: c_int = 4008;
pub const OPUS_SET_COMPLEXITY_REQUEST: c_int = 4010;
pub const OPUS_SET_INBAND_FEC_REQUEST: c_int = 4012;
pub const OPUS_SET_PACKET_LOSS_PERC_REQUEST: c_int = 4014;
pub const OPUS_SET_DTX_REQUEST: c_int = 4016;
pub const OPUS_SET_VBR_CONSTRAINT_REQUEST: c_int = 4020;
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

#[repr(C)]
pub struct OpusMSEncoder {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct OpusMSDecoder {
    _opaque: [u8; 0],
}

// ---------------------------------------------------------------------------
// Public Opus C API
// ---------------------------------------------------------------------------
// The C reference library (opus_ref) is compiled and linked by the main ropus
// crate's build.rs. No #[link] attribute needed here — the symbols are already
// available through the ropus dependency's link directives.
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
    pub fn opus_encode_float(
        st: *mut OpusEncoder,
        pcm: *const f32,
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
    pub fn opus_decode_float(
        st: *mut OpusDecoder,
        data: *const c_uchar,
        len: opus_int32,
        pcm: *mut f32,
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
    pub fn opus_repacketizer_out_range(
        rp: *mut OpusRepacketizer,
        begin: c_int,
        end: c_int,
        data: *mut c_uchar,
        maxlen: opus_int32,
    ) -> opus_int32;
    pub fn opus_repacketizer_get_nb_frames(rp: *mut OpusRepacketizer) -> c_int;

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
    pub fn opus_multistream_surround_encoder_create(
        Fs: opus_int32,
        channels: c_int,
        mapping_family: c_int,
        streams: *mut c_int,
        coupled_streams: *mut c_int,
        mapping: *mut c_uchar,
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

/// Configuration for the C-side encoder, mirroring the Rust prologue setters
/// in the differential fuzz targets so byte-exact compare stays valid.
#[derive(Clone, Copy)]
pub struct CEncodeConfig {
    pub bitrate: i32,
    pub complexity: i32,
    pub application: i32,
    pub vbr: i32,
    pub vbr_constraint: i32,
    pub inband_fec: i32,
    pub dtx: i32,
    pub loss_perc: i32,
}

#[inline]
unsafe fn apply_c_encoder_config(enc: *mut OpusEncoder, cfg: &CEncodeConfig) {
    unsafe {
        opus_encoder_ctl(enc, OPUS_SET_BITRATE_REQUEST, cfg.bitrate);
        opus_encoder_ctl(enc, OPUS_SET_VBR_REQUEST, cfg.vbr);
        opus_encoder_ctl(enc, OPUS_SET_VBR_CONSTRAINT_REQUEST, cfg.vbr_constraint);
        opus_encoder_ctl(enc, OPUS_SET_INBAND_FEC_REQUEST, cfg.inband_fec);
        opus_encoder_ctl(enc, OPUS_SET_DTX_REQUEST, cfg.dtx);
        opus_encoder_ctl(enc, OPUS_SET_PACKET_LOSS_PERC_REQUEST, cfg.loss_perc);
        opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY_REQUEST, cfg.complexity);
    }
}

/// Encode PCM samples using the C reference. Creates an encoder, configures it,
/// encodes one frame, destroys the encoder. Returns compressed bytes or an error code.
pub fn c_encode(
    pcm: &[i16],
    frame_size: i32,
    sample_rate: i32,
    channels: i32,
    cfg: &CEncodeConfig,
) -> Result<Vec<u8>, i32> {
    unsafe {
        let mut error: c_int = 0;
        let enc = opus_encoder_create(sample_rate, channels, cfg.application, &mut error);
        if enc.is_null() || error != OPUS_OK {
            if !enc.is_null() {
                opus_encoder_destroy(enc);
            }
            return Err(error);
        }

        apply_c_encoder_config(enc, cfg);

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

/// Float-PCM variant of `c_encode`. Mirrors `OpusEncoder::encode_float`.
pub fn c_encode_float(
    pcm: &[f32],
    frame_size: i32,
    sample_rate: i32,
    channels: i32,
    cfg: &CEncodeConfig,
) -> Result<Vec<u8>, i32> {
    unsafe {
        let mut error: c_int = 0;
        let enc = opus_encoder_create(sample_rate, channels, cfg.application, &mut error);
        if enc.is_null() || error != OPUS_OK {
            if !enc.is_null() {
                opus_encoder_destroy(enc);
            }
            return Err(error);
        }

        apply_c_encoder_config(enc, cfg);

        let max_data_bytes: opus_int32 = 4000;
        let mut out = vec![0u8; max_data_bytes as usize];

        let ret = opus_encode_float(
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

/// Float-PCM variant of `c_decode`. Mirrors `OpusDecoder::decode_float`.
pub fn c_decode_float(data: &[u8], sample_rate: i32, channels: i32) -> Result<Vec<f32>, i32> {
    unsafe {
        let mut error: c_int = 0;
        let dec = opus_decoder_create(sample_rate, channels, &mut error);
        if dec.is_null() || error != OPUS_OK {
            if !dec.is_null() {
                opus_decoder_destroy(dec);
            }
            return Err(error);
        }

        let max_frame = 5760 * channels as usize;
        let mut pcm = vec![0f32; max_frame];

        let ret = opus_decode_float(
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

/// Encode multiple sequential frames using the C reference, keeping encoder state
/// across frames. `frame_cfgs` must be the same length as `pcm_frames`; each
/// config is applied via CTL setters before its corresponding `opus_encode` call,
/// mirroring the Rust target's per-frame setter shuffle so byte-exact compare
/// stays valid. The encoder is created with `frame_cfgs[0].application`; mid-
/// stream application changes are not supported by the underlying API.
/// Returns a Vec of compressed packets (one per frame) or an error.
pub fn c_encode_multiframe(
    pcm_frames: &[&[i16]],
    frame_cfgs: &[CEncodeConfig],
    frame_size: i32,
    sample_rate: i32,
    channels: i32,
) -> Result<Vec<Vec<u8>>, i32> {
    if pcm_frames.len() != frame_cfgs.len() || frame_cfgs.is_empty() {
        return Err(-1);
    }
    unsafe {
        let mut error: c_int = 0;
        let enc = opus_encoder_create(sample_rate, channels, frame_cfgs[0].application, &mut error);
        if enc.is_null() || error != OPUS_OK {
            if !enc.is_null() {
                opus_encoder_destroy(enc);
            }
            return Err(error);
        }

        let max_data_bytes: opus_int32 = 4000;
        let mut results = Vec::with_capacity(pcm_frames.len());

        for (pcm, cfg) in pcm_frames.iter().zip(frame_cfgs.iter()) {
            apply_c_encoder_config(enc, cfg);

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

/// Result of a C-side cat-then-out-then-out_range repacketizer run, mirroring
/// what the Rust `fuzz_repacketizer_seq` target performs against
/// `OpusRepacketizer`.
pub struct CRepackOutcome {
    /// `cat()` return code for each input packet (in input order).
    pub cat_rets: Vec<i32>,
    /// Frame count after all `cat()` calls.
    pub nb_frames: i32,
    /// `out()` return code (negative on error, otherwise byte length written).
    pub out_ret: i32,
    /// Bytes written by `out()` (truncated to `out_ret` when non-negative).
    pub out_buf: Vec<u8>,
    /// `out_range()` return code (negative on error, otherwise byte length).
    pub out_range_ret: i32,
    /// Bytes written by `out_range()`.
    pub out_range_buf: Vec<u8>,
}

/// Cat each packet via the C `OpusRepacketizer`, then call `out()` and
/// `out_range(begin, end)` on the accumulated state. Returns all C-side
/// outputs so the Rust fuzz target can compare step-by-step.
pub fn c_repack_cat_then_out_range(
    packets: &[&[u8]],
    begin: i32,
    end: i32,
) -> CRepackOutcome {
    let total_len: usize = packets.iter().map(|p| p.len()).sum();
    let buf_cap = total_len + 1024;

    unsafe {
        let rp = opus_repacketizer_create();
        if rp.is_null() {
            return CRepackOutcome {
                cat_rets: vec![-1; packets.len()],
                nb_frames: -1,
                out_ret: -1,
                out_buf: Vec::new(),
                out_range_ret: -1,
                out_range_buf: Vec::new(),
            };
        }
        opus_repacketizer_init(rp);

        let mut cat_rets = Vec::with_capacity(packets.len());
        for pkt in packets {
            let r = opus_repacketizer_cat(rp, pkt.as_ptr(), pkt.len() as opus_int32);
            cat_rets.push(r as i32);
            if r != OPUS_OK {
                // C reference stops accepting frames after the first cat error
                // (state is poisoned for that frame), but later cats may still
                // succeed if the input is independently valid. Mirror the Rust
                // target by continuing and recording each return.
            }
        }

        let nb_frames = opus_repacketizer_get_nb_frames(rp);

        let mut out_buf = vec![0u8; buf_cap];
        let out_ret = opus_repacketizer_out(rp, out_buf.as_mut_ptr(), buf_cap as opus_int32);
        if out_ret > 0 {
            out_buf.truncate(out_ret as usize);
        } else {
            out_buf.clear();
        }

        let mut out_range_buf = vec![0u8; buf_cap];
        let out_range_ret = opus_repacketizer_out_range(
            rp,
            begin,
            end,
            out_range_buf.as_mut_ptr(),
            buf_cap as opus_int32,
        );
        if out_range_ret > 0 {
            out_range_buf.truncate(out_range_ret as usize);
        } else {
            out_range_buf.clear();
        }

        opus_repacketizer_destroy(rp);

        CRepackOutcome {
            cat_rets,
            nb_frames,
            out_ret,
            out_buf,
            out_range_ret,
            out_range_buf,
        }
    }
}

// ---------------------------------------------------------------------------
// Multistream wrappers
// ---------------------------------------------------------------------------

/// Configuration for the C-side multistream encoder, mirroring the Rust
/// baseline `apply_initial_config` setters in `fuzz_multistream`. The runtime
/// setter shuffle is mirrored separately via `apply_c_ms_setter_sequence` so
/// byte-exact compare stays valid for encode/roundtrip differential checks.
#[derive(Clone, Copy)]
pub struct CMSEncodeConfig {
    pub bitrate: i32,
    pub complexity: i32,
    pub application: i32,
    pub vbr: i32,
    pub inband_fec: i32,
    pub dtx: i32,
    pub loss_perc: i32,
}

// OPUS_SET_LFE = 4030 in opus_multistream.h
const OPUS_SET_LFE_REQUEST: c_int = 4030;

#[inline]
unsafe fn apply_c_ms_baseline_config(enc: *mut OpusMSEncoder, cfg: &CMSEncodeConfig) {
    unsafe {
        opus_multistream_encoder_ctl(enc, OPUS_SET_BITRATE_REQUEST, cfg.bitrate);
        opus_multistream_encoder_ctl(enc, OPUS_SET_VBR_REQUEST, cfg.vbr);
        opus_multistream_encoder_ctl(enc, OPUS_SET_INBAND_FEC_REQUEST, cfg.inband_fec);
        opus_multistream_encoder_ctl(enc, OPUS_SET_DTX_REQUEST, cfg.dtx);
        opus_multistream_encoder_ctl(enc, OPUS_SET_PACKET_LOSS_PERC_REQUEST, cfg.loss_perc);
        opus_multistream_encoder_ctl(enc, OPUS_SET_COMPLEXITY_REQUEST, cfg.complexity);
    }
}

/// Mirror of the Rust-side `apply_ms_encoder_setter_sequence`, applied to a C
/// `OpusMSEncoder` so the runtime-setter shuffle runs symmetrically on both
/// codecs and the byte-exact differential remains valid. Must match the Rust
/// helper byte-for-byte: same modulo, same range derivations, same bitrate
/// formula. Errors from individual CTL calls are ignored (mirrors Rust's
/// `let _ = enc.set_X(...)`).
pub fn apply_c_ms_setter_sequence(enc: *mut OpusMSEncoder, bytes: &[u8]) {
    unsafe {
        for chunk in bytes.chunks_exact(2).take(8) {
            match chunk[0] % 7 {
                0 => {
                    let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let bitrate = 6000_i32 + (raw as i32 % 504_001);
                    opus_multistream_encoder_ctl(enc, OPUS_SET_BITRATE_REQUEST, bitrate);
                }
                1 => {
                    opus_multistream_encoder_ctl(
                        enc,
                        OPUS_SET_COMPLEXITY_REQUEST,
                        (chunk[1] % 11) as c_int,
                    );
                }
                2 => {
                    opus_multistream_encoder_ctl(
                        enc,
                        OPUS_SET_VBR_REQUEST,
                        (chunk[1] & 1) as c_int,
                    );
                }
                3 => {
                    opus_multistream_encoder_ctl(
                        enc,
                        OPUS_SET_INBAND_FEC_REQUEST,
                        (chunk[1] % 3) as c_int,
                    );
                }
                4 => {
                    opus_multistream_encoder_ctl(
                        enc,
                        OPUS_SET_DTX_REQUEST,
                        (chunk[1] & 1) as c_int,
                    );
                }
                5 => {
                    opus_multistream_encoder_ctl(
                        enc,
                        OPUS_SET_PACKET_LOSS_PERC_REQUEST,
                        (chunk[1] % 101) as c_int,
                    );
                }
                6 => {
                    opus_multistream_encoder_ctl(
                        enc,
                        OPUS_SET_LFE_REQUEST,
                        (chunk[1] & 1) as c_int,
                    );
                }
                _ => unreachable!(),
            }
        }
    }
}

/// Outcome of a C-side multistream encode. `mapping`, `streams`, and
/// `coupled_streams` are populated by `opus_multistream_surround_encoder_create`;
/// callers should assert they match the Rust-side equivalents to detect
/// mapping-derivation drift.
pub struct CMSEncodeOutcome {
    pub packet: Vec<u8>,
    pub mapping: Vec<u8>,
    pub streams: i32,
    pub coupled_streams: i32,
}

/// Encode interleaved i16 PCM using the C reference multistream encoder built
/// via `opus_multistream_surround_encoder_create`. C computes its own mapping
/// from `mapping_family`/`channels`, and the caller asserts byte-equality with
/// the Rust-side mapping. After construction the baseline config is applied,
/// then the runtime-setter shuffle (`setter_bytes`) is applied symmetrically
/// with the Rust target.
pub fn c_ms_encode(
    pcm: &[i16],
    frame_size: i32,
    sample_rate: i32,
    channels: i32,
    mapping_family: i32,
    cfg: &CMSEncodeConfig,
    setter_bytes: &[u8],
) -> Result<CMSEncodeOutcome, i32> {
    unsafe {
        let mut error: c_int = 0;
        let mut c_streams: c_int = 0;
        let mut c_coupled: c_int = 0;
        let mut c_mapping = vec![0u8; channels.max(1) as usize];
        let enc = opus_multistream_surround_encoder_create(
            sample_rate,
            channels,
            mapping_family,
            &mut c_streams,
            &mut c_coupled,
            c_mapping.as_mut_ptr(),
            cfg.application,
            &mut error,
        );
        if enc.is_null() || error != OPUS_OK {
            if !enc.is_null() {
                opus_multistream_encoder_destroy(enc);
            }
            return Err(error);
        }

        apply_c_ms_baseline_config(enc, cfg);
        apply_c_ms_setter_sequence(enc, setter_bytes);

        let max_data_bytes: opus_int32 = 4000 * c_streams.max(1);
        let mut out = vec![0u8; max_data_bytes as usize];

        let ret = opus_multistream_encode(
            enc,
            pcm.as_ptr(),
            frame_size,
            out.as_mut_ptr(),
            max_data_bytes,
        );

        opus_multistream_encoder_destroy(enc);

        if ret < 0 {
            Err(ret as i32)
        } else {
            out.truncate(ret as usize);
            Ok(CMSEncodeOutcome {
                packet: out,
                mapping: c_mapping,
                streams: c_streams as i32,
                coupled_streams: c_coupled as i32,
            })
        }
    }
}

/// Decode a multistream packet using the C reference multistream decoder.
pub fn c_ms_decode(
    data: &[u8],
    sample_rate: i32,
    channels: i32,
    streams: i32,
    coupled_streams: i32,
    mapping: &[u8],
    frame_size: i32,
) -> Result<Vec<i16>, i32> {
    unsafe {
        let mut error: c_int = 0;
        let dec = opus_multistream_decoder_create(
            sample_rate,
            channels,
            streams,
            coupled_streams,
            mapping.as_ptr(),
            &mut error,
        );
        if dec.is_null() || error != OPUS_OK {
            if !dec.is_null() {
                opus_multistream_decoder_destroy(dec);
            }
            return Err(error);
        }

        let max_pcm = frame_size as usize * channels as usize;
        let mut pcm = vec![0i16; max_pcm];

        let ret = opus_multistream_decode(
            dec,
            data.as_ptr(),
            data.len() as opus_int32,
            pcm.as_mut_ptr(),
            frame_size,
            0,
        );

        opus_multistream_decoder_destroy(dec);

        if ret < 0 {
            Err(ret as i32)
        } else {
            pcm.truncate(ret as usize * channels as usize);
            Ok(pcm)
        }
    }
}
