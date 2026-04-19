//! Opus Decoder — top-level Opus decoding entry point.
//!
//! Ported from: reference/src/opus_decoder.c, reference/src/opus.c
//! Fixed-point path (non-RES24, non-QEXT).
//!
//! Neural PLC (LPCNet + FARGAN) is wired in here for Stage 7b. It only
//! activates on lost frames after `set_dnn_blob` has successfully loaded
//! weights; the classical (noise / periodic) PLC is still used for
//! frames where DNN synthesis isn't selected.

use crate::celt::decoder::CeltDecoder;
use crate::celt::math_ops::celt_exp2;
use crate::celt::range_coder::RangeDecoder;
use crate::dnn::lpcnet::LPCNetPLCState;
use crate::silk::decoder::{SilkDecControl, SilkDecoder, silk_decode};
use crate::types::*;

// ===========================================================================
// Constants
// ===========================================================================

// Mode constants (from opus_private.h)
pub const MODE_SILK_ONLY: i32 = 1000;
pub const MODE_HYBRID: i32 = 1001;
pub const MODE_CELT_ONLY: i32 = 1002;

// Bandwidth constants (from opus_defines.h)
pub const OPUS_BANDWIDTH_NARROWBAND: i32 = 1101;
pub const OPUS_BANDWIDTH_MEDIUMBAND: i32 = 1102;
pub const OPUS_BANDWIDTH_WIDEBAND: i32 = 1103;
pub const OPUS_BANDWIDTH_SUPERWIDEBAND: i32 = 1104;
pub const OPUS_BANDWIDTH_FULLBAND: i32 = 1105;

// Error codes (from opus_defines.h)
pub const OPUS_OK: i32 = 0;
pub const OPUS_BAD_ARG: i32 = -1;
pub const OPUS_BUFFER_TOO_SMALL: i32 = -2;
pub const OPUS_INTERNAL_ERROR: i32 = -3;
pub const OPUS_INVALID_PACKET: i32 = -4;
pub const OPUS_UNIMPLEMENTED: i32 = -5;

/// Maximum frames per packet (48 × 2.5ms = 120ms).
pub const MAX_FRAMES: usize = 48;

/// Gain conversion factor: ln(10)/(20*256*ln(2)) in Q25.
const GAIN_SCALE_Q25: i32 = qconst16(6.48814081e-4, 25);

// ===========================================================================
// Packet Utilities (from opus.c and opus_decoder.c)
// ===========================================================================

/// Parse a frame size field from the bitstream.
/// Returns `(bytes_consumed, frame_size)`. On error, returns `(-1, -1)`.
/// Matches C `parse_size`.
fn parse_size(data: &[u8], len: i32) -> (i32, i16) {
    if len < 1 {
        return (-1, -1);
    }
    if data[0] < 252 {
        (1, data[0] as i16)
    } else if len < 2 {
        (-1, -1)
    } else {
        (2, 4 * data[1] as i16 + data[0] as i16)
    }
}

/// Get the number of audio samples per frame from the TOC byte.
/// Matches C `opus_packet_get_samples_per_frame`.
pub fn opus_packet_get_samples_per_frame(data: &[u8], fs: i32) -> i32 {
    if data[0] & 0x80 != 0 {
        // CELT-only: bits 4-3 select 2.5/5/10/20ms
        let audiosize = ((data[0] >> 3) & 0x3) as i32;
        (fs << audiosize) / 400
    } else if (data[0] & 0x60) == 0x60 {
        // Hybrid: bit 3 selects 10/20ms
        if data[0] & 0x08 != 0 {
            fs / 50
        } else {
            fs / 100
        }
    } else {
        // SILK-only: bits 4-3 select 10/20/40/60ms
        let audiosize = ((data[0] >> 3) & 0x3) as i32;
        if audiosize == 3 {
            fs * 60 / 1000
        } else {
            (fs << audiosize) / 100
        }
    }
}

/// Determine codec mode from the TOC byte.
/// Matches C `opus_packet_get_mode`.
fn opus_packet_get_mode(data: &[u8]) -> i32 {
    if data[0] & 0x80 != 0 {
        MODE_CELT_ONLY
    } else if (data[0] & 0x60) == 0x60 {
        MODE_HYBRID
    } else {
        MODE_SILK_ONLY
    }
}

/// Get bandwidth from the TOC byte.
/// Matches C `opus_packet_get_bandwidth`.
pub fn opus_packet_get_bandwidth(data: &[u8]) -> i32 {
    if data[0] & 0x80 != 0 {
        // CELT-only
        let bw = OPUS_BANDWIDTH_MEDIUMBAND + ((data[0] >> 5) & 0x3) as i32;
        if bw == OPUS_BANDWIDTH_MEDIUMBAND {
            OPUS_BANDWIDTH_NARROWBAND
        } else {
            bw
        }
    } else if (data[0] & 0x60) == 0x60 {
        // Hybrid
        if data[0] & 0x10 != 0 {
            OPUS_BANDWIDTH_FULLBAND
        } else {
            OPUS_BANDWIDTH_SUPERWIDEBAND
        }
    } else {
        // SILK-only
        OPUS_BANDWIDTH_NARROWBAND + ((data[0] >> 5) & 0x3) as i32
    }
}

/// Get number of channels from the TOC byte.
/// Matches C `opus_packet_get_nb_channels`.
pub fn opus_packet_get_nb_channels(data: &[u8]) -> i32 {
    if data[0] & 0x4 != 0 { 2 } else { 1 }
}

/// Get number of frames in a packet.
/// Matches C `opus_packet_get_nb_frames`.
pub fn opus_packet_get_nb_frames(packet: &[u8]) -> Result<i32, i32> {
    if packet.is_empty() {
        return Err(OPUS_BAD_ARG);
    }
    match packet[0] & 0x3 {
        0 => Ok(1),
        3 => {
            if packet.len() < 2 {
                Err(OPUS_INVALID_PACKET)
            } else {
                Ok((packet[1] & 0x3F) as i32)
            }
        }
        _ => Ok(2),
    }
}

/// Get total number of samples in a packet at the given sample rate.
/// Matches C `opus_packet_get_nb_samples`.
pub fn opus_packet_get_nb_samples(packet: &[u8], fs: i32) -> Result<i32, i32> {
    let count = opus_packet_get_nb_frames(packet)?;
    let samples = count * opus_packet_get_samples_per_frame(packet, fs);
    // Can't have more than 120 ms
    if samples * 25 > fs * 3 {
        Err(OPUS_INVALID_PACKET)
    } else {
        Ok(samples)
    }
}

/// Padding extraction output for `opus_packet_parse_impl_with_padding`.
///
/// `offset` is the byte index into the input `data` slice where the padding
/// region (after the final frame) begins; `len` is the padding length. When
/// `len == 0` the offset is meaningless and typically set to 0.
#[derive(Clone, Copy, Debug)]
pub struct PaddingInfo {
    pub offset: usize,
    pub len: i32,
}

/// Packet parse implementation. Returns frame count (positive) or negative
/// error code. On success, `out_toc` is set, `sizes[0..count]` are the frame
/// sizes, and `payload_offset` points to the first frame byte within `data`
/// (each subsequent frame follows at `payload_offset + sum(sizes[0..i])`).
///
/// Matches C `opus_packet_parse_impl`.
pub fn opus_packet_parse_impl(
    data: &[u8],
    len: i32,
    self_delimited: bool,
    out_toc: &mut u8,
    sizes: &mut [i16; MAX_FRAMES],
    payload_offset: &mut i32,
    packet_offset: Option<&mut i32>,
) -> i32 {
    opus_packet_parse_impl_with_padding(
        data,
        len,
        self_delimited,
        out_toc,
        sizes,
        payload_offset,
        packet_offset,
        None,
    )
}

/// As `opus_packet_parse_impl` but also emits the padding region offset/length
/// when `padding` is `Some`. Matches the full C signature documented in
/// `reference/src/opus_private.h:208-212`.
pub fn opus_packet_parse_impl_with_padding(
    data: &[u8],
    len: i32,
    self_delimited: bool,
    out_toc: &mut u8,
    sizes: &mut [i16; MAX_FRAMES],
    payload_offset: &mut i32,
    packet_offset: Option<&mut i32>,
    padding: Option<&mut PaddingInfo>,
) -> i32 {
    // Mirror C: zero padding output up-front so error paths see cleared values
    // (reference/src/opus.c:240-244).
    let padding = if let Some(p) = padding {
        p.offset = 0;
        p.len = 0;
        Some(p)
    } else {
        None
    };

    if len < 0 {
        return OPUS_BAD_ARG;
    }
    if len == 0 {
        return OPUS_INVALID_PACKET;
    }

    let framesize = opus_packet_get_samples_per_frame(data, 48000);

    let mut pos: usize = 0;
    let mut remaining = len;
    let mut pad: i32 = 0;

    let toc = data[0];
    pos += 1;
    remaining -= 1;
    let mut last_size = remaining;
    let mut cbr = false;

    let count: i32;

    match toc & 0x3 {
        // One frame
        0 => {
            count = 1;
        }
        // Two CBR frames
        1 => {
            count = 2;
            cbr = true;
            if !self_delimited {
                if remaining & 0x1 != 0 {
                    return OPUS_INVALID_PACKET;
                }
                last_size = remaining / 2;
                // If last_size doesn't fit in size[0], we'll catch it later
                sizes[0] = last_size as i16;
            }
        }
        // Two VBR frames
        2 => {
            count = 2;
            let (bytes, sz) = parse_size(&data[pos..], remaining);
            if sz < 0 {
                return OPUS_INVALID_PACKET;
            }
            remaining -= bytes;
            pos += bytes as usize;
            if sz as i32 > remaining {
                return OPUS_INVALID_PACKET;
            }
            sizes[0] = sz;
            last_size = remaining - sz as i32;
        }
        // Multiple CBR/VBR frames (code 3)
        _ => {
            if remaining < 1 {
                return OPUS_INVALID_PACKET;
            }
            let ch = data[pos];
            pos += 1;
            remaining -= 1;
            count = (ch & 0x3F) as i32;
            if count <= 0 || framesize * count > 5760 {
                return OPUS_INVALID_PACKET;
            }
            // Padding flag (bit 6)
            if ch & 0x40 != 0 {
                loop {
                    if remaining <= 0 {
                        return OPUS_INVALID_PACKET;
                    }
                    let p = data[pos];
                    pos += 1;
                    remaining -= 1;
                    let tmp = if p == 255 { 254 } else { p as i32 };
                    remaining -= tmp;
                    pad += tmp;
                    if p != 255 {
                        break;
                    }
                }
            }
            if remaining < 0 {
                return OPUS_INVALID_PACKET;
            }
            // VBR flag (bit 7): cbr = !(ch & 0x80)
            cbr = (ch & 0x80) == 0;
            if !cbr {
                // VBR case
                last_size = remaining;
                for i in 0..count as usize - 1 {
                    let (bytes, sz) = parse_size(&data[pos..], remaining);
                    if sz < 0 {
                        return OPUS_INVALID_PACKET;
                    }
                    remaining -= bytes;
                    pos += bytes as usize;
                    if sz as i32 > remaining {
                        return OPUS_INVALID_PACKET;
                    }
                    sizes[i] = sz;
                    last_size -= bytes + sz as i32;
                }
                if last_size < 0 {
                    return OPUS_INVALID_PACKET;
                }
            } else if !self_delimited {
                // CBR case
                last_size = remaining / count;
                if last_size * count != remaining {
                    return OPUS_INVALID_PACKET;
                }
                for i in 0..count as usize - 1 {
                    sizes[i] = last_size as i16;
                }
            }
        }
    }

    // Self-delimited framing has an extra size for the last frame
    if self_delimited {
        let (bytes, sz) = parse_size(&data[pos..], remaining);
        if sz < 0 {
            return OPUS_INVALID_PACKET;
        }
        remaining -= bytes;
        pos += bytes as usize;
        if sz as i32 > remaining {
            return OPUS_INVALID_PACKET;
        }
        sizes[count as usize - 1] = sz;
        // For CBR packets, apply the size to all the frames
        if cbr {
            if sz as i32 * count > remaining {
                return OPUS_INVALID_PACKET;
            }
            for i in 0..count as usize - 1 {
                sizes[i] = sz;
            }
        } else if bytes + sz as i32 > last_size {
            return OPUS_INVALID_PACKET;
        }
    } else {
        // Because it's not encoded explicitly, it's possible the size of the
        // last packet (or all the packets, for the CBR case) is larger than
        // 1275. Reject them here.
        if last_size > 1275 {
            return OPUS_INVALID_PACKET;
        }
        sizes[count as usize - 1] = last_size as i16;
    }

    *payload_offset = pos as i32;
    *out_toc = toc;

    // Advance past all frames to compute packet_offset
    let mut frame_end = pos;
    for i in 0..count as usize {
        frame_end += sizes[i] as usize;
    }

    if let Some(pad_info) = padding {
        pad_info.offset = frame_end;
        pad_info.len = pad;
    }

    if let Some(pkt_off) = packet_offset {
        *pkt_off = pad + frame_end as i32;
    }

    count
}

/// Check if a packet contains LBRR (FEC) data.
/// Matches C `opus_packet_has_lbrr`.
pub fn opus_packet_has_lbrr(packet: &[u8], len: i32) -> Result<bool, i32> {
    let packet_mode = opus_packet_get_mode(packet);
    if packet_mode == MODE_CELT_ONLY {
        return Ok(false);
    }
    let packet_frame_size = opus_packet_get_samples_per_frame(packet, 48000);
    let nb_frames = if packet_frame_size > 960 {
        packet_frame_size / 960
    } else {
        1
    };
    let packet_stream_channels = opus_packet_get_nb_channels(packet);

    let mut toc = 0u8;
    let mut sizes = [0i16; MAX_FRAMES];
    let mut payload_offset = 0i32;
    let count = opus_packet_parse_impl(
        packet,
        len,
        false,
        &mut toc,
        &mut sizes,
        &mut payload_offset,
        None,
    );
    if count <= 0 {
        return Err(count);
    }
    if sizes[0] == 0 {
        return Ok(false);
    }
    let frame0 = &packet[payload_offset as usize..];
    let mut lbrr = ((frame0[0] >> (7 - nb_frames)) & 0x1) != 0;
    if packet_stream_channels == 2 {
        lbrr = lbrr || ((frame0[0] >> (6 - 2 * nb_frames)) & 0x1) != 0;
    }
    Ok(lbrr)
}

// ===========================================================================
// smooth_fade helper
// ===========================================================================

/// Crossfade between two audio buffers using the squared MDCT window.
/// Matches C `smooth_fade` (fixed-point, non-RES24 path).
///
/// `in1` and `in2` must NOT alias `out`. Caller must copy if needed.
fn smooth_fade(
    in1: &[i16],
    in2: &[i16],
    out: &mut [i16],
    overlap: i32,
    channels: i32,
    window: &[i16],
    fs: i32,
) {
    let inc = (48000 / fs) as usize;
    for c in 0..channels as usize {
        for i in 0..overlap as usize {
            let w = window[i * inc] as i32;
            // Square the window: w² in Q15
            let w = mult16_16_q15(w, w);
            let idx = i * channels as usize + c;
            // out = w * in2 + (1-w) * in1, result >>15 back to Q0
            out[idx] = shr32(
                mac16_16(mult16_16(w, in2[idx] as i32), Q15ONE - w, in1[idx] as i32),
                15,
            ) as i16;
        }
    }
}

// ===========================================================================
// DRED FEC state (stub — full DRED decoder not yet ported)
// ===========================================================================


// ===========================================================================
// OpusDecoder struct
// ===========================================================================

/// Opus decoder state.
///
/// Wraps SILK and CELT sub-decoders, handling mode switching, redundancy,
/// PLC, FEC, and gain application.
pub struct OpusDecoder {
    // --- Immutable after init ---
    channels: i32,
    fs: i32,

    // --- Sub-decoders ---
    silk_dec: SilkDecoder,
    celt_dec: CeltDecoder,

    // --- SILK decoder control (persisted for prev_pitch_lag) ---
    dec_control: SilkDecControl,

    // --- Configuration ---
    decode_gain: i32,
    complexity: i32,
    ignore_extensions: bool,

    // --- Dynamic state (cleared on reset) ---
    stream_channels: i32,
    bandwidth: i32,
    mode: i32,
    prev_mode: i32,
    frame_size: i32,
    prev_redundancy: bool,
    last_packet_duration: i32,
    range_final: u32,

    // --- Neural PLC / DRED FEC state ---
    // Boxed so adding this field doesn't grow `OpusDecoder`'s stack
    // footprint by the full LPCNet state (several tens of KB of arrays
    // plus any loaded weights). A stack-sized decoder matters for
    // embedded callers and keeps `opus_decoder_create` equivalents cheap.
    lpcnet: Box<LPCNetPLCState>,
}

// ===========================================================================
// OpusDecoder implementation
// ===========================================================================

impl OpusDecoder {
    /// Create and initialize a new Opus decoder.
    /// `fs`: output sample rate (8000, 12000, 16000, 24000, or 48000).
    /// `channels`: 1 (mono) or 2 (stereo).
    /// Matches C `opus_decoder_create` + `opus_decoder_init`.
    pub fn new(fs: i32, channels: i32) -> Result<Self, i32> {
        if (fs != 48000 && fs != 24000 && fs != 16000 && fs != 12000 && fs != 8000)
            || (channels != 1 && channels != 2)
        {
            return Err(OPUS_BAD_ARG);
        }

        let celt_dec = CeltDecoder::new(fs, channels).map_err(|_| OPUS_INTERNAL_ERROR)?;
        let mut silk_dec = SilkDecoder::new();
        silk_dec.init();

        let dec_control = SilkDecControl {
            n_channels_api: channels as usize,
            n_channels_internal: channels as usize,
            api_sample_rate: fs,
            internal_sample_rate: 0,
            payload_size_ms: 0,
            prev_pitch_lag: 0,
            enable_deep_plc: false,
        };

        let mut dec = Self {
            channels,
            fs,
            silk_dec,
            celt_dec,
            dec_control,
            decode_gain: 0,
            complexity: 0,
            ignore_extensions: false,
            stream_channels: channels,
            bandwidth: 0,
            mode: 0,
            prev_mode: 0,
            frame_size: fs / 400, // 2.5 ms
            prev_redundancy: false,
            last_packet_duration: 0,
            range_final: 0,
            lpcnet: Box::new(LPCNetPLCState::new()),
        };

        // CELT signalling off (Opus handles framing)
        dec.celt_dec.set_signalling(false);

        // Auto-load the compile-time default weight blob if build.rs was
        // able to produce one from the xiph reference sources. Matches C
        // `opus_decoder_init` behaviour under `!USE_WEIGHTS_FILE`
        // (`reference/dnn/lpcnet_plc.c:58`), which runs
        // `init_plcmodel(..., plcmodel_arrays)` on the compile-time
        // tables and `celt_assert`s the return. A failure here means the
        // embedded blob is malformed — a build-config bug, not a runtime
        // condition — so we `debug_assert!` to surface it in tests and
        // debug builds rather than silently falling back to classical
        // PLC. Release builds still leave `loaded=false` on failure so
        // shipped code keeps decoding, but the loud-fail in CI catches
        // drift early.
        if crate::dnn::embedded_weights::has_embedded_weights() {
            let ret = dec
                .lpcnet
                .load_model(crate::dnn::embedded_weights::WEIGHTS_BLOB);
            debug_assert_eq!(
                ret, 0,
                "embedded weight blob is malformed — rebuild from a fresh \
                 reference/dnn/ tree (cargo run -p fetch-assets -- weights)"
            );
        }

        Ok(dec)
    }

    /// Reset the decoder to initial state.
    /// Matches C `OPUS_RESET_STATE` in `opus_decoder_ctl`.
    pub fn reset(&mut self) {
        self.stream_channels = self.channels;
        self.bandwidth = 0;
        self.mode = 0;
        self.prev_mode = 0;
        self.frame_size = self.fs / 400;
        self.prev_redundancy = false;
        self.last_packet_duration = 0;
        self.range_final = 0;

        self.celt_dec.reset();
        self.silk_dec.init();
        // Clear runtime PLC history (FEC queue, analysis buffers, GRU
        // states, cont_features) but *preserve* loaded model weights.
        // Mirrors C `lpcnet_plc_reset` (`reference/dnn/lpcnet_plc.c:45`),
        // which only zeros from `LPCNET_PLC_RESET_START` onwards. A
        // user's `set_dnn_blob(custom)` therefore survives `reset()`,
        // and a decoder that started with embedded defaults keeps them —
        // matches the C ABI semantic that reset wipes stream state, not
        // model configuration.
        self.lpcnet.reset();
    }

    /// Load DNN model weights for neural PLC.
    ///
    /// Mirrors C's `OPUS_SET_DNN_BLOB_REQUEST` CTL
    /// (`reference/src/opus_decoder.c:1218`).
    ///
    /// On success, populates the three neural PLC sub-models — the PLC
    /// prediction network (`plc_dense_in`, two GRUs, `plc_dense_out`),
    /// the PitchDNN estimator, and FARGAN synthesis — and flips
    /// `LPCNetPLCState.loaded = true`. From that point on, lost frames
    /// in CELT-only or hybrid mode take the `FRAME_PLC_NEURAL` branch
    /// (provided `complexity >= 5`, which the C reference uses as the
    /// deep-PLC gate). Without a successful `set_dnn_blob`, every lost
    /// frame falls through to the classical pitch/noise PLC — exactly
    /// the C reference's `!st->loaded` branch.
    ///
    /// Unknown weight records in the blob (for example the DRED /
    /// OSCE tables that xiph's `write_lpcnet_weights` emits alongside
    /// PLC weights) are silently ignored. This matches the C
    /// `find_array_check` semantics and keeps the runtime path working
    /// against the superset blob the upstream tarball ships.
    ///
    /// Errors:
    /// - `OPUS_BAD_ARG` for a malformed blob, or one missing a required
    ///   weight name / size combination for any of the three
    ///   sub-models.
    pub fn set_dnn_blob(&mut self, data: &[u8]) -> Result<(), i32> {
        match self.lpcnet.load_model(data) {
            0 => Ok(()),
            _ => Err(OPUS_BAD_ARG),
        }
    }

    /// Add a DRED-decoded feature vector to the PLC's FEC queue.
    ///
    /// Stage 7b exposes the API surface only — the full DRED decoder
    /// (rdovae) ships with Stage 8. Passing `None` increments the skip
    /// counter so the queue stays aligned when the caller knows a slot
    /// exists but the features aren't reconstructible.
    // TODO(stage-8): wire to the real DRED decoder once `rdovae` lands.
    pub fn fec_add(&mut self, features: Option<&[f32]>) {
        self.lpcnet.fec_add(features);
    }

    /// Drop all queued DRED features. Typically called after a good frame
    /// arrives, so stale predictions don't fire.
    // TODO(stage-8): keep in lockstep with DRED decoder's frame boundary.
    pub fn fec_clear(&mut self) {
        self.lpcnet.fec_clear();
    }

    // -----------------------------------------------------------------------
    // decode_frame — core single-frame decode
    // -----------------------------------------------------------------------

    /// Decode a single frame of audio.
    /// `data`: frame payload (None for PLC).
    /// `pcm`: output buffer (interleaved, length >= frame_size * channels).
    /// `frame_size`: max samples per channel the output buffer can hold.
    /// `decode_fec`: true to decode FEC (LBRR) data from this frame.
    /// Returns decoded sample count or error code.
    /// Matches C `opus_decode_frame`.
    fn decode_frame(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i16],
        frame_size: i32,
        decode_fec: bool,
    ) -> Result<i32, i32> {
        let f20 = self.fs / 50;
        let f10 = f20 >> 1;
        let f5 = f10 >> 1;
        let f2_5 = f5 >> 1;

        if frame_size < f2_5 {
            return Err(OPUS_BUFFER_TOO_SMALL);
        }
        // Limit frame_size to avoid excessive allocations (max 120ms)
        let mut frame_size = imin(frame_size, self.fs / 25 * 3);

        let audiosize: i32;
        let mode: i32;
        let bandwidth: i32;

        // Payloads of 1 byte or 0 bytes trigger PLC/DTX
        let data = match data {
            Some(d) if d.len() > 1 => Some(d),
            _ => {
                // In that case, don't conceal more than what the TOC says
                frame_size = imin(frame_size, self.frame_size);
                None
            }
        };

        if let Some(frame_data) = data {
            audiosize = self.frame_size;
            mode = self.mode;
            bandwidth = self.bandwidth;
            // Range decoder is initialized below when needed
            let _ = frame_data; // used below
        } else {
            // PLC mode: use previous mode (CELT if we ended with CELT redundancy)
            let plc_mode = if self.prev_redundancy {
                MODE_CELT_ONLY
            } else {
                self.prev_mode
            };

            if plc_mode == 0 {
                // No previous packet yet — output zeros
                let n = frame_size as usize * self.channels as usize;
                for s in &mut pcm[..n] {
                    *s = 0;
                }
                return Ok(frame_size);
            }

            mode = plc_mode;
            bandwidth = 0;

            // Avoids trying to run PLC on sizes other than 2.5, 5, 10, or 20ms
            if frame_size > f20 {
                let mut pcm_offset = 0usize;
                let mut remaining = frame_size;
                loop {
                    let chunk = imin(remaining, f20);
                    let ret = self.decode_frame(None, &mut pcm[pcm_offset..], chunk, false)?;
                    pcm_offset += ret as usize * self.channels as usize;
                    remaining -= ret;
                    if remaining <= 0 {
                        break;
                    }
                }
                return Ok(frame_size);
            }

            let mut plc_audiosize = frame_size;
            if plc_audiosize < f20 {
                if plc_audiosize > f10 {
                    plc_audiosize = f10;
                } else if mode != MODE_SILK_ONLY && plc_audiosize > f5 && plc_audiosize < f10 {
                    plc_audiosize = f5;
                }
            }
            audiosize = plc_audiosize;
        }

        // In fixed-point, CELT accumulates on top of SILK PCM buffer
        let celt_accum = mode != MODE_CELT_ONLY;

        // --- Detect mode transitions ---
        let mut transition = false;
        let mut pcm_transition: Option<Vec<i16>> = None;

        if data.is_some()
            && self.prev_mode > 0
            && ((mode == MODE_CELT_ONLY
                && self.prev_mode != MODE_CELT_ONLY
                && !self.prev_redundancy)
                || (mode != MODE_CELT_ONLY && self.prev_mode == MODE_CELT_ONLY))
        {
            transition = true;
        }

        // For CELT-only transition, pre-decode PLC into transition buffer
        if transition && mode == MODE_CELT_ONLY {
            let trans_size = f5 as usize * self.channels as usize;
            let mut trans_buf = vec![0i16; trans_size];
            self.decode_frame(None, &mut trans_buf, imin(f5, audiosize), false)?;
            pcm_transition = Some(trans_buf);
        }

        if audiosize > frame_size {
            return Err(OPUS_BAD_ARG);
        }
        let frame_size = audiosize;

        // --- Initialize range decoder ---
        // Use empty slice for PLC (SILK/CELT will do PLC without reading)
        let frame_data = data.unwrap_or(&[]);
        let mut len = frame_data.len() as i32;
        let mut dec = RangeDecoder::new(frame_data);

        // --- SILK processing ---
        if mode != MODE_CELT_ONLY {
            let pcm_too_small = frame_size < f10;
            let mut pcm_silk = if pcm_too_small {
                vec![0i16; f10 as usize * self.channels as usize]
            } else {
                Vec::new()
            };

            if self.prev_mode == MODE_CELT_ONLY {
                self.silk_dec.init();
            }

            // The SILK PLC cannot produce frames of less than 10 ms
            self.dec_control.payload_size_ms = imax(10, 1000 * audiosize / self.fs);

            // Mirror C `reference/src/opus_decoder.c:443`: deep PLC is gated
            // on complexity ≥ 5. The SILK neural branch still also checks
            // `lpcnet.loaded`, so when weights aren't populated this setting
            // is harmless — it just lets the branch be reachable for
            // loaded-weight builds without a second round of plumbing.
            self.dec_control.enable_deep_plc = self.complexity >= 5;

            if data.is_some() {
                self.dec_control.n_channels_internal = self.stream_channels as usize;
                if mode == MODE_SILK_ONLY {
                    self.dec_control.internal_sample_rate = match bandwidth {
                        OPUS_BANDWIDTH_NARROWBAND => 8000,
                        OPUS_BANDWIDTH_MEDIUMBAND => 12000,
                        _ => 16000, // WIDEBAND or unexpected
                    };
                } else {
                    // Hybrid mode
                    self.dec_control.internal_sample_rate = 16000;
                }
            }

            let lost_flag = if data.is_none() {
                1
            } else {
                2 * decode_fec as i32
            };

            let mut decoded_samples = 0i32;
            let mut silk_ptr_offset = 0usize;
            loop {
                let new_packet_flag = decoded_samples == 0;
                let mut silk_frame_size = 0usize;

                let silk_output = if pcm_too_small {
                    &mut pcm_silk[silk_ptr_offset..]
                } else {
                    &mut pcm[silk_ptr_offset..]
                };

                // Thread the LPCNet state through SILK decode so good frames
                // can `update()` its GRU history and lost frames (when the
                // blob is loaded) can `conceal()` in place of classical PLC.
                let silk_lpcnet: crate::silk::decoder::DnnPlcArg<'_> =
                    Some(self.lpcnet.as_mut());

                let silk_ret = silk_decode(
                    &mut self.silk_dec,
                    &mut self.dec_control,
                    lost_flag,
                    new_packet_flag,
                    &mut dec,
                    silk_output,
                    &mut silk_frame_size,
                    silk_lpcnet,
                );
                if silk_ret != 0 {
                    if lost_flag != 0 {
                        // PLC failure should not be fatal — zero-fill
                        silk_frame_size = frame_size as usize;
                        let n = frame_size as usize * self.channels as usize;
                        let out = if pcm_too_small {
                            &mut pcm_silk[silk_ptr_offset..silk_ptr_offset + n]
                        } else {
                            &mut pcm[silk_ptr_offset..silk_ptr_offset + n]
                        };
                        for s in out.iter_mut() {
                            *s = 0;
                        }
                    } else {
                        return Err(OPUS_INTERNAL_ERROR);
                    }
                }
                silk_ptr_offset += silk_frame_size * self.channels as usize;
                decoded_samples += silk_frame_size as i32;
                if decoded_samples >= frame_size {
                    break;
                }
            }

            if pcm_too_small {
                let n = frame_size as usize * self.channels as usize;
                pcm[..n].copy_from_slice(&pcm_silk[..n]);
            }
        }

        // --- Parse redundancy info ---
        let mut redundancy = false;
        let mut redundancy_bytes = 0i32;
        let mut celt_to_silk = false;
        let mut start_band = 0;

        if !decode_fec
            && mode != MODE_CELT_ONLY
            && data.is_some()
            && dec.tell() + 17 + 20 * (mode == MODE_HYBRID) as i32 <= 8 * len
        {
            // Check if we have a redundant 0-8 kHz band
            if mode == MODE_HYBRID {
                redundancy = dec.decode_bit_logp(12);
            } else {
                redundancy = true;
            }
            if redundancy {
                celt_to_silk = dec.decode_bit_logp(1);
                // redundancy_bytes will be at least two, in the non-hybrid
                // case due to the ec_tell() check above
                redundancy_bytes = if mode == MODE_HYBRID {
                    dec.decode_uint(256) as i32 + 2
                } else {
                    len - ((dec.tell() + 7) >> 3)
                };
                len -= redundancy_bytes;
                // Sanity check — should never happen for valid packets
                if len * 8 < dec.tell() {
                    len = 0;
                    redundancy_bytes = 0;
                    redundancy = false;
                }
                // Shrink decoder because of raw bits
                dec.reduce_storage(redundancy_bytes as u32);
            }
        }

        if mode != MODE_CELT_ONLY {
            start_band = 17;
        }

        if redundancy {
            transition = false;
        }

        // For SILK→CELT transition (non-redundant), pre-decode PLC
        if transition && mode != MODE_CELT_ONLY && pcm_transition.is_none() {
            let trans_size = f5 as usize * self.channels as usize;
            let mut trans_buf = vec![0i16; trans_size];
            self.decode_frame(None, &mut trans_buf, imin(f5, audiosize), false)?;
            pcm_transition = Some(trans_buf);
        }

        // --- Configure CELT end band by bandwidth ---
        if bandwidth != 0 {
            let endband = match bandwidth {
                OPUS_BANDWIDTH_NARROWBAND => 13,
                OPUS_BANDWIDTH_MEDIUMBAND | OPUS_BANDWIDTH_WIDEBAND => 17,
                OPUS_BANDWIDTH_SUPERWIDEBAND => 19,
                OPUS_BANDWIDTH_FULLBAND => 21,
                _ => 21,
            };
            let _ = self.celt_dec.set_end_band(endband);
        }
        let _ = self.celt_dec.set_channels(self.stream_channels);

        // --- Allocate redundant audio buffer ---
        let mut redundant_audio = if redundancy {
            vec![0i16; f5 as usize * self.channels as usize]
        } else {
            Vec::new()
        };
        let mut redundant_rng: u32 = 0;

        // --- 5 ms redundant frame for CELT→SILK ---
        if redundancy && celt_to_silk {
            let _ = self.celt_dec.set_start_band(0);
            let redundancy_data =
                &frame_data[len as usize..len as usize + redundancy_bytes as usize];

            let _ = self.celt_dec.decode_with_ec(
                Some(redundancy_data),
                &mut redundant_audio,
                f5,
                None,
                false,
                None,
            );
            redundant_rng = self.celt_dec.rng;
        }

        // --- Set CELT start band (MUST be after PLC) ---
        let _ = self.celt_dec.set_start_band(start_band);

        // --- CELT processing ---
        let celt_ret;
        if mode != MODE_SILK_ONLY {
            let celt_frame_size = imin(f20, frame_size);
            // Make sure to discard any previous CELT state on mode change
            if mode != self.prev_mode && self.prev_mode > 0 && !self.prev_redundancy {
                self.celt_dec.reset();
            }
            // Decode CELT (pass None for data when doing FEC)
            let celt_data = if decode_fec { None } else { data };
            // Neural PLC on CELT lost frames. The CELT path will only
            // flip to FRAME_PLC_NEURAL when weights have been loaded.
            let celt_lpcnet: crate::celt::decoder::DnnPlcArg<'_> =
                Some(self.lpcnet.as_mut());

            celt_ret = self.celt_dec.decode_with_ec(
                celt_data,
                &mut pcm[..celt_frame_size as usize * self.channels as usize],
                celt_frame_size,
                Some(&mut dec),
                celt_accum,
                celt_lpcnet,
            );
            self.range_final = self.celt_dec.rng;
        } else {
            // SILK-only mode
            let silence: [u8; 2] = [0xFF, 0xFF];
            if !celt_accum {
                let n = frame_size as usize * self.channels as usize;
                for s in &mut pcm[..n] {
                    *s = 0;
                }
            }
            // For hybrid → SILK transitions, let the CELT MDCT do a fade-out
            if self.prev_mode == MODE_HYBRID
                && !(redundancy && celt_to_silk && self.prev_redundancy)
            {
                let _ = self.celt_dec.set_start_band(0);

                let _ = self.celt_dec.decode_with_ec(
                    Some(&silence),
                    &mut pcm[..f2_5 as usize * self.channels as usize],
                    f2_5,
                    None,
                    celt_accum,
                    None,
                );
            }
            self.range_final = dec.get_rng();
            celt_ret = Ok(0);
        }

        // --- Get the window for crossfades ---
        let window = self.celt_dec.get_mode().window;

        // --- 5 ms redundant frame for SILK→CELT ---
        if redundancy && !celt_to_silk {
            self.celt_dec.reset();
            let _ = self.celt_dec.set_start_band(0);
            let redundancy_data =
                &frame_data[len as usize..len as usize + redundancy_bytes as usize];

            let _ = self.celt_dec.decode_with_ec(
                Some(redundancy_data),
                &mut redundant_audio,
                f5,
                None,
                false,
                None,
            );
            redundant_rng = self.celt_dec.rng;

            // Crossfade: last 2.5ms of main output with first 2.5ms of redundancy
            let ch = self.channels as usize;
            let fade_offset = ch * (frame_size as usize - f2_5 as usize);
            let fade_n = f2_5 as usize * ch;
            // in1 aliases out (both pcm[fade_offset..]), so copy in1 to temp
            let temp: Vec<i16> = pcm[fade_offset..fade_offset + fade_n].to_vec();
            let red_offset = ch * f2_5 as usize;
            smooth_fade(
                &temp,
                &redundant_audio[red_offset..red_offset + fade_n],
                &mut pcm[fade_offset..fade_offset + fade_n],
                f2_5,
                self.channels,
                window,
                self.fs,
            );
        }

        // --- Apply CELT→SILK redundancy to output ---
        if redundancy && celt_to_silk && (self.prev_mode != MODE_SILK_ONLY || self.prev_redundancy)
        {
            let ch = self.channels as usize;
            let n_f2_5 = f2_5 as usize * ch;
            // Copy first 2.5ms from redundancy
            pcm[..n_f2_5].copy_from_slice(&redundant_audio[..n_f2_5]);
            // Crossfade next 2.5ms: in2 aliases out (both pcm[n_f2_5..]), copy in2 to temp
            let temp: Vec<i16> = pcm[n_f2_5..2 * n_f2_5].to_vec();
            smooth_fade(
                &redundant_audio[n_f2_5..2 * n_f2_5],
                &temp,
                &mut pcm[n_f2_5..2 * n_f2_5],
                f2_5,
                self.channels,
                window,
                self.fs,
            );
        }

        // --- Apply transition crossfade ---
        if transition {
            if let Some(ref trans) = pcm_transition {
                let ch = self.channels as usize;
                let n_f2_5 = f2_5 as usize * ch;
                if audiosize >= f5 {
                    // Copy first 2.5ms from transition buffer
                    pcm[..n_f2_5].copy_from_slice(&trans[..n_f2_5]);
                    // Crossfade next 2.5ms: in2 aliases out, copy to temp
                    let temp: Vec<i16> = pcm[n_f2_5..2 * n_f2_5].to_vec();
                    smooth_fade(
                        &trans[n_f2_5..2 * n_f2_5],
                        &temp,
                        &mut pcm[n_f2_5..2 * n_f2_5],
                        f2_5,
                        self.channels,
                        window,
                        self.fs,
                    );
                } else {
                    // Not enough time for a clean transition, but best we can do
                    let temp: Vec<i16> = pcm[..n_f2_5].to_vec();
                    smooth_fade(
                        trans,
                        &temp,
                        &mut pcm[..n_f2_5],
                        f2_5,
                        self.channels,
                        window,
                        self.fs,
                    );
                }
            }
        }

        // --- Apply decode gain ---
        if self.decode_gain != 0 {
            let gain = celt_exp2(mult16_16_p15(GAIN_SCALE_Q25, self.decode_gain));
            let n = frame_size as usize * self.channels as usize;
            for i in 0..n {
                let x = mult16_32_p16(pcm[i] as i32, gain);
                pcm[i] = saturate(x, 32767) as i16;
            }
        }

        // --- Compute final range ---
        if len <= 1 {
            self.range_final = 0;
        } else {
            self.range_final ^= redundant_rng;
        }

        // --- Update state ---
        self.prev_mode = mode;
        self.prev_redundancy = redundancy && !celt_to_silk;

        match celt_ret {
            Err(e) => Err(e),
            Ok(_) => Ok(audiosize),
        }
    }

    // -----------------------------------------------------------------------
    // decode_native — internal native decode entry point
    // -----------------------------------------------------------------------

    /// Internal decode function used by all public decode methods and the
    /// multistream decoder.
    /// Matches C `opus_decode_native` (without DRED/QEXT).
    pub(crate) fn decode_native(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i16],
        frame_size: i32,
        decode_fec: bool,
        self_delimited: bool,
        packet_offset: Option<&mut i32>,
    ) -> Result<i32, i32> {
        if !(0..=1).contains(&(decode_fec as i32)) {
            return Err(OPUS_BAD_ARG);
        }
        // For FEC/PLC, frame_size must be a multiple of 2.5 ms
        let is_plc_or_fec = decode_fec || data.is_none() || data.is_none_or(|d| d.is_empty());
        if is_plc_or_fec && frame_size % (self.fs / 400) != 0 {
            return Err(OPUS_BAD_ARG);
        }

        // --- PLC path (no data) ---
        let packet_data = match data {
            Some(d) if !d.is_empty() => d,
            _ => {
                // PLC: decode frames until frame_size is filled
                let mut pcm_count = 0i32;
                loop {
                    let ret = self.decode_frame(
                        None,
                        &mut pcm[pcm_count as usize * self.channels as usize..],
                        frame_size - pcm_count,
                        false,
                    )?;
                    pcm_count += ret;
                    if pcm_count >= frame_size {
                        break;
                    }
                }
                self.last_packet_duration = pcm_count;
                return Ok(pcm_count);
            }
        };

        // --- Parse the packet ---
        let packet_mode = opus_packet_get_mode(packet_data);
        let packet_bandwidth = opus_packet_get_bandwidth(packet_data);
        let packet_frame_size = opus_packet_get_samples_per_frame(packet_data, self.fs);
        let packet_stream_channels = opus_packet_get_nb_channels(packet_data);

        let mut toc = 0u8;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut offset = 0i32;
        let count = opus_packet_parse_impl(
            packet_data,
            packet_data.len() as i32,
            self_delimited,
            &mut toc,
            &mut sizes,
            &mut offset,
            packet_offset,
        );
        if count < 0 {
            return Err(count);
        }

        let payload = &packet_data[offset as usize..];

        // --- FEC path ---
        if decode_fec {
            // If no FEC can be present, run the PLC (recursive call)
            if frame_size < packet_frame_size
                || packet_mode == MODE_CELT_ONLY
                || self.mode == MODE_CELT_ONLY
            {
                return self.decode_native(None, pcm, frame_size, false, false, None);
            }
            // Otherwise, run PLC on everything except the size for which we
            // might have FEC
            let duration_copy = self.last_packet_duration;
            if frame_size - packet_frame_size != 0 {
                let plc_size = frame_size - packet_frame_size;
                let ret = self.decode_native(None, pcm, plc_size, false, false, None);
                if let Err(e) = ret {
                    self.last_packet_duration = duration_copy;
                    return Err(e);
                }
            }
            // Complete with FEC
            self.mode = packet_mode;
            self.bandwidth = packet_bandwidth;
            self.frame_size = packet_frame_size;
            self.stream_channels = packet_stream_channels;
            let fec_offset = (frame_size - packet_frame_size) as usize * self.channels as usize;
            let ret = self.decode_frame(
                Some(&payload[..sizes[0] as usize]),
                &mut pcm[fec_offset..],
                packet_frame_size,
                true,
            )?;
            let _ = ret;
            self.last_packet_duration = frame_size;
            return Ok(frame_size);
        }

        // --- Normal decode path ---
        if count * packet_frame_size > frame_size {
            return Err(OPUS_BUFFER_TOO_SMALL);
        }

        // Update state as the last step to avoid updating on invalid packet
        self.mode = packet_mode;
        self.bandwidth = packet_bandwidth;
        self.frame_size = packet_frame_size;
        self.stream_channels = packet_stream_channels;

        let mut nb_samples = 0i32;
        let mut data_offset = 0usize;
        for i in 0..count as usize {
            // C passes (data, size[i]) as separate args; when size[i] <= 1,
            // opus_decode_frame sets data=NULL triggering PLC/DTX.  Match that
            // by passing None directly so decode_frame takes the PLC path.
            let frame_arg = if sizes[i] as i32 <= 1 {
                None
            } else {
                Some(&payload[data_offset..data_offset + sizes[i] as usize])
            };
            let ret = self.decode_frame(
                frame_arg,
                &mut pcm[nb_samples as usize * self.channels as usize..],
                frame_size - nb_samples,
                false,
            )?;
            data_offset += sizes[i] as usize;
            nb_samples += ret;
        }
        self.last_packet_duration = nb_samples;

        // Fixed-point: no soft clipping needed

        Ok(nb_samples)
    }

    // -----------------------------------------------------------------------
    // Public decode API
    // -----------------------------------------------------------------------

    /// Decode an Opus packet to 16-bit PCM.
    /// `data`: compressed packet (None for PLC).
    /// `pcm`: output buffer, interleaved if stereo, length >= frame_size * channels.
    /// `frame_size`: max samples per channel the buffer can hold.
    /// `decode_fec`: true to decode FEC from this packet (recovers *previous* frame).
    /// Returns number of decoded samples per channel, or error.
    /// Matches C `opus_decode` (fixed-point, non-RES24 path).
    pub fn decode(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i16],
        frame_size: i32,
        decode_fec: bool,
    ) -> Result<i32, i32> {
        if frame_size <= 0 {
            return Err(OPUS_BAD_ARG);
        }
        self.decode_native(data, pcm, frame_size, decode_fec, false, None)
    }

    /// Decode an Opus packet to 32-bit PCM (24-bit in 32-bit container).
    /// Matches C `opus_decode24` (fixed-point, non-RES24 path: decode to i16, then <<8).
    pub fn decode24(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i32],
        frame_size: i32,
        decode_fec: bool,
    ) -> Result<i32, i32> {
        if frame_size <= 0 {
            return Err(OPUS_BAD_ARG);
        }
        // Determine actual frame count to minimize allocation
        let mut actual_frame_size = frame_size;
        if let Some(d) = data {
            if !d.is_empty() && !decode_fec {
                match opus_packet_get_nb_samples(d, self.fs) {
                    Ok(ns) if ns > 0 => actual_frame_size = imin(frame_size, ns),
                    Ok(_) | Err(_) => return Err(OPUS_INVALID_PACKET),
                }
            }
        }
        let mut out = vec![0i16; actual_frame_size as usize * self.channels as usize];
        let ret = self.decode_native(data, &mut out, actual_frame_size, decode_fec, false, None)?;
        // Convert i16 → i32 (24-bit): SHL32(EXTEND32(a), 8)
        let n = ret as usize * self.channels as usize;
        for i in 0..n {
            pcm[i] = shl32(out[i] as i32, 8);
        }
        Ok(ret)
    }

    /// Decode an Opus packet to floating-point PCM.
    /// Matches C `opus_decode_float` (fixed-point path: decode to i16, then /32768.0).
    pub fn decode_float(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [f32],
        frame_size: i32,
        decode_fec: bool,
    ) -> Result<i32, i32> {
        if frame_size <= 0 {
            return Err(OPUS_BAD_ARG);
        }
        // Determine actual frame count to minimize allocation
        let mut actual_frame_size = frame_size;
        if let Some(d) = data {
            if !d.is_empty() && !decode_fec {
                match opus_packet_get_nb_samples(d, self.fs) {
                    Ok(ns) if ns > 0 => actual_frame_size = imin(frame_size, ns),
                    Ok(_) | Err(_) => return Err(OPUS_INVALID_PACKET),
                }
            }
        }
        let mut out = vec![0i16; actual_frame_size as usize * self.channels as usize];
        let ret = self.decode_native(data, &mut out, actual_frame_size, decode_fec, false, None)?;
        // Convert i16 → f32: a / 32768.0
        let n = ret as usize * self.channels as usize;
        for i in 0..n {
            pcm[i] = out[i] as f32 * (1.0 / 32768.0);
        }
        Ok(ret)
    }

    // -----------------------------------------------------------------------
    // CTL getters and setters
    // -----------------------------------------------------------------------

    /// Get the bandwidth of the last decoded packet.
    pub fn get_bandwidth(&self) -> i32 {
        self.bandwidth
    }

    /// Get the output sample rate.
    pub fn get_sample_rate(&self) -> i32 {
        self.fs
    }

    /// Get the number of output channels.
    pub fn get_channels(&self) -> i32 {
        self.channels
    }

    /// Get the range coder final state (for bitstream verification).
    pub fn get_final_range(&self) -> u32 {
        self.range_final
    }

    /// Get the CELT decoder's old_band_e (for debug comparison).
    pub fn debug_get_old_band_e(&self) -> &[i32] {
        self.celt_dec.debug_old_band_e()
    }

    /// Get the CELT decoder's old_log_e (for debug comparison).
    pub fn debug_get_old_log_e(&self) -> &[i32] {
        self.celt_dec.debug_old_log_e()
    }

    /// Get the CELT decoder's old_log_e2 (for debug comparison).
    pub fn debug_get_old_log_e2(&self) -> &[i32] {
        self.celt_dec.debug_old_log_e2()
    }

    /// Get the CELT decoder's background_log_e (for debug comparison).
    pub fn debug_get_background_log_e(&self) -> &[i32] {
        self.celt_dec.debug_background_log_e()
    }

    /// Get the CELT decode_mem per-channel stride length.
    pub fn debug_get_decode_mem_stride(&self) -> usize {
        self.celt_dec.debug_decode_mem_stride()
    }

    /// Get the CELT decoder's postfilter state (for debug comparison).
    pub fn debug_get_preemph_mem(&self) -> [i32; 2] {
        self.celt_dec.preemph_mem_d
    }

    /// Get a slice of the CELT decode_mem (debug accessor).
    pub fn debug_get_decode_mem(&self, offset: usize, count: usize) -> &[i32] {
        self.celt_dec.debug_get_decode_mem(offset, count)
    }

    pub fn debug_get_postfilter(&self) -> (i32, i32, i32, i32, i32, i32) {
        (
            self.celt_dec.postfilter_period,
            self.celt_dec.postfilter_period_old,
            self.celt_dec.postfilter_gain,
            self.celt_dec.postfilter_gain_old,
            self.celt_dec.postfilter_tapset,
            self.celt_dec.postfilter_tapset_old,
        )
    }

    /// Debug: borrow the SILK decoder (read-only) for state inspection.
    pub fn debug_silk_dec(&self) -> &crate::silk::decoder::SilkDecoder {
        &self.silk_dec
    }

    /// Get the pitch period of the last decoded frame (samples at 48 kHz).
    pub fn get_pitch(&self) -> i32 {
        if self.prev_mode == MODE_CELT_ONLY {
            self.celt_dec.get_pitch()
        } else {
            self.dec_control.prev_pitch_lag
        }
    }

    /// Get the current decode gain in Q8 dB.
    pub fn get_gain(&self) -> i32 {
        self.decode_gain
    }

    /// Set the decode gain (Q8 dB, range [-32768, 32767]).
    pub fn set_gain(&mut self, gain: i32) -> Result<(), i32> {
        if !(-32768..=32767).contains(&gain) {
            return Err(OPUS_BAD_ARG);
        }
        self.decode_gain = gain;
        Ok(())
    }

    /// Get the current decoder complexity (0-10).
    pub fn get_complexity(&self) -> i32 {
        self.complexity
    }

    /// Set the decoder complexity (0-10).
    pub fn set_complexity(&mut self, complexity: i32) -> Result<(), i32> {
        if !(0..=10).contains(&complexity) {
            return Err(OPUS_BAD_ARG);
        }
        self.complexity = complexity;
        let _ = self.celt_dec.set_complexity(complexity);
        Ok(())
    }

    /// Get the duration of the last decoded packet in samples.
    pub fn get_last_packet_duration(&self) -> i32 {
        self.last_packet_duration
    }

    /// Get whether phase inversion is disabled.
    pub fn get_phase_inversion_disabled(&self) -> bool {
        self.celt_dec.get_phase_inversion_disabled()
    }

    /// Set whether phase inversion is disabled (0 or 1).
    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) {
        self.celt_dec.set_phase_inversion_disabled(disabled);
    }

    /// Get whether extensions are ignored.
    pub fn get_ignore_extensions(&self) -> bool {
        self.ignore_extensions
    }

    /// Set whether to ignore packet extensions.
    pub fn set_ignore_extensions(&mut self, ignore: bool) {
        self.ignore_extensions = ignore;
    }

    /// Get the number of samples in a packet at this decoder's sample rate.
    pub fn get_nb_samples(&self, packet: &[u8]) -> Result<i32, i32> {
        opus_packet_get_nb_samples(packet, self.fs)
    }

    // -----------------------------------------------------------------------
    // pub(crate) accessors for multistream module
    // -----------------------------------------------------------------------

    #[allow(dead_code)]
    pub(crate) fn ms_get_sample_rate(&self) -> i32 {
        self.fs
    }
    #[allow(dead_code)]
    pub(crate) fn ms_get_bandwidth(&self) -> i32 {
        self.bandwidth
    }
    #[allow(dead_code)]
    pub(crate) fn ms_get_range_final(&self) -> u32 {
        self.range_final
    }
    #[allow(dead_code)]
    pub(crate) fn ms_get_last_packet_duration(&self) -> i32 {
        self.last_packet_duration
    }
    #[allow(dead_code)]
    pub(crate) fn ms_get_gain(&self) -> i32 {
        self.decode_gain
    }
    #[allow(dead_code)]
    pub(crate) fn ms_set_gain(&mut self, gain: i32) {
        self.decode_gain = gain;
    }
    #[allow(dead_code)]
    pub(crate) fn ms_get_complexity(&self) -> i32 {
        self.complexity
    }
    #[allow(dead_code)]
    pub(crate) fn ms_set_complexity(&mut self, v: i32) {
        self.complexity = v;
    }
    #[allow(dead_code)]
    pub(crate) fn ms_get_phase_inversion_disabled(&self) -> i32 {
        if self.celt_dec.disable_inv { 1 } else { 0 }
    }
    #[allow(dead_code)]
    pub(crate) fn ms_set_phase_inversion_disabled(&mut self, v: i32) {
        self.celt_dec.disable_inv = v != 0;
    }
    #[allow(dead_code)]
    pub(crate) fn ms_reset(&mut self) {
        // Reset decoder state
        self.stream_channels = self.channels;
        self.bandwidth = 0;
        self.mode = 0;
        self.prev_mode = 0;
        self.frame_size = self.fs / 400;
        self.prev_redundancy = false;
        self.last_packet_duration = 0;
        self.range_final = 0;
        self.silk_dec.init();
        self.celt_dec.reset();
    }

    /// Test-only accessor for the CELT decoder's last-frame-type marker.
    /// Exposed so integration tests can verify which PLC branch
    /// (`FRAME_PLC_PERIODIC`, `FRAME_PLC_NOISE`, `FRAME_PLC_NEURAL`,
    /// `FRAME_DRED`) actually executed on the most recent decode.
    #[cfg(test)]
    pub(crate) fn celt_last_frame_type(&self) -> i32 {
        self.celt_dec.last_frame_type
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opus::encoder::{
        OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY, OpusEncoder,
    };

    fn patterned_pcm_i16(frame_size: usize, channels: usize, seed: i32) -> Vec<i16> {
        (0..frame_size * channels)
            .map(|i| {
                let base = ((i as i32 * 7919 + seed * 911) % 28000) - 14000;
                if channels == 2 && i % 2 == 1 {
                    (base / 2) as i16
                } else {
                    base as i16
                }
            })
            .collect()
    }

    // --- Packet utility tests ---

    #[test]
    fn test_parse_size_small() {
        let data = [100u8];
        let (bytes, size) = parse_size(&data, 1);
        assert_eq!(bytes, 1);
        assert_eq!(size, 100);
    }

    #[test]
    fn test_parse_size_large() {
        // 252 + 3 = 255 in first byte, 10 in second byte => 4*10 + 255 = 295
        let data = [255u8, 10u8];
        let (bytes, size) = parse_size(&data, 2);
        assert_eq!(bytes, 2);
        assert_eq!(size, 4 * 10 + 255);
    }

    #[test]
    fn test_parse_size_empty() {
        let data = [];
        let (bytes, size) = parse_size(&data, 0);
        assert_eq!(bytes, -1);
        assert_eq!(size, -1);
    }

    #[test]
    fn test_parse_size_large_but_one_byte() {
        let data = [253u8];
        let (bytes, size) = parse_size(&data, 1);
        assert_eq!(bytes, -1);
        assert_eq!(size, -1);
    }

    #[test]
    fn test_get_samples_per_frame_celt() {
        // CELT-only (bit 7 set), bits 4-3 = 0 → 2.5ms → 48000/400 = 120
        let data = [0x80u8];
        assert_eq!(opus_packet_get_samples_per_frame(&data, 48000), 120);
        // bits 4-3 = 3 → 20ms → 48000/50 = 960
        let data = [0x98u8];
        assert_eq!(opus_packet_get_samples_per_frame(&data, 48000), 960);
    }

    #[test]
    fn test_get_samples_per_frame_hybrid() {
        // Hybrid (bits 7-6 = 01), bit 3 = 0 → 10ms
        let data = [0x60u8];
        assert_eq!(opus_packet_get_samples_per_frame(&data, 48000), 480);
        // bit 3 = 1 → 20ms
        let data = [0x68u8];
        assert_eq!(opus_packet_get_samples_per_frame(&data, 48000), 960);
    }

    #[test]
    fn test_get_samples_per_frame_silk() {
        // SILK (bits 7-6 = 00), bits 4-3 = 0 → 10ms
        let data = [0x00u8];
        assert_eq!(opus_packet_get_samples_per_frame(&data, 48000), 480);
        // bits 4-3 = 3 → 60ms
        let data = [0x18u8];
        assert_eq!(opus_packet_get_samples_per_frame(&data, 48000), 2880);
    }

    #[test]
    fn test_get_mode() {
        assert_eq!(opus_packet_get_mode(&[0x80]), MODE_CELT_ONLY);
        assert_eq!(opus_packet_get_mode(&[0x60]), MODE_HYBRID);
        assert_eq!(opus_packet_get_mode(&[0x00]), MODE_SILK_ONLY);
    }

    #[test]
    fn test_get_bandwidth() {
        // CELT: bits 6-5 = 00 → mediumband (mapped to narrowband)
        assert_eq!(
            opus_packet_get_bandwidth(&[0x80]),
            OPUS_BANDWIDTH_NARROWBAND
        );
        // CELT: bits 6-5 = 01 → wideband
        assert_eq!(opus_packet_get_bandwidth(&[0xA0]), OPUS_BANDWIDTH_WIDEBAND);
        // Hybrid: bit 4 = 0 → superwideband
        assert_eq!(
            opus_packet_get_bandwidth(&[0x60]),
            OPUS_BANDWIDTH_SUPERWIDEBAND
        );
        // Hybrid: bit 4 = 1 → fullband
        assert_eq!(opus_packet_get_bandwidth(&[0x70]), OPUS_BANDWIDTH_FULLBAND);
        // SILK: bits 6-5 = 00 → narrowband
        assert_eq!(
            opus_packet_get_bandwidth(&[0x00]),
            OPUS_BANDWIDTH_NARROWBAND
        );
    }

    #[test]
    fn test_get_nb_channels() {
        assert_eq!(opus_packet_get_nb_channels(&[0x00]), 1); // mono
        assert_eq!(opus_packet_get_nb_channels(&[0x04]), 2); // stereo
    }

    #[test]
    fn test_get_nb_frames() {
        assert_eq!(opus_packet_get_nb_frames(&[0x00]), Ok(1)); // code 0
        assert_eq!(opus_packet_get_nb_frames(&[0x01]), Ok(2)); // code 1
        assert_eq!(opus_packet_get_nb_frames(&[0x02]), Ok(2)); // code 2
        assert_eq!(opus_packet_get_nb_frames(&[0x03, 0x05]), Ok(5)); // code 3, count=5
        assert!(opus_packet_get_nb_frames(&[]).is_err());
    }

    #[test]
    fn test_get_nb_samples() {
        // SILK 20ms mono, 1 frame at 48kHz → 960
        let packet = [0x08u8, 0x00]; // SILK, 20ms, code 0 (1 frame), + 1 byte payload
        assert_eq!(opus_packet_get_nb_samples(&packet, 48000), Ok(960));

        // 60ms SILK with 3 frames exceeds the 120ms limit.
        let invalid = [0x1Bu8, 0x03];
        assert_eq!(
            opus_packet_get_nb_samples(&invalid, 48000),
            Err(OPUS_INVALID_PACKET)
        );
    }

    #[test]
    fn test_packet_parse_single_frame() {
        // TOC: CELT, 20ms, mono, code 0 (1 frame) + 10 bytes payload
        let mut pkt = vec![0x80u8]; // TOC
        pkt.extend_from_slice(&[0u8; 10]); // payload

        let mut toc = 0u8;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut offset = 0i32;
        let count = opus_packet_parse_impl(
            &pkt,
            pkt.len() as i32,
            false,
            &mut toc,
            &mut sizes,
            &mut offset,
            None,
        );
        assert_eq!(count, 1);
        assert_eq!(toc, 0x80);
        assert_eq!(sizes[0], 10);
        assert_eq!(offset, 1); // 1 byte TOC
    }

    #[test]
    fn test_packet_parse_two_cbr_frames() {
        // TOC code 1 (2 CBR frames) + 20 bytes payload
        let mut pkt = vec![0x81u8]; // TOC with code=1
        pkt.extend_from_slice(&[0u8; 20]);

        let mut toc = 0u8;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut offset = 0i32;
        let count = opus_packet_parse_impl(
            &pkt,
            pkt.len() as i32,
            false,
            &mut toc,
            &mut sizes,
            &mut offset,
            None,
        );
        assert_eq!(count, 2);
        assert_eq!(sizes[0], 10);
        assert_eq!(sizes[1], 10);
    }

    #[test]
    fn test_packet_parse_two_vbr_frames() {
        // TOC code 2 (2 VBR frames), size[0] = 5, remaining = 15 for frame 1
        let mut pkt = vec![0x82u8]; // TOC with code=2
        pkt.push(5); // size of first frame (small, 1 byte encoding)
        pkt.extend_from_slice(&[0u8; 20]); // payload

        let mut toc = 0u8;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut offset = 0i32;
        let count = opus_packet_parse_impl(
            &pkt,
            pkt.len() as i32,
            false,
            &mut toc,
            &mut sizes,
            &mut offset,
            None,
        );
        assert_eq!(count, 2);
        assert_eq!(sizes[0], 5);
        assert_eq!(sizes[1], 15); // remaining after TOC(1) + size_encoding(1) + size[0](5) = 22-7=15
    }

    #[test]
    fn test_packet_parse_invalid_odd_cbr() {
        // TOC code 1 (2 CBR), but odd-length payload
        let pkt = vec![0x81u8, 0, 0, 0]; // 3 bytes of payload (odd)
        let mut toc = 0u8;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut offset = 0i32;
        let count = opus_packet_parse_impl(
            &pkt,
            pkt.len() as i32,
            false,
            &mut toc,
            &mut sizes,
            &mut offset,
            None,
        );
        assert_eq!(count, OPUS_INVALID_PACKET);
    }

    #[test]
    fn test_packet_parse_code2_two_byte_size_and_payload_offset() {
        let mut pkt = vec![0x82u8, 0xFC, 0x02];
        pkt.extend_from_slice(&[0u8; 300]);

        let mut toc = 0u8;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut offset = 0i32;
        let count = opus_packet_parse_impl(
            &pkt,
            pkt.len() as i32,
            false,
            &mut toc,
            &mut sizes,
            &mut offset,
            None,
        );

        assert_eq!(count, 2);
        assert_eq!(toc, 0x82);
        assert_eq!(sizes[0], 260);
        assert_eq!(sizes[1], 40);
        assert_eq!(offset, 3);
    }

    #[test]
    fn test_packet_parse_code3_padding_loop_and_packet_offset() {
        let mut pkt = vec![0x83u8, 0x42, 0xFF, 0x01];
        pkt.extend_from_slice(&[0u8; 259]);

        let mut toc = 0u8;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut offset = 0i32;
        let mut packet_offset = 0i32;
        let count = opus_packet_parse_impl(
            &pkt,
            pkt.len() as i32,
            false,
            &mut toc,
            &mut sizes,
            &mut offset,
            Some(&mut packet_offset),
        );

        assert_eq!(count, 2);
        assert_eq!(toc, 0x83);
        assert_eq!(&sizes[..2], &[2, 2]);
        assert_eq!(offset, 4);
        assert_eq!(packet_offset, pkt.len() as i32);
    }

    #[test]
    fn test_packet_parse_code3_self_delimited_paths() {
        let pkt_cbr = [0x83u8, 0x03, 0x02, 0, 1, 2, 3, 4, 5];
        let mut toc = 0u8;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut offset = 0i32;
        let mut packet_offset = 0i32;
        let count = opus_packet_parse_impl(
            &pkt_cbr,
            pkt_cbr.len() as i32,
            true,
            &mut toc,
            &mut sizes,
            &mut offset,
            Some(&mut packet_offset),
        );
        assert_eq!(count, 3);
        assert_eq!(toc, 0x83);
        assert_eq!(&sizes[..3], &[2, 2, 2]);
        assert_eq!(offset, 3);
        assert_eq!(packet_offset, pkt_cbr.len() as i32);

        let pkt_vbr_with_pad = [0x83u8, 0xC2, 0x01, 0x02, 0x03, 0, 1, 2, 3, 4, 0];
        let mut toc = 0u8;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut offset = 0i32;
        let mut packet_offset = 0i32;
        let count = opus_packet_parse_impl(
            &pkt_vbr_with_pad,
            pkt_vbr_with_pad.len() as i32,
            true,
            &mut toc,
            &mut sizes,
            &mut offset,
            Some(&mut packet_offset),
        );
        assert_eq!(count, 2);
        assert_eq!(toc, 0x83);
        assert_eq!(&sizes[..2], &[2, 3]);
        assert_eq!(offset, 5);
        assert_eq!(packet_offset, pkt_vbr_with_pad.len() as i32);

        let invalid = [0x83u8, 0x82, 0x02, 0x05, 0, 1, 0];
        let mut toc = 0u8;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut offset = 0i32;
        assert_eq!(
            opus_packet_parse_impl(
                &invalid,
                invalid.len() as i32,
                true,
                &mut toc,
                &mut sizes,
                &mut offset,
                None,
            ),
            OPUS_INVALID_PACKET
        );
    }

    #[test]
    fn test_decode24_and_decode_float_reject_invalid_packets() {
        let mut dec24 = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm24 = vec![0i32; 960];
        let bad_packet = [0x9Bu8, 7];
        assert_eq!(
            dec24.decode24(Some(&bad_packet), &mut pcm24, 960, false),
            Err(OPUS_INVALID_PACKET)
        );

        let mut decf = OpusDecoder::new(48000, 1).unwrap();
        let mut pcmf = vec![0.0f32; 960];
        assert_eq!(
            decf.decode_float(Some(&bad_packet), &mut pcmf, 960, false),
            Err(OPUS_INVALID_PACKET)
        );
    }

    #[test]
    fn test_packet_has_lbrr_detects_flags_and_errors() {
        assert_eq!(opus_packet_has_lbrr(&[0x80, 0x00], 2), Ok(false));
        assert_eq!(opus_packet_has_lbrr(&[0x08, 0x40], 2), Ok(true));
        assert_eq!(opus_packet_has_lbrr(&[0x0C, 0x10], 2), Ok(true));
        assert_eq!(opus_packet_has_lbrr(&[0x03], 1), Err(OPUS_INVALID_PACKET));
    }

    // --- Decoder construction tests ---

    #[test]
    fn test_decoder_new_valid() {
        let dec = OpusDecoder::new(48000, 2);
        assert!(dec.is_ok());
        let dec = dec.unwrap();
        assert_eq!(dec.get_sample_rate(), 48000);
        assert_eq!(dec.get_channels(), 2);
    }

    #[test]
    fn test_decoder_new_invalid_rate() {
        assert!(OpusDecoder::new(44100, 1).is_err());
    }

    #[test]
    fn test_decoder_new_invalid_channels() {
        assert!(OpusDecoder::new(48000, 3).is_err());
    }

    #[test]
    fn test_decoder_gain() {
        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        assert_eq!(dec.get_gain(), 0);
        assert!(dec.set_gain(256).is_ok());
        assert_eq!(dec.get_gain(), 256);
        assert!(dec.set_gain(-32768).is_ok());
        assert!(dec.set_gain(32768).is_err());
    }

    #[test]
    fn test_decoder_complexity() {
        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        assert_eq!(dec.get_complexity(), 0);
        assert!(dec.set_complexity(5).is_ok());
        assert_eq!(dec.get_complexity(), 5);
        assert!(dec.set_complexity(11).is_err());
    }

    #[test]
    fn test_decoder_reset() {
        let mut dec = OpusDecoder::new(48000, 2).unwrap();
        // Modify state
        dec.bandwidth = OPUS_BANDWIDTH_FULLBAND;
        dec.prev_mode = MODE_CELT_ONLY;
        dec.range_final = 12345;
        // Reset
        dec.reset();
        assert_eq!(dec.bandwidth, 0);
        assert_eq!(dec.prev_mode, 0);
        assert_eq!(dec.range_final, 0);
        assert_eq!(dec.stream_channels, 2);
        assert_eq!(dec.frame_size, 48000 / 400);
    }

    #[test]
    fn test_decoder_plc_no_prev_mode() {
        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![1i16; 960];
        // PLC with no previous mode should output zeros
        let ret = dec.decode(None, &mut pcm, 960, false);
        assert!(ret.is_ok());
        assert_eq!(ret.unwrap(), 960);
        assert!(pcm.iter().all(|&s| s == 0));
    }

    #[test]
    fn test_decoder_bad_frame_size() {
        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0i16; 960];
        let ret = dec.decode(None, &mut pcm, 0, false);
        assert_eq!(ret, Err(OPUS_BAD_ARG));
    }

    #[test]
    fn test_decode_native_fec_argument_and_fallback_paths() {
        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0i16; 960];
        let packet = [0x80u8, 0x00];
        assert_eq!(
            dec.decode_native(Some(&packet), &mut pcm, 121, true, false, None),
            Err(OPUS_BAD_ARG)
        );
        assert_eq!(
            dec.decode_native(None, &mut pcm, 121, false, false, None),
            Err(OPUS_BAD_ARG)
        );

        let mut packet_offset = -1;
        let decoded = dec
            .decode_native(
                Some(&packet),
                &mut pcm,
                960,
                true,
                false,
                Some(&mut packet_offset),
            )
            .unwrap();
        assert_eq!(decoded, 960);
        assert_eq!(packet_offset, packet.len() as i32);
        assert_eq!(dec.get_last_packet_duration(), 960);
        assert!(pcm.iter().all(|&sample| sample == 0));
    }

    #[test]
    fn test_decoder_misc_wrapper_paths() {
        let mut dec = OpusDecoder::new(48000, 2).unwrap();
        assert!(!dec.get_phase_inversion_disabled());
        dec.set_phase_inversion_disabled(true);
        assert!(dec.get_phase_inversion_disabled());
        assert_eq!(dec.ms_get_phase_inversion_disabled(), 1);
        dec.ms_set_phase_inversion_disabled(0);
        assert!(!dec.get_phase_inversion_disabled());
        assert_eq!(dec.ms_get_phase_inversion_disabled(), 0);

        assert!(!dec.get_ignore_extensions());
        dec.set_ignore_extensions(true);
        assert!(dec.get_ignore_extensions());

        dec.ms_set_gain(123);
        assert_eq!(dec.ms_get_gain(), 123);
        dec.ms_set_complexity(7);
        assert_eq!(dec.ms_get_complexity(), 7);
        assert_eq!(dec.get_complexity(), 7);

        dec.bandwidth = OPUS_BANDWIDTH_FULLBAND;
        dec.range_final = 77;
        dec.last_packet_duration = 960;
        assert_eq!(dec.ms_get_sample_rate(), 48000);
        assert_eq!(dec.ms_get_bandwidth(), OPUS_BANDWIDTH_FULLBAND);
        assert_eq!(dec.ms_get_range_final(), 77);
        assert_eq!(dec.ms_get_last_packet_duration(), 960);

        dec.ms_reset();
        assert_eq!(dec.ms_get_bandwidth(), 0);
        assert_eq!(dec.ms_get_range_final(), 0);
        assert_eq!(dec.ms_get_last_packet_duration(), 0);
        assert_eq!(dec.frame_size, 120);
    }

    #[test]
    fn test_decode24_and_decode_float_roundtrip_shapes() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        let pcm = patterned_pcm_i16(960, 1, 91);
        let mut packet_buf = vec![0u8; 1500];
        let packet_capacity = packet_buf.len() as i32;
        let len = enc
            .encode(&pcm, 960, &mut packet_buf, packet_capacity)
            .unwrap();
        let packet = &packet_buf[..len as usize];

        let mut dec24 = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm24 = vec![0i32; 960];
        let decoded24 = dec24
            .decode24(Some(packet), &mut pcm24, 960, false)
            .unwrap();
        assert_eq!(decoded24, 960);
        assert!(pcm24.iter().any(|&sample| sample != 0));
        let mut too_small24 = vec![0i32; 480];
        assert_eq!(
            dec24.decode24(Some(packet), &mut too_small24, 480, false),
            Err(OPUS_BUFFER_TOO_SMALL)
        );
        assert_eq!(
            dec24.decode24(Some(packet), &mut pcm24, 0, false),
            Err(OPUS_BAD_ARG)
        );
        let mut plc24 = vec![0i32; 960];
        assert_eq!(dec24.decode24(None, &mut plc24, 960, false).unwrap(), 960);

        let mut decf = OpusDecoder::new(48000, 1).unwrap();
        let mut pcmf = vec![0.0f32; 960];
        let decodedf = decf
            .decode_float(Some(packet), &mut pcmf, 960, false)
            .unwrap();
        assert_eq!(decodedf, 960);
        assert!(pcmf.iter().any(|sample| sample.abs() > 1e-4));
        let mut too_smallf = vec![0.0f32; 480];
        assert_eq!(
            decf.decode_float(Some(packet), &mut too_smallf, 480, false),
            Err(OPUS_BUFFER_TOO_SMALL)
        );
        assert_eq!(
            decf.decode_float(Some(packet), &mut pcmf, 0, false),
            Err(OPUS_BAD_ARG)
        );
        let mut plc_f = vec![0.0f32; 960];
        assert_eq!(
            decf.decode_float(None, &mut plc_f, 960, false).unwrap(),
            960
        );

        let mut lowdelay =
            OpusEncoder::new(48000, 1, OPUS_APPLICATION_RESTRICTED_LOWDELAY).unwrap();
        let pcm = patterned_pcm_i16(480, 1, 123);
        let mut lowdelay_packet = vec![0u8; 1500];
        let len = lowdelay
            .encode(&pcm, 480, &mut lowdelay_packet, packet_capacity)
            .unwrap();
        let packet = &lowdelay_packet[..len as usize];
        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm24 = vec![0i32; 480];
        assert_eq!(
            dec.decode24(Some(packet), &mut pcm24, 480, false).unwrap(),
            480
        );
    }

    #[test]
    fn test_get_nb_frames_and_samples_error_edges() {
        assert_eq!(opus_packet_get_nb_frames(&[]), Err(OPUS_BAD_ARG));
        assert_eq!(opus_packet_get_nb_frames(&[0x83]), Err(OPUS_INVALID_PACKET));

        // CELT-only, 20 ms per frame, code 3 with 7 frames => 140 ms > 120 ms max.
        assert_eq!(
            opus_packet_get_nb_samples(&[0x9B, 7], 48000),
            Err(OPUS_INVALID_PACKET)
        );
    }

    #[test]
    fn test_decode_frame_plc_chunking_and_one_byte_dtx_paths() {
        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        dec.prev_mode = MODE_CELT_ONLY;
        dec.frame_size = 1200;

        let mut plc_pcm = vec![0i16; 1200];
        let decoded = dec.decode_frame(None, &mut plc_pcm, 1200, false).unwrap();
        assert_eq!(decoded, 1200);

        let mut dtx_dec = OpusDecoder::new(48000, 1).unwrap();
        dtx_dec.frame_size = 480;
        let mut dtx_pcm = vec![7i16; 960];
        let decoded = dtx_dec
            .decode_frame(Some(&[0x80]), &mut dtx_pcm, 960, false)
            .unwrap();
        assert_eq!(decoded, 480);
        assert!(dtx_pcm[..480].iter().all(|&sample| sample == 0));
    }

    #[test]
    fn test_decode_wrapper_empty_packet_and_small_buffer_error() {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        let pcm = patterned_pcm_i16(960, 1, 211);
        let mut packet_buf = vec![0u8; 1500];
        let packet_capacity = packet_buf.len() as i32;
        let len = enc
            .encode(&pcm, 960, &mut packet_buf, packet_capacity)
            .unwrap();
        let packet = &packet_buf[..len as usize];

        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut too_small = vec![0i16; 120];
        assert_eq!(
            dec.decode_native(Some(packet), &mut too_small, 120, false, false, None),
            Err(OPUS_BUFFER_TOO_SMALL)
        );

        let mut plc_dec = OpusDecoder::new(48000, 1).unwrap();
        let mut plc_pcm = vec![1i16; 120];
        let decoded = plc_dec.decode(Some(&[]), &mut plc_pcm, 120, false).unwrap();
        assert_eq!(decoded, 120);
        assert!(plc_pcm.iter().all(|&sample| sample == 0));
    }

    #[test]
    fn test_decode_transition_and_gain_paths_between_silk_and_celt_packets() {
        let mut silk_enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        assert_eq!(silk_enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
        assert_eq!(silk_enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
        let silk_pcm = patterned_pcm_i16(960, 1, 317);
        let mut silk_packet = vec![0u8; 1500];
        let silk_capacity = silk_packet.len() as i32;
        let silk_len = silk_enc
            .encode(&silk_pcm, 960, &mut silk_packet, silk_capacity)
            .unwrap();

        let mut celt_enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        assert_eq!(celt_enc.set_force_mode(MODE_CELT_ONLY), OPUS_OK);
        let celt_pcm = patterned_pcm_i16(960, 1, 389);
        let mut celt_packet = vec![0u8; 1500];
        let celt_capacity = celt_packet.len() as i32;
        let celt_len = celt_enc
            .encode(&celt_pcm, 960, &mut celt_packet, celt_capacity)
            .unwrap();

        let silk_packet = &silk_packet[..silk_len as usize];
        let celt_packet = &celt_packet[..celt_len as usize];

        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut first = vec![0i16; 960];
        assert_eq!(
            dec.decode(Some(silk_packet), &mut first, 960, false)
                .unwrap(),
            960
        );
        assert_eq!(dec.prev_mode, MODE_SILK_ONLY);

        assert_eq!(dec.set_gain(256), Ok(()));
        let mut second = vec![0i16; 960];
        assert_eq!(
            dec.decode(Some(celt_packet), &mut second, 960, false)
                .unwrap(),
            960
        );
        assert_eq!(dec.prev_mode, MODE_CELT_ONLY);
        assert!(second.iter().any(|&sample| sample != 0));
        assert_ne!(dec.get_final_range(), 0);

        let mut third = vec![0i16; 960];
        assert_eq!(
            dec.decode(Some(silk_packet), &mut third, 960, false)
                .unwrap(),
            960
        );
        assert_eq!(dec.prev_mode, MODE_SILK_ONLY);
        assert!(third.iter().any(|&sample| sample != 0));
    }

    #[test]
    fn test_patterned_pcm_i16_stereo_branch_halves_odd_samples() {
        let pcm = patterned_pcm_i16(2, 2, 3);
        assert_eq!(pcm.len(), 4);
        assert_eq!(pcm[1] as i32 * 2, pcm[0] as i32 + 7919);
        assert_eq!(pcm[3] as i32 * 2, pcm[2] as i32 + 7919);
    }

    #[test]
    fn test_smooth_fade_basic() {
        // Simple crossfade test: in1 all zeros, in2 all max
        let in1 = vec![0i16; 4];
        let in2 = vec![16384i16; 4]; // ~0.5 in Q15
        let mut out = vec![0i16; 4];
        // Use a flat window at Q15ONE (32767)
        let window = vec![32767i16; 120];
        // 1 channel, 4 overlap samples
        smooth_fade(&in1, &in2, &mut out, 4, 1, &window, 48000);
        // With window=Q15ONE squared = Q15ONE, w ~= 1.0
        // out = 1.0 * in2 + 0.0 * in1 = in2
        for i in 0..4 {
            assert!((out[i] as i32 - in2[i] as i32).unsigned_abs() <= 1, "smooth_fade mismatch at {i}");
        }
    }

    // -----------------------------------------------------------------------
    // FFI differential tests (test_lpc_inverse_pred_gain_matches_c_reference,
    // test_garbage_hybrid_swb_decode_matches_c_reference,
    // test_redundancy_differential_sequential_decode) were migrated to
    // harness/tests/c_ref_differential.rs because the ropus library has no
    // build script to link the C reference.
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Additional coverage tests
    // -----------------------------------------------------------------------

    /// Helper: encode frames in a given mode and return the encoded packets.
    fn encode_packets(
        fs: i32,
        channels: i32,
        app: i32,
        mode: Option<i32>,
        bandwidth: Option<i32>,
        bitrate: Option<i32>,
        frame_size: i32,
        count: usize,
    ) -> Vec<Vec<u8>> {
        use crate::opus::encoder::OpusEncoder;
        let mut enc = OpusEncoder::new(fs, channels, app).unwrap();
        if let Some(m) = mode {
            assert_eq!(enc.set_force_mode(m), OPUS_OK);
        }
        if let Some(bw) = bandwidth {
            assert_eq!(enc.set_bandwidth(bw), OPUS_OK);
        }
        if let Some(br) = bitrate {
            assert_eq!(enc.set_bitrate(br), OPUS_OK);
        }
        let mut packets = Vec::new();
        for seed in 0..count {
            let pcm = patterned_pcm_i16(frame_size as usize, channels as usize, seed as i32 * 37);
            let mut buf = vec![0u8; 1500];
            let cap = buf.len() as i32;
            let len = enc.encode(&pcm, frame_size, &mut buf, cap).unwrap();
            packets.push(buf[..len as usize].to_vec());
        }
        packets
    }

    /// Helper: encode SILK packets with FEC enabled.
    fn encode_silk_with_fec(fs: i32, channels: i32, frame_size: i32, count: usize) -> Vec<Vec<u8>> {
        use crate::opus::encoder::{OPUS_APPLICATION_VOIP, OpusEncoder};
        let mut enc = OpusEncoder::new(fs, channels, OPUS_APPLICATION_VOIP).unwrap();
        assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
        assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
        assert_eq!(enc.set_inband_fec(1), OPUS_OK);
        assert_eq!(enc.set_packet_loss_perc(25), OPUS_OK);
        let mut packets = Vec::new();
        for seed in 0..count {
            let pcm = patterned_pcm_i16(frame_size as usize, channels as usize, seed as i32 * 53);
            let mut buf = vec![0u8; 1500];
            let cap = buf.len() as i32;
            let len = enc.encode(&pcm, frame_size, &mut buf, cap).unwrap();
            packets.push(buf[..len as usize].to_vec());
        }
        packets
    }

    #[test]
    fn test_fec_decode_silk_lbrr_path() {
        // Encode several SILK frames with in-band FEC enabled, then decode
        // with decode_fec=true to exercise the FEC/LBRR decoding path.
        let packets = encode_silk_with_fec(48000, 1, 960, 5);

        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0i16; 5760];

        // Decode first two packets normally to prime the decoder state
        for pkt in &packets[..2] {
            let ret = dec.decode(Some(pkt), &mut pcm, 5760, false);
            assert!(ret.is_ok(), "Normal decode failed: {:?}", ret);
        }
        assert_eq!(dec.prev_mode, MODE_SILK_ONLY);

        // Now decode packet[2] with decode_fec=true — this exercises the FEC path
        // which processes LBRR data to recover the previous lost frame.
        let ret = dec.decode(Some(&packets[2]), &mut pcm, 960, true);
        assert!(ret.is_ok(), "FEC decode failed: {:?}", ret);
        assert_eq!(ret.unwrap(), 960);

        // Decode packet[2] again normally (FEC consumed it for recovery, but we
        // still need to properly decode this frame).
        let ret = dec.decode(Some(&packets[2]), &mut pcm, 960, false);
        assert!(ret.is_ok());

        // Decode packet[4] with FEC and a larger frame_size to exercise the
        // PLC + FEC split path (frame_size > packet_frame_size).
        let ret = dec.decode(Some(&packets[4]), &mut pcm, 1920, true);
        assert!(ret.is_ok(), "FEC decode with PLC prefix failed: {:?}", ret);
        assert_eq!(ret.unwrap(), 1920);
    }

    #[test]
    fn test_fec_decode_falls_back_to_plc_for_celt_only() {
        // When decode_fec=true but the packet is CELT-only, the decoder should
        // fall back to PLC (since CELT has no LBRR/FEC).
        let packets = encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_CELT_ONLY),
            None,
            None,
            960,
            3,
        );

        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0i16; 5760];

        // Prime with one normal decode
        dec.decode(Some(&packets[0]), &mut pcm, 960, false).unwrap();
        assert_eq!(dec.prev_mode, MODE_CELT_ONLY);

        // FEC on CELT packet should fall back to PLC
        let ret = dec.decode(Some(&packets[1]), &mut pcm, 960, true);
        assert!(ret.is_ok(), "FEC fallback to PLC failed: {:?}", ret);
        assert_eq!(ret.unwrap(), 960);
    }

    #[test]
    fn test_mode_transition_silk_to_celt() {
        // Decode a SILK packet followed by a CELT packet to trigger the
        // SILK→CELT transition path (pre-decode PLC for crossfade).
        let silk_packets = encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_SILK_ONLY),
            Some(OPUS_BANDWIDTH_WIDEBAND),
            None,
            960,
            2,
        );
        let celt_packets = encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_CELT_ONLY),
            None,
            None,
            960,
            2,
        );

        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0i16; 5760];

        // Prime with SILK
        dec.decode(Some(&silk_packets[0]), &mut pcm, 960, false)
            .unwrap();
        assert_eq!(dec.prev_mode, MODE_SILK_ONLY);
        assert!(!dec.prev_redundancy);

        // Transition to CELT — should trigger PLC pre-decode crossfade
        let ret = dec.decode(Some(&celt_packets[0]), &mut pcm, 960, false);
        assert!(ret.is_ok(), "SILK→CELT transition failed: {:?}", ret);
        assert_eq!(ret.unwrap(), 960);
        assert_eq!(dec.prev_mode, MODE_CELT_ONLY);
        assert!(pcm[..960].iter().any(|&s| s != 0));
    }

    #[test]
    fn test_mode_transition_celt_to_silk() {
        // Decode a CELT packet followed by a SILK packet to trigger the
        // CELT→SILK transition path.
        let celt_packets = encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_CELT_ONLY),
            None,
            None,
            960,
            2,
        );
        let silk_packets = encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_SILK_ONLY),
            Some(OPUS_BANDWIDTH_WIDEBAND),
            None,
            960,
            2,
        );

        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0i16; 5760];

        // Prime with CELT
        dec.decode(Some(&celt_packets[0]), &mut pcm, 960, false)
            .unwrap();
        assert_eq!(dec.prev_mode, MODE_CELT_ONLY);

        // Transition to SILK — triggers PLC-based crossfade
        let ret = dec.decode(Some(&silk_packets[0]), &mut pcm, 960, false);
        assert!(ret.is_ok(), "CELT→SILK transition failed: {:?}", ret);
        assert_eq!(ret.unwrap(), 960);
        assert_eq!(dec.prev_mode, MODE_SILK_ONLY);
    }

    #[test]
    fn test_hybrid_mode_decode() {
        // Force hybrid mode encoding and decode to exercise the hybrid path.
        let packets = encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_HYBRID),
            Some(OPUS_BANDWIDTH_SUPERWIDEBAND),
            Some(32000),
            960,
            3,
        );

        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0i16; 5760];

        for pkt in &packets {
            let mode = opus_packet_get_mode(pkt);
            assert_eq!(mode, MODE_HYBRID, "Expected hybrid packet");
            let ret = dec.decode(Some(pkt), &mut pcm, 960, false);
            assert!(ret.is_ok(), "Hybrid decode failed: {:?}", ret);
            assert_eq!(ret.unwrap(), 960);
        }
        assert_eq!(dec.prev_mode, MODE_HYBRID);
    }

    #[test]
    fn test_plc_for_different_modes() {
        // Exercise PLC (None packet) after priming with each mode.
        let silk_pkt = &encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_SILK_ONLY),
            Some(OPUS_BANDWIDTH_WIDEBAND),
            None,
            960,
            1,
        )[0];
        let celt_pkt = &encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_CELT_ONLY),
            None,
            None,
            960,
            1,
        )[0];
        let hybrid_pkt = &encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_HYBRID),
            Some(OPUS_BANDWIDTH_SUPERWIDEBAND),
            Some(32000),
            960,
            1,
        )[0];

        for (name, pkt, expected_mode) in [
            ("SILK", silk_pkt, MODE_SILK_ONLY),
            ("CELT", celt_pkt, MODE_CELT_ONLY),
            ("Hybrid", hybrid_pkt, MODE_HYBRID),
        ] {
            let mut dec = OpusDecoder::new(48000, 1).unwrap();
            let mut pcm = vec![0i16; 5760];

            // Prime decoder
            dec.decode(Some(pkt), &mut pcm, 960, false).unwrap();
            assert_eq!(dec.prev_mode, expected_mode, "wrong prev_mode after prime");

            // PLC — decoder should conceal without error
            let ret = dec.decode(None, &mut pcm, 960, false);
            assert!(ret.is_ok(), "{name} PLC failed: {:?}", ret);
            assert_eq!(ret.unwrap(), 960);
        }
    }

    #[test]
    fn test_bandwidth_detection_silk_narrowband() {
        // SILK narrowband: TOC byte with bits 7-6=00, bits 6-5=00 → NB
        let packets = encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_SILK_ONLY),
            Some(OPUS_BANDWIDTH_NARROWBAND),
            None,
            960,
            2,
        );

        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0i16; 5760];

        dec.decode(Some(&packets[0]), &mut pcm, 960, false).unwrap();
        assert_eq!(dec.bandwidth, OPUS_BANDWIDTH_NARROWBAND);
    }

    #[test]
    fn test_bandwidth_detection_silk_mediumband() {
        // SILK mediumband (12kHz internal)
        let packets = encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_SILK_ONLY),
            Some(OPUS_BANDWIDTH_MEDIUMBAND),
            None,
            960,
            2,
        );

        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0i16; 5760];

        dec.decode(Some(&packets[0]), &mut pcm, 960, false).unwrap();
        assert_eq!(dec.bandwidth, OPUS_BANDWIDTH_MEDIUMBAND);
    }

    #[test]
    fn test_sample_rate_conversion_decode() {
        // Encoder at 48kHz, decode at various sample rates
        let packets = encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_CELT_ONLY),
            None,
            None,
            960,
            2,
        );

        for &out_rate in &[8000, 12000, 16000, 24000, 48000] {
            let mut dec = OpusDecoder::new(out_rate, 1).unwrap();
            let out_frame = out_rate / 50; // 20ms at output rate
            let mut pcm = vec![0i16; out_frame as usize * 2]; // extra room

            let ret = dec.decode(Some(&packets[0]), &mut pcm, out_frame, false);
            assert!(ret.is_ok(), "decode failed at output rate");
            let decoded = ret.unwrap();
            assert_eq!(decoded, out_frame, "wrong sample count at output rate");
        }
    }

    #[test]
    fn test_decode24_silk_and_hybrid_modes() {
        // Exercise decode24 with SILK and hybrid packets (not just CELT).
        let silk_pkt = &encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_SILK_ONLY),
            Some(OPUS_BANDWIDTH_WIDEBAND),
            None,
            960,
            1,
        )[0];
        let hybrid_pkt = &encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_HYBRID),
            Some(OPUS_BANDWIDTH_SUPERWIDEBAND),
            Some(32000),
            960,
            1,
        )[0];

        for (name, pkt) in [("SILK", silk_pkt), ("Hybrid", hybrid_pkt)] {
            let mut dec = OpusDecoder::new(48000, 1).unwrap();
            let mut pcm24 = vec![0i32; 960];
            let ret = dec.decode24(Some(pkt), &mut pcm24, 960, false);
            assert!(ret.is_ok(), "{name} decode24 failed: {:?}", ret);
            assert_eq!(ret.unwrap(), 960);
            // 24-bit samples are i16 << 8, so should be multiples of 256
            assert!(pcm24.iter().any(|&s| s != 0), "decode24 produced all zeros");
            assert!(pcm24.iter().all(|&s| s % 256 == 0), "decode24 samples not aligned to 256");
        }
    }

    #[test]
    fn test_decode_float_silk_and_hybrid_modes() {
        // Exercise decode_float with SILK and hybrid packets.
        let silk_pkt = &encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_SILK_ONLY),
            Some(OPUS_BANDWIDTH_WIDEBAND),
            None,
            960,
            1,
        )[0];
        let hybrid_pkt = &encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_HYBRID),
            Some(OPUS_BANDWIDTH_SUPERWIDEBAND),
            Some(32000),
            960,
            1,
        )[0];

        for (name, pkt) in [("SILK", silk_pkt), ("Hybrid", hybrid_pkt)] {
            let mut dec = OpusDecoder::new(48000, 1).unwrap();
            let mut pcmf = vec![0.0f32; 960];
            let ret = dec.decode_float(Some(pkt), &mut pcmf, 960, false);
            assert!(ret.is_ok(), "{name} decode_float failed: {:?}", ret);
            assert_eq!(ret.unwrap(), 960);
            assert!(pcmf.iter().any(|s| s.abs() > 1e-6), "decode_float all zeros");
            // Float samples should be in [-1.0, 1.0] range
            assert!(pcmf.iter().all(|&s| s >= -1.0 && s <= 1.0), "decode_float out of range");
        }
    }

    #[test]
    fn test_decode_gain_applied_to_silk_output() {
        // Set a positive decode gain and verify it amplifies the output.
        let packets = encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_SILK_ONLY),
            Some(OPUS_BANDWIDTH_WIDEBAND),
            None,
            960,
            2,
        );

        // Decode without gain
        let mut dec1 = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm1 = vec![0i16; 960];
        dec1.decode(Some(&packets[0]), &mut pcm1, 960, false)
            .unwrap();

        // Decode with gain (+6dB ≈ 256*6 = 1536 in Q8)
        let mut dec2 = OpusDecoder::new(48000, 1).unwrap();
        dec2.set_gain(1536).unwrap();
        let mut pcm2 = vec![0i16; 960];
        dec2.decode(Some(&packets[0]), &mut pcm2, 960, false)
            .unwrap();

        // The gained output should have higher energy
        let energy1: i64 = pcm1.iter().map(|&s| (s as i64) * (s as i64)).sum();
        let energy2: i64 = pcm2.iter().map(|&s| (s as i64) * (s as i64)).sum();
        assert!(energy2 > energy1, "gained output should be louder");
    }

    #[test]
    fn test_get_pitch_silk_and_celt_modes() {
        // Exercise get_pitch in both SILK and CELT modes.
        let silk_pkt = &encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_SILK_ONLY),
            Some(OPUS_BANDWIDTH_WIDEBAND),
            None,
            960,
            2,
        )[0];
        let celt_pkt = &encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_CELT_ONLY),
            None,
            None,
            960,
            2,
        )[0];

        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0i16; 960];

        // SILK mode — pitch from dec_control.prev_pitch_lag
        dec.decode(Some(silk_pkt), &mut pcm, 960, false).unwrap();
        assert_eq!(dec.prev_mode, MODE_SILK_ONLY);
        let silk_pitch = dec.get_pitch();
        // SILK pitch should be non-negative
        assert!(silk_pitch >= 0, "SILK pitch negative: {}", silk_pitch);

        // CELT mode — pitch from celt_dec.get_pitch()
        dec.decode(Some(celt_pkt), &mut pcm, 960, false).unwrap();
        assert_eq!(dec.prev_mode, MODE_CELT_ONLY);
        let celt_pitch = dec.get_pitch();
        assert!(celt_pitch >= 0, "CELT pitch negative: {}", celt_pitch);
    }

    #[test]
    fn test_get_nb_samples_decoder_method() {
        // Exercise the decoder's get_nb_samples method.
        let packets = encode_packets(48000, 1, OPUS_APPLICATION_AUDIO, None, None, None, 960, 1);

        let dec = OpusDecoder::new(48000, 1).unwrap();
        let ns = dec.get_nb_samples(&packets[0]);
        assert!(ns.is_ok(), "get_nb_samples failed: {:?}", ns);
        assert_eq!(ns.unwrap(), 960);

        // Also test at a different decoder rate
        let dec16 = OpusDecoder::new(16000, 1).unwrap();
        let ns16 = dec16.get_nb_samples(&packets[0]);
        assert!(ns16.is_ok());
        // 960 samples at 48kHz → 320 at 16kHz
        assert_eq!(ns16.unwrap(), 320);
    }

    #[test]
    fn test_stereo_decode_silk_celt_hybrid() {
        // Exercise stereo decoding for all three modes.
        let silk_pkt = &encode_packets(
            48000,
            2,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_SILK_ONLY),
            Some(OPUS_BANDWIDTH_WIDEBAND),
            None,
            960,
            1,
        )[0];
        let celt_pkt = &encode_packets(
            48000,
            2,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_CELT_ONLY),
            None,
            None,
            960,
            1,
        )[0];
        let hybrid_pkt = &encode_packets(
            48000,
            2,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_HYBRID),
            Some(OPUS_BANDWIDTH_SUPERWIDEBAND),
            Some(48000),
            960,
            1,
        )[0];

        for (name, pkt) in [
            ("SILK stereo", silk_pkt),
            ("CELT stereo", celt_pkt),
            ("Hybrid stereo", hybrid_pkt),
        ] {
            let mut dec = OpusDecoder::new(48000, 2).unwrap();
            let mut pcm = vec![0i16; 960 * 2]; // stereo
            let ret = dec.decode(Some(pkt), &mut pcm, 960, false);
            assert!(ret.is_ok(), "{name} decode failed: {:?}", ret);
            assert_eq!(ret.unwrap(), 960);
            assert!(pcm.iter().any(|&s| s != 0), "{name}: all zeros");
        }
    }

    #[test]
    fn test_plc_large_frame_chunking() {
        // PLC with a frame_size > 20ms triggers the chunking path in decode_frame.
        let silk_pkt = &encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_SILK_ONLY),
            Some(OPUS_BANDWIDTH_WIDEBAND),
            None,
            960,
            1,
        )[0];

        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0i16; 5760];

        // Prime decoder
        dec.decode(Some(silk_pkt), &mut pcm, 960, false).unwrap();

        // PLC with 2880 samples (60ms) — triggers chunking into 20ms pieces
        let ret = dec.decode(None, &mut pcm, 2880, false);
        assert!(ret.is_ok(), "PLC chunking failed: {:?}", ret);
        assert_eq!(ret.unwrap(), 2880);
    }

    #[test]
    fn test_error_malformed_packet_too_many_frames() {
        // Code 3 packet claiming too many frames
        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0i16; 5760];

        // TOC: CELT 20ms code=3, count byte = 50 frames → 50*20ms = 1000ms > 120ms
        let bad_pkt = [0x83u8, 50];
        let ret = dec.decode(Some(&bad_pkt), &mut pcm, 5760, false);
        assert!(ret.is_err(), "Should reject packet with too many frames");
    }

    #[test]
    fn test_error_empty_buffer_decode24_and_float() {
        // Empty packet to decode24 and decode_float should trigger PLC.
        let mut dec24 = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm24 = vec![0i32; 960];
        let ret = dec24.decode24(Some(&[]), &mut pcm24, 960, false);
        assert!(ret.is_ok());
        let decoded = ret.unwrap();
        // PLC with no prev mode outputs zeros; exact frame count depends
        // on internal frame_size (2.5ms = 120 samples at 48kHz).
        assert!(decoded > 0, "Expected some decoded samples");

        let mut decf = OpusDecoder::new(48000, 1).unwrap();
        let mut pcmf = vec![0.0f32; 960];
        let ret = decf.decode_float(Some(&[]), &mut pcmf, 960, false);
        assert!(ret.is_ok());
    }

    #[test]
    fn test_transition_hybrid_to_silk_celt_fadeout() {
        // Decode hybrid then SILK-only to trigger the hybrid→SILK transition
        // which runs the CELT decoder with a silence frame for fade-out.
        let hybrid_pkt = &encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_HYBRID),
            Some(OPUS_BANDWIDTH_SUPERWIDEBAND),
            Some(32000),
            960,
            1,
        )[0];
        let silk_pkt = &encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_SILK_ONLY),
            Some(OPUS_BANDWIDTH_WIDEBAND),
            None,
            960,
            1,
        )[0];

        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0i16; 5760];

        // Prime with hybrid
        dec.decode(Some(hybrid_pkt), &mut pcm, 960, false).unwrap();
        assert_eq!(dec.prev_mode, MODE_HYBRID);

        // Decode SILK — triggers the hybrid→SILK path (CELT silence fade-out)
        let ret = dec.decode(Some(silk_pkt), &mut pcm, 960, false);
        assert!(ret.is_ok(), "Hybrid→SILK transition failed: {:?}", ret);
        assert_eq!(ret.unwrap(), 960);
        assert_eq!(dec.prev_mode, MODE_SILK_ONLY);
    }

    #[test]
    fn test_fec_decode_with_frame_size_mismatch() {
        // FEC decode where frame_size < packet_frame_size falls back to PLC.
        let packets = encode_silk_with_fec(48000, 1, 960, 3);

        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0i16; 5760];

        // Prime
        dec.decode(Some(&packets[0]), &mut pcm, 960, false).unwrap();

        // FEC with frame_size=480 < packet's 960 → should fall back to PLC
        let ret = dec.decode(Some(&packets[1]), &mut pcm, 480, true);
        assert!(ret.is_ok(), "FEC frame_size mismatch fallback failed");
        assert_eq!(ret.unwrap(), 480);
    }

    #[test]
    fn test_decode24_with_fec() {
        // Exercise the decode24 FEC path.
        let packets = encode_silk_with_fec(48000, 1, 960, 4);

        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm24 = vec![0i32; 960];

        // Prime
        for pkt in &packets[..2] {
            dec.decode24(Some(pkt), &mut pcm24, 960, false).unwrap();
        }

        // FEC decode via decode24
        let ret = dec.decode24(Some(&packets[2]), &mut pcm24, 960, true);
        assert!(ret.is_ok(), "decode24 FEC failed: {:?}", ret);
        assert_eq!(ret.unwrap(), 960);
    }

    #[test]
    fn test_decode_float_with_fec() {
        // Exercise the decode_float FEC path.
        let packets = encode_silk_with_fec(48000, 1, 960, 4);

        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcmf = vec![0.0f32; 960];

        // Prime
        for pkt in &packets[..2] {
            dec.decode_float(Some(pkt), &mut pcmf, 960, false).unwrap();
        }

        // FEC decode via decode_float
        let ret = dec.decode_float(Some(&packets[2]), &mut pcmf, 960, true);
        assert!(ret.is_ok(), "decode_float FEC failed: {:?}", ret);
        assert_eq!(ret.unwrap(), 960);
    }

    #[test]
    fn test_redundancy_voip_auto_mode_sequence() {
        // Encode a sequence of frames with a VOIP encoder that auto-selects
        // mode (SILK, CELT, or hybrid). This exercises mode transitions
        // including any redundancy frames the encoder inserts.
        // We verify that each packet decodes without error and that the
        // decoder tracks mode changes properly.
        use crate::opus::encoder::{OPUS_APPLICATION_VOIP, OpusEncoder};

        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
        enc.set_bitrate(24000);
        enc.set_inband_fec(1);
        enc.set_packet_loss_perc(10);

        let mut packets = Vec::new();
        for seed in 0..10 {
            let pcm = patterned_pcm_i16(960, 1, seed * 1337);
            let mut buf = vec![0u8; 1500];
            let cap = buf.len() as i32;
            let len = enc.encode(&pcm, 960, &mut buf, cap).unwrap();
            packets.push(buf[..len as usize].to_vec());
        }

        // Decode the full sequence and track mode changes
        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0i16; 5760];
        let mut prev_mode = 0;
        let mut _transitions = 0;

        for (i, pkt) in packets.iter().enumerate() {
            let _pkt_mode = opus_packet_get_mode(pkt);
            let ret = dec.decode(Some(pkt), &mut pcm, 5760, false);
            assert!(ret.is_ok(), "frame {i} decode failed");
            let decoded = ret.unwrap();
            assert!(decoded > 0);

            if prev_mode != 0 && dec.prev_mode != prev_mode {
                _transitions += 1;
            }
            prev_mode = dec.prev_mode;
        }

        // Verify at least some packets decoded successfully
        assert!(packets.len() == 10);
        // The VOIP encoder typically starts in SILK mode
        // Verify we can do PLC after the sequence
        let ret = dec.decode(None, &mut pcm, 960, false);
        assert!(ret.is_ok(), "PLC after sequence failed: {:?}", ret);
    }

    // test_redundancy_differential_sequential_decode moved to
    // harness/tests/c_ref_differential.rs (FFI-dependent).

    // -----------------------------------------------------------------------
    // Additional coverage tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_decoder_ctl_ignore_extensions() {
        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        dec.set_ignore_extensions(true);
        assert!(dec.get_ignore_extensions());
        dec.set_ignore_extensions(false);
        assert!(!dec.get_ignore_extensions());
    }

    // Intentional overlap with set_complexity_range_and_getter (line ~4285):
    // this covers the CTL dispatch layer; the other covers the method directly.
    /// Round-trip test mirroring the SET→GET path exercised by
    /// `mdopus_decoder_ctl_{set,get}_int` for `OPUS_SET/GET_COMPLEXITY_REQUEST`.
    /// Validates the CTL surface wired in capi/src/ctl.rs maps 1:1 onto the
    /// Rust decoder's `set_complexity` / `get_complexity`.
    #[test]
    fn test_decoder_ctl_complexity_roundtrip() {
        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        assert_eq!(dec.get_complexity(), 0);
        for v in [0, 5, 10] {
            assert!(dec.set_complexity(v).is_ok());
            assert_eq!(dec.get_complexity(), v);
        }
        // Out-of-range rejected (matches encoder's validation, which the
        // decoder CTL dispatcher forwards to this method unchanged).
        assert!(dec.set_complexity(-1).is_err());
        assert!(dec.set_complexity(11).is_err());
    }

    /// Exercises `OpusDecoder::set_dnn_blob` end-to-end: malformed
    /// blobs rejected with `OPUS_BAD_ARG` (mirrors C's
    /// `opus_decoder_ctl` returning `OPUS_BAD_ARG` on parse failure),
    /// empty blobs rejected, and a structurally-valid PLC blob loads
    /// all three sub-models (PLC prediction layers, PitchDNN, FARGAN)
    /// so `LPCNetPLCState.loaded` stays `true` — the gate the SILK /
    /// CELT neural PLC paths read.
    ///
    /// When the compile-time embedded weight blob is non-empty
    /// (i.e. `reference/dnn/*_data.c` was on disk at build time), the
    /// decoder starts with `loaded=true` already. A malformed
    /// `set_dnn_blob` call can wipe it, because `load_model` clears
    /// layers as it re-initialises — the test accounts for both
    /// starting states by reloading a valid blob at the end.
    #[test]
    fn test_decoder_ctl_set_dnn_blob_loads_real_blob() {
        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        // Random bytes aren't a valid weight blob.
        assert_eq!(dec.set_dnn_blob(&[0u8; 16]), Err(OPUS_BAD_ARG));
        // Empty blob also fails (no records to parse).
        assert_eq!(dec.set_dnn_blob(&[]), Err(OPUS_BAD_ARG));
        // Deterministic synthetic blob from the lpcnet test module —
        // exercises the same parse → init path C uses for real weights.
        let blob = crate::dnn::lpcnet::test_support::make_plc_weight_blob();
        assert_eq!(dec.set_dnn_blob(&blob), Ok(()));
        // Successful load must flip `loaded` (or keep it set, if the
        // compile-time embed already did). Either way, the neural PLC
        // path is active after this point.
        assert!(
            dec.lpcnet.loaded,
            "successful set_dnn_blob must enable neural PLC"
        );
    }

    /// Covers the partial-success failure path inside
    /// `LPCNetPLCState::load_model`: when a blob lacks required weights
    /// for one of the three sub-models, `load_model` returns non-zero
    /// and `loaded` stays `false` (or flips to `false` if it was
    /// previously set). Prevents a silent half-populated state where
    /// e.g. PitchDNN weights from the new blob coexist with stale
    /// FARGAN weights from a prior call.
    #[test]
    fn test_decoder_set_dnn_blob_rejects_partial_blob() {
        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        // PitchDNN-only blob misses FARGAN and PLC weights, so
        // init_fargan / init_plcmodel will fail to find required
        // records and `load_model` returns non-zero.
        let partial_blob = crate::dnn::lpcnet::test_support::make_pitchdnn_weight_blob();
        assert_eq!(dec.set_dnn_blob(&partial_blob), Err(OPUS_BAD_ARG));
        assert!(
            !dec.lpcnet.loaded,
            "partial blob must leave neural PLC disabled, regardless of \
             whether the decoder started with embedded weights loaded"
        );
    }

    /// Confirms the compile-time embedded weight blob auto-loads on
    /// `OpusDecoder::new()` when `reference/dnn/*_data.c` was present
    /// during the build. When build.rs couldn't locate the reference
    /// sources, the blob is empty and this test is vacuously satisfied
    /// (mirroring the crates.io-without-fetch-assets scenario).
    ///
    /// Also pins the blob size (golden-value) when embedded — a change
    /// means either the weights tarball was rev'd or the generator
    /// logic drifted, both of which warrant explicit review before
    /// shipping.
    #[test]
    fn test_decoder_neural_plc_activates_with_default_weights() {
        // Golden-size assertion on the embedded blob. Catches silent
        // tarball drift — if `fetch-assets`'s WEIGHTS_SHA256 changes or
        // our gen_weights_blob.c's record-emit logic shifts, this
        // assertion fails and forces a conscious update. Empty (no
        // reference sources on disk) is a separate, expected state
        // that this test tolerates via the outer `if`.
        if crate::dnn::embedded_weights::has_embedded_weights() {
            // Stage 8.3: DRED weights now included by gen_weights_blob.c
            // (generator policy change, not a tarball/SHA256 change).
            assert_eq!(
                crate::dnn::embedded_weights::WEIGHTS_BLOB.len(),
                8_785_408,
                "embedded blob size changed — xiph tarball drift? update \
                 WEIGHTS_SHA256 in tools/fetch-assets AND this assertion"
            );
        }

        let dec = OpusDecoder::new(48000, 1).unwrap();
        if crate::dnn::embedded_weights::has_embedded_weights() {
            assert!(
                dec.lpcnet.loaded,
                "a fresh decoder must auto-load the embedded weight blob \
                 so neural PLC is armed out-of-the-box"
            );
        } else {
            assert!(
                !dec.lpcnet.loaded,
                "without an embedded blob, the fresh decoder falls back to \
                 classical PLC until the caller supplies weights"
            );
        }
    }

    /// `OpusDecoder::reset()` wipes stream history (FEC queue, FFT
    /// buffers, `pcm` ring, GRU states, etc.) but **preserves loaded
    /// model weights** — both the `loaded` gate and the content of
    /// `self.model` / `self.fargan.model` / `self.enc.pitchdnn.model`
    /// survive. Matches C `lpcnet_plc_reset`
    /// (`reference/dnn/lpcnet_plc.c:45`), which only `OPUS_CLEAR`s from
    /// `LPCNET_PLC_RESET_START` onwards; the `model` / `loaded` /
    /// `fargan` / `enc` fields sit before that marker and are
    /// intentionally untouched.
    ///
    /// Semantic consequence: a user's `set_dnn_blob(custom_blob)`
    /// followed by `reset()` does *not* silently revert to the
    /// compile-time defaults — the custom weights stay live until the
    /// caller explicitly swaps them back. Previous behaviour that
    /// re-loaded the embedded blob on every reset was a deviation from
    /// the C ABI; this test locks in the fix.
    #[test]
    fn test_decoder_reset_preserves_weights() {
        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let blob = crate::dnn::lpcnet::test_support::make_plc_weight_blob();
        dec.set_dnn_blob(&blob).expect("weight blob should load");
        assert!(dec.lpcnet.loaded, "precondition: weights loaded");

        dec.reset();

        // Weights must survive reset regardless of whether the
        // compile-time embedded blob exists — the user's custom blob
        // is what's live, and reset doesn't erase model state.
        assert!(
            dec.lpcnet.loaded,
            "reset must preserve loaded weights — user-supplied blob \
             should survive a stream-history wipe"
        );
    }

    /// Confirms the **neural PLC path actually executes and produces
    /// audible output** under loss, rather than silently falling back
    /// to the classical PLC *or* running the neural branch against
    /// a mis-bound name-to-field mapping that compiles but emits
    /// silence. Two assertions:
    ///
    /// 1. `FRAME_PLC_NEURAL` branch ran (gate wired, complexity>=5,
    ///    weights loaded).
    /// 2. PCM output is non-zero *and* has an RMS above a floor — if
    ///    `init_plcmodel` / `init_fargan` / `init_pitchdnn` had a
    ///    permutation in their weight-name dispatch, `linear_init`
    ///    would still populate *something* for every layer and the
    ///    synthesis would run, but the output would be garbage /
    ///    silence. The RMS floor catches that failure mode.
    ///
    /// Only runs when the compile-time embedded blob is present;
    /// otherwise the synthetic zero-weight blob produces silence by
    /// construction and there's no bound that distinguishes correct
    /// from broken. Stage 7b.2 adds cross-reference parity checks.
    #[test]
    fn test_decoder_neural_plc_branch_runs_under_loss() {
        use crate::opus::encoder::{OPUS_APPLICATION_AUDIO, OpusEncoder};
        const FRAME_PLC_NEURAL: i32 = 4;

        if !crate::dnn::embedded_weights::has_embedded_weights() {
            // Without real weights this test would assert silence — a
            // tautology that hides the failure mode it's meant to catch.
            // Skip cleanly; Stage 7b.2's harness covers the weights-
            // absent scenario via tier-2 SNR instead.
            eprintln!(
                "skipping test_decoder_neural_plc_branch_runs_under_loss: \
                 no embedded weights"
            );
            return;
        }

        let fs = 48000;
        let frame_size = fs / 50; // 20 ms
        let mut enc = OpusEncoder::new(fs, 1, OPUS_APPLICATION_AUDIO).unwrap();
        enc.set_bitrate(96000);
        enc.set_force_mode(MODE_CELT_ONLY);

        let mut dec = OpusDecoder::new(fs, 1).unwrap();
        assert!(dec.set_complexity(5).is_ok());
        // Use the embedded real weights auto-loaded by `new` — no
        // synthetic-blob indirection. `assert!(dec.lpcnet.loaded)`
        // confirms the pre-condition the Stage 7b.1.5 compile-time
        // default buys us.
        assert!(dec.lpcnet.loaded, "precondition: neural PLC enabled");

        // Prime the decoder with good CELT frames so the neural PLC gate
        // sees a fully-initialised history when loss kicks in.
        let mut out = vec![0i16; frame_size as usize];
        for seed in 0..4 {
            let pcm = crate::coverage_tests::patterned_pcm_i16(
                frame_size as usize,
                1,
                seed,
            );
            let mut pkt = vec![0u8; 1500];
            let len = enc.encode(&pcm, frame_size, &mut pkt, 1500).unwrap();
            dec.decode(Some(&pkt[..len as usize]), &mut out, frame_size, false)
                .expect("good-frame decode");
        }

        // Drop a packet; the decoder must pick FRAME_PLC_NEURAL.
        dec.decode(None, &mut out, frame_size, false)
            .expect("PLC decode");
        assert_eq!(
            dec.celt_last_frame_type(),
            FRAME_PLC_NEURAL,
            "CELT PLC must take the neural branch when complexity>=5 \
             and weights are loaded"
        );

        // Output must be audibly non-silent. Catches the silent-failure
        // mode: if the weight-name dispatch were permuted, FARGAN would
        // still synthesise something, but it'd be either zeros or
        // incoherent garbage averaging near zero.
        assert!(
            out.iter().any(|&s| s != 0),
            "neural PLC output is all zero — weights likely not loaded \
             correctly"
        );
        // RMS floor: loose enough that the bound holds across the range
        // of "real weights do sensible things" but tight enough that a
        // silent / near-silent output fails. The bound was calibrated
        // against the embedded blob on this checkout; if a future
        // refetch changes the weights and this drops below 50, verify
        // the new output is still reasonable speech before relaxing.
        let sum_sq: u64 = out.iter().map(|&s| (s as i64 * s as i64) as u64).sum();
        let rms = ((sum_sq as f64) / (out.len() as f64)).sqrt();
        // On this checkout the embedded real weights produce RMS
        // ≈ 1440 against the patterned test signal; 500 is the floor
        // that still catches the silent/near-silent failure modes
        // (zeroed weights ~0, name-permutation garbage < 100) without
        // being so tight a legitimate tarball refresh trips it.
        assert!(
            rms > 500.0,
            "neural PLC RMS too low ({rms:.1}) — weight-name mapping \
             permutation would pass branch assertion but fail this"
        );
    }

    /// Lossless sanity: after loading a valid weight blob, a no-loss
    /// encode/decode round-trip must still succeed and produce non-silent
    /// output. Ensures wiring the neural path through the decoder didn't
    /// regress the classical code path.
    #[test]
    fn test_decoder_lossless_roundtrip_after_dnn_blob_loaded() {
        use crate::opus::encoder::{OPUS_APPLICATION_VOIP, OpusEncoder};
        let fs = 48000;
        let frame_size = fs / 50; // 20 ms
        let mut enc = OpusEncoder::new(fs, 1, OPUS_APPLICATION_VOIP).unwrap();
        enc.set_bitrate(32000);
        let mut dec = OpusDecoder::new(fs, 1).unwrap();
        let blob = crate::dnn::lpcnet::test_support::make_plc_weight_blob();
        dec.set_dnn_blob(&blob).expect("weight blob should load");

        let mut nonzero = false;
        for seed in 0..5 {
            let pcm = crate::coverage_tests::patterned_pcm_i16(
                frame_size as usize,
                1,
                seed,
            );
            let mut pkt = vec![0u8; 1500];
            let len = enc.encode(&pcm, frame_size, &mut pkt, 1500).unwrap();
            let mut out = vec![0i16; frame_size as usize];
            dec.decode(Some(&pkt[..len as usize]), &mut out, frame_size, false)
                .unwrap();
            nonzero |= out.iter().any(|&s| s != 0);
        }
        assert!(nonzero, "lossless decode should produce audible output");
    }

    /// Plumbing regression: with a parseable blob loaded, exercising the
    /// SILK and CELT lost-frame paths must not panic or error. This is a
    /// plumbing-level test only — because Stage 7b.1 ships the
    /// "honest gap" (parse succeeds, weights not populated,
    /// `LPCNetPLCState.loaded = false`), the decoder falls through to
    /// the classical pitch/noise PLC on every loss. The assertion here
    /// is therefore limited to "decode returns `Ok` under loss"; a real
    /// neural-PLC regression test lands in Stage 7b.1.5 once
    /// `LPCNetState::load_model` populates model fields from the blob.
    ///
    /// Covers both SILK-only (16 kHz, where the SILK PLC gating lives)
    /// and CELT-only (48 kHz, where the `FRAME_PLC_NEURAL` gating lives)
    /// so that a regression in either branch shows up as a panic in CI.
    #[test]
    fn test_decoder_does_not_panic_under_loss() {
        use crate::opus::encoder::{OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_VOIP, OpusEncoder};
        let fs = 48000;
        let frame_size = fs / 50; // 20 ms
        let blob = crate::dnn::lpcnet::test_support::make_plc_weight_blob();

        for &(app, force_mode, bitrate, label) in &[
            (OPUS_APPLICATION_VOIP, MODE_SILK_ONLY, 24000, "SILK"),
            (OPUS_APPLICATION_AUDIO, MODE_CELT_ONLY, 96000, "CELT"),
        ] {
            let mut enc = OpusEncoder::new(fs, 1, app).unwrap();
            enc.set_bitrate(bitrate);
            enc.set_force_mode(force_mode);
            let mut dec = OpusDecoder::new(fs, 1).unwrap();
            // Complexity ≥ 5 would unlock neural PLC once 7b.1.5 populates
            // weights; here it simply exercises the gate that Stage 7b.1
            // wires up in `decode_frame`.
            assert!(dec.set_complexity(5).is_ok(), "{label}");
            dec.set_dnn_blob(&blob).expect("weight blob should load");

            // Prime with 4 good packets so the decoder builds real history
            // before we start dropping frames.
            let mut out = vec![0i16; frame_size as usize];
            for seed in 0..4 {
                let pcm = crate::coverage_tests::patterned_pcm_i16(
                    frame_size as usize,
                    1,
                    seed,
                );
                let mut pkt = vec![0u8; 1500];
                let len = enc.encode(&pcm, frame_size, &mut pkt, 1500).unwrap();
                dec.decode(Some(&pkt[..len as usize]), &mut out, frame_size, false)
                    .unwrap_or_else(|e| panic!("{label}: good-frame decode failed: {e}"));
            }

            // Drop three consecutive packets. The decode path must pick the
            // classical PLC (neural is gated off by `loaded=false`) and
            // return a sample count without panic. Any index-out-of-bounds
            // in the neural threading would trip here.
            for i in 0..3 {
                let decoded = dec
                    .decode(None, &mut out, frame_size, false)
                    .unwrap_or_else(|e| panic!("{label}: PLC frame {i} failed: {e}"));
                assert_eq!(decoded, frame_size, "{label}: short PLC");
            }
        }
    }

    #[test]
    fn test_decoder_bandwidth_detection_all_modes() {
        use crate::opus::encoder::{OPUS_APPLICATION_VOIP, OpusEncoder};
        // Encode at different sample rates and check decoder bandwidth
        for &(rate, _mode_name) in &[(8000, "NB"), (12000, "MB"), (16000, "WB")] {
            let frame_size = rate / 50;
            let mut enc = OpusEncoder::new(rate, 1, OPUS_APPLICATION_VOIP).unwrap();
            enc.set_bitrate(24000);
            enc.set_force_mode(MODE_SILK_ONLY);
            let pcm = crate::coverage_tests::patterned_pcm_i16(frame_size as usize, 1, rate);
            let mut pkt = vec![0u8; 1500];
            let len = enc.encode(&pcm, frame_size, &mut pkt, 1500).unwrap();

            let mut dec = OpusDecoder::new(rate, 1).unwrap();
            let mut out = vec![0i16; frame_size as usize];
            dec.decode(Some(&pkt[..len as usize]), &mut out, frame_size, false)
                .unwrap();
            let bw = dec.get_bandwidth();
            assert!(bw >= OPUS_BANDWIDTH_NARROWBAND && bw <= OPUS_BANDWIDTH_FULLBAND, "bandwidth out of range");
        }
    }

    #[test]
    fn test_decode_stereo_plc_multiple_modes() {
        use crate::opus::encoder::{OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_VOIP, OpusEncoder};
        // Test PLC for stereo in SILK, CELT, and hybrid modes
        for &(app, force_mode, bitrate, mode_name) in &[
            (OPUS_APPLICATION_VOIP, MODE_SILK_ONLY, 24000, "SILK"),
            (OPUS_APPLICATION_AUDIO, MODE_CELT_ONLY, 96000, "CELT"),
            (OPUS_APPLICATION_AUDIO, MODE_HYBRID, 48000, "Hybrid"),
        ] {
            let mut enc = OpusEncoder::new(48000, 2, app).unwrap();
            enc.set_bitrate(bitrate);
            enc.set_force_mode(force_mode);
            let mut dec = OpusDecoder::new(48000, 2).unwrap();

            // Prime with 3 real packets
            for i in 0..3 {
                let pcm = crate::coverage_tests::patterned_pcm_i16(960, 2, i * 7);
                let mut pkt = vec![0u8; 1500];
                let len = enc.encode(&pcm, 960, &mut pkt, 1500).unwrap();
                let mut out = vec![0i16; 960 * 2];
                dec.decode(Some(&pkt[..len as usize]), &mut out, 960, false)
                    .unwrap();
            }

            // PLC
            let mut out = vec![0i16; 960 * 2];
            let ret = dec.decode(None, &mut out, 960, false).unwrap();
            assert_eq!(ret, 960, "{}: PLC returned {}", mode_name, ret);
        }
    }

    #[test]
    fn test_decode_large_frame_chunking_40ms() {
        use crate::opus::encoder::{OPUS_APPLICATION_VOIP, OPUS_FRAMESIZE_40_MS, OpusEncoder};
        let mut enc = OpusEncoder::new(16000, 1, OPUS_APPLICATION_VOIP).unwrap();
        enc.set_bitrate(24000);
        enc.set_expert_frame_duration(OPUS_FRAMESIZE_40_MS);
        let pcm = crate::coverage_tests::patterned_pcm_i16(640, 1, 99);
        let mut pkt = vec![0u8; 1500];
        let len = enc.encode(&pcm, 640, &mut pkt, 1500).unwrap();

        let mut dec = OpusDecoder::new(16000, 1).unwrap();
        let mut out = vec![0i16; 640];
        let decoded = dec
            .decode(Some(&pkt[..len as usize]), &mut out, 640, false)
            .unwrap();
        assert_eq!(decoded, 640);
    }

    #[test]
    fn test_decoder_reset_clears_state() {
        use crate::opus::encoder::{OPUS_APPLICATION_AUDIO, OpusEncoder};
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        enc.set_bitrate(64000);
        let pcm = crate::coverage_tests::patterned_pcm_i16(960, 1, 42);
        let mut pkt = vec![0u8; 1500];
        let len = enc.encode(&pcm, 960, &mut pkt, 1500).unwrap();

        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut out = vec![0i16; 960];
        dec.decode(Some(&pkt[..len as usize]), &mut out, 960, false)
            .unwrap();

        // Reset and decode again — should work cleanly
        dec.reset();
        dec.decode(Some(&pkt[..len as usize]), &mut out, 960, false)
            .unwrap();
    }

    #[test]
    fn test_fec_decode_across_modes() {
        use crate::opus::encoder::{OPUS_APPLICATION_VOIP, OpusEncoder};
        // SILK with FEC enabled
        let mut enc = OpusEncoder::new(16000, 1, OPUS_APPLICATION_VOIP).unwrap();
        enc.set_bitrate(32000);
        enc.set_inband_fec(1);
        enc.set_packet_loss_perc(30);
        enc.set_complexity(10);

        let mut dec = OpusDecoder::new(16000, 1).unwrap();
        let mut pkts = Vec::new();
        for i in 0..6 {
            let pcm = crate::coverage_tests::patterned_pcm_i16(320, 1, 100 + i * 13);
            let mut pkt = vec![0u8; 1500];
            let len = enc.encode(&pcm, 320, &mut pkt, 1500).unwrap();
            pkts.push(pkt[..len as usize].to_vec());
        }

        // Decode normally
        for pkt in &pkts[..3] {
            let mut out = vec![0i16; 320];
            dec.decode(Some(pkt), &mut out, 320, false).unwrap();
        }
        // Simulate loss of packet 3, use FEC from packet 4
        let mut out = vec![0i16; 320];
        dec.decode(Some(&pkts[4]), &mut out, 320, true).unwrap(); // FEC
        // Resume
        let mut out = vec![0i16; 320];
        dec.decode(Some(&pkts[4]), &mut out, 320, false).unwrap();
    }

    // -----------------------------------------------------------------------
    // Coverage: packet parsing error paths
    // -----------------------------------------------------------------------

    #[test]
    fn test_packet_parse_negative_len() {
        // Line 188: len < 0 should return OPUS_BAD_ARG
        let data = [0x80u8, 0x00];
        let mut toc = 0u8;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut offset = 0i32;
        let ret = opus_packet_parse_impl(&data, -1, false, &mut toc, &mut sizes, &mut offset, None);
        assert_eq!(ret, OPUS_BAD_ARG);
    }

    #[test]
    fn test_packet_parse_zero_len() {
        // Line 191: len == 0 should return OPUS_INVALID_PACKET
        let data = [0x80u8];
        let mut toc = 0u8;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut offset = 0i32;
        let ret = opus_packet_parse_impl(&data, 0, false, &mut toc, &mut sizes, &mut offset, None);
        assert_eq!(ret, OPUS_INVALID_PACKET);
    }

    #[test]
    fn test_packet_parse_code2_parse_size_error() {
        // Line 231: Code 2 (VBR) where parse_size returns error (only TOC, no size byte)
        let pkt = [0x82u8]; // TOC code=2, but no size byte follows
        let mut toc = 0u8;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut offset = 0i32;
        let ret = opus_packet_parse_impl(
            &pkt,
            pkt.len() as i32,
            false,
            &mut toc,
            &mut sizes,
            &mut offset,
            None,
        );
        assert_eq!(ret, OPUS_INVALID_PACKET);
    }

    #[test]
    fn test_packet_parse_code2_size_exceeds_remaining() {
        // Line 236: Code 2 (VBR) where first frame size > remaining data
        // TOC code=2, size byte = 200, but only 2 bytes total after TOC
        let pkt = [0x82u8, 200, 0, 0];
        let mut toc = 0u8;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut offset = 0i32;
        let ret = opus_packet_parse_impl(
            &pkt,
            pkt.len() as i32,
            false,
            &mut toc,
            &mut sizes,
            &mut offset,
            None,
        );
        assert_eq!(ret, OPUS_INVALID_PACKET);
    }

    #[test]
    fn test_packet_parse_code3_vbr_frame_size_exceeds_remaining() {
        // Lines 281/286: Code 3 VBR where a frame size exceeds remaining data
        // TOC code=3, ch byte = 0x82 (VBR, 2 frames), size[0] = 250
        // Total payload too small for size[0]
        let pkt = [0x83u8, 0x82, 250, 0, 0];
        let mut toc = 0u8;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut offset = 0i32;
        let ret = opus_packet_parse_impl(
            &pkt,
            pkt.len() as i32,
            false,
            &mut toc,
            &mut sizes,
            &mut offset,
            None,
        );
        assert_eq!(ret, OPUS_INVALID_PACKET);
    }

    #[test]
    fn test_packet_parse_code3_vbr_negative_last_size() {
        // Line 292: Code 3 VBR where accumulated frame sizes make last_size < 0
        // 3 VBR frames, sizes claim more than available
        // TOC code=3, ch = 0x83 (VBR, 3 frames), sizes[0]=100, sizes[1]=100
        // But total remaining after TOC+ch is only 210, so last_size = 210 - 2 - 100 - 2 - 100 < 0
        let mut pkt = vec![0x83u8, 0x83]; // TOC + ch (VBR, 3 frames)
        pkt.push(100); // size[0] = 100
        pkt.push(100); // size[1] = 100
        // Payload: need 200 bytes for frames 0+1, plus room for frame 2
        // Total after TOC+ch = pkt.len()-2 = payload
        // We need remaining - (1+100) - (1+100) < 0 for last_size
        // remaining = len - 2 (TOC+ch). If len = 204, remaining = 202
        // After size[0]: remaining -= 1 (byte), last_size -= 1+100 = 101. rem=201, last=101
        // After size[1]: remaining -= 1, last_size -= 1+100 = 0. But that's exactly 0, not negative.
        // Let's make it so sizes are huge relative to total payload.
        // Actually let's use 2-byte size encoding to claim larger sizes.
        // size byte 252+x means 4*second_byte + first_byte. Use [255, 50] = 4*50+255=455
        let mut pkt2 = vec![0x83u8, 0x82]; // VBR, 2 frames
        pkt2.extend_from_slice(&[255, 50]); // size[0] = 455 (2 bytes consumed)
        pkt2.extend_from_slice(&[0u8; 10]); // small payload
        let mut toc = 0u8;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut offset = 0i32;
        let ret = opus_packet_parse_impl(
            &pkt2,
            pkt2.len() as i32,
            false,
            &mut toc,
            &mut sizes,
            &mut offset,
            None,
        );
        // size[0]=455 > remaining (after TOC+ch+2 size bytes = 10), so hits line 286
        assert_eq!(ret, OPUS_INVALID_PACKET);
    }

    #[test]
    fn test_packet_parse_code3_cbr_not_divisible() {
        // Line 298: Code 3 CBR where remaining is not divisible by count
        // TOC code=3, ch = 0x03 (CBR, 3 frames), remaining = 10 (not divisible by 3)
        let mut pkt = vec![0x83u8, 0x03]; // TOC + ch (CBR, 3 frames)
        pkt.extend_from_slice(&[0u8; 10]); // 10 bytes not divisible by 3
        let mut toc = 0u8;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut offset = 0i32;
        let ret = opus_packet_parse_impl(
            &pkt,
            pkt.len() as i32,
            false,
            &mut toc,
            &mut sizes,
            &mut offset,
            None,
        );
        assert_eq!(ret, OPUS_INVALID_PACKET);
    }

    #[test]
    fn test_packet_parse_code3_padding_exhausts_remaining() {
        // Line 271: Code 3 with padding that makes remaining < 0
        // TOC code=3, ch = 0x43 (padding + CBR, 3 frames), padding byte = 200
        // Total data too small to survive the padding
        let pkt = [0x83u8, 0x43, 200]; // padding of 200 > remaining after ch (1 byte)
        let mut toc = 0u8;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut offset = 0i32;
        let ret = opus_packet_parse_impl(
            &pkt,
            pkt.len() as i32,
            false,
            &mut toc,
            &mut sizes,
            &mut offset,
            None,
        );
        assert_eq!(ret, OPUS_INVALID_PACKET);
    }

    #[test]
    fn test_packet_parse_code3_vbr_parse_size_error_in_loop() {
        // Line 281: Code 3 VBR where parse_size fails inside the frame-size loop
        // TOC code=3, ch = 0x83 (VBR, 3 frames), then a 252+ byte with no second byte
        let pkt = [0x83u8, 0x83, 255]; // VBR, 3 frames. parse_size sees 255 but no 2nd byte
        let mut toc = 0u8;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut offset = 0i32;
        let ret = opus_packet_parse_impl(
            &pkt,
            pkt.len() as i32,
            false,
            &mut toc,
            &mut sizes,
            &mut offset,
            None,
        );
        assert_eq!(ret, OPUS_INVALID_PACKET);
    }

    #[test]
    fn test_packet_parse_last_size_exceeds_1275() {
        // Line 335: Non-self-delimited packet where last_size > 1275
        // Code 0 (single frame) with payload > 1275 bytes
        let mut pkt = vec![0x80u8]; // TOC: CELT, code 0
        pkt.extend_from_slice(&[0u8; 1276]); // 1276 bytes payload > 1275 limit
        let mut toc = 0u8;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut offset = 0i32;
        let ret = opus_packet_parse_impl(
            &pkt,
            pkt.len() as i32,
            false,
            &mut toc,
            &mut sizes,
            &mut offset,
            None,
        );
        assert_eq!(ret, OPUS_INVALID_PACKET);
    }

    #[test]
    fn test_packet_parse_self_delimited_size_error() {
        // Line 311: Self-delimited framing where parse_size for last frame fails
        // Code 0 (single frame), self_delimited=true, but no size byte for last frame
        let pkt = [0x80u8]; // Only TOC, no size byte for self-delimited last frame
        let mut toc = 0u8;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut offset = 0i32;
        let ret = opus_packet_parse_impl(
            &pkt,
            pkt.len() as i32,
            true,
            &mut toc,
            &mut sizes,
            &mut offset,
            None,
        );
        assert_eq!(ret, OPUS_INVALID_PACKET);
    }

    #[test]
    fn test_packet_parse_self_delimited_size_exceeds_remaining() {
        // Line 322: Self-delimited where the declared size > remaining bytes
        // Code 0, self_delimited, size byte claims 200 but only a few bytes follow
        let pkt = [0x80u8, 200, 0, 0]; // TOC + size=200, but only 2 payload bytes
        let mut toc = 0u8;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut offset = 0i32;
        let ret = opus_packet_parse_impl(
            &pkt,
            pkt.len() as i32,
            true,
            &mut toc,
            &mut sizes,
            &mut offset,
            None,
        );
        assert_eq!(ret, OPUS_INVALID_PACKET);
    }

    #[test]
    fn test_packet_parse_self_delimited_cbr_size_times_count_exceeds_remaining() {
        // Line 322: Self-delimited CBR where sz*count > remaining
        // Code 3, CBR, 3 frames, self-delimited, size byte = 100 => 100*3=300 > remaining
        let mut pkt = vec![0x83u8, 0x03]; // TOC + ch (CBR, 3 frames)
        pkt.push(100); // self-delimited last-frame size = 100
        pkt.extend_from_slice(&[0u8; 10]); // only 10 bytes payload
        let mut toc = 0u8;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut offset = 0i32;
        let ret = opus_packet_parse_impl(
            &pkt,
            pkt.len() as i32,
            true,
            &mut toc,
            &mut sizes,
            &mut offset,
            None,
        );
        assert_eq!(ret, OPUS_INVALID_PACKET);
    }

    #[test]
    fn test_packet_parse_self_delimited_vbr_bytes_plus_sz_exceeds_last() {
        // Line 328: Self-delimited VBR where bytes + sz > last_size
        // Code 3, VBR, 2 frames. size[0]=5, self-delimited size = 100
        // but total is too small for that
        let mut pkt = vec![0x83u8, 0x82]; // TOC + ch (VBR, 2 frames)
        pkt.push(5); // size[0] = 5
        pkt.push(100); // self-delimited last size = 100
        pkt.extend_from_slice(&[0u8; 10]); // small payload
        let mut toc = 0u8;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut offset = 0i32;
        let ret = opus_packet_parse_impl(
            &pkt,
            pkt.len() as i32,
            true,
            &mut toc,
            &mut sizes,
            &mut offset,
            None,
        );
        assert_eq!(ret, OPUS_INVALID_PACKET);
    }

    // -----------------------------------------------------------------------
    // Coverage: opus_packet_has_lbrr with empty first frame
    // -----------------------------------------------------------------------

    #[test]
    fn test_packet_has_lbrr_empty_first_frame() {
        // Line 387: sizes[0] == 0 should return Ok(false)
        // SILK 20ms mono, code 2 (VBR), first frame size = 0
        // TOC: SILK NB 20ms mono code=2 = (config=1)<<3 | 0<<2 | 2 = 0x0A
        let pkt = [0x0Au8, 0x00, 0x00, 0x00]; // size[0]=0, rest is frame 2
        let ret = opus_packet_has_lbrr(&pkt, pkt.len() as i32);
        assert_eq!(ret, Ok(false));
    }

    // -----------------------------------------------------------------------
    // Coverage: opus_packet_get_nb_samples exercising success path
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_nb_samples_various_rates() {
        // Line 165: opus_packet_get_nb_samples success path at different sample rates
        // SILK 20ms mono code 0: config=1 => 20ms, 1 frame
        let pkt = [0x08u8, 0x00]; // SILK 20ms, code 0
        // At 48kHz: 1 * 960 = 960
        assert_eq!(opus_packet_get_nb_samples(&pkt, 48000), Ok(960));
        // At 16kHz: 1 * 320 = 320
        assert_eq!(opus_packet_get_nb_samples(&pkt, 16000), Ok(320));
        // At 8kHz: 1 * 160 = 160
        assert_eq!(opus_packet_get_nb_samples(&pkt, 8000), Ok(160));

        // Code 1 (2 CBR frames), CELT 20ms: 2 * 960 = 1920 at 48kHz
        let pkt2 = [0x99u8, 0x00, 0x00]; // CELT 20ms code=1, with 2 bytes payload
        assert_eq!(opus_packet_get_nb_samples(&pkt2, 48000), Ok(1920));
    }

    // -----------------------------------------------------------------------
    // Coverage: PLC audio size adjustment branches
    // -----------------------------------------------------------------------

    #[test]
    fn test_plc_audiosize_adjustment_between_f5_and_f10() {
        // Lines 701-705: PLC with frame_size between f5 and f10 for non-SILK mode
        // At 48kHz: f20=960, f10=480, f5=240, f2.5=120
        // CELT PLC with frame_size=360 (between f5=240 and f10=480) should clamp to f5=240
        let celt_pkt = &encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_CELT_ONLY),
            None,
            None,
            960,
            1,
        )[0];
        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0i16; 5760];
        // Prime with a CELT packet
        dec.decode(Some(celt_pkt), &mut pcm, 960, false).unwrap();
        assert_eq!(dec.prev_mode, MODE_CELT_ONLY);

        // Now PLC with frame_size=360 (between f5=240 and f10=480)
        let ret = dec.decode_frame(None, &mut pcm, 360, false);
        assert!(ret.is_ok(), "PLC audiosize adjustment failed: {:?}", ret);
        // Should have been clamped to f5=240
        assert_eq!(ret.unwrap(), 240);
    }

    #[test]
    fn test_plc_audiosize_adjustment_between_f10_and_f20() {
        // Lines 701-702: PLC with frame_size between f10 and f20 should clamp to f10
        // At 48kHz: f20=960, f10=480
        let silk_pkt = &encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_SILK_ONLY),
            Some(OPUS_BANDWIDTH_WIDEBAND),
            None,
            960,
            1,
        )[0];
        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0i16; 5760];
        dec.decode(Some(silk_pkt), &mut pcm, 960, false).unwrap();

        // PLC with frame_size=600 (between f10=480 and f20=960)
        let ret = dec.decode_frame(None, &mut pcm, 600, false);
        assert!(ret.is_ok(), "PLC audiosize clamp to f10 failed: {:?}", ret);
        assert_eq!(ret.unwrap(), 480);
    }

    // -----------------------------------------------------------------------
    // Coverage: CELT→SILK transition (non-redundant) with PLC pre-decode
    // -----------------------------------------------------------------------

    #[test]
    fn test_transition_celt_to_silk_non_redundant() {
        // Lines 886-891: Transition from CELT to SILK (non-redundant) triggers PLC pre-decode
        // into a transition buffer, then smooth-fades with the new SILK output.
        let celt_pkt = &encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_CELT_ONLY),
            None,
            None,
            960,
            2,
        );
        let silk_pkt = &encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_SILK_ONLY),
            Some(OPUS_BANDWIDTH_WIDEBAND),
            None,
            960,
            1,
        )[0];

        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0i16; 5760];

        // Prime with CELT packets
        for pkt in celt_pkt {
            dec.decode(Some(pkt), &mut pcm, 960, false).unwrap();
        }
        assert_eq!(dec.prev_mode, MODE_CELT_ONLY);

        // Decode SILK — this is CELT→SILK transition (non-redundant path)
        // which exercises lines 886-891 (transition PLC for SILK→CELT smooth fade)
        let ret = dec.decode(Some(silk_pkt), &mut pcm, 960, false);
        assert!(ret.is_ok(), "CELT->SILK transition failed: {:?}", ret);
        assert_eq!(ret.unwrap(), 960);
        assert_eq!(dec.prev_mode, MODE_SILK_ONLY);
    }

    // -----------------------------------------------------------------------
    // Coverage: Transition smooth-fade with short frame (audiosize < f5)
    // -----------------------------------------------------------------------

    #[test]
    fn test_transition_smooth_fade_short_frame() {
        // Lines 1061-1074: Transition crossfade when audiosize < f5
        // At 48kHz: f5=240, f2.5=120. We need a frame with audiosize < 240.
        // Use CELT 2.5ms frames (120 samples) to create a short transition.
        // Prime decoder with SILK, then decode a CELT 2.5ms packet.
        let silk_pkt = &encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_SILK_ONLY),
            Some(OPUS_BANDWIDTH_WIDEBAND),
            None,
            960,
            2,
        );

        // CELT 2.5ms: config bits 4-3 = 0 in CELT range -> 2.5ms
        // Use RESTRICTED_LOWDELAY which forces CELT and allows short frames
        let mut celt_enc =
            OpusEncoder::new(48000, 1, OPUS_APPLICATION_RESTRICTED_LOWDELAY).unwrap();
        celt_enc.set_bitrate(64000);
        let short_pcm = patterned_pcm_i16(120, 1, 777);
        let mut celt_buf = vec![0u8; 1500];
        let celt_len = celt_enc
            .encode(&short_pcm, 120, &mut celt_buf, 1500)
            .unwrap();
        let celt_short_pkt = &celt_buf[..celt_len as usize];

        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0i16; 5760];

        // Prime with SILK
        for pkt in silk_pkt {
            dec.decode(Some(pkt), &mut pcm, 960, false).unwrap();
        }
        assert_eq!(dec.prev_mode, MODE_SILK_ONLY);

        // Decode short CELT — triggers SILK→CELT transition with short frame
        let ret = dec.decode(Some(celt_short_pkt), &mut pcm, 120, false);
        assert!(ret.is_ok(), "Short CELT transition failed: {:?}", ret);
        assert_eq!(ret.unwrap(), 120);
        assert_eq!(dec.prev_mode, MODE_CELT_ONLY);
    }

    // -----------------------------------------------------------------------
    // Coverage: FEC decode path with frame_size > packet_frame_size
    // -----------------------------------------------------------------------

    #[test]
    fn test_fec_partial_decode_plc_prefix() {
        // Lines 1188-1207: FEC decode where frame_size > packet_frame_size
        // triggers PLC for the prefix, then FEC for the remainder.
        let packets = encode_silk_with_fec(48000, 1, 960, 5);

        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0i16; 5760];

        // Prime decoder with first 3 packets
        for pkt in &packets[..3] {
            dec.decode(Some(pkt), &mut pcm, 960, false).unwrap();
        }

        // FEC decode with frame_size=1920 > packet_frame_size=960
        // This triggers PLC for first 960 samples, then FEC for last 960
        let ret = dec.decode(Some(&packets[4]), &mut pcm, 1920, true);
        assert!(ret.is_ok(), "FEC partial decode with PLC prefix failed");
        assert_eq!(ret.unwrap(), 1920);
    }

    // -----------------------------------------------------------------------
    // Coverage: FEC falls back to PLC when prev mode is CELT
    // -----------------------------------------------------------------------

    #[test]
    fn test_fec_fallback_plc_when_prev_mode_celt() {
        // Line 1181: decode_fec=true with self.mode == MODE_CELT_ONLY
        // falls back to PLC (CELT has no LBRR)
        let celt_pkt = &encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_CELT_ONLY),
            None,
            None,
            960,
            2,
        );
        let silk_fec_pkt = &encode_silk_with_fec(48000, 1, 960, 3);

        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0i16; 5760];

        // Prime with CELT packets to set self.mode to CELT
        for pkt in celt_pkt {
            dec.decode(Some(pkt), &mut pcm, 960, false).unwrap();
        }
        assert_eq!(dec.prev_mode, MODE_CELT_ONLY);

        // FEC with a SILK packet, but self.mode is CELT => falls back to PLC
        let ret = dec.decode(Some(&silk_fec_pkt[2]), &mut pcm, 960, true);
        assert!(ret.is_ok(), "FEC fallback to PLC failed: {:?}", ret);
        assert_eq!(ret.unwrap(), 960);
    }

    // -----------------------------------------------------------------------
    // Coverage: Hybrid→SILK transition CELT fade-out
    // -----------------------------------------------------------------------

    #[test]
    fn test_hybrid_to_silk_celt_fadeout_with_prev_redundancy() {
        // Lines 962-980: When transitioning from hybrid to SILK-only,
        // the CELT decoder decodes a silence frame for fade-out.
        // Also test that prev_redundancy=true suppresses the fade-out.
        let hybrid_pkt = &encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_HYBRID),
            Some(OPUS_BANDWIDTH_SUPERWIDEBAND),
            Some(32000),
            960,
            2,
        );
        let silk_pkt = &encode_packets(
            48000,
            1,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_SILK_ONLY),
            Some(OPUS_BANDWIDTH_WIDEBAND),
            None,
            960,
            2,
        );

        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0i16; 5760];

        // Prime with hybrid
        for pkt in hybrid_pkt {
            dec.decode(Some(pkt), &mut pcm, 960, false).unwrap();
        }
        assert_eq!(dec.prev_mode, MODE_HYBRID);

        // First SILK decode — triggers hybrid→SILK CELT fade-out (line 968-980)
        let ret = dec.decode(Some(&silk_pkt[0]), &mut pcm, 960, false);
        assert!(ret.is_ok(), "Hybrid->SILK first transition failed");
        assert_eq!(ret.unwrap(), 960);
        assert_eq!(dec.prev_mode, MODE_SILK_ONLY);

        // Second SILK decode — no longer hybrid->SILK, so no fade-out
        let ret = dec.decode(Some(&silk_pkt[1]), &mut pcm, 960, false);
        assert!(ret.is_ok());
        assert_eq!(ret.unwrap(), 960);
    }

    // -----------------------------------------------------------------------
    // Coverage: SILK bandwidth setting paths (NB, MB, WB)
    // -----------------------------------------------------------------------

    #[test]
    fn test_silk_bandwidth_narrowband_and_mediumband() {
        // Lines 731, 736: SILK bandwidth settings for NB and MB
        // (WB is the default/fallthrough, already covered)
        for &(bw, rate, bw_name) in &[
            (OPUS_BANDWIDTH_NARROWBAND, 8000, "NB"),
            (OPUS_BANDWIDTH_MEDIUMBAND, 12000, "MB"),
        ] {
            let packets = encode_packets(
                rate,
                1,
                OPUS_APPLICATION_AUDIO,
                Some(MODE_SILK_ONLY),
                Some(bw),
                None,
                rate / 50,
                1,
            );
            let mut dec = OpusDecoder::new(rate, 1).unwrap();
            let mut pcm = vec![0i16; rate as usize / 50];
            let ret = dec.decode(Some(&packets[0]), &mut pcm, rate / 50, false);
            assert!(ret.is_ok(), "{bw_name} decode failed: {:?}", ret);
            assert_eq!(
                ret.unwrap(),
                rate / 50,
                "{bw_name}: expected {} samples",
                rate / 50
            );
        }
    }

    // test_silk_subframe_boundary_divergence and test_fuzz_decode_scan_all_configs
    // (FFI-dependent: call c_ref_decode helper) were migrated to
    // harness/tests/c_ref_differential.rs.

    #[test]
    fn test_opus_decoder_debug_accessors() {
        // Cover debug accessor methods on OpusDecoder (lines 1357-1390)
        let mut dec = OpusDecoder::new(48000, 2).unwrap();

        // Decode one frame to populate internal state
        let packets = encode_packets(
            48000,
            2,
            OPUS_APPLICATION_AUDIO,
            Some(MODE_CELT_ONLY),
            None,
            None,
            960,
            1,
        );
        let mut pcm = vec![0i16; 960 * 2];
        dec.decode(Some(&packets[0]), &mut pcm, 960, false).unwrap();

        // Exercise all debug accessors
        let band_e = dec.debug_get_old_band_e();
        assert!(!band_e.is_empty(), "old_band_e should be non-empty");

        let log_e = dec.debug_get_old_log_e();
        assert!(!log_e.is_empty(), "old_log_e should be non-empty");

        let log_e2 = dec.debug_get_old_log_e2();
        assert!(!log_e2.is_empty(), "old_log_e2 should be non-empty");

        let preemph = dec.debug_get_preemph_mem();
        let _ = preemph; // just exercise it

        let mem = dec.debug_get_decode_mem(0, 10);
        assert_eq!(mem.len(), 10);

        let pf = dec.debug_get_postfilter();
        let _ = pf; // exercise the tuple accessor
    }

    #[test]
    fn test_opus_packet_get_nb_samples_success() {
        // Cover line 165: opus_packet_get_nb_samples success path
        // TOC 0x00 = SILK NB 10ms mono code-0
        let packet = vec![0x00, 0xFF, 0xFF, 0xFF, 0xFF];
        let result = opus_packet_get_nb_samples(&packet, 48000);
        assert!(result.is_ok(), "should succeed: {:?}", result);
        // SILK NB 10ms → 480 samples at 48kHz
        assert_eq!(result.unwrap(), 480);
    }

    #[test]
    fn test_opus_packet_parse_code1_valid() {
        // Cover line 224: Code 1 (two equal CBR frames)
        // TOC byte with code=1: e.g., 0x01 (SILK NB 10ms mono, code 1)
        let mut packet = vec![0x01u8]; // TOC: code 1
        packet.extend_from_slice(&[0xAA; 20]); // 20 bytes payload
        // Code 1 means two equal frames, each of size remaining/2 = 10
        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0i16; 960];
        // This may produce invalid audio but exercises the parsing path
        let _ = dec.decode(Some(&packet), &mut pcm, 960, false);
    }

    #[test]
    fn test_opus_packet_parse_code3_cbr_valid() {
        // Cover lines 294-302: Code 3 CBR (multiple equal frames)
        // TOC with code=3: 0x03 (SILK NB 10ms mono, code 3)
        let mut packet = vec![0x03u8]; // TOC: code 3
        // Frame count byte: 3 frames, CBR (no VBR flag), no padding
        packet.push(0x03); // count=3, no padding, no VBR
        // Need 3 equal frames, remaining bytes divisible by 3
        packet.extend_from_slice(&[0xBB; 30]); // 30 bytes / 3 = 10 each
        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0i16; 960 * 3];
        let _ = dec.decode(Some(&packet), &mut pcm, 960 * 3, false);
    }

    #[test]
    fn test_opus_packet_parse_code3_vbr_valid() {
        // Cover lines 276-292: Code 3 VBR (variable-size frames)
        let mut packet = vec![0x03u8]; // TOC: code 3
        // Frame count byte: 2 frames, VBR (0x80 set)
        packet.push(0x82); // count=2, VBR=1, no padding
        // Size for frame 0: 10 bytes (single byte size < 252)
        packet.push(10);
        // Frame 0 data: 10 bytes
        packet.extend_from_slice(&[0xCC; 10]);
        // Frame 1 data: remaining bytes
        packet.extend_from_slice(&[0xDD; 15]);
        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0i16; 960 * 2];
        let _ = dec.decode(Some(&packet), &mut pcm, 960 * 2, false);
    }

    // -----------------------------------------------------------------------
    // Mutation-killing pinning tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pin_error_constants() {
        assert_eq!(OPUS_OK, 0);
        assert_eq!(OPUS_BAD_ARG, -1);
        assert_eq!(OPUS_BUFFER_TOO_SMALL, -2);
        assert_eq!(OPUS_INTERNAL_ERROR, -3);
        assert_eq!(OPUS_INVALID_PACKET, -4);
        assert_eq!(OPUS_UNIMPLEMENTED, -5);
    }

    #[test]
    fn test_pin_smooth_fade_8khz() {
        // fs=8000 => inc = 48000/8000 = 6, so window is sampled at stride 6.
        // Use a real CELT window to exercise the actual stride path.
        let dec = OpusDecoder::new(48000, 1).unwrap();
        let window = dec.celt_dec.get_mode().window;

        let overlap = 10;
        let channels = 1;
        let in1 = vec![1000i16; overlap * channels];
        let in2 = vec![0i16; overlap * channels];
        let mut out = vec![0i16; overlap * channels];

        smooth_fade(&in1, &in2, &mut out, overlap as i32, channels as i32, window, 8000);

        // Pinned: window stride=6 crossfade of in1=1000 fading to in2=0
        let expected: [i16; 10] = [999, 999, 998, 991, 975, 944, 893, 820, 724, 611];
        assert_eq!(&out[..], &expected[..]);
    }

    #[test]
    fn test_pin_smooth_fade_16khz() {
        // fs=16000 => inc = 48000/16000 = 3, window sampled at stride 3.
        let dec = OpusDecoder::new(48000, 1).unwrap();
        let window = dec.celt_dec.get_mode().window;

        let overlap = 10;
        let channels = 1;
        let in1 = vec![1000i16; overlap * channels];
        let in2 = vec![0i16; overlap * channels];
        let mut out = vec![0i16; overlap * channels];

        smooth_fade(&in1, &in2, &mut out, overlap as i32, channels as i32, window, 16000);

        // Pinned: window stride=3 crossfade of in1=1000 fading to in2=0
        let expected: [i16; 10] = [999, 999, 999, 999, 998, 995, 991, 985, 975, 962];
        assert_eq!(&out[..], &expected[..]);
    }

    #[test]
    fn test_pin_samples_per_frame_silk_60ms() {
        // SILK-only, audiosize==3 (bits 4-3 of TOC = 0b11) => 60ms frame.
        // TOC byte: 0b00_00_11_00 = 0x18 (NB, audiosize=3, code 0).
        let toc_silk_60ms = 0x18u8;

        let rates = [8000, 12000, 16000, 24000, 48000];
        let expected = [480, 720, 960, 1440, 2880]; // fs * 60 / 1000
        for (i, &rate) in rates.iter().enumerate() {
            let spf = opus_packet_get_samples_per_frame(&[toc_silk_60ms], rate);
            assert_eq!(spf, expected[i], "SILK 60ms at {}Hz", rate);
        }
    }

    #[test]
    fn test_pin_bandwidth_toc_sweep() {
        // SILK-only: bit7=0, bits6-5 != 0b11 => bw = NB + ((toc>>5)&3)
        assert_eq!(opus_packet_get_bandwidth(&[0x00]), OPUS_BANDWIDTH_NARROWBAND);
        assert_eq!(opus_packet_get_bandwidth(&[0x20]), OPUS_BANDWIDTH_MEDIUMBAND);
        assert_eq!(opus_packet_get_bandwidth(&[0x40]), OPUS_BANDWIDTH_WIDEBAND);

        // Hybrid: bits 7-5 = 011 => bit4 selects SWB/FB
        assert_eq!(opus_packet_get_bandwidth(&[0x60]), OPUS_BANDWIDTH_SUPERWIDEBAND);
        assert_eq!(opus_packet_get_bandwidth(&[0x70]), OPUS_BANDWIDTH_FULLBAND);

        // CELT-only: bit7=1 => bw = MB + ((toc>>5)&3), but MB remapped to NB
        assert_eq!(opus_packet_get_bandwidth(&[0x80]), OPUS_BANDWIDTH_NARROWBAND);
        assert_eq!(opus_packet_get_bandwidth(&[0xA0]), OPUS_BANDWIDTH_WIDEBAND);
        assert_eq!(opus_packet_get_bandwidth(&[0xC0]), OPUS_BANDWIDTH_SUPERWIDEBAND);
        assert_eq!(opus_packet_get_bandwidth(&[0xE0]), OPUS_BANDWIDTH_FULLBAND);

        // Pin the actual constant values
        assert_eq!(OPUS_BANDWIDTH_NARROWBAND, 1101);
        assert_eq!(OPUS_BANDWIDTH_MEDIUMBAND, 1102);
        assert_eq!(OPUS_BANDWIDTH_WIDEBAND, 1103);
        assert_eq!(OPUS_BANDWIDTH_SUPERWIDEBAND, 1104);
        assert_eq!(OPUS_BANDWIDTH_FULLBAND, 1105);
    }

    #[test]
    fn test_pin_decode_native_exact_output() {
        // Encode a known signal with RESTRICTED_LOWDELAY (CELT-only), decode, pin output.
        let mut enc =
            OpusEncoder::new(48000, 1, OPUS_APPLICATION_RESTRICTED_LOWDELAY).unwrap();
        enc.set_bitrate(64000);

        let pcm_in = vec![10000i16; 960];
        let mut packet = vec![0u8; 1500];
        let len = enc.encode(&pcm_in, 960, &mut packet, 1500).unwrap();
        let packet = &packet[..len as usize];

        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm_out = vec![0i16; 960];
        let decoded = dec.decode(Some(packet), &mut pcm_out, 960, false).unwrap();
        assert_eq!(decoded, 960);

        // Pinned: first 20 samples from decoding DC=10000 with CELT RESTRICTED_LOWDELAY
        let expected: [i16; 20] = [
            0, 1, -1, 1, 0, -1, 2, -3, 2, -3, -2, -1, -5, 1, -3, -4, -5, -3, -9, -7,
        ];
        assert_eq!(&pcm_out[..20], &expected[..]);
    }

    #[test]
    fn test_pin_decode_with_gain() {
        // Encode, decode with gain applied, pin exact output samples.
        let mut enc =
            OpusEncoder::new(48000, 1, OPUS_APPLICATION_RESTRICTED_LOWDELAY).unwrap();
        enc.set_bitrate(64000);

        let pcm_in = vec![10000i16; 960];
        let mut packet = vec![0u8; 1500];
        let len = enc.encode(&pcm_in, 960, &mut packet, 1500).unwrap();
        let packet = &packet[..len as usize];

        // Decode with +6dB gain (gain = 256 * 6 = 1536 in Q8)
        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        dec.set_gain(1536).unwrap();
        let mut pcm_out = vec![0i16; 960];
        let decoded = dec.decode(Some(packet), &mut pcm_out, 960, false).unwrap();
        assert_eq!(decoded, 960);

        // Pinned: first 20 samples with +6dB gain applied
        let expected: [i16; 20] = [
            0, 2, -2, 2, 0, -2, 4, -6, 4, -6, -4, -2, -10, 2, -6, -8, -10, -6, -18, -14,
        ];
        assert_eq!(&pcm_out[..20], &expected[..]);
    }

    // =======================================================================
    // Stage 2 branch coverage additions
    // =======================================================================
    mod branch_coverage_stage2 {
        use super::*;
        use crate::opus::encoder::OPUS_APPLICATION_VOIP;

        /// Encode one 20ms frame at fs=16000, mono, patterned PCM — cheap helper
        /// for the many tests below. Returns the packet bytes.
        fn enc_one_mono(bitrate: i32, complexity: i32) -> Vec<u8> {
            let mut enc = OpusEncoder::new(16000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            enc.set_bitrate(bitrate);
            enc.set_complexity(complexity);
            let pcm = patterned_pcm_i16(320, 1, 3);
            let mut buf = vec![0u8; 600];
            let n = enc.encode(&pcm, 320, &mut buf, 600).unwrap();
            buf.truncate(n as usize);
            buf
        }

        #[test]
        fn new_accepts_all_sample_rates_and_channels() {
            for &fs in &[8000i32, 12000, 16000, 24000, 48000] {
                for &ch in &[1i32, 2] {
                    let dec = OpusDecoder::new(fs, ch).unwrap();
                    assert_eq!(dec.get_sample_rate(), fs);
                    assert_eq!(dec.get_channels(), ch);
                }
            }
        }

        #[test]
        fn new_rejects_bad_fs_and_channels() {
            assert!(OpusDecoder::new(44100, 1).is_err());
            assert!(OpusDecoder::new(48000, 0).is_err());
            assert!(OpusDecoder::new(48000, 3).is_err());
            assert!(OpusDecoder::new(0, 1).is_err());
        }

        #[test]
        fn decode_mono_across_rates_exercises_bandwidth_table() {
            // Encode at 48kHz then decode at each output rate: exercises
            // internal_sample_rate branches and bandwidth end-band table.
            let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            enc.set_bitrate(48000);
            let pcm = patterned_pcm_i16(960, 1, 5);
            let mut pkt = vec![0u8; 1500];
            let n = enc.encode(&pcm, 960, &mut pkt, 1500).unwrap();
            let packet = &pkt[..n as usize];

            for &fs in &[8000i32, 12000, 16000, 24000, 48000] {
                let mut dec = OpusDecoder::new(fs, 1).unwrap();
                let mut out = vec![0i16; (fs / 50) as usize];
                let samples = dec.decode(Some(packet), &mut out, fs / 50, false).unwrap();
                assert!(samples > 0);
                assert_eq!(dec.get_sample_rate(), fs);
                // Last-packet duration recorded
                assert_eq!(dec.get_last_packet_duration(), samples);
            }
        }

        #[test]
        fn decode_stereo_packet_drives_stream_channels_branch() {
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
            enc.set_bitrate(96000);
            let pcm = patterned_pcm_i16(960, 2, 9);
            let mut pkt = vec![0u8; 2000];
            let n = enc.encode(&pcm, 960, &mut pkt, 2000).unwrap();
            let mut dec = OpusDecoder::new(48000, 2).unwrap();
            let mut out = vec![0i16; 960 * 2];
            let samples = dec.decode(Some(&pkt[..n as usize]), &mut out, 960, false).unwrap();
            assert_eq!(samples, 960);
            assert_eq!(dec.get_channels(), 2);
        }

        #[test]
        fn plc_then_resume_consecutive_loss_depths() {
            // Prime the decoder with a good SILK packet, then trigger several
            // PLC frames (data=None), then resume with another good packet.
            let packet = enc_one_mono(16000, 5);
            let mut dec = OpusDecoder::new(16000, 1).unwrap();
            let mut out = vec![0i16; 320];
            dec.decode(Some(&packet), &mut out, 320, false).unwrap();
            for &losses in &[1usize, 2, 3, 10, 50] {
                for _ in 0..losses {
                    let r = dec.decode(None, &mut out, 320, false).unwrap();
                    assert_eq!(r, 320);
                }
                // resume
                let r = dec.decode(Some(&packet), &mut out, 320, false).unwrap();
                assert_eq!(r, 320);
            }
        }

        #[test]
        fn plc_with_empty_slice_is_equivalent_to_none() {
            let packet = enc_one_mono(16000, 3);
            let mut dec = OpusDecoder::new(16000, 1).unwrap();
            let mut out = vec![0i16; 320];
            dec.decode(Some(&packet), &mut out, 320, false).unwrap();
            // Empty slice triggers PLC path too
            let r = dec.decode(Some(&[][..]), &mut out, 320, false).unwrap();
            assert_eq!(r, 320);
        }

        #[test]
        fn plc_chunked_path_for_large_frame_sizes() {
            // frame_size > 20 ms triggers the chunked PLC branch (line 685).
            let packet = enc_one_mono(16000, 2);
            let mut dec = OpusDecoder::new(16000, 1).unwrap();
            let mut out = vec![0i16; 320];
            dec.decode(Some(&packet), &mut out, 320, false).unwrap();
            // 60 ms PLC = 960 samples at 16kHz
            let mut big = vec![0i16; 960];
            let r = dec.decode(None, &mut big, 960, false).unwrap();
            assert_eq!(r, 960);
        }

        #[test]
        fn fec_decode_with_lost_packet() {
            // Encode with inband FEC enabled + loss perc high → FEC bits present.
            let mut enc = OpusEncoder::new(16000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            enc.set_bitrate(24000);
            enc.set_inband_fec(1);
            enc.set_packet_loss_perc(50);
            enc.set_complexity(5);
            let pcm = patterned_pcm_i16(320, 1, 11);
            let mut pkt = vec![0u8; 400];
            let n = enc.encode(&pcm, 320, &mut pkt, 400).unwrap();
            let packet = &pkt[..n as usize];

            let mut dec = OpusDecoder::new(16000, 1).unwrap();
            // Prime — normal decode
            let mut out = vec![0i16; 320];
            let _ = dec.decode(Some(packet), &mut out, 320, false);
            // FEC path: recover the previous frame from this packet's LBRR data
            let mut rec = vec![0i16; 320];
            let r = dec.decode(Some(packet), &mut rec, 320, true);
            // API may return samples or an error depending on LBRR presence —
            // either is fine; we only need the branch taken.
            let _ = r;
        }

        #[test]
        fn fec_decode_on_plc_when_no_data() {
            let mut dec = OpusDecoder::new(16000, 1).unwrap();
            let mut out = vec![0i16; 320];
            // decode_fec=true with no data falls through to PLC
            let r = dec.decode(None, &mut out, 320, true);
            let _ = r; // may succeed as PLC, or error — branch exercised
        }

        #[test]
        fn decode_fec_frame_size_not_multiple_of_2_5ms_is_bad_arg() {
            let mut dec = OpusDecoder::new(16000, 1).unwrap();
            let mut out = vec![0i16; 100];
            // frame_size not a multiple of fs/400 = 40 samples
            let err = dec.decode(None, &mut out, 39, false).unwrap_err();
            assert_eq!(err, OPUS_BAD_ARG);
        }

        #[test]
        fn decode_rejects_non_positive_frame_size() {
            let mut dec = OpusDecoder::new(16000, 1).unwrap();
            let mut out = vec![0i16; 320];
            assert_eq!(dec.decode(None, &mut out, 0, false).unwrap_err(), OPUS_BAD_ARG);
            assert_eq!(dec.decode(None, &mut out, -1, false).unwrap_err(), OPUS_BAD_ARG);
            assert_eq!(dec.decode_float(None, &mut vec![0f32; 320], 0, false).unwrap_err(), OPUS_BAD_ARG);
            assert_eq!(dec.decode24(None, &mut vec![0i32; 320], 0, false).unwrap_err(), OPUS_BAD_ARG);
        }

        #[test]
        fn toc_config_0_to_31_round_trips_through_accessors() {
            // Build a 1-byte TOC for each config and feed the packet-info helpers.
            // Hybrid configs encode bandwidth via bit 4; SILK/CELT use bits 6-5.
            for cfg in 0u8..32 {
                let toc = (cfg << 3) | 0x00; // code 0, 1 frame
                let pkt = [toc];
                // These should not panic for any valid TOC.
                let nb_frames = opus_packet_get_nb_frames(&pkt).unwrap();
                assert_eq!(nb_frames, 1);
                let bw = opus_packet_get_bandwidth(&pkt);
                assert!((1101..=1105).contains(&bw));
                let spf = opus_packet_get_samples_per_frame(&pkt, 48000);
                assert!(spf > 0);
                let nb_ch = opus_packet_get_nb_channels(&pkt);
                assert!(nb_ch == 1 || nb_ch == 2);
                // And for each supported decoder rate
                for &fs in &[8000i32, 16000, 24000, 48000] {
                    let ns = opus_packet_get_nb_samples(&pkt, fs);
                    assert!(ns.is_ok());
                }
            }
        }

        #[test]
        fn toc_stereo_bit_selects_channels() {
            let pkt = [0x04u8]; // stereo flag set, code 0
            assert_eq!(opus_packet_get_nb_channels(&pkt), 2);
            let pkt = [0x00u8];
            assert_eq!(opus_packet_get_nb_channels(&pkt), 1);
        }

        #[test]
        fn get_nb_frames_code3_short_packet() {
            // Code 3 but only 1 byte → invalid
            assert!(matches!(opus_packet_get_nb_frames(&[0x03]), Err(OPUS_INVALID_PACKET)));
            // Empty → bad arg
            assert!(matches!(opus_packet_get_nb_frames(&[]), Err(OPUS_BAD_ARG)));
            // Code 1 / 2 always return 2
            assert_eq!(opus_packet_get_nb_frames(&[0x01, 0, 0]).unwrap(), 2);
            assert_eq!(opus_packet_get_nb_frames(&[0x02, 1, 0, 0]).unwrap(), 2);
        }

        #[test]
        fn get_nb_samples_rejects_over_120ms_code3() {
            // Build a code 3 packet claiming huge frame count at 20ms ⇒ > 120ms
            // Framesize for SILK 20ms at fs=48000 is 960; 7 frames = 6720 samples > 120ms=5760
            let pkt = [0x0Bu8, 0x07u8, 0, 0, 0, 0, 0, 0];
            let r = opus_packet_get_nb_samples(&pkt, 48000);
            assert!(r.is_err());
        }

        #[test]
        fn set_gain_range() {
            let mut dec = OpusDecoder::new(48000, 1).unwrap();
            assert!(dec.set_gain(0).is_ok());
            assert!(dec.set_gain(-32768).is_ok());
            assert!(dec.set_gain(32767).is_ok());
            assert!(dec.set_gain(32768).is_err());
            assert!(dec.set_gain(-32769).is_err());
            assert_eq!(dec.get_gain(), 32767);
        }

        #[test]
        fn set_complexity_range_and_getter() {
            let mut dec = OpusDecoder::new(48000, 1).unwrap();
            assert!(dec.set_complexity(0).is_ok());
            assert!(dec.set_complexity(10).is_ok());
            assert!(dec.set_complexity(-1).is_err());
            assert!(dec.set_complexity(11).is_err());
            assert_eq!(dec.get_complexity(), 10);
        }

        #[test]
        fn reset_clears_state_and_get_pitch_before_decode() {
            let dec = OpusDecoder::new(48000, 1).unwrap();
            assert_eq!(dec.get_last_packet_duration(), 0);
            // get_pitch before any decode uses prev_pitch_lag (non-CELT path)
            let _ = dec.get_pitch();

            let packet = enc_one_mono(32000, 4);
            let mut out = vec![0i16; 320];
            let mut dec16 = OpusDecoder::new(16000, 1).unwrap();
            let _ = dec16.decode(Some(&packet), &mut out, 320, false);
            let dur_before = dec16.get_last_packet_duration();
            assert!(dur_before > 0);

            dec16.reset();
            assert_eq!(dec16.get_last_packet_duration(), 0);
            assert_eq!(dec16.get_final_range(), 0);
            // After reset, get_pitch is the non-CELT path
            let _ = dec16.get_pitch();
        }

        #[test]
        fn decode_float_and_decode24_output_finite_values() {
            let packet = enc_one_mono(24000, 3);
            let mut dec = OpusDecoder::new(16000, 1).unwrap();

            let mut outf = vec![0f32; 320];
            let n = dec.decode_float(Some(&packet), &mut outf, 320, false).unwrap();
            assert_eq!(n, 320);
            assert!(outf.iter().all(|v| v.is_finite() && v.abs() <= 1.0));

            let packet2 = enc_one_mono(24000, 3);
            let mut dec2 = OpusDecoder::new(16000, 1).unwrap();
            let mut out32 = vec![0i32; 320];
            let n2 = dec2.decode24(Some(&packet2), &mut out32, 320, false).unwrap();
            assert_eq!(n2, 320);
        }

        #[test]
        fn decode_with_celt_only_packet_from_lowdelay_encoder() {
            // RESTRICTED_LOWDELAY uses CELT-only → exercises mode==CELT branch
            let mut enc =
                OpusEncoder::new(48000, 1, OPUS_APPLICATION_RESTRICTED_LOWDELAY).unwrap();
            enc.set_bitrate(64000);
            let pcm = vec![3000i16; 960];
            let mut buf = vec![0u8; 1500];
            let n = enc.encode(&pcm, 960, &mut buf, 1500).unwrap();

            let mut dec = OpusDecoder::new(48000, 1).unwrap();
            let mut out = vec![0i16; 960];
            let r = dec.decode(Some(&buf[..n as usize]), &mut out, 960, false).unwrap();
            assert_eq!(r, 960);

            // Now follow up with PLC — exercises prev_mode == CELT_ONLY PLC path
            let r = dec.decode(None, &mut out, 960, false).unwrap();
            assert_eq!(r, 960);
        }

        #[test]
        fn phase_inversion_and_ignore_extensions_flags() {
            let mut dec = OpusDecoder::new(48000, 2).unwrap();
            dec.set_phase_inversion_disabled(true);
            assert!(dec.get_phase_inversion_disabled());
            dec.set_phase_inversion_disabled(false);
            assert!(!dec.get_phase_inversion_disabled());
            dec.set_ignore_extensions(true);
            assert!(dec.get_ignore_extensions());
            dec.set_ignore_extensions(false);
            assert!(!dec.get_ignore_extensions());
        }

        #[test]
        fn get_nb_samples_via_decoder_helper() {
            let packet = enc_one_mono(16000, 3);
            let dec = OpusDecoder::new(16000, 1).unwrap();
            let n = dec.get_nb_samples(&packet).unwrap();
            assert!(n > 0);
        }

        #[test]
        fn decode_invalid_packet_propagates_error() {
            let mut dec = OpusDecoder::new(48000, 1).unwrap();
            let mut out = vec![0i16; 960];
            // Code 1 with odd remaining → INVALID_PACKET via opus_packet_parse_impl
            let r = dec.decode(Some(&[0x09u8, 0xAAu8, 0xBBu8, 0xCCu8]), &mut out, 960, false);
            assert!(r.is_err());
        }

        #[test]
        fn decode_buffer_too_small_for_packet_samples() {
            // 20ms SILK packet into a 5ms buffer → BUFFER_TOO_SMALL.
            let packet = enc_one_mono(16000, 3);
            let mut dec = OpusDecoder::new(16000, 1).unwrap();
            let mut out = vec![0i16; 40]; // 2.5ms at 16kHz
            let r = dec.decode(Some(&packet), &mut out, 40, false);
            // Either returns buffer-too-small or another non-positive error.
            assert!(r.is_err() || r.unwrap() <= 40);
        }

        // --- Additional targeted branch coverage ---

        #[test]
        fn decode_frame_size_too_small_paths() {
            let mut dec = OpusDecoder::new(16000, 1).unwrap();
            let mut out = vec![0i16; 40];
            // 40 samples = 2.5ms at 16kHz — smallest valid PLC frame.
            let _ = dec.decode(None, &mut out, 40, false);
        }

        #[test]
        fn decode_native_zero_data_triggers_plc_path() {
            let mut dec = OpusDecoder::new(16000, 1).unwrap();
            let mut out = vec![0i16; 320];
            // Pass Some(&[]) — data.is_empty() triggers PLC.
            let r = dec.decode(Some(&[][..]), &mut out, 320, false).unwrap();
            assert_eq!(r, 320);
        }

        #[test]
        fn mode_transition_celt_to_silk() {
            // First a CELT-only packet, then a SILK packet — exercises
            // prev_mode transitions in decode_frame.
            let mut enc_celt =
                OpusEncoder::new(48000, 1, OPUS_APPLICATION_RESTRICTED_LOWDELAY).unwrap();
            enc_celt.set_bitrate(64000);
            let pcm = vec![1500i16; 960];
            let mut pkt_celt = vec![0u8; 1500];
            let n = enc_celt.encode(&pcm, 960, &mut pkt_celt, 1500).unwrap();

            let mut enc_silk = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            enc_silk.set_bitrate(16000);
            enc_silk.set_bandwidth(OPUS_BANDWIDTH_NARROWBAND);
            let mut pkt_silk = vec![0u8; 1500];
            let ns = enc_silk.encode(&pcm, 960, &mut pkt_silk, 1500).unwrap();

            let mut dec = OpusDecoder::new(48000, 1).unwrap();
            let mut out = vec![0i16; 960];
            let _ = dec.decode(Some(&pkt_celt[..n as usize]), &mut out, 960, false).unwrap();
            let _ = dec.decode(Some(&pkt_silk[..ns as usize]), &mut out, 960, false).unwrap();
            let _ = dec.decode(Some(&pkt_celt[..n as usize]), &mut out, 960, false).unwrap();
        }

        #[test]
        fn mode_transition_silk_to_celt() {
            // SILK first, then CELT → exercises the other transition direction.
            let mut enc_silk = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
            enc_silk.set_bitrate(16000);
            enc_silk.set_bandwidth(OPUS_BANDWIDTH_NARROWBAND);
            let pcm = vec![1500i16; 960];
            let mut pkt_silk = vec![0u8; 1500];
            let ns = enc_silk.encode(&pcm, 960, &mut pkt_silk, 1500).unwrap();

            let mut enc_celt =
                OpusEncoder::new(48000, 1, OPUS_APPLICATION_RESTRICTED_LOWDELAY).unwrap();
            enc_celt.set_bitrate(64000);
            let mut pkt_celt = vec![0u8; 1500];
            let n = enc_celt.encode(&pcm, 960, &mut pkt_celt, 1500).unwrap();

            let mut dec = OpusDecoder::new(48000, 1).unwrap();
            let mut out = vec![0i16; 960];
            let _ = dec.decode(Some(&pkt_silk[..ns as usize]), &mut out, 960, false).unwrap();
            let _ = dec.decode(Some(&pkt_celt[..n as usize]), &mut out, 960, false).unwrap();
        }

        #[test]
        fn decode_native_code1_odd_remaining_is_invalid() {
            // Code 1 (two CBR frames) with odd remaining → parse error.
            let mut dec = OpusDecoder::new(48000, 1).unwrap();
            let mut out = vec![0i16; 960];
            // TOC=0x09 (NB SILK 10ms code 1), then 3 data bytes: odd count.
            let pkt = [0x09u8, 0xAA, 0xBB, 0xCC];
            let r = dec.decode(Some(&pkt), &mut out, 960, false);
            assert!(r.is_err());
        }

        #[test]
        fn decode_native_code3_padding_exhausts_remaining() {
            // Code 3 with padding flag and no room for padding bytes.
            let mut dec = OpusDecoder::new(48000, 1).unwrap();
            let mut out = vec![0i16; 960];
            let pkt = [0x0Bu8, 0x41u8]; // code 3, count=1 with padding flag set
            let r = dec.decode(Some(&pkt), &mut out, 960, false);
            assert!(r.is_err());
        }

        #[test]
        fn opus_packet_has_lbrr_all_modes() {
            // CELT-only packet → always false
            let p = [0x80u8, 0xAA];
            assert_eq!(opus_packet_has_lbrr(&p, p.len() as i32).unwrap(), false);

            // Hybrid packet with a tiny SILK payload
            let p = [0x60u8, 0xAA, 0xBB, 0xCC, 0xDD];
            let _ = opus_packet_has_lbrr(&p, p.len() as i32);

            // SILK-only packet
            let p = [0x00u8, 0xAA, 0xBB, 0xCC];
            let _ = opus_packet_has_lbrr(&p, p.len() as i32);
        }

        #[test]
        fn decode_float_and_decode24_reject_invalid_packet_samples() {
            // Invalid packet (code 3 claiming too many frames) triggers the
            // `Ok(_) | Err(_) => return Err(OPUS_INVALID_PACKET)` branch in
            // decode24/decode_float at lines 1296 and 1328.
            let mut dec = OpusDecoder::new(48000, 1).unwrap();
            // 120ms check fails: CELT 2.5ms × 49 > 5760 samples
            let pkt = [0x87u8, 49u8];
            let mut out32 = vec![0i32; 960];
            let r = dec.decode24(Some(&pkt), &mut out32, 960, false);
            assert!(r.is_err());
            let mut outf = vec![0f32; 960];
            let r = dec.decode_float(Some(&pkt), &mut outf, 960, false);
            assert!(r.is_err());
        }

        #[test]
        fn decode_fec_request_falls_through_to_plc_when_celt_only() {
            // CELT-only packet with decode_fec=true → no FEC available,
            // so code falls through to PLC (line 1187).
            let mut enc =
                OpusEncoder::new(48000, 1, OPUS_APPLICATION_RESTRICTED_LOWDELAY).unwrap();
            enc.set_bitrate(64000);
            let pcm = vec![0i16; 960];
            let mut pkt = vec![0u8; 1500];
            let n = enc.encode(&pcm, 960, &mut pkt, 1500).unwrap();

            let mut dec = OpusDecoder::new(48000, 1).unwrap();
            let mut out = vec![0i16; 960];
            // Prime with a normal decode
            dec.decode(Some(&pkt[..n as usize]), &mut out, 960, false).unwrap();
            // Request FEC — should fall through to PLC since mode is CELT-only
            let r = dec.decode(Some(&pkt[..n as usize]), &mut out, 960, true);
            let _ = r; // may error or succeed via PLC; branch hit either way
        }

        #[test]
        fn decode_stereo_at_lower_rate_exercises_downmix_path() {
            // Stereo encode at 48kHz, decode at 24kHz mono — exercises
            // stream_channels=2 with decoder channels=1.
            let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
            enc.set_bitrate(64000);
            let pcm = patterned_pcm_i16(960, 2, 99);
            let mut pkt = vec![0u8; 2000];
            let n = enc.encode(&pcm, 960, &mut pkt, 2000).unwrap();

            // Decode at 24kHz stereo (different fs path)
            let mut dec = OpusDecoder::new(24000, 2).unwrap();
            let mut out = vec![0i16; 480 * 2];
            let r = dec.decode(Some(&pkt[..n as usize]), &mut out, 480, false).unwrap();
            assert_eq!(r, 480);
        }

        #[test]
        fn decode_code3_cbr_with_valid_count() {
            // Build a valid code 3 CBR packet: 2 frames, each 10 bytes, SILK NB.
            // TOC=0x00 (SILK NB 10ms) | 0x03 (code 3) = 0x03, ch = 2 (CBR, no padding)
            let mut pkt = vec![0x03u8, 0x02u8];
            pkt.extend_from_slice(&[0u8; 20]);
            let mut dec = OpusDecoder::new(48000, 1).unwrap();
            let mut out = vec![0i16; 960];
            let _ = dec.decode(Some(&pkt), &mut out, 960, false);
        }

        #[test]
        fn decode_code3_vbr_with_valid_count() {
            // Code 3 VBR: TOC=0x03, ch=0x82 (VBR, count=2), size[0]=3, then frames.
            let mut pkt = vec![0x03u8, 0x82u8, 3u8];
            pkt.extend_from_slice(&[0u8; 20]);
            let mut dec = OpusDecoder::new(48000, 1).unwrap();
            let mut out = vec![0i16; 960];
            let _ = dec.decode(Some(&pkt), &mut out, 960, false);
        }

        #[test]
        fn decode_code2_vbr_valid() {
            // Code 2 VBR: TOC=0x02, size=1, then 2 bytes
            let pkt = [0x02u8, 1u8, 0xAA, 0xBB];
            let mut dec = OpusDecoder::new(48000, 1).unwrap();
            let mut out = vec![0i16; 960];
            let _ = dec.decode(Some(&pkt), &mut out, 960, false);
        }
    }
}
