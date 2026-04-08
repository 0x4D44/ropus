//! Opus Decoder — top-level Opus decoding entry point.
//!
//! Ported from: reference/src/opus_decoder.c, reference/src/opus.c
//! Fixed-point path (non-RES24, non-QEXT, non-DNN).

use crate::celt::decoder::CeltDecoder;
use crate::celt::math_ops::celt_exp2;
use crate::celt::range_coder::RangeDecoder;
use crate::silk::decoder::{SilkDecControl, SilkDecoder, silk_decode};
use crate::types::*;

/// Create a no-op CELT DNN PLC argument (None when dnn is enabled, () when disabled).
/// Used for CELT decode calls that always have data (redundancy, silence frames).
#[inline(always)]
fn celt_lpcnet_noop<'a>() -> crate::celt::decoder::DnnPlcArg<'a> {
    #[cfg(feature = "dnn")]
    {
        None
    }
    #[cfg(not(feature = "dnn"))]
    {
        ()
    }
}

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
const MAX_FRAMES: usize = 48;

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

/// Internal packet parse implementation.
/// Returns frame count (positive) or negative error code.
/// Matches C `opus_packet_parse_impl`.
fn opus_packet_parse_impl(
    data: &[u8],
    len: i32,
    self_delimited: bool,
    out_toc: &mut u8,
    sizes: &mut [i16; MAX_FRAMES],
    payload_offset: &mut i32,
    packet_offset: Option<&mut i32>,
) -> i32 {
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

/// Placeholder for DRED (Deep REDundancy) state.
/// Full DRED decoding requires the rdovae module which is not yet ported.
/// This struct provides the API surface for future integration.
#[cfg(feature = "dnn")]
pub struct DredState {
    pub fec_features: Vec<f32>,
    pub nb_latents: i32,
    pub dred_offset: i32,
    pub process_stage: i32,
}

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

    // --- DNN neural PLC state (behind feature flag) ---
    #[cfg(feature = "dnn")]
    lpcnet: crate::dnn::lpcnet::LPCNetPLCState,
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
            #[cfg(feature = "dnn")]
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
            #[cfg(feature = "dnn")]
            lpcnet: crate::dnn::lpcnet::LPCNetPLCState::new(),
        };

        // CELT signalling off (Opus handles framing)
        dec.celt_dec.set_signalling(false);

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
        #[cfg(feature = "dnn")]
        self.lpcnet.reset();
    }

    /// Load DNN model weights for neural PLC.
    /// Matches C `OPUS_SET_DNN_BLOB_REQUEST` CTL.
    /// Returns `Ok(())` on success, `Err(error_code)` on failure.
    #[cfg(feature = "dnn")]
    pub fn set_dnn_blob(&mut self, data: &[u8]) -> Result<(), i32> {
        let ret = self.lpcnet.load_model(data);
        if ret == 0 { Ok(()) } else { Err(ret) }
    }

    /// Decode with DRED FEC support.
    /// When DRED features are available and the packet is missing, feeds FEC
    /// features to the LPCNet PLC before the decode/PLC path.
    /// Matches C `opus_decode_native` with DRED support.
    ///
    /// Note: Full DRED support requires the DRED decoder (rdovae), which is
    /// not yet ported. This method provides the API wiring for future use.
    #[cfg(feature = "dnn")]
    pub fn decode_with_dred(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i16],
        frame_size: i32,
        dred: Option<&DredState>,
        _dred_offset: i32,
    ) -> Result<i32, i32> {
        // If DRED state is provided and the packet is missing, feed FEC features
        if data.is_none() {
            if let Some(dred) = dred {
                // Feed DRED FEC features to LPCNet PLC
                let nb_features = dred.nb_latents as usize;
                for i in 0..nb_features {
                    let start = i * crate::dnn::lpcnet::NB_FEATURES;
                    let end = start + crate::dnn::lpcnet::NB_FEATURES;
                    if end <= dred.fec_features.len() {
                        self.lpcnet.fec_add(Some(&dred.fec_features[start..end]));
                    }
                }
            }
        }

        // Delegate to the normal decode path.
        // Note: fec_clear() is NOT called here. FEC features are consumed
        // incrementally via fec_read_pos during PLC. The caller is responsible
        // for clearing stale FEC data when a good packet arrives.
        self.decode(data, pcm, frame_size, false)
    }

    /// Clear DRED FEC buffer.
    #[cfg(feature = "dnn")]
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

                #[cfg(feature = "dnn")]
                let silk_lpcnet: crate::silk::decoder::DnnPlcArg<'_> = Some(&mut self.lpcnet);
                #[cfg(not(feature = "dnn"))]
                let silk_lpcnet: crate::silk::decoder::DnnPlcArg<'_> = ();

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
                celt_lpcnet_noop(),
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
            // Pass lpcnet for CELT PLC (neural concealment on lost frames).
            #[cfg(feature = "dnn")]
            let celt_lpcnet: crate::celt::decoder::DnnPlcArg<'_> = Some(&mut self.lpcnet);
            #[cfg(not(feature = "dnn"))]
            let celt_lpcnet: crate::celt::decoder::DnnPlcArg<'_> = ();

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
                    celt_lpcnet_noop(),
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
                celt_lpcnet_noop(),
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
            let frame_data = &payload[data_offset..data_offset + sizes[i] as usize];
            let ret = self.decode_frame(
                Some(frame_data),
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
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "dnn")]
    use crate::dnn::core::{WEIGHT_BLOCK_SIZE, WEIGHT_BLOB_VERSION, WEIGHT_TYPE_FLOAT};
    use crate::opus::encoder::{
        OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY, OpusEncoder,
    };

    #[cfg(feature = "dnn")]
    fn append_float_record(blob: &mut Vec<u8>, name: &str, count: usize) {
        let payload_len = count * 4;
        let mut record = vec![0u8; WEIGHT_BLOCK_SIZE + payload_len];
        record[4..8].copy_from_slice(&WEIGHT_BLOB_VERSION.to_ne_bytes());
        record[8..12].copy_from_slice(&WEIGHT_TYPE_FLOAT.to_ne_bytes());
        record[12..16].copy_from_slice(&(payload_len as i32).to_ne_bytes());
        record[16..20].copy_from_slice(&(payload_len as i32).to_ne_bytes());
        record[20..20 + name.len()].copy_from_slice(name.as_bytes());
        blob.extend_from_slice(&record);
    }

    #[cfg(feature = "dnn")]
    fn append_int8_record(blob: &mut Vec<u8>, name: &str, count: usize) {
        let payload_len = count;
        let mut record = vec![0u8; WEIGHT_BLOCK_SIZE + payload_len];
        record[4..8].copy_from_slice(&WEIGHT_BLOB_VERSION.to_ne_bytes());
        record[8..12].copy_from_slice(&crate::dnn::core::WEIGHT_TYPE_INT8.to_ne_bytes());
        record[12..16].copy_from_slice(&(payload_len as i32).to_ne_bytes());
        record[16..20].copy_from_slice(&(payload_len as i32).to_ne_bytes());
        record[20..20 + name.len()].copy_from_slice(name.as_bytes());
        blob.extend_from_slice(&record);
    }

    #[cfg(feature = "dnn")]
    fn make_pitchdnn_weight_blob() -> Vec<u8> {
        let mut blob = Vec::new();

        for name in [
            "dense_if_upsampler_1_bias",
            "dense_if_upsampler_1_subias",
            "dense_if_upsampler_1_scale",
        ] {
            append_float_record(&mut blob, name, 64);
        }
        append_int8_record(&mut blob, "dense_if_upsampler_1_weights_int8", 88 * 64);

        for name in [
            "dense_if_upsampler_2_bias",
            "dense_if_upsampler_2_subias",
            "dense_if_upsampler_2_scale",
        ] {
            append_float_record(&mut blob, name, 64);
        }
        append_int8_record(&mut blob, "dense_if_upsampler_2_weights_int8", 64 * 64);

        append_float_record(&mut blob, "conv2d_1_bias", 4);
        append_float_record(&mut blob, "conv2d_1_weight_float", 4 * 3 * 3);
        append_float_record(&mut blob, "conv2d_2_bias", 1);
        append_float_record(&mut blob, "conv2d_2_weight_float", 4 * 3 * 3);

        for name in [
            "dense_downsampler_bias",
            "dense_downsampler_subias",
            "dense_downsampler_scale",
        ] {
            append_float_record(&mut blob, name, 64);
        }
        append_int8_record(&mut blob, "dense_downsampler_weights_int8", 288 * 64);

        for name in [
            "gru_1_input_bias",
            "gru_1_input_subias",
            "gru_1_input_scale",
        ] {
            append_float_record(&mut blob, name, 3 * 64);
        }
        append_int8_record(&mut blob, "gru_1_input_weights_int8", 64 * 3 * 64);

        for name in [
            "gru_1_recurrent_bias",
            "gru_1_recurrent_subias",
            "gru_1_recurrent_scale",
        ] {
            append_float_record(&mut blob, name, 3 * 64);
        }
        append_int8_record(&mut blob, "gru_1_recurrent_weights_int8", 64 * 3 * 64);

        for name in [
            "dense_final_upsampler_bias",
            "dense_final_upsampler_subias",
            "dense_final_upsampler_scale",
        ] {
            append_float_record(&mut blob, name, 3 * 64);
        }
        append_int8_record(&mut blob, "dense_final_upsampler_weights_int8", 64 * 3 * 64);

        blob
    }

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
        assert_eq!(decf.decode_float(None, &mut plc_f, 960, false).unwrap(), 960);

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
        assert_eq!(dec.decode(Some(silk_packet), &mut first, 960, false).unwrap(), 960);
        assert_eq!(dec.prev_mode, MODE_SILK_ONLY);

        assert_eq!(dec.set_gain(256), Ok(()));
        let mut second = vec![0i16; 960];
        assert_eq!(dec.decode(Some(celt_packet), &mut second, 960, false).unwrap(), 960);
        assert_eq!(dec.prev_mode, MODE_CELT_ONLY);
        assert!(second.iter().any(|&sample| sample != 0));
        assert_ne!(dec.get_final_range(), 0);

        let mut third = vec![0i16; 960];
        assert_eq!(dec.decode(Some(silk_packet), &mut third, 960, false).unwrap(), 960);
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
    #[cfg(feature = "dnn")]
    fn test_dred_and_debug_accessor_paths() {
        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        assert_eq!(dec.set_dnn_blob(&[1, 2, 3]), Err(-1));

        let blob = make_pitchdnn_weight_blob();
        assert_eq!(dec.set_dnn_blob(&blob), Ok(()));

        let dred = DredState {
            fec_features: vec![0.5; crate::dnn::lpcnet::NB_FEATURES + 3],
            nb_latents: 2,
            dred_offset: 0,
            process_stage: 0,
        };
        let mut pcm = vec![0i16; 120];
        assert_eq!(
            dec.decode_with_dred(None, &mut pcm, 120, Some(&dred), 0),
            Ok(120)
        );
        dec.fec_clear();

        assert_eq!(dec.get_final_range(), 0);
        assert!(!dec.debug_get_old_band_e().is_empty());
        assert!(!dec.debug_get_old_log_e().is_empty());
        assert!(!dec.debug_get_old_log_e2().is_empty());
        assert_eq!(dec.debug_get_preemph_mem(), [0, 0]);
        assert_eq!(dec.debug_get_decode_mem(0, 4).len(), 4);
        assert_eq!(dec.debug_get_postfilter(), (0, 0, 0, 0, 0, 0));

        dec.prev_mode = MODE_CELT_ONLY;
        assert_eq!(dec.get_pitch(), dec.celt_dec.get_pitch());
        dec.prev_mode = MODE_SILK_ONLY;
        dec.dec_control.prev_pitch_lag = 123;
        assert_eq!(dec.get_pitch(), 123);
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
            assert!(
                (out[i] as i32 - in2[i] as i32).unsigned_abs() <= 1,
                "out[{}] = {}, expected ~{}",
                i,
                out[i],
                in2[i]
            );
        }
    }
}
