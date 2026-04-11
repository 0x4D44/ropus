//! Opus Multistream and Projection — multi-channel audio multiplexing.
//!
//! Ported from:
//!   reference/src/opus_multistream.c
//!   reference/src/opus_multistream_encoder.c
//!   reference/src/opus_multistream_decoder.c
//!   reference/src/opus_projection_encoder.c
//!   reference/src/opus_projection_decoder.c
//!   reference/src/mapping_matrix.c
//!
//! Fixed-point path (FIXED_POINT, non-RES24, non-QEXT).

use crate::celt::bands::compute_band_energies;
use crate::celt::encoder::{celt_preemphasis, clt_mdct_forward, get_fft_state};
use crate::celt::math_ops::{celt_log2, isqrt32};
use crate::celt::modes::{bitrate_to_bits, bits_to_bitrate, resampling_factor, CELTMode};
use crate::celt::quant_bands::amp2log2;
use crate::types::*;

use super::decoder::{
    opus_packet_get_nb_samples, OpusDecoder, MODE_CELT_ONLY, OPUS_BAD_ARG, OPUS_BUFFER_TOO_SMALL,
    OPUS_INTERNAL_ERROR, OPUS_INVALID_PACKET, OPUS_OK, OPUS_UNIMPLEMENTED,
};
use super::decoder::{
    OPUS_BANDWIDTH_FULLBAND, OPUS_BANDWIDTH_NARROWBAND, OPUS_BANDWIDTH_SUPERWIDEBAND,
    OPUS_BANDWIDTH_WIDEBAND,
};
use super::encoder::{
    frame_size_select, OpusEncoder, OPUS_AUTO, OPUS_BITRATE_MAX, OPUS_FRAMESIZE_ARG,
};
use super::repacketizer::OpusRepacketizer;

// ===========================================================================
// Constants
// ===========================================================================

const MAX_OVERLAP: usize = 120;

/// Max size in case the encoder decides to return six frames (6 x 20 ms = 120 ms).
const MS_FRAME_TMP: usize = 6 * 1275 + 12;

// ===========================================================================
// MappingType
// ===========================================================================

/// Multistream mapping type — controls surround processing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MappingType {
    /// No surround processing (family 0, 255).
    None,
    /// Surround analysis enabled (family 1, channels > 2).
    Surround,
    /// Ambisonics mode (family 2).
    Ambisonics,
}

// ===========================================================================
// ChannelLayout
// ===========================================================================

/// Channel layout — maps output channels to streams.
#[derive(Clone)]
pub struct ChannelLayout {
    pub nb_channels: i32,
    pub nb_streams: i32,
    pub nb_coupled_streams: i32,
    pub mapping: [u8; 256],
}

impl ChannelLayout {
    fn new() -> Self {
        Self {
            nb_channels: 0,
            nb_streams: 0,
            nb_coupled_streams: 0,
            mapping: [0u8; 256],
        }
    }
}

/// Validate that all mapping entries reference valid stream channels.
/// Matches C `validate_layout`.
fn validate_layout(layout: &ChannelLayout) -> bool {
    let max_channel = layout.nb_streams + layout.nb_coupled_streams;
    if max_channel > 255 {
        return false;
    }
    for i in 0..layout.nb_channels as usize {
        if layout.mapping[i] >= max_channel as u8 && layout.mapping[i] != 255 {
            return false;
        }
    }
    true
}

/// Find the output channel mapped to the left of coupled stream `stream_id`,
/// starting the search after `prev`. Returns -1 if not found.
/// Matches C `get_left_channel`.
fn get_left_channel(layout: &ChannelLayout, stream_id: i32, prev: i32) -> i32 {
    let start = if prev < 0 { 0 } else { prev + 1 };
    for i in start..layout.nb_channels {
        if layout.mapping[i as usize] == (stream_id * 2) as u8 {
            return i;
        }
    }
    -1
}

/// Find the output channel mapped to the right of coupled stream `stream_id`,
/// starting the search after `prev`. Returns -1 if not found.
/// Matches C `get_right_channel`.
fn get_right_channel(layout: &ChannelLayout, stream_id: i32, prev: i32) -> i32 {
    let start = if prev < 0 { 0 } else { prev + 1 };
    for i in start..layout.nb_channels {
        if layout.mapping[i as usize] == (stream_id * 2 + 1) as u8 {
            return i;
        }
    }
    -1
}

/// Find the output channel mapped to mono stream `stream_id`,
/// starting the search after `prev`. Returns -1 if not found.
/// Matches C `get_mono_channel`.
fn get_mono_channel(layout: &ChannelLayout, stream_id: i32, prev: i32) -> i32 {
    let start = if prev < 0 { 0 } else { prev + 1 };
    for i in start..layout.nb_channels {
        if layout.mapping[i as usize] == (stream_id + layout.nb_coupled_streams) as u8 {
            return i;
        }
    }
    -1
}

/// Validate that every stream has at least one channel assigned.
/// Matches C `validate_encoder_layout`.
fn validate_encoder_layout(layout: &ChannelLayout) -> bool {
    for s in 0..layout.nb_streams {
        if s < layout.nb_coupled_streams {
            if get_left_channel(layout, s, -1) == -1 {
                return false;
            }
            if get_right_channel(layout, s, -1) == -1 {
                return false;
            }
        } else if get_mono_channel(layout, s, -1) == -1 {
            return false;
        }
    }
    true
}

// ===========================================================================
// Vorbis channel mappings
// ===========================================================================

struct VorbisLayout {
    nb_streams: i32,
    nb_coupled_streams: i32,
    mapping: [u8; 8],
}

/// Index is nb_channels - 1. Matches C `vorbis_mappings[8]`.
static VORBIS_MAPPINGS: [VorbisLayout; 8] = [
    VorbisLayout {
        nb_streams: 1,
        nb_coupled_streams: 0,
        mapping: [0, 0, 0, 0, 0, 0, 0, 0],
    },
    VorbisLayout {
        nb_streams: 1,
        nb_coupled_streams: 1,
        mapping: [0, 1, 0, 0, 0, 0, 0, 0],
    },
    VorbisLayout {
        nb_streams: 2,
        nb_coupled_streams: 1,
        mapping: [0, 2, 1, 0, 0, 0, 0, 0],
    },
    VorbisLayout {
        nb_streams: 2,
        nb_coupled_streams: 2,
        mapping: [0, 1, 2, 3, 0, 0, 0, 0],
    },
    VorbisLayout {
        nb_streams: 3,
        nb_coupled_streams: 2,
        mapping: [0, 4, 1, 2, 3, 0, 0, 0],
    },
    VorbisLayout {
        nb_streams: 4,
        nb_coupled_streams: 2,
        mapping: [0, 4, 1, 2, 3, 5, 0, 0],
    },
    VorbisLayout {
        nb_streams: 4,
        nb_coupled_streams: 3,
        mapping: [0, 4, 1, 2, 3, 5, 6, 0],
    },
    VorbisLayout {
        nb_streams: 5,
        nb_coupled_streams: 3,
        mapping: [0, 6, 1, 2, 3, 4, 5, 7],
    },
];

// ===========================================================================
// Ambisonics validation
// ===========================================================================

/// Validate ambisonics channel count and compute stream/coupled counts.
/// Valid: (N+1)^2 or (N+1)^2 + 2, for N=0..14.
/// Matches C `validate_ambisonics`.
fn validate_ambisonics(
    nb_channels: i32,
    nb_streams: Option<&mut i32>,
    nb_coupled_streams: Option<&mut i32>,
) -> bool {
    if nb_channels < 1 || nb_channels > 227 {
        return false;
    }
    let order_plus_one = isqrt32(nb_channels as u32) as i32;
    let acn_channels = order_plus_one * order_plus_one;
    let nondiegetic_channels = nb_channels - acn_channels;

    if nondiegetic_channels != 0 && nondiegetic_channels != 2 {
        return false;
    }

    if let Some(ns) = nb_streams {
        *ns = acn_channels + (nondiegetic_channels != 0) as i32;
    }
    if let Some(ncs) = nb_coupled_streams {
        *ncs = (nondiegetic_channels != 0) as i32;
    }
    true
}

// ===========================================================================
// Channel position for surround analysis
// ===========================================================================

/// Assign spatial positions for surround mixing analysis.
/// 0=don't mix (LFE), 1=left, 2=center, 3=right.
/// Matches C `channel_pos`.
fn channel_pos(channels: i32, pos: &mut [i32; 8]) {
    pos.fill(0);
    if channels == 4 {
        pos[0] = 1;
        pos[1] = 3;
        pos[2] = 1;
        pos[3] = 3;
    } else if channels == 3 || channels == 5 || channels == 6 {
        pos[0] = 1;
        pos[1] = 2;
        pos[2] = 3;
        pos[3] = 1;
        pos[4] = 3;
        pos[5] = 0;
    } else if channels == 7 {
        pos[0] = 1;
        pos[1] = 2;
        pos[2] = 3;
        pos[3] = 1;
        pos[4] = 3;
        pos[5] = 2;
        pos[6] = 0;
    } else if channels == 8 {
        pos[0] = 1;
        pos[1] = 2;
        pos[2] = 3;
        pos[3] = 1;
        pos[4] = 3;
        pos[5] = 1;
        pos[6] = 3;
        pos[7] = 0;
    }
}

// ===========================================================================
// logSum — approximate log2(2^a + 2^b)
// ===========================================================================

/// Fixed-point GCONST macro: value in Q(DB_SHIFT=24) format.
const fn gconst(x: f64) -> i32 {
    (x * ((1i64 << DB_SHIFT) as f64) + 0.5) as i32
}

/// Rough approximation of log2(2^a + 2^b) in fixed-point.
/// Matches C `logSum` (fixed-point path).
fn log_sum(a: i32, b: i32) -> i32 {
    static DIFF_TABLE: [i32; 17] = [
        gconst(0.5000000),
        gconst(0.2924813),
        gconst(0.1609640),
        gconst(0.0849625),
        gconst(0.0437314),
        gconst(0.0221971),
        gconst(0.0111839),
        gconst(0.0056136),
        gconst(0.0028123),
        // Table only needs 9 entries for diff < 8.0 (indices 0..8 + interpolation to 9)
        // But C code has 17 entries; pad with very small values for safety.
        gconst(0.0014060),
        gconst(0.0007030),
        gconst(0.0003515),
        gconst(0.0001757),
        gconst(0.0000879),
        gconst(0.0000439),
        gconst(0.0000220),
        gconst(0.0000110),
    ];

    let (max_val, diff) = if a > b { (a, a - b) } else { (b, b - a) };
    // Inverted comparison to catch large values (NaN-like guard from C)
    if !(diff < gconst(8.0)) {
        return max_val;
    }
    // low = floor(2*diff) in Q(DB_SHIFT-1)
    let low = (diff >> (DB_SHIFT - 1)) as usize;
    // frac in Q16 for interpolation
    let frac = vshr32(diff - ((low as i32) << (DB_SHIFT - 1)), DB_SHIFT - 16);
    max_val + DIFF_TABLE[low] + mult16_32_q15(frac, DIFF_TABLE[low + 1] - DIFF_TABLE[low])
}

/// Variable shift right — handles both positive and negative shifts.
#[inline(always)]
fn vshr32(x: i32, shift: i32) -> i32 {
    if shift >= 0 {
        x >> shift
    } else {
        x << (-shift)
    }
}

/// 16×32 multiply keeping Q15 result.
#[inline(always)]
fn mult16_32_q15(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> 15) as i32
}

// ===========================================================================
// Surround analysis
// ===========================================================================

/// Compute per-channel band energies and signal-to-mask ratios for surround mode.
/// Matches C `surround_analysis` (fixed-point path, no float NaN guard).
fn surround_analysis(
    celt_mode: &CELTMode,
    pcm: &[i16],
    band_log_e: &mut [i32],
    mem: &mut [i32],
    preemph_mem: &mut [i32],
    len: i32,
    overlap: i32,
    channels: i32,
    rate: i32,
) {
    let mut pos = [0i32; 8];
    let upsample = resampling_factor(rate);
    let frame_size = len * upsample;

    // LM = log2(frame_size / shortMdctSize)
    let mut lm = 0i32;
    while lm < celt_mode.max_lm {
        if (celt_mode.short_mdct_size << lm) == frame_size {
            break;
        }
        lm += 1;
    }

    let freq_size = celt_mode.short_mdct_size << lm;
    let ov = overlap as usize;

    let mut inp = vec![0i32; (frame_size + overlap) as usize];
    let mut x = vec![0i16; len as usize];
    let mut freq = vec![0i32; freq_size as usize];

    channel_pos(channels, &mut pos);

    let mut mask_log_e = [[-gconst(28.0); 21]; 3];

    for c in 0..channels as usize {
        let nb_frames = frame_size / freq_size;
        // Copy overlap memory
        inp[..ov].copy_from_slice(&mem[c * ov..(c + 1) * ov]);

        // Extract channel c from interleaved PCM
        for i in 0..len as usize {
            x[i] = pcm[i * channels as usize + c];
        }

        // Apply pre-emphasis
        celt_preemphasis(
            &x,
            0,
            &mut inp,
            ov,
            frame_size,
            1,
            upsample,
            &celt_mode.preemph,
            &mut preemph_mem[c],
            0,
        );

        let mut band_e = [0i32; 21];
        for frame in 0..nb_frames {
            let mut tmp_e = [0i32; 21];
            let shift = celt_mode.max_lm - lm;
            let fft_state = get_fft_state(shift);
            // Compute trig table offset, matching C: trig += N for each shift level
            let mdct = &crate::celt::mdct::MDCT_48000_960;
            let mut trig_offset = 0usize;
            let mut trig_n = mdct.n;
            for _ in 0..shift {
                trig_n >>= 1;
                trig_offset += trig_n as usize;
            }
            let trig = &mdct.trig[trig_offset..];
            clt_mdct_forward(
                &inp[(freq_size * frame) as usize..],
                &mut freq,
                celt_mode.window,
                overlap,
                shift,
                1,
                fft_state,
                trig,
            );
            if upsample != 1 {
                let bound = (freq_size / upsample) as usize;
                for i in 0..bound {
                    freq[i] *= upsample;
                }
                for i in bound..freq_size as usize {
                    freq[i] = 0;
                }
            }
            compute_band_energies(celt_mode, &freq, &mut tmp_e, 21, 1, lm);
            for i in 0..21 {
                band_e[i] = band_e[i].max(tmp_e[i]);
            }
        }
        amp2log2(celt_mode, 21, 21, &band_e, &mut band_log_e[21 * c..], 1);

        // Apply spreading function: -6 dB/band up, -12 dB/band down
        for i in 1..21 {
            let idx = 21 * c + i;
            band_log_e[idx] = band_log_e[idx].max(band_log_e[idx - 1] - gconst(1.0));
        }
        for i in (0..20).rev() {
            let idx = 21 * c + i;
            band_log_e[idx] = band_log_e[idx].max(band_log_e[idx + 1] - gconst(2.0));
        }

        // Accumulate into left/right/center masks
        if pos[c] == 1 {
            for i in 0..21 {
                mask_log_e[0][i] = log_sum(mask_log_e[0][i], band_log_e[21 * c + i]);
            }
        } else if pos[c] == 3 {
            for i in 0..21 {
                mask_log_e[2][i] = log_sum(mask_log_e[2][i], band_log_e[21 * c + i]);
            }
        } else if pos[c] == 2 {
            for i in 0..21 {
                mask_log_e[0][i] = log_sum(mask_log_e[0][i], band_log_e[21 * c + i] - gconst(0.5));
                mask_log_e[2][i] = log_sum(mask_log_e[2][i], band_log_e[21 * c + i] - gconst(0.5));
            }
        }

        // Save overlap for next frame
        mem[c * ov..(c + 1) * ov]
            .copy_from_slice(&inp[frame_size as usize..(frame_size + overlap) as usize]);
    }

    // Center mask = min(left, right)
    for i in 0..21 {
        mask_log_e[1][i] = mask_log_e[0][i].min(mask_log_e[2][i]);
    }

    // Channel offset: 0.5 * log2(2/(channels-1))
    // qconst32(2.0, 14) = 32768, celt_log2 returns Q(DB_SHIFT)
    let channel_offset = celt_log2(32768 / (channels - 1)) >> 1;
    for c in 0..3 {
        for i in 0..21 {
            mask_log_e[c][i] += channel_offset;
        }
    }

    // Subtract mask to produce SMR per channel
    for c in 0..channels as usize {
        if pos[c] != 0 {
            let mask = &mask_log_e[(pos[c] - 1) as usize];
            for i in 0..21 {
                band_log_e[21 * c + i] -= mask[i];
            }
        } else {
            for i in 0..21 {
                band_log_e[21 * c + i] = 0;
            }
        }
    }
}

// ===========================================================================
// Rate allocation
// ===========================================================================

/// Surround-aware bitrate allocation across streams.
/// Matches C `surround_rate_allocation`.
fn surround_rate_allocation(
    layout: &ChannelLayout,
    lfe_stream: i32,
    bitrate_bps: i32,
    rate: &mut [i32],
    frame_size: i32,
    fs: i32,
) {
    let nb_lfe = if lfe_stream != -1 { 1 } else { 0 };
    let nb_coupled = layout.nb_coupled_streams;
    let nb_uncoupled = layout.nb_streams - nb_coupled - nb_lfe;
    let nb_normal = 2 * nb_coupled + nb_uncoupled;

    // Give each non-LFE channel enough bits per channel for coding band energy
    let channel_offset = 40 * imax(50, fs / frame_size);

    let bitrate = if bitrate_bps == OPUS_AUTO {
        nb_normal * (channel_offset + fs + 10000) + 8000 * nb_lfe
    } else if bitrate_bps == OPUS_BITRATE_MAX {
        nb_normal * 750000 + nb_lfe * 128000
    } else {
        bitrate_bps
    };

    // LFE allocation: never exceed 1/20 of total for non-energy part
    let lfe_offset = imin(bitrate / 20, 3000) + 15 * imax(50, fs / frame_size);

    // Per-stream starting offset (models coupling savings)
    let stream_offset = imax(
        0,
        imin(
            20000,
            (bitrate - channel_offset * nb_normal - lfe_offset * nb_lfe) / nb_normal / 2,
        ),
    );

    let coupled_ratio: i32 = 512; // Q8: 2.0x
    let lfe_ratio: i32 = 32; // Q8: 0.125x

    let total = (nb_uncoupled << 8) + coupled_ratio * nb_coupled + nb_lfe * lfe_ratio;
    let channel_rate = (256
        * (bitrate
            - lfe_offset * nb_lfe
            - stream_offset * (nb_coupled + nb_uncoupled)
            - channel_offset * nb_normal) as i64
        / total as i64) as i32;

    for i in 0..layout.nb_streams as usize {
        let i32_i = i as i32;
        if i32_i < layout.nb_coupled_streams {
            rate[i] =
                2 * channel_offset + imax(0, stream_offset + (channel_rate * coupled_ratio >> 8));
        } else if i32_i != lfe_stream {
            rate[i] = channel_offset + imax(0, stream_offset + channel_rate);
        } else {
            rate[i] = imax(0, lfe_offset + (channel_rate * lfe_ratio >> 8));
        }
    }
}

/// Equal bitrate allocation for ambisonics.
/// Matches C `ambisonics_rate_allocation`.
fn ambisonics_rate_allocation(
    layout: &ChannelLayout,
    bitrate_bps: i32,
    rate: &mut [i32],
    frame_size: i32,
    fs: i32,
) {
    let nb_channels = layout.nb_streams + layout.nb_coupled_streams;

    let total_rate = if bitrate_bps == OPUS_AUTO {
        (layout.nb_coupled_streams + layout.nb_streams) * (fs + 60 * fs / frame_size)
            + layout.nb_streams * 15000
    } else if bitrate_bps == OPUS_BITRATE_MAX {
        nb_channels * 750000
    } else {
        bitrate_bps
    };

    let per_stream_rate = total_rate / layout.nb_streams;
    for i in 0..layout.nb_streams as usize {
        rate[i] = per_stream_rate;
    }
}

// ===========================================================================
// Packet validation
// ===========================================================================

/// Validate a multistream packet: parse all sub-packets, verify consistent frame count.
/// Matches C `opus_multistream_packet_validate`.
fn multistream_packet_validate(data: &[u8], len: i32, nb_streams: i32, fs: i32) -> i32 {
    let mut offset = 0usize;
    let mut remaining = len;
    let mut samples = 0i32;

    for s in 0..nb_streams {
        if remaining <= 0 {
            return OPUS_INVALID_PACKET;
        }
        let sub_data = &data[offset..];
        let self_delimited = s != nb_streams - 1;

        // Parse packet to find offset
        let parse_result = parse_multistream_subpacket(sub_data, remaining, self_delimited);
        if parse_result < 0 {
            return parse_result;
        }
        let packet_offset = parse_result;

        let tmp_samples = match opus_packet_get_nb_samples(&sub_data[..packet_offset as usize], fs)
        {
            Ok(n) => n,
            Err(e) => return e,
        };
        if s != 0 && samples != tmp_samples {
            return OPUS_INVALID_PACKET;
        }
        samples = tmp_samples;
        offset += packet_offset as usize;
        remaining -= packet_offset;
    }
    samples
}

/// Parse a self-delimiting or standard sub-packet to find its byte length.
/// Returns packet_offset (total bytes consumed), or negative error code.
/// Simplified version of opus_packet_parse_impl — only extracts offset.
fn parse_multistream_subpacket(data: &[u8], len: i32, self_delimited: bool) -> i32 {
    if len < 1 {
        return OPUS_INVALID_PACKET;
    }

    let toc = data[0];
    let mut pos: usize = 1;
    let mut remaining = len - 1;

    // Determine number of frames from code
    let code = toc & 0x3;
    match code {
        0 => {}
        1 => {}
        2 => {}
        3 => {
            // VBR/CBR code 3
            if remaining < 1 {
                return OPUS_INVALID_PACKET;
            }
            let ch = data[pos];
            pos += 1;
            remaining -= 1;
            let count = (ch & 0x3F) as i32;
            if count == 0 || (count > 48) {
                return OPUS_INVALID_PACKET;
            }

            // Parse padding
            if ch & 0x40 != 0 {
                // Padding
                loop {
                    if remaining < 1 {
                        return OPUS_INVALID_PACKET;
                    }
                    let p = data[pos] as i32;
                    pos += 1;
                    remaining -= 1;
                    remaining -= if p == 255 { 254 } else { p };
                    if remaining < 0 {
                        return OPUS_INVALID_PACKET;
                    }
                    if p < 255 {
                        break;
                    }
                }
            }

            // VBR or CBR
            if ch & 0x80 != 0 {
                // VBR — parse (count-1) frame sizes
                for _ in 0..(count - 1) {
                    let (consumed, sz) = parse_size_field(&data[pos..], remaining);
                    if consumed < 0 {
                        return OPUS_INVALID_PACKET;
                    }
                    pos += consumed as usize;
                    remaining -= consumed;
                    remaining -= sz as i32;
                    if remaining < 0 {
                        return OPUS_INVALID_PACKET;
                    }
                }
            }
            // CBR — all frames same size, determined by remaining/count
        }
        _ => unreachable!(),
    }

    // Handle self-delimiting: parse the self-delimiting size field
    if self_delimited {
        let (consumed, sz) = parse_size_field(&data[pos..], remaining);
        if consumed < 0 {
            return OPUS_INVALID_PACKET;
        }
        pos += consumed as usize;
        remaining -= consumed;

        // For self-delimiting, the last frame size is explicit
        if code == 0 {
            remaining = sz as i32;
        } else if code == 1 {
            remaining = 2 * sz as i32;
        } else if code == 2 {
            // Code 2: first frame size is in the packet, second is the self-delim size
            // Need to parse the first frame size
            let (c2, s2) = parse_size_field(&data[pos..], remaining);
            if c2 < 0 {
                return OPUS_INVALID_PACKET;
            }
            remaining = s2 as i32 + sz as i32;
            pos += c2 as usize;
        } else {
            // Code 3: self-delim size is the last frame size, already accounted for
            remaining += sz as i32;
        }
        if remaining < 0 {
            return OPUS_INVALID_PACKET;
        }
    }

    pos as i32 + remaining
}

/// Parse a 1- or 2-byte size field.
fn parse_size_field(data: &[u8], len: i32) -> (i32, i16) {
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

// ===========================================================================
// Channel copy functions (encoder side)
// ===========================================================================

/// Extract a single channel from interleaved i16 PCM to a mono buffer.
/// Matches C `opus_copy_channel_in_short`.
fn copy_channel_in_short(
    dst: &mut [i16],
    dst_stride: usize,
    src: &[i16],
    src_stride: usize,
    src_channel: usize,
    frame_size: usize,
) {
    for i in 0..frame_size {
        dst[i * dst_stride] = src[i * src_stride + src_channel];
    }
}

// ===========================================================================
// Channel copy functions (decoder side)
// ===========================================================================

/// Route decoded i16 audio to output channel.
/// If `src` is None, writes silence.
/// Matches C `opus_copy_channel_out_short`.
fn copy_channel_out_short(
    dst: &mut [i16],
    dst_stride: usize,
    dst_channel: usize,
    src: Option<&[i16]>,
    src_stride: usize,
    frame_size: usize,
) {
    match src {
        Some(s) => {
            for i in 0..frame_size {
                dst[i * dst_stride + dst_channel] = s[i * src_stride];
            }
        }
        None => {
            for i in 0..frame_size {
                dst[i * dst_stride + dst_channel] = 0;
            }
        }
    }
}

// ===========================================================================
// OpusMSEncoder
// ===========================================================================

/// Opus multistream encoder — encodes multi-channel audio into concatenated sub-packets.
pub struct OpusMSEncoder {
    layout: ChannelLayout,
    lfe_stream: i32,
    bitrate_bps: i32,
    application: i32,
    variable_duration: i32,
    mapping_type: MappingType,
    encoders: Vec<OpusEncoder>,
    /// Window memory for surround analysis (channels * MAX_OVERLAP).
    window_mem: Vec<i32>,
    /// Pre-emphasis memory for surround analysis (one per channel).
    preemph_mem: Vec<i32>,
}

impl OpusMSEncoder {
    /// Create and initialize a multistream encoder.
    /// Matches C `opus_multistream_encoder_create`.
    pub fn new(
        fs: i32,
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        mapping: &[u8],
        application: i32,
    ) -> Result<Self, i32> {
        Self::new_impl(
            fs,
            channels,
            streams,
            coupled_streams,
            mapping,
            application,
            MappingType::None,
            -1,
        )
    }

    /// Create a surround encoder with automatic mapping.
    /// Matches C `opus_multistream_surround_encoder_create`.
    pub fn new_surround(
        fs: i32,
        channels: i32,
        mapping_family: i32,
        application: i32,
    ) -> Result<(Self, i32, i32, Vec<u8>), i32> {
        if channels > 255 || channels < 1 {
            return Err(OPUS_BAD_ARG);
        }

        let streams: i32;
        let coupled_streams: i32;
        let mut mapping = vec![0u8; channels as usize];
        let mut lfe_stream: i32 = -1;
        let mapping_type: MappingType;

        if mapping_family == 0 {
            if channels == 1 {
                streams = 1;
                coupled_streams = 0;
                mapping[0] = 0;
            } else if channels == 2 {
                streams = 1;
                coupled_streams = 1;
                mapping[0] = 0;
                mapping[1] = 1;
            } else {
                return Err(OPUS_UNIMPLEMENTED);
            }
        } else if mapping_family == 1 && channels <= 8 && channels >= 1 {
            let vm = &VORBIS_MAPPINGS[(channels - 1) as usize];
            streams = vm.nb_streams;
            coupled_streams = vm.nb_coupled_streams;
            for i in 0..channels as usize {
                mapping[i] = vm.mapping[i];
            }
            if channels >= 6 {
                lfe_stream = streams - 1;
            }
        } else if mapping_family == 255 {
            streams = channels;
            coupled_streams = 0;
            for i in 0..channels as usize {
                mapping[i] = i as u8;
            }
        } else if mapping_family == 2 {
            let mut ns = 0i32;
            let mut ncs = 0i32;
            if !validate_ambisonics(channels, Some(&mut ns), Some(&mut ncs)) {
                return Err(OPUS_BAD_ARG);
            }
            streams = ns;
            coupled_streams = ncs;
            // Ambisonics mapping: mono streams first, then coupled
            for i in 0..(streams - coupled_streams) as usize {
                mapping[i] = (i as i32 + coupled_streams * 2) as u8;
            }
            for i in 0..(coupled_streams * 2) as usize {
                mapping[i + (streams - coupled_streams) as usize] = i as u8;
            }
        } else {
            return Err(OPUS_UNIMPLEMENTED);
        }

        if channels > 2 && mapping_family == 1 {
            mapping_type = MappingType::Surround;
        } else if mapping_family == 2 {
            mapping_type = MappingType::Ambisonics;
        } else {
            mapping_type = MappingType::None;
        }

        let encoder = Self::new_impl(
            fs,
            channels,
            streams,
            coupled_streams,
            &mapping,
            application,
            mapping_type,
            lfe_stream,
        )?;

        Ok((encoder, streams, coupled_streams, mapping))
    }

    fn new_impl(
        fs: i32,
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        mapping: &[u8],
        application: i32,
        mapping_type: MappingType,
        lfe_stream: i32,
    ) -> Result<Self, i32> {
        if channels > 255
            || channels < 1
            || coupled_streams > streams
            || streams < 1
            || coupled_streams < 0
            || streams > 255 - coupled_streams
            || streams + coupled_streams > channels
        {
            return Err(OPUS_BAD_ARG);
        }

        let mut layout = ChannelLayout::new();
        layout.nb_channels = channels;
        layout.nb_streams = streams;
        layout.nb_coupled_streams = coupled_streams;
        for i in 0..channels as usize {
            layout.mapping[i] = mapping[i];
        }
        if !validate_layout(&layout) {
            return Err(OPUS_BAD_ARG);
        }
        if !validate_encoder_layout(&layout) {
            return Err(OPUS_BAD_ARG);
        }
        if mapping_type == MappingType::Ambisonics && !validate_ambisonics(channels, None, None) {
            return Err(OPUS_BAD_ARG);
        }

        // Create sub-encoders
        let mut encoders = Vec::with_capacity(streams as usize);
        for s in 0..streams {
            let ch = if s < coupled_streams { 2 } else { 1 };
            let mut enc = OpusEncoder::new(fs, ch, application)?;
            if s == lfe_stream {
                enc.ms_set_lfe(1);
            }
            encoders.push(enc);
        }

        let (window_mem, preemph_mem) = if mapping_type == MappingType::Surround {
            (
                vec![0i32; channels as usize * MAX_OVERLAP],
                vec![0i32; channels as usize],
            )
        } else {
            (Vec::new(), Vec::new())
        };

        Ok(Self {
            layout,
            lfe_stream,
            bitrate_bps: OPUS_AUTO,
            application,
            variable_duration: OPUS_FRAMESIZE_ARG,
            mapping_type,
            encoders,
            window_mem,
            preemph_mem,
        })
    }

    /// Encode multi-channel PCM to a multistream packet.
    /// Returns the number of bytes written, or a negative error code.
    /// Matches C `opus_multistream_encode`.
    pub fn encode(
        &mut self,
        pcm: &[i16],
        frame_size: i32,
        data: &mut [u8],
        max_data_bytes: i32,
    ) -> Result<i32, i32> {
        self.encode_native(pcm, frame_size, data, max_data_bytes, 16)
    }

    /// Internal encode — top-level multistream encoding orchestrator.
    /// Matches C `opus_multistream_encode_native` (fixed-point, no analysis downmix).
    fn encode_native(
        &mut self,
        pcm: &[i16],
        analysis_frame_size: i32,
        data: &mut [u8],
        max_data_bytes: i32,
        lsb_depth: i32,
    ) -> Result<i32, i32> {
        let fs = self.encoders[0].fs;
        let vbr = self.encoders[0].ms_get_vbr();

        let frame_size = frame_size_select(analysis_frame_size, self.variable_duration, fs);
        if frame_size <= 0 {
            return Err(OPUS_BAD_ARG);
        }

        // Smallest packet the encoder can produce
        let mut smallest_packet = self.layout.nb_streams * 2 - 1;
        if fs / frame_size == 10 {
            smallest_packet += self.layout.nb_streams;
        }
        if max_data_bytes < smallest_packet {
            return Err(OPUS_BUFFER_TOO_SMALL);
        }

        let mut buf = vec![0i16; 2 * frame_size as usize];
        let mut band_smr = vec![0i32; 21 * self.layout.nb_channels as usize];
        let mut band_log_e = [0i32; 42];

        // Surround analysis
        if self.mapping_type == MappingType::Surround {
            if let Some(celt_mode) = self.encoders[0].ms_get_celt_mode() {
                surround_analysis(
                    celt_mode,
                    pcm,
                    &mut band_smr,
                    &mut self.window_mem,
                    &mut self.preemph_mem,
                    frame_size,
                    celt_mode.overlap,
                    self.layout.nb_channels,
                    fs,
                );
            }
        }

        // Rate allocation
        let mut bitrates = vec![0i32; self.layout.nb_streams as usize];
        let rate_sum = self.rate_allocation(&mut bitrates, frame_size, fs);

        let mut max_data_bytes = max_data_bytes;
        if vbr == 0 {
            if self.bitrate_bps == OPUS_AUTO {
                max_data_bytes = imin(
                    max_data_bytes,
                    (bitrate_to_bits(rate_sum, fs, frame_size) + 4) / 8,
                );
            } else if self.bitrate_bps != OPUS_BITRATE_MAX {
                max_data_bytes = imin(
                    max_data_bytes,
                    imax(
                        smallest_packet,
                        (bitrate_to_bits(self.bitrate_bps, fs, frame_size) + 4) / 8,
                    ),
                );
            }
        }

        // Configure and encode each stream
        for s in 0..self.layout.nb_streams as usize {
            let enc = &mut self.encoders[s];
            enc.ms_set_user_bitrate(bitrates[s]);

            if self.mapping_type == MappingType::Surround {
                let equiv_rate =
                    if self.bitrate_bps == OPUS_AUTO || self.bitrate_bps == OPUS_BITRATE_MAX {
                        bitrates.iter().sum::<i32>()
                    } else {
                        self.bitrate_bps
                    };
                let mut adj_rate = equiv_rate;
                if frame_size * 50 < fs {
                    adj_rate -= 60 * (fs / frame_size - 50) * self.layout.nb_channels;
                }
                if adj_rate > 10000 * self.layout.nb_channels {
                    enc.ms_set_bandwidth(OPUS_BANDWIDTH_FULLBAND);
                } else if adj_rate > 7000 * self.layout.nb_channels {
                    enc.ms_set_bandwidth(OPUS_BANDWIDTH_SUPERWIDEBAND);
                } else if adj_rate > 5000 * self.layout.nb_channels {
                    enc.ms_set_bandwidth(OPUS_BANDWIDTH_WIDEBAND);
                } else {
                    enc.ms_set_bandwidth(OPUS_BANDWIDTH_NARROWBAND);
                }
                if (s as i32) < self.layout.nb_coupled_streams {
                    enc.ms_set_force_mode(MODE_CELT_ONLY);
                    enc.ms_set_force_channels(2);
                }
            } else if self.mapping_type == MappingType::Ambisonics {
                enc.ms_set_force_mode(MODE_CELT_ONLY);
            }
        }

        // Encode each stream
        let mut tot_size: i32 = 0;
        let mut data_offset: usize = 0;
        let mut tmp_data = [0u8; MS_FRAME_TMP];

        for s in 0..self.layout.nb_streams as usize {
            let s_i32 = s as i32;

            // Extract channel(s) for this stream
            if s_i32 < self.layout.nb_coupled_streams {
                let left = get_left_channel(&self.layout, s_i32, -1) as usize;
                let right = get_right_channel(&self.layout, s_i32, -1) as usize;
                copy_channel_in_short(
                    &mut buf,
                    2,
                    pcm,
                    self.layout.nb_channels as usize,
                    left,
                    frame_size as usize,
                );
                copy_channel_in_short(
                    &mut buf[1..],
                    2,
                    pcm,
                    self.layout.nb_channels as usize,
                    right,
                    frame_size as usize,
                );
                if self.mapping_type == MappingType::Surround {
                    for i in 0..21 {
                        band_log_e[i] = band_smr[21 * left + i];
                        band_log_e[21 + i] = band_smr[21 * right + i];
                    }
                }
            } else {
                let chan = get_mono_channel(&self.layout, s_i32, -1) as usize;
                copy_channel_in_short(
                    &mut buf,
                    1,
                    pcm,
                    self.layout.nb_channels as usize,
                    chan,
                    frame_size as usize,
                );
                if self.mapping_type == MappingType::Surround {
                    for i in 0..21 {
                        band_log_e[i] = band_smr[21 * chan + i];
                    }
                }
            }

            // Compute max bytes for this stream
            let mut curr_max = max_data_bytes - tot_size;
            // Reserve space for remaining streams
            curr_max -= imax(0, 2 * (self.layout.nb_streams - s_i32 - 1) - 1);
            if fs / frame_size == 10 {
                curr_max -= self.layout.nb_streams - s_i32 - 1;
            }
            curr_max = imin(curr_max, MS_FRAME_TMP as i32);
            // Account for self-delimiting overhead
            if s_i32 != self.layout.nb_streams - 1 {
                curr_max -= if curr_max > 253 { 2 } else { 1 };
            }

            // CBR last-stream adjustment
            if vbr == 0 && s_i32 == self.layout.nb_streams - 1 {
                self.encoders[s].ms_set_user_bitrate(bits_to_bitrate(curr_max * 8, fs, frame_size));
            }

            // Encode this stream
            let enc = &mut self.encoders[s];
            let len = enc.encode_native(&buf, frame_size, &mut tmp_data, curr_max, lsb_depth)?;

            // Repacketize with self-delimiting framing
            let mut rp = OpusRepacketizer::new();
            let ret = rp.cat(&tmp_data[..len as usize], len);
            if ret != OPUS_OK {
                return Err(OPUS_INTERNAL_ERROR);
            }
            let nb_frames = rp.get_nb_frames();
            let is_self_delimited = s_i32 != self.layout.nb_streams - 1;
            let pad_to_max = vbr == 0 && s_i32 == self.layout.nb_streams - 1;
            let out_len = rp.out_range_impl(
                0,
                nb_frames as usize,
                &mut data[data_offset..],
                max_data_bytes - tot_size,
                is_self_delimited,
                pad_to_max,
                &[],
            );
            if out_len < 0 {
                return Err(out_len);
            }
            data_offset += out_len as usize;
            tot_size += out_len;
        }

        Ok(tot_size)
    }

    /// Compute bitrate allocation, returning the total rate sum.
    fn rate_allocation(&self, rate: &mut [i32], frame_size: i32, fs: i32) -> i32 {
        if self.mapping_type == MappingType::Ambisonics {
            ambisonics_rate_allocation(&self.layout, self.bitrate_bps, rate, frame_size, fs);
        } else {
            surround_rate_allocation(
                &self.layout,
                self.lfe_stream,
                self.bitrate_bps,
                rate,
                frame_size,
                fs,
            );
        }
        let mut rate_sum = 0i32;
        for i in 0..self.layout.nb_streams as usize {
            rate[i] = imax(rate[i], 500);
            rate_sum += rate[i];
        }
        rate_sum
    }

    /// Get/set encoder parameters. Matches C `opus_multistream_encoder_ctl`.
    pub fn set_bitrate(&mut self, value: i32) -> i32 {
        if value != OPUS_AUTO && value != OPUS_BITRATE_MAX {
            if value <= 0 {
                return OPUS_BAD_ARG;
            }
            let clamped = imin(
                750000 * self.layout.nb_channels,
                imax(500 * self.layout.nb_channels, value),
            );
            self.bitrate_bps = clamped;
        } else {
            self.bitrate_bps = value;
        }
        OPUS_OK
    }

    pub fn get_bitrate(&self) -> i32 {
        // Sum of per-stream bitrates
        let mut total = 0i32;
        for enc in &self.encoders {
            total += enc.ms_get_bitrate();
        }
        total
    }

    /// Get combined final range (XOR of all sub-stream ranges).
    pub fn get_final_range(&self) -> u32 {
        let mut val = 0u32;
        for enc in &self.encoders {
            val ^= enc.range_final;
        }
        val
    }

    /// Set a parameter on all sub-encoders.
    pub fn set_complexity(&mut self, v: i32) -> i32 {
        for enc in &mut self.encoders {
            enc.ms_set_complexity(v);
        }
        OPUS_OK
    }

    pub fn set_vbr(&mut self, v: i32) -> i32 {
        for enc in &mut self.encoders {
            enc.ms_set_vbr(v);
        }
        OPUS_OK
    }

    pub fn set_vbr_constraint(&mut self, v: i32) -> i32 {
        for enc in &mut self.encoders {
            enc.ms_set_vbr_constraint(v);
        }
        OPUS_OK
    }

    pub fn set_signal(&mut self, v: i32) -> i32 {
        for enc in &mut self.encoders {
            enc.ms_set_signal(v);
        }
        OPUS_OK
    }

    pub fn set_bandwidth(&mut self, v: i32) -> i32 {
        for enc in &mut self.encoders {
            enc.ms_set_bandwidth(v);
        }
        OPUS_OK
    }

    pub fn set_max_bandwidth(&mut self, v: i32) -> i32 {
        for enc in &mut self.encoders {
            enc.ms_set_max_bandwidth(v);
        }
        OPUS_OK
    }

    pub fn set_inband_fec(&mut self, v: i32) -> i32 {
        for enc in &mut self.encoders {
            enc.ms_set_inband_fec(v);
        }
        OPUS_OK
    }

    pub fn set_packet_loss_perc(&mut self, v: i32) -> i32 {
        for enc in &mut self.encoders {
            enc.ms_set_packet_loss_perc(v);
        }
        OPUS_OK
    }

    pub fn set_dtx(&mut self, v: i32) -> i32 {
        for enc in &mut self.encoders {
            enc.ms_set_dtx(v);
        }
        OPUS_OK
    }

    pub fn set_lsb_depth(&mut self, v: i32) -> i32 {
        for enc in &mut self.encoders {
            enc.ms_set_lsb_depth(v);
        }
        OPUS_OK
    }

    pub fn set_prediction_disabled(&mut self, v: i32) -> i32 {
        for enc in &mut self.encoders {
            enc.ms_set_prediction_disabled(v);
        }
        OPUS_OK
    }

    pub fn set_phase_inversion_disabled(&mut self, v: i32) -> i32 {
        for enc in &mut self.encoders {
            enc.ms_set_phase_inversion_disabled(v);
        }
        OPUS_OK
    }

    pub fn set_force_mode(&mut self, v: i32) -> i32 {
        for enc in &mut self.encoders {
            enc.ms_set_force_mode(v);
        }
        OPUS_OK
    }

    pub fn set_force_channels(&mut self, v: i32) -> i32 {
        for enc in &mut self.encoders {
            enc.ms_set_force_channels(v);
        }
        OPUS_OK
    }

    pub fn set_application(&mut self, v: i32) -> i32 {
        self.application = v;
        for enc in &mut self.encoders {
            enc.ms_set_application(v);
        }
        OPUS_OK
    }

    pub fn get_application(&self) -> i32 {
        self.application
    }

    pub fn set_expert_frame_duration(&mut self, v: i32) {
        self.variable_duration = v;
    }

    pub fn get_expert_frame_duration(&self) -> i32 {
        self.variable_duration
    }

    pub fn get_sample_rate(&self) -> i32 {
        self.encoders[0].fs
    }

    pub fn get_lookahead(&self) -> i32 {
        self.encoders[0].ms_get_lookahead()
    }

    pub fn get_encoder(&self, stream_id: usize) -> Option<&OpusEncoder> {
        self.encoders.get(stream_id)
    }

    pub fn get_encoder_mut(&mut self, stream_id: usize) -> Option<&mut OpusEncoder> {
        self.encoders.get_mut(stream_id)
    }

    /// Reset all encoder state.
    pub fn reset(&mut self) {
        if self.mapping_type == MappingType::Surround {
            self.preemph_mem.fill(0);
            self.window_mem.fill(0);
        }
        for enc in &mut self.encoders {
            enc.reset();
        }
    }

    pub fn nb_streams(&self) -> i32 {
        self.layout.nb_streams
    }
    pub fn nb_coupled_streams(&self) -> i32 {
        self.layout.nb_coupled_streams
    }
}

// ===========================================================================
// OpusMSDecoder
// ===========================================================================

/// Opus multistream decoder — decodes concatenated sub-packets to multi-channel PCM.
pub struct OpusMSDecoder {
    layout: ChannelLayout,
    decoders: Vec<OpusDecoder>,
}

impl OpusMSDecoder {
    /// Create and initialize a multistream decoder.
    /// Matches C `opus_multistream_decoder_create`.
    pub fn new(
        fs: i32,
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        mapping: &[u8],
    ) -> Result<Self, i32> {
        if channels > 255
            || channels < 1
            || coupled_streams > streams
            || streams < 1
            || coupled_streams < 0
            || streams > 255 - coupled_streams
        {
            return Err(OPUS_BAD_ARG);
        }

        let mut layout = ChannelLayout::new();
        layout.nb_channels = channels;
        layout.nb_streams = streams;
        layout.nb_coupled_streams = coupled_streams;
        for i in 0..channels as usize {
            layout.mapping[i] = mapping[i];
        }
        if !validate_layout(&layout) {
            return Err(OPUS_BAD_ARG);
        }

        let mut decoders = Vec::with_capacity(streams as usize);
        for s in 0..streams {
            let ch = if s < coupled_streams { 2 } else { 1 };
            decoders.push(OpusDecoder::new(fs, ch)?);
        }

        Ok(Self { layout, decoders })
    }

    /// Decode a multistream packet to interleaved i16 PCM.
    /// Matches C `opus_multistream_decode`.
    pub fn decode(
        &mut self,
        data: Option<&[u8]>,
        len: i32,
        pcm: &mut [i16],
        frame_size: i32,
        decode_fec: bool,
    ) -> Result<i32, i32> {
        self.decode_native(data, len, pcm, frame_size, decode_fec)
    }

    /// Internal decode. Matches C `opus_multistream_decode_native` (fixed-point).
    fn decode_native(
        &mut self,
        data: Option<&[u8]>,
        len: i32,
        pcm: &mut [i16],
        mut frame_size: i32,
        decode_fec: bool,
    ) -> Result<i32, i32> {
        if frame_size <= 0 {
            return Err(OPUS_BAD_ARG);
        }
        let fs = self.decoders[0].ms_get_sample_rate();
        // Limit frame_size to avoid excessive allocation
        frame_size = imin(frame_size, fs / 25 * 3);

        let mut buf = vec![0i16; 2 * frame_size as usize];
        let do_plc = data.is_none() || len == 0;

        if !do_plc && len < 0 {
            return Err(OPUS_BAD_ARG);
        }
        if !do_plc && len < 2 * self.layout.nb_streams - 1 {
            return Err(OPUS_INVALID_PACKET);
        }

        // Validate the multistream packet
        if !do_plc {
            let raw_data = data.unwrap();
            let ret = multistream_packet_validate(raw_data, len, self.layout.nb_streams, fs);
            if ret < 0 {
                return Err(ret);
            } else if ret > frame_size {
                return Err(OPUS_BUFFER_TOO_SMALL);
            }
        }

        let mut data_offset: usize = 0;
        let mut remaining = len;

        for s in 0..self.layout.nb_streams as usize {
            let s_i32 = s as i32;

            if !do_plc && remaining <= 0 {
                return Err(OPUS_INTERNAL_ERROR);
            }

            let mut packet_offset: i32 = 0;
            let sub_data = if do_plc {
                None
            } else {
                Some(&data.unwrap()[data_offset..])
            };
            let self_delimited = s_i32 != self.layout.nb_streams - 1;

            let dec = &mut self.decoders[s];
            let ret = dec.decode_native(
                sub_data,
                &mut buf,
                frame_size,
                decode_fec,
                self_delimited,
                Some(&mut packet_offset),
            )?;

            if !do_plc {
                data_offset += packet_offset as usize;
                remaining -= packet_offset;
            }

            frame_size = ret;

            // Route decoded audio to output channels
            if s_i32 < self.layout.nb_coupled_streams {
                // Coupled stream: route left and right
                let mut prev = -1;
                loop {
                    let chan = get_left_channel(&self.layout, s_i32, prev);
                    if chan == -1 {
                        break;
                    }
                    copy_channel_out_short(
                        pcm,
                        self.layout.nb_channels as usize,
                        chan as usize,
                        Some(&buf),
                        2, // stride 2 for interleaved stereo
                        frame_size as usize,
                    );
                    prev = chan;
                }
                prev = -1;
                loop {
                    let chan = get_right_channel(&self.layout, s_i32, prev);
                    if chan == -1 {
                        break;
                    }
                    copy_channel_out_short(
                        pcm,
                        self.layout.nb_channels as usize,
                        chan as usize,
                        Some(&buf[1..]),
                        2,
                        frame_size as usize,
                    );
                    prev = chan;
                }
            } else {
                // Mono stream
                let mut prev = -1;
                loop {
                    let chan = get_mono_channel(&self.layout, s_i32, prev);
                    if chan == -1 {
                        break;
                    }
                    copy_channel_out_short(
                        pcm,
                        self.layout.nb_channels as usize,
                        chan as usize,
                        Some(&buf),
                        1,
                        frame_size as usize,
                    );
                    prev = chan;
                }
            }
        }

        // Handle muted channels (mapping == 255)
        for c in 0..self.layout.nb_channels as usize {
            if self.layout.mapping[c] == 255 {
                copy_channel_out_short(
                    pcm,
                    self.layout.nb_channels as usize,
                    c,
                    None,
                    0,
                    frame_size as usize,
                );
            }
        }

        Ok(frame_size)
    }

    /// Get combined final range (XOR of all sub-stream ranges).
    pub fn get_final_range(&self) -> u32 {
        let mut val = 0u32;
        for dec in &self.decoders {
            val ^= dec.ms_get_range_final();
        }
        val
    }

    pub fn get_sample_rate(&self) -> i32 {
        self.decoders[0].ms_get_sample_rate()
    }

    pub fn get_bandwidth(&self) -> i32 {
        self.decoders[0].ms_get_bandwidth()
    }

    pub fn get_last_packet_duration(&self) -> i32 {
        self.decoders[0].ms_get_last_packet_duration()
    }

    pub fn set_gain(&mut self, gain: i32) -> i32 {
        for dec in &mut self.decoders {
            dec.ms_set_gain(gain);
        }
        OPUS_OK
    }

    pub fn get_gain(&self) -> i32 {
        self.decoders[0].ms_get_gain()
    }

    pub fn set_phase_inversion_disabled(&mut self, v: i32) -> i32 {
        for dec in &mut self.decoders {
            dec.ms_set_phase_inversion_disabled(v);
        }
        OPUS_OK
    }

    pub fn get_phase_inversion_disabled(&self) -> i32 {
        self.decoders[0].ms_get_phase_inversion_disabled()
    }

    pub fn set_complexity(&mut self, v: i32) -> i32 {
        for dec in &mut self.decoders {
            dec.ms_set_complexity(v);
        }
        OPUS_OK
    }

    pub fn get_complexity(&self) -> i32 {
        self.decoders[0].ms_get_complexity()
    }

    /// Reset all decoder state.
    pub fn reset(&mut self) {
        for dec in &mut self.decoders {
            dec.ms_reset();
        }
    }

    pub fn get_decoder(&self, stream_id: usize) -> Option<&OpusDecoder> {
        self.decoders.get(stream_id)
    }

    pub fn get_decoder_mut(&mut self, stream_id: usize) -> Option<&mut OpusDecoder> {
        self.decoders.get_mut(stream_id)
    }

    pub fn nb_streams(&self) -> i32 {
        self.layout.nb_streams
    }
    pub fn nb_coupled_streams(&self) -> i32 {
        self.layout.nb_coupled_streams
    }
}

// ===========================================================================
// MappingMatrix
// ===========================================================================

/// Mapping matrix for ambisonics projection.
/// Stores Q15 fixed-point coefficients in column-major order.
/// Matches C `MappingMatrix`.
#[derive(Clone)]
pub struct MappingMatrix {
    pub rows: i32,
    pub cols: i32,
    /// Gain in dB, S7.8 fixed-point format.
    pub gain: i32,
    /// Matrix data in column-major order: element(row, col) = data[rows * col + row].
    pub data: Vec<i16>,
}

impl MappingMatrix {
    /// Create a new mapping matrix from the given data.
    pub fn new(rows: i32, cols: i32, gain: i32, data: &[i16]) -> Self {
        Self {
            rows,
            cols,
            gain,
            data: data.to_vec(),
        }
    }

    /// Column-major index.
    #[inline(always)]
    fn idx(&self, row: i32, col: i32) -> usize {
        (self.rows * col + row) as usize
    }

    /// Multiply one output row by all input columns (encode-side).
    /// Fixed-point path: Q15 × Q15 → Q30, shifted >> 15 with rounding.
    /// Matches C `mapping_matrix_multiply_channel_in_short`.
    pub fn multiply_channel_in_short(
        &self,
        input: &[i16],
        input_rows: i32,
        output: &mut [i16],
        output_row: i32,
        output_rows: i32,
        frame_size: i32,
    ) {
        for i in 0..frame_size as usize {
            let mut tmp: i32 = 0;
            for col in 0..input_rows {
                // Q15 × Q15 >> 8 = Q22 accumulation
                tmp += ((self.data[self.idx(output_row, col)] as i32)
                    * (input[(input_rows as usize) * i + col as usize] as i32))
                    >> 8;
            }
            // Final shift: (tmp + 64) >> 7, total shift = 8+7 = 15
            let out_idx = output_rows as usize * i + output_row as usize;
            output[out_idx] = sat16((tmp + 64) >> 7);
        }
    }

    /// Scatter one input column to all output rows (decode-side).
    /// Fixed-point: Q15 × Q15 → Q30, shift >> 15 with rounding. Accumulates (+=).
    /// Matches C `mapping_matrix_multiply_channel_out_short`.
    pub fn multiply_channel_out_short(
        &self,
        input: &[i16],
        input_row: i32,
        input_rows: i32,
        output: &mut [i16],
        output_rows: i32,
        frame_size: i32,
    ) {
        for i in 0..frame_size as usize {
            let input_sample = input[input_rows as usize * i + input_row as usize] as i32;
            for row in 0..output_rows {
                let tmp = (self.data[self.idx(row, input_row)] as i32) * input_sample;
                let out_idx = (output_rows as usize) * i + row as usize;
                // Accumulate with rounding: += (tmp + 16384) >> 15
                output[out_idx] = sat16(output[out_idx] as i32 + ((tmp + 16384) >> 15));
            }
        }
    }
}

/// Validate matrix dimensions for OGG header constraints.
/// Matches C `mapping_matrix_get_size`.
fn mapping_matrix_valid_size(rows: i32, cols: i32) -> bool {
    if rows > 255 || cols > 255 {
        return false;
    }
    let size = rows as i64 * cols as i64 * 2;
    if size > 65004 {
        return false;
    }
    true
}

// ===========================================================================
// Static ambisonics matrices
// ===========================================================================

// Include the pre-computed ambisonics mixing/demixing matrices.
// All data is Q15 fixed-point, column-major order.

include!("multistream_tables.rs");

// ===========================================================================
// Projection helpers
// ===========================================================================

/// Get order_plus_one from channel count for ambisonics projection.
fn get_order_plus_one_from_channels(channels: i32) -> Result<i32, i32> {
    if channels < 1 || channels > 227 {
        return Err(OPUS_BAD_ARG);
    }
    let order_plus_one = isqrt32(channels as u32) as i32;
    let acn_channels = order_plus_one * order_plus_one;
    let nondiegetic = channels - acn_channels;
    if nondiegetic != 0 && nondiegetic != 2 {
        return Err(OPUS_BAD_ARG);
    }
    Ok(order_plus_one)
}

/// Get stream counts for projection (mapping family 3).
fn get_streams_from_channels(channels: i32, mapping_family: i32) -> Result<(i32, i32, i32), i32> {
    if mapping_family != 3 {
        return Err(OPUS_BAD_ARG);
    }
    let order_plus_one = get_order_plus_one_from_channels(channels)?;
    let streams = (channels + 1) / 2;
    let coupled_streams = channels / 2;
    Ok((streams, coupled_streams, order_plus_one))
}

/// Get pre-computed mixing matrix for the given ambisonics order.
fn get_mixing_matrix_for_order(order_plus_one: i32) -> Result<MappingMatrix, i32> {
    match order_plus_one {
        2 => Ok(MappingMatrix::new(6, 6, 0, &FOA_MIXING_DATA)),
        3 => Ok(MappingMatrix::new(11, 11, 0, &SOA_MIXING_DATA)),
        4 => Ok(MappingMatrix::new(18, 18, 0, &TOA_MIXING_DATA)),
        5 => Ok(MappingMatrix::new(27, 27, 0, &FOURTHOA_MIXING_DATA)),
        6 => Ok(MappingMatrix::new(38, 38, 0, &FIFTHOA_MIXING_DATA)),
        _ => Err(OPUS_BAD_ARG),
    }
}

/// Get pre-computed demixing matrix for the given ambisonics order.
fn get_demixing_matrix_for_order(order_plus_one: i32) -> Result<MappingMatrix, i32> {
    match order_plus_one {
        2 => Ok(MappingMatrix::new(6, 6, 0, &FOA_DEMIXING_DATA)),
        3 => Ok(MappingMatrix::new(11, 11, 3050, &SOA_DEMIXING_DATA)),
        4 => Ok(MappingMatrix::new(18, 18, 0, &TOA_DEMIXING_DATA)),
        5 => Ok(MappingMatrix::new(27, 27, 0, &FOURTHOA_DEMIXING_DATA)),
        6 => Ok(MappingMatrix::new(38, 38, 0, &FIFTHOA_DEMIXING_DATA)),
        _ => Err(OPUS_BAD_ARG),
    }
}

// ===========================================================================
// OpusProjectionEncoder
// ===========================================================================

/// Opus projection encoder — wraps a multistream encoder with matrix mixing.
/// Used for ambisonics (mapping family 3).
pub struct OpusProjectionEncoder {
    mixing_matrix: MappingMatrix,
    demixing_matrix: MappingMatrix,
    ms_encoder: OpusMSEncoder,
}

impl OpusProjectionEncoder {
    /// Create a projection ambisonics encoder.
    /// Matches C `opus_projection_ambisonics_encoder_create`.
    pub fn new(
        fs: i32,
        channels: i32,
        mapping_family: i32,
        application: i32,
    ) -> Result<(Self, i32, i32), i32> {
        let (streams, coupled_streams, order_plus_one) =
            get_streams_from_channels(channels, mapping_family)?;

        let mixing_matrix = get_mixing_matrix_for_order(order_plus_one)?;
        let demixing_matrix = get_demixing_matrix_for_order(order_plus_one)?;

        // Verify matrix dimensions match stream configuration
        if streams + coupled_streams > mixing_matrix.rows
            || channels > mixing_matrix.cols
            || channels > demixing_matrix.rows
            || streams + coupled_streams > demixing_matrix.cols
        {
            return Err(OPUS_BAD_ARG);
        }

        // Trivial mapping: each channel maps to itself
        let mapping: Vec<u8> = (0..channels as u8).collect();

        let ms_encoder = OpusMSEncoder::new(
            fs,
            channels,
            streams,
            coupled_streams,
            &mapping,
            application,
        )?;

        Ok((
            Self {
                mixing_matrix,
                demixing_matrix,
                ms_encoder,
            },
            streams,
            coupled_streams,
        ))
    }

    /// Encode with matrix mixing applied.
    /// Matches C `opus_projection_encode`.
    pub fn encode(
        &mut self,
        pcm: &[i16],
        frame_size: i32,
        data: &mut [u8],
        max_data_bytes: i32,
    ) -> Result<i32, i32> {
        // Apply mixing matrix: transform input channels to stream channels
        let nb_streams = self.ms_encoder.layout.nb_streams;
        let nb_coupled = self.ms_encoder.layout.nb_coupled_streams;
        let input_channels = self.ms_encoder.layout.nb_channels;
        let total_stream_channels = nb_streams + nb_coupled;

        // Create a mixed PCM buffer
        let mut mixed = vec![0i16; total_stream_channels as usize * frame_size as usize];

        // Apply mixing matrix channel by channel
        for ch in 0..total_stream_channels as usize {
            self.mixing_matrix.multiply_channel_in_short(
                pcm,
                input_channels,
                &mut mixed,
                ch as i32,
                total_stream_channels,
                frame_size,
            );
        }

        // Encode the mixed buffer using the multistream encoder
        self.ms_encoder
            .encode(&mixed, frame_size, data, max_data_bytes)
    }

    /// Get the demixing matrix size (for Ogg encapsulation).
    pub fn get_demixing_matrix_size(&self) -> i32 {
        let nb_streams = self.ms_encoder.layout.nb_streams;
        let nb_coupled = self.ms_encoder.layout.nb_coupled_streams;
        let nb_channels = self.ms_encoder.layout.nb_channels;
        nb_channels * (nb_streams + nb_coupled) * 2 // sizeof(i16) = 2
    }

    /// Get the demixing matrix gain (S7.8 dB).
    pub fn get_demixing_matrix_gain(&self) -> i32 {
        self.demixing_matrix.gain
    }

    /// Get the demixing matrix data in little-endian byte format.
    pub fn get_demixing_matrix(&self) -> Vec<u8> {
        let nb_streams = self.ms_encoder.layout.nb_streams;
        let nb_coupled = self.ms_encoder.layout.nb_coupled_streams;
        let nb_channels = self.ms_encoder.layout.nb_channels;
        let nb_input_streams = nb_streams + nb_coupled;
        let nb_output_streams = nb_channels;

        let mut result = vec![0u8; (nb_input_streams * nb_output_streams * 2) as usize];
        let mut l = 0;
        for i in 0..nb_input_streams {
            for j in 0..nb_output_streams {
                let k = (self.demixing_matrix.rows * i + j) as usize;
                let val = self.demixing_matrix.data[k];
                result[2 * l] = val as u8;
                result[2 * l + 1] = (val >> 8) as u8;
                l += 1;
            }
        }
        result
    }

    pub fn get_final_range(&self) -> u32 {
        self.ms_encoder.get_final_range()
    }
    pub fn set_bitrate(&mut self, v: i32) -> i32 {
        self.ms_encoder.set_bitrate(v)
    }
    pub fn get_bitrate(&self) -> i32 {
        self.ms_encoder.get_bitrate()
    }
    pub fn set_complexity(&mut self, v: i32) -> i32 {
        self.ms_encoder.set_complexity(v)
    }
    pub fn set_vbr(&mut self, v: i32) -> i32 {
        self.ms_encoder.set_vbr(v)
    }
    pub fn reset(&mut self) {
        self.ms_encoder.reset();
    }
}

// ===========================================================================
// OpusProjectionDecoder
// ===========================================================================

/// Opus projection decoder — wraps a multistream decoder with matrix demixing.
pub struct OpusProjectionDecoder {
    demixing_matrix: MappingMatrix,
    ms_decoder: OpusMSDecoder,
}

impl OpusProjectionDecoder {
    /// Create a projection decoder from a demixing matrix.
    /// The `demixing_matrix_bytes` is in little-endian i16 format.
    /// Matches C `opus_projection_decoder_create`.
    pub fn new(
        fs: i32,
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        demixing_matrix_bytes: &[u8],
        demixing_matrix_size: i32,
    ) -> Result<Self, i32> {
        let nb_input_streams = streams + coupled_streams;
        let expected_size = nb_input_streams * channels * 2;
        if expected_size != demixing_matrix_size {
            return Err(OPUS_BAD_ARG);
        }

        if !mapping_matrix_valid_size(channels, nb_input_streams) {
            return Err(OPUS_BAD_ARG);
        }

        // Convert little-endian bytes to i16 with sign extension
        let n_elements = (nb_input_streams * channels) as usize;
        let mut buf = vec![0i16; n_elements];
        for i in 0..n_elements {
            let lo = demixing_matrix_bytes[2 * i] as u32;
            let hi = demixing_matrix_bytes[2 * i + 1] as u32;
            let s = (hi << 8) | lo;
            // Sign-extend from 16 bits
            buf[i] = (((s & 0xFFFF) ^ 0x8000).wrapping_sub(0x8000)) as i16;
        }

        let demixing_matrix = MappingMatrix::new(channels, nb_input_streams, 0, &buf);

        // Trivial mapping: each input channel pairs with a matrix column
        let mapping: Vec<u8> = (0..channels as u8).collect();

        let ms_decoder = OpusMSDecoder::new(fs, channels, streams, coupled_streams, &mapping)?;

        Ok(Self {
            demixing_matrix,
            ms_decoder,
        })
    }

    /// Decode with matrix demixing applied.
    /// Matches C `opus_projection_decode`.
    pub fn decode(
        &mut self,
        data: Option<&[u8]>,
        len: i32,
        pcm: &mut [i16],
        frame_size: i32,
        decode_fec: bool,
    ) -> Result<i32, i32> {
        let nb_channels = self.ms_decoder.layout.nb_channels;
        let nb_streams = self.ms_decoder.layout.nb_streams;
        let nb_coupled = self.ms_decoder.layout.nb_coupled_streams;
        let total_stream_channels = nb_streams + nb_coupled;

        // Decode to an intermediate buffer with stream channel layout
        let mut stream_pcm = vec![0i16; total_stream_channels as usize * frame_size as usize];
        let ret = self
            .ms_decoder
            .decode(data, len, &mut stream_pcm, frame_size, decode_fec)?;

        // Zero the output buffer (accumulation pattern requires this)
        for v in pcm[..nb_channels as usize * ret as usize].iter_mut() {
            *v = 0;
        }

        // Apply demixing matrix: scatter each stream channel to all output channels
        for ch in 0..total_stream_channels {
            self.demixing_matrix.multiply_channel_out_short(
                &stream_pcm,
                ch,
                total_stream_channels,
                pcm,
                nb_channels,
                ret,
            );
        }

        Ok(ret)
    }

    pub fn get_final_range(&self) -> u32 {
        self.ms_decoder.get_final_range()
    }
    pub fn get_sample_rate(&self) -> i32 {
        self.ms_decoder.get_sample_rate()
    }
    pub fn set_gain(&mut self, v: i32) -> i32 {
        self.ms_decoder.set_gain(v)
    }
    pub fn get_gain(&self) -> i32 {
        self.ms_decoder.get_gain()
    }
    pub fn reset(&mut self) {
        self.ms_decoder.reset();
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opus::encoder::{
        OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_FRAMESIZE_20_MS,
        OPUS_FRAMESIZE_40_MS, OPUS_SIGNAL_MUSIC,
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

    #[test]
    fn test_validate_layout_valid() {
        let mut layout = ChannelLayout::new();
        layout.nb_channels = 2;
        layout.nb_streams = 1;
        layout.nb_coupled_streams = 1;
        layout.mapping[0] = 0;
        layout.mapping[1] = 1;
        assert!(validate_layout(&layout));
    }

    #[test]
    fn test_validate_layout_invalid() {
        let mut layout = ChannelLayout::new();
        layout.nb_channels = 2;
        layout.nb_streams = 1;
        layout.nb_coupled_streams = 1;
        layout.mapping[0] = 0;
        layout.mapping[1] = 5; // invalid: exceeds max_channel
        assert!(!validate_layout(&layout));
    }

    #[test]
    fn test_validate_layout_muted_channel() {
        let mut layout = ChannelLayout::new();
        layout.nb_channels = 2;
        layout.nb_streams = 1;
        layout.nb_coupled_streams = 0;
        layout.mapping[0] = 0;
        layout.mapping[1] = 255; // muted
        assert!(validate_layout(&layout));
    }

    #[test]
    fn test_get_channels() {
        let mut layout = ChannelLayout::new();
        layout.nb_channels = 6;
        layout.nb_streams = 4;
        layout.nb_coupled_streams = 2;
        // 5.1 mapping
        layout.mapping[0] = 0; // FL -> coupled 0 left
        layout.mapping[1] = 4; // C  -> mono 2
        layout.mapping[2] = 1; // FR -> coupled 0 right
        layout.mapping[3] = 2; // RL -> coupled 1 left
        layout.mapping[4] = 3; // RR -> coupled 1 right
        layout.mapping[5] = 5; // LFE -> mono 3

        assert_eq!(get_left_channel(&layout, 0, -1), 0);
        assert_eq!(get_right_channel(&layout, 0, -1), 2);
        assert_eq!(get_left_channel(&layout, 1, -1), 3);
        assert_eq!(get_right_channel(&layout, 1, -1), 4);
        assert_eq!(get_mono_channel(&layout, 2, -1), 1); // stream 2 = mono, mapping = 0+2=2 → channel mapping[1]=4=2+2 ✗
    }

    #[test]
    fn test_validate_ambisonics() {
        // FOA: 4 channels = 2^2
        let mut ns = 0;
        let mut ncs = 0;
        assert!(validate_ambisonics(4, Some(&mut ns), Some(&mut ncs)));
        assert_eq!(ns, 4); // 4 ACN channels, no coupled
        assert_eq!(ncs, 0);

        // FOA + 2 non-diegetic: 6 channels
        assert!(validate_ambisonics(6, Some(&mut ns), Some(&mut ncs)));
        assert_eq!(ns, 5); // 4 ACN + 1 coupled
        assert_eq!(ncs, 1);

        // SOA: 9 channels = 3^2
        assert!(validate_ambisonics(9, Some(&mut ns), Some(&mut ncs)));
        assert_eq!(ns, 9);
        assert_eq!(ncs, 0);

        // Invalid: 5 channels
        assert!(!validate_ambisonics(5, None, None));

        // Invalid: > 227
        assert!(!validate_ambisonics(228, None, None));
    }

    #[test]
    fn test_ms_encoder_create_mono() {
        let enc = OpusMSEncoder::new(48000, 1, 1, 0, &[0], OPUS_APPLICATION_AUDIO);
        assert!(enc.is_ok());
    }

    #[test]
    fn test_ms_encoder_create_stereo() {
        let enc = OpusMSEncoder::new(48000, 2, 1, 1, &[0, 1], OPUS_APPLICATION_AUDIO);
        assert!(enc.is_ok());
    }

    #[test]
    fn test_ms_encoder_create_invalid() {
        // streams > channels
        let enc = OpusMSEncoder::new(48000, 1, 2, 0, &[0, 1], OPUS_APPLICATION_AUDIO);
        assert!(enc.is_err());
    }

    #[test]
    fn test_ms_decoder_create() {
        let dec = OpusMSDecoder::new(48000, 2, 1, 1, &[0, 1]);
        assert!(dec.is_ok());
    }

    #[test]
    fn test_ms_decoder_create_invalid() {
        // coupled > streams
        let dec = OpusMSDecoder::new(48000, 2, 1, 2, &[0, 1]);
        assert!(dec.is_err());
    }

    #[test]
    fn test_surround_encoder_family0() {
        let result = OpusMSEncoder::new_surround(48000, 2, 0, OPUS_APPLICATION_AUDIO);
        assert!(result.is_ok());
        let (enc, streams, coupled, mapping) = result.unwrap();
        assert_eq!(streams, 1);
        assert_eq!(coupled, 1);
        assert_eq!(&mapping, &[0, 1]);
        assert_eq!(enc.nb_streams(), 1);
    }

    #[test]
    fn test_surround_encoder_family1() {
        let result = OpusMSEncoder::new_surround(48000, 6, 1, OPUS_APPLICATION_AUDIO);
        assert!(result.is_ok());
        let (enc, streams, coupled, mapping) = result.unwrap();
        assert_eq!(streams, 4);
        assert_eq!(coupled, 2);
        assert_eq!(&mapping[..6], &[0, 4, 1, 2, 3, 5]);
        assert_eq!(enc.nb_streams(), 4);
        assert_eq!(enc.nb_coupled_streams(), 2);
    }

    #[test]
    fn test_surround_encoder_family255() {
        let result = OpusMSEncoder::new_surround(48000, 3, 255, OPUS_APPLICATION_AUDIO);
        assert!(result.is_ok());
        let (enc, streams, coupled, mapping) = result.unwrap();
        assert_eq!(streams, 3);
        assert_eq!(coupled, 0);
        assert_eq!(&mapping[..3], &[0, 1, 2]);
        assert_eq!(enc.nb_streams(), 3);
    }

    #[test]
    fn test_surround_family_zero_one_and_constructor_guards() {
        let (enc0, streams0, coupled0, mapping0) =
            OpusMSEncoder::new_surround(48000, 1, 0, OPUS_APPLICATION_AUDIO).unwrap();
        assert_eq!(streams0, 1);
        assert_eq!(coupled0, 0);
        assert_eq!(mapping0[0], 0);
        assert_eq!(enc0.nb_streams(), 1);

        let (enc1, streams1, coupled1, mapping1) =
            OpusMSEncoder::new_surround(48000, 1, 1, OPUS_APPLICATION_AUDIO).unwrap();
        assert_eq!(streams1, 1);
        assert_eq!(coupled1, 0);
        assert_eq!(mapping1[0], 0);
        assert_eq!(enc1.nb_streams(), 1);

        assert!(matches!(
            OpusMSEncoder::new_surround(48000, 4, 3, OPUS_APPLICATION_AUDIO),
            Err(OPUS_UNIMPLEMENTED)
        ));
        assert!(matches!(
            OpusMSEncoder::new(48000, 2, 0, 0, &[0, 1], OPUS_APPLICATION_AUDIO),
            Err(OPUS_BAD_ARG)
        ));
        assert!(matches!(
            OpusMSEncoder::new(48000, 256, 1, 0, &vec![0u8; 256], OPUS_APPLICATION_AUDIO),
            Err(OPUS_BAD_ARG)
        ));
        assert!(matches!(
            OpusMSDecoder::new(48000, 0, 1, 0, &[0]),
            Err(OPUS_BAD_ARG)
        ));
        assert!(matches!(
            OpusMSDecoder::new(48000, 2, 0, 0, &[0, 255]),
            Err(OPUS_BAD_ARG)
        ));
    }

    #[test]
    fn test_surround_encode_and_muted_decode_paths() {
        let (mut enc, streams, coupled, _mapping) =
            OpusMSEncoder::new_surround(48000, 6, 1, OPUS_APPLICATION_AUDIO).unwrap();
        assert_eq!(streams, 4);
        assert_eq!(coupled, 2);

        assert_eq!(enc.set_vbr(0), OPUS_OK);
        assert_eq!(enc.set_bitrate(24_000), OPUS_OK);

        let pcm = patterned_pcm_i16(480, 6, 17);
        let mut packet = vec![0u8; 4000];
        let len = enc.encode(&pcm, 480, &mut packet, 4000).unwrap();
        assert!(len > 0);
        assert_ne!(enc.get_final_range(), 0);

        let mono_pcm = patterned_pcm_i16(960, 1, 23);
        let mut mono_enc =
            OpusMSEncoder::new(48000, 1, 1, 0, &[0], OPUS_APPLICATION_AUDIO).unwrap();
        let mut mono_packet = vec![0u8; 4000];
        let mono_len = mono_enc
            .encode(&mono_pcm, 960, &mut mono_packet, 4000)
            .unwrap();
        assert!(mono_len > 0);

        let mut muted_dec = OpusMSDecoder::new(48000, 2, 1, 0, &[0, 255]).unwrap();
        let mut muted_out = vec![123i16; 960 * 2];
        let muted_decoded = muted_dec
            .decode(
                Some(&mono_packet[..mono_len as usize]),
                mono_len,
                &mut muted_out,
                960,
                false,
            )
            .unwrap();
        assert_eq!(muted_decoded, 960);
        assert!(muted_out.iter().step_by(2).any(|&sample| sample != 123));
        assert!(muted_out
            .iter()
            .skip(1)
            .step_by(2)
            .all(|&sample| sample == 0));
    }

    #[test]
    fn test_mapping_matrix_multiply_identity() {
        // 2x2 identity matrix (Q15)
        let data: [i16; 4] = [32767, 0, 0, 32767]; // column-major
        let mat = MappingMatrix::new(2, 2, 0, &data);

        let input: [i16; 4] = [1000, 2000, 3000, 4000]; // 2 channels, 2 samples
        let mut output = [0i16; 4];

        // Multiply channel 0 (first row)
        mat.multiply_channel_in_short(&input, 2, &mut output, 0, 2, 2);
        // Multiply channel 1 (second row)
        mat.multiply_channel_in_short(&input, 2, &mut output, 1, 2, 2);

        // Should be close to identity (within rounding)
        assert!((output[0] as i32 - 1000).abs() <= 1);
        assert!((output[1] as i32 - 2000).abs() <= 1);
        assert!((output[2] as i32 - 3000).abs() <= 1);
        assert!((output[3] as i32 - 4000).abs() <= 1);
    }

    #[test]
    fn test_mapping_matrix_multiply_out() {
        // 2x2 identity matrix (Q15)
        let data: [i16; 4] = [32767, 0, 0, 32767];
        let mat = MappingMatrix::new(2, 2, 0, &data);

        let input: [i16; 4] = [1000, 2000, 3000, 4000]; // 2 channels, 2 samples
        let mut output = [0i16; 4];

        // Scatter channel 0
        mat.multiply_channel_out_short(&input, 0, 2, &mut output, 2, 2);
        // Scatter channel 1
        mat.multiply_channel_out_short(&input, 1, 2, &mut output, 2, 2);

        assert!((output[0] as i32 - 1000).abs() <= 1);
        assert!((output[1] as i32 - 2000).abs() <= 1);
        assert!((output[2] as i32 - 3000).abs() <= 1);
        assert!((output[3] as i32 - 4000).abs() <= 1);
    }

    #[test]
    fn test_log_sum_equal_values() {
        // log2(2^a + 2^a) = log2(2 * 2^a) = a + 1
        // In fixed-point (Q24), a = 4.0 → 4 << 24 = 67108864
        // Expected: ~5.0 → 83886080
        let a = gconst(4.0);
        let result = log_sum(a, a);
        // log_sum should approximate a + 0.5 (from the table), giving ~4.5
        // Actually log2(2^a + 2^a) = a + 1, but the function uses base-2 logs
        // With equal inputs, diff=0, so result = max + diff_table[0] = a + 0.5
        let expected = a + gconst(0.5);
        assert!((result - expected).abs() < gconst(0.01));
    }

    #[test]
    fn test_log_sum_large_diff() {
        // When difference >= 8, return max
        let a = gconst(10.0);
        let b = gconst(1.0);
        let result = log_sum(a, b);
        assert_eq!(result, a);
    }

    #[test]
    fn test_ms_encode_decode_mono() {
        let mut enc = OpusMSEncoder::new(48000, 1, 1, 0, &[0], OPUS_APPLICATION_AUDIO).unwrap();
        let mut dec = OpusMSDecoder::new(48000, 1, 1, 0, &[0]).unwrap();

        let frame_size = 960;
        let pcm_in = vec![0i16; frame_size];
        let mut packet = vec![0u8; 4000];
        let mut pcm_out = vec![0i16; frame_size];

        let len = enc
            .encode(&pcm_in, frame_size as i32, &mut packet, 4000)
            .unwrap();
        assert!(len > 0);

        let samples = dec
            .decode(
                Some(&packet[..len as usize]),
                len,
                &mut pcm_out,
                frame_size as i32,
                false,
            )
            .unwrap();
        assert_eq!(samples, frame_size as i32);
    }

    #[test]
    fn test_ms_encode_decode_stereo() {
        let mut enc = OpusMSEncoder::new(48000, 2, 1, 1, &[0, 1], OPUS_APPLICATION_AUDIO).unwrap();
        let mut dec = OpusMSDecoder::new(48000, 2, 1, 1, &[0, 1]).unwrap();

        let frame_size = 960;
        let pcm_in = vec![0i16; frame_size * 2];
        let mut packet = vec![0u8; 4000];
        let mut pcm_out = vec![0i16; frame_size * 2];

        let len = enc
            .encode(&pcm_in, frame_size as i32, &mut packet, 4000)
            .unwrap();
        assert!(len > 0);

        let samples = dec
            .decode(
                Some(&packet[..len as usize]),
                len,
                &mut pcm_out,
                frame_size as i32,
                false,
            )
            .unwrap();
        assert_eq!(samples, frame_size as i32);
    }

    #[test]
    fn test_final_range_xor() {
        let mut enc = OpusMSEncoder::new(48000, 2, 1, 1, &[0, 1], OPUS_APPLICATION_AUDIO).unwrap();
        let frame_size = 960;
        let pcm = vec![0i16; frame_size * 2];
        let mut packet = vec![0u8; 4000];
        enc.encode(&pcm, frame_size as i32, &mut packet, 4000)
            .unwrap();
        let range = enc.get_final_range();
        // Just verify it's non-zero after encoding
        assert_ne!(range, 0);
    }

    #[test]
    fn test_decoder_plc() {
        let mut dec = OpusMSDecoder::new(48000, 1, 1, 0, &[0]).unwrap();
        let frame_size = 960;
        let mut pcm_out = vec![0i16; frame_size];

        // First, encode a real frame for the decoder to have state
        let mut enc = OpusMSEncoder::new(48000, 1, 1, 0, &[0], OPUS_APPLICATION_AUDIO).unwrap();
        let pcm_in = vec![0i16; frame_size];
        let mut packet = vec![0u8; 4000];
        let len = enc
            .encode(&pcm_in, frame_size as i32, &mut packet, 4000)
            .unwrap();
        dec.decode(
            Some(&packet[..len as usize]),
            len,
            &mut pcm_out,
            frame_size as i32,
            false,
        )
        .unwrap();

        // Now do PLC (None data)
        let samples = dec
            .decode(None, 0, &mut pcm_out, frame_size as i32, false)
            .unwrap();
        assert_eq!(samples, frame_size as i32);
    }

    #[test]
    fn test_channel_pos_5_1() {
        let mut pos = [0i32; 8];
        channel_pos(6, &mut pos);
        assert_eq!(pos[0], 1); // FL = left
        assert_eq!(pos[1], 2); // C = center
        assert_eq!(pos[2], 3); // FR = right
        assert_eq!(pos[3], 1); // RL = left
        assert_eq!(pos[4], 3); // RR = right
        assert_eq!(pos[5], 0); // LFE = don't mix
    }

    #[test]
    fn test_rate_allocation_auto() {
        let mut layout = ChannelLayout::new();
        layout.nb_channels = 2;
        layout.nb_streams = 1;
        layout.nb_coupled_streams = 1;

        let mut rate = [0i32; 1];
        surround_rate_allocation(&layout, -1, OPUS_AUTO, &mut rate, 960, 48000);
        assert!(rate[0] > 0);
    }

    #[test]
    fn test_ambisonics_rate_allocation() {
        let mut layout = ChannelLayout::new();
        layout.nb_channels = 4;
        layout.nb_streams = 4;
        layout.nb_coupled_streams = 0;

        let mut rate = [0i32; 4];
        ambisonics_rate_allocation(&layout, 128000, &mut rate, 960, 48000);
        // Equal allocation
        assert_eq!(rate[0], rate[1]);
        assert_eq!(rate[1], rate[2]);
        assert_eq!(rate[2], rate[3]);
        assert_eq!(rate[0], 32000);
    }

    #[test]
    fn test_mapping_matrix_valid_size_limits() {
        assert!(mapping_matrix_valid_size(2, 2));
        assert!(mapping_matrix_valid_size(255, 127)); // 255*127*2 = 64770 < 65004
        assert!(!mapping_matrix_valid_size(256, 1)); // exceeds 255
        assert!(!mapping_matrix_valid_size(255, 128)); // 255*128*2 = 65280 > 65004
    }

    #[test]
    fn test_validate_encoder_layout_missing_channels() {
        let mut coupled = ChannelLayout::new();
        coupled.nb_channels = 2;
        coupled.nb_streams = 1;
        coupled.nb_coupled_streams = 1;
        coupled.mapping[0] = 0;
        coupled.mapping[1] = 255;
        assert!(!validate_encoder_layout(&coupled));

        let mut mono = ChannelLayout::new();
        mono.nb_channels = 3;
        mono.nb_streams = 2;
        mono.nb_coupled_streams = 1;
        mono.mapping[0] = 0;
        mono.mapping[1] = 1;
        mono.mapping[2] = 255;
        assert!(!validate_encoder_layout(&mono));
    }

    #[test]
    fn test_parse_size_field_and_subpacket_variants() {
        assert_eq!(parse_size_field(&[], 0), (-1, -1));
        assert_eq!(parse_size_field(&[251], 1), (1, 251));
        assert_eq!(parse_size_field(&[252], 1), (-1, -1));
        assert_eq!(parse_size_field(&[252, 1], 2), (2, 256));

        assert_eq!(parse_multistream_subpacket(&[0x00], 1, false), 1);
        assert_eq!(
            parse_multistream_subpacket(&[0x00, 0x05, 0, 0, 0, 0, 0], 7, true),
            7
        );
        assert_eq!(
            parse_multistream_subpacket(&[0x01, 0x03, 0, 0, 0, 0, 0, 0], 8, true),
            8
        );
        assert_eq!(
            parse_multistream_subpacket(&[0x02, 0x02, 0x03, 0, 0, 0, 0, 0], 8, true),
            8
        );
        assert_eq!(parse_multistream_subpacket(&[0x03, 0x02], 2, false), 2);
        assert!(parse_multistream_subpacket(&[0x03, 0x42, 0x01], 3, false) < 0);
        assert_eq!(
            parse_multistream_subpacket(&[0x03, 0x82, 0x01, 0x00], 4, false),
            3
        );
        assert_eq!(parse_multistream_subpacket(&[0x03, 0x02, 0x01], 3, true), 4);
        assert!(parse_multistream_subpacket(&[0x03], 1, false) < 0);
        assert!(parse_multistream_subpacket(&[0x03, 0x00], 2, false) < 0);
        assert!(parse_multistream_subpacket(&[0x03, 0xB1], 2, false) < 0);
        assert!(parse_multistream_subpacket(&[0x03, 0x41, 0xFF], 3, false) < 0);
    }

    #[test]
    fn test_channel_pos_branches_and_matrix_helpers() {
        let mut pos = [0i32; 8];

        channel_pos(4, &mut pos);
        assert_eq!(pos, [1, 3, 1, 3, 0, 0, 0, 0]);

        channel_pos(7, &mut pos);
        assert_eq!(pos, [1, 2, 3, 1, 3, 2, 0, 0]);

        channel_pos(8, &mut pos);
        assert_eq!(pos, [1, 2, 3, 1, 3, 1, 3, 0]);

        channel_pos(2, &mut pos);
        assert_eq!(pos, [0, 0, 0, 0, 0, 0, 0, 0]);

        assert!(mapping_matrix_valid_size(1, 1));
        assert!(!mapping_matrix_valid_size(256, 1));
        assert!(!mapping_matrix_valid_size(255, 128));

        let mixing = get_mixing_matrix_for_order(3).unwrap();
        let demixing = get_demixing_matrix_for_order(3).unwrap();
        assert_eq!((mixing.rows, mixing.cols, mixing.gain), (11, 11, 0));
        assert_eq!(
            (demixing.rows, demixing.cols, demixing.gain),
            (11, 11, 3050)
        );

        assert!(matches!(get_mixing_matrix_for_order(7), Err(OPUS_BAD_ARG)));
        assert!(matches!(
            get_demixing_matrix_for_order(7),
            Err(OPUS_BAD_ARG)
        ));
    }

    #[test]
    fn test_channel_pos_three_and_five_paths() {
        let mut pos = [0i32; 8];

        channel_pos(3, &mut pos);
        assert_eq!(pos, [1, 2, 3, 1, 3, 0, 0, 0]);

        channel_pos(5, &mut pos);
        assert_eq!(pos, [1, 2, 3, 1, 3, 0, 0, 0]);
    }

    #[test]
    fn test_matrix_order_paths_four_five_six() {
        for (order_plus_one, expected_dim) in [(4, 18), (5, 27), (6, 38)] {
            let mixing = get_mixing_matrix_for_order(order_plus_one).unwrap();
            let demixing = get_demixing_matrix_for_order(order_plus_one).unwrap();
            assert_eq!((mixing.rows, mixing.cols, mixing.gain), (expected_dim, expected_dim, 0));
            assert_eq!(
                (demixing.rows, demixing.cols, demixing.gain),
                (expected_dim, expected_dim, 0)
            );
        }
    }

    #[test]
    fn test_layout_helper_branches_and_encoder_constructor_errors() {
        let mut layout = ChannelLayout::new();
        layout.nb_channels = 4;
        layout.nb_streams = 2;
        layout.nb_coupled_streams = 1;
        layout.mapping[0] = 0;
        layout.mapping[1] = 0;
        layout.mapping[2] = 1;
        layout.mapping[3] = 2;

        assert!(validate_layout(&layout));
        assert!(validate_encoder_layout(&layout));
        assert_eq!(get_left_channel(&layout, 0, -1), 0);
        assert_eq!(get_left_channel(&layout, 0, 0), 1);
        assert_eq!(get_right_channel(&layout, 0, -1), 2);
        assert_eq!(get_right_channel(&layout, 0, 2), -1);
        assert_eq!(get_mono_channel(&layout, 1, -1), 3);
        assert_eq!(get_mono_channel(&layout, 1, 3), -1);

        let mut missing_left = layout.clone();
        missing_left.mapping[0] = 1;
        missing_left.mapping[1] = 1;
        assert!(!validate_encoder_layout(&missing_left));

        let mut missing_right = layout.clone();
        missing_right.mapping[2] = 2;
        assert!(!validate_encoder_layout(&missing_right));

        let mut missing_mono = layout.clone();
        missing_mono.mapping[3] = 255;
        assert!(!validate_encoder_layout(&missing_mono));

        let mut too_many_channels = layout.clone();
        too_many_channels.nb_channels = 256;
        too_many_channels.nb_streams = 200;
        too_many_channels.nb_coupled_streams = 56;
        assert!(!validate_layout(&too_many_channels));

        assert!(matches!(
            OpusMSEncoder::new(12345, 1, 1, 0, &[0], OPUS_APPLICATION_AUDIO),
            Err(OPUS_BAD_ARG)
        ));
        assert!(matches!(
            OpusMSEncoder::new(48000, 2, 1, 1, &[0, 255], OPUS_APPLICATION_AUDIO),
            Err(OPUS_BAD_ARG)
        ));
    }

    #[test]
    fn test_parse_multistream_subpacket_branch_matrix() {
        assert_eq!(parse_multistream_subpacket(&[0x00, 0x00], 2, true), 2);
        assert_eq!(parse_multistream_subpacket(&[0x01, 0x00], 2, true), 2);
        assert_eq!(parse_multistream_subpacket(&[0x02, 0x00, 0x00], 3, true), 3);
        assert_eq!(parse_multistream_subpacket(&[0x03, 0x03, 0x00], 3, true), 3);

        // Padding branch, code 3 CBR path.
        assert_eq!(
            parse_multistream_subpacket(&[0x03, 0x43, 0x01, 0xAA, 0xBB], 5, false),
            4
        );

        // VBR branch, code 3 with two size fields and no self-delimiting size.
        assert_eq!(
            parse_multistream_subpacket(
                &[0x03, 0x83, 0x01, 0x01, 0x01, 0xAA, 0xBB, 0xCC],
                8,
                false
            ),
            6
        );

        // Self-delimited code 2 path parses the explicit last frame size.
        assert_eq!(parse_multistream_subpacket(&[0x02, 0x00, 0x00], 3, true), 3);
    }

    #[test]
    fn test_encoder_accessors_reset_and_child_propagation() {
        let (mut enc, streams, coupled, mapping) =
            OpusMSEncoder::new_surround(48000, 6, 1, OPUS_APPLICATION_AUDIO).unwrap();

        assert_eq!(streams, 4);
        assert_eq!(coupled, 2);
        assert_eq!(&mapping[..6], &[0, 4, 1, 2, 3, 5]);
        assert_eq!(enc.get_application(), OPUS_APPLICATION_AUDIO);
        assert_eq!(enc.get_sample_rate(), 48000);
        assert_eq!(
            enc.get_lookahead(),
            enc.get_encoder(0).unwrap().ms_get_lookahead()
        );
        assert!(enc.get_encoder(1).is_some());
        assert!(enc.get_encoder(99).is_none());
        assert!(enc.get_encoder_mut(0).is_some());

        assert_eq!(
            enc.set_application(OPUS_APPLICATION_RESTRICTED_LOWDELAY),
            OPUS_OK
        );
        assert_eq!(enc.get_application(), OPUS_APPLICATION_RESTRICTED_LOWDELAY);
        enc.set_expert_frame_duration(OPUS_FRAMESIZE_40_MS);
        assert_eq!(enc.get_expert_frame_duration(), OPUS_FRAMESIZE_40_MS);
        assert_eq!(enc.set_complexity(7), OPUS_OK);
        assert_eq!(enc.set_vbr(0), OPUS_OK);
        assert_eq!(enc.set_vbr_constraint(1), OPUS_OK);
        assert_eq!(enc.set_signal(OPUS_SIGNAL_MUSIC), OPUS_OK);
        assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_FULLBAND), OPUS_OK);
        assert_eq!(enc.set_max_bandwidth(OPUS_BANDWIDTH_WIDEBAND), OPUS_OK);
        assert_eq!(enc.set_inband_fec(1), OPUS_OK);
        assert_eq!(enc.set_packet_loss_perc(12), OPUS_OK);
        assert_eq!(enc.set_dtx(1), OPUS_OK);
        assert_eq!(enc.set_lsb_depth(12), OPUS_OK);
        assert_eq!(enc.set_prediction_disabled(1), OPUS_OK);
        assert_eq!(enc.set_phase_inversion_disabled(1), OPUS_OK);
        assert_eq!(enc.set_force_mode(MODE_CELT_ONLY), OPUS_OK);
        assert_eq!(enc.set_force_channels(1), OPUS_OK);

        {
            let child = enc.get_encoder_mut(0).unwrap();
            child.ms_set_variable_duration(OPUS_FRAMESIZE_20_MS);
            child.ms_set_lsb_depth(14);
        }

        let child = enc.get_encoder(0).unwrap();
        assert_eq!(child.ms_get_complexity(), 7);
        assert_eq!(child.ms_get_vbr(), 0);
        assert_eq!(child.ms_get_variable_duration(), OPUS_FRAMESIZE_20_MS);
        assert_eq!(child.ms_get_lsb_depth(), 14);
        assert_eq!(child.ms_get_lookahead(), enc.get_lookahead());

        enc.preemph_mem[0] = 17;
        enc.window_mem[0] = 29;
        enc.reset();
        assert!(enc.preemph_mem.iter().all(|&v| v == 0));
        assert!(enc.window_mem.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_encoder_bitrate_special_values_and_non_surround_reset() {
        let mut enc = OpusMSEncoder::new(48000, 2, 1, 1, &[0, 1], OPUS_APPLICATION_AUDIO).unwrap();
        let channels = enc.layout.nb_channels;

        assert_eq!(enc.set_bitrate(0), OPUS_BAD_ARG);

        assert_eq!(enc.set_bitrate(1), OPUS_OK);
        assert_eq!(enc.bitrate_bps, 500 * channels);

        assert_eq!(enc.set_bitrate(2_000_000_000), OPUS_OK);
        assert_eq!(enc.bitrate_bps, 750000 * channels);

        assert_eq!(enc.set_bitrate(OPUS_AUTO), OPUS_OK);
        assert_eq!(enc.bitrate_bps, OPUS_AUTO);

        assert_eq!(enc.set_bitrate(OPUS_BITRATE_MAX), OPUS_OK);
        assert_eq!(enc.bitrate_bps, OPUS_BITRATE_MAX);

        enc.reset();
        assert!(enc.preemph_mem.is_empty());
        assert!(enc.window_mem.is_empty());
    }

    #[test]
    fn test_projection_helper_branches_and_family2_constructor() {
        assert_eq!(get_order_plus_one_from_channels(0), Err(OPUS_BAD_ARG));
        assert_eq!(get_order_plus_one_from_channels(4), Ok(2));
        assert_eq!(get_order_plus_one_from_channels(6), Ok(2));
        assert_eq!(get_order_plus_one_from_channels(13), Err(OPUS_BAD_ARG));
        assert_eq!(get_order_plus_one_from_channels(228), Err(OPUS_BAD_ARG));

        assert_eq!(get_streams_from_channels(4, 2), Err(OPUS_BAD_ARG));
        assert_eq!(get_streams_from_channels(4, 3), Ok((2, 2, 2)));
        assert_eq!(get_streams_from_channels(6, 3), Ok((3, 3, 2)));

        assert!(get_mixing_matrix_for_order(2).is_ok());
        assert!(get_demixing_matrix_for_order(3).is_ok());
        assert!(matches!(get_mixing_matrix_for_order(7), Err(OPUS_BAD_ARG)));
        assert!(matches!(
            get_demixing_matrix_for_order(7),
            Err(OPUS_BAD_ARG)
        ));

        let (enc, streams, coupled, mapping) =
            OpusMSEncoder::new_surround(48000, 4, 2, OPUS_APPLICATION_AUDIO).unwrap();
        assert_eq!(streams, 4);
        assert_eq!(coupled, 0);
        assert_eq!(&mapping[..4], &[0, 1, 2, 3]);
        assert_eq!(enc.nb_streams(), 4);
        assert_eq!(enc.nb_coupled_streams(), 0);
    }

    #[test]
    fn test_projection_encoder_wrapper_methods() {
        let (mut enc, streams, coupled) =
            OpusProjectionEncoder::new(48000, 9, 3, OPUS_APPLICATION_AUDIO).unwrap();
        assert_eq!(streams, 5);
        assert_eq!(coupled, 4);

        assert_eq!(enc.set_bitrate(OPUS_AUTO), OPUS_OK);
        assert_eq!(enc.ms_encoder.bitrate_bps, OPUS_AUTO);
        assert_eq!(enc.set_bitrate(OPUS_BITRATE_MAX), OPUS_OK);
        assert_eq!(enc.ms_encoder.bitrate_bps, OPUS_BITRATE_MAX);
        assert_eq!(enc.set_complexity(4), OPUS_OK);
        assert_eq!(enc.set_vbr(0), OPUS_OK);
        assert_eq!(enc.get_bitrate(), enc.ms_encoder.get_bitrate());

        let pcm = patterned_pcm_i16(960, 9, 71);
        let mut packet = vec![0u8; 5000];
        let len = enc.encode(&pcm, 960, &mut packet, 5000).unwrap();
        assert!(len > 0);
        assert_eq!(enc.get_final_range(), enc.ms_encoder.get_final_range());
        assert_eq!(enc.ms_encoder.get_encoder(0).unwrap().ms_get_complexity(), 4);
        assert_eq!(enc.ms_encoder.get_encoder(0).unwrap().ms_get_vbr(), 0);

        enc.reset();
        assert_eq!(enc.get_final_range(), 0);
    }

    #[test]
    fn test_ms_decoder_wrapper_methods_multi_stream() {
        let (mut enc, streams, coupled, mapping) =
            OpusMSEncoder::new_surround(48000, 6, 1, OPUS_APPLICATION_AUDIO).unwrap();
        let pcm = patterned_pcm_i16(960, 6, 73);
        let mut packet = vec![0u8; 4000];
        let len = enc.encode(&pcm, 960, &mut packet, 4000).unwrap();

        let mut dec = OpusMSDecoder::new(48000, 6, streams, coupled, &mapping).unwrap();
        let mut out = vec![0i16; 960 * 6];
        assert_eq!(
            dec.decode(Some(&packet[..len as usize]), len, &mut out, 960, false)
                .unwrap(),
            960
        );

        let _ = dec.get_bandwidth();
        assert_eq!(dec.get_last_packet_duration(), 960);
        assert_eq!(dec.set_gain(222), OPUS_OK);
        assert_eq!(dec.get_gain(), 222);
        assert_eq!(dec.set_phase_inversion_disabled(1), OPUS_OK);
        assert_eq!(dec.get_phase_inversion_disabled(), 1);
        assert_eq!(dec.set_complexity(6), OPUS_OK);
        assert_eq!(dec.get_complexity(), 6);

        for i in 0..streams as usize {
            let child = dec.get_decoder(i).unwrap();
            assert_eq!(child.ms_get_gain(), 222);
            assert_eq!(child.ms_get_phase_inversion_disabled(), 1);
            assert_eq!(child.ms_get_complexity(), 6);
        }
        assert!(dec.get_decoder(streams as usize).is_none());
        assert!(dec.get_decoder_mut(0).is_some());
        assert_ne!(dec.get_final_range(), 0);

        dec.reset();
        assert_eq!(dec.get_final_range(), 0);
        assert_eq!(dec.get_last_packet_duration(), 0);
    }

    #[test]
    fn test_ms_decoder_rejects_invalid_lengths_and_small_frames() {
        let (mut enc, streams, coupled, mapping) =
            OpusMSEncoder::new_surround(48000, 6, 1, OPUS_APPLICATION_AUDIO)
                .expect("surround multistream encoder");
        let mut packet = vec![0u8; 4000];
        let pcm = patterned_pcm_i16(960, 6, 17);
        let len = enc
            .encode(&pcm, 960, &mut packet, 4000)
            .expect("encode valid multistream packet");

        let mut dec = OpusMSDecoder::new(48000, 6, streams, coupled, &mapping)
            .expect("surround multistream decoder");
        let mut out = vec![0i16; 960 * 6];

        assert_eq!(
            dec.decode(Some(&packet[..len as usize]), -1, &mut out, 960, false),
            Err(OPUS_BAD_ARG)
        );
        assert_eq!(
            dec.decode(Some(&packet[..1]), 1, &mut out, 960, false),
            Err(OPUS_INVALID_PACKET)
        );
        assert_eq!(
            dec.decode(Some(&packet[..len as usize]), len, &mut out, 0, false),
            Err(OPUS_BAD_ARG)
        );
        assert_eq!(
            dec.decode(Some(&packet[..len as usize]), 0, &mut out, 960, false),
            Ok(960)
        );
        assert_eq!(
            dec.decode(Some(&packet[..len as usize]), len, &mut out, 120, false),
            Err(OPUS_BUFFER_TOO_SMALL)
        );
        assert_eq!(
            dec.decode(Some(&packet[..len as usize]), len, &mut out, 960, false),
            Ok(960)
        );
        assert!(out.iter().any(|&sample| sample != 0));
    }

    #[test]
    fn test_decoder_accessors_reset_and_child_propagation() {
        let mut enc = OpusMSEncoder::new(48000, 2, 1, 1, &[0, 1], OPUS_APPLICATION_AUDIO).unwrap();
        let pcm = patterned_pcm_i16(960, 2, 31);
        let mut packet = vec![0u8; 4000];
        let len = enc.encode(&pcm, 960, &mut packet, 4000).unwrap();

        let mut dec = OpusMSDecoder::new(48000, 2, 1, 1, &[0, 1]).unwrap();
        let mut out = vec![0i16; 960 * 2];
        assert_eq!(dec.get_sample_rate(), 48000);
        assert_eq!(dec.get_bandwidth(), 0);
        assert_eq!(dec.get_last_packet_duration(), 0);
        assert!(dec.get_decoder(0).is_some());
        assert!(dec.get_decoder(1).is_none());
        assert!(dec.get_decoder_mut(0).is_some());

        dec.set_gain(123);
        dec.set_phase_inversion_disabled(1);
        dec.set_complexity(7);
        assert_eq!(dec.get_gain(), 123);
        assert_eq!(dec.get_phase_inversion_disabled(), 1);
        assert_eq!(dec.get_complexity(), 7);

        let decoded = dec
            .decode(Some(&packet[..len as usize]), len, &mut out, 960, false)
            .unwrap();
        assert_eq!(decoded, 960);
        assert_ne!(dec.get_final_range(), 0);
        assert_eq!(dec.get_last_packet_duration(), 960);

        dec.reset();
        assert_eq!(dec.get_final_range(), 0);
        assert_eq!(dec.get_last_packet_duration(), 0);
        assert_eq!(dec.get_gain(), 123);
        assert_eq!(dec.get_phase_inversion_disabled(), 1);
        assert_eq!(dec.get_complexity(), 7);
    }

    #[test]
    fn test_surround_and_ambisonics_construction_paths() {
        assert!(matches!(
            OpusMSEncoder::new_surround(48000, 3, 0, OPUS_APPLICATION_AUDIO),
            Err(OPUS_UNIMPLEMENTED)
        ));
        assert!(matches!(
            OpusMSEncoder::new_surround(48000, 9, 1, OPUS_APPLICATION_AUDIO),
            Err(OPUS_UNIMPLEMENTED)
        ));
        assert!(matches!(
            OpusMSEncoder::new_surround(48000, 5, 2, OPUS_APPLICATION_AUDIO),
            Err(OPUS_BAD_ARG)
        ));

        let (mut enc, streams, coupled, mapping) =
            OpusMSEncoder::new_surround(48000, 9, 2, OPUS_APPLICATION_AUDIO)
                .expect("ambisonics multistream encoder");
        assert_eq!(streams, 9);
        assert_eq!(coupled, 0);
        assert_eq!(&mapping[..9], &[0, 1, 2, 3, 4, 5, 6, 7, 8]);

        let pcm = patterned_pcm_i16(960, 9, 23);
        let mut packet = vec![0u8; 4000];
        let len = enc
            .encode(&pcm, 960, &mut packet, 4000)
            .expect("ambisonics encode");
        assert!(len > 0);
        assert_eq!(enc.nb_streams(), 9);
        assert_eq!(enc.nb_coupled_streams(), 0);
        assert!(enc.get_final_range() != 0);

        let mut dec = OpusMSDecoder::new(48000, 9, streams, coupled, &mapping)
            .expect("ambisonics multistream decoder");
        let mut out = vec![0i16; 960 * 9];
        let decoded = dec
            .decode(Some(&packet[..len as usize]), len, &mut out, 960, false)
            .expect("ambisonics decode");
        assert_eq!(decoded, 960);
        assert_eq!(dec.nb_streams(), 9);
        assert_eq!(dec.nb_coupled_streams(), 0);
        assert!(out.iter().any(|&sample| sample != 0));
    }

    #[test]
    fn test_projection_encoder_decoder_roundtrip_and_invalid_inputs() {
        assert!(matches!(
            OpusProjectionEncoder::new(48000, 5, 3, OPUS_APPLICATION_AUDIO),
            Err(OPUS_BAD_ARG)
        ));
        assert!(matches!(
            OpusProjectionEncoder::new(48000, 4, 2, OPUS_APPLICATION_AUDIO),
            Err(OPUS_BAD_ARG)
        ));
        assert!(matches!(
            OpusProjectionDecoder::new(48000, 5, 5, 0, &[0u8; 48], 48),
            Err(OPUS_BAD_ARG)
        ));

        let (mut enc, streams, coupled) =
            OpusProjectionEncoder::new(48000, 9, 3, OPUS_APPLICATION_AUDIO)
                .expect("projection encoder");
        assert_eq!(streams, 5);
        assert_eq!(coupled, 4);
        assert_eq!(enc.get_demixing_matrix_gain(), 3050);
        assert_eq!(enc.get_demixing_matrix_size(), 9 * (streams + coupled) * 2);

        let demixing = enc.get_demixing_matrix();
        let mut dec = OpusProjectionDecoder::new(
            48000,
            9,
            streams,
            coupled,
            &demixing,
            enc.get_demixing_matrix_size(),
        )
        .expect("projection decoder");
        assert_eq!(dec.get_sample_rate(), 48000);

        let pcm = vec![0i16; 960 * 9];
        let mut packet = vec![0u8; 4000];
        let len = enc
            .encode(&pcm, 960, &mut packet, 4000)
            .expect("projection encode");
        assert!(len > 0);

        let mut out = vec![1i16; 960 * 9];
        let decoded = dec
            .decode(Some(&packet[..len as usize]), len, &mut out, 960, false)
            .expect("projection decode");
        assert_eq!(decoded, 960);
        assert!(out.iter().all(|&sample| sample == 0));
    }

    #[test]
    fn test_projection_decoder_wrapper_methods() {
        let (mut enc, streams, coupled) =
            OpusProjectionEncoder::new(48000, 9, 3, OPUS_APPLICATION_AUDIO).unwrap();
        let demixing = enc.get_demixing_matrix();
        let matrix_size = enc.get_demixing_matrix_size();
        let mut dec = OpusProjectionDecoder::new(48000, 9, streams, coupled, &demixing, matrix_size)
            .unwrap();

        let pcm = patterned_pcm_i16(960, 9, 79);
        let mut packet = vec![0u8; 5000];
        let len = enc.encode(&pcm, 960, &mut packet, 5000).unwrap();
        let mut out = vec![0i16; 960 * 9];
        assert_eq!(
            dec.decode(Some(&packet[..len as usize]), len, &mut out, 960, false)
                .unwrap(),
            960
        );

        assert_eq!(dec.get_sample_rate(), 48000);
        assert_ne!(dec.get_final_range(), 0);
        assert_eq!(dec.set_gain(345), OPUS_OK);
        assert_eq!(dec.get_gain(), 345);
        dec.reset();
        assert_eq!(dec.get_final_range(), 0);
    }

    // --- Coverage additions: ambisonics, coupled streams, error handling ---

    #[test]
    fn test_ambisonics_invalid_channel_counts() {
        // Family 2 = ambisonics. Invalid channel counts for ambisonics.
        for ch in [0, 2, 5, 7, 8, 10] {
            let result = OpusMSEncoder::new_surround(48000, ch, 2, OPUS_APPLICATION_AUDIO);
            assert!(result.is_err(), "channels={ch} should be invalid for ambisonics family 2");
        }
    }

    #[test]
    fn test_ambisonics_valid_channel_counts() {
        // Family 2 = ambisonics. Valid: N^2 or N^2 + 2.
        // N=1: 1, 3; N=2: 4, 6; N=3: 9, 11; N=4: 16
        for ch in [1, 3, 4, 6, 9, 11, 16] {
            let result = OpusMSEncoder::new_surround(48000, ch, 2, OPUS_APPLICATION_AUDIO);
            assert!(result.is_ok(), "channels={ch} should be valid for ambisonics, got {:?}", result.err());
        }
    }

    #[test]
    fn test_ms_encoder_coupled_stereo_roundtrip() {
        let mut enc = OpusMSEncoder::new(48000, 2, 1, 1, &[0, 1], OPUS_APPLICATION_AUDIO).expect("stereo ms encoder");
        let mut dec = OpusMSDecoder::new(48000, 2, 1, 1, &[0, 1]).expect("stereo ms decoder");
        let pcm = patterned_pcm_i16(960, 2, 42);
        let mut packet = vec![0u8; 4000];
        let len = enc.encode(&pcm, 960, &mut packet, 4000).expect("encode");
        assert!(len > 0);
        let mut out = vec![0i16; 960 * 2];
        let decoded = dec.decode(Some(&packet[..len as usize]), len, &mut out, 960, false).expect("decode");
        assert_eq!(decoded, 960);
        assert!(out.iter().any(|&s| s != 0));
    }

    #[test]
    fn test_ms_encoder_zero_channels_error() {
        let result = OpusMSEncoder::new(48000, 0, 0, 0, &[], OPUS_APPLICATION_AUDIO);
        assert!(result.is_err());
    }

    #[test]
    fn test_ms_decoder_coupled_exceeds_streams_v2() {
        let result = OpusMSDecoder::new(48000, 2, 1, 2, &[0, 1]);
        assert!(result.is_err(), "coupled=2 > streams=1 should fail");
    }

    #[test]
    fn test_ms_surround_invalid_family_v2() {
        let result = OpusMSEncoder::new_surround(48000, 2, 3, OPUS_APPLICATION_AUDIO);
        assert!(result.is_err(), "family=3 should be UNIMPLEMENTED");
    }

    #[test]
    fn test_ms_decoder_plc_stereo() {
        let mut enc = OpusMSEncoder::new(48000, 2, 1, 1, &[0, 1], OPUS_APPLICATION_AUDIO).expect("stereo ms encoder");
        let mut dec = OpusMSDecoder::new(48000, 2, 1, 1, &[0, 1]).expect("stereo ms decoder");
        let pcm = patterned_pcm_i16(960, 2, 99);
        let mut packet = vec![0u8; 4000];
        let len = enc.encode(&pcm, 960, &mut packet, 4000).expect("encode");
        let mut out = vec![0i16; 960 * 2];
        dec.decode(Some(&packet[..len as usize]), len, &mut out, 960, false).expect("decode");
        let mut plc_out = vec![0i16; 960 * 2];
        let result = dec.decode(None, 0, &mut plc_out, 960, false);
        assert!(result.is_ok(), "PLC decode should succeed");
    }

    #[test]
    fn test_validate_layout_max_channels_boundary() {
        let result = OpusMSDecoder::new(48000, 2, 255, 1, &[0, 1]);
        assert!(result.is_err(), "streams=255 + coupled=1 = 256 > 255 should fail");
    }
}
