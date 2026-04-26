//! Decode any symphonia-supported input file to interleaved f32 PCM.
//!
//! Opus tracks are routed through the `ropus` decoder; everything else uses
//! symphonia's native decoder for the codec.

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use colored::*;

use ropus::{DecodeMode, Decoder as RopusDecoder};

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{
    CODEC_TYPE_NULL, CODEC_TYPE_OPUS, Decoder as SymphoniaDecoder, DecoderOptions,
};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSource, MediaSourceStream};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::consts::OPUS_SR;
use crate::util::channel_count_to_ropus;

pub struct DecodedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: usize,
}

/// Internal codec pipeline: either symphonia's native decoder for the track,
/// or ropus driven by symphonia's Ogg demuxer for Opus tracks. Centralising
/// the routing here means `commands::play`, `commands::encode` and any future
/// caller goes through the same demuxer/decoder for every supported input
/// format.
///
/// The Opus variant is boxed so the two variants are similar in size — the
/// inline `RopusDecoder` is much larger than `Box<dyn SymphoniaDecoder>` and
/// would otherwise trip `clippy::large_enum_variant`.
struct OpusState {
    dec: RopusDecoder,
    pre_skip: usize,
    channels: usize,
    /// Samples already trimmed off the head. `commands::play` does not seek,
    /// so this only ever monotonically increases until pre_skip is consumed.
    skipped: usize,
}

enum CodecPipeline {
    Native(Box<dyn SymphoniaDecoder>),
    Opus(Box<OpusState>),
}

pub fn decode_to_f32(path: &Path) -> Result<DecodedAudio> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let hint_ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_string);
    decode_reader(Box::new(file), hint_ext.as_deref())
        .with_context(|| format!("decoding {}", path.display()))
}

/// Decode an arbitrary symphonia `MediaSource`. `hint_ext` is an optional file
/// extension (no leading dot) used by symphonia's probe to narrow down the
/// container format. Callers feeding `Cursor<Vec<u8>>` from stdin can pass
/// `None` — probe falls back to magic-byte sniffing.
///
/// Note: symphonia's probe chain can require backward seeks for format
/// sniffing, which is why `MediaSource` requires `Seek`. Wrapping stdin in a
/// `Cursor<Vec<u8>>` (buffered up-front) satisfies that requirement, at the
/// cost of buffering the whole input in memory.
pub fn decode_reader(source: Box<dyn MediaSource>, hint_ext: Option<&str>) -> Result<DecodedAudio> {
    let mss = MediaSourceStream::new(source, Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = hint_ext {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("probing input")?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("no decodable audio track"))?;

    let track_id = track.id;
    let codec_params = track.codec_params.clone();

    let sample_rate = codec_params
        .sample_rate
        .ok_or_else(|| anyhow!("track has no sample rate"))?;
    let channels = codec_params
        .channels
        .map(|c| c.count())
        .ok_or_else(|| anyhow!("track has no channel info"))?;

    // Branch on codec: Opus goes through ropus (we deliberately don't enable
    // symphonia's stub Opus decoder). Everything else uses the native
    // symphonia decoder for that codec.
    let mut pipeline = if codec_params.codec == CODEC_TYPE_OPUS {
        let opus_channels = channel_count_to_ropus(channels)?;
        let dec = RopusDecoder::new(OPUS_SR, opus_channels)
            .map_err(|e| anyhow!("decoder init failed: {e}"))?;
        // Resolve OpusHead.pre_skip without falling back to a magic constant.
        // Symphonia's Ogg demuxer surfaces it as `codec_params.delay`; failing
        // that, we parse it directly out of the OpusHead bytes that the
        // demuxer hands us in `codec_params.extra_data`. If neither is
        // present the file is malformed (every valid Ogg Opus stream begins
        // with an OpusHead packet), so we bail rather than silently inserting
        // a guess that would shift playback.
        let pre_skip = if let Some(d) = codec_params.delay {
            d as usize
        } else if let Some(buf) = codec_params.extra_data.as_deref() {
            if buf.len() >= 19 && &buf[..8] == b"OpusHead" {
                u16::from_le_bytes([buf[10], buf[11]]) as usize
            } else {
                bail!(
                    "opus track has no pre_skip metadata (no codec delay and \
                     no OpusHead extra_data)"
                );
            }
        } else {
            bail!(
                "opus track has no pre_skip metadata (no codec delay and \
                 no OpusHead extra_data)"
            );
        };
        CodecPipeline::Opus(Box::new(OpusState {
            dec,
            pre_skip,
            channels,
            skipped: 0,
        }))
    } else {
        let dec = symphonia::default::get_codecs()
            .make(&codec_params, &DecoderOptions::default())
            .context("creating decoder for track")?;
        CodecPipeline::Native(dec)
    };

    let mut interleaved: Vec<f32> = Vec::with_capacity(1 << 20);
    let mut sample_buf: Option<SampleBuffer<f32>> = None;
    let max_per_ch = (OPUS_SR / 1000 * 120) as usize;
    let mut opus_scratch: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                bail!("symphonia: stream reset required (unsupported)")
            }
            Err(e) => return Err(e).context("reading next packet"),
        };
        if packet.track_id() != track_id {
            continue;
        }

        match &mut pipeline {
            CodecPipeline::Native(decoder) => match decoder.decode(&packet) {
                Ok(decoded) => {
                    if sample_buf.is_none() {
                        let spec = *decoded.spec();
                        let dur = decoded.capacity() as u64;
                        sample_buf = Some(SampleBuffer::<f32>::new(dur, spec));
                    }
                    if let Some(buf) = sample_buf.as_mut() {
                        buf.copy_interleaved_ref(decoded);
                        interleaved.extend_from_slice(buf.samples());
                    }
                }
                Err(SymphoniaError::DecodeError(_)) => {
                    // Skip corrupt packets.
                    continue;
                }
                Err(e) => return Err(e).context("decoding packet"),
            },
            CodecPipeline::Opus(state) => {
                let OpusState {
                    dec,
                    pre_skip,
                    channels: ch,
                    skipped,
                } = state.as_mut();
                if opus_scratch.len() != max_per_ch * *ch {
                    opus_scratch = vec![0f32; max_per_ch * *ch];
                }
                let n = match dec.decode_float(&packet.data, &mut opus_scratch, DecodeMode::Normal)
                {
                    Ok(n) => n,
                    Err(e) => {
                        // Match the native path: swallow per-packet decode
                        // failures rather than aborting the whole file.
                        eprintln!("{} opus packet: {e}", "warning:".yellow());
                        continue;
                    }
                };
                let total = n * *ch;
                let frame = &opus_scratch[..total];
                // Trim the leading pre_skip samples (in interleaved units)
                // before they reach the output buffer.
                let want_trim = pre_skip.saturating_mul(*ch).saturating_sub(*skipped);
                if want_trim >= total {
                    *skipped += total;
                } else {
                    interleaved.extend_from_slice(&frame[want_trim..]);
                    *skipped += want_trim;
                }
            }
        }
    }

    Ok(DecodedAudio {
        samples: interleaved,
        sample_rate,
        channels,
    })
}
