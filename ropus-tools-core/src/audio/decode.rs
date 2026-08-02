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
use crate::container::ogg::{parse_opus_head, validate_opus_audio_packet};
use crate::ui::escape_terminal_path;
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
    channels: usize,
    pre_skip: usize,
    end_granule: usize,
}

enum CodecPipeline {
    Native(Box<dyn SymphoniaDecoder>),
    Opus(Box<OpusState>),
}

pub(crate) const MIN_GAIN_DB: f32 = -128.0;
pub(crate) const MAX_GAIN_DB: f32 = 32_767.0 / 256.0;

pub fn decode_to_f32(path: &Path) -> Result<DecodedAudio> {
    decode_to_f32_with_gain(path, 0.0)
}

/// Decode a file to interleaved f32 PCM, applying `gain_db` at the codec
/// boundary. Opus combines the user gain with `OpusHead.output_gain` and calls
/// `Decoder::set_gain`; other codecs use the same linear dB multiplier after
/// native decoding.
pub fn decode_to_f32_with_gain(path: &Path, gain_db: f32) -> Result<DecodedAudio> {
    let file =
        File::open(path).with_context(|| format!("opening {}", escape_terminal_path(path)))?;
    let hint_ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_string);
    decode_reader_with_gain(Box::new(file), hint_ext.as_deref(), gain_db)
        .with_context(|| format!("decoding {}", escape_terminal_path(path)))
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
    decode_reader_with_gain(source, hint_ext, 0.0)
}

/// Decode an arbitrary symphonia `MediaSource` with a user-requested dB gain.
/// Opus applies the gain in the decoder so its i16 saturation semantics match
/// direct decode; non-Opus codecs retain the f32 post-decode behaviour used by
/// the player before this gain-aware entry point existed.
pub fn decode_reader_with_gain(
    source: Box<dyn MediaSource>,
    hint_ext: Option<&str>,
    gain_db: f32,
) -> Result<DecodedAudio> {
    validate_gain_db(gain_db)?;
    let mss = MediaSourceStream::new(source, Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = hint_ext {
        hint.with_extension(ext);
    }

    let format_options = FormatOptions {
        enable_gapless: true,
        ..FormatOptions::default()
    };
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_options, &MetadataOptions::default())
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
        let opus_head = codec_params
            .extra_data
            .as_deref()
            .map(parse_opus_head)
            .transpose()
            .context("parsing OpusHead")?;
        let opus_channels = channel_count_to_ropus(channels)?;
        let mut dec = RopusDecoder::new(OPUS_SR, opus_channels)
            .map_err(|e| anyhow!("decoder init failed: {e}"))?;
        let header_gain_q8 = opus_head.map_or(0, |head| i32::from(head.output_gain));
        let user_gain_q8 = gain_db_to_q8(gain_db)?;
        let total_gain_q8 = header_gain_q8
            .checked_add(user_gain_q8)
            .ok_or_else(|| anyhow!("Opus header and user gain overflow"))?;
        if !(-32_768..=32_767).contains(&total_gain_q8) {
            bail!(
                "Opus header and user gain total {total_gain_q8} Q8 is outside decoder range [-32768, 32767]"
            );
        }
        if total_gain_q8 != 0 {
            dec.set_gain(total_gain_q8)
                .map_err(|e| anyhow!("set_gain({total_gain_q8} Q8) failed: {e}"))?;
        }
        // Keep the OpusHead delay separate from Symphonia's inferred codec
        // delay. The latter includes page-level padding for some streams,
        // while RFC 7845's audible start is always the OpusHead pre-skip.
        let pre_skip = opus_head
            .map(|head| head.pre_skip as usize)
            .or_else(|| codec_params.delay.map(|d| d as usize))
            .ok_or_else(|| {
                anyhow!(
                    "opus track has no pre_skip metadata (no OpusHead extra_data or codec delay)"
                )
            })?;
        let end_granule = codec_params
            .n_frames
            .ok_or_else(|| anyhow!("opus track ended before an Ogg EOS granule was established"))?;
        let end_granule = usize::try_from(end_granule)
            .map_err(|_| anyhow!("Opus EOS granule does not fit in this platform's usize"))?;
        if end_granule < pre_skip {
            bail!("Opus EOS granule {end_granule} is smaller than pre-skip {pre_skip}");
        }
        CodecPipeline::Opus(Box::new(OpusState {
            dec,
            channels,
            pre_skip,
            end_granule,
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
    let opus_gain_applied = matches!(&pipeline, CodecPipeline::Opus(_));
    let opus_timeline = match &pipeline {
        CodecPipeline::Opus(state) => Some((state.pre_skip, state.end_granule, state.channels)),
        CodecPipeline::Native(_) => None,
    };

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
                    dec, channels: ch, ..
                } = state.as_mut();
                validate_opus_audio_packet(&packet.data).context("validating Opus audio packet")?;
                if opus_scratch.len() != max_per_ch * *ch {
                    opus_scratch = vec![0f32; max_per_ch * *ch];
                }
                let n = match dec.decode_float(&packet.data, &mut opus_scratch, DecodeMode::Normal)
                {
                    Ok(n) => n,
                    Err(e) => {
                        // Match the native path: swallow per-packet decode
                        // failures rather than aborting the whole file.
                        eprintln!(
                            "{} opus packet: {}",
                            "warning:".yellow(),
                            crate::ui::escape_terminal_text(&e.to_string())
                        );
                        continue;
                    }
                };
                let total = n * *ch;
                let frame = &opus_scratch[..total];
                // Symphonia exposes the packet timeline and EOS granule, but
                // its Ogg probe queues packets before the final page's trim
                // metadata is known. Apply the RFC 7845 endpoint explicitly
                // after decoding so every caller gets the same exact range.
                interleaved.extend_from_slice(frame);
            }
        }
    }

    if let Some((pre_skip, end_granule, channels)) = opus_timeline {
        let start_samples = pre_skip
            .checked_mul(channels)
            .ok_or_else(|| anyhow!("Opus pre-skip sample count overflows"))?;
        let end_samples = end_granule
            .checked_mul(channels)
            .ok_or_else(|| anyhow!("Opus EOS sample count overflows"))?;
        if interleaved.len() < end_samples {
            bail!(
                "decoded {} samples, but Opus EOS granule requires {} samples",
                interleaved.len(),
                end_samples
            );
        }
        interleaved = interleaved[start_samples..end_samples].to_vec();
    }

    if !opus_gain_applied && gain_db != 0.0 {
        let multiplier = gain_db_to_multiplier(gain_db);
        for sample in &mut interleaved {
            *sample *= multiplier;
        }
    }

    // A corrupt stream may have every packet rejected above, or a valid
    // stream's pre-skip may consume its only decoded frame. Never hand an
    // empty track to playback: an empty sink looks like successful playback
    // and can make loop modes spin forever.
    if interleaved.is_empty() {
        bail!("decoded input contains no audio samples");
    }

    Ok(DecodedAudio {
        samples: interleaved,
        sample_rate,
        channels,
    })
}

pub(crate) fn validate_gain_db(gain_db: f32) -> Result<()> {
    if !gain_db.is_finite() {
        bail!("gain must be a finite dB value (got {gain_db})");
    }
    if !(MIN_GAIN_DB..=MAX_GAIN_DB).contains(&gain_db) {
        bail!("gain {gain_db} dB out of range [{MIN_GAIN_DB}, {MAX_GAIN_DB}]");
    }
    Ok(())
}

pub(crate) fn gain_db_to_q8(gain_db: f32) -> Result<i32> {
    validate_gain_db(gain_db)?;
    let q8 = (f64::from(gain_db) * 256.0).round();
    if !(i32::MIN as f64..=i32::MAX as f64).contains(&q8) {
        bail!("gain {gain_db} dB does not fit in Q8");
    }
    Ok(q8 as i32)
}

fn gain_db_to_multiplier(gain_db: f32) -> f32 {
    10.0_f32.powf(gain_db / 20.0)
}
