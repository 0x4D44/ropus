//! Subcommand implementations: encode, decode, info, play.

use std::fs::File;
use std::io::{BufReader, BufWriter};

use anyhow::{Context, Result, anyhow};
use colored::*;

use ropus::{
    Application, Bitrate, DecodeMode, Decoder as RopusDecoder, Encoder, Signal,
};

use ogg::reading::PacketReader;
use ogg::writing::{PacketWriteEndInfo, PacketWriter};

use crate::audio::decode::{DecodedAudio, decode_to_f32};
use crate::audio::resample::resample_to_48k;
use crate::audio::wav::write_wav_pcm16;
use crate::cli::{DecodeArgs, EncodeArgs, InfoArgs, PlayArgs};
use crate::consts::{
    FRAME_SAMPLES_PER_CH, FRAMES_PER_PACKET, MAX_OPUS_FRAME_BYTES, MAX_PACKET_BYTES, OPUS_SR,
};
use crate::container::ogg::{
    OGG_STREAM_SERIAL, build_opus_head, build_opus_tags, parse_opus_head, read_last_granule,
    read_opus_tags,
};
use crate::ui::{format_num, heading, ok};
use crate::util::{channel_count_to_ropus, with_extension};

// ---------------------------------------------------------------------------
// encode / transcode
// ---------------------------------------------------------------------------

pub(crate) fn encode(args: EncodeArgs) -> Result<()> {
    // Guard the encoder's output buffer sizing. If a future `--frame-duration`
    // flag wires in (40/60/80/100/120 ms = 2/3/4/5/6 frames per packet) and
    // FRAMES_PER_PACKET is bumped without updating MAX_PACKET_BYTES (or vice
    // versa), this fires immediately in debug builds rather than overflowing
    // the buffer at runtime.
    debug_assert_eq!(
        MAX_PACKET_BYTES,
        MAX_OPUS_FRAME_BYTES * FRAMES_PER_PACKET,
        "MAX_PACKET_BYTES must equal MAX_OPUS_FRAME_BYTES * FRAMES_PER_PACKET; \
         update both if --frame-duration becomes configurable"
    );

    heading("encode");
    println!("input    {}", args.input.display().to_string().cyan());

    let output = args
        .output
        .clone()
        .unwrap_or_else(|| with_extension(&args.input, "opus"));
    println!("output   {}", output.display().to_string().cyan());

    // 1. Decode the input to interleaved f32 PCM.
    let DecodedAudio {
        samples,
        sample_rate,
        channels,
    } = decode_to_f32(&args.input).context("decoding input")?;
    println!(
        "decoded  {} samples, {} Hz, {} ch",
        format_num(samples.len() as u64).bright_white(),
        sample_rate.to_string().bright_white(),
        channels.to_string().bright_white(),
    );

    // 2. Resample to 48 kHz if needed.
    let pcm_48k = if sample_rate == OPUS_SR {
        samples
    } else {
        println!("resample {} Hz -> {} Hz", sample_rate, OPUS_SR);
        resample_to_48k(&samples, sample_rate, channels)
            .context("resampling to 48 kHz")?
    };
    println!(
        "resampled {} samples @ 48 kHz",
        format_num(pcm_48k.len() as u64).bright_white(),
    );

    // 3. Build the encoder.
    let opus_channels = channel_count_to_ropus(channels)?;
    let mut builder =
        Encoder::builder(OPUS_SR, opus_channels, Application::from(args.application));
    if let Some(b) = args.bitrate {
        builder = builder.bitrate(Bitrate::Bits(b));
    }
    if let Some(c) = args.complexity {
        builder = builder.complexity(c);
    }
    builder = builder.signal(Signal::Auto);
    if args.cbr {
        builder = builder.vbr(false);
    } else {
        builder = builder.vbr(true);
    }
    let mut encoder = builder
        .build()
        .map_err(|e| anyhow!("encoder build failed: {e}"))?;

    // Query the encoder for its actual lookahead in 48 kHz samples; that is
    // exactly the value RFC 7845 requires in OpusHead.pre_skip (typically
    // 312; 120 in OPUS_APPLICATION_RESTRICTED_LOWDELAY). Real libopus values
    // are always well under 65 535; a value that doesn't fit in u16 means the
    // encoder is in a broken state, so we bail loudly rather than silently
    // capping at u16::MAX and producing an OpusHead with a wrong pre_skip.
    let lookahead = encoder.lookahead();
    let pre_skip = u16::try_from(lookahead).map_err(|_| {
        anyhow!(
            "encoder lookahead {} does not fit in u16 — likely corrupt encoder state",
            lookahead
        )
    })?;

    // 4. Open Ogg writer.
    let file = File::create(&output)
        .with_context(|| format!("creating output file {}", output.display()))?;
    let mut writer = PacketWriter::new(BufWriter::new(file));

    // 5. Emit OpusHead and OpusTags headers (each on its own page per RFC 7845).
    let head = build_opus_head(channels as u8, sample_rate, pre_skip);
    writer
        .write_packet(head, OGG_STREAM_SERIAL, PacketWriteEndInfo::EndPage, 0)
        .context("writing OpusHead page")?;

    let tags = build_opus_tags("ropus-cli");
    writer
        .write_packet(tags, OGG_STREAM_SERIAL, PacketWriteEndInfo::EndPage, 0)
        .context("writing OpusTags page")?;

    // 6. Encode and write data packets in 20 ms frames.
    let frame_interleaved = FRAME_SAMPLES_PER_CH * channels;
    let mut packet_buf = vec![0u8; MAX_PACKET_BYTES];
    let mut samples_written: u64 = 0;
    let mut packet_count: u64 = 0;
    let total_chunks = pcm_48k.len() / frame_interleaved;
    let remainder_len = pcm_48k.len() - total_chunks * frame_interleaved;
    let has_tail = remainder_len > 0;
    let chunks = pcm_48k.chunks_exact(frame_interleaved);
    let remainder_owned: Vec<f32> = chunks.remainder().to_vec();
    for (idx, chunk) in chunks.enumerate() {
        let n = encoder
            .encode_float(chunk, &mut packet_buf)
            .map_err(|e| anyhow!("encode failed: {e}"))?;
        // Advance the granule position by the per-channel frame length;
        // RFC 7845 sec. 4 mandates granule position is in 48 kHz samples.
        samples_written += FRAME_SAMPLES_PER_CH as u64;
        let is_last = idx + 1 == total_chunks && !has_tail;
        let end_info = if is_last {
            PacketWriteEndInfo::EndStream
        } else {
            PacketWriteEndInfo::NormalPacket
        };
        writer
            .write_packet(
                packet_buf[..n].to_vec(),
                OGG_STREAM_SERIAL,
                end_info,
                samples_written,
            )
            .context("writing Opus data page")?;
        packet_count += 1;
    }

    // Pad and encode any tail (silence-pad to a 20 ms frame).
    if has_tail {
        let mut frame_buf = vec![0.0f32; frame_interleaved];
        frame_buf[..remainder_owned.len()].copy_from_slice(&remainder_owned);
        let n = encoder
            .encode_float(&frame_buf, &mut packet_buf)
            .map_err(|e| anyhow!("encode failed (tail): {e}"))?;
        samples_written += FRAME_SAMPLES_PER_CH as u64;
        writer
            .write_packet(
                packet_buf[..n].to_vec(),
                OGG_STREAM_SERIAL,
                PacketWriteEndInfo::EndStream,
                samples_written,
            )
            .context("writing tail Opus packet")?;
        packet_count += 1;
    }

    println!(
        "wrote    {} packets, {} samples (granule)",
        format_num(packet_count).bright_white(),
        format_num(samples_written).bright_white(),
    );
    ok(&format!("encoded -> {}", output.display()));
    Ok(())
}

// ---------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------

pub(crate) fn decode(args: DecodeArgs) -> Result<()> {
    heading("decode");
    println!("input    {}", args.input.display().to_string().cyan());

    let output = args
        .output
        .clone()
        .unwrap_or_else(|| with_extension(&args.input, "wav"));
    println!("output   {}", output.display().to_string().cyan());

    let file = File::open(&args.input)
        .with_context(|| format!("opening {}", args.input.display()))?;
    let mut reader = PacketReader::new(BufReader::new(file));

    // Header packet: OpusHead.
    let head_pkt = reader
        .read_packet()?
        .ok_or_else(|| anyhow!("no packets found in input"))?;
    let head = parse_opus_head(&head_pkt.data)?;
    println!(
        "header   ch={} input_sr={} pre_skip={}",
        head.channels.to_string().bright_white(),
        head.input_sample_rate.to_string().bright_white(),
        head.pre_skip.to_string().bright_white(),
    );

    // Tags packet: OpusTags. Verify magic to catch malformed files (e.g. a
    // stripped tags page would otherwise silently consume the first audio
    // packet).
    read_opus_tags(&mut reader).context("reading OpusTags packet")?;

    let opus_channels = channel_count_to_ropus(head.channels as usize)?;
    let mut decoder = RopusDecoder::new(OPUS_SR, opus_channels)
        .map_err(|e| anyhow!("decoder init failed: {e}"))?;

    // Maximum 120 ms of decoded samples at 48 kHz, per channel.
    let max_per_ch = (OPUS_SR / 1000 * 120) as usize;
    let mut decoded = vec![0i16; max_per_ch * opus_channels.count()];

    // We accumulate decoded samples then write a single WAV when finished.
    let mut all_pcm: Vec<i16> = Vec::with_capacity(1 << 20);
    let mut packet_count: u64 = 0;
    while let Some(pkt) = reader.read_packet()? {
        let n = decoder
            .decode(&pkt.data, &mut decoded, DecodeMode::Normal)
            .map_err(|e| anyhow!("decode failed: {e}"))?;
        // n is samples per channel.
        let total = n * opus_channels.count();
        all_pcm.extend_from_slice(&decoded[..total]);
        packet_count += 1;
    }

    // Trim the leading pre-skip samples.
    let pre_skip = head.pre_skip as usize * opus_channels.count();
    let pre_skip = pre_skip.min(all_pcm.len());
    let trimmed = &all_pcm[pre_skip..];

    write_wav_pcm16(&output, trimmed, OPUS_SR, opus_channels.count() as u16)
        .context("writing WAV")?;

    println!(
        "decoded  {} packets, {} samples ({} after pre-skip)",
        format_num(packet_count).bright_white(),
        format_num(all_pcm.len() as u64).bright_white(),
        format_num(trimmed.len() as u64).bright_white(),
    );
    ok(&format!("decoded -> {}", output.display()));
    Ok(())
}

// ---------------------------------------------------------------------------
// info
// ---------------------------------------------------------------------------

pub(crate) fn info(args: InfoArgs) -> Result<()> {
    heading("info");
    println!("file     {}", args.input.display().to_string().cyan());

    let file = File::open(&args.input)
        .with_context(|| format!("opening {}", args.input.display()))?;
    let file_len = file
        .metadata()
        .ok()
        .map(|m| m.len())
        .unwrap_or(0);
    let mut reader = PacketReader::new(BufReader::new(file));

    let head_pkt = reader
        .read_packet()?
        .ok_or_else(|| anyhow!("empty file"))?;
    let head = parse_opus_head(&head_pkt.data)?;
    // Capture the OpusHead's stream serial — this identifies the logical Opus
    // bitstream we care about in a multiplexed Ogg file. We need it to filter
    // the reverse-scan in `read_last_granule` to the right stream.
    let target_serial = head_pkt.stream_serial();
    read_opus_tags(&mut reader).context("reading OpusTags packet")?;

    let opus_channels = channel_count_to_ropus(head.channels as usize)?;

    // Fast path: locate the last Ogg page belonging to our serial and read
    // its absolute granule position. Per RFC 7845 §4 that is the total stream
    // length in 48 kHz samples, including pre_skip warm-up.
    let mut fast_file = File::open(&args.input)
        .with_context(|| format!("opening {} for granule scan", args.input.display()))?;
    let absgp_opt =
        read_last_granule(&mut fast_file, target_serial).context("scanning for last Ogg page")?;

    let sample_count = if let Some(absgp) = absgp_opt {
        // absgp is the cumulative per-channel sample count at end-of-stream
        // (48 kHz). Subtract pre_skip to get the real decoded duration.
        absgp.saturating_sub(head.pre_skip as u64)
    } else {
        // Truncated or unknown end (absgp == 0xFFFF_FFFF_FFFF_FFFF). Fall back
        // to the slow path: decode every packet to recover the sample count.
        let mut decoder = RopusDecoder::new(OPUS_SR, opus_channels)
            .map_err(|e| anyhow!("decoder init failed: {e}"))?;
        let max_per_ch = (OPUS_SR / 1000 * 120) as usize;
        let mut decoded = vec![0i16; max_per_ch * opus_channels.count()];

        let mut packet_idx: u64 = 0;
        let mut sample_count: u64 = 0;
        while let Some(pkt) = reader.read_packet()? {
            // We need to know how many samples the packet decodes to. The
            // simplest correct approach is to actually decode it and count.
            match decoder.decode(&pkt.data, &mut decoded, DecodeMode::Normal) {
                Ok(n) => sample_count += n as u64,
                Err(e) => {
                    eprintln!("{} packet {}: {e}", "warning:".yellow(), packet_idx);
                }
            }
            packet_idx += 1;
        }
        // Subtract pre_skip from the total so the reported duration matches the
        // fast path (which subtracts pre_skip from absgp).
        sample_count.saturating_sub(head.pre_skip as u64)
    };

    let duration_s = sample_count as f64 / OPUS_SR as f64;
    let avg_kbps = if duration_s > 0.0 {
        (file_len as f64 * 8.0) / duration_s / 1000.0
    } else {
        0.0
    };

    println!("container       {}", "Ogg".bright_white());
    println!("opus_version    {}", head.version.to_string().bright_white());
    println!("channels        {}", head.channels.to_string().bright_white());
    println!(
        "input_sr        {} Hz",
        head.input_sample_rate.to_string().bright_white()
    );
    println!(
        "decode_sr       {} Hz",
        OPUS_SR.to_string().bright_white()
    );
    println!("pre_skip        {}", head.pre_skip.to_string().bright_white());
    println!(
        "channel_mapping {}",
        head.channel_mapping.to_string().bright_white()
    );
    println!(
        "total_samples   {} ({} per ch)",
        format_num(sample_count * opus_channels.count() as u64).bright_white(),
        format_num(sample_count).bright_white(),
    );
    println!(
        "duration        {} s",
        format!("{:.3}", duration_s).bright_white()
    );
    println!("file_size       {} bytes", format_num(file_len).bright_white());
    println!(
        "avg_bitrate     {} kbps",
        format!("{:.1}", avg_kbps).bright_white()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// play
// ---------------------------------------------------------------------------

pub(crate) fn play(args: PlayArgs) -> Result<()> {
    heading("play");
    println!("file     {}", args.input.display().to_string().cyan());

    // Decode to interleaved f32 through one unified pipeline: symphonia demuxes
    // every container, and `decode_to_f32` routes Opus tracks through ropus
    // while everything else uses symphonia's native decoders.
    let DecodedAudio {
        samples,
        sample_rate,
        channels,
    } = decode_to_f32(&args.input)?;

    let channels_u16 = u16::try_from(channels).map_err(|_| anyhow!("channel count overflow"))?;
    println!(
        "audio    {} samples, {} Hz, {} ch",
        format_num(samples.len() as u64).bright_white(),
        sample_rate.to_string().bright_white(),
        channels_u16.to_string().bright_white(),
    );

    // Try to open the default audio device. If that fails (no device, headless
    // environment, etc.) print a clear message instead of panicking.
    let (_stream, handle) = match rodio::OutputStream::try_default() {
        Ok(pair) => pair,
        Err(e) => {
            return Err(anyhow!("no default audio output device available: {e}"));
        }
    };
    let sink = rodio::Sink::try_new(&handle).map_err(|e| anyhow!("creating sink failed: {e}"))?;

    if let Some(v) = args.volume {
        sink.set_volume(v.clamp(0.0, 1.0));
    }

    let source = rodio::buffer::SamplesBuffer::new(channels_u16, sample_rate, samples);
    sink.append(source);
    println!("playing  (Ctrl-C to stop)");
    sink.sleep_until_end();
    ok("playback finished");
    Ok(())
}
