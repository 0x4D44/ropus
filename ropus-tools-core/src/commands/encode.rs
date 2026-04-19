//! Encode: any symphonia-supported input → Ogg Opus.

use std::fs::File;
use std::io::BufWriter;

use anyhow::{Context, Result, anyhow};
use colored::*;

use ropus::{Bitrate, Encoder, Signal};

use ogg::writing::{PacketWriteEndInfo, PacketWriter};

use crate::audio::decode::{DecodedAudio, decode_to_f32};
use crate::audio::resample::resample_to_48k;
use crate::consts::{
    FRAME_SAMPLES_PER_CH, FRAMES_PER_PACKET, MAX_OPUS_FRAME_BYTES, MAX_PACKET_BYTES, OPUS_SR,
};
use crate::container::ogg::{
    OGG_STREAM_SERIAL, OpusTags, build_opus_head,
};
use crate::options::EncodeOptions;
use crate::ui::{format_num, heading, ok};
use crate::util::{channel_count_to_ropus, with_extension};

pub fn encode(opts: EncodeOptions) -> Result<()> {
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
    println!("input    {}", opts.input.display().to_string().cyan());

    let output = opts
        .output
        .clone()
        .unwrap_or_else(|| with_extension(&opts.input, "opus"));
    println!("output   {}", output.display().to_string().cyan());

    // 1. Decode the input to interleaved f32 PCM.
    let DecodedAudio {
        samples,
        sample_rate,
        channels,
    } = decode_to_f32(&opts.input).context("decoding input")?;
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
    let mut builder = Encoder::builder(OPUS_SR, opus_channels, opts.application);
    if let Some(b) = opts.bitrate {
        builder = builder.bitrate(Bitrate::Bits(b));
    }
    if let Some(c) = opts.complexity {
        builder = builder.complexity(c);
    }
    builder = builder.signal(Signal::Auto);
    builder = builder.vbr(opts.vbr);
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

    let tags = OpusTags {
        vendor: opts.vendor.clone(),
        comments: opts.comments.clone(),
    }
    .encode();
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
