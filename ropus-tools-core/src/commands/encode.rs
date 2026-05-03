//! Encode: any symphonia-supported input → Ogg Opus.

use std::fs::File;
use std::io::{BufWriter, Cursor, Read, Write};

use anyhow::{Context, Result, anyhow, bail};
use colored::*;

use ropus::{Bitrate, Encoder};

use ogg::writing::{PacketWriteEndInfo, PacketWriter};

use crate::audio::decode::{DecodedAudio, decode_reader, decode_to_f32};
use crate::audio::downmix::downmix_to_mono;
use crate::audio::resample::resample;
use crate::consts::{MAX_OPUS_FRAME_BYTES, MAX_PACKET_BYTES, MAX_SUBFRAMES_PER_PACKET, OPUS_SR};
use crate::container::ogg::{OGG_STREAM_SERIAL, OpusTags, build_opus_head};
use crate::container::picture::{
    MAX_PICTURE_BYTES, base64_encode, build_picture_block, detect_format,
};
use crate::options::EncodeOptions;
use crate::ui::{format_num, heading, ok};
use crate::util::{channel_count_to_ropus, is_stdio_sentinel, with_extension};

use ropus::FrameDuration;

/// Per-channel sample count for each supported `FrameDuration` at 48 kHz.
/// `2.5 ms * 48 = 120` samples (exact). Kept as a function rather than a table
/// so adding a new FrameDuration variant upstream fails loudly at compile
/// time via an unmatched arm.
fn frame_samples_per_ch(d: FrameDuration) -> usize {
    match d {
        FrameDuration::Ms2_5 => 120,
        FrameDuration::Ms5 => 240,
        FrameDuration::Ms10 => 480,
        FrameDuration::Ms20 => 960,
        FrameDuration::Ms40 => 1920,
        FrameDuration::Ms60 => 2880,
        FrameDuration::Ms80 => 3840,
        FrameDuration::Ms100 => 4800,
        FrameDuration::Ms120 => 5760,
        // `Argument` means "infer from buffer size", which would leave our
        // chunking math guessing. The CLI surfaces explicit ms values only
        // (FrameSizeArg → FrameDuration skips this variant), so reaching
        // this arm means a library caller constructed an invalid
        // EncodeOptions.
        FrameDuration::Argument => {
            unreachable!("CLI never emits FrameDuration::Argument — only explicit durations")
        }
    }
}

pub fn encode(opts: EncodeOptions) -> Result<()> {
    // Guard the encoder's output buffer sizing. At `--framesize` ≥ 40 ms,
    // libopus packs 2..6 sub-frames into a code-3 packet and uses the full
    // output buffer as its repacketise budget, so sizing the buffer for just
    // one sub-frame silently caps multi-frame packets and collapses quality
    // on high-bitrate CBR. We size for the worst case (6 × 1275 bytes) across
    // every frame duration.
    debug_assert_eq!(
        MAX_PACKET_BYTES,
        MAX_OPUS_FRAME_BYTES * MAX_SUBFRAMES_PER_PACKET,
        "MAX_PACKET_BYTES must equal MAX_OPUS_FRAME_BYTES * MAX_SUBFRAMES_PER_PACKET"
    );

    // Resolve the output path first. `-` and "input is stdin with no explicit
    // -o" both map to stdout (there's no sensible filename to derive from a
    // pipe). Detect stdout early so every banner/progress `println!` below can
    // route through `report!` and land on stderr instead — mixing progress
    // text with the Ogg bitstream on stdout corrupts downstream consumers.
    let input_is_stdin = is_stdio_sentinel(&opts.input);
    let output_path: std::path::PathBuf = match opts.output.clone() {
        Some(p) => p,
        None if input_is_stdin => std::path::PathBuf::from("-"),
        None => with_extension(&opts.input, "opus"),
    };
    let output_is_stdout = is_stdio_sentinel(&output_path);

    // Print progress/banner lines. Gated on output-sink: stdout gets the
    // bitstream, so progress must go to stderr in that case.
    macro_rules! report {
        ($($arg:tt)*) => {
            if output_is_stdout {
                eprintln!($($arg)*);
            } else {
                println!($($arg)*);
            }
        };
    }
    if output_is_stdout {
        eprintln!("{}", "encode".bright_yellow().bold());
    } else {
        heading("encode");
    }
    report!(
        "input    {}",
        if input_is_stdin {
            "<stdin>".cyan().to_string()
        } else {
            opts.input.display().to_string().cyan().to_string()
        }
    );
    report!(
        "output   {}",
        if output_is_stdout {
            "<stdout>".cyan().to_string()
        } else {
            output_path.display().to_string().cyan().to_string()
        }
    );

    // 1. Decode the input to interleaved f32 PCM. Stdin path buffers all bytes
    //    into a `Vec<u8>` (symphonia's probe chain needs `Seek` for format
    //    sniffing); a multi-GB pipe will use multi-GB RAM. Accepted per HLD.
    let DecodedAudio {
        samples,
        sample_rate,
        channels,
    } = if input_is_stdin {
        let mut buf = Vec::new();
        std::io::stdin()
            .lock()
            .read_to_end(&mut buf)
            .context("reading stdin into buffer")?;
        decode_reader(Box::new(Cursor::new(buf)), None).context("decoding stdin input")?
    } else {
        decode_to_f32(&opts.input).context("decoding input")?
    };
    report!(
        "decoded  {} samples, {} Hz, {} ch",
        format_num(samples.len() as u64).bright_white(),
        sample_rate.to_string().bright_white(),
        channels.to_string().bright_white(),
    );

    // 2. Optional stereo → mono downmix. Must happen before resample so the
    //    resampler sees the post-mix channel count; the encoder, OpusHead,
    //    and resampler all need to agree.
    let (samples, channels) = if opts.downmix_to_mono && channels > 1 {
        let mixed = downmix_to_mono(&samples, channels).context("downmixing stereo to mono")?;
        report!("downmix  {} ch -> 1 ch", channels);
        (mixed, 1usize)
    } else {
        (samples, channels)
    };

    // 3. Resample to 48 kHz if needed.
    let pcm_48k = if sample_rate == OPUS_SR {
        samples
    } else {
        report!("resample {} Hz -> {} Hz", sample_rate, OPUS_SR);
        resample(&samples, sample_rate, OPUS_SR, channels).context("resampling to 48 kHz")?
    };
    report!(
        "resampled {} samples @ 48 kHz",
        format_num(pcm_48k.len() as u64).bright_white(),
    );

    // 4. Build the encoder.
    let opus_channels = channel_count_to_ropus(channels)?;
    let mut builder = Encoder::builder(OPUS_SR, opus_channels, opts.application);
    if let Some(b) = opts.bitrate {
        builder = builder.bitrate(Bitrate::Bits(b));
    }
    if let Some(c) = opts.complexity {
        builder = builder.complexity(c);
    }
    builder = builder.signal(opts.signal);
    builder = builder.vbr(opts.vbr);
    builder = builder.vbr_constraint(opts.vbr_constraint);
    builder = builder.frame_duration(opts.frame_duration);
    if opts.expect_loss > 0 {
        builder = builder.packet_loss_perc(opts.expect_loss);
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

    // 5. Open Ogg writer. For `-` we route to locked stdout; for everything
    //    else we create the file. `PacketWriter` is generic over any
    //    `Write` so both sinks plug in identically.
    let sink: Box<dyn Write> = if output_is_stdout {
        Box::new(BufWriter::new(std::io::stdout().lock()))
    } else {
        let file = File::create(&output_path)
            .with_context(|| format!("creating output file {}", output_path.display()))?;
        Box::new(BufWriter::new(file))
    };
    let mut writer = PacketWriter::new(sink);

    // The caller's `--serial N` overrides the library's default constant.
    let serial = opts.serial.unwrap_or(OGG_STREAM_SERIAL);

    // 6. Emit OpusHead and OpusTags headers (each on its own page per RFC 7845).
    let head = build_opus_head(channels as u8, sample_rate, pre_skip);
    writer
        .write_packet(head, serial, PacketWriteEndInfo::EndPage, 0)
        .context("writing OpusHead page")?;

    // Optional --picture: read bytes, detect format, build METADATA_BLOCK_PICTURE,
    // base64-encode, and prepend to the user comments. "Prepend" is a deliberate
    // choice — opus-tools emits it before user-supplied comments, and keeping
    // the same order means differential testing against opus-tools stays clean.
    let mut comments = opts.comments.clone();
    if let Some(pic_path) = opts.picture_path.as_ref() {
        // Stat first and reject oversize files *before* reading them into
        // memory. Avoids a 5 GiB allocation on obvious user error (dropped-in
        // video file, etc.) and gives a clear message instead of OOM.
        let meta = std::fs::metadata(pic_path)
            .with_context(|| format!("reading picture metadata {}", pic_path.display()))?;
        if meta.len() > MAX_PICTURE_BYTES {
            bail!(
                "picture file {} is {} bytes; refusing > {} bytes (use a smaller cover image)",
                pic_path.display(),
                meta.len(),
                MAX_PICTURE_BYTES,
            );
        }
        let data = std::fs::read(pic_path)
            .with_context(|| format!("reading picture file {}", pic_path.display()))?;
        if data.is_empty() {
            bail!("picture file {} is empty", pic_path.display());
        }
        let format = detect_format(&data)
            .with_context(|| format!("detecting picture format for {}", pic_path.display()))?;
        let block = build_picture_block(format, &data)
            .with_context(|| format!("building picture block for {}", pic_path.display()))?;
        let b64 = base64_encode(&block);
        comments.insert(0, format!("METADATA_BLOCK_PICTURE={b64}"));
        report!(
            "picture  {} ({} bytes, {})",
            pic_path.display().to_string().cyan(),
            format_num(data.len() as u64).bright_white(),
            format.mime(),
        );
    }

    let tags = OpusTags {
        vendor: opts.vendor.clone(),
        comments,
    }
    .encode();
    writer
        .write_packet(tags, serial, PacketWriteEndInfo::EndPage, 0)
        .context("writing OpusTags page")?;

    // 7. Encode and write data packets in chunks of the chosen frame size.
    let frame_samples_ch = frame_samples_per_ch(opts.frame_duration);
    let frame_interleaved = frame_samples_ch * channels;
    let mut packet_buf = vec![0u8; MAX_PACKET_BYTES];
    let mut samples_written: u64 = 0;
    let mut packet_count: u64 = 0;
    let mut payload_bytes: u64 = 0;
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
        samples_written += frame_samples_ch as u64;
        payload_bytes += n as u64;
        let is_last = idx + 1 == total_chunks && !has_tail;
        let end_info = if is_last {
            PacketWriteEndInfo::EndStream
        } else {
            PacketWriteEndInfo::NormalPacket
        };
        writer
            .write_packet(packet_buf[..n].to_vec(), serial, end_info, samples_written)
            .context("writing Opus data page")?;
        packet_count += 1;
    }

    // Pad and encode any tail (silence-pad to the chosen frame size).
    if has_tail {
        let mut frame_buf = vec![0.0f32; frame_interleaved];
        frame_buf[..remainder_owned.len()].copy_from_slice(&remainder_owned);
        let n = encoder
            .encode_float(&frame_buf, &mut packet_buf)
            .map_err(|e| anyhow!("encode failed (tail): {e}"))?;
        samples_written += frame_samples_ch as u64;
        payload_bytes += n as u64;
        writer
            .write_packet(
                packet_buf[..n].to_vec(),
                serial,
                PacketWriteEndInfo::EndStream,
                samples_written,
            )
            .context("writing tail Opus packet")?;
        packet_count += 1;
    }

    // Drain the Ogg packet writer and the BufWriter underneath before
    // returning. `BufWriter::drop` swallows write errors, so relying on
    // drop lets broken pipes (`ropusenc … -o - | head -c 100`) pass as
    // clean exits with truncated output. Explicitly flushing surfaces the
    // error via `?` instead.
    let mut sink = writer.into_inner();
    sink.flush().context("flushing Ogg output")?;

    report!(
        "wrote    {} packets, {} samples (granule)",
        format_num(packet_count).bright_white(),
        format_num(samples_written).bright_white(),
    );
    // Average payload bitrate (Opus packets only — excludes Ogg framing
    // overhead, which is what `--bitrate` controls and what users compare
    // against the target). RFC 7845 fixes the granule clock at 48 kHz, so
    // duration_seconds = samples_written / OPUS_SR.
    if let Some(avg_bps) = (payload_bytes * 8 * (OPUS_SR as u64)).checked_div(samples_written) {
        report!(
            "bitrate  {} kbps avg (payload)",
            format!("{:.1}", avg_bps as f64 / 1000.0).bright_white(),
        );
    }
    let dest = if output_is_stdout {
        "<stdout>".to_string()
    } else {
        output_path.display().to_string()
    };
    if output_is_stdout {
        eprintln!("{}", format!("encoded -> {dest}").green());
    } else {
        ok(&format!("encoded -> {dest}"));
    }
    Ok(())
}
