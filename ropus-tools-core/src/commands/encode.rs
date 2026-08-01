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
use crate::ui::{escape_terminal_path, format_num, heading, ok};
use crate::util::{
    channel_count_to_ropus, is_stdio_sentinel, noncolliding_default_output,
    reject_input_output_alias,
};

use ropus::FrameDuration;

/// Per-channel sample count for each supported `FrameDuration` at 48 kHz.
/// `2.5 ms * 48 = 120` samples (exact). Kept as a function rather than a table
/// so adding a new FrameDuration variant upstream fails loudly at compile
/// time via an unmatched arm.
fn frame_samples_per_ch(d: FrameDuration) -> Result<usize> {
    Ok(match d {
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
        FrameDuration::Argument => bail!(
            "frame duration Argument is not supported; choose an explicit duration from 2.5 to 120 ms"
        ),
    })
}

fn validate_encode_options(opts: &EncodeOptions) -> Result<()> {
    frame_samples_per_ch(opts.frame_duration)?;
    if let Some(complexity) = opts.complexity
        && complexity > 10
    {
        bail!("complexity {complexity} out of range (accepted: 0..=10)");
    }
    if let Some(bitrate) = opts.bitrate {
        if bitrate == 0 {
            bail!("bitrate must be greater than zero");
        }
        if bitrate > i32::MAX as u32 {
            bail!("bitrate {bitrate} bps exceeds the libopus i32::MAX limit");
        }
    }
    if opts.expect_loss > 100 {
        bail!(
            "expected packet loss {} out of range (accepted: 0..=100)",
            opts.expect_loss
        );
    }
    Ok(())
}

pub fn encode(opts: EncodeOptions) -> Result<()> {
    validate_encode_options(&opts)?;

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
        None => noncolliding_default_output(&opts.input, "opus", "encoded")?,
    };
    let output_is_stdout = is_stdio_sentinel(&output_path);
    reject_input_output_alias(&opts.input, &output_path)?;

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
            escape_terminal_path(&opts.input).cyan().to_string()
        }
    );
    report!(
        "output   {}",
        if output_is_stdout {
            "<stdout>".cyan().to_string()
        } else {
            escape_terminal_path(&output_path).cyan().to_string()
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

    // A valid Ogg Opus stream needs at least one audio packet carrying EOS.
    // Reject empty decoded input before creating the destination so callers
    // never receive a header-only, unterminated logical stream.
    if pcm_48k.is_empty() {
        bail!("decoded input contains no audio samples");
    }
    if pcm_48k.len() % channels != 0 {
        bail!(
            "decoded input has {} interleaved samples for {} channels",
            pcm_48k.len(),
            channels
        );
    }
    let source_samples_ch = pcm_48k.len() / channels;
    let source_samples_ch_u64 = source_samples_ch as u64;

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
        let file = File::create(&output_path).with_context(|| {
            format!(
                "creating output file {}",
                escape_terminal_path(&output_path)
            )
        })?;
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
        let meta = std::fs::metadata(pic_path).with_context(|| {
            format!(
                "reading picture metadata {}",
                escape_terminal_path(pic_path)
            )
        })?;
        if meta.len() > MAX_PICTURE_BYTES {
            bail!(
                "picture file {} is {} bytes; refusing > {} bytes (use a smaller cover image)",
                escape_terminal_path(pic_path),
                meta.len(),
                MAX_PICTURE_BYTES,
            );
        }
        let data = std::fs::read(pic_path)
            .with_context(|| format!("reading picture file {}", escape_terminal_path(pic_path)))?;
        if data.is_empty() {
            bail!("picture file {} is empty", escape_terminal_path(pic_path));
        }
        let format = detect_format(&data).with_context(|| {
            format!(
                "detecting picture format for {}",
                escape_terminal_path(pic_path)
            )
        })?;
        let block = build_picture_block(format, &data).with_context(|| {
            format!(
                "building picture block for {}",
                escape_terminal_path(pic_path)
            )
        })?;
        let b64 = base64_encode(&block);
        comments.insert(0, format!("METADATA_BLOCK_PICTURE={b64}"));
        report!(
            "picture  {} ({} bytes, {})",
            escape_terminal_path(pic_path).cyan(),
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
    // The encoder delay represented by pre_skip must be drained with trailing
    // silence, otherwise exact-frame inputs decode short. Packet rounding may
    // add more silence, but the EOS granule trims that padding by declaring the
    // exact endpoint: source samples + pre_skip, in the fixed 48 kHz clock.
    let frame_samples_ch = frame_samples_per_ch(opts.frame_duration)?;
    let frame_interleaved = frame_samples_ch * channels;
    let final_granule = source_samples_ch_u64
        .checked_add(u64::from(pre_skip))
        .ok_or_else(|| anyhow!("source duration plus pre-skip overflows the Ogg granule clock"))?;
    let packet_target_u64 = final_granule.div_ceil(frame_samples_ch as u64);
    let packet_target = usize::try_from(packet_target_u64)
        .map_err(|_| anyhow!("required Opus packet count does not fit in usize"))?;

    let mut packet_buf = vec![0u8; MAX_PACKET_BYTES];
    let mut padded_frame = vec![0.0f32; frame_interleaved];
    let mut submitted_samples_ch: u64 = 0;
    let mut packet_count: u64 = 0;
    let mut payload_bytes: u64 = 0;
    for idx in 0..packet_target {
        let start = idx
            .checked_mul(frame_interleaved)
            .ok_or_else(|| anyhow!("Opus input frame offset overflow"))?;
        let source_start = start.min(pcm_48k.len());
        let source_end = start.saturating_add(frame_interleaved).min(pcm_48k.len());
        let source_chunk = &pcm_48k[source_start..source_end];
        let encode_chunk = if source_chunk.len() == frame_interleaved {
            source_chunk
        } else {
            padded_frame.fill(0.0);
            padded_frame[..source_chunk.len()].copy_from_slice(source_chunk);
            &padded_frame
        };

        let n = encoder
            .encode_float(encode_chunk, &mut packet_buf)
            .map_err(|e| anyhow!("encode failed: {e}"))?;
        submitted_samples_ch += frame_samples_ch as u64;
        payload_bytes += n as u64;
        let is_last = idx + 1 == packet_target;
        let end_info = if is_last {
            PacketWriteEndInfo::EndStream
        } else {
            PacketWriteEndInfo::NormalPacket
        };
        let granule = if is_last {
            final_granule
        } else {
            submitted_samples_ch
        };
        writer
            .write_packet(packet_buf[..n].to_vec(), serial, end_info, granule)
            .context("writing Opus data page")?;
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
        format_num(final_granule).bright_white(),
    );
    // Average payload bitrate (Opus packets only — excludes Ogg framing
    // overhead, which is what `--bitrate` controls and what users compare
    // against the target). RFC 7845 fixes the granule clock at 48 kHz, so
    // duration_seconds = source_samples / OPUS_SR (pre-skip and packet padding
    // are not audible source duration).
    if let Some(avg_bps) = (payload_bytes * 8 * (OPUS_SR as u64)).checked_div(source_samples_ch_u64)
    {
        report!(
            "bitrate  {} kbps avg (payload)",
            format!("{:.1}", avg_bps as f64 / 1000.0).bright_white(),
        );
    }
    let dest = if output_is_stdout {
        "<stdout>".to_string()
    } else {
        escape_terminal_path(&output_path)
    };
    if output_is_stdout {
        eprintln!("{}", format!("encoded -> {dest}").green());
    } else {
        ok(&format!("encoded -> {dest}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Application, Signal};

    fn valid_options() -> EncodeOptions {
        EncodeOptions {
            input: "missing-input.wav".into(),
            output: Some("missing-output.opus".into()),
            bitrate: Some(64_000),
            complexity: Some(10),
            application: Application::Audio,
            vbr: true,
            vbr_constraint: false,
            signal: Signal::Auto,
            frame_duration: FrameDuration::Ms20,
            expect_loss: 0,
            downmix_to_mono: false,
            serial: None,
            picture_path: None,
            vendor: "test".to_string(),
            comments: Vec::new(),
        }
    }

    #[test]
    fn public_encode_options_reject_invalid_values_before_io() {
        let mut opts = valid_options();
        opts.frame_duration = FrameDuration::Argument;
        assert!(validate_encode_options(&opts).is_err());

        let mut opts = valid_options();
        opts.complexity = Some(11);
        assert!(validate_encode_options(&opts).is_err());

        let mut opts = valid_options();
        opts.bitrate = Some(u32::MAX);
        assert!(validate_encode_options(&opts).is_err());

        let mut opts = valid_options();
        opts.expect_loss = 101;
        assert!(validate_encode_options(&opts).is_err());
    }

    #[test]
    fn encode_rejects_direct_input_output_alias_before_decode() {
        let path = std::env::temp_dir().join(format!(
            "ropus_encode_alias_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock")
                .as_nanos()
        ));
        let original = b"source bytes that must survive".to_vec();
        std::fs::write(&path, &original).expect("write input");

        let mut opts = valid_options();
        opts.input = path.clone();
        opts.output = Some(path.clone());
        let error = encode(opts).expect_err("direct alias must be rejected");
        assert!(error.to_string().contains("same file"));
        assert_eq!(std::fs::read(&path).expect("read input"), original);
        std::fs::remove_file(path).expect("remove input");
    }
}
