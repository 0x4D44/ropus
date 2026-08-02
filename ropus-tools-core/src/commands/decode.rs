//! Decode: Ogg Opus → WAV (i16 or f32) or raw interleaved PCM.
//!
//! Pipeline (strict order — see HLD "ropusdec gaps"):
//!   packets → decode@48k (i16 or f32) → set_gain(header + --gain)
//!          → trim OpusHead.pre_skip samples @ 48 kHz
//!          → resample 48 kHz → --rate HZ (if set)
//!          → dither (i16 path only, unless --no-dither)
//!          → write WAV / raw
//!
//! Pre-skip must happen *before* the resample. The codec emits silence as part
//! of its warm-up; resampling that silence alongside real audio smears the
//! boundary and shifts the first-sample alignment by up to one resampler
//! kernel width.

use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor, Read, Write};
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use colored::*;

use ropus::{DecodeMode, Decoder as RopusDecoder};

use ogg::reading::PacketReader;

use crate::audio::decode::gain_db_to_q8;
use crate::audio::dither::{DITHER_SEED, PACKET_LOSS_SEED, Xorshift32, quantize_to_i16};
use crate::audio::resample::resample;
use crate::audio::wav::{
    write_wav_float32, write_wav_float32_to, write_wav_pcm16, write_wav_pcm16_to,
};
use crate::consts::OPUS_SR;
use crate::container::ogg::{
    OpusTags, UNKNOWN_GRANULE, parse_opus_head, validate_opus_audio_packet,
};
use crate::container::toc::decode_toc;
use crate::options::{DecodeOptions, OutputPolicy};
use crate::ui::{escape_terminal_path, escape_terminal_text, format_num, heading, ok};
use crate::util::{
    channel_count_to_ropus, is_stdio_sentinel, noncolliding_default_output,
    reject_input_output_alias,
};

/// Accepted output sample-rate range for `--rate`. Mirrors the WAV-supported
/// band (8 kHz for narrowband telephony up to 192 kHz high-res). rubato can
/// technically resample further in either direction, but rates outside this
/// band are almost always a user typo and the resulting WAV won't play in most
/// tools.
const MIN_OUTPUT_RATE: u32 = 8_000;
const MAX_OUTPUT_RATE: u32 = 192_000;

/// Return the per-channel decoded duration advertised by an Opus packet's
/// TOC. Simulated loss must size PLC from the packet being skipped, not from a
/// previous packet: Opus streams may change duration between packets and the
/// first packet has no previous duration at all.
fn packet_duration_samples(packet: &[u8], max_per_ch: usize) -> Result<usize> {
    let toc = decode_toc(packet).ok_or_else(|| anyhow!("lost Opus packet has no TOC byte"))?;
    let frame_count = usize::from(
        toc.frames
            .ok_or_else(|| anyhow!("lost Opus packet has an incomplete code-3 TOC"))?,
    );
    if frame_count == 0 {
        bail!("lost Opus packet advertises zero frames");
    }
    let frame_samples = usize::try_from(
        u64::from(toc.frame_size_cms)
            .checked_mul(u64::from(OPUS_SR))
            .ok_or_else(|| anyhow!("Opus TOC frame duration overflows"))?
            / 100_000,
    )
    .map_err(|_| anyhow!("Opus TOC frame duration does not fit in usize"))?;
    if frame_samples == 0 {
        bail!("lost Opus packet advertises a zero-duration frame");
    }
    let duration = frame_samples
        .checked_mul(frame_count)
        .ok_or_else(|| anyhow!("Opus packet duration overflows"))?;
    if duration > max_per_ch {
        bail!("lost Opus packet duration {duration} exceeds decoder maximum {max_per_ch} samples");
    }
    Ok(duration)
}

/// Type-erased `Read + Seek` bound used to plumb both `File`-backed and
/// `Cursor<Vec<u8>>`-backed (stdin) sources through the same `PacketReader`.
trait ReadSeek: std::io::Read + std::io::Seek {}
impl<T: std::io::Read + std::io::Seek> ReadSeek for T {}

pub fn decode(opts: DecodeOptions) -> Result<()> {
    decode_with_policy(opts, OutputPolicy::default())
}

pub fn decode_with_policy(opts: DecodeOptions, policy: OutputPolicy) -> Result<()> {
    // Validate all public option values before opening the input or creating
    // the output. This keeps GUI/plugin callers on the same safe boundary as
    // the Clap wrapper and prevents invalid options from consuming a stream.
    let user_gain_q8 =
        gain_db_to_q8(opts.gain_db).map_err(|e| anyhow!("--gain validation failed: {e}"))?;
    if opts.packet_loss_pct > 100 {
        bail!(
            "--packet-loss {} out of range (accepted: 0..=100)",
            opts.packet_loss_pct
        );
    }

    // Validate --rate before opening any files.
    if let Some(rate) = opts.rate
        && !(MIN_OUTPUT_RATE..=MAX_OUTPUT_RATE).contains(&rate)
    {
        // Likely a unit mix-up: 48 is 48 Hz (invalid), user probably meant
        // 48 kHz = 48000. Nudge them.
        if (8..=192).contains(&rate) {
            bail!(
                "--rate {rate} out of range (accepted: {MIN_OUTPUT_RATE}..={MAX_OUTPUT_RATE} Hz) \
                 (did you mean {}?)",
                rate * 1000
            );
        }
        bail!("--rate {rate} out of range (accepted: {MIN_OUTPUT_RATE}..={MAX_OUTPUT_RATE} Hz)");
    }

    // Resolve the output path. `-` and "input is stdin with no explicit -o"
    // both map to stdout (no sensible filename to derive from a pipe). Detect
    // stdout early so progress lines route to stderr — raw WAV/PCM bytes and
    // coloured banner text don't mix on the same fd.
    let input_is_stdin = is_stdio_sentinel(&opts.input);
    let default_ext = if opts.raw { "pcm" } else { "wav" };
    let output_path: std::path::PathBuf = match opts.output.clone() {
        Some(p) => p,
        None if input_is_stdin => std::path::PathBuf::from("-"),
        None => noncolliding_default_output(&opts.input, default_ext, "decoded")?,
    };
    let output_is_stdout = is_stdio_sentinel(&output_path);
    reject_input_output_alias(&opts.input, &output_path)?;

    // Progress/banner lines. Gated on output-sink so that piping bytes to
    // stdout doesn't mix with the banner text.
    macro_rules! report {
        ($($arg:tt)*) => {
            if !policy.quiet && output_is_stdout {
                eprintln!($($arg)*);
            } else if !policy.quiet {
                println!($($arg)*);
            }
        };
    }
    if !policy.quiet {
        if output_is_stdout {
            eprintln!("{}", "decode".bright_yellow().bold());
        } else {
            heading("decode");
        }
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

    // `PacketReader` takes any `Read + Seek`, so `Cursor<Vec<u8>>` from stdin
    // plugs in identically to a `File`. Stdin path buffers the whole stream
    // into a `Vec<u8>` first — memory-bounded by input size; documented risk.
    let mut reader: PacketReader<Box<dyn ReadSeek>> = if input_is_stdin {
        let mut buf = Vec::new();
        std::io::stdin()
            .lock()
            .read_to_end(&mut buf)
            .context("reading stdin into buffer")?;
        PacketReader::new(Box::new(Cursor::new(buf)))
    } else {
        let file = File::open(&opts.input)
            .with_context(|| format!("opening {}", escape_terminal_path(&opts.input)))?;
        PacketReader::new(Box::new(BufReader::new(file)))
    };

    // Header packet: OpusHead.
    let head_pkt = reader
        .read_packet()?
        .ok_or_else(|| anyhow!("no packets found in input"))?;
    let stream_serial = head_pkt.stream_serial();
    let head = parse_opus_head(&head_pkt.data)?;
    report!(
        "header   ch={} input_sr={} pre_skip={}",
        head.channels.to_string().bright_white(),
        head.input_sample_rate.to_string().bright_white(),
        head.pre_skip.to_string().bright_white(),
    );

    // Tags packet: OpusTags. Parse rather than verify-only so malformed files
    // (stripped tags page, truncated lengths, non-UTF-8 vendor) fail here with
    // a useful error instead of silently consuming the first audio packet.
    let tags_pkt = reader
        .read_packet()?
        .ok_or_else(|| anyhow!("expected OpusTags packet, got end of stream"))?;
    if tags_pkt.stream_serial() != stream_serial {
        bail!("OpusTags packet belongs to a different Ogg stream");
    }
    let tags = OpusTags::parse(&tags_pkt.data).context("parsing OpusTags packet")?;
    report!(
        "tags     vendor={}, {} comments",
        format!("\"{}\"", escape_terminal_text(&tags.vendor)).bright_white(),
        tags.comments.len().to_string().bright_white(),
    );

    let opus_channels = channel_count_to_ropus(head.channels as usize)?;
    let ch_count = opus_channels.count();

    let mut decoder = RopusDecoder::new(OPUS_SR, opus_channels)
        .map_err(|e| anyhow!("decoder init failed: {e}"))?;

    // Combine header's Q8 output_gain with --gain DB (also Q8 after conversion)
    // and apply via set_gain. libopus range-checks the sum ([-32768, 32767]);
    // the positive endpoint is one Q8 step below +128 dB.
    let total_gain_q8 = (head.output_gain as i32)
        .checked_add(user_gain_q8)
        .ok_or_else(|| anyhow!("header and user gain overflow in Q8"))?;
    if !(-32_768..=32_767).contains(&total_gain_q8) {
        bail!(
            "header and user gain total {total_gain_q8} Q8 is outside decoder range [-32768, 32767]"
        );
    }
    if total_gain_q8 != 0 {
        decoder
            .set_gain(total_gain_q8)
            .map_err(|e| anyhow!("set_gain({total_gain_q8} Q8) failed: {e}"))?;
    }

    // Decode pipeline selection. We decode directly to whichever precision
    // we ultimately need:
    //   • `--float`       -> user asked for f32 output
    //   • `--rate HZ`     -> resample operates on f32 (rubato's native unit)
    //   • dither enabled  -> dither needs true f32 samples; adding ±1 LSB
    //                        noise to an already-quantised i16 only flips the
    //                        LSB and does none of the decorrelation work.
    // For the pure i16 path (no float, no resample, no dither) we keep i16
    // end-to-end — no f32 round-trip, no silent ±1 LSB drift versus the raw
    // decoder output.
    let use_float_decode = opts.float || opts.rate.is_some() || opts.dither;

    // Active-flags banner. Print each configuration knob that deviates from the
    // default so a user looking at output diffs can see immediately which flags
    // were in effect.
    report!(
        "format   {}",
        (if opts.float { "f32" } else { "i16" }).bright_white()
    );
    if opts.gain_db != 0.0 {
        report!(
            "gain     header={} Q8 + user={:.2} dB -> {} Q8",
            head.output_gain.to_string().bright_white(),
            opts.gain_db,
            total_gain_q8.to_string().bright_white(),
        );
    }
    if !opts.float {
        report!(
            "dither   {}",
            (if opts.dither { "on" } else { "off" }).bright_white()
        );
    }
    let output_rate = opts.rate.unwrap_or(OPUS_SR);
    if let Some(rate) = opts.rate {
        report!(
            "rate     {} -> {} Hz",
            OPUS_SR.to_string().bright_white(),
            rate.to_string().bright_white(),
        );
    }
    if opts.packet_loss_pct > 0 {
        report!(
            "loss     simulating {}% packet drops (deterministic seed)",
            opts.packet_loss_pct.to_string().bright_white(),
        );
    }
    if opts.raw {
        report!("mode     raw (no WAV header)");
    }

    // Maximum per-channel samples decodable from one packet (120 ms @ 48 kHz).
    let max_per_ch = (OPUS_SR / 1000 * 120) as usize;

    // PRNGs. Construct unconditionally — a fresh xorshift32 is eight bytes and
    // never observed when the corresponding flag is off (short-circuits below).
    let mut dither_rng = Xorshift32::new(DITHER_SEED);
    let mut loss_rng = Xorshift32::new(PACKET_LOSS_SEED);

    let mut packet_count: u64 = 0;
    let mut dropped_count: u64 = 0;
    let mut eos_granule: Option<u64> = None;

    // We accumulate decoded samples and write a single output when finished.
    // Branch on the decode precision once; the inner loops are otherwise
    // identical.
    let (all_pcm_i16, all_pcm_f32) = if use_float_decode {
        let mut scratch = vec![0.0f32; max_per_ch * ch_count];
        let mut acc: Vec<f32> = Vec::with_capacity(1 << 20);
        while let Some(pkt) = reader.read_packet()? {
            if pkt.stream_serial() != stream_serial {
                continue;
            }
            validate_opus_audio_packet(&pkt.data)
                .with_context(|| format!("validating Opus audio packet {packet_count}"))?;
            let lost = opts.packet_loss_pct > 0
                && (loss_rng.next_u32() % 100) < u32::from(opts.packet_loss_pct);
            let n = if lost {
                dropped_count += 1;
                let frame_samples = packet_duration_samples(&pkt.data, max_per_ch)?;
                let plc_len = frame_samples
                    .checked_mul(ch_count)
                    .ok_or_else(|| anyhow!("PLC output sample count overflows"))?;
                let n = decoder
                    .decode_float(&[], &mut scratch[..plc_len], DecodeMode::Normal)
                    .map_err(|e| anyhow!("decode_float PLC failed: {e}"))?;
                if n != frame_samples {
                    bail!("PLC decoded {n} samples for a {frame_samples}-sample lost packet");
                }
                n
            } else {
                decoder
                    .decode_float(&pkt.data, &mut scratch, DecodeMode::Normal)
                    .map_err(|e| anyhow!("decode_float failed: {e}"))?
            };
            let total = n * ch_count;
            acc.extend_from_slice(&scratch[..total]);
            packet_count += 1;
            if pkt.last_in_stream() {
                eos_granule = Some(pkt.absgp_page());
                break;
            }
        }
        (Vec::new(), acc)
    } else {
        let mut scratch = vec![0i16; max_per_ch * ch_count];
        let mut acc: Vec<i16> = Vec::with_capacity(1 << 20);
        while let Some(pkt) = reader.read_packet()? {
            if pkt.stream_serial() != stream_serial {
                continue;
            }
            validate_opus_audio_packet(&pkt.data)
                .with_context(|| format!("validating Opus audio packet {packet_count}"))?;
            let lost = opts.packet_loss_pct > 0
                && (loss_rng.next_u32() % 100) < u32::from(opts.packet_loss_pct);
            let n = if lost {
                dropped_count += 1;
                let frame_samples = packet_duration_samples(&pkt.data, max_per_ch)?;
                let plc_len = frame_samples
                    .checked_mul(ch_count)
                    .ok_or_else(|| anyhow!("PLC output sample count overflows"))?;
                let n = decoder
                    .decode(&[], &mut scratch[..plc_len], DecodeMode::Normal)
                    .map_err(|e| anyhow!("decode PLC failed: {e}"))?;
                if n != frame_samples {
                    bail!("PLC decoded {n} samples for a {frame_samples}-sample lost packet");
                }
                n
            } else {
                decoder
                    .decode(&pkt.data, &mut scratch, DecodeMode::Normal)
                    .map_err(|e| anyhow!("decode failed: {e}"))?
            };
            let total = n * ch_count;
            acc.extend_from_slice(&scratch[..total]);
            packet_count += 1;
            if pkt.last_in_stream() {
                eos_granule = Some(pkt.absgp_page());
                break;
            }
        }
        (acc, Vec::new())
    };

    let eos_granule =
        eos_granule.ok_or_else(|| anyhow!("selected Opus stream ended before an Ogg EOS page"))?;
    if eos_granule == UNKNOWN_GRANULE {
        bail!("selected Opus stream has an unknown EOS granule position");
    }
    let eos_granule = usize::try_from(eos_granule)
        .map_err(|_| anyhow!("EOS granule position does not fit in this platform's usize"))?;
    let pre_skip_ch = usize::from(head.pre_skip);
    if eos_granule < pre_skip_ch {
        bail!("EOS granule {eos_granule} is smaller than OpusHead pre-skip {pre_skip_ch}");
    }

    // Trim the leading pre-skip samples. Applied at 48 kHz *before* any
    // resample (see module docstring).
    let pre_skip_samples = head.pre_skip as usize * ch_count;
    let total_before_trim = if use_float_decode {
        all_pcm_f32.len()
    } else {
        all_pcm_i16.len()
    };
    let eos_samples = eos_granule
        .checked_mul(ch_count)
        .ok_or_else(|| anyhow!("EOS granule sample count overflows"))?;
    if total_before_trim < eos_samples {
        bail!(
            "decoded {} samples, but EOS granule requires {} samples",
            total_before_trim,
            eos_samples
        );
    }
    let pre_skip = pre_skip_samples;

    // If --rate was requested, resample after pre-skip and before dither/write.
    // Resample always happens on f32 (rubato's native unit); reaching this
    // branch implies `use_float_decode == true`, so we already have f32 samples
    // in `all_pcm_f32`.
    let need_resample = output_rate != OPUS_SR;

    if use_float_decode {
        let trimmed_f32: &[f32] = &all_pcm_f32[pre_skip..eos_samples];
        let resampled: Vec<f32> = if need_resample {
            resample(trimmed_f32, OPUS_SR, output_rate, ch_count)
                .context("resampling decoded PCM")?
        } else {
            trimmed_f32.to_vec()
        };
        if opts.float {
            // User asked for f32 output: write the f32 samples directly.
            write_output_samples(
                &output_path,
                output_is_stdout,
                OutputData::Float(&resampled),
                output_rate,
                ch_count as u16,
                opts.raw,
            )?;
            report_and_return(
                packet_count,
                dropped_count,
                total_before_trim as u64,
                resampled.len() as u64,
                &output_path,
                output_is_stdout,
                policy.quiet,
            )
        } else {
            // i16 output via the f32 pipeline (triggered by --rate or dither).
            // `opts.dither` drives whether `quantize_to_i16` adds TPDF noise
            // before the round-and-clamp.
            let i16_out = quantize_to_i16(&resampled, opts.dither, &mut dither_rng);
            write_output_samples(
                &output_path,
                output_is_stdout,
                OutputData::I16(&i16_out),
                output_rate,
                ch_count as u16,
                opts.raw,
            )?;
            report_and_return(
                packet_count,
                dropped_count,
                total_before_trim as u64,
                i16_out.len() as u64,
                &output_path,
                output_is_stdout,
                policy.quiet,
            )
        }
    } else {
        // Pure i16 path: no resample, no dither, no float request. Pass the
        // decoder's i16 output straight to the writer — no f32 round-trip,
        // so every sample survives bit-identical to what ropus emitted.
        let trimmed_i16: &[i16] = &all_pcm_i16[pre_skip..eos_samples];
        write_output_samples(
            &output_path,
            output_is_stdout,
            OutputData::I16(trimmed_i16),
            output_rate,
            ch_count as u16,
            opts.raw,
        )?;
        report_and_return(
            packet_count,
            dropped_count,
            total_before_trim as u64,
            trimmed_i16.len() as u64,
            &output_path,
            output_is_stdout,
            policy.quiet,
        )
    }
}

enum OutputData<'a> {
    I16(&'a [i16]),
    Float(&'a [f32]),
}

/// Open the output sink (locked stdout or a newly-created file) and invoke
/// the caller-supplied closure with a mutable `Write` reference. Consolidates
/// the four raw-or-WAV × path-or-stdout combinations into one flush point.
fn with_output_sink<F>(output: &Path, output_is_stdout: bool, body: F) -> Result<()>
where
    F: FnOnce(&mut dyn Write) -> Result<()>,
{
    if output_is_stdout {
        let stdout = std::io::stdout();
        let mut w = BufWriter::new(stdout.lock());
        body(&mut w)?;
        w.flush()?;
    } else {
        let f = File::create(output)
            .with_context(|| format!("creating {}", escape_terminal_path(output)))?;
        let mut w = BufWriter::new(f);
        body(&mut w)?;
        w.flush()?;
    }
    Ok(())
}

fn write_output_samples(
    output: &Path,
    output_is_stdout: bool,
    data: OutputData<'_>,
    sample_rate: u32,
    channels: u16,
    raw: bool,
) -> Result<()> {
    match (data, raw) {
        (OutputData::I16(samples), true) => with_output_sink(output, output_is_stdout, |w| {
            for s in samples {
                w.write_all(&s.to_le_bytes())?;
            }
            Ok(())
        }),
        (OutputData::I16(samples), false) => {
            if output_is_stdout {
                with_output_sink(output, true, |w| {
                    write_wav_pcm16_to(w, samples, sample_rate, channels).context("writing WAV")
                })
            } else {
                write_wav_pcm16(output, samples, sample_rate, channels).context("writing WAV")
            }
        }
        (OutputData::Float(samples), true) => with_output_sink(output, output_is_stdout, |w| {
            for s in samples {
                w.write_all(&s.to_le_bytes())?;
            }
            Ok(())
        }),
        (OutputData::Float(samples), false) => {
            if output_is_stdout {
                with_output_sink(output, true, |w| {
                    write_wav_float32_to(w, samples, sample_rate, channels)
                        .context("writing float WAV")
                })
            } else {
                write_wav_float32(output, samples, sample_rate, channels)
                    .context("writing float WAV")
            }
        }
    }
}

fn report_and_return(
    packet_count: u64,
    dropped_count: u64,
    total_samples: u64,
    emitted_samples: u64,
    output: &Path,
    output_is_stdout: bool,
    quiet: bool,
) -> Result<()> {
    if quiet {
        return Ok(());
    }
    // Mirror the progress-banner gating inside `decode()`: progress lines
    // must not land on stdout when it's the bitstream sink.
    let line = format!(
        "decoded  {} packets{}, {} samples ({} emitted)",
        format_num(packet_count).bright_white(),
        if dropped_count > 0 {
            format!(" ({} dropped for PLC)", format_num(dropped_count))
                .yellow()
                .to_string()
        } else {
            String::new()
        },
        format_num(total_samples).bright_white(),
        format_num(emitted_samples).bright_white(),
    );
    let dest = if output_is_stdout {
        "<stdout>".to_string()
    } else {
        escape_terminal_path(output)
    };
    if output_is_stdout {
        eprintln!("{line}");
        eprintln!("{}", format!("decoded -> {dest}").green());
    } else {
        println!("{line}");
        ok(&format!("decoded -> {dest}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packet_duration_samples_reads_single_frame_duration() {
        assert_eq!(packet_duration_samples(&[0x00], 5_760).unwrap(), 480);
        assert_eq!(packet_duration_samples(&[0x18], 5_760).unwrap(), 2_880);
    }

    #[test]
    fn packet_duration_samples_reads_code_three_frame_count() {
        assert_eq!(packet_duration_samples(&[0x03, 0x02], 5_760).unwrap(), 960);
    }

    #[test]
    fn packet_duration_samples_does_not_carry_duration_between_packets() {
        let packets = [[0x00], [0x18], [0x00]];
        let durations: Vec<_> = packets
            .iter()
            .map(|packet| packet_duration_samples(packet, 5_760).unwrap())
            .collect();
        assert_eq!(durations, [480, 2_880, 480]);
    }

    #[test]
    fn packet_duration_samples_rejects_incomplete_or_oversized_toc() {
        assert!(packet_duration_samples(&[], 5_760).is_err());
        assert!(packet_duration_samples(&[0x03], 5_760).is_err());
        assert!(packet_duration_samples(&[0x03, 0x00], 5_760).is_err());
        assert!(packet_duration_samples(&[0x03, 0x3f], 5_760).is_err());
    }

    #[test]
    fn public_decode_options_reject_invalid_loss_before_io() {
        let output = std::env::temp_dir().join(format!(
            "ropus_invalid_loss_{}_{}.wav",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock")
                .as_nanos()
        ));
        let err = decode(DecodeOptions {
            input: "missing-input.opus".into(),
            output: Some(output.clone()),
            float: false,
            raw: false,
            rate: None,
            gain_db: 0.0,
            dither: true,
            packet_loss_pct: 101,
        })
        .expect_err("loss above 100 must fail before opening input");
        assert!(format!("{err:#}").contains("packet-loss"));
        assert!(!output.exists(), "invalid options must not create output");
    }

    #[test]
    fn public_decode_options_reject_unrepresentable_gain_before_io() {
        let output = std::env::temp_dir().join(format!(
            "ropus_invalid_gain_{}_{}.wav",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock")
                .as_nanos()
        ));
        let err = decode(DecodeOptions {
            input: "missing-input.opus".into(),
            output: Some(output.clone()),
            float: false,
            raw: false,
            rate: None,
            gain_db: 128.0,
            dither: true,
            packet_loss_pct: 0,
        })
        .expect_err("+128 dB cannot fit decoder Q8");
        assert!(format!("{err:#}").contains("gain"));
        assert!(!output.exists(), "invalid options must not create output");
    }

    #[test]
    fn decode_rejects_direct_input_output_alias_before_reading() {
        let path = std::env::temp_dir().join(format!(
            "ropus_decode_alias_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock")
                .as_nanos()
        ));
        let original = b"source bytes that must survive".to_vec();
        std::fs::write(&path, &original).expect("write input");

        let error = decode(DecodeOptions {
            input: path.clone(),
            output: Some(path.clone()),
            float: false,
            raw: false,
            rate: None,
            gain_db: 0.0,
            dither: true,
            packet_loss_pct: 0,
        })
        .expect_err("direct alias must be rejected");
        assert!(error.to_string().contains("same file"));
        assert_eq!(std::fs::read(&path).expect("read input"), original);
        std::fs::remove_file(path).expect("remove input");
    }
}
