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
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use colored::*;

use ropus::{DecodeMode, Decoder as RopusDecoder};

use ogg::reading::PacketReader;

use crate::audio::dither::{DITHER_SEED, PACKET_LOSS_SEED, Xorshift32, quantize_to_i16};
use crate::audio::resample::resample;
use crate::audio::wav::{write_wav_float32, write_wav_pcm16};
use crate::consts::OPUS_SR;
use crate::container::ogg::{OpusTags, parse_opus_head};
use crate::options::DecodeOptions;
use crate::ui::{format_num, heading, ok};
use crate::util::{channel_count_to_ropus, with_extension};

/// Accepted output sample-rate range for `--rate`. Mirrors the WAV-supported
/// band (8 kHz for narrowband telephony up to 192 kHz high-res). rubato can
/// technically resample further in either direction, but rates outside this
/// band are almost always a user typo and the resulting WAV won't play in most
/// tools.
const MIN_OUTPUT_RATE: u32 = 8_000;
const MAX_OUTPUT_RATE: u32 = 192_000;

pub fn decode(opts: DecodeOptions) -> Result<()> {
    heading("decode");
    println!("input    {}", opts.input.display().to_string().cyan());

    // Validate --gain before opening any files. NaN or ±∞ would saturate to 0
    // when cast to the Q8 i32 later, silently ignoring the flag; surface that
    // as a clean error instead.
    if !opts.gain_db.is_finite() {
        bail!("--gain must be finite, got {}", opts.gain_db);
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
        bail!(
            "--rate {rate} out of range (accepted: {MIN_OUTPUT_RATE}..={MAX_OUTPUT_RATE} Hz)"
        );
    }

    // Default output extension matches the chosen container: `.wav` for WAV
    // writers, `.pcm` for `--raw` (an unlabeled binary blob is less confusing
    // to land on disk than one that lies about being a WAV).
    let default_ext = if opts.raw { "pcm" } else { "wav" };
    let output = opts
        .output
        .clone()
        .unwrap_or_else(|| with_extension(&opts.input, default_ext));
    println!("output   {}", output.display().to_string().cyan());

    let file = File::open(&opts.input)
        .with_context(|| format!("opening {}", opts.input.display()))?;
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

    // Tags packet: OpusTags. Parse rather than verify-only so malformed files
    // (stripped tags page, truncated lengths, non-UTF-8 vendor) fail here with
    // a useful error instead of silently consuming the first audio packet.
    let tags_pkt = reader
        .read_packet()?
        .ok_or_else(|| anyhow!("expected OpusTags packet, got end of stream"))?;
    let tags = OpusTags::parse(&tags_pkt.data).context("parsing OpusTags packet")?;
    println!(
        "tags     vendor={}, {} comments",
        format!("\"{}\"", tags.vendor).bright_white(),
        tags.comments.len().to_string().bright_white(),
    );

    let opus_channels = channel_count_to_ropus(head.channels as usize)?;
    let ch_count = opus_channels.count();

    let mut decoder = RopusDecoder::new(OPUS_SR, opus_channels)
        .map_err(|e| anyhow!("decoder init failed: {e}"))?;

    // Combine header's Q8 output_gain with --gain DB (also Q8 after conversion)
    // and apply via set_gain. libopus range-checks the sum ([-32768, 32767])
    // which translates to ±128 dB; out of range surfaces as a clean error.
    let user_gain_q8 = (opts.gain_db * 256.0).round() as i32;
    let total_gain_q8 = head.output_gain as i32 + user_gain_q8;
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
    println!(
        "format   {}",
        (if opts.float { "f32" } else { "i16" })
            .bright_white()
    );
    if opts.gain_db != 0.0 {
        println!(
            "gain     header={} Q8 + user={:.2} dB -> {} Q8",
            head.output_gain.to_string().bright_white(),
            opts.gain_db,
            total_gain_q8.to_string().bright_white(),
        );
    }
    if !opts.float {
        println!(
            "dither   {}",
            (if opts.dither { "on" } else { "off" }).bright_white()
        );
    }
    let output_rate = opts.rate.unwrap_or(OPUS_SR);
    if let Some(rate) = opts.rate {
        println!(
            "rate     {} -> {} Hz",
            OPUS_SR.to_string().bright_white(),
            rate.to_string().bright_white(),
        );
    }
    if opts.packet_loss_pct > 0 {
        println!(
            "loss     simulating {}% packet drops (deterministic seed)",
            opts.packet_loss_pct.to_string().bright_white(),
        );
    }
    if opts.raw {
        println!("mode     raw (no WAV header)");
    }

    // Maximum per-channel samples decodable from one packet (120 ms @ 48 kHz).
    let max_per_ch = (OPUS_SR / 1000 * 120) as usize;

    // PRNGs. Construct unconditionally — a fresh xorshift32 is eight bytes and
    // never observed when the corresponding flag is off (short-circuits below).
    let mut dither_rng = Xorshift32::new(DITHER_SEED);
    let mut loss_rng = Xorshift32::new(PACKET_LOSS_SEED);

    let mut packet_count: u64 = 0;
    let mut dropped_count: u64 = 0;

    // Per-channel frame size of the last successfully-decoded real packet.
    // Used to size the PLC output buffer on simulated loss — libopus' PLC
    // fills whatever buffer it's given, so handing it a 120 ms scratch for a
    // 20 ms stream inflates the output 6× per drop. When no real packet has
    // decoded yet (first-packet loss), fall back to 960 samples (20 ms at
    // 48 kHz) — matches libopus `opusdec`'s behaviour and is the modal Opus
    // frame duration on the wire.
    let mut last_frame_samples: usize = 960;

    // We accumulate decoded samples and write a single output when finished.
    // Branch on the decode precision once; the inner loops are otherwise
    // identical.
    let (all_pcm_i16, all_pcm_f32) = if use_float_decode {
        let mut scratch = vec![0.0f32; max_per_ch * ch_count];
        let mut acc: Vec<f32> = Vec::with_capacity(1 << 20);
        while let Some(pkt) = reader.read_packet()? {
            let lost = opts.packet_loss_pct > 0
                && (loss_rng.next_u32() % 100) < u32::from(opts.packet_loss_pct);
            let n = if lost {
                dropped_count += 1;
                let plc_len = last_frame_samples * ch_count;
                decoder
                    .decode_float(&[], &mut scratch[..plc_len], DecodeMode::Normal)
                    .map_err(|e| anyhow!("decode_float PLC failed: {e}"))?
            } else {
                let n = decoder
                    .decode_float(&pkt.data, &mut scratch, DecodeMode::Normal)
                    .map_err(|e| anyhow!("decode_float failed: {e}"))?;
                last_frame_samples = n;
                n
            };
            let total = n * ch_count;
            acc.extend_from_slice(&scratch[..total]);
            packet_count += 1;
        }
        (Vec::new(), acc)
    } else {
        let mut scratch = vec![0i16; max_per_ch * ch_count];
        let mut acc: Vec<i16> = Vec::with_capacity(1 << 20);
        while let Some(pkt) = reader.read_packet()? {
            let lost = opts.packet_loss_pct > 0
                && (loss_rng.next_u32() % 100) < u32::from(opts.packet_loss_pct);
            let n = if lost {
                dropped_count += 1;
                let plc_len = last_frame_samples * ch_count;
                decoder
                    .decode(&[], &mut scratch[..plc_len], DecodeMode::Normal)
                    .map_err(|e| anyhow!("decode PLC failed: {e}"))?
            } else {
                let n = decoder
                    .decode(&pkt.data, &mut scratch, DecodeMode::Normal)
                    .map_err(|e| anyhow!("decode failed: {e}"))?;
                last_frame_samples = n;
                n
            };
            let total = n * ch_count;
            acc.extend_from_slice(&scratch[..total]);
            packet_count += 1;
        }
        (acc, Vec::new())
    };

    // Trim the leading pre-skip samples. Applied at 48 kHz *before* any
    // resample (see module docstring).
    let pre_skip_samples = head.pre_skip as usize * ch_count;
    let total_before_trim = if use_float_decode {
        all_pcm_f32.len()
    } else {
        all_pcm_i16.len()
    };
    let pre_skip = pre_skip_samples.min(total_before_trim);

    // If --rate was requested, resample after pre-skip and before dither/write.
    // Resample always happens on f32 (rubato's native unit); reaching this
    // branch implies `use_float_decode == true`, so we already have f32 samples
    // in `all_pcm_f32`.
    let need_resample = output_rate != OPUS_SR;

    if use_float_decode {
        let trimmed_f32: &[f32] = &all_pcm_f32[pre_skip..];
        let resampled: Vec<f32> = if need_resample {
            resample(trimmed_f32, OPUS_SR, output_rate, ch_count)
                .context("resampling decoded PCM")?
        } else {
            trimmed_f32.to_vec()
        };
        if opts.float {
            // User asked for f32 output: write the f32 samples directly.
            write_output_samples(
                &output,
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
                &output,
            )
        } else {
            // i16 output via the f32 pipeline (triggered by --rate or dither).
            // `opts.dither` drives whether `quantize_to_i16` adds TPDF noise
            // before the round-and-clamp.
            let i16_out = quantize_to_i16(&resampled, opts.dither, &mut dither_rng);
            write_output_samples(
                &output,
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
                &output,
            )
        }
    } else {
        // Pure i16 path: no resample, no dither, no float request. Pass the
        // decoder's i16 output straight to the writer — no f32 round-trip,
        // so every sample survives bit-identical to what ropus emitted.
        let trimmed_i16: &[i16] = &all_pcm_i16[pre_skip..];
        write_output_samples(
            &output,
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
            &output,
        )
    }
}

enum OutputData<'a> {
    I16(&'a [i16]),
    Float(&'a [f32]),
}

fn write_output_samples(
    output: &Path,
    data: OutputData<'_>,
    sample_rate: u32,
    channels: u16,
    raw: bool,
) -> Result<()> {
    match data {
        OutputData::I16(samples) => {
            if raw {
                let f = File::create(output)
                    .with_context(|| format!("creating {}", output.display()))?;
                let mut w = BufWriter::new(f);
                for s in samples {
                    w.write_all(&s.to_le_bytes())?;
                }
                w.flush()?;
                Ok(())
            } else {
                write_wav_pcm16(output, samples, sample_rate, channels).context("writing WAV")
            }
        }
        OutputData::Float(samples) => {
            if raw {
                let f = File::create(output)
                    .with_context(|| format!("creating {}", output.display()))?;
                let mut w = BufWriter::new(f);
                for s in samples {
                    w.write_all(&s.to_le_bytes())?;
                }
                w.flush()?;
                Ok(())
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
) -> Result<()> {
    println!(
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
    ok(&format!("decoded -> {}", output.display()));
    Ok(())
}
