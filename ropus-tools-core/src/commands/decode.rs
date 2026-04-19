//! Decode: Ogg Opus → 16-bit PCM WAV.

use std::fs::File;
use std::io::BufReader;

use anyhow::{Context, Result, anyhow};
use colored::*;

use ropus::{DecodeMode, Decoder as RopusDecoder};

use ogg::reading::PacketReader;

use crate::audio::wav::write_wav_pcm16;
use crate::consts::OPUS_SR;
use crate::container::ogg::{parse_opus_head, read_opus_tags};
use crate::options::DecodeOptions;
use crate::ui::{format_num, heading, ok};
use crate::util::{channel_count_to_ropus, with_extension};

pub fn decode(opts: DecodeOptions) -> Result<()> {
    heading("decode");
    println!("input    {}", opts.input.display().to_string().cyan());

    let output = opts
        .output
        .clone()
        .unwrap_or_else(|| with_extension(&opts.input, "wav"));
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

    // Tags packet: OpusTags. Verify magic to catch malformed files (e.g. a
    // stripped tags page would otherwise silently consume the first audio
    // packet).
    read_opus_tags(&mut reader).context("reading OpusTags packet")?;

    let opus_channels = channel_count_to_ropus(head.channels as usize)?;
    let mut decoder = RopusDecoder::new(OPUS_SR, opus_channels)
        .map_err(|e| anyhow!("decoder init failed: {e}"))?;

    // RFC 7845 §5.1: decoders MUST apply OpusHead.output_gain. ropus::Decoder
    // does it in fixed-point before the i16 clamp — preserves precision at low
    // volumes and is differential-tested against the C reference.
    if head.output_gain != 0 {
        decoder
            .set_gain(head.output_gain as i32)
            .map_err(|e| anyhow!("set_gain from OpusHead failed: {e}"))?;
    }

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
