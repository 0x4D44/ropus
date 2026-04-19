//! Info: print stream info for an Opus file.

use std::fs::File;
use std::io::BufReader;

use anyhow::{Context, Result, anyhow};
use colored::*;

use ropus::{DecodeMode, Decoder as RopusDecoder};

use ogg::reading::PacketReader;

use crate::consts::OPUS_SR;
use crate::container::ogg::{OpusTags, parse_opus_head, read_last_granule};
use crate::options::InfoOptions;
use crate::ui::{format_num, heading};
use crate::util::channel_count_to_ropus;

pub fn info(opts: InfoOptions) -> Result<()> {
    heading("info");
    println!("file     {}", opts.input.display().to_string().cyan());

    let file = File::open(&opts.input)
        .with_context(|| format!("opening {}", opts.input.display()))?;
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
    // Parse the OpusTags packet so malformed files fail here rather than
    // later. Step 5 of the opus-tools-parity HLD adds a pretty display of
    // vendor + comments; for now we just consume the packet and discard the
    // parsed struct (dropping the `_tags` binding).
    let tags_pkt = reader
        .read_packet()?
        .ok_or_else(|| anyhow!("expected OpusTags packet, got end of stream"))?;
    let _tags = OpusTags::parse(&tags_pkt.data).context("parsing OpusTags packet")?;

    let opus_channels = channel_count_to_ropus(head.channels as usize)?;

    // Fast path: locate the last Ogg page belonging to our serial and read
    // its absolute granule position. Per RFC 7845 §4 that is the total stream
    // length in 48 kHz samples, including pre_skip warm-up.
    let mut fast_file = File::open(&opts.input)
        .with_context(|| format!("opening {} for granule scan", opts.input.display()))?;
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
