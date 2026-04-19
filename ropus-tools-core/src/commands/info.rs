//! Info: print stream info for an Opus file.
//!
//! Three output modes, selected by `InfoOptions`:
//!   1. Default: multi-line human-readable block mirroring `opusinfo`.
//!   2. `--extended` (-e): the default block plus a per-packet TOC decode and
//!      a per-gap list.
//!   3. `--query KEY` (-q): one named value, no banner, no decoration. Intended
//!      for shell pipelines; stricter than `--quiet --no-color`.

use std::fs::File;
use std::io::BufReader;

use anyhow::{Context, Result, anyhow};
use colored::*;

use ropus::{DecodeMode, Decoder as RopusDecoder};

use ogg::reading::PacketReader;

use crate::consts::OPUS_SR;
use crate::container::ogg::{
    GranuleGap, OpusHead, OpusTags, detect_granule_gaps, parse_opus_head, read_last_granule,
    read_page_granules,
};
use crate::container::toc::decode_toc;
use crate::options::InfoOptions;
use crate::ui::heading;
use crate::util::channel_count_to_ropus;

/// Parsed summary assembled once and consumed by every output mode. Having all
/// three modes compute from the same struct guarantees the default block, the
/// extended output, and `--query` never drift in what they report.
struct InfoSummary {
    head: OpusHead,
    tags: OpusTags,
    /// Per-channel decoded sample count, post pre-skip trim, at 48 kHz.
    sample_count: u64,
    /// Total file size in bytes. Zero if the metadata call failed (rare, on
    /// stdin or unusual filesystems).
    file_len: u64,
    /// TOC byte 0/1 for each data packet in file order. Byte 1 is `None` when
    /// the packet has fewer than 2 bytes.
    packets: Vec<(u8, Option<u8>)>,
    /// Per-page granule positions for the target stream, sentinel-filtered.
    /// Used for gap detection; not emitted directly.
    page_granules: Vec<u64>,
}

impl InfoSummary {
    fn duration_s(&self) -> f64 {
        self.sample_count as f64 / OPUS_SR as f64
    }

    fn avg_kbps(&self) -> f64 {
        let d = self.duration_s();
        if d > 0.0 {
            (self.file_len as f64 * 8.0) / d / 1000.0
        } else {
            0.0
        }
    }

    fn gaps(&self) -> Vec<GranuleGap> {
        detect_granule_gaps(&self.page_granules)
    }
}

pub fn info(opts: InfoOptions) -> Result<()> {
    let summary = collect_summary(&opts.input)?;

    // `--query` is a strict scripting mode: skip the heading, skip the banner,
    // skip any colored text. The main.rs caller already short-circuited the
    // banner when `opts.query.is_some()` (see ropusinfo/src/main.rs), so here
    // we just emit the bare value and return.
    if let Some(key) = &opts.query {
        return emit_query(&summary, key);
    }

    heading("info");
    print_default_block(&opts.input, &summary);

    if opts.extended {
        print_extended(&summary);
    } else {
        // In default mode, still warn about granule gaps — they indicate
        // truncation or muxer bugs and the user probably wants to know.
        let gaps = summary.gaps();
        if !gaps.is_empty() {
            println!(
                "{} {} granule gap(s) detected",
                "WARN:".yellow().bold(),
                gaps.len()
            );
        }
    }

    Ok(())
}

fn collect_summary(input: &std::path::Path) -> Result<InfoSummary> {
    let file =
        File::open(input).with_context(|| format!("opening {}", input.display()))?;
    let file_len = file.metadata().ok().map(|m| m.len()).unwrap_or(0);
    let mut reader = PacketReader::new(BufReader::new(file));

    let head_pkt = reader
        .read_packet()?
        .ok_or_else(|| anyhow!("empty file"))?;
    let head = parse_opus_head(&head_pkt.data)?;
    // Capture the OpusHead's stream serial — this identifies the logical Opus
    // bitstream we care about in a multiplexed Ogg file.
    let target_serial = head_pkt.stream_serial();

    let tags_pkt = reader
        .read_packet()?
        .ok_or_else(|| anyhow!("expected OpusTags packet, got end of stream"))?;
    let tags = OpusTags::parse(&tags_pkt.data).context("parsing OpusTags packet")?;

    let opus_channels = channel_count_to_ropus(head.channels as usize)?;

    // Fast path: last-page granule position. Slow path only kicks in when the
    // last page has the unknown-granule sentinel (truncated files).
    let mut fast_file = File::open(input)
        .with_context(|| format!("opening {} for granule scan", input.display()))?;
    let absgp_opt =
        read_last_granule(&mut fast_file, target_serial).context("scanning for last Ogg page")?;

    // Walk every data packet, capturing at most 2 bytes of each for TOC decode.
    // We still need to decode on the slow path to recover the true sample
    // count; the extended-mode cost is one buffer-of-TOC-bytes over the
    // existing loop, negligible compared to the decode work.
    let max_per_ch = (OPUS_SR / 1000 * 120) as usize;
    let mut decoded = vec![0i16; max_per_ch * opus_channels.count()];
    let mut packets: Vec<(u8, Option<u8>)> = Vec::new();
    let mut slow_sample_count: u64 = 0;
    // We lazily spin up the decoder only when the slow path needs it; on a
    // healthy file (absgp known) we walk packets purely for their TOC bytes.
    let need_slow = absgp_opt.is_none();
    let mut decoder = if need_slow {
        Some(
            RopusDecoder::new(OPUS_SR, opus_channels)
                .map_err(|e| anyhow!("decoder init failed: {e}"))?,
        )
    } else {
        None
    };

    let mut packet_idx: u64 = 0;
    while let Some(pkt) = reader.read_packet()? {
        let b0 = pkt.data.first().copied().unwrap_or(0);
        let b1 = pkt.data.get(1).copied();
        packets.push((b0, b1));

        if let Some(dec) = decoder.as_mut() {
            match dec.decode(&pkt.data, &mut decoded, DecodeMode::Normal) {
                Ok(n) => slow_sample_count += n as u64,
                Err(e) => {
                    eprintln!("{} packet {}: {e}", "warning:".yellow(), packet_idx);
                }
            }
        }
        packet_idx += 1;
    }

    let sample_count = match absgp_opt {
        Some(absgp) => absgp.saturating_sub(head.pre_skip as u64),
        None => slow_sample_count.saturating_sub(head.pre_skip as u64),
    };

    // Separate pass for per-page granules: the `ogg` crate's PacketReader
    // coalesces packets across pages and doesn't expose per-page absgp, so
    // we re-open the file and walk the raw Ogg frames ourselves. Used only
    // for gap detection — cheap (a single sequential read).
    let mut gap_file = File::open(input)
        .with_context(|| format!("opening {} for granule-gap scan", input.display()))?;
    let page_granules = read_page_granules(&mut gap_file, target_serial)
        .context("scanning page granules")?;

    Ok(InfoSummary {
        head,
        tags,
        sample_count,
        file_len,
        packets,
        page_granules,
    })
}

/// Emit the default multi-line block. Format intentionally mirrors
/// `opus-tools`' `opusinfo` so users scripting around grep-style parsers keep
/// their muscle memory; deviations are only where ropus simply doesn't have
/// the equivalent field.
fn print_default_block(input: &std::path::Path, s: &InfoSummary) {
    println!(
        "Input File: {}",
        input.display().to_string().cyan()
    );
    println!("Channels: {}", s.head.channels.to_string().bright_white());
    println!(
        "Sample rate (input): {} Hz",
        s.head.input_sample_rate.to_string().bright_white()
    );
    println!("Pre-skip: {}", s.head.pre_skip.to_string().bright_white());
    println!("Output gain: {}", format_output_gain(s.head.output_gain));
    println!(
        "Channel mapping family: {}",
        s.head.channel_mapping.to_string().bright_white()
    );
    println!("Vendor: {}", s.tags.vendor.bright_white());
    if s.tags.comments.is_empty() {
        println!("User comments: (none)");
    } else {
        println!("User comments:");
        for c in &s.tags.comments {
            // Two-space indent, bare `KEY=value` text — matches opusinfo and
            // keeps any grep/awk pipeline on the consumer side trivial.
            println!("  {c}");
        }
    }
    // Raw digits (no thousands commas) for byte-count fields — the HLD
    // example writes `Total data length: 42312 bytes`, and scripts diffing
    // against opusinfo output rely on the unformatted integer.
    println!(
        "Total data length: {} bytes",
        s.file_len.to_string().bright_white()
    );
    println!(
        "Playback length: {}",
        format_playback_length(s.duration_s()).bright_white()
    );
    println!(
        "Average bitrate: {} kb/s",
        format!("{:.1}", s.avg_kbps()).bright_white()
    );
}

fn print_extended(s: &InfoSummary) {
    println!("Packets:");
    for (i, &(b0, b1_opt)) in s.packets.iter().enumerate() {
        let mut bytes = vec![b0];
        if let Some(b1) = b1_opt {
            bytes.push(b1);
        }
        let toc = match decode_toc(&bytes) {
            Some(t) => t,
            None => continue, // empty packet; decode_toc only returns None for 0-byte input
        };
        let ch = if toc.stereo { 2 } else { 1 };
        let frames_str = toc
            .frames
            .map(|n| n.to_string())
            .unwrap_or_else(|| "?".to_string());
        // Per-frame duration * frame count = packet duration. Use integer
        // arithmetic on cms so we avoid float-format drift; print the sum as a
        // trimmed ms value when the total is a whole ms.
        let total_cms = (toc.frame_size_cms as u64) * toc.frames.unwrap_or(1) as u64;
        let dur_str = if total_cms.is_multiple_of(100) {
            format!("{}ms", total_cms / 100)
        } else {
            format!("{}.{}ms", total_cms / 100, (total_cms % 100) / 10)
        };
        println!(
            "  #{:04}: TOC=0x{:02X} mode={} bw={} ch={} frames={} dur={}",
            i,
            b0,
            toc.mode.label(toc.bandwidth),
            toc.bandwidth.label(),
            ch,
            frames_str,
            dur_str,
        );
    }

    let gaps = s.gaps();
    if gaps.is_empty() {
        println!("Gaps: none");
    } else {
        println!("Gaps:");
        for g in gaps {
            println!(
                "  gap: page={}, from={}, to={}",
                g.page, g.from, g.to
            );
        }
    }
}

/// Convert the Q8-dB `output_gain` i16 from OpusHead to a human string.
/// Always `X.Y dB` with one decimal place so the default block and the
/// `--query gain` value share the same representation for zero (both emit
/// `0.0`). Scripts diffing the two paths get identical output.
fn format_output_gain(gain_q8: i16) -> String {
    format!("{:.1} dB", gain_q8 as f32 / 256.0)
}

/// Format the playback length as `Xm Y.YYs` (minutes + seconds), adding an
/// `Hh` prefix for files over one hour. Minutes are omitted for sub-minute
/// files. Matches opusinfo's display shape.
fn format_playback_length(seconds: f64) -> String {
    let total_secs = seconds;
    let hours = (total_secs / 3600.0).floor() as u64;
    let after_hours = total_secs - (hours as f64) * 3600.0;
    let minutes = (after_hours / 60.0).floor() as u64;
    let secs = after_hours - (minutes as f64) * 60.0;
    if hours > 0 {
        format!("{}h {}m {:.2}s", hours, minutes, secs)
    } else if minutes > 0 {
        format!("{}m {:.2}s", minutes, secs)
    } else {
        format!("{:.2}s", secs)
    }
}

/// Handle `--query KEY`. Prints a bare value to stdout on success; writes to
/// stderr and returns an error for unknown keys (the caller exits with 2 via
/// `prelude::run`, preserving our uniform error formatting).
fn emit_query(s: &InfoSummary, key: &str) -> Result<()> {
    let lower = key.to_ascii_lowercase();
    // Special-case `comment:KEY` up front: the colon suffix is variable, so a
    // match arm that binds the rest is cleaner than a giant lookup table.
    if let Some(rest) = lower.strip_prefix("comment:") {
        // Missing comment is not an error — empty stdout + exit 0 keeps the
        // caller's `if ropusinfo -q comment:artist x.opus | grep -q .; then …`
        // idiom working.
        if let Some(v) = s.tags.get(rest) {
            println!("{v}");
        } else {
            println!();
        }
        return Ok(());
    }

    match lower.as_str() {
        "channels" => println!("{}", s.head.channels),
        "samplerate" => println!("{}", s.head.input_sample_rate),
        "preskip" => println!("{}", s.head.pre_skip),
        "gain" => {
            // Q8 → float dB, same formatter as the default block — but without
            // the " dB" suffix so scripts can feed it straight into bc/awk.
            println!("{:.1}", s.head.output_gain as f32 / 256.0);
        }
        "duration" => {
            // Six decimal places is enough for sub-microsecond precision at
            // 48 kHz and matches the resolution of the sample_count we derive
            // it from.
            println!("{:.6}", s.duration_s());
        }
        "bitrate" => {
            // Integer bps, rounded. avg_kbps() returns kb/s as f64; multiply
            // and round to get an integer bps value the user can feed into a
            // `< 128000` kind of test.
            let bps = (s.avg_kbps() * 1000.0).round() as u64;
            println!("{bps}");
        }
        "vendor" => println!("{}", s.tags.vendor),
        _ => {
            // `prelude::run` prepends `error:` and the anyhow chain; instead
            // of that shape we want the exact opus-tools-style message and
            // exit code 2 (the shell reserves 1 for generic failure).
            // Print directly, then bail to bubble up ExitCode::FAILURE — and
            // note the deviation: our `prelude::run` maps errors to
            // ExitCode::FAILURE (1), not 2. The HLD says exit 2 for unknown
            // keys; to keep the mapping clean, emit the error message here
            // and use `std::process::exit(2)` so we don't depend on the
            // prelude's exit-code behaviour for this one case.
            eprintln!("ropusinfo: unknown query key: {key}");
            std::process::exit(2);
        }
    }
    Ok(())
}

