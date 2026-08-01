//! Info: print stream info for an Opus file.
//!
//! Three output modes, selected by `InfoOptions`:
//!   1. Default: multi-line human-readable block mirroring `opusinfo`.
//!   2. `--extended` (-e): the default block plus a per-packet TOC decode and
//!      a per-gap list.
//!   3. `--query KEY` (-q): one named value, no banner, no decoration. Intended
//!      for shell pipelines; stricter than `--quiet --no-color`.

use std::fs::File;
use std::io::{BufReader, IsTerminal, Read, Seek};

use anyhow::{Context, Result, anyhow};
use colored::*;

use ropus::{DecodeMode, Decoder as RopusDecoder};

use ogg::reading::PacketReader;

use crate::consts::OPUS_SR;
use crate::container::ogg::{
    GranuleGap, OpusHead, OpusTags, detect_granule_gaps, parse_opus_head, read_last_granule,
    read_page_granules, validate_opus_audio_packet,
};
use crate::container::toc::decode_toc;
use crate::options::InfoOptions;
use crate::ui::{escape_terminal_path, escape_terminal_text, format_query_value, heading};
use crate::util::channel_count_to_ropus;

/// The small, fixed set of values exposed by `--query`.
///
/// Parsing this before opening the input is deliberate: an invalid query must
/// not turn into an input-file error, and a valid scalar query must not fall
/// through the human-summary collector.
#[derive(Debug, Clone, PartialEq, Eq)]
enum QueryKey {
    Channels,
    SampleRate,
    PreSkip,
    Gain,
    Duration,
    Bitrate,
    Vendor,
    Comment(String),
}

impl QueryKey {
    fn parse(raw: &str) -> Result<Self> {
        let lower = raw.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("comment:") {
            return Ok(Self::Comment(rest.to_owned()));
        }

        match lower.as_str() {
            "channels" => Ok(Self::Channels),
            "samplerate" => Ok(Self::SampleRate),
            "preskip" => Ok(Self::PreSkip),
            "gain" => Ok(Self::Gain),
            "duration" => Ok(Self::Duration),
            "bitrate" => Ok(Self::Bitrate),
            "vendor" => Ok(Self::Vendor),
            _ => Err(anyhow!("unknown query key: {raw}")),
        }
    }
}

/// Validate a query key without opening its input file.
///
/// The CLI uses this to retain opus-tools' exit code 2 for an unknown key,
/// while library callers receive a normal `anyhow::Error` from `info`.
pub fn validate_query_key(raw: &str) -> Result<()> {
    QueryKey::parse(raw).map(|_| ())
}

/// Parsed summary shared by the two human-readable output modes. Query mode
/// deliberately uses smaller plans below so scalar lookups do not assemble
/// this whole structure.
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
    // `--query` is a strict scripting mode: skip the heading, skip the banner,
    // skip any colored text. The main.rs caller already short-circuited the
    // banner when `opts.query.is_some()` (see ropusinfo/src/main.rs), so here
    // we just emit the bare value and return. Parse before opening the input:
    // an unknown key should not be masked by a missing or unreadable file.
    if let Some(raw_key) = &opts.query {
        let key = QueryKey::parse(raw_key)?;
        return collect_query(&opts.input, &key);
    }

    let summary = collect_summary(&opts.input, opts.extended)?;

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

fn collect_summary(input: &std::path::Path, retain_packets: bool) -> Result<InfoSummary> {
    let file =
        File::open(input).with_context(|| format!("opening {}", escape_terminal_path(input)))?;
    let file_len = file.metadata().ok().map(|m| m.len()).unwrap_or(0);
    let mut reader = PacketReader::new(BufReader::new(file));

    let head_pkt = reader.read_packet()?.ok_or_else(|| anyhow!("empty file"))?;
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
        .with_context(|| format!("opening {} for granule scan", escape_terminal_path(input)))?;
    let absgp_opt =
        read_last_granule(&mut fast_file, target_serial).context("scanning for last Ogg page")?;

    // Walk every data packet, capturing at most 2 bytes of each for TOC decode.
    // We still need to decode on the slow path to recover the true sample
    // count; the extended-mode cost is one buffer-of-TOC-bytes over the
    // existing loop, negligible compared to the decode work.
    let max_per_ch = (OPUS_SR / 1000 * 120) as usize;
    let mut decoded = vec![0i16; max_per_ch * opus_channels.count()];
    // Packet TOCs are only retained for `--extended`; default human output
    // needs neither the bytes nor their per-packet allocation.
    let mut packets = retain_packets.then(Vec::new);
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
        validate_opus_audio_packet(&pkt.data)
            .with_context(|| format!("validating Opus audio packet {packet_idx}"))?;
        let b0 = pkt.data.first().copied().unwrap_or(0);
        let b1 = pkt.data.get(1).copied();
        if let Some(tocs) = packets.as_mut() {
            tocs.push((b0, b1));
        }

        if let Some(dec) = decoder.as_mut() {
            match dec.decode(&pkt.data, &mut decoded, DecodeMode::Normal) {
                Ok(n) => slow_sample_count += n as u64,
                Err(e) => {
                    eprintln!(
                        "{} packet {}: {}",
                        "warning:".yellow(),
                        packet_idx,
                        escape_terminal_text(&e.to_string())
                    );
                }
            }
        }
        packet_idx += 1;
    }

    let pre_skip = head.pre_skip as u64;
    let sample_count = match absgp_opt {
        Some(absgp) => absgp
            .checked_sub(pre_skip)
            .ok_or_else(|| anyhow!("final granule {absgp} is before pre-skip {pre_skip}"))?,
        None => slow_sample_count.checked_sub(pre_skip).ok_or_else(|| {
            anyhow!("decoded sample count {slow_sample_count} is smaller than pre-skip {pre_skip}")
        })?,
    };

    // Separate pass for per-page granules: the `ogg` crate's PacketReader
    // coalesces packets across pages and doesn't expose per-page absgp, so
    // we re-open the file and walk the raw Ogg frames ourselves. Used only
    // for gap detection — cheap (a single sequential read).
    let mut gap_file = File::open(input).with_context(|| {
        format!(
            "opening {} for granule-gap scan",
            escape_terminal_path(input)
        )
    })?;
    let page_granules =
        read_page_granules(&mut gap_file, target_serial).context("scanning page granules")?;

    Ok(InfoSummary {
        head,
        tags,
        sample_count,
        file_len,
        packets: packets.unwrap_or_default(),
        page_granules,
    })
}

/// Emit the default multi-line block. Format intentionally mirrors
/// `opus-tools`' `opusinfo` so users scripting around grep-style parsers keep
/// their muscle memory; deviations are only where ropus simply doesn't have
/// the equivalent field.
fn print_default_block(input: &std::path::Path, s: &InfoSummary) {
    println!("Input File: {}", escape_terminal_path(input).cyan());
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
    println!(
        "Vendor: {}",
        escape_terminal_text(&s.tags.vendor).bright_white()
    );
    if s.tags.comments.is_empty() {
        println!("User comments: (none)");
    } else {
        println!("User comments:");
        for c in &s.tags.comments {
            // Two-space indent, bare `KEY=value` text — matches opusinfo and
            // keeps any grep/awk pipeline on the consumer side trivial.
            println!("  {}", escape_terminal_text(c));
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
            println!("  gap: page={}, from={}, to={}", g.page, g.from, g.to);
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

/// Read just the OpusHead packet and stream serial. Fixed scalar queries use
/// this path and return before OpusTags or any audio packet is read.
fn read_head(input: &std::path::Path) -> Result<(OpusHead, u32, u64)> {
    let file =
        File::open(input).with_context(|| format!("opening {}", escape_terminal_path(input)))?;
    let file_len = file.metadata().ok().map(|m| m.len()).unwrap_or(0);
    let (head, serial) = read_head_from(file)?;
    Ok((head, serial, file_len))
}

fn read_head_from<R: Read + Seek>(source: R) -> Result<(OpusHead, u32)> {
    let mut reader = PacketReader::new(BufReader::new(source));
    let head_pkt = reader.read_packet()?.ok_or_else(|| anyhow!("empty file"))?;
    let head = parse_opus_head(&head_pkt.data)?;
    Ok((head, head_pkt.stream_serial()))
}

/// Read OpusHead and OpusTags, but no audio packets. Tag queries use this
/// bounded packet plan; scalar queries never call it.
fn read_head_and_tags(input: &std::path::Path) -> Result<(OpusHead, OpusTags, u32, u64)> {
    let file =
        File::open(input).with_context(|| format!("opening {}", escape_terminal_path(input)))?;
    let file_len = file.metadata().ok().map(|m| m.len()).unwrap_or(0);
    let mut reader = PacketReader::new(BufReader::new(file));
    let head_pkt = reader.read_packet()?.ok_or_else(|| anyhow!("empty file"))?;
    let head = parse_opus_head(&head_pkt.data)?;
    let target_serial = head_pkt.stream_serial();
    let tags_pkt = reader
        .read_packet()?
        .ok_or_else(|| anyhow!("expected OpusTags packet, got end of stream"))?;
    let tags = OpusTags::parse(&tags_pkt.data).context("parsing OpusTags packet")?;
    Ok((head, tags, target_serial, file_len))
}

/// Derive the sample count for duration/bitrate without building a human
/// summary. The normal case reads only the bounded trailing Ogg window; the
/// decoder fallback is reserved for truncated streams whose EOS granule is
/// unknown.
fn query_sample_count(input: &std::path::Path, head: OpusHead, target_serial: u32) -> Result<u64> {
    let mut fast_file = File::open(input)
        .with_context(|| format!("opening {} for granule scan", escape_terminal_path(input)))?;
    let absgp_opt =
        read_last_granule(&mut fast_file, target_serial).context("scanning for last Ogg page")?;

    let sample_count = if let Some(absgp) = absgp_opt {
        absgp
            .checked_sub(head.pre_skip as u64)
            .ok_or_else(|| anyhow!("final granule {absgp} is before pre-skip {}", head.pre_skip))?
    } else {
        decode_sample_count(input, head)?
    };
    Ok(sample_count)
}

/// Slow duration fallback for truncated streams. This decodes packets in a
/// bounded-memory loop and does not retain their TOCs or tag strings.
fn decode_sample_count(input: &std::path::Path, head: OpusHead) -> Result<u64> {
    let file =
        File::open(input).with_context(|| format!("opening {}", escape_terminal_path(input)))?;
    let mut reader = PacketReader::new(BufReader::new(file));
    // Skip OpusHead and OpusTags; the first packet is validated against the
    // caller's head, while the tags payload is intentionally not parsed.
    reader.read_packet()?.ok_or_else(|| anyhow!("empty file"))?;
    reader
        .read_packet()?
        .ok_or_else(|| anyhow!("expected OpusTags packet, got end of stream"))?;

    let opus_channels = channel_count_to_ropus(head.channels as usize)?;
    let max_per_ch = (OPUS_SR / 1000 * 120) as usize;
    let mut decoded = vec![0i16; max_per_ch * opus_channels.count()];
    let mut sample_count = 0u64;
    let mut packet_idx = 0u64;
    let mut decoder = RopusDecoder::new(OPUS_SR, opus_channels)
        .map_err(|e| anyhow!("decoder init failed: {e}"))?;
    while let Some(pkt) = reader.read_packet()? {
        validate_opus_audio_packet(&pkt.data)
            .with_context(|| format!("validating Opus audio packet {packet_idx}"))?;
        match decoder.decode(&pkt.data, &mut decoded, DecodeMode::Normal) {
            Ok(n) => sample_count += n as u64,
            Err(e) => {
                eprintln!(
                    "{} packet {}: {}",
                    "warning:".yellow(),
                    packet_idx,
                    escape_terminal_text(&e.to_string())
                );
            }
        }
        packet_idx += 1;
    }
    sample_count
        .checked_sub(head.pre_skip as u64)
        .ok_or_else(|| {
            anyhow!(
                "decoded sample count {sample_count} is smaller than pre-skip {}",
                head.pre_skip
            )
        })
}

/// Execute a query-specific collection plan, then reuse the normal formatter.
fn collect_query(input: &std::path::Path, key: &QueryKey) -> Result<()> {
    match key {
        QueryKey::Channels | QueryKey::SampleRate | QueryKey::PreSkip | QueryKey::Gain => {
            let (head, _serial, file_len) = read_head(input)?;
            let summary = InfoSummary {
                head,
                tags: OpusTags::default(),
                sample_count: 0,
                file_len,
                packets: Vec::new(),
                page_granules: Vec::new(),
            };
            emit_query(&summary, key)
        }
        QueryKey::Vendor | QueryKey::Comment(_) => {
            let (head, tags, _serial, file_len) = read_head_and_tags(input)?;
            let summary = InfoSummary {
                head,
                tags,
                sample_count: 0,
                file_len,
                packets: Vec::new(),
                page_granules: Vec::new(),
            };
            emit_query(&summary, key)
        }
        QueryKey::Duration | QueryKey::Bitrate => {
            let (head, target_serial, file_len) = read_head(input)?;
            let sample_count = query_sample_count(input, head, target_serial)?;
            let summary = InfoSummary {
                head,
                tags: OpusTags::default(),
                sample_count,
                file_len,
                packets: Vec::new(),
                page_granules: Vec::new(),
            };
            emit_query(&summary, key)
        }
    }
}

/// Handle a validated `--query KEY`. Prints a bare value to stdout on success.
fn emit_query(s: &InfoSummary, key: &QueryKey) -> Result<()> {
    let stdout_is_tty = std::io::stdout().is_terminal();
    match key {
        QueryKey::Comment(rest) => {
            // Missing comment is not an error — empty stdout + exit 0 keeps the
            // caller's `if ropusinfo -q comment:artist x.opus | grep -q .; then …`
            // idiom working.
            if let Some(v) = s.tags.get(rest) {
                println!("{}", format_query_value(v, stdout_is_tty));
            } else {
                println!();
            }
        }
        QueryKey::Channels => println!("{}", s.head.channels),
        QueryKey::SampleRate => println!("{}", s.head.input_sample_rate),
        QueryKey::PreSkip => println!("{}", s.head.pre_skip),
        QueryKey::Gain => {
            // Q8 → float dB, same formatter as the default block — but without
            // the " dB" suffix so scripts can feed it straight into bc/awk.
            println!("{:.1}", s.head.output_gain as f32 / 256.0);
        }
        QueryKey::Duration => {
            // Six decimal places is enough for sub-microsecond precision at
            // 48 kHz and matches the resolution of the sample_count we derive
            // it from.
            println!("{:.6}", s.duration_s());
        }
        QueryKey::Bitrate => {
            // Integer bps, rounded. avg_kbps() returns kb/s as f64; multiply
            // and round to get an integer bps value the user can feed into a
            // `< 128000` kind of test.
            let bps = (s.avg_kbps() * 1000.0).round() as u64;
            println!("{bps}");
        }
        QueryKey::Vendor => println!("{}", format_query_value(&s.tags.vendor, stdout_is_tty)),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, Cursor};

    use ogg::writing::{PacketWriteEndInfo, PacketWriter};

    struct BoundedReader {
        inner: Cursor<Vec<u8>>,
        limit: usize,
    }

    impl Read for BoundedReader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let pos = self.inner.position() as usize;
            if pos >= self.limit {
                return Err(io::Error::other("bounded-reader limit exceeded"));
            }
            let remaining = self.limit - pos;
            let read_len = buf.len().min(remaining);
            self.inner.read(&mut buf[..read_len])
        }
    }

    impl Seek for BoundedReader {
        fn seek(&mut self, position: io::SeekFrom) -> io::Result<u64> {
            self.inner.seek(position)
        }
    }

    fn stream_with_large_tags() -> (Vec<u8>, usize) {
        let serial = 0xC0DE_C0DE;
        let head = [
            b'O', b'p', b'u', b's', b'H', b'e', b'a', b'd', 1, 1, 0, 0, 0x80, 0xbb, 0, 0, 0, 0, 0,
        ];
        let tags = OpusTags {
            vendor: "vendor".to_owned(),
            comments: vec![format!("COMMENT={}", "x".repeat(128 * 1024))],
        };
        let mut output = Cursor::new(Vec::new());
        {
            let mut writer = PacketWriter::new(&mut output);
            writer
                .write_packet(&head[..], serial, PacketWriteEndInfo::EndPage, 0)
                .expect("write head");
            writer
                .write_packet(tags.encode(), serial, PacketWriteEndInfo::EndPage, 0)
                .expect("write tags");
            writer
                .write_packet(&[0u8], serial, PacketWriteEndInfo::EndStream, 960)
                .expect("write data");
        }
        // The first page consists of a 27-byte header, one lacing byte, and
        // the 19-byte OpusHead packet. A scalar query may stop at this bound;
        // attempting to read the large OpusTags page is a regression.
        (output.into_inner(), 27 + 1 + head.len())
    }

    #[test]
    fn query_key_is_validated_without_opening_input() {
        let error = validate_query_key("gargle").expect_err("unknown key must fail");
        assert!(error.to_string().contains("unknown query key"));

        let error = info(InfoOptions {
            input: std::path::PathBuf::from("definitely-missing.opus"),
            extended: false,
            query: Some("gargle".to_owned()),
        })
        .expect_err("unknown key must win before file open");
        assert!(error.to_string().contains("unknown query key"));
    }

    #[test]
    fn fixed_header_plan_stops_before_large_tags_packet() {
        let (bytes, first_page_len) = stream_with_large_tags();
        let mut reader = BoundedReader {
            inner: Cursor::new(bytes),
            limit: first_page_len,
        };
        let (head, serial) = read_head_from(&mut reader).expect("head fits in bound");
        assert_eq!(head.channels, 1);
        assert_eq!(serial, 0xC0DE_C0DE);
        assert_eq!(reader.inner.position() as usize, first_page_len);
    }
}
