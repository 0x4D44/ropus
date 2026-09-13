//! Info: print stream info for an Opus file.
//!
//! Three output modes, selected by `InfoOptions`:
//!   1. Default: multi-line human-readable block mirroring `opusinfo`.
//!   2. `--extended` (-e): the default block plus a per-packet TOC decode and
//!      a per-gap list.
//!   3. `--query KEY` (-q): one named value, no banner, no decoration. Intended
//!      for shell pipelines; stricter than `--quiet --no-color`.

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{BufReader, IsTerminal, Read, Seek};

use anyhow::{Context, Result, anyhow};
use colored::*;

use ropus::{DecodeMode, Decoder as RopusDecoder};

use crate::consts::{MAX_PACKET_BYTES, OPUS_SR};
use crate::container::ogg::{
    GranuleGap, OpusHead, OpusTags, detect_granule_gaps, parse_opus_head, read_last_granule,
    read_page_granules, validate_opus_audio_packet, validate_opus_header_stream,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SampleCount {
    Exact(u64),
    Estimate { samples: u64, error_count: u64 },
}

impl SampleCount {
    fn value(self) -> u64 {
        match self {
            Self::Exact(samples) | Self::Estimate { samples, .. } => samples,
        }
    }

    fn annotation(self) -> Option<String> {
        match self {
            Self::Exact(_) => None,
            Self::Estimate { error_count, .. } => {
                Some(format!("estimate; {error_count} packet error(s) skipped"))
            }
        }
    }

    fn from_value(samples: u64, error_count: u64) -> Self {
        if error_count == 0 {
            Self::Exact(samples)
        } else {
            Self::Estimate {
                samples,
                error_count,
            }
        }
    }
}

/// Maximum packet sizes accepted by the info reader.
///
/// OpusHead is fixed-size for the channel mapping family supported by this
/// crate. Tags are deliberately finite because they are user-controlled text;
/// audio packets are bounded by the largest packet our encoder can produce.
const MAX_OPUS_HEAD_PACKET_BYTES: usize = 19;
const MAX_OPUS_TAGS_PACKET_BYTES: usize = 1024 * 1024;
const MAX_HEADER_PENDING_BYTES: usize = MAX_OPUS_TAGS_PACKET_BYTES;

#[derive(Debug)]
struct InfoPacket {
    data: Vec<u8>,
    stream_serial: u32,
}

impl InfoPacket {
    fn stream_serial(&self) -> u32 {
        self.stream_serial
    }
}

/// Page-streamed Ogg packet reader for info plans.
///
/// `ogg::reading::PacketReader` keeps every continued packet in a growing
/// overlap cache before a caller can validate it. This reader parses one
/// physical page at a time, discards unrelated logical streams, and checks the
/// selected packet's role-specific budget before copying a lace into it.
struct BoundedPacketReader<R: Read> {
    source: R,
    target_serial: Option<u32>,
    pending: Option<Vec<u8>>,
    other_pending: HashMap<u32, Vec<u8>>,
    packet_count: u64,
    ready: VecDeque<InfoPacket>,
}

impl<R: Read> BoundedPacketReader<R> {
    fn new(source: R) -> Self {
        Self {
            source,
            target_serial: None,
            pending: None,
            other_pending: HashMap::new(),
            packet_count: 0,
            ready: VecDeque::new(),
        }
    }

    fn for_serial(source: R, target_serial: u32) -> Self {
        Self {
            target_serial: Some(target_serial),
            ..Self::new(source)
        }
    }

    fn packet_limit(packet_count: u64) -> usize {
        match packet_count {
            0 => MAX_OPUS_HEAD_PACKET_BYTES,
            1 => MAX_OPUS_TAGS_PACKET_BYTES,
            _ => MAX_PACKET_BYTES,
        }
    }

    fn read_packet(&mut self) -> Result<Option<InfoPacket>> {
        loop {
            if let Some(packet) = self.ready.pop_front() {
                return Ok(Some(packet));
            }

            let Some(page) = self.read_page()? else {
                if self.pending.is_some() {
                    return Err(anyhow!("truncated continued Ogg packet"));
                }
                return Ok(None);
            };

            let serial = page.serial;
            if self.target_serial.is_none() {
                // As with PacketReader, the first packet in a normal Opus file
                // identifies the stream. Selecting the first page serial also
                // avoids retaining arbitrary pre-header multiplexed streams.
                self.target_serial = Some(serial);
            }
            if self.target_serial == Some(serial) {
                self.process_page(serial, page.header_type, &page.lacing, &page.body)?;
            }
        }
    }

    /// Read the next packet while the header pair is being established. The
    /// second packet must be surfaced even when a malformed/multiplexed file
    /// puts it on another logical stream, so callers can reject that pair.
    fn read_header_packet(&mut self) -> Result<Option<InfoPacket>> {
        loop {
            if let Some(packet) = self.ready.pop_front() {
                return Ok(Some(packet));
            }
            let Some(page) = self.read_page()? else {
                return Ok(None);
            };
            if self.target_serial == Some(page.serial) {
                self.process_page(page.serial, page.header_type, &page.lacing, &page.body)?;
            } else if self.packet_count == 1 {
                self.process_other_header_page(&page)?;
            }
        }
    }

    fn process_other_header_page(&mut self, page: &InfoPage) -> Result<()> {
        let continued = page.header_type & 0x01 != 0;
        if continued != self.other_pending.contains_key(&page.serial) {
            return Err(anyhow!("invalid Ogg continued-packet flag"));
        }
        let pending_len = self.other_pending.get(&page.serial).map_or(0, Vec::len);
        Self::check_lacing_budget(1, pending_len, &page.lacing)?;
        let other_pending_bytes = self.other_pending.values().map(Vec::len).sum::<usize>();
        let mut final_pending_len = pending_len;
        for &lace in &page.lacing {
            final_pending_len = final_pending_len
                .checked_add(lace as usize)
                .ok_or_else(|| anyhow!("header continuation size overflow"))?;
            if lace < 255 {
                final_pending_len = 0;
            }
        }
        let total_after_page = other_pending_bytes
            .saturating_sub(pending_len)
            .checked_add(final_pending_len)
            .ok_or_else(|| anyhow!("header continuation size overflow"))?;
        if total_after_page > MAX_HEADER_PENDING_BYTES {
            return Err(anyhow!("header continuation memory budget exceeded"));
        }

        let mut offset = 0usize;
        if !continued && !page.lacing.is_empty() {
            self.other_pending.insert(page.serial, Vec::new());
        }
        for (index, &lace) in page.lacing.iter().enumerate() {
            let lace_len = lace as usize;
            let packet = self
                .other_pending
                .get_mut(&page.serial)
                .ok_or_else(|| anyhow!("missing Ogg packet continuation"))?;
            packet.extend_from_slice(&page.body[offset..offset + lace_len]);
            offset += lace_len;
            if lace < 255 {
                let data = self
                    .other_pending
                    .remove(&page.serial)
                    .expect("pending packet exists");
                self.ready.push_back(InfoPacket {
                    data,
                    stream_serial: page.serial,
                });
                if index + 1 < page.lacing.len() {
                    self.other_pending.insert(page.serial, Vec::new());
                }
            }
        }
        Ok(())
    }

    fn process_page(
        &mut self,
        serial: u32,
        header_type: u8,
        lacing: &[u8],
        body: &[u8],
    ) -> Result<()> {
        let continued = header_type & 0x01 != 0;
        if continued != self.pending.is_some() {
            return Err(anyhow!("invalid Ogg continued-packet flag"));
        }
        self.check_page_budget(lacing)?;

        let mut offset = 0usize;
        let mut lace_index = 0usize;
        if !continued && !lacing.is_empty() {
            self.pending = Some(Vec::new());
        }
        while lace_index < lacing.len() {
            let lace_len = lacing[lace_index] as usize;
            self.append_pending(&body[offset..offset + lace_len])?;
            offset += lace_len;
            lace_index += 1;
            if lace_len < 255 {
                self.finish_pending(serial);
                if lace_index < lacing.len() {
                    self.pending = Some(Vec::new());
                }
            }
        }
        debug_assert_eq!(offset, body.len());
        Ok(())
    }

    fn check_page_budget(&self, lacing: &[u8]) -> Result<()> {
        let packet_len = self.pending.as_ref().map_or(0, Vec::len);
        Self::check_lacing_budget(self.packet_count, packet_len, lacing)
    }

    fn check_lacing_budget(
        mut packet_count: u64,
        mut packet_len: usize,
        lacing: &[u8],
    ) -> Result<()> {
        for &lace in lacing {
            let lace_len = lace as usize;
            let limit = Self::packet_limit(packet_count);
            packet_len = packet_len
                .checked_add(lace_len)
                .ok_or_else(|| anyhow!("Ogg packet length overflow"))?;
            if packet_len > limit {
                return Err(anyhow!(
                    "Ogg packet {} exceeds info limit of {} bytes",
                    packet_count,
                    limit
                ));
            }
            if lace < 255 {
                packet_count += 1;
                packet_len = 0;
            }
        }
        Ok(())
    }

    fn append_pending(&mut self, bytes: &[u8]) -> Result<()> {
        let limit = Self::packet_limit(self.packet_count);
        let pending = self
            .pending
            .as_mut()
            .ok_or_else(|| anyhow!("missing Ogg packet continuation"))?;
        if bytes.len() > limit.saturating_sub(pending.len()) {
            return Err(anyhow!(
                "Ogg packet {} exceeds info limit of {} bytes",
                self.packet_count,
                limit
            ));
        }
        pending.extend_from_slice(bytes);
        Ok(())
    }

    fn finish_pending(&mut self, serial: u32) {
        let data = self
            .pending
            .take()
            .expect("finished packet must be pending");
        self.ready.push_back(InfoPacket {
            data,
            stream_serial: serial,
        });
        self.packet_count += 1;
    }

    fn read_page(&mut self) -> Result<Option<InfoPage>> {
        let Some(header) = self.read_header()? else {
            return Ok(None);
        };
        if header[4] != 0 {
            return Err(anyhow!(
                "unsupported Ogg stream structure version {}",
                header[4]
            ));
        }

        let segment_count = header[26] as usize;
        let mut lacing = vec![0u8; segment_count];
        self.source
            .read_exact(&mut lacing)
            .context("reading Ogg lacing table")?;
        let serial = u32::from_le_bytes([header[14], header[15], header[16], header[17]]);
        if self.target_serial.is_none() {
            self.target_serial = Some(serial);
        }
        if self.target_serial == Some(serial) {
            let continued = header[5] & 0x01 != 0;
            if continued != self.pending.is_some() {
                return Err(anyhow!("invalid Ogg continued-packet flag"));
            }
            // The lacing table gives us the exact bytes that would be added
            // to each packet, so reject before allocating the page body.
            self.check_page_budget(&lacing)?;
        }
        let body_len = lacing.iter().map(|&lace| lace as usize).sum::<usize>();
        let mut body = vec![0u8; body_len];
        self.source
            .read_exact(&mut body)
            .context("reading Ogg page body")?;

        let mut page = Vec::with_capacity(27 + segment_count + body_len);
        page.extend_from_slice(&header);
        page.extend_from_slice(&lacing);
        page.extend_from_slice(&body);
        let expected = u32::from_le_bytes([header[22], header[23], header[24], header[25]]);
        if ogg_crc32(&page) != expected {
            return Err(anyhow!("Ogg page checksum mismatch"));
        }

        Ok(Some(InfoPage {
            serial,
            header_type: header[5],
            lacing,
            body,
        }))
    }

    fn read_header(&mut self) -> Result<Option<[u8; 27]>> {
        let mut capture = [0u8; 4];
        let mut filled = 0usize;
        loop {
            let n = self.source.read(&mut capture[filled..filled + 1])?;
            if n == 0 {
                if filled == 0 {
                    return Ok(None);
                }
                return Err(anyhow!("truncated Ogg capture pattern"));
            }
            filled += 1;
            if filled == capture.len() {
                if &capture == b"OggS" {
                    let mut header = [0u8; 27];
                    header[..4].copy_from_slice(&capture);
                    self.source.read_exact(&mut header[4..])?;
                    return Ok(Some(header));
                }
                capture.copy_within(1.., 0);
                filled = 3;
            }
        }
    }
}

struct InfoPage {
    serial: u32,
    header_type: u8,
    lacing: Vec<u8>,
    body: Vec<u8>,
}

fn ogg_crc32(page: &[u8]) -> u32 {
    const POLY: u32 = 0x04c1_1db7;
    let mut crc = 0u32;
    for (index, &byte) in page.iter().enumerate() {
        let byte = if (22..26).contains(&index) { 0 } else { byte };
        crc ^= u32::from(byte) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 {
                (crc << 1) ^ POLY
            } else {
                crc << 1
            };
        }
    }
    crc
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
    /// Human output may carry an explicitly labelled estimate when fallback
    /// decoding skipped malformed packets.
    sample_count: SampleCount,
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
        self.sample_count.value() as f64 / OPUS_SR as f64
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
    let mut reader = BoundedPacketReader::new(BufReader::new(file));

    let head_pkt = reader.read_packet()?.ok_or_else(|| anyhow!("empty file"))?;
    let head = parse_opus_head(&head_pkt.data)?;
    // Capture the OpusHead's stream serial — this identifies the logical Opus
    // bitstream we care about in a multiplexed Ogg file.
    let target_serial = head_pkt.stream_serial();

    let tags_pkt = reader
        .read_header_packet()?
        .ok_or_else(|| anyhow!("expected OpusTags packet, got end of stream"))?;
    validate_opus_header_stream(target_serial, tags_pkt.stream_serial())?;
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
    let mut packet_errors: u64 = 0;
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
        if let Err(error) = validate_opus_audio_packet(&pkt.data) {
            if pkt.data.is_empty() || retain_packets {
                return Err(error)
                    .with_context(|| format!("validating Opus audio packet {packet_idx}"));
            }

            // Default human output may retain a labelled estimate when a
            // malformed nonempty packet cannot contribute decoded samples.
            // Extended output and strict queries take the error path above.
            packet_errors += 1;
            eprintln!(
                "{} packet {}: {}",
                "warning:".yellow(),
                packet_idx,
                escape_terminal_text(&error.to_string())
            );
            packet_idx += 1;
            continue;
        }
        let b0 = pkt.data.first().copied().unwrap_or(0);
        let b1 = pkt.data.get(1).copied();
        if let Some(tocs) = packets.as_mut() {
            tocs.push((b0, b1));
        }

        if let Some(dec) = decoder.as_mut() {
            match dec.decode(&pkt.data, &mut decoded, DecodeMode::Normal) {
                Ok(n) => slow_sample_count += n as u64,
                Err(e) => {
                    packet_errors += 1;
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
        Some(absgp) => SampleCount::from_value(
            absgp
                .checked_sub(pre_skip)
                .ok_or_else(|| anyhow!("final granule {absgp} is before pre-skip {pre_skip}"))?,
            packet_errors,
        ),
        None => SampleCount::from_value(
            slow_sample_count.checked_sub(pre_skip).ok_or_else(|| {
                anyhow!(
                    "decoded sample count {slow_sample_count} is smaller than pre-skip {pre_skip}"
                )
            })?,
            packet_errors,
        ),
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
    let duration = format_playback_length(s.duration_s()).bright_white();
    let bitrate = format!("{:.1}", s.avg_kbps()).bright_white();
    if let Some(annotation) = s.sample_count.annotation() {
        println!("Playback length: {duration} ({annotation})");
        println!("Average bitrate: {bitrate} kb/s ({annotation})");
    } else {
        println!("Playback length: {duration}");
        println!("Average bitrate: {bitrate} kb/s");
    }
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
        // trimmed ms value when the total is a whole ms. Keep the duration
        // unknown if a future TOC source cannot provide its frame count.
        let dur_str = match toc.frames {
            Some(frames) => {
                let total_cms = (toc.frame_size_cms as u64) * u64::from(frames);
                if total_cms.is_multiple_of(100) {
                    format!("{}ms", total_cms / 100)
                } else {
                    format!("{}.{}ms", total_cms / 100, (total_cms % 100) / 10)
                }
            }
            None => "?".to_string(),
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
    let total_secs = (seconds * 100.0).round() / 100.0;
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
    let mut reader = BoundedPacketReader::new(BufReader::new(source));
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
    let mut reader = BoundedPacketReader::new(BufReader::new(file));
    let head_pkt = reader.read_packet()?.ok_or_else(|| anyhow!("empty file"))?;
    let head = parse_opus_head(&head_pkt.data)?;
    let target_serial = head_pkt.stream_serial();
    let tags_pkt = reader
        .read_header_packet()?
        .ok_or_else(|| anyhow!("expected OpusTags packet, got end of stream"))?;
    validate_opus_header_stream(target_serial, tags_pkt.stream_serial())?;
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
        validate_query_packets(input, target_serial)?;
        absgp
            .checked_sub(head.pre_skip as u64)
            .ok_or_else(|| anyhow!("final granule {absgp} is before pre-skip {}", head.pre_skip))?
    } else {
        decode_sample_count(input, head, target_serial)?
    };
    Ok(sample_count)
}

/// Validate every target-stream audio packet before trusting a fast-path
/// granule. Strict scalar queries must never print a value from a stream that
/// contains a malformed packet, even when an EOS granule is available.
fn validate_query_packets(input: &std::path::Path, target_serial: u32) -> Result<()> {
    let file =
        File::open(input).with_context(|| format!("opening {}", escape_terminal_path(input)))?;
    let mut reader = BoundedPacketReader::for_serial(BufReader::new(file), target_serial);
    let _head_pkt = reader.read_packet()?.ok_or_else(|| anyhow!("empty file"))?;
    let tags_pkt = reader
        .read_header_packet()?
        .ok_or_else(|| anyhow!("expected OpusTags packet, got end of stream"))?;
    validate_opus_header_stream(target_serial, tags_pkt.stream_serial())?;

    let mut packet_idx = 0u64;
    while let Some(pkt) = reader.read_packet()? {
        if pkt.stream_serial() == target_serial {
            validate_opus_audio_packet(&pkt.data)
                .with_context(|| format!("validating Opus audio packet {packet_idx}"))?;
            packet_idx += 1;
        }
    }
    Ok(())
}

/// Slow duration fallback for truncated streams. This decodes packets in a
/// bounded-memory loop and does not retain their TOCs or tag strings.
fn decode_sample_count(input: &std::path::Path, head: OpusHead, target_serial: u32) -> Result<u64> {
    let file =
        File::open(input).with_context(|| format!("opening {}", escape_terminal_path(input)))?;
    let mut reader = BoundedPacketReader::for_serial(BufReader::new(file), target_serial);
    // Skip OpusHead and OpusTags; the first packet is validated against the
    // caller's head, while the tags payload is intentionally not parsed.
    let head_pkt = reader.read_packet()?.ok_or_else(|| anyhow!("empty file"))?;
    let tags_pkt = reader
        .read_header_packet()?
        .ok_or_else(|| anyhow!("expected OpusTags packet, got end of stream"))?;
    validate_opus_header_stream(head_pkt.stream_serial(), tags_pkt.stream_serial())?;

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
        let n = decoder
            .decode(&pkt.data, &mut decoded, DecodeMode::Normal)
            .map_err(|e| anyhow!("decoding Opus audio packet {packet_idx}: {e}"))?;
        sample_count += n as u64;
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
                sample_count: SampleCount::Exact(0),
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
                sample_count: SampleCount::Exact(0),
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
                sample_count: SampleCount::Exact(sample_count),
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

    fn build_page(serial: u32, sequence: u32, header_type: u8, lacing: &[u8]) -> Vec<u8> {
        let body_len = lacing.iter().map(|&lace| lace as usize).sum::<usize>();
        let mut page = Vec::with_capacity(27 + lacing.len() + body_len);
        page.extend_from_slice(b"OggS");
        page.push(0);
        page.push(header_type);
        page.extend_from_slice(&0u64.to_le_bytes());
        page.extend_from_slice(&serial.to_le_bytes());
        page.extend_from_slice(&sequence.to_le_bytes());
        page.extend_from_slice(&0u32.to_le_bytes());
        page.push(lacing.len() as u8);
        page.extend_from_slice(lacing);
        page.extend(std::iter::repeat_n(0xA5, body_len));
        let crc = ogg_crc32(&page);
        page[22..26].copy_from_slice(&crc.to_le_bytes());
        page
    }

    fn append_packet(output: &mut Vec<u8>, serial: u32, sequence: &mut u32, length: usize) {
        let mut remaining = length;
        let mut first_page = true;
        while remaining > 0 {
            let chunk = remaining.min(255);
            let mut header_type = if first_page { 0x02 } else { 0x01 };
            remaining -= chunk;
            if remaining == 0 {
                header_type |= 0x04;
            }
            let lacing = if chunk == 255 && remaining == 0 {
                // A zero-length lace terminates a packet whose size is an
                // exact multiple of 255.
                vec![255, 0]
            } else {
                vec![chunk as u8]
            };
            output.extend_from_slice(&build_page(serial, *sequence, header_type, &lacing));
            *sequence += 1;
            first_page = false;
        }
    }

    fn stream_with_audio_packet(length: usize) -> Vec<u8> {
        let serial = 0xABCD_1234;
        let mut output = Vec::new();
        let mut sequence = 0;
        append_packet(&mut output, serial, &mut sequence, 19);
        append_packet(&mut output, serial, &mut sequence, 16);
        append_packet(&mut output, serial, &mut sequence, length);
        output
    }

    fn stream_with_packet_lengths(lengths: &[usize]) -> Vec<u8> {
        let serial = 0xABCD_1234;
        let mut output = Vec::new();
        let mut sequence = 0;
        for &length in lengths {
            append_packet(&mut output, serial, &mut sequence, length);
        }
        output
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

    #[test]
    fn bounded_reader_accepts_head_packet_at_limit() {
        let bytes = stream_with_packet_lengths(&[MAX_OPUS_HEAD_PACKET_BYTES]);
        let mut reader = BoundedPacketReader::new(Cursor::new(bytes));
        assert_eq!(
            reader.read_packet().unwrap().unwrap().data.len(),
            MAX_OPUS_HEAD_PACKET_BYTES
        );
    }

    #[test]
    fn bounded_reader_rejects_oversized_head_packet() {
        let bytes = stream_with_packet_lengths(&[MAX_OPUS_HEAD_PACKET_BYTES + 1]);
        let mut reader = BoundedPacketReader::new(Cursor::new(bytes));
        let error = match reader.read_packet() {
            Err(error) => error,
            Ok(Some(_)) => panic!("OpusHead over budget must be rejected"),
            Ok(None) => panic!("OpusHead packet disappeared"),
        };
        assert!(error.to_string().contains("packet 0"));
    }

    #[test]
    fn bounded_reader_accepts_tags_packet_at_limit() {
        let bytes = stream_with_packet_lengths(&[19, MAX_OPUS_TAGS_PACKET_BYTES]);
        let mut reader = BoundedPacketReader::new(Cursor::new(bytes));
        assert_eq!(reader.read_packet().unwrap().unwrap().data.len(), 19);
        assert_eq!(
            reader.read_packet().unwrap().unwrap().data.len(),
            MAX_OPUS_TAGS_PACKET_BYTES
        );
    }

    #[test]
    fn bounded_reader_rejects_oversized_tags_packet() {
        let bytes = stream_with_packet_lengths(&[19, MAX_OPUS_TAGS_PACKET_BYTES + 1]);
        let mut reader = BoundedPacketReader::new(Cursor::new(bytes));
        reader.read_packet().unwrap().unwrap();
        let error = match reader.read_packet() {
            Err(error) => error,
            Ok(Some(_)) => panic!("OpusTags over budget must be rejected"),
            Ok(None) => panic!("OpusTags packet disappeared"),
        };
        assert!(error.to_string().contains("packet 1"));
    }

    #[test]
    fn bounded_reader_accepts_audio_packet_at_limit_across_continued_pages() {
        let bytes = stream_with_audio_packet(MAX_PACKET_BYTES);
        let mut reader = BoundedPacketReader::new(Cursor::new(bytes));
        assert_eq!(reader.read_packet().unwrap().unwrap().data.len(), 19);
        assert_eq!(reader.read_packet().unwrap().unwrap().data.len(), 16);
        assert_eq!(
            reader.read_packet().unwrap().unwrap().data.len(),
            MAX_PACKET_BYTES
        );
    }

    #[test]
    fn bounded_reader_rejects_oversized_continued_audio_packet() {
        let bytes = stream_with_audio_packet(MAX_PACKET_BYTES + 1);
        let mut reader = BoundedPacketReader::new(Cursor::new(bytes));
        reader.read_packet().unwrap().unwrap();
        reader.read_packet().unwrap().unwrap();
        let error = match reader.read_packet() {
            Err(error) => error,
            Ok(Some(_)) => panic!("audio packet over budget must be rejected"),
            Ok(None) => panic!("audio packet disappeared"),
        };
        assert!(error.to_string().contains("exceeds info limit"));
    }

    #[test]
    fn bounded_reader_skips_interleaved_logical_stream_pages() {
        let target = 0xABCD_1234;
        let other = 0x1020_3040;
        let mut bytes = Vec::new();
        let mut target_sequence = 0;
        let mut other_sequence = 0;
        append_packet(&mut bytes, target, &mut target_sequence, 19);
        append_packet(&mut bytes, other, &mut other_sequence, 64);
        append_packet(&mut bytes, target, &mut target_sequence, 16);
        append_packet(&mut bytes, other, &mut other_sequence, 64);
        append_packet(&mut bytes, target, &mut target_sequence, 1);

        let mut reader = BoundedPacketReader::for_serial(Cursor::new(bytes), target);
        assert_eq!(reader.read_packet().unwrap().unwrap().data.len(), 19);
        assert_eq!(reader.read_packet().unwrap().unwrap().data.len(), 16);
        assert_eq!(reader.read_packet().unwrap().unwrap().data.len(), 1);
    }

    #[test]
    fn header_reader_tracks_interleaved_non_target_continuations_by_serial() {
        let target = 0xABCD_1234;
        let pending_other = 0x1020_3040;
        let completed_other = 0x5060_7080;
        let mut bytes = Vec::new();
        let mut target_sequence = 0;
        append_packet(&mut bytes, target, &mut target_sequence, 19);
        bytes.extend_from_slice(&build_page(pending_other, 0, 0x02, &[255]));
        bytes.extend_from_slice(&build_page(completed_other, 0, 0x02, &[16]));

        let mut reader = BoundedPacketReader::new(Cursor::new(bytes));
        reader.read_packet().unwrap().unwrap();
        let packet = reader
            .read_header_packet()
            .unwrap()
            .expect("completed interleaved header packet");
        assert_eq!(packet.stream_serial(), completed_other);
        assert_eq!(packet.data.len(), 16);
    }

    #[test]
    fn playback_length_rounds_before_minute_carry() {
        let one_sample = 1.0 / 48_000.0;

        assert_eq!(format_playback_length(60.0 - one_sample), "1m 0.00s");
    }

    #[test]
    fn playback_length_rounds_before_hour_carry() {
        let one_sample = 1.0 / 48_000.0;

        assert_eq!(format_playback_length(3600.0 - one_sample), "1h 0m 0.00s");
    }

    #[test]
    fn playback_length_preserves_sub_minute_shape() {
        assert_eq!(format_playback_length(12.345), "12.35s");
    }
}
