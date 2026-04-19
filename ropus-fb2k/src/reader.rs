//! `OggOpusReader<R>` — the crate's demuxer + decoder driver.
//!
//! Responsibilities:
//!
//! * open an Ogg Opus stream,
//! * parse `OpusHead` (rejecting channel_mapping != 0 up front),
//! * parse `OpusTags`,
//! * reverse-scan the final Ogg page to learn total duration and
//!   average bitrate (seekable inputs only — unseekable streams open
//!   successfully with zero duration and `-1` bitrate),
//! * drive a lazily-constructed `OpusDecoder` for `decode_next`,
//! * expose the fields `RopusFb2kInfo` needs plus the parsed tags.
//!
//! Seek (`ropus_fb2k_seek`) is not yet implemented; the page-walk index
//! that the real seek path will build lives alongside this struct.

use std::fmt;
use std::io::{Read, Seek, SeekFrom};

use ogg::reading::PacketReader;
use ropus::OpusDecoder;

use crate::io::AbortTag;
use crate::tags::{self, ParsedTags, TagError};

/// Output sample rate used by every decoded frame. `OpusHead::input_sample_rate`
/// is informational-only per RFC 7845; the codec always produces 48 kHz.
pub(crate) const OPUS_SAMPLE_RATE_HZ: i32 = 48_000;

/// Maximum per-channel samples an Opus frame can produce (120 ms @ 48 kHz).
/// Sizes the scratch buffer `decode_next` hands to `OpusDecoder::decode_float`.
pub(crate) const MAX_FRAME_SAMPLES_PER_CH: usize = 5760;

/// Size of an `OpusHead` packet for channel_mapping=0 (mono/stereo). Family 1
/// payloads are longer, but we reject them before inspecting the extra bytes.
const OPUS_HEAD_MIN_LEN: usize = 19;

/// Supported channel-mapping family. Matches HLD sec. 2: surround and
/// chained-stream support are explicitly out of scope.
const SUPPORTED_MAPPING_FAMILY: u8 = 0;

/// Parsed `OpusHead` fields we carry forward. Matches the RFC 7845 sec. 5.1
/// layout (little-endian multi-byte fields, channel_mapping_family at the
/// end of the 19-byte prefix).
#[derive(Debug, Clone, Copy)]
pub(crate) struct OpusHead {
    #[allow(dead_code)] // surfaced in tests; not yet on the C ABI.
    pub(crate) version: u8,
    pub(crate) channels: u8,
    pub(crate) pre_skip: u16,
    /// Informational-only per RFC 7845 (we always decode at 48 kHz). Kept
    /// so future work can surface it alongside the other OpusHead fields.
    #[allow(dead_code)]
    pub(crate) input_sample_rate: u32,
    #[allow(dead_code)]
    pub(crate) output_gain: i16,
    pub(crate) channel_mapping: u8,
}

/// Errors the reader surfaces, coarsely aligned with the C-level status
/// codes. The FFI shim maps each variant to an appropriate negative status
/// and a human message. Crate-private: C callers only ever see the mapped
/// `c_int`, and integration tests drive the C path.
#[derive(Debug)]
pub(crate) enum ReaderError {
    Io(std::io::Error),
    /// The stream doesn't look like a valid Ogg Opus file — missing magic,
    /// malformed OpusHead / OpusTags, truncated, etc.
    InvalidStream(String),
    /// The stream *is* well-formed Opus but uses features we don't implement
    /// (surround, chained streams).
    Unsupported(String),
    /// The IO callbacks indicated the caller requested cancellation.
    Aborted,
}

impl fmt::Display for ReaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReaderError::Io(e) => write!(f, "I/O error: {e}"),
            ReaderError::InvalidStream(s) => write!(f, "invalid Opus stream: {s}"),
            ReaderError::Unsupported(s) => write!(f, "unsupported Opus feature: {s}"),
            ReaderError::Aborted => f.write_str("aborted"),
        }
    }
}

impl std::error::Error for ReaderError {}

/// Classify an `std::io::Error` as abort or generic I/O. `CallbackReader`
/// tags the abort path by wrapping an `AbortTag` marker inside the error;
/// we detect it via `downcast_ref` on the direct payload *and* by walking
/// the `Error::source()` chain in case an intermediate layer wraps our
/// error in its own type along the way.
fn classify_io_error(e: std::io::Error) -> ReaderError {
    if io_error_is_abort(&e) {
        ReaderError::Aborted
    } else {
        ReaderError::Io(e)
    }
}

/// Returns true iff the payload of `e` (or any link of its `source()` chain)
/// is an `AbortTag`. The payload lives inside `io::Error::get_ref()`, which
/// isn't visited by `Error::source()` by default — we check it explicitly
/// at every level.
fn io_error_is_abort(e: &std::io::Error) -> bool {
    if let Some(inner) = e.get_ref()
        && source_chain_contains_abort(inner)
    {
        return true;
    }
    false
}

fn source_chain_contains_abort(e: &(dyn std::error::Error + 'static)) -> bool {
    let mut cur: Option<&(dyn std::error::Error + 'static)> = Some(e);
    while let Some(node) = cur {
        if node.downcast_ref::<AbortTag>().is_some() {
            return true;
        }
        if let Some(io_err) = node.downcast_ref::<std::io::Error>()
            && io_error_is_abort(io_err)
        {
            return true;
        }
        cur = node.source();
    }
    false
}

impl From<TagError> for ReaderError {
    fn from(e: TagError) -> Self {
        ReaderError::InvalidStream(format!("OpusTags: {e}"))
    }
}

/// Top-level Ogg Opus reader. Holds the parsed header, the parsed tags, and
/// the underlying `PacketReader<R>` positioned just after the OpusTags page
/// (ready for the decode loop to start pulling audio packets).
///
/// `PacketReader` doesn't derive `Debug`, so we hand-write the impl and skip
/// it in the output — tests that `expect_err` the `Result` only need the
/// error side printable, which already implements `Debug`.
pub(crate) struct OggOpusReader<R: Read + Seek> {
    head: OpusHead,
    tags: ParsedTags,
    /// Serial of the logical bitstream we latched onto on page 1. Used to
    /// filter the reverse-scan to the right stream (matches the pattern in
    /// `ropus-cli/src/container/ogg.rs::read_last_granule`).
    stream_serial: u32,
    /// `PacketReader` kept alive for the decode loop. `Option` so a future
    /// seek path can `take()` it, rebuild the index, and re-seat without
    /// fighting the borrow checker. Never `None` between `open` and `close`
    /// today.
    packet_reader: Option<PacketReader<R>>,
    /// Total decoded samples per channel after pre-skip trim, derived from a
    /// reverse-scan of the last Ogg page's absolute granule position. `0`
    /// means unknown (unseekable IO, unknown size, or sentinel granule).
    total_samples: u64,
    /// Average bitrate in bits/sec, computed at open time from `file_size *
    /// 8 * 48_000 / total_samples`. `-1` means unknown.
    nominal_bitrate: i32,
    /// Low-level decoder, lazily constructed on the first `decode_next` call
    /// so open remains cheap for fb2k's info-scan path. Persists across
    /// calls — Opus decode state is stateful and must not be reset between
    /// packets within a single playback. The future seek path will reset it.
    decoder: Option<OpusDecoder>,
    /// Scratch buffer handed to `OpusDecoder::decode_float`, sized once to
    /// `MAX_FRAME_SAMPLES_PER_CH * channels` (~46 KiB stereo) and reused
    /// across calls. Allocating per-call on the audio thread stressed the
    /// Windows heap; a reader-owned buffer keeps realtime-adjacent decode
    /// allocation-free after the first call.
    decode_scratch: Vec<f32>,
    /// Per-channel samples remaining to trim from the leading edge of the
    /// decoded stream, per RFC 7845 §4.2. Initialised to `head.pre_skip` on
    /// open; counted down by `decode_next` as it drops the first samples.
    pre_skip_remaining: u32,
    // Hand-off note for the seek implementation: the full page-walk index
    // (Vec<(granule, offset)>) will live here, built lazily on first seek.
    // Placing the Option<..> field between `pre_skip_remaining` and the end
    // keeps this struct compact.
}

impl<R: Read + Seek> fmt::Debug for OggOpusReader<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OggOpusReader")
            .field("head", &self.head)
            .field("vendor", &self.tags.vendor)
            .field("comments", &self.tags.comments.len())
            .field("stream_serial", &self.stream_serial)
            .finish()
    }
}

impl<R: Read + Seek> OggOpusReader<R> {
    /// Open a stream. Reads the first two Ogg pages (OpusHead, OpusTags)
    /// and, if the IO supports seeking with a known size, reverse-scans
    /// from EOF for the last page's granule position to populate
    /// `total_samples` and `nominal_bitrate`.
    ///
    /// `flags` is accepted for both the info-only fast path and the full
    /// decode path; both paths currently reverse-scan identically. A future
    /// release will replace the full path with a streaming page-walk that
    /// also populates the seek index.
    ///
    /// `file_size_hint` is the total stream length in bytes if the IO layer
    /// knows it (from the `size` callback), or `None` for live streams. It's
    /// plumbed in from the FFI shim because our `Seek` adapter needs it to
    /// resolve `SeekFrom::End(_)` and because the bitrate formula consumes
    /// it directly.
    pub(crate) fn open(
        reader: R,
        _flags: u32,
        file_size_hint: Option<u64>,
    ) -> Result<Self, ReaderError> {
        let mut packet_reader = PacketReader::new(reader);

        // --- Page 1: OpusHead ---------------------------------------
        let head_pkt = packet_reader
            .read_packet()
            .map_err(ogg_err_to_reader)?
            .ok_or_else(|| ReaderError::InvalidStream("empty stream".into()))?;
        let stream_serial = head_pkt.stream_serial();
        let head = parse_opus_head(&head_pkt.data)?;

        if head.channel_mapping != SUPPORTED_MAPPING_FAMILY {
            return Err(ReaderError::Unsupported(format!(
                "channel_mapping_family={} (only family 0 supported)",
                head.channel_mapping
            )));
        }
        if head.channels == 0 || head.channels > 2 {
            return Err(ReaderError::Unsupported(format!(
                "channel count {} unsupported in family 0",
                head.channels
            )));
        }

        // --- Page 2: OpusTags ---------------------------------------
        let tags_pkt = packet_reader
            .read_packet()
            .map_err(ogg_err_to_reader)?
            .ok_or_else(|| ReaderError::InvalidStream("missing OpusTags page".into()))?;
        let tags = tags::parse(&tags_pkt.data)?;

        // --- Reverse-scan for last granule ---------------------------------
        //
        // For a live HTTP stream the caller provides no `size` and no `seek`,
        // in which case we skip this and report `total_samples = 0`,
        // `nominal_bitrate = -1`. We purposely *don't* fail open — HLD §4.3
        // explicitly permits "unseekable / unknown size" with zero duration.
        //
        // When a scan is needed we drop the current `PacketReader`, run the
        // scan on the bare reader (so we can seek freely without fighting
        // `PacketReader`'s internal state — the `ogg` crate's `get_mut` is
        // explicitly marked as capable of corrupting that state), then
        // construct a fresh `PacketReader` from the bare reader and use
        // `seek_bytes(Start(resume_pos))` to flip its internal `has_seeked`
        // flag. Without that flag, a fresh `PacketReader` landing mid-stream
        // refuses the first non-first-page from a previously-unseen serial.
        //
        // For unseekable streams we can't run the scan (no way to return to
        // the audio start) and we can't `seek_bytes` a fresh `PacketReader`
        // either. Instead we keep the original `PacketReader` alive — it
        // has already registered the stream serial via OpusHead, so its
        // subsequent page reads hit the `Occupied` branch in the ogg crate
        // and tolerate mid-stream pages without needing `has_seeked`.
        let (packet_reader, total_samples, nominal_bitrate) = match file_size_hint {
            Some(size) if size > 0 => {
                let mut bare_reader = packet_reader.into_inner();
                let resume_pos = bare_reader.stream_position().map_err(classify_io_error)?;
                let (total_samples, nominal_bitrate) = compute_duration_and_bitrate(
                    &mut bare_reader,
                    stream_serial,
                    head.pre_skip,
                    size,
                )?;
                let mut fresh = PacketReader::new(bare_reader);
                fresh.seek_bytes(SeekFrom::Start(resume_pos))
                    .map_err(classify_io_error)?;
                (fresh, total_samples, nominal_bitrate)
            }
            _ => (packet_reader, 0u64, -1i32),
        };

        Ok(Self {
            head,
            tags,
            stream_serial,
            packet_reader: Some(packet_reader),
            total_samples,
            nominal_bitrate,
            decoder: None,
            decode_scratch: Vec::new(),
            pre_skip_remaining: head.pre_skip as u32,
        })
    }

    pub(crate) fn channels(&self) -> u8 {
        self.head.channels
    }

    pub(crate) fn pre_skip(&self) -> u16 {
        self.head.pre_skip
    }

    /// Informational-only per RFC 7845; we never surface it through the C
    /// ABI (we always decode at 48 kHz). Kept for diagnostic logging.
    #[allow(dead_code)]
    pub(crate) fn input_sample_rate(&self) -> u32 {
        self.head.input_sample_rate
    }

    pub(crate) fn vendor(&self) -> &str {
        &self.tags.vendor
    }

    pub(crate) fn tags(&self) -> &ParsedTags {
        &self.tags
    }

    pub(crate) fn total_samples(&self) -> u64 {
        self.total_samples
    }

    pub(crate) fn nominal_bitrate(&self) -> i32 {
        self.nominal_bitrate
    }

    /// Pull the next Opus packet, decode it into the caller's interleaved
    /// float buffer, and return the number of samples-per-channel written.
    /// Returns `Ok(0)` at end-of-stream.
    ///
    /// On the first successful call after open, the leading `pre_skip`
    /// samples are transparently dropped before returning — so the caller
    /// never sees the encoder warm-up. If a whole packet is consumed by
    /// pre-skip the reader loops to the next packet; this matters because
    /// pre-skip can exceed a 2.5 ms frame (312 lookahead for typical
    /// encoders > 120 samples) and would otherwise surface as a spurious
    /// "empty" return value.
    pub(crate) fn decode_next(
        &mut self,
        out_interleaved: &mut [f32],
        max_samples_per_ch: usize,
    ) -> Result<i32, ReaderError> {
        let channels = self.head.channels as usize;
        // Lazy-init the low-level decoder AND the scratch buffer together.
        // Deferring until the first decode keeps open() cheap for fb2k
        // library scans that never decode; re-using `decode_scratch` across
        // calls keeps the audio-thread path allocation-free.
        if self.decoder.is_none() {
            let dec = OpusDecoder::new(OPUS_SAMPLE_RATE_HZ, channels as i32)
                .map_err(|code| {
                    ReaderError::InvalidStream(format!(
                        "OpusDecoder init failed (code {code})"
                    ))
                })?;
            self.decoder = Some(dec);
            self.decode_scratch = vec![0f32; MAX_FRAME_SAMPLES_PER_CH * channels];
        }

        let packet_reader = self
            .packet_reader
            .as_mut()
            .expect("packet_reader is always Some between open and drop");
        let decoder = self
            .decoder
            .as_mut()
            .expect("decoder was just lazy-initialised");
        let scratch = self.decode_scratch.as_mut_slice();

        loop {
            let pkt = packet_reader.read_packet().map_err(ogg_err_to_reader)?;
            let Some(pkt) = pkt else {
                return Ok(0); // clean EOF
            };

            let decoded = decoder
                .decode_float(
                    Some(&pkt.data),
                    scratch,
                    MAX_FRAME_SAMPLES_PER_CH as i32,
                    false,
                )
                .map_err(|code| {
                    ReaderError::InvalidStream(format!(
                        "OpusDecoder decode_float failed (code {code})"
                    ))
                })? as usize;

            // Trim pre-skip from the leading edge, possibly consuming the
            // whole packet on the first few 2.5 ms sub-frames. A live packet
            // yielding zero samples is not a normal pre-skip case — it means
            // the decoder produced nothing for a non-empty packet, which
            // would cause the loop to spin silently draining packets. Fail
            // fast instead.
            if decoded == 0 {
                return Err(ReaderError::InvalidStream(
                    "decoder returned 0 samples for a live packet during pre-skip".into(),
                ));
            }

            let (drop, kept) = if self.pre_skip_remaining > 0 {
                let drop = (self.pre_skip_remaining as usize).min(decoded);
                self.pre_skip_remaining -= drop as u32;
                (drop, decoded - drop)
            } else {
                (0, decoded)
            };

            if kept == 0 {
                // Packet entirely consumed by pre-skip — grab the next one.
                continue;
            }

            // The FFI shim enforces `max_samples_per_ch >= 5760 ==
            // MAX_FRAME_SAMPLES_PER_CH`, and Opus packets produce at most
            // 5760 samples/ch, so `kept` cannot exceed the caller's buffer
            // in any currently-reachable path. If that invariant ever
            // changes, fail loudly — silently truncating decoded audio is
            // the worst possible failure mode.
            debug_assert!(
                kept <= max_samples_per_ch,
                "caller buffer smaller than FFI contract guarantees"
            );
            if kept > max_samples_per_ch {
                return Err(ReaderError::InvalidStream(format!(
                    "decoded {kept} samples/ch exceeds caller buffer {max_samples_per_ch}"
                )));
            }

            let src_start = drop * channels;
            let src_end = src_start + kept * channels;
            out_interleaved[..kept * channels].copy_from_slice(&scratch[src_start..src_end]);

            return Ok(kept as i32);
        }
    }
}

/// Scan the last 128 KiB of the stream for the last Ogg page whose serial
/// matches `target_serial`; from its granule position compute total decoded
/// samples (post-pre-skip) and the average bitrate in bits/sec.
///
/// Sibling to `ropus-cli/src/container/ogg.rs::read_last_granule` — the same
/// RFC 3533 bit-twiddling; deliberately copied rather than shared because
/// making `ropus-fb2k` depend on `ropus-cli` for a 60-line helper would
/// couple two otherwise-independent binaries just to save duplication.
///
/// Preserves the underlying reader's cursor: restores it to wherever it was
/// before the scan so the caller's `PacketReader` state remains consistent
/// when we hand back. Truncated files (file_size < 27 bytes or no matching
/// page in the trailing window) silently produce `(0, -1)` — that's a
/// degraded but non-fatal open.
fn compute_duration_and_bitrate<R: Read + Seek>(
    reader: &mut R,
    target_serial: u32,
    pre_skip: u16,
    file_size: u64,
) -> Result<(u64, i32), ReaderError> {
    let saved_pos = reader.stream_position().map_err(classify_io_error)?;
    let absgp = read_last_granule(reader, target_serial, file_size).map_err(classify_io_error)?;
    // Best-effort restore so PacketReader's view of the stream is unchanged.
    // If the restore fails the caller will surface the next IO error on its
    // next read — we can't do anything more useful here.
    reader
        .seek(SeekFrom::Start(saved_pos))
        .map_err(classify_io_error)?;

    let Some(absgp) = absgp else {
        return Ok((0, -1));
    };
    let total_samples = absgp.saturating_sub(pre_skip as u64);
    if total_samples == 0 {
        return Ok((0, -1));
    }
    // bits/sec = file_size * 8 * sample_rate / total_samples. Compute in u128
    // to avoid overflow on very long, very large files (a 10 GB file with
    // 24 h of audio is well inside u128).
    let num: u128 = (file_size as u128)
        .saturating_mul(8)
        .saturating_mul(OPUS_SAMPLE_RATE_HZ as u128);
    let bitrate = num / total_samples as u128;
    // Clamp into i32 — bitrates above 2 Gbps don't exist in the real world
    // and the fb2k shim just surfaces an int anyway.
    let bitrate = i32::try_from(bitrate).unwrap_or(i32::MAX);
    Ok((total_samples, bitrate))
}

/// Reverse-scan for the absolute granule position of the last Ogg page
/// belonging to `target_serial`. Returns `Ok(None)` if the granule is the
/// unknown-sentinel (`0xFFFF_FFFF_FFFF_FFFF`) or no matching page is found
/// in the trailing 128 KiB. `file_size` is passed in rather than re-derived
/// from `Seek::seek(End(0))` because `CallbackReader` already cached it at
/// construction and another round-trip through the callback is pointless.
// NOTE: this is a line-for-line copy of ropus-cli's read_last_granule. Duplication
// is intentional for now; once the seek path lands, post-release cleanup should
// factor both copies into a shared helper (likely in `ropus::container::ogg`).
// Do NOT sync bug-fixes manually — when that sync need arises, do the extraction
// instead.
fn read_last_granule<R: Read + Seek>(
    reader: &mut R,
    target_serial: u32,
    file_size: u64,
) -> std::io::Result<Option<u64>> {
    /// Absolute cap on how far back we scan. RFC 3533 limits an Ogg page to
    /// ~65 KiB (27-byte header + 255 × 255 lacing bytes), so 128 KiB reliably
    /// spans a max-sized final page even with trailing junk.
    const SCAN_WINDOW: u64 = 128 * 1024;
    const HEADER_LEN: usize = 27;
    const UNKNOWN_GRANULE: u64 = 0xFFFF_FFFF_FFFF_FFFF;

    if file_size < HEADER_LEN as u64 {
        return Ok(None);
    }

    let read_len = SCAN_WINDOW.min(file_size);
    let start = file_size - read_len;
    reader.seek(SeekFrom::Start(start))?;

    let mut buf = vec![0u8; read_len as usize];
    reader.read_exact(&mut buf)?;

    // Walk back byte-by-byte for the b"OggS" capture pattern, validating
    // the fixed-layout header fields before trusting the granule. Matches
    // the reference helper in `ropus-cli/src/container/ogg.rs`.
    let mut i = buf.len().saturating_sub(4);
    loop {
        if i + HEADER_LEN <= buf.len()
            && &buf[i..i + 4] == b"OggS"
            // stream_structure_version must be 0 per RFC 3533 §6.
            && buf[i + 4] == 0
        {
            let absgp = u64::from_le_bytes([
                buf[i + 6],
                buf[i + 7],
                buf[i + 8],
                buf[i + 9],
                buf[i + 10],
                buf[i + 11],
                buf[i + 12],
                buf[i + 13],
            ]);
            let serial = u32::from_le_bytes([
                buf[i + 14],
                buf[i + 15],
                buf[i + 16],
                buf[i + 17],
            ]);
            if serial == target_serial {
                if absgp == UNKNOWN_GRANULE {
                    return Ok(None);
                }
                return Ok(Some(absgp));
            }
        }
        if i == 0 {
            return Ok(None);
        }
        i -= 1;
    }
}

/// Parse the first 19 bytes of an OpusHead packet. Deliberately accepts any
/// `version >= 1` because `OpusHead` was designed to be forward-compatible
/// by bumping the *high* nibble only (RFC 7845 sec. 5.1).
fn parse_opus_head(data: &[u8]) -> Result<OpusHead, ReaderError> {
    if data.len() < OPUS_HEAD_MIN_LEN {
        return Err(ReaderError::InvalidStream(format!(
            "OpusHead packet is {} bytes, expected >= {}",
            data.len(),
            OPUS_HEAD_MIN_LEN
        )));
    }
    if &data[..8] != b"OpusHead" {
        return Err(ReaderError::InvalidStream(
            "OpusHead magic missing".into(),
        ));
    }
    let version = data[8];
    if version == 0 || (version & 0xF0) != 0 {
        // Major-version mismatch per RFC 7845 sec. 5.1: "The upper nibble
        // indicates the major version; implementations MUST reject any
        // stream with a higher major version."
        return Err(ReaderError::Unsupported(format!(
            "OpusHead version 0x{version:02x} unsupported"
        )));
    }
    Ok(OpusHead {
        version,
        channels: data[9],
        pre_skip: u16::from_le_bytes([data[10], data[11]]),
        input_sample_rate: u32::from_le_bytes([data[12], data[13], data[14], data[15]]),
        output_gain: i16::from_le_bytes([data[16], data[17]]),
        channel_mapping: data[18],
    })
}

fn ogg_err_to_reader(e: ogg::OggReadError) -> ReaderError {
    match e {
        ogg::OggReadError::ReadError(io) => classify_io_error(io),
        other => ReaderError::InvalidStream(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Build a minimal valid OpusHead packet.
    pub(crate) fn build_opus_head(channels: u8, pre_skip: u16, mapping: u8) -> Vec<u8> {
        let mut h = Vec::with_capacity(19);
        h.extend_from_slice(b"OpusHead");
        h.push(1); // version
        h.push(channels);
        h.extend_from_slice(&pre_skip.to_le_bytes());
        h.extend_from_slice(&48_000u32.to_le_bytes());
        h.extend_from_slice(&0i16.to_le_bytes());
        h.push(mapping);
        h
    }

    #[test]
    fn parse_opus_head_minimal_ok() {
        let h = parse_opus_head(&build_opus_head(2, 312, 0)).expect("ok");
        assert_eq!(h.channels, 2);
        assert_eq!(h.pre_skip, 312);
        assert_eq!(h.channel_mapping, 0);
        assert_eq!(h.input_sample_rate, 48_000);
    }

    #[test]
    fn parse_opus_head_rejects_short() {
        let short = vec![0u8; 10];
        let err = parse_opus_head(&short).unwrap_err();
        assert!(matches!(err, ReaderError::InvalidStream(_)));
    }

    #[test]
    fn parse_opus_head_rejects_bad_magic() {
        let mut h = build_opus_head(2, 312, 0);
        h[0] = b'X';
        let err = parse_opus_head(&h).unwrap_err();
        assert!(matches!(err, ReaderError::InvalidStream(_)));
    }

    #[test]
    fn parse_opus_head_rejects_future_major_version() {
        let mut h = build_opus_head(2, 312, 0);
        h[8] = 0x10; // major nibble set
        let err = parse_opus_head(&h).unwrap_err();
        assert!(matches!(err, ReaderError::Unsupported(_)));
    }

    #[test]
    fn open_rejects_empty() {
        let err = OggOpusReader::open(Cursor::new(Vec::<u8>::new()), 0, Some(0)).unwrap_err();
        assert!(matches!(err, ReaderError::InvalidStream(_)));
    }
}
