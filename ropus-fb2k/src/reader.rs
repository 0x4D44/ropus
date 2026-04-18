//! `OggOpusReader<R>` — header + tags parse for M1.
//!
//! The real fun (seek index, decode loop, ReplayGain conversion) lives in
//! M2/M3. For now we only need to:
//!
//! * open an Ogg Opus stream,
//! * parse `OpusHead` (rejecting channel_mapping != 0 up front),
//! * parse `OpusTags`,
//! * expose the fields `RopusFb2kInfo` needs (channels, pre_skip,
//!   input_sample_rate) plus the parsed tags.
//!
//! Nothing here owns a decoder yet.

use std::fmt;
use std::io::{Read, Seek};

use ogg::reading::PacketReader;

use crate::io::AbortTag;
use crate::tags::{self, ParsedTags, TagError};

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
    #[allow(dead_code)] // surfaced in tests and future M2 work; not yet on the C ABI.
    pub(crate) version: u8,
    pub(crate) channels: u8,
    pub(crate) pre_skip: u16,
    /// Informational-only per RFC 7845 (we always decode at 48 kHz). Kept
    /// so M2 can surface it alongside the other OpusHead fields if wanted.
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
/// (ready for M2's decode loop to start pulling audio packets).
///
/// `PacketReader` doesn't derive `Debug`, so we hand-write the impl and skip
/// it in the output — tests that `expect_err` the `Result` only need the
/// error side printable, which already implements `Debug`.
pub(crate) struct OggOpusReader<R: Read + Seek> {
    head: OpusHead,
    tags: ParsedTags,
    /// Serial of the logical bitstream we latched onto on page 1. M2 uses
    /// this to filter a reverse-scan to the right stream (matching what
    /// `ropus-cli info` already does, see `ropus-cli/src/container/ogg.rs`).
    #[allow(dead_code)]
    stream_serial: u32,
    /// `PacketReader` kept alive for M2 to pull audio packets from. Holding
    /// the reader as an `Option` so M2 can `take()` it for FEC-friendly
    /// manual loops without fighting the borrow checker.
    #[allow(dead_code)]
    packet_reader: Option<PacketReader<R>>,
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
    /// and stops. M1 scope — no page-walk, no decoder.
    ///
    /// `_flags` is accepted for forward compatibility with M2's fast-path;
    /// M1 ignores it. We take it rather than inventing a second constructor
    /// later, so the call site doesn't churn when M2 lands.
    pub(crate) fn open(reader: R, _flags: u32) -> Result<Self, ReaderError> {
        let mut packet_reader = PacketReader::new(reader);

        // --- Page 1: OpusHead ------------------------------------------------
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

        // --- Page 2: OpusTags ------------------------------------------------
        let tags_pkt = packet_reader
            .read_packet()
            .map_err(ogg_err_to_reader)?
            .ok_or_else(|| ReaderError::InvalidStream("missing OpusTags page".into()))?;
        let tags = tags::parse(&tags_pkt.data)?;

        Ok(Self {
            head,
            tags,
            stream_serial,
            packet_reader: Some(packet_reader),
        })
    }

    pub(crate) fn channels(&self) -> u8 {
        self.head.channels
    }

    pub(crate) fn pre_skip(&self) -> u16 {
        self.head.pre_skip
    }

    /// Informational-only per RFC 7845; M1 never surfaces it through the
    /// C ABI (we decode at 48 kHz). Kept for M2's diagnostic logging.
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
        let err = OggOpusReader::open(Cursor::new(Vec::<u8>::new()), 0).unwrap_err();
        assert!(matches!(err, ReaderError::InvalidStream(_)));
    }
}
