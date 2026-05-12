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
//! * expose the fields `RopusFb2kInfo` needs plus the parsed tags,
//! * `seek(sample_pos)` with 80 ms pre-roll per RFC 7845 §4.2 — page index
//!   built lazily on first seek so the info-only fast path stays cheap.
//!
//! Pre-skip and post-seek pre-roll are unified via a single
//! `next_sample_abs_pos` counter (in pre-skip-inclusive 48 kHz units — the
//! same semantics as Ogg Opus granules). On first decode the counter sits at
//! `0` and the target is `pre_skip`; on seek the counter jumps to the chosen
//! index entry's `start_granule` and the target becomes `sample_pos +
//! pre_skip`. `decode_next` drops leading samples until the counter catches
//! up to the target, unifying the two discard regimes.

use std::fmt;
use std::io::{Read, Seek, SeekFrom};

use ogg::reading::PacketReader;
use ropus::OpusDecoder;

use crate::io::AbortTag;
use crate::tags::{self, ParsedTags, ReplayGainInfo, TagError};

/// Output sample rate used by every decoded frame. `OpusHead::input_sample_rate`
/// is informational-only per RFC 7845; the codec always produces 48 kHz.
pub(crate) const OPUS_SAMPLE_RATE_HZ: i32 = 48_000;

/// Maximum per-channel samples an Opus frame can produce.
/// Opus frame = up to 120 ms at 48 kHz = 5760 samples per channel. RFC 6716 §2.
/// Sizes the scratch buffer `decode_next` hands to `OpusDecoder::decode_float`;
/// also re-exported via `lib.rs::MIN_OUT_SAMPLES_PER_CH` as the minimum
/// caller-buffer size advertised in `ropus_fb2k.h`.
pub(crate) const MAX_FRAME_SAMPLES_PER_CH: usize = 5760;

/// Size of an `OpusHead` packet for channel_mapping=0 (mono/stereo). Family 1
/// payloads are longer, but we reject them before inspecting the extra bytes.
const OPUS_HEAD_MIN_LEN: usize = 19;

/// Supported channel-mapping family. Matches HLD sec. 2: surround and
/// chained-stream support are explicitly out of scope.
const SUPPORTED_MAPPING_FAMILY: u8 = 0;

/// Seek pre-roll in 48 kHz samples per RFC 7845 §4.2. 80 ms × 48 kHz = 3840.
/// Matches libopusfile's `op_pcm_seek`: we rewind this many samples before
/// `sample_pos`, let the decoder converge, then silently discard the
/// pre-roll before returning real audio to the caller.
pub(crate) const PRE_ROLL_SAMPLES: u64 = 3_840;

/// Ogg page sentinel for "granule unknown" (RFC 3533 §6). Reverse-scan and
/// index walk both treat pages carrying this value as continuation pages
/// that don't advance the timeline.
const UNKNOWN_GRANULE: u64 = 0xFFFF_FFFF_FFFF_FFFF;

/// Ogg page capture pattern per RFC 3533 §6.
const OGG_CAPTURE: &[u8; 4] = b"OggS";

/// Fixed Ogg page header length (capture + structure + flags + granule +
/// serial + seq + crc + segment_count).
const OGG_HEADER_LEN: usize = 27;

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

/// One entry in the lazily-built page index.
///
/// `start_granule` is the number of decoded samples (in pre-skip-inclusive
/// 48 kHz units per RFC 7845 §4.2) the decoder has produced *before* any of
/// this page's packets are consumed. For the first audio page that's `0`;
/// for subsequent pages it's the end granule of the previous page.
/// `byte_offset` is the absolute file offset where the `OggS` capture
/// pattern of the page header starts.
///
/// Storing the start-granule (not the end-granule) lets `seek` set
/// `next_sample_abs_pos = entry.start_granule` directly after repositioning
/// the reader — no off-by-one reasoning required.
///
/// Continuation pages (granule == `0xFFFF_FFFF_FFFF_FFFF`) are NOT indexed
/// — they don't advance the timeline on their own, and seeking into one
/// would land the decoder mid-packet.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PageIndexEntry {
    pub(crate) start_granule: u64,
    pub(crate) byte_offset: u64,
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
    /// Cached ReplayGain values extracted from `tags` at open time. Stored
    /// here so `ropus_fb2k_get_info` is a pure read — we don't want to
    /// re-walk the tag list on every metadata query.
    replaygain: ReplayGainInfo,
    /// Serial of the logical bitstream we latched onto on page 1. Used to
    /// filter the reverse-scan and the page-index walk to the right stream
    /// (matches the pattern in `ropus-cli/src/container/ogg.rs`).
    stream_serial: u32,
    /// `PacketReader` kept alive for the decode loop. `Option` so the seek
    /// path can `take()` it, reposition the underlying reader, then re-seat
    /// a fresh `PacketReader` without fighting the borrow checker.
    packet_reader: Option<PacketReader<R>>,
    /// Byte offset of the first audio page — i.e. the stream position right
    /// after the OpusTags page was consumed. The page-index walk starts
    /// here. Cached at open time because we temporarily drop the
    /// `PacketReader` for the reverse-scan.
    audio_start_offset: u64,
    /// Total file size as advertised by the IO `size` callback. `None` for
    /// unseekable / unknown-size streams. Used by the page-index walk as a
    /// bound (otherwise we have no stop condition for the scan).
    file_size: Option<u64>,
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
    /// packets within a single playback. `seek` resets it.
    decoder: Option<OpusDecoder>,
    /// Scratch buffer handed to `OpusDecoder::decode_float`, sized once to
    /// `MAX_FRAME_SAMPLES_PER_CH * channels` (~46 KiB stereo) and reused
    /// across calls. Allocating per-call on the audio thread stressed the
    /// Windows heap; a reader-owned buffer keeps realtime-adjacent decode
    /// allocation-free after the first call.
    decode_scratch: Vec<f32>,
    /// Absolute 48 kHz sample position of the *next* sample about to be
    /// produced by the decoder, in pre-skip-inclusive units (matches Ogg
    /// Opus granule semantics per RFC 7845 §4.2). On open: starts at 0 and
    /// the target is `pre_skip` so the first `pre_skip` decoded samples are
    /// dropped. On seek: jumps to the start-page's `start_granule`; the
    /// target becomes `sample_pos + pre_skip`. The unified counter replaces
    /// the separate `pre_skip_remaining` field from M2.
    next_sample_abs_pos: u64,
    /// Target absolute sample position (same units as `next_sample_abs_pos`)
    /// at which the *caller* starts seeing samples. `decode_next` discards
    /// decoded samples while `next_sample_abs_pos < target_abs_pos`.
    /// Initialised to `pre_skip` on open; set to `sample_pos + pre_skip` on
    /// seek.
    target_abs_pos: u64,
    /// Page index for seek — built lazily on the first seek call to keep
    /// `open()` cheap (fb2k's library-scan path never needs it). `None`
    /// between open and first seek; `Some(empty)` is a legal degenerate
    /// result (stream had no indexable pages).
    page_index: Option<Vec<PageIndexEntry>>,
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

        // Capture the position right after the OpusTags page — the first
        // byte of the first audio page. Used by the seek path as the
        // starting point for the page-index walk.
        let mut bare_reader = packet_reader.into_inner();
        let audio_start_offset = bare_reader.stream_position().map_err(classify_io_error)?;

        // --- Reverse-scan for last granule ---------------------------------
        //
        // For a live HTTP stream the caller provides no `size` and no `seek`,
        // in which case we skip this and report `total_samples = 0`,
        // `nominal_bitrate = -1`. HLD §4.3 permits "unseekable / unknown
        // size" with zero duration.
        let (packet_reader, total_samples, nominal_bitrate) = match file_size_hint {
            Some(size) if size > 0 => {
                let (total_samples, nominal_bitrate) = compute_duration_and_bitrate(
                    &mut bare_reader,
                    stream_serial,
                    head.pre_skip,
                    size,
                )?;
                let mut fresh = PacketReader::new(bare_reader);
                fresh
                    .seek_bytes(SeekFrom::Start(audio_start_offset))
                    .map_err(classify_io_error)?;
                (fresh, total_samples, nominal_bitrate)
            }
            _ => {
                // Unseekable / unknown-size: wrap the bare reader back up
                // without touching its cursor — PacketReader has already
                // consumed the OpusTags page, so the bare reader sits at
                // `audio_start_offset` which is exactly where we want.
                (PacketReader::new(bare_reader), 0u64, -1i32)
            }
        };

        let replaygain = tags::extract_replaygain(&tags);

        Ok(Self {
            head,
            tags,
            replaygain,
            stream_serial,
            packet_reader: Some(packet_reader),
            audio_start_offset,
            file_size: file_size_hint,
            total_samples,
            nominal_bitrate,
            decoder: None,
            decode_scratch: Vec::new(),
            // Unified discard model: counter starts at 0, target is
            // `pre_skip`, so `decode_next` drops the first `pre_skip`
            // samples before returning anything to the caller.
            next_sample_abs_pos: 0,
            target_abs_pos: head.pre_skip as u64,
            page_index: None,
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

    pub(crate) fn replaygain(&self) -> ReplayGainInfo {
        self.replaygain
    }

    /// Pull the next Opus packet, decode it into the caller's interleaved
    /// float buffer, and return `(samples_per_ch, bytes_consumed)`.
    /// Returns `Ok((0, 0))` at end-of-stream — `bytes_consumed` is forced
    /// to 0 at EOF regardless of how many pre-roll packets the call drained
    /// before hitting end-of-stream (HLD §4.2 contract; the C++ shim never
    /// reads the value when samples == 0, but the simpler invariant is
    /// worth preserving).
    ///
    /// `bytes_consumed` is the sum of `pkt.data.len()` for every Ogg packet
    /// read by this call — including packets silently dropped because their
    /// samples fell entirely within the pre-skip / post-seek pre-roll window
    /// — so the caller can compute an instantaneous bitrate that accounts
    /// for every byte of compressed payload pulled from the stream.
    ///
    /// Discard logic is unified across pre-skip (first call) and post-seek
    /// pre-roll (after `seek`): we drop samples until `next_sample_abs_pos`
    /// reaches `target_abs_pos`. If a whole packet is consumed by the
    /// discard the reader loops to the next one rather than returning zero
    /// — pre-skip + pre-roll combined can exceed a 2.5 ms frame.
    pub(crate) fn decode_next(
        &mut self,
        out_interleaved: &mut [f32],
        max_samples_per_ch: usize,
    ) -> Result<(i32, u64), ReaderError> {
        // Test-only panic injection. Compiled out unless the `test-panic`
        // feature is on; see `lib.rs::ropus_fb2k_test_set_panic_flag` for
        // the sibling FFI hook the integration test uses to arm it.
        #[cfg(feature = "test-panic")]
        {
            if crate::test_panic_should_fire() {
                panic!("ropus-fb2k test-panic hook fired in decode_next");
            }
        }

        let channels = self.head.channels as usize;
        // Lazy-init the low-level decoder AND the scratch buffer together.
        // Deferring until the first decode keeps open() cheap for fb2k
        // library scans that never decode; re-using `decode_scratch` across
        // calls keeps the audio-thread path allocation-free.
        if self.decoder.is_none() {
            let dec = OpusDecoder::new(OPUS_SAMPLE_RATE_HZ, channels as i32).map_err(|code| {
                ReaderError::InvalidStream(format!("OpusDecoder init failed (code {code})"))
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

        // Accumulated encoded payload bytes consumed by this call. Includes
        // every packet read — both the one whose samples we ultimately return
        // and any earlier packets fully consumed by the pre-skip / post-seek
        // pre-roll discard window. Reported back so the C++ shim can compute
        // a live VBR bitrate that doesn't drop pre-roll-only callbacks (HLD
        // §4.3 "Rust side").
        let mut bytes_consumed: u64 = 0;

        loop {
            let pkt = packet_reader.read_packet().map_err(ogg_err_to_reader)?;
            let Some(pkt) = pkt else {
                // Clean EOF. Per HLD §4.2 the EOF return reports
                // `bytes_consumed == 0`, even if this call had already
                // accumulated payload from pre-roll packets that all got
                // discarded before EOF was hit. The C++ shim never reads
                // the value when samples == 0 anyway, but the simpler
                // contract is worth preserving — and it pins the test that
                // would otherwise still pass if we leaked the accumulator.
                return Ok((0, 0));
            };
            bytes_consumed += pkt.data.len() as u64;

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

            // A zero-sample packet is not a normal discard case — it means
            // the decoder produced nothing for a non-empty packet, which
            // would cause the loop to spin silently draining packets. Fail
            // fast instead.
            if decoded == 0 {
                return Err(ReaderError::InvalidStream(
                    "decoder returned 0 samples for a live packet".into(),
                ));
            }

            // Unified discard: the samples just produced cover absolute
            // positions [next_sample_abs_pos, next_sample_abs_pos+decoded).
            // Drop leading samples while we're below target_abs_pos.
            let discard = if self.next_sample_abs_pos < self.target_abs_pos {
                let gap = self.target_abs_pos - self.next_sample_abs_pos;
                (gap as usize).min(decoded)
            } else {
                0
            };
            let kept = decoded - discard;

            // Advance the counter by *every* sample the decoder produced —
            // both discarded and kept. The counter models the decoder's
            // absolute output position, which is what the next seek wants.
            self.next_sample_abs_pos = self.next_sample_abs_pos.saturating_add(decoded as u64);

            if kept == 0 {
                // Packet entirely consumed by discard — grab the next one.
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

            let src_start = discard * channels;
            let src_end = src_start + kept * channels;
            out_interleaved[..kept * channels].copy_from_slice(&scratch[src_start..src_end]);

            return Ok((kept as i32, bytes_consumed));
        }
    }

    // -----------------------------------------------------------------
    // Seek + page index (HLD §4.2 / §4.3)
    // -----------------------------------------------------------------

    /// Seek the decoder to a per-channel sample position (48 kHz, post-pre-skip).
    ///
    /// Behaviour:
    /// 1. Reject *any* seek (including `0`) on unseekable streams — pretending
    ///    `seek(0)` works without actually rewinding would silently mislead
    ///    callers that already pumped `decode_next`. The HLD §5.2
    ///    `decode_can_seek` will return false for these streams via the C++
    ///    shim, so this is the right contract.
    /// 2. For seekable streams of unknown duration (`total_samples == 0`),
    ///    accept only `sample_pos == 0` and rewind to audio start; non-zero
    ///    targets return `InvalidStream`.
    /// 3. Otherwise: clamp `sample_pos` to `[0, total_samples]`. If the
    ///    target lands inside the pre-skip window, short-circuit straight
    ///    to a rewind-to-audio-start (no page index walk needed).
    /// 4. Lazily build the page index if it isn't already.
    /// 5. Convert to granule (pre-skip-inclusive), rewind 80 ms, binary-search
    ///    the index for the largest entry with `start_granule <= rewind_to`.
    ///    If none, pick the first entry.
    /// 6. Seek the underlying reader to that page's byte offset and reset
    ///    the Opus decoder (OPUS_RESET_STATE semantics).
    /// 7. Set `next_sample_abs_pos = entry.start_granule` and
    ///    `target_abs_pos = sample_pos + pre_skip`. `decode_next` then
    ///    silently discards the pre-roll before returning real audio.
    pub(crate) fn seek(&mut self, sample_pos: u64) -> Result<(), ReaderError> {
        // Unseekable streams have no `seek` IO callback; we surface this as
        // InvalidStream (→ -4) rather than Unsupported (→ -5) because by
        // this point we've already accepted the stream via open(): the fb2k
        // contract for `exception_io_unsupported_format` is "fall through to
        // next input decoder", which is the wrong signal once we own the
        // stream. We reject *every* target — including 0 — because a
        // `seek(0)` that silently does nothing after decode_next has already
        // run is a contract violation: the caller thinks they rewound but
        // didn't. The C++ shim's `decode_can_seek` returns false here, so
        // fb2k won't ever issue this call in practice.
        if self.file_size.is_none() {
            return Err(ReaderError::InvalidStream(
                "cannot rewind unseekable stream".into(),
            ));
        }

        // Seekable but unknown duration: only seek-to-zero is meaningful.
        // Unlike the unseekable branch we *do* honour `sample_pos == 0` here
        // because the underlying IO can actually rewind us back to audio
        // start, so the operation has real effect.
        if self.total_samples == 0 {
            if sample_pos != 0 {
                return Err(ReaderError::InvalidStream(
                    "stream duration unknown; seek only to 0 supported".into(),
                ));
            }
            return self.rewind_to_audio_start();
        }

        // Clamp to the legal per-channel sample range. fb2k sometimes seeks
        // past the end on a scrub; HLD §4.2 says we clamp rather than error.
        let sample_pos = sample_pos.min(self.total_samples);

        // Granule units (pre-skip-inclusive, matches Ogg page granules).
        let pre_skip = self.head.pre_skip as u64;
        let target_granule = sample_pos.saturating_add(pre_skip);

        // Early-file shortcut: if the user is seeking somewhere inside the
        // pre-skip preamble, the post-rewind state is identical to a fresh
        // open — we'd land at the very first audio page and discard up to
        // `pre_skip`. Skip the index walk in that case (saves the disk-walk
        // entirely on a `seek(0)` that hasn't seeked before).
        if target_granule <= pre_skip {
            return self.rewind_to_audio_start();
        }

        // Saturation is intentional for early-file seeks: when
        // `target_granule < PRE_ROLL_SAMPLES` the rewind point floors at 0,
        // and the page-index search below correctly picks the very first
        // entry in that case.
        let rewind_to = target_granule.saturating_sub(PRE_ROLL_SAMPLES);

        // Build the index on demand.
        if self.page_index.is_none() {
            self.build_page_index()?;
        }
        let index = self
            .page_index
            .as_ref()
            .expect("page_index populated above");
        if index.is_empty() {
            return Err(ReaderError::InvalidStream(
                "seek: page index is empty".into(),
            ));
        }

        // Largest entry with start_granule <= rewind_to. `partition_point`
        // returns the insertion index of the first entry where the
        // predicate fails; the last satisfying entry is at `pp - 1`. If
        // all entries have start_granule > rewind_to we fall back to
        // index[0] (rewind to the very start).
        let pp = index.partition_point(|e| e.start_granule <= rewind_to);
        let start_page = if pp == 0 { index[0] } else { index[pp - 1] };

        // Take the PacketReader apart so we can reposition the bare reader,
        // then re-seat a fresh PacketReader at the new offset. A fresh
        // reader + `seek_bytes` flips the `has_seeked` flag the ogg crate
        // needs to tolerate mid-stream page reads.
        let packet_reader = self
            .packet_reader
            .take()
            .expect("packet_reader is always Some between open and drop");
        let mut bare_reader = packet_reader.into_inner();
        bare_reader
            .seek(SeekFrom::Start(start_page.byte_offset))
            .map_err(classify_io_error)?;
        let mut fresh = PacketReader::new(bare_reader);
        fresh
            .seek_bytes(SeekFrom::Start(start_page.byte_offset))
            .map_err(classify_io_error)?;
        self.packet_reader = Some(fresh);

        // OPUS_RESET_STATE semantics. If the decoder hasn't been created
        // yet (seek-before-first-decode), there's nothing to reset — the
        // lazy-init path in `decode_next` will fresh-build it next time.
        if let Some(dec) = self.decoder.as_mut() {
            dec.reset();
        }

        // Wire up the unified counter. The next decoded sample's absolute
        // position is `start_page.start_granule`; we want the caller to
        // start seeing samples at absolute granule `target_granule`.
        self.next_sample_abs_pos = start_page.start_granule;
        self.target_abs_pos = target_granule;

        Ok(())
    }

    /// Re-seat the reader at the first audio page and clear the decoder so
    /// the next `decode_next` produces samples starting from absolute
    /// granule 0 (which the unified discard logic then trims down to
    /// `pre_skip`). Shared by the main seek path's early-file shortcut and
    /// the `total_samples == 0` + `sample_pos == 0` branch.
    fn rewind_to_audio_start(&mut self) -> Result<(), ReaderError> {
        // Same reposition pattern M2 established for the post-reverse-scan
        // re-seat: take the bare reader out of the PacketReader, seek it,
        // then reseat a fresh PacketReader (the ogg crate's `has_seeked`
        // flag must be flipped via `seek_bytes` for mid-stream reads to
        // tolerate the move).
        let packet_reader = self
            .packet_reader
            .take()
            .expect("packet_reader is always Some between open and drop");
        let mut bare_reader = packet_reader.into_inner();
        bare_reader
            .seek(SeekFrom::Start(self.audio_start_offset))
            .map_err(classify_io_error)?;
        let mut fresh = PacketReader::new(bare_reader);
        fresh
            .seek_bytes(SeekFrom::Start(self.audio_start_offset))
            .map_err(classify_io_error)?;
        self.packet_reader = Some(fresh);

        // OPUS_RESET_STATE if the decoder exists; otherwise the lazy-init
        // path in `decode_next` will build a fresh one.
        if let Some(dec) = self.decoder.as_mut() {
            dec.reset();
        }

        // Match the open()-time state: the decoder hasn't produced anything
        // yet (next_sample_abs_pos = 0), and the discard target is the
        // pre-skip count so the first `pre_skip` decoded samples are
        // dropped before the caller sees real audio.
        self.next_sample_abs_pos = 0;
        self.target_abs_pos = self.head.pre_skip as u64;

        Ok(())
    }

    /// Walk Ogg pages from `audio_start_offset` to EOF, recording
    /// `(start_granule, byte_offset)` for every page whose serial matches
    /// ours. Lazily called on first seek.
    ///
    /// Preserves the current `PacketReader`'s view by seating a fresh one
    /// over the same bare reader at the same byte position on return.
    fn build_page_index(&mut self) -> Result<(), ReaderError> {
        let file_size = self
            .file_size
            .ok_or_else(|| ReaderError::Unsupported("stream is not seekable".into()))?;

        let packet_reader = self
            .packet_reader
            .take()
            .expect("packet_reader is always Some between open and drop");
        let mut bare_reader = packet_reader.into_inner();

        let resume_pos = bare_reader
            .stream_position()
            .unwrap_or(self.audio_start_offset);

        let index_result = scan_pages(
            &mut bare_reader,
            self.audio_start_offset,
            file_size,
            self.stream_serial,
        );

        // CRITICAL: always re-seat a PacketReader regardless of whether
        // the post-scan seek-restore succeeds. If we failed here and left
        // `self.packet_reader = None`, a retry of `seek` would panic on
        // `take().expect(...)` inside `ffi_guard!`, masking the real
        // aborted-state error (caller would see -6 INTERNAL instead of
        // -3 ABORTED). Best-effort restore to `resume_pos`; if that
        // fails, fall back to `audio_start_offset` so subsequent calls
        // at least see a valid-looking stream position. The caller still
        // gets the original error (abort / IO).
        let _ = bare_reader.seek(SeekFrom::Start(resume_pos));
        let mut fresh = PacketReader::new(bare_reader);
        let _ = fresh.seek_bytes(SeekFrom::Start(resume_pos));
        self.packet_reader = Some(fresh);

        self.page_index = Some(index_result?);
        Ok(())
    }
}

/// Walk Ogg pages between `start_offset` and `file_size`, recording
/// `(start_granule, byte_offset)` pairs for pages whose serial matches
/// `target_serial` and whose granule is not the "unknown" sentinel.
///
/// `start_granule` is the accumulated end-granule of the *previous* matching
/// page (0 for the first). This matches the semantics `seek` wants:
/// "where is the decoder cursor BEFORE this page's packets are decoded?"
///
/// An IO error tagged with `AbortTag` (via `CallbackReader::check_abort`)
/// bubbles up as `ReaderError::Aborted`, so a long walk on a big file can
/// be cancelled mid-scan.
///
/// Malformed bytes past the last good page are treated as end-of-scan
/// rather than a hard error — consistent with `read_last_granule`'s
/// tolerance for truncated trailers. A partial index still yields correct
/// seeks within the indexed prefix.
fn scan_pages<R: Read + Seek>(
    reader: &mut R,
    start_offset: u64,
    file_size: u64,
    target_serial: u32,
) -> Result<Vec<PageIndexEntry>, ReaderError> {
    let mut entries: Vec<PageIndexEntry> = Vec::new();
    if start_offset >= file_size {
        return Ok(entries);
    }

    reader
        .seek(SeekFrom::Start(start_offset))
        .map_err(classify_io_error)?;

    let mut header = [0u8; OGG_HEADER_LEN];
    let mut offset = start_offset;
    let mut running_granule: u64 = 0;

    loop {
        if offset + OGG_HEADER_LEN as u64 > file_size {
            break;
        }

        match reader.read_exact(&mut header) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(classify_io_error(e)),
        }

        // Validate capture + stream_structure_version. We don't attempt
        // recapture — our walk starts at a known page boundary and moves
        // by exact page sizes, so a mismatch means truncation.
        if &header[..4] != OGG_CAPTURE || header[4] != 0 {
            break;
        }

        let absgp = u64::from_le_bytes([
            header[6], header[7], header[8], header[9], header[10], header[11], header[12],
            header[13],
        ]);
        let serial = u32::from_le_bytes([header[14], header[15], header[16], header[17]]);
        let page_segments = header[26] as usize;

        // Read the lacing table for payload length. `page_segments` is
        // 0..=255 so the `[u8; 255]` scratch always fits.
        let mut lacing = [0u8; 255];
        let lacing_slice = &mut lacing[..page_segments];
        match reader.read_exact(lacing_slice) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(classify_io_error(e)),
        }
        let payload_len: usize = lacing_slice.iter().map(|&b| b as usize).sum();
        let page_total = OGG_HEADER_LEN + page_segments + payload_len;

        // Only index pages of our stream. Chained-stream support would
        // detect a second OpusHead and stop here (HLD §2 out-of-scope).
        // Continuation pages (absgp == UNKNOWN_GRANULE) don't advance the
        // timeline on their own; we skip them without updating
        // `running_granule`.
        if serial == target_serial && absgp != UNKNOWN_GRANULE {
            entries.push(PageIndexEntry {
                start_granule: running_granule,
                byte_offset: offset,
            });
            running_granule = absgp;
        }

        let next_offset = offset + page_total as u64;
        if next_offset <= offset || next_offset > file_size {
            break;
        }
        reader
            .seek(SeekFrom::Start(next_offset))
            .map_err(classify_io_error)?;
        offset = next_offset;
    }

    Ok(entries)
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
            let serial = u32::from_le_bytes([buf[i + 14], buf[i + 15], buf[i + 16], buf[i + 17]]);
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
        return Err(ReaderError::InvalidStream("OpusHead magic missing".into()));
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
