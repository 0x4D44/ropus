//! Integration tests for the `ropus-fb2k` C ABI.
//!
//! Every test here drives the public C entry points (`ropus_fb2k_open`,
//! `ropus_fb2k_read_tags`, …) — the `open_rust` rlib helper was deleted
//! in an earlier review pass. Shared fixture + `MemIo` helpers live in
//! `tests/common/mod.rs`; each test pulls only what it needs from there.
//!
//! Fixtures are constructed by encoding a short silence stream via
//! `ropus::Encoder` into an in-memory Ogg container, which matches the
//! encode path used by `ropus-cli` — any future change there that breaks
//! our fixture is a change we want to notice.

mod common;

use std::io::Cursor;

use ogg::reading::PacketReader;
use ropus::OpusDecoder;

use common::{
    FIXTURE_STREAM_SERIAL, MemIo, build_opus_fixture, build_opus_fixture_audio_source,
    build_opus_fixture_audio_source_with_output_gain, build_opus_fixture_with_audio_packets,
    build_opus_fixture_with_final_granule, build_opus_head, last_error_string,
    minimal_opus_fixture, open_from_bytes, open_from_bytes_info_only, open_from_bytes_without_seek,
    opus_fixture_with_artist_alice, read_tags_collect, surround_family_fixture,
};

use ropus_fb2k::{
    ROPUS_FB2K_ABORTED, ROPUS_FB2K_BAD_ARG, ROPUS_FB2K_INVALID_STREAM, ROPUS_FB2K_IO,
    ROPUS_FB2K_UNSUPPORTED, RopusFb2kInfo,
};

// ---------------------------------------------------------------------------
// Garbage input is rejected as INVALID_STREAM.
// ---------------------------------------------------------------------------

#[test]
fn open_rejects_garbage() {
    let mut bytes = vec![0u8; 1024];
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(37).wrapping_add(11);
    }
    let (_io, handle) = open_from_bytes(bytes);
    assert!(handle.is_null(), "garbage must not parse");

    let code = unsafe { ropus_fb2k::ropus_fb2k_last_error_code() };
    assert_eq!(
        code,
        ROPUS_FB2K_INVALID_STREAM,
        "garbage must surface as INVALID_STREAM (last_error={:?})",
        last_error_string()
    );
}

// ---------------------------------------------------------------------------
// Truncating a *valid* fixture at various points must surface a clean
// negative status — never a panic (`-6 INTERNAL`), never UB. Every cut in
// this test lands inside an Ogg page (mid-header or mid-body), so the `ogg`
// crate's short-page detection catches the problem before our OpusHead /
// OpusTags parser ever sees a packet. Acceptable codes are `-4 INVALID_STREAM`
// (parser detected corruption) or `-2 IO` (read returned short before the
// parser could react).
//
// The `tags::read_u32_le` / `tags::take` length-check discipline (vendor
// or comment length larger than the packet) is NOT exercised by these cuts
// — the ogg crate rejects the short page first. Coverage for that parser
// path lives in:
//   * `corrupt_opus_tags_body_rejected_invalid_stream` below, which wraps a
//     hand-crafted malformed body in a valid Ogg page so the parser sees it;
//   * `src/tags.rs::tests::rejects_truncated_comment_length` at the unit
//     level.
// ---------------------------------------------------------------------------

#[test]
fn open_rejects_truncated_valid_fixture() {
    // Use a multi-packet fixture so cuts hit a representative spread of
    // boundaries: inside OpusHead, between OpusHead and OpusTags, inside
    // OpusTags vendor-length, between OpusTags and the audio pages, and
    // mid-audio. Each cut point exercises a different parse step.
    // Smaller fixture is enough — we want cuts that hit the OpusHead
    // page (bytes ~0..47), the OpusTags page (~47..103), and beyond into
    // the audio area where the parser has already accepted the headers
    // but the tail is missing. Five packets of silence pack into a single
    // ~50-byte audio page in practice, so total fixture is ~150 bytes.
    let full = build_opus_fixture_with_audio_packets("ropus-fb2k-test", &[], 5, Some(312));
    let len = full.len();
    // Cuts spread across all parser stages. We deliberately do *not*
    // include cuts that lop off only the trailing 1-4 bytes: the reader
    // is intentionally lenient about a torn tail (the OpusHead +
    // OpusTags pages plus a complete audio page are enough to open with
    // valid metadata; the missing bytes just shorten the apparent
    // duration). Tests for the parser-discipline cuts are what matter.
    let cuts = [
        10usize, // mid first Ogg page header
        27,      // immediately after the first Ogg page header
        30,      // mid OpusHead packet payload (inside the 19 body bytes)
        50,      // around the OpusHead/OpusTags page boundary
        60,      // mid OpusTags Ogg page header
        80,      // mid OpusTags vendor / comment-count area
    ];

    for &cut in &cuts {
        if cut == 0 || cut >= len {
            continue;
        }
        let truncated = full[..cut].to_vec();
        let (_io, handle) = open_from_bytes(truncated);
        if !handle.is_null() {
            unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
            panic!("cut={cut} (fixture len={len}) must not parse as a valid stream");
        }
        let code = unsafe { ropus_fb2k::ropus_fb2k_last_error_code() };
        assert!(
            code == ROPUS_FB2K_INVALID_STREAM || code == ROPUS_FB2K_IO,
            "cut={cut} got code={code}, expected INVALID_STREAM or IO \
             (last_error={:?})",
            last_error_string()
        );
    }
}

// ---------------------------------------------------------------------------
// A valid Ogg page carrying a malformed `OpusTags` body (vendor_length wildly
// exceeds the packet size) must surface INVALID_STREAM from the parser's
// length-check discipline (`tags::read_u32_le` / `tags::take`), not from Ogg
// framing. Complements `open_rejects_truncated_valid_fixture` (which only
// tests page-level short reads) and the unit test
// `src/tags.rs::tests::rejects_truncated_comment_length`.
// ---------------------------------------------------------------------------

#[test]
fn corrupt_opus_tags_body_rejected_invalid_stream() {
    use ogg::writing::{PacketWriteEndInfo, PacketWriter};

    // Hand-craft a malformed OpusTags packet body: magic + `vendor_length =
    // u32::MAX - 5` + no actual vendor bytes. `tags::read_u32_le` reads the
    // length, `tags::take` then checks `cur.len() < n` and returns
    // `TagError::Truncated`, which maps to INVALID_STREAM.
    let mut bad_tags = Vec::with_capacity(16);
    bad_tags.extend_from_slice(b"OpusTags");
    bad_tags.extend_from_slice(&(u32::MAX - 5).to_le_bytes());

    // Wrap the page-1 OpusHead and page-2 corrupt-body packet in valid Ogg
    // framing via the same `ogg::PacketWriter` the healthy fixtures use, so
    // the page headers and CRCs are correct — the only malformation is
    // inside the OpusTags packet payload.
    let mut out = Vec::with_capacity(256);
    let mut writer = PacketWriter::new(&mut out);
    writer
        .write_packet(
            build_opus_head(2, 48_000, 312),
            FIXTURE_STREAM_SERIAL,
            PacketWriteEndInfo::EndPage,
            0,
        )
        .expect("write OpusHead page");
    writer
        .write_packet(
            bad_tags,
            FIXTURE_STREAM_SERIAL,
            PacketWriteEndInfo::EndStream,
            0,
        )
        .expect("write corrupt OpusTags page");
    drop(writer);

    let (_io, handle) = open_from_bytes(out);
    assert!(
        handle.is_null(),
        "malformed OpusTags body must not parse as a valid stream"
    );
    let code = unsafe { ropus_fb2k::ropus_fb2k_last_error_code() };
    assert_eq!(
        code,
        ROPUS_FB2K_INVALID_STREAM,
        "corrupt vendor length must surface as INVALID_STREAM (last_error={:?})",
        last_error_string()
    );
}

// ---------------------------------------------------------------------------
// Valid file populates OpusHead fields in `RopusFb2kInfo`.
// ---------------------------------------------------------------------------

#[test]
fn open_parses_header() {
    let (_io, handle) = open_from_bytes(minimal_opus_fixture().to_vec());
    assert!(!handle.is_null(), "valid fixture must parse");

    let mut info = zeroed_info();
    let rc = unsafe { ropus_fb2k::ropus_fb2k_get_info(handle, &mut info) };
    assert_eq!(rc, 0);
    assert_eq!(info.sample_rate, 48_000);
    assert_eq!(info.channels, 2);
    assert!(
        info.pre_skip > 0,
        "encoder always reports non-zero pre_skip (typically 312)"
    );
    assert!(
        info.rg_track_gain.is_nan(),
        "ReplayGain stays NaN until tag mapping lands"
    );

    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

// ---------------------------------------------------------------------------
// Valid file surfaces VENDOR (synthetic) + user comments via read_tags.
// ---------------------------------------------------------------------------

#[test]
fn open_parses_tags() {
    let (_io, handle) = open_from_bytes(opus_fixture_with_artist_alice().to_vec());
    assert!(!handle.is_null(), "valid fixture must parse");

    let pairs = read_tags_collect(handle);
    assert!(
        pairs
            .iter()
            .any(|(k, v)| k == "VENDOR" && v == "ropus-fb2k-test"),
        "expected synthetic VENDOR entry in {pairs:?}"
    );
    assert!(
        pairs.iter().any(|(k, v)| k == "ARTIST" && v == "Alice"),
        "expected ARTIST=Alice in {pairs:?}"
    );

    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

// ---------------------------------------------------------------------------
// Separate fixture: zero user comments is also fine.
// ---------------------------------------------------------------------------

#[test]
fn empty_comment_block_reports_only_vendor() {
    let (_io, handle) = open_from_bytes(minimal_opus_fixture().to_vec());
    assert!(!handle.is_null(), "valid fixture must parse");

    let pairs = read_tags_collect(handle);
    assert_eq!(
        pairs.len(),
        1,
        "only the synthetic VENDOR entry; got {pairs:?}"
    );
    assert_eq!(pairs[0].0, "VENDOR");

    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

// ---------------------------------------------------------------------------
// Cover-art blobs parse successfully but are filtered from the callback.
// ---------------------------------------------------------------------------

#[test]
fn metadata_block_picture_is_filtered_from_callback() {
    let bytes = build_opus_fixture(
        "ropus-fb2k-test",
        &[
            ("METADATA_BLOCK_PICTURE", "base64-of-a-huge-image"),
            ("ARTIST", "Alice"),
        ],
    );
    let (_io, handle) = open_from_bytes(bytes);
    assert!(!handle.is_null(), "valid fixture must parse");

    let pairs = read_tags_collect(handle);
    assert!(
        pairs.iter().all(|(k, _)| k != "METADATA_BLOCK_PICTURE"),
        "METADATA_BLOCK_PICTURE must be filtered at the callback boundary; got {pairs:?}"
    );
    // The surrounding comments still make it through.
    assert!(
        pairs.iter().any(|(k, v)| k == "ARTIST" && v == "Alice"),
        "ARTIST should survive filtering; got {pairs:?}"
    );

    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

// ---------------------------------------------------------------------------
// Family 1 (surround) OpusHead is rejected as UNSUPPORTED.
// ---------------------------------------------------------------------------

#[test]
fn open_rejects_mapping_family_1() {
    let (_io, handle) = open_from_bytes(surround_family_fixture());
    assert!(handle.is_null(), "mapping family 1 must reject");

    let code = unsafe { ropus_fb2k::ropus_fb2k_last_error_code() };
    assert_eq!(
        code,
        ROPUS_FB2K_UNSUPPORTED,
        "family 1 must surface as UNSUPPORTED (last_error={:?})",
        last_error_string()
    );
}

// ---------------------------------------------------------------------------
// Abort during open: first `read` returns, then the abort callback trips
// the `CallbackReader` guard and open bails with ABORTED.
// ---------------------------------------------------------------------------

#[test]
fn abort_halts_open() {
    let io = MemIo::new(minimal_opus_fixture().to_vec()).with_abort_flag();
    let fb2k_io = io.io();
    let handle = unsafe { ropus_fb2k::ropus_fb2k_open(&fb2k_io, 0) };
    assert!(handle.is_null(), "open with abort flagged must return null");

    let msg = last_error_string();
    assert!(
        msg.contains("aborted"),
        "last-error should mention 'aborted', got {msg:?}"
    );
    let code = unsafe { ropus_fb2k::ropus_fb2k_last_error_code() };
    assert_eq!(code, ROPUS_FB2K_ABORTED);
}

// ---------------------------------------------------------------------------
// Seek-to-zero on a valid seekable fixture succeeds (pins M3 semantics).
// ---------------------------------------------------------------------------

#[test]
fn seek_to_zero_succeeds() {
    let (_io, handle) = open_from_bytes(minimal_opus_fixture().to_vec());
    assert!(!handle.is_null());

    let rc_seek = unsafe { ropus_fb2k::ropus_fb2k_seek(handle, 0) };
    assert_eq!(
        rc_seek,
        0,
        "seek(0) must succeed (last_error={:?})",
        last_error_string()
    );

    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

// ---------------------------------------------------------------------------
// INFO_ONLY flag is accepted and populates duration via reverse-scan.
// ---------------------------------------------------------------------------

#[test]
fn info_only_flag_is_accepted() {
    let (_io, handle) = open_from_bytes_info_only(minimal_opus_fixture().to_vec());
    assert!(
        !handle.is_null(),
        "ROPUS_FB2K_OPEN_INFO_ONLY must be accepted (last_error={:?})",
        last_error_string()
    );
    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

// ---------------------------------------------------------------------------
// `last_error_code` returns the right sentinel for UNSUPPORTED. Exercises
// the new FFI entry point explicitly.
// ---------------------------------------------------------------------------

#[test]
fn last_error_code_reports_unsupported_for_mapping_family_1() {
    let (_io, handle) = open_from_bytes(surround_family_fixture());
    assert!(handle.is_null());
    let code = unsafe { ropus_fb2k::ropus_fb2k_last_error_code() };
    assert_eq!(code, ROPUS_FB2K_UNSUPPORTED);
}

// ---------------------------------------------------------------------------
// decode_next: fb2k path matches a direct ropus decode of the same packets,
// pre-skip trimmed consistently on both sides.
// ---------------------------------------------------------------------------

#[test]
fn decode_wiring_matches_direct_ogg_path() {
    // Both arms use the same `ropus::OpusDecoder`, so this is a wiring /
    // pre-skip-trim test — NOT a codec oracle. Decoder correctness is
    // covered separately via the C-reference comparison harness in
    // `tests/harness/`.
    //
    // ~100 ms stereo = 5 × 20 ms packets. Small enough that both paths
    // finish quickly; large enough to exercise multiple packets in a row.
    let bytes = build_opus_fixture_with_audio_packets("ropus-fb2k-test", &[], 5, None);

    let path_a = decode_through_fb2k(bytes.clone());
    let path_b = decode_reference_direct(&bytes);

    assert_eq!(
        path_a.len(),
        path_b.len(),
        "post-pre-skip sample counts must match (fb2k={}, reference={})",
        path_a.len(),
        path_b.len()
    );
    assert_eq!(
        path_a, path_b,
        "fb2k decode stream must equal direct OpusDecoder stream bit-for-bit"
    );
    assert!(
        !path_a.is_empty(),
        "fixture must produce some decoded samples"
    );
}

// ---------------------------------------------------------------------------
// A partial final frame must stop at the stream's absolute EOS granule rather
// than exposing the codec padding that follows it.
// ---------------------------------------------------------------------------

#[test]
fn decode_stops_at_partial_final_granule() {
    const PRE_SKIP: u16 = 312;
    const EOS_GRANULE: u64 = 1_440; // 960 + 480: half of the second frame.
    let bytes = build_opus_fixture_with_final_granule(
        "ropus-fb2k-test",
        &[],
        2,
        Some(PRE_SKIP),
        EOS_GRANULE,
    );

    let (_io, handle) = open_from_bytes(bytes.clone());
    assert!(!handle.is_null(), "partial-final-frame fixture must open");
    let mut info = zeroed_info();
    assert_eq!(
        unsafe { ropus_fb2k::ropus_fb2k_get_info(handle, &mut info) },
        0
    );
    assert_eq!(info.total_samples, (EOS_GRANULE - PRE_SKIP as u64));
    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };

    let decoded = decode_through_fb2k(bytes);
    assert_eq!(
        decoded.len(),
        (EOS_GRANULE - PRE_SKIP as u64) as usize * 2,
        "decoded PCM must stop at EOS instead of including final-frame padding"
    );
}

#[test]
fn open_rejects_final_granule_before_pre_skip() {
    let bytes = build_opus_fixture_with_final_granule("ropus-fb2k-test", &[], 1, Some(312), 100);
    let (_io, handle) = open_from_bytes(bytes);
    assert!(handle.is_null(), "impossible EOS granule must be rejected");
    assert_eq!(
        unsafe { ropus_fb2k::ropus_fb2k_last_error_code() },
        ROPUS_FB2K_INVALID_STREAM
    );
}

// ---------------------------------------------------------------------------
// decode_next returns 0 at EOF after all packets drained.
// ---------------------------------------------------------------------------

#[test]
fn decode_returns_zero_at_eof() {
    let (_io, handle) = open_from_bytes(minimal_opus_fixture().to_vec());
    assert!(!handle.is_null());

    let mut scratch = vec![0f32; 5760 * 2];
    let mut bytes_consumed = 0u64;
    // Drain every non-zero return first.
    loop {
        let rc = unsafe {
            ropus_fb2k::ropus_fb2k_decode_next(
                handle,
                scratch.as_mut_ptr(),
                5760,
                &mut bytes_consumed,
            )
        };
        assert!(
            rc >= 0,
            "decode_next must not error before EOF (got {rc}, last_error={:?})",
            last_error_string()
        );
        if rc == 0 {
            break;
        }
    }
    // Subsequent calls keep returning 0.
    let rc = unsafe {
        ropus_fb2k::ropus_fb2k_decode_next(handle, scratch.as_mut_ptr(), 5760, &mut bytes_consumed)
    };
    assert_eq!(rc, 0, "post-EOF decode_next stays at 0");

    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

// ---------------------------------------------------------------------------
// decode_next's `out_bytes_consumed` parameter, summed across every call until
// EOF, must equal the total encoded packet-payload bytes the underlying Ogg
// stream carries (i.e. every audio packet's `data.len()` from page 3 onward,
// skipping the OpusHead + OpusTags header packets). Pins HLD §4.3: the Rust
// side reports raw bytes-consumed and the C++ side does the EWMA.
// ---------------------------------------------------------------------------

#[test]
fn decode_next_reports_bytes_consumed_correctly() {
    // Multi-packet fixture so the sum is non-trivially > 0 and exercises the
    // accumulation across the decode_next loop.
    let bytes = build_opus_fixture_with_audio_packets("ropus-fb2k-test", &[], 20, Some(312));

    // Path A: drive the FFI; sum every `bytes_consumed` written by
    // decode_next. Includes the EOF call (which writes 0).
    let (_io, handle) = open_from_bytes(bytes.clone());
    assert!(
        !handle.is_null(),
        "fixture must open (last_error={:?})",
        last_error_string()
    );
    let mut buf = vec![0f32; 5760 * 2];
    let mut ffi_sum: u64 = 0;
    let eof_bytes_consumed: u64;
    loop {
        let mut bytes_consumed = 0u64;
        let rc = unsafe {
            ropus_fb2k::ropus_fb2k_decode_next(handle, buf.as_mut_ptr(), 5760, &mut bytes_consumed)
        };
        assert!(
            rc >= 0,
            "decode_next must not error before EOF (got {rc}, last_error={:?})",
            last_error_string()
        );
        if rc == 0 {
            // Pin the HLD §4.2 EOF contract: at end-of-stream the EOF call
            // itself must report bytes_consumed == 0, regardless of how
            // many pre-roll packets it drained before hitting EOF. A
            // running sum alone would mask a regression where the EOF
            // return leaked the accumulator.
            eof_bytes_consumed = bytes_consumed;
            break;
        }
        ffi_sum += bytes_consumed;
    }
    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
    assert_eq!(
        eof_bytes_consumed, 0,
        "EOF return must zero bytes_consumed (HLD §4.2)"
    );

    // Path B: parse the same bytes via `ogg::PacketReader` directly; sum the
    // `data.len()` of every packet AFTER OpusHead + OpusTags. That's the
    // ground truth for "encoded payload bytes the decoder consumes".
    let mut pr = PacketReader::new(Cursor::new(bytes));
    let _head = pr.read_packet().expect("read OpusHead").expect("OpusHead");
    let _tags = pr.read_packet().expect("read OpusTags").expect("OpusTags");
    let mut direct_sum: u64 = 0;
    while let Some(pkt) = pr.read_packet().expect("packet read") {
        direct_sum += pkt.data.len() as u64;
    }

    assert!(
        direct_sum > 0,
        "fixture must contain audio packets (got direct_sum=0)"
    );
    assert_eq!(
        ffi_sum, direct_sum,
        "bytes_consumed summed across decode_next calls ({ffi_sum}) must equal \
         the Ogg-level payload total ({direct_sum})"
    );
}

// ---------------------------------------------------------------------------
// abort callback mid-decode surfaces as -3 ABORTED.
// ---------------------------------------------------------------------------

#[test]
fn decode_propagates_abort() {
    // Build a fixture and let `open` complete first (it performs several
    // read calls: 2 pages + one 128 KiB reverse-scan), then flip `abort`
    // *after* open so decode_next is the thing that trips. Directly
    // asserting on a fixed `abort_after_n_reads` value was flaky — the
    // reverse-scan's read-count depends on file size and buffer chunking.
    let bytes = build_opus_fixture_with_audio_packets("ropus-fb2k-test", &[], 20, None);
    let io = MemIo::new(bytes);
    let fb2k_io = io.io();
    let handle = unsafe { ropus_fb2k::ropus_fb2k_open(&fb2k_io, 0) };
    assert!(
        !handle.is_null(),
        "open must succeed (last_error={:?})",
        last_error_string()
    );

    // Flip the abort flag now. The next decode call will poll it on the
    // very first `check_abort` inside `CallbackReader` and unwind with -3.
    io.set_aborting();

    // Snapshot IO-layer call counts so we can prove the abort path actually
    // exercised the IO layer. If internal buffering ever changed to the
    // point where `decode_next` returned without polling the IO at all,
    // neither `read_calls` nor `abort_calls` would move. We check the
    // union so the test stays meaningful regardless of whether the abort
    // trips in `check_abort` (before a byte-read) or in the read callback
    // itself.
    let reads_before = io.read_calls();
    let abort_calls_before = io.abort_calls();

    let mut scratch = vec![0f32; 5760 * 2];
    let mut bytes_consumed = 0u64;
    let rc = unsafe {
        ropus_fb2k::ropus_fb2k_decode_next(handle, scratch.as_mut_ptr(), 5760, &mut bytes_consumed)
    };
    assert_eq!(
        rc, ROPUS_FB2K_ABORTED,
        "decode_next must surface -3 ABORTED"
    );

    let code = unsafe { ropus_fb2k::ropus_fb2k_last_error_code() };
    assert_eq!(code, ROPUS_FB2K_ABORTED);

    assert!(
        io.read_calls() > reads_before || io.abort_calls() > abort_calls_before,
        "abort test vacuous — decode_next returned without polling the IO layer"
    );

    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

// ---------------------------------------------------------------------------
// decode_next rejects a buffer smaller than 120 ms (5760 samples/ch).
// ---------------------------------------------------------------------------

#[test]
fn decode_rejects_small_buffer() {
    let (_io, handle) = open_from_bytes(minimal_opus_fixture().to_vec());
    assert!(!handle.is_null());

    let mut scratch = vec![0f32; 4096 * 2];
    let mut bytes_consumed = 0u64;
    let rc = unsafe {
        ropus_fb2k::ropus_fb2k_decode_next(handle, scratch.as_mut_ptr(), 4096, &mut bytes_consumed)
    };
    assert_eq!(rc, ROPUS_FB2K_BAD_ARG);

    let code = unsafe { ropus_fb2k::ropus_fb2k_last_error_code() };
    assert_eq!(code, ROPUS_FB2K_BAD_ARG);
    assert!(
        last_error_string().contains("5760"),
        "error message should mention the minimum; got {:?}",
        last_error_string()
    );

    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

// ---------------------------------------------------------------------------
// info populates total_samples via reverse-scan of the last granule.
// ---------------------------------------------------------------------------

#[test]
fn info_populates_total_samples() {
    // 20 × 20 ms packets = 400 ms. Final granule = 20 × 960 = 19 200.
    // We pin pre_skip to 312 (typical libopus lookahead) so the expected
    // value is exact rather than encoder-dependent.
    const PACKETS: usize = 20;
    const PRE_SKIP: u16 = 312;
    let bytes =
        build_opus_fixture_with_audio_packets("ropus-fb2k-test", &[], PACKETS, Some(PRE_SKIP));
    let (_io, handle) = open_from_bytes(bytes);
    assert!(
        !handle.is_null(),
        "fixture must parse (last_error={:?})",
        last_error_string()
    );

    let mut info = zeroed_info();
    let rc = unsafe { ropus_fb2k::ropus_fb2k_get_info(handle, &mut info) };
    assert_eq!(rc, 0);

    let expected = (PACKETS as u64) * 960 - PRE_SKIP as u64;
    assert_eq!(
        info.total_samples, expected,
        "total_samples must be granule - pre_skip"
    );

    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

#[test]
fn duration_ignores_fake_eos_header_inside_payload() {
    const PACKETS: usize = 20;
    const PRE_SKIP: u16 = 312;
    const FAKE_GRANULE: u64 = 9_999_999;

    let mut bytes =
        build_opus_fixture_with_audio_packets("ropus-fb2k-test", &[], PACKETS, Some(PRE_SKIP));

    // This has the right capture/version/serial and an EOS flag, but its
    // zero checksum is invalid. Embed it in the real final page payload after
    // updating that page's lacing and CRC; the old byte-by-byte scan trusted
    // the payload bytes as a new final page.
    let mut fake = Vec::with_capacity(27);
    fake.extend_from_slice(b"OggS");
    fake.push(0);
    fake.push(0x04);
    fake.extend_from_slice(&FAKE_GRANULE.to_le_bytes());
    fake.extend_from_slice(&FIXTURE_STREAM_SERIAL.to_le_bytes());
    fake.extend_from_slice(&0u32.to_le_bytes());
    fake.extend_from_slice(&0u32.to_le_bytes());
    fake.push(0);
    let page_start = bytes
        .windows(4)
        .rposition(|window| window == b"OggS")
        .expect("fixture has a final Ogg page");
    let segment_count = bytes[page_start + 26] as usize;
    assert!(segment_count > 0);
    let last_lacing = page_start + 27 + segment_count - 1;
    assert!(bytes[last_lacing] as usize + fake.len() <= u8::MAX as usize);
    bytes[last_lacing] += fake.len() as u8;
    bytes.extend_from_slice(&fake);
    set_ogg_page_crc(&mut bytes, page_start);

    let (_io, handle) = open_from_bytes(bytes);
    assert!(
        !handle.is_null(),
        "fixture must parse: {}",
        last_error_string()
    );
    let mut info = zeroed_info();
    assert_eq!(
        unsafe { ropus_fb2k::ropus_fb2k_get_info(handle, &mut info) },
        0
    );
    assert_eq!(
        info.total_samples,
        PACKETS as u64 * 960 - PRE_SKIP as u64,
        "invalid trailing page must not replace the real EOS granule"
    );
    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

fn set_ogg_page_crc(bytes: &mut [u8], page_start: usize) {
    bytes[page_start + 22..page_start + 26].fill(0);
    let mut crc = 0u32;
    for &byte in &bytes[page_start..] {
        crc ^= (byte as u32) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 {
                (crc << 1) ^ 0x04C1_1DB7
            } else {
                crc << 1
            };
        }
    }
    bytes[page_start + 22..page_start + 26].copy_from_slice(&crc.to_le_bytes());
}

// ---------------------------------------------------------------------------
// info populates nominal_bitrate within a tight window around the
// computed-from-fixture expected value.
// ---------------------------------------------------------------------------

#[test]
fn info_populates_nominal_bitrate() {
    // Pin the expected value to the exact formula the reader uses:
    // bits/sec = file_size * 8 * 48_000 / total_samples
    // where total_samples = PACKETS * 960 - pre_skip.
    //
    // A regression that returned `file_size` or `total_samples` directly,
    // or a stale sentinel, would miss this window by orders of magnitude.
    const PACKETS: usize = 20;
    const PRE_SKIP: u16 = 312;
    let bytes =
        build_opus_fixture_with_audio_packets("ropus-fb2k-test", &[], PACKETS, Some(PRE_SKIP));
    let file_size = bytes.len() as u64;
    let total_samples = (PACKETS as u64) * 960 - PRE_SKIP as u64;
    let expected = (file_size * 8 * 48_000 / total_samples) as i32;

    let (_io, handle) = open_from_bytes(bytes);
    assert!(!handle.is_null());

    let mut info = zeroed_info();
    let rc = unsafe { ropus_fb2k::ropus_fb2k_get_info(handle, &mut info) };
    assert_eq!(rc, 0);

    // 10% window. The fixture is deterministic (silence input, same encoder
    // settings) so in practice `info.nominal_bitrate == expected`, but the
    // window absorbs any future encoder tuning that changes packet sizes
    // slightly without hiding an order-of-magnitude regression.
    let low = expected - expected / 10;
    let high = expected + expected / 10;
    assert!(
        info.nominal_bitrate >= low && info.nominal_bitrate <= high,
        "nominal_bitrate {} outside expected window [{low}, {high}] \
         (file_size={file_size}, total_samples={total_samples})",
        info.nominal_bitrate
    );

    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

// ---------------------------------------------------------------------------
// Open is bounded by a small constant number of reads — we read the first
// two Ogg pages + a single 128-KiB reverse-scan for the last granule, NOT
// the whole file. An accidental regression to a full-file page-walk would
// produce O(file_size / buffer_size) reads and trip this ceiling.
// ---------------------------------------------------------------------------

#[test]
fn open_uses_bounded_reads() {
    // Observed < 10 in practice; 20 gives some slack for ogg buffer size
    // changes but catches an accidental full-file scan (which would read
    // hundreds of times on a 20-packet fixture).
    //
    // The INFO_ONLY flag is passed because the reverse-scan behaviour is
    // what's under test; today both open paths reverse-scan identically,
    // so the flag is incidental to the assertion (and the test will stay
    // meaningful when the flag starts differentiating).
    let bytes = build_opus_fixture_with_audio_packets("ropus-fb2k-test", &[], 20, Some(312));
    let (io, handle) = open_from_bytes_info_only(bytes);
    assert!(!handle.is_null());

    let reads = io.read_calls();
    assert!(
        reads < 20,
        "open read too many chunks: {reads} (expected < 20; a full-file scan would be hundreds)"
    );

    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

// ---------------------------------------------------------------------------
// An unseekable stream (no `seek` callback) still opens, but
// total_samples / nominal_bitrate fall back to the "unknown" sentinels.
// ---------------------------------------------------------------------------

#[test]
fn info_without_seek_returns_zero_total_samples() {
    let bytes = build_opus_fixture_with_audio_packets("ropus-fb2k-test", &[], 20, Some(312));
    let (_io, handle) = open_from_bytes_without_seek(bytes, 0);
    assert!(
        !handle.is_null(),
        "open must succeed on unseekable streams (last_error={:?})",
        last_error_string()
    );

    let mut info = zeroed_info();
    let rc = unsafe { ropus_fb2k::ropus_fb2k_get_info(handle, &mut info) };
    assert_eq!(rc, 0);
    assert_eq!(
        info.total_samples, 0,
        "unseekable stream reports zero duration"
    );
    assert_eq!(
        info.nominal_bitrate, -1,
        "unseekable stream reports unknown bitrate"
    );

    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

// ===========================================================================
// M3 — seek with pre-roll, ReplayGain
// ===========================================================================

/// Generate a 20 ms stereo packet of a 1 kHz sine at 0.5 amplitude starting
/// at packet index `i`. Each packet advances phase by 960 samples so the
/// waveform is continuous across packet boundaries.
fn sine_pcm_for_packet(i: usize) -> Vec<i16> {
    const FREQ_HZ: f32 = 1000.0;
    const SR_HZ: f32 = 48_000.0;
    const AMPL: f32 = 0.5 * i16::MAX as f32;
    let base = i * 960;
    let mut pcm = vec![0i16; 960 * 2];
    for n in 0..960 {
        let t = (base + n) as f32 / SR_HZ;
        let s = (2.0 * std::f32::consts::PI * FREQ_HZ * t).sin() * AMPL;
        let v = s as i16;
        pcm[n * 2] = v;
        pcm[n * 2 + 1] = v;
    }
    pcm
}

fn rms(samples: &[f32]) -> f32 {
    let mean_square = samples
        .iter()
        .map(|sample| f64::from(*sample) * f64::from(*sample))
        .sum::<f64>()
        / samples.len() as f64;
    mean_square.sqrt() as f32
}

fn decode_seek_window(bytes: Vec<u8>, sample_pos: u64, samples_per_channel: usize) -> Vec<f32> {
    let (_io, handle) = open_from_bytes(bytes);
    assert!(!handle.is_null(), "seek fixture failed to open");
    let rc_seek = unsafe { ropus_fb2k::ropus_fb2k_seek(handle, sample_pos) };
    assert_eq!(rc_seek, 0, "seek failed: {}", last_error_string());

    let mut out = Vec::with_capacity(samples_per_channel * 2);
    let mut buf = vec![0f32; 5760 * 2];
    while out.len() < samples_per_channel * 2 {
        let mut bytes_consumed = 0u64;
        let rc = unsafe {
            ropus_fb2k::ropus_fb2k_decode_next(handle, buf.as_mut_ptr(), 5760, &mut bytes_consumed)
        };
        assert!(rc > 0, "seek window ended early with {rc}");
        out.extend_from_slice(&buf[..rc as usize * 2]);
    }
    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
    out.truncate(samples_per_channel * 2);
    out
}

#[test]
fn opus_head_output_gain_applies_before_and_after_seek() {
    const AUDIO_PACKETS: usize = 30;
    const PRE_SKIP: u16 = 312;
    const SEEK_SAMPLE: u64 = 10_000;
    const WINDOW_SAMPLES: usize = 960;
    const POSITIVE_GAIN: i16 = 6 * 256;
    const NEGATIVE_GAIN: i16 = -6 * 256;

    let baseline = build_opus_fixture_audio_source_with_output_gain(
        "ropus-fb2k-test",
        &[],
        AUDIO_PACKETS,
        Some(PRE_SKIP),
        0,
        sine_pcm_for_packet,
    );
    let positive = build_opus_fixture_audio_source_with_output_gain(
        "ropus-fb2k-test",
        &[],
        AUDIO_PACKETS,
        Some(PRE_SKIP),
        POSITIVE_GAIN,
        sine_pcm_for_packet,
    );
    let negative = build_opus_fixture_audio_source_with_output_gain(
        "ropus-fb2k-test",
        &[],
        AUDIO_PACKETS,
        Some(PRE_SKIP),
        NEGATIVE_GAIN,
        sine_pcm_for_packet,
    );

    let baseline_full = decode_through_fb2k(baseline.clone());
    let positive_full = decode_through_fb2k(positive.clone());
    let negative_full = decode_through_fb2k(negative.clone());
    let baseline_rms = rms(&baseline_full);
    assert!(
        rms(&positive_full) > baseline_rms * 1.35,
        "+6 dB OpusHead gain did not amplify decoded audio: baseline={baseline_rms}, gained={}",
        rms(&positive_full)
    );
    assert!(
        rms(&negative_full) < baseline_rms * 0.75,
        "-6 dB OpusHead gain did not attenuate decoded audio: baseline={baseline_rms}, gained={}",
        rms(&negative_full)
    );

    let baseline_seek = decode_seek_window(baseline, SEEK_SAMPLE, WINDOW_SAMPLES);
    let positive_seek = decode_seek_window(positive, SEEK_SAMPLE, WINDOW_SAMPLES);
    let negative_seek = decode_seek_window(negative, SEEK_SAMPLE, WINDOW_SAMPLES);
    let baseline_seek_rms = rms(&baseline_seek);
    assert!(
        rms(&positive_seek) > baseline_seek_rms * 1.35,
        "+6 dB gain was lost after seek reset: baseline={baseline_seek_rms}, gained={}",
        rms(&positive_seek)
    );
    assert!(
        rms(&negative_seek) < baseline_seek_rms * 0.75,
        "-6 dB gain was lost after seek reset: baseline={baseline_seek_rms}, gained={}",
        rms(&negative_seek)
    );
}

// ---------------------------------------------------------------------------
// seek_round_trip_within_tolerance — core M3 guarantee. Decode the same
// region via a contiguous pass and via seek + pre-roll, then compare RMS.
// ---------------------------------------------------------------------------

#[test]
fn seek_round_trip_within_tolerance() {
    // 30 s stereo sine @ 48 kHz = 1500 × 20 ms packets. pre_skip pinned
    // so the math lines up on paper: target_granule after seek =
    // sample_pos + 312, so the first returned sample has absolute
    // position 480 000 (= 10.0 s per-channel).
    const PACKETS: usize = 1500;
    const PRE_SKIP: u16 = 312;
    const SAMPLES_PER_PACKET: usize = 960;

    let bytes = build_opus_fixture_audio_source(
        "ropus-fb2k-test",
        &[],
        PACKETS,
        Some(PRE_SKIP),
        sine_pcm_for_packet,
    );

    // Path A: contiguous reference decode. Capture samples 10.0–10.2 s.
    let reference = decode_through_fb2k(bytes.clone());
    assert_eq!(
        reference.len(),
        (PACKETS * SAMPLES_PER_PACKET - PRE_SKIP as usize) * 2,
        "contiguous decode should yield total_samples × channels"
    );
    const SEEK_SAMPLE: usize = 480_000; // 10 s per-channel
    const WINDOW_SAMPLES: usize = 9_600; // 200 ms per-channel
    let ref_start = SEEK_SAMPLE * 2;
    let ref_end = ref_start + WINDOW_SAMPLES * 2;
    let ref_window = &reference[ref_start..ref_end];

    // Path B: open, seek, decode next 200 ms.
    let (_io, handle) = open_from_bytes(bytes);
    assert!(!handle.is_null(), "open failed: {}", last_error_string());

    let rc_seek = unsafe { ropus_fb2k::ropus_fb2k_seek(handle, SEEK_SAMPLE as u64) };
    assert_eq!(rc_seek, 0, "seek failed: {}", last_error_string());

    let mut post_seek: Vec<f32> = Vec::with_capacity(WINDOW_SAMPLES * 2);
    let mut buf = vec![0f32; 5760 * 2];
    let mut bytes_consumed = 0u64;
    while post_seek.len() < WINDOW_SAMPLES * 2 {
        let rc = unsafe {
            ropus_fb2k::ropus_fb2k_decode_next(handle, buf.as_mut_ptr(), 5760, &mut bytes_consumed)
        };
        assert!(
            rc > 0,
            "decode_next after seek returned {rc} (last_error={:?})",
            last_error_string()
        );
        post_seek.extend_from_slice(&buf[..rc as usize * 2]);
    }
    post_seek.truncate(WINDOW_SAMPLES * 2);

    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };

    // Per-sample RMS difference. With an 80 ms pre-roll both decoders
    // should have fully converged; HLD §7 tolerance is 1e-3.
    let sum_sq: f64 = ref_window
        .iter()
        .zip(post_seek.iter())
        .map(|(a, b)| {
            let d = (*a as f64) - (*b as f64);
            d * d
        })
        .sum();
    let rms = (sum_sq / ref_window.len() as f64).sqrt();

    // Observed RMS = 0.0 post-M3; 1e-5 catches any non-trivial convergence
    // regression while tolerating FP32 rounding edge cases. Sine-wave fixture
    // is stationary, so a future stronger oracle using time-varying content
    // should tighten further.
    assert!(
        rms <= 1e-5,
        "seek round-trip RMS {rms} exceeds 1e-5 tolerance"
    );
}

// ---------------------------------------------------------------------------
// seek_to_zero_is_valid — after seek(0) the next decode_next returns the
// first real audio sample (post-pre-skip), matching a fresh-open decode.
// ---------------------------------------------------------------------------

#[test]
fn seek_to_zero_is_valid() {
    let bytes = build_opus_fixture_with_audio_packets("ropus-fb2k-test", &[], 5, Some(312));

    // Reference: contiguous decode of the whole fixture.
    let reference = decode_through_fb2k(bytes.clone());
    assert!(!reference.is_empty(), "fixture must yield samples");

    // Open a second handle; seek(0); decode everything. Seek(0) exercises
    // the lazy `decoder = None` → `reset` path — if the reset is broken
    // the first packet after seek will produce nothing, or wrong samples.
    let (_io, handle) = open_from_bytes(bytes);
    assert!(!handle.is_null());
    let rc = unsafe { ropus_fb2k::ropus_fb2k_seek(handle, 0) };
    assert_eq!(
        rc,
        0,
        "seek(0) must return OK (last_error={:?})",
        last_error_string()
    );

    // The very first decode_next after seek(0) must produce samples —
    // confirms the lazy decoder init + reset path runs through to real
    // output.
    let mut buf = vec![0f32; 5760 * 2];
    let mut bytes_consumed = 0u64;
    let first = unsafe {
        ropus_fb2k::ropus_fb2k_decode_next(handle, buf.as_mut_ptr(), 5760, &mut bytes_consumed)
    };
    assert!(
        first > 0,
        "first decode_next after seek(0) must return samples, got {first}"
    );
    let mut out: Vec<f32> = Vec::new();
    out.extend_from_slice(&buf[..first as usize * 2]);

    // Drain the rest.
    loop {
        let rc = unsafe {
            ropus_fb2k::ropus_fb2k_decode_next(handle, buf.as_mut_ptr(), 5760, &mut bytes_consumed)
        };
        assert!(rc >= 0);
        if rc == 0 {
            break;
        }
        out.extend_from_slice(&buf[..rc as usize * 2]);
    }
    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };

    assert_eq!(
        out.len(),
        reference.len(),
        "seek(0) + drain must yield the same sample count as a fresh decode"
    );

    // Sample-content equality: seek(0) must put us back at the very first
    // audio sample. This is a far tighter check than length-equality alone
    // and catches regressions where the reset landed us on a later packet.
    assert_eq!(
        out, reference,
        "seek(0) + drain must match fresh-open samples bit-for-bit"
    );
}

// ---------------------------------------------------------------------------
// seek_zero_after_decode_rewinds — regression test for the silent
// no-op-on-`seek(0)` bug. Earlier the seek path's `total_samples == 0` and
// unseekable branches both early-returned `Ok(())` for `sample_pos == 0`
// without resetting the decoder or repositioning the packet reader. After
// `decode_next` had already drained N packets, that left the decoder
// mid-stream while the caller thought they had rewound — so the next
// `decode_next` returned packet N+1 instead of packet 1.
//
// Here we drain a chunk, seek(0), then decode again and assert the first
// emitted sample equals the first emitted sample from a fresh handle.
// ---------------------------------------------------------------------------

#[test]
fn seek_zero_after_decode_rewinds() {
    let bytes = build_opus_fixture_with_audio_packets("ropus-fb2k-test", &[], 5, Some(312));

    // Capture the very first samples-per-channel and float buffer a *fresh*
    // handle produces from its first `decode_next` call. With pre_skip=312
    // and a 960-sample packet, that's 960 - 312 = 648 samples/ch (one
    // partial packet), distinct from packet 2's 960 samples/ch — the size
    // alone is enough to detect the bug if the seek silently no-ops.
    let (fresh_n, fresh_buf) = {
        let (_io, handle) = open_from_bytes(bytes.clone());
        assert!(!handle.is_null());
        let mut buf = vec![0f32; 5760 * 2];
        let mut bytes_consumed = 0u64;
        let n = unsafe {
            ropus_fb2k::ropus_fb2k_decode_next(handle, buf.as_mut_ptr(), 5760, &mut bytes_consumed)
        };
        assert!(n > 0, "fresh-handle first decode must produce samples");
        let captured: Vec<f32> = buf[..n as usize * 2].to_vec();
        unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
        (n, captured)
    };

    // Drain a packet on a separate handle; *then* seek(0) and decode again.
    // With the bug, the next decode_next returns the *second* packet (a
    // full 960 samples/ch with different content); with the fix, the next
    // decode_next reproduces the fresh-handle first chunk.
    let (_io, handle) = open_from_bytes(bytes);
    assert!(!handle.is_null());
    let mut buf = vec![0f32; 5760 * 2];
    let mut bytes_consumed = 0u64;

    let first_decode = unsafe {
        ropus_fb2k::ropus_fb2k_decode_next(handle, buf.as_mut_ptr(), 5760, &mut bytes_consumed)
    };
    assert!(first_decode > 0, "pre-seek decode must produce samples");

    let rc = unsafe { ropus_fb2k::ropus_fb2k_seek(handle, 0) };
    assert_eq!(
        rc,
        0,
        "seek(0) after decode must return OK (last_error={:?})",
        last_error_string()
    );

    let post_seek_first = unsafe {
        ropus_fb2k::ropus_fb2k_decode_next(handle, buf.as_mut_ptr(), 5760, &mut bytes_consumed)
    };
    assert!(
        post_seek_first > 0,
        "first decode after seek(0) must produce samples, got {post_seek_first}"
    );

    // Sample-count and bit-for-bit equality. If seek(0) silently did
    // nothing, post_seek_first would be 960 (full packet 2) rather than
    // fresh_n (= 648, partial packet 1 after pre-skip trim).
    assert_eq!(
        post_seek_first, fresh_n,
        "post-seek samples-per-ch must match fresh-handle first-call count \
         (got {post_seek_first}, expected {fresh_n}; a no-op seek would have \
         returned the next packet's 960 samples/ch instead)"
    );
    let got = &buf[..post_seek_first as usize * 2];
    assert_eq!(
        got,
        fresh_buf.as_slice(),
        "post-seek samples must match fresh-handle samples bit-for-bit"
    );

    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

// ---------------------------------------------------------------------------
// seek_to_nonzero_then_decode — mirror of seek_to_zero_is_valid for a
// non-zero target. Seek to a sample in the middle of the fixture, decode,
// and confirm the first emitted samples match the corresponding offset in
// a contiguous decode. This exercises the index-walk + pre-roll discard
// through to real output.
// ---------------------------------------------------------------------------

#[test]
fn seek_to_nonzero_then_decode() {
    const PACKETS: usize = 30;
    const PRE_SKIP: u16 = 312;
    const SAMPLES_PER_PACKET: usize = 960;
    let bytes =
        build_opus_fixture_with_audio_packets("ropus-fb2k-test", &[], PACKETS, Some(PRE_SKIP));

    // Reference: contiguous decode. Length = PACKETS*960 - pre_skip per
    // channel × 2 channels.
    let reference = decode_through_fb2k(bytes.clone());
    let total_per_ch = PACKETS * SAMPLES_PER_PACKET - PRE_SKIP as usize;
    assert_eq!(reference.len(), total_per_ch * 2);

    // Seek to 10000 samples in — well inside the fixture, past pre-roll
    // so pre-roll discard is non-trivial.
    const SEEK_SAMPLE: u64 = 10_000;
    const WINDOW_PER_CH: usize = 960; // one packet's worth

    let (_io, handle) = open_from_bytes(bytes);
    assert!(!handle.is_null());
    let rc = unsafe { ropus_fb2k::ropus_fb2k_seek(handle, SEEK_SAMPLE) };
    assert_eq!(rc, 0, "seek failed: {}", last_error_string());

    let mut post: Vec<f32> = Vec::with_capacity(WINDOW_PER_CH * 2);
    let mut buf = vec![0f32; 5760 * 2];
    let mut bytes_consumed = 0u64;
    while post.len() < WINDOW_PER_CH * 2 {
        let rc = unsafe {
            ropus_fb2k::ropus_fb2k_decode_next(handle, buf.as_mut_ptr(), 5760, &mut bytes_consumed)
        };
        assert!(rc > 0, "decode_next after seek returned {rc}");
        post.extend_from_slice(&buf[..rc as usize * 2]);
    }
    post.truncate(WINDOW_PER_CH * 2);
    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };

    // Compare against the same slice of the contiguous decode. Silence
    // fixture → identical samples; any pre-roll miss or reset bug would
    // make these diverge.
    let ref_start = SEEK_SAMPLE as usize * 2;
    let ref_end = ref_start + WINDOW_PER_CH * 2;
    let ref_window = &reference[ref_start..ref_end];
    assert_eq!(
        post, ref_window,
        "post-seek decode must match the same window in a contiguous decode"
    );
}

// ---------------------------------------------------------------------------
// decode_next_excludes_preroll_bytes_after_seek — regression for the live-VBR
// bitrate FFI contract (HLD §4.2). After a seek that lands well inside the
// fixture, the 80 ms pre-roll discards roughly 4 × 20 ms packets before the
// first kept sample. Those discarded packets' payload bytes must NOT show up
// in the post-seek `decode_next` call's `bytes_consumed` — the C++ EWMA at
// `input_ropus_opus.cpp:300` computes `inst_bps = bytes * 8 * srate /
// kept_samples`, so leaking pre-roll bytes here would inflate the
// instantaneous bitrate by ~5× for ~1 s of playback, visibly skewing the
// status-bar readout.
//
// Oracle: compare against a no-seek baseline taken from the very first
// `decode_next` call on the same fixture (which also exercises the pre-skip
// trim path). With the bug the post-seek call's bytes_consumed would be
// multi-packet-sized; with the fix it should be within a small multiple of
// the no-seek baseline (one kept partial packet either way).
// ---------------------------------------------------------------------------

#[test]
fn decode_next_excludes_preroll_bytes_after_seek() {
    // 30 packets ≈ 600 ms. Seek to 10 000 samples (~208 ms) — well past the
    // first few packets so the 80 ms pre-roll spans ~4 fully-discarded
    // packets before any kept sample lands. Matches the fixture shape used
    // by `seek_to_nonzero_then_decode`.
    const PACKETS: usize = 30;
    const PRE_SKIP: u16 = 312;
    const SEEK_SAMPLE: u64 = 10_000;
    let bytes =
        build_opus_fixture_with_audio_packets("ropus-fb2k-test", &[], PACKETS, Some(PRE_SKIP));

    // Baseline: no-seek decode_next on a fresh handle. The first kept frame
    // post-pre-skip-trim is 960 - 312 = 648 samples and consumes ONE packet
    // payload. This is the reference for "kept samples per single-packet
    // payload" — the post-seek call should land in the same ballpark, not
    // ~5× higher.
    let baseline_bytes = {
        let (_io, handle) = open_from_bytes(bytes.clone());
        assert!(!handle.is_null(), "fixture must open");
        let mut buf = vec![0f32; 5760 * 2];
        let mut bc = 0u64;
        let rc =
            unsafe { ropus_fb2k::ropus_fb2k_decode_next(handle, buf.as_mut_ptr(), 5760, &mut bc) };
        assert!(rc > 0, "baseline decode must produce samples (got {rc})");
        unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
        bc
    };
    assert!(
        baseline_bytes > 0,
        "baseline bytes_consumed must be positive"
    );

    // Open a fresh handle, seek past the pre-roll window, then call
    // decode_next exactly ONCE. With the fix `bytes_consumed` reflects only
    // the kept-sample-producing packet(s); with the bug it would also
    // include ~4 fully-discarded pre-roll packets.
    let (_io, handle) = open_from_bytes(bytes);
    assert!(!handle.is_null(), "fixture must open");
    let rc_seek = unsafe { ropus_fb2k::ropus_fb2k_seek(handle, SEEK_SAMPLE) };
    assert_eq!(rc_seek, 0, "seek failed: {}", last_error_string());

    let mut buf = vec![0f32; 5760 * 2];
    let mut post_seek_bytes = 0u64;
    let rc = unsafe {
        ropus_fb2k::ropus_fb2k_decode_next(handle, buf.as_mut_ptr(), 5760, &mut post_seek_bytes)
    };
    assert!(
        rc > 0,
        "post-seek decode must produce samples (got {rc}, last_error={:?})",
        last_error_string()
    );
    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };

    // Core assertion: bytes_consumed must NOT be multi-packet-sized. With the
    // bug we'd expect ~5× baseline (4 discarded pre-roll packets + 1 kept
    // packet, all of similar size for a silence fixture). The 2× window
    // tolerates ordinary packet-size variance for partial-frame landings
    // while still flagging the ~5× regression unambiguously.
    let bound = baseline_bytes * 2;
    assert!(
        post_seek_bytes <= bound,
        "post-seek bytes_consumed = {post_seek_bytes} exceeds 2× baseline ({baseline_bytes}) — \
         pre-roll bytes are leaking into the kept-sample byte count, which would skew the \
         C++ EWMA. Expected <= {bound}."
    );
    // Sanity: it shouldn't be zero either — the kept packet was real.
    assert!(
        post_seek_bytes > 0,
        "post-seek bytes_consumed must be positive (got 0)"
    );
}

// ---------------------------------------------------------------------------
// seek_past_end_clamps — seek beyond total_samples returns OK (clamped);
// next decode is effectively EOF (zero samples).
// ---------------------------------------------------------------------------

#[test]
fn seek_past_end_clamps() {
    let bytes = build_opus_fixture_with_audio_packets("ropus-fb2k-test", &[], 5, Some(312));

    let (_io, handle) = open_from_bytes(bytes);
    assert!(!handle.is_null());

    let mut info = zeroed_info();
    let rc = unsafe { ropus_fb2k::ropus_fb2k_get_info(handle, &mut info) };
    assert_eq!(rc, 0);
    assert!(info.total_samples > 0);

    // Seek way past the end.
    let rc_seek = unsafe { ropus_fb2k::ropus_fb2k_seek(handle, info.total_samples + 10_000) };
    assert_eq!(
        rc_seek,
        0,
        "seek past end must clamp (last_error={:?})",
        last_error_string()
    );

    // After clamp + pre-roll discard, decode_next drains the tail of the
    // fixture and then returns 0 (EOF). Any post-seek samples are
    // post-target-granule by construction.
    let mut total_post_seek = 0usize;
    let mut buf = vec![0f32; 5760 * 2];
    let mut bytes_consumed = 0u64;
    loop {
        let rc = unsafe {
            ropus_fb2k::ropus_fb2k_decode_next(handle, buf.as_mut_ptr(), 5760, &mut bytes_consumed)
        };
        assert!(rc >= 0, "decode after clamp returned {rc}");
        if rc == 0 {
            break;
        }
        total_post_seek += rc as usize;
    }
    // The clamp target is the exact EOS granule, so the pre-roll is discarded
    // and no final-frame padding may leak after the end seek.
    assert!(
        total_post_seek == 0,
        "seek(total_samples) produced {total_post_seek} per-ch samples; expected exact EOF"
    );

    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

// ---------------------------------------------------------------------------
// seek_on_unseekable_returns_invalid_stream — without a seek IO callback,
// *any* seek (including zero) returns -4 INVALID_STREAM. The previous
// behaviour silently returned OK on `sample_pos == 0` even after the caller
// had already pumped `decode_next`, which left the decoder mid-stream while
// the caller thought they had rewound — a contract violation. We now reject
// up front; the C++ shim's `decode_can_seek` returns false for unseekable
// streams so fb2k won't issue this call in practice anyway.
// ---------------------------------------------------------------------------

#[test]
fn seek_on_unseekable_returns_invalid_stream_even_for_zero() {
    let bytes = build_opus_fixture_with_audio_packets("ropus-fb2k-test", &[], 20, Some(312));
    let (_io, handle) = open_from_bytes_without_seek(bytes, 0);
    assert!(
        !handle.is_null(),
        "unseekable open must succeed: {}",
        last_error_string()
    );

    // Both a non-zero target (used to be rejected) and zero (used to be
    // silently accepted as a no-op) must now consistently return INVALID_STREAM.
    for &target in &[1_000u64, 0u64] {
        let rc = unsafe { ropus_fb2k::ropus_fb2k_seek(handle, target) };
        assert_eq!(
            rc, ROPUS_FB2K_INVALID_STREAM,
            "seek({target}) on unseekable stream must return INVALID_STREAM"
        );

        let code = unsafe { ropus_fb2k::ropus_fb2k_last_error_code() };
        assert_eq!(code, ROPUS_FB2K_INVALID_STREAM);
        let msg = last_error_string().to_lowercase();
        assert!(
            msg.contains("seek") || msg.contains("seekable") || msg.contains("rewind"),
            "last-error should mention seek/seekable/rewind; got {msg:?}"
        );
    }

    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

// ---------------------------------------------------------------------------
// seek_propagates_abort_during_index_build — flip abort after open
// succeeds; first seek triggers lazy index build which must return -3
// ABORTED without corrupting the reader.
// ---------------------------------------------------------------------------

#[test]
fn seek_propagates_abort_during_index_build() {
    // 500 ms = 25 × 20 ms packets, so the index walk does enough read
    // calls that the abort fires mid-scan rather than racing to EOF.
    let bytes = build_opus_fixture_with_audio_packets("ropus-fb2k-test", &[], 25, Some(312));
    let io = MemIo::new(bytes);
    let fb2k_io = io.io();
    let handle = unsafe { ropus_fb2k::ropus_fb2k_open(&fb2k_io, 0) };
    assert!(
        !handle.is_null(),
        "open must succeed: {}",
        last_error_string()
    );

    // Flip abort now; first seek builds the index, which polls abort via
    // CallbackReader on every read.
    io.set_aborting();

    let rc = unsafe { ropus_fb2k::ropus_fb2k_seek(handle, 10_000) };
    assert_eq!(
        rc, ROPUS_FB2K_ABORTED,
        "seek during index build must surface -3"
    );
    let code = unsafe { ropus_fb2k::ropus_fb2k_last_error_code() };
    assert_eq!(code, ROPUS_FB2K_ABORTED);

    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

// ---------------------------------------------------------------------------
// seek_during_abort_recovery_no_panic — after a seek() aborts mid-scan,
// `packet_reader` must still be seated so a *second* seek doesn't panic on
// `take().expect(...)`. If the recovery path were broken, `ffi_guard!`
// would catch the panic and return -6 INTERNAL, masking what is really
// an ABORTED state. Regression test for the recovery-restructure fix.
// ---------------------------------------------------------------------------

#[test]
fn seek_during_abort_recovery_no_panic() {
    let bytes = build_opus_fixture_with_audio_packets("ropus-fb2k-test", &[], 25, Some(312));
    let io = MemIo::new(bytes);
    let fb2k_io = io.io();
    let handle = unsafe { ropus_fb2k::ropus_fb2k_open(&fb2k_io, 0) };
    assert!(
        !handle.is_null(),
        "open must succeed: {}",
        last_error_string()
    );

    // First seek aborts mid-scan.
    io.set_aborting();
    let rc1 = unsafe { ropus_fb2k::ropus_fb2k_seek(handle, 10_000) };
    assert_eq!(rc1, ROPUS_FB2K_ABORTED, "first seek must return -3 ABORTED");

    // Second seek call. With the recovery fix, `packet_reader` has been
    // re-seated so this must not panic inside the take/expect inside
    // build_page_index. We don't care whether it succeeds or aborts again,
    // only that the code reaches an orderly error path — anything other
    // than -6 INTERNAL is evidence the panic guard did NOT fire.
    let rc2 = unsafe { ropus_fb2k::ropus_fb2k_seek(handle, 10_000) };
    assert_ne!(
        rc2,
        ropus_fb2k::ROPUS_FB2K_INTERNAL,
        "second seek must not surface -6 INTERNAL (indicates panic); got {rc2}"
    );
    // With abort still held high, the second seek should keep reporting
    // ABORTED from the same code path.
    assert_eq!(
        rc2, ROPUS_FB2K_ABORTED,
        "second seek should cleanly report -3 ABORTED; got {rc2}"
    );

    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

// ---------------------------------------------------------------------------
// Failed seeks must preserve the reader and permit a retry.
// ---------------------------------------------------------------------------

#[test]
fn failed_target_seek_keeps_reader_usable() {
    let bytes = build_opus_fixture_with_audio_packets("ropus-fb2k-test", &[], 25, Some(312));
    let (io, handle) = open_from_bytes(bytes);
    assert!(!handle.is_null(), "open failed: {}", last_error_string());

    // Build the lazy page index first. The injected failure then belongs to
    // the target reposition itself, not to the scan's rollback.
    assert_eq!(
        unsafe { ropus_fb2k::ropus_fb2k_seek(handle, 10_000) },
        0,
        "initial seek failed: {}",
        last_error_string()
    );
    io.fail_next_seek();
    let failed = unsafe { ropus_fb2k::ropus_fb2k_seek(handle, 12_000) };
    assert_eq!(failed, ROPUS_FB2K_IO, "target seek failure must be I/O");

    // The failed callback must not leave packet_reader empty or reset the
    // decoder state. A one-shot failure lets the following seek prove that
    // the same handle remains operational.
    assert_eq!(
        unsafe { ropus_fb2k::ropus_fb2k_seek(handle, 12_000) },
        0,
        "retry after target seek failure failed: {}",
        last_error_string()
    );
    let mut buf = vec![0f32; 5760 * 2];
    let mut bytes_consumed = 0u64;
    let decoded = unsafe {
        ropus_fb2k::ropus_fb2k_decode_next(handle, buf.as_mut_ptr(), 5760, &mut bytes_consumed)
    };
    assert!(decoded > 0, "retry seek must leave the reader decodable");
    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

#[test]
fn failed_rewind_keeps_reader_usable() {
    let bytes = build_opus_fixture_with_audio_packets("ropus-fb2k-test", &[], 5, Some(312));
    let (io, handle) = open_from_bytes(bytes);
    assert!(!handle.is_null(), "open failed: {}", last_error_string());

    io.fail_next_seek();
    let failed = unsafe { ropus_fb2k::ropus_fb2k_seek(handle, 0) };
    assert_eq!(failed, ROPUS_FB2K_IO, "rewind failure must be I/O");
    assert_eq!(
        unsafe { ropus_fb2k::ropus_fb2k_seek(handle, 0) },
        0,
        "retry after rewind failure failed: {}",
        last_error_string()
    );

    let mut buf = vec![0f32; 5760 * 2];
    let mut bytes_consumed = 0u64;
    let decoded = unsafe {
        ropus_fb2k::ropus_fb2k_decode_next(handle, buf.as_mut_ptr(), 5760, &mut bytes_consumed)
    };
    assert!(decoded > 0, "retry rewind must leave the reader decodable");
    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

#[test]
fn aborted_index_scan_resumes_original_content_after_clear() {
    let bytes =
        build_opus_fixture_audio_source("ropus-fb2k-test", &[], 25, Some(312), sine_pcm_for_packet);

    let (_reference_io, reference_handle) = open_from_bytes(bytes.clone());
    assert!(!reference_handle.is_null());
    let mut reference_buf = vec![0f32; 5760 * 2];
    let mut reference_bytes = 0u64;
    let first = unsafe {
        ropus_fb2k::ropus_fb2k_decode_next(
            reference_handle,
            reference_buf.as_mut_ptr(),
            5760,
            &mut reference_bytes,
        )
    };
    assert!(first > 0);
    let mut reference_next_buf = vec![0f32; 5760 * 2];
    let mut reference_next_bytes = 0u64;
    let next = unsafe {
        ropus_fb2k::ropus_fb2k_decode_next(
            reference_handle,
            reference_next_buf.as_mut_ptr(),
            5760,
            &mut reference_next_bytes,
        )
    };
    assert!(next > 0);
    let reference_next = &reference_next_buf[..next as usize * 2];
    unsafe { ropus_fb2k::ropus_fb2k_close(reference_handle) };

    let (io, handle) = open_from_bytes(bytes);
    assert!(!handle.is_null());
    let mut first_buf = vec![0f32; 5760 * 2];
    let mut first_bytes = 0u64;
    let first_after_open = unsafe {
        ropus_fb2k::ropus_fb2k_decode_next(handle, first_buf.as_mut_ptr(), 5760, &mut first_bytes)
    };
    assert_eq!(first_after_open, first);
    assert_eq!(
        &first_buf[..first as usize * 2],
        &reference_buf[..first as usize * 2]
    );

    // The scan and its rollback both observe the active abort. The reader
    // must return ABORTED without losing the pre-seek decode position.
    io.set_aborting();
    let aborted = unsafe { ropus_fb2k::ropus_fb2k_seek(handle, 10_000) };
    assert_eq!(aborted, ROPUS_FB2K_ABORTED);
    io.clear_aborting();

    let mut recovered_buf = vec![0f32; 5760 * 2];
    let mut recovered_bytes = 0u64;
    let recovered = unsafe {
        ropus_fb2k::ropus_fb2k_decode_next(
            handle,
            recovered_buf.as_mut_ptr(),
            5760,
            &mut recovered_bytes,
        )
    };
    assert_eq!(
        recovered, next,
        "clear-abort decode must resume with samples"
    );
    assert_eq!(
        &recovered_buf[..recovered as usize * 2],
        reference_next,
        "clear-abort decode must resume the original packet position"
    );
    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

// ---------------------------------------------------------------------------
// ReplayGain mapping per HLD §5.5
// ---------------------------------------------------------------------------

#[test]
fn rg_track_gain_r128() {
    // -1280 Q7.8 = -5 dB R128. Add +5 dB RG offset → 0 dB.
    let bytes = build_opus_fixture("v", &[("R128_TRACK_GAIN", "-1280")]);
    let (_io, handle) = open_from_bytes(bytes);
    assert!(!handle.is_null());

    let mut info = zeroed_info();
    assert_eq!(
        unsafe { ropus_fb2k::ropus_fb2k_get_info(handle, &mut info) },
        0
    );
    assert!(
        (info.rg_track_gain - 0.0).abs() < 1e-3,
        "expected 0 dB, got {}",
        info.rg_track_gain
    );

    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

#[test]
fn rg_legacy_overrides_r128() {
    let bytes = build_opus_fixture(
        "v",
        &[
            ("R128_TRACK_GAIN", "-1280"),          // would map to 0 dB
            ("REPLAYGAIN_TRACK_GAIN", "-6.75 dB"), // should win
        ],
    );
    let (_io, handle) = open_from_bytes(bytes);
    assert!(!handle.is_null());

    let mut info = zeroed_info();
    assert_eq!(
        unsafe { ropus_fb2k::ropus_fb2k_get_info(handle, &mut info) },
        0
    );
    assert!(
        (info.rg_track_gain - (-6.75)).abs() < 1e-3,
        "legacy must win; got {}",
        info.rg_track_gain
    );

    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

#[test]
fn rg_peak_linear_passthrough() {
    let bytes = build_opus_fixture("v", &[("R128_TRACK_PEAK", "0.95")]);
    let (_io, handle) = open_from_bytes(bytes);
    assert!(!handle.is_null());

    let mut info = zeroed_info();
    assert_eq!(
        unsafe { ropus_fb2k::ropus_fb2k_get_info(handle, &mut info) },
        0
    );
    assert!((info.rg_track_peak - 0.95).abs() < 1e-3);

    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

#[test]
fn rg_absent_tags_are_nan() {
    // Plain fixture, no RG tags at all.
    let bytes = build_opus_fixture("v", &[("ARTIST", "Alice")]);
    let (_io, handle) = open_from_bytes(bytes);
    assert!(!handle.is_null());

    let mut info = zeroed_info();
    assert_eq!(
        unsafe { ropus_fb2k::ropus_fb2k_get_info(handle, &mut info) },
        0
    );
    assert!(info.rg_track_gain.is_nan(), "track_gain must be NaN");
    assert!(info.rg_album_gain.is_nan(), "album_gain must be NaN");
    assert!(info.rg_track_peak.is_nan(), "track_peak must be NaN");
    assert!(info.rg_album_peak.is_nan(), "album_peak must be NaN");

    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

#[test]
fn rg_malformed_tag_is_nan() {
    let bytes = build_opus_fixture("v", &[("R128_TRACK_GAIN", "not_a_number")]);
    let (_io, handle) = open_from_bytes(bytes);
    assert!(!handle.is_null());

    let mut info = zeroed_info();
    assert_eq!(
        unsafe { ropus_fb2k::ropus_fb2k_get_info(handle, &mut info) },
        0
    );
    assert!(info.rg_track_gain.is_nan(), "malformed → NaN");

    // Decode/playback still works after a malformed tag — bad metadata
    // must not break the audio path.
    let mut buf = vec![0f32; 5760 * 2];
    let mut bytes_consumed = 0u64;
    let rc = unsafe {
        ropus_fb2k::ropus_fb2k_decode_next(handle, buf.as_mut_ptr(), 5760, &mut bytes_consumed)
    };
    assert!(rc >= 0, "decode must still work; got {rc}");

    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

// ---------------------------------------------------------------------------
// panic_in_decode_surfaces_internal_code — proves `ffi_guard!` turns a
// deep panic in `decode_next` into the unified `-6 INTERNAL` last-error
// code, plus the per-entry `-1 BAD_ARG` return sentinel and a human-
// readable message containing "internal panic". Gated behind the
// `test-panic` feature flag because it depends on a hidden FFI hook
// (`ropus_fb2k_test_set_panic_flag`) that arms a thread-local panic flag.
// CI runs `cargo test -p ropus-fb2k --features test-panic` to exercise
// this case; the default `cargo test` skips it.
// ---------------------------------------------------------------------------

#[cfg(feature = "test-panic")]
unsafe extern "C" {
    fn ropus_fb2k_test_set_panic_flag(on: bool);
}

#[cfg(feature = "test-panic")]
#[test]
fn panic_in_decode_surfaces_internal_code() {
    use ropus_fb2k::ROPUS_FB2K_INTERNAL;

    // RAII guard: if any assertion below panics before we reach the manual
    // disarm, cargo's test runner may reuse this OS thread for a subsequent
    // test on the same thread, and the armed panic flag would fire inside
    // that unrelated test's `decode_next` call — producing a confusing
    // secondary failure that hides the original. Drop-order guarantees the
    // flag is cleared on *any* exit path (pass, assertion panic, or
    // early-return), so parallel / sequential reuse is always safe.
    struct DisarmPanicFlag;
    impl Drop for DisarmPanicFlag {
        fn drop(&mut self) {
            unsafe { ropus_fb2k_test_set_panic_flag(false) };
        }
    }
    let _guard = DisarmPanicFlag;

    let (_io, handle) = open_from_bytes(minimal_opus_fixture().to_vec());
    assert!(!handle.is_null(), "fixture must open");

    // Arm the panic flag, then call decode_next. The hook fires at the top
    // of `decode_next` before any real work; `ffi_guard!` should catch the
    // unwind and:
    //   * return the per-entry sentinel (`-1 BAD_ARG` — see ffi_guard! call
    //     site in lib.rs::ropus_fb2k_decode_next),
    //   * write `ROPUS_FB2K_INTERNAL` to the last-error-code slot,
    //   * stash a "internal panic" message in the last-error string slot.
    unsafe { ropus_fb2k_test_set_panic_flag(true) };
    let mut buf = vec![0f32; 5760 * 2];
    let mut bytes_consumed = 0u64;
    let rc = unsafe {
        ropus_fb2k::ropus_fb2k_decode_next(handle, buf.as_mut_ptr(), 5760, &mut bytes_consumed)
    };
    // Per-entry sentinel — ffi_guard!'s on_panic for decode_next is BAD_ARG.
    assert_eq!(
        rc, ROPUS_FB2K_BAD_ARG,
        "decode_next entry sentinel must be BAD_ARG (got {rc})"
    );
    // Unified class code in the last-error slot.
    let code = unsafe { ropus_fb2k::ropus_fb2k_last_error_code() };
    assert_eq!(
        code, ROPUS_FB2K_INTERNAL,
        "last_error_code must be INTERNAL after panic (got {code})"
    );
    // Message text — must mention "internal panic" so a C caller surfacing
    // the message can distinguish the panic from a regular bad-arg error.
    let msg = last_error_string().to_lowercase();
    assert!(
        msg.contains("internal panic"),
        "last_error message must mention 'internal panic'; got {msg:?}"
    );

    // Flag is disarmed by `_guard`'s Drop impl; no manual call needed here.
    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

// ---------------------------------------------------------------------------
// Helpers local to the test file
// ---------------------------------------------------------------------------

fn zeroed_info() -> RopusFb2kInfo {
    RopusFb2kInfo {
        sample_rate: 0,
        channels: 0,
        pre_skip: 0,
        total_samples: 0,
        nominal_bitrate: 0,
        rg_track_gain: 0.0,
        rg_album_gain: 0.0,
        rg_track_peak: 0.0,
        rg_album_peak: 0.0,
    }
}

/// Path A in `decode_matches_reference_path`: drive the full fb2k C ABI
/// (`ropus_fb2k_open` + `ropus_fb2k_decode_next`) and collect every sample
/// the caller would see (pre-skip already trimmed by the reader).
fn decode_through_fb2k(bytes: Vec<u8>) -> Vec<f32> {
    let (_io, handle) = open_from_bytes(bytes);
    assert!(!handle.is_null(), "path-A open failed");
    let mut info = zeroed_info();
    let rc = unsafe { ropus_fb2k::ropus_fb2k_get_info(handle, &mut info) };
    assert_eq!(rc, 0);
    let ch = info.channels as usize;

    let mut out = Vec::new();
    let mut buf = vec![0f32; 5760 * ch];
    let mut bytes_consumed = 0u64;
    loop {
        let rc = unsafe {
            ropus_fb2k::ropus_fb2k_decode_next(handle, buf.as_mut_ptr(), 5760, &mut bytes_consumed)
        };
        assert!(rc >= 0, "path-A decode failure {rc}");
        if rc == 0 {
            break;
        }
        out.extend_from_slice(&buf[..rc as usize * ch]);
    }
    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
    out
}

/// Path B in `decode_matches_reference_path`: parse the same bytes via the
/// `ogg` crate and feed packets to a fresh low-level `OpusDecoder` instance.
/// Applies the same pre-skip trim that the fb2k reader does, so the two
/// streams should line up post-trim.
fn decode_reference_direct(bytes: &[u8]) -> Vec<f32> {
    let mut pr = PacketReader::new(Cursor::new(bytes.to_vec()));

    // Page 1: OpusHead — parse just enough to learn channels + pre_skip.
    let head_pkt = pr.read_packet().unwrap().expect("OpusHead");
    assert!(head_pkt.data.len() >= 19);
    assert_eq!(&head_pkt.data[..8], b"OpusHead");
    let channels = head_pkt.data[9] as usize;
    let pre_skip = u16::from_le_bytes([head_pkt.data[10], head_pkt.data[11]]);

    // Page 2: OpusTags — we don't care about payload for this path.
    let _tags_pkt = pr.read_packet().unwrap().expect("OpusTags");

    let mut dec = OpusDecoder::new(48_000, channels as i32).expect("decoder inits");
    let mut pre_skip_remaining = pre_skip as usize;
    let mut out = Vec::new();
    let mut scratch = vec![0f32; 5760 * channels];
    while let Some(pkt) = pr.read_packet().unwrap() {
        let n = dec
            .decode_float(Some(&pkt.data), &mut scratch, 5760, false)
            .expect("decode") as usize;
        let drop = pre_skip_remaining.min(n);
        pre_skip_remaining -= drop;
        let kept = n - drop;
        if kept == 0 {
            continue;
        }
        let start = drop * channels;
        let end = start + kept * channels;
        out.extend_from_slice(&scratch[start..end]);
    }
    out
}
