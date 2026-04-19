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
    build_opus_fixture, build_opus_fixture_with_audio_packets, last_error_string,
    minimal_opus_fixture, open_from_bytes, open_from_bytes_info_only,
    open_from_bytes_without_seek, opus_fixture_with_artist_alice, read_tags_collect,
    surround_family_fixture, MemIo,
};

use ropus_fb2k::{
    RopusFb2kInfo, ROPUS_FB2K_ABORTED, ROPUS_FB2K_BAD_ARG, ROPUS_FB2K_INVALID_STREAM,
    ROPUS_FB2K_UNSUPPORTED,
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
        code, ROPUS_FB2K_INVALID_STREAM,
        "garbage must surface as INVALID_STREAM (last_error={:?})",
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
    assert!(info.rg_track_gain.is_nan(), "ReplayGain stays NaN until tag mapping lands");

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
        pairs
            .iter()
            .any(|(k, v)| k == "ARTIST" && v == "Alice"),
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
    assert_eq!(pairs.len(), 1, "only the synthetic VENDOR entry; got {pairs:?}");
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
        pairs
            .iter()
            .any(|(k, v)| k == "ARTIST" && v == "Alice"),
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
        code, ROPUS_FB2K_UNSUPPORTED,
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
// Tripwire: seek is not yet wired — updates when the seek path lands,
// until then this pins the ABI contract (return code + unified last-error
// code) so an accidental change surfaces as a test failure.
// ---------------------------------------------------------------------------

#[test]
fn seek_currently_unsupported() {
    let (_io, handle) = open_from_bytes(minimal_opus_fixture().to_vec());
    assert!(!handle.is_null());

    let rc_seek = unsafe { ropus_fb2k::ropus_fb2k_seek(handle, 0) };
    assert_eq!(
        rc_seek, ROPUS_FB2K_UNSUPPORTED,
        "seek returns UNSUPPORTED until it's wired"
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
    assert!(!path_a.is_empty(), "fixture must produce some decoded samples");
}

// ---------------------------------------------------------------------------
// decode_next returns 0 at EOF after all packets drained.
// ---------------------------------------------------------------------------

#[test]
fn decode_returns_zero_at_eof() {
    let (_io, handle) = open_from_bytes(minimal_opus_fixture().to_vec());
    assert!(!handle.is_null());

    let mut scratch = vec![0f32; 5760 * 2];
    // Drain every non-zero return first.
    loop {
        let rc = unsafe {
            ropus_fb2k::ropus_fb2k_decode_next(handle, scratch.as_mut_ptr(), 5760)
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
        ropus_fb2k::ropus_fb2k_decode_next(handle, scratch.as_mut_ptr(), 5760)
    };
    assert_eq!(rc, 0, "post-EOF decode_next stays at 0");

    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
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
    assert!(!handle.is_null(), "open must succeed (last_error={:?})", last_error_string());

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
    let rc = unsafe {
        ropus_fb2k::ropus_fb2k_decode_next(handle, scratch.as_mut_ptr(), 5760)
    };
    assert_eq!(rc, ROPUS_FB2K_ABORTED, "decode_next must surface -3 ABORTED");

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
    let rc = unsafe {
        ropus_fb2k::ropus_fb2k_decode_next(handle, scratch.as_mut_ptr(), 4096)
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
    let bytes = build_opus_fixture_with_audio_packets(
        "ropus-fb2k-test",
        &[],
        PACKETS,
        Some(PRE_SKIP),
    );
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
    let bytes = build_opus_fixture_with_audio_packets(
        "ropus-fb2k-test",
        &[],
        PACKETS,
        Some(PRE_SKIP),
    );
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
    assert_eq!(info.total_samples, 0, "unseekable stream reports zero duration");
    assert_eq!(info.nominal_bitrate, -1, "unseekable stream reports unknown bitrate");

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
    loop {
        let rc = unsafe {
            ropus_fb2k::ropus_fb2k_decode_next(handle, buf.as_mut_ptr(), 5760)
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

