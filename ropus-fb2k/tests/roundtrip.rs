//! Integration tests for the M1 `ropus-fb2k` C ABI.
//!
//! Every test here drives the public C entry points (`ropus_fb2k_open`,
//! `ropus_fb2k_read_tags`, …) — the `open_rust` rlib helper was deleted in
//! the M1 code-review pass. Shared fixture + `MemIo` helpers live in
//! `tests/common/mod.rs`; each test pulls only what it needs from there.
//!
//! Fixtures are constructed by encoding a short silence stream via
//! `ropus::Encoder` into an in-memory Ogg container, which matches the
//! encode path used by `ropus-cli` — any future change there that breaks
//! our fixture is a change we want to notice.

mod common;

use common::{
    build_opus_fixture, last_error_string, minimal_opus_fixture, open_from_bytes,
    open_from_bytes_info_only, opus_fixture_with_artist_alice, read_tags_collect,
    surround_family_fixture, MemIo,
};

use ropus_fb2k::{
    RopusFb2kInfo, ROPUS_FB2K_ABORTED, ROPUS_FB2K_INVALID_STREAM, ROPUS_FB2K_UNSUPPORTED,
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
    assert!(info.rg_track_gain.is_nan(), "M1 leaves ReplayGain as NaN");

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
// M1 stubs return UNSUPPORTED. DELETE WHEN M2 LANDS — these stubs become
// real code.
// ---------------------------------------------------------------------------

#[test]
fn stubs_return_unsupported_m1_only() {
    let (_io, handle) = open_from_bytes(minimal_opus_fixture().to_vec());
    assert!(!handle.is_null());

    let mut scratch = vec![0f32; 5760 * 2];
    let rc_decode =
        unsafe { ropus_fb2k::ropus_fb2k_decode_next(handle, scratch.as_mut_ptr(), 5760) };
    assert_eq!(
        rc_decode, ROPUS_FB2K_UNSUPPORTED,
        "decode_next is an UNSUPPORTED stub in M1"
    );

    let rc_seek = unsafe { ropus_fb2k::ropus_fb2k_seek(handle, 0) };
    assert_eq!(
        rc_seek, ROPUS_FB2K_UNSUPPORTED,
        "seek is an UNSUPPORTED stub in M1"
    );

    unsafe { ropus_fb2k::ropus_fb2k_close(handle) };
}

// ---------------------------------------------------------------------------
// INFO_ONLY flag is accepted without error. M2 will fill total_samples and
// nominal_bitrate under this flag; this test only asserts handle validity.
// ---------------------------------------------------------------------------

#[test]
fn info_only_flag_is_accepted() {
    let (_io, handle) = open_from_bytes_info_only(minimal_opus_fixture().to_vec());
    assert!(
        !handle.is_null(),
        "ROPUS_FB2K_OPEN_INFO_ONLY must be accepted by M1 (last_error={:?})",
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

