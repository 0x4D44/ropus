//! Conformance test: `reference/tests/test_opus_decode.c`.
//!
//! Exercises PLC, zero-length input, null pointer handling, CTL behaviour
//! (RESET_STATE, GET_FINAL_RANGE, GET_LAST_PACKET_DURATION), fuzzes
//! valid-prefix packets across configs, and finishes with `test_soft_clip`
//! which drives `opus_pcm_soft_clip` directly.
//!
//! This file is its own `[[test]]` binary (cargo auto-discovery). See
//! `padding.rs` for the rationale — `test_opus_common.h` declares a common
//! `regression_test(void)` that both decode and encode tests define, so
//! they must live in separate binaries.
//!
//! As with `padding`, requires `-- --test-threads=1` — the reference tests
//! use their own process-image-global RNG.

use std::hint::black_box;
use std::os::raw::{c_char, c_int};

// Compiled by build.rs as `libtest_opus_decode.a`, with
// `-Dmain=test_opus_decode_main`.
unsafe extern "C" {
    fn test_opus_decode_main(argc: c_int, argv: *const *const c_char) -> c_int;
}

/// Pull every `#[unsafe(no_mangle)]` symbol the decode test could resolve out
/// of the `capi` rlib. Expanded beyond padding.rs to cover the packet
/// inspection helpers (`opus_packet_get_*`, `opus_decoder_get_nb_samples`),
/// the int/uint32 CTL dispatchers, and `opus_pcm_soft_clip`.
fn force_link() {
    // Library-level
    black_box(mdopus_capi::opus_strerror as *const ());
    black_box(mdopus_capi::opus_get_version_string as *const ());
    // Encoder (test doesn't exercise these, but link them so the rlib is
    // pulled and inadvertent drift doesn't mask the failure).
    black_box(mdopus_capi::encoder::opus_encoder_get_size as *const ());
    black_box(mdopus_capi::encoder::opus_encoder_create as *const ());
    black_box(mdopus_capi::encoder::opus_encoder_init as *const ());
    black_box(mdopus_capi::encoder::opus_encoder_destroy as *const ());
    black_box(mdopus_capi::encoder::opus_encode as *const ());
    black_box(mdopus_capi::encoder::opus_encode_float as *const ());
    // Decoder core
    black_box(mdopus_capi::decoder::opus_decoder_get_size as *const ());
    black_box(mdopus_capi::decoder::opus_decoder_create as *const ());
    black_box(mdopus_capi::decoder::opus_decoder_init as *const ());
    black_box(mdopus_capi::decoder::opus_decoder_destroy as *const ());
    black_box(mdopus_capi::decoder::opus_decode as *const ());
    black_box(mdopus_capi::decoder::opus_decode_float as *const ());
    black_box(mdopus_capi::decoder::opus_packet_pad as *const ());
    black_box(mdopus_capi::decoder::opus_packet_unpad as *const ());
    // Packet-TOC accessors
    black_box(mdopus_capi::decoder::opus_packet_get_bandwidth as *const ());
    black_box(mdopus_capi::decoder::opus_packet_get_samples_per_frame as *const ());
    black_box(mdopus_capi::decoder::opus_packet_get_nb_channels as *const ());
    black_box(mdopus_capi::decoder::opus_packet_get_nb_frames as *const ());
    black_box(mdopus_capi::decoder::opus_packet_get_nb_samples as *const ());
    black_box(mdopus_capi::decoder::opus_packet_has_lbrr as *const ());
    black_box(mdopus_capi::decoder::opus_decoder_get_nb_samples as *const ());
    // Soft clip
    black_box(mdopus_capi::decoder::opus_pcm_soft_clip as *const ());
    // CTL shim
    black_box(mdopus_capi::ctl::mdopus_encoder_ctl_reset as *const ());
    black_box(mdopus_capi::ctl::mdopus_decoder_ctl_reset as *const ());
    black_box(mdopus_capi::ctl::mdopus_decoder_ctl_set_int as *const ());
    black_box(mdopus_capi::ctl::mdopus_decoder_ctl_get_int as *const ());
    black_box(mdopus_capi::ctl::mdopus_decoder_ctl_get_uint32 as *const ());
}

#[test]
fn decode() {
    force_link();
    let rc = unsafe { test_opus_decode_main(0, std::ptr::null()) };
    assert_eq!(rc, 0, "test_opus_decode returned {}", rc);
}
