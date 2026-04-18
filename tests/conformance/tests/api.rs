//! Conformance test: `reference/tests/test_opus_api.c`.
//!
//! 1916-line API contract fuzzer from xiph/opus. Exercises encoder, decoder,
//! multistream encoder, multistream decoder, repacketizer, and packet helpers.
//! Built as its own static lib via build.rs with `-Dmain=test_opus_api_main`.
//!
//! Requires `-- --test-threads=1`; the reference tests share process-global
//! RNG state (`test_opus_common.h`).

use std::hint::black_box;
use std::os::raw::{c_char, c_int};

unsafe extern "C" {
    fn test_opus_api_main(argc: c_int, argv: *const *const c_char) -> c_int;
}

/// Pull every `#[unsafe(no_mangle)]` symbol the api test could resolve out of
/// the `capi` rlib.
fn force_link() {
    // Library-level
    black_box(mdopus_capi::opus_strerror as *const ());
    black_box(mdopus_capi::opus_get_version_string as *const ());

    // Encoder
    black_box(mdopus_capi::encoder::opus_encoder_get_size as *const ());
    black_box(mdopus_capi::encoder::opus_encoder_create as *const ());
    black_box(mdopus_capi::encoder::opus_encoder_init as *const ());
    black_box(mdopus_capi::encoder::opus_encoder_destroy as *const ());
    black_box(mdopus_capi::encoder::opus_encode as *const ());
    black_box(mdopus_capi::encoder::opus_encode_float as *const ());

    // Decoder
    black_box(mdopus_capi::decoder::opus_decoder_get_size as *const ());
    black_box(mdopus_capi::decoder::opus_decoder_create as *const ());
    black_box(mdopus_capi::decoder::opus_decoder_init as *const ());
    black_box(mdopus_capi::decoder::opus_decoder_destroy as *const ());
    black_box(mdopus_capi::decoder::opus_decode as *const ());
    black_box(mdopus_capi::decoder::opus_decode_float as *const ());
    black_box(mdopus_capi::decoder::opus_packet_pad as *const ());
    black_box(mdopus_capi::decoder::opus_packet_unpad as *const ());
    black_box(mdopus_capi::decoder::opus_packet_get_bandwidth as *const ());
    black_box(mdopus_capi::decoder::opus_packet_get_samples_per_frame as *const ());
    black_box(mdopus_capi::decoder::opus_packet_get_nb_channels as *const ());
    black_box(mdopus_capi::decoder::opus_packet_get_nb_frames as *const ());
    black_box(mdopus_capi::decoder::opus_packet_get_nb_samples as *const ());
    black_box(mdopus_capi::decoder::opus_packet_has_lbrr as *const ());
    black_box(mdopus_capi::decoder::opus_decoder_get_nb_samples as *const ());
    black_box(mdopus_capi::decoder::opus_pcm_soft_clip as *const ());

    // Multistream encoder
    black_box(mdopus_capi::ms_encoder::opus_multistream_encoder_get_size as *const ());
    black_box(mdopus_capi::ms_encoder::opus_multistream_surround_encoder_get_size as *const ());
    black_box(mdopus_capi::ms_encoder::opus_multistream_encoder_create as *const ());
    black_box(mdopus_capi::ms_encoder::opus_multistream_surround_encoder_create as *const ());
    black_box(mdopus_capi::ms_encoder::opus_multistream_encoder_init as *const ());
    black_box(mdopus_capi::ms_encoder::opus_multistream_surround_encoder_init as *const ());
    black_box(mdopus_capi::ms_encoder::opus_multistream_encoder_destroy as *const ());
    black_box(mdopus_capi::ms_encoder::opus_multistream_encode as *const ());
    black_box(mdopus_capi::ms_encoder::opus_multistream_encode_float as *const ());

    // Multistream decoder
    black_box(mdopus_capi::ms_decoder::opus_multistream_decoder_get_size as *const ());
    black_box(mdopus_capi::ms_decoder::opus_multistream_decoder_create as *const ());
    black_box(mdopus_capi::ms_decoder::opus_multistream_decoder_init as *const ());
    black_box(mdopus_capi::ms_decoder::opus_multistream_decoder_destroy as *const ());
    black_box(mdopus_capi::ms_decoder::opus_multistream_decode as *const ());
    black_box(mdopus_capi::ms_decoder::opus_multistream_decode_float as *const ());

    // Repacketizer + MS pad/unpad
    black_box(mdopus_capi::repacketizer::opus_repacketizer_get_size as *const ());
    black_box(mdopus_capi::repacketizer::opus_repacketizer_create as *const ());
    black_box(mdopus_capi::repacketizer::opus_repacketizer_init as *const ());
    black_box(mdopus_capi::repacketizer::opus_repacketizer_destroy as *const ());
    black_box(mdopus_capi::repacketizer::opus_repacketizer_cat as *const ());
    black_box(mdopus_capi::repacketizer::opus_repacketizer_out as *const ());
    black_box(mdopus_capi::repacketizer::opus_repacketizer_out_range as *const ());
    black_box(mdopus_capi::repacketizer::opus_repacketizer_get_nb_frames as *const ());
    black_box(mdopus_capi::repacketizer::opus_multistream_packet_pad as *const ());
    black_box(mdopus_capi::repacketizer::opus_multistream_packet_unpad as *const ());

    // Packet parse
    black_box(mdopus_capi::packet_parse::opus_packet_parse as *const ());

    // CTL shim entry points
    black_box(mdopus_capi::ctl::mdopus_encoder_ctl_reset as *const ());
    black_box(mdopus_capi::ctl::mdopus_encoder_ctl_set_int as *const ());
    black_box(mdopus_capi::ctl::mdopus_encoder_ctl_get_int as *const ());
    black_box(mdopus_capi::ctl::mdopus_encoder_ctl_get_uint32 as *const ());
    black_box(mdopus_capi::ctl::mdopus_decoder_ctl_reset as *const ());
    black_box(mdopus_capi::ctl::mdopus_decoder_ctl_set_int as *const ());
    black_box(mdopus_capi::ctl::mdopus_decoder_ctl_get_int as *const ());
    black_box(mdopus_capi::ctl::mdopus_decoder_ctl_get_uint32 as *const ());
    black_box(mdopus_capi::ctl::mdopus_ms_encoder_ctl_reset as *const ());
    black_box(mdopus_capi::ctl::mdopus_ms_encoder_ctl_set_int as *const ());
    black_box(mdopus_capi::ctl::mdopus_ms_encoder_ctl_get_int as *const ());
    black_box(mdopus_capi::ctl::mdopus_ms_encoder_ctl_get_uint32 as *const ());
    black_box(mdopus_capi::ctl::mdopus_ms_encoder_ctl_get_encoder_state as *const ());
    black_box(mdopus_capi::ctl::mdopus_ms_decoder_ctl_reset as *const ());
    black_box(mdopus_capi::ctl::mdopus_ms_decoder_ctl_set_int as *const ());
    black_box(mdopus_capi::ctl::mdopus_ms_decoder_ctl_get_int as *const ());
    black_box(mdopus_capi::ctl::mdopus_ms_decoder_ctl_get_uint32 as *const ());
    black_box(mdopus_capi::ctl::mdopus_ms_decoder_ctl_get_decoder_state as *const ());
}

#[test]
fn api() {
    force_link();
    let rc = unsafe { test_opus_api_main(0, std::ptr::null()) };
    assert_eq!(rc, 0, "test_opus_api returned {}", rc);
}
