//! Conformance test: `reference/tests/test_opus_encode.c`.
//!
//! 756-line encoder contract test that combines targeted API probes
//! (`OPUS_SET_FORCE_MODE(-2)` returns `OPUS_BAD_ARG`, `opus_encode` with
//! invalid frame size returns `OPUS_BAD_ARG`) with long encode+decode
//! loops across mode/rate/bandwidth/frame-size matrices for both the
//! single-stream encoder and a 2-stream dual-mono multistream encoder.
//! Exercises `opus_packet_pad`, `opus_packet_unpad`,
//! `opus_multistream_packet_pad`, `opus_multistream_packet_unpad`, and
//! `opus_packet_parse` as part of the loop.
//!
//! `test_opus_encode.c` also calls `regression_test()`, defined upstream in
//! `opus_encode_regressions.c`. With Pieces A+B landed, `build.rs` compiles
//! that file as an extra so the 11 historical crash repros (7 unconditional
//! + 3 float-API + 1 projection overflow) run against our codec. The five
//! `ENABLE_QEXT`/`ENABLE_DRED`-guarded repros compile out.
//!
//! This file is its own `[[test]]` binary (cargo auto-discovery). Running
//! under `-- --test-threads=1` is required; the reference tests share
//! process-global RNG state in `test_opus_common.h`.

use std::hint::black_box;
use std::os::raw::{c_char, c_int};

// Compiled by build.rs as `libtest_opus_encode.a`, with
// `-Dmain=test_opus_encode_main`.
unsafe extern "C" {
    fn test_opus_encode_main(argc: c_int, argv: *const *const c_char) -> c_int;
}

/// Pull every `#[unsafe(no_mangle)]` symbol `test_opus_encode.c` and the
/// `opus_encode_regressions.c` extra can resolve out of the `capi` rlib.
/// Exhaustive — we prefer over-linking to a mysterious undefined symbol at
/// link time.
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

    // Repacketizer + MS pad/unpad (MS pad/unpad is called from
    // test_opus_encode.c lines 569, 574, 580).
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

    // Packet parse (line 636).
    black_box(mdopus_capi::packet_parse::opus_packet_parse as *const ());

    // CTL shim entry points — OPUS_SET_FORCE_MODE routes into the encoder /
    // ms-encoder set_int dispatchers. OPUS_GET_PREDICTION_DISABLED for MS is
    // called on line 542.
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

    // Projection encoder — pulled in by `opus_encode_regressions.c`'s
    // `projection_overflow{,2,3}` repros. Decoder surface is not touched
    // by regression_test() but we force-link it for parity with the
    // projection test binary (keeps the symbol set closed).
    black_box(
        mdopus_capi::projection::opus_projection_ambisonics_encoder_get_size as *const (),
    );
    black_box(
        mdopus_capi::projection::opus_projection_ambisonics_encoder_create as *const (),
    );
    black_box(mdopus_capi::projection::opus_projection_ambisonics_encoder_init as *const ());
    black_box(mdopus_capi::projection::opus_projection_encoder_destroy as *const ());
    black_box(mdopus_capi::projection::opus_projection_encode as *const ());
    black_box(mdopus_capi::projection::opus_projection_encode_float as *const ());
    black_box(mdopus_capi::ctl::mdopus_proj_encoder_ctl_reset as *const ());
    black_box(mdopus_capi::ctl::mdopus_proj_encoder_ctl_set_int as *const ());
    black_box(mdopus_capi::ctl::mdopus_proj_encoder_ctl_get_int as *const ());
    black_box(mdopus_capi::ctl::mdopus_proj_encoder_ctl_get_uint32 as *const ());
    black_box(
        mdopus_capi::ctl::mdopus_proj_encoder_ctl_get_demixing_matrix as *const (),
    );
}

#[test]
fn encode() {
    force_link();
    let rc = unsafe { test_opus_encode_main(0, std::ptr::null()) };
    assert_eq!(rc, 0, "test_opus_encode returned {}", rc);
}
