#![cfg(not(no_reference))]
//! Conformance test: `reference/tests/test_opus_projection.c`.
//!
//! Exercises two surfaces:
//!   1. The ambisonics mapping-matrix API (`mapping_matrix_*`), compiled
//!      from `reference/src/mapping_matrix.c` verbatim into this test's
//!      static lib. Pure C-reference against hand-specified expected
//!      output — doesn't touch our Rust codec.
//!   2. The public `opus_projection_*` encode/decode path, which goes
//!      through our `capi` crate's thin wrappers over
//!      `ropus::opus::multistream::OpusProjection{Encoder,Decoder}`.
//!
//! Built by `build.rs` with `-Dmain=test_opus_projection_main`. Single
//! `#[test]` binary per cargo convention.
//!
//! Requires `-- --test-threads=1` for parity with the other conformance
//! drivers (reference tests share process-global RNG state in
//! `test_opus_common.h`).

use std::hint::black_box;
use std::os::raw::{c_char, c_int};

// Compiled by build.rs as `libtest_opus_projection.a`, with
// `-Dmain=test_opus_projection_main`.
unsafe extern "C" {
    fn test_opus_projection_main(argc: c_int, argv: *const *const c_char) -> c_int;
}

/// Take the address of every `#[unsafe(no_mangle)]` capi symbol the C test
/// may call, forcing the linker to pull them out of the `mdopus_capi` rlib.
fn force_link() {
    // Library-level
    black_box(mdopus_capi::opus_strerror as *const ());
    black_box(mdopus_capi::opus_get_version_string as *const ());

    // Projection encoder
    black_box(mdopus_capi::projection::opus_projection_ambisonics_encoder_get_size as *const ());
    black_box(mdopus_capi::projection::opus_projection_ambisonics_encoder_create as *const ());
    black_box(mdopus_capi::projection::opus_projection_ambisonics_encoder_init as *const ());
    black_box(mdopus_capi::projection::opus_projection_encoder_destroy as *const ());
    black_box(mdopus_capi::projection::opus_projection_encode as *const ());
    black_box(mdopus_capi::projection::opus_projection_encode_float as *const ());

    // Projection decoder
    black_box(mdopus_capi::projection::opus_projection_decoder_get_size as *const ());
    black_box(mdopus_capi::projection::opus_projection_decoder_create as *const ());
    black_box(mdopus_capi::projection::opus_projection_decoder_init as *const ());
    black_box(mdopus_capi::projection::opus_projection_decoder_destroy as *const ());
    black_box(mdopus_capi::projection::opus_projection_decode as *const ());
    black_box(mdopus_capi::projection::opus_projection_decode_float as *const ());

    // Projection-specific CTL dispatchers (reached through the varargs
    // shim `opus_projection_encoder_ctl` in `ctl_shim.c`).
    black_box(mdopus_capi::ctl::mdopus_proj_encoder_ctl_reset as *const ());
    black_box(mdopus_capi::ctl::mdopus_proj_encoder_ctl_set_int as *const ());
    black_box(mdopus_capi::ctl::mdopus_proj_encoder_ctl_get_int as *const ());
    black_box(mdopus_capi::ctl::mdopus_proj_encoder_ctl_get_uint32 as *const ());
    black_box(mdopus_capi::ctl::mdopus_proj_encoder_ctl_get_demixing_matrix as *const ());
    black_box(mdopus_capi::ctl::mdopus_proj_decoder_ctl_reset as *const ());
    black_box(mdopus_capi::ctl::mdopus_proj_decoder_ctl_set_int as *const ());
    black_box(mdopus_capi::ctl::mdopus_proj_decoder_ctl_get_int as *const ());
    black_box(mdopus_capi::ctl::mdopus_proj_decoder_ctl_get_uint32 as *const ());
}

#[test]
fn projection() {
    force_link();
    let rc = unsafe { test_opus_projection_main(0, std::ptr::null()) };
    assert_eq!(rc, 0, "test_opus_projection returned {}", rc);
}
