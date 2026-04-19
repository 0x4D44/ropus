#![cfg(not(no_reference))]
//! Conformance test: `reference/tests/test_opus_padding.c`.
//!
//! Cargo auto-discovers each `tests/<name>.rs` as its own integration-test
//! binary. Keeping one reference test per file is load-bearing: the shared
//! `test_opus_common.h` declares `void regression_test(void)`, which each
//! `test_opus_*.c` file then defines — co-linking two of them produces
//! LNK2005 (duplicate symbol).
//!
//! Tests **must** run single-threaded (`-- --test-threads=1`) because the
//! reference tests share module-global RNG state (`Rz`/`Rw`/`iseed` in
//! `test_opus_common.h`) within their process image.
//!
//! Linker note: the `capi` crate exposes `#[unsafe(no_mangle)]` symbols the
//! C tests link against. The Rust linker only pulls rlib symbols that are
//! actually referenced from Rust code, so `force_link()` takes the address of
//! every symbol the C tests might call. `std::hint::black_box` prevents the
//! optimiser from constant-folding the references away.

use std::hint::black_box;
use std::os::raw::{c_char, c_int};

// Compiled by build.rs as `libtest_opus_padding.a`, with `-Dmain=test_opus_padding_main`.
unsafe extern "C" {
    fn test_opus_padding_main(argc: c_int, argv: *const *const c_char) -> c_int;
}

/// Take the address of every `#[unsafe(no_mangle)]` symbol the C tests might
/// call, so the linker pulls them out of the `capi` rlib. The lib crate name
/// is `mdopus_capi` per its `Cargo.toml` `[lib] name`.
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
    // CTL
    black_box(mdopus_capi::ctl::mdopus_encoder_ctl_reset as *const ());
    black_box(mdopus_capi::ctl::mdopus_decoder_ctl_reset as *const ());
}

#[test]
fn padding() {
    force_link();
    let rc = unsafe { test_opus_padding_main(0, std::ptr::null()) };
    assert_eq!(rc, 0, "test_opus_padding returned {}", rc);
}
