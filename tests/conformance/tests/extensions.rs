//! Conformance test: `reference/tests/test_opus_extensions.c`.
//!
//! Exercises the extension iterator, the five public
//! `opus_packet_extensions_*` entry points, plus `opus_packet_parse_impl`
//! and `opus_repacketizer_out_range_impl` (both declared in the upstream
//! private header). Built as its own static lib via build.rs with
//! `-Dmain=test_opus_extensions_main`.
//!
//! Requires `-- --test-threads=1` like the other conformance tests: shared
//! `test_opus_common.h` RNG state (`Rz`, `Rw`, `iseed`) lives at module
//! scope and would race between threads.
//!
//! The final subtest (`test_random_extensions_parse`) loops 100_000_000
//! times over random parse+generate cycles — this is the slowest part of
//! the suite. Expect wall-clock on the order of several minutes in
//! release mode; `mdtimeout 600` from the phase-5 brief is the budget.

use std::hint::black_box;
use std::os::raw::{c_char, c_int};

unsafe extern "C" {
    fn test_opus_extensions_main(argc: c_int, argv: *const *const c_char) -> c_int;
}

/// Pull every `#[unsafe(no_mangle)]` symbol `test_opus_extensions.c` can
/// resolve out of the `capi` rlib.
fn force_link() {
    // Library-level
    black_box(mdopus_capi::opus_strerror as *const ());
    black_box(mdopus_capi::opus_get_version_string as *const ());

    // Repacketizer + out_range_impl
    black_box(mdopus_capi::repacketizer::opus_repacketizer_get_size as *const ());
    black_box(mdopus_capi::repacketizer::opus_repacketizer_create as *const ());
    black_box(mdopus_capi::repacketizer::opus_repacketizer_init as *const ());
    black_box(mdopus_capi::repacketizer::opus_repacketizer_destroy as *const ());
    black_box(mdopus_capi::repacketizer::opus_repacketizer_cat as *const ());
    black_box(mdopus_capi::repacketizer::opus_repacketizer_out as *const ());
    black_box(mdopus_capi::repacketizer::opus_repacketizer_out_range as *const ());
    black_box(mdopus_capi::repacketizer::opus_repacketizer_out_range_impl as *const ());
    black_box(mdopus_capi::repacketizer::opus_repacketizer_get_nb_frames as *const ());

    // Packet parse (test line 691)
    black_box(mdopus_capi::packet_parse::opus_packet_parse as *const ());
    black_box(mdopus_capi::packet_parse::opus_packet_parse_impl as *const ());

    // Extensions
    black_box(mdopus_capi::extensions::opus_packet_extensions_count as *const ());
    black_box(mdopus_capi::extensions::opus_packet_extensions_count_ext as *const ());
    black_box(mdopus_capi::extensions::opus_packet_extensions_parse as *const ());
    black_box(mdopus_capi::extensions::opus_packet_extensions_parse_ext as *const ());
    black_box(mdopus_capi::extensions::opus_packet_extensions_generate as *const ());
}

#[test]
fn extensions() {
    force_link();
    let rc = unsafe { test_opus_extensions_main(0, std::ptr::null()) };
    assert_eq!(rc, 0, "test_opus_extensions returned {}", rc);
}
