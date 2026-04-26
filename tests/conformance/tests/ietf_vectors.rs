#![cfg(not(no_reference))]
//! Conformance test: IETF RFC 6716 / RFC 8251 bitstream vectors.
//!
//! This test drives `opus_demo -d` against the 12 IETF test vector
//! bitstreams in both mono and stereo, then runs `opus_compare` to
//! validate the decoded PCM against the reference PCM shipped alongside
//! the bitstreams. Mirrors `reference/tests/run_vectors.sh`.
//!
//! Vectors must be fetched first via `tools/fetch_ietf_vectors.sh` (or
//! `.ps1` on Windows) — they land in `tests/vectors/ietf/`, which is
//! gitignored. If the directory is missing the test panics with a
//! helpful message.
//!
//! Serialisation: like the other conformance tests, `opus_demo.c` has
//! file-scope globals (RNG state in `test_opus_common.h`, FILE* handles
//! mid-function). Calling its `main()` once per vector in a single
//! process requires strict serialisation — run with `-- --test-threads=1`.
//!
//! `opus_compare.c` returns 0 on quality-threshold pass, non-zero on
//! fail. Per the shell script's behaviour at lines 90-95, we accept
//! either the `.dec` reference OR the `m.dec` (alternative vector)
//! reference matching — they exist to accommodate implementation
//! flexibility in the reference.

use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

// Compiled by build.rs as `libopus_demo.a` / `libopus_compare.a` with their
// `main` symbols renamed. Our driver assembles the same argv that
// run_vectors.sh would build and calls these entry points in-process.
unsafe extern "C" {
    fn opus_demo_main(argc: c_int, argv: *const *const c_char) -> c_int;
    fn opus_compare_main(argc: c_int, argv: *const *const c_char) -> c_int;
}

/// Force the linker to pull `#[unsafe(no_mangle)]` symbols from the
/// `capi` rlib so the opus_demo static lib can resolve `opus_decode24`,
/// `opus_decoder_create`, etc. Same idiom as the other conformance tests.
fn force_link() {
    use std::hint::black_box;
    black_box(mdopus_capi::opus_strerror as *const ());
    black_box(mdopus_capi::opus_get_version_string as *const ());
    black_box(mdopus_capi::decoder::opus_decoder_get_size as *const ());
    black_box(mdopus_capi::decoder::opus_decoder_create as *const ());
    black_box(mdopus_capi::decoder::opus_decoder_init as *const ());
    black_box(mdopus_capi::decoder::opus_decoder_destroy as *const ());
    black_box(mdopus_capi::decoder::opus_decode as *const ());
    black_box(mdopus_capi::decoder::opus_decode24 as *const ());
    black_box(mdopus_capi::decoder::opus_decode_float as *const ());
    black_box(mdopus_capi::decoder::opus_packet_get_bandwidth as *const ());
    black_box(mdopus_capi::decoder::opus_packet_get_nb_channels as *const ());
    black_box(mdopus_capi::decoder::opus_packet_get_nb_frames as *const ());
    black_box(mdopus_capi::decoder::opus_packet_get_nb_samples as *const ());
    black_box(mdopus_capi::decoder::opus_packet_get_samples_per_frame as *const ());
    black_box(mdopus_capi::decoder::opus_decoder_get_nb_samples as *const ());
    black_box(mdopus_capi::decoder::opus_pcm_soft_clip as *const ());
    // Encoder side — opus_demo's encode arm is dead under `-d`, but the
    // symbols are unconditionally referenced so we still have to resolve
    // them at link time.
    black_box(mdopus_capi::encoder::opus_encoder_get_size as *const ());
    black_box(mdopus_capi::encoder::opus_encoder_create as *const ());
    black_box(mdopus_capi::encoder::opus_encoder_init as *const ());
    black_box(mdopus_capi::encoder::opus_encoder_destroy as *const ());
    black_box(mdopus_capi::encoder::opus_encode as *const ());
    black_box(mdopus_capi::encoder::opus_encode_float as *const ());
    // Multistream — opus_demo includes `opus_multistream.h` and may link
    // MS decoder create/destroy even in decode-only mode (the #if 0 block
    // at opus_demo.c:306 is disabled so these aren't called, but the
    // header drags declarations in).
    black_box(mdopus_capi::ms_decoder::opus_multistream_decoder_create as *const ());
    black_box(mdopus_capi::ms_decoder::opus_multistream_decoder_init as *const ());
    black_box(mdopus_capi::ms_decoder::opus_multistream_decoder_destroy as *const ());
    black_box(mdopus_capi::ms_decoder::opus_multistream_decode as *const ());
    black_box(mdopus_capi::ms_decoder::opus_multistream_decode24 as *const ());
    black_box(mdopus_capi::ms_decoder::opus_multistream_decode_float as *const ());
    black_box(mdopus_capi::ms_decoder::opus_multistream_decoder_get_size as *const ());
    // CTL dispatch.
    black_box(mdopus_capi::ctl::mdopus_decoder_ctl_reset as *const ());
    black_box(mdopus_capi::ctl::mdopus_decoder_ctl_set_int as *const ());
    black_box(mdopus_capi::ctl::mdopus_decoder_ctl_get_int as *const ());
    black_box(mdopus_capi::ctl::mdopus_decoder_ctl_get_uint32 as *const ());
    black_box(mdopus_capi::ctl::mdopus_encoder_ctl_reset as *const ());
    black_box(mdopus_capi::ctl::mdopus_encoder_ctl_set_int as *const ());
    black_box(mdopus_capi::ctl::mdopus_encoder_ctl_get_int as *const ());
    black_box(mdopus_capi::ctl::mdopus_encoder_ctl_get_uint32 as *const ());
}

fn vectors_dir() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // manifest = <repo>/tests/conformance → <repo>/tests/vectors/ietf
    manifest.parent().unwrap().join("vectors").join("ietf")
}

/// A per-test output directory. Uses an atomic counter plus the process
/// ID to avoid collisions even though we serialise the tests.
fn test_output_dir(stem: &str, channels: i32) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let c = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "ropus_ietf_{}_{}_ch{}_{}",
        std::process::id(),
        stem,
        channels,
        c
    ));
    std::fs::create_dir_all(&dir).expect("create test output dir");
    dir
}

fn path_to_cstring(p: &Path) -> CString {
    CString::new(p.to_string_lossy().into_owned()).expect("path has no NUL byte")
}

/// Run `opus_demo -d <rate> <channels> -ignore_extensions <bit> <out>`.
fn run_opus_demo(rate: i32, channels: i32, bit: &Path, out: &Path) -> i32 {
    let argv_strings: Vec<CString> = vec![
        CString::new("opus_demo").unwrap(),
        CString::new("-d").unwrap(),
        CString::new(rate.to_string()).unwrap(),
        CString::new(channels.to_string()).unwrap(),
        CString::new("-ignore_extensions").unwrap(),
        path_to_cstring(bit),
        path_to_cstring(out),
    ];
    let argv_ptrs: Vec<*const c_char> = argv_strings.iter().map(|s| s.as_ptr()).collect();
    unsafe { opus_demo_main(argv_ptrs.len() as c_int, argv_ptrs.as_ptr()) }
}

/// Run `opus_compare [-s] -r <rate> <ref> <out>`.
fn run_opus_compare(channels: i32, rate: i32, reference: &Path, out: &Path) -> i32 {
    let mut argv_strings: Vec<CString> = vec![CString::new("opus_compare").unwrap()];
    if channels == 2 {
        argv_strings.push(CString::new("-s").unwrap());
    }
    argv_strings.push(CString::new("-r").unwrap());
    argv_strings.push(CString::new(rate.to_string()).unwrap());
    argv_strings.push(path_to_cstring(reference));
    argv_strings.push(path_to_cstring(out));

    let argv_ptrs: Vec<*const c_char> = argv_strings.iter().map(|s| s.as_ptr()).collect();
    unsafe { opus_compare_main(argv_ptrs.len() as c_int, argv_ptrs.as_ptr()) }
}

/// Drive one vector × channels combination.
fn run_one(stem: &str, channels: i32, rate: i32) {
    force_link();

    let vectors = vectors_dir();
    if !vectors.exists() {
        panic!(
            "IETF vectors not present at {}. \
             Run `tools/fetch_ietf_vectors.sh` (or `.ps1` on Windows) first.",
            vectors.display()
        );
    }

    let bit = vectors.join(format!("{stem}.bit"));
    assert!(bit.exists(), "bitstream missing: {}", bit.display());

    let tmp = test_output_dir(stem, channels);
    let out = tmp.join(format!("{stem}_{channels}ch.pcm"));

    let rc = run_opus_demo(rate, channels, &bit, &out);
    assert_eq!(
        rc,
        0,
        "opus_demo -d {rate} {channels} -ignore_extensions {} {}: exit {rc}",
        bit.display(),
        out.display()
    );

    // Reference PCM. run_vectors.sh tries both `testvectorNN.dec` and
    // `testvectorNNm.dec` and passes if either matches — the `m` variant
    // is an alternative reference PCM for implementations that make
    // slightly different (still RFC-compliant) choices.
    let ref_primary = vectors.join(format!("{stem}.dec"));
    let ref_alt = vectors.join(format!("{stem}m.dec"));

    let r_primary = if ref_primary.exists() {
        run_opus_compare(channels, rate, &ref_primary, &out)
    } else {
        -1
    };
    let r_alt = if ref_alt.exists() {
        run_opus_compare(channels, rate, &ref_alt, &out)
    } else {
        -1
    };

    assert!(
        r_primary == 0 || r_alt == 0,
        "{stem} ch{channels} @ {rate}: neither primary ({}: rc={r_primary}) \
         nor alternative ({}: rc={r_alt}) reference matched",
        ref_primary.display(),
        ref_alt.display()
    );

    // Best-effort cleanup; OK to leave files on failure for debugging.
    let _ = std::fs::remove_dir_all(&tmp);
}

// ---------------------------------------------------------------------------
// The 12 IETF vectors × {mono, stereo} = 24 tests.
// Vectors are decoded at 48 kHz (the RFC-mandated test rate).
// ---------------------------------------------------------------------------

macro_rules! vector_test {
    ($name:ident, $stem:expr, $ch:expr) => {
        #[test]
        fn $name() {
            run_one($stem, $ch, 48000);
        }
    };
    ($name:ident, $stem:expr, $ch:expr, ignore = $reason:expr) => {
        #[test]
        #[ignore = $reason]
        fn $name() {
            run_one($stem, $ch, 48000);
        }
    };
}

vector_test!(testvector01_mono, "testvector01", 1);
vector_test!(testvector01_stereo, "testvector01", 2);
vector_test!(testvector02_mono, "testvector02", 1);
vector_test!(testvector02_stereo, "testvector02", 2);
vector_test!(testvector03_mono, "testvector03", 1);
vector_test!(testvector03_stereo, "testvector03", 2);
vector_test!(testvector04_mono, "testvector04", 1);
vector_test!(testvector04_stereo, "testvector04", 2);
vector_test!(testvector05_mono, "testvector05", 1);
vector_test!(testvector05_stereo, "testvector05", 2);
vector_test!(testvector06_mono, "testvector06", 1);
vector_test!(testvector06_stereo, "testvector06", 2);
vector_test!(testvector07_mono, "testvector07", 1);
vector_test!(testvector07_stereo, "testvector07", 2);
vector_test!(testvector08_mono, "testvector08", 1);
vector_test!(testvector08_stereo, "testvector08", 2);
vector_test!(testvector09_mono, "testvector09", 1);
vector_test!(testvector09_stereo, "testvector09", 2);
vector_test!(testvector10_mono, "testvector10", 1);
vector_test!(testvector10_stereo, "testvector10", 2);
vector_test!(testvector11_mono, "testvector11", 1);
vector_test!(testvector11_stereo, "testvector11", 2);
vector_test!(testvector12_mono, "testvector12", 1);
vector_test!(testvector12_stereo, "testvector12", 2);
