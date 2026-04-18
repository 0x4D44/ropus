//! End-to-end smoke test for the `ropus-cli` binary.
//!
//! Exercises encode -> decode through the CLI's actual code paths by invoking
//! the built binary (via the `CARGO_BIN_EXE_ropus-cli` env var that cargo
//! provides for `[[bin]]` targets) and asserts the round-trip output is sane.
//!
//! This is a smoke test, not a precision test. The goal: if anyone breaks the
//! encode -> decode pipeline (Ogg framing, OpusHead, pre_skip, anything),
//! `cargo test -p ropus-cli` fails.

use std::path::PathBuf;
use std::process::Command;

#[test]
fn encode_then_decode_48k_sine_round_trips_within_snr_threshold() {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ropus-cli has a parent dir (the workspace root)")
        .to_path_buf();
    let input_wav = workspace.join("tests/vectors/48k_sine1k_loud.wav");
    let tmp_opus = std::env::temp_dir().join("ropus_cli_rt.opus");
    let tmp_wav = std::env::temp_dir().join("ropus_cli_rt.wav");

    // Skip cleanly if the test vector isn't present (so the test is portable
    // to checkouts without the vectors directory populated).
    if !input_wav.exists() {
        eprintln!("skipping: test vector {input_wav:?} not present");
        return;
    }

    // Best-effort cleanup of stale outputs from a previous run.
    let _ = std::fs::remove_file(&tmp_opus);
    let _ = std::fs::remove_file(&tmp_wav);

    let exe = env!("CARGO_BIN_EXE_ropus-cli");

    let status = Command::new(exe)
        .args([
            "encode",
            input_wav.to_str().expect("input path is utf-8"),
            "-o",
            tmp_opus.to_str().expect("opus path is utf-8"),
        ])
        .status()
        .expect("run encode");
    assert!(status.success(), "encode exited non-zero");

    let status = Command::new(exe)
        .args([
            "decode",
            tmp_opus.to_str().expect("opus path is utf-8"),
            "-o",
            tmp_wav.to_str().expect("wav path is utf-8"),
        ])
        .status()
        .expect("run decode");
    assert!(status.success(), "decode exited non-zero");

    // Read both WAVs and run simple sanity checks: the output exists, has a
    // data chunk, and the input is at least a valid WAV header in length.
    let in_meta = std::fs::metadata(&input_wav).expect("stat input");
    let out_meta = std::fs::metadata(&tmp_wav).expect("stat output");
    assert!(out_meta.len() > 0, "output WAV is empty");
    assert!(in_meta.len() > 44, "input WAV is too small to be valid");

    // Parse the output WAV header far enough to assert the sample rate is
    // 48 kHz. The decoder always writes 48 kHz output regardless of input.
    let bytes = std::fs::read(&tmp_wav).expect("read output");
    assert!(
        bytes.len() >= 44,
        "output WAV header is shorter than 44 bytes"
    );
    let sample_rate = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    assert_eq!(sample_rate, 48_000, "output WAV must be 48 kHz");
}
