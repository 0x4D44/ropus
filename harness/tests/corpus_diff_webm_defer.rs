#![cfg(not(no_reference))]
//! Integration test for the corpus_diff WebM defer path.
//!
//! Verifies that a directory containing only a WebM file produces:
//! - exit code 3 (all candidates deferred)
//! - a per-file `DEFER ... reason=webm-matroska-container-deferred` stdout line

use std::fs;
use std::process::Command;

#[test]
fn webm_only_directory_defers_with_exit_three_and_reason_line() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let webm_path = tmp.path().join("zero-byte.webm");
    fs::write(&webm_path, b"").expect("create zero-byte webm");

    let bin = env!("CARGO_BIN_EXE_corpus_diff");
    let output = Command::new(bin)
        .arg(tmp.path())
        .output()
        .expect("run corpus_diff");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("STDOUT:\n{stdout}\nSTDERR:\n{stderr}");

    assert_eq!(
        output.status.code(),
        Some(3),
        "expected exit 3 for all-deferred set; got {:?}\n{combined}",
        output.status.code()
    );
    let defer_line_present = stdout.lines().any(|line| {
        line.starts_with("  DEFER ") && line.contains("reason=webm-matroska-container-deferred")
    });
    assert!(
        defer_line_present,
        "missing per-file DEFER line with webm-matroska-container-deferred reason\n{combined}"
    );
}
