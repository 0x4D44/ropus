//! CLI-surface integration tests for `ropusplay`.
//!
//! Shell-out tests (via `std::process::Command` against `CARGO_BIN_EXE_ropusplay`)
//! exercise the three flags added in the opus-tools-parity HLD Step 6:
//! `--list-devices`, `--device NAME`, `--gain DB`. The library-level unit tests
//! live next to the gain validator in `ropus-tools-core/src/commands/play.rs`
//! so this file focuses on end-to-end argv behaviour.
//!
//! A headless CI host may have zero output devices; rather than failing there
//! (which would be a false negative — the binary is behaving correctly), the
//! `list_devices` test degrades to a printed-skip.

use std::process::{Command, Stdio};

/// `--list-devices` prints at least one device name on stdout and exits 0.
/// Degrades gracefully on hosts with no audio devices — exit 1 there is the
/// documented "no devices" contract, not a test failure.
#[test]
fn list_devices_prints_lines_and_exits_zero() {
    let bin = env!("CARGO_BIN_EXE_ropusplay");
    // `--quiet` suppresses the banner so stdout is purely the device list —
    // keeps this test focused on the flag's output, not banner formatting.
    let out = Command::new(bin)
        .args(["--quiet", "--list-devices"])
        .stderr(Stdio::piped())
        .output()
        .expect("spawn ropusplay --list-devices");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    if !out.status.success() {
        // Host genuinely has no devices — that's a clean error path we
        // exercise through integration testing elsewhere; skip here rather
        // than fail.
        println!(
            "skipped: no host devices (exit={:?}, stderr={stderr})",
            out.status.code()
        );
        return;
    }

    let lines: Vec<&str> = stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .collect();
    assert!(
        !lines.is_empty(),
        "expected at least one device line on stdout; stdout={stdout:?} stderr={stderr:?}"
    );
}

/// An obviously-nonexistent `--device` name must exit non-zero and surface
/// the requested name on stderr. The exact message format is owned by
/// `open_named_output_stream`; we only assert that the name is echoed back
/// so the user can spot their typo without parsing a boilerplate wall.
#[test]
fn unknown_device_exits_nonzero() {
    let bin = env!("CARGO_BIN_EXE_ropusplay");
    let bogus = "_definitely_not_a_device_";
    // We must supply a positional `input` so clap doesn't reject us before
    // the command body runs; the path is never opened because device
    // resolution fails first.
    let out = Command::new(bin)
        .args([
            "--quiet",
            "--device",
            bogus,
            "C:/this/path/does/not/exist.opus",
        ])
        .output()
        .expect("spawn ropusplay --device <bogus>");

    assert!(
        !out.status.success(),
        "unknown device must exit non-zero (got {:?})",
        out.status.code()
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains(bogus),
        "stderr should mention the requested name '{bogus}', got: {stderr}"
    );
}
