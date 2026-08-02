//! CLI-surface integration tests for `ropusplay`.
//!
//! Shell-out tests (via `std::process::Command` against `CARGO_BIN_EXE_ropusplay`)
//! exercise the three flags added in the opus-tools-parity HLD Step 6:
//! `--list-devices`, `--device NAME`, `--gain DB`. The library-level unit tests
//! live next to the gain validator in `ropus-tools-core/src/commands/play.rs`
//! so this file focuses on end-to-end argv behaviour.
//!
//! A headless CI host may have zero output devices. The test accepts only the
//! command's structured no-device error; panics, argument failures, and other
//! enumeration errors remain test failures.

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
        let lower = stderr.to_ascii_lowercase();
        assert_eq!(
            out.status.code(),
            Some(1),
            "unexpected --list-devices failure: stderr={stderr:?}"
        );
        assert!(
            lower.contains("no output devices available"),
            "only the structured no-device outcome may be accepted; stderr={stderr:?}"
        );
        return;
    }

    let lines: Vec<&str> = stdout.lines().filter(|l| !l.trim().is_empty()).collect();
    assert!(
        !lines.is_empty(),
        "expected at least one device line on stdout; stdout={stdout:?} stderr={stderr:?}"
    );
}

#[test]
fn list_devices_without_quiet_has_no_banner_pollution() {
    let bin = env!("CARGO_BIN_EXE_ropusplay");
    let out = Command::new(bin)
        .args(["--no-color", "--list-devices"])
        .stderr(Stdio::piped())
        .output()
        .expect("spawn ropusplay --list-devices without quiet");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !out.status.success() {
        assert_eq!(
            out.status.code(),
            Some(1),
            "unexpected --list-devices failure: stderr={stderr:?}"
        );
        assert!(
            stderr
                .to_ascii_lowercase()
                .contains("no output devices available"),
            "only the structured no-device outcome may be accepted; stderr={stderr:?}"
        );
        return;
    }

    assert!(
        !stdout.contains("(build "),
        "device list must not contain the ropusplay banner: stdout={stdout:?}"
    );
    assert!(
        !stdout.contains('\x1b'),
        "device list must not contain ANSI escapes: stdout={stdout:?}"
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
