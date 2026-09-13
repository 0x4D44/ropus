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

use std::io::{self, Read};
use std::process::{Child, Command, Output, Stdio};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const PLAYER_CHILD_TIMEOUT: Duration = Duration::from_secs(30);
const HANGING_CHILD_TIMEOUT: Duration = Duration::from_millis(250);

fn run_child_with_timeout(
    mut command: Command,
    description: &str,
    timeout: Duration,
) -> io::Result<Output> {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;

        command.process_group(0);
    }

    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stdout_reader = spawn_reader(child.stdout.take().expect("stdout was piped"));
    let stderr_reader = spawn_reader(child.stderr.take().expect("stderr was piped"));
    let deadline = Instant::now() + timeout;

    let status = loop {
        match child.try_wait()? {
            Some(status) => break status,
            None if Instant::now() >= deadline => {
                terminate_process_tree(&mut child);
                let _ = child.wait();
                let _ = join_reader(stdout_reader);
                let _ = join_reader(stderr_reader);
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!(
                        "{description} timed out after {timeout:?}; process was killed and reaped"
                    ),
                ));
            }
            None => thread::sleep(Duration::from_millis(10)),
        }
    };

    Ok(Output {
        status,
        stdout: join_reader(stdout_reader)?,
        stderr: join_reader(stderr_reader)?,
    })
}

fn run_player(args: &[&str], description: &str) -> io::Result<Output> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_ropusplay"));
    command.args(args);
    run_child_with_timeout(command, description, PLAYER_CHILD_TIMEOUT)
}

fn spawn_reader(mut pipe: impl Read + Send + 'static) -> JoinHandle<io::Result<Vec<u8>>> {
    thread::spawn(move || {
        let mut bytes = Vec::new();
        pipe.read_to_end(&mut bytes)?;
        Ok(bytes)
    })
}

fn join_reader(reader: JoinHandle<io::Result<Vec<u8>>>) -> io::Result<Vec<u8>> {
    reader
        .join()
        .map_err(|_| io::Error::other("child output reader panicked"))?
}

fn assert_no_device_failure_stdout(output: &Output, description: &str) {
    assert!(
        output.stdout.is_empty(),
        "{description} must keep stdout empty on the no-device branch; stdout={:?} stderr={:?}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output
            .stdout
            .iter()
            .all(|&byte| byte >= 0x20 && !(0x80..=0x9F).contains(&byte)),
        "{description} must not emit terminal controls on stdout; stdout={:?}",
        String::from_utf8_lossy(&output.stdout)
    );
}

fn terminate_process_tree(child: &mut Child) {
    let pid = child.id().to_string();

    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/PID", &pid, "/T", "/F"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }

    #[cfg(unix)]
    {
        let process_group = format!("-{pid}");
        let _ = Command::new("kill")
            .args(["-KILL", &process_group])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }

    #[cfg(not(any(unix, windows)))]
    let _ = pid;

    let _ = child.kill();
}

#[test]
fn clap_errors_escape_invalid_values_before_formatting() {
    let hostile =
        "not-a-number\u{0007}\u{001B}]0;title\u{0085}\u{009B}\r\n\u{061C}\u{200E}\u{200F}\u{202E}";
    let output = Command::new(env!("CARGO_BIN_EXE_ropusplay"))
        .args(["--no-color", "--volume", hostile, "input.opus"])
        .output()
        .expect("spawn ropusplay with invalid hostile volume");

    assert_eq!(output.status.code(), Some(2));
    assert!(!output.stderr.contains(&0));
    assert!(!output.stderr.contains(&0x07));
    assert!(!output.stderr.contains(&0x1B));
    let stderr = String::from_utf8_lossy(&output.stderr);
    for escaped in [
        r"\u{0007}",
        r"\u{001B}",
        r"\u{0085}",
        r"\u{009B}",
        r"\u{000D}",
        r"\u{000A}",
        r"\u{061C}",
        r"\u{200E}",
        r"\u{200F}",
        r"\u{202E}",
    ] {
        assert!(stderr.contains(escaped), "missing {escaped} in {stderr:?}");
    }
    assert!(!stderr.chars().any(|c| {
        matches!(
            c,
            '\u{0085}' | '\u{009B}' | '\u{061C}' | '\u{200E}' | '\u{200F}' | '\u{202E}'
        )
    }));
}

/// `--list-devices` prints at least one device name on stdout and exits 0.
/// Degrades gracefully on hosts with no audio devices — exit 1 there is the
/// documented "no devices" contract, not a test failure.
#[test]
fn list_devices_prints_lines_and_exits_zero() {
    // `--quiet` suppresses the banner so stdout is purely the device list —
    // keeps this test focused on the flag's output, not banner formatting.
    let out = run_player(&["--quiet", "--list-devices"], "ropusplay --list-devices")
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
        assert_no_device_failure_stdout(&out, "ropusplay --list-devices");
        return;
    }

    let lines: Vec<&str> = stdout.lines().collect();
    assert!(
        !lines.is_empty(),
        "expected at least one device line on stdout; stdout={stdout:?} stderr={stderr:?}"
    );
    assert!(
        lines.iter().all(|line| !line.is_empty()),
        "device listing must contain exactly one non-empty name per line; stdout={stdout:?}"
    );
    assert!(
        stdout.ends_with('\n'),
        "device listing must end with a newline; stdout={stdout:?}"
    );
}

#[test]
fn list_devices_without_quiet_has_no_banner_pollution() {
    let out = run_player(
        &["--no-color", "--list-devices"],
        "ropusplay --no-color --list-devices",
    )
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
        assert_no_device_failure_stdout(&out, "ropusplay --no-color --list-devices");
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
    let bogus = "_definitely_not_a_device_";
    // We must supply a positional `input` so clap doesn't reject us before
    // the command body runs; the path is never opened because device
    // resolution fails first.
    let out = run_player(
        &[
            "--quiet",
            "--device",
            bogus,
            "C:/this/path/does/not/exist.opus",
        ],
        "ropusplay --device <bogus>",
    )
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

#[test]
fn hanging_helper_child() {
    if std::env::var_os("ROPUSPLAY_HANG_HELPER").is_some() {
        loop {
            thread::sleep(Duration::from_secs(60));
        }
    }
}

#[test]
fn child_timeout_reports_distinct_error_after_cleanup() {
    let mut command = Command::new(std::env::current_exe().expect("locate CLI test binary"));
    command
        .args(["hanging_helper_child", "--exact", "--nocapture"])
        .env("ROPUSPLAY_HANG_HELPER", "1");

    let error = run_child_with_timeout(
        command,
        "ropusplay hanging test child",
        HANGING_CHILD_TIMEOUT,
    )
    .expect_err("hanging child must time out");

    assert_eq!(
        error.kind(),
        io::ErrorKind::TimedOut,
        "timeout must have a distinct error kind: {error}"
    );
    assert!(
        error.to_string().contains("timed out") && error.to_string().contains("killed and reaped"),
        "timeout diagnostic must identify cleanup: {error}"
    );
}
