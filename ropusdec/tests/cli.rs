//! Integration tests for the `ropusdec` binary's `-` stdin/stdout sentinel.
//!
//! Same architecture as `ropusenc/tests/cli.rs`: shell out to the built
//! binary (via `CARGO_BIN_EXE_ropusdec`) because the stdin/stdout sentinel
//! branch only activates when argv is `-`, and in-process library calls
//! can't force that without also hijacking the test runner's stdin/stdout.
//!
//! Each test first encodes a synthetic sine via the library (`ropus_tools_core`
//! exposes `commands::encode`) to produce a valid Ogg Opus stream, then pipes
//! that stream into `ropusdec -` and checks the output format on stdout.

use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ropus_tools_core::commands;
use ropus_tools_core::options::EncodeOptions;

/// Synthesise a 1 kHz 48 kHz mono sine WAV on disk and return its path. The
/// caller owns cleanup. Duplicated from `ropusenc/tests/cli.rs`; sharing via
/// a `tests/common/` submodule is more ceremony than it's worth for 20 lines.
fn write_sine_wav_tmp(seconds: u32, freq_hz: f32, tag: &str) -> PathBuf {
    let nonce = format!(
        "{}_{}_{}",
        tag,
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let path = std::env::temp_dir().join(format!("ropusdec_cli_{nonce}.wav"));

    let sr: u32 = 48_000;
    let channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let byte_rate = sr * u32::from(channels) * u32::from(bits_per_sample) / 8;
    let block_align = channels * bits_per_sample / 8;
    let num_samples = sr * seconds;
    let data_size = num_samples * u32::from(block_align);
    let riff_size = 36 + data_size;

    let mut out = Vec::with_capacity((44 + data_size) as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sr.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());

    let two_pi = std::f32::consts::TAU;
    for n in 0..num_samples {
        let t = n as f32 / sr as f32;
        let s = (two_pi * freq_hz * t).sin() * 0.6;
        let q = (s * 32767.0) as i16;
        out.extend_from_slice(&q.to_le_bytes());
    }
    std::fs::write(&path, &out).expect("write synth WAV");
    path
}

/// Encode a short sine WAV to Opus using the library (not the binary). Returns
/// the Opus bytes for use as test input to `ropusdec`.
fn encode_sine_to_opus_bytes(tag: &str) -> Vec<u8> {
    let wav_path = write_sine_wav_tmp(1, 1000.0, tag);
    let opus_path = std::env::temp_dir().join(format!(
        "ropusdec_cli_{}_{}.opus",
        tag,
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));

    let enc_opts = EncodeOptions {
        input: wav_path.clone(),
        output: Some(opus_path.clone()),
        bitrate: Some(64_000),
        complexity: None,
        application: ropus_tools_core::Application::Audio,
        vbr: true,
        vbr_constraint: false,
        signal: ropus_tools_core::Signal::Auto,
        frame_duration: ropus_tools_core::FrameDuration::Ms20,
        expect_loss: 0,
        downmix_to_mono: false,
        serial: None,
        picture_path: None,
        vendor: "ropusdec-cli-test".to_string(),
        comments: Vec::new(),
    };
    commands::encode(enc_opts).expect("encode fixture");

    let bytes = std::fs::read(&opus_path).expect("read fixture opus");
    let _ = std::fs::remove_file(&wav_path);
    let _ = std::fs::remove_file(&opus_path);
    bytes
}

fn ropusdec_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_ropusdec"))
}

fn temp_path(tag: &str, extension: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after Unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "ropusdec_{tag}_{}_{}.{}",
        std::process::id(),
        nonce,
        extension
    ))
}

const CHILD_TIMEOUT: Duration = Duration::from_secs(30);
const CHILD_POLL_INTERVAL: Duration = Duration::from_millis(10);

#[derive(Debug)]
enum ChildCapture {
    Completed(Output),
    TimedOut(Output),
}

fn run_child(command: &mut Command, input: Option<&[u8]>) -> io::Result<ChildCapture> {
    run_child_with_timeout(command, input, CHILD_TIMEOUT)
}

fn run_child_with_timeout(
    command: &mut Command,
    input: Option<&[u8]>,
    timeout: Duration,
) -> io::Result<ChildCapture> {
    configure_process_group(command);
    command
        .stdin(if input.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn()?;

    let stdin_thread = input.map(|bytes| {
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "child stdin was not piped"));
        let bytes = bytes.to_vec();
        thread::spawn(move || {
            let mut stdin = stdin?;
            stdin.write_all(&bytes)
        })
    });

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "child stdout was not piped"));
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "child stderr was not piped"));
    let (stdout, stderr) = match (stdout, stderr) {
        (Ok(stdout), Ok(stderr)) => (stdout, stderr),
        (Err(error), _) | (_, Err(error)) => {
            let _ = terminate_process_tree(&mut child);
            if let Some(stdin_thread) = stdin_thread {
                let _ = stdin_thread.join();
            }
            return Err(error);
        }
    };

    let stdout_thread = thread::spawn(move || {
        let mut bytes = Vec::new();
        let mut stdout = stdout;
        stdout.read_to_end(&mut bytes)?;
        Ok(bytes)
    });
    let stderr_thread = thread::spawn(move || {
        let mut bytes = Vec::new();
        let mut stderr = stderr;
        stderr.read_to_end(&mut bytes)?;
        Ok(bytes)
    });

    let deadline = Instant::now() + timeout;
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) if Instant::now() >= deadline => {
                timed_out = true;
                break terminate_process_tree(&mut child);
            }
            Ok(None) => {
                let remaining = deadline.saturating_duration_since(Instant::now());
                thread::sleep(remaining.min(CHILD_POLL_INTERVAL));
            }
            Err(error) => {
                let cleanup = terminate_process_tree(&mut child);
                let _ = join_output_reader(stdout_thread);
                let _ = join_output_reader(stderr_thread);
                if let Some(stdin_thread) = stdin_thread {
                    let _ = stdin_thread.join();
                }
                return Err(combine_wait_error(error, cleanup));
            }
        }
    }?;

    if let Some(stdin_thread) = stdin_thread {
        stdin_thread
            .join()
            .map_err(|_| io::Error::other("child stdin writer panicked"))??;
    }
    let output = Output {
        status,
        stdout: join_output_reader(stdout_thread)?,
        stderr: join_output_reader(stderr_thread)?,
    };
    Ok(if timed_out {
        ChildCapture::TimedOut(output)
    } else {
        ChildCapture::Completed(output)
    })
}

fn run_child_expect_complete(command: &mut Command, input: Option<&[u8]>) -> Output {
    match run_child(command, input).expect("run child") {
        ChildCapture::Completed(output) => output,
        ChildCapture::TimedOut(output) => panic!(
            "child timed out after {CHILD_TIMEOUT:?}; stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        ),
    }
}

fn combine_wait_error(wait_error: io::Error, cleanup: io::Result<ExitStatus>) -> io::Error {
    match cleanup {
        Ok(_) => wait_error,
        Err(cleanup_error) => io::Error::other(format!(
            "failed while waiting for child: {wait_error}; cleanup also failed: {cleanup_error}"
        )),
    }
}

fn terminate_process_tree(child: &mut Child) -> io::Result<ExitStatus> {
    let tree_error = kill_process_tree(child.id()).err();
    if let Some(tree_error) = tree_error {
        let direct_error = child.kill().err();
        let status = child.wait();
        return match status {
            Ok(status) => Err(io::Error::other(format!(
                "process-tree cleanup failed: {tree_error}; direct child was reaped with {status}{}",
                direct_error
                    .as_ref()
                    .map(|error| format!("; direct kill also failed: {error}"))
                    .unwrap_or_default()
            ))),
            Err(wait_error) => Err(io::Error::other(format!(
                "process-tree cleanup failed: {tree_error}; waiting for direct child also failed: \
                 {wait_error}{}",
                direct_error
                    .as_ref()
                    .map(|error| format!("; direct kill also failed: {error}"))
                    .unwrap_or_default()
            ))),
        };
    }
    child.wait()
}

#[cfg(unix)]
fn configure_process_group(command: &mut Command) {
    use std::os::unix::process::CommandExt;

    command.process_group(0);
}

#[cfg(not(unix))]
fn configure_process_group(_command: &mut Command) {}

#[cfg(unix)]
fn kill_process_tree(pid: u32) -> io::Result<()> {
    let group = format!("-{pid}");
    let status = Command::new("kill")
        .args(["-KILL", &group])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "kill process group for child {pid} exited with {status}"
        )))
    }
}

#[cfg(windows)]
fn kill_process_tree(pid: u32) -> io::Result<()> {
    let status = Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "taskkill for child {pid} exited with {status}"
        )))
    }
}

#[cfg(not(any(unix, windows)))]
fn kill_process_tree(pid: u32) -> io::Result<()> {
    Err(io::Error::other(format!(
        "process-tree termination is unsupported for child {pid}"
    )))
}

fn join_output_reader(handle: thread::JoinHandle<io::Result<Vec<u8>>>) -> io::Result<Vec<u8>> {
    handle
        .join()
        .map_err(|_| io::Error::other("child output reader panicked"))?
}

#[test]
fn hanging_helper_child() {
    if std::env::var_os("ROPUSDEC_HANG_HELPER").is_some() {
        loop {
            thread::sleep(Duration::from_secs(60));
        }
    }
}

#[test]
fn child_timeout_kills_and_reaps_hanging_helper() {
    let mut command = Command::new(std::env::current_exe().expect("locate test binary"));
    command
        .args(["--exact", "hanging_helper_child"])
        .env("ROPUSDEC_HANG_HELPER", "1");
    let started = Instant::now();
    let result = run_child_with_timeout(&mut command, None, Duration::from_millis(100))
        .expect("supervisor should return a timeout");

    match result {
        ChildCapture::TimedOut(output) => {
            assert!(
                !output.status.success(),
                "hanging helper unexpectedly succeeded"
            );
            assert!(
                started.elapsed() < Duration::from_secs(2),
                "timeout took too long: {:?}",
                started.elapsed()
            );
        }
        ChildCapture::Completed(output) => {
            panic!("hanging helper completed unexpectedly: {:?}", output.status);
        }
    }
}

#[test]
fn stdin_opus_to_stdout_wav() {
    // Build a known Opus stream in-process, then pipe it to `ropusdec - -o -`.
    // Expect stdout to carry a valid WAV: starts with RIFF, contains WAVE at
    // offset 8..12. Stderr absorbs banner/progress text.
    let opus = encode_sine_to_opus_bytes("stdin_wav");

    let mut command = Command::new(ropusdec_bin());
    command.args(["--no-color", "-", "-o", "-"]);
    let output = run_child_expect_complete(&mut command, Some(&opus));
    assert!(
        output.status.success(),
        "ropusdec exited {:?}; stderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
    assert!(
        output.stdout.len() >= 12,
        "stdout too short ({} bytes) to be a WAV header",
        output.stdout.len()
    );
    assert_eq!(
        &output.stdout[..4],
        b"RIFF",
        "stdout must start with 'RIFF'; got: {:02x?}",
        &output.stdout[..16]
    );
    assert_eq!(
        &output.stdout[8..12],
        b"WAVE",
        "WAV must contain 'WAVE' at offset 8..12; got: {:02x?}",
        &output.stdout[..16]
    );
}

#[test]
fn stdin_opus_to_stdout_with_o_attached() {
    // Regression for the argv sniffer: `-o-` (short flag with attached `-`
    // value) must route the banner to stderr and leave stdout as the clean
    // WAV byte stream. Mirrors the ropusenc `-o-` regression.
    let opus = encode_sine_to_opus_bytes("stdin_o_attached");

    let mut command = Command::new(ropusdec_bin());
    command.args(["--no-color", "-", "-o-"]);
    let output = run_child_expect_complete(&mut command, Some(&opus));
    assert!(
        output.status.success(),
        "ropusdec exited {:?}; stderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
    assert_eq!(
        &output.stdout[..4],
        b"RIFF",
        "stdout must start with 'RIFF' when -o- is used; got: {:02x?}",
        &output.stdout[..output.stdout.len().min(16)]
    );
    assert_eq!(
        &output.stdout[8..12],
        b"WAVE",
        "WAV must contain 'WAVE' at offset 8..12; got: {:02x?}",
        &output.stdout[..output.stdout.len().min(16)]
    );
}

#[test]
fn stdin_after_value_option_uses_clean_implicit_stdout() {
    // Regression for ROP-BUG-FLUX-00053: `44100` used to be mistaken for
    // the positional input by the prelude's raw argv scanner. The typed CLI
    // maps the later `-` input to implicit stdout, so WAV bytes must begin at
    // byte zero with no banner prefix.
    let opus = encode_sine_to_opus_bytes("stdin_after_rate");

    let mut command = Command::new(ropusdec_bin());
    command.args(["--no-color", "--rate", "44100", "-"]);
    let output = run_child_expect_complete(&mut command, Some(&opus));
    assert!(
        output.status.success(),
        "ropusdec exited {:?}; stderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
    assert_eq!(
        output.stdout.get(..4),
        Some(b"RIFF".as_slice()),
        "implicit stdout must start with RIFF; got: {:02x?}",
        &output.stdout[..output.stdout.len().min(16)]
    );
    assert_eq!(
        output.stdout.get(8..12),
        Some(b"WAVE".as_slice()),
        "implicit stdout must contain WAVE at bytes 8..12"
    );
}

#[test]
fn stdout_raw_float_has_no_header() {
    // `--raw --float` means 4-byte-aligned f32 LE on stdout, no WAV
    // container. Assert: no RIFF, no WAVE, byte count divisible by 4
    // (sizeof f32 × 1-channel mono output).
    let opus = encode_sine_to_opus_bytes("raw_float");

    let mut command = Command::new(ropusdec_bin());
    command.args(["--no-color", "--raw", "--float", "-", "-o", "-"]);
    let output = run_child_expect_complete(&mut command, Some(&opus));
    assert!(
        output.status.success(),
        "ropusdec exited {:?}; stderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
    assert!(
        !output.stdout.is_empty(),
        "raw-float stdout unexpectedly empty"
    );
    assert!(
        output.stdout.len().is_multiple_of(4),
        "raw f32 output length {} must be divisible by 4 (channels × sizeof f32)",
        output.stdout.len()
    );
    assert!(
        !output.stdout.windows(4).any(|w| w == b"RIFF"),
        "raw output must contain no 'RIFF' marker"
    );
    assert!(
        !output.stdout.windows(4).any(|w| w == b"WAVE"),
        "raw output must contain no 'WAVE' marker"
    );
}

#[test]
fn quiet_success_suppresses_informational_output() {
    let opus = encode_sine_to_opus_bytes("quiet_success");
    let output_path = temp_path("quiet_success", "wav");

    let mut command = Command::new(ropusdec_bin());
    command.args([
        "--quiet",
        "--no-color",
        "-",
        "-o",
        output_path.to_str().expect("temporary path is UTF-8"),
    ]);
    let result = run_child_expect_complete(&mut command, Some(&opus));
    assert!(
        result.status.success(),
        "quiet decode failed: stderr={:?}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        result.stdout.is_empty(),
        "quiet decode wrote stdout: {:?}",
        result.stdout
    );
    assert!(
        result.stderr.is_empty(),
        "quiet decode wrote stderr: {:?}",
        result.stderr
    );

    let decoded = std::fs::read(&output_path).expect("read quiet decode output");
    assert_eq!(&decoded[..4], b"RIFF", "quiet decode output is not WAV");
    let _ = std::fs::remove_file(output_path);
}

#[test]
fn quiet_failure_preserves_errors_without_progress_reports() {
    let output_path = temp_path("quiet_failure", "wav");
    let mut command = Command::new(ropusdec_bin());
    command.args([
        "--quiet",
        "--no-color",
        "missing-ropusdec-input.opus",
        "-o",
        output_path.to_str().expect("temporary path is UTF-8"),
    ]);
    let result = run_child_expect_complete(&mut command, None);

    assert!(!result.status.success(), "missing input must fail");
    assert!(
        result.stdout.is_empty(),
        "quiet failure wrote stdout: {:?}",
        result.stdout
    );
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("error:"),
        "quiet failure must report an error: {stderr:?}"
    );
    assert!(
        !stderr.contains("input    "),
        "quiet failure leaked input report: {stderr:?}"
    );
    assert!(
        !stderr.contains("output   "),
        "quiet failure leaked output report: {stderr:?}"
    );
    assert!(
        !output_path.exists(),
        "failed decode must not create output"
    );
}
