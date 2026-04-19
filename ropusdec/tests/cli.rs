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

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

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

#[test]
fn stdin_opus_to_stdout_wav() {
    // Build a known Opus stream in-process, then pipe it to `ropusdec - -o -`.
    // Expect stdout to carry a valid WAV: starts with RIFF, contains WAVE at
    // offset 8..12. Stderr absorbs banner/progress text.
    let opus = encode_sine_to_opus_bytes("stdin_wav");

    let mut child = Command::new(ropusdec_bin())
        .arg("--no-color")
        .arg("-")
        .arg("-o")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ropusdec");

    {
        let mut stdin = child.stdin.take().expect("stdin piped");
        stdin.write_all(&opus).expect("write opus into child stdin");
    }

    let output = child.wait_with_output().expect("wait_with_output");
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

    let mut child = Command::new(ropusdec_bin())
        .arg("--no-color")
        .arg("-")
        .arg("-o-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ropusdec");

    {
        let mut stdin = child.stdin.take().expect("stdin piped");
        stdin.write_all(&opus).expect("write opus into child stdin");
    }

    let output = child.wait_with_output().expect("wait_with_output");
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
fn stdout_raw_float_has_no_header() {
    // `--raw --float` means 4-byte-aligned f32 LE on stdout, no WAV
    // container. Assert: no RIFF, no WAVE, byte count divisible by 4
    // (sizeof f32 × 1-channel mono output).
    let opus = encode_sine_to_opus_bytes("raw_float");

    let mut child = Command::new(ropusdec_bin())
        .arg("--no-color")
        .arg("--raw")
        .arg("--float")
        .arg("-")
        .arg("-o")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ropusdec");

    {
        let mut stdin = child.stdin.take().expect("stdin piped");
        stdin.write_all(&opus).expect("write opus into child stdin");
    }

    let output = child.wait_with_output().expect("wait_with_output");
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
