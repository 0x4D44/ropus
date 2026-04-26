//! Integration tests for the `ropusenc` binary's `-` stdin/stdout sentinel.
//!
//! In-process library tests can't exercise the sentinel because `PathBuf("-")`
//! branches inside `commands::encode` — by the time a library caller reaches
//! that branch, stdin/stdout is the test harness' own handles, not a pipe we
//! control. So we shell out to the built `ropusenc` binary via
//! `CARGO_BIN_EXE_ropusenc` and drive it through `std::process::Command`.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

/// Write a canonical 16-bit PCM mono WAV with a 1 kHz sine at 48 kHz to the
/// caller-supplied byte buffer. Duplicates the synth helper in
/// `ropus-tools-core/tests/round_trip.rs` because that one lives in a separate
/// crate and integration-test files don't share code directly without a
/// shared helper module (keeping this test file self-contained beats the
/// ceremony of a `tests/common/` submodule for one synth function).
fn synth_sine_wav_bytes(seconds: u32, freq_hz: f32) -> Vec<u8> {
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
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
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
    out
}

fn ropusenc_bin() -> PathBuf {
    // Cargo sets `CARGO_BIN_EXE_<name>` for each binary target in the same
    // crate; integration tests get the rebuilt binary, so no need to run
    // `cargo build` beforehand.
    PathBuf::from(env!("CARGO_BIN_EXE_ropusenc"))
}

#[test]
fn stdin_to_stdout_round_trip() {
    // Pipe 0.1 s of synthesised 48 kHz mono sine into `ropusenc - -o -`.
    // Assert the process exits zero and stdout starts with the Ogg magic
    // `"OggS"`. Uses `--no-color` to keep banner text ANSI-free just in case
    // a future refactor writes any of it to stdout by accident.
    let wav = synth_sine_wav_bytes(1, 1000.0);

    let mut child = Command::new(ropusenc_bin())
        .arg("--no-color")
        .arg("-")
        .arg("-o")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ropusenc");

    {
        let mut stdin = child.stdin.take().expect("stdin piped");
        stdin.write_all(&wav).expect("write WAV into child stdin");
        // Drop closes stdin → child sees EOF → command loop proceeds to encode.
    }

    let output = child.wait_with_output().expect("wait_with_output");
    assert!(
        output.status.success(),
        "ropusenc exited {:?}; stderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
    assert!(
        output.stdout.len() > 4,
        "stdout length {} — expected at least an Ogg page header",
        output.stdout.len()
    );
    assert_eq!(
        &output.stdout[..4],
        b"OggS",
        "stdout must start with Ogg magic 'OggS', got: {:02x?}",
        &output.stdout[..output.stdout.len().min(16)]
    );
}

#[test]
fn stdin_to_stdout_round_trip_with_o_attached() {
    // Regression for the argv sniffer: clap accepts `-o-` as the short flag
    // with an attached `-` value. The earlier sniffer missed it and routed
    // the banner to stdout, corrupting the Ogg stream. Assert stdout still
    // starts with `OggS` when the flag is spelled `-o-`.
    let wav = synth_sine_wav_bytes(1, 1000.0);

    let mut child = Command::new(ropusenc_bin())
        .arg("--no-color")
        .arg("-")
        .arg("-o-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ropusenc");

    {
        let mut stdin = child.stdin.take().expect("stdin piped");
        stdin.write_all(&wav).expect("write WAV into child stdin");
    }

    let output = child.wait_with_output().expect("wait_with_output");
    assert!(
        output.status.success(),
        "ropusenc exited {:?}; stderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
    assert_eq!(
        &output.stdout[..4],
        b"OggS",
        "stdout must start with 'OggS' when -o- is used; got: {:02x?}",
        &output.stdout[..output.stdout.len().min(16)]
    );
}

#[test]
fn stdin_to_stdout_round_trip_with_equals() {
    // Regression for the argv sniffer: `--output=-` must also route the
    // banner to stderr. Mirrors the `-o-` regression above but for the
    // long-form spelling.
    let wav = synth_sine_wav_bytes(1, 1000.0);

    let mut child = Command::new(ropusenc_bin())
        .arg("--no-color")
        .arg("-")
        .arg("--output=-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ropusenc");

    {
        let mut stdin = child.stdin.take().expect("stdin piped");
        stdin.write_all(&wav).expect("write WAV into child stdin");
    }

    let output = child.wait_with_output().expect("wait_with_output");
    assert!(
        output.status.success(),
        "ropusenc exited {:?}; stderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
    assert_eq!(
        &output.stdout[..4],
        b"OggS",
        "stdout must start with 'OggS' when --output=- is used; got: {:02x?}",
        &output.stdout[..output.stdout.len().min(16)]
    );
}

#[test]
fn stdout_has_no_banner_pollution() {
    // With `-o -` the banner, heading, and all progress/report lines must
    // land on stderr. stdout is the bitstream and must start exactly with
    // the Ogg magic. `--no-color` also guarantees no ANSI escape codes sneak
    // through some other code path.
    let wav = synth_sine_wav_bytes(1, 1000.0);

    let mut child = Command::new(ropusenc_bin())
        .arg("--no-color")
        .arg("-")
        .arg("-o")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ropusenc");

    {
        let mut stdin = child.stdin.take().expect("stdin piped");
        stdin.write_all(&wav).expect("write WAV into child stdin");
    }

    let output = child.wait_with_output().expect("wait_with_output");
    assert!(
        output.status.success(),
        "ropusenc failed: {:?}",
        output.status
    );

    // First four bytes of stdout are the Ogg magic. No leading ANSI, no
    // leading text.
    assert_eq!(
        &output.stdout[..4],
        b"OggS",
        "stdout must start with 'OggS'; got: {:02x?}",
        &output.stdout[..output.stdout.len().min(16)]
    );

    // The banner strings belong on stderr. Use a generic banner-indicator
    // check rather than exact-match because the timestamp/sha vary between
    // builds. Presence of `ropusenc` and `encode` in stderr is enough.
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("ropusenc"),
        "stderr must carry the banner; got: {stderr:?}"
    );
    assert!(
        stderr.contains("encode"),
        "stderr must carry the 'encode' heading; got: {stderr:?}"
    );
    // Stdout must not contain banner heading text. The OpusTags vendor
    // string legitimately contains "ropusenc" inside the encoded payload,
    // so we target banner-specific phrasing ("encode" heading, progress
    // labels) that the command function would print, never the codec's
    // metadata tags.
    let stdout_head = &output.stdout[..output.stdout.len().min(64)];
    assert!(
        !stdout_head.windows(5).any(|w| w == b"input"),
        "stdout must not carry the 'input' progress label"
    );
    assert!(
        !stdout_head.windows(6).any(|w| w == b"output"),
        "stdout must not carry the 'output' progress label"
    );
    assert!(
        !stdout_head.windows(7).any(|w| w == b"decoded"),
        "stdout must not carry the 'decoded' progress label"
    );
}
