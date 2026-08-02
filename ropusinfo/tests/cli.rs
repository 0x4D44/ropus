//! CLI-level integration tests for ropusinfo.
//!
//! These tests shell out to the built `ropusinfo` binary via
//! `CARGO_BIN_EXE_ropusinfo` (exposed automatically for integration tests in
//! the same crate that owns the binary). In-process library tests live in
//! `ropus-tools-core/tests/round_trip.rs`; anything that needs exit-code or
//! stdout-capture semantics lives here instead.

use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use ropus_tools_core::audio::wav::write_wav_pcm16;
use ropus_tools_core::commands;
use ropus_tools_core::options::EncodeOptions;

/// Produce a short `.opus` file in the system temp directory and return its
/// path. Callers own cleanup. `comments` populates the OpusTags comment list
/// verbatim — pass `["ARTIST=Foo", "TITLE=Bar"]` to test tag queries.
fn encode_tmp_opus(tag: &str, comments: Vec<String>) -> PathBuf {
    let nonce = format!(
        "{}_{}_{}",
        tag,
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let input_wav = std::env::temp_dir().join(format!("ropusinfo_cli_{nonce}.wav"));
    let tmp_opus = std::env::temp_dir().join(format!("ropusinfo_cli_{nonce}.opus"));
    let samples: Vec<i16> = (0..48_000)
        .map(|n| {
            let phase = std::f32::consts::TAU * 1_000.0 * n as f32 / 48_000.0;
            (phase.sin() * 19_000.0) as i16
        })
        .collect();
    write_wav_pcm16(&input_wav, &samples, 48_000, 1).expect("write deterministic WAV fixture");

    let enc_opts = EncodeOptions {
        input: input_wav.clone(),
        output: Some(tmp_opus.clone()),
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
        vendor: "ropusinfo-cli-test".to_string(),
        comments,
    };
    commands::encode(enc_opts).expect("encode fixture for CLI test");
    let _ = std::fs::remove_file(input_wav);
    tmp_opus
}

/// Run `ropusinfo` with the given args and return (stdout, stderr, exit_code).
/// Panics on spawn failure — we want a clear test failure, not a hidden one.
fn run_ropusinfo(args: &[&str]) -> (String, String, i32) {
    let bin = env!("CARGO_BIN_EXE_ropusinfo");
    // Disable ANSI color, but leave the banner enabled for default-mode tests.
    // Query mode suppresses it based on its typed option, so this helper covers
    // both default and controlled output paths.
    let mut cmd = Command::new(bin);
    cmd.arg("--no-color");
    for a in args {
        cmd.arg(a);
    }
    let out = cmd.output().expect("spawn ropusinfo");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let code = out.status.code().unwrap_or(-1);
    (stdout, stderr, code)
}

fn ogg_crc32(page: &[u8]) -> u32 {
    const POLY: u32 = 0x04C1_1DB7;
    let mut crc = 0u32;
    for (index, &byte) in page.iter().enumerate() {
        let byte = if (22..26).contains(&index) { 0 } else { byte };
        crc ^= u32::from(byte) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 {
                (crc << 1) ^ POLY
            } else {
                crc << 1
            };
        }
    }
    crc
}

fn build_ogg_page(
    serial: u32,
    sequence: u32,
    header_type: u8,
    granule: u64,
    payload: &[u8],
) -> Vec<u8> {
    assert!(payload.len() <= u8::MAX as usize);
    let mut page = Vec::with_capacity(28 + payload.len());
    page.extend_from_slice(b"OggS");
    page.push(0); // stream structure version
    page.push(header_type);
    page.extend_from_slice(&granule.to_le_bytes());
    page.extend_from_slice(&serial.to_le_bytes());
    page.extend_from_slice(&sequence.to_le_bytes());
    page.extend_from_slice(&0u32.to_le_bytes()); // CRC placeholder
    page.push(1); // one lacing segment
    page.push(payload.len() as u8);
    page.extend_from_slice(payload);
    let crc = ogg_crc32(&page);
    page[22..26].copy_from_slice(&crc.to_le_bytes());
    page
}

fn write_incomplete_invalid_opus(tag: &str) -> PathBuf {
    let nonce = format!(
        "{}_{}_{}",
        tag,
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let path = std::env::temp_dir().join(format!("ropusinfo_cli_{nonce}.opus"));
    let serial = 0x1234_5678;
    let unknown_granule = u64::MAX;

    let mut head = b"OpusHead".to_vec();
    head.extend_from_slice(&[1, 1]); // version, mono channel count
    head.extend_from_slice(&0u16.to_le_bytes()); // pre-skip
    head.extend_from_slice(&48_000u32.to_le_bytes());
    head.extend_from_slice(&0i16.to_le_bytes()); // output gain
    head.push(0); // family 0 mapping

    let mut tags = b"OpusTags".to_vec();
    tags.extend_from_slice(&0u32.to_le_bytes()); // empty vendor
    tags.extend_from_slice(&0u32.to_le_bytes()); // zero comments

    // Non-empty but invalid Opus payload: the container validator accepts it,
    // while the codec rejects it during the strict fallback decode.
    let invalid_audio = [0x9B, 7];
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&build_ogg_page(serial, 0, 0x02, 0, &head));
    bytes.extend_from_slice(&build_ogg_page(serial, 1, 0, 0, &tags));
    bytes.extend_from_slice(&build_ogg_page(
        serial,
        2,
        0x04,
        unknown_granule,
        &invalid_audio,
    ));
    std::fs::write(&path, bytes).expect("write incomplete invalid Opus fixture");
    path
}

#[test]
fn info_default_output_contains_expected_fields() {
    let opus = encode_tmp_opus("default", Vec::new());

    let (stdout, _stderr, code) = run_ropusinfo(&[opus.to_str().expect("path utf8")]);
    assert_eq!(code, 0, "exit code 0 expected, got {code}");

    // Hit the load-bearing field names. Bright-white ANSI wrapping is stripped
    // by `--no-color`, so literal substring matches are reliable.
    for expected in &[
        "Input File:",
        "Channels:",
        "Sample rate (input):",
        "Pre-skip:",
        "Output gain:",
        "Channel mapping family:",
        "Vendor:",
        "Total data length:",
        "Playback length:",
        "Average bitrate:",
    ] {
        assert!(
            stdout.contains(expected),
            "default output missing `{expected}`; got:\n{stdout}"
        );
    }

    let _ = std::fs::remove_file(&opus);
}

#[test]
fn info_extended_lists_per_packet_toc() {
    let opus = encode_tmp_opus("extended", Vec::new());

    let (stdout, _stderr, code) = run_ropusinfo(&["--extended", opus.to_str().expect("path utf8")]);
    assert_eq!(code, 0, "exit code 0 expected, got {code}");

    assert!(
        stdout.contains("Packets:"),
        "extended output missing `Packets:` section; got:\n{stdout}"
    );

    // Tighten the per-packet assertion to specifically check packet 0. A
    // swapped NB/WB row in the TOC config table would still produce
    // "CELT-" / "Hybrid-" / "SILK-" text somewhere in the output, so just
    // checking that any mode label appears doesn't guard against table
    // regressions. Pick out the `#0000:` line and verify two invariants:
    //
    //   1. `mode=<one-of-the-nine-known-labels>` appears
    //   2. `dur=20ms` — we pass `--framesize 20` above, so every packet
    //      on this fixture is a single 20 ms frame. (code=0 → 1 frame,
    //      frame duration 20 ms → dur=20ms.)
    //
    // We deliberately don't pin the specific mode/bw combo: the ropus
    // encoder may pick any of SILK/Hybrid/CELT for a 1 kHz tone at 64 kbps
    // and that's within spec. What matters is that the label shape is
    // correct and the frame duration matches our encoder setting.
    let line0 = stdout
        .lines()
        .find(|l| l.contains("#0000:"))
        .unwrap_or_else(|| panic!("extended output missing `#0000:` line; got:\n{stdout}"));

    let known_modes = [
        "SILK-NB",
        "SILK-MB",
        "SILK-WB",
        "Hybrid-SWB",
        "Hybrid-FB",
        "CELT-NB",
        "CELT-WB",
        "CELT-SWB",
        "CELT-FB",
    ];
    let has_known_mode = known_modes
        .iter()
        .any(|m| line0.contains(&format!("mode={m}")));
    assert!(
        has_known_mode,
        "packet 0 TOC line missing `mode=<known-label>`; got: `{line0}`"
    );
    assert!(
        line0.contains("dur=20ms"),
        "packet 0 TOC line missing `dur=20ms` (fixture uses --framesize 20); got: `{line0}`"
    );

    let _ = std::fs::remove_file(&opus);
}

#[test]
fn info_query_duration_returns_bare_number() {
    let opus = encode_tmp_opus("query_dur", Vec::new());

    let (stdout, _stderr, code) =
        run_ropusinfo(&["--query", "duration", opus.to_str().expect("path utf8")]);
    assert_eq!(code, 0, "exit code 0 expected, got {code}");

    // Only one line, and it parses as a float. No banner, no key prefix, no
    // unit suffix. The `.trim()` is defensive: on Windows the println line
    // ending is `\r\n`, which `parse::<f64>` doesn't like.
    let lines: Vec<_> = stdout.lines().collect();
    assert_eq!(lines.len(), 1, "expected one stdout line, got {lines:?}");
    let v: f64 = lines[0]
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("duration was not a bare float: `{}`", lines[0]));
    assert!(v > 0.0, "duration should be positive, got {v}");

    let _ = std::fs::remove_file(&opus);
}

#[test]
fn info_query_attached_short_form_returns_bare_number_without_quiet() {
    // Regression for ROP-BUG-FLUX-00053: Clap accepts `-q=duration`, but the
    // old prelude scanner did not recognise it and printed a banner before
    // the query scalar. Do not add `--quiet` here: query mode itself must be
    // the authoritative reason the banner is suppressed.
    let opus = encode_tmp_opus("query_attached", Vec::new());

    let out = Command::new(env!("CARGO_BIN_EXE_ropusinfo"))
        .args([
            "--no-color",
            "-q=duration",
            opus.to_str().expect("path utf8"),
        ])
        .output()
        .expect("run ropusinfo");

    assert_eq!(
        out.status.code(),
        Some(0),
        "query failed; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("query stdout is UTF-8");
    let lines: Vec<_> = stdout.lines().collect();
    assert_eq!(
        lines.len(),
        1,
        "expected one bare query line, got {lines:?}"
    );
    let duration: f64 = lines[0]
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("duration was not a bare float: {:?}", lines[0]));
    assert!(
        duration > 0.0,
        "duration should be positive, got {duration}"
    );

    let _ = std::fs::remove_file(&opus);
}

#[test]
fn info_query_comment_artist_returns_value() {
    let opus = encode_tmp_opus(
        "query_artist",
        vec!["ARTIST=Foo".to_string(), "TITLE=Bar".to_string()],
    );

    let (stdout, _stderr, code) = run_ropusinfo(&[
        "--query",
        "comment:artist",
        opus.to_str().expect("path utf8"),
    ]);
    assert_eq!(code, 0);
    assert_eq!(stdout.trim_end(), "Foo");

    // Case insensitivity: `COMMENT:ARTIST` should also resolve `ARTIST=Foo`.
    let (stdout_upper, _, code_upper) = run_ropusinfo(&[
        "--query",
        "COMMENT:ARTIST",
        opus.to_str().expect("path utf8"),
    ]);
    assert_eq!(code_upper, 0);
    assert_eq!(stdout_upper.trim_end(), "Foo");

    let _ = std::fs::remove_file(&opus);
}

#[test]
fn info_query_missing_comment_is_empty_exit_0() {
    let opus = encode_tmp_opus("query_missing", Vec::new());

    let (stdout, _stderr, code) = run_ropusinfo(&[
        "--query",
        "comment:nonexistent",
        opus.to_str().expect("path utf8"),
    ]);
    assert_eq!(code, 0, "missing comment must be exit 0 for scriptability");
    // Output is a single empty line (one println!()); callers can `grep -q .`
    // to detect absence.
    assert!(
        stdout.trim().is_empty(),
        "missing comment must produce empty stdout, got `{stdout}`"
    );

    let _ = std::fs::remove_file(&opus);
}

#[test]
fn info_query_unknown_key_exits_2() {
    let opus = encode_tmp_opus("query_unknown", Vec::new());

    let (_stdout, stderr, code) =
        run_ropusinfo(&["--query", "gargle", opus.to_str().expect("path utf8")]);
    assert_eq!(code, 2, "unknown key must exit 2, got {code}");
    assert!(
        stderr.contains("unknown query key"),
        "stderr should explain the error; got `{stderr}`"
    );

    let _ = std::fs::remove_file(&opus);
}

#[test]
fn strict_duration_and_bitrate_reject_incomplete_decode_without_scalar() {
    let opus = write_incomplete_invalid_opus("strict_incomplete");

    for query in ["duration", "bitrate"] {
        let out = Command::new(env!("CARGO_BIN_EXE_ropusinfo"))
            .args([
                "--no-color",
                "--query",
                query,
                opus.to_str().expect("path utf8"),
            ])
            .output()
            .expect("run strict query");

        assert!(
            !out.status.success(),
            "{query} must fail for an invalid fallback packet; stderr={:?}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            out.stdout.is_empty(),
            "{query} must not emit a partial scalar: {:?}",
            String::from_utf8_lossy(&out.stdout)
        );
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("decoding Opus audio packet"),
            "{query} error must identify strict packet decoding: {:?}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    let human = Command::new(env!("CARGO_BIN_EXE_ropusinfo"))
        .args(["--no-color", opus.to_str().expect("path utf8")])
        .output()
        .expect("run human info diagnostic");
    assert!(
        human.status.success(),
        "human info should retain a diagnostic estimate: {:?}",
        String::from_utf8_lossy(&human.stderr)
    );
    assert!(
        String::from_utf8_lossy(&human.stdout).contains("estimate; 1 packet error(s) skipped"),
        "human info must label its estimate: {:?}",
        String::from_utf8_lossy(&human.stdout)
    );

    let _ = std::fs::remove_file(opus);
}
