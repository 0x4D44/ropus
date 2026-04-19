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

use ropus_tools_core::commands;
use ropus_tools_core::options::EncodeOptions;

/// Produce a short `.opus` file in the system temp directory and return its
/// path. Callers own cleanup. `comments` populates the OpusTags comment list
/// verbatim — pass `["ARTIST=Foo", "TITLE=Bar"]` to test tag queries.
fn encode_tmp_opus(tag: &str, comments: Vec<String>) -> PathBuf {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ropusinfo has a parent dir (workspace root)")
        .to_path_buf();
    let input_wav = workspace.join("tests/vectors/48k_sine1k_loud.wav");

    let nonce = format!(
        "{}_{}_{}",
        tag,
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let tmp_opus = std::env::temp_dir().join(format!("ropusinfo_cli_{nonce}.opus"));

    // Skip the round-trip encode if the test vector isn't present — leaves the
    // calling test free to return early with a SKIP marker instead of failing
    // on the missing fixture, matching round_trip.rs' convention.
    if !input_wav.exists() {
        return tmp_opus;
    }

    let enc_opts = EncodeOptions {
        input: input_wav,
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
    tmp_opus
}

/// Run `ropusinfo` with the given args and return (stdout, stderr, exit_code).
/// Panics on spawn failure — we want a clear test failure, not a hidden one.
fn run_ropusinfo(args: &[&str]) -> (String, String, i32) {
    let bin = env!("CARGO_BIN_EXE_ropusinfo");
    // `--no-color` and `--quiet` keep the output stable across terminals and
    // strip the banner so assertions can focus on the block itself. Query mode
    // implicitly disables both already; passing them unconditionally makes the
    // helper safe to reuse for the default-block tests too.
    let mut cmd = Command::new(bin);
    cmd.arg("--no-color");
    cmd.arg("--quiet");
    for a in args {
        cmd.arg(a);
    }
    let out = cmd.output().expect("spawn ropusinfo");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let code = out.status.code().unwrap_or(-1);
    (stdout, stderr, code)
}

/// True when the test fixture is absent. Returns a clear SKIP message so a
/// silent skip is visible in CI logs rather than looking like a pass.
fn skip_if_no_fixture(path: &std::path::Path, test_name: &str) -> bool {
    if !path.exists() {
        eprintln!(
            "SKIPPING {test_name}: opus fixture {path:?} was not built \
             (the underlying WAV vector was probably missing)"
        );
        return true;
    }
    false
}

#[test]
fn info_default_output_contains_expected_fields() {
    let opus = encode_tmp_opus("default", Vec::new());
    if skip_if_no_fixture(&opus, "info_default_output_contains_expected_fields") {
        return;
    }

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
    if skip_if_no_fixture(&opus, "info_extended_lists_per_packet_toc") {
        return;
    }

    let (stdout, _stderr, code) =
        run_ropusinfo(&["--extended", opus.to_str().expect("path utf8")]);
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
        "SILK-NB", "SILK-MB", "SILK-WB",
        "Hybrid-SWB", "Hybrid-FB",
        "CELT-NB", "CELT-WB", "CELT-SWB", "CELT-FB",
    ];
    let has_known_mode = known_modes.iter().any(|m| line0.contains(&format!("mode={m}")));
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
    if skip_if_no_fixture(&opus, "info_query_duration_returns_bare_number") {
        return;
    }

    let (stdout, _stderr, code) = run_ropusinfo(&[
        "--query",
        "duration",
        opus.to_str().expect("path utf8"),
    ]);
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
fn info_query_comment_artist_returns_value() {
    let opus = encode_tmp_opus(
        "query_artist",
        vec!["ARTIST=Foo".to_string(), "TITLE=Bar".to_string()],
    );
    if skip_if_no_fixture(&opus, "info_query_comment_artist_returns_value") {
        return;
    }

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
    if skip_if_no_fixture(&opus, "info_query_missing_comment_is_empty_exit_0") {
        return;
    }

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
    if skip_if_no_fixture(&opus, "info_query_unknown_key_exits_2") {
        return;
    }

    let (_stdout, stderr, code) = run_ropusinfo(&[
        "--query",
        "gargle",
        opus.to_str().expect("path utf8"),
    ]);
    assert_eq!(code, 2, "unknown key must exit 2, got {code}");
    assert!(
        stderr.contains("unknown query key"),
        "stderr should explain the error; got `{stderr}`"
    );

    let _ = std::fs::remove_file(&opus);
}
