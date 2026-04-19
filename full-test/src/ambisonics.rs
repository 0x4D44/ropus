//! Stage 3 — ambisonics projection roundtrip.
//!
//! Invokes `cargo run --release -p ropus-harness --bin projection_roundtrip`
//! and scrapes stdout for the summary block. The harness binary prints a
//! per-fixture line of the form
//!
//! ```text
//!   PASS order 1: 5 frames, encode 5/5, decode 5/5
//!   FAIL order 2: 5 frames, encode 4/5, decode 5/5
//! ```
//!
//! plus a final
//!
//! ```text
//! PASS: all fixtures byte-exact encode AND sample-exact decode
//! ```
//!
//! on success (or `FAIL` on failure). The parser here degrades to empty data
//! rather than panicking on anything it doesn't recognise — same defensive
//! style as the Stage 2 cargo parser.
//!
//! Cargo exits 101 for build failures; the binary itself exits 1 for runtime
//! mismatches. We use that distinction to set `build_failed` without pattern-
//! matching on error text.

use std::process::{Command, Stdio};
use std::time::Instant;

use colored::Colorize;
use serde::Serialize;

/// A single per-order outcome scraped from the harness summary block.
#[derive(Debug, Clone, Serialize)]
pub struct OrderOutcome {
    pub order: u8,
    pub passed: bool,
    pub detail: String,
}

/// Stage 3 result. `skipped` covers the flag path; `build_failed` covers a
/// cargo compile error (exit 101). An otherwise-clean run with no summary
/// marker leaves `overall_pass=false` with an empty `per_order` vector — the
/// report surfaces that as "ambisonics: no summary produced".
#[derive(Debug, Clone, Serialize)]
pub struct AmbisonicsResult {
    pub skipped: bool,
    pub skip_reason: Option<String>,
    pub build_failed: bool,
    pub duration_ms: u64,
    pub overall_pass: bool,
    pub per_order: Vec<OrderOutcome>,
}

impl AmbisonicsResult {
    /// Construct a "stage disabled" outcome used by the `--skip-ambisonics`
    /// flag and the upstream-build-failure chaining rule from the HLD.
    pub fn skipped(reason: impl Into<String>) -> Self {
        Self {
            skipped: true,
            skip_reason: Some(reason.into()),
            build_failed: false,
            duration_ms: 0,
            overall_pass: false,
            per_order: Vec::new(),
        }
    }

    /// True when the stage was skipped outright, or when every fixture passed
    /// and there was no build failure. Matches the HLD's "green means every
    /// enabled stage passed" contract.
    pub fn all_passed(&self) -> bool {
        if self.skipped {
            return true;
        }
        !self.build_failed && self.overall_pass
    }
}

/// Build the cargo argument vector for Stage 3. Factored out so the command-
/// shape contract can be unit-tested without spawning cargo.
fn build_cargo_args() -> Vec<&'static str> {
    vec![
        "run",
        "--release",
        "-p",
        "ropus-harness",
        "--bin",
        "projection_roundtrip",
    ]
}

/// Run Stage 3. Returns a populated `AmbisonicsResult`; never panics.
pub fn run() -> AmbisonicsResult {
    let start = Instant::now();
    eprintln!(
        "{} cargo run --release -p ropus-harness --bin projection_roundtrip",
        "[ambisonics]".cyan().bold()
    );

    // `CARGO_TERM_COLOR=never` forces plain cargo output regardless of any
    // terminal-detection logic on the developer box, keeping the stdout /
    // stderr parsers simple.
    let output = Command::new("cargo")
        .env("CARGO_TERM_COLOR", "never")
        .args(build_cargo_args())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();

    let duration_ms = start.elapsed().as_millis() as u64;

    let (stdout, stderr, exit_code, spawn_err) = match output {
        Ok(o) => (
            String::from_utf8_lossy(&o.stdout).into_owned(),
            String::from_utf8_lossy(&o.stderr).into_owned(),
            o.status.code(),
            None,
        ),
        Err(e) => (String::new(), String::new(), None, Some(e.to_string())),
    };

    if let Some(msg) = spawn_err {
        eprintln!("  {} failed to spawn cargo: {msg}", "FAIL".red());
        return AmbisonicsResult {
            skipped: false,
            skip_reason: Some(format!("failed to spawn cargo: {msg}")),
            build_failed: true,
            duration_ms,
            overall_pass: false,
            per_order: Vec::new(),
        };
    }

    // Bound memory before scanning. The harness is short-lived but cargo's
    // build chatter on a cold cache can push into MiB territory.
    let (stdout, _) = crate::issues::cap_stderr(&stdout);
    let (stderr, _) = crate::issues::cap_stderr(&stderr);

    // Cargo uses exit code 101 for compile errors; the binary itself uses 1
    // for runtime fixture mismatches. Distinguishing these lets the report
    // skip downstream stages correctly without pattern-matching `error[E`.
    let build_failed = exit_code == Some(101) || detect_build_failure(&stderr);
    let (overall_pass, per_order) = parse_summary(&stdout);

    let banner = if build_failed {
        "BUILD-FAIL".red()
    } else if overall_pass {
        "PASS".green()
    } else {
        "FAIL".red()
    };
    eprintln!(
        "  {} {} orders in {:.2}s",
        banner,
        per_order.len(),
        duration_ms as f64 / 1000.0,
    );

    AmbisonicsResult {
        skipped: false,
        skip_reason: None,
        build_failed,
        duration_ms,
        overall_pass,
        per_order,
    }
}

/// Parse the per-order + overall marker lines in the stdout stream emitted
/// by `projection_roundtrip`.
///
/// Returns `(overall_pass, per_order)`. `overall_pass` is true iff the trailing
/// `PASS: all fixtures byte-exact encode AND sample-exact decode` line is
/// present; anything else (missing marker, empty output, garbage) is treated
/// as fail. The `=== SUMMARY ===` header itself is not matched — we scan the
/// whole stream for the marker and per-order shapes, so a harness that
/// reorders or drops the header still produces useful data.
///
/// The per-order scrape is independent of the overall marker: we still want
/// "order 3 failed" visible in the report when the summary is present but
/// reports mismatches.
pub fn parse_summary(stdout: &str) -> (bool, Vec<OrderOutcome>) {
    let overall_pass = stdout
        .lines()
        .any(|l| l.trim() == "PASS: all fixtures byte-exact encode AND sample-exact decode");

    let mut per_order: Vec<OrderOutcome> = Vec::new();
    for raw in stdout.lines() {
        if let Some(outcome) = parse_order_line(raw) {
            per_order.push(outcome);
        }
    }
    (overall_pass, per_order)
}

/// Parse a single per-order summary line: `  PASS order N: <detail>` or
/// `  FAIL order N: <detail>`. Returns `None` on any malformed shape.
fn parse_order_line(line: &str) -> Option<OrderOutcome> {
    let trimmed = line.trim_start();
    let (passed, rest) = if let Some(rest) = trimmed.strip_prefix("PASS order ") {
        (true, rest)
    } else if let Some(rest) = trimmed.strip_prefix("FAIL order ") {
        (false, rest)
    } else {
        return None;
    };
    // `rest` looks like "1: 5 frames, encode 5/5, decode 5/5". Split on the
    // first colon to peel off the numeric order.
    let (order_str, detail) = rest.split_once(':')?;
    let order: u8 = order_str.trim().parse().ok()?;
    Some(OrderOutcome {
        order,
        passed,
        detail: detail.trim().to_string(),
    })
}

/// Fallback build-failure detector for the unusual case where cargo returns
/// a non-101 exit code but the build still didn't finish. Mirrors the
/// `cargo_parse.rs` heuristic.
fn detect_build_failure(stderr: &str) -> bool {
    stderr.lines().any(|line| {
        let t = line.trim_start();
        t.starts_with("error[E") || t.starts_with("error: could not compile")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Copied from `harness/src/bin/projection_roundtrip.rs` — the actual
    /// format emitted on a clean 5-order pass.
    const HAPPY_STDOUT: &str = "\
--- order 1 ---
Loaded tests/vectors/ambisonic/ambisonic_order1_100ms.wav: 48000 Hz, 4 channels, 4800 samples
  Processed 5 frames (960 samples/frame x 4 channels)
  Encode: 5/5 frames byte-identical
  Decode: 5/5 frames sample-identical
--- order 2 ---
Loaded tests/vectors/ambisonic/ambisonic_order2_100ms.wav: 48000 Hz, 9 channels, 4800 samples
  Processed 5 frames (960 samples/frame x 9 channels)
  Encode: 5/5 frames byte-identical
  Decode: 5/5 frames sample-identical
--- order 3 ---
Loaded tests/vectors/ambisonic/ambisonic_order3_100ms.wav: 48000 Hz, 16 channels, 4800 samples
  Processed 5 frames (960 samples/frame x 16 channels)
  Encode: 5/5 frames byte-identical
  Decode: 5/5 frames sample-identical
--- order 4 ---
Loaded tests/vectors/ambisonic/ambisonic_order4_100ms.wav: 48000 Hz, 25 channels, 4800 samples
  Processed 5 frames (960 samples/frame x 25 channels)
  Encode: 5/5 frames byte-identical
  Decode: 5/5 frames sample-identical
--- order 5 ---
Loaded tests/vectors/ambisonic/ambisonic_order5_100ms.wav: 48000 Hz, 36 channels, 4800 samples
  Processed 5 frames (960 samples/frame x 36 channels)
  Encode: 5/5 frames byte-identical
  Decode: 5/5 frames sample-identical

=== SUMMARY ===
  PASS order 1: 5 frames, encode 5/5, decode 5/5
  PASS order 2: 5 frames, encode 5/5, decode 5/5
  PASS order 3: 5 frames, encode 5/5, decode 5/5
  PASS order 4: 5 frames, encode 5/5, decode 5/5
  PASS order 5: 5 frames, encode 5/5, decode 5/5
PASS: all fixtures byte-exact encode AND sample-exact decode
";

    /// A run where order 2 drifted on decode but the rest held.
    const FAILING_STDOUT: &str = "\
=== SUMMARY ===
  PASS order 1: 5 frames, encode 5/5, decode 5/5
  FAIL order 2: 5 frames, encode 5/5, decode 4/5
    first decode divergence: frame 3, sample 12
  PASS order 3: 5 frames, encode 5/5, decode 5/5
  PASS order 4: 5 frames, encode 5/5, decode 5/5
  PASS order 5: 5 frames, encode 5/5, decode 5/5
FAIL
";

    #[test]
    fn build_cargo_args_shape_is_stable() {
        // The HLD spells out the exact invocation; this test guards against
        // drift if someone edits the flag list casually.
        let args = build_cargo_args();
        assert_eq!(
            args,
            vec![
                "run",
                "--release",
                "-p",
                "ropus-harness",
                "--bin",
                "projection_roundtrip",
            ]
        );
    }

    #[test]
    fn happy_summary_parses_all_five_orders_as_pass() {
        let (overall, per_order) = parse_summary(HAPPY_STDOUT);
        assert!(overall);
        assert_eq!(per_order.len(), 5);
        for (i, o) in per_order.iter().enumerate() {
            assert_eq!(o.order, (i + 1) as u8);
            assert!(o.passed, "order {} should pass", o.order);
            assert!(
                o.detail.contains("encode 5/5"),
                "detail missing encode count: {}",
                o.detail
            );
            assert!(o.detail.contains("decode 5/5"));
        }
    }

    #[test]
    fn mismatch_summary_flags_failed_order() {
        let (overall, per_order) = parse_summary(FAILING_STDOUT);
        assert!(!overall);
        assert_eq!(per_order.len(), 5);
        // Exactly one failing order.
        let failing: Vec<&OrderOutcome> = per_order.iter().filter(|o| !o.passed).collect();
        assert_eq!(failing.len(), 1);
        assert_eq!(failing[0].order, 2);
        assert!(failing[0].detail.contains("decode 4/5"));
    }

    #[test]
    fn empty_output_is_not_a_pass() {
        let (overall, per_order) = parse_summary("");
        assert!(!overall);
        assert!(per_order.is_empty());
    }

    #[test]
    fn only_missing_marker_defaults_to_fail() {
        // Summary table without the final "PASS: all fixtures ..." marker.
        let s = "\
=== SUMMARY ===
  PASS order 1: 5 frames, encode 5/5, decode 5/5
";
        let (overall, per_order) = parse_summary(s);
        assert!(!overall);
        assert_eq!(per_order.len(), 1);
        assert!(per_order[0].passed);
    }

    #[test]
    fn garbled_lines_do_not_panic() {
        let s = "\
  PASS order abc: nope
  FAIL order : also nope
  not-an-order-line
  PASS order 7: ok detail here
";
        let (_, per_order) = parse_summary(s);
        // Only the well-formed trailing line survives.
        assert_eq!(per_order.len(), 1);
        assert_eq!(per_order[0].order, 7);
        assert!(per_order[0].passed);
    }

    #[test]
    fn detect_build_failure_fires_on_compile_errors() {
        let stderr = "\
   Compiling ropus-harness v0.1.0
error[E0432]: unresolved import `foo::bar`
  --> harness/src/bin/projection_roundtrip.rs:10:5
error: could not compile `ropus-harness` due to previous error
";
        assert!(detect_build_failure(stderr));
    }

    #[test]
    fn detect_build_failure_quiet_on_clean_run() {
        let stderr = "\
   Compiling ropus-harness v0.1.0
    Finished release [optimized] target(s) in 12.34s
     Running `target/release/projection_roundtrip`
";
        assert!(!detect_build_failure(stderr));
    }

    #[test]
    fn skipped_is_green() {
        let o = AmbisonicsResult::skipped("--skip-ambisonics");
        assert!(o.skipped);
        assert!(o.all_passed());
        assert_eq!(o.skip_reason.as_deref(), Some("--skip-ambisonics"));
    }

    #[test]
    fn build_failure_fails_stage() {
        let o = AmbisonicsResult {
            skipped: false,
            skip_reason: None,
            build_failed: true,
            duration_ms: 12,
            overall_pass: false,
            per_order: Vec::new(),
        };
        assert!(!o.all_passed());
    }

    #[test]
    fn clean_pass_is_green() {
        let o = AmbisonicsResult {
            skipped: false,
            skip_reason: None,
            build_failed: false,
            duration_ms: 42,
            overall_pass: true,
            per_order: vec![OrderOutcome {
                order: 1,
                passed: true,
                detail: "ok".to_string(),
            }],
        };
        assert!(o.all_passed());
    }

    #[test]
    fn mismatched_order_fails_stage() {
        // overall_pass=false with a per-order mismatch → stage red.
        let o = AmbisonicsResult {
            skipped: false,
            skip_reason: None,
            build_failed: false,
            duration_ms: 42,
            overall_pass: false,
            per_order: vec![OrderOutcome {
                order: 2,
                passed: false,
                detail: "encode 4/5".to_string(),
            }],
        };
        assert!(!o.all_passed());
    }
}
