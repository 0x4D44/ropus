//! Stage 2 — tests (+ optional coverage).
//!
//! Dispatches a single cargo invocation based on `--skip-coverage`:
//!
//! - **Coverage OFF** (the default):
//!   `cargo llvm-cov --json --output-path <tmp>.json test --workspace --lib
//!   --tests --no-fail-fast -- --test-threads=1 [--skip testvector]`.
//! - **Coverage ON** (`--skip-coverage`):
//!   `cargo test --workspace --lib --tests --no-fail-fast -- --test-threads=1
//!   [--skip testvector]`.
//!
//! The trailing `--skip ::testvector` is injected iff Stage 0 reported
//! `ietf_vectors_present == false`. HLD proposes `--exclude conformance`
//! for this case, but that flag removes the *entire* conformance crate
//! (30 tests) rather than just the 24 IETF vector probes, which would
//! silently reduce the conformance surface.
//!
//! libtest's `--skip` is a substring match, not a path-prefix match, so a
//! bare `--skip testvector` would also drop any future test whose name
//! merely *contains* the word `testvector`. We use `--skip ::testvector`:
//! the leading `::` pins the match to a module-path component boundary,
//! which only the 24 IETF probes (`conformance::ietf_vectors::testvectorNN`)
//! satisfy today. Verified via `cargo test -p conformance -- --list`.
//! Documented here and in the Phase 2 journal.
//!
//! Stdout/stderr are captured separately because cargo sends per-test
//! outcome lines to stdout and build progress / diagnostics to stderr;
//! `cargo_parse::parse` needs both.

use std::process::{Command, Stdio};
use std::time::Instant;

use colored::Colorize;

use crate::cargo_parse::{self, TestsResult};
use crate::issues;
use crate::llvm_cov_parse::{self, CoverageResult};

/// Stage 2 result, combining test parse output with optional coverage
/// metrics and the skip-reason reporting hook.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub tests: TestsResult,
    pub coverage: Option<CoverageResult>,
    pub coverage_skipped: bool,
}

impl Outcome {
    /// Construct a "stage disabled" outcome. Not used by the Phase 2 dispatch
    /// (no flag currently marks stage 2 as skipped-by-policy — `--quick`
    /// keeps it running per the HLD), but retained because the report
    /// envelope tests depend on it, and Phase 4 may wire an upstream-build-
    /// failure skip path through here.
    #[cfg(test)]
    pub fn skipped(reason: impl Into<String>) -> Self {
        let tests = TestsResult {
            skipped: true,
            skip_reason: Some(reason.into()),
            ..TestsResult::default()
        };
        Self {
            tests,
            coverage: None,
            coverage_skipped: true,
        }
    }

    /// A Stage 2 run is "green" when it was skipped outright, or when it
    /// ran without build failure and no test failed. Ignored tests don't
    /// count (matches HLD § PASS / FAIL / WARN semantics).
    pub fn all_passed(&self) -> bool {
        if self.tests.skipped {
            return true;
        }
        !self.tests.build_failed && self.tests.total_failed == 0
    }
}

/// Build the cargo argument vector for Stage 2. Factored out so the
/// command-shape contract can be unit-tested without spawning cargo.
///
/// - `skip_coverage=false` → `llvm-cov --json --output-path <path> test ...`.
/// - `skip_coverage=true`  → `test ...` (plain cargo test, no instrumentation).
/// - `ietf_vectors_present=false` appends `--skip ::testvector`.
fn build_cargo_args(
    skip_coverage: bool,
    ietf_vectors_present: bool,
    cov_path: Option<&std::path::Path>,
) -> Vec<std::ffi::OsString> {
    use std::ffi::OsString;
    let mut args: Vec<OsString> = Vec::with_capacity(16);
    if skip_coverage {
        for s in ["test", "--workspace", "--lib", "--tests", "--no-fail-fast"] {
            args.push(OsString::from(s));
        }
    } else {
        args.push(OsString::from("llvm-cov"));
        args.push(OsString::from("--json"));
        args.push(OsString::from("--output-path"));
        args.push(OsString::from(
            cov_path.expect("tempfile present when !skip_coverage"),
        ));
        for s in ["test", "--workspace", "--lib", "--tests", "--no-fail-fast"] {
            args.push(OsString::from(s));
        }
    }
    args.push(OsString::from("--"));
    args.push(OsString::from("--test-threads=1"));
    if !ietf_vectors_present {
        // `--skip` is libtest's substring filter; prefix with `::` to pin
        // the match to a module-path component boundary so only the IETF
        // `conformance::ietf_vectors::testvectorNN` tests are dropped.
        args.push(OsString::from("--skip"));
        args.push(OsString::from("::testvector"));
    }
    args
}

/// Canonical skip-reason text for the IETF-vectors fallback. Matched by unit
/// tests and surfaced in the JSON envelope.
fn skip_reason_text() -> String {
    "tests/vectors/ietf/testvector01.bit missing — IETF probes \
     (testvector01..24) filtered out; run tools/fetch_ietf_vectors.sh \
     to enable"
        .to_string()
}

/// Run Stage 2. `ietf_vectors_present` comes from Stage 0; when false we
/// inject `--skip testvector` with a reason captured for the HTML report.
pub fn run(skip_coverage: bool, ietf_vectors_present: bool) -> Outcome {
    let start = Instant::now();
    let (label, header) = if skip_coverage {
        (
            "tests",
            "cargo test --workspace --lib --tests --no-fail-fast -- --test-threads=1".to_string(),
        )
    } else {
        (
            "tests+coverage",
            "cargo llvm-cov --json --output-path <tmp>.json test --workspace --lib --tests \
             --no-fail-fast -- --test-threads=1"
                .to_string(),
        )
    };
    eprintln!("{} {}", format!("[{label}]").cyan().bold(), header);

    // Prepare the coverage output path (only used when coverage is enabled).
    // We bind the `NamedTempFile` guard to a local variable for its full
    // lifetime — it lives until the end of `run()` and is cleaned up there,
    // well after `cargo_llvm_cov` has finished writing.
    let cov_tmp: Option<tempfile::NamedTempFile> = if skip_coverage {
        None
    } else {
        match tempfile::Builder::new()
            .prefix("ropus-llvm-cov-")
            .suffix(".json")
            .tempfile()
        {
            Ok(f) => Some(f),
            Err(e) => {
                // If we can't create a tempfile, surface it as a build
                // failure so downstream stages know not to trust the run.
                let tests = TestsResult {
                    build_failed: true,
                    skip_reason: Some(format!("failed to create llvm-cov tempfile: {e}")),
                    duration_ms: start.elapsed().as_millis() as u64,
                    ..TestsResult::default()
                };
                return Outcome {
                    tests,
                    coverage: None,
                    coverage_skipped: false,
                };
            }
        }
    };

    let cov_path = cov_tmp.as_ref().map(|f| f.path().to_path_buf());

    let args = build_cargo_args(skip_coverage, ietf_vectors_present, cov_path.as_deref());
    let skip_reason: Option<String> = if ietf_vectors_present {
        None
    } else {
        Some(skip_reason_text())
    };

    let mut cmd = Command::new("cargo");
    cmd.args(&args);

    let output = cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).output();

    let duration_ms = start.elapsed().as_millis() as u64;

    let (stdout, stderr, spawn_err): (String, String, Option<String>) = match output {
        Ok(o) => (
            String::from_utf8_lossy(&o.stdout).into_owned(),
            String::from_utf8_lossy(&o.stderr).into_owned(),
            None,
        ),
        Err(e) => (String::new(), String::new(), Some(e.to_string())),
    };

    // Bound memory use before parsing. A ~45 min run can emit many MiB of
    // stderr (build chatter plus per-test failure backtraces); `cap_stderr`
    // truncates both buffers at 1 MiB with an inline note. Per-test outcome
    // lines (~80 B × ≤2000 tests ≈ <200 KB) and `test result:` summaries sit
    // well under the cap, so the authoritative totals are preserved.
    let (stdout, _) = issues::cap_stderr(&stdout);
    let (stderr, _) = issues::cap_stderr(&stderr);

    let mut tests = cargo_parse::parse(&stdout, &stderr);
    tests.duration_ms = duration_ms;
    tests.skip_reason = skip_reason;
    if let Some(e) = spawn_err {
        tests.build_failed = true;
        tests.skip_reason = Some(format!("failed to spawn cargo: {e}"));
    }

    // Coverage: resolve to Some(result) only when cargo actually wrote a
    // non-empty, well-formed JSON document. `tempfile::Builder::tempfile()`
    // creates the path on disk, so `read_to_string` on a build failure
    // returns `Ok("")` — which would parse to a zeros-default result.
    // Setting coverage=None with a skip-reason keeps "coverage unavailable"
    // and "coverage 0/0 = 0%" distinguishable in the envelope.
    let coverage = if skip_coverage {
        tests.coverage_skip_reason = Some("--skip-coverage passed".to_string());
        None
    } else {
        match cov_path.as_ref().map(std::fs::read_to_string) {
            Some(Ok(text)) if text.is_empty() => {
                tests.coverage_skip_reason = Some(
                    "llvm-cov produced no output (likely a build failure before instrumentation \
                     ran)"
                        .to_string(),
                );
                None
            }
            Some(Ok(text)) => match serde_json::from_str::<serde_json::Value>(&text) {
                Ok(_) => Some(llvm_cov_parse::parse(&text)),
                Err(e) => {
                    tests.coverage_skip_reason =
                        Some(format!("llvm-cov output was not valid JSON: {e}"));
                    None
                }
            },
            Some(Err(e)) => {
                tests.coverage_skip_reason = Some(format!("failed to read llvm-cov output: {e}"));
                None
            }
            None => {
                tests.coverage_skip_reason = Some("llvm-cov output path not available".to_string());
                None
            }
        }
    };

    // `cov_tmp` stays bound through the read_to_string above; it drops here.
    drop(cov_tmp);

    let banner = if tests.build_failed {
        "BUILD-FAIL".red()
    } else if tests.total_failed > 0 {
        "FAIL".red()
    } else {
        "PASS".green()
    };
    eprintln!(
        "  {} {} passed, {} failed, {} ignored in {:.2}s",
        banner,
        tests.total_passed,
        tests.total_failed,
        tests.total_ignored,
        duration_ms as f64 / 1000.0,
    );

    Outcome {
        tests,
        coverage,
        coverage_skipped: skip_coverage,
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::cargo_parse::TestsResult;
    use std::path::Path;

    fn outcome_with(tests: TestsResult) -> Outcome {
        Outcome {
            tests,
            coverage: None,
            coverage_skipped: true,
        }
    }

    fn args_as_strings(args: &[std::ffi::OsString]) -> Vec<String> {
        args.iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn skip_coverage_builds_plain_test_command() {
        let args = build_cargo_args(true, true, None);
        let s = args_as_strings(&args);
        assert_eq!(
            s,
            vec![
                "test",
                "--workspace",
                "--lib",
                "--tests",
                "--no-fail-fast",
                "--",
                "--test-threads=1",
            ]
        );
    }

    #[test]
    fn coverage_on_builds_llvm_cov_command_with_path() {
        let p = Path::new("/tmp/sample.json");
        let args = build_cargo_args(false, true, Some(p));
        let s = args_as_strings(&args);
        // Leading chunk is the llvm-cov wrapper.
        assert_eq!(s[0], "llvm-cov");
        assert_eq!(s[1], "--json");
        assert_eq!(s[2], "--output-path");
        assert_eq!(s[3], "/tmp/sample.json");
        // Test subcommand starts at index 4.
        assert_eq!(
            &s[4..9],
            &["test", "--workspace", "--lib", "--tests", "--no-fail-fast"]
        );
        // Trailing pass-through to libtest.
        assert_eq!(&s[9..], &["--", "--test-threads=1"]);
    }

    #[test]
    fn missing_ietf_vectors_appends_skip_filter() {
        let args = build_cargo_args(true, false, None);
        let s = args_as_strings(&args);
        // The tail after `--test-threads=1` is `--skip ::testvector`.
        // The leading `::` pins the match to a module-path boundary so we
        // don't accidentally drop an unrelated future test whose name merely
        // contains "testvector".
        let tail: &[String] = &s[s.len() - 2..];
        assert_eq!(tail, &["--skip".to_string(), "::testvector".to_string()]);
    }

    #[test]
    fn present_ietf_vectors_does_not_append_skip_filter() {
        let args = build_cargo_args(true, true, None);
        let s = args_as_strings(&args);
        assert!(!s.iter().any(|a| a == "--skip"));
        assert!(!s.iter().any(|a| a == "::testvector"));
    }

    #[test]
    fn skip_reason_text_references_fetch_script() {
        let r = skip_reason_text();
        assert!(r.contains("testvector01.bit"));
        assert!(r.contains("fetch_ietf_vectors.sh"));
    }

    #[test]
    fn skipped_outcome_is_green() {
        let o = Outcome::skipped("stage disabled");
        assert!(o.all_passed());
        assert!(o.tests.skipped);
        assert_eq!(o.tests.skip_reason.as_deref(), Some("stage disabled"));
    }

    #[test]
    fn build_failure_fails_stage() {
        let t = TestsResult {
            build_failed: true,
            ..TestsResult::default()
        };
        assert!(!outcome_with(t).all_passed());
    }

    #[test]
    fn any_test_failure_fails_stage() {
        let t = TestsResult {
            total_passed: 100,
            total_failed: 1,
            ..TestsResult::default()
        };
        assert!(!outcome_with(t).all_passed());
    }

    #[test]
    fn clean_pass_is_green() {
        let t = TestsResult {
            total_passed: 42,
            ..TestsResult::default()
        };
        assert!(outcome_with(t).all_passed());
    }

    #[test]
    fn ignored_tests_do_not_fail_stage() {
        let t = TestsResult {
            total_passed: 10,
            total_ignored: 5,
            ..TestsResult::default()
        };
        assert!(outcome_with(t).all_passed());
    }
}
