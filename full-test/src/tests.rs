//! Stage 2 — tests (+ optional coverage).
//!
//! Dispatches a single cargo invocation based on `--skip-coverage`:
//!
//! - **Coverage OFF** (the default):
//!   `cargo llvm-cov --json --ignore-run-fail --output-path <tmp>.json test
//!   --workspace [profile package selection] --lib --tests -- --test-threads=1
//!   [--skip testvector]`.
//!   (`--ignore-run-fail` keeps the report written even when a test fails;
//!   it implies `--no-fail-fast` and the two are mutually exclusive.)
//! - **Coverage ON** (`--skip-coverage`):
//!   `cargo test --workspace [profile package selection] --lib --tests
//!   --no-fail-fast -- --test-threads=1 [--skip testvector]`.
//!
//! The trailing `--skip testvector` is injected only when Stage 0 could not
//! provision the IETF vectors. We still run the rest of Stage 2 for signal, but
//! add a synthetic conformance failure so the gate cannot pass green without
//! the spec vectors.
//!
//! libtest's `--skip` is a substring match. The IETF vector integration tests
//! are named `testvectorNN_mono` / `testvectorNN_stereo`, so the filter must
//! be the flat `testvector` token.
//!
//! Stdout/stderr are captured separately because cargo sends per-test
//! outcome lines to stdout and build progress / diagnostics to stderr;
//! `cargo_parse::parse` needs both.

use std::process::{Command, Stdio};
use std::time::Instant;

use colored::Colorize;

use crate::cargo_parse::{
    self, BinaryResult, Outcome as TestOutcomeKind, TestOutcome, TestsResult,
};
use crate::ietf_vectors::IetfVectorProvision;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage2Profile {
    FullWorkspace,
    QuickReleasePreflight,
}

impl Stage2Profile {
    fn package_args(self) -> &'static [&'static str] {
        match self {
            Self::FullWorkspace => &[],
            Self::QuickReleasePreflight => &[
                "--exclude",
                "conformance",
                "--exclude",
                "ropus-harness-control",
                "--exclude",
                "ropus-harness-deep-plc",
            ],
        }
    }
}

/// Build the cargo argument vector for Stage 2. Factored out so the
/// command-shape contract can be unit-tested without spawning cargo.
///
/// - `skip_coverage=false` → `llvm-cov --json --output-path <path> test ...`.
/// - `skip_coverage=true`  → `test ...` (plain cargo test, no instrumentation).
/// - `skip_ietf_vectors=true` appends `--skip testvector`.
fn build_cargo_args(
    skip_coverage: bool,
    skip_ietf_vectors: bool,
    profile: Stage2Profile,
    cov_path: Option<&std::path::Path>,
) -> Vec<std::ffi::OsString> {
    use std::ffi::OsString;
    let mut args: Vec<OsString> = Vec::with_capacity(24);
    if skip_coverage {
        for s in ["test", "--workspace"] {
            args.push(OsString::from(s));
        }
        args.extend(profile.package_args().iter().map(OsString::from));
        for s in ["--lib", "--tests", "--no-fail-fast"] {
            args.push(OsString::from(s));
        }
    } else {
        // `cargo llvm-cov` refuses to emit its JSON report when a test binary
        // exits non-zero (exit 101 from libtest). `--ignore-run-fail` keeps
        // report generation on after a test failure; it internally forces
        // `--no-fail-fast`, and the two flags are mutually exclusive on the
        // CLI, so we drop the explicit `--no-fail-fast` on this path.
        // Verified against cargo-llvm-cov 0.8.5 in Phase 5 smoke run.
        args.push(OsString::from("llvm-cov"));
        args.push(OsString::from("--json"));
        args.push(OsString::from("--ignore-run-fail"));
        args.push(OsString::from("--output-path"));
        args.push(OsString::from(
            cov_path.expect("tempfile present when !skip_coverage"),
        ));
        for s in ["test", "--workspace"] {
            args.push(OsString::from(s));
        }
        args.extend(profile.package_args().iter().map(OsString::from));
        for s in ["--lib", "--tests"] {
            args.push(OsString::from(s));
        }
    }
    args.push(OsString::from("--"));
    args.push(OsString::from("--test-threads=1"));
    if skip_ietf_vectors {
        args.push(OsString::from("--skip"));
        args.push(OsString::from("testvector"));
    }
    args
}

fn stage_header(skip_coverage: bool, skip_ietf_vectors: bool, profile: Stage2Profile) -> String {
    let mut header = if skip_coverage {
        "cargo test --workspace".to_string()
    } else {
        "cargo llvm-cov --json --ignore-run-fail --output-path <tmp>.json test --workspace"
            .to_string()
    };
    for arg in profile.package_args() {
        header.push(' ');
        header.push_str(arg);
    }
    if skip_coverage {
        header.push_str(" --lib --tests --no-fail-fast -- --test-threads=1");
    } else {
        header.push_str(" --lib --tests -- --test-threads=1");
    }
    if skip_ietf_vectors {
        header.push_str(" --skip testvector");
    }
    header
}

/// Canonical failure text for an unavailable IETF vector set. Matched by unit
/// tests and surfaced in the JSON envelope.
fn unavailable_reason_text(ietf_vectors: &IetfVectorProvision) -> String {
    let reason = ietf_vectors
        .reason
        .as_deref()
        .unwrap_or("fetch script did not produce a complete vector set");
    format!(
        "IETF vectors unavailable after provisioning; filtered \
         testvectorNN_mono/stereo probes to preserve non-IETF \
         Stage 2 signal. {reason}"
    )
}

/// Run Stage 2. If Stage 0 could not provision IETF vectors, filter the 24
/// vector probes but add a synthetic failure so the gate stays honest.
pub fn run(
    skip_coverage: bool,
    ietf_vectors: &IetfVectorProvision,
    profile: Stage2Profile,
) -> Outcome {
    let start = Instant::now();
    let skip_ietf_vectors = !ietf_vectors.available();
    let label = if skip_coverage {
        "tests"
    } else {
        "tests+coverage"
    };
    let header = stage_header(skip_coverage, skip_ietf_vectors, profile);
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

    let args = build_cargo_args(
        skip_coverage,
        skip_ietf_vectors,
        profile,
        cov_path.as_deref(),
    );
    let skip_reason: Option<String> = if skip_ietf_vectors {
        Some(unavailable_reason_text(ietf_vectors))
    } else {
        None
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
    if skip_ietf_vectors {
        append_ietf_provisioning_failure(&mut tests, ietf_vectors);
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

fn append_ietf_provisioning_failure(tests: &mut TestsResult, ietf_vectors: &IetfVectorProvision) {
    const FQN: &str = "conformance::ietf_vectors::provisioning";
    const BINARY: &str = "ietf_vectors";

    tests.total_failed += 1;
    tests.failed_test_names.push(FQN.to_string());
    tests.per_test.push(TestOutcome {
        fqn: FQN.to_string(),
        outcome: TestOutcomeKind::Fail,
        binary: BINARY.to_string(),
    });
    match tests.binaries.iter_mut().find(|b| b.name == BINARY) {
        Some(binary) => binary.failed += 1,
        None => tests.binaries.push(BinaryResult {
            name: BINARY.to_string(),
            passed: 0,
            failed: 1,
            ignored: 0,
            duration_ms: 0,
        }),
    }
    if tests.skip_reason.is_none() {
        tests.skip_reason = Some(unavailable_reason_text(ietf_vectors));
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::cargo_parse::TestsResult;
    use crate::ietf_vectors::ProvisionStatus;
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
        let args = build_cargo_args(true, false, Stage2Profile::FullWorkspace, None);
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
        let args = build_cargo_args(false, false, Stage2Profile::FullWorkspace, Some(p));
        let s = args_as_strings(&args);
        // Leading chunk is the llvm-cov wrapper. `--ignore-run-fail` keeps
        // report generation running when a test fails — without it cargo
        // llvm-cov skips writing the JSON and coverage reports as
        // "unavailable" even on an otherwise-valid run (fixed in Phase 5).
        assert_eq!(s[0], "llvm-cov");
        assert_eq!(s[1], "--json");
        assert_eq!(s[2], "--ignore-run-fail");
        assert_eq!(s[3], "--output-path");
        assert_eq!(s[4], "/tmp/sample.json");
        // Test subcommand starts at index 5. Note: no explicit `--no-fail-fast`
        // here because `--ignore-run-fail` is mutually exclusive with it and
        // implies it internally.
        assert_eq!(&s[5..9], &["test", "--workspace", "--lib", "--tests"]);
        // Trailing pass-through to libtest.
        assert_eq!(&s[9..], &["--", "--test-threads=1"]);
    }

    #[test]
    fn coverage_on_excludes_explicit_no_fail_fast_flag() {
        // Regression guard: --ignore-run-fail and --no-fail-fast are mutually
        // exclusive on cargo-llvm-cov's CLI. Putting both back would cause a
        // hard error at invocation time. See Phase 5 smoke-run findings.
        let p = Path::new("/tmp/sample.json");
        let args = build_cargo_args(false, false, Stage2Profile::FullWorkspace, Some(p));
        let s = args_as_strings(&args);
        assert!(s.contains(&"--ignore-run-fail".to_string()));
        assert!(!s.contains(&"--no-fail-fast".to_string()));
    }

    #[test]
    fn missing_ietf_vectors_appends_skip_filter() {
        let args = build_cargo_args(true, true, Stage2Profile::FullWorkspace, None);
        let s = args_as_strings(&args);
        // The tail after `--test-threads=1` is `--skip testvector`, matching
        // the flat IETF integration-test names (`testvectorNN_mono/stereo`).
        let tail: &[String] = &s[s.len() - 2..];
        assert_eq!(tail, &["--skip".to_string(), "testvector".to_string()]);
    }

    #[test]
    fn present_ietf_vectors_does_not_append_skip_filter() {
        let args = build_cargo_args(true, false, Stage2Profile::FullWorkspace, None);
        let s = args_as_strings(&args);
        assert!(!s.iter().any(|a| a == "--skip"));
        assert!(!s.iter().any(|a| a == "testvector"));
    }

    #[test]
    fn quick_release_preflight_skip_coverage_uses_package_profile() {
        let args = build_cargo_args(true, false, Stage2Profile::QuickReleasePreflight, None);
        let s = args_as_strings(&args);
        assert_eq!(
            s,
            vec![
                "test",
                "--workspace",
                "--exclude",
                "conformance",
                "--exclude",
                "ropus-harness-control",
                "--exclude",
                "ropus-harness-deep-plc",
                "--lib",
                "--tests",
                "--no-fail-fast",
                "--",
                "--test-threads=1",
            ]
        );
    }

    #[test]
    fn missing_ietf_vectors_and_quick_release_preflight_append_only_vector_skip() {
        let args = build_cargo_args(true, true, Stage2Profile::QuickReleasePreflight, None);
        let s = args_as_strings(&args);
        let tail: &[String] = &s[s.len() - 2..];
        assert_eq!(tail, &["--skip".to_string(), "testvector".to_string()]);
        assert!(!s.iter().any(|a| a == "extensions"));
    }

    #[test]
    fn quick_release_preflight_coverage_uses_package_profile() {
        let p = Path::new("/tmp/sample.json");
        let args = build_cargo_args(false, false, Stage2Profile::QuickReleasePreflight, Some(p));
        let s = args_as_strings(&args);
        assert_eq!(
            s,
            vec![
                "llvm-cov",
                "--json",
                "--ignore-run-fail",
                "--output-path",
                "/tmp/sample.json",
                "test",
                "--workspace",
                "--exclude",
                "conformance",
                "--exclude",
                "ropus-harness-control",
                "--exclude",
                "ropus-harness-deep-plc",
                "--lib",
                "--tests",
                "--",
                "--test-threads=1",
            ]
        );
    }

    #[test]
    fn normal_stage_2_does_not_use_quick_package_profile() {
        let args = build_cargo_args(true, false, Stage2Profile::FullWorkspace, None);
        let s = args_as_strings(&args);
        assert!(!s.iter().any(|a| a == "--exclude"));
        assert!(!s.iter().any(|a| a == "conformance"));
    }

    #[test]
    fn stage_header_matches_quick_release_preflight_command_shape() {
        assert_eq!(
            stage_header(true, true, Stage2Profile::QuickReleasePreflight),
            "cargo test --workspace --exclude conformance --exclude ropus-harness-control --exclude ropus-harness-deep-plc --lib --tests --no-fail-fast -- --test-threads=1 --skip testvector"
        );
    }

    #[test]
    fn stage_header_keeps_normal_stage_2_unchanged() {
        assert_eq!(
            stage_header(true, false, Stage2Profile::FullWorkspace),
            "cargo test --workspace --lib --tests --no-fail-fast -- --test-threads=1"
        );
    }

    #[test]
    fn unavailable_reason_text_explains_filtered_vectors() {
        let provision = IetfVectorProvision {
            status: ProvisionStatus::Unavailable,
            attempted_fetch: true,
            script: None,
            exit_code: Some(7),
            reason: Some("network unavailable".to_string()),
        };
        let r = unavailable_reason_text(&provision);
        assert!(r.contains("IETF vectors unavailable"));
        assert!(r.contains("testvectorNN"));
        assert!(r.contains("network unavailable"));
    }

    #[test]
    fn unavailable_vectors_append_synthetic_failure() {
        let provision = IetfVectorProvision {
            status: ProvisionStatus::Unavailable,
            attempted_fetch: true,
            script: None,
            exit_code: Some(7),
            reason: Some("network unavailable".to_string()),
        };
        let mut t = TestsResult {
            total_passed: 10,
            ..TestsResult::default()
        };
        append_ietf_provisioning_failure(&mut t, &provision);

        assert_eq!(t.total_passed, 10);
        assert_eq!(t.total_failed, 1);
        assert_eq!(
            t.failed_test_names,
            vec!["conformance::ietf_vectors::provisioning".to_string()]
        );
        assert_eq!(t.binaries.len(), 1);
        assert_eq!(t.binaries[0].name, "ietf_vectors");
        assert_eq!(t.binaries[0].failed, 1);
        assert!(t.skip_reason.unwrap().contains("network unavailable"));
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
