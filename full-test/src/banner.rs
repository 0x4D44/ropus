//! Banner classification — PASS / FAIL / WARN.
//!
//! HLD § PASS / FAIL / WARN codifies the three overall-outcome states. The
//! mapping is a small enough piece of logic to live in one place, but important
//! enough to deserve a table-driven test so edge cases (bench crashes, build
//! failures, skipped stages) don't silently flip categories.
//!
//! Contract summary:
//!
//! - **PASS** — stages 1, 2, 3 all green. Stage 4 irrelevant.
//! - **FAIL** — any test failure, any clippy/fmt error, ambisonics mismatch,
//!   a Stage 2 build failure, a release-thresholded benchmark gate failure, or
//!   a claimed platform/sanitizer breadth failure.
//! - **WARN** — stages 1-3 green but stage 4 has crashes or ratio > 1.15×.
//!
//! Exit code: PASS→0, WARN→0, FAIL→1. The WARN→0 behaviour is deliberate —
//! pre-commit runs shouldn't spuriously block developers on bench noise.

use crate::ambisonics::AmbisonicsResult;
use crate::bench::BenchResult;
use crate::corpus::Outcome as CorpusOutcome;
use crate::fuzz::Outcome as FuzzOutcome;
use crate::platform::Outcome as PlatformOutcome;
use crate::preflight::Outcome as PreflightOutcome;
use crate::quality::Outcome as QualityOutcome;
use crate::tests::Outcome as TestsOutcome;

/// The banner classification shown at the top of the report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Banner {
    Pass,
    Fail,
    Warn,
}

impl Banner {
    /// Label rendered in the HTML badge and the stdout one-liner.
    pub fn label(self) -> &'static str {
        match self {
            Banner::Pass => "PASS",
            Banner::Fail => "FAIL",
            Banner::Warn => "WARN",
        }
    }

    /// CSS background colour for the header badge. Kept dark enough for the
    /// dark theme but obvious against a white label.
    pub fn badge_color(self) -> &'static str {
        match self {
            Banner::Pass => "#16a34a",
            Banner::Fail => "#dc2626",
            Banner::Warn => "#d97706",
        }
    }

    /// Exit code per HLD § PASS/FAIL/WARN. WARN does not block exit so
    /// `--quick` pre-commit runs don't trip on bench noise.
    pub fn exit_code(self) -> u8 {
        match self {
            Banner::Pass | Banner::Warn => 0,
            Banner::Fail => 1,
        }
    }
}

/// Bench WARN threshold. Hard-coded at 1.15 = 15% slower than C reference.
/// Rationale: this is a regression check, not a tuning target. Variance
/// floor at --iters=30 is ~±3% based on Phase 3 DA analysis; 15% is well
/// above noise. If real variance turns out higher, expose as a CLI flag.
/// Do not change without measurement. See `wrk_journals/2026.04.19 - DA -
/// full-test-phase4.md` and HLD § Decisions.
pub const BENCH_WARN_RATIO: f64 = 1.15;

/// Test-only convenience wrapper that fills in a default "no platform claim"
/// outcome. Production code path goes through [`classify_with_platform`]
/// directly because `main.rs` always builds a real `PlatformOutcome`.
#[cfg(test)]
fn classify(
    quality: &QualityOutcome,
    tests: &TestsOutcome,
    ambisonics: &AmbisonicsResult,
    bench: &BenchResult,
    fuzz: &FuzzOutcome,
    corpus: &CorpusOutcome,
    preflight: &PreflightOutcome,
) -> Banner {
    let platform = crate::platform::Outcome::not_claimed();
    classify_with_platform(
        quality, tests, ambisonics, bench, fuzz, corpus, &platform, preflight,
    )
}

// Each parameter is the outcome of an independent pipeline stage; the rule
// table below cross-references all of them. Bundling into a struct would
// only rename the field count, not reduce the surface area.
#[allow(clippy::too_many_arguments)]
pub fn classify_with_platform(
    quality: &QualityOutcome,
    tests: &TestsOutcome,
    ambisonics: &AmbisonicsResult,
    bench: &BenchResult,
    fuzz: &FuzzOutcome,
    corpus: &CorpusOutcome,
    platform: &PlatformOutcome,
    preflight: &PreflightOutcome,
) -> Banner {
    // FAIL predicates. Each stage's `all_passed()` is the canonical "green"
    // signal (skipped stages included). A stage-level build failure also
    // trips FAIL — it implies downstream stages skipped on upstream build
    // failure, and we don't want a FAIL to decay to WARN through a chain
    // of "well technically every row was skipped".
    let stage2_failed = tests.tests.build_failed || !tests.all_passed();
    let quality_failed = !quality.all_passed();
    let ambisonics_failed = ambisonics.build_failed || !ambisonics.all_passed();
    let fuzz_failed = fuzz.banner_fail();
    let corpus_failed = corpus.banner_fail();
    let platform_failed = platform.banner_fail();
    let preflight_failed = preflight.banner_blocking_missing();
    let bench_failed = bench.banner_fail();
    if stage2_failed
        || quality_failed
        || ambisonics_failed
        || bench_failed
        || fuzz_failed
        || corpus_failed
        || platform_failed
        || preflight_failed
    {
        return Banner::Fail;
    }

    // WARN predicates. Stages 1-3 are green; stage 4 remains yellow for
    // local/default benchmark anomalies. In release-thresholded runs, severe
    // benchmark failures were handled above; ratios above the local WARN
    // threshold still render yellow without becoming a hard release failure.
    let bench_ratio_warn = bench
        .vectors
        .iter()
        .any(|v| ratio_exceeds_warn(v.enc_ratio) || ratio_exceeds_warn(v.dec_ratio));
    if !bench.all_passed() || bench_ratio_warn || fuzz.banner_warn() {
        return Banner::Warn;
    }

    Banner::Pass
}

/// True when a Rust/C ratio is both present and above the WARN threshold.
/// `None` means the sample didn't produce a timing — we don't WARN on missing
/// data; that's a skip or crash already surfaced elsewhere.
fn ratio_exceeds_warn(ratio: Option<f64>) -> bool {
    match ratio {
        Some(r) if r.is_finite() => r > BENCH_WARN_RATIO,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ambisonics::OrderOutcome;
    use crate::bench::{BenchProfile, VectorBench};
    use crate::cargo_parse::TestsResult;
    use crate::quality::Check;
    use std::time::Duration;

    fn quality_pass() -> QualityOutcome {
        QualityOutcome {
            skipped: false,
            checks: vec![Check {
                name: "fmt",
                passed: true,
                duration: Duration::from_millis(10),
                issues: Vec::new(),
            }],
        }
    }

    fn quality_fail() -> QualityOutcome {
        QualityOutcome {
            skipped: false,
            checks: vec![Check {
                name: "clippy",
                passed: false,
                duration: Duration::from_millis(10),
                issues: vec!["error".to_string()],
            }],
        }
    }

    fn tests_pass() -> TestsOutcome {
        TestsOutcome {
            tests: TestsResult {
                total_passed: 100,
                ..TestsResult::default()
            },
            coverage: None,
            coverage_skipped: true,
        }
    }

    fn tests_fail() -> TestsOutcome {
        TestsOutcome {
            tests: TestsResult {
                total_passed: 99,
                total_failed: 1,
                ..TestsResult::default()
            },
            coverage: None,
            coverage_skipped: true,
        }
    }

    fn tests_build_failed() -> TestsOutcome {
        TestsOutcome {
            tests: TestsResult {
                build_failed: true,
                ..TestsResult::default()
            },
            coverage: None,
            coverage_skipped: true,
        }
    }

    fn ambisonics_pass() -> AmbisonicsResult {
        AmbisonicsResult {
            skipped: false,
            skip_reason: None,
            build_failed: false,
            duration_ms: 0,
            overall_pass: true,
            per_order: vec![OrderOutcome {
                order: 1,
                passed: true,
                detail: "ok".to_string(),
            }],
        }
    }

    fn ambisonics_fail() -> AmbisonicsResult {
        AmbisonicsResult {
            skipped: false,
            skip_reason: None,
            build_failed: false,
            duration_ms: 0,
            overall_pass: false,
            per_order: vec![OrderOutcome {
                order: 2,
                passed: false,
                detail: "mismatch".to_string(),
            }],
        }
    }

    fn bench_pass() -> BenchResult {
        BenchResult {
            profile: BenchProfile::ObservedWarnOnly,
            skipped: false,
            skip_reason: None,
            build_failed: false,
            duration_ms: 0,
            vectors: vec![bench_row(Some(0.85), Some(0.9), false)],
        }
    }

    fn bench_with_crash() -> BenchResult {
        BenchResult {
            profile: BenchProfile::ObservedWarnOnly,
            skipped: false,
            skip_reason: None,
            build_failed: false,
            duration_ms: 0,
            vectors: vec![bench_row(None, None, true)],
        }
    }

    fn bench_with_regression() -> BenchResult {
        BenchResult {
            profile: BenchProfile::ObservedWarnOnly,
            skipped: false,
            skip_reason: None,
            build_failed: false,
            duration_ms: 0,
            vectors: vec![bench_row(Some(1.25), Some(0.9), false)],
        }
    }

    fn bench_row(enc: Option<f64>, dec: Option<f64>, crashed: bool) -> VectorBench {
        VectorBench {
            label: "fake".to_string(),
            bitrate: 64_000,
            skipped: false,
            skip_reason: None,
            crashed,
            crash_reason: if crashed {
                Some("boom".to_string())
            } else {
                None
            },
            c_encode_ms: Some(100.0),
            rust_encode_ms: enc.map(|r| 100.0 * r),
            enc_ratio: enc,
            c_decode_ms: Some(50.0),
            rust_decode_ms: dec.map(|r| 50.0 * r),
            dec_ratio: dec,
        }
    }

    fn release_bench_row(enc: Option<f64>, dec: Option<f64>) -> VectorBench {
        VectorBench {
            label: "SILK NB 8k mono noise".to_string(),
            bitrate: 16_000,
            skipped: false,
            skip_reason: None,
            crashed: false,
            crash_reason: None,
            c_encode_ms: Some(100.0),
            rust_encode_ms: enc.map(|r| 100.0 * r),
            enc_ratio: enc,
            c_decode_ms: Some(100.0),
            rust_decode_ms: dec.map(|r| 100.0 * r),
            dec_ratio: dec,
        }
    }

    fn preflight_pass() -> PreflightOutcome {
        PreflightOutcome::inactive(&crate::ietf_vectors::IetfVectorProvision::present())
    }

    fn preflight_fail() -> PreflightOutcome {
        let tmp = tempfile::tempdir().expect("tempdir");
        crate::preflight::capture(
            tmp.path(),
            crate::preflight::PreflightPolicy::ReleaseCoreSmokeNoNeuralClaim,
            &crate::ietf_vectors::IetfVectorProvision::present(),
        )
    }

    fn fuzz_pass() -> crate::fuzz::Outcome {
        crate::fuzz::Outcome {
            mode: crate::fuzz::Mode::FullSanity,
            status: crate::fuzz::Status::Pass,
            duration_ms: 0,
            command: Vec::new(),
            targets: Vec::new(),
            issues: Vec::new(),
            stdout: String::new(),
            stderr: String::new(),
        }
    }

    fn fuzz_fail() -> crate::fuzz::Outcome {
        crate::fuzz::Outcome {
            status: crate::fuzz::Status::Fail,
            issues: vec!["fuzz_decode build failed".to_string()],
            ..fuzz_pass()
        }
    }

    fn fuzz_warn() -> crate::fuzz::Outcome {
        crate::fuzz::Outcome {
            mode: crate::fuzz::Mode::InventoryOnly,
            status: crate::fuzz::Status::Warn,
            issues: vec!["cargo-fuzz not checked".to_string()],
            ..fuzz_pass()
        }
    }

    fn corpus_pass() -> crate::corpus::Outcome {
        crate::corpus::Outcome::not_claimed_for_tests()
    }

    fn corpus_fail() -> crate::corpus::Outcome {
        crate::corpus::Outcome::failing_for_tests("corpus_diff exited 1")
    }

    fn platform_fail() -> crate::platform::Outcome {
        crate::platform::Outcome::failing_for_tests("generic x86_64 smoke failed")
    }

    #[test]
    fn all_green_is_pass() {
        let b = classify(
            &quality_pass(),
            &tests_pass(),
            &ambisonics_pass(),
            &bench_pass(),
            &fuzz_pass(),
            &corpus_pass(),
            &preflight_pass(),
        );
        assert_eq!(b, Banner::Pass);
    }

    #[test]
    fn all_skipped_stages_is_pass() {
        // Quality + bench skipped, ambisonics skipped, tests skipped → PASS.
        // Skipped stages are green by contract.
        let quality = QualityOutcome::skipped();
        let tests = TestsOutcome {
            tests: TestsResult {
                skipped: true,
                skip_reason: Some("not run".to_string()),
                ..TestsResult::default()
            },
            coverage: None,
            coverage_skipped: true,
        };
        let amb = AmbisonicsResult::skipped("--skip-ambisonics");
        let bench = BenchResult::skipped("--skip-benchmarks");
        assert_eq!(
            classify(
                &quality,
                &tests,
                &amb,
                &bench,
                &fuzz_pass(),
                &corpus_pass(),
                &preflight_pass()
            ),
            Banner::Pass
        );
    }

    #[test]
    fn test_failure_is_fail() {
        let b = classify(
            &quality_pass(),
            &tests_fail(),
            &ambisonics_pass(),
            &bench_pass(),
            &fuzz_pass(),
            &corpus_pass(),
            &preflight_pass(),
        );
        assert_eq!(b, Banner::Fail);
    }

    #[test]
    fn stage2_build_failure_is_fail() {
        let b = classify(
            &quality_pass(),
            &tests_build_failed(),
            &ambisonics_pass(),
            &bench_pass(),
            &fuzz_pass(),
            &corpus_pass(),
            &preflight_pass(),
        );
        assert_eq!(b, Banner::Fail);
    }

    #[test]
    fn quality_failure_is_fail() {
        let b = classify(
            &quality_fail(),
            &tests_pass(),
            &ambisonics_pass(),
            &bench_pass(),
            &fuzz_pass(),
            &corpus_pass(),
            &preflight_pass(),
        );
        assert_eq!(b, Banner::Fail);
    }

    #[test]
    fn ambisonics_mismatch_is_fail() {
        let b = classify(
            &quality_pass(),
            &tests_pass(),
            &ambisonics_fail(),
            &bench_pass(),
            &fuzz_pass(),
            &corpus_pass(),
            &preflight_pass(),
        );
        assert_eq!(b, Banner::Fail);
    }

    #[test]
    fn ambisonics_build_failure_is_fail() {
        let amb = AmbisonicsResult {
            skipped: false,
            skip_reason: None,
            build_failed: true,
            duration_ms: 0,
            overall_pass: false,
            per_order: Vec::new(),
        };
        let b = classify(
            &quality_pass(),
            &tests_pass(),
            &amb,
            &bench_pass(),
            &fuzz_pass(),
            &corpus_pass(),
            &preflight_pass(),
        );
        assert_eq!(b, Banner::Fail);
    }

    #[test]
    fn bench_crash_is_warn() {
        let b = classify(
            &quality_pass(),
            &tests_pass(),
            &ambisonics_pass(),
            &bench_with_crash(),
            &fuzz_pass(),
            &corpus_pass(),
            &preflight_pass(),
        );
        assert_eq!(b, Banner::Warn);
    }

    #[test]
    fn bench_regression_is_warn() {
        let b = classify(
            &quality_pass(),
            &tests_pass(),
            &ambisonics_pass(),
            &bench_with_regression(),
            &fuzz_pass(),
            &corpus_pass(),
            &preflight_pass(),
        );
        assert_eq!(b, Banner::Warn);
    }

    #[test]
    fn bench_build_fail_is_warn() {
        // Even when the prebuild blew up, stages 1-3 are still valid — we
        // treat the bench stage as WARN-only per HLD § PASS/FAIL/WARN.
        let bench = BenchResult {
            profile: BenchProfile::ObservedWarnOnly,
            skipped: false,
            skip_reason: Some("ropus-compare build failed".to_string()),
            build_failed: true,
            duration_ms: 0,
            vectors: Vec::new(),
        };
        let b = classify(
            &quality_pass(),
            &tests_pass(),
            &ambisonics_pass(),
            &bench,
            &fuzz_pass(),
            &corpus_pass(),
            &preflight_pass(),
        );
        assert_eq!(b, Banner::Warn);
    }

    #[test]
    fn release_thresholded_bench_build_fail_is_fail() {
        let bench = BenchResult {
            profile: BenchProfile::ReleaseThresholded,
            skipped: false,
            skip_reason: Some("ropus-compare build failed".to_string()),
            build_failed: true,
            duration_ms: 0,
            vectors: Vec::new(),
        };
        let b = classify(
            &quality_pass(),
            &tests_pass(),
            &ambisonics_pass(),
            &bench,
            &fuzz_pass(),
            &corpus_pass(),
            &preflight_pass(),
        );
        assert_eq!(b, Banner::Fail);
    }

    #[test]
    fn release_thresholded_skip_benchmarks_is_fail() {
        let bench = BenchResult::skipped_with_profile(
            BenchProfile::ReleaseThresholded,
            "--skip-benchmarks disables release benchmark gate",
        );
        let b = classify(
            &quality_pass(),
            &tests_pass(),
            &ambisonics_pass(),
            &bench,
            &fuzz_pass(),
            &corpus_pass(),
            &preflight_pass(),
        );
        assert_eq!(b, Banner::Fail);
    }

    #[test]
    fn release_thresholded_ratio_breach_is_fail() {
        let bench = BenchResult {
            profile: BenchProfile::ReleaseThresholded,
            skipped: false,
            skip_reason: None,
            build_failed: false,
            duration_ms: 0,
            vectors: vec![release_bench_row(Some(2.0), Some(0.9))],
        };
        let b = classify(
            &quality_pass(),
            &tests_pass(),
            &ambisonics_pass(),
            &bench,
            &fuzz_pass(),
            &corpus_pass(),
            &preflight_pass(),
        );
        assert_eq!(b, Banner::Fail);
    }

    #[test]
    fn bench_at_threshold_is_not_warn() {
        // `BENCH_WARN_RATIO` is strict-greater; a row exactly at the
        // threshold stays green.
        let bench = BenchResult {
            profile: BenchProfile::ObservedWarnOnly,
            skipped: false,
            skip_reason: None,
            build_failed: false,
            duration_ms: 0,
            vectors: vec![bench_row(Some(BENCH_WARN_RATIO), Some(0.9), false)],
        };
        let b = classify(
            &quality_pass(),
            &tests_pass(),
            &ambisonics_pass(),
            &bench,
            &fuzz_pass(),
            &corpus_pass(),
            &preflight_pass(),
        );
        assert_eq!(b, Banner::Pass);
    }

    #[test]
    fn bench_with_nan_ratio_is_not_warn() {
        // A NaN / inf ratio is never treated as a regression; it's surfaced
        // elsewhere as a missing-data row.
        let bench = BenchResult {
            profile: BenchProfile::ObservedWarnOnly,
            skipped: false,
            skip_reason: None,
            build_failed: false,
            duration_ms: 0,
            vectors: vec![bench_row(Some(f64::NAN), Some(f64::INFINITY), false)],
        };
        let b = classify(
            &quality_pass(),
            &tests_pass(),
            &ambisonics_pass(),
            &bench,
            &fuzz_pass(),
            &corpus_pass(),
            &preflight_pass(),
        );
        assert_eq!(b, Banner::Pass);
    }

    #[test]
    fn fail_trumps_warn() {
        // A failed test + a crashed bench row → still FAIL, not WARN.
        let b = classify(
            &quality_pass(),
            &tests_fail(),
            &ambisonics_pass(),
            &bench_with_crash(),
            &fuzz_pass(),
            &corpus_pass(),
            &preflight_pass(),
        );
        assert_eq!(b, Banner::Fail);
    }

    #[test]
    fn release_preflight_missing_core_asset_is_fail() {
        let b = classify(
            &quality_pass(),
            &tests_pass(),
            &ambisonics_pass(),
            &bench_pass(),
            &fuzz_pass(),
            &corpus_pass(),
            &preflight_fail(),
        );
        assert_eq!(b, Banner::Fail);
    }

    #[test]
    fn fuzz_failure_is_fail() {
        let b = classify(
            &quality_pass(),
            &tests_pass(),
            &ambisonics_pass(),
            &bench_pass(),
            &fuzz_fail(),
            &corpus_pass(),
            &preflight_pass(),
        );
        assert_eq!(b, Banner::Fail);
    }

    #[test]
    fn claimed_corpus_failure_is_fail() {
        let b = classify(
            &quality_pass(),
            &tests_pass(),
            &ambisonics_pass(),
            &bench_pass(),
            &fuzz_pass(),
            &corpus_fail(),
            &preflight_pass(),
        );
        assert_eq!(b, Banner::Fail);
    }

    #[test]
    fn claimed_platform_breadth_failure_is_fail() {
        let b = classify_with_platform(
            &quality_pass(),
            &tests_pass(),
            &ambisonics_pass(),
            &bench_pass(),
            &fuzz_pass(),
            &corpus_pass(),
            &platform_fail(),
            &preflight_pass(),
        );
        assert_eq!(b, Banner::Fail);
    }

    #[test]
    fn quick_fuzz_tool_absence_warns_without_fail() {
        let b = classify(
            &quality_pass(),
            &tests_pass(),
            &ambisonics_pass(),
            &bench_pass(),
            &fuzz_warn(),
            &corpus_pass(),
            &preflight_pass(),
        );
        assert_eq!(b, Banner::Warn);
    }

    #[test]
    fn exit_code_mapping() {
        assert_eq!(Banner::Pass.exit_code(), 0);
        assert_eq!(Banner::Warn.exit_code(), 0);
        assert_eq!(Banner::Fail.exit_code(), 1);
    }

    #[test]
    fn labels_and_colors_are_stable() {
        // The HTML renderer pulls these verbatim. A drift test keeps the
        // public surface stable.
        assert_eq!(Banner::Pass.label(), "PASS");
        assert_eq!(Banner::Fail.label(), "FAIL");
        assert_eq!(Banner::Warn.label(), "WARN");
        assert!(Banner::Pass.badge_color().starts_with('#'));
        assert!(Banner::Fail.badge_color().starts_with('#'));
        assert!(Banner::Warn.badge_color().starts_with('#'));
    }
}
