//! Phase 2 JSON envelope.
//!
//! The HLD's Stage 5 is an HTML report; Phase 4 will build that. Through
//! Phase 2 we emit a structured JSON blob so the supervisor has a stable
//! contract to parse. Field shape matches the task brief verbatim.

use serde_json::{Value, json};

use crate::quality::{Check, Outcome as QualityOutcome};
use crate::setup::SetupInfo;
use crate::tests::Outcome as TestsOutcome;

pub struct Envelope<'a> {
    pub setup: &'a SetupInfo,
    pub quality: &'a QualityOutcome,
    pub tests: &'a TestsOutcome,
    pub exit_code: u8,
}

impl Envelope<'_> {
    pub fn to_json(&self) -> Value {
        json!({
            "setup": setup_to_json(self.setup),
            "stages": {
                "quality": quality_to_json(self.quality),
                "tests": tests_to_json(self.tests),
            },
            "exit_code": self.exit_code,
        })
    }
}

fn setup_to_json(s: &SetupInfo) -> Value {
    json!({
        "commit": s.commit,
        "branch": s.branch,
        "version": s.version,
        "ietf_vectors_present": s.ietf_vectors_present,
        "flags": {
            "quick": s.options_snapshot.quick,
            "skip_quality": s.options_snapshot.skip_quality,
            "skip_coverage": s.options_snapshot.skip_coverage,
            "skip_benchmarks": s.options_snapshot.skip_benchmarks,
            "skip_ambisonics": s.options_snapshot.skip_ambisonics,
        },
    })
}

fn quality_to_json(o: &QualityOutcome) -> Value {
    json!({
        "skipped": o.skipped,
        "checks": o.checks.iter().map(check_to_json).collect::<Vec<_>>(),
    })
}

fn check_to_json(c: &Check) -> Value {
    json!({
        "name": c.name,
        "passed": c.passed,
        "duration_ms": c.duration.as_millis() as u64,
        "issues": c.issues,
    })
}

fn tests_to_json(o: &TestsOutcome) -> Value {
    // Defer to serde's derived representation for the inner types, then
    // splice in the coverage payload + a couple of convenience fields the
    // supervisor wants at the top level.
    //
    // We re-serialise via `serde_json::to_value` rather than `json!({...})`
    // so the field set stays in sync with the struct definitions in
    // `cargo_parse.rs` / `llvm_cov_parse.rs` without a manual mirror here.
    let tests_v = serde_json::to_value(&o.tests).unwrap_or(Value::Null);
    // Coverage field: an object when coverage ran, Null when --skip-coverage
    // or when the JSON couldn't be read. The HLD explicitly spells out that
    // this is null in the skipped case.
    let coverage_v = match &o.coverage {
        Some(c) => serde_json::to_value(c).unwrap_or(Value::Null),
        None => Value::Null,
    };
    let mut obj = match tests_v {
        Value::Object(m) => m,
        _ => serde_json::Map::new(),
    };
    obj.insert("coverage".to_string(), coverage_v);
    obj.insert(
        "coverage_skipped".to_string(),
        Value::Bool(o.coverage_skipped),
    );
    Value::Object(obj)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::cargo_parse::{BinaryResult, Outcome as OutcomeKind, TestOutcome, TestsResult};
    use crate::cli::Options;
    use crate::llvm_cov_parse::{CoverageMetrics, CoverageResult};
    use crate::tests::Outcome as TestsStageOutcome;

    fn dummy_setup() -> SetupInfo {
        SetupInfo {
            commit: "dd3fb17".to_string(),
            branch: "main".to_string(),
            version: "0.9.0".to_string(),
            ietf_vectors_present: true,
            options_snapshot: Options::default(),
        }
    }

    fn quality_ok() -> QualityOutcome {
        QualityOutcome {
            skipped: false,
            checks: vec![
                Check {
                    name: "fmt",
                    passed: true,
                    duration: Duration::from_millis(1_240),
                    issues: Vec::new(),
                },
                Check {
                    name: "clippy",
                    passed: true,
                    duration: Duration::from_millis(58_310),
                    issues: Vec::new(),
                },
            ],
        }
    }

    fn tests_populated() -> TestsStageOutcome {
        let t = TestsResult {
            duration_ms: 2_736_412,
            binaries: vec![BinaryResult {
                name: "ropus".to_string(),
                passed: 1021,
                failed: 0,
                ignored: 12,
                duration_ms: 2_700_000,
            }],
            per_test: vec![TestOutcome {
                fqn: "silk::decoder::tests::test_alpha".to_string(),
                outcome: OutcomeKind::Pass,
                binary: "ropus".to_string(),
            }],
            total_passed: 1021,
            total_failed: 0,
            total_ignored: 12,
            ..TestsResult::default()
        };
        let coverage = CoverageResult {
            ropus: CoverageMetrics {
                lines_covered: 1000,
                lines_total: 1200,
                functions_covered: 200,
                functions_total: 220,
                branches_covered: 500,
                branches_total: 600,
                regions_covered: 1800,
                regions_total: 2000,
            },
            workspace: CoverageMetrics {
                lines_covered: 1500,
                lines_total: 1800,
                ..CoverageMetrics::default()
            },
        };
        TestsStageOutcome {
            tests: t,
            coverage: Some(coverage),
            coverage_skipped: false,
        }
    }

    #[test]
    fn phase2_envelope_has_stages_tests() {
        let setup = dummy_setup();
        let quality = quality_ok();
        let tests = tests_populated();
        let env = Envelope {
            setup: &setup,
            quality: &quality,
            tests: &tests,
            exit_code: 0,
        };
        let v = env.to_json();

        assert_eq!(v["stages"]["tests"]["total_passed"], 1021);
        assert_eq!(v["stages"]["tests"]["total_failed"], 0);
        assert_eq!(v["stages"]["tests"]["total_ignored"], 12);
        assert_eq!(v["stages"]["tests"]["duration_ms"], 2_736_412);
        assert_eq!(v["stages"]["tests"]["build_failed"], false);
        assert_eq!(v["stages"]["tests"]["skipped"], false);
        assert_eq!(v["stages"]["tests"]["coverage_skipped"], false);

        // Coverage sub-object shape.
        let cov = &v["stages"]["tests"]["coverage"];
        assert_eq!(cov["ropus"]["lines_covered"], 1000);
        assert_eq!(cov["ropus"]["lines_total"], 1200);
        assert_eq!(cov["ropus"]["branches_total"], 600);
        assert_eq!(cov["workspace"]["lines_total"], 1800);
    }

    #[test]
    fn skip_coverage_yields_null_coverage() {
        let setup = dummy_setup();
        let quality = quality_ok();
        let t = TestsResult {
            total_passed: 50,
            ..TestsResult::default()
        };
        let tests = TestsStageOutcome {
            tests: t,
            coverage: None,
            coverage_skipped: true,
        };
        let env = Envelope {
            setup: &setup,
            quality: &quality,
            tests: &tests,
            exit_code: 0,
        };
        let v = env.to_json();
        assert!(v["stages"]["tests"]["coverage"].is_null());
        assert_eq!(v["stages"]["tests"]["coverage_skipped"], true);
    }

    #[test]
    fn skipped_tests_stage_serialises_with_reason() {
        let setup = dummy_setup();
        let quality = quality_ok();
        let tests = TestsStageOutcome::skipped("stage disabled via flag");
        let env = Envelope {
            setup: &setup,
            quality: &quality,
            tests: &tests,
            exit_code: 0,
        };
        let v = env.to_json();
        assert_eq!(v["stages"]["tests"]["skipped"], true);
        assert_eq!(
            v["stages"]["tests"]["skip_reason"],
            "stage disabled via flag"
        );
    }

    #[test]
    fn build_failure_surfaces_on_stage() {
        let setup = dummy_setup();
        let quality = quality_ok();
        let t = TestsResult {
            build_failed: true,
            ..TestsResult::default()
        };
        let tests = TestsStageOutcome {
            tests: t,
            coverage: None,
            coverage_skipped: false,
        };
        let env = Envelope {
            setup: &setup,
            quality: &quality,
            tests: &tests,
            exit_code: 1,
        };
        let v = env.to_json();
        assert_eq!(v["stages"]["tests"]["build_failed"], true);
        assert_eq!(v["exit_code"], 1);
    }

    // ---- Phase 1 envelope tests retained for regression coverage. ----

    #[test]
    fn serialises_phase1_envelope_with_expected_shape() {
        let setup = dummy_setup();
        let quality = quality_ok();
        let tests = TestsStageOutcome::skipped("phase 1 compat");
        let env = Envelope {
            setup: &setup,
            quality: &quality,
            tests: &tests,
            exit_code: 0,
        };
        let v = env.to_json();

        assert_eq!(v["setup"]["commit"], "dd3fb17");
        assert_eq!(v["setup"]["branch"], "main");
        assert_eq!(v["setup"]["version"], "0.9.0");
        assert_eq!(v["setup"]["ietf_vectors_present"], true);
        assert_eq!(v["setup"]["flags"]["quick"], false);

        let checks = v["stages"]["quality"]["checks"].as_array().unwrap();
        assert_eq!(checks.len(), 2);
        assert_eq!(checks[0]["name"], "fmt");
        assert_eq!(checks[0]["passed"], true);
        assert_eq!(checks[0]["duration_ms"], 1_240);
        assert!(checks[0]["issues"].as_array().unwrap().is_empty());
        assert_eq!(checks[1]["name"], "clippy");
        assert_eq!(checks[1]["duration_ms"], 58_310);

        assert_eq!(v["exit_code"], 0);
    }

    #[test]
    fn skipped_quality_stage_serialises_empty_checks() {
        let setup = dummy_setup();
        let quality = QualityOutcome::skipped();
        let tests = TestsStageOutcome::skipped("phase 1 compat");
        let env = Envelope {
            setup: &setup,
            quality: &quality,
            tests: &tests,
            exit_code: 0,
        };
        let v = env.to_json();
        assert_eq!(v["stages"]["quality"]["skipped"], true);
        assert_eq!(
            v["stages"]["quality"]["checks"].as_array().unwrap().len(),
            0
        );
    }

    #[test]
    fn failing_quality_check_propagates_issues() {
        let setup = dummy_setup();
        let quality = QualityOutcome {
            skipped: false,
            checks: vec![Check {
                name: "clippy",
                passed: false,
                duration: Duration::from_secs(1),
                issues: vec!["error[E0308]: mismatched types".to_string()],
            }],
        };
        let tests = TestsStageOutcome::skipped("phase 1 compat");
        let env = Envelope {
            setup: &setup,
            quality: &quality,
            tests: &tests,
            exit_code: 1,
        };
        let v = env.to_json();
        let checks = v["stages"]["quality"]["checks"].as_array().unwrap();
        assert_eq!(checks[0]["passed"], false);
        let issues = checks[0]["issues"].as_array().unwrap();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0], "error[E0308]: mismatched types");
        assert_eq!(v["exit_code"], 1);
    }
}
