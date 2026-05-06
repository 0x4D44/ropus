//! Phase 3 JSON envelope.
//!
//! The HLD's Stage 5 is an HTML report; Phase 4 will build that. Through
//! Phase 3 we emit a structured JSON blob so the supervisor has a stable
//! contract to parse. Field shape matches the task brief verbatim.

use serde_json::{Value, json};

use crate::ambisonics::AmbisonicsResult;
use crate::bench::BenchResult;
use crate::ietf_vectors::IetfVectorProvision;
use crate::preflight::{AssetProbe, Outcome as PreflightOutcome};
use crate::quality::{Check, Outcome as QualityOutcome};
use crate::setup::SetupInfo;
use crate::tests::Outcome as TestsOutcome;

pub struct Envelope<'a> {
    pub setup: &'a SetupInfo,
    pub quality: &'a QualityOutcome,
    pub tests: &'a TestsOutcome,
    pub ambisonics: &'a AmbisonicsResult,
    pub bench: &'a BenchResult,
    pub exit_code: u8,
}

impl Envelope<'_> {
    pub fn to_json(&self) -> Value {
        json!({
            "setup": setup_to_json(self.setup),
            "stages": {
                "quality": quality_to_json(self.quality),
                "tests": tests_to_json(self.tests),
                "ambisonics": ambisonics_to_json(self.ambisonics),
                "bench": bench_to_json(self.bench),
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
        "ietf_vectors": ietf_vectors_to_json(&s.ietf_vectors),
        "preflight": preflight_to_json(&s.preflight),
        "flags": {
            "quick": s.options_snapshot.quick,
            "skip_quality": s.options_snapshot.skip_quality,
            "skip_coverage": s.options_snapshot.skip_coverage,
            "skip_benchmarks": s.options_snapshot.skip_benchmarks,
            "skip_ambisonics": s.options_snapshot.skip_ambisonics,
            "release_preflight": s.options_snapshot.release_preflight,
        },
    })
}

fn ietf_vectors_to_json(v: &IetfVectorProvision) -> Value {
    json!({
        "status": v.status_label(),
        "available": v.available(),
        "attempted_fetch": v.attempted_fetch,
        "script": v.script.as_ref().map(|p| p.to_string_lossy().replace('\\', "/")),
        "exit_code": v.exit_code,
        "reason": v.reason,
    })
}

fn preflight_to_json(p: &PreflightOutcome) -> Value {
    json!({
        "release_preflight": p.release_preflight,
        "profile": p.profile,
        "banner_blocking_missing": p.banner_blocking_missing(),
        "assets": p.assets.iter().map(preflight_asset_to_json).collect::<Vec<_>>(),
    })
}

fn preflight_asset_to_json(asset: &AssetProbe) -> Value {
    json!({
        "key": asset.key,
        "label": asset.label,
        "requirement": asset.requirement.as_str(),
        "status": asset.status.as_str(),
        "available": asset.status.available(),
        "probes": asset.probes,
        "note": asset.note,
        "banner_blocking": asset.banner_blocking,
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

fn ambisonics_to_json(o: &AmbisonicsResult) -> Value {
    // Reuse serde's derive on `AmbisonicsResult`/`OrderOutcome` rather than
    // hand-writing a field mirror — keeps the JSON in lock-step with the
    // struct shape. Falls back to Null defensively (matches the style of
    // `tests_to_json`).
    serde_json::to_value(o).unwrap_or(Value::Null)
}

fn bench_to_json(o: &BenchResult) -> Value {
    // Same splice pattern as ambisonics/tests: derive-Serialize on the
    // outcome struct, then hand the `Value` to the envelope verbatim.
    serde_json::to_value(o).unwrap_or(Value::Null)
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
    use crate::ambisonics::{AmbisonicsResult, OrderOutcome};
    use crate::bench::{BenchResult, VectorBench};
    use crate::cargo_parse::{BinaryResult, Outcome as OutcomeKind, TestOutcome, TestsResult};
    use crate::cli::Options;
    use crate::ietf_vectors::IetfVectorProvision;
    use crate::llvm_cov_parse::{CoverageMetrics, CoverageResult};
    use crate::tests::Outcome as TestsStageOutcome;

    fn asset_json<'a>(assets: &'a [Value], key: &str) -> &'a Value {
        assets
            .iter()
            .find(|asset| asset["key"] == key)
            .unwrap_or_else(|| panic!("asset {key} not found"))
    }

    fn dummy_setup() -> SetupInfo {
        SetupInfo {
            commit: "dd3fb17".to_string(),
            branch: "main".to_string(),
            version: "0.9.0".to_string(),
            ietf_vectors: IetfVectorProvision::present(),
            ietf_vectors_present: true,
            preflight: crate::preflight::Outcome::inactive(&IetfVectorProvision::present()),
            options_snapshot: Options::default(),
        }
    }

    /// Default-skipped ambisonics/bench results for Phase 2-era tests that
    /// don't care about the new stages. Keeps the old assertions intact
    /// while still exercising the new envelope contract.
    fn skipped_ambisonics() -> AmbisonicsResult {
        AmbisonicsResult::skipped("phase 2 compat")
    }

    fn skipped_bench() -> BenchResult {
        BenchResult::skipped("phase 2 compat")
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
        let amb = skipped_ambisonics();
        let bench = skipped_bench();
        let env = Envelope {
            setup: &setup,
            quality: &quality,
            tests: &tests,
            ambisonics: &amb,
            bench: &bench,
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
        let amb = skipped_ambisonics();
        let bench = skipped_bench();
        let env = Envelope {
            setup: &setup,
            quality: &quality,
            tests: &tests,
            ambisonics: &amb,
            bench: &bench,
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
        let amb = skipped_ambisonics();
        let bench = skipped_bench();
        let env = Envelope {
            setup: &setup,
            quality: &quality,
            tests: &tests,
            ambisonics: &amb,
            bench: &bench,
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
        let amb = skipped_ambisonics();
        let bench = skipped_bench();
        let env = Envelope {
            setup: &setup,
            quality: &quality,
            tests: &tests,
            ambisonics: &amb,
            bench: &bench,
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
        let amb = skipped_ambisonics();
        let bench = skipped_bench();
        let env = Envelope {
            setup: &setup,
            quality: &quality,
            tests: &tests,
            ambisonics: &amb,
            bench: &bench,
            exit_code: 0,
        };
        let v = env.to_json();

        assert_eq!(v["setup"]["commit"], "dd3fb17");
        assert_eq!(v["setup"]["branch"], "main");
        assert_eq!(v["setup"]["version"], "0.9.0");
        assert_eq!(v["setup"]["ietf_vectors_present"], true);
        assert_eq!(v["setup"]["ietf_vectors"]["status"], "present");
        assert_eq!(v["setup"]["ietf_vectors"]["available"], true);
        assert_eq!(v["setup"]["ietf_vectors"]["attempted_fetch"], false);
        assert_eq!(v["setup"]["preflight"]["profile"], "core");
        assert_eq!(v["setup"]["preflight"]["release_preflight"], false);
        assert_eq!(
            v["setup"]["preflight"]["assets"].as_array().unwrap().len(),
            6
        );
        assert_eq!(v["setup"]["flags"]["quick"], false);
        assert_eq!(v["setup"]["flags"]["release_preflight"], false);

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
    fn serialises_preflight_asset_contract() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut setup = dummy_setup();
        setup.options_snapshot.release_preflight = true;
        setup.preflight =
            crate::preflight::capture(tmp.path(), true, &IetfVectorProvision::present());
        let quality = quality_ok();
        let tests = TestsStageOutcome::skipped("preflight contract");
        let amb = skipped_ambisonics();
        let bench = skipped_bench();
        let env = Envelope {
            setup: &setup,
            quality: &quality,
            tests: &tests,
            ambisonics: &amb,
            bench: &bench,
            exit_code: 1,
        };
        let v = env.to_json();
        let preflight = &v["setup"]["preflight"];
        let assets = preflight["assets"].as_array().expect("preflight assets");

        assert_eq!(preflight["profile"], "core");
        assert_eq!(preflight["release_preflight"], true);
        assert_eq!(preflight["banner_blocking_missing"], true);
        assert_eq!(assets.len(), 6);

        let fixed = asset_json(assets, "fixed_reference");
        assert_eq!(fixed["requirement"], "required");
        assert_eq!(fixed["status"], "missing_required");
        assert_eq!(fixed["available"], false);
        assert_eq!(fixed["banner_blocking"], true);

        let ietf = asset_json(assets, "ietf_vectors");
        assert_eq!(ietf["requirement"], "required");
        assert_eq!(ietf["status"], "present_required");
        assert_eq!(ietf["banner_blocking"], false);

        let dnn = asset_json(assets, "dnn_base_weights");
        assert_eq!(dnn["requirement"], "optional");
        assert_eq!(dnn["status"], "missing_optional");
        assert_eq!(dnn["banner_blocking"], false);

        assert_eq!(v["setup"]["flags"]["release_preflight"], true);
    }

    #[test]
    fn skipped_quality_stage_serialises_empty_checks() {
        let setup = dummy_setup();
        let quality = QualityOutcome::skipped();
        let tests = TestsStageOutcome::skipped("phase 1 compat");
        let amb = skipped_ambisonics();
        let bench = skipped_bench();
        let env = Envelope {
            setup: &setup,
            quality: &quality,
            tests: &tests,
            ambisonics: &amb,
            bench: &bench,
            exit_code: 0,
        };
        let v = env.to_json();
        assert_eq!(v["stages"]["quality"]["skipped"], true);
        assert_eq!(
            v["stages"]["quality"]["checks"].as_array().unwrap().len(),
            0
        );
    }

    // ---- Phase 3: ambisonics + bench envelope coverage ----

    fn ambisonics_populated() -> AmbisonicsResult {
        AmbisonicsResult {
            skipped: false,
            skip_reason: None,
            build_failed: false,
            duration_ms: 14_200,
            overall_pass: true,
            per_order: (1u8..=5u8)
                .map(|o| OrderOutcome {
                    order: o,
                    passed: true,
                    detail: format!("5 frames, encode 5/5, decode 5/5 (order {o})"),
                })
                .collect(),
        }
    }

    fn bench_populated() -> BenchResult {
        BenchResult {
            skipped: false,
            skip_reason: None,
            build_failed: false,
            duration_ms: 1_800_000,
            vectors: vec![
                VectorBench {
                    label: "SILK NB 8k mono noise".to_string(),
                    bitrate: 16_000,
                    skipped: false,
                    skip_reason: None,
                    crashed: false,
                    crash_reason: None,
                    c_encode_ms: Some(100.0),
                    rust_encode_ms: Some(75.0),
                    enc_ratio: Some(0.75),
                    c_decode_ms: Some(20.0),
                    rust_decode_ms: Some(22.0),
                    dec_ratio: Some(1.1),
                },
                VectorBench {
                    label: "MUSIC 48k stereo".to_string(),
                    bitrate: 128_000,
                    skipped: true,
                    skip_reason: Some(
                        "fixture missing at tests/vectors/music_48k_stereo.wav".to_string(),
                    ),
                    crashed: false,
                    crash_reason: None,
                    c_encode_ms: None,
                    rust_encode_ms: None,
                    enc_ratio: None,
                    c_decode_ms: None,
                    rust_decode_ms: None,
                    dec_ratio: None,
                },
            ],
        }
    }

    #[test]
    fn phase3_envelope_exposes_ambisonics_stage() {
        let setup = dummy_setup();
        let quality = quality_ok();
        let tests = tests_populated();
        let amb = ambisonics_populated();
        let bench = skipped_bench();
        let env = Envelope {
            setup: &setup,
            quality: &quality,
            tests: &tests,
            ambisonics: &amb,
            bench: &bench,
            exit_code: 0,
        };
        let v = env.to_json();

        assert_eq!(v["stages"]["ambisonics"]["skipped"], false);
        assert_eq!(v["stages"]["ambisonics"]["build_failed"], false);
        assert_eq!(v["stages"]["ambisonics"]["overall_pass"], true);
        assert_eq!(v["stages"]["ambisonics"]["duration_ms"], 14_200);
        let per_order = v["stages"]["ambisonics"]["per_order"].as_array().unwrap();
        assert_eq!(per_order.len(), 5);
        assert_eq!(per_order[0]["order"], 1);
        assert_eq!(per_order[0]["passed"], true);
        assert!(
            per_order[0]["detail"]
                .as_str()
                .unwrap()
                .contains("encode 5/5")
        );
    }

    #[test]
    fn phase3_envelope_exposes_bench_stage() {
        let setup = dummy_setup();
        let quality = quality_ok();
        let tests = tests_populated();
        let amb = skipped_ambisonics();
        let bench = bench_populated();
        let env = Envelope {
            setup: &setup,
            quality: &quality,
            tests: &tests,
            ambisonics: &amb,
            bench: &bench,
            exit_code: 0,
        };
        let v = env.to_json();

        assert_eq!(v["stages"]["bench"]["skipped"], false);
        assert_eq!(v["stages"]["bench"]["duration_ms"], 1_800_000);
        let vectors = v["stages"]["bench"]["vectors"].as_array().unwrap();
        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors[0]["label"], "SILK NB 8k mono noise");
        assert_eq!(vectors[0]["bitrate"], 16_000);
        assert_eq!(vectors[0]["c_encode_ms"], 100.0);
        assert_eq!(vectors[0]["rust_encode_ms"], 75.0);
        assert_eq!(vectors[0]["enc_ratio"], 0.75);
        assert_eq!(vectors[0]["dec_ratio"], 1.1);
        assert_eq!(vectors[1]["skipped"], true);
        assert!(vectors[1]["c_encode_ms"].is_null());
        assert!(vectors[1]["enc_ratio"].is_null());
        assert_eq!(
            vectors[1]["skip_reason"],
            "fixture missing at tests/vectors/music_48k_stereo.wav"
        );
    }

    #[test]
    fn skipped_ambisonics_stage_serialises_with_reason() {
        let setup = dummy_setup();
        let quality = quality_ok();
        let tests = tests_populated();
        let amb = AmbisonicsResult::skipped("--skip-ambisonics");
        let bench = skipped_bench();
        let env = Envelope {
            setup: &setup,
            quality: &quality,
            tests: &tests,
            ambisonics: &amb,
            bench: &bench,
            exit_code: 0,
        };
        let v = env.to_json();
        assert_eq!(v["stages"]["ambisonics"]["skipped"], true);
        assert_eq!(
            v["stages"]["ambisonics"]["skip_reason"],
            "--skip-ambisonics"
        );
    }

    #[test]
    fn skipped_bench_stage_serialises_with_reason() {
        let setup = dummy_setup();
        let quality = quality_ok();
        let tests = tests_populated();
        let amb = skipped_ambisonics();
        let bench = BenchResult::skipped("--quick");
        let env = Envelope {
            setup: &setup,
            quality: &quality,
            tests: &tests,
            ambisonics: &amb,
            bench: &bench,
            exit_code: 0,
        };
        let v = env.to_json();
        assert_eq!(v["stages"]["bench"]["skipped"], true);
        assert_eq!(v["stages"]["bench"]["skip_reason"], "--quick");
    }

    #[test]
    fn upstream_build_failure_propagates_to_ambisonics_and_bench_skip_reasons() {
        // Matches the HLD chaining rule: when Stage 2 reports build_failed,
        // Stage 3 and Stage 4 are skipped with reason "upstream build failure".
        // main.rs is responsible for threading the skip; the envelope just
        // has to round-trip the reason verbatim.
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
        let amb = AmbisonicsResult::skipped("upstream build failure");
        let bench = BenchResult::skipped("upstream build failure");
        let env = Envelope {
            setup: &setup,
            quality: &quality,
            tests: &tests,
            ambisonics: &amb,
            bench: &bench,
            exit_code: 1,
        };
        let v = env.to_json();
        assert_eq!(
            v["stages"]["ambisonics"]["skip_reason"],
            "upstream build failure"
        );
        assert_eq!(
            v["stages"]["bench"]["skip_reason"],
            "upstream build failure"
        );
        assert_eq!(v["stages"]["tests"]["build_failed"], true);
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
        let amb = skipped_ambisonics();
        let bench = skipped_bench();
        let env = Envelope {
            setup: &setup,
            quality: &quality,
            tests: &tests,
            ambisonics: &amb,
            bench: &bench,
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
