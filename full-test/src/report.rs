//! Phase 1 JSON envelope.
//!
//! The HLD's Stage 5 is an HTML report; Phase 4 will build that. Phase 1 just
//! prints a structured JSON blob so the supervisor has something concrete to
//! parse. Field shape matches the task brief verbatim.

use serde_json::{Value, json};

use crate::quality::{Check, Outcome};
use crate::setup::SetupInfo;

pub struct Envelope<'a> {
    pub setup: &'a SetupInfo,
    pub quality: &'a Outcome,
    pub exit_code: u8,
}

impl<'a> Envelope<'a> {
    pub fn to_json(&self) -> Value {
        json!({
            "setup": setup_to_json(self.setup),
            "stages": {
                "quality": outcome_to_json(self.quality),
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

fn outcome_to_json(o: &Outcome) -> Value {
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::cli::Options;

    fn dummy_setup() -> SetupInfo {
        SetupInfo {
            commit: "dd3fb17".to_string(),
            branch: "main".to_string(),
            version: "0.9.0".to_string(),
            ietf_vectors_present: true,
            options_snapshot: Options::default(),
        }
    }

    #[test]
    fn serialises_phase1_envelope_with_expected_shape() {
        let setup = dummy_setup();
        let outcome = Outcome {
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
        };
        let env = Envelope {
            setup: &setup,
            quality: &outcome,
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
    fn skipped_stage_serialises_empty_checks() {
        let setup = dummy_setup();
        let outcome = Outcome::skipped();
        let env = Envelope {
            setup: &setup,
            quality: &outcome,
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
    fn failing_check_propagates_issues() {
        let setup = dummy_setup();
        let outcome = Outcome {
            skipped: false,
            checks: vec![Check {
                name: "clippy",
                passed: false,
                duration: Duration::from_secs(1),
                issues: vec!["error[E0308]: mismatched types".to_string()],
            }],
        };
        let env = Envelope {
            setup: &setup,
            quality: &outcome,
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
