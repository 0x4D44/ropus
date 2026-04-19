//! Stage 1 — quality checks: `cargo fmt --check`, then `cargo clippy
//! --workspace --all-targets -- -D warnings`.
//!
//! Both run sequentially; clippy runs even if fmt failed, because the two
//! signals are independent and developers want both diagnostics surfaced
//! from a single invocation.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use colored::Colorize;

use crate::issues;

#[derive(Debug, Clone)]
pub struct Check {
    pub name: &'static str,
    pub passed: bool,
    pub duration: Duration,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Outcome {
    pub skipped: bool,
    pub checks: Vec<Check>,
}

impl Outcome {
    pub fn skipped() -> Self {
        Self {
            skipped: true,
            checks: Vec::new(),
        }
    }

    /// True when the stage ran cleanly OR was skipped. A skipped stage does
    /// not fail the overall exit code (that's the HLD contract).
    pub fn all_passed(&self) -> bool {
        self.skipped || self.checks.iter().all(|c| c.passed)
    }
}

/// Run the full quality stage: fmt then clippy.
pub fn run() -> Outcome {
    let checks = vec![run_fmt(), run_clippy()];
    Outcome {
        skipped: false,
        checks,
    }
}

fn run_fmt() -> Check {
    eprintln!("{} cargo fmt --check", "[quality]".cyan().bold());
    let start = Instant::now();
    // `cargo fmt --check` already applies to the whole workspace (rustfmt's
    // default); HLD explicitly says no `--all`.
    let output = Command::new("cargo")
        .args(["fmt", "--check"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();
    let duration = start.elapsed();
    let (passed, issues) = match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            // rustfmt's "needs formatting" signal lives on stdout, not stderr;
            // scan both so our issues list catches something useful on failure.
            let (combined, _) = issues::cap_stderr(&format!("{stderr}\n{stdout}"));
            // Distinguish cargo fmt's three relevant exit codes:
            //   0 — all files formatted (pass).
            //   1 — formatting violations found (normal fail; surface issues).
            //   2 — invalid args (hard invocation error).
            //   anything else (incl. signal-kill / rustfmt missing) — tool
            //   unavailable; record distinctly so it isn't mistaken for a
            //   check failure.
            match out.status.code() {
                Some(0) => (true, issues::extract(&combined)),
                Some(1) => (false, issues::extract(&combined)),
                Some(2) => (false, vec!["error: cargo fmt invocation error".to_string()]),
                other => (
                    false,
                    vec![format!(
                        "error: cargo fmt unavailable (exit: {})",
                        match other {
                            Some(c) => c.to_string(),
                            None => "signal".to_string(),
                        }
                    )],
                ),
            }
        }
        Err(e) => (
            false,
            vec![format!("error: failed to spawn cargo fmt: {e}")],
        ),
    };
    eprintln!(
        "  {} in {:.2}s",
        if passed { "PASS".green() } else { "FAIL".red() },
        duration.as_secs_f64()
    );
    Check {
        name: "fmt",
        passed,
        duration,
        issues,
    }
}

fn run_clippy() -> Check {
    eprintln!(
        "{} cargo clippy --workspace --all-targets -- -D warnings",
        "[quality]".cyan().bold()
    );
    let start = Instant::now();
    let output = Command::new("cargo")
        .args([
            "clippy",
            "--workspace",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();
    let duration = start.elapsed();
    let (passed, issues) = match output {
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let (capped, _) = issues::cap_stderr(&stderr);
            (out.status.success(), issues::extract(&capped))
        }
        Err(e) => (
            false,
            vec![format!("error: failed to spawn cargo clippy: {e}")],
        ),
    };
    eprintln!(
        "  {} ({} issue{}) in {:.2}s",
        if passed { "PASS".green() } else { "FAIL".red() },
        issues.len(),
        if issues.len() == 1 { "" } else { "s" },
        duration.as_secs_f64()
    );
    Check {
        name: "clippy",
        passed,
        duration,
        issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outcome_skipped_is_green_for_exit_code() {
        let o = Outcome::skipped();
        assert!(o.skipped);
        assert!(o.all_passed());
    }

    #[test]
    fn outcome_with_failing_check_is_red() {
        let o = Outcome {
            skipped: false,
            checks: vec![Check {
                name: "fmt",
                passed: false,
                duration: Duration::from_millis(10),
                issues: vec!["error: bad".to_string()],
            }],
        };
        assert!(!o.all_passed());
    }

    #[test]
    fn outcome_with_all_passing_checks_is_green() {
        let o = Outcome {
            skipped: false,
            checks: vec![
                Check {
                    name: "fmt",
                    passed: true,
                    duration: Duration::from_millis(10),
                    issues: Vec::new(),
                },
                Check {
                    name: "clippy",
                    passed: true,
                    duration: Duration::from_millis(20),
                    issues: Vec::new(),
                },
            ],
        };
        assert!(o.all_passed());
    }
}
