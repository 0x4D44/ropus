//! Platform/sanitizer release-breadth gate.
//!
//! This stage is deliberately narrow. Default and quick profiles make no
//! platform/sanitizer claim. Non-quick release preflight claims one generic
//! x86_64 `ropus --lib` smoke on x86_64 hosts and one sanitizer-backed claim
//! through the existing cargo-fuzz full-sanity lane.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use serde::Serialize;
use serde::ser::Serializer;

use crate::cli::Options;
use crate::fuzz::{Mode as FuzzMode, Outcome as FuzzOutcome, Status as FuzzStatus};

pub const GENERIC_PROFILE_PROOF_NOTE: &str = "generic x86_64 command verified 2026-05-06 with `cargo rustc -p ropus --lib -- --print cfg`: default x86-64-v3 emitted AVX2/BMI/FMA features, while `RUSTFLAGS='-C target-cpu=x86-64'` emitted only fxsr/sse/sse2 CPU features";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    DefaultNoClaim,
    QuickReleaseNoClaim,
    ReleaseBreadth,
}

impl Serialize for Profile {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl Profile {
    pub fn from_flags(quick: bool, release_preflight: bool) -> Self {
        if !release_preflight {
            Self::DefaultNoClaim
        } else if quick {
            Self::QuickReleaseNoClaim
        } else {
            Self::ReleaseBreadth
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::DefaultNoClaim => "default-no-platform-sanitizer-claim",
            Self::QuickReleaseNoClaim => "quick-release-no-platform-sanitizer-claim",
            Self::ReleaseBreadth => "release-platform-sanitizer-breadth",
        }
    }

    pub fn claimed(self) -> bool {
        matches!(self, Self::ReleaseBreadth)
    }

    pub fn claim_note(self) -> &'static str {
        match self {
            Self::DefaultNoClaim => {
                "default full-test does not claim platform or sanitizer breadth"
            }
            Self::QuickReleaseNoClaim => {
                "quick release-preflight is core smoke only; platform and sanitizer breadth are not claimed"
            }
            Self::ReleaseBreadth => {
                "non-quick release-preflight claims generic x86_64 smoke where applicable plus sanitizer-backed fuzz full-sanity"
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LaneStatus {
    NotClaimed,
    NotApplicable,
    Unsupported,
    Pass,
    Fail,
}

impl LaneStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotClaimed => "not_claimed",
            Self::NotApplicable => "not_applicable",
            Self::Unsupported => "unsupported",
            Self::Pass => "pass",
            Self::Fail => "fail",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LaneOutcome {
    pub id: &'static str,
    pub label: &'static str,
    pub status: LaneStatus,
    pub claimed: bool,
    pub release_blocking: bool,
    pub command: Vec<String>,
    pub duration_ms: u64,
    pub note: String,
    pub issues: Vec<String>,
}

impl LaneOutcome {
    fn not_claimed(id: &'static str, label: &'static str, note: impl Into<String>) -> Self {
        Self {
            id,
            label,
            status: LaneStatus::NotClaimed,
            claimed: false,
            release_blocking: false,
            command: Vec::new(),
            duration_ms: 0,
            note: note.into(),
            issues: Vec::new(),
        }
    }

    fn not_applicable(id: &'static str, label: &'static str, note: impl Into<String>) -> Self {
        Self {
            id,
            label,
            status: LaneStatus::NotApplicable,
            claimed: false,
            release_blocking: false,
            command: Vec::new(),
            duration_ms: 0,
            note: note.into(),
            issues: Vec::new(),
        }
    }

    fn claimed(
        id: &'static str,
        label: &'static str,
        status: LaneStatus,
        command: Vec<String>,
        duration_ms: u64,
        note: impl Into<String>,
        issues: Vec<String>,
    ) -> Self {
        Self {
            id,
            label,
            status,
            claimed: true,
            release_blocking: true,
            command,
            duration_ms,
            note: note.into(),
            issues,
        }
    }

    pub fn banner_fail(&self) -> bool {
        self.release_blocking && matches!(self.status, LaneStatus::Fail | LaneStatus::Unsupported)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Outcome {
    pub profile: Profile,
    pub profile_name: &'static str,
    pub claimed: bool,
    pub claim_note: &'static str,
    pub host_arch: String,
    pub duration_ms: u64,
    pub generic_x86_64: LaneOutcome,
    pub sanitizer: LaneOutcome,
    pub issues: Vec<String>,
}

impl Outcome {
    pub fn banner_fail(&self) -> bool {
        self.generic_x86_64.banner_fail() || self.sanitizer.banner_fail()
    }

    pub fn release_blocking_issues(&self) -> Vec<String> {
        self.generic_x86_64
            .issues
            .iter()
            .chain(self.sanitizer.issues.iter())
            .cloned()
            .collect()
    }

    /// Test-only helper: build a "no platform claim" outcome anchored to
    /// the host architecture. Production never branches into a no-claim
    /// outcome — `run` always produces a real one — so this exists solely
    /// for `banner::classify`'s test wrapper.
    #[cfg(test)]
    pub fn not_claimed() -> Self {
        outcome_without_claim(Profile::DefaultNoClaim, std::env::consts::ARCH)
    }

    #[cfg(test)]
    pub fn not_claimed_for_tests() -> Self {
        outcome_without_claim(Profile::DefaultNoClaim, "x86_64")
    }

    #[cfg(test)]
    pub fn failing_for_tests(issue: impl Into<String>) -> Self {
        let issue = issue.into();
        let mut out = outcome_without_claim(Profile::ReleaseBreadth, "x86_64");
        out.claimed = true;
        out.generic_x86_64 = LaneOutcome::claimed(
            "generic-x86_64-smoke",
            "Generic x86_64 smoke",
            LaneStatus::Fail,
            generic_x86_64_command(),
            12,
            GENERIC_PROFILE_PROOF_NOTE,
            vec![issue.clone()],
        );
        out.sanitizer = LaneOutcome::claimed(
            "asan-fuzz-sanity",
            "ASan fuzz sanity",
            LaneStatus::Pass,
            vec!["timeout".to_string(), "300".to_string()],
            50,
            "sanitizer breadth is satisfied only by the existing cargo-fuzz full-sanity lane; quick fuzz inventory is not sanitizer coverage",
            Vec::new(),
        );
        out.issues = vec![issue];
        out
    }
}

pub fn run(options: &Options, fuzz: &FuzzOutcome) -> Outcome {
    let root = workspace_root();
    run_with_root_and_arch(
        options,
        fuzz,
        &root,
        std::env::consts::ARCH,
        execute_generic_smoke,
    )
}

fn run_with_root_and_arch<F>(
    options: &Options,
    fuzz: &FuzzOutcome,
    root: &Path,
    host_arch: &str,
    mut run_generic: F,
) -> Outcome
where
    F: FnMut(&[String], &Path) -> GenericRun,
{
    let started = Instant::now();
    let profile = Profile::from_flags(options.quick, options.release_preflight);
    let generic_x86_64 = generic_lane(profile, host_arch, root, &mut run_generic);
    let sanitizer = sanitizer_lane(profile, fuzz);
    let mut issues = Vec::new();
    issues.extend(generic_x86_64.issues.iter().cloned());
    issues.extend(sanitizer.issues.iter().cloned());

    Outcome {
        profile,
        profile_name: profile.as_str(),
        claimed: profile.claimed(),
        claim_note: profile.claim_note(),
        host_arch: host_arch.to_string(),
        duration_ms: started.elapsed().as_millis() as u64,
        generic_x86_64,
        sanitizer,
        issues,
    }
}

#[cfg(test)]
fn outcome_without_claim(profile: Profile, host_arch: &str) -> Outcome {
    let generic = LaneOutcome::not_claimed(
        "generic-x86_64-smoke",
        "Generic x86_64 smoke",
        profile.claim_note(),
    );
    let sanitizer =
        LaneOutcome::not_claimed("asan-fuzz-sanity", "ASan fuzz sanity", profile.claim_note());
    Outcome {
        profile,
        profile_name: profile.as_str(),
        claimed: profile.claimed(),
        claim_note: profile.claim_note(),
        host_arch: host_arch.to_string(),
        duration_ms: 0,
        generic_x86_64: generic,
        sanitizer,
        issues: Vec::new(),
    }
}

fn generic_lane<F>(
    profile: Profile,
    host_arch: &str,
    root: &Path,
    run_generic: &mut F,
) -> LaneOutcome
where
    F: FnMut(&[String], &Path) -> GenericRun,
{
    if profile != Profile::ReleaseBreadth {
        return LaneOutcome::not_claimed(
            "generic-x86_64-smoke",
            "Generic x86_64 smoke",
            profile.claim_note(),
        );
    }

    if host_arch != "x86_64" {
        return LaneOutcome::not_applicable(
            "generic-x86_64-smoke",
            "Generic x86_64 smoke",
            format!("host architecture is {host_arch}; no x86_64 lane is claimed"),
        );
    }

    let command = generic_x86_64_command();
    match run_generic(&command, root) {
        GenericRun::Success { duration_ms } => LaneOutcome::claimed(
            "generic-x86_64-smoke",
            "Generic x86_64 smoke",
            LaneStatus::Pass,
            command,
            duration_ms,
            GENERIC_PROFILE_PROOF_NOTE,
            Vec::new(),
        ),
        GenericRun::NonZero {
            code,
            duration_ms,
            output_tail,
        } => {
            let code = code
                .map(|c| c.to_string())
                .unwrap_or_else(|| "terminated by signal".to_string());
            LaneOutcome::claimed(
                "generic-x86_64-smoke",
                "Generic x86_64 smoke",
                LaneStatus::Fail,
                command,
                duration_ms,
                GENERIC_PROFILE_PROOF_NOTE,
                vec![format!(
                    "generic x86_64 smoke command failed with status {code}: {output_tail}"
                )],
            )
        }
        GenericRun::LaunchError { error, duration_ms } => LaneOutcome::claimed(
            "generic-x86_64-smoke",
            "Generic x86_64 smoke",
            LaneStatus::Unsupported,
            command,
            duration_ms,
            GENERIC_PROFILE_PROOF_NOTE,
            vec![format!(
                "generic x86_64 smoke command could not launch: {error}"
            )],
        ),
    }
}

fn sanitizer_lane(profile: Profile, fuzz: &FuzzOutcome) -> LaneOutcome {
    if profile != Profile::ReleaseBreadth {
        return LaneOutcome::not_claimed(
            "asan-fuzz-sanity",
            "ASan fuzz sanity",
            profile.claim_note(),
        );
    }

    let note = "sanitizer breadth is satisfied only by the existing cargo-fuzz full-sanity lane; quick fuzz inventory is not sanitizer coverage";
    match (fuzz.mode, fuzz.status) {
        (FuzzMode::FullSanity, FuzzStatus::Pass) => LaneOutcome::claimed(
            "asan-fuzz-sanity",
            "ASan fuzz sanity",
            LaneStatus::Pass,
            fuzz.command.clone(),
            fuzz.duration_ms,
            note,
            Vec::new(),
        ),
        (FuzzMode::FullSanity, status) => {
            let mut issues = vec![format!(
                "sanitizer breadth requires fuzz full-sanity pass; observed status {}",
                status.as_str()
            )];
            issues.extend(fuzz.issues.iter().cloned());
            LaneOutcome::claimed(
                "asan-fuzz-sanity",
                "ASan fuzz sanity",
                LaneStatus::Fail,
                fuzz.command.clone(),
                fuzz.duration_ms,
                note,
                issues,
            )
        }
        (mode, status) => LaneOutcome::claimed(
            "asan-fuzz-sanity",
            "ASan fuzz sanity",
            LaneStatus::Fail,
            fuzz.command.clone(),
            fuzz.duration_ms,
            note,
            vec![format!(
                "sanitizer breadth requires fuzz full-sanity; observed mode {} with status {}",
                mode.as_str(),
                status.as_str()
            )],
        ),
    }
}

pub fn generic_x86_64_command() -> Vec<String> {
    vec![
        "timeout".to_string(),
        "900".to_string(),
        "env".to_string(),
        "CARGO_TARGET_DIR=target/platform-breadth/x86_64-generic".to_string(),
        "RUSTFLAGS=-C target-cpu=x86-64".to_string(),
        "cargo".to_string(),
        "test".to_string(),
        "-p".to_string(),
        "ropus".to_string(),
        "--lib".to_string(),
        "--".to_string(),
        "--test-threads=1".to_string(),
    ]
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum GenericRun {
    Success {
        duration_ms: u64,
    },
    NonZero {
        code: Option<i32>,
        duration_ms: u64,
        output_tail: String,
    },
    LaunchError {
        error: String,
        duration_ms: u64,
    },
}

fn execute_generic_smoke(command: &[String], root: &Path) -> GenericRun {
    let started = Instant::now();
    let output = Command::new(&command[0])
        .args(&command[1..])
        .current_dir(root)
        .output();
    let duration_ms = started.elapsed().as_millis() as u64;

    match output {
        Ok(output) if output.status.success() => GenericRun::Success { duration_ms },
        Ok(output) => GenericRun::NonZero {
            code: output.status.code(),
            duration_ms,
            output_tail: combined_output_tail(
                &String::from_utf8_lossy(&output.stdout),
                &String::from_utf8_lossy(&output.stderr),
            ),
        },
        Err(error) => GenericRun::LaunchError {
            error: error.to_string(),
            duration_ms,
        },
    }
}

fn combined_output_tail(stdout: &str, stderr: &str) -> String {
    let stdout = tail(stdout);
    let stderr = tail(stderr);
    match (stdout.as_str(), stderr.as_str()) {
        ("(no output captured)", "(no output captured)") => "(no output captured)".to_string(),
        ("(no output captured)", stderr) => format!("stderr:\n{stderr}"),
        (stdout, "(no output captured)") => format!("stdout:\n{stdout}"),
        (stdout, stderr) => format!("stdout:\n{stdout}\nstderr:\n{stderr}"),
    }
}

fn tail(text: &str) -> String {
    let lines = text
        .lines()
        .rev()
        .take(6)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>();
    if lines.is_empty() {
        "(no output captured)".to_string()
    } else {
        lines.join("\n")
    }
}

fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().map(Path::to_path_buf).unwrap_or(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(quick: bool, release_preflight: bool) -> Options {
        Options {
            quick,
            release_preflight,
            ..Options::default()
        }
    }

    fn fuzz(mode: FuzzMode, status: FuzzStatus) -> FuzzOutcome {
        FuzzOutcome {
            mode,
            status,
            duration_ms: 50,
            command: vec!["timeout".to_string(), "300".to_string()],
            targets: Vec::new(),
            issues: if status == FuzzStatus::Fail {
                vec!["fuzz sanity failed".to_string()]
            } else {
                Vec::new()
            },
            stdout: String::new(),
            stderr: String::new(),
        }
    }

    fn run_synthetic(
        options: Options,
        fuzz: FuzzOutcome,
        host_arch: &str,
        generic: GenericRun,
    ) -> Outcome {
        let tmp = tempfile::tempdir().expect("tempdir");
        run_with_root_and_arch(&options, &fuzz, tmp.path(), host_arch, |_cmd, _root| {
            generic.clone()
        })
    }

    #[test]
    fn policy_selects_expected_profiles() {
        assert_eq!(Profile::from_flags(false, false), Profile::DefaultNoClaim);
        assert_eq!(Profile::from_flags(true, false), Profile::DefaultNoClaim);
        assert_eq!(
            Profile::from_flags(true, true),
            Profile::QuickReleaseNoClaim
        );
        assert_eq!(Profile::from_flags(false, true), Profile::ReleaseBreadth);
    }

    #[test]
    fn command_is_bounded_and_sets_generic_rustflags() {
        assert_eq!(
            generic_x86_64_command(),
            vec![
                "timeout",
                "900",
                "env",
                "CARGO_TARGET_DIR=target/platform-breadth/x86_64-generic",
                "RUSTFLAGS=-C target-cpu=x86-64",
                "cargo",
                "test",
                "-p",
                "ropus",
                "--lib",
                "--",
                "--test-threads=1",
            ]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>()
        );
    }

    #[test]
    fn default_profile_does_not_claim_or_run_breadth_lanes() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let fuzz = fuzz(FuzzMode::FullSanity, FuzzStatus::Pass);
        let mut called = false;
        let outcome = run_with_root_and_arch(
            &options(false, false),
            &fuzz,
            tmp.path(),
            "x86_64",
            |_cmd, _root| {
                called = true;
                GenericRun::Success { duration_ms: 1 }
            },
        );

        assert!(!called);
        assert_eq!(outcome.profile, Profile::DefaultNoClaim);
        assert_eq!(outcome.generic_x86_64.status, LaneStatus::NotClaimed);
        assert_eq!(outcome.sanitizer.status, LaneStatus::NotClaimed);
        assert!(!outcome.banner_fail());
    }

    #[test]
    fn quick_release_preflight_does_not_treat_inventory_as_sanitizer_coverage() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let fuzz = fuzz(FuzzMode::InventoryOnly, FuzzStatus::Pass);
        let mut called = false;
        let outcome = run_with_root_and_arch(
            &options(true, true),
            &fuzz,
            tmp.path(),
            "x86_64",
            |_cmd, _root| {
                called = true;
                GenericRun::Success { duration_ms: 1 }
            },
        );

        assert!(!called);
        assert_eq!(outcome.profile, Profile::QuickReleaseNoClaim);
        assert_eq!(outcome.generic_x86_64.status, LaneStatus::NotClaimed);
        assert_eq!(outcome.sanitizer.status, LaneStatus::NotClaimed);
        assert!(!outcome.banner_fail());
    }

    #[test]
    fn release_preflight_on_x86_64_claims_passing_generic_and_sanitizer_lanes() {
        let outcome = run_synthetic(
            options(false, true),
            fuzz(FuzzMode::FullSanity, FuzzStatus::Pass),
            "x86_64",
            GenericRun::Success { duration_ms: 17 },
        );

        assert_eq!(outcome.profile, Profile::ReleaseBreadth);
        assert_eq!(outcome.generic_x86_64.status, LaneStatus::Pass);
        assert!(outcome.generic_x86_64.claimed);
        assert_eq!(outcome.sanitizer.status, LaneStatus::Pass);
        assert!(outcome.sanitizer.claimed);
        assert!(!outcome.banner_fail());
        assert!(outcome.generic_x86_64.note.contains("only fxsr/sse/sse2"));
    }

    #[test]
    fn release_preflight_off_x86_64_reports_generic_lane_not_applicable() {
        let outcome = run_synthetic(
            options(false, true),
            fuzz(FuzzMode::FullSanity, FuzzStatus::Pass),
            "aarch64",
            GenericRun::Success { duration_ms: 17 },
        );

        assert_eq!(outcome.generic_x86_64.status, LaneStatus::NotApplicable);
        assert!(!outcome.generic_x86_64.claimed);
        assert_eq!(outcome.sanitizer.status, LaneStatus::Pass);
        assert!(!outcome.banner_fail());
    }

    #[test]
    fn release_preflight_fails_when_sanitizer_full_sanity_did_not_run() {
        let outcome = run_synthetic(
            options(false, true),
            fuzz(FuzzMode::InventoryOnly, FuzzStatus::Pass),
            "x86_64",
            GenericRun::Success { duration_ms: 17 },
        );

        assert_eq!(outcome.sanitizer.status, LaneStatus::Fail);
        assert!(outcome.banner_fail());
        assert!(outcome.sanitizer.issues[0].contains("requires fuzz full-sanity"));
    }

    #[test]
    fn release_preflight_fails_when_fuzz_full_sanity_fails() {
        let outcome = run_synthetic(
            options(false, true),
            fuzz(FuzzMode::FullSanity, FuzzStatus::Fail),
            "x86_64",
            GenericRun::Success { duration_ms: 17 },
        );

        assert_eq!(outcome.sanitizer.status, LaneStatus::Fail);
        assert!(outcome.banner_fail());
        assert!(
            outcome
                .release_blocking_issues()
                .iter()
                .any(|issue| issue.contains("fuzz sanity failed"))
        );
    }

    #[test]
    fn generic_nonzero_exit_is_release_blocking_fail() {
        let outcome = run_synthetic(
            options(false, true),
            fuzz(FuzzMode::FullSanity, FuzzStatus::Pass),
            "x86_64",
            GenericRun::NonZero {
                code: Some(101),
                duration_ms: 17,
                output_tail: "stdout:\ntest failed".to_string(),
            },
        );

        assert_eq!(outcome.generic_x86_64.status, LaneStatus::Fail);
        assert!(outcome.banner_fail());
        assert!(outcome.generic_x86_64.issues[0].contains("status 101"));
    }

    #[test]
    fn generic_launch_error_is_release_blocking_unsupported() {
        let outcome = run_synthetic(
            options(false, true),
            fuzz(FuzzMode::FullSanity, FuzzStatus::Pass),
            "x86_64",
            GenericRun::LaunchError {
                error: "timeout not found".to_string(),
                duration_ms: 0,
            },
        );

        assert_eq!(outcome.generic_x86_64.status, LaneStatus::Unsupported);
        assert!(outcome.banner_fail());
        assert!(outcome.generic_x86_64.issues[0].contains("could not launch"));
    }
}
