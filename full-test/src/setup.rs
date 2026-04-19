//! Stage 0 — environment capture and process-priority setup.
//!
//! Runs unconditionally before any timed stage. Not surfaced in the HTML
//! report (Phase 4), but its fields feed into the envelope and Phase 2+
//! decisions (e.g. the IETF-vectors pre-flight).

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::cli::Options;

#[derive(Debug, Clone)]
pub struct SetupInfo {
    pub commit: String,
    pub branch: String,
    pub version: String,
    pub ietf_vectors_present: bool,
    // Mirror the CLI flags so the JSON envelope can round-trip them for
    // supervisor logs. Phase 2+ consumes these directly.
    pub options_snapshot: Options,
}

/// Capture the runtime context: git commit + branch, workspace version, and
/// the IETF-vectors pre-flight.
pub fn capture(options: &Options) -> SetupInfo {
    let commit = git_short_sha().unwrap_or_else(|| "unknown".to_string());
    SetupInfo {
        branch: resolve_branch(&commit),
        commit,
        version: workspace_version().unwrap_or_else(|| "unknown".to_string()),
        ietf_vectors_present: ietf_vectors_present(),
        options_snapshot: options.clone(),
    }
}

fn git_short_sha() -> Option<String> {
    run_trimmed(Command::new("git").args(["rev-parse", "--short", "HEAD"]))
}

fn git_branch() -> Option<String> {
    run_trimmed(Command::new("git").args(["rev-parse", "--abbrev-ref", "HEAD"]))
}

/// Resolve the branch name, substituting `detached@<sha>` when `git rev-parse
/// --abbrev-ref HEAD` returns the literal `"HEAD"` (the signal for a detached
/// checkout). Falls back to `"unknown"` outside a git repo.
fn resolve_branch(commit_short: &str) -> String {
    format_branch(git_branch(), commit_short)
}

/// Pure helper used by `resolve_branch`; factored out for unit testing.
fn format_branch(raw: Option<String>, commit_short: &str) -> String {
    match raw {
        Some(b) if b == "HEAD" => format!("detached@{commit_short}"),
        Some(b) => b,
        None => "unknown".to_string(),
    }
}

fn run_trimmed(cmd: &mut Command) -> Option<String> {
    let out = cmd.output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

/// Look up the workspace version via `cargo metadata`.
///
/// `cargo metadata --no-deps` does not expose `[workspace.package]` under a
/// top-level key, so we scan `packages[]` for the `ropus` crate — that is the
/// authoritative workspace version today.
fn workspace_version() -> Option<String> {
    let out = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;

    let packages = json.get("packages")?.as_array()?;
    for pkg in packages {
        if pkg.get("name").and_then(|n| n.as_str()) == Some("ropus")
            && let Some(v) = pkg.get("version").and_then(|s| s.as_str())
        {
            return Some(v.to_string());
        }
    }
    None
}

fn ietf_vectors_present() -> bool {
    // The conformance suite runs `testvector01`..`testvector12` (each in mono
    // and stereo variants, so 24 tests from 12 `.bit` files). A partial fetch
    // that delivered only a subset would pass a single-file probe and then
    // blow up on the missing ones mid-Stage-2. Require all 12 present.
    // See HLD § Stage 0.
    let root = workspace_root().join("tests").join("vectors").join("ietf");
    all_ietf_bitstreams_present(&root)
}

/// True when every `testvectorNN.bit` (NN=01..=12) exists under `root`.
/// Factored out for unit testing against a temp dir. Pure — no I/O on `root`
/// beyond `is_file()` on each candidate.
fn all_ietf_bitstreams_present(root: &Path) -> bool {
    (1..=12).all(|n| root.join(format!("testvector{n:02}.bit")).is_file())
}

fn workspace_root() -> PathBuf {
    // `CARGO_MANIFEST_DIR` points at full-test/; go up one.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().map(Path::to_path_buf).unwrap_or(manifest)
}

/// Drop the process (and thus all child cargo invocations) to a below-normal
/// scheduling class so full-test runs happily in the background while Arthur
/// keeps working. Mirrors mddosem's pattern (see
/// `C:\language\mddosem\src\bin\full_test.rs` lines ~86-130).
pub fn set_low_priority() {
    #[cfg(unix)]
    {
        // SAFETY: `nice` is async-signal-safe and always succeeds for positive
        // increments. We intentionally ignore the return value (the new nice
        // value) since we don't care what it was clamped to.
        unsafe {
            nice(10);
        }
    }
    #[cfg(windows)]
    {
        const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x0000_4000;
        // SAFETY: `GetCurrentProcess` returns a pseudo-handle (-1) that never
        // needs to be closed. `SetPriorityClass` is documented thread-safe.
        unsafe {
            SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS);
        }
    }
}

#[cfg(unix)]
unsafe extern "C" {
    fn nice(inc: i32) -> i32;
}

#[cfg(windows)]
unsafe extern "system" {
    fn GetCurrentProcess() -> *mut core::ffi::c_void;
    fn SetPriorityClass(h_process: *mut core::ffi::c_void, dw_priority_class: u32) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_does_not_panic_and_fields_populate() {
        // We don't assert specific values — the repo's commit/branch/version
        // change under us — only that the capture path returns strings rather
        // than panicking and that the IETF probe resolves to some boolean.
        let info = capture(&Options::default());
        assert!(!info.commit.is_empty());
        assert!(!info.branch.is_empty());
        assert!(!info.version.is_empty());
        // Boolean: either true or false is fine; force a read.
        let _: bool = info.ietf_vectors_present;
    }

    #[test]
    fn workspace_root_points_at_manifest_parent() {
        let root = workspace_root();
        // The root Cargo.toml must live here by construction.
        assert!(
            root.join("Cargo.toml").is_file(),
            "root: {}",
            root.display()
        );
    }

    #[test]
    fn format_branch_detects_detached_head() {
        assert_eq!(
            format_branch(Some("HEAD".to_string()), "abc1234"),
            "detached@abc1234"
        );
    }

    #[test]
    fn format_branch_passes_through_named_branch() {
        assert_eq!(format_branch(Some("main".to_string()), "abc1234"), "main");
    }

    #[test]
    fn format_branch_falls_back_to_unknown_outside_repo() {
        assert_eq!(format_branch(None, "abc1234"), "unknown");
    }

    #[test]
    fn ietf_preflight_requires_all_twelve_bitstreams() {
        // Empty dir → false.
        let tmp = tempfile::TempDir::new().expect("temp dir");
        assert!(!all_ietf_bitstreams_present(tmp.path()));

        // Only testvectors 01..06 → still false (partial fetch scenario).
        for n in 1..=6 {
            let p = tmp.path().join(format!("testvector{n:02}.bit"));
            std::fs::write(&p, b"").expect("write stub");
        }
        assert!(!all_ietf_bitstreams_present(tmp.path()));

        // Fill in 07..12 → true.
        for n in 7..=12 {
            let p = tmp.path().join(format!("testvector{n:02}.bit"));
            std::fs::write(&p, b"").expect("write stub");
        }
        assert!(all_ietf_bitstreams_present(tmp.path()));
    }
}
