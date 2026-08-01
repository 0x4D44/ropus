//! Shared build-time provenance policy for the four command-line crates.
//!
//! A package can be built from a normal clone, a linked worktree, a detached
//! checkout, or a source archive nested inside another repository. Git is
//! useful only when its discovered top-level is the workspace that owns this
//! package; otherwise the banner must say `unknown` rather than identify the
//! consumer repository.

use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Discovery {
    pub sha: Option<String>,
    pub watch_paths: Vec<PathBuf>,
}

/// Emit Cargo environment variables and the complete set of Git files that
/// can change the selected revision. Each CLI build script is a tiny wrapper
/// around this function so the policy cannot drift between binaries.
#[allow(dead_code)]
pub fn run() {
    let manifest_dir = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR")
            .unwrap_or_else(|| PathBuf::from(".").into_os_string()),
    );
    println!("cargo:rerun-if-changed=build.rs");
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir
            .join("../ropus-tools-core/src/build_provenance.rs")
            .display()
    );

    let workspace_root = manifest_dir
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| manifest_dir.clone());
    let discovery = discover(&manifest_dir, &workspace_root);
    for path in &discovery.watch_paths {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let now = build_timestamp();
    println!("cargo:rustc-env=BUILD_TIMESTAMP={now}");
    println!(
        "cargo:rustc-env=BUILD_GIT_SHA={}",
        discovery.sha.as_deref().unwrap_or("unknown")
    );
}

/// Resolve the current repository only when its top-level matches the
/// workspace root that owns `manifest_dir`.
pub fn discover(manifest_dir: &Path, workspace_root: &Path) -> Discovery {
    let Some(discovered_top) =
        git_value(manifest_dir, &["rev-parse", "--show-toplevel"]).map(PathBuf::from)
    else {
        return Discovery::default();
    };
    let Some(discovered_top) = canonical_path(&discovered_top) else {
        return Discovery::default();
    };
    let Some(workspace_root) = canonical_path(workspace_root) else {
        return Discovery::default();
    };
    if discovered_top != workspace_root {
        return Discovery::default();
    }

    let Some(head_path) = git_path(manifest_dir, "HEAD") else {
        return Discovery::default();
    };
    let mut watch_paths = vec![head_path.clone()];
    if let Some(ref_name) = git_value(manifest_dir, &["symbolic-ref", "--quiet", "HEAD"])
        && ref_name.starts_with("refs/")
        && let Some(ref_path) = git_path(manifest_dir, &ref_name)
    {
        push_unique(&mut watch_paths, ref_path);
    }
    // A repository may keep the branch target in packed-refs instead of a
    // loose refs/heads file. Watching it costs nothing and covers that case.
    if let Some(packed_refs) = git_path(manifest_dir, "packed-refs") {
        push_unique(&mut watch_paths, packed_refs);
    }

    let sha = git_value(manifest_dir, &["rev-parse", "--verify", "--short", "HEAD"])
        .filter(|value| !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_hexdigit()));
    Discovery { sha, watch_paths }
}

fn push_unique(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.iter().any(|existing| existing == &path) {
        paths.push(path);
    }
}

fn canonical_path(path: &Path) -> Option<PathBuf> {
    std::fs::canonicalize(path).ok()
}

fn git_path(manifest_dir: &Path, name: &str) -> Option<PathBuf> {
    let path = PathBuf::from(git_value(manifest_dir, &["rev-parse", "--git-path", name])?);
    if path.is_absolute() {
        Some(path)
    } else {
        Some(manifest_dir.join(path))
    }
}

fn git_value(manifest_dir: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .current_dir(manifest_dir)
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?.trim().to_string();
    (!value.is_empty()).then_some(value)
}

#[cfg(not(test))]
fn build_timestamp() -> String {
    chrono::Utc::now()
        .format("%Y-%m-%d %H:%M:%S UTC")
        .to_string()
}

#[cfg(test)]
#[allow(dead_code)]
fn build_timestamp() -> String {
    "test timestamp".to_string()
}
