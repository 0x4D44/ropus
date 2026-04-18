use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Refresh build timestamp + git SHA whenever HEAD moves (new commit,
    // branch switch, etc.). Without this, BUILD_GIT_SHA can become stale.
    // Only emit the rerun hint if the path exists — when installed via
    // `cargo install --path` from outside the workspace, the build runs in
    // a temp dir without a parent `.git`, and we don't want to wedge cargo
    // on a non-existent path.
    let head = Path::new("../.git/HEAD");
    if head.exists() {
        println!("cargo:rerun-if-changed=../.git/HEAD");
    }

    let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string();
    println!("cargo:rustc-env=BUILD_TIMESTAMP={now}");

    // Optional: short git SHA for the banner. If git isn't installed, we
    // aren't in a repo, or HEAD can't be read, fall back to "unknown" so
    // `cargo install` from a packaged tarball still succeeds.
    let sha = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=BUILD_GIT_SHA={sha}");
}
