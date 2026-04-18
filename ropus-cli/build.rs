use std::process::Command;

fn main() {
    // Refresh build timestamp + git SHA whenever HEAD moves (new commit,
    // branch switch, etc.). Without this, BUILD_GIT_SHA can become stale.
    println!("cargo:rerun-if-changed=../.git/HEAD");
    println!("cargo:rerun-if-changed=build.rs");

    let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string();
    println!("cargo:rustc-env=BUILD_TIMESTAMP={now}");

    // Optional: short git SHA for the banner.
    let sha = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=BUILD_GIT_SHA={sha}");
}
