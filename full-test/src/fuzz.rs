//! Phase 1.2 fuzz sanity gate.
//!
//! The shell runner owns cargo-fuzz command details. This module owns
//! full-test policy: default runs do not require fuzz tooling, non-quick
//! release preflight runs the bounded sanity command, and quick release
//! preflight only inventories manifest targets plus committed crash files.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use serde::Serialize;

use crate::cli::Options;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    NotRequested,
    InventoryOnly,
    FullSanity,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::NotRequested => "not_requested",
            Mode::InventoryOnly => "inventory_only",
            Mode::FullSanity => "full_sanity",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    NotRequested,
    Pass,
    Warn,
    Fail,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::NotRequested => "not_requested",
            Status::Pass => "pass",
            Status::Warn => "warn",
            Status::Fail => "fail",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TargetSummary {
    pub name: String,
    pub build: String,
    pub crashes: usize,
    pub replay: String,
}

impl TargetSummary {
    fn inventory(name: String, crashes: usize) -> Self {
        Self {
            name,
            build: "not_checked".to_string(),
            crashes,
            replay: if crashes == 0 {
                "skip".to_string()
            } else {
                "not_checked".to_string()
            },
        }
    }

    #[cfg(test)]
    fn passed(&self) -> bool {
        self.build != "fail" && self.replay != "fail"
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Outcome {
    pub mode: Mode,
    pub status: Status,
    pub duration_ms: u64,
    pub command: Vec<String>,
    pub targets: Vec<TargetSummary>,
    pub issues: Vec<String>,
    pub stdout: String,
    pub stderr: String,
}

impl Outcome {
    pub fn not_requested() -> Self {
        Self {
            mode: Mode::NotRequested,
            status: Status::NotRequested,
            duration_ms: 0,
            command: Vec::new(),
            targets: Vec::new(),
            issues: Vec::new(),
            stdout: String::new(),
            stderr: String::new(),
        }
    }

    pub fn banner_fail(&self) -> bool {
        self.status == Status::Fail
    }

    pub fn banner_warn(&self) -> bool {
        self.status == Status::Warn
    }
}

pub fn run(options: &Options) -> Outcome {
    let root = workspace_root();
    run_with_root(options, &root)
}

fn run_with_root(options: &Options, root: &Path) -> Outcome {
    match selected_mode(options) {
        Mode::NotRequested => Outcome::not_requested(),
        Mode::InventoryOnly => inventory_only(root),
        Mode::FullSanity => run_full_sanity(root),
    }
}

fn selected_mode(options: &Options) -> Mode {
    if !options.release_preflight {
        Mode::NotRequested
    } else if options.quick {
        Mode::InventoryOnly
    } else {
        Mode::FullSanity
    }
}

fn run_full_sanity(root: &Path) -> Outcome {
    let command = sanity_command();
    let started = Instant::now();
    let output = Command::new(&command[0])
        .args(&command[1..])
        .current_dir(root)
        .output();
    let duration_ms = started.elapsed().as_millis() as u64;

    match output {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
            let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
            let mut issues = Vec::new();
            if !output.status.success() {
                issues.push(match output.status.code() {
                    Some(code) => format!("fuzz sanity command exited with status {code}"),
                    None => "fuzz sanity command terminated by signal".to_string(),
                });
            }
            Outcome {
                mode: Mode::FullSanity,
                status: if output.status.success() {
                    Status::Pass
                } else {
                    Status::Fail
                },
                duration_ms,
                command,
                targets: parse_summary_lines(&stdout),
                issues,
                stdout,
                stderr,
            }
        }
        Err(e) => Outcome {
            mode: Mode::FullSanity,
            status: Status::Fail,
            duration_ms,
            command,
            targets: Vec::new(),
            issues: vec![format!("failed to launch fuzz sanity command: {e}")],
            stdout: String::new(),
            stderr: String::new(),
        },
    }
}

fn inventory_only(root: &Path) -> Outcome {
    let started = Instant::now();
    let manifest = root.join("tests").join("fuzz").join("Cargo.toml");
    let crashes_dir = root.join("tests").join("fuzz").join("crashes");
    let mut issues = Vec::new();
    let mut targets = Vec::new();
    let mut inventory_failed = false;

    let declared = match discover_targets(&manifest) {
        Ok(t) if !t.is_empty() => t,
        Ok(_) => {
            inventory_failed = true;
            issues.push(format!(
                "no fuzz targets declared in {}",
                display_path(&manifest)
            ));
            Vec::new()
        }
        Err(e) => {
            inventory_failed = true;
            issues.push(e);
            Vec::new()
        }
    };

    if !declared.is_empty() {
        let declared_set: BTreeSet<&str> = declared.iter().map(String::as_str).collect();
        for target in &declared {
            targets.push(TargetSummary::inventory(
                target.clone(),
                crash_count_for(&crashes_dir.join(target)),
            ));
        }
        let undeclared = undeclared_crash_bins(&crashes_dir, &declared_set);
        inventory_failed |= !undeclared.is_empty();
        issues.extend(undeclared);
    }

    let cargo_fuzz_available = cargo_fuzz_on_path();
    let nightly_ok = nightly_available();
    let status = if inventory_failed {
        Status::Fail
    } else if !cargo_fuzz_available || !nightly_ok {
        if !cargo_fuzz_available {
            issues.push(
                "cargo-fuzz not checked by quick preflight: cargo-fuzz not found on PATH"
                    .to_string(),
            );
        }
        if !nightly_ok {
            issues.push(
                "nightly not checked by quick preflight: cargo +nightly is unavailable".to_string(),
            );
        }
        Status::Warn
    } else {
        Status::Pass
    };

    Outcome {
        mode: Mode::InventoryOnly,
        status,
        duration_ms: started.elapsed().as_millis() as u64,
        command: Vec::new(),
        targets,
        issues,
        stdout: String::new(),
        stderr: String::new(),
    }
}

pub fn sanity_command() -> Vec<String> {
    vec![
        "timeout".to_string(),
        "300".to_string(),
        "bash".to_string(),
        "tools/fuzz_run.sh".to_string(),
        "--sanity".to_string(),
    ]
}

fn parse_summary_lines(stdout: &str) -> Vec<TargetSummary> {
    stdout
        .lines()
        .filter_map(parse_summary_line)
        .collect::<Vec<_>>()
}

fn parse_summary_line(line: &str) -> Option<TargetSummary> {
    let mut parts = line.split_whitespace();
    let name = parts.next()?;
    if !name.starts_with("fuzz_") {
        return None;
    }
    let mut build = None;
    let mut crashes = None;
    let mut replay = None;
    for part in parts {
        if let Some(value) = part.strip_prefix("build=") {
            build = Some(value.to_string());
        } else if let Some(value) = part.strip_prefix("crashes=") {
            crashes = value.parse::<usize>().ok();
        } else if let Some(value) = part.strip_prefix("replay=") {
            replay = Some(value.to_string());
        }
    }
    Some(TargetSummary {
        name: name.to_string(),
        build: build?,
        crashes: crashes?,
        replay: replay?,
    })
}

fn discover_targets(manifest: &Path) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(manifest)
        .map_err(|e| format!("failed to read {}: {e}", display_path(manifest)))?;
    let mut targets = Vec::new();
    let mut in_bin = false;
    let mut seen_name = false;
    for raw in text.lines() {
        let line = raw.trim();
        if is_bin_header(line) {
            if in_bin && !seen_name {
                return Err(format!(
                    "missing name in [[bin]] entry in {}",
                    display_path(manifest)
                ));
            }
            in_bin = true;
            seen_name = false;
            continue;
        }
        if line.starts_with("[[bin]]") {
            return Err(format!(
                "malformed [[bin]] header in {}: {raw}",
                display_path(manifest)
            ));
        }
        if line.starts_with('[') {
            if in_bin && !seen_name {
                return Err(format!(
                    "missing name in [[bin]] entry in {}",
                    display_path(manifest)
                ));
            }
            in_bin = false;
            seen_name = false;
            continue;
        }
        if in_bin {
            let Some(value) = target_name_value(line) else {
                continue;
            };
            if !(value.starts_with('"')
                && value.ends_with('"')
                && value.len() >= 3
                && !value[1..value.len() - 1].contains('"'))
            {
                return Err(format!(
                    "malformed fuzz target name in {}",
                    display_path(manifest)
                ));
            }
            targets.push(value[1..value.len() - 1].to_string());
            seen_name = true;
        }
    }
    if in_bin && !seen_name {
        return Err(format!(
            "missing name in [[bin]] entry in {}",
            display_path(manifest)
        ));
    }
    Ok(targets)
}

fn is_bin_header(line: &str) -> bool {
    let before_comment = line.split('#').next().unwrap_or(line).trim_end();
    before_comment == "[[bin]]"
}

fn target_name_value(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("name")?;
    let rest = rest.trim_start();
    let rest = rest.strip_prefix('=')?.trim_start();
    Some(rest.split('#').next().unwrap_or(rest).trim_end())
}

fn undeclared_crash_bins(crashes_dir: &Path, declared: &BTreeSet<&str>) -> Vec<String> {
    let mut issues = Vec::new();
    let Ok(entries) = std::fs::read_dir(crashes_dir) else {
        return issues;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if declared.contains(name) {
            continue;
        }
        if crash_count_for(&path) > 0 {
            issues.push(format!(
                "undeclared fuzz crash corpus contains committed .bin files: {}",
                display_path(&path)
            ));
        }
    }
    issues.sort();
    issues
}

fn crash_count_for(dir: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut count = 0;
    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        let path = entry.path();
        if file_type.is_dir() {
            count += crash_count_for(&path);
        } else if file_type.is_file() && path.extension().is_some_and(|ext| ext == "bin") {
            count += 1;
        }
    }
    count
}

fn cargo_fuzz_on_path() -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|dir| executable_exists(dir.join("cargo-fuzz")))
}

fn nightly_available() -> bool {
    Command::new("cargo")
        .arg("+nightly")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success())
}

fn executable_exists(path: PathBuf) -> bool {
    if path.is_file() {
        return true;
    }
    #[cfg(windows)]
    {
        path.with_extension("exe").is_file()
    }
    #[cfg(not(windows))]
    {
        false
    }
}

fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().map(Path::to_path_buf).unwrap_or(manifest)
}

fn display_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().expect("parent")).expect("create parent");
        std::fs::write(path, text).expect("write test file");
    }

    fn manifest(root: &Path, body: &str) {
        write(&root.join("tests/fuzz/Cargo.toml"), body);
    }

    fn options(quick: bool, release_preflight: bool) -> Options {
        Options {
            quick,
            release_preflight,
            ..Options::default()
        }
    }

    #[test]
    fn policy_selects_expected_mode() {
        assert_eq!(selected_mode(&options(false, false)), Mode::NotRequested);
        assert_eq!(selected_mode(&options(true, false)), Mode::NotRequested);
        assert_eq!(selected_mode(&options(false, true)), Mode::FullSanity);
        assert_eq!(selected_mode(&options(true, true)), Mode::InventoryOnly);
    }

    #[test]
    fn full_sanity_command_is_bounded_and_named() {
        assert_eq!(
            sanity_command(),
            vec!["timeout", "300", "bash", "tools/fuzz_run.sh", "--sanity"]
        );
    }

    #[test]
    fn summary_line_parser_accepts_shell_shape() {
        let row = parse_summary_line("fuzz_decode build=pass crashes=4 replay=pass")
            .expect("summary row");
        assert_eq!(row.name, "fuzz_decode");
        assert_eq!(row.build, "pass");
        assert_eq!(row.crashes, 4);
        assert_eq!(row.replay, "pass");
        assert!(row.passed());
    }

    #[test]
    fn default_run_is_not_requested() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let outcome = run_with_root(&Options::default(), tmp.path());
        assert_eq!(outcome.mode, Mode::NotRequested);
        assert_eq!(outcome.status, Status::NotRequested);
        assert!(!outcome.banner_fail());
    }

    #[test]
    fn inventory_counts_declared_crashes_and_allows_empty_dirs() {
        let tmp = tempfile::tempdir().expect("tempdir");
        manifest(
            tmp.path(),
            r#"
[[bin]]
name = "fuzz_decode"
path = "fuzz_targets/fuzz_decode.rs"

[[bin]]
name = "fuzz_encode"
path = "fuzz_targets/fuzz_encode.rs"
"#,
        );
        write(
            &tmp.path()
                .join("tests/fuzz/crashes/fuzz_decode/regression.bin"),
            "boom",
        );
        write(
            &tmp.path().join("tests/fuzz/crashes/fuzz_encode/.gitkeep"),
            "",
        );

        let outcome = inventory_only(tmp.path());
        assert_ne!(outcome.status, Status::Fail);
        assert_eq!(outcome.targets.len(), 2);
        assert_eq!(outcome.targets[0].crashes, 1);
        assert_eq!(outcome.targets[0].replay, "not_checked");
        assert_eq!(outcome.targets[1].crashes, 0);
        assert_eq!(outcome.targets[1].replay, "skip");
    }

    #[test]
    fn inventory_counts_declared_crashes_recursively() {
        let tmp = tempfile::tempdir().expect("tempdir");
        manifest(
            tmp.path(),
            r#"
[[bin]]
name = "fuzz_decode"
path = "fuzz_targets/fuzz_decode.rs"
"#,
        );
        write(
            &tmp.path()
                .join("tests/fuzz/crashes/fuzz_decode/nested/regression.bin"),
            "boom",
        );

        let outcome = inventory_only(tmp.path());
        assert_ne!(outcome.status, Status::Fail);
        assert_eq!(outcome.targets.len(), 1);
        assert_eq!(outcome.targets[0].crashes, 1);
    }

    #[test]
    fn inventory_fails_undeclared_bin_crash_files() {
        let tmp = tempfile::tempdir().expect("tempdir");
        manifest(
            tmp.path(),
            r#"
[[bin]]
name = "fuzz_decode"
path = "fuzz_targets/fuzz_decode.rs"
"#,
        );
        write(
            &tmp.path()
                .join("tests/fuzz/crashes/fuzz_removed/regression.bin"),
            "boom",
        );

        let outcome = inventory_only(tmp.path());
        assert_eq!(outcome.status, Status::Fail);
        assert!(
            outcome
                .issues
                .iter()
                .any(|issue| issue.contains("undeclared fuzz crash corpus"))
        );
    }

    #[test]
    fn inventory_fails_nested_undeclared_bin_crash_files() {
        let tmp = tempfile::tempdir().expect("tempdir");
        manifest(
            tmp.path(),
            r#"
[[bin]]
name = "fuzz_decode"
path = "fuzz_targets/fuzz_decode.rs"
"#,
        );
        write(
            &tmp.path()
                .join("tests/fuzz/crashes/fuzz_removed/nested/regression.bin"),
            "boom",
        );

        let outcome = inventory_only(tmp.path());
        assert_eq!(outcome.status, Status::Fail);
        assert!(
            outcome
                .issues
                .iter()
                .any(|issue| issue.contains("undeclared fuzz crash corpus"))
        );
    }

    #[test]
    fn inventory_ignores_undeclared_non_bin_placeholders() {
        let tmp = tempfile::tempdir().expect("tempdir");
        manifest(
            tmp.path(),
            r#"
[[bin]]
name = "fuzz_decode"
path = "fuzz_targets/fuzz_decode.rs"
"#,
        );
        write(
            &tmp.path().join("tests/fuzz/crashes/fuzz_removed/.gitkeep"),
            "",
        );

        let outcome = inventory_only(tmp.path());
        assert_ne!(outcome.status, Status::Fail);
        assert!(
            !outcome
                .issues
                .iter()
                .any(|issue| issue.contains("undeclared fuzz crash corpus"))
        );
    }

    #[test]
    fn malformed_bin_header_is_fail() {
        let tmp = tempfile::tempdir().expect("tempdir");
        manifest(
            tmp.path(),
            r#"
[[bin]]bad
name = "fuzz_decode"
path = "fuzz_targets/fuzz_decode.rs"
"#,
        );

        let outcome = inventory_only(tmp.path());
        assert_eq!(outcome.status, Status::Fail);
        assert!(
            outcome
                .issues
                .iter()
                .any(|issue| issue.contains("malformed [[bin]] header"))
        );
    }

    #[test]
    fn name_suffix_is_not_accepted_as_target_name() {
        let tmp = tempfile::tempdir().expect("tempdir");
        manifest(
            tmp.path(),
            r#"
[[bin]]
name_suffix = "fuzz_decode"
path = "fuzz_targets/fuzz_decode.rs"
"#,
        );

        let outcome = inventory_only(tmp.path());
        assert_eq!(outcome.status, Status::Fail);
        assert!(
            outcome
                .issues
                .iter()
                .any(|issue| issue.contains("missing name"))
        );
    }

    #[test]
    fn malformed_manifest_is_fail() {
        let tmp = tempfile::tempdir().expect("tempdir");
        manifest(
            tmp.path(),
            r#"
[[bin]]
path = "fuzz_targets/fuzz_decode.rs"
"#,
        );

        let outcome = inventory_only(tmp.path());
        assert_eq!(outcome.status, Status::Fail);
        assert!(
            outcome
                .issues
                .iter()
                .any(|issue| issue.contains("missing name"))
        );
    }
}
