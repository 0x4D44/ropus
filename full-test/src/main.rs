//! full-test — local validation runner for the ropus workspace.
//!
//! Phase 4 scope: Stage 0 setup, Stage 1 quality, Stage 2 tests (+ optional
//! coverage), Stage 3 ambisonics roundtrip, Stage 4 native benchmark sweep,
//! Stage 5 HTML report generation. See
//! `wrk_docs/2026.04.19 - HLD - full-test-runner.md`.
//!
//! The primary artefact is the HTML report at
//! `tests/results/full_test_<YYYYMMDD_HHMMSS>.html`; the JSON envelope is kept
//! behind `--emit-json` for supervisor log plumbing.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

mod ambisonics;
mod banner;
mod bench;
mod cargo_parse;
mod cli;
mod fuzz;
mod html;
mod ietf_vectors;
mod issues;
mod llvm_cov_parse;
mod preflight;
mod quality;
mod report;
mod setup;
mod tests;

fn main() -> ExitCode {
    // Parse CLI first; --help / unknown flags must exit before we touch the
    // process priority so we don't mutate the shell's scheduler class on a
    // misspelled flag.
    let args: Vec<String> = std::env::args().skip(1).collect();
    let options = match cli::parse(&args) {
        cli::ParseOutcome::Options(o) => o,
        cli::ParseOutcome::HelpRequested => {
            cli::print_help();
            return ExitCode::from(0);
        }
        cli::ParseOutcome::Error(msg) => {
            eprintln!("error: {msg}");
            eprintln!();
            cli::print_help();
            return ExitCode::from(2);
        }
    };

    // Stage 0 — setup capture (always runs). Capture happens BEFORE we lower
    // priority so `cargo metadata` (the slowest piece of stage 0) runs at
    // normal scheduling class; priority-lowering applies from here onward and
    // propagates to the child cargo invocations in later stages.
    let setup_info = setup::capture(&options);
    setup::set_low_priority();

    // Stage 1 — quality.
    let quality_outcome = if options.skip_quality || options.quick {
        quality::Outcome::skipped()
    } else {
        quality::run()
    };

    // Stage 2 — tests (+ optional coverage).
    //
    // HLD § Flags: `--quick` skips stages 1 and 4 only; stage 2 still runs.
    // When `--quick` is combined with `--skip-coverage`, stage 2 downgrades
    // to plain `cargo test` — which is already what happens here, since the
    // `run()` call sees `skip_coverage=true`.
    let tests_outcome = tests::run(
        options.skip_coverage,
        &setup_info.ietf_vectors,
        stage2_profile(&options),
    );

    // Phase 1.2 — bounded fuzz sanity gate. Default full-test does not ask
    // for fuzz tooling. Release preflight either runs the bounded shell sanity
    // path or, under --quick, only inventories declared targets and committed
    // crash files.
    let fuzz_outcome = fuzz::run(&options);

    // HLD § Stages: if stage 2 failed to *compile* (not fails a test — fails
    // to compile), stages 3 and 4 are marked "skipped (upstream build
    // failure)" rather than attempted. They depend on the same compiled
    // artefacts and would repeat the same error with no new signal.
    let upstream_build_failed = tests_outcome.tests.build_failed;

    // Stage 3 — ambisonics roundtrip.
    let ambisonics_outcome = if options.skip_ambisonics {
        ambisonics::AmbisonicsResult::skipped("--skip-ambisonics")
    } else if upstream_build_failed {
        ambisonics::AmbisonicsResult::skipped("upstream build failure")
    } else {
        ambisonics::run()
    };

    // Stage 4 — native benchmark sweep.
    //
    // HLD § Flags: `--quick` skips stages 1 and 4.
    let bench_outcome = if options.skip_benchmarks || options.quick {
        let reason = if options.quick {
            "--quick"
        } else {
            "--skip-benchmarks"
        };
        bench::BenchResult::skipped(reason)
    } else if upstream_build_failed {
        bench::BenchResult::skipped("upstream build failure")
    } else {
        bench::run()
    };

    // Banner classification centralises the PASS/FAIL/WARN rules per
    // HLD § PASS / FAIL / WARN. The exit code is derived from the banner —
    // PASS and WARN both map to 0 so pre-commit `--quick` runs aren't
    // spuriously blocked on bench ratio noise.
    let banner_kind = banner::classify(
        &quality_outcome,
        &tests_outcome,
        &ambisonics_outcome,
        &bench_outcome,
        &fuzz_outcome,
        &setup_info.preflight,
    );
    let exit_code: u8 = banner_kind.exit_code();

    // Stage 5 — HTML report.
    let commit_subject = resolve_commit_subject();
    let timestamp = chrono::Local::now();
    let report_ctx = html::ReportContext {
        commit_sha: &setup_info.commit,
        branch: &setup_info.branch,
        version: &setup_info.version,
        commit_subject: &commit_subject,
        timestamp,
        banner: banner_kind,
        ietf_vectors: setup_info.ietf_vectors.clone(),
        ietf_vectors_present: setup_info.ietf_vectors_present,
        preflight: setup_info.preflight.clone(),
        options: &options,
        quality: &quality_outcome,
        tests: &tests_outcome,
        fuzz: &fuzz_outcome,
        ambisonics: &ambisonics_outcome,
        bench: &bench_outcome,
    };
    let html_body = html::render(&report_ctx);
    let report_path = match write_report(&html_body, timestamp) {
        Ok(p) => Some(p),
        Err(e) => {
            eprintln!("error: failed to write HTML report: {e}");
            None
        }
    };

    // Stdout one-liner — makes the terminal output self-evidently useful
    // even when the caller hasn't opened the HTML.
    print_summary_line(
        &banner_kind,
        &setup_info,
        &tests_outcome,
        report_path.as_deref(),
    );

    // `--emit-json` keeps the envelope available for supervisors that want
    // to round-trip the structured shape; elided by default (the HTML is
    // the primary artefact per HLD § Output format).
    if options.emit_json {
        let envelope = report::Envelope {
            setup: &setup_info,
            quality: &quality_outcome,
            tests: &tests_outcome,
            fuzz: &fuzz_outcome,
            ambisonics: &ambisonics_outcome,
            bench: &bench_outcome,
            exit_code,
        };
        match serde_json::to_string_pretty(&envelope.to_json()) {
            Ok(s) => println!("{s}"),
            Err(e) => {
                eprintln!("error: failed to serialize report: {e}");
                return ExitCode::from(1);
            }
        }
    }

    ExitCode::from(exit_code)
}

/// Resolve `git log -1 --format=%s` for the header metadata. Defaults to
/// `(no subject available)` outside a git checkout rather than panicking or
/// propagating a spawn error — the report must always render.
fn resolve_commit_subject() -> String {
    let out = Command::new("git")
        .args(["log", "-1", "--format=%s"])
        .output();
    match out {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if s.is_empty() {
                "(no subject available)".to_string()
            } else {
                s
            }
        }
        _ => "(no subject available)".to_string(),
    }
}

/// Write the HTML body to `tests/results/full_test_<stamp>.html`. Creates the
/// parent directory as needed. Returns the full path on success so the
/// stdout summary line can point at it.
fn write_report(body: &str, timestamp: chrono::DateTime<chrono::Local>) -> Result<PathBuf, String> {
    let root = workspace_root();
    let results_dir = root.join("tests").join("results");
    std::fs::create_dir_all(&results_dir)
        .map_err(|e| format!("creating {}: {e}", results_dir.display()))?;
    let stamp = timestamp.format("%Y%m%d_%H%M%S").to_string();
    let path = results_dir.join(format!("full_test_{stamp}.html"));
    std::fs::write(&path, body).map_err(|e| format!("writing {}: {e}", path.display()))?;
    Ok(path)
}

/// Workspace root derived from this crate's manifest dir. Mirrors the pattern
/// from `setup.rs` / `bench.rs`.
fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().map(Path::to_path_buf).unwrap_or(manifest)
}

/// Emit the single-line stdout summary. Format per HLD:
///
/// ```text
/// PASS — ropus 0.9.0 @ main/35896f2 — 1247 passed, 0 failed, 12 ignored — report: tests/results/full_test_20260419_214301.html
/// ```
fn print_summary_line(
    banner: &banner::Banner,
    setup: &setup::SetupInfo,
    tests: &tests::Outcome,
    report_path: Option<&Path>,
) {
    let passed = tests.tests.total_passed;
    let failed = tests.tests.total_failed;
    let ignored = tests.tests.total_ignored;
    let report_seg = match report_path {
        Some(p) => format!(" — report: {}", relpath(p)),
        None => String::new(),
    };
    println!(
        "{label} — ropus {version} @ {branch}/{sha} — {passed} passed, {failed} failed, {ignored} ignored{report}",
        label = banner.label(),
        version = setup.version,
        branch = setup.branch,
        sha = setup.commit,
        report = report_seg,
    );
}

/// Render a path relative to the workspace root if it lives under it, else
/// fall back to the absolute form. Keeps the stdout one-liner readable when
/// run from the workspace root (the common case).
fn relpath(p: &Path) -> String {
    let root = workspace_root();
    match p.strip_prefix(&root) {
        Ok(rel) => rel.to_string_lossy().replace('\\', "/"),
        Err(_) => p.to_string_lossy().into_owned(),
    }
}

fn stage2_profile(options: &cli::Options) -> tests::Stage2Profile {
    match preflight::PreflightPolicy::from_flags(options.quick, options.release_preflight) {
        preflight::PreflightPolicy::ReleaseCoreSmokeNoNeuralClaim => {
            tests::Stage2Profile::QuickReleasePreflight
        }
        preflight::PreflightPolicy::DefaultReportOnly
        | preflight::PreflightPolicy::ReleaseCorePlusNeuralDredGate => {
            tests::Stage2Profile::FullWorkspace
        }
    }
}

#[cfg(test)]
mod tests_unit {
    use super::*;

    fn options(quick: bool, release_preflight: bool) -> cli::Options {
        cli::Options {
            quick,
            release_preflight,
            ..cli::Options::default()
        }
    }

    #[test]
    fn quick_release_preflight_stage2_profile_requires_both_flags() {
        assert_eq!(
            stage2_profile(&options(true, true)),
            tests::Stage2Profile::QuickReleasePreflight
        );
        assert_eq!(
            stage2_profile(&options(true, false)),
            tests::Stage2Profile::FullWorkspace
        );
        assert_eq!(
            stage2_profile(&options(false, true)),
            tests::Stage2Profile::FullWorkspace
        );
        assert_eq!(
            stage2_profile(&options(false, false)),
            tests::Stage2Profile::FullWorkspace
        );
    }
}
