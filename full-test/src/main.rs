//! full-test — local validation runner for the ropus workspace.
//!
//! Phase 3 scope: Stage 0 setup, Stage 1 quality, Stage 2 tests (+ optional
//! coverage), Stage 3 ambisonics roundtrip, Stage 4 native benchmark sweep.
//! Stage 5 (HTML report) is Phase 4. See
//! `wrk_docs/2026.04.19 - HLD - full-test-runner.md`.
//!
//! Phase 3 output remains a structured JSON blob on stdout; the HTML report
//! belongs to Phase 4.

use std::process::ExitCode;

mod ambisonics;
mod bench;
mod cargo_parse;
mod cli;
mod issues;
mod llvm_cov_parse;
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
    let tests_outcome = tests::run(options.skip_coverage, setup_info.ietf_vectors_present);

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

    // Stage 5 (HTML report) lands in Phase 4. The TODO stays here so the
    // dispatch structure remains visible.
    // TODO(phase 4): Stage 5 HTML report generation.

    // Phase 3 envelope: setup + quality + tests + ambisonics + bench. Overall
    // pass means every enabled stage passed; skipped stages are treated as
    // green. Bench anomalies are WARNs (handled in Phase 4), never FAILs.
    let exit_code: u8 = if quality_outcome.all_passed()
        && tests_outcome.all_passed()
        && ambisonics_outcome.all_passed()
        && bench_outcome.all_passed()
    {
        0
    } else {
        1
    };
    let envelope = report::Envelope {
        setup: &setup_info,
        quality: &quality_outcome,
        tests: &tests_outcome,
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
    ExitCode::from(exit_code)
}
