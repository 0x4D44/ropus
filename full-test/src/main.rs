//! full-test — local validation runner for the ropus workspace.
//!
//! Phase 2 scope: Stage 0 setup, Stage 1 quality, Stage 2 tests (+ optional
//! coverage). Stages 3 (ambisonics), 4 (benchmarks), and 5 (HTML report) are
//! later phases. See `wrk_docs/2026.04.19 - HLD - full-test-runner.md`.
//!
//! Phase 2 output is a structured JSON blob on stdout; the HTML report belongs
//! to Phase 4.

use std::process::ExitCode;

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

    // Stages 3 (ambisonics), 4 (benchmarks), 5 (HTML report) land in later
    // phases. Their dispatch TODOs live here so the structure stays visible.
    // TODO(phase 3): if !options.skip_ambisonics && !options.quick { stage 3 }
    // TODO(phase 4): if !options.skip_benchmarks && !options.quick { stage 4 }
    // TODO(phase 4): Stage 5 HTML report generation.

    // Phase 2 envelope: setup + quality + tests. Overall pass means every
    // enabled stage passed; skipped stages are treated as green.
    let exit_code: u8 = if quality_outcome.all_passed() && tests_outcome.all_passed() {
        0
    } else {
        1
    };
    let envelope = report::Envelope {
        setup: &setup_info,
        quality: &quality_outcome,
        tests: &tests_outcome,
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
