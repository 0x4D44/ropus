//! full-test — local validation runner for the ropus workspace.
//!
//! Phase 1 scaffold: CLI parsing, stage 0 setup capture, and stage 1 quality
//! checks (`cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`).
//! Later phases add tests+coverage, ambisonics roundtrip, benchmarks, and an
//! HTML report. See `wrk_docs/2026.04.19 - HLD - full-test-runner.md`.
//!
//! Phase 1 output is a single JSON blob on stdout; the HTML report belongs to
//! Phase 4.

use std::process::ExitCode;

mod cli;
mod issues;
mod quality;
mod report;
mod setup;

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

    // Phase 1 envelope: setup + quality only. Later stages are stubs.
    let exit_code: u8 = if quality_outcome.all_passed() { 0 } else { 1 };
    let envelope = report::Envelope {
        setup: &setup_info,
        quality: &quality_outcome,
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
