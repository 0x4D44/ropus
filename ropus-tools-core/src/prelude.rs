//! Shared main-entry helpers: argv sniffing for `-q/--quiet` and `--no-color`
//! before clap runs, and `run(Result<()>) -> ExitCode` for uniform
//! `anyhow`-chain error printing across the four binaries.

use std::env;
use std::process::ExitCode;

use colored::*;

/// Flags sniffed from argv before clap runs.
pub struct PreludeFlags {
    pub quiet: bool,
    pub no_color: bool,
}

/// Inspect raw argv for `-q`/`--quiet` and `--no-color` and apply the color
/// override if requested. Returns the sniffed flags so the caller can skip
/// the banner.
///
/// Runs before clap so the flags still take effect when the user passes
/// `--help` or `--version` (clap exits before our main body would otherwise
/// see them).
pub fn run_prelude() -> PreludeFlags {
    let raw: Vec<String> = env::args().collect();
    let quiet = raw.iter().any(|a| a == "-q" || a == "--quiet");
    let no_color = raw.iter().any(|a| a == "--no-color");

    if no_color {
        colored::control::set_override(false);
    }

    PreludeFlags { quiet, no_color }
}

/// Turn a command's `anyhow::Result<()>` into a process exit code, printing the
/// full error chain to stderr on failure.
pub fn run(result: anyhow::Result<()>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{} {}", "error:".red().bold(), e);
            for cause in e.chain().skip(1) {
                eprintln!("  {} {}", "caused by:".red(), cause);
            }
            ExitCode::FAILURE
        }
    }
}
