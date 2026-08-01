//! Shared main-entry helpers for typed CLI output routing and uniform
//! `anyhow`-chain error printing across the four binaries.

use std::path::Path;
use std::process::ExitCode;

use colored::*;

/// Apply the colour override from an already-parsed CLI flag.
pub fn configure_color(no_color: bool) {
    if no_color {
        colored::control::set_override(false);
    }
}

/// Decide whether a typed input/output pair routes binary data to stdout.
///
/// An explicit output wins. Without one, stdin input (`-`) implies stdout
/// because no filename can be derived from a pipe. Call this only after the
/// owning CLI parser has resolved option arity and end-of-options semantics.
pub fn output_is_stdout(input: &Path, output: Option<&Path>) -> bool {
    output.map_or(input.as_os_str() == "-", |path| path.as_os_str() == "-")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_paths_select_stdout_without_reparsing_argv() {
        assert!(output_is_stdout(Path::new("-"), None));
        assert!(output_is_stdout(
            Path::new("input.wav"),
            Some(Path::new("-"))
        ));
        assert!(!output_is_stdout(
            Path::new("-"),
            Some(Path::new("output.opus"))
        ));
        assert!(!output_is_stdout(Path::new("input.wav"), None));
    }
}
