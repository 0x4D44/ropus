//! Shared main-entry helpers for typed CLI output routing and uniform
//! `anyhow`-chain error printing across the four binaries.

use std::path::Path;
use std::process::ExitCode;

use colored::*;

use crate::ui::escape_terminal_text;

/// Apply the colour override from an already-parsed CLI flag.
pub fn configure_color(no_color: bool) {
    if no_color {
        colored::control::set_override(false);
    }
}

/// Detect the standalone `--no-color` flag before Clap renders help or parse
/// errors. The scan stops at `--`, so a positional filename with that spelling
/// remains data rather than changing parser output policy.
pub fn no_color_requested() -> bool {
    no_color_in(std::env::args_os().skip(1))
}

fn no_color_in<I>(args: I) -> bool
where
    I: IntoIterator,
    I::Item: AsRef<std::ffi::OsStr>,
{
    for arg in args {
        let arg = arg.as_ref();
        if arg == "--" {
            break;
        }
        if arg == "--no-color" {
            return true;
        }
    }
    false
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
            eprintln!(
                "{} {}",
                "error:".red().bold(),
                escape_terminal_text(&e.to_string())
            );
            for cause in e.chain().skip(1) {
                eprintln!(
                    "  {} {}",
                    "caused by:".red(),
                    escape_terminal_text(&cause.to_string())
                );
            }
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_color_flag_stops_at_end_of_options() {
        assert!(!no_color_in([
            std::ffi::OsString::from("input.opus"),
            std::ffi::OsString::from("--"),
            std::ffi::OsString::from("--no-color"),
        ]));
        assert!(no_color_in([
            std::ffi::OsString::from("input.opus"),
            std::ffi::OsString::from("--no-color"),
        ]));
    }

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
