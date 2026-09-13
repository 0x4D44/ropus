//! Shared main-entry helpers for typed CLI output routing and uniform
//! `anyhow`-chain error printing across the four binaries.

use std::ffi::OsString;
use std::path::Path;
use std::process::ExitCode;

use colored::*;

use crate::ui::escape_terminal_text;

/// Parse a command's real argv without allowing Clap to print unescaped input.
///
/// The command factory is evaluated once for the normal parse and again only
/// when that parse fails. The retry receives an escaped copy of every argv
/// item, so Clap constructs its normal styled error from safe context values.
/// Successful parses retain the original OsStrings, including values after
/// the end-of-options marker. The callbacks are kept in the macro expansion
/// so this crate does not need to depend on the CLI-only Clap crate.
#[macro_export]
macro_rules! try_get_matches {
    ($command_factory:expr) => {{
        let argv = ::std::env::args_os().collect::<::std::vec::Vec<_>>();
        match ($command_factory)().try_get_matches_from(argv.clone()) {
            Ok(matches) => matches,
            Err(error) => {
                let use_stderr = error.use_stderr();
                let exit_code = error.exit_code();
                let safe_argv = $crate::prelude::escape_terminal_argv(argv);

                match ($command_factory)().try_get_matches_from(safe_argv) {
                    Err(safe_error)
                        if safe_error.kind() == error.kind()
                            && safe_error.use_stderr() == use_stderr
                            && safe_error.exit_code() == exit_code =>
                    {
                        let _ = safe_error.print();
                    }
                    Err(safe_error) => {
                        // A changed error kind, stream, or status means the
                        // retry no longer proves equivalent parser behavior.
                        // Escape the safe retry's complete rendering and keep
                        // the original destination and exit code.
                        $crate::prelude::print_escaped_cli_error(&safe_error, use_stderr);
                    }
                    Ok(_) => {
                        // Escaping should not change whether argv is valid,
                        // but retain a fail-closed path if a parser's custom
                        // semantics make the retry diverge.
                        $crate::prelude::print_escaped_cli_error(&error, use_stderr);
                    }
                }
                ::std::process::exit(exit_code);
            }
        }
    }};
}

/// Escape argv items for a Clap error-only retry.
pub fn escape_terminal_argv(argv: Vec<OsString>) -> Vec<OsString> {
    argv.into_iter()
        .map(|arg| OsString::from(escape_terminal_text(&arg.to_string_lossy())))
        .collect()
}

/// Fail-closed fallback for a parser whose escaped retry unexpectedly succeeds.
///
/// This path may lose Clap's layout and styling, but it cannot emit a raw
/// terminal control from the original error. Normal errors use Clap's own
/// formatter on the escaped retry above.
pub fn print_escaped_cli_error(error: &impl std::fmt::Display, use_stderr: bool) {
    use std::io::Write as _;

    let text = escape_terminal_text(&error.to_string());
    if use_stderr {
        let mut stream = std::io::stderr().lock();
        let _ = stream.write_all(text.as_bytes());
        let _ = stream.flush();
    } else {
        let mut stream = std::io::stdout().lock();
        let _ = stream.write_all(text.as_bytes());
        let _ = stream.flush();
    }
}

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
    fn escaped_argv_covers_controls_and_preserves_parser_tokens() {
        let argv = vec![
            OsString::from("ropus\u{001B}]0;argv0\u{0007}\u{061C}\u{200E}\u{200F}"),
            OsString::from("--"),
            OsString::from("--bad\u{0000}\u{0007}\u{001B}]0;title\u{0085}\u{009B}\r\n\u{202E}"),
        ];

        assert_eq!(
            escape_terminal_argv(argv),
            vec![
                OsString::from(r"ropus\u{001B}]0;argv0\u{0007}\u{061C}\u{200E}\u{200F}"),
                OsString::from("--"),
                OsString::from(
                    r"--bad\u{0000}\u{0007}\u{001B}]0;title\u{0085}\u{009B}\u{000D}\u{000A}\u{202E}"
                ),
            ]
        );
    }

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
