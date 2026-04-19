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
    /// Argv points to stdout as the bitstream sink — either explicit
    /// `-o -` / `--output -` or implicit (input `-` with no `-o`). Banner
    /// must go to stderr in that case so it doesn't mix with the bytes.
    /// Approximate — the authoritative decision still happens inside the
    /// command, but this sniff is enough to steer the banner correctly.
    pub output_is_stdout: bool,
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
    let flags = scan_argv(&raw);

    if flags.no_color {
        colored::control::set_override(false);
    }

    flags
}

/// Pure argv scan used by `run_prelude` and exercised directly from tests.
/// Kept separate from the `colored` override side-effect so tests can drive
/// it with synthetic argv without mutating process-global colour state.
pub fn scan_argv(raw: &[String]) -> PreludeFlags {
    let quiet = raw.iter().any(|a| a == "-q" || a == "--quiet");
    let no_color = raw.iter().any(|a| a == "--no-color");

    // Argv-level sentinel sniff for stdout sink. Patterns accepted:
    //   • explicit:  `-o -`, `-o-`, `-o=-`, `--output -`, `--output=-`
    //   • implicit:  input is `-` and no explicit `-o` anywhere
    // clap accepts all five short+long spellings above; missing any of them
    // routes the banner to stdout alongside the Ogg bytes and corrupts the
    // stream. Positional-input detection uses "first non-flag argument"
    // because each tool's input is the only positional. Doesn't need to be
    // perfect — the command function reruns the same check authoritatively.
    let mut explicit_stdout = false;
    let mut has_explicit_output = false;
    let mut input_is_stdin = false;
    let mut iter = raw.iter().skip(1).peekable();
    let mut seen_positional = false;
    while let Some(a) = iter.next() {
        if a == "-o" || a == "--output" {
            // Bare `-o` / `--output` with next arg as value. Must be checked
            // before the `-o<value>` branch because `strip_prefix("-o")` on
            // bare `-o` would return Some("") and we'd misclassify.
            has_explicit_output = true;
            if let Some(next) = iter.peek()
                && next.as_str() == "-"
            {
                explicit_stdout = true;
            }
        } else if let Some(rest) = a.strip_prefix("--output=") {
            has_explicit_output = true;
            if rest == "-" {
                explicit_stdout = true;
            }
        } else if let Some(rest) = a.strip_prefix("-o") {
            // `-o<value>` short-form with attached value: covers `-o-`
            // (rest == "-"), `-o=-` (rest == "=-"), and `-oout.opus`
            // (rest == "out.opus"). An `=` immediately after `-o` is clap's
            // short-with-equals spelling; strip it so both `-o-` and `-o=-`
            // collapse to the same value check.
            //
            // Reachable only when `a` starts with `-o` but is neither bare
            // `-o` (handled above) nor `--output...` (starts with `--`, so
            // the `-o` prefix check fails). Empty `rest` is impossible here
            // because the bare-`-o` branch caught it.
            has_explicit_output = true;
            let value = rest.strip_prefix('=').unwrap_or(rest);
            if value == "-" {
                explicit_stdout = true;
            }
        } else if !a.starts_with('-') || a == "-" {
            // First non-flag argument is the positional input. clap's
            // conventions allow `-` as a positional so we permit it here.
            if !seen_positional {
                seen_positional = true;
                if a == "-" {
                    input_is_stdin = true;
                }
            }
        }
    }
    let output_is_stdout = explicit_stdout || (input_is_stdin && !has_explicit_output);

    PreludeFlags { quiet, no_color, output_is_stdout }
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

    fn argv(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn output_is_stdout_detects_o_space_dash() {
        let flags = scan_argv(&argv(&["ropusenc", "in.wav", "-o", "-"]));
        assert!(flags.output_is_stdout);
    }

    #[test]
    fn output_is_stdout_detects_o_attached_dash() {
        // Regression: clap accepts `-o-` (short flag with value glued on).
        // Sniff must fire or the banner corrupts the Ogg stream on stdout.
        let flags = scan_argv(&argv(&["ropusenc", "in.wav", "-o-"]));
        assert!(flags.output_is_stdout);
    }

    #[test]
    fn output_is_stdout_detects_o_equals_dash() {
        // Regression: clap accepts `-o=-` too.
        let flags = scan_argv(&argv(&["ropusenc", "in.wav", "-o=-"]));
        assert!(flags.output_is_stdout);
    }

    #[test]
    fn output_is_stdout_detects_output_space_dash() {
        let flags = scan_argv(&argv(&["ropusenc", "in.wav", "--output", "-"]));
        assert!(flags.output_is_stdout);
    }

    #[test]
    fn output_is_stdout_detects_output_equals_dash() {
        let flags = scan_argv(&argv(&["ropusenc", "in.wav", "--output=-"]));
        assert!(flags.output_is_stdout);
    }

    #[test]
    fn output_is_stdout_false_for_file_output() {
        let flags = scan_argv(&argv(&["ropusenc", "in.wav", "-o", "out.opus"]));
        assert!(!flags.output_is_stdout);
    }

    #[test]
    fn output_is_stdout_false_for_file_output_attached() {
        // `-oout.opus` is clap's short-form with attached value. Must not be
        // mistaken for stdout: value is `out.opus`, not `-`.
        let flags = scan_argv(&argv(&["ropusenc", "in.wav", "-oout.opus"]));
        assert!(!flags.output_is_stdout);
    }
}
