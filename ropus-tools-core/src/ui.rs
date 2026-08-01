//! Banner, headings and small text-formatting helpers.

use std::fmt::Write as _;
use std::path::Path;

use colored::*;

/// Print the "<name> <version> (build <ts>, sha <sha>)" banner line to
/// stdout.
///
/// The library is binary-agnostic, so each binary passes its own
/// `env!("CARGO_PKG_NAME")` / `CARGO_PKG_VERSION` / `BUILD_TIMESTAMP` /
/// `BUILD_GIT_SHA` values in.
pub fn print_banner(name: &str, version: &str, timestamp: &str, sha: &str) {
    let name = name.bright_cyan().bold();
    let version = version.bright_white();
    let suffix = format!("(build {timestamp}, sha {sha})").dimmed();
    println!("{name} {version} {suffix}");
}

/// Same as [`print_banner`] but writes to stderr. Used by `ropusenc`/`ropusdec`
/// when the bitstream is piped to stdout — the banner's ANSI codes and text
/// would otherwise corrupt the byte stream downstream consumers see.
pub fn print_banner_stderr(name: &str, version: &str, timestamp: &str, sha: &str) {
    let name = name.bright_cyan().bold();
    let version = version.bright_white();
    let suffix = format!("(build {timestamp}, sha {sha})").dimmed();
    eprintln!("{name} {version} {suffix}");
}

pub fn heading(text: &str) {
    println!("{}", text.bright_yellow().bold());
}

pub fn ok(text: &str) {
    println!("{}", text.green());
}

/// Escape untrusted text for a terminal-facing single line.
///
/// C0/C1 controls (including ESC, OSC bytes, BEL, CR, and LF), Unicode line
/// separators, and bidi overrides are rendered as `\\u{NNNN}` sequences.
/// Backslashes are escaped too, making the encoding reversible rather than
/// ambiguous with a literal escape sequence. Printable Unicode remains
/// unchanged.
pub fn escape_terminal_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            c if is_terminal_control(c) => {
                write!(out, "\\u{{{code:04X}}}", code = c as u32)
                    .expect("writing to a String cannot fail");
            }
            c => out.push(c),
        }
    }
    out
}

fn is_terminal_control(c: char) -> bool {
    c.is_control()
        || matches!(
            c,
            '\u{2028}'
                | '\u{2029}'
                | '\u{202A}'..='\u{202E}'
                | '\u{2066}'..='\u{2069}'
        )
}

/// Escape a path after lossy conversion of non-UTF-8 platform bytes.
pub fn escape_terminal_path(path: &Path) -> String {
    escape_terminal_text(&path.to_string_lossy())
}

/// Format a machine-query value according to the explicit output policy:
/// preserve bytes for redirected/piped stdout, but escape controls when a
/// human is viewing the query directly in a terminal.
pub fn format_query_value(value: &str, stdout_is_tty: bool) -> String {
    if stdout_is_tty {
        escape_terminal_text(value)
    } else {
        value.to_owned()
    }
}

/// Format an integer with thousands separators using ASCII commas.
pub fn format_num(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(bytes.len() + bytes.len() / 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_terminal_text_covers_c0_c1_and_backslash() {
        let input = "ok\\line\0\x07\x1B]0;title\r\n\u{0085}\u{009B}\u{2028}\u{202E}31m";
        assert_eq!(
            escape_terminal_text(input),
            r"ok\\line\u{0000}\u{0007}\u{001B}]0;title\u{000D}\u{000A}\u{0085}\u{009B}\u{2028}\u{202E}31m"
        );
    }

    #[test]
    fn query_values_are_raw_only_when_stdout_is_not_a_tty() {
        let value = "name\n\u{001B}[31m";
        assert_eq!(format_query_value(value, true), r"name\u{000A}\u{001B}[31m");
        assert_eq!(format_query_value(value, false), value);
    }
}
