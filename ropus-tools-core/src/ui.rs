//! Banner, headings and small text-formatting helpers.

use std::fmt::{self, Write as _};
use std::io::{self, Write};
use std::path::Path;

use colored::*;

/// A fallible, shared sink for terminal-facing informational output.
///
/// The stream handle stays in this layer, and each line takes a short lock.
/// Callers can therefore handle a closed pipe without a formatting panic and
/// without holding a terminal lock across unrelated work.
pub struct Ui<'a> {
    target: UiTarget<'a>,
}

enum UiTarget<'a> {
    Stdout(io::Stdout),
    Stderr(io::Stderr),
    Borrowed(&'a mut dyn Write),
}

impl<'a> Ui<'a> {
    pub fn new(writer: &'a mut dyn Write) -> Self {
        Self {
            target: UiTarget::Borrowed(writer),
        }
    }

    pub fn for_output(to_stderr: bool) -> Ui<'static> {
        Ui {
            target: if to_stderr {
                UiTarget::Stderr(io::stderr())
            } else {
                UiTarget::Stdout(io::stdout())
            },
        }
    }

    pub fn line(&mut self, args: fmt::Arguments<'_>) -> io::Result<()> {
        match &mut self.target {
            UiTarget::Stdout(stream) => {
                let mut writer = stream.lock();
                write_line(&mut writer, args)
            }
            UiTarget::Stderr(stream) => {
                let mut writer = stream.lock();
                write_line(&mut writer, args)
            }
            UiTarget::Borrowed(writer) => write_line(*writer, args),
        }
    }

    pub fn flush(&mut self) -> io::Result<()> {
        match &mut self.target {
            UiTarget::Stdout(stream) => stream.lock().flush(),
            UiTarget::Stderr(stream) => stream.lock().flush(),
            UiTarget::Borrowed(writer) => writer.flush(),
        }
    }

    pub fn banner(
        &mut self,
        name: &str,
        version: &str,
        timestamp: &str,
        sha: &str,
    ) -> io::Result<()> {
        let name = name.bright_cyan().bold();
        let version = version.bright_white();
        let suffix = format!("(build {timestamp}, sha {sha})").dimmed();
        self.line(format_args!("{name} {version} {suffix}"))
    }

    pub fn heading(&mut self, text: &str) -> io::Result<()> {
        self.line(format_args!("{}", text.bright_yellow().bold()))
    }

    pub fn ok(&mut self, text: &str) -> io::Result<()> {
        self.line(format_args!("{}", text.green()))
    }
}

fn write_line(writer: &mut dyn Write, args: fmt::Arguments<'_>) -> io::Result<()> {
    writer.write_fmt(args)?;
    writer.write_all(b"\n")
}

/// Run a UI operation against a locked stdout or stderr stream.
pub fn with_locked_writer<T>(to_stderr: bool, f: impl FnOnce(&mut dyn Write) -> T) -> T {
    if to_stderr {
        let stderr = io::stderr();
        let mut writer = stderr.lock();
        f(&mut writer)
    } else {
        let stdout = io::stdout();
        let mut writer = stdout.lock();
        f(&mut writer)
    }
}

/// Write the "<name> <version> (build <ts>, sha <sha>)" banner line to a
/// caller-provided stream.
pub fn write_banner(
    writer: &mut dyn Write,
    name: &str,
    version: &str,
    timestamp: &str,
    sha: &str,
) -> io::Result<()> {
    Ui::new(writer).banner(name, version, timestamp, sha)
}

/// Print the banner to stdout, retaining the historical infallible API for
/// non-encoding callers. Closed stdout is deliberately ignored here; the
/// encoding CLI uses [`write_banner`] so it can stop before opening output.
///
/// The library is binary-agnostic, so each binary passes its own
/// `env!("CARGO_PKG_NAME")` / `CARGO_PKG_VERSION` / `BUILD_TIMESTAMP` /
/// `BUILD_GIT_SHA` values in.
pub fn print_banner(name: &str, version: &str, timestamp: &str, sha: &str) {
    let _ = with_locked_writer(false, |writer| {
        write_banner(writer, name, version, timestamp, sha)
    });
}

/// Same as [`print_banner`] but writes to stderr. Used by `ropusenc`/`ropusdec`
/// when the bitstream is piped to stdout — the banner's ANSI codes and text
/// would otherwise corrupt the byte stream downstream consumers see.
pub fn print_banner_stderr(name: &str, version: &str, timestamp: &str, sha: &str) {
    let _ = with_locked_writer(true, |writer| {
        write_banner(writer, name, version, timestamp, sha)
    });
}

pub fn heading(text: &str) {
    let _ = with_locked_writer(false, |writer| Ui::new(writer).heading(text));
}

pub fn ok(text: &str) {
    let _ = with_locked_writer(false, |writer| Ui::new(writer).ok(text));
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
                | '\u{061C}'
                | '\u{200E}'
                | '\u{200F}'
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
        let input = "ok\\line\0\x07\x1B]0;title\r\n\u{0085}\u{009B}\u{061C}\u{200E}\u{200F}\u{2028}\u{202E}31m";
        assert_eq!(
            escape_terminal_text(input),
            r"ok\\line\u{0000}\u{0007}\u{001B}]0;title\u{000D}\u{000A}\u{0085}\u{009B}\u{061C}\u{200E}\u{200F}\u{2028}\u{202E}31m"
        );
    }

    #[test]
    fn query_values_are_raw_only_when_stdout_is_not_a_tty() {
        let value = "name\n\u{001B}[31m";
        assert_eq!(format_query_value(value, true), r"name\u{000A}\u{001B}[31m");
        assert_eq!(format_query_value(value, false), value);
    }
}
