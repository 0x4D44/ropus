//! Banner, headings and small text-formatting helpers.

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
