//! Banner, headings and small text-formatting helpers.

use colored::*;

pub(crate) fn print_banner() {
    let name = env!("CARGO_PKG_NAME").bright_cyan().bold();
    let version = env!("CARGO_PKG_VERSION").bright_white();
    let timestamp = env!("BUILD_TIMESTAMP");
    let sha = env!("BUILD_GIT_SHA");
    let suffix = format!("(build {timestamp}, sha {sha})").dimmed();
    println!("{name} {version} {suffix}");
}

pub(crate) fn heading(text: &str) {
    println!("{}", text.bright_yellow().bold());
}

pub(crate) fn ok(text: &str) {
    println!("{}", text.green());
}

/// Format an integer with thousands separators using ASCII commas.
pub(crate) fn format_num(n: u64) -> String {
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
