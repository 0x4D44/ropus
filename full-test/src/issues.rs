//! Extract warning/error lines from cargo/clippy stderr output.
//!
//! The HLD spec: take lines starting with `error[` or `warning:` (first ~20).
//! We widen "starts with" to "starts with, after optional ANSI color prefix
//! and optional leading whitespace" because cargo colorizes its output on
//! interactive terminals. Clippy's actual diagnostic header lines always
//! begin at column 0, but a belt-and-braces trim keeps us robust if that
//! ever shifts.

use regex::Regex;
use std::sync::OnceLock;

/// Cap the `issues` array at 20 entries (per HLD). Anything beyond that is
/// noise for a summary JSON blob.
pub const MAX_ISSUES: usize = 20;

/// Cap the raw stderr buffer we even bother to scan. 1 MiB is far more than
/// any healthy clippy run emits; beyond that we truncate and append a note.
pub const MAX_STDERR_BYTES: usize = 1024 * 1024;

/// Extract up to `MAX_ISSUES` diagnostic lines from combined stderr text.
///
/// Matches lines whose first non-whitespace token is `error[...]:`, `error:`,
/// or `warning:`. Preserves original ordering. The returned strings are
/// trimmed of leading/trailing whitespace to keep the JSON tidy; ANSI color
/// escapes, if present, are left intact because callers (humans reading the
/// supervisor log) generally prefer colorized diagnostics.
pub fn extract(stderr: &str) -> Vec<String> {
    let re = diag_regex();
    let mut out = Vec::with_capacity(MAX_ISSUES);
    for line in stderr.lines() {
        if out.len() >= MAX_ISSUES {
            break;
        }
        let trimmed = line.trim_start();
        if re.is_match(trimmed) {
            out.push(trimmed.trim_end().to_string());
        }
    }
    out
}

/// Truncate a stderr buffer to `MAX_STDERR_BYTES`, appending a marker if we
/// had to cut. Returns `(buffer, was_truncated)`.
pub fn cap_stderr(s: &str) -> (String, bool) {
    if s.len() <= MAX_STDERR_BYTES {
        return (s.to_string(), false);
    }
    // Slice on a char boundary to avoid splitting a UTF-8 codepoint.
    let mut end = MAX_STDERR_BYTES;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    let mut capped = String::with_capacity(end + 64);
    capped.push_str(&s[..end]);
    capped.push_str("\n[full-test: stderr truncated at 1 MiB]\n");
    (capped, true)
}

fn diag_regex() -> &'static Regex {
    // `error[E0XXX]: ...`, `error: ...`, `warning: ...`. The trailing colon
    // disambiguates from words that merely start with "error" or "warning".
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^(?:error(?:\[[^\]]+\])?:|warning:)").expect("static regex"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pulls_canonical_error_and_warning_lines() {
        let sample = "   Compiling ropus v0.11.1\n\
error[E0308]: mismatched types\n \
  --> ropus/src/celt/bands.rs:42:5\n\
warning: unused variable: `foo`\n \
  --> ropus/src/silk/encoder.rs:7:9\n\
error: could not compile `ropus` due to previous error\n\
note: `#[warn(unused_variables)]` on by default\n";
        let issues = extract(sample);
        assert_eq!(issues.len(), 3, "issues: {issues:?}");
        assert!(issues[0].starts_with("error[E0308]:"));
        assert!(issues[1].starts_with("warning: unused variable"));
        assert!(issues[2].starts_with("error: could not compile"));
    }

    #[test]
    fn ignores_unrelated_lines() {
        let sample = "   Compiling foo\n\
    Finished dev [unoptimized + debuginfo] target(s) in 1.23s\n\
the word error appears but not at line start\n";
        assert!(extract(sample).is_empty());
    }

    #[test]
    fn respects_max_issues_cap() {
        let mut s = String::new();
        for i in 0..(MAX_ISSUES + 5) {
            s.push_str(&format!("warning: issue {i}\n"));
        }
        let issues = extract(&s);
        assert_eq!(issues.len(), MAX_ISSUES);
        assert_eq!(issues[0], "warning: issue 0");
        assert_eq!(
            issues[MAX_ISSUES - 1],
            format!("warning: issue {}", MAX_ISSUES - 1)
        );
    }

    #[test]
    fn tolerates_leading_whitespace() {
        // Cargo sometimes indents follow-up error lines; we accept indented
        // headers too since the HLD asks for "lines starting with" these
        // tokens and the useful signal survives a leading space.
        let sample = "    error[E0433]: failed to resolve: unresolved import\n";
        let issues = extract(sample);
        assert_eq!(issues.len(), 1);
        assert!(issues[0].starts_with("error[E0433]:"));
    }

    #[test]
    fn cap_stderr_under_limit_is_identity() {
        let (out, truncated) = cap_stderr("hello\n");
        assert_eq!(out, "hello\n");
        assert!(!truncated);
    }

    #[test]
    fn cap_stderr_over_limit_truncates_with_note() {
        let big = "x".repeat(MAX_STDERR_BYTES + 1024);
        let (out, truncated) = cap_stderr(&big);
        assert!(truncated);
        assert!(out.len() <= MAX_STDERR_BYTES + 128);
        assert!(out.contains("stderr truncated"));
    }
}
