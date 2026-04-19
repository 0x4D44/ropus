//! Parse `cargo test` (and `cargo llvm-cov ... test ...`) stdout/stderr.
//!
//! Cargo's test harness output has three moving parts:
//!
//! 1. **Per-binary headers** on stderr:
//!    `Running unittests src/lib.rs (target/debug/deps/ropus-abcdef.exe)` or
//!    `Running tests\integration-0123.exe`. We group test outcomes under the
//!    most-recently-seen header.
//!
//! 2. **Per-test outcome lines** on stdout:
//!    `test <fully::qualified::name> ... ok|FAILED|ignored`.
//!    Multi-line failure bodies (`---- ... stdout ----`, panic traces, etc.)
//!    sit below the `test result:` line. We ignore them: attributing per-line
//!    output to specific tests is fragile and out of scope for the summary
//!    envelope — "this test failed" is sufficient; the HTML report can link
//!    to the raw log.
//!
//! 3. **Per-binary result summary** on stdout:
//!    `test result: ok. N passed; N failed; N ignored; N measured; N filtered out; finished in T.Ts`.
//!    Provides authoritative totals per binary; we prefer these over hand
//!    counting outcome lines (a harness panic can produce outcome lines
//!    without a matching summary, and vice-versa).
//!
//! Build failures are detected on stderr via `error[E<digits>]:` or the
//! literal `could not compile` / `aborting due to`. When either fires we set
//! `build_failed=true` so the supervisor can mark downstream stages as
//! "skipped (upstream build failure)" per the HLD.

use std::sync::OnceLock;

use regex::Regex;
use serde::Serialize;

/// Outcome of a single test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    Pass,
    Fail,
    Ignored,
}

/// Flat per-test record. The `binary` label is the most-recently-seen cargo
/// header (stem of the binary path) — useful for the HTML module breakdown.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TestOutcome {
    pub fqn: String,
    pub outcome: Outcome,
    pub binary: String,
}

/// Aggregated per-binary totals. Pulled from the `test result:` summary line.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct BinaryResult {
    pub name: String,
    pub passed: u32,
    pub failed: u32,
    pub ignored: u32,
    pub duration_ms: u64,
}

/// Stage 2 parse result. Serialisable so `report::Envelope` can embed it.
#[derive(Debug, Clone, Default, Serialize)]
pub struct TestsResult {
    pub skipped: bool,
    pub skip_reason: Option<String>,
    pub build_failed: bool,
    /// Stage 2 wall-clock, populated by the dispatcher (not the parser).
    pub duration_ms: u64,
    pub binaries: Vec<BinaryResult>,
    pub per_test: Vec<TestOutcome>,
    pub total_passed: u32,
    pub total_failed: u32,
    pub total_ignored: u32,
    pub failed_test_names: Vec<String>,
    pub ignored_test_names: Vec<String>,
    /// Human-readable explanation of why `coverage` is `None`, when it is.
    /// Distinct from the overall `skip_reason`: this records *why the
    /// coverage sub-report is unavailable* (file missing, empty, malformed,
    /// or `--skip-coverage`). Phase 4's HTML surfaces this so "no coverage"
    /// never silently looks like "0/0".
    pub coverage_skip_reason: Option<String>,
}

/// Parse combined cargo output. `stdout` carries per-test lines + summaries;
/// `stderr` carries `Running …` headers, compile progress, and build errors.
///
/// The parser is tolerant: malformed or missing lines degrade to empty data
/// rather than panicking. The only hard error signal is `build_failed=true`.
/// ANSI colour escapes are stripped before regex matching: cargo/libtest
/// suppress colour under `Stdio::piped()` on most platforms, but some
/// Windows terminal emulators leak TTY semantics into pipes, so we handle
/// it defensively rather than relying on absence.
pub fn parse(stdout: &str, stderr: &str) -> TestsResult {
    let stdout = strip_ansi(stdout);
    let stderr = strip_ansi(stderr);
    let (stdout, stderr) = (stdout.as_str(), stderr.as_str());

    let mut result = TestsResult {
        build_failed: detect_build_failure(stderr),
        ..TestsResult::default()
    };

    // Derive per-binary labels from stderr `Running …` headers, in order.
    let binary_labels = parse_running_headers(stderr);
    let mut binary_idx: usize = 0;
    let mut current_binary: String = binary_labels
        .first()
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());

    let test_line_re = test_line_regex();
    let result_line_re = result_line_regex();

    for line in stdout.lines() {
        // When we see a `test result:` summary, we take that as the end of
        // the current binary's output. Attribute it, then advance the label
        // pointer so the next outcome block picks up the next header. This
        // keeps binary-attribution correct even if a stray header line made
        // it into stdout somehow.
        if let Some(summary) = parse_result_line(result_line_re, line) {
            let BinarySummary {
                passed,
                failed,
                ignored,
                duration_ms,
            } = summary;
            result.binaries.push(BinaryResult {
                name: current_binary.clone(),
                passed,
                failed,
                ignored,
                duration_ms,
            });
            result.total_passed += passed;
            result.total_failed += failed;
            result.total_ignored += ignored;
            binary_idx += 1;
            current_binary = binary_labels
                .get(binary_idx)
                .cloned()
                .unwrap_or_else(|| "unknown".to_string());
            continue;
        }

        if let Some(caps) = test_line_re.captures(line) {
            let fqn = caps.get(1).map(|m| m.as_str()).unwrap_or_default();
            let tail = caps.get(2).map(|m| m.as_str()).unwrap_or_default();
            // Skip benchmark-harness output (`bench: 1,234 ns/iter (+/- 56)`)
            // entirely: not a pass, not a fail, not an ignore. Stage 2 runs
            // `--lib --tests` so benches don't run today, but a future
            // cargo flag tweak shouldn't silently inflate the pass count.
            if tail.trim_start().starts_with("bench:") {
                continue;
            }
            let outcome = classify_outcome(tail);
            let name = fqn.to_string();
            match outcome {
                Outcome::Fail => result.failed_test_names.push(name.clone()),
                Outcome::Ignored => result.ignored_test_names.push(name.clone()),
                Outcome::Pass => {}
            }
            result.per_test.push(TestOutcome {
                fqn: name,
                outcome,
                binary: current_binary.clone(),
            });
        }
    }

    result
}

fn classify_outcome(tail: &str) -> Outcome {
    // `tail` is the post-`...` remainder: "ok", "FAILED", "ignored", and very
    // occasionally "ignored, <reason>" for `#[ignore = "..."]`. Be generous.
    let normalised = tail.trim().to_ascii_lowercase();
    if normalised.starts_with("failed") {
        Outcome::Fail
    } else if normalised.starts_with("ignored") {
        Outcome::Ignored
    } else {
        // Default to Pass for "ok" and anything else we don't recognise.
        // Unknown tails are rare (custom harnesses?) and counting them as
        // pass matches what the `test result:` summary will say — the
        // authoritative totals come from that summary anyway.
        Outcome::Pass
    }
}

struct BinarySummary {
    passed: u32,
    failed: u32,
    ignored: u32,
    duration_ms: u64,
}

fn parse_result_line(re: &Regex, line: &str) -> Option<BinarySummary> {
    let caps = re.captures(line.trim_start())?;
    let passed = caps.name("passed")?.as_str().parse().ok()?;
    let failed = caps.name("failed")?.as_str().parse().ok()?;
    let ignored = caps.name("ignored")?.as_str().parse().ok()?;
    // Duration is optional: older cargo versions may omit the trailing
    // "finished in T.Ts" fragment. Default to 0 if absent or unparseable.
    let duration_ms = caps
        .name("secs")
        .and_then(|m| m.as_str().parse::<f64>().ok())
        .map(|s| (s * 1000.0) as u64)
        .unwrap_or(0);
    Some(BinarySummary {
        passed,
        failed,
        ignored,
        duration_ms,
    })
}

fn parse_running_headers(stderr: &str) -> Vec<String> {
    // Cargo emits:
    //     Running unittests src/lib.rs (target/debug/deps/ropus-abcdef.exe)
    //     Running tests\integration-0123.exe
    //     Running tests/integration-0123
    // We extract the binary stem (no extension, no hash trailer) because
    // the hash varies between builds and makes the per-binary key unstable
    // across runs.
    let re = running_header_regex();
    let mut out = Vec::new();
    for line in stderr.lines() {
        if let Some(caps) = re.captures(line.trim_start()) {
            // Group 1 = parenthesised real path (unittests form);
            // group 2 = bare path (tests form). Whichever matched wins.
            let path = caps
                .get(1)
                .or_else(|| caps.get(2))
                .map(|m| m.as_str())
                .unwrap_or_default();
            out.push(binary_stem(path));
        }
    }
    out
}

/// Strip directory, extension, and the 16-hex-char cargo hash suffix from
/// `target/debug/deps/ropus-abc123def456.exe` → `ropus`.
fn binary_stem(path: &str) -> String {
    let base = path.rsplit(['/', '\\']).next().unwrap_or(path);
    let stem = base.strip_suffix(".exe").unwrap_or(base);
    // Hash suffix is exactly 16 hex chars after a trailing dash.
    let hash_re = hash_suffix_regex();
    hash_re.replace(stem, "").into_owned()
}

/// Remove ANSI CSI escape sequences (colour, cursor, etc.) from a string.
///
/// Splits on `ESC` (0x1B); for the first chunk we keep everything, and for
/// each subsequent chunk we discard up to and including the first `m`
/// terminator (the common case for colour codes like `\x1b[32m`). Chunks
/// that don't contain an `m` are kept verbatim — an incomplete escape is
/// unusual, and the parser downstream can cope with spurious bytes far
/// better than with silent drops.
fn strip_ansi(s: &str) -> String {
    if !s.contains('\x1b') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut parts = s.split('\x1b');
    if let Some(head) = parts.next() {
        out.push_str(head);
    }
    for chunk in parts {
        match chunk.split_once('m') {
            Some((_esc, rest)) => out.push_str(rest),
            None => out.push_str(chunk),
        }
    }
    out
}

fn detect_build_failure(stderr: &str) -> bool {
    // Two independent signals; either fires. `error[E<digits>]:` is a hard
    // rustc diagnostic; "could not compile" is cargo's own summary line;
    // "aborting due to" shows up under rustc's own summary. Keep all three
    // so a borderline log doesn't silently pass us.
    let re = build_fail_regex();
    stderr.lines().any(|line| re.is_match(line.trim_start()))
}

// ------------------------------------------------------------------------
// Regex cache
// ------------------------------------------------------------------------

fn test_line_regex() -> &'static Regex {
    // `test <name> ... <tail>` — the stable cargo/libtest format since ~2015.
    // We require whitespace+three-dots+whitespace to avoid matching "test
    // result: ok..." lines (which have no "..." before the outcome).
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^test\s+(\S+)\s+\.\.\.\s+(.+)$").expect("static regex"))
}

fn result_line_regex() -> &'static Regex {
    // `test result: ok. N passed; N failed; N ignored; N measured; N filtered out; finished in T.Ts`
    // The "finished in" fragment is optional (see comment in parse_result_line).
    // `measured` and `filtered out` counts are captured but ignored — we
    // don't expose either in the envelope today.
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"^test result:\s+\S+\.\s+(?P<passed>\d+)\s+passed;\s+(?P<failed>\d+)\s+failed;\s+(?P<ignored>\d+)\s+ignored;\s+\d+\s+measured;\s+\d+\s+filtered out(?:;\s+finished in\s+(?P<secs>[0-9]+(?:\.[0-9]+)?)s)?",
        )
        .expect("static regex")
    })
}

fn running_header_regex() -> &'static Regex {
    // Match either `Running tests/path` or `Running unittests path (real/path.exe)`.
    // We capture the parenthesised path when present (it is the actual
    // binary), falling back to the bare path.
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^Running\s+(?:(?:unittests|tests)\s+)?(?:\S+?\s+\((\S+)\)|(\S+))")
            .expect("static regex")
    })
}

fn hash_suffix_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"-[0-9a-f]{16}$").expect("static regex"))
}

fn build_fail_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^(?:error\[E\d+\]:|error: could not compile|error: aborting due to)")
            .expect("static regex")
    })
}

// ------------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const HAPPY_STDOUT: &str = "\n\
running 3 tests\n\
test silk::decoder::tests::test_alpha ... ok\n\
test silk::decoder::tests::test_beta ... ok\n\
test silk::decoder::tests::test_gamma ... ok\n\
\n\
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.42s\n\
\n\
running 2 tests\n\
test celt::bands::tests::test_delta ... ok\n\
test celt::bands::tests::test_epsilon ... ok\n\
\n\
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s\n";

    const HAPPY_STDERR: &str = "\
   Compiling ropus v0.11.1\n\
    Finished test [unoptimized + debuginfo] target(s) in 4.12s\n\
     Running unittests src/lib.rs (target/debug/deps/ropus-0123456789abcdef.exe)\n\
     Running tests\\integration-fedcba9876543210.exe\n";

    #[test]
    fn happy_path_counts_roll_up() {
        let r = parse(HAPPY_STDOUT, HAPPY_STDERR);
        assert!(!r.build_failed);
        assert_eq!(r.total_passed, 5);
        assert_eq!(r.total_failed, 0);
        assert_eq!(r.total_ignored, 0);
        assert_eq!(r.binaries.len(), 2);
        assert_eq!(r.per_test.len(), 5);
        assert!(r.failed_test_names.is_empty());
        assert!(r.ignored_test_names.is_empty());
    }

    #[test]
    fn binary_labels_strip_path_and_hash_suffix() {
        let r = parse(HAPPY_STDOUT, HAPPY_STDERR);
        assert_eq!(r.binaries[0].name, "ropus", "binaries: {:?}", r.binaries);
        assert_eq!(r.binaries[1].name, "integration");
    }

    #[test]
    fn duration_in_result_line_parsed() {
        let r = parse(HAPPY_STDOUT, HAPPY_STDERR);
        assert_eq!(r.binaries[0].duration_ms, 420);
        assert_eq!(r.binaries[1].duration_ms, 1000);
    }

    #[test]
    fn per_test_binary_attribution_follows_headers() {
        let r = parse(HAPPY_STDOUT, HAPPY_STDERR);
        let ropus_tests: Vec<_> = r
            .per_test
            .iter()
            .filter(|t| t.binary == "ropus")
            .map(|t| t.fqn.as_str())
            .collect();
        assert_eq!(
            ropus_tests,
            vec![
                "silk::decoder::tests::test_alpha",
                "silk::decoder::tests::test_beta",
                "silk::decoder::tests::test_gamma",
            ]
        );
    }

    #[test]
    fn build_failure_sets_flag() {
        let stdout = "";
        let stderr = "\
   Compiling ropus v0.11.1\n\
error[E0432]: unresolved import `foo::bar`\n \
  --> ropus/src/celt/encoder.rs:10:5\n\
error: could not compile `ropus` (lib) due to 1 previous error\n";
        let r = parse(stdout, stderr);
        assert!(r.build_failed);
        assert_eq!(r.total_passed, 0);
        assert_eq!(r.total_failed, 0);
    }

    #[test]
    fn mixed_pass_fail_captures_failed_names() {
        let stdout = "\
running 3 tests\n\
test mod_a::tests::fails_loudly ... FAILED\n\
test mod_a::tests::passes ... ok\n\
test mod_a::tests::also_fails ... FAILED\n\
\n\
failures:\n\
\n\
---- mod_a::tests::fails_loudly stdout ----\n\
thread 'mod_a::tests::fails_loudly' panicked at 'boom', src/lib.rs:3:5\n\
\n\
failures:\n\
    mod_a::tests::fails_loudly\n\
    mod_a::tests::also_fails\n\
\n\
test result: FAILED. 1 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.10s\n";
        let stderr = "     Running tests\\suite-aaaaaaaaaaaaaaaa.exe\n";
        let r = parse(stdout, stderr);
        assert!(!r.build_failed);
        assert_eq!(r.total_passed, 1);
        assert_eq!(r.total_failed, 2);
        assert_eq!(r.total_ignored, 0);
        assert_eq!(r.failed_test_names.len(), 2);
        assert!(
            r.failed_test_names
                .contains(&"mod_a::tests::fails_loudly".to_string())
        );
        assert!(
            r.failed_test_names
                .contains(&"mod_a::tests::also_fails".to_string())
        );
    }

    #[test]
    fn ignored_tests_tracked_in_list() {
        let stdout = "\
running 2 tests\n\
test slow::tests::expensive ... ignored\n\
test slow::tests::annotated ... ignored, slow on CI\n\
\n\
test result: ok. 0 passed; 0 failed; 2 ignored; 0 measured; 0 filtered out; finished in 0.00s\n";
        let stderr = "     Running tests\\slow-0000000000000000.exe\n";
        let r = parse(stdout, stderr);
        assert_eq!(r.total_ignored, 2);
        assert_eq!(
            r.ignored_test_names,
            vec![
                "slow::tests::expensive".to_string(),
                "slow::tests::annotated".to_string(),
            ]
        );
    }

    #[test]
    fn outcome_classifier_handles_spacing_and_case() {
        assert_eq!(classify_outcome("ok"), Outcome::Pass);
        assert_eq!(classify_outcome("  ok  "), Outcome::Pass);
        assert_eq!(classify_outcome("FAILED"), Outcome::Fail);
        assert_eq!(classify_outcome("failed"), Outcome::Fail);
        assert_eq!(classify_outcome("ignored"), Outcome::Ignored);
        assert_eq!(classify_outcome("ignored, reason"), Outcome::Ignored);
    }

    #[test]
    fn binary_stem_handles_windows_and_unix_paths() {
        assert_eq!(
            binary_stem("target/debug/deps/ropus-0123456789abcdef.exe"),
            "ropus"
        );
        assert_eq!(
            binary_stem("target\\debug\\deps\\ropus-0123456789abcdef.exe"),
            "ropus"
        );
        assert_eq!(
            binary_stem("target/debug/deps/ropus-0123456789abcdef"),
            "ropus"
        );
        // No hash → preserved.
        assert_eq!(binary_stem("tests/integration.exe"), "integration");
    }

    #[test]
    fn malformed_result_line_does_not_panic() {
        // Missing "filtered out" counter — degrades gracefully.
        let stdout = "\
running 1 test\n\
test a::b ... ok\n\
test result: ok. 1 passed;\n";
        let stderr = "     Running tests\\a-0000000000000000.exe\n";
        let r = parse(stdout, stderr);
        // Outcome line still collected; summary line drops on the floor.
        assert_eq!(r.per_test.len(), 1);
        assert_eq!(r.total_passed, 0, "summary line should have been ignored");
    }

    #[test]
    fn missing_running_headers_tolerated() {
        // No `Running …` lines — every test attributed to "unknown".
        let stdout = "\
running 1 test\n\
test a::b ... ok\n\
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n";
        let r = parse(stdout, "");
        assert_eq!(r.per_test.len(), 1);
        assert_eq!(r.per_test[0].binary, "unknown");
        assert_eq!(r.total_passed, 1);
    }

    #[test]
    fn ansi_coloured_test_lines_parse_cleanly() {
        // Simulate cargo leaking colour through a pipe (observed on some
        // Windows terminals). `\x1b[32mok\x1b[0m` etc. must not break the
        // regex match; strip_ansi handles it before parse sees the lines.
        let stdout = "\x1b[0m\n\
running 2 tests\n\
test \x1b[32mmod_a::tests::a\x1b[0m ... \x1b[32mok\x1b[0m\n\
test \x1b[32mmod_a::tests::b\x1b[0m ... \x1b[31mFAILED\x1b[0m\n\
\n\
test result: \x1b[31mFAILED\x1b[0m. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.50s\n";
        let stderr = "     Running tests\\suite-0000000000000000.exe\n";
        let r = parse(stdout, stderr);
        assert_eq!(r.total_passed, 1);
        assert_eq!(r.total_failed, 1);
        assert_eq!(r.per_test.len(), 2);
        assert_eq!(r.failed_test_names, vec!["mod_a::tests::b".to_string()]);
    }

    #[test]
    fn bench_output_lines_do_not_inflate_pass_count() {
        // libtest's bench format: `test name ... bench: 1,234 ns/iter (+/- 56)`.
        // These must neither count as a pass nor as a failure; they're
        // outside Stage 2's scope (we run --lib --tests, not --benches).
        let stdout = "\
running 2 tests\n\
test mod_a::tests::passes ... ok\n\
test mod_a::tests::perf_probe ... bench:       1,234 ns/iter (+/- 56)\n\
\n\
test result: ok. 1 passed; 0 failed; 0 ignored; 1 measured; 0 filtered out; finished in 0.01s\n";
        let stderr = "     Running tests\\suite-0000000000000000.exe\n";
        let r = parse(stdout, stderr);
        // Summary-line totals drive the per-binary counts; the bench line
        // is measured=1, which we don't count. Crucially, the bench name
        // must not appear in per_test.
        assert_eq!(r.total_passed, 1);
        assert_eq!(r.total_failed, 0);
        assert_eq!(r.per_test.len(), 1);
        assert_eq!(r.per_test[0].fqn, "mod_a::tests::passes");
    }

    #[test]
    fn unittests_src_lib_header_recognised() {
        // Confirm the "Running unittests src/lib.rs (path/to/exe)" form's
        // inner parenthesised path wins over the bare path.
        let stderr =
            "     Running unittests src/lib.rs (target/debug/deps/ropus-0123456789abcdef.exe)\n";
        let labels = parse_running_headers(stderr);
        assert_eq!(labels, vec!["ropus".to_string()]);
    }
}
