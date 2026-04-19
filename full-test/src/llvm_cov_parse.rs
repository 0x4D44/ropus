//! Parse `cargo llvm-cov --json` output.
//!
//! The JSON is llvm-cov's `export -format=text` shape, wrapped by cargo:
//!
//! ```text
//! {
//!   "type": "llvm.coverage.json.export",
//!   "version": "...",
//!   "data": [
//!     {
//!       "files": [
//!         {
//!           "filename": "...",
//!           "summary": {
//!             "lines":     { "count": N, "covered": N, "percent": F },
//!             "functions": { "count": N, "covered": N, "percent": F },
//!             "branches":  { "count": N, "covered": N, "percent": F },
//!             "regions":   { "count": N, "covered": N, "percent": F }
//!           }
//!         },
//!         ...
//!       ],
//!       "totals": { "lines": {...}, "functions": {...}, ... }
//!     }
//!   ]
//! }
//! ```
//!
//! We compute **two rollups** from `data[0].files[]`. Paths are normalised
//! to forward slashes first (via `normalise_path`), so every predicate below
//! can match a single canonical form:
//!
//! - **ropus headline**: files whose normalised path contains `ropus/src/`.
//!   This is the codec proper.
//! - **Workspace total**: everything under the workspace, excluding known
//!   non-codec paths (`reference/`, `tests/vectors/`, `target/`, and the
//!   `full-test/` runner crate itself to avoid self-measurement pollution).
//!   The C reference source is bundled under `reference/` and would skew
//!   the denominator if counted.
//!
//! We deliberately re-derive both rollups from `files[]` rather than read the
//! `data[0].totals` block — the latter is a workspace-wide total with no way
//! to surface the ropus subset separately.

use serde::Serialize;

/// Coverage counters for a single rollup.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct CoverageMetrics {
    pub lines_covered: u64,
    pub lines_total: u64,
    pub functions_covered: u64,
    pub functions_total: u64,
    pub branches_covered: u64,
    pub branches_total: u64,
    pub regions_covered: u64,
    pub regions_total: u64,
}

impl CoverageMetrics {
    fn accumulate(&mut self, s: &FileSummary) {
        self.lines_covered += s.lines.covered;
        self.lines_total += s.lines.count;
        self.functions_covered += s.functions.covered;
        self.functions_total += s.functions.count;
        self.branches_covered += s.branches.covered;
        self.branches_total += s.branches.count;
        self.regions_covered += s.regions.covered;
        self.regions_total += s.regions.count;
    }
}

/// Aggregate coverage report — the headline + footnote rollups the HLD asks
/// for. Either or both may be zero-filled if the JSON is missing, malformed,
/// or filters out every file.
#[derive(Debug, Clone, Default, Serialize)]
pub struct CoverageResult {
    pub ropus: CoverageMetrics,
    pub workspace: CoverageMetrics,
}

/// Parse a raw llvm-cov JSON blob into the two rollups.
///
/// Returns an empty `CoverageResult` (everything zero) if the JSON is
/// unparseable, has no `data` array, or fails the shape we expect. The
/// supervisor will see zero totals and surface "coverage unavailable" in the
/// HTML report — a no-panic contract is more valuable than hard-erroring
/// here because Stage 2 already captured the test results.
pub fn parse(json_text: &str) -> CoverageResult {
    let v: serde_json::Value = match serde_json::from_str(json_text) {
        Ok(v) => v,
        Err(_) => return CoverageResult::default(),
    };
    let files = match v.get("data").and_then(|d| d.as_array()).and_then(|arr| {
        arr.first()
            .and_then(|d0| d0.get("files"))
            .and_then(|f| f.as_array())
    }) {
        Some(f) => f,
        None => return CoverageResult::default(),
    };

    let mut out = CoverageResult::default();
    for file in files {
        let filename = match file.get("filename").and_then(|s| s.as_str()) {
            Some(s) => s,
            None => continue,
        };
        let summary = match extract_file_summary(file) {
            Some(s) => s,
            None => continue,
        };

        let normalised = normalise_path(filename);

        if is_workspace_path(&normalised) {
            out.workspace.accumulate(&summary);
            if is_ropus_src(&normalised) {
                out.ropus.accumulate(&summary);
            }
        }
    }
    out
}

/// Canonicalise to forward slashes for predicate matching. Keeps both path
/// flavours in a single filter without duplicating every check.
fn normalise_path(p: &str) -> String {
    p.replace('\\', "/")
}

/// True when the file lives under `ropus/src/`. Input is already normalised
/// to forward slashes (see `normalise_path`). The `/ropus/src/`-with-
/// leading-slash match is the general case; we also accept a path that
/// *begins* with `ropus/src/` for relative filenames.
fn is_ropus_src(path_fwd: &str) -> bool {
    path_fwd.contains("/ropus/src/") || path_fwd.starts_with("ropus/src/")
}

/// True when the file should count toward the workspace rollup: rejects the
/// C reference sources, the test vector fixtures, any build artefact that
/// slipped into llvm-cov's output, and the `full-test/` crate itself. The
/// `full-test` exclusion is deliberate — running `full-test` via
/// `cargo llvm-cov test --workspace` instruments the runner's own code, which
/// would pollute the denominator with code that measures coverage rather
/// than code that implements the codec. Keeping the allow-list shaped as a
/// deny-list of non-codec paths means new crates are captured automatically.
fn is_workspace_path(path_fwd: &str) -> bool {
    // Exclude non-Rust reference sources, fixture paths, build artefacts, and
    // the full-test runner crate itself (self-measurement).
    const EXCLUDE_SEGMENTS: &[&str] =
        &["/reference/", "/tests/vectors/", "/target/", "/full-test/"];
    if EXCLUDE_SEGMENTS.iter().any(|seg| path_fwd.contains(seg)) {
        return false;
    }
    // Also reject absolute paths that start with any excluded segment (no
    // leading slash on the match side).
    const EXCLUDE_PREFIXES: &[&str] = &["reference/", "tests/vectors/", "target/", "full-test/"];
    if EXCLUDE_PREFIXES.iter().any(|pre| path_fwd.starts_with(pre)) {
        return false;
    }
    true
}

struct FileSummary {
    lines: Counter,
    functions: Counter,
    branches: Counter,
    regions: Counter,
}

#[derive(Default, Clone, Copy)]
struct Counter {
    count: u64,
    covered: u64,
}

fn extract_file_summary(file: &serde_json::Value) -> Option<FileSummary> {
    let summary = file.get("summary")?;
    Some(FileSummary {
        lines: extract_counter(summary.get("lines")),
        functions: extract_counter(summary.get("functions")),
        branches: extract_counter(summary.get("branches")),
        regions: extract_counter(summary.get("regions")),
    })
}

fn extract_counter(obj: Option<&serde_json::Value>) -> Counter {
    let obj = match obj {
        Some(o) => o,
        None => return Counter::default(),
    };
    let count = obj.get("count").and_then(|v| v.as_u64()).unwrap_or(0);
    let covered = obj.get("covered").and_then(|v| v.as_u64()).unwrap_or(0);
    Counter { count, covered }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-built fixture mimicking cargo-llvm-cov's export shape. We use
    /// three files to cover the three filter states: ropus src (headline +
    /// workspace), non-ropus workspace crate (workspace only), and reference
    /// path (excluded entirely).
    const FIXTURE_JSON: &str = r#"{
      "type": "llvm.coverage.json.export",
      "version": "2.0.1",
      "data": [{
        "files": [
          {
            "filename": "C:\\work\\mdopus\\ropus\\src\\silk\\decoder.rs",
            "summary": {
              "lines":     { "count": 100, "covered": 80, "percent": 80.0 },
              "functions": { "count":  20, "covered": 18, "percent": 90.0 },
              "branches":  { "count":  50, "covered": 40, "percent": 80.0 },
              "regions":   { "count": 200, "covered": 160, "percent": 80.0 }
            }
          },
          {
            "filename": "C:\\work\\mdopus\\ropus-tools-core\\src\\lib.rs",
            "summary": {
              "lines":     { "count": 50, "covered": 25, "percent": 50.0 },
              "functions": { "count": 10, "covered":  5, "percent": 50.0 },
              "branches":  { "count": 20, "covered": 10, "percent": 50.0 },
              "regions":   { "count": 80, "covered": 40, "percent": 50.0 }
            }
          },
          {
            "filename": "C:\\work\\mdopus\\reference\\silk\\decoder.c",
            "summary": {
              "lines":     { "count": 999, "covered": 999, "percent": 100.0 },
              "functions": { "count": 999, "covered": 999, "percent": 100.0 },
              "branches":  { "count": 999, "covered": 999, "percent": 100.0 },
              "regions":   { "count": 999, "covered": 999, "percent": 100.0 }
            }
          }
        ],
        "totals": {
          "lines":     { "count": 1149, "covered": 1104, "percent": 96.08 },
          "functions": { "count": 1029, "covered": 1022, "percent": 99.32 },
          "branches":  { "count": 1069, "covered": 1049, "percent": 98.13 },
          "regions":   { "count": 1279, "covered": 1199, "percent": 93.75 }
        }
      }]
    }"#;

    #[test]
    fn ropus_rollup_filters_to_ropus_src_only() {
        let r = parse(FIXTURE_JSON);
        // Only the ropus\src\silk\decoder.rs file counts here.
        assert_eq!(r.ropus.lines_total, 100);
        assert_eq!(r.ropus.lines_covered, 80);
        assert_eq!(r.ropus.functions_total, 20);
        assert_eq!(r.ropus.functions_covered, 18);
        assert_eq!(r.ropus.branches_total, 50);
        assert_eq!(r.ropus.branches_covered, 40);
        assert_eq!(r.ropus.regions_total, 200);
        assert_eq!(r.ropus.regions_covered, 160);
    }

    #[test]
    fn workspace_rollup_sums_rust_crates_excluding_reference() {
        let r = parse(FIXTURE_JSON);
        // ropus + ropus-tools-core; reference\decoder.c excluded.
        assert_eq!(r.workspace.lines_total, 150);
        assert_eq!(r.workspace.lines_covered, 105);
        assert_eq!(r.workspace.functions_total, 30);
        assert_eq!(r.workspace.functions_covered, 23);
        assert_eq!(r.workspace.branches_total, 70);
        assert_eq!(r.workspace.branches_covered, 50);
        assert_eq!(r.workspace.regions_total, 280);
        assert_eq!(r.workspace.regions_covered, 200);
    }

    #[test]
    fn malformed_json_returns_empty_result() {
        let r = parse("not json at all");
        assert_eq!(r.ropus, CoverageMetrics::default());
        assert_eq!(r.workspace, CoverageMetrics::default());
    }

    #[test]
    fn missing_data_array_returns_empty_result() {
        let r = parse(r#"{"type": "llvm.coverage.json.export"}"#);
        assert_eq!(r.ropus, CoverageMetrics::default());
        assert_eq!(r.workspace, CoverageMetrics::default());
    }

    #[test]
    fn missing_summary_fields_degrade_to_zero_not_panic() {
        // Files without a `summary` key or with missing counters are skipped
        // cleanly. The second file has a partial summary (lines only).
        let json = r#"{
          "data": [{
            "files": [
              {
                "filename": "ropus/src/a.rs"
              },
              {
                "filename": "ropus/src/b.rs",
                "summary": {
                  "lines": { "count": 10, "covered": 5 }
                }
              }
            ]
          }]
        }"#;
        let r = parse(json);
        // a.rs: no summary → counters stay zero.
        // b.rs: only `lines` counter present; functions/branches/regions
        // stay zero. Final rollup: lines 10/5, rest 0/0.
        assert_eq!(r.ropus.lines_total, 10);
        assert_eq!(r.ropus.lines_covered, 5);
        assert_eq!(r.ropus.functions_total, 0);
        assert_eq!(r.ropus.branches_total, 0);
        assert_eq!(r.ropus.regions_total, 0);
    }

    #[test]
    fn unix_paths_match_filters() {
        let json = r#"{
          "data": [{
            "files": [
              {
                "filename": "/home/arthur/mdopus/ropus/src/opus/encoder.rs",
                "summary": {
                  "lines":     { "count": 10, "covered": 7 },
                  "functions": { "count":  2, "covered": 2 },
                  "branches":  { "count":  4, "covered": 3 },
                  "regions":   { "count": 15, "covered": 14 }
                }
              }
            ]
          }]
        }"#;
        let r = parse(json);
        assert_eq!(r.ropus.lines_total, 10);
        assert_eq!(r.ropus.lines_covered, 7);
        assert_eq!(r.workspace.lines_total, 10);
    }

    #[test]
    fn full_test_paths_excluded_from_both_rollups() {
        // Running `full-test` via `cargo llvm-cov test --workspace`
        // instruments the runner's own code; that must not contaminate
        // the ropus headline or the workspace denominator.
        let json = r#"{
          "data": [{
            "files": [
              {
                "filename": "C:\\work\\mdopus\\full-test\\src\\main.rs",
                "summary": {
                  "lines": { "count": 500, "covered": 400 }
                }
              },
              {
                "filename": "/home/arthur/mdopus/full-test/src/tests.rs",
                "summary": {
                  "lines": { "count": 300, "covered": 200 }
                }
              }
            ]
          }]
        }"#;
        let r = parse(json);
        assert_eq!(r.ropus.lines_total, 0);
        assert_eq!(r.workspace.lines_total, 0);
    }

    #[test]
    fn target_paths_excluded_from_workspace() {
        // Build-script output that landed in llvm-cov's file list must not
        // inflate the workspace rollup.
        let json = r#"{
          "data": [{
            "files": [
              {
                "filename": "C:\\work\\mdopus\\target\\debug\\build\\foo-0/out/generated.rs",
                "summary": {
                  "lines": { "count": 1000, "covered": 1000 }
                }
              }
            ]
          }]
        }"#;
        let r = parse(json);
        assert_eq!(r.workspace.lines_total, 0);
        assert_eq!(r.ropus.lines_total, 0);
    }
}
