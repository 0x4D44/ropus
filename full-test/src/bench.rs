//! Stage 4 — native benchmark sweep.
//!
//! The HLD is explicit: **do not shell out** to `tools/bench_sweep.sh`. Instead
//! we loop `ropus-compare bench <wav> --iters 30 --repeats 1 --bitrate <bps>`
//! natively, parse the `│ C encode │ … │ <ms>ms │ …` table rows, and compute
//! encode/decode ratios per vector. This removes the hard dependency on
//! Git-Bash-on-PATH that breaks bare Windows installs.
//!
//! The vector set mirrors `tools/bench_sweep.sh` (lines 21-33) verbatim so
//! the README-tracking `--iters=50` runs and the full-test regression runs
//! cover the same ground.
//!
//! Strategy:
//!
//! 1. One-time `cargo build --release --bin ropus-compare` so the per-vector
//!    loop spawns the binary directly (no cargo round-trips in the hot path).
//! 2. For each vector, skip cleanly if the WAV is missing (fuzz probes /
//!    assets-not-fetched environments shouldn't turn into a FAIL).
//! 3. Parse the bench report table. Output uses Unicode box-drawing chars so
//!    splitting on `│` gives us the four columns — we grab the 4th, strip
//!    `ms` and whitespace, parse the float.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use colored::Colorize;
use serde::Serialize;

/// One vector in the sweep. Mirrors `tools/bench_sweep.sh` lines 21-30.
#[derive(Debug, Clone, Copy)]
struct VectorSpec {
    label: &'static str,
    relative_path: &'static str,
    bitrate: u32,
}

/// Canonical vector set. Paths are workspace-relative; resolved against
/// `CARGO_MANIFEST_DIR` at runtime so the runner works from any cwd.
const VECTORS: &[VectorSpec] = &[
    VectorSpec {
        label: "SILK NB 8k mono noise",
        relative_path: "tests/vectors/8000hz_mono_noise.wav",
        bitrate: 16_000,
    },
    VectorSpec {
        label: "SILK WB 16k mono noise",
        relative_path: "tests/vectors/16000hz_mono_noise.wav",
        bitrate: 24_000,
    },
    VectorSpec {
        label: "Hybrid 24k mono noise",
        relative_path: "tests/vectors/24000hz_mono_noise.wav",
        bitrate: 32_000,
    },
    VectorSpec {
        label: "CELT FB 48k mono noise",
        relative_path: "tests/vectors/48000hz_mono_noise.wav",
        bitrate: 64_000,
    },
    VectorSpec {
        label: "CELT FB 48k stereo noise",
        relative_path: "tests/vectors/48000hz_stereo_noise.wav",
        bitrate: 96_000,
    },
    VectorSpec {
        label: "CELT 48k mono sine 1k loud",
        relative_path: "tests/vectors/48k_sine1k_loud.wav",
        bitrate: 64_000,
    },
    VectorSpec {
        label: "CELT 48k mono sweep",
        relative_path: "tests/vectors/48k_sweep.wav",
        bitrate: 64_000,
    },
    VectorSpec {
        label: "CELT 48k mono square 1k",
        relative_path: "tests/vectors/48k_square1k.wav",
        bitrate: 64_000,
    },
    VectorSpec {
        label: "SPEECH 48k mono (SAPI TTS)",
        relative_path: "tests/vectors/speech_48k_mono.wav",
        bitrate: 64_000,
    },
    VectorSpec {
        label: "MUSIC 48k stereo",
        relative_path: "tests/vectors/music_48k_stereo.wav",
        bitrate: 128_000,
    },
];

/// The four timings we pull out of a single `ropus-compare bench` run.
#[derive(Debug, Clone, Copy, Default)]
pub struct BenchTimings {
    pub c_encode_ms: Option<f64>,
    pub rust_encode_ms: Option<f64>,
    pub c_decode_ms: Option<f64>,
    pub rust_decode_ms: Option<f64>,
}

/// Per-vector bench row used by the report.
///
/// Two distinct failure modes are tracked separately:
///
/// - `skipped=true`, `crashed=false` — fixture missing on disk. Expected on
///   assets-not-fetched environments; the row is informational.
/// - `skipped=false`, `crashed=true` — `ropus-compare` was invoked but failed
///   (non-zero exit, parser drift, spawn error). Elevates the stage to WARN
///   per HLD § PASS/FAIL/WARN ("bench is WARN-only on regressions; only
///   crashes elevate").
#[derive(Debug, Clone, Serialize)]
pub struct VectorBench {
    pub label: String,
    pub bitrate: u32,
    pub skipped: bool,
    pub skip_reason: Option<String>,
    pub crashed: bool,
    pub crash_reason: Option<String>,
    pub c_encode_ms: Option<f64>,
    pub rust_encode_ms: Option<f64>,
    pub enc_ratio: Option<f64>,
    pub c_decode_ms: Option<f64>,
    pub rust_decode_ms: Option<f64>,
    pub dec_ratio: Option<f64>,
}

/// Stage 4 result aggregate. `skipped` is stage-level (set when
/// `--skip-benchmarks`, `--quick`, or upstream-build-failure fires).
/// `build_failed` is stage-level too — flipped when the one-time prebuild
/// fails (either because cargo returned non-zero, or the binary isn't on
/// disk after a nominally-successful build).
#[derive(Debug, Clone, Serialize)]
pub struct BenchResult {
    pub skipped: bool,
    pub skip_reason: Option<String>,
    pub build_failed: bool,
    pub duration_ms: u64,
    pub vectors: Vec<VectorBench>,
}

impl BenchResult {
    /// Construct a "stage disabled" outcome. Used by `--skip-benchmarks`,
    /// `--quick`, and the upstream-build-failure chaining rule from the HLD.
    pub fn skipped(reason: impl Into<String>) -> Self {
        Self {
            skipped: true,
            skip_reason: Some(reason.into()),
            build_failed: false,
            duration_ms: 0,
            vectors: Vec::new(),
        }
    }

    /// Per HLD § PASS/FAIL/WARN: bench is WARN-only on regressions; only
    /// crashes elevate. Slow ratios, missing fixtures, and a clean skip all
    /// keep `all_passed()` true; a `crashed` vector is the one signal that
    /// flips the stage non-green. Stage-level build failure also fails the
    /// stage since no vector ever ran.
    pub fn all_passed(&self) -> bool {
        if self.build_failed {
            return false;
        }
        !self.vectors.iter().any(|v| v.crashed)
    }
}

/// Workspace root, derived from `CARGO_MANIFEST_DIR` (points at `full-test/`).
fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().map(Path::to_path_buf).unwrap_or(manifest)
}

/// Dispatch the whole stage. Always returns a populated result; never panics.
pub fn run() -> BenchResult {
    let start = Instant::now();
    let root = workspace_root();

    eprintln!(
        "{} building ropus-compare once, then sweeping {} vectors",
        "[bench]".cyan().bold(),
        VECTORS.len()
    );

    // One-time build of `ropus-compare`. Spawning `cargo run` per vector
    // round-trips through cargo's lock each time; a single prebuild keeps
    // the hot loop pure process invocations.
    if let Some(err) = prebuild_ropus_compare() {
        let duration_ms = start.elapsed().as_millis() as u64;
        eprintln!("  {} ropus-compare build failed: {err}", "BUILD-FAIL".red());
        return BenchResult {
            skipped: false,
            skip_reason: Some(format!("ropus-compare build failed: {err}")),
            build_failed: true,
            duration_ms,
            vectors: Vec::new(),
        };
    }

    let bin = ropus_compare_path(&root);
    // Defence against a prebuild that claims success but leaves nothing on
    // disk (stale lock, concurrent delete, exotic `CARGO_TARGET_DIR`). Without
    // this guard every per-vector `Command::new(bin)` would hit
    // `ErrorKind::NotFound` and surface as 10 identical spawn-failure rows.
    if !bin.is_file() {
        let duration_ms = start.elapsed().as_millis() as u64;
        eprintln!(
            "  {} ropus-compare binary not on disk after build",
            "BUILD-FAIL".red()
        );
        return BenchResult {
            skipped: false,
            skip_reason: Some("ropus-compare binary not on disk after build".to_string()),
            build_failed: true,
            duration_ms,
            vectors: Vec::new(),
        };
    }
    let mut vectors: Vec<VectorBench> = Vec::with_capacity(VECTORS.len());

    for spec in VECTORS {
        let wav = root.join(spec.relative_path);
        if !wav.is_file() {
            eprintln!(
                "  {} {} — missing {}",
                "SKIP".yellow(),
                spec.label,
                spec.relative_path
            );
            vectors.push(VectorBench {
                label: spec.label.to_string(),
                bitrate: spec.bitrate,
                skipped: true,
                skip_reason: Some(format!("fixture missing at {}", spec.relative_path)),
                crashed: false,
                crash_reason: None,
                c_encode_ms: None,
                rust_encode_ms: None,
                enc_ratio: None,
                c_decode_ms: None,
                rust_decode_ms: None,
                dec_ratio: None,
            });
            continue;
        }

        let vb = run_one_vector(&bin, spec, &wav);
        vectors.push(vb);
    }

    let duration_ms = start.elapsed().as_millis() as u64;
    eprintln!(
        "  {} {} vectors in {:.2}s",
        "DONE".green(),
        vectors.len(),
        duration_ms as f64 / 1000.0
    );

    BenchResult {
        skipped: false,
        skip_reason: None,
        build_failed: false,
        duration_ms,
        vectors,
    }
}

/// Prebuild `ropus-compare` in release mode. Returns `Some(err)` on failure
/// so the caller can short-circuit the per-vector loop.
fn prebuild_ropus_compare() -> Option<String> {
    // `CARGO_TERM_COLOR=never` suppresses ANSI escapes in cargo's own output
    // regardless of terminal detection on the developer box. Keeps stderr
    // parsing (e.g. `detect_build_failure`) robust.
    let output = Command::new("cargo")
        .env("CARGO_TERM_COLOR", "never")
        .args(["build", "--release", "--bin", "ropus-compare"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();
    match output {
        Ok(o) if o.status.success() => None,
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            let (stderr, _) = crate::issues::cap_stderr(&stderr);
            let hint = stderr
                .lines()
                .find(|l| {
                    let t = l.trim_start();
                    t.starts_with("error[E") || t.starts_with("error: ")
                })
                .unwrap_or("cargo build exited non-zero")
                .trim()
                .to_string();
            Some(hint)
        }
        Err(e) => Some(format!("failed to spawn cargo: {e}")),
    }
}

/// Location of the prebuilt `ropus-compare` binary. Windows uses `.exe`; Unix
/// doesn't. Matching `bench_sweep.sh`'s `ropus-compare.exe` suffix on Windows.
fn ropus_compare_path(root: &Path) -> PathBuf {
    let mut p = root.join("target").join("release");
    if cfg!(windows) {
        p.push("ropus-compare.exe");
    } else {
        p.push("ropus-compare");
    }
    p
}

/// Run a single vector through `ropus-compare bench` and parse the timings.
fn run_one_vector(bin: &Path, spec: &VectorSpec, wav: &Path) -> VectorBench {
    eprintln!(
        "  {} {} (bitrate={})",
        "[bench]".cyan().bold(),
        spec.label,
        spec.bitrate
    );
    // Force plain output: `--color=never` on the subcommand plus
    // `CARGO_TERM_COLOR=never` in the env hardens the Unicode-box-drawing
    // parser against any future terminal-detection logic.
    let output = Command::new(bin)
        .env("CARGO_TERM_COLOR", "never")
        .args([
            "bench",
            wav.to_string_lossy().as_ref(),
            "--iters",
            "30",
            "--repeats",
            "1",
            "--bitrate",
            &spec.bitrate.to_string(),
            "--color=never",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();

    match output {
        Ok(o) if o.status.success() => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            let timings = parse_bench_stdout(&stdout);
            build_vector_row(spec, timings, None)
        }
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            let reason = stderr
                .lines()
                .next()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| {
                    format!(
                        "ropus-compare exited with status {}",
                        o.status.code().unwrap_or(-1)
                    )
                });
            build_vector_row(spec, BenchTimings::default(), Some(reason))
        }
        Err(e) => build_vector_row(
            spec,
            BenchTimings::default(),
            Some(format!("failed to spawn ropus-compare: {e}")),
        ),
    }
}

/// Fold a parsed `BenchTimings` + optional crash reason into the report row
/// shape. Kept pure so the report-shape tests don't have to spawn processes.
///
/// `crash_reason` being `Some` means `ropus-compare` was invoked but failed
/// (non-zero exit, parser drift, spawn error) — distinct from the
/// fixture-missing path in `run()`, which never reaches this function. The
/// crash path flips `crashed=true` so `all_passed()` can surface WARNs per
/// HLD § PASS/FAIL/WARN.
///
/// TODO: if `harness::print_bench_report` (`harness/src/main.rs`) changes its
/// column widths or swaps the Unicode box chars, `SAMPLE_BENCH_STDOUT` and
/// `parse_bench_stdout` need a matching refresh. That function is the
/// source-of-truth for the parser.
fn build_vector_row(
    spec: &VectorSpec,
    timings: BenchTimings,
    crash_reason: Option<String>,
) -> VectorBench {
    let enc_ratio = match (timings.rust_encode_ms, timings.c_encode_ms) {
        (Some(r), Some(c)) if c > 0.0 => Some(r / c),
        _ => None,
    };
    let dec_ratio = match (timings.rust_decode_ms, timings.c_decode_ms) {
        (Some(r), Some(c)) if c > 0.0 => Some(r / c),
        _ => None,
    };
    VectorBench {
        label: spec.label.to_string(),
        bitrate: spec.bitrate,
        skipped: false,
        skip_reason: None,
        crashed: crash_reason.is_some(),
        crash_reason,
        c_encode_ms: timings.c_encode_ms,
        rust_encode_ms: timings.rust_encode_ms,
        enc_ratio,
        c_decode_ms: timings.c_decode_ms,
        rust_decode_ms: timings.rust_decode_ms,
        dec_ratio,
    }
}

/// Parse the Unicode-box-drawn `print_bench_report` table from
/// `harness/src/main.rs`. Format (cols):
///
/// ```text
/// │ C encode     │     30 │   1234.567ms │        1234   │
/// │ Rust encode  │     30 │    987.654ms │        1872   │
/// │ C decode     │     30 │     42.123ms │       43941   │
/// │ Rust decode  │     30 │     38.456ms │       48123   │
/// ```
///
/// We split each line on `│` and pull the 4th column (3rd index, zero-based).
/// Missing rows map to `None` rather than panicking — a partial table is
/// still worth surfacing whatever ran.
pub fn parse_bench_stdout(stdout: &str) -> BenchTimings {
    let mut t = BenchTimings::default();
    for line in stdout.lines() {
        // Fast reject: only consider lines that start with the left wall of
        // the table.
        let trimmed = line.trim_start();
        if !trimmed.starts_with('\u{2502}') {
            continue;
        }
        let cols: Vec<&str> = trimmed.split('\u{2502}').collect();
        // Expected: ["", " C encode     ", "     30 ", "   1234.567ms ", "        1234   ", ""]
        if cols.len() < 5 {
            continue;
        }
        let label_col = cols[1].trim();
        let time_col = cols[3].trim();
        let ms = match parse_ms(time_col) {
            Some(v) => v,
            None => continue,
        };
        match label_col {
            "C encode" => t.c_encode_ms = Some(ms),
            "Rust encode" => t.rust_encode_ms = Some(ms),
            "C decode" => t.c_decode_ms = Some(ms),
            "Rust decode" => t.rust_decode_ms = Some(ms),
            _ => {}
        }
    }
    t
}

/// Parse a `"1234.567ms"` (or `"  1234.567ms  "`) into an f64. Returns
/// `None` on any parse error — the caller treats that as "no sample".
fn parse_ms(s: &str) -> Option<f64> {
    let num = s.trim().strip_suffix("ms").unwrap_or(s.trim()).trim();
    num.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_BENCH_STDOUT: &str = "\
=== ropus benchmark ===
Input : tests/vectors/48k_sweep.wav (48000 Hz, 1 ch, 10.00s, 500 frames @ 20ms)
Config: bitrate=64000, complexity=10, CBR
Iters : 30 (each iter encodes/decodes ALL frames)

Warming up (1 iter each)...

Benchmarking encode...
Benchmarking decode...

\u{250C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{252C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{252C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{252C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}
\u{2502} Operation    \u{2502}  Iters \u{2502}  ms/iter     \u{2502}  frames/sec   \u{2502}
\u{251C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2524}
\u{2502} C encode     \u{2502}     30 \u{2502}   1234.567ms \u{2502}        1234   \u{2502}
\u{2502} Rust encode  \u{2502}     30 \u{2502}    987.654ms \u{2502}        1872   \u{2502}
\u{2502} C decode     \u{2502}     30 \u{2502}     42.123ms \u{2502}       43941   \u{2502}
\u{2502} Rust decode  \u{2502}     30 \u{2502}     38.456ms \u{2502}       48123   \u{2502}
\u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2534}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2534}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2534}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2518}

  encode : Rust is 1.25x FASTER than C (987.654ms vs 1234.567ms per iter)
  decode : Rust is 1.10x FASTER than C (38.456ms vs 42.123ms per iter)
";

    #[test]
    fn parses_all_four_timings_from_sample() {
        let t = parse_bench_stdout(SAMPLE_BENCH_STDOUT);
        assert_eq!(t.c_encode_ms, Some(1234.567));
        assert_eq!(t.rust_encode_ms, Some(987.654));
        assert_eq!(t.c_decode_ms, Some(42.123));
        assert_eq!(t.rust_decode_ms, Some(38.456));
    }

    #[test]
    fn missing_table_yields_all_none() {
        let t = parse_bench_stdout("=== ropus benchmark ===\nno table here\n");
        assert!(t.c_encode_ms.is_none());
        assert!(t.rust_encode_ms.is_none());
        assert!(t.c_decode_ms.is_none());
        assert!(t.rust_decode_ms.is_none());
    }

    #[test]
    fn partial_table_surfaces_what_it_can() {
        // Only the encode rows made it (harness crashed mid-decode).
        let s = "\
\u{2502} C encode     \u{2502}     30 \u{2502}   100.000ms \u{2502}        1234   \u{2502}
\u{2502} Rust encode  \u{2502}     30 \u{2502}    80.000ms \u{2502}        1872   \u{2502}
thread 'main' panicked at 'decode crash'
";
        let t = parse_bench_stdout(s);
        assert_eq!(t.c_encode_ms, Some(100.0));
        assert_eq!(t.rust_encode_ms, Some(80.0));
        assert!(t.c_decode_ms.is_none());
        assert!(t.rust_decode_ms.is_none());
    }

    #[test]
    fn parse_ms_strips_suffix_and_whitespace() {
        assert_eq!(parse_ms("1234.567ms"), Some(1234.567));
        assert_eq!(parse_ms("   42.1ms  "), Some(42.1));
        assert_eq!(parse_ms("0ms"), Some(0.0));
        assert!(parse_ms("not a number").is_none());
        // Non-numeric input is rejected outright by `parse()`; the fallback
        // branch below triggers only for bare numeric text ("42" with no
        // `ms` suffix), which is never what the harness actually emits.
        assert!(parse_ms("abc").is_none());
    }

    #[test]
    fn build_vector_row_computes_ratios_from_timings() {
        let spec = VECTORS[0];
        let t = BenchTimings {
            c_encode_ms: Some(100.0),
            rust_encode_ms: Some(75.0),
            c_decode_ms: Some(50.0),
            rust_decode_ms: Some(55.0),
        };
        let row = build_vector_row(&spec, t, None);
        assert!(!row.skipped);
        assert!(!row.crashed);
        assert_eq!(row.enc_ratio, Some(0.75));
        assert_eq!(row.dec_ratio, Some(1.1));
        assert_eq!(row.c_encode_ms, Some(100.0));
        assert_eq!(row.rust_encode_ms, Some(75.0));
    }

    #[test]
    fn build_vector_row_with_zero_c_time_leaves_ratio_none() {
        // A 0.0 baseline would yield inf; we prefer None so downstream
        // report code doesn't have to special-case non-finite floats.
        let spec = VECTORS[0];
        let t = BenchTimings {
            c_encode_ms: Some(0.0),
            rust_encode_ms: Some(1.0),
            c_decode_ms: Some(0.0),
            rust_decode_ms: Some(1.0),
        };
        let row = build_vector_row(&spec, t, None);
        assert!(row.enc_ratio.is_none());
        assert!(row.dec_ratio.is_none());
    }

    #[test]
    fn build_vector_row_missing_timings_are_none() {
        let spec = VECTORS[0];
        let row = build_vector_row(&spec, BenchTimings::default(), None);
        assert!(row.enc_ratio.is_none());
        assert!(row.dec_ratio.is_none());
        assert!(row.c_encode_ms.is_none());
        assert!(row.rust_encode_ms.is_none());
    }

    #[test]
    fn build_vector_row_with_crash_reason_marks_crashed() {
        let spec = VECTORS[0];
        let row = build_vector_row(
            &spec,
            BenchTimings::default(),
            Some("bench runner crashed".to_string()),
        );
        assert!(row.crashed);
        assert!(!row.skipped);
        assert_eq!(row.crash_reason.as_deref(), Some("bench runner crashed"));
    }

    #[test]
    fn crashed_vector_fails_stage_but_missing_fixture_does_not() {
        // Phase 3 CR item 3: distinguish bench crash from missing fixture.
        // A crashed vector elevates the stage to non-green (WARN per HLD
        // § PASS/FAIL/WARN); a skipped (fixture-missing) vector does not.
        let spec = VECTORS[0];
        let crashed_row = build_vector_row(
            &spec,
            BenchTimings::default(),
            Some("ropus-compare exited with status 1".to_string()),
        );
        let skipped_row = VectorBench {
            label: spec.label.to_string(),
            bitrate: spec.bitrate,
            skipped: true,
            skip_reason: Some(format!("fixture missing at {}", spec.relative_path)),
            crashed: false,
            crash_reason: None,
            c_encode_ms: None,
            rust_encode_ms: None,
            enc_ratio: None,
            c_decode_ms: None,
            rust_decode_ms: None,
            dec_ratio: None,
        };

        let only_skipped = BenchResult {
            skipped: false,
            skip_reason: None,
            build_failed: false,
            duration_ms: 1000,
            vectors: vec![skipped_row.clone()],
        };
        assert!(
            only_skipped.all_passed(),
            "missing-fixture skip should not fail the stage"
        );

        let with_crash = BenchResult {
            skipped: false,
            skip_reason: None,
            build_failed: false,
            duration_ms: 1000,
            vectors: vec![skipped_row, crashed_row],
        };
        assert!(
            !with_crash.all_passed(),
            "a crashed vector should flip all_passed() to false"
        );
    }

    #[test]
    fn build_failed_fails_stage() {
        // A stage-level build failure (e.g. `ropus-compare` binary missing on
        // disk after a nominally-successful build) means nothing ran; the
        // stage should surface non-green rather than claiming all_passed.
        let r = BenchResult {
            skipped: false,
            skip_reason: Some("ropus-compare binary not on disk after build".to_string()),
            build_failed: true,
            duration_ms: 500,
            vectors: Vec::new(),
        };
        assert!(!r.all_passed());
    }

    #[test]
    fn vector_table_matches_bench_sweep_script_order() {
        // The HLD requires the canonical list to match `tools/bench_sweep.sh`
        // lines 21-30 (10 vectors, specific order). This guards against
        // silent edits.
        assert_eq!(VECTORS.len(), 10);
        assert_eq!(VECTORS[0].label, "SILK NB 8k mono noise");
        assert_eq!(VECTORS[0].bitrate, 16_000);
        assert_eq!(VECTORS[9].label, "MUSIC 48k stereo");
        assert_eq!(VECTORS[9].bitrate, 128_000);
    }

    #[test]
    fn all_vector_paths_are_under_tests_vectors() {
        // Defensive: catch copy-paste errors that would write outside the
        // fixture tree. `tools/bench_sweep.sh` anchors every entry with
        // `$ROOT/tests/vectors/`; we mirror that invariant.
        for v in VECTORS {
            assert!(
                v.relative_path.starts_with("tests/vectors/"),
                "vector {:?} escaped tests/vectors/",
                v.label
            );
        }
    }

    #[test]
    fn skipped_stage_is_green() {
        // The stage-disabled skip path keeps `all_passed()` true and populates
        // the reason field for the report.
        let r = BenchResult::skipped("--skip-benchmarks");
        assert!(r.skipped);
        assert_eq!(r.skip_reason.as_deref(), Some("--skip-benchmarks"));
        assert!(r.all_passed());
        assert!(r.vectors.is_empty());
    }

    #[test]
    fn empty_bench_result_is_all_passed() {
        // No vectors ran, no crashes, no build failure — stage stays green.
        // Phase 4's banner logic relies on this shape for the --skip paths.
        let r = BenchResult {
            skipped: false,
            skip_reason: None,
            build_failed: false,
            duration_ms: 1000,
            vectors: Vec::new(),
        };
        assert!(r.all_passed());
    }

    #[test]
    fn ropus_compare_path_uses_exe_on_windows_only() {
        let p = ropus_compare_path(Path::new("/tmp/root"));
        let s = p.to_string_lossy();
        if cfg!(windows) {
            assert!(s.ends_with("ropus-compare.exe"));
        } else {
            assert!(s.ends_with("ropus-compare"));
            assert!(!s.ends_with(".exe"));
        }
        assert!(s.contains("target"));
        assert!(s.contains("release"));
    }
}
