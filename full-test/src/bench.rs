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
use serde::ser::Serializer;

/// Benchmark coverage profile selected from CLI flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchProfile {
    /// No performance coverage is claimed. Used by `--quick` and local
    /// `--skip-benchmarks` runs.
    NotClaimed,
    /// Local/default benchmark visibility. Ratios above `BENCH_WARN_RATIO`
    /// remain WARN-only and exit 0.
    ObservedWarnOnly,
    /// Release-preflight benchmark gate. Required rows must produce complete
    /// timings and stay within per-vector severe-regression thresholds.
    ReleaseThresholded,
}

impl BenchProfile {
    pub fn from_flags(quick: bool, release_preflight: bool, skip_benchmarks: bool) -> Self {
        if release_preflight && !quick {
            Self::ReleaseThresholded
        } else if quick || skip_benchmarks {
            Self::NotClaimed
        } else {
            Self::ObservedWarnOnly
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotClaimed => "not-claimed",
            Self::ObservedWarnOnly => "observed-warn-only",
            Self::ReleaseThresholded => "release-thresholded",
        }
    }

    pub fn claimed(self) -> bool {
        !matches!(self, Self::NotClaimed)
    }

    pub fn claim_note(self) -> &'static str {
        match self {
            Self::NotClaimed => "performance coverage is not claimed by this profile",
            Self::ObservedWarnOnly => {
                "benchmark ratios are observed and reported; regressions are WARN-only"
            }
            Self::ReleaseThresholded => {
                "thresholded smoke over the canonical 10 benchmark vectors is release-blocking"
            }
        }
    }

    pub fn threshold_source(self) -> Option<&'static str> {
        match self {
            Self::ReleaseThresholded => Some(THRESHOLD_SOURCE),
            _ => None,
        }
    }

    fn release_thresholded(self) -> bool {
        matches!(self, Self::ReleaseThresholded)
    }
}

impl Serialize for BenchProfile {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

/// Row-level release threshold classification for JSON/HTML reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchThresholdStatus {
    Pass,
    Fail,
    MissingRequiredFixture,
    Crashed,
    MissingTiming,
    MissingThreshold,
}

impl BenchThresholdStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
            Self::MissingRequiredFixture => "missing_required_fixture",
            Self::Crashed => "crashed",
            Self::MissingTiming => "missing_timing",
            Self::MissingThreshold => "missing_threshold",
        }
    }
}

impl Serialize for BenchThresholdStatus {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct BenchThresholdRow {
    pub label: String,
    pub relative_path: Option<String>,
    pub enc_baseline_ratio: Option<f64>,
    pub dec_baseline_ratio: Option<f64>,
    pub enc_release_fail_ratio: Option<f64>,
    pub dec_release_fail_ratio: Option<f64>,
    pub enc_status: BenchThresholdStatus,
    pub dec_status: BenchThresholdStatus,
    pub issue: Option<String>,
}

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

#[derive(Debug, Clone, Copy)]
struct ThresholdSpec {
    label: &'static str,
    relative_path: &'static str,
    enc_baseline_ratio: f64,
    dec_baseline_ratio: f64,
    enc_release_fail_ratio: f64,
    dec_release_fail_ratio: f64,
}

/// Recalibrated 2026-05-11 against a fresh 5-sweep median baseline measured on
/// i7-14700KF / WSL2 / x86-64-v3 with the bench harness fixed to construct the
/// codec once outside the timed loop (commit 54fef85 — prior runs measured
/// constructor cost, not steady-state decode throughput). Per-row baseline is
/// the median of 5 clean sweeps; release_fail is computed by the same formula
/// as the original calibration. See `wrk_journals/2026.05.11 - JRN -
/// perf-recalibration.md`.
pub const THRESHOLD_SOURCE: &str = "recalibrated 2026-05-11 from 5-sweep median on i7-14700KF/WSL2/x86-64-v3 with bench-amortise fix; release_fail=max(baseline+0.05, baseline*1.20, 1.05), rounded up";

const THRESHOLDS: &[ThresholdSpec] = &[
    ThresholdSpec {
        label: "SILK NB 8k mono noise",
        relative_path: "tests/vectors/8000hz_mono_noise.wav",
        enc_baseline_ratio: 0.97,
        dec_baseline_ratio: 0.95,
        enc_release_fail_ratio: 1.17,
        dec_release_fail_ratio: 1.14,
    },
    ThresholdSpec {
        label: "SILK WB 16k mono noise",
        relative_path: "tests/vectors/16000hz_mono_noise.wav",
        enc_baseline_ratio: 0.96,
        dec_baseline_ratio: 1.10,
        enc_release_fail_ratio: 1.16,
        dec_release_fail_ratio: 1.32,
    },
    ThresholdSpec {
        label: "Hybrid 24k mono noise",
        relative_path: "tests/vectors/24000hz_mono_noise.wav",
        enc_baseline_ratio: 0.97,
        dec_baseline_ratio: 1.02,
        enc_release_fail_ratio: 1.17,
        dec_release_fail_ratio: 1.23,
    },
    ThresholdSpec {
        label: "CELT FB 48k mono noise",
        relative_path: "tests/vectors/48000hz_mono_noise.wav",
        enc_baseline_ratio: 1.13,
        dec_baseline_ratio: 0.98,
        enc_release_fail_ratio: 1.36,
        dec_release_fail_ratio: 1.18,
    },
    ThresholdSpec {
        label: "CELT FB 48k stereo noise",
        relative_path: "tests/vectors/48000hz_stereo_noise.wav",
        enc_baseline_ratio: 1.10,
        dec_baseline_ratio: 1.07,
        enc_release_fail_ratio: 1.32,
        dec_release_fail_ratio: 1.29,
    },
    ThresholdSpec {
        label: "CELT 48k mono sine 1k loud",
        relative_path: "tests/vectors/48k_sine1k_loud.wav",
        enc_baseline_ratio: 1.00,
        dec_baseline_ratio: 0.99,
        enc_release_fail_ratio: 1.20,
        dec_release_fail_ratio: 1.19,
    },
    ThresholdSpec {
        label: "CELT 48k mono sweep",
        relative_path: "tests/vectors/48k_sweep.wav",
        enc_baseline_ratio: 0.96,
        dec_baseline_ratio: 1.00,
        enc_release_fail_ratio: 1.16,
        dec_release_fail_ratio: 1.20,
    },
    ThresholdSpec {
        label: "CELT 48k mono square 1k",
        relative_path: "tests/vectors/48k_square1k.wav",
        enc_baseline_ratio: 1.20,
        dec_baseline_ratio: 0.92,
        enc_release_fail_ratio: 1.44,
        dec_release_fail_ratio: 1.11,
    },
    ThresholdSpec {
        label: "SPEECH 48k mono (SAPI TTS)",
        relative_path: "tests/vectors/speech_48k_mono.wav",
        enc_baseline_ratio: 1.20,
        dec_baseline_ratio: 1.01,
        enc_release_fail_ratio: 1.44,
        dec_release_fail_ratio: 1.22,
    },
    ThresholdSpec {
        label: "MUSIC 48k stereo",
        relative_path: "tests/vectors/music_48k_stereo.wav",
        enc_baseline_ratio: 1.16,
        dec_baseline_ratio: 1.07,
        enc_release_fail_ratio: 1.40,
        dec_release_fail_ratio: 1.29,
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
    pub profile: BenchProfile,
    pub skipped: bool,
    pub skip_reason: Option<String>,
    pub build_failed: bool,
    pub duration_ms: u64,
    pub vectors: Vec<VectorBench>,
}

impl BenchResult {
    /// Test-only convenience wrapper that defaults the profile to
    /// `BenchProfile::NotClaimed`. Production sites always know which
    /// profile applies and call [`Self::skipped_with_profile`] directly.
    #[cfg(test)]
    pub fn skipped(reason: impl Into<String>) -> Self {
        Self::skipped_with_profile(BenchProfile::NotClaimed, reason)
    }

    pub fn skipped_with_profile(profile: BenchProfile, reason: impl Into<String>) -> Self {
        Self {
            profile,
            skipped: true,
            skip_reason: Some(reason.into()),
            build_failed: false,
            duration_ms: 0,
            vectors: Vec::new(),
        }
    }

    /// Per default/local HLD semantics, bench anomalies are WARN-only. In the
    /// release-thresholded profile, `all_passed` also reflects threshold-gate
    /// failures so callers cannot mistake a disabled or incomplete release
    /// performance lane for green evidence.
    pub fn all_passed(&self) -> bool {
        if self.banner_fail() || self.build_failed {
            return false;
        }
        !self.vectors.iter().any(|v| v.crashed)
    }

    pub fn banner_fail(&self) -> bool {
        self.profile.release_thresholded() && !self.release_blocking_issues().is_empty()
    }

    pub fn release_blocking_issues(&self) -> Vec<String> {
        if !self.profile.release_thresholded() {
            return Vec::new();
        }

        let mut issues = Vec::new();
        if self.skipped {
            let reason = self.skip_reason.as_deref().unwrap_or("benchmarks skipped");
            issues.push(format!(
                "release-thresholded benchmark gate did not run: {reason}"
            ));
        }
        if self.build_failed {
            let reason = self
                .skip_reason
                .as_deref()
                .unwrap_or("ropus-compare build failed");
            issues.push(format!("benchmark build failed: {reason}"));
        }

        let mut gateable_rows = 0usize;
        for row in &self.vectors {
            let Some(threshold) = threshold_for_label(&row.label) else {
                issues.push(format!("missing threshold spec for {}", row.label));
                continue;
            };
            if row.skipped {
                let reason = row.skip_reason.as_deref().unwrap_or("fixture missing");
                issues.push(format!(
                    "required benchmark fixture missing for {}: {reason}",
                    row.label
                ));
                continue;
            }
            if row.crashed {
                let reason = row.crash_reason.as_deref().unwrap_or("no reason captured");
                issues.push(format!("benchmark row crashed for {}: {reason}", row.label));
                continue;
            }
            if !has_complete_timings(row) {
                issues.push(format!(
                    "benchmark row has incomplete timings for {}",
                    row.label
                ));
                continue;
            }
            let enc = row.enc_ratio.filter(|r| r.is_finite());
            let dec = row.dec_ratio.filter(|r| r.is_finite());
            match enc {
                Some(r) if r > threshold.enc_release_fail_ratio => issues.push(format!(
                    "{} encode ratio {r:.3}x exceeds release threshold {:.3}x",
                    row.label, threshold.enc_release_fail_ratio
                )),
                Some(_) => {}
                None => issues.push(format!(
                    "benchmark row has missing encode ratio for {}",
                    row.label
                )),
            }
            match dec {
                Some(r) if r > threshold.dec_release_fail_ratio => issues.push(format!(
                    "{} decode ratio {r:.3}x exceeds release threshold {:.3}x",
                    row.label, threshold.dec_release_fail_ratio
                )),
                Some(_) => {}
                None => issues.push(format!(
                    "benchmark row has missing decode ratio for {}",
                    row.label
                )),
            }
            if enc.is_some() && dec.is_some() {
                gateable_rows += 1;
            }
        }

        if gateable_rows == 0 {
            issues.push("zero gateable benchmark rows produced complete timings".to_string());
        }

        issues
    }

    pub fn threshold_rows(&self) -> Vec<BenchThresholdRow> {
        if !self.profile.release_thresholded() {
            return Vec::new();
        }
        self.vectors
            .iter()
            .map(|row| {
                let threshold = threshold_for_label(&row.label);
                let enc_status = threshold_status(row, threshold, Operation::Encode, row.enc_ratio);
                let dec_status = threshold_status(row, threshold, Operation::Decode, row.dec_ratio);
                let issue = release_row_issue(row, threshold);
                BenchThresholdRow {
                    label: row.label.clone(),
                    relative_path: threshold.map(|t| t.relative_path.to_string()),
                    enc_baseline_ratio: threshold.map(|t| t.enc_baseline_ratio),
                    dec_baseline_ratio: threshold.map(|t| t.dec_baseline_ratio),
                    enc_release_fail_ratio: threshold.map(|t| t.enc_release_fail_ratio),
                    dec_release_fail_ratio: threshold.map(|t| t.dec_release_fail_ratio),
                    enc_status,
                    dec_status,
                    issue,
                }
            })
            .collect()
    }
}

/// Workspace root, derived from `CARGO_MANIFEST_DIR` (points at `full-test/`).
fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().map(Path::to_path_buf).unwrap_or(manifest)
}

/// Dispatch the whole stage. Always returns a populated result; never panics.
pub fn run(profile: BenchProfile) -> BenchResult {
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
            profile,
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
            profile,
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
        profile,
        skipped: false,
        skip_reason: None,
        build_failed: false,
        duration_ms,
        vectors,
    }
}

#[derive(Debug, Clone, Copy)]
enum Operation {
    Encode,
    Decode,
}

fn threshold_for_label(label: &str) -> Option<&'static ThresholdSpec> {
    THRESHOLDS.iter().find(|t| t.label == label)
}

fn has_complete_timings(row: &VectorBench) -> bool {
    row.c_encode_ms.is_some()
        && row.rust_encode_ms.is_some()
        && row.c_decode_ms.is_some()
        && row.rust_decode_ms.is_some()
}

fn threshold_status(
    row: &VectorBench,
    threshold: Option<&ThresholdSpec>,
    operation: Operation,
    ratio: Option<f64>,
) -> BenchThresholdStatus {
    if row.skipped {
        return BenchThresholdStatus::MissingRequiredFixture;
    }
    if row.crashed {
        return BenchThresholdStatus::Crashed;
    }
    let Some(threshold) = threshold else {
        return BenchThresholdStatus::MissingThreshold;
    };
    if !has_complete_timings(row) {
        return BenchThresholdStatus::MissingTiming;
    }
    let Some(ratio) = ratio.filter(|r| r.is_finite()) else {
        return BenchThresholdStatus::MissingTiming;
    };
    let limit = match operation {
        Operation::Encode => threshold.enc_release_fail_ratio,
        Operation::Decode => threshold.dec_release_fail_ratio,
    };
    if ratio > limit {
        BenchThresholdStatus::Fail
    } else {
        BenchThresholdStatus::Pass
    }
}

fn release_row_issue(row: &VectorBench, threshold: Option<&ThresholdSpec>) -> Option<String> {
    if row.skipped {
        return Some(
            row.skip_reason
                .clone()
                .unwrap_or_else(|| "required fixture missing".to_string()),
        );
    }
    if row.crashed {
        return Some(
            row.crash_reason
                .clone()
                .unwrap_or_else(|| "benchmark row crashed".to_string()),
        );
    }
    let threshold = threshold?;
    if !has_complete_timings(row) {
        return Some("missing complete encode/decode timings".to_string());
    }
    if let Some(ratio) = row.enc_ratio.filter(|r| r.is_finite())
        && ratio > threshold.enc_release_fail_ratio
    {
        return Some(format!(
            "encode ratio {ratio:.3}x exceeds {:.3}x",
            threshold.enc_release_fail_ratio
        ));
    }
    if let Some(ratio) = row.dec_ratio.filter(|r| r.is_finite())
        && ratio > threshold.dec_release_fail_ratio
    {
        return Some(format!(
            "decode ratio {ratio:.3}x exceeds {:.3}x",
            threshold.dec_release_fail_ratio
        ));
    }
    None
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
            profile: BenchProfile::ObservedWarnOnly,
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
            profile: BenchProfile::ObservedWarnOnly,
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
            profile: BenchProfile::ObservedWarnOnly,
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
        assert_eq!(r.profile, BenchProfile::NotClaimed);
        assert_eq!(r.skip_reason.as_deref(), Some("--skip-benchmarks"));
        assert!(r.all_passed());
        assert!(r.vectors.is_empty());
    }

    #[test]
    fn empty_bench_result_is_all_passed() {
        // No vectors ran, no crashes, no build failure — stage stays green.
        // Phase 4's banner logic relies on this shape for the --skip paths.
        let r = BenchResult {
            profile: BenchProfile::ObservedWarnOnly,
            skipped: false,
            skip_reason: None,
            build_failed: false,
            duration_ms: 1000,
            vectors: Vec::new(),
        };
        assert!(r.all_passed());
    }

    #[test]
    fn benchmark_profile_follows_release_and_skip_flags() {
        assert_eq!(
            BenchProfile::from_flags(false, false, false),
            BenchProfile::ObservedWarnOnly
        );
        assert_eq!(
            BenchProfile::from_flags(false, false, true),
            BenchProfile::NotClaimed
        );
        assert_eq!(
            BenchProfile::from_flags(true, true, false),
            BenchProfile::NotClaimed
        );
        assert_eq!(
            BenchProfile::from_flags(false, true, false),
            BenchProfile::ReleaseThresholded
        );
        assert_eq!(
            BenchProfile::from_flags(false, true, true),
            BenchProfile::ReleaseThresholded
        );
    }

    #[test]
    fn release_thresholded_skip_is_release_blocking() {
        let r = BenchResult::skipped_with_profile(
            BenchProfile::ReleaseThresholded,
            "--skip-benchmarks disables release performance gate",
        );
        assert!(r.banner_fail());
        let issues = r.release_blocking_issues();
        assert!(issues[0].contains("did not run"));
        assert!(issues.iter().any(|issue| issue.contains("zero gateable")));
    }

    #[test]
    fn release_thresholded_missing_required_fixture_is_blocking() {
        let spec = VECTORS[0];
        let r = BenchResult {
            profile: BenchProfile::ReleaseThresholded,
            skipped: false,
            skip_reason: None,
            build_failed: false,
            duration_ms: 1000,
            vectors: vec![VectorBench {
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
            }],
        };
        assert!(r.banner_fail());
        assert!(
            r.release_blocking_issues()
                .iter()
                .any(|issue| issue.contains("required benchmark fixture missing"))
        );
        let rows = r.threshold_rows();
        assert_eq!(
            rows[0].enc_status,
            BenchThresholdStatus::MissingRequiredFixture
        );
    }

    #[test]
    fn release_thresholded_missing_timings_are_blocking() {
        let spec = VECTORS[0];
        let row = build_vector_row(
            &spec,
            BenchTimings {
                c_encode_ms: Some(100.0),
                rust_encode_ms: Some(110.0),
                c_decode_ms: None,
                rust_decode_ms: None,
            },
            None,
        );
        let r = BenchResult {
            profile: BenchProfile::ReleaseThresholded,
            skipped: false,
            skip_reason: None,
            build_failed: false,
            duration_ms: 1000,
            vectors: vec![row],
        };
        assert!(r.banner_fail());
        assert!(
            r.release_blocking_issues()
                .iter()
                .any(|issue| issue.contains("incomplete timings"))
        );
        let rows = r.threshold_rows();
        assert_eq!(rows[0].dec_status, BenchThresholdStatus::MissingTiming);
    }

    #[test]
    fn release_thresholded_missing_threshold_spec_is_blocking() {
        let row = VectorBench {
            label: "local exploratory vector".to_string(),
            bitrate: 64_000,
            skipped: false,
            skip_reason: None,
            crashed: false,
            crash_reason: None,
            c_encode_ms: Some(100.0),
            rust_encode_ms: Some(100.0),
            enc_ratio: Some(1.0),
            c_decode_ms: Some(100.0),
            rust_decode_ms: Some(100.0),
            dec_ratio: Some(1.0),
        };
        let r = BenchResult {
            profile: BenchProfile::ReleaseThresholded,
            skipped: false,
            skip_reason: None,
            build_failed: false,
            duration_ms: 1000,
            vectors: vec![row],
        };
        assert!(r.banner_fail());
        assert!(
            r.release_blocking_issues()
                .iter()
                .any(|issue| issue.contains("missing threshold spec"))
        );
        let rows = r.threshold_rows();
        assert_eq!(rows[0].enc_status, BenchThresholdStatus::MissingThreshold);
        assert_eq!(rows[0].dec_status, BenchThresholdStatus::MissingThreshold);
    }

    #[test]
    fn release_thresholded_crashed_row_is_blocking() {
        let spec = VECTORS[0];
        let row = build_vector_row(
            &spec,
            BenchTimings::default(),
            Some("ropus-compare exited with status 1".to_string()),
        );
        let r = BenchResult {
            profile: BenchProfile::ReleaseThresholded,
            skipped: false,
            skip_reason: None,
            build_failed: false,
            duration_ms: 1000,
            vectors: vec![row],
        };
        assert!(r.banner_fail());
        assert!(
            r.release_blocking_issues()
                .iter()
                .any(|issue| issue.contains("benchmark row crashed"))
        );
        let rows = r.threshold_rows();
        assert_eq!(rows[0].enc_status, BenchThresholdStatus::Crashed);
        assert_eq!(rows[0].dec_status, BenchThresholdStatus::Crashed);
    }

    #[test]
    fn release_thresholded_ratio_breach_is_blocking_per_operation() {
        let spec = VECTORS[0];
        let row = build_vector_row(
            &spec,
            BenchTimings {
                c_encode_ms: Some(100.0),
                rust_encode_ms: Some(200.0),
                c_decode_ms: Some(100.0),
                rust_decode_ms: Some(80.0),
            },
            None,
        );
        let r = BenchResult {
            profile: BenchProfile::ReleaseThresholded,
            skipped: false,
            skip_reason: None,
            build_failed: false,
            duration_ms: 1000,
            vectors: vec![row],
        };
        assert!(r.banner_fail());
        assert!(
            r.release_blocking_issues()
                .iter()
                .any(|issue| issue.contains("encode ratio 2.000x exceeds"))
        );
        let rows = r.threshold_rows();
        assert_eq!(rows[0].enc_status, BenchThresholdStatus::Fail);
        assert_eq!(rows[0].dec_status, BenchThresholdStatus::Pass);
        assert_eq!(rows[0].enc_release_fail_ratio, Some(1.26));
    }

    #[test]
    fn release_thresholded_complete_rows_pass_threshold_gate() {
        let spec = VECTORS[0];
        let row = build_vector_row(
            &spec,
            BenchTimings {
                c_encode_ms: Some(100.0),
                rust_encode_ms: Some(110.0),
                c_decode_ms: Some(100.0),
                rust_decode_ms: Some(90.0),
            },
            None,
        );
        let r = BenchResult {
            profile: BenchProfile::ReleaseThresholded,
            skipped: false,
            skip_reason: None,
            build_failed: false,
            duration_ms: 1000,
            vectors: vec![row],
        };
        assert!(!r.banner_fail());
        assert!(r.release_blocking_issues().is_empty());
        let rows = r.threshold_rows();
        assert_eq!(rows[0].enc_status, BenchThresholdStatus::Pass);
        assert_eq!(rows[0].dec_status, BenchThresholdStatus::Pass);
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
