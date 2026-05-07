//! Real-world/generated corpus gate.
//!
//! This stage keeps corpus language honest. Default and quick release
//! preflight runs report manifest/provisioning status without making a corpus
//! coverage claim. Non-quick release preflight generates a tiny non-reference
//! Ogg Opus smoke from an existing WAV fixture using FFmpeg's native `opus`
//! encoder, then requires `corpus_diff` to decode and match it against the C
//! reference.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::Instant;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::cli::Options;

/// Subprocess seam used by the corpus stage. The production runner shells out
/// to `std::process::Command`; tests substitute a fake that records calls and
/// scripts responses, so the parser/pinning/dispatch logic can be exercised
/// without touching FFmpeg or `corpus_diff` on the host.
pub trait CommandRunner {
    fn run(&self, program: &str, args: &[String]) -> std::io::Result<Output>;
}

pub struct RealCommandRunner;

impl CommandRunner for RealCommandRunner {
    fn run(&self, program: &str, args: &[String]) -> std::io::Result<Output> {
        Command::new(program).args(args).output()
    }
}

pub const ORACLE_NOTE: &str = "exact PCM parity vs C reference for supported Ogg family-0 packet decode only; no WebM/player semantics, seek, output gain, pre-skip, granule, FEC, or PLC coverage";

const MANIFEST_REL: &str = "tests/vectors/real_world/corpus_manifest.toml";
const CORPUS_DIR_REL: &str = "tests/vectors/real_world";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    ReportOnly,
    QuickNoClaim,
    GeneratedSmoke,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ReportOnly => "report_only",
            Self::QuickNoClaim => "quick_no_claim",
            Self::GeneratedSmoke => "generated_smoke",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    NotClaimed,
    Pass,
    Fail,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotClaimed => "not_claimed",
            Self::Pass => "pass",
            Self::Fail => "fail",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EntryOutcome {
    pub id: String,
    pub required: bool,
    pub corpus_class: String,
    pub path: String,
    pub source_fixture: String,
    pub encoder: String,
    pub container_support: String,
    pub oracle: String,
    pub local_path_present: bool,
    pub generated: bool,
    pub compared: bool,
    pub bytes: Option<u64>,
    pub sha256: Option<String>,
    pub status: String,
    pub note: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CorpusDiffSummary {
    pub candidates: usize,
    pub decoded_and_compared: usize,
    pub zero_audio: usize,
    pub skipped: usize,
    pub deferred: usize,
    pub mismatched: usize,
    pub panicked: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Outcome {
    pub mode: Mode,
    pub status: Status,
    pub claimed: bool,
    pub claim_note: String,
    pub oracle_note: &'static str,
    pub manifest_path: String,
    pub required_entries: usize,
    pub compared_required_entries: usize,
    pub pinned_entries_matched: usize,
    pub duration_ms: u64,
    pub command: Vec<String>,
    pub ffmpeg_version: Option<String>,
    pub corpus_diff_summary: Option<CorpusDiffSummary>,
    pub entries: Vec<EntryOutcome>,
    pub issues: Vec<String>,
    pub stdout: String,
    pub stderr: String,
}

impl Outcome {
    pub fn banner_fail(&self) -> bool {
        self.claimed && self.status == Status::Fail
    }

    #[cfg(test)]
    pub fn not_claimed_for_tests() -> Self {
        Self {
            mode: Mode::ReportOnly,
            status: Status::NotClaimed,
            claimed: false,
            claim_note: "default full-test report-only; real-world corpus coverage is not claimed"
                .to_string(),
            oracle_note: ORACLE_NOTE,
            manifest_path: MANIFEST_REL.to_string(),
            required_entries: 1,
            compared_required_entries: 0,
            pinned_entries_matched: 0,
            duration_ms: 0,
            command: Vec::new(),
            ffmpeg_version: None,
            corpus_diff_summary: None,
            entries: Vec::new(),
            issues: Vec::new(),
            stdout: String::new(),
            stderr: String::new(),
        }
    }

    #[cfg(test)]
    pub fn failing_for_tests(issue: impl Into<String>) -> Self {
        let mut out = Self::not_claimed_for_tests();
        out.mode = Mode::GeneratedSmoke;
        out.status = Status::Fail;
        out.claimed = true;
        out.claim_note =
            "non-quick release-preflight claims generated real-world corpus gate".to_string();
        out.issues.push(issue.into());
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Manifest {
    path: PathBuf,
    entries: Vec<ManifestEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ManifestEntry {
    id: String,
    path: String,
    corpus_class: String,
    encoder: String,
    source_fixture: String,
    generation_command: String,
    license: String,
    container_support: String,
    required: bool,
    max_size_bytes: u64,
    oracle: String,
    bitrate_kbps: u32,
    channels: u32,
    expected_sha256: Option<String>,
}

impl ManifestEntry {
    fn is_required_supported_generated(&self) -> bool {
        self.required
            && self.corpus_class == "generated-non-reference"
            && self.container_support == "supported-ogg-family0"
    }
}

#[derive(Default)]
struct EntryBuilder {
    id: Option<String>,
    path: Option<String>,
    corpus_class: Option<String>,
    encoder: Option<String>,
    source_fixture: Option<String>,
    generation_command: Option<String>,
    license: Option<String>,
    container_support: Option<String>,
    required: Option<bool>,
    max_size_bytes: Option<u64>,
    oracle: Option<String>,
    bitrate_kbps: Option<u32>,
    channels: Option<u32>,
    expected_sha256: Option<String>,
}

pub fn run(options: &Options) -> Outcome {
    let root = workspace_root();
    run_with_root(options, &root)
}

fn run_with_root(options: &Options, root: &Path) -> Outcome {
    match selected_mode(options) {
        Mode::ReportOnly => report_without_claim(root, Mode::ReportOnly),
        Mode::QuickNoClaim => report_without_claim(root, Mode::QuickNoClaim),
        Mode::GeneratedSmoke => run_generated_smoke(root),
    }
}

fn selected_mode(options: &Options) -> Mode {
    if !options.release_preflight {
        Mode::ReportOnly
    } else if options.quick {
        Mode::QuickNoClaim
    } else {
        Mode::GeneratedSmoke
    }
}

fn report_without_claim(root: &Path, mode: Mode) -> Outcome {
    let started = Instant::now();
    let mut issues = Vec::new();
    let manifest = match load_manifest(root) {
        Ok(manifest) => Some(manifest),
        Err(e) => {
            issues.push(format!("real-world corpus manifest unavailable: {e}"));
            None
        }
    };
    let manifest_path = manifest
        .as_ref()
        .map(|m| display_path(&m.path))
        .unwrap_or_else(|| display_path(&root.join(MANIFEST_REL)));
    let entries = manifest
        .as_ref()
        .map(|m| {
            m.entries
                .iter()
                .map(|entry| entry_report_only(root, entry, mode))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let required_entries = entries.iter().filter(|entry| entry.required).count();
    Outcome {
        mode,
        status: Status::NotClaimed,
        claimed: false,
        claim_note: match mode {
            Mode::ReportOnly => {
                "default full-test report-only; real-world corpus coverage is not claimed"
                    .to_string()
            }
            Mode::QuickNoClaim => {
                "quick release-preflight core smoke only; real-world corpus coverage is not claimed"
                    .to_string()
            }
            Mode::GeneratedSmoke => unreachable!("generated mode claims coverage"),
        },
        oracle_note: ORACLE_NOTE,
        manifest_path,
        required_entries,
        compared_required_entries: 0,
        pinned_entries_matched: 0,
        duration_ms: started.elapsed().as_millis() as u64,
        command: Vec::new(),
        ffmpeg_version: None,
        corpus_diff_summary: None,
        entries,
        issues,
        stdout: String::new(),
        stderr: String::new(),
    }
}

fn run_generated_smoke(root: &Path) -> Outcome {
    let runner = RealCommandRunner;
    let probe = detect_ffmpeg(&runner);
    run_generated_smoke_with_runner(root, probe, &runner)
}

#[cfg(test)]
fn run_generated_smoke_with_probe(root: &Path, probe: FfmpegProbe) -> Outcome {
    run_generated_smoke_with_runner(root, probe, &RealCommandRunner)
}

fn run_generated_smoke_with_runner(
    root: &Path,
    probe: FfmpegProbe,
    runner: &dyn CommandRunner,
) -> Outcome {
    let started = Instant::now();
    let mut issues = Vec::new();
    let manifest = match load_manifest(root) {
        Ok(manifest) => manifest,
        Err(e) => {
            return failure_outcome(
                root,
                Vec::new(),
                started,
                Vec::new(),
                None,
                None,
                vec![format!("real-world corpus manifest unavailable: {e}")],
                String::new(),
                String::new(),
                0,
            );
        }
    };
    let required = manifest
        .entries
        .iter()
        .filter(|entry| entry.is_required_supported_generated())
        .collect::<Vec<_>>();
    if required.is_empty() {
        return failure_outcome(
            root,
            manifest_entries(root, &manifest, false, false),
            started,
            Vec::new(),
            None,
            None,
            vec!["manifest has no required generated supported Ogg family-0 entries".to_string()],
            String::new(),
            String::new(),
            0,
        );
    }

    for entry in &required {
        let fixture = root.join(&entry.source_fixture);
        if !fixture.is_file() {
            issues.push(format!(
                "required source fixture for {} is missing: {}",
                entry.id,
                display_path(&fixture)
            ));
        }
    }
    if !issues.is_empty() {
        return failure_outcome(
            root,
            manifest_entries(root, &manifest, false, false),
            started,
            Vec::new(),
            None,
            None,
            issues,
            String::new(),
            String::new(),
            0,
        );
    }

    let ffmpeg_version: Option<String> = match probe {
        FfmpegProbe::Available { version } => Some(version),
        FfmpegProbe::Missing(reason) => {
            return failure_outcome(
                root,
                manifest_entries(root, &manifest, false, false),
                started,
                Vec::new(),
                None,
                None,
                vec![format!(
                    "FFmpeg native opus generation unavailable: {reason}"
                )],
                String::new(),
                String::new(),
                0,
            );
        }
        FfmpegProbe::NoNativeOpus { version, detail } => {
            return failure_outcome(
                root,
                manifest_entries(root, &manifest, false, false),
                started,
                Vec::new(),
                Some(version),
                None,
                vec![format!(
                    "FFmpeg is present but native -c:a opus encoder is unavailable: {detail}"
                )],
                String::new(),
                String::new(),
                0,
            );
        }
    };
    let ffmpeg_label: String = ffmpeg_version.as_deref().unwrap_or("unknown").to_string();

    let temp_parent = root.join("target").join("real_world_corpus");
    if let Err(e) = fs::create_dir_all(&temp_parent) {
        return failure_outcome(
            root,
            manifest_entries(root, &manifest, false, false),
            started,
            Vec::new(),
            ffmpeg_version,
            None,
            vec![format!(
                "failed to create temporary corpus parent {}: {e}",
                display_path(&temp_parent)
            )],
            String::new(),
            String::new(),
            0,
        );
    }
    let tempdir = match tempfile::Builder::new()
        .prefix("ropus-real-world-corpus-")
        .tempdir_in(&temp_parent)
    {
        Ok(dir) => dir,
        Err(e) => {
            return failure_outcome(
                root,
                manifest_entries(root, &manifest, false, false),
                started,
                Vec::new(),
                ffmpeg_version,
                None,
                vec![format!("failed to create temporary corpus directory: {e}")],
                String::new(),
                String::new(),
                0,
            );
        }
    };

    let mut entry_outcomes = Vec::new();
    let mut pinned_matched: usize = 0;
    let pinned_required_total: usize = manifest
        .entries
        .iter()
        .filter(|e| e.is_required_supported_generated() && e.expected_sha256.is_some())
        .count();
    for entry in &manifest.entries {
        if !entry.is_required_supported_generated() {
            entry_outcomes.push(entry_report_only(root, entry, Mode::GeneratedSmoke));
            continue;
        }
        let filename = match Path::new(&entry.path).file_name() {
            Some(name) => name,
            None => {
                issues.push(format!(
                    "manifest entry {} has no output filename",
                    entry.id
                ));
                continue;
            }
        };
        let output_path = tempdir.path().join(filename);
        match generate_entry(runner, root, entry, &output_path) {
            Ok(()) => {
                let meta = fs::metadata(&output_path).ok();
                let bytes = meta.as_ref().map(|m| m.len());
                if let Some(bytes) = bytes
                    && bytes > entry.max_size_bytes
                {
                    issues.push(format!(
                        "{} generated {} bytes, exceeding manifest max_size_bytes {}",
                        entry.id, bytes, entry.max_size_bytes
                    ));
                }
                let sha256 = sha256_of_opus_payload(&output_path).ok();
                let mut entry_status = "generated".to_string();
                let mut entry_note =
                    "generated in a temporary directory; media remains untracked".to_string();
                if let Some(expected) = &entry.expected_sha256 {
                    match &sha256 {
                        Some(produced) if produced == expected => {
                            pinned_matched += 1;
                            entry_status = "pinned_match".to_string();
                            entry_note = format!(
                                "{ORACLE_NOTE}; SHA256 matches manifest pin (ffmpeg={ffmpeg_label})"
                            );
                        }
                        Some(produced) => {
                            issues.push(format!(
                                "pinned SHA256 mismatch for {}: produced={} expected={} (ffmpeg={})",
                                entry.id, produced, expected, ffmpeg_label
                            ));
                            entry_status = "pinned_mismatch".to_string();
                        }
                        None => {
                            issues.push(format!(
                                "{} produced no SHA256 to compare against pinned digest (ffmpeg={})",
                                entry.id, ffmpeg_label
                            ));
                            entry_status = "pinned_unverifiable".to_string();
                        }
                    }
                }
                entry_outcomes.push(EntryOutcome {
                    id: entry.id.clone(),
                    required: entry.required,
                    corpus_class: entry.corpus_class.clone(),
                    path: entry.path.clone(),
                    source_fixture: entry.source_fixture.clone(),
                    encoder: entry.encoder.clone(),
                    container_support: entry.container_support.clone(),
                    oracle: entry.oracle.clone(),
                    local_path_present: root.join(CORPUS_DIR_REL).join(&entry.path).is_file(),
                    generated: true,
                    compared: false,
                    bytes,
                    sha256,
                    status: entry_status,
                    note: entry_note,
                });
            }
            Err(e) => {
                issues.push(format!("failed to generate {}: {e}", entry.id));
                entry_outcomes.push(entry_failure(root, entry, e));
            }
        }
    }

    if !issues.is_empty() {
        return failure_outcome(
            root,
            entry_outcomes,
            started,
            Vec::new(),
            ffmpeg_version,
            None,
            issues,
            String::new(),
            String::new(),
            pinned_matched,
        );
    }

    let command = corpus_diff_command(tempdir.path());
    let argv: Vec<String> = command[1..].to_vec();
    let output = runner.run(&command[0], &argv);
    match output {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
            let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
            let summary = parse_corpus_diff_summary(&stdout);
            let mut issues = Vec::new();
            match output.status.code() {
                Some(0) => {}
                Some(2) => issues.push(format!(
                    "corpus_diff exited 2 (no candidates) under release-preflight; corpus claim cannot be satisfied by an empty directory (ffmpeg={ffmpeg_label})"
                )),
                Some(3) => issues.push(format!(
                    "corpus_diff exited 3 (all candidates deferred); a claimed corpus must include at least one non-deferred entry (ffmpeg={ffmpeg_label})"
                )),
                Some(code) => issues.push(format!(
                    "corpus_diff exited with status {code} (ffmpeg={ffmpeg_label})"
                )),
                None => issues.push(format!(
                    "corpus_diff terminated by signal (ffmpeg={ffmpeg_label})"
                )),
            }
            match &summary {
                Some(summary) if summary.decoded_and_compared >= required.len() => {}
                Some(summary) => issues.push(format!(
                    "corpus_diff decoded-and-compared {} required file(s), expected at least {} (ffmpeg={})",
                    summary.decoded_and_compared,
                    required.len(),
                    ffmpeg_label
                )),
                None => issues.push(format!(
                    "corpus_diff did not emit CORPUS_DIFF_SUMMARY (ffmpeg={ffmpeg_label})"
                )),
            }
            if pinned_matched < pinned_required_total {
                issues.push(format!(
                    "fewer pinned entries matched than required: {pinned_matched}/{pinned_required_total} (ffmpeg={ffmpeg_label})"
                ));
            }
            if issues.is_empty() {
                for entry in &mut entry_outcomes {
                    if entry.required && entry.generated {
                        entry.compared = true;
                        if entry.status != "pinned_match" {
                            entry.status = "matched".to_string();
                            entry.note = ORACLE_NOTE.to_string();
                        }
                    }
                }
                Outcome {
                    mode: Mode::GeneratedSmoke,
                    status: Status::Pass,
                    claimed: true,
                    claim_note:
                        "non-quick release-preflight claims generated real-world corpus gate"
                            .to_string(),
                    oracle_note: ORACLE_NOTE,
                    manifest_path: display_path(&manifest.path),
                    required_entries: required.len(),
                    compared_required_entries: required.len(),
                    pinned_entries_matched: pinned_matched,
                    duration_ms: started.elapsed().as_millis() as u64,
                    command,
                    ffmpeg_version,
                    corpus_diff_summary: summary,
                    entries: entry_outcomes,
                    issues,
                    stdout,
                    stderr,
                }
            } else {
                failure_outcome(
                    root,
                    entry_outcomes,
                    started,
                    command,
                    ffmpeg_version,
                    summary,
                    issues,
                    stdout,
                    stderr,
                    pinned_matched,
                )
            }
        }
        Err(e) => failure_outcome(
            root,
            entry_outcomes,
            started,
            command,
            ffmpeg_version,
            None,
            vec![format!(
                "failed to launch corpus_diff: {e} (ffmpeg={ffmpeg_label})"
            )],
            String::new(),
            String::new(),
            pinned_matched,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn failure_outcome(
    root: &Path,
    entries: Vec<EntryOutcome>,
    started: Instant,
    command: Vec<String>,
    ffmpeg_version: Option<String>,
    corpus_diff_summary: Option<CorpusDiffSummary>,
    issues: Vec<String>,
    stdout: String,
    stderr: String,
    pinned_matched: usize,
) -> Outcome {
    Outcome {
        mode: Mode::GeneratedSmoke,
        status: Status::Fail,
        claimed: true,
        claim_note: "non-quick release-preflight claims generated real-world corpus gate"
            .to_string(),
        oracle_note: ORACLE_NOTE,
        manifest_path: display_path(&root.join(MANIFEST_REL)),
        required_entries: entries.iter().filter(|entry| entry.required).count(),
        compared_required_entries: entries
            .iter()
            .filter(|entry| entry.required && entry.compared)
            .count(),
        pinned_entries_matched: pinned_matched,
        duration_ms: started.elapsed().as_millis() as u64,
        command,
        ffmpeg_version,
        corpus_diff_summary,
        entries,
        issues,
        stdout,
        stderr,
    }
}

fn entry_report_only(root: &Path, entry: &ManifestEntry, mode: Mode) -> EntryOutcome {
    let local_path_present = root.join(CORPUS_DIR_REL).join(&entry.path).is_file();
    let note = match mode {
        Mode::ReportOnly => "report-only; default full-test makes no real-world corpus claim",
        Mode::QuickNoClaim => {
            "report-only; quick release-preflight makes no real-world corpus claim"
        }
        Mode::GeneratedSmoke => "not required by generated smoke gate",
    };
    EntryOutcome {
        id: entry.id.clone(),
        required: entry.required,
        corpus_class: entry.corpus_class.clone(),
        path: entry.path.clone(),
        source_fixture: entry.source_fixture.clone(),
        encoder: entry.encoder.clone(),
        container_support: entry.container_support.clone(),
        oracle: entry.oracle.clone(),
        local_path_present,
        generated: false,
        compared: false,
        bytes: None,
        sha256: None,
        status: if local_path_present {
            "present_local".to_string()
        } else {
            "not_provisioned".to_string()
        },
        note: note.to_string(),
    }
}

fn entry_failure(root: &Path, entry: &ManifestEntry, note: String) -> EntryOutcome {
    let mut out = entry_report_only(root, entry, Mode::GeneratedSmoke);
    out.status = "failed".to_string();
    out.note = note;
    out
}

fn manifest_entries(
    root: &Path,
    manifest: &Manifest,
    generated: bool,
    compared: bool,
) -> Vec<EntryOutcome> {
    manifest
        .entries
        .iter()
        .map(|entry| {
            let mut outcome = entry_report_only(root, entry, Mode::GeneratedSmoke);
            outcome.generated = generated && entry.is_required_supported_generated();
            outcome.compared = compared && entry.is_required_supported_generated();
            outcome
        })
        .collect()
}

fn generate_entry(
    runner: &dyn CommandRunner,
    root: &Path,
    entry: &ManifestEntry,
    output_path: &Path,
) -> Result<(), String> {
    let source = root.join(&entry.source_fixture);
    let source_str = source
        .to_str()
        .ok_or_else(|| format!("non-UTF-8 source path: {}", display_path(&source)))?
        .to_string();
    let output_str = output_path
        .to_str()
        .ok_or_else(|| format!("non-UTF-8 output path: {}", display_path(output_path)))?
        .to_string();
    // The pin contract hashes the Opus packet payload bytes only (see
    // `sha256_of_opus_payload`), so we don't need FFmpeg's bitexact flags to
    // suppress the muxer's randomised stream serial or vendor strings written
    // into OpusTags. Both vary across runs; neither feeds into the digest.
    let args: Vec<String> = vec![
        "-y".to_string(),
        "-hide_banner".to_string(),
        "-loglevel".to_string(),
        "error".to_string(),
        "-i".to_string(),
        source_str,
        "-c:a".to_string(),
        "opus".to_string(),
        "-strict".to_string(),
        "-2".to_string(),
        "-b:a".to_string(),
        format!("{}k", entry.bitrate_kbps),
        "-ac".to_string(),
        format!("{}", entry.channels),
        output_str,
    ];
    let output = runner
        .run("ffmpeg", &args)
        .map_err(|e| format!("failed to launch ffmpeg: {e}"))?;
    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(match output.status.code() {
            Some(code) => format!("ffmpeg exited with status {code}: {}", stderr.trim()),
            None => format!("ffmpeg terminated by signal: {}", stderr.trim()),
        })
    }
}

fn corpus_diff_command(dir: &Path) -> Vec<String> {
    vec![
        "cargo".to_string(),
        "run".to_string(),
        "-p".to_string(),
        "ropus-harness".to_string(),
        "--bin".to_string(),
        "corpus_diff".to_string(),
        "--".to_string(),
        display_path(dir),
    ]
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FfmpegProbe {
    Available { version: String },
    Missing(String),
    NoNativeOpus { version: String, detail: String },
}

fn detect_ffmpeg(runner: &dyn CommandRunner) -> FfmpegProbe {
    let version = match runner.run("ffmpeg", &["-version".to_string()]) {
        Ok(output) if output.status.success() => first_line(&output.stdout, &output.stderr)
            .unwrap_or_else(|| "ffmpeg version unknown".to_string()),
        Ok(output) => {
            let detail = first_line(&output.stdout, &output.stderr)
                .unwrap_or_else(|| "ffmpeg -version failed".to_string());
            return FfmpegProbe::Missing(detail);
        }
        Err(e) => return FfmpegProbe::Missing(e.to_string()),
    };
    let encoders = match runner.run(
        "ffmpeg",
        &["-hide_banner".to_string(), "-encoders".to_string()],
    ) {
        Ok(output) if output.status.success() => {
            let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
            text.push_str(&String::from_utf8_lossy(&output.stderr));
            text
        }
        Ok(output) => {
            let detail = first_line(&output.stdout, &output.stderr)
                .unwrap_or_else(|| "ffmpeg -encoders failed".to_string());
            return FfmpegProbe::NoNativeOpus { version, detail };
        }
        Err(e) => {
            return FfmpegProbe::NoNativeOpus {
                version,
                detail: e.to_string(),
            };
        }
    };
    if has_native_opus_encoder(&encoders) {
        FfmpegProbe::Available { version }
    } else {
        FfmpegProbe::NoNativeOpus {
            version,
            detail: "encoder list did not contain an encoder named exactly `opus`".to_string(),
        }
    }
}

fn has_native_opus_encoder(encoders: &str) -> bool {
    encoders.lines().any(|line| {
        let mut parts = line.split_whitespace();
        let flags = parts.next();
        let name = parts.next();
        matches!((flags, name), (Some(flags), Some("opus")) if flags.starts_with('A'))
    })
}

fn first_line(stdout: &[u8], stderr: &[u8]) -> Option<String> {
    let mut text = String::from_utf8_lossy(stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(stderr));
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(ToOwned::to_owned)
}

/// Hash an `.opus` file's encoder output, ignoring framing and metadata.
///
/// Walks the Ogg pages of `path`, drops the first two pages (`OpusHead` and
/// `OpusTags` per RFC 7845 §3) — both depend on FFmpeg version, muxer state,
/// and vendor strings — then concatenates the page payload bytes from every
/// remaining page and hashes the result. The pin therefore fires only on
/// real encoder drift, not on muxer-layer variation (FFmpeg randomises the
/// Ogg stream serial number per run).
fn sha256_of_opus_payload(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| format!("reading {}: {e}", display_path(path)))?;
    let payload = read_opus_payload(&bytes)?;
    Ok(sha256_hex_of_bytes(&payload))
}

/// Extract the concatenated Opus packet payload bytes from an in-memory Ogg
/// stream, skipping the first two pages (OpusHead + OpusTags).
fn read_opus_payload(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut page_idx = 0usize;
    while i < data.len() {
        if i + 27 > data.len() {
            return Err(format!(
                "truncated Ogg page header at offset {i} (page {page_idx})"
            ));
        }
        if &data[i..i + 4] != b"OggS" {
            return Err(format!(
                "missing OggS capture pattern at offset {i} (page {page_idx})"
            ));
        }
        let n_segments = data[i + 26] as usize;
        let segments_end = i + 27 + n_segments;
        if segments_end > data.len() {
            return Err(format!(
                "truncated segment table at offset {i} (page {page_idx})"
            ));
        }
        let payload_len: usize = data[i + 27..segments_end].iter().map(|b| *b as usize).sum();
        let payload_end = segments_end + payload_len;
        if payload_end > data.len() {
            return Err(format!(
                "truncated payload at offset {} (page {page_idx})",
                segments_end
            ));
        }
        if page_idx >= 2 {
            out.extend_from_slice(&data[segments_end..payload_end]);
        }
        page_idx += 1;
        i = payload_end;
    }
    if page_idx < 3 {
        return Err(format!(
            "Ogg stream has only {page_idx} page(s); expected at least one audio page after OpusHead+OpusTags"
        ));
    }
    Ok(out)
}

fn sha256_hex_of_bytes(data: &[u8]) -> String {
    let digest = Sha256::digest(data);
    let mut hex = String::with_capacity(64);
    for byte in digest.iter() {
        hex.push_str(&format!("{:02x}", byte));
    }
    hex
}

fn parse_corpus_diff_summary(stdout: &str) -> Option<CorpusDiffSummary> {
    let line = stdout
        .lines()
        .find(|line| line.starts_with("CORPUS_DIFF_SUMMARY "))?;
    let mut summary = CorpusDiffSummary {
        candidates: 0,
        decoded_and_compared: 0,
        zero_audio: 0,
        skipped: 0,
        deferred: 0,
        mismatched: 0,
        panicked: 0,
    };
    for part in line["CORPUS_DIFF_SUMMARY ".len()..].split_whitespace() {
        let (key, value) = part.split_once('=')?;
        let value = value.parse::<usize>().ok()?;
        match key {
            "candidates" => summary.candidates = value,
            "decoded_and_compared" => summary.decoded_and_compared = value,
            "zero_audio" => summary.zero_audio = value,
            "skipped" => summary.skipped = value,
            "deferred" => summary.deferred = value,
            "mismatched" => summary.mismatched = value,
            "panicked" => summary.panicked = value,
            _ => return None,
        }
    }
    Some(summary)
}

fn load_manifest(root: &Path) -> Result<Manifest, String> {
    let path = root.join(MANIFEST_REL);
    let text =
        fs::read_to_string(&path).map_err(|e| format!("reading {}: {e}", display_path(&path)))?;
    let entries = parse_manifest_entries(&text)?;
    if entries.is_empty() {
        return Err("manifest has no entries".to_string());
    }
    Ok(Manifest { path, entries })
}

fn parse_manifest_entries(text: &str) -> Result<Vec<ManifestEntry>, String> {
    let mut entries = Vec::new();
    let mut current: Option<EntryBuilder> = None;

    for (line_no, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "[[entries]]" {
            if let Some(builder) = current.take() {
                entries.push(builder.finish(line_no)?);
            }
            current = Some(EntryBuilder::default());
            continue;
        }
        let Some(builder) = current.as_mut() else {
            return Err(format!(
                "line {} appears before first [[entries]] block",
                line_no + 1
            ));
        };
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| format!("line {} is not key = value", line_no + 1))?;
        builder.set(key.trim(), value.trim(), line_no + 1)?;
    }

    if let Some(builder) = current.take() {
        entries.push(builder.finish(text.lines().count() + 1)?);
    }
    Ok(entries)
}

impl EntryBuilder {
    fn set(&mut self, key: &str, value: &str, line_no: usize) -> Result<(), String> {
        match key {
            "id" => self.id = Some(parse_string(value, line_no)?),
            "path" => self.path = Some(parse_string(value, line_no)?),
            "class" => self.corpus_class = Some(parse_string(value, line_no)?),
            "encoder" => self.encoder = Some(parse_string(value, line_no)?),
            "source_fixture" => self.source_fixture = Some(parse_string(value, line_no)?),
            "generation_command" => self.generation_command = Some(parse_string(value, line_no)?),
            "license" => self.license = Some(parse_string(value, line_no)?),
            "container_support" => self.container_support = Some(parse_string(value, line_no)?),
            "required" => self.required = Some(parse_bool(value, line_no)?),
            "max_size_bytes" => self.max_size_bytes = Some(parse_u64(value, line_no)?),
            "oracle" => self.oracle = Some(parse_string(value, line_no)?),
            "bitrate_kbps" => {
                let n = parse_u64(value, line_no)?;
                if !(6..=510).contains(&n) {
                    return Err(format!(
                        "line {line_no} bitrate_kbps must be in 6..=510, got {n}"
                    ));
                }
                self.bitrate_kbps = Some(n as u32);
            }
            "channels" => {
                let n = parse_u64(value, line_no)?;
                if n != 1 && n != 2 {
                    return Err(format!("line {line_no} channels must be 1 or 2, got {n}"));
                }
                self.channels = Some(n as u32);
            }
            "expected_sha256" => {
                let raw = parse_string(value, line_no)?;
                if raw.len() != 64
                    || !raw
                        .chars()
                        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
                {
                    return Err(format!(
                        "line {line_no} expected_sha256 must be a lowercase 64-hex-char string"
                    ));
                }
                self.expected_sha256 = Some(raw);
            }
            other => return Err(format!("line {line_no} has unknown key `{other}`")),
        }
        Ok(())
    }

    fn finish(self, line_no: usize) -> Result<ManifestEntry, String> {
        Ok(ManifestEntry {
            id: required_field(self.id, "id", line_no)?,
            path: required_field(self.path, "path", line_no)?,
            corpus_class: required_field(self.corpus_class, "class", line_no)?,
            encoder: required_field(self.encoder, "encoder", line_no)?,
            source_fixture: required_field(self.source_fixture, "source_fixture", line_no)?,
            generation_command: required_field(
                self.generation_command,
                "generation_command",
                line_no,
            )?,
            license: required_field(self.license, "license", line_no)?,
            container_support: required_field(
                self.container_support,
                "container_support",
                line_no,
            )?,
            required: required_field(self.required, "required", line_no)?,
            max_size_bytes: required_field(self.max_size_bytes, "max_size_bytes", line_no)?,
            oracle: required_field(self.oracle, "oracle", line_no)?,
            bitrate_kbps: required_field(self.bitrate_kbps, "bitrate_kbps", line_no)?,
            channels: required_field(self.channels, "channels", line_no)?,
            expected_sha256: self.expected_sha256,
        })
    }
}

fn parse_string(value: &str, line_no: usize) -> Result<String, String> {
    let value = value
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .ok_or_else(|| format!("line {line_no} expected a quoted string"))?;
    Ok(value.to_string())
}

fn parse_bool(value: &str, line_no: usize) -> Result<bool, String> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!("line {line_no} expected true or false")),
    }
}

fn parse_u64(value: &str, line_no: usize) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|e| format!("line {line_no} expected unsigned integer: {e}"))
}

fn required_field<T>(value: Option<T>, name: &str, line_no: usize) -> Result<T, String> {
    value.ok_or_else(|| format!("entry ending before line {line_no} missing `{name}`"))
}

fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().map(Path::to_path_buf).unwrap_or(manifest)
}

fn display_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write(root: &Path, rel: &str, body: &str) {
        let path = root.join(rel);
        fs::create_dir_all(path.parent().expect("fixture has parent")).expect("mkdir");
        fs::write(path, body).expect("write fixture");
    }

    fn manifest_text() -> &'static str {
        r#"
[[entries]]
id = "ffmpeg-native-sine-32k"
path = "ffmpeg-native-sine-32k.opus"
class = "generated-non-reference"
encoder = "FFmpeg native opus"
source_fixture = "tests/vectors/48k_sine1k_loud.wav"
generation_command = "ffmpeg -y -hide_banner -loglevel error -i tests/vectors/48k_sine1k_loud.wav -c:a opus -strict -2 -b:a 32k -ac 1 tests/vectors/real_world/ffmpeg-native-sine-32k.opus"
license = "Generated locally from checked-in synthetic WAV fixture; generated media remains untracked."
container_support = "supported-ogg-family0"
required = true
max_size_bytes = 100000
oracle = "exact-pcm-parity-vs-c-reference"
bitrate_kbps = 32
channels = 1
"#
    }

    fn write_manifest(root: &Path) {
        write(root, MANIFEST_REL, manifest_text());
    }

    #[test]
    fn manifest_parser_loads_required_generated_entry() {
        let entries = parse_manifest_entries(manifest_text()).expect("parse manifest");

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, "ffmpeg-native-sine-32k");
        assert!(entries[0].is_required_supported_generated());
        assert_eq!(entries[0].max_size_bytes, 100000);
    }

    #[test]
    fn default_report_only_does_not_claim_or_fail_when_manifest_is_absent() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let outcome = run_with_root(&Options::default(), tmp.path());

        assert_eq!(outcome.mode, Mode::ReportOnly);
        assert_eq!(outcome.status, Status::NotClaimed);
        assert!(!outcome.claimed);
        assert!(!outcome.banner_fail());
        assert!(outcome.claim_note.contains("not claimed"));
    }

    #[test]
    fn quick_release_preflight_explicitly_makes_no_corpus_claim() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write_manifest(tmp.path());
        let options = Options {
            quick: true,
            release_preflight: true,
            ..Options::default()
        };
        let outcome = run_with_root(&options, tmp.path());

        assert_eq!(outcome.mode, Mode::QuickNoClaim);
        assert_eq!(outcome.status, Status::NotClaimed);
        assert!(!outcome.claimed);
        assert!(!outcome.banner_fail());
        assert!(outcome.claim_note.contains("not claimed"));
        assert_eq!(outcome.required_entries, 1);
    }

    #[test]
    fn non_quick_release_preflight_missing_manifest_fails_claim() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let options = Options {
            release_preflight: true,
            ..Options::default()
        };
        let outcome = run_with_root(&options, tmp.path());

        assert_eq!(outcome.mode, Mode::GeneratedSmoke);
        assert_eq!(outcome.status, Status::Fail);
        assert!(outcome.claimed);
        assert!(outcome.banner_fail());
        assert!(outcome.issues[0].contains("manifest unavailable"));
    }

    #[test]
    fn non_quick_release_preflight_missing_ffmpeg_fails_as_tooling() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write_manifest(tmp.path());
        write(tmp.path(), "tests/vectors/48k_sine1k_loud.wav", "fake wav");

        let outcome = run_generated_smoke_with_probe(
            tmp.path(),
            FfmpegProbe::Missing("not found on PATH".to_string()),
        );

        assert_eq!(outcome.status, Status::Fail);
        assert!(outcome.claimed);
        assert!(outcome.banner_fail());
        assert!(
            outcome
                .issues
                .iter()
                .any(|issue| issue.contains("FFmpeg native opus generation unavailable"))
        );
    }

    #[test]
    fn ffmpeg_without_native_opus_fails_as_missing_encoder() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write_manifest(tmp.path());
        write(tmp.path(), "tests/vectors/48k_sine1k_loud.wav", "fake wav");

        let outcome = run_generated_smoke_with_probe(
            tmp.path(),
            FfmpegProbe::NoNativeOpus {
                version: "ffmpeg version test".to_string(),
                detail: "encoder list did not contain opus".to_string(),
            },
        );

        assert_eq!(outcome.status, Status::Fail);
        assert_eq!(
            outcome.ffmpeg_version.as_deref(),
            Some("ffmpeg version test")
        );
        assert!(
            outcome
                .issues
                .iter()
                .any(|issue| issue.contains("native -c:a opus encoder is unavailable"))
        );
    }

    #[test]
    fn native_opus_encoder_detection_ignores_libopus() {
        let encoders =
            " A..... libopus              libopus Opus\n V..... png                  PNG";
        assert!(!has_native_opus_encoder(encoders));

        let encoders =
            " A..... opus                 Opus\n A..... libopus              libopus Opus";
        assert!(has_native_opus_encoder(encoders));
    }

    #[test]
    fn parses_corpus_diff_summary_line() {
        let summary = parse_corpus_diff_summary(
            "noise\nCORPUS_DIFF_SUMMARY candidates=1 decoded_and_compared=1 zero_audio=0 skipped=0 deferred=0 mismatched=0 panicked=0\n",
        )
        .expect("summary");

        assert_eq!(
            summary,
            CorpusDiffSummary {
                candidates: 1,
                decoded_and_compared: 1,
                zero_audio: 0,
                skipped: 0,
                deferred: 0,
                mismatched: 0,
                panicked: 0,
            }
        );
    }

    #[test]
    fn parse_corpus_diff_summary_reads_deferred_key() {
        let summary = parse_corpus_diff_summary(
            "CORPUS_DIFF_SUMMARY candidates=4 decoded_and_compared=1 zero_audio=0 skipped=0 deferred=2 mismatched=1 panicked=0\n",
        )
        .expect("summary");
        assert_eq!(summary.deferred, 2);
        assert_eq!(summary.candidates, 4);
        assert_eq!(summary.mismatched, 1);

        let older = parse_corpus_diff_summary(
            "CORPUS_DIFF_SUMMARY candidates=1 decoded_and_compared=1 zero_audio=0 skipped=0 mismatched=0 panicked=0\n",
        )
        .expect("legacy summary");
        assert_eq!(older.deferred, 0);
    }

    #[test]
    fn manifest_parser_accepts_expected_sha256() {
        let body = format!(
            r#"
[[entries]]
id = "ffmpeg-native-sine-32k"
path = "ffmpeg-native-sine-32k.opus"
class = "generated-non-reference"
encoder = "FFmpeg native opus"
source_fixture = "tests/vectors/48k_sine1k_loud.wav"
generation_command = "noop"
license = "local"
container_support = "supported-ogg-family0"
required = true
max_size_bytes = 100000
oracle = "exact-pcm-parity-vs-c-reference"
bitrate_kbps = 32
channels = 1
expected_sha256 = "{}"
"#,
            "0".repeat(64)
        );
        let entries = parse_manifest_entries(&body).expect("parse manifest");
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].expected_sha256.as_deref(),
            Some(&*"0".repeat(64))
        );
    }

    #[test]
    fn manifest_parser_rejects_malformed_expected_sha256() {
        let bad_hex = r#"
[[entries]]
id = "x"
path = "x.opus"
class = "generated-non-reference"
encoder = "FFmpeg native opus"
source_fixture = "tests/vectors/48k_sine1k_loud.wav"
generation_command = "noop"
license = "local"
container_support = "supported-ogg-family0"
required = true
max_size_bytes = 100000
oracle = "exact-pcm-parity-vs-c-reference"
bitrate_kbps = 32
channels = 1
expected_sha256 = "not-hex-and-too-short"
"#;
        let err = parse_manifest_entries(bad_hex).expect_err("must reject");
        assert!(err.contains("expected_sha256"), "err={err}");
        assert!(err.contains("line "), "err={err}");

        let wrong_len = format!(
            r#"
[[entries]]
id = "x"
path = "x.opus"
class = "generated-non-reference"
encoder = "FFmpeg native opus"
source_fixture = "tests/vectors/48k_sine1k_loud.wav"
generation_command = "noop"
license = "local"
container_support = "supported-ogg-family0"
required = true
max_size_bytes = 100000
oracle = "exact-pcm-parity-vs-c-reference"
bitrate_kbps = 32
channels = 1
expected_sha256 = "{}"
"#,
            "abc"
        );
        let err = parse_manifest_entries(&wrong_len).expect_err("must reject short");
        assert!(err.contains("expected_sha256"), "err={err}");
    }

    #[test]
    fn command_runner_seam_threads_through_generate_entry() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write(tmp.path(), "tests/vectors/48k_sine1k_loud.wav", "fake wav");
        let entry = ManifestEntry {
            id: "x".to_string(),
            path: "x.opus".to_string(),
            corpus_class: "generated-non-reference".to_string(),
            encoder: "FFmpeg native opus".to_string(),
            source_fixture: "tests/vectors/48k_sine1k_loud.wav".to_string(),
            generation_command: "noop".to_string(),
            license: "local".to_string(),
            container_support: "supported-ogg-family0".to_string(),
            required: true,
            max_size_bytes: 100000,
            oracle: "exact-pcm-parity-vs-c-reference".to_string(),
            bitrate_kbps: 24,
            channels: 1,
            expected_sha256: None,
        };
        let out = tmp.path().join("x.opus");
        let runner = RecordingRunner::new_writing(b"opusbytes");
        let res = generate_entry(&runner, tmp.path(), &entry, &out);
        assert!(res.is_ok(), "generate_entry: {:?}", res);
        let calls = runner.calls();
        assert_eq!(calls.len(), 1, "expected exactly one ffmpeg call");
        let (program, args) = &calls[0];
        assert_eq!(program, "ffmpeg");
        assert!(args.iter().any(|a| a == "-b:a"));
        let bitrate_idx = args.iter().position(|a| a == "-b:a").unwrap();
        assert_eq!(args[bitrate_idx + 1], "24k");
        let ac_idx = args.iter().position(|a| a == "-ac").unwrap();
        assert_eq!(args[ac_idx + 1], "1");
    }

    #[test]
    fn manifest_parser_validates_bitrate_and_channels() {
        let too_many_channels = r#"
[[entries]]
id = "x"
path = "x.opus"
class = "generated-non-reference"
encoder = "FFmpeg native opus"
source_fixture = "tests/vectors/48k_sine1k_loud.wav"
generation_command = "noop"
license = "local"
container_support = "supported-ogg-family0"
required = true
max_size_bytes = 100000
oracle = "exact-pcm-parity-vs-c-reference"
bitrate_kbps = 32
channels = 3
expected_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
"#;
        let err = parse_manifest_entries(too_many_channels).expect_err("reject channels=3");
        assert!(err.contains("channels"), "err={err}");

        let zero_bitrate = r#"
[[entries]]
id = "x"
path = "x.opus"
class = "generated-non-reference"
encoder = "FFmpeg native opus"
source_fixture = "tests/vectors/48k_sine1k_loud.wav"
generation_command = "noop"
license = "local"
container_support = "supported-ogg-family0"
required = true
max_size_bytes = 100000
oracle = "exact-pcm-parity-vs-c-reference"
bitrate_kbps = 0
channels = 1
expected_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
"#;
        let err = parse_manifest_entries(zero_bitrate).expect_err("reject bitrate=0");
        assert!(err.contains("bitrate_kbps"), "err={err}");

        let ok = r#"
[[entries]]
id = "x"
path = "x.opus"
class = "generated-non-reference"
encoder = "FFmpeg native opus"
source_fixture = "tests/vectors/48k_sine1k_loud.wav"
generation_command = "noop"
license = "local"
container_support = "supported-ogg-family0"
required = true
max_size_bytes = 100000
oracle = "exact-pcm-parity-vs-c-reference"
bitrate_kbps = 510
channels = 2
"#;
        let entries = parse_manifest_entries(ok).expect("accepts valid bounds");
        assert_eq!(entries[0].bitrate_kbps, 510);
        assert_eq!(entries[0].channels, 2);
    }

    #[cfg(unix)]
    #[test]
    fn pinned_entry_sha256_match_is_pass() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write(tmp.path(), "tests/vectors/48k_sine1k_loud.wav", "fake wav");
        let audio_payload = b"deterministic-payload-A".to_vec();
        let expected = sha256_hex(&audio_payload);
        let ogg = synthetic_ogg_stream(&audio_payload);
        let manifest_body = format!(
            r#"
[[entries]]
id = "ffmpeg-native-sine-32k"
path = "ffmpeg-native-sine-32k.opus"
class = "generated-non-reference"
encoder = "FFmpeg native opus"
source_fixture = "tests/vectors/48k_sine1k_loud.wav"
generation_command = "noop"
license = "local"
container_support = "supported-ogg-family0"
required = true
max_size_bytes = 100000
oracle = "exact-pcm-parity-vs-c-reference"
bitrate_kbps = 32
channels = 1
expected_sha256 = "{expected}"
"#,
        );
        write(tmp.path(), MANIFEST_REL, &manifest_body);
        let runner = ScriptedRunner::new(&ogg, b"CORPUS_DIFF_SUMMARY candidates=1 decoded_and_compared=1 zero_audio=0 skipped=0 deferred=0 mismatched=0 panicked=0\n");
        let outcome = run_generated_smoke_with_runner(
            tmp.path(),
            FfmpegProbe::Available {
                version: "ffmpeg version test".to_string(),
            },
            &runner,
        );
        assert_eq!(outcome.status, Status::Pass, "issues={:?}", outcome.issues);
        assert_eq!(outcome.pinned_entries_matched, 1);
    }

    #[cfg(unix)]
    #[test]
    fn pinned_entry_sha256_mismatch_is_fail_with_ffmpeg_version() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write(tmp.path(), "tests/vectors/48k_sine1k_loud.wav", "fake wav");
        let audio_payload = b"actual-payload-B".to_vec();
        let actual_digest = sha256_hex(&audio_payload);
        let ogg = synthetic_ogg_stream(&audio_payload);
        let pinned_digest = "1".repeat(64);
        let manifest_body = format!(
            r#"
[[entries]]
id = "ffmpeg-native-sine-32k"
path = "ffmpeg-native-sine-32k.opus"
class = "generated-non-reference"
encoder = "FFmpeg native opus"
source_fixture = "tests/vectors/48k_sine1k_loud.wav"
generation_command = "noop"
license = "local"
container_support = "supported-ogg-family0"
required = true
max_size_bytes = 100000
oracle = "exact-pcm-parity-vs-c-reference"
bitrate_kbps = 32
channels = 1
expected_sha256 = "{pinned_digest}"
"#,
        );
        write(tmp.path(), MANIFEST_REL, &manifest_body);
        let runner = ScriptedRunner::new(&ogg, b"CORPUS_DIFF_SUMMARY candidates=1 decoded_and_compared=1 zero_audio=0 skipped=0 deferred=0 mismatched=0 panicked=0\n");
        let outcome = run_generated_smoke_with_runner(
            tmp.path(),
            FfmpegProbe::Available {
                version: "ffmpeg version 9.9.9-test".to_string(),
            },
            &runner,
        );
        assert_eq!(outcome.status, Status::Fail);
        assert_eq!(outcome.pinned_entries_matched, 0);
        let combined = outcome.issues.join("\n");
        assert!(
            combined.contains(&actual_digest),
            "missing actual: {combined}"
        );
        assert!(
            combined.contains(&pinned_digest),
            "missing pinned: {combined}"
        );
        assert!(
            combined.contains("ffmpeg version 9.9.9-test"),
            "missing version: {combined}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn corpus_diff_exit_two_under_release_preflight_is_fail() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write(tmp.path(), "tests/vectors/48k_sine1k_loud.wav", "fake wav");
        let manifest_body = unpinned_manifest();
        write(tmp.path(), MANIFEST_REL, &manifest_body);
        let runner = ScriptedRunner::new_with_status(b"data", b"", 2);
        let outcome = run_generated_smoke_with_runner(
            tmp.path(),
            FfmpegProbe::Available {
                version: "ffmpeg version test".to_string(),
            },
            &runner,
        );
        assert_eq!(outcome.status, Status::Fail);
        assert!(
            outcome.issues.iter().any(|i| i.contains("no candidates")),
            "issues: {:?}",
            outcome.issues
        );
    }

    #[cfg(unix)]
    #[test]
    fn corpus_diff_exit_three_under_release_preflight_is_fail() {
        let tmp = tempfile::tempdir().expect("tempdir");
        write(tmp.path(), "tests/vectors/48k_sine1k_loud.wav", "fake wav");
        let manifest_body = unpinned_manifest();
        write(tmp.path(), MANIFEST_REL, &manifest_body);
        let runner = ScriptedRunner::new_with_status(
            b"data",
            b"CORPUS_DIFF_SUMMARY candidates=2 decoded_and_compared=0 zero_audio=0 skipped=0 deferred=2 mismatched=0 panicked=0\n",
            3,
        );
        let outcome = run_generated_smoke_with_runner(
            tmp.path(),
            FfmpegProbe::Available {
                version: "ffmpeg version test".to_string(),
            },
            &runner,
        );
        assert_eq!(outcome.status, Status::Fail);
        assert!(
            outcome
                .issues
                .iter()
                .any(|i| i.contains("all candidates deferred")),
            "issues: {:?}",
            outcome.issues
        );
    }

    #[test]
    fn sha256_hex_of_bytes_matches_fips_180_2_vectors() {
        // FIPS-180-2 (NIST) test vectors for SHA-256.
        assert_eq!(
            sha256_hex_of_bytes(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex_of_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // Multi-block FIPS test vector (length 56 bytes — straddles the
        // padding/length boundary that trips naive implementations).
        let multi_block = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".as_ref();
        assert_eq!(
            sha256_hex_of_bytes(multi_block),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn read_opus_payload_returns_only_audio_pages() {
        // Two header pages + one audio page: payload extraction returns the
        // audio payload only.
        let audio = b"audio-bytes";
        let stream = synthetic_ogg_stream(audio);
        let extracted = read_opus_payload(&stream).expect("parse 3-page stream");
        assert_eq!(extracted, audio);
    }

    #[test]
    fn read_opus_payload_concatenates_multiple_audio_pages() {
        let mut stream = Vec::new();
        stream.extend_from_slice(&make_ogg_page(
            b"OpusHead\x01\x01\x00\x00\x80\xbb\x00\x00\x00\x00\x00",
            0,
        ));
        stream.extend_from_slice(&make_ogg_page(b"OpusTags", 1));
        stream.extend_from_slice(&make_ogg_page(b"audio-1", 2));
        stream.extend_from_slice(&make_ogg_page(b"audio-2", 3));
        let extracted = read_opus_payload(&stream).expect("parse 4-page stream");
        assert_eq!(extracted, b"audio-1audio-2".to_vec());
    }

    #[test]
    fn read_opus_payload_errors_on_truncated_header() {
        // Less than 27 bytes (the minimum Ogg page header length).
        assert!(read_opus_payload(b"OggS\x00\x00\x00").is_err());
    }

    #[test]
    fn read_opus_payload_errors_on_missing_capture_pattern() {
        // 27 bytes (header-length minimum) of zeros: passes the length
        // check but fails the OggS magic check.
        let stream = vec![0u8; 27];
        let err = read_opus_payload(&stream).expect_err("must reject");
        assert!(err.contains("OggS"), "err={err}");
    }

    #[test]
    fn live_manifest_required_id_set_is_exact() {
        // Reads the live committed manifest and asserts the required-id set
        // matches exactly. R4 mitigation: a future contributor cannot quietly
        // expand the corpus claim by adding a fourth required entry.
        let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .to_path_buf();
        let manifest_path = workspace_root.join(MANIFEST_REL);
        let text = fs::read_to_string(&manifest_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", manifest_path.display()));
        let entries = parse_manifest_entries(&text).expect("parse live manifest");
        let mut required_ids: Vec<String> = entries
            .iter()
            .filter(|e| e.required)
            .map(|e| e.id.clone())
            .collect();
        required_ids.sort();
        let mut expected: Vec<String> = vec![
            "ffmpeg-native-sine-32k".to_string(),
            "ffmpeg-native-sine-24k-mono".to_string(),
            "ffmpeg-native-speech-40k-mono".to_string(),
        ];
        expected.sort();
        assert_eq!(
            required_ids, expected,
            "required-id set drifted; HLD signoff required for any new required entry"
        );
    }

    fn unpinned_manifest() -> String {
        r#"
[[entries]]
id = "ffmpeg-native-sine-32k"
path = "ffmpeg-native-sine-32k.opus"
class = "generated-non-reference"
encoder = "FFmpeg native opus"
source_fixture = "tests/vectors/48k_sine1k_loud.wav"
generation_command = "noop"
license = "local"
container_support = "supported-ogg-family0"
required = true
max_size_bytes = 100000
oracle = "exact-pcm-parity-vs-c-reference"
bitrate_kbps = 32
channels = 1
"#
        .to_string()
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        sha256_hex_of_bytes(bytes)
    }

    /// Build a minimal Ogg page wrapping `payload`, with sequence number
    /// `seq`. Sufficient for the parser under test; CRC is left zero (the
    /// parser does not validate CRC).
    fn make_ogg_page(payload: &[u8], seq: u32) -> Vec<u8> {
        let mut page = Vec::new();
        page.extend_from_slice(b"OggS"); // capture pattern
        page.push(0); // stream structure version
        page.push(0); // header_type
        page.extend_from_slice(&0u64.to_le_bytes()); // granule position
        page.extend_from_slice(&0u32.to_le_bytes()); // serial number
        page.extend_from_slice(&seq.to_le_bytes()); // page sequence number
        page.extend_from_slice(&0u32.to_le_bytes()); // crc placeholder
        // Segment table: split payload into 255-byte chunks.
        let mut remaining = payload.len();
        let mut seg_lengths: Vec<u8> = Vec::new();
        while remaining > 255 {
            seg_lengths.push(255);
            remaining -= 255;
        }
        seg_lengths.push(remaining as u8);
        page.push(seg_lengths.len() as u8);
        page.extend_from_slice(&seg_lengths);
        page.extend_from_slice(payload);
        page
    }

    /// Build a 3-page Ogg stream: an OpusHead-like header, an OpusTags-like
    /// header, and one audio page carrying `audio_payload`. The parser under
    /// test only inspects page framing, not magic strings, so dummy header
    /// payloads are sufficient.
    fn synthetic_ogg_stream(audio_payload: &[u8]) -> Vec<u8> {
        let mut stream = Vec::new();
        stream.extend_from_slice(&make_ogg_page(b"OpusHead-stub", 0));
        stream.extend_from_slice(&make_ogg_page(b"OpusTags-stub", 1));
        stream.extend_from_slice(&make_ogg_page(audio_payload, 2));
        stream
    }

    /// Test runner: records every call and writes a fixed payload for any
    /// invocation that includes a final positional arg recognised as an
    /// output path under the tempdir.
    struct RecordingRunner {
        payload: Vec<u8>,
        calls: std::sync::Mutex<Vec<(String, Vec<String>)>>,
    }

    impl RecordingRunner {
        fn new_writing(payload: &[u8]) -> Self {
            Self {
                payload: payload.to_vec(),
                calls: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn calls(&self) -> Vec<(String, Vec<String>)> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl CommandRunner for RecordingRunner {
        fn run(&self, program: &str, args: &[String]) -> std::io::Result<std::process::Output> {
            self.calls
                .lock()
                .unwrap()
                .push((program.to_string(), args.to_vec()));
            // last argv item is the output path for our ffmpeg invocations.
            if let Some(out) = args.last()
                && (out.ends_with(".opus") || out.ends_with(".ogg"))
            {
                fs::write(out, &self.payload).expect("test write payload");
            }
            Ok(std::process::Output {
                status: std::process::ExitStatus::default(),
                stdout: Vec::new(),
                stderr: Vec::new(),
            })
        }
    }

    /// Test runner used for end-to-end smoke tests: writes a fixed payload
    /// to ffmpeg invocations and returns scripted stdout/exit-status to
    /// `corpus_diff` invocations. Routing relies on `program == "ffmpeg"` —
    /// the corpus stage's only two subprocesses are FFmpeg (for generation)
    /// and `cargo run -p ropus-harness --bin corpus_diff` (the comparison
    /// invocation), so this binary check uniquely distinguishes them.
    /// Unix-gated because constructing a synthetic exit code uses the
    /// `os::unix::process::ExitStatusExt` trait; a portable rewrite is out
    /// of scope here.
    #[cfg(unix)]
    struct ScriptedRunner {
        payload: Vec<u8>,
        diff_stdout: Vec<u8>,
        diff_exit: i32,
    }

    #[cfg(unix)]
    impl ScriptedRunner {
        fn new(payload: &[u8], diff_stdout: &[u8]) -> Self {
            Self {
                payload: payload.to_vec(),
                diff_stdout: diff_stdout.to_vec(),
                diff_exit: 0,
            }
        }
        fn new_with_status(payload: &[u8], diff_stdout: &[u8], diff_exit: i32) -> Self {
            Self {
                payload: payload.to_vec(),
                diff_stdout: diff_stdout.to_vec(),
                diff_exit,
            }
        }
    }

    #[cfg(unix)]
    impl CommandRunner for ScriptedRunner {
        fn run(&self, program: &str, args: &[String]) -> std::io::Result<std::process::Output> {
            if program == "ffmpeg" {
                if let Some(out) = args.last()
                    && (out.ends_with(".opus") || out.ends_with(".ogg"))
                {
                    fs::write(out, &self.payload).expect("test write payload");
                }
                return Ok(std::process::Output {
                    status: std::process::ExitStatus::default(),
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                });
            }
            // corpus_diff invocation
            use std::os::unix::process::ExitStatusExt;
            let status = std::process::ExitStatus::from_raw((self.diff_exit & 0xff) << 8);
            Ok(std::process::Output {
                status,
                stdout: self.diff_stdout.clone(),
                stderr: Vec::new(),
            })
        }
    }
}
