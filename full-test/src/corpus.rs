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
use std::process::Command;
use std::time::Instant;

use serde::Serialize;

use crate::cli::Options;

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
    run_generated_smoke_with_probe(root, detect_ffmpeg())
}

fn run_generated_smoke_with_probe(root: &Path, probe: FfmpegProbe) -> Outcome {
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
        );
    }

    let ffmpeg_version = match probe {
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
            );
        }
    };

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
            );
        }
    };

    let mut entry_outcomes = Vec::new();
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
        match generate_entry(root, entry, &output_path) {
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
                let sha256 = sha256_of(&output_path).ok();
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
                    status: "generated".to_string(),
                    note: "generated in a temporary directory; media remains untracked".to_string(),
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
        );
    }

    let command = corpus_diff_command(tempdir.path());
    let output = Command::new(&command[0])
        .args(&command[1..])
        .current_dir(root)
        .output();
    match output {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
            let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
            let summary = parse_corpus_diff_summary(&stdout);
            let mut issues = Vec::new();
            if !output.status.success() {
                issues.push(match output.status.code() {
                    Some(code) => format!("corpus_diff exited with status {code}"),
                    None => "corpus_diff terminated by signal".to_string(),
                });
            }
            match &summary {
                Some(summary) if summary.decoded_and_compared >= required.len() => {}
                Some(summary) => issues.push(format!(
                    "corpus_diff decoded-and-compared {} required file(s), expected at least {}",
                    summary.decoded_and_compared,
                    required.len()
                )),
                None => issues.push("corpus_diff did not emit CORPUS_DIFF_SUMMARY".to_string()),
            }
            if issues.is_empty() {
                for entry in &mut entry_outcomes {
                    if entry.required && entry.generated {
                        entry.compared = true;
                        entry.status = "matched".to_string();
                        entry.note = ORACLE_NOTE.to_string();
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
            vec![format!("failed to launch corpus_diff: {e}")],
            String::new(),
            String::new(),
        ),
    }
}

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

fn generate_entry(root: &Path, entry: &ManifestEntry, output_path: &Path) -> Result<(), String> {
    let source = root.join(&entry.source_fixture);
    let output = Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            source
                .to_str()
                .ok_or_else(|| format!("non-UTF-8 source path: {}", display_path(&source)))?,
            "-c:a",
            "opus",
            "-strict",
            "-2",
            "-b:a",
            "32k",
            output_path
                .to_str()
                .ok_or_else(|| format!("non-UTF-8 output path: {}", display_path(output_path)))?,
        ])
        .current_dir(root)
        .output()
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

fn detect_ffmpeg() -> FfmpegProbe {
    let version = match Command::new("ffmpeg").arg("-version").output() {
        Ok(output) if output.status.success() => first_line(&output.stdout, &output.stderr)
            .unwrap_or_else(|| "ffmpeg version unknown".to_string()),
        Ok(output) => {
            let detail = first_line(&output.stdout, &output.stderr)
                .unwrap_or_else(|| "ffmpeg -version failed".to_string());
            return FfmpegProbe::Missing(detail);
        }
        Err(e) => return FfmpegProbe::Missing(e.to_string()),
    };
    let encoders = match Command::new("ffmpeg")
        .args(["-hide_banner", "-encoders"])
        .output()
    {
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

fn sha256_of(path: &Path) -> Result<String, String> {
    for (program, args) in [
        ("sha256sum", vec![display_path(path)]),
        (
            "shasum",
            vec!["-a".to_string(), "256".to_string(), display_path(path)],
        ),
    ] {
        let output = Command::new(program).args(args).output();
        let output = match output {
            Ok(output) => output,
            Err(_) => continue,
        };
        if !output.status.success() {
            continue;
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        if let Some(hash) = stdout.split_whitespace().next()
            && hash.len() == 64
            && hash.chars().all(|c| c.is_ascii_hexdigit())
        {
            return Ok(hash.to_ascii_lowercase());
        }
    }
    Err("need sha256sum or shasum on PATH".to_string())
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
generation_command = "ffmpeg -y -hide_banner -loglevel error -i tests/vectors/48k_sine1k_loud.wav -c:a opus -strict -2 -b:a 32k tests/vectors/real_world/ffmpeg-native-sine-32k.opus"
license = "Generated locally from checked-in synthetic WAV fixture; generated media remains untracked."
container_support = "supported-ogg-family0"
required = true
max_size_bytes = 100000
oracle = "exact-pcm-parity-vs-c-reference"
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
            "noise\nCORPUS_DIFF_SUMMARY candidates=1 decoded_and_compared=1 zero_audio=0 skipped=0 mismatched=0 panicked=0\n",
        )
        .expect("summary");

        assert_eq!(
            summary,
            CorpusDiffSummary {
                candidates: 1,
                decoded_and_compared: 1,
                zero_audio: 0,
                skipped: 0,
                mismatched: 0,
                panicked: 0,
            }
        );
    }
}
