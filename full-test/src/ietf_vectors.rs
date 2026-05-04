//! IETF RFC 6716 / RFC 8251 vector provisioning for the validation gate.
//!
//! The vectors stay unvendored under `tests/vectors/ietf/`, but `full-test`
//! owns the gate semantics: before Stage 2 can claim conformance, the 12
//! bitstreams and matching reference PCM files must be present or fetched.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::issues;

const FETCH_TIMEOUT: Duration = Duration::from_secs(15 * 60);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProvisionStatus {
    Present,
    Provisioned,
    Unavailable,
}

impl ProvisionStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ProvisionStatus::Present => "present",
            ProvisionStatus::Provisioned => "provisioned",
            ProvisionStatus::Unavailable => "unavailable",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IetfVectorProvision {
    pub status: ProvisionStatus,
    pub attempted_fetch: bool,
    pub script: Option<PathBuf>,
    pub exit_code: Option<i32>,
    pub reason: Option<String>,
}

impl IetfVectorProvision {
    pub fn present() -> Self {
        Self {
            status: ProvisionStatus::Present,
            attempted_fetch: false,
            script: None,
            exit_code: None,
            reason: None,
        }
    }

    fn provisioned(script: PathBuf, exit_code: Option<i32>) -> Self {
        Self {
            status: ProvisionStatus::Provisioned,
            attempted_fetch: true,
            script: Some(script),
            exit_code,
            reason: None,
        }
    }

    fn unavailable(script: PathBuf, attempt: FetchAttempt, post_check: String) -> Self {
        Self {
            status: ProvisionStatus::Unavailable,
            attempted_fetch: true,
            script: Some(script),
            exit_code: attempt.exit_code,
            reason: Some(format_unavailable_reason(&attempt, &post_check)),
        }
    }

    pub fn available(&self) -> bool {
        matches!(
            self.status,
            ProvisionStatus::Present | ProvisionStatus::Provisioned
        )
    }

    pub fn status_label(&self) -> &'static str {
        self.status.as_str()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchAttempt {
    pub success: bool,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub spawn_error: Option<String>,
    pub timed_out: bool,
}

impl FetchAttempt {
    #[cfg(test)]
    fn success() -> Self {
        Self {
            success: true,
            exit_code: Some(0),
            stdout: String::new(),
            stderr: String::new(),
            spawn_error: None,
            timed_out: false,
        }
    }

    #[cfg(test)]
    fn failure(message: &str) -> Self {
        Self {
            success: false,
            exit_code: Some(7),
            stdout: String::new(),
            stderr: message.to_string(),
            spawn_error: None,
            timed_out: false,
        }
    }
}

/// Ensure the workspace has a complete IETF vector set, fetching it if needed.
pub fn provision(workspace_root: &Path) -> IetfVectorProvision {
    provision_with_runner(workspace_root, run_fetch_script)
}

fn provision_with_runner<F>(workspace_root: &Path, mut run_fetch: F) -> IetfVectorProvision
where
    F: FnMut(&Path, &Path) -> FetchAttempt,
{
    let vectors = vectors_dir(workspace_root);
    if vector_set_complete(&vectors) {
        return IetfVectorProvision::present();
    }

    let script = fetch_script_path(workspace_root);
    let attempt = run_fetch(workspace_root, &script);
    if attempt.success && vector_set_complete(&vectors) {
        return IetfVectorProvision::provisioned(script, attempt.exit_code);
    }

    let post_check = missing_vector_summary(&vectors);
    IetfVectorProvision::unavailable(script, attempt, post_check)
}

pub fn vectors_dir(workspace_root: &Path) -> PathBuf {
    workspace_root.join("tests").join("vectors").join("ietf")
}

/// Complete means all 12 bitstreams and at least one reference PCM per vector.
pub fn vector_set_complete(root: &Path) -> bool {
    missing_vector_entries(root).is_empty()
}

fn missing_vector_summary(root: &Path) -> String {
    let missing = missing_vector_entries(root);
    if missing.is_empty() {
        "post-fetch vector check passed".to_string()
    } else {
        let mut text = format!("{} vector file group(s) incomplete", missing.len());
        let preview = missing
            .iter()
            .take(6)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        if !preview.is_empty() {
            text.push_str(": ");
            text.push_str(&preview);
        }
        if missing.len() > 6 {
            text.push_str(", ...");
        }
        text
    }
}

fn missing_vector_entries(root: &Path) -> Vec<String> {
    let mut missing = Vec::new();
    for n in 1..=12 {
        let stem = format!("testvector{n:02}");
        if !root.join(format!("{stem}.bit")).is_file() {
            missing.push(format!("{stem}.bit"));
        }
        let primary = root.join(format!("{stem}.dec"));
        let alt = root.join(format!("{stem}m.dec"));
        if !primary.is_file() && !alt.is_file() {
            missing.push(format!("{stem}.dec or {stem}m.dec"));
        }
    }
    missing
}

fn fetch_script_path(workspace_root: &Path) -> PathBuf {
    let name = if cfg!(windows) {
        "fetch_ietf_vectors.ps1"
    } else {
        "fetch_ietf_vectors.sh"
    };
    workspace_root.join("tools").join(name)
}

fn run_fetch_script(workspace_root: &Path, script: &Path) -> FetchAttempt {
    let mut stdout_file = match tempfile::tempfile() {
        Ok(file) => file,
        Err(e) => {
            return fetch_attempt_spawn_error(format!("failed to create stdout temp file: {e}"));
        }
    };
    let mut stderr_file = match tempfile::tempfile() {
        Ok(file) => file,
        Err(e) => {
            return fetch_attempt_spawn_error(format!("failed to create stderr temp file: {e}"));
        }
    };
    let stdout_for_child = match stdout_file.try_clone() {
        Ok(file) => file,
        Err(e) => {
            return fetch_attempt_spawn_error(format!("failed to clone stdout temp file: {e}"));
        }
    };
    let stderr_for_child = match stderr_file.try_clone() {
        Ok(file) => file,
        Err(e) => {
            return fetch_attempt_spawn_error(format!("failed to clone stderr temp file: {e}"));
        }
    };

    let mut cmd = if cfg!(windows) {
        let mut c = Command::new("powershell");
        c.args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]);
        c.arg(script);
        c
    } else {
        let mut c = Command::new("bash");
        c.arg(script);
        c
    };
    cmd.current_dir(workspace_root)
        .stdout(Stdio::from(stdout_for_child))
        .stderr(Stdio::from(stderr_for_child));

    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(e) => return fetch_attempt_spawn_error(e.to_string()),
    };

    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => {
                return collect_fetch_output(child, &mut stdout_file, &mut stderr_file, false);
            }
            Ok(None) if start.elapsed() >= FETCH_TIMEOUT => {
                let _ = child.kill();
                return collect_fetch_output(child, &mut stdout_file, &mut stderr_file, true);
            }
            Ok(None) => thread::sleep(Duration::from_millis(100)),
            Err(e) => {
                let _ = child.kill();
                return fetch_attempt_spawn_error(format!(
                    "failed while waiting for fetch script: {e}"
                ));
            }
        }
    }
}

fn collect_fetch_output(
    mut child: Child,
    stdout_file: &mut std::fs::File,
    stderr_file: &mut std::fs::File,
    timed_out: bool,
) -> FetchAttempt {
    let status = match child.wait() {
        Ok(status) => status,
        Err(e) => {
            return FetchAttempt {
                success: false,
                exit_code: None,
                stdout: read_capped_tempfile(stdout_file),
                stderr: read_capped_tempfile(stderr_file),
                spawn_error: Some(format!("failed to collect fetch script status: {e}")),
                timed_out,
            };
        }
    };

    let stdout = read_capped_tempfile(stdout_file);
    let stderr = read_capped_tempfile(stderr_file);
    FetchAttempt {
        success: status.success() && !timed_out,
        exit_code: status.code(),
        stdout,
        stderr,
        spawn_error: None,
        timed_out,
    }
}

fn read_capped_tempfile(file: &mut std::fs::File) -> String {
    if file.seek(SeekFrom::Start(0)).is_err() {
        return String::new();
    }
    let mut bytes = Vec::new();
    if file.read_to_end(&mut bytes).is_err() {
        return String::new();
    }
    let (text, _) = issues::cap_stderr(&String::from_utf8_lossy(&bytes));
    text
}

fn fetch_attempt_spawn_error(error: String) -> FetchAttempt {
    FetchAttempt {
        success: false,
        exit_code: None,
        stdout: String::new(),
        stderr: String::new(),
        spawn_error: Some(error),
        timed_out: false,
    }
}

fn format_unavailable_reason(attempt: &FetchAttempt, post_check: &str) -> String {
    if attempt.timed_out {
        return format!(
            "fetch script timed out after {} seconds; {post_check}",
            FETCH_TIMEOUT.as_secs()
        );
    }
    if let Some(e) = &attempt.spawn_error {
        return format!("failed to spawn fetch script: {e}; {post_check}");
    }

    let code = attempt
        .exit_code
        .map(|c| c.to_string())
        .unwrap_or_else(|| "signal".to_string());
    let diagnostic = first_nonempty(&attempt.stderr)
        .or_else(|| first_nonempty(&attempt.stdout))
        .unwrap_or_else(|| "fetch script produced no diagnostic".to_string());

    if attempt.success {
        format!("fetch script exited 0 but vectors are incomplete: {post_check}")
    } else {
        format!("fetch script exited {code}: {diagnostic}; {post_check}")
    }
}

fn first_nonempty(s: &str) -> Option<String> {
    let line = s.lines().find(|line| !line.trim().is_empty())?;
    Some(shorten(line.trim(), 320))
}

fn shorten(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &s[..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn write_complete_vectors(root: &Path) {
        std::fs::create_dir_all(root).expect("vector dir");
        for n in 1..=12 {
            let stem = format!("testvector{n:02}");
            std::fs::write(root.join(format!("{stem}.bit")), b"bit").expect("bit");
            std::fs::write(root.join(format!("{stem}.dec")), b"dec").expect("dec");
        }
    }

    #[test]
    fn complete_requires_bits_and_reference_pcm() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        assert!(!vector_set_complete(tmp.path()));

        for n in 1..=12 {
            let stem = format!("testvector{n:02}");
            std::fs::write(tmp.path().join(format!("{stem}.bit")), b"bit").expect("bit");
        }
        assert!(!vector_set_complete(tmp.path()));

        for n in 1..=12 {
            let stem = format!("testvector{n:02}");
            let path = if n % 2 == 0 {
                tmp.path().join(format!("{stem}m.dec"))
            } else {
                tmp.path().join(format!("{stem}.dec"))
            };
            std::fs::write(path, b"dec").expect("dec");
        }
        assert!(vector_set_complete(tmp.path()));
    }

    #[test]
    fn present_vectors_do_not_invoke_fetch() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let vectors = vectors_dir(tmp.path());
        write_complete_vectors(&vectors);
        let called = Cell::new(false);

        let result = provision_with_runner(tmp.path(), |_, _| {
            called.set(true);
            FetchAttempt::failure("should not run")
        });

        assert_eq!(result.status, ProvisionStatus::Present);
        assert!(!result.attempted_fetch);
        assert!(!called.get());
    }

    #[test]
    fn successful_fetch_that_completes_vectors_is_provisioned() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let vectors = vectors_dir(tmp.path());

        let result = provision_with_runner(tmp.path(), |_, _| {
            write_complete_vectors(&vectors);
            FetchAttempt::success()
        });

        assert_eq!(result.status, ProvisionStatus::Provisioned);
        assert!(result.available());
        assert!(result.attempted_fetch);
        assert_eq!(result.exit_code, Some(0));
    }

    #[test]
    fn failed_fetch_is_unavailable_without_live_network() {
        let tmp = tempfile::TempDir::new().expect("temp dir");

        let result = provision_with_runner(tmp.path(), |_, _| {
            FetchAttempt::failure("network unavailable")
        });

        assert_eq!(result.status, ProvisionStatus::Unavailable);
        assert!(!result.available());
        assert!(result.attempted_fetch);
        assert_eq!(result.exit_code, Some(7));
        let reason = result.reason.as_deref().unwrap_or_default();
        assert!(reason.contains("network unavailable"), "{reason}");
        assert!(reason.contains("incomplete"), "{reason}");
    }

    #[test]
    fn successful_fetch_with_incomplete_vectors_is_unavailable() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let vectors = vectors_dir(tmp.path());

        let result = provision_with_runner(tmp.path(), |_, _| {
            std::fs::create_dir_all(&vectors).expect("vector dir");
            std::fs::write(vectors.join("testvector01.bit"), b"bit").expect("bit");
            FetchAttempt::success()
        });

        assert_eq!(result.status, ProvisionStatus::Unavailable);
        let reason = result.reason.as_deref().unwrap_or_default();
        assert!(reason.contains("exited 0"), "{reason}");
        assert!(
            reason.contains("testvector01.dec or testvector01m.dec"),
            "{reason}"
        );
    }
}
