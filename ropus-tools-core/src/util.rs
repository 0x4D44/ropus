//! Miscellaneous small helpers shared by multiple commands.

use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use ropus::Channels as RopusChannels;

pub fn channel_count_to_ropus(n: usize) -> Result<RopusChannels> {
    match n {
        1 => Ok(RopusChannels::Mono),
        2 => Ok(RopusChannels::Stereo),
        other => bail!("unsupported channel count {other} (ropus supports mono/stereo)"),
    }
}

pub fn with_extension(path: &Path, ext: &str) -> PathBuf {
    let mut p = path.to_path_buf();
    p.set_extension(ext);
    p
}

/// Returns true if `path` is the stdin/stdout sentinel `-` used by
/// `ropusenc` and `ropusdec`. Centralised here so every command compares the
/// sentinel identically (OsStr-level equality, no lossy string conversion).
pub fn is_stdio_sentinel(path: &Path) -> bool {
    path.as_os_str() == "-"
}

/// Return whether two existing paths resolve to the same filesystem object.
///
/// The platform metadata keys follow symlinks and identify hard links without
/// opening the output for writing. A missing candidate is not an alias; the
/// eventual output create/open reports its own error.
fn paths_refer_to_same_file(input: &Path, output: &Path) -> Result<bool> {
    // Do this cheap lexical check even when the output does not exist yet. It
    // also covers equivalent `.`/`..` spellings without relying on a platform
    // canonicalisation call.
    if input.exists() && normalize_lexical_path(input)? == normalize_lexical_path(output)? {
        return Ok(true);
    }

    let input_metadata = match std::fs::metadata(input) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error).context("reading input metadata for identity check"),
    };
    let output_metadata = match std::fs::metadata(output) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error).context("reading output metadata for identity check"),
    };
    Ok(
        match (
            metadata_identity(&input_metadata),
            metadata_identity(&output_metadata),
        ) {
            (Some(input_id), Some(output_id)) => input_id == output_id,
            _ => false,
        },
    )
}

#[cfg(unix)]
fn metadata_identity(metadata: &std::fs::Metadata) -> Option<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;

    Some((metadata.dev(), metadata.ino()))
}

#[cfg(windows)]
fn metadata_identity(metadata: &std::fs::Metadata) -> Option<(u64, u64, u64, u32)> {
    use std::os::windows::fs::MetadataExt;

    // The stable Windows metadata surface exposes these values. Hard links
    // share all four, while treating a coincident tuple as an alias is the
    // safe failure mode: it refuses a write instead of risking truncation.
    Some((
        metadata.creation_time(),
        metadata.last_write_time(),
        metadata.file_size(),
        metadata.file_attributes(),
    ))
}

#[cfg(not(any(unix, windows)))]
fn metadata_identity(_metadata: &std::fs::Metadata) -> Option<()> {
    None
}

/// Reject an output path that would truncate the input path.
///
/// Standard-stream sentinels are intentionally exempt: `-` means a pipe, not
/// a second filesystem name. Call this before decoding or creating output.
pub fn reject_input_output_alias(input: &Path, output: &Path) -> Result<()> {
    if is_stdio_sentinel(input) || is_stdio_sentinel(output) {
        return Ok(());
    }
    if paths_refer_to_same_file(input, output)? {
        bail!("input and output refer to the same file; choose a different output path");
    }
    Ok(())
}

/// A regular-file destination that is committed only after the complete output
/// has flushed successfully. The temporary lives beside the final path so the
/// final rename is atomic on the same filesystem.
pub(crate) struct AtomicOutput {
    output_path: PathBuf,
    temp_path: PathBuf,
    committed: bool,
}

impl AtomicOutput {
    pub(crate) fn create(output_path: &Path) -> Result<(Self, File)> {
        let parent = output_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let file_name = output_path.file_name().ok_or_else(|| {
            anyhow::anyhow!("output path {} has no file name", output_path.display())
        })?;
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let pid = std::process::id();

        for attempt in 0..100u32 {
            let mut temp_name = OsString::from(".");
            temp_name.push(file_name);
            temp_name.push(format!(".ropus-tmp-{pid}-{timestamp}-{attempt}"));
            let temp_path = parent.join(temp_name);
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp_path)
            {
                Ok(file) => {
                    return Ok((
                        Self {
                            output_path: output_path.to_path_buf(),
                            temp_path,
                            committed: false,
                        },
                        file,
                    ));
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!(
                            "creating temporary output beside {}",
                            crate::ui::escape_terminal_path(output_path)
                        )
                    });
                }
            }
        }
        bail!(
            "could not create a unique temporary output beside {}",
            crate::ui::escape_terminal_path(output_path)
        )
    }

    pub(crate) fn commit(mut self) -> Result<()> {
        atomic_replace(&self.temp_path, &self.output_path)?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for AtomicOutput {
    fn drop(&mut self) {
        if !self.committed {
            let _ = std::fs::remove_file(&self.temp_path);
        }
    }
}

#[cfg(not(windows))]
fn atomic_replace(temp_path: &Path, output_path: &Path) -> Result<()> {
    std::fs::rename(temp_path, output_path).with_context(|| {
        format!(
            "replacing output {} with flushed temporary",
            crate::ui::escape_terminal_path(output_path)
        )
    })
}

#[cfg(windows)]
fn atomic_replace(temp_path: &Path, output_path: &Path) -> Result<()> {
    use std::os::windows::ffi::OsStrExt;

    unsafe extern "system" {
        #[link_name = "MoveFileExW"]
        fn move_file_ex_w(from: *const u16, to: *const u16, flags: u32) -> i32;
    }

    let from: Vec<u16> = temp_path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let to: Vec<u16> = output_path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    const MOVEFILE_REPLACE_EXISTING: u32 = 0x1;
    const MOVEFILE_WRITE_THROUGH: u32 = 0x8;
    let replaced = unsafe {
        move_file_ex_w(
            from.as_ptr(),
            to.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if replaced == 0 {
        return Err(std::io::Error::last_os_error()).with_context(|| {
            format!(
                "replacing output {} with flushed temporary",
                crate::ui::escape_terminal_path(output_path)
            )
        });
    }
    Ok(())
}

/// Choose a deterministic default destination that cannot be the source.
///
/// `suffix` differentiates a source whose existing extension already equals
/// the requested destination extension (`song.opus` → `song.encoded.opus`, or
/// `song.wav` containing Opus data → `song.decoded.wav`).
pub fn noncolliding_default_output(input: &Path, extension: &str, suffix: &str) -> Result<PathBuf> {
    let candidate = with_extension(input, extension);
    if !paths_refer_to_same_file(input, &candidate)? {
        return Ok(candidate);
    }

    let stem = input
        .file_stem()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("output");
    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    for index in 0..1000u32 {
        let name = if index == 0 {
            format!("{stem}.{suffix}.{extension}")
        } else {
            format!("{stem}.{suffix}.{index}.{extension}")
        };
        let candidate = parent.join(name);
        if !paths_refer_to_same_file(input, &candidate)? {
            return Ok(candidate);
        }
    }
    bail!(
        "could not choose a non-colliding output path for {}",
        input.display()
    )
}

fn normalize_lexical_path(path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .context("resolving current directory for input/output identity check")?
            .join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normalized.pop();
            }
            _ => normalized.push(component.as_os_str()),
        }
    }
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_dir() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ropus-path-{}-{nonce}", std::process::id()));
        fs::create_dir(&dir).expect("create temporary identity-test directory");
        dir
    }

    #[test]
    fn direct_and_lexical_aliases_are_rejected() {
        let dir = test_dir();
        let input = dir.join("input.wav");
        fs::write(&input, b"source").expect("write input");
        assert!(reject_input_output_alias(&input, &input).is_err());
        let lexical = dir.join("nested").join("..").join("input.wav");
        assert!(reject_input_output_alias(&input, &lexical).is_err());
        let output = dir.join("output.opus");
        assert!(reject_input_output_alias(&input, &output).is_ok());
        fs::remove_dir_all(dir).expect("remove identity-test directory");
    }

    #[cfg(unix)]
    #[test]
    fn symlink_and_hard_link_aliases_are_rejected() {
        use std::os::unix::fs::symlink;

        let dir = test_dir();
        let input = dir.join("input.wav");
        fs::write(&input, b"source").expect("write input");
        let hard = dir.join("hard.wav");
        fs::hard_link(&input, &hard).expect("create hard link");
        let link = dir.join("link.wav");
        symlink(&input, &link).expect("create symlink");
        assert!(reject_input_output_alias(&input, &hard).is_err());
        assert!(reject_input_output_alias(&input, &link).is_err());
        fs::remove_dir_all(dir).expect("remove identity-test directory");
    }

    #[test]
    fn default_extension_collision_gets_a_safe_suffix() {
        let dir = test_dir();
        let input = dir.join("song.opus");
        fs::write(&input, b"source").expect("write input");
        let output = noncolliding_default_output(&input, "opus", "encoded")
            .expect("choose noncolliding output");
        assert_eq!(output, dir.join("song.encoded.opus"));
        assert!(reject_input_output_alias(&input, &output).is_ok());
        fs::remove_dir_all(dir).expect("remove identity-test directory");
    }
}
