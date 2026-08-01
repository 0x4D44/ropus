//! Miscellaneous small helpers shared by multiple commands.

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
