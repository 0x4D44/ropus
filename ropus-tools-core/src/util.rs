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
    #[cfg(windows)]
    destination_existed: bool,
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

        #[cfg(windows)]
        let destination_existed = match std::fs::metadata(output_path) {
            Ok(_) => true,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "checking existing output {} before atomic replacement",
                        crate::ui::escape_terminal_path(output_path)
                    )
                });
            }
        };

        for attempt in 0..100u32 {
            let mut temp_name = OsString::from(".");
            temp_name.push(file_name);
            temp_name.push(format!(".ropus-tmp-{pid}-{timestamp}-{attempt}"));
            let temp_path = parent.join(temp_name);
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;

                // New outputs are owner-only until an existing destination's
                // mode is copied immediately before publication.
                options.mode(0o600);
            }
            match options.open(&temp_path) {
                Ok(file) => {
                    return Ok((
                        Self {
                            output_path: output_path.to_path_buf(),
                            temp_path,
                            committed: false,
                            #[cfg(windows)]
                            destination_existed,
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
        #[cfg(unix)]
        preserve_existing_permissions(&self.temp_path, &self.output_path)?;

        #[cfg(windows)]
        atomic_replace(&self.temp_path, &self.output_path, self.destination_existed)?;
        #[cfg(not(windows))]
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

#[cfg(unix)]
fn preserve_existing_permissions(temp_path: &Path, output_path: &Path) -> Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    let metadata = match std::fs::metadata(output_path) {
        Ok(metadata) => metadata,
        // The destination may have been removed while the output was being
        // produced. Keep the explicit secure new-file mode in that case.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "reading existing output permissions for {}",
                    crate::ui::escape_terminal_path(output_path)
                )
            });
        }
    };
    let mode = metadata.mode() & 0o7777;
    std::fs::set_permissions(temp_path, std::fs::Permissions::from_mode(mode)).with_context(|| {
        format!(
            "applying existing output permissions before replacing {}",
            crate::ui::escape_terminal_path(output_path)
        )
    })
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
fn atomic_replace(temp_path: &Path, output_path: &Path, destination_existed: bool) -> Result<()> {
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStrExt;

    unsafe extern "system" {
        #[link_name = "MoveFileExW"]
        fn move_file_ex_w(from: *const u16, to: *const u16, flags: u32) -> i32;
        #[link_name = "ReplaceFileW"]
        fn replace_file_w(
            replaced: *const u16,
            replacement: *const u16,
            backup: *const u16,
            flags: u32,
            exclude: *mut c_void,
            reserved: *mut c_void,
        ) -> i32;
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
    const MOVEFILE_WRITE_THROUGH: u32 = 0x8;
    let replaced = unsafe {
        if destination_existed {
            // ReplaceFileW carries the destination's security descriptor onto
            // the replacement. Passing no ignore-ACL flag makes preservation
            // failure abort the transaction instead of weakening the ACL.
            replace_file_w(
                to.as_ptr(),
                from.as_ptr(),
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        } else {
            // A new destination must not overwrite a file that appeared while
            // the encoder was running with an ACL we never inspected.
            move_file_ex_w(from.as_ptr(), to.as_ptr(), MOVEFILE_WRITE_THROUGH)
        }
    };
    if replaced == 0 {
        return Err(std::io::Error::last_os_error()).with_context(|| {
            format!(
                "publishing output {} while preserving destination security",
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
    use std::io::Write;
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

    #[test]
    fn abandoned_atomic_output_preserves_existing_destination() {
        let dir = test_dir();
        let output = dir.join("output.opus");
        fs::write(&output, b"original").expect("write original output");

        #[cfg(unix)]
        {
            use std::os::unix::fs::{PermissionsExt, set_permissions};

            set_permissions(&output, fs::Permissions::from_mode(0o640))
                .expect("restrict original output");
        }

        #[cfg(windows)]
        let security_before = security_descriptor(&output);

        let (atomic, mut temp) = AtomicOutput::create(&output).expect("create atomic output");
        temp.write_all(b"partial replacement")
            .expect("write partial replacement");
        drop(temp);
        drop(atomic);

        assert_eq!(
            fs::read(&output).expect("read original output"),
            b"original"
        );
        #[cfg(unix)]
        assert_eq!(output_mode(&output), 0o640);
        #[cfg(windows)]
        assert_eq!(security_descriptor(&output), security_before);
        fs::remove_dir_all(dir).expect("remove atomic-output directory");
    }

    #[cfg(unix)]
    #[test]
    fn atomic_output_uses_safe_new_mode_and_preserves_existing_mode() {
        use std::os::unix::fs::{PermissionsExt, set_permissions};

        let dir = test_dir();
        let existing = dir.join("existing.opus");
        fs::write(&existing, b"original").expect("write original output");
        set_permissions(&existing, fs::Permissions::from_mode(0o640))
            .expect("set existing output mode");

        let (atomic, mut temp) = AtomicOutput::create(&existing).expect("create existing output");
        assert_eq!(output_mode_from_file(&temp), 0o600);
        temp.write_all(b"replacement").expect("write replacement");
        temp.flush().expect("flush replacement");
        drop(temp);
        atomic.commit().expect("commit replacement");
        assert_eq!(
            fs::read(&existing).expect("read replacement"),
            b"replacement"
        );
        assert_eq!(output_mode(&existing), 0o640);

        let new_output = dir.join("new.opus");
        let (atomic, mut temp) = AtomicOutput::create(&new_output).expect("create new output");
        assert_eq!(output_mode_from_file(&temp), 0o600);
        temp.write_all(b"new output").expect("write new output");
        temp.flush().expect("flush new output");
        drop(temp);
        atomic.commit().expect("commit new output");
        assert_eq!(output_mode(&new_output), 0o600);
        fs::remove_dir_all(dir).expect("remove atomic-output directory");
    }

    #[cfg(windows)]
    #[test]
    fn atomic_output_preserves_existing_acl() {
        let dir = test_dir();
        let output = dir.join("existing.opus");
        let inherited_reference = dir.join("inherited-reference.opus");
        fs::write(&output, b"original").expect("write original output");
        fs::write(&inherited_reference, b"reference").expect("write inherited reference");
        restrict_file_acl_for_test(&output);
        let restricted = security_descriptor(&output);
        let inherited = security_descriptor(&inherited_reference);
        assert_ne!(
            restricted, inherited,
            "test fixture must use an ACL different from the directory default"
        );

        let (atomic, mut temp) = AtomicOutput::create(&output).expect("create existing output");
        temp.write_all(b"replacement").expect("write replacement");
        temp.flush().expect("flush replacement");
        drop(temp);
        atomic.commit().expect("commit replacement");

        assert_eq!(fs::read(&output).expect("read replacement"), b"replacement");
        assert_eq!(security_descriptor(&output), restricted);
        fs::remove_dir_all(dir).expect("remove atomic-output directory");
    }

    #[cfg(windows)]
    #[test]
    fn atomic_output_new_file_inherits_directory_acl() {
        let dir = test_dir();
        let reference = dir.join("inherited-reference.opus");
        let output = dir.join("new.opus");
        fs::write(&reference, b"reference").expect("write inherited reference");
        let inherited = security_descriptor(&reference);

        let (atomic, mut temp) = AtomicOutput::create(&output).expect("create new output");
        temp.write_all(b"new output").expect("write new output");
        temp.flush().expect("flush new output");
        drop(temp);
        atomic.commit().expect("commit new output");

        assert_eq!(security_descriptor(&output), inherited);
        fs::remove_dir_all(dir).expect("remove atomic-output directory");
    }

    #[cfg(windows)]
    #[test]
    fn atomic_output_does_not_replace_destination_that_appears_late() {
        let dir = test_dir();
        let output = dir.join("new.opus");
        let (atomic, mut temp) = AtomicOutput::create(&output).expect("create new output");
        temp.write_all(b"replacement").expect("write replacement");
        temp.flush().expect("flush replacement");
        drop(temp);

        fs::write(&output, b"raced destination").expect("create raced destination");
        let security_before = security_descriptor(&output);
        assert!(
            atomic.commit().is_err(),
            "late destination must not be replaced"
        );
        assert_eq!(
            fs::read(&output).expect("read raced destination"),
            b"raced destination"
        );
        assert_eq!(security_descriptor(&output), security_before);
        fs::remove_dir_all(dir).expect("remove atomic-output directory");
    }

    #[cfg(unix)]
    fn output_mode(path: &Path) -> u32 {
        use std::os::unix::fs::PermissionsExt;

        output_mode_from_file(&fs::File::open(path).expect("open output for mode"))
    }

    #[cfg(unix)]
    fn output_mode_from_file(file: &fs::File) -> u32 {
        use std::os::unix::fs::PermissionsExt;

        file.metadata()
            .expect("read output mode")
            .permissions()
            .mode()
            & 0o7777
    }

    #[cfg(windows)]
    fn restrict_file_acl_for_test(path: &Path) {
        use std::process::Command;

        let whoami = Command::new("whoami")
            .output()
            .expect("resolve current Windows account");
        assert!(whoami.status.success(), "whoami failed: {whoami:?}");
        let account = String::from_utf8_lossy(&whoami.stdout).trim().to_owned();
        assert!(!account.is_empty(), "whoami returned no account");
        let grant = format!("{account}:F");
        let result = Command::new("icacls")
            .arg(path)
            .arg("/inheritance:r")
            .arg("/grant:r")
            .arg(&grant)
            .output()
            .expect("run icacls");
        assert!(
            result.status.success(),
            "icacls failed: {}{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
    }

    #[cfg(windows)]
    fn security_descriptor(path: &Path) -> Vec<u8> {
        use std::ffi::c_void;
        use std::os::windows::ffi::OsStrExt;

        #[link(name = "advapi32")]
        unsafe extern "system" {
            #[link_name = "GetFileSecurityW"]
            fn get_file_security_w(
                file_name: *const u16,
                requested_information: u32,
                security_descriptor: *mut c_void,
                descriptor_length: u32,
                length_needed: *mut u32,
            ) -> i32;
        }

        let file_name: Vec<u16> = path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        const DACL_SECURITY_INFORMATION: u32 = 0x0000_0004;
        let mut length_needed = 0;
        let first = unsafe {
            get_file_security_w(
                file_name.as_ptr(),
                DACL_SECURITY_INFORMATION,
                std::ptr::null_mut(),
                0,
                &mut length_needed,
            )
        };
        assert_eq!(first, 0, "GetFileSecurityW unexpectedly succeeded");
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(122),
            "GetFileSecurityW did not request a larger buffer"
        );
        let mut descriptor = vec![0u8; length_needed as usize];
        let result = unsafe {
            get_file_security_w(
                file_name.as_ptr(),
                DACL_SECURITY_INFORMATION,
                descriptor.as_mut_ptr().cast(),
                descriptor.len() as u32,
                &mut length_needed,
            )
        };
        assert_ne!(result, 0, "GetFileSecurityW failed");
        descriptor.truncate(length_needed as usize);
        descriptor
    }
}
