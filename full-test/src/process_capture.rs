//! Bounded, concurrent child-process output capture.
//!
//! `std::process::Command::output` drains both pipes into unbounded vectors.
//! The full-test runner executes noisy, long-lived tools, so it must drain
//! stdout and stderr concurrently while retaining only a bounded prefix and
//! tail from each stream.

use std::collections::VecDeque;
use std::io::{self, Read};
use std::process::{Child, Command, Output, Stdio};
use std::thread;

/// Maximum retained bytes per captured stream, including the truncation note.
pub const MAX_CAPTURE_BYTES: usize = 1024 * 1024;

const TRUNCATION_MARKER: &[u8] = b"\n[full-test: output truncated; kept prefix and tail]\n";
const PREFIX_BYTES: usize = (MAX_CAPTURE_BYTES - TRUNCATION_MARKER.len()) / 2;
const TAIL_BYTES: usize = MAX_CAPTURE_BYTES - TRUNCATION_MARKER.len() - PREFIX_BYTES;
const READ_CHUNK_BYTES: usize = 16 * 1024;

/// Run a command with both output pipes drained concurrently and bounded.
pub fn output(command: &mut Command) -> io::Result<Output> {
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let child = command.spawn()?;
    collect(child)
}

fn collect(mut child: Child) -> io::Result<Output> {
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "child stdout was not piped"));
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "child stderr was not piped"));

    let (stdout, stderr) = match (stdout, stderr) {
        (Ok(stdout), Ok(stderr)) => (stdout, stderr),
        (Err(error), _) | (_, Err(error)) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }
    };

    let stdout_thread = thread::spawn(move || read_bounded_bytes(stdout));
    let stderr_thread = thread::spawn(move || read_bounded_bytes(stderr));
    let status = child.wait();
    let stdout = join_reader(stdout_thread)?;
    let stderr = join_reader(stderr_thread)?;

    Ok(Output {
        status: status?,
        stdout,
        stderr,
    })
}

fn join_reader(handle: thread::JoinHandle<io::Result<Vec<u8>>>) -> io::Result<Vec<u8>> {
    handle
        .join()
        .map_err(|_| io::Error::other("child output reader panicked"))?
}

/// Read a stream while retaining at most [`MAX_CAPTURE_BYTES`] bytes.
///
/// Before the cap is crossed, the complete stream is retained. Once it is
/// crossed, the result keeps a prefix and a tail with an explicit marker so
/// diagnostics preserve both the command's beginning and final failure lines.
pub fn read_bounded<R: Read>(mut reader: R) -> io::Result<String> {
    let bytes = read_bounded_bytes(&mut reader)?;
    Ok(bytes_to_string(&bytes))
}

fn bytes_to_string(bytes: &[u8]) -> String {
    let mut text = String::from_utf8_lossy(bytes).into_owned();
    if text.len() > MAX_CAPTURE_BYTES {
        let mut end = MAX_CAPTURE_BYTES;
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
    }
    text
}

fn read_bounded_bytes<R: Read>(mut reader: R) -> io::Result<Vec<u8>> {
    let mut capture = BoundedCapture::new();
    let mut chunk = [0u8; READ_CHUNK_BYTES];
    loop {
        let read = reader.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        capture.push(&chunk[..read]);
    }
    Ok(capture.finish())
}

enum CaptureState {
    Complete(Vec<u8>),
    Truncated { prefix: Vec<u8>, tail: VecDeque<u8> },
}

struct BoundedCapture {
    state: CaptureState,
}

impl BoundedCapture {
    fn new() -> Self {
        Self {
            state: CaptureState::Complete(Vec::new()),
        }
    }

    fn push(&mut self, bytes: &[u8]) {
        match &mut self.state {
            CaptureState::Complete(data) => {
                if data.len().saturating_add(bytes.len()) <= MAX_CAPTURE_BYTES {
                    data.extend_from_slice(bytes);
                    return;
                }

                let mut combined = std::mem::take(data);
                combined.extend_from_slice(bytes);
                let prefix = combined[..PREFIX_BYTES].to_vec();
                let tail_start = combined.len().saturating_sub(TAIL_BYTES);
                let tail = combined[tail_start..].iter().copied().collect();
                self.state = CaptureState::Truncated { prefix, tail };
            }
            CaptureState::Truncated { tail, .. } => {
                for byte in bytes {
                    if tail.len() == TAIL_BYTES {
                        tail.pop_front();
                    }
                    tail.push_back(*byte);
                }
            }
        }
    }

    fn finish(self) -> Vec<u8> {
        match self.state {
            CaptureState::Complete(data) => data,
            CaptureState::Truncated { prefix, tail } => {
                let mut output = Vec::with_capacity(MAX_CAPTURE_BYTES);
                output.extend_from_slice(&prefix);
                output.extend_from_slice(TRUNCATION_MARKER);
                output.extend(tail);
                output
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn oversized_stream_keeps_bounded_prefix_and_tail() {
        let prefix = b"prefix\n";
        let tail = b"tail\n";
        let mut input = Vec::with_capacity(MAX_CAPTURE_BYTES * 2);
        input.extend_from_slice(prefix);
        input.resize(MAX_CAPTURE_BYTES * 2, b'x');
        input.extend_from_slice(tail);

        let captured = read_bounded(Cursor::new(input)).expect("capture");

        assert!(captured.len() <= MAX_CAPTURE_BYTES);
        assert!(captured.starts_with("prefix\n"));
        assert!(captured.ends_with("tail\n"));
        assert!(captured.contains("output truncated"));
    }

    #[test]
    fn short_stream_is_preserved_exactly() {
        let captured = read_bounded(Cursor::new(b"stdout\n".to_vec())).expect("capture");
        assert_eq!(captured, "stdout\n");
    }

    #[test]
    fn invalid_utf8_does_not_expand_past_the_cap() {
        let captured = read_bounded(Cursor::new(vec![0xff; MAX_CAPTURE_BYTES])).expect("capture");
        assert!(captured.len() <= MAX_CAPTURE_BYTES);
    }

    #[cfg(unix)]
    #[test]
    fn child_status_and_both_streams_survive_bounded_capture() {
        let mut command = Command::new("sh");
        command.args([
            "-c",
            "printf 'stdout-start'; printf 'stderr-tail' >&2; exit 7",
        ]);

        let output = output(&mut command).expect("run command");
        assert_eq!(output.status.code(), Some(7));
        assert_eq!(output.stdout, b"stdout-start");
        assert_eq!(output.stderr, b"stderr-tail");
    }
}
