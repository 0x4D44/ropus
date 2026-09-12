//! Stage 7b.3 diagnostic: measure the fixed-vs-float arithmetic gap of
//! classical SILK PLC under the tier-2 packet and loss conditions.
//!
//! This is a control diagnostic, not a ceiling or calibration oracle for the
//! neural PLC gate. Classical LPC/LTP concealment and neural concealment have
//! different error-transfer functions, so a classical SNR cannot bound the
//! direct Rust-vs-C neural comparison in `harness-deep-plc/tests/tier2_snr.rs`.
//!
//! The test proves its own identity: both decoder binaries receive the same
//! packet file, announce their fixed/float classical mode and complexity,
//! decode a non-silent signal, and diverge on the lossy path. The measured
//! classical SNR stays inside a broad diagnostic interval so a broken control
//! harness cannot silently become the new neural calibration source.
//!
//! 1. Synthesise the same deterministic 2-second PCM as `tier2_snr.rs`.
//! 2. Encode once with ropus at 16 kbps, complexity 10 (identical bitstream
//!    to a float-mode encode, per xiph's bit-exact encoder guarantee).
//! 3. Mark frame indices under the same `is_lost(i) = i > 0 && i % 7 == 0`
//!    pattern and write the whole stream (packet bytes + lost flags) to a
//!    tempfile.
//! 4. Build + run `ctrl_decode_fixed` (from the `harness` crate) and
//!    `ctrl_decode_float` (from the `harness-deep-plc` crate) back-to-back.
//!    Both request complexity = 4 so the float side's compiled-in DEEP_PLC
//!    stays dormant and classical SILK PLC fills every lost frame — the
//!    fixed side never had DEEP_PLC anyway.
//! 5. Read both PCMs back from disk and compute SNR(fixed, float). This is
//!    the control number.
//!
//! The neural gate is calibrated only by its direct Rust-vs-C neural cases;
//! this experiment supplies context about the classical arithmetic gap.

use std::env;
use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ropus::{OPUS_APPLICATION_VOIP, OPUS_OK, OpusEncoder};

// ---------------------------------------------------------------------------
// Constants — mirror tier2_snr.rs exactly so the loss pattern and packet set
// are comparable.
// ---------------------------------------------------------------------------

const FS: i32 = 48_000;
const CHANNELS: i32 = 1;
const FRAME_MS: i32 = 20;
const FRAME_SIZE: i32 = FS * FRAME_MS / 1000; // 960
const BITRATE: i32 = 16_000;
const ENC_COMPLEXITY: i32 = 10;
const SIGNAL_DURATION_MS: i32 = 2_000;
const TOTAL_FRAMES: i32 = SIGNAL_DURATION_MS / FRAME_MS;
const LOSS_INTERVAL_FRAMES: usize = 7;
const EXPECTED_LOST_FRAMES: &[usize] = &[7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91];

const LOST_BIT: u32 = 0x8000_0000;
const CONTROL_COMPLEXITY: i32 = 4;
const CONTROL_SNR_MIN_DB: f64 = 35.0;
const CONTROL_SNR_MAX_DB: f64 = 50.0;
const CONTROL_LOSSLESS_MIN_DB: f64 = 80.0;
const CONTROL_SIGNAL_MIN_MEAN_SQUARE: f64 = 1_000_000.0;
const CONTROL_OUTPUT_MIN_MEAN_SQUARE: f64 = 100_000.0;
const FIXED_MODE_MARKER: &str = "control-mode=classical fixed-point complexity=4";
const FLOAT_MODE_MARKER: &str = "control-mode=classical float complexity=4 deep_plc=disabled";
const CONTROL_COMMAND_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const CONTROL_POLL_INTERVAL: Duration = Duration::from_millis(25);

// Loss pattern is byte-for-byte identical to `tier2_snr.rs::is_lost`.
fn is_lost(frame_idx: usize) -> bool {
    frame_idx > 0
        && frame_idx.is_multiple_of(LOSS_INTERVAL_FRAMES)
        && frame_idx + LOSS_INTERVAL_FRAMES <= TOTAL_FRAMES as usize
}

#[test]
fn loss_pattern_contains_only_complete_recovery_cycles() {
    let losses: Vec<_> = (0..TOTAL_FRAMES as usize).filter(|&i| is_lost(i)).collect();

    assert_eq!(losses, EXPECTED_LOST_FRAMES);
    assert!(
        losses
            .iter()
            .all(|&i| i + LOSS_INTERVAL_FRAMES <= TOTAL_FRAMES as usize)
    );
}

// Synthetic PCM: bit-identical to `tier2_snr.rs::synth_reference_pcm()`.
// Kept inline to avoid coupling crate dependencies just for this helper.
fn synth_reference_pcm() -> Vec<i16> {
    let n_samples = (FS as usize) * (SIGNAL_DURATION_MS as usize) / 1000;
    let mut pcm = Vec::with_capacity(n_samples);
    let mut rng: u32 = 0xC0FFEE_u32;
    for i in 0..n_samples {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        let noise = ((rng as i32) >> 22) as f64 / 512.0;

        let t = i as f64 / FS as f64;
        let env = 0.5 + 0.5 * (2.0 * std::f64::consts::PI * 2.0 * t).sin().abs();
        let tone1 = (2.0 * std::f64::consts::PI * 220.0 * t).sin();
        let tone2 = (2.0 * std::f64::consts::PI * 880.0 * t).sin();
        let sample = env * (0.6 * tone1 + 0.35 * tone2) + 0.05 * noise;
        let s_i16 = (sample.clamp(-1.0, 1.0) * 28_000.0) as i16;
        pcm.push(s_i16);
    }
    pcm
}

fn encode_with_ropus(pcm: &[i16]) -> Vec<Vec<u8>> {
    let mut enc =
        OpusEncoder::new(FS, CHANNELS, OPUS_APPLICATION_VOIP).expect("ropus encoder_create failed");
    assert_eq!(enc.set_bitrate(BITRATE), OPUS_OK);
    assert_eq!(enc.set_complexity(ENC_COMPLEXITY), OPUS_OK);

    let frame_samples = FRAME_SIZE as usize;
    let expected_frames = pcm.len() / frame_samples;
    let mut packets = Vec::with_capacity(expected_frames);
    for frame_idx in 0..expected_frames {
        let start = frame_idx * frame_samples;
        let frame = &pcm[start..start + frame_samples];
        let mut buf = vec![0u8; 4000];
        let cap = buf.len() as i32;
        let n = enc
            .encode(frame, FRAME_SIZE, &mut buf, cap)
            .unwrap_or_else(|e| panic!("encode frame {frame_idx}: {e}"));
        assert!(n > 0, "frame {frame_idx}: empty packet");
        buf.truncate(n as usize);
        packets.push(buf);
    }
    packets
}

fn write_packets_file(
    path: &Path,
    packets: &[Vec<u8>],
    drop_pattern: impl Fn(usize) -> bool,
) -> std::io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);

    // Header.
    w.write_all(&(packets.len() as u32).to_le_bytes())?;
    w.write_all(&(FS as u32).to_le_bytes())?;
    w.write_all(&(CHANNELS as u32).to_le_bytes())?;
    w.write_all(&(FRAME_SIZE as u32).to_le_bytes())?;

    // Per-frame records.
    for (i, pkt) in packets.iter().enumerate() {
        if drop_pattern(i) {
            // LOST: flag bit set, zero-length, no payload.
            w.write_all(&LOST_BIT.to_le_bytes())?;
        } else {
            let len = pkt.len() as u32;
            assert!(len & LOST_BIT == 0, "packet too large for our encoding");
            w.write_all(&len.to_le_bytes())?;
            w.write_all(pkt)?;
        }
    }
    w.flush()
}

fn read_pcm_file(path: &Path) -> std::io::Result<Vec<i16>> {
    let mut f = File::open(path)?;
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)?;
    assert!(bytes.len().is_multiple_of(2), "odd-length PCM file");
    let mut pcm = Vec::with_capacity(bytes.len() / 2);
    for ch in bytes.chunks_exact(2) {
        pcm.push(i16::from_le_bytes([ch[0], ch[1]]));
    }
    Ok(pcm)
}

/// SNR(test vs ref) in dB. Matches the formula used by `tier2_snr.rs`.
fn compute_snr_db(ref_pcm: &[i16], test: &[i16]) -> f64 {
    assert_eq!(
        ref_pcm.len(),
        test.len(),
        "SNR inputs differ in length: {} vs {}",
        ref_pcm.len(),
        test.len()
    );
    let mut signal_power = 0.0_f64;
    let mut noise_power = 0.0_f64;
    for (&r, &t) in ref_pcm.iter().zip(test.iter()) {
        let r_f = r as f64;
        let t_f = t as f64;
        signal_power += r_f * r_f;
        let err = t_f - r_f;
        noise_power += err * err;
    }
    let n = ref_pcm.len() as f64;
    signal_power /= n;
    noise_power /= n;
    if noise_power == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (signal_power / noise_power).log10()
}

fn mean_square_energy(pcm: &[i16]) -> f64 {
    assert!(!pcm.is_empty(), "energy requires non-empty PCM");
    pcm.iter()
        .map(|&sample| {
            let sample = sample as f64;
            sample * sample
        })
        .sum::<f64>()
        / pcm.len() as f64
}

fn packet_stream_fingerprint(packets: &[Vec<u8>]) -> u64 {
    // Small deterministic FNV-1a fingerprint: enough to prove both child
    // decoders consumed the same non-empty packet stream without a new hash
    // dependency in this test-only crate.
    let mut hash = 0xcbf29ce484222325u64;
    for packet in packets {
        for &byte in packet {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn first_divergent(a: &[i16], b: &[i16]) -> Option<usize> {
    a.iter().zip(b.iter()).position(|(x, y)| x != y)
}

// ---------------------------------------------------------------------------
// Tempfile management — each control invocation gets its own OS-temp directory
// so concurrent cargo test processes never share packet or PCM artifacts.
// ---------------------------------------------------------------------------

fn workspace_root() -> PathBuf {
    // This file lives under `harness-control/tests/`. Walk up two.
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    manifest_dir
        .parent()
        .expect("manifest_dir has a parent")
        .to_path_buf()
}

static NEXT_TMP_ID: AtomicU64 = AtomicU64::new(0);

struct CtrlTempDir {
    path: PathBuf,
}

impl CtrlTempDir {
    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for CtrlTempDir {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.path) {
            eprintln!("failed to clean {}: {error}", self.path.display());
        }
    }
}

fn ctrl_tmp_dir() -> CtrlTempDir {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let process_id = std::process::id();
    let sequence = NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed);
    let root = env::temp_dir();

    for attempt in 0..16 {
        let dir = root.join(format!(
            "ropus-harness-control-{process_id}-{now}-{sequence}-{attempt}"
        ));
        match std::fs::create_dir(&dir) {
            Ok(()) => return CtrlTempDir { path: dir },
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => panic!("create {}: {error}", dir.display()),
        }
    }
    panic!("failed to create a unique control temp directory after 16 attempts");
}

#[test]
fn control_temp_dirs_are_unique_and_cleaned_up() {
    let (first_path, second_path);
    {
        let first = ctrl_tmp_dir();
        let second = ctrl_tmp_dir();
        assert_ne!(first.path(), second.path());
        assert!(first.path().is_dir());
        assert!(second.path().is_dir());
        first_path = first.path().to_path_buf();
        second_path = second.path().to_path_buf();
    }
    assert!(!first_path.exists());
    assert!(!second_path.exists());
}

#[derive(Debug)]
enum ControlCapture {
    Completed(Output),
    TimedOut(Output),
}

fn run_command_with_timeout(
    command: &mut Command,
    timeout: Duration,
) -> io::Result<ControlCapture> {
    configure_process_group(command);
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command.spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "child stdout was not piped"));
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "child stderr was not piped"));
    let (mut stdout, mut stderr) = match (stdout, stderr) {
        (Ok(stdout), Ok(stderr)) => (stdout, stderr),
        (Err(error), _) | (_, Err(error)) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }
    };

    let stdout_thread = thread::spawn(move || {
        let mut bytes = Vec::new();
        stdout.read_to_end(&mut bytes)?;
        Ok(bytes)
    });
    let stderr_thread = thread::spawn(move || {
        let mut bytes = Vec::new();
        stderr.read_to_end(&mut bytes)?;
        Ok(bytes)
    });
    let deadline = Instant::now() + timeout;
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) if Instant::now() >= deadline => {
                timed_out = true;
                break terminate_process_tree(&mut child);
            }
            Ok(None) => {
                let remaining = deadline.saturating_duration_since(Instant::now());
                thread::sleep(remaining.min(CONTROL_POLL_INTERVAL));
            }
            Err(error) => {
                let cleanup = terminate_process_tree(&mut child);
                let _ = join_output_reader(stdout_thread);
                let _ = join_output_reader(stderr_thread);
                return Err(combine_wait_error(error, cleanup));
            }
        }
    }?;
    let stdout = join_output_reader(stdout_thread)?;
    let stderr = join_output_reader(stderr_thread)?;
    let output = Output {
        status,
        stdout,
        stderr,
    };
    Ok(if timed_out {
        ControlCapture::TimedOut(output)
    } else {
        ControlCapture::Completed(output)
    })
}

fn combine_wait_error(wait_error: io::Error, cleanup: io::Result<ExitStatus>) -> io::Error {
    match cleanup {
        Ok(_) => wait_error,
        Err(cleanup_error) => io::Error::other(format!(
            "failed while waiting for child: {wait_error}; process-tree cleanup also failed: \
             {cleanup_error}"
        )),
    }
}

fn terminate_process_tree(child: &mut Child) -> io::Result<ExitStatus> {
    let tree_error = kill_process_tree(child.id()).err();
    if let Some(tree_error) = tree_error {
        let direct_error = child.kill().err();
        let status = child.wait();
        return match status {
            Ok(status) => Err(io::Error::other(format!(
                "process-tree cleanup failed: {tree_error}; direct child was reaped with {status}{}",
                direct_error
                    .as_ref()
                    .map(|error| format!("; direct kill also failed: {error}"))
                    .unwrap_or_default()
            ))),
            Err(wait_error) => Err(io::Error::other(format!(
                "process-tree cleanup failed: {tree_error}; waiting for direct child also failed: \
                 {wait_error}{}",
                direct_error
                    .as_ref()
                    .map(|error| format!("; direct kill also failed: {error}"))
                    .unwrap_or_default()
            ))),
        };
    }
    child.wait()
}

#[cfg(unix)]
fn configure_process_group(command: &mut Command) {
    use std::os::unix::process::CommandExt;

    command.process_group(0);
}

#[cfg(not(unix))]
fn configure_process_group(_command: &mut Command) {}

#[cfg(unix)]
fn kill_process_tree(pid: u32) -> io::Result<()> {
    let group = format!("-{pid}");
    let status = Command::new("kill")
        .args(["-KILL", &group])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "kill process group for child {pid} exited with {status}"
        )))
    }
}

#[cfg(windows)]
fn kill_process_tree(pid: u32) -> io::Result<()> {
    let status = Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "taskkill for child {pid} exited with {status}"
        )))
    }
}

#[cfg(not(any(unix, windows)))]
fn kill_process_tree(pid: u32) -> io::Result<()> {
    Err(io::Error::other(format!(
        "process-tree termination is unsupported for child {pid}"
    )))
}

fn join_output_reader(handle: thread::JoinHandle<io::Result<Vec<u8>>>) -> io::Result<Vec<u8>> {
    handle
        .join()
        .map_err(|_| io::Error::other("child output reader panicked"))?
}

#[cfg(any(unix, windows))]
#[test]
fn hanging_control_child_is_killed_and_reports_a_distinct_timeout() {
    let mut command = hanging_command();
    let started = Instant::now();
    let result = run_command_with_timeout(&mut command, Duration::from_millis(100))
        .expect("supervisor should return a timeout result");

    match result {
        ControlCapture::TimedOut(output) => {
            assert!(!output.status.success());
            assert!(
                started.elapsed() < Duration::from_secs(1),
                "timeout took too long: {:?}",
                started.elapsed()
            );
        }
        ControlCapture::Completed(output) => {
            panic!("hanging child completed unexpectedly: {:?}", output.status);
        }
    }
}

#[cfg(unix)]
fn hanging_command() -> Command {
    let mut command = Command::new("sh");
    command.args(["-c", "sleep 2 & wait"]);
    command
}

#[cfg(windows)]
fn hanging_command() -> Command {
    let mut command = Command::new("powershell");
    command.args([
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "Start-Process -FilePath powershell -ArgumentList '-NoLogo','-NoProfile','-NonInteractive','-Command','Start-Sleep -Seconds 2' -Wait",
    ]);
    command
}

fn run_decoder(package: &str, bin: &str, packets: &Path, pcm_out: &Path, mode_marker: &str) {
    // Use `cargo run` rather than locating the binary directly — keeps this
    // test agnostic to the target triple + release vs debug dir names. The
    // dep binaries rebuild only if they or their C sources changed, so the
    // first invocation pays a one-time cost, subsequent runs are cheap.
    let t0 = Instant::now();
    let mut command = Command::new(env!("CARGO"));
    command
        .args([
            "run",
            "--quiet",
            "-p",
            package,
            "--bin",
            bin,
            "--release",
            "--",
        ])
        .arg(packets)
        .arg(pcm_out)
        .current_dir(workspace_root());
    let result = run_command_with_timeout(&mut command, CONTROL_COMMAND_TIMEOUT)
        .unwrap_or_else(|e| panic!("cargo run -p {package} --bin {bin} failed: {e}"));
    let elapsed = t0.elapsed();
    let output = match result {
        ControlCapture::Completed(output) => output,
        ControlCapture::TimedOut(output) => panic!(
            "cargo run -p {package} --bin {bin} timed out after {elapsed:?}; stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        ),
    };
    assert!(
        output.status.success(),
        "cargo run -p {package} --bin {bin} exited with {:?} ({elapsed:?}); stderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(mode_marker),
        "{bin} did not announce expected control mode marker {mode_marker:?}; stderr: {stderr}"
    );
    eprintln!("  {bin}: decoded in {elapsed:.2?}");
}

// ---------------------------------------------------------------------------
// The test itself.
// ---------------------------------------------------------------------------

#[test]
fn ctrl_fixed_vs_float_classical_snr() {
    eprintln!("=== Stage 7b.3 control experiment: C-fixed-classical vs C-float-classical ===");
    assert_eq!(
        CONTROL_COMPLEXITY, 4,
        "control markers and decoder configuration must stay at classical complexity"
    );

    // 1. Reference PCM + encode — identical to tier2_snr.rs's preamble.
    let pcm_in = synth_reference_pcm();
    let packets = encode_with_ropus(&pcm_in);
    assert_eq!(
        packets.len() as i32,
        TOTAL_FRAMES,
        "encoder emitted {} packets, expected {}",
        packets.len(),
        TOTAL_FRAMES
    );
    let n_lost = (0..packets.len()).filter(|&i| is_lost(i)).count();
    assert!(
        n_lost > 5,
        "loss pattern lost only {n_lost} packets — not meaningful"
    );
    let expected_lost = EXPECTED_LOST_FRAMES.len();
    assert_eq!(
        n_lost, expected_lost,
        "control loss pattern changed: expected {expected_lost} lost frames"
    );
    let input_energy = mean_square_energy(&pcm_in);
    assert!(
        input_energy > CONTROL_SIGNAL_MIN_MEAN_SQUARE,
        "control input is not energetic enough: mean-square={input_energy:.1}"
    );
    let packet_fingerprint = packet_stream_fingerprint(&packets);
    assert_ne!(
        packet_fingerprint, 0,
        "encoded control packet stream is empty"
    );

    // 2. Write packets to a tempfile. Both decoders read the same bytes so
    // there's no chance the two sides see a different frame-by-frame stream.
    let tmp = ctrl_tmp_dir();
    let packets_path = tmp.path().join("ctrl_packets.bin");
    write_packets_file(&packets_path, &packets, is_lost).expect("write packets file");
    let packet_file_before = std::fs::read(&packets_path).expect("read control packet file");
    eprintln!(
        "  wrote {} frames ({} lost) to {}",
        packets.len(),
        n_lost,
        packets_path.display()
    );

    // 3. Run both decoders.
    let fixed_pcm_path = tmp.path().join("ctrl_fixed.pcm");
    let float_pcm_path = tmp.path().join("ctrl_float.pcm");
    run_decoder(
        "ropus-harness",
        "ctrl_decode_fixed",
        &packets_path,
        &fixed_pcm_path,
        FIXED_MODE_MARKER,
    );
    run_decoder(
        "ropus-harness-deep-plc",
        "ctrl_decode_float",
        &packets_path,
        &float_pcm_path,
        FLOAT_MODE_MARKER,
    );

    assert_eq!(
        std::fs::read(&packets_path).expect("read packets after fixed/float decode"),
        packet_file_before,
        "control decoder mutated the shared packet stream"
    );

    // 4. Read back + compute SNR.
    let pcm_fixed = read_pcm_file(&fixed_pcm_path).expect("read fixed pcm");
    let pcm_float = read_pcm_file(&float_pcm_path).expect("read float pcm");
    assert_eq!(
        pcm_fixed.len(),
        pcm_float.len(),
        "PCM lengths differ: fixed={} float={}",
        pcm_fixed.len(),
        pcm_float.len()
    );
    assert_eq!(
        pcm_fixed.len(),
        (TOTAL_FRAMES as usize) * (FRAME_SIZE as usize) * (CHANNELS as usize),
        "PCM length inconsistent with expected frames*frame_size*channels"
    );
    let fixed_energy = mean_square_energy(&pcm_fixed);
    let float_energy = mean_square_energy(&pcm_float);
    assert!(
        fixed_energy > CONTROL_OUTPUT_MIN_MEAN_SQUARE
            && float_energy > CONTROL_OUTPUT_MIN_MEAN_SQUARE,
        "classical control output lost signal energy: fixed={fixed_energy:.1}, float={float_energy:.1}"
    );

    let snr = compute_snr_db(&pcm_float, &pcm_fixed);
    let first_diverge = first_divergent(&pcm_float, &pcm_fixed);

    eprintln!("===");
    eprintln!("  n_lost          = {n_lost}");
    eprintln!("  total samples   = {}", pcm_fixed.len());
    eprintln!("  SNR(fixed vs float, classical PLC) = {:.2} dB", snr);
    eprintln!("  first divergent sample index = {first_diverge:?}");
    eprintln!("  packet fingerprint = {packet_fingerprint:016x}");
    eprintln!("===");

    assert!(
        first_diverge.is_some(),
        "classical fixed/float control unexpectedly produced identical lossy PCM"
    );
    assert!(
        (CONTROL_SNR_MIN_DB..=CONTROL_SNR_MAX_DB).contains(&snr),
        "classical control SNR {snr:.2} dB is outside diagnostic interval \
         [{CONTROL_SNR_MIN_DB:.0}, {CONTROL_SNR_MAX_DB:.0}] — inspect mode, \
         packet identity, and first divergent sample {first_diverge:?}"
    );
}

/// Lossless sanity-check baseline: SNR(fixed vs float) with NO packet loss.
/// Expected: ~90 dB (matches the reasoning in `tier2_snr.rs` lossless
/// regression). Confirms the harness itself is working — the lossy number
/// above is a PLC-path result, not pervasive corruption.
#[test]
fn ctrl_fixed_vs_float_classical_snr_lossless() {
    eprintln!("=== Stage 7b.3 control: lossless baseline SNR(C-fixed, C-float) ===");

    let pcm_in = synth_reference_pcm();
    let packets = encode_with_ropus(&pcm_in);
    assert!(
        mean_square_energy(&pcm_in) > CONTROL_SIGNAL_MIN_MEAN_SQUARE,
        "lossless control input is not energetic"
    );
    let packet_fingerprint = packet_stream_fingerprint(&packets);
    assert_ne!(
        packet_fingerprint, 0,
        "encoded lossless packet stream is empty"
    );

    let tmp = ctrl_tmp_dir();
    let packets_path = tmp.path().join("ctrl_packets_lossless.bin");
    write_packets_file(&packets_path, &packets, |_| false).expect("write packets");
    let packet_file_before = std::fs::read(&packets_path).expect("read lossless packet file");

    let fixed_pcm_path = tmp.path().join("ctrl_fixed_lossless.pcm");
    let float_pcm_path = tmp.path().join("ctrl_float_lossless.pcm");
    run_decoder(
        "ropus-harness",
        "ctrl_decode_fixed",
        &packets_path,
        &fixed_pcm_path,
        FIXED_MODE_MARKER,
    );
    run_decoder(
        "ropus-harness-deep-plc",
        "ctrl_decode_float",
        &packets_path,
        &float_pcm_path,
        FLOAT_MODE_MARKER,
    );

    assert_eq!(
        std::fs::read(&packets_path).expect("read lossless packets after decode"),
        packet_file_before,
        "lossless control decoder mutated the shared packet stream"
    );

    let pcm_fixed = read_pcm_file(&fixed_pcm_path).expect("read fixed pcm");
    let pcm_float = read_pcm_file(&float_pcm_path).expect("read float pcm");
    let snr = compute_snr_db(&pcm_float, &pcm_fixed);
    let first_diverge = first_divergent(&pcm_float, &pcm_fixed);
    let fixed_energy = mean_square_energy(&pcm_fixed);
    let float_energy = mean_square_energy(&pcm_float);

    eprintln!("===");
    eprintln!(
        "  SNR(fixed vs float, NO loss) = {:.2} dB, first diverge at {:?}",
        snr, first_diverge
    );
    eprintln!("  mean-square energy fixed={fixed_energy:.1}, float={float_energy:.1}");
    eprintln!("  packet fingerprint = {packet_fingerprint:016x}");
    eprintln!("===");

    assert!(
        fixed_energy > CONTROL_OUTPUT_MIN_MEAN_SQUARE
            && float_energy > CONTROL_OUTPUT_MIN_MEAN_SQUARE,
        "lossless control output lost signal energy: fixed={fixed_energy:.1}, float={float_energy:.1}"
    );
    assert!(
        first_diverge.is_some(),
        "lossless fixed/float control unexpectedly produced identical PCM"
    );
    assert!(
        snr >= CONTROL_LOSSLESS_MIN_DB,
        "Lossless SNR {snr:.2} dB is below the fixed/float diagnostic baseline \
         of {CONTROL_LOSSLESS_MIN_DB:.0} dB — first divergent sample {first_diverge:?}"
    );
}
