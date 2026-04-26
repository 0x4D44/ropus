//! Stage 7b.3 control experiment: measure the fixed-vs-float arithmetic-gap
//! ceiling of classical SILK PLC under the exact tier-2 test conditions.
//!
//! ## What this test decides
//!
//! Tier-2 SNR (ropus vs C-float with DEEP_PLC) currently sits at 51.79 dB. The
//! HLD gate is 60 dB. The devils-advocate hypothesis is that the 60 dB gate is
//! unachievable *by construction* because ropus is fixed-point and the
//! harness's C reference for tier-2 is float — ± 1 LSB quantisation drift on
//! every frame, amplified ~9× by the sLPC back-copy step in `silk/PLC.c`,
//! ceiling the achievable SNR at ~50 dB regardless of whether the neural PLC
//! path is bit-exact.
//!
//! If the hypothesis is right, the **classical** SILK PLC (no neural path at
//! all) should show a similar ceiling when compared across fixed vs float
//! under the same loss pattern — nothing but the 1-LSB arithmetic gap
//! multiplied by the IIR recurrence.
//!
//! ## How the test works
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
//! The tier-2 gate reports SNR(rust, float-C). This test reports
//! SNR(fixed-C, float-C). The relationship is: the tier-2 number can never
//! exceed the control number (the noise floor is the same 1-LSB gap
//! multiplied by the same IIR), modulo Rust vs C divergence on integer ops
//! — which is supposed to be zero (tier-1 bit-exact).
//!
//! ## Interpretation cheat-sheet
//!
//! - control ≈ 50 dB → 60 dB gate is ceiling-limited, amend HLD.
//! - control > 60 dB → ropus still has a port bug biting us.
//! - 55-65 dB → partial: some ceiling, some residual bug.

use std::env;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

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

const LOST_BIT: u32 = 0x8000_0000;

// Loss pattern is byte-for-byte identical to `tier2_snr.rs::is_lost`.
fn is_lost(frame_idx: usize) -> bool {
    frame_idx > 0 && frame_idx.is_multiple_of(7)
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

fn first_divergent(a: &[i16], b: &[i16]) -> Option<usize> {
    a.iter().zip(b.iter()).position(|(x, y)| x != y)
}

// ---------------------------------------------------------------------------
// Tempfile management — we keep the paths under `target/tmp/` so they survive
// the test's working-directory reset across `cargo` invocations and aren't
// scattered under the OS tempdir.
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

fn ctrl_tmp_dir() -> PathBuf {
    let root = workspace_root();
    let dir = root.join("target").join("harness-control-tmp");
    std::fs::create_dir_all(&dir).expect("create target/harness-control-tmp");
    dir
}

fn run_decoder(package: &str, bin: &str, packets: &Path, pcm_out: &Path) {
    // Use `cargo run` rather than locating the binary directly — keeps this
    // test agnostic to the target triple + release vs debug dir names. The
    // dep binaries rebuild only if they or their C sources changed, so the
    // first invocation pays a one-time cost, subsequent runs are cheap.
    let t0 = Instant::now();
    let status = Command::new(env!("CARGO"))
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
        .current_dir(workspace_root())
        .status()
        .unwrap_or_else(|e| panic!("cargo run -p {package} --bin {bin} failed: {e}"));
    let elapsed = t0.elapsed();
    assert!(
        status.success(),
        "cargo run -p {package} --bin {bin} exited with {status:?} ({elapsed:?})"
    );
    eprintln!("  {bin}: decoded in {elapsed:.2?}");
}

// ---------------------------------------------------------------------------
// The test itself.
// ---------------------------------------------------------------------------

#[test]
fn ctrl_fixed_vs_float_classical_snr() {
    eprintln!("=== Stage 7b.3 control experiment: C-fixed-classical vs C-float-classical ===");

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

    // 2. Write packets to a tempfile. Both decoders read the same bytes so
    // there's no chance the two sides see a different frame-by-frame stream.
    let tmp = ctrl_tmp_dir();
    let packets_path = tmp.join("ctrl_packets.bin");
    write_packets_file(&packets_path, &packets, is_lost).expect("write packets file");
    eprintln!(
        "  wrote {} frames ({} lost) to {}",
        packets.len(),
        n_lost,
        packets_path.display()
    );

    // 3. Run both decoders.
    let fixed_pcm_path = tmp.join("ctrl_fixed.pcm");
    let float_pcm_path = tmp.join("ctrl_float.pcm");
    run_decoder(
        "ropus-harness",
        "ctrl_decode_fixed",
        &packets_path,
        &fixed_pcm_path,
    );
    run_decoder(
        "ropus-harness-deep-plc",
        "ctrl_decode_float",
        &packets_path,
        &float_pcm_path,
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

    let snr = compute_snr_db(&pcm_float, &pcm_fixed);
    let first_diverge = first_divergent(&pcm_float, &pcm_fixed);

    eprintln!("===");
    eprintln!("  n_lost          = {n_lost}");
    eprintln!("  total samples   = {}", pcm_fixed.len());
    eprintln!("  SNR(fixed vs float, classical PLC) = {:.2} dB", snr);
    eprintln!("  first divergent sample index = {first_diverge:?}");
    eprintln!("===");

    // The control experiment has no pass/fail threshold — the measured SNR
    // IS the answer. But we sanity-check against catastrophe (any value
    // below 20 dB would indicate something catastrophically wrong with the
    // harness itself, not the arithmetic gap we're trying to measure).
    assert!(
        snr > 20.0,
        "control SNR {snr:.2} dB is catastrophically low — the harness is broken, \
         not measuring the arithmetic gap. Inspect first divergent sample {first_diverge:?}."
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

    let tmp = ctrl_tmp_dir();
    let packets_path = tmp.join("ctrl_packets_lossless.bin");
    write_packets_file(&packets_path, &packets, |_| false).expect("write packets");

    let fixed_pcm_path = tmp.join("ctrl_fixed_lossless.pcm");
    let float_pcm_path = tmp.join("ctrl_float_lossless.pcm");
    run_decoder(
        "ropus-harness",
        "ctrl_decode_fixed",
        &packets_path,
        &fixed_pcm_path,
    );
    run_decoder(
        "ropus-harness-deep-plc",
        "ctrl_decode_float",
        &packets_path,
        &float_pcm_path,
    );

    let pcm_fixed = read_pcm_file(&fixed_pcm_path).expect("read fixed pcm");
    let pcm_float = read_pcm_file(&float_pcm_path).expect("read float pcm");
    let snr = compute_snr_db(&pcm_float, &pcm_fixed);
    let first_diverge = first_divergent(&pcm_float, &pcm_fixed);

    eprintln!("===");
    eprintln!(
        "  SNR(fixed vs float, NO loss) = {:.2} dB, first diverge at {:?}",
        snr, first_diverge
    );
    eprintln!("===");

    // Loose lower bound: the lossless classical path across fixed/float
    // should be very close to identical — 60 dB gives plenty of slack
    // (tier2_snr.rs's lossless regression uses 80 dB against Rust-fixed).
    assert!(
        snr > 60.0,
        "Lossless SNR {snr:.2} dB is below a safe baseline — something is \
         wrong beyond the arithmetic gap we're trying to measure."
    );
}
