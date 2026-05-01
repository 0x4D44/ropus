#![no_main]
use libfuzzer_sys::arbitrary::{self, Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use ropus::opus::decoder::OpusDecoder;
use ropus::opus::encoder::{
    OpusEncoder, OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY,
    OPUS_APPLICATION_VOIP,
};
use std::cell::RefCell;
use std::sync::Once;

#[path = "c_reference.rs"]
mod c_reference;

// --------------------------------------------------------------------------- //
// Panic-capture: on Windows, Rust assertions in libfuzzer-sys trigger __fastfail
// which bypasses libFuzzer's crash-artifact writer. Install a panic hook that
// saves the current input to FUZZ_PANIC_CAPTURE_DIR (or `fuzz_crashes/`) so we
// get reproducible crash files even when libFuzzer fails to persist them.
// --------------------------------------------------------------------------- //
thread_local! {
    static CURRENT_INPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn init_panic_capture() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            CURRENT_INPUT.with(|cell| {
                let bytes = cell.borrow();
                if !bytes.is_empty() {
                    use std::collections::hash_map::DefaultHasher;
                    use std::hash::{Hash, Hasher};
                    let mut hasher = DefaultHasher::new();
                    bytes.hash(&mut hasher);
                    let hash = hasher.finish();
                    let dir = std::env::var("FUZZ_PANIC_CAPTURE_DIR")
                        .unwrap_or_else(|_| "fuzz_crashes".to_string());
                    let _ = std::fs::create_dir_all(&dir);
                    let path = std::path::Path::new(&dir).join(format!("crash_{:016x}.bin", hash));
                    match std::fs::write(&path, bytes.as_slice()) {
                        Ok(()) => eprintln!(
                            "[PANIC CAPTURE] Saved {} bytes to {}",
                            bytes.len(),
                            path.display()
                        ),
                        Err(e) => eprintln!(
                            "[PANIC CAPTURE] Failed to write {}: {}",
                            path.display(),
                            e
                        ),
                    }
                }
            });
            prev(info);
        }));
    });
}

const SAMPLE_RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];
const APPLICATIONS: [i32; 3] = [
    OPUS_APPLICATION_VOIP,
    OPUS_APPLICATION_AUDIO,
    OPUS_APPLICATION_RESTRICTED_LOWDELAY,
];

/// Map a u16 to a bitrate in the valid Opus range (6000..=510000).
/// Mirrors `byte_to_bitrate` from Stream A's encode targets.
fn raw_to_bitrate(raw: u16) -> i32 {
    6000 + (raw as i32 % 504_001)
}

/// 16-byte digest of the structured input — enough to identify the shape of
/// a crash (rates, channel count, frame count, total PCM size, first-frame
/// config) without the ~41 MB/s allocation churn that `format!("{:?}", ...)`
/// on a 61 KB-per-iteration input produced. The libFuzzer artifact directory
/// already preserves the raw bytes that triggered the panic; this is the
/// supplementary hint the panic hook writes alongside.
fn input_fingerprint(input: &MultiframeInput) -> [u8; 16] {
    let mut fp = [0u8; 16];
    fp[0] = input.sample_rate_idx;
    fp[1] = input.channels_minus_one;
    fp[2] = input.application_idx;
    fp[3] = input.frames.len() as u8;
    let total_pcm: usize = input.frames.iter().map(|f| f.pcm_bytes.len()).sum();
    fp[4..8].copy_from_slice(&(total_pcm as u32).to_le_bytes());
    if let Some(first) = input.frames.first() {
        fp[8..10].copy_from_slice(&first.bitrate_raw.to_le_bytes());
        fp[10] = first.complexity;
        fp[11] = first.fec;
        fp[12] = first.loss_perc;
        fp[13] = first.vbr as u8 | ((first.dtx as u8) << 1);
    }
    fp
}

// --------------------------------------------------------------------------- //
// Structured input — `Arbitrary` lets libFuzzer produce well-shaped inputs that
// the modulo-byte mutator on the original target couldn't reach. Frame count
// and per-frame PCM length are explicitly bounded so each iteration completes
// in <200 ms and inputs stay within libFuzzer's max_len budget.
// --------------------------------------------------------------------------- //
#[derive(Debug)]
struct MultiframeInput {
    sample_rate_idx: u8,
    channels_minus_one: u8,
    application_idx: u8,
    frames: Vec<FrameConfig>,
}

#[derive(Debug)]
struct FrameConfig {
    bitrate_raw: u16,
    complexity: u8,
    vbr: bool,
    fec: u8,
    dtx: bool,
    loss_perc: u8,
    pcm_bytes: Vec<u8>,
}

impl<'a> Arbitrary<'a> for MultiframeInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let sample_rate_idx = u.int_in_range(0..=4)?;
        let channels_minus_one = u.int_in_range(0..=1)?;
        let application_idx = u.int_in_range(0..=2)?;
        let n_frames = u.int_in_range(2..=16)?;

        let sample_rate = SAMPLE_RATES[sample_rate_idx as usize];
        let channels = channels_minus_one as i32 + 1;
        let frame_size = (sample_rate / 50) as usize;
        let pcm_bytes_needed = frame_size * channels as usize * 2;

        let mut frames = Vec::with_capacity(n_frames);
        for _ in 0..n_frames {
            frames.push(FrameConfig::arbitrary_for_shape(u, pcm_bytes_needed)?);
        }
        Ok(Self {
            sample_rate_idx,
            channels_minus_one,
            application_idx,
            frames,
        })
    }
}

impl FrameConfig {
    fn arbitrary_for_shape<'a>(
        u: &mut Unstructured<'a>,
        pcm_bytes_needed: usize,
    ) -> arbitrary::Result<Self> {
        Ok(Self {
            bitrate_raw: u.arbitrary()?,
            // Cap at 9 to dodge the analysis.c divergence class
            // (Campaign 9, 2026-04-19): C builds with DISABLE_FLOAT_API=off,
            // so complexity ≥ 10 ∧ sr ≥ 16000 ∧ app != RESTRICTED_SILK
            // produces an AnalysisInfo on the C side that the Rust port
            // doesn't yet emit (analysis stage 6 still in flight).
            complexity: u.int_in_range(0..=9)?,
            vbr: u.arbitrary()?,
            fec: u.int_in_range(0..=2)?,
            dtx: u.arbitrary()?,
            loss_perc: u.int_in_range(0..=100)?,
            pcm_bytes: u.bytes(pcm_bytes_needed)?.to_vec(),
        })
    }
}

// Multiframe differential fuzz target.
//
// Encodes 2-16 sequential frames through a single encoder instance, applying a
// fresh setter shuffle (bitrate / complexity / vbr / fec / dtx / loss_perc)
// before each frame to model intra-stream config drift (adaptive bitrate,
// FEC toggling on push-to-talk). Compares Rust vs C reference frame-by-frame.
// This catches state-accumulation bugs that single-frame fuzzing is
// structurally blind to — the encoder carries prediction state, gain history,
// and mode decisions across frames.
fuzz_target!(|input: MultiframeInput| {
    init_panic_capture();
    // Best-effort capture of a 16-byte fingerprint for crash repro. libFuzzer
    // keeps the raw byte input in the artifact directory; we just write a
    // shape hint (rates, frame count, total PCM size, first-frame config).
    CURRENT_INPUT.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.clear();
        buf.extend_from_slice(&input_fingerprint(&input));
    });

    let sample_rate = SAMPLE_RATES[input.sample_rate_idx as usize];
    let channels = input.channels_minus_one as i32 + 1;
    let application = APPLICATIONS[input.application_idx as usize];
    let frame_size = sample_rate / 50; // 20 ms
    let samples_per_frame = frame_size as usize * channels as usize;

    // Decode each frame's PCM bytes to i16 samples. The Arbitrary impl
    // guarantees pcm_bytes.len() == samples_per_frame * 2, so chunks_exact
    // never drops a tail.
    let mut pcm_frames: Vec<Vec<i16>> = Vec::with_capacity(input.frames.len());
    for fc in &input.frames {
        let pcm: Vec<i16> = fc
            .pcm_bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        pcm_frames.push(pcm);
    }

    // Per-frame C-side configs mirror the Rust setter calls below.
    let frame_cfgs: Vec<c_reference::CEncodeConfig> = input
        .frames
        .iter()
        .map(|fc| c_reference::CEncodeConfig {
            bitrate: raw_to_bitrate(fc.bitrate_raw),
            complexity: fc.complexity as i32,
            application,
            vbr: if fc.vbr { 1 } else { 0 },
            vbr_constraint: 0,
            inband_fec: fc.fec as i32,
            dtx: if fc.dtx { 1 } else { 0 },
            loss_perc: fc.loss_perc as i32,
        })
        .collect();

    // --- Rust multiframe encode ---
    let mut rust_enc = match OpusEncoder::new(sample_rate, channels, application) {
        Ok(e) => e,
        Err(_) => return,
    };

    let mut rust_packets: Vec<Vec<u8>> = Vec::with_capacity(input.frames.len());
    for (pcm, cfg) in pcm_frames.iter().zip(frame_cfgs.iter()) {
        rust_enc.set_bitrate(cfg.bitrate);
        rust_enc.set_vbr(cfg.vbr);
        rust_enc.set_vbr_constraint(cfg.vbr_constraint);
        rust_enc.set_inband_fec(cfg.inband_fec);
        rust_enc.set_dtx(cfg.dtx);
        rust_enc.set_packet_loss_perc(cfg.loss_perc);
        rust_enc.set_complexity(cfg.complexity);

        let mut out = vec![0u8; 4000];
        match rust_enc.encode(pcm, frame_size, &mut out, 4000) {
            Ok(len) => {
                out.truncate(len as usize);
                rust_packets.push(out);
            }
            Err(_) => return,
        }
    }

    // --- C reference multiframe encode ---
    let frame_refs: Vec<&[i16]> = pcm_frames.iter().map(|f| f.as_slice()).collect();
    let c_packets = match c_reference::c_encode_multiframe(
        &frame_refs,
        &frame_cfgs,
        frame_size,
        sample_rate,
        channels,
    ) {
        Ok(p) => p,
        Err(_) => return,
    };

    // --- Differential comparison: frame-by-frame byte-exact ---
    assert_eq!(
        rust_packets.len(),
        c_packets.len(),
        "Frame count mismatch: Rust={}, C={}, sr={sample_rate}, ch={channels}, frames={}",
        rust_packets.len(),
        c_packets.len(),
        input.frames.len(),
    );

    for (i, (rust_pkt, c_pkt)) in rust_packets.iter().zip(c_packets.iter()).enumerate() {
        let cfg = &frame_cfgs[i];
        assert_eq!(
            rust_pkt.len(),
            c_pkt.len(),
            "Frame {i}/{} length mismatch: Rust={}, C={}, \
             sr={sample_rate}, ch={channels}, app={application}, \
             br={}, cx={}, vbr={}, fec={}, dtx={}, loss={}",
            input.frames.len(),
            rust_pkt.len(),
            c_pkt.len(),
            cfg.bitrate,
            cfg.complexity,
            cfg.vbr,
            cfg.inband_fec,
            cfg.dtx,
            cfg.loss_perc,
        );

        assert_eq!(
            rust_pkt, c_pkt,
            "Frame {i}/{} byte mismatch: \
             sr={sample_rate}, ch={channels}, app={application}, \
             br={}, cx={}, vbr={}, fec={}, dtx={}, loss={}, len={}",
            input.frames.len(),
            cfg.bitrate,
            cfg.complexity,
            cfg.vbr,
            cfg.inband_fec,
            cfg.dtx,
            cfg.loss_perc,
            rust_pkt.len(),
        );
    }

    // --- Semantic invariant: each encoded packet is parseable ---
    for (i, pkt) in rust_packets.iter().enumerate() {
        if !pkt.is_empty() {
            match ropus::opus::decoder::opus_packet_get_nb_frames(pkt) {
                Ok(n) if n > 0 => {}
                Ok(n) => panic!(
                    "Frame {i}: encoded packet reports {n} frames, len={}",
                    pkt.len()
                ),
                Err(e) => panic!(
                    "Frame {i}: encoded packet not parseable, len={}, err={e}",
                    pkt.len()
                ),
            }
        }
    }

    // --- Semantic invariant: decode the encoded packets, verify sample counts ---
    let mut rust_dec = match OpusDecoder::new(sample_rate, channels) {
        Ok(d) => d,
        Err(_) => return,
    };
    for (i, pkt) in rust_packets.iter().enumerate() {
        let mut decoded = vec![0i16; samples_per_frame];
        match rust_dec.decode(Some(pkt), &mut decoded, frame_size, false) {
            Ok(samples) => {
                assert_eq!(
                    samples as usize,
                    frame_size as usize,
                    "Frame {i}: decoded sample count {samples} != frame_size {frame_size}"
                );
            }
            Err(e) => {
                panic!("Frame {i}: Rust failed to decode its own encoded packet: {e}");
            }
        }
    }
});
