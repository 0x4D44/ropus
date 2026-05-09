#![cfg(not(no_reference))]
//! Stage 2 (TDD) — DRED bitrate-plumbing differential test.
//!
//! Exercises the F33/F48/F53/quantizer fixes by encoding the same fixed
//! PCM stream through both the Rust `OpusEncoder` and the xiph C reference
//! (via the existing `dred_encode_shim.c` harness) with DRED active, and
//! asserts the per-frame packet bytes match.
//!
//! ## Configuration parity
//!
//! The C shim (`ropus_test_c_encoder_new`) hardcodes:
//!   - bitrate           = 32_000 bps
//!   - complexity        = 5
//!   - FEC               = on
//!   - packet_loss_perc  = 20
//!   - VBR               = (default; not disabled — shim does NOT call
//!                          `OPUS_SET_VBR(0)`)
//!   - dred_duration     = caller-supplied
//!
//! The Stage 2 HLD asks for `loss=15, VBR off, 5 frames, 32 kbps`. We
//! cannot disable VBR or change `loss` without editing the shim — both of
//! which are out of scope for Stage 2. We therefore use the shim's fixed
//! parameters and configure the Rust side to match exactly. Stage 3
//! reviewers may grow the shim if needed.
//!
//! ## Failure semantics
//!
//! The two functions under test (`compute_dred_bitrate`,
//! `estimate_dred_bitrate`) and the call-site fixes don't exist yet —
//! Stage 2 only stubs them with `unimplemented!()`. Encoding a frame
//! through the Rust path will therefore panic during Stage 2 runs.
//! This is the desired Stage 2 signal. Once Stage 3 lands, this test
//! must pass byte-equal — and if f32 drift turns up, the assertion
//! drops to a Tier-2 SNR check with explicit `eprintln!` diagnostic
//! and a comment naming the suspect f32 op (per HLD §6 risk row 1).

use std::fs;
use std::path::PathBuf;

use ropus::dnn::embedded_weights::WEIGHTS_BLOB;
use ropus::opus::encoder::{OPUS_APPLICATION_VOIP as ROPUS_APP_VOIP, OpusEncoder};

use ropus_harness_deep_plc::{
    OPUS_APPLICATION_VOIP, ropus_test_c_encoder_encode, ropus_test_c_encoder_free,
    ropus_test_c_encoder_new,
};

const TARGET_FS: i32 = 48000;
const FRAME_SIZE: i32 = 960; // 20 ms at 48 kHz.
const MAX_PACKET: usize = 1500;
const NUM_FRAMES: usize = 5;
const DRED_DURATION_2_5MS: i32 = 100; // 250 ms of DRED payload.

// Mirror the C shim's hard-coded knobs (see comment at top of file).
const SHIM_BITRATE: i32 = 32_000;
const SHIM_COMPLEXITY: i32 = 5;
const SHIM_PACKET_LOSS: i32 = 20;
const SHIM_USE_FEC: i32 = 1;

// ---------------------------------------------------------------------------
// WAV utility (copied from `dred_integrated_encode.rs`).
// ---------------------------------------------------------------------------

struct Wav {
    sample_rate: u32,
    channels: u16,
    samples: Vec<i16>,
}

fn read_wav(path: &PathBuf) -> Wav {
    let data = fs::read(path).unwrap_or_else(|e| {
        panic!("cannot read {}: {}", path.display(), e);
    });
    assert!(data.len() >= 44, "WAV too small");
    assert_eq!(&data[0..4], b"RIFF");
    assert_eq!(&data[8..12], b"WAVE");
    let mut pos = 12;
    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut bits_per_sample = 0u16;
    let mut pcm: Vec<i16> = Vec::new();
    while pos + 8 <= data.len() {
        let id = &data[pos..pos + 4];
        let sz = u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]])
            as usize;
        if id == b"fmt " {
            channels = u16::from_le_bytes([data[pos + 10], data[pos + 11]]);
            sample_rate = u32::from_le_bytes([
                data[pos + 12],
                data[pos + 13],
                data[pos + 14],
                data[pos + 15],
            ]);
            bits_per_sample = u16::from_le_bytes([data[pos + 22], data[pos + 23]]);
        } else if id == b"data" {
            assert_eq!(bits_per_sample, 16, "only 16-bit PCM supported");
            let bytes = &data[pos + 8..pos + 8 + sz];
            pcm = bytes
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect();
        }
        pos += 8 + sz;
        if !sz.is_multiple_of(2) {
            pos += 1;
        }
    }
    assert!(sample_rate > 0, "no fmt chunk");
    assert!(!pcm.is_empty(), "no data chunk");
    Wav {
        sample_rate,
        channels,
        samples: pcm,
    }
}

fn vectors_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("tests")
        .join("vectors")
        .join(name)
}

fn weights_or_skip(tag: &str) -> bool {
    if WEIGHTS_BLOB.is_empty() {
        eprintln!("{tag}: WEIGHTS_BLOB empty — skipping. Run `cargo run -p fetch-assets -- all`.");
        return false;
    }
    true
}

fn encode_rust_frames(samples: &[i16]) -> Vec<Vec<u8>> {
    let mut enc = OpusEncoder::new(TARGET_FS, 1, ROPUS_APP_VOIP)
        .expect("OpusEncoder::new(48k, mono, VOIP) should succeed");
    assert_eq!(enc.set_bitrate(SHIM_BITRATE), 0);
    assert_eq!(enc.set_complexity(SHIM_COMPLEXITY), 0);
    assert_eq!(enc.set_inband_fec(SHIM_USE_FEC), 0);
    assert_eq!(enc.set_packet_loss_perc(SHIM_PACKET_LOSS), 0);
    assert_eq!(enc.set_dred_duration(DRED_DURATION_2_5MS), 0);

    let mut out = Vec::with_capacity(NUM_FRAMES);
    for i in 0..NUM_FRAMES {
        let start = i * FRAME_SIZE as usize;
        let end = start + FRAME_SIZE as usize;
        assert!(end <= samples.len(), "wav too short for {NUM_FRAMES} frames");
        let mut packet = vec![0u8; MAX_PACKET];
        let nb = enc
            .encode(&samples[start..end], FRAME_SIZE, &mut packet, MAX_PACKET as i32)
            .expect("encode should succeed");
        packet.truncate(nb as usize);
        out.push(packet);
    }
    out
}

fn encode_c_frames(samples: &[i16]) -> Vec<Vec<u8>> {
    let enc = unsafe {
        ropus_test_c_encoder_new(TARGET_FS, 1, OPUS_APPLICATION_VOIP, DRED_DURATION_2_5MS)
    };
    assert!(!enc.is_null(), "ropus_test_c_encoder_new returned NULL");
    let mut out = Vec::with_capacity(NUM_FRAMES);
    for i in 0..NUM_FRAMES {
        let start = i * FRAME_SIZE as usize;
        let end = start + FRAME_SIZE as usize;
        assert!(end <= samples.len(), "wav too short for {NUM_FRAMES} frames");
        let mut packet = vec![0u8; MAX_PACKET];
        let nb = unsafe {
            ropus_test_c_encoder_encode(
                enc,
                samples[start..end].as_ptr(),
                FRAME_SIZE,
                packet.as_mut_ptr(),
                MAX_PACKET as i32,
            )
        };
        assert!(nb > 0, "C encoder returned non-positive: {nb}");
        packet.truncate(nb as usize);
        out.push(packet);
    }
    unsafe { ropus_test_c_encoder_free(enc) };
    out
}

// ---------------------------------------------------------------------------
// The differential.
// ---------------------------------------------------------------------------

#[test]
fn rust_and_c_dred_packets_match_byte_for_byte() {
    if !weights_or_skip("dred_bitrate_plumbing_diff") {
        return;
    }

    let wav_path = vectors_path("48000hz_mono_sine440.wav");
    let wav = read_wav(&wav_path);
    assert_eq!(wav.sample_rate, TARGET_FS as u32);
    assert_eq!(wav.channels, 1);

    // Encode the same first 5 frames with both encoders. After Stage 3 lands
    // F33 + F48 + F53 + the per-frame quantizer fix, this loop must complete.
    // During Stage 2 it panics inside `encode` because the stubs are
    // `unimplemented!()` — that's the intended TDD failure.
    let rust_packets = encode_rust_frames(&wav.samples);
    let c_packets = encode_c_frames(&wav.samples);

    assert_eq!(rust_packets.len(), c_packets.len());
    for (i, (rs, cs)) in rust_packets.iter().zip(c_packets.iter()).enumerate() {
        assert_eq!(
            rs, cs,
            "frame {i} mismatch: \n  rust ({} bytes): {:02x?}\n  c    ({} bytes): {:02x?}",
            rs.len(),
            rs,
            cs.len(),
            cs
        );
    }
}
