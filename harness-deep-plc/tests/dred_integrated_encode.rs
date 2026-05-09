#![cfg(not(no_reference))]
//! Stage 8.8 acceptance gate — integrated `OpusEncoder` wiring for DRED.
//!
//! Three format-level cross-compatibility tests (the HLD's exit criterion
//! is "packets C can parse and vice versa", not byte-exactness; 8.6's
//! upstream LPCNet drift still blocks byte-exact end-to-end):
//!
//! 1. **Rust encoder → C parser.** Drive the Rust `OpusEncoder` with
//!    `set_dred_duration(100)` and encode a WAV frame. Feed the emitted
//!    packet to the C reference's `opus_dred_parse`. Assert success and
//!    sensible `OpusDRED` fields (`nb_latents > 0`, `process_stage >= 1`).
//!
//! 2. **C encoder → Rust parser.** Mirror image — drive the C encoder via
//!    `ropus_test_c_encoder_*`, parse with Rust `OpusDREDDecoder::parse`,
//!    assert the Rust side also sees a populated extension.
//!
//! 3. **Rust encoder → Rust parser.** Self-consistent round-trip. Confirms
//!    the encoder emits what the matching Rust decoder expects to see
//!    without relying on either C path.
//!
//! All three operate on the same 48 kHz mono sine-wave WAV as 8.6's
//! payload-diff test, so if a future upstream LPCNet fix flips 8.6 from
//! ignored to passing, these tests also tighten automatically.

use std::fs;
use std::path::PathBuf;

use ropus::dnn::dred::DRED_MAX_FRAMES;
use ropus::dnn::embedded_weights::WEIGHTS_BLOB;
use ropus::opus::dred::OpusDREDDecoder;
use ropus::opus::encoder::OPUS_APPLICATION_VOIP as ROPUS_APP_VOIP;
use ropus::opus::encoder::OpusEncoder;

use ropus_harness_deep_plc::{
    OPUS_APPLICATION_VOIP, ropus_test_c_dred_parse, ropus_test_c_encoder_encode,
    ropus_test_c_encoder_free, ropus_test_c_encoder_new,
};

// ---------------------------------------------------------------------------
// WAV utility + fixtures (copied from the 8.6 test — pure-tone input keeps
// the VAD mostly active so DRED always has something to emit).
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

// ---------------------------------------------------------------------------
// Encode N frames with the Rust `OpusEncoder` using DRED. Returns the
// per-frame packet byte vectors.
// ---------------------------------------------------------------------------

const DRED_DURATION_2_5MS: i32 = 100; // 250 ms of payload (matches HLD example).
const TARGET_FS: i32 = 48000;
const FRAME_SIZE: i32 = 960; // 20 ms @ 48 kHz.
const MAX_PACKET: usize = 1500;

fn encode_rust_frames(samples: &[i16], channels: i32, num_frames: usize) -> Vec<Vec<u8>> {
    let mut enc = OpusEncoder::new(TARGET_FS, channels, ROPUS_APP_VOIP)
        .expect("OpusEncoder::new(48k, mono, VOIP) should succeed");
    // Match the C shim's setup verbatim. FEC + non-zero packet loss are
    // required to make `compute_dred_bitrate` allocate any chunks at all
    // — without them, both Rust and C set `dred_target_chunks = 0` and
    // skip DRED emission entirely.
    assert_eq!(enc.set_bitrate(32000), 0);
    assert_eq!(enc.set_complexity(5), 0);
    assert_eq!(enc.set_inband_fec(1), 0);
    assert_eq!(enc.set_packet_loss_perc(20), 0);
    assert_eq!(enc.set_dred_duration(DRED_DURATION_2_5MS), 0);
    assert_eq!(enc.get_dred_duration(), DRED_DURATION_2_5MS);

    let mut out = Vec::with_capacity(num_frames);
    let frame_stride = (FRAME_SIZE as usize) * (channels as usize);
    for i in 0..num_frames {
        let start = i * frame_stride;
        let end = start + frame_stride;
        assert!(
            end <= samples.len(),
            "wav too short for {num_frames} frames"
        );
        let frame = &samples[start..end];
        let mut packet = vec![0u8; MAX_PACKET];
        let nb = enc
            .encode(frame, FRAME_SIZE, &mut packet, MAX_PACKET as i32)
            .expect("encode should succeed");
        packet.truncate(nb as usize);
        out.push(packet);
    }
    out
}

// Encode with the C reference (via shim) — mirror of `encode_rust_frames`.
fn encode_c_frames(samples: &[i16], channels: i32, num_frames: usize) -> Vec<Vec<u8>> {
    let enc = unsafe {
        ropus_test_c_encoder_new(
            TARGET_FS,
            channels,
            OPUS_APPLICATION_VOIP,
            DRED_DURATION_2_5MS,
        )
    };
    assert!(!enc.is_null(), "ropus_test_c_encoder_new returned NULL");
    let mut out = Vec::with_capacity(num_frames);
    let frame_stride = (FRAME_SIZE as usize) * (channels as usize);
    for i in 0..num_frames {
        let start = i * frame_stride;
        let end = start + frame_stride;
        assert!(
            end <= samples.len(),
            "wav too short for {num_frames} frames"
        );
        let frame = &samples[start..end];
        let mut packet = vec![0u8; MAX_PACKET];
        let nb = unsafe {
            ropus_test_c_encoder_encode(
                enc,
                frame.as_ptr(),
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
// Test 1: Rust encoder → C parser.
// ---------------------------------------------------------------------------

#[test]
fn rust_encoded_packets_parse_on_c_reference() {
    if !weights_or_skip("dred_integrated_encode") {
        return;
    }

    let wav_path = vectors_path("48000hz_mono_sine440.wav");
    let wav = read_wav(&wav_path);
    assert_eq!(wav.sample_rate, TARGET_FS as u32);
    assert_eq!(wav.channels, 1);

    // Encode ~500 ms — enough for the DRED buffer to accumulate chunks.
    let frames = encode_rust_frames(&wav.samples, 1, 25);

    // Walk frames until we find one with a DRED extension the C parser
    // acknowledges. DRED doesn't emit on every frame (warm-up + activity
    // gating), so we scan rather than hard-assert on index 0.
    let mut found = false;
    let mut per_frame_nb_latents = Vec::new();
    for (i, pkt) in frames.iter().enumerate() {
        let mut nb_latents: i32 = -1;
        let mut process_stage: i32 = -1;
        let mut dred_offset: i32 = -1;
        let ret = unsafe {
            ropus_test_c_dred_parse(
                pkt.as_ptr(),
                pkt.len() as i32,
                2 * TARGET_FS,
                TARGET_FS,
                &mut nb_latents,
                &mut process_stage,
                &mut dred_offset,
            )
        };
        eprintln!(
            "[rust→C] frame {i} ({} bytes): parse_ret={ret} nb_latents={nb_latents} \
             process_stage={process_stage} dred_offset={dred_offset}",
            pkt.len()
        );
        per_frame_nb_latents.push(nb_latents);
        if ret >= 0 && nb_latents >= 1 {
            assert!(
                process_stage >= 1,
                "process_stage must be >= 1 on parsed DRED"
            );
            assert!(
                nb_latents as usize <= DRED_MAX_FRAMES,
                "nb_latents {nb_latents} exceeds DRED_MAX_FRAMES"
            );
            found = true;
        }
    }
    assert!(
        found,
        "expected at least one frame to carry a C-parseable DRED extension; \
         nb_latents per frame = {:?}",
        per_frame_nb_latents
    );
}

// ---------------------------------------------------------------------------
// Test 2: C encoder → Rust parser.
// ---------------------------------------------------------------------------

#[test]
fn c_encoded_packets_parse_with_rust_decoder() {
    if !weights_or_skip("dred_integrated_encode") {
        return;
    }

    let wav_path = vectors_path("48000hz_mono_sine440.wav");
    let wav = read_wav(&wav_path);
    let frames = encode_c_frames(&wav.samples, 1, 25);

    let decoder = OpusDREDDecoder::new();
    assert!(
        decoder.loaded(),
        "OpusDREDDecoder embedded weights should be available when WEIGHTS_BLOB is nonempty"
    );

    let mut found = false;
    let mut per_frame = Vec::new();
    for (i, pkt) in frames.iter().enumerate() {
        let dred = decoder
            .parse(pkt, 2 * TARGET_FS, TARGET_FS)
            .expect("parse should not error on a well-formed packet");
        eprintln!(
            "[C→rust] frame {i} ({} bytes): nb_latents={} process_stage={} dred_offset={}",
            pkt.len(),
            dred.nb_latents,
            dred.process_stage,
            dred.dred_offset
        );
        per_frame.push(dred.nb_latents);
        if dred.nb_latents >= 1 {
            assert!(dred.process_stage >= 1);
            assert!(dred.nb_latents as usize <= DRED_MAX_FRAMES);
            found = true;
        }
    }
    assert!(
        found,
        "expected at least one C-encoded frame to carry a Rust-parseable \
         DRED extension; nb_latents per frame = {:?}",
        per_frame
    );
}

// ---------------------------------------------------------------------------
// Test 3: Rust encoder → Rust parser (self-consistency).
// ---------------------------------------------------------------------------

#[test]
fn rust_encoder_decoder_dred_roundtrip() {
    if !weights_or_skip("dred_integrated_encode") {
        return;
    }

    let wav_path = vectors_path("48000hz_mono_sine440.wav");
    let wav = read_wav(&wav_path);
    let frames = encode_rust_frames(&wav.samples, 1, 25);

    let decoder = OpusDREDDecoder::new();
    assert!(decoder.loaded());

    let mut found_with_latents = false;
    for (i, pkt) in frames.iter().enumerate() {
        let mut dred = decoder
            .parse(pkt, 2 * TARGET_FS, TARGET_FS)
            .expect("parse should not error");
        if dred.nb_latents < 1 {
            continue;
        }
        // Found one — drive it through `process` to populate fec_features,
        // proving the full parse → process chain closes. Matches the
        // 8.7 round-trip test's contract.
        let ret = decoder.process(&mut dred);
        assert_eq!(
            ret, 0,
            "process should succeed on parsed packet (frame {i})"
        );
        assert_eq!(dred.process_stage, 2);
        // At least one feature must be nonzero on a live signal frame.
        let any_nonzero = dred.fec_features.iter().any(|f| *f != 0.0);
        assert!(
            any_nonzero,
            "fec_features all-zero after process on frame {i} — RDOVAE decoder not driven"
        );
        found_with_latents = true;
        break;
    }
    assert!(
        found_with_latents,
        "round-trip found no packet with DRED latents to drive process() on"
    );
}
