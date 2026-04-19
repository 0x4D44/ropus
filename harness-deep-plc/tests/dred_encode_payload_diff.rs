//! Stage 8.6 acceptance gate — byte-exact DRED payload test on a WAV input.
//!
//! Tier 1 target (per `wrk_docs/2026.04.19 - HLD - dred-port.md` staging
//! row 8.6): byte-exact against the xiph C reference on WAV input. There
//! is no meaningful tier 2 fallback for integer range-coded payload bytes
//! — if any byte diverges, the test fails and someone investigates.
//!
//! Setup:
//! - C side: `dred_encoder.c` + `dred_coding.c` linked via
//!   `harness-deep-plc/build.rs`, exposed through the opaque-handle
//!   `dred_encode_shim.c`. The C reference is compiled with
//!   `ENABLE_DEEP_PLC` + `!USE_WEIGHTS_FILE`, so `dred_encoder_init`
//!   auto-loads the compile-time RDOVAE / pitchdnn tables.
//! - Rust side: `DREDEnc::new` auto-loads the embedded blob (same
//!   upstream checkpoint both sides pin to). Then `compute_latents` +
//!   `encode_silk_frame` produce the same byte stream.
//!
//! WAV: `tests/vectors/48000hz_mono_sine440.wav` (1 s, 48 kHz mono, i16).
//! Using a pure tone keeps the VAD mostly-active so the "only silence"
//! early-return paths don't kick in and hide differential bugs.

use std::fs;
use std::os::raw::c_void;
use std::path::PathBuf;

use ropus::dnn::dred::{DRED_MAX_DATA_SIZE, DRED_MAX_FRAMES, DREDEnc};
use ropus::dnn::embedded_weights::WEIGHTS_BLOB;

use ropus_harness_deep_plc::{
    ropus_test_dred_compute_latents, ropus_test_dred_encode_silk_frame,
    ropus_test_dredenc_copy_input_buffer, ropus_test_dredenc_copy_latents,
    ropus_test_dredenc_copy_lpcnet_features, ropus_test_dredenc_copy_resample_mem,
    ropus_test_dredenc_copy_state, ropus_test_dredenc_dred_offset, ropus_test_dredenc_free,
    ropus_test_dredenc_input_buffer_fill, ropus_test_dredenc_latent_offset,
    ropus_test_dredenc_latents_buffer_fill, ropus_test_dredenc_new,
};

/// Convert a 16-bit PCM sample to the f32 scale the DRED encoder expects
/// (same convention `opus_encoder.c` uses — divide by 32768 before
/// invoking `dred_compute_latents`).
fn pcm_i16_to_f32(s: i16) -> f32 {
    s as f32 * (1.0 / 32768.0)
}

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

/// Guard: require the embedded DRED weight blob so both sides initialise
/// from the same compile-time checkpoint. Emit a loud skip on bare trees.
fn weights_or_skip() -> bool {
    if WEIGHTS_BLOB.is_empty() {
        eprintln!(
            "dred_encode_payload_diff: WEIGHTS_BLOB empty — skipping. \
             Run `cargo run -p fetch-assets -- all` to populate."
        );
        return false;
    }
    true
}

/// Locate `tests/vectors/48000hz_mono_sine440.wav` relative to the
/// workspace root. `CARGO_MANIFEST_DIR` is set per crate by cargo at
/// build time.
fn vectors_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("tests")
        .join("vectors")
        .join(name)
}

/// Index of the first diverging byte, or None on byte-exact.
fn first_byte_divergent(a: &[u8], b: &[u8]) -> Option<(usize, u8, u8)> {
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if x != y {
            return Some((i, x, y));
        }
    }
    if a.len() != b.len() {
        return Some((a.len().min(b.len()), 0, 0));
    }
    None
}

/// Index of the first diverging f32 (bit-pattern compare), None on
/// bit-exact. Used to diagnose where divergence starts (resampler vs
/// features vs RDOVAE) when the payload bytes don't line up.
fn first_f32_divergent(a: &[f32], b: &[f32]) -> Option<(usize, f32, f32)> {
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if x.to_bits() != y.to_bits() {
            return Some((i, x, y));
        }
    }
    None
}

#[test]
#[ignore = "Stage 7 LPCNet drift: Ex[]/FFT-path bit-exactness pending. See JRN 2026.04.19 stage8-dred-port 8.6."]
fn dred_encode_silk_frame_bytes_match_c_reference() {
    if !weights_or_skip() {
        return;
    }

    let path = vectors_path("48000hz_mono_sine440.wav");
    let wav = read_wav(&path);
    assert_eq!(wav.sample_rate, 48_000);
    assert_eq!(wav.channels, 1);

    let fs = wav.sample_rate as i32;
    let channels = wav.channels as i32;

    // --- Instantiate C + Rust DRED encoders ---
    let c_enc = unsafe { ropus_test_dredenc_new(fs, channels) };
    assert!(
        !c_enc.is_null(),
        "C dred_encoder_init failed (weights not loaded?)"
    );
    let mut r_enc = DREDEnc::new(fs, channels);
    assert!(
        r_enc.loaded,
        "Rust DREDEnc embedded-blob load failed — check gen_weights_blob.c"
    );

    // Sanity-check initial state matches C.
    {
        let c_fill = unsafe { ropus_test_dredenc_input_buffer_fill(c_enc as *const c_void) };
        assert_eq!(
            c_fill, r_enc.input_buffer_fill,
            "initial input_buffer_fill mismatch: C={} Rust={}",
            c_fill, r_enc.input_buffer_fill
        );
    }

    // --- Drive both encoders ---
    //
    // The C reference encodes DRED by processing `frame_size` PCM samples
    // per Opus frame (20 ms = 960 samples at 48 kHz), with a
    // `total_buffer` (= pre-roll + lookahead) passed as `extra_delay`.
    // For a standalone harness we pick a fixed 20 ms frame and a typical
    // Opus lookahead — 312 samples at 48 kHz (`Fs*6.5/1000`), the same
    // bound `opus_encoder.c` uses in its setup. The exact value only
    // matters if it matches on both sides; the test just needs
    // deterministic, identical inputs to both.
    let frame_size = (fs * 20 / 1000) as usize; // 960 samples / 20 ms
    let extra_delay: i32 = 312; // ~6.5 ms lookahead @ 48 kHz
    let total_frames = wav.samples.len() / (frame_size * channels as usize);
    // Convert the whole WAV to float once.
    let pcm_f: Vec<f32> = wav.samples.iter().map(|&s| pcm_i16_to_f32(s)).collect();

    // `activity_mem` is a `4 * DRED_MAX_FRAMES` = 416-byte ring buffer
    // the enclosing `OpusEncoder` maintains via its VAD. For the harness
    // we force "always active" (all 1s) so the encode-silk-frame logic
    // doesn't short-circuit on silence. Identical mem on both sides
    // keeps the test deterministic.
    let mut activity_mem_c = vec![1u8; 4 * DRED_MAX_FRAMES];
    let activity_mem_r = vec![1u8; 4 * DRED_MAX_FRAMES];

    // Buffer sizing — the outer Opus encoder passes DRED_MAX_DATA_SIZE as
    // an upper bound to `dred_encode_silk_frame`. We do the same.
    let max_bytes = DRED_MAX_DATA_SIZE;
    let max_chunks: i32 = 32; // Matches typical dred_chunks computed in opus_encoder.c.
    let q0: i32 = 6; // DRED_ENC_Q0 default from dred_config.h.
    let d_q: i32 = 1; // Smallest nonzero dQ — a sensible default.
    let qmax: i32 = 15; // DRED_ENC_Q1 default.

    // Accumulate latents over enough frames for the encoder to have
    // meaningful payload to emit. The C code emits DRED once per Opus
    // packet so we call both sides in lock-step, compare internal state
    // after each compute, and compare payload bytes after the last
    // encode_silk_frame.
    let mut c_payloads: Vec<Vec<u8>> = Vec::new();
    let mut r_payloads: Vec<Vec<u8>> = Vec::new();

    for fi in 0..total_frames {
        let sample_start = fi * frame_size * channels as usize;
        let sample_end = sample_start + frame_size * channels as usize;
        let frame_pcm = &pcm_f[sample_start..sample_end];

        // Both sides ingest the same frame.
        unsafe {
            ropus_test_dred_compute_latents(
                c_enc,
                frame_pcm.as_ptr(),
                frame_size as i32,
                extra_delay,
            );
        }
        r_enc.compute_latents(frame_pcm, frame_size as i32, extra_delay);

        // Compare resample_mem + input_buffer (upstream of features /
        // RDOVAE / entropy). If the resampler diverges, everything below
        // is wasted effort.
        let mut c_resample = [0.0f32; 9];
        unsafe {
            ropus_test_dredenc_copy_resample_mem(
                c_enc as *const c_void,
                c_resample.as_mut_ptr(),
                9,
            );
        }
        if let Some((idx, cv, rv)) = first_f32_divergent(&c_resample, &r_enc.resample_mem) {
            panic!(
                "frame {fi}: resample_mem drift at index {idx}: C={cv} Rust={rv} — \
                 filter_df2t implementation differs"
            );
        }
        let buf_fill_after = {
            let n = 2 * ropus::dnn::dred::DRED_FRAME_SIZE;
            let mut c_in = vec![0.0f32; n];
            unsafe {
                ropus_test_dredenc_copy_input_buffer(
                    c_enc as *const c_void,
                    c_in.as_mut_ptr(),
                    n as i32,
                );
            }
            if let Some((idx, cv, rv)) = first_f32_divergent(&c_in, &r_enc.input_buffer[..n]) {
                panic!(
                    "frame {fi}: input_buffer drift at index {idx}: C={cv} Rust={rv} — \
                     resampler / quantisation differs"
                );
            }
            n
        };
        let _ = buf_fill_after;

        // Compare state before encoding (early-warning diagnostics).
        let c_fill = unsafe { ropus_test_dredenc_latents_buffer_fill(c_enc as *const c_void) };
        let c_off = unsafe { ropus_test_dredenc_dred_offset(c_enc as *const c_void) };
        let c_latoff = unsafe { ropus_test_dredenc_latent_offset(c_enc as *const c_void) };
        assert_eq!(
            c_fill, r_enc.latents_buffer_fill,
            "frame {fi}: latents_buffer_fill mismatch C={} Rust={}",
            c_fill, r_enc.latents_buffer_fill
        );
        assert_eq!(
            c_off, r_enc.dred_offset,
            "frame {fi}: dred_offset mismatch C={} Rust={}",
            c_off, r_enc.dred_offset
        );
        assert_eq!(
            c_latoff, r_enc.latent_offset,
            "frame {fi}: latent_offset mismatch C={} Rust={}",
            c_latoff, r_enc.latent_offset
        );
        // Spot-check the LPCNet features once we have any — this catches
        // upstream drift from the feature extractor independently of the
        // RDOVAE forward pass.
        if c_fill > 0 {
            let mut c_feat = vec![0.0f32; 36];
            unsafe {
                ropus_test_dredenc_copy_lpcnet_features(
                    c_enc as *const c_void,
                    c_feat.as_mut_ptr(),
                    36,
                );
            }
            let r_feat = &r_enc.lpcnet_enc_state.features[..36];
            if let Some((idx, cv, rv)) = first_f32_divergent(&c_feat, r_feat) {
                panic!(
                    "frame {fi}: LPCNet features drift at index {idx}: C={cv} Rust={rv} — \
                     pre-existing Stage 7 LPCNet feature-extractor issue (not DRED)"
                );
            }
        }

        // Spot-check the first 25 latents once we have any.
        if c_fill > 0 {
            let mut c_lat = vec![0.0f32; 25];
            unsafe {
                ropus_test_dredenc_copy_latents(c_enc as *const c_void, c_lat.as_mut_ptr(), 25);
            }
            let r_lat = &r_enc.latents_buffer[..25];
            if let Some((idx, cv, rv)) = first_f32_divergent(&c_lat, r_lat) {
                panic!(
                    "frame {fi}: latents drift at index {idx}: C={cv} Rust={rv} — \
                     upstream of entropy coding (resampler/features/RDOVAE)"
                );
            }
            // Also spot-check state bank.
            let mut c_state = vec![0.0f32; 50];
            unsafe {
                ropus_test_dredenc_copy_state(c_enc as *const c_void, c_state.as_mut_ptr(), 50);
            }
            let r_state = &r_enc.state_buffer[..50];
            if let Some((idx, cv, rv)) = first_f32_divergent(&c_state, r_state) {
                panic!(
                    "frame {fi}: state drift at index {idx}: C={cv} Rust={rv} — \
                     upstream of entropy coding (resampler/features/RDOVAE)"
                );
            }
        }

        // Encode (only once we have enough latents for a useful
        // payload — matches the outer-encoder pattern where DRED isn't
        // emitted until buffers fill up). Call it every frame past the
        // first few so we exercise the encoder repeatedly.
        if c_fill >= 4 {
            let mut c_buf = vec![0u8; max_bytes];
            let mut r_buf = vec![0u8; max_bytes];
            let c_len = unsafe {
                ropus_test_dred_encode_silk_frame(
                    c_enc,
                    c_buf.as_mut_ptr(),
                    max_chunks,
                    max_bytes as i32,
                    q0,
                    d_q,
                    qmax,
                    activity_mem_c.as_mut_ptr(),
                )
            };
            let r_len = r_enc.encode_silk_frame(
                &mut r_buf,
                max_chunks,
                max_bytes,
                q0,
                d_q,
                qmax,
                &activity_mem_r,
            );
            assert_eq!(
                c_len, r_len,
                "frame {fi}: payload length mismatch C={} Rust={}",
                c_len, r_len
            );
            if c_len > 0 {
                c_buf.truncate(c_len as usize);
                r_buf.truncate(r_len as usize);
                if let Some((i, cv, rv)) = first_byte_divergent(&c_buf, &r_buf) {
                    panic!(
                        "frame {fi}: payload byte {} differs: C=0x{:02x} Rust=0x{:02x} \
                         (total len {}). C payload hex = {}, Rust payload hex = {}",
                        i,
                        cv,
                        rv,
                        c_len,
                        to_hex(&c_buf),
                        to_hex(&r_buf),
                    );
                }
                c_payloads.push(c_buf);
                r_payloads.push(r_buf);
            }
        }
    }

    unsafe {
        ropus_test_dredenc_free(c_enc);
    }

    assert!(
        !c_payloads.is_empty(),
        "no DRED payloads were emitted — test exercised nothing"
    );
    eprintln!(
        "Tier 1 achieved: {} byte-exact DRED payload emission(s) matched C reference.",
        c_payloads.len()
    );
}

/// Lowercase hex-encode a byte slice for panic diagnostics.
fn to_hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for &x in b {
        s.push_str(&format!("{:02x}", x));
    }
    s
}
