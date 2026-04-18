//! Integration tests that exercise the Rust codec against the C reference.
//!
//! These tests were previously inside the `ropus` library's `#[cfg(test)]`
//! modules, but the library itself has no build script and so cannot link
//! `opus_ref`. The harness crate links `opus_ref` via `build.rs`, so these
//! FFI-dependent differential tests live here instead.
//!
//! Tests migrated from:
//!   - `ropus/src/opus/decoder.rs` (test_lpc_inverse_pred_gain_matches_c_reference,
//!     test_garbage_hybrid_swb_decode_matches_c_reference,
//!     test_redundancy_differential_sequential_decode)
//!   - `ropus/src/silk/encoder.rs` (test_bug5_encode_8khz_stereo_71kbps_matches_c_reference)

#![allow(clippy::too_many_arguments)]

use ropus::opus::decoder::OpusDecoder;
use ropus::opus::encoder::{OPUS_APPLICATION_VOIP, OpusEncoder};
use ropus::silk::common::silk_lpc_inverse_pred_gain;

use std::os::raw::c_int;

// Minimal FFI declarations for the C reference (linked via harness/build.rs as
// the `opus_ref` static library).
unsafe extern "C" {
    fn opus_decoder_create(
        fs: i32,
        channels: c_int,
        error: *mut c_int,
    ) -> *mut std::ffi::c_void;
    fn opus_decode(
        st: *mut std::ffi::c_void,
        data: *const u8,
        len: i32,
        pcm: *mut i16,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
    fn opus_decoder_destroy(st: *mut std::ffi::c_void);
    fn silk_LPC_inverse_pred_gain_c(a_q12: *const i16, order: c_int) -> i32;

    fn opus_encoder_create(
        fs: i32,
        channels: c_int,
        application: c_int,
        error: *mut c_int,
    ) -> *mut u8;
    fn opus_encode(
        st: *mut u8,
        pcm: *const i16,
        frame_size: c_int,
        data: *mut u8,
        max_data_bytes: i32,
    ) -> i32;
    fn opus_encoder_destroy(st: *mut u8);
    fn opus_encoder_ctl(st: *mut u8, request: c_int, ...) -> c_int;
}

const OPUS_SET_BITRATE_REQUEST: c_int = 4002;
const OPUS_SET_VBR_REQUEST: c_int = 4006;
const OPUS_SET_COMPLEXITY_REQUEST: c_int = 4010;

/// Generate a patterned PCM buffer that is deterministic across runs.
/// Copied verbatim from the original test helper in ropus/src/opus/decoder.rs.
fn patterned_pcm_i16(frame_size: usize, channels: usize, seed: i32) -> Vec<i16> {
    (0..frame_size * channels)
        .map(|i| {
            let base = ((i as i32 * 7919 + seed * 911) % 28000) - 14000;
            if channels == 2 && i % 2 == 1 {
                (base / 2) as i16
            } else {
                base as i16
            }
        })
        .collect()
}

/// Decode a packet with the C reference, returning PCM or the C error code.
fn c_ref_decode(packet: &[u8], sr: i32, ch: i32) -> Result<Vec<i16>, i32> {
    unsafe {
        let mut err: c_int = 0;
        let dec = opus_decoder_create(sr, ch, &mut err);
        if dec.is_null() || err != 0 {
            if !dec.is_null() {
                opus_decoder_destroy(dec);
            }
            return Err(err);
        }
        let max_frame = 5760;
        let mut pcm = vec![0i16; max_frame as usize * ch as usize];
        let ret = opus_decode(
            dec,
            packet.as_ptr(),
            packet.len() as i32,
            pcm.as_mut_ptr(),
            max_frame,
            0,
        );
        opus_decoder_destroy(dec);
        if ret < 0 {
            Err(ret)
        } else {
            pcm.truncate(ret as usize * ch as usize);
            Ok(pcm)
        }
    }
}

// -----------------------------------------------------------------------------
// silk_LPC_inverse_pred_gain differential
// -----------------------------------------------------------------------------

#[test]
fn test_lpc_inverse_pred_gain_matches_c_reference() {
    // Test with many different coefficient patterns to find divergences
    let mut mismatches = 0u32;
    let patterns: &[(&str, Vec<i16>)] = &[
        // Patterns that exercise extreme values
        ("all_max", vec![i16::MAX; 16]),
        ("all_min", vec![i16::MIN; 16]),
        (
            "alternating",
            (0..16)
                .map(|i| if i % 2 == 0 { 4000i16 } else { -4000 })
                .collect(),
        ),
        ("near_unity_sum", {
            // Sum close to 4096 (DC stability boundary)
            let mut v = vec![256i16; 16];
            v[0] = 4095 - 15 * 256;
            v
        }),
        ("garbage_ff", vec![0x0FFFu16 as i16; 16]),
        ("extreme_mixed", {
            let mut v = vec![0i16; 16];
            for i in 0..16 {
                v[i] = ((i as i32 * 8191 + 3) % 8191 - 4095) as i16;
            }
            v
        }),
    ];
    for (name, coeffs) in patterns {
        let order = coeffs.len().min(16);
        let rust_result = silk_lpc_inverse_pred_gain(&coeffs[..order], order);
        let c_result =
            unsafe { silk_LPC_inverse_pred_gain_c(coeffs.as_ptr(), order as c_int) };
        if rust_result != c_result {
            eprintln!(
                "MISMATCH {name}: rust={rust_result}, c={c_result}, \
                 coeffs={:?}",
                &coeffs[..order]
            );
            mismatches += 1;
        }
    }

    // Also brute-force random-looking patterns
    for seed in 0u32..1000 {
        let mut coeffs = [0i16; 16];
        for i in 0..16 {
            let v = ((seed.wrapping_mul(2654435761).wrapping_add(i as u32 * 7919)) >> 16) as i16;
            coeffs[i] = v;
        }
        let rust_result = silk_lpc_inverse_pred_gain(&coeffs, 16);
        let c_result = unsafe { silk_LPC_inverse_pred_gain_c(coeffs.as_ptr(), 16) };
        if rust_result != c_result {
            eprintln!(
                "MISMATCH seed={seed}: rust={rust_result}, c={c_result}, \
                 coeffs={coeffs:?}"
            );
            mismatches += 1;
            if mismatches >= 5 {
                break;
            }
        }
    }

    assert_eq!(
        mismatches, 0,
        "mismatches found between Rust and C silk_lpc_inverse_pred_gain"
    );
}

// -----------------------------------------------------------------------------
// Garbage Hybrid SWB packet decode differential
// -----------------------------------------------------------------------------

#[test]
fn test_garbage_hybrid_swb_decode_matches_c_reference() {
    // TOC 0x60 = config 12 = Hybrid SWB, 10ms, mono, code 0 (1 frame)
    // Use the 0xFF pattern which triggers the mismatch found by fuzz testing.
    // The bug was in silk_decode_pulses: missing ICDF table offset after 10
    // LSB shifts, causing different bit consumption from the range coder.
    let toc = 0x60u8;
    let sr = 48000;
    let ch = 1;
    let max_frame = 5760;
    let payload = vec![0xFFu8; 255];

    let mut packet = vec![toc];
    packet.extend_from_slice(&payload);

    // Rust decode
    let mut rust_dec = OpusDecoder::new(sr, ch).unwrap();
    let mut rust_pcm = vec![0i16; max_frame as usize * ch as usize];
    let rust_ret = rust_dec.decode(Some(&packet), &mut rust_pcm, max_frame, false);

    // C reference decode
    let c_ret = c_ref_decode(&packet, sr, ch);

    let rust_samples = rust_ret.unwrap() as usize;
    let c_pcm = c_ret.unwrap();
    let c_samples = c_pcm.len() / ch as usize;
    assert_eq!(rust_samples, c_samples, "Sample count mismatch");

    let rust_slice = &rust_pcm[..rust_samples * ch as usize];
    assert_eq!(
        rust_slice,
        &c_pcm[..],
        "Garbage hybrid SWB (0xFF payload) PCM mismatch: Rust and C differ"
    );
}

// -----------------------------------------------------------------------------
// Sequential VOIP decode differential (redundancy / mode transitions)
// -----------------------------------------------------------------------------

#[test]
fn test_redundancy_differential_sequential_decode() {
    // Differential test: decode a sequence of VOIP packets with both
    // the Rust decoder and the C reference, comparing output at each step.
    let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
    enc.set_bitrate(24000);

    let mut packets = Vec::new();
    for seed in 0..6 {
        let pcm = patterned_pcm_i16(960, 1, seed * 1337);
        let mut buf = vec![0u8; 1500];
        let cap = buf.len() as i32;
        let len = enc.encode(&pcm, 960, &mut buf, cap).unwrap();
        packets.push(buf[..len as usize].to_vec());
    }

    // Decode sequentially with both Rust and C, comparing each frame
    let mut rust_dec = OpusDecoder::new(48000, 1).unwrap();
    let sr = 48000;
    let ch = 1;
    let max_frame = 5760;

    unsafe {
        let mut err: c_int = 0;
        let c_dec = opus_decoder_create(sr, ch, &mut err);
        assert!(!c_dec.is_null() && err == 0);

        for (i, pkt) in packets.iter().enumerate() {
            let mut rust_pcm = vec![0i16; max_frame as usize];
            let rust_ret = rust_dec
                .decode(Some(pkt), &mut rust_pcm, max_frame, false)
                .unwrap();

            let mut c_pcm = vec![0i16; max_frame as usize];
            let c_ret = opus_decode(
                c_dec,
                pkt.as_ptr(),
                pkt.len() as i32,
                c_pcm.as_mut_ptr(),
                max_frame,
                0,
            );
            assert!(c_ret > 0, "C decode failed at frame {i}: {c_ret}");
            assert_eq!(rust_ret, c_ret, "frame {i}: sample count mismatch");
            assert_eq!(
                &rust_pcm[..rust_ret as usize],
                &c_pcm[..c_ret as usize],
                "frame {i}: PCM mismatch"
            );
        }

        opus_decoder_destroy(c_dec);
    }
}

// -----------------------------------------------------------------------------
// Bug #5 regression: 8kHz stereo @ 71kbps encode matches C reference
// -----------------------------------------------------------------------------

/// Regression test for Bug #5: encode mismatch at 8kHz/stereo/71kbps.
///
/// `silk_inner_prod_aligned_scale` used i64 accumulation, but the C reference
/// uses i32 accumulation that wraps on overflow. This produces different
/// stereo predictor values, cascading into different rate allocation and
/// trailing zeros in the Rust output.
#[test]
fn test_bug5_encode_8khz_stereo_71kbps_matches_c_reference() {
    let sample_rate: i32 = 8000;
    let channels: i32 = 2;
    let application: i32 = 2048; // OPUS_APPLICATION_VOIP
    let bitrate: i32 = 71535;
    let complexity: i32 = 2;
    let frame_size: i32 = sample_rate / 50; // 20ms = 160 samples

    // Deterministic stereo PCM using a simple LCG for reproducibility.
    // The fuzz campaign found the mismatch with pseudo-random data that
    // exercises the stereo predictor path where i32 accumulation wrapping
    // in silk_inner_prod_aligned_scale diverges from i64.
    let total_samples = frame_size as usize * channels as usize; // 320
    let mut pcm = vec![0i16; total_samples];
    let mut rng: u64 = 0xDEAD_BEEF_CAFE_BABE;
    for s in pcm.iter_mut() {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *s = (rng >> 33) as i16;
    }

    // --- Create both encoders with identical config ---
    let c_enc = unsafe {
        let mut error: c_int = 0;
        let enc = opus_encoder_create(sample_rate, channels, application, &mut error);
        assert!(
            !enc.is_null() && error == 0,
            "C encoder create failed: {error}"
        );
        opus_encoder_ctl(enc, OPUS_SET_BITRATE_REQUEST, bitrate);
        opus_encoder_ctl(enc, OPUS_SET_VBR_REQUEST, 0 as c_int);
        opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY_REQUEST, complexity);
        enc
    };

    let mut rust_enc = OpusEncoder::new(sample_rate, channels, application)
        .expect("Rust encoder create failed");
    rust_enc.set_bitrate(bitrate);
    rust_enc.set_vbr(0);
    rust_enc.set_complexity(complexity);

    // Encode several frames to build up encoder state (stereo predictor
    // smoothing, HP filter state, etc.) — the overflow may only manifest
    // after the encoder has warmed up.
    let num_frames = 10;
    for frame_idx in 0..num_frames {
        // Generate per-frame PCM from the running RNG
        for s in pcm.iter_mut() {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *s = (rng >> 33) as i16;
        }

        let c_out = unsafe {
            let mut out = vec![0u8; 4000];
            let ret = opus_encode(c_enc, pcm.as_ptr(), frame_size, out.as_mut_ptr(), 4000);
            assert!(ret > 0, "C encode failed on frame {frame_idx}: {ret}");
            out.truncate(ret as usize);
            out
        };

        let mut rust_out = vec![0u8; 4000];
        let rust_len = rust_enc
            .encode(&pcm, frame_size, &mut rust_out, 4000)
            .unwrap_or_else(|e| panic!("Rust encode failed on frame {frame_idx}: {e}"))
            as usize;

        assert_eq!(
            rust_len,
            c_out.len(),
            "Frame {frame_idx}: output length mismatch: Rust={rust_len}, C={}",
            c_out.len()
        );
        assert_eq!(
            &rust_out[..rust_len],
            &c_out[..],
            "Frame {frame_idx}: byte mismatch (bug #5: silk_inner_prod_aligned_scale overflow)"
        );
    }

    unsafe { opus_encoder_destroy(c_enc) };
}

// -----------------------------------------------------------------------------
// Targeted diagnostic: SILK subframe-boundary divergence across bandwidths / SRs
// -----------------------------------------------------------------------------

/// Targeted test: narrow down whether divergence is in SILK WB or the resampler.
/// Test WB 20ms at both 8kHz (needs resampler) and 16kHz (native WB, no resampler).
#[test]
fn test_silk_subframe_boundary_divergence() {
    let max_frame = 5760;
    let payload = vec![0xFFu8; 50];

    // Test WB 20ms mono (config 9, toc=0x48) at different output sample rates
    let configs: &[(&str, u8, i32, i32)] = &[
        // (label, toc, sr, ch)
        ("NB 20ms @ 8kHz", 0x08, 8000, 1),
        ("WB 20ms @ 8kHz", 0x48, 8000, 1),   // resamples 16k→8k
        ("WB 20ms @ 16kHz", 0x48, 16000, 1), // native 16k, no resampler
        ("WB 20ms @ 24kHz", 0x48, 24000, 1), // resamples 16k→24k
        ("WB 20ms @ 48kHz", 0x48, 48000, 1), // resamples 16k→48k
        ("MB 20ms @ 8kHz", 0x28, 8000, 1),   // resamples 12k→8k
        ("MB 20ms @ 12kHz", 0x28, 12000, 1), // native 12k
        ("Hyb SWB 20ms @ 48kHz", 0x68, 48000, 1), // hybrid, native
        ("NB 20ms stereo @ 8kHz", 0x0C, 8000, 1), // stereo TOC on mono dec
        ("NB 20ms stereo @ 8kHz ch2", 0x0C, 8000, 2), // actual stereo
    ];

    for &(label, toc, sr, ch) in configs {
        let mut packet = vec![toc];
        packet.extend_from_slice(&payload);

        let mut rust_dec = OpusDecoder::new(sr, ch).unwrap();
        let mut rust_pcm = vec![0i16; max_frame as usize * ch as usize];
        let rust_ret = rust_dec.decode(Some(&packet), &mut rust_pcm, max_frame, false);

        let c_ret = c_ref_decode(&packet, sr, ch);

        match (&rust_ret, &c_ret) {
            (Ok(rn), Ok(cp)) => {
                let n = *rn as usize;
                let rust_slice = &rust_pcm[..n];
                if rust_slice != &cp[..] {
                    let first = rust_slice
                        .iter()
                        .zip(cp.iter())
                        .position(|(a, b)| a != b)
                        .unwrap_or(0);
                    let max_d = rust_slice
                        .iter()
                        .zip(cp.iter())
                        .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs())
                        .max()
                        .unwrap_or(0);
                    eprintln!(
                        "  {label}: DIVERGES — samples={n}, first_diff={first}, max_diff={max_d}"
                    );
                    // Show first 5 mismatching pairs
                    for (i, (r, c)) in rust_slice
                        .iter()
                        .zip(cp.iter())
                        .enumerate()
                        .filter(|(_, (r, c))| r != c)
                        .take(5)
                    {
                        eprintln!("    [{i}]: Rust={r}, C={c}, diff={}", *r as i32 - *c as i32);
                    }
                } else {
                    eprintln!("  {label}: MATCHES — samples={n}");
                }
            }
            _ => {
                eprintln!("  {label}: error result — skipping");
            }
        }
    }
}

/// Broad scan: decode garbage packets at every TOC config × sample rate × channel count.
/// Reports which configurations diverge between Rust and C.
/// Ignored by default: known SILK garbage-input divergences (Bug #8).
/// Run with `cargo test -- --ignored test_fuzz_decode_scan_all_configs` to diagnose.
#[test]
#[ignore]
fn test_fuzz_decode_scan_all_configs() {
    let sample_rates = [8000, 12000, 16000, 24000, 48000];
    let channels_opts = [1, 2];
    // Test a few payload patterns that are likely to exercise edge cases
    let payloads: &[(&str, Vec<u8>)] = &[
        ("0xFF×50", vec![0xFF; 50]),
        ("0x00×50", vec![0x00; 50]),
        ("ramp", (0u8..50).collect()),
        ("0x80×50", vec![0x80; 50]),
    ];
    let mut mismatches = Vec::new();

    for &sr in &sample_rates {
        for &ch in &channels_opts {
            for toc in 0u8..=255 {
                // Only test every 4th TOC (the frame code bits don't affect
                // the mode/bandwidth, just framing)
                if toc & 0x03 != 0 {
                    continue;
                }
                for (pat_name, payload) in payloads {
                    let mut packet = vec![toc];
                    packet.extend_from_slice(payload);

                    // Rust decode
                    let mut rust_dec = match OpusDecoder::new(sr, ch) {
                        Ok(d) => d,
                        Err(_) => continue,
                    };
                    let max_frame = 5760;
                    let mut rust_pcm = vec![0i16; max_frame as usize * ch as usize];
                    let rust_ret =
                        rust_dec.decode(Some(&packet), &mut rust_pcm, max_frame, false);

                    // C decode
                    let c_ret = c_ref_decode(&packet, sr, ch);

                    match (&rust_ret, &c_ret) {
                        (Ok(rust_n), Ok(c_pcm)) => {
                            let n = *rust_n as usize * ch as usize;
                            let c_n = c_pcm.len();
                            if n != c_n {
                                mismatches.push(format!(
                                    "sr={sr} ch={ch} toc=0x{toc:02X} pat={pat_name}: \
                                     sample count Rust={} C={}",
                                    rust_n,
                                    c_n / ch as usize
                                ));
                                continue;
                            }
                            let rust_slice = &rust_pcm[..n];
                            if rust_slice != &c_pcm[..] {
                                // Find first mismatch index
                                let first_diff = rust_slice
                                    .iter()
                                    .zip(c_pcm.iter())
                                    .position(|(a, b)| a != b)
                                    .unwrap_or(0);
                                let max_diff = rust_slice
                                    .iter()
                                    .zip(c_pcm.iter())
                                    .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs())
                                    .max()
                                    .unwrap_or(0);
                                mismatches.push(format!(
                                    "sr={sr} ch={ch} toc=0x{toc:02X} pat={pat_name}: \
                                     PCM mismatch first_diff_idx={first_diff} max_diff={max_diff} \
                                     samples={}",
                                    rust_n
                                ));
                            }
                        }
                        (Err(_), Err(_)) => {} // both error — fine
                        (Ok(n), Err(e)) => {
                            mismatches.push(format!(
                                "sr={sr} ch={ch} toc=0x{toc:02X} pat={pat_name}: \
                                 Rust OK({n}) but C err({e})"
                            ));
                        }
                        (Err(e), Ok(c_pcm)) => {
                            mismatches.push(format!(
                                "sr={sr} ch={ch} toc=0x{toc:02X} pat={pat_name}: \
                                 Rust err({e}) but C OK({})",
                                c_pcm.len() / ch as usize
                            ));
                        }
                    }
                }
            }
        }
    }

    if !mismatches.is_empty() {
        eprintln!("\n=== DECODE DIVERGENCES FOUND: {} ===", mismatches.len());
        for m in &mismatches {
            eprintln!("  {m}");
        }
        panic!(
            "{} decode divergence(s) found — see stderr for details",
            mismatches.len()
        );
    }
}
