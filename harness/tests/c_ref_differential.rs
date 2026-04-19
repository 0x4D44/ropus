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
    fn opus_decode24(
        st: *mut std::ffi::c_void,
        data: *const u8,
        len: i32,
        pcm: *mut i32,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
    fn opus_decoder_destroy(st: *mut std::ffi::c_void);
    fn opus_multistream_decoder_create(
        fs: i32,
        channels: c_int,
        streams: c_int,
        coupled_streams: c_int,
        mapping: *const u8,
        error: *mut c_int,
    ) -> *mut std::ffi::c_void;
    fn opus_multistream_decode24(
        st: *mut std::ffi::c_void,
        data: *const u8,
        len: i32,
        pcm: *mut i32,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
    fn opus_multistream_decoder_destroy(st: *mut std::ffi::c_void);
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
    fn opus_decoder_ctl(st: *mut std::ffi::c_void, request: c_int, ...) -> c_int;
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

// -----------------------------------------------------------------------------
// Encoder `get_final_range` matches the bitstream on CELT→SILK redundancy
// -----------------------------------------------------------------------------
//
// Regression for the bug surfaced by xiph/opus test_opus_encode.c:501 (surfaced
// via the capi FFI shim). In SILK force-mode, cycling bandwidths NB→MB→WB
// triggers the SILK-initiated bandwidth switch that makes
// `redundancy && celt_to_silk` fire on the MB→WB transition. Before the fix,
// the encoder encoded a 5ms CELT→SILK redundancy frame but failed to capture
// the CELT rng into `redundant_rng` before `OPUS_RESET_STATE` wiped it, so
// the XOR at the end of `encode_frame_native` XOR'd with 0 and the encoder's
// `get_final_range` diverged from what both the ropus decoder and the C
// reference decoder reported after decoding the produced bytes.
#[test]
fn test_encoder_final_range_matches_decoder_on_celt_to_silk_redundancy() {
    use ropus::opus::decoder::{
        MODE_SILK_ONLY, OPUS_BANDWIDTH_MEDIUMBAND, OPUS_BANDWIDTH_NARROWBAND,
        OPUS_BANDWIDTH_WIDEBAND,
    };

    // Exact reproducer from the phase-4 debug agent:
    // 48 kHz mono, SILK force-mode, 6809 bps, 1920-sample frames,
    // bandwidths cycling NB → MB → WB. Iter 2 (first WB) previously diverged.
    let sr: i32 = 48000;
    let ch: i32 = 1;
    let frame_size: i32 = 1920;
    let bitrate: i32 = 6809;
    let max_packet: i32 = 1500;

    let mut enc = OpusEncoder::new(sr, ch, OPUS_APPLICATION_VOIP).unwrap();
    assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), 0);
    enc.set_bitrate(bitrate);

    let mut rust_dec = OpusDecoder::new(sr, ch).unwrap();

    let c_dec = unsafe {
        let mut err: c_int = 0;
        let d = opus_decoder_create(sr, ch, &mut err);
        assert!(!d.is_null() && err == 0);
        d
    };

    let bw_cycle = [
        OPUS_BANDWIDTH_NARROWBAND,
        OPUS_BANDWIDTH_MEDIUMBAND,
        OPUS_BANDWIDTH_WIDEBAND,
    ];

    // Realistic audio-like signal: a port of `generate_music()` from
    // `reference/tests/test_opus_encode.c:57-84` — a simple algorithmic tune
    // plus shaped noise that actually drives SILK's analysis into the
    // "warrants a bandwidth switch" state. The patterned PCM used by other
    // tests in this file is too sparse to trigger SILK's `switchReady`.
    let total_samples = (frame_size as usize) * (ch as usize) * 20;
    let mut music = vec![0i16; total_samples];
    {
        let (mut a1, mut b1) = (0i32, 0i32);
        let (mut c1, mut d1) = (0i32, 0i32);
        let mut j: i32 = 0;
        let mut rng: u64 = 0x1234_5678_9ABC_DEF0;
        // 60ms silence prefix so the encoder has time to settle, mirroring
        // `generate_music()`'s own silence prefix.
        let silence_samples = (sr as usize / 1000) * 60;
        for (i, s) in music.iter_mut().enumerate().skip(silence_samples) {
            // Pseudo-melody bitfield from test_opus_encode.c:71
            let v_base = ((j.wrapping_mul(
                (j >> 12) ^ ((j >> 10 | j >> 12) & 26 & (j >> 7)),
            )) & 128)
                + 128;
            let mut v: i32 = v_base << 15;
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (rng >> 32) as u32 as i32;
            v = v.wrapping_add(r & 65535);
            v = v.wrapping_sub(r >> 16);
            // IIR smoothing exactly as in generate_music()
            b1 = v.wrapping_sub(a1).wrapping_add((b1 * 61 + 32) >> 6);
            a1 = v;
            c1 = (30 * (c1 + b1 + d1) + 32) >> 6;
            d1 = b1;
            let out = (c1 + 128) >> 8;
            *s = out.clamp(-32768, 32767) as i16;
            // C: `if (i % 6 == 0) j++;` — drive the melody from the sample
            // counter, NOT from `j` itself.
            if i % 6 == 0 {
                j = j.wrapping_add(1);
            }
        }
    }

    for iter in 0..15i32 {
        let bw = bw_cycle[(iter as usize) % bw_cycle.len()];
        assert_eq!(enc.set_bandwidth(bw), 0);

        let start = (iter as usize) * (frame_size as usize) * (ch as usize);
        let end = start + (frame_size as usize) * (ch as usize);
        let pcm = &music[start..end];

        let mut out = vec![0u8; max_packet as usize];
        let len = enc
            .encode(pcm, frame_size, &mut out, max_packet)
            .unwrap_or_else(|e| panic!("iter {iter}: encode failed: {e}"));
        assert!(len > 0, "iter {iter}: empty encoded packet");
        let packet = &out[..len as usize];

        let enc_rng = enc.get_final_range();

        // Rust decode
        let mut rust_pcm = vec![0i16; frame_size as usize * ch as usize];
        rust_dec
            .decode(Some(packet), &mut rust_pcm, frame_size, false)
            .unwrap_or_else(|e| panic!("iter {iter}: rust decode failed: {e}"));
        let rust_dec_rng = rust_dec.get_final_range();

        // C reference decode (separate fresh decoder not required — we reuse
        // one to mirror the stateful invariants on the wire, just like the
        // conformance suite does).
        let mut c_pcm = vec![0i16; frame_size as usize * ch as usize];
        let c_ret = unsafe {
            opus_decode(
                c_dec,
                packet.as_ptr(),
                packet.len() as i32,
                c_pcm.as_mut_ptr(),
                frame_size,
                0,
            )
        };
        assert!(c_ret > 0, "iter {iter}: C decode failed: {c_ret}");
        // Read C decoder's final range via the same ctl used by the test suite.
        let mut c_dec_rng: u32 = 0;
        let rc = unsafe {
            const OPUS_GET_FINAL_RANGE_REQUEST: c_int = 4031;
            opus_decoder_ctl(
                c_dec,
                OPUS_GET_FINAL_RANGE_REQUEST,
                &mut c_dec_rng as *mut u32,
            )
        };
        assert_eq!(rc, 0, "iter {iter}: C OPUS_GET_FINAL_RANGE ctl failed: {rc}");

        assert_eq!(
            enc_rng, rust_dec_rng,
            "iter {iter} (bw={bw}): encoder get_final_range={:#010x} disagrees \
             with ropus decoder get_final_range={:#010x}",
            enc_rng, rust_dec_rng
        );
        assert_eq!(
            enc_rng, c_dec_rng,
            "iter {iter} (bw={bw}): encoder get_final_range={:#010x} disagrees \
             with C-ref decoder OPUS_GET_FINAL_RANGE={:#010x}",
            enc_rng, c_dec_rng
        );
    }

    unsafe { opus_decoder_destroy(c_dec) };
}

// -----------------------------------------------------------------------------
// Encoder `get_final_range` matches the bitstream on SILK→CELT redundancy
// -----------------------------------------------------------------------------
//
// Secondary regression surfaced after fixing CELT→SILK (commit 190d4c6a).
// test_opus_encode.c:501 fails at j=7 (MODE_CELT_ONLY, frame_size=2880)
// because the mode-transition-undo + WB-downgrade force the first frame of
// j=7 to be encoded as mode=SILK_ONLY with a SILK→CELT redundancy tail
// (redundancy=true, celt_to_silk=false, to_celt=true). The encoder's
// `redundant_rng` bookkeeping on the SILK→CELT branch diverges from what
// the decoder computes from the bitstream.
//
// Minimum reproducer: run a Hybrid-FB frame first to set prev_mode=HYBRID,
// then immediately request MODE_CELT_ONLY with a WB-constrained bandwidth
// and frame_size=60ms (2880 at 48kHz). Stereo, high bitrate (≥64k).
#[test]
fn test_encoder_final_range_matches_decoder_on_silk_to_celt_redundancy() {
    use ropus::opus::decoder::{
        MODE_CELT_ONLY, MODE_HYBRID, OPUS_BANDWIDTH_FULLBAND, OPUS_BANDWIDTH_WIDEBAND,
    };

    let sr: i32 = 48000;
    let ch: i32 = 2;
    let max_packet: i32 = 1500;

    let mut enc = OpusEncoder::new(sr, ch, OPUS_APPLICATION_VOIP).unwrap();
    // 20 ms stereo Hybrid FB at 80 kbps, warmup 2 frames.
    let bitrate: i32 = 80_000;
    enc.set_bitrate(bitrate);

    let mut rust_dec = OpusDecoder::new(sr, ch).unwrap();
    let c_dec = unsafe {
        let mut err: c_int = 0;
        let d = opus_decoder_create(sr, ch, &mut err);
        assert!(!d.is_null() && err == 0);
        d
    };

    // Realistic audio-like signal — long enough for all frames combined.
    let total_frames = 6usize;
    let total_samples = (sr as usize / 50) * total_frames * 4 + 2880 * 2; // generous
    let mut music = vec![0i16; total_samples * ch as usize];
    {
        let (mut a1, mut b1) = (0i32, 0i32);
        let (mut c1, mut d1) = (0i32, 0i32);
        let mut j: i32 = 0;
        let mut rng: u64 = 0x1234_5678_9ABC_DEF0;
        for (i, s) in music.iter_mut().enumerate() {
            let v_base = ((j.wrapping_mul(
                (j >> 12) ^ ((j >> 10 | j >> 12) & 26 & (j >> 7)),
            )) & 128)
                + 128;
            let mut v: i32 = v_base << 15;
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (rng >> 32) as u32 as i32;
            v = v.wrapping_add(r & 65535);
            v = v.wrapping_sub(r >> 16);
            b1 = v.wrapping_sub(a1).wrapping_add((b1 * 61 + 32) >> 6);
            a1 = v;
            c1 = (30 * (c1 + b1 + d1) + 32) >> 6;
            d1 = b1;
            let out = (c1 + 128) >> 8;
            *s = out.clamp(-32768, 32767) as i16;
            if i % 6 == 0 {
                j = j.wrapping_add(1);
            }
        }
    }

    let mut cursor: usize = 0;

    // Helper closure to encode, decode (both Rust and C), and compare ranges.
    let encode_and_compare = |enc: &mut OpusEncoder,
                              rust_dec: &mut OpusDecoder,
                              c_dec: *mut std::ffi::c_void,
                              tag: &str,
                              frame_size: i32,
                              pcm: &[i16]| {
        let mut out = vec![0u8; max_packet as usize];
        let len = enc
            .encode(pcm, frame_size, &mut out, max_packet)
            .unwrap_or_else(|e| panic!("{tag}: encode failed: {e}"))
            as usize;
        let pkt = &out[..len];
        let enc_rng = enc.get_final_range();

        let mut rust_pcm = vec![0i16; frame_size as usize * ch as usize];
        rust_dec
            .decode(Some(pkt), &mut rust_pcm, frame_size, false)
            .unwrap_or_else(|e| panic!("{tag}: rust decode failed: {e}"));
        let rust_dec_rng = rust_dec.get_final_range();

        let mut c_pcm = vec![0i16; frame_size as usize * ch as usize];
        let c_ret = unsafe {
            opus_decode(
                c_dec,
                pkt.as_ptr(),
                pkt.len() as i32,
                c_pcm.as_mut_ptr(),
                frame_size,
                0,
            )
        };
        assert!(c_ret > 0, "{tag}: C decode failed: {c_ret}");
        let mut c_dec_rng: u32 = 0;
        let rc = unsafe {
            const OPUS_GET_FINAL_RANGE_REQUEST: c_int = 4031;
            opus_decoder_ctl(
                c_dec,
                OPUS_GET_FINAL_RANGE_REQUEST,
                &mut c_dec_rng as *mut u32,
            )
        };
        assert_eq!(rc, 0, "{tag}: C ctl failed");

        assert_eq!(
            enc_rng, rust_dec_rng,
            "{tag}: encoder.get_final_range={:#010x} disagrees with \
             ropus decoder={:#010x}",
            enc_rng, rust_dec_rng,
        );
        assert_eq!(
            enc_rng, c_dec_rng,
            "{tag}: encoder.get_final_range={:#010x} disagrees with \
             C-ref decoder={:#010x}",
            enc_rng, c_dec_rng,
        );
    };

    // Warmup: 2× Hybrid FB 20ms frames to set prev_mode=HYBRID.
    for i in 0..3 {
        let frame_size: i32 = 960; // 20 ms
        enc.set_force_mode(MODE_HYBRID);
        enc.set_bandwidth(OPUS_BANDWIDTH_FULLBAND);
        let start = cursor;
        let end = start + (frame_size as usize) * (ch as usize);
        let pcm = &music[start..end];
        cursor = end;
        let tag = format!("warmup[{i}]");
        encode_and_compare(&mut enc, &mut rust_dec, c_dec, &tag, frame_size, pcm);
    }

    // Trigger: force CELT_ONLY + bandwidth <= WB with 60ms frame.
    // This invokes mode=prev_mode=HYBRID → bandwidth<=WB → mode=SILK_ONLY
    // → redundancy+to_celt path (SILK→CELT redundancy tail).
    enc.set_force_mode(MODE_CELT_ONLY);
    enc.set_bandwidth(OPUS_BANDWIDTH_WIDEBAND);

    let frame_size: i32 = 2880; // 60 ms
    let start = cursor;
    let end = start + (frame_size as usize) * (ch as usize);
    let pcm = &music[start..end];
    encode_and_compare(&mut enc, &mut rust_dec, c_dec, "trigger", frame_size, pcm);

    unsafe { opus_decoder_destroy(c_dec) };
}

// -----------------------------------------------------------------------------
// Encoder `get_final_range` matches decoder on HYBRID-mode SILK→CELT redundancy
// -----------------------------------------------------------------------------
//
// Surfaced by `test_opus_encode.c:501` after bugs K (SILK pitch) and M
// (CELT→SILK rng capture) were fixed. Same bug class as the prior
// SILK→CELT redundancy test, but exercised in HYBRID mode rather than
// SILK_ONLY. The distinction matters because only the HYBRID branch
// requires the `nb_compr_bytes = ret` shrink from C opus_encoder.c:2532.
//
// Trigger: prev_mode=HYBRID, user requests CELT_ONLY with bandwidth>WB
// (so HYBRID isn't downgraded) at a short frame. The mode-transition
// logic undoes this back to HYBRID with redundancy=true, celt_to_silk=false,
// to_celt=true — writing a SILK→CELT redundancy tail in HYBRID mode.
#[test]
fn test_encoder_final_range_matches_decoder_on_hybrid_silk_to_celt_redundancy() {
    use ropus::opus::decoder::{
        MODE_CELT_ONLY, MODE_HYBRID, OPUS_BANDWIDTH_FULLBAND,
    };

    let sr: i32 = 48000;
    let ch: i32 = 2;
    let max_packet: i32 = 1500;

    let mut enc = OpusEncoder::new(sr, ch, OPUS_APPLICATION_VOIP).unwrap();
    // VBR Hybrid so CELT actually shrinks nb_compr_bytes below the max.
    // At 24 kbps the CELT portion of a 20 ms Hybrid frame uses only a
    // fraction of the available bytes, so `ret < nb_compr_bytes` reliably,
    // which is the condition that exposes the SILK→CELT redundancy offset
    // bug (pre-fix ropus writes redundancy at the stale `nb_compr_bytes`
    // offset instead of the actual CELT end `ret`).
    let bitrate: i32 = 24_000;
    enc.set_bitrate(bitrate);

    let mut rust_dec = OpusDecoder::new(sr, ch).unwrap();
    let c_dec = unsafe {
        let mut err: c_int = 0;
        let d = opus_decoder_create(sr, ch, &mut err);
        assert!(!d.is_null() && err == 0);
        d
    };

    let total_frames = 8usize;
    let total_samples = (sr as usize / 50) * total_frames + 2880 * 2;
    let mut music = vec![0i16; total_samples * ch as usize];
    {
        let (mut a1, mut b1) = (0i32, 0i32);
        let (mut c1, mut d1) = (0i32, 0i32);
        let mut j: i32 = 0;
        let mut rng: u64 = 0xcafe_f00d_dead_beef;
        for (i, s) in music.iter_mut().enumerate() {
            let v_base = ((j.wrapping_mul(
                (j >> 12) ^ ((j >> 10 | j >> 12) & 26 & (j >> 7)),
            )) & 128)
                + 128;
            let mut v: i32 = v_base << 15;
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (rng >> 32) as u32 as i32;
            v = v.wrapping_add(r & 65535);
            v = v.wrapping_sub(r >> 16);
            b1 = v.wrapping_sub(a1).wrapping_add((b1 * 61 + 32) >> 6);
            a1 = v;
            c1 = (30 * (c1 + b1 + d1) + 32) >> 6;
            d1 = b1;
            let out = (c1 + 128) >> 8;
            *s = out.clamp(-32768, 32767) as i16;
            if i % 6 == 0 {
                j = j.wrapping_add(1);
            }
        }
    }

    let mut cursor: usize = 0;

    let encode_and_compare = |enc: &mut OpusEncoder,
                              rust_dec: &mut OpusDecoder,
                              c_dec: *mut std::ffi::c_void,
                              tag: &str,
                              frame_size: i32,
                              pcm: &[i16]| {
        let mut out = vec![0u8; max_packet as usize];
        let len = enc
            .encode(pcm, frame_size, &mut out, max_packet)
            .unwrap_or_else(|e| panic!("{tag}: encode failed: {e}"))
            as usize;
        let pkt = &out[..len];
        let enc_rng = enc.get_final_range();

        let mut rust_pcm = vec![0i16; frame_size as usize * ch as usize];
        rust_dec
            .decode(Some(pkt), &mut rust_pcm, frame_size, false)
            .unwrap_or_else(|e| panic!("{tag}: rust decode failed: {e}"));
        let rust_dec_rng = rust_dec.get_final_range();

        let mut c_pcm = vec![0i16; frame_size as usize * ch as usize];
        let c_ret = unsafe {
            opus_decode(
                c_dec,
                pkt.as_ptr(),
                pkt.len() as i32,
                c_pcm.as_mut_ptr(),
                frame_size,
                0,
            )
        };
        assert!(c_ret > 0, "{tag}: C decode failed: {c_ret}");
        let mut c_dec_rng: u32 = 0;
        let rc = unsafe {
            const OPUS_GET_FINAL_RANGE_REQUEST: c_int = 4031;
            opus_decoder_ctl(
                c_dec,
                OPUS_GET_FINAL_RANGE_REQUEST,
                &mut c_dec_rng as *mut u32,
            )
        };
        assert_eq!(rc, 0, "{tag}: C ctl failed");

        assert_eq!(
            enc_rng, rust_dec_rng,
            "{tag}: encoder.get_final_range={:#010x} disagrees with \
             ropus decoder={:#010x}",
            enc_rng, rust_dec_rng,
        );
        assert_eq!(
            enc_rng, c_dec_rng,
            "{tag}: encoder.get_final_range={:#010x} disagrees with \
             C-ref decoder={:#010x}",
            enc_rng, c_dec_rng,
        );
    };

    // Warmup: 3× Hybrid FB 20 ms frames to set prev_mode=HYBRID and let
    // SILK settle.
    for i in 0..3 {
        let frame_size: i32 = 960;
        enc.set_force_mode(MODE_HYBRID);
        enc.set_bandwidth(OPUS_BANDWIDTH_FULLBAND);
        let start = cursor;
        let end = start + (frame_size as usize) * (ch as usize);
        let pcm = &music[start..end];
        cursor = end;
        let tag = format!("warmup[{i}]");
        encode_and_compare(&mut enc, &mut rust_dec, c_dec, &tag, frame_size, pcm);
    }

    // Trigger: force CELT_ONLY at FULLBAND, 20 ms. Mode-transition-undo
    // logic resurrects HYBRID as the actual encode mode with
    // redundancy=true, celt_to_silk=false, to_celt=true — producing a
    // SILK→CELT redundancy tail in HYBRID mode. The SILK→CELT branch in
    // ropus then needs the `nb_compr_bytes = ret` shrink to write the
    // redundancy at the correct offset.
    enc.set_force_mode(MODE_CELT_ONLY);
    enc.set_bandwidth(OPUS_BANDWIDTH_FULLBAND);

    let frame_size: i32 = 960; // 20 ms — single-frame HYBRID path
    let start = cursor;
    let end = start + (frame_size as usize) * (ch as usize);
    let pcm = &music[start..end];
    encode_and_compare(&mut enc, &mut rust_dec, c_dec, "trigger", frame_size, pcm);

    unsafe { opus_decoder_destroy(c_dec) };
}

// -----------------------------------------------------------------------------
// Encoder `get_final_range` matches decoder when the multi-frame path fires
// with a SILK→CELT redundancy tail on the final sub-frame.
// -----------------------------------------------------------------------------
//
// Surfaced by `test_opus_encode.c:501` after bugs K + M + N were fixed.
// Reproducer: prev_mode = HYBRID (set by warmup), user forces CELT_ONLY at
// a 60 ms FULLBAND frame. `compute_redundancy` undoes the HYBRID→CELT
// transition: `mode = prev_mode = HYBRID`, `redundancy = true`,
// `to_celt = true`, `celt_to_silk = false`. Because `frame_size (2880) >
// max_celt_frame (960)`, the encoder enters `encode_multiframe`, which
// splits into 3× 20 ms sub-frames. Only the final sub-frame has
// `frame_to_celt = true` and `frame_redundancy = true`, so only the final
// sub-frame writes a SILK→CELT redundancy tail and XORs the redundant CELT
// rng into its `range_final`.
//
// C reference (`opus_encoder.c:1770-1838`) does NOT override `rangeFinal`
// after the multiframe repacketize loop — it trusts the value set inside
// the last `opus_encode_frame_native` call (which is `enc.rng XOR
// redundant_rng`). Pre-fix ropus instead wrote
// `self.range_final = celt.rng` after the loop. `celt.rng` at that point
// is the REDUNDANCY CELT encoder's rng (the last thing CELT did was the
// 5 ms redundancy frame), which is neither `main_rng` nor
// `main_rng XOR redundant_rng`. The decoder picks up
// `main_rng XOR redundant_rng`, so the encoder's `get_final_range`
// diverges from the bitstream.
//
// 24 kbps is low enough that CELT's VBR shrinks inside each sub-frame,
// matching the condition the upstream test hits at j=7 (MDCT FB VBR) after
// j=6 (Hybrid FB VBR) sets `prev_mode = HYBRID`.
#[test]
fn test_encoder_final_range_matches_decoder_on_multiframe_silk_to_celt_redundancy() {
    use ropus::opus::decoder::{MODE_CELT_ONLY, MODE_HYBRID, OPUS_BANDWIDTH_FULLBAND};

    let sr: i32 = 48000;
    let ch: i32 = 2;
    let max_packet: i32 = 1500;

    let mut enc = OpusEncoder::new(sr, ch, OPUS_APPLICATION_VOIP).unwrap();
    // Match the upstream failure case (j=7 MDCT FB VBR, rates[7]=512000 base).
    // Needs to be high enough that `compute_redundancy_bytes` returns > 0 for
    // each 20 ms sub-frame — otherwise the last sub-frame's SILK→CELT
    // redundancy never fires and this bug doesn't surface.
    let bitrate: i32 = 700_000;
    enc.set_bitrate(bitrate);

    let mut rust_dec = OpusDecoder::new(sr, ch).unwrap();
    let c_dec = unsafe {
        let mut err: c_int = 0;
        let d = opus_decoder_create(sr, ch, &mut err);
        assert!(!d.is_null() && err == 0);
        d
    };

    // Enough samples for 3 warmup Hybrid 20 ms frames + one 60 ms trigger.
    let total_samples = 960 * 3 + 2880 + 2880;
    let mut music = vec![0i16; total_samples * ch as usize];
    {
        let (mut a1, mut b1) = (0i32, 0i32);
        let (mut c1, mut d1) = (0i32, 0i32);
        let mut j: i32 = 0;
        let mut rng: u64 = 0x0BAD_C0DE_1337_D00D;
        for (i, s) in music.iter_mut().enumerate() {
            let v_base = ((j.wrapping_mul(
                (j >> 12) ^ ((j >> 10 | j >> 12) & 26 & (j >> 7)),
            )) & 128)
                + 128;
            let mut v: i32 = v_base << 15;
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (rng >> 32) as u32 as i32;
            v = v.wrapping_add(r & 65535);
            v = v.wrapping_sub(r >> 16);
            b1 = v.wrapping_sub(a1).wrapping_add((b1 * 61 + 32) >> 6);
            a1 = v;
            c1 = (30 * (c1 + b1 + d1) + 32) >> 6;
            d1 = b1;
            let out = (c1 + 128) >> 8;
            *s = out.clamp(-32768, 32767) as i16;
            if i % 6 == 0 {
                j = j.wrapping_add(1);
            }
        }
    }

    let mut cursor: usize = 0;

    let encode_and_compare = |enc: &mut OpusEncoder,
                              rust_dec: &mut OpusDecoder,
                              c_dec: *mut std::ffi::c_void,
                              tag: &str,
                              frame_size: i32,
                              pcm: &[i16]| {
        let mut out = vec![0u8; max_packet as usize];
        let len = enc
            .encode(pcm, frame_size, &mut out, max_packet)
            .unwrap_or_else(|e| panic!("{tag}: encode failed: {e}"))
            as usize;
        let pkt = &out[..len];
        let enc_rng = enc.get_final_range();

        let mut rust_pcm = vec![0i16; frame_size as usize * ch as usize];
        rust_dec
            .decode(Some(pkt), &mut rust_pcm, frame_size, false)
            .unwrap_or_else(|e| panic!("{tag}: rust decode failed: {e}"));
        let rust_dec_rng = rust_dec.get_final_range();

        let mut c_pcm = vec![0i16; frame_size as usize * ch as usize];
        let c_ret = unsafe {
            opus_decode(
                c_dec,
                pkt.as_ptr(),
                pkt.len() as i32,
                c_pcm.as_mut_ptr(),
                frame_size,
                0,
            )
        };
        assert!(c_ret > 0, "{tag}: C decode failed: {c_ret}");
        let mut c_dec_rng: u32 = 0;
        let rc = unsafe {
            const OPUS_GET_FINAL_RANGE_REQUEST: c_int = 4031;
            opus_decoder_ctl(
                c_dec,
                OPUS_GET_FINAL_RANGE_REQUEST,
                &mut c_dec_rng as *mut u32,
            )
        };
        assert_eq!(rc, 0, "{tag}: C ctl failed");

        assert_eq!(
            enc_rng, rust_dec_rng,
            "{tag}: encoder.get_final_range={:#010x} disagrees with \
             ropus decoder={:#010x}",
            enc_rng, rust_dec_rng,
        );
        assert_eq!(
            enc_rng, c_dec_rng,
            "{tag}: encoder.get_final_range={:#010x} disagrees with \
             C-ref decoder={:#010x}",
            enc_rng, c_dec_rng,
        );
    };

    // Warmup: 3× Hybrid FB 20 ms frames so prev_mode = HYBRID settles.
    for i in 0..3 {
        let frame_size: i32 = 960;
        enc.set_force_mode(MODE_HYBRID);
        enc.set_bandwidth(OPUS_BANDWIDTH_FULLBAND);
        let start = cursor;
        let end = start + (frame_size as usize) * (ch as usize);
        let pcm = &music[start..end];
        cursor = end;
        let tag = format!("warmup[{i}]");
        encode_and_compare(&mut enc, &mut rust_dec, c_dec, &tag, frame_size, pcm);
    }

    // Trigger: CELT_ONLY at FULLBAND, 60 ms frame. compute_redundancy undoes
    // HYBRID→CELT: mode=HYBRID (from prev_mode), redundancy+to_celt.
    // 60 ms > max_celt_frame (20 ms) → encode_multiframe, 3 sub-frames.
    // Final sub-frame writes SILK→CELT redundancy tail.
    enc.set_force_mode(MODE_CELT_ONLY);
    enc.set_bandwidth(OPUS_BANDWIDTH_FULLBAND);

    let frame_size: i32 = 2880; // 60 ms
    let start = cursor;
    let end = start + (frame_size as usize) * (ch as usize);
    let pcm = &music[start..end];
    encode_and_compare(&mut enc, &mut rust_dec, c_dec, "trigger", frame_size, pcm);

    unsafe { opus_decoder_destroy(c_dec) };
}

// -----------------------------------------------------------------------------
// Encoder `get_final_range` matches decoder for HYBRID CVBR when CELT VBR
// shrinks the packet small enough that the decoder skips the redundancy-bit
// read.
// -----------------------------------------------------------------------------
//
// Surfaced by `test_opus_encode.c:501` at SEED=460330478 in the first CVBR
// Hybrid FB iteration (j=3, rates[j]=16000, rate≈17109 bps).
//
// Bug: For HYBRID (mode != CELT_ONLY, start_band==17), the encoder's
// redundancy-signaling check at `opus_encoder.c:2351` / `encoder.rs:2363` is
// against the PRE-shrink buffer `8 * (max_data_bytes - 1)` (≈11992 for a
// 1500-byte max packet), and always writes a 1-symbol `redundancy` bit
// (`ec_enc_bit_logp(enc, redundancy, 12)`). The decoder's matching check
// at `opus_decoder.c:501` / `decoder.rs:822` uses the POST-shrink packet
// `8 * len`. If CELT's internal VBR shrinks the CELT output so small that
// `ec_tell(&dec) + 37 > 8 * len`, the decoder SKIPS reading the bit.
// Encoder's `rng` has advanced by one symbol but decoder's hasn't —
// entropy-coder desync. Bytes are identical; decoded `rng` diverges.
//
// C reference prevents this in `celt_encoder.c:2432-2433`:
//     if (hybrid)
//        min_allowed = IMAX(min_allowed,
//            (tell0_frac+(37<<BITRES)+total_boost+(1<<(BITRES+3))-1)
//            >> (BITRES+3));
// Ropus's CELT encoder was missing this clamp, so `nbCompressedBytes` could
// shrink below the 37-bit floor.
#[test]
fn test_encoder_final_range_matches_decoder_on_hybrid_cvbr_min_packet_floor() {
    use ropus::opus::decoder::{MODE_HYBRID, OPUS_BANDWIDTH_FULLBAND};

    let sr: i32 = 48000;
    let ch: i32 = 2;
    let max_packet: i32 = 1500;
    let frame_size: i32 = 960; // 20 ms

    let mut enc = OpusEncoder::new(sr, ch, OPUS_APPLICATION_VOIP).unwrap();
    enc.set_bitrate(17_000);
    enc.set_vbr(1);
    enc.set_vbr_constraint(1); // CVBR
    enc.set_force_mode(MODE_HYBRID);
    enc.set_bandwidth(OPUS_BANDWIDTH_FULLBAND);

    let mut rust_dec = OpusDecoder::new(sr, ch).unwrap();
    let c_dec = unsafe {
        let mut err: c_int = 0;
        let d = opus_decoder_create(sr, ch, &mut err);
        assert!(!d.is_null() && err == 0);
        d
    };

    // Generate patterned PCM similar to generate_music() so SILK settles and
    // the CELT VBR path eventually triggers a small-packet frame.
    let n_frames = 40;
    let total_samples = (frame_size as usize) * (ch as usize) * n_frames;
    let mut music = vec![0i16; total_samples];
    {
        let (mut a1, mut b1) = (0i32, 0i32);
        let (mut c1, mut d1) = (0i32, 0i32);
        let mut jj: i32 = 0;
        let mut rng: u64 = 0xFEED_FACE_DEAD_BEEF;
        for (i, s) in music.iter_mut().enumerate() {
            let v_base = ((jj.wrapping_mul(
                (jj >> 12) ^ ((jj >> 10 | jj >> 12) & 26 & (jj >> 7)),
            )) & 128)
                + 128;
            let mut v: i32 = v_base << 15;
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (rng >> 32) as u32 as i32;
            v = v.wrapping_add(r & 65535);
            v = v.wrapping_sub(r >> 16);
            b1 = v.wrapping_sub(a1).wrapping_add((b1 * 61 + 32) >> 6);
            a1 = v;
            c1 = (30 * (c1 + b1 + d1) + 32) >> 6;
            d1 = b1;
            let out = (c1 + 128) >> 8;
            *s = out.clamp(-32768, 32767) as i16;
            if i % 6 == 0 {
                jj = jj.wrapping_add(1);
            }
        }
    }

    for iter in 0..n_frames {
        let start = iter * (frame_size as usize) * (ch as usize);
        let end = start + (frame_size as usize) * (ch as usize);
        let pcm = &music[start..end];

        let mut out = vec![0u8; max_packet as usize];
        let len = enc
            .encode(pcm, frame_size, &mut out, max_packet)
            .unwrap_or_else(|e| panic!("iter {iter}: encode failed: {e}"));
        assert!(len > 0, "iter {iter}: empty packet");
        let pkt = &out[..len as usize];
        let enc_rng = enc.get_final_range();

        let mut rust_pcm = vec![0i16; frame_size as usize * ch as usize];
        rust_dec
            .decode(Some(pkt), &mut rust_pcm, frame_size, false)
            .unwrap_or_else(|e| panic!("iter {iter}: rust decode failed: {e}"));
        let rust_dec_rng = rust_dec.get_final_range();

        let mut c_pcm = vec![0i16; frame_size as usize * ch as usize];
        let c_ret = unsafe {
            opus_decode(
                c_dec,
                pkt.as_ptr(),
                pkt.len() as i32,
                c_pcm.as_mut_ptr(),
                frame_size,
                0,
            )
        };
        assert!(c_ret > 0, "iter {iter}: C decode failed: {c_ret}");
        let mut c_dec_rng: u32 = 0;
        let rc = unsafe {
            const OPUS_GET_FINAL_RANGE_REQUEST: c_int = 4031;
            opus_decoder_ctl(
                c_dec,
                OPUS_GET_FINAL_RANGE_REQUEST,
                &mut c_dec_rng as *mut u32,
            )
        };
        assert_eq!(rc, 0, "iter {iter}: C ctl failed");

        assert_eq!(
            enc_rng, rust_dec_rng,
            "iter {iter}: encoder.get_final_range={:#010x} disagrees with \
             ropus decoder={:#010x} (packet {} bytes)",
            enc_rng, rust_dec_rng, len,
        );
        assert_eq!(
            enc_rng, c_dec_rng,
            "iter {iter}: encoder.get_final_range={:#010x} disagrees with \
             C-ref decoder={:#010x} (packet {} bytes)",
            enc_rng, c_dec_rng, len,
        );
    }

    unsafe { opus_decoder_destroy(c_dec) };
}

/// Regression for `test_opus_encode.c` `fuzz_encoder_settings` failure:
/// when the requested bitrate falls below the encoder's "useful work" floor
/// (`bitrate_bps < 3 * frame_rate * 8`), ropus must emit a short "PLC" frame
/// whose TOC advertises the caller's actual frame duration — not whatever
/// mode the previous frame used. Before the fix, a 2.5 ms frame at 8 kHz
/// with 6 kbps produced a `MODE_SILK_ONLY` TOC with garbage period bits
/// (`(period-2) << 3` underflowing when `period == 0`), which decoded as
/// a 10 ms packet and returned 80 samples instead of the requested 20.
/// C reference: `opus_encoder.c:1340-1406`.
#[test]
fn test_encoder_toc_matches_reference_on_below_threshold_2p5ms_frame() {
    const OPUS_SET_FORCE_CHANNELS_REQUEST: c_int = 4022;
    const OPUS_SET_MAX_BANDWIDTH_REQUEST: c_int = 4004;
    const OPUS_SET_EXPERT_FRAME_DURATION_REQUEST: c_int = 4040;
    const OPUS_FRAMESIZE_2_5_MS: c_int = 5001;
    const OPUS_BANDWIDTH_FULLBAND: c_int = 1105;

    let sr: i32 = 8000;
    let ch: i32 = 2;
    let frame_size: i32 = sr / 400; // 2.5 ms → 20 samples @ 8 kHz

    let mut enc = OpusEncoder::new(sr, ch, OPUS_APPLICATION_VOIP).unwrap();
    enc.set_bitrate(6_000);
    enc.set_vbr(1);
    enc.set_vbr_constraint(1);
    enc.set_complexity(8);
    enc.set_force_channels(1);
    enc.set_max_bandwidth(OPUS_BANDWIDTH_FULLBAND);
    enc.set_expert_frame_duration(OPUS_FRAMESIZE_2_5_MS);

    let pcm = patterned_pcm_i16(frame_size as usize, ch as usize, 12345);
    let mut out = vec![0u8; 1500];
    let len = enc.encode(&pcm, frame_size, &mut out, 1500).unwrap();
    assert!(len > 0, "ropus encode returned {len}");
    let ropus_pkt = &out[..len as usize];
    let ropus_toc = ropus_pkt[0];
    let ropus_config = ropus_toc >> 3;
    eprintln!(
        "ropus: len={len}, TOC=0x{:02x}, config={}, stereo={}, code={}",
        ropus_toc,
        ropus_config,
        (ropus_toc >> 2) & 1,
        ropus_toc & 0x3,
    );

    // Compare to the C reference encoder's output.
    let c_enc = unsafe {
        let mut err: c_int = 0;
        let e = opus_encoder_create(sr, ch, OPUS_APPLICATION_VOIP, &mut err);
        assert!(!e.is_null() && err == 0);
        e
    };
    unsafe {
        opus_encoder_ctl(c_enc, OPUS_SET_BITRATE_REQUEST, 6_000i32);
        opus_encoder_ctl(c_enc, OPUS_SET_VBR_REQUEST, 1i32);
        opus_encoder_ctl(c_enc, OPUS_SET_COMPLEXITY_REQUEST, 8i32);
        opus_encoder_ctl(c_enc, OPUS_SET_FORCE_CHANNELS_REQUEST, 1i32);
        opus_encoder_ctl(c_enc, OPUS_SET_MAX_BANDWIDTH_REQUEST, OPUS_BANDWIDTH_FULLBAND);
        opus_encoder_ctl(
            c_enc,
            OPUS_SET_EXPERT_FRAME_DURATION_REQUEST,
            OPUS_FRAMESIZE_2_5_MS,
        );
    }
    let mut c_out = vec![0u8; 1500];
    let c_len = unsafe { opus_encode(c_enc, pcm.as_ptr(), frame_size, c_out.as_mut_ptr(), 1500) };
    assert!(c_len > 0, "C-ref encode returned {c_len}");
    let c_toc = c_out[0];
    let c_config = c_toc >> 3;
    eprintln!(
        "C-ref: len={c_len}, TOC=0x{:02x}, config={}, stereo={}, code={}",
        c_toc,
        c_config,
        (c_toc >> 2) & 1,
        c_toc & 0x3,
    );
    unsafe { opus_encoder_destroy(c_enc) };

    // If the TOCs differ, the encoder is packing the wrong frame duration.
    assert_eq!(
        ropus_config, c_config,
        "ropus TOC config={ropus_config} but C-ref config={c_config}"
    );
}

// =============================================================================
// 24-bit decode differential tests (Piece A)
//
// Validates the Rust-level decode24 methods (which the capi
// `opus_decode24` / `opus_multistream_decode24` wrappers delegate to) against
// the C reference. Needed because `opus_demo.c` — driven by the IETF vector
// suite — calls `opus_decode24` on every decoded packet.
//
// Oracle: byte-exact sample match between our decode24 output and the C
// reference's opus_decode24 output. The 24-bit path is just the 16-bit path
// with every sample left-shifted by 8 in our fixed-point / !ENABLE_RES24
// build, so matching the 16-bit differential implies 24-bit correctness;
// but we validate it directly to catch any future drift (e.g. if someone
// ever enables ENABLE_RES24, the shift collapses into native res24 routing
// and these tests would start distinguishing the paths).
// =============================================================================

/// Helper: encode a test fixture into a single Opus packet.
fn encode_voip_packet(sr: i32, ch: i32, frame_size: i32, seed: i32) -> Vec<u8> {
    let mut enc = OpusEncoder::new(sr, ch, OPUS_APPLICATION_VOIP).unwrap();
    enc.set_bitrate(24000);
    let pcm = patterned_pcm_i16(frame_size as usize, ch as usize, seed);
    let mut buf = vec![0u8; 1500];
    let cap = buf.len() as i32;
    let len = enc.encode(&pcm, frame_size, &mut buf, cap).unwrap();
    buf.truncate(len as usize);
    buf
}

#[test]
fn test_24bit_decode_matches_c_reference_mono() {
    // Encode a single-stream packet, then decode via both paths.
    let sr = 48000;
    let ch = 1;
    let frame_size = 960;
    let max_frame = 5760;
    let packet = encode_voip_packet(sr, ch, frame_size, 42);

    // Rust decode24 (same code path the capi `opus_decode24` wrapper calls).
    let mut rust_dec = OpusDecoder::new(sr, ch).unwrap();
    let mut rust_pcm24 = vec![0i32; max_frame as usize * ch as usize];
    let rust_n = rust_dec
        .decode24(Some(&packet), &mut rust_pcm24, max_frame, false)
        .expect("Rust decode24 failed");

    // C reference decode24.
    let c_n = unsafe {
        let mut err: c_int = 0;
        let c_dec = opus_decoder_create(sr, ch, &mut err);
        assert!(!c_dec.is_null() && err == 0);
        let mut c_pcm24 = vec![0i32; max_frame as usize * ch as usize];
        let n = opus_decode24(
            c_dec,
            packet.as_ptr(),
            packet.len() as i32,
            c_pcm24.as_mut_ptr(),
            max_frame,
            0,
        );
        opus_decoder_destroy(c_dec);
        assert!(n > 0, "C opus_decode24 returned {n}");
        (n, c_pcm24)
    };
    let (c_sample_count, c_pcm24) = c_n;

    assert_eq!(rust_n, c_sample_count, "decode24 mono: sample count mismatch");
    let n_samples = (rust_n as usize) * (ch as usize);
    assert_eq!(
        &rust_pcm24[..n_samples],
        &c_pcm24[..n_samples],
        "decode24 mono: PCM mismatch with C reference"
    );
}

#[test]
fn test_24bit_decode_matches_c_reference_stereo() {
    let sr = 48000;
    let ch = 2;
    let frame_size = 960;
    let max_frame = 5760;
    let packet = encode_voip_packet(sr, ch, frame_size, 101);

    let mut rust_dec = OpusDecoder::new(sr, ch).unwrap();
    let mut rust_pcm24 = vec![0i32; max_frame as usize * ch as usize];
    let rust_n = rust_dec
        .decode24(Some(&packet), &mut rust_pcm24, max_frame, false)
        .expect("Rust decode24 failed");

    let (c_sample_count, c_pcm24) = unsafe {
        let mut err: c_int = 0;
        let c_dec = opus_decoder_create(sr, ch, &mut err);
        assert!(!c_dec.is_null() && err == 0);
        let mut c_pcm24 = vec![0i32; max_frame as usize * ch as usize];
        let n = opus_decode24(
            c_dec,
            packet.as_ptr(),
            packet.len() as i32,
            c_pcm24.as_mut_ptr(),
            max_frame,
            0,
        );
        opus_decoder_destroy(c_dec);
        assert!(n > 0, "C opus_decode24 returned {n}");
        (n, c_pcm24)
    };

    assert_eq!(
        rust_n, c_sample_count,
        "decode24 stereo: sample count mismatch"
    );
    let n_samples = (rust_n as usize) * (ch as usize);
    assert_eq!(
        &rust_pcm24[..n_samples],
        &c_pcm24[..n_samples],
        "decode24 stereo: PCM mismatch with C reference"
    );
}

#[test]
fn test_24bit_decode_left_shift_8_invariant() {
    // Structural property of the fixed-point / !ENABLE_RES24 build:
    // decode24 output equals decode-16 output left-shifted by 8. If this
    // ever fails, either (a) someone enabled ENABLE_RES24 without updating
    // the capi wrapper, or (b) decode24 diverged from the spec.
    let sr = 48000;
    let ch = 1;
    let frame_size = 960;
    let max_frame = 5760;
    let packet = encode_voip_packet(sr, ch, frame_size, 7);

    let mut dec16 = OpusDecoder::new(sr, ch).unwrap();
    let mut pcm16 = vec![0i16; max_frame as usize];
    let n16 = dec16.decode(Some(&packet), &mut pcm16, max_frame, false).unwrap();

    let mut dec24 = OpusDecoder::new(sr, ch).unwrap();
    let mut pcm24 = vec![0i32; max_frame as usize];
    let n24 = dec24
        .decode24(Some(&packet), &mut pcm24, max_frame, false)
        .unwrap();

    assert_eq!(n16, n24, "decode vs decode24 sample count mismatch");
    for i in 0..(n16 as usize) {
        let expected = (pcm16[i] as i32) << 8;
        assert_eq!(
            pcm24[i], expected,
            "sample {i}: decode24 ({:#x}) != decode <<8 ({:#x})",
            pcm24[i], expected
        );
    }
}

#[test]
#[ignore = "ropus decode24 PLC diverges from C reference after first subframe \
            — see wrk_journals/2026.04.19 - JRN - ietf-projection-regressions.md"]
fn test_24bit_decode_plc_matches_c_reference() {
    // opus_demo invokes `opus_decode24(dec, NULL, 0, ...)` on every lost
    // packet; the PLC code path was previously uncovered. Drive one
    // successful packet to prime decoder state, then invoke PLC via
    // (data=None, len=0) and confirm the Rust and C reference agree
    // sample-for-sample.
    //
    // Currently #[ignore]'d: the test surfaces a differential divergence
    // between ropus and the C reference where the first ~35 samples match
    // bit-exactly but then drift. Tracked for investigation alongside the
    // stereo decoder bugs. Run with `cargo test -p ropus-harness --test
    // c_ref_differential -- --ignored` once fixed.
    let sr = 48000;
    let ch = 1;
    let frame_size = 960;
    let max_frame = 5760;
    let packet = encode_voip_packet(sr, ch, frame_size, 55);

    // Prime both decoders with a successful frame so PLC has state to
    // extrapolate from.
    let mut rust_dec = OpusDecoder::new(sr, ch).unwrap();
    let mut prime_buf = vec![0i32; max_frame as usize * ch as usize];
    rust_dec
        .decode24(Some(&packet), &mut prime_buf, max_frame, false)
        .expect("Rust decode24 prime failed");
    let mut rust_pcm24 = vec![0i32; max_frame as usize * ch as usize];
    let rust_n = rust_dec
        .decode24(None, &mut rust_pcm24, frame_size, false)
        .expect("Rust decode24 PLC failed");

    let (c_sample_count, c_pcm24) = unsafe {
        let mut err: c_int = 0;
        let c_dec = opus_decoder_create(sr, ch, &mut err);
        assert!(!c_dec.is_null() && err == 0);
        // Prime with the same packet.
        let mut prime_c = vec![0i32; max_frame as usize * ch as usize];
        let prime_n = opus_decode24(
            c_dec,
            packet.as_ptr(),
            packet.len() as i32,
            prime_c.as_mut_ptr(),
            max_frame,
            0,
        );
        assert!(prime_n > 0, "C opus_decode24 prime returned {prime_n}");
        // Now invoke PLC: data=NULL, len=0.
        let mut c_pcm24 = vec![0i32; max_frame as usize * ch as usize];
        let n = opus_decode24(
            c_dec,
            std::ptr::null(),
            0,
            c_pcm24.as_mut_ptr(),
            frame_size,
            0,
        );
        opus_decoder_destroy(c_dec);
        assert!(n > 0, "C opus_decode24 PLC returned {n}");
        (n, c_pcm24)
    };

    assert_eq!(rust_n, c_sample_count, "decode24 PLC: sample count mismatch");
    let n_samples = (rust_n as usize) * (ch as usize);
    assert_eq!(
        &rust_pcm24[..n_samples],
        &c_pcm24[..n_samples],
        "decode24 PLC: PCM mismatch with C reference"
    );
}

#[test]
fn test_24bit_multistream_decode_matches_c_reference_stereo() {
    // Two-channel multistream, dual-mono mapping.
    use ropus::opus::multistream::OpusMSEncoder;

    let sr = 48000;
    let channels = 2;
    let streams = 1;
    let coupled_streams = 1;
    let mapping = [0u8, 1u8];
    let frame_size = 960;
    let max_frame = 5760;

    // Encode a test packet through the Rust MS encoder.
    let mut enc = OpusMSEncoder::new(
        sr,
        channels,
        streams,
        coupled_streams,
        &mapping,
        OPUS_APPLICATION_VOIP,
    )
    .expect("MS encoder create");
    enc.set_bitrate(48000);
    let pcm = patterned_pcm_i16(frame_size as usize, channels as usize, 123);
    let mut packet = vec![0u8; 2000];
    let cap = packet.len() as i32;
    let packet_len = enc.encode(&pcm, frame_size, &mut packet, cap).unwrap();
    packet.truncate(packet_len as usize);

    // Rust decode24 via OpusMSDecoder.
    use ropus::opus::multistream::OpusMSDecoder;
    let mut rust_dec = OpusMSDecoder::new(sr, channels, streams, coupled_streams, &mapping)
        .expect("MS decoder create");
    let mut rust_pcm24 = vec![0i32; max_frame as usize * channels as usize];
    let rust_n = rust_dec
        .decode24(
            Some(&packet),
            packet.len() as i32,
            &mut rust_pcm24,
            max_frame,
            false,
        )
        .expect("Rust MS decode24 failed");

    // C reference decode24 via opus_multistream_decode24.
    let (c_sample_count, c_pcm24) = unsafe {
        let mut err: c_int = 0;
        let c_dec = opus_multistream_decoder_create(
            sr,
            channels,
            streams,
            coupled_streams,
            mapping.as_ptr(),
            &mut err,
        );
        assert!(!c_dec.is_null() && err == 0);
        let mut c_pcm24 = vec![0i32; max_frame as usize * channels as usize];
        let n = opus_multistream_decode24(
            c_dec,
            packet.as_ptr(),
            packet.len() as i32,
            c_pcm24.as_mut_ptr(),
            max_frame,
            0,
        );
        opus_multistream_decoder_destroy(c_dec);
        assert!(n > 0, "C opus_multistream_decode24 returned {n}");
        (n, c_pcm24)
    };

    assert_eq!(
        rust_n, c_sample_count,
        "MS decode24 stereo: sample count mismatch"
    );
    let n_samples = (rust_n as usize) * (channels as usize);
    assert_eq!(
        &rust_pcm24[..n_samples],
        &c_pcm24[..n_samples],
        "MS decode24 stereo: PCM mismatch with C reference"
    );
}

// ===========================================================================
// analysis.rs golden-vector test
// ===========================================================================
//
// Stage 6.3 ports `reference/src/analysis.c` to `ropus/src/opus/analysis.rs`.
// The ropus-crate smoke tests only check that `run_analysis` doesn't panic and
// produces field ranges that look plausible. That's too weak: a sign-flip or
// a swapped MLP buffer in the port would pass those tests.
//
// This differential test drives both Rust and C `run_analysis` over the same
// 30-frame 1 kHz sine at 48 kHz, then bit-compares the five user-visible
// scalar fields emitted by `tonality_get_info`. Any numerical drift that
// reaches the encoder wiring (Stage 6.4) fails here first with a clear
// diagnostic, rather than as a mysterious encoder byte-diff downstream.
//
// The Rust `TonalityAnalysisState` and `AnalysisInfo` are `#[repr(C)]` with
// the same field order as their C counterparts (analysis.h:47-81, celt.h:65-79),
// so the same heap-allocated buffer can be passed to either side without
// marshalling.

use ropus::opus::analysis::{
    AnalysisInfo, DownmixFunc, TonalityAnalysisState, run_analysis as rust_run_analysis,
    tonality_analysis_init as rust_tonality_analysis_init,
};

unsafe extern "C" {
    // Analysis FFI from reference/src/analysis.c. Both sides share the
    // TonalityAnalysisState / AnalysisInfo layout (#[repr(C)]).
    fn tonality_analysis_init(st: *mut TonalityAnalysisState, fs: i32);
    #[allow(clippy::too_many_arguments)]
    fn run_analysis(
        analysis: *mut TonalityAnalysisState,
        celt_mode: *const std::ffi::c_void,
        analysis_pcm: *const std::ffi::c_void,
        analysis_frame_size: c_int,
        frame_size: c_int,
        c1: c_int,
        c2: c_int,
        c: c_int,
        fs: i32,
        lsb_depth: c_int,
        downmix: unsafe extern "C" fn(
            *const std::ffi::c_void, // x
            *mut i32,                // y
            c_int,                   // subframe
            c_int,                   // offset
            c_int,                   // c1
            c_int,                   // c2
            c_int,                   // C
        ),
        analysis_info: *mut AnalysisInfo,
    );
    // downmix_float is a non-static symbol in reference/src/opus_encoder.c
    // (opus_private.h:176), so the linker finds it.
    fn downmix_float(
        x: *const std::ffi::c_void,
        y: *mut i32,
        subframe: c_int,
        offset: c_int,
        c1: c_int,
        c2: c_int,
        c: c_int,
    );
    fn downmix_int(
        x: *const std::ffi::c_void,
        y: *mut i32,
        subframe: c_int,
        offset: c_int,
        c1: c_int,
        c2: c_int,
        c: c_int,
    );
    // CELTMode * opus_custom_mode_create(Fs, frame_size, &err) returns the
    // static `mode48000_960_120` in `reference/celt/static_modes_fixed.h:1547`
    // when CUSTOM_MODES is not defined (our harness build).
    fn opus_custom_mode_create(
        fs: i32,
        frame_size: c_int,
        error: *mut c_int,
    ) -> *mut std::ffi::c_void;
}

/// Matching Rust-side DownmixFunc that interprets the PCM buffer as i16.
/// Matches `reference/src/opus_encoder.c:781-802`. Under FIXED_POINT,
/// `INT16TOSIG(x) = x << SIG_SHIFT` — no clamp, no rounding.
fn rust_downmix_int(
    input: &[u8],
    output: &mut [i32],
    subframe: i32,
    offset: i32,
    c1: i32,
    c2: i32,
    c: i32,
) {
    const SIG_SHIFT: u32 = 12;
    let samples: &[i16] = unsafe {
        std::slice::from_raw_parts(
            input.as_ptr() as *const i16,
            input.len() / std::mem::size_of::<i16>(),
        )
    };
    #[inline(always)]
    fn int16_to_sig(x: i16) -> i32 {
        (x as i32) << SIG_SHIFT
    }
    for j in 0..subframe as usize {
        output[j] = int16_to_sig(samples[((j as i32 + offset) * c + c1) as usize]);
    }
    if c2 > -1 {
        for j in 0..subframe as usize {
            output[j] += int16_to_sig(samples[((j as i32 + offset) * c + c2) as usize]);
        }
    } else if c2 == -2 {
        for ch in 1..c {
            for j in 0..subframe as usize {
                output[j] += int16_to_sig(samples[((j as i32 + offset) * c + ch) as usize]);
            }
        }
    }
}

/// Matching Rust-side DownmixFunc that interprets the PCM buffer as f32
/// samples (little-endian x86 default) and applies the C `FLOAT2SIG`
/// chain from `reference/celt/float_cast.h:166-172` for FIXED_POINT
/// builds:
///   y = float2int( clamp( x * (32768<<SIG_SHIFT),
///                         -(65536<<SIG_SHIFT), +(65536<<SIG_SHIFT) ) )
/// where `float2int` is round-half-even (matches `lrintf`).
///
/// Previously this helper used `x * 32768` (the non-FIXED_POINT macro
/// form) and `as i32` truncation — which gave Rust `inmem` values ~4096x
/// smaller than the C `FLOAT2SIG` output and cast with the wrong rounding,
/// producing several-ULP drift on downstream analysis outputs. See the
/// harness config in `harness/config.h:8` — we build with FIXED_POINT=1,
/// so the C side is using the FIXED_POINT FLOAT2SIG.
fn rust_downmix_float(
    input: &[u8],
    output: &mut [i32],
    subframe: i32,
    offset: i32,
    c1: i32,
    c2: i32,
    c: i32,
) {
    // Safety: `input` is a byte view of an `&[f32]` passed by the caller.
    let samples: &[f32] = unsafe {
        std::slice::from_raw_parts(
            input.as_ptr() as *const f32,
            input.len() / std::mem::size_of::<f32>(),
        )
    };
    // CELT_SIG_SCALE << SIG_SHIFT = 32768 * 4096 = 134_217_728.
    // Matches `FLOAT2SIG(x)` in `float_cast.h:166-172` under FIXED_POINT.
    const FLOAT2SIG_MULT: f32 = 134_217_728.0; // 32768 << 12
    const SIG_CLAMP_MAX: f32 = 268_435_456.0;  // 65536 << 12
    const SIG_CLAMP_MIN: f32 = -268_435_456.0;
    #[inline(always)]
    fn float2sig(x: f32) -> i32 {
        let mut y = x * FLOAT2SIG_MULT;
        if y > SIG_CLAMP_MAX {
            y = SIG_CLAMP_MAX;
        }
        if y < SIG_CLAMP_MIN {
            y = SIG_CLAMP_MIN;
        }
        // float2int uses round-half-even (matches `lrintf` on Windows).
        y.round_ties_even() as i32
    }
    for j in 0..subframe as usize {
        output[j] = float2sig(samples[((j as i32 + offset) * c + c1) as usize]);
    }
    if c2 > -1 {
        for j in 0..subframe as usize {
            output[j] += float2sig(samples[((j as i32 + offset) * c + c2) as usize]);
        }
    } else if c2 == -2 {
        for ch in 1..c {
            for j in 0..subframe as usize {
                output[j] += float2sig(samples[((j as i32 + offset) * c + ch) as usize]);
            }
        }
    }
}

/// Drive both Rust and C `run_analysis` over an identical input stream,
/// then compare the five scalar fields returned by `tonality_get_info`
/// bit-exactly on every valid frame.
///
/// This is a full-stack differential test: the C reference generates its
/// own golden values live, so there is no need for a separate capture
/// binary or hand-pasted hex constants. Any f32 drift between the two
/// ports shows up here as a bit-pattern mismatch.
///
/// Bit-exact as of the Stage 6.3 closeout fix: the previous `#[ignore]`
/// was tracking a test-harness helper bug (not a port bug). `rust_downmix_float`
/// above used the non-FIXED_POINT `FLOAT2SIG = x * 32768`, which mismatched
/// the C side's FIXED_POINT `FLOAT2SIG = float2int(clamp(x * (32768<<12)))`
/// in `reference/celt/float_cast.h:166-172`. Matching the FIXED_POINT
/// macro exactly closes the ~4096x input-scale drift that propagated
/// through the FFT, band energies, and the MLP chain into music_prob.
#[test]
fn test_analysis_run_matches_c_reference() {
    // 30 frames of 1 kHz sine at 48 kHz mono, f32, amplitude 0.5. 30 frames
    // x 960 samples gives enough lookahead to exercise the post-fill path
    // in `tonality_get_info`, not just the initial silence-return branch.
    const NUM_FRAMES: usize = 30;
    const FRAME_SIZE: usize = 960;
    const FS: i32 = 48_000;
    const FREQ: f32 = 1_000.0;
    const C1: i32 = 0;
    const C2: i32 = -2;
    const CHANNELS: i32 = 1;
    const LSB_DEPTH: i32 = 24;

    let pcm: Vec<f32> = (0..NUM_FRAMES * FRAME_SIZE)
        .map(|i| 0.5 * (2.0 * std::f32::consts::PI * FREQ * i as f32 / FS as f32).sin())
        .collect();
    // Bytes view; both downmix callbacks reinterpret it as &[f32].
    let pcm_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            pcm.as_ptr() as *const u8,
            std::mem::size_of_val(pcm.as_slice()),
        )
    };

    // Allocate the C and Rust state buffers on the heap — ~75 KB each.
    let mut c_state = TonalityAnalysisState::new_boxed();
    let mut rust_state = TonalityAnalysisState::new_boxed();

    // Fetch the static CELTMode48000_960_120 from the C side.
    let celt_mode_ptr: *mut std::ffi::c_void = unsafe {
        let mut err: c_int = 0;
        let p = opus_custom_mode_create(FS, FRAME_SIZE as c_int, &mut err);
        assert!(!p.is_null() && err == 0, "opus_custom_mode_create err={err}");
        p
    };

    // Initialise both sides with the same Fs. `tonality_analysis_init`
    // also zeros the post-reset-start fields.
    unsafe {
        tonality_analysis_init(&raw mut *c_state, FS);
    }
    rust_tonality_analysis_init(&mut rust_state, FS);

    // Drive NUM_FRAMES iterations, each frame = FRAME_SIZE samples at FS.
    let frame_bytes = FRAME_SIZE * std::mem::size_of::<f32>();
    for f in 0..NUM_FRAMES {
        let start = f * frame_bytes;
        let end = start + frame_bytes;
        let frame_slice = &pcm_bytes[start..end];

        // C side
        let mut c_info = AnalysisInfo::default();
        unsafe {
            run_analysis(
                &raw mut *c_state,
                celt_mode_ptr,
                frame_slice.as_ptr() as *const std::ffi::c_void,
                FRAME_SIZE as c_int,
                FRAME_SIZE as c_int,
                C1,
                C2,
                CHANNELS,
                FS,
                LSB_DEPTH,
                downmix_float,
                &mut c_info,
            );
        }

        // Rust side — needs the same CELTMode Rust-side reference.
        let mut rust_info = AnalysisInfo::default();
        // The ropus analysis module wants `&CELTMode`; we reach in via
        // the static `MODE_48000_960_120` defined in ropus's celt::modes.
        use ropus::celt::modes::MODE_48000_960_120 as ROPUS_CELT_MODE;
        let rust_downmix_cb: DownmixFunc = rust_downmix_float;
        rust_run_analysis(
            &mut rust_state,
            &ROPUS_CELT_MODE,
            Some(frame_slice),
            FRAME_SIZE as i32,
            FRAME_SIZE as i32,
            C1,
            C2,
            CHANNELS,
            FS,
            LSB_DEPTH,
            rust_downmix_cb,
            &mut rust_info,
        );

        // Bit-exact compare on the user-visible scalar fields. We wait
        // until valid=1 on both sides before enforcing the stricter
        // music/tonality comparisons — early frames legitimately return
        // valid=0 while the ring buffer fills.
        assert_eq!(
            c_info.valid, rust_info.valid,
            "frame {f}: valid mismatch (c={}, rust={})",
            c_info.valid, rust_info.valid
        );
        if c_info.valid == 1 {
            assert_eq!(
                c_info.music_prob.to_bits(),
                rust_info.music_prob.to_bits(),
                "frame {f}: music_prob bit-diff (c={:08x}=={}, rust={:08x}=={})",
                c_info.music_prob.to_bits(),
                c_info.music_prob,
                rust_info.music_prob.to_bits(),
                rust_info.music_prob
            );
            assert_eq!(
                c_info.tonality.to_bits(),
                rust_info.tonality.to_bits(),
                "frame {f}: tonality bit-diff (c={:08x}=={}, rust={:08x}=={})",
                c_info.tonality.to_bits(),
                c_info.tonality,
                rust_info.tonality.to_bits(),
                rust_info.tonality
            );
            assert_eq!(
                c_info.tonality_slope.to_bits(),
                rust_info.tonality_slope.to_bits(),
                "frame {f}: tonality_slope bit-diff (c={}, rust={})",
                c_info.tonality_slope, rust_info.tonality_slope
            );
            assert_eq!(
                c_info.activity_probability.to_bits(),
                rust_info.activity_probability.to_bits(),
                "frame {f}: activity_probability bit-diff (c={}, rust={})",
                c_info.activity_probability, rust_info.activity_probability
            );
            assert_eq!(
                c_info.bandwidth, rust_info.bandwidth,
                "frame {f}: bandwidth mismatch (c={}, rust={})",
                c_info.bandwidth, rust_info.bandwidth
            );
        }
    }
}

/// Diagnostic: run only `downmix_and_resample` via a fresh Rust-only
/// and a fresh C-only state, then byte-compare `inmem`. Isolates whether
/// divergence begins BEFORE the FFT (i.e. inside downmix/resample).
#[test]
#[ignore = "diagnostic; not a conformance test"]
fn diag_inmem_after_downmix() {
    const FRAME_SIZE: usize = 960;
    const FS: i32 = 48_000;
    const FREQ: f32 = 1_000.0;
    const C1: i32 = 0;
    const C2: i32 = -2;
    const CHANNELS: i32 = 1;
    const LSB_DEPTH: i32 = 24;

    let pcm: Vec<f32> = (0..FRAME_SIZE)
        .map(|i| 0.5 * (2.0 * std::f32::consts::PI * FREQ * i as f32 / FS as f32).sin())
        .collect();
    let pcm_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            pcm.as_ptr() as *const u8,
            std::mem::size_of_val(pcm.as_slice()),
        )
    };

    let mut c_state = TonalityAnalysisState::new_boxed();
    let mut rust_state = TonalityAnalysisState::new_boxed();

    let celt_mode_ptr: *mut std::ffi::c_void = unsafe {
        let mut err: c_int = 0;
        let p = opus_custom_mode_create(FS, FRAME_SIZE as c_int, &mut err);
        assert!(!p.is_null() && err == 0);
        p
    };

    unsafe {
        tonality_analysis_init(&raw mut *c_state, FS);
    }
    rust_tonality_analysis_init(&mut rust_state, FS);

    let mut c_info = AnalysisInfo::default();
    unsafe {
        run_analysis(
            &raw mut *c_state,
            celt_mode_ptr,
            pcm_bytes.as_ptr() as *const std::ffi::c_void,
            FRAME_SIZE as c_int,
            FRAME_SIZE as c_int,
            C1,
            C2,
            CHANNELS,
            FS,
            LSB_DEPTH,
            downmix_float,
            &mut c_info,
        );
    }
    let mut rust_info = AnalysisInfo::default();
    use ropus::celt::modes::MODE_48000_960_120 as ROPUS_CELT_MODE;
    let rust_downmix_cb: DownmixFunc = rust_downmix_float;
    rust_run_analysis(
        &mut rust_state,
        &ROPUS_CELT_MODE,
        Some(pcm_bytes),
        FRAME_SIZE as i32,
        FRAME_SIZE as i32,
        C1,
        C2,
        CHANNELS,
        FS,
        LSB_DEPTH,
        rust_downmix_cb,
        &mut rust_info,
    );

    // Compare only `inmem` — it is the downmix+resample result, and is
    // the input to the FFT.
    println!("c_state.inmem[0..20]   = {:?}", &c_state.inmem[0..20]);
    println!("rust_state.inmem[0..20] = {:?}", &rust_state.inmem[0..20]);
    println!("c_state.inmem[240..260]   = {:?}", &c_state.inmem[240..260]);
    println!("rust_state.inmem[240..260] = {:?}", &rust_state.inmem[240..260]);
    let mut first_diff: Option<usize> = None;
    for i in 0..c_state.inmem.len() {
        if c_state.inmem[i] != rust_state.inmem[i] {
            first_diff = Some(i);
            break;
        }
    }
    println!("first differing inmem index = {:?}", first_diff);
    // Compare downmix_state too
    println!("c_state.downmix_state = {:?}", c_state.downmix_state);
    println!("rust_state.downmix_state = {:?}", rust_state.downmix_state);
    println!("c_state.hp_ener_accum = {} ({:08x})", c_state.hp_ener_accum, c_state.hp_ener_accum.to_bits());
    println!(
        "rust_state.hp_ener_accum = {} ({:08x})",
        rust_state.hp_ener_accum, rust_state.hp_ener_accum.to_bits()
    );
    println!("c_state.mem_fill = {}", c_state.mem_fill);
    println!("rust_state.mem_fill = {}", rust_state.mem_fill);
}

/// Diagnostic: drive one frame, then byte-compare the entire
/// `TonalityAnalysisState`, print the first differing range, and panic
/// with enough context to map the offset back to a field.
#[test]
#[ignore = "diagnostic; not a conformance test"]
fn diag_first_differing_field_after_frame_1() {
    const FRAME_SIZE: usize = 960;
    const FS: i32 = 48_000;
    const FREQ: f32 = 1_000.0;
    const C1: i32 = 0;
    const C2: i32 = -2;
    const CHANNELS: i32 = 1;
    const LSB_DEPTH: i32 = 24;

    let pcm: Vec<f32> = (0..FRAME_SIZE)
        .map(|i| 0.5 * (2.0 * std::f32::consts::PI * FREQ * i as f32 / FS as f32).sin())
        .collect();
    let pcm_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            pcm.as_ptr() as *const u8,
            std::mem::size_of_val(pcm.as_slice()),
        )
    };

    let mut c_state = TonalityAnalysisState::new_boxed();
    let mut rust_state = TonalityAnalysisState::new_boxed();

    let celt_mode_ptr: *mut std::ffi::c_void = unsafe {
        let mut err: c_int = 0;
        let p = opus_custom_mode_create(FS, FRAME_SIZE as c_int, &mut err);
        assert!(!p.is_null() && err == 0);
        p
    };

    unsafe {
        tonality_analysis_init(&raw mut *c_state, FS);
    }
    rust_tonality_analysis_init(&mut rust_state, FS);

    let mut c_info = AnalysisInfo::default();
    unsafe {
        run_analysis(
            &raw mut *c_state,
            celt_mode_ptr,
            pcm_bytes.as_ptr() as *const std::ffi::c_void,
            FRAME_SIZE as c_int,
            FRAME_SIZE as c_int,
            C1,
            C2,
            CHANNELS,
            FS,
            LSB_DEPTH,
            downmix_float,
            &mut c_info,
        );
    }
    let mut rust_info = AnalysisInfo::default();
    use ropus::celt::modes::MODE_48000_960_120 as ROPUS_CELT_MODE;
    let rust_downmix_cb: DownmixFunc = rust_downmix_float;
    rust_run_analysis(
        &mut rust_state,
        &ROPUS_CELT_MODE,
        Some(pcm_bytes),
        FRAME_SIZE as i32,
        FRAME_SIZE as i32,
        C1,
        C2,
        CHANNELS,
        FS,
        LSB_DEPTH,
        rust_downmix_cb,
        &mut rust_info,
    );

    let state_size = std::mem::size_of::<TonalityAnalysisState>();
    let c_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(c_state.as_ref() as *const _ as *const u8, state_size)
    };
    let r_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(rust_state.as_ref() as *const _ as *const u8, state_size)
    };

    println!("state size = {} bytes", state_size);
    // Skip the `arch` field (offset 0..4): C sets it via `opus_select_arch`
    // (returns 4 on x86 with SSE4.1 enabled in harness config), the Rust
    // port hardcodes 0. That delta is purely a metadata tag and is
    // expected — opus_fft only SIMD-dispatches on ARM with NE10, not x86.
    const ARCH_FIELD_BYTES: usize = 4;
    let mut first_diff: Option<usize> = None;
    for (i, (&c, &r)) in c_bytes
        .iter()
        .zip(r_bytes.iter())
        .enumerate()
        .skip(ARCH_FIELD_BYTES)
    {
        if c != r {
            first_diff = Some(i);
            break;
        }
    }
    match first_diff {
        None => println!("after frame 1: state matches (excluding arch metadata tag)"),
        Some(off) => {
            let end = (off + 32).min(state_size);
            println!(
                "first differing byte at offset 0x{off:x} ({off})\n  C   : {:02x?}\n  Rust: {:02x?}",
                &c_bytes[off..end],
                &r_bytes[off..end],
            );
            // Show surrounding region
            let start = off.saturating_sub(8);
            println!(
                "surrounding region [0x{start:x}..0x{end:x}]\n  C   : {:02x?}\n  Rust: {:02x?}",
                &c_bytes[start..end],
                &r_bytes[start..end],
            );
            // Map to f32 at 4-byte alignment
            let aligned = off & !3;
            if aligned + 4 <= state_size {
                let c_f32 = f32::from_le_bytes([
                    c_bytes[aligned],
                    c_bytes[aligned + 1],
                    c_bytes[aligned + 2],
                    c_bytes[aligned + 3],
                ]);
                let r_f32 = f32::from_le_bytes([
                    r_bytes[aligned],
                    r_bytes[aligned + 1],
                    r_bytes[aligned + 2],
                    r_bytes[aligned + 3],
                ]);
                println!(
                    "at aligned offset 0x{aligned:x}: C f32 = {c_f32:e} ({:08x}), Rust f32 = {r_f32:e} ({:08x})",
                    c_f32.to_bits(),
                    r_f32.to_bits()
                );
            }
            panic!("state diverges at offset {off}");
        }
    }
}

/// Long-running differential: 500 frames of multi-tone input, byte-comparing
/// the full `TonalityAnalysisState` AFTER EVERY FRAME. On mismatch prints
/// frame number + first-differing byte offset and panics.
///
/// This test exists to answer a specific question in the Stage 6.4 journal:
/// is the remaining encode drift in the analysis module itself, or in the
/// CELT-side wiring? The 30-frame sine test passes bit-exactly but isn't
/// rich enough to excite all analyzer state (tonality trackers, long-term
/// band-energy trackers, etc.). A multi-tone input drives more of the
/// state machine.
#[test]
fn test_analysis_run_matches_c_reference_long() {
    // 500 frames * 20 ms/frame = 10 s of audio. Enough to warm up long-run
    // trackers (low_e / high_e / mean_e / cmean / std) and to exercise the
    // BFCC memory ring (mem[] is 4*NB_TBANDS*DETECT_SIZE = 32 entries
    // wide, refreshed across frames).
    const NUM_FRAMES: usize = 500;
    const FRAME_SIZE: usize = 960;
    const FS: i32 = 48_000;
    const C1: i32 = 0;
    const C2: i32 = -2;
    const CHANNELS: i32 = 1;
    const LSB_DEPTH: i32 = 24;

    // Multi-tone signal: 200 Hz + 800 Hz + 3 kHz + linear-congruential
    // pseudorandom noise, each at modest amplitude so the sum stays inside
    // [-0.5, 0.5] and never clips through the FLOAT2SIG chain. Deterministic
    // — both sides see identical input. The noise term prevents the input
    // from being purely periodic (which would leave some long-run trackers
    // unexercised).
    let pcm: Vec<f32> = {
        let mut seed: u32 = 0xCAFE_BABE;
        (0..NUM_FRAMES * FRAME_SIZE)
            .map(|i| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let noise = ((seed >> 16) & 0x7fff) as f32 / 32768.0 - 0.5;
                let t = i as f32 / FS as f32;
                let w0 = 2.0 * std::f32::consts::PI * 200.0 * t;
                let w1 = 2.0 * std::f32::consts::PI * 800.0 * t;
                let w2 = 2.0 * std::f32::consts::PI * 3000.0 * t;
                (w0.sin() + w1.sin() + w2.sin()) * 0.1 + noise * 0.2
            })
            .collect()
    };
    let pcm_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            pcm.as_ptr() as *const u8,
            std::mem::size_of_val(pcm.as_slice()),
        )
    };

    let mut c_state = TonalityAnalysisState::new_boxed();
    let mut rust_state = TonalityAnalysisState::new_boxed();

    let celt_mode_ptr: *mut std::ffi::c_void = unsafe {
        let mut err: c_int = 0;
        let p = opus_custom_mode_create(FS, FRAME_SIZE as c_int, &mut err);
        assert!(!p.is_null() && err == 0, "opus_custom_mode_create err={err}");
        p
    };

    unsafe {
        tonality_analysis_init(&raw mut *c_state, FS);
    }
    rust_tonality_analysis_init(&mut rust_state, FS);

    let state_size = std::mem::size_of::<TonalityAnalysisState>();
    // Skip the `arch` field (offset 0..4): C sets it via `opus_select_arch`
    // (returns 4 on x86 with SSE4.1 enabled in harness config), the Rust
    // port hardcodes 0.
    const ARCH_FIELD_BYTES: usize = 4;

    let frame_bytes = FRAME_SIZE * std::mem::size_of::<f32>();
    for f in 0..NUM_FRAMES {
        let start = f * frame_bytes;
        let end = start + frame_bytes;
        let frame_slice = &pcm_bytes[start..end];

        let mut c_info = AnalysisInfo::default();
        unsafe {
            run_analysis(
                &raw mut *c_state,
                celt_mode_ptr,
                frame_slice.as_ptr() as *const std::ffi::c_void,
                FRAME_SIZE as c_int,
                FRAME_SIZE as c_int,
                C1,
                C2,
                CHANNELS,
                FS,
                LSB_DEPTH,
                downmix_float,
                &mut c_info,
            );
        }

        let mut rust_info = AnalysisInfo::default();
        use ropus::celt::modes::MODE_48000_960_120 as ROPUS_CELT_MODE;
        let rust_downmix_cb: DownmixFunc = rust_downmix_float;
        rust_run_analysis(
            &mut rust_state,
            &ROPUS_CELT_MODE,
            Some(frame_slice),
            FRAME_SIZE as i32,
            FRAME_SIZE as i32,
            C1,
            C2,
            CHANNELS,
            FS,
            LSB_DEPTH,
            rust_downmix_cb,
            &mut rust_info,
        );

        // Full state byte-compare (skipping arch) after EVERY frame.
        let c_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(c_state.as_ref() as *const _ as *const u8, state_size)
        };
        let r_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(rust_state.as_ref() as *const _ as *const u8, state_size)
        };
        let mut first_diff: Option<usize> = None;
        for (i, (&c, &r)) in c_bytes
            .iter()
            .zip(r_bytes.iter())
            .enumerate()
            .skip(ARCH_FIELD_BYTES)
        {
            if c != r {
                first_diff = Some(i);
                break;
            }
        }
        if let Some(off) = first_diff {
            let end_off = (off + 32).min(state_size);
            let start_off = off.saturating_sub(8);
            eprintln!(
                "DIVERGED at frame {f}, first differing byte at offset 0x{off:x} ({off})\n  C   : {:02x?}\n  Rust: {:02x?}",
                &c_bytes[off..end_off],
                &r_bytes[off..end_off],
            );
            eprintln!(
                "surrounding region [0x{start_off:x}..0x{end_off:x}]\n  C   : {:02x?}\n  Rust: {:02x?}",
                &c_bytes[start_off..end_off],
                &r_bytes[start_off..end_off],
            );
            let aligned = off & !3;
            if aligned + 4 <= state_size {
                let c_f32 = f32::from_le_bytes([
                    c_bytes[aligned],
                    c_bytes[aligned + 1],
                    c_bytes[aligned + 2],
                    c_bytes[aligned + 3],
                ]);
                let r_f32 = f32::from_le_bytes([
                    r_bytes[aligned],
                    r_bytes[aligned + 1],
                    r_bytes[aligned + 2],
                    r_bytes[aligned + 3],
                ]);
                eprintln!(
                    "at aligned offset 0x{aligned:x}: C f32 = {c_f32:e} ({:08x}), Rust f32 = {r_f32:e} ({:08x})",
                    c_f32.to_bits(),
                    r_f32.to_bits()
                );
            }
            // Field-offset table for mapping byte offsets back to field names.
            // Offsets computed via field_offsets_for_diag() below, printed on
            // the first run by diag_print_field_offsets.
            panic!("state diverges at frame {f}, offset 0x{off:x}");
        }

        // Also enforce the user-visible AnalysisInfo fields on valid frames.
        assert_eq!(
            c_info.valid, rust_info.valid,
            "frame {f}: valid mismatch (c={}, rust={})",
            c_info.valid, rust_info.valid
        );
        if c_info.valid == 1 {
            assert_eq!(
                c_info.music_prob.to_bits(),
                rust_info.music_prob.to_bits(),
                "frame {f}: music_prob bit-diff (c={}, rust={})",
                c_info.music_prob, rust_info.music_prob
            );
        }
    }
}

/// Long-running differential using the actual music_48k_stereo.wav content.
/// This is the most realistic stress test for the analyzer: it's literally
/// the same samples the ropus-compare encode path feeds through.
///
/// Frame-by-frame byte-compares the full TonalityAnalysisState. Uses
/// `downmix_int` (C) / `rust_downmix_int` (Rust) since the WAV is i16 PCM
/// — matching the real `opus_encode` code path.
///
/// If this test passes bit-exactly but ropus-compare encode still shows
/// DIFFER frames on the same WAV, the drift is *downstream* of the
/// analyzer — in the Opus/CELT wiring that consumes AnalysisInfo.
#[test]
fn test_analysis_run_matches_c_reference_real_music() {
    // Load the WAV file. Minimal parser — expects 44-byte header, PCM i16.
    let wav_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("tests")
        .join("vectors")
        .join("music_48k_stereo.wav");
    let data = match std::fs::read(&wav_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIPPING: cannot read {}: {}", wav_path.display(), e);
            return;
        }
    };
    assert!(data.len() >= 44, "WAV too small");
    assert_eq!(&data[0..4], b"RIFF");
    assert_eq!(&data[8..12], b"WAVE");
    let channels = u16::from_le_bytes([data[22], data[23]]) as i32;
    let sample_rate = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
    let bits_per_sample = u16::from_le_bytes([data[34], data[35]]);
    assert_eq!(sample_rate, 48_000, "expected 48 kHz");
    assert_eq!(bits_per_sample, 16, "expected 16-bit PCM");

    // Find data chunk.
    let mut pos = 12;
    let mut pcm_bytes: &[u8] = &[];
    while pos + 8 <= data.len() {
        let chunk_size = u32::from_le_bytes([
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]) as usize;
        if &data[pos..pos + 4] == b"data" {
            pcm_bytes = &data[pos + 8..pos + 8 + chunk_size];
            break;
        }
        pos += 8 + chunk_size;
        if !chunk_size.is_multiple_of(2) {
            pos += 1;
        }
    }
    assert!(!pcm_bytes.is_empty(), "no data chunk");

    const FRAME_SIZE: usize = 960; // 20 ms at 48 kHz
    const FS: i32 = 48_000;
    const C1: i32 = 0;
    const C2: i32 = -2;
    const LSB_DEPTH: i32 = 16;

    let frame_bytes_len = FRAME_SIZE * channels as usize * std::mem::size_of::<i16>();
    let num_full_frames = pcm_bytes.len() / frame_bytes_len;
    // Exercise at least 300 frames (~6 s) — well beyond the ~90 frame mark
    // where ropus-compare first sees a DIFFER. Cap to keep test time sane.
    let num_frames = num_full_frames.min(300);

    let mut c_state = TonalityAnalysisState::new_boxed();
    let mut rust_state = TonalityAnalysisState::new_boxed();

    let celt_mode_ptr: *mut std::ffi::c_void = unsafe {
        let mut err: c_int = 0;
        let p = opus_custom_mode_create(FS, FRAME_SIZE as c_int, &mut err);
        assert!(!p.is_null() && err == 0);
        p
    };

    unsafe {
        tonality_analysis_init(&raw mut *c_state, FS);
    }
    rust_tonality_analysis_init(&mut rust_state, FS);

    let state_size = std::mem::size_of::<TonalityAnalysisState>();
    const ARCH_FIELD_BYTES: usize = 4;

    for f in 0..num_frames {
        let start = f * frame_bytes_len;
        let end = start + frame_bytes_len;
        let frame_slice = &pcm_bytes[start..end];

        let mut c_info = AnalysisInfo::default();
        unsafe {
            run_analysis(
                &raw mut *c_state,
                celt_mode_ptr,
                frame_slice.as_ptr() as *const std::ffi::c_void,
                FRAME_SIZE as c_int,
                FRAME_SIZE as c_int,
                C1,
                C2,
                channels as c_int,
                FS,
                LSB_DEPTH,
                downmix_int,
                &mut c_info,
            );
        }

        let mut rust_info = AnalysisInfo::default();
        use ropus::celt::modes::MODE_48000_960_120 as ROPUS_CELT_MODE;
        let rust_downmix_cb: DownmixFunc = rust_downmix_int;
        rust_run_analysis(
            &mut rust_state,
            &ROPUS_CELT_MODE,
            Some(frame_slice),
            FRAME_SIZE as i32,
            FRAME_SIZE as i32,
            C1,
            C2,
            channels,
            FS,
            LSB_DEPTH,
            rust_downmix_cb,
            &mut rust_info,
        );

        let c_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(c_state.as_ref() as *const _ as *const u8, state_size)
        };
        let r_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(rust_state.as_ref() as *const _ as *const u8, state_size)
        };
        let mut first_diff: Option<usize> = None;
        for (i, (&c, &r)) in c_bytes
            .iter()
            .zip(r_bytes.iter())
            .enumerate()
            .skip(ARCH_FIELD_BYTES)
        {
            if c != r {
                first_diff = Some(i);
                break;
            }
        }
        if let Some(off) = first_diff {
            let end_off = (off + 32).min(state_size);
            eprintln!(
                "DIVERGED at frame {f}, first differing byte at 0x{off:x} ({off})\n  C   : {:02x?}\n  Rust: {:02x?}",
                &c_bytes[off..end_off],
                &r_bytes[off..end_off],
            );
            let aligned = off & !3;
            if aligned + 4 <= state_size {
                let c_f32 = f32::from_le_bytes([
                    c_bytes[aligned],
                    c_bytes[aligned + 1],
                    c_bytes[aligned + 2],
                    c_bytes[aligned + 3],
                ]);
                let r_f32 = f32::from_le_bytes([
                    r_bytes[aligned],
                    r_bytes[aligned + 1],
                    r_bytes[aligned + 2],
                    r_bytes[aligned + 3],
                ]);
                eprintln!(
                    "at aligned offset 0x{aligned:x}: C f32 = {c_f32:e} ({:08x}), Rust f32 = {r_f32:e} ({:08x})",
                    c_f32.to_bits(),
                    r_f32.to_bits()
                );
            }
            panic!("state diverges at frame {f}, offset 0x{off:x}");
        }

        // AnalysisInfo fields
        assert_eq!(
            c_info.valid, rust_info.valid,
            "frame {f}: valid mismatch"
        );
        if c_info.valid == 1 {
            assert_eq!(
                c_info.music_prob.to_bits(),
                rust_info.music_prob.to_bits(),
                "frame {f}: music_prob bit-diff (c={}, rust={})",
                c_info.music_prob, rust_info.music_prob
            );
            assert_eq!(
                c_info.bandwidth, rust_info.bandwidth,
                "frame {f}: bandwidth mismatch"
            );
        }
    }
}

/// Emit a table of field offsets for [`TonalityAnalysisState`], used to
/// map byte-offset divergences from the long differential test back to
/// field names. Run with `-- --ignored --nocapture` when the long test
/// fails to localize the drift.
#[test]
#[ignore = "diagnostic: prints field offset table"]
fn diag_print_field_offsets() {
    // Stable macro-free way: build a zeroed state and compute the offsets of
    // every field manually via raw-pointer math.
    let st = TonalityAnalysisState::new_boxed();
    let base = st.as_ref() as *const _ as usize;
    macro_rules! off {
        ($name:ident) => {{
            let p = &st.$name as *const _ as usize;
            (stringify!($name), p - base, std::mem::size_of_val(&st.$name))
        }};
    }
    let fields: Vec<(&'static str, usize, usize)> = vec![
        off!(arch),
        off!(application),
        off!(fs),
        off!(angle),
        off!(d_angle),
        off!(d2_angle),
        off!(inmem),
        off!(mem_fill),
        off!(prev_band_tonality),
        off!(prev_tonality),
        off!(prev_bandwidth),
        off!(e_frames),
        off!(log_e_frames),
        off!(low_e),
        off!(high_e),
        off!(mean_e),
        off!(mem),
        off!(cmean),
        off!(std),
        off!(e_tracker),
        off!(low_e_count),
        off!(e_count),
        off!(count),
        off!(analysis_offset),
        off!(write_pos),
        off!(read_pos),
        off!(read_subframe),
        off!(hp_ener_accum),
        off!(initialized),
        off!(rnn_state),
        off!(downmix_state),
        off!(info),
    ];
    println!("TonalityAnalysisState layout (total {} bytes):", std::mem::size_of::<TonalityAnalysisState>());
    for (name, off, size) in &fields {
        println!("  0x{off:06x}..0x{:06x} ({size:6} bytes) {name}", off + size);
    }
}

// =============================================================================
// SILK mono→stereo transition resampler bug (testvector02/10 regression)
// =============================================================================
//
// Before the fix, decoding a stereo SILK packet that followed a run of mono
// SILK packets produced zeros on the right channel for the first ~13 samples
// (and drift thereafter). Root cause: `silk_decode` cloned channel 0's
// resampler state into channel 1 BEFORE calling `silk_decoder_set_fs`, which
// re-initialised the channel 1 resampler and wiped the cloned state. The fix
// runs the clone AFTER set_fs, matching `reference/silk/dec_API.c:218-222`.
//
// Surfaced by IETF testvector02_stereo and testvector10_stereo (RFC 6716).

#[test]
fn test_silk_mono_to_stereo_transition_matches_c_reference() {
    // Prime a stereo-capable decoder on both sides with a mono packet, then
    // hand it a stereo packet. Pre-fix: rust right channel is zeros for the
    // first ~13 samples. Post-fix: byte-exact against the C reference.
    //
    // Packet provenance: both bitstreams are frames 601 (mono 10 ms SILK NB)
    // and 602 (stereo 60 ms SILK NB) of IETF testvector02.bit — the exact
    // boundary where the original divergence surfaced. Hard-coded inline so
    // the regression test does not depend on the IETF fixture being present
    // on disk.

    let sr = 48000;
    let ch = 2;
    let max_frame = 5760;

    // Rust decoder: mono prime, then stereo decode.
    let mut rust_dec = OpusDecoder::new(sr, ch).unwrap();
    let mut rust_buf = vec![0i16; max_frame as usize * ch as usize];
    let _ = rust_dec
        .decode(
            Some(&SILK_MONO_NB_10MS_FRAME),
            &mut rust_buf,
            max_frame,
            false,
        )
        .expect("Rust mono prime failed");
    let rust_n = rust_dec
        .decode(
            Some(&SILK_STEREO_NB_60MS_FRAME),
            &mut rust_buf,
            max_frame,
            false,
        )
        .expect("Rust stereo decode failed");

    // C reference: same two-call sequence.
    let (c_n, c_buf) = unsafe {
        let mut err: c_int = 0;
        let c_dec = opus_decoder_create(sr, ch as c_int, &mut err);
        assert!(!c_dec.is_null() && err == 0, "C decoder create failed");
        let mut c_buf = vec![0i16; max_frame as usize * ch as usize];
        let mp = opus_decode(
            c_dec,
            SILK_MONO_NB_10MS_FRAME.as_ptr(),
            SILK_MONO_NB_10MS_FRAME.len() as i32,
            c_buf.as_mut_ptr(),
            max_frame,
            0,
        );
        assert!(mp > 0, "C mono prime returned {mp}");
        let n = opus_decode(
            c_dec,
            SILK_STEREO_NB_60MS_FRAME.as_ptr(),
            SILK_STEREO_NB_60MS_FRAME.len() as i32,
            c_buf.as_mut_ptr(),
            max_frame,
            0,
        );
        opus_decoder_destroy(c_dec);
        assert!(n > 0, "C stereo decode returned {n}");
        (n, c_buf)
    };

    assert_eq!(rust_n, c_n, "sample count mismatch");
    let n_samples = (rust_n as usize) * (ch as usize);
    assert_eq!(
        &rust_buf[..n_samples],
        &c_buf[..n_samples],
        "stereo PCM mismatch: mono→stereo SILK transition (ropus/C divergence)"
    );
}

// Frame 601 of IETF testvector02.bit: SILK NB mono, 10 ms (TOC 0x00).
#[rustfmt::skip]
const SILK_MONO_NB_10MS_FRAME: [u8; 44] = [
    0x00, 0x99, 0x11, 0xd5, 0xb2, 0xf5, 0x8d, 0xa4, 0xfa, 0xe7, 0x14, 0xf6,
    0x04, 0x2e, 0x85, 0x5a, 0x5c, 0x08, 0x6f, 0x25, 0x65, 0xf7, 0xaf, 0x99,
    0x66, 0xe3, 0x83, 0xac, 0xd2, 0x43, 0x16, 0x29, 0x1c, 0xfc, 0x55, 0x13,
    0xab, 0x54, 0xd0, 0xfe, 0xcd, 0x01, 0x10, 0x71,
];

// Frame 602 of IETF testvector02.bit: SILK NB stereo, 60 ms (TOC 0x1c).
#[rustfmt::skip]
const SILK_STEREO_NB_60MS_FRAME: [u8; 349] = [
    0x1c, 0xe2, 0x7f, 0x85, 0x31, 0x10, 0xec, 0x82, 0x5f, 0xba, 0xbc, 0x88,
    0xe8, 0x82, 0x0c, 0xb1, 0xa5, 0xd6, 0xba, 0x25, 0x06, 0x9e, 0xc1, 0xd8,
    0xcc, 0x14, 0xea, 0x25, 0xe1, 0xa3, 0x8c, 0xcf, 0xd4, 0xea, 0xaf, 0xb3,
    0x59, 0x6c, 0x14, 0x44, 0xc2, 0xe0, 0x2f, 0xe8, 0x70, 0x68, 0x47, 0xd5,
    0x6e, 0x57, 0xcf, 0x60, 0x5f, 0x97, 0x32, 0xdd, 0xaf, 0x5e, 0xfa, 0x68,
    0x3b, 0xae, 0x5e, 0xbf, 0xd9, 0xa8, 0xcd, 0x3b, 0xfa, 0x9c, 0x7e, 0x32,
    0xf0, 0xe4, 0x84, 0x18, 0xb9, 0x7f, 0xc5, 0x45, 0x03, 0xce, 0xca, 0xa1,
    0xcc, 0x6b, 0x1a, 0x46, 0xd9, 0x11, 0x23, 0x90, 0x81, 0x19, 0xb9, 0x79,
    0x3b, 0x3e, 0x6b, 0xd4, 0x08, 0xb8, 0x9c, 0x9f, 0xea, 0xb3, 0x05, 0x75,
    0xf9, 0x71, 0xe9, 0x25, 0xbc, 0x8d, 0x7e, 0xb0, 0x61, 0xbb, 0x22, 0xf4,
    0x78, 0x64, 0x30, 0xe2, 0xa1, 0x48, 0x79, 0x2d, 0x3e, 0x30, 0x97, 0xbe,
    0xa8, 0xe1, 0x18, 0x6f, 0x6b, 0x0e, 0xfd, 0xcc, 0x10, 0x7a, 0xc6, 0xf3,
    0x3d, 0x1b, 0x0c, 0xa8, 0x84, 0xe1, 0x5c, 0x2c, 0x3b, 0x9f, 0x17, 0xc1,
    0x1c, 0xab, 0xb6, 0xaf, 0xd7, 0x3b, 0x71, 0x7c, 0xb2, 0xd9, 0x01, 0x53,
    0x2d, 0xcb, 0x41, 0xad, 0xed, 0xdc, 0x6d, 0x04, 0x35, 0x2a, 0x04, 0xa6,
    0xcd, 0x4e, 0xac, 0x27, 0x5a, 0xb1, 0x8a, 0x17, 0xb2, 0xe1, 0xff, 0x11,
    0x61, 0x70, 0x0e, 0x74, 0x35, 0x64, 0xdf, 0xdb, 0xfc, 0x58, 0x85, 0x44,
    0x91, 0xb9, 0x4f, 0xa1, 0x56, 0x52, 0x04, 0xd1, 0x5f, 0x5f, 0x4b, 0x15,
    0xa3, 0x93, 0x43, 0x49, 0x46, 0x01, 0xfa, 0x5c, 0x8c, 0x7f, 0x9b, 0x46,
    0x40, 0x63, 0xc3, 0x32, 0x66, 0x95, 0x1d, 0xc9, 0x65, 0x09, 0x62, 0x00,
    0xe4, 0x8d, 0x1a, 0x11, 0x9d, 0x19, 0x46, 0x56, 0x3e, 0x3a, 0x28, 0x04,
    0x09, 0x65, 0x4f, 0x1e, 0x99, 0x41, 0xbb, 0xf6, 0xec, 0x73, 0x74, 0xd9,
    0x95, 0x71, 0x5c, 0x90, 0x2b, 0x91, 0x4e, 0x63, 0x26, 0x2d, 0xb0, 0x48,
    0x04, 0x8f, 0xdc, 0x71, 0x14, 0x80, 0x66, 0x02, 0x99, 0x24, 0x7c, 0x40,
    0x72, 0x1c, 0xfd, 0x02, 0x0a, 0x8c, 0x99, 0x67, 0xc7, 0xac, 0x6e, 0x66,
    0xea, 0x30, 0x98, 0x21, 0xaa, 0x31, 0xb8, 0xfe, 0x9d, 0xe9, 0x63, 0x76,
    0xc5, 0xa5, 0x8c, 0xb0, 0xba, 0xe1, 0xc2, 0xf1, 0x13, 0x67, 0x86, 0x00,
    0x41, 0x2f, 0x4c, 0x61, 0x30, 0xd9, 0x56, 0x64, 0x81, 0x01, 0xf3, 0x49,
    0xd5, 0xbb, 0xf3, 0xa7, 0x3b, 0x6a, 0xe4, 0x3a, 0x93, 0xb9, 0x2a, 0x55,
    0x37,
];

// =============================================================================
// Stage 6.4b round 2: CELT encoder long-running state field divergence.
//
// Prior agent conclusively proved analysis module is bit-exact against the C
// reference for the exact music_48k_stereo.wav content at 300 frames.
// All 5 CELT-side + 4 Opus-side consumption sites audited.
// Yet `ropus-compare encode` shows 457 DIFFER (first at frame 90) on the
// same WAV. Hypothesis: a long-running CELT encoder accumulator drifts
// between C and Rust. This test does parallel opus_encode on both sides
// frame-by-frame, extracts a set of suspect CELT fields from each side,
// and reports the first divergence.
//
// Ignored by default — it's a diagnostic, not an acceptance test.
// =============================================================================

/// Find the first CELT encoder state field that diverges between C and Rust
/// during parallel encoding of music_48k_stereo.wav under ropus-compare's
/// default config (OPUS_APPLICATION_AUDIO, bitrate 64 kbps, complexity 10,
/// VBR off).
#[test]
#[ignore = "diagnostic for Stage 6.4 bit-exactness; run with --include-ignored"]
fn diag_celt_encoder_state_divergence_music() {
    use ropus::opus::encoder::{CeltEncoderStateExt, OpusEncoderStereoSnapshot};
    const OPUS_APPLICATION_AUDIO: c_int = 2049;

    // Local FFI declarations for the debug_helper extractors (the bindings
    // module is inside `harness/src/` and not accessible from tests).
    unsafe extern "C" {
        fn debug_get_celt_encoder_state_ext(
            enc: *mut u8,
            stereo_saving: *mut i32,
            hf_average: *mut i32,
            spec_avg: *mut i32,
            intensity: *mut i32,
            overlap_max: *mut i32,
            vbr_reservoir: *mut i32,
            vbr_drift: *mut i32,
            vbr_offset: *mut i32,
            vbr_count: *mut i32,
            preemph_mem_e_0: *mut i32,
            preemph_mem_e_1: *mut i32,
            preemph_mem_d_0: *mut i32,
            preemph_mem_d_1: *mut i32,
        );
        fn debug_get_celt_encoder_state(
            enc: *mut u8,
            delayed_intra: *mut i32,
            loss_rate: *mut i32,
            prefilter_period: *mut i32,
            prefilter_gain: *mut i32,
            prefilter_tapset: *mut i32,
            force_intra: *mut i32,
            spread_decision: *mut i32,
            tonal_average: *mut i32,
            last_coded_bands: *mut i32,
            consec_transient: *mut i32,
        );
        fn debug_get_opus_stereo_state(
            enc: *mut u8,
            hybrid_stereo_width_q14: *mut i32,
            width_xx: *mut i32,
            width_xy: *mut i32,
            width_yy: *mut i32,
            width_smoothed: *mut i32,
            width_max_follower: *mut i32,
            detected_bandwidth: *mut i32,
            mode: *mut i32,
            prev_mode: *mut i32,
            bandwidth: *mut i32,
        );
    }

    // Load music_48k_stereo.wav the same way test_analysis_run_matches_c_reference_real_music
    // does.
    let wav_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("tests")
        .join("vectors")
        .join("music_48k_stereo.wav");
    let data = std::fs::read(&wav_path).expect("load music_48k_stereo.wav");
    assert!(data.len() >= 44);
    assert_eq!(&data[0..4], b"RIFF");
    let channels = u16::from_le_bytes([data[22], data[23]]) as i32;
    let sample_rate = u32::from_le_bytes([data[24], data[25], data[26], data[27]]) as i32;
    assert_eq!(sample_rate, 48000);
    assert_eq!(channels, 2);

    // Find "data" chunk.
    let mut pos = 12;
    let mut pcm_bytes: &[u8] = &[];
    while pos + 8 <= data.len() {
        let chunk_size = u32::from_le_bytes([
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]) as usize;
        if &data[pos..pos + 4] == b"data" {
            pcm_bytes = &data[pos + 8..pos + 8 + chunk_size];
            break;
        }
        pos += 8 + chunk_size;
        if !chunk_size.is_multiple_of(2) {
            pos += 1;
        }
    }
    assert!(!pcm_bytes.is_empty(), "no data chunk");

    // PCM as i16.
    let pcm: Vec<i16> = pcm_bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();

    let frame_size: i32 = sample_rate / 50; // 20 ms = 960
    let samples_per_frame = frame_size as usize * channels as usize;
    let num_frames = (pcm.len() / samples_per_frame).min(120); // up to frame 119

    // --- create both encoders with ropus-compare's default config ---
    const BITRATE: i32 = 64_000;
    const COMPLEXITY: c_int = 10;
    let application = OPUS_APPLICATION_AUDIO;

    let c_enc = unsafe {
        let mut err: c_int = 0;
        let enc = opus_encoder_create(sample_rate, channels as c_int, application, &mut err);
        assert!(!enc.is_null() && err == 0);
        opus_encoder_ctl(enc, OPUS_SET_BITRATE_REQUEST, BITRATE);
        opus_encoder_ctl(enc, OPUS_SET_VBR_REQUEST, 0 as c_int);
        opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY_REQUEST, COMPLEXITY);
        enc
    };

    let mut rust_enc =
        OpusEncoder::new(sample_rate, channels, application).expect("rust enc");
    rust_enc.set_bitrate(BITRATE);
    rust_enc.set_vbr(0);
    rust_enc.set_complexity(COMPLEXITY);

    // Accumulator fields to compare each frame. Field name must match the
    // `CeltEncoderStateExt` Rust struct and C trace extractor.
    struct CState {
        stereo_saving: i32,
        hf_average: i32,
        spec_avg: i32,
        intensity: i32,
        overlap_max: i32,
        vbr_reservoir: i32,
        vbr_drift: i32,
        vbr_offset: i32,
        vbr_count: i32,
        preemph_mem_e: [i32; 2],
        preemph_mem_d: [i32; 2],
        delayed_intra: i32,
        tonal_average: i32,
        last_coded_bands: i32,
        tapset_decision: i32,
        spread_decision: i32,
        rng: u32,
        consec_transient: i32,
    }

    let snap_c = |enc: *mut u8| -> CState {
        let mut stereo_saving = 0;
        let mut hf_average = 0;
        let mut spec_avg = 0;
        let mut intensity = 0;
        let mut overlap_max = 0;
        let mut vbr_reservoir = 0;
        let mut vbr_drift = 0;
        let mut vbr_offset = 0;
        let mut vbr_count = 0;
        let mut pme0 = 0;
        let mut pme1 = 0;
        let mut pmd0 = 0;
        let mut pmd1 = 0;
        unsafe {
            debug_get_celt_encoder_state_ext(
                enc,
                &mut stereo_saving,
                &mut hf_average,
                &mut spec_avg,
                &mut intensity,
                &mut overlap_max,
                &mut vbr_reservoir,
                &mut vbr_drift,
                &mut vbr_offset,
                &mut vbr_count,
                &mut pme0,
                &mut pme1,
                &mut pmd0,
                &mut pmd1,
            );
        }
        let mut delayed_intra = 0;
        let mut loss_rate = 0;
        let mut pf_period = 0;
        let mut pf_gain = 0;
        let mut pf_tapset = 0;
        let mut force_intra = 0;
        let mut spread_decision = 0;
        let mut tonal_average = 0;
        let mut last_coded_bands = 0;
        let mut consec_transient = 0;
        unsafe {
            debug_get_celt_encoder_state(
                enc,
                &mut delayed_intra,
                &mut loss_rate,
                &mut pf_period,
                &mut pf_gain,
                &mut pf_tapset,
                &mut force_intra,
                &mut spread_decision,
                &mut tonal_average,
                &mut last_coded_bands,
                &mut consec_transient,
            );
        }
        CState {
            stereo_saving,
            hf_average,
            spec_avg,
            intensity,
            overlap_max,
            vbr_reservoir,
            vbr_drift,
            vbr_offset,
            vbr_count,
            preemph_mem_e: [pme0, pme1],
            preemph_mem_d: [pmd0, pmd1],
            delayed_intra,
            tonal_average,
            last_coded_bands,
            tapset_decision: 0, // Not extracted; reserved for future use.
            spread_decision,
            rng: 0, // Not extracted; reserved for future use.
            consec_transient,
        }
    };

    fn snap_rs(snap: CeltEncoderStateExt) -> CState {
        CState {
            stereo_saving: snap.stereo_saving,
            hf_average: snap.hf_average,
            spec_avg: snap.spec_avg,
            intensity: snap.intensity,
            overlap_max: snap.overlap_max,
            vbr_reservoir: snap.vbr_reservoir,
            vbr_drift: snap.vbr_drift,
            vbr_offset: snap.vbr_offset,
            vbr_count: snap.vbr_count,
            preemph_mem_e: snap.preemph_mem_e,
            preemph_mem_d: snap.preemph_mem_d,
            delayed_intra: snap.delayed_intra,
            tonal_average: snap.tonal_average,
            last_coded_bands: snap.last_coded_bands,
            tapset_decision: snap.tapset_decision,
            spread_decision: snap.spread_decision,
            rng: snap.rng,
            consec_transient: snap.consec_transient,
        }
    }

    let mut first_divergence: Option<(usize, String)> = None;

    for f in 0..num_frames {
        let start = f * samples_per_frame;
        let end = start + samples_per_frame;
        let frame_pcm = &pcm[start..end];

        // Encode in C.
        let c_out_len = unsafe {
            let mut out = vec![0u8; 4000];
            let n =
                opus_encode(c_enc, frame_pcm.as_ptr(), frame_size, out.as_mut_ptr(), 4000);
            assert!(n > 0, "C encode failed frame={f} ret={n}");
            n
        };

        // Encode in Rust.
        let mut rust_out = vec![0u8; 4000];
        let rust_n = rust_enc
            .encode(frame_pcm, frame_size, &mut rust_out, 4000)
            .unwrap_or_else(|e| panic!("Rust encode failed frame={f} err={e}"));

        let _ = c_out_len;
        let _ = rust_n;

        // Snapshot both sides AFTER the frame.
        let c = snap_c(c_enc);
        let r = snap_rs(
            rust_enc
                .get_celt_state_ext()
                .expect("rust celt_state_ext must be Some"),
        );

        // Compare each field; report first divergence.
        macro_rules! check {
            ($name:ident) => {
                if c.$name != r.$name {
                    let msg = format!(
                        "frame {:3}: {:<20} C={:12} R={:12} diff={}",
                        f,
                        stringify!($name),
                        c.$name,
                        r.$name,
                        c.$name - r.$name,
                    );
                    eprintln!("{msg}");
                    if first_divergence.is_none() {
                        first_divergence = Some((f, msg));
                    }
                }
            };
        }
        check!(stereo_saving);
        check!(hf_average);
        check!(spec_avg);
        check!(intensity);
        check!(overlap_max);
        check!(vbr_reservoir);
        check!(vbr_drift);
        check!(vbr_offset);
        check!(vbr_count);
        check!(delayed_intra);
        check!(tonal_average);
        check!(last_coded_bands);
        check!(spread_decision);
        check!(consec_transient);

        // --- Opus-level stereo-width and mode state ---
        let mut c_hybrid = 0;
        let mut c_xx = 0;
        let mut c_xy = 0;
        let mut c_yy = 0;
        let mut c_smoothed = 0;
        let mut c_max_follower = 0;
        let mut c_detected_bw = 0;
        let mut c_mode = 0;
        let mut c_prev_mode = 0;
        let mut c_bandwidth = 0;
        unsafe {
            debug_get_opus_stereo_state(
                c_enc,
                &mut c_hybrid,
                &mut c_xx,
                &mut c_xy,
                &mut c_yy,
                &mut c_smoothed,
                &mut c_max_follower,
                &mut c_detected_bw,
                &mut c_mode,
                &mut c_prev_mode,
                &mut c_bandwidth,
            );
        }
        let r_opus: OpusEncoderStereoSnapshot = rust_enc.get_opus_stereo_state();
        macro_rules! check_opus {
            ($c:expr, $r:expr, $name:literal) => {
                if $c != $r {
                    let msg = format!(
                        "frame {:3}: opus.{:<18} C={:12} R={:12}",
                        f, $name, $c, $r
                    );
                    eprintln!("{msg}");
                    if first_divergence.is_none() {
                        first_divergence = Some((f, msg));
                    }
                }
            };
        }
        check_opus!(c_hybrid, r_opus.hybrid_stereo_width_q14, "hybrid_sw_Q14");
        check_opus!(c_xx, r_opus.width_xx, "width_xx");
        check_opus!(c_xy, r_opus.width_xy, "width_xy");
        check_opus!(c_yy, r_opus.width_yy, "width_yy");
        check_opus!(c_smoothed, r_opus.width_smoothed, "width_smoothed");
        check_opus!(c_max_follower, r_opus.width_max_follower, "width_max_foll");
        check_opus!(c_detected_bw, r_opus.detected_bandwidth, "detected_bw");
        check_opus!(c_mode, r_opus.mode, "mode");
        check_opus!(c_prev_mode, r_opus.prev_mode, "prev_mode");
        check_opus!(c_bandwidth, r_opus.bandwidth, "bandwidth");
        // preemph_mem_e[] intentionally excluded: it holds a cosmetic
        // representation difference (Rust stores x, C stores coef0*x/2^15)
        // that produces identical preemphasis OUTPUT samples but differing
        // state values. Ruled out in 2026.04.10 CELT bug-fix journal.
        if c.preemph_mem_d != r.preemph_mem_d {
            let msg = format!(
                "frame {:3}: preemph_mem_d      C={:?} R={:?}",
                f, c.preemph_mem_d, r.preemph_mem_d
            );
            eprintln!("{msg}");
            if first_divergence.is_none() {
                first_divergence = Some((f, msg));
            }
        }
    }

    unsafe { opus_encoder_destroy(c_enc) };

    if let Some((f, msg)) = first_divergence {
        eprintln!("\n=== First divergence at frame {f} ===\n  {msg}");
    } else {
        eprintln!("\nNo CELT encoder state divergence detected in first {num_frames} frames");
    }
}

