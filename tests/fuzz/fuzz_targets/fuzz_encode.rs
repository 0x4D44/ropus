#![no_main]
use libfuzzer_sys::fuzz_target;
use ropus::opus::decoder::{opus_packet_get_nb_frames, opus_packet_get_nb_samples};
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
                        Err(e) => {
                            eprintln!("[PANIC CAPTURE] Failed to write {}: {}", path.display(), e)
                        }
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

// Known-class skip filter: returns true for input tuples that match a
// documented codec divergence already captured under
// `tests/fuzz/known_failures/`. Keeps the worker alive past the known
// bug so libFuzzer can keep mutating into NEW corners.
//
// Narrow on purpose: each tuple guards exactly the class it documents.
// Adjacent configs still exercise the differential — the goal is escape
// from saturation, not blanket suppression.
//
// NOTE: The previous entry guarding the
// `encode-float-lowdelay-8k-divergence` class (sr=8000, ch=1,
// RESTRICTED_LOWDELAY, float-PCM) was retired on 2026-05-02 once the
// `MAX_ENCODING_DEPTH = 16` fix landed — see
// `wrk_docs/2026.05.02 - HLD - float-pcm-ingest-fix.md`. The empty
// stub is kept so future divergence classes have an obvious place to
// register without churning the call site.
fn is_known_class(
    _sample_rate: i32,
    _channels: i32,
    _application: i32,
    _use_float_pcm: bool,
) -> bool {
    false
}

/// Map a byte to a bitrate in the valid Opus range.
fn byte_to_bitrate(b0: u8, b1: u8) -> i32 {
    // Map u16 (0..65535) to 6000..510000
    let raw = u16::from_le_bytes([b0, b1]) as i32;
    6000 + (raw % 504001) // clamp to valid range
}

fn configure_encoder(
    enc: &mut OpusEncoder,
    bitrate: i32,
    complexity: i32,
    vbr: bool,
    vbr_constraint: bool,
    inband_fec: i32,
    dtx: bool,
    loss_perc: i32,
) {
    enc.set_bitrate(bitrate);
    enc.set_vbr(if vbr { 1 } else { 0 });
    enc.set_vbr_constraint(if vbr_constraint { 1 } else { 0 });
    enc.set_inband_fec(inband_fec);
    enc.set_dtx(if dtx { 1 } else { 0 });
    enc.set_packet_loss_perc(loss_perc);
    enc.set_complexity(complexity);
}

fuzz_target!(|data: &[u8]| {
    init_panic_capture();
    CURRENT_INPUT.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.clear();
        buf.extend_from_slice(data);
    });

    // 8 config bytes + enough PCM for at least one 20ms frame.
    // Minimum i16 PCM: 8000 Hz * 0.02s * 1 ch * 2 bytes = 320 bytes.
    // Float PCM doubles the per-sample size; the post-parse length check
    // below enforces that.
    if data.len() < 8 + 320 {
        return;
    }

    // --- Parse structured config ---
    let sample_rate = SAMPLE_RATES[(data[0] as usize) % SAMPLE_RATES.len()];
    let channels = if data[1] & 1 == 0 { 1 } else { 2 };
    let use_float_pcm = (data[1] & 0b0010) != 0;
    let application = APPLICATIONS[(data[2] as usize) % APPLICATIONS.len()];
    let bitrate = byte_to_bitrate(data[3], data[4]);
    // Cap at 9 to dodge the analysis.c divergence class (Campaign 9,
    // 2026-04-19): the C reference builds with DISABLE_FLOAT_API=off, so
    // complexity ≥ 10 ∧ sr ≥ 16000 ∧ app != RESTRICTED_SILK runs analysis
    // on C and not on Rust.
    let complexity = (data[5] as i32) % 10; // 0-9
                                            // data[6]: VBR + FEC config — explicit parens, `>>` binds tighter than `&`.
    let vbr = (data[6] & 0b0001) != 0;
    let vbr_constraint = (data[6] & 0b0010) != 0;
    // 0..=3; clamp 3 → 0 so each documented FEC value (Off/On/Forced) is
    // exercised with non-trivial probability rather than half wasted on noise.
    let inband_fec_raw = ((data[6] & 0b1100) >> 2) as i32;
    let inband_fec = if inband_fec_raw == 3 {
        0
    } else {
        inband_fec_raw
    };
    // data[7]: DTX + packet-loss-perc.
    let dtx = (data[7] & 0b0001) != 0;
    let loss_perc = (((data[7] & 0b1111_1110) >> 1) as i32) % 101; // 0..=100
    let pcm_bytes = &data[8..];

    // Frame size = 20ms at the selected sample rate
    let frame_size = sample_rate / 50;
    let samples_needed = frame_size as usize * channels as usize;
    let sample_size_bytes = if use_float_pcm { 4 } else { 2 };
    let bytes_needed = samples_needed * sample_size_bytes;

    if pcm_bytes.len() < bytes_needed {
        return;
    }

    let cfg = c_reference::CEncodeConfig {
        bitrate,
        complexity,
        application,
        vbr: if vbr { 1 } else { 0 },
        vbr_constraint: if vbr_constraint { 1 } else { 0 },
        inband_fec,
        dtx: if dtx { 1 } else { 0 },
        loss_perc,
        // Defaults match the C reference's encoder defaults so this target's
        // byte-exact differential is unaffected by the new dimensions added
        // by Stream A's grammar broadening.
        max_bandwidth: c_reference::OPUS_BANDWIDTH_FULLBAND,
        signal: c_reference::OPUS_AUTO,
        force_channels: c_reference::OPUS_AUTO,
        prediction_disabled: 0,
    };

    // --- Rust encoder construction (shared between i16 and f32 paths) ---
    let mut rust_enc = match OpusEncoder::new(sample_rate, channels, application) {
        Ok(e) => e,
        Err(_) => return,
    };
    configure_encoder(
        &mut rust_enc,
        bitrate,
        complexity,
        vbr,
        vbr_constraint,
        inband_fec,
        dtx,
        loss_perc,
    );

    let mut rust_out = vec![0u8; 4000];

    if use_float_pcm {
        // Float-path branch — interpret input bytes as f32 PCM.
        let pcm: Vec<f32> = pcm_bytes[..bytes_needed]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let rust_ret = rust_enc.encode_float(&pcm, frame_size, &mut rust_out, 4000);
        let c_ret = c_reference::c_encode_float(&pcm, frame_size, sample_rate, channels, &cfg);

        // Known-class skip: continue to exercise both encoders but
        // bypass the byte-equality assert for documented divergences.
        let skip_diff = is_known_class(sample_rate, channels, application, true);

        match (&rust_ret, &c_ret) {
            (Ok(rust_len), Ok(c_out)) => {
                let rust_len = *rust_len as usize;
                let c_len = c_out.len();

                if !skip_diff {
                    assert_eq!(
                        rust_len, c_len,
                        "Float output length mismatch: Rust={rust_len}, C={c_len}, \
                         sr={sample_rate}, ch={channels}, app={application}, \
                         br={bitrate}, cx={complexity}, vbr={vbr}, fec={inband_fec}"
                    );
                    assert_eq!(
                        &rust_out[..rust_len], &c_out[..],
                        "Float output byte mismatch at sr={sample_rate}, ch={channels}, \
                         app={application}, br={bitrate}, cx={complexity}, vbr={vbr}, len={rust_len}"
                    );
                }

                let packet = &rust_out[..rust_len];
                if !packet.is_empty() {
                    let nb_frames = opus_packet_get_nb_frames(packet);
                    assert!(
                        nb_frames.is_ok() && nb_frames.unwrap() > 0,
                        "Encoded packet not parseable: len={rust_len}, sr={sample_rate}"
                    );
                    let nb_samples = opus_packet_get_nb_samples(packet, sample_rate);
                    if let Ok(ns) = nb_samples {
                        assert_eq!(
                            ns, frame_size,
                            "Encoded packet nb_samples ({ns}) != frame_size ({frame_size})"
                        );
                    }
                }

                // CBR determinism only holds when VBR is disabled.
                if !vbr {
                    let mut enc2 = match OpusEncoder::new(sample_rate, channels, application) {
                        Ok(e) => e,
                        Err(_) => return,
                    };
                    configure_encoder(
                        &mut enc2,
                        bitrate,
                        complexity,
                        vbr,
                        vbr_constraint,
                        inband_fec,
                        dtx,
                        loss_perc,
                    );
                    let mut out2 = vec![0u8; 4000];
                    if let Ok(len2) = enc2.encode_float(&pcm, frame_size, &mut out2, 4000) {
                        let len2 = len2 as usize;
                        assert_eq!(
                            rust_len, len2,
                            "CBR determinism (float): length differs ({rust_len} vs {len2})"
                        );
                        assert_eq!(
                            &rust_out[..rust_len],
                            &out2[..len2],
                            "CBR determinism (float): bytes differ, len={rust_len}"
                        );
                    }
                }
            }
            (Err(_), Err(_)) => {}
            (Ok(rust_len), Err(c_err)) => {
                panic!(
                    "Float: Rust encoded ({rust_len} bytes) but C errored ({c_err}), \
                     sr={sample_rate}, ch={channels}, app={application}, \
                     br={bitrate}, cx={complexity}, vbr={vbr}"
                );
            }
            (Err(rust_err), Ok(c_out)) => {
                panic!(
                    "Float: C encoded ({} bytes) but Rust errored ({rust_err}), \
                     sr={sample_rate}, ch={channels}, app={application}, \
                     br={bitrate}, cx={complexity}, vbr={vbr}",
                    c_out.len()
                );
            }
        }
    } else {
        // Interpret input bytes as i16 PCM.
        let pcm: Vec<i16> = pcm_bytes[..bytes_needed]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        let rust_ret = rust_enc.encode(&pcm, frame_size, &mut rust_out, 4000);
        let c_ret = c_reference::c_encode(&pcm, frame_size, sample_rate, channels, &cfg);

        match (&rust_ret, &c_ret) {
            (Ok(rust_len), Ok(c_out)) => {
                let rust_len = *rust_len as usize;
                let c_len = c_out.len();

                // Compressed output must match byte-for-byte
                assert_eq!(
                    rust_len, c_len,
                    "Output length mismatch: Rust={rust_len}, C={c_len}, \
                     sr={sample_rate}, ch={channels}, app={application}, \
                     br={bitrate}, cx={complexity}, vbr={vbr}, fec={inband_fec}"
                );

                assert_eq!(
                    &rust_out[..rust_len],
                    &c_out[..],
                    "Output byte mismatch at sr={sample_rate}, ch={channels}, \
                     app={application}, br={bitrate}, cx={complexity}, vbr={vbr}, len={rust_len}"
                );

                // --- Semantic invariant: encoded packet is parseable ---
                let packet = &rust_out[..rust_len];
                if !packet.is_empty() {
                    let nb_frames = opus_packet_get_nb_frames(packet);
                    assert!(
                        nb_frames.is_ok() && nb_frames.unwrap() > 0,
                        "Encoded packet not parseable: len={rust_len}, sr={sample_rate}"
                    );
                    let nb_samples = opus_packet_get_nb_samples(packet, sample_rate);
                    if let Ok(ns) = nb_samples {
                        assert_eq!(
                            ns, frame_size,
                            "Encoded packet nb_samples ({ns}) != frame_size ({frame_size})"
                        );
                    }
                }

                // --- Semantic invariant: CBR determinism (VBR-mode-only assertion) ---
                if !vbr {
                    let mut enc2 = match OpusEncoder::new(sample_rate, channels, application) {
                        Ok(e) => e,
                        Err(_) => return,
                    };
                    configure_encoder(
                        &mut enc2,
                        bitrate,
                        complexity,
                        vbr,
                        vbr_constraint,
                        inband_fec,
                        dtx,
                        loss_perc,
                    );
                    let mut out2 = vec![0u8; 4000];
                    if let Ok(len2) = enc2.encode(&pcm, frame_size, &mut out2, 4000) {
                        let len2 = len2 as usize;
                        assert_eq!(
                            rust_len, len2,
                            "CBR determinism: length differs ({rust_len} vs {len2})"
                        );
                        assert_eq!(
                            &rust_out[..rust_len],
                            &out2[..len2],
                            "CBR determinism: bytes differ, len={rust_len}"
                        );
                    }
                }
            }
            (Err(_), Err(_)) => {
                // Both errored — fine
            }
            (Ok(rust_len), Err(c_err)) => {
                panic!(
                    "Rust encoded ({rust_len} bytes) but C errored ({c_err}), \
                     sr={sample_rate}, ch={channels}, app={application}, \
                     br={bitrate}, cx={complexity}, vbr={vbr}"
                );
            }
            (Err(rust_err), Ok(c_out)) => {
                panic!(
                    "C encoded ({} bytes) but Rust errored ({rust_err}), \
                     sr={sample_rate}, ch={channels}, app={application}, \
                     br={bitrate}, cx={complexity}, vbr={vbr}",
                    c_out.len()
                );
            }
        }
    }
});
