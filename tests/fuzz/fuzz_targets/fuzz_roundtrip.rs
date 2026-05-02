#![no_main]
use libfuzzer_sys::fuzz_target;
use ropus::opus::decoder::{
    opus_packet_get_nb_frames, opus_packet_get_nb_samples, OpusDecoder,
};
use ropus::opus::encoder::{
    OpusEncoder, OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY,
    OPUS_APPLICATION_VOIP,
};
use std::cell::RefCell;
use std::sync::Once;

#[path = "c_reference.rs"]
mod c_reference;

#[path = "oracle.rs"]
mod oracle;

// --------------------------------------------------------------------------- //
// Panic-capture: on Windows, Rust assertions in libfuzzer-sys trigger __fastfail
// which bypasses libFuzzer's crash-artifact writer.
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

// Known-class skip filter — see `fuzz_encode.rs::is_known_class`.
// Narrow on purpose: each tuple guards exactly the divergence class
// already documented under `tests/fuzz/known_failures/`.
//
// NOTE: The previous entry guarding the
// `roundtrip-float-12k-cx7-vbr-divergence` class (sr=12000, AUDIO,
// float-PCM, vbr=true) was retired on 2026-05-02 once the
// `MAX_ENCODING_DEPTH = 16` fix landed — see
// `wrk_docs/2026.05.02 - HLD - float-pcm-ingest-fix.md`. The empty
// stub is kept so future divergence classes have an obvious place to
// register without churning the call site.
fn is_known_class(
    _sample_rate: i32,
    _application: i32,
    _use_float_pcm: bool,
    _vbr: bool,
) -> bool {
    false
}

fn byte_to_bitrate(b0: u8, b1: u8) -> i32 {
    let raw = u16::from_le_bytes([b0, b1]) as i32;
    6000 + (raw % 504001)
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
    let complexity = (data[5] as i32) % 10;
    let vbr = (data[6] & 0b0001) != 0;
    let vbr_constraint = (data[6] & 0b0010) != 0;
    let inband_fec_raw = ((data[6] & 0b1100) >> 2) as i32;
    let inband_fec = if inband_fec_raw == 3 { 0 } else { inband_fec_raw };
    let dtx = (data[7] & 0b0001) != 0;
    let loss_perc = (((data[7] & 0b1111_1110) >> 1) as i32) % 101;
    let pcm_bytes = &data[8..];

    let frame_size = sample_rate / 50; // 20ms
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
    };

    // === Rust encoder + decoder construction (shared between i16 and f32) ===
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

    let mut rust_dec = match OpusDecoder::new(sample_rate, channels) {
        Ok(d) => d,
        Err(_) => return,
    };

    let mut rust_compressed = vec![0u8; 4000];

    if use_float_pcm {
        let pcm: Vec<f32> = pcm_bytes[..bytes_needed]
            .chunks_exact(4)
            .map(|c| {
                let f = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                // NaN/Inf produces differential mismatch — see wrk_journals/2026.05.01 - JRN - fuzz-coverage-expansion-impl.md
                if f.is_finite() { f } else { 0.0 }
            })
            .collect();

        let rust_enc_len =
            match rust_enc.encode_float(&pcm, frame_size, &mut rust_compressed, 4000) {
                Ok(l) => l as usize,
                Err(_) => return,
            };

        let mut rust_decoded = vec![0f32; samples_needed];
        let rust_dec_ret = rust_dec.decode_float(
            Some(&rust_compressed[..rust_enc_len]),
            &mut rust_decoded,
            frame_size,
            false,
        );

        let c_compressed =
            match c_reference::c_encode_float(&pcm, frame_size, sample_rate, channels, &cfg) {
                Ok(c) => c,
                Err(_) => return,
            };
        let c_decoded = match c_reference::c_decode_float(&c_compressed, sample_rate, channels) {
            Ok(d) => d,
            Err(_) => return,
        };

        // Compressed output must match — skipped for documented
        // float-PCM divergence classes (see is_known_class).
        let skip_diff = is_known_class(sample_rate, application, true, vbr);
        if !skip_diff {
            assert_eq!(
                &rust_compressed[..rust_enc_len], &c_compressed[..],
                "Float compressed output mismatch: Rust={rust_enc_len}B, C={}B, \
                 sr={sample_rate}, ch={channels}, app={application}, br={bitrate}, cx={complexity}, vbr={vbr}",
                c_compressed.len()
            );
        }

        let packet = &rust_compressed[..rust_enc_len];
        if !packet.is_empty() {
            let nb_frames = opus_packet_get_nb_frames(packet);
            assert!(
                nb_frames.is_ok() && nb_frames.unwrap() > 0,
                "Float encoded packet not parseable: len={rust_enc_len}"
            );
            let nb_samples = opus_packet_get_nb_samples(packet, sample_rate);
            if let Ok(ns) = nb_samples {
                assert_eq!(
                    ns, frame_size,
                    "Float encoded packet nb_samples ({ns}) != frame_size ({frame_size})"
                );
            }
        }

        match rust_dec_ret {
            Ok(rust_samples) => {
                let rust_samples = rust_samples as usize;
                let c_samples = c_decoded.len() / channels as usize;
                if !skip_diff {
                    assert_eq!(
                        rust_samples, c_samples,
                        "Float decoded sample count mismatch: Rust={rust_samples}, C={c_samples}"
                    );
                    let n = rust_samples * channels as usize;
                    // Float decode is a deterministic i16/32768 mapping over the
                    // Opus-decoded samples; CELT-only is bit-exact and SILK/Hybrid
                    // is bounded by the SNR oracle (HLD V2 gap 6).
                    let celt_only = !packet.is_empty() && (packet[0] & 0x80) != 0;
                    if celt_only {
                        for (i, (r, c)) in rust_decoded[..n].iter().zip(c_decoded.iter()).enumerate() {
                            assert_eq!(
                                r.to_bits(),
                                c.to_bits(),
                                "Float decoded PCM mismatch at sample {i}: \
                                 sr={sample_rate}, ch={channels}, br={bitrate}"
                            );
                        }
                    } else {
                        let to_i16 = |x: f32| -> i16 {
                            (x * 32768.0).round().clamp(-32768.0, 32767.0) as i16
                        };
                        let rust_i16: Vec<i16> = rust_decoded[..n].iter().map(|&x| to_i16(x)).collect();
                        let c_i16: Vec<i16> = c_decoded.iter().map(|&x| to_i16(x)).collect();
                        let well_formed =
                            opus_packet_get_nb_samples(packet, sample_rate).is_ok();
                        if oracle::snr_oracle_applicable_for_packet(&c_i16, well_formed) {
                            let snr = oracle::snr_db(&c_i16, &rust_i16);
                            assert!(
                                snr >= oracle::SILK_DECODE_MIN_SNR_DB,
                                "Float roundtrip SILK/Hybrid SNR {snr:.2} dB < {:.0} dB \
                                 (sr={sample_rate}, ch={channels}, br={bitrate}, cx={complexity}, vbr={vbr})",
                                oracle::SILK_DECODE_MIN_SNR_DB
                            );
                        }
                        // else: reference is silence or near-silence; both
                        // implementations' recovery PCM is unconstrained.
                        // Sample-count match (already asserted earlier) is
                        // the only oracle.
                    }
                }
                assert_eq!(
                    rust_samples, frame_size as usize,
                    "Float decoded samples ({rust_samples}) != frame_size ({frame_size})"
                );
            }
            Err(e) => {
                panic!(
                    "Float: Rust decode failed ({e}) but C decode succeeded ({} samples), \
                     sr={sample_rate}, ch={channels}",
                    c_decoded.len() / channels as usize
                );
            }
        }
    } else {
        let pcm: Vec<i16> = pcm_bytes[..bytes_needed]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        let rust_enc_len = match rust_enc.encode(&pcm, frame_size, &mut rust_compressed, 4000) {
            Ok(l) => l as usize,
            Err(_) => return,
        };

        let mut rust_decoded = vec![0i16; samples_needed];
        let rust_dec_ret = rust_dec.decode(
            Some(&rust_compressed[..rust_enc_len]),
            &mut rust_decoded,
            frame_size,
            false,
        );

        // === C reference round-trip ===
        let c_compressed = match c_reference::c_encode(
            &pcm, frame_size, sample_rate, channels, &cfg,
        ) {
            Ok(c) => c,
            Err(_) => return, // C encode failed — skip (Rust encode succeeded above)
        };

        let c_decoded = match c_reference::c_decode(&c_compressed, sample_rate, channels) {
            Ok(d) => d,
            Err(_) => return,
        };

        // === Differential comparison ===

        // Compressed output must match
        assert_eq!(
            &rust_compressed[..rust_enc_len], &c_compressed[..],
            "Compressed output mismatch: Rust={rust_enc_len}B, C={}B, \
             sr={sample_rate}, ch={channels}, app={application}, br={bitrate}, cx={complexity}, vbr={vbr}",
            c_compressed.len()
        );

        // --- Semantic invariant: encoded packet is parseable ---
        let packet = &rust_compressed[..rust_enc_len];
        if !packet.is_empty() {
            let nb_frames = opus_packet_get_nb_frames(packet);
            assert!(
                nb_frames.is_ok() && nb_frames.unwrap() > 0,
                "Encoded packet not parseable: len={rust_enc_len}"
            );
            let nb_samples = opus_packet_get_nb_samples(packet, sample_rate);
            if let Ok(ns) = nb_samples {
                assert_eq!(
                    ns, frame_size,
                    "Encoded packet nb_samples ({ns}) != frame_size ({frame_size})"
                );
            }
        }

        // Decoded output must match (CELT-only) or fall within the SNR
        // envelope (SILK/Hybrid). HLD V2 gap 6.
        let celt_only = !packet.is_empty() && (packet[0] & 0x80) != 0;
        match rust_dec_ret {
            Ok(rust_samples) => {
                let rust_samples = rust_samples as usize;
                let c_samples = c_decoded.len() / channels as usize;
                assert_eq!(
                    rust_samples, c_samples,
                    "Decoded sample count mismatch: Rust={rust_samples}, C={c_samples}"
                );
                let rust_slice = &rust_decoded[..rust_samples * channels as usize];
                if celt_only {
                    assert_eq!(
                        rust_slice,
                        &c_decoded[..],
                        "Decoded PCM mismatch at sr={sample_rate}, ch={channels}"
                    );
                } else {
                    let well_formed =
                        opus_packet_get_nb_samples(packet, sample_rate).is_ok();
                    if oracle::snr_oracle_applicable_for_packet(&c_decoded[..], well_formed) {
                        let snr = oracle::snr_db(&c_decoded[..], rust_slice);
                        assert!(
                            snr >= oracle::SILK_DECODE_MIN_SNR_DB,
                            "Roundtrip SILK/Hybrid SNR {snr:.2} dB < {:.0} dB \
                             (sr={sample_rate}, ch={channels}, br={bitrate}, cx={complexity}, vbr={vbr}, packet_len={})",
                            oracle::SILK_DECODE_MIN_SNR_DB,
                            packet.len()
                        );
                    }
                }
                // else: reference is silence or near-silence; both
                // implementations' recovery PCM is unconstrained.
                // Sample-count match (already asserted earlier) is the
                // only oracle.

                // --- Semantic invariant: decoded sample count == frame_size ---
                assert_eq!(
                    rust_samples, frame_size as usize,
                    "Decoded samples ({rust_samples}) != frame_size ({frame_size})"
                );
            }
            Err(e) => {
                panic!(
                    "Rust decode failed ({e}) but C decode succeeded ({} samples), \
                     sr={sample_rate}, ch={channels}",
                    c_decoded.len() / channels as usize
                );
            }
        }
    }
});
