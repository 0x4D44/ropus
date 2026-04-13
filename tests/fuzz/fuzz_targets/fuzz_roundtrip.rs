#![no_main]
use libfuzzer_sys::fuzz_target;
use mdopus::opus::decoder::{
    opus_packet_get_nb_frames, opus_packet_get_nb_samples, OpusDecoder,
};
use mdopus::opus::encoder::{
    OpusEncoder, OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY,
    OPUS_APPLICATION_VOIP,
};
use std::cell::RefCell;
use std::sync::Once;

#[path = "c_reference.rs"]
mod c_reference;

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

fn byte_to_bitrate(b0: u8, b1: u8) -> i32 {
    let raw = u16::from_le_bytes([b0, b1]) as i32;
    6000 + (raw % 504001)
}

fuzz_target!(|data: &[u8]| {
    init_panic_capture();
    CURRENT_INPUT.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.clear();
        buf.extend_from_slice(data);
    });

    if data.len() < 6 + 320 {
        return;
    }

    // --- Parse structured config ---
    let sample_rate = SAMPLE_RATES[(data[0] as usize) % SAMPLE_RATES.len()];
    let channels = if data[1] & 1 == 0 { 1 } else { 2 };
    let application = APPLICATIONS[(data[2] as usize) % APPLICATIONS.len()];
    let bitrate = byte_to_bitrate(data[3], data[4]);
    let complexity = (data[5] as i32) % 11;
    let pcm_bytes = &data[6..];

    let frame_size = sample_rate / 50; // 20ms
    let samples_needed = frame_size as usize * channels as usize;
    let bytes_needed = samples_needed * 2;

    if pcm_bytes.len() < bytes_needed {
        return;
    }

    let pcm: Vec<i16> = pcm_bytes[..bytes_needed]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    // === Rust round-trip ===
    let mut rust_enc = match OpusEncoder::new(sample_rate, channels, application) {
        Ok(e) => e,
        Err(_) => return,
    };
    rust_enc.set_bitrate(bitrate);
    rust_enc.set_vbr(0);
    rust_enc.set_complexity(complexity);

    let mut rust_compressed = vec![0u8; 4000];
    let rust_enc_len = match rust_enc.encode(&pcm, frame_size, &mut rust_compressed, 4000) {
        Ok(l) => l as usize,
        Err(_) => return,
    };

    let mut rust_dec = match OpusDecoder::new(sample_rate, channels) {
        Ok(d) => d,
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
        &pcm, frame_size, sample_rate, channels, bitrate, complexity, application,
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
         sr={sample_rate}, ch={channels}, app={application}, br={bitrate}, cx={complexity}",
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

    // Decoded output must match
    match rust_dec_ret {
        Ok(rust_samples) => {
            let rust_samples = rust_samples as usize;
            let c_samples = c_decoded.len() / channels as usize;
            assert_eq!(
                rust_samples, c_samples,
                "Decoded sample count mismatch: Rust={rust_samples}, C={c_samples}"
            );
            assert_eq!(
                &rust_decoded[..rust_samples * channels as usize],
                &c_decoded[..],
                "Decoded PCM mismatch at sr={sample_rate}, ch={channels}"
            );

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
});
