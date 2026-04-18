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

// Rust-only safety fuzz target for encode+decode roundtrip — extended with
// semantic invariants and LOWDELAY differential comparison.
//
// Tests: panics, OOB, sample count correctness, decode determinism, LOWDELAY diff.

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

    let sample_rate = SAMPLE_RATES[(data[0] as usize) % SAMPLE_RATES.len()];
    let channels = if data[1] & 1 == 0 { 1 } else { 2 };
    let application = APPLICATIONS[(data[2] as usize) % APPLICATIONS.len()];
    let bitrate = byte_to_bitrate(data[3], data[4]);
    let complexity = (data[5] as i32) % 11;
    let pcm_bytes = &data[6..];

    let frame_size = sample_rate / 50;
    let samples_needed = frame_size as usize * channels as usize;
    let bytes_needed = samples_needed * 2;

    if pcm_bytes.len() < bytes_needed {
        return;
    }

    let pcm: Vec<i16> = pcm_bytes[..bytes_needed]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    // Encode
    let mut enc = match OpusEncoder::new(sample_rate, channels, application) {
        Ok(e) => e,
        Err(_) => return,
    };
    enc.set_bitrate(bitrate);
    enc.set_vbr(0);
    enc.set_complexity(complexity);

    let mut compressed = vec![0u8; 4000];
    let enc_len = match enc.encode(&pcm, frame_size, &mut compressed, 4000) {
        Ok(l) => l as usize,
        Err(_) => return,
    };
    let packet = &compressed[..enc_len];

    // --- Semantic invariant: encoded packet is parseable ---
    if !packet.is_empty() {
        let nb_frames = opus_packet_get_nb_frames(packet);
        assert!(
            nb_frames.is_ok() && nb_frames.unwrap() > 0,
            "Encoded packet not parseable: len={enc_len}"
        );
        let nb_samples = opus_packet_get_nb_samples(packet, sample_rate);
        if let Ok(ns) = nb_samples {
            assert_eq!(
                ns, frame_size,
                "Encoded packet nb_samples ({ns}) != frame_size ({frame_size})"
            );
        }
    }

    // Decode — must not panic
    let mut dec = match OpusDecoder::new(sample_rate, channels) {
        Ok(d) => d,
        Err(_) => return,
    };
    let mut decoded = vec![0i16; samples_needed];
    let dec_ret = dec.decode(Some(packet), &mut decoded, frame_size, false);

    // --- Semantic invariant: decoded sample count == frame_size ---
    if let Ok(dec_samples) = dec_ret {
        assert_eq!(
            dec_samples, frame_size,
            "Decoded sample count ({dec_samples}) != frame_size ({frame_size}), \
             sr={sample_rate}, ch={channels}"
        );

        // --- Semantic invariant: decode determinism ---
        let mut dec2 = match OpusDecoder::new(sample_rate, channels) {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut decoded2 = vec![0i16; samples_needed];
        if let Ok(n2) = dec2.decode(Some(packet), &mut decoded2, frame_size, false) {
            assert_eq!(
                dec_samples, n2,
                "Decode determinism: sample count differs ({dec_samples} vs {n2})"
            );
            assert_eq!(
                &decoded[..dec_samples as usize * channels as usize],
                &decoded2[..n2 as usize * channels as usize],
                "Decode determinism: PCM output differs for same encoded packet"
            );
        }
    }

    // --- Differential for LOWDELAY (CELT-only) ---
    if application == OPUS_APPLICATION_RESTRICTED_LOWDELAY {
        // Compare compressed output
        let c_compressed = c_reference::c_encode(
            &pcm, frame_size, sample_rate, channels, bitrate, complexity, application,
        );
        if let Ok(c_pkt) = &c_compressed {
            assert_eq!(
                packet, &c_pkt[..],
                "LOWDELAY roundtrip: compressed output mismatch, \
                 sr={sample_rate}, ch={channels}, br={bitrate}"
            );
        }

        // Compare decoded output
        if let (Ok(dec_samples), Ok(c_pkt)) = (dec_ret, c_compressed) {
            let c_decoded = c_reference::c_decode(&c_pkt, sample_rate, channels);
            if let Ok(c_pcm) = c_decoded {
                let n = dec_samples as usize * channels as usize;
                assert_eq!(
                    &decoded[..n], &c_pcm[..],
                    "LOWDELAY roundtrip: decoded PCM mismatch, \
                     sr={sample_rate}, ch={channels}, br={bitrate}"
                );
            }
        }
    }
});
