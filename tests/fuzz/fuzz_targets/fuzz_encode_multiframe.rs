#![no_main]
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

fn byte_to_bitrate(b0: u8, b1: u8) -> i32 {
    let raw = u16::from_le_bytes([b0, b1]) as i32;
    6000 + (raw % 504001)
}

// Multiframe differential fuzz target.
//
// Encodes 5-10 sequential frames through a single encoder instance, comparing
// Rust vs C reference frame-by-frame. This catches state accumulation bugs that
// single-frame fuzzing is structurally blind to — the encoder carries prediction
// state, gain history, and mode decisions across frames.
fuzz_target!(|data: &[u8]| {
    init_panic_capture();
    CURRENT_INPUT.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.clear();
        buf.extend_from_slice(data);
    });

    // 7 config bytes + enough PCM for at least 5 frames at minimum config (8kHz mono 20ms)
    // Minimum PCM per frame: 160 samples * 2 bytes = 320 bytes
    // Minimum total: 7 + 5 * 320 = 1607 bytes
    if data.len() < 7 + 5 * 320 {
        return;
    }

    // --- Parse structured config ---
    let sample_rate = SAMPLE_RATES[(data[0] as usize) % SAMPLE_RATES.len()];
    let channels = if data[1] & 1 == 0 { 1 } else { 2 };
    let application = APPLICATIONS[(data[2] as usize) % APPLICATIONS.len()];
    let bitrate = byte_to_bitrate(data[3], data[4]);
    let complexity = (data[5] as i32) % 11;
    // Frame count: 5-10
    let num_frames = 5 + ((data[6] as usize) % 6);
    let pcm_bytes = &data[7..];

    let frame_size = sample_rate / 50; // 20ms
    let samples_per_frame = frame_size as usize * channels as usize;
    let bytes_per_frame = samples_per_frame * 2;
    let total_bytes_needed = bytes_per_frame * num_frames;

    if pcm_bytes.len() < total_bytes_needed {
        return;
    }

    // Split input into frames of i16 PCM
    let mut pcm_frames: Vec<Vec<i16>> = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let start = i * bytes_per_frame;
        let frame_bytes = &pcm_bytes[start..start + bytes_per_frame];
        let pcm: Vec<i16> = frame_bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        pcm_frames.push(pcm);
    }

    // --- Rust multiframe encode ---
    let mut rust_enc = match OpusEncoder::new(sample_rate, channels, application) {
        Ok(e) => e,
        Err(_) => return,
    };
    rust_enc.set_bitrate(bitrate);
    rust_enc.set_vbr(0);
    rust_enc.set_complexity(complexity);

    let mut rust_packets: Vec<Vec<u8>> = Vec::with_capacity(num_frames);
    for pcm in &pcm_frames {
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
        frame_size,
        sample_rate,
        channels,
        bitrate,
        complexity,
        application,
    ) {
        Ok(p) => p,
        Err(_) => return,
    };

    // --- Differential comparison: frame-by-frame ---
    assert_eq!(
        rust_packets.len(),
        c_packets.len(),
        "Frame count mismatch: Rust={}, C={}, sr={sample_rate}, ch={channels}, frames={num_frames}",
        rust_packets.len(),
        c_packets.len()
    );

    for (i, (rust_pkt, c_pkt)) in rust_packets.iter().zip(c_packets.iter()).enumerate() {
        assert_eq!(
            rust_pkt.len(),
            c_pkt.len(),
            "Frame {i}/{num_frames} length mismatch: Rust={}, C={}, \
             sr={sample_rate}, ch={channels}, app={application}, br={bitrate}, cx={complexity}",
            rust_pkt.len(),
            c_pkt.len()
        );

        assert_eq!(
            rust_pkt, c_pkt,
            "Frame {i}/{num_frames} byte mismatch: \
             sr={sample_rate}, ch={channels}, app={application}, br={bitrate}, cx={complexity}, \
             len={}",
            rust_pkt.len()
        );
    }

    // --- Semantic invariant: each encoded packet is parseable ---
    for (i, pkt) in rust_packets.iter().enumerate() {
        if !pkt.is_empty() {
            let nb_frames = ropus::opus::decoder::opus_packet_get_nb_frames(pkt);
            assert!(
                nb_frames.is_ok() && nb_frames.unwrap() > 0,
                "Frame {i}: encoded packet not parseable, len={}",
                pkt.len()
            );
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
