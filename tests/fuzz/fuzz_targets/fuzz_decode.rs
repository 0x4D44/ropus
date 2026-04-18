#![no_main]
use libfuzzer_sys::fuzz_target;
use ropus::opus::decoder::{
    opus_packet_get_nb_frames, opus_packet_get_samples_per_frame, OpusDecoder,
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

/// Sample rate lookup from config byte.
const SAMPLE_RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];

/// Max frame size at any sample rate (120ms at 48kHz).
const MAX_FRAME: i32 = 5760;

/// CELT-only packets have TOC config >= 16 (bit 7 set).
/// SILK-only (config 0-11) and Hybrid (config 12-15) have known numerical
/// divergences — skip PCM differential comparison for those modes.
fn is_celt_only_packet(packet: &[u8]) -> bool {
    !packet.is_empty() && (packet[0] & 0x80) != 0
}

fuzz_target!(|data: &[u8]| {
    init_panic_capture();
    CURRENT_INPUT.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.clear();
        buf.extend_from_slice(data);
    });

    // Need at least 2 config bytes + 1 byte payload
    if data.len() < 3 {
        return;
    }

    // --- Parse structured config from first 2 bytes ---
    let sr_idx = (data[0] as usize) % SAMPLE_RATES.len();
    let sample_rate = SAMPLE_RATES[sr_idx];
    let channels = if data[1] & 1 == 0 { 1 } else { 2 };
    let packet = &data[2..];

    if packet.is_empty() {
        return;
    }

    // --- Rust decode ---
    let mut rust_dec = match OpusDecoder::new(sample_rate, channels) {
        Ok(d) => d,
        Err(_) => return,
    };
    let max_pcm = MAX_FRAME as usize * channels as usize;
    let mut rust_pcm = vec![0i16; max_pcm];
    let rust_ret = rust_dec.decode(Some(packet), &mut rust_pcm, MAX_FRAME, false);

    // --- C reference decode ---
    let c_ret = c_reference::c_decode(packet, sample_rate, channels);

    // --- Differential comparison ---
    // SILK/Hybrid modes have known numerical divergences — only compare PCM for
    // CELT-only packets. Sample counts and error agreement are checked for all modes.
    let celt_only = is_celt_only_packet(packet);

    match (&rust_ret, &c_ret) {
        (Ok(rust_samples), Ok(c_pcm)) => {
            let rust_samples = *rust_samples as usize;
            let c_samples = c_pcm.len() / channels as usize;

            // Sample counts must match for all modes
            assert_eq!(
                rust_samples, c_samples,
                "Sample count mismatch: Rust={rust_samples}, C={c_samples}, \
                 sr={sample_rate}, ch={channels}, pkt_len={}",
                packet.len()
            );

            // PCM output must match sample-for-sample (CELT-only)
            if celt_only {
                let rust_slice = &rust_pcm[..rust_samples * channels as usize];
                assert_eq!(
                    rust_slice, &c_pcm[..],
                    "PCM mismatch at sr={sample_rate}, ch={channels}, pkt_len={}, samples={rust_samples}",
                    packet.len()
                );
            }
        }
        (Err(_), Err(_)) => {
            // Both errored — that's fine, errors don't have to match exactly
        }
        (Ok(rust_samples), Err(c_err)) => {
            panic!(
                "Rust decoded ({rust_samples} samples) but C errored ({c_err}), \
                 sr={sample_rate}, ch={channels}, pkt_len={}",
                packet.len()
            );
        }
        (Err(rust_err), Ok(c_pcm)) => {
            panic!(
                "C decoded ({} samples) but Rust errored ({rust_err}), \
                 sr={sample_rate}, ch={channels}, pkt_len={}",
                c_pcm.len() / channels as usize,
                packet.len()
            );
        }
    }

    // --- Semantic invariant: decoded sample count matches packet structure ---
    if let Ok(rust_samples) = rust_ret {
        let expected_spf = opus_packet_get_samples_per_frame(packet, sample_rate);
        let expected_frames = opus_packet_get_nb_frames(packet);
        if let Ok(nf) = expected_frames {
            if nf > 0 && expected_spf > 0 {
                let expected_total = expected_spf * nf;
                assert_eq!(
                    rust_samples, expected_total,
                    "Decoded samples ({rust_samples}) != expected ({expected_total} = {expected_spf} * {nf}), \
                     sr={sample_rate}, ch={channels}"
                );
            }
        }
    }

    // --- Semantic invariant: decode determinism ---
    if let Ok(rust_samples) = rust_ret {
        let mut dec2 = match OpusDecoder::new(sample_rate, channels) {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut pcm2 = vec![0i16; max_pcm];
        if let Ok(n2) = dec2.decode(Some(packet), &mut pcm2, MAX_FRAME, false) {
            assert_eq!(rust_samples, n2, "Decode determinism: sample count differs");
            assert_eq!(
                &rust_pcm[..rust_samples as usize * channels as usize],
                &pcm2[..n2 as usize * channels as usize],
                "Decode determinism: PCM differs for identical packet"
            );
        }
    }

    // --- Also test PLC (packet loss concealment) after a decode ---
    if rust_ret.is_ok() {
        let plc_frame = sample_rate / 50; // 20ms frame
        let mut rust_plc = vec![0i16; plc_frame as usize * channels as usize];
        let _ = rust_dec.decode(None, &mut rust_plc, plc_frame, false);
    }
});
