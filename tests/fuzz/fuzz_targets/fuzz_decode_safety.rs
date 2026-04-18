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

// Rust-only safety fuzz target for the decoder — extended with differential
// comparison for CELT-only packets (where SILK numerical divergence is not an issue).
//
// Tests: panics, OOB, infinite loops, and correctness for CELT-only paths.

const SAMPLE_RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];
const MAX_FRAME: i32 = 5760;

// CELT-only packets have TOC config >= 16 (bit 7 of TOC byte set).
// Hybrid mode (config 12-15) also has SWB/FB bandwidth but includes a SILK
// component with known numerical divergences — must NOT be treated as CELT-only.
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

    if data.len() < 3 {
        return;
    }

    let sr_idx = (data[0] as usize) % SAMPLE_RATES.len();
    let sample_rate = SAMPLE_RATES[sr_idx];
    let channels = if data[1] & 1 == 0 { 1 } else { 2 };
    let packet = &data[2..];

    if packet.is_empty() {
        return;
    }

    let mut dec = match OpusDecoder::new(sample_rate, channels) {
        Ok(d) => d,
        Err(_) => return,
    };
    let max_pcm = MAX_FRAME as usize * channels as usize;
    let mut pcm = vec![0i16; max_pcm];

    // Decode the packet — must not panic
    let rust_ret = dec.decode(Some(packet), &mut pcm, MAX_FRAME, false);

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

    // --- Differential comparison for CELT-only packets ---
    if is_celt_only_packet(packet) {
        let c_ret = c_reference::c_decode(packet, sample_rate, channels);

        match (&rust_ret, &c_ret) {
            (Ok(rust_samples), Ok(c_pcm)) => {
                let rust_samples = *rust_samples as usize;
                let c_samples = c_pcm.len() / channels as usize;

                assert_eq!(
                    rust_samples, c_samples,
                    "CELT-only sample count mismatch: Rust={rust_samples}, C={c_samples}, \
                     sr={sample_rate}, ch={channels}, pkt_len={}",
                    packet.len()
                );

                let rust_slice = &pcm[..rust_samples * channels as usize];
                assert_eq!(
                    rust_slice, &c_pcm[..],
                    "CELT-only PCM mismatch: sr={sample_rate}, ch={channels}, pkt_len={}, \
                     samples={rust_samples}",
                    packet.len()
                );
            }
            (Err(_), Err(_)) => {}
            (Ok(rust_samples), Err(c_err)) => {
                panic!(
                    "CELT-only: Rust decoded ({rust_samples} samples) but C errored ({c_err}), \
                     sr={sample_rate}, ch={channels}, pkt_len={}",
                    packet.len()
                );
            }
            (Err(rust_err), Ok(c_pcm)) => {
                panic!(
                    "CELT-only: C decoded ({} samples) but Rust errored ({rust_err}), \
                     sr={sample_rate}, ch={channels}, pkt_len={}",
                    c_pcm.len() / channels as usize,
                    packet.len()
                );
            }
        }
    }

    // --- Semantic invariant: decode determinism ---
    // Decoding the same packet with a fresh decoder must produce identical output
    if rust_ret.is_ok() {
        let mut dec2 = match OpusDecoder::new(sample_rate, channels) {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut pcm2 = vec![0i16; max_pcm];
        let ret2 = dec2.decode(Some(packet), &mut pcm2, MAX_FRAME, false);
        if let (Ok(n1), Ok(n2)) = (rust_ret, ret2) {
            assert_eq!(n1, n2, "Decode determinism: sample count differs");
            assert_eq!(
                &pcm[..n1 as usize * channels as usize],
                &pcm2[..n2 as usize * channels as usize],
                "Decode determinism: PCM output differs for identical packet"
            );
        }
    }

    // Also test PLC after decode
    let plc_frame = sample_rate / 50;
    let mut plc_pcm = vec![0i16; plc_frame as usize * channels as usize];
    let _ = dec.decode(None, &mut plc_pcm, plc_frame, false);

    // Test FEC decode
    let _ = dec.decode(Some(packet), &mut pcm, MAX_FRAME, true);
});
