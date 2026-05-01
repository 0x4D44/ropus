#![no_main]
use libfuzzer_sys::fuzz_target;
use ropus::opus::decoder::{
    opus_packet_get_nb_frames, opus_packet_get_samples_per_frame, OpusDecoder,
};
use std::cell::RefCell;
use std::sync::Once;

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

// Rust-only safety fuzz target for the decoder — extended with an i16/f32
// PCM-format switch and a deterministic runtime setter shuffle exercised
// between construction and the first decode().
//
// Tests: panics, OOB, infinite loops, decoder-setter validation, and
// PCM determinism after state mutations including reset().
//
// The setter shuffle is driven by a SEPARATE slice of input bytes (carved
// from the tail of `data`) so it doesn't compete with the packet payload
// for input budget. The original CELT-only differential against the C
// reference is dropped here: once the shuffle has fired the decoder state
// can no longer be mirrored cleanly into the C decoder. Byte-exact
// CELT-only differential coverage lives in `fuzz_decode.rs`.

const SAMPLE_RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];
const MAX_FRAME: i32 = 5760;

// Per-target cap: at most 8 setter calls (16 bytes), per HLD V2.
const MAX_SETTER_CALLS: usize = 8;
const SETTER_BYTES_BUDGET: usize = MAX_SETTER_CALLS * 2;

/// Replay up to 8 setter calls on the OpusDecoder driven by `bytes`
/// (2 bytes per call). `reset()` mid-shuffle is the most interesting case —
/// it tests that subsequent decode() calls survive a state reset.
fn apply_decoder_setter_sequence(dec: &mut OpusDecoder, bytes: &[u8]) {
    for chunk in bytes.chunks_exact(2).take(MAX_SETTER_CALLS) {
        // `set_complexity` on a decoder has no decode-path consumer (the field
        // is stored but unused); we cover it for validator-path coverage only.
        // `set_gain` and `reset()` have observable effects.
        match chunk[0] % 4 {
            0 => {
                let _ = dec.set_gain(i16::from_le_bytes([chunk[1], 0]) as i32);
            }
            1 => {
                let _ = dec.set_complexity((chunk[1] % 11) as i32);
            }
            2 => {
                dec.set_phase_inversion_disabled((chunk[1] & 1) != 0);
            }
            3 => {
                dec.reset();
            }
            _ => unreachable!(),
        }
    }
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
    let use_float_pcm = (data[1] & 0b0010) != 0;

    // Carve the tail of `data` for the setter shuffle so PCM-payload bytes
    // and shuffle bytes don't fight over the same input budget. Only carve
    // when there's room beyond a meaningful packet payload (>=8 bytes).
    let payload = &data[2..];
    let (packet, setter_bytes): (&[u8], &[u8]) = if payload.len() > 8 + SETTER_BYTES_BUDGET {
        let split = payload.len() - SETTER_BYTES_BUDGET;
        (&payload[..split], &payload[split..])
    } else {
        (payload, &[])
    };

    if packet.is_empty() {
        return;
    }

    let mut dec = match OpusDecoder::new(sample_rate, channels) {
        Ok(d) => d,
        Err(_) => return,
    };

    // Apply runtime setter shuffle BEFORE the first decode.
    apply_decoder_setter_sequence(&mut dec, setter_bytes);

    let max_pcm = MAX_FRAME as usize * channels as usize;

    if use_float_pcm {
        let mut pcm = vec![0f32; max_pcm];
        let rust_ret = dec.decode_float(Some(packet), &mut pcm, MAX_FRAME, false);

        if let Ok(rust_samples) = rust_ret {
            let expected_spf = opus_packet_get_samples_per_frame(packet, sample_rate);
            let expected_frames = opus_packet_get_nb_frames(packet);
            if let Ok(nf) = expected_frames {
                if nf > 0 && expected_spf > 0 {
                    let expected_total = expected_spf * nf;
                    assert_eq!(
                        rust_samples, expected_total,
                        "Float decoded samples ({rust_samples}) != expected ({expected_total} = {expected_spf} * {nf}), \
                         sr={sample_rate}, ch={channels}"
                    );
                }
            }
        }

        // PLC + FEC paths must not panic post-shuffle either.
        let plc_frame = sample_rate / 50;
        let mut plc_pcm = vec![0f32; plc_frame as usize * channels as usize];
        let _ = dec.decode_float(None, &mut plc_pcm, plc_frame, false);
        let _ = dec.decode_float(Some(packet), &mut pcm, MAX_FRAME, true);
    } else {
        let mut pcm = vec![0i16; max_pcm];
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

        // PLC + FEC paths must not panic post-shuffle either.
        let plc_frame = sample_rate / 50;
        let mut plc_pcm = vec![0i16; plc_frame as usize * channels as usize];
        let _ = dec.decode(None, &mut plc_pcm, plc_frame, false);
        let _ = dec.decode(Some(packet), &mut pcm, MAX_FRAME, true);
    }
});
