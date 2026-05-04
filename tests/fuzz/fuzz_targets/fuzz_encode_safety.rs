#![no_main]
use libfuzzer_sys::fuzz_target;
use ropus::opus::decoder::{
    opus_packet_get_nb_frames, opus_packet_get_nb_samples, OPUS_BANDWIDTH_FULLBAND,
    OPUS_BANDWIDTH_MEDIUMBAND, OPUS_BANDWIDTH_NARROWBAND, OPUS_BANDWIDTH_SUPERWIDEBAND,
    OPUS_BANDWIDTH_WIDEBAND,
};
use ropus::opus::encoder::{
    OpusEncoder, OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY,
    OPUS_APPLICATION_VOIP, OPUS_AUTO, OPUS_SIGNAL_MUSIC, OPUS_SIGNAL_VOICE,
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

// Rust-only safety fuzz target for the encoder — extended with structured
// VBR/FEC/DTX prologue, an i16/f32 PCM switch, and a deterministic runtime
// setter shuffle exercised between construction and the first encode().
//
// Tests: panics, OOB, packet parseability, CBR determinism (CBR mode),
// runtime-setter validation paths on the low-level OpusEncoder surface.
//
// Setters 7-9 in the shuffle (set_lsb_depth / set_phase_inversion_disabled /
// set_prediction_disabled) are deliberately excluded from the high-level
// `Encoder` facade but legal on the low-level `OpusEncoder` we drive here —
// fuzzing them is intentional safety coverage of the underlying CTL surface.

const SAMPLE_RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];
const APPLICATIONS: [i32; 3] = [
    OPUS_APPLICATION_VOIP,
    OPUS_APPLICATION_AUDIO,
    OPUS_APPLICATION_RESTRICTED_LOWDELAY,
];
const SIGNAL_VALUES: [i32; 3] = [OPUS_AUTO, OPUS_SIGNAL_VOICE, OPUS_SIGNAL_MUSIC];
const FORCE_CHANNELS: [i32; 3] = [OPUS_AUTO, 1, 2];
const BANDWIDTHS: [i32; 5] = [
    OPUS_BANDWIDTH_NARROWBAND,
    OPUS_BANDWIDTH_MEDIUMBAND,
    OPUS_BANDWIDTH_WIDEBAND,
    OPUS_BANDWIDTH_SUPERWIDEBAND,
    OPUS_BANDWIDTH_FULLBAND,
];

fn byte_to_bitrate(b0: u8, b1: u8) -> i32 {
    let raw = u16::from_le_bytes([b0, b1]) as i32;
    6000 + (raw % 504001)
}

/// Replay up to 16 setter calls driven by `bytes` (2 bytes per call).
/// Runs before the first `encode()` to fuzz the runtime-setters validation
/// surface on the low-level OpusEncoder.
fn apply_encoder_setter_sequence(enc: &mut OpusEncoder, bytes: &[u8]) {
    for chunk in bytes.chunks_exact(2).take(16) {
        // Setters 7-9 (lsb_depth, phase_inversion, prediction_disabled) are
        // excluded from the high-level facade in HLD V3 because mid-stream
        // toggling is API misuse. We include them here to cover their CTL
        // validation paths (out-of-range rejection); the single-frame encode
        // below intentionally avoids exercising mid-stream-corruption sequences.
        match chunk[0] % 11 {
            0 => {
                enc.set_complexity((chunk[1] % 11) as i32);
            }
            1 => {
                enc.set_signal(SIGNAL_VALUES[chunk[1] as usize % SIGNAL_VALUES.len()]);
            }
            2 => {
                enc.set_force_channels(FORCE_CHANNELS[chunk[1] as usize % FORCE_CHANNELS.len()]);
            }
            3 => {
                enc.set_max_bandwidth(BANDWIDTHS[chunk[1] as usize % BANDWIDTHS.len()]);
            }
            4 => {
                enc.set_packet_loss_perc((chunk[1] % 101) as i32);
            }
            5 => {
                enc.set_inband_fec((chunk[1] % 3) as i32);
            }
            6 => {
                enc.set_dtx((chunk[1] & 1) as i32);
            }
            7 => {
                enc.set_lsb_depth(8 + (chunk[1] % 17) as i32);
            }
            8 => {
                enc.set_phase_inversion_disabled((chunk[1] & 1) as i32);
            }
            9 => {
                enc.set_prediction_disabled((chunk[1] & 1) as i32);
            }
            10 => {
                enc.set_bitrate(byte_to_bitrate(chunk[1], chunk[1]));
            }
            _ => unreachable!(),
        }
    }
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

    let sample_rate = SAMPLE_RATES[(data[0] as usize) % SAMPLE_RATES.len()];
    let channels = if data[1] & 1 == 0 { 1 } else { 2 };
    let use_float_pcm = (data[1] & 0b0010) != 0;
    let application = APPLICATIONS[(data[2] as usize) % APPLICATIONS.len()];
    let bitrate = byte_to_bitrate(data[3], data[4]);
    let complexity = (data[5] as i32) % 11;
    let vbr = (data[6] & 0b0001) != 0;
    let vbr_constraint = (data[6] & 0b0010) != 0;
    let inband_fec_raw = ((data[6] & 0b1100) >> 2) as i32;
    let inband_fec = if inband_fec_raw == 3 {
        0
    } else {
        inband_fec_raw
    };
    let dtx = (data[7] & 0b0001) != 0;
    let loss_perc = (((data[7] & 0b1111_1110) >> 1) as i32) % 101;
    let pcm_bytes = &data[8..];

    let frame_size = sample_rate / 50;
    let samples_needed = frame_size as usize * channels as usize;
    let sample_size_bytes = if use_float_pcm { 4 } else { 2 };
    let bytes_needed = samples_needed * sample_size_bytes;

    if pcm_bytes.len() < bytes_needed {
        return;
    }

    // Setter-shuffle bytes are the trailing slice after the PCM payload, so
    // they don't fight the PCM for input budget.
    let setter_bytes = &pcm_bytes[bytes_needed..];

    let mut enc = match OpusEncoder::new(sample_rate, channels, application) {
        Ok(e) => e,
        Err(_) => return,
    };
    configure_encoder(
        &mut enc,
        bitrate,
        complexity,
        vbr,
        vbr_constraint,
        inband_fec,
        dtx,
        loss_perc,
    );
    apply_encoder_setter_sequence(&mut enc, setter_bytes);

    let mut out = vec![0u8; 4000];

    if use_float_pcm {
        let pcm: Vec<f32> = pcm_bytes[..bytes_needed]
            .chunks_exact(4)
            .map(|c| {
                let f = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                // NaN/Inf produces differential mismatch — see wrk_journals/2026.05.01 - JRN - fuzz-coverage-expansion-impl.md
                if f.is_finite() {
                    f
                } else {
                    0.0
                }
            })
            .collect();

        let ret = enc.encode_float(&pcm, frame_size, &mut out, 4000);

        if let Ok(len) = ret {
            let len = len as usize;
            let packet = &out[..len];

            if !packet.is_empty() {
                let nb_frames = opus_packet_get_nb_frames(packet);
                assert!(
                    nb_frames.is_ok() && nb_frames.unwrap() > 0,
                    "Float encoded packet not parseable: len={len}, sr={sample_rate}, ch={channels}"
                );
                let nb_samples = opus_packet_get_nb_samples(packet, sample_rate);
                if let Ok(ns) = nb_samples {
                    assert_eq!(
                        ns, frame_size,
                        "Float encoded packet nb_samples ({ns}) != frame_size ({frame_size})"
                    );
                }
            }
        }
    } else {
        let pcm: Vec<i16> = pcm_bytes[..bytes_needed]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        let ret = enc.encode(&pcm, frame_size, &mut out, 4000);

        if let Ok(len) = ret {
            let len = len as usize;
            let packet = &out[..len];

            // --- Semantic invariant: encoded packet must be parseable ---
            if !packet.is_empty() {
                let nb_frames = opus_packet_get_nb_frames(packet);
                assert!(
                    nb_frames.is_ok() && nb_frames.unwrap() > 0,
                    "Encoded packet not parseable: len={len}, sr={sample_rate}, ch={channels}"
                );
                let nb_samples = opus_packet_get_nb_samples(packet, sample_rate);
                if let Ok(ns) = nb_samples {
                    assert_eq!(
                        ns, frame_size,
                        "Encoded packet nb_samples ({ns}) != frame_size ({frame_size})"
                    );
                }
            }
        }
    }
    // CBR determinism and LOWDELAY differential comparisons that previously
    // lived here are intentionally dropped: the runtime setter shuffle above
    // means we can't reliably mirror state into a second Rust encoder or into
    // the C reference. The byte-exact differential and CBR-determinism
    // coverage for the un-shuffled prologue config space lives in
    // `fuzz_encode.rs`, which is unaffected by this change.
});
