#![no_main]
use libfuzzer_sys::fuzz_target;
use mdopus::opus::decoder::OpusDecoder;

#[path = "c_reference.rs"]
mod c_reference;

/// Sample rate lookup from config byte.
const SAMPLE_RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];

/// Max frame size at any sample rate (120ms at 48kHz).
const MAX_FRAME: i32 = 5760;

fuzz_target!(|data: &[u8]| {
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
    match (&rust_ret, &c_ret) {
        (Ok(rust_samples), Ok(c_pcm)) => {
            let rust_samples = *rust_samples as usize;
            let c_samples = c_pcm.len() / channels as usize;

            // Sample counts must match
            assert_eq!(
                rust_samples, c_samples,
                "Sample count mismatch: Rust={rust_samples}, C={c_samples}, \
                 sr={sample_rate}, ch={channels}, pkt_len={}",
                packet.len()
            );

            // PCM output must match sample-for-sample
            let rust_slice = &rust_pcm[..rust_samples * channels as usize];
            assert_eq!(
                rust_slice, &c_pcm[..],
                "PCM mismatch at sr={sample_rate}, ch={channels}, pkt_len={}, samples={rust_samples}",
                packet.len()
            );
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

    // --- Also test PLC (packet loss concealment) after a decode ---
    // Only do PLC if the initial decode succeeded (decoder is in valid state)
    if rust_ret.is_ok() {
        let plc_frame = sample_rate / 50; // 20ms frame
        let mut rust_plc = vec![0i16; plc_frame as usize * channels as usize];
        let _ = rust_dec.decode(None, &mut rust_plc, plc_frame, false);
    }
});
