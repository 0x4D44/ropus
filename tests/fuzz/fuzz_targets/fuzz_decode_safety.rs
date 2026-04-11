#![no_main]
use libfuzzer_sys::fuzz_target;
use mdopus::opus::decoder::OpusDecoder;

/// Rust-only safety fuzz target for the decoder.
/// Tests for panics, OOB, infinite loops — no C comparison.
/// This exercises SILK/CELT/Hybrid decode paths with arbitrary data.

const SAMPLE_RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];
const MAX_FRAME: i32 = 5760;

fuzz_target!(|data: &[u8]| {
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

    // Decode the packet — just must not panic
    let _ = dec.decode(Some(packet), &mut pcm, MAX_FRAME, false);

    // Also test PLC after decode
    let plc_frame = sample_rate / 50;
    let mut plc_pcm = vec![0i16; plc_frame as usize * channels as usize];
    let _ = dec.decode(None, &mut plc_pcm, plc_frame, false);

    // Test FEC decode
    let _ = dec.decode(Some(packet), &mut pcm, MAX_FRAME, true);
});
