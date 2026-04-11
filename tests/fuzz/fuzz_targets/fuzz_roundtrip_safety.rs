#![no_main]
use libfuzzer_sys::fuzz_target;
use mdopus::opus::decoder::OpusDecoder;
use mdopus::opus::encoder::{
    OpusEncoder, OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY,
    OPUS_APPLICATION_VOIP,
};

/// Rust-only safety fuzz target for encode+decode roundtrip.
/// Tests for panics, OOB, infinite loops — no C comparison.

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

    // Decode what we just encoded — must not panic
    let mut dec = match OpusDecoder::new(sample_rate, channels) {
        Ok(d) => d,
        Err(_) => return,
    };
    let mut decoded = vec![0i16; samples_needed];
    let _ = dec.decode(Some(&compressed[..enc_len]), &mut decoded, frame_size, false);
});
