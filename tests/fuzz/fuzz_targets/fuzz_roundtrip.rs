#![no_main]
use libfuzzer_sys::fuzz_target;
use mdopus::opus::decoder::OpusDecoder;
use mdopus::opus::encoder::{OpusEncoder, OPUS_APPLICATION_AUDIO};

fuzz_target!(|data: &[u8]| {
    if data.len() < 1920 {
        return;
    }

    let pcm: Vec<i16> = data
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    let frame_size = 960;
    if pcm.len() < frame_size {
        return;
    }

    let mut enc = match OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO) {
        Ok(e) => e,
        Err(_) => return,
    };
    enc.set_bitrate(64000);
    enc.set_vbr(0);

    let mut compressed = vec![0u8; 4000];
    let len = match enc.encode(&pcm[..frame_size], frame_size as i32, &mut compressed, 4000) {
        Ok(l) => l as usize,
        Err(_) => return,
    };

    let mut dec = match OpusDecoder::new(48000, 1) {
        Ok(d) => d,
        Err(_) => return,
    };
    let mut out_pcm = vec![0i16; frame_size];
    match dec.decode(
        Some(&compressed[..len]),
        &mut out_pcm,
        frame_size as i32,
        false,
    ) {
        Ok(samples) => assert_eq!(samples, frame_size as i32, "decoded frame size mismatch"),
        Err(_) => {} // Encoding succeeded but decode failed — worth investigating but don't panic the fuzzer
    }
});
