#![no_main]
use libfuzzer_sys::fuzz_target;
use mdopus::opus::decoder::OpusDecoder;
use mdopus::opus::encoder::{OpusEncoder, OPUS_APPLICATION_AUDIO};

fuzz_target!(|data: &[u8]| {
    // Need at least 960 samples * 2 bytes = 1920 bytes for one 20ms frame at 48kHz
    if data.len() < 1920 {
        return;
    }

    // Interpret input bytes as i16 PCM
    let pcm: Vec<i16> = data
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    let mut enc = match OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO) {
        Ok(e) => e,
        Err(_) => return,
    };
    enc.set_bitrate(64000);
    enc.set_vbr(0);

    let mut out = vec![0u8; 4000];
    let ret = enc.encode(&pcm[..960], 960, &mut out, 4000);

    if let Ok(len) = ret {
        // Verify the output is decodable
        let mut dec = match OpusDecoder::new(48000, 1) {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut dec_pcm = vec![0i16; 960];
        let _ = dec.decode(Some(&out[..len as usize]), &mut dec_pcm, 960, false);
    }
});
