#![no_main]
use libfuzzer_sys::fuzz_target;
use mdopus::opus::decoder::OpusDecoder;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    // Try decoding at 48kHz mono (most common config)
    let mut dec = match OpusDecoder::new(48000, 1) {
        Ok(d) => d,
        Err(_) => return,
    };

    let mut pcm = vec![0i16; 5760]; // max frame size
    // Decode — may return error, that's fine. Must not panic.
    let _ = dec.decode(Some(data), &mut pcm, 5760, false);

    // Also try PLC after receiving data
    let _ = dec.decode(None, &mut pcm, 960, false);
});
