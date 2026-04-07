#![no_main]
use libfuzzer_sys::fuzz_target;
use mdopus::opus::repacketizer::OpusRepacketizer;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let mut rp = OpusRepacketizer::new();
    rp.init();

    // Try to cat the data as a packet
    let len = data.len() as i32;
    let _ = rp.cat(data, len);

    // Try to extract output
    let mut out = vec![0u8; data.len() + 256];
    let maxlen = out.len() as i32;
    let _ = rp.out(&mut out, maxlen);
});
