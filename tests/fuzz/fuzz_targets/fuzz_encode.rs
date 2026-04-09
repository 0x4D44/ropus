#![no_main]
use libfuzzer_sys::fuzz_target;
use mdopus::opus::encoder::{
    OpusEncoder, OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY,
    OPUS_APPLICATION_VOIP,
};

#[path = "c_reference.rs"]
mod c_reference;

const SAMPLE_RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];
const APPLICATIONS: [i32; 3] = [
    OPUS_APPLICATION_VOIP,
    OPUS_APPLICATION_AUDIO,
    OPUS_APPLICATION_RESTRICTED_LOWDELAY,
];

/// Map a byte to a bitrate in the valid Opus range.
fn byte_to_bitrate(b0: u8, b1: u8) -> i32 {
    // Map u16 (0..65535) to 6000..510000
    let raw = u16::from_le_bytes([b0, b1]) as i32;
    6000 + (raw % 504001) // clamp to valid range
}

fuzz_target!(|data: &[u8]| {
    // 6 config bytes + enough PCM for at least one 20ms frame
    // Minimum PCM: 8000 Hz * 0.02s * 1 ch * 2 bytes = 320 bytes
    if data.len() < 6 + 320 {
        return;
    }

    // --- Parse structured config ---
    let sample_rate = SAMPLE_RATES[(data[0] as usize) % SAMPLE_RATES.len()];
    let channels = if data[1] & 1 == 0 { 1 } else { 2 };
    let application = APPLICATIONS[(data[2] as usize) % APPLICATIONS.len()];
    let bitrate = byte_to_bitrate(data[3], data[4]);
    let complexity = (data[5] as i32) % 11; // 0-10
    let pcm_bytes = &data[6..];

    // Frame size = 20ms at the selected sample rate
    let frame_size = sample_rate / 50;
    let samples_needed = frame_size as usize * channels as usize;
    let bytes_needed = samples_needed * 2;

    if pcm_bytes.len() < bytes_needed {
        return;
    }

    // Interpret input bytes as i16 PCM
    let pcm: Vec<i16> = pcm_bytes[..bytes_needed]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    // --- Rust encode ---
    let mut rust_enc = match OpusEncoder::new(sample_rate, channels, application) {
        Ok(e) => e,
        Err(_) => return,
    };
    rust_enc.set_bitrate(bitrate);
    rust_enc.set_vbr(0);
    rust_enc.set_complexity(complexity);

    let mut rust_out = vec![0u8; 4000];
    let rust_ret = rust_enc.encode(&pcm, frame_size, &mut rust_out, 4000);

    // --- C reference encode ---
    let c_ret = c_reference::c_encode(&pcm, frame_size, sample_rate, channels, bitrate, complexity, application);

    // --- Differential comparison ---
    match (&rust_ret, &c_ret) {
        (Ok(rust_len), Ok(c_out)) => {
            let rust_len = *rust_len as usize;
            let c_len = c_out.len();

            // Compressed output must match byte-for-byte
            assert_eq!(
                rust_len, c_len,
                "Output length mismatch: Rust={rust_len}, C={c_len}, \
                 sr={sample_rate}, ch={channels}, app={application}, \
                 br={bitrate}, cx={complexity}"
            );

            assert_eq!(
                &rust_out[..rust_len], &c_out[..],
                "Output byte mismatch at sr={sample_rate}, ch={channels}, \
                 app={application}, br={bitrate}, cx={complexity}, len={rust_len}"
            );
        }
        (Err(_), Err(_)) => {
            // Both errored — fine
        }
        (Ok(rust_len), Err(c_err)) => {
            panic!(
                "Rust encoded ({rust_len} bytes) but C errored ({c_err}), \
                 sr={sample_rate}, ch={channels}, app={application}, \
                 br={bitrate}, cx={complexity}"
            );
        }
        (Err(rust_err), Ok(c_out)) => {
            panic!(
                "C encoded ({} bytes) but Rust errored ({rust_err}), \
                 sr={sample_rate}, ch={channels}, app={application}, \
                 br={bitrate}, cx={complexity}",
                c_out.len()
            );
        }
    }
});
