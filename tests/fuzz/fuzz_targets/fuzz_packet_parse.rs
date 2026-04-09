#![no_main]
use libfuzzer_sys::fuzz_target;
use mdopus::opus::decoder::{
    opus_packet_get_bandwidth, opus_packet_get_nb_channels, opus_packet_get_nb_frames,
    opus_packet_get_nb_samples, opus_packet_get_samples_per_frame,
};

#[path = "c_reference.rs"]
mod c_reference;

const SAMPLE_RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    // --- Bandwidth ---
    let rust_bw = opus_packet_get_bandwidth(data);
    let c_bw = c_reference::c_packet_get_bandwidth(data);
    assert_eq!(
        rust_bw, c_bw,
        "get_bandwidth mismatch: Rust={rust_bw}, C={c_bw}, first_byte=0x{:02x}, len={}",
        data[0],
        data.len()
    );

    // --- Nb channels ---
    let rust_ch = opus_packet_get_nb_channels(data);
    let c_ch = c_reference::c_packet_get_nb_channels(data);
    assert_eq!(
        rust_ch, c_ch,
        "get_nb_channels mismatch: Rust={rust_ch}, C={c_ch}, first_byte=0x{:02x}",
        data[0]
    );

    // --- Nb frames ---
    let rust_frames = opus_packet_get_nb_frames(data).unwrap_or_else(|e| e);
    let c_frames = c_reference::c_packet_get_nb_frames(data);
    assert_eq!(
        rust_frames, c_frames,
        "get_nb_frames mismatch: Rust={rust_frames}, C={c_frames}, \
         first_byte=0x{:02x}, len={}",
        data[0],
        data.len()
    );

    // --- Samples per frame and nb samples at ALL sample rates ---
    for &sr in &SAMPLE_RATES {
        let rust_spf = opus_packet_get_samples_per_frame(data, sr);
        let c_spf = c_reference::c_packet_get_samples_per_frame(data, sr);
        assert_eq!(
            rust_spf, c_spf,
            "get_samples_per_frame mismatch at sr={sr}: Rust={rust_spf}, C={c_spf}, \
             first_byte=0x{:02x}",
            data[0]
        );

        let rust_ns = opus_packet_get_nb_samples(data, sr).unwrap_or_else(|e| e);
        let c_ns = c_reference::c_packet_get_nb_samples(data, sr);
        assert_eq!(
            rust_ns, c_ns,
            "get_nb_samples mismatch at sr={sr}: Rust={rust_ns}, C={c_ns}, \
             first_byte=0x{:02x}, len={}",
            data[0],
            data.len()
        );
    }
});
