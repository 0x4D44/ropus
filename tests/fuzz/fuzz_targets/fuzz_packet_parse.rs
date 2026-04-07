#![no_main]
use libfuzzer_sys::fuzz_target;
use mdopus::opus::decoder::{
    opus_packet_get_bandwidth, opus_packet_get_nb_channels, opus_packet_get_nb_frames,
    opus_packet_get_nb_samples, opus_packet_get_samples_per_frame,
};

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let _ = opus_packet_get_bandwidth(data);
    let _ = opus_packet_get_nb_channels(data);
    let _ = opus_packet_get_nb_frames(data);
    let _ = opus_packet_get_samples_per_frame(data, 48000);
    let _ = opus_packet_get_nb_samples(data, 48000);

    // Also test with other sample rates
    let _ = opus_packet_get_samples_per_frame(data, 8000);
    let _ = opus_packet_get_nb_samples(data, 16000);
});
