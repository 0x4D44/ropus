#[path = "../fuzz_targets/frame_duration.rs"]
mod frame_duration;

use frame_duration::{
    legal_frame_duration_index, legal_frame_duration_label, legal_frame_size_samples_per_channel,
    LEGAL_FRAME_DURATION_MS_X2,
};

const SAMPLE_RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];
const EXPECTED_FRAME_SIZES: [[i32; 6]; 5] = [
    [20, 40, 80, 160, 320, 480],
    [30, 60, 120, 240, 480, 720],
    [40, 80, 160, 320, 640, 960],
    [60, 120, 240, 480, 960, 1440],
    [120, 240, 480, 960, 1920, 2880],
];

#[test]
fn legal_duration_metadata_is_stable() {
    assert_eq!(LEGAL_FRAME_DURATION_MS_X2, [5, 10, 20, 40, 80, 120]);

    let labels: Vec<_> = (0..6).map(legal_frame_duration_label).collect();
    assert_eq!(labels, ["2.5ms", "5ms", "10ms", "20ms", "40ms", "60ms"]);
}

#[test]
fn legal_frame_sizes_match_supported_opus_sample_rates() {
    for (sample_rate, expected_sizes) in SAMPLE_RATES.into_iter().zip(EXPECTED_FRAME_SIZES) {
        for (selector, expected_frame_size) in expected_sizes.into_iter().enumerate() {
            assert_eq!(
                legal_frame_size_samples_per_channel(sample_rate, selector as u8),
                expected_frame_size,
                "sample_rate={sample_rate}, selector={selector}"
            );
        }
    }
}

#[test]
fn selector_wraps_without_rejecting_inputs() {
    for selector in 0u8..=u8::MAX {
        assert_eq!(
            legal_frame_duration_index(selector),
            selector as usize % LEGAL_FRAME_DURATION_MS_X2.len()
        );
    }

    assert_eq!(legal_frame_duration_label(6), "2.5ms");
    assert_eq!(legal_frame_duration_label(7), "5ms");
}

#[test]
fn roundtrip_uses_existing_data7_high_bits_as_selector() {
    let selected_labels: Vec<_> = (0u8..=7)
        .map(|high_bits| {
            let data7 = high_bits << 5;
            legal_frame_duration_label(data7 >> 5)
        })
        .collect();

    assert_eq!(
        selected_labels,
        ["2.5ms", "5ms", "10ms", "20ms", "40ms", "60ms", "2.5ms", "5ms"]
    );
}
