#[path = "../fuzz_targets/frame_duration.rs"]
mod frame_duration;

use frame_duration::{
    legal_frame_duration_index, legal_frame_duration_label, legal_frame_size_samples_per_channel,
    multiframe_shape_from_byte, plc_seq_frame_duration_selector, LEGAL_FRAME_DURATION_MS_X2,
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

#[test]
fn multiframe_shape_byte_selects_legal_duration_and_frame_count() {
    let mut seen = [false; 6];

    for byte in 0u8..=u8::MAX {
        let (selector, n_frames) = multiframe_shape_from_byte(byte);
        assert!(selector < 6, "byte={byte}, selector={selector}");
        assert!(
            (2..=16).contains(&n_frames),
            "byte={byte}, n_frames={n_frames}"
        );
        seen[selector as usize] = true;
    }

    assert_eq!(seen, [true; 6]);
}

#[test]
fn multiframe_shape_reaches_all_selectors_compactly() {
    let selected_labels: Vec<_> = (0u8..=35)
        .map(|byte| {
            let (selector, _) = multiframe_shape_from_byte(byte);
            legal_frame_duration_label(selector)
        })
        .collect();

    assert!(selected_labels.contains(&"2.5ms"));
    assert!(selected_labels.contains(&"5ms"));
    assert!(selected_labels.contains(&"10ms"));
    assert!(selected_labels.contains(&"20ms"));
    assert!(selected_labels.contains(&"40ms"));
    assert!(selected_labels.contains(&"60ms"));
}

#[test]
fn plc_seq_uses_drop_mask_high_bits_as_selector() {
    assert_eq!(plc_seq_frame_duration_selector(0), 0);
    assert_eq!(plc_seq_frame_duration_selector(1 << 29), 1);
    assert_eq!(plc_seq_frame_duration_selector(1 << 30), 2);
    assert_eq!(plc_seq_frame_duration_selector(1 << 31), 4);
    assert_eq!(plc_seq_frame_duration_selector(0x0000_0fff), 0);

    assert_eq!(
        legal_frame_duration_label(plc_seq_frame_duration_selector(6 << 29)),
        "2.5ms"
    );
    assert_eq!(
        legal_frame_duration_label(plc_seq_frame_duration_selector(7 << 29)),
        "5ms"
    );
}
