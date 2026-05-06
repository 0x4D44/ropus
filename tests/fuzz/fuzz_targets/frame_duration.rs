pub const LEGAL_FRAME_DURATION_MS_X2: [u16; 6] = [5, 10, 20, 40, 80, 120];

pub fn legal_frame_duration_index(selector: u8) -> usize {
    selector as usize % LEGAL_FRAME_DURATION_MS_X2.len()
}

pub fn legal_frame_size_samples_per_channel(sample_rate: i32, selector: u8) -> i32 {
    match legal_frame_duration_index(selector) {
        0 => sample_rate / 400,
        1 => sample_rate / 200,
        2 => sample_rate / 100,
        3 => sample_rate / 50,
        4 => sample_rate / 25,
        5 => 3 * sample_rate / 50,
        _ => unreachable!(),
    }
}

pub fn legal_frame_duration_label(selector: u8) -> &'static str {
    match legal_frame_duration_index(selector) {
        0 => "2.5ms",
        1 => "5ms",
        2 => "10ms",
        3 => "20ms",
        4 => "40ms",
        5 => "60ms",
        _ => unreachable!(),
    }
}
