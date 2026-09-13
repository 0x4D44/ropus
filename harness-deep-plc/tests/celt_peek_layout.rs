#![cfg(not(no_reference))]
//! Positive-path guard for the private CELT decoder mirror used by the peek shim.

use ropus::opus::decoder::MODE_CELT_ONLY;
use ropus::{OPUS_APPLICATION_AUDIO, OPUS_OK, OpusEncoder};
use ropus_harness_deep_plc::CRefFloatDecoder;

#[test]
fn celt_peeks_read_trailing_state_after_decode() {
    const FS: i32 = 48_000;
    const CHANNELS: i32 = 1;
    const FRAME_SIZE: i32 = 960;

    let mut encoder = OpusEncoder::new(FS, CHANNELS, OPUS_APPLICATION_AUDIO).unwrap();
    assert_eq!(encoder.set_force_mode(MODE_CELT_ONLY), OPUS_OK);
    assert_eq!(encoder.set_bitrate(64_000), OPUS_OK);
    let pcm: Vec<i16> = (0..FRAME_SIZE as usize)
        .map(|i| {
            let t = i as f64 / FS as f64;
            (28_000.0 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()) as i16
        })
        .collect();
    let mut packet = vec![0u8; 4_000];
    let packet_capacity = packet.len() as i32;
    let packet_len = encoder
        .encode(&pcm, FRAME_SIZE, &mut packet, packet_capacity)
        .unwrap() as usize;
    packet.truncate(packet_len);
    assert!(packet.first().is_some_and(|&toc| toc & 0x80 != 0));

    let mut decoder = CRefFloatDecoder::new(FS, CHANNELS).expect("C decoder create");
    let mut decoded = vec![0i16; FRAME_SIZE as usize];
    assert_eq!(
        decoder
            .decode(Some(&packet), &mut decoded, FRAME_SIZE, false)
            .unwrap(),
        FRAME_SIZE
    );
    assert!(decoded.iter().any(|&sample| sample != 0));
    assert_eq!(decoder.decode_mem_stride(), 2_048 + 120);

    let count = decoder.nb_ebands() * 2;
    let old_band_e = decoder
        .peek_old_band_e(0, count)
        .expect("valid oldBandE range");
    assert!(old_band_e.iter().all(|value| value.is_finite()));
    assert!(
        old_band_e.iter().any(|&value| value != 0.0),
        "a decoded non-silent CELT frame must update oldBandE"
    );
}
