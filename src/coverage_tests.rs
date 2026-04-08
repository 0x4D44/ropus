use crate::opus::decoder::{MODE_CELT_ONLY, MODE_SILK_ONLY, OPUS_OK, OpusDecoder};
use crate::opus::encoder::{
    OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_APPLICATION_VOIP,
    OPUS_FRAMESIZE_5_MS, OPUS_FRAMESIZE_40_MS, OPUS_SIGNAL_VOICE, OpusEncoder,
};
use crate::opus::multistream::{OpusMSDecoder, OpusMSEncoder};

fn patterned_pcm_i16(frame_size: usize, channels: usize, seed: i32) -> Vec<i16> {
    (0..frame_size * channels)
        .map(|i| {
            let base = ((i as i32 * 7919 + seed * 911) % 28000) - 14000;
            if channels == 2 && i % 2 == 1 {
                (base / 2) as i16
            } else {
                base as i16
            }
        })
        .collect()
}

fn patterned_pcm_f32(frame_size: usize, channels: usize, seed: i32) -> Vec<f32> {
    patterned_pcm_i16(frame_size, channels, seed)
        .into_iter()
        .map(|sample| sample as f32 / 32768.0)
        .collect()
}

#[test]
fn public_api_silk_only_roundtrip_and_plc() {
    let mut enc = OpusEncoder::new(16000, 1, OPUS_APPLICATION_VOIP).unwrap();
    assert_eq!(enc.set_force_mode(MODE_SILK_ONLY), OPUS_OK);
    assert_eq!(enc.set_bitrate(32000), OPUS_OK);
    assert_eq!(enc.set_vbr(1), OPUS_OK);
    assert_eq!(enc.set_vbr_constraint(1), OPUS_OK);
    assert_eq!(enc.set_inband_fec(1), OPUS_OK);
    assert_eq!(enc.set_packet_loss_perc(20), OPUS_OK);
    assert_eq!(enc.set_dtx(1), OPUS_OK);
    assert_eq!(enc.set_signal(OPUS_SIGNAL_VOICE), OPUS_OK);
    assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_40_MS), OPUS_OK);
    assert_eq!(enc.get_expert_frame_duration(), OPUS_FRAMESIZE_40_MS);
    assert_eq!(enc.get_channels(), 1);
    assert_eq!(enc.get_sample_rate(), 16000);

    let mut dec = OpusDecoder::new(16000, 1).unwrap();

    for frame_idx in 0..3 {
        let pcm = patterned_pcm_i16(640, 1, 100 + frame_idx * 17);
        let mut packet = vec![0u8; 1500];
        let packet_capacity = packet.len() as i32;
        let len = enc.encode(&pcm, 640, &mut packet, packet_capacity).unwrap();
        let packet = &packet[..len as usize];

        assert!(len > 1);
        assert_eq!(enc.get_mode(), MODE_SILK_ONLY);
        assert!(enc.get_final_range() > 0);
        assert_eq!(dec.get_nb_samples(packet).unwrap(), 640);

        let mut out = vec![0i16; 640];
        let decoded = dec.decode(Some(packet), &mut out, 640, false).unwrap();
        assert_eq!(decoded, 640);
        assert!(out.iter().any(|&sample| sample != 0));
        assert_eq!(dec.get_last_packet_duration(), 640);

        let mut fec_out = vec![0i16; 640];
        let decoded_fec = dec.decode(Some(packet), &mut fec_out, 640, true).unwrap();
        assert_eq!(decoded_fec, 640);
    }

    let mut plc = vec![0i16; 640];
    assert_eq!(dec.decode(None, &mut plc, 640, false).unwrap(), 640);
}

#[test]
fn public_api_restricted_lowdelay_float_roundtrip() {
    let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_RESTRICTED_LOWDELAY).unwrap();
    assert_eq!(enc.set_vbr(0), OPUS_OK);
    assert_eq!(enc.set_complexity(10), OPUS_OK);
    assert_eq!(enc.set_expert_frame_duration(OPUS_FRAMESIZE_5_MS), OPUS_OK);
    assert_eq!(enc.set_phase_inversion_disabled(1), OPUS_OK);
    assert_eq!(enc.set_force_channels(2), OPUS_OK);

    let pcm = patterned_pcm_f32(240, 2, 43);
    let mut packet = vec![0u8; 1500];
    let packet_capacity = packet.len() as i32;
    let len = enc
        .encode_float(&pcm, 240, &mut packet, packet_capacity)
        .unwrap();
    let packet = &packet[..len as usize];

    assert!(len > 0);
    assert_eq!(enc.get_mode(), MODE_CELT_ONLY);
    assert!(enc.get_final_range() > 0);

    let mut dec = OpusDecoder::new(48000, 2).unwrap();
    dec.set_phase_inversion_disabled(true);

    let mut out = vec![0f32; 240 * 2];
    let decoded = dec
        .decode_float(Some(packet), &mut out, 240, false)
        .unwrap();
    assert_eq!(decoded, 240);
    assert!(out.iter().any(|sample| sample.abs() > 1e-4));
}

#[test]
fn public_api_decode24_roundtrip() {
    let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
    let pcm = patterned_pcm_i16(960, 1, 77);
    let mut packet = vec![0u8; 1500];
    let packet_capacity = packet.len() as i32;
    let len = enc.encode(&pcm, 960, &mut packet, packet_capacity).unwrap();
    let packet = &packet[..len as usize];

    let mut dec = OpusDecoder::new(48000, 1).unwrap();
    let mut pcm24 = vec![0i32; 960];
    let decoded = dec.decode24(Some(packet), &mut pcm24, 960, false).unwrap();
    assert_eq!(decoded, 960);
    assert!(pcm24.iter().any(|&sample| sample != 0));
}

#[test]
fn public_api_multistream_surround_roundtrip() {
    let (mut enc, streams, coupled, mapping) =
        OpusMSEncoder::new_surround(48000, 6, 1, OPUS_APPLICATION_AUDIO).unwrap();
    let mut dec = OpusMSDecoder::new(48000, 6, streams, coupled, &mapping).unwrap();

    assert_eq!(enc.nb_streams(), streams);
    assert_eq!(enc.nb_coupled_streams(), coupled);

    let pcm = patterned_pcm_i16(960, 6, 9);
    let mut packet = vec![0u8; 4000];
    let packet_capacity = packet.len() as i32;
    let len = enc.encode(&pcm, 960, &mut packet, packet_capacity).unwrap();
    assert!(len > 0);
    assert!(enc.get_final_range() > 0);

    let mut out = vec![0i16; 960 * 6];
    let decoded = dec
        .decode(Some(&packet[..len as usize]), len, &mut out, 960, false)
        .unwrap();
    assert_eq!(decoded, 960);
    assert!(out.iter().any(|&sample| sample != 0));
    assert!(dec.get_final_range() > 0);
}
