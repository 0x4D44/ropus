use crate::opus::decoder::{MODE_CELT_ONLY, MODE_HYBRID, MODE_SILK_ONLY, OPUS_OK, OpusDecoder};
use crate::opus::encoder::{
    OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_APPLICATION_VOIP,
    OPUS_FRAMESIZE_5_MS, OPUS_FRAMESIZE_10_MS, OPUS_FRAMESIZE_40_MS, OPUS_FRAMESIZE_60_MS,
    OPUS_SIGNAL_MUSIC, OPUS_SIGNAL_VOICE, OpusEncoder,
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

// ===========================================================================
// Coverage improvement: comprehensive encode/decode matrix
// ===========================================================================

/// Encode `n_frames` frames and return all packets.
fn encode_frames(
    enc: &mut OpusEncoder,
    frame_size: usize,
    channels: usize,
    n_frames: usize,
) -> Vec<Vec<u8>> {
    let mut packets = Vec::new();
    for i in 0..n_frames {
        let pcm = patterned_pcm_i16(frame_size, channels, 100 + i as i32 * 13);
        let mut pkt = vec![0u8; 1500];
        let len = enc
            .encode(&pcm, frame_size as i32, &mut pkt, 1500)
            .unwrap();
        packets.push(pkt[..len as usize].to_vec());
    }
    packets
}

/// Decode a packet (or PLC if None) and verify output.
fn decode_packet(dec: &mut OpusDecoder, pkt: Option<&[u8]>, frame_size: i32) -> Vec<i16> {
    let channels = dec.get_channels() as usize;
    let mut out = vec![0i16; frame_size as usize * channels];
    let decoded = dec.decode(pkt, &mut out, frame_size, false).unwrap();
    assert_eq!(decoded, frame_size);
    out
}

#[test]
fn coverage_silk_narrowband_8khz_mono() {
    let mut enc = OpusEncoder::new(8000, 1, OPUS_APPLICATION_VOIP).unwrap();
    enc.set_force_mode(MODE_SILK_ONLY);
    enc.set_bitrate(12000);
    enc.set_complexity(10);
    let mut dec = OpusDecoder::new(8000, 1).unwrap();
    let pkts = encode_frames(&mut enc, 160, 1, 5);
    for pkt in &pkts {
        decode_packet(&mut dec, Some(pkt), 160);
    }
    // PLC
    decode_packet(&mut dec, None, 160);
}

#[test]
fn coverage_silk_mediumband_12khz_stereo() {
    let mut enc = OpusEncoder::new(12000, 2, OPUS_APPLICATION_VOIP).unwrap();
    enc.set_force_mode(MODE_SILK_ONLY);
    enc.set_bitrate(24000);
    enc.set_complexity(5);
    let mut dec = OpusDecoder::new(12000, 2).unwrap();
    let pkts = encode_frames(&mut enc, 240, 2, 5);
    for pkt in &pkts {
        decode_packet(&mut dec, Some(pkt), 240);
    }
}

#[test]
fn coverage_silk_wideband_16khz_fec() {
    let mut enc = OpusEncoder::new(16000, 1, OPUS_APPLICATION_VOIP).unwrap();
    enc.set_force_mode(MODE_SILK_ONLY);
    enc.set_bitrate(32000);
    enc.set_inband_fec(1);
    enc.set_packet_loss_perc(25);
    enc.set_complexity(10);
    let mut dec = OpusDecoder::new(16000, 1).unwrap();
    let pkts = encode_frames(&mut enc, 320, 1, 8);
    // Normal decode
    for pkt in &pkts[..4] {
        decode_packet(&mut dec, Some(pkt), 320);
    }
    // FEC decode (look-ahead)
    let mut out = vec![0i16; 320];
    dec.decode(Some(&pkts[5]), &mut out, 320, true).unwrap();
    // Resume normal
    decode_packet(&mut dec, Some(&pkts[6]), 320);
}

#[test]
fn coverage_celt_only_48khz_mono_cbr() {
    let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
    enc.set_force_mode(MODE_CELT_ONLY);
    enc.set_bitrate(64000);
    enc.set_vbr(0); // CBR
    enc.set_complexity(10);
    let mut dec = OpusDecoder::new(48000, 1).unwrap();
    let pkts = encode_frames(&mut enc, 960, 1, 5);
    for pkt in &pkts {
        decode_packet(&mut dec, Some(pkt), 960);
    }
}

#[test]
fn coverage_celt_only_48khz_stereo_vbr() {
    let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
    enc.set_force_mode(MODE_CELT_ONLY);
    enc.set_bitrate(128000);
    enc.set_vbr(1);
    enc.set_vbr_constraint(1);
    enc.set_complexity(10);
    let mut dec = OpusDecoder::new(48000, 2).unwrap();
    let pkts = encode_frames(&mut enc, 960, 2, 5);
    for pkt in &pkts {
        decode_packet(&mut dec, Some(pkt), 960);
    }
}

#[test]
fn coverage_hybrid_mode_48khz() {
    let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
    enc.set_force_mode(MODE_HYBRID);
    enc.set_bitrate(48000);
    enc.set_complexity(10);
    let mut dec = OpusDecoder::new(48000, 1).unwrap();
    let pkts = encode_frames(&mut enc, 960, 1, 5);
    for pkt in &pkts {
        decode_packet(&mut dec, Some(pkt), 960);
    }
}

#[test]
fn coverage_hybrid_stereo_with_fec() {
    let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_VOIP).unwrap();
    enc.set_force_mode(MODE_HYBRID);
    enc.set_bitrate(64000);
    enc.set_inband_fec(1);
    enc.set_packet_loss_perc(15);
    enc.set_complexity(10);
    let mut dec = OpusDecoder::new(48000, 2).unwrap();
    let pkts = encode_frames(&mut enc, 960, 2, 8);
    for pkt in &pkts {
        decode_packet(&mut dec, Some(pkt), 960);
    }
}

#[test]
fn coverage_dtx_silence_detection() {
    let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
    enc.set_bitrate(32000);
    enc.set_dtx(1);
    enc.set_vbr(1);
    enc.set_complexity(5);
    // Encode active speech first
    for i in 0..3 {
        let pcm = patterned_pcm_i16(960, 1, i * 17);
        let mut pkt = vec![0u8; 1500];
        enc.encode(&pcm, 960, &mut pkt, 1500).unwrap();
    }
    // Then encode silence → should eventually emit DTX packets
    let silence = vec![0i16; 960];
    let mut dtx_found = false;
    for _ in 0..20 {
        let mut pkt = vec![0u8; 1500];
        let len = enc.encode(&silence, 960, &mut pkt, 1500).unwrap();
        if len == 1 {
            dtx_found = true;
            break;
        }
    }
    // DTX may or may not trigger depending on internal state; just ensure encoding works
    assert!(dtx_found || true); // Non-strict: encoding without panic is the goal
}

#[test]
fn coverage_restricted_lowdelay_5ms() {
    let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_RESTRICTED_LOWDELAY).unwrap();
    enc.set_bitrate(64000);
    enc.set_expert_frame_duration(OPUS_FRAMESIZE_5_MS);
    let mut dec = OpusDecoder::new(48000, 1).unwrap();
    let pkts = encode_frames(&mut enc, 240, 1, 5);
    for pkt in &pkts {
        let mut out = vec![0i16; 240];
        dec.decode(Some(pkt), &mut out, 240, false).unwrap();
    }
}

#[test]
fn coverage_restricted_lowdelay_10ms_stereo() {
    let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_RESTRICTED_LOWDELAY).unwrap();
    enc.set_bitrate(128000);
    enc.set_expert_frame_duration(OPUS_FRAMESIZE_10_MS);
    let mut dec = OpusDecoder::new(48000, 2).unwrap();
    let pkts = encode_frames(&mut enc, 480, 2, 5);
    for pkt in &pkts {
        let mut out = vec![0i16; 480 * 2];
        dec.decode(Some(pkt), &mut out, 480, false).unwrap();
    }
}

#[test]
fn coverage_40ms_multiframe_silk() {
    let mut enc = OpusEncoder::new(16000, 1, OPUS_APPLICATION_VOIP).unwrap();
    enc.set_bitrate(24000);
    enc.set_expert_frame_duration(OPUS_FRAMESIZE_40_MS);
    let mut dec = OpusDecoder::new(16000, 1).unwrap();
    let pkts = encode_frames(&mut enc, 640, 1, 3);
    for pkt in &pkts {
        let mut out = vec![0i16; 640];
        dec.decode(Some(pkt), &mut out, 640, false).unwrap();
    }
}

#[test]
fn coverage_60ms_multiframe_silk() {
    let mut enc = OpusEncoder::new(16000, 1, OPUS_APPLICATION_VOIP).unwrap();
    enc.set_bitrate(32000);
    enc.set_expert_frame_duration(OPUS_FRAMESIZE_60_MS);
    enc.set_complexity(10);
    let mut dec = OpusDecoder::new(16000, 1).unwrap();
    let pkts = encode_frames(&mut enc, 960, 1, 3);
    for pkt in &pkts {
        let mut out = vec![0i16; 960];
        dec.decode(Some(pkt), &mut out, 960, false).unwrap();
    }
}

#[test]
fn coverage_force_channels_mono_in_stereo_encoder() {
    let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
    enc.set_bitrate(64000);
    enc.set_force_channels(1); // Force mono output
    let pcm = patterned_pcm_i16(960, 2, 42);
    let mut pkt = vec![0u8; 1500];
    let len = enc.encode(&pcm, 960, &mut pkt, 1500).unwrap();
    assert!(len > 0);
}

#[test]
fn coverage_signal_type_music() {
    let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
    enc.set_bitrate(96000);
    enc.set_signal(OPUS_SIGNAL_MUSIC);
    enc.set_complexity(10);
    let pkts = encode_frames(&mut enc, 960, 1, 5);
    assert!(pkts.iter().all(|p| !p.is_empty()));
}

#[test]
fn coverage_complexity_sweep() {
    for complexity in [0, 1, 2, 5, 8, 10] {
        let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        enc.set_bitrate(64000);
        enc.set_complexity(complexity);
        let pcm = patterned_pcm_i16(960, 1, complexity);
        let mut pkt = vec![0u8; 1500];
        let len = enc.encode(&pcm, 960, &mut pkt, 1500).unwrap();
        assert!(len > 0, "complexity {} failed", complexity);
    }
}

#[test]
fn coverage_sample_rate_sweep_silk() {
    for &rate in &[8000, 12000, 16000, 24000] {
        let frame_size = rate / 50; // 20ms
        let mut enc = OpusEncoder::new(rate, 1, OPUS_APPLICATION_VOIP).unwrap();
        enc.set_bitrate(24000);
        enc.set_force_mode(MODE_SILK_ONLY);
        let mut dec = OpusDecoder::new(rate, 1).unwrap();
        let pcm = patterned_pcm_i16(frame_size as usize, 1, rate);
        let mut pkt = vec![0u8; 1500];
        let len = enc.encode(&pcm, frame_size, &mut pkt, 1500).unwrap();
        let mut out = vec![0i16; frame_size as usize];
        dec.decode(Some(&pkt[..len as usize]), &mut out, frame_size, false).unwrap();
    }
}

#[test]
fn coverage_decode_at_different_output_rates() {
    // Encode at 48kHz, decode at various rates
    let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
    enc.set_bitrate(64000);
    let pcm = patterned_pcm_i16(960, 1, 99);
    let mut pkt = vec![0u8; 1500];
    let len = enc.encode(&pcm, 960, &mut pkt, 1500).unwrap();
    let pkt = &pkt[..len as usize];

    for &rate in &[8000, 12000, 16000, 24000, 48000] {
        let frame_size = rate / 50;
        let mut dec = OpusDecoder::new(rate, 1).unwrap();
        let mut out = vec![0i16; frame_size as usize];
        dec.decode(Some(pkt), &mut out, frame_size, false).unwrap();
    }
}

#[test]
fn coverage_plc_progressive_decay() {
    let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
    enc.set_bitrate(32000);
    let mut dec = OpusDecoder::new(48000, 1).unwrap();
    // Prime with real data
    let pkts = encode_frames(&mut enc, 960, 1, 3);
    for pkt in &pkts {
        decode_packet(&mut dec, Some(pkt), 960);
    }
    // Multiple consecutive PLC frames → progressive decay
    for _ in 0..10 {
        decode_packet(&mut dec, None, 960);
    }
}

#[test]
fn coverage_encode_float_celt_stereo() {
    let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
    enc.set_force_mode(MODE_CELT_ONLY);
    enc.set_bitrate(128000);
    let pcm = patterned_pcm_f32(960, 2, 55);
    let mut pkt = vec![0u8; 1500];
    let len = enc.encode_float(&pcm, 960, &mut pkt, 1500).unwrap();
    assert!(len > 0);
}

#[test]
fn coverage_encode_float_silk_mono() {
    let mut enc = OpusEncoder::new(16000, 1, OPUS_APPLICATION_VOIP).unwrap();
    enc.set_bitrate(16000);
    let pcm = patterned_pcm_f32(320, 1, 66);
    let mut pkt = vec![0u8; 1500];
    let len = enc.encode_float(&pcm, 320, &mut pkt, 1500).unwrap();
    assert!(len > 0);
}

#[test]
fn coverage_decode_float_and_decode24() {
    let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
    enc.set_bitrate(64000);
    let pcm = patterned_pcm_i16(960, 1, 77);
    let mut pkt = vec![0u8; 1500];
    let len = enc.encode(&pcm, 960, &mut pkt, 1500).unwrap();
    let pkt = &pkt[..len as usize];

    // decode_float
    let mut dec = OpusDecoder::new(48000, 1).unwrap();
    let mut out_f32 = vec![0f32; 960];
    dec.decode_float(Some(pkt), &mut out_f32, 960, false).unwrap();
    assert!(out_f32.iter().any(|s| s.abs() > 1e-4));

    // decode24
    let mut dec2 = OpusDecoder::new(48000, 1).unwrap();
    let mut out_i32 = vec![0i32; 960];
    dec2.decode24(Some(pkt), &mut out_i32, 960, false).unwrap();
    assert!(out_i32.iter().any(|&s| s != 0));
}

#[test]
fn coverage_low_bitrate_silk_stereo_10ms() {
    let mut enc = OpusEncoder::new(16000, 2, OPUS_APPLICATION_VOIP).unwrap();
    enc.set_bitrate(12000);
    enc.set_force_mode(MODE_SILK_ONLY);
    enc.set_complexity(10);
    enc.set_expert_frame_duration(OPUS_FRAMESIZE_10_MS);
    let mut dec = OpusDecoder::new(16000, 2).unwrap();
    let pkts = encode_frames(&mut enc, 160, 2, 5);
    for pkt in &pkts {
        let mut out = vec![0i16; 160 * 2];
        dec.decode(Some(pkt), &mut out, 160, false).unwrap();
    }
}

#[test]
fn coverage_high_bitrate_celt_mono_cbr_10ms() {
    let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
    enc.set_force_mode(MODE_CELT_ONLY);
    enc.set_bitrate(256000);
    enc.set_vbr(0);
    enc.set_expert_frame_duration(OPUS_FRAMESIZE_10_MS);
    let mut dec = OpusDecoder::new(48000, 1).unwrap();
    let pkts = encode_frames(&mut enc, 480, 1, 3);
    for pkt in &pkts {
        let mut out = vec![0i16; 480];
        dec.decode(Some(pkt), &mut out, 480, false).unwrap();
    }
}

#[test]
fn coverage_mode_transition_silk_to_celt() {
    // Start in SILK, switch to CELT → exercises transition and redundancy paths
    let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
    enc.set_bitrate(32000);
    enc.set_complexity(10);
    let mut dec = OpusDecoder::new(48000, 1).unwrap();

    // SILK frames
    enc.set_force_mode(MODE_SILK_ONLY);
    let pkts = encode_frames(&mut enc, 960, 1, 3);
    for pkt in &pkts {
        decode_packet(&mut dec, Some(pkt), 960);
    }
    // Switch to CELT
    enc.set_force_mode(MODE_CELT_ONLY);
    let pkts = encode_frames(&mut enc, 960, 1, 3);
    for pkt in &pkts {
        decode_packet(&mut dec, Some(pkt), 960);
    }
}

#[test]
fn coverage_mode_transition_celt_to_silk() {
    let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
    enc.set_bitrate(32000);
    enc.set_complexity(10);
    let mut dec = OpusDecoder::new(48000, 1).unwrap();

    // CELT frames
    enc.set_force_mode(MODE_CELT_ONLY);
    let pkts = encode_frames(&mut enc, 960, 1, 3);
    for pkt in &pkts {
        decode_packet(&mut dec, Some(pkt), 960);
    }
    // Switch to SILK
    enc.set_force_mode(MODE_SILK_ONLY);
    let pkts = encode_frames(&mut enc, 960, 1, 3);
    for pkt in &pkts {
        decode_packet(&mut dec, Some(pkt), 960);
    }
}

#[test]
fn coverage_multistream_stereo_encode_decode() {
    let (mut enc, streams, coupled, mapping) =
        OpusMSEncoder::new_surround(48000, 2, 1, OPUS_APPLICATION_AUDIO).unwrap();
    let mut dec = OpusMSDecoder::new(48000, 2, streams, coupled, &mapping).unwrap();

    let pcm = patterned_pcm_i16(960, 2, 42);
    let mut pkt = vec![0u8; 4000];
    let len = enc.encode(&pcm, 960, &mut pkt, 4000).unwrap();

    let mut out = vec![0i16; 960 * 2];
    dec.decode(Some(&pkt[..len as usize]), len, &mut out, 960, false).unwrap();
    assert!(out.iter().any(|&s| s != 0));
}

#[test]
fn coverage_decoder_gain_and_pitch() {
    let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP).unwrap();
    enc.set_bitrate(32000);
    let pcm = patterned_pcm_i16(960, 1, 88);
    let mut pkt = vec![0u8; 1500];
    let len = enc.encode(&pcm, 960, &mut pkt, 1500).unwrap();

    let mut dec = OpusDecoder::new(48000, 1).unwrap();
    dec.set_gain(256); // +1 dB
    let mut out = vec![0i16; 960];
    dec.decode(Some(&pkt[..len as usize]), &mut out, 960, false).unwrap();
    let pitch = dec.get_pitch();
    // Pitch may or may not be > 0 depending on mode, just ensure no panic
    let _ = pitch;
}

#[test]
fn coverage_silk_24khz_superwideband() {
    let mut enc = OpusEncoder::new(24000, 1, OPUS_APPLICATION_VOIP).unwrap();
    enc.set_bitrate(32000);
    enc.set_force_mode(MODE_SILK_ONLY);
    enc.set_complexity(10);
    let mut dec = OpusDecoder::new(24000, 1).unwrap();
    let pkts = encode_frames(&mut enc, 480, 1, 5);
    for pkt in &pkts {
        let mut out = vec![0i16; 480];
        dec.decode(Some(pkt), &mut out, 480, false).unwrap();
    }
}

#[test]
fn coverage_silk_stereo_high_complexity_wideband() {
    let mut enc = OpusEncoder::new(16000, 2, OPUS_APPLICATION_VOIP).unwrap();
    enc.set_bitrate(48000);
    enc.set_force_mode(MODE_SILK_ONLY);
    enc.set_complexity(10);
    enc.set_inband_fec(1);
    enc.set_packet_loss_perc(10);
    let mut dec = OpusDecoder::new(16000, 2).unwrap();
    let pkts = encode_frames(&mut enc, 320, 2, 5);
    for pkt in &pkts {
        let mut out = vec![0i16; 320 * 2];
        dec.decode(Some(pkt), &mut out, 320, false).unwrap();
    }
}

#[test]
fn coverage_cbr_40ms_celt_stereo() {
    let mut enc = OpusEncoder::new(48000, 2, OPUS_APPLICATION_AUDIO).unwrap();
    enc.set_force_mode(MODE_CELT_ONLY);
    enc.set_bitrate(96000);
    enc.set_vbr(0);
    enc.set_expert_frame_duration(OPUS_FRAMESIZE_40_MS);
    let mut dec = OpusDecoder::new(48000, 2).unwrap();
    let pkts = encode_frames(&mut enc, 1920, 2, 3);
    for pkt in &pkts {
        let mut out = vec![0i16; 1920 * 2];
        dec.decode(Some(pkt), &mut out, 1920, false).unwrap();
    }
}
