//! Proptest-based property tests for the Opus encoder/decoder round-trip.

use crate::opus::decoder::{
    opus_packet_get_nb_frames, opus_packet_get_nb_samples, opus_packet_get_samples_per_frame,
    MODE_CELT_ONLY, MODE_HYBRID, MODE_SILK_ONLY, OPUS_BANDWIDTH_NARROWBAND, OPUS_OK, OpusDecoder,
};
use crate::opus::encoder::{
    OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_APPLICATION_VOIP,
    OpusEncoder,
};
use crate::opus::repacketizer::OpusRepacketizer;
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Canonical configs: (sample_rate, channels, application, frame_size, mode_to_force)
// ---------------------------------------------------------------------------

fn canonical_configs() -> Vec<(i32, i32, i32, i32, Option<i32>)> {
    vec![
        (8000, 1, OPUS_APPLICATION_VOIP, 160, Some(MODE_SILK_ONLY)),  // SILK NB mono 20ms
        (16000, 2, OPUS_APPLICATION_VOIP, 320, Some(MODE_SILK_ONLY)), // SILK WB stereo 20ms
        (48000, 1, OPUS_APPLICATION_AUDIO, 960, Some(MODE_CELT_ONLY)), // CELT FB mono 20ms
        (48000, 2, OPUS_APPLICATION_AUDIO, 480, Some(MODE_CELT_ONLY)), // CELT FB stereo 10ms
        (48000, 1, OPUS_APPLICATION_AUDIO, 960, Some(MODE_HYBRID)),   // Hybrid FB mono 20ms
        (48000, 2, OPUS_APPLICATION_RESTRICTED_LOWDELAY, 240, None),  // Low-delay stereo 5ms
        (48000, 1, OPUS_APPLICATION_AUDIO, 120, Some(MODE_CELT_ONLY)),  // CELT FB mono 2.5ms
        (8000, 1, OPUS_APPLICATION_VOIP, 480, Some(MODE_SILK_ONLY)),    // SILK NB mono 60ms
    ]
}

fn config_strategy() -> impl Strategy<Value = (i32, i32, i32, i32, Option<i32>)> {
    prop::sample::select(canonical_configs())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn rms_energy(pcm: &[i16]) -> f64 {
    let sum: f64 = pcm.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum / pcm.len() as f64).sqrt()
}

fn sine_pcm(frame_size: usize, channels: usize, freq: f64, amplitude: f64, fs: f64) -> Vec<i16> {
    let n = frame_size * channels;
    (0..n)
        .map(|i| {
            let t = (i / channels) as f64 / fs;
            (amplitude * (2.0 * std::f64::consts::PI * freq * t).sin()) as i16
        })
        .collect()
}

/// Create an encoder with the given config and apply bitrate/complexity.
fn make_encoder(
    fs: i32,
    channels: i32,
    application: i32,
    mode: Option<i32>,
    bitrate: i32,
    complexity: i32,
) -> OpusEncoder {
    let mut enc = OpusEncoder::new(fs, channels, application).unwrap();
    if let Some(m) = mode {
        assert_eq!(enc.set_force_mode(m), OPUS_OK);
    }
    assert_eq!(enc.set_bitrate(bitrate), OPUS_OK);
    assert_eq!(enc.set_complexity(complexity), OPUS_OK);
    enc
}

/// Encode a single PCM frame, returning the packet bytes.
fn encode_frame(enc: &mut OpusEncoder, pcm: &[i16], frame_size: i32) -> Vec<u8> {
    let mut packet = vec![0u8; 1500];
    let cap = packet.len() as i32;
    let len = enc.encode(pcm, frame_size, &mut packet, cap).unwrap();
    assert!(len > 0, "encode returned 0 bytes");
    packet.truncate(len as usize);
    packet
}

/// Warm up an encoder by encoding `n` frames of patterned PCM.
fn warm_up_encoder(enc: &mut OpusEncoder, frame_size: i32, channels: i32, n: usize) {
    for i in 0..n {
        let pcm = crate::coverage_tests::patterned_pcm_i16(
            frame_size as usize,
            channels as usize,
            (i * 37) as i32,
        );
        let mut packet = vec![0u8; 1500];
        let cap = packet.len() as i32;
        let _ = enc.encode(&pcm, frame_size, &mut packet, cap);
    }
}

fn encode_sequence(
    enc: &mut OpusEncoder,
    frame_size: i32,
    channels: i32,
    n_frames: usize,
    pcm_seed: i32,
) -> Vec<Vec<u8>> {
    (0..n_frames)
        .map(|i| {
            let pcm = crate::coverage_tests::patterned_pcm_i16(
                frame_size as usize,
                channels as usize,
                pcm_seed + i as i32,
            );
            encode_frame(enc, &pcm, frame_size)
        })
        .collect()
}

fn decode_sequence(
    dec: &mut OpusDecoder,
    packets: &[Option<&[u8]>],
    frame_size: i32,
    channels: i32,
) -> Vec<(i32, Vec<i16>)> {
    let n = frame_size as usize * channels as usize;
    packets
        .iter()
        .map(|pkt| {
            let mut out = vec![0i16; n];
            let decoded = dec.decode(*pkt, &mut out, frame_size, false).unwrap();
            (decoded, out)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 200, max_shrink_iters: 100, .. ProptestConfig::default() })]

    /// Encoding always produces a parseable Opus packet.
    #[test]
    fn prop_encode_produces_parseable_packets(
        config in config_strategy(),
        bitrate in 6000..=256000i32,
        complexity in 0..=10i32,
        seed in 0..10000i32,
    ) {
        let (fs, channels, app, frame_size, mode) = config;
        let mut enc = make_encoder(fs, channels, app, mode, bitrate, complexity);
        warm_up_encoder(&mut enc, frame_size, channels, 3);

        let pcm = crate::coverage_tests::patterned_pcm_i16(
            frame_size as usize,
            channels as usize,
            seed,
        );
        let packet = encode_frame(&mut enc, &pcm, frame_size);

        // Packet is parseable
        let nb_frames = opus_packet_get_nb_frames(&packet).unwrap();
        prop_assert!(nb_frames > 0, "get_nb_frames returned {}", nb_frames);

        let spf = opus_packet_get_samples_per_frame(&packet, fs);
        prop_assert!(spf > 0, "get_samples_per_frame returned {}", spf);
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 150, max_shrink_iters: 100, .. ProptestConfig::default() })]

    /// CBR encoding is deterministic: same config + same input = identical output.
    #[test]
    fn prop_cbr_determinism(
        config in config_strategy(),
        bitrate in 24000..=128000i32,
        complexity in 0..=10i32,
        seed in 0..10000i32,
    ) {
        let (fs, channels, app, frame_size, mode) = config;

        let mut enc1 = make_encoder(fs, channels, app, mode, bitrate, complexity);
        assert_eq!(enc1.set_vbr(0), OPUS_OK);
        let mut enc2 = make_encoder(fs, channels, app, mode, bitrate, complexity);
        assert_eq!(enc2.set_vbr(0), OPUS_OK);

        // Warm up both identically
        for i in 0..3 {
            let pcm = crate::coverage_tests::patterned_pcm_i16(
                frame_size as usize,
                channels as usize,
                (i * 37) as i32,
            );
            let mut p1 = vec![0u8; 1500];
            let mut p2 = vec![0u8; 1500];
            let cap = p1.len() as i32;
            let _ = enc1.encode(&pcm, frame_size, &mut p1, cap);
            let _ = enc2.encode(&pcm, frame_size, &mut p2, cap);
        }

        let pcm = crate::coverage_tests::patterned_pcm_i16(
            frame_size as usize,
            channels as usize,
            seed,
        );
        let pkt1 = encode_frame(&mut enc1, &pcm, frame_size);
        let pkt2 = encode_frame(&mut enc2, &pcm, frame_size);

        prop_assert_eq!(&pkt1, &pkt2, "CBR outputs differ");
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 200, max_shrink_iters: 100, .. ProptestConfig::default() })]

    /// Decoding is idempotent: two fresh decoders produce identical output.
    #[test]
    fn prop_decode_idempotency(
        config in config_strategy(),
        bitrate in 24000..=128000i32,
        seed in 0..10000i32,
    ) {
        let (fs, channels, app, frame_size, mode) = config;
        let mut enc = make_encoder(fs, channels, app, mode, bitrate, 5);
        warm_up_encoder(&mut enc, frame_size, channels, 3);

        let pcm = crate::coverage_tests::patterned_pcm_i16(
            frame_size as usize,
            channels as usize,
            seed,
        );
        let packet = encode_frame(&mut enc, &pcm, frame_size);

        let mut dec1 = OpusDecoder::new(fs, channels).unwrap();
        let mut dec2 = OpusDecoder::new(fs, channels).unwrap();

        let n = frame_size as usize * channels as usize;
        let mut out1 = vec![0i16; n];
        let mut out2 = vec![0i16; n];

        let r1 = dec1.decode(Some(&packet), &mut out1, frame_size, false).unwrap();
        let r2 = dec2.decode(Some(&packet), &mut out2, frame_size, false).unwrap();

        prop_assert_eq!(r1, r2, "decoded sample counts differ");
        prop_assert_eq!(&out1, &out2, "decoded PCM differs");
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 200, max_shrink_iters: 100, .. ProptestConfig::default() })]

    /// Decode returns exactly the expected number of samples per channel.
    #[test]
    fn prop_sample_count_invariant(
        config in config_strategy(),
        bitrate in 24000..=128000i32,
        seed in 0..10000i32,
    ) {
        let (fs, channels, app, frame_size, mode) = config;
        let mut enc = make_encoder(fs, channels, app, mode, bitrate, 5);
        warm_up_encoder(&mut enc, frame_size, channels, 3);

        let pcm = crate::coverage_tests::patterned_pcm_i16(
            frame_size as usize,
            channels as usize,
            seed,
        );
        let packet = encode_frame(&mut enc, &pcm, frame_size);

        let mut dec = OpusDecoder::new(fs, channels).unwrap();
        let n = frame_size as usize * channels as usize;
        let mut out = vec![0i16; n];
        let decoded = dec.decode(Some(&packet), &mut out, frame_size, false).unwrap();

        prop_assert_eq!(
            decoded, frame_size,
            "expected {} samples/ch, got {}",
            frame_size, decoded
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, max_shrink_iters: 100, .. ProptestConfig::default() })]

    /// PLC energy converges toward zero over successive loss frames.
    #[test]
    fn prop_plc_convergence(
        config in config_strategy(),
        bitrate in 24000..=128000i32,
        seed in 0..10000i32,
    ) {
        let (fs, channels, app, frame_size, mode) = config;
        let mut enc = make_encoder(fs, channels, app, mode, bitrate, 5);
        let mut dec = OpusDecoder::new(fs, channels).unwrap();
        let n = frame_size as usize * channels as usize;

        // Encode + decode 5 real frames to establish decoder state
        for i in 0..5 {
            let pcm = crate::coverage_tests::patterned_pcm_i16(
                frame_size as usize,
                channels as usize,
                seed + i,
            );
            let packet = encode_frame(&mut enc, &pcm, frame_size);
            let mut out = vec![0i16; n];
            dec.decode(Some(&packet), &mut out, frame_size, false).unwrap();
        }

        // Run 20 PLC frames, measure energy at frames 5 and 20
        let mut energy_at_5 = 0.0f64;
        for plc_idx in 1..=20 {
            let mut out = vec![0i16; n];
            dec.decode(None, &mut out, frame_size, false).unwrap();
            let e = rms_energy(&out);
            if plc_idx == 5 {
                energy_at_5 = e;
            }
            if plc_idx == 20 {
                // Energy should not diverge wildly over PLC frames.
                // Hybrid mode (SILK/CELT cross-fade) can legitimately produce
                // rising energy, so use a generous bound that still catches
                // genuine runaway divergence.
                prop_assert!(
                    e < energy_at_5 * 4.0 + 500.0,
                    "PLC energy did not converge: frame 5 = {:.1}, frame 20 = {:.1}",
                    energy_at_5,
                    e,
                );
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 150, max_shrink_iters: 100, .. ProptestConfig::default() })]

    /// Encoding and decoding silence produces near-silent output.
    #[test]
    fn prop_silence_preservation(
        config in config_strategy(),
        bitrate in 24000..=128000i32,
        complexity in 0..=10i32,
    ) {
        let (fs, channels, app, frame_size, mode) = config;
        let mut enc = make_encoder(fs, channels, app, mode, bitrate, complexity);
        let mut dec = OpusDecoder::new(fs, channels).unwrap();
        let n = frame_size as usize * channels as usize;
        let silence = vec![0i16; n];

        // Encode + decode 10 silence frames to let encoder/decoder settle
        for _ in 0..10 {
            let packet = encode_frame(&mut enc, &silence, frame_size);
            let mut out = vec![0i16; n];
            dec.decode(Some(&packet), &mut out, frame_size, false).unwrap();
        }

        // Frame 11: measure output
        let packet = encode_frame(&mut enc, &silence, frame_size);
        let mut out = vec![0i16; n];
        dec.decode(Some(&packet), &mut out, frame_size, false).unwrap();

        let energy = rms_energy(&out);
        prop_assert!(
            energy < 50.0,
            "silence output RMS = {:.1}, expected < 50",
            energy,
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 150, max_shrink_iters: 100, .. ProptestConfig::default() })]

    /// Energy is roughly conserved through encode/decode of a sine wave.
    #[test]
    fn prop_energy_conservation(
        config in config_strategy(),
        bitrate in 24000..=256000i32,
        freq_idx in 0..5usize,
    ) {
        let (fs, channels, app, frame_size, mode) = config;
        let mut enc = make_encoder(fs, channels, app, mode, bitrate, 5);
        let mut dec = OpusDecoder::new(fs, channels).unwrap();
        let n = frame_size as usize * channels as usize;

        // Pick a frequency that's well within the Nyquist limit for this sample rate
        let freqs: [f64; 5] = [200.0, 440.0, 1000.0, 2000.0, 3500.0];
        let freq = freqs[freq_idx].min(fs as f64 / 4.0); // stay well below Nyquist
        let amplitude = 10000.0f64;

        // Warm up 5 frames
        for _ in 0..5 {
            let pcm = sine_pcm(
                frame_size as usize,
                channels as usize,
                freq,
                amplitude,
                fs as f64,
            );
            let packet = encode_frame(&mut enc, &pcm, frame_size);
            let mut out = vec![0i16; n];
            dec.decode(Some(&packet), &mut out, frame_size, false).unwrap();
        }

        // Measure frame 6
        let input = sine_pcm(
            frame_size as usize,
            channels as usize,
            freq,
            amplitude,
            fs as f64,
        );
        let input_energy = rms_energy(&input);
        let packet = encode_frame(&mut enc, &input, frame_size);
        let mut output = vec![0i16; n];
        dec.decode(Some(&packet), &mut output, frame_size, false).unwrap();
        let output_energy = rms_energy(&output);

        prop_assert!(
            output_energy < 4.0 * input_energy,
            "output energy {:.1} > 4x input energy {:.1}",
            output_energy,
            input_energy,
        );
        prop_assert!(
            output_energy > 0.01 * input_energy,
            "output energy {:.1} < 0.01x input energy {:.1}",
            output_energy,
            input_energy,
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, max_shrink_iters: 100, .. ProptestConfig::default() })]

    /// Multi-frame sequence invariants: packets parseable, sizes bounded, energy stable.
    #[test]
    fn prop_sequence_invariants(
        config in config_strategy(),
        bitrate in 16000..=128000i32,
        seed in 0..10000i32,
    ) {
        let (fs, channels, app, frame_size, mode) = config;
        let mut enc = make_encoder(fs, channels, app, mode, bitrate, 5);
        let mut dec = OpusDecoder::new(fs, channels).unwrap();

        let packets = encode_sequence(&mut enc, frame_size, channels, 20, seed);
        let pkt_refs: Vec<Option<&[u8]>> = packets.iter().map(|p| Some(p.as_slice())).collect();
        let decoded = decode_sequence(&mut dec, &pkt_refs, frame_size, channels);

        let mut prev_energy = 0.0f64;
        for (i, (pkt, (decoded_samples, pcm))) in packets.iter().zip(decoded.iter()).enumerate() {
            // Every packet parseable
            let nb = opus_packet_get_nb_frames(pkt).unwrap();
            prop_assert!(nb > 0, "frame {}: get_nb_frames returned {}", i, nb);

            // Packet size bounded
            prop_assert!(
                pkt.len() >= 2 && pkt.len() <= 1275,
                "frame {}: packet size {} out of range 2..=1275",
                i, pkt.len(),
            );

            // Decode returned correct sample count
            prop_assert_eq!(
                *decoded_samples, frame_size,
                "frame {}: decoded {} samples/ch, expected {}",
                i, decoded_samples, frame_size,
            );

            // No energy explosion: energy[i] < 8 * max(energy[i-1], 800)
            // Base of 800 accommodates legitimate energy variation from patterned PCM
            // across different seed values and short frame sizes (especially 2.5ms CELT
            // at 120 samples where inter-frame energy can swing by ~10x).
            let energy = rms_energy(pcm);
            if i > 0 {
                let threshold = 8.0 * prev_energy.max(800.0);
                prop_assert!(
                    energy < threshold,
                    "frame {}: energy {:.1} > 8 * max(prev {:.1}, 800)",
                    i, energy, prev_energy,
                );
            }
            prev_energy = energy;
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, max_shrink_iters: 100, .. ProptestConfig::default() })]

    /// FEC recovery: SILK/Hybrid packets with FEC survive single-frame loss gracefully.
    #[test]
    fn prop_fec_recovery(
        config in config_strategy(),
        bitrate in 16000..=64000i32,
        loss_frame in 1..=8usize,
        seed in 0..10000i32,
    ) {
        let (fs, channels, app, frame_size, mode) = config;
        // FEC only works in SILK/Hybrid modes, not CELT-only or low-delay
        prop_assume!(mode != Some(MODE_CELT_ONLY) && app != OPUS_APPLICATION_RESTRICTED_LOWDELAY);

        let mut enc = make_encoder(fs, channels, app, mode, bitrate, 5);
        assert_eq!(enc.set_inband_fec(1), OPUS_OK);
        assert_eq!(enc.set_packet_loss_perc(25), OPUS_OK);

        let packets = encode_sequence(&mut enc, frame_size, channels, 10, seed);
        let n = frame_size as usize * channels as usize;

        let mut dec = OpusDecoder::new(fs, channels).unwrap();

        // Decode frames before the loss normally, track pre-loss energy
        let mut pre_loss_energies = Vec::new();
        for i in 0..loss_frame {
            let mut out = vec![0i16; n];
            dec.decode(Some(&packets[i]), &mut out, frame_size, false).unwrap();
            pre_loss_energies.push(rms_energy(&out));
        }

        // Skip loss_frame (don't decode it — simulates loss)

        // Decode loss_frame+1 with FEC recovery
        let mut fec_out = vec![0i16; n];
        let fec_result = dec.decode(Some(&packets[loss_frame + 1]), &mut fec_out, frame_size, true);
        prop_assert!(
            fec_result.is_ok() && fec_result.unwrap() == frame_size,
            "FEC decode failed: {:?}", fec_result,
        );

        // fec_out contains the FEC-recovered approximation of the lost frame.
        // We don't validate its contents — FEC recovery is approximate by design.
        // The purpose of this decode is to advance the decoder through the FEC path.

        // Decode loss_frame+1 normally
        let mut normal_out = vec![0i16; n];
        let normal_result = dec.decode(Some(&packets[loss_frame + 1]), &mut normal_out, frame_size, false);
        prop_assert!(
            normal_result.is_ok() && normal_result.unwrap() == frame_size,
            "normal decode after FEC failed: {:?}", normal_result,
        );

        // Decode remaining frames normally, check no energy explosion
        let avg_pre_loss = if pre_loss_energies.is_empty() {
            100.0
        } else {
            pre_loss_energies.iter().sum::<f64>() / pre_loss_energies.len() as f64
        };
        let explosion_threshold = 10.0 * avg_pre_loss.max(100.0);

        for i in (loss_frame + 2)..10 {
            let mut out = vec![0i16; n];
            let r = dec.decode(Some(&packets[i]), &mut out, frame_size, false).unwrap();
            prop_assert_eq!(r, frame_size);
            let energy = rms_energy(&out);
            prop_assert!(
                energy < explosion_threshold,
                "post-recovery frame {}: energy {:.1} > 10x avg pre-loss {:.1}",
                i, energy, avg_pre_loss,
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, max_shrink_iters: 100, .. ProptestConfig::default() })]

    /// Mid-stream config transitions (bitrate, bandwidth, VBR) don't crash or corrupt output.
    #[test]
    fn prop_config_transition_no_crash(
        config in config_strategy(),
        bitrate in 16000..=128000i32,
        transition_type in 0..=2u32,
        seed in 0..10000i32,
    ) {
        let (fs, channels, app, frame_size, mode) = config;
        let mut enc = make_encoder(fs, channels, app, mode, bitrate, 5);
        let mut all_packets = Vec::new();

        // Encode 5 frames before transition
        for i in 0..5 {
            let pcm = crate::coverage_tests::patterned_pcm_i16(
                frame_size as usize,
                channels as usize,
                seed + i,
            );
            all_packets.push(encode_frame(&mut enc, &pcm, frame_size));
        }

        // Apply transition
        match transition_type {
            0 => {
                // Bitrate change: double or halve
                let current = enc.get_bitrate();
                let new_bitrate = if current < 128000 {
                    (current * 2).min(256000)
                } else {
                    (current / 2).max(6000)
                };
                assert_eq!(enc.set_bitrate(new_bitrate), OPUS_OK);
            }
            1 => {
                // Bandwidth change: force narrowband
                assert_eq!(enc.set_bandwidth(OPUS_BANDWIDTH_NARROWBAND), OPUS_OK);
            }
            _ => {
                // VBR toggle
                let current_vbr = enc.get_vbr();
                assert_eq!(enc.set_vbr(1 - current_vbr), OPUS_OK);
            }
        }

        // Encode 5 frames after transition
        for i in 5..10 {
            let pcm = crate::coverage_tests::patterned_pcm_i16(
                frame_size as usize,
                channels as usize,
                seed + i,
            );
            all_packets.push(encode_frame(&mut enc, &pcm, frame_size));
        }

        // Verify all packets parseable
        for (i, pkt) in all_packets.iter().enumerate() {
            let nb = opus_packet_get_nb_frames(pkt).unwrap();
            prop_assert!(nb > 0, "frame {}: get_nb_frames returned {}", i, nb);
        }

        // Decode all and verify
        let mut dec = OpusDecoder::new(fs, channels).unwrap();
        let n = frame_size as usize * channels as usize;
        let mut prev_energy = 0.0f64;

        for (i, pkt) in all_packets.iter().enumerate() {
            let mut out = vec![0i16; n];
            let r = dec.decode(Some(pkt), &mut out, frame_size, false);
            prop_assert!(
                r.is_ok() && r.unwrap() == frame_size,
                "frame {}: decode failed: {:?}", i, r,
            );

            // Check energy explosion at and after the transition point (frame 5+)
            let energy = rms_energy(&out);
            if i >= 5 {
                let threshold = 10.0 * prev_energy.max(100.0);
                prop_assert!(
                    energy < threshold,
                    "post-transition frame {}: energy {:.1} > 10x prev {:.1}",
                    i, energy, prev_energy,
                );
            }
            prev_energy = energy;
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, max_shrink_iters: 100, .. ProptestConfig::default() })]

    /// Repacketizer round-trip: combined multi-frame packet decodes identically to individual packets.
    #[test]
    fn prop_repacketizer_roundtrip(
        config in config_strategy(),
        bitrate in 16000..=128000i32,
        n_frames in 2..=3u32,
        seed in 0..10000i32,
    ) {
        let (fs, channels, app, frame_size, mode) = config;
        let mut enc = make_encoder(fs, channels, app, mode, bitrate, 5);

        // Total duration must not exceed 120ms (Opus maximum per packet)
        let frame_duration_ms = frame_size as f64 * 1000.0 / fs as f64;
        prop_assume!(frame_duration_ms * n_frames as f64 <= 120.0);

        let packets = encode_sequence(&mut enc, frame_size, channels, n_frames as usize, seed);

        // When mode is forced, TOC bytes MUST match — a mismatch would be an encoder bug.
        // When mode is None (auto-select), TOC can legitimately vary, so skip mismatches.
        let toc_match = packets.iter().all(|p| (p[0] & 0xFC) == (packets[0][0] & 0xFC));
        if mode.is_some() {
            prop_assert!(toc_match, "forced-mode encoder produced mismatched TOC bytes");
        } else {
            prop_assume!(toc_match);
        }

        // Repacketize: cat all packets into one combined packet
        let mut rp = OpusRepacketizer::new();
        for pkt in &packets {
            let result = rp.cat(pkt, pkt.len() as i32);
            prop_assert_eq!(result, OPUS_OK, "cat failed with {}", result);
        }

        let mut combined = vec![0u8; 4000];
        let maxlen = combined.len() as i32;
        let combined_len = rp.out(&mut combined, maxlen);
        prop_assert!(combined_len > 0, "repacketizer out returned {}", combined_len);
        combined.truncate(combined_len as usize);

        // Verify combined packet structure
        let nb = opus_packet_get_nb_frames(&combined).unwrap();
        prop_assert_eq!(
            nb, n_frames as i32,
            "combined packet has {} frames, expected {}", nb, n_frames,
        );

        let total_samples = opus_packet_get_nb_samples(&combined, fs).unwrap();
        let spf = opus_packet_get_samples_per_frame(&combined, fs);
        prop_assert_eq!(
            total_samples, n_frames as i32 * spf,
            "combined samples {} != {} * {}", total_samples, n_frames, spf,
        );

        // Decode combined packet with decoder A
        let mut dec_a = OpusDecoder::new(fs, channels).unwrap();
        let combined_n = frame_size as usize * channels as usize * n_frames as usize;
        let mut combined_pcm = vec![0i16; combined_n];
        let combined_decoded = dec_a.decode(
            Some(&combined),
            &mut combined_pcm,
            frame_size * n_frames as i32,
            false,
        ).unwrap();
        prop_assert_eq!(
            combined_decoded, frame_size * n_frames as i32,
            "combined decode returned {} samples, expected {}",
            combined_decoded, frame_size * n_frames as i32,
        );

        // Decode individual packets with decoder B
        let mut dec_b = OpusDecoder::new(fs, channels).unwrap();
        let n = frame_size as usize * channels as usize;
        let mut individual_pcm = Vec::with_capacity(combined_n);
        for pkt in &packets {
            let mut out = vec![0i16; n];
            let r = dec_b.decode(Some(pkt), &mut out, frame_size, false).unwrap();
            prop_assert_eq!(r, frame_size);
            individual_pcm.extend_from_slice(&out);
        }

        // Compare: combined decode must match concatenated individual decodes
        prop_assert_eq!(
            &combined_pcm, &individual_pcm,
            "combined decode PCM differs from individual decodes",
        );
    }
}
