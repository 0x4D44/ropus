//! Proptest-based property tests for the Opus encoder/decoder round-trip.

use crate::opus::decoder::{
    opus_packet_get_nb_frames, opus_packet_get_samples_per_frame, MODE_CELT_ONLY, MODE_HYBRID,
    MODE_SILK_ONLY, OPUS_OK, OpusDecoder,
};
use crate::opus::encoder::{
    OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_APPLICATION_VOIP,
    OpusEncoder,
};
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
                // Energy should decrease (or at least not increase significantly)
                // Use generous bound: energy_20 < energy_5 + small tolerance
                // If energy_at_5 is already near zero, both are fine
                prop_assert!(
                    e < energy_at_5 + 100.0,
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
