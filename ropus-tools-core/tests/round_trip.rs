//! End-to-end smoke test for the core library.
//!
//! Calls `commands::encode` and `commands::decode` as library functions (no
//! subprocess) and checks the round-tripped WAV tracks the input fixture
//! sample-for-sample. The fixture is already 48 kHz so the encoder's resample
//! step is identity; alignment reduces to truncating both buffers to their
//! shared length (decode has already applied pre_skip). If anyone breaks the
//! encode → decode pipeline (Ogg framing, OpusHead, pre_skip, gain
//! application, channel handling, anything), the SNR collapses and this test
//! fails.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use ropus_tools_core::commands;
use ropus_tools_core::options::{DecodeOptions, EncodeOptions};

/// Parse a 16-bit PCM WAV (mono or stereo) into interleaved i16 samples and
/// return (samples, sample_rate, channels). Assumes the canonical RIFF/fmt/data
/// layout emitted by our own WAV writer — good enough for test fixtures.
fn read_pcm16_wav(path: &std::path::Path) -> (Vec<i16>, u32, u16) {
    let bytes = std::fs::read(path).expect("read WAV");
    assert!(bytes.len() >= 44, "WAV shorter than canonical 44-byte header");
    let channels = u16::from_le_bytes([bytes[22], bytes[23]]);
    let sample_rate = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    // data chunk starts at offset 36 with "data" + u32 size, samples follow at 44.
    assert_eq!(&bytes[36..40], b"data", "expected canonical data chunk at 36");
    let data_len = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]) as usize;
    let end = (44 + data_len).min(bytes.len());
    let raw = &bytes[44..end];
    let mut samples = Vec::with_capacity(raw.len() / 2);
    for chunk in raw.chunks_exact(2) {
        samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
    }
    (samples, sample_rate, channels)
}

#[test]
fn encode_then_decode_48k_sine_round_trips_with_snr_above_20_db() {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ropus-tools-core has a parent dir (the workspace root)")
        .to_path_buf();
    let input_wav = workspace.join("tests/vectors/48k_sine1k_loud.wav");

    // Per-run unique temp names so parallel test invocations don't race on the
    // same paths.
    let nonce = format!(
        "{}_{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let tmp_opus = std::env::temp_dir().join(format!("ropus_tools_core_rt_{nonce}.opus"));
    let tmp_wav = std::env::temp_dir().join(format!("ropus_tools_core_rt_{nonce}.wav"));

    // Skip cleanly if the test vector isn't present (so the test is portable
    // to checkouts without the vectors directory populated). Loud eprintln so
    // a silent skip is visible in CI logs rather than looking like a pass.
    if !input_wav.exists() {
        eprintln!(
            "SKIPPING round_trip: test vector {input_wav:?} not present \
             (no SNR was computed, this is not a real pass)"
        );
        return;
    }

    let enc_opts = EncodeOptions {
        input: input_wav.clone(),
        output: Some(tmp_opus.clone()),
        bitrate: Some(64_000),
        complexity: None,
        application: ropus_tools_core::Application::Audio,
        vbr: true,
        vendor: "ropus-tools-core-test".to_string(),
        comments: Vec::new(),
    };
    commands::encode(enc_opts).expect("encode");

    let dec_opts = DecodeOptions {
        input: tmp_opus.clone(),
        output: Some(tmp_wav.clone()),
    };
    commands::decode(dec_opts).expect("decode");

    let out_meta = std::fs::metadata(&tmp_wav).expect("stat output");
    assert!(out_meta.len() > 44, "output WAV is empty or header-only");

    let (in_samples, in_sr, in_ch) = read_pcm16_wav(&input_wav);
    let (out_samples, out_sr, out_ch) = read_pcm16_wav(&tmp_wav);
    assert_eq!(in_sr, 48_000, "test fixture must be 48 kHz");
    assert_eq!(out_sr, 48_000, "output WAV must be 48 kHz");
    assert_eq!(
        in_ch, out_ch,
        "channel count mismatch between input and round-tripped output"
    );

    let n = in_samples.len().min(out_samples.len());
    assert!(n > 0, "no overlapping samples to compare");

    let sig_power: f64 =
        in_samples[..n].iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / n as f64;
    let noise_power: f64 = in_samples[..n]
        .iter()
        .zip(&out_samples[..n])
        .map(|(&a, &b)| ((a as f64) - (b as f64)).powi(2))
        .sum::<f64>()
        / n as f64;
    let snr_db = 10.0 * (sig_power / noise_power.max(f64::MIN_POSITIVE)).log10();

    // 1 kHz sine at 64 kbps Audio VBR comfortably clears 40 dB through libopus;
    // 20 dB is a generous floor that still catches real breakage (gain drift,
    // channel duplication, sample-level corruption that previously hid behind
    // total-power equivalence).
    assert!(
        snr_db > 20.0,
        "round-trip SNR {snr_db:.2} dB is below 20 dB floor \
         (sig_power={sig_power}, noise_power={noise_power}, n={n})"
    );

    // Best-effort cleanup of this run's outputs.
    let _ = std::fs::remove_file(&tmp_opus);
    let _ = std::fs::remove_file(&tmp_wav);
}
