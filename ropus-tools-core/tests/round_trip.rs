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
        vbr_constraint: false,
        signal: ropus_tools_core::Signal::Auto,
        frame_duration: ropus_tools_core::FrameDuration::Ms20,
        expect_loss: 0,
        downmix_to_mono: false,
        serial: None,
        picture_path: None,
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

/// Minimal Ogg packet walk to pull the OpusTags packet out of a freshly-encoded
/// file. Uses the `ogg` crate's `PacketReader` the same way `commands::decode`
/// does: packet 0 is OpusHead, packet 1 is OpusTags.
fn read_opus_tags_from_file(path: &std::path::Path) -> ropus_tools_core::container::ogg::OpusTags {
    use ogg::reading::PacketReader;
    use ropus_tools_core::container::ogg::OpusTags;

    let file = std::fs::File::open(path).expect("open opus");
    let mut reader = PacketReader::new(std::io::BufReader::new(file));
    let _head = reader
        .read_packet()
        .expect("read OpusHead")
        .expect("packet 0");
    let tags_pkt = reader
        .read_packet()
        .expect("read OpusTags")
        .expect("packet 1");
    OpusTags::parse(&tags_pkt.data).expect("parse tags")
}

#[test]
fn encode_with_metadata_flags_round_trips_tags() {
    // Exercises the Step 3 metadata plumbing end-to-end: pass
    // `--artist X --title Y` equivalents through `EncodeOptions.comments`,
    // encode to a real Ogg file, then parse the OpusTags page back out.
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace parent")
        .to_path_buf();
    let input_wav = workspace.join("tests/vectors/48k_sine1k_loud.wav");

    if !input_wav.exists() {
        eprintln!(
            "SKIPPING encode_with_metadata_flags_round_trips_tags: \
             test vector {input_wav:?} not present"
        );
        return;
    }

    let nonce = format!(
        "{}_{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let tmp_opus = std::env::temp_dir().join(format!("ropus_tags_rt_{nonce}.opus"));

    let enc_opts = EncodeOptions {
        input: input_wav,
        output: Some(tmp_opus.clone()),
        bitrate: Some(64_000),
        complexity: None,
        application: ropus_tools_core::Application::Audio,
        vbr: true,
        vbr_constraint: false,
        signal: ropus_tools_core::Signal::Auto,
        frame_duration: ropus_tools_core::FrameDuration::Ms20,
        expect_loss: 0,
        downmix_to_mono: false,
        serial: None,
        picture_path: None,
        vendor: "ropus-tools-core-test".to_string(),
        comments: vec![
            "ARTIST=X".to_string(),
            "TITLE=Y".to_string(),
        ],
    };
    commands::encode(enc_opts).expect("encode with tags");

    let tags = read_opus_tags_from_file(&tmp_opus);
    assert_eq!(tags.get("ARTIST"), Some("X"), "artist round-tripped");
    assert_eq!(tags.get("TITLE"), Some("Y"), "title round-tripped");
    assert_eq!(
        tags.vendor, "ropus-tools-core-test",
        "vendor round-tripped"
    );

    let _ = std::fs::remove_file(&tmp_opus);
}

#[test]
fn encode_with_custom_serial_writes_that_serial_to_ogg_pages() {
    // --serial N must override the hardcoded OGG_STREAM_SERIAL. Walk the
    // raw bytes of the first OggS page and verify the serial field (offset
    // 14..18 inside the page header, little-endian per RFC 3533) matches
    // what we asked for.
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace parent")
        .to_path_buf();
    let input_wav = workspace.join("tests/vectors/48k_sine1k_loud.wav");
    if !input_wav.exists() {
        eprintln!("SKIPPING encode_with_custom_serial: test vector missing");
        return;
    }

    let nonce = format!(
        "{}_{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let tmp_opus = std::env::temp_dir().join(format!("ropus_serial_{nonce}.opus"));

    let enc_opts = EncodeOptions {
        input: input_wav,
        output: Some(tmp_opus.clone()),
        bitrate: Some(64_000),
        complexity: None,
        application: ropus_tools_core::Application::Audio,
        vbr: true,
        vbr_constraint: false,
        signal: ropus_tools_core::Signal::Auto,
        frame_duration: ropus_tools_core::FrameDuration::Ms20,
        expect_loss: 0,
        downmix_to_mono: false,
        serial: Some(0xAABB_CCDD),
        picture_path: None,
        vendor: "ropus-tools-core-test".to_string(),
        comments: Vec::new(),
    };
    commands::encode(enc_opts).expect("encode with custom serial");

    let bytes = std::fs::read(&tmp_opus).expect("read opus");
    assert!(bytes.len() >= 27, "output shorter than one Ogg page header");
    assert_eq!(&bytes[..4], b"OggS", "first 4 bytes must be OggS");
    let serial = u32::from_le_bytes([bytes[14], bytes[15], bytes[16], bytes[17]]);
    assert_eq!(serial, 0xAABB_CCDD, "custom --serial must appear on page");

    let _ = std::fs::remove_file(&tmp_opus);
}

/// Write a canonical 16-bit PCM mono WAV with a 1 kHz sine at 48 kHz. Minimal
/// RIFF header matching the layout `read_pcm16_wav` above already parses.
fn write_sine_wav(path: &std::path::Path, seconds: u32, freq_hz: f32) {
    let sr: u32 = 48_000;
    let channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let byte_rate = sr * u32::from(channels) * u32::from(bits_per_sample) / 8;
    let block_align = channels * bits_per_sample / 8;
    let num_samples = sr * seconds;
    let data_size = num_samples * u32::from(block_align);
    let riff_size = 36 + data_size;

    let mut out = Vec::with_capacity((44 + data_size) as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sr.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());

    let two_pi = std::f32::consts::TAU;
    for n in 0..num_samples {
        let t = n as f32 / sr as f32;
        let s = (two_pi * freq_hz * t).sin() * 0.6; // -4.4 dBFS peak
        let q = (s * 32767.0) as i16;
        out.extend_from_slice(&q.to_le_bytes());
    }

    std::fs::write(path, &out).expect("write synthetic WAV");
}

/// Regression test for the packet-buffer blocker: encode 5 s of 1 kHz sine at
/// `--framesize 120 --bitrate 128000 --hard-cbr`. With the old 1275-byte
/// packet buffer, libopus' repacketiser clamps each 6-sub-frame code-3 packet
/// to ~1275 bytes, starving the CBR target and producing a file roughly
/// one-sixth the expected size. Raising `MAX_PACKET_BYTES` to
/// `MAX_OPUS_FRAME_BYTES * MAX_SUBFRAMES_PER_PACKET` restores the full budget.
///
/// Expected size: 128 kbps × 5 s = 640 000 bits = 80 000 bytes of Opus data.
/// The OpusHead and OpusTags pages plus Ogg framing add a few hundred bytes.
/// We accept ±10% (72 000..88 000 bytes of *encoded-data total*), which comfortably
/// catches a 6× undersize but tolerates normal framing overhead.
#[test]
fn encode_framesize_120_hard_cbr_128k_fills_packet_buffer() {
    let nonce = format!(
        "{}_{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let tmp_wav_in = std::env::temp_dir().join(format!("ropus_fs120_in_{nonce}.wav"));
    let tmp_opus = std::env::temp_dir().join(format!("ropus_fs120_{nonce}.opus"));

    write_sine_wav(&tmp_wav_in, 5, 1000.0);

    let enc_opts = EncodeOptions {
        input: tmp_wav_in.clone(),
        output: Some(tmp_opus.clone()),
        bitrate: Some(128_000),
        complexity: None,
        application: ropus_tools_core::Application::Audio,
        // hard CBR: vbr=false, vbr_constraint=false — the CLI maps --hard-cbr
        // to this pair in ropusenc/src/main.rs.
        vbr: false,
        vbr_constraint: false,
        signal: ropus_tools_core::Signal::Auto,
        frame_duration: ropus_tools_core::FrameDuration::Ms120,
        expect_loss: 0,
        downmix_to_mono: false,
        serial: None,
        picture_path: None,
        vendor: "ropus-tools-core-test".to_string(),
        comments: Vec::new(),
    };
    commands::encode(enc_opts).expect("encode framesize 120 hard-cbr 128k");

    // File must decode cleanly end-to-end — no malformed packet framing.
    let tmp_wav_out = std::env::temp_dir().join(format!("ropus_fs120_out_{nonce}.wav"));
    let dec_opts = DecodeOptions {
        input: tmp_opus.clone(),
        output: Some(tmp_wav_out.clone()),
    };
    commands::decode(dec_opts).expect("decode the CBR-at-120ms output");

    let file_size = std::fs::metadata(&tmp_opus).expect("stat opus").len();
    let expected_payload: u64 = 128_000 / 8 * 5; // 80 000 bytes
    let min = expected_payload * 90 / 100; // 72 000
    let max = expected_payload * 110 / 100; // 88 000
    assert!(
        file_size >= min && file_size <= max,
        "opus file size {file_size} bytes outside ±10% of {expected_payload} \
         (min={min}, max={max}). A too-small file indicates the packet buffer \
         cap was limiting multi-sub-frame packets."
    );

    let _ = std::fs::remove_file(&tmp_wav_in);
    let _ = std::fs::remove_file(&tmp_opus);
    let _ = std::fs::remove_file(&tmp_wav_out);
}
