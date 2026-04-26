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
    assert!(
        bytes.len() >= 44,
        "WAV shorter than canonical 44-byte header"
    );
    let channels = u16::from_le_bytes([bytes[22], bytes[23]]);
    let sample_rate = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    // data chunk starts at offset 36 with "data" + u32 size, samples follow at 44.
    assert_eq!(
        &bytes[36..40],
        b"data",
        "expected canonical data chunk at 36"
    );
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
        float: false,
        raw: false,
        rate: None,
        gain_db: 0.0,
        dither: true,
        packet_loss_pct: 0,
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

    let sig_power: f64 = in_samples[..n]
        .iter()
        .map(|&x| (x as f64).powi(2))
        .sum::<f64>()
        / n as f64;
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
        comments: vec!["ARTIST=X".to_string(), "TITLE=Y".to_string()],
    };
    commands::encode(enc_opts).expect("encode with tags");

    let tags = read_opus_tags_from_file(&tmp_opus);
    assert_eq!(tags.get("ARTIST"), Some("X"), "artist round-tripped");
    assert_eq!(tags.get("TITLE"), Some("Y"), "title round-tripped");
    assert_eq!(tags.vendor, "ropus-tools-core-test", "vendor round-tripped");

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
        float: false,
        raw: false,
        rate: None,
        gain_db: 0.0,
        dither: true,
        packet_loss_pct: 0,
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

/// Shared encode fixture: synth a 1 s 1 kHz sine WAV, encode to Opus, return
/// the Opus path. Callers own cleanup of that path. Keeps every decode-flag
/// test from re-duplicating the 20 lines of `EncodeOptions` boilerplate.
fn encode_tmp_sine_opus(tag: &str) -> PathBuf {
    let nonce = format!(
        "{}_{}_{}",
        tag,
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let wav_in = std::env::temp_dir().join(format!("ropus_dec_in_{nonce}.wav"));
    let opus_path = std::env::temp_dir().join(format!("ropus_dec_{nonce}.opus"));

    write_sine_wav(&wav_in, 1, 1000.0);

    let enc_opts = EncodeOptions {
        input: wav_in.clone(),
        output: Some(opus_path.clone()),
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
    commands::encode(enc_opts).expect("encode sine fixture");
    let _ = std::fs::remove_file(&wav_in);

    opus_path
}

#[test]
fn decode_with_rate_44100_writes_that_rate() {
    // --rate 44100 must resample *after* pre-skip trim and write a WAV whose
    // fmt chunk reports 44 100 Hz. Any regression that skips the rename
    // (sample_rate still hard-coded to 48 kHz in write_wav_pcm16) fails here.
    let opus_path = encode_tmp_sine_opus("rate44100");
    let wav_out = std::env::temp_dir().join(format!(
        "ropus_dec_rate_{}_{}.wav",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));

    let dec_opts = DecodeOptions {
        input: opus_path.clone(),
        output: Some(wav_out.clone()),
        float: false,
        raw: false,
        rate: Some(44_100),
        gain_db: 0.0,
        dither: true,
        packet_loss_pct: 0,
    };
    commands::decode(dec_opts).expect("decode --rate 44100");

    let (_, out_sr, _) = read_pcm16_wav(&wav_out);
    assert_eq!(out_sr, 44_100, "WAV header must advertise target rate");

    let _ = std::fs::remove_file(&opus_path);
    let _ = std::fs::remove_file(&wav_out);
}

#[test]
fn decode_float_round_trip_produces_valid_wav() {
    // --float: WAV format code 3, fact chunk, bits_per_sample = 32, and the
    // correct interleaved f32 sample count.
    let opus_path = encode_tmp_sine_opus("float");
    let wav_out = std::env::temp_dir().join(format!(
        "ropus_dec_float_{}_{}.wav",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));

    let dec_opts = DecodeOptions {
        input: opus_path.clone(),
        output: Some(wav_out.clone()),
        float: true,
        raw: false,
        rate: None,
        gain_db: 0.0,
        dither: true, // ignored in float mode
        packet_loss_pct: 0,
    };
    commands::decode(dec_opts).expect("decode --float");

    let bytes = std::fs::read(&wav_out).expect("read float wav");
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    assert_eq!(&bytes[12..16], b"fmt ");
    let fmt_size = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    assert_eq!(fmt_size, 18, "float WAV needs fmt chunk size 18");
    let format_code = u16::from_le_bytes([bytes[20], bytes[21]]);
    assert_eq!(format_code, 3, "format code must be IEEE float (3)");
    let bits = u16::from_le_bytes([bytes[34], bytes[35]]);
    assert_eq!(bits, 32);
    // fact chunk immediately follows the 18-byte fmt body (plus the 8-byte
    // chunk header): "fact" at offset 38..42.
    assert_eq!(&bytes[38..42], b"fact");
    assert_eq!(&bytes[50..54], b"data");
    let data_bytes = u32::from_le_bytes([bytes[54], bytes[55], bytes[56], bytes[57]]) as usize;
    assert!(data_bytes > 0, "float WAV must contain sample data");
    assert!(
        data_bytes.is_multiple_of(4),
        "data size must be multiple of 4 bytes"
    );

    let _ = std::fs::remove_file(&opus_path);
    let _ = std::fs::remove_file(&wav_out);
}

#[test]
fn decode_gain_plus_header_gain_clamps_cleanly() {
    // Header gain 0 + --gain 200 dB => total Q8 far beyond libopus' ±128 dB
    // range. `set_gain` must surface an error via anyhow rather than panic
    // or silently wrap. We only assert that decode() returns Err — the exact
    // message is not load-bearing.
    let opus_path = encode_tmp_sine_opus("bad_gain");
    let wav_out = std::env::temp_dir().join(format!(
        "ropus_dec_bad_gain_{}_{}.wav",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));

    let dec_opts = DecodeOptions {
        input: opus_path.clone(),
        output: Some(wav_out.clone()),
        float: false,
        raw: false,
        rate: None,
        gain_db: 200.0, // 200 dB * 256 = 51200 Q8, out of [-32768, 32767]
        dither: true,
        packet_loss_pct: 0,
    };
    let err = commands::decode(dec_opts).expect_err("200 dB must surface as Err");
    let msg = format!("{err:#}");
    assert!(
        msg.to_ascii_lowercase().contains("gain"),
        "error should mention gain, got: {msg}"
    );

    let _ = std::fs::remove_file(&opus_path);
    let _ = std::fs::remove_file(&wav_out);
}

#[test]
fn decode_rejects_nan_gain() {
    // NaN `--gain` would saturate to 0 when cast to the Q8 i32 later,
    // silently ignoring the flag. The validator must surface it as a
    // clean error before opening any files.
    let opus_path = encode_tmp_sine_opus("nan_gain");
    let wav_out = std::env::temp_dir().join(format!(
        "ropus_dec_nan_gain_{}_{}.wav",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));

    let dec_opts = DecodeOptions {
        input: opus_path.clone(),
        output: Some(wav_out.clone()),
        float: false,
        raw: false,
        rate: None,
        gain_db: f32::NAN,
        dither: true,
        packet_loss_pct: 0,
    };
    let err = commands::decode(dec_opts).expect_err("NaN gain must surface as Err");
    let msg = format!("{err:#}").to_ascii_lowercase();
    assert!(
        msg.contains("gain") && msg.contains("finite"),
        "error should mention gain + finite, got: {msg}"
    );

    // Also reject +∞.
    let dec_opts_inf = DecodeOptions {
        input: opus_path.clone(),
        output: Some(wav_out.clone()),
        float: false,
        raw: false,
        rate: None,
        gain_db: f32::INFINITY,
        dither: true,
        packet_loss_pct: 0,
    };
    assert!(
        commands::decode(dec_opts_inf).is_err(),
        "+∞ gain must surface as Err"
    );

    let _ = std::fs::remove_file(&opus_path);
    let _ = std::fs::remove_file(&wav_out);
}

#[test]
fn decode_packet_loss_zero_is_bit_identical_to_no_flag() {
    // `packet_loss_pct = 0` must short-circuit the PRNG entirely so the
    // output matches a default-flag decode byte-for-byte. This is the safety
    // net: the flag exists but "off" means off.
    let opus_path = encode_tmp_sine_opus("loss0");
    let tag_a = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let wav_default = std::env::temp_dir().join(format!(
        "ropus_loss_default_{}_{}.wav",
        std::process::id(),
        tag_a
    ));
    let wav_explicit = std::env::temp_dir().join(format!(
        "ropus_loss_explicit0_{}_{}.wav",
        std::process::id(),
        tag_a
    ));

    let base = DecodeOptions {
        input: opus_path.clone(),
        output: Some(wav_default.clone()),
        float: false,
        raw: false,
        rate: None,
        gain_db: 0.0,
        dither: true,
        packet_loss_pct: 0,
    };
    commands::decode(base).expect("decode default");

    let explicit = DecodeOptions {
        input: opus_path.clone(),
        output: Some(wav_explicit.clone()),
        float: false,
        raw: false,
        rate: None,
        gain_db: 0.0,
        dither: true,
        packet_loss_pct: 0,
    };
    commands::decode(explicit).expect("decode packet_loss_pct=0");

    let a = std::fs::read(&wav_default).expect("read a");
    let b = std::fs::read(&wav_explicit).expect("read b");
    assert_eq!(
        a, b,
        "packet_loss_pct=0 must match no-flag output byte-for-byte"
    );

    let _ = std::fs::remove_file(&opus_path);
    let _ = std::fs::remove_file(&wav_default);
    let _ = std::fs::remove_file(&wav_explicit);
}

#[test]
fn decode_packet_loss_nonzero_produces_valid_output() {
    // With 10% simulated loss PLC must fill the gaps and the output must
    // decode without error. Length is approximately (source seconds * rate),
    // which for a 1 s 48 kHz fixture lands ~48 000 samples (minus pre-skip).
    let opus_path = encode_tmp_sine_opus("loss10");
    let wav_out = std::env::temp_dir().join(format!(
        "ropus_loss10_{}_{}.wav",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));

    let dec_opts = DecodeOptions {
        input: opus_path.clone(),
        output: Some(wav_out.clone()),
        float: false,
        raw: false,
        rate: None,
        gain_db: 0.0,
        dither: true,
        packet_loss_pct: 10,
    };
    commands::decode(dec_opts).expect("decode with --packet-loss 10");

    let (samples, sr, _) = read_pcm16_wav(&wav_out);
    assert_eq!(sr, 48_000);
    // 1 s of mono audio: expect at least 40 000 samples after pre-skip.
    assert!(
        samples.len() > 40_000,
        "decoded length {} too short — PLC did not fill dropped packets",
        samples.len()
    );

    let _ = std::fs::remove_file(&opus_path);
    let _ = std::fs::remove_file(&wav_out);
}

/// Pull the raw OpusHead packet bytes from a freshly-encoded Ogg Opus file.
/// Mirror of `read_opus_tags_from_file` but stops after packet 0 so the caller
/// can feed it into `container::ogg::parse_opus_head` directly.
fn read_opus_head_from_file(path: &std::path::Path) -> ropus_tools_core::container::ogg::OpusHead {
    use ogg::reading::PacketReader;
    use ropus_tools_core::container::ogg::parse_opus_head;

    let file = std::fs::File::open(path).expect("open opus");
    let mut reader = PacketReader::new(std::io::BufReader::new(file));
    let head_pkt = reader
        .read_packet()
        .expect("read OpusHead")
        .expect("packet 0");
    parse_opus_head(&head_pkt.data).expect("parse OpusHead")
}

/// Encode-then-decode a 1 s mono sine under the given signal hint and
/// confirm the pipeline doesn't panic / error and produces non-silent PCM.
/// Asserting the encoder actually *used* the hint is out of scope — that's a
/// codec-internal choice — but the flag must plumb through cleanly.
fn signal_hint_round_trip(tag: &str, signal: ropus_tools_core::Signal) {
    let nonce = format!(
        "{}_{}_{}",
        tag,
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let tmp_wav_in = std::env::temp_dir().join(format!("ropus_sig_in_{nonce}.wav"));
    let tmp_opus = std::env::temp_dir().join(format!("ropus_sig_{nonce}.opus"));
    let tmp_wav_out = std::env::temp_dir().join(format!("ropus_sig_out_{nonce}.wav"));

    write_sine_wav(&tmp_wav_in, 1, 1000.0);

    let enc_opts = EncodeOptions {
        input: tmp_wav_in.clone(),
        output: Some(tmp_opus.clone()),
        bitrate: Some(64_000),
        complexity: None,
        application: ropus_tools_core::Application::Audio,
        vbr: true,
        vbr_constraint: false,
        signal,
        frame_duration: ropus_tools_core::FrameDuration::Ms20,
        expect_loss: 0,
        downmix_to_mono: false,
        serial: None,
        picture_path: None,
        vendor: "ropus-tools-core-test".to_string(),
        comments: Vec::new(),
    };
    commands::encode(enc_opts).expect("encode with signal hint");

    let dec_opts = DecodeOptions {
        input: tmp_opus.clone(),
        output: Some(tmp_wav_out.clone()),
        float: false,
        raw: false,
        rate: None,
        gain_db: 0.0,
        dither: true,
        packet_loss_pct: 0,
    };
    commands::decode(dec_opts).expect("decode signal-hinted file");

    let (samples, sr, _) = read_pcm16_wav(&tmp_wav_out);
    assert_eq!(sr, 48_000, "output WAV must be 48 kHz");
    assert!(
        samples.iter().any(|&s| s != 0),
        "decoded PCM is all-zero — signal hint likely broke the encode path"
    );

    let _ = std::fs::remove_file(&tmp_wav_in);
    let _ = std::fs::remove_file(&tmp_opus);
    let _ = std::fs::remove_file(&tmp_wav_out);
}

#[test]
fn encode_with_music_signal_hint_round_trips() {
    // Step-4 coverage: pinning `Signal::Music` must plumb through the encoder
    // builder without panic or error. We don't assert the bitstream changed —
    // that's codec-internal — only that the flag reaches the encoder cleanly
    // and a valid, non-zero PCM round-trip survives.
    signal_hint_round_trip("music", ropus_tools_core::Signal::Music);
}

#[test]
fn encode_with_voice_signal_hint_round_trips() {
    // Mirror of the Music test for `Signal::Voice`; same plumbing assertion,
    // different enum variant. Together they cover both sides of `--music` /
    // `--speech` mapping in the CLI.
    signal_hint_round_trip("voice", ropus_tools_core::Signal::Voice);
}

#[test]
fn encode_with_downmix_mono_produces_mono_output() {
    // Step-4 coverage: `downmix_to_mono: true` on a stereo input must
    // collapse to a 1-channel stream end-to-end. The critical assertion is
    // `channels == 1` in OpusHead — if the flag didn't plumb through, the
    // encoder would see the original 2-channel PCM and write channels=2.
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace parent")
        .to_path_buf();
    let stereo_wav = workspace.join("tests/vectors/48000hz_stereo_sine440.wav");
    if !stereo_wav.exists() {
        eprintln!(
            "SKIPPING encode_with_downmix_mono_produces_mono_output: \
             stereo fixture {stereo_wav:?} not present"
        );
        return;
    }

    // Sanity: the fixture really is stereo. If someone swaps it for a mono
    // file by accident, the test degenerates into a tautology and this catches it.
    let (_, _, in_ch) = read_pcm16_wav(&stereo_wav);
    assert_eq!(in_ch, 2, "downmix test needs a stereo input fixture");

    let nonce = format!(
        "{}_{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let tmp_opus = std::env::temp_dir().join(format!("ropus_downmix_{nonce}.opus"));
    let tmp_wav_out = std::env::temp_dir().join(format!("ropus_downmix_out_{nonce}.wav"));

    let enc_opts = EncodeOptions {
        input: stereo_wav,
        output: Some(tmp_opus.clone()),
        bitrate: Some(64_000),
        complexity: None,
        application: ropus_tools_core::Application::Audio,
        vbr: true,
        vbr_constraint: false,
        signal: ropus_tools_core::Signal::Auto,
        frame_duration: ropus_tools_core::FrameDuration::Ms20,
        expect_loss: 0,
        downmix_to_mono: true,
        serial: None,
        picture_path: None,
        vendor: "ropus-tools-core-test".to_string(),
        comments: Vec::new(),
    };
    commands::encode(enc_opts).expect("encode with downmix_to_mono");

    // Load-bearing check: OpusHead.channels must be 1.
    let head = read_opus_head_from_file(&tmp_opus);
    assert_eq!(
        head.channels, 1,
        "downmix_to_mono must write channels=1 into OpusHead, got {}",
        head.channels
    );

    // Secondary check: the decoded WAV reports 1 channel and its interleaved
    // sample count is evenly divisible by 1 (trivially true, but the channel
    // field in the fmt chunk is what matters).
    let dec_opts = DecodeOptions {
        input: tmp_opus.clone(),
        output: Some(tmp_wav_out.clone()),
        float: false,
        raw: false,
        rate: None,
        gain_db: 0.0,
        dither: true,
        packet_loss_pct: 0,
    };
    commands::decode(dec_opts).expect("decode downmixed file");
    let (samples, _, out_ch) = read_pcm16_wav(&tmp_wav_out);
    assert_eq!(out_ch, 1, "decoded WAV must report 1 channel");
    assert!(!samples.is_empty(), "decoded WAV has no sample data");

    let _ = std::fs::remove_file(&tmp_opus);
    let _ = std::fs::remove_file(&tmp_wav_out);
}

#[test]
fn encode_with_picture_flag_embeds_metadata_block_picture() {
    // Step-4 coverage: `picture_path: Some(path)` must wrap the file in a
    // METADATA_BLOCK_PICTURE structure, base64-encode it, and prepend it to
    // the Vorbis comments in OpusTags. The detector only inspects the first
    // handful of bytes, so a hand-rolled 32-byte PNG-magic-plus-filler buffer
    // is enough to pass `detect_format`.
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace parent")
        .to_path_buf();
    let input_wav = workspace.join("tests/vectors/48k_sine1k_loud.wav");
    if !input_wav.exists() {
        eprintln!(
            "SKIPPING encode_with_picture_flag_embeds_metadata_block_picture: \
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
    let tmp_png = std::env::temp_dir().join(format!("ropus_pic_{nonce}.png"));
    let tmp_opus = std::env::temp_dir().join(format!("ropus_pic_{nonce}.opus"));

    // Minimal PNG stand-in: the 8-byte PNG signature followed by enough
    // filler bytes that `build_picture_block` emits a realistic payload.
    // `detect_format` only looks at bytes 0..8 for the PNG magic, so the
    // rest is opaque — we never ask anything to actually decode the image.
    let mut png_bytes: Vec<u8> = vec![
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG magic
    ];
    png_bytes.extend_from_slice(&[0xAAu8; 24]); // 24 bytes of filler -> 32 total
    std::fs::write(&tmp_png, &png_bytes).expect("write fake PNG fixture");

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
        picture_path: Some(tmp_png.clone()),
        vendor: "ropus-tools-core-test".to_string(),
        comments: Vec::new(),
    };
    commands::encode(enc_opts).expect("encode with picture_path");

    // Parse OpusTags back out, find the METADATA_BLOCK_PICTURE entry, and
    // confirm it carries a non-trivial base64 payload. A METADATA_BLOCK_PICTURE
    // that plumbed correctly will be well over 50 chars: 32 bytes of header +
    // 9-byte MIME + 32-byte payload ≈ 73 raw bytes → ~100 chars base64.
    let tags = read_opus_tags_from_file(&tmp_opus);
    let pic = tags
        .comments
        .iter()
        .find(|c| c.starts_with("METADATA_BLOCK_PICTURE="))
        .expect("METADATA_BLOCK_PICTURE comment must be present");
    let b64 = pic
        .strip_prefix("METADATA_BLOCK_PICTURE=")
        .expect("prefix just matched");
    assert!(
        b64.len() >= 50,
        "picture payload only {} chars (< 50) — likely unplumbed: full comment = {}",
        b64.len(),
        pic
    );
    // base64 output must be a multiple of 4 (RFC 4648 canonical form).
    assert!(
        b64.len().is_multiple_of(4),
        "base64 payload length {} is not a multiple of 4",
        b64.len()
    );

    let _ = std::fs::remove_file(&tmp_png);
    let _ = std::fs::remove_file(&tmp_opus);
}
