//! mdopus-compare: CLI tool that compares C reference opus output against
//! the Rust implementation, byte-for-byte / sample-for-sample.

#![allow(
    clippy::needless_range_loop,
    clippy::manual_range_contains,
    clippy::unnecessary_cast,
    clippy::collapsible_if,
    clippy::identity_op,
    clippy::manual_is_variant_and,
    clippy::manual_memcpy,
    clippy::items_after_test_module,
    clippy::single_match,
    clippy::unnecessary_unwrap
)]

#[path = "bindings.rs"]
mod bindings;

use std::fs;
use std::path::Path;
use std::process;

// ---------------------------------------------------------------------------
// WAV reading (minimal, 16-bit PCM only)
// ---------------------------------------------------------------------------

struct WavData {
    sample_rate: u32,
    channels: u16,
    samples: Vec<i16>,
}

fn read_wav(path: &Path) -> WavData {
    let data = fs::read(path).unwrap_or_else(|e| {
        eprintln!("ERROR: cannot read {}: {}", path.display(), e);
        process::exit(1);
    });
    if data.len() < 44 {
        eprintln!("ERROR: file too small to be a WAV: {}", path.display());
        process::exit(1);
    }
    if &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        eprintln!("ERROR: not a WAV file: {}", path.display());
        process::exit(1);
    }

    // Find "fmt " chunk
    let mut pos = 12;
    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut bits_per_sample;
    let mut fmt_found = false;

    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size =
            u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]])
                as usize;
        if chunk_id == b"fmt " {
            if chunk_size < 16 {
                eprintln!("ERROR: fmt chunk too small");
                process::exit(1);
            }
            let audio_format = u16::from_le_bytes([data[pos + 8], data[pos + 9]]);
            if audio_format != 1 {
                eprintln!(
                    "ERROR: only PCM WAV supported (got format {})",
                    audio_format
                );
                process::exit(1);
            }
            channels = u16::from_le_bytes([data[pos + 10], data[pos + 11]]);
            sample_rate = u32::from_le_bytes([
                data[pos + 12],
                data[pos + 13],
                data[pos + 14],
                data[pos + 15],
            ]);
            bits_per_sample = u16::from_le_bytes([data[pos + 22], data[pos + 23]]);
            if bits_per_sample != 16 {
                eprintln!("ERROR: only 16-bit PCM supported (got {})", bits_per_sample);
                process::exit(1);
            }
            fmt_found = true;
        }
        if chunk_id == b"data" {
            if !fmt_found {
                eprintln!("ERROR: data chunk before fmt chunk");
                process::exit(1);
            }
            let sample_data = &data[pos + 8..pos + 8 + chunk_size];
            let samples: Vec<i16> = sample_data
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]))
                .collect();
            return WavData {
                sample_rate,
                channels,
                samples,
            };
        }
        pos += 8 + chunk_size;
        // WAV chunks are word-aligned
        if !chunk_size.is_multiple_of(2) {
            pos += 1;
        }
    }
    eprintln!("ERROR: no data chunk found in WAV");
    process::exit(1);
}

// ---------------------------------------------------------------------------
// Comparison statistics
// ---------------------------------------------------------------------------

struct CompareStats {
    total: usize,
    matching: usize,
    first_diff_offset: Option<usize>,
    max_diff: i32,
}

fn compare_bytes(a: &[u8], b: &[u8]) -> CompareStats {
    let total = a.len().max(b.len());
    let mut matching = 0usize;
    let mut first_diff_offset = None;
    let mut max_diff: i32 = 0;

    for i in 0..total {
        let va = a.get(i).copied().unwrap_or(0);
        let vb = b.get(i).copied().unwrap_or(0);
        let diff = (va as i32 - vb as i32).abs();
        if diff == 0 {
            matching += 1;
        } else {
            if first_diff_offset.is_none() {
                first_diff_offset = Some(i);
            }
            max_diff = max_diff.max(diff);
        }
    }
    CompareStats {
        total,
        matching,
        first_diff_offset,
        max_diff,
    }
}

fn compare_samples(a: &[i16], b: &[i16]) -> CompareStats {
    let total = a.len().max(b.len());
    let mut matching = 0usize;
    let mut first_diff_offset = None;
    let mut max_diff: i32 = 0;

    for i in 0..total {
        let va = a.get(i).copied().unwrap_or(0) as i32;
        let vb = b.get(i).copied().unwrap_or(0) as i32;
        let diff = (va - vb).abs();
        if diff == 0 {
            matching += 1;
        } else {
            if first_diff_offset.is_none() {
                first_diff_offset = Some(i);
            }
            max_diff = max_diff.max(diff);
        }
    }
    CompareStats {
        total,
        matching,
        first_diff_offset,
        max_diff,
    }
}

fn print_result(label: &str, stats: &CompareStats, a: &[u8], b: &[u8]) {
    match stats.first_diff_offset {
        None => {
            println!("{}: PASS ({} items, all match)", label, stats.total);
        }
        Some(offset) => {
            println!(
                "{}: FAIL at offset {} (total: {}, matching: {}, max_diff: {})",
                label, offset, stats.total, stats.matching, stats.max_diff
            );
            // Hex dump around first difference
            let start = offset.saturating_sub(16);
            let end = (offset + 48).min(a.len().max(b.len()));
            println!("  C ref:  {}", hex_line(a, start, end));
            println!("  Rust:   {}", hex_line(b, start, end));
            println!(
                "  {}^ offset {}",
                " ".repeat(10 + (offset - start) * 3),
                offset
            );
        }
    }
}

fn hex_line(data: &[u8], start: usize, end: usize) -> String {
    (start..end)
        .map(|i| {
            if i < data.len() {
                format!("{:02x}", data[i])
            } else {
                "--".to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn print_sample_result(label: &str, stats: &CompareStats, a: &[i16], b: &[i16]) {
    match stats.first_diff_offset {
        None => {
            println!("{}: PASS ({} samples, all match)", label, stats.total);
        }
        Some(offset) => {
            println!(
                "{}: FAIL at sample {} (total: {}, matching: {}, max_diff: {})",
                label, offset, stats.total, stats.matching, stats.max_diff
            );
            let start = offset.saturating_sub(4);
            let end = (offset + 12).min(a.len().max(b.len()));
            print!("  C ref: ");
            for i in start..end {
                let v = a.get(i).copied().unwrap_or(0);
                if i == offset {
                    print!("[{:6}]", v);
                } else {
                    print!(" {:6} ", v);
                }
            }
            println!();
            print!("  Rust:  ");
            for i in start..end {
                let v = b.get(i).copied().unwrap_or(0);
                if i == offset {
                    print!("[{:6}]", v);
                } else {
                    print!(" {:6} ", v);
                }
            }
            println!();
        }
    }
}

// ---------------------------------------------------------------------------
// Encode configuration
// ---------------------------------------------------------------------------

#[allow(dead_code)]
#[derive(Clone)]
struct EncodeConfig {
    sample_rate: i32,
    channels: i32,
    bitrate: i32,
    complexity: i32,
    application: i32,
    vbr: i32,
    vbr_constraint: i32,
    frame_ms: f64,
    fec: i32,
    packet_loss_pct: i32,
    dtx: i32,
    signal: i32,
    bandwidth: i32,
    force_channels: i32,
    max_bandwidth: i32,
    lsb_depth: i32,
    prediction_disabled: i32,
    phase_inversion_disabled: i32,
    force_mode: i32,
}

impl EncodeConfig {
    fn new(sample_rate: i32, channels: i32) -> Self {
        Self {
            sample_rate,
            channels,
            bitrate: 64000,
            complexity: 10,
            application: bindings::OPUS_APPLICATION_AUDIO,
            vbr: 0,
            vbr_constraint: 0,
            frame_ms: 20.0,
            fec: 0,
            packet_loss_pct: 0,
            dtx: 0,
            signal: bindings::OPUS_AUTO,
            bandwidth: bindings::OPUS_AUTO,
            force_channels: bindings::OPUS_AUTO,
            max_bandwidth: bindings::OPUS_BANDWIDTH_FULLBAND,
            lsb_depth: 24,
            prediction_disabled: 0,
            phase_inversion_disabled: 0,
            force_mode: bindings::OPUS_AUTO,
        }
    }

    fn frame_size(&self) -> usize {
        (self.sample_rate as f64 * self.frame_ms / 1000.0) as usize
    }
}

// ---------------------------------------------------------------------------
// Deterministic noise generator
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn generate_noise(sample_rate: i32, channels: i32, duration_secs: f64, seed: u64) -> Vec<i16> {
    let num_samples = (sample_rate as f64 * duration_secs) as usize * channels as usize;
    let mut rng = seed;
    let mut samples = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Use bits 33-48 directly as i16 — uniform over full i16 range
        samples.push((rng >> 33) as i16);
    }
    samples
}

/// Generate alternating noise/silence signal for DTX testing.
/// Pattern: 200ms noise, 200ms silence, repeated for duration.
fn generate_dtx_signal(sample_rate: i32, channels: i32, duration_secs: f64, seed: u64) -> Vec<i16> {
    let total_samples = (sample_rate as f64 * duration_secs) as usize * channels as usize;
    let chunk = (sample_rate as usize * channels as usize) / 5; // 200ms in samples
    let mut rng = seed;
    let mut samples = Vec::with_capacity(total_samples);
    let mut noise_phase = true;
    while samples.len() < total_samples {
        let remaining = total_samples - samples.len();
        let n = chunk.min(remaining);
        for _ in 0..n {
            if noise_phase {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                samples.push((rng >> 33) as i16);
            } else {
                samples.push(0);
            }
        }
        noise_phase = !noise_phase;
    }
    samples
}

// ---------------------------------------------------------------------------
// C reference encode/decode via FFI
// ---------------------------------------------------------------------------

fn c_encode_cfg(pcm: &[i16], cfg: &EncodeConfig) -> Vec<u8> {
    unsafe {
        let mut error: i32 = 0;
        let enc = bindings::opus_encoder_create(
            cfg.sample_rate,
            cfg.channels,
            cfg.application,
            &mut error,
        );
        if enc.is_null() || error != bindings::OPUS_OK {
            eprintln!(
                "ERROR: C opus_encoder_create failed: {}",
                bindings::error_string(error)
            );
            process::exit(1);
        }

        macro_rules! ctl {
            ($enc:expr, $req:expr, $val:expr) => {{
                let ret = bindings::opus_encoder_ctl($enc, $req, $val);
                if ret != bindings::OPUS_OK {
                    eprintln!(
                        "WARNING: C opus_encoder_ctl({}, {}) failed: {}",
                        $req,
                        $val,
                        bindings::error_string(ret)
                    );
                }
            }};
        }

        // Apply all CTL settings
        ctl!(enc, bindings::OPUS_SET_BITRATE_REQUEST, cfg.bitrate);
        ctl!(enc, bindings::OPUS_SET_COMPLEXITY_REQUEST, cfg.complexity);
        ctl!(enc, bindings::OPUS_SET_VBR_REQUEST, cfg.vbr);
        ctl!(
            enc,
            bindings::OPUS_SET_VBR_CONSTRAINT_REQUEST,
            cfg.vbr_constraint
        );
        ctl!(enc, bindings::OPUS_SET_INBAND_FEC_REQUEST, cfg.fec);
        ctl!(
            enc,
            bindings::OPUS_SET_PACKET_LOSS_PERC_REQUEST,
            cfg.packet_loss_pct
        );
        ctl!(enc, bindings::OPUS_SET_DTX_REQUEST, cfg.dtx);
        ctl!(enc, bindings::OPUS_SET_SIGNAL_REQUEST, cfg.signal);
        ctl!(enc, bindings::OPUS_SET_BANDWIDTH_REQUEST, cfg.bandwidth);
        ctl!(
            enc,
            bindings::OPUS_SET_FORCE_CHANNELS_REQUEST,
            cfg.force_channels
        );
        ctl!(
            enc,
            bindings::OPUS_SET_MAX_BANDWIDTH_REQUEST,
            cfg.max_bandwidth
        );
        ctl!(enc, bindings::OPUS_SET_LSB_DEPTH_REQUEST, cfg.lsb_depth);
        ctl!(
            enc,
            bindings::OPUS_SET_PREDICTION_DISABLED_REQUEST,
            cfg.prediction_disabled
        );
        ctl!(
            enc,
            bindings::OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST,
            cfg.phase_inversion_disabled
        );
        ctl!(enc, bindings::OPUS_SET_FORCE_MODE_REQUEST, cfg.force_mode);

        let frame_size = cfg.frame_size();
        let samples_per_frame = frame_size * cfg.channels as usize;
        let max_packet = 4000;
        let mut output = Vec::new();
        let mut packet = vec![0u8; max_packet];

        let mut pos = 0;
        while pos + samples_per_frame <= pcm.len() {
            let ret = bindings::opus_encode(
                enc,
                pcm[pos..].as_ptr(),
                frame_size as i32,
                packet.as_mut_ptr(),
                max_packet as i32,
            );
            if ret < 0 {
                eprintln!(
                    "ERROR: C opus_encode failed: {}",
                    bindings::error_string(ret)
                );
                bindings::opus_encoder_destroy(enc);
                process::exit(1);
            }
            // Prepend 2-byte length header for framing
            output.extend_from_slice(&(ret as u16).to_le_bytes());
            output.extend_from_slice(&packet[..ret as usize]);
            pos += samples_per_frame;
        }

        bindings::opus_encoder_destroy(enc);
        output
    }
}

fn c_encode(
    pcm: &[i16],
    sample_rate: i32,
    channels: i32,
    bitrate: i32,
    complexity: i32,
) -> Vec<u8> {
    let mut cfg = EncodeConfig::new(sample_rate, channels);
    cfg.bitrate = bitrate;
    cfg.complexity = complexity;
    c_encode_cfg(pcm, &cfg)
}

fn c_decode_cfg(encoded: &[u8], cfg: &EncodeConfig) -> Vec<i16> {
    unsafe {
        let mut error: i32 = 0;
        let dec = bindings::opus_decoder_create(cfg.sample_rate, cfg.channels, &mut error);
        if dec.is_null() || error != bindings::OPUS_OK {
            eprintln!(
                "ERROR: C opus_decoder_create failed: {}",
                bindings::error_string(error)
            );
            process::exit(1);
        }

        let frame_size = cfg.frame_size();
        let mut pcm = vec![0i16; frame_size * cfg.channels as usize];
        let mut output = Vec::new();

        let mut pos = 0;
        while pos + 2 <= encoded.len() {
            let pkt_len = u16::from_le_bytes([encoded[pos], encoded[pos + 1]]) as usize;
            pos += 2;
            if pos + pkt_len > encoded.len() {
                eprintln!("ERROR: truncated packet at offset {}", pos);
                break;
            }

            let ret = bindings::opus_decode(
                dec,
                encoded[pos..].as_ptr(),
                pkt_len as i32,
                pcm.as_mut_ptr(),
                frame_size as i32,
                0,
            );
            if ret < 0 {
                eprintln!(
                    "ERROR: C opus_decode failed: {}",
                    bindings::error_string(ret)
                );
                bindings::opus_decoder_destroy(dec);
                process::exit(1);
            }
            output.extend_from_slice(&pcm[..ret as usize * cfg.channels as usize]);
            pos += pkt_len;
        }

        bindings::opus_decoder_destroy(dec);
        output
    }
}

fn c_decode(encoded: &[u8], sample_rate: i32, channels: i32) -> Vec<i16> {
    let cfg = EncodeConfig::new(sample_rate, channels);
    c_decode_cfg(encoded, &cfg)
}

// ---------------------------------------------------------------------------
// Rust implementation encode/decode
// ---------------------------------------------------------------------------

fn rust_encode_cfg(pcm: &[i16], cfg: &EncodeConfig) -> Vec<u8> {
    use mdopus::opus::encoder::OpusEncoder;

    let mut enc = match OpusEncoder::new(cfg.sample_rate, cfg.channels, cfg.application) {
        Ok(e) => e,
        Err(code) => {
            eprintln!("ERROR: Rust OpusEncoder::new failed: {}", code);
            return Vec::new();
        }
    };

    fn check_ctl(name: &str, ret: i32) {
        if ret != 0 {
            eprintln!("WARNING: Rust {} failed: {}", name, ret);
        }
    }

    // Apply all settings
    check_ctl("set_bitrate", enc.set_bitrate(cfg.bitrate));
    check_ctl("set_complexity", enc.set_complexity(cfg.complexity));
    check_ctl("set_vbr", enc.set_vbr(cfg.vbr));
    check_ctl(
        "set_vbr_constraint",
        enc.set_vbr_constraint(cfg.vbr_constraint),
    );
    check_ctl("set_inband_fec", enc.set_inband_fec(cfg.fec));
    check_ctl(
        "set_packet_loss_perc",
        enc.set_packet_loss_perc(cfg.packet_loss_pct),
    );
    check_ctl("set_dtx", enc.set_dtx(cfg.dtx));
    check_ctl("set_signal", enc.set_signal(cfg.signal));
    check_ctl("set_bandwidth", enc.set_bandwidth(cfg.bandwidth));
    check_ctl(
        "set_force_channels",
        enc.set_force_channels(cfg.force_channels),
    );
    check_ctl(
        "set_max_bandwidth",
        enc.set_max_bandwidth(cfg.max_bandwidth),
    );
    check_ctl("set_lsb_depth", enc.set_lsb_depth(cfg.lsb_depth));
    check_ctl(
        "set_prediction_disabled",
        enc.set_prediction_disabled(cfg.prediction_disabled),
    );
    check_ctl(
        "set_phase_inversion_disabled",
        enc.set_phase_inversion_disabled(cfg.phase_inversion_disabled),
    );
    check_ctl("set_force_mode", enc.set_force_mode(cfg.force_mode));

    let frame_size = cfg.frame_size();
    let samples_per_frame = frame_size * cfg.channels as usize;
    let max_packet = 4000;
    let mut output = Vec::new();
    let mut packet = vec![0u8; max_packet];

    let mut pos = 0;
    while pos + samples_per_frame <= pcm.len() {
        match enc.encode(
            &pcm[pos..pos + samples_per_frame],
            frame_size as i32,
            &mut packet,
            max_packet as i32,
        ) {
            Ok(ret) => {
                let ret = ret as usize;
                output.extend_from_slice(&(ret as u16).to_le_bytes());
                output.extend_from_slice(&packet[..ret]);
            }
            Err(code) => {
                eprintln!(
                    "ERROR: Rust encode failed at frame {}: {}",
                    pos / samples_per_frame,
                    code
                );
                return output; // Return partial output for debugging
            }
        }
        pos += samples_per_frame;
    }
    output
}

fn rust_encode(
    pcm: &[i16],
    sample_rate: i32,
    channels: i32,
    bitrate: i32,
    complexity: i32,
) -> Vec<u8> {
    let mut cfg = EncodeConfig::new(sample_rate, channels);
    cfg.bitrate = bitrate;
    cfg.complexity = complexity;
    rust_encode_cfg(pcm, &cfg)
}

fn rust_decode_cfg(encoded: &[u8], cfg: &EncodeConfig) -> Vec<i16> {
    use mdopus::opus::decoder::OpusDecoder;

    let mut dec = match OpusDecoder::new(cfg.sample_rate, cfg.channels) {
        Ok(d) => d,
        Err(code) => {
            eprintln!("ERROR: Rust OpusDecoder::new failed: {}", code);
            return Vec::new();
        }
    };

    let frame_size = cfg.frame_size();
    let mut pcm = vec![0i16; frame_size * cfg.channels as usize];
    let mut output = Vec::new();

    let mut pos = 0;
    while pos + 2 <= encoded.len() {
        let pkt_len = u16::from_le_bytes([encoded[pos], encoded[pos + 1]]) as usize;
        pos += 2;
        if pos + pkt_len > encoded.len() {
            eprintln!("ERROR: truncated packet at offset {}", pos);
            break;
        }

        match dec.decode(
            Some(&encoded[pos..pos + pkt_len]),
            &mut pcm,
            frame_size as i32,
            false,
        ) {
            Ok(ret) => {
                output.extend_from_slice(&pcm[..ret as usize * cfg.channels as usize]);
            }
            Err(code) => {
                eprintln!(
                    "ERROR: Rust decode failed at packet offset {}: {}",
                    pos, code
                );
                return output; // Return partial for debugging
            }
        }
        pos += pkt_len;
    }
    output
}

fn rust_decode(encoded: &[u8], sample_rate: i32, channels: i32) -> Vec<i16> {
    let cfg = EncodeConfig::new(sample_rate, channels);
    rust_decode_cfg(encoded, &cfg)
}

// ---------------------------------------------------------------------------
// Unit-level module comparison tests
// ---------------------------------------------------------------------------

fn unit_range_coder() -> bool {
    println!("--- range coder comparison ---");
    let mut pass = true;

    // Test: encode a sequence of uints, decode back, compare
    let test_values: Vec<(u32, u32)> = vec![
        (0, 10),
        (5, 10),
        (9, 10),
        (0, 256),
        (127, 256),
        (255, 256),
        (0, 65536),
        (12345, 65536),
        (65535, 65536),
    ];

    let mut buf = vec![0u8; 1024];

    // C encode
    let buf_len = buf.len() as u32;
    unsafe {
        let mut enc = std::mem::zeroed::<bindings::ec_enc>();
        bindings::ec_enc_init(&mut enc, buf.as_mut_ptr(), buf_len);

        for &(val, ft) in &test_values {
            bindings::ec_enc_uint(&mut enc, val, ft);
        }
        bindings::ec_enc_done(&mut enc);

        // C decode -- pass original storage size, not just offs, because
        // ec_enc_uint uses raw bits at the END of the buffer.
        let mut dec = std::mem::zeroed::<bindings::ec_dec>();
        bindings::ec_dec_init(&mut dec, buf.as_mut_ptr(), buf_len);

        for (i, &(expected, ft)) in test_values.iter().enumerate() {
            let decoded = bindings::ec_dec_uint(&mut dec, ft);
            if decoded != expected {
                println!(
                    "  C range coder MISMATCH at index {}: encoded {}, decoded {} (ft={})",
                    i, expected, decoded, ft
                );
                pass = false;
            }
        }
        if pass {
            println!(
                "  C range coder: encode/decode roundtrip OK ({} values)",
                test_values.len()
            );
        }

        // Test bit_logp
        let logp_tests: Vec<(i32, u32)> = vec![(0, 1), (1, 1), (0, 2), (1, 2), (0, 8), (1, 8)];
        let mut buf2 = vec![0u8; 256];
        let mut enc2 = std::mem::zeroed::<bindings::ec_enc>();
        bindings::ec_enc_init(&mut enc2, buf2.as_mut_ptr(), buf2.len() as u32);
        for &(val, logp) in &logp_tests {
            bindings::ec_enc_bit_logp(&mut enc2, val, logp);
        }
        bindings::ec_enc_done(&mut enc2);
        let enc2_len = enc2.offs as usize;

        let mut dec2 = std::mem::zeroed::<bindings::ec_dec>();
        bindings::ec_dec_init(&mut dec2, buf2.as_mut_ptr(), enc2_len as u32);
        for (i, &(expected, logp)) in logp_tests.iter().enumerate() {
            let decoded = bindings::ec_dec_bit_logp(&mut dec2, logp);
            if decoded != expected {
                println!(
                    "  C bit_logp MISMATCH at index {}: encoded {}, decoded {} (logp={})",
                    i, expected, decoded, logp
                );
                pass = false;
            }
        }
        if pass {
            println!("  C bit_logp: roundtrip OK ({} values)", logp_tests.len());
        }
    }

    // TODO: compare against Rust range coder when implemented
    println!("  (Rust range coder not yet implemented -- C self-test only)");
    pass
}

fn cmd_unit(module: &str) -> bool {
    match module {
        "range_coder" | "entcode" => unit_range_coder(),
        "all" => {
            let mut pass = true;
            pass &= unit_range_coder();
            // Future modules: fft, mdct, pitch, etc.
            pass
        }
        _ => {
            eprintln!(
                "ERROR: unknown module '{}'. Available: range_coder, all",
                module
            );
            false
        }
    }
}

// ---------------------------------------------------------------------------
// CLI commands
// ---------------------------------------------------------------------------

fn cmd_encode(wav_path: &str, bitrate: i32, complexity: i32) {
    let wav = read_wav(Path::new(wav_path));
    println!(
        "Input: {} ({} Hz, {} ch, {} samples)",
        wav_path,
        wav.sample_rate,
        wav.channels,
        wav.samples.len() / wav.channels as usize
    );

    let sr = wav.sample_rate as i32;
    let ch = wav.channels as i32;

    println!(
        "Encoding with C reference (bitrate={}, complexity={})...",
        bitrate, complexity
    );
    let c_encoded = c_encode(&wav.samples, sr, ch, bitrate, complexity);
    println!("  C encoded: {} bytes", c_encoded.len());

    println!("Encoding with Rust implementation...");
    let rust_encoded = rust_encode(&wav.samples, sr, ch, bitrate, complexity);
    println!("  Rust encoded: {} bytes", rust_encoded.len());

    if rust_encoded.is_empty() {
        println!("\nencode: SKIP (Rust encoder not yet implemented)");
        return;
    }

    let stats = compare_bytes(&c_encoded, &rust_encoded);
    println!();
    print_result("encode", &stats, &c_encoded, &rust_encoded);

    // Per-frame comparison
    println!();
    println!("=== Per-frame packet comparison ===");
    let c_packets = parse_packets(&c_encoded);
    let r_packets = parse_packets(&rust_encoded);
    println!(
        "  C packets: {}, Rust packets: {}",
        c_packets.len(),
        r_packets.len()
    );

    let n = c_packets.len().min(r_packets.len());
    let mut first_diff_frame = None;
    for i in 0..n {
        let cp = &c_packets[i];
        let rp = &r_packets[i];
        if cp == rp {
            println!("  Frame {:3}: {} bytes - MATCH", i, cp.len());
        } else {
            let pkt_stats = compare_bytes(cp, rp);
            println!(
                "  Frame {:3}: {} bytes - DIFFER at byte {} (max_diff={})",
                i,
                cp.len(),
                pkt_stats.first_diff_offset.unwrap_or(0),
                pkt_stats.max_diff
            );
            if first_diff_frame.is_none() {
                first_diff_frame = Some(i);
                // Dump hex around the difference
                if let Some(off) = pkt_stats.first_diff_offset {
                    let start = off.saturating_sub(8);
                    let end = (off + 24).min(cp.len().max(rp.len()));
                    println!("    C pkt:  {}", hex_line(cp, start, end));
                    println!("    R pkt:  {}", hex_line(rp, start, end));
                    println!("    {}^ byte {}", " ".repeat(12 + (off - start) * 3), off);
                    // Dump first 4 bytes (TOC + possibly SILK/CELT headers)
                    println!("    TOC byte: C=0x{:02x} R=0x{:02x}", cp[0], rp[0]);
                    // Count total differing bytes in this packet
                    let mut ndiff = 0;
                    for j in 0..cp.len().min(rp.len()) {
                        if cp[j] != rp[j] {
                            ndiff += 1;
                            println!(
                                "    Byte {}: C=0x{:02x} R=0x{:02x} diff={}",
                                j,
                                cp[j],
                                rp[j],
                                (cp[j] as i32 - rp[j] as i32).abs()
                            );
                        }
                    }
                    println!("    Total differing bytes in frame: {}", ndiff);
                }
            }
        }
    }
}

fn cmd_encode_framecompare(
    wav_path: &str,
    bitrate: i32,
    complexity: i32,
    application: i32,
    signal: i32,
) {
    // Do per-frame encode: each frame independently through C and Rust,
    // comparing the encoder state after each frame
    let wav = read_wav(Path::new(wav_path));
    let sr = wav.sample_rate as i32;
    let ch = wav.channels as i32;
    let frame_size = (sr / 50) as usize;
    let samples_per_frame = frame_size * ch as usize;
    let max_packet = 4000;

    // C encoder
    let c_enc = unsafe {
        let mut error: i32 = 0;
        let enc = bindings::opus_encoder_create(sr, ch, application, &mut error);
        bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_BITRATE_REQUEST, bitrate);
        bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_COMPLEXITY_REQUEST, complexity);
        bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_VBR_REQUEST, 0i32);
        if signal != -1000 {
            bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_SIGNAL_REQUEST, signal);
        }
        enc
    };

    // Rust encoder
    use mdopus::opus::encoder::OpusEncoder;
    let mut r_enc = OpusEncoder::new(sr, ch, application).unwrap();
    r_enc.set_bitrate(bitrate);
    r_enc.set_complexity(complexity);
    r_enc.set_vbr(0);
    if signal != -1000 {
        r_enc.set_signal(signal);
    }

    let mut pos = 0;
    let mut frame_idx = 0;
    let mut c_pkt = vec![0u8; max_packet];
    let mut r_pkt = vec![0u8; max_packet];

    while pos + samples_per_frame <= wav.samples.len() {
        let pcm = &wav.samples[pos..pos + samples_per_frame];

        // C encode
        let c_len = unsafe {
            bindings::opus_encode(
                c_enc,
                pcm.as_ptr(),
                frame_size as i32,
                c_pkt.as_mut_ptr(),
                max_packet as i32,
            )
        };

        // Rust encode
        let r_len = r_enc
            .encode(pcm, frame_size as i32, &mut r_pkt, max_packet as i32)
            .unwrap_or(-1);

        let cl = c_len as usize;
        let rl = r_len as usize;

        // Query bandwidths
        let c_bw = unsafe {
            let mut bw: i32 = 0;
            bindings::opus_encoder_ctl(
                c_enc,
                bindings::OPUS_GET_BANDWIDTH_REQUEST,
                &mut bw as *mut i32,
            );
            bw
        };
        let r_bw = r_enc.get_bandwidth();

        if c_pkt[..cl] == r_pkt[..rl] {
            println!(
                "Frame {:3}: {} bytes - MATCH (C_bw={} R_bw={} C_toc=0x{:02x} R_toc=0x{:02x})",
                frame_idx, cl, c_bw, r_bw, c_pkt[0], r_pkt[0]
            );
        } else {
            println!(
                "Frame {:3}: C={} bytes, R={} bytes - DIFFER (C_bw={} R_bw={} C_toc=0x{:02x} R_toc=0x{:02x})",
                frame_idx, cl, rl, c_bw, r_bw, c_pkt[0], r_pkt[0]
            );
            let min_len = cl.min(rl);
            let mut ndiff = 0;
            for j in 0..min_len {
                if c_pkt[j] != r_pkt[j] {
                    ndiff += 1;
                    if ndiff <= 5 {
                        println!(
                            "  byte {:3}: C=0x{:02x} R=0x{:02x} diff={}",
                            j,
                            c_pkt[j],
                            r_pkt[j],
                            (c_pkt[j] as i32 - r_pkt[j] as i32).abs()
                        );
                    }
                }
            }
            if ndiff > 5 {
                println!("  ... and {} more differing bytes", ndiff - 5);
            }
            println!("  Total differing bytes: {}", ndiff);

            // For the first differing frame, dump the full C and Rust packets
            // around the difference region
            let stats = compare_bytes(&c_pkt[..cl], &r_pkt[..rl]);
            if let Some(off) = stats.first_diff_offset {
                let start = off.saturating_sub(16);
                let end = (off + 32).min(cl.max(rl));
                println!("  C pkt:  {}", hex_line(&c_pkt[..cl], start, end));
                println!("  R pkt:  {}", hex_line(&r_pkt[..rl], start, end));
            }
        }

        pos += samples_per_frame;
        frame_idx += 1;
    }

    unsafe {
        bindings::opus_encoder_destroy(c_enc);
    }
}

fn parse_packets(data: &[u8]) -> Vec<Vec<u8>> {
    let mut packets = Vec::new();
    let mut pos = 0;
    while pos + 2 <= data.len() {
        let pkt_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;
        if pos + pkt_len > data.len() {
            break;
        }
        packets.push(data[pos..pos + pkt_len].to_vec());
        pos += pkt_len;
    }
    packets
}

fn cmd_decode(opus_path: &str) {
    let data = fs::read(opus_path).unwrap_or_else(|e| {
        eprintln!("ERROR: cannot read {}: {}", opus_path, e);
        process::exit(1);
    });

    // Assume our simple framing format (length-prefixed packets) at 48kHz stereo
    let sample_rate = 48000i32;
    let channels = 2i32;

    println!("Decoding {} with C reference...", opus_path);
    let c_pcm = c_decode(&data, sample_rate, channels);
    println!("  C decoded: {} samples", c_pcm.len());

    println!("Decoding with Rust implementation...");
    let rust_pcm = rust_decode(&data, sample_rate, channels);
    println!("  Rust decoded: {} samples", rust_pcm.len());

    if rust_pcm.is_empty() {
        println!("\ndecode: SKIP (Rust decoder not yet implemented)");
        return;
    }

    let stats = compare_samples(&c_pcm, &rust_pcm);
    println!();
    print_sample_result("decode", &stats, &c_pcm, &rust_pcm);
}

fn cmd_roundtrip(wav_path: &str, bitrate: i32) {
    let wav = read_wav(Path::new(wav_path));
    println!(
        "Input: {} ({} Hz, {} ch, {} samples)",
        wav_path,
        wav.sample_rate,
        wav.channels,
        wav.samples.len() / wav.channels as usize
    );

    let sr = wav.sample_rate as i32;
    let ch = wav.channels as i32;
    let complexity = 10;

    // C roundtrip
    println!("C reference roundtrip (bitrate={})...", bitrate);
    let c_encoded = c_encode(&wav.samples, sr, ch, bitrate, complexity);
    let c_decoded = c_decode(&c_encoded, sr, ch);
    println!(
        "  C: {} bytes encoded -> {} samples decoded",
        c_encoded.len(),
        c_decoded.len()
    );

    // Rust roundtrip
    println!("Rust implementation roundtrip...");
    let rust_encoded = rust_encode(&wav.samples, sr, ch, bitrate, complexity);

    if rust_encoded.is_empty() {
        println!("\nroundtrip: SKIP (Rust encoder not yet implemented)");
        return;
    }

    let rust_decoded = rust_decode(&rust_encoded, sr, ch);
    println!(
        "  Rust: {} bytes encoded -> {} samples decoded",
        rust_encoded.len(),
        rust_decoded.len()
    );

    // Compare final PCM
    let stats = compare_samples(&c_decoded, &rust_decoded);
    println!();
    print_sample_result("roundtrip", &stats, &c_decoded, &rust_decoded);

    // --- Decode-only comparison (same bitstream) ---
    println!();
    println!("=== Decode-only comparison (C-encoded data decoded by both) ===");
    let rust_decoded_c_data = rust_decode(&c_encoded, sr, ch);
    println!(
        "  Rust decoded C data: {} samples",
        rust_decoded_c_data.len()
    );
    let dec_stats = compare_samples(&c_decoded, &rust_decoded_c_data);
    print_sample_result("decode-only", &dec_stats, &c_decoded, &rust_decoded_c_data);
}

// ---------------------------------------------------------------------------
// Math function comparison (C vs Rust)
// ---------------------------------------------------------------------------

fn cmd_mathcompare() {
    use mdopus::celt::math_ops;
    use mdopus::celt::vq;

    // First: check OPUS_FAST_INT64 setting
    let fast_int64 = unsafe { bindings::debug_c_opus_fast_int64() };
    println!("C OPUS_FAST_INT64 = {}", fast_int64);
    println!();

    // Test celt_sqrt32
    println!("--- celt_sqrt32 comparison ---");
    let test_vals: Vec<i32> = vec![
        0,
        1,
        100,
        10000,
        1_000_000,
        100_000_000,
        1_000_000_000,
        1_073_741_823,
        536_870_912,
        268_435_456,
        16_777_216,
        42,
        12345,
        999_999_999,
        7,
        255,
        65535,
    ];
    let mut sqrt_pass = true;
    for &x in &test_vals {
        let c_val = unsafe { bindings::debug_c_celt_sqrt32(x) };
        let r_val = math_ops::celt_sqrt32(x);
        if c_val != r_val {
            println!(
                "  MISMATCH: celt_sqrt32({}) C={} R={} diff={}",
                x,
                c_val,
                r_val,
                c_val - r_val
            );
            sqrt_pass = false;
        }
    }
    if sqrt_pass {
        println!("  celt_sqrt32: {} values PASS", test_vals.len());
    }

    // Test frac_div32
    println!("--- frac_div32 comparison ---");
    let div_tests: Vec<(i32, i32)> = vec![
        (1000, 2000),
        (100, 300),
        (12345, 67890),
        (1, 2),
        (1_000_000, 2_000_000),
        (536_870_912, 1_073_741_824),
        (100_000_000, 200_000_001),
        (999_999, 1_000_001),
    ];
    let mut div_pass = true;
    for &(a, b) in &div_tests {
        let c_val = unsafe { bindings::debug_c_frac_div32(a, b) };
        let r_val = math_ops::frac_div32(a, b);
        if c_val != r_val {
            println!(
                "  MISMATCH: frac_div32({}, {}) C={} R={} diff={}",
                a,
                b,
                c_val,
                r_val,
                c_val - r_val
            );
            div_pass = false;
        }
    }
    if div_pass {
        println!("  frac_div32: {} values PASS", div_tests.len());
    }

    // Test celt_atan_norm
    println!("--- celt_atan_norm comparison ---");
    let atan_tests: Vec<i32> = vec![
        0,
        1,
        -1,
        536_870_912,
        -536_870_912,
        1_073_741_824,
        -1_073_741_824, // exact +-1.0
        268_435_456,
        805_306_368,
        100_000_000,
        -100_000_000,
        1_000_000,
        -1_000_000,
    ];
    let mut atan_pass = true;
    for &x in &atan_tests {
        let c_val = unsafe { bindings::debug_c_celt_atan_norm(x) };
        let r_val = math_ops::celt_atan_norm(x);
        if c_val != r_val {
            println!(
                "  MISMATCH: celt_atan_norm({}) C={} R={} diff={}",
                x,
                c_val,
                r_val,
                c_val - r_val
            );
            atan_pass = false;
        }
    }
    if atan_pass {
        println!("  celt_atan_norm: {} values PASS", atan_tests.len());
    }

    // Test celt_atan2p_norm
    println!("--- celt_atan2p_norm comparison ---");
    let atan2_tests: Vec<(i32, i32)> = vec![
        (0, 0),
        (0, 100),
        (100, 0),
        (100, 100),
        (1_000_000, 2_000_000),
        (2_000_000, 1_000_000),
        (536_870_912, 1_073_741_824),
        (1_073_741_824, 536_870_912),
    ];
    let mut atan2_pass = true;
    for &(y, x) in &atan2_tests {
        let c_val = unsafe { bindings::debug_c_celt_atan2p_norm(y, x) };
        let r_val = math_ops::celt_atan2p_norm(y, x);
        if c_val != r_val {
            println!(
                "  MISMATCH: celt_atan2p_norm({}, {}) C={} R={} diff={}",
                y,
                x,
                c_val,
                r_val,
                c_val - r_val
            );
            atan2_pass = false;
        }
    }
    if atan2_pass {
        println!("  celt_atan2p_norm: {} values PASS", atan2_tests.len());
    }

    // Test stereo_itheta with realistic data
    println!("--- stereo_itheta comparison ---");
    let n = 8;
    // Typical Q24 norm vectors
    let test_vectors: Vec<(Vec<i32>, Vec<i32>)> = vec![
        // Equal energy
        (vec![1 << 24; 8], vec![1 << 24; 8]),
        // Unequal energy
        (
            vec![1 << 23, 1 << 22, 1 << 21, 1 << 20, 0, 0, 0, 0],
            vec![1 << 21, 1 << 20, 1 << 19, 1 << 18, 0, 0, 0, 0],
        ),
        // Sine-like
        (
            vec![
                0, 5_930_000, 11_180_000, 14_900_000, 16_777_216, 14_900_000, 11_180_000, 5_930_000,
            ],
            vec![
                16_777_216,
                14_900_000,
                11_180_000,
                5_930_000,
                0,
                -5_930_000,
                -11_180_000,
                -14_900_000,
            ],
        ),
    ];
    let mut sitheta_pass = true;
    for (i, (xv, yv)) in test_vectors.iter().enumerate() {
        // Non-stereo path
        let c_val = unsafe { bindings::debug_c_stereo_itheta(xv.as_ptr(), yv.as_ptr(), 0, n) };
        let r_val = vq::stereo_itheta(xv, yv, false, n as usize);
        if c_val != r_val {
            println!(
                "  MISMATCH: stereo_itheta (non-stereo, vec #{}) C={} R={} diff={}",
                i,
                c_val,
                r_val,
                c_val - r_val
            );
            sitheta_pass = false;
        }
    }
    if sitheta_pass {
        println!("  stereo_itheta: {} vectors PASS", test_vectors.len());
    }

    // Test celt_rsqrt_norm32
    println!("--- celt_rsqrt_norm32 comparison ---");
    let rsqrt_tests: Vec<i32> = vec![
        536_870_912,   // 0.25 Q31
        1_073_741_824, // 0.5 Q31
        1_610_612_736, // 0.75 Q31
        2_000_000_000, // ~0.93 Q31
        2_147_483_646, // ~1.0 Q31
        600_000_000,
        800_000_000,
        1_200_000_000,
        1_500_000_000,
        1_900_000_000,
    ];
    let mut rsqrt_pass = true;
    for &x in &rsqrt_tests {
        let c_val = unsafe { bindings::debug_c_celt_rsqrt_norm32(x) };
        let r_val = math_ops::celt_rsqrt_norm32(x);
        if c_val != r_val {
            println!(
                "  MISMATCH: celt_rsqrt_norm32({}) C={} R={} diff={}",
                x,
                c_val,
                r_val,
                c_val - r_val
            );
            rsqrt_pass = false;
        }
    }
    if rsqrt_pass {
        println!("  celt_rsqrt_norm32: {} values PASS", rsqrt_tests.len());
    }

    // Test normalise_residual gain computation
    println!("--- normalise_residual gain comparison ---");
    use mdopus::types::{mult32_32_q31, vshr32};
    let gain_tests: Vec<(i32, i32)> = vec![
        (1, 2_147_483_647), // typical ryy, Q31ONE gain
        (4, 2_147_483_647),
        (16, 2_147_483_647),
        (100, 2_147_483_647),
        (1000, 1_500_000_000), // moderate gain
        (10000, 1_000_000_000),
        (1, 536_870_912), // small gain
    ];
    let mut gain_pass = true;
    for &(ryy, gain) in &gain_tests {
        let c_val = unsafe { bindings::debug_c_normalise_residual_g(ryy, gain) };
        // Replicate the gain computation
        let k = math_ops::celt_ilog2(ryy) >> 1;
        let t = vshr32(ryy, 2 * (k - 7) - 15);
        let r_val = mult32_32_q31(math_ops::celt_rsqrt_norm32(t), gain);
        if c_val != r_val {
            println!(
                "  MISMATCH: normalise_residual_g(ryy={}, gain={}) C={} R={} diff={}",
                ryy,
                gain,
                c_val,
                r_val,
                c_val - r_val
            );
            gain_pass = false;
        }
    }
    if gain_pass {
        println!(
            "  normalise_residual gain: {} values PASS",
            gain_tests.len()
        );
    }

    // Test stereo_itheta with actual encoder data (from traces)
    println!("--- stereo_itheta with actual encoder vectors ---");
    // First divergent vectors (from eoffs=57, n=16, band=13)
    let x_c: Vec<i32> = vec![
        4594765, 42154, -2191998, -7903840, -8283225, -948462, 3372303, 2950766, 927383, -1686152,
        1011692, 400460, -1243537, -1201384, -126461, -252922,
    ];
    let x_r: Vec<i32> = vec![
        4594765, 42154, -2191998, -7903841, -8283225, -948462, 3372303, 2950766, 927383, -1686152,
        1011692, 400461, -1243537, -1201384, -126461, -252922,
    ];
    let y_c: Vec<i32> = vec![
        843076, 2950768, 1517537, -1032768, 2086613, -2465998, 505847, -2992920, 5184916, -210772,
        -2782153, 358306, -2002309, -4742306, -885233, -1053848,
    ];
    let y_r: Vec<i32> = vec![
        843076, 2950768, 1517538, -1032768, 2086613, -2465998, 505847, -2992920, 5184916, -210772,
        -2782154, 358306, -2002309, -4742306, -885233, -1053848,
    ];

    // C with C-vectors
    let c_with_c = unsafe { bindings::debug_c_stereo_itheta(x_c.as_ptr(), y_c.as_ptr(), 0, 16) };
    // C with R-vectors (to see the difference caused by input change)
    let c_with_r = unsafe { bindings::debug_c_stereo_itheta(x_r.as_ptr(), y_r.as_ptr(), 0, 16) };
    // Rust with C-vectors
    let r_with_c = vq::stereo_itheta(&x_c, &y_c, false, 16);
    // Rust with R-vectors
    let r_with_r = vq::stereo_itheta(&x_r, &y_r, false, 16);

    println!("  C(C-vectors):  {}", c_with_c);
    println!(
        "  C(R-vectors):  {} (diff from C-input: {})",
        c_with_r,
        c_with_r - c_with_c
    );
    println!(
        "  R(C-vectors):  {} (diff from C: {})",
        r_with_c,
        r_with_c - c_with_c
    );
    println!(
        "  R(R-vectors):  {} (diff from C(C): {})",
        r_with_r,
        r_with_r - c_with_c
    );

    // Exhaustive sweep: celt_sqrt32 on values that produce results near
    // the itheta range used in encoding
    println!("--- Exhaustive celt_sqrt32 sweep near typical values ---");
    let mut sqrt_mismatches = 0;
    // Sweep around powers of 2 (common norm energy values)
    for base in [
        1 << 14,
        1 << 16,
        1 << 18,
        1 << 20,
        1 << 22,
        1 << 24,
        1 << 26,
        1 << 28,
    ] {
        for delta in -1000..=1000 {
            let x = base + delta;
            if x <= 0 || x >= 1_073_741_824 {
                continue;
            }
            let c_val = unsafe { bindings::debug_c_celt_sqrt32(x) };
            let r_val = math_ops::celt_sqrt32(x);
            if c_val != r_val {
                sqrt_mismatches += 1;
                if sqrt_mismatches <= 5 {
                    println!(
                        "  MISMATCH: celt_sqrt32({}) C={} R={} diff={}",
                        x,
                        c_val,
                        r_val,
                        c_val - r_val
                    );
                }
            }
        }
    }
    if sqrt_mismatches > 0 {
        println!("  celt_sqrt32 sweep: {} mismatches found!", sqrt_mismatches);
    } else {
        println!("  celt_sqrt32 sweep: 16000 values PASS");
    }

    // Exhaustive sweep: celt_rsqrt_norm32 over entire valid range (200K points)
    println!("--- Exhaustive celt_rsqrt_norm32 sweep (200K) ---");
    let mut rsqrt32_mismatches = 0u64;
    let mut rsqrt32_tested = 0u64;
    // Valid range is [2^29, 2^31-1] — test 200K uniform points
    let rsqrt32_lo = 536_870_912i64; // 2^29
    let rsqrt32_hi = 2_147_483_647i64; // 2^31-1
    for i in 0..200_000 {
        let x = (rsqrt32_lo + i * (rsqrt32_hi - rsqrt32_lo) / 200_000) as i32;
        let c_val = unsafe { bindings::debug_c_celt_rsqrt_norm32(x) };
        let r_val = math_ops::celt_rsqrt_norm32(x);
        rsqrt32_tested += 1;
        if c_val != r_val {
            rsqrt32_mismatches += 1;
            if rsqrt32_mismatches <= 5 {
                println!(
                    "  MISMATCH: celt_rsqrt_norm32({}) C={} R={} diff={}",
                    x,
                    c_val,
                    r_val,
                    c_val - r_val
                );
            }
        }
    }
    // Also test boundary values
    for &x in &[536_870_912i32, 1_073_741_824, 2_147_483_647] {
        let c_val = unsafe { bindings::debug_c_celt_rsqrt_norm32(x) };
        let r_val = math_ops::celt_rsqrt_norm32(x);
        rsqrt32_tested += 1;
        if c_val != r_val {
            rsqrt32_mismatches += 1;
            if rsqrt32_mismatches <= 5 {
                println!(
                    "  MISMATCH: celt_rsqrt_norm32({}) C={} R={} diff={}",
                    x,
                    c_val,
                    r_val,
                    c_val - r_val
                );
            }
        }
    }
    if rsqrt32_mismatches > 0 {
        println!(
            "  celt_rsqrt_norm32 sweep: {} mismatches out of {}!",
            rsqrt32_mismatches, rsqrt32_tested
        );
    } else {
        println!("  celt_rsqrt_norm32 sweep: {} values PASS", rsqrt32_tested);
    }

    // Exhaustive sweep: normalise_residual gain computation
    println!("--- Exhaustive normalise_residual gain sweep ---");
    let mut gain_mismatches = 0;
    let gain_value = 2_147_483_647i32; // Q31ONE
    for i in 1..10000 {
        let ryy = i;
        let c_val = unsafe { bindings::debug_c_normalise_residual_g(ryy, gain_value) };
        let k = math_ops::celt_ilog2(ryy) >> 1;
        let t = vshr32(ryy, 2 * (k - 7) - 15);
        let r_val = mult32_32_q31(math_ops::celt_rsqrt_norm32(t), gain_value);
        if c_val != r_val {
            gain_mismatches += 1;
            if gain_mismatches <= 5 {
                println!(
                    "  MISMATCH: normalise_g(ryy={}, gain=Q31ONE) C={} R={} diff={}",
                    ryy,
                    c_val,
                    r_val,
                    c_val - r_val
                );
            }
        }
    }
    // Also test with non-Q31ONE gains
    for gi in [500_000_000i32, 1_000_000_000, 1_500_000_000, 2_000_000_000] {
        for i in 1..2000 {
            let ryy = i;
            let c_val = unsafe { bindings::debug_c_normalise_residual_g(ryy, gi) };
            let k = math_ops::celt_ilog2(ryy) >> 1;
            let t = vshr32(ryy, 2 * (k - 7) - 15);
            let r_val = mult32_32_q31(math_ops::celt_rsqrt_norm32(t), gi);
            if c_val != r_val {
                gain_mismatches += 1;
                if gain_mismatches <= 5 {
                    println!(
                        "  MISMATCH: normalise_g(ryy={}, gain={}) C={} R={} diff={}",
                        ryy,
                        gi,
                        c_val,
                        r_val,
                        c_val - r_val
                    );
                }
            }
        }
    }
    if gain_mismatches > 0 {
        println!("  normalise_g sweep: {} mismatches found!", gain_mismatches);
    } else {
        println!("  normalise_g sweep: 17999 values PASS");
    }

    // -----------------------------------------------------------------------
    // celt_rsqrt_norm — exhaustive sweep over Q16 domain [0.25, ~1.0)
    // -----------------------------------------------------------------------
    println!("--- celt_rsqrt_norm exhaustive sweep ---");
    let mut rsqrt_norm_mismatches = 0u64;
    let rsqrt_norm_count: u64 = (65535 - 16384 + 1) as u64; // 49,152 values
    for x in 16384..=65535i32 {
        let c_val = unsafe { bindings::debug_c_celt_rsqrt_norm(x) };
        let r_val = math_ops::celt_rsqrt_norm(x);
        if c_val != r_val {
            rsqrt_norm_mismatches += 1;
            if rsqrt_norm_mismatches <= 5 {
                println!(
                    "  MISMATCH: celt_rsqrt_norm({}) C={} R={} diff={}",
                    x,
                    c_val,
                    r_val,
                    c_val - r_val
                );
            }
        }
    }
    if rsqrt_norm_mismatches > 0 {
        println!(
            "  celt_rsqrt_norm sweep: {} mismatches out of {}!",
            rsqrt_norm_mismatches, rsqrt_norm_count
        );
    } else {
        println!("  celt_rsqrt_norm sweep: {} values PASS", rsqrt_norm_count);
    }

    // -----------------------------------------------------------------------
    // celt_rcp_norm16 — exhaustive sweep over Q15 domain [0.5, ~1.0)
    // -----------------------------------------------------------------------
    println!("--- celt_rcp_norm16 exhaustive sweep ---");
    let mut rcp16_mismatches = 0u64;
    let rcp16_count: u64 = (32767 - 16384 + 1) as u64; // 16,384 values
    for x in 16384..=32767i32 {
        let c_val = unsafe { bindings::debug_c_celt_rcp_norm16(x) };
        let r_val = math_ops::celt_rcp_norm16(x);
        if c_val != r_val {
            rcp16_mismatches += 1;
            if rcp16_mismatches <= 5 {
                println!(
                    "  MISMATCH: celt_rcp_norm16({}) C={} R={} diff={}",
                    x,
                    c_val,
                    r_val,
                    c_val - r_val
                );
            }
        }
    }
    if rcp16_mismatches > 0 {
        println!(
            "  celt_rcp_norm16 sweep: {} mismatches out of {}!",
            rcp16_mismatches, rcp16_count
        );
    } else {
        println!("  celt_rcp_norm16 sweep: {} values PASS", rcp16_count);
    }

    // -----------------------------------------------------------------------
    // celt_rcp_norm32 — 200K-point sweep over [2^30, 2^31-1]
    // -----------------------------------------------------------------------
    println!("--- celt_rcp_norm32 sweep (200K) ---");
    let mut rcp32_mismatches = 0u64;
    let mut rcp32_tested = 0u64;
    let rcp32_lo = 1_073_741_824i64; // 2^30
    let rcp32_hi = 2_147_483_647i64; // 2^31-1
    for i in 0..200_000 {
        let x = (rcp32_lo + i * (rcp32_hi - rcp32_lo) / 200_000) as i32;
        let c_val = unsafe { bindings::debug_c_celt_rcp_norm32(x) };
        let r_val = math_ops::celt_rcp_norm32(x);
        rcp32_tested += 1;
        if c_val != r_val {
            rcp32_mismatches += 1;
            if rcp32_mismatches <= 5 {
                println!(
                    "  MISMATCH: celt_rcp_norm32({}) C={} R={} diff={}",
                    x,
                    c_val,
                    r_val,
                    c_val - r_val
                );
            }
        }
    }
    // Also test boundary values
    for &x in &[1_073_741_824i32, 1_610_612_736, 2_147_483_647] {
        let c_val = unsafe { bindings::debug_c_celt_rcp_norm32(x) };
        let r_val = math_ops::celt_rcp_norm32(x);
        rcp32_tested += 1;
        if c_val != r_val {
            rcp32_mismatches += 1;
            if rcp32_mismatches <= 5 {
                println!(
                    "  MISMATCH: celt_rcp_norm32({}) C={} R={} diff={}",
                    x,
                    c_val,
                    r_val,
                    c_val - r_val
                );
            }
        }
    }
    if rcp32_mismatches > 0 {
        println!(
            "  celt_rcp_norm32 sweep: {} mismatches out of {}!",
            rcp32_mismatches, rcp32_tested
        );
    } else {
        println!("  celt_rcp_norm32 sweep: {} values PASS", rcp32_tested);
    }

    // -----------------------------------------------------------------------
    // celt_rcp — strategic multi-range sweep
    // -----------------------------------------------------------------------
    println!("--- celt_rcp strategic sweep ---");
    let mut rcp_mismatches = 0u64;
    let mut rcp_tested = 0u64;

    // Small values: 1..=1000 (every value)
    for x in 1..=1000i32 {
        let c_val = unsafe { bindings::debug_c_celt_rcp(x) };
        let r_val = math_ops::celt_rcp(x);
        rcp_tested += 1;
        if c_val != r_val {
            rcp_mismatches += 1;
            if rcp_mismatches <= 5 {
                println!(
                    "  MISMATCH: celt_rcp({}) C={} R={} diff={}",
                    x,
                    c_val,
                    r_val,
                    c_val - r_val
                );
            }
        }
    }

    // Medium values: 1000..100_000, sampled at ~1000 points
    for i in 0..1000 {
        let x = 1000 + (i as i64 * 99_000 / 1000) as i32;
        let c_val = unsafe { bindings::debug_c_celt_rcp(x) };
        let r_val = math_ops::celt_rcp(x);
        rcp_tested += 1;
        if c_val != r_val {
            rcp_mismatches += 1;
            if rcp_mismatches <= 5 {
                println!(
                    "  MISMATCH: celt_rcp({}) C={} R={} diff={}",
                    x,
                    c_val,
                    r_val,
                    c_val - r_val
                );
            }
        }
    }

    // Large values: 100_000..2_147_483_647, sampled at ~1000 points
    for i in 0..1000 {
        let x = (100_000i64 + i * (2_147_483_647i64 - 100_000) / 1000) as i32;
        let c_val = unsafe { bindings::debug_c_celt_rcp(x) };
        let r_val = math_ops::celt_rcp(x);
        rcp_tested += 1;
        if c_val != r_val {
            rcp_mismatches += 1;
            if rcp_mismatches <= 5 {
                println!(
                    "  MISMATCH: celt_rcp({}) C={} R={} diff={}",
                    x,
                    c_val,
                    r_val,
                    c_val - r_val
                );
            }
        }
    }

    // Powers of 2: 1, 2, 4, ..., 2^30
    for k in 0..=30 {
        let x = 1i32 << k;
        let c_val = unsafe { bindings::debug_c_celt_rcp(x) };
        let r_val = math_ops::celt_rcp(x);
        rcp_tested += 1;
        if c_val != r_val {
            rcp_mismatches += 1;
            if rcp_mismatches <= 5 {
                println!(
                    "  MISMATCH: celt_rcp(2^{} = {}) C={} R={} diff={}",
                    k,
                    x,
                    c_val,
                    r_val,
                    c_val - r_val
                );
            }
        }
    }

    // Near powers: 2^k +/- 1 for k=1..30
    for k in 1..=30 {
        let base = 1i32 << k;
        for &delta in &[-1i32, 1] {
            let x = base + delta;
            if x < 1 {
                continue;
            }
            let c_val = unsafe { bindings::debug_c_celt_rcp(x) };
            let r_val = math_ops::celt_rcp(x);
            rcp_tested += 1;
            if c_val != r_val {
                rcp_mismatches += 1;
                if rcp_mismatches <= 5 {
                    println!(
                        "  MISMATCH: celt_rcp(2^{}+{} = {}) C={} R={} diff={}",
                        k,
                        delta,
                        x,
                        c_val,
                        r_val,
                        c_val - r_val
                    );
                }
            }
        }
    }

    if rcp_mismatches > 0 {
        println!(
            "  celt_rcp sweep: {} mismatches out of {}!",
            rcp_mismatches, rcp_tested
        );
    } else {
        println!("  celt_rcp sweep: {} values PASS", rcp_tested);
    }
}

// ---------------------------------------------------------------------------
// Range decoder nbits comparison
// ---------------------------------------------------------------------------

fn cmd_rng_test() {
    use mdopus::celt::range_coder::RangeDecoder;

    // Extract the frame 4 payload from the 48k square wave at 128kbps
    // First, encode it with C
    let wav = read_wav(Path::new("tests/vectors/48k_square1k.wav"));
    let c_encoded = c_encode(&wav.samples, 48000, 1, 128000, 10);

    // Extract frame 4's Opus packet
    let mut pos = 0;
    let mut pkt_data: &[u8] = &[];
    for i in 0..5 {
        let pkt_len = u16::from_le_bytes([c_encoded[pos], c_encoded[pos + 1]]) as usize;
        pos += 2;
        println!(
            "Packet {}: len={} TOC=0x{:02x} (config={} s={} code={})",
            i,
            pkt_len,
            c_encoded[pos],
            c_encoded[pos] >> 3,
            (c_encoded[pos] >> 2) & 1,
            c_encoded[pos] & 3
        );
        if i == 4 {
            pkt_data = &c_encoded[pos..pos + pkt_len];
        }
        pos += pkt_len;
    }

    // The CELT payload is after the TOC byte
    let celt_payload = &pkt_data[1..];
    println!("Frame 4 CELT payload: {} bytes", celt_payload.len());

    // Create a Rust range decoder from this payload
    let mut dec = RangeDecoder::new(celt_payload);
    println!(
        "After init: tell={} nbits={} rng={:08x}",
        dec.tell(),
        dec.debug_nbits_total(),
        dec.get_rng()
    );

    // Read the same header as the CELT decoder would:
    // 1. Silence check (tell == 1)
    let tell0 = dec.tell();
    let silence = if tell0 >= celt_payload.len() as i32 * 8 {
        true
    } else if tell0 == 1 {
        dec.decode_bit_logp(15)
    } else {
        false
    };
    println!(
        "After silence: tell={} nbits={} silence={}",
        dec.tell(),
        dec.debug_nbits_total(),
        silence
    );

    // 2. Postfilter
    let tell1 = dec.tell();
    let total_bits = celt_payload.len() as i32 * 8;
    let mut pf_pitch = 0;
    let mut pf_gain = 0;
    let mut pf_tapset = 0;
    if tell1 + 16 <= total_bits {
        let has_pf = dec.decode_bit_logp(1);
        if has_pf {
            let octave = dec.decode_uint(6);
            pf_pitch = (16 << octave) as i32 + dec.decode_bits(4 + octave) as i32 - 1;
            let qg = dec.decode_bits(3) as i32;
            pf_gain = 3072 * (qg + 1);
            let tell2 = dec.tell();
            if tell2 + 2 <= total_bits {
                let tapset_icdf: [u8; 3] = [2, 1, 0];
                pf_tapset = dec.decode_icdf(&tapset_icdf, 2);
            }
        }
    }
    println!(
        "After PF: tell={} nbits={} pitch={} gain={} tapset={}",
        dec.tell(),
        dec.debug_nbits_total(),
        pf_pitch,
        pf_gain,
        pf_tapset
    );

    // 3. Transient (LM=3 > 0)
    let tell3 = dec.tell();
    let mut is_trans = false;
    if tell3 + 3 <= total_bits {
        is_trans = dec.decode_bit_logp(3);
    }
    println!(
        "After transient: tell={} nbits={} is_trans={}",
        dec.tell(),
        dec.debug_nbits_total(),
        is_trans
    );

    // 4. Intra energy
    let tell4 = dec.tell();
    let intra = if tell4 + 3 <= total_bits {
        if dec.decode_bit_logp(3) { 1 } else { 0 }
    } else {
        0
    };
    println!(
        "After intra: tell={} nbits={} intra={}",
        dec.tell(),
        dec.debug_nbits_total(),
        intra
    );

    // Coarse energy decode with Rust
    {
        use mdopus::celt::modes::MODE_48000_960_120;
        use mdopus::celt::quant_bands::unquant_coarse_energy;
        let mode = &MODE_48000_960_120;
        let mut old_e = vec![0i32; 42]; // 2*21
        // Set old_e to the known state from frame 3
        // (We'll skip this since we want to test the range decoder itself, not the energy values)
        // Just use zeros for now -- the key is comparing nbits_total

        let before_coarse_nbits = dec.debug_nbits_total();
        let _before_coarse_rng = dec.get_rng();
        unquant_coarse_energy(mode, 0, 21, &mut old_e, intra, &mut dec, 1, 3);
        println!(
            "After Rust coarse: tell={} nbits={} rng={:08x} band0={}",
            dec.tell(),
            dec.debug_nbits_total(),
            dec.get_rng(),
            old_e[0]
        );
        println!(
            "  nbits consumed in coarse: {}",
            dec.debug_nbits_total() - before_coarse_nbits
        );
    }

    // Now create a C range decoder from the same payload and compare
    println!("\nC equivalent:");
    unsafe {
        let mut c_dec: bindings::ec_dec = std::mem::zeroed();
        bindings::ec_dec_init(
            &mut c_dec,
            celt_payload.as_ptr() as *mut _,
            celt_payload.len() as u32,
        );
        println!(
            "After init: nbits={} rng={:08x} tell={}",
            c_dec.nbits_total,
            c_dec.rng,
            c_dec.nbits_total - 32i32.wrapping_sub(c_dec.rng.leading_zeros() as i32)
        );

        // Silence
        let c_silence = bindings::ec_dec_bit_logp(&mut c_dec, 15);
        println!(
            "After silence: nbits={} silence={}",
            c_dec.nbits_total, c_silence
        );

        // Postfilter
        let c_has_pf = bindings::ec_dec_bit_logp(&mut c_dec, 1);
        println!(
            "After PF flag: nbits={} has_pf={}",
            c_dec.nbits_total, c_has_pf
        );
        if c_has_pf != 0 {
            let c_octave = bindings::ec_dec_uint(&mut c_dec, 6);
            let c_pf_pitch = (16 << c_octave) as i32
                + bindings::ec_dec_bits(&mut c_dec, 4 + c_octave) as i32
                - 1;
            let c_qg = bindings::ec_dec_bits(&mut c_dec, 3);
            let tapset_icdf: [u8; 3] = [2, 1, 0];
            let c_tapset = bindings::ec_dec_icdf(&mut c_dec, tapset_icdf.as_ptr(), 2);
            println!(
                "After PF decode: nbits={} pitch={} gain={} tapset={}",
                c_dec.nbits_total,
                c_pf_pitch,
                3072 * (c_qg as i32 + 1),
                c_tapset
            );
        }

        // Transient
        let c_trans = bindings::ec_dec_bit_logp(&mut c_dec, 3);
        println!(
            "After transient: nbits={} is_trans={}",
            c_dec.nbits_total, c_trans
        );

        // Intra
        let c_intra = bindings::ec_dec_bit_logp(&mut c_dec, 3);
        println!("After intra: nbits={} intra={}", c_dec.nbits_total, c_intra);
    }
}

// ---------------------------------------------------------------------------
// Per-frame decode comparison (diagnostic)
// ---------------------------------------------------------------------------

fn cmd_decode_framecompare(wav_path: &str, bitrate: i32, application: i32) {
    let wav = read_wav(Path::new(wav_path));
    println!(
        "Input: {} ({} Hz, {} ch, {} samples)",
        wav_path,
        wav.sample_rate,
        wav.channels,
        wav.samples.len() / wav.channels as usize
    );

    let sr = wav.sample_rate as i32;
    let ch = wav.channels as i32;

    // Encode with C reference using specified application
    let mut cfg = EncodeConfig::new(sr, ch);
    cfg.bitrate = bitrate;
    cfg.application = application;
    let c_encoded = c_encode_cfg(&wav.samples, &cfg);
    println!("C encoded: {} bytes", c_encoded.len());

    // Now decode frame-by-frame with both C and Rust, comparing each frame
    let frame_size = (sr / 50) as usize; // 20ms
    let frame_samples = frame_size * ch as usize;

    // C decoder
    let c_dec;
    unsafe {
        let mut error: i32 = 0;
        c_dec = bindings::opus_decoder_create(sr, ch, &mut error);
        assert!(!c_dec.is_null() && error == bindings::OPUS_OK);
    }

    // Rust decoder
    use mdopus::opus::decoder::OpusDecoder;
    let mut rust_dec = OpusDecoder::new(sr, ch).expect("Rust decoder init");

    let mut c_pcm = vec![0i16; frame_samples];
    let mut rust_pcm = vec![0i16; frame_samples];
    let mut pos = 0;
    let mut frame_idx = 0;
    let mut first_fail_frame = -1i32;

    while pos + 2 <= c_encoded.len() {
        let pkt_len = u16::from_le_bytes([c_encoded[pos], c_encoded[pos + 1]]) as usize;
        pos += 2;
        if pos + pkt_len > c_encoded.len() {
            break;
        }
        let pkt = &c_encoded[pos..pos + pkt_len];

        // C decode
        let c_ret = unsafe {
            bindings::opus_decode(
                c_dec,
                pkt.as_ptr(),
                pkt_len as i32,
                c_pcm.as_mut_ptr(),
                frame_size as i32,
                0,
            )
        };
        assert!(c_ret >= 0);

        // Rust decode
        let rust_ret = rust_dec
            .decode(Some(pkt), &mut rust_pcm, frame_size as i32, false)
            .expect("Rust decode");

        // Get final range from C decoder (CTL 4031 = OPUS_GET_FINAL_RANGE)
        let mut c_range: u32 = 0;
        unsafe {
            bindings::opus_decoder_ctl(c_dec, 4031i32, &mut c_range as *mut u32);
        }
        let rust_range = rust_dec.get_final_range();

        // Compare
        let c_count = c_ret as usize * ch as usize;
        let r_count = rust_ret as usize * ch as usize;
        let mut max_diff = 0i32;
        let mut first_diff: Option<usize> = None;
        let len = c_count.min(r_count);
        for i in 0..len {
            let d = (c_pcm[i] as i32 - rust_pcm[i] as i32).abs();
            if d > max_diff {
                max_diff = d;
            }
            if d > 0 && first_diff.is_none() {
                first_diff = Some(i);
            }
        }

        let range_match = c_range == rust_range;
        if let Some(fd) = first_diff {
            println!(
                "Frame {:3}: pkt_len={:5} FAIL max_diff={:6} first_diff_at={} (sample {} overall) range: C={:08x} R={:08x} {}",
                frame_idx,
                pkt_len,
                max_diff,
                fd,
                frame_idx * frame_samples + fd,
                c_range,
                rust_range,
                if range_match { "MATCH" } else { "MISMATCH" }
            );
            if first_fail_frame < 0 {
                first_fail_frame = frame_idx as i32;
                // Print first few samples around divergence
                let start = fd.saturating_sub(4);
                let end = (fd + 12).min(len);
                print!("  C:    ");
                for i in start..end {
                    if i == fd {
                        print!("[{:6}]", c_pcm[i]);
                    } else {
                        print!(" {:6} ", c_pcm[i]);
                    }
                }
                println!();
                print!("  Rust: ");
                for i in start..end {
                    if i == fd {
                        print!("[{:6}]", rust_pcm[i]);
                    } else {
                        print!(" {:6} ", rust_pcm[i]);
                    }
                }
                println!();
            }
        } else if !range_match {
            println!(
                "Frame {:3}: pkt_len={:5} PCM match but RANGE MISMATCH: C={:08x} R={:08x}",
                frame_idx, pkt_len, c_range, rust_range
            );
        } else {
            println!(
                "Frame {:3}: pkt_len={:5} PASS ({} samples) range={:08x}",
                frame_idx, pkt_len, len, c_range
            );
        }

        // Compare old_band_e arrays
        let mut c_old_band_e = vec![0i32; 42]; // max 21 bands * 2
        let nb_ebands =
            unsafe { bindings::debug_get_celt_old_band_e(c_dec, c_old_band_e.as_mut_ptr(), 42) }
                as usize;
        let r_old_band_e = rust_dec.debug_get_old_band_e();
        let total_bands = 2 * nb_ebands;

        let mut band_diff = false;
        for i in 0..total_bands.min(r_old_band_e.len()) {
            if c_old_band_e[i] != r_old_band_e[i] {
                if !band_diff {
                    println!(
                        "  oldBandE MISMATCH at band {} (of {}): C={} R={} diff={}",
                        i,
                        total_bands,
                        c_old_band_e[i],
                        r_old_band_e[i],
                        c_old_band_e[i] - r_old_band_e[i]
                    );
                }
                band_diff = true;
            }
        }
        // For the last passing frame and first failing frames, dump all bands
        if frame_idx <= 5 {
            // Print first few bands for C and Rust
            let show = total_bands.min(r_old_band_e.len()).min(10);
            print!("  C  bands[0..{}]:", show);
            for i in 0..show {
                print!(" {}", c_old_band_e[i]);
            }
            println!();
            print!("  R  bands[0..{}]:", show);
            for i in 0..show {
                print!(" {}", r_old_band_e[i]);
            }
            println!();
            // Show second half (ch1 for mono copy)
            print!("  C  bands[21..{}]:", 21 + show.min(5));
            for i in 21..21 + show.min(5) {
                if i < total_bands {
                    print!(" {}", c_old_band_e[i]);
                }
            }
            println!();
            print!("  R  bands[21..{}]:", 21 + show.min(5));
            for i in 21..21 + show.min(5) {
                if i < total_bands {
                    print!(" {}", r_old_band_e[i]);
                }
            }
            println!();
        }

        // Compare old_log_e arrays
        let mut c_old_log_e = vec![0i32; 42];
        let mut c_old_log_e2 = vec![0i32; 42];
        unsafe {
            bindings::debug_get_celt_old_log_e(
                c_dec,
                c_old_log_e.as_mut_ptr(),
                c_old_log_e2.as_mut_ptr(),
                42,
            );
        }
        let r_old_log_e = rust_dec.debug_get_old_log_e();
        let r_old_log_e2 = rust_dec.debug_get_old_log_e2();
        let mut log_e_diff = false;
        for i in 0..total_bands.min(r_old_log_e.len()) {
            if c_old_log_e[i] != r_old_log_e[i] {
                if !log_e_diff {
                    println!(
                        "  oldLogE MISMATCH at band {} (of {}): C={} R={} diff={}",
                        i,
                        total_bands,
                        c_old_log_e[i],
                        r_old_log_e[i],
                        c_old_log_e[i] - r_old_log_e[i]
                    );
                }
                log_e_diff = true;
            }
            if c_old_log_e2[i] != r_old_log_e2[i] {
                if !log_e_diff {
                    println!(
                        "  oldLogE2 MISMATCH at band {} (of {}): C={} R={} diff={}",
                        i,
                        total_bands,
                        c_old_log_e2[i],
                        r_old_log_e2[i],
                        c_old_log_e2[i] - r_old_log_e2[i]
                    );
                }
                log_e_diff = true;
            }
        }

        // Compare postfilter state
        let (r_pf_period, r_pf_period_old, r_pf_gain, r_pf_gain_old, r_pf_tapset, r_pf_tapset_old) =
            rust_dec.debug_get_postfilter();
        let (
            mut c_pf_period,
            mut c_pf_period_old,
            mut c_pf_gain,
            mut c_pf_gain_old,
            mut c_pf_tapset,
            mut c_pf_tapset_old,
        ) = (0i32, 0i32, 0i32, 0i32, 0i32, 0i32);
        unsafe {
            bindings::debug_get_celt_postfilter(
                c_dec,
                &mut c_pf_period,
                &mut c_pf_period_old,
                &mut c_pf_gain,
                &mut c_pf_gain_old,
                &mut c_pf_tapset,
                &mut c_pf_tapset_old,
            );
        }
        if c_pf_period != r_pf_period
            || c_pf_period_old != r_pf_period_old
            || c_pf_gain != r_pf_gain
            || c_pf_gain_old != r_pf_gain_old
            || c_pf_tapset != r_pf_tapset
            || c_pf_tapset_old != r_pf_tapset_old
        {
            println!(
                "  postfilter MISMATCH: C=({},{},{},{},{},{}) R=({},{},{},{},{},{})",
                c_pf_period,
                c_pf_period_old,
                c_pf_gain,
                c_pf_gain_old,
                c_pf_tapset,
                c_pf_tapset_old,
                r_pf_period,
                r_pf_period_old,
                r_pf_gain,
                r_pf_gain_old,
                r_pf_tapset,
                r_pf_tapset_old,
            );
        }

        pos += pkt_len;
        frame_idx += 1;
    }

    // --- Single-frame isolation test ---
    // Decode frame at first_fail_frame from fresh decoders
    if first_fail_frame >= 0 {
        let ff = first_fail_frame as usize;
        println!("\n=== Isolating frame {} for fresh-decoder test ===", ff);

        // Extract the packet for the failing frame
        let mut p = 0usize;
        let mut target_pkt: Option<&[u8]> = None;
        for fidx in 0..=ff {
            if p + 2 > c_encoded.len() {
                break;
            }
            let pl = u16::from_le_bytes([c_encoded[p], c_encoded[p + 1]]) as usize;
            p += 2;
            if fidx == ff {
                target_pkt = Some(&c_encoded[p..p + pl]);
            }
            p += pl;
        }

        if let Some(pkt) = target_pkt {
            // Create fresh decoders
            let fresh_c;
            unsafe {
                let mut error: i32 = 0;
                fresh_c = bindings::opus_decoder_create(sr, ch, &mut error);
                assert!(!fresh_c.is_null() && error == bindings::OPUS_OK);
            }
            let mut fresh_rust = OpusDecoder::new(sr, ch).expect("Rust decoder");

            // Decode all frames up to (but not including) the failing frame
            let mut p2 = 0usize;
            for _fidx in 0..ff {
                if p2 + 2 > c_encoded.len() {
                    break;
                }
                let pl = u16::from_le_bytes([c_encoded[p2], c_encoded[p2 + 1]]) as usize;
                p2 += 2;
                let pk = &c_encoded[p2..p2 + pl];
                // C decode
                unsafe {
                    bindings::opus_decode(
                        fresh_c,
                        pk.as_ptr(),
                        pl as i32,
                        c_pcm.as_mut_ptr(),
                        frame_size as i32,
                        0,
                    );
                }
                // Rust decode
                let _ = fresh_rust.decode(Some(pk), &mut rust_pcm, frame_size as i32, false);
                p2 += pl;
            }

            // Compare state before decoding the failing frame
            let mut c_be = vec![0i32; 42];
            let c_nb =
                unsafe { bindings::debug_get_celt_old_band_e(fresh_c, c_be.as_mut_ptr(), 42) }
                    as usize;
            let r_be = fresh_rust.debug_get_old_band_e();

            println!("State before frame {}:", ff);
            let mut state_match = true;
            for i in 0..(2 * c_nb).min(r_be.len()) {
                if c_be[i] != r_be[i] {
                    println!(
                        "  band[{}]: C={} R={} diff={}",
                        i,
                        c_be[i],
                        r_be[i],
                        c_be[i] - r_be[i]
                    );
                    state_match = false;
                }
            }
            if state_match {
                println!("  All bands match before frame {}", ff);
            }

            // Before decoding frame, get tell from Rust decoder
            // (We can't easily get tell from C mid-decode, so we compare tells post-decode)

            // Now decode the failing frame
            unsafe {
                let ret = bindings::opus_decode(
                    fresh_c,
                    pkt.as_ptr(),
                    pkt.len() as i32,
                    c_pcm.as_mut_ptr(),
                    frame_size as i32,
                    0,
                );
                assert!(ret >= 0);
            }
            let _ = fresh_rust.decode(Some(pkt), &mut rust_pcm, frame_size as i32, false);

            // Compare state after
            let c_nb =
                unsafe { bindings::debug_get_celt_old_band_e(fresh_c, c_be.as_mut_ptr(), 42) }
                    as usize;
            let r_be = fresh_rust.debug_get_old_band_e();
            println!("State after frame {}:", ff);
            for i in 0..(2 * c_nb).min(r_be.len()).min(5) {
                println!(
                    "  band[{}]: C={} R={} diff={}",
                    i,
                    c_be[i],
                    r_be[i],
                    c_be[i] - r_be[i]
                );
            }

            // Test C-side energy decode with the same state
            let mut c_test_bands = vec![0i32; 42];
            let _r_be_slice = fresh_rust.debug_get_old_band_e();
            // Re-fetch the matching pre-decode state (decode the first ff frames again)
            {
                let fresh_c2;
                unsafe {
                    let mut error: i32 = 0;
                    fresh_c2 = bindings::opus_decoder_create(sr, ch, &mut error);
                    assert!(!fresh_c2.is_null() && error == bindings::OPUS_OK);
                }
                let mut p3 = 0usize;
                for _fidx in 0..ff {
                    if p3 + 2 > c_encoded.len() {
                        break;
                    }
                    let pl = u16::from_le_bytes([c_encoded[p3], c_encoded[p3 + 1]]) as usize;
                    p3 += 2;
                    let pk = &c_encoded[p3..p3 + pl];
                    unsafe {
                        bindings::opus_decode(
                            fresh_c2,
                            pk.as_ptr(),
                            pl as i32,
                            c_pcm.as_mut_ptr(),
                            frame_size as i32,
                            0,
                        );
                    }
                    p3 += pl;
                }
                let _pre_nb = unsafe {
                    bindings::debug_get_celt_old_band_e(fresh_c2, c_test_bands.as_mut_ptr(), 42)
                };
                unsafe {
                    bindings::opus_decoder_destroy(fresh_c2);
                }
            }

            // Now test: call C energy decode with the pre-decode state
            // The pkt is the Opus packet (with TOC). We need the CELT payload.
            // For CELT-only mode, TOC is 1 byte, then the rest is payload.
            // Actually, opus_packet_parse strips the TOC. The data passed to
            // decode_frame is already the payload. In our case, the Opus packet
            // has TOC byte 0xFC (CELT-only 20ms), so payload = pkt[1..].
            // But our extract already strips TOC via opus_packet_parse... no,
            // actually the c_encoded format uses our custom framing (len+data).
            // The data IS the full Opus packet including TOC.
            // The decode_frame receives frame_data = payload after TOC.
            // For code 0 (single frame), payload starts at byte 1.
            let celt_payload = &pkt[1..]; // Skip TOC byte
            println!(
                "CELT payload: {} bytes, first bytes: {:02x} {:02x} {:02x} {:02x}",
                celt_payload.len(),
                celt_payload.first().unwrap_or(&0),
                celt_payload.get(1).unwrap_or(&0),
                celt_payload.get(2).unwrap_or(&0),
                celt_payload.get(3).unwrap_or(&0)
            );

            // Call C energy decode
            let mut c_fine = vec![0i32; 21];
            unsafe {
                bindings::debug_c_decode_energy(
                    celt_payload.as_ptr(),
                    celt_payload.len() as i32,
                    c_test_bands.as_mut_ptr(),
                    c_fine.as_mut_ptr(),
                    ch,
                    3, // LM=3 for 960 samples at 48kHz
                );
            }
            println!("C energy-test after coarse:");
            for i in 0..5 {
                println!("  band[{}]: C_test={}", i, c_test_bands[i]);
            }

            // Compare PCM
            let mut pcm_diff = 0i32;
            let mut first_pcm_diff: Option<usize> = None;
            for i in 0..frame_samples {
                let d = (c_pcm[i] as i32 - rust_pcm[i] as i32).abs();
                if d > pcm_diff {
                    pcm_diff = d;
                }
                if d > 0 && first_pcm_diff.is_none() {
                    first_pcm_diff = Some(i);
                }
            }
            println!("PCM: max_diff={} first_diff={:?}", pcm_diff, first_pcm_diff);

            // Get final ranges
            let mut c_r: u32 = 0;
            unsafe {
                bindings::opus_decoder_ctl(fresh_c, 4031i32, &mut c_r as *mut u32);
            }
            let r_r = fresh_rust.get_final_range();
            println!(
                "Range: C={:08x} R={:08x} {}",
                c_r,
                r_r,
                if c_r == r_r { "MATCH" } else { "MISMATCH" }
            );

            unsafe {
                bindings::opus_decoder_destroy(fresh_c);
            }
        }
    }

    unsafe {
        bindings::opus_decoder_destroy(c_dec);
    }
}

// ---------------------------------------------------------------------------
// Benchmark: C vs Rust performance comparison
// ---------------------------------------------------------------------------

struct BenchResult {
    label: String,
    iters: u32,
    total_secs: f64,
    frames: usize,
}

impl BenchResult {
    fn per_iter_ms(&self) -> f64 {
        (self.total_secs * 1000.0) / self.iters as f64
    }
    fn frames_per_sec(&self) -> f64 {
        (self.frames as f64 * self.iters as f64) / self.total_secs
    }
}

fn bench_encode_c(
    pcm: &[i16],
    sample_rate: i32,
    channels: i32,
    bitrate: i32,
    complexity: i32,
    iters: u32,
) -> BenchResult {
    let frame_size = (sample_rate / 50) as usize;
    let samples_per_frame = frame_size * channels as usize;
    let max_packet = 4000;
    let mut packet = vec![0u8; max_packet];
    let num_frames = pcm.len() / samples_per_frame;

    let start = std::time::Instant::now();
    for _ in 0..iters {
        unsafe {
            let mut error: i32 = 0;
            let enc = bindings::opus_encoder_create(
                sample_rate,
                channels,
                bindings::OPUS_APPLICATION_AUDIO,
                &mut error,
            );
            bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_BITRATE_REQUEST, bitrate);
            bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_COMPLEXITY_REQUEST, complexity);
            bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_VBR_REQUEST, 0i32);

            let mut pos = 0;
            while pos + samples_per_frame <= pcm.len() {
                let ret = bindings::opus_encode(
                    enc,
                    pcm[pos..].as_ptr(),
                    frame_size as i32,
                    packet.as_mut_ptr(),
                    max_packet as i32,
                );
                let _ = std::hint::black_box(ret);
                pos += samples_per_frame;
            }
            bindings::opus_encoder_destroy(enc);
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    BenchResult {
        label: "C encode".to_string(),
        iters,
        total_secs: elapsed,
        frames: num_frames,
    }
}

fn bench_encode_rust(
    pcm: &[i16],
    sample_rate: i32,
    channels: i32,
    bitrate: i32,
    complexity: i32,
    iters: u32,
) -> BenchResult {
    use mdopus::opus::encoder::{OPUS_APPLICATION_AUDIO, OpusEncoder};

    let frame_size = (sample_rate / 50) as usize;
    let samples_per_frame = frame_size * channels as usize;
    let max_packet = 4000;
    let mut packet = vec![0u8; max_packet];
    let num_frames = pcm.len() / samples_per_frame;

    let start = std::time::Instant::now();
    for _ in 0..iters {
        let mut enc = OpusEncoder::new(sample_rate, channels, OPUS_APPLICATION_AUDIO).unwrap();
        enc.set_bitrate(bitrate);
        enc.set_complexity(complexity);
        enc.set_vbr(0);

        let mut pos = 0;
        while pos + samples_per_frame <= pcm.len() {
            let ret = enc.encode(
                &pcm[pos..pos + samples_per_frame],
                frame_size as i32,
                &mut packet,
                max_packet as i32,
            );
            let _ = std::hint::black_box(ret);
            pos += samples_per_frame;
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    BenchResult {
        label: "Rust encode".to_string(),
        iters,
        total_secs: elapsed,
        frames: num_frames,
    }
}

fn bench_decode_c(encoded: &[u8], sample_rate: i32, channels: i32, iters: u32) -> BenchResult {
    let frame_size = (sample_rate / 50) as usize;
    let mut pcm = vec![0i16; frame_size * channels as usize];

    // Count frames in encoded stream
    let mut num_frames = 0;
    {
        let mut pos = 0;
        while pos + 2 <= encoded.len() {
            let pkt_len = u16::from_le_bytes([encoded[pos], encoded[pos + 1]]) as usize;
            pos += 2 + pkt_len;
            num_frames += 1;
        }
    }

    let start = std::time::Instant::now();
    for _ in 0..iters {
        unsafe {
            let mut error: i32 = 0;
            let dec = bindings::opus_decoder_create(sample_rate, channels, &mut error);

            let mut pos = 0;
            while pos + 2 <= encoded.len() {
                let pkt_len = u16::from_le_bytes([encoded[pos], encoded[pos + 1]]) as usize;
                pos += 2;
                if pos + pkt_len > encoded.len() {
                    break;
                }
                let ret = bindings::opus_decode(
                    dec,
                    encoded[pos..].as_ptr(),
                    pkt_len as i32,
                    pcm.as_mut_ptr(),
                    frame_size as i32,
                    0,
                );
                let _ = std::hint::black_box(ret);
                pos += pkt_len;
            }
            bindings::opus_decoder_destroy(dec);
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    BenchResult {
        label: "C decode".to_string(),
        iters,
        total_secs: elapsed,
        frames: num_frames,
    }
}

fn bench_decode_rust(encoded: &[u8], sample_rate: i32, channels: i32, iters: u32) -> BenchResult {
    use mdopus::opus::decoder::OpusDecoder;

    let frame_size = (sample_rate / 50) as usize;
    let mut pcm = vec![0i16; frame_size * channels as usize];

    let mut num_frames = 0;
    {
        let mut pos = 0;
        while pos + 2 <= encoded.len() {
            let pkt_len = u16::from_le_bytes([encoded[pos], encoded[pos + 1]]) as usize;
            pos += 2 + pkt_len;
            num_frames += 1;
        }
    }

    let start = std::time::Instant::now();
    for _ in 0..iters {
        let mut dec = OpusDecoder::new(sample_rate, channels).unwrap();

        let mut pos = 0;
        while pos + 2 <= encoded.len() {
            let pkt_len = u16::from_le_bytes([encoded[pos], encoded[pos + 1]]) as usize;
            pos += 2;
            if pos + pkt_len > encoded.len() {
                break;
            }
            let ret = dec.decode(
                Some(&encoded[pos..pos + pkt_len]),
                &mut pcm,
                frame_size as i32,
                false,
            );
            let _ = std::hint::black_box(ret);
            pos += pkt_len;
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    BenchResult {
        label: "Rust decode".to_string(),
        iters,
        total_secs: elapsed,
        frames: num_frames,
    }
}

fn print_bench_report(results: &[BenchResult]) {
    println!("┌──────────────┬────────┬──────────────┬───────────────┐");
    println!("│ Operation    │  Iters │  ms/iter     │  frames/sec   │");
    println!("├──────────────┼────────┼──────────────┼───────────────┤");
    for r in results {
        println!(
            "│ {:<12} │ {:>6} │ {:>10.3}ms │ {:>11.0}   │",
            r.label,
            r.iters,
            r.per_iter_ms(),
            r.frames_per_sec(),
        );
    }
    println!("└──────────────┴────────┴──────────────┴───────────────┘");

    // Print ratios
    println!();
    // Pair up C vs Rust for each operation
    let mut i = 0;
    while i + 1 < results.len() {
        let c_res = &results[i];
        let r_res = &results[i + 1];
        let ratio = r_res.total_secs / c_res.total_secs;
        let op = if c_res.label.contains("encode") {
            "encode"
        } else {
            "decode"
        };
        if ratio > 1.0 {
            println!(
                "  {} : Rust is {:.2}x SLOWER than C ({:.3}ms vs {:.3}ms per iter)",
                op,
                ratio,
                r_res.per_iter_ms(),
                c_res.per_iter_ms(),
            );
        } else {
            println!(
                "  {} : Rust is {:.2}x FASTER than C ({:.3}ms vs {:.3}ms per iter)",
                op,
                1.0 / ratio,
                r_res.per_iter_ms(),
                c_res.per_iter_ms(),
            );
        }
        i += 2;
    }
}

// ---------------------------------------------------------------------------
// PLC (Packet Loss Concealment) comparison
// ---------------------------------------------------------------------------

fn plc_decode_c(
    sr: i32,
    ch: i32,
    frame_size: usize,
    packets: &[Vec<u8>],
    drops: &[usize],
) -> Vec<i16> {
    unsafe {
        let mut error: i32 = 0;
        let dec = bindings::opus_decoder_create(sr, ch, &mut error);
        if dec.is_null() || error != bindings::OPUS_OK {
            eprintln!(
                "ERROR: C opus_decoder_create failed: {}",
                bindings::error_string(error)
            );
            process::exit(1);
        }

        let samples_per_frame = frame_size * ch as usize;
        let mut pcm = vec![0i16; samples_per_frame];
        let mut output = Vec::new();

        for (i, pkt) in packets.iter().enumerate() {
            if drops.contains(&i) {
                // PLC: decode with NULL data
                let ret = bindings::opus_decode(
                    dec,
                    std::ptr::null(),
                    0,
                    pcm.as_mut_ptr(),
                    frame_size as i32,
                    0,
                );
                if ret > 0 {
                    output.extend_from_slice(&pcm[..ret as usize * ch as usize]);
                }
            } else {
                let ret = bindings::opus_decode(
                    dec,
                    pkt.as_ptr(),
                    pkt.len() as i32,
                    pcm.as_mut_ptr(),
                    frame_size as i32,
                    0,
                );
                if ret > 0 {
                    output.extend_from_slice(&pcm[..ret as usize * ch as usize]);
                }
            }
        }

        bindings::opus_decoder_destroy(dec);
        output
    }
}

fn plc_decode_rust(
    sr: i32,
    ch: i32,
    frame_size: usize,
    packets: &[Vec<u8>],
    drops: &[usize],
) -> Vec<i16> {
    use mdopus::opus::decoder::OpusDecoder;

    let mut dec = match OpusDecoder::new(sr, ch) {
        Ok(d) => d,
        Err(code) => {
            eprintln!("ERROR: Rust OpusDecoder::new failed: {}", code);
            process::exit(1);
        }
    };

    let samples_per_frame = frame_size * ch as usize;
    let mut pcm = vec![0i16; samples_per_frame];
    let mut output = Vec::new();

    for (i, pkt) in packets.iter().enumerate() {
        if drops.contains(&i) {
            // PLC: decode with None
            match dec.decode(None, &mut pcm, frame_size as i32, false) {
                Ok(ret) => {
                    output.extend_from_slice(&pcm[..ret as usize * ch as usize]);
                }
                Err(_) => {} // PLC can fail on first frame etc.
            }
        } else {
            match dec.decode(Some(pkt), &mut pcm, frame_size as i32, false) {
                Ok(ret) => output.extend_from_slice(&pcm[..ret as usize * ch as usize]),
                Err(e) => {
                    eprintln!("PLC Rust decode error at frame {}: {}", i, e);
                }
            }
        }
    }

    output
}

fn cmd_plc(wav_path: &str, bitrate: i32) {
    let wav = read_wav(Path::new(wav_path));
    let sr = wav.sample_rate as i32;
    let ch = wav.channels as i32;
    let frame_size = (sr / 50) as usize; // 20ms frames

    // Encode with C encoder (deterministic baseline bitstream)
    let mut cfg = EncodeConfig::new(sr, ch);
    cfg.bitrate = bitrate;
    let encoded = c_encode_cfg(&wav.samples, &cfg);

    // Parse into individual packets
    let packets = parse_packets(&encoded);

    println!(
        "PLC test: {} Hz, {} ch, {} frames, bitrate={}",
        sr,
        ch,
        packets.len(),
        bitrate
    );

    let patterns: &[(&str, Vec<usize>)] = &[
        ("isolated", vec![5, 15, 25, 35, 45]),
        ("burst", vec![20, 21, 22]),
        ("heavy", (0..packets.len()).filter(|i| i % 3 == 2).collect()),
    ];

    let mut all_pass = true;

    for (name, drops) in patterns {
        // Skip patterns that reference frames beyond what we have
        let effective_drops: Vec<usize> = drops
            .iter()
            .copied()
            .filter(|&d| d < packets.len())
            .collect();

        let c_pcm = plc_decode_c(sr, ch, frame_size, &packets, &effective_drops);
        let rust_pcm = plc_decode_rust(sr, ch, frame_size, &packets, &effective_drops);

        let stats = compare_samples(&c_pcm, &rust_pcm);
        if stats.first_diff_offset.is_none() {
            println!(
                "  {}: PASS ({} samples, {} drops)",
                name,
                stats.total,
                effective_drops.len()
            );
        } else {
            println!(
                "  {}: FAIL at sample {} (max_diff={}, {} drops)",
                name,
                stats.first_diff_offset.unwrap(),
                stats.max_diff,
                effective_drops.len()
            );
            all_pass = false;
        }
    }

    if !all_pass {
        process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// FEC (Forward Error Correction) decode helpers
// ---------------------------------------------------------------------------

fn fec_decode_c(
    sr: i32,
    ch: i32,
    frame_size: usize,
    packets: &[Vec<u8>],
    drops: &[usize],
) -> Vec<i16> {
    unsafe {
        let mut error: i32 = 0;
        let dec = bindings::opus_decoder_create(sr, ch, &mut error);
        if dec.is_null() || error != bindings::OPUS_OK {
            eprintln!(
                "ERROR: C opus_decoder_create failed: {}",
                bindings::error_string(error)
            );
            process::exit(1);
        }

        let samples_per_frame = frame_size * ch as usize;
        let mut pcm = vec![0i16; samples_per_frame];
        let mut output = Vec::new();

        for i in 0..packets.len() {
            let is_dropped = drops.contains(&i);
            let prev_dropped = i > 0 && drops.contains(&(i - 1));

            if is_dropped {
                // This frame is dropped — use PLC
                let ret = bindings::opus_decode(
                    dec,
                    std::ptr::null(),
                    0,
                    pcm.as_mut_ptr(),
                    frame_size as i32,
                    0,
                );
                if ret > 0 {
                    output.extend_from_slice(&pcm[..ret as usize * ch as usize]);
                }
            } else if prev_dropped {
                // Previous frame was dropped, current is available.
                // Recover previous frame using FEC from current packet.
                let ret = bindings::opus_decode(
                    dec,
                    packets[i].as_ptr(),
                    packets[i].len() as i32,
                    pcm.as_mut_ptr(),
                    frame_size as i32,
                    1, // decode_fec=1
                );
                if ret > 0 {
                    output.extend_from_slice(&pcm[..ret as usize * ch as usize]);
                }
                // Now decode current frame normally
                let ret = bindings::opus_decode(
                    dec,
                    packets[i].as_ptr(),
                    packets[i].len() as i32,
                    pcm.as_mut_ptr(),
                    frame_size as i32,
                    0,
                );
                if ret > 0 {
                    output.extend_from_slice(&pcm[..ret as usize * ch as usize]);
                }
            } else {
                // Normal decode
                let ret = bindings::opus_decode(
                    dec,
                    packets[i].as_ptr(),
                    packets[i].len() as i32,
                    pcm.as_mut_ptr(),
                    frame_size as i32,
                    0,
                );
                if ret > 0 {
                    output.extend_from_slice(&pcm[..ret as usize * ch as usize]);
                }
            }
        }

        bindings::opus_decoder_destroy(dec);
        output
    }
}

fn fec_decode_rust(
    sr: i32,
    ch: i32,
    frame_size: usize,
    packets: &[Vec<u8>],
    drops: &[usize],
) -> Vec<i16> {
    use mdopus::opus::decoder::OpusDecoder;

    let mut dec = match OpusDecoder::new(sr, ch) {
        Ok(d) => d,
        Err(code) => {
            eprintln!("ERROR: Rust OpusDecoder::new failed: {}", code);
            process::exit(1);
        }
    };

    let samples_per_frame = frame_size * ch as usize;
    let mut pcm = vec![0i16; samples_per_frame];
    let mut output = Vec::new();

    for i in 0..packets.len() {
        let is_dropped = drops.contains(&i);
        let prev_dropped = i > 0 && drops.contains(&(i - 1));

        if is_dropped {
            // This frame is dropped — use PLC
            match dec.decode(None, &mut pcm, frame_size as i32, false) {
                Ok(ret) => output.extend_from_slice(&pcm[..ret as usize * ch as usize]),
                Err(_) => {} // PLC can fail on first frame etc.
            }
        } else if prev_dropped {
            // Previous frame was dropped, current is available.
            // Recover previous frame using FEC from current packet.
            match dec.decode(Some(&packets[i]), &mut pcm, frame_size as i32, true) {
                Ok(ret) => output.extend_from_slice(&pcm[..ret as usize * ch as usize]),
                Err(e) => {
                    eprintln!("FEC Rust FEC-decode error at frame {}: {}", i, e);
                }
            }
            // Now decode current frame normally
            match dec.decode(Some(&packets[i]), &mut pcm, frame_size as i32, false) {
                Ok(ret) => output.extend_from_slice(&pcm[..ret as usize * ch as usize]),
                Err(e) => {
                    eprintln!("FEC Rust normal-decode error at frame {}: {}", i, e);
                }
            }
        } else {
            // Normal decode
            match dec.decode(Some(&packets[i]), &mut pcm, frame_size as i32, false) {
                Ok(ret) => output.extend_from_slice(&pcm[..ret as usize * ch as usize]),
                Err(e) => {
                    eprintln!("FEC Rust decode error at frame {}: {}", i, e);
                }
            }
        }
    }

    output
}

fn cmd_fec(wav_path: &str, bitrate: i32, loss_pct: i32) {
    let wav = read_wav(Path::new(wav_path));
    let sr = wav.sample_rate as i32;
    let ch = wav.channels as i32;
    let frame_size = (sr / 50) as usize;

    println!(
        "FEC test: {} Hz, {} ch, bitrate={}, loss_pct={}",
        sr, ch, bitrate, loss_pct
    );

    let mut all_pass = true;

    // Encode with FEC enabled
    let mut cfg = EncodeConfig::new(sr, ch);
    cfg.bitrate = bitrate;
    cfg.application = bindings::OPUS_APPLICATION_VOIP; // FEC is most relevant for VOIP
    cfg.fec = 1;
    cfg.packet_loss_pct = loss_pct;

    let c_enc = c_encode_cfg(&wav.samples, &cfg);
    let rust_enc = rust_encode_cfg(&wav.samples, &cfg);

    // 1. Compare encoded bytes
    let enc_stats = compare_bytes(&c_enc, &rust_enc);
    if enc_stats.first_diff_offset.is_none() {
        println!("  encode (FEC enabled): PASS ({} bytes)", c_enc.len());
    } else {
        println!(
            "  encode (FEC enabled): FAIL at byte {} (max_diff={})",
            enc_stats.first_diff_offset.unwrap(),
            enc_stats.max_diff
        );
        all_pass = false;
    }

    // 2. FEC recovery test: simulate packet loss and recover via FEC
    let packets = parse_packets(&c_enc);
    let drops: Vec<usize> = vec![5, 10, 15, 20, 25]
        .into_iter()
        .filter(|&d| d < packets.len())
        .collect();

    // Decode with FEC recovery using C decoder
    let c_fec_pcm = fec_decode_c(sr, ch, frame_size, &packets, &drops);
    // Decode with FEC recovery using Rust decoder
    let rust_fec_pcm = fec_decode_rust(sr, ch, frame_size, &packets, &drops);

    let dec_stats = compare_samples(&c_fec_pcm, &rust_fec_pcm);
    if dec_stats.first_diff_offset.is_none() {
        println!(
            "  FEC decode recovery: PASS ({} samples, {} drops)",
            dec_stats.total,
            drops.len()
        );
    } else {
        println!(
            "  FEC decode recovery: FAIL at sample {} (max_diff={}, {} drops)",
            dec_stats.first_diff_offset.unwrap(),
            dec_stats.max_diff,
            drops.len()
        );
        all_pass = false;
    }

    if !all_pass {
        process::exit(1);
    }
}

fn cmd_bench(wav_path: &str, bitrate: i32, complexity: i32, iters: u32) {
    let wav = read_wav(Path::new(wav_path));
    let sr = wav.sample_rate as i32;
    let ch = wav.channels as i32;
    let frame_size = (sr / 50) as usize;
    let samples_per_frame = frame_size * ch as usize;
    let num_frames = wav.samples.len() / samples_per_frame;
    let duration_sec = wav.samples.len() as f64 / (sr as f64 * ch as f64);

    println!("=== mdopus benchmark ===");
    println!(
        "Input : {} ({} Hz, {} ch, {:.2}s, {} frames @ 20ms)",
        wav_path, sr, ch, duration_sec, num_frames,
    );
    println!(
        "Config: bitrate={}, complexity={}, CBR",
        bitrate, complexity
    );
    println!("Iters : {} (each iter encodes/decodes ALL frames)", iters);
    println!();

    // --- Warmup ---
    println!("Warming up (1 iter each)...");
    let _ = bench_encode_c(&wav.samples, sr, ch, bitrate, complexity, 1);
    let _ = bench_encode_rust(&wav.samples, sr, ch, bitrate, complexity, 1);
    println!();

    // --- Encode benchmark ---
    println!("Benchmarking encode...");
    let c_enc = bench_encode_c(&wav.samples, sr, ch, bitrate, complexity, iters);
    let r_enc = bench_encode_rust(&wav.samples, sr, ch, bitrate, complexity, iters);

    // Generate encoded data for decode benchmark (use C encoder output)
    let encoded = c_encode(&wav.samples, sr, ch, bitrate, complexity);

    // --- Decode benchmark ---
    println!("Benchmarking decode...");
    let _ = bench_decode_c(&encoded, sr, ch, 1); // warmup
    let _ = bench_decode_rust(&encoded, sr, ch, 1);

    let c_dec = bench_decode_c(&encoded, sr, ch, iters);
    let r_dec = bench_decode_rust(&encoded, sr, ch, iters);

    println!();
    print_bench_report(&[c_enc, r_enc, c_dec, r_dec]);
}

// ---------------------------------------------------------------------------
// Sweep: exhaustive parameter-space comparison
// ---------------------------------------------------------------------------

struct SweepCase {
    cfg: EncodeConfig,
    label: String,
}

fn resolve_wav_path(sample_rate: i32, channels: i32) -> String {
    format!(
        "tests/vectors/{}hz_{}_noise.wav",
        sample_rate,
        if channels == 1 { "mono" } else { "stereo" }
    )
}

fn sweep_cases() -> Vec<SweepCase> {
    let mut cases = Vec::new();

    // Helper to push a case with a generated label
    macro_rules! push {
        ($sr:expr, $ch:expr, $br:expr, $frame_ms:expr, $vbr:expr, $cx:expr, $app:expr, $app_label:expr) => {{
            let mut cfg = EncodeConfig::new($sr, $ch);
            cfg.bitrate = $br;
            cfg.frame_ms = $frame_ms;
            cfg.vbr = $vbr;
            cfg.complexity = $cx;
            cfg.application = $app;
            let mode = if $sr >= 48000 {
                "CELT"
            } else if $sr <= 12000 {
                "SILK"
            } else {
                "Hybrid"
            };
            let vbr_s = if $vbr == 0 { "CBR" } else { "VBR" };
            let ch_s = if $ch == 1 { "1ch" } else { "2ch" };
            let br_k = if $br >= 1000 {
                format!("{}kbps", $br / 1000)
            } else {
                format!("{}bps", $br)
            };
            let label = format!(
                "{} {}k/{} {} {} {}ms cx{} {}",
                mode,
                $sr / 1000,
                ch_s,
                vbr_s,
                br_k,
                $frame_ms,
                $cx,
                $app_label
            );
            cases.push(SweepCase { cfg, label });
        }};
    }

    let audio = bindings::OPUS_APPLICATION_AUDIO;
    let voip = bindings::OPUS_APPLICATION_VOIP;
    let lowdelay = bindings::OPUS_APPLICATION_RESTRICTED_LOWDELAY;

    // ---------------------------------------------------------------
    // CELT core (48000 Hz, mono)
    // ---------------------------------------------------------------

    // Bitrate sweep at 20ms, cx10, AUDIO, CBR
    for &br in &[24000, 64000, 128000, 510000] {
        push!(48000, 1, br, 20.0, 0, 10, audio, "AUDIO");
    }

    // Frame duration sweep at 64kbps, cx10, AUDIO, CBR
    for &ms in &[2.5_f64, 5.0, 10.0, 20.0] {
        push!(48000, 1, 64000, ms, 0, 10, audio, "AUDIO");
    }

    // VBR at key bitrates
    for &br in &[24000, 64000, 128000] {
        push!(48000, 1, br, 20.0, 1, 10, audio, "AUDIO");
    }

    // Application modes at 64kbps 20ms
    push!(48000, 1, 64000, 20.0, 0, 10, voip, "VOIP");
    push!(48000, 1, 64000, 20.0, 0, 10, lowdelay, "LOWDELAY");

    // Complexity sweep at 64kbps 20ms AUDIO CBR
    for &cx in &[0, 5] {
        push!(48000, 1, 64000, 20.0, 0, cx, audio, "AUDIO");
    }

    // CELT VBR + different complexities
    push!(48000, 1, 64000, 20.0, 1, 0, audio, "AUDIO");
    push!(48000, 1, 64000, 20.0, 1, 5, audio, "AUDIO");

    // CELT short frames + different bitrates
    push!(48000, 1, 128000, 2.5, 0, 10, audio, "AUDIO");
    push!(48000, 1, 510000, 5.0, 0, 10, audio, "AUDIO");
    push!(48000, 1, 24000, 10.0, 0, 10, audio, "AUDIO");

    // CELT LOWDELAY short frames
    push!(48000, 1, 64000, 2.5, 0, 10, lowdelay, "LOWDELAY");
    push!(48000, 1, 64000, 5.0, 0, 10, lowdelay, "LOWDELAY");
    push!(48000, 1, 64000, 10.0, 0, 10, lowdelay, "LOWDELAY");

    // CELT VOIP + VBR
    push!(48000, 1, 64000, 20.0, 1, 10, voip, "VOIP");

    // ---------------------------------------------------------------
    // SILK core (8000 Hz, mono)
    // ---------------------------------------------------------------

    // Bitrate sweep at 20ms, cx10, VOIP, CBR
    for &br in &[6000, 8000, 12000, 24000] {
        push!(8000, 1, br, 20.0, 0, 10, voip, "VOIP");
    }

    // Frame duration sweep at 12kbps
    for &ms in &[10.0_f64, 20.0, 40.0, 60.0] {
        push!(8000, 1, 12000, ms, 0, 10, voip, "VOIP");
    }

    // VBR
    for &br in &[6000, 12000, 24000] {
        push!(8000, 1, br, 20.0, 1, 10, voip, "VOIP");
    }

    // Complexity sweep
    for &cx in &[0, 5] {
        push!(8000, 1, 12000, 20.0, 0, cx, voip, "VOIP");
    }

    // SILK AUDIO mode
    push!(8000, 1, 12000, 20.0, 0, 10, audio, "AUDIO");
    push!(8000, 1, 24000, 20.0, 0, 10, audio, "AUDIO");

    // SILK long frames + VBR
    push!(8000, 1, 12000, 40.0, 1, 10, voip, "VOIP");
    push!(8000, 1, 12000, 60.0, 1, 10, voip, "VOIP");

    // SILK min/max bitrate
    push!(8000, 1, 6000, 60.0, 0, 0, voip, "VOIP");
    push!(8000, 1, 24000, 10.0, 0, 10, voip, "VOIP");

    // ---------------------------------------------------------------
    // Hybrid (16000 Hz, mono)
    // ---------------------------------------------------------------

    // Bitrate sweep crosses SILK/hybrid/CELT boundaries
    for &br in &[12000, 16000, 24000, 32000, 64000] {
        push!(16000, 1, br, 20.0, 0, 10, audio, "AUDIO");
    }

    // Frame duration sweep
    for &ms in &[10.0_f64, 20.0] {
        push!(16000, 1, 32000, ms, 0, 10, audio, "AUDIO");
    }

    // 16kHz long frames (SILK multi-frame at wideband)
    push!(16000, 1, 16000, 40.0, 0, 10, voip, "VOIP");
    push!(16000, 1, 16000, 60.0, 0, 10, voip, "VOIP");

    // VBR
    for &br in &[16000, 32000, 64000] {
        push!(16000, 1, br, 20.0, 1, 10, audio, "AUDIO");
    }

    // Complexity sweep
    for &cx in &[0, 5] {
        push!(16000, 1, 32000, 20.0, 0, cx, audio, "AUDIO");
    }

    // VOIP mode
    push!(16000, 1, 16000, 20.0, 0, 10, voip, "VOIP");
    push!(16000, 1, 32000, 20.0, 0, 10, voip, "VOIP");

    // 24kHz hybrid
    for &br in &[16000, 32000, 64000] {
        push!(24000, 1, br, 20.0, 0, 10, audio, "AUDIO");
    }
    push!(24000, 1, 32000, 20.0, 1, 10, audio, "AUDIO");

    // ---------------------------------------------------------------
    // Stereo
    // ---------------------------------------------------------------

    // CELT stereo
    for &br in &[64000, 128000, 510000] {
        push!(48000, 2, br, 20.0, 0, 10, audio, "AUDIO");
    }
    push!(48000, 2, 128000, 10.0, 0, 10, audio, "AUDIO");
    push!(48000, 2, 128000, 20.0, 1, 10, audio, "AUDIO");
    push!(48000, 2, 64000, 20.0, 0, 5, audio, "AUDIO");

    // SILK stereo
    for &br in &[12000, 24000] {
        push!(8000, 2, br, 20.0, 0, 10, voip, "VOIP");
    }
    push!(8000, 2, 12000, 20.0, 1, 10, voip, "VOIP");
    push!(8000, 2, 24000, 40.0, 0, 10, voip, "VOIP");

    // Hybrid stereo
    for &br in &[24000, 64000] {
        push!(16000, 2, br, 20.0, 0, 10, audio, "AUDIO");
    }
    push!(16000, 2, 32000, 20.0, 1, 10, audio, "AUDIO");
    push!(24000, 2, 64000, 20.0, 0, 10, audio, "AUDIO");

    // Stereo VOIP + LOWDELAY
    push!(48000, 2, 64000, 20.0, 0, 10, voip, "VOIP");
    push!(48000, 2, 64000, 20.0, 0, 10, lowdelay, "LOWDELAY");

    // ---------------------------------------------------------------
    // Application mode comparisons (same config, different app)
    // ---------------------------------------------------------------

    for &app in &[audio, voip, lowdelay] {
        let lbl = match app {
            x if x == audio => "AUDIO",
            x if x == voip => "VOIP",
            _ => "LOWDELAY",
        };
        push!(48000, 1, 64000, 10.0, 0, 10, app, lbl);
    }

    for &app in &[audio, voip] {
        let lbl = if app == audio { "AUDIO" } else { "VOIP" };
        push!(16000, 1, 24000, 20.0, 0, 10, app, lbl);
    }

    // LOWDELAY at different sample rates (2.5ms + 5ms only at 48k; 10/20ms at 16k)
    push!(48000, 1, 128000, 2.5, 0, 10, lowdelay, "LOWDELAY");
    push!(16000, 1, 32000, 10.0, 0, 10, lowdelay, "LOWDELAY");
    push!(16000, 1, 32000, 20.0, 0, 10, lowdelay, "LOWDELAY");

    // LOWDELAY at narrowband (forces CELT at 8kHz)
    push!(8000, 1, 12000, 10.0, 0, 10, lowdelay, "LOWDELAY");
    push!(8000, 1, 12000, 20.0, 0, 10, lowdelay, "LOWDELAY");

    // ---------------------------------------------------------------
    // Signal hints
    // ---------------------------------------------------------------

    // VOICE vs MUSIC at 16k
    {
        let mut cfg = EncodeConfig::new(16000, 1);
        cfg.bitrate = 24000;
        cfg.signal = bindings::OPUS_SIGNAL_VOICE;
        cases.push(SweepCase {
            cfg,
            label: "Hybrid 16k/1ch CBR 24kbps 20ms cx10 AUDIO sig=VOICE".into(),
        });
    }
    {
        let mut cfg = EncodeConfig::new(16000, 1);
        cfg.bitrate = 24000;
        cfg.signal = bindings::OPUS_SIGNAL_MUSIC;
        cases.push(SweepCase {
            cfg,
            label: "Hybrid 16k/1ch CBR 24kbps 20ms cx10 AUDIO sig=MUSIC".into(),
        });
    }
    // VOICE vs MUSIC at 48k
    {
        let mut cfg = EncodeConfig::new(48000, 1);
        cfg.bitrate = 64000;
        cfg.signal = bindings::OPUS_SIGNAL_VOICE;
        cases.push(SweepCase {
            cfg,
            label: "CELT 48k/1ch CBR 64kbps 20ms cx10 AUDIO sig=VOICE".into(),
        });
    }
    {
        let mut cfg = EncodeConfig::new(48000, 1);
        cfg.bitrate = 64000;
        cfg.signal = bindings::OPUS_SIGNAL_MUSIC;
        cases.push(SweepCase {
            cfg,
            label: "CELT 48k/1ch CBR 64kbps 20ms cx10 AUDIO sig=MUSIC".into(),
        });
    }
    // Signal hints with VOIP application
    {
        let mut cfg = EncodeConfig::new(16000, 1);
        cfg.bitrate = 24000;
        cfg.application = voip;
        cfg.signal = bindings::OPUS_SIGNAL_VOICE;
        cases.push(SweepCase {
            cfg,
            label: "Hybrid 16k/1ch CBR 24kbps 20ms cx10 VOIP sig=VOICE".into(),
        });
    }
    {
        let mut cfg = EncodeConfig::new(48000, 1);
        cfg.bitrate = 64000;
        cfg.application = voip;
        cfg.signal = bindings::OPUS_SIGNAL_MUSIC;
        cases.push(SweepCase {
            cfg,
            label: "CELT 48k/1ch CBR 64kbps 20ms cx10 VOIP sig=MUSIC".into(),
        });
    }

    // ---------------------------------------------------------------
    // Bandwidth forcing (48k only, mono)
    // ---------------------------------------------------------------

    for &(bw, bw_label) in &[
        (bindings::OPUS_BANDWIDTH_NARROWBAND, "NB"),
        (bindings::OPUS_BANDWIDTH_WIDEBAND, "WB"),
        (bindings::OPUS_BANDWIDTH_SUPERWIDEBAND, "SWB"),
        (bindings::OPUS_BANDWIDTH_FULLBAND, "FB"),
    ] {
        let mut cfg = EncodeConfig::new(48000, 1);
        cfg.bitrate = 64000;
        cfg.bandwidth = bw;
        let label = format!("CELT 48k/1ch CBR 64kbps 20ms cx10 AUDIO bw={}", bw_label);
        cases.push(SweepCase { cfg, label });
    }

    // MEDIUMBAND bandwidth forcing
    {
        let mut cfg = EncodeConfig::new(48000, 1);
        cfg.bitrate = 64000;
        cfg.bandwidth = bindings::OPUS_BANDWIDTH_MEDIUMBAND;
        cases.push(SweepCase {
            cfg,
            label: "CELT 48k/1ch CBR 64kbps 20ms cx10 AUDIO bw=MB".into(),
        });
    }

    // Bandwidth forcing at lower bitrate
    for &(bw, bw_label) in &[
        (bindings::OPUS_BANDWIDTH_NARROWBAND, "NB"),
        (bindings::OPUS_BANDWIDTH_WIDEBAND, "WB"),
    ] {
        let mut cfg = EncodeConfig::new(48000, 1);
        cfg.bitrate = 24000;
        cfg.bandwidth = bw;
        let label = format!("CELT 48k/1ch CBR 24kbps 20ms cx10 AUDIO bw={}", bw_label);
        cases.push(SweepCase { cfg, label });
    }

    // Max bandwidth limiting
    {
        let mut cfg = EncodeConfig::new(48000, 1);
        cfg.bitrate = 128000;
        cfg.max_bandwidth = bindings::OPUS_BANDWIDTH_WIDEBAND;
        cases.push(SweepCase {
            cfg,
            label: "CELT 48k/1ch CBR 128kbps 20ms cx10 AUDIO maxbw=WB".into(),
        });
    }
    {
        let mut cfg = EncodeConfig::new(48000, 1);
        cfg.bitrate = 128000;
        cfg.max_bandwidth = bindings::OPUS_BANDWIDTH_SUPERWIDEBAND;
        cases.push(SweepCase {
            cfg,
            label: "CELT 48k/1ch CBR 128kbps 20ms cx10 AUDIO maxbw=SWB".into(),
        });
    }

    // Bandwidth forcing at 16kHz
    {
        let mut cfg = EncodeConfig::new(16000, 1);
        cfg.bitrate = 24000;
        cfg.bandwidth = bindings::OPUS_BANDWIDTH_NARROWBAND;
        cases.push(SweepCase {
            cfg,
            label: "Hybrid 16k/1ch CBR 24kbps 20ms cx10 AUDIO bw=NB".into(),
        });
    }
    {
        let mut cfg = EncodeConfig::new(16000, 1);
        cfg.bitrate = 24000;
        cfg.bandwidth = bindings::OPUS_BANDWIDTH_WIDEBAND;
        cases.push(SweepCase {
            cfg,
            label: "Hybrid 16k/1ch CBR 24kbps 20ms cx10 AUDIO bw=WB".into(),
        });
    }

    // ---------------------------------------------------------------
    // Edge cases
    // ---------------------------------------------------------------

    // Minimum bitrate at all sample rates
    for &sr in &[8000, 16000, 24000, 48000] {
        push!(sr, 1, 6000, 20.0, 0, 10, audio, "AUDIO");
    }

    // Maximum bitrate at all sample rates
    for &sr in &[8000, 16000, 24000, 48000] {
        push!(sr, 1, 510000, 20.0, 0, 10, audio, "AUDIO");
    }

    // Complexity 0 at all sample rates
    for &sr in &[8000, 16000, 24000, 48000] {
        push!(sr, 1, 32000, 20.0, 0, 0, audio, "AUDIO");
    }

    // 2.5ms frames at 48k (CELT-only minimum)
    push!(48000, 1, 64000, 2.5, 0, 10, audio, "AUDIO");

    // 60ms frames at 8k (SILK maximum multi-frame)
    push!(8000, 1, 12000, 60.0, 0, 10, voip, "VOIP");

    // 12kHz sample rate (less common)
    push!(12000, 1, 16000, 20.0, 0, 10, audio, "AUDIO");
    push!(12000, 1, 32000, 20.0, 0, 10, voip, "VOIP");
    push!(12000, 2, 32000, 20.0, 0, 10, audio, "AUDIO");

    // FEC enabled (at VOIP mode which benefits from it)
    {
        let mut cfg = EncodeConfig::new(16000, 1);
        cfg.bitrate = 24000;
        cfg.application = voip;
        cfg.fec = 1;
        cfg.packet_loss_pct = 10;
        cases.push(SweepCase {
            cfg,
            label: "Hybrid 16k/1ch CBR 24kbps 20ms cx10 VOIP FEC+10%loss".into(),
        });
    }
    {
        let mut cfg = EncodeConfig::new(8000, 1);
        cfg.bitrate = 12000;
        cfg.application = voip;
        cfg.fec = 1;
        cfg.packet_loss_pct = 20;
        cases.push(SweepCase {
            cfg,
            label: "SILK 8k/1ch CBR 12kbps 20ms cx10 VOIP FEC+20%loss".into(),
        });
    }

    // FEC + stereo
    {
        let mut cfg = EncodeConfig::new(48000, 2);
        cfg.bitrate = 64000;
        cfg.application = voip;
        cfg.fec = 1;
        cfg.packet_loss_pct = 20;
        cases.push(SweepCase {
            cfg,
            label: "CELT 48k/2ch CBR 64kbps 20ms cx10 VOIP FEC+20%loss".into(),
        });
    }
    {
        let mut cfg = EncodeConfig::new(16000, 2);
        cfg.bitrate = 24000;
        cfg.application = voip;
        cfg.fec = 1;
        cfg.packet_loss_pct = 20;
        cases.push(SweepCase {
            cfg,
            label: "Hybrid 16k/2ch CBR 24kbps 20ms cx10 VOIP FEC+20%loss".into(),
        });
    }

    // FEC + long frames
    {
        let mut cfg = EncodeConfig::new(8000, 1);
        cfg.bitrate = 12000;
        cfg.frame_ms = 40.0;
        cfg.application = voip;
        cfg.fec = 1;
        cfg.packet_loss_pct = 20;
        cases.push(SweepCase {
            cfg,
            label: "SILK 8k/1ch CBR 12kbps 40ms cx10 VOIP FEC+20%loss".into(),
        });
    }
    {
        let mut cfg = EncodeConfig::new(8000, 1);
        cfg.bitrate = 12000;
        cfg.frame_ms = 60.0;
        cfg.application = voip;
        cfg.fec = 1;
        cfg.packet_loss_pct = 20;
        cases.push(SweepCase {
            cfg,
            label: "SILK 8k/1ch CBR 12kbps 60ms cx10 VOIP FEC+20%loss".into(),
        });
    }

    // DTX enabled
    {
        let mut cfg = EncodeConfig::new(8000, 1);
        cfg.bitrate = 12000;
        cfg.application = voip;
        cfg.dtx = 1;
        cases.push(SweepCase {
            cfg,
            label: "SILK 8k/1ch CBR 12kbps 20ms cx10 VOIP DTX".into(),
        });
    }
    {
        let mut cfg = EncodeConfig::new(16000, 1);
        cfg.bitrate = 24000;
        cfg.application = voip;
        cfg.dtx = 1;
        cases.push(SweepCase {
            cfg,
            label: "Hybrid 16k/1ch CBR 24kbps 20ms cx10 VOIP DTX".into(),
        });
    }

    // Prediction disabled
    {
        let mut cfg = EncodeConfig::new(48000, 1);
        cfg.bitrate = 64000;
        cfg.prediction_disabled = 1;
        cases.push(SweepCase {
            cfg,
            label: "CELT 48k/1ch CBR 64kbps 20ms cx10 AUDIO nopred".into(),
        });
    }

    // Phase inversion disabled (stereo)
    {
        let mut cfg = EncodeConfig::new(48000, 2);
        cfg.bitrate = 128000;
        cfg.phase_inversion_disabled = 1;
        cases.push(SweepCase {
            cfg,
            label: "CELT 48k/2ch CBR 128kbps 20ms cx10 AUDIO nophase".into(),
        });
    }

    // ---------------------------------------------------------------
    // Additional CELT coverage
    // ---------------------------------------------------------------

    // More bitrate/frame-size combos at 48k
    push!(48000, 1, 32000, 20.0, 0, 10, audio, "AUDIO");
    push!(48000, 1, 96000, 20.0, 0, 10, audio, "AUDIO");
    push!(48000, 1, 256000, 20.0, 0, 10, audio, "AUDIO");
    push!(48000, 1, 32000, 10.0, 0, 10, audio, "AUDIO");
    push!(48000, 1, 128000, 10.0, 0, 10, audio, "AUDIO");
    push!(48000, 1, 128000, 5.0, 0, 10, audio, "AUDIO");
    push!(48000, 1, 256000, 2.5, 0, 10, audio, "AUDIO");

    // CELT VBR + various frame sizes
    push!(48000, 1, 64000, 10.0, 1, 10, audio, "AUDIO");
    push!(48000, 1, 64000, 5.0, 1, 10, audio, "AUDIO");
    push!(48000, 1, 128000, 20.0, 1, 10, audio, "AUDIO");

    // CELT complexities with different bitrates
    push!(48000, 1, 24000, 20.0, 0, 0, audio, "AUDIO");
    push!(48000, 1, 128000, 20.0, 0, 0, audio, "AUDIO");
    push!(48000, 1, 24000, 20.0, 0, 5, audio, "AUDIO");
    push!(48000, 1, 128000, 20.0, 0, 5, audio, "AUDIO");

    // CELT VOIP at various bitrates
    push!(48000, 1, 24000, 20.0, 0, 10, voip, "VOIP");
    push!(48000, 1, 128000, 20.0, 0, 10, voip, "VOIP");
    push!(48000, 1, 128000, 10.0, 0, 10, voip, "VOIP");

    // CELT LOWDELAY VBR
    push!(48000, 1, 64000, 10.0, 1, 10, lowdelay, "LOWDELAY");
    push!(48000, 1, 128000, 5.0, 0, 10, lowdelay, "LOWDELAY");

    // ---------------------------------------------------------------
    // Additional SILK coverage
    // ---------------------------------------------------------------

    // SILK at different complexities with different bitrates
    push!(8000, 1, 6000, 20.0, 0, 0, voip, "VOIP");
    push!(8000, 1, 24000, 20.0, 0, 0, voip, "VOIP");
    push!(8000, 1, 6000, 20.0, 0, 5, voip, "VOIP");
    push!(8000, 1, 24000, 20.0, 0, 5, voip, "VOIP");

    // SILK longer frames with different bitrates
    push!(8000, 1, 6000, 40.0, 0, 10, voip, "VOIP");
    push!(8000, 1, 24000, 40.0, 0, 10, voip, "VOIP");
    push!(8000, 1, 8000, 60.0, 0, 10, voip, "VOIP");
    push!(8000, 1, 24000, 60.0, 0, 10, voip, "VOIP");

    // SILK VBR longer frames
    push!(8000, 1, 12000, 40.0, 1, 5, voip, "VOIP");
    push!(8000, 1, 8000, 60.0, 1, 10, voip, "VOIP");

    // SILK AUDIO mode at more bitrates
    push!(8000, 1, 6000, 20.0, 0, 10, audio, "AUDIO");
    push!(8000, 1, 8000, 20.0, 0, 10, audio, "AUDIO");

    // SILK 10ms frames
    push!(8000, 1, 8000, 10.0, 0, 10, voip, "VOIP");
    push!(8000, 1, 24000, 10.0, 0, 5, voip, "VOIP");

    // ---------------------------------------------------------------
    // Additional Hybrid coverage
    // ---------------------------------------------------------------

    // 16k more bitrate/complexity combos
    push!(16000, 1, 12000, 20.0, 0, 0, audio, "AUDIO");
    push!(16000, 1, 64000, 20.0, 0, 0, audio, "AUDIO");
    push!(16000, 1, 12000, 20.0, 0, 5, audio, "AUDIO");
    push!(16000, 1, 64000, 20.0, 0, 5, audio, "AUDIO");

    // 16k VBR + different complexities
    push!(16000, 1, 24000, 20.0, 1, 0, audio, "AUDIO");
    push!(16000, 1, 24000, 20.0, 1, 5, audio, "AUDIO");

    // 16k 10ms more bitrates
    push!(16000, 1, 16000, 10.0, 0, 10, audio, "AUDIO");
    push!(16000, 1, 64000, 10.0, 0, 10, audio, "AUDIO");

    // 16k VOIP more combos
    push!(16000, 1, 12000, 20.0, 0, 10, voip, "VOIP");
    push!(16000, 1, 64000, 20.0, 0, 10, voip, "VOIP");
    push!(16000, 1, 16000, 20.0, 1, 10, voip, "VOIP");

    // 24k more combos
    push!(24000, 1, 12000, 20.0, 0, 10, audio, "AUDIO");
    push!(24000, 1, 128000, 20.0, 0, 10, audio, "AUDIO");
    push!(24000, 1, 64000, 20.0, 1, 10, audio, "AUDIO");
    push!(24000, 1, 32000, 10.0, 0, 10, audio, "AUDIO");

    // 24k VOIP
    push!(24000, 1, 32000, 20.0, 0, 10, voip, "VOIP");
    push!(24000, 1, 16000, 20.0, 0, 10, voip, "VOIP");

    // ---------------------------------------------------------------
    // Additional stereo coverage
    // ---------------------------------------------------------------

    // CELT stereo more bitrates/frames
    push!(48000, 2, 32000, 20.0, 0, 10, audio, "AUDIO");
    push!(48000, 2, 256000, 20.0, 0, 10, audio, "AUDIO");
    push!(48000, 2, 64000, 10.0, 0, 10, audio, "AUDIO");
    push!(48000, 2, 128000, 5.0, 0, 10, audio, "AUDIO");
    push!(48000, 2, 64000, 20.0, 1, 10, audio, "AUDIO");
    push!(48000, 2, 128000, 20.0, 0, 0, audio, "AUDIO");
    push!(48000, 2, 128000, 20.0, 0, 5, audio, "AUDIO");

    // SILK stereo more combos
    push!(8000, 2, 6000, 20.0, 0, 10, voip, "VOIP");
    push!(8000, 2, 24000, 20.0, 1, 10, voip, "VOIP");
    push!(8000, 2, 12000, 40.0, 0, 10, voip, "VOIP");

    // Hybrid stereo more combos
    push!(16000, 2, 32000, 20.0, 0, 10, audio, "AUDIO");
    push!(16000, 2, 64000, 20.0, 1, 10, audio, "AUDIO");
    push!(16000, 2, 16000, 20.0, 0, 10, voip, "VOIP");

    // 24k stereo
    push!(24000, 2, 32000, 20.0, 0, 10, audio, "AUDIO");
    push!(24000, 2, 128000, 20.0, 0, 10, audio, "AUDIO");

    // 12k stereo
    push!(12000, 2, 24000, 20.0, 0, 10, audio, "AUDIO");
    push!(12000, 2, 64000, 20.0, 0, 10, voip, "VOIP");

    // ---------------------------------------------------------------
    // Additional FEC / DTX / special mode combos
    // ---------------------------------------------------------------

    // FEC at different rates
    {
        let mut cfg = EncodeConfig::new(8000, 1);
        cfg.bitrate = 24000;
        cfg.application = voip;
        cfg.fec = 1;
        cfg.packet_loss_pct = 5;
        cases.push(SweepCase {
            cfg,
            label: "SILK 8k/1ch CBR 24kbps 20ms cx10 VOIP FEC+5%loss".into(),
        });
    }
    {
        let mut cfg = EncodeConfig::new(48000, 1);
        cfg.bitrate = 64000;
        cfg.application = voip;
        cfg.fec = 1;
        cfg.packet_loss_pct = 10;
        cases.push(SweepCase {
            cfg,
            label: "CELT 48k/1ch CBR 64kbps 20ms cx10 VOIP FEC+10%loss".into(),
        });
    }

    // DTX at CELT rate
    {
        let mut cfg = EncodeConfig::new(48000, 1);
        cfg.bitrate = 64000;
        cfg.application = voip;
        cfg.dtx = 1;
        cases.push(SweepCase {
            cfg,
            label: "CELT 48k/1ch CBR 64kbps 20ms cx10 VOIP DTX".into(),
        });
    }

    // Force channels (force mono in stereo encoder)
    {
        let mut cfg = EncodeConfig::new(48000, 2);
        cfg.bitrate = 128000;
        cfg.force_channels = 1;
        cases.push(SweepCase {
            cfg,
            label: "CELT 48k/2ch CBR 128kbps 20ms cx10 AUDIO forcemono".into(),
        });
    }
    {
        let mut cfg = EncodeConfig::new(48000, 2);
        cfg.bitrate = 128000;
        cfg.force_channels = 2;
        cases.push(SweepCase {
            cfg,
            label: "CELT 48k/2ch CBR 128kbps 20ms cx10 AUDIO forcestereo".into(),
        });
    }

    // LSB depth variations
    {
        let mut cfg = EncodeConfig::new(48000, 1);
        cfg.bitrate = 64000;
        cfg.lsb_depth = 16;
        cases.push(SweepCase {
            cfg,
            label: "CELT 48k/1ch CBR 64kbps 20ms cx10 AUDIO lsb=16".into(),
        });
    }
    {
        let mut cfg = EncodeConfig::new(48000, 1);
        cfg.bitrate = 64000;
        cfg.lsb_depth = 8;
        cases.push(SweepCase {
            cfg,
            label: "CELT 48k/1ch CBR 64kbps 20ms cx10 AUDIO lsb=8".into(),
        });
    }

    // VBR constraint
    {
        let mut cfg = EncodeConfig::new(48000, 1);
        cfg.bitrate = 64000;
        cfg.vbr = 1;
        cfg.vbr_constraint = 1;
        cases.push(SweepCase {
            cfg,
            label: "CELT 48k/1ch CVBR 64kbps 20ms cx10 AUDIO".into(),
        });
    }
    {
        let mut cfg = EncodeConfig::new(16000, 1);
        cfg.bitrate = 24000;
        cfg.vbr = 1;
        cfg.vbr_constraint = 1;
        cases.push(SweepCase {
            cfg,
            label: "Hybrid 16k/1ch CVBR 24kbps 20ms cx10 AUDIO".into(),
        });
    }
    {
        let mut cfg = EncodeConfig::new(8000, 1);
        cfg.bitrate = 12000;
        cfg.application = voip;
        cfg.vbr = 1;
        cfg.vbr_constraint = 1;
        cases.push(SweepCase {
            cfg,
            label: "SILK 8k/1ch CVBR 12kbps 20ms cx10 VOIP".into(),
        });
    }

    cases
}

fn cmd_sweep(filter: Option<&str>, stop_on_fail: bool) {
    let cases = sweep_cases();
    let total = cases.len();
    let filtered: Vec<&SweepCase> = if let Some(f) = filter {
        let f_lower = f.to_lowercase();
        cases
            .iter()
            .filter(|c| c.label.to_lowercase().contains(&f_lower))
            .collect()
    } else {
        cases.iter().collect()
    };

    println!(
        "sweep: {} configurations ({} after filter)",
        total,
        filtered.len()
    );
    println!();

    let mut pass = 0;
    let mut fail = 0;
    let mut skip = 0;

    for (i, case) in filtered.iter().enumerate() {
        // Load signal: use generated noise/silence for DTX, WAV file otherwise
        let pcm = if case.cfg.dtx != 0 {
            generate_dtx_signal(case.cfg.sample_rate, case.cfg.channels, 1.0, 42)
        } else {
            let wav_path = resolve_wav_path(case.cfg.sample_rate, case.cfg.channels);
            let wav = read_wav(Path::new(&wav_path));
            wav.samples
        };

        // Encode with both
        let c_enc = c_encode_cfg(&pcm, &case.cfg);
        let rust_enc = rust_encode_cfg(&pcm, &case.cfg);

        if rust_enc.is_empty() {
            println!("[{:4}/{}] SKIP  {}", i + 1, filtered.len(), case.label);
            skip += 1;
            continue;
        }

        // Compare encoded bytes
        let enc_match = c_enc == rust_enc;

        // Decode C-encoded data with both decoders
        let c_dec = c_decode_cfg(&c_enc, &case.cfg);
        let rust_dec = rust_decode_cfg(&c_enc, &case.cfg);
        let dec_match = c_dec == rust_dec;

        if enc_match && dec_match {
            println!("[{:4}/{}] PASS  {}", i + 1, filtered.len(), case.label);
            pass += 1;
        } else {
            println!("[{:4}/{}] FAIL  {}", i + 1, filtered.len(), case.label);
            if !enc_match {
                let stats = compare_bytes(&c_enc, &rust_enc);
                println!(
                    "  encode: {} bytes differ (first at offset {}, max_diff={})",
                    stats.total - stats.matching,
                    stats.first_diff_offset.unwrap_or(0),
                    stats.max_diff
                );
            }
            if !dec_match {
                let stats = compare_samples(&c_dec, &rust_dec);
                println!(
                    "  decode: {} samples differ (first at {}, max_diff={})",
                    stats.total - stats.matching,
                    stats.first_diff_offset.unwrap_or(0),
                    stats.max_diff
                );
            }
            fail += 1;
            if stop_on_fail {
                println!("\nsweep: stopped on first failure");
                break;
            }
        }
    }

    println!();
    println!("sweep: {} PASS, {} FAIL, {} SKIP", pass, fail, skip);

    if fail > 0 {
        process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// DTX (Discontinuous Transmission) test
// ---------------------------------------------------------------------------

fn dtx_test_one(sr: i32, ch: i32, pcm: &[i16], bitrate: i32) -> bool {
    println!("DTX test: {} Hz, {} ch, bitrate={}", sr, ch, bitrate);

    let mut all_pass = true;

    let mut cfg = EncodeConfig::new(sr, ch);
    cfg.bitrate = bitrate;
    cfg.application = bindings::OPUS_APPLICATION_VOIP; // DTX most relevant for VOIP
    cfg.dtx = 1;

    let c_enc = c_encode_cfg(pcm, &cfg);
    let rust_enc = rust_encode_cfg(pcm, &cfg);

    // Compare encoded bytes
    let enc_stats = compare_bytes(&c_enc, &rust_enc);
    if enc_stats.first_diff_offset.is_none() {
        println!("  encode: PASS ({} bytes)", c_enc.len());
    } else {
        println!(
            "  encode: FAIL at byte {} (max_diff={})",
            enc_stats.first_diff_offset.unwrap(),
            enc_stats.max_diff
        );
        all_pass = false;
    }

    // Analyze packet sizes to verify DTX behavior
    let c_packets = parse_packets(&c_enc);
    let r_packets = parse_packets(&rust_enc);

    let c_dtx_frames = c_packets.iter().filter(|p| p.len() <= 2).count();
    let r_dtx_frames = r_packets.iter().filter(|p| p.len() <= 2).count();
    println!(
        "  C DTX frames: {}/{}, Rust DTX frames: {}/{}",
        c_dtx_frames,
        c_packets.len(),
        r_dtx_frames,
        r_packets.len()
    );

    // Decode both and compare
    let c_dec = c_decode_cfg(&c_enc, &cfg);
    let rust_dec = rust_decode_cfg(&c_enc, &cfg);

    let dec_stats = compare_samples(&c_dec, &rust_dec);
    if dec_stats.first_diff_offset.is_none() {
        println!("  decode: PASS ({} samples)", dec_stats.total);
    } else {
        println!(
            "  decode: FAIL at sample {} (max_diff={})",
            dec_stats.first_diff_offset.unwrap(),
            dec_stats.max_diff
        );
        all_pass = false;
    }

    all_pass
}

fn cmd_dtx(wav_path: &str, bitrate: i32) {
    let mut all_pass = true;

    if wav_path == "generate" {
        // Generate test signals at multiple sample rates
        for &(test_sr, test_ch) in &[(48000, 1), (16000, 1), (8000, 1)] {
            let signal = generate_dtx_signal(test_sr, test_ch, 2.0, 42);
            if !dtx_test_one(test_sr, test_ch, &signal, bitrate) {
                all_pass = false;
            }
        }
        if !all_pass {
            process::exit(1);
        }
        return;
    }

    let wav = read_wav(Path::new(wav_path));
    if !dtx_test_one(
        wav.sample_rate as i32,
        wav.channels as i32,
        &wav.samples,
        bitrate,
    ) {
        process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// API validation: compare C and Rust error handling for invalid inputs
// ---------------------------------------------------------------------------

fn cmd_api() {
    println!("API validation: testing error handling parity between C and Rust");
    println!();
    let mut pass = 0u32;
    let mut fail = 0u32;

    // -----------------------------------------------------------------------
    // 1. Encoder creation -- invalid and valid parameters
    // -----------------------------------------------------------------------

    println!("--- Encoder creation ---");

    macro_rules! test_enc_create {
        ($sr:expr, $ch:expr, $app:expr, $label:expr) => {{
            let c_result = unsafe {
                let mut err: i32 = 0;
                let enc = bindings::opus_encoder_create($sr, $ch, $app, &mut err);
                if !enc.is_null() {
                    bindings::opus_encoder_destroy(enc);
                }
                err
            };
            let r_result = match mdopus::opus::encoder::OpusEncoder::new($sr, $ch, $app) {
                Ok(_) => 0i32,
                Err(e) => e,
            };
            // Both succeed or both fail
            let ok = (c_result == 0) == (r_result == 0);
            if ok {
                println!("  PASS  {}: C={}, Rust={}", $label, c_result, r_result);
                pass += 1;
            } else {
                println!("  FAIL  {}: C={}, Rust={}", $label, c_result, r_result);
                fail += 1;
            }
        }};
    }

    let audio = bindings::OPUS_APPLICATION_AUDIO;
    let voip = bindings::OPUS_APPLICATION_VOIP;
    let lowdelay = bindings::OPUS_APPLICATION_RESTRICTED_LOWDELAY;

    // Invalid sample rates
    test_enc_create!(44100, 1, audio, "enc_create(44100, 1, AUDIO)");
    test_enc_create!(0, 1, audio, "enc_create(0, 1, AUDIO)");
    test_enc_create!(-1, 1, audio, "enc_create(-1, 1, AUDIO)");
    test_enc_create!(96000, 1, audio, "enc_create(96000, 1, AUDIO)");
    test_enc_create!(11025, 1, audio, "enc_create(11025, 1, AUDIO)");

    // Invalid channels
    test_enc_create!(48000, 0, audio, "enc_create(48000, 0, AUDIO)");
    test_enc_create!(48000, 3, audio, "enc_create(48000, 3, AUDIO)");
    test_enc_create!(48000, -1, audio, "enc_create(48000, -1, AUDIO)");
    test_enc_create!(48000, 255, audio, "enc_create(48000, 255, AUDIO)");

    // Invalid application
    test_enc_create!(48000, 1, 9999, "enc_create(48000, 1, 9999)");
    test_enc_create!(48000, 1, 0, "enc_create(48000, 1, 0)");
    test_enc_create!(48000, 1, -1, "enc_create(48000, 1, -1)");

    // Valid cases -- all five valid sample rates
    test_enc_create!(48000, 1, audio, "enc_create(48000, 1, AUDIO)");
    test_enc_create!(48000, 2, voip, "enc_create(48000, 2, VOIP)");
    test_enc_create!(8000, 1, lowdelay, "enc_create(8000, 1, LOWDELAY)");
    test_enc_create!(12000, 1, audio, "enc_create(12000, 1, AUDIO)");
    test_enc_create!(16000, 2, voip, "enc_create(16000, 2, VOIP)");
    test_enc_create!(24000, 1, audio, "enc_create(24000, 1, AUDIO)");

    println!();

    // -----------------------------------------------------------------------
    // 2. Decoder creation -- invalid and valid parameters
    // -----------------------------------------------------------------------

    println!("--- Decoder creation ---");

    macro_rules! test_dec_create {
        ($sr:expr, $ch:expr, $label:expr) => {{
            let c_result = unsafe {
                let mut err: i32 = 0;
                let dec = bindings::opus_decoder_create($sr, $ch, &mut err);
                if !dec.is_null() {
                    bindings::opus_decoder_destroy(dec);
                }
                err
            };
            let r_result = match mdopus::opus::decoder::OpusDecoder::new($sr, $ch) {
                Ok(_) => 0i32,
                Err(e) => e,
            };
            let ok = (c_result == 0) == (r_result == 0);
            if ok {
                println!("  PASS  {}: C={}, Rust={}", $label, c_result, r_result);
                pass += 1;
            } else {
                println!("  FAIL  {}: C={}, Rust={}", $label, c_result, r_result);
                fail += 1;
            }
        }};
    }

    // Invalid sample rates
    test_dec_create!(44100, 1, "dec_create(44100, 1)");
    test_dec_create!(0, 1, "dec_create(0, 1)");
    test_dec_create!(-1, 1, "dec_create(-1, 1)");
    test_dec_create!(96000, 1, "dec_create(96000, 1)");

    // Invalid channels
    test_dec_create!(48000, 0, "dec_create(48000, 0)");
    test_dec_create!(48000, 3, "dec_create(48000, 3)");
    test_dec_create!(48000, -1, "dec_create(48000, -1)");

    // Valid cases
    test_dec_create!(48000, 1, "dec_create(48000, 1)");
    test_dec_create!(48000, 2, "dec_create(48000, 2)");
    test_dec_create!(8000, 1, "dec_create(8000, 1)");
    test_dec_create!(8000, 2, "dec_create(8000, 2)");
    test_dec_create!(12000, 1, "dec_create(12000, 1)");
    test_dec_create!(16000, 2, "dec_create(16000, 2)");
    test_dec_create!(24000, 1, "dec_create(24000, 1)");

    println!();

    // -----------------------------------------------------------------------
    // 3. Encoder CTL -- invalid values
    // -----------------------------------------------------------------------

    println!("--- Encoder CTL ---");

    // Create a valid C encoder and Rust encoder for CTL tests
    let c_enc = unsafe {
        let mut err: i32 = 0;
        let enc = bindings::opus_encoder_create(48000, 2, audio, &mut err);
        assert!(
            !enc.is_null() && err == 0,
            "C encoder creation failed for CTL tests"
        );
        enc
    };
    let mut r_enc = mdopus::opus::encoder::OpusEncoder::new(48000, 2, audio)
        .expect("Rust encoder creation failed for CTL tests");

    macro_rules! test_ctl {
        ($c_req:expr, $c_val:expr, $rust_call:expr, $label:expr) => {{
            let c_ret = unsafe { bindings::opus_encoder_ctl(c_enc, $c_req, $c_val) };
            let r_ret: i32 = $rust_call;
            // Both succeed (==0) or both fail (!=0)
            let ok = (c_ret == 0) == (r_ret == 0);
            if ok {
                println!("  PASS  {}: C={}, Rust={}", $label, c_ret, r_ret);
                pass += 1;
            } else {
                println!("  FAIL  {}: C={}, Rust={}", $label, c_ret, r_ret);
                fail += 1;
            }
        }};
    }

    // Bitrate
    test_ctl!(
        bindings::OPUS_SET_BITRATE_REQUEST,
        -2i32,
        r_enc.set_bitrate(-2),
        "set_bitrate(-2)"
    );
    test_ctl!(
        bindings::OPUS_SET_BITRATE_REQUEST,
        300i32,
        r_enc.set_bitrate(300),
        "set_bitrate(300)"
    );
    test_ctl!(
        bindings::OPUS_SET_BITRATE_REQUEST,
        2000000i32,
        r_enc.set_bitrate(2000000),
        "set_bitrate(2000000)"
    );
    test_ctl!(
        bindings::OPUS_SET_BITRATE_REQUEST,
        64000i32,
        r_enc.set_bitrate(64000),
        "set_bitrate(64000) [valid]"
    );
    test_ctl!(
        bindings::OPUS_SET_BITRATE_REQUEST,
        500i32,
        r_enc.set_bitrate(500),
        "set_bitrate(500) [min valid]"
    );

    // Complexity
    test_ctl!(
        bindings::OPUS_SET_COMPLEXITY_REQUEST,
        -1i32,
        r_enc.set_complexity(-1),
        "set_complexity(-1)"
    );
    test_ctl!(
        bindings::OPUS_SET_COMPLEXITY_REQUEST,
        11i32,
        r_enc.set_complexity(11),
        "set_complexity(11)"
    );
    test_ctl!(
        bindings::OPUS_SET_COMPLEXITY_REQUEST,
        0i32,
        r_enc.set_complexity(0),
        "set_complexity(0) [valid]"
    );
    test_ctl!(
        bindings::OPUS_SET_COMPLEXITY_REQUEST,
        10i32,
        r_enc.set_complexity(10),
        "set_complexity(10) [valid]"
    );

    // VBR
    test_ctl!(
        bindings::OPUS_SET_VBR_REQUEST,
        -1i32,
        r_enc.set_vbr(-1),
        "set_vbr(-1)"
    );
    test_ctl!(
        bindings::OPUS_SET_VBR_REQUEST,
        2i32,
        r_enc.set_vbr(2),
        "set_vbr(2)"
    );
    test_ctl!(
        bindings::OPUS_SET_VBR_REQUEST,
        0i32,
        r_enc.set_vbr(0),
        "set_vbr(0) [valid]"
    );
    test_ctl!(
        bindings::OPUS_SET_VBR_REQUEST,
        1i32,
        r_enc.set_vbr(1),
        "set_vbr(1) [valid]"
    );

    // VBR constraint
    test_ctl!(
        bindings::OPUS_SET_VBR_CONSTRAINT_REQUEST,
        -1i32,
        r_enc.set_vbr_constraint(-1),
        "set_vbr_constraint(-1)"
    );
    test_ctl!(
        bindings::OPUS_SET_VBR_CONSTRAINT_REQUEST,
        2i32,
        r_enc.set_vbr_constraint(2),
        "set_vbr_constraint(2)"
    );

    // Inband FEC
    test_ctl!(
        bindings::OPUS_SET_INBAND_FEC_REQUEST,
        -1i32,
        r_enc.set_inband_fec(-1),
        "set_inband_fec(-1)"
    );
    test_ctl!(
        bindings::OPUS_SET_INBAND_FEC_REQUEST,
        3i32,
        r_enc.set_inband_fec(3),
        "set_inband_fec(3)"
    );

    // Packet loss percentage
    test_ctl!(
        bindings::OPUS_SET_PACKET_LOSS_PERC_REQUEST,
        -1i32,
        r_enc.set_packet_loss_perc(-1),
        "set_packet_loss_perc(-1)"
    );
    test_ctl!(
        bindings::OPUS_SET_PACKET_LOSS_PERC_REQUEST,
        101i32,
        r_enc.set_packet_loss_perc(101),
        "set_packet_loss_perc(101)"
    );

    // DTX
    test_ctl!(
        bindings::OPUS_SET_DTX_REQUEST,
        -1i32,
        r_enc.set_dtx(-1),
        "set_dtx(-1)"
    );
    test_ctl!(
        bindings::OPUS_SET_DTX_REQUEST,
        2i32,
        r_enc.set_dtx(2),
        "set_dtx(2)"
    );

    // Signal
    test_ctl!(
        bindings::OPUS_SET_SIGNAL_REQUEST,
        0i32,
        r_enc.set_signal(0),
        "set_signal(0)"
    );
    test_ctl!(
        bindings::OPUS_SET_SIGNAL_REQUEST,
        9999i32,
        r_enc.set_signal(9999),
        "set_signal(9999)"
    );
    test_ctl!(
        bindings::OPUS_SET_SIGNAL_REQUEST,
        bindings::OPUS_SIGNAL_VOICE,
        r_enc.set_signal(mdopus::opus::encoder::OPUS_SIGNAL_VOICE),
        "set_signal(VOICE) [valid]"
    );

    // Bandwidth
    test_ctl!(
        bindings::OPUS_SET_BANDWIDTH_REQUEST,
        0i32,
        r_enc.set_bandwidth(0),
        "set_bandwidth(0)"
    );
    test_ctl!(
        bindings::OPUS_SET_BANDWIDTH_REQUEST,
        9999i32,
        r_enc.set_bandwidth(9999),
        "set_bandwidth(9999)"
    );

    // Max bandwidth
    test_ctl!(
        bindings::OPUS_SET_MAX_BANDWIDTH_REQUEST,
        0i32,
        r_enc.set_max_bandwidth(0),
        "set_max_bandwidth(0)"
    );
    test_ctl!(
        bindings::OPUS_SET_MAX_BANDWIDTH_REQUEST,
        9999i32,
        r_enc.set_max_bandwidth(9999),
        "set_max_bandwidth(9999)"
    );

    // LSB depth
    test_ctl!(
        bindings::OPUS_SET_LSB_DEPTH_REQUEST,
        7i32,
        r_enc.set_lsb_depth(7),
        "set_lsb_depth(7)"
    );
    test_ctl!(
        bindings::OPUS_SET_LSB_DEPTH_REQUEST,
        25i32,
        r_enc.set_lsb_depth(25),
        "set_lsb_depth(25)"
    );
    test_ctl!(
        bindings::OPUS_SET_LSB_DEPTH_REQUEST,
        16i32,
        r_enc.set_lsb_depth(16),
        "set_lsb_depth(16) [valid]"
    );

    // Prediction disabled
    test_ctl!(
        bindings::OPUS_SET_PREDICTION_DISABLED_REQUEST,
        -1i32,
        r_enc.set_prediction_disabled(-1),
        "set_prediction_disabled(-1)"
    );
    test_ctl!(
        bindings::OPUS_SET_PREDICTION_DISABLED_REQUEST,
        2i32,
        r_enc.set_prediction_disabled(2),
        "set_prediction_disabled(2)"
    );

    // Force channels
    test_ctl!(
        bindings::OPUS_SET_FORCE_CHANNELS_REQUEST,
        0i32,
        r_enc.set_force_channels(0),
        "set_force_channels(0)"
    );
    test_ctl!(
        bindings::OPUS_SET_FORCE_CHANNELS_REQUEST,
        3i32,
        r_enc.set_force_channels(3),
        "set_force_channels(3)"
    );

    unsafe {
        bindings::opus_encoder_destroy(c_enc);
    }

    println!();

    // -----------------------------------------------------------------------
    // 4. Decode with invalid inputs
    // -----------------------------------------------------------------------

    println!("--- Decode with invalid inputs ---");

    // First encode a valid packet with C reference so we have real data
    let c_enc_tmp = unsafe {
        let mut err: i32 = 0;
        let enc = bindings::opus_encoder_create(48000, 1, audio, &mut err);
        assert!(!enc.is_null() && err == 0);
        enc
    };

    let silence = vec![0i16; 960]; // 20ms at 48kHz mono
    let mut c_pkt = vec![0u8; 4000];
    let c_pkt_len = unsafe {
        bindings::opus_encode(c_enc_tmp, silence.as_ptr(), 960, c_pkt.as_mut_ptr(), 4000)
    };
    assert!(c_pkt_len > 0, "C encode of silence failed");
    let c_pkt_len = c_pkt_len as usize;
    unsafe {
        bindings::opus_encoder_destroy(c_enc_tmp);
    }

    let valid_pkt = &c_pkt[..c_pkt_len];

    // Helper: test decode with a byte slice (non-null data)
    fn test_decode_data(
        data: &[u8],
        c_data_len: i32,
        frame_size: i32,
        label: &str,
        pass: &mut u32,
        fail: &mut u32,
    ) {
        // C decoder -- fresh each time to avoid state contamination
        let c_dec = unsafe {
            let mut err: i32 = 0;
            let dec = bindings::opus_decoder_create(48000, 1, &mut err);
            assert!(!dec.is_null() && err == 0);
            dec
        };
        let mut c_pcm = vec![0i16; 5760];
        let c_ret = unsafe {
            bindings::opus_decode(
                c_dec,
                data.as_ptr(),
                c_data_len,
                c_pcm.as_mut_ptr(),
                frame_size,
                0,
            )
        };
        unsafe {
            bindings::opus_decoder_destroy(c_dec);
        }

        // Rust decoder -- fresh each time
        let mut r_dec =
            mdopus::opus::decoder::OpusDecoder::new(48000, 1).expect("Rust decoder failed");
        let mut r_pcm = vec![0i16; 5760];
        let r_data: Option<&[u8]> = if c_data_len > 0 {
            Some(&data[..c_data_len as usize])
        } else {
            Some(&[])
        };
        let r_ret = match r_dec.decode(r_data, &mut r_pcm, frame_size, false) {
            Ok(n) => n,
            Err(e) => e,
        };

        let ok = (c_ret >= 0) == (r_ret >= 0);
        if ok {
            println!("  PASS  {}: C={}, Rust={}", label, c_ret, r_ret);
            *pass += 1;
        } else {
            println!("  FAIL  {}: C={}, Rust={}", label, c_ret, r_ret);
            *fail += 1;
        }
    }

    // Helper: test decode with NULL/None data (PLC path)
    fn test_decode_null(frame_size: i32, label: &str, pass: &mut u32, fail: &mut u32) {
        let c_dec = unsafe {
            let mut err: i32 = 0;
            let dec = bindings::opus_decoder_create(48000, 1, &mut err);
            assert!(!dec.is_null() && err == 0);
            dec
        };
        let mut c_pcm = vec![0i16; 5760];
        let c_ret = unsafe {
            bindings::opus_decode(
                c_dec,
                std::ptr::null(),
                0,
                c_pcm.as_mut_ptr(),
                frame_size,
                0,
            )
        };
        unsafe {
            bindings::opus_decoder_destroy(c_dec);
        }

        let mut r_dec =
            mdopus::opus::decoder::OpusDecoder::new(48000, 1).expect("Rust decoder failed");
        let mut r_pcm = vec![0i16; 5760];
        let r_ret = match r_dec.decode(None, &mut r_pcm, frame_size, false) {
            Ok(n) => n,
            Err(e) => e,
        };

        let ok = (c_ret >= 0) == (r_ret >= 0);
        if ok {
            println!("  PASS  {}: C={}, Rust={}", label, c_ret, r_ret);
            *pass += 1;
        } else {
            println!("  FAIL  {}: C={}, Rust={}", label, c_ret, r_ret);
            *fail += 1;
        }
    }

    // Valid decode (baseline)
    test_decode_data(
        valid_pkt,
        c_pkt_len as i32,
        960,
        "decode(valid, 960)",
        &mut pass,
        &mut fail,
    );

    // Null/None data -- triggers PLC
    test_decode_null(960, "decode(NULL, 960) [PLC]", &mut pass, &mut fail);

    // Frame size 0
    test_decode_data(
        valid_pkt,
        c_pkt_len as i32,
        0,
        "decode(valid, fs=0)",
        &mut pass,
        &mut fail,
    );

    // Empty packet (0-length data pointer)
    test_decode_data(
        valid_pkt,
        0,
        960,
        "decode(data, len=0, 960)",
        &mut pass,
        &mut fail,
    );

    // Very small frame size
    test_decode_data(
        valid_pkt,
        c_pkt_len as i32,
        1,
        "decode(valid, fs=1)",
        &mut pass,
        &mut fail,
    );

    // Large frame size (5760 = 120ms at 48kHz -- max supported)
    test_decode_data(
        valid_pkt,
        c_pkt_len as i32,
        5760,
        "decode(valid, fs=5760)",
        &mut pass,
        &mut fail,
    );

    // Corrupted packet: single byte
    let bad_pkt: [u8; 1] = [0xFF];
    test_decode_data(
        &bad_pkt,
        1,
        960,
        "decode([0xFF], 960)",
        &mut pass,
        &mut fail,
    );

    // Null data with frame_size 0
    test_decode_null(0, "decode(NULL, fs=0)", &mut pass, &mut fail);

    // Note: negative length (C `len=-1`) cannot be tested through Rust's slice
    // API -- Rust prevents negative-length slices by construction. The C reference
    // returns OPUS_INVALID_PACKET for negative len; the Rust API prevents this
    // class of error at the type level. Skipped.

    println!();
    println!("========================================");
    println!("API validation: {} PASS, {} FAIL", pass, fail);
    println!("========================================");
    if fail > 0 {
        process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Packet introspection comparison (C vs Rust)
// ---------------------------------------------------------------------------

fn cmd_packets(wav_path: &str) {
    use mdopus::opus::decoder::{
        opus_packet_get_bandwidth, opus_packet_get_nb_channels, opus_packet_get_nb_frames,
        opus_packet_get_nb_samples, opus_packet_get_samples_per_frame,
    };

    println!("Packet introspection comparison: C vs Rust");
    println!();

    // Generate packets at several configs to cover different TOC bytes
    let configs: &[(&str, i32, i32, i32, i32)] = &[
        (
            "CELT 48k mono",
            48000,
            1,
            64000,
            bindings::OPUS_APPLICATION_AUDIO,
        ),
        (
            "VOIP 48k mono",
            48000,
            1,
            64000,
            bindings::OPUS_APPLICATION_VOIP,
        ),
        (
            "SILK 8k mono",
            8000,
            1,
            12000,
            bindings::OPUS_APPLICATION_VOIP,
        ),
        (
            "Hybrid 16k mono",
            16000,
            1,
            24000,
            bindings::OPUS_APPLICATION_AUDIO,
        ),
        (
            "Stereo 48k",
            48000,
            2,
            128000,
            bindings::OPUS_APPLICATION_AUDIO,
        ),
    ];

    // Also encode from the provided WAV for realistic packets
    let wav = read_wav(Path::new(wav_path));
    let wav_sr = wav.sample_rate as i32;
    let wav_ch = wav.channels as i32;

    let mut pass = 0u32;
    let mut fail = 0u32;

    for (label, sr, ch, br, app) in configs {
        println!("--- {} (sr={}, ch={}, br={}) ---", label, sr, ch, br);

        // Generate noise at the right rate/channels
        let pcm = generate_noise(*sr, *ch, 0.5, 42);
        let mut cfg = EncodeConfig::new(*sr, *ch);
        cfg.bitrate = *br;
        cfg.application = *app;
        let encoded = c_encode_cfg(&pcm, &cfg);
        let packets = parse_packets(&encoded);

        for (i, pkt) in packets.iter().take(5).enumerate() {
            if pkt.is_empty() {
                continue;
            }
            // C introspection
            let c_bw = unsafe { bindings::opus_packet_get_bandwidth(pkt.as_ptr()) };
            let c_ch = unsafe { bindings::opus_packet_get_nb_channels(pkt.as_ptr()) };
            let c_nf =
                unsafe { bindings::opus_packet_get_nb_frames(pkt.as_ptr(), pkt.len() as i32) };
            let c_spf = unsafe { bindings::opus_packet_get_samples_per_frame(pkt.as_ptr(), *sr) };
            let c_ns = unsafe {
                bindings::opus_packet_get_nb_samples(pkt.as_ptr(), pkt.len() as i32, *sr)
            };

            // Rust introspection
            let r_bw = opus_packet_get_bandwidth(pkt);
            let r_ch = opus_packet_get_nb_channels(pkt);
            let r_nf = opus_packet_get_nb_frames(pkt).unwrap_or(-999);
            let r_spf = opus_packet_get_samples_per_frame(pkt, *sr);
            let r_ns = opus_packet_get_nb_samples(pkt, *sr).unwrap_or(-999);

            // Compare each function
            let funcs: &[(&str, i32, i32)] = &[
                ("get_bandwidth", c_bw, r_bw),
                ("get_nb_channels", c_ch, r_ch),
                ("get_nb_frames", c_nf, r_nf),
                ("get_samples_per_frame", c_spf, r_spf),
                ("get_nb_samples", c_ns, r_ns),
            ];

            for (fname, c_val, r_val) in funcs {
                if c_val == r_val {
                    pass += 1;
                } else {
                    fail += 1;
                    println!(
                        "  pkt[{}] {}: FAIL (C={}, Rust={}, TOC=0x{:02x})",
                        i, fname, c_val, r_val, pkt[0]
                    );
                }
            }
        }
    }

    // Also test with WAV-derived packets
    println!("--- WAV-derived ({} Hz, {} ch) ---", wav_sr, wav_ch);
    let wav_cfg = EncodeConfig::new(wav_sr, wav_ch);
    let wav_encoded = c_encode_cfg(&wav.samples, &wav_cfg);
    let wav_packets = parse_packets(&wav_encoded);

    for (i, pkt) in wav_packets.iter().take(5).enumerate() {
        if pkt.is_empty() {
            continue;
        }
        let c_bw = unsafe { bindings::opus_packet_get_bandwidth(pkt.as_ptr()) };
        let c_ch = unsafe { bindings::opus_packet_get_nb_channels(pkt.as_ptr()) };
        let c_nf = unsafe { bindings::opus_packet_get_nb_frames(pkt.as_ptr(), pkt.len() as i32) };
        let c_spf = unsafe { bindings::opus_packet_get_samples_per_frame(pkt.as_ptr(), wav_sr) };
        let c_ns =
            unsafe { bindings::opus_packet_get_nb_samples(pkt.as_ptr(), pkt.len() as i32, wav_sr) };

        let r_bw = opus_packet_get_bandwidth(pkt);
        let r_ch = opus_packet_get_nb_channels(pkt);
        let r_nf = opus_packet_get_nb_frames(pkt).unwrap_or(-999);
        let r_spf = opus_packet_get_samples_per_frame(pkt, wav_sr);
        let r_ns = opus_packet_get_nb_samples(pkt, wav_sr).unwrap_or(-999);

        let funcs: &[(&str, i32, i32)] = &[
            ("get_bandwidth", c_bw, r_bw),
            ("get_nb_channels", c_ch, r_ch),
            ("get_nb_frames", c_nf, r_nf),
            ("get_samples_per_frame", c_spf, r_spf),
            ("get_nb_samples", c_ns, r_ns),
        ];

        for (fname, c_val, r_val) in funcs {
            if c_val == r_val {
                pass += 1;
            } else {
                fail += 1;
                println!(
                    "  pkt[{}] {}: FAIL (C={}, Rust={}, TOC=0x{:02x})",
                    i, fname, c_val, r_val, pkt[0]
                );
            }
        }
    }

    println!();
    println!("========================================");
    println!("packets: {} PASS, {} FAIL", pass, fail);
    println!("========================================");
    if fail > 0 {
        process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Decode format comparison (i16, f32, i32/24-bit)
// ---------------------------------------------------------------------------

fn compare_f32_samples(a: &[f32], b: &[f32]) -> (usize, usize, Option<usize>, f32) {
    let total = a.len().max(b.len());
    let mut matching = 0usize;
    let mut first_diff = None;
    let mut max_diff: f32 = 0.0;

    for i in 0..total {
        let va = a.get(i).copied().unwrap_or(0.0);
        let vb = b.get(i).copied().unwrap_or(0.0);
        let diff = (va - vb).abs();
        if diff == 0.0 {
            matching += 1;
        } else {
            if first_diff.is_none() {
                first_diff = Some(i);
            }
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }
    (total, matching, first_diff, max_diff)
}

fn compare_i32_samples(a: &[i32], b: &[i32]) -> (usize, usize, Option<usize>, i32) {
    let total = a.len().max(b.len());
    let mut matching = 0usize;
    let mut first_diff = None;
    let mut max_diff: i32 = 0;

    for i in 0..total {
        let va = a.get(i).copied().unwrap_or(0);
        let vb = b.get(i).copied().unwrap_or(0);
        let diff = (va - vb).abs();
        if diff == 0 {
            matching += 1;
        } else {
            if first_diff.is_none() {
                first_diff = Some(i);
            }
            max_diff = max_diff.max(diff);
        }
    }
    (total, matching, first_diff, max_diff)
}

fn cmd_decode_formats(wav_path: &str) {
    use mdopus::opus::decoder::OpusDecoder as RustDecoder;

    println!("Decode format comparison: C vs Rust (i16, f32, i32/24-bit)");
    println!();

    let wav = read_wav(Path::new(wav_path));
    let sr = wav.sample_rate as i32;
    let ch = wav.channels as i32;

    // Encode with C to get reference packets
    let cfg = EncodeConfig::new(sr, ch);
    let encoded = c_encode_cfg(&wav.samples, &cfg);
    let packets = parse_packets(&encoded);

    println!(
        "Input: {} ({} Hz, {} ch, {} packets)",
        wav_path,
        sr,
        ch,
        packets.len()
    );

    let frame_size = cfg.frame_size();
    let frame_samples = frame_size * ch as usize;

    let mut pass = 0u32;
    let mut fail = 0u32;

    // --- i16 decode comparison ---
    println!();
    println!("--- i16 decode ---");
    {
        let c_dec = unsafe {
            let mut error: i32 = 0;
            let dec = bindings::opus_decoder_create(sr, ch, &mut error);
            if dec.is_null() || error != bindings::OPUS_OK {
                eprintln!(
                    "ERROR: C opus_decoder_create failed: {}",
                    bindings::error_string(error)
                );
                process::exit(1);
            }
            dec
        };
        let mut r_dec = match RustDecoder::new(sr, ch) {
            Ok(d) => d,
            Err(code) => {
                eprintln!("ERROR: Rust OpusDecoder::new failed: {}", code);
                process::exit(1);
            }
        };

        let mut c_pcm = vec![0i16; frame_samples];
        let mut r_pcm = vec![0i16; frame_samples];
        let mut total_match = 0usize;
        let mut total_samples = 0usize;
        let mut first_fail_frame = None;

        for (i, pkt) in packets.iter().enumerate() {
            let c_ret = unsafe {
                bindings::opus_decode(
                    c_dec,
                    pkt.as_ptr(),
                    pkt.len() as i32,
                    c_pcm.as_mut_ptr(),
                    frame_size as i32,
                    0,
                )
            };
            let r_ret = r_dec
                .decode(Some(pkt), &mut r_pcm, frame_size as i32, false)
                .unwrap_or(-1);

            if c_ret > 0 && r_ret > 0 {
                let n = c_ret as usize * ch as usize;
                total_samples += n;
                let stats = compare_samples(&c_pcm[..n], &r_pcm[..n]);
                total_match += stats.matching;
                if stats.first_diff_offset.is_some() && first_fail_frame.is_none() {
                    first_fail_frame = Some(i);
                }
            }
        }

        if total_match == total_samples {
            println!("  i16: PASS ({} samples, all match)", total_samples);
            pass += 1;
        } else {
            let diff_count = total_samples - total_match;
            println!(
                "  i16: FAIL ({}/{} samples match, first diff at frame {:?})",
                total_match, total_samples, first_fail_frame
            );
            fail += 1;
            let _ = diff_count; // suppress unused warning
        }

        unsafe {
            bindings::opus_decoder_destroy(c_dec);
        }
    }

    // --- f32 decode comparison ---
    // C fixed-point build doesn't export opus_decode_float (DISABLE_FLOAT_API),
    // so we decode with C (i16) and convert to f32 the same way the C fixed-point
    // code does: sample * (1.0 / 32768.0). Then compare against Rust decode_float.
    println!();
    println!("--- f32 decode ---");
    {
        let c_dec = unsafe {
            let mut error: i32 = 0;
            let dec = bindings::opus_decoder_create(sr, ch, &mut error);
            if dec.is_null() || error != bindings::OPUS_OK {
                eprintln!(
                    "ERROR: C opus_decoder_create failed: {}",
                    bindings::error_string(error)
                );
                process::exit(1);
            }
            dec
        };
        let mut r_dec = match RustDecoder::new(sr, ch) {
            Ok(d) => d,
            Err(code) => {
                eprintln!("ERROR: Rust OpusDecoder::new failed: {}", code);
                process::exit(1);
            }
        };

        let mut c_pcm_16 = vec![0i16; frame_samples];
        let mut r_pcm_f = vec![0.0f32; frame_samples];
        let mut total_match = 0usize;
        let mut total_samples = 0usize;
        let mut first_fail_frame = None;
        let mut worst_diff: f32 = 0.0;

        for (i, pkt) in packets.iter().enumerate() {
            let c_ret = unsafe {
                bindings::opus_decode(
                    c_dec,
                    pkt.as_ptr(),
                    pkt.len() as i32,
                    c_pcm_16.as_mut_ptr(),
                    frame_size as i32,
                    0,
                )
            };
            let r_ret = r_dec
                .decode_float(Some(pkt), &mut r_pcm_f, frame_size as i32, false)
                .unwrap_or(-1);

            if c_ret > 0 && r_ret > 0 {
                let n = c_ret as usize * ch as usize;
                total_samples += n;
                // Convert C i16 to f32 the same way C fixed-point does
                let c_pcm_f: Vec<f32> = c_pcm_16[..n]
                    .iter()
                    .map(|&s| s as f32 * (1.0 / 32768.0))
                    .collect();
                let (_, m, fd, md) = compare_f32_samples(&c_pcm_f, &r_pcm_f[..n]);
                total_match += m;
                if md > worst_diff {
                    worst_diff = md;
                }
                if fd.is_some() && first_fail_frame.is_none() {
                    first_fail_frame = Some(i);
                }
            }
        }

        if total_match == total_samples {
            println!("  f32: PASS ({} samples, all match)", total_samples);
            pass += 1;
        } else {
            println!(
                "  f32: FAIL ({}/{} samples match, max_diff={:.9}, first diff at frame {:?})",
                total_match, total_samples, worst_diff, first_fail_frame
            );
            fail += 1;
        }

        unsafe {
            bindings::opus_decoder_destroy(c_dec);
        }
    }

    // --- i32/24-bit decode comparison ---
    // C fixed-point opus_decode in 24-bit mode: decode to i16 then <<8.
    // The C API doesn't have a direct opus_decode24() export, so we emulate
    // by calling C opus_decode (i16) and shifting left by 8, then comparing
    // against the Rust decode24.
    println!();
    println!("--- i32/24-bit decode ---");
    {
        let c_dec = unsafe {
            let mut error: i32 = 0;
            let dec = bindings::opus_decoder_create(sr, ch, &mut error);
            if dec.is_null() || error != bindings::OPUS_OK {
                eprintln!(
                    "ERROR: C opus_decoder_create failed: {}",
                    bindings::error_string(error)
                );
                process::exit(1);
            }
            dec
        };
        let mut r_dec = match RustDecoder::new(sr, ch) {
            Ok(d) => d,
            Err(code) => {
                eprintln!("ERROR: Rust OpusDecoder::new failed: {}", code);
                process::exit(1);
            }
        };

        let mut c_pcm_16 = vec![0i16; frame_samples];
        let mut r_pcm_32 = vec![0i32; frame_samples];
        let mut total_match = 0usize;
        let mut total_samples = 0usize;
        let mut first_fail_frame = None;
        let mut worst_diff: i32 = 0;

        for (i, pkt) in packets.iter().enumerate() {
            let c_ret = unsafe {
                bindings::opus_decode(
                    c_dec,
                    pkt.as_ptr(),
                    pkt.len() as i32,
                    c_pcm_16.as_mut_ptr(),
                    frame_size as i32,
                    0,
                )
            };
            let r_ret = r_dec
                .decode24(Some(pkt), &mut r_pcm_32, frame_size as i32, false)
                .unwrap_or(-1);

            if c_ret > 0 && r_ret > 0 {
                let n = c_ret as usize * ch as usize;
                total_samples += n;
                // C reference (fixed-point): i16 shifted left by 8
                let c_pcm_32: Vec<i32> = c_pcm_16[..n].iter().map(|&s| (s as i32) << 8).collect();
                let (_, m, fd, md) = compare_i32_samples(&c_pcm_32, &r_pcm_32[..n]);
                total_match += m;
                if md > worst_diff {
                    worst_diff = md;
                }
                if fd.is_some() && first_fail_frame.is_none() {
                    first_fail_frame = Some(i);
                }
            }
        }

        if total_match == total_samples {
            println!("  i32/24-bit: PASS ({} samples, all match)", total_samples);
            pass += 1;
        } else {
            println!(
                "  i32/24-bit: FAIL ({}/{} samples match, max_diff={}, first diff at frame {:?})",
                total_match, total_samples, worst_diff, first_fail_frame
            );
            fail += 1;
        }

        unsafe {
            bindings::opus_decoder_destroy(c_dec);
        }
    }

    println!();
    println!("========================================");
    println!("decode-formats: {} PASS, {} FAIL", pass, fail);
    println!("========================================");
    if fail > 0 {
        process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Longsoak: long-duration encode/decode to catch state accumulation bugs
// ---------------------------------------------------------------------------

fn cmd_longsoak(duration_secs: i32, sample_rate: i32) {
    let ch = 1; // mono for simplicity
    let pcm = generate_noise(sample_rate, ch, duration_secs as f64, 12345);
    let frame_size = (sample_rate / 50) as usize; // 20ms frames
    let frame_samples = frame_size * ch as usize;
    let num_frames = pcm.len() / frame_samples;

    println!(
        "Longsoak: {} Hz, {} ch, {}s ({} frames)",
        sample_rate, ch, duration_secs, num_frames
    );

    // Encode with C and Rust
    let mut cfg = EncodeConfig::new(sample_rate, ch);
    cfg.bitrate = 64000;
    let c_enc_data = c_encode_cfg(&pcm, &cfg);
    let rust_enc_data = rust_encode_cfg(&pcm, &cfg);

    // Compare encoded output
    let enc_stats = compare_bytes(&c_enc_data, &rust_enc_data);
    if enc_stats.first_diff_offset.is_none() {
        println!("  encode: PASS ({} bytes)", c_enc_data.len());
    } else {
        println!(
            "  encode: FAIL at byte {} (of {}, max_diff={})",
            enc_stats.first_diff_offset.unwrap(),
            enc_stats.total,
            enc_stats.max_diff
        );
    }

    // Frame-by-frame decode with final_range tracking
    let packets = parse_packets(&c_enc_data);
    println!(
        "  decode: {} packets, frame_size={}",
        packets.len(),
        frame_size
    );

    // Create C decoder
    let c_dec = unsafe {
        let mut error: i32 = 0;
        let dec = bindings::opus_decoder_create(sample_rate, ch, &mut error);
        if dec.is_null() || error != bindings::OPUS_OK {
            eprintln!(
                "ERROR: C opus_decoder_create failed: {}",
                bindings::error_string(error)
            );
            process::exit(1);
        }
        dec
    };

    // Create Rust decoder
    let mut rust_dec =
        mdopus::opus::decoder::OpusDecoder::new(sample_rate, ch).unwrap_or_else(|e| {
            eprintln!("ERROR: Rust OpusDecoder::new failed: {}", e);
            process::exit(1);
        });

    let mut c_pcm_buf = vec![0i16; frame_samples];
    let mut rust_pcm_buf = vec![0i16; frame_samples];
    let mut first_pcm_fail: Option<usize> = None;
    let mut first_range_fail: Option<usize> = None;
    let mut total_sample_diffs = 0u64;
    let mut max_sample_diff = 0i32;

    for (i, pkt) in packets.iter().enumerate() {
        // C decode
        let c_ret = unsafe {
            bindings::opus_decode(
                c_dec,
                pkt.as_ptr(),
                pkt.len() as i32,
                c_pcm_buf.as_mut_ptr(),
                frame_size as i32,
                0,
            )
        };
        if c_ret < 0 {
            eprintln!(
                "  frame {}: C decode error: {}",
                i,
                bindings::error_string(c_ret)
            );
            break;
        }

        // Rust decode
        let rust_ret = match rust_dec.decode(Some(pkt), &mut rust_pcm_buf, frame_size as i32, false)
        {
            Ok(r) => r,
            Err(e) => {
                eprintln!("  frame {}: Rust decode error: {}", i, e);
                break;
            }
        };

        // Get final range from both
        let mut c_range: u32 = 0;
        unsafe {
            bindings::opus_decoder_ctl(
                c_dec,
                bindings::OPUS_GET_FINAL_RANGE_REQUEST,
                &mut c_range as *mut u32,
            );
        }
        let rust_range = rust_dec.get_final_range();

        // Compare PCM
        let c_count = c_ret as usize * ch as usize;
        let r_count = rust_ret as usize * ch as usize;
        let len = c_count.min(r_count);
        for j in 0..len {
            let d = (c_pcm_buf[j] as i32 - rust_pcm_buf[j] as i32).abs();
            if d > 0 {
                total_sample_diffs += 1;
                max_sample_diff = max_sample_diff.max(d);
                if first_pcm_fail.is_none() {
                    first_pcm_fail = Some(i);
                    println!(
                        "  frame {:5}: PCM MISMATCH first_diff_sample={} max_diff={} c_samples={} rust_samples={}",
                        i, j, d, c_count, r_count
                    );
                }
            }
        }

        // Compare final_range
        if c_range != rust_range && first_range_fail.is_none() {
            first_range_fail = Some(i);
            println!(
                "  frame {:5}: RANGE MISMATCH C={:08x} Rust={:08x}",
                i, c_range, rust_range
            );
        }

        // Progress reporting every 250 frames
        if (i + 1) % 250 == 0 || i + 1 == packets.len() {
            print!(
                "\r  progress: {}/{} frames ({:.0}%)",
                i + 1,
                packets.len(),
                (i + 1) as f64 / packets.len() as f64 * 100.0
            );
        }
    }
    println!();

    unsafe {
        bindings::opus_decoder_destroy(c_dec);
    }

    // Summary
    println!();
    println!("  --- Longsoak summary ---");
    println!("  frames decoded: {}", packets.len());
    match first_pcm_fail {
        None => println!("  PCM:   PASS (all samples match)"),
        Some(f) => println!(
            "  PCM:   FAIL (first at frame {}, {} total diffs, max_diff={})",
            f, total_sample_diffs, max_sample_diff
        ),
    }
    match first_range_fail {
        None => println!("  Range: PASS (all final_range values match)"),
        Some(f) => println!("  Range: FAIL (first mismatch at frame {})", f),
    }
    if first_pcm_fail.is_none() && first_range_fail.is_none() {
        println!("  OVERALL: PASS");
    } else {
        println!("  OVERALL: FAIL");
        process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Torture: extended soak with random parameter variation mid-stream
// ---------------------------------------------------------------------------

/// Get current process RSS in bytes (for memory leak detection).
#[allow(unused_variables, unreachable_code)]
fn get_rss_bytes() -> Option<usize> {
    #[cfg(target_os = "windows")]
    {
        #[repr(C)]
        #[allow(non_snake_case)]
        struct ProcessMemoryCounters {
            cb: u32,
            page_fault_count: u32,
            peak_working_set_size: usize,
            working_set_size: usize,
            quota_peak_paged_pool_usage: usize,
            quota_paged_pool_usage: usize,
            quota_peak_non_paged_pool_usage: usize,
            quota_non_paged_pool_usage: usize,
            pagefile_usage: usize,
            peak_pagefile_usage: usize,
        }
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut core::ffi::c_void;
            fn K32GetProcessMemoryInfo(
                h_process: *mut core::ffi::c_void,
                ppsmem_counters: *mut ProcessMemoryCounters,
                cb: u32,
            ) -> i32;
        }
        unsafe {
            let mut pmc = std::mem::zeroed::<ProcessMemoryCounters>();
            pmc.cb = std::mem::size_of::<ProcessMemoryCounters>() as u32;
            if K32GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) != 0 {
                return Some(pmc.working_set_size);
            }
        }
        None
    }
    #[cfg(not(target_os = "windows"))]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("VmRSS:") {
                    let trimmed = rest.trim().trim_end_matches(" kB").trim();
                    if let Ok(kb) = trimmed.parse::<usize>() {
                        return Some(kb * 1024);
                    }
                }
            }
        }
        None
    }
}

/// Advance PRNG state (LCG, same constants as generate_noise).
fn torture_rng(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

/// Pick a random element from a slice.
fn torture_rng_choice<T: Copy>(state: &mut u64, choices: &[T]) -> T {
    choices[(torture_rng(state) % choices.len() as u64) as usize]
}

/// Generate a random encoder config for torture testing.
/// Only varies parameters settable mid-stream via CTL (not sample_rate, channels, application).
fn random_torture_config(rng: &mut u64, sample_rate: i32, channels: i32) -> EncodeConfig {
    let mut cfg = EncodeConfig::new(sample_rate, channels);

    cfg.bitrate = torture_rng_choice(
        rng,
        &[
            6000, 8000, 10000, 16000, 24000, 32000, 48000, 64000, 96000, 128000, 256000, 320000,
            510000,
        ],
    );
    cfg.complexity = (torture_rng(rng) % 11) as i32;
    cfg.vbr = (torture_rng(rng) % 2) as i32;
    cfg.vbr_constraint = if cfg.vbr != 0 {
        (torture_rng(rng) % 2) as i32
    } else {
        0
    };
    cfg.signal = torture_rng_choice(
        rng,
        &[
            bindings::OPUS_AUTO,
            bindings::OPUS_SIGNAL_VOICE,
            bindings::OPUS_SIGNAL_MUSIC,
        ],
    );
    cfg.max_bandwidth = torture_rng_choice(
        rng,
        &[
            bindings::OPUS_BANDWIDTH_NARROWBAND,
            bindings::OPUS_BANDWIDTH_MEDIUMBAND,
            bindings::OPUS_BANDWIDTH_WIDEBAND,
            bindings::OPUS_BANDWIDTH_SUPERWIDEBAND,
            bindings::OPUS_BANDWIDTH_FULLBAND,
        ],
    );
    cfg.bandwidth = bindings::OPUS_AUTO;
    // Weight toward AUTO to let the encoder decide, but occasionally force a mode
    cfg.force_mode = torture_rng_choice(
        rng,
        &[
            bindings::OPUS_AUTO,
            bindings::OPUS_AUTO,
            bindings::OPUS_AUTO,
            1000, // SILK_ONLY
            1002, // CELT_ONLY
        ],
    );
    cfg.fec = (torture_rng(rng) % 2) as i32;
    cfg.packet_loss_pct = if cfg.fec != 0 {
        torture_rng_choice(rng, &[0, 1, 5, 10, 20, 30])
    } else {
        0
    };
    cfg.dtx = (torture_rng(rng) % 2) as i32;
    cfg.force_channels = if channels > 1 {
        torture_rng_choice(rng, &[bindings::OPUS_AUTO, 1, 2])
    } else {
        bindings::OPUS_AUTO
    };
    cfg.lsb_depth = torture_rng_choice(rng, &[8, 16, 24]);
    cfg.prediction_disabled = (torture_rng(rng) % 2) as i32;
    cfg.phase_inversion_disabled = (torture_rng(rng) % 2) as i32;
    // Frame sizes safe for all modes: 10ms, 20ms, 40ms, 60ms. Weight toward 20ms.
    cfg.frame_ms = torture_rng_choice(rng, &[10.0, 20.0, 20.0, 20.0, 40.0, 60.0]);

    cfg
}

/// Apply an EncodeConfig to a live C encoder via CTL calls.
unsafe fn apply_config_to_c_encoder(enc: *mut bindings::OpusEncoder, cfg: &EncodeConfig) {
    unsafe {
        bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_BITRATE_REQUEST, cfg.bitrate);
        bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_COMPLEXITY_REQUEST, cfg.complexity);
        bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_VBR_REQUEST, cfg.vbr);
        bindings::opus_encoder_ctl(
            enc,
            bindings::OPUS_SET_VBR_CONSTRAINT_REQUEST,
            cfg.vbr_constraint,
        );
        bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_INBAND_FEC_REQUEST, cfg.fec);
        bindings::opus_encoder_ctl(
            enc,
            bindings::OPUS_SET_PACKET_LOSS_PERC_REQUEST,
            cfg.packet_loss_pct,
        );
        bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_DTX_REQUEST, cfg.dtx);
        bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_SIGNAL_REQUEST, cfg.signal);
        bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_BANDWIDTH_REQUEST, cfg.bandwidth);
        bindings::opus_encoder_ctl(
            enc,
            bindings::OPUS_SET_FORCE_CHANNELS_REQUEST,
            cfg.force_channels,
        );
        bindings::opus_encoder_ctl(
            enc,
            bindings::OPUS_SET_MAX_BANDWIDTH_REQUEST,
            cfg.max_bandwidth,
        );
        bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_LSB_DEPTH_REQUEST, cfg.lsb_depth);
        bindings::opus_encoder_ctl(
            enc,
            bindings::OPUS_SET_PREDICTION_DISABLED_REQUEST,
            cfg.prediction_disabled,
        );
        bindings::opus_encoder_ctl(
            enc,
            bindings::OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST,
            cfg.phase_inversion_disabled,
        );
        bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_FORCE_MODE_REQUEST, cfg.force_mode);
    }
}

/// Apply an EncodeConfig to a live Rust encoder.
fn apply_config_to_rust_encoder(enc: &mut mdopus::opus::encoder::OpusEncoder, cfg: &EncodeConfig) {
    enc.set_bitrate(cfg.bitrate);
    enc.set_complexity(cfg.complexity);
    enc.set_vbr(cfg.vbr);
    enc.set_vbr_constraint(cfg.vbr_constraint);
    enc.set_inband_fec(cfg.fec);
    enc.set_packet_loss_perc(cfg.packet_loss_pct);
    enc.set_dtx(cfg.dtx);
    enc.set_signal(cfg.signal);
    enc.set_bandwidth(cfg.bandwidth);
    enc.set_force_channels(cfg.force_channels);
    enc.set_max_bandwidth(cfg.max_bandwidth);
    enc.set_lsb_depth(cfg.lsb_depth);
    enc.set_prediction_disabled(cfg.prediction_disabled);
    enc.set_phase_inversion_disabled(cfg.phase_inversion_disabled);
    enc.set_force_mode(cfg.force_mode);
}

/// Format an EncodeConfig as a compact diagnostic string.
fn format_torture_config(cfg: &EncodeConfig) -> String {
    let mode_str = match cfg.force_mode {
        -1000 => "AUTO",
        1000 => "SILK",
        1001 => "HYBRID",
        1002 => "CELT",
        _ => "??",
    };
    let bw_str = match cfg.max_bandwidth {
        1101 => "NB",
        1102 => "MB",
        1103 => "WB",
        1104 => "SWB",
        1105 => "FB",
        _ => "AUTO",
    };
    let sig_str = match cfg.signal {
        -1000 => "AUTO",
        3001 => "VOICE",
        3002 => "MUSIC",
        _ => "??",
    };
    format!(
        "br={} mode={} bw={} cx={} vbr={} fec={}/{} dtx={} sig={} frame={}ms",
        cfg.bitrate,
        mode_str,
        bw_str,
        cfg.complexity,
        cfg.vbr,
        cfg.fec,
        cfg.packet_loss_pct,
        cfg.dtx,
        sig_str,
        cfg.frame_ms
    )
}

// ---------------------------------------------------------------------------
// Section 6: Periodic state comparison (HLD 2026.04.08)
// ---------------------------------------------------------------------------
// Every N frames during a torture run we sample the C and Rust encoder /
// decoder state via OPUS_GET_* CTLs and compare. A drifting internal counter
// that has not yet produced a divergent packet is caught within one polling
// window. Strict by default per Decision 5 — the project is bit-exact, so
// "state differs but packets still match" is a latent bug, not noise.
//
// OPUS_GET_IN_DTX is deliberately absent: the Rust encoder does not expose a
// `get_in_dtx` getter and the HLD forbids `src/` changes. All other dynamic
// encoder/decoder state we can read is covered.

/// Read one i32 from a C encoder CTL. Returns 0 on error (CTL should not
/// fail with a GET request — if it does, comparison will flag it).
unsafe fn c_enc_get_i32(enc: *mut bindings::OpusEncoder, req: i32) -> i32 {
    let mut val: i32 = 0;
    unsafe { bindings::opus_encoder_ctl(enc, req, &mut val as *mut i32) };
    val
}

/// Read one u32 from a C encoder CTL.
unsafe fn c_enc_get_u32(enc: *mut bindings::OpusEncoder, req: i32) -> u32 {
    let mut val: u32 = 0;
    unsafe { bindings::opus_encoder_ctl(enc, req, &mut val as *mut u32) };
    val
}

/// Read one i32 from a C decoder CTL.
unsafe fn c_dec_get_i32(dec: *mut bindings::OpusDecoder, req: i32) -> i32 {
    let mut val: i32 = 0;
    unsafe { bindings::opus_decoder_ctl(dec, req, &mut val as *mut i32) };
    val
}

/// Read one u32 from a C decoder CTL.
unsafe fn c_dec_get_u32(dec: *mut bindings::OpusDecoder, req: i32) -> u32 {
    let mut val: u32 = 0;
    unsafe { bindings::opus_decoder_ctl(dec, req, &mut val as *mut u32) };
    val
}

/// Compare every observable state getter on the C and Rust encoder/decoder
/// pairs. Returns an empty vec if everything matches; otherwise a list of
/// human-readable diff lines.
fn state_checkpoint_diffs(
    c_enc: *mut bindings::OpusEncoder,
    r_enc: &mdopus::opus::encoder::OpusEncoder,
    c_dec: *mut bindings::OpusDecoder,
    r_dec: &mdopus::opus::decoder::OpusDecoder,
) -> Vec<String> {
    let mut diffs: Vec<String> = Vec::new();

    // ---- Encoder state ----
    unsafe {
        macro_rules! cmp_enc_i32 {
            ($req:expr, $rust:expr, $name:literal) => {{
                let c_val = c_enc_get_i32(c_enc, $req);
                let r_val: i32 = $rust;
                if c_val != r_val {
                    diffs.push(format!("enc.{}: C={} R={}", $name, c_val, r_val));
                }
            }};
        }
        cmp_enc_i32!(
            bindings::OPUS_GET_BITRATE_REQUEST,
            r_enc.get_bitrate(),
            "bitrate"
        );
        cmp_enc_i32!(
            bindings::OPUS_GET_MAX_BANDWIDTH_REQUEST,
            r_enc.get_max_bandwidth(),
            "max_bandwidth"
        );
        cmp_enc_i32!(bindings::OPUS_GET_VBR_REQUEST, r_enc.get_vbr(), "vbr");
        cmp_enc_i32!(
            bindings::OPUS_GET_BANDWIDTH_REQUEST,
            r_enc.get_bandwidth(),
            "bandwidth"
        );
        cmp_enc_i32!(
            bindings::OPUS_GET_COMPLEXITY_REQUEST,
            r_enc.get_complexity(),
            "complexity"
        );
        cmp_enc_i32!(
            bindings::OPUS_GET_INBAND_FEC_REQUEST,
            r_enc.get_inband_fec(),
            "inband_fec"
        );
        cmp_enc_i32!(
            bindings::OPUS_GET_PACKET_LOSS_PERC_REQUEST,
            r_enc.get_packet_loss_perc(),
            "packet_loss_perc"
        );
        cmp_enc_i32!(bindings::OPUS_GET_DTX_REQUEST, r_enc.get_dtx(), "dtx");
        cmp_enc_i32!(
            bindings::OPUS_GET_VBR_CONSTRAINT_REQUEST,
            r_enc.get_vbr_constraint(),
            "vbr_constraint"
        );
        cmp_enc_i32!(
            bindings::OPUS_GET_FORCE_CHANNELS_REQUEST,
            r_enc.get_force_channels(),
            "force_channels"
        );
        cmp_enc_i32!(
            bindings::OPUS_GET_SIGNAL_REQUEST,
            r_enc.get_signal(),
            "signal"
        );
        cmp_enc_i32!(
            bindings::OPUS_GET_LOOKAHEAD_REQUEST,
            r_enc.get_lookahead(),
            "lookahead"
        );
        cmp_enc_i32!(
            bindings::OPUS_GET_SAMPLE_RATE_REQUEST,
            r_enc.get_sample_rate(),
            "sample_rate"
        );
        cmp_enc_i32!(
            bindings::OPUS_GET_LSB_DEPTH_REQUEST,
            r_enc.get_lsb_depth(),
            "lsb_depth"
        );
        cmp_enc_i32!(
            bindings::OPUS_GET_PREDICTION_DISABLED_REQUEST,
            r_enc.get_prediction_disabled(),
            "prediction_disabled"
        );
        cmp_enc_i32!(
            bindings::OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST,
            r_enc.get_phase_inversion_disabled(),
            "phase_inversion_disabled"
        );

        // Final range is u32.
        let c_rng = c_enc_get_u32(c_enc, bindings::OPUS_GET_FINAL_RANGE_REQUEST);
        let r_rng = r_enc.get_final_range();
        if c_rng != r_rng {
            diffs.push(format!(
                "enc.final_range: C=0x{:08x} R=0x{:08x}",
                c_rng, r_rng
            ));
        }

        // Internal HP/mode state via debug helper
        let mut c_hp = [0i32; 4];
        let mut c_hp_smth2 = 0i32;
        let mut c_mode = 0i32;
        let mut c_stream_ch = 0i32;
        let mut c_bw = 0i32;
        bindings::debug_get_encoder_hp_state(
            c_enc,
            c_hp.as_mut_ptr(),
            &mut c_hp_smth2,
            &mut c_mode,
            &mut c_stream_ch,
            &mut c_bw,
        );
        let r_hp = r_enc.get_hp_mem();
        let r_hp_smth2 = r_enc.get_variable_hp_smth2();
        if c_hp != r_hp {
            diffs.push(format!("enc.hp_mem: C={:?} R={:?}", c_hp, r_hp));
        }
        if c_hp_smth2 != r_hp_smth2 {
            diffs.push(format!(
                "enc.variable_hp_smth2: C={} R={}",
                c_hp_smth2, r_hp_smth2
            ));
        }

        // SILK encoder internal state
        let mut c_silk_fs_khz = 0i32;
        let mut c_silk_frame_length = 0i32;
        let mut c_silk_nb_subfr = 0i32;
        let mut c_silk_input_buf_ix = 0i32;
        let mut c_silk_n_frames_per_packet = 0i32;
        let mut c_silk_packet_size_ms = 0i32;
        let mut c_silk_first_frame_after_reset = 0i32;
        let mut c_silk_controlled_since_last_payload = 0i32;
        let mut c_silk_prefill_flag = 0i32;
        let mut c_silk_n_frames_encoded = 0i32;
        let mut c_silk_speech_activity_q8 = 0i32;
        let mut c_silk_signal_type = 0i32;
        let mut c_silk_input_quality_bands_q15 = 0i32;
        bindings::debug_get_silk_state(
            c_enc,
            &mut c_silk_fs_khz,
            &mut c_silk_frame_length,
            &mut c_silk_nb_subfr,
            &mut c_silk_input_buf_ix,
            &mut c_silk_n_frames_per_packet,
            &mut c_silk_packet_size_ms,
            &mut c_silk_first_frame_after_reset,
            &mut c_silk_controlled_since_last_payload,
            &mut c_silk_prefill_flag,
            &mut c_silk_n_frames_encoded,
            &mut c_silk_speech_activity_q8,
            &mut c_silk_signal_type,
            &mut c_silk_input_quality_bands_q15,
        );
        if let Some(rs) = r_enc.get_silk_state() {
            macro_rules! cmp_silk {
                ($c:expr, $r:expr, $name:literal) => {
                    if $c != $r {
                        diffs.push(format!("SILK-{}: C={} R={}", $name, $c, $r));
                    }
                };
            }
            cmp_silk!(c_silk_fs_khz, rs.fs_khz, "fs_khz");
            cmp_silk!(c_silk_frame_length, rs.frame_length, "frame_length");
            cmp_silk!(c_silk_nb_subfr, rs.nb_subfr, "nb_subfr");
            cmp_silk!(c_silk_input_buf_ix, rs.input_buf_ix, "input_buf_ix");
            cmp_silk!(
                c_silk_n_frames_per_packet,
                rs.n_frames_per_packet,
                "n_frames_per_packet"
            );
            cmp_silk!(c_silk_packet_size_ms, rs.packet_size_ms, "packet_size_ms");
            cmp_silk!(
                c_silk_first_frame_after_reset,
                rs.first_frame_after_reset,
                "first_frame_after_reset"
            );
            cmp_silk!(
                c_silk_controlled_since_last_payload,
                rs.controlled_since_last_payload,
                "controlled_since_last_payload"
            );
            cmp_silk!(c_silk_prefill_flag, rs.prefill_flag, "prefill_flag");
            cmp_silk!(
                c_silk_n_frames_encoded,
                rs.n_frames_encoded,
                "n_frames_encoded"
            );
            cmp_silk!(
                c_silk_speech_activity_q8,
                rs.speech_activity_q8,
                "speech_activity_q8"
            );
            cmp_silk!(c_silk_signal_type, rs.signal_type, "signal_type");
            cmp_silk!(
                c_silk_input_quality_bands_q15,
                rs.input_quality_bands_q15,
                "input_quality_bands_q15"
            );
        }
    }

    // ---- Decoder state ----
    unsafe {
        macro_rules! cmp_dec_i32 {
            ($req:expr, $rust:expr, $name:literal) => {{
                let c_val = c_dec_get_i32(c_dec, $req);
                let r_val: i32 = $rust;
                if c_val != r_val {
                    diffs.push(format!("dec.{}: C={} R={}", $name, c_val, r_val));
                }
            }};
        }
        cmp_dec_i32!(
            bindings::OPUS_GET_BANDWIDTH_REQUEST,
            r_dec.get_bandwidth(),
            "bandwidth"
        );
        cmp_dec_i32!(
            bindings::OPUS_GET_SAMPLE_RATE_REQUEST,
            r_dec.get_sample_rate(),
            "sample_rate"
        );
        cmp_dec_i32!(bindings::OPUS_GET_PITCH_REQUEST, r_dec.get_pitch(), "pitch");
        cmp_dec_i32!(bindings::OPUS_GET_GAIN_REQUEST, r_dec.get_gain(), "gain");
        cmp_dec_i32!(
            bindings::OPUS_GET_LAST_PACKET_DURATION_REQUEST,
            r_dec.get_last_packet_duration(),
            "last_packet_duration"
        );

        // phase_inversion_disabled is bool on Rust decoder side.
        let c_piv = c_dec_get_i32(c_dec, bindings::OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST);
        let r_piv = r_dec.get_phase_inversion_disabled() as i32;
        if c_piv != r_piv {
            diffs.push(format!(
                "dec.phase_inversion_disabled: C={} R={}",
                c_piv, r_piv
            ));
        }

        let c_rng = c_dec_get_u32(c_dec, bindings::OPUS_GET_FINAL_RANGE_REQUEST);
        let r_rng = r_dec.get_final_range();
        if c_rng != r_rng {
            diffs.push(format!(
                "dec.final_range: C=0x{:08x} R=0x{:08x}",
                c_rng, r_rng
            ));
        }
    }

    diffs
}

// ---------------------------------------------------------------------------
// Section 3: Deterministic transition bursts (HLD 2026.04.08)
// ---------------------------------------------------------------------------
// Every burst_interval frames the random walk is suspended and a small
// scripted sequence is injected. Bursts target the redundancy / prefill /
// stereo→mono transition branches in src/opus/encoder.rs:1452-1505 that a
// uniform random walk rarely hits by chance. Selection is deterministic
// round-robin (Decision 8 rationale: reproducibility > diversity).

/// A single step in a burst: the config mutation, plus how many frames to
/// hold it. The mutation is applied on top of a neutral baseline so burst
/// steps do not accidentally inherit state from the random walk.
#[derive(Clone, Debug)]
struct BurstStep {
    /// Description of the mutation, for diagnostics.
    label: &'static str,
    /// Number of frames to hold this step.
    hold: usize,
    /// Function that mutates a neutral baseline config in place.
    mutate: fn(&mut EncodeConfig),
}

/// A single named burst: 3 consecutive steps.
#[derive(Clone, Debug)]
struct Burst {
    name: &'static str,
    steps: [BurstStep; 3],
    /// Minimum channel count a burst requires (1 = any, 2 = stereo-only).
    min_channels: i32,
}

/// Build a neutral baseline config that burst steps mutate. Keeps settings
/// predictable so each burst exercises its targeted branch rather than
/// whatever quirks the previous random config left on the stack.
fn burst_neutral_baseline(sample_rate: i32, channels: i32) -> EncodeConfig {
    let mut cfg = EncodeConfig::new(sample_rate, channels);
    cfg.bitrate = 64000;
    cfg.complexity = 10;
    cfg.vbr = 1;
    cfg.vbr_constraint = 0;
    cfg.signal = bindings::OPUS_AUTO;
    cfg.bandwidth = bindings::OPUS_AUTO;
    cfg.max_bandwidth = bindings::OPUS_BANDWIDTH_FULLBAND;
    cfg.force_channels = bindings::OPUS_AUTO;
    cfg.force_mode = bindings::OPUS_AUTO;
    cfg.fec = 0;
    cfg.packet_loss_pct = 0;
    cfg.dtx = 0;
    cfg.lsb_depth = 24;
    cfg.prediction_disabled = 0;
    cfg.phase_inversion_disabled = 0;
    cfg.frame_ms = 20.0;
    cfg
}

// Mode constants used by force_mode: values are the same as the C MODE_*
// constants in celt/modes.h — 1000=SILK_ONLY, 1001=HYBRID, 1002=CELT_ONLY.
const BURST_MODE_SILK: i32 = 1000;
const BURST_MODE_HYBRID: i32 = 1001;
const BURST_MODE_CELT: i32 = 1002;

/// The burst table. Index is deterministic (frame_idx/burst_interval) mod
/// len, so a seed-and-bug report reproduces exactly. Appending new bursts
/// only changes the schedule on future runs. Order must remain stable.
static BURSTS: &[Burst] = &[
    Burst {
        name: "silk_celt_silk",
        min_channels: 1,
        steps: [
            BurstStep {
                label: "SILK 20ms",
                hold: 1,
                mutate: |c| {
                    c.force_mode = BURST_MODE_SILK;
                    c.frame_ms = 20.0;
                    c.max_bandwidth = bindings::OPUS_BANDWIDTH_WIDEBAND;
                    c.bitrate = 32000;
                },
            },
            BurstStep {
                label: "CELT 20ms",
                hold: 1,
                mutate: |c| {
                    c.force_mode = BURST_MODE_CELT;
                    c.frame_ms = 20.0;
                    c.bitrate = 128000;
                },
            },
            BurstStep {
                label: "SILK 20ms",
                hold: 1,
                mutate: |c| {
                    c.force_mode = BURST_MODE_SILK;
                    c.frame_ms = 20.0;
                    c.max_bandwidth = bindings::OPUS_BANDWIDTH_WIDEBAND;
                    c.bitrate = 32000;
                },
            },
        ],
    },
    Burst {
        name: "celt_silk_celt",
        min_channels: 1,
        steps: [
            BurstStep {
                label: "CELT 20ms",
                hold: 1,
                mutate: |c| {
                    c.force_mode = BURST_MODE_CELT;
                    c.frame_ms = 20.0;
                    c.bitrate = 128000;
                },
            },
            BurstStep {
                label: "SILK 20ms",
                hold: 1,
                mutate: |c| {
                    c.force_mode = BURST_MODE_SILK;
                    c.frame_ms = 20.0;
                    c.max_bandwidth = bindings::OPUS_BANDWIDTH_WIDEBAND;
                    c.bitrate = 32000;
                },
            },
            BurstStep {
                label: "CELT 20ms",
                hold: 1,
                mutate: |c| {
                    c.force_mode = BURST_MODE_CELT;
                    c.frame_ms = 20.0;
                    c.bitrate = 128000;
                },
            },
        ],
    },
    Burst {
        name: "silk_celt_silk_short",
        min_channels: 1,
        steps: [
            BurstStep {
                label: "SILK 10ms",
                hold: 1,
                mutate: |c| {
                    c.force_mode = BURST_MODE_SILK;
                    c.frame_ms = 10.0;
                    c.max_bandwidth = bindings::OPUS_BANDWIDTH_WIDEBAND;
                    c.bitrate = 32000;
                },
            },
            BurstStep {
                label: "CELT 10ms",
                hold: 1,
                mutate: |c| {
                    c.force_mode = BURST_MODE_CELT;
                    c.frame_ms = 10.0;
                    c.bitrate = 128000;
                },
            },
            BurstStep {
                label: "SILK 10ms",
                hold: 1,
                mutate: |c| {
                    c.force_mode = BURST_MODE_SILK;
                    c.frame_ms = 10.0;
                    c.max_bandwidth = bindings::OPUS_BANDWIDTH_WIDEBAND;
                    c.bitrate = 32000;
                },
            },
        ],
    },
    Burst {
        name: "hybrid_celt_hybrid",
        min_channels: 1,
        steps: [
            BurstStep {
                label: "HYBRID 20ms",
                hold: 1,
                mutate: |c| {
                    c.force_mode = BURST_MODE_HYBRID;
                    c.frame_ms = 20.0;
                    c.max_bandwidth = bindings::OPUS_BANDWIDTH_SUPERWIDEBAND;
                    c.bitrate = 48000;
                },
            },
            BurstStep {
                label: "CELT 20ms",
                hold: 1,
                mutate: |c| {
                    c.force_mode = BURST_MODE_CELT;
                    c.frame_ms = 20.0;
                    c.bitrate = 128000;
                },
            },
            BurstStep {
                label: "HYBRID 20ms",
                hold: 1,
                mutate: |c| {
                    c.force_mode = BURST_MODE_HYBRID;
                    c.frame_ms = 20.0;
                    c.max_bandwidth = bindings::OPUS_BANDWIDTH_SUPERWIDEBAND;
                    c.bitrate = 48000;
                },
            },
        ],
    },
    Burst {
        name: "stereo_mono_stereo",
        min_channels: 2,
        steps: [
            BurstStep {
                label: "ch=2",
                hold: 1,
                mutate: |c| {
                    c.force_channels = 2;
                    c.bitrate = 96000;
                },
            },
            BurstStep {
                label: "ch=1",
                hold: 1,
                mutate: |c| {
                    c.force_channels = 1;
                    c.bitrate = 96000;
                },
            },
            BurstStep {
                label: "ch=2",
                hold: 1,
                mutate: |c| {
                    c.force_channels = 2;
                    c.bitrate = 96000;
                },
            },
        ],
    },
    Burst {
        name: "stereo_mono_auto",
        min_channels: 2,
        steps: [
            BurstStep {
                label: "ch=2",
                hold: 1,
                mutate: |c| {
                    c.force_channels = 2;
                    c.bitrate = 96000;
                },
            },
            BurstStep {
                label: "ch=1",
                hold: 1,
                mutate: |c| {
                    c.force_channels = 1;
                    c.bitrate = 96000;
                },
            },
            BurstStep {
                label: "ch=AUTO",
                hold: 1,
                mutate: |c| {
                    c.force_channels = bindings::OPUS_AUTO;
                    c.bitrate = 96000;
                },
            },
        ],
    },
    Burst {
        name: "vbr_cbr_at_low_br",
        min_channels: 1,
        steps: [
            BurstStep {
                label: "vbr=1 br=8k",
                hold: 1,
                mutate: |c| {
                    c.vbr = 1;
                    c.vbr_constraint = 0;
                    c.bitrate = 8000;
                },
            },
            BurstStep {
                label: "vbr=0 br=8k",
                hold: 1,
                mutate: |c| {
                    c.vbr = 0;
                    c.bitrate = 8000;
                },
            },
            BurstStep {
                label: "vbr=1 br=8k",
                hold: 1,
                mutate: |c| {
                    c.vbr = 1;
                    c.vbr_constraint = 0;
                    c.bitrate = 8000;
                },
            },
        ],
    },
    Burst {
        name: "celt_framesize_2p5_10_20",
        min_channels: 1,
        // Exercise CELT's short-block path (2.5 ms → 10 ms → 20 ms).
        // CELT's valid frame sizes are 2.5, 5, 10, 20 ms. 60 ms is
        // SILK-only so mixing it with force_mode=CELT would produce
        // rejected frames and waste the burst slot. Sticking to
        // CELT-valid sizes tests the MDCT length transitions that
        // the devil's-advocate review (2026.04.09) called out as the
        // real coverage gap.
        steps: [
            BurstStep {
                label: "CELT 2.5ms",
                hold: 1,
                mutate: |c| {
                    c.force_mode = BURST_MODE_CELT;
                    c.frame_ms = 2.5;
                    c.bitrate = 128000;
                },
            },
            BurstStep {
                label: "CELT 10ms",
                hold: 1,
                mutate: |c| {
                    c.force_mode = BURST_MODE_CELT;
                    c.frame_ms = 10.0;
                    c.bitrate = 128000;
                },
            },
            BurstStep {
                label: "CELT 20ms",
                hold: 1,
                mutate: |c| {
                    c.force_mode = BURST_MODE_CELT;
                    c.frame_ms = 20.0;
                    c.bitrate = 128000;
                },
            },
        ],
    },
    Burst {
        name: "silk_framesize_60_40_20",
        min_channels: 1,
        // Companion SILK-side burst: test long-block frame-size
        // transitions within SILK mode. force_mode=SILK keeps us on
        // the SILK path; 60, 40, 20 ms are all valid SILK sizes.
        steps: [
            BurstStep {
                label: "SILK 60ms",
                hold: 1,
                mutate: |c| {
                    c.force_mode = BURST_MODE_SILK;
                    c.frame_ms = 60.0;
                    c.max_bandwidth = bindings::OPUS_BANDWIDTH_WIDEBAND;
                    c.bitrate = 24000;
                },
            },
            BurstStep {
                label: "SILK 40ms",
                hold: 1,
                mutate: |c| {
                    c.force_mode = BURST_MODE_SILK;
                    c.frame_ms = 40.0;
                    c.max_bandwidth = bindings::OPUS_BANDWIDTH_WIDEBAND;
                    c.bitrate = 24000;
                },
            },
            BurstStep {
                label: "SILK 20ms",
                hold: 1,
                mutate: |c| {
                    c.force_mode = BURST_MODE_SILK;
                    c.frame_ms = 20.0;
                    c.max_bandwidth = bindings::OPUS_BANDWIDTH_WIDEBAND;
                    c.bitrate = 24000;
                },
            },
        ],
    },
    Burst {
        name: "framesize_20_40_20",
        min_channels: 1,
        steps: [
            BurstStep {
                label: "20ms",
                hold: 1,
                mutate: |c| {
                    c.frame_ms = 20.0;
                },
            },
            BurstStep {
                label: "40ms",
                hold: 1,
                mutate: |c| {
                    c.frame_ms = 40.0;
                },
            },
            BurstStep {
                label: "20ms",
                hold: 1,
                mutate: |c| {
                    c.frame_ms = 20.0;
                },
            },
        ],
    },
    Burst {
        name: "bitrate_crash_6k_510k_6k",
        min_channels: 1,
        steps: [
            BurstStep {
                label: "br=6k",
                hold: 1,
                mutate: |c| {
                    c.bitrate = 6000;
                },
            },
            BurstStep {
                label: "br=510k",
                hold: 1,
                mutate: |c| {
                    c.bitrate = 510000;
                },
            },
            BurstStep {
                label: "br=6k",
                hold: 1,
                mutate: |c| {
                    c.bitrate = 6000;
                },
            },
        ],
    },
    Burst {
        name: "bandwidth_NB_FB_NB",
        min_channels: 1,
        steps: [
            BurstStep {
                label: "NB",
                hold: 1,
                mutate: |c| {
                    c.max_bandwidth = bindings::OPUS_BANDWIDTH_NARROWBAND;
                },
            },
            BurstStep {
                label: "FB",
                hold: 1,
                mutate: |c| {
                    c.max_bandwidth = bindings::OPUS_BANDWIDTH_FULLBAND;
                },
            },
            BurstStep {
                label: "NB",
                hold: 1,
                mutate: |c| {
                    c.max_bandwidth = bindings::OPUS_BANDWIDTH_NARROWBAND;
                },
            },
        ],
    },
    Burst {
        name: "fec_off_on_off",
        min_channels: 1,
        steps: [
            BurstStep {
                label: "fec=0",
                hold: 1,
                mutate: |c| {
                    c.fec = 0;
                    c.packet_loss_pct = 0;
                },
            },
            BurstStep {
                label: "fec=1 loss20",
                hold: 1,
                mutate: |c| {
                    c.fec = 1;
                    c.packet_loss_pct = 20;
                },
            },
            BurstStep {
                label: "fec=0",
                hold: 1,
                mutate: |c| {
                    c.fec = 0;
                    c.packet_loss_pct = 0;
                },
            },
        ],
    },
];

/// Apply a burst step on top of the neutral baseline.
fn apply_burst_step(baseline: &EncodeConfig, step: &BurstStep) -> EncodeConfig {
    let mut cfg = baseline.clone();
    (step.mutate)(&mut cfg);
    cfg
}

/// Pick the burst at a given frame_idx / burst_interval boundary, filtering
/// out bursts that require more channels than the test has.
fn select_burst(frame_idx: usize, burst_interval: usize, channels: i32) -> &'static Burst {
    // Filter to usable bursts first so the rotation is stable relative to
    // the channel count (i.e. a mono run and a stereo run both cycle
    // through the same subset deterministically).
    let usable: Vec<&'static Burst> = BURSTS
        .iter()
        .filter(|b| b.min_channels <= channels)
        .collect();
    assert!(
        !usable.is_empty(),
        "no bursts usable at channels={}",
        channels
    );
    let boundary = frame_idx / burst_interval.max(1);
    usable[boundary % usable.len()]
}

// ---------------------------------------------------------------------------
// Config provenance for diagnostics (HLD Decision 8).
// ---------------------------------------------------------------------------
// When a mismatch fires inside a torture run, the diagnostic needs to say
// whether the frame came from the random walk or from a scripted burst. A
// small enum wrapping the active config makes that unambiguous.
#[derive(Clone, Debug)]
enum CurrentCfgSource {
    Random,
    Burst {
        name: &'static str,
        step_idx: usize,
        step_label: &'static str,
    },
}

impl CurrentCfgSource {
    fn format(&self) -> String {
        match self {
            CurrentCfgSource::Random => "random".to_string(),
            CurrentCfgSource::Burst {
                name,
                step_idx,
                step_label,
            } => {
                format!("burst={} step={}:{}", name, step_idx, step_label)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Section 2: Cross-decoding (HLD 2026.04.08)
// ---------------------------------------------------------------------------
// In `--cross-decode` mode we run every encoded packet (both the C packet
// and the Rust packet) through both a C decoder and a Rust decoder, so the
// 2x2 matrix is:
//
//              decode with C    decode with Rust
//   encode C:      A                  B
//   encode R:      C                  D
//
// Strong pass (when encoders agree byte-for-byte): A == B == C == D.
// Weak pass (when encoders diverge): A == B AND C == D — i.e. decoders
// agree on each encoder's output individually.
//
// Two columns of the matrix (B,D) and (A,C) each see a fixed encoder's
// stream, so the "mode-transition state across packets" constraint from the
// HLD means we need four *separate* decoder instances: one dedicated to
// each (encoder, decoder) pair.

/// Run a single packet through a C decoder and return PCM + final_range.
/// Overwrites `out` with the decoded samples. Returns `None` if decode
/// failed.
unsafe fn c_decode_packet(
    dec: *mut bindings::OpusDecoder,
    pkt: &[u8],
    out: &mut [i16],
    max_per_ch: i32,
) -> Option<(i32, u32)> {
    let n = unsafe {
        bindings::opus_decode(
            dec,
            pkt.as_ptr(),
            pkt.len() as i32,
            out.as_mut_ptr(),
            max_per_ch,
            0,
        )
    };
    if n < 0 {
        return None;
    }
    let mut rng: u32 = 0;
    unsafe {
        bindings::opus_decoder_ctl(
            dec,
            bindings::OPUS_GET_FINAL_RANGE_REQUEST,
            &mut rng as *mut u32,
        );
    }
    Some((n, rng))
}

/// Run a single packet through a Rust decoder and return (nsamples, range).
fn rust_decode_packet(
    dec: &mut mdopus::opus::decoder::OpusDecoder,
    pkt: &[u8],
    out: &mut [i16],
    max_per_ch: i32,
) -> Option<(i32, u32)> {
    match dec.decode(Some(pkt), out, max_per_ch, false) {
        Ok(n) => Some((n, dec.get_final_range())),
        Err(_) => None,
    }
}

#[allow(clippy::too_many_arguments, clippy::manual_is_multiple_of)]
fn cmd_torture(
    duration_secs: i32,
    seed: u64,
    change_interval: usize,
    sample_rate: i32,
    channels: i32,
    cross_decode: bool,
    burst_interval: usize,
    state_check_interval: usize,
    state_check_strict: bool,
) {
    println!("=== Torture Test ===");
    println!(
        "Duration: {}s, Seed: {}, Config change every {} frames",
        duration_secs, seed, change_interval
    );
    println!("Sample rate: {} Hz, Channels: {}", sample_rate, channels);
    println!(
        "Cross-decode: {}, Burst interval: {} (0=disabled), State check interval: {} (0=disabled), strict={}",
        cross_decode, burst_interval, state_check_interval, state_check_strict,
    );
    println!();

    let rss_start = get_rss_bytes();
    if let Some(rss) = rss_start {
        println!("RSS at start: {:.1} MB", rss as f64 / 1_048_576.0);
    }

    // Generate PCM for the full duration
    let pcm = generate_noise(sample_rate, channels, duration_secs as f64, seed);

    // --- Create C encoder + decoder ---
    let c_enc = unsafe {
        let mut err: i32 = 0;
        let enc = bindings::opus_encoder_create(
            sample_rate,
            channels,
            bindings::OPUS_APPLICATION_AUDIO,
            &mut err,
        );
        if enc.is_null() || err != bindings::OPUS_OK {
            eprintln!(
                "ERROR: C encoder create failed: {}",
                bindings::error_string(err)
            );
            process::exit(1);
        }
        enc
    };
    let c_dec_cs = unsafe {
        // Primary C decoder: fed the C-encoded stream.
        let mut err: i32 = 0;
        let dec = bindings::opus_decoder_create(sample_rate, channels, &mut err);
        if dec.is_null() || err != bindings::OPUS_OK {
            eprintln!(
                "ERROR: C decoder (C stream) create failed: {}",
                bindings::error_string(err)
            );
            process::exit(1);
        }
        dec
    };

    // --- Create Rust encoder + decoder ---
    let mut rust_enc = mdopus::opus::encoder::OpusEncoder::new(
        sample_rate,
        channels,
        bindings::OPUS_APPLICATION_AUDIO,
    )
    .unwrap_or_else(|e| {
        eprintln!("ERROR: Rust encoder create failed: {}", e);
        process::exit(1);
    });
    let mut rust_dec_cs = mdopus::opus::decoder::OpusDecoder::new(sample_rate, channels)
        .unwrap_or_else(|e| {
            eprintln!("ERROR: Rust decoder (C stream) create failed: {}", e);
            process::exit(1);
        });

    // --- Section 2: cross-decode extras ---
    // When --cross-decode is on we also route the Rust-encoded stream
    // through a dedicated C decoder and a dedicated Rust decoder. Those
    // are allocated lazily so the default torture run has unchanged
    // memory footprint.
    let c_dec_rs: *mut bindings::OpusDecoder = if cross_decode {
        unsafe {
            let mut err: i32 = 0;
            let dec = bindings::opus_decoder_create(sample_rate, channels, &mut err);
            if dec.is_null() || err != bindings::OPUS_OK {
                eprintln!(
                    "ERROR: C decoder (Rust stream) create failed: {}",
                    bindings::error_string(err)
                );
                process::exit(1);
            }
            dec
        }
    } else {
        std::ptr::null_mut()
    };
    let mut rust_dec_rs_opt = if cross_decode {
        Some(
            mdopus::opus::decoder::OpusDecoder::new(sample_rate, channels).unwrap_or_else(|e| {
                eprintln!("ERROR: Rust decoder (Rust stream) create failed: {}", e);
                process::exit(1);
            }),
        )
    } else {
        None
    };

    // Buffers — 120ms is the absolute max Opus frame at any sample rate
    let max_frame_samples_per_ch = sample_rate as usize * 120 / 1000;
    let max_frame_samples = max_frame_samples_per_ch * channels as usize;
    let mut c_pkt = vec![0u8; 4000];
    let mut rust_pkt = vec![0u8; 4000];
    // Matrix cells:
    //   pcm_a = C decoder fed C packet
    //   pcm_b = Rust decoder fed C packet
    //   pcm_c = C decoder fed Rust packet   (cross-decode only)
    //   pcm_d = Rust decoder fed Rust packet (cross-decode only)
    let mut pcm_a = vec![0i16; max_frame_samples];
    let mut pcm_b = vec![0i16; max_frame_samples];
    let mut pcm_c = if cross_decode {
        vec![0i16; max_frame_samples]
    } else {
        Vec::new()
    };
    let mut pcm_d = if cross_decode {
        vec![0i16; max_frame_samples]
    } else {
        Vec::new()
    };

    let mut rng = seed.wrapping_add(0xDEAD_BEEF); // Offset from PCM seed
    let mut frame_idx: usize = 0;
    let mut pcm_pos: usize = 0;
    let mut configs_tested: usize = 0;
    let mut encode_mismatches: usize = 0;
    let mut decode_mismatches: usize = 0;
    let mut range_mismatches: usize = 0;
    let mut cross_decode_mismatches: usize = 0;
    let mut state_check_mismatches: usize = 0;
    let mut max_pcm_diff: i32 = 0;
    let mut first_fail_frame: Option<usize> = None;
    // Once the encoders have produced any divergent packet, the four
    // cross-decoder instances no longer share identical input histories
    // — so the "strong" A==C / B==D checks would false-positive on
    // subsequent frames that happen to match byte-for-byte. Track
    // whether we've ever seen a divergence and only run the strong
    // checks while this is false. The weak C==D check is still valid
    // on any divergent frame because c_dec_rs and rust_dec_rs share a
    // history (both fed rust_pkt from frame 0).
    let mut prior_divergence_seen = false;

    // --- Section 3: burst scheduler state ---
    // When a burst is active, `burst_current` is Some and the state machine
    // walks the 3 steps. On burst completion the pre-burst config is
    // restored so the random walk picks up where it left off.
    let mut burst_current: Option<&'static Burst> = None;
    let mut burst_step_idx: usize = 0;
    let mut burst_step_frames_done: usize = 0;
    let mut burst_saved_cfg: Option<EncodeConfig> = None;
    let mut bursts_fired: usize = 0;
    // Burst mismatches get their own 5-line print budget so random-walk
    // noise does not starve burst diagnostics (devil's-advocate finding
    // #15, 2026.04.09). The burst label is far more actionable than a
    // random config dump — pinpointing burst=name step=k is the whole
    // point of the scheduled sequences.
    let mut burst_mismatches_printed: usize = 0;

    let mut current_cfg = EncodeConfig::new(sample_rate, channels);
    let mut current_cfg_source = CurrentCfgSource::Random;
    let mut current_frame_size = current_cfg.frame_size();

    unsafe { apply_config_to_c_encoder(c_enc, &current_cfg) };
    apply_config_to_rust_encoder(&mut rust_enc, &current_cfg);

    let start_time = std::time::Instant::now();

    loop {
        let frame_samples = current_frame_size * channels as usize;
        if pcm_pos + frame_samples > pcm.len() {
            break;
        }

        // --- Config selection state machine ---
        // Priority order per frame:
        //   1. Advance an in-progress burst (or finish it).
        //   2. If burst_interval boundary, start a new burst.
        //   3. Otherwise, maybe run a random-walk config change.
        let mut cfg_changed = false;

        if let Some(burst) = burst_current {
            burst_step_frames_done += 1;
            let current_step_hold = burst.steps[burst_step_idx].hold;
            if burst_step_frames_done > current_step_hold {
                burst_step_idx += 1;
                burst_step_frames_done = 1;
                if burst_step_idx >= burst.steps.len() {
                    // Burst complete — restore the pre-burst random-walk config.
                    if let Some(saved) = burst_saved_cfg.take() {
                        current_cfg = saved;
                    }
                    current_cfg_source = CurrentCfgSource::Random;
                    burst_current = None;
                    burst_step_idx = 0;
                    burst_step_frames_done = 0;
                    cfg_changed = true;
                } else {
                    let base = burst_neutral_baseline(sample_rate, channels);
                    current_cfg = apply_burst_step(&base, &burst.steps[burst_step_idx]);
                    current_cfg_source = CurrentCfgSource::Burst {
                        name: burst.name,
                        step_idx: burst_step_idx,
                        step_label: burst.steps[burst_step_idx].label,
                    };
                    cfg_changed = true;
                }
            }
        } else {
            // Not currently in a burst.
            let start_burst =
                burst_interval > 0 && frame_idx > 0 && frame_idx % burst_interval == 0;
            if start_burst {
                let burst = select_burst(frame_idx, burst_interval, channels);
                burst_saved_cfg = Some(current_cfg.clone());
                burst_current = Some(burst);
                burst_step_idx = 0;
                burst_step_frames_done = 1;
                bursts_fired += 1;
                let base = burst_neutral_baseline(sample_rate, channels);
                current_cfg = apply_burst_step(&base, &burst.steps[0]);
                current_cfg_source = CurrentCfgSource::Burst {
                    name: burst.name,
                    step_idx: 0,
                    step_label: burst.steps[0].label,
                };
                cfg_changed = true;
            } else if frame_idx > 0 && frame_idx % change_interval == 0 {
                current_cfg = random_torture_config(&mut rng, sample_rate, channels);
                current_cfg_source = CurrentCfgSource::Random;
                cfg_changed = true;
                configs_tested += 1;
            }
        }

        if cfg_changed {
            unsafe { apply_config_to_c_encoder(c_enc, &current_cfg) };
            apply_config_to_rust_encoder(&mut rust_enc, &current_cfg);
            current_frame_size = current_cfg.frame_size();

            // Re-check PCM availability after frame size change
            let frame_samples = current_frame_size * channels as usize;
            if pcm_pos + frame_samples > pcm.len() {
                break;
            }
        }

        let frame_samples = current_frame_size * channels as usize;

        // --- Encode with C ---
        // Once per frame, compute whether we are currently inside a
        // burst. Diagnostic print gates consult this so that burst
        // failures get their own 5-line budget (devil's-advocate
        // finding #15) — a burst mismatch is much more actionable than
        // a random-walk mismatch because the burst name + step
        // pinpoints the exact code path.
        let in_burst = matches!(current_cfg_source, CurrentCfgSource::Burst { .. });
        let burst_print_budget_ok = burst_mismatches_printed < 5;

        let c_len = unsafe {
            bindings::opus_encode(
                c_enc,
                pcm[pcm_pos..].as_ptr(),
                current_frame_size as i32,
                c_pkt.as_mut_ptr(),
                c_pkt.len() as i32,
            )
        };

        // --- Encode with Rust ---
        let rust_pkt_len = rust_pkt.len() as i32;
        let rust_len = match rust_enc.encode(
            &pcm[pcm_pos..pcm_pos + frame_samples],
            current_frame_size as i32,
            &mut rust_pkt,
            rust_pkt_len,
        ) {
            Ok(n) => n,
            Err(e) => {
                if c_len >= 0 {
                    encode_mismatches += 1;
                    let print_ok = encode_mismatches <= 5 || (in_burst && burst_print_budget_ok);
                    if print_ok {
                        if in_burst {
                            burst_mismatches_printed += 1;
                        }
                        println!(
                            "ENCODE ERROR frame {}: Rust failed ({}), C ok ({} bytes)",
                            frame_idx, e, c_len
                        );
                        println!(
                            "  {} / src: {}",
                            format_torture_config(&current_cfg),
                            current_cfg_source.format()
                        );
                    }
                    if first_fail_frame.is_none() {
                        first_fail_frame = Some(frame_idx);
                    }
                }
                pcm_pos += frame_samples;
                frame_idx += 1;
                continue;
            }
        };

        // Both failed — skip (valid: some configs are rejected by both)
        if c_len < 0 {
            pcm_pos += frame_samples;
            frame_idx += 1;
            continue;
        }

        let c_len = c_len as usize;
        let rust_len = rust_len as usize;

        // --- Compare encoded bytes ---
        let packets_equal = c_len == rust_len && c_pkt[..c_len] == rust_pkt[..rust_len];
        if !packets_equal {
            encode_mismatches += 1;
            prior_divergence_seen = true;
            let print_ok = encode_mismatches <= 5 || (in_burst && burst_print_budget_ok);
            if print_ok {
                if in_burst {
                    burst_mismatches_printed += 1;
                }
                println!(
                    "ENCODE MISMATCH frame {} (config #{}): C={} bytes, Rust={} bytes",
                    frame_idx, configs_tested, c_len, rust_len
                );
                println!(
                    "  {} / src: {}",
                    format_torture_config(&current_cfg),
                    current_cfg_source.format()
                );
                if c_len == rust_len {
                    for k in 0..c_len {
                        if c_pkt[k] != rust_pkt[k] {
                            println!(
                                "  First diff byte {}: C=0x{:02x} Rust=0x{:02x}",
                                k, c_pkt[k], rust_pkt[k]
                            );
                            break;
                        }
                    }
                }
            }
            if first_fail_frame.is_none() {
                first_fail_frame = Some(frame_idx);
            }
        }

        // --- Decode (2x2 matrix if cross-decode, else 1x2) ---
        // A: c_dec_cs  fed C packet
        // B: rust_dec_cs fed C packet
        // C: c_dec_rs  fed Rust packet (cross_decode only)
        // D: rust_dec_rs fed Rust packet (cross_decode only)
        let a = unsafe {
            c_decode_packet(
                c_dec_cs,
                &c_pkt[..c_len],
                &mut pcm_a,
                max_frame_samples_per_ch as i32,
            )
        };
        let b = rust_decode_packet(
            &mut rust_dec_cs,
            &c_pkt[..c_len],
            &mut pcm_b,
            max_frame_samples_per_ch as i32,
        );
        let c_res = if cross_decode {
            unsafe {
                c_decode_packet(
                    c_dec_rs,
                    &rust_pkt[..rust_len],
                    &mut pcm_c,
                    max_frame_samples_per_ch as i32,
                )
            }
        } else {
            None
        };
        let d_res = if cross_decode {
            rust_decode_packet(
                rust_dec_rs_opt.as_mut().unwrap(),
                &rust_pkt[..rust_len],
                &mut pcm_d,
                max_frame_samples_per_ch as i32,
            )
        } else {
            None
        };

        // --- A vs B check (this is the existing "Rust decoder matches C
        //     decoder on the C-encoded stream" check, unchanged semantics).
        match (a, b) {
            (Some((an, ar)), Some((bn, br))) => {
                let an_samples = an as usize * channels as usize;
                let bn_samples = bn as usize * channels as usize;
                let cmp_len = an_samples.min(bn_samples);
                let mut local_max_diff = 0i32;
                for j in 0..cmp_len {
                    let d = (pcm_a[j] as i32 - pcm_b[j] as i32).abs();
                    if d > local_max_diff {
                        local_max_diff = d;
                    }
                }
                if local_max_diff > 0 {
                    decode_mismatches += 1;
                    max_pcm_diff = max_pcm_diff.max(local_max_diff);
                    let print_ok = decode_mismatches <= 5 || (in_burst && burst_print_budget_ok);
                    if print_ok {
                        if in_burst {
                            burst_mismatches_printed += 1;
                        }
                        println!(
                            "DECODE MISMATCH frame {}: max PCM diff {} (A vs B, c_len={})",
                            frame_idx, local_max_diff, c_len
                        );
                        println!(
                            "  {} / src: {}",
                            format_torture_config(&current_cfg),
                            current_cfg_source.format()
                        );
                    }
                    if first_fail_frame.is_none() {
                        first_fail_frame = Some(frame_idx);
                    }
                }
                if ar != br {
                    range_mismatches += 1;
                    let print_ok = range_mismatches <= 5 || (in_burst && burst_print_budget_ok);
                    if print_ok {
                        if in_burst {
                            burst_mismatches_printed += 1;
                        }
                        println!(
                            "RANGE MISMATCH frame {}: A.rng=0x{:08x} B.rng=0x{:08x}",
                            frame_idx, ar, br
                        );
                        println!(
                            "  {} / src: {}",
                            format_torture_config(&current_cfg),
                            current_cfg_source.format()
                        );
                    }
                    if first_fail_frame.is_none() {
                        first_fail_frame = Some(frame_idx);
                    }
                }
            }
            (Some(_), None) => {
                decode_mismatches += 1;
                let print_ok = decode_mismatches <= 5 || (in_burst && burst_print_budget_ok);
                if print_ok {
                    if in_burst {
                        burst_mismatches_printed += 1;
                    }
                    println!(
                        "DECODE ERROR frame {}: Rust decoder rejected C packet",
                        frame_idx
                    );
                    println!(
                        "  {} / src: {}",
                        format_torture_config(&current_cfg),
                        current_cfg_source.format()
                    );
                }
                if first_fail_frame.is_none() {
                    first_fail_frame = Some(frame_idx);
                }
            }
            (None, Some(_)) => {
                decode_mismatches += 1;
                let print_ok = decode_mismatches <= 5 || (in_burst && burst_print_budget_ok);
                if print_ok {
                    if in_burst {
                        burst_mismatches_printed += 1;
                    }
                    println!(
                        "DECODE ERROR frame {}: C decoder rejected C packet (Rust accepted)",
                        frame_idx
                    );
                    println!(
                        "  {} / src: {}",
                        format_torture_config(&current_cfg),
                        current_cfg_source.format()
                    );
                }
                if first_fail_frame.is_none() {
                    first_fail_frame = Some(frame_idx);
                }
            }
            (None, None) => {
                // Both decoders rejected — not a divergence.
            }
        }

        // --- Cross-decode checks (Section 2) ---
        // The strong check (A vs C, B vs D on packets_equal frames) is
        // only meaningful while the four decoder instances share an
        // identical input history. Once the encoders have diverged on
        // any prior frame, decoder histories differ forever and the
        // strong check false-positives. `prior_divergence_seen` gates it.
        // The weak check (C vs D on divergent frames) is always valid:
        // c_dec_rs and rust_dec_rs both see the Rust-encoded stream from
        // frame 0, so their histories match regardless of encoder drift.
        if cross_decode {
            if packets_equal && !prior_divergence_seen {
                // Compare A vs C (same bytes, different decoder instance:
                // should be bit-identical because they've seen identical
                // input sequences from frame 0).
                let cross_pcm_fail = match (a, c_res) {
                    (Some((an, ar)), Some((cn, cr))) => {
                        let an_samples = an as usize * channels as usize;
                        let cn_samples = cn as usize * channels as usize;
                        let cmp_len = an_samples.min(cn_samples);
                        let mut diff = false;
                        for j in 0..cmp_len {
                            if pcm_a[j] != pcm_c[j] {
                                diff = true;
                                break;
                            }
                        }
                        diff || ar != cr
                    }
                    (None, None) => false,
                    _ => true,
                };
                if cross_pcm_fail {
                    cross_decode_mismatches += 1;
                    let print_ok =
                        cross_decode_mismatches <= 5 || (in_burst && burst_print_budget_ok);
                    if print_ok {
                        if in_burst {
                            burst_mismatches_printed += 1;
                        }
                        println!(
                            "CROSS-DECODE MISMATCH (A vs C, packets equal) @ frame {}",
                            frame_idx
                        );
                        println!(
                            "  {} / src: {}",
                            format_torture_config(&current_cfg),
                            current_cfg_source.format()
                        );
                    }
                    if first_fail_frame.is_none() {
                        first_fail_frame = Some(frame_idx);
                    }
                }
                // Compare B vs D.
                let cross_rust_fail = match (b, d_res) {
                    (Some((bn, br)), Some((dn, dr))) => {
                        let bn_samples = bn as usize * channels as usize;
                        let dn_samples = dn as usize * channels as usize;
                        let cmp_len = bn_samples.min(dn_samples);
                        let mut diff = false;
                        for j in 0..cmp_len {
                            if pcm_b[j] != pcm_d[j] {
                                diff = true;
                                break;
                            }
                        }
                        diff || br != dr
                    }
                    (None, None) => false,
                    _ => true,
                };
                if cross_rust_fail {
                    cross_decode_mismatches += 1;
                    let print_ok =
                        cross_decode_mismatches <= 5 || (in_burst && burst_print_budget_ok);
                    if print_ok {
                        if in_burst {
                            burst_mismatches_printed += 1;
                        }
                        println!(
                            "CROSS-DECODE MISMATCH (B vs D, packets equal) @ frame {}",
                            frame_idx
                        );
                        println!(
                            "  {} / src: {}",
                            format_torture_config(&current_cfg),
                            current_cfg_source.format()
                        );
                    }
                    if first_fail_frame.is_none() {
                        first_fail_frame = Some(frame_idx);
                    }
                }
            } else if !packets_equal {
                // Packets diverged. Weak criterion: each encoder's output
                // must decode the same way under both decoders. A vs B is
                // already covered; still to cover is C vs D.
                let weak_fail_cd = match (c_res, d_res) {
                    (Some((cn, cr)), Some((dn, dr))) => {
                        let cn_samples = cn as usize * channels as usize;
                        let dn_samples = dn as usize * channels as usize;
                        let cmp_len = cn_samples.min(dn_samples);
                        let mut diff = false;
                        for j in 0..cmp_len {
                            if pcm_c[j] != pcm_d[j] {
                                diff = true;
                                break;
                            }
                        }
                        diff || cr != dr
                    }
                    (None, None) => false,
                    _ => true,
                };
                if weak_fail_cd {
                    cross_decode_mismatches += 1;
                    let print_ok =
                        cross_decode_mismatches <= 5 || (in_burst && burst_print_budget_ok);
                    if print_ok {
                        if in_burst {
                            burst_mismatches_printed += 1;
                        }
                        println!(
                            "CROSS-DECODE MISMATCH (C vs D, packets diverged) @ frame {}",
                            frame_idx
                        );
                        println!(
                            "  {} / src: {}",
                            format_torture_config(&current_cfg),
                            current_cfg_source.format()
                        );
                    }
                    if first_fail_frame.is_none() {
                        first_fail_frame = Some(frame_idx);
                    }
                }
            }
        }

        pcm_pos += frame_samples;
        frame_idx += 1;

        // --- Section 6: periodic state comparison ---
        // In cross-decode mode the four decoder instances have diverging
        // histories after the first encoder mismatch, so we deliberately
        // checkpoint only the column (c_dec_cs vs rust_dec_cs) — the
        // primary C-stream pair. That is the only comparison that has
        // a meaningful bit-exact invariant: both decoders consumed the
        // C-encoded stream from frame 0, so they must agree on every
        // getter. The rust-stream column (c_dec_rs vs rust_dec_rs) has
        // the same invariant but is redundant in terms of drift
        // detection. Documented per devil's-advocate finding #4.
        if state_check_interval > 0 && frame_idx % state_check_interval == 0 {
            let diffs = state_checkpoint_diffs(c_enc, &rust_enc, c_dec_cs, &rust_dec_cs);
            if !diffs.is_empty() {
                state_check_mismatches += 1;
                if state_check_mismatches <= 5 {
                    println!(
                        "STATE CHECK MISMATCH @ frame {} ({} diffs):",
                        frame_idx,
                        diffs.len()
                    );
                    for dline in &diffs {
                        println!("  {}", dline);
                    }
                    println!(
                        "  cfg: {} / src: {}",
                        format_torture_config(&current_cfg),
                        current_cfg_source.format()
                    );
                }
                if first_fail_frame.is_none() {
                    first_fail_frame = Some(frame_idx);
                }
                if state_check_strict {
                    eprintln!();
                    eprintln!("STATE CHECK STRICT MODE: aborting at frame {}", frame_idx);
                    break;
                }
            }
        }

        // Progress every 500 frames
        if frame_idx % 500 == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let pct = pcm_pos as f64 / pcm.len() as f64 * 100.0;
            eprint!(
                "\r  [{:.0}s] frame {} ({:.1}%), cfgs: {}, bursts: {}, enc_err: {}, dec_err: {}, rng_err: {}, xdec_err: {}, state_err: {}   ",
                elapsed,
                frame_idx,
                pct,
                configs_tested,
                bursts_fired,
                encode_mismatches,
                decode_mismatches,
                range_mismatches,
                cross_decode_mismatches,
                state_check_mismatches,
            );
        }
    }
    eprintln!();

    // Final state checkpoint at end of test — catches drift that never
    // crossed an interval boundary.
    if state_check_interval > 0 && frame_idx > 0 {
        let diffs = state_checkpoint_diffs(c_enc, &rust_enc, c_dec_cs, &rust_dec_cs);
        if !diffs.is_empty() {
            state_check_mismatches += 1;
            println!(
                "STATE CHECK MISMATCH @ end-of-test frame {} ({} diffs):",
                frame_idx,
                diffs.len()
            );
            for dline in &diffs {
                println!("  {}", dline);
            }
            println!(
                "  cfg: {} / src: {}",
                format_torture_config(&current_cfg),
                current_cfg_source.format()
            );
            if first_fail_frame.is_none() {
                first_fail_frame = Some(frame_idx);
            }
        }
    }

    let elapsed = start_time.elapsed();
    let rss_end = get_rss_bytes();

    unsafe {
        bindings::opus_encoder_destroy(c_enc);
        bindings::opus_decoder_destroy(c_dec_cs);
        if !c_dec_rs.is_null() {
            bindings::opus_decoder_destroy(c_dec_rs);
        }
    }
    drop(rust_dec_rs_opt);

    println!();
    println!("=== Torture Test Summary ===");
    println!("  Duration:        {:.1}s", elapsed.as_secs_f64());
    println!("  Frames encoded:  {}", frame_idx);
    println!("  Configs tested:  {}", configs_tested);
    println!("  Bursts fired:    {}", bursts_fired);
    println!("  Encode errors:   {}", encode_mismatches);
    println!("  Decode errors:   {}", decode_mismatches);
    println!("  Range errors:    {}", range_mismatches);
    if cross_decode {
        println!("  X-decode errors: {}", cross_decode_mismatches);
    }
    if state_check_interval > 0 {
        println!("  State errors:    {}", state_check_mismatches);
    }
    if max_pcm_diff > 0 {
        println!("  Max PCM diff:    {}", max_pcm_diff);
    }
    if let (Some(start), Some(end)) = (rss_start, rss_end) {
        let delta = end as i64 - start as i64;
        println!(
            "  RSS: {:.1} MB -> {:.1} MB ({:+.1} MB)",
            start as f64 / 1_048_576.0,
            end as f64 / 1_048_576.0,
            delta as f64 / 1_048_576.0
        );
        if delta > 50 * 1024 * 1024 {
            println!("  WARNING: RSS grew >50 MB — possible memory leak");
        }
    }

    let all_pass = encode_mismatches == 0
        && decode_mismatches == 0
        && range_mismatches == 0
        && cross_decode_mismatches == 0
        && (!state_check_strict || state_check_mismatches == 0);
    if all_pass {
        println!("  RESULT: PASS");
    } else {
        println!(
            "  RESULT: FAIL (first failure at frame {})",
            first_fail_frame.unwrap_or(0)
        );
        process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Mode Transition Tests: deterministic sequences exercising risky transitions
// ---------------------------------------------------------------------------

/// Run a sequence of (config, frame_count) steps through paired C/Rust encoders.
/// Returns true if all frames match.
fn run_transition_sequence(
    label: &str,
    sample_rate: i32,
    channels: i32,
    steps: &[(EncodeConfig, usize)],
    cross_decode: bool,
) -> bool {
    let frame_size = (sample_rate / 50) as usize; // 20ms frames
    let total_frames: usize = steps.iter().map(|(_, n)| *n).sum();
    let duration = total_frames as f64 * 0.02; // 20ms per frame
    let pcm = generate_noise(sample_rate, channels, duration + 0.1, 54321);

    print!("  {}: ", label);

    let c_enc = unsafe {
        let mut err: i32 = 0;
        let enc = bindings::opus_encoder_create(
            sample_rate,
            channels,
            bindings::OPUS_APPLICATION_AUDIO,
            &mut err,
        );
        if enc.is_null() || err != 0 {
            println!("SKIP (C encoder failed)");
            return true;
        }
        enc
    };
    let mut rust_enc = match mdopus::opus::encoder::OpusEncoder::new(
        sample_rate,
        channels,
        bindings::OPUS_APPLICATION_AUDIO,
    ) {
        Ok(e) => e,
        Err(_) => {
            println!("SKIP (Rust encoder failed)");
            unsafe { bindings::opus_encoder_destroy(c_enc) };
            return true;
        }
    };

    // Primary decoders: fed the C-encoded stream.
    let c_dec_cs = unsafe {
        let mut err: i32 = 0;
        let dec = bindings::opus_decoder_create(sample_rate, channels, &mut err);
        if dec.is_null() || err != 0 {
            println!("SKIP (C decoder failed)");
            bindings::opus_encoder_destroy(c_enc);
            return true;
        }
        dec
    };
    let mut rust_dec_cs = match mdopus::opus::decoder::OpusDecoder::new(sample_rate, channels) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP (Rust decoder failed)");
            unsafe {
                bindings::opus_encoder_destroy(c_enc);
                bindings::opus_decoder_destroy(c_dec_cs);
            };
            return true;
        }
    };

    // Cross-decode extras: separate decoders fed the Rust-encoded stream.
    let c_dec_rs: *mut bindings::OpusDecoder = if cross_decode {
        unsafe {
            let mut err: i32 = 0;
            let dec = bindings::opus_decoder_create(sample_rate, channels, &mut err);
            if dec.is_null() || err != 0 {
                println!("SKIP (C decoder for Rust stream failed)");
                bindings::opus_encoder_destroy(c_enc);
                bindings::opus_decoder_destroy(c_dec_cs);
                return true;
            }
            dec
        }
    } else {
        std::ptr::null_mut()
    };
    let mut rust_dec_rs_opt = if cross_decode {
        match mdopus::opus::decoder::OpusDecoder::new(sample_rate, channels) {
            Ok(d) => Some(d),
            Err(_) => {
                println!("SKIP (Rust decoder for Rust stream failed)");
                unsafe {
                    bindings::opus_encoder_destroy(c_enc);
                    bindings::opus_decoder_destroy(c_dec_cs);
                    if !c_dec_rs.is_null() {
                        bindings::opus_decoder_destroy(c_dec_rs);
                    }
                }
                return true;
            }
        }
    } else {
        None
    };

    let max_dec_per_ch = sample_rate as usize * 120 / 1000;
    let max_frame_samples = max_dec_per_ch * channels as usize;
    let mut c_pkt = vec![0u8; 4000];
    let mut rust_pkt = vec![0u8; 4000];
    let mut pcm_a = vec![0i16; max_frame_samples];
    let mut pcm_b = vec![0i16; max_frame_samples];
    let mut pcm_c = if cross_decode {
        vec![0i16; max_frame_samples]
    } else {
        Vec::new()
    };
    let mut pcm_d = if cross_decode {
        vec![0i16; max_frame_samples]
    } else {
        Vec::new()
    };

    let frame_samples = frame_size * channels as usize;
    let mut pcm_pos: usize = 0;
    let mut frame_idx: usize = 0;
    let mut mismatches: usize = 0;
    let mut cross_mismatches: usize = 0;
    let mut first_fail: Option<(usize, usize)> = None; // (step, frame_within_step)
    // See cmd_torture for the rationale: the strong A==C / B==D checks
    // are only meaningful while the four decoder instances share an
    // identical input history. `prior_divergence_seen` gates them.
    let mut prior_divergence_seen = false;

    for (step_idx, (cfg, num_frames)) in steps.iter().enumerate() {
        unsafe { apply_config_to_c_encoder(c_enc, cfg) };
        apply_config_to_rust_encoder(&mut rust_enc, cfg);

        for f in 0..*num_frames {
            if pcm_pos + frame_samples > pcm.len() {
                break;
            }

            let c_len = unsafe {
                bindings::opus_encode(
                    c_enc,
                    pcm[pcm_pos..].as_ptr(),
                    frame_size as i32,
                    c_pkt.as_mut_ptr(),
                    c_pkt.len() as i32,
                )
            };
            let rpkt_len = rust_pkt.len() as i32;
            let rust_len = rust_enc
                .encode(
                    &pcm[pcm_pos..pcm_pos + frame_samples],
                    frame_size as i32,
                    &mut rust_pkt,
                    rpkt_len,
                )
                .unwrap_or(-1);

            if c_len < 0 && rust_len < 0 {
                // Both reject — fine
            } else if c_len >= 0 && rust_len >= 0 {
                let cl = c_len as usize;
                let rl = rust_len as usize;
                let packets_equal = cl == rl && c_pkt[..cl] == rust_pkt[..rl];
                if !packets_equal {
                    mismatches += 1;
                    prior_divergence_seen = true;
                    if first_fail.is_none() {
                        first_fail = Some((step_idx, f));
                        // Dump first mismatch details
                        eprintln!(
                            "    DIAG step={} frame={}: C={} bytes Rust={} bytes",
                            step_idx, f, cl, rl
                        );
                        if cl > 0 && rl > 0 {
                            eprintln!("    C_toc=0x{:02x} R_toc=0x{:02x}", c_pkt[0], rust_pkt[0]);
                        }
                        let min_len = cl.min(rl).min(16);
                        eprint!("    C_pkt:");
                        for b in &c_pkt[..min_len] {
                            eprint!(" {:02x}", b);
                        }
                        eprintln!();
                        eprint!("    R_pkt:");
                        for b in &rust_pkt[..min_len] {
                            eprint!(" {:02x}", b);
                        }
                        eprintln!();
                        // Find first diff
                        let cmp_len = cl.min(rl);
                        for k in 0..cmp_len {
                            if c_pkt[k] != rust_pkt[k] {
                                eprintln!(
                                    "    First diff at byte {}: C=0x{:02x} R=0x{:02x}",
                                    k, c_pkt[k], rust_pkt[k]
                                );
                                break;
                            }
                        }
                    }
                }

                // 2x2 decode matrix. A/B always run; C/D only in cross-decode.
                let a = unsafe {
                    c_decode_packet(c_dec_cs, &c_pkt[..cl], &mut pcm_a, max_dec_per_ch as i32)
                };
                let b = rust_decode_packet(
                    &mut rust_dec_cs,
                    &c_pkt[..cl],
                    &mut pcm_b,
                    max_dec_per_ch as i32,
                );
                let c_res = if cross_decode {
                    unsafe {
                        c_decode_packet(
                            c_dec_rs,
                            &rust_pkt[..rl],
                            &mut pcm_c,
                            max_dec_per_ch as i32,
                        )
                    }
                } else {
                    None
                };
                let d_res = if cross_decode {
                    rust_decode_packet(
                        rust_dec_rs_opt.as_mut().unwrap(),
                        &rust_pkt[..rl],
                        &mut pcm_d,
                        max_dec_per_ch as i32,
                    )
                } else {
                    None
                };

                // A vs B: existing check.
                match (a, b) {
                    (Some((an, ar)), Some((bn, br))) => {
                        let n =
                            (an as usize * channels as usize).min(bn as usize * channels as usize);
                        let mut pcm_diff = false;
                        for j in 0..n {
                            if pcm_a[j] != pcm_b[j] {
                                pcm_diff = true;
                                break;
                            }
                        }
                        if pcm_diff || ar != br {
                            mismatches += 1;
                            if first_fail.is_none() {
                                first_fail = Some((step_idx, f));
                            }
                        }
                    }
                    (None, None) => {}
                    _ => {
                        mismatches += 1;
                        if first_fail.is_none() {
                            first_fail = Some((step_idx, f));
                        }
                    }
                }

                // Cross-decode comparisons. Strong A/C + B/D checks
                // gated on no prior divergence — see cmd_torture.
                if cross_decode {
                    if packets_equal && !prior_divergence_seen {
                        // A vs C and B vs D must also match.
                        let ac_fail = match (a, c_res) {
                            (Some((an, ar)), Some((cn, cr))) => {
                                let n = (an as usize * channels as usize)
                                    .min(cn as usize * channels as usize);
                                let mut diff = false;
                                for j in 0..n {
                                    if pcm_a[j] != pcm_c[j] {
                                        diff = true;
                                        break;
                                    }
                                }
                                diff || ar != cr
                            }
                            (None, None) => false,
                            _ => true,
                        };
                        let bd_fail = match (b, d_res) {
                            (Some((bn, br)), Some((dn, dr))) => {
                                let n = (bn as usize * channels as usize)
                                    .min(dn as usize * channels as usize);
                                let mut diff = false;
                                for j in 0..n {
                                    if pcm_b[j] != pcm_d[j] {
                                        diff = true;
                                        break;
                                    }
                                }
                                diff || br != dr
                            }
                            (None, None) => false,
                            _ => true,
                        };
                        if ac_fail || bd_fail {
                            cross_mismatches += 1;
                            if first_fail.is_none() {
                                first_fail = Some((step_idx, f));
                            }
                        }
                    } else if !packets_equal {
                        // Weak: only C == D on divergent frames.
                        let cd_fail = match (c_res, d_res) {
                            (Some((cn, cr)), Some((dn, dr))) => {
                                let n = (cn as usize * channels as usize)
                                    .min(dn as usize * channels as usize);
                                let mut diff = false;
                                for j in 0..n {
                                    if pcm_c[j] != pcm_d[j] {
                                        diff = true;
                                        break;
                                    }
                                }
                                diff || cr != dr
                            }
                            (None, None) => false,
                            _ => true,
                        };
                        if cd_fail {
                            cross_mismatches += 1;
                            if first_fail.is_none() {
                                first_fail = Some((step_idx, f));
                            }
                        }
                    }
                }
            } else {
                // One succeeded, one failed
                mismatches += 1;
                if first_fail.is_none() {
                    first_fail = Some((step_idx, f));
                }
            }

            pcm_pos += frame_samples;
            frame_idx += 1;
        }
    }

    unsafe {
        bindings::opus_encoder_destroy(c_enc);
        bindings::opus_decoder_destroy(c_dec_cs);
        if !c_dec_rs.is_null() {
            bindings::opus_decoder_destroy(c_dec_rs);
        }
    }
    drop(rust_dec_rs_opt);

    let total = mismatches + cross_mismatches;
    if total == 0 {
        if cross_decode {
            println!("PASS ({} frames, cross-decode ok)", frame_idx);
        } else {
            println!("PASS ({} frames)", frame_idx);
        }
        true
    } else {
        let (s, f) = first_fail.unwrap_or((0, 0));
        if cross_decode {
            println!(
                "FAIL ({} primary + {} cross-decode mismatches in {} frames, first at step {} frame {})",
                mismatches, cross_mismatches, frame_idx, s, f
            );
        } else {
            println!(
                "FAIL ({} mismatches in {} frames, first at step {} frame {})",
                mismatches, frame_idx, s, f
            );
        }
        false
    }
}

fn cmd_transitions(cross_decode: bool) {
    println!("=== Mode Transition Tests ===");
    if cross_decode {
        println!("(cross-decode enabled)");
    }
    println!();
    let mut all_pass = true;

    // Frames per step — enough to let encoder state settle after each transition
    let n = 100;

    // --- Test 1: SILK -> CELT -> SILK mode cycling ---
    {
        let sr = 48000;
        let ch = 1;
        let mut silk = EncodeConfig::new(sr, ch);
        silk.force_mode = 1000; // SILK_ONLY
        silk.bitrate = 16000;
        silk.max_bandwidth = bindings::OPUS_BANDWIDTH_WIDEBAND;

        let mut celt = EncodeConfig::new(sr, ch);
        celt.force_mode = 1002; // CELT_ONLY
        celt.bitrate = 128000;
        celt.max_bandwidth = bindings::OPUS_BANDWIDTH_FULLBAND;

        all_pass &= run_transition_sequence(
            "SILK->CELT->SILK->CELT (mono)",
            sr,
            ch,
            &[
                (silk.clone(), n),
                (celt.clone(), n),
                (silk.clone(), n),
                (celt.clone(), n),
            ],
            cross_decode,
        );
    }

    // --- Test 2: Bandwidth sweep NB -> FB -> NB ---
    {
        let sr = 48000;
        let ch = 1;
        let bws = [
            bindings::OPUS_BANDWIDTH_NARROWBAND,
            bindings::OPUS_BANDWIDTH_MEDIUMBAND,
            bindings::OPUS_BANDWIDTH_WIDEBAND,
            bindings::OPUS_BANDWIDTH_SUPERWIDEBAND,
            bindings::OPUS_BANDWIDTH_FULLBAND,
            bindings::OPUS_BANDWIDTH_SUPERWIDEBAND,
            bindings::OPUS_BANDWIDTH_WIDEBAND,
            bindings::OPUS_BANDWIDTH_MEDIUMBAND,
            bindings::OPUS_BANDWIDTH_NARROWBAND,
        ];
        let steps: Vec<(EncodeConfig, usize)> = bws
            .iter()
            .map(|&bw| {
                let mut cfg = EncodeConfig::new(sr, ch);
                cfg.bitrate = 64000;
                cfg.max_bandwidth = bw;
                (cfg, n / 2)
            })
            .collect();
        all_pass &=
            run_transition_sequence("BW sweep NB->FB->NB (mono)", sr, ch, &steps, cross_decode);
    }

    // --- Test 3: VBR on/off cycling ---
    {
        let sr = 48000;
        let ch = 1;
        let mut vbr_on = EncodeConfig::new(sr, ch);
        vbr_on.vbr = 1;
        vbr_on.vbr_constraint = 1;
        vbr_on.bitrate = 64000;

        let mut vbr_off = EncodeConfig::new(sr, ch);
        vbr_off.vbr = 0;
        vbr_off.bitrate = 64000;

        all_pass &= run_transition_sequence(
            "VBR on->off->on->off (mono)",
            sr,
            ch,
            &[
                (vbr_on.clone(), n),
                (vbr_off.clone(), n),
                (vbr_on.clone(), n),
                (vbr_off.clone(), n),
            ],
            cross_decode,
        );
    }

    // --- Test 4: VOIP-like -> AUDIO-like switching via signal + force_mode ---
    {
        let sr = 48000;
        let ch = 1;
        let mut voip_like = EncodeConfig::new(sr, ch);
        voip_like.signal = bindings::OPUS_SIGNAL_VOICE;
        voip_like.bitrate = 16000;
        voip_like.force_mode = 1000; // SILK
        voip_like.max_bandwidth = bindings::OPUS_BANDWIDTH_WIDEBAND;
        voip_like.fec = 1;
        voip_like.packet_loss_pct = 10;

        let mut audio_like = EncodeConfig::new(sr, ch);
        audio_like.signal = bindings::OPUS_SIGNAL_MUSIC;
        audio_like.bitrate = 128000;
        audio_like.force_mode = 1002; // CELT
        audio_like.max_bandwidth = bindings::OPUS_BANDWIDTH_FULLBAND;
        audio_like.fec = 0;
        audio_like.packet_loss_pct = 0;

        all_pass &= run_transition_sequence(
            "VOIP-like->AUDIO-like->VOIP-like (mono)",
            sr,
            ch,
            &[
                (voip_like.clone(), n),
                (audio_like.clone(), n),
                (voip_like.clone(), n),
                (audio_like.clone(), n),
            ],
            cross_decode,
        );
    }

    // --- Test 5: DTX on/off cycling ---
    {
        let sr = 48000;
        let ch = 1;
        let mut dtx_on = EncodeConfig::new(sr, ch);
        dtx_on.dtx = 1;
        dtx_on.bitrate = 16000;
        dtx_on.signal = bindings::OPUS_SIGNAL_VOICE;

        let mut dtx_off = EncodeConfig::new(sr, ch);
        dtx_off.dtx = 0;
        dtx_off.bitrate = 64000;

        all_pass &= run_transition_sequence(
            "DTX on->off->on (mono)",
            sr,
            ch,
            &[
                (dtx_on.clone(), n),
                (dtx_off.clone(), n),
                (dtx_on.clone(), n),
            ],
            cross_decode,
        );
    }

    // --- Test 6: Stereo force-channels switching ---
    {
        let sr = 48000;
        let ch = 2;
        let mut stereo = EncodeConfig::new(sr, ch);
        stereo.bitrate = 96000;
        stereo.force_channels = 2;

        let mut mono_forced = EncodeConfig::new(sr, ch);
        mono_forced.bitrate = 96000;
        mono_forced.force_channels = 1;

        let mut auto_ch = EncodeConfig::new(sr, ch);
        auto_ch.bitrate = 96000;
        auto_ch.force_channels = bindings::OPUS_AUTO;

        all_pass &= run_transition_sequence(
            "Stereo->ForceMono->Auto->Stereo",
            sr,
            ch,
            &[
                (stereo.clone(), n),
                (mono_forced.clone(), n),
                (auto_ch.clone(), n),
                (stereo.clone(), n),
            ],
            cross_decode,
        );
    }

    // --- Test 7: Bitrate extremes ---
    {
        let sr = 48000;
        let ch = 1;
        let bitrates = [6000, 510000, 6000, 128000, 6000];
        let steps: Vec<(EncodeConfig, usize)> = bitrates
            .iter()
            .map(|&br| {
                let mut cfg = EncodeConfig::new(sr, ch);
                cfg.bitrate = br;
                (cfg, n)
            })
            .collect();
        all_pass &= run_transition_sequence(
            "Bitrate extremes 6k->510k->6k (mono)",
            sr,
            ch,
            &steps,
            cross_decode,
        );
    }

    // --- Test 8: FEC + packet loss cycling ---
    {
        let sr = 48000;
        let ch = 1;
        let mut fec_off = EncodeConfig::new(sr, ch);
        fec_off.fec = 0;
        fec_off.bitrate = 32000;

        let mut fec_low = EncodeConfig::new(sr, ch);
        fec_low.fec = 1;
        fec_low.packet_loss_pct = 5;
        fec_low.bitrate = 32000;

        let mut fec_high = EncodeConfig::new(sr, ch);
        fec_high.fec = 1;
        fec_high.packet_loss_pct = 30;
        fec_high.bitrate = 32000;

        all_pass &= run_transition_sequence(
            "FEC off->low_loss->high_loss->off (mono)",
            sr,
            ch,
            &[
                (fec_off.clone(), n),
                (fec_low.clone(), n),
                (fec_high.clone(), n),
                (fec_off.clone(), n),
            ],
            cross_decode,
        );
    }

    println!();
    if all_pass {
        println!("=== All transition tests PASSED ===");
    } else {
        println!("=== Some transition tests FAILED ===");
        process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Quality: SNR sanity check for encode-decode roundtrip
// ---------------------------------------------------------------------------

/// Compute SNR in dB between original and decoded signals.
/// The codec introduces an algorithmic delay of `delay` samples, so
/// decoded[delay..] corresponds to original[0..]. We compare the
/// overlapping region: original[0..n-delay] vs decoded[delay..n].
fn compute_snr(original: &[i16], decoded: &[i16], delay: usize) -> f64 {
    let orig_len = original.len();
    let dec_len = decoded.len();
    if delay >= dec_len || delay >= orig_len {
        return f64::NEG_INFINITY;
    }
    // Compare original[0..compare_len] vs decoded[delay..delay+compare_len]
    let compare_len = (orig_len - delay).min(dec_len - delay);
    let mut signal_power = 0.0f64;
    let mut noise_power = 0.0f64;
    for i in 0..compare_len {
        let s = original[i] as f64;
        let d = decoded[delay + i] as f64;
        signal_power += s * s;
        noise_power += (s - d) * (s - d);
    }
    if noise_power == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (signal_power / noise_power).log10()
}

fn cmd_quality(wav_path: &str) {
    let wav = read_wav(Path::new(wav_path));
    let sr = wav.sample_rate as i32;
    let ch = wav.channels as i32;

    let bitrates = [16000, 32000, 64000, 128000, 256000];

    // Account for codec algorithmic delay: decoded output is shifted by
    // the encoder's lookahead. Opus has ~6.5ms total delay at 48kHz
    // (2.5ms analysis + 4ms delay compensation). We round to the nearest
    // frame boundary for simplicity.
    let delay = ((sr / 400 + sr / 250) as usize) * ch as usize;

    println!(
        "Quality (SNR) check: {} Hz, {} ch, {} samples (delay compensation: {} samples)",
        sr,
        ch,
        wav.samples.len() / ch as usize,
        delay / ch as usize
    );
    println!("{:>10} {:>10} {:>8}", "Bitrate", "SNR (dB)", "Status");

    for &br in &bitrates {
        let mut cfg = EncodeConfig::new(sr, ch);
        cfg.bitrate = br;

        // Encode and decode with Rust
        let encoded = rust_encode_cfg(&wav.samples, &cfg);
        if encoded.is_empty() {
            println!("{:>10} {:>10} {:>8}", br, "N/A", "SKIP");
            continue;
        }
        let decoded = rust_decode_cfg(&encoded, &cfg);
        if decoded.is_empty() {
            println!("{:>10} {:>10} {:>8}", br, "N/A", "SKIP");
            continue;
        }

        let snr = compute_snr(&wav.samples, &decoded, delay);

        // Minimum expected SNR: conservative sanity thresholds.
        // These detect "codec is producing garbage", not quality targets.
        // Noise and short signals will legitimately have low SNR.
        let min_snr = match br {
            b if b <= 16000 => -5.0,
            b if b <= 32000 => -2.0,
            b if b <= 64000 => 0.0,
            b if b <= 128000 => 0.0,
            _ => 0.0,
        };

        let status = if snr >= min_snr { "OK" } else { "LOW" };
        println!("{:>10} {:>9.1} {:>8}", br, snr, status);
    }
}

// ---------------------------------------------------------------------------
// Repacketizer comparison
// ---------------------------------------------------------------------------

fn cmd_repacketizer(wav_path: &str) {
    let wav = read_wav(Path::new(wav_path));
    let sr = wav.sample_rate as i32;
    let ch = wav.channels as i32;
    let cfg = EncodeConfig::new(sr, ch);
    let encoded = c_encode_cfg(&wav.samples, &cfg);
    let packets = parse_packets(&encoded);

    if packets.len() < 3 {
        eprintln!(
            "ERROR: need at least 3 encoded packets, got {}",
            packets.len()
        );
        process::exit(1);
    }

    println!(
        "Repacketizer test: {} packets from {} Hz {} ch",
        packets.len(),
        sr,
        ch
    );
    println!();

    let mut pass = 0u32;
    let mut fail = 0u32;

    // -----------------------------------------------------------------------
    // Test 1: Cat 3 packets and compare merged output
    // -----------------------------------------------------------------------
    println!("--- Test 1: cat 3 packets, compare merged output ---");
    {
        let c_out = unsafe {
            let rp = bindings::opus_repacketizer_create();
            assert!(!rp.is_null(), "C opus_repacketizer_create returned null");
            for i in 0..3 {
                let ret = bindings::opus_repacketizer_cat(
                    rp,
                    packets[i].as_ptr(),
                    packets[i].len() as i32,
                );
                assert!(
                    ret == bindings::OPUS_OK,
                    "C repacketizer_cat failed: {}",
                    ret
                );
            }
            let nb = bindings::opus_repacketizer_get_nb_frames(rp);
            println!("  C repacketizer: {} frames after cat", nb);

            let mut out = vec![0u8; 4000];
            let out_len = bindings::opus_repacketizer_out(rp, out.as_mut_ptr(), out.len() as i32);
            assert!(out_len > 0, "C repacketizer_out failed: {}", out_len);
            bindings::opus_repacketizer_destroy(rp);
            out.truncate(out_len as usize);
            out
        };

        let rust_out = {
            use mdopus::opus::repacketizer::OpusRepacketizer;
            let mut rp = OpusRepacketizer::new();
            for i in 0..3 {
                let ret = rp.cat(&packets[i], packets[i].len() as i32);
                assert!(ret == 0, "Rust repacketizer cat failed: {}", ret);
            }
            let nb = rp.get_nb_frames();
            println!("  Rust repacketizer: {} frames after cat", nb);

            let mut out = vec![0u8; 4000];
            let maxlen = out.len() as i32;
            let out_len = rp.out(&mut out, maxlen);
            assert!(out_len > 0, "Rust repacketizer out failed: {}", out_len);
            out.truncate(out_len as usize);
            out
        };

        let stats = compare_bytes(&c_out, &rust_out);
        print_result("repack_cat3_out", &stats, &c_out, &rust_out);
        if stats.first_diff_offset.is_none() {
            pass += 1;
        } else {
            fail += 1;
        }
    }

    // -----------------------------------------------------------------------
    // Test 2: out_range — extract frame 1 only from 3 catted packets
    // -----------------------------------------------------------------------
    println!();
    println!("--- Test 2: out_range(1,2) from 3 catted packets ---");
    {
        let c_out = unsafe {
            let rp = bindings::opus_repacketizer_create();
            for i in 0..3 {
                bindings::opus_repacketizer_cat(rp, packets[i].as_ptr(), packets[i].len() as i32);
            }
            let mut out = vec![0u8; 4000];
            let out_len =
                bindings::opus_repacketizer_out_range(rp, 1, 2, out.as_mut_ptr(), out.len() as i32);
            assert!(out_len > 0, "C repacketizer_out_range failed: {}", out_len);
            bindings::opus_repacketizer_destroy(rp);
            out.truncate(out_len as usize);
            out
        };

        let rust_out = {
            use mdopus::opus::repacketizer::OpusRepacketizer;
            let mut rp = OpusRepacketizer::new();
            for i in 0..3 {
                rp.cat(&packets[i], packets[i].len() as i32);
            }
            let mut out = vec![0u8; 4000];
            let maxlen = out.len() as i32;
            let out_len = rp.out_range(1, 2, &mut out, maxlen);
            assert!(
                out_len > 0,
                "Rust repacketizer out_range failed: {}",
                out_len
            );
            out.truncate(out_len as usize);
            out
        };

        let stats = compare_bytes(&c_out, &rust_out);
        print_result("repack_out_range(1,2)", &stats, &c_out, &rust_out);
        if stats.first_diff_offset.is_none() {
            pass += 1;
        } else {
            fail += 1;
        }
    }

    // -----------------------------------------------------------------------
    // Test 3: out_range(0,2) — first two frames
    // -----------------------------------------------------------------------
    println!();
    println!("--- Test 3: out_range(0,2) from 3 catted packets ---");
    {
        let c_out = unsafe {
            let rp = bindings::opus_repacketizer_create();
            for i in 0..3 {
                bindings::opus_repacketizer_cat(rp, packets[i].as_ptr(), packets[i].len() as i32);
            }
            let mut out = vec![0u8; 4000];
            let out_len =
                bindings::opus_repacketizer_out_range(rp, 0, 2, out.as_mut_ptr(), out.len() as i32);
            assert!(
                out_len > 0,
                "C repacketizer_out_range(0,2) failed: {}",
                out_len
            );
            bindings::opus_repacketizer_destroy(rp);
            out.truncate(out_len as usize);
            out
        };

        let rust_out = {
            use mdopus::opus::repacketizer::OpusRepacketizer;
            let mut rp = OpusRepacketizer::new();
            for i in 0..3 {
                rp.cat(&packets[i], packets[i].len() as i32);
            }
            let mut out = vec![0u8; 4000];
            let maxlen = out.len() as i32;
            let out_len = rp.out_range(0, 2, &mut out, maxlen);
            assert!(
                out_len > 0,
                "Rust repacketizer out_range(0,2) failed: {}",
                out_len
            );
            out.truncate(out_len as usize);
            out
        };

        let stats = compare_bytes(&c_out, &rust_out);
        print_result("repack_out_range(0,2)", &stats, &c_out, &rust_out);
        if stats.first_diff_offset.is_none() {
            pass += 1;
        } else {
            fail += 1;
        }
    }

    // -----------------------------------------------------------------------
    // Test 4: pad and unpad a single packet
    // -----------------------------------------------------------------------
    println!();
    println!("--- Test 4: pad and unpad ---");
    {
        let orig = &packets[0];
        let orig_len = orig.len() as i32;
        let padded_len = orig_len + 64;

        // C pad
        let c_padded = unsafe {
            let mut buf = vec![0u8; padded_len as usize];
            buf[..orig.len()].copy_from_slice(orig);
            let ret = bindings::opus_packet_pad(buf.as_mut_ptr(), orig_len, padded_len);
            assert!(
                ret == bindings::OPUS_OK,
                "C opus_packet_pad failed: {}",
                ret
            );
            buf
        };

        // Rust pad
        let rust_padded = {
            use mdopus::opus::repacketizer::opus_packet_pad;
            let mut buf = vec![0u8; padded_len as usize];
            buf[..orig.len()].copy_from_slice(orig);
            let ret = opus_packet_pad(&mut buf, orig_len, padded_len);
            assert!(ret == 0, "Rust opus_packet_pad failed: {}", ret);
            buf
        };

        let stats = compare_bytes(&c_padded, &rust_padded);
        print_result("pad", &stats, &c_padded, &rust_padded);
        if stats.first_diff_offset.is_none() {
            pass += 1;
        } else {
            fail += 1;
        }

        // C unpad
        let c_unpadded = unsafe {
            let mut buf = c_padded.clone();
            let ret = bindings::opus_packet_unpad(buf.as_mut_ptr(), padded_len);
            assert!(ret > 0, "C opus_packet_unpad failed: {}", ret);
            buf.truncate(ret as usize);
            buf
        };

        // Rust unpad
        let rust_unpadded = {
            use mdopus::opus::repacketizer::opus_packet_unpad;
            let mut buf = rust_padded.clone();
            let ret = opus_packet_unpad(&mut buf, padded_len);
            assert!(ret > 0, "Rust opus_packet_unpad failed: {}", ret);
            buf.truncate(ret as usize);
            buf
        };

        let stats = compare_bytes(&c_unpadded, &rust_unpadded);
        print_result("unpad", &stats, &c_unpadded, &rust_unpadded);
        if stats.first_diff_offset.is_none() {
            pass += 1;
        } else {
            fail += 1;
        }

        // Verify unpadded matches original
        let stats = compare_bytes(orig, &c_unpadded);
        print_result("unpad_vs_original", &stats, orig, &c_unpadded);
        if stats.first_diff_offset.is_none() {
            pass += 1;
        } else {
            fail += 1;
        }
    }

    // -----------------------------------------------------------------------
    // Test 5: init / reset and re-use
    // -----------------------------------------------------------------------
    println!();
    println!("--- Test 5: init/reset re-use ---");
    {
        let c_out = unsafe {
            let rp = bindings::opus_repacketizer_create();
            bindings::opus_repacketizer_cat(rp, packets[0].as_ptr(), packets[0].len() as i32);
            bindings::opus_repacketizer_init(rp);
            bindings::opus_repacketizer_cat(rp, packets[1].as_ptr(), packets[1].len() as i32);
            bindings::opus_repacketizer_cat(rp, packets[2].as_ptr(), packets[2].len() as i32);
            let mut out = vec![0u8; 4000];
            let out_len = bindings::opus_repacketizer_out(rp, out.as_mut_ptr(), out.len() as i32);
            assert!(
                out_len > 0,
                "C repacketizer_out after init failed: {}",
                out_len
            );
            bindings::opus_repacketizer_destroy(rp);
            out.truncate(out_len as usize);
            out
        };

        let rust_out = {
            use mdopus::opus::repacketizer::OpusRepacketizer;
            let mut rp = OpusRepacketizer::new();
            rp.cat(&packets[0], packets[0].len() as i32);
            rp.init();
            rp.cat(&packets[1], packets[1].len() as i32);
            rp.cat(&packets[2], packets[2].len() as i32);
            let mut out = vec![0u8; 4000];
            let maxlen = out.len() as i32;
            let out_len = rp.out(&mut out, maxlen);
            assert!(
                out_len > 0,
                "Rust repacketizer out after init failed: {}",
                out_len
            );
            out.truncate(out_len as usize);
            out
        };

        let stats = compare_bytes(&c_out, &rust_out);
        print_result("repack_init_reuse", &stats, &c_out, &rust_out);
        if stats.first_diff_offset.is_none() {
            pass += 1;
        } else {
            fail += 1;
        }
    }

    // -----------------------------------------------------------------------
    // Test 6: get_nb_frames parity
    // -----------------------------------------------------------------------
    println!();
    println!("--- Test 6: get_nb_frames parity ---");
    {
        for count in 1..=3.min(packets.len()) {
            let c_nb = unsafe {
                let rp = bindings::opus_repacketizer_create();
                for i in 0..count {
                    bindings::opus_repacketizer_cat(
                        rp,
                        packets[i].as_ptr(),
                        packets[i].len() as i32,
                    );
                }
                let nb = bindings::opus_repacketizer_get_nb_frames(rp);
                bindings::opus_repacketizer_destroy(rp);
                nb
            };

            let r_nb = {
                use mdopus::opus::repacketizer::OpusRepacketizer;
                let mut rp = OpusRepacketizer::new();
                for i in 0..count {
                    rp.cat(&packets[i], packets[i].len() as i32);
                }
                rp.get_nb_frames()
            };

            if c_nb == r_nb {
                println!(
                    "  PASS  nb_frames after {} cat: C={}, Rust={}",
                    count, c_nb, r_nb
                );
                pass += 1;
            } else {
                println!(
                    "  FAIL  nb_frames after {} cat: C={}, Rust={}",
                    count, c_nb, r_nb
                );
                fail += 1;
            }
        }
    }

    println!();
    println!("========================================");
    println!("repacketizer: {} PASS, {} FAIL", pass, fail);
    println!("========================================");
    if fail > 0 {
        process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Multistream comparison
// ---------------------------------------------------------------------------

fn cmd_multistream() {
    println!("Multistream test: stereo (mapping family 0)");
    println!();

    let sr = 48000i32;
    let channels = 2i32;
    let streams = 1i32;
    let coupled = 1i32;
    let mapping = [0u8, 1u8];
    let application = bindings::OPUS_APPLICATION_AUDIO;
    let frame_size = 960; // 20ms at 48kHz
    let bitrate = 64000i32;

    let pcm = generate_noise(sr, channels, 0.5, 42);
    let num_frames = pcm.len() / (frame_size * channels as usize);

    println!(
        "  {} samples, {} frames of {} at {} Hz {} ch",
        pcm.len() / channels as usize,
        num_frames,
        frame_size,
        sr,
        channels
    );
    println!();

    let mut pass = 0u32;
    let mut fail = 0u32;

    // -----------------------------------------------------------------------
    // Test 1: Encode with both C and Rust, compare packets
    // -----------------------------------------------------------------------
    println!("--- Test 1: multistream encode comparison ---");

    let c_packets: Vec<Vec<u8>> = unsafe {
        let mut err: i32 = 0;
        let enc = bindings::opus_multistream_encoder_create(
            sr,
            channels,
            streams,
            coupled,
            mapping.as_ptr(),
            application,
            &mut err,
        );
        assert!(
            !enc.is_null() && err == bindings::OPUS_OK,
            "C multistream_encoder_create failed: {}",
            err
        );

        bindings::opus_multistream_encoder_ctl(enc, bindings::OPUS_SET_BITRATE_REQUEST, bitrate);
        bindings::opus_multistream_encoder_ctl(enc, bindings::OPUS_SET_COMPLEXITY_REQUEST, 10i32);
        bindings::opus_multistream_encoder_ctl(enc, bindings::OPUS_SET_VBR_REQUEST, 0i32);

        let mut pkts = Vec::new();
        let mut pos = 0;
        let samples_per_frame = frame_size * channels as usize;
        let mut out = vec![0u8; 4000];
        while pos + samples_per_frame <= pcm.len() {
            let ret = bindings::opus_multistream_encode(
                enc,
                pcm[pos..].as_ptr(),
                frame_size as i32,
                out.as_mut_ptr(),
                out.len() as i32,
            );
            assert!(ret > 0, "C multistream_encode failed: {}", ret);
            pkts.push(out[..ret as usize].to_vec());
            pos += samples_per_frame;
        }
        bindings::opus_multistream_encoder_destroy(enc);
        pkts
    };

    let rust_packets: Vec<Vec<u8>> = {
        use mdopus::opus::multistream::OpusMSEncoder;
        let mut enc = OpusMSEncoder::new(sr, channels, streams, coupled, &mapping, application)
            .expect("Rust OpusMSEncoder::new failed");

        enc.set_bitrate(bitrate);
        enc.set_complexity(10);
        enc.set_vbr(0);

        let mut pkts = Vec::new();
        let mut pos = 0;
        let samples_per_frame = frame_size * channels as usize;
        let mut out = vec![0u8; 4000];
        while pos + samples_per_frame <= pcm.len() {
            let maxlen = out.len() as i32;
            let ret = enc
                .encode(
                    &pcm[pos..pos + samples_per_frame],
                    frame_size as i32,
                    &mut out,
                    maxlen,
                )
                .expect("Rust multistream encode failed");
            pkts.push(out[..ret as usize].to_vec());
            pos += samples_per_frame;
        }
        pkts
    };

    println!(
        "  C encoded {} packets, Rust encoded {} packets",
        c_packets.len(),
        rust_packets.len()
    );
    {
        let mut all_match = true;
        let frame_count = c_packets.len().min(rust_packets.len());
        for i in 0..frame_count {
            let stats = compare_bytes(&c_packets[i], &rust_packets[i]);
            if stats.first_diff_offset.is_some() {
                println!(
                    "  Frame {}: FAIL (C {} bytes, Rust {} bytes, first_diff @{})",
                    i,
                    c_packets[i].len(),
                    rust_packets[i].len(),
                    stats.first_diff_offset.unwrap()
                );
                all_match = false;
            }
        }
        if c_packets.len() != rust_packets.len() {
            all_match = false;
        }
        if all_match {
            println!(
                "  ms_encode: PASS ({} packets, all byte-exact)",
                frame_count
            );
            pass += 1;
        } else {
            println!("  ms_encode: FAIL");
            fail += 1;
        }
    }

    // -----------------------------------------------------------------------
    // Test 2: Decode C-encoded packets with both C and Rust, compare PCM
    // -----------------------------------------------------------------------
    println!();
    println!("--- Test 2: multistream decode comparison (C-encoded packets) ---");

    let c_decoded: Vec<i16> = unsafe {
        let mut err: i32 = 0;
        let dec = bindings::opus_multistream_decoder_create(
            sr,
            channels,
            streams,
            coupled,
            mapping.as_ptr(),
            &mut err,
        );
        assert!(
            !dec.is_null() && err == bindings::OPUS_OK,
            "C multistream_decoder_create failed: {}",
            err
        );

        let mut output = Vec::new();
        let mut pcm_buf = vec![0i16; frame_size * channels as usize];
        for pkt in &c_packets {
            let ret = bindings::opus_multistream_decode(
                dec,
                pkt.as_ptr(),
                pkt.len() as i32,
                pcm_buf.as_mut_ptr(),
                frame_size as i32,
                0,
            );
            assert!(ret > 0, "C multistream_decode failed: {}", ret);
            output.extend_from_slice(&pcm_buf[..ret as usize * channels as usize]);
        }
        bindings::opus_multistream_decoder_destroy(dec);
        output
    };

    let rust_decoded: Vec<i16> = {
        use mdopus::opus::multistream::OpusMSDecoder;
        let mut dec = OpusMSDecoder::new(sr, channels, streams, coupled, &mapping)
            .expect("Rust OpusMSDecoder::new failed");

        let mut output = Vec::new();
        let mut pcm_buf = vec![0i16; frame_size * channels as usize];
        for pkt in &c_packets {
            let ret = dec
                .decode(
                    Some(pkt.as_slice()),
                    pkt.len() as i32,
                    &mut pcm_buf,
                    frame_size as i32,
                    false,
                )
                .expect("Rust multistream decode failed");
            output.extend_from_slice(&pcm_buf[..ret as usize * channels as usize]);
        }
        output
    };

    println!(
        "  C decoded {} samples, Rust decoded {} samples",
        c_decoded.len() / channels as usize,
        rust_decoded.len() / channels as usize
    );
    {
        let stats = compare_samples(&c_decoded, &rust_decoded);
        print_sample_result("ms_decode", &stats, &c_decoded, &rust_decoded);
        if stats.first_diff_offset.is_none() {
            pass += 1;
        } else {
            fail += 1;
        }
    }

    // -----------------------------------------------------------------------
    // Test 3: Full roundtrip — encode with Rust, decode with Rust, vs C roundtrip
    // -----------------------------------------------------------------------
    println!();
    println!("--- Test 3: multistream roundtrip (C enc+dec vs Rust enc+dec) ---");

    let c_roundtrip: Vec<i16> = unsafe {
        let mut err: i32 = 0;
        let dec = bindings::opus_multistream_decoder_create(
            sr,
            channels,
            streams,
            coupled,
            mapping.as_ptr(),
            &mut err,
        );
        assert!(!dec.is_null() && err == bindings::OPUS_OK);

        let mut output = Vec::new();
        let mut pcm_buf = vec![0i16; frame_size * channels as usize];
        for pkt in &c_packets {
            let ret = bindings::opus_multistream_decode(
                dec,
                pkt.as_ptr(),
                pkt.len() as i32,
                pcm_buf.as_mut_ptr(),
                frame_size as i32,
                0,
            );
            if ret > 0 {
                output.extend_from_slice(&pcm_buf[..ret as usize * channels as usize]);
            }
        }
        bindings::opus_multistream_decoder_destroy(dec);
        output
    };

    let rust_roundtrip: Vec<i16> = {
        use mdopus::opus::multistream::OpusMSDecoder;
        let mut dec = OpusMSDecoder::new(sr, channels, streams, coupled, &mapping)
            .expect("Rust OpusMSDecoder::new failed");

        let mut output = Vec::new();
        let mut pcm_buf = vec![0i16; frame_size * channels as usize];
        for pkt in &rust_packets {
            match dec.decode(
                Some(pkt.as_slice()),
                pkt.len() as i32,
                &mut pcm_buf,
                frame_size as i32,
                false,
            ) {
                Ok(ret) => {
                    output.extend_from_slice(&pcm_buf[..ret as usize * channels as usize]);
                }
                Err(e) => {
                    eprintln!(
                        "  WARNING: Rust multistream decode failed on Rust-encoded packet: {}",
                        e
                    );
                    break;
                }
            }
        }
        output
    };

    println!(
        "  C roundtrip {} samples, Rust roundtrip {} samples",
        c_roundtrip.len() / channels as usize,
        rust_roundtrip.len() / channels as usize
    );
    {
        let stats = compare_samples(&c_roundtrip, &rust_roundtrip);
        print_sample_result("ms_roundtrip", &stats, &c_roundtrip, &rust_roundtrip);
        if stats.first_diff_offset.is_none() {
            pass += 1;
        } else {
            fail += 1;
        }
    }

    println!();
    println!("========================================");
    println!("multistream: {} PASS, {} FAIL", pass, fail);
    println!("========================================");
    if fail > 0 {
        process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Test-all: sequential full test suite
// ---------------------------------------------------------------------------

fn cmd_test_all() {
    println!("=== mdopus full test suite ===");
    println!();

    // Each command calls process::exit(1) on failure, so if any fails
    // the remaining tests won't run. This is acceptable for a sequential
    // test runner -- fix failures before running the full suite.

    println!("--- [1/11] api ---");
    cmd_api();
    println!();

    println!("--- [2/11] sweep ---");
    cmd_sweep(None, false);
    println!();

    println!("--- [3/11] plc (48k mono noise) ---");
    cmd_plc("tests/vectors/48000hz_mono_noise.wav", 64000);
    println!();

    println!("--- [4/11] plc (8k mono noise) ---");
    cmd_plc("tests/vectors/8000hz_mono_noise.wav", 12000);
    println!();

    println!("--- [5/11] fec (48k mono noise) ---");
    cmd_fec("tests/vectors/48000hz_mono_noise.wav", 64000, 20);
    println!();

    println!("--- [6/11] dtx (generate) ---");
    cmd_dtx("generate", 24000);
    println!();

    println!("--- [7/11] packets (48k mono noise) ---");
    cmd_packets("tests/vectors/48000hz_mono_noise.wav");
    println!();

    println!("--- [8/11] decode-formats (48k mono noise) ---");
    cmd_decode_formats("tests/vectors/48000hz_mono_noise.wav");
    println!();

    println!("--- [9/11] repacketizer (48k mono noise) ---");
    cmd_repacketizer("tests/vectors/48000hz_mono_noise.wav");
    println!();

    println!("--- [10/11] multistream ---");
    cmd_multistream();
    println!();

    println!("--- [11/11] longsoak (5s) ---");
    cmd_longsoak(5, 48000);
    println!();

    // Quality is a sanity check that doesn't fail on LOW results,
    // so run it last as informational.
    println!("--- [bonus] quality (48k mono sine440) ---");
    cmd_quality("tests/vectors/48000hz_mono_sine440.wav");
    println!();

    println!("========================================");
    println!("=== ALL TESTS PASSED ===");
    println!("========================================");
}

#[cfg(test)]
mod coverage_smoke_tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn harness_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn smoke_core_harness_commands() {
        let _guard = harness_lock().lock().unwrap();

        cmd_api();
        cmd_packets("tests/vectors/48000hz_mono_noise.wav");
        cmd_decode_formats("tests/vectors/48000hz_mono_noise.wav");
        cmd_repacketizer("tests/vectors/48000hz_mono_noise.wav");
    }

    #[test]
    fn smoke_resilience_and_multistream_commands() {
        let _guard = harness_lock().lock().unwrap();

        let wav = read_wav(Path::new("tests/vectors/48000hz_mono_noise.wav"));
        let sr = wav.sample_rate as i32;
        let ch = wav.channels as i32;
        let frame_size = (sr / 50) as usize;

        let mut plc_cfg = EncodeConfig::new(sr, ch);
        plc_cfg.bitrate = 64000;
        let plc_packets = parse_packets(&c_encode_cfg(&wav.samples, &plc_cfg));
        let plc_output = plc_decode_rust(sr, ch, frame_size, &plc_packets, &[5, 15, 25]);
        assert!(
            !plc_output.is_empty(),
            "PLC smoke path should conceal dropped packets"
        );

        let mut fec_cfg = EncodeConfig::new(sr, ch);
        fec_cfg.bitrate = 64000;
        fec_cfg.fec = 1;
        fec_cfg.packet_loss_pct = 20;
        let fec_packets = parse_packets(&c_encode_cfg(&wav.samples, &fec_cfg));
        let fec_output = fec_decode_rust(sr, ch, frame_size, &fec_packets, &[5, 15, 25]);
        assert!(
            !fec_output.is_empty(),
            "FEC smoke path should recover dropped packets"
        );

        let dtx_signal = generate_dtx_signal(16000, 1, 1.0, 42);
        let mut dtx_cfg = EncodeConfig::new(16000, 1);
        dtx_cfg.bitrate = 24000;
        dtx_cfg.application = bindings::OPUS_APPLICATION_VOIP;
        dtx_cfg.dtx = 1;
        let dtx_encoded = rust_encode_cfg(&dtx_signal, &dtx_cfg);
        assert!(
            !dtx_encoded.is_empty(),
            "DTX smoke path should produce encoded output"
        );
        let dtx_decoded = rust_decode_cfg(&dtx_encoded, &dtx_cfg);
        assert!(
            !dtx_decoded.is_empty(),
            "DTX smoke path should decode output"
        );

        cmd_multistream();
        cmd_quality("tests/vectors/48000hz_mono_sine440.wav");
    }

    #[test]
    fn smoke_targeted_sweep_matrix() {
        let _guard = harness_lock().lock().unwrap();

        for filter in [
            "SILK 8k/1ch CBR 12kbps 20ms cx10 VOIP",
            "SILK 8k/1ch CBR 24kbps 20ms cx10 VOIP FEC+5%loss",
            "Hybrid 16k/1ch CBR 32kbps 20ms cx10 AUDIO",
            "Hybrid 16k/1ch CVBR 24kbps 20ms cx10 AUDIO",
            "CELT 48k/1ch CBR 64kbps 20ms cx10 AUDIO",
            "CELT 48k/1ch CBR 64kbps 20ms cx10 VOIP DTX",
            "CELT 48k/2ch CBR 128kbps 20ms cx10 AUDIO forcestereo",
            "CELT 48k/1ch CBR 64kbps 2.5ms cx10 LOWDELAY",
            // Additional for coverage:
            "SILK 12k/1ch CBR 16kbps 20ms cx10 VOIP", // mediumband
            "SILK 16k/2ch CBR 32kbps 20ms cx10 VOIP", // stereo WB SILK
            "SILK 16k/1ch CVBR 24kbps 40ms cx10 VOIP", // 40ms frames
            "SILK 16k/1ch CBR 24kbps 60ms cx10 VOIP", // 60ms frames
            "SILK 8k/2ch CBR 24kbps 20ms cx10 VOIP",  // stereo NB
            "SILK 16k/1ch CBR 32kbps 20ms cx10 VOIP FEC+25%loss", // high loss FEC
            "Hybrid 48k/2ch CBR 64kbps 20ms cx10 AUDIO", // hybrid stereo
            "Hybrid 48k/1ch CVBR 48kbps 20ms cx10 AUDIO", // hybrid CVBR
            "CELT 48k/1ch CVBR 32kbps 20ms cx10 AUDIO", // CELT CVBR low
            "CELT 48k/2ch CVBR 96kbps 20ms cx10 AUDIO", // CELT stereo CVBR
            "CELT 48k/1ch CBR 64kbps 10ms cx10 LOWDELAY", // 10ms lowdelay
            "CELT 48k/2ch CBR 128kbps 5ms cx10 LOWDELAY", // stereo lowdelay
        ] {
            cmd_sweep(Some(filter), true);
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn usage() {
    eprintln!("mdopus-compare: compare C reference opus vs Rust implementation");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  mdopus-compare encode <input.wav> [--bitrate N] [--complexity N]");
    eprintln!("  mdopus-compare decode <input.opus>");
    eprintln!("  mdopus-compare roundtrip <input.wav> [--bitrate N]");
    eprintln!("  mdopus-compare plc <input.wav> [--bitrate N]");
    eprintln!("  mdopus-compare fec <input.wav> [--bitrate N] [--loss-pct N]");
    eprintln!("  mdopus-compare dtx <input.wav|generate> [--bitrate N]");
    eprintln!("  mdopus-compare sweep [filter] [--stop-on-fail]");
    eprintln!("  mdopus-compare bench <input.wav> [--bitrate N] [--complexity N] [--iters N]");
    eprintln!("  mdopus-compare longsoak [--duration N] [--sample-rate N]");
    eprintln!(
        "  mdopus-compare torture [--duration N] [--seed N] [--change-interval N] [--sample-rate N] [--channels N]"
    );
    eprintln!("  mdopus-compare transitions");
    eprintln!("  mdopus-compare quality <input.wav>");
    eprintln!("  mdopus-compare packets <input.wav>");
    eprintln!("  mdopus-compare decode-formats <input.wav>");
    eprintln!("  mdopus-compare repacketizer <input.wav>");
    eprintln!("  mdopus-compare multistream");
    eprintln!("  mdopus-compare unit <module_name>");
    eprintln!("  mdopus-compare api");
    eprintln!("  mdopus-compare test-all");
    eprintln!();
    eprintln!("MODULES for 'unit':");
    eprintln!("  range_coder   Range coder (entenc/entdec)");
    eprintln!("  all           Run all module tests");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("  --bitrate N      Target bitrate in bps (default: 64000)");
    eprintln!("  --complexity N   Encoder complexity 0-10 (default: 10)");
    eprintln!("  --iters N        Benchmark iterations (default: 10)");
    eprintln!("  --loss-pct N     Packet loss percentage for FEC (default: 20)");
    eprintln!("  --duration N     Longsoak duration in seconds (default: 30)");
    eprintln!("  --sample-rate N  Sample rate for longsoak (default: 48000)");
    eprintln!("  --stop-on-fail   (sweep) Stop after first failing configuration");
    eprintln!();
    eprintln!("SWEEP FILTER EXAMPLES:");
    eprintln!("  mdopus-compare sweep CELT          Only CELT configurations");
    eprintln!("  mdopus-compare sweep SILK          Only SILK configurations");
    eprintln!("  mdopus-compare sweep stereo        Only stereo (2ch) configurations");
    eprintln!("  mdopus-compare sweep VBR           Only VBR configurations");
}

fn parse_option(args: &[String], flag: &str, default: i32) -> i32 {
    for i in 0..args.len() {
        if args[i] == flag {
            if i + 1 < args.len() {
                return args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("ERROR: invalid value for {}", flag);
                    process::exit(1);
                });
            }
        }
    }
    default
}

fn cmd_hp_cutoff_compare() {
    // Compare C and Rust hp_cutoff for stereo data
    // Use the same noise generator as the sweep tests
    let pcm = generate_noise(48000, 2, 0.02, 42); // 20ms of stereo 48kHz noise
    let len = 960; // 48000 * 0.02 = 960 samples per channel
    let cutoff_hz = 60; // actual initial cutoff for CELT mode
    let fs = 48000;

    // C hp_cutoff
    let mut c_out = vec![0i16; pcm.len()];
    let mut c_hp_mem = [0i32; 4];
    unsafe {
        bindings::debug_c_hp_cutoff_stereo(
            pcm.as_ptr(),
            cutoff_hz,
            c_out.as_mut_ptr(),
            c_hp_mem.as_mut_ptr(),
            len,
            fs,
        );
    }

    // Rust hp_cutoff
    use mdopus::opus::encoder::hp_cutoff_debug;
    let mut r_out = vec![0i16; pcm.len()];
    let mut r_hp_mem = [0i32; 4];
    hp_cutoff_debug(
        &pcm,
        cutoff_hz,
        &mut r_out,
        &mut r_hp_mem,
        len as usize,
        2,
        fs,
    );

    // Compare
    let mut ndiff = 0;
    let mut first_diff = None;
    for i in 0..pcm.len() {
        if c_out[i] != r_out[i] {
            if ndiff < 10 {
                println!(
                    "  sample[{}]: C={} R={} diff={}",
                    i,
                    c_out[i],
                    r_out[i],
                    (c_out[i] as i32 - r_out[i] as i32).abs()
                );
            }
            if first_diff.is_none() {
                first_diff = Some(i);
            }
            ndiff += 1;
        }
    }

    if ndiff == 0 {
        println!("hp_cutoff stereo: MATCH ({} samples compared)", pcm.len());
    } else {
        println!(
            "hp_cutoff stereo: {} samples DIFFER (first at {}, total={})",
            ndiff,
            first_diff.unwrap_or(0),
            pcm.len()
        );
    }

    // Also compare hp_mem state
    if c_hp_mem == r_hp_mem {
        println!("hp_mem state: MATCH");
    } else {
        println!("hp_mem state: DIFFER");
        for i in 0..4 {
            println!("  hp_mem[{}]: C={} R={}", i, c_hp_mem[i], r_hp_mem[i]);
        }
    }
}

fn cmd_debug_decode_25ms() {
    // Debug: decode 2.5ms frames at 48k/1ch/64kbps AUDIO and compare per-frame
    use mdopus::opus::decoder::OpusDecoder as RustDecoder;

    let sr = 48000;
    let ch = 1;
    let bitrate = 64000;
    let frame_ms = 2.5f64;
    let frame_size = (sr as f64 * frame_ms / 1000.0) as usize; // 120

    // Generate noise input
    let pcm = generate_noise(sr, ch, 0.1, 42); // 100ms

    // Encode with C reference (using AUDIO mode)
    let mut cfg = EncodeConfig::new(sr, ch);
    cfg.bitrate = bitrate;
    cfg.frame_ms = frame_ms;
    cfg.application = bindings::OPUS_APPLICATION_AUDIO;
    let encoded = c_encode_cfg(&pcm, &cfg);

    println!(
        "Encoded {} bytes from {} samples (frame_size={})",
        encoded.len(),
        pcm.len(),
        frame_size
    );

    // Create C decoder
    let c_dec = unsafe {
        let mut error: i32 = 0;
        let dec = bindings::opus_decoder_create(sr, ch, &mut error);
        assert!(!dec.is_null() && error == bindings::OPUS_OK);
        dec
    };

    // Create Rust decoder
    let mut rust_dec = RustDecoder::new(sr, ch).expect("Rust decoder init");

    let mut c_pcm = vec![0i16; frame_size * ch as usize];
    let mut rust_pcm = vec![0i16; frame_size * ch as usize];
    let mut pos = 0;
    let mut frame_idx = 0;

    while pos + 2 <= encoded.len() {
        let pkt_len = u16::from_le_bytes([encoded[pos], encoded[pos + 1]]) as usize;
        pos += 2;
        if pos + pkt_len > encoded.len() {
            break;
        }
        let pkt = &encoded[pos..pos + pkt_len];

        // C decode
        let c_ret = unsafe {
            bindings::opus_decode(
                c_dec,
                pkt.as_ptr(),
                pkt_len as i32,
                c_pcm.as_mut_ptr(),
                frame_size as i32,
                0,
            )
        };
        assert!(c_ret >= 0);

        // Get C final range
        let mut c_range: u32 = 0;
        unsafe {
            bindings::opus_decoder_ctl(c_dec, 4031i32, &mut c_range as *mut u32);
        }

        // Rust decode
        let rust_ret = rust_dec
            .decode(Some(pkt), &mut rust_pcm, frame_size as i32, false)
            .expect("Rust decode");
        let rust_range = rust_dec.get_final_range();

        // Compare
        let count = (c_ret as usize).min(rust_ret as usize) * ch as usize;
        let mut max_diff = 0i32;
        let mut first_diff: Option<usize> = None;
        let mut ndiff = 0usize;
        for i in 0..count {
            let d = (c_pcm[i] as i32 - rust_pcm[i] as i32).abs();
            if d != 0 {
                ndiff += 1;
                if first_diff.is_none() {
                    first_diff = Some(i);
                }
                if d > max_diff {
                    max_diff = d;
                }
            }
        }

        // Compare C vs Rust internal state
        if frame_idx < 3 {
            // C old_band_e
            let mut c_band_e = [0i32; 42];
            unsafe {
                bindings::debug_get_celt_old_band_e(c_dec, c_band_e.as_mut_ptr(), 42);
            }
            let r_band_e = rust_dec.debug_get_old_band_e();
            let mut be_diff = false;
            for i in 0..42.min(r_band_e.len()) {
                if c_band_e[i] != r_band_e[i] {
                    be_diff = true;
                    break;
                }
            }
            if be_diff {
                println!("  old_band_e DIFFER after frame {}:", frame_idx);
                for i in 0..21.min(r_band_e.len()) {
                    if c_band_e[i] != r_band_e[i] {
                        println!("    band[{}]: C={} R={}", i, c_band_e[i], r_band_e[i]);
                    }
                }
            } else {
                println!("  old_band_e: MATCH after frame {}", frame_idx);
            }

            // Compare decode_mem overlap region after frame 0
            // DECODE_BUFFER_SIZE=2048, overlap=120
            // out_syn_off = 2048 - 120 = 1928
            // MDCT writes to [1928..2168] (240 elements)
            // After frame, buffer shifts left by 120: old [1928..2168] -> [1808..2048]
            // So after shift, the "future overlap" is at [1928..2048] (overlaps with new out_syn start)
            // But that's the current frame's output. The new overlap is what the NEXT frame will see.
            // After the shift, decode_mem[1928..2048] = old [2048..2168] = overlap tail from this frame.
            // Let's read positions 2048..2168 BEFORE the shift (which already happened).
            // Actually the shift already happened. So the "overlap tail" is now at positions
            // 1928..2048 of the new buffer state.
            // More precisely: after frame decode, the buffer was shifted.
            // new[0..2048-120+120] = old[120..2168] = old[120..2168]
            // So new[1928..2048] = old[2048..2168] = the MDCT's [120..240] output region
            // Compare the full MDCT output region: [out_syn_off..out_syn_off+240]
            // out_syn_off = DECODE_BUFFER_SIZE - N = 2048 - 120 = 1928
            // But after the buffer shift (which happens before MDCT), the MDCT writes to [1928..2168].
            // Let's read the full MDCT region.
            let overlap_start = 1928usize;
            let overlap_count = 240usize;
            let mut c_overlap = vec![0i32; overlap_count];
            unsafe {
                bindings::debug_get_celt_decode_mem(
                    c_dec,
                    overlap_start as i32,
                    overlap_count as i32,
                    c_overlap.as_mut_ptr(),
                );
            }
            let r_overlap = rust_dec.debug_get_decode_mem(overlap_start, overlap_count);
            let mut ov_ndiff = 0;
            let mut ov_max_diff = 0i32;
            for i in 0..overlap_count {
                let d = (c_overlap[i] - r_overlap[i]).abs();
                if d != 0 {
                    ov_ndiff += 1;
                    if d > ov_max_diff {
                        ov_max_diff = d;
                    }
                }
            }
            if ov_ndiff > 0 {
                println!(
                    "  decode_mem overlap [{}-{}]: {} diffs, max_diff={}",
                    overlap_start,
                    overlap_start + overlap_count,
                    ov_ndiff,
                    ov_max_diff
                );
                let mut shown = 0;
                for i in 0..overlap_count {
                    let d = c_overlap[i] - r_overlap[i];
                    if d != 0 && shown < 20 {
                        println!(
                            "    mem[{}]: C={} R={} (diff={})",
                            overlap_start + i,
                            c_overlap[i],
                            r_overlap[i],
                            d
                        );
                        shown += 1;
                    }
                }
            } else {
                println!(
                    "  decode_mem overlap [{}-{}]: MATCH",
                    overlap_start,
                    overlap_start + overlap_count
                );
            }
        }

        if ndiff == 0 {
            println!(
                "Frame {:3}: {} bytes, {} samples - MATCH (range: C={:#010x} R={:#010x})",
                frame_idx, pkt_len, c_ret, c_range, rust_range
            );
        } else {
            println!(
                "Frame {:3}: {} bytes, {} samples - DIFFER: {} diffs, first at sample {}, max_diff={} (range: C={:#010x} R={:#010x})",
                frame_idx,
                pkt_len,
                c_ret,
                ndiff,
                first_diff.unwrap(),
                max_diff,
                c_range,
                rust_range
            );
            // Print first 10 differing samples
            let mut shown = 0;
            for i in 0..count {
                let d = (c_pcm[i] as i32 - rust_pcm[i] as i32).abs();
                if d != 0 && shown < 10 {
                    println!(
                        "  sample[{}]: C={} R={} (diff={})",
                        i,
                        c_pcm[i],
                        rust_pcm[i],
                        c_pcm[i] as i32 - rust_pcm[i] as i32
                    );
                    shown += 1;
                }
            }
        }

        pos += pkt_len;
        frame_idx += 1;
    }

    unsafe {
        bindings::opus_decoder_destroy(c_dec);
    }
}

fn cmd_debug_mdct_compare() {
    use mdopus::celt::mdct::{MDCT_48000_960, clt_mdct_backward};
    use mdopus::celt::modes::MODE_48000_960_120;

    let mode = &MODE_48000_960_120;
    let l = &MDCT_48000_960;
    let overlap = 120i32;

    // Test with shift=3 (the problematic case: LM=0, 2.5ms frames)
    let shift = 3;
    let stride = 1;
    let n_mdct = 240; // 1920 >> 3
    let n2 = 120; // N/2

    // Generate a non-trivial frequency-domain input
    let mut freq = vec![0i32; n2];
    for i in 0..n2 {
        freq[i] = ((i as i32 * 31337 + 12345) % 65536) - 32768;
        freq[i] <<= 8; // Scale up to realistic signal levels
    }

    // Test 1: zero overlap (like frame 0)
    {
        let overlap_buf = vec![0i32; overlap as usize];
        let mut c_out = vec![0i32; n_mdct];
        let mut r_out = vec![0i32; n_mdct];

        // C MDCT backward
        unsafe {
            bindings::debug_clt_mdct_backward(
                freq.as_ptr(),
                c_out.as_mut_ptr(),
                overlap_buf.as_ptr(),
                overlap,
                shift,
                stride,
                n_mdct as i32,
            );
        }

        // Rust MDCT backward
        for i in 0..overlap as usize {
            r_out[i] = overlap_buf[i];
        }
        clt_mdct_backward(l, &freq, &mut r_out, mode.window, overlap, shift, stride);

        // Compare ALL positions (not just 0..120)
        let mut ndiff = 0;
        let mut max_diff = 0i32;
        for i in 0..n_mdct {
            let d = (c_out[i] - r_out[i]).abs();
            if d != 0 {
                ndiff += 1;
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
        println!(
            "Test 1 (zero overlap, shift=3): {} diffs out of {}, max_diff={}",
            ndiff, n_mdct, max_diff
        );
        if ndiff > 0 {
            let mut shown = 0;
            for i in 0..n_mdct {
                let d = c_out[i] - r_out[i];
                if d != 0 && shown < 20 {
                    println!("  out[{}]: C={} R={} (diff={})", i, c_out[i], r_out[i], d);
                    shown += 1;
                }
            }
        }
    }

    // Test 2: non-zero overlap (like frame 1)
    {
        // Use the post-rotation tail from Test 1 as the overlap for Test 2
        let mut overlap_buf = vec![0i32; overlap as usize];
        // First run Rust MDCT to get frame 0's output, then extract overlap region
        let mut frame0_out = vec![0i32; n_mdct];
        clt_mdct_backward(
            l,
            &freq,
            &mut frame0_out,
            mode.window,
            overlap,
            shift,
            stride,
        );
        // The "future overlap" is positions [120..180]
        for i in 0..60 {
            overlap_buf[i] = frame0_out[120 + i];
        }
        // positions 60..120 would be zeros (as they would be in actual decode buffer)

        // Generate different freq data for frame 1
        let mut freq2 = vec![0i32; n2];
        for i in 0..n2 {
            freq2[i] = ((i as i32 * 54321 + 98765) % 65536) - 32768;
            freq2[i] <<= 8;
        }

        let mut c_out = vec![0i32; n_mdct];
        let mut r_out = vec![0i32; n_mdct];

        // C MDCT backward with overlap
        unsafe {
            bindings::debug_clt_mdct_backward(
                freq2.as_ptr(),
                c_out.as_mut_ptr(),
                overlap_buf.as_ptr(),
                overlap,
                shift,
                stride,
                n_mdct as i32,
            );
        }

        // Rust MDCT backward with overlap
        for i in 0..overlap as usize {
            r_out[i] = overlap_buf[i];
        }
        clt_mdct_backward(l, &freq2, &mut r_out, mode.window, overlap, shift, stride);

        // Compare ALL positions
        let mut ndiff = 0;
        let mut max_diff = 0i32;
        for i in 0..n_mdct {
            let d = (c_out[i] - r_out[i]).abs();
            if d != 0 {
                ndiff += 1;
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
        println!(
            "Test 2 (non-zero overlap, shift=3): {} diffs out of {}, max_diff={}",
            ndiff, n_mdct, max_diff
        );
        if ndiff > 0 {
            let mut shown = 0;
            for i in 0..n_mdct {
                let d = c_out[i] - r_out[i];
                if d != 0 && shown < 20 {
                    println!("  out[{}]: C={} R={} (diff={})", i, c_out[i], r_out[i], d);
                    shown += 1;
                }
            }
        }
    }

    // Test 3: shift=0 (the working case) with non-zero overlap
    {
        let shift0 = 0;
        let n_mdct0 = 1920;
        let n2_0 = 960;

        let mut freq0 = vec![0i32; n2_0];
        for i in 0..n2_0 {
            freq0[i] = ((i as i32 * 31337 + 12345) % 65536) - 32768;
            freq0[i] <<= 8;
        }

        let overlap_buf = vec![12345i32; overlap as usize]; // non-zero overlap

        let mut c_out = vec![0i32; n_mdct0];
        let mut r_out = vec![0i32; n_mdct0];

        // C MDCT backward
        unsafe {
            bindings::debug_clt_mdct_backward(
                freq0.as_ptr(),
                c_out.as_mut_ptr(),
                overlap_buf.as_ptr(),
                overlap,
                shift0,
                stride,
                n_mdct0 as i32,
            );
        }

        // Rust MDCT backward
        for i in 0..overlap as usize {
            r_out[i] = overlap_buf[i];
        }
        clt_mdct_backward(l, &freq0, &mut r_out, mode.window, overlap, shift0, stride);

        let mut ndiff = 0;
        let mut max_diff = 0i32;
        for i in 0..n_mdct0 {
            let d = (c_out[i] - r_out[i]).abs();
            if d != 0 {
                ndiff += 1;
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
        println!(
            "Test 3 (non-zero overlap, shift=0): {} diffs out of {}, max_diff={}",
            ndiff, n_mdct0, max_diff
        );
        if ndiff > 0 {
            let mut shown = 0;
            for i in 0..n_mdct0 {
                let d = c_out[i] - r_out[i];
                if d != 0 && shown < 20 {
                    println!("  out[{}]: C={} R={} (diff={})", i, c_out[i], r_out[i], d);
                    shown += 1;
                }
            }
        }
    }
}

fn cmd_debug_stereo_voip() {
    use mdopus::opus::encoder::{OpusEncoder as RustEncoder, hp_cutoff_debug};

    // First: test HP cutoff at 8kHz stereo
    {
        let wav_path = resolve_wav_path(8000, 2);
        let wav = read_wav(Path::new(&wav_path));
        let test_pcm = &wav.samples[..640]; // 160 samples * 2 channels = 320 interleaved
        let len = 160;
        let cutoff_hz = 60;
        let fs = 8000;

        let mut c_out = vec![0i16; test_pcm.len()];
        let mut c_hp_mem = [0i32; 4];
        unsafe {
            bindings::debug_c_hp_cutoff_stereo(
                test_pcm.as_ptr(),
                cutoff_hz,
                c_out.as_mut_ptr(),
                c_hp_mem.as_mut_ptr(),
                len,
                fs,
            );
        }

        let mut r_out = vec![0i16; test_pcm.len()];
        let mut r_hp_mem = [0i32; 4];
        hp_cutoff_debug(
            test_pcm,
            cutoff_hz,
            &mut r_out,
            &mut r_hp_mem,
            len as usize,
            2,
            fs,
        );

        let ndiff = c_out
            .iter()
            .zip(r_out.iter())
            .filter(|(a, b)| a != b)
            .count();
        if ndiff == 0 {
            println!("HP cutoff 8kHz stereo: MATCH ({} samples)", test_pcm.len());
        } else {
            println!("HP cutoff 8kHz stereo: {} DIFFER", ndiff);
            for i in 0..test_pcm.len().min(20) {
                if c_out[i] != r_out[i] {
                    println!("  sample[{}]: C={} R={}", i, c_out[i], r_out[i]);
                }
            }
        }
        if c_hp_mem != r_hp_mem {
            println!("HP mem 8kHz: DIFFER C={:?} R={:?}", c_hp_mem, r_hp_mem);
        } else {
            println!("HP mem 8kHz: MATCH");
        }
    }

    // Test LR_to_MS directly
    {
        use mdopus::silk::encoder::{StereoEncState, silk_stereo_lr_to_ms};
        let fl = 80usize; // 8kHz * 10ms = 80 samples (internal frame)
        let fs_khz = 8;
        // Use some non-trivial input data
        let x1: Vec<i16> = (0..fl)
            .map(|i| ((i as i32 * 1234 + 5678) % 20000 - 10000) as i16)
            .collect();
        let x2: Vec<i16> = (0..fl)
            .map(|i| ((i as i32 * 4321 + 8765) % 20000 - 10000) as i16)
            .collect();

        // C version
        let mut c_mid = vec![0i16; fl + 2];
        let mut c_side = vec![0i16; fl + 2];
        let mut c_pred_ix = [0i8; 6];
        let mut c_mid_only = 0i8;
        let mut c_rates = [0i32; 2];
        unsafe {
            bindings::debug_c_stereo_lr_to_ms(
                x1.as_ptr(),
                x2.as_ptr(),
                c_mid.as_mut_ptr(),
                c_side.as_mut_ptr(),
                c_pred_ix.as_mut_ptr(),
                &mut c_mid_only,
                c_rates.as_mut_ptr(),
                24000,
                0,
                0,
                fs_khz,
                fl as i32,
            );
        }

        // Rust version
        let mut r_state = StereoEncState::default();
        let mut r_x1 = x1.clone();
        let mut r_x2 = x2.clone();
        let mut r_pred_ix = [[0i8; 3]; 2];
        let mut r_mid_only = 0i8;
        let mut r_rates = [0i32; 2];
        silk_stereo_lr_to_ms(
            &mut r_state,
            &mut r_x1,
            &mut r_x2,
            &mut r_pred_ix,
            &mut r_mid_only,
            &mut r_rates,
            24000,
            0,
            false,
            fs_khz,
            fl,
        );

        // C returns mid in c_mid[0..fl+2] (inputBuf layout) and side in c_side[0..fl+1]
        // The mid signal at inputBuf[2..fl+2] should match Rust x1[0..fl]
        // The side signal at c_side[1..fl+1] should match Rust x2[0..fl]
        let c_mid_signal = &c_mid[2..fl + 2];
        let c_side_signal = &c_side[1..fl + 1];

        let mid_ndiff = c_mid_signal
            .iter()
            .zip(r_x1.iter())
            .filter(|(a, b)| a != b)
            .count();
        let side_ndiff = c_side_signal
            .iter()
            .zip(r_x2.iter())
            .filter(|(a, b)| a != b)
            .count();

        println!("\nLR_to_MS comparison (fl={}, fs_kHz={}):", fl, fs_khz);
        if mid_ndiff == 0 {
            println!("  Mid signal: MATCH ({} samples)", fl);
        } else {
            println!("  Mid signal: {} DIFFER", mid_ndiff);
            let mut shown = 0;
            for i in 0..fl {
                if c_mid_signal[i] != r_x1[i] && shown < 10 {
                    println!("    mid[{}]: C={} R={}", i, c_mid_signal[i], r_x1[i]);
                    shown += 1;
                }
            }
        }
        if side_ndiff == 0 {
            println!("  Side signal: MATCH ({} samples)", fl);
        } else {
            println!("  Side signal: {} DIFFER", side_ndiff);
            let mut shown = 0;
            for i in 0..fl {
                if c_side_signal[i] != r_x2[i] && shown < 10 {
                    println!("    side[{}]: C={} R={}", i, c_side_signal[i], r_x2[i]);
                    shown += 1;
                }
            }
        }
        println!(
            "  C overlap: mid[0..2]={:?} side[0..2]={:?}",
            &c_mid[0..2],
            &c_side[0..2]
        );
        println!(
            "  R state: s_mid={:?} s_side={:?}",
            r_state.s_mid, r_state.s_side
        );
        println!("  C pred_ix={:?} mid_only={}", c_pred_ix, c_mid_only);
        println!("  R pred_ix={:?} mid_only={}", r_pred_ix, r_mid_only);
        println!("  C rates={:?}", c_rates);
        println!("  R rates={:?}", r_rates);
    }

    let sr = 8000i32;
    let ch = 2i32;
    let bitrate = 24000i32;
    let frame_ms = 20.0f64;
    let frame_size = (sr as f64 * frame_ms / 1000.0) as usize;
    let samples_per_frame = frame_size * ch as usize;

    let wav_path = resolve_wav_path(sr, ch);
    let wav = read_wav(Path::new(&wav_path));
    let pcm = wav.samples;

    println!(
        "\nDebug stereo VOIP: {}Hz {}ch {}bps frame_size={}",
        sr, ch, bitrate, frame_size
    );
    println!("Total PCM samples: {}", pcm.len());

    let c_enc = unsafe {
        let mut error: i32 = 0;
        let enc =
            bindings::opus_encoder_create(sr, ch, bindings::OPUS_APPLICATION_VOIP, &mut error);
        assert!(!enc.is_null() && error == bindings::OPUS_OK);
        bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_BITRATE_REQUEST, bitrate);
        bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_COMPLEXITY_REQUEST, 10);
        enc
    };

    let mut r_enc = RustEncoder::new(sr, ch, bindings::OPUS_APPLICATION_VOIP).unwrap();
    r_enc.set_bitrate(bitrate);
    r_enc.set_complexity(10);

    let max_packet = 4000;
    let mut c_pkt = vec![0u8; max_packet];
    let mut r_pkt = vec![0u8; max_packet];

    let mut pos = 0;
    let mut frame_idx = 0;
    while pos + samples_per_frame <= pcm.len() && frame_idx < 50 {
        let frame_pcm = &pcm[pos..pos + samples_per_frame];

        let c_ret = unsafe {
            bindings::opus_encode(
                c_enc,
                frame_pcm.as_ptr(),
                frame_size as i32,
                c_pkt.as_mut_ptr(),
                max_packet as i32,
            )
        };

        let r_ret = r_enc
            .encode(frame_pcm, frame_size as i32, &mut r_pkt, max_packet as i32)
            .unwrap_or_else(|e| {
                eprintln!("Rust encode error: {}", e);
                0
            });

        let c_bytes = &c_pkt[..c_ret as usize];
        let r_bytes = &r_pkt[..r_ret as usize];

        // Get encoder state
        let r_mode = r_enc.get_mode();
        let r_stream_ch = r_enc.get_stream_channels();
        let r_bw = r_enc.get_bandwidth();
        let r_range = r_enc.get_final_range();
        let c_range = unsafe {
            let mut v: u32 = 0;
            bindings::opus_encoder_ctl(
                c_enc,
                bindings::OPUS_GET_FINAL_RANGE_REQUEST,
                &mut v as *mut u32,
            );
            v
        };
        let c_bw = unsafe {
            let mut v: i32 = 0;
            bindings::opus_encoder_ctl(
                c_enc,
                bindings::OPUS_GET_BANDWIDTH_REQUEST,
                &mut v as *mut i32,
            );
            v
        };

        if c_bytes == r_bytes {
            println!(
                "  frame {}: MATCH ({} bytes) mode={} sch={} bw=C{}/R{}",
                frame_idx, c_ret, r_mode, r_stream_ch, c_bw, r_bw
            );
        } else {
            let ndiff = c_bytes
                .iter()
                .zip(r_bytes.iter())
                .filter(|(a, b)| a != b)
                .count();
            println!(
                "  frame {}: DIFFER c_len={} r_len={} byte_diffs={} mode={} sch={} bw=C{}/R{}",
                frame_idx, c_ret, r_ret, ndiff, r_mode, r_stream_ch, c_bw, r_bw
            );
            println!("    TOC: C=0x{:02x} R=0x{:02x}", c_bytes[0], r_bytes[0]);
            println!("    range: C=0x{:08x} R=0x{:08x}", c_range, r_range);
            let min_len = (c_ret as usize).min(r_ret as usize);
            let mut shown = 0;
            for i in 1..min_len {
                if c_bytes[i] != r_bytes[i] && shown < 5 {
                    println!(
                        "    byte[{}]: C=0x{:02x} R=0x{:02x}",
                        i, c_bytes[i], r_bytes[i]
                    );
                    shown += 1;
                }
            }
            let r_hp_smth2 = r_enc.get_variable_hp_smth2();
            println!("    Rust hp_smth2={}", r_hp_smth2);
            break;
        }

        pos += samples_per_frame;
        frame_idx += 1;
    }

    unsafe {
        bindings::opus_encoder_destroy(c_enc);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage();
        process::exit(1);
    }

    println!("C reference opus: {}", bindings::version_string());
    println!();

    match args[1].as_str() {
        "encode" => {
            if args.len() < 3 {
                eprintln!("ERROR: encode requires an input WAV file");
                process::exit(1);
            }
            let bitrate = parse_option(&args, "--bitrate", 64000);
            let complexity = parse_option(&args, "--complexity", 10);
            cmd_encode(&args[2], bitrate, complexity);
        }
        "decode" => {
            if args.len() < 3 {
                eprintln!("ERROR: decode requires an input file");
                process::exit(1);
            }
            cmd_decode(&args[2]);
        }
        "roundtrip" => {
            if args.len() < 3 {
                eprintln!("ERROR: roundtrip requires an input WAV file");
                process::exit(1);
            }
            let bitrate = parse_option(&args, "--bitrate", 64000);
            cmd_roundtrip(&args[2], bitrate);
        }
        "framecompare" => {
            if args.len() < 3 {
                eprintln!("ERROR: framecompare requires an input WAV file");
                process::exit(1);
            }
            let bitrate = parse_option(&args, "--bitrate", 64000);
            let complexity = parse_option(&args, "--complexity", 10);
            let application =
                parse_option(&args, "--application", bindings::OPUS_APPLICATION_AUDIO);
            let signal = parse_option(&args, "--signal", -1000);
            cmd_encode_framecompare(&args[2], bitrate, complexity, application, signal);
        }
        "unit" => {
            if args.len() < 3 {
                eprintln!("ERROR: unit requires a module name");
                process::exit(1);
            }
            let pass = cmd_unit(&args[2]);
            if !pass {
                process::exit(1);
            }
        }
        "mathcompare" => {
            cmd_mathcompare();
        }
        "rngtest" => {
            cmd_rng_test();
        }
        "bench" => {
            if args.len() < 3 {
                eprintln!("ERROR: bench requires an input WAV file");
                process::exit(1);
            }
            let bitrate = parse_option(&args, "--bitrate", 64000);
            let complexity = parse_option(&args, "--complexity", 10);
            let iters = parse_option(&args, "--iters", 10) as u32;
            cmd_bench(&args[2], bitrate, complexity, iters);
        }
        "decodecompare" => {
            if args.len() < 3 {
                eprintln!("ERROR: decodecompare requires an input WAV file");
                process::exit(1);
            }
            let bitrate = parse_option(&args, "--bitrate", 64000);
            let application =
                parse_option(&args, "--application", bindings::OPUS_APPLICATION_AUDIO);
            cmd_decode_framecompare(&args[2], bitrate, application);
        }
        "plc" => {
            if args.len() < 3 {
                eprintln!("ERROR: plc requires an input WAV file");
                process::exit(1);
            }
            let bitrate = parse_option(&args, "--bitrate", 64000);
            cmd_plc(&args[2], bitrate);
        }
        "fec" => {
            if args.len() < 3 {
                eprintln!("ERROR: fec requires an input WAV file");
                process::exit(1);
            }
            let bitrate = parse_option(&args, "--bitrate", 64000);
            let loss_pct = parse_option(&args, "--loss-pct", 20);
            cmd_fec(&args[2], bitrate, loss_pct);
        }
        "dtx" => {
            if args.len() < 3 {
                eprintln!("ERROR: dtx requires an input WAV file or 'generate'");
                process::exit(1);
            }
            let bitrate = parse_option(&args, "--bitrate", 24000);
            cmd_dtx(&args[2], bitrate);
        }
        "sweep" => {
            let stop_on_fail = args.iter().any(|a| a == "--stop-on-fail");
            // Find filter: first positional arg after "sweep" that isn't a flag
            let filter = args.get(2).and_then(|s| {
                if s.starts_with("--") {
                    None
                } else {
                    Some(s.as_str())
                }
            });
            cmd_sweep(filter, stop_on_fail);
        }
        "api" => {
            cmd_api();
        }
        "packets" => {
            if args.len() < 3 {
                eprintln!("ERROR: packets requires an input WAV file");
                process::exit(1);
            }
            cmd_packets(&args[2]);
        }
        "decode-formats" => {
            if args.len() < 3 {
                eprintln!("ERROR: decode-formats requires an input WAV file");
                process::exit(1);
            }
            cmd_decode_formats(&args[2]);
        }
        "longsoak" => {
            let duration = parse_option(&args, "--duration", 30);
            let sr = parse_option(&args, "--sample-rate", 48000);
            cmd_longsoak(duration, sr);
        }
        "torture" => {
            let duration = parse_option(&args, "--duration", 1800); // 30 min default
            let seed_i = parse_option(&args, "--seed", 42);
            let interval = parse_option(&args, "--change-interval", 50);
            let sr = parse_option(&args, "--sample-rate", 48000);
            let ch = parse_option(&args, "--channels", 1);
            // Section 2: --cross-decode enables the 2x2 decoder matrix
            // Section 3: --burst-interval controls the deterministic burst
            //            scheduler (0 disables; default 200 = ~4s between
            //            bursts at 20ms frames).
            // Section 6: --state-check-interval enables periodic state
            //            comparison (default 1000 frames ≈ 20s).
            //            --state-check-loose downgrades state mismatches to
            //            warnings. Strict-by-default per HLD Decision 5.
            let cross_decode = args.iter().any(|a| a == "--cross-decode");
            let burst_interval = parse_option(&args, "--burst-interval", 200) as usize;
            let state_check_interval = parse_option(&args, "--state-check-interval", 1000) as usize;
            let state_check_strict = !args.iter().any(|a| a == "--state-check-loose");
            cmd_torture(
                duration,
                seed_i as u64,
                interval as usize,
                sr,
                ch,
                cross_decode,
                burst_interval,
                state_check_interval,
                state_check_strict,
            );
        }
        "transitions" => {
            // Section 2 (Decision 3): cross-decoding is also wired into
            // cmd_transitions because that command is exactly where
            // encoder divergence is most likely to hide.
            let cross_decode = args.iter().any(|a| a == "--cross-decode");
            cmd_transitions(cross_decode);
        }
        "quality" => {
            if args.len() < 3 {
                eprintln!("ERROR: quality requires an input WAV file");
                process::exit(1);
            }
            cmd_quality(&args[2]);
        }
        "repacketizer" => {
            if args.len() < 3 {
                eprintln!("ERROR: repacketizer requires an input WAV file");
                process::exit(1);
            }
            cmd_repacketizer(&args[2]);
        }
        "multistream" => {
            cmd_multistream();
        }
        "hp-compare" => {
            cmd_hp_cutoff_compare();
        }
        "debug-decode-25ms" => {
            cmd_debug_decode_25ms();
        }
        "debug-stereo-voip" => {
            cmd_debug_stereo_voip();
        }
        "debug-mdct-compare" => {
            cmd_debug_mdct_compare();
        }
        "test-all" => {
            cmd_test_all();
        }
        "--help" | "-h" | "help" => {
            usage();
        }
        other => {
            eprintln!("ERROR: unknown command '{}'", other);
            usage();
            process::exit(1);
        }
    }
}
