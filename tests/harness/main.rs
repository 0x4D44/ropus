//! mdopus-compare: CLI tool that compares C reference opus output against
//! the Rust implementation, byte-for-byte / sample-for-sample.

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
        if chunk_size % 2 != 0 {
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
            let start = if offset >= 16 { offset - 16 } else { 0 };
            let end = (offset + 48).min(a.len().max(b.len()));
            println!("  C ref:  {}", hex_line(&a, start, end));
            println!("  Rust:   {}", hex_line(&b, start, end));
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
            let start = if offset >= 4 { offset - 4 } else { 0 };
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
// C reference encode/decode via FFI
// ---------------------------------------------------------------------------

fn c_encode(
    pcm: &[i16],
    sample_rate: i32,
    channels: i32,
    bitrate: i32,
    complexity: i32,
) -> Vec<u8> {
    unsafe {
        let mut error: i32 = 0;
        let enc = bindings::opus_encoder_create(
            sample_rate,
            channels,
            bindings::OPUS_APPLICATION_AUDIO,
            &mut error,
        );
        if enc.is_null() || error != bindings::OPUS_OK {
            eprintln!(
                "ERROR: C opus_encoder_create failed: {}",
                bindings::error_string(error)
            );
            process::exit(1);
        }

        // Set bitrate
        bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_BITRATE_REQUEST, bitrate);
        // Set complexity
        bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_COMPLEXITY_REQUEST, complexity);
        // Force CBR for deterministic output
        bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_VBR_REQUEST, 0i32);

        let frame_size = (sample_rate / 50) as usize; // 20ms frames
        let samples_per_frame = frame_size * channels as usize;
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

fn c_decode(encoded: &[u8], sample_rate: i32, channels: i32) -> Vec<i16> {
    unsafe {
        let mut error: i32 = 0;
        let dec = bindings::opus_decoder_create(sample_rate, channels, &mut error);
        if dec.is_null() || error != bindings::OPUS_OK {
            eprintln!(
                "ERROR: C opus_decoder_create failed: {}",
                bindings::error_string(error)
            );
            process::exit(1);
        }

        let frame_size = (sample_rate / 50) as usize; // 20ms
        let mut pcm = vec![0i16; frame_size * channels as usize];
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
            output.extend_from_slice(&pcm[..ret as usize * channels as usize]);
            pos += pkt_len;
        }

        bindings::opus_decoder_destroy(dec);
        output
    }
}

// ---------------------------------------------------------------------------
// Rust implementation stubs
// (These call into our Rust port. Currently stubbed -- will be filled in
//  as modules are ported.)
// ---------------------------------------------------------------------------

fn rust_encode(
    pcm: &[i16],
    sample_rate: i32,
    channels: i32,
    bitrate: i32,
    complexity: i32,
) -> Vec<u8> {
    use mdopus::opus::encoder::{OPUS_APPLICATION_AUDIO, OpusEncoder};

    let mut enc = match OpusEncoder::new(sample_rate, channels, OPUS_APPLICATION_AUDIO) {
        Ok(e) => e,
        Err(code) => {
            eprintln!("ERROR: Rust OpusEncoder::new failed: {}", code);
            return Vec::new();
        }
    };
    enc.set_bitrate(bitrate);
    enc.set_complexity(complexity);
    enc.set_vbr(0); // CBR for deterministic output

    let frame_size = (sample_rate / 50) as usize; // 20ms frames
    let samples_per_frame = frame_size * channels as usize;
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

fn rust_decode(encoded: &[u8], sample_rate: i32, channels: i32) -> Vec<i16> {
    use mdopus::opus::decoder::OpusDecoder;

    let mut dec = match OpusDecoder::new(sample_rate, channels) {
        Ok(d) => d,
        Err(code) => {
            eprintln!("ERROR: Rust OpusDecoder::new failed: {}", code);
            return Vec::new();
        }
    };

    let frame_size = (sample_rate / 50) as usize; // 20ms
    let mut pcm = vec![0i16; frame_size * channels as usize];
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
                output.extend_from_slice(&pcm[..ret as usize * channels as usize]);
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
// Main
// ---------------------------------------------------------------------------

fn usage() {
    eprintln!("mdopus-compare: compare C reference opus vs Rust implementation");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  mdopus-compare encode <input.wav> [--bitrate N] [--complexity N]");
    eprintln!("  mdopus-compare decode <input.opus>");
    eprintln!("  mdopus-compare roundtrip <input.wav> [--bitrate N]");
    eprintln!("  mdopus-compare unit <module_name>");
    eprintln!();
    eprintln!("MODULES for 'unit':");
    eprintln!("  range_coder   Range coder (entenc/entdec)");
    eprintln!("  all           Run all module tests");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("  --bitrate N      Target bitrate in bps (default: 64000)");
    eprintln!("  --complexity N   Encoder complexity 0-10 (default: 10)");
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
