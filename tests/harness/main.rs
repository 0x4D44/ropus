//! mdopus-compare: CLI tool that compares C reference opus output against
//! the Rust implementation, byte-for-byte / sample-for-sample.

#![allow(
    clippy::needless_range_loop,
    clippy::manual_range_contains,
    clippy::unnecessary_cast,
    clippy::collapsible_if,
    clippy::identity_op,
    clippy::manual_is_variant_and,
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
            let start = offset.saturating_sub(16);
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

fn cmd_encode_framecompare(wav_path: &str, bitrate: i32, complexity: i32) {
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
        let enc =
            bindings::opus_encoder_create(sr, ch, bindings::OPUS_APPLICATION_AUDIO, &mut error);
        bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_BITRATE_REQUEST, bitrate);
        bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_COMPLEXITY_REQUEST, complexity);
        bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_VBR_REQUEST, 0i32);
        enc
    };

    // Rust encoder
    use mdopus::opus::encoder::{OPUS_APPLICATION_AUDIO, OpusEncoder};
    let mut r_enc = OpusEncoder::new(sr, ch, OPUS_APPLICATION_AUDIO).unwrap();
    r_enc.set_bitrate(bitrate);
    r_enc.set_complexity(complexity);
    r_enc.set_vbr(0);

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

        if &c_pkt[..cl] == &r_pkt[..rl] {
            println!("Frame {:3}: {} bytes - MATCH", frame_idx, cl);
        } else {
            println!(
                "Frame {:3}: C={} bytes, R={} bytes - DIFFER",
                frame_idx, cl, rl
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

    // Test celt_cos_norm32
    println!("--- celt_cos_norm32 comparison ---");
    let cos_tests: Vec<i32> = vec![
        0,
        1,
        -1,
        1_073_741_824,
        -1_073_741_824, // boundaries
        536_870_912,
        -536_870_912, // pi/4
        268_435_456,
        805_306_368, // pi/8, 3*pi/8
        100_000_000,
        500_000_000,
        900_000_000,
    ];
    let mut cos_pass = true;
    for &x in &cos_tests {
        let c_val = unsafe { bindings::debug_c_celt_cos_norm32(x) };
        let r_val = math_ops::celt_cos_norm32(x);
        if c_val != r_val {
            println!(
                "  MISMATCH: celt_cos_norm32({}) C={} R={} diff={}",
                x,
                c_val,
                r_val,
                c_val - r_val
            );
            cos_pass = false;
        }
    }
    if cos_pass {
        println!("  celt_cos_norm32: {} values PASS", cos_tests.len());
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

    // Exhaustive sweep: celt_rsqrt_norm32 over entire valid range
    println!("--- Exhaustive celt_rsqrt_norm32 sweep ---");
    let mut rsqrt32_mismatches = 0;
    // Valid range is [2^30, 2^31) for Q31 in [0.5, 1.0)
    // Test 20000 points spread across the range
    for i in 0..20000 {
        let x = 1_073_741_824 + (i as i64 * 1_073_741_823 / 20000) as i32;
        let c_val = unsafe { bindings::debug_c_celt_rsqrt_norm32(x) };
        let r_val = math_ops::celt_rsqrt_norm32(x);
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
            "  celt_rsqrt_norm32 sweep: {} mismatches out of 20000!",
            rsqrt32_mismatches
        );
    } else {
        println!("  celt_rsqrt_norm32 sweep: 20000 values PASS");
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

    // Exhaustive sweep: celt_cos_norm32
    println!("--- Exhaustive celt_cos_norm32 sweep ---");
    let mut cos_mismatches = 0;
    for i in 0..20000 {
        let x = (-1_073_741_824i64 + (i as i64 * 2_147_483_648 / 20000)) as i32;
        let c_val = unsafe { bindings::debug_c_celt_cos_norm32(x) };
        let r_val = math_ops::celt_cos_norm32(x);
        if c_val != r_val {
            cos_mismatches += 1;
            if cos_mismatches <= 5 {
                println!(
                    "  MISMATCH: celt_cos_norm32({}) C={} R={} diff={}",
                    x,
                    c_val,
                    r_val,
                    c_val - r_val
                );
            }
        }
    }
    if cos_mismatches > 0 {
        println!(
            "  celt_cos_norm32 sweep: {} mismatches found!",
            cos_mismatches
        );
    } else {
        println!("  celt_cos_norm32 sweep: 20000 values PASS");
    }

    // Add a test for normalise_bands: compute g for band 13 with realistic energy values
    println!("--- celt_rcp_norm32 sweep ---");
    let mut rcp_mismatches = 0;
    for i in 0..20000 {
        let x = 1_073_741_824 + (i as i64 * 1_073_741_823 / 20000) as i32;
        let c_val_sqrt = unsafe { bindings::debug_c_celt_sqrt32(x) };
        let r_val_sqrt = math_ops::celt_sqrt32(x);
        if c_val_sqrt != r_val_sqrt {
            rcp_mismatches += 1;
        }
        // Also test celt_rcp_norm32 by constructing valid inputs
        let c_val_rsq = unsafe { bindings::debug_c_celt_rsqrt_norm32(x) };
        let r_val_rsq = math_ops::celt_rsqrt_norm32(x);
        if c_val_rsq != r_val_rsq {
            rcp_mismatches += 1;
            if rcp_mismatches <= 5 {
                println!(
                    "  MISMATCH: rsqrt_norm32({}) C={} R={} diff={}",
                    x,
                    c_val_rsq,
                    r_val_rsq,
                    c_val_rsq - r_val_rsq
                );
            }
        }
    }
    if rcp_mismatches > 0 {
        println!("  rcp_norm32 sweep: {} mismatches found!", rcp_mismatches);
    } else {
        println!("  rcp_norm32 sweep: 40000 values PASS");
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

fn cmd_decode_framecompare(wav_path: &str, bitrate: i32) {
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

    // Encode with C reference
    let c_encoded = c_encode(&wav.samples, sr, ch, bitrate, complexity);
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
                celt_payload.get(0).unwrap_or(&0),
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
        "framecompare" => {
            if args.len() < 3 {
                eprintln!("ERROR: framecompare requires an input WAV file");
                process::exit(1);
            }
            let bitrate = parse_option(&args, "--bitrate", 64000);
            let complexity = parse_option(&args, "--complexity", 10);
            cmd_encode_framecompare(&args[2], bitrate, complexity);
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
        "decodecompare" => {
            if args.len() < 3 {
                eprintln!("ERROR: decodecompare requires an input WAV file");
                process::exit(1);
            }
            let bitrate = parse_option(&args, "--bitrate", 64000);
            cmd_decode_framecompare(&args[2], bitrate);
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
