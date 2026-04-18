//! Reproduce fuzz_roundtrip crash: compare C vs Rust encoder for specific configs.
//!
//! Usage: cargo run --release --bin repro-fuzz-roundtrip [crash_file]

#![allow(
    clippy::needless_range_loop,
    clippy::unnecessary_cast,
    clippy::collapsible_if,
    clippy::identity_op
)]

#[path = "../tests/harness/bindings.rs"]
mod bindings;

use std::os::raw::c_int;

use mdopus::opus::encoder::{
    OpusEncoder as RustEncoder, OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY,
    OPUS_APPLICATION_VOIP,
};

const SAMPLE_RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];
const APPLICATIONS: [i32; 3] = [OPUS_APPLICATION_VOIP, OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY];

fn byte_to_bitrate(b0: u8, b1: u8) -> i32 {
    let raw = u16::from_le_bytes([b0, b1]) as i32;
    6000 + (raw % 504001)
}

fn compare_encode(
    sample_rate: i32,
    channels: i32,
    application: i32,
    bitrate: i32,
    complexity: i32,
    pcm: &[i16],
    frame_size: i32,
) {
    println!("Config: sr={sample_rate}, ch={channels}, app={application}, br={bitrate}, cx={complexity}, fs={frame_size}");

    // Rust encode
    let mut rust_enc = RustEncoder::new(sample_rate, channels, application).unwrap();
    rust_enc.set_bitrate(bitrate);
    rust_enc.set_vbr(0);
    rust_enc.set_complexity(complexity);

    let mut rust_out = vec![0u8; 4000];
    let rust_len = match rust_enc.encode(pcm, frame_size, &mut rust_out, 4000) {
        Ok(l) => l as usize,
        Err(e) => {
            println!("  Rust encode FAILED: {e}");
            return;
        }
    };
    println!("  Rust: {rust_len} bytes, TOC=0x{:02X}", rust_out[0]);

    // C encode
    unsafe {
        let mut error: c_int = 0;
        let c_enc = bindings::opus_encoder_create(sample_rate, channels, application, &mut error);
        if c_enc.is_null() || error != 0 {
            println!("  C encoder create FAILED: {error}");
            return;
        }
        bindings::opus_encoder_ctl(c_enc, bindings::OPUS_SET_BITRATE_REQUEST, bitrate);
        bindings::opus_encoder_ctl(c_enc, bindings::OPUS_SET_VBR_REQUEST, 0 as c_int);
        bindings::opus_encoder_ctl(c_enc, bindings::OPUS_SET_COMPLEXITY_REQUEST, complexity);

        let mut c_out = vec![0u8; 4000];
        let c_len = bindings::opus_encode(
            c_enc,
            pcm.as_ptr(),
            frame_size,
            c_out.as_mut_ptr(),
            4000,
        );
        bindings::opus_encoder_destroy(c_enc);

        if c_len < 0 {
            println!("  C encode FAILED: {c_len}");
            return;
        }
        let c_len = c_len as usize;
        println!("  C:    {c_len} bytes, TOC=0x{:02X}", c_out[0]);

        if rust_len != c_len || rust_out[..rust_len] != c_out[..c_len] {
            println!("  *** DIVERGENCE ***");
            // Decode TOC
            decode_toc(rust_out[0], "Rust");
            decode_toc(c_out[0], "C");
            // Show first bytes
            let show = rust_len.min(c_len).min(32);
            println!("  Rust bytes[0..{show}]: {:02X?}", &rust_out[..show]);
            println!("  C    bytes[0..{show}]: {:02X?}", &c_out[..show]);
            // Find first difference
            for i in 0..rust_len.min(c_len) {
                if rust_out[i] != c_out[i] {
                    println!("  First diff at byte {i}: Rust=0x{:02X} C=0x{:02X}", rust_out[i], c_out[i]);
                    break;
                }
            }
        } else {
            println!("  MATCH");
        }
    }
}

fn decode_toc(toc: u8, label: &str) {
    let stereo = (toc & 0x04) != 0;
    let config = (toc >> 3) & 0x1F;
    let code = toc & 0x03;

    // Mode and bandwidth from config
    let (mode, bw, frame_dur) = if config < 12 {
        // SILK-only
        let bw = match config / 4 {
            0 => "NB",
            1 => "MB",
            2 => "WB",
            _ => "?",
        };
        let dur = match config % 4 {
            0 => "10ms",
            1 => "20ms",
            2 => "40ms",
            3 => "60ms",
            _ => "?",
        };
        ("SILK", bw, dur)
    } else if config < 16 {
        // Hybrid
        let bw = if config < 14 { "SWB" } else { "FB" };
        let dur = if config % 2 == 0 { "10ms" } else { "20ms" };
        ("Hybrid", bw, dur)
    } else {
        // CELT-only
        let bw = match (config - 16) / 4 {
            0 => "NB",
            1 => "WB",
            2 => "SWB",
            3 => "FB",
            _ => "?",
        };
        let dur = match (config - 16) % 4 {
            0 => "2.5ms",
            1 => "5ms",
            2 => "10ms",
            3 => "20ms",
            _ => "?",
        };
        ("CELT", bw, dur)
    };
    println!("  {label} TOC: {mode} {bw} {frame_dur} {}{} (code={code})",
        if stereo { "stereo" } else { "mono" },
        if code > 0 { format!(" code={code}") } else { String::new() }
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 {
        // Load crash file
        let data = std::fs::read(&args[1]).expect("cannot read crash file");
        println!("Crash file: {} ({} bytes)", args[1], data.len());

        if data.len() < 6 + 320 {
            println!("Too short for fuzz_roundtrip format");
            return;
        }

        let sample_rate = SAMPLE_RATES[(data[0] as usize) % SAMPLE_RATES.len()];
        let channels = if data[1] & 1 == 0 { 1 } else { 2 };
        let application = APPLICATIONS[(data[2] as usize) % APPLICATIONS.len()];
        let bitrate = byte_to_bitrate(data[3], data[4]);
        let complexity = (data[5] as i32) % 11;
        let pcm_bytes = &data[6..];
        let frame_size = sample_rate / 50;
        let samples_needed = frame_size as usize * channels as usize;
        let bytes_needed = samples_needed * 2;

        println!("Parsed: sr={sample_rate} ch={channels} app={application} br={bitrate} cx={complexity} frame_size={frame_size}");

        if pcm_bytes.len() < bytes_needed {
            println!("Not enough PCM: need {bytes_needed}, have {}", pcm_bytes.len());
            return;
        }

        let pcm: Vec<i16> = pcm_bytes[..bytes_needed]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        compare_encode(sample_rate, channels, application, bitrate, complexity, &pcm, frame_size);
    }

    // Also test the known divergence case from the assertion message
    println!("\n=== Known divergence case ===");
    let sr = 24000;
    let ch = 2;
    let app = 2048; // AUDIO
    let br = 71279;
    let cx = 3;
    let frame_size = sr / 50; // 480
    let samples = frame_size as usize * ch as usize;
    let pcm = vec![0i16; samples];
    compare_encode(sr, ch, app, br, cx, &pcm, frame_size);

    // Test the captured crash file config
    println!("\n=== Captured crash config ===");
    let sr = 12000;
    let ch = 1;
    let app = 2048; // AUDIO
    let br = 10000;
    let cx = 3;
    let frame_size = sr / 50; // 240
    let samples = frame_size as usize * ch as usize;
    let pcm = vec![0i16; samples];
    compare_encode(sr, ch, app, br, cx, &pcm, frame_size);

    // Sweep nearby bitrates to find edge cases
    println!("\n=== Bitrate sweep at 24kHz stereo AUDIO cx=3 ===");
    for br in (5000..100000).step_by(1000) {
        let sr = 24000;
        let ch = 2;
        let frame_size = sr / 50;
        let samples = frame_size as usize * ch as usize;
        let pcm = vec![0i16; samples];
        compare_encode(sr, ch, 2048, br, 3, &pcm, frame_size);
    }
}
