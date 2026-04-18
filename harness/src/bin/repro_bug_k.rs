//! Reproducer for Campaign 8 Bug K (differential encode divergence).
//!
//! Input: `fuzz_crashes/crash_b7c3ad5aeeffb545.bin`
//! Expected: frame 2 diverges between Rust and C encoders at 12 kHz stereo VOIP.
//!
//! Usage: cargo run --release --bin repro_bug_k [path]

#![allow(clippy::needless_range_loop, clippy::identity_op)]

#[path = "../bindings.rs"]
mod bindings;

use std::os::raw::c_int;

use ropus::opus::encoder::{
    OpusEncoder as RustEncoder, OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY,
    OPUS_APPLICATION_VOIP,
};

const SAMPLE_RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];
const APPLICATIONS: [i32; 3] = [
    OPUS_APPLICATION_VOIP,
    OPUS_APPLICATION_AUDIO,
    OPUS_APPLICATION_RESTRICTED_LOWDELAY,
];

fn byte_to_bitrate(b0: u8, b1: u8) -> i32 {
    let raw = u16::from_le_bytes([b0, b1]) as i32;
    6000 + (raw % 504001)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "fuzz_crashes/crash_b7c3ad5aeeffb545.bin".to_string());
    let data = std::fs::read(&path).expect("cannot read crash file");
    println!("Loaded {} bytes from {path}", data.len());

    assert!(data.len() >= 7 + 5 * 320, "too small for harness");

    let sample_rate = SAMPLE_RATES[(data[0] as usize) % SAMPLE_RATES.len()];
    let channels = if data[1] & 1 == 0 { 1i32 } else { 2 };
    let application = APPLICATIONS[(data[2] as usize) % APPLICATIONS.len()];
    let bitrate = byte_to_bitrate(data[3], data[4]);
    let complexity = (data[5] as i32) % 11;
    let num_frames = 5 + ((data[6] as usize) % 6);
    let pcm_bytes = &data[7..];

    let frame_size = sample_rate / 50;
    let samples_per_frame = frame_size as usize * channels as usize;
    let bytes_per_frame = samples_per_frame * 2;
    let total_bytes_needed = bytes_per_frame * num_frames;

    println!(
        "Config: sr={sample_rate} ch={channels} app={application} \
         br={bitrate} cx={complexity} fs={frame_size} n_frames={num_frames}"
    );
    assert!(
        pcm_bytes.len() >= total_bytes_needed,
        "not enough PCM (need {total_bytes_needed} have {})",
        pcm_bytes.len()
    );

    let mut pcm_frames: Vec<Vec<i16>> = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let start = i * bytes_per_frame;
        let frame_bytes = &pcm_bytes[start..start + bytes_per_frame];
        let pcm: Vec<i16> = frame_bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        pcm_frames.push(pcm);
    }

    // Summary per frame
    for (fi, f) in pcm_frames.iter().enumerate() {
        let min = *f.iter().min().unwrap_or(&0);
        let max = *f.iter().max().unwrap_or(&0);
        let abs_sum: i64 = f.iter().map(|&s| (s as i64).abs()).sum();
        println!(
            "  frame {fi}: len={}, min={min}, max={max}, abs_avg={}, first5={:?}",
            f.len(),
            abs_sum / f.len() as i64,
            &f[..5.min(f.len())]
        );
    }

    // Rust encode
    let mut rust_enc = RustEncoder::new(sample_rate, channels, application).unwrap();
    rust_enc.set_bitrate(bitrate);
    rust_enc.set_vbr(0);
    rust_enc.set_complexity(complexity);
    let mut rust_packets: Vec<Vec<u8>> = Vec::with_capacity(num_frames);
    for pcm in &pcm_frames {
        let mut out = vec![0u8; 4000];
        let len = rust_enc.encode(pcm, frame_size, &mut out, 4000).unwrap();
        out.truncate(len as usize);
        rust_packets.push(out);
    }

    // C encode
    let mut c_packets: Vec<Vec<u8>> = Vec::with_capacity(num_frames);
    unsafe {
        let mut error: c_int = 0;
        let enc = bindings::opus_encoder_create(sample_rate, channels, application, &mut error);
        assert!(!enc.is_null() && error == 0);
        bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_BITRATE_REQUEST, bitrate);
        bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_VBR_REQUEST, 0 as c_int);
        bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_COMPLEXITY_REQUEST, complexity);
        for pcm in &pcm_frames {
            let mut out = vec![0u8; 4000];
            let len = bindings::opus_encode(enc, pcm.as_ptr(), frame_size, out.as_mut_ptr(), 4000);
            assert!(len >= 0, "C encode failed: {len}");
            out.truncate(len as usize);
            c_packets.push(out);
        }
        bindings::opus_encoder_destroy(enc);
    }

    // Compare
    let mut first_mismatch: Option<usize> = None;
    for i in 0..rust_packets.len().min(c_packets.len()) {
        if rust_packets[i] != c_packets[i] {
            first_mismatch = Some(i);
            break;
        }
    }
    match first_mismatch {
        None => println!("All {} frames match -- NOT REPRODUCED!", rust_packets.len()),
        Some(i) => {
            println!(
                "REPRO HIT: frame {i}/{num_frames} diverges, len rust={} c={}",
                rust_packets[i].len(),
                c_packets[i].len()
            );
            for j in 0..i {
                println!("  frame {j} MATCH len={}", rust_packets[j].len());
            }
            let rp = &rust_packets[i];
            let cp = &c_packets[i];
            let mut first_diff = None;
            for k in 0..rp.len().min(cp.len()) {
                if rp[k] != cp[k] {
                    first_diff = Some(k);
                    break;
                }
            }
            println!(
                "  first byte diff at offset {:?}",
                first_diff
            );
            println!("  rust = {:?}", rp);
            println!("  c    = {:?}", cp);
        }
    }
}
