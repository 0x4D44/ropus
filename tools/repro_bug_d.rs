//! Reproducer for Campaign 8 Bug D (multiframe encode divergence).
//!
//! Configuration (from fuzz log):
//!   sr=8000, ch=1, application=OPUS_APPLICATION_VOIP (2048),
//!   bitrate=12000, complexity=5, num_frames=10
//! Frame 4/10 diverges at byte offset 9; packet length 30 bytes.
//!
//! Note: the bug note mis-reported application=2048 as LOWDELAY, but 2048 is
//! OPUS_APPLICATION_VOIP. VOIP at 8 kHz is the SILK-dominated path (config 1,
//! SILK-NB 20ms, TOC first byte 0x0B matches the log).
//!
//! Strategy: try a handful of deterministic PCM seeds until one reproduces the
//! frame-4 divergence. We also decode the first bytes of the observed packet to
//! help confirm we are on the right encoder path.
//!
//! Usage: cargo run --release --bin repro_bug_d

#![allow(
    dead_code,
    clippy::needless_range_loop,
    clippy::unnecessary_cast,
    clippy::collapsible_if,
    clippy::identity_op
)]

#[path = "../tests/harness/bindings.rs"]
mod bindings;

use std::os::raw::c_int;

use mdopus::opus::encoder::{OpusEncoder as RustEncoder, OPUS_APPLICATION_VOIP};

const SAMPLE_RATE: i32 = 8000;
const CHANNELS: i32 = 1;
const APPLICATION: i32 = OPUS_APPLICATION_VOIP;
const BITRATE: i32 = 12000;
const COMPLEXITY: i32 = 5;
const FRAME_SIZE: i32 = 160; // 20 ms at 8 kHz
const NUM_FRAMES: usize = 10;

/// Simple xorshift PRNG — deterministic given seed.
fn xorshift_next(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn seed_noise(seed: u64, amp: i16) -> Vec<Vec<i16>> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    if s == 0 {
        s = 1;
    }
    let mut frames = Vec::with_capacity(NUM_FRAMES);
    for _ in 0..NUM_FRAMES {
        let mut frame = Vec::with_capacity(FRAME_SIZE as usize);
        for _ in 0..FRAME_SIZE {
            let r = xorshift_next(&mut s) as i32;
            let sample = ((r & 0xFFFF) as i32 - 0x8000) as i32;
            let scaled = (sample * amp as i32) >> 15;
            frame.push(scaled as i16);
        }
        frames.push(frame);
    }
    frames
}

fn seed_sine(hz: f32, amp: i16) -> Vec<Vec<i16>> {
    let mut frames = Vec::with_capacity(NUM_FRAMES);
    let mut phase: f64 = 0.0;
    let step = 2.0 * std::f64::consts::PI * hz as f64 / SAMPLE_RATE as f64;
    for _ in 0..NUM_FRAMES {
        let mut frame = Vec::with_capacity(FRAME_SIZE as usize);
        for _ in 0..FRAME_SIZE {
            let s = (phase.sin() * amp as f64) as i16;
            frame.push(s);
            phase += step;
        }
        frames.push(frame);
    }
    frames
}

fn seed_silence() -> Vec<Vec<i16>> {
    vec![vec![0i16; FRAME_SIZE as usize]; NUM_FRAMES]
}

fn seed_ramp() -> Vec<Vec<i16>> {
    let mut frames = Vec::with_capacity(NUM_FRAMES);
    let mut v: i32 = 0;
    for _ in 0..NUM_FRAMES {
        let mut frame = Vec::with_capacity(FRAME_SIZE as usize);
        for _ in 0..FRAME_SIZE {
            frame.push(v as i16);
            v = (v + 37) & 0x7FFF;
        }
        frames.push(frame);
    }
    frames
}

fn rust_encode_all(frames: &[Vec<i16>]) -> Vec<Vec<u8>> {
    let mut enc = RustEncoder::new(SAMPLE_RATE, CHANNELS, APPLICATION).unwrap();
    enc.set_bitrate(BITRATE);
    enc.set_vbr(0);
    enc.set_complexity(COMPLEXITY);

    let mut pkts = Vec::with_capacity(frames.len());
    for f in frames {
        let mut out = vec![0u8; 4000];
        let len = enc.encode(f, FRAME_SIZE, &mut out, 4000).unwrap();
        out.truncate(len as usize);
        pkts.push(out);
    }
    pkts
}

fn c_encode_all(frames: &[Vec<i16>]) -> Vec<Vec<u8>> {
    unsafe {
        let mut error: c_int = 0;
        let enc = bindings::opus_encoder_create(SAMPLE_RATE, CHANNELS, APPLICATION, &mut error);
        assert!(!enc.is_null() && error == 0);
        bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_BITRATE_REQUEST, BITRATE);
        bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_VBR_REQUEST, 0 as c_int);
        bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_COMPLEXITY_REQUEST, COMPLEXITY);

        let mut pkts = Vec::with_capacity(frames.len());
        for f in frames {
            let mut out = vec![0u8; 4000];
            let len = bindings::opus_encode(enc, f.as_ptr(), FRAME_SIZE, out.as_mut_ptr(), 4000);
            assert!(len >= 0, "C encode failed: {len}");
            out.truncate(len as usize);
            pkts.push(out);
        }
        bindings::opus_encoder_destroy(enc);
        pkts
    }
}

fn compare(name: &str, frames: &[Vec<i16>]) -> Option<usize> {
    let rp = rust_encode_all(frames);
    let cp = c_encode_all(frames);

    let mut first_diff = None;
    for i in 0..rp.len().min(cp.len()) {
        if rp[i] != cp[i] {
            first_diff = Some(i);
            break;
        }
    }
    match first_diff {
        Some(i) => {
            println!(
                "[{name}] frame {i}/{n} DIVERGES  rust_len={} c_len={}",
                rp[i].len(),
                cp[i].len(),
                n = rp.len()
            );
            let up_to = rp[i].len().max(cp[i].len()).min(30);
            println!("  rust[0..{up_to}]: {:?}", &rp[i][..rp[i].len().min(up_to)]);
            println!("  c   [0..{up_to}]: {:?}", &cp[i][..cp[i].len().min(up_to)]);
            for j in 0..i {
                println!("  frame {j} matched len={}", rp[j].len());
            }
        }
        None => {
            println!("[{name}] all {} frames match", rp.len());
        }
    }
    first_diff
}

/// Parse a fuzz corpus file the same way fuzz_encode_multiframe.rs does.
/// Returns (sr, ch, app, br, cx, num_frames, pcm_frames) if it would exercise
/// the same config path as Bug D, else None.
fn parse_corpus(data: &[u8]) -> Option<(i32, i32, i32, i32, i32, usize, Vec<Vec<i16>>)> {
    const SAMPLE_RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];
    const APPLICATIONS: [i32; 3] = [
        OPUS_APPLICATION_VOIP,
        mdopus::opus::encoder::OPUS_APPLICATION_AUDIO,
        mdopus::opus::encoder::OPUS_APPLICATION_RESTRICTED_LOWDELAY,
    ];

    if data.len() < 7 + 5 * 320 {
        return None;
    }
    let sample_rate = SAMPLE_RATES[(data[0] as usize) % SAMPLE_RATES.len()];
    let channels = if data[1] & 1 == 0 { 1 } else { 2 };
    let application = APPLICATIONS[(data[2] as usize) % APPLICATIONS.len()];
    let raw = u16::from_le_bytes([data[3], data[4]]) as i32;
    let bitrate = 6000 + (raw % 504001);
    let complexity = (data[5] as i32) % 11;
    let num_frames = 5 + ((data[6] as usize) % 6);
    let pcm_bytes = &data[7..];

    let frame_size = sample_rate / 50;
    let samples_per_frame = frame_size as usize * channels as usize;
    let bytes_per_frame = samples_per_frame * 2;
    let total_bytes_needed = bytes_per_frame * num_frames;
    if pcm_bytes.len() < total_bytes_needed {
        return None;
    }

    let mut frames = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let start = i * bytes_per_frame;
        let fb = &pcm_bytes[start..start + bytes_per_frame];
        let pcm: Vec<i16> = fb
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        frames.push(pcm);
    }
    Some((sample_rate, channels, application, bitrate, complexity, num_frames, frames))
}

fn main() {
    println!(
        "Config: sr={SAMPLE_RATE} ch={CHANNELS} app={APPLICATION} \
         br={BITRATE} cx={COMPLEXITY} fs={FRAME_SIZE} n_frames={NUM_FRAMES}"
    );

    // --- Phase 1: scan corpus for inputs that match the target config and
    // actually diverge across both impls. ---
    let corpus_dir = std::path::Path::new("tests/fuzz/corpus/fuzz_encode_multiframe");
    let mut scanned = 0usize;
    let mut matching_cfg = 0usize;
    let mut hits = 0usize;

    if let Ok(rd) = std::fs::read_dir(corpus_dir) {
        for entry in rd.flatten() {
            let path = entry.path();
            let data = match std::fs::read(&path) {
                Ok(d) => d,
                Err(_) => continue,
            };
            scanned += 1;
            let parsed = match parse_corpus(&data) {
                Some(p) => p,
                None => continue,
            };
            let (sr, ch, app, br, cx, nf, frames) = parsed;
            // Match the Bug D config exactly.
            if !(sr == SAMPLE_RATE
                && ch == CHANNELS
                && app == APPLICATION
                && br == BITRATE
                && cx == COMPLEXITY
                && nf == NUM_FRAMES)
            {
                continue;
            }
            matching_cfg += 1;

            let rp = rust_encode_all(&frames);
            let cp = c_encode_all(&frames);
            for i in 0..rp.len().min(cp.len()) {
                if rp[i] != cp[i] {
                    hits += 1;
                    println!(
                        ">>> CORPUS REPRO file={} frame={i} rust_len={} c_len={}",
                        path.file_name().unwrap().to_string_lossy(),
                        rp[i].len(),
                        cp[i].len()
                    );
                    let up_to = 30.min(rp[i].len().max(cp[i].len()));
                    println!("    rust: {:?}", &rp[i][..rp[i].len().min(up_to)]);
                    println!("    c   : {:?}", &cp[i][..cp[i].len().min(up_to)]);
                    break;
                }
            }
            if hits >= 3 {
                break;
            }
        }
    } else {
        println!("Could not read corpus directory");
    }
    println!(
        "Corpus scan: {scanned} files, {matching_cfg} matched target config, {hits} diverged"
    );
    if hits > 0 {
        return;
    }

    // --- Phase 2: Relax bitrate/complexity/frames constraints: any 8kHz mono
    // VOIP multiframe input. The exact bitrate may still trigger it. ---
    let mut relaxed_hits = 0usize;
    let mut relaxed_scanned = 0usize;
    if let Ok(rd) = std::fs::read_dir(corpus_dir) {
        for entry in rd.flatten() {
            let path = entry.path();
            let data = match std::fs::read(&path) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let parsed = match parse_corpus(&data) {
                Some(p) => p,
                None => continue,
            };
            let (sr, ch, app, _br, _cx, _nf, frames) = parsed;
            if !(sr == SAMPLE_RATE && ch == CHANNELS && app == APPLICATION) {
                continue;
            }
            relaxed_scanned += 1;
            // Always run with the Bug D config for the harness, regardless of
            // what the corpus config byte said — we want to see if any PCM
            // triggers the bug under the target config.
            let rp = rust_encode_all(&frames[..NUM_FRAMES.min(frames.len())]);
            let cp = c_encode_all(&frames[..NUM_FRAMES.min(frames.len())]);
            for i in 0..rp.len().min(cp.len()) {
                if rp[i] != cp[i] {
                    relaxed_hits += 1;
                    println!(
                        ">>> RELAXED CORPUS REPRO file={} frame={i} rust_len={} c_len={}",
                        path.file_name().unwrap().to_string_lossy(),
                        rp[i].len(),
                        cp[i].len()
                    );
                    if relaxed_hits >= 3 {
                        break;
                    }
                    break;
                }
            }
            if relaxed_hits >= 3 {
                break;
            }
        }
    }
    println!(
        "Relaxed corpus scan: {relaxed_scanned} 8kHz-mono-VOIP files, {relaxed_hits} diverged"
    );
    if relaxed_hits > 0 {
        return;
    }

    // --- Phase 2.5: mix-and-match. For each matching-config corpus file,
    // replace its PCM with PRNG noise of varying seeds and amplitudes. ---
    let mut mm_hits = 0usize;
    let mut mm_tries = 0usize;
    let matching_files: Vec<std::path::PathBuf> = if let Ok(rd) = std::fs::read_dir(corpus_dir) {
        rd.flatten()
            .map(|e| e.path())
            .filter(|p| {
                std::fs::read(p)
                    .ok()
                    .and_then(|d| parse_corpus(&d))
                    .map(|p| {
                        let (sr, ch, app, br, cx, nf, _) = p;
                        sr == SAMPLE_RATE
                            && ch == CHANNELS
                            && app == APPLICATION
                            && br == BITRATE
                            && cx == COMPLEXITY
                            && nf == NUM_FRAMES
                    })
                    .unwrap_or(false)
            })
            .collect()
    } else {
        vec![]
    };
    println!("{} matching-config corpus files for mix-and-match", matching_files.len());

    // Transition patterns: silence -> noise -> silence, ramp up, speech-like.
    let make_mixed = |seed: u64, pattern: u8| -> Vec<Vec<i16>> {
        let mut out: Vec<Vec<i16>> = Vec::with_capacity(NUM_FRAMES);
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        if s == 0 {
            s = 1;
        }
        for fi in 0..NUM_FRAMES {
            let amp = match pattern {
                0 => {
                    // silence then ramp up
                    if fi < 3 {
                        0i16
                    } else {
                        ((fi - 2) as i32 * 3000).min(24000) as i16
                    }
                }
                1 => {
                    // noise burst on frame 3 and 4
                    if fi == 3 || fi == 4 {
                        16000
                    } else {
                        500
                    }
                }
                2 => {
                    // increasing amplitude
                    ((fi + 1) as i32 * 2000).min(30000) as i16
                }
                3 => {
                    // decreasing amplitude
                    ((NUM_FRAMES - fi) as i32 * 2000).min(30000) as i16
                }
                _ => 8000,
            };
            let mut frame = Vec::with_capacity(FRAME_SIZE as usize);
            for _ in 0..FRAME_SIZE {
                let r = xorshift_next(&mut s) as i32;
                let sample = ((r & 0xFFFF) as i32 - 0x8000) as i32;
                let scaled = (sample * amp as i32) >> 15;
                frame.push(scaled as i16);
            }
            out.push(frame);
        }
        out
    };

    for pattern in 0..4u8 {
        for seed in 1..=200u64 {
            mm_tries += 1;
            let frames = make_mixed(seed, pattern);
            let rp = rust_encode_all(&frames);
            let cp = c_encode_all(&frames);
            for i in 0..rp.len().min(cp.len()) {
                if rp[i] != cp[i] {
                    mm_hits += 1;
                    println!(
                        ">>> MIX REPRO pattern={pattern} seed={seed} frame={i} rust_len={} c_len={}",
                        rp[i].len(),
                        cp[i].len()
                    );
                    if mm_hits >= 3 {
                        break;
                    }
                    break;
                }
            }
            if mm_hits >= 3 {
                break;
            }
        }
        if mm_hits >= 3 {
            break;
        }
    }
    println!("Mix tries: {mm_tries}, hits: {mm_hits}");
    if mm_hits > 0 {
        return;
    }

    // --- Phase 3: broad PRNG search (unchanged). ---

    // Try a range of seeds; stop on first reproduction.
    // Broad PRNG search: try many seeds silently, only print divergences.
    let mut hits = 0usize;
    let mut searched = 0usize;
    let amps: &[i16] = &[2048, 4096, 8192, 16384, 24576, 32767];

    for amp in amps {
        for seed in 1..=500u64 {
            searched += 1;
            let frames = seed_noise(seed, *amp);
            let rp = rust_encode_all(&frames);
            let cp = c_encode_all(&frames);
            let mut first_diff = None;
            for i in 0..rp.len().min(cp.len()) {
                if rp[i] != cp[i] {
                    first_diff = Some(i);
                    break;
                }
            }
            if let Some(i) = first_diff {
                hits += 1;
                println!(
                    ">>> REPRO seed={seed} amp={amp} frame={i} rust_len={} c_len={}",
                    rp[i].len(),
                    cp[i].len()
                );
                let up_to = 30.min(rp[i].len().max(cp[i].len()));
                println!("    rust: {:?}", &rp[i][..rp[i].len().min(up_to)]);
                println!("    c   : {:?}", &cp[i][..cp[i].len().min(up_to)]);
                if hits >= 3 {
                    break;
                }
            }
        }
        if hits >= 3 {
            break;
        }
    }

    // Also a few sine sweeps with low amplitude (close to silence threshold).
    for (hz, amp) in &[
        (50.0f32, 500i16),
        (80.0, 1000),
        (120.0, 2000),
        (200.0, 100),
        (300.0, 200),
        (400.0, 300),
        (700.0, 500),
        (1500.0, 100),
        (3000.0, 50),
    ] {
        searched += 1;
        let frames = seed_sine(*hz, *amp);
        let rp = rust_encode_all(&frames);
        let cp = c_encode_all(&frames);
        for i in 0..rp.len().min(cp.len()) {
            if rp[i] != cp[i] {
                hits += 1;
                println!(
                    ">>> REPRO sine hz={hz} amp={amp} frame={i} rust_len={} c_len={}",
                    rp[i].len(),
                    cp[i].len()
                );
                let up_to = 30.min(rp[i].len().max(cp[i].len()));
                println!("    rust: {:?}", &rp[i][..rp[i].len().min(up_to)]);
                println!("    c   : {:?}", &cp[i][..cp[i].len().min(up_to)]);
                break;
            }
        }
    }

    println!("Searched {searched} seeds; {hits} reproductions");
}
