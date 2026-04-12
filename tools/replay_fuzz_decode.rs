//! Replay tool for fuzz_decode corpus.
//!
//! Iterates every file in `tests/fuzz/corpus/fuzz_decode/`, reproduces the
//! fuzz target's decode (Rust + C reference), and reports any CELT-only seed
//! whose PCM output differs between the two implementations. Also matches
//! against panic fingerprints captured from the overnight fuzz run.
//!
//! Usage:
//!   cargo run --release --bin replay_fuzz_decode --no-default-features \
//!       --features simd -- [corpus_dir] [--copy-to <crashes_dir>] \
//!       [--fingerprints <panics.json>]

#![allow(clippy::too_many_arguments)]

#[path = "../tests/harness/bindings.rs"]
mod bindings;

use mdopus::opus::decoder::OpusDecoder;
use std::fs;
use std::os::raw::c_int;
use std::path::{Path, PathBuf};

/// Sample rate lookup from config byte (mirrors fuzz_decode.rs).
const SAMPLE_RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];
/// Max frame size at any sample rate (120 ms @ 48 kHz).
const MAX_FRAME: i32 = 5760;

/// Target panic signatures from the fuzz run: (sample_rate, channels, pkt_len, samples).
const TARGETS: &[(i32, i32, usize, usize)] = &[
    (48000, 1, 75, 720),
    (48000, 1, 33, 2280),
    (16000, 1, 90, 240),
    (48000, 1, 124, 720),
];

fn is_celt_only_packet(packet: &[u8]) -> bool {
    !packet.is_empty() && (packet[0] & 0x80) != 0
}

/// Decode using the C reference library. Returns interleaved PCM or a negative error code.
fn c_decode(data: &[u8], sample_rate: i32, channels: i32) -> Result<Vec<i16>, i32> {
    unsafe {
        let mut error: c_int = 0;
        let dec = bindings::opus_decoder_create(sample_rate, channels, &mut error);
        if dec.is_null() || error != bindings::OPUS_OK {
            if !dec.is_null() {
                bindings::opus_decoder_destroy(dec);
            }
            return Err(error);
        }
        let max_frame = MAX_FRAME as usize * channels as usize;
        let mut pcm = vec![0i16; max_frame];
        let ret = bindings::opus_decode(
            dec,
            data.as_ptr(),
            data.len() as i32,
            pcm.as_mut_ptr(),
            MAX_FRAME,
            0,
        );
        bindings::opus_decoder_destroy(dec);
        if ret < 0 {
            Err(ret)
        } else {
            pcm.truncate(ret as usize * channels as usize);
            Ok(pcm)
        }
    }
}

fn rust_decode(data: &[u8], sample_rate: i32, channels: i32) -> Result<Vec<i16>, i32> {
    let mut dec = OpusDecoder::new(sample_rate, channels)?;
    let max_pcm = MAX_FRAME as usize * channels as usize;
    let mut pcm = vec![0i16; max_pcm];
    let n = dec.decode(Some(data), &mut pcm, MAX_FRAME, false)?;
    pcm.truncate(n as usize * channels as usize);
    Ok(pcm)
}

/// A panic fingerprint loaded from JSON (subset we care about).
#[derive(Debug)]
struct Fingerprint {
    sr: i32,
    ch: i32,
    pkt_len: usize,
    samples: usize,
    rust_pcm: Vec<i16>, // "left" from the assertion
    c_pcm: Vec<i16>,    // "right" from the assertion
    label: String,      // "F1a" .. "F1d"
}

fn parse_pcm_array(s: &str, key: &str) -> Option<Vec<i16>> {
    // Look for `"key": [n, n, n, ...]`
    let needle = format!("\"{}\"", key);
    let idx = s.find(&needle)?;
    let rest = &s[idx + needle.len()..];
    let lb = rest.find('[')?;
    let rb = rest.find(']')?;
    let inner = &rest[lb + 1..rb];
    let mut out = Vec::with_capacity(4096);
    for tok in inner.split(',') {
        let t = tok.trim();
        if t.is_empty() {
            continue;
        }
        let v: i16 = t.parse().ok()?;
        out.push(v);
    }
    Some(out)
}

fn parse_int_field(s: &str, key: &str) -> Option<i64> {
    let needle = format!("\"{}\"", key);
    let idx = s.find(&needle)?;
    let rest = &s[idx + needle.len()..];
    let colon = rest.find(':')?;
    let after = &rest[colon + 1..];
    // Skip whitespace, read digits until non-digit
    let bytes = after.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    let start = i;
    while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'-') {
        i += 1;
    }
    after[start..i].parse().ok()
}

fn load_fingerprints(path: &Path) -> Vec<Fingerprint> {
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("warning: cannot read {}: {e}", path.display());
            return Vec::new();
        }
    };
    // Minimal hand-rolled parser for the array-of-objects JSON produced by the extract script.
    let mut fps = Vec::new();
    let mut i = 0;
    let bytes = text.as_bytes();
    while i < bytes.len() {
        if bytes[i] != b'{' {
            i += 1;
            continue;
        }
        // Find matching brace (depth-aware, but rust_pcm arrays don't contain braces).
        let mut depth = 0i32;
        let start = i;
        while i < bytes.len() {
            match bytes[i] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        i += 1;
                        break;
                    }
                }
                _ => {}
            }
            i += 1;
        }
        let obj = &text[start..i];
        let sr = parse_int_field(obj, "sr").unwrap_or(-1) as i32;
        let ch = parse_int_field(obj, "ch").unwrap_or(-1) as i32;
        let pkt_len = parse_int_field(obj, "pkt_len").unwrap_or(-1) as usize;
        let samples = parse_int_field(obj, "samples").unwrap_or(-1) as usize;
        let left = parse_pcm_array(obj, "left").unwrap_or_default();
        let right = parse_pcm_array(obj, "right").unwrap_or_default();
        if sr < 0 || ch < 0 || left.is_empty() || right.is_empty() {
            continue;
        }
        // Auto-label F1a..F1d by TARGETS order
        let label = TARGETS
            .iter()
            .position(|&(tsr, tch, tpl, tns)| {
                tsr == sr && tch == ch && tpl == pkt_len && tns == samples
            })
            .map(|i| format!("F1{}", ['a', 'b', 'c', 'd'][i]))
            .unwrap_or_else(|| "???".into());
        fps.push(Fingerprint {
            sr,
            ch,
            pkt_len,
            samples,
            rust_pcm: left,
            c_pcm: right,
            label,
        });
    }
    fps
}

/// Find the first divergence index between two PCM slices (if any).
fn first_divergence(a: &[i16], b: &[i16]) -> Option<usize> {
    a.iter().zip(b.iter()).position(|(x, y)| x != y)
}

struct Finding {
    path: PathBuf,
    sample_rate: i32,
    channels: i32,
    pkt_len: usize,
    samples: usize,
    first_diff_idx: usize,
    first_diffs: Vec<(i16, i16)>,
    matched_target: Option<usize>, // index into TARGETS
    matched_fingerprint: Option<String>,
}

fn scan_file(path: &Path, fingerprints: &[Fingerprint]) -> Option<Finding> {
    let bytes = fs::read(path).ok()?;
    if bytes.len() < 3 {
        return None;
    }
    let sr_idx = (bytes[0] as usize) % SAMPLE_RATES.len();
    let sample_rate = SAMPLE_RATES[sr_idx];
    let channels: i32 = if bytes[1] & 1 == 0 { 1 } else { 2 };
    let packet = &bytes[2..];
    if packet.is_empty() {
        return None;
    }
    // Mirror the fuzz target: only CELT-only packets trigger the PCM assertion.
    if !is_celt_only_packet(packet) {
        return None;
    }

    // Fast filter: only process packets matching at least one target signature.
    let fp_match = fingerprints.iter().find(|fp| {
        fp.sr == sample_rate
            && fp.ch == channels
            && fp.pkt_len == packet.len()
    });
    // We still do a full decode even if no target match, to surface other divergences.

    let rust_ret = rust_decode(packet, sample_rate, channels);
    let c_ret = c_decode(packet, sample_rate, channels);

    let (rust_pcm, c_pcm) = match (rust_ret, c_ret) {
        (Ok(r), Ok(c)) => (r, c),
        _ => return None, // one erred; not a PCM mismatch (other assertions handle this)
    };

    // Check fingerprint match first (more specific).
    let mut matched_fp_label: Option<String> = None;
    if let Some(fp) = fp_match {
        if rust_pcm == fp.rust_pcm && c_pcm == fp.c_pcm {
            matched_fp_label = Some(fp.label.clone());
        } else if rust_pcm == fp.rust_pcm {
            matched_fp_label = Some(format!("{}-rust-only", fp.label));
        } else if c_pcm == fp.c_pcm {
            matched_fp_label = Some(format!("{}-c-only", fp.label));
        }
    }
    // Also check ALL fingerprints (not just fp_match) for any PCM fingerprint match.
    if matched_fp_label.is_none() {
        for fp in fingerprints {
            if rust_pcm == fp.rust_pcm {
                matched_fp_label = Some(format!("{}-rust-pcm-any-sig", fp.label));
                break;
            }
            if c_pcm == fp.c_pcm {
                matched_fp_label = Some(format!("{}-c-pcm-any-sig", fp.label));
                break;
            }
        }
    }

    if rust_pcm.len() != c_pcm.len() {
        let samples = rust_pcm.len() / channels as usize;
        return Some(Finding {
            path: path.to_path_buf(),
            sample_rate,
            channels,
            pkt_len: packet.len(),
            samples,
            first_diff_idx: 0,
            first_diffs: vec![],
            matched_target: None,
            matched_fingerprint: matched_fp_label,
        });
    }

    if rust_pcm == c_pcm && matched_fp_label.is_none() {
        return None;
    }

    // Find first divergence and capture up to 4 samples starting there.
    let first_diff_idx = first_divergence(&rust_pcm, &c_pcm).unwrap_or(usize::MAX);
    let mut first_diffs = Vec::with_capacity(4);
    if first_diff_idx != usize::MAX {
        for i in first_diff_idx..(first_diff_idx + 4).min(rust_pcm.len()) {
            first_diffs.push((rust_pcm[i], c_pcm[i]));
        }
    }

    let samples = rust_pcm.len() / channels as usize;
    let matched_target = TARGETS.iter().position(|&(sr, ch, pl, ns)| {
        sr == sample_rate && ch == channels && pl == packet.len() && ns == samples
    });

    // If neither target nor fingerprint nor pcm mismatch, skip.
    if matched_target.is_none() && matched_fp_label.is_none() && first_diffs.is_empty() {
        return None;
    }

    Some(Finding {
        path: path.to_path_buf(),
        sample_rate,
        channels,
        pkt_len: packet.len(),
        samples,
        first_diff_idx,
        first_diffs,
        matched_target,
        matched_fingerprint: matched_fp_label,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut corpus_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fuzz/corpus/fuzz_decode");
    let mut copy_to: Option<PathBuf> = None;
    let mut fingerprints_path: Option<PathBuf> = Some(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tools/fuzz_decode_panics.json"),
    );

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--copy-to" => {
                i += 1;
                copy_to = Some(PathBuf::from(&args[i]));
            }
            "--fingerprints" => {
                i += 1;
                fingerprints_path = Some(PathBuf::from(&args[i]));
            }
            "--no-fingerprints" => {
                fingerprints_path = None;
            }
            other if !other.starts_with("--") => {
                corpus_dir = PathBuf::from(other);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let fingerprints: Vec<Fingerprint> = fingerprints_path
        .as_deref()
        .map(load_fingerprints)
        .unwrap_or_default();
    if !fingerprints.is_empty() {
        println!("loaded {} panic fingerprint(s)", fingerprints.len());
        for fp in &fingerprints {
            println!(
                "  {} sr={} ch={} pkt_len={} samples={}",
                fp.label, fp.sr, fp.ch, fp.pkt_len, fp.samples
            );
        }
    }

    println!("scanning {}", corpus_dir.display());
    let entries: Vec<PathBuf> = match fs::read_dir(&corpus_dir) {
        Ok(r) => r.filter_map(|e| e.ok().map(|e| e.path())).collect(),
        Err(e) => {
            eprintln!("read_dir failed: {e}");
            std::process::exit(1);
        }
    };
    println!("{} files", entries.len());

    let mut findings: Vec<Finding> = Vec::new();
    let mut celt_only_total = 0usize;
    let mut matched_total = [0usize; 4];
    for p in &entries {
        if !p.is_file() {
            continue;
        }
        if let Some(f) = scan_file(p, &fingerprints) {
            findings.push(f);
        }
        if let Ok(bytes) = fs::read(p) {
            if bytes.len() >= 3 {
                let sr_idx = (bytes[0] as usize) % SAMPLE_RATES.len();
                let sr = SAMPLE_RATES[sr_idx];
                let ch: i32 = if bytes[1] & 1 == 0 { 1 } else { 2 };
                let pkt = &bytes[2..];
                if !pkt.is_empty() && is_celt_only_packet(pkt) {
                    celt_only_total += 1;
                    for (i, &(tsr, tch, tpl, _tns)) in TARGETS.iter().enumerate() {
                        if tsr == sr && tch == ch && tpl == pkt.len() {
                            matched_total[i] += 1;
                        }
                    }
                }
            }
        }
    }
    println!("CELT-only corpus entries: {celt_only_total}");
    for (i, c) in matched_total.iter().enumerate() {
        let letter = ['a', 'b', 'c', 'd'][i];
        println!("  F1{letter} (sr/ch/pkt_len match, any payload): {c}");
    }

    // Diagnostic: for each target, list sample counts produced by matching corpus files.
    println!("\n=== Diagnostic: sample counts for matching seeds ===");
    for (i, &(tsr, tch, tpl, tns)) in TARGETS.iter().enumerate() {
        let letter = ['a', 'b', 'c', 'd'][i];
        println!("F1{letter} target samples={tns}:");
        for p in &entries {
            if !p.is_file() {
                continue;
            }
            let Ok(bytes) = fs::read(p) else { continue };
            if bytes.len() < 3 {
                continue;
            }
            let sr_idx = (bytes[0] as usize) % SAMPLE_RATES.len();
            let sr = SAMPLE_RATES[sr_idx];
            let ch: i32 = if bytes[1] & 1 == 0 { 1 } else { 2 };
            let pkt = &bytes[2..];
            if sr != tsr || ch != tch || pkt.len() != tpl || !is_celt_only_packet(pkt) {
                continue;
            }
            // Decode with Rust and C and show actual sample counts.
            let rs = rust_decode(pkt, sr, ch);
            let cs = c_decode(pkt, sr, ch);
            let rust_n = match &rs {
                Ok(v) => v.len() as i32 / ch,
                Err(e) => -*e,
            };
            let c_n = match &cs {
                Ok(v) => v.len() as i32 / ch,
                Err(e) => -*e,
            };
            let eq = match (&rs, &cs) {
                (Ok(r), Ok(c)) => r == c,
                _ => false,
            };
            println!(
                "    {} rust_n={} c_n={} pcm_eq={}",
                p.file_name().unwrap().to_string_lossy(),
                rust_n,
                c_n,
                eq
            );
        }
    }

    println!("\n=== Findings ===");
    println!("{} file(s) produced diverging PCM or matched fingerprint", findings.len());
    for f in &findings {
        let tag = f
            .matched_target
            .map(|i| format!("F1{}", ['a', 'b', 'c', 'd'][i]))
            .unwrap_or_else(|| "???".into());
        println!(
            "\n[{}{}] {}",
            tag,
            f.matched_fingerprint
                .as_ref()
                .map(|l| format!(" /FP={l}"))
                .unwrap_or_default(),
            f.path.file_name().unwrap().to_string_lossy()
        );
        println!(
            "    sr={} ch={} pkt_len={} samples={}",
            f.sample_rate, f.channels, f.pkt_len, f.samples
        );
        if !f.first_diffs.is_empty() {
            println!("    first divergence at sample index {}", f.first_diff_idx);
            print!("    first diffs (rust, c): ");
            for (r, c) in &f.first_diffs {
                print!("({r},{c}) ");
            }
            println!();
        }
    }

    println!("\n=== Target coverage ===");
    for (i, tgt) in TARGETS.iter().enumerate() {
        let tag = format!("F1{}", ['a', 'b', 'c', 'd'][i]);
        let matching: Vec<_> = findings
            .iter()
            .filter(|f| f.matched_target == Some(i) || f.matched_fingerprint.as_deref() == Some(tag.as_str()))
            .collect();
        println!(
            "{} (sr={} ch={} pkt_len={} samples={}): {} seed(s)",
            tag, tgt.0, tgt.1, tgt.2, tgt.3, matching.len()
        );
        for f in matching {
            println!("    - {}", f.path.file_name().unwrap().to_string_lossy());
        }
    }

    if let Some(dst_dir) = copy_to {
        if let Err(e) = fs::create_dir_all(&dst_dir) {
            eprintln!("cannot create {}: {}", dst_dir.display(), e);
            std::process::exit(1);
        }
        println!("\n=== Copying regression artifacts to {} ===", dst_dir.display());
        for (i, _tgt) in TARGETS.iter().enumerate() {
            let letter = ['a', 'b', 'c', 'd'][i];
            let tag = format!("F1{letter}");
            let matching: Vec<_> = findings
                .iter()
                .filter(|f| f.matched_target == Some(i) || f.matched_fingerprint.as_deref() == Some(tag.as_str()))
                .collect();
            for f in matching {
                let hash = f
                    .path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_string();
                let short = hash.chars().take(12).collect::<String>();
                let out_name = format!("finding1_{letter}_{short}.bin");
                let out_path = dst_dir.join(&out_name);
                match fs::copy(&f.path, &out_path) {
                    Ok(_) => println!(
                        "    copied {} -> {}",
                        f.path.file_name().unwrap().to_string_lossy(),
                        out_name
                    ),
                    Err(e) => eprintln!(
                        "    copy failed {} -> {}: {}",
                        f.path.display(),
                        out_path.display(),
                        e
                    ),
                }
            }
        }
    }
}
