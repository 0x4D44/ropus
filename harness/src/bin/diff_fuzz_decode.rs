//! Differential PCM decode tool for fuzz crash files.
//!
//! Reads a single fuzz_decode crash file (2-byte header + Opus packet),
//! decodes with both the Rust ropus decoder and the C reference (via FFI),
//! and prints the first divergence index plus a context window.
//!
//! Usage:
//!   cargo run --release --bin diff_fuzz_decode -- <path1> [<path2> ...]
//!
//! Header layout (mirrors fuzz_decode.rs):
//!   data[0]%5 → sample rate index into [8000,12000,16000,24000,48000]
//!   data[1]&1 → channels (0→mono, 1→stereo)
//!   data[2..] → Opus packet bytes

#[path = "../bindings.rs"]
mod bindings;

use ropus::opus::decoder::{
    OpusDecoder, opus_packet_get_nb_frames, opus_packet_get_samples_per_frame,
};
use std::fs;
use std::os::raw::c_int;
use std::path::Path;

/// Compare Rust and C CELT decoder state. Reads via debug accessors and prints
/// any differences. Returns true if all matched.
fn compare_celt_state(
    label: &str,
    rust: &OpusDecoder,
    c_dec: *mut bindings::OpusDecoder,
    nb_ebands: usize,
) -> bool {
    let mut all_match = true;

    // postfilter
    let (rp, rpo, rg, rgo, rt, rto) = rust.debug_get_postfilter();
    let mut cp: i32 = 0;
    let mut cpo: i32 = 0;
    let mut cg: i32 = 0;
    let mut cgo: i32 = 0;
    let mut ct: i32 = 0;
    let mut cto: i32 = 0;
    unsafe {
        bindings::debug_get_celt_postfilter(
            c_dec, &mut cp, &mut cpo, &mut cg, &mut cgo, &mut ct, &mut cto,
        );
    }
    if (rp, rpo, rg, rgo, rt, rto) != (cp, cpo, cg, cgo, ct, cto) {
        println!(
            "    {label}: postfilter rust=({rp},{rpo},{rg},{rgo},{rt},{rto}) c=({cp},{cpo},{cg},{cgo},{ct},{cto})"
        );
        all_match = false;
    }

    // preemph_mem
    let r_pre = rust.debug_get_preemph_mem();
    let mut c_pre = [0i32; 2];
    unsafe {
        bindings::debug_get_celt_preemph_mem(c_dec, c_pre.as_mut_ptr());
    }
    if r_pre != c_pre {
        println!("    {label}: preemph rust={r_pre:?} c={c_pre:?}");
        all_match = false;
    }

    // old_band_e
    let r_be = rust.debug_get_old_band_e();
    let max_be = r_be.len() as i32;
    let mut c_be = vec![0i32; max_be as usize];
    let n_be = unsafe { bindings::debug_get_celt_old_band_e(c_dec, c_be.as_mut_ptr(), max_be) };
    let n_be = n_be as usize;
    if r_be[..n_be] != c_be[..n_be] {
        // Find first diff
        let i = r_be[..n_be]
            .iter()
            .zip(c_be[..n_be].iter())
            .position(|(a, b)| a != b);
        println!(
            "    {label}: old_band_e first diff at {i:?} (len Rust={}, C={})",
            r_be.len(),
            n_be
        );
        if let Some(idx) = i {
            let lo = idx.saturating_sub(1);
            let hi = (idx + 4).min(n_be);
            for j in lo..hi {
                println!("       [{j:3}] rust={} c={}", r_be[j], c_be[j]);
            }
        }
        all_match = false;
    }

    // old_log_e and old_log_e2
    let r_le = rust.debug_get_old_log_e();
    let r_le2 = rust.debug_get_old_log_e2();
    let max_le = r_le.len() as i32;
    let mut c_le = vec![0i32; max_le as usize];
    let mut c_le2 = vec![0i32; max_le as usize];
    let n_le = unsafe {
        bindings::debug_get_celt_old_log_e(c_dec, c_le.as_mut_ptr(), c_le2.as_mut_ptr(), max_le)
    };
    let n_le = n_le as usize;
    if r_le[..n_le] != c_le[..n_le] {
        let i = r_le[..n_le]
            .iter()
            .zip(c_le[..n_le].iter())
            .position(|(a, b)| a != b);
        println!("    {label}: old_log_e first diff at {i:?}");
        all_match = false;
    }
    if r_le2[..n_le] != c_le2[..n_le] {
        let i = r_le2[..n_le]
            .iter()
            .zip(c_le2[..n_le].iter())
            .position(|(a, b)| a != b);
        println!("    {label}: old_log_e2 first diff at {i:?}");
        all_match = false;
    }

    // decode_mem: decode_buffer_size = 2048, plus overlap = 120, so the
    // per-channel allocation is 2168 samples. Read both regions.
    let mem_count = 2168;
    let r_mem = rust.debug_get_decode_mem(0, mem_count);
    let mut c_mem = vec![0i32; mem_count];
    unsafe {
        bindings::debug_get_celt_decode_mem(c_dec, 0, mem_count as c_int, c_mem.as_mut_ptr());
    }
    if r_mem != c_mem {
        let i = r_mem
            .iter()
            .zip(c_mem.iter())
            .position(|(a, b)| a != b)
            .unwrap();
        let last = r_mem
            .iter()
            .zip(c_mem.iter())
            .rposition(|(a, b)| a != b)
            .unwrap();
        let n_diff = r_mem
            .iter()
            .zip(c_mem.iter())
            .filter(|(a, b)| a != b)
            .count();
        println!("    {label}: decode_mem diff: first={i} last={last} n_diff={n_diff}/{mem_count}");
        let lo = i.saturating_sub(2);
        let hi = (i + 6).min(mem_count);
        for j in lo..hi {
            println!(
                "       [{j:5}] rust={:10} c={:10}{}",
                r_mem[j],
                c_mem[j],
                if r_mem[j] != c_mem[j] { " *" } else { "" }
            );
        }
        all_match = false;
    }

    let _ = nb_ebands;
    all_match
}

const SAMPLE_RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];
const MAX_FRAME: i32 = 5760;

/// Parse a code-3 VBR packet's per-sub-frame sizes (excluding ToC + frame_byte +
/// padding-length bytes). Returns (per_frame_sizes, payload_offset).
/// Mirrors `parse_size` + `opus_packet_parse_impl` for code-3 VBR; returns None
/// on parse failure or non-code-3 input.
fn parse_code3_vbr_sizes(packet: &[u8]) -> Option<(Vec<i32>, usize)> {
    if packet.len() < 2 {
        return None;
    }
    let toc = packet[0];
    if toc & 0x3 != 3 {
        return None;
    }
    let ch = packet[1];
    if ch & 0x80 == 0 {
        // CBR; not what we want here
        return None;
    }
    let count = (ch & 0x3f) as usize;
    if count == 0 {
        return None;
    }
    let mut pos = 2usize;
    let mut remaining = packet.len() as i32 - 2;
    let mut pad: i32 = 0;
    if ch & 0x40 != 0 {
        // padding bytes
        loop {
            if remaining <= 0 || pos >= packet.len() {
                return None;
            }
            let p = packet[pos];
            pos += 1;
            remaining -= 1;
            let tmp = if p == 255 { 254 } else { p as i32 };
            remaining -= tmp;
            pad += tmp;
            if p != 255 {
                break;
            }
        }
    }
    if remaining < 0 {
        return None;
    }
    let mut sizes = vec![0i32; count];
    let mut last_size = remaining;
    for slot in sizes.iter_mut().take(count - 1) {
        if pos >= packet.len() {
            return None;
        }
        let (bytes, sz) = if packet[pos] < 252 {
            (1i32, packet[pos] as i32)
        } else if pos + 1 >= packet.len() {
            return None;
        } else {
            (2i32, 4 * packet[pos + 1] as i32 + packet[pos] as i32)
        };
        if sz < 0 || sz > remaining {
            return None;
        }
        remaining -= bytes;
        pos += bytes as usize;
        if sz > remaining {
            return None;
        }
        *slot = sz;
        last_size -= bytes + sz;
    }
    if !(0..=1275).contains(&last_size) {
        return None;
    }
    sizes[count - 1] = last_size;
    let _ = pad;
    Some((sizes, pos))
}

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

/// Build a synthetic single-frame (code-0) Opus packet by replacing the TOC
/// code bits with 00 (single frame). All other TOC bits (config, stereo) are
/// preserved from the original. The body is just the sub-frame's payload.
fn synth_code0_packet(orig_toc: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + payload.len());
    out.push(orig_toc & !0x3); // clear code bits → 00
    out.extend_from_slice(payload);
    out
}

/// Decode a multi-frame code-3 packet by splitting it into per-sub-frame
/// code-0 packets and decoding each through both Rust and C decoders that
/// share state across calls. After each sub-frame, compares Rust vs C PCM and
/// prints any per-frame divergence.
fn split_decode(
    packet: &[u8],
    sample_rate: i32,
    channels: i32,
    sizes: &[i32],
    payload_off: usize,
    spf: i32,
) {
    let toc = packet[0];
    println!("\n  --- per-sub-frame split decode ---");

    // Set up persistent decoders
    let mut rust_dec = match OpusDecoder::new(sample_rate, channels) {
        Ok(d) => d,
        Err(e) => {
            println!("  rust dec init err {e}");
            return;
        }
    };
    let c_dec = unsafe {
        let mut error: c_int = 0;
        let d = bindings::opus_decoder_create(sample_rate, channels, &mut error);
        if d.is_null() || error != bindings::OPUS_OK {
            println!("  C dec init err {error}");
            return;
        }
        d
    };

    let mut sub_off = payload_off;
    let max_pcm = spf as usize * channels as usize;
    let mut rust_buf = vec![0i16; max_pcm];
    let mut c_buf = vec![0i16; max_pcm];

    let mut first_diff_seen = false;
    for (i, &sz) in sizes.iter().enumerate() {
        let sub_payload = &packet[sub_off..sub_off + sz as usize];
        sub_off += sz as usize;
        let sub_packet = synth_code0_packet(toc, sub_payload);

        let r_n = rust_dec
            .decode(Some(&sub_packet), &mut rust_buf, spf, false)
            .unwrap_or_else(|e| {
                println!("    [{i:2}] rust err {e}");
                -1
            });
        let c_n = unsafe {
            bindings::opus_decode(
                c_dec,
                sub_packet.as_ptr(),
                sub_packet.len() as i32,
                c_buf.as_mut_ptr(),
                spf,
                0,
            )
        };

        if r_n != c_n {
            println!("    [{i:2}] sz={sz:3} sample-count diff rust={r_n} c={c_n}");
            continue;
        }
        let n = r_n as usize * channels as usize;
        let eq = rust_buf[..n] == c_buf[..n];
        let tag = if eq { "==" } else { "!=" };
        let first_diff = if eq {
            String::new()
        } else {
            let idx = rust_buf[..n]
                .iter()
                .zip(c_buf[..n].iter())
                .position(|(a, b)| a != b)
                .unwrap();
            format!(" first_div@{idx} rust={} c={}", rust_buf[idx], c_buf[idx])
        };
        println!(
            "    [{i:2}] sz={sz:3} n={r_n:3} {tag}{}{}",
            first_diff,
            if !eq && !first_diff_seen {
                first_diff_seen = true;
                " <-- FIRST DIVERGE"
            } else {
                ""
            }
        );
    }

    unsafe { bindings::opus_decoder_destroy(c_dec) };
}

fn analyze(path: &Path) {
    println!("\n========================================");
    println!("file: {}", path.display());
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            println!("  ERR read: {e}");
            return;
        }
    };
    println!("  raw len = {} B", bytes.len());
    if bytes.len() < 3 {
        println!("  too small");
        return;
    }
    let sr_idx = (bytes[0] as usize) % SAMPLE_RATES.len();
    let sample_rate = SAMPLE_RATES[sr_idx];
    let channels: i32 = if bytes[1] & 1 == 0 { 1 } else { 2 };
    let packet = &bytes[2..];
    println!(
        "  header: sr_byte=0x{:02x} sr={} ch_byte=0x{:02x} ch={}",
        bytes[0], sample_rate, bytes[1], channels
    );
    println!("  packet len = {} B", packet.len());
    if packet.is_empty() {
        return;
    }
    println!("  TOC = 0x{:02x}", packet[0]);
    let spf = opus_packet_get_samples_per_frame(packet, sample_rate);
    let nf = opus_packet_get_nb_frames(packet).unwrap_or(-1);
    println!("  spf={} nf={}", spf, nf);
    let mut _split_args: Option<(Vec<i32>, usize, i32)> = None;
    if packet.len() >= 2 {
        let toc = packet[0];
        let code = toc & 0x3;
        let cfg = (toc >> 3) & 0x1f;
        let stereo = (toc >> 2) & 0x1;
        println!("  TOC: cfg={} stereo={} code={}", cfg, stereo, code);
        if code == 3 {
            let frame_byte = packet[1];
            let vbr = (frame_byte >> 7) & 1;
            let pad = (frame_byte >> 6) & 1;
            let m = frame_byte & 0x3f;
            println!(
                "  code-3 frame_byte=0x{:02x} vbr={} pad={} M={}",
                frame_byte, vbr, pad, m
            );
            if let Some((sizes, payload_off)) = parse_code3_vbr_sizes(packet) {
                println!("  payload_off={} sub-frame sizes:", payload_off);
                let mut acc = 0i32;
                for (i, sz) in sizes.iter().enumerate() {
                    let acc_before = acc;
                    acc += spf;
                    let dtx = if *sz <= 1 { " (DTX/PLC)" } else { "" };
                    println!(
                        "    [{:2}] size={:4} sample_range=[{:5}..{:5}){}",
                        i, sz, acc_before, acc, dtx
                    );
                }
                // After we report the main divergence below, also do per-sub-frame split decode.
                _split_args = Some((sizes, payload_off, spf));
            }
        }
    }

    let rust_ret = rust_decode(packet, sample_rate, channels);
    let c_ret = c_decode(packet, sample_rate, channels);

    match (&rust_ret, &c_ret) {
        (Ok(r), Ok(c)) => {
            let r_n = r.len() / channels as usize;
            let c_n = c.len() / channels as usize;
            println!("  rust: ok, samples={r_n} (raw_len={})", r.len());
            println!("  C   : ok, samples={c_n} (raw_len={})", c.len());
            if r.len() != c.len() {
                println!("  *** sample-count mismatch ***");
                return;
            }
            // Find first divergence (interleaved index)
            let div = r.iter().zip(c.iter()).position(|(a, b)| a != b);
            match div {
                None => println!("  PCM exactly equal."),
                Some(i) => {
                    let total = r.len();
                    println!("  *** FIRST DIVERGENCE at interleaved index {i} of {total}");
                    let sample_idx = i / channels as usize;
                    let ch_in_sample = i % channels as usize;
                    println!(
                        "      sample_idx={sample_idx}, ch={ch_in_sample}, total_samples={}",
                        total / channels as usize
                    );
                    // Context: 3 before, 8 after, both channels printed.
                    let lo = i.saturating_sub(3 * channels as usize);
                    let hi = (i + 8 * channels as usize).min(total);
                    println!("      context [{lo}..{hi}):");
                    for j in lo..hi {
                        let mark = if j == i {
                            " <-- DIVERGE"
                        } else if r[j] != c[j] {
                            " *"
                        } else {
                            ""
                        };
                        println!(
                            "        [{j:5}] rust={:6} c={:6} diff={:6}{}",
                            r[j],
                            c[j],
                            r[j] as i32 - c[j] as i32,
                            mark
                        );
                    }
                    // Count total mismatches
                    let n_diff = r.iter().zip(c.iter()).filter(|(a, b)| a != b).count();
                    println!("      total mismatched indices: {n_diff}/{total}");
                }
            }
        }
        (Err(e), Ok(_)) => println!("  rust ERR={e}, C ok"),
        (Ok(_), Err(e)) => println!("  rust ok, C ERR={e}"),
        (Err(re), Err(ce)) => println!("  both errored rust={re} c={ce}"),
    }

    // Per-sub-frame split-decode comparison
    if let Some((sizes, payload_off, spf)) = _split_args {
        split_decode(packet, sample_rate, channels, &sizes, payload_off, spf);

        // Reduce-test: try shrinking the PLC sequence and see if the bug
        // still appears. We feed: sub_frame[0], k PLC frames, sub_frame[last_real]
        // and check whether divergence appears.
        let toc = packet[0];
        // Find first non-DTX and last non-DTX sub-frames.
        let first_real = sizes.iter().position(|&s| s > 1);
        let last_real = sizes.iter().rposition(|&s| s > 1);
        if let (Some(fr), Some(lr)) = (first_real, last_real)
            && fr != lr
        {
            let mut off = payload_off;
            for &s in &sizes[..fr] {
                off += s as usize;
            }
            let first_payload = packet[off..off + sizes[fr] as usize].to_vec();
            let mut off2 = payload_off;
            for &s in &sizes[..lr] {
                off2 += s as usize;
            }
            let last_payload = packet[off2..off2 + sizes[lr] as usize].to_vec();
            println!("\n  --- minimal repro: 1 normal + k PLC + 1 normal ---");
            println!(
                "  first sub-frame [{}] sz={}, last [{}] sz={}",
                fr, sizes[fr], lr, sizes[lr]
            );
            // First, with k=1, dump full PCM and CELT state after each step.
            {
                println!("\n  --- step-by-step k=1 dump (with state) ---");
                let mut rd = OpusDecoder::new(sample_rate, channels).unwrap();
                let cd = unsafe {
                    let mut e: c_int = 0;
                    let d = bindings::opus_decoder_create(sample_rate, channels, &mut e);
                    assert!(!d.is_null() && e == 0);
                    d
                };
                let max_pcm = spf as usize * channels as usize;
                let mut rb = vec![0i16; max_pcm];
                let mut cb = vec![0i16; max_pcm];

                println!("  state at fresh decoder:");
                compare_celt_state("FRESH", &rd, cd, 21);

                let p1 = synth_code0_packet(toc, &first_payload);
                let _ = rd.decode(Some(&p1), &mut rb, spf, false);
                let _ = unsafe {
                    bindings::opus_decode(cd, p1.as_ptr(), p1.len() as i32, cb.as_mut_ptr(), spf, 0)
                };
                let n = spf as usize * channels as usize;
                let neq = rb[..n] == cb[..n];
                println!("  after p1 normal: pcm_eq={neq}");
                let _ = compare_celt_state("AFTER_P1", &rd, cd, 21);

                let plc1 = vec![toc & !0x3];
                let _ = rd.decode(Some(&plc1), &mut rb, spf, false);
                let _ = unsafe {
                    bindings::opus_decode(
                        cd,
                        plc1.as_ptr(),
                        plc1.len() as i32,
                        cb.as_mut_ptr(),
                        spf,
                        0,
                    )
                };
                let neq = rb[..n] == cb[..n];
                println!("  after PLC1     : pcm_eq={neq}");
                let _ = compare_celt_state("AFTER_PLC1", &rd, cd, 21);

                let p2 = synth_code0_packet(toc, &last_payload);
                let _ = rd.decode(Some(&p2), &mut rb, spf, false);
                let _ = unsafe {
                    bindings::opus_decode(cd, p2.as_ptr(), p2.len() as i32, cb.as_mut_ptr(), spf, 0)
                };
                let neq = rb[..n] == cb[..n];
                let first_diff = rb[..n].iter().zip(cb[..n].iter()).position(|(a, b)| a != b);
                println!("  after p2 normal: pcm_eq={neq} first_diff_idx={first_diff:?}");
                let _ = compare_celt_state("AFTER_P2", &rd, cd, 21);
                unsafe { bindings::opus_decoder_destroy(cd) };
            }

            for k in [0usize, 1, 2, 3, 4, 5, 8, 12].iter().copied() {
                let mut rd = OpusDecoder::new(sample_rate, channels).unwrap();
                let cd = unsafe {
                    let mut e: c_int = 0;
                    let d = bindings::opus_decoder_create(sample_rate, channels, &mut e);
                    assert!(!d.is_null() && e == 0);
                    d
                };
                let max_pcm = spf as usize * channels as usize;
                let mut rb = vec![0i16; max_pcm];
                let mut cb = vec![0i16; max_pcm];

                // 1 normal
                let p1 = synth_code0_packet(toc, &first_payload);
                let _ = rd.decode(Some(&p1), &mut rb, spf, false);
                let _ = unsafe {
                    bindings::opus_decode(cd, p1.as_ptr(), p1.len() as i32, cb.as_mut_ptr(), spf, 0)
                };

                // k PLC frames (use a 1-byte size-1 packet → triggers PLC inside decode_frame)
                // PLC is triggered by len <= 1, so feed a 1-byte packet.
                let plc1 = vec![toc & !0x3]; // single byte = TOC only, len=1
                for _ in 0..k {
                    let _ = rd.decode(Some(&plc1), &mut rb, spf, false);
                    let _ = unsafe {
                        bindings::opus_decode(
                            cd,
                            plc1.as_ptr(),
                            plc1.len() as i32,
                            cb.as_mut_ptr(),
                            spf,
                            0,
                        )
                    };
                }

                // Final normal
                let p2 = synth_code0_packet(toc, &last_payload);
                let r_n = rd.decode(Some(&p2), &mut rb, spf, false).unwrap_or(-1);
                let c_n = unsafe {
                    bindings::opus_decode(cd, p2.as_ptr(), p2.len() as i32, cb.as_mut_ptr(), spf, 0)
                };

                let n = r_n as usize * channels as usize;
                let eq = r_n == c_n && rb[..n] == cb[..n];
                let first_diff = if eq {
                    String::new()
                } else {
                    let i = rb[..n]
                        .iter()
                        .zip(cb[..n].iter())
                        .position(|(a, b)| a != b)
                        .unwrap_or(0);
                    format!(" first_div@{i} rust={} c={}", rb[i], cb[i])
                };
                println!("    k={k:2} {}{}", if eq { "==" } else { "!=" }, first_diff);

                unsafe { bindings::opus_decoder_destroy(cd) };
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: diff_fuzz_decode <crashfile> [<crashfile2> ...]");
        std::process::exit(2);
    }
    for arg in &args[1..] {
        analyze(Path::new(arg));
    }
}
