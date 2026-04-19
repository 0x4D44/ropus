//! PCM-level drift measurement: encode the same input with both the C
//! reference and the Rust port, decode each with the C reference decoder,
//! and compare the decoded PCM streams sample-by-sample.
//!
//! Answers the question: does the tier-1 byte-exact residual (94.9% on
//! music, 98.6% on speech per Stage 6 journal) translate to audible audio
//! quality loss, or do different-but-equally-valid bitstreams decode to
//! perceptually equivalent PCM?
//!
//! Both encoders use the same default profile as `ropus-compare encode`:
//! 64 kbps CBR, complexity 10, 20 ms frames, application AUDIO. Each
//! packet stream is fed to its own C-reference decoder instance so the
//! decoder state evolves independently per stream.

#![allow(clippy::too_many_arguments)]

use ropus_harness::bindings;

use std::ffi::c_int;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

// ---------------------------------------------------------------------------
// Minimal WAV reader (16-bit PCM only)
// ---------------------------------------------------------------------------

struct Wav {
    sample_rate: u32,
    channels: u16,
    samples: Vec<i16>,
}

fn read_wav(path: &Path) -> Wav {
    let data = fs::read(path).unwrap_or_else(|e| {
        eprintln!("ERROR: cannot read {}: {}", path.display(), e);
        process::exit(1);
    });
    assert!(data.len() >= 44, "WAV too small");
    assert_eq!(&data[0..4], b"RIFF");
    assert_eq!(&data[8..12], b"WAVE");

    let mut pos = 12;
    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut bits_per_sample = 0u16;
    let mut pcm: Vec<i16> = Vec::new();

    while pos + 8 <= data.len() {
        let id = &data[pos..pos + 4];
        let sz = u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]])
            as usize;
        if id == b"fmt " {
            channels = u16::from_le_bytes([data[pos + 10], data[pos + 11]]);
            sample_rate = u32::from_le_bytes([
                data[pos + 12],
                data[pos + 13],
                data[pos + 14],
                data[pos + 15],
            ]);
            bits_per_sample = u16::from_le_bytes([data[pos + 22], data[pos + 23]]);
        } else if id == b"data" {
            assert_eq!(bits_per_sample, 16, "only 16-bit PCM supported");
            let bytes = &data[pos + 8..pos + 8 + sz];
            pcm = bytes
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect();
        }
        pos += 8 + sz;
        if !sz.is_multiple_of(2) {
            pos += 1;
        }
    }
    assert!(sample_rate > 0, "no fmt chunk");
    assert!(!pcm.is_empty(), "no data chunk");
    Wav {
        sample_rate,
        channels,
        samples: pcm,
    }
}

// ---------------------------------------------------------------------------
// Apply matching CTL config to C encoder.
// ---------------------------------------------------------------------------

fn configure_c_encoder(enc: *mut bindings::OpusEncoder, bitrate: i32, complexity: i32) {
    unsafe {
        macro_rules! ctl {
            ($req:expr, $val:expr) => {{
                let ret = bindings::opus_encoder_ctl(enc, $req, $val);
                assert_eq!(ret, bindings::OPUS_OK, "C CTL {} failed: {}", $req, ret);
            }};
        }
        ctl!(bindings::OPUS_SET_BITRATE_REQUEST, bitrate);
        ctl!(bindings::OPUS_SET_COMPLEXITY_REQUEST, complexity);
        ctl!(bindings::OPUS_SET_VBR_REQUEST, 0_i32);
        ctl!(bindings::OPUS_SET_VBR_CONSTRAINT_REQUEST, 0_i32);
        ctl!(bindings::OPUS_SET_INBAND_FEC_REQUEST, 0_i32);
        ctl!(bindings::OPUS_SET_PACKET_LOSS_PERC_REQUEST, 0_i32);
        ctl!(bindings::OPUS_SET_DTX_REQUEST, 0_i32);
        ctl!(bindings::OPUS_SET_SIGNAL_REQUEST, bindings::OPUS_AUTO);
        ctl!(bindings::OPUS_SET_BANDWIDTH_REQUEST, bindings::OPUS_AUTO);
        ctl!(
            bindings::OPUS_SET_FORCE_CHANNELS_REQUEST,
            bindings::OPUS_AUTO
        );
        ctl!(
            bindings::OPUS_SET_MAX_BANDWIDTH_REQUEST,
            bindings::OPUS_BANDWIDTH_FULLBAND
        );
        ctl!(bindings::OPUS_SET_LSB_DEPTH_REQUEST, 24_i32);
        ctl!(bindings::OPUS_SET_PREDICTION_DISABLED_REQUEST, 0_i32);
        ctl!(bindings::OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, 0_i32);
        ctl!(bindings::OPUS_SET_FORCE_MODE_REQUEST, bindings::OPUS_AUTO);
    }
}

fn configure_rust_encoder(
    enc: &mut ropus::opus::encoder::OpusEncoder,
    bitrate: i32,
    complexity: i32,
) {
    enc.set_bitrate(bitrate);
    enc.set_complexity(complexity);
    enc.set_vbr(0);
    enc.set_vbr_constraint(0);
    enc.set_inband_fec(0);
    enc.set_packet_loss_perc(0);
    enc.set_dtx(0);
    enc.set_signal(bindings::OPUS_AUTO);
    enc.set_bandwidth(bindings::OPUS_AUTO);
    enc.set_force_channels(bindings::OPUS_AUTO);
    enc.set_max_bandwidth(bindings::OPUS_BANDWIDTH_FULLBAND);
    enc.set_lsb_depth(24);
    enc.set_prediction_disabled(0);
    enc.set_phase_inversion_disabled(0);
    enc.set_force_mode(bindings::OPUS_AUTO);
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

struct Metrics {
    label: String,
    frames_total: usize,
    frames_packets_identical: usize,
    frames_pcm_identical: usize,
    first_pkt_divergence_frame: Option<usize>,
    first_pcm_divergence_frame: Option<usize>,
    samples_total: usize,
    samples_differ: usize,
    max_abs_diff: i32,
    sum_sq_err: f64,        // (r_dec - c_dec)^2
    sum_sq_ref: f64,        // c_dec^2 (reference signal for differential SNR)
    sum_sq_input: f64,      // input^2 (reference for codec-transparency SNR)
    sum_sq_c_vs_input: f64, // (c_dec - input_aligned)^2
    sum_sq_r_vs_input: f64, // (r_dec - input_aligned)^2
    samples_vs_input: usize,
    max_abs_ref: i32,
    c_total_bytes: usize,
    r_total_bytes: usize,
    worst_frames: Vec<(usize, f64, i32)>, // (frame, rms, max_abs_diff)
}

impl Metrics {
    fn new(label: String) -> Self {
        Self {
            label,
            frames_total: 0,
            frames_packets_identical: 0,
            frames_pcm_identical: 0,
            first_pkt_divergence_frame: None,
            first_pcm_divergence_frame: None,
            samples_total: 0,
            samples_differ: 0,
            max_abs_diff: 0,
            sum_sq_err: 0.0,
            sum_sq_ref: 0.0,
            sum_sq_input: 0.0,
            sum_sq_c_vs_input: 0.0,
            sum_sq_r_vs_input: 0.0,
            samples_vs_input: 0,
            max_abs_ref: 0,
            c_total_bytes: 0,
            r_total_bytes: 0,
            worst_frames: Vec::new(),
        }
    }

    fn report(&self, duration_s: f64) {
        let rms_err = (self.sum_sq_err / self.samples_total as f64).sqrt();
        let rms_ref = (self.sum_sq_ref / self.samples_total as f64).sqrt();
        let snr_db = if rms_err > 0.0 {
            20.0 * (rms_ref / rms_err).log10()
        } else {
            f64::INFINITY
        };
        let frames = self.frames_total as f64;
        let c_bitrate = (self.c_total_bytes * 8) as f64 / duration_s;
        let r_bitrate = (self.r_total_bytes * 8) as f64 / duration_s;
        let diff_pct = 100.0 * self.samples_differ as f64 / self.samples_total as f64;
        let peak_vs_ref_pct = 100.0 * self.max_abs_diff as f64 / self.max_abs_ref.max(1) as f64;

        println!("\n====== {} ======", self.label);
        println!(
            "Frames: {} ({:.2} s at 48 kHz / 20 ms frames)",
            self.frames_total, duration_s
        );
        println!(
            "Packets byte-identical   : {}/{} ({:.2}%)",
            self.frames_packets_identical,
            self.frames_total,
            100.0 * self.frames_packets_identical as f64 / frames
        );
        println!(
            "Decoded frames identical : {}/{} ({:.2}%)",
            self.frames_pcm_identical,
            self.frames_total,
            100.0 * self.frames_pcm_identical as f64 / frames
        );
        match self.first_pkt_divergence_frame {
            Some(f) => println!("First packet divergence  : frame {f}"),
            None => println!("First packet divergence  : (none — packets bit-exact)"),
        }
        match self.first_pcm_divergence_frame {
            Some(f) => println!("First PCM divergence     : frame {f}"),
            None => println!("First PCM divergence     : (none — PCM bit-exact)"),
        }
        println!(
            "Samples: {} total, {} differing ({:.3}%)",
            self.samples_total, self.samples_differ, diff_pct
        );
        println!(
            "Peak abs diff            : {} LSB  ({:.3}% of peak signal {})",
            self.max_abs_diff, peak_vs_ref_pct, self.max_abs_ref
        );
        println!("RMS error                : {:.2} LSB", rms_err);
        println!("RMS reference            : {:.2} LSB", rms_ref);
        println!("SNR (ref / err)          : {:.2} dB", snr_db);
        println!(
            "Bitrate  C: {:.0} bps  |  Rust: {:.0} bps  |  delta: {:+.3}%",
            c_bitrate,
            r_bitrate,
            100.0 * (r_bitrate - c_bitrate) / c_bitrate
        );

        // --- Three-way: input vs c_decoded vs r_decoded ---
        // This is the perceptually meaningful comparison: even if C and
        // Rust disagree, what matters is whether Rust's encoding is a
        // worse approximation of the original input than C's.
        if self.samples_vs_input > 0 {
            let rms_input = (self.sum_sq_input / self.samples_vs_input as f64).sqrt();
            let rms_c_err = (self.sum_sq_c_vs_input / self.samples_vs_input as f64).sqrt();
            let rms_r_err = (self.sum_sq_r_vs_input / self.samples_vs_input as f64).sqrt();
            let snr_c = if rms_c_err > 0.0 {
                20.0 * (rms_input / rms_c_err).log10()
            } else {
                f64::INFINITY
            };
            let snr_r = if rms_r_err > 0.0 {
                20.0 * (rms_input / rms_r_err).log10()
            } else {
                f64::INFINITY
            };
            println!("\nCodec transparency (vs original input WAV, 312-sample pre-skip applied):");
            println!(
                "  C ref  : SNR {:>6.2} dB   (RMS err {:.2} LSB)",
                snr_c, rms_c_err
            );
            println!(
                "  Rust   : SNR {:>6.2} dB   (RMS err {:.2} LSB)",
                snr_r, rms_r_err
            );
            let delta = snr_r - snr_c;
            println!(
                "  Delta  : {:+.3} dB  ({})",
                delta,
                if delta.abs() < 0.1 {
                    "equivalent"
                } else if delta > 0.0 {
                    "Rust marginally better"
                } else {
                    "Rust marginally worse"
                }
            );
        }
        if !self.worst_frames.is_empty() {
            println!("Worst frames by RMS error:");
            println!("  frame       rms_err   max_abs_diff");
            for (f, rms, max_d) in &self.worst_frames {
                println!("  {:>6}   {:>10.2}   {:>10}", f, rms, max_d);
            }
        }
    }
}

fn run_one(
    label: &str,
    wav_path: &Path,
    max_frames: Option<usize>,
    bitrate: i32,
    complexity: i32,
) -> Metrics {
    let wav = read_wav(wav_path);
    assert_eq!(wav.sample_rate, 48_000, "expected 48 kHz WAV");
    let sr = wav.sample_rate as i32;
    let ch = wav.channels as i32;
    let frame_size = 960usize; // 20 ms at 48 kHz
    let samples_per_frame = frame_size * wav.channels as usize;
    let max_packet = 4000usize;

    let total_frames_available = wav.samples.len() / samples_per_frame;
    let total_frames = match max_frames {
        Some(n) => n.min(total_frames_available),
        None => total_frames_available,
    };

    // --- Create encoders + decoders ---
    let mut err: c_int = 0;
    let c_enc = unsafe {
        bindings::opus_encoder_create(sr, ch, bindings::OPUS_APPLICATION_AUDIO, &mut err)
    };
    assert!(
        !c_enc.is_null() && err == bindings::OPUS_OK,
        "C enc create failed"
    );
    configure_c_encoder(c_enc, bitrate, complexity);

    let mut r_enc =
        ropus::opus::encoder::OpusEncoder::new(sr, ch, bindings::OPUS_APPLICATION_AUDIO)
            .expect("Rust enc create");
    configure_rust_encoder(&mut r_enc, bitrate, complexity);

    let c_dec_a = unsafe { bindings::opus_decoder_create(sr, ch, &mut err) };
    assert!(
        !c_dec_a.is_null() && err == bindings::OPUS_OK,
        "C dec A create failed"
    );
    let c_dec_b = unsafe { bindings::opus_decoder_create(sr, ch, &mut err) };
    assert!(
        !c_dec_b.is_null() && err == bindings::OPUS_OK,
        "C dec B create failed"
    );

    let mut metrics = Metrics::new(label.to_string());
    let mut c_pkt = vec![0u8; max_packet];
    let mut r_pkt = vec![0u8; max_packet];
    let mut c_pcm = vec![0i16; samples_per_frame];
    let mut r_pcm = vec![0i16; samples_per_frame];

    // Buffer all decoded PCM for end-of-stream alignment vs input.
    // Opus FB has 312-sample pre-skip per channel at 48 kHz.
    let mut c_pcm_all: Vec<i16> = Vec::with_capacity(total_frames * samples_per_frame);
    let mut r_pcm_all: Vec<i16> = Vec::with_capacity(total_frames * samples_per_frame);

    // We use a min-heap surrogate by inserting-then-trimming for top 5 worst frames.
    let mut per_frame_rms: Vec<(usize, f64, i32)> = Vec::with_capacity(total_frames);

    for f in 0..total_frames {
        let start = f * samples_per_frame;
        let end = start + samples_per_frame;
        let pcm_in = &wav.samples[start..end];

        // Encode with C
        let c_len = unsafe {
            bindings::opus_encode(
                c_enc,
                pcm_in.as_ptr(),
                frame_size as c_int,
                c_pkt.as_mut_ptr(),
                max_packet as bindings::opus_int32,
            )
        };
        assert!(c_len > 0, "C encode failed at frame {f}: {c_len}");
        let c_len = c_len as usize;

        // Encode with Rust
        let r_len = match r_enc.encode(pcm_in, frame_size as i32, &mut r_pkt, max_packet as i32) {
            Ok(n) => n as usize,
            Err(code) => panic!("Rust encode failed at frame {f}: {code}"),
        };

        metrics.c_total_bytes += c_len;
        metrics.r_total_bytes += r_len;

        let pkt_equal = c_len == r_len && c_pkt[..c_len] == r_pkt[..r_len];
        if pkt_equal {
            metrics.frames_packets_identical += 1;
        } else if metrics.first_pkt_divergence_frame.is_none() {
            metrics.first_pkt_divergence_frame = Some(f);
        }

        // Decode C packet with C decoder A
        let c_dec_len = unsafe {
            bindings::opus_decode(
                c_dec_a,
                c_pkt.as_ptr(),
                c_len as bindings::opus_int32,
                c_pcm.as_mut_ptr(),
                frame_size as c_int,
                0,
            )
        };
        assert_eq!(c_dec_len, frame_size as c_int, "C decode wrong frame size");

        // Decode Rust packet with C decoder B
        let r_dec_len = unsafe {
            bindings::opus_decode(
                c_dec_b,
                r_pkt.as_ptr(),
                r_len as bindings::opus_int32,
                r_pcm.as_mut_ptr(),
                frame_size as c_int,
                0,
            )
        };
        assert_eq!(
            r_dec_len, frame_size as c_int,
            "Rust-packet decode wrong frame size"
        );

        // Compare PCM
        let mut frame_differs = false;
        let mut frame_ss_err = 0.0f64;
        let mut frame_max_abs_diff = 0i32;
        for i in 0..samples_per_frame {
            let c_s = c_pcm[i] as i32;
            let r_s = r_pcm[i] as i32;
            let diff = r_s - c_s;
            let abs_diff = diff.abs();
            let abs_ref = c_s.abs();

            metrics.sum_sq_err += (diff as f64) * (diff as f64);
            metrics.sum_sq_ref += (c_s as f64) * (c_s as f64);
            if abs_ref > metrics.max_abs_ref {
                metrics.max_abs_ref = abs_ref;
            }
            if abs_diff > 0 {
                metrics.samples_differ += 1;
                frame_differs = true;
                if abs_diff > metrics.max_abs_diff {
                    metrics.max_abs_diff = abs_diff;
                }
                if abs_diff > frame_max_abs_diff {
                    frame_max_abs_diff = abs_diff;
                }
                frame_ss_err += (diff as f64) * (diff as f64);
            }
        }
        metrics.samples_total += samples_per_frame;
        if !frame_differs {
            metrics.frames_pcm_identical += 1;
        } else if metrics.first_pcm_divergence_frame.is_none() {
            metrics.first_pcm_divergence_frame = Some(f);
        }
        let frame_rms = (frame_ss_err / samples_per_frame as f64).sqrt();
        per_frame_rms.push((f, frame_rms, frame_max_abs_diff));
        metrics.frames_total += 1;

        // Retain full decoded streams for alignment-vs-input comparison.
        c_pcm_all.extend_from_slice(&c_pcm);
        r_pcm_all.extend_from_slice(&r_pcm);
    }

    unsafe {
        bindings::opus_encoder_destroy(c_enc);
        bindings::opus_decoder_destroy(c_dec_a);
        bindings::opus_decoder_destroy(c_dec_b);
    }

    // --- Align decoded vs input (skip pre-skip samples) ---
    // Opus FB pre-skip is 312 samples / channel at 48 kHz.
    const PRESKIP_PER_CH: usize = 312;
    let preskip = PRESKIP_PER_CH * wav.channels as usize;
    let total_decoded = c_pcm_all.len(); // same as r_pcm_all.len()
    let input_used = &wav.samples[..total_frames * samples_per_frame];
    if total_decoded > preskip {
        let cmp_len = (total_decoded - preskip).min(input_used.len() - preskip);
        for i in 0..cmp_len {
            let x = input_used[i] as f64;
            let c = c_pcm_all[i + preskip] as f64;
            let r = r_pcm_all[i + preskip] as f64;
            let dc = c - x;
            let dr = r - x;
            metrics.sum_sq_input += x * x;
            metrics.sum_sq_c_vs_input += dc * dc;
            metrics.sum_sq_r_vs_input += dr * dr;
        }
        metrics.samples_vs_input = cmp_len;
    }

    per_frame_rms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    metrics.worst_frames = per_frame_rms.into_iter().take(5).collect();
    metrics
}

pub fn main() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let vectors = PathBuf::from(manifest_dir)
        .join("..")
        .join("tests")
        .join("vectors");

    // Same profile as ropus-compare encode defaults (which produced the
    // 94.9% music / 98.6% speech byte-exact headline).
    let bitrate = 64_000;
    let complexity = 10;

    // Frame cap (optional). Set via env var PCM_DRIFT_FRAMES for a quick
    // sanity run. Default = process the whole clip.
    let max_frames = std::env::var("PCM_DRIFT_FRAMES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());

    let music = vectors.join("music_48k_stereo.wav");
    let speech = vectors.join("speech_48k_mono.wav");

    let cases = [
        ("music (48 kHz stereo)", music),
        ("speech (48 kHz mono)", speech),
    ];

    for (label, path) in cases {
        if !path.exists() {
            eprintln!("SKIP {}: {} missing", label, path.display());
            continue;
        }
        let m = run_one(label, &path, max_frames, bitrate, complexity);
        let duration_s = (m.frames_total as f64) * 0.020; // 20 ms frames
        m.report(duration_s);
    }

    println!(
        "\nProfile: {} bps CBR, complexity {}, 20 ms, AUDIO, FB",
        bitrate, complexity
    );
    println!("Each Rust packet is decoded by an independent C reference decoder,");
    println!("so state drift is isolated from the Rust decoder path.");
}
