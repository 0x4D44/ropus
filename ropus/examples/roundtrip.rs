//! WAV in → encode → decode → WAV out, in a single process.
//!
//! Usage: `cargo run --example roundtrip -- input.wav output.wav --bitrate 64000`
//!
//! Reports SNR of the centre region of the decoded output vs the original
//! PCM. Opus is lossy and has algorithmic latency, so SNR will be finite and
//! we skip the first samples to avoid the encoder's pre-roll.

use std::env;
use std::error::Error;

use ropus::{Application, Bitrate, Channels, Decoder, Encoder};

// `#[path]` is needed because Cargo's example auto-discovery treats every
// file under examples/ as a binary target unless it's in a subdirectory.
#[path = "common/wav.rs"]
mod wav;

const FRAME_MS: u32 = 20;
const COMPLEXITY: u8 = 10;
const MAX_PACKET: usize = 4000;
/// 120 ms at 48 kHz — the largest frame Opus can emit.
const MAX_FRAME_PER_CHANNEL: usize = 5_760;
/// Skip this many samples per channel from the start when measuring SNR, to
/// step past Opus's algorithmic look-ahead / pre-roll. 1000 samples at 48 kHz
/// is ~21 ms, comfortably past the codec's ~6.5 ms look-ahead.
const SNR_WARMUP_SAMPLES: usize = 1000;
/// Maximum delay the SNR aligner will search for, per channel. Opus's reported
/// look-ahead at 48 kHz is ~312 samples; 2048 gives plenty of headroom for
/// any mode the encoder picks (low-bitrate SILK modes can push the optimal
/// alignment well past the 1k mark).
const SNR_MAX_LAG_SAMPLES: usize = 2048;

fn main() -> Result<(), Box<dyn Error>> {
    let (in_path, out_path, bitrate) = parse_args()?;

    let input = wav::read(&in_path)?;
    let channels = match input.channels {
        1 => Channels::Mono,
        2 => Channels::Stereo,
        n => return Err(format!("unsupported channel count {n}").into()),
    };
    let ch = channels.count();
    let frame_per_channel = (input.sample_rate * FRAME_MS / 1000) as usize;
    let frame_interleaved = frame_per_channel * ch;

    println!(
        "input:  {} Hz, {} ch, {} samples ({:.3} s)",
        input.sample_rate,
        ch,
        input.frames(),
        input.duration_secs()
    );

    let mut encoder = Encoder::builder(input.sample_rate, channels, Application::Audio)
        .bitrate(Bitrate::Bits(bitrate))
        .complexity(COMPLEXITY)
        .build()?;
    let mut decoder = Decoder::new(input.sample_rate, channels)?;

    let mut packet = vec![0u8; MAX_PACKET];
    let mut decoded_frame = vec![0i16; MAX_FRAME_PER_CHANNEL * ch];
    let mut decoded: Vec<i16> = Vec::with_capacity(input.samples.len());
    let mut packets: usize = 0;

    let mut pcm_frame = vec![0i16; frame_interleaved];
    let mut idx = 0usize;
    while idx < input.samples.len() {
        let take = (input.samples.len() - idx).min(frame_interleaved);
        pcm_frame[..take].copy_from_slice(&input.samples[idx..idx + take]);
        if take < frame_interleaved {
            pcm_frame[take..].fill(0);
        }
        idx += take;

        let n = encoder.encode(&pcm_frame, &mut packet)?;
        let samples_per_channel = decoder.decode(&packet[..n], &mut decoded_frame, false)?;
        decoded.extend_from_slice(&decoded_frame[..samples_per_channel * ch]);
        packets += 1;
    }

    let decoded_frames = decoded.len() / ch;
    println!(
        "output: {} packets, {} samples ({:.3} s)",
        packets,
        decoded_frames,
        decoded_frames as f64 / input.sample_rate as f64
    );

    // Opus shifts the decoded output by its algorithmic look-ahead, so a naive
    // sample-aligned compare gives garbage SNR for periodic signals like a
    // sine wave. Search for the best alignment within a small lag window.
    let snr = compute_snr_db(
        &input.samples,
        &decoded,
        ch,
        SNR_WARMUP_SAMPLES,
        SNR_MAX_LAG_SAMPLES,
    );
    match snr {
        Some((db, lag)) => {
            println!(
                "SNR (centre region, skip {} samples, best lag {} samples): {:.2} dB",
                SNR_WARMUP_SAMPLES, lag, db
            );
            if lag == SNR_MAX_LAG_SAMPLES {
                eprintln!(
                    "warning: SNR lag search saturated at {SNR_MAX_LAG_SAMPLES}; reported SNR may underestimate quality"
                );
            }
        }
        None => println!("SNR: not enough overlap to compute"),
    }

    wav::write(&out_path, input.sample_rate, ch as u16, &decoded)?;
    println!("wrote:  {}", out_path);
    Ok(())
}

fn parse_args() -> Result<(String, String, u32), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let usage = || {
        format!(
            "usage: {} <input.wav> <output.wav> --bitrate <bps>",
            args.first().map(String::as_str).unwrap_or("roundtrip")
        )
    };
    if args.len() != 5 || args[3] != "--bitrate" {
        return Err(usage().into());
    }
    let bitrate: u32 = args[4]
        .parse()
        .map_err(|e| format!("invalid --bitrate {:?}: {e}", args[4]))?;
    Ok((args[1].clone(), args[2].clone(), bitrate))
}

/// Sum-of-squares SNR (dB) of `decoded` against `orig`, with a small lag
/// search to compensate for Opus's algorithmic delay. Returns the best SNR
/// and the lag (in per-channel samples) at which it was measured. Returns
/// `None` if the overlap window is empty.
///
/// `decoded[lag*ch ..]` is compared against `orig[warmup*ch ..]`, sweeping
/// `lag` over `0..=max_lag_per_channel`.
fn compute_snr_db(
    orig: &[i16],
    decoded: &[i16],
    ch: usize,
    warmup_per_channel: usize,
    max_lag_per_channel: usize,
) -> Option<(f64, usize)> {
    let warmup = warmup_per_channel * ch;
    let mut best: Option<(f64, usize)> = None;

    for lag_per_ch in 0..=max_lag_per_channel {
        let lag = lag_per_ch * ch;
        // Aligned window length capped by both buffers.
        let usable = orig.len().saturating_sub(warmup).min(
            decoded.len().saturating_sub(warmup + lag),
        );
        if usable == 0 {
            continue;
        }
        let mut signal_power = 0.0f64;
        let mut noise_power = 0.0f64;
        for i in 0..usable {
            let s = orig[warmup + i] as f64;
            let d = decoded[warmup + lag + i] as f64;
            signal_power += s * s;
            let e = s - d;
            noise_power += e * e;
        }
        if signal_power == 0.0 {
            // All-silence input — SNR is undefined.
            return None;
        }
        let snr = if noise_power == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (signal_power / noise_power).log10()
        };
        match best {
            Some((prev, _)) if snr <= prev => {}
            _ => best = Some((snr, lag_per_ch)),
        }
    }
    best
}
