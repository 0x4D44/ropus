//! One-shot fixture generator: writes a deterministic N-channel ambisonics WAV
//! for a given order.
//!
//! Usage: `rustc make_ambisonic_fixture.rs -o mk && ./mk <order> <output-path>`
//!
//! `order` is the ambisonics order (1 through 5), corresponding to:
//!   order 1 = FOA,  4 channels ( = (1+1)^2 )
//!   order 2 = SOA,  9 channels ( = (2+1)^2 )
//!   order 3 = TOA, 16 channels ( = (3+1)^2 )
//!   order 4 = 4HOA, 25 channels
//!   order 5 = 5HOA, 36 channels
//!
//! Note: the projection API uses `order_plus_one` internally, which equals
//! `order + 1`. We accept the user-facing ambisonics "order" here because
//! that is the standard terminology in ambisonics (FOA=1, SOA=2, etc.).
//!
//! Produces a 100 ms, 48 kHz, 16-bit PCM WAV. The W (omni) channel carries a
//! gentle sine sweep; remaining ACN channels carry small deterministic content
//! derived from the sample index and the channel index so that every fixture
//! is reproducible and every channel has non-trivial content.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <order 1..=5> [output-path]", args[0]);
        std::process::exit(1);
    }
    let order: u32 = args[1].parse().unwrap_or_else(|_| {
        eprintln!("ERROR: <order> must be an integer 1..=5");
        std::process::exit(1);
    });
    if !(1..=5).contains(&order) {
        eprintln!("ERROR: <order> must be 1..=5 (got {order})");
        std::process::exit(1);
    }
    let channels_u32: u32 = (order + 1) * (order + 1);
    let path = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| format!("ambisonic_order{order}_100ms.wav"));

    let sample_rate: u32 = 48000;
    let channels: u16 = channels_u32 as u16;
    let bits_per_sample: u16 = 16;
    // 100 ms: long enough to exercise multiple 20 ms Opus frames (5 × 20 ms
    // at 48 kHz) while keeping every fixture small.
    let duration_s: f64 = 0.1;
    let num_samples = (sample_rate as f64 * duration_s) as u32;

    // Synthesis scheme (deterministic, per-channel):
    //   ch 0 (W, omni):   sine sweep 200 Hz -> 800 Hz at -20 dBFS
    //   ch 1 (Y, ACN):    W * 0.5 * sin(phase * 0.5)
    //   ch 2 (Z, ACN):    W * 0.5 * cos(phase * 0.5)
    //   ch 3 (X, ACN):    W * 0.5 * sin(phase * 0.25)
    //   ch k (k >= 4):    small per-channel deterministic dither derived from
    //                     (n, k) so every higher-order channel has non-trivial
    //                     content but the amplitude is small enough that the
    //                     codec does not waste bits on pure noise.
    //
    // Using a single phase accumulator keeps the whole fixture reproducible
    // from (order, sample_rate, duration) alone.
    let mut samples: Vec<i16> = Vec::with_capacity((num_samples * channels as u32) as usize);
    let amp: f64 = 0.1 * 32767.0; // -20 dBFS
    let mut phase: f64 = 0.0;
    for n in 0..num_samples {
        let t = n as f64 / sample_rate as f64;
        let f_hz: f64 = 200.0 + 600.0 * t;
        phase += 2.0 * std::f64::consts::PI * f_hz / sample_rate as f64;

        let w = amp * phase.sin();
        for ch in 0..channels as u32 {
            let v: f64 = match ch {
                0 => w,
                1 => w * 0.5 * (phase * 0.5).sin(),
                2 => w * 0.5 * (phase * 0.5).cos(),
                3 => w * 0.5 * (phase * 0.25).sin(),
                _ => {
                    // Deterministic small dither: LCG-ish combination of
                    // channel and sample indices.
                    let mix: i64 = (n as i64) * 37 + (ch as i64) * 61;
                    let v = ((mix.rem_euclid(2048)) - 1024) as f64;
                    v // roughly -1024..=1023 out of 32767  (~ -30 dBFS)
                }
            };
            // saturate to i16
            let s = v.round().clamp(-32768.0, 32767.0) as i16;
            samples.push(s);
        }
    }

    let data_bytes: Vec<u8> = samples
        .iter()
        .flat_map(|s| s.to_le_bytes().to_vec())
        .collect();
    let data_len = data_bytes.len() as u32;
    let byte_rate: u32 = sample_rate * channels as u32 * (bits_per_sample as u32 / 8);
    let block_align: u16 = channels * (bits_per_sample / 8);

    let mut wav: Vec<u8> = Vec::with_capacity(44 + data_bytes.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36u32 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&bits_per_sample.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.extend_from_slice(&data_bytes);

    std::fs::write(&path, &wav).expect("write wav");
    println!(
        "Wrote {} (order={}, {} channels, {} bytes)",
        path,
        order,
        channels,
        wav.len()
    );
}
