//! Interleaved f32 sample-rate conversion via rubato's `SincFixedIn`.
//!
//! Generalised to an arbitrary target rate so the encoder (always 48 kHz) and
//! decoder (driven by `--rate HZ`, default 48 kHz) share one implementation.
//! Callers pass both source and target explicitly — no hidden 48 kHz default.

use anyhow::{Result, anyhow, bail};

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

pub fn resample(
    interleaved: &[f32],
    from_sr: u32,
    to_sr: u32,
    channels: usize,
) -> Result<Vec<f32>> {
    if channels == 0 {
        bail!("zero-channel input");
    }
    if from_sr == 0 {
        bail!("zero-Hz input");
    }
    if to_sr == 0 {
        bail!("zero-Hz target");
    }
    let ratio = to_sr as f64 / from_sr as f64;

    let frames_in = interleaved.len() / channels;
    if frames_in == 0 {
        return Ok(Vec::new());
    }

    // Deinterleave into per-channel planar buffers.
    let mut planar: Vec<Vec<f32>> = (0..channels)
        .map(|_| Vec::with_capacity(frames_in))
        .collect();
    for frame in interleaved.chunks_exact(channels) {
        for (ch, s) in frame.iter().enumerate() {
            planar[ch].push(*s);
        }
    }

    // Choose a chunk size that comfortably fits multiple sinc kernels.
    let chunk = 1024usize;

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::Blackman2,
    };

    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, chunk, channels)
        .map_err(|e| anyhow!("rubato construction failed: {e}"))?;
    // SincFixedIn warms up by emitting `output_delay()` frames of leading
    // silence before any real output. We must skip those frames or the
    // decoded WAV starts with a click/shifted silence.
    let resampler_delay = resampler.output_delay();

    // Pad each channel so the *output* covers the warm-up plus the target
    // length. Without this, a single 20 ms Opus frame (960 in) resampled
    // 48→44.1 can have its entire 882-frame target swallowed by the
    // warm-up delay, leaving zero real output. We over-pad by `resampler_delay
    // / ratio` input frames (silence) so the resampler has enough lead-in to
    // emit the full target. Then round up to a chunk multiple so `process`
    // never sees a short final block.
    let warmup_in = (resampler_delay as f64 / ratio).ceil() as usize;
    let needed_in = frames_in + warmup_in;
    let rem = needed_in % chunk;
    let total_in = if rem == 0 {
        needed_in
    } else {
        needed_in + (chunk - rem)
    };
    let pad = total_in - frames_in;
    if pad > 0 {
        for ch in &mut planar {
            ch.extend(std::iter::repeat_n(0.0f32, pad));
        }
    }

    let mut out_planar: Vec<Vec<f32>> = (0..channels).map(|_| Vec::new()).collect();

    let mut pos = 0;
    while pos < total_in {
        let end = (pos + chunk).min(total_in);
        // Slices for this chunk.
        let chunk_in: Vec<&[f32]> = planar.iter().map(|c| &c[pos..end]).collect();
        let processed = resampler
            .process(&chunk_in, None)
            .map_err(|e| anyhow!("rubato process failed: {e}"))?;
        for (ch, samples) in processed.into_iter().enumerate() {
            out_planar[ch].extend_from_slice(&samples);
        }
        pos = end;
    }

    // Drop the leading resampler delay (silence warm-up), then trim to the
    // expected output length (frames_in * ratio rounded). If the resampler
    // produced fewer frames than expected (input shorter than the warm-up
    // window), cap target_frames to what we actually have.
    let target_frames = (frames_in as f64 * ratio).round() as usize;
    let available = out_planar[0].len().saturating_sub(resampler_delay);
    let out_frames = target_frames.min(available);
    let start = resampler_delay.min(out_planar[0].len());

    // Interleave from out_planar[ch][start .. start + out_frames].
    let mut out = Vec::with_capacity(out_frames * channels);
    for f in 0..out_frames {
        out.extend(out_planar.iter().map(|chan| chan[start + f]));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_generalised_48k_to_44100_has_expected_length() {
        // Generate a short 1 kHz sine at 48 kHz and resample to 44.1 kHz.
        // Output frame count should be approximately `frames_in * 44100/48000`,
        // within ±1% to allow for resampler warm-up trimming.
        let sr_in: u32 = 48_000;
        let sr_out: u32 = 44_100;
        let seconds = 1;
        let frames_in = sr_in as usize * seconds;
        let channels = 1;

        let two_pi = std::f32::consts::TAU;
        let samples: Vec<f32> = (0..frames_in)
            .map(|n| (two_pi * 1000.0 * n as f32 / sr_in as f32).sin() * 0.5)
            .collect();

        let out = resample(&samples, sr_in, sr_out, channels).expect("resample");

        let expected = sr_out as usize * seconds;
        let diff = out.len().abs_diff(expected);
        let tolerance = expected / 100; // ±1 %
        assert!(
            diff <= tolerance,
            "resampled frame count {} out of tolerance (expected {}, diff {})",
            out.len(),
            expected,
            diff
        );
    }

    #[test]
    fn resample_identity_rate_round_trips_length() {
        // Source rate == target rate: the body still runs through rubato (no
        // short-circuit in this function), so we just verify the length lands
        // inside the standard resampler warm-up tolerance.
        let sr: u32 = 48_000;
        let samples = vec![0.1_f32; sr as usize];
        let out = resample(&samples, sr, sr, 1).expect("identity resample");
        let diff = out.len().abs_diff(samples.len());
        assert!(
            diff <= samples.len() / 100,
            "identity resample length drift too large: in={}, out={}",
            samples.len(),
            out.len()
        );
    }

    #[test]
    fn resample_rejects_zero_args() {
        assert!(resample(&[1.0_f32], 0, 48_000, 1).is_err());
        assert!(resample(&[1.0_f32], 48_000, 0, 1).is_err());
        assert!(resample(&[1.0_f32], 48_000, 48_000, 0).is_err());
    }

    #[test]
    fn resample_one_opus_frame_48k_to_44100_produces_nonempty_output() {
        // Regression: a single 20 ms Opus frame (960 samples @ 48 kHz) must
        // produce ~882 samples @ 44.1 kHz. Before padding for warm-up delay,
        // the SincFixedIn output-delay (~hundreds of frames) swallowed the
        // full target window for this short input, producing zero bytes.
        let sr_in: u32 = 48_000;
        let sr_out: u32 = 44_100;
        let channels = 1;
        let frames_in = 960usize;

        let two_pi = std::f32::consts::TAU;
        let samples: Vec<f32> = (0..frames_in)
            .map(|n| (two_pi * 1000.0 * n as f32 / sr_in as f32).sin() * 0.5)
            .collect();

        let out = resample(&samples, sr_in, sr_out, channels).expect("resample one frame");

        // Expected ~882 frames. Accept ±1% to tolerate rounding around the
        // trim boundary; what we really care about is "not zero".
        let expected = (frames_in as f64 * (sr_out as f64 / sr_in as f64)).round() as usize;
        let diff = out.len().abs_diff(expected);
        assert!(
            !out.is_empty(),
            "one-frame 48→44.1 kHz resample returned zero samples"
        );
        assert!(
            diff <= expected / 100 + 1,
            "one-frame 48→44.1 kHz resample length {} off from expected {} (diff {})",
            out.len(),
            expected,
            diff
        );
    }
}
