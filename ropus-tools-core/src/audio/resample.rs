//! Sample-rate conversion to the Opus native rate (48 kHz) via rubato's
//! `SincFixedIn` resampler.

use anyhow::{Result, anyhow, bail};

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use crate::consts::OPUS_SR;

pub fn resample_to_48k(interleaved: &[f32], from_sr: u32, channels: usize) -> Result<Vec<f32>> {
    if channels == 0 {
        bail!("zero-channel input");
    }
    if from_sr == 0 {
        bail!("zero-Hz input");
    }
    let ratio = OPUS_SR as f64 / from_sr as f64;

    let frames_in = interleaved.len() / channels;
    if frames_in == 0 {
        return Ok(Vec::new());
    }

    // Deinterleave into per-channel planar buffers.
    let mut planar: Vec<Vec<f32>> = (0..channels).map(|_| Vec::with_capacity(frames_in)).collect();
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

    // Pad each channel to a multiple of the resampler chunk size.
    let pad = chunk - (frames_in % chunk);
    if pad != chunk {
        for ch in &mut planar {
            ch.extend(std::iter::repeat_n(0.0f32, pad));
        }
    }

    let total_in = planar[0].len();
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
