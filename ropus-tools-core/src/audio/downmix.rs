//! Stereo → mono downmix helper used by `ropusenc --downmix mono`.
//!
//! Only `L + R` averaging is supported. opus-tools' `--downmix stereo` flag is
//! for surround → stereo and is out of scope here (we reject >2-channel input
//! anyway per the HLD). Mono input passes through unchanged.

use anyhow::{Result, bail};

/// Mix interleaved stereo f32 samples down to mono via `(L + R) * 0.5`.
///
/// Behaviour per channel count:
///   - 1 channel (mono): no-op, returns input clone.
///   - 2 channels (stereo): averages paired samples, returns half as many.
///   - anything else: errors cleanly. ropus only accepts mono/stereo input,
///     so this is a reinforced guard rather than a new restriction.
pub fn downmix_to_mono(interleaved: &[f32], channels: usize) -> Result<Vec<f32>> {
    match channels {
        1 => Ok(interleaved.to_vec()),
        2 => {
            let frames = interleaved.len() / 2;
            let mut out = Vec::with_capacity(frames);
            for frame in interleaved.chunks_exact(2) {
                out.push((frame[0] + frame[1]) * 0.5);
            }
            Ok(out)
        }
        n => {
            bail!("--downmix mono cannot handle {n}-channel input (only 1 or 2 channels supported)")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmix_stereo_to_mono() {
        // Four interleaved stereo samples → two mono samples.
        let input = [0.0, 1.0, 0.5, -0.5, 1.0, -1.0, 0.25, 0.75];
        let out = downmix_to_mono(&input, 2).expect("downmix stereo");
        assert_eq!(out.len(), 4, "stereo → mono halves the sample count");
        assert!((out[0] - 0.5).abs() < f32::EPSILON);
        assert!((out[1] - 0.0).abs() < f32::EPSILON);
        assert!((out[2] - 0.0).abs() < f32::EPSILON);
        assert!((out[3] - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn downmix_mono_is_noop() {
        let input = [0.25, -0.5, 0.75, 1.0];
        let out = downmix_to_mono(&input, 1).expect("downmix mono");
        assert_eq!(out, input);
    }

    #[test]
    fn downmix_rejects_surround() {
        // 3-channel input: should error cleanly, not panic or lose samples.
        let input = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let err = downmix_to_mono(&input, 3).expect_err("must reject surround");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("cannot handle 3-channel"),
            "expected channel-count error, got: {msg}"
        );
    }

    #[test]
    fn downmix_zero_channels_errors() {
        let input: [f32; 0] = [];
        assert!(downmix_to_mono(&input, 0).is_err());
    }

    #[test]
    fn downmix_stereo_empty_input_returns_empty() {
        let input: [f32; 0] = [];
        let out = downmix_to_mono(&input, 2).expect("empty stereo downmixes");
        assert!(out.is_empty());
    }
}
