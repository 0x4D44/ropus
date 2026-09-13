//! Triangular-PDF (TPDF) dither for f32 → i16 quantisation.
//!
//! Decorrelates quantisation error from the signal: instead of adding a
//! deterministic bias (truncation) or a signal-correlated sawtooth (round-half),
//! we add ±1 LSB of triangular-distributed noise. That turns the error into
//! uniform-spectrum dither rather than audible quantisation distortion on
//! low-amplitude material — the same trick `opusdec` applies by default.
//!
//! The RNG is a hand-rolled xorshift32: eight lines, no deps, fixed seed so the
//! output is reproducible across runs. Reproducibility matters far more for a
//! test-and-diff pipeline than cryptographic unpredictability.
//!
//! Default seeds are deliberately **distinct** so dither and packet-loss
//! streams drawn from the same PRNG family do not share state. Callers that
//! want a different seed (e.g. A/B comparison) can instantiate
//! [`Xorshift32`] directly.

/// Tiny xorshift32 PRNG. State must be non-zero; zero is a fixed point and
/// collapses the sequence. All default seeds in this module are non-zero.
#[derive(Debug, Clone)]
pub struct Xorshift32 {
    state: u32,
}

impl Xorshift32 {
    /// Construct with an explicit non-zero seed. Panics on `seed == 0`
    /// because xorshift cannot escape the all-zero state.
    pub fn new(seed: u32) -> Self {
        assert!(seed != 0, "xorshift32 seed must be non-zero");
        Self { state: seed }
    }

    /// Advance the PRNG and return the next 32-bit sample.
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }
}

/// Fixed seed for the dither PRNG stream.
pub const DITHER_SEED: u32 = 0xCAFE_F00D;
/// Fixed seed for the packet-loss PRNG stream. Distinct from `DITHER_SEED`
/// so loss decisions don't perturb dither output (or vice-versa).
pub const PACKET_LOSS_SEED: u32 = 0x1234_5678;

/// Quantise one interleaved f32 buffer to i16 with optional TPDF dither.
///
/// * `samples` are assumed to be in the nominal `[-1.0, 1.0]` range emitted by
///   `Decoder::decode_float`. Out-of-range values are hard-clamped after the
///   dither-noise addition.
/// * When `dither` is `false`, output is `(sample * 32768.0).round()` with a
///   clamp into i16 range. Round-to-nearest (ties-away-from-zero via `f32::round`)
///   is what `opus-tools opusdec` uses — bare `as i16` truncates toward zero
///   and biases negative samples by +0.5 LSB.
/// * When `dither` is `true`, a ±1 LSB triangular-PDF noise is added before
///   the rounding clamp. Each noise sample is the difference of two
///   full-width uniform draws; `rng` is advanced twice per output sample.
pub fn quantize_to_i16(samples: &[f32], dither: bool, rng: &mut Xorshift32) -> Vec<i16> {
    let mut out = Vec::with_capacity(samples.len());
    if dither {
        for &s in samples {
            // TPDF: difference of two independent uniform [0,1) draws gives
            // a triangular sample in (-1, 1) LSB. Use the full PRNG width;
            // taking only one bit would preserve the rounding bias this
            // dither is meant to decorrelate.
            let a = next_uniform(rng);
            let b = next_uniform(rng);
            let noise = a - b;
            let scaled = s * 32768.0 + noise;
            out.push(scaled.clamp(-32768.0, 32767.0).round() as i16);
        }
    } else {
        for &s in samples {
            let scaled = s * 32768.0;
            out.push(scaled.clamp(-32768.0, 32767.0).round() as i16);
        }
    }
    out
}

/// Map one full-width xorshift draw to a uniform `[0, 1)` float. The division
/// is performed in `f64` so the `u32::MAX` endpoint remains below one before
/// the result is narrowed for the f32 audio calculation.
#[inline]
fn next_uniform(rng: &mut Xorshift32) -> f32 {
    (f64::from(rng.next_u32()) / 4_294_967_296.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xorshift_is_deterministic_for_same_seed() {
        let mut a = Xorshift32::new(1);
        let mut b = Xorshift32::new(1);
        for _ in 0..1000 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn xorshift_differs_between_seeds() {
        let mut a = Xorshift32::new(1);
        let mut b = Xorshift32::new(2);
        // Different seeds must diverge within the first handful of draws.
        let mut any_diff = false;
        for _ in 0..16 {
            if a.next_u32() != b.next_u32() {
                any_diff = true;
                break;
            }
        }
        assert!(
            any_diff,
            "different seeds must not produce identical streams"
        );
    }

    #[test]
    fn dither_disabled_matches_naive_i16() {
        // Naive path: sample * 32768, clamp, round-to-nearest into i16.
        // dither=false must produce the identical sequence — same rounding
        // rule as the dithered branch, just without the noise term.
        let samples = vec![0.0, 0.5, -0.5, 0.999_99, -1.0, 0.123_456_f32];
        let mut rng = Xorshift32::new(DITHER_SEED);
        let out = quantize_to_i16(&samples, false, &mut rng);
        let expected: Vec<i16> = samples
            .iter()
            .map(|&s| (s * 32768.0).clamp(-32768.0, 32767.0).round() as i16)
            .collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn dither_enabled_tracks_unquantized_fractional_input() {
        // Constant input: a small non-zero value. Dither-on output must
        // differ from the rounded no-dither value by at most 1 LSB per
        // sample, and the mean across many samples must stay close to the
        // unquantized scaled input (TPDF decorrelates rounding error).
        //
        // The old one-bit draws could only add -1, 0, or +1, and therefore
        // left the fractional part at 0.8 unchanged. Full-width uniform
        // draws make the output distribution track the true 3276.8 value.
        let value = 0.1_f32;
        let n = 8192usize;
        let input = vec![value; n];
        let rounded = (value * 32768.0).round() as i16;

        let mut rng = Xorshift32::new(DITHER_SEED);
        let out = quantize_to_i16(&input, true, &mut rng);

        for (i, &s) in out.iter().enumerate() {
            let diff = (s as i32 - rounded as i32).abs();
            assert!(
                diff <= 1,
                "sample {i}: dithered {s} differs from rounded {rounded} by {diff} > 1 LSB"
            );
        }
        let sum: f64 = out.iter().map(|&s| s as f64).sum();
        let mean = sum / n as f64;
        let upper_count = out.iter().filter(|&&sample| sample > rounded).count();
        assert!(
            upper_count < n / 10,
            "upper tail {upper_count}/{n} is too large for fractional TPDF dither"
        );
        let scaled = f64::from(value) * 32768.0;
        assert!(
            (mean - scaled).abs() < 0.1,
            "mean {mean} should track unquantized scaled input {scaled}"
        );
    }

    #[test]
    fn dither_xorshift_is_deterministic() {
        // Same input + same seed must produce bit-identical output. This is
        // what makes `--packet-loss N` and dither-on runs reproducible for
        // diff-testing.
        let samples: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.01).sin() * 0.5).collect();

        let mut rng1 = Xorshift32::new(DITHER_SEED);
        let out1 = quantize_to_i16(&samples, true, &mut rng1);
        let mut rng2 = Xorshift32::new(DITHER_SEED);
        let out2 = quantize_to_i16(&samples, true, &mut rng2);

        assert_eq!(out1, out2);
    }
}
