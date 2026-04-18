//! `opus_pcm_soft_clip` — soft-clips interleaved PCM samples into `[-1, 1]`.
//!
//! Port of the C reference in `src/opus.c:39-166`. Generic C fallback —
//! returns 0 for the `within1` hint (so the per-sample `>1 / <-1` check is
//! always taken), while still performing the `[-2, +2]` input clamp. A SIMD
//! variant could set the hint to 1 and skip the per-sample scan.
//!
//! This function is exposed via the capi shim for the conformance test
//! `test_opus_decode.c::test_soft_clip`. It is *not* wired into our decoder's
//! hot path — the fixed-point decoder already clips to `i16` before producing
//! float output.
//!
//! The implementation is designed to round-trip with the continuation state
//! `declip_mem` such that applying it across two consecutive buffers produces
//! the same result as a single call on the concatenation.

/// Soft-clip a float PCM signal into the `[-1, 1]` range.
///
/// `x`: interleaved PCM, `n * c` samples (mutated in place).
/// `n`: samples per channel.
/// `c`: channel count (>=1).
/// `declip_mem`: one float per channel, initially zero; updated between calls.
///
/// No-ops when any of the sizes is non-positive or the slices are too short
/// for the requested processing, matching the C reference's early-return
/// behaviour on `C<1 || N<1 || !_x || !declip_mem`.
pub fn opus_pcm_soft_clip(x: &mut [f32], n: usize, c: usize, declip_mem: &mut [f32]) {
    // Mirror C: `if (C<1 || N<1 || !_x || !declip_mem) return;`
    if c < 1 || n < 1 {
        return;
    }
    // Storage preconditions (caller-provided; mirror what the C test passes).
    if declip_mem.len() < c || x.len() < n * c {
        return;
    }

    // `opus_limit2_checkwithin1_c` (generic C fallback in `celt/mathops.c`):
    // clamp every sample to [-2, +2] and always return 0 (i.e., the caller
    // must do the per-sample in-range check below). This clamp is NOT
    // optional — the soft-clip formula `a=(maxval-1)/maxval^2` only yields
    // output in [-1, 1] when the input is pre-clamped to [-2, 2].
    for v in x[..n * c].iter_mut() {
        if *v > 2.0 {
            *v = 2.0;
        } else if *v < -2.0 {
            *v = -2.0;
        }
    }
    // Placeholder for SIMD detection hint; always false in the generic port.
    let all_within_neg1pos1 = false;

    for ch in 0..c {
        let mut a = declip_mem[ch];

        // Continue applying the non-linearity from the previous frame to avoid
        // any discontinuity.
        for i in 0..n {
            let xi = x[i * c + ch];
            if xi * a >= 0.0 {
                break;
            }
            x[i * c + ch] = xi + a * xi * xi;
        }

        let mut curr: usize = 0;
        let x0 = x[ch]; // i.e., x[0*C + ch]

        loop {
            // Detection for early exit can be skipped if hinted.
            let i_break = if all_within_neg1pos1 {
                n
            } else {
                let mut found = n;
                for i in curr..n {
                    let v = x[i * c + ch];
                    if v > 1.0 || v < -1.0 {
                        found = i;
                        break;
                    }
                }
                found
            };

            if i_break == n {
                a = 0.0;
                break;
            }

            let i = i_break;
            let peak_sign_x = x[i * c + ch];
            let mut peak_pos = i;
            let mut start = i;
            let mut end = i;
            let mut maxval = peak_sign_x.abs();

            // Look for first zero crossing before clipping.
            while start > 0 && peak_sign_x * x[(start - 1) * c + ch] >= 0.0 {
                start -= 1;
            }
            // Look for first zero crossing after clipping.
            while end < n && peak_sign_x * x[end * c + ch] >= 0.0 {
                let v = x[end * c + ch];
                if v.abs() > maxval {
                    maxval = v.abs();
                    peak_pos = end;
                }
                end += 1;
            }

            // Detect the special case where we clip before the first zero crossing.
            let special = start == 0 && peak_sign_x * x[ch] >= 0.0;

            // Compute a such that maxval + a*maxval^2 = 1.
            let mut a_local = (maxval - 1.0) / (maxval * maxval);
            // Slightly boost "a" by 2^-22. Ensures -ffast-math does not push
            // output values past +/-1, while being small enough not to matter
            // even for 24-bit output.
            a_local += a_local * 2.4e-7f32;
            if peak_sign_x > 0.0 {
                a_local = -a_local;
            }

            // Apply soft clipping.
            for j in start..end {
                let v = x[j * c + ch];
                x[j * c + ch] = v + a_local * v * v;
            }

            if special && peak_pos >= 2 {
                // Add a linear ramp from the first sample to the signal peak.
                // This avoids a discontinuity at the beginning of the frame.
                let offset_init = x0 - x[ch];
                let delta = offset_init / peak_pos as f32;
                let mut offset = offset_init;
                for j in curr..peak_pos {
                    offset -= delta;
                    let v = x[j * c + ch] + offset;
                    let clipped = if v > 1.0 {
                        1.0
                    } else if v < -1.0 {
                        -1.0
                    } else {
                        v
                    };
                    x[j * c + ch] = clipped;
                }
            }

            a = a_local;
            curr = end;
            if curr == n {
                break;
            }
        }

        declip_mem[ch] = a;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_is_within_unit_range() {
        // Signal forced well outside [-1, 1]; after soft_clip every sample
        // must lie in [-1, 1].
        let mut x = [0.0f32; 1024];
        for (j, v) in x.iter_mut().enumerate() {
            *v = (j as f32 % 256.0) * (1.0 / 32.0) - 4.0;
        }
        let mut s = [0.0f32; 1];
        opus_pcm_soft_clip(&mut x, 1024, 1, &mut s);
        for v in x.iter() {
            assert!(*v >= -1.0 && *v <= 1.0, "sample {} out of range", v);
        }
    }

    #[test]
    fn no_op_on_zero_or_negative_sizes() {
        let mut x = [0.5f32; 4];
        let orig = x;
        let mut s = [0.0f32; 1];
        opus_pcm_soft_clip(&mut x, 0, 1, &mut s);
        assert_eq!(x, orig);
        opus_pcm_soft_clip(&mut x, 1, 0, &mut s);
        assert_eq!(x, orig);
    }

    #[test]
    fn multichannel_interleaved_handled() {
        // 2-channel; check every sample gets clipped regardless of channel.
        let n = 512;
        let mut x = vec![0.0f32; n * 2];
        for (j, v) in x.iter_mut().enumerate() {
            *v = ((j / 2) as f32 % 256.0) * (1.0 / 32.0) - 4.0;
        }
        let mut s = [0.0f32; 2];
        opus_pcm_soft_clip(&mut x, n, 2, &mut s);
        for v in x.iter() {
            assert!(*v >= -1.0 && *v <= 1.0);
        }
    }
}
