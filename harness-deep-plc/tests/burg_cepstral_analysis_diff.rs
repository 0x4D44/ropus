#![cfg(not(no_reference))]
//! Differential test: `burg_cepstral_analysis` Rust vs C reference.
//!
//! Companion to the f32/f64 parity bundle in
//! `wrk_docs/2026.05.07 - HLD - burg-cepstrum-pow-fix.md`. Asserts
//! bit-for-bit (`f32` bit-pattern) equality on all `2 * NB_BANDS = 36`
//! outputs across two deterministic fixtures:
//!
//! 1. 440 Hz sine, FRAME_SIZE = 160 samples — narrow-band, smooth `burg_lpc`
//!    magnitudes, exercises the typical voiced-frame regime.
//! 2. White noise from a deterministic xorshift seed, FRAME_SIZE = 160 — flat
//!    spectrum, broader `burg_lpc` magnitude coverage to surface any
//!    f32/f64 deviation that the sine fixture might mask through coefficient
//!    smallness.
//!
//! `burg_cepstral_analysis` has no internal state — both halves of the
//! frame are processed independently inside `compute_burg_cepstrum`, and
//! the function takes only the 160-sample input buffer plus an output
//! pointer. No priming or warm-up frame is required.

use ropus::dnn::lpcnet::{FRAME_SIZE, NB_BANDS, burg_cepstral_analysis};
use ropus_harness_deep_plc::{OPUS_OK, ropus_test_burg_cepstral_analysis};

#[path = "support/finite_oracle.rs"]
#[allow(dead_code)] // The shared module also exposes slice-only guards.
mod finite_oracle;
use finite_oracle::assert_finite_pair;

const NB_OUTPUTS: usize = 2 * NB_BANDS;

fn first_f32_divergence(a: &[f32], b: &[f32]) -> Option<(usize, f32, f32)> {
    assert_finite_pair("Burg cepstral differential output", a, b);
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if x.to_bits() != y.to_bits() {
            return Some((i, x, y));
        }
    }
    None
}

fn run_diff(fixture_name: &str, x: &[f32; FRAME_SIZE]) {
    let mut ceps_rust = [0.0f32; NB_OUTPUTS];
    burg_cepstral_analysis(&mut ceps_rust, x);

    let mut ceps_c = [0.0f32; NB_OUTPUTS];
    let c_ret = unsafe {
        ropus_test_burg_cepstral_analysis(
            x.as_ptr(),
            ceps_c.as_mut_ptr(),
            x.len() as i32,
            ceps_c.len() as i32,
        )
    };
    assert_eq!(c_ret, OPUS_OK, "C Burg shim rejected valid buffer lengths");

    if let Some((i, rust_value, c_value)) = first_f32_divergence(&ceps_rust, &ceps_c) {
        let r_bits = rust_value.to_bits();
        let c_bits = c_value.to_bits();
        assert_eq!(
            r_bits,
            c_bits,
            "fixture={fixture_name}: ceps[{i}] f32 bits differ (rust=0x{r_bits:08x} = {r:?}, c=0x{c_bits:08x} = {c:?})",
            r = rust_value,
            c = c_value,
        );
    }
}

#[cfg(test)]
mod oracle_tests {
    use super::first_f32_divergence;

    #[test]
    #[should_panic(expected = "non-finite")]
    fn first_f32_divergence_rejects_same_nan() {
        let nan = f32::NAN;
        let _ = first_f32_divergence(&[nan], &[nan]);
    }

    #[test]
    #[should_panic(expected = "equal lengths")]
    fn first_f32_divergence_rejects_unequal_lengths() {
        let _ = first_f32_divergence(&[0.0], &[]);
    }
}

#[test]
fn burg_cepstral_analysis_sine440_bit_exact() {
    // 440 Hz sine at the LPCNet feature-extractor 16 kHz internal rate,
    // amplitude scaled to typical voiced loudness (~0.5 full-scale).
    const FS: f32 = 16_000.0;
    const FREQ: f32 = 440.0;
    let mut x = [0.0f32; FRAME_SIZE];
    for (i, sample) in x.iter_mut().enumerate() {
        let t = i as f32 / FS;
        *sample = 0.5 * (2.0 * std::f32::consts::PI * FREQ * t).sin();
    }
    run_diff("sine440", &x);
}

#[test]
fn burg_cepstral_analysis_white_noise_bit_exact() {
    // Deterministic xorshift32 PRNG (Marsaglia). Same seed picked once and
    // never changed — adding noise here is a regression guard, not a
    // statistical claim. The fixture's job is to broaden `burg_lpc[i]`
    // magnitude coverage relative to the sine.
    let mut state: u32 = 0x1234_5678;
    let mut x = [0.0f32; FRAME_SIZE];
    for sample in x.iter_mut() {
        // xorshift32
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        // Map u32 -> [-0.5, 0.5) deterministically.
        let u = (state as f32) / (u32::MAX as f32);
        *sample = u - 0.5;
    }
    run_diff("white_noise", &x);
}
