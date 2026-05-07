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
use ropus_harness_deep_plc::ropus_test_burg_cepstral_analysis;

const NB_OUTPUTS: usize = 2 * NB_BANDS;

fn run_diff(fixture_name: &str, x: &[f32; FRAME_SIZE]) {
    let mut ceps_rust = [0.0f32; NB_OUTPUTS];
    burg_cepstral_analysis(&mut ceps_rust, x);

    let mut ceps_c = [0.0f32; NB_OUTPUTS];
    unsafe {
        ropus_test_burg_cepstral_analysis(x.as_ptr(), ceps_c.as_mut_ptr());
    }

    for i in 0..NB_OUTPUTS {
        let r_bits = ceps_rust[i].to_bits();
        let c_bits = ceps_c[i].to_bits();
        assert_eq!(
            r_bits,
            c_bits,
            "fixture={fixture_name}: ceps[{i}] f32 bits differ (rust=0x{r_bits:08x} = {r:?}, c=0x{c_bits:08x} = {c:?})",
            r = ceps_rust[i],
            c = ceps_c[i],
        );
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
