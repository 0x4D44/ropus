#![cfg(not(no_reference))]
//! Stage 8.4 differential test — RDOVAE encoder forward pass.
//!
//! Tier 1 target (per `wrk_docs/2026.04.19 - HLD - dred-port.md` staging
//! row 8.4): bit-exact against the xiph C reference on a synthetic input.
//! Tier 2 fallback: SNR ≥ 60 dB.
//!
//! Setup:
//! - C side: `dred_rdovae_enc.c` linked via `harness-deep-plc/build.rs`,
//!   exposed through `harness-deep-plc/dred_enc_shim.c` opaque handles.
//! - Rust side: `ropus::dnn::dred::RDOVAEEncState::encode_dframe`,
//!   model-initialised from the same compile-time weight blob the C side
//!   uses (xiph generates both from the same pinned checkpoint).
//!
//! The test runs multiple consecutive frames so the GRU hidden state and
//! dilated conv memories evolve non-trivially — a single-frame comparison
//! wouldn't exercise the recurrent paths.

use ropus::dnn::core::parse_weights;
use ropus::dnn::dred::{
    DRED_LATENT_DIM, DRED_NUM_FEATURES, DRED_STATE_DIM, RDOVAEEncState, init_rdovaeenc,
};
use ropus::dnn::embedded_weights::WEIGHTS_BLOB;

use ropus_harness_deep_plc::{
    ropus_test_dred_rdovae_encode_dframe, ropus_test_rdovae_enc_state_free,
    ropus_test_rdovae_enc_state_new, ropus_test_rdovaeenc_free, ropus_test_rdovaeenc_new,
};

/// How many consecutive frames to push through before declaring divergence
/// a pass. Covers at least two dilation=2 state rotations (2 frames) and a
/// few extra to catch accumulation drift.
const NUM_FRAMES: usize = 8;

/// Deterministic 40-wide feature input for frame `f`. Small magnitudes so
/// tanh saturation doesn't hide numerical drift; explicit integer-seeded
/// xorshift so the signal is bit-identical across platforms.
fn synth_input(f: usize) -> [f32; 2 * DRED_NUM_FEATURES] {
    // Seed the xorshift so frame 0 already diverges from a uniform ramp —
    // avoids any suspicion that we're testing on zero state.
    let mut rng: u32 = 0xC0FFEE_u32.wrapping_add(f as u32 * 0x9E3779B9);
    let mut out = [0.0f32; 2 * DRED_NUM_FEATURES];
    for (i, x) in out.iter_mut().enumerate() {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        // Map 32-bit signed RNG to [-0.3, 0.3], a typical LPCNet feature range.
        let s = (rng as i32) as f64 / (i32::MAX as f64);
        let phase = (i as f64) * 0.11 + (f as f64) * 0.37;
        *x = (s * 0.2 + phase.sin() * 0.1) as f32;
    }
    out
}

fn compute_snr_db(reference: &[f32], test: &[f32]) -> f64 {
    assert_eq!(reference.len(), test.len());
    let mut sig_power = 0.0f64;
    let mut err_power = 0.0f64;
    for (&r, &t) in reference.iter().zip(test.iter()) {
        let r64 = r as f64;
        let t64 = t as f64;
        sig_power += r64 * r64;
        let e = t64 - r64;
        err_power += e * e;
    }
    let n = reference.len() as f64;
    let sig = sig_power / n;
    let err = err_power / n;
    if err == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (sig / err).log10()
}

/// Index of the first diverging element (if any). Returns None on bit-exact.
fn first_divergent(a: &[f32], b: &[f32]) -> Option<(usize, f32, f32)> {
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if x.to_bits() != y.to_bits() {
            return Some((i, x, y));
        }
    }
    None
}

/// Guard against running without an embedded DRED weight blob — if
/// `fetch-assets -- weights` wasn't run the crate compiles but the test
/// has nothing to compare. Emit a loud skip, don't silently pass.
fn weights_or_skip() -> bool {
    if WEIGHTS_BLOB.is_empty() {
        eprintln!(
            "dred_rdovae_enc_diff: WEIGHTS_BLOB empty — skipping. \
             Run `cargo run -p fetch-assets -- all` to populate."
        );
        return false;
    }
    true
}

#[test]
fn rdovae_encode_dframe_matches_c_reference() {
    if !weights_or_skip() {
        return;
    }

    // --- C side model + state ---
    // SAFETY: shim returns a heap-allocated struct, or NULL on failure.
    let c_model = unsafe { ropus_test_rdovaeenc_new() };
    assert!(!c_model.is_null(), "C ropus_test_rdovaeenc_new failed");
    let c_state = unsafe { ropus_test_rdovae_enc_state_new() };
    assert!(
        !c_state.is_null(),
        "C ropus_test_rdovae_enc_state_new failed"
    );

    // --- Rust side model + state ---
    let arrays = parse_weights(WEIGHTS_BLOB).expect("parse_weights WEIGHTS_BLOB");
    let rust_model = init_rdovaeenc(&arrays).expect("init_rdovaeenc from embedded blob");
    let mut rust_state = RDOVAEEncState::default();

    // Track running tier-1 / tier-2 outcomes across frames.
    let mut all_bit_exact = true;
    let mut worst_snr_latents = f64::INFINITY;
    let mut worst_snr_state = f64::INFINITY;
    let mut first_drift_frame: Option<usize> = None;
    let mut first_drift_details: Option<(String, usize, f32, f32)> = None;

    for f in 0..NUM_FRAMES {
        let input = synth_input(f);

        // --- C forward pass ---
        let mut c_latents = vec![0.0f32; DRED_LATENT_DIM];
        let mut c_initial_state = vec![0.0f32; DRED_STATE_DIM];
        unsafe {
            ropus_test_dred_rdovae_encode_dframe(
                c_state,
                c_model,
                c_latents.as_mut_ptr(),
                c_initial_state.as_mut_ptr(),
                input.as_ptr(),
            );
        }

        // --- Rust forward pass ---
        let mut r_latents = vec![0.0f32; DRED_LATENT_DIM];
        let mut r_initial_state = vec![0.0f32; DRED_STATE_DIM];
        rust_state.encode_dframe(&rust_model, &mut r_latents, &mut r_initial_state, &input);

        // Tier 1 — bit-exact. `to_bits()` comparison handles NaN / -0.0
        // edge cases without tripping PartialEq quirks.
        let latents_diverge = first_divergent(&c_latents, &r_latents);
        let state_diverge = first_divergent(&c_initial_state, &r_initial_state);
        if latents_diverge.is_some() || state_diverge.is_some() {
            all_bit_exact = false;
            if first_drift_frame.is_none() {
                first_drift_frame = Some(f);
                first_drift_details = latents_diverge
                    .map(|(i, c, r)| ("latents".into(), i, c, r))
                    .or_else(|| state_diverge.map(|(i, c, r)| ("state".into(), i, c, r)));
            }
        }

        // Tier 2 — SNR dB on this frame's outputs.
        let snr_l = compute_snr_db(&c_latents, &r_latents);
        let snr_s = compute_snr_db(&c_initial_state, &r_initial_state);
        if snr_l < worst_snr_latents {
            worst_snr_latents = snr_l;
        }
        if snr_s < worst_snr_state {
            worst_snr_state = snr_s;
        }
        eprintln!(
            "frame {f}: SNR(latents)={:>7.2} dB   SNR(state)={:>7.2} dB   \
             bit-exact-latents={}   bit-exact-state={}",
            if snr_l.is_infinite() {
                f64::INFINITY
            } else {
                snr_l
            },
            if snr_s.is_infinite() {
                f64::INFINITY
            } else {
                snr_s
            },
            latents_diverge.is_none(),
            state_diverge.is_none(),
        );
    }

    // --- Free C handles before assertions so a failure doesn't leak. ---
    unsafe {
        ropus_test_rdovaeenc_free(c_model);
        ropus_test_rdovae_enc_state_free(c_state);
    }

    if all_bit_exact {
        eprintln!("Tier 1 achieved: bit-exact for all {NUM_FRAMES} frames.");
        return;
    }

    // Tier-1 miss — fall back to tier-2 SNR check.
    eprintln!(
        "Tier 1 drift: first divergence at frame {:?}, detail {:?}",
        first_drift_frame, first_drift_details
    );
    eprintln!(
        "Tier 2 guard: worst SNR latents = {:.2} dB, worst SNR state = {:.2} dB",
        worst_snr_latents, worst_snr_state
    );

    const TIER2_THRESHOLD_DB: f64 = 60.0;
    assert!(
        worst_snr_latents >= TIER2_THRESHOLD_DB,
        "Tier-2 failure on latents: worst SNR {:.2} dB < {:.0} dB. \
         First drift: frame={:?}, detail={:?}",
        worst_snr_latents,
        TIER2_THRESHOLD_DB,
        first_drift_frame,
        first_drift_details
    );
    assert!(
        worst_snr_state >= TIER2_THRESHOLD_DB,
        "Tier-2 failure on initial_state: worst SNR {:.2} dB < {:.0} dB. \
         First drift: frame={:?}, detail={:?}",
        worst_snr_state,
        TIER2_THRESHOLD_DB,
        first_drift_frame,
        first_drift_details
    );
}
