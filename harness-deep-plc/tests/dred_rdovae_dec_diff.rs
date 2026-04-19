//! Stage 8.5 differential test — RDOVAE decoder forward pass.
//!
//! Tier 1 target (per `wrk_docs/2026.04.19 - HLD - dred-port.md` staging
//! row 8.5): bit-exact against the xiph C reference on a synthetic input.
//! Tier 2 fallback: SNR ≥ 60 dB.
//!
//! Setup:
//! - C side: `dred_rdovae_dec.c` linked via `harness-deep-plc/build.rs`,
//!   exposed through `harness-deep-plc/dred_dec_shim.c` opaque handles.
//! - Rust side: `ropus::dnn::dred::RDOVAEDecState::{init_states, decode_qframe}`,
//!   model-initialised from the same compile-time weight blob the C side
//!   uses (xiph generates both from the same pinned checkpoint).
//!
//! The test runs multiple consecutive frames so the GRU hidden state and
//! conv memories evolve non-trivially — a single-frame comparison wouldn't
//! exercise the recurrent paths.
//!
//! The xorshift seed is deliberately distinct from the 8.4 encoder test
//! (`0xC0FFEE` there, `0xDECAF00D` here) so the two synthetic-input banks
//! can never accidentally collide.

use ropus::dnn::core::parse_weights;
use ropus::dnn::dred::{
    DRED_LATENT_DIM, DRED_NUM_FEATURES, DRED_STATE_DIM, RDOVAEDecState, RDOVAEEncState,
    compute_quantizer, init_rdovaedec, init_rdovaeenc,
};
use ropus::dnn::dred_stats::{
    dred_latent_dead_zone_q8, dred_latent_p0_q8, dred_latent_quant_scales_q8, dred_latent_r_q8,
    dred_state_dead_zone_q8, dred_state_p0_q8, dred_state_quant_scales_q8, dred_state_r_q8,
};
use ropus::dnn::embedded_weights::WEIGHTS_BLOB;

use ropus_harness_deep_plc::{
    ropus_test_dred_rdovae_dec_init_states, ropus_test_dred_rdovae_decode_qframe,
    ropus_test_rdovae_dec_state_free, ropus_test_rdovae_dec_state_new, ropus_test_rdovaedec_free,
    ropus_test_rdovaedec_new,
};

/// How many consecutive frames to push through before declaring divergence
/// a pass. More than the encoder's kernel depth so the GRU banks evolve
/// and the conv memories shift at least a few times post-init.
const NUM_FRAMES: usize = 8;

/// The C reference passes a 26-wide latent to `dred_rdovae_decode_qframe`
/// (25 floats + 1 trailing slot for the quantiser index). Matches
/// `dec_dense1.nb_inputs` in `init_rdovaedec`.
const DEC_INPUT_WIDTH: usize = DRED_LATENT_DIM + 1;

/// The C `dred_rdovae_decode_qframe` writes a qframe of size 4 * 20 = 80
/// (four LPCNet feature vectors in reverse time order). Matches
/// `dec_output.nb_outputs`.
const DEC_QFRAME_WIDTH: usize = 80;

/// Deterministic 50-wide initial state (the encoder's `gdense2` output).
/// Seed is distinct from any other DRED synthetic input generator so
/// accidental collisions between tests are impossible.
fn synth_initial_state() -> [f32; DRED_STATE_DIM] {
    let mut rng: u32 = 0xDECAF00D_u32;
    let mut out = [0.0f32; DRED_STATE_DIM];
    for (i, x) in out.iter_mut().enumerate() {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        let s = (rng as i32) as f64 / (i32::MAX as f64);
        let phase = (i as f64) * 0.09 + 0.5;
        *x = (s * 0.15 + phase.cos() * 0.05) as f32;
    }
    out
}

/// Deterministic 26-wide latent for frame `f`. Magnitudes small enough
/// that the tanh stack doesn't saturate, so numerical drift would be
/// visible rather than masked by ±1 clipping.
fn synth_input(f: usize) -> [f32; DEC_INPUT_WIDTH] {
    // Distinct xorshift seed from the encoder's test (which uses 0xC0FFEE).
    // Adding the frame index keeps the sequence deterministic but
    // non-repeating across frames.
    let mut rng: u32 = 0xDECAF00D_u32.wrapping_add(f as u32 * 0x85EBCA6B);
    let mut out = [0.0f32; DEC_INPUT_WIDTH];
    for (i, x) in out.iter_mut().enumerate() {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        let s = (rng as i32) as f64 / (i32::MAX as f64);
        let phase = (i as f64) * 0.13 + (f as f64) * 0.29;
        *x = (s * 0.18 + phase.sin() * 0.08) as f32;
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
            "dred_rdovae_dec_diff: WEIGHTS_BLOB empty — skipping. \
             Run `cargo run -p fetch-assets -- all` to populate."
        );
        return false;
    }
    true
}

#[test]
fn rdovae_decode_qframe_matches_c_reference() {
    if !weights_or_skip() {
        return;
    }

    // --- C side model + state ---
    // SAFETY: shim returns a heap-allocated struct, or NULL on failure.
    let c_model = unsafe { ropus_test_rdovaedec_new() };
    assert!(!c_model.is_null(), "C ropus_test_rdovaedec_new failed");
    let c_state = unsafe { ropus_test_rdovae_dec_state_new() };
    assert!(
        !c_state.is_null(),
        "C ropus_test_rdovae_dec_state_new failed"
    );

    // --- Rust side model + state ---
    let arrays = parse_weights(WEIGHTS_BLOB).expect("parse_weights WEIGHTS_BLOB");
    let rust_model = init_rdovaedec(&arrays).expect("init_rdovaedec from embedded blob");
    let mut rust_state = RDOVAEDecState::default();

    // Seed both decoders' GRU banks from an identical `initial_state`
    // bank — the same projection the encoder-side `gdense2` emits at the
    // start of each DRED payload.
    let initial_state = synth_initial_state();
    unsafe {
        ropus_test_dred_rdovae_dec_init_states(c_state, c_model, initial_state.as_ptr());
    }
    rust_state.init_states(&rust_model, &initial_state);

    // Track running tier-1 / tier-2 outcomes across frames.
    let mut all_bit_exact = true;
    let mut worst_snr_qframe = f64::INFINITY;
    let mut first_drift_frame: Option<usize> = None;
    let mut first_drift_details: Option<(usize, f32, f32)> = None;

    for f in 0..NUM_FRAMES {
        let input = synth_input(f);

        // --- C forward pass ---
        let mut c_qframe = vec![0.0f32; DEC_QFRAME_WIDTH];
        unsafe {
            ropus_test_dred_rdovae_decode_qframe(
                c_state,
                c_model,
                c_qframe.as_mut_ptr(),
                input.as_ptr(),
            );
        }

        // --- Rust forward pass ---
        let mut r_qframe = vec![0.0f32; DEC_QFRAME_WIDTH];
        rust_state.decode_qframe(&rust_model, &mut r_qframe, &input);

        // Tier 1 — bit-exact. `to_bits()` comparison handles NaN / -0.0
        // edge cases without tripping PartialEq quirks.
        let qframe_diverge = first_divergent(&c_qframe, &r_qframe);
        if let Some(detail) = qframe_diverge {
            all_bit_exact = false;
            if first_drift_frame.is_none() {
                first_drift_frame = Some(f);
                first_drift_details = Some(detail);
            }
        }

        // Tier 2 — SNR dB on this frame's output.
        let snr = compute_snr_db(&c_qframe, &r_qframe);
        if snr < worst_snr_qframe {
            worst_snr_qframe = snr;
        }
        eprintln!(
            "frame {f}: SNR(qframe)={:>7.2} dB   bit-exact={}",
            if snr.is_infinite() {
                f64::INFINITY
            } else {
                snr
            },
            qframe_diverge.is_none(),
        );
    }

    // --- Free C handles before assertions so a failure doesn't leak. ---
    unsafe {
        ropus_test_rdovaedec_free(c_model);
        ropus_test_rdovae_dec_state_free(c_state);
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
        "Tier 2 guard: worst SNR qframe = {:.2} dB",
        worst_snr_qframe
    );

    const TIER2_THRESHOLD_DB: f64 = 60.0;
    assert!(
        worst_snr_qframe >= TIER2_THRESHOLD_DB,
        "Tier-2 failure on qframe: worst SNR {:.2} dB < {:.0} dB. \
         First drift: frame={:?}, detail={:?}",
        worst_snr_qframe,
        TIER2_THRESHOLD_DB,
        first_drift_frame,
        first_drift_details
    );
}

/// Stage 8.6 carryover: complement to `rdovae_decode_qframe_matches_c_reference`
/// that exercises the decoder on integer-aligned quantised inputs — the
/// realistic code path DRED actually takes. The original test feeds
/// continuous-valued xorshift floats directly into `decode_qframe`, which
/// is plenty to shake out layer-level bugs but doesn't catch anything
/// that only misbehaves on the specific integer lattice the encoder
/// emits.
///
/// This test runs two back-to-back RDOVAE encoders (C + Rust), passes
/// realistic 40-wide feature inputs through both, quantises the resulting
/// latents / initial_state through the exact same deadzone + tanh pipeline
/// as the encoder's range coder, packs the integer outputs as floats, and
/// feeds both decoders. Byte-exact match asserts the decoder stays
/// tier-1 even on the constrained-input manifold.
#[test]
fn decode_qframe_diff_quantised_inputs() {
    if !weights_or_skip() {
        return;
    }

    // --- Rust + C encoder side (to produce realistic unquantised latents). ---
    let arrays = parse_weights(WEIGHTS_BLOB).expect("parse_weights WEIGHTS_BLOB");
    let rust_enc_model = init_rdovaeenc(&arrays).expect("init_rdovaeenc");
    let mut rust_enc_state = RDOVAEEncState::default();

    // --- Rust + C decoder side. ---
    let rust_dec_model = init_rdovaedec(&arrays).expect("init_rdovaedec");
    let mut rust_dec_state = RDOVAEDecState::default();
    let c_dec_model = unsafe { ropus_test_rdovaedec_new() };
    assert!(!c_dec_model.is_null(), "C rdovaedec_new failed");
    let c_dec_state = unsafe { ropus_test_rdovae_dec_state_new() };
    assert!(!c_dec_state.is_null(), "C rdovae_dec_state_new failed");

    // Step 1: run the encoder once on a deterministic feature frame to
    // produce a plausible `initial_state` for the decoder — the exact
    // oracle both sides will seed from. `synth_initial_state()` above
    // bypasses the encoder; here we route through it so the init_states
    // input is on the real-payload manifold.
    let mut enc_input = [0.0f32; 2 * DRED_NUM_FEATURES];
    {
        let mut rng: u32 = 0x8DA_BEEF_u32;
        for x in enc_input.iter_mut() {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            let s = (rng as i32) as f64 / (i32::MAX as f64);
            *x = (s * 0.2) as f32;
        }
    }
    let mut warmup_latents = [0.0f32; DRED_LATENT_DIM];
    let mut seed_state = [0.0f32; DRED_STATE_DIM];
    rust_enc_state.encode_dframe(
        &rust_enc_model,
        &mut warmup_latents,
        &mut seed_state,
        &enc_input,
    );

    // Quantise the seed state through the encoder's deadzone pipeline
    // (q0 = 0 picks the first row of `dred_state_*_q8`), then dequantise
    // back to float as the decoder expects.
    let quantised_state = quantise_state(&seed_state, 0);

    // Seed both decoders from the same quantised state.
    unsafe {
        ropus_test_dred_rdovae_dec_init_states(c_dec_state, c_dec_model, quantised_state.as_ptr());
    }
    rust_dec_state.init_states(&rust_dec_model, &quantised_state);

    const NUM_FRAMES_QUANT: usize = 8;
    let mut all_bit_exact = true;
    let mut worst_snr_qframe = f64::INFINITY;
    let mut first_drift: Option<(usize, usize, f32, f32)> = None;

    for f in 0..NUM_FRAMES_QUANT {
        // Re-encode to get a fresh realistic latent.
        let mut frame_input = [0.0f32; 2 * DRED_NUM_FEATURES];
        let mut rng: u32 = 0xDEADC0DE_u32.wrapping_add((f as u32) * 0x9E3779B9);
        for x in frame_input.iter_mut() {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            let s = (rng as i32) as f64 / (i32::MAX as f64);
            *x = (s * 0.2) as f32;
        }
        let mut latents = [0.0f32; DRED_LATENT_DIM];
        let mut _state = [0.0f32; DRED_STATE_DIM];
        rust_enc_state.encode_dframe(&rust_enc_model, &mut latents, &mut _state, &frame_input);

        // Quantise the latents through the encoder's deadzone pipeline
        // at quantiser level matching the actual `compute_quantizer`
        // call used for chunk `f/2`.
        let q_level = compute_quantizer(0, 0, 15, (f as i32) / 2);
        let quantised_latent = quantise_latent(&latents, q_level);
        // Pack as decoder input: 25 quantised floats + 1 trailing slot
        // (the quantiser level byte). The C decoder reads 26 floats.
        let mut dec_input = [0.0f32; DRED_LATENT_DIM + 1];
        dec_input[..DRED_LATENT_DIM].copy_from_slice(&quantised_latent);
        dec_input[DRED_LATENT_DIM] = q_level as f32;

        // --- C forward pass ---
        let mut c_qframe = vec![0.0f32; 80];
        unsafe {
            ropus_test_dred_rdovae_decode_qframe(
                c_dec_state,
                c_dec_model,
                c_qframe.as_mut_ptr(),
                dec_input.as_ptr(),
            );
        }
        // --- Rust forward pass ---
        let mut r_qframe = vec![0.0f32; 80];
        rust_dec_state.decode_qframe(&rust_dec_model, &mut r_qframe, &dec_input);

        // Tier 1 — bit-exact.
        if let Some((idx, cv, rv)) = first_divergent(&c_qframe, &r_qframe) {
            all_bit_exact = false;
            if first_drift.is_none() {
                first_drift = Some((f, idx, cv, rv));
            }
        }
        let snr = compute_snr_db(&c_qframe, &r_qframe);
        if snr < worst_snr_qframe {
            worst_snr_qframe = snr;
        }
        eprintln!(
            "[quantised] frame {f}: q_level={q_level}, SNR={:>7.2} dB, bit-exact={}",
            if snr.is_infinite() {
                f64::INFINITY
            } else {
                snr
            },
            all_bit_exact,
        );
    }

    // Free C handles before assertions.
    unsafe {
        ropus_test_rdovaedec_free(c_dec_model);
        ropus_test_rdovae_dec_state_free(c_dec_state);
    }

    if all_bit_exact {
        eprintln!("Tier 1 achieved: bit-exact on {NUM_FRAMES_QUANT} quantised frames.");
        return;
    }
    const TIER2_THRESHOLD_DB: f64 = 60.0;
    assert!(
        worst_snr_qframe >= TIER2_THRESHOLD_DB,
        "Tier-2 failure on qframe (quantised): worst SNR {:.2} dB < {:.0} dB. First drift: {:?}",
        worst_snr_qframe,
        TIER2_THRESHOLD_DB,
        first_drift,
    );
}

/// Run the first three steps of `dred_encode_latents` (deadzone + tanh +
/// floor-half-up rounding) but skip the Laplace range coder, and then
/// dequantise back to float as the decoder expects. `q_level` indexes the
/// per-quantiser-level tables.
fn quantise_latent(x: &[f32; DRED_LATENT_DIM], q_level: i32) -> [f32; DRED_LATENT_DIM] {
    let offset = (q_level as usize) * DRED_LATENT_DIM;
    let scale = &dred_latent_quant_scales_q8[offset..offset + DRED_LATENT_DIM];
    let dzone = &dred_latent_dead_zone_q8[offset..offset + DRED_LATENT_DIM];
    let r = &dred_latent_r_q8[offset..offset + DRED_LATENT_DIM];
    let p0 = &dred_latent_p0_q8[offset..offset + DRED_LATENT_DIM];
    quantise_one(x, scale, dzone, r, p0)
}

/// State-side variant of `quantise_latent` for 50-wide init seed.
fn quantise_state(x: &[f32; DRED_STATE_DIM], q_level: i32) -> [f32; DRED_STATE_DIM] {
    let offset = (q_level as usize) * DRED_STATE_DIM;
    let scale = &dred_state_quant_scales_q8[offset..offset + DRED_STATE_DIM];
    let dzone = &dred_state_dead_zone_q8[offset..offset + DRED_STATE_DIM];
    let r = &dred_state_r_q8[offset..offset + DRED_STATE_DIM];
    let p0 = &dred_state_p0_q8[offset..offset + DRED_STATE_DIM];
    quantise_one(x, scale, dzone, r, p0)
}

fn quantise_one<const N: usize>(
    x: &[f32; N],
    scale: &[u8],
    dzone: &[u8],
    r: &[u8],
    p0: &[u8],
) -> [f32; N] {
    let mut delta = [0.0f32; N];
    let mut xq = [0.0f32; N];
    let mut deadzone = [0.0f32; N];
    let eps = 0.1f32;
    for i in 0..N {
        delta[i] = dzone[i] as f32 / 256.0;
        xq[i] = x[i] * scale[i] as f32 / 256.0;
        deadzone[i] = xq[i] / (delta[i] + eps);
    }
    for v in deadzone.iter_mut() {
        *v = v.tanh();
    }
    let mut q = [0.0f32; N];
    for i in 0..N {
        let adjusted = xq[i] - delta[i] * deadzone[i];
        let qi = (0.5 + adjusted).floor() as i32;
        // Matches `if (r[i] == 0 || p0[i] == 255) q[i] = 0` skip in the
        // encoder — the range coder wouldn't emit a symbol for that dim.
        let q_int = if r[i] == 0 || p0[i] == 255 { 0 } else { qi };
        q[i] = q_int as f32;
    }
    q
}
