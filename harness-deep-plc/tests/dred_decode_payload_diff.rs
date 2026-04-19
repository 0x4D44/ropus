//! Stage 8.7 C-differential test — `dred_ec_decode`.
//!
//! The Rust round-trip (in `ropus/src/dnn/dred.rs::tests`) proves
//! `encode_silk_frame` + `ec_decode` form an inverse pair. This test adds
//! the missing oracle: **the C reference emitter and the Rust decoder
//! agree bit-for-bit**, *and* **the C and Rust decoders agree bit-for-bit
//! on the same C-emitted bytes**. Together these close the compensating-
//! error gap 8.6 flagged.
//!
//! Flow:
//!   1. Poke synthetic latents + state into both a C `DREDEnc` and a Rust
//!      `DREDEnc` (no RDOVAE upstream — buffers set via new 8.7 shims).
//!   2. Encode on the C side to byte buffer `c_bytes`.
//!   3. Decode `c_bytes` with BOTH the Rust `OpusDred::ec_decode` and the
//!      C `dred_ec_decode`, assert byte-equal state + latents + process
//!      metadata between the two sides.
//!   4. As a symmetric cross-check, Rust-encode the same inputs to
//!      `r_bytes` and assert `c_bytes == r_bytes` (regression guard against
//!      any encoder drift that Stage 8.6 didn't already catch).
//!
//! Because the payload is integer range-coded, there's no float drift to
//! worry about: tier 1 is the only acceptable outcome.

use ropus::dnn::dred::{
    DRED_LATENT_DIM, DRED_MAX_DATA_SIZE, DRED_MAX_FRAMES, DRED_NUM_REDUNDANCY_FRAMES,
    DRED_STATE_DIM, DREDEnc, OpusDred, compute_quantizer,
};

use ropus_harness_deep_plc::{
    ropus_test_dred_ec_decode, ropus_test_dred_encode_silk_frame, ropus_test_dredenc_free,
    ropus_test_dredenc_new, ropus_test_dredenc_set_bookkeeping,
    ropus_test_dredenc_set_latents_buffer, ropus_test_dredenc_set_state_buffer,
};

/// Deterministic xorshift-driven latent magnitudes. Mildly scaled so the
/// quantiser produces a mix of positive, negative, and zero outputs.
fn synth_float(seed: &mut u32) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    let s = (*seed as i32) as f64 / (i32::MAX as f64);
    (s * 2.5) as f32
}

/// Convenience: index into `a` / `b` and return the first differing
/// f32 (via `to_bits()` for NaN / -0.0 robustness). `None` if bit-exact.
fn first_f32_divergence(a: &[f32], b: &[f32]) -> Option<(usize, f32, f32)> {
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if x.to_bits() != y.to_bits() {
            return Some((i, x, y));
        }
    }
    None
}

#[test]
fn dred_ec_decode_matches_c_on_c_emitted_payload() {
    // Constants sized for a small but coverage-meaningful test.
    const NUM_CHUNKS: i32 = 4;
    const NUM_LATENT_FRAMES_WRITTEN: usize = (2 * NUM_CHUNKS) as usize;
    let q0: i32 = 6;
    let d_q: i32 = 0;
    let qmax: i32 = 15;

    // --- Synthesise inputs (shared by both C and Rust sides) ---
    let mut seed = 0xF00DCAFE_u32;
    let mut state_input = [0.0f32; DRED_STATE_DIM];
    for x in state_input.iter_mut() {
        *x = synth_float(&mut seed);
    }
    // The encoder reads chunk `i` (i = 0, 2, 4, ...) from
    // `latents_buffer[i * DRED_LATENT_DIM]`, so we only need to populate
    // even frames. Odd frames are never read by `dred_encode_silk_frame`,
    // so we leave them at zero to keep the buffer minimal.
    let mut latents_input = [0.0f32; NUM_LATENT_FRAMES_WRITTEN * DRED_LATENT_DIM];
    for chunk in 0..(NUM_CHUNKS as usize) {
        for i in 0..DRED_LATENT_DIM {
            latents_input[2 * chunk * DRED_LATENT_DIM + i] = synth_float(&mut seed);
        }
    }
    let activity_mem = vec![1u8; 4 * DRED_MAX_FRAMES];

    // --- C side: allocate, seed, encode ---
    let c_enc = unsafe { ropus_test_dredenc_new(48000, 1) };
    assert!(
        !c_enc.is_null(),
        "C dredenc_new failed (weights blob probably absent)"
    );
    unsafe {
        ropus_test_dredenc_set_state_buffer(c_enc, state_input.as_ptr(), DRED_STATE_DIM as i32);
        ropus_test_dredenc_set_latents_buffer(
            c_enc,
            latents_input.as_ptr(),
            latents_input.len() as i32,
        );
        ropus_test_dredenc_set_bookkeeping(
            c_enc,
            0,                  /* latent_offset */
            2 * NUM_CHUNKS + 2, /* latents_buffer_fill — + slack chunk */
            0,                  /* dred_offset */
            0,                  /* last_extra_dred_offset */
        );
    }
    let mut c_bytes = vec![0u8; DRED_MAX_DATA_SIZE];
    let c_nbytes = unsafe {
        ropus_test_dred_encode_silk_frame(
            c_enc,
            c_bytes.as_mut_ptr(),
            NUM_CHUNKS,
            DRED_MAX_DATA_SIZE as i32,
            q0,
            d_q,
            qmax,
            activity_mem.as_ptr() as *mut _,
        )
    };
    unsafe { ropus_test_dredenc_free(c_enc) };
    assert!(
        c_nbytes > 0,
        "C encoder produced 0 bytes — buffer fill wrong?"
    );
    c_bytes.truncate(c_nbytes as usize);

    // --- Rust side: same inputs, same encoder ---
    let mut r_enc = DREDEnc::new_unloaded(48000, 1);
    r_enc.loaded = true;
    r_enc.latent_offset = 0;
    r_enc.latents_buffer_fill = 2 * NUM_CHUNKS + 2;
    r_enc.dred_offset = 0;
    r_enc.last_extra_dred_offset = 0;
    r_enc.state_buffer[..DRED_STATE_DIM].copy_from_slice(&state_input);
    r_enc.latents_buffer[..latents_input.len()].copy_from_slice(&latents_input);
    let mut r_bytes = vec![0u8; DRED_MAX_DATA_SIZE];
    let r_nbytes = r_enc.encode_silk_frame(
        &mut r_bytes,
        NUM_CHUNKS,
        DRED_MAX_DATA_SIZE,
        q0,
        d_q,
        qmax,
        &activity_mem,
    );
    assert!(r_nbytes > 0);
    r_bytes.truncate(r_nbytes as usize);

    // --- Byte-level cross-check: C encoder and Rust encoder agree ---
    // This is technically Stage 8.6 territory, but 8.6 was gated on
    // LPCNet drift that 8.7's direct-buffer-poke avoids — so with the
    // same byte-level inputs, the encoders must match. Any drift here
    // would invalidate the downstream decoder diff test.
    assert_eq!(
        c_bytes, r_bytes,
        "C and Rust encoders disagree on the same direct-poke inputs — \
         this is purely the integer range coder, so drift here points to \
         a real bug in `encode_silk_frame` / `dred_encode_latents`."
    );

    // --- Decode via Rust ---
    let mut rust_dred = OpusDred::default();
    let rust_nb = rust_dred.ec_decode(&c_bytes, 2 * NUM_CHUNKS, 0);
    assert!(rust_nb > 0);

    // --- Decode via C ---
    let mut c_state = [0.0f32; DRED_STATE_DIM];
    let mut c_latents = [0.0f32; (DRED_NUM_REDUNDANCY_FRAMES / 2) * (DRED_LATENT_DIM + 1)];
    let mut c_nb_latents: i32 = 0;
    let mut c_process_stage: i32 = 0;
    let mut c_dred_offset: i32 = 0;
    let c_ret = unsafe {
        ropus_test_dred_ec_decode(
            c_bytes.as_ptr(),
            c_bytes.len() as i32,
            2 * NUM_CHUNKS,
            0,
            c_state.as_mut_ptr(),
            c_latents.as_mut_ptr(),
            &mut c_nb_latents,
            &mut c_process_stage,
            &mut c_dred_offset,
        )
    };

    // --- Metadata parity ---
    assert_eq!(
        rust_nb, c_ret,
        "C and Rust decoders report different chunk counts"
    );
    assert_eq!(rust_dred.nb_latents, c_nb_latents);
    assert_eq!(rust_dred.process_stage, c_process_stage);
    assert_eq!(rust_dred.dred_offset, c_dred_offset);

    // --- State bit-exact ---
    if let Some((i, cv, rv)) = first_f32_divergence(&c_state, &rust_dred.state) {
        panic!(
            "state[{i}] diverges: C={:?} ({:#x}) Rust={:?} ({:#x})",
            cv,
            cv.to_bits(),
            rv,
            rv.to_bits(),
        );
    }

    // --- Latents bit-exact across the full buffer ---
    if let Some((i, cv, rv)) = first_f32_divergence(&c_latents, &rust_dred.latents) {
        panic!(
            "latents[{i}] diverges: C={:?} ({:#x}) Rust={:?} ({:#x})",
            cv,
            cv.to_bits(),
            rv,
            rv.to_bits(),
        );
    }

    eprintln!(
        "Tier 1 achieved on C-emitted payload: {} chunks, {} bytes, \
         bit-exact across state + latents on both C/Rust decoders.",
        rust_nb,
        c_bytes.len(),
    );

    // Sanity-check one of the latent trailing slots (`q_level*.125 - 1`
    // tag). If compute_quantizer is reachable and the header was parsed,
    // this slot must equal the formula. Catches subtle off-by-one errors
    // where the 26th slot isn't written.
    for chunk in 0..(rust_nb as usize) {
        let q_level = compute_quantizer(q0, d_q, qmax, chunk as i32);
        let expected = (q_level as f32) * 0.125 - 1.0;
        let base = chunk * (DRED_LATENT_DIM + 1);
        assert_eq!(
            rust_dred.latents[base + DRED_LATENT_DIM].to_bits(),
            expected.to_bits(),
            "chunk[{chunk}] q-tag slot not written",
        );
    }
}
