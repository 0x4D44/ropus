//! Positive-control test for Stage 3 of the Cluster A HLD V2 trace work:
//! confirms that running an encode populates the `ropus::silk_trace`
//! ring with V2 tuples (`boundary_id >= 100`).
//!
//! Why this exists: Stage 3 wired 10 V2 trace pushes inside
//! `silk_encode_frame_fix` (`/home/md/language/ropus/ropus/src/silk/encoder.rs`).
//! Stage 4 (the diff-tool side) is not yet landed, so the
//! `fuzz_repro_diff` print path can't yet decode V2 tuples. This test
//! is the cheapest sanity check that the producer side actually
//! populates the ring — independent of the diff tool.
//!
//! It runs entirely against the Rust codec (no FFI), and is gated on
//! `trace-silk-encode` being enabled (which `ropus-harness` does
//! unconditionally — see `harness/Cargo.toml`).

#![cfg(not(no_reference))]

use ropus::opus::encoder::{OPUS_APPLICATION_VOIP, OpusEncoder};

/// Deterministic PCM pattern (mirrors `patterned_pcm_i16` from
/// `c_ref_differential.rs` but kept local to avoid cross-test dep).
fn patterned_pcm_i16(frame_size: usize, channels: usize, seed: i32) -> Vec<i16> {
    let mut out = vec![0i16; frame_size * channels];
    let mut s = seed.wrapping_mul(2_147_483_647) as u32 | 1;
    for sample in out.iter_mut() {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        // Limit amplitude so the encoder doesn't clip; full range isn't needed.
        *sample = ((s as i32) >> 16) as i16 / 4;
    }
    out
}

#[test]
fn silk_trace_v2_tuples_emitted_on_silk_encode() {
    // 16 kHz mono VOIP — a SILK-only configuration so `silk_encode_frame_fix`
    // is exercised and V2 boundaries 100..=109 get a chance to fire.
    let mut enc = OpusEncoder::new(16000, 1, OPUS_APPLICATION_VOIP)
        .expect("Rust encoder create failed");
    enc.set_bitrate(16000);

    // Clear any residue from a prior test in the same process.
    ropus::silk_trace::clear();

    // Encode a few 20 ms frames so we cover both the prefill path and at
    // least one rate-control-loop iteration (b106/107/108/109).
    let frame_size = 320; // 20 ms at 16 kHz
    let mut buf = vec![0u8; 1500];
    for seed in 0..4 {
        let pcm = patterned_pcm_i16(frame_size, 1, seed * 1337);
        let cap = buf.len() as i32;
        let _len = enc
            .encode(&pcm, frame_size as i32, &mut buf, cap)
            .expect("encode failed");
    }

    let snap = ropus::silk_trace::snapshot();
    let v2_count = snap.iter().filter(|t| t.boundary_id >= 100).count();
    let v1_count = snap.iter().filter(|t| t.boundary_id < 100 && t.boundary_id > 0).count();

    // Expect at least one V2 tuple. Stage 3 wired 10 boundaries, but
    // some (e.g. prefill-skipped 100, last-iter-only 109) won't fire on
    // every frame. A single V2 tuple is sufficient evidence the wiring
    // is reachable.
    assert!(
        v2_count > 0,
        "expected at least one V2 trace tuple (boundary_id >= 100), got {} total tuples (V1: {})",
        snap.len(),
        v1_count,
    );
}
