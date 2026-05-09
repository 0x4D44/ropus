#![cfg(not(no_reference))]
//! Stage-5 (apply-feedback) — direct FFI scalar fixture for the DRED
//! bitrate helpers.
//!
//! ## Why this test exists
//!
//! Stage 4's reviewers flagged the original Tier-1 differential at the
//! C shim's hard-coded settings as a no-op gate (`dred_bitrate_bps = 0`
//! → none of the f32 ops contribute). The Stage-5 pivot moved to a
//! "drive `compute_dred_bitrate > 0` and assert byte-exact packets"
//! integration test (`dred_bitrate_plumbing_nonzero_diff.rs`), but at
//! the DA's counterexample config that test trips a pre-existing
//! SILK divergence (24 dB SNR vs the 60 dB Tier-2 floor) that has
//! nothing to do with the DRED port. The supervisor's resolution was
//! to ground the f32 ops against C **directly** rather than
//! transitively through the encode pipeline.
//!
//! This test does exactly that: it links C's `compute_dred_bitrate` and
//! `estimate_dred_bitrate` (verbatim copies sit in
//! `harness-deep-plc/dred_encode_shim.c` because both are `static` in
//! `opus_encoder.c`) and calls them alongside the Rust port for ≥10
//! input vectors. Every observable scalar — return value plus all four
//! out-parameter fields — must match byte-exact. If any input vector
//! mismatches the test fails with the exact `(input → C → Rust)`
//! triple so the supervisor can arbitrate.
//!
//! See the journal entry "2026-05-09 16:30 — Stage 4 reviewers complete"
//! for the devil's advocate finding that demanded oracle-grounded
//! fixtures and the supervisor's pivot decision on top of that.

use ropus::opus::encoder::{ropus_test_compute_dred_bitrate, ropus_test_estimate_dred_bitrate};

use ropus_harness_deep_plc::{ropus_c_compute_dred_bitrate, ropus_c_estimate_dred_bitrate};

use std::os::raw::c_int;

// ---------------------------------------------------------------------------
// FFI shims for the C `ropus_c_*` wrappers.
// ---------------------------------------------------------------------------

/// Call the C reference's `estimate_dred_bitrate` and return
/// `(estimated_bits, target_chunks)`.
fn c_estimate_dred_bitrate(q0: i32, d_q: i32, qmax: i32, duration: i32, target_bits: i32) -> (i32, i32) {
    let mut tc: c_int = 0;
    let bits = unsafe {
        ropus_c_estimate_dred_bitrate(q0, d_q, qmax, duration, target_bits, &mut tc as *mut c_int)
    };
    (bits, tc)
}

/// Call the C reference's `compute_dred_bitrate` and return
/// `(dred_bitrate, q0, d_q, qmax, target_chunks)` matching the order
/// returned by `ropus_test_compute_dred_bitrate`.
fn c_compute_dred_bitrate(
    use_in_band_fec: i32,
    packet_loss_perc: i32,
    fs: i32,
    dred_duration: i32,
    bitrate_bps: i32,
    frame_size: i32,
) -> (i32, i32, i32, i32, i32) {
    let mut q0: c_int = 0;
    let mut d_q: c_int = 0;
    let mut qmax: c_int = 0;
    let mut target_chunks: c_int = 0;
    let dred_bitrate = unsafe {
        ropus_c_compute_dred_bitrate(
            use_in_band_fec,
            packet_loss_perc,
            fs,
            dred_duration,
            bitrate_bps,
            frame_size,
            &mut q0 as *mut c_int,
            &mut d_q as *mut c_int,
            &mut qmax as *mut c_int,
            &mut target_chunks as *mut c_int,
        )
    };
    (dred_bitrate, q0, d_q, qmax, target_chunks)
}

// ---------------------------------------------------------------------------
// estimate_dred_bitrate — 5 vectors mirroring the Stage-2 unit-test goldens.
// ---------------------------------------------------------------------------

/// `(q0, d_q, qmax, duration, target_bits)`. Matches the tuples in
/// `ropus/src/opus/encoder.rs::test_estimate_dred_bitrate_golden`.
const ESTIMATE_VECTORS: &[(i32, i32, i32, i32, i32)] = &[
    (15, 5, 15, 4, 1_000_000),
    (9, 3, 15, 100, 1_000_000),
    (4, 5, 15, 100, 10_000),
    (4, 3, 15, 8, 8000),
    (15, 5, 15, 104, 0),
];

#[test]
fn estimate_dred_bitrate_byte_exact_against_c_ref() {
    let mut failures: Vec<String> = Vec::new();
    for &(q0, d_q, qmax, duration, target_bits) in ESTIMATE_VECTORS.iter() {
        let c = c_estimate_dred_bitrate(q0, d_q, qmax, duration, target_bits);
        let r = ropus_test_estimate_dred_bitrate(q0, d_q, qmax, duration, target_bits);
        if c != r {
            failures.push(format!(
                "estimate_dred_bitrate(q0={q0}, dQ={d_q}, qmax={qmax}, dur={duration}, tgt={target_bits}): \
                 C returned (bits={}, tc={}); Rust returned (bits={}, tc={})",
                c.0, c.1, r.0, r.1
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "estimate_dred_bitrate diverged from C ref:\n  - {}",
        failures.join("\n  - ")
    );
}

// ---------------------------------------------------------------------------
// compute_dred_bitrate — 5+ vectors covering FEC on/off, loss zero/15/30,
// bitrate low/mid/high, frame sizes 160/320/480/960, and sample rates
// 8k/16k/24k/48k.
// ---------------------------------------------------------------------------

/// `(label, use_in_band_fec, packet_loss_perc, fs, dred_duration,
/// bitrate_bps, frame_size)`. The label feeds into failure messages so
/// a mismatch immediately points back at the row that caused it.
const COMPUTE_VECTORS: &[(&str, i32, i32, i32, i32, i32, i32)] = &[
    // FEC on, loss=0 → dred_frac = MIN16(.7, 0) = 0 → target_dred=0 →
    // target_chunks=0 → return 0. Smoke-test for the all-zero path
    // through the FEC branch.
    ("fec1_loss0_48k_960_64k", 1, 0, 48_000, 100, 64_000, 960),
    // FEC on, loss=15 → dred_frac = MIN16(.7, 0.45) = 0.45.
    // 16 kHz / 320 / 40 kbps / dur=100 — same shape as the DA
    // counterexample but with FEC enabled, and here the bitrate
    // exceeds bitrate_offset=20000 by enough to make q0 valid.
    ("fec1_loss15_16k_320_40k", 1, 15, 16_000, 100, 40_000, 320),
    // FEC off, loss=30 → MIN16(.8, .55+.30)=.8 → DA's counterexample
    // settings driven straight through `compute_dred_bitrate`. This
    // is the row whose transitive integration test trips SILK drift.
    // Direct FFI sidesteps that entirely.
    ("fec0_loss30_16k_320_40k", 0, 30, 16_000, 320, 40_000, 320),
    // FEC off, loss=0 → 12*0/100 = 0 → return 0. Covers the
    // `packet_loss_perc <= 5` branch's degenerate case.
    ("fec0_loss0_24k_480_24k", 0, 0, 24_000, 50, 24_000, 480),
    // FEC off, loss=2 → 12*2/100 = 0.24 (the small-loss rate-driven
    // branch). 8 kHz / 160 sample frame / 12 kbps stresses the lowest
    // sample rate and the smallest typical frame size.
    ("fec0_loss2_8k_160_12k", 0, 2, 8_000, 200, 12_000, 160),
    // High-bitrate path: bitrate-offset > 36000 picks dQ=3 instead of
    // 5, so this row directly exercises the dQ branch.
    ("fec1_loss10_48k_960_120k", 1, 10, 48_000, 100, 120_000, 960),
    // dred_duration = 0 → max_dred_bits=0 + target_chunks=0 → return 0
    // unconditionally. Confirms the disable path matches between
    // implementations.
    ("dred_off_48k_960_64k", 1, 10, 48_000, 0, 64_000, 960),
    // Very low bitrate where `bitrate_bps - bitrate_offset` goes
    // negative; `IMAX(1, ...)` keeps EC_ILOG in domain. Confirms both
    // implementations clamp identically.
    ("low_bitrate_underflow", 0, 30, 16_000, 100, 5_000, 320),
    // Mid-range loss (FEC=0, loss=10 → MIN16(.8, .55+.1)=0.65). 24 kHz
    // covers the SWB sample rate; 480-sample frame is 20 ms there.
    ("fec0_loss10_24k_480_32k", 0, 10, 24_000, 100, 32_000, 480),
    // 48 kHz, very short DRED duration (4) — exercises the
    // `dred_chunks = (duration+5)/4` minimum path inside
    // estimate_dred_bitrate when called via compute_dred_bitrate.
    ("short_dred_48k_960_48k", 0, 8, 48_000, 4, 48_000, 960),
];

#[test]
fn compute_dred_bitrate_byte_exact_against_c_ref() {
    let mut failures: Vec<String> = Vec::new();
    for &(label, fec, loss, fs, dur, bitrate, frame) in COMPUTE_VECTORS.iter() {
        let c = c_compute_dred_bitrate(fec, loss, fs, dur, bitrate, frame);
        let r = ropus_test_compute_dred_bitrate(fec, loss, fs, dur, bitrate, frame);
        if c != r {
            failures.push(format!(
                "{label}: compute_dred_bitrate(fec={fec}, loss={loss}, fs={fs}, dur={dur}, \
                 bitrate={bitrate}, frame={frame})\n      C    = (dred={}, q0={}, dQ={}, qmax={}, tc={})\n      Rust = (dred={}, q0={}, dQ={}, qmax={}, tc={})",
                c.0, c.1, c.2, c.3, c.4, r.0, r.1, r.2, r.3, r.4,
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "compute_dred_bitrate diverged from C ref:\n  - {}",
        failures.join("\n  - ")
    );
}
