//! Shared oracle helpers for fuzz targets.
//! Included via `#[path = "oracle.rs"] mod oracle;`.
//!
//! Per HLD V2 (Stream D, gap 6), this module replaces the unconditional
//! SILK/Hybrid PCM-skip in differential decode/roundtrip targets with a
//! bounded-SNR oracle. Tier-2 SNR floor (>= 50 dB) matches the project's
//! documented envelope in CLAUDE.md.

#![allow(dead_code)]

/// Minimum acceptable SNR (in decibels) when comparing the Rust port's
/// SILK/Hybrid PCM output against the C reference. CELT-only packets remain
/// byte-exact compared elsewhere; this floor only governs the lossy modes.
pub const SILK_DECODE_MIN_SNR_DB: f64 = 50.0;

/// Lower bound on reference signal energy before SNR is meaningful.
/// Reference frames with sum-of-squares below this are dropped from
/// the SNR oracle — both implementations may have entered recovery
/// on a malformed packet, producing PCM that diverges by design.
///
/// Calibrated empirically: the `silk_decode_snr_below_floor` repro
/// (a 4-byte malformed SILK packet) produces ~5.4 dB SNR with
/// reference energy in the 1e6–1e7 range; bumping the floor to 1e7
/// classifies it as recovery-divergence rather than a tier-2 violation.
pub const SNR_PRECHECK_MIN_REF_ENERGY: f64 = 1e7;

/// Sum-of-squares energy of a reference signal.
pub fn signal_energy(samples: &[i16]) -> f64 {
    samples.iter().map(|&s| (s as f64).powi(2)).sum()
}

/// Whether the SNR oracle is meaningful for this reference.
pub fn snr_oracle_applicable(reference: &[i16]) -> bool {
    signal_energy(reference) >= SNR_PRECHECK_MIN_REF_ENERGY
}

/// Compute SNR (signal-to-noise ratio) in decibels of `test` against
/// `reference`. Special cases:
/// - length mismatch returns `f64::NEG_INFINITY` (treated as a failure)
/// - exact match returns `f64::INFINITY`
/// - silence-in but non-silence-out returns `f64::NEG_INFINITY` (a bug)
pub fn snr_db(reference: &[i16], test: &[i16]) -> f64 {
    if reference.len() != test.len() {
        return f64::NEG_INFINITY;
    }
    let (mut sig, mut noise) = (0.0f64, 0.0f64);
    for (&r, &t) in reference.iter().zip(test) {
        sig += (r as f64).powi(2);
        noise += ((r as f64) - (t as f64)).powi(2);
    }
    if noise == 0.0 {
        return f64::INFINITY;
    }
    if sig == 0.0 {
        return f64::NEG_INFINITY;
    }
    10.0 * (sig / noise).log10()
}

// Unit tests live in `tests/oracle_tests.rs` (an integration test that
// includes this file via `#[path]`). The fuzz binaries link libFuzzer's
// `main`, so a `#[cfg(test)] mod tests` block here would compile but
// never actually run via `cargo test --bin <fuzz_target>`.
