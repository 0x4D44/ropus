//! Integration tests for the shared fuzz-target oracle helpers.
//!
//! `tests/fuzz/fuzz_targets/oracle.rs` is included into each fuzz binary via
//! `#[path = "oracle.rs"] mod oracle;`. Those fuzz binaries link libFuzzer's
//! `main`, which intercepts the libtest CLI and prevents `cargo test --bin
//! <fuzz_target>` from actually running unit tests inside the fuzz module.
//!
//! Pulling the same source into a regular integration test gives the helper
//! a normal libtest entry point, so the SNR oracle's invariants are
//! validated without standing up a separate library crate (HLD V2 Stream D
//! goal: no extra helper module beyond `oracle.rs`).

#[path = "../fuzz_targets/oracle.rs"]
mod oracle;

use oracle::*;

#[test]
fn identical_signals_return_infinity() {
    let s: Vec<i16> = (0..1000).map(|i| (i as i16).wrapping_mul(13)).collect();
    let snr = snr_db(&s, &s);
    assert!(snr.is_infinite() && snr > 0.0, "expected +INF, got {snr}");
}

#[test]
fn one_lsb_off_is_about_96_db() {
    // 16-bit signal with one-LSB error per sample -> SNR ~= 6.02 * 16 = ~96 dB.
    let r: Vec<i16> = (0..1000).map(|i| ((i % 256) as i16) << 7).collect();
    let t: Vec<i16> = r.iter().map(|x| x.wrapping_add(1)).collect();
    let snr = snr_db(&r, &t);
    assert!(snr > 70.0 && snr < 110.0, "expected ~96 dB, got {snr}");
}

#[test]
fn silence_in_noise_out_returns_neg_infinity() {
    let zeros = vec![0i16; 100];
    let noisy: Vec<i16> = (0..100).map(|i| (i % 10) as i16).collect();
    let snr = snr_db(&zeros, &noisy);
    assert!(snr.is_infinite() && snr < 0.0, "expected -INF, got {snr}");
}

#[test]
fn length_mismatch_returns_neg_infinity() {
    let a = vec![100i16; 50];
    let b = vec![100i16; 100];
    let snr = snr_db(&a, &b);
    assert!(snr.is_infinite() && snr < 0.0, "expected -INF, got {snr}");
}

#[test]
fn snr_oracle_applicable_for_typical_signal() {
    // ~1000-amplitude sinusoidal-ish content: well above the energy floor.
    let r: Vec<i16> = (0..960)
        .map(|i| ((i * 31) as i16).wrapping_add(1000))
        .collect();
    assert!(
        snr_oracle_applicable(&r),
        "expected oracle to apply, energy={}",
        signal_energy(&r)
    );
}

#[test]
fn snr_oracle_skipped_for_silence_or_near_silence() {
    let zeros = vec![0i16; 1000];
    assert!(
        !snr_oracle_applicable(&zeros),
        "expected oracle to skip pure silence"
    );
    // [1, -1, 1, -1, ...] over 1000 samples: energy = 1000, well below 1e6.
    let near_silence: Vec<i16> = (0..1000).map(|i| if i & 1 == 0 { 1 } else { -1 }).collect();
    assert!(
        !snr_oracle_applicable(&near_silence),
        "expected oracle to skip ±1 LSB toggle, energy={}",
        signal_energy(&near_silence)
    );
}

#[test]
fn snr_oracle_for_packet_skips_when_not_well_formed() {
    // Loud reference (well above energy floor) but the packet didn't validate.
    // Mirrors the silk_decode_recovery_divergence_loud finding (2026-05-01):
    // both decoders enter recovery on a malformed packet, their PCM diverges
    // wildly, and asserting SNR would produce a false positive.
    let loud: Vec<i16> = (0..960)
        .map(|i| ((i * 31) as i16).wrapping_add(20_000))
        .collect();
    assert!(
        snr_oracle_applicable(&loud),
        "preflight: loud signal must pass the energy precheck"
    );
    assert!(
        !snr_oracle_applicable_for_packet(&loud, false),
        "expected packet-not-well-formed to override the energy precheck"
    );
}

#[test]
fn snr_oracle_for_packet_passes_when_both_gates_clear() {
    let loud: Vec<i16> = (0..960)
        .map(|i| ((i * 31) as i16).wrapping_add(20_000))
        .collect();
    assert!(snr_oracle_applicable_for_packet(&loud, true));
}

#[test]
fn snr_oracle_for_packet_skips_silence_even_if_well_formed() {
    let zeros = vec![0i16; 1000];
    assert!(!snr_oracle_applicable_for_packet(&zeros, true));
}

#[test]
fn standalone_recovery_fixture_classifies_as_recovery_or_dtx() {
    let data = include_bytes!("../known_failures/silk_decode_recovery_divergence_loud/crash.bin");
    let packet = &data[2..];
    assert_eq!(
        classify_decode_packet(packet, 48_000),
        DecodeOracleClass::RecoveryOrDtxOnly
    );

    let loud: Vec<i16> = (0..960)
        .map(|i| ((i * 31) as i16).wrapping_add(20_000))
        .collect();
    assert!(snr_oracle_applicable(&loud));
    assert!(!snr_oracle_applicable_for_decode_class(
        &loud,
        DecodeOracleClass::RecoveryOrDtxOnly
    ));
}

#[test]
fn silk_coded_packet_classifies_as_snr_comparable() {
    let packet = [0x00, 0xaa, 0xbb, 0xcc];
    assert_eq!(
        classify_decode_packet(&packet, 48_000),
        DecodeOracleClass::SilkHybridCodedComparable
    );

    let loud: Vec<i16> = (0..960)
        .map(|i| ((i * 31) as i16).wrapping_add(20_000))
        .collect();
    assert!(snr_oracle_applicable_for_decode_class(
        &loud,
        DecodeOracleClass::SilkHybridCodedComparable
    ));
}

#[test]
fn hybrid_highband_packet_uses_sample_count_only_at_high_rates() {
    let packet = [0x60, 0xaa, 0xbb, 0xcc];
    assert_eq!(
        classify_decode_packet(&packet, 48_000),
        DecodeOracleClass::SampleCountOnly
    );
    assert_eq!(
        classify_decode_packet(&packet, 16_000),
        DecodeOracleClass::SilkHybridCodedComparable
    );
}

#[test]
fn typical_encode_decode_drift_passes_threshold() {
    // Synthetic ~1% RMS drift -- should be well above the 50 dB floor.
    let r: Vec<i16> = (0..960)
        .map(|i| ((i * 31) as i16).wrapping_add(1000))
        .collect();
    let t: Vec<i16> = r
        .iter()
        .enumerate()
        .map(|(idx, &x)| {
            if idx % 100 == 0 {
                x.wrapping_add(50)
            } else {
                x
            }
        })
        .collect();
    let snr = snr_db(&r, &t);
    assert!(
        snr >= SILK_DECODE_MIN_SNR_DB,
        "expected >= {SILK_DECODE_MIN_SNR_DB} dB, got {snr}"
    );
}
