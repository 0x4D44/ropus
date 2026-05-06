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
use std::collections::HashSet;
use std::fs;
use std::path::Path;

const FUZZ_DECODE_SAMPLE_RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];

fn fuzz_decode_packet_and_sample_rate(input: &[u8]) -> Option<(&[u8], i32)> {
    if input.len() < 3 {
        return None;
    }
    let sample_rate =
        FUZZ_DECODE_SAMPLE_RATES[(input[0] as usize) % FUZZ_DECODE_SAMPLE_RATES.len()];
    Some((&input[2..], sample_rate))
}

fn record_fuzz_decode_dir(counts: &mut DecodeOracleClassCounts, dir: &Path) {
    let mut paths: Vec<_> = fs::read_dir(dir)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", dir.display()))
        .map(|entry| {
            entry
                .unwrap_or_else(|err| panic!("failed to read entry in {}: {err}", dir.display()))
                .path()
        })
        .filter(|path| path.is_file())
        .collect();
    paths.sort();

    for path in paths {
        let input = fs::read(&path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
        if let Some((packet, sample_rate)) = fuzz_decode_packet_and_sample_rate(&input) {
            counts.record(classify_decode_packet(packet, sample_rate));
        }
    }
}

#[test]
fn decode_oracle_classes_are_self_describing() {
    let mut stable_names = HashSet::new();
    for class in DECODE_ORACLE_CLASSES {
        assert!(
            stable_names.insert(class.stable_name()),
            "duplicate stable name for {class:?}"
        );
        assert!(!class.stable_name().is_empty());
        assert!(!class.asserted_property().is_empty());
        assert!(!class.matrix_class().is_empty());

        if class.is_weakened() {
            assert!(
                class
                    .unasserted_property()
                    .is_some_and(|property| !property.is_empty()),
                "weakened class {class:?} must describe its oracle debt"
            );
        } else {
            assert_eq!(
                class.unasserted_property(),
                None,
                "strong class {class:?} should not report oracle debt"
            );
        }
    }
}

#[test]
fn decode_oracle_weakened_class_boundaries_are_explicit() {
    assert!(!DecodeOracleClass::CeltCodedComparable.is_weakened());
    assert!(!DecodeOracleClass::SilkHybridCodedComparable.is_weakened());
    assert!(DecodeOracleClass::SampleCountOnly.is_weakened());
    assert!(DecodeOracleClass::RecoveryOrDtxOnly.is_weakened());
    assert!(DecodeOracleClass::ErrorAgreementOnly.is_weakened());

    assert_eq!(
        DecodeOracleClass::CeltCodedComparable.stable_name(),
        "celt-coded-comparable"
    );
    assert_eq!(
        DecodeOracleClass::SilkHybridCodedComparable.stable_name(),
        "silk-hybrid-snr-comparable"
    );
    assert_eq!(
        DecodeOracleClass::SampleCountOnly.stable_name(),
        "sample-count-only"
    );
    assert_eq!(
        DecodeOracleClass::RecoveryOrDtxOnly.stable_name(),
        "recovery-or-dtx-only"
    );
    assert_eq!(
        DecodeOracleClass::ErrorAgreementOnly.stable_name(),
        "error-agreement-only"
    );
}

#[test]
fn decode_oracle_class_counts_track_strong_and_weakened_totals() {
    let mut counts = DecodeOracleClassCounts::new();
    for class in [
        DecodeOracleClass::CeltCodedComparable,
        DecodeOracleClass::SilkHybridCodedComparable,
        DecodeOracleClass::SampleCountOnly,
        DecodeOracleClass::SampleCountOnly,
        DecodeOracleClass::RecoveryOrDtxOnly,
        DecodeOracleClass::ErrorAgreementOnly,
    ] {
        counts.record(class);
    }

    assert_eq!(counts.total(), 6);
    assert_eq!(counts.strong_total(), 2);
    assert_eq!(counts.weakened_total(), 4);
    assert_eq!(counts.count_for(DecodeOracleClass::SampleCountOnly), 2);
    assert_eq!(
        counts
            .rows()
            .map(|row| (row.class.stable_name(), row.count)),
        [
            ("celt-coded-comparable", 1),
            ("silk-hybrid-snr-comparable", 1),
            ("sample-count-only", 2),
            ("recovery-or-dtx-only", 1),
            ("error-agreement-only", 1),
        ]
    );
}

#[test]
fn decode_oracle_report_is_deterministic_and_includes_zero_counts() {
    let mut counts = DecodeOracleClassCounts::new();
    counts.record(DecodeOracleClass::SilkHybridCodedComparable);
    counts.record(DecodeOracleClass::ErrorAgreementOnly);

    let report = counts.format_report("unit fixture");
    assert_eq!(
        report,
        concat!(
            "decode oracle class report: unit fixture\n",
            "total: 2\n",
            "strong: 1\n",
            "weakened: 1\n",
            "classes:\n",
            "- class: celt-coded-comparable\n",
            "  count: 0\n",
            "  debt: no\n",
            "  asserts: Exact PCM/float-bit parity where the target path supports it.\n",
            "  does_not_assert: none\n",
            "- class: silk-hybrid-snr-comparable\n",
            "  count: 1\n",
            "  debt: no\n",
            "  asserts: SNR-gated parity when the reference energy floor is met.\n",
            "  does_not_assert: none\n",
            "- class: sample-count-only\n",
            "  count: 0\n",
            "  debt: yes\n",
            "  asserts: Decode result symmetry and sample-count/shape parity only.\n",
            "  does_not_assert: PCM sample values, exact PCM parity, and SNR threshold are not asserted for attacker-controlled high-band Hybrid decode inputs.\n",
            "- class: recovery-or-dtx-only\n",
            "  count: 0\n",
            "  debt: yes\n",
            "  asserts: Decode result symmetry and sample-count/shape parity only.\n",
            "  does_not_assert: PCM sample values, exact PCM parity, and SNR threshold are not asserted when at least one SILK/Hybrid sub-frame routes through PLC/DTX-style recovery.\n",
            "- class: error-agreement-only\n",
            "  count: 1\n",
            "  debt: yes\n",
            "  asserts: Decode success/error symmetry only; sample count only when both sides decode.\n",
            "  does_not_assert: PCM sample values, exact PCM parity, SNR threshold, and in some cases sample-count expectations are not asserted because packet structure is not usable for PCM comparison.\n",
        )
    );
}

#[test]
fn summarize_decode_packets_records_class_distribution() {
    let packets = [
        (&[0x80, 0xaa][..], 48_000),
        (&[0x00, 0xaa, 0xbb, 0xcc][..], 16_000),
        (&[0x60, 0xaa, 0xbb, 0xcc][..], 48_000),
        (&[][..], 48_000),
    ];
    let counts = summarize_decode_packets(packets);

    assert_eq!(counts.total(), 4);
    assert_eq!(counts.count_for(DecodeOracleClass::CeltCodedComparable), 1);
    assert_eq!(
        counts.count_for(DecodeOracleClass::SilkHybridCodedComparable),
        1
    );
    assert_eq!(counts.count_for(DecodeOracleClass::SampleCountOnly), 1);
    assert_eq!(counts.count_for(DecodeOracleClass::ErrorAgreementOnly), 1);
}

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
fn malformed_packet_classifies_as_error_agreement_only() {
    assert_eq!(
        classify_decode_packet(&[], 48_000),
        DecodeOracleClass::ErrorAgreementOnly
    );
}

#[test]
fn celt_coded_packet_classifies_as_comparable() {
    let packet = [0x80, 0xaa];
    assert_eq!(
        classify_decode_packet(&packet, 48_000),
        DecodeOracleClass::CeltCodedComparable
    );
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

#[test]
#[ignore = "operator report; run with --ignored --nocapture"]
fn decode_oracle_class_distribution_report() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut counts = DecodeOracleClassCounts::new();

    record_fuzz_decode_dir(&mut counts, &root.join("seeds/fuzz_decode"));
    record_fuzz_decode_dir(&mut counts, &root.join("crashes/fuzz_decode"));

    let recovery =
        include_bytes!("../known_failures/silk_decode_recovery_divergence_loud/crash.bin");
    if let Some((packet, sample_rate)) = fuzz_decode_packet_and_sample_rate(recovery) {
        counts.record(classify_decode_packet(packet, sample_rate));
    }

    println!(
        "{}",
        counts.format_report("committed fuzz_decode seeds, crashes, and known failures")
    );
}
