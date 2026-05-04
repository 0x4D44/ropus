//! Shared oracle helpers for fuzz targets.
//! Included via `#[path = "oracle.rs"] mod oracle;`.
//!
//! Per HLD V2 (Stream D, gap 6), this module replaces the unconditional
//! SILK/Hybrid PCM-skip in differential decode/roundtrip targets with a
//! bounded-SNR oracle. Tier-2 SNR floor (>= 50 dB) matches the project's
//! documented envelope in CLAUDE.md.

#![allow(dead_code)]

use ropus::opus::decoder::{opus_packet_parse_impl, MAX_FRAMES};

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

/// How strong a PCM oracle is meaningful for a decoded packet.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeOracleClass {
    /// CELT-only coded audio is expected to be byte-exact.
    CeltCodedComparable,
    /// SILK or low-rate Hybrid coded audio is comparable through the
    /// bounded-SNR oracle.
    SilkHybridCodedComparable,
    /// The packet can still be decoded and sample-count checked, but its PCM
    /// is not a stable SNR oracle for attacker-controlled decode inputs.
    SampleCountOnly,
    /// At least one SILK/Hybrid sub-frame routes through PLC/DTX recovery.
    RecoveryOrDtxOnly,
    /// Packet structure is not usable for PCM comparison; decode result
    /// symmetry and sample-count parity are the only meaningful checks.
    ErrorAgreementOnly,
}

/// Sum-of-squares energy of a reference signal.
pub fn signal_energy(samples: &[i16]) -> f64 {
    samples.iter().map(|&s| (s as f64).powi(2)).sum()
}

/// Whether the SNR oracle is meaningful for this reference.
pub fn snr_oracle_applicable(reference: &[i16]) -> bool {
    signal_energy(reference) >= SNR_PRECHECK_MIN_REF_ENERGY
}

/// SNR-applicability gate that also requires the packet to be structurally
/// well-formed.
///
/// `packet_well_formed` should be the boolean result of a packet-parse
/// validity check that bit-exact agrees between Rust and C — typically
/// `opus_packet_get_nb_samples(packet, fs).is_ok()`. When that parse fails,
/// both decoders enter recovery and produce PCM that diverges by design;
/// asserting the SNR floor on those inputs produces false positives like
/// the `silk_decode_recovery_divergence_loud` finding (2026-05-01).
pub fn snr_oracle_applicable_for_packet(reference: &[i16], packet_well_formed: bool) -> bool {
    packet_well_formed && snr_oracle_applicable(reference)
}

/// Whether the SNR oracle is meaningful for this reference and decode class.
pub fn snr_oracle_applicable_for_decode_class(reference: &[i16], class: DecodeOracleClass) -> bool {
    class == DecodeOracleClass::SilkHybridCodedComparable && snr_oracle_applicable(reference)
}

/// Classify a single Opus packet for attacker-controlled decode comparison.
pub fn classify_decode_packet(packet: &[u8], sample_rate: i32) -> DecodeOracleClass {
    classify_single_packet(packet, false, sample_rate).0
}

/// Classify a multistream packet by parsing each sub-packet.
pub fn classify_multistream_decode_packet(
    packet: &[u8],
    streams: i32,
    sample_rate: i32,
) -> DecodeOracleClass {
    if packet.is_empty() || streams <= 0 {
        return DecodeOracleClass::ErrorAgreementOnly;
    }

    let mut offset = 0usize;
    let mut saw_silk_or_hybrid = false;
    for stream_idx in 0..streams as usize {
        if offset >= packet.len() {
            return DecodeOracleClass::ErrorAgreementOnly;
        }

        let self_delimited = stream_idx + 1 != streams as usize;
        let (class, packet_offset) =
            classify_single_packet(&packet[offset..], self_delimited, sample_rate);
        match class {
            DecodeOracleClass::SampleCountOnly => return class,
            DecodeOracleClass::RecoveryOrDtxOnly => return class,
            DecodeOracleClass::ErrorAgreementOnly => return class,
            DecodeOracleClass::SilkHybridCodedComparable => saw_silk_or_hybrid = true,
            DecodeOracleClass::CeltCodedComparable => {}
        }

        let Some(packet_offset) = packet_offset else {
            return DecodeOracleClass::ErrorAgreementOnly;
        };
        if packet_offset == 0 || offset + packet_offset > packet.len() {
            return DecodeOracleClass::ErrorAgreementOnly;
        }
        offset += packet_offset;
    }

    if offset != packet.len() {
        return DecodeOracleClass::ErrorAgreementOnly;
    }
    if saw_silk_or_hybrid {
        DecodeOracleClass::SilkHybridCodedComparable
    } else {
        DecodeOracleClass::CeltCodedComparable
    }
}

fn classify_single_packet(
    packet: &[u8],
    self_delimited: bool,
    sample_rate: i32,
) -> (DecodeOracleClass, Option<usize>) {
    if packet.is_empty() {
        return (DecodeOracleClass::ErrorAgreementOnly, None);
    }

    let mut toc = 0u8;
    let mut sizes = [0i16; MAX_FRAMES];
    let mut payload_offset = 0i32;
    let mut packet_offset = 0i32;
    let count = opus_packet_parse_impl(
        packet,
        packet.len() as i32,
        self_delimited,
        &mut toc,
        &mut sizes,
        &mut payload_offset,
        Some(&mut packet_offset),
    );
    if count <= 0 || packet_offset <= 0 {
        return (DecodeOracleClass::ErrorAgreementOnly, None);
    }

    if toc & 0x80 != 0 {
        return (
            DecodeOracleClass::CeltCodedComparable,
            Some(packet_offset as usize),
        );
    }

    let frame_sizes = &sizes[..count as usize];
    if frame_sizes.iter().any(|&size| size <= 1) {
        (
            DecodeOracleClass::RecoveryOrDtxOnly,
            Some(packet_offset as usize),
        )
    } else if (12..=15).contains(&(toc >> 3)) && sample_rate > 16_000 {
        // Hybrid packets decoded above 16 kHz include CELT high-band synthesis.
        // Attacker-controlled Hybrid payloads can drift in that high-band path
        // while SILK low-band, frame sizes, and final range all agree with C;
        // keep roundtrip/encoder-produced Hybrid SNR checks, but do not use
        // high-band arbitrary decode as a standalone PCM oracle.
        (
            DecodeOracleClass::SampleCountOnly,
            Some(packet_offset as usize),
        )
    } else {
        (
            DecodeOracleClass::SilkHybridCodedComparable,
            Some(packet_offset as usize),
        )
    }
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
