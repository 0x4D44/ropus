# Oracle calibration: multistream SILK/Hybrid recovery divergence

## Repro

The 4 crash files are libFuzzer Arbitrary inputs for `fuzz_multistream`
that triggered the SILK/Hybrid SNR oracle assertion at
`fuzz_multistream.rs:457` during the 24h campaign hour 1.

```
cargo +nightly fuzz run --fuzz-dir tests/fuzz fuzz_multistream \
    tests/fuzz/known_failures/multistream-decode-recovery-divergence/crash-00.bin
```

## What this is

Same class as the standalone `silk_decode_recovery_divergence_loud`
finding: a structurally-parseable but malformed packet decodes
successfully on both sides with sample-count parity, but the PCM
outputs diverge wildly (3-7 dB SNR vs the 50 dB tier-2 floor). Both
implementations are in some recovery / DTX path producing unconstrained
PCM.

Captured at `family=0, packet_len=10` and `family=0, packet_len=8`
across the 4 repros — all single-stream cases where the multistream
wrapper effectively forwards to the standalone decoder, so this is
*the same bug surface* re-finding the same recovery-divergence class.

## Resolution applied (campaign hour 1)

The SNR floor was disabled in `fuzz_multistream.rs` for SILK/Hybrid
decode (CELT-only remains byte-exact). Sample-count parity is the
only oracle on the SILK/Hybrid sub-packet path now. This restores the
pre-Stream-D safety baseline for multistream and unblocks the campaign.

The 4 captured repros are kept in this directory as a regression marker
for any future re-enablement of the SNR floor — once the
recovery-divergence root cause is fixed, the oracle can be re-armed
and these inputs should pass.

## Severity

Low (oracle calibration, not a codec bug per se). The underlying
recovery-divergence is real but produces only "low-fidelity recovery
PCM diverges between implementations" — not a functional bug under
well-formed input.

## Detected

2026-05-02 ~23:36 → 00:30 BST, 24h fuzz campaign hour 1. All 4
`fuzz_multistream` workers tripped within minutes of campaign start.

## Fix scope

Two complementary follow-ups:
1. Fix the underlying SILK recovery-divergence so malformed-but-
   parseable packets produce equivalent PCM on both sides.
2. Or build a structural per-sub-frame validity gate for multistream
   payloads (parse out the embedded TOC code and frame sizes) so the
   SNR floor only fires on inputs whose all sub-frames have non-zero
   bytes.

Either gives back the tier-2 SNR coverage we lost in the campaign-hour-1
mitigation.

## 2026-05-03 Worker B triage

Verdict: **defer**, gated on the standalone
`tests/fuzz/known_failures/silk_decode_recovery_divergence_loud/` fix or a
future per-sub-packet structural-validity gate.

Focused replay now locks the classification in
`tests/fuzz/tests/multistream_decode_asymmetries.rs`: all four multistream
fixtures decode `Ok` on both Rust and C, sample counts match, the packets are
SILK/Hybrid rather than CELT-only, and only recovery PCM diverges below the
50 dB SNR floor. These repros stay in this directory as markers until the
gating SILK recovery class is fixed and the multistream SILK/Hybrid SNR oracle
can be re-armed.

## 2026-05-04 resolution

The SILK/Hybrid SNR oracle has been re-armed for attacker-controlled
multistream decode when every parsed sub-packet is coded-comparable. The
classifier walks each multistream sub-packet with the same self-delimited
framing convention used by the decoder. If any SILK/Hybrid sub-frame has size
`<= 1`, the packet is treated as recovery/DTX and only sample-count parity is
asserted.

All four fixtures in this directory classify as recovery/DTX because they are
family-0 mono decode inputs whose payloads contain a code-2 final sub-frame of
0 bytes. They remain tracked as regression markers for the classifier.
