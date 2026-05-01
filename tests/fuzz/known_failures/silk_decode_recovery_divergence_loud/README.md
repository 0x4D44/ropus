# SILK/Hybrid decode recovery-divergence — loud reference output

## Summary

`fuzz_decode` (i16 path) panics on the 7-byte input
`04 00 62 03 bb bb cc` with:

```
thread '<unnamed>' panicked at fuzz_targets/fuzz_decode.rs:237:21:
SILK/Hybrid decode SNR 6.06 dB < 50 dB (sr=48000, ch=1, pkt_len=5, samples=960)
```

## What this is

Header parses as: sr=48000 Hz, mono, i16-PCM mode. Packet body is
`62 03 bb bb cc` (TOC = 0x62 -> Hybrid SWB 10 ms, code 2; payload 4 bytes).

The packet is malformed but parses far enough that both the C reference and
the Rust decoder return `Ok(...)`. Their PCM outputs diverge wildly — same
recovery-divergence class as the deleted `silk_decode_snr_below_floor`
finding, except this packet's reference signal energy (~6.75e11) is well
above the SNR-precheck floor (1e7). Both decoders produce loud full-scale
recovery PCM but with no meaningful relationship to each other.

## Why the energy precheck doesn't filter this

The precheck (`SNR_PRECHECK_MIN_REF_ENERGY = 1e7` in
`tests/fuzz/fuzz_targets/oracle.rs`) was calibrated against the previous
4-byte-packet repro, which had quiet recovery output. This packet has
loud recovery output — energy is not the right axis to distinguish
recovery-divergence from genuine tier-2 drift.

The proper resolution is one of:

1. Add a complementary oracle gate: only assert SNR when both decoders
   agree on the *first* frame (i.e. neither has entered recovery).
   Recovery state is observable on the C side via OPUS_GET_LAST_PACKET_DURATION
   or by re-parsing the packet TOC and comparing to expected frame count.
2. Shrink the SNR oracle to packets that survive an encode->decode
   round-trip — guarantees they are well-formed by construction.
3. Fix the decoder-side error-recovery divergence so malformed-but-
   parseable packets produce equivalent PCM on both sides.

Stream D's energy precheck is sufficient for the original 4-byte case
(deleted) but not for the broader class. Tracking here for follow-up.

## Reproducer

```
cargo +nightly fuzz run --fuzz-dir tests/fuzz fuzz_decode \
    tests/fuzz/known_failures/silk_decode_recovery_divergence_loud/crash.bin
```
