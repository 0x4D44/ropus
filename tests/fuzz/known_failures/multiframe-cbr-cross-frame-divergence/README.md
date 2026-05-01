# Differential finding: multiframe CBR cross-frame divergence

## Repro

The 4 crash files in this directory are libFuzzer-format Arbitrary
inputs for `fuzz_encode_multiframe` that trigger a per-frame
byte-mismatch between `OpusEncoder::encode` (Rust) and `opus_encode`
(C reference) under **CBR** with **all toggling flags off**:
`vbr=0, fec=0, dtx=0, loss_perc=0`.

```
cargo +nightly fuzz run --fuzz-dir tests/fuzz fuzz_encode_multiframe \
    tests/fuzz/known_failures/multiframe-cbr-cross-frame-divergence/crash-00.bin
```

## Symptom

Representative case (frame 1/2 mismatch):
- `sr=8000, ch=2, application=2048 (VOIP), bitrate=50544`
- `complexity=0, vbr=0, fec=0, dtx=0, loss_perc=0`
- Frame 0 of Rust matches C byte-for-byte.
- Frame 1 of Rust diverges from C starting around byte 13.

This is **distinct** from the earlier
`multiframe-vbr-fec-dtx-divergence` finding (which required
VBR+FEC+DTX toggling per frame). This new class exhibits divergence
under pure CBR with no setter changes — meaning the divergence is
purely state-accumulation in the encoder pipeline across frames,
independent of any setter-driven re-derivation.

## Likely root cause

Rust's encoder carries some per-frame state that is updated from frame
0's encode and used in frame 1's encode, in a way that doesn't match
the C reference's update schedule. Candidates:

- SILK decoder state used in inband-FEC lookahead (even with FEC off).
- Pitch lag / LPC coefficient memory.
- VBR rate-control "carry" state that is computed even when VBR is off.
- Stereo prediction memory at sr=8000 ch=2.

A side-by-side trace of `silk_Encode` (or `opus_encode_native`) on the
two implementations between frame 0 and frame 1 should pinpoint the
diverging field.

## Severity

Medium. CBR is a common deployment shape; cross-frame state
accumulation drift means long streams will diverge more and more from
the C reference even though the per-frame setter API is unchanged.
Single-frame fuzzing (Stream A targets) remains bit-exact, so this
only affects callers that re-use a single `OpusEncoder` across frames
(which is the standard pattern).

## Detected

2026-05-02 ~00:00 BST, 24h fuzz campaign hour 1. Killed all 4
`fuzz_encode_multiframe` workers within ~30 min. 4 unique
fingerprints captured — likely all the same root-cause class but at
different bitrates / sample rates.

## Fix scope

Separate PR. Out of scope for the 24h campaign — needs a side-by-side
trace, not blind fixing.
