# Differential finding: VBR + FEC + DTX cross-frame divergence

## Repro

The 6 crash files in this directory are libFuzzer-format inputs for
`fuzz_encode_multiframe` that trigger a per-frame byte-mismatch between
`OpusEncoder::encode` (Rust) and `opus_encode` (C reference) when the
encoder has VBR + inband-FEC + DTX toggled across frames.

To replay one:
```
cargo +nightly fuzz run --fuzz-dir tests/fuzz fuzz_encode_multiframe \
    tests/fuzz/known_failures/multiframe-vbr-fec-dtx-divergence/crash-XXXX
```

## Symptom

For a 2-frame stream at sr=8000, ch=2, VOIP application, bitrate≈59713,
complexity=0, with `vbr=1, fec=2, dtx=1, loss_perc=7` (representative
case): **frame 1** of Rust output is 164 bytes; frame 1 of C output is
181 bytes. Compressed bytes diverge.

The 6 captured repros are different SHA-1 inputs — they exercise
distinct field combinations within the same general state-interaction
class. None of them appear in single-frame fuzzing (Stream A's targets
all pass), so the bug is multi-frame-only.

## Likely root cause

Rust's encoder rate-control state caches differently from C's when
`set_inband_fec` and `set_dtx` are toggled between frames in VBR mode.
Suspect path: `silk_mode.use_in_band_fec` / `silk_mode.use_dtx` are
written by the setter, but a derived rate-control field
(`rangeFinal`, `bitsFinal`, or `silk_mode.bitRate` recomputation) is
re-derived on a different schedule between Rust and C. Needs a
side-by-side trace.

## Workaround currently in place

None at the codec level. Stream B's `fuzz_encode_multiframe`
deliberately exercises this path; the repros remain in this directory
so they aren't replayed as regression seeds in `tests/fuzz/artifacts/`.

## Severity

Medium. Per-frame VBR + FEC + DTX toggling is realistic for adaptive-
bitrate WebRTC and push-to-talk usage. Single-frame and CBR paths
remain bit-exact.

## Fix scope

Separate PR. Investigation steps:
1. Diff `opus_encode_native` between Rust and C around the rate-control
   recomputation block when `silk_mode.use_in_band_fec` flips.
2. Trace one of the smaller repros through both encoders.
3. Either align the recomputation order (preferred) or relax the
   differential oracle to skip per-frame toggling combinations
   (last resort — would lose real coverage).

## Detected

2026-05-01 Stream B implementation of
`wrk_docs/2026.05.01 - HLD - fuzz-coverage-expansion V2.md`. Smoke run
of rebuilt `fuzz_encode_multiframe` triggered 4 crashes in 30 seconds;
2 additional crashes triggered during the post-feedback verification
smoke run.
