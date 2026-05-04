# Resolved differential finding: NaN in encode_float

Status: fixed on 2026-05-04 by matching the C `FLOAT2INT16`
`MAX32`/`MIN32` clamp order in `ropus/src/types.rs::float2int16`.
The directory is retained as a triage record because concurrent fuzz
work may still reference known-failure paths.

## Repro
The crash file in this directory is a libFuzzer-format input that triggers a
differential mismatch between `OpusEncoder::encode_float` (Rust) and
`opus_encode_float` (C reference) when fed an f32 PCM buffer containing
NaN.

Post-fix replay commands:

```bash
timeout 180s cargo +nightly fuzz run --fuzz-dir tests/fuzz fuzz_encode \
  tests/fuzz/known_failures/nan-encode-float/crash-54acd282f7bd446d92811b719f183d6eea294f84

timeout 180s cargo +nightly fuzz run --fuzz-dir tests/fuzz fuzz_roundtrip \
  tests/fuzz/known_failures/nan-encode-float/crash-54acd282f7bd446d92811b719f183d6eea294f84
```

Both commands passed on 2026-05-04 after removing the differential parser
masking in `fuzz_encode` and `fuzz_roundtrip` (`fuzz_encode` fixed-input
replay executed in `5 ms`; `fuzz_roundtrip` fixed-input replay executed in
`69 ms`).

## Symptom
Rust's `float2int16` (`ropus/src/types.rs::float2int16`) returns `0` for
NaN inputs. The C reference returns `-32768`. Subsequent encode bytes
diverge.

## Regression marker
The minimized fixture has also been promoted to:

- `tests/fuzz/corpus/fuzz_encode/regression-nan-encode-float`
- `tests/fuzz/corpus/fuzz_roundtrip/regression-nan-encode-float`

The safety fuzz targets still keep their broader non-finite-to-zero
workaround for now because they can drive high-complexity analysis with
the original float buffer. That is separate from this fixed
`FLOAT2INT16` parity issue.

## Detected
2026-04-30 overnight fuzz preparatory smoke run (Stream A implementation
of `wrk_docs/2026.05.01 - HLD - fuzz-coverage-expansion V2.md`).
