# Differential finding: NaN/Inf in encode_float

## Repro
The crash file in this directory is a libFuzzer-format input that triggers a
differential mismatch between `OpusEncoder::encode_float` (Rust) and
`opus_encode_float` (C reference) when fed an f32 PCM buffer containing
NaN or Inf.

## Symptom
Rust's `float2int16` (`ropus/src/types.rs::float2int16`) returns `0` for
NaN inputs. The C reference returns `-32768`. Subsequent encode bytes
diverge.

## Workaround currently in place
The float-PCM fuzz targets (`fuzz_encode`, `fuzz_encode_safety`,
`fuzz_roundtrip`, `fuzz_roundtrip_safety`) replace NaN/Inf with `0.0`
immediately after `f32::from_le_bytes`, so the codec divergence is not
the focus of those campaigns. Sub-normals pass through unchanged and
produce identical output on both sides.

## Fix scope
Probably one line in `ropus/src/types.rs::float2int16` — match C's
`_mm_cvttps_epi32`-style NaN handling (clamp to `-32768`).

## Detected
2026-04-30 overnight fuzz preparatory smoke run (Stream A implementation
of `wrk_docs/2026.05.01 - HLD - fuzz-coverage-expansion V2.md`).
