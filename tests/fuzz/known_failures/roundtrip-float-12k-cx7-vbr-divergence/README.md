# Differential finding: float-PCM roundtrip divergence at sr=12k, cx=7, VBR

## Repro

The 2 crash files are libFuzzer-format inputs for `fuzz_roundtrip`
that trigger a compressed-output byte-mismatch between Rust and C
when encoding **float PCM** at **sr=12000, application=AUDIO,
complexity=7, vbr=true**.

```
cargo +nightly fuzz run --fuzz-dir tests/fuzz fuzz_roundtrip \
    tests/fuzz/known_failures/roundtrip-float-12k-cx7-vbr-divergence/crash-00.bin
```

## Symptom

Representative case:
- `sr=12000, ch=2, application=2049 (AUDIO), bitrate=71281, cx=7,
  vbr=true`
- Rust encode produces a **289-byte** packet.
- C encode produces a **179-byte** packet.

The first ~13 bytes match (TOC + early VBR size fields), then the
streams diverge. Different *length* outputs means this is a
rate-control / frame-size decision divergence, not just a coefficient
quantization difference.

## Why this isn't the analysis-divergence class

Campaign 9 (2026-04-19) flagged a divergence class at
`complexity ≥ 10 ∧ sr ≥ 16000 ∧ app != RESTRICTED_SILK` due to the
missing analysis port. This finding is at **complexity=7** (below
the threshold) and **sr=12000** (also below the threshold), so it
isn't that class.

The float-PCM input path *is* a Stream A addition (HLD V2 gap 11,
2026-05-01). It's plausible the float-PCM resampling or the
`encode_float` → internal-i16 conversion pipeline has a bitrate-side
divergence that wasn't there for the i16 path.

## Severity

Medium. AUDIO application + VBR + complexity 7 is a normal
high-quality voice-and-music encode setting; sr=12000 is unusual but
legal (Opus auto-resamples internally). Stereo float-PCM with these
settings is probably a real-world configuration in some deployments.

## Detected

2026-05-02 ~23:56 BST 2026-05-01 (T+20 min into 24h campaign). Killed
both `fuzz_roundtrip` workers. 2 unique fingerprints — almost
certainly the same root-cause class.

## Fix scope

Separate PR. Investigation: trace `opus_encode_native` from float-PCM
ingest through `silk_Encode_Frame` rate-control on both Rust and C at
this exact configuration. Look for divergence in `useDTX`,
`use_in_band_fec`, `complexity`-derived parameters, or the float→i16
conversion pipeline.
