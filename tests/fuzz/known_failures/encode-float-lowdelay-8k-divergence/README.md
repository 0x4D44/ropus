# Differential finding: float-PCM single-frame encode divergence at sr=8k LOW_DELAY

## Repro

The crash file is a libFuzzer raw-bytes input for `fuzz_encode` that
triggers a compressed-output byte-mismatch between Rust and C on
**single-frame** float-PCM encode at sr=8000.

```
cargo +nightly fuzz run --fuzz-dir tests/fuzz fuzz_encode \
    tests/fuzz/known_failures/encode-float-lowdelay-8k-divergence/crash-00.bin
```

## Symptom

Variants captured (sr=8000, ch=1, RESTRICTED_LOWDELAY (2051),
bitrate=65760, complexity=3, float-PCM input):

| Repro | vbr | fec | Outcome |
|---|---|---|---|
| crash-00 | 0 | 0 | 164B = 164B (bytes diverge) |
| crash-01-vbr-len-mismatch | 1 | 0 | Rust=308B vs C=123B (length divergence) |

The CBR case produces the same-length-different-bytes pattern; the
VBR case produces an outright length divergence — Rust packs more
than 2× the bytes C does for the same input. Cross-mode variation
matches the multistream-encode-bytes-divergence shape (where the
4 variants span CBR/VBR/cx 4/6/9 too).

## Why this is significant

This is a **single-frame** divergence — distinct from the cross-frame
state-accumulation classes (`multiframe-cbr-cross-frame-divergence`,
`multistream-encode-bytes-divergence`). Single-frame fuzzing has
been bit-exact for the i16 PCM path since Stream A landed; this
finding is in the **float-PCM** ingest path that Stream A added
(HLD V2 gap 11) and that the campaign's float-PCM coverage was
designed to exercise.

Compare with `roundtrip-float-12k-cx7-vbr-divergence/` — different
config (sr=12000 ch=2 AUDIO cx=7 VBR vs sr=8000 ch=1 LOWDELAY cx=3
CBR) but plausibly the same root cause: float→i16 conversion or
float-path codebook lookup divergence between Rust and C.

The fuzz_encode worker w0 ran for 6 hours with cov=7829 before
hitting this — meaning libFuzzer's mutator needed sustained
exploration to land on the specific input shape. Worth investigating
both this and the roundtrip-float case together since they probably
share a root cause.

## Severity

Medium. RESTRICTED_LOWDELAY application is used by real-time voice
deployments (WebRTC, VoIP) where the float PCM ingest is normal.
Byte-level divergence under these settings means low-latency voice
streams will silently produce a different bitstream from the C
reference — likely still decodable, but differential-test failures
suggest the underlying DSP arithmetic path differs in ways that
could surface as quality issues.

## Detected

2026-05-02 hours 6 + 12 of 24h fuzz campaign. crash-00 killed
fuzz_encode w0 at hour 6 (CBR variant). crash-01-vbr-len-mismatch
killed fuzz_encode w1 at hour 12 (VBR variant). Both fuzz_encode
workers retired for the rest of the campaign — same encode-path
class will keep being hit on restart.

## Fix scope

Separate PR. Investigate jointly with
`roundtrip-float-12k-cx7-vbr-divergence/` — both involve float-PCM
input through `OpusEncoder::encode_float`. Likely candidates:
- `ropus/src/types.rs::float2int16` rounding/clamping difference
- Float-domain analysis weights (analysis stage 6 still in flight)
- Float-PCM resampling path inside the encoder
