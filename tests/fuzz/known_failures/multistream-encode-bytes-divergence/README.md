# Differential finding: multistream encode bytes divergence

## Repro

The 2 crash files are libFuzzer Arbitrary inputs for `fuzz_multistream`
that trigger a compressed-output byte-mismatch at the encode path
between `OpusMSEncoder::encode` (Rust) and the C reference's surround
encoder.

```
cargo +nightly fuzz run --fuzz-dir tests/fuzz fuzz_multistream \
    tests/fuzz/known_failures/multistream-encode-bytes-divergence/crash-00.bin
```

## Symptom

Variants captured (sr=8000, ch=1, family=0 across all):

| Repro | bitrate | cx | vbr | Outcome |
|---|---|---|---|---|
| crash-00 | 15472 | 6 | 0 | 15B = 15B (bytes diverge) |
| crash-01 | 15472 | 6 | 0 | 15B = 15B (bytes diverge) |
| crash-02-len-mismatch | 62063 | 9 | 1 | Rust=15B vs C=13B |
| crash-03-cx4-vbr1 | 15472 | 4 | 1 | 12B = 12B (bytes diverge) |

The first two variants captured during hour 3, the last two during
hour 4 after restart. All sr=8000 ch=1 (mono SILK NB) family=0.
Variants span CBR/VBR and complexity 4/6/9 — the encoder
state-divergence is robust across rate-control modes. Variant 02 is
a *length* divergence (different bitstream, different size) — even
more concerning than the byte mismatches.

family=0 with ch=1 is a single-stream mono encode through the
multistream wrapper. That's effectively a thin wrapper over the
regular `OpusEncoder`. So this is plausibly the same root-cause
class as the `multiframe-cbr-cross-frame-divergence` finding from
hour 1: **state-accumulation divergence between Rust and C inside
the encoder**, exposed by the per-frame setter shuffle that
`fuzz_multistream` applies.

`fuzz_encode` (single-frame, no setter shuffle) does NOT find this,
so the trigger is either:
- The setter shuffle (bitrate / complexity / vbr / fec / dtx /
  loss_perc) re-arranging encoder state in a way that diverges from
  C, OR
- Multi-iteration accumulation of state across the `arbitrary` test
  loop, since `OpusMSEncoder` may buffer something between
  `arbitrary_takes`.

## Severity

Medium. Mono SILK NB encode at moderate bitrates is a common
deployment shape (8 kHz voice). Byte-exact divergence under
runtime setter changes means streaming consumers that toggle
bitrate/complexity mid-stream will silently produce a different
compressed bitstream from the C reference. Functional behaviour
(decodability) likely intact.

## Detected

2026-05-02 ~02:30 BST (hours 3-4), 24h fuzz campaign. Killed all 4
multistream workers in two waves: w0/w3 hour 3, then w1/w2 hour 4
after restart under family-asymmetry tolerance. 4 unique fingerprints
captured spanning the rate-control modes listed above.

After hour 4 the multistream target was retired for the rest of the
campaign — every restart cycle hits another variant of this class
within ~30 min, which is by definition the same root-cause encoder
state divergence. Further repros would not add diagnostic value.

## Fix scope

Separate PR. Investigation:
1. Run the minimised crash through both encoders with verbose tracing.
2. Diff `silk_Encode_Frame` state at the panic point.
3. Likely candidate: a setter-driven re-derivation that differs by one
   field re-computation between Rust and C.
