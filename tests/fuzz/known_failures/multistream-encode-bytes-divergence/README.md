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

Representative case:
- `sr=8000, ch=1, family=0, bitrate=15472, complexity=6, vbr=0,
  packet_len=15`
- Rust and C produce 15-byte compressed packets (length matches), but
  the bytes differ.

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

2026-05-02 ~02:30 BST, 24h fuzz campaign hour 3. Killed 2 of the 4
restarted multistream workers (w0, w3). 2 unique fingerprints
captured.

## Fix scope

Separate PR. Investigation:
1. Run the minimised crash through both encoders with verbose tracing.
2. Diff `silk_Encode_Frame` state at the panic point.
3. Likely candidate: a setter-driven re-derivation that differs by one
   field re-computation between Rust and C.
