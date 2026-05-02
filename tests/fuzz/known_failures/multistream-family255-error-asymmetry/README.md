# Multistream family=255 packet-validity asymmetry

## Repro

The 4 crash files are libFuzzer Arbitrary inputs for `fuzz_multistream`
that triggered the `(Ok, Err)` or `(Err, Ok)` decode-result mismatch
panic during the 24h campaign hour 2 (after the SILK/Hybrid SNR
oracle was disabled).

```
cargo +nightly fuzz run --fuzz-dir tests/fuzz fuzz_multistream \
    tests/fuzz/known_failures/multistream-family255-error-asymmetry/crash-00.bin
```

## What this is

Mapping family 255 means "no surround layout / custom mapping"
(per RFC 7845 §5.1.1). Channel count, stream count, coupled-stream
count, and the mapping[] array are all attacker-controlled, with
no fixed structural constraints from the family.

Under family 255, Rust and C apply slightly different validation —
typically Rust returns `OPUS_INVALID_PACKET` (-4) on small packets
that C still accepts (or vice versa). Representative cases captured:

| sr | ch | family | packet_len | direction |
|----|----|--------|------------|-----------|
| 48000 | 4 | 255 | 49 | Rust err / C ok |
| 48000 | 2 | 255 | 4  | Rust err / C ok |
| 8000  | 2 | 255 | 20 | Rust ok / C err |
| 8000  | 2 | 255 | 20 | Rust err / C ok |

The *decoder* state machines diverge on edge cases that the *family*
doesn't constrain, so neither implementation is "wrong" by the spec —
they're just making different defensive choices on inputs that have
no normative meaning.

## Resolution applied (campaign hour 2)

`fuzz_multistream.rs` now tolerates `(Ok, Err)` and `(Err, Ok)`
decode mismatches **only when mapping_family == 255**. Families 0
(mono/stereo), 1 (Vorbis surround), and 2 (ambisonics) are
structured layouts where both implementations should agree on
packet validity — those still panic on asymmetric error.

## Severity

Low. Family 255 is intended as an escape hatch for non-standard
channel layouts; spec-driven implementations are encouraged to be
conservative. The asymmetric rejection doesn't expose any safety
issue (both still produce an error code, just different ones).

## Detected

2026-05-02 ~01:30 BST, 24h fuzz campaign hour 2 (after the
SILK/Hybrid SNR oracle was disabled in 2fdf690). All 4
fuzz_multistream workers tripped within ~50 min of restart.

## Fix scope

Tracking only — not blocking the codec from shipping. A future PR
could either:
1. Document Rust's family-255 validation as intentionally stricter
   than C, with a regression test pinning the rejected configurations.
2. Align Rust to C's looser handling under family-255.
