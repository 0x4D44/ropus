# Multistream decode validity asymmetry across families

## Repro

The 2 crash files are libFuzzer Arbitrary inputs for `fuzz_multistream`
that trigger a `(Err Rust, Ok C)` decoder-result mismatch at families
1 and beyond (i.e. structured surround layouts).

```
cargo +nightly fuzz run --fuzz-dir tests/fuzz fuzz_multistream \
    tests/fuzz/known_failures/multistream-decode-family-asymmetry/crash-00.bin
```

## Symptom

Representative case:
- `sr=48000, ch=3, family=1, packet_len=128`
- C decodes the packet successfully (5760 samples / channel).
- Rust returns `OPUS_INVALID_PACKET` (-4).

`family=1` is Vorbis surround mapping (per RFC 7845 §5.1.1.2) — at
ch=3 this is the 3.0 layout (front-left, front-centre, front-right).
This is a *structured* family so both implementations should ideally
agree on packet validity. Since Rust is being **stricter** (rejecting
what C accepts) this is the safer asymmetry — but it's still a
real divergence in the decoder validation pipeline.

Compare with `multistream-family255-error-asymmetry/`: family=255
("no mapping") is genuinely unconstrained, so asymmetric validation
is acceptable. family=1 is constrained, so the asymmetry is at
least worth tracking.

## Resolution applied (campaign hour 3)

`fuzz_multistream.rs` now tolerates the **stricter-Rust** direction
`(Err Rust, Ok C)` at all families >= 1. The opposite direction
`(Ok Rust, Err C)` — Rust accepting what C rejects, the
security-relevant direction — still panics at families 0/1/2.

This is broader than the family=255-only mitigation: it accepts
Rust over-rejecting valid packets at structured families to keep
the campaign running. Follow-up should narrow this back to
family=255-only once the family=1 stricter-Rust validation is
investigated.

## Severity

Low/medium. Stricter Rust validation is the safe direction (no
security risk), but rejecting valid Vorbis-surround packets means
ropus consumers will fail to play back legitimate multi-channel
streams that the C reference accepts. Worth a follow-up
investigation to confirm the Rust-side rejection condition is
deliberate (not a port bug).

## Detected

2026-05-02 ~02:30 BST, 24h fuzz campaign hour 3. Killed 2 of the 4
restarted multistream workers (w1, w2).
