# Safety finding: OpusMSDecoder sub-packet length OOB panic

## Repro
The crash files in this directory are libFuzzer-format inputs (Arbitrary-
shaped) for `fuzz_multistream` that panic `OpusMSDecoder::decode` on a
short, attacker-controlled payload.

```
cargo +nightly fuzz run --fuzz-dir tests/fuzz fuzz_multistream \
    tests/fuzz/known_failures/multistream-decode-subpacket-oob/crash-f23bd5f24ad728d0f3835d054ee2b9199808d2e9
```

## Symptom
Panic site: `ropus/src/opus/multistream.rs:624`

```
thread '<unnamed>' panicked at ropus/src/opus/multistream.rs:624:69:
range end index 1278 out of range for slice of length 20
```

The decoder iterates streams, calls `parse_multistream_subpacket` to
discover each sub-packet's byte length, and then slices

```rust
&sub_data[..packet_offset as usize]
```

to feed `opus_packet_get_nb_samples`. `parse_multistream_subpacket` does
not bound `packet_offset` against the actual remaining sub-packet length,
so an Opus self-delimited size prefix > sub-packet bytes turns into a
slice OOB panic in safe Rust.

## Severity
**P0 safety vulnerability — denial-of-service.** Any caller of
`OpusMSDecoder::decode` on untrusted bytes can be crashed with a
20-byte attacker-controlled multistream packet. The Rust slice bounds
check converts what would be a C-side `OPUS_INVALID_PACKET` into a
process-killing panic.

## Threat model
Untrusted-input decode is a real and common deployment shape for an
Opus codec: RTP/WebRTC media servers, VoIP clients, Ogg/Opus file
parsers, browser audio pipelines, and any service that ingests
third-party media all expose `OpusMSDecoder::decode` to attacker-
controlled bytes. A single malformed packet panics the process —
trivial DoS against any consumer of `ropus` on this code path.

The C reference returns `OPUS_INVALID_PACKET` on the same input
(graceful handling), so this is a **Rust-port-only regression**:
upstream `xiph/opus` is unaffected.

## Suggested fix
The fix is straightforward: bounds-check `packet_offset` against
`sub_data.len()` before slicing in `ropus/src/opus/multistream.rs:624`
and the surrounding subpacket-iteration loop. Alternatively, have
`parse_multistream_subpacket` return `OPUS_INVALID_PACKET` whenever
the encoded self-delimited length exceeds the remaining buffer — that
mirrors what the C reference does and keeps the error path in one
place.

## Detected
2026-04-30 Stream C smoke run of `fuzz_multistream`
(`wrk_docs/2026.05.01 - HLD - fuzz-coverage-expansion V2.md`).
First reproducer landed at iteration ~2.7M into a 30 s campaign.
