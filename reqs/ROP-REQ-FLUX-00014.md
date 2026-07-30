# ROP-REQ-FLUX-00014 — Complete DRED payload reconstruction to PCM

- **State:** Draft
- **Priority:** Should
- **Area:** ropus/opus-dred
- **Raised:** 2026-07-31
- **Implemented-by:** —
- **Satisfied-by:** —
- **Violated-by:** —
- **Flow:** heavy
- **Claimed-by:** —
- **State history:** Draft (2026-07-31, raised via `deltic reqs new` model=gpt-5.6-sol@high)

## Statement

The system must reconstruct parsed DRED features into i16, 24-bit-in-i32, and f32 PCM through OpusDecoder using the existing FARGAN synthesis/model path, instead of returning OPUS_UNIMPLEMENTED from /Users/md/language/ropus/ropus/src/opus/dred.rs:228-280. The implementation must preserve the reference offset, frame-size, model-loading, and sample-format semantics and provide parity-focused coverage.

## Notes
