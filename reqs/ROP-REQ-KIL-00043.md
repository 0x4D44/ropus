# ROP-REQ-KIL-00043 — Give ropus-fb2k independent cross-boundary test oracles

- **State:** Draft
- **Priority:** Should
- **Area:** ropus-fb2k/testing
- **Raised:** 2026-08-22T06:11:12Z
- **Discovery source:** Agent
- **Implemented-by:** —
- **Satisfied-by:** —
- **Violated-by:** —
- **Depends-on:** —
- **Design:** —
- **Flow:** heavy
- **Claimed-by:** —
- **State history:** Draft (2026-08-22T06:11:12Z, raised via `deltic reqs new` model=gpt-5.6-sol@high)

## Statement

The ropus-fb2k acceptance suite must prove the Rust-to-C component boundary with independent, position-sensitive oracles. Replace periodic sine and silence seek fixtures in ropus-fb2k/tests/roundtrip.rs:889-902 and :1245-1292 with deterministic non-periodic or packet-index-coded audio; keep the current direct-decoder comparison as a wiring test but add at least one end-to-end comparison against the independent C Opus reference; compile a small C probe against ropus-fb2k/include/ropus_fb2k.h to verify ABI sizes and offsets; and make the normal integration policy execute the test-panic coverage currently gated at tests/roundtrip.rs:1821-1883. Acceptance requires each oracle to fail against a deliberately corrupted position, ABI declaration, or panic path before returning green. This is consolidated test-infrastructure debt from the 2026-08-22 static review; it does not replace the focused regressions required by filed defects.

## Notes
