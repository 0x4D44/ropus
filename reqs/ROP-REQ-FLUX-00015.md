# ROP-REQ-FLUX-00015 — Remove recurring heap allocation from codec real-time hot paths

- **State:** Draft
- **Priority:** Should
- **Area:** ropus/dnn-celt-hot-paths
- **Raised:** 2026-07-31
- **Implemented-by:** —
- **Satisfied-by:** —
- **Violated-by:** —
- **Flow:** heavy
- **Claimed-by:** —
- **State history:** Draft (2026-07-31, raised via `deltic reqs new` model=gpt-5.6-sol@high)

## Statement

The system must reuse bounded scratch storage across frames in neural PLC/FARGAN and routine CELT/Opus conversion hot paths so steady-state encode, decode, and concealment do not allocate per layer or subframe. Static review counted at least 69 Vec allocations per 10 ms FARGAN synthesis frame across /Users/md/language/ropus/ropus/src/dnn/core.rs:630-765 and /Users/md/language/ropus/ropus/src/dnn/fargan.rs:659-962; runtime impact remains to be measured. Preserve fixed-point operation order and bit-exact output.

## Notes
