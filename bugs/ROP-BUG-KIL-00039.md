# ROP-BUG-KIL-00039 — fb2k decode FFI constructs slices from an unchecked caller length

- **State:** Closed
- **Priority:** Should
- **Severity:** High
- **Area:** ropus-fb2k/ffi
- **Raised:** 2026-08-22T06:10:46Z
- **Discovery source:** Agent
- **Owner:** -
- **Owner role:** -
- **Owner run:** -
- **Owner host:** -
- **Owner branch:** -
- **Owner base:** -
- **Owner fingerprint:** -
- **Owner since:** -
- **Owner until:** -
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T06:10:46Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-13T05:47:02Z, deltic:auto role=fix run=fix-20260913T053344Z-b3540b17 branch=task/bug-ROP-BUG-KIL-00039-run-fix-20260913T053344Z-b3540b17 code=52e564d49df573c6848bbd7fefb3cbd1e8e3a6dd gate=manual) -> Closed (2026-09-13T16:43:36Z, independent two-eyes verification model=codex@xhigh, verifier=CRUCIBLE, fixer=deltic:auto, fix=52e564d49df573c6848bbd7fefb3cbd1e8e3a6dd)

## Observation

Static review at HEAD 3e0f6c1. ropus-fb2k/src/lib.rs:419-434 checks only the minimum output length, then computes max_samples_per_ch * channels unchecked and passes it to from_raw_parts_mut. A hostile or invalid large size_t can wrap the multiplication or exceed the Rust slice isize::MAX byte limit, invoking undefined behavior rather than returning BAD_ARG; the C header declares no upper bound. Expected: use checked arithmetic, enforce the slice byte bound, and preferably construct only the fixed maximum span the decoder can write. Add SIZE_MAX and boundary regression coverage. Static review only; no app, build, test, or harness ran.

## Fix

### Verification summary (2026-09-13)

- Re-ran `checked_interleaved_len_rejects_overflow_and_byte_limit`; it passed, and the ropus-fb2k package gate passed all 111 tests.
- A red control removed the `f32` element-size bound; the overflow-and-byte-limit assertion failed, and the fix was restored.

## Notes
