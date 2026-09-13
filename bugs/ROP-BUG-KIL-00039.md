# ROP-BUG-KIL-00039 — fb2k decode FFI constructs slices from an unchecked caller length

- **State:** Fixed
- **Priority:** Should
- **Severity:** High
- **Area:** ropus-fb2k/ffi
- **Raised:** 2026-08-22T06:10:46Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T152436Z-0a0e8667
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KIL-00039-run-verify-20260913T152436Z-0a0e8667
- **Owner base:** a4abc0806c3538586da5b838d3f656eeee8dd67e
- **Owner fingerprint:** sha256:e4e189f9f09276729c2dce502b18cf09c74e4c7ed87d3b17ed0c31bbff20e256
- **Owner since:** 2026-09-13T15:24:36Z
- **Owner until:** 2026-09-13T17:24:36Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T06:10:46Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-13T05:47:02Z, deltic:auto role=fix run=fix-20260913T053344Z-b3540b17 branch=task/bug-ROP-BUG-KIL-00039-run-fix-20260913T053344Z-b3540b17 code=52e564d49df573c6848bbd7fefb3cbd1e8e3a6dd gate=manual)

## Observation

Static review at HEAD 3e0f6c1. ropus-fb2k/src/lib.rs:419-434 checks only the minimum output length, then computes max_samples_per_ch * channels unchecked and passes it to from_raw_parts_mut. A hostile or invalid large size_t can wrap the multiplication or exceed the Rust slice isize::MAX byte limit, invoking undefined behavior rather than returning BAD_ARG; the C header declares no upper bound. Expected: use checked arithmetic, enforce the slice byte bound, and preferably construct only the fixed maximum span the decoder can write. Add SIZE_MAX and boundary regression coverage. Static review only; no app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
