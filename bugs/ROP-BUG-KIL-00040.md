# ROP-BUG-KIL-00040 — fb2k tag callbacks can invalidate the borrowed reader handle

- **State:** Fixed
- **Priority:** Should
- **Severity:** High
- **Area:** ropus-fb2k/ffi-reentrancy
- **Raised:** 2026-08-22T06:10:46Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T152510Z-28573cd2
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KIL-00040-run-verify-20260913T152510Z-28573cd2
- **Owner base:** 70a53a97cdb21e28bcf7b6b65d8cc073f23c127b
- **Owner fingerprint:** sha256:00ef2383b8c7508417487990c1236ec80821245314c8906f025fbdce9f7e236b
- **Owner since:** 2026-09-13T15:25:10Z
- **Owner until:** 2026-09-13T17:25:10Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T06:10:46Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-13T06:02:58Z, deltic:auto role=fix run=fix-20260913T054357Z-6ab93113 branch=task/bug-ROP-BUG-KIL-00040-run-fix-20260913T054357Z-6ab93113 code=cedc4bf51bf84fadbfb80134c4020e5bfa3aaa96 gate=manual)

## Observation

Static review at HEAD 3e0f6c1. ropus-fb2k/src/lib.rs:339-364 holds a shared Rust reference and iterates reader-owned strings across arbitrary C callbacks at :342 and :357. The public header does not forbid reentrancy, so a callback can call ropus_fb2k_close or a mutating API such as decode_next or seek; the outer function then resumes through a freed handle or overlapping mutable borrow. Expected: snapshot callback data before the first callback and enforce a clear reentrancy or handle-lifetime policy, with callbacks that close and mutate the handle as regressions. Static review only; no app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
