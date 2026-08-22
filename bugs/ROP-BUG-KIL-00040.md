# ROP-BUG-KIL-00040 — fb2k tag callbacks can invalidate the borrowed reader handle

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** ropus-fb2k/ffi-reentrancy
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
- **State history:** Open (2026-08-22T06:10:46Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Static review at HEAD 3e0f6c1. ropus-fb2k/src/lib.rs:339-364 holds a shared Rust reference and iterates reader-owned strings across arbitrary C callbacks at :342 and :357. The public header does not forbid reentrancy, so a callback can call ropus_fb2k_close or a mutating API such as decode_next or seek; the outer function then resumes through a freed handle or overlapping mutable borrow. Expected: snapshot callback data before the first callback and enforce a clear reentrancy or handle-lifetime policy, with callbacks that close and mutate the handle as regressions. Static review only; no app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
