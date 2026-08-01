# ROP-BUG-FLUX-00007 — Safe public DSP helpers can trigger unchecked-indexing UB

- **State:** Closed
- **Priority:** Must
- **Severity:** High
- **Area:** ropus/silk-celt-api
- **Raised:** 2026-07-31
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T035908Z-p73731-n642592000-c1 branch=task/bug-ROP-BUG-FLUX-00007-run-fix-20260801T035908Z-p73731-n642592000-c1 code=860662501a8bc9565c56edaf6f013a6ec98d5bf3 gate=manual) -> Closed (2026-08-01, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main dc05a88; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static source inspection confirmed that the safe public silk_lpc_analysis_filter at /Users/md/language/ropus/ropus/src/silk/common.rs:433 passes caller-controlled indices into the uc macro at /Users/md/language/ropus/ropus/src/lib.rs:227, which uses get_unchecked. For example, output length 2, signal length 1, coefficient length 1, len 2, and order 1 reaches an unchecked read of signal index 1. Expected: every safe public API validates slice and dimension invariants, or the API is explicitly unsafe with a complete safety contract. Actual: an ordinary safe call can cause undefined behavior. Static review only; no code was executed.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-01, independent two-eyes, host flux)

- Commit `8606625` changes `silk_lpc_analysis_filter` (`ropus/src/silk/common.rs:433`) from
  `pub` to `pub(crate)`, removing it from the safe public surface entirely (the described root
  cause: no external caller can reach the unchecked-indexing helper any more).
- A new `compile_fail` doctest at `ropus/src/lib.rs:48` proves external code can no longer name
  the function.
- `cargo clippy -p ropus --all-targets --locked -- -D warnings` clean; `cargo test -p ropus
  --locked`: 1888 passed, 0 failed, including the compile-fail doctest ("all doctests ran").
