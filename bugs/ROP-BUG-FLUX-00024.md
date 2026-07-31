# ROP-BUG-FLUX-00024 — Fuzz replay drops one-sided decoder status divergences

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness/fuzz-replay
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at `origin/main` `d0ab87e`. `/Users/md/language/ropus/harness/src/bin_inner/replay_fuzz_decode.rs:235` through `:240` returns `None` for every decode result except `(Ok, Ok)`, so both Rust-success/C-error and Rust-error/C-success seeds disappear from the replay findings. The actual oracle at `/Users/md/language/ropus/tests/fuzz/fuzz_targets/fuzz_decode.rs:245` through `:258` panics on exactly those asymmetric outcomes. Expected: replay reproduces every failure class enforced by the fuzz target. Actual: a corpus seed can fail the target but produce no replay finding. Fix: model decoder status in `Finding`, ignore only `(Err, Err)` as the target does, report both one-sided cases, and add a pure four-quadrant outcome-classifier test. Static review only; no fuzz target or replay binary ran.

## Fix

<unfixed — raised only>

## Notes
