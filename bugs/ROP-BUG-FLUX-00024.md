# ROP-BUG-FLUX-00024 — Fuzz replay drops one-sided decoder status divergences

- **State:** Closed
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T193543Z-p55189-n217293000-c1 branch=task/bug-ROP-BUG-FLUX-00024-run-fix-20260801T193543Z-p55189-n217293000-c1 code=b834cea gate=manual) -> Closed (2026-08-01, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main dc05a88; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at `origin/main` `d0ab87e`. `/Users/md/language/ropus/harness/src/bin_inner/replay_fuzz_decode.rs:235` through `:240` returns `None` for every decode result except `(Ok, Ok)`, so both Rust-success/C-error and Rust-error/C-success seeds disappear from the replay findings. The actual oracle at `/Users/md/language/ropus/tests/fuzz/fuzz_targets/fuzz_decode.rs:245` through `:258` panics on exactly those asymmetric outcomes. Expected: replay reproduces every failure class enforced by the fuzz target. Actual: a corpus seed can fail the target but produce no replay finding. Fix: model decoder status in `Finding`, ignore only `(Err, Err)` as the target does, report both one-sided cases, and add a pure four-quadrant outcome-classifier test. Static review only; no fuzz target or replay binary ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-01, independent two-eyes, host flux)

- Fix commit `b834cea` makes `harness/src/bin_inner/replay_fuzz_decode.rs`'s `Finding` model
  decoder status and report both one-sided `(Ok, Err)`/`(Err, Ok)` divergences, ignoring only
  `(Err, Err)` — matching the oracle in `tests/fuzz/fuzz_targets/fuzz_decode.rs`.
- New test `decoder_status_classifier_covers_all_four_quadrants` passes.
- `cargo clippy -p ropus-harness --all-targets --locked -- -D warnings` clean; `cargo test -p
  ropus-harness --locked`: every suite green, 0 failed.
