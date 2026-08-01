# ROP-BUG-FLUX-00044 — Simulated packet loss uses the wrong PLC duration

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-tools-core/decode-plc
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T214001Z-p1439-n246664000-c1 branch=task/bug-ROP-BUG-FLUX-00044-run-fix-20260801T214001Z-p1439-n246664000-c1 code=a77085d3c65041495f2e6cb303114575a77b4fb6 gate=manual) -> Closed (2026-08-02, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main 3528b9e; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main ac7ff8a. /Users/md/language/ropus/ropus-tools-core/src/commands/decode.rs:240-293 initializes PLC at 960 samples and sizes each dropped packet from the previous successful decode. The current packet TOC is available but ignored, so first-packet loss and duration transitions alter the output timeline for non-20 ms streams. /Users/md/language/ropus/ropus-tools-core/tests/round_trip.rs:669-701 covers only 20 ms and asserts a broad lower length bound. Fix: derive each deliberately dropped packet duration from its own TOC, validate it, and assert exact no-loss/loss timeline equality for 10 ms, 60 ms, and duration-switch streams. Static review only; no decoder or test ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-02, independent two-eyes, host flux)

- Fix commit `a77085d` derives each deliberately-dropped packet's duration from its own TOC
  in `ropus-tools-core/src/commands/decode.rs` instead of reusing the previous packet's size.
- Regression re-verified by construction: spliced the fix's new tests
  (`decode_packet_loss_preserves_10ms_and_60ms_timelines`,
  `decode_packet_loss_preserves_duration_switch_timeline`) onto the pre-fix `decode.rs` in a
  scratch worktree at `a77085d3~1` — `decode_packet_loss_preserves_10ms_and_60ms_timelines`
  panicked (`decoded 16320 samples, but EOS granule requires 48312 samples`), confirming the
  root cause. Both tests pass at the current tree.
- `cargo clippy -p ropus-tools-core --all-targets --locked -- -D warnings` clean; `cargo test
  -p ropus-tools-core --locked`: 129 lib + 22 integration passed, 0 failed.
