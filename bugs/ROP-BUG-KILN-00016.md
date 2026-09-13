# ROP-BUG-KILN-00016 — Classical control uses stale packet-loss recovery horizon

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness-control/loss-pattern
- **Raised:** 2026-08-16T07:49:45Z
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
- **State history:** Open (2026-08-16T07:49:45Z, raised via `deltic bugs new`) -> Fixed (2026-09-12T23:42:37Z, deltic:auto role=fix run=fix-20260912T233905Z-2a8132f9 branch=task/bug-ROP-BUG-KILN-00016-run-fix-20260912T233905Z-2a8132f9 code=ce30efbd0b93799d3839f064b631638af21714da gate=manual) -> Closed (2026-09-13T14:33:07Z, independent two-eyes verification on host CRUCIBLE, model=codex@xhigh, at origin/main 19c6ac33c0e5148c5443ea062567b0154ff05fc0; fixer was a different actor)

## Observation

Static review at origin/main a97b6f11. harness-control/tests/control_snr.rs:65-68 claims to mirror the tier-2 loss pattern but drops every positive multiple of seven through frame 98, and lines 303-306 expect 14 losses. The live tier-2 contract at harness-deep-plc/tests/tier2_snr.rs:42-66 requires a complete seven-frame recovery horizon and drops only frames 7 through 91, 13 losses. Expected: the classical control measures the documented tier-2 packet and recovery conditions. Actual: its final loss ends mid-cycle, so its aggregate SNR is not directly comparable to the live scenario. Fix: share the interval and complete-cycle predicate or copy the horizon guard exactly, then assert the exact expected loss indexes. Static inspection and git history only; no app, build, test, decoder, or harness ran.

## Fix

Implemented in `harness-control/tests/control_snr.rs` and integrated at code
commit `ce30efbd0b93799d3839f064b631638af21714da`.

- The control loss predicate now shares the seven-frame interval and requires
  a complete recovery horizon before dropping a packet.
- Added an exact loss-index oracle for frames 7 through 91 (13 losses), and
  updated the control assertion to use that expected set.

Verification:

- `$null | deltic timeout 180 cargo test -p ropus-harness-control --test control_snr loss_pattern_contains_only_complete_recovery_cycles` — 1 passed, 0 failed.
- `deltic timeout 120 cargo check -p ropus-harness-control` — passed.
- `deltic timeout 120 cargo fmt --all -- --check` — passed.
- `git diff --check` — passed.
- Red proof: temporarily removed the recovery-horizon guard; the exact-index
  test failed because frame 98 was included. The guard was restored and the
  focused test passed.

## Notes
