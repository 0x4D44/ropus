# ROP-BUG-KILN-00003 — Coordinator can mark failed or collided reviews complete

- **State:** Closed
- **Priority:** Should
- **Severity:** High
- **Area:** tools/coordinator-review
- **Raised:** 2026-08-13T17:17:35Z
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
- **State history:** Open (2026-08-13T17:17:35Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-13T04:32:34Z, deltic:auto role=fix run=fix-20260913T042804Z-c2488a79 branch=task/bug-ROP-BUG-KILN-00003-run-fix-20260913T042804Z-c2488a79 code=2fab4b0e9f54cc7b7e396b3c957836c8ac67319f gate=manual) -> Closed (2026-09-13T17:49:57Z, independent two-eyes verification model=codex@xhigh, verifier=CRUCIBLE, fixer=deltic:auto, fix=2fab4b0e9f54cc7b7e396b3c957836c8ac67319)

## Observation

Observation: tools/coordinator.py ignores review_ok, saves whatever output is available, and unconditionally marks a module reviewed. Parallel Codex calls also name output files from integer seconds, so calls starting within one second can overwrite or read each other output. Expected: a unique output file per invocation and reviewed state only after a successful attributable review. Actual: failed or cross-wired reviews advance the checkpoint.

## Fix

### Independent verification summary (2026-09-13)

- Re-ran the coordinator review/checkpoint regressions; `tools.test_coordinator` passed all 11 tests.
- A red control allowed a failed or collided review to advance the checkpoint; `test_failed_or_empty_review_does_not_advance_checkpoint` failed, and the fix was restored.

## Notes
