# ROP-BUG-KILN-00014 — Coordinator checkpoint writes can corrupt resumable state

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** tools/coordinator-state
- **Raised:** 2026-08-13T17:17:41Z
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
- **State history:** Open (2026-08-13T17:17:41Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-13T04:41:58Z, deltic:auto role=fix run=fix-20260913T043331Z-bb6ca5a0 branch=task/bug-ROP-BUG-KILN-00014-run-fix-20260913T043331Z-bb6ca5a0 code=aa6227ab4a3df7170c488ef633d787e207d201f0 gate=manual)

## Observation

Observation: tools/coordinator.py truncates coordinator_state.json and writes JSON directly in place. Process termination, disk-full, or write failure can leave invalid JSON; load_state then parses it without recovery and status, run, and resume crash. Expected: write, flush, and atomically replace a same-directory temporary file, with a clear recovery error. Actual: an interrupted checkpoint can permanently wedge the coordinator.

## Fix

<unfixed — raised only>

## Notes
