# ROP-BUG-KILN-00014 — Coordinator checkpoint writes can corrupt resumable state

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** tools/coordinator-state
- **Raised:** 2026-08-13T17:17:41Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T150530Z-9996e271
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KILN-00014-run-verify-20260913T150530Z-9996e271
- **Owner base:** f03d92ad3923dc31779e2c77e795916a14a0d366
- **Owner fingerprint:** sha256:ade119c2f0165cee8178b643d59412bfeda0bdf2bd1771d262d2247628a8691d
- **Owner since:** 2026-09-13T15:05:30Z
- **Owner until:** 2026-09-13T17:05:30Z
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
