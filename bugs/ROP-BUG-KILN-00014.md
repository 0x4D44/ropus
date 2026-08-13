# ROP-BUG-KILN-00014 — Coordinator checkpoint writes can corrupt resumable state

- **State:** Open
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
- **State history:** Open (2026-08-13T17:17:41Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Observation: tools/coordinator.py truncates coordinator_state.json and writes JSON directly in place. Process termination, disk-full, or write failure can leave invalid JSON; load_state then parses it without recovery and status, run, and resume crash. Expected: write, flush, and atomically replace a same-directory temporary file, with a clear recovery error. Actual: an interrupted checkpoint can permanently wedge the coordinator.

## Fix

<unfixed — raised only>

## Notes
