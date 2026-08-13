# ROP-BUG-KILN-00003 — Coordinator can mark failed or collided reviews complete

- **State:** Open
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
- **State history:** Open (2026-08-13T17:17:35Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Observation: tools/coordinator.py ignores review_ok, saves whatever output is available, and unconditionally marks a module reviewed. Parallel Codex calls also name output files from integer seconds, so calls starting within one second can overwrite or read each other output. Expected: a unique output file per invocation and reviewed state only after a successful attributable review. Actual: failed or cross-wired reviews advance the checkpoint.

## Fix

<unfixed — raised only>

## Notes
