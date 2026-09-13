# ROP-BUG-KILN-00003 — Coordinator can mark failed or collided reviews complete

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** tools/coordinator-review
- **Raised:** 2026-08-13T17:17:35Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260913T042804Z-c2488a79
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KILN-00003-run-fix-20260913T042804Z-c2488a79
- **Owner base:** 2317498964c1eb946d072e4f70dab95810123398
- **Owner fingerprint:** -
- **Owner since:** 2026-09-13T04:28:04Z
- **Owner until:** 2026-09-13T06:28:04Z
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
