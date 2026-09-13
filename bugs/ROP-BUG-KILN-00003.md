# ROP-BUG-KILN-00003 — Coordinator can mark failed or collided reviews complete

- **State:** Fixed
- **Priority:** Should
- **Severity:** High
- **Area:** tools/coordinator-review
- **Raised:** 2026-08-13T17:17:35Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T170327Z-67746778
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KILN-00003-run-verify-20260913T170327Z-67746778
- **Owner base:** d0c03fcfd737cec728baedc0309f0b59d8b69ef0
- **Owner fingerprint:** sha256:2f053987bf17f085030487ce1b0af095bc39a4d6f7b1e0472f75681630642ec1
- **Owner since:** 2026-09-13T17:03:27Z
- **Owner until:** 2026-09-13T19:03:27Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-13T17:17:35Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-13T04:32:34Z, deltic:auto role=fix run=fix-20260913T042804Z-c2488a79 branch=task/bug-ROP-BUG-KILN-00003-run-fix-20260913T042804Z-c2488a79 code=2fab4b0e9f54cc7b7e396b3c957836c8ac67319f gate=manual)

## Observation

Observation: tools/coordinator.py ignores review_ok, saves whatever output is available, and unconditionally marks a module reviewed. Parallel Codex calls also name output files from integer seconds, so calls starting within one second can overwrite or read each other output. Expected: a unique output file per invocation and reviewed state only after a successful attributable review. Actual: failed or cross-wired reviews advance the checkpoint.

## Fix

<unfixed — raised only>

## Notes
