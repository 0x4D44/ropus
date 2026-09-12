# ROP-BUG-KILN-00010 — Trace fixer crashes in build-failure recovery

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** tools/trace-fix
- **Raised:** 2026-08-13T17:17:38Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260912T223742Z-7a97ad5d
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KILN-00010-run-fix-20260912T223742Z-7a97ad5d
- **Owner base:** b3ff2a1612d182381d880a4320ef798a83b483b6
- **Owner fingerprint:** -
- **Owner since:** 2026-09-12T22:37:42Z
- **Owner until:** 2026-09-13T00:37:42Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-13T17:17:38Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Observation: tools/trace_fix.py calls invoke_claude in both build-failure recovery branches, but that function is not defined or imported; the file defines invoke_agent and invoke_codex instead. Expected: a failed post-agent build invokes the supported fallback and continues or reports failure. Actual: the recovery path raises NameError and aborts.

## Fix

<unfixed — raised only>

## Notes
