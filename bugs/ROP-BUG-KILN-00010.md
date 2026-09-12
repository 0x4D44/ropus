# ROP-BUG-KILN-00010 — Trace fixer crashes in build-failure recovery

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** tools/trace-fix
- **Raised:** 2026-08-13T17:17:38Z
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
- **State history:** Open (2026-08-13T17:17:38Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-12T22:44:29Z, deltic:auto role=fix run=fix-20260912T223742Z-7a97ad5d branch=task/bug-ROP-BUG-KILN-00010-run-fix-20260912T223742Z-7a97ad5d code=52c8d7e03e75a1d87ae1e5483b1b006b1b001a91 gate=manual)

## Observation

Observation: tools/trace_fix.py calls invoke_claude in both build-failure recovery branches, but that function is not defined or imported; the file defines invoke_agent and invoke_codex instead. Expected: a failed post-agent build invokes the supported fallback and continues or reports failure. Actual: the recovery path raises NameError and aborts.

## Fix

<unfixed — raised only>

## Notes
