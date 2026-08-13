# ROP-BUG-KILN-00010 — Trace fixer crashes in build-failure recovery

- **State:** Open
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
- **State history:** Open (2026-08-13T17:17:38Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Observation: tools/trace_fix.py calls invoke_claude in both build-failure recovery branches, but that function is not defined or imported; the file defines invoke_agent and invoke_codex instead. Expected: a failed post-agent build invokes the supported fallback and continues or reports failure. Actual: the recovery path raises NameError and aborts.

## Fix

<unfixed — raised only>

## Notes
