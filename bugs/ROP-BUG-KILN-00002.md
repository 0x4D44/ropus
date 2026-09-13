# ROP-BUG-KILN-00002 — Coordinator targets obsolete paths and a nonexistent integration test

- **State:** Closed
- **Priority:** Should
- **Severity:** High
- **Area:** tools/coordinator-layout
- **Raised:** 2026-08-13T17:17:34Z
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
- **State history:** Open (2026-08-13T17:17:34Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-13T04:26:58Z, deltic:auto role=fix run=fix-20260913T041557Z-7a283f3a branch=task/bug-ROP-BUG-KILN-00002-run-fix-20260913T041557Z-7a283f3a code=6adf91311c54ae19fb2c9fb3cc856db956a56142 gate=manual) -> Closed (2026-09-13T17:49:57Z, independent two-eyes verification model=codex@xhigh, verifier=CRUCIBLE, fixer=deltic:auto, fix=6adf91311c54ae19fb2c9fb3cc856db956a56142)

## Observation

Observation: tools/coordinator.py directs implementation and review agents to root src paths and tests/harness files, but the current codec crate is under ropus/src and the active harness is a workspace member under harness. Its integration phase also runs cargo test --test integration, while no workspace manifest declares that test target. Expected: coordinator paths and validation commands resolve to current workspace members. Actual: agents are directed to non-built locations and the final phase cannot pass.

## Fix

### Independent verification summary (2026-09-13)

- Re-ran the coordinator layout, path, and integration-target regressions; `tools.test_coordinator` passed all 11 tests.
- A red control restored obsolete root paths and the nonexistent integration target; the coordinator path/command assertions failed, and the fix was restored.

## Notes
