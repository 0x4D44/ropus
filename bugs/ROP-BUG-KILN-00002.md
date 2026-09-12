# ROP-BUG-KILN-00002 — Coordinator targets obsolete paths and a nonexistent integration test

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** tools/coordinator-layout
- **Raised:** 2026-08-13T17:17:34Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260912T212329Z-0805dcf7
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KILN-00002-run-fix-20260912T212329Z-0805dcf7
- **Owner base:** bbfb65646418d78f37b8064eff7ff4b7ccc01756
- **Owner fingerprint:** -
- **Owner since:** 2026-09-12T21:23:29Z
- **Owner until:** 2026-09-12T23:23:29Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-13T17:17:34Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Observation: tools/coordinator.py directs implementation and review agents to root src paths and tests/harness files, but the current codec crate is under ropus/src and the active harness is a workspace member under harness. Its integration phase also runs cargo test --test integration, while no workspace manifest declares that test target. Expected: coordinator paths and validation commands resolve to current workspace members. Actual: agents are directed to non-built locations and the final phase cannot pass.

## Fix

<unfixed — raised only>

## Notes
