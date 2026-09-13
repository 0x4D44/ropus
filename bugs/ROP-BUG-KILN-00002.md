# ROP-BUG-KILN-00002 — Coordinator targets obsolete paths and a nonexistent integration test

- **State:** Fixed
- **Priority:** Should
- **Severity:** High
- **Area:** tools/coordinator-layout
- **Raised:** 2026-08-13T17:17:34Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T170049Z-cd13254c
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KILN-00002-run-verify-20260913T170049Z-cd13254c
- **Owner base:** ba09d5ecf3a8e333058eb745ba9454ec006b7c00
- **Owner fingerprint:** sha256:9833050675c0f5c0ff20bb2bd45a69346ca69eff8ed31b8dc5d5530ed8a65411
- **Owner since:** 2026-09-13T17:00:49Z
- **Owner until:** 2026-09-13T19:00:49Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-13T17:17:34Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-13T04:26:58Z, deltic:auto role=fix run=fix-20260913T041557Z-7a283f3a branch=task/bug-ROP-BUG-KILN-00002-run-fix-20260913T041557Z-7a283f3a code=6adf91311c54ae19fb2c9fb3cc856db956a56142 gate=manual)

## Observation

Observation: tools/coordinator.py directs implementation and review agents to root src paths and tests/harness files, but the current codec crate is under ropus/src and the active harness is a workspace member under harness. Its integration phase also runs cargo test --test integration, while no workspace manifest declares that test target. Expected: coordinator paths and validation commands resolve to current workspace members. Actual: agents are directed to non-built locations and the final phase cannot pass.

## Fix

<unfixed — raised only>

## Notes
