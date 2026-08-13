# ROP-BUG-KILN-00002 — Coordinator targets obsolete paths and a nonexistent integration test

- **State:** Open
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
- **State history:** Open (2026-08-13T17:17:34Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Observation: tools/coordinator.py directs implementation and review agents to root src paths and tests/harness files, but the current codec crate is under ropus/src and the active harness is a workspace member under harness. Its integration phase also runs cargo test --test integration, while no workspace manifest declares that test target. Expected: coordinator paths and validation commands resolve to current workspace members. Actual: agents are directed to non-built locations and the final phase cannot pass.

## Fix

<unfixed — raised only>

## Notes
