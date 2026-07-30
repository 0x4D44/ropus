# ROP-BUG-FLUX-00001 — Workspace rustfmt gate fails on ropus-fb2k roundtrip tests

- **State:** Open
- **Priority:** Must
- **Severity:** Low
- **Area:** validation/ropus-fb2k
- **Raised:** 2026-07-30
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260730T172357Z-p67668-n901388000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00001-run-fix-20260730T172357Z-p67668-n901388000-c1
- **Owner base:** 3c1e2f71383dada8f41e7d3cf17495fdb834727c
- **Owner fingerprint:** -
- **Owner since:** 2026-07-30T17:23:57Z
- **Owner until:** 2026-07-30T19:23:57Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-30, raised via `deltic bugs new`)

## Observation

Observed on macOS in a clean task worktree at origin/main c2658d87e44886ce4210be3fae798fe6913cd40e. Reproduce: cargo fmt --all -- --check. Expected: exit 0 with no diff. Actual: exit 1 and rustfmt diffs throughout ropus-fb2k/tests/roundtrip.rs, first at line 432. Repeated once unchanged.

## Fix

<unfixed — raised only>

## Notes
