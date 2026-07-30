# ROP-BUG-FLUX-00001 — Workspace rustfmt gate fails on ropus-fb2k roundtrip tests

- **State:** Fixed
- **Priority:** Must
- **Severity:** Low
- **Area:** validation/ropus-fb2k
- **Raised:** 2026-07-30
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
- **State history:** Open (2026-07-30, raised via `deltic bugs new`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T205437Z-p43457-n353831000-c1 branch=task/bug-ROP-BUG-FLUX-00001-run-fix-20260730T205437Z-p43457-n353831000-c1 code=4ced261 gate=manual)

## Observation

Observed on macOS in a clean task worktree at origin/main c2658d87e44886ce4210be3fae798fe6913cd40e. Reproduce: cargo fmt --all -- --check. Expected: exit 0 with no diff. Actual: exit 1 and rustfmt diffs throughout ropus-fb2k/tests/roundtrip.rs, first at line 432. Repeated once unchanged.

## Fix

<unfixed — raised only>

## Notes
