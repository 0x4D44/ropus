# ROP-BUG-FLUX-00006 — Fuzz integration gate depends on absent ignored reference checkout

- **State:** Fixed
- **Priority:** Must
- **Severity:** Medium
- **Area:** integration/fuzz-assets
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
- **State history:** Open (2026-07-30, raised via `deltic bugs new`) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T151741Z-p75566-n609680000-c1 branch=task/bug-ROP-BUG-FLUX-00006-run-fix-20260801T151741Z-p75566-n609680000-c1 code=04249b7 gate=manual)

## Observation

The harness and fuzz components run `cargo check --manifest-path tests/fuzz/Cargo.toml --locked`, while `tests/fuzz/build.rs` panics when `reference/celt/bands.c` is absent. The `reference/` directory is gitignored and contains no tracked files, so a clean Deltic task worktree cannot satisfy this focused gate without an out-of-band asset fetch. Expected the integration gate to be reproducible from a clean worktree or to provision its pinned prerequisite explicitly.

## Fix

<unfixed — raised only>

## Notes
