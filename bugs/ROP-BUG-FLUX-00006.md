# ROP-BUG-FLUX-00006 — Fuzz integration gate depends on absent ignored reference checkout

- **State:** Open
- **Priority:** Must
- **Severity:** Medium
- **Area:** integration/fuzz-assets
- **Raised:** 2026-07-30
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T151741Z-p75566-n609680000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00006-run-fix-20260801T151741Z-p75566-n609680000-c1
- **Owner base:** 63f5edc9effcec58ef68878c1aa08b6d7b4ca772
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T15:17:41Z
- **Owner until:** 2026-08-01T17:17:41Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-30, raised via `deltic bugs new`)

## Observation

The harness and fuzz components run `cargo check --manifest-path tests/fuzz/Cargo.toml --locked`, while `tests/fuzz/build.rs` panics when `reference/celt/bands.c` is absent. The `reference/` directory is gitignored and contains no tracked files, so a clean Deltic task worktree cannot satisfy this focused gate without an out-of-band asset fetch. Expected the integration gate to be reproducible from a clean worktree or to provision its pinned prerequisite explicitly.

## Fix

<unfixed — raised only>

## Notes
