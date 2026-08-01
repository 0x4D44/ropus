# ROP-BUG-FLUX-00006 — Fuzz integration gate depends on absent ignored reference checkout

- **State:** Closed
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
- **State history:** Open (2026-07-30, raised via `deltic bugs new`) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T151741Z-p75566-n609680000-c1 branch=task/bug-ROP-BUG-FLUX-00006-run-fix-20260801T151741Z-p75566-n609680000-c1 code=04249b7 gate=manual) -> Closed (2026-08-01, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main dc05a88; fixer was a prior automated fix session, verifier is a different actor)

## Observation

The harness and fuzz components run `cargo check --manifest-path tests/fuzz/Cargo.toml --locked`, while `tests/fuzz/build.rs` panics when `reference/celt/bands.c` is absent. The `reference/` directory is gitignored and contains no tracked files, so a clean Deltic task worktree cannot satisfy this focused gate without an out-of-band asset fetch. Expected the integration gate to be reproducible from a clean worktree or to provision its pinned prerequisite explicitly.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-01, independent two-eyes, host flux)

- From a clean task worktree (no prior `reference/` checkout), `cargo run -p fetch-assets
  --locked -- reference` followed by `cargo check --manifest-path tests/fuzz/Cargo.toml
  --locked` succeeds without a manual out-of-band asset fetch.
- `.deltic-integrate.toml`'s `harness` and `fuzz` components both now run
  `cargo run -p fetch-assets --locked -- reference` immediately before the fuzz `cargo check`,
  so the gate is reproducible from a bare worktree as the bug required.
