# ROP-BUG-FLUX-00005 — Fuzz integration gate does not check rustfmt

- **State:** Closed
- **Priority:** Must
- **Severity:** Low
- **Area:** integration/fuzz-format
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
- **State history:** Open (2026-07-30, raised via `deltic bugs new`) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T134403Z-p37671-n150269000-c1 branch=task/bug-ROP-BUG-FLUX-00005-run-fix-20260801T134403Z-p37671-n150269000-c1 code=e4a6ec3 gate=manual) -> Closed (2026-08-01, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main dc05a88; fixer was a prior automated fix session, verifier is a different actor)

## Observation

The fuzz component in `.deltic-integrate.toml` runs `cargo fmt --all -- --check` from the repository root, but `tests/fuzz` is excluded from the root workspace and declares its own workspace. Cargo metadata confirms the two workspace roots are distinct, so edits under `tests/fuzz` can pass the selected gate without being checked by rustfmt. Expected the fuzz gate to run rustfmt against `tests/fuzz/Cargo.toml`.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-01, independent two-eyes, host flux)

- `cargo fmt --manifest-path tests/fuzz/Cargo.toml -- --check` exits 0 from a clean worktree.
- `.deltic-integrate.toml`'s `fuzz` component now runs this exact command as its own gate step,
  separate from the root `cargo fmt --all -- --check`, so edits under `tests/fuzz` (a distinct
  Cargo workspace per `cargo metadata`) can no longer bypass formatting review.
