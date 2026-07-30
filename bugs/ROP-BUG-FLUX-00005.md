# ROP-BUG-FLUX-00005 — Fuzz integration gate does not check rustfmt

- **State:** Open
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
- **State history:** Open (2026-07-30, raised via `deltic bugs new`)

## Observation

The fuzz component in `.deltic-integrate.toml` runs `cargo fmt --all -- --check` from the repository root, but `tests/fuzz` is excluded from the root workspace and declares its own workspace. Cargo metadata confirms the two workspace roots are distinct, so edits under `tests/fuzz` can pass the selected gate without being checked by rustfmt. Expected the fuzz gate to run rustfmt against `tests/fuzz/Cargo.toml`.

## Fix

<unfixed — raised only>

## Notes
