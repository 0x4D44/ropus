# ROP-BUG-FLUX-00005 — Fuzz integration gate does not check rustfmt

- **State:** Open
- **Priority:** Must
- **Severity:** Low
- **Area:** integration/fuzz-format
- **Raised:** 2026-07-30
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T134403Z-p37671-n150269000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00005-run-fix-20260801T134403Z-p37671-n150269000-c1
- **Owner base:** d0b6102b118f40f6eda2c4c17175d3dc20617781
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T13:44:03Z
- **Owner until:** 2026-08-01T15:44:03Z
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
