# ROP-BUG-KILN-00012 — PGO benchmark does not measure the profile-guided binary

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** tools/pgo
- **Raised:** 2026-08-13T17:17:39Z
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
- **State history:** Open (2026-08-13T17:17:39Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Observation: tools/pgo_build.sh invokes cargo run for ropus-compare from the virtual workspace manifest without selecting the harness package, then scopes profile-use RUSTFLAGS only to the preceding cargo build. The benchmark command can fail package selection; once selected, Cargo can rebuild without the PGO fingerprint. Expected: train and directly execute the same explicitly selected instrumented and profile-use binary. Actual: the advertised PGO measurement is unavailable or measures a default release rebuild.

## Fix

<unfixed — raised only>

## Notes
