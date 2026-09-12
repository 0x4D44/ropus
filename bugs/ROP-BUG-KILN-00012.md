# ROP-BUG-KILN-00012 — PGO benchmark does not measure the profile-guided binary

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** tools/pgo
- **Raised:** 2026-08-13T17:17:39Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260912T230146Z-5d89dcd9
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KILN-00012-run-fix-20260912T230146Z-5d89dcd9
- **Owner base:** 0e1b23fb08b76dd0aeb77223a1b6578a783a30c0
- **Owner fingerprint:** -
- **Owner since:** 2026-09-12T23:01:46Z
- **Owner until:** 2026-09-13T01:01:46Z
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
