# ROP-BUG-KILN-00012 — PGO benchmark does not measure the profile-guided binary

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** tools/pgo
- **Raised:** 2026-08-13T17:17:39Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T150309Z-688023b5
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KILN-00012-run-verify-20260913T150309Z-688023b5
- **Owner base:** 25f07192744685e6348c2a3e4f30ac90f2082062
- **Owner fingerprint:** sha256:c620830ac2ece353ab148f08f2de7ce1ed5fed11e66dcc9256c681042a680556
- **Owner since:** 2026-09-13T15:03:09Z
- **Owner until:** 2026-09-13T17:03:09Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-13T17:17:39Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-12T23:10:39Z, deltic:auto role=fix run=fix-20260912T230146Z-5d89dcd9 branch=task/bug-ROP-BUG-KILN-00012-run-fix-20260912T230146Z-5d89dcd9 code=05b5c28bb2809135d28c9f8adefa1556a44521ad gate=manual)

## Observation

Observation: tools/pgo_build.sh invokes cargo run for ropus-compare from the virtual workspace manifest without selecting the harness package, then scopes profile-use RUSTFLAGS only to the preceding cargo build. The benchmark command can fail package selection; once selected, Cargo can rebuild without the PGO fingerprint. Expected: train and directly execute the same explicitly selected instrumented and profile-use binary. Actual: the advertised PGO measurement is unavailable or measures a default release rebuild.

## Fix

Integrated code commit `05b5c28bb2809135d28c9f8adefa1556a44521ad` now builds
the explicitly selected `ropus-harness`/`ropus-compare` target with one fixed
target directory. Training and both benchmark runs execute the resulting
binary directly, while the launcher records its SHA-256 before and after each
measurement. Focused coverage lives in `tools/test_pgo_build.py`.

Validation evidence:

- Regression proof: replacing `run_benchmark`'s direct binary invocation with
  the old `cargo run` command made
  `test_builds_and_measures_the_selected_pgo_binary_directly` fail on its own
  `assertEqual(len(cargo_lines), 3, cargo_lines)` assertion after observing
  five Cargo calls, including two `ARGS=run` entries.
- After restoration, `python -m unittest -v tools.test_pgo_build` passed the
  selected fake-tool integration test. It observed three package-selected
  builds, profile-generate/profile-use flags, direct baseline/training/PGO
  invocations, and unchanged binary identities around both benchmarks.
- Normalized `bash -n tools/pgo_build.sh` and
  `python -m py_compile tools/test_pgo_build.py` passed. The exact real command
  `deltic timeout 180 cargo build --release --manifest-path
  harness/Cargo.toml --target-dir target --package ropus-harness --bin
  ropus-compare` also completed successfully; only the repository's existing
  missing-reference/DNN-data warnings were emitted.

## Notes
