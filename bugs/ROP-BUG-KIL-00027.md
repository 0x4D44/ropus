# ROP-BUG-KIL-00027 — build.rs sentinel guard misses partially-extracted weights tarball

- **State:** Closed
- **Priority:** Could
- **Severity:** Low
- **Area:** harness-deep-plc/build
- **Raised:** 2026-08-19T10:46:21Z
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
- **State history:** Open (2026-08-19T10:46:21Z, raised via `deltic bugs new`) -> Fixed (2026-09-13T02:03:50Z, deltic:auto role=fix run=fix-20260913T015731Z-54a4eb27 branch=task/bug-ROP-BUG-KIL-00027-run-fix-20260913T015731Z-54a4eb27 code=2d8b1e7dabbac7b3e9ce532a1c63c3170e247080 gate=manual) -> Closed (2026-09-13T16:28:29Z, independent two-eyes verification model=codex@xhigh, verifier=CRUCIBLE, fixer=deltic:auto, fix=2d8b1e7dabbac7b3e9ce532a1c63c3170e247080)

## Observation

Static review. harness-deep-plc/build.rs:91-93 gates the friendly `no_reference` downgrade on just three sentinel files (`celt/bands.c`, `dnn/fargan_data.c`, `dnn/plc_data.c`), but the compiled source lists need more files from the same weights tarball: `dnn/pitchdnn_data.c` (:286) and four `dnn/dred_rdovae_*_data.c` files (:305-309). A partially-extracted tarball (e.g. `fetch-assets` interrupted mid-extraction) passes the sentinel check, and the build then dies with a raw cc "No such file or directory" deep in `reference/` instead of the actionable "run `cargo run -p fetch-assets -- all`" warning the guard exists to give (:101-107). Loud either way, so Low severity — but the guard fails at its one job in a reachable state. Fix: extend the sentinel list to cover every `*_data.c` the source lists compile (or probe the lists themselves). Static inspection only; no build ran.

## Fix

### Verification summary (2026-09-13, independent verifier)

- The deep-PLC build/check gate passed with the complete reference asset set.
- Red control: removing `reference/dnn/pitchdnn_data.c` from an otherwise provisioned reference tree produced the missing-data warning during `cargo check -p ropus-harness-deep-plc`; the file was restored.

## Notes
