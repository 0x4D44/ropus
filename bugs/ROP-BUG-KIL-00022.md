# ROP-BUG-KIL-00022 — Tier-2 SNR oracles accept degenerate identical or silent output as passing

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness-deep-plc/tests
- **Raised:** 2026-08-19T10:46:14Z
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
- **State history:** Open (2026-08-19T10:46:14Z, raised via `deltic bugs new`) -> Fixed (2026-09-13T00:34:45Z, deltic:auto role=fix run=fix-20260913T002411Z-a6538224 branch=task/bug-ROP-BUG-KIL-00022-run-fix-20260913T002411Z-a6538224 code=45442c80dc736b6fa5c7f833a288c750b984d7fe gate=manual)

## Observation

Static review. harness-deep-plc/tests/tier2_snr.rs:221-223 returns `f64::INFINITY` when `noise_power == 0.0`, and the release gate `dnn_plc_tier2_snr_above_50db` (:276) plus `dnn_plc_tier2_lossless_regression` (:393) assert only `snr > threshold`, so bit-identical output — including both decoders emitting all-zero PCM — passes as INFINITY. The file's own calibration helper treats identical output as a fault (`assert!(first_diverge.is_some(), "... unexpectedly identical")`, :316-319), and the lossless test's comment (:360-366) documents that identical output is impossible in a healthy run (float-C vs fixed-Rust differ by 1 LSB, measured ~90 dB) — so INFINITY is always anomalous, yet the release gates accept it. Relatedly, harness-deep-plc/tests/dred_bitrate_plumbing_nonzero_diff.rs:260-266 checks `noise == 0.0 → INFINITY` before `sig == 0.0 → NEG_INFINITY`, making the degenerate-signal guard dead for exactly the both-silent case it exists to catch. Expected: a degenerate result (identical output, or zero signal energy) fails the gate. Actual: it passes at any threshold. Fix: assert `first_diverge.is_some()` (or a signal-energy floor) in both tier2_snr gates, and reorder the checks in nonzero_diff so both-silent returns NEG_INFINITY. Static inspection only; no build or test ran.

## Fix

Implemented in `harness-deep-plc/tests/tier2_snr.rs` and
`harness-deep-plc/tests/dred_bitrate_plumbing_nonzero_diff.rs`, integrated at
code commit `45442c80dc736b6fa5c7f833a288c750b984d7fe`.

- The tier-2 `compute_snr_db` helper now returns `NEG_INFINITY` for zero signal
  or zero noise, so identical, silent, or otherwise degenerate output cannot
  satisfy either SNR threshold.
- The nonzero-DRED `snr_db` helper checks zero signal before zero noise and
  returns `NEG_INFINITY` for both degenerate cases.
- Added four focused unit tests covering identical non-silent and both-silent
  PCM in both helpers.

Verification:

- `$null | deltic timeout 300 cargo test -p ropus-harness-deep-plc --test tier2_snr --test dred_bitrate_plumbing_nonzero_diff snr_tests -- --nocapture` — 4 passed, 0 failed.
- `$null | deltic timeout 600 cargo test -p ropus-harness-deep-plc --test tier2_snr --test dred_bitrate_plumbing_nonzero_diff` — 9 passed, 0 failed, including both tier-2 gates, the calibration matrix, and the nonzero-DRED gate.
- `$null | deltic timeout 900 cargo check -p ropus-harness-deep-plc`, `cargo fmt --all -- --check`, and `git diff --check` — passed.
- Red proof: temporarily restored the old zero-noise `INFINITY` behavior in both helpers. The identical non-silent and both-silent regression tests failed with `left: inf` and `right: -inf`. The fixed guards were restored before the green runs.
- Test setup fetched the pinned C reference and DNN weights with `cargo run -p fetch-assets -- all`.

## Notes
