# ROP-BUG-KIL-00022 — Tier-2 SNR oracles accept degenerate identical or silent output as passing

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness-deep-plc/tests
- **Raised:** 2026-08-19T10:46:14Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260913T002411Z-a6538224
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KIL-00022-run-fix-20260913T002411Z-a6538224
- **Owner base:** 1f9ad6c7d15c45a97a998ee49f619c19c46a39f5
- **Owner fingerprint:** -
- **Owner since:** 2026-09-13T00:24:11Z
- **Owner until:** 2026-09-13T02:24:11Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-19T10:46:14Z, raised via `deltic bugs new`)

## Observation

Static review. harness-deep-plc/tests/tier2_snr.rs:221-223 returns `f64::INFINITY` when `noise_power == 0.0`, and the release gate `dnn_plc_tier2_snr_above_50db` (:276) plus `dnn_plc_tier2_lossless_regression` (:393) assert only `snr > threshold`, so bit-identical output — including both decoders emitting all-zero PCM — passes as INFINITY. The file's own calibration helper treats identical output as a fault (`assert!(first_diverge.is_some(), "... unexpectedly identical")`, :316-319), and the lossless test's comment (:360-366) documents that identical output is impossible in a healthy run (float-C vs fixed-Rust differ by 1 LSB, measured ~90 dB) — so INFINITY is always anomalous, yet the release gates accept it. Relatedly, harness-deep-plc/tests/dred_bitrate_plumbing_nonzero_diff.rs:260-266 checks `noise == 0.0 → INFINITY` before `sig == 0.0 → NEG_INFINITY`, making the degenerate-signal guard dead for exactly the both-silent case it exists to catch. Expected: a degenerate result (identical output, or zero signal energy) fails the gate. Actual: it passes at any threshold. Fix: assert `first_diverge.is_some()` (or a signal-energy floor) in both tier2_snr gates, and reorder the checks in nonzero_diff so both-silent returns NEG_INFINITY. Static inspection only; no build or test ran.

## Fix

<unfixed — raised only>

## Notes
