# ROP-BUG-KIL-00023 — Bit-exact differential gates accept shared NaN and silently truncate on length mismatch

- **State:** Open
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
- **State history:** Open (2026-08-19T10:46:14Z, raised via `deltic bugs new`)

## Observation

Static review. Three test files run bit-exact `to_bits()` comparisons without the shared non-finite guard that tests/support/finite_oracle.rs exists to provide (prior fix wave ROP-BUG-FLUX-00069 covered the RDOVAE tests only): (1) harness-deep-plc/tests/dred_decode_payload_diff.rs:48-55 `first_f32_divergence`, gating state/latents at :194 and :205; (2) tests/dred_encode_payload_diff.rs:141-148 `first_f32_divergent`, gating resample_mem/input_buffer/features/latents/state at :258, :274, :316, :331, :343; (3) tests/burg_cepstral_analysis_diff.rs:35-45, raw bit-pattern `assert_eq!` on all 36 cepstral outputs. A shared numerical blow-up (both sides producing the canonical quiet NaN, e.g. log of a non-positive Burg magnitude) reads as "bit-exact match" and the gates print "Tier 1 achieved". The unguarded helpers also use `.zip()`, which silently truncates on length mismatch and reports the short comparison as bit-exact (finite_oracle's `assert_finite_pair` checks equal lengths at finite_oracle.rs:15-19). Additionally tests/dred_integrated_encode.rs:433 uses `any(|f| *f != 0.0)` as the sole value assertion on `process()` output — `NaN != 0.0` is true, so all-NaN features satisfy the "RDOVAE decoder driven" gate. Expected: non-finite values fail before parity logic, per finite_oracle.rs's doc and lessons_learnt.md. Fix: `#[path]`-import finite_oracle and call `assert_finite_pair` in the three files' helpers; change the any-nonzero check to `f.is_finite() && *f != 0.0` plus an all-finite assertion. Static inspection only; no build or test ran.

## Fix

<unfixed — raised only>

## Notes
