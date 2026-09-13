# ROP-BUG-KIL-00023 — Bit-exact differential gates accept shared NaN and silently truncate on length mismatch

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
- **State history:** Open (2026-08-19T10:46:14Z, raised via `deltic bugs new`) -> Fixed (2026-09-13T00:48:35Z, deltic:auto role=fix run=fix-20260913T003640Z-30a2db99 branch=task/bug-ROP-BUG-KIL-00023-run-fix-20260913T003640Z-30a2db99 code=c0715f0560034ad0fec0130c652d9ce808d718b7 gate=manual)

## Observation

Static review. Three test files run bit-exact `to_bits()` comparisons without the shared non-finite guard that tests/support/finite_oracle.rs exists to provide (prior fix wave ROP-BUG-FLUX-00069 covered the RDOVAE tests only): (1) harness-deep-plc/tests/dred_decode_payload_diff.rs:48-55 `first_f32_divergence`, gating state/latents at :194 and :205; (2) tests/dred_encode_payload_diff.rs:141-148 `first_f32_divergent`, gating resample_mem/input_buffer/features/latents/state at :258, :274, :316, :331, :343; (3) tests/burg_cepstral_analysis_diff.rs:35-45, raw bit-pattern `assert_eq!` on all 36 cepstral outputs. A shared numerical blow-up (both sides producing the canonical quiet NaN, e.g. log of a non-positive Burg magnitude) reads as "bit-exact match" and the gates print "Tier 1 achieved". The unguarded helpers also use `.zip()`, which silently truncates on length mismatch and reports the short comparison as bit-exact (finite_oracle's `assert_finite_pair` checks equal lengths at finite_oracle.rs:15-19). Additionally tests/dred_integrated_encode.rs:433 uses `any(|f| *f != 0.0)` as the sole value assertion on `process()` output — `NaN != 0.0` is true, so all-NaN features satisfy the "RDOVAE decoder driven" gate. Expected: non-finite values fail before parity logic, per finite_oracle.rs's doc and lessons_learnt.md. Fix: `#[path]`-import finite_oracle and call `assert_finite_pair` in the three files' helpers; change the any-nonzero check to `f.is_finite() && *f != 0.0` plus an all-finite assertion. Static inspection only; no build or test ran.

## Fix

Implemented in four `harness-deep-plc/tests/` files and integrated at code
commit `c0715f0560034ad0fec0130c652d9ce808d718b7`.

- Added the shared `finite_oracle::assert_finite_pair` guard to the DRED decode,
  DRED encode, and Burg cepstral comparison helpers. It rejects non-finite
  values before `to_bits()` parity checks and rejects unequal lengths before
  `.zip()` can truncate the comparison.
- Hardened the integrated DRED decoder check with
  `finite_oracle::assert_finite_slice` and a finite nonzero-feature predicate,
  so all-NaN features cannot satisfy the liveness gate.
- Added focused same-NaN and unequal-length helper tests, plus finite-feature
  acceptance and all-NaN rejection tests. The existing ignored encoder parity
  gate remains ignored for its documented LPCNet drift, but its helper is now
  protected when explicitly run.

Verification:

- `$null | deltic timeout 900 cargo test -p ropus-harness-deep-plc --test dred_decode_payload_diff --test dred_encode_payload_diff --test burg_cepstral_analysis_diff --test dred_integrated_encode` — 32 passed, 0 failed; 1 existing encoder parity test remained ignored.
- `$null | deltic timeout 900 cargo check -p ropus-harness-deep-plc`, `cargo fmt --all -- --check`, and `git diff --check` — passed.
- Red proof: temporarily removed the finite guard from the Burg helper; its same-NaN and unequal-length `#[should_panic]` tests failed. Removing the integrated finite-slice guard also made `all_nan_fec_features_are_rejected` fail at the weaker all-zero assertion. Both guards were restored before the green suite.
- Test setup fetched the pinned C reference and DNN weights with `cargo run -p fetch-assets -- all`.

## Notes
