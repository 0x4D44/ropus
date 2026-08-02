# ROP-BUG-FLUX-00069 — Neural differential oracles accept non-finite output as passing

- **State:** Closed
- **Priority:** Must
- **Severity:** Medium
- **Area:** harness-deep-plc/neural-oracles
- **Raised:** 2026-07-31
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-02, deltic:auto role=fix run=fix-20260802T011840Z-p16348-n137842000-c1 branch=task/bug-ROP-BUG-FLUX-00069-run-fix-20260802T011840Z-p16348-n137842000-c1 code=0159cf8 gate=manual) -> Closed (2026-08-02, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main 60de518; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main b65f812. /Users/md/language/ropus/harness-deep-plc/tests/dred_lpcnet_feature_drift.rs:152-203 reduces drift with ordered comparisons; a NaN difference makes every comparison false, leaves maxima and counts at zero, and passes the bounds at :509-545. /Users/md/language/ropus/harness-deep-plc/tests/dred_rdovae_enc_diff.rs:55-72 and :162-227 plus /Users/md/language/ropus/harness-deep-plc/tests/dred_rdovae_dec_diff.rs:92-109, :201-247, and :371-403 have the same false-green shape: a one-sided NaN makes Tier 1 divergent, but NaN SNR never replaces the initial positive infinity, so Tier 2 passes. Equal NaN bit patterns can also return early as bit-exact. Expected: non-finite neural input, output, or metrics fail immediately. Actual: catastrophic non-finite inference can be reported as zero drift, infinite SNR, or parity. Fix: require finite values before bit/SNR/drift evaluation, treat any non-finite result as an immediate failure, and add same-NaN, one-sided-NaN, positive/negative infinity, and finite-control helper tests. Static review only; no code, build, test, model, or harness ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-02, independent two-eyes, host flux)

- Fix commit `0159cf8` adds a new shared `harness-deep-plc/tests/support/finite_oracle.rs`
  module (`finite_slice_error`, `finite_pair_error` and their `assert_*` wrappers) and wires
  explicit finiteness assertions into `dred_lpcnet_feature_drift.rs`,
  `dred_rdovae_enc_diff.rs`, and `dred_rdovae_dec_diff.rs` before any bit-exact/SNR/drift
  math runs. Confirmed by reading the diff: `lpcnet_feature_drift_report` now asserts
  `assert_finite_pair`/finite-diff/finite-rms at each stage, exactly where the bug describes
  NaN silently leaving maxima at their optimistic initial values.
- The `finite_oracle::tests` module (`same_nan_is_rejected`, `one_sided_nan_is_rejected`,
  `positive_and_negative_infinity_are_rejected`, `finite_control_is_accepted`) is a
  self-contained regression suite that directly targets the described false-green shapes and
  needs no C reference to run; it is new in this commit and did not exist before the fix.
- Rebuilt against the real xiph/opus C reference (vendored under `reference/`, linked via the
  `cc` crate) on a fresh worktree at `origin/main` `60de518` and reran every touched test
  file: `cargo test -p ropus-harness-deep-plc --test dred_lpcnet_feature_drift --test
  dred_rdovae_enc_diff --test dred_rdovae_dec_diff` — 20 passed, 0 failed, including the four
  finite-oracle unit cases and the live Rust-vs-C differential/drift cases
  (`test_dred_lpcnet_feature_drift_is_bounded_against_c_reference`,
  `rdovae_decode_qframe_matches_c_reference`, `rdovae_encode_dframe_matches_c_reference`).
- `cargo clippy -p ropus-harness-deep-plc --all-targets --locked -- -D warnings`: clean.
