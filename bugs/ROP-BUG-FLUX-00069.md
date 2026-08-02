# ROP-BUG-FLUX-00069 — Neural differential oracles accept non-finite output as passing

- **State:** Fixed
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-02, deltic:auto role=fix run=fix-20260802T011840Z-p16348-n137842000-c1 branch=task/bug-ROP-BUG-FLUX-00069-run-fix-20260802T011840Z-p16348-n137842000-c1 code=0159cf8 gate=manual)

## Observation

Static review at origin/main b65f812. /Users/md/language/ropus/harness-deep-plc/tests/dred_lpcnet_feature_drift.rs:152-203 reduces drift with ordered comparisons; a NaN difference makes every comparison false, leaves maxima and counts at zero, and passes the bounds at :509-545. /Users/md/language/ropus/harness-deep-plc/tests/dred_rdovae_enc_diff.rs:55-72 and :162-227 plus /Users/md/language/ropus/harness-deep-plc/tests/dred_rdovae_dec_diff.rs:92-109, :201-247, and :371-403 have the same false-green shape: a one-sided NaN makes Tier 1 divergent, but NaN SNR never replaces the initial positive infinity, so Tier 2 passes. Equal NaN bit patterns can also return early as bit-exact. Expected: non-finite neural input, output, or metrics fail immediately. Actual: catastrophic non-finite inference can be reported as zero drift, infinite SNR, or parity. Fix: require finite values before bit/SNR/drift evaluation, treat any non-finite result as an immediate failure, and add same-NaN, one-sided-NaN, positive/negative infinity, and finite-control helper tests. Static review only; no code, build, test, model, or harness ran.

## Fix

<unfixed — raised only>

## Notes
