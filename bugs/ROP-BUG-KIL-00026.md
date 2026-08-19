# ROP-BUG-KIL-00026 — RDOVAE Tier-1 gates silently downgrade to 60 dB Tier-2 on bit-exactness loss

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness-deep-plc/tests
- **Raised:** 2026-08-19T10:46:20Z
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
- **State history:** Open (2026-08-19T10:46:20Z, raised via `deltic bugs new`)

## Observation

Static review. The RDOVAE differential gates claim Tier-1 (bit-exact) status — CLAUDE.md records "Tier-1 ... on ... RDOVAE forward pass" — but enforce only Tier-2: harness-deep-plc/tests/dred_rdovae_enc_diff.rs:206-239 and dred_rdovae_dec_diff.rs:235-259 plus :407-417 take `if all_bit_exact { eprintln!("Tier 1 achieved ..."); return; }` and otherwise fall back to a 60 dB SNR assertion. A regression that breaks bit-exactness but stays above 60 dB passes green; the only signal is an `eprintln!` hidden by `cargo test`'s capture. Expected: an achieved tier is ratcheted (as dred_lpcnet_feature_drift.rs does with its locked bounds and BIT_EXACT_FEATURE_INDICES). Actual: the tier can silently degrade. Fix: assert `all_bit_exact` (keeping the SNR diagnostics in the failure message), or gate the fallback behind an explicit env/feature opt-out documented as a tier downgrade. Static inspection only; no build or test ran.

## Fix

<unfixed — raised only>

## Notes
