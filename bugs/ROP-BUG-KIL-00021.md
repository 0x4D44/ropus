# ROP-BUG-KIL-00021 — DRED differential gates skip-and-pass when WEIGHTS_BLOB is empty

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** harness-deep-plc/tests
- **Raised:** 2026-08-19T10:45:49Z
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
- **State history:** Open (2026-08-19T10:45:49Z, raised via `deltic bugs new`)

## Observation

Static review of harness-deep-plc/ at the current worktree head. Nine live (non-`#[ignore]`) DRED differential gates begin with `if !weights_or_skip() { return; }` — a bare return from `#[test]` is a PASS. Sites: harness-deep-plc/tests/dred_rdovae_enc_diff.rs:111, dred_rdovae_dec_diff.rs:148 and :278, dred_lpcnet_feature_drift.rs:446, dred_integrated_encode.rs:297, :355, :405, dred_bitrate_plumbing_nonzero_diff.rs:321, dred_bitrate_plumbing_diff.rs:195. The helper's own comment says "Emit a loud skip, don't silently pass", but `cargo test` hides the `eprintln!` without `--nocapture`. Trigger: `ropus/build.rs:59-70` and :88-94 write an EMPTY blob with only a `cargo:warning` on any `try_build_blob` failure (compiler/linker hiccup, cross-build), while harness-deep-plc/build.rs:91-93 gates `no_reference` only on three reference files existing — so the crate compiles, runs, and reports the entire DRED differential surface green with zero comparisons performed. Expected: assets missing means red or visibly ignored, never a silent pass — per lessons_learnt.md ("never print a skip and pass") and the pattern already in this crate at dred_dtx_first_frame_diff.rs:52-57 (`require_weights()` asserts). Inconsistently, tier2_snr.rs never checks the blob and fails loudly at ~9 dB in the same state. Fix: replace `weights_or_skip` with the asserting `require_weights` pattern (or `#[ignore]` + loud failure when run explicitly) in all nine sites. Static inspection only; no build or test ran.

## Fix

<unfixed — raised only>

## Notes
