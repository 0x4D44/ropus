# ROP-BUG-KIL-00021 — DRED differential gates skip-and-pass when WEIGHTS_BLOB is empty

- **State:** Fixed
- **Priority:** Should
- **Severity:** High
- **Area:** harness-deep-plc/tests
- **Raised:** 2026-08-19T10:45:49Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T143625Z-73e3b84e
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KIL-00021-run-verify-20260913T143625Z-73e3b84e
- **Owner base:** aef200b719fc910da03d66843da9f19ad9206332
- **Owner fingerprint:** sha256:feae18b8ec84847e21d77075273c84ad3907da7626396db395b08ee7ff74c543
- **Owner since:** 2026-09-13T14:36:25Z
- **Owner until:** 2026-09-13T16:36:25Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-19T10:45:49Z, raised via `deltic bugs new`) -> Fixed (2026-09-13T00:22:10Z, deltic:auto role=fix run=fix-20260913T001300Z-29800934 branch=task/bug-ROP-BUG-KIL-00021-run-fix-20260913T001300Z-29800934 code=0959774e1aaf2ad36f99b259509a1657de843b57 gate=manual)

## Observation

Static review of harness-deep-plc/ at the current worktree head. Nine live (non-`#[ignore]`) DRED differential gates begin with `if !weights_or_skip() { return; }` — a bare return from `#[test]` is a PASS. Sites: harness-deep-plc/tests/dred_rdovae_enc_diff.rs:111, dred_rdovae_dec_diff.rs:148 and :278, dred_lpcnet_feature_drift.rs:446, dred_integrated_encode.rs:297, :355, :405, dred_bitrate_plumbing_nonzero_diff.rs:321, dred_bitrate_plumbing_diff.rs:195. The helper's own comment says "Emit a loud skip, don't silently pass", but `cargo test` hides the `eprintln!` without `--nocapture`. Trigger: `ropus/build.rs:59-70` and :88-94 write an EMPTY blob with only a `cargo:warning` on any `try_build_blob` failure (compiler/linker hiccup, cross-build), while harness-deep-plc/build.rs:91-93 gates `no_reference` only on three reference files existing — so the crate compiles, runs, and reports the entire DRED differential surface green with zero comparisons performed. Expected: assets missing means red or visibly ignored, never a silent pass — per lessons_learnt.md ("never print a skip and pass") and the pattern already in this crate at dred_dtx_first_frame_diff.rs:52-57 (`require_weights()` asserts). Inconsistently, tier2_snr.rs never checks the blob and fails loudly at ~9 dB in the same state. Fix: replace `weights_or_skip` with the asserting `require_weights` pattern (or `#[ignore]` + loud failure when run explicitly) in all nine sites. Static inspection only; no build or test ran.

## Fix

Implemented in six `harness-deep-plc/tests/` files and integrated at code
commit `0959774e1aaf2ad36f99b259509a1657de843b57`.

- Replaced `weights_or_skip` with a `require_weights` assertion in the six
  files containing the nine live DRED differential gates.
- An empty `WEIGHTS_BLOB` now fails the test with an actionable
  `fetch-assets -- all` message instead of returning from `#[test]` and being
  reported as a pass.
- The already-ignored `dred_encode_payload_diff` test remains unchanged; the
  nine non-ignored gates named in the observation are all covered.

Verification:

- `$null | deltic timeout 1200 cargo test -p ropus-harness-deep-plc --test dred_rdovae_enc_diff --test dred_rdovae_dec_diff --test dred_lpcnet_feature_drift --test dred_integrated_encode --test dred_bitrate_plumbing_nonzero_diff --test dred_bitrate_plumbing_diff` — 27 passed, 0 failed, including all nine affected gates.
- `$null | deltic timeout 900 cargo check -p ropus-harness-deep-plc` — passed.
- `cargo fmt --all -- --check` and `git diff --check` — passed.
- Red proof: temporarily added an intentional `#error` to `ropus/build/gen_weights_blob.c`, forcing an empty embedded blob. `dred_rdovae_enc_diff` then failed at `require_weights` (4 support tests passed, the differential gate failed). The generator was restored before the green validation.
- Test setup fetched the pinned C reference and DNN weights with `cargo run -p fetch-assets -- all`.

## Notes
