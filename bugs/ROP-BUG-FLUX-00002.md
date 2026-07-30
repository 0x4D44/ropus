# ROP-BUG-FLUX-00002 — Full-test benchmark threshold assertions are stale

- **State:** Open
- **Priority:** Must
- **Severity:** Medium
- **Area:** full-test/benchmarks
- **Raised:** 2026-07-30
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260730T205505Z-p44875-n625225000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00002-run-fix-20260730T205505Z-p44875-n625225000-c1
- **Owner base:** acfb6ad4dabe3086e6ba9d7f9548a5d84ef820fa
- **Owner fingerprint:** -
- **Owner since:** 2026-07-30T20:55:05Z
- **Owner until:** 2026-07-30T22:55:05Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-30, raised via `deltic bugs new`)

## Observation

Observed on macOS in a clean task worktree at origin/main c2658d87e44886ce4210be3fae798fe6913cd40e. Reproduce: cargo test -p full-test --locked. Expected: all 223 tests pass. Actual: 220 pass and three fail: bench::tests::release_thresholded_ratio_breach_is_blocking_per_operation expects Some(1.26) but gets Some(1.17); html::tests::release_thresholded_benchmarks_render_claim_and_threshold_rows and report::tests::release_thresholded_bench_json_exposes_threshold_contract still expect the text 'initial calibration'. The focused command reproduced unchanged after cargo test --workspace --locked first exposed it.

## Fix

<unfixed — raised only>

## Notes
