# ROP-BUG-FLUX-00002 — Full-test benchmark threshold assertions are stale

- **State:** Closed
- **Priority:** Must
- **Severity:** Medium
- **Area:** full-test/benchmarks
- **Raised:** 2026-07-30
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
- **State history:** Open (2026-07-30, raised via `deltic bugs new`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T205505Z-p44875-n625225000-c1 branch=task/bug-ROP-BUG-FLUX-00002-run-fix-20260730T205505Z-p44875-n625225000-c1 code=6f8c777 gate=manual) -> Closed (2026-07-31, independent two-eyes verification on host KILN, model=claude, at origin/main 6ccb736; fails-before reproduced at 6f8c777^, passes-after on trunk)

## Observation

Observed on macOS in a clean task worktree at origin/main c2658d87e44886ce4210be3fae798fe6913cd40e. Reproduce: cargo test -p full-test --locked. Expected: all 223 tests pass. Actual: 220 pass and three fail: bench::tests::release_thresholded_ratio_breach_is_blocking_per_operation expects Some(1.26) but gets Some(1.17); html::tests::release_thresholded_benchmarks_render_claim_and_threshold_rows and report::tests::release_thresholded_bench_json_exposes_threshold_contract still expect the text 'initial calibration'. The focused command reproduced unchanged after cargo test --workspace --locked first exposed it.

## Fix

Commit `6f8c777` aligned three stale test expectations in `full-test` with the production
threshold table: `enc_release_fail_ratio` 1.26 -> 1.17 in `full-test/src/bench.rs`, and the
threshold-source text `initial calibration` -> `recalibrated 2026-05-11` in
`full-test/src/html.rs` and `full-test/src/report.rs`.

### Verification summary (2026-07-31, independent two-eyes, host KILN / Windows x86_64)

Verified at `origin/main` 6ccb736 by an actor other than the fixer.

- **Fails before.** Detached worktree at `6f8c777^` (4ced261), `cargo test -p full-test
  --locked -- release_thresholded`: exactly the three recorded tests fail, with the recorded
  messages — `assertion left == right failed, left: Some(1.17), right: Some(1.26)` at
  `bench.rs:1339`, and the two `initial calibration` assertions at `report.rs:909` and
  `html.rs:1753`. This matches the original observation verbatim.
- **Passes after.** On 6ccb736 the same filter runs 12 tests, 0 failed, including all three;
  the whole suite `cargo test -p full-test --locked` is green (220 passed, 0 failed). The
  total differs from the 223 in the observation because the suite itself has changed since.
- **Root cause addressed, not papered over.** The production table was deliberately
  recalibrated in commit `6267221` ("perf: recalibrate README + release thresholds against
  fixed bench"), which moved row 0 `enc_release_fail_ratio` to 1.17 and rewrote
  `X86_64_THRESHOLD_SOURCE`, but left these three assertions behind. The tests, not the
  thresholds, were wrong; a sibling test already asserted 1.17. The changed assertions read
  the same production constants they are meant to gate, so the fix restores the contract
  rather than weakening it.
- **Still correct on Apple Silicon.** Later commits added `APPLE_AARCH64_THRESHOLDS` and made
  `release_thresholded_ratio_breach_is_blocking_per_operation` arch-aware
  (`bench.rs:1466`: 1.20 on aarch64-macos, 1.17 elsewhere), so the hardcoded 1.17 introduced
  by this fix is no longer arch-fragile on trunk.

## Notes

### Unrelated gate failure seen during verification (2026-07-31, host KILN)

`cargo clippy -p full-test --all-targets --locked -- -D warnings` fails on Windows:
`unpinned_manifest` (`full-test/src/corpus.rs:1743`) and `sha256_hex`
(`full-test/src/corpus.rs:1763`) are only used by `#[cfg(unix)]` tests, so they are dead code
on Windows and `-D warnings` rejects them. This is **not** caused by this bug's fix — it
reproduces identically at `6f8c777^`, and the fix touched only `bench.rs`, `html.rs`, and
`report.rs`. Recorded here so the next reader does not mistake it for a regression; it needs
its own ledger entry.
