# ROP-BUG-CRUCIBLE-00009 — Malformed benchmark output is treated as a successful default result

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** full-test/benchmark-reporting
- **Raised:** 2026-08-14T15:50:24Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T145838Z-c592c114
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-CRUCIBLE-00009-run-verify-20260913T145838Z-c592c114
- **Owner base:** de22b3d3f19f7c70d45748e863394bca4dc92498
- **Owner fingerprint:** sha256:1948cb78448e2da559a8d54f2cdfe0d1dab7d7874ced79c928828c47d483383e
- **Owner since:** 2026-09-13T14:58:38Z
- **Owner until:** 2026-09-13T16:58:38Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-14T15:50:24Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-09-12T23:28:08Z, deltic:auto role=fix run=fix-20260912T232023Z-645ea5f3 branch=task/bug-ROP-BUG-CRUCIBLE-00009-run-fix-20260912T232023Z-645ea5f3 code=954c6d6979d4d830cecead7357721b351d317f69 gate=manual)

## Observation

Static review at origin/main bb54eb50. A zero-exit ropus-compare with malformed or partial stdout produces missing timings at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\bench.rs:870-875, but build_vector_row at :913-939 marks the row non-crashed and BenchResult::all_passed at :488-497 accepts it in the default profile. This contradicts the parser-drift contract at :903-907 and allows the normal report to show PASS instead of WARN. Release-thresholded mode separately rejects missing timings, so the release gate is not bypassed. Fix: treat any incomplete successful parse as a structured anomaly or crash in every profile and add an oracle connecting partial output to the default banner. Static review only; no benchmark, app, test, or harness ran.

## Fix

Implemented in `full-test/src/bench.rs` and integrated at code commit
`954c6d6979d4d830cecead7357721b351d317f69`.

- `BenchTimings::has_complete_finite_values` requires all four parsed timing
  values to be present and finite.
- `build_vector_row` now turns an incomplete successful parse into a structured
  crash anomaly, preserving any partial timings for diagnostics. This makes
  `BenchResult::all_passed` reject the false-green default result while keeping
  missing fixtures as informational skips.
- Added coverage for the default observed profile and release-thresholded
  reporting.

Verification:

- `$null | deltic timeout 180 cargo test -p full-test bench::tests` — 25 passed,
  0 failed.
- `deltic timeout 120 cargo fmt --all -- --check` — passed.
- `git diff --check` — passed.
- Red proof: temporarily disabled the new classifier guard; the regression test
  failed with `partial successful output must be anomalous`. The guard was then
  restored and the focused suite passed.

## Notes
