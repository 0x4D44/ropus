# ROP-BUG-CRUCIBLE-00009 — Malformed benchmark output is treated as a successful default result

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** full-test/benchmark-reporting
- **Raised:** 2026-08-14T15:50:24Z
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
- **State history:** Open (2026-08-14T15:50:24Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main bb54eb50. A zero-exit ropus-compare with malformed or partial stdout produces missing timings at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\bench.rs:870-875, but build_vector_row at :913-939 marks the row non-crashed and BenchResult::all_passed at :488-497 accepts it in the default profile. This contradicts the parser-drift contract at :903-907 and allows the normal report to show PASS instead of WARN. Release-thresholded mode separately rejects missing timings, so the release gate is not bypassed. Fix: treat any incomplete successful parse as a structured anomaly or crash in every profile and add an oracle connecting partial output to the default banner. Static review only; no benchmark, app, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
