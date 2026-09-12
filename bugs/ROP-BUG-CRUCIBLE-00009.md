# ROP-BUG-CRUCIBLE-00009 — Malformed benchmark output is treated as a successful default result

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** full-test/benchmark-reporting
- **Raised:** 2026-08-14T15:50:24Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260912T232023Z-645ea5f3
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-CRUCIBLE-00009-run-fix-20260912T232023Z-645ea5f3
- **Owner base:** 8f877a2f6540240caef8dc8f34f11267855949b7
- **Owner fingerprint:** -
- **Owner since:** 2026-09-12T23:20:23Z
- **Owner until:** 2026-09-13T01:20:23Z
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
