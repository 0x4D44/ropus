# ROP-BUG-CRUCIBLE-00015 — Ambisonics parse failures render as zero failures in HTML

- **State:** Fixed
- **Priority:** Could
- **Severity:** Low
- **Area:** full-test/html-report
- **Raised:** 2026-08-14T15:50:30Z
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
- **State history:** Open (2026-08-14T15:50:30Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-09-13T04:29:12Z, deltic:auto role=fix run=fix-20260913T042425Z-28ef7170 branch=task/bug-ROP-BUG-CRUCIBLE-00015-run-fix-20260913T042425Z-28ef7170 code=f1d39d9fecbcfd47d9afd9ac3e55801171a6d472 gate=manual)

## Observation

Static review at origin/main bb54eb50. A missing or malformed ambisonics summary yields overall_pass false and no order rows at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\ambisonics.rs:138-164, which correctly makes the overall banner fail through all_passed. The HTML phase row at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\html.rs:454-482 derives failures only from per_order and build_failed, so it displays total 0, passed 0, failed 0 with no fail class. Fix: render a parse or stage failure when overall_pass is false even with no rows, and add a no-summary AmbisonicsResult fixture. Static review only; no report rendering, app, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
