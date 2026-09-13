# ROP-BUG-CRUCIBLE-00013 — ANSI-colored Cargo diagnostics disappear from full-test reports

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** full-test/diagnostics
- **Raised:** 2026-08-14T15:50:28Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260913T040222Z-7a6d28d7
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-CRUCIBLE-00013-run-fix-20260913T040222Z-7a6d28d7
- **Owner base:** a24b5428d6b49ae6e5102f71bd308d6b9f6c43cc
- **Owner fingerprint:** -
- **Owner since:** 2026-09-13T04:02:22Z
- **Owner until:** 2026-09-13T06:02:22Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-14T15:50:28Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main bb54eb50. C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\issues.rs:21-38 says ANSI escapes may remain, but its regex at :60-65 only matches lines beginning directly with error or warning. C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\quality.rs:58-68 and :116-131 do not force plain Cargo output before extraction. With CARGO_TERM_COLOR=always, the stage still fails but its actionable issue list can be empty. Fix: strip ANSI for classification while retaining desired display text, or force plain output consistently; add colored error and warning fixtures. Static review only; no Cargo command, app, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
