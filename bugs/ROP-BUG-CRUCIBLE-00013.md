# ROP-BUG-CRUCIBLE-00013 — ANSI-colored Cargo diagnostics disappear from full-test reports

- **State:** Fixed
- **Priority:** Could
- **Severity:** Low
- **Area:** full-test/diagnostics
- **Raised:** 2026-08-14T15:50:28Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T150341Z-f7799433
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-CRUCIBLE-00013-run-verify-20260913T150341Z-f7799433
- **Owner base:** 9a22fe6006e4a6712fff97903b59a090605b1292
- **Owner fingerprint:** sha256:ae87349ea3f2904362e8bd2a1a81ac72d3b4f817276ab5c94555b95677a0f02d
- **Owner since:** 2026-09-13T15:03:41Z
- **Owner until:** 2026-09-13T17:03:41Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-14T15:50:28Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-09-13T04:14:38Z, deltic:auto role=fix run=fix-20260913T040222Z-7a6d28d7 branch=task/bug-ROP-BUG-CRUCIBLE-00013-run-fix-20260913T040222Z-7a6d28d7 code=5f696be1c74754fc5f36f3cf384465aed2102a3d gate=manual)

## Observation

Static review at origin/main bb54eb50. C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\issues.rs:21-38 says ANSI escapes may remain, but its regex at :60-65 only matches lines beginning directly with error or warning. C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\quality.rs:58-68 and :116-131 do not force plain Cargo output before extraction. With CARGO_TERM_COLOR=always, the stage still fails but its actionable issue list can be empty. Fix: strip ANSI for classification while retaining desired display text, or force plain output consistently; add colored error and warning fixtures. Static review only; no Cargo command, app, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
