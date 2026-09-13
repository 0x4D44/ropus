# ROP-BUG-CRUCIBLE-00014 — Fuzz inventory warnings count unchecked targets as passed in HTML

- **State:** Fixed
- **Priority:** Could
- **Severity:** Low
- **Area:** full-test/html-report
- **Raised:** 2026-08-14T15:50:29Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T150455Z-c1b3cef6
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-CRUCIBLE-00014-run-verify-20260913T150455Z-c1b3cef6
- **Owner base:** d827d9fa301b653ecf742f1b3f3a294e83b3fc6f
- **Owner fingerprint:** sha256:f59c5d41756637e4f3e819ac06b5ded74b5dae7a1c997381b1af60600379c727
- **Owner since:** 2026-09-13T15:04:55Z
- **Owner until:** 2026-09-13T17:04:55Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-14T15:50:29Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-09-13T04:23:34Z, deltic:auto role=fix run=fix-20260913T041555Z-db56fe5c branch=task/bug-ROP-BUG-CRUCIBLE-00014-run-fix-20260913T041555Z-db56fe5c code=f2f14943deefd32b595c7c50b77a8aa47bcaab75 gate=manual)

## Observation

Static review at origin/main bb54eb50. Quick release preflight creates inventory target rows with build and replay states not_checked at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\fuzz.rs:63-74 and a Warn stage status. C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\html.rs:523-555 counts every Warn target as passed, so the primary phase summary can say all unchecked targets passed while also showing warn. Fix: count only explicit pass states, classify unchecked rows as skipped or unchecked, and add an InventoryOnly phase-summary assertion. Static review only; no report rendering, app, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
