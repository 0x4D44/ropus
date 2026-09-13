# ROP-BUG-CRUCIBLE-00014 — Fuzz inventory warnings count unchecked targets as passed in HTML

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** full-test/html-report
- **Raised:** 2026-08-14T15:50:29Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260913T041555Z-db56fe5c
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-CRUCIBLE-00014-run-fix-20260913T041555Z-db56fe5c
- **Owner base:** 94da478fa598473af57fef3627085e6dba5e923b
- **Owner fingerprint:** -
- **Owner since:** 2026-09-13T04:15:55Z
- **Owner until:** 2026-09-13T06:15:55Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-14T15:50:29Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main bb54eb50. Quick release preflight creates inventory target rows with build and replay states not_checked at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\fuzz.rs:63-74 and a Warn stage status. C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\html.rs:523-555 counts every Warn target as passed, so the primary phase summary can say all unchecked targets passed while also showing warn. Fix: count only explicit pass states, classify unchecked rows as skipped or unchecked, and add an InventoryOnly phase-summary assertion. Static review only; no report rendering, app, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
