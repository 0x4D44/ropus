# ROP-BUG-CRUCIBLE-00014 — Fuzz inventory warnings count unchecked targets as passed in HTML

- **State:** Closed
- **Priority:** Could
- **Severity:** Low
- **Area:** full-test/html-report
- **Raised:** 2026-08-14T15:50:29Z
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
- **State history:** Open (2026-08-14T15:50:29Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-09-13T04:23:34Z, deltic:auto role=fix run=fix-20260913T041555Z-db56fe5c branch=task/bug-ROP-BUG-CRUCIBLE-00014-run-fix-20260913T041555Z-db56fe5c code=f2f14943deefd32b595c7c50b77a8aa47bcaab75 gate=manual) -> Closed (2026-09-13T17:49:57Z, independent two-eyes verification model=codex@xhigh, verifier=CRUCIBLE, fixer=deltic:auto, fix=f2f14943deefd32b595c7c50b77a8aa47bcaab75)

## Observation

Static review at origin/main bb54eb50. Quick release preflight creates inventory target rows with build and replay states not_checked at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\fuzz.rs:63-74 and a Warn stage status. C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\html.rs:523-555 counts every Warn target as passed, so the primary phase summary can say all unchecked targets passed while also showing warn. Fix: count only explicit pass states, classify unchecked rows as skipped or unchecked, and add an InventoryOnly phase-summary assertion. Static review only; no report rendering, app, test, or harness ran.

## Fix

### Independent verification summary (2026-09-13)

- Re-ran `phase_summary_marks_unchecked_inventory_targets_skipped`; the full-test package gate passed all 244 tests.
- A red control counted unchecked inventory warnings as passed; the phase-summary assertion failed, and the fix was restored.

## Notes
