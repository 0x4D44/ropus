# ROP-BUG-CRUCIBLE-00012 — JSON envelope omits the emit-json option

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** full-test/json-report
- **Raised:** 2026-08-14T15:50:27Z
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
- **State history:** Open (2026-08-14T15:50:27Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main bb54eb50. Options includes emit_json at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\cli.rs:19-27 and SetupInfo says its snapshot mirrors CLI flags at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\setup.rs:22-24. However setup_to_json omits emit_json at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\report.rs:50-66. Supervisors receiving the opt-in envelope cannot round-trip the invocation as promised. Fix: serialize flags.emit_json and add true and false report assertions. Static review only; no app, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
