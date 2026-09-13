# ROP-BUG-CRUCIBLE-00012 — JSON envelope omits the emit-json option

- **State:** Fixed
- **Priority:** Could
- **Severity:** Low
- **Area:** full-test/json-report
- **Raised:** 2026-08-14T15:50:27Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T150234Z-beb4cb70
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-CRUCIBLE-00012-run-verify-20260913T150234Z-beb4cb70
- **Owner base:** 94bd198b4efcc162eba491ba444b50da78e9a4a5
- **Owner fingerprint:** sha256:809947e7df14cf8d9a8ef5127c8ec37157901ab62d7dc47685e907cde8c133a8
- **Owner since:** 2026-09-13T15:02:34Z
- **Owner until:** 2026-09-13T17:02:34Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-14T15:50:27Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-09-13T04:01:36Z, deltic:auto role=fix run=fix-20260913T035627Z-0a069269 branch=task/bug-ROP-BUG-CRUCIBLE-00012-run-fix-20260913T035627Z-0a069269 code=cf191aa7138723d0cdafe94b890c7e6cd4ca882f gate=manual)

## Observation

Static review at origin/main bb54eb50. Options includes emit_json at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\cli.rs:19-27 and SetupInfo says its snapshot mirrors CLI flags at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\setup.rs:22-24. However setup_to_json omits emit_json at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\report.rs:50-66. Supervisors receiving the opt-in envelope cannot round-trip the invocation as promised. Fix: serialize flags.emit_json and add true and false report assertions. Static review only; no app, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
