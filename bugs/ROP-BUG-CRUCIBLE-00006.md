# ROP-BUG-CRUCIBLE-00006 — Full-test child processes can run forever

- **State:** Closed
- **Priority:** Must
- **Severity:** High
- **Area:** full-test/subprocess-supervision
- **Raised:** 2026-08-14T15:50:22Z
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
- **State history:** Open (2026-08-14T15:50:22Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-09-12T21:06:54Z, deltic:auto role=fix run=fix-20260912T205144Z-5d46f11b branch=task/bug-ROP-BUG-CRUCIBLE-00006-run-fix-20260912T205144Z-5d46f11b code=eed31d60f719892a6adaefa5cd65493bd559cd77 gate=manual) -> Closed (2026-09-13T17:49:57Z, independent two-eyes verification model=codex@xhigh, verifier=CRUCIBLE, fixer=deltic:auto, fix=eed31d60f719892a6adaefa5cd65493bd559cd77)

## Observation

Static review at origin/main bb54eb50. C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\process_capture.rs:21-58 caps retained stdout and stderr but waits for the child without any deadline, process-tree termination, or descendant cleanup. Every Cargo, benchmark, fuzz, corpus, and platform command using this helper can therefore wedge the entire overnight run if the child hangs or a descendant keeps an inherited pipe open. Expected: every external stage has a bounded lifetime and reaps its process tree. Actual: output size is bounded but process lifetime is not. Fix: add a timeout-aware, cross-platform supervisor that kills and reaps descendants, returns a distinct timeout result, and has a hanging-child regression oracle. Static review only; no app, child command, test, or harness ran.

## Fix

### Independent verification summary (2026-09-13)

- Re-ran `hanging_child_is_killed_and_reports_a_distinct_timeout`; the full-test package gate passed all 244 tests.
- A red control cleared the timeout classification flag; the hanging-child regression failed, and the fix was restored.

## Notes
