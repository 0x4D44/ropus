# ROP-BUG-CRUCIBLE-00006 — Full-test child processes can run forever

- **State:** Open
- **Priority:** Must
- **Severity:** High
- **Area:** full-test/subprocess-supervision
- **Raised:** 2026-08-14T15:50:22Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260912T205144Z-5d46f11b
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-CRUCIBLE-00006-run-fix-20260912T205144Z-5d46f11b
- **Owner base:** 53572f78dffca731fdebe6ea98d6b55cac6319e6
- **Owner fingerprint:** -
- **Owner since:** 2026-09-12T20:51:44Z
- **Owner until:** 2026-09-12T22:51:44Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-14T15:50:22Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main bb54eb50. C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\process_capture.rs:21-58 caps retained stdout and stderr but waits for the child without any deadline, process-tree termination, or descendant cleanup. Every Cargo, benchmark, fuzz, corpus, and platform command using this helper can therefore wedge the entire overnight run if the child hangs or a descendant keeps an inherited pipe open. Expected: every external stage has a bounded lifetime and reaps its process tree. Actual: output size is bounded but process lifetime is not. Fix: add a timeout-aware, cross-platform supervisor that kills and reaps descendants, returns a distinct timeout result, and has a hanging-child regression oracle. Static review only; no app, child command, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
