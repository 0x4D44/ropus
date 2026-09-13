# ROP-BUG-CRUCIBLE-00006 — Full-test child processes can run forever

- **State:** Fixed
- **Priority:** Must
- **Severity:** High
- **Area:** full-test/subprocess-supervision
- **Raised:** 2026-08-14T15:50:22Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T145457Z-5e0d8f9b
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-CRUCIBLE-00006-run-verify-20260913T145457Z-5e0d8f9b
- **Owner base:** 4a88ca71a80eaac314bab60dc1736f2ce021ad79
- **Owner fingerprint:** sha256:88aad1f7371d5784a019985455d10cca34bda6f5cb7da3bb6b9f0ce0705dbdb6
- **Owner since:** 2026-09-13T14:54:57Z
- **Owner until:** 2026-09-13T16:54:57Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-14T15:50:22Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-09-12T21:06:54Z, deltic:auto role=fix run=fix-20260912T205144Z-5d46f11b branch=task/bug-ROP-BUG-CRUCIBLE-00006-run-fix-20260912T205144Z-5d46f11b code=eed31d60f719892a6adaefa5cd65493bd559cd77 gate=manual)

## Observation

Static review at origin/main bb54eb50. C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\process_capture.rs:21-58 caps retained stdout and stderr but waits for the child without any deadline, process-tree termination, or descendant cleanup. Every Cargo, benchmark, fuzz, corpus, and platform command using this helper can therefore wedge the entire overnight run if the child hangs or a descendant keeps an inherited pipe open. Expected: every external stage has a bounded lifetime and reaps its process tree. Actual: output size is bounded but process lifetime is not. Fix: add a timeout-aware, cross-platform supervisor that kills and reaps descendants, returns a distinct timeout result, and has a hanging-child regression oracle. Static review only; no app, child command, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
