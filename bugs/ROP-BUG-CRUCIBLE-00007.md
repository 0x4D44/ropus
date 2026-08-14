# ROP-BUG-CRUCIBLE-00007 — Release preflight launches Unix-only wrappers on Windows

- **State:** Open
- **Priority:** Must
- **Severity:** High
- **Area:** full-test/windows-preflight
- **Raised:** 2026-08-14T15:50:23Z
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
- **State history:** Open (2026-08-14T15:50:23Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main bb54eb50 on the primary Windows platform. C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\fuzz.rs:259-266 launches timeout then bash, while C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\platform.rs:420-434 launches timeout then env. Native Windows has timeout.exe as an interactive delay utility and no env executable in this review environment, so the required non-quick release-preflight lanes cannot launch correctly on the platform promised by C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\wrk_docs\2026.04.19 - HLD - full-test-runner.md:33-34. Expected: release preflight works on native Windows without GNU wrappers. Fix: set environment variables through Command, implement timeout supervision in Rust, choose the platform script explicitly, and add cfg(windows) command-shape tests. Static inspection and system command lookup only; no app, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
