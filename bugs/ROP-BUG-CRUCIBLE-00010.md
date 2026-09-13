# ROP-BUG-CRUCIBLE-00010 — Benchmark runner ignores custom Cargo target directories

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** full-test/benchmark-launch
- **Raised:** 2026-08-14T15:50:25Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260913T034834Z-f0420b73
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-CRUCIBLE-00010-run-fix-20260913T034834Z-f0420b73
- **Owner base:** 6e595598d641eba5370e61616a81bb9871f2b9ff
- **Owner fingerprint:** -
- **Owner since:** 2026-09-13T03:48:34Z
- **Owner until:** 2026-09-13T05:48:34Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-14T15:50:25Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main bb54eb50. C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\bench.rs:804-812 builds ropus-compare through Cargo, which honors CARGO_TARGET_DIR, but :835-842 always looks under workspace-root\target\release. With a standard custom target directory, the build succeeds and :645-650 then reports the binary missing. Expected: the runner invokes the artifact Cargo just built. Fix: resolve CARGO_TARGET_DIR consistently or set one explicit target directory for both build and lookup, then add a custom-target-path unit oracle. The active review environment uses the default workspace target, so runtime reproduction was not attempted. No app, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
