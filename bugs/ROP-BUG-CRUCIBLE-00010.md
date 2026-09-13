# ROP-BUG-CRUCIBLE-00010 — Benchmark runner ignores custom Cargo target directories

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** full-test/benchmark-launch
- **Raised:** 2026-08-14T15:50:25Z
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
- **State history:** Open (2026-08-14T15:50:25Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-09-13T03:55:37Z, deltic:auto role=fix run=fix-20260913T034834Z-f0420b73 branch=task/bug-ROP-BUG-CRUCIBLE-00010-run-fix-20260913T034834Z-f0420b73 code=77730451a2c7b312ea1582ed4fbc84eb823958f5 gate=manual) -> Closed (2026-09-13T17:49:57Z, independent two-eyes verification model=codex@xhigh, verifier=CRUCIBLE, fixer=deltic:auto, fix=77730451a2c7b312ea1582ed4fbc84eb823958f5)

## Observation

Static review at origin/main bb54eb50. C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\bench.rs:804-812 builds ropus-compare through Cargo, which honors CARGO_TARGET_DIR, but :835-842 always looks under workspace-root\target\release. With a standard custom target directory, the build succeeds and :645-650 then reports the binary missing. Expected: the runner invokes the artifact Cargo just built. Fix: resolve CARGO_TARGET_DIR consistently or set one explicit target directory for both build and lookup, then add a custom-target-path unit oracle. The active review environment uses the default workspace target, so runtime reproduction was not attempted. No app, test, or harness ran.

## Fix

### Independent verification summary (2026-09-13)

- Re-ran `custom_target_dir_is_resolved_from_workspace_root`; the full-test package gate passed all 244 tests.
- A red control ignored the configured Cargo target directory; the custom-target path assertion failed, and the fix was restored.

## Notes
