# ROP-BUG-KILN-00001 — Autonomous fix scripts stage unrelated workspace files

- **State:** Open
- **Priority:** Must
- **Severity:** High
- **Area:** tools/automation-checkpoints
- **Raised:** 2026-08-13T17:16:40Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260912T204020Z-277ce191
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KILN-00001-run-fix-20260912T204020Z-277ce191
- **Owner base:** ea4c682e41817fe4d1a4cc2abb0d4a862fb5cc8a
- **Owner fingerprint:** -
- **Owner since:** 2026-09-12T20:40:20Z
- **Owner until:** 2026-09-12T22:40:20Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-13T17:16:40Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Observation: tools/bisect_fix.py and tools/trace_fix.py periodically run git add -A from the repository root, ignore add and commit status, and then log that a checkpoint was committed. In a shared or dirty worktree this captures unrelated source, generated files, or accidentally unignored secrets and can misreport a rejected commit as successful. Expected: autonomous checkpoints stage an explicit allowlist in a dedicated worktree and report the real command status. Actual: every changed and untracked path is staged indiscriminately.

## Fix

<unfixed — raised only>

## Notes
