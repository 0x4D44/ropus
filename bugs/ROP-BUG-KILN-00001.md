# ROP-BUG-KILN-00001 — Autonomous fix scripts stage unrelated workspace files

- **State:** Open
- **Priority:** Must
- **Severity:** High
- **Area:** tools/automation-checkpoints
- **Raised:** 2026-08-13T17:16:40Z
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
- **State history:** Open (2026-08-13T17:16:40Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Observation: tools/bisect_fix.py and tools/trace_fix.py periodically run git add -A from the repository root, ignore add and commit status, and then log that a checkpoint was committed. In a shared or dirty worktree this captures unrelated source, generated files, or accidentally unignored secrets and can misreport a rejected commit as successful. Expected: autonomous checkpoints stage an explicit allowlist in a dedicated worktree and report the real command status. Actual: every changed and untracked path is staged indiscriminately.

## Fix

<unfixed — raised only>

## Notes
