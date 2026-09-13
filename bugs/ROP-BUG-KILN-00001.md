# ROP-BUG-KILN-00001 — Autonomous fix scripts stage unrelated workspace files

- **State:** Closed
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
- **State history:** Open (2026-08-13T17:16:40Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-12T21:00:44Z, deltic:auto role=fix run=fix-20260912T204020Z-277ce191 branch=task/bug-ROP-BUG-KILN-00001-run-fix-20260912T204020Z-277ce191 code=6c6b9f7 gate=manual) -> Closed (2026-09-13T17:49:57Z, independent two-eyes verification model=codex@xhigh, verifier=CRUCIBLE, fixer=deltic:auto, fix=6c6b9f7)

## Observation

Observation: tools/bisect_fix.py and tools/trace_fix.py periodically run git add -A from the repository root, ignore add and commit status, and then log that a checkpoint was committed. In a shared or dirty worktree this captures unrelated source, generated files, or accidentally unignored secrets and can misreport a rejected commit as successful. Expected: autonomous checkpoints stage an explicit allowlist in a dedicated worktree and report the real command status. Actual: every changed and untracked path is staged indiscriminately.

## Fix

`tools/checkpoint.py` now requires a linked worktree, stages only tracked changes
under the explicit `ropus/src` allowlist, rejects pre-staged paths outside that
allowlist, and reports non-zero Git add/commit results. Both autonomous loops
refuse the primary checkout before building or invoking another agent.

Validation: `$null | python -m unittest tools.test_checkpoint -v` selected 4
tests and passed; both tool `--help` imports and `py_compile` also passed.

Fails-before proof: temporarily changing the checkpoint add command from
`git add --update -- <allowlist>` to `git add -A` made
`test_only_allowlisted_tracked_changes_are_committed` fail on its
`unrelated tracked work was staged` assertion. The allowlist change was restored
before commit `b38fc9c`, integrated as `6c6b9f7`.

### Independent verification summary (2026-09-13)

- Re-ran `test_only_allowlisted_tracked_changes_are_committed`; the checkpoint suite passed all 4 tests, with tool help imports and Python compilation also passing.
- A red control changed the checkpoint staging command to `git add -A`; the test detected unrelated tracked work, and the allowlist fix was restored.
## Notes
