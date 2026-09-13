# ROP-BUG-KILN-00001 — Autonomous fix scripts stage unrelated workspace files

- **State:** Fixed
- **Priority:** Must
- **Severity:** High
- **Area:** tools/automation-checkpoints
- **Raised:** 2026-08-13T17:16:40Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T144821Z-f4993ff9
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KILN-00001-run-verify-20260913T144821Z-f4993ff9
- **Owner base:** b216934d6b99df31bf707f6f7c0c66d22f1ff617
- **Owner fingerprint:** sha256:469e5650dca1fbaedfe5cb31fd58e9249a2741e6872ddf1d8b1e9b7f446d4aac
- **Owner since:** 2026-09-13T14:48:21Z
- **Owner until:** 2026-09-13T16:48:21Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-13T17:16:40Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-12T21:00:44Z, deltic:auto role=fix run=fix-20260912T204020Z-277ce191 branch=task/bug-ROP-BUG-KILN-00001-run-fix-20260912T204020Z-277ce191 code=6c6b9f7 gate=manual)

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

## Notes
