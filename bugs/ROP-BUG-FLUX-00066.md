# ROP-BUG-FLUX-00066 — Playback discards output and terminal restoration failures

- **State:** Fixed
- **Priority:** Should
- **Severity:** Low
- **Area:** ropusplay/terminal-io
- **Raised:** 2026-07-31
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-02, deltic:auto role=fix run=fix-20260802T011105Z-p7367-n149474000-c1 branch=task/bug-ROP-BUG-FLUX-00066-run-fix-20260802T011105Z-p7367-n149474000-c1 code=465a04a gate=manual)

## Observation

Static review at origin/main e5d7113. /Users/md/language/ropus/ropus-tools-core/src/commands/play.rs:100,129,138,299,376,384,414,419,424,434 discards every stdout flush result, including the script-consumed --list-devices block, so incomplete delivery can still exit zero. RawModeGuard::drop at :54-57 also discards disable_raw_mode failure; a normal playback exit can report success while leaving the terminal in raw mode without a diagnostic. Fix: route output through a fallible writer, propagate write/flush failures, explicitly restore raw mode on normal and error exits while retaining Drop as unwind fallback, and preserve both errors if playback and restoration fail. Add injected failing-writer and terminal-restore oracles. Static review only; no terminal, player, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
