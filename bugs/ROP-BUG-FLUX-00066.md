# ROP-BUG-FLUX-00066 — Playback discards output and terminal restoration failures

- **State:** Closed
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-02, deltic:auto role=fix run=fix-20260802T011105Z-p7367-n149474000-c1 branch=task/bug-ROP-BUG-FLUX-00066-run-fix-20260802T011105Z-p7367-n149474000-c1 code=465a04a gate=manual) -> Closed (2026-08-02, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main 66f0954; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main e5d7113. /Users/md/language/ropus/ropus-tools-core/src/commands/play.rs:100,129,138,299,376,384,414,419,424,434 discards every stdout flush result, including the script-consumed --list-devices block, so incomplete delivery can still exit zero. RawModeGuard::drop at :54-57 also discards disable_raw_mode failure; a normal playback exit can report success while leaving the terminal in raw mode without a diagnostic. Fix: route output through a fallible writer, propagate write/flush failures, explicitly restore raw mode on normal and error exits while retaining Drop as unwind fallback, and preserve both errors if playback and restoration fail. Add injected failing-writer and terminal-restore oracles. Static review only; no terminal, player, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-02, independent two-eyes, host flux)

- Fix commit `465a04a` reworks `ropus-tools-core/src/commands/play.rs`'s `RawModeGuard`
  from a bare `Drop`-only impl into a generic type parameterized on an injectable
  `TerminalMode`, routes stdout through a fallible writer, and explicitly restores raw mode
  on normal and error exits (retaining `Drop` only as an unwind fallback), preserving both
  playback and restoration errors when both fail. New unit tests
  (`write_output_propagates_write_failures`, `write_output_propagates_flush_failures`,
  `raw_mode_restore_reports_failure_and_drop_retries`,
  `playback_and_restore_errors_are_both_preserved`) inject failing writers/terminal-mode
  stubs matching the observation.
- The fallible-writer seam and explicit-restore logic are both new in this commit (the
  pre-fix `play.rs` has a bare unit-struct `RawModeGuard` with no injectable mode and no
  fallible writer), so these tests cannot exist or pass without the fix — confirmed by
  reading `git diff 465a04a~1 465a04a -- ropus-tools-core/src/commands/play.rs`.
- `cargo test -p ropus-tools-core` (full crate, including `round_trip.rs`): 23 integration
  + 47 unit tests passed, 0 failed.
- `cargo clippy -p ropus-tools-core --all-targets --locked -- -D warnings`: clean.
