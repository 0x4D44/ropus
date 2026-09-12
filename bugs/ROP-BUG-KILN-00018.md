# ROP-BUG-KILN-00018 — Control decoder subprocesses have no execution deadline

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness-control/subprocess-supervision
- **Raised:** 2026-08-16T07:50:04Z
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
- **State history:** Open (2026-08-16T07:50:04Z, raised via `deltic bugs new`) -> Fixed (2026-09-12T23:56:44Z, deltic:auto role=fix run=fix-20260912T235011Z-6ec2c4c6 branch=task/bug-ROP-BUG-KILN-00018-run-fix-20260912T235011Z-6ec2c4c6 code=89ff8918d856a2812318da2d1a9e3cada7074217 gate=manual)

## Observation

Static review at origin/main a97b6f11. harness-control/tests/control_snr.rs:245-260 runs nested Cargo and decoder processes with Command::output, which has no deadline, polling, termination, or descendant cleanup. A Cargo lock, build, or C decoder hang can therefore wedge a direct control test indefinitely instead of returning a bounded failure; the component gate at .deltic-integrate.toml:101-108 invokes this test package. Expected: every external control stage has a generous but finite lifetime and reaps its process tree. Actual: process lifetime is unbounded. Fix: use a cross-platform timeout-aware runner that terminates and reaps the Cargo/decoder tree, reports timeout distinctly, and has a hanging-child regression oracle. This is distinct from ROP-BUG-CRUCIBLE-00006, which covers full-test/src/process_capture.rs and the full-test runner only. Static inspection only; no subprocess, app, build, test, decoder, or harness ran.

## Fix

Implemented in `harness-control/tests/control_snr.rs` and integrated at code
commit `89ff8918d856a2812318da2d1a9e3cada7074217`.

- Replaced unbounded `Command::output` calls with a timeout-aware runner that
  drains both output pipes, polls a 15-minute deadline, kills the full process
  tree (`kill` process groups on Unix and `taskkill /T` on Windows), and reaps
  the direct child.
- Decoder failures now distinguish timeout diagnostics from non-zero exits.
- Added a hanging-child regression test that completes in under one second.

Verification:

- `$null | deltic timeout 180 cargo test -p ropus-harness-control --test control_snr hanging_control_child_is_killed_and_reports_a_distinct_timeout` — 1 passed, 0 failed.
- `$null | deltic timeout 180 cargo test -p ropus-harness-control --test control_snr control_temp_dirs_are_unique_and_cleaned_up` — 1 passed, 0 failed.
- `$null | deltic timeout 180 cargo test -p ropus-harness-control --test control_snr loss_pattern_contains_only_complete_recovery_cycles` — 1 passed, 0 failed.
- `deltic timeout 120 cargo check -p ropus-harness-control` — passed.
- `deltic timeout 120 cargo fmt --all -- --check` — passed.
- Red proof: temporarily changed the runner to ignore its caller timeout; the
  hanging-child test took 3.2 seconds and failed its one-second bound. The
  caller-supplied deadline was restored and the focused tests passed.

The full decoder SNR tests were not run; this checkout lacks the optional DNN
weights and those tests are expected to fail for that environment reason.

## Notes
