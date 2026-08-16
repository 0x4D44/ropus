# ROP-BUG-KILN-00018 — Control decoder subprocesses have no execution deadline

- **State:** Open
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
- **State history:** Open (2026-08-16T07:50:04Z, raised via `deltic bugs new`)

## Observation

Static review at origin/main a97b6f11. harness-control/tests/control_snr.rs:245-260 runs nested Cargo and decoder processes with Command::output, which has no deadline, polling, termination, or descendant cleanup. A Cargo lock, build, or C decoder hang can therefore wedge a direct control test indefinitely instead of returning a bounded failure; the component gate at .deltic-integrate.toml:101-108 invokes this test package. Expected: every external control stage has a generous but finite lifetime and reaps its process tree. Actual: process lifetime is unbounded. Fix: use a cross-platform timeout-aware runner that terminates and reaps the Cargo/decoder tree, reports timeout distinctly, and has a hanging-child regression oracle. This is distinct from ROP-BUG-CRUCIBLE-00006, which covers full-test/src/process_capture.rs and the full-test runner only. Static inspection only; no subprocess, app, build, test, decoder, or harness ran.

## Fix

<unfixed — raised only>

## Notes
