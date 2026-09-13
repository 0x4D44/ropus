# ROP-BUG-KIL-00055 — ropusdec CLI tests can wait forever on child processes

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropusdec/tests
- **Raised:** 2026-08-22T09:55:56Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260913T073655Z-b222ba5e
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KIL-00055-run-fix-20260913T073655Z-b222ba5e
- **Owner base:** 3d86b003265ff4f93e04c5c2f6d2486e04d50762
- **Owner fingerprint:** -
- **Owner since:** 2026-09-13T07:36:55Z
- **Owner until:** 2026-09-13T09:36:55Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T09:55:56Z, raised via `deltic bugs new`)

## Observation

Static review at `1c337e8751383e5e3a60009ce73e23283571edf0`.
`ropusdec/tests/cli.rs:132-148,181-196,225-238,265-283,314-334,359-368`
starts the decoder and waits through `wait_with_output()` or `output()` with no deadline,
termination path, or explicit reap after timeout. A decoder regression that waits for input,
deadlocks, or otherwise stops making progress can therefore wedge `cargo test -p ropusdec`
indefinitely instead of returning a bounded diagnostic failure. Existing
`ROP-BUG-CRUCIBLE-00006` covers the `full-test` runner and `ROP-BUG-KILN-00018` covers
`harness-control`; neither owns this crate. Expected: every child test has a generous finite
deadline and always terminates and reaps the child. Fix: centralize process setup and capture in
a timeout-aware helper, kill and reap on expiry, and add a deliberately hanging helper-child
oracle that proves the timeout path. Static source inspection only; no process, binary, or test
ran.

## Fix

<unfixed — raised only>

## Notes
