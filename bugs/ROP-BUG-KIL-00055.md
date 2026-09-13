# ROP-BUG-KIL-00055 — ropusdec CLI tests can wait forever on child processes

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropusdec/tests
- **Raised:** 2026-08-22T09:55:56Z
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
- **State history:** Open (2026-08-22T09:55:56Z, raised via `deltic bugs new`) -> Fixed (2026-09-13T07:47:37Z, deltic:auto role=fix run=fix-20260913T073655Z-b222ba5e branch=task/bug-ROP-BUG-KIL-00055-run-fix-20260913T073655Z-b222ba5e code=4234e4109e96571261674e6b315469868d9935ea gate=manual) -> Closed (2026-09-13T16:47:37Z, independent two-eyes verification model=codex@xhigh, verifier=CRUCIBLE, fixer=deltic:auto, fix=4234e4109e96571261674e6b315469868d9935ea)

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

### Verification summary (2026-09-13)

- Re-ran `process_capture::tests::hanging_child_is_killed_and_reports_a_distinct_timeout`; it passed, and the full-test gate passed.
- A red control cleared the timeout flag; the test observed the hanging child completed unexpectedly and failed, so the fix was restored.

## Notes
