# ROP-BUG-KIL-00055 — ropusdec CLI tests can wait forever on child processes

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropusdec/tests
- **Raised:** 2026-08-22T09:55:56Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T153239Z-3ad69c09
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KIL-00055-run-verify-20260913T153239Z-3ad69c09
- **Owner base:** 1a4abfde85b8ffd3a015addc23fb9cfed01b650d
- **Owner fingerprint:** sha256:7c3f8452a8c8991b646b403908ffdce50af487fe06fe39a0d4c7bf631db56740
- **Owner since:** 2026-09-13T15:32:39Z
- **Owner until:** 2026-09-13T17:32:39Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T09:55:56Z, raised via `deltic bugs new`) -> Fixed (2026-09-13T07:47:37Z, deltic:auto role=fix run=fix-20260913T073655Z-b222ba5e branch=task/bug-ROP-BUG-KIL-00055-run-fix-20260913T073655Z-b222ba5e code=4234e4109e96571261674e6b315469868d9935ea gate=manual)

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
