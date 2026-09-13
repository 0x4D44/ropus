# ROP-BUG-KILN-00008 — Fuzz runner reports all clear after fuzzer failure

- **State:** Fixed
- **Priority:** Should
- **Severity:** High
- **Area:** tools/fuzz-runner
- **Raised:** 2026-08-13T17:17:37Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T145805Z-f2ae4044
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KILN-00008-run-verify-20260913T145805Z-f2ae4044
- **Owner base:** f0c2c4a5250eed228928d61d2ca6e0ca0420f9a7
- **Owner fingerprint:** sha256:235cf2e36226f66859b2f52d998843d95fbd20ca3e4d35adfcf7d8548f8c1e62
- **Owner since:** 2026-09-13T14:58:05Z
- **Owner until:** 2026-09-13T16:58:05Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-13T17:17:37Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-12T22:17:41Z, deltic:auto role=fix run=fix-20260912T220200Z-dc7a268b branch=task/bug-ROP-BUG-KILN-00008-run-fix-20260912T220200Z-dc7a268b code=75ae57c27f69e412188a328c19f56a4962a229a8 gate=manual)

## Observation

Observation: tools/fuzz_run.sh captures every cargo-fuzz exit status but the final exit decision checks only whether artifact files were found. A startup, sanitizer, invalid-option, signal, or runtime failure that creates no artifact is summarized with its nonzero status and then reported as All clear with exit zero. The documented no-diff option also only sets an unused variable. Expected: any child failure is non-green and no-diff changes target behavior or is rejected. Actual: failed or misconfigured campaigns can pass.

## Fix

Integrated code commit `75ae57c27f69e412188a328c19f56a4962a229a8` now counts
nonzero cargo-fuzz target exits separately from findings and fails the campaign
when either condition occurs. The shell and Windows PowerShell runners reject
the documented but unsupported `--no-diff` option instead of silently ignoring it.
Focused coverage lives in `tools/test_fuzz_run.py`.

Validation evidence:

- Before the fix, the isolated fake-cargo harness ran four tests and failed its
  child-failure assertion (the runner returned zero for exit 7 with no artifact)
  and its `--no-diff` assertion (the option was accepted and cargo started).
- Regression proof: restoring the artifact-only final decision made
  `test_child_failure_without_artifact_is_nonzero` fail; restoring the ignored
  `--no-diff` parser made `test_no_diff_is_rejected_before_starting_cargo` fail.
- After the fix, `python -m unittest -v tools.test_fuzz_run` passed all four tests.
- A normalized `bash -n` parse passed, and the PowerShell runner rejected
  `--no-diff` with exit status 1 before starting cargo.

## Notes
