# ROP-BUG-KILN-00008 — Fuzz runner reports all clear after fuzzer failure

- **State:** Closed
- **Priority:** Should
- **Severity:** High
- **Area:** tools/fuzz-runner
- **Raised:** 2026-08-13T17:17:37Z
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
- **State history:** Open (2026-08-13T17:17:37Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-12T22:17:41Z, deltic:auto role=fix run=fix-20260912T220200Z-dc7a268b branch=task/bug-ROP-BUG-KILN-00008-run-fix-20260912T220200Z-dc7a268b code=75ae57c27f69e412188a328c19f56a4962a229a8 gate=manual) -> Closed (2026-09-13T17:49:57Z, independent two-eyes verification model=codex@xhigh, verifier=CRUCIBLE, fixer=deltic:auto, fix=75ae57c27f69e412188a328c19f56a4962a229a8)

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

### Independent verification summary (2026-09-13)

- Re-ran `python -m unittest -v tools.test_fuzz_run`; all 4 tests passed.
- Red controls restored artifact-only success and ignored `--no-diff`; the child-failure and pre-start rejection tests failed, and both fixes were restored.
## Notes
