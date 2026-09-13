# ROP-BUG-KILN-00005 — Integration fix loop persists stale pre-fix results

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** tools/integration-results
- **Raised:** 2026-08-13T17:17:36Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T171101Z-61bee2ad
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KILN-00005-run-verify-20260913T171101Z-61bee2ad
- **Owner base:** b584433645a0a1d493670a68f877a9ebd18a4211
- **Owner fingerprint:** sha256:945657d10e7a0dbc48217d23280bfd912bc9b1d59af83b6f5caf61e2a7660042
- **Owner since:** 2026-09-13T17:11:01Z
- **Owner until:** 2026-09-13T19:11:01Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-13T17:17:36Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-12T21:49:12Z, deltic:auto role=fix run=fix-20260912T214538Z-0291dd54 branch=task/bug-ROP-BUG-KILN-00005-run-fix-20260912T214538Z-0291dd54 code=c1ed1b7016405e6261d08d8a189f0e60aae808fe gate=manual)

## Observation

Observation: tools/integrate.py rebinds results locally after each fix-loop retest and returns only a boolean. On success, cmd_run saves its original pre-fix list, so integration_results.json and the status command continue to report failures after the run announced success. Expected: the final verified result set is saved. Actual: persisted status describes the initial failing run.

## Fix

Integrated code commit `c1ed1b7016405e6261d08d8a189f0e60aae808fe` now replaces
the caller-owned result list in place after every fix-loop retest. `cmd_run`
therefore saves the verified post-fix results instead of its original failing
list. Regression coverage is in `tools/test_integrity.py`.

Validation evidence:

- Before the fix, an isolated mocked retest returned success while the caller
  still held `[{'passed': False, ...}]` instead of the verified passing result.
- Regression proof: mutating the in-place update back to a local rebinding made
  `$null | python -m unittest -v
  tools.test_integrity.IntegrateIntegrityTests.test_fix_loop_updates_caller_results_after_retest`
  fail with exit code 1.
- After restoration, `$null | deltic timeout 120 python -m unittest -v
  tools.test_integrity tools.test_checkpoint` passed all 16 tests.
- `$null | deltic timeout 120 python -m py_compile tools/integrate.py
  tools/bisect_fix.py tools/test_integrity.py` passed.

## Notes
