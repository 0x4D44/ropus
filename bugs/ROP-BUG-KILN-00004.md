# ROP-BUG-KILN-00004 — Legacy integration tools pass when required fixtures are absent

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** tools/test-integrity
- **Raised:** 2026-08-13T17:17:35Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T170717Z-4caafa1d
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KILN-00004-run-verify-20260913T170717Z-4caafa1d
- **Owner base:** fc3d6c376d280c7e6f85fc711d439280499c419e
- **Owner fingerprint:** sha256:9106168444dd5d67dd3ef829497b9dfdd3feacf5f04ce30fb5b0245ad4405888
- **Owner since:** 2026-09-13T17:07:17Z
- **Owner until:** 2026-09-13T19:07:17Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-13T17:17:35Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-12T21:43:11Z, deltic:auto role=fix run=fix-20260912T213422Z-1ed60022 branch=task/bug-ROP-BUG-KILN-00004-run-fix-20260912T213422Z-1ed60022 code=e7243c2ef7f50c475cb91e604aa38aac5e6722f2 gate=manual)

## Observation

Observation: tools/integrate.py skips missing configured WAV files and returns success when its results list has no failures, including an empty list. tools/bisect_fix.py similarly maps missing WAV files to an indeterminate value, excludes them from its failure count, and can declare all zero tests passing; its scan and test commands also return zero after observed failures. Expected: missing required fixtures or a failing comparator produce a nonzero result. Actual: fresh or incomplete corpora can yield false-green automation.

## Fix

Integrated code commit `e7243c2ef7f50c475cb91e604aa38aac5e6722f2` now treats
missing configured WAV fixtures and empty comparison results as failures. The
legacy integration and bisect commands propagate comparator failures with a
nonzero exit status, and their fixer loops stop without invoking a codec fixer
when a required fixture is absent. Focused coverage lives in
`tools/test_integrity.py`.

Validation evidence:

- Before the fix, an isolated missing-corpus run returned
  `integrate.run_all_tests(...) == []`; `bisect_fix.scan_all(...)` reported a
  missing fixture with `passed=None` and zero counted failures.
- Regression proof: mutating the missing-fixture result back to `passed=None`
  made `$null | python -m unittest -v
  tools.test_integrity.BisectIntegrityTests.test_missing_wav_is_a_failure`
  fail with exit code 1.
- After restoration, `$null | deltic timeout 120 python -m unittest -v
  tools.test_integrity tools.test_checkpoint` passed all 15 tests.
- `$null | deltic timeout 120 python -m py_compile tools/integrate.py
  tools/bisect_fix.py tools/test_integrity.py` passed.

## Notes
