# ROP-BUG-KILN-00004 — Legacy integration tools pass when required fixtures are absent

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** tools/test-integrity
- **Raised:** 2026-08-13T17:17:35Z
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
- **State history:** Open (2026-08-13T17:17:35Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-12T21:43:11Z, deltic:auto role=fix run=fix-20260912T213422Z-1ed60022 branch=task/bug-ROP-BUG-KILN-00004-run-fix-20260912T213422Z-1ed60022 code=e7243c2ef7f50c475cb91e604aa38aac5e6722f2 gate=manual) -> Closed (2026-09-13T17:49:57Z, independent two-eyes verification model=codex@xhigh, verifier=CRUCIBLE, fixer=deltic:auto, fix=e7243c2ef7f50c475cb91e604aa38aac5e6722f2)

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

### Independent verification summary (2026-09-13)

- Re-ran the missing-fixture, empty-result, and failure-propagation regressions; `tools.test_integrity` and checkpoint tests passed all 15 tests.
- A red control treated a missing fixture as indeterminate; `test_missing_wav_is_a_failure` failed, and the fix was restored.
## Notes
