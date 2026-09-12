# ROP-BUG-KILN-00005 — Integration fix loop persists stale pre-fix results

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** tools/integration-results
- **Raised:** 2026-08-13T17:17:36Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260912T214538Z-0291dd54
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KILN-00005-run-fix-20260912T214538Z-0291dd54
- **Owner base:** 6af8e815dfc0ce43797a282dc0d09e61007a2018
- **Owner fingerprint:** -
- **Owner since:** 2026-09-12T21:45:38Z
- **Owner until:** 2026-09-12T23:45:38Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-13T17:17:36Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Observation: tools/integrate.py rebinds results locally after each fix-loop retest and returns only a boolean. On success, cmd_run saves its original pre-fix list, so integration_results.json and the status command continue to report failures after the run announced success. Expected: the final verified result set is saved. Actual: persisted status describes the initial failing run.

## Fix

<unfixed — raised only>

## Notes
