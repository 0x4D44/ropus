# ROP-BUG-KILN-00005 — Integration fix loop persists stale pre-fix results

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** tools/integration-results
- **Raised:** 2026-08-13T17:17:36Z
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
- **State history:** Open (2026-08-13T17:17:36Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Observation: tools/integrate.py rebinds results locally after each fix-loop retest and returns only a boolean. On success, cmd_run saves its original pre-fix list, so integration_results.json and the status command continue to report failures after the run announced success. Expected: the final verified result set is saved. Actual: persisted status describes the initial failing run.

## Fix

<unfixed — raised only>

## Notes
