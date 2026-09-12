# ROP-BUG-KILN-00004 — Legacy integration tools pass when required fixtures are absent

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** tools/test-integrity
- **Raised:** 2026-08-13T17:17:35Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260912T213422Z-1ed60022
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KILN-00004-run-fix-20260912T213422Z-1ed60022
- **Owner base:** 486d400497ad83c7228aba67acadd4cc73a653bd
- **Owner fingerprint:** -
- **Owner since:** 2026-09-12T21:34:22Z
- **Owner until:** 2026-09-12T23:34:22Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-13T17:17:35Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Observation: tools/integrate.py skips missing configured WAV files and returns success when its results list has no failures, including an empty list. tools/bisect_fix.py similarly maps missing WAV files to an indeterminate value, excludes them from its failure count, and can declare all zero tests passing; its scan and test commands also return zero after observed failures. Expected: missing required fixtures or a failing comparator produce a nonzero result. Actual: fresh or incomplete corpora can yield false-green automation.

## Fix

<unfixed — raised only>

## Notes
