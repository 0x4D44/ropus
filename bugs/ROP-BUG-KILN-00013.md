# ROP-BUG-KILN-00013 — Benchmark sweep parser no longer matches harness output

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** tools/bench-sweep
- **Raised:** 2026-08-13T17:17:40Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260912T231226Z-df922e2e
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KILN-00013-run-fix-20260912T231226Z-df922e2e
- **Owner base:** b990b49ac29de1abc0a55aa0c06e1b3fa3640f50
- **Owner fingerprint:** -
- **Owner since:** 2026-09-12T23:12:26Z
- **Owner until:** 2026-09-13T01:12:26Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-13T17:17:40Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Observation: tools/bench_sweep.sh expects legacy lines beginning with encode or decode and exits under set -e when grep finds none, while the current harness prints box-table rows beginning with a vertical separator and labels C encode, Rust encode, C decode, and Rust decode. Its per-vector failure handler also does not aggregate a nonzero final status. Expected: parse the current structured rows and fail the sweep if any required vector fails. Actual: the first successful benchmark aborts before producing the summary.

## Fix

<unfixed — raised only>

## Notes
