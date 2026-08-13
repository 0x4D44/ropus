# ROP-BUG-KILN-00013 — Benchmark sweep parser no longer matches harness output

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** tools/bench-sweep
- **Raised:** 2026-08-13T17:17:40Z
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
- **State history:** Open (2026-08-13T17:17:40Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Observation: tools/bench_sweep.sh expects legacy lines beginning with encode or decode and exits under set -e when grep finds none, while the current harness prints box-table rows beginning with a vertical separator and labels C encode, Rust encode, C decode, and Rust decode. Its per-vector failure handler also does not aggregate a nonzero final status. Expected: parse the current structured rows and fail the sweep if any required vector fails. Actual: the first successful benchmark aborts before producing the summary.

## Fix

<unfixed — raised only>

## Notes
