# ROP-BUG-KILN-00013 — Benchmark sweep parser no longer matches harness output

- **State:** Closed
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
- **State history:** Open (2026-08-13T17:17:40Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-12T23:18:28Z, deltic:auto role=fix run=fix-20260912T231226Z-df922e2e branch=task/bug-ROP-BUG-KILN-00013-run-fix-20260912T231226Z-df922e2e code=ceb21caa6c1638619319fee2d06b2681288231b4 gate=manual) -> Closed (2026-09-13T17:49:57Z, independent two-eyes verification model=codex@xhigh, verifier=CRUCIBLE, fixer=deltic:auto, fix=ceb21caa6c1638619319fee2d06b2681288231b4)

## Observation

Observation: tools/bench_sweep.sh expects legacy lines beginning with encode or decode and exits under set -e when grep finds none, while the current harness prints box-table rows beginning with a vertical separator and labels C encode, Rust encode, C decode, and Rust decode. Its per-vector failure handler also does not aggregate a nonzero final status. Expected: parse the current structured rows and fail the sweep if any required vector fails. Actual: the first successful benchmark aborts before producing the summary.

## Fix

Integrated code commit `ceb21caa6c1638619319fee2d06b2681288231b4` now parses
the current four-row `C encode`/`Rust encode`/`C decode`/`Rust decode` timing
table through one strict helper. It rejects missing, duplicate, nonnumeric, or
non-finite timings, and aggregates benchmark-process and parser failures into
the final sweep status. Focused coverage lives in
`tools/test_bench_sweep.py`.

Validation evidence:

- Regression proof: restoring the old legacy `grep` gate made
  `test_current_table_is_parsed_into_summary` fail on its own
  `assertEqual(result.returncode, 0, ...)` assertion because the sweep exited
  before parsing the current table.
- After restoration, `python -m unittest -v tools.test_bench_sweep` passed all
  four selected tests. They cover current output and ratios, missing and
  non-finite rows, and mixed vector success with a nonzero final status.
- Normalized `bash -n tools/bench_sweep.sh` and
  `python -m py_compile tools/test_bench_sweep.py` passed. A live benchmark was
  not available because the repository lacks `reference/celt/bands.c`; the
  focused fake-binary harness supplied the captured current-format output.

### Independent verification summary (2026-09-13)

- Re-ran the four benchmark-sweep parser tests; all passed, with shell/Python validation also passing.
- A red control restored the legacy grep gate; `test_current_table_is_parsed_into_summary` failed before parsing the current table, and the fix was restored.
## Notes
