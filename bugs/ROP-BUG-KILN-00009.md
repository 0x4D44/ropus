# ROP-BUG-KILN-00009 — Long fuzz launchers hide worker failures and accept unsafe durations

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** tools/fuzz-launchers
- **Raised:** 2026-08-13T17:17:38Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T172538Z-a86ea38f
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KILN-00009-run-verify-20260913T172538Z-a86ea38f
- **Owner base:** ae9cda9f35207db29a7cdf4b59afc0ab75e9b687
- **Owner fingerprint:** sha256:ce1e0413037c9fed29ebd40bc271be2f5855033d160cb64f6e37fbb2fab38c09
- **Owner since:** 2026-09-13T17:25:38Z
- **Owner until:** 2026-09-13T19:25:38Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-13T17:17:38Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-12T22:35:45Z, deltic:auto role=fix run=fix-20260912T222100Z-8af8029e branch=task/bug-ROP-BUG-KILN-00009-run-fix-20260912T222100Z-8af8029e code=10a0102dd42c1184ceacb6f7749e69677744d55d gate=manual)

## Observation

Observation: tools/overnight_fuzz_launch.sh and tools/fuzz_24h_launch_v2.sh launch background workers, do not retain or aggregate each wait status, and print completion after an unchecked wait, allowing failed campaigns to exit zero. They also evaluate an unvalidated duration argument in Bash arithmetic; arithmetic recursively evaluates crafted variable contents. Expected: validate a bounded decimal duration, wait every PID, and fail on any worker failure. Actual: invalid or failed campaigns can look complete, and untrusted wrapper input can execute during arithmetic evaluation.

## Fix

Integrated code commit `10a0102dd42c1184ceacb6f7749e69677744d55d` now validates
both launcher durations as positive decimal values from 1 through 86400 seconds
before arithmetic, retains every worker PID, waits on every worker, and returns
nonzero with target-labelled failure details when any worker fails. Focused
coverage lives in `tools/test_fuzz_launchers.py`.

Validation evidence:

- Regression proof: mutating the v2 worker-status branch to `if false` made
  `test_mixed_worker_outcomes_fail_for_both_launchers` fail because the v2
  launcher returned zero after a fake target exited 7. Mutating its duration
  bound to `if false` made `test_duration_rejects_zero_and_values_above_the_day_limit`
  fail for both rejected inputs. Mutating the overnight status branch likewise
  made the mixed-outcome test fail because that launcher returned zero.
- The same fake-cargo harness also rejected a command-substitution duration
  without creating its marker file, proving the input was not evaluated.
- After restoration, `python -m unittest -v tools.test_fuzz_launchers` passed all
  four tests, covering both launchers, mixed outcomes, success, bounds, and injection-shaped input.
- Normalized `bash -n` parses passed for both launcher scripts.

## Notes
