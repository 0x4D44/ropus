# ROP-BUG-KILN-00009 — Long fuzz launchers hide worker failures and accept unsafe durations

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** tools/fuzz-launchers
- **Raised:** 2026-08-13T17:17:38Z
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
- **State history:** Open (2026-08-13T17:17:38Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-12T22:35:45Z, deltic:auto role=fix run=fix-20260912T222100Z-8af8029e branch=task/bug-ROP-BUG-KILN-00009-run-fix-20260912T222100Z-8af8029e code=10a0102dd42c1184ceacb6f7749e69677744d55d gate=manual) -> Closed (2026-09-13T17:49:57Z, independent two-eyes verification model=codex@xhigh, verifier=CRUCIBLE, fixer=deltic:auto, fix=10a0102dd42c1184ceacb6f7749e69677744d55d)

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

### Independent verification summary (2026-09-13)

- Re-ran `python -m unittest -v tools.test_fuzz_launchers`; all 4 tests passed, covering both launchers, mixed outcomes, bounds, and injection-shaped input.
- Red controls hid worker failure and disabled duration bounds; the mixed-outcome and invalid-duration regressions failed, and the fixes were restored.
## Notes
