# ROP-BUG-KILN-00009 — Long fuzz launchers hide worker failures and accept unsafe durations

- **State:** Open
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
- **State history:** Open (2026-08-13T17:17:38Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Observation: tools/overnight_fuzz_launch.sh and tools/fuzz_24h_launch_v2.sh launch background workers, do not retain or aggregate each wait status, and print completion after an unchecked wait, allowing failed campaigns to exit zero. They also evaluate an unvalidated duration argument in Bash arithmetic; arithmetic recursively evaluates crafted variable contents. Expected: validate a bounded decimal duration, wait every PID, and fail on any worker failure. Actual: invalid or failed campaigns can look complete, and untrusted wrapper input can execute during arithmetic evaluation.

## Fix

<unfixed — raised only>

## Notes
