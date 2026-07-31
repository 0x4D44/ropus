# ROP-BUG-FLUX-00057 — Required integration tests silently pass when vectors are absent

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-tools-core/test-integrity
- **Raised:** 2026-07-31
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main ac7ff8a. Five behavioral tests at /Users/md/language/ropus/ropus-tools-core/tests/round_trip.rs:66-74,182-187,238-240,815-820,900-905 print SKIPPING and return success when committed fixtures are missing. Cargo therefore reports green without exercising core round-trip, metadata, serial, downmix, or picture behavior; stderr text does not alter test status. Fix: generate deterministic fixtures in-test or make missing required vectors a hard failure. If any fixture is intentionally optional, move that coverage to an explicitly ignored/manual test and retain a generated mandatory oracle. Static review only; no test ran.

## Fix

<unfixed — raised only>

## Notes

### Confirmed in all `ropusinfo` CLI behavior tests (2026-07-31)

`/Users/md/language/ropus/ropusinfo/tests/cli.rs:16-42,86-287` lets all six
tests return success when the shared vector is absent. Its helper at :65-83
also injects `--quiet --no-color` into every invocation, so it cannot catch
banner-routing or default-color regressions. Make required fixtures
deterministic and mandatory, then test default and controlled output paths
separately. Static review at `origin/main` `6a312e1`; no test ran.
