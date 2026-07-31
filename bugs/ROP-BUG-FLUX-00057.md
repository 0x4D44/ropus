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

### `ropusplay` device test can skip every failure (2026-07-31)

`/Users/md/language/ropus/ropusplay/tests/cli.rs:18-48` labels every nonzero
status from `--list-devices` as a headless-host skip, including panics, argument
regressions, and enumeration failures. Its forced `--quiet` plus nonempty-line
assertion also cannot detect a leaked banner. Skip only a structured
no-device outcome, fail every unrelated status, and use an injected
deterministic enumerator for mandatory empty/nonempty and exact-output
coverage. Keep real hardware enumeration explicitly manual or ignored. Static
review at `origin/main` `e5d7113`; no binary, device, or test ran.

### DTX/DRED first-frame gate succeeds when inconclusive (2026-07-31)

`/Users/md/language/ropus/harness-deep-plc/tests/dred_dtx_first_frame_diff.rs:189-205`
returns success when it never observes a suitable multi-frame DRED packet.
When it does observe one with `ext_frame == 0`, `:208-225` only logs that DTX
did not fire and still passes. The test was meant to replace the deleted
tautological unit oracle, as recorded at
`/Users/md/language/ropus/ropus/src/opus/encoder.rs:8924-8929`, so regressing
the first-frame shift can now remain green. Pre-warm deterministic DTX state,
independently prove the first sub-frame was dropped, require DRED presence, and
assert its extension frame equals the observed `dtx_count`; an inconclusive
fixture must fail or be explicitly ignored. Static review at `origin/main`
`b65f812`; no encoder or test ran.
