# ROP-BUG-FLUX-00057 — Required integration tests silently pass when vectors are absent

- **State:** Closed
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-02, deltic:auto role=fix run=fix-20260801T232638Z-p40721-n439187000-c1 branch=task/bug-ROP-BUG-FLUX-00057-run-fix-20260801T232638Z-p40721-n439187000-c1 code=033b529c520a4a2af11cda3ed69c0171d2dcb95c gate=manual) -> Closed (2026-08-02, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main 43a61b9; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main ac7ff8a. Five behavioral tests at /Users/md/language/ropus/ropus-tools-core/tests/round_trip.rs:66-74,182-187,238-240,815-820,900-905 print SKIPPING and return success when committed fixtures are missing. Cargo therefore reports green without exercising core round-trip, metadata, serial, downmix, or picture behavior; stderr text does not alter test status. Fix: generate deterministic fixtures in-test or make missing required vectors a hard failure. If any fixture is intentionally optional, move that coverage to an explicitly ignored/manual test and retain a generated mandatory oracle. Static review only; no test ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-02, independent two-eyes, host flux)

Covers the main observation plus all three notes (`ropusinfo` behavior tests, `ropusplay`
device test, DTX/DRED gate).

- Fix commit `033b529` replaces external test-vector dependencies with in-test deterministic
  WAV generation in `ropus-tools-core/tests/round_trip.rs` and `ropusinfo/tests/cli.rs`,
  tightens `ropusplay/tests/cli.rs` to accept only the structured no-device error, and makes
  `harness-deep-plc/tests/dred_dtx_first_frame_diff.rs` hard-fail on an inconclusive fixture
  (now `#[ignore]`d by default rather than silently returning success).
- Regression re-verified directly: at the pre-fix commit (`033b529c~1`), moving
  `tests/vectors/48k_sine1k_loud.wav` aside and running
  `encode_then_decode_48k_sine_round_trips_with_snr_above_20_db` reported `test result: ok`
  in 0.00s — the false-green this bug describes (SKIPPING, no real work, still "passes").
  Repeating the same removal at the current (fixed) tree: the test still ran a full
  encode/decode/SNR check against its own generated fixture and passed for real (`ok` in
  0.06s, with genuine encode/decode log output, not an instant no-op).
  `ropusinfo/tests/cli.rs` and `ropusplay/tests/cli.rs` diffs confirmed by reading: both now
  generate their own fixtures / require a specific error string rather than treating any
  failure as a skip, matching the notes exactly.
  `dred_dtx_first_frame_diff.rs`'s test module is compiled only under
  `#[cfg(not(no_reference))]` (gated on the git-ignored C reference), so it could not be
  independently re-executed in this environment; the code-level fix (`found.expect(...)`
  replacing a silent `return`, plus explicit `#[ignore]`) was confirmed by reading the diff.
- `cargo clippy -p ropus-tools-core --all-targets --locked -- -D warnings` clean; `cargo test
  -p ropus-tools-core -p ropusinfo -p ropusplay`: 132+22+2+7+2 passed, 0 failed.

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
