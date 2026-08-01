# ROP-BUG-FLUX-00010 — Multistream decoder setters bypass validation and CELT complexity propagation

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus/opus-multistream-ctl
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T153302Z-p26375-n308063000-c1 branch=task/bug-ROP-BUG-FLUX-00010-run-fix-20260801T153302Z-p26375-n308063000-c1 code=ab04c06f3666903778861f5d6084eb28ee9caf74 gate=manual) -> Closed (2026-08-01, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main dc05a88; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static source inspection confirmed that OpusMSDecoder set_gain, set_phase_inversion_disabled, and set_complexity at /Users/md/language/ropus/ropus/src/opus/multistream.rs:1966-1992 call raw ms_set helpers and always return OPUS_OK. The helpers at /Users/md/language/ropus/ropus/src/opus/decoder.rs:1582-1592 skip the public setter validation; ms_set_complexity also fails to call CeltDecoder::set_complexity, unlike OpusDecoder::set_complexity at :1518. Expected: invalid CTL values return OPUS_BAD_ARG and valid complexity updates both Opus and CELT state. Actual: invalid values persist and valid complexity leaves CELT stale. Static review only; no code was executed.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-01, independent two-eyes, host flux)

- Commit `ab04c06` routes `OpusDecoder::ms_set_gain` / `ms_set_complexity` /
  `ms_set_phase_inversion_disabled` (`ropus/src/opus/decoder.rs:1584-1596`) through the
  already-validated single-stream setters and returns `Result<(), i32>`; `ms_set_complexity`
  now calls the real `set_complexity`, which also updates the CELT decoder (previously only
  `OpusDecoder::set_complexity` did). `OpusMSDecoder::set_gain/set_phase_inversion_disabled/
  set_complexity` (`multistream.rs:1971-2001`) now propagate the first failing stream's error.
- Existing setter tests updated to `.unwrap()` the new `Result`s and continue to pass.
- `cargo test -p ropus --locked`: 1888 passed, 0 failed.
