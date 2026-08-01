# ROP-BUG-FLUX-00011 — Multistream expert frame duration setter accepts invalid values

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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T153753Z-p44527-n851873000-c1 branch=task/bug-ROP-BUG-FLUX-00011-run-fix-20260801T153753Z-p44527-n851873000-c1 code=11355cdede39e3e904fe42ecbb90322bf75005c2 gate=manual) -> Closed (2026-08-01, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main dc05a88; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static source inspection confirmed that OpusMSEncoder::set_expert_frame_duration at /Users/md/language/ropus/ropus/src/opus/multistream.rs:1524 stores every i32 and returns no status. The equivalent OpusEncoder setter at /Users/md/language/ropus/ropus/src/opus/encoder.rs:3975 accepts only OPUS_FRAMESIZE_ARG or defined 2.5-120 ms values and returns OPUS_BAD_ARG otherwise. Expected: the multistream setter rejects invalid CTL values without mutation. Actual: it stores an invalid state and defers failure until encode. Static review only; no code was executed.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-01, independent two-eyes, host flux)

- Commit `11355cd` makes `OpusMSEncoder::set_expert_frame_duration`
  (`ropus/src/opus/multistream.rs:1527`) return `i32` and reject any value that is neither
  `OPUS_FRAMESIZE_ARG` nor within `OPUS_FRAMESIZE_2_5_MS..=OPUS_FRAMESIZE_120_MS`, matching the
  single-stream encoder's contract.
- New test `encoder_expert_frame_duration_rejects_invalid_values` confirms a valid 20 ms value
  is accepted and reflected by the getter, then that `4999` and `120ms+1` are both rejected
  with `OPUS_BAD_ARG` and leave the prior duration unchanged.
- `cargo test -p ropus --locked`: 1888 passed, 0 failed.
