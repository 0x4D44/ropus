# ROP-BUG-FLUX-00011 — Multistream expert frame duration setter accepts invalid values

- **State:** Open
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Static source inspection confirmed that OpusMSEncoder::set_expert_frame_duration at /Users/md/language/ropus/ropus/src/opus/multistream.rs:1524 stores every i32 and returns no status. The equivalent OpusEncoder setter at /Users/md/language/ropus/ropus/src/opus/encoder.rs:3975 accepts only OPUS_FRAMESIZE_ARG or defined 2.5-120 ms values and returns OPUS_BAD_ARG otherwise. Expected: the multistream setter rejects invalid CTL values without mutation. Actual: it stores an invalid state and defers failure until encode. Static review only; no code was executed.

## Fix

<unfixed — raised only>

## Notes
