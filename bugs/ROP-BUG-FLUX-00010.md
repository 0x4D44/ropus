# ROP-BUG-FLUX-00010 — Multistream decoder setters bypass validation and CELT complexity propagation

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

Static source inspection confirmed that OpusMSDecoder set_gain, set_phase_inversion_disabled, and set_complexity at /Users/md/language/ropus/ropus/src/opus/multistream.rs:1966-1992 call raw ms_set helpers and always return OPUS_OK. The helpers at /Users/md/language/ropus/ropus/src/opus/decoder.rs:1582-1592 skip the public setter validation; ms_set_complexity also fails to call CeltDecoder::set_complexity, unlike OpusDecoder::set_complexity at :1518. Expected: invalid CTL values return OPUS_BAD_ARG and valid complexity updates both Opus and CELT state. Actual: invalid values persist and valid complexity leaves CELT stale. Static review only; no code was executed.

## Fix

<unfixed — raised only>

## Notes
