# ROP-BUG-FLUX-00010 — Multistream decoder setters bypass validation and CELT complexity propagation

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus/opus-multistream-ctl
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T153302Z-p26375-n308063000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00010-run-fix-20260801T153302Z-p26375-n308063000-c1
- **Owner base:** 61125abac5ca5b9f5d31e286e90db14d9e9352ab
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T15:33:02Z
- **Owner until:** 2026-08-01T17:33:02Z
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
