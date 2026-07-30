# ROP-BUG-FLUX-00013 — Extension parsers trust caller counts beyond safe slice bounds

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus/opus-extensions
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

Static source inspection confirmed two safe public panic paths. opus_packet_extensions_parse at /Users/md/language/ropus/ropus/src/opus/repacketizer.rs:652 trusts nb_extensions rather than extensions.len() before indexing at :675. opus_packet_extensions_parse_ext at /Users/md/language/ropus/ropus/src/opus/extensions.rs:104 accepts negative per-frame counts into a wrapping prefix sum, and at :132-136 can cast a negative index or trust declared capacity beyond the output slice. The iterator also lacks a release check that len is within data.len(). Expected: inconsistent lengths, capacities, and counts return OPUS_BAD_ARG or OPUS_BUFFER_TOO_SMALL. Actual: safe inputs can panic. Static review only; no code was executed.

## Fix

<unfixed — raised only>

## Notes
