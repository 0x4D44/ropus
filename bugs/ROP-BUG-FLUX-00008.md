# ROP-BUG-FLUX-00008 — Multistream constructors panic on undersized mapping and matrix slices

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus/opus-multistream
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

Static source inspection confirmed that OpusMSEncoder::new and OpusMSDecoder::new copy channels entries from mapping without checking mapping.len() at /Users/md/language/ropus/ropus/src/opus/multistream.rs:915 and :1611. OpusProjectionDecoder::new similarly trusts demixing_matrix_size instead of demixing_matrix_bytes.len() before indexing at :2379. Expected: malformed safe-slice configuration returns OPUS_BAD_ARG from these Result constructors. Actual: an undersized slice causes an indexing panic. Static review only; no code was executed.

## Fix

<unfixed — raised only>

## Notes
