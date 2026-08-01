# ROP-BUG-FLUX-00008 — Multistream constructors panic on undersized mapping and matrix slices

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus/opus-multistream
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T152126Z-p90074-n005769000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00008-run-fix-20260801T152126Z-p90074-n005769000-c1
- **Owner base:** 5b655a3deab55d87a0b73c25006c6a0c26bb5a3c
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T15:21:26Z
- **Owner until:** 2026-08-01T17:21:26Z
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
