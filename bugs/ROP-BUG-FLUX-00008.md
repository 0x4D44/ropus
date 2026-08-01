# ROP-BUG-FLUX-00008 — Multistream constructors panic on undersized mapping and matrix slices

- **State:** Closed
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T152126Z-p90074-n005769000-c1 branch=task/bug-ROP-BUG-FLUX-00008-run-fix-20260801T152126Z-p90074-n005769000-c1 code=cfc23f68f1543c8f37d121621874fcebdc26a3c1 gate=manual) -> Closed (2026-08-01, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main dc05a88; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static source inspection confirmed that OpusMSEncoder::new and OpusMSDecoder::new copy channels entries from mapping without checking mapping.len() at /Users/md/language/ropus/ropus/src/opus/multistream.rs:915 and :1611. OpusProjectionDecoder::new similarly trusts demixing_matrix_size instead of demixing_matrix_bytes.len() before indexing at :2379. Expected: malformed safe-slice configuration returns OPUS_BAD_ARG from these Result constructors. Actual: an undersized slice causes an indexing panic. Static review only; no code was executed.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-01, independent two-eyes, host flux)

- Commit `cfc23f6` adds `mapping.len() < channels as usize` bounds checks to
  `OpusMSEncoder::new` and `OpusMSDecoder::new` (`ropus/src/opus/multistream.rs:907,1606`), and
  replaces `OpusProjectionDecoder::new`'s size-only check with checked-arithmetic dimension
  validation plus a `demixing_matrix_bytes.len()` bound (`:2368`), matching the described root
  cause exactly.
- New test `constructors_reject_undersized_safe_slices_without_panicking`
  (`ropus/src/opus/multistream.rs:4134`) drives all three constructors under
  `std::panic::catch_unwind` and asserts `Err(OPUS_BAD_ARG)` instead of a panic.
- `cargo test -p ropus --locked`: 1888 passed, 0 failed.
