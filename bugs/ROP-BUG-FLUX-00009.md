# ROP-BUG-FLUX-00009 — Projection codec casts negative frame sizes before validation

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus/opus-projection
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T152816Z-p13950-n626310000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00009-run-fix-20260801T152816Z-p13950-n626310000-c1
- **Owner base:** 023aa94459111e07ca26fc39290c61a3e1b76a08
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T15:28:16Z
- **Owner until:** 2026-08-01T17:28:16Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Static source inspection confirmed that OpusProjectionEncoder::encode and OpusProjectionDecoder::decode cast frame_size to usize while sizing temporary Vec buffers at /Users/md/language/ropus/ropus/src/opus/multistream.rs:2263 and :2415, before the underlying codec can reject a non-positive frame size. Expected: a negative frame size returns OPUS_BAD_ARG before allocation arithmetic. Actual: -1 becomes a huge usize and can cause capacity-overflow panic or allocation abort. Static review only; no code was executed.

## Fix

<unfixed — raised only>

## Notes
