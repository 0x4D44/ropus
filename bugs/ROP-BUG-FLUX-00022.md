# ROP-BUG-FLUX-00022 — Projection gate passes fixtures without processing a frame

- **State:** Open
- **Priority:** Must
- **Severity:** High
- **Area:** harness/projection-gate
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T041534Z-p18394-n733330000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00022-run-fix-20260801T041534Z-p18394-n733330000-c1
- **Owner base:** 970262d59daa199e346ab3ec1cf0dd55ac47b512
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T04:15:34Z
- **Owner until:** 2026-08-01T06:15:34Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at `origin/main` `d0ab87e`. `/Users/md/language/ropus/harness/src/bin_inner/projection_roundtrip.rs:344` floors the input length to complete 20 ms frames, so an empty or shorter valid WAV produces `total_frames == 0`. The loop at `:358` then performs no encode or decode, while `FixtureResult::passed()` at `:260` checks only zero mismatch counters. The driver prints the release-consumed global PASS marker and exits 0 at `:538` through `:577`. Expected: every passing fixture proves at least one encoded and decoded frame. Actual: a truncated or empty fixture can satisfy the ambisonics gate without exercising audio. Fix: reject zero complete frames, include processed-frame count in `passed()`, reject partial-only input, and add empty/short fixture regression cases. Static review only; no binary, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
