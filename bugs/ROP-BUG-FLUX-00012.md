# ROP-BUG-FLUX-00012 — Multistream decoder reset retains neural PLC history

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus/opus-multistream-reset
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T165929Z-p68457-n903210000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00012-run-fix-20260801T165929Z-p68457-n903210000-c1
- **Owner base:** 2a1a00522940976e92c0ac4691b1989cd6e0a38f
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T16:59:29Z
- **Owner until:** 2026-08-01T18:59:29Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Static source inspection confirmed that OpusMSDecoder::reset at /Users/md/language/ropus/ropus/src/opus/multistream.rs:2000 routes through OpusDecoder::ms_reset at /Users/md/language/ropus/ropus/src/opus/decoder.rs:1602. That helper resets framing, SILK, and CELT state but omits lpcnet.reset(), which the canonical OpusDecoder::reset performs at :628 while preserving model weights. Expected: reset clears all per-stream neural PLC, FEC, analysis, and GRU history. Actual: multistream reuse can carry neural concealment history across streams. Static review only; output divergence was not executed under this pass.

## Fix

<unfixed — raised only>

## Notes
