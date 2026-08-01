# ROP-BUG-FLUX-00012 — Multistream decoder reset retains neural PLC history

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus/opus-multistream-reset
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T165929Z-p68457-n903210000-c1 branch=task/bug-ROP-BUG-FLUX-00012-run-fix-20260801T165929Z-p68457-n903210000-c1 code=a38df9bf41fa784bf865b319431117badec02959 gate=manual)

## Observation

Static source inspection confirmed that OpusMSDecoder::reset at /Users/md/language/ropus/ropus/src/opus/multistream.rs:2000 routes through OpusDecoder::ms_reset at /Users/md/language/ropus/ropus/src/opus/decoder.rs:1602. That helper resets framing, SILK, and CELT state but omits lpcnet.reset(), which the canonical OpusDecoder::reset performs at :628 while preserving model weights. Expected: reset clears all per-stream neural PLC, FEC, analysis, and GRU history. Actual: multistream reuse can carry neural concealment history across streams. Static review only; output divergence was not executed under this pass.

## Fix

<unfixed — raised only>

## Notes
