# ROP-BUG-FLUX-00042 — Encoder truncates or pads source audio through incorrect granule accounting

- **State:** Open
- **Priority:** Must
- **Severity:** High
- **Area:** ropus-tools-core/encode-timeline
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main ac7ff8a. /Users/md/language/ropus/ropus-tools-core/src/commands/encode.rs:179-191 writes encoder lookahead as pre_skip, but :261-310 feeds only source frames rounded to a packet and records full submitted-frame counts as granules. It never drains lookahead or sets EOS to source_frames + pre_skip; exact-frame inputs decode pre_skip samples short, partial inputs can retain padding, and zero-frame input writes headers without any EOS page. /Users/md/language/ropus/ropus-tools-core/tests/round_trip.rs:120-131 compares only the shared shorter prefix and hides the length error. Expected: preserve the exact source frame count and emit a closed valid stream. Fix: drain with silence through lookahead plus packet rounding, set final EOS granule to source_frames + pre_skip, define/reject empty input, and assert exact decoded lengths for exact and partial frames. Static review only; no encoder or test ran.

## Fix

<unfixed — raised only>

## Notes
