# ROP-BUG-FLUX-00050 — Public interleaved-audio helpers silently truncate malformed frames

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** ropus-tools-core/audio-shape
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T223058Z-p54209-n151716000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00050-run-fix-20260801T223058Z-p54209-n151716000-c1
- **Owner base:** 91f3cda11353c500351efbda5e21f4967cae666f
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T22:30:58Z
- **Owner until:** 2026-08-02T00:30:58Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main ac7ff8a. /Users/md/language/ropus/ropus-tools-core/src/audio/downmix.rs:16-25 and audio/resample.rs:13-43 use chunks_exact/division without rejecting len modulo channels, silently dropping an incomplete frame. audio/wav.rs:33-47 permits zero channels while the float writer rejects it; :93-114 can truncate fact-frame division, and both header paths use unchecked rate/alignment multiplication. Normal command buffers appear aligned, but these modules are public. Fix: reject zero/unsupported channels, zero rates, and unaligned sample lengths at every public boundary; use checked header arithmetic and add malformed-shape tests. Static review only; no helper or test ran.

## Fix

<unfixed — raised only>

## Notes
