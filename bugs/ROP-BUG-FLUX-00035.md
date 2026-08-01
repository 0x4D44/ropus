# ROP-BUG-FLUX-00035 — fb2k decoder ignores OpusHead output gain

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-fb2k/decode
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T205144Z-p55683-n378703000-c1 branch=task/bug-ROP-BUG-FLUX-00035-run-fix-20260801T205144Z-p55683-n378703000-c1 code=a6154866325a9ada8754a835bfdad185e0e3c93f gate=manual)

## Observation

Static review at origin/main dcfd694. /Users/md/language/ropus/ropus-fb2k/src/reader.rs:74-85 and :1024-1051 parse and store the signed Q7.8 OpusHead output_gain, but decoder construction at :452-463 never applies it and no later production path reads it. The C info ABI also does not expose it. Valid nonzero-gain files therefore play at the wrong level, distinct from optional ReplayGain tags. Expected: the decoder applies header output gain exactly once, including after seek reset. Fix: call the existing OpusDecoder Q8 gain setter during lazy initialization and parameterize fixtures with positive and negative header gains to compare amplitude before and after seeks. Static review only; no playback or test ran.

## Fix

<unfixed — raised only>

## Notes
