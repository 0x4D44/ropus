# ROP-BUG-FLUX-00035 — fb2k decoder ignores OpusHead output gain

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-fb2k/decode
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T205144Z-p55683-n378703000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00035-run-fix-20260801T205144Z-p55683-n378703000-c1
- **Owner base:** 24ec808ceeedde351c591c6d9a4c3346c4689bb9
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T20:51:44Z
- **Owner until:** 2026-08-01T22:51:44Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main dcfd694. /Users/md/language/ropus/ropus-fb2k/src/reader.rs:74-85 and :1024-1051 parse and store the signed Q7.8 OpusHead output_gain, but decoder construction at :452-463 never applies it and no later production path reads it. The C info ABI also does not expose it. Valid nonzero-gain files therefore play at the wrong level, distinct from optional ReplayGain tags. Expected: the decoder applies header output gain exactly once, including after seek reset. Fix: call the existing OpusDecoder Q8 gain setter during lazy initialization and parameterize fixtures with positive and negative header gains to compare amplitude before and after seeks. Static review only; no playback or test ran.

## Fix

<unfixed — raised only>

## Notes
