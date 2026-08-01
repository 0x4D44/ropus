# ROP-BUG-FLUX-00050 — Public interleaved-audio helpers silently truncate malformed frames

- **State:** Closed
- **Priority:** Could
- **Severity:** Low
- **Area:** ropus-tools-core/audio-shape
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T223058Z-p54209-n151716000-c1 branch=task/bug-ROP-BUG-FLUX-00050-run-fix-20260801T223058Z-p54209-n151716000-c1 code=c8300e1 gate=manual) -> Closed (2026-08-02, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main 3528b9e; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main ac7ff8a. /Users/md/language/ropus/ropus-tools-core/src/audio/downmix.rs:16-25 and audio/resample.rs:13-43 use chunks_exact/division without rejecting len modulo channels, silently dropping an incomplete frame. audio/wav.rs:33-47 permits zero channels while the float writer rejects it; :93-114 can truncate fact-frame division, and both header paths use unchecked rate/alignment multiplication. Normal command buffers appear aligned, but these modules are public. Fix: reject zero/unsupported channels, zero rates, and unaligned sample lengths at every public boundary; use checked header arithmetic and add malformed-shape tests. Static review only; no helper or test ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-02, independent two-eyes, host flux)

- Fix commit `c8300e1` adds shape validation to `audio/downmix.rs`, `audio/resample.rs`, and
  `audio/wav.rs` (reject zero/unaligned channel counts, zero sample rates, and unaligned
  interleaved lengths before doing checked header/frame-count arithmetic).
- Regression re-verified by construction: the fix's 4 new tests were spliced onto the
  pre-fix versions of these 3 files at `c8300e1~1` — all 4 failed
  (`downmix_rejects_incomplete_stereo_frame`,
  `resample_rejects_incomplete_interleaved_frame`,
  `pcm_wav_rejects_invalid_shape_and_header_arithmetic_before_writing`,
  `float_wav_rejects_invalid_shape_before_writing`), each panicking on the missing
  validation (e.g. "odd stereo input must error", "zero channels"). All 4 pass at the
  current tree.
- `cargo clippy -p ropus-tools-core --all-targets --locked -- -D warnings` clean; `cargo test
  -p ropus-tools-core --locked`: 129 lib + 22 integration passed, 0 failed.
