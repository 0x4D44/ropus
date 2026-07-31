# ROP-BUG-FLUX-00064 — All-invalid tracks count as successful empty playback

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropusplay/decode-completeness
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

Static review at origin/main e5d7113. /Users/md/language/ropus/ropus-tools-core/src/audio/decode.rs:160-235 skips every native or Opus packet decode failure and still returns Ok even when it emitted no audio. /Users/md/language/ropus/ropus-tools-core/src/commands/play.rs:132-151 treats that empty result as a successful track and resets consecutive_errors; :174-211,444-470 appends an empty sink and classifies its immediate drain as TrackFinished. LoopMode::Single and LoopMode::All can therefore re-decode a corrupt track forever, while Off exits zero without playing audio. Fix: return a typed failure unless at least one valid frame survives the defined pre-skip/completeness policy; explicitly decide truly zero-duration stream behavior; add header-only, all-invalid, and pre-skip-consumes-all fixtures across Off, All, and Single. Static review only; no decoder, player, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
