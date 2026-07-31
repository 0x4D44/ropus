# ROP-BUG-FLUX-00046 — Play and Opus transcode ignore OpusHead output gain

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-tools-core/shared-decode-gain
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

Static review at origin/main ac7ff8a. Direct decode applies OpusHead output_gain at /Users/md/language/ropus/ropus-tools-core/src/commands/decode.rs:167-175. The separate shared decoder at /Users/md/language/ropus/ropus-tools-core/src/audio/decode.rs:42-49,111-147,196-225 stores only pre_skip and never calls set_gain. ropusplay and Opus-to-Opus encode use that path at commands/play.rs:132-177 and commands/encode.rs:112-128. Valid nonzero-gain files therefore play or transcode at a different amplitude than ropusdec. Fix: parse the full header, apply Q8 output gain once, and add paired zero/-6 dB shared-decode fixtures. Static review only; no playback, transcode, or test ran.

## Fix

<unfixed — raised only>

## Notes
