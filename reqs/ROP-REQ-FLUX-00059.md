# ROP-REQ-FLUX-00059 — Unify Opus header, stream, and decode policy

- **State:** Draft
- **Priority:** Should
- **Area:** ropus-tools-core/opus-decode-pipeline
- **Raised:** 2026-07-31
- **Implemented-by:** —
- **Satisfied-by:** —
- **Violated-by:** —
- **Flow:** heavy
- **Claimed-by:** —
- **State history:** Draft (2026-07-31, raised via `deltic reqs new` model=gpt-5.6-sol@xhigh)

## Statement

Consolidate the duplicated Opus decode policy in /Users/md/language/ropus/ropus-tools-core/src/commands/decode.rs and /Users/md/language/ropus/ropus-tools-core/src/audio/decode.rs into one reusable pipeline with configurable sinks. The copies have already drifted on OpusHead validation, selected logical-stream handling, pre-skip and EOS trimming, output gain, empty/corrupt packet policy, and gapless metadata. This produced tracked defects ROP-BUG-FLUX-00043, ROP-BUG-FLUX-00045, ROP-BUG-FLUX-00046, and ROP-BUG-FLUX-00056. Define one validated header/stream state machine and timing contract; make ropusdec, ropusplay, and ropusenc transcode adapters select output policy without reimplementing demux/decode semantics. Add independent fixtures for nonzero gain, foreign serials, partial EOS, and malformed empty packets. Raised from static review at origin/main ac7ff8a; no decoder or test ran.

## Notes

### `ropusplay` acceptance detail (2026-07-31)

The unified pipeline must expose completeness, selected-stream EOS, and whether
any valid audio frame survived pre-skip. It must return a typed failure for the
all-invalid case tracked by `ROP-BUG-FLUX-00064`, reject the empty-packet PLC
case in `ROP-BUG-FLUX-00056`, trim exact EOS under `ROP-BUG-FLUX-00043`, and
combine header plus user Q8 gain under `ROP-BUG-FLUX-00046` before publishing a
track. Add header-only, all-invalid, pre-skip-consumes-all, malformed-empty,
partial-EOS, and signed-gain fixtures shared across play, decode, and
transcode. Static review at `origin/main` `e5d7113`; no decoder, player, or
test ran.
