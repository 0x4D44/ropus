# ROP-BUG-FLUX-00043 — Decode, play, and transcode emit samples beyond the EOS granule

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-tools-core/decode-timeline
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

Static review at origin/main ac7ff8a. /Users/md/language/ropus/ropus-tools-core/src/commands/decode.rs:237-370 appends every decoded packet and removes only leading pre_skip; it never uses last_in_stream or absgp_page. /Users/md/language/ropus/ropus-tools-core/src/audio/decode.rs:83-90,196-227 uses default non-gapless Symphonia options and its custom Opus path also trims only the head. Valid files with a partial final packet therefore expose encoder padding in ropusdec, ropusplay, and Opus transcode. Expected under RFC 7845 section 4.4: samples after the EOS granule are discarded. Fix: retain selected-stream timing, enable/apply packet end trim or clamp final output to EOS minus pre_skip, reject impossible granules, and add independently-granulated i16/float fixtures. Static review only; no decoder or test ran.

## Fix

<unfixed — raised only>

## Notes

### Additional `ropusdec` EOF trigger (2026-07-31)

Static review at `origin/main` `bfe19ba` confirmed that
`/Users/md/language/ropus/ropus-tools-core/src/commands/decode.rs:255-300`
also treats physical EOF after a complete non-EOS page as successful decode.
The loop stops on `Ok(None)` without requiring `last_in_stream` or retaining the
final absolute granule. It can therefore publish a plausible truncated WAV/PCM
with exit zero, not only expose padding beyond a present EOS granule.

The fix and oracle must require selected-stream EOS before publication and
reject physical EOF before EOS. Use an independently built stream truncated
after a complete non-EOS page so the test does not depend on the encoder tracked
by `ROP-BUG-FLUX-00042`. This was a static review; no decoder or test ran.

### `ropusplay` completion acceptance (2026-07-31)

`/Users/md/language/ropus/ropus-tools-core/src/audio/decode.rs:160-235`
also treats physical `UnexpectedEof` as successful completion and exposes the
result directly to playback at
`/Users/md/language/ropus/ropus-tools-core/src/commands/play.rs:132-193`.
Require selected-stream EOS and exact end trimming before a track is published.
Tighten `/Users/md/language/ropus/ropus-tools-core/tests/round_trip.rs:120-134`
from shared-prefix comparison to an independently granulated exact sample-count
oracle. Wholly undecodable tracks are tracked separately as
`ROP-BUG-FLUX-00064`. Static review at `origin/main` `e5d7113`; no decoder,
player, or test ran.
