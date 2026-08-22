# ROP-BUG-KIL-00047 — Info and playback accept OpusTags from another logical stream

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-tools-core/stream-validation
- **Raised:** 2026-08-22T07:33:51Z
- **Discovery source:** Agent
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
- **State history:** Open (2026-08-22T07:33:51Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at HEAD f9a3871. commands/info.rs:189-198 records the OpusHead stream serial but parses the next physical packet as OpusTags without checking that serial; read_head_and_tags at :442-454 repeats the omission. commands/play.rs:617-630 likewise treats the second physical packet as display tags without validating the Head or matching serial. A malformed or multiplexed Ogg can therefore report or display metadata from a different logical stream, while commands/decode.rs:191-209 correctly rejects that mismatch. Fix by selecting and validating OpusHead and OpusTags as one stream-scoped header pair, shared across consumers; for unsupported multiplexing, reject cross-serial headers explicitly. Add Head serial A plus Tags serial B fixtures for default info, every strict query, and playback label fallback. This finding does not require general multiplexed-audio support, which remains outside the signed chained-stream scope. Static inspection only; no app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
