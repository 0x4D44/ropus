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
