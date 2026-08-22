# ROP-BUG-KIL-00042 — fb2k open accepts malformed Opus header sequences

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-fb2k/headers
- **Raised:** 2026-08-22T06:10:46Z
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
- **State history:** Open (2026-08-22T06:10:46Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Static review at HEAD 3e0f6c1. ropus-fb2k/src/reader.rs:315-340 accepts the first physical packet as OpusHead and the next as OpusTags without verifying required BOS/page placement, zero granules, or that both packets share stream_serial. parse_opus_head at :1235-1262 also accepts trailing bytes for version 1, whose defined family-0 layout is exactly 19 bytes; extra fields are reserved for later compatible minor versions. Malformed input can therefore combine headers from different logical streams or violate mandatory layout while opening successfully. Expected: validate header stream identity, ordering/page flags and granules, require the version-1 family-0 length, while preserving RFC-compatible extensions for minor versions 2 through 15. Add one fixture per rejected invariant. Static review only; no app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
