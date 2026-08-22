# ROP-BUG-KIL-00059 — ropusinfo Ogg packet assembly has no memory bound

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** ropusinfo/input-limits
- **Raised:** 2026-08-22T12:29:57Z
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
- **State history:** Open (2026-08-22T12:29:57Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at HEAD 1c8b85f. ropus-tools-core/src/commands/info.rs:187-198,233-257,446-453,481-497,504-526 reads OpusHead, OpusTags, and audio through ogg::PacketReader before applying any packet-size budget. Cargo.lock:978-984 pins ogg 0.9.2; its reading.rs:408-423 and :499-508 retains every continued-page fragment and allocates the complete packet before returning it. A crafted Ogg packet continued across the file can therefore exhaust memory or abort ropusinfo before parse_opus_head, OpusTags::parse, or validate_opus_audio_packet can reject it. Default and extended summaries plus channels, vendor, comment, duration, and bitrate queries all reach this assembly path. Expected: reject over-budget header, tag, and audio packets while consuming lacing, before retaining continuation data. Fix with a bounded selected-stream Ogg reader, RFC-compatible audio limits, explicit metadata budgets, and boundary fixtures at and above each limit. This is the ropusinfo counterpart of ROP-BUG-KIL-00033 and should coordinate with ROP-REQ-FLUX-00058/00059, but those records do not cover this command path. Static inspection only; no app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
