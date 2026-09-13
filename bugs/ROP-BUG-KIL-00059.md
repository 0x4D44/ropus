# ROP-BUG-KIL-00059 — ropusinfo Ogg packet assembly has no memory bound

- **State:** Closed
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
- **State history:** Open (2026-08-22T12:29:57Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-09-13T08:33:44Z, deltic:auto role=fix run=fix-20260913T080515Z-7df35780 branch=task/bug-ROP-BUG-KIL-00059-run-fix-20260913T080515Z-7df35780 code=1e6486c2365da3dfcd09e8bdd6745ddd03814086 gate=manual) -> Closed (2026-09-13T16:47:37Z, independent two-eyes verification model=codex@xhigh, verifier=CRUCIBLE, fixer=deltic:auto, fix=1e6486c2365da3dfcd09e8bdd6745ddd03814086)

## Observation

Static review at HEAD 1c8b85f. ropus-tools-core/src/commands/info.rs:187-198,233-257,446-453,481-497,504-526 reads OpusHead, OpusTags, and audio through ogg::PacketReader before applying any packet-size budget. Cargo.lock:978-984 pins ogg 0.9.2; its reading.rs:408-423 and :499-508 retains every continued-page fragment and allocates the complete packet before returning it. A crafted Ogg packet continued across the file can therefore exhaust memory or abort ropusinfo before parse_opus_head, OpusTags::parse, or validate_opus_audio_packet can reject it. Default and extended summaries plus channels, vendor, comment, duration, and bitrate queries all reach this assembly path. Expected: reject over-budget header, tag, and audio packets while consuming lacing, before retaining continuation data. Fix with a bounded selected-stream Ogg reader, RFC-compatible audio limits, explicit metadata budgets, and boundary fixtures at and above each limit. This is the ropusinfo counterpart of ROP-BUG-KIL-00033 and should coordinate with ROP-REQ-FLUX-00058/00059, but those records do not cover this command path. Static inspection only; no app, build, test, or harness ran.

## Fix

### Verification summary (2026-09-13)

- Re-ran `bounded_reader_rejects_oversized_continued_audio_packet`; it passed, and the relevant package gates passed.
- A red control raised the audio packet limit by one byte; the oversized continued packet was accepted and the regression panicked, so the fix was restored.

## Notes
