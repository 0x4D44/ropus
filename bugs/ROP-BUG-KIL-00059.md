# ROP-BUG-KIL-00059 — ropusinfo Ogg packet assembly has no memory bound

- **State:** Fixed
- **Priority:** Should
- **Severity:** High
- **Area:** ropusinfo/input-limits
- **Raised:** 2026-08-22T12:29:57Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T153758Z-3cb111b0
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KIL-00059-run-verify-20260913T153758Z-3cb111b0
- **Owner base:** 08e2207e39193985415be3fe9299cdde96ddef18
- **Owner fingerprint:** sha256:a35b4eda3c913e86d6791bc0405e7e4679fb2bb9fb2a361a2607edfd1ae2d8ef
- **Owner since:** 2026-09-13T15:37:58Z
- **Owner until:** 2026-09-13T17:37:58Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T12:29:57Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-09-13T08:33:44Z, deltic:auto role=fix run=fix-20260913T080515Z-7df35780 branch=task/bug-ROP-BUG-KIL-00059-run-fix-20260913T080515Z-7df35780 code=1e6486c2365da3dfcd09e8bdd6745ddd03814086 gate=manual)

## Observation

Static review at HEAD 1c8b85f. ropus-tools-core/src/commands/info.rs:187-198,233-257,446-453,481-497,504-526 reads OpusHead, OpusTags, and audio through ogg::PacketReader before applying any packet-size budget. Cargo.lock:978-984 pins ogg 0.9.2; its reading.rs:408-423 and :499-508 retains every continued-page fragment and allocates the complete packet before returning it. A crafted Ogg packet continued across the file can therefore exhaust memory or abort ropusinfo before parse_opus_head, OpusTags::parse, or validate_opus_audio_packet can reject it. Default and extended summaries plus channels, vendor, comment, duration, and bitrate queries all reach this assembly path. Expected: reject over-budget header, tag, and audio packets while consuming lacing, before retaining continuation data. Fix with a bounded selected-stream Ogg reader, RFC-compatible audio limits, explicit metadata budgets, and boundary fixtures at and above each limit. This is the ropusinfo counterpart of ROP-BUG-KIL-00033 and should coordinate with ROP-REQ-FLUX-00058/00059, but those records do not cover this command path. Static inspection only; no app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
