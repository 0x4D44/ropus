# ROP-BUG-KIL-00033 — fb2k Ogg packets and metadata have no allocation bounds

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** ropus-fb2k/input-limits
- **Raised:** 2026-08-22T06:10:44Z
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
- **State history:** Open (2026-08-22T06:10:44Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Static review at HEAD 3e0f6c1. ropus-fb2k/src/reader.rs:312-340 and :534 assemble header and audio packets through ogg::PacketReader without a size limit; continued pages are retained until the complete packet exists. ropus-fb2k/src/tags.rs:105-139 then clones the vendor and every comment, including cover art filtered only later at src/lib.rs:350-356. A crafted comment header or padded audio packet can exhaust process memory before validation. Expected: enforce explicit RFC 7845-compatible packet and metadata budgets while assembling input, reject oversized audio packets, and avoid retaining filtered blobs. Add boundary fixtures at and over each limit. Static review only; no app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
