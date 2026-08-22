# ROP-BUG-KIL-00035 — fb2k decodes zero-octet Ogg audio packets as PLC

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-fb2k/malformed-audio
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

Static review at HEAD 3e0f6c1. ropus-fb2k/src/reader.rs:550-558 passes Some(&pkt.data) to decode_float even when the Ogg packet is empty. ropus/src/opus/decoder.rs:1394-1405 treats an empty slice like packet-loss concealment, so malformed container input fabricates audio instead of returning INVALID_STREAM. RFC 7845 section 3 requires zero-octet Ogg audio packets to be treated as malformed. Expected: reject an empty container packet before codec decode and add a malformed-empty fixture. ROP-BUG-FLUX-00056 fixed the same class only in ropus-tools-core, not this component. Static review only; no app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
