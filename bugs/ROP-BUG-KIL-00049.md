# ROP-BUG-KIL-00049 — Packet and repacketizer lengths can exceed Rust slice bounds

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** ropus/opus-packet
- **Raised:** 2026-08-22T08:28:17Z
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
- **State history:** Open (2026-08-22T08:28:17Z, raised via `deltic bugs new` model=gpt-5.6-sol@max)

## Observation

Static review at HEAD 3972b03. ropus/src/opus/decoder.rs:208-243 accepts an explicit len without checking len <= data.len(); ropus/src/opus/repacketizer.rs:1035-1080 carries the same contract through OpusRepacketizer::cat. out_range_impl at repacketizer.rs:1097-1200 trusts maxlen rather than data.len(), while pad/unpad paths at :1359-1510 slice by len/new_len. A safe caller can pass a short slice plus a larger claimed length or capacity and trigger a panic or a false-successful parse. Expected: safe public APIs reject inconsistent slice and length pairs with OPUS_BAD_ARG or OPUS_BUFFER_TOO_SMALL. Fix: validate every source and destination capacity before access, parse only a bounded subslice, and add short-slice/oversized-length boundary tests. This is distinct from closed ROP-BUG-FLUX-00013, which fixed extension parsing. Static inspection only; no code, app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
