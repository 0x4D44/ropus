# ROP-BUG-KIL-00049 — Packet and repacketizer lengths can exceed Rust slice bounds

- **State:** Fixed
- **Priority:** Should
- **Severity:** High
- **Area:** ropus/opus-packet
- **Raised:** 2026-08-22T08:28:17Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T152941Z-866a8287
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KIL-00049-run-verify-20260913T152941Z-866a8287
- **Owner base:** 73d9a6409dea6413006d9fe9b0136d245f00fafe
- **Owner fingerprint:** sha256:35069a2a1c111efba594f843d2fe5b7421f8bc1b7164131e317ccd6fb5a93849
- **Owner since:** 2026-09-13T15:29:41Z
- **Owner until:** 2026-09-13T17:29:41Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T08:28:17Z, raised via `deltic bugs new` model=gpt-5.6-sol@max) -> Fixed (2026-09-13T07:02:54Z, deltic:auto role=fix run=fix-20260913T064739Z-6d8940d7 branch=task/bug-ROP-BUG-KIL-00049-run-fix-20260913T064739Z-6d8940d7 code=2a9229badbdd69cc290a3d01835939068fdfe846 gate=manual)

## Observation

Static review at HEAD 3972b03. ropus/src/opus/decoder.rs:208-243 accepts an explicit len without checking len <= data.len(); ropus/src/opus/repacketizer.rs:1035-1080 carries the same contract through OpusRepacketizer::cat. out_range_impl at repacketizer.rs:1097-1200 trusts maxlen rather than data.len(), while pad/unpad paths at :1359-1510 slice by len/new_len. A safe caller can pass a short slice plus a larger claimed length or capacity and trigger a panic or a false-successful parse. Expected: safe public APIs reject inconsistent slice and length pairs with OPUS_BAD_ARG or OPUS_BUFFER_TOO_SMALL. Fix: validate every source and destination capacity before access, parse only a bounded subslice, and add short-slice/oversized-length boundary tests. This is distinct from closed ROP-BUG-FLUX-00013, which fixed extension parsing. Static inspection only; no code, app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
