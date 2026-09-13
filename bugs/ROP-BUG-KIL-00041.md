# ROP-BUG-KIL-00041 — fb2k applies malformed R128 gain tags

- **State:** Fixed
- **Priority:** Could
- **Severity:** Medium
- **Area:** ropus-fb2k/tags
- **Raised:** 2026-08-22T06:10:46Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T152553Z-527e34f3
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KIL-00041-run-verify-20260913T152553Z-527e34f3
- **Owner base:** 09f6ca6c84a8d6e966ef5f2a14b0276e6d75295c
- **Owner fingerprint:** sha256:ec999bf4ffa12d22a77df30c37727eb0e780c88c4107e93e6dff959a0f5f023c
- **Owner since:** 2026-09-13T15:25:53Z
- **Owner until:** 2026-09-13T17:25:53Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T06:10:46Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-13T05:59:53Z, deltic:auto role=fix run=fix-20260913T054837Z-0da42b8a branch=task/bug-ROP-BUG-KIL-00041-run-fix-20260913T054837Z-0da42b8a code=4218c1a0f206d87739a7ae291331f25c1035d496 gate=manual)

## Observation

Static review at HEAD 3e0f6c1. ropus-fb2k/src/tags.rs:286-292 trims whitespace, parses unrestricted i32 text, clamps after converting to dB, and extract_replaygain at :219-245 lets later duplicates overwrite earlier values. RFC 7845 section 5.2.1 requires at most one tag, no whitespace, no more than six ASCII characters, and a raw integer in -32768..=32767. Values such as -32769 are accepted and can apply near-muting gain; some valid high raw values are rejected by the unrelated +/-127 dB post-conversion clamp. Expected: validate the exact grammar and raw range, handle duplicates explicitly as invalid metadata, and cover all boundaries. Static review only; no app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
