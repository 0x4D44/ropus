# ROP-BUG-KIL-00035 — fb2k decodes zero-octet Ogg audio packets as PLC

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-fb2k/malformed-audio
- **Raised:** 2026-08-22T06:10:44Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T152230Z-45022563
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KIL-00035-run-verify-20260913T152230Z-45022563
- **Owner base:** 64e97e40e5ddeb1b8ebac443df226de92653815a
- **Owner fingerprint:** sha256:d0e7c6365722a4155942c3f2e77a53667573cc98986841654e8b3d6840a73cff
- **Owner since:** 2026-09-13T15:22:30Z
- **Owner until:** 2026-09-13T17:22:30Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T06:10:44Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-13T05:12:03Z, deltic:auto role=fix run=fix-20260913T045820Z-394d7194 branch=task/bug-ROP-BUG-KIL-00035-run-fix-20260913T045820Z-394d7194 code=737a1ef597985d986a58c27d594a90300f8a230d gate=manual)

## Observation

Static review at HEAD 3e0f6c1. ropus-fb2k/src/reader.rs:550-558 passes Some(&pkt.data) to decode_float even when the Ogg packet is empty. ropus/src/opus/decoder.rs:1394-1405 treats an empty slice like packet-loss concealment, so malformed container input fabricates audio instead of returning INVALID_STREAM. RFC 7845 section 3 requires zero-octet Ogg audio packets to be treated as malformed. Expected: reject an empty container packet before codec decode and add a malformed-empty fixture. ROP-BUG-FLUX-00056 fixed the same class only in ropus-tools-core, not this component. Static review only; no app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
