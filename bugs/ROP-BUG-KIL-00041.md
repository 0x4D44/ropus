# ROP-BUG-KIL-00041 — fb2k applies malformed R128 gain tags

- **State:** Closed
- **Priority:** Could
- **Severity:** Medium
- **Area:** ropus-fb2k/tags
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
- **State history:** Open (2026-08-22T06:10:46Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-13T05:59:53Z, deltic:auto role=fix run=fix-20260913T054837Z-0da42b8a branch=task/bug-ROP-BUG-KIL-00041-run-fix-20260913T054837Z-0da42b8a code=4218c1a0f206d87739a7ae291331f25c1035d496 gate=manual) -> Closed (2026-09-13T16:43:36Z, independent two-eyes verification model=codex@xhigh, verifier=CRUCIBLE, fixer=deltic:auto, fix=4218c1a0f206d87739a7ae291331f25c1035d496)

## Observation

Static review at HEAD 3e0f6c1. ropus-fb2k/src/tags.rs:286-292 trims whitespace, parses unrestricted i32 text, clamps after converting to dB, and extract_replaygain at :219-245 lets later duplicates overwrite earlier values. RFC 7845 section 5.2.1 requires at most one tag, no whitespace, no more than six ASCII characters, and a raw integer in -32768..=32767. Values such as -32769 are accepted and can apply near-muting gain; some valid high raw values are rejected by the unrelated +/-127 dB post-conversion clamp. Expected: validate the exact grammar and raw range, handle duplicates explicitly as invalid metadata, and cover all boundaries. Static review only; no app, build, test, or harness ran.

## Fix

### Verification summary (2026-09-13)

- Re-ran `r128_gain_requires_exact_raw_i16_grammar`; it passed, and the ropus-fb2k package gate passed all 111 tests.
- A red control widened raw R128 parsing from `i16` to `i32`; malformed-tag rejection failed, and the fix was restored.

## Notes
