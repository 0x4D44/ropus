# ROP-BUG-KIL-00033 — fb2k Ogg packets and metadata have no allocation bounds

- **State:** Closed
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
- **State history:** Open (2026-08-22T06:10:44Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-13T04:57:26Z, deltic:auto role=fix run=fix-20260913T043003Z-2ba26a6c branch=task/bug-ROP-BUG-KIL-00033-run-fix-20260913T043003Z-2ba26a6c code=e8afd6bc8b86bc9f25f92c54edbeeb172f7c1607 gate=manual) -> Closed (2026-09-13T16:33:35Z, independent two-eyes verification model=codex@xhigh, verifier=CRUCIBLE, fixer=deltic:auto, fix=e8afd6bc8b86bc9f25f92c54edbeeb172f7c1607)

## Observation

Static review at HEAD 3e0f6c1. ropus-fb2k/src/reader.rs:312-340 and :534 assemble header and audio packets through ogg::PacketReader without a size limit; continued pages are retained until the complete packet exists. ropus-fb2k/src/tags.rs:105-139 then clones the vendor and every comment, including cover art filtered only later at src/lib.rs:350-356. A crafted comment header or padded audio packet can exhaust process memory before validation. Expected: enforce explicit RFC 7845-compatible packet and metadata budgets while assembling input, reject oversized audio packets, and avoid retaining filtered blobs. Add boundary fixtures at and over each limit. Static review only; no app, build, test, or harness ran.

## Fix

### Verification summary (2026-09-13, independent verifier)

- Re-ran the bounded Opus packet boundary tests; both the at-limit and oversized continued-packet cases passed. The `ropus-fb2k` 111-test and `ropus-tools-core` 197-test package gates passed.
- Red control: changing the packet-length comparison from `>` to `>=` made the at-limit regression fail its own `packet at limit must parse` assertion. The mutation was restored.

## Notes
