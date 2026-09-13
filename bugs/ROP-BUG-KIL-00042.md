# ROP-BUG-KIL-00042 — fb2k open accepts malformed Opus header sequences

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-fb2k/headers
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
- **State history:** Open (2026-08-22T06:10:46Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-13T06:23:36Z, deltic:auto role=fix run=fix-20260913T060050Z-4c1ccbb6 branch=task/bug-ROP-BUG-KIL-00042-run-fix-20260913T060050Z-4c1ccbb6 code=7415c6e9c940a0c0ebc1293a84b55e7c35b24b89 gate=manual) -> Closed (2026-09-13T16:43:36Z, independent two-eyes verification model=codex@xhigh, verifier=CRUCIBLE, fixer=deltic:auto, fix=7415c6e9c940a0c0ebc1293a84b55e7c35b24b89)

## Observation

Static review at HEAD 3e0f6c1. ropus-fb2k/src/reader.rs:315-340 accepts the first physical packet as OpusHead and the next as OpusTags without verifying required BOS/page placement, zero granules, or that both packets share stream_serial. parse_opus_head at :1235-1262 also accepts trailing bytes for version 1, whose defined family-0 layout is exactly 19 bytes; extra fields are reserved for later compatible minor versions. Malformed input can therefore combine headers from different logical streams or violate mandatory layout while opening successfully. Expected: validate header stream identity, ordering/page flags and granules, require the version-1 family-0 length, while preserving RFC-compatible extensions for minor versions 2 through 15. Add one fixture per rejected invariant. Static review only; no app, build, test, or harness ran.

## Fix

### Verification summary (2026-09-13)

- Re-ran `open_rejects_nonzero_id_header_granule`; it passed, and the ropus-fb2k package gate passed all 111 tests.
- A red control disabled the non-zero header-granule check; malformed-header rejection failed, and the fix was restored.

## Notes
