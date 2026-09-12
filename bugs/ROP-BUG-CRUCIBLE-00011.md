# ROP-BUG-CRUCIBLE-00011 — Oversized corpus output is fully buffered after rejection

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** full-test/corpus-resource-bounds
- **Raised:** 2026-08-14T15:50:26Z
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
- **State history:** Open (2026-08-14T15:50:26Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main bb54eb50. C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\corpus.rs:482-491 records that a generated file exceeds max_size_bytes, but immediately continues to sha256_of_opus_payload at :492. That function reads the complete file and builds a second payload buffer at :906-955, so an already-rejected runaway output still consumes unbounded memory and time. Expected: max_size_bytes is an effective resource boundary. Fix: stop processing after the size breach or stream-parse and hash with a hard byte cap; add an oversized sparse-file oracle that proves no full allocation. Static review only; no corpus generation, app, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
