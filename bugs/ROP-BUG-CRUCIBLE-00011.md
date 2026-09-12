# ROP-BUG-CRUCIBLE-00011 — Oversized corpus output is fully buffered after rejection

- **State:** Fixed
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
- **State history:** Open (2026-08-14T15:50:26Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-09-12T23:37:29Z, deltic:auto role=fix run=fix-20260912T233205Z-c8473729 branch=task/bug-ROP-BUG-CRUCIBLE-00011-run-fix-20260912T233205Z-c8473729 code=33c0a0442aad8b19d4b96ea420653b32873138e6 gate=manual)

## Observation

Static review at origin/main bb54eb50. C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\corpus.rs:482-491 records that a generated file exceeds max_size_bytes, but immediately continues to sha256_of_opus_payload at :492. That function reads the complete file and builds a second payload buffer at :906-955, so an already-rejected runaway output still consumes unbounded memory and time. Expected: max_size_bytes is an effective resource boundary. Fix: stop processing after the size breach or stream-parse and hash with a hard byte cap; add an oversized sparse-file oracle that proves no full allocation. Static review only; no corpus generation, app, test, or harness ran.

## Fix

Implemented in `full-test/src/corpus.rs` and integrated at code commit
`33c0a0442aad8b19d4b96ea420653b32873138e6`.

- The generated-output size check now marks oversized entries and skips payload
  hashing, so `max_size_bytes` is an effective memory and work boundary.
- Added a sparse-file regression oracle with an injected hasher that panics if
  an oversized output reaches `fs::read`; the report retains the byte count,
  `oversized` status, and no digest.

Verification:

- `$null | deltic timeout 180 cargo test -p full-test corpus::tests` — 20
  passed, 0 failed.
- `deltic timeout 180 cargo check -p full-test` — passed.
- `deltic timeout 120 cargo fmt --all -- --check` — passed.
- `git diff --check` — passed.
- Red proof: temporarily restored unconditional hashing; the sparse-file test
  failed with `oversized output must not reach the payload hasher`. The guard
  was restored and the focused suite passed.

## Notes
