# ROP-BUG-FLUX-00052 — Info performs full packet and page scans for every query

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-tools-core/info-scale
- **Raised:** 2026-07-31
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T224659Z-p8638-n131126000-c1 branch=task/bug-ROP-BUG-FLUX-00052-run-fix-20260801T224659Z-p8638-n131126000-c1 code=3400cf5 gate=manual) -> Closed (2026-08-02, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main 3528b9e; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main ac7ff8a. /Users/md/language/ropus/ropus-tools-core/src/commands/info.rs:68-83 builds the complete summary before selecting output mode, :125-182 retains every packet TOC and always performs the raw granule scan, and /Users/md/language/ropus/ropus-tools-core/src/container/ogg.rs:218-267 reads the full file into memory and stores every page granule. Even scalar queries therefore use memory and work proportional to the whole input, with compressed bytes plus per-packet/page vectors. Fix: select a query-specific collection plan, stream page parsing, retain TOCs only for extended output, and stop once the requested scalar can be finalized. Add a bounded-reader oracle. Static review only; no info command or measurement ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-02, independent two-eyes, host flux)

Covers the main observation plus the query-specific collection acceptance note. Fixed by
the same commit as `ROP-BUG-FLUX-00054`; see that bug's closure note for the shared
unknown-query evidence.

- Fix commit `3400cf5` rewrites `ropus-tools-core/src/commands/info.rs` to parse the query
  key before opening the input, select a query-specific collection plan (header/tag plan vs.
  streaming page-scan plan), and stream raw Ogg page scans instead of reading the whole file
  and retaining every packet TOC.
- Regression re-verified by construction: the fix's new tests
  (`query_key_is_validated_without_opening_input`,
  `fixed_header_plan_stops_before_large_tags_packet`) reference `validate_query_key` and
  `read_head_from`, neither of which existed pre-fix at `3400cf5~1` (compile error),
  confirming there was no query-specific plan and no early-validation boundary before this
  fix. Both tests pass at the current tree.
- `cargo clippy -p ropus-tools-core --all-targets --locked -- -D warnings` clean; `cargo test
  -p ropus-tools-core -p ropusinfo --locked`: 129 passed, 0 failed (plus the 2 named tests
  individually confirmed passing).

### Query-specific collection acceptance (2026-07-31)

Parse and validate `QueryKey` before opening the input. Use a header/tag plan
for fixed metadata, a streaming page plan for duration and bitrate, and retain
packet TOCs only for extended human output. Non-tag scalar queries must not
allocate tag strings or picture payloads. Prove the plans with bounded-reader
and large-comment oracles. Static review of `ropusinfo` at `origin/main`
`6a312e1`; no command or measurement ran.
