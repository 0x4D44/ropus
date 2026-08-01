# ROP-BUG-FLUX-00052 — Info performs full packet and page scans for every query

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-tools-core/info-scale
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T224659Z-p8638-n131126000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00052-run-fix-20260801T224659Z-p8638-n131126000-c1
- **Owner base:** 77d8a93bd64e7d956ef52dbdd08d094b9ece2bc0
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T22:46:59Z
- **Owner until:** 2026-08-02T00:46:59Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main ac7ff8a. /Users/md/language/ropus/ropus-tools-core/src/commands/info.rs:68-83 builds the complete summary before selecting output mode, :125-182 retains every packet TOC and always performs the raw granule scan, and /Users/md/language/ropus/ropus-tools-core/src/container/ogg.rs:218-267 reads the full file into memory and stores every page granule. Even scalar queries therefore use memory and work proportional to the whole input, with compressed bytes plus per-packet/page vectors. Fix: select a query-specific collection plan, stream page parsing, retain TOCs only for extended output, and stop once the requested scalar can be finalized. Add a bounded-reader oracle. Static review only; no info command or measurement ran.

## Fix

<unfixed — raised only>

## Notes

### Query-specific collection acceptance (2026-07-31)

Parse and validate `QueryKey` before opening the input. Use a header/tag plan
for fixed metadata, a streaming page plan for duration and bitrate, and retain
packet TOCs only for extended human output. Non-tag scalar queries must not
allocate tag strings or picture payloads. Prove the plans with bounded-reader
and large-comment oracles. Static review of `ropusinfo` at `origin/main`
`6a312e1`; no command or measurement ran.
