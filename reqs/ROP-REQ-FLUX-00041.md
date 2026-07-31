# ROP-REQ-FLUX-00041 — Centralize validated Ogg page metadata parsing

- **State:** Draft
- **Priority:** Should
- **Area:** container/ogg-page-parsing
- **Raised:** 2026-07-31
- **Implemented-by:** —
- **Satisfied-by:** —
- **Violated-by:** —
- **Flow:** heavy
- **Claimed-by:** —
- **State history:** Draft (2026-07-31, raised via `deltic reqs new` model=gpt-5.6-sol@xhigh)

## Statement

The workspace must use one neutral, validated Ogg page metadata parser for capture/version, flags, granule, serial, lacing, page extent, and an explicit checksum policy. The bounded reverse-duration scan and forward seek-index walk may retain separate traversal policies, but they must not duplicate byte offsets or acceptance rules. Migrate /Users/md/language/ropus/ropus-fb2k/src/reader.rs:822-902 and :964-1018 plus the source-noted CLI copy at :959-963. The acceptance oracle must feed identical valid, truncated, fake-capture, wrong-serial, and malformed-extent pages through every consumer and prove consistent metadata or rejection. This structural extraction supports ROP-BUG-FLUX-00037 but does not replace its immediate fix.

## Notes
