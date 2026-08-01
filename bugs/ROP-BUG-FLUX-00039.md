# ROP-BUG-FLUX-00039 — fb2k OpusTags parser accepts invalid Vorbis field names

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** ropus-fb2k/tags
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T211851Z-p19786-n399480000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00039-run-fix-20260801T211851Z-p19786-n399480000-c1
- **Owner base:** 263675259150c0edd99865f84bb515f997d7b8c8
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T21:18:51Z
- **Owner until:** 2026-08-01T23:18:51Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main dcfd694. /Users/md/language/ropus/ropus-fb2k/src/tags.rs:11 documents the Vorbis field-name grammar, but parse at :114-128 rejects only missing separators and empty keys. Control bytes, non-ASCII UTF-8, and bytes outside ASCII 0x20 through 0x7D are accepted, uppercased, and surfaced through /Users/md/language/ropus/ropus-fb2k/src/lib.rs:350-380. Expected: malformed comment field names are rejected before reaching host metadata callbacks. Fix: validate every key byte against ASCII 0x20 through 0x7D excluding equals, add a typed invalid-key error, and cover control and non-ASCII cases. Static review only; no parser or test ran.

## Fix

<unfixed — raised only>

## Notes
