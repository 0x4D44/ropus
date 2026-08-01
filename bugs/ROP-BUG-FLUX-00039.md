# ROP-BUG-FLUX-00039 — fb2k OpusTags parser accepts invalid Vorbis field names

- **State:** Closed
- **Priority:** Could
- **Severity:** Low
- **Area:** ropus-fb2k/tags
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T211851Z-p19786-n399480000-c1 branch=task/bug-ROP-BUG-FLUX-00039-run-fix-20260801T211851Z-p19786-n399480000-c1 code=2f8c07a7376d54ef8f2b4e0f1affee51b2dc710f gate=manual) -> Closed (2026-08-01, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main 9e51e0a; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main dcfd694. /Users/md/language/ropus/ropus-fb2k/src/tags.rs:11 documents the Vorbis field-name grammar, but parse at :114-128 rejects only missing separators and empty keys. Control bytes, non-ASCII UTF-8, and bytes outside ASCII 0x20 through 0x7D are accepted, uppercased, and surfaced through /Users/md/language/ropus/ropus-fb2k/src/lib.rs:350-380. Expected: malformed comment field names are rejected before reaching host metadata callbacks. Fix: validate every key byte against ASCII 0x20 through 0x7D excluding equals, add a typed invalid-key error, and cover control and non-ASCII cases. Static review only; no parser or test ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-01, independent two-eyes, host flux)

- Fix commit `2f8c07a` validates every OpusTags comment-key byte against the Vorbis ASCII
  field-name grammar (0x20-0x7D, excluding `=`) before normalization in
  `ropus-fb2k/src/tags.rs`, returning a typed parser error for control-byte and non-ASCII
  keys instead of surfacing them to host metadata callbacks.
- New tests `rejects_control_byte_in_field_name`, `rejects_non_ascii_field_name` (unit) and
  `open_rejects_invalid_tag_field_names` (C ABI integration) pass.
- `cargo clippy -p ropus-fb2k --all-targets --locked -- -D warnings` clean; `cargo test -p
  ropus-fb2k --locked`: 80 passed, 0 failed.
