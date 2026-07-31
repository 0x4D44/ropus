# ROP-BUG-FLUX-00060 — Picture preprocessing can destroy output and bypass its size cap

- **State:** Open
- **Priority:** Must
- **Severity:** High
- **Area:** ropusenc/picture-output-safety
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main b4e2c31. /Users/md/language/ropus/ropusenc/src/main.rs:131-134,334-347 passes --picture independently from output. /Users/md/language/ropus/ropus-tools-core/src/commands/encode.rs:193-212 creates or truncates the destination and writes OpusHead before :219-243 stats, reads, and validates the picture. A missing, empty, oversized, invalid, or unreadable picture therefore destroys a pre-existing destination before the command fails; an output path that aliases the picture destroys the cover itself. The separate metadata and fs::read opens at :223-234 also let replacement or growth bypass MAX_PICTURE_BYTES and allocate without a hard bound. Expected: invalid auxiliary input leaves existing files unchanged and the cap is enforced during the read. Fix: open picture once, read at most MAX_PICTURE_BYTES + 1, validate/build its tag before any output mutation, reject output aliases against every input, then write regular output through a same-directory temporary and atomically replace only after successful flush. Add missing/empty/oversized/invalid and direct/symlink/hardlink alias tests that preserve sentinel files. Static review only; no application or test ran.

## Fix

<unfixed — raised only>

## Notes
