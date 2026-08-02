# ROP-BUG-FLUX-00060 — Picture preprocessing can destroy output and bypass its size cap

- **State:** Closed
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-02, deltic:auto role=fix run=fix-20260801T233850Z-p80645-n650518000-c1 branch=task/bug-ROP-BUG-FLUX-00060-run-fix-20260801T233850Z-p80645-n650518000-c1 code=62bff7d072c4e55faa4a3a873fb141ad9591d61d gate=manual) -> Closed (2026-08-02, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main 43a61b9; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main b4e2c31. /Users/md/language/ropus/ropusenc/src/main.rs:131-134,334-347 passes --picture independently from output. /Users/md/language/ropus/ropus-tools-core/src/commands/encode.rs:193-212 creates or truncates the destination and writes OpusHead before :219-243 stats, reads, and validates the picture. A missing, empty, oversized, invalid, or unreadable picture therefore destroys a pre-existing destination before the command fails; an output path that aliases the picture destroys the cover itself. The separate metadata and fs::read opens at :223-234 also let replacement or growth bypass MAX_PICTURE_BYTES and allocate without a hard bound. Expected: invalid auxiliary input leaves existing files unchanged and the cap is enforced during the read. Fix: open picture once, read at most MAX_PICTURE_BYTES + 1, validate/build its tag before any output mutation, reject output aliases against every input, then write regular output through a same-directory temporary and atomically replace only after successful flush. Add missing/empty/oversized/invalid and direct/symlink/hardlink alias tests that preserve sentinel files. Static review only; no application or test ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-02, independent two-eyes, host flux)

- Fix commit `62bff7d` reworks `ropus-tools-core/src/commands/encode.rs` to read and
  validate the picture (bounded to `MAX_PICTURE_BYTES + 1`) fully before opening the
  destination, reject an output path that aliases the picture (or input), and write regular
  outputs through a same-directory temp file that is atomically renamed into place only
  after a successful flush.
- Regression re-verified by construction: the fix's 3 new tests
  (`picture_failures_preserve_existing_output`,
  `picture_output_direct_alias_is_rejected_without_destroying_picture`,
  `picture_output_link_aliases_are_rejected_without_destroying_picture`) were spliced onto
  the pre-fix `encode.rs` at `62bff7d~1` — all 3 failed: the alias tests found no "same
  file" rejection in the error text, and the preserve-existing-output test hit an
  out-of-order error (metadata read failing after truncation had already begun) instead of
  the expected clean pre-mutation failure. All 3 pass at the current tree.
- `cargo clippy -p ropus-tools-core --all-targets --locked -- -D warnings` clean; `cargo test
  -p ropus-tools-core --locked`: 132 lib + 22 integration passed, 0 failed.
