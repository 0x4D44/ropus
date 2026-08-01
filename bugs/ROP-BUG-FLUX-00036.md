# ROP-BUG-FLUX-00036 — fb2k seek failures can corrupt or lose reader state

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-fb2k/seek
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T205757Z-p68527-n499317000-c1 branch=task/bug-ROP-BUG-FLUX-00036-run-fix-20260801T205757Z-p68527-n499317000-c1 code=af9124baf688473a2cb80b8ed12bebbdc699197f gate=manual) -> Closed (2026-08-01, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main dc05a88; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main dcfd694. /Users/md/language/ropus/ropus-fb2k/src/reader.rs:688-700 and :729-741 take packet_reader, then propagate callback-backed seek errors before restoring Some, so a transient IO or abort can leave the handle empty and the next operation becomes INTERNAL through invariant expects. Index rollback at :776-801 also ignores two abort-aware restore failures and can re-seat at the scan interruption cursor; clearing cancellation can then resume from the wrong place. The repeated bare-reader seek plus PacketReader seek_bytes widens the failure window. Expected: failed or cancelled seek/index operations preserve the prior readable position or explicitly mark the handle terminal without a later panic. Fix: centralize one transactional reseat/rollback helper, use a restoration path that cancellation cannot suppress, remove duplicate host seeks, and test target-seek failure, rewind failure, and abort-clear-resume content. Static review only; no seek or test ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-01, independent two-eyes, host flux)

- Fix commit `af9124b` centralizes a transactional reseat/rollback helper for seek/index
  failures in `ropus-fb2k/src/reader.rs`, replacing the prior bare-reader-plus-`PacketReader`
  double-seek path and its two abort-aware restore failures.
- New tests `failed_rewind_keeps_reader_usable`, `failed_target_seek_keeps_reader_usable`,
  `seek_during_abort_recovery_no_panic`, `seek_propagates_abort_during_index_build`, and
  `aborted_index_scan_resumes_original_content_after_clear` cover target-seek failure, rewind
  failure, and abort-clear-resume as required.
- `cargo clippy -p ropus-fb2k --all-targets --locked -- -D warnings` clean; `cargo test -p
  ropus-fb2k --locked`: 75 passed, 0 failed.
