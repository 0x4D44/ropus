# ROP-BUG-FLUX-00036 — fb2k seek failures can corrupt or lose reader state

- **State:** Open
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main dcfd694. /Users/md/language/ropus/ropus-fb2k/src/reader.rs:688-700 and :729-741 take packet_reader, then propagate callback-backed seek errors before restoring Some, so a transient IO or abort can leave the handle empty and the next operation becomes INTERNAL through invariant expects. Index rollback at :776-801 also ignores two abort-aware restore failures and can re-seat at the scan interruption cursor; clearing cancellation can then resume from the wrong place. The repeated bare-reader seek plus PacketReader seek_bytes widens the failure window. Expected: failed or cancelled seek/index operations preserve the prior readable position or explicitly mark the handle terminal without a later panic. Fix: centralize one transactional reseat/rollback helper, use a restoration path that cancellation cannot suppress, remove duplicate host seeks, and test target-seek failure, rewind failure, and abort-clear-resume content. Static review only; no seek or test ran.

## Fix

<unfixed — raised only>

## Notes
