# ROP-BUG-CRUCIBLE-00004 — Extension shims abort on large caller capacities

- **State:** Open
- **Priority:** Could
- **Severity:** Medium
- **Area:** capi/extensions-allocation
- **Raised:** 2026-08-14T14:26:28Z
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
- **State history:** Open (2026-08-14T14:26:28Z, raised via `deltic bugs new`)

## Observation

The extension parse, parse_ext, and generate shims allocate temporary Vec storage directly from caller-controlled extension counts with vec! or Vec::with_capacity at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\extensions.rs:146, :260, and :325. The repacketizer extension path repeats this at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\repacketizer.rs:308. Only negative counts are rejected. A large nonnegative capacity can therefore enter Rust infallible allocation and abort the process; ffi_guard cannot catch allocator aborts. The upstream C paths process caller storage directly and do not require an equivalent duplicate allocation. Expected: the adapter returns OPUS_ALLOC_FAIL or rejects a count above a packet-derived bound. Actual: shim-owned temporary allocation can terminate the conformance process. Bound counts by packet/output limits, allocate fallibly with try_reserve_exact, and add failpoint plus large-capacity regression coverage.

## Fix

<unfixed — raised only>

## Notes
