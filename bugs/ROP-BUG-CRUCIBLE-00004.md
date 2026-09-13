# ROP-BUG-CRUCIBLE-00004 — Extension shims abort on large caller capacities

- **State:** Fixed
- **Priority:** Could
- **Severity:** Medium
- **Area:** capi/extensions-allocation
- **Raised:** 2026-08-14T14:26:28Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T145145Z-5cd86e1c
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-CRUCIBLE-00004-run-verify-20260913T145145Z-5cd86e1c
- **Owner base:** df7b326d246ac1e747e622ffe30c55852814d657
- **Owner fingerprint:** sha256:b92fa1b337fedb4fa0522d1e411da4b0f1b547973dff6dc66bcdc95a29d97259
- **Owner since:** 2026-09-13T14:51:45Z
- **Owner until:** 2026-09-13T16:51:45Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-14T14:26:28Z, raised via `deltic bugs new`) -> Fixed (2026-09-13T03:32:43Z, deltic:auto role=fix run=fix-20260913T031705Z-c7a149f8 branch=task/bug-ROP-BUG-CRUCIBLE-00004-run-fix-20260913T031705Z-c7a149f8 code=177fb542895f8ff33963f4da87b5e12523ce02c0 gate=manual)

## Observation

The extension parse, parse_ext, and generate shims allocate temporary Vec storage directly from caller-controlled extension counts with vec! or Vec::with_capacity at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\extensions.rs:146, :260, and :325. The repacketizer extension path repeats this at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\repacketizer.rs:308. Only negative counts are rejected. A large nonnegative capacity can therefore enter Rust infallible allocation and abort the process; ffi_guard cannot catch allocator aborts. The upstream C paths process caller storage directly and do not require an equivalent duplicate allocation. Expected: the adapter returns OPUS_ALLOC_FAIL or rejects a count above a packet-derived bound. Actual: shim-owned temporary allocation can terminate the conformance process. Bound counts by packet/output limits, allocate fallibly with try_reserve_exact, and add failpoint plus large-capacity regression coverage.

## Fix

<unfixed — raised only>

## Notes
