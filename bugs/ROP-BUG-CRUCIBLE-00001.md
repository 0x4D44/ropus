# ROP-BUG-CRUCIBLE-00001 — C ABI allocation failure reporting misses nested codec construction

- **State:** Fixed
- **Priority:** Could
- **Severity:** Medium
- **Area:** capi/allocation-errors
- **Raised:** 2026-08-14T14:26:03Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T141756Z-c6a564eb
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-CRUCIBLE-00001-run-verify-20260913T141756Z-c6a564eb
- **Owner base:** a1cde3b320a635db3e4372d435336ed4e56b3429
- **Owner fingerprint:** sha256:2f781e9c0ea50f07fdbf7f60df87c16222f69c9097a915212089c3a045670280
- **Owner since:** 2026-09-13T14:17:56Z
- **Owner until:** 2026-09-13T16:17:56Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-14T14:26:03Z, raised via `deltic bugs new`) -> Fixed (2026-09-13T03:01:22Z, deltic:auto role=fix run=fix-20260913T022358Z-e9856244 branch=task/bug-ROP-BUG-CRUCIBLE-00001-run-fix-20260913T022358Z-e9856244 code=a4a5fd4f70c850e7420fb44a36e44849da77b0c2 gate=manual)

## Observation

CAPI create/init wrappers call OpusEncoder::new, OpusDecoder::new, multistream constructors, and projection constructors before capi::alloc::try_box (for example C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\encoder.rs:167 and C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\decoder.rs:211). Those core constructors perform infallible Vec and Box allocations, including C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\ropus\src\opus\encoder.rs:1539 and C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\ropus\src\opus\decoder.rs:574, so allocator exhaustion can still abort before the wrapper can return OPUS_ALLOC_FAIL. The existing failpoint tests in C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\lib.rs:182 only instrument CAPI helper allocations and therefore do not prove constructor-wide fallibility. Expected: every allocation in a valid C create/init call fails recoverably with OPUS_ALLOC_FAIL. Actual: nested allocation failure can terminate the process. Add fallible core construction or a shared allocation seam, stage all nested state before publishing handles, and inject failures inside the core constructors. This is residual behavior after closed ROP-BUG-FLUX-00017, so it requires a new record.

## Fix

<unfixed — raised only>

## Notes
