# ROP-BUG-CRUCIBLE-00001 — C ABI allocation failure reporting misses nested codec construction

- **State:** Open
- **Priority:** Could
- **Severity:** Medium
- **Area:** capi/allocation-errors
- **Raised:** 2026-08-14T14:26:03Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260913T022358Z-e9856244
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-CRUCIBLE-00001-run-fix-20260913T022358Z-e9856244
- **Owner base:** ea8d8ba9c6fe0acad2f8a8181f009f43e1317792
- **Owner fingerprint:** -
- **Owner since:** 2026-09-13T02:23:58Z
- **Owner until:** 2026-09-13T04:23:58Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-14T14:26:03Z, raised via `deltic bugs new`)

## Observation

CAPI create/init wrappers call OpusEncoder::new, OpusDecoder::new, multistream constructors, and projection constructors before capi::alloc::try_box (for example C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\encoder.rs:167 and C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\decoder.rs:211). Those core constructors perform infallible Vec and Box allocations, including C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\ropus\src\opus\encoder.rs:1539 and C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\ropus\src\opus\decoder.rs:574, so allocator exhaustion can still abort before the wrapper can return OPUS_ALLOC_FAIL. The existing failpoint tests in C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\lib.rs:182 only instrument CAPI helper allocations and therefore do not prove constructor-wide fallibility. Expected: every allocation in a valid C create/init call fails recoverably with OPUS_ALLOC_FAIL. Actual: nested allocation failure can terminate the process. Add fallible core construction or a shared allocation seam, stage all nested state before publishing handles, and inject failures inside the core constructors. This is residual behavior after closed ROP-BUG-FLUX-00017, so it requires a new record.

## Fix

<unfixed — raised only>

## Notes
