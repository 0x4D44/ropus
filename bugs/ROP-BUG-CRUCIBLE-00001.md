# ROP-BUG-CRUCIBLE-00001 — C ABI allocation failure reporting misses nested codec construction

- **State:** Closed
- **Priority:** Could
- **Severity:** Medium
- **Area:** capi/allocation-errors
- **Raised:** 2026-08-14T14:26:03Z
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
- **State history:** Open (2026-08-14T14:26:03Z, raised via `deltic bugs new`) -> Fixed (2026-09-13T03:01:22Z, deltic:auto role=fix run=fix-20260913T022358Z-e9856244 branch=task/bug-ROP-BUG-CRUCIBLE-00001-run-fix-20260913T022358Z-e9856244 code=a4a5fd4f70c850e7420fb44a36e44849da77b0c2 gate=manual) -> Closed (2026-09-13T17:49:57Z, independent two-eyes verification model=codex@xhigh, verifier=CRUCIBLE, fixer=deltic:auto, fix=a4a5fd4f70c850e7420fb44a36e44849da77b0c2)

## Observation

CAPI create/init wrappers call OpusEncoder::new, OpusDecoder::new, multistream constructors, and projection constructors before capi::alloc::try_box (for example C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\encoder.rs:167 and C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\decoder.rs:211). Those core constructors perform infallible Vec and Box allocations, including C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\ropus\src\opus\encoder.rs:1539 and C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\ropus\src\opus\decoder.rs:574, so allocator exhaustion can still abort before the wrapper can return OPUS_ALLOC_FAIL. The existing failpoint tests in C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\lib.rs:182 only instrument CAPI helper allocations and therefore do not prove constructor-wide fallibility. Expected: every allocation in a valid C create/init call fails recoverably with OPUS_ALLOC_FAIL. Actual: nested allocation failure can terminate the process. Add fallible core construction or a shared allocation seam, stage all nested state before publishing handles, and inject failures inside the core constructors. This is residual behavior after closed ROP-BUG-FLUX-00017, so it requires a new record.

## Fix

### Independent verification summary (2026-09-13)

- Re-ran the C-API allocation-failure regressions, including encoder and decoder nested construction, embedded-model loading, in-place init, multistream construction, and projection parameter publication; `cargo test -p capi --locked` passed 21 Rust tests and 1 C test.
- A red control disabled the failpoint's zero-budget branch; all 14 allocation tests failed (the first assertion poisoned the shared lock for the remainder), and the guard was restored before the green gate.

## Notes
