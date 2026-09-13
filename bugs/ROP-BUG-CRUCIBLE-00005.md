# ROP-BUG-CRUCIBLE-00005 — C ABI scratch buffers allocate before frame-size validation

- **State:** Closed
- **Priority:** Could
- **Severity:** Medium
- **Area:** capi/frame-size-allocation
- **Raised:** 2026-08-14T14:26:39Z
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
- **State history:** Open (2026-08-14T14:26:39Z, raised via `deltic bugs new`) -> Fixed (2026-09-13T03:47:46Z, deltic:auto role=fix run=fix-20260913T033334Z-41a4b194 branch=task/bug-ROP-BUG-CRUCIBLE-00005-run-fix-20260913T033334Z-41a4b194 code=8b51233d92cf326d07f5e3944b28223f5f0a01c2 gate=manual) -> Closed (2026-09-13T17:49:57Z, independent two-eyes verification model=codex@xhigh, verifier=CRUCIBLE, fixer=deltic:auto, fix=8b51233d92cf326d07f5e3944b28223f5f0a01c2)

## Observation

opus_multistream_decode_float sizes and allocates an i16 scratch Vec from the raw caller frame_size at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\ms_decoder.rs:441 before the core multistream decoder clamps to 120 ms. Projection decode similarly allocates stream_pcm from the raw frame size at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\ropus\src\opus\multistream.rs:2436 before entering the clamped multistream path, and projection float encode allocates conversion/mixed buffers before frame-size selection at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\projection.rs:410 and C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\ropus\src\opus\multistream.rs:2272. A very large positive frame_size can therefore trigger infallible allocation and abort, whereas the upstream native paths clamp before scratch allocation. Expected: oversized capacity is clamped or rejected before allocation. Actual: wrapper/core staging allocates from the unvalidated value. Apply the codec maximum before every temporary allocation, use checked fallible allocation, and add 120-ms boundary plus oversized PLC/float/projection regression tests. This is distinct residual behavior from closed ROP-BUG-FLUX-00009, which covered only negative frame-size casts.

## Fix

### Independent verification summary (2026-09-13)

- Re-ran the multistream and projection float/decode scratch-allocation regressions; the C-API gate passed 21 Rust tests and 1 C test.
- A red control bypassed the root fallible scratch-allocation seam; the oversized-frame allocation regression failed, and the fix was restored.

## Notes
