# ROP-BUG-CRUCIBLE-00005 — C ABI scratch buffers allocate before frame-size validation

- **State:** Fixed
- **Priority:** Could
- **Severity:** Medium
- **Area:** capi/frame-size-allocation
- **Raised:** 2026-08-14T14:26:39Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T145351Z-b35d717a
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-CRUCIBLE-00005-run-verify-20260913T145351Z-b35d717a
- **Owner base:** 7b23bca4dcd715e2227425ffbe377caec6838d36
- **Owner fingerprint:** sha256:df225ba81764d63b894afedb46e35cfcc10717af98f7008a10bf8e6b6e511902
- **Owner since:** 2026-09-13T14:53:51Z
- **Owner until:** 2026-09-13T16:53:51Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-14T14:26:39Z, raised via `deltic bugs new`) -> Fixed (2026-09-13T03:47:46Z, deltic:auto role=fix run=fix-20260913T033334Z-41a4b194 branch=task/bug-ROP-BUG-CRUCIBLE-00005-run-fix-20260913T033334Z-41a4b194 code=8b51233d92cf326d07f5e3944b28223f5f0a01c2 gate=manual)

## Observation

opus_multistream_decode_float sizes and allocates an i16 scratch Vec from the raw caller frame_size at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\ms_decoder.rs:441 before the core multistream decoder clamps to 120 ms. Projection decode similarly allocates stream_pcm from the raw frame size at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\ropus\src\opus\multistream.rs:2436 before entering the clamped multistream path, and projection float encode allocates conversion/mixed buffers before frame-size selection at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\projection.rs:410 and C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\ropus\src\opus\multistream.rs:2272. A very large positive frame_size can therefore trigger infallible allocation and abort, whereas the upstream native paths clamp before scratch allocation. Expected: oversized capacity is clamped or rejected before allocation. Actual: wrapper/core staging allocates from the unvalidated value. Apply the codec maximum before every temporary allocation, use checked fallible allocation, and add 120-ms boundary plus oversized PLC/float/projection regression tests. This is distinct residual behavior from closed ROP-BUG-FLUX-00009, which covered only negative frame-size casts.

## Fix

<unfixed — raised only>

## Notes
