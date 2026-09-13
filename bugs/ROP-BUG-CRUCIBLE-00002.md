# ROP-BUG-CRUCIBLE-00002 — Projection decoder size query accepts impossible stream counts

- **State:** Fixed
- **Priority:** Could
- **Severity:** Low
- **Area:** capi/projection-size
- **Raised:** 2026-08-14T14:26:12Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T144908Z-e79ad037
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-CRUCIBLE-00002-run-verify-20260913T144908Z-e79ad037
- **Owner base:** df45dd194f5ae9ef65cffeb1ffef63414f237285
- **Owner fingerprint:** sha256:ed4300e6b48cabd9a77163bcd7e3a2d6f68978ac8ee633b29ecdef0dd6feeba9
- **Owner since:** 2026-09-13T14:49:08Z
- **Owner until:** 2026-09-13T16:49:08Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-14T14:26:12Z, raised via `deltic bugs new`) -> Fixed (2026-09-13T03:08:29Z, deltic:auto role=fix run=fix-20260913T030213Z-4e80ff37 branch=task/bug-ROP-BUG-CRUCIBLE-00002-run-fix-20260913T030213Z-4e80ff37 code=4efc3d4282c44912ff37bcf5330993558c413600 gate=manual)

## Observation

opus_projection_decoder_get_size validates only streams >= 1, coupled_streams <= streams, and channels <= 255 at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\projection.rs:428. It omits the documented streams <= 255 and streams + coupled_streams <= 255 limits used by the multistream decoder, then proj_dec_size_for performs unchecked c_int multiplication at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\projection.rs:56. For example, streams=256 and coupled_streams=0 returns a positive size although construction rejects the configuration; larger values can wrap in release or panic behind ffi_guard in debug. Expected: invalid dimensions and arithmetic overflow return zero consistently. Actual: callers can receive a plausible, negative, or build-dependent size. Share the constructor dimension validation, use checked arithmetic, and add 255/256, total-channel, and overflow boundary tests.

## Fix

<unfixed — raised only>

## Notes
