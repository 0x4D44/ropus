# ROP-BUG-CRUCIBLE-00002 — Projection decoder size query accepts impossible stream counts

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** capi/projection-size
- **Raised:** 2026-08-14T14:26:12Z
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
- **State history:** Open (2026-08-14T14:26:12Z, raised via `deltic bugs new`)

## Observation

opus_projection_decoder_get_size validates only streams >= 1, coupled_streams <= streams, and channels <= 255 at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\projection.rs:428. It omits the documented streams <= 255 and streams + coupled_streams <= 255 limits used by the multistream decoder, then proj_dec_size_for performs unchecked c_int multiplication at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\projection.rs:56. For example, streams=256 and coupled_streams=0 returns a positive size although construction rejects the configuration; larger values can wrap in release or panic behind ffi_guard in debug. Expected: invalid dimensions and arithmetic overflow return zero consistently. Actual: callers can receive a plausible, negative, or build-dependent size. Share the constructor dimension validation, use checked arithmetic, and add 255/256, total-channel, and overflow boundary tests.

## Fix

<unfixed — raised only>

## Notes
