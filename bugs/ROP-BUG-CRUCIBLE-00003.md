# ROP-BUG-CRUCIBLE-00003 — Encoder size queries ignore mapping-family validity

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** capi/encoder-size
- **Raised:** 2026-08-14T14:26:19Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260913T030923Z-b691a808
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-CRUCIBLE-00003-run-fix-20260913T030923Z-b691a808
- **Owner base:** eab0924f97279ef4c472584a03fbc1d80f632443
- **Owner fingerprint:** -
- **Owner since:** 2026-09-13T03:09:23Z
- **Owner until:** 2026-09-13T05:09:23Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-14T14:26:19Z, raised via `deltic bugs new`)

## Observation

opus_multistream_surround_encoder_get_size discards mapping_family at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\ms_encoder.rs:205, and opus_projection_ambisonics_encoder_get_size does the same at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\projection.rs:221. Their constructors validate supported families and channel layouts, so unsupported families and impossible ambisonics channel counts receive positive allocation sizes before create/init rejects them. The upstream Xiph implementations derive stream/layout data from mapping_family and return zero when that validation fails. Expected: each size query accepts exactly the configuration domain supported by its matching constructor. Actual: invalid configurations return usable-looking sizes. Reuse pure constructor validation in both queries and add unsupported-family plus channel-layout boundary tests.

## Fix

<unfixed — raised only>

## Notes
