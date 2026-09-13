# ROP-BUG-CRUCIBLE-00003 — Encoder size queries ignore mapping-family validity

- **State:** Fixed
- **Priority:** Could
- **Severity:** Low
- **Area:** capi/encoder-size
- **Raised:** 2026-08-14T14:26:19Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T145033Z-028303e2
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-CRUCIBLE-00003-run-verify-20260913T145033Z-028303e2
- **Owner base:** 26359222879ad594bd302c29a316a77b85b35277
- **Owner fingerprint:** sha256:98aa0c7d0d1f2cc4764f4ffe4239f827805980594e70ef4e0e605afcc8b48197
- **Owner since:** 2026-09-13T14:50:33Z
- **Owner until:** 2026-09-13T16:50:33Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-14T14:26:19Z, raised via `deltic bugs new`) -> Fixed (2026-09-13T03:16:11Z, deltic:auto role=fix run=fix-20260913T030923Z-b691a808 branch=task/bug-ROP-BUG-CRUCIBLE-00003-run-fix-20260913T030923Z-b691a808 code=5e28a369f43a655078477042d7cc03a311008ea6 gate=manual)

## Observation

opus_multistream_surround_encoder_get_size discards mapping_family at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\ms_encoder.rs:205, and opus_projection_ambisonics_encoder_get_size does the same at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\projection.rs:221. Their constructors validate supported families and channel layouts, so unsupported families and impossible ambisonics channel counts receive positive allocation sizes before create/init rejects them. The upstream Xiph implementations derive stream/layout data from mapping_family and return zero when that validation fails. Expected: each size query accepts exactly the configuration domain supported by its matching constructor. Actual: invalid configurations return usable-looking sizes. Reuse pure constructor validation in both queries and add unsupported-family plus channel-layout boundary tests.

## Fix

<unfixed — raised only>

## Notes
