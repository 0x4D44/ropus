# ROP-BUG-CRUCIBLE-00003 — Encoder size queries ignore mapping-family validity

- **State:** Closed
- **Priority:** Could
- **Severity:** Low
- **Area:** capi/encoder-size
- **Raised:** 2026-08-14T14:26:19Z
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
- **State history:** Open (2026-08-14T14:26:19Z, raised via `deltic bugs new`) -> Fixed (2026-09-13T03:16:11Z, deltic:auto role=fix run=fix-20260913T030923Z-b691a808 branch=task/bug-ROP-BUG-CRUCIBLE-00003-run-fix-20260913T030923Z-b691a808 code=5e28a369f43a655078477042d7cc03a311008ea6 gate=manual) -> Closed (2026-09-13T17:49:57Z, independent two-eyes verification model=codex@xhigh, verifier=CRUCIBLE, fixer=deltic:auto, fix=5e28a369f43a655078477042d7cc03a311008ea6)

## Observation

opus_multistream_surround_encoder_get_size discards mapping_family at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\ms_encoder.rs:205, and opus_projection_ambisonics_encoder_get_size does the same at C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-143801\capi\src\projection.rs:221. Their constructors validate supported families and channel layouts, so unsupported families and impossible ambisonics channel counts receive positive allocation sizes before create/init rejects them. The upstream Xiph implementations derive stream/layout data from mapping_family and return zero when that validation fails. Expected: each size query accepts exactly the configuration domain supported by its matching constructor. Actual: invalid configurations return usable-looking sizes. Reuse pure constructor validation in both queries and add unsupported-family plus channel-layout boundary tests.

## Fix

### Independent verification summary (2026-09-13)

- Re-ran `surround_encoder_size_matches_mapping_family_and_channel_domain` and `projection_encoder_size_matches_mapping_family_and_channel_domain`; the C-API gate passed 21 Rust tests and 1 C test.
- A red control disabled mapping-family/channel-domain validation; the invalid configuration assertions failed, and the fix was restored.

## Notes
