# ROP-BUG-KIL-00052 — Public FEC queue accepts short vectors and overflows its fixed capacity

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus/dnn-fec
- **Raised:** 2026-08-22T08:28:50Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T153107Z-374a28d1
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KIL-00052-run-verify-20260913T153107Z-374a28d1
- **Owner base:** cec5c64f4797e5edfcc33733a44b19fc8f8596af
- **Owner fingerprint:** sha256:15b20f781d6f445b18a8de8dc4fc84888619b509ec180ad4fc2760545a0707e8
- **Owner since:** 2026-09-13T15:31:07Z
- **Owner until:** 2026-09-13T17:31:07Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T08:28:50Z, raised via `deltic bugs new` model=gpt-5.6-sol@max) -> Fixed (2026-09-13T07:36:07Z, deltic:auto role=fix run=fix-20260913T072127Z-e1d5e2fc branch=task/bug-ROP-BUG-KIL-00052-run-fix-20260913T072127Z-e1d5e2fc code=3fd7739a39901394fa321119b739b049d3418f43 gate=manual)

## Observation

Static review at HEAD 3972b03. OpusDecoder::fec_add at ropus/src/opus/decoder.rs:663-672 forwards any Option slice. LPCNetPLCState::fec_add at ropus/src/dnn/lpcnet.rs:2340-2348 copies f[..NB_FEATURES] without checking the feature width and protects the fixed 104-entry queue only with debug_assert. A safe caller passing fewer than 20 features or adding the 105th feature panics; release builds have no queue-capacity guard. Expected: malformed feature widths and a full queue return a stable error without changing queue state. Fix: make the public operation checked, validate the exact feature width and fec_fill_pos before slicing, define overflow behavior, and add short-vector plus capacity-boundary tests. Static inspection only; no code, app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
