# ROP-BUG-KIL-00052 — Public FEC queue accepts short vectors and overflows its fixed capacity

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus/dnn-fec
- **Raised:** 2026-08-22T08:28:50Z
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
- **State history:** Open (2026-08-22T08:28:50Z, raised via `deltic bugs new` model=gpt-5.6-sol@max)

## Observation

Static review at HEAD 3972b03. OpusDecoder::fec_add at ropus/src/opus/decoder.rs:663-672 forwards any Option slice. LPCNetPLCState::fec_add at ropus/src/dnn/lpcnet.rs:2340-2348 copies f[..NB_FEATURES] without checking the feature width and protects the fixed 104-entry queue only with debug_assert. A safe caller passing fewer than 20 features or adding the 105th feature panics; release builds have no queue-capacity guard. Expected: malformed feature widths and a full queue return a stable error without changing queue state. Fix: make the public operation checked, validate the exact feature width and fec_fill_pos before slicing, define overflow behavior, and add short-vector plus capacity-boundary tests. Static inspection only; no code, app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
