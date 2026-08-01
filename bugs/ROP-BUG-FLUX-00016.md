# ROP-BUG-FLUX-00016 — Multistream decoder C CTL omits complexity requests

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** capi/multistream-decoder-ctl
- **Raised:** 2026-07-31
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new`) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T171344Z-p80782-n124719000-c1 branch=task/bug-ROP-BUG-FLUX-00016-run-fix-20260801T171344Z-p80782-n124719000-c1 code=e05659c779c4d1ae354bcaa70d78bd415b6571f4 gate=manual)

## Observation

A valid opus_multistream_decoder_ctl(st, OPUS_SET_COMPLEXITY(v)) or OPUS_GET_COMPLEXITY(out) call falls through to OPUS_UNIMPLEMENTED. capi/src/ctl_shim.c:422-463 routes gain and phase inversion but omits request codes 4010/4011; capi/src/ctl.rs:667-715 has no matching Rust arms. The underlying OpusMSDecoder already exposes set_complexity/get_complexity at ropus/src/opus/multistream.rs:1988-1997. Expected: the C ABI fans the setter across streams and returns the first stream's value, matching the vendored multistream API. Actual: both requests return OPUS_UNIMPLEMENTED. Fix: add both request codes to the C varargs switch and Rust dispatcher, validate 0..=10 before mutation, and add focused C-driven set/get/invalid-value coverage.

## Fix

<unfixed — raised only>

## Notes
