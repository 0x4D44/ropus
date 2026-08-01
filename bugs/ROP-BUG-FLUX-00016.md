# ROP-BUG-FLUX-00016 — Multistream decoder C CTL omits complexity requests

- **State:** Closed
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new`) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T171344Z-p80782-n124719000-c1 branch=task/bug-ROP-BUG-FLUX-00016-run-fix-20260801T171344Z-p80782-n124719000-c1 code=e05659c779c4d1ae354bcaa70d78bd415b6571f4 gate=manual) -> Closed (2026-08-01, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main dc05a88; fixer was a prior automated fix session, verifier is a different actor)

## Observation

A valid opus_multistream_decoder_ctl(st, OPUS_SET_COMPLEXITY(v)) or OPUS_GET_COMPLEXITY(out) call falls through to OPUS_UNIMPLEMENTED. capi/src/ctl_shim.c:422-463 routes gain and phase inversion but omits request codes 4010/4011; capi/src/ctl.rs:667-715 has no matching Rust arms. The underlying OpusMSDecoder already exposes set_complexity/get_complexity at ropus/src/opus/multistream.rs:1988-1997. Expected: the C ABI fans the setter across streams and returns the first stream's value, matching the vendored multistream API. Actual: both requests return OPUS_UNIMPLEMENTED. Fix: add both request codes to the C varargs switch and Rust dispatcher, validate 0..=10 before mutation, and add focused C-driven set/get/invalid-value coverage.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-01, independent two-eyes, host flux)

- Commit `e05659c` adds request codes 4010/4011 (`OPUS_SET_COMPLEXITY`/`OPUS_GET_COMPLEXITY`)
  to both `capi/src/ctl_shim.c`'s multistream decoder varargs switch and `capi/src/ctl.rs`'s
  Rust dispatcher, fanning the setter across streams and validating `0..=10` before mutation,
  matching `OpusMSDecoder::set_complexity/get_complexity`.
- New tests `multistream_decoder_complexity_ctl_round_trip_and_validation` (Rust, in
  `capi/src/ctl.rs`) and `c_ctl_multistream_decoder_complexity_round_trip_and_validation`
  (C-driven, `capi/tests/ms_decoder_ctl.rs`) both pass — satisfying the fix's own requirement
  for focused C-driven set/get/invalid-value coverage.
- `cargo clippy -p capi --all-targets --locked -- -D warnings` clean; `cargo test -p capi
  --locked`: 8 passed, 0 failed.
