# ROP-BUG-KIL-00034 — fb2k reverse duration scan trusts nested pages and permits quadratic CRC work

- **State:** Fixed
- **Priority:** Should
- **Severity:** High
- **Area:** ropus-fb2k/ogg-scan
- **Raised:** 2026-08-22T06:10:44Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T152157Z-64e7633f
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KIL-00034-run-verify-20260913T152157Z-64e7633f
- **Owner base:** da292747611099f6f066eed810cd1704bcc4db1e
- **Owner fingerprint:** sha256:e90a845a32c2e3e03eb4d928034dd0b91fd09f30775d95d1d3be6f9c46bb74bc
- **Owner since:** 2026-09-13T15:21:57Z
- **Owner until:** 2026-09-13T17:21:57Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T06:10:44Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-13T05:05:47Z, deltic:auto role=fix run=fix-20260913T044247Z-1ac01d52 branch=task/bug-ROP-BUG-KIL-00034-run-fix-20260913T044247Z-1ac01d52 code=b8465a5a28b2eca8f22fa7c28eaacc72490f11d7 gate=manual)

## Observation

Static review at HEAD 3e0f6c1. ropus-fb2k/src/reader.rs:1127-1145 tests every byte offset in the trailing 128 KiB, while parse_duration_page at :1151-1210 validates only the local candidate extent and CRC, not that the offset is a physical Ogg page boundary. A CRC-valid fake EOS page embedded in a real page payload can therefore supply an arbitrary final granule; many structurally valid candidates also drive the bit-at-a-time CRC loop at :1215-1229 into quadratic work. Expected: derive duration from boundary-proven pages in a one-pass or anchored scan, with a regression containing a CRC-valid nested fake page and an adversarial candidate-density case. ROP-BUG-FLUX-00037 covered only the earlier unchecked-candidate form; this is the residual boundary and complexity defect. Static review only; no app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
