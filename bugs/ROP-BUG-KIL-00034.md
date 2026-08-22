# ROP-BUG-KIL-00034 — fb2k reverse duration scan trusts nested pages and permits quadratic CRC work

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** ropus-fb2k/ogg-scan
- **Raised:** 2026-08-22T06:10:44Z
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
- **State history:** Open (2026-08-22T06:10:44Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Static review at HEAD 3e0f6c1. ropus-fb2k/src/reader.rs:1127-1145 tests every byte offset in the trailing 128 KiB, while parse_duration_page at :1151-1210 validates only the local candidate extent and CRC, not that the offset is a physical Ogg page boundary. A CRC-valid fake EOS page embedded in a real page payload can therefore supply an arbitrary final granule; many structurally valid candidates also drive the bit-at-a-time CRC loop at :1215-1229 into quadratic work. Expected: derive duration from boundary-proven pages in a one-pass or anchored scan, with a regression containing a CRC-valid nested fake page and an adversarial candidate-density case. ROP-BUG-FLUX-00037 covered only the earlier unchecked-candidate form; this is the residual boundary and complexity defect. Static review only; no app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
