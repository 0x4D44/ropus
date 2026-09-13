# ROP-BUG-KIL-00045 — Reverse Ogg duration scan trusts CRC-valid pages nested in payload

- **State:** Fixed
- **Priority:** Should
- **Severity:** High
- **Area:** ropus-tools-core/ogg-scan
- **Raised:** 2026-08-22T07:33:49Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T152739Z-73e442cb
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KIL-00045-run-verify-20260913T152739Z-73e442cb
- **Owner base:** 9330432cc1696c9e6d6068111c00c665ab1df6cc
- **Owner fingerprint:** sha256:bf85bf4d55531fdc8d1e69de390ca1fda40190a3bc534f1b67c3fd92bb928155
- **Owner since:** 2026-09-13T15:27:39Z
- **Owner until:** 2026-09-13T17:27:39Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T07:33:49Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-09-13T06:27:05Z, deltic:auto role=fix run=fix-20260913T061508Z-59d07029 branch=task/bug-ROP-BUG-KIL-00045-run-fix-20260913T061508Z-59d07029 code=c0a0ad64523b263b511dd601371bce037705b84d gate=manual)

## Observation

Static review at HEAD f9a3871. ropus-tools-core/src/container/ogg.rs:391-412 tests every byte in the final 128 KiB, while parse_duration_page at :417-476 validates only the local candidate extent and CRC. It does not prove that the candidate begins at a physical Ogg page boundary or belongs to the physical page sequence. A complete CRC-valid fake EOS page embedded in the final real page payload can therefore be encountered after the real page header and supply an arbitrary final granule to ropusinfo; repeated large candidates also make the bitwise CRC loop at :481-495 perform avoidable superlinear work. The regression at :598-605 invalidates the embedded candidate CRC, so it does not exercise this residual. Fix with a boundary-anchored linear page walk or equivalent provenance proof; add a CRC-valid nested EOS fixture and an adversarial candidate-density bound. This is residual after closed ROP-BUG-FLUX-00047, which added local extent/CRC/EOS checks but not boundary provenance. Static inspection only; no app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
