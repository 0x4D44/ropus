# ROP-BUG-KIL-00045 — Reverse Ogg duration scan trusts CRC-valid pages nested in payload

- **State:** Closed
- **Priority:** Should
- **Severity:** High
- **Area:** ropus-tools-core/ogg-scan
- **Raised:** 2026-08-22T07:33:49Z
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
- **State history:** Open (2026-08-22T07:33:49Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-09-13T06:27:05Z, deltic:auto role=fix run=fix-20260913T061508Z-59d07029 branch=task/bug-ROP-BUG-KIL-00045-run-fix-20260913T061508Z-59d07029 code=c0a0ad64523b263b511dd601371bce037705b84d gate=manual) -> Closed (2026-09-13T16:47:37Z, independent two-eyes verification model=codex@xhigh, verifier=CRUCIBLE, fixer=deltic:auto, fix=c0a0ad64523b263b511dd601371bce037705b84d)

## Observation

Static review at HEAD f9a3871. ropus-tools-core/src/container/ogg.rs:391-412 tests every byte in the final 128 KiB, while parse_duration_page at :417-476 validates only the local candidate extent and CRC. It does not prove that the candidate begins at a physical Ogg page boundary or belongs to the physical page sequence. A complete CRC-valid fake EOS page embedded in the final real page payload can therefore be encountered after the real page header and supply an arbitrary final granule to ropusinfo; repeated large candidates also make the bitwise CRC loop at :481-495 perform avoidable superlinear work. The regression at :598-605 invalidates the embedded candidate CRC, so it does not exercise this residual. Fix with a boundary-anchored linear page walk or equivalent provenance proof; add a CRC-valid nested EOS fixture and an adversarial candidate-density bound. This is residual after closed ROP-BUG-FLUX-00047, which added local extent/CRC/EOS checks but not boundary provenance. Static inspection only; no app, build, test, or harness ran.

## Fix

### Verification summary (2026-09-13)

- Re-ran `read_last_granule_skips_crc_valid_page_shaped_payload`; it passed, and the `ropus-tools-core` package gate passed all 197 tests.
- A red control changed the final-page seek from `Start(0)` to `End(0)`; the regression returned `None` instead of granule 42, and the fix was restored.

## Notes
