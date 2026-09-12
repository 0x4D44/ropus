# ROP-BUG-KILN-00006 — Legacy fuzz seed generator emits shifted encode inputs

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** tools/fuzz-seeds
- **Raised:** 2026-08-13T17:17:36Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260912T215101Z-135631f9
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KILN-00006-run-fix-20260912T215101Z-135631f9
- **Owner base:** 77b72833b5013065a472b47bcaaec688f61a2269
- **Owner fingerprint:** -
- **Owner since:** 2026-09-12T21:51:01Z
- **Owner until:** 2026-09-12T23:51:01Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-13T17:17:36Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Observation: tools/generate_fuzz_seeds.py documents and emits a six-byte encode header, while the current fuzz_encode, fuzz_roundtrip, and safety targets require and parse eight configuration bytes. The first two PCM bytes become VBR, FEC, DTX, and loss fields and are removed from audio. Expected: generated seeds encode the requested configuration under the live target grammar. Actual: their configuration and PCM are shifted; a second generator already carries the current format.

## Fix

<unfixed — raised only>

## Notes
