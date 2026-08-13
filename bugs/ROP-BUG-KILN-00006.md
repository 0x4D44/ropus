# ROP-BUG-KILN-00006 — Legacy fuzz seed generator emits shifted encode inputs

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** tools/fuzz-seeds
- **Raised:** 2026-08-13T17:17:36Z
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
- **State history:** Open (2026-08-13T17:17:36Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Observation: tools/generate_fuzz_seeds.py documents and emits a six-byte encode header, while the current fuzz_encode, fuzz_roundtrip, and safety targets require and parse eight configuration bytes. The first two PCM bytes become VBR, FEC, DTX, and loss fields and are removed from audio. Expected: generated seeds encode the requested configuration under the live target grammar. Actual: their configuration and PCM are shifted; a second generator already carries the current format.

## Fix

<unfixed — raised only>

## Notes
