# ROP-BUG-KILN-00006 — Legacy fuzz seed generator emits shifted encode inputs

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** tools/fuzz-seeds
- **Raised:** 2026-08-13T17:17:36Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T145534Z-b42895da
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KILN-00006-run-verify-20260913T145534Z-b42895da
- **Owner base:** e883a1d8e2e1608ec3d802a8f2743da05da0ee52
- **Owner fingerprint:** sha256:fdfc6bafd40b2ce112b81938eed6f946292aab2c19e0023beab083e7d57f3516
- **Owner since:** 2026-09-13T14:55:34Z
- **Owner until:** 2026-09-13T16:55:34Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-13T17:17:36Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-12T22:00:01Z, deltic:auto role=fix run=fix-20260912T215101Z-135631f9 branch=task/bug-ROP-BUG-KILN-00006-run-fix-20260912T215101Z-135631f9 code=72e3e9fbcd4d9c785f9a63e49d3b681ef560ada4 gate=manual)

## Observation

Observation: tools/generate_fuzz_seeds.py documents and emits a six-byte encode header, while the current fuzz_encode, fuzz_roundtrip, and safety targets require and parse eight configuration bytes. The first two PCM bytes become VBR, FEC, DTX, and loss fields and are removed from audio. Expected: generated seeds encode the requested configuration under the live target grammar. Actual: their configuration and PCM are shifted; a second generator already carries the current format.

## Fix

Integrated code commit `72e3e9fbcd4d9c785f9a63e49d3b681ef560ada4` updates
`tools/generate_fuzz_seeds.py` to emit the live eight-byte encode prologue,
including VBR/FEC and DTX/loss controls. Generated encode and roundtrip seeds
now begin PCM at byte 8. The helper matches the newer
`tools/gen_fuzz_seeds.py` header builder, and focused coverage lives in
`tools/test_generate_fuzz_seeds.py`.

Validation evidence:

- Before the fix, `encode_config(...)` returned six bytes and a marker PCM
  sample began at offset 6; the live Rust parsers consume bytes 0–7 as config.
- Regression proof: the new header test failed when the emitted DTX/loss byte
  was deliberately replaced with zero.
- After restoration, `$null | deltic timeout 120 python -m unittest -v
  tools.test_generate_fuzz_seeds tools.test_integrity tools.test_checkpoint`
  passed all 18 tests.
- The legacy header matched `gen_fuzz_seeds.make_encode_header(...)` byte for
  byte for the same configuration, and Python compilation passed.

## Notes
