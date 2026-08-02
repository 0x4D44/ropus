# ROP-BUG-FLUX-00063 — Strict info queries return partial estimates after fallback decode errors

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropusinfo/query-integrity
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-02, deltic:auto role=fix run=fix-20260802T001734Z-p11561-n537342000-c1 branch=task/bug-ROP-BUG-FLUX-00063-run-fix-20260802T001734Z-p11561-n537342000-c1 code=e70bb9c27c49258090ad97c8a0558584b09089a0 gate=manual)

## Observation

Static review at origin/main 6a312e1. /Users/md/language/ropus/ropus-tools-core/src/commands/info.rs:118-165 enters a decode fallback when the reverse scan cannot supply a final granule. At :151-157 it prints a warning for a malformed nonempty Opus packet, omits that packet from slow_sample_count, and continues. Strict --query duration and --query bitrate then print the incomplete value and return success at :333-345. Scripts receive a plausible scalar with exit 0 even though the promised true sample count was not recovered. Fix: carry an explicit completeness state; strict scalar queries must return a typed failure on any packet/decode error, while human diagnostic output may show a clearly labelled estimate. Add a CRC-valid Ogg stream with no usable final granule and one invalid Opus audio packet; assert duration/bitrate fail nonzero and emit no scalar. Static review only; no info command, decoder, or test ran.

## Fix

<unfixed — raised only>

## Notes
