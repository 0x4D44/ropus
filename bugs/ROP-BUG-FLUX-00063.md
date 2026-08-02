# ROP-BUG-FLUX-00063 — Strict info queries return partial estimates after fallback decode errors

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropusinfo/query-integrity
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260802T001734Z-p11561-n537342000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00063-run-fix-20260802T001734Z-p11561-n537342000-c1
- **Owner base:** 700867c2b2668f1cf877d175c05b0a5ebd24fc7a
- **Owner fingerprint:** -
- **Owner since:** 2026-08-02T00:17:34Z
- **Owner until:** 2026-08-02T02:17:34Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main 6a312e1. /Users/md/language/ropus/ropus-tools-core/src/commands/info.rs:118-165 enters a decode fallback when the reverse scan cannot supply a final granule. At :151-157 it prints a warning for a malformed nonempty Opus packet, omits that packet from slow_sample_count, and continues. Strict --query duration and --query bitrate then print the incomplete value and return success at :333-345. Scripts receive a plausible scalar with exit 0 even though the promised true sample count was not recovered. Fix: carry an explicit completeness state; strict scalar queries must return a typed failure on any packet/decode error, while human diagnostic output may show a clearly labelled estimate. Add a CRC-valid Ogg stream with no usable final granule and one invalid Opus audio packet; assert duration/bitrate fail nonzero and emit no scalar. Static review only; no info command, decoder, or test ran.

## Fix

<unfixed — raised only>

## Notes
