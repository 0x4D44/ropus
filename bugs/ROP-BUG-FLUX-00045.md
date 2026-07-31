# ROP-BUG-FLUX-00045 — OpusHead parser accepts unsupported mappings and incompatible versions

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-tools-core/opus-head
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main ac7ff8a. /Users/md/language/ropus/ropus-tools-core/src/container/ogg.rs:175-190 checks only packet length and magic, then accepts every version, channel count, mapping family, and header shape. Consumers at /Users/md/language/ropus/ropus-tools-core/src/commands/decode.rs:136-165 and commands/info.rs:105-116 instantiate the family-0 mono/stereo path, so unsupported multistream or incompatible future headers can be misinterpreted. Fix: validate compatible version, nonzero supported channels, mapping family 0, exact family-0 shape, and reject unsupported families before decoding; add a malformed-header table. Static review only; no parser or test ran.

## Fix

<unfixed — raised only>

## Notes
