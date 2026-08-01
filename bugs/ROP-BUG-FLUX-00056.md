# ROP-BUG-FLUX-00056 — Empty Ogg audio packets become fabricated PLC audio and TOC data

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-tools-core/empty-packets
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T231813Z-p13207-n706702000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00056-run-fix-20260801T231813Z-p13207-n706702000-c1
- **Owner base:** 7511b5e7ed97cd8092ed6b22e671ab63b29bb701
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T23:18:13Z
- **Owner until:** 2026-08-02T01:18:13Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main ac7ff8a. /Users/md/language/ropus/ropus-tools-core/src/commands/decode.rs:255-291 passes every packet payload to decode; the same code uses an empty slice for deliberate PLC, so a zero-length container packet synthesizes up to the requested frame size instead of failing. /Users/md/language/ropus/ropus-tools-core/src/audio/decode.rs:196-225 repeats the issue. commands/info.rs:145-153 maps missing TOC to byte zero and can decode the empty payload as PLC in slow summary mode, fabricating valid-looking data. RFC 7845 treats zero-octet audio packets as malformed. Fix: reject or explicitly flag empty container packets before codec and TOC paths; reserve empty slices for deliberate PLC only; add malformed-empty fixtures. Static review only; no decoder or test ran.

## Fix

<unfixed — raised only>

## Notes
