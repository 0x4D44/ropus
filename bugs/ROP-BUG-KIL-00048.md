# ROP-BUG-KIL-00048 — Decode alias guard can be raced into truncating the input

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** ropus-tools-core/path-safety
- **Raised:** 2026-08-22T07:33:52Z
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
- **State history:** Open (2026-08-22T07:33:52Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at HEAD f9a3871. commands/decode.rs:122-134 checks input/output identity before the full decode, then opens the caller path much later with File::create at :501-519 or through audio/wav.rs:85-90 and :143-148. A concurrent replacement of the destination with a symlink or hard link to the source after preflight makes the final create follow that alias and truncate the input; any write failure also destroys the previous destination. commands/encode.rs:103-176 already avoids this class by writing a same-directory create-new temporary and publishing by rename. Fix decode regular-file output through the same atomic-output abstraction, retaining stdout streaming; revalidate stable identity at publication if needed. Add a deterministic seam that replaces the destination after preflight and proves the source and prior destination survive both success and injected write failure. This is residual after closed ROP-BUG-FLUX-00055, whose direct/symlink/hard-link tests cover only aliases present during preflight. Static inspection only; no file, app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
