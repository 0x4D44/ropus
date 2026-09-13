# ROP-BUG-KIL-00048 — Decode alias guard can be raced into truncating the input

- **State:** Fixed
- **Priority:** Should
- **Severity:** High
- **Area:** ropus-tools-core/path-safety
- **Raised:** 2026-08-22T07:33:52Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T152913Z-b7edd814
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KIL-00048-run-verify-20260913T152913Z-b7edd814
- **Owner base:** b8cbd598b19b5c7b4cb75ab4715c1c6370f66bc5
- **Owner fingerprint:** sha256:b649078658cdf159d1efc4184c8db0c07cab27a7296c96cb524a0b818d949bf3
- **Owner since:** 2026-09-13T15:29:13Z
- **Owner until:** 2026-09-13T17:29:13Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T07:33:52Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-09-13T06:56:22Z, deltic:auto role=fix run=fix-20260913T064335Z-3cad2e56 branch=task/bug-ROP-BUG-KIL-00048-run-fix-20260913T064335Z-3cad2e56 code=30db9faac35d8f8777fae84499d1bf4102c9a53a gate=manual)

## Observation

Static review at HEAD f9a3871. commands/decode.rs:122-134 checks input/output identity before the full decode, then opens the caller path much later with File::create at :501-519 or through audio/wav.rs:85-90 and :143-148. A concurrent replacement of the destination with a symlink or hard link to the source after preflight makes the final create follow that alias and truncate the input; any write failure also destroys the previous destination. commands/encode.rs:103-176 already avoids this class by writing a same-directory create-new temporary and publishing by rename. Fix decode regular-file output through the same atomic-output abstraction, retaining stdout streaming; revalidate stable identity at publication if needed. Add a deterministic seam that replaces the destination after preflight and proves the source and prior destination survive both success and injected write failure. This is residual after closed ROP-BUG-FLUX-00055, whose direct/symlink/hard-link tests cover only aliases present during preflight. Static inspection only; no file, app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
