# ROP-BUG-KILN-00011 — foobar2000 SDK fetch authenticates only by byte length

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** tools/fetch-fb2k-sdk
- **Raised:** 2026-08-13T17:17:39Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260912T224610Z-86e5a93a
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KILN-00011-run-fix-20260912T224610Z-86e5a93a
- **Owner base:** c962ee67997481a700875bdbac2bdd4fa25dfedd
- **Owner fingerprint:** -
- **Owner since:** 2026-09-12T22:46:10Z
- **Owner until:** 2026-09-13T00:46:10Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-13T17:17:39Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Observation: tools/fetch-fb2k-sdk.ps1 downloads executable build headers and sources, checks only a fixed Content-Length, computes a SHA-256, prints it, and extracts without comparing it to a pin or signature. A same-length replacement at the endpoint is admitted into subsequent native builds. Expected: compare a pinned cryptographic digest or trusted signature before extraction. Actual: size is the sole authenticity decision.

## Fix

<unfixed — raised only>

## Notes
