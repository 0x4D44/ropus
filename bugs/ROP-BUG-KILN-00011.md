# ROP-BUG-KILN-00011 — foobar2000 SDK fetch authenticates only by byte length

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** tools/fetch-fb2k-sdk
- **Raised:** 2026-08-13T17:17:39Z
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
- **State history:** Open (2026-08-13T17:17:39Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Observation: tools/fetch-fb2k-sdk.ps1 downloads executable build headers and sources, checks only a fixed Content-Length, computes a SHA-256, prints it, and extracts without comparing it to a pin or signature. A same-length replacement at the endpoint is admitted into subsequent native builds. Expected: compare a pinned cryptographic digest or trusted signature before extraction. Actual: size is the sole authenticity decision.

## Fix

<unfixed — raised only>

## Notes
