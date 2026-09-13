# ROP-BUG-KILN-00011 — foobar2000 SDK fetch authenticates only by byte length

- **State:** Fixed
- **Priority:** Should
- **Severity:** High
- **Area:** tools/fetch-fb2k-sdk
- **Raised:** 2026-08-13T17:17:39Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T150154Z-a7b08628
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KILN-00011-run-verify-20260913T150154Z-a7b08628
- **Owner base:** 5047e7f85b144cadb866255f66edd3cbf7e97734
- **Owner fingerprint:** sha256:dd946f60f13192bd8f55e5945e4339f619a7e7499f5c3329c4aad6bfc66ec1ef
- **Owner since:** 2026-09-13T15:01:54Z
- **Owner until:** 2026-09-13T17:01:54Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-13T17:17:39Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-12T22:58:17Z, deltic:auto role=fix run=fix-20260912T224610Z-86e5a93a branch=task/bug-ROP-BUG-KILN-00011-run-fix-20260912T224610Z-86e5a93a code=5906099df304b499f495f3cdc896d2fa77b30a8b gate=manual)

## Observation

Observation: tools/fetch-fb2k-sdk.ps1 downloads executable build headers and sources, checks only a fixed Content-Length, computes a SHA-256, prints it, and extracts without comparing it to a pin or signature. A same-length replacement at the endpoint is admitted into subsequent native builds. Expected: compare a pinned cryptographic digest or trusted signature before extraction. Actual: size is the sole authenticity decision.

## Fix

Integrated code commit `5906099df304b499f495f3cdc896d2fa77b30a8b` now pins the
official `SDK-2025-03-07.7z` SHA-256 and verifies it before extraction. With
`-Force`, extraction happens in a temporary staging directory; the existing
SDK is replaced only after the staged `foobar2000` and `pfc` directories pass
the layout check. Focused coverage lives in
`tools/test_fetch_fb2k_sdk.py`.

Validation evidence:

- Regression proof: mutating the SHA-256 guard to
  `$false -and $sha -ne $ExpectedSha256` made
  `test_same_length_wrong_archive_fails_before_replacing_sdk` fail on its own
  `assertIn("SHA-256 mismatch", output)` assertion after output reached the
  extraction step.
- After restoration, `python -m unittest -v tools.test_fetch_fb2k_sdk` passed
  the selected same-length tampered-archive test. It observed the mismatch,
  no extraction output, and an unchanged prior SDK marker.
- `python -m py_compile tools/test_fetch_fb2k_sdk.py` and PowerShell parser
  validation passed. A local `pwsh -NoProfile -File tools/fetch-fb2k-sdk.ps1
  -Force` run against the official archive also passed the pinned digest and
  extracted-layout checks.

## Notes
