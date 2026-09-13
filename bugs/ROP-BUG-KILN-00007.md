# ROP-BUG-KILN-00007 — Asset fetch accepts an unpinned C reference checkout

- **State:** Fixed
- **Priority:** Must
- **Severity:** High
- **Area:** tools/fetch-assets
- **Raised:** 2026-08-13T17:17:37Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T171849Z-c75da419
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KILN-00007-run-verify-20260913T171849Z-c75da419
- **Owner base:** be5671a69a0e0784671c8a31d5321f8c76439d21
- **Owner fingerprint:** sha256:ba5335e07da95cea32d51c64dd4303e675065bd591154616edffa7f54cb437ea
- **Owner since:** 2026-09-13T17:18:49Z
- **Owner until:** 2026-09-13T19:18:49Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-13T17:17:37Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-12T20:50:37Z, deltic:auto role=fix run=fix-20260912T204139Z-9f5cf2d3 branch=task/bug-ROP-BUG-KILN-00007-run-fix-20260912T204139Z-9f5cf2d3 code=3460ad377b0abec83a63ffc8afdd857324b6824a gate=manual)

## Observation

Observation: fetch_reference returns success whenever reference/celt/bands.c exists, even if git HEAD differs from the pinned commit or cannot be read. Downstream builds and differential tests can then use an arbitrary or unknown C oracle while provisioning reports success. Expected: only the pinned commit is accepted unless an explicit override is chosen. Actual: mismatch is a warning with exit zero.

## Fix

<unfixed — raised only>

## Notes
