# ROP-BUG-KILN-00007 — Asset fetch accepts an unpinned C reference checkout

- **State:** Open
- **Priority:** Must
- **Severity:** High
- **Area:** tools/fetch-assets
- **Raised:** 2026-08-13T17:17:37Z
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
- **State history:** Open (2026-08-13T17:17:37Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Observation: fetch_reference returns success whenever reference/celt/bands.c exists, even if git HEAD differs from the pinned commit or cannot be read. Downstream builds and differential tests can then use an arbitrary or unknown C oracle while provisioning reports success. Expected: only the pinned commit is accepted unless an explicit override is chosen. Actual: mismatch is a warning with exit zero.

## Fix

<unfixed — raised only>

## Notes
