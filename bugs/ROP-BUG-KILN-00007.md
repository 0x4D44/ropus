# ROP-BUG-KILN-00007 — Asset fetch accepts an unpinned C reference checkout

- **State:** Closed
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
- **State history:** Open (2026-08-13T17:17:37Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-12T20:50:37Z, deltic:auto role=fix run=fix-20260912T204139Z-9f5cf2d3 branch=task/bug-ROP-BUG-KILN-00007-run-fix-20260912T204139Z-9f5cf2d3 code=3460ad377b0abec83a63ffc8afdd857324b6824a gate=manual) -> Closed (2026-09-13T17:49:57Z, independent two-eyes verification model=codex@xhigh, verifier=CRUCIBLE, fixer=deltic:auto, fix=3460ad377b0abec83a63ffc8afdd857324b6824a)

## Observation

Observation: fetch_reference returns success whenever reference/celt/bands.c exists, even if git HEAD differs from the pinned commit or cannot be read. Downstream builds and differential tests can then use an arbitrary or unknown C oracle while provisioning reports success. Expected: only the pinned commit is accepted unless an explicit override is chosen. Actual: mismatch is a warning with exit zero.

## Fix

### Independent verification summary (2026-09-13)

- Re-ran the fetch-assets reference validation tests; `cargo test -p fetch-assets --locked` passed both tests, including mismatched and unreadable existing-reference cases.
- A red control accepted an unpinned existing reference checkout; the direct validation regression failed, and the fix was restored.

## Notes
