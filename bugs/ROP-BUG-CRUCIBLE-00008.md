# ROP-BUG-CRUCIBLE-00008 — Fuzz sanity can pass without verified target results

- **State:** Fixed
- **Priority:** Must
- **Severity:** High
- **Area:** full-test/fuzz-gate
- **Raised:** 2026-08-14T15:50:23Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T172043Z-dbd8eda0
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-CRUCIBLE-00008-run-verify-20260913T172043Z-dbd8eda0
- **Owner base:** dd67a22ba0c87480528e4f6bd1735e96ea39e4e3
- **Owner fingerprint:** sha256:a80295bb2036e698d1face448b381471972fb6028ea7d4c6c5f8dbc1594736e9
- **Owner since:** 2026-09-13T17:20:43Z
- **Owner until:** 2026-09-13T19:20:43Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-14T15:50:23Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-09-12T21:22:11Z, deltic:auto role=fix run=fix-20260912T210803Z-0eec838b branch=task/bug-ROP-BUG-CRUCIBLE-00008-run-fix-20260912T210803Z-0eec838b code=4ffb5a4b78ed7a76673a9ac3606743495ce4ada6 gate=manual)

## Observation

Static review at origin/main bb54eb50. C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\fuzz.rs:149-173 sets FullSanity status to Pass solely from a zero process exit, while parsed target summaries at :169 may be empty or incomplete. C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\html.rs:634-657 then renders no fuzz target rows without changing the green status. A zero-exit no-op, truncated output, or wrapper/parser drift can therefore create a false-green release fuzz claim. Expected: green proves the declared target set built and every committed crash replay was assessed. Fix: compare summaries against manifest-discovered targets, validate allowed field states and the terminal result marker, fail closed on missing or duplicate rows, and add zero-output plus incomplete-output oracles. Static review only; no fuzz command, app, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
