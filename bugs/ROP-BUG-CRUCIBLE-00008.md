# ROP-BUG-CRUCIBLE-00008 — Fuzz sanity can pass without verified target results

- **State:** Open
- **Priority:** Must
- **Severity:** High
- **Area:** full-test/fuzz-gate
- **Raised:** 2026-08-14T15:50:23Z
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
- **State history:** Open (2026-08-14T15:50:23Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main bb54eb50. C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\fuzz.rs:149-173 sets FullSanity status to Pass solely from a zero process exit, while parsed target summaries at :169 may be empty or incomplete. C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\html.rs:634-657 then renders no fuzz target rows without changing the green status. A zero-exit no-op, truncated output, or wrapper/parser drift can therefore create a false-green release fuzz claim. Expected: green proves the declared target set built and every committed crash replay was assessed. Fix: compare summaries against manifest-discovered targets, validate allowed field states and the terminal result marker, fail closed on missing or duplicate rows, and add zero-output plus incomplete-output oracles. Static review only; no fuzz command, app, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
