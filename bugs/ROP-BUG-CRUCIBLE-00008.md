# ROP-BUG-CRUCIBLE-00008 — Fuzz sanity can pass without verified target results

- **State:** Closed
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
- **State history:** Open (2026-08-14T15:50:23Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-09-12T21:22:11Z, deltic:auto role=fix run=fix-20260912T210803Z-0eec838b branch=task/bug-ROP-BUG-CRUCIBLE-00008-run-fix-20260912T210803Z-0eec838b code=4ffb5a4b78ed7a76673a9ac3606743495ce4ada6 gate=manual) -> Closed (2026-09-13T17:49:57Z, independent two-eyes verification model=codex@xhigh, verifier=CRUCIBLE, fixer=deltic:auto, fix=4ffb5a4b78ed7a76673a9ac3606743495ce4ada6)

## Observation

Static review at origin/main bb54eb50. C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\fuzz.rs:149-173 sets FullSanity status to Pass solely from a zero process exit, while parsed target summaries at :169 may be empty or incomplete. C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\html.rs:634-657 then renders no fuzz target rows without changing the green status. A zero-exit no-op, truncated output, or wrapper/parser drift can therefore create a false-green release fuzz claim. Expected: green proves the declared target set built and every committed crash replay was assessed. Fix: compare summaries against manifest-discovered targets, validate allowed field states and the terminal result marker, fail closed on missing or duplicate rows, and add zero-output plus incomplete-output oracles. Static review only; no fuzz command, app, test, or harness ran.

## Fix

### Independent verification summary (2026-09-13)

- Re-ran the complete, incomplete, duplicate, and malformed fuzz-sanity report regressions; `cargo test -p full-test --locked` passed all 244 tests.
- A red control relaxed zero-output or incomplete-target validation; the sanity oracle failed, and the fix was restored.

## Notes
