# ROP-BUG-KILN-00008 — Fuzz runner reports all clear after fuzzer failure

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** tools/fuzz-runner
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

Observation: tools/fuzz_run.sh captures every cargo-fuzz exit status but the final exit decision checks only whether artifact files were found. A startup, sanitizer, invalid-option, signal, or runtime failure that creates no artifact is summarized with its nonzero status and then reported as All clear with exit zero. The documented no-diff option also only sets an unused variable. Expected: any child failure is non-green and no-diff changes target behavior or is rejected. Actual: failed or misconfigured campaigns can pass.

## Fix

<unfixed — raised only>

## Notes
