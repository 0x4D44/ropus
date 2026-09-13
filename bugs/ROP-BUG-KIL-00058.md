# ROP-BUG-KIL-00058 — Closed progress pipes can panic and abort or misreport encoding

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropusenc/cli-output
- **Raised:** 2026-08-22T11:25:51Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260913T082236Z-5e65a9c7
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KIL-00058-run-fix-20260913T082236Z-5e65a9c7
- **Owner base:** 6fd4004c380e9d067164b5785778d703dffe0a53
- **Owner fingerprint:** -
- **Owner since:** 2026-09-13T08:22:36Z
- **Owner until:** 2026-09-13T10:22:36Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T11:25:51Z, raised via `deltic bugs new`)

## Observation

Static review at baseline a463e758 found infallible println/eprintln progress and banner writes in ropusenc/src/main.rs:292-307 and ropus-tools-core/src/commands/encode.rs:299-312,566-570. Rust standard output macros panic when a downstream pipe closes. A normal pipeline such as ropusenc input.wav -o output.opus piped to a consumer that exits early can therefore return a panic status; writes before commit discard the temporary output, while the final write happens after commit and can report failure even though the output was published (commit is at encode.rs:540-542). Expected: a closed diagnostic/progress pipe never panics or creates an ambiguous output result. Fix: use fallible locked writers through the shared UI layer, define BrokenPipe as graceful termination, and add regressions for closure before and after commit. No application or test execution was performed in this review pass.

## Fix

<unfixed — raised only>

## Notes
