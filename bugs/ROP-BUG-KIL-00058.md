# ROP-BUG-KIL-00058 — Closed progress pipes can panic and abort or misreport encoding

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropusenc/cli-output
- **Raised:** 2026-08-22T11:25:51Z
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
- **State history:** Open (2026-08-22T11:25:51Z, raised via `deltic bugs new`) -> Fixed (2026-09-13T09:16:37Z, deltic:auto role=fix run=fix-20260913T082236Z-5e65a9c7 branch=task/bug-ROP-BUG-KIL-00058-run-fix-20260913T082236Z-5e65a9c7 code=9252c99f3df1de964b4a96219e63240f34ed20c4 gate=manual) -> Closed (2026-09-13T16:47:37Z, independent two-eyes verification model=codex@xhigh, verifier=CRUCIBLE, fixer=deltic:auto, fix=9252c99f3df1de964b4a96219e63240f34ed20c4)

## Observation

Static review at baseline a463e758 found infallible println/eprintln progress and banner writes in ropusenc/src/main.rs:292-307 and ropus-tools-core/src/commands/encode.rs:299-312,566-570. Rust standard output macros panic when a downstream pipe closes. A normal pipeline such as ropusenc input.wav -o output.opus piped to a consumer that exits early can therefore return a panic status; writes before commit discard the temporary output, while the final write happens after commit and can report failure even though the output was published (commit is at encode.rs:540-542). Expected: a closed diagnostic/progress pipe never panics or creates an ambiguous output result. Fix: use fallible locked writers through the shared UI layer, define BrokenPipe as graceful termination, and add regressions for closure before and after commit. No application or test execution was performed in this review pass.

## Fix

### Verification summary (2026-09-13)

- Re-ran `broken_progress_pipe_before_commit_preserves_destination_and_cleans_temp`; it passed, and the relevant package gates passed.
- A red control skipped the pre-commit progress flush; the regression returned `Ok` instead of the expected error and failed, so the fix was restored.

## Notes
