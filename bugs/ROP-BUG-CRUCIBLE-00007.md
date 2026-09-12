# ROP-BUG-CRUCIBLE-00007 — Release preflight launches Unix-only wrappers on Windows

- **State:** Fixed
- **Priority:** Must
- **Severity:** High
- **Area:** full-test/windows-preflight
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
- **State history:** Open (2026-08-14T15:50:23Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-09-12T21:27:35Z, deltic:auto role=fix run=fix-20260912T210409Z-0ca7c43d branch=task/bug-ROP-BUG-CRUCIBLE-00007-run-fix-20260912T210409Z-0ca7c43d code=a57b406 gate=manual)

## Observation

Static review at origin/main bb54eb50 on the primary Windows platform. C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\fuzz.rs:259-266 launches timeout then bash, while C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\platform.rs:420-434 launches timeout then env. Native Windows has timeout.exe as an interactive delay utility and no env executable in this review environment, so the required non-quick release-preflight lanes cannot launch correctly on the platform promised by C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\wrk_docs\2026.04.19 - HLD - full-test-runner.md:33-34. Expected: release preflight works on native Windows without GNU wrappers. Fix: set environment variables through Command, implement timeout supervision in Rust, choose the platform script explicitly, and add cfg(windows) command-shape tests. Static inspection and system command lookup only; no app, test, or harness ran.

## Fix

Replaced the Unix-only `timeout`/`bash` and `timeout`/`env` command chains used by
release preflight. `full-test` now selects `tools/fuzz_run.ps1` through native
PowerShell on Windows, sets Cargo environment variables through `Command`, and
uses the shared Rust process supervisor with 300-second fuzz and 900-second
platform deadlines. The HTML expectation was updated for the native command
shape.

Focused regression coverage:

- `$null | deltic timeout 240 cargo test -p full-test --locked` — 238 passed.
- `cargo fmt --all -- --check` — passed.
- `powershell -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File tools/fuzz_run.ps1 --list` — passed and listed all 14 manifest targets.

Fails-before proof: temporarily restoring the old `timeout 300 bash` command
made `full_sanity_command_uses_native_powershell_without_unix_wrappers` fail
with the old command vector. Temporarily removing the `CARGO_TARGET_DIR` and
`RUSTFLAGS` assignments made `generic_command_sets_environment_on_command`
fail on the missing environment assertion. Both root-cause mutations were
restored before commit.

Fix provenance: local commit `de4d7e3`; integrated fix commit `a57b406`.

## Notes
