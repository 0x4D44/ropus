# ROP-BUG-FLUX-00001 — Workspace rustfmt gate fails on ropus-fb2k roundtrip tests

- **State:** Closed
- **Priority:** Must
- **Severity:** Low
- **Area:** validation/ropus-fb2k
- **Raised:** 2026-07-30
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
- **State history:** Open (2026-07-30, raised via `deltic bugs new`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T205437Z-p43457-n353831000-c1 branch=task/bug-ROP-BUG-FLUX-00001-run-fix-20260730T205437Z-p43457-n353831000-c1 code=4ced261 gate=manual) -> Closed (2026-07-31, independent two-eyes verification on host KILN, model=claude, at origin/main 6ccb736; fixer was Codex, verifier is a different actor and a different machine)

## Observation

Observed on macOS in a clean task worktree at origin/main c2658d87e44886ce4210be3fae798fe6913cd40e. Reproduce: cargo fmt --all -- --check. Expected: exit 0 with no diff. Actual: exit 1 and rustfmt diffs throughout ropus-fb2k/tests/roundtrip.rs, first at line 432. Repeated once unchanged.

## Fix

Commit `4ced261` applied the workspace rustfmt output to `ropus-fb2k/tests/roundtrip.rs`
(19 insertions, 105 deletions — call-argument reflow only, no semantic change).

### Verification summary (2026-07-31, independent two-eyes, host KILN / Windows x86_64)

Verified at `origin/main` 6ccb736 by an actor other than the fixer.

- **Fails before.** Extracted `ropus-fb2k/tests/roundtrip.rs` at `4ced261^` together with its
  `tests/common/mod.rs` sibling and ran `rustfmt --edition 2024 --check`: exit 1 with 248 lines
  of diff, **first diff at line 432** — matching the recorded observation exactly.
- **Passes after.** `cargo fmt --all -- --check` on 6ccb736: exit 0, no diff.
- **Root cause addressed.** The defect was unformatted committed source, not a rustfmt
  configuration problem; reformatting the file is the correct and minimal fix.
- **Regression coverage.** The repo's own formatting gate is the regression test:
  `cargo fmt --all -- --check` is the first step of every component gate and of the fallback
  gate in `.deltic-integrate.toml`, so a recurrence blocks integration.
- **No semantic drift from the reformat.** `cargo clippy -p ropus-fb2k --all-targets --locked
  -- -D warnings` green; `cargo test -p ropus-fb2k --locked` green (33 unit + 35 roundtrip
  tests pass, 0 failed).

## Notes
