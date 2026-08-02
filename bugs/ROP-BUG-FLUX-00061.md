# ROP-BUG-FLUX-00061 — CLI build provenance can be stale or identify an unrelated repository

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropusenc/build-provenance
- **Raised:** 2026-07-31
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-02, deltic:auto role=fix run=fix-20260801T235001Z-p20648-n265745000-c1 branch=task/bug-ROP-BUG-FLUX-00061-run-fix-20260801T235001Z-p20648-n265745000-c1 code=82f1c1fe3a11e9623f33d4ad4134235b9e37c992 gate=manual) -> Closed (2026-08-02, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main 66f0954; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main b4e2c31. /Users/md/language/ropus/ropusenc/build.rs:5-16 promises fresh commit metadata but watches only ../.git/HEAD. In a linked worktree the repository-root .git is a file, so that watch is omitted; in a normal symbolic-HEAD clone, a same-branch commit updates the referenced ref rather than HEAD. The unrestricted git rev-parse at :23-35 can also walk upward and label vendored source with an unrelated consumer repository SHA. Cargo may therefore reuse stale BUILD_GIT_SHA and BUILD_TIMESTAMP values or embed the wrong repository identity. Fix: resolve the actual Git directory and symbolic-ref target, watch HEAD plus the target ref, verify the discovered top-level is this workspace, and emit unknown outside it. Add normal-clone, linked-worktree, detached-HEAD, same-branch-commit, packaged, and vendored-layout oracles. The same script is copied in the sibling CLIs, so centralize the corrected policy while fixing it. Static review only; no build or application ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-02, independent two-eyes, host flux)

- Fix commit `82f1c1f` adds a centralized `ropus-tools-core/src/build_provenance.rs`
  module (`discover()`), used by all four `build.rs` scripts (`ropusenc`, `ropusdec`,
  `ropusinfo`, `ropusplay`), replacing the byte-identical unsafe `git rev-parse` scripts
  the bug and its notes describe. Confirmed by reading: each `build.rs` is now a 5-line
  delegator to the shared module.
- The regression suite `ropus-tools-core/tests/build_provenance.rs` (new file, added by
  the fix) is a strong fails-before-fix case: it directly includes the new module and did
  not exist prior to `82f1c1f`, so it could not compile or pass without the fix. Its four
  cases (`normal_clone_tracks_head_and_same_branch_ref`,
  `linked_worktree_resolves_common_ref_without_using_root_git_path`,
  `detached_head_still_reports_commit_and_watches_head`,
  `vendored_and_packaged_layouts_return_unknown`) exercise exactly the linked-worktree,
  same-branch-ref, and unrelated-ancestor-repository scenarios in the observation and its
  three per-CLI notes.
- `cargo test -p ropus-tools-core --test build_provenance`: 4 passed, 0 failed.
- `cargo clippy -p ropus-tools-core --all-targets --locked -- -D warnings`: clean.
- `cargo build --workspace`: clean, no warnings.

### Confirmed in `ropusdec` (2026-07-31)

`/Users/md/language/ropus/ropusdec/build.rs:1-36` is byte-identical to the
script originally cited above. It has the same missing linked-worktree and
symbolic-ref watches and the same unrestricted ancestor-repository lookup.
Treat this as one four-CLI fix and run the normal clone, linked worktree,
detached HEAD, packaged, and vendored-layout oracles against every binary.
Static review at `origin/main` `bfe19ba`; no build ran.

### Confirmed in `ropusinfo` (2026-07-31)

`/Users/md/language/ropus/ropusinfo/build.rs:1-36` is byte-identical to the
same provenance script. Include `ropusinfo` in the shared correction and the
normal-clone, linked-worktree, detached-HEAD, packaged, and vendored-layout
oracles. Static review at `origin/main` `6a312e1`; no build ran.

### Confirmed in `ropusplay` (2026-07-31)

`/Users/md/language/ropus/ropusplay/build.rs:1-36` is the fourth
byte-identical provenance script. The shared correction and normal-clone,
linked-worktree, detached-HEAD, packaged, and vendored-layout oracles must
cover all four CLI binaries. Static review at `origin/main` `e5d7113`; no
build ran.
