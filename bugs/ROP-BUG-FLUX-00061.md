# ROP-BUG-FLUX-00061 — CLI build provenance can be stale or identify an unrelated repository

- **State:** Open
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main b4e2c31. /Users/md/language/ropus/ropusenc/build.rs:5-16 promises fresh commit metadata but watches only ../.git/HEAD. In a linked worktree the repository-root .git is a file, so that watch is omitted; in a normal symbolic-HEAD clone, a same-branch commit updates the referenced ref rather than HEAD. The unrestricted git rev-parse at :23-35 can also walk upward and label vendored source with an unrelated consumer repository SHA. Cargo may therefore reuse stale BUILD_GIT_SHA and BUILD_TIMESTAMP values or embed the wrong repository identity. Fix: resolve the actual Git directory and symbolic-ref target, watch HEAD plus the target ref, verify the discovered top-level is this workspace, and emit unknown outside it. Add normal-clone, linked-worktree, detached-HEAD, same-branch-commit, packaged, and vendored-layout oracles. The same script is copied in the sibling CLIs, so centralize the corrected policy while fixing it. Static review only; no build or application ran.

## Fix

<unfixed — raised only>

## Notes
