# ROP-BUG-FLUX-00061 — CLI build provenance can be stale or identify an unrelated repository

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropusenc/build-provenance
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T235001Z-p20648-n265745000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00061-run-fix-20260801T235001Z-p20648-n265745000-c1
- **Owner base:** 3c465a5819592cc3e4ceeab6ba65a36b87a7561a
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T23:50:01Z
- **Owner until:** 2026-08-02T01:50:01Z
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
