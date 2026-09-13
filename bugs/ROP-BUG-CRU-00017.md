# ROP-BUG-CRU-00017 — fb2k warnings-as-errors lint rejects existing page-index loop

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-fb2k/lint
- **Raised:** 2026-09-13T05:57:45Z
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
- **State history:** Open (2026-09-13T05:57:45Z, raised via `deltic bugs new --land` model=gpt-5.6-luna@max) -> Fixed (2026-09-13T06:01:23Z, deltic:auto role=fix run=fix-20260913T055817Z-e16ac468 branch=task/bug-ROP-BUG-CRU-00017-run-fix-20260913T055817Z-e16ac468 code=7540f7a9be1f0984840c2fe1efa0c528f7c9a51e gate=manual)

## Observation

cargo clippy -p ropus-fb2k --all-targets -- -D warnings fails on pre-existing ropus-fb2k/src/reader.rs:1193-1272 with clippy::while-let-loop and two clippy::manual-is-multiple-of diagnostics. The same command fails twice on the claimed origin base, while normal clippy passes with only those three warnings. This blocks the documented warnings-as-errors validation gate for unrelated changes.

Evidence fingerprint: `trunk-red:v1:ropus`


## Fix

<unfixed — raised only>

## Notes
