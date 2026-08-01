# ROP-BUG-FLUX-00054 — Info library command exits the embedding process on unknown query

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-tools-core/info-api
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-02, deltic:auto role=fix run=fix-20260801T230021Z-p56272-n121291000-c1 branch=task/bug-ROP-BUG-FLUX-00054-run-fix-20260801T230021Z-p56272-n121291000-c1 code=3400cf5 gate=manual) -> Closed (2026-08-02, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main 3528b9e; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main ac7ff8a. The crate advertises GUI/plugin use at /Users/md/language/ropus/ropus-tools-core/src/options.rs:1-6 and exports commands::info as Result<()> at commands/mod.rs:1-12. Unknown query handling at commands/info.rs:305-308,347-358 writes stderr and calls std::process::exit(2), bypassing caller recovery, cleanup, and the Result contract. Fix: return a typed UnknownQuery error or command outcome; let only ropusinfo main map it to exit code 2; add an in-process error test and a separate CLI exit-code test. Static review only; no command or test ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-02, independent two-eyes, host flux)

Covers the main observation plus the `ropusinfo` validation-order note. Fixed by the same
commit as `ROP-BUG-FLUX-00052` (`3400cf5`) — the query-collection-plan rework and the
exit-code fix were one change: `info()` now returns `Result` and only `ropusinfo/src/main.rs`
maps an unknown-query error to exit code 2, via the new `commands::validate_query_key`
export.

- Confirmed by reading the current tree: `grep -n process::exit
  ropus-tools-core/src/commands/info.rs` returns no matches — the library function no longer
  calls `std::process::exit`. `ropusinfo/src/main.rs` now calls
  `commands::validate_query_key(query)` before `commands::info(opts)` and only the CLI's
  `main` returns `ExitCode::from(2)`.
- Regression re-verified: `query_key_is_validated_without_opening_input` in
  `ropus-tools-core/src/commands/info.rs` asserts `info(...)` returns
  `Err` (not a process exit) for an unknown query against a nonexistent file, and that the
  error is available before any file I/O. `validate_query_key` did not exist pre-fix at
  `3400cf5~1` (compile error against the fix's own test), confirming the library previously
  had no typed-error path for this case. The test passes at the current tree.
- `cargo clippy -p ropus-tools-core --all-targets --locked -- -D warnings` clean; `cargo test
  -p ropus-tools-core -p ropusinfo --locked`: 129 passed, 0 failed.

### `ropusinfo` validation order (2026-07-31)

`/Users/md/language/ropus/ropus-tools-core/src/commands/info.rs:68-76,305-358`
opens and fully inspects the input before validating the query name. An unknown
query against a missing or unreadable file therefore reports an I/O error
instead of the documented query error. Parse to a typed query before any file
I/O, return the typed error through the library boundary, and let only
`ropusinfo` map it to exit 2. Static review at `origin/main` `6a312e1`; no
command or test ran.
