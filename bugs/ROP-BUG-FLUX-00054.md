# ROP-BUG-FLUX-00054 — Info library command exits the embedding process on unknown query

- **State:** Open
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main ac7ff8a. The crate advertises GUI/plugin use at /Users/md/language/ropus/ropus-tools-core/src/options.rs:1-6 and exports commands::info as Result<()> at commands/mod.rs:1-12. Unknown query handling at commands/info.rs:305-308,347-358 writes stderr and calls std::process::exit(2), bypassing caller recovery, cleanup, and the Result contract. Fix: return a typed UnknownQuery error or command outcome; let only ropusinfo main map it to exit code 2; add an in-process error test and a separate CLI exit-code test. Static review only; no command or test ran.

## Fix

<unfixed — raised only>

## Notes
