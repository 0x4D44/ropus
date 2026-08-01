# ROP-BUG-FLUX-00021 — Post-hoc child output cap does not bound memory

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** full-test/subprocess-output
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T191145Z-p32846-n335474000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00021-run-fix-20260801T191145Z-p32846-n335474000-c1
- **Owner base:** 09f2eb4308a735c3d7fcc6997bcef9bf753d53fb
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T19:11:45Z
- **Owner until:** 2026-08-01T21:11:45Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main d2a0fb9. Observation: `/Users/md/language/ropus/full-test/src/tests.rs:253` through `:275` uses `Command::output` to buffer complete stdout and stderr, copies both buffers into owned strings, and only then applies the advertised 1 MiB cap. A noisy 45-minute test or build can exhaust runner memory before truncation. The same unbounded capture pattern appears in `/Users/md/language/ropus/full-test/src/fuzz.rs:141`, `/Users/md/language/ropus/full-test/src/corpus.rs:20`, and `/Users/md/language/ropus/full-test/src/platform.rs:453`; `/Users/md/language/ropus/full-test/src/ietf_vectors.rs:301` similarly reads a complete tempfile before capping it. Expected: retained diagnostic output has a real bounded-memory contract. Fix: add one subprocess capture helper that drains both pipes concurrently into bounded prefix and tail buffers or tempfiles, size-check or stream coverage JSON, and read fetch logs through a bounded reader. Add an oracle that emits beyond the cap and proves retained memory and output stay bounded while preserving status and tail diagnostics. Static review only; no runner, build, tests, or harness executed.

## Fix

<unfixed — raised only>

## Notes
