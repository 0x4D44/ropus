# ROP-BUG-FLUX-00021 — Post-hoc child output cap does not bound memory

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** full-test/subprocess-output
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T191145Z-p32846-n335474000-c1 branch=task/bug-ROP-BUG-FLUX-00021-run-fix-20260801T191145Z-p32846-n335474000-c1 code=ac7a438f34462858c0498f0d1b5f64e78b5f0b89 gate=manual) -> Closed (2026-08-01, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main dc05a88; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main d2a0fb9. Observation: `/Users/md/language/ropus/full-test/src/tests.rs:253` through `:275` uses `Command::output` to buffer complete stdout and stderr, copies both buffers into owned strings, and only then applies the advertised 1 MiB cap. A noisy 45-minute test or build can exhaust runner memory before truncation. The same unbounded capture pattern appears in `/Users/md/language/ropus/full-test/src/fuzz.rs:141`, `/Users/md/language/ropus/full-test/src/corpus.rs:20`, and `/Users/md/language/ropus/full-test/src/platform.rs:453`; `/Users/md/language/ropus/full-test/src/ietf_vectors.rs:301` similarly reads a complete tempfile before capping it. Expected: retained diagnostic output has a real bounded-memory contract. Fix: add one subprocess capture helper that drains both pipes concurrently into bounded prefix and tail buffers or tempfiles, size-check or stream coverage JSON, and read fetch logs through a bounded reader. Add an oracle that emits beyond the cap and proves retained memory and output stay bounded while preserving status and tail diagnostics. Static review only; no runner, build, tests, or harness executed.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-01, independent two-eyes, host flux)

- Commit `ac7a438` adds one shared subprocess-capture helper
  (`full-test/src/process_capture.rs`) that drains stdout/stderr concurrently into bounded
  prefix+tail buffers, applied across `tests.rs`, `fuzz.rs`, `corpus.rs`, and `platform.rs`;
  coverage JSON is now size-checked before read.
- New tests `child_status_and_both_streams_survive_bounded_capture`,
  `oversized_stream_keeps_bounded_prefix_and_tail`, and
  `oversized_coverage_file_is_rejected_before_read` pass, proving retained memory and output
  stay bounded while status/tail diagnostics survive.
- `cargo clippy -p full-test --all-targets --locked -- -D warnings` clean; `cargo test -p
  full-test --locked`: 238 passed, 0 failed.
