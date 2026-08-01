# ROP-BUG-FLUX-00031 — Numeric CLI options allow huge work and arithmetic panics

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness/cli-validation
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T201842Z-p6993-n992405000-c1 branch=task/bug-ROP-BUG-FLUX-00031-run-fix-20260801T201842Z-p6993-n992405000-c1 code=2c7fed8 gate=manual) -> Closed (2026-08-01, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main dc05a88; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at `origin/main` `d0ab87e`. The generic parser at `/Users/md/language/ropus/harness/src/cli.rs:9277` accepts any `i32`, then dispatch casts values to unsigned counts at `:10587` through `:10693`. `bench --iters -1` becomes 4,294,967,295 iterations; `torture --change-interval 0` reaches modulo zero at `:6991`; negative intervals become huge `usize` values; and unchecked duration/rate/channel values feed allocation arithmetic. `longsoak --sample-rate 0` also reaches division by zero. Fix: use command-specific typed parsers with supported Opus rates/channels, positive bounded durations and iterations, a nonzero change interval, explicit zero-only disable flags, and checked sample-count arithmetic. Add boundary/negative/overflow parser cases without running long workloads. Static review only; no command ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-01, independent two-eyes, host flux)

- Fix commit `2c7fed8` replaces the generic `i32`-then-cast CLI parsing with command-specific
  typed parsers enforcing supported Opus rates/channels, positive bounded durations/iterations,
  a nonzero change interval, and checked sample-count arithmetic.
- New tests `rejects_invalid_numeric_ranges` and `checked_sample_count_handles_negative_overflow_and_cap`
  pass.
- `cargo clippy -p ropus-harness --all-targets --locked -- -D warnings` clean; `cargo test -p
  ropus-harness --locked`: every suite green, 0 failed.
