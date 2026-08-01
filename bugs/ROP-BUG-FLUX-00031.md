# ROP-BUG-FLUX-00031 — Numeric CLI options allow huge work and arithmetic panics

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness/cli-validation
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T201842Z-p6993-n992405000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00031-run-fix-20260801T201842Z-p6993-n992405000-c1
- **Owner base:** 3aaf0a38663ec2fd4ad97da8f2437501ed2fc162
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T20:18:42Z
- **Owner until:** 2026-08-01T22:18:42Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at `origin/main` `d0ab87e`. The generic parser at `/Users/md/language/ropus/harness/src/cli.rs:9277` accepts any `i32`, then dispatch casts values to unsigned counts at `:10587` through `:10693`. `bench --iters -1` becomes 4,294,967,295 iterations; `torture --change-interval 0` reaches modulo zero at `:6991`; negative intervals become huge `usize` values; and unchecked duration/rate/channel values feed allocation arithmetic. `longsoak --sample-rate 0` also reaches division by zero. Fix: use command-specific typed parsers with supported Opus rates/channels, positive bounded durations and iterations, a nonzero change interval, explicit zero-only disable flags, and checked sample-count arithmetic. Add boundary/negative/overflow parser cases without running long workloads. Static review only; no command ran.

## Fix

<unfixed — raised only>

## Notes
