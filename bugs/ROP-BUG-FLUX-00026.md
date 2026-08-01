# ROP-BUG-FLUX-00026 — Benchmark measures Rust with unbounded trace instrumentation

- **State:** Fixed
- **Priority:** Must
- **Severity:** High
- **Area:** harness/benchmark-tracing
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T075011Z-p48083-n853583000-c1 branch=task/bug-ROP-BUG-FLUX-00026-run-fix-20260801T075011Z-p48083-n853583000-c1 code=a7dc551f0248eac8b617022fe69b460df405a5e1 gate=manual)

## Observation

Static review at `origin/main` `d0ab87e`. `/Users/md/language/ropus/harness/Cargo.toml:9` through `:14` unconditionally enables `trace-silk-encode`. The Rust trace sink at `/Users/md/language/ropus/ropus/src/lib.rs:173` through `:188` pushes every tuple into an unbounded process-global `Mutex<Vec<_>>`; the benchmark at `/Users/md/language/ropus/harness/src/cli.rs:2722` through `:2757` neither disables nor clears it. SILK encode timing therefore includes locking, allocation, payload copies, and continually growing retained memory. The C trace is capped, so the comparison is asymmetric. Expected: Stage 4 measures production codec work. Actual: it measures an instrumented Rust build against a differently instrumented C build. Fix: use uninstrumented dependencies/reference code for benchmarks and keep trace features in diagnostic-only targets; add a build-contract assertion that the benchmark artifact has tracing disabled. Static review only; no benchmark ran and runtime magnitude was not measured.

## Fix

<unfixed — raised only>

## Notes
