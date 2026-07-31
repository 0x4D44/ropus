# ROP-BUG-FLUX-00027 — Benchmark times C construction but excludes Rust construction

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness/benchmark-timing
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

Static review at `origin/main` `d0ab87e`. The C encode timer starts before `opus_encoder_create` and ends after destroy at `/Users/md/language/ropus/harness/src/cli.rs:2682` through `:2713`, while Rust construction and CTLs occur before its timer at `:2738` through `:2757`. C decode repeats the mismatch at `:2781` through `:2812` versus Rust at `:2837` through `:2859`. The comments and `/Users/md/language/ropus/wrk_journals/2026.05.11 - JRN - perf-recalibration.md:49` state that both sides construct once outside timing, so source contradicts the calibrated methodology. Fix: start C timers after successful construction/configuration and stop before destruction, ideally through one shared lifecycle helper; add an injected ordering test. Static review only; no benchmark ran, so ratio impact is unverified.

## Fix

<unfixed — raised only>

## Notes
