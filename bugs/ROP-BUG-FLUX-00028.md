# ROP-BUG-FLUX-00028 — Torture leak check counts its duration-sized input as leaked memory

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness/torture-memory
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T194939Z-p71483-n396912000-c1 branch=task/bug-ROP-BUG-FLUX-00028-run-fix-20260801T194939Z-p71483-n396912000-c1 code=627fb6d gate=manual) -> Closed (2026-08-01, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main dc05a88; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at `origin/main` `d0ab87e`. Torture captures its RSS baseline at `/Users/md/language/ropus/harness/src/cli.rs:6775` before allocating the entire duration's PCM at `:6780`. `generate_noise` at `:378` allocates `sample_rate * duration * channels` samples; the 1,800-second default therefore retains 172.8 MB mono or 345.6 MB stereo. End RSS is sampled at `:7491` while the PCM and other buffers are still live, then a 50 MB increase is labeled a possible leak at `:7521` through `:7531`. Expected: the check detects steady-state growth. Actual: normal fixed input storage deterministically dominates the leak delta and memory use grows with requested duration. Fix: generate frames incrementally, take the baseline after fixed setup/warmup, compare steady-state samples or post-teardown RSS, and test the leak classifier independently of fixed allocations. Static review only; no torture run or RSS measurement occurred.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-01, independent two-eyes, host flux)

- Fix commit `627fb6d` moves the torture RSS baseline to after fixed setup/warmup and
  reclassifies growth via a steady-state comparison, so the duration-sized PCM buffer no longer
  dominates the leak delta.
- New test `rss_growth_classifier_ignores_fixed_baseline_and_shrink` passes.
- `cargo clippy -p ropus-harness --all-targets --locked -- -D warnings` clean; `cargo test -p
  ropus-harness --locked`: every suite green, 0 failed.
