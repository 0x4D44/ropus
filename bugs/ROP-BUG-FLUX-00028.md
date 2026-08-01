# ROP-BUG-FLUX-00028 — Torture leak check counts its duration-sized input as leaked memory

- **State:** Fixed
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T194939Z-p71483-n396912000-c1 branch=task/bug-ROP-BUG-FLUX-00028-run-fix-20260801T194939Z-p71483-n396912000-c1 code=627fb6d gate=manual)

## Observation

Static review at `origin/main` `d0ab87e`. Torture captures its RSS baseline at `/Users/md/language/ropus/harness/src/cli.rs:6775` before allocating the entire duration's PCM at `:6780`. `generate_noise` at `:378` allocates `sample_rate * duration * channels` samples; the 1,800-second default therefore retains 172.8 MB mono or 345.6 MB stereo. End RSS is sampled at `:7491` while the PCM and other buffers are still live, then a 50 MB increase is labeled a possible leak at `:7521` through `:7531`. Expected: the check detects steady-state growth. Actual: normal fixed input storage deterministically dominates the leak delta and memory use grows with requested duration. Fix: generate frames incrementally, take the baseline after fixed setup/warmup, compare steady-state samples or post-teardown RSS, and test the leak classifier independently of fixed allocations. Static review only; no torture run or RSS measurement occurred.

## Fix

<unfixed — raised only>

## Notes
