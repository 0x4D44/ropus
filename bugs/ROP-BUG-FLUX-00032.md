# ROP-BUG-FLUX-00032 — Stereo Phase-C trace maps overwrite earlier channel records

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness/phase-c-trace
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

Static review at `origin/main` `d0ab87e`. Phase-C trace comparison intentionally keys tuples only by `(boundary_id, iter)` at `/Users/md/language/ropus/harness/src/bin_inner/fuzz_repro_diff.rs:1628` through `:1636`, then collects them into `BTreeMap` at `:1663` through `:1670`. The C traced encoder invokes the inner function per channel and stamps the channel at `/Users/md/language/ropus/harness/silk_enc_api_traced.c:624` through `:632`; Rust emits the same per-call tuples with a `-1` channel sentinel at `/Users/md/language/ropus/ropus/src/silk/encoder.rs:7052`. In stereo, duplicate keys overwrite the earlier channel record on both sides, so a channel-0-only divergence can disappear. Fix: align by boundary, iteration, and occurrence ordinal, or propagate a real Rust channel; reject duplicate logical keys instead of lossy collection. Add a two-channel synthetic trace with divergence only in the first occurrence. Static review only; no repro binary ran.

## Fix

<unfixed — raised only>

## Notes
