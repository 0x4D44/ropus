# ROP-BUG-FLUX-00004 — Apple Silicon neural PLC SNR misses tier-2 threshold

- **State:** Fixed
- **Priority:** Must
- **Severity:** Medium
- **Area:** harness-deep-plc/arm
- **Raised:** 2026-07-30
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
- **State history:** Open (2026-07-30, raised via `deltic bugs new`) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T045808Z-p81818-n066143000-c1 branch=task/bug-ROP-BUG-FLUX-00004-run-fix-20260801T045808Z-p81818-n066143000-c1 code=1c8b95c9144570a9dab887b12b58963e45752734 gate=manual)

## Observation

On aarch64-apple-darwin with the pinned reference and weights assets, cargo test -p ropus-harness-deep-plc --test tier2_snr --locked -- --nocapture reports SNR(rust vs C)=35.97 dB for 14 lost packets, below the required 50 dB; the lossless control is 90.14 dB. Forcing the C oracle scalar with DISABLE_NEON changes the failing result to 37.34 dB, so the failure is not fixed by disabling NEON. Expected the native Apple Silicon oracle to meet the existing 50 dB tier-2 threshold.

## Fix

<unfixed — raised only>

## Notes
