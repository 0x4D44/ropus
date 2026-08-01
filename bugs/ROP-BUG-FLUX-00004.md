# ROP-BUG-FLUX-00004 — Apple Silicon neural PLC SNR misses tier-2 threshold

- **State:** Open
- **Priority:** Must
- **Severity:** Medium
- **Area:** harness-deep-plc/arm
- **Raised:** 2026-07-30
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T045808Z-p81818-n066143000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00004-run-fix-20260801T045808Z-p81818-n066143000-c1
- **Owner base:** 1f23a752184feed4029a98505a3c227f353d4325
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T04:58:08Z
- **Owner until:** 2026-08-01T06:58:08Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-30, raised via `deltic bugs new`)

## Observation

On aarch64-apple-darwin with the pinned reference and weights assets, cargo test -p ropus-harness-deep-plc --test tier2_snr --locked -- --nocapture reports SNR(rust vs C)=35.97 dB for 14 lost packets, below the required 50 dB; the lossless control is 90.14 dB. Forcing the C oracle scalar with DISABLE_NEON changes the failing result to 37.34 dB, so the failure is not fixed by disabling NEON. Expected the native Apple Silicon oracle to meet the existing 50 dB tier-2 threshold.

## Fix

<unfixed — raised only>

## Notes
