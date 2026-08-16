# ROP-BUG-KILN-00019 — Lossless control accepts incomplete matching PCM outputs

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness-control/lossless-oracle
- **Raised:** 2026-08-16T07:50:15Z
- **Discovery source:** Agent
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
- **State history:** Open (2026-08-16T07:50:15Z, raised via `deltic bugs new`)

## Observation

Static review at origin/main a97b6f11. The lossy control requires exactly TOTAL_FRAMES * FRAME_SIZE * CHANNELS samples at harness-control/tests/control_snr.rs:357-369, but the lossless path at lines 450-479 checks only equal child lengths, energy, divergence, and SNR. Matching nonempty truncated prefixes can satisfy every lossless assertion, so a shared early-success or publication regression in the child decoders is false-green. The closed ROP-BUG-FLUX-00033 note says a shared exact-length validator landed, but fix commit b05bb88 did not modify harness-control and the current check is absent; this record tracks the residual gap without rewriting the closed ledger entry. Expected: both control paths reject incomplete PCM. Fix: share one exact output-length validator, assert expected packet count and input frame alignment, and check nonempty packets directly rather than relying on a nonzero FNV offset. Static inspection and history only; no app, build, test, decoder, or harness ran.

## Fix

<unfixed — raised only>

## Notes
