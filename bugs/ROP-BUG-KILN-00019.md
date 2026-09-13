# ROP-BUG-KILN-00019 — Lossless control accepts incomplete matching PCM outputs

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness-control/lossless-oracle
- **Raised:** 2026-08-16T07:50:15Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T142914Z-2d9312c2
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KILN-00019-run-verify-20260913T142914Z-2d9312c2
- **Owner base:** 45fe8d93b3b7f78ea399d27d23f2126562a13caf
- **Owner fingerprint:** sha256:61ef5ae8eeb169794fa7f1c6d7d6b69f87e822ee1da93a737c098ccf0fdce73a
- **Owner since:** 2026-09-13T14:29:14Z
- **Owner until:** 2026-09-13T16:29:14Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-16T07:50:15Z, raised via `deltic bugs new`) -> Fixed (2026-09-13T00:10:24Z, deltic:auto role=fix run=fix-20260912T235836Z-974d1259 branch=task/bug-ROP-BUG-KILN-00019-run-fix-20260912T235836Z-974d1259 code=cbb6717cb3939751f33866ad354d96a7c16dccc7 gate=manual)

## Observation

Static review at origin/main a97b6f11. The lossy control requires exactly TOTAL_FRAMES * FRAME_SIZE * CHANNELS samples at harness-control/tests/control_snr.rs:357-369, but the lossless path at lines 450-479 checks only equal child lengths, energy, divergence, and SNR. Matching nonempty truncated prefixes can satisfy every lossless assertion, so a shared early-success or publication regression in the child decoders is false-green. The closed ROP-BUG-FLUX-00033 note says a shared exact-length validator landed, but fix commit b05bb88 did not modify harness-control and the current check is absent; this record tracks the residual gap without rewriting the closed ledger entry. Expected: both control paths reject incomplete PCM. Fix: share one exact output-length validator, assert expected packet count and input frame alignment, and check nonempty packets directly rather than relying on a nonzero FNV offset. Static inspection and history only; no app, build, test, decoder, or harness ran.

## Fix

Implemented in `harness-control/tests/control_snr.rs` and integrated at code
commit `cbb6717cb3939751f33866ad354d96a7c16dccc7`.

- `expected_frame_count` now requires input PCM to align to
  `FRAME_SIZE * CHANNELS`; `encode_with_ropus` uses it instead of silently
  truncating a partial final frame.
- `assert_expected_packet_stream` requires the exact frame count and rejects
  every empty packet. Both lossy and lossless controls call it directly.
- `assert_exact_pcm_length` requires each decoder to produce exactly
  `TOTAL_FRAMES * FRAME_SIZE * CHANNELS` samples before energy or SNR checks.
  Both controls call it, so matching truncated PCM cannot pass the lossless
  oracle.
- Replaced the nonzero FNV-fingerprint checks with direct packet-shape checks
  and added `control_shape_oracles_reject_truncated_data` for the two failure
  shapes.

Verification:

- `$null | deltic timeout 180 cargo test -p ropus-harness-control --test control_snr control_shape_oracles_reject_truncated_data` — 1 passed, 0 failed.
- `$null | deltic timeout 900 cargo test -p ropus-harness-control --test control_snr ctrl_fixed_vs_float_classical_snr_lossless -- --nocapture` — 1 passed, 0 failed; lossless SNR 90.14 dB and 96,000 output samples.
- `$null | deltic timeout 900 cargo test -p ropus-harness-control --test control_snr ctrl_fixed_vs_float_classical_snr -- --nocapture` — 2 passed, 0 failed; lossy SNR 42.35 dB and 96,000 output samples.
- `cargo check -p ropus-harness-control`, `cargo fmt --all -- --check`, and `git diff --check` — passed.
- Red proof: temporarily made `assert_exact_pcm_length` a no-op; the focused shape test failed. The guard was restored before the green runs.
- Test setup fetched the pinned C reference and DNN weights with `cargo run -p fetch-assets -- reference` and `cargo run -p fetch-assets -- weights`.

## Notes
