# ROP-BUG-KIL-00028 — LPCNet drift-lock bounds are host-architecture-specific despite scalar-only claim

- **State:** Fixed
- **Priority:** Could
- **Severity:** Medium
- **Area:** harness-deep-plc/tests
- **Raised:** 2026-08-19T10:46:21Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T150937Z-72ec299d
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KIL-00028-run-verify-20260913T150937Z-72ec299d
- **Owner base:** f3963d17a22be1c90232f3c95c00db9b401eeb96
- **Owner fingerprint:** sha256:f685256c04a1407e573295bfe084b36ed9ea501ca1bb48473aa366ae87ba926a
- **Owner since:** 2026-09-13T15:09:37Z
- **Owner until:** 2026-09-13T17:09:37Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-19T10:46:21Z, raised via `deltic bugs new`) -> Fixed (2026-09-13T02:22:32Z, deltic:auto role=fix run=fix-20260913T020444Z-d46a930b branch=task/bug-ROP-BUG-KIL-00028-run-fix-20260913T020444Z-d46a930b code=cdd1b1f3d81794f4318a2a86016bf76cc4653dfb gate=manual)

## Observation

Static review. harness-deep-plc/config.h:26-28 claims "Keep SIMD off in this harness — scalar only", but that comment governs translation units, not xiph's header-only DNN vector kernels selected by compiler predefines. build.rs:338 defines `DISABLE_NEON` (ARM escape) and nothing for x86 — and the crate's own drift analysis proves the x86 SIMD kernel is live: tests/dred_lpcnet_feature_drift.rs:102-117 root-causes the locked feat-18 residual to `reference/dnn/vec_avx.h:486`'s `_mm_rcp_ps`. Consequence: the locked drift bounds (:118-140, including the 34-index `BIT_EXACT_FEATURE_INDICES` equality lock) are host-architecture artefacts — locked against SSE/AVX C output on x86 while an AArch64 host (forced scalar C) compares against different C numbers, so the same gate proves different properties per host, ungated and unrecorded. This is the same fault family as closed ROP-BUG-FLUX-00004 (Apple Silicon tier-2 SNR miss). Confidence: the doc contradiction and the in-repo root-cause note are verified by reading; per-arch runtime behaviour is inference (reference/ absent on this host). Fix: gate the locked constants by `cfg(target_arch)` with per-arch values (or force the scalar C path on x86 too, if xiph's vec.h dispatch allows), record which kernel set produced each lock, and correct the config.h comment. Static inspection only; no build or test ran.

## Fix

<unfixed — raised only>

## Notes
