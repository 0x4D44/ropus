# ROP-BUG-KIL-00053 — DRED decoder trusts zero sample rates and mutable latent counts

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus/opus-dred
- **Raised:** 2026-08-22T08:29:02Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T153137Z-494dadf4
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KIL-00053-run-verify-20260913T153137Z-494dadf4
- **Owner base:** 423fe3d80fb5787e654d4a503f5f7c6b91c1a746
- **Owner fingerprint:** sha256:683a01317c4ad669409905758ac71924fe7e2a341b5ba443893678595c4c2679
- **Owner since:** 2026-09-13T15:31:37Z
- **Owner until:** 2026-09-13T17:31:37Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T08:29:02Z, raised via `deltic bugs new` model=gpt-5.6-sol@max) -> Fixed (2026-09-13T07:32:09Z, deltic:auto role=fix run=fix-20260913T072248Z-cd39eaea branch=task/bug-ROP-BUG-KIL-00053-run-fix-20260913T072248Z-cd39eaea code=09268fca39046c1815c5def42523f3d6b599510c gate=manual)

## Observation

Static review at HEAD 3972b03. For a packet containing a DRED payload, OpusDREDDecoder::parse at ropus/src/opus/dred.rs:116-145 divides by sampling_rate without validating it, so sampling_rate zero panics. OpusDREDDecoder::process at :159-190 casts dred.nb_latents to usize and slices fixed latent/output arrays, while OpusDred exposes nb_latents and process_stage as public mutable fields at ropus/src/dnn/dred.rs:2366-2373. A safe caller can set a negative or oversized latent count with stage 1 and trigger a panic. Expected: invalid rates and state dimensions return OPUS_BAD_ARG without processing. Fix: validate the supported positive sample rates, enforce nb_latents within the fixed storage limit before casting, and add zero-rate plus negative/oversized-state tests. Existing ROP-REQ-FLUX-00014 covers unimplemented PCM reconstruction, not these argument bounds. Static inspection only; no code, app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
