# ROP-BUG-KIL-00053 — DRED decoder trusts zero sample rates and mutable latent counts

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus/opus-dred
- **Raised:** 2026-08-22T08:29:02Z
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
- **State history:** Open (2026-08-22T08:29:02Z, raised via `deltic bugs new` model=gpt-5.6-sol@max)

## Observation

Static review at HEAD 3972b03. For a packet containing a DRED payload, OpusDREDDecoder::parse at ropus/src/opus/dred.rs:116-145 divides by sampling_rate without validating it, so sampling_rate zero panics. OpusDREDDecoder::process at :159-190 casts dred.nb_latents to usize and slices fixed latent/output arrays, while OpusDred exposes nb_latents and process_stage as public mutable fields at ropus/src/dnn/dred.rs:2366-2373. A safe caller can set a negative or oversized latent count with stage 1 and trigger a panic. Expected: invalid rates and state dimensions return OPUS_BAD_ARG without processing. Fix: validate the supported positive sample rates, enforce nb_latents within the fixed storage limit before casting, and add zero-rate plus negative/oversized-state tests. Existing ROP-REQ-FLUX-00014 covers unimplemented PCM reconstruction, not these argument bounds. Static inspection only; no code, app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
