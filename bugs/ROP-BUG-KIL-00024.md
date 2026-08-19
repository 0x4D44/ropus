# ROP-BUG-KIL-00024 — DRED shim FFI lacks buffer-length and CTL contracts

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness-deep-plc/shims
- **Raised:** 2026-08-19T10:46:15Z
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
- **State history:** Open (2026-08-19T10:46:15Z, raised via `deltic bugs new`)

## Observation

Static review. The DRED test-shim FFI surface has no length or configuration contracts, unlike c/peek.c which validates every access via `peek_valid_range` (peek.c:94-97). Four facets: (1) harness-deep-plc/dred_encode_shim.c:103-139 and :156-167 — eight `copy_*`/`set_*` helpers `memcpy(n * sizeof(float))` with caller-supplied `int n`, no bound against the real field capacity and no negativity check (negative `n` promotes to a huge `size_t`); the `set_*` pair writes into the heap-allocated `DREDEnc`. (2) Output sizes are C-macro-driven with nothing tying them to the Rust constants sizing the destination buffers: `ropus_test_dred_ec_decode` (dred_encode_shim.c:190-218) memcpys `DRED_STATE_DIM`/`(DRED_NUM_REDUNDANCY_FRAMES/2)*(DRED_LATENT_DIM+1)` floats into arrays sized from ropus's own constants (tests/dred_decode_payload_diff.rs:165-166); same shape in dred_enc_shim.c:49-64, dred_dec_shim.c:64-77, c/burg_thunk.c:18-20. A reference-side constant bump becomes silent stack/heap corruption presenting as a bogus divergence, not a red assertion. (3) dred_encode_shim.c:253-265 — five `opus_encoder_ctl` setter returns ignored (only DRED_DURATION is checked and read back at :266-279); a rejected CTL leaves the C encoder differently configured from the Rust encoder it is differenced against, absorbed by the tests' tolerance envelopes; `err != OPUS_OK && enc != NULL` also leaks the encoder. (4) tests/dred_decode_payload_diff.rs:82 + :115 passes a shared borrow as `activity_mem.as_ptr() as *mut _` through the shim's non-const `unsigned char *` (dred_encode_shim.c:73) — UB if `dred_encode_silk_frame` writes; the sibling encode test correctly uses `let mut` + `as_mut_ptr()`. Fix: validate `n` against real capacities in every copy/set shim, add `_Static_assert`s (or out_len parameters) tying C write sizes to the header constants, assert all CTL returns, and make the Rust caller's buffer `mut`. Static inspection only; no build or test ran.

## Fix

<unfixed — raised only>

## Notes
