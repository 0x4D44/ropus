# ROP-REQ-KIL-00029 — Add a C differential gate for OpusDREDDecoder::process reconstructed features

- **State:** Draft
- **Priority:** Should
- **Area:** harness-deep-plc/tests
- **Raised:** 2026-08-19T10:46:27Z
- **Discovery source:** Agent
- **Implemented-by:** —
- **Satisfied-by:** —
- **Violated-by:** —
- **Depends-on:** —
- **Design:** —
- **Flow:** light
- **Claimed-by:** —
- **State history:** Draft (2026-08-19T10:46:27Z, raised via `deltic reqs new`)

## Statement

The DRED reconstruction chain must have a C differential gate for its final stage: `OpusDREDDecoder::process` / `fec_features`. Today harness-deep-plc gates the chain only in halves — `dred_decode_payload_diff.rs` covers `ec_decode` against C and `dred_rdovae_dec_diff.rs` covers `decode_qframe` on synthetic latents — while the composed `process()` output is compared against nothing (harness-deep-plc/src/lib.rs exposes `ropus_test_c_dred_parse` but no binding for the C `opus_dred_process`). The only value-level assertion on reconstructed features anywhere is the NaN-permissive nonzero check at tests/dred_integrated_encode.rs:433 (raised as ROP-BUG-KIL-00023). Acceptance: a shim exposing the C `opus_dred_process`, plus a differential test that feeds both sides the same parsed packet and bounds the reconstructed-feature drift with finite-oracle validation.

## Notes
