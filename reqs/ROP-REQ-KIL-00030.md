# ROP-REQ-KIL-00030 — Consolidate duplicated harness-deep-plc test support into tests/support

- **State:** Draft
- **Priority:** Should
- **Area:** harness-deep-plc/tests
- **Raised:** 2026-08-19T10:46:33Z
- **Discovery source:** Agent
- **Implemented-by:** —
- **Satisfied-by:** —
- **Violated-by:** —
- **Depends-on:** —
- **Design:** —
- **Flow:** light
- **Claimed-by:** —
- **State history:** Draft (2026-08-19T10:46:33Z, raised via `deltic reqs new`)

## Statement

harness-deep-plc's test-support code must live once under tests/support/ (the crate already uses the `#[path = "support/..."]` pattern for finite_oracle.rs). Today ~250 lines are copy-pasted per file: the RIFF/WAV parser appears verbatim in five tests (dred_bitrate_plumbing_diff.rs:57-108, dred_bitrate_plumbing_nonzero_diff.rs:89-140, dred_encode_payload_diff.rs:46-99, dred_integrated_encode.rs:44-95, dred_lpcnet_feature_drift.rs:370-423); `vectors_path` in six; `weights_or_skip` in seven with a live signature drift (`(tag: &str)` vs `()`); and four independently-written SNR helpers (currently equivalent, verified). The drift-prone duplication is the mechanism behind ROP-BUG-KIL-00021 needing a nine-site fix. Scope is within-crate; the cross-crate fixture centralization is separately tracked as ROP-REQ-KILN-00020. Acceptance: one shared module per helper, all tests importing it, no remaining per-file copies.

## Notes
