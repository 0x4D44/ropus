# ROP-REQ-KIL-00031 — Verify reference checkout provenance and mirrored-layout constants at build time

- **State:** Draft
- **Priority:** Should
- **Area:** harness-deep-plc/build
- **Raised:** 2026-08-19T10:46:33Z
- **Discovery source:** Agent
- **Implemented-by:** —
- **Satisfied-by:** —
- **Violated-by:** —
- **Depends-on:** —
- **Design:** —
- **Flow:** heavy
- **Claimed-by:** —
- **State history:** Draft (2026-08-19T10:46:33Z, raised via `deltic reqs new`)

## Statement

The harness build must verify at build time that the `reference/` checkout is the pinned commit and that every hand-mirrored layout/constant matches it. Today the pin exists only at fetch time — tools/fetch-assets/src/main.rs:99-112 prints and accepts a mismatched pre-existing HEAD — and the sole build-time drift detector is one FNV-1a fingerprint over the `dred_bits_table` block (harness-deep-plc/build.rs:58-70), which covers neither the harness's own copied function bodies nor any of the mirrored values the shims depend on (`CELTDecoder`/`silk_decoder_state` prefixes, `PLC_UPDATE_SAMPLES`, `DECODE_BUFFER_SIZE`, `NB_BANDS`, `FRAME_SIZE`, `NB_TOTAL_FEATURES`, `DRED_*` dims, `DEC_QFRAME_WIDTH`). ROP-BUG-KIL-00024 and ROP-BUG-KIL-00025 fix the acute instances; this req is the systemic guard: a build-time reference-commit check (warn-or-fail on mismatch) plus `_Static_assert`s/fingerprints tying each mirrored constant and struct prefix to the reference headers, so a reference update turns into a red build instead of silent misreads. Acceptance: deliberately perturbing a mirrored constant or the reference HEAD fails the build with an actionable message.

## Notes
