# ROP-BUG-FLUX-00056 — Empty Ogg audio packets become fabricated PLC audio and TOC data

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-tools-core/empty-packets
- **Raised:** 2026-07-31
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-02, deltic:auto role=fix run=fix-20260801T231813Z-p13207-n706702000-c1 branch=task/bug-ROP-BUG-FLUX-00056-run-fix-20260801T231813Z-p13207-n706702000-c1 code=6d5e94ca025ad1954ee066f89d1c56c0aeaf1c8a gate=manual) -> Closed (2026-08-02, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main 3528b9e; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main ac7ff8a. /Users/md/language/ropus/ropus-tools-core/src/commands/decode.rs:255-291 passes every packet payload to decode; the same code uses an empty slice for deliberate PLC, so a zero-length container packet synthesizes up to the requested frame size instead of failing. /Users/md/language/ropus/ropus-tools-core/src/audio/decode.rs:196-225 repeats the issue. commands/info.rs:145-153 maps missing TOC to byte zero and can decode the empty payload as PLC in slow summary mode, fabricating valid-looking data. RFC 7845 treats zero-octet audio packets as malformed. Fix: reject or explicitly flag empty container packets before codec and TOC paths; reserve empty slices for deliberate PLC only; add malformed-empty fixtures. Static review only; no decoder or test ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-02, independent two-eyes, host flux)

- Fix commit `6d5e94c` rejects zero-length container audio packets before they reach the
  codec or TOC paths in `commands/decode.rs`, `audio/decode.rs`, and `commands/info.rs`, and
  adds `ropus-tools-core/src/container/ogg.rs` support for flagging them.
- Regression re-verified by construction: the fix's new end-to-end test
  (`empty_ogg_audio_packet_is_rejected_without_plc_or_toc_fabrication` in
  `tests/round_trip.rs`) was run against the pre-fix tree at `6d5e94ca~1` (an integration
  test, decoupled from the source split, so no splicing was needed) — it panicked
  (`container decode must reject empty Opus audio: ()`), confirming the empty packet decoded
  as fabricated PLC audio instead of being rejected. The test passes at the current tree.
- `cargo clippy -p ropus-tools-core --all-targets --locked -- -D warnings` clean; `cargo test
  -p ropus-tools-core --locked`: 129 lib + 22 integration passed, 0 failed.
