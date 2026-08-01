# ROP-BUG-FLUX-00045 — OpusHead parser accepts unsupported mappings and incompatible versions

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-tools-core/opus-head
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T215213Z-p43735-n092100000-c1 branch=task/bug-ROP-BUG-FLUX-00045-run-fix-20260801T215213Z-p43735-n092100000-c1 code=3669b4974cb4752b6040731d6a9c1dce00389899 gate=manual) -> Closed (2026-08-02, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main 3528b9e; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main ac7ff8a. /Users/md/language/ropus/ropus-tools-core/src/container/ogg.rs:175-190 checks only packet length and magic, then accepts every version, channel count, mapping family, and header shape. Consumers at /Users/md/language/ropus/ropus-tools-core/src/commands/decode.rs:136-165 and commands/info.rs:105-116 instantiate the family-0 mono/stereo path, so unsupported multistream or incompatible future headers can be misinterpreted. Fix: validate compatible version, nonzero supported channels, mapping family 0, exact family-0 shape, and reject unsupported families before decoding; add a malformed-header table. Static review only; no parser or test ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-02, independent two-eyes, host flux)

- Fix commit `3669b49` adds version/channel/mapping-family/shape validation to
  `parse_opus_head` in `ropus-tools-core/src/container/ogg.rs`.
- Regression re-verified by construction: spliced the fix's new tests onto the pre-fix
  `ogg.rs` at `3669b497~1` — `opus_head_parse_rejects_malformed_header_matrix` panicked
  (`case zero version must reject`), confirming the pre-fix parser accepted an incompatible
  header. All 3 new tests pass at the current tree.
- `cargo clippy -p ropus-tools-core --all-targets --locked -- -D warnings` clean; `cargo test
  -p ropus-tools-core --locked`: 129 lib + 22 integration passed, 0 failed.
