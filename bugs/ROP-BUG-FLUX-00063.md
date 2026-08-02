# ROP-BUG-FLUX-00063 — Strict info queries return partial estimates after fallback decode errors

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropusinfo/query-integrity
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-02, deltic:auto role=fix run=fix-20260802T001734Z-p11561-n537342000-c1 branch=task/bug-ROP-BUG-FLUX-00063-run-fix-20260802T001734Z-p11561-n537342000-c1 code=e70bb9c27c49258090ad97c8a0558584b09089a0 gate=manual) -> Closed (2026-08-02, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main 66f0954; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main 6a312e1. /Users/md/language/ropus/ropus-tools-core/src/commands/info.rs:118-165 enters a decode fallback when the reverse scan cannot supply a final granule. At :151-157 it prints a warning for a malformed nonempty Opus packet, omits that packet from slow_sample_count, and continues. Strict --query duration and --query bitrate then print the incomplete value and return success at :333-345. Scripts receive a plausible scalar with exit 0 even though the promised true sample count was not recovered. Fix: carry an explicit completeness state; strict scalar queries must return a typed failure on any packet/decode error, while human diagnostic output may show a clearly labelled estimate. Add a CRC-valid Ogg stream with no usable final granule and one invalid Opus audio packet; assert duration/bitrate fail nonzero and emit no scalar. Static review only; no info command, decoder, or test ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-02, independent two-eyes, host flux)

- Fix commit `e70bb9c` adds an explicit completeness state to
  `ropus-tools-core/src/commands/info.rs`'s decode fallback: strict `--query duration` and
  `--query bitrate` now return a typed failure on any packet/decode error instead of the
  incomplete scalar, while human diagnostic output may still show a labelled estimate. New
  regression coverage lands in `ropusinfo/tests/cli.rs`
  (`strict_duration_and_bitrate_reject_incomplete_decode_without_scalar`), exactly matching
  the CRC-valid/no-final-granule/malformed-packet fixture the observation calls for.
- Fails-before/passes-after re-verified directly: reverted `info.rs` to its pre-fix version
  (`e70bb9c~1`) and ran the new test — it failed with `duration must fail for an invalid
  fallback packet; stderr="warning: packet 0: malformed Opus packet\n"`, reproducing the
  bug (a plausible scalar despite the decode gap). Restored the fix and reran: `cargo test
  -p ropusinfo --test cli` — 8 passed, 0 failed.
- `cargo clippy -p ropus-tools-core -p ropusinfo --all-targets --locked -- -D warnings`:
  clean.
