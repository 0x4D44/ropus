# ROP-BUG-FLUX-00047 — Reverse Ogg duration scan trusts header-shaped payload bytes

- **State:** Closed
- **Priority:** Should
- **Severity:** Low
- **Area:** ropus-tools-core/ogg-scan
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T220951Z-p81763-n296538000-c1 branch=task/bug-ROP-BUG-FLUX-00047-run-fix-20260801T220951Z-p81763-n296538000-c1 code=1d39892 gate=manual) -> Closed (2026-08-02, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main 3528b9e; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main ac7ff8a. /Users/md/language/ropus/ropus-tools-core/src/container/ogg.rs:324-379 accepts a reverse-scan candidate after checking only OggS, version zero, and serial. It does not validate lacing, full page extent, checksum, sequence, flags, or EOS. /Users/md/language/ropus/ropus-tools-core/src/commands/info.rs:118-165 trusts the resulting granule for duration and bitrate. Crafted trailing or payload bytes can therefore supply an arbitrary final position. Fix: use the validated page-metadata parser tracked by ROP-REQ-FLUX-00041, adopt an explicit CRC/EOS policy, reject final granule below pre_skip, and add fake-header cases. Static review only; no file or test ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-02, independent two-eyes, host flux)

Covers the main observation plus the `ropusinfo` final-page note.

- Fix commit `1d39892` rewrites `read_last_granule` in `ropus-tools-core/src/container/ogg.rs`
  to validate CRC, EOS, lacing extent, and page sequence before trusting a candidate's
  granule, replacing the prior header-shape-only check.
- Regression re-verified by construction: spliced the fix's new tests onto the pre-fix
  `read_last_granule` (keeping the pre-fix validation logic, adding only the CRC/constant
  helpers the tests need to build fixtures) at `1d39892~1` — 3 of 8 tests failed exactly as
  predicted: `read_last_granule_skips_bad_crc_candidate`,
  `read_last_granule_skips_valid_non_eos_trailer`, and
  `read_last_granule_skips_header_shaped_payload` all fabricated a granule of 999 instead of
  correctly falling back to 42. All 8 tests pass at the current tree.
- `cargo clippy -p ropus-tools-core --all-targets --locked -- -D warnings` clean; `cargo test
  -p ropus-tools-core --locked`: 129 lib + 22 integration passed, 0 failed.

### Additional `ropusinfo` final-page trigger (2026-07-31)

`/Users/md/language/ropus/ropus-tools-core/src/container/ogg.rs:346-373`
also accepts a matching non-EOS candidate and does not prove the candidate's
declared page extent or checksum before
`/Users/md/language/ropus/ropus-tools-core/src/commands/info.rs:118-165`
publishes its granule as the stream duration. Add fake `OggS`, truncated-page,
bad-CRC, and non-EOS trailing candidates to the shared validated-page oracle.
Static review at `origin/main` `6a312e1`; no command or test ran.
