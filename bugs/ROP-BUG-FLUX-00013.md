# ROP-BUG-FLUX-00013 — Extension parsers trust caller counts beyond safe slice bounds

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus/opus-extensions
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T170535Z-p74388-n441287000-c1 branch=task/bug-ROP-BUG-FLUX-00013-run-fix-20260801T170535Z-p74388-n441287000-c1 code=d937f3ba844c7900e4170648c4c75520f8153ff1 gate=manual) -> Closed (2026-08-01, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main dc05a88; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static source inspection confirmed two safe public panic paths. opus_packet_extensions_parse at /Users/md/language/ropus/ropus/src/opus/repacketizer.rs:652 trusts nb_extensions rather than extensions.len() before indexing at :675. opus_packet_extensions_parse_ext at /Users/md/language/ropus/ropus/src/opus/extensions.rs:104 accepts negative per-frame counts into a wrapping prefix sum, and at :132-136 can cast a negative index or trust declared capacity beyond the output slice. The iterator also lacks a release check that len is within data.len(). Expected: inconsistent lengths, capacities, and counts return OPUS_BAD_ARG or OPUS_BUFFER_TOO_SMALL. Actual: safe inputs can panic. Static review only; no code was executed.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-01, independent two-eyes, host flux)

- Commit `d937f3b` adds length/capacity checks to `opus_packet_extensions_count_ext` and
  `opus_packet_extensions_parse_ext` (`ropus/src/opus/extensions.rs`): both now reject
  `len > data.len()`, `opus_packet_extensions_parse_ext` also validates `*nb_extensions` against
  `extensions.len()` and rejects negative per-frame counts with `checked_add` instead of the
  prior `wrapping_add` prefix sum.
- New tests in `ropus/src/opus/extensions.rs` (e.g. `parse_ext_rejects_negative_frame_count`)
  and existing `repacketizer.rs` tests (`test_packet_extensions_parse_buffer_too_small`,
  `test_packet_extensions_parse_rejects_short_output_slice`,
  `test_parse_negative_len_returns_bad_arg`) cover both public panic paths and the invalid
  iterator length.
- `cargo test -p ropus --locked`: 1888 passed, 0 failed.
