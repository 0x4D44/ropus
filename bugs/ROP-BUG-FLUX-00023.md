# ROP-BUG-FLUX-00023 — Corpus diff counts malformed or partially decoded streams as matches

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness/corpus-diff
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T192005Z-p39305-n112351000-c1 branch=task/bug-ROP-BUG-FLUX-00023-run-fix-20260801T192005Z-p39305-n112351000-c1 code=a978c614b2d75330b33eaf814a78d00fa159f8cf gate=manual) -> Closed (2026-08-01, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main dc05a88; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at `origin/main` `d0ab87e`. `/Users/md/language/ropus/harness/src/bin_inner/corpus_diff.rs:185` discards the second Ogg packet without verifying `OpusTags`, so an audio packet in that position is silently omitted. After any successful audio packet, matching decoder errors at `:265` through `:280` break the loop and return `FileOutcome::Match` at `:305`, using the already accumulated nonzero samples. `RunStats` then counts the file as `decoded_and_compared` and can exit 0 at `:372` through `:390`. This contradicts the file's promise at `:7` through `:11` to decode every audio packet and the signed definition of `decoded_and_compared` as successful complete decode. Fix: validate the `OpusTags` magic, return a structural skip/non-green outcome for malformed or partially decoded streams, and count a match only after clean EOF. Add missing-tags and valid-prefix/malformed-tail cases. Static review only; no corpus, binary, build, or test ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-01, independent two-eyes, host flux)

- Fix commit `a978c61` makes `harness/src/bin_inner/corpus_diff.rs` validate the `OpusTags`
  magic and return a structural non-match for malformed or partially-decoded streams, counting
  a match only after clean EOF.
- New tests `opus_tags_magic_is_required`, `missing_opus_tags_is_malformed`,
  `valid_prefix_with_malformed_tail_is_not_a_match`,
  `matching_decoder_error_after_valid_prefix_is_malformed`, and
  `corpus_malformed_is_non_green_even_with_a_valid_prefix` pass, covering missing-tags and
  valid-prefix/malformed-tail cases.
- `cargo clippy -p ropus-harness --all-targets --locked -- -D warnings` clean; `cargo test -p
  ropus-harness --locked`: every suite green, 0 failed.
