# ROP-BUG-FLUX-00023 — Corpus diff counts malformed or partially decoded streams as matches

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness/corpus-diff
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T192005Z-p39305-n112351000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00023-run-fix-20260801T192005Z-p39305-n112351000-c1
- **Owner base:** 8dadd242c1baa9ac722d73e44d6f3709bcd3cdfc
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T19:20:05Z
- **Owner until:** 2026-08-01T21:20:05Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at `origin/main` `d0ab87e`. `/Users/md/language/ropus/harness/src/bin_inner/corpus_diff.rs:185` discards the second Ogg packet without verifying `OpusTags`, so an audio packet in that position is silently omitted. After any successful audio packet, matching decoder errors at `:265` through `:280` break the loop and return `FileOutcome::Match` at `:305`, using the already accumulated nonzero samples. `RunStats` then counts the file as `decoded_and_compared` and can exit 0 at `:372` through `:390`. This contradicts the file's promise at `:7` through `:11` to decode every audio packet and the signed definition of `decoded_and_compared` as successful complete decode. Fix: validate the `OpusTags` magic, return a structural skip/non-green outcome for malformed or partially decoded streams, and count a match only after clean EOF. Add missing-tags and valid-prefix/malformed-tail cases. Static review only; no corpus, binary, build, or test ran.

## Fix

<unfixed — raised only>

## Notes
