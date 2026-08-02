# ROP-BUG-FLUX-00064 — All-invalid tracks count as successful empty playback

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropusplay/decode-completeness
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-02, deltic:auto role=fix run=fix-20260802T002554Z-p32406-n239300000-c1 branch=task/bug-ROP-BUG-FLUX-00064-run-fix-20260802T002554Z-p32406-n239300000-c1 code=55a6de7 gate=manual) -> Closed (2026-08-02, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main 66f0954; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main e5d7113. /Users/md/language/ropus/ropus-tools-core/src/audio/decode.rs:160-235 skips every native or Opus packet decode failure and still returns Ok even when it emitted no audio. /Users/md/language/ropus/ropus-tools-core/src/commands/play.rs:132-151 treats that empty result as a successful track and resets consecutive_errors; :174-211,444-470 appends an empty sink and classifies its immediate drain as TrackFinished. LoopMode::Single and LoopMode::All can therefore re-decode a corrupt track forever, while Off exits zero without playing audio. Fix: return a typed failure unless at least one valid frame survives the defined pre-skip/completeness policy; explicitly decide truly zero-duration stream behavior; add header-only, all-invalid, and pre-skip-consumes-all fixtures across Off, All, and Single. Static review only; no decoder, player, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-02, independent two-eyes, host flux)

- Fix commit `55a6de7` makes `ropus-tools-core/src/audio/decode.rs`'s shared decode path
  return a typed failure when no samples survive packet recovery and Opus pre-skip
  trimming, instead of an empty `Ok`. New header-only, all-invalid, and pre-skip-consumes-
  all regression fixtures land in `ropus-tools-core/tests/round_trip.rs`
  (`playback_rejects_header_only_all_invalid_and_pre_skip_only_audio`).
- Fails-before/passes-after re-verified directly: reverted `decode.rs` to its pre-fix
  version (`55a6de7~1`) and ran the new test — it failed with `zero-playback fixture must
  fail`, reproducing the bug (empty decode reported as success). Restored the fix and
  reran: `cargo test -p ropus-tools-core --test round_trip` — 23 passed, 0 failed.
- `cargo clippy -p ropus-tools-core --all-targets --locked -- -D warnings`: clean.
