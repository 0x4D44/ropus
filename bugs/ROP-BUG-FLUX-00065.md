# ROP-BUG-FLUX-00065 — Playback status width miscounts Unicode terminal cells

- **State:** Closed
- **Priority:** Should
- **Severity:** Low
- **Area:** ropusplay/terminal-layout
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-02, deltic:auto role=fix run=fix-20260802T003410Z-p49291-n070544000-c1 branch=task/bug-ROP-BUG-FLUX-00065-run-fix-20260802T003410Z-p49291-n070544000-c1 code=4977e79 gate=manual) -> Closed (2026-08-02, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main 66f0954; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main e5d7113. The signed player design requires the status line to fit the current terminal width. /Users/md/language/ropus/ropus-tools-core/src/commands/play.rs:578-608,669-686 instead uses chars().count() as display columns and truncates by Unicode scalar count. Wide CJK and emoji labels can therefore wrap past the requested width, while combining sequences are truncated too aggressively; the ASCII-only oracle at :1116-1139 validates the same incorrect metric. Fix: use a supported terminal display-width and grapheme-aware truncation policy, bound work to the visible prefix, and add wide, combining, emoji, narrow-terminal, and resize cases that assert display-cell width. Static review only; no terminal, player, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-02, independent two-eyes, host flux)

- Fix commit `4977e79` adds the `unicode-width` and `unicode-segmentation` crates
  (`ropus-tools-core/Cargo.toml`) and reworks `ropus-tools-core/src/commands/play.rs`'s
  status-line width and truncation logic to measure terminal cell width and truncate on
  grapheme-cluster boundaries instead of scalar `chars().count()`. New unit tests
  (`status_line_wide_labels_fit_terminal_cells`,
  `truncate_to_fit_preserves_combining_and_emoji_graphemes`,
  `status_line_resize_recomputes_unicode_budget`) exercise wide CJK, combining, emoji, and
  resize cases matching the observation.
- The width/truncation logic and its tests are both new in this commit (the pre-fix
  `play.rs` has neither), so the tests cannot exist or pass without the fix — confirmed by
  reading `git diff 4977e79~1 4977e79 -- ropus-tools-core/src/commands/play.rs`.
- `cargo test -p ropus-tools-core --lib commands::play`: 47 passed, 0 failed (includes the
  three new width/truncation cases plus all pre-existing `play` unit tests).
- `cargo clippy -p ropus-tools-core --all-targets --locked -- -D warnings`: clean.
