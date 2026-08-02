# ROP-BUG-FLUX-00065 — Playback status width miscounts Unicode terminal cells

- **State:** Open
- **Priority:** Should
- **Severity:** Low
- **Area:** ropusplay/terminal-layout
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260802T003410Z-p49291-n070544000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00065-run-fix-20260802T003410Z-p49291-n070544000-c1
- **Owner base:** c684e7e0ed36c2e64057a2ec303e1449221a3760
- **Owner fingerprint:** -
- **Owner since:** 2026-08-02T00:34:10Z
- **Owner until:** 2026-08-02T02:34:10Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main e5d7113. The signed player design requires the status line to fit the current terminal width. /Users/md/language/ropus/ropus-tools-core/src/commands/play.rs:578-608,669-686 instead uses chars().count() as display columns and truncates by Unicode scalar count. Wide CJK and emoji labels can therefore wrap past the requested width, while combining sequences are truncated too aggressively; the ASCII-only oracle at :1116-1139 validates the same incorrect metric. Fix: use a supported terminal display-width and grapheme-aware truncation policy, bound work to the visible prefix, and add wide, combining, emoji, narrow-terminal, and resize cases that assert display-cell width. Static review only; no terminal, player, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
