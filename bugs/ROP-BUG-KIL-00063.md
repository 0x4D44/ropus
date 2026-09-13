# ROP-BUG-KIL-00063 — Playback status overflows very narrow terminals

- **State:** Fixed
- **Priority:** Could
- **Severity:** Low
- **Area:** ropusplay/terminal-layout
- **Raised:** 2026-08-22T12:55:44Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T153652Z-ab5b0c4e
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KIL-00063-run-verify-20260913T153652Z-ab5b0c4e
- **Owner base:** 14df511e3ba8e92260f4171d8947c4c0310b4cbf
- **Owner fingerprint:** sha256:313e2639f519d706f4ed48e103fdf3596ecbb2eb01b3a1668c26b35f5a976fce
- **Owner since:** 2026-09-13T15:36:52Z
- **Owner until:** 2026-09-13T17:36:52Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T12:55:44Z, raised via `deltic bugs new`) -> Fixed (2026-09-13T09:52:27Z, deltic:auto role=fix run=fix-20260913T094216Z-e0af6809 branch=task/bug-ROP-BUG-KIL-00063-run-fix-20260913T094216Z-e0af6809 code=7d4e806110910f6ca3ab36316fb518e93d38f069 gate=manual)

## Observation

Static review at `d18b7aa7d7b60cc8799428e6785cf5be326fb1ce` found that
`D:\worktrees\ropus\20260822-REV-ROP-CDX@KILN-code-review-133528\ropus-tools-core\src\commands\play.rs:682-711` promises the interactive status line will
respect `cols`, but its narrow branch always emits a fixed prefix and suffix. For a one-track,
one-minute file those fields already occupy 20 terminal cells; when `cols < 20`,
`truncate_to_fit` at `D:\worktrees\ropus\20260822-REV-ROP-CDX@KILN-code-review-133528\ropus-tools-core\src\commands\play.rs:784-789` only removes the title and the returned line still exceeds the
terminal width. Very narrow terminals therefore wrap or repaint across lines. The decoding
line at `D:\worktrees\ropus\20260822-REV-ROP-CDX@KILN-code-review-133528\ropus-tools-core\src\commands\play.rs:189-194` has the same fixed-prefix underflow when its prefix alone exceeds `cols`.
Expected: every interactive render is at most the current terminal cell width, including zero
and tiny widths. Fix: use progressive layouts that drop duration, track count, and glyph when
their fixed fields do not fit, and add a property/table oracle over widths from zero through
the normal layout threshold plus resize cases. This is residual after closed
`ROP-BUG-FLUX-00065`, which fixed Unicode cell/grapheme measurement but tests no width below
40. Static source inspection only; no terminal, application, test, build, or exploratory
harness ran.

## Fix

<unfixed — raised only>

## Notes
