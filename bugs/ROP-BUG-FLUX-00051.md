# ROP-BUG-FLUX-00051 — Untrusted metadata and paths can inject terminal control sequences

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-tools-core/terminal-output
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main ac7ff8a. OpusTags accepts arbitrary valid UTF-8 at /Users/md/language/ropus/ropus-tools-core/src/container/ogg.rs:42-114. Human-facing output interpolates vendor/comments and paths without control escaping at commands/info.rs:190-212, commands/decode.rs:155-159, and commands/play.rs:167,346-375,550-608. ESC, OSC, BEL, carriage return, newline, and C1 controls can alter or forge terminal output, especially in raw-mode playback status. Fix: centralize one TTY-safe single-line formatter for untrusted text and paths; preserve raw machine-query values only under an explicit non-TTY policy; test all control classes. Static review only; no terminal or test ran.

## Fix

<unfixed — raised only>

## Notes

### Additional `ropusplay` device-name sinks (2026-07-31)

Raw cpal device names and the requested name are also emitted at
`/Users/md/language/ropus/ropus-tools-core/src/commands/play.rs:266-299,314-338`.
Include them in the single-line control-escaping policy. The machine-facing
device-list format needs an explicit reversible encoding so embedded CR/LF or
terminal controls cannot create forged device rows while exact selection
remains possible. Add deterministic device names containing ESC, OSC, BEL,
CR/LF, and C1 controls. Static review at `origin/main` `e5d7113`; no device,
terminal, player, or test ran.
