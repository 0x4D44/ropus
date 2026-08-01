# ROP-BUG-FLUX-00051 — Untrusted metadata and paths can inject terminal control sequences

- **State:** Closed
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T223552Z-p68771-n967796000-c1 branch=task/bug-ROP-BUG-FLUX-00051-run-fix-20260801T223552Z-p68771-n967796000-c1 code=495b9ae gate=manual) -> Closed (2026-08-02, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main 3528b9e; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main ac7ff8a. OpusTags accepts arbitrary valid UTF-8 at /Users/md/language/ropus/ropus-tools-core/src/container/ogg.rs:42-114. Human-facing output interpolates vendor/comments and paths without control escaping at commands/info.rs:190-212, commands/decode.rs:155-159, and commands/play.rs:167,346-375,550-608. ESC, OSC, BEL, carriage return, newline, and C1 controls can alter or forge terminal output, especially in raw-mode playback status. Fix: centralize one TTY-safe single-line formatter for untrusted text and paths; preserve raw machine-query values only under an explicit non-TTY policy; test all control classes. Static review only; no terminal or test ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-02, independent two-eyes, host flux)

Covers the main observation plus the `ropusplay` device-name sinks note.

- Fix commit `495b9ae` adds `escape_terminal_text`/`escape_terminal_path`/
  `format_query_value` to `ropus-tools-core/src/ui.rs` (new module functions; C0/C1
  controls, line separators, and bidi overrides become `\u{NNNN}`, with backslash escaped
  for reversibility) and wires them into `info.rs`, `decode.rs`, and `play.rs` call sites,
  including device-name output.
- Regression re-verified by construction: the 2 new `play.rs` tests were spliced onto the
  pre-fix `play.rs` (keeping the old unescaped `format_status_line`/`format_device_list`
  bodies, with `ui.rs` brought in as pure additive functions) at `495b9ae~1` — both failed:
  `status_line_escapes_untrusted_track_labels` asserted no raw `\n`/ESC (found them), and
  `format_device_list_uses_reversible_control_escaping` showed the raw CR/LF/ESC/C1 bytes
  passing straight through unescaped. All 4 new tests (2 in `ui.rs`, 2 in `play.rs`) pass at
  the current tree.
- `cargo clippy -p ropus-tools-core --all-targets --locked -- -D warnings` clean; `cargo test
  -p ropus-tools-core --locked`: 129 lib + 22 integration passed, 0 failed.

### Additional `ropusplay` device-name sinks (2026-07-31)

Raw cpal device names and the requested name are also emitted at
`/Users/md/language/ropus/ropus-tools-core/src/commands/play.rs:266-299,314-338`.
Include them in the single-line control-escaping policy. The machine-facing
device-list format needs an explicit reversible encoding so embedded CR/LF or
terminal controls cannot create forged device rows while exact selection
remains possible. Add deterministic device names containing ESC, OSC, BEL,
CR/LF, and C1 controls. Static review at `origin/main` `e5d7113`; no device,
terminal, player, or test ran.
