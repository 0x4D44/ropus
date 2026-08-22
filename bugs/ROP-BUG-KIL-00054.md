# ROP-BUG-KIL-00054 — Clap parse errors echo terminal controls from argv

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropusdec/cli-output
- **Raised:** 2026-08-22T09:55:50Z
- **Discovery source:** Agent
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
- **State history:** Open (2026-08-22T09:55:50Z, raised via `deltic bugs new`)

## Observation

Static review at `1c337e8751383e5e3a60009ce73e23283571edf0`. `ropusdec/src/main.rs:85-93`
parses with Clap's `get_matches()`. On failure, that API renders the error and exits before
`prelude::run()` can apply the repository's terminal-control escaping. `Cargo.lock:267-270`
pins `clap_builder` 4.6.0; its `src/error/format.rs:222-225,321-327,353-359`
interpolates rejected values and unknown arguments verbatim. An invalid `--gain` value or
unknown dash-leading filename containing ESC, OSC, CR, LF, or other controls therefore reaches
stderr raw and can alter terminal state or forge output. Expected: all terminal-facing
untrusted text uses the reversible escaping policy from `ropus-tools-core/src/ui.rs`. Actual:
Clap exits before that policy is reached. This is residual after closed
`ROP-BUG-FLUX-00051`, which covers command reporting and error chains after successful parsing.
Fix: add one shared safe Clap entry point that uses `try_get_matches`, renders parse failures
without trusting raw context strings, escapes attacker-controlled controls, and preserves only
known formatter styling; route every CLI through it and add invalid-value and unknown-argument
fixtures containing C0/C1, OSC, CR/LF, and bidi controls. Static source and locked-dependency
inspection only; no binary or test ran.

## Fix

<unfixed — raised only>

## Notes
