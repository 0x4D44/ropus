# ROP-BUG-KIL-00062 — Headless device-list tests bypass stdout invariants

- **State:** Open
- **Priority:** Should
- **Severity:** Low
- **Area:** ropusplay/tests
- **Raised:** 2026-08-22T12:55:38Z
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
- **State history:** Open (2026-08-22T12:55:38Z, raised via `deltic bugs new`)

## Observation

Static review at `d18b7aa7d7b60cc8799428e6785cf5be326fb1ce` found that both
accepted no-device branches in `D:\worktrees\ropus\20260822-REV-ROP-CDX@KILN-code-review-133528\ropusplay\tests\cli.rs:32-43,64-76` return before checking
stdout. On a headless host, removing the `!args.list_devices` banner guard at
`D:\worktrees\ropus\20260822-REV-ROP-CDX@KILN-code-review-133528\ropusplay\src\main.rs:127`, emitting an ANSI sequence before the structured no-device error,
or replacing enumeration with an implementation that always returns that accepted phrase can
leave the suite green. The first test forces `--quiet`; the second is the only non-quiet
banner oracle and skips its banner/ANSI assertions on exactly this branch. Expected: the
headless path still proves stdout is empty and control-free, while a deterministic nonempty
device seam proves exact one-name-per-line success output. Fix: assert the failure branch's
stdout invariants before returning and inject empty/nonempty device enumerators for mandatory
hardware-free coverage; keep real hardware enumeration manual or explicitly ignored. This is
a residual after closed `ROP-BUG-FLUX-00057`, whose fix narrowed accepted failures but did not
make these output assertions unconditional. Static source inspection only; no application,
test, build, device enumeration, or exploratory harness ran.

## Fix

<unfixed — raised only>

## Notes
