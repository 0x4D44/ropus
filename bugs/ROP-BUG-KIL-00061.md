# ROP-BUG-KIL-00061 — ropusplay CLI tests can wait forever on child processes

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropusplay/tests
- **Raised:** 2026-08-22T12:55:34Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T153555Z-087971ad
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KIL-00061-run-verify-20260913T153555Z-087971ad
- **Owner base:** a14e84b7a8568a61a4bf2fe0650c5488d05394b1
- **Owner fingerprint:** sha256:430708781df740ff19851ac1797c5c4ec37faf8cccce141d1712c5f8cc75c1c1
- **Owner since:** 2026-09-13T15:35:55Z
- **Owner until:** 2026-09-13T17:35:55Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T12:55:34Z, raised via `deltic bugs new`) -> Fixed (2026-09-13T14:16:46Z, deltic:auto role=fix run=fix-20260913T141503Z-a056b67d branch=task/bug-ROP-BUG-KIL-00061-run-fix-20260913T141503Z-a056b67d code=a05a40fe2446f74ae1cbe488e03f98eca11fa744 gate=manual)

## Observation

Static review at `d18b7aa7d7b60cc8799428e6785cf5be326fb1ce` found that
`D:\worktrees\ropus\20260822-REV-ROP-CDX@KILN-code-review-133528\ropusplay\tests\cli.rs:23-27,56-60,100-108` launches the player with
`Command::output()`. Every call waits without a deadline, timeout result, kill path, or
explicit reap. A cpal device-enumeration stall, audio-driver hang, or child regression can
therefore wedge `cargo test -p ropusplay` indefinitely instead of returning a bounded failure.
Expected: external test children have a generous finite lifetime and are killed and reaped on
expiry. Fix: route all three invocations through one cross-platform timeout-aware helper, return
a distinct timeout diagnostic, and prove the helper with a deliberately hanging child. This is
the `ropusplay` instance of the subprocess-lifetime class; open `ROP-BUG-KIL-00055` owns
`ropusdec`, `ROP-BUG-KILN-00018` owns `harness-control`, and
`ROP-BUG-CRUCIBLE-00006` owns `full-test`. Static source inspection only; no process,
application, test, build, or exploratory harness ran.

## Fix

<unfixed — raised only>

## Notes
