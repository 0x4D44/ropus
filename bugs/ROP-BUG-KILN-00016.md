# ROP-BUG-KILN-00016 — Classical control uses stale packet-loss recovery horizon

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness-control/loss-pattern
- **Raised:** 2026-08-16T07:49:45Z
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
- **State history:** Open (2026-08-16T07:49:45Z, raised via `deltic bugs new`)

## Observation

Static review at origin/main a97b6f11. harness-control/tests/control_snr.rs:65-68 claims to mirror the tier-2 loss pattern but drops every positive multiple of seven through frame 98, and lines 303-306 expect 14 losses. The live tier-2 contract at harness-deep-plc/tests/tier2_snr.rs:42-66 requires a complete seven-frame recovery horizon and drops only frames 7 through 91, 13 losses. Expected: the classical control measures the documented tier-2 packet and recovery conditions. Actual: its final loss ends mid-cycle, so its aggregate SNR is not directly comparable to the live scenario. Fix: share the interval and complete-cycle predicate or copy the horizon guard exactly, then assert the exact expected loss indexes. Static inspection and git history only; no app, build, test, decoder, or harness ran.

## Fix

<unfixed — raised only>

## Notes
