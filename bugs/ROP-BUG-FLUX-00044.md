# ROP-BUG-FLUX-00044 — Simulated packet loss uses the wrong PLC duration

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-tools-core/decode-plc
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

Static review at origin/main ac7ff8a. /Users/md/language/ropus/ropus-tools-core/src/commands/decode.rs:240-293 initializes PLC at 960 samples and sizes each dropped packet from the previous successful decode. The current packet TOC is available but ignored, so first-packet loss and duration transitions alter the output timeline for non-20 ms streams. /Users/md/language/ropus/ropus-tools-core/tests/round_trip.rs:669-701 covers only 20 ms and asserts a broad lower length bound. Fix: derive each deliberately dropped packet duration from its own TOC, validate it, and assert exact no-loss/loss timeline equality for 10 ms, 60 ms, and duration-switch streams. Static review only; no decoder or test ran.

## Fix

<unfixed — raised only>

## Notes
