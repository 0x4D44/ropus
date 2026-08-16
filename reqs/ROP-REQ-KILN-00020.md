# ROP-REQ-KILN-00020 — Centralize the shared PLC control fixture and loss policy

- **State:** Draft
- **Priority:** Should
- **Area:** harness-control/shared-test-support
- **Raised:** 2026-08-16T07:50:25Z
- **Discovery source:** Agent
- **Implemented-by:** —
- **Satisfied-by:** —
- **Violated-by:** —
- **Depends-on:** —
- **Design:** —
- **Flow:** heavy
- **Claimed-by:** —
- **State history:** Draft (2026-08-16T07:50:25Z, raised via `deltic reqs new`)

## Statement

The test infrastructure must define the deterministic PLC control PCM fixture, codec constants, and complete-cycle loss policy once for harness-control and the live tier-2 oracle, while allowing explicitly different diagnostic patterns to remain local and named. Today harness-control/tests/control_snr.rs:46-91 and harness-deep-plc/tests/tier2_snr.rs:29-94 duplicate this contract; the control copy already drifted to 14 losses through frame 98 while tier-2 requires 13 losses through frame 91. Acceptance should prove both consumers use the same shared scenario and exact loss-index oracle, so a future fixture or horizon change cannot land in only one crate. This is narrow test-support debt beyond the immediate ROP-BUG-KILN-00016 predicate correction.

## Notes
