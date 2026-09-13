# ROP-BUG-KIL-00057 — Malformed packets can flood and block ropusenc diagnostics

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-tools-core/decode-diagnostics
- **Raised:** 2026-08-22T11:25:39Z
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
- **State history:** Open (2026-08-22T11:25:39Z, raised via `deltic bugs new`) -> Fixed (2026-09-13T08:21:01Z, deltic:auto role=fix run=fix-20260913T075442Z-340b3010 branch=task/bug-ROP-BUG-KIL-00057-run-fix-20260913T075442Z-340b3010 code=2c7775c91958c4dd5d220c9dddef19d83d4b8281 gate=manual) -> Closed (2026-09-13T16:47:37Z, independent two-eyes verification model=codex@xhigh, verifier=CRUCIBLE, fixer=deltic:auto, fix=2c7775c91958c4dd5d220c9dddef19d83d4b8281)

## Observation

Static review at baseline a463e758 found that decode_input prints one warning for every malformed nonempty Opus packet and then continues (ropus-tools-core/src/audio/decode.rs:244-260). This diagnostic bypasses the OutputPolicy passed by ropusenc (ropusenc/src/main.rs:390-393), so --quiet does not suppress it. An input containing many decoder-rejected packets can generate unbounded stderr output; when stderr is a pipe whose reader does not drain, the encoder can block once the pipe fills. Expected: malformed-packet diagnostics are bounded, respect quiet mode, and cannot turn invalid input into an output/log denial of service. Fix: route decoder diagnostics through a policy-aware sink, cap or aggregate repeated failures, and report a final count; add a bounded-output quiet-mode regression oracle. No application or test execution was performed in this review pass.

## Fix

### Verification summary (2026-09-13)

- Re-ran `malformed_packet_diagnostics_are_bounded_and_aggregated`; it passed, and the relevant package gates passed.
- A red control removed the diagnostic cap; the test emitted 13 diagnostics instead of 4 and failed its count assertion, so the fix was restored.

## Notes
