# ROP-BUG-KIL-00057 — Malformed packets can flood and block ropusenc diagnostics

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-tools-core/decode-diagnostics
- **Raised:** 2026-08-22T11:25:39Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T153351Z-1fe07d9e
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KIL-00057-run-verify-20260913T153351Z-1fe07d9e
- **Owner base:** 04d481bd7b7158f3b5b8d9c9fa65ccb9541bdf47
- **Owner fingerprint:** sha256:50df18f730ed24bd07167ecaa3d986acb3c9c8cbcb95960c687722dbfaaa9615
- **Owner since:** 2026-09-13T15:33:51Z
- **Owner until:** 2026-09-13T17:33:51Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T11:25:39Z, raised via `deltic bugs new`) -> Fixed (2026-09-13T08:21:01Z, deltic:auto role=fix run=fix-20260913T075442Z-340b3010 branch=task/bug-ROP-BUG-KIL-00057-run-fix-20260913T075442Z-340b3010 code=2c7775c91958c4dd5d220c9dddef19d83d4b8281 gate=manual)

## Observation

Static review at baseline a463e758 found that decode_input prints one warning for every malformed nonempty Opus packet and then continues (ropus-tools-core/src/audio/decode.rs:244-260). This diagnostic bypasses the OutputPolicy passed by ropusenc (ropusenc/src/main.rs:390-393), so --quiet does not suppress it. An input containing many decoder-rejected packets can generate unbounded stderr output; when stderr is a pipe whose reader does not drain, the encoder can block once the pipe fills. Expected: malformed-packet diagnostics are bounded, respect quiet mode, and cannot turn invalid input into an output/log denial of service. Fix: route decoder diagnostics through a policy-aware sink, cap or aggregate repeated failures, and report a final count; add a bounded-output quiet-mode regression oracle. No application or test execution was performed in this review pass.

## Fix

<unfixed — raised only>

## Notes
