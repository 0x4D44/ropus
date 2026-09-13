# ROP-BUG-KIL-00057 — Malformed packets can flood and block ropusenc diagnostics

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-tools-core/decode-diagnostics
- **Raised:** 2026-08-22T11:25:39Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260913T075442Z-340b3010
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KIL-00057-run-fix-20260913T075442Z-340b3010
- **Owner base:** 5783bccf951b25104fa8b555f9980dc254e99eed
- **Owner fingerprint:** -
- **Owner since:** 2026-09-13T07:54:42Z
- **Owner until:** 2026-09-13T09:54:42Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T11:25:39Z, raised via `deltic bugs new`)

## Observation

Static review at baseline a463e758 found that decode_input prints one warning for every malformed nonempty Opus packet and then continues (ropus-tools-core/src/audio/decode.rs:244-260). This diagnostic bypasses the OutputPolicy passed by ropusenc (ropusenc/src/main.rs:390-393), so --quiet does not suppress it. An input containing many decoder-rejected packets can generate unbounded stderr output; when stderr is a pipe whose reader does not drain, the encoder can block once the pipe fills. Expected: malformed-packet diagnostics are bounded, respect quiet mode, and cannot turn invalid input into an output/log denial of service. Fix: route decoder diagnostics through a policy-aware sink, cap or aggregate repeated failures, and report a final count; add a bounded-output quiet-mode regression oracle. No application or test execution was performed in this review pass.

## Fix

<unfixed — raised only>

## Notes
