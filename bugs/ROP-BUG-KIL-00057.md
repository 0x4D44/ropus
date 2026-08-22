# ROP-BUG-KIL-00057 — Malformed packets can flood and block ropusenc diagnostics

- **State:** Open
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
- **State history:** Open (2026-08-22T11:25:39Z, raised via `deltic bugs new`)

## Observation

Static review at baseline a463e758 found that decode_input prints one warning for every malformed nonempty Opus packet and then continues (ropus-tools-core/src/audio/decode.rs:244-260). This diagnostic bypasses the OutputPolicy passed by ropusenc (ropusenc/src/main.rs:390-393), so --quiet does not suppress it. An input containing many decoder-rejected packets can generate unbounded stderr output; when stderr is a pipe whose reader does not drain, the encoder can block once the pipe fills. Expected: malformed-packet diagnostics are bounded, respect quiet mode, and cannot turn invalid input into an output/log denial of service. Fix: route decoder diagnostics through a policy-aware sink, cap or aggregate repeated failures, and report a final count; add a bounded-output quiet-mode regression oracle. No application or test execution was performed in this review pass.

## Fix

<unfixed — raised only>

## Notes
