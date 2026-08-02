# ROP-BUG-FLUX-00072 — Classical PLC control cannot calibrate the neural acceptance gate

- **State:** Fixed
- **Priority:** Must
- **Severity:** High
- **Area:** harness-control/oracle-calibration
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-02T01:55:56Z, deltic:auto role=fix run=fix-20260802T013612Z-p39664-n393663000-c1 branch=task/bug-ROP-BUG-FLUX-00072-run-fix-20260802T013612Z-p39664-n393663000-c1 code=39487bd gate=manual)

## Observation

Static review at origin/main 1ae9e50. /Users/md/language/ropus/harness-control/tests/control_snr.rs:14-17 measures classical SILK PLC, but :35-39 claims its fixed-vs-float SNR bounds the neural tier-2 result and :41-45 still frames lowering the old 60 dB gate as a pending decision. The repository’s own live oracle refutes that relation: /Users/md/language/ropus/harness-deep-plc/tests/tier2_snr.rs:190-208 records neural SNR 51.79 dB above the classical 42.33 dB result because the algorithms propagate arithmetic error differently, yet uses the classical result to justify a 50 dB neural threshold. The control also asserts only lower bounds at /Users/md/language/ropus/harness-control/tests/control_snr.rs:317-335 and :368-386, so identical outputs, wrong build flavours, or accidental neural activation can pass without proving the intended arithmetic gap. Expected: a gate is calibrated by a like-for-like neural oracle and the control proves its own build flavours, classical mode, signal energy, divergence, and accepted interval. Actual: an unrelated classical error-transfer function is treated as a neural ceiling even though observed neural output exceeds it. Fix: make the classical experiment diagnostic only; establish neural-path-specific cross-precision or per-layer error bounds across supported platforms, fixtures, and loss patterns; set a justified regression margin; assert control identity/mode/energy/divergence; and update the stale 60 dB narrative and HLD rationale. Static review only; no build, test, decoder, model, control experiment, or harness ran.

## Fix

<unfixed — raised only>

## Notes
