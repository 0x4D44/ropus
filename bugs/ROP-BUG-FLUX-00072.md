# ROP-BUG-FLUX-00072 — Classical PLC control cannot calibrate the neural acceptance gate

- **State:** Closed
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-02T01:55:56Z, deltic:auto role=fix run=fix-20260802T013612Z-p39664-n393663000-c1 branch=task/bug-ROP-BUG-FLUX-00072-run-fix-20260802T013612Z-p39664-n393663000-c1 code=39487bd gate=manual) -> Closed (2026-08-02, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main 60de518; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main 1ae9e50. /Users/md/language/ropus/harness-control/tests/control_snr.rs:14-17 measures classical SILK PLC, but :35-39 claims its fixed-vs-float SNR bounds the neural tier-2 result and :41-45 still frames lowering the old 60 dB gate as a pending decision. The repository’s own live oracle refutes that relation: /Users/md/language/ropus/harness-deep-plc/tests/tier2_snr.rs:190-208 records neural SNR 51.79 dB above the classical 42.33 dB result because the algorithms propagate arithmetic error differently, yet uses the classical result to justify a 50 dB neural threshold. The control also asserts only lower bounds at /Users/md/language/ropus/harness-control/tests/control_snr.rs:317-335 and :368-386, so identical outputs, wrong build flavours, or accidental neural activation can pass without proving the intended arithmetic gap. Expected: a gate is calibrated by a like-for-like neural oracle and the control proves its own build flavours, classical mode, signal energy, divergence, and accepted interval. Actual: an unrelated classical error-transfer function is treated as a neural ceiling even though observed neural output exceeds it. Fix: make the classical experiment diagnostic only; establish neural-path-specific cross-precision or per-layer error bounds across supported platforms, fixtures, and loss patterns; set a justified regression margin; assert control identity/mode/energy/divergence; and update the stale 60 dB narrative and HLD rationale. Static review only; no build, test, decoder, model, control experiment, or harness ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-02, independent two-eyes, host flux)

- Fix commit `39487bd` reframes `harness-control/tests/control_snr.rs`'s doc comments to
  state the classical fixed-vs-float experiment is diagnostic-only and cannot bound the
  neural path, and adds control-mode markers (`FIXED_MODE_MARKER`/`FLOAT_MODE_MARKER`
  asserted against decoder stderr), a packet-stream fingerprint (`assert_ne!` proving the
  classical PLC actually altered the damaged stream), signal-energy floors, and an explicit
  `[min, max]` SNR interval (previously only a lower bound). `harness-deep-plc/tests/
  tier2_snr.rs` removes the stale "classical ceiling justifies 50 dB" narrative and adds
  `dnn_plc_neural_cross_precision_calibration_matrix`, a direct Rust-vs-C neural calibration
  across multiple fixtures and loss intervals with its own `NEURAL_CALIBRATION_MIN_SNR_DB`
  margin — replacing the classical-SNR justification the bug objects to. The referenced HLD
  (`wrk_docs/2026.04.19 - HLD - stages-6-8-relaxed-oracles.md`) is updated to match.
  Confirmed by reading both diffs against the observation.
- Rebuilt against the real C reference on a fresh worktree at `origin/main` `60de518`:
  `cargo test -p ropus-harness-control --test control_snr` — 2 passed (both classical-mode
  cases, ~33s, genuine C-reference fixed/float decode); `cargo test -p ropus-harness-deep-plc
  --test tier2_snr` — 4 passed, 0 failed, including the new calibration-matrix test and the
  existing lossless/50dB/loss-pattern cases.
- `cargo clippy -p ropus-harness-control -p ropus-harness-deep-plc --all-targets --locked --
  -D warnings`: clean.
