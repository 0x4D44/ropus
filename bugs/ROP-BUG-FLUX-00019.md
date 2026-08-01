# ROP-BUG-FLUX-00019 — Stage 2 ignores nonzero Cargo exit status

- **State:** Closed
- **Priority:** Must
- **Severity:** High
- **Area:** full-test/stage2
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T040926Z-p98732-n591719000-c1 branch=task/bug-ROP-BUG-FLUX-00019-run-fix-20260801T040926Z-p98732-n591719000-c1 code=bd9d3ceadc7a2f33df3641c239692661f4dd5307 gate=manual) -> Closed (2026-08-01, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main dc05a88; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main d2a0fb9. Observation: `/Users/md/language/ropus/full-test/src/tests.rs:256` captures cargo output, but lines 260-267 discard `Output.status`. Only spawn failure sets `build_failed` at lines 280-283. `/Users/md/language/ropus/full-test/src/cargo_parse.rs:88` through `:101` derives failure only from parsed summaries and narrow compiler-error text, while `/Users/md/language/ropus/full-test/src/tests.rs:72` through `:76` treats zero parsed failures as green. A signal-killed test binary, rejected Cargo invocation, or missing/truncated summary can therefore return nonzero while Stage 2 and the overall validation report PASS. Expected: every unsuccessful Cargo status makes Stage 2 non-green. Fix: retain `ExitStatus`, force a structured execution failure on any unsuccessful status not already explained, and add injected cases for nonzero with empty or unrecognized output, signal termination, and zero with a valid summary. Static review only; no runner, build, tests, or harness executed.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-01, independent two-eyes, host flux)

- Commit `bd9d3ce` retains Cargo's `ExitStatus` in `full-test/src/tests.rs` and forces a
  structured execution failure for any unsuccessful status not already explained by a parsed
  summary, covering signal termination and empty/unrecognized output.
- New tests `nonzero_cargo_with_empty_output_fails_stage`,
  `nonzero_cargo_with_unrecognized_output_fails_stage`, `signal_exit_status_maps_to_signal_termination`,
  `signal_terminated_cargo_fails_stage`, and `recognized_test_failure_explains_nonzero_cargo_status`
  cover exactly the injected cases the fix called for;
  `successful_cargo_with_valid_summary_remains_green` proves the legitimate-pass path is
  untouched.
- `cargo clippy -p full-test --all-targets --locked -- -D warnings` clean; `cargo test -p
  full-test --locked`: 238 passed, 0 failed.
