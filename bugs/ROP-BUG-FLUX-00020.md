# ROP-BUG-FLUX-00020 — Primary HTML report write failure still exits successfully

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** full-test/reporting
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T190442Z-p27224-n474403000-c1 branch=task/bug-ROP-BUG-FLUX-00020-run-fix-20260801T190442Z-p27224-n474403000-c1 code=acd764f9a547b2590df887461c3d484bbfe52377 gate=manual) -> Closed (2026-08-01, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main dc05a88; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main d2a0fb9. Observation: `/Users/md/language/ropus/full-test/src/main.rs:175` through `:180` logs an HTML write error and reduces it to `report_path = None`, but line 216 returns the earlier PASS or WARN banner exit code unchanged. The same module at lines 7-10 and `/Users/md/language/ropus/wrk_docs/2026.04.19 - HLD - full-test-runner.md:70` define HTML as the primary artifact. A read-only results directory, full filesystem, or other I/O failure can therefore produce no report while automation receives exit 0. Expected: failure to deliver the primary report is nonzero and explicit in the summary. Fix: propagate write failure into the final exit status, preserve the diagnostic, use atomic temp-file plus rename to avoid partial reports, and add a write-failure seam proving PASS and WARN cannot exit 0 without the artifact. Static review only; no runner, build, tests, or harness executed.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-01, independent two-eyes, host flux)

- Commit `acd764f` propagates HTML report write failure into the final exit status in
  `full-test/src/main.rs`, using an atomic temp-file-plus-rename write.
- New tests `report_write_failure_is_fatal_for_pass_and_warn` and
  `report_writer_replaces_target_atomically` pass, proving neither PASS nor WARN can exit 0
  without the primary artifact.
- `cargo clippy -p full-test --all-targets --locked -- -D warnings` clean; `cargo test -p
  full-test --locked`: 238 passed, 0 failed.
