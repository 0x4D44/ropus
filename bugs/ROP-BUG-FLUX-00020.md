# ROP-BUG-FLUX-00020 — Primary HTML report write failure still exits successfully

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** full-test/reporting
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T190442Z-p27224-n474403000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00020-run-fix-20260801T190442Z-p27224-n474403000-c1
- **Owner base:** 15205f922bf09eafa042578407d6c39362a71bbf
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T19:04:42Z
- **Owner until:** 2026-08-01T21:04:42Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main d2a0fb9. Observation: `/Users/md/language/ropus/full-test/src/main.rs:175` through `:180` logs an HTML write error and reduces it to `report_path = None`, but line 216 returns the earlier PASS or WARN banner exit code unchanged. The same module at lines 7-10 and `/Users/md/language/ropus/wrk_docs/2026.04.19 - HLD - full-test-runner.md:70` define HTML as the primary artifact. A read-only results directory, full filesystem, or other I/O failure can therefore produce no report while automation receives exit 0. Expected: failure to deliver the primary report is nonzero and explicit in the summary. Fix: propagate write failure into the final exit status, preserve the diagnostic, use atomic temp-file plus rename to avoid partial reports, and add a write-failure seam proving PASS and WARN cannot exit 0 without the artifact. Static review only; no runner, build, tests, or harness executed.

## Fix

<unfixed — raised only>

## Notes
