# ROP-REQ-CRUCIBLE-00016 — Centralize full-test run policy and subprocess execution

- **State:** Draft
- **Priority:** Should
- **Area:** full-test/runner-architecture
- **Raised:** 2026-08-14T15:50:31Z
- **Discovery source:** Agent
- **Implemented-by:** —
- **Satisfied-by:** —
- **Violated-by:** —
- **Depends-on:** —
- **Design:** —
- **Flow:** heavy
- **Claimed-by:** —
- **State history:** Draft (2026-08-14T15:50:31Z, raised via `deltic reqs new` model=gpt-5.6-sol@xhigh)

## Statement

Create one typed WorkspaceContext, RunProfile or StagePlan, and CommandSpec or ProcessRunner owned by full-test main. Today workspace-root discovery, current-directory behavior, quick and release claim policy, environment setup, timeout policy, capture, and command logging are duplicated across C:\worktrees\ropus\20260814-REV-ROP-CDX@CRUCIBLE-code-review-160501\full-test\src\main.rs, setup.rs, quality.rs, tests.rs, fuzz.rs, corpus.rs, platform.rs, ambisonics.rs, bench.rs, and ietf_vectors.rs. The duplication has already produced inconsistent cwd behavior, Unix-only Windows launch shapes, and missing lifetime bounds. Acceptance: each stage consumes one immutable plan; every subprocess declares argv, root cwd, environment, deadline, and capture policy; Windows and Unix launch behavior have pure command-shape tests; report claims derive from the same plan rather than reimplementing flag logic. Keep specialized IETF tempfile capture behind the shared interface. This is structural debt beyond any single bug fix and needs coordinated design.

## Notes
