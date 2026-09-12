# ROP-BUG-KILN-00017 — Concurrent control tests race through shared artifact paths

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness-control/artifact-isolation
- **Raised:** 2026-08-16T07:49:54Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260912T234409Z-29934271
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KILN-00017-run-fix-20260912T234409Z-29934271
- **Owner base:** b3f8c45f99c998fda84c517a2f890d969f5b7e0d
- **Owner fingerprint:** -
- **Owner since:** 2026-09-12T23:44:09Z
- **Owner until:** 2026-09-13T01:44:09Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-16T07:49:54Z, raised via `deltic bugs new`)

## Observation

Static review at origin/main a97b6f11. harness-control/tests/control_snr.rs:232-237 always uses target/harness-control-tmp, while lines 321-334 and 422-428 reuse fixed packet and PCM filenames. Two cargo test processes in the same checkout can truncate or read those files while the other process validates, decodes, or measures them, producing corrupted inputs, partial outputs, or false failures. Expected: each control invocation owns isolated artifacts. Actual: process boundaries do not isolate the shared paths. Fix: allocate a unique per-test temporary directory with cleanup on drop and pass only paths inside it to both children; add an overlapping-invocation path-isolation oracle. Static inspection only; no concurrent command, app, build, test, decoder, or harness ran.

## Fix

<unfixed — raised only>

## Notes
