# ROP-BUG-KIL-00056 — Atomic encode replacement can weaken destination permissions

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-tools-core/atomic-output
- **Raised:** 2026-08-22T11:25:31Z
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
- **State history:** Open (2026-08-22T11:25:31Z, raised via `deltic bugs new`)

## Observation

Static review at baseline a463e758 found that AtomicOutput::create creates a same-directory temporary file with default process permissions (ropus-tools-core/src/commands/encode.rs:132-135), then AtomicOutput::commit replaces the destination with that temporary inode (ropus-tools-core/src/commands/encode.rs:180-182). Re-encoding over an existing restricted file can therefore replace Unix mode 0600 with the umask-derived mode (commonly 0644), and can analogously replace a restrictive Windows ACL with parent-inherited permissions. Expected: atomic replacement preserves the existing destination security metadata, while new outputs follow an explicit safe policy. Fix: snapshot and apply the existing mode/ACL to the temporary file before publication, define the new-file policy, and add Unix permission plus Windows ACL regression oracles that also verify failure preserves the original. No application or test execution was performed in this review pass.

## Fix

<unfixed — raised only>

## Notes
