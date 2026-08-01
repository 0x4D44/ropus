# ROP-BUG-FLUX-00018 — C ABI version string reports stale package version

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** capi/version-metadata
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T184952Z-p9388-n274066000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00018-run-fix-20260801T184952Z-p9388-n274066000-c1
- **Owner base:** d9aba8a74e4bcb28dc16d815a71c31114d0b9c05
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T18:49:52Z
- **Owner until:** 2026-08-01T20:49:52Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31, raised via `deltic bugs new`)

## Observation

capi/Cargo.toml:3 declares version 0.2.2, but capi/src/lib.rs:122 hard-codes libopus mdopus-capi-0.1.0-fixed and opus_get_version_string returns it at lines 152-156. The literal has not changed across package version bumps. Expected: the exported diagnostic string identifies the built capi version. Actual: every 0.2.x build claims 0.1.0. Fix: derive the NUL-terminated string from CARGO_PKG_VERSION (or build-generated metadata) and add a focused assertion tying it to the manifest version.

## Fix

<unfixed — raised only>

## Notes
