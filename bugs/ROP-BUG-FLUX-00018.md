# ROP-BUG-FLUX-00018 — C ABI version string reports stale package version

- **State:** Fixed
- **Priority:** Could
- **Severity:** Low
- **Area:** capi/version-metadata
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new`) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T184952Z-p9388-n274066000-c1 branch=task/bug-ROP-BUG-FLUX-00018-run-fix-20260801T184952Z-p9388-n274066000-c1 code=21e119ce43dbe3e6db1e5db91d3ae57e5901701a gate=manual)

## Observation

capi/Cargo.toml:3 declares version 0.2.2, but capi/src/lib.rs:122 hard-codes libopus mdopus-capi-0.1.0-fixed and opus_get_version_string returns it at lines 152-156. The literal has not changed across package version bumps. Expected: the exported diagnostic string identifies the built capi version. Actual: every 0.2.x build claims 0.1.0. Fix: derive the NUL-terminated string from CARGO_PKG_VERSION (or build-generated metadata) and add a focused assertion tying it to the manifest version.

## Fix

<unfixed — raised only>

## Notes
