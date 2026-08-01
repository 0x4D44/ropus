# ROP-BUG-FLUX-00025 — C reference build can stay stale after source changes

- **State:** Open
- **Priority:** Must
- **Severity:** High
- **Area:** harness/build-invalidation
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T042107Z-p26455-n773376000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00025-run-fix-20260801T042107Z-p26455-n773376000-c1
- **Owner base:** 48499a983943963ef69fb9ef0f6651fa7585b23d
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T04:21:07Z
- **Owner until:** 2026-08-01T06:21:07Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at `origin/main` `d0ab87e`. `/Users/md/language/ropus/harness/build.rs:336` through `:388` registers roughly one hundred C-reference sources, but the explicit Cargo watches at `:19` through `:23` and `:394` through `:400` cover only two reference sources and local harness files. Cargo's documented rule is that once any `rerun-if-changed` directives are emitted, only named inputs trigger the build script. An edit to an unwatched reference source or header can therefore leave the previously linked `libopus_ref` in place, making differential and performance results compare against stale C code. Fix: centralize source registration so every compiled source and relevant header/directory emits a matching watch, then assert one-to-one coverage in pure build-manifest logic. Static review only; no build or oracle ran.

## Fix

<unfixed — raised only>

## Notes

### Float-reference harness and copied bitrate oracle (2026-07-31)

Static review at `origin/main` `b65f812` confirmed the same invalidation failure
in the float DEEP_PLC harness. `/Users/md/language/ropus/harness-deep-plc/build.rs:57-259,289-330`
compiles more than one hundred reference sources, while explicit watches cover
only three reference files at `:22-33` and local harness files at `:333-339`.
The one-manifest fix must cover both fixed and float reference builds and their
headers.

The direct bitrate oracle has a second stale-reference path even after Cargo
rebuilds correctly. `/Users/md/language/ropus/harness-deep-plc/dred_encode_shim.c:349-478`
copies the vendor-static bitrate table and function bodies, and
`/Users/md/language/ropus/harness-deep-plc/tests/dred_compute_bitrate_ffi_diff.rs:18-25`
calls that copy as the C side. A vendor change can therefore leave a rebuilt
but stale “reference” oracle. Expose a wrapper from the live implementation
translation unit, or enforce a pinned source-block hash as part of the
one-to-one manifest oracle. This was a static review; the absent `reference/`
tree made current copied-body parity **UNVERIFIED**.
