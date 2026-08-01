# ROP-BUG-FLUX-00025 — C reference build can stay stale after source changes

- **State:** Closed
- **Priority:** Must
- **Severity:** High
- **Area:** harness/build-invalidation
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T042107Z-p26455-n773376000-c1 branch=task/bug-ROP-BUG-FLUX-00025-run-fix-20260801T042107Z-p26455-n773376000-c1 code=a575cf3cd3a10d6be820e37df734b25b71887ee1 gate=manual) -> Closed (2026-08-01, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main dc05a88; fixer was a prior automated fix session, verifier is a different actor)

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


### Verification — Closed (2026-08-01, independent two-eyes, host flux)

Covers both the fixed-point harness observation and the float DEEP_PLC / DRED-oracle note.

- Fix commit `a575cf3` centralizes source registration in `harness/build.rs` behind a shared
  `ReferenceBuildManifest` (`harness/reference_build_manifest.rs`), asserting
  `manifest.sources() == manifest.watched_sources()` live at build time, and
  `harness-deep-plc/build.rs` reuses the *same* module (`#[path = "../harness/
  reference_build_manifest.rs"]`) for its own ~100+ float reference sources — one manifest
  covers both harnesses as the note required. Both `cargo clippy -p ropus-harness` and
  `cargo clippy -p ropus-harness-deep-plc` built clean, so both assertions passed live.
- The copied DRED bitrate oracle in `harness-deep-plc/dred_encode_shim.c` is now guarded by
  `verify_copied_dred_bitrate_oracle`, an FNV-1a64 fingerprint of the live
  `reference/src/opus_encoder.c` block pinned in `build.rs`; this also passed live during the
  same build (a mismatch would have panicked the build script), resolving the previously
  UNVERIFIED copied-body-parity gap.
- New harness tests `compiled_sources_have_exactly_one_file_watch`,
  `escaping_reference_root_is_rejected`, and `duplicate_source_is_rejected_before_compilation`
  pass.
- `cargo test -p ropus-harness --locked` and `cargo test -p ropus-harness-deep-plc --locked`:
  every suite green, 0 failed.
