# ROP-BUG-FLUX-00025 — C reference build can stay stale after source changes

- **State:** Open
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at `origin/main` `d0ab87e`. `/Users/md/language/ropus/harness/build.rs:336` through `:388` registers roughly one hundred C-reference sources, but the explicit Cargo watches at `:19` through `:23` and `:394` through `:400` cover only two reference sources and local harness files. Cargo's documented rule is that once any `rerun-if-changed` directives are emitted, only named inputs trigger the build script. An edit to an unwatched reference source or header can therefore leave the previously linked `libopus_ref` in place, making differential and performance results compare against stale C code. Fix: centralize source registration so every compiled source and relevant header/directory emits a matching watch, then assert one-to-one coverage in pure build-manifest logic. Static review only; no build or oracle ran.

## Fix

<unfixed — raised only>

## Notes
