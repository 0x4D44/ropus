# ROP-BUG-FLUX-00003 — C reference harness compiles x86 CPU probe on Apple Silicon

- **State:** Open
- **Priority:** Must
- **Severity:** Medium
- **Area:** harness/build
- **Raised:** 2026-07-30
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260730T174531Z-p35016-n442157000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00003-run-fix-20260730T174531Z-p35016-n442157000-c1
- **Owner base:** 96d7f26de3b20381777f92c34634cbca7bc44c02
- **Owner fingerprint:** -
- **Owner since:** 2026-07-30T17:45:31Z
- **Owner until:** 2026-07-30T19:45:31Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-30, raised via `deltic bugs new`)

## Observation

Observed on Apple Silicon macOS in a clean task worktree based on origin/main ae278a4d315b8f922181afc2e481f9b5534bf680 after provisioning the repository's pinned xiph/opus reference tree with cargo run -p fetch-assets -- all. Reproduce: cargo clippy --locked --all-targets -- -D warnings, or the Deltic automatic integration gate. Expected: the C reference harness selects sources for aarch64-apple-darwin and compiles. Actual: harness/build.rs compiles reference/celt/x86/x86cpu.c; Apple Clang rejects cpuid.h as x86-only and reports invalid x86 assembly constraints. The same failure is documented in the active macOS/ARM optimisation worktree.

## Fix

<unfixed — raised only>

## Notes
