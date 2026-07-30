# ROP-BUG-FLUX-00003 — C reference harness compiles x86 CPU probe on Apple Silicon

- **State:** Fixed
- **Priority:** Must
- **Severity:** Medium
- **Area:** harness/build
- **Raised:** 2026-07-30
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
- **State history:** Open (2026-07-30, raised via `deltic bugs new`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T205358Z-p41959-n128536000-c1 branch=task/bug-ROP-BUG-FLUX-00003-run-fix-20260730T205358Z-p41959-n128536000-c1 code=a08e2fa gate=manual)

## Observation

Observed on Apple Silicon macOS in a clean task worktree based on origin/main ae278a4d315b8f922181afc2e481f9b5534bf680 after provisioning the repository's pinned xiph/opus reference tree with cargo run -p fetch-assets -- all. Reproduce: cargo clippy --locked --all-targets -- -D warnings, or the Deltic automatic integration gate. Expected: the C reference harness selects sources for aarch64-apple-darwin and compiles. Actual: harness/build.rs compiles reference/celt/x86/x86cpu.c; Apple Clang rejects cpuid.h as x86-only and reports invalid x86 assembly constraints. The same failure is documented in the active macOS/ARM optimisation worktree.

## Fix

<unfixed — raised only>

## Notes
