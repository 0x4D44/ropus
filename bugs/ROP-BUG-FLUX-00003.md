# ROP-BUG-FLUX-00003 — C reference harness compiles x86 CPU probe on Apple Silicon

- **State:** Closed
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
- **State history:** Open (2026-07-30, raised via `deltic bugs new`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T205358Z-p41959-n128536000-c1 branch=task/bug-ROP-BUG-FLUX-00003-run-fix-20260730T205358Z-p41959-n128536000-c1 code=a08e2fa gate=manual) -> Closed (2026-08-01, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main dc05a88; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Observed on Apple Silicon macOS in a clean task worktree based on origin/main ae278a4d315b8f922181afc2e481f9b5534bf680 after provisioning the repository's pinned xiph/opus reference tree with cargo run -p fetch-assets -- all. Reproduce: cargo clippy --locked --all-targets -- -D warnings, or the Deltic automatic integration gate. Expected: the C reference harness selects sources for aarch64-apple-darwin and compiles. Actual: harness/build.rs compiles reference/celt/x86/x86cpu.c; Apple Clang rejects cpuid.h as x86-only and reports invalid x86 assembly constraints. The same failure is documented in the active macOS/ARM optimisation worktree.

## Fix

Commit `a08e2fa` gated the x86 reference sources and `-msse4.1` behind
`CARGO_CFG_TARGET_ARCH` in `harness/build.rs` and `tests/fuzz/build.rs`, and wrapped the
`OPUS_X86_MAY_HAVE_*` / `CPU_INFO_BY_C` macros in `harness/config.h` behind an x86 preprocessor
guard. Later commits (`ca425df`, `b2124c9`, `dad075e`) moved those macros out of `config.h`
into arch-conditional `cc::Build::define` calls in `harness/build.rs` and added the
`OPUS_ARM_*_NEON_INTR` defines for aarch64.

## Notes

### Verification attempt — NOT closed (2026-07-31, host KILN / Windows x86_64)

Left at **Fixed**: this host cannot reproduce the original observation, so closing it here
would be a fabricated closure. What was and was not established:

- **Not verifiable here.** The recorded symptom is Apple Clang rejecting
  `reference/celt/x86/x86cpu.c` on `aarch64-apple-darwin`. Reproducing it needs an Apple
  Silicon macOS host; this verifier ran on Windows x86_64, which takes the x86 branch and
  never compiles the failing path. There is no host-independent regression test either — the
  only coverage is the harness build itself on ARM, so the check must run on the platform.
- **Established by inspection only (inference, not observation).** The arch gating is present
  and intact on trunk 6ccb736: `harness/build.rs:16` derives `target_is_x86` from
  `CARGO_CFG_TARGET_ARCH` and guards the `-msse4.1` flag and both x86 source lists
  (`harness/build.rs:305,339,382`); `tests/fuzz/build.rs` mirrors it
  (`tests/fuzz/build.rs:217,240,264`); the x86 RTCD macros are now defined only under the x86
  branch (`harness/build.rs:307`). That addresses the stated root cause.
- **Next step for closure.** A verifier on Apple Silicon macOS should run the recorded repro —
  `cargo clippy --locked --all-targets -- -D warnings` with the pinned xiph/opus tree
  provisioned via `cargo run -p fetch-assets -- all` — and close on that evidence.


### Verification — Closed (2026-08-01, independent two-eyes, host flux / Apple Silicon macOS)

Verified on the exact platform the observation requires (aarch64-apple-darwin), which the
prior KILN (Windows x86_64) verification pass could not reach.

- `cargo run -p fetch-assets --locked -- reference` provisions the pinned xiph/opus tree.
- `cargo clippy -p ropus-harness --all-targets --locked -- -D warnings` compiles clean:
  `reference/celt/x86/x86cpu.c` is never entered; `harness/build.rs` gates the x86 source
  lists and `-msse4.1` behind `CARGO_CFG_TARGET_ARCH` as recorded in the fix.
- `cargo check --manifest-path tests/fuzz/Cargo.toml --locked` also compiles clean, confirming
  `tests/fuzz/build.rs` mirrors the same arch gating.
- Root cause addressed: Apple Clang never sees the x86 CPU-probe translation unit on this
  target. This directly satisfies the prior verifier's documented "next step for closure".
