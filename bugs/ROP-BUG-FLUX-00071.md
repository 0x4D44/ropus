# ROP-BUG-FLUX-00071 — Integrated DRED encoder gate ignores malformed packets after one success

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness-deep-plc/dred-parse-oracle
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-02, deltic:auto role=fix run=fix-20260802T013109Z-p32726-n965228000-c1 branch=task/bug-ROP-BUG-FLUX-00071-run-fix-20260802T013109Z-p32726-n965228000-c1 code=5c76cfc gate=manual) -> Closed (2026-08-02, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main 60de518; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main b65f812. /Users/md/language/ropus/harness-deep-plc/tests/dred_integrated_encode.rs:219-251 calls the C parser for every Rust-emitted packet but ignores negative return codes. Any one packet with ret >= 0 and nb_latents >= 1 satisfies the final found assertion at :252-257, so an early valid DRED packet can mask later malformed output. The Rust self-path at :325-349 likewise stops after the first DRED-bearing packet and never processes later extensions. Expected: every emitted Opus packet is structurally accepted, every DRED-bearing extension parses and processes, and at least one packet proves DRED presence. Actual: the gate proves only one success. Fix: require nonnegative C parse status for every packet, independently identify DRED-bearing packets, require every such extension to parse and process with bounded fields, and keep a separate at-least-one-DRED assertion. Static review only; no code, build, test, encoder, parser, or harness ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-02, independent two-eyes, host flux)

- Fix commit `5c76cfc` extracts `validate_c_parse_frame` and `validate_rust_dred_frame` in
  `dred_integrated_encode.rs`, called unconditionally (`.unwrap_or_else(|e| panic!(...))`)
  for every packet in both `rust_encoded_packets_parse_on_c_reference` and
  `c_encoded_packets_parse_with_rust_decoder`, replacing the old `if ret >= 0 && nb_latents
  >= 1 { ... }` guard that silently skipped negative-return packets after an earlier
  success. A new `validation_tests` module directly proves the fix:
  `malformed_packet_after_valid_packet_is_not_masked` asserts a valid frame 0
  (`validate_c_parse_frame(0, 720, 1, 2)` succeeds) followed by a malformed frame 1
  (`validate_c_parse_frame(1, -1, 0, -1)` returns `Err`) — exactly the masking scenario the
  bug describes, and the equivalent negative case for the Rust-side validator. Under the
  pre-fix loop shape (`if ret >= 0 && ... { assert }`) a `ret = -1` packet falls outside the
  branch and is silently skipped, so this validator and its unit tests could not exist before
  the fix.
- Rebuilt against the real C reference on a fresh worktree at `origin/main` `60de518`:
  `cargo test -p ropus-harness-deep-plc --test dred_integrated_encode` — 5 passed, 0 failed,
  including both new `validation_tests` cases and the three live Rust<->C DRED round-trip
  cases.
- `cargo clippy -p ropus-harness-deep-plc --all-targets --locked -- -D warnings`: clean.
