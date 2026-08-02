# ROP-BUG-FLUX-00071 — Integrated DRED encoder gate ignores malformed packets after one success

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness-deep-plc/dred-parse-oracle
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260802T013109Z-p32726-n965228000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00071-run-fix-20260802T013109Z-p32726-n965228000-c1
- **Owner base:** 468ce05b2c89b65c665389eae64030a7fa861807
- **Owner fingerprint:** -
- **Owner since:** 2026-08-02T01:31:09Z
- **Owner until:** 2026-08-02T03:31:09Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main b65f812. /Users/md/language/ropus/harness-deep-plc/tests/dred_integrated_encode.rs:219-251 calls the C parser for every Rust-emitted packet but ignores negative return codes. Any one packet with ret >= 0 and nb_latents >= 1 satisfies the final found assertion at :252-257, so an early valid DRED packet can mask later malformed output. The Rust self-path at :325-349 likewise stops after the first DRED-bearing packet and never processes later extensions. Expected: every emitted Opus packet is structurally accepted, every DRED-bearing extension parses and processes, and at least one packet proves DRED presence. Actual: the gate proves only one success. Fix: require nonnegative C parse status for every packet, independently identify DRED-bearing packets, require every such extension to parse and process with bounded fields, and keep a separate at-least-one-DRED assertion. Static review only; no code, build, test, encoder, parser, or harness ran.

## Fix

<unfixed — raised only>

## Notes
