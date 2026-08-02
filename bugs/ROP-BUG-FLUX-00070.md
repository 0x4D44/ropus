# ROP-BUG-FLUX-00070 — DRED bitrate-plumbing gates do not prove their claimed behavior

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness-deep-plc/dred-bitrate-oracles
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260802T012532Z-p27320-n993770000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00070-run-fix-20260802T012532Z-p27320-n993770000-c1
- **Owner base:** 27d6f4334686ddc5eca39891fe328291631cf6ef
- **Owner fingerprint:** -
- **Owner since:** 2026-08-02T01:25:32Z
- **Owner until:** 2026-08-02T03:25:32Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main b65f812. /Users/md/language/ropus/harness-deep-plc/tests/dred_bitrate_plumbing_diff.rs:1-35 and :200-267 name and describe byte-for-byte DRED parity, but the body admits DRED bitrate is zero and compares only the TOC plus cumulative size within 20 percent. The nonzero variant at /Users/md/language/ropus/harness-deep-plc/tests/dred_bitrate_plumbing_nonzero_diff.rs:276-311 checks only ordinary good-packet PCM SNR; it never asserts nonzero DRED bitrate, target chunks, extension presence, or parsed latents. Its claimed direct scalar counterexample also differs: /Users/md/language/ropus/harness-deep-plc/tests/dred_compute_bitrate_ffi_diff.rs:143-148 uses duration 320 rather than the integration configuration duration 100. Expected: a DRED-active gate proves the exact scalar configuration and observes DRED on emitted packets. Actual: DRED may be disabled or absent while both tests pass. Fix: rename the zero-bit test as a coarse cross-precision SILK envelope diagnostic, add the exact duration-100 scalar vector and assert nonzero bitrate/chunks, parse both packet streams, and require populated DRED extensions before applying PCM side checks. Static review only; no code, build, test, encoder, decoder, or harness ran.

## Fix

<unfixed — raised only>

## Notes
