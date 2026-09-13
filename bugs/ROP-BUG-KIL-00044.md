# ROP-BUG-KIL-00044 — Dither quantizer uses integer noise that preserves rounding bias

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-tools-core/audio-output
- **Raised:** 2026-08-22T07:33:48Z
- **Discovery source:** Agent
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
- **State history:** Open (2026-08-22T07:33:48Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-09-13T06:13:22Z, deltic:auto role=fix run=fix-20260913T060406Z-cf71a882 branch=task/bug-ROP-BUG-KIL-00044-run-fix-20260913T060406Z-cf71a882 code=961bbe4480cd57e3305e1d5907aa4d8838f2ba11 gate=manual)

## Observation

Static review at HEAD f9a3871. ropus-tools-core/src/audio/dither.rs:65-73 claims triangular-PDF dither matching opusdec, but it subtracts two one-bit draws, producing only integer noise {-1,0,+1} before round(). For every non-clipped input, round(x + integer) equals round(x) + integer, so the expected output remains the already-rounded value and the original fractional quantization bias is not decorrelated. The signed parity design at wrk_docs/2026.04.19 - HLD - opus-tools-parity.md:233-241 requires TPDF dither. Existing tests at dither.rs:131-161 check only a one-LSB envelope and mean near the rounded value, so they cannot distinguish this defect. Fix by mapping full-width independent RNG draws to uniform fractional samples before differencing, then add a distribution/mean-error oracle against the unquantized scaled input and a focused differential against opusdec. Static inspection only; no app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
