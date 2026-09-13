# ROP-BUG-KIL-00060 — Playback length formatting misses minute and hour carries

- **State:** Fixed
- **Priority:** Could
- **Severity:** Low
- **Area:** ropusinfo/output-format
- **Raised:** 2026-08-22T12:30:10Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T153526Z-7997107e
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KIL-00060-run-verify-20260913T153526Z-7997107e
- **Owner base:** 33a76bb1803e0119e789c9f814841c816d5ae34e
- **Owner fingerprint:** sha256:2cc55602ed8d55758ac640feec790e606d59abff59fc2316163c56d427fdd18c
- **Owner since:** 2026-09-13T15:35:26Z
- **Owner until:** 2026-09-13T17:35:26Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T12:30:10Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-09-13T09:41:17Z, deltic:auto role=fix run=fix-20260913T093557Z-4ede6a37 branch=task/bug-ROP-BUG-KIL-00060-run-fix-20260913T093557Z-4ede6a37 code=d22f1fb7b5f4010ef70041e8cf8dc35f2cfc6463 gate=manual)

## Observation

Static review at HEAD 1c8b85f. ropus-tools-core/src/commands/info.rs:408-419 decomposes the unrounded f64 duration into hours, minutes, and seconds, then rounds only the seconds field to two decimals. A valid 48 kHz sample count one sample below one hour is 3599.999979 seconds: hours remains 0, minutes remains 59, and the final formatter rounds seconds to 60.00, so default ropusinfo output becomes 59m 60.00s instead of 1h 0m 0.00s. The same carry failure exists immediately below any minute boundary. Expected: displayed seconds remain below 60 and rounding carries into minutes/hours. Fix by rounding the total duration to centiseconds before decomposition, then deriving normalized hours, minutes, and seconds; add one-sample-below minute/hour boundary cases. Static inspection only; no app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
