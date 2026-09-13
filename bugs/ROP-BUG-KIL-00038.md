# ROP-BUG-KIL-00038 — fb2k float decode allocates on every audio packet

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-fb2k/realtime
- **Raised:** 2026-08-22T06:10:45Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260913T152408Z-f4fa9fc5
- **Owner host:** CRUCIBLE
- **Owner branch:** task/bug-ROP-BUG-KIL-00038-run-verify-20260913T152408Z-f4fa9fc5
- **Owner base:** a011399f31b201cc94bc8137bb770904db58aba2
- **Owner fingerprint:** sha256:0eb014342943ae6129b1d0b2fefd79855c9b5a427e348420b265056c788022b1
- **Owner since:** 2026-09-13T15:24:08Z
- **Owner until:** 2026-09-13T17:24:08Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T06:10:45Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-13T05:41:32Z, deltic:auto role=fix run=fix-20260913T052001Z-4b7e6dfe branch=task/bug-ROP-BUG-KIL-00038-run-fix-20260913T052001Z-4b7e6dfe code=f77319fd27aca6843cde9a0bedb150a1d9182d66 gate=manual)

## Observation

Static review at HEAD 3e0f6c1. ropus-fb2k/src/reader.rs:484-500 says the reusable decode scratch keeps the audio-thread path allocation-free after lazy initialization. The called ropus/src/opus/decoder.rs:1394-1405 nevertheless allocates a fresh Vec<i16> inside every decode_float invocation. Typical 20 ms audio therefore performs about 50 heap allocations per second on the real-time-adjacent path, risking avoidable jitter and contradicting the documented invariant. Expected: supply reusable integer scratch or a buffer-taking decoder path, and add an allocation-count assertion after initialization. Static review only; no app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
