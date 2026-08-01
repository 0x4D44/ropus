# ROP-BUG-FLUX-00034 — fb2k decoder emits EOS padding beyond the final granule

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-fb2k/decode
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T204523Z-p42766-n152455000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00034-run-fix-20260801T204523Z-p42766-n152455000-c1
- **Owner base:** 0895b1c2813331feebbd710cb2bcbaa0871697f9
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T20:45:23Z
- **Owner until:** 2026-08-01T22:45:23Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main dcfd694. /Users/md/language/ropus/ropus-fb2k/src/reader.rs:487-578 decodes and returns every final-packet sample after leading discard, while /Users/md/language/ropus/ropus-fb2k/src/reader.rs:918-950 derives total_samples from the trimmed final Ogg granule. The decode path never caps output at pre_skip plus total_samples or records EOS. A valid source whose length is not a frame multiple therefore returns encoder padding beyond its advertised duration; /Users/md/language/ropus/ropus-fb2k/tests/roundtrip.rs:1170-1191 even permits one full packet after seek-to-end. Expected: delivered PCM ends exactly at the EOS granule and seek(total_samples) immediately yields EOF. Fix: clamp the selected stream final packet to its absolute EOS granule, persist EOF, reject impossible granules, add a partial-final-frame fixture, and tighten the end-seek oracle to zero samples. Static review only; no decoder or test ran.

## Fix

<unfixed — raised only>

## Notes
