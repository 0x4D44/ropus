# ROP-BUG-FLUX-00049 — Picture detector rejects valid JPEG marker layouts

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** ropus-tools-core/picture-detection
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main ac7ff8a. /Users/md/language/ropus/ropus-tools-core/src/container/picture.rs:27-44 recognizes JPEG only when SOI is followed by APP0, APP1, or DQT. Valid JPEG can begin with other APPn, COM, table, or frame markers, including APP2 ICC and APP14 Adobe, so valid cover art is rejected. Fix: use the JPEG SOI signature under an explicit MIME-detection policy, or perform real parsing if validation is required, and add APP2/APP14 fixtures. Static review only; no image or test ran.

## Fix

<unfixed — raised only>

## Notes
