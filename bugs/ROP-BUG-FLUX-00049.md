# ROP-BUG-FLUX-00049 — Picture detector rejects valid JPEG marker layouts

- **State:** Closed
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T222739Z-p40717-n324296000-c1 branch=task/bug-ROP-BUG-FLUX-00049-run-fix-20260801T222739Z-p40717-n324296000-c1 code=e583f48 gate=manual) -> Closed (2026-08-02, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main 3528b9e; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main ac7ff8a. /Users/md/language/ropus/ropus-tools-core/src/container/picture.rs:27-44 recognizes JPEG only when SOI is followed by APP0, APP1, or DQT. Valid JPEG can begin with other APPn, COM, table, or frame markers, including APP2 ICC and APP14 Adobe, so valid cover art is rejected. Fix: use the JPEG SOI signature under an explicit MIME-detection policy, or perform real parsing if validation is required, and add APP2/APP14 fixtures. Static review only; no image or test ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-02, independent two-eyes, host flux)

- Fix commit `e583f48` changes `detect_format` in `ropus-tools-core/src/container/picture.rs`
  to recognize JPEG by the SOI signature (`FF D8 FF`) rather than a fixed APP0/APP1/DQT
  marker allow-list.
- Regression re-verified by construction: the fix's new tests
  (`detect_format_recognises_jpeg_app2_icc_profile`,
  `detect_format_recognises_jpeg_app14_adobe`) were spliced onto the pre-fix `picture.rs` at
  `e583f48~1` — both panicked with "unrecognised picture format (need JPEG or PNG)",
  confirming valid APP2/APP14 JPEGs were rejected. Both pass at the current tree.
- `cargo clippy -p ropus-tools-core --all-targets --locked -- -D warnings` clean; `cargo test
  -p ropus-tools-core --locked`: 129 lib + 22 integration passed, 0 failed.
