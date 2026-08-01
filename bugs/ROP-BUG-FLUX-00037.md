# ROP-BUG-FLUX-00037 — fb2k duration scan accepts fake Ogg headers inside payload data

- **State:** Fixed
- **Priority:** Should
- **Severity:** Low
- **Area:** ropus-fb2k/ogg-scan
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T210854Z-p93472-n107963000-c1 branch=task/bug-ROP-BUG-FLUX-00037-run-fix-20260801T210854Z-p93472-n107963000-c1 code=69a312c8b5ed9b3640859b48c93f1caabde1270e gate=manual)

## Observation

Static review at origin/main dcfd694. /Users/md/language/ropus/ropus-fb2k/src/reader.rs:964-1018 walks backward byte by byte and accepts the first bytes matching OggS, version zero, and the selected serial. It does not validate lacing, full page extent, flags, boundary provenance, or checksum. A crafted matching header-shaped sequence after the real final page header can supply an arbitrary granule, corrupting total_samples, nominal bitrate, and seek clamping. Expected: duration comes only from a structurally valid page of the selected stream. Fix: parse and validate the complete candidate page, preferably including checksum and EOS policy, and add a fixture with a fake matching header inside final payload or trailing data. Static review only; no parser or test ran.

## Fix

<unfixed — raised only>

## Notes
