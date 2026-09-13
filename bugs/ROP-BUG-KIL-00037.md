# ROP-BUG-KIL-00037 — fb2k seek index can consume memory proportional to every Ogg page

- **State:** Closed
- **Priority:** Should
- **Severity:** High
- **Area:** ropus-fb2k/seek
- **Raised:** 2026-08-22T06:10:45Z
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
- **State history:** Open (2026-08-22T06:10:45Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-09-13T05:32:30Z, deltic:auto role=fix run=fix-20260913T051308Z-26fe97e8 branch=task/bug-ROP-BUG-KIL-00037-run-fix-20260913T051308Z-26fe97e8 code=b54ff049b1be16e0fb9b97b3c4cad1ff41f8ad82 gate=manual) -> Closed (2026-09-13T16:33:35Z, independent two-eyes verification model=codex@xhigh, verifier=CRUCIBLE, fixer=deltic:auto, fix=b54ff049b1be16e0fb9b97b3c4cad1ff41f8ad82)

## Observation

Static review at HEAD 3e0f6c1. The first nonzero seek calls build_page_index at ropus-fb2k/src/reader.rs:748-751. scan_pages at :957-1036 walks to file_size and pushes one 16-byte entry for every matching page with a known granule at :1018-1024, with no cap or sparsification. A large file made of small pages can force hundreds of megabytes or more of index allocation on first seek. Expected: use bounded sparse indexing or bisection-based seeking, stop at the selected EOS, and cover a high-page-count input with a memory-bound oracle. Static review only; no app, build, test, or harness ran.

## Fix

### Verification summary (2026-09-13, independent verifier)

- Re-ran `page_index_is_sparse_and_stops_at_selected_eos`; the bounded sparse index behavior passed with the `ropus-fb2k` package gate.
- Red control: disabling the maximum-index guard made the test exceed its entry bound and fail its own assertion. The mutation was restored.

## Notes
