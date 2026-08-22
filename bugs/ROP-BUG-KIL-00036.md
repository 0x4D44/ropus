# ROP-BUG-KIL-00036 — fb2k reports clean EOF when the selected Ogg stream is truncated

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-fb2k/truncation
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
- **State history:** Open (2026-08-22T06:10:45Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Static review at HEAD 3e0f6c1. ropus-fb2k/src/reader.rs:534-540 maps physical EOF to successful end-of-stream without proving the selected logical stream reached an EOS page. The reverse scan also deliberately degrades a missing EOS to unknown duration at :1050-1052. A torn audio tail can therefore open and play as a silent partial success, contrary to the component HLD at wrk_docs/2026.04.18 - HLD - foobar2000 opus decoder component.md:593-594, which requires truncated or malformed Ogg to return INVALID_STREAM. Expected: make the product contract explicit and enforce it consistently; under the current HLD, reject missing selected-stream EOS and add truncated-tail coverage. Static review only; no app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
