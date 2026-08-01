# ROP-BUG-FLUX-00038 — fb2k decoder crosses the first chained logical stream boundary

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-fb2k/demux
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T211447Z-p8705-n395922000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00038-run-fix-20260801T211447Z-p8705-n395922000-c1
- **Owner base:** 11134dd208a01c76458dac2d4e5b839fdea4ad80
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T21:14:47Z
- **Owner until:** 2026-08-01T23:14:47Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main dcfd694. The signed design at /Users/md/language/ropus/wrk_docs/2026.04.18 - HLD - foobar2000 opus decoder component.md:45-50 and :683-684 says chained files play the first logical stream and stop. /Users/md/language/ropus/ropus-fb2k/src/reader.rs:297 and :361 record its serial, but decode_next at :487-578 checks neither packet serial nor selected-stream EOS. It therefore reads onward and feeds the next chain OpusHead into the existing decoder, producing INVALID_STREAM instead of clean EOF. Expected: the component returns EOF at the first selected logical stream EOS. Fix: track packet EOS/serial, stop before later chains, and add a two-chain fixture. Multiplexed-stream support is outside this bug because this pass did not verify that contract. Static review only; no decode or test ran.

## Fix

<unfixed — raised only>

## Notes
