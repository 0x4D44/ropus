# ROP-BUG-FLUX-00038 — fb2k decoder crosses the first chained logical stream boundary

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-fb2k/demux
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T211447Z-p8705-n395922000-c1 branch=task/bug-ROP-BUG-FLUX-00038-run-fix-20260801T211447Z-p8705-n395922000-c1 code=80ef0e528cf8b077bed9dbe3cee2897241579498 gate=manual) -> Closed (2026-08-01, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main 9e51e0a; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main dcfd694. The signed design at /Users/md/language/ropus/wrk_docs/2026.04.18 - HLD - foobar2000 opus decoder component.md:45-50 and :683-684 says chained files play the first logical stream and stop. /Users/md/language/ropus/ropus-fb2k/src/reader.rs:297 and :361 record its serial, but decode_next at :487-578 checks neither packet serial nor selected-stream EOS. It therefore reads onward and feeds the next chain OpusHead into the existing decoder, producing INVALID_STREAM instead of clean EOF. Expected: the component returns EOF at the first selected logical stream EOS. Fix: track packet EOS/serial, stop before later chains, and add a two-chain fixture. Multiplexed-stream support is outside this bug because this pass did not verify that contract. Static review only; no decode or test ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-01, independent two-eyes, host flux)

- Fix commit `80ef0e5` tracks the selected `PacketReader` stream serial and last-in-stream
  state in `ropus-fb2k/src/reader.rs`, so `decode_next` stops at the first selected logical
  stream's EOS instead of feeding a later chain's `OpusHead` into the active decoder; the
  sticky EOS state resets on seek.
- New test `chained_stream_stops_at_first_logical_eos` passes.
- `cargo clippy -p ropus-fb2k --all-targets --locked -- -D warnings` clean; `cargo test -p
  ropus-fb2k --locked`: 80 passed, 0 failed.
