# ROP-BUG-FLUX-00068 — Safe float-reference decoder methods permit out-of-bounds C access

- **State:** Closed
- **Priority:** Must
- **Severity:** High
- **Area:** harness-deep-plc/ffi-safety
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T051116Z-p36074-n718775000-c1 branch=task/bug-ROP-BUG-FLUX-00068-run-fix-20260801T051116Z-p36074-n718775000-c1 code=26659708e647f38f9f2d69c95708e382be449b3e gate=manual) -> Closed (2026-08-01, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main dc05a88; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main b65f812. /Users/md/language/ropus/harness-deep-plc/src/lib.rs:293-325 exposes safe CRefFloatDecoder::decode but does not retain the decoder channel count or validate that pcm.len() can hold frame_size * channels before passing pcm.as_mut_ptr() to opus_decode. A safe stereo call with an undersized slice lets C write beyond Rust-managed memory. Safe peek methods at :337-369 and :407-410 also accept arbitrary signed offsets and counts; /Users/md/language/ropus/harness-deep-plc/c/peek.c:105-163 and :221-226 index those ranges without bounds checks, permitting C out-of-bounds reads, while negative counts become huge Rust allocations. Expected: every safe Rust method proves the full FFI buffer/range contract. Actual: ordinary safe calls can trigger undefined behavior. Fix: retain channels, reject non-positive or overflowing dimensions and packet lengths, validate output and peek ranges against authoritative capacities, return Result, and make any irreducibly unchecked raw-state operation unsafe with a complete contract. Add boundary tests that prove invalid inputs never enter C. Static review only; no code, build, test, decoder, or harness ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-01, independent two-eyes, host flux)

- Fix commit `2665970` makes `harness-deep-plc/src/lib.rs`'s `CRefFloatDecoder` retain its
  channel count and validate `pcm.len()` against `frame_size * channels` before calling
  `opus_decode`; safe peek methods now validate ranges against authoritative capacities before
  entering `c/peek.c`.
- New tests `decode_rejects_invalid_dimensions_before_entering_c`,
  `invalid_peeks_return_before_entering_c`,
  `peek_ranges_reject_negative_empty_overflowing_and_past_end`, and
  `runtime_peek_capacities_are_sanity_checked` pass, proving invalid inputs never reach C.
- `cargo clippy -p ropus-harness-deep-plc --all-targets --locked -- -D warnings` clean; `cargo
  test -p ropus-harness-deep-plc --locked`: every suite green, 0 failed.
