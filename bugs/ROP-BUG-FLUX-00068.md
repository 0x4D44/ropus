# ROP-BUG-FLUX-00068 — Safe float-reference decoder methods permit out-of-bounds C access

- **State:** Open
- **Priority:** Must
- **Severity:** High
- **Area:** harness-deep-plc/ffi-safety
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T051116Z-p36074-n718775000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00068-run-fix-20260801T051116Z-p36074-n718775000-c1
- **Owner base:** dc5d3e6a6f9ab5bea01e41fd6498dec007114eba
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T05:11:16Z
- **Owner until:** 2026-08-01T07:11:16Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main b65f812. /Users/md/language/ropus/harness-deep-plc/src/lib.rs:293-325 exposes safe CRefFloatDecoder::decode but does not retain the decoder channel count or validate that pcm.len() can hold frame_size * channels before passing pcm.as_mut_ptr() to opus_decode. A safe stereo call with an undersized slice lets C write beyond Rust-managed memory. Safe peek methods at :337-369 and :407-410 also accept arbitrary signed offsets and counts; /Users/md/language/ropus/harness-deep-plc/c/peek.c:105-163 and :221-226 index those ranges without bounds checks, permitting C out-of-bounds reads, while negative counts become huge Rust allocations. Expected: every safe Rust method proves the full FFI buffer/range contract. Actual: ordinary safe calls can trigger undefined behavior. Fix: retain channels, reject non-positive or overflowing dimensions and packet lengths, validate output and peek ranges against authoritative capacities, return Result, and make any irreducibly unchecked raw-state operation unsafe with a complete contract. Add boundary tests that prove invalid inputs never enter C. Static review only; no code, build, test, decoder, or harness ran.

## Fix

<unfixed — raised only>

## Notes
