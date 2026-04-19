# Campaign 9 — Bug L: analysis `frame_tonality` subtract underflow

**Date**: 2026-04-19
**Status**: fixed
**Targets panicking**: `fuzz_encode`, `fuzz_roundtrip`, `fuzz_encode_safety`,
`fuzz_roundtrip_safety`, `fuzz_encode_multiframe` (5 / 9)
**Retraction note**: the original "Bug L" on this date was retracted as
not-a-bug (Stage 6 analysis-off divergence). With Stage 6 6.3 / 6.4 now
landed, a *new* panic turned up at the same letter slot — this note
replaces the earlier triage file in full.

## Config that triggers it

Every input that drives the encoder down the analysis path trips it,
which after Stage 6 is essentially any call with
`complexity >= 10 && 16_000 <= Fs <= 48_000 &&
application != OPUS_APPLICATION_RESTRICTED_SILK`. The five canonical
repros live in `C:\language\mdopus\fuzz_crashes\`:

- `crash_24da1ad627e2a5a7.bin` — fuzz_encode_safety (1286 B, pure-Rust)
- `crash_44ecced56f288382.bin` — fuzz_encode (646 B)
- `crash_8606db8d3057c654.bin` — fuzz_roundtrip (646 B)
- `crash_cde9a65042489ae3.bin` — fuzz_encode_multiframe (7008 B)
- `crash_f727a37351a8ab5a.bin` — fuzz_roundtrip_safety (646 B)

Because `fuzz_encode_safety` has no differential comparison, this is a
genuine Rust panic, not a mismatch.

## Root cause

`ropus/src/opus/analysis.rs:1247` (pre-fix):

```rust
if b >= NB_TBANDS - NB_TONAL_SKIP_BANDS {
    frame_tonality -= band_tonality[b - NB_TBANDS + NB_TONAL_SKIP_BANDS];
}
```

`NB_TBANDS = 18`, `NB_TONAL_SKIP_BANDS = 9`, and the outer loop runs
`b` over `0..NB_TBANDS`, so the guard fires for `b in 9..=17`. In the
C reference (`reference/src/analysis.c:719`) the index is evaluated
as signed `int` arithmetic: `b - 18 + 9` gives `b - 9 ∈ {0..=8}`. In
the Rust port every term was left as `usize`, and Rust evaluates the
subtraction left-to-right — `b - NB_TBANDS` underflows for *every*
value of `b` admitted by the guard (since `b < NB_TBANDS`). Release
builds without overflow-checks silently wrapped to a huge index and
would have OOB-panicked on the array access; the cargo-fuzz profile
enables `overflow-checks`, which catches the underflow first at the
subtraction itself — hence the panic message "attempt to subtract
with overflow".

This is an unconditional panic on the analysis-hot path, introduced
when Stage 6.4 wired `run_analysis` into the encoder.

## Fix

`ropus/src/opus/analysis.rs:1247` — reorder the additive chain so
each partial sum stays non-negative in `usize`. The guard already
ensures `b + NB_TONAL_SKIP_BANDS >= NB_TBANDS`:

```rust
frame_tonality -= band_tonality[b + NB_TONAL_SKIP_BANDS - NB_TBANDS];
```

Bit-exact to the C reference — same index values, same addition order
(up to reassociation in a narrow range where no intermediate result
can exceed `usize::MAX`).

## Verification

1. `cargo test -p ropus --lib --release` — 1804 pass; the single
   pre-existing failure (`prop_sequence_invariants`, energy-budget
   property) reproduces on the pristine tree and is unrelated.

2. Rebuilt the 5 affected fuzz binaries individually (the other 4
   fuzzers are still running and hold locks on their `.exe` files —
   per task guardrails, not touched):

   ```
   cargo +nightly fuzz build --fuzz-dir tests/fuzz fuzz_encode
   cargo +nightly fuzz build --fuzz-dir tests/fuzz fuzz_roundtrip
   cargo +nightly fuzz build --fuzz-dir tests/fuzz fuzz_encode_safety
   cargo +nightly fuzz build --fuzz-dir tests/fuzz fuzz_roundtrip_safety
   cargo +nightly fuzz build --fuzz-dir tests/fuzz fuzz_encode_multiframe
   ```

3. Re-ran every repro against its matching fresh binary:

   | Target | Repro | Result |
   |--|--|--|
   | fuzz_encode_safety | `crash_24da1ad627e2a5a7.bin` | clean exit (232 ms) |
   | fuzz_encode | `crash_44ecced56f288382.bin` | clean exit (166 ms) |
   | fuzz_roundtrip | `crash_8606db8d3057c654.bin` | clean exit (31 ms) |
   | fuzz_encode_multiframe | `crash_cde9a65042489ae3.bin` | clean exit (100 ms) |
   | fuzz_roundtrip_safety | `crash_f727a37351a8ab5a.bin` | clean exit (122 ms) |

   No more "attempt to subtract with overflow". (The differential
   targets may still mismatch on bitstream bytes because Stage 6.4 is
   only at the "tier-1 95% floor" — see the campaign 9 journal — but
   the panic itself is gone.)

## Related spots not addressed

A quick scan of the neighbourhood of `b - NB_TBANDS`-style indexing
in `analysis.rs` turned up one echo:

- `ropus/src/opus/analysis.rs:1258`: `(b as f32 - NB_TBANDS as f32)`
  inside the `max_frame_tonality` update. Safe — cast to `f32` first,
  signed-float arithmetic.

No other occurrences. The surrounding `leakage_from[b-1]` /
`band_log2[b-1]` / `TBANDS[b-1]` references are all guarded by
`b >= 1`.

## Files touched

- `C:\language\mdopus\ropus\src\opus\analysis.rs:1247-1254` — fix.
- `C:\language\mdopus\notes\campaign9_bug_L.md` — this file (full rewrite).
- `C:\language\mdopus\wrk_journals\2026.04.19 - JRN - fuzz campaign 9.md`
  — Bug L section added.
