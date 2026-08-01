# ROP-BUG-FLUX-00046 — Play and Opus transcode ignore OpusHead output gain

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-tools-core/shared-decode-gain
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T215919Z-p58759-n146521000-c1 branch=task/bug-ROP-BUG-FLUX-00046-run-fix-20260801T215919Z-p58759-n146521000-c1 code=127a7c07c7a70040154057ea800e9b0de2073b61 gate=manual) -> Closed (2026-08-02, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main 3528b9e; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main ac7ff8a. Direct decode applies OpusHead output_gain at /Users/md/language/ropus/ropus-tools-core/src/commands/decode.rs:167-175. The separate shared decoder at /Users/md/language/ropus/ropus-tools-core/src/audio/decode.rs:42-49,111-147,196-225 stores only pre_skip and never calls set_gain. ropusplay and Opus-to-Opus encode use that path at commands/play.rs:132-177 and commands/encode.rs:112-128. Valid nonzero-gain files therefore play or transcode at a different amplitude than ropusdec. Fix: parse the full header, apply Q8 output gain once, and add paired zero/-6 dB shared-decode fixtures. Static review only; no playback, transcode, or test ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-02, independent two-eyes, host flux)

- Fix commit `127a7c0` adds `decode_to_f32_with_gain` (Opus header + user gain applied via
  `Decoder::set_gain`) to `ropus-tools-core/src/audio/decode.rs` and wires it into
  `commands/play.rs`.
- Regression re-verified by construction: the fix's new test
  `shared_decode_applies_header_and_user_gain_for_playback_and_transcode` was spliced onto
  the pre-fix tree at `127a7c07~1` — it failed to compile (`cannot find function
  decode_to_f32_with_gain`), confirming the shared decode path had no gain-application
  capability at all before this fix. The test passes at the current tree.
- `cargo clippy -p ropus-tools-core --all-targets --locked -- -D warnings` clean; `cargo test
  -p ropus-tools-core --locked`: 129 lib + 22 integration passed, 0 failed.

### `ropusplay --gain` is not decoder gain (2026-07-31)

The signed parity design requires decoder `set_gain` semantics, but
`/Users/md/language/ropus/ropus-tools-core/src/commands/play.rs:153-158,227-255`
multiplies already decoded float PCM. The actual decoder applies Q8 gain before
the saturating i16 clamp at
`/Users/md/language/ropus/ropus/src/opus/decoder.rs:1152-1159,1382-1411`.
Positive gain can therefore clip and quantize differently from `ropusdec`, in
addition to omitting `OpusHead.output_gain`. Combine header and user Q8 gain in
the decoder, define non-Opus policy, and add zero, negative, and positive
saturation fixtures plus a CLI mapping case. Static review at `origin/main`
`e5d7113`; no decoder, player, or test ran.
