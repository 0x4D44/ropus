# Campaign 9 — "Bug L": not a bug

**Date**: 2026-04-19
**Status**: NOT A CODE BUG. Expected divergence from Stage 6 in-flight state.
**Disposition**: No fix applied. No commit. Update journal; optionally tighten
fuzz targets to skip analysis-triggering inputs (out of scope for this task).

## Summary

All four differential fuzz-target panics captured in the first 5 minutes of
campaign 9 have the same root cause: **the C reference encoder now runs
`analysis.c` / `mlp.c` (tonality analysis) at complexity >= 10 on
Fs in [16k, 48k], while the Rust port of `analysis.c` has not yet landed
(Stage 6.3 of the deferred-work closeout).** This divergence was introduced
*intentionally* by commit `3768181` ("build(harness): align scalar-C config
for Stage 6 analysis port") and is documented in the commit message,
`harness/config.h:20-25`, and
`wrk_journals/2026.04.19 - JRN - stage6-analysis-port.md`.

The task brief's "prime suspect" (the MLP port in commit `621105f` having a
subtle numerical bug) is incorrect. `ropus/src/opus/mlp.rs` and
`ropus/src/opus/mlp_data.rs` are **dead code** — neither is referenced
outside its own module. Grep confirms:

```
$ grep -rn "mlp::\|mlp_data::" ropus/src/
(no matches)
```

`ropus/src/opus/mod.rs` declares them `pub(crate) mod mlp; pub(crate) mod mlp_data;`
and nothing else in the encoder touches either module. MLP is inert until
Stage 6.3 (`analysis.c` port) and 6.4 (wire-in) land. Therefore MLP cannot
be the cause of any encoder bitstream divergence.

## Evidence

### 1. The intentional config change

`harness/config.h`, which `tests/fuzz/build.rs` includes via
`.include(&harness_dir)`, was modified by commit `3768181`:

```diff
-/* Disable float analysis API for bit-exact fixed-point comparison.
-   The Rust implementation doesn't have the analysis module, so we need
-   to disable it in C to get an apples-to-apples comparison. */
-#define DISABLE_FLOAT_API 1
+/* Float analysis API is enabled on both sides. ...
+   encode-comparison tests that exercise the analysis path will diverge. ...
+   Do NOT re-gate analysis by adding DISABLE_FLOAT_API back unless the
+   closeout is abandoned. */
```

Before this commit the C reference compiled `analysis.c` / `mlp.c` /
`mlp_data.c` as no-ops (gated by `#ifndef DISABLE_FLOAT_API`). After this
commit the C reference really runs tonality analysis and uses its output
to drive mode/bandwidth/trim decisions.

### 2. The analysis run condition

From `reference/src/opus_encoder.c:1250-1252`:

```c
#ifdef FIXED_POINT
    if (st->silk_mode.complexity >= 10 && st->Fs>=16000 && st->Fs<=48000
        && st->application != OPUS_APPLICATION_RESTRICTED_SILK)
```

Note the gate is `!= RESTRICTED_SILK` — not `!= RESTRICTED_LOWDELAY`.
LOWDELAY DOES run analysis. This refutes the brief's assumption that
"LOWDELAY bypasses analysis, so it's likely a distinct root cause."

All four crashes have `complexity == 10` and `Fs >= 16000`. Decoded from
the structured fuzz inputs:

| Crash | sr | ch | app | cx | Analysis runs? |
|-------|----|----|-----|-------|----------------|
| `crash_b8161d8197510a08` (encode) | 16000 | 2 | AUDIO | 10 | yes |
| `crash_c6c3ecaf8a8f274a` (roundtrip) | 16000 | 2 | AUDIO | 10 | yes |
| `crash_bf89cf0445559397` (roundtrip_safety LOWDELAY) | 16000 | 2 | LOWDELAY | 10 | **yes** |
| `crash_69b470b83a09824e` (encode_multiframe) | 24000 | 1 | AUDIO | 10 | yes |

### 3. How analysis perturbs the bitstream

The `AnalysisInfo` struct populated by `run_analysis` feeds three places:

1. **Mode selection** (`opus_encoder.c:1279-1305`): `analysis_info.music_prob`
   drives `voice_ratio`; `analysis_info.bandwidth` drives `detected_bandwidth`.
   These change the SILK/Hybrid/CELT-only mode pick. **Explains the AUDIO
   crashes where Rust TOC=188 (CELT-only) vs C TOC=79 (Hybrid) at 16kHz
   stereo 32kbps** — music_prob pushes C toward SILK-family, Rust (no
   analysis) picks CELT-only by default.

2. **DTX activity gating** (`opus_encoder.c:1461`): `useDTX &&
   !(analysis_info.valid || is_silence)` — only matters when DTX enabled.
   Not the driver here.

3. **CELT encoder** (`opus_encoder.c:2417-2418`):
   ```c
   if (redundancy || st->mode != MODE_SILK_ONLY)
      celt_encoder_ctl(celt_enc, CELT_SET_ANALYSIS(analysis_info));
   ```
   `alloc_trim_analysis()` in `reference/celt/celt_encoder.c:935-941` then
   reads `analysis->tonality_slope`:
   ```c
   if (analysis->valid) {
      trim -= MAX16(-QCONST16(2.f, 8), MIN16(QCONST16(2.f, 8),
            (opus_val16)(QCONST16(2.f, 8)*(analysis->tonality_slope+.05f))));
   }
   ```
   This shifts `trim_index` by 1-2 units, which directly changes CELT
   bitstream bytes. **Explains the LOWDELAY crash where the TOC matches
   (byte 0 = 188 on both sides, same CELT-only mode) but bytes 16+ diverge
   mid-payload.** Rust sets `analysis.valid = false` by default
   (`ropus/src/celt/encoder.rs:127,3192`), so the Rust trim path never
   takes the tonality-slope adjustment. **Same story for the multiframe
   crash: first 8 bytes match (shared TOC+header), divergence starts where
   CELT payload begins.**

### 4. Reproduction (all four panic as expected)

```
$ ASAN_DIR="/c/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"
$ export PATH="$ASAN_DIR:$PATH"
$ export ASAN_OPTIONS="detect_odr_violation=0:detect_leaks=0"
$ export FUZZ_PANIC_CAPTURE_DIR=/c/language/mdopus/fuzz_crashes

$ ./tests/fuzz/target/x86_64-pc-windows-msvc/release/fuzz_encode.exe \
    fuzz_crashes/crash_b8161d8197510a08.bin
thread '<unnamed>' panicked: Output byte mismatch at sr=16000, ch=2, app=2049, br=32000, cx=10, len=80
  left (Rust):  [188, ...]   # CELT-only FB stereo 20ms
 right (C):     [ 79, ...]   # Hybrid FB stereo

$ ./tests/fuzz/target/x86_64-pc-windows-msvc/release/fuzz_roundtrip_safety.exe \
    fuzz_crashes/crash_bf89cf0445559397.bin
thread '<unnamed>' panicked: LOWDELAY roundtrip: compressed output mismatch
  left:  [188, 127, 175, 212, 91, 87, 220, 5, 94, 236, 6, 60, 77, 155, 69, 152, 28, ...]
  right: [188, 127, 175, 212, 91, 87, 220, 5, 94, 236, 6, 60, 77, 155, 69, 152, 23, ...]
                                                                          ^^^ divergence at byte 16 (mid CELT payload)

$ ./tests/fuzz/target/x86_64-pc-windows-msvc/release/fuzz_encode_multiframe.exe \
    fuzz_crashes/crash_69b470b83a09824e.bin
thread '<unnamed>' panicked: Frame 1/7 byte mismatch: sr=24000, ch=1, app=2049, br=64000, cx=10
  TOCs match (216), payload diverges at byte 8.
```

## Conclusion

No code fix. No commit. The root cause is a deliberately-introduced
divergence between the C reference (analysis on) and the Rust encoder
(no analysis yet) that will be resolved when Stage 6.3 / 6.4 land. The
brief's MLP-is-buggy theory is refuted by MLP being dead code that no
call site references.

## Recommendations (out of scope for this task)

While this is not "bug L", the differential fuzz targets (`fuzz_encode`,
`fuzz_roundtrip`, `fuzz_encode_multiframe`, and `fuzz_roundtrip_safety`'s
LOWDELAY comparison block) will keep spuriously panicking on every
analysis-triggering input for the duration of Stage 6. Two mitigations
to consider (consult before implementing):

1. **Gate the differential assertion** at the top of each differential
   fuzz target:

   ```rust
   // Stage 6 in flight: C runs analysis, Rust doesn't. Skip inputs that
   // would trigger the tonality analyzer — they'll spuriously mismatch.
   let analysis_runs =
       complexity >= 10 && sample_rate >= 16000 && sample_rate <= 48000
       && application != OPUS_APPLICATION_RESTRICTED_SILK;
   if analysis_runs {
       return;
   }
   ```

   This lets the fuzzers productively explore the SILK-only / low-complexity
   space while Stage 6 is in flight.

2. **Alternatively**, re-enable `DISABLE_FLOAT_API 1` in the fuzz build
   only (fork `tests/fuzz/config.h` from `harness/config.h` and add the
   define back). This restores apples-to-apples comparison in the fuzz
   setup while preserving the Stage 6 red-signal in the harness
   differential tests. The Stage 6 journal explicitly asks not to do
   this for the harness, but the fuzz setup is a separate consumer and
   the reasoning (visible red signal) is already provided by the harness.

Either mitigation should be coordinated with the campaign 9 operator
and the Stage 6 agent — out of scope for this task per guardrails.

## Also: the 5 running fuzzers are fine

`fuzz_decode`, `fuzz_decode_safety`, `fuzz_encode_safety`,
`fuzz_repacketizer`, `fuzz_packet_parse` are either pure-Rust safety
targets or decode-side differentials. Decode doesn't run analysis
(it's encoder-only), and the safety targets don't do differential
comparison. So they're unaffected by this divergence and should keep
running.

## Files referenced

- `C:\language\mdopus\harness\config.h:20-25` — the warning against
  re-adding DISABLE_FLOAT_API.
- `C:\language\mdopus\reference\src\opus_encoder.c:1247-1263` —
  analysis run condition.
- `C:\language\mdopus\reference\src\opus_encoder.c:2416-2419` —
  CELT_SET_ANALYSIS wiring.
- `C:\language\mdopus\reference\celt\celt_encoder.c:935-941` —
  tonality_slope trim adjustment.
- `C:\language\mdopus\ropus\src\opus\mod.rs:4-5` — mlp / mlp_data
  declared but no call sites.
- `C:\language\mdopus\ropus\src\opus\encoder.rs:2564-2568` — Rust
  encoder never calls `SetAnalysis` on CELT.
- `C:\language\mdopus\wrk_journals\2026.04.19 - JRN - stage6-analysis-port.md`
  — Stage 6 plan and known-divergence disclaimer.
- Fuzz crash artifacts: `C:\language\mdopus\fuzz_crashes\crash_*.bin`.
