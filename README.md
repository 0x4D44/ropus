# ropus

A Rust port of [xiph/opus](https://github.com/xiph/opus) with a bit-exact comparison workflow against the C reference codec.

For library usage and the crates.io quickstart, see [`ropus/README.md`](ropus/README.md) or the crates.io page.

## What this project is

This repository is a Cargo workspace: the published library crate lives in `ropus/`, with a C comparison harness, four end-user CLIs, a foobar2000 plugin, and a C ABI shim alongside for parity testing and integration.

- `ropus/src/` contains codec modules (`celt`, `silk`, `opus`, `dnn`) and shared types in `types.rs`.
- End-user CLI binaries, one crate per command: `ropusenc/` (encode any symphonia-supported input to Ogg Opus), `ropusdec/` (decode Ogg Opus to WAV or raw PCM), `ropusinfo/` (print stream metadata), `ropusplay/` (play via the default output device).
- `ropus-fb2k/` is a foobar2000 input-component backend — Ogg Opus demux + ropus decoder behind a stable C ABI — paired with the C++ SDK adapter in `fb2k-ropus/`.
- `harness/` builds C Opus as `opus_ref` and runs `ropus-compare`.
- `capi/` exposes the Rust codec through the libopus C ABI.
- `tests/conformance/` is the workspace conformance suite; `tests/vectors/` stores deterministic WAV fixtures.
- `tools/` hosts the coordination script for multi-step implementation workflows.

## Quick setup

```bash
git clone https://github.com/0x4D44/ropus.git
cd ropus

# Fetch external build assets (pinned, idempotent, cached).
#   `reference` clones xiph/opus at a pinned commit.
#   `weights`   downloads the DNN model tarball (~135 MB) and extracts into reference/dnn/.
#   `all`       does both, in order.
cargo run -p fetch-assets -- all

cargo build
```

Required:

- Rust 2024 (`1.88+`; required for let-chains in the neural PLC wiring)
- C compiler toolchain (MSVC/GCC/Clang)
- `git`, `curl`, `tar` on `PATH` (all three ship with Windows 10 1803+, macOS, and
  every mainstream Linux distro — used by `fetch-assets`)
- `reference/` directory containing the Opus C source tree (produced by
  `cargo run -p fetch-assets -- reference`)

If you run `cargo build` before fetching the reference tree, the harness
`build.rs` degrades gracefully: it emits a `cargo:warning` pointing at the
`fetch-assets` command and sets `cfg(no_reference)`, so `cargo build` at the
workspace root succeeds and FFI-dependent binaries in `harness/` stub out.
Invoking a stubbed binary prints the fetch hint at runtime. Fetch the
reference first if you intend to drive the differential-comparison path.

## Core commands

```bash
cargo build
cargo run --bin ropus-compare -- encode tests/vectors/48k_sine1k_loud.wav
cargo run --bin ropus-compare -- decode tests/vectors/48k_impulse.wav
cargo run --bin ropus-compare -- roundtrip tests/vectors/48k_sine1k_loud.wav --bitrate 64000
cargo run --bin ropus-compare -- unit range_coder
cargo run --bin ropus-compare -- bench tests/vectors/48k_sine1k_loud.wav --iters 10
cargo test
cargo llvm-cov
```

`ropus-compare` also supports:
- `encode`, `decode`, `roundtrip`
- `framecompare`, `decodecompare`, `mathcompare`, `rngtest`, `bench`
- `unit <module>` for module checks

### End-user CLIs

Four workspace binaries wrap the codec for day-to-day use. Each lives in its
own crate (`ropusenc/`, `ropusdec/`, `ropusinfo/`, `ropusplay/`) and can be
built and run via `cargo run -p <name>` or installed with
`cargo install --path <crate>`.

```bash
cargo run -p ropusenc  -- input.wav -o output.opus           # encode → Ogg Opus
cargo run -p ropusdec  -- input.opus -o output.wav           # decode Ogg Opus → WAV
cargo run -p ropusinfo -- input.opus                         # print stream metadata
cargo run -p ropusplay -- input.opus                         # play via default output device
```

`ropusenc` accepts any input format handled by
[`symphonia`](https://crates.io/crates/symphonia) (WAV, FLAC, MP3, AAC, …)
and emits a standard Ogg-Opus file per RFC 7845. `ropusdec` can target
either 16-bit-PCM WAV, 32-bit-float WAV, or raw interleaved PCM.

### foobar2000 plugin

A Windows foobar2000 input-component backed by ropus lives in
`ropus-fb2k/` (Rust, decoder + Ogg demux behind a stable C ABI) plus
`fb2k-ropus/` (C++ SDK adapter). Build the C++ side against the foobar2000
SDK; it links the Rust static library produced by `cargo build -p ropus-fb2k`.

## Testing scope

We validate ropus against several independent oracles. This section documents what is and is not covered today, so the compatibility story is legible at a glance.

### What we test

- **Per-module unit tests.** Each `ropus/src/{celt,silk,opus,dnn}/` module ships Rust unit tests that exercise leaf functions against hand-specified expected values.
- **Differential FFI harness** (`harness/`). Links the C reference as `libopus_ref` and compares encode/decode output byte-for-byte. Drives `ropus-compare encode|decode|roundtrip|framecompare|decodecompare|mathcompare|rngtest`.
- **`cargo-fuzz` campaigns** (`tests/fuzz/`). Long-running differential fuzz with the C reference as oracle; weekly cadence documented in `tests/fuzz/README.md`.
- **Official xiph/opus test suite** (`tests/conformance/`). All 7 reference test binaries — `padding`, `decode`, `api`, `encode`, `extensions`, `ietf_vectors`, and `projection` — compile unmodified against our C ABI shim (`capi/`) and pass. This includes the full `regression_test()` body from `test_opus_encode.c` (11 historical crash repros from `opus_encode_regressions.c` compiled verbatim; 5 QEXT/DRED-gated branches compile out) and `test_opus_projection.c`'s ambisonics `mapping_matrix_*` math plus the public `opus_projection_*` encode/decode surface. Full breakdown in [`tests/conformance/README.md`](tests/conformance/README.md). Invoke with `cargo test -p conformance -- --test-threads=1` (the reference test harness has module-global state; single-threaded execution is required).
- **IETF RFC 6716 / RFC 8251 bitstream vectors.** The canonical spec-level oracle. The 12 reference bitstreams × {mono, stereo} are driven through `opus_demo -d` + `opus_compare` (both upstream `.c` files compiled verbatim) and asserted to pass the RFC-mandated quality threshold. Fetch once with `tools/fetch_ietf_vectors.sh` (or `.ps1` on Windows) — vectors are ~71 MB and stay out of the repo. Run as part of `cargo test -p conformance --test ietf_vectors -- --test-threads=1`. Status: all 24 vectors pass.
- **Ambisonics (`channel_mapping == 3`) roundtrip.** `harness/src/bin/projection_roundtrip.rs` verifies byte-exact encode and sample-exact decode across first-through-fifth-order ambisonic fixtures.

### What we don't test (and why)

- **`test_opus_dred.c`.** DRED (Deep REDundancy) is Opus 1.5+, not part of RFC 6716. The encoder-side port + decoder-parse/process path shipped under `wrk_docs/2026.04.19 - HLD - dred-port.md` (Stage 8). The one remaining piece — FARGAN-driven PCM reconstruction that `test_opus_dred.c` exercises — is tracked in [`wrk_docs/2026.04.19 - HLD - fargan-dred-joint-followup.md`](wrk_docs/2026.04.19%20-%20HLD%20-%20fargan-dred-joint-followup.md) and **currently deferred**. Rationale: zero in-tree consumers today — capi does not expose `opus_decoder_dred_decode{,24,_float}`, and no harness, CLI, or conformance target calls them — so landing it is strict libopus C-ABI parity rather than a correctness gate. Scope estimate: ~150-200 lines of Rust (shared inner + three thin i16/i32/f32 wrappers, following the existing `decode_native` pattern) plus tier-2 SNR calibration against a C-vs-C float-mode control run, roughly 1-2 days of focused work. Will land when capi drop-in parity becomes a ship requirement, or sooner if a consumer appears.
- **`test_opus_custom.c`.** Custom modes (`opus_custom_*`, `#ifdef CUSTOM_MODES`) allow non-standard frame sizes and are not interoperable with RFC 6716 streams. Effectively deprecated upstream and out of scope for a "public Opus" crate.
- **Platform-specific SIMD (NEON, SSE4.1, AVX2 intrinsics).** ropus uses the `wide` crate for portable SIMD. Hand-tuned platform paths are an intentional deferral — see `wrk_journals/` for rationale.
- **QEXT (Opus 2.0 extensions).** Not ported. All QEXT-gated branches of the conformance tests compile out with `ENABLE_QEXT` undefined.

## Repository layout

```text
ropus/            # workspace root
├── ropus/        # published library crate (name = "ropus")
│   └── src/
│       ├── celt/
│       ├── silk/
│       ├── opus/
│       ├── dnn/
│       ├── lib.rs
│       └── types.rs
├── ropusenc/     # CLI: encode → Ogg Opus (symphonia-backed input)
├── ropusdec/     # CLI: decode Ogg Opus → WAV or raw PCM
├── ropusinfo/    # CLI: Ogg Opus stream metadata
├── ropusplay/    # CLI: play audio via the default output device
├── ropus-fb2k/   # foobar2000 input-component backend (Rust, stable C ABI)
├── fb2k-ropus/   # foobar2000 SDK adapter (C++) loading ropus-fb2k
├── harness/      # comparison binary + bindings (ropus-compare)
├── capi/         # C ABI shim (libopus-compatible)
├── tests/
│   ├── conformance/  # workspace conformance suite
│   └── vectors/      # test media
└── tools/
    └── coordinator.py
```

## Performance

Rust / C reference ratio (lower is better) measured via `tools/bench_sweep.sh --iters=30`
on an Opus 1.5.2 C reference baseline, current `master`. Encode ratio is what you pay
to encode; decode ratio is what you pay to decode. <1.0 means Rust is *faster* than
the C reference (which dispatches to hand-tuned SSE4.1/AVX2 at runtime).

| Vector                   | Encode | Decode |
|--------------------------|:------:|:------:|
| SILK NB 8k mono noise    | 1.05×  | 0.69×  |
| SILK WB 16k mono noise   | 1.14×  | 0.89×  |
| Hybrid 24k mono noise    | 1.11×  | 0.90×  |
| CELT FB 48k mono noise   | 1.08×  | 0.92×  |
| CELT FB 48k stereo noise | 1.04×  | 0.94×  |
| CELT 48k sine 1k loud    | 0.94×  | 1.05×  |
| CELT 48k sweep           | 0.96×  | 0.99×  |
| CELT 48k square 1k       | 1.03×  | 1.03×  |
| SPEECH 48k mono (TTS)    | 0.84×  | 0.98×  |
| MUSIC 48k stereo         | 1.01×  | 0.98×  |
| **Mean**                 | **1.02×** | **0.95×** |

Three vectors encode *faster* than C (sine, sweep, SPEECH) with MUSIC
essentially at parity (1.01×); the remaining six run 3-14% slower, dominated
by SILK where the C reference dispatches to hand-tuned SSE. Decode is faster
or at parity on eight of ten vectors, with CELT full-band sine (1.05×) and
square (1.03×) running slightly behind. Full measurement log:
[`wrk_journals/2026.04.19 - JRN - avx2-baseline.md`](wrk_journals/2026.04.19%20-%20JRN%20-%20avx2-baseline.md).

### Non-standard release profile

Two deliberate tunings in the release profile, both empirically justified:

1. **`Cargo.toml` sets `lto = "thin"`**, not the more common `"fat"`. A bake-off
   across all four `{ThinLTO, FatLTO} × {no-PGO, PGO}` cells showed **ThinLTO
   without PGO** is the best mean performer by ~6–8 percentage points of encode
   throughput over FatLTO. PGO is marginal at best and actively harmful in some
   cases (CELT-sweep encode regression of +31pp under ThinLTO). Full data:
   [`wrk_journals/2026.04.19 - JRN - lto-pgo-bakeoff.md`](wrk_journals/2026.04.19%20-%20JRN%20-%20lto-pgo-bakeoff.md).

2. **`.cargo/config.toml` sets `target-cpu = "x86-64-v3"`** (Haswell 2013 /
   Excavator 2015 baseline — AVX2 + FMA3 + BMI1/2). This unblocks LLVM to emit
   modern SIMD from the existing `wide::i32x4` kernels and to vectorize scalar
   hot paths at 256-bit width. The one-line change shifted encode mean from
   1.19× to 1.02× and decode from 0.99× to 0.95× — larger than the combined
   effect of all three stages of indexing work. Full data:
   [`wrk_journals/2026.04.19 - JRN - avx2-baseline.md`](wrk_journals/2026.04.19%20-%20JRN%20-%20avx2-baseline.md).

PGO is still available opt-in via `tools/pgo_build.sh` for one-off measurements,
but is not part of the canonical build. Users on pre-2013 CPUs (Sandy/Ivy Bridge,
pre-Excavator AMD — ~2% of current x86 machines per Steam HW survey) can rebuild
with `RUSTFLAGS="-C target-cpu=x86-64-v2"` or `=x86-64`.

### Reproducing the numbers

```bash
cargo build --release --bin ropus-compare
bash tools/bench_sweep.sh --iters=50
```

The sweep runs both the C reference and the Rust port on each vector, reports
per-iter medians, and prints the ratio shown above.

## Development notes

- Keep codec changes parity-first: validate numeric behavior with the harness after each meaningful change.
- `unsafe` is expected mainly in harness/FFI glue; prefer safe implementations in codec modules.
- Commit messages in history are short and imperative (`Fix ...`, `Add ...`, `Enable ...`).

## License

This repo is an implementation based on the xiph/opus codebase. Refer to the upstream project license in the `reference/` source.
