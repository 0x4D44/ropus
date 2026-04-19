# ropus

A Rust port of [xiph/opus](https://github.com/xiph/opus) with a bit-exact comparison workflow against the C reference codec.

For library usage and the crates.io quickstart, see [`ropus/README.md`](ropus/README.md) or the crates.io page.

## What this project is

This repository is a Cargo workspace: the published library crate lives in `ropus/`, with a C comparison harness, end-user CLI, and C ABI shim alongside for parity testing and integration.

- `ropus/src/` contains codec modules (`celt`, `silk`, `opus`, `dnn`) and shared types in `types.rs`.
- `ropus-cli/` is the end-user CLI (encode/decode/transcode/play).
- `harness/` builds C Opus as `opus_ref` and runs `ropus-compare`.
- `capi/` exposes the Rust codec through the libopus C ABI.
- `tests/conformance/` is the workspace conformance suite; `tests/vectors/` stores deterministic WAV fixtures.
- `tools/` hosts the coordination script for multi-step implementation workflows.

## Quick setup

```bash
git clone https://github.com/0x4D44/ropus.git
cd ropus

git clone https://github.com/xiph/opus.git reference

cargo build
```

Required:

- Rust 2024 (`1.85+`)
- C compiler toolchain (MSVC/GCC/Clang)
- `reference/` directory containing the Opus C source tree

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
├── ropus-cli/    # end-user CLI (encode/decode/transcode/play)
├── harness/      # comparison binary + bindings (ropus-compare)
├── capi/         # C ABI shim (libopus-compatible)
├── tests/
│   ├── conformance/  # workspace conformance suite
│   └── vectors/      # test media
└── tools/
    └── coordinator.py
```

## Performance

Rust / C reference ratio (lower is better) measured via `tools/bench_sweep.sh --iters=50`
on an Opus 1.5.2 C reference baseline, current `master` with Stage 1–3 indexing
work landed. Encode ratio is what you pay to encode; decode ratio is what you pay
to decode. <1.0 means Rust is *faster* than the C reference (SSE path).

| Vector                   | Encode | Decode |
|--------------------------|:------:|:------:|
| SILK NB 8k mono noise    | 1.06×  | 0.68×  |
| SILK WB 16k mono noise   | 1.42×  | 0.93×  |
| Hybrid 24k mono noise    | 1.34×  | 0.94×  |
| CELT FB 48k mono noise   | 1.27×  | 0.97×  |
| CELT FB 48k stereo noise | 1.17×  | 1.00×  |
| CELT 48k sine 1k loud    | 1.04×  | 1.11×  |
| CELT 48k sweep           | 1.06×  | 1.04×  |
| CELT 48k square 1k       | 1.28×  | 1.10×  |
| SPEECH 48k mono (TTS)    | 1.21×  | 1.09×  |
| MUSIC 48k stereo         | 1.09×  | 1.01×  |
| **Mean**                 | **1.19×** | **0.99×** |

### Non-standard release profile

`Cargo.toml` sets `lto = "thin"`, not the more common `"fat"`. This is deliberate:
an empirical bake-off across all four `{ThinLTO, FatLTO} × {no-PGO, PGO}` cells
showed **ThinLTO without PGO** is the best mean performer on this codec —
by ~6-8 percentage points of encode throughput over FatLTO. PGO is marginal
at best and actively harmful in some cases (it introduces a CELT-sweep encode
regression of +31pp under ThinLTO). Full data and rationale live in
[`wrk_journals/2026.04.19 - JRN - lto-pgo-bakeoff.md`](wrk_journals/2026.04.19%20-%20JRN%20-%20lto-pgo-bakeoff.md).

PGO is still available opt-in via `tools/pgo_build.sh` for one-off
measurements, but is not part of the canonical build.

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
