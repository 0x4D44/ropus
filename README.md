# mdopus

A Rust port of [xiph/opus](https://github.com/xiph/opus) with a bit-exact comparison workflow against the C reference codec.

For library usage and the crates.io quickstart, see [`ropus/README.md`](ropus/README.md) or the crates.io page.

## What this project is

mdopus is a Cargo workspace: the published library crate lives in `ropus/`, with a C comparison harness alongside for parity testing.

- `ropus/src/` contains codec modules (`celt`, `silk`, `opus`, `dnn`) and shared types in `types.rs`.
- `harness/` builds C Opus as `opus_ref` and runs `ropus-compare`.
- `tests/vectors/` stores deterministic WAV fixtures used by CLI checks.
- `tools/` hosts the coordination script for multi-step implementation workflows.

## Quick setup

```bash
git clone <repo-url> mdopus
cd mdopus

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
mdopus/
├── ropus/         # published library crate
│   └── src/
│       ├── celt/
│       ├── silk/
│       ├── opus/
│       ├── dnn/
│       ├── lib.rs
│       └── types.rs
├── harness/       # comparison binary + bindings
├── tests/
│   └── vectors/   # test media
└── tools/
    └── coordinator.py
```

## Development notes

- Keep codec changes parity-first: validate numeric behavior with the harness after each meaningful change.
- `unsafe` is expected mainly in harness/FFI glue; prefer safe implementations in codec modules.
- Commit messages in history are short and imperative (`Fix ...`, `Add ...`, `Enable ...`).

## License

This repo is an implementation based on the xiph/opus codebase. Refer to the upstream project license in the `reference/` source.
