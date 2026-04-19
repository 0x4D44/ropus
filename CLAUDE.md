# ropus - Opus Audio Codec in Rust

Production-quality Rust port of the [xiph/opus](https://github.com/xiph/opus) C codec (fixed-point).
~36k lines of Rust across 36 source files covering all 26 modules: CELT, SILK, Opus, and DNN.

Published at [github.com/0x4D44/ropus](https://github.com/0x4D44/ropus). Reference C source lives in `reference/` (git-ignored; clone from xiph/opus).

Note: the workspace directory on disk is still named `mdopus/` — only the crate/repo identity was renamed to `ropus`.

## Project Status

All 26 modules ported. Tier-1 (bit-exact against the C reference) on range
coder, CELT primitives, integer paths, RDOVAE forward pass, and the DRED
range-coded payload. Tier-2 (SNR-bounded, typically >= 50/60 dB) where
float-path drift exists: Analysis, LPCNet features, DEEP_PLC, and DRED
WAV-level reconstruction.

Currently in the **integration** phase — full encode/decode round-trip validation.

## Architecture

```
mdopus/                     # workspace root (disk name)
├── Cargo.toml              # workspace manifest
├── ropus/                  # published library crate (name = "ropus")
│   └── src/
│       ├── lib.rs          # Crate root, module declarations
│       ├── types.rs        # Fixed-point types, Q-format macros (arch.h, fixed_generic.h)
│       ├── celt/           # CELT codec: range coder, FFT/MDCT, bands, pitch, VQ, encoder/decoder
│       ├── silk/           # SILK codec: tables, common utils, encoder, decoder
│       ├── opus/           # Opus top-level: encoder, decoder, multistream, repacketizer
│       └── dnn/            # Neural enhancement: core inference, LPCNet, FARGAN, OSCE, DRED, etc.
├── ropus-cli/              # end-user CLI: encode/decode/transcode/play
├── harness/                # FFI comparison harness (builds C reference via cc crate)
│   ├── build.rs            # Compiles xiph/opus C source into libopus_ref
│   ├── src/main.rs         # ropus-compare CLI: encode/decode/roundtrip, byte-for-byte compare
│   └── src/bin/            # repro/trace/fuzz-replay binaries
├── capi/                   # C ABI shim exposing the Rust codec through the libopus API
├── tests/
│   ├── conformance/        # workspace member: conformance suite
│   ├── fuzz/               # cargo-fuzz targets (excluded from workspace)
│   └── vectors/            # deterministic WAV fixtures
├── assets/                 # Architecture docs and code review notes per module
├── tools/                  # Coordinator, integration scripts, bisect/trace utilities
├── notes/                  # Working scratchpad and investigation logs
├── wrk_docs/ wrk_journals/ # HLDs and investigation journals
└── logs/                   # Coordinator run logs (git-ignored)
```

## Development Workflow

Multi-agent orchestration:
- **Claude CLI** (`claude -p`) — primary implementation agent
- **Codex CLI** (`codex exec`) — adversarial code review
- **Coordinator** (`tools/coordinator.py`) — orchestrates phases and gates

Phases: Document → HLD → Test Harness → Implement → Integrate

## Implementation Order

Modules ported bottom-up by dependency:

### Core codec
1. Range coder (entropy) — `celt/range_coder.rs`, `ec_ctx.rs`
2. Math utilities — `celt/math_ops.rs`, `lpc.rs`
3. Combinatorial coding — `celt/cwrs.rs`
4. Band processing — `celt/bands.rs`
5. FFT/MDCT transforms — `celt/fft.rs`, `mdct.rs`
6. Pitch detection — `celt/pitch.rs`
7. Band quantization — `celt/quant_bands.rs`
8. Vector quantization — `celt/vq.rs`
9. Rate allocation — `celt/rate.rs`
10. Mode configuration — `celt/modes.rs`
11. CELT decoder — `celt/decoder.rs`
12. CELT encoder — `celt/encoder.rs`
13. SILK common — `silk/common.rs`, `tables.rs`
14. SILK decoder — `silk/decoder.rs`
15. SILK encoder — `silk/encoder.rs`
16. Opus decoder — `opus/decoder.rs`
17. Opus encoder — `opus/encoder.rs`
18. Multistream — `opus/multistream.rs`
19. Repacketizer — `opus/repacketizer.rs`

### DNN (neural enhancement, Opus 1.4+)
20. DNN core — `dnn/core.rs` (inference engine)
21. LPCNet — `dnn/lpcnet.rs`
22. FARGAN — `dnn/fargan.rs`
23. Lossgen — `dnn/lossgen.rs`
24. PitchDNN — `dnn/pitchdnn.rs`

## Coding Rules

- Safe Rust only; `unsafe` permitted only for FFI to C reference in test harness
- **Bit-exact numerical output** is a hard requirement — every sample must match
- Fixed-point arithmetic must match C reference precisely (uses `OPUS_FAST_INT64` path)
- SIMD enabled by default via the `wide` crate (safe, portable); no platform-specific intrinsics (ARM, MIPS, x86 skipped)
- Match C reference API semantics; Rust API can be idiomatic on top
- Clippy allows for C-port patterns (see `lib.rs` preamble)

## Testing

The comparison harness (`harness/`, crate `ropus-harness`) links the C reference via FFI and compares
outputs directly in-process:

```
ropus-compare encode <input.wav> [--bitrate N] [--complexity N]
ropus-compare decode <input.opus>
ropus-compare roundtrip <input.wav> [--bitrate N]
```

Test modes:
- **Encode**: WAV → compressed bytes, compare C vs Rust output byte-for-byte
- **Decode**: Opus → PCM samples, compare sample-for-sample
- **Round-trip**: encode then decode, compare final PCM

## Commands

```bash
# Build
cargo build

# Run comparison harness
cargo run --bin ropus-compare -- encode tests/vectors/48k_sine1k_loud.wav
cargo run --bin ropus-compare -- roundtrip tests/vectors/48k_sine1k_loud.wav

# Coordinator
python tools/coordinator.py run
python tools/coordinator.py status

# Code coverage
cargo llvm-cov
```
