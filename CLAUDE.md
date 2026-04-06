# mdopus - Opus Audio Codec in Rust

Production-quality Rust port of the [xiph/opus](https://github.com/xiph/opus) C codec (fixed-point).
~36k lines of Rust across 36 source files covering all 26 modules: CELT, SILK, Opus, and DNN.

Reference C source lives in `reference/` (git-ignored; clone from xiph/opus).

## Project Status

All 26 modules ported and passing bit-exact comparison tests against the C reference.
Currently in the **integration** phase — full encode/decode round-trip validation.

## Architecture

```
mdopus/
├── src/
│   ├── lib.rs              # Crate root, module declarations
│   ├── types.rs            # Fixed-point types, Q-format macros (arch.h, fixed_generic.h)
│   ├── celt/               # CELT codec: range coder, FFT/MDCT, bands, pitch, VQ, encoder/decoder
│   ├── silk/               # SILK codec: tables, common utils, encoder, decoder
│   ├── opus/               # Opus top-level: encoder, decoder, multistream, repacketizer
│   └── dnn/                # Neural enhancement: core inference, LPCNet, FARGAN, OSCE, DRED, etc.
├── tests/harness/          # FFI comparison harness (builds C reference via cc crate)
│   ├── build.rs            # Compiles xiph/opus C source into libopus_ref
│   ├── bindings.rs         # Raw FFI declarations for C reference API
│   └── main.rs             # CLI: encode/decode with both impls, compare byte-for-byte
├── assets/                 # Architecture docs and code review notes per module
├── tools/                  # Coordinator, integration scripts, bisect/trace utilities
├── notes/                  # Working scratchpad and investigation logs
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
- No platform-specific SIMD (ARM, MIPS, x86 intrinsics skipped)
- Match C reference API semantics; Rust API can be idiomatic on top
- Clippy allows for C-port patterns (see `lib.rs` preamble)

## Testing

The comparison harness (`tests/harness/`) links the C reference via FFI and compares
outputs directly in-process:

```
mdopus-compare encode <input.wav> [--bitrate N] [--complexity N]
mdopus-compare decode <input.opus>
mdopus-compare roundtrip <input.wav> [--bitrate N]
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
cargo run --bin mdopus-compare -- encode assets/test.wav
cargo run --bin mdopus-compare -- roundtrip assets/test.wav

# Coordinator
python tools/coordinator.py run
python tools/coordinator.py status

# Code coverage
cargo llvm-cov
```
