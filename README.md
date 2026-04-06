# mdopus

A production-quality Rust port of the [Opus audio codec](https://opus-codec.org/) (xiph/opus), targeting **bit-exact** output against the C reference implementation.

## What is Opus?

[Opus](https://opus-codec.org/) is the leading open audio codec, standardized as [RFC 6716](https://tools.ietf.org/html/rfc6716). It handles everything from low-bitrate speech (6 kbps) to high-fidelity music (510 kbps), with frame sizes from 2.5 ms to 60 ms, and supports 1–255 channels. Opus is used by WebRTC, Discord, Zoom, YouTube, Spotify, and most modern VoIP/streaming platforms.

Opus combines two codecs internally:
- **SILK** — optimized for speech (derived from the Skype codec)
- **CELT** — optimized for music and general audio (low-delay MDCT codec)

The Opus layer dynamically switches between SILK, CELT, or a hybrid of both depending on the audio content, bitrate, and latency requirements.

## Why a Rust port?

- **Memory safety** without garbage collection — eliminates entire classes of C bugs (buffer overflows, use-after-free, undefined behavior from signed overflow)
- **Bit-exact compatibility** — not an approximation or reimplementation; every PCM sample and compressed byte matches the C reference
- **Single-crate, safe Rust** — no `unsafe` in the codec itself (only in the FFI test harness)
- **No SIMD, no platform-specific code** — portable fixed-point implementation

## Project Status

| Component | Modules | Status |
|-----------|---------|--------|
| CELT codec | Range coder, FFT/MDCT, bands, pitch, VQ, encoder, decoder | All passing |
| SILK codec | Tables, common utils, encoder, decoder | All passing |
| Opus layer | Encoder, decoder, multistream, repacketizer | All passing |
| DNN/neural | Core inference, LPCNet, FARGAN, OSCE, DRED, lossgen, pitchdnn | All passing |

All 26 modules pass bit-exact comparison tests against the C reference. Currently in integration testing — validating full encode/decode round-trips across diverse audio content and codec configurations.

## Architecture

```
mdopus/
├── src/
│   ├── lib.rs          # Crate root
│   ├── types.rs        # Fixed-point types and Q-format arithmetic
│   ├── celt/           # CELT codec (17 files, ~13k lines)
│   │   ├── range_coder.rs, ec_ctx.rs    # Entropy coding
│   │   ├── fft.rs, mdct.rs              # Transforms
│   │   ├── bands.rs, pitch.rs, vq.rs    # Signal processing
│   │   ├── encoder.rs, decoder.rs       # CELT enc/dec
│   │   └── ...                          # modes, tables, rate, quant_bands, etc.
│   ├── silk/           # SILK codec (5 files, ~10k lines)
│   │   ├── common.rs, tables.rs         # Shared utilities and lookup tables
│   │   ├── encoder.rs                   # SILK encoder pipeline
│   │   └── decoder.rs                   # SILK decoder pipeline
│   ├── opus/           # Opus top-level (5 files, ~7k lines)
│   │   ├── encoder.rs, decoder.rs       # Mode switching, redundancy, PLC
│   │   ├── multistream.rs              # Multi-channel support
│   │   └── repacketizer.rs             # Packet manipulation
│   └── dnn/            # Neural enhancement (6 files, ~5k lines)
│       ├── core.rs                      # Inference engine (dense, GRU, conv)
│       ├── lpcnet.rs, fargan.rs         # Neural vocoders
│       └── pitchdnn.rs, lossgen.rs      # Neural pitch, loss modeling
└── tests/harness/      # FFI comparison against C reference
    ├── build.rs        # Compiles xiph/opus via cc crate
    ├── bindings.rs     # Raw FFI declarations
    └── main.rs         # CLI comparison tool
```

**~36,000 lines of Rust** across 36 source files, ported from the xiph/opus C codebase.

## Building

### Prerequisites

- **Rust** 2024 edition (1.85+)
- **C compiler** (MSVC, GCC, or Clang) — needed to build the C reference for testing
- **xiph/opus source** — cloned into `reference/`

### Setup

```bash
# Clone this repo
git clone https://github.com/user/mdopus.git
cd mdopus

# Clone the C reference (needed for the comparison test harness)
git clone https://github.com/xiph/opus.git reference

# Build
cargo build

# Run tests
cargo run --bin mdopus-compare -- roundtrip path/to/test.wav
```

### Build Configuration

The project uses fixed-point arithmetic (`FIXED_POINT` in C terms) with 64-bit intermediates (`OPUS_FAST_INT64`). Debug builds disable overflow checks to match C wrapping semantics:

```toml
[profile.dev]
overflow-checks = false
```

## Testing

The test harness compiles the C reference into a static library via the `cc` crate, then calls both implementations on the same input and compares outputs byte-for-byte.

```bash
# Encode a WAV file with both C and Rust, compare compressed output
cargo run --bin mdopus-compare -- encode input.wav

# Decode an Opus file with both, compare PCM output
cargo run --bin mdopus-compare -- decode input.opus

# Full round-trip: encode then decode, compare final PCM
cargo run --bin mdopus-compare -- roundtrip input.wav --bitrate 64000

# Code coverage
cargo llvm-cov
```

Output shows byte-level / sample-level match statistics with hex dumps at any divergence point.

## How the Port Works

This is a **structural port**, not a rewrite. Each C function maps to a Rust function with identical logic, preserving:

- Exact integer arithmetic (wrapping, truncation, shift behavior)
- Exact floating-point literals (coefficient tables, conversion constants)
- Control flow structure (loop bounds, branch order, early returns)
- Variable naming (adapted to snake_case but recognizable)

The port uses the fixed-point code paths exclusively — no floating-point codec paths are ported. Platform-specific SIMD intrinsics (ARM NEON, x86 SSE, MIPS DSP) are skipped; all operations use portable C/Rust equivalents.

### Bit-exactness Guarantee

The C reference relies on specific integer overflow wrapping behavior (undefined in C, but well-defined on all target platforms). Rust's `Wrapping` semantics match this exactly. The test harness validates that for any input:

- `rust_encode(input) == c_encode(input)` (byte-for-byte)
- `rust_decode(input) == c_decode(input)` (sample-for-sample)

## Development Process

This port was built using a multi-agent AI orchestration approach:

1. **Document** — Architecture analysis of each C module
2. **HLD** — High-level design mapping C structures to Rust idioms
3. **Test Harness** — FFI comparison framework built first
4. **Implement** — Bottom-up port, module by module, validated against C reference
5. **Integrate** — Full codec round-trip testing

Agents:
- **Claude CLI** — primary implementation
- **Codex CLI** — adversarial code review
- **Coordinator** (`tools/coordinator.py`) — phase orchestration and gate checks

## License

This port follows the licensing of the original [xiph/opus](https://github.com/xiph/opus) codec, which is released under a BSD-style license. See the opus project for full license terms.

## Acknowledgments

- [Xiph.Org Foundation](https://xiph.org/) for the Opus codec
- [Jean-Marc Valin](https://jmvalin.ca/), Koen Vos, and Timothy B. Terriberry — original Opus authors
- [RFC 6716](https://tools.ietf.org/html/rfc6716) — Opus codec specification
