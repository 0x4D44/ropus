# mdopus - Opus Audio Codec in Rust

Production-quality Rust port of the [xiph/opus](https://github.com/xiph/opus) C codec.
Reference C source is in `reference/` (git-ignored, cloned from xiph/opus).

## Architecture

- `src/` - Rust implementation (the port)
- `reference/` - xiph/opus C source (read-only reference, git-ignored)
- `assets/` - Generated architecture docs and HLD
- `tools/` - Coordinator script and utilities
- `notes/` - Working scratchpad
- `logs/` - Coordinator run logs (git-ignored)

## Development Workflow

This project uses a multi-agent orchestration approach:
- **Claude CLI** (`claude -p`) is the primary implementation agent
- **Codex CLI** (`codex exec`) provides adversarial code review
- **Coordinator** (`tools/coordinator.py`) orchestrates phases and checks gates
- Phases: Document -> HLD -> Test Harness -> Implement -> Integrate
- The Rust implementation must produce **bit-exact output** matching the C reference

## Implementation Order

Modules are ported bottom-up by dependency:

### Core codec
1. Range coder (entropy coding) - `celt/entcode.c`, `entenc.c`, `entdec.c`
2. Math utilities - `celt/mathops.c`, `celt_lpc.c`
3. Combinatorial coding - `celt/cwrs.c`
4. Band processing - `celt/bands.c`
5. FFT/MDCT transforms - `celt/kiss_fft.c`, `mdct.c`
6. Pitch detection - `celt/pitch.c`
7. Band quantization - `celt/quant_bands.c`
8. Vector quantization - `celt/vq.c`
9. Mode configuration - `celt/modes.c`
10. CELT decoder - `celt/celt_decoder.c`
11. CELT encoder - `celt/celt_encoder.c`
12. SILK common - `silk/` (tables, utilities)
13. SILK decoder - `silk/` (decoder pipeline)
14. SILK encoder - `silk/` (encoder pipeline)
15. Opus decoder - `src/opus_decoder.c`
16. Opus encoder - `src/opus_encoder.c`
17. Multistream - `src/opus_multistream*.c`
18. Repacketizer - `src/repacketizer.c`

### DNN (neural enhancement, Opus 1.4+)
19. DNN core - `dnn/nnet.c`, `nndsp.c` (inference engine)
20. LPCNet - `dnn/lpcnet.c` (neural speech codec)
21. OSCE - `dnn/osce.c` (speech coding enhancement)
22. FARGAN - `dnn/fargan.c` (neural waveform generator)
23. FWGAN - `dnn/fwgan.c` (frequency-domain waveform generator)
24. DRED - `dnn/dred_*.c` (deep redundancy coding)
25. Lossgen - `dnn/lossgen.c` (packet loss generator)
26. PitchDNN - `dnn/pitchdnn.c` (neural pitch detection)

## Coding Rules

- Safe Rust only; `unsafe` permitted only for FFI to C reference in test harness
- Bit-exact numerical output is a hard requirement
- Fixed-point arithmetic must match C reference precisely
- No platform-specific SIMD in initial port (ARM, MIPS, x86 intrinsics skipped)
- Match C reference API semantics; Rust API can be idiomatic on top

## Testing

The test harness (`tests/compare/`) links the C reference via FFI and compares
outputs directly in-process. Test modes:
- Encode WAV -> compare compressed bytes
- Decode Opus -> compare PCM samples
- Round-trip: encode then decode, compare final PCM

## Commands

```bash
# Run coordinator
python tools/coordinator.py run

# Run specific phase
python tools/coordinator.py phase document

# Check status
python tools/coordinator.py status

# Build & test
cargo build
cargo test
```
