I now have all the information needed. Here is the complete HLD document.

---

# High-Level Design: mdopus — Opus Audio Codec in Rust

## 1. Rust Module Structure

### 1.1 Crate Organization

Single crate (`mdopus`) with internal modules. This keeps build simplicity while allowing fine-grained visibility control. A workspace split (e.g., separate `mdopus-celt`, `mdopus-silk` crates) adds compile-time cost and cross-crate generic friction without proportional benefit at this scale.

```
mdopus/
├── src/
│   ├── lib.rs                     # Public API, re-exports
│   ├── types.rs                   # Primitive types, Q-format newtypes, macros
│   ├── error.rs                   # OpusError enum, Result alias
│   │
│   ├── entropy/
│   │   ├── mod.rs
│   │   ├── range_coder.rs         # ec_ctx → RangeEncoder, RangeDecoder
│   │   ├── laplace.rs             # Laplace entropy coding
│   │   └── tables.rs              # Correction tables, SMALL_DIV_TABLE
│   │
│   ├── celt/
│   │   ├── mod.rs
│   │   ├── math_ops.rs            # isqrt32, celt_rcp, celt_sqrt, celt_log2, celt_exp2, trig
│   │   ├── lpc.rs                 # _celt_lpc, celt_fir, celt_iir, _celt_autocorr
│   │   ├── cwrs.rs                # PVQ codebook enumeration (encode/decode_pulses)
│   │   ├── bands.rs               # Band processing, normalization, quant_all_bands
│   │   ├── fft.rs                 # kiss_fft forward/inverse
│   │   ├── mdct.rs                # clt_mdct_forward/backward
│   │   ├── pitch.rs               # pitch_downsample, pitch_search, remove_doubling, xcorr
│   │   ├── quant_bands.rs         # Energy quantization, amp2Log2, coarse/fine/finalise
│   │   ├── vq.rs                  # PVQ search, alg_quant, alg_unquant, exp_rotation
│   │   ├── rate.rs                # Bit allocation (clt_compute_allocation)
│   │   ├── modes.rs               # CELTMode, static mode tables
│   │   ├── decoder.rs             # CeltDecoder (PLC, comb filter, synthesis)
│   │   ├── encoder.rs             # CeltEncoder (pre-filter, transient, VBR)
│   │   └── tables.rs              # Static tables (PVQ_U_DATA, eBands, allocVectors, etc.)
│   │
│   ├── silk/
│   │   ├── mod.rs
│   │   ├── common.rs              # Tables, sorting, bwexpander, lin2log, sigm_Q15
│   │   ├── nlsf.rs                # NLSF codec, codebooks, A2NLSF, NLSF2A
│   │   ├── resampler.rs           # Polyphase resampler
│   │   ├── lpc.rs                 # SILK LPC analysis, inverse_pred_gain, LPC_fit
│   │   ├── pitch.rs               # SILK pitch estimation, lag codebook
│   │   ├── noise_shape.rs         # NSQ, NSQ_del_dec
│   │   ├── shell_coder.rs         # Shell coding for excitation pulses
│   │   ├── stereo.rs              # LR↔MS, stereo prediction
│   │   ├── vad.rs                 # Voice activity detection
│   │   ├── plc.rs                 # Packet loss concealment
│   │   ├── cng.rs                 # Comfort noise generation
│   │   ├── decoder.rs             # SilkDecoder (decode_frame, decode_core)
│   │   ├── encoder.rs             # SilkEncoder (encode_frame, indices, pulses)
│   │   └── tables.rs              # iCDF tables, gain tables, LTP codebooks
│   │
│   ├── opus/
│   │   ├── mod.rs
│   │   ├── packet.rs              # TOC parsing, packet inspection utilities
│   │   ├── decoder.rs             # OpusDecoder (mode switching, redundancy, PLC)
│   │   ├── encoder.rs             # OpusEncoder (mode/bandwidth selection, analysis)
│   │   ├── multistream.rs         # Multistream encoder/decoder, channel mapping
│   │   ├── projection.rs          # Ambisonics projection matrices
│   │   └── repacketizer.rs        # Packet merge/split/pad, extension iterator
│   │
│   └── dnn/                       # Phase 2 (deferred)
│       ├── mod.rs
│       ├── nnet.rs                # Dense, GRU, Conv1D/2D, GLU primitives
│       ├── weights.rs             # Weight blob parsing, LinearLayer
│       ├── lpcnet.rs              # LPCNet features, synthesis, PLC
│       ├── osce.rs                # LACE/NoLACE/BBWENet post-filter
│       ├── fargan.rs              # FARGAN vocoder
│       ├── fwgan.rs               # FWGAN vocoder
│       ├── dred.rs                # Deep redundancy (RDOVAE)
│       ├── pitchdnn.rs            # Neural pitch estimator
│       └── lossgen.rs             # Packet loss generator (test utility)
│
├── tests/
│   └── compare/                   # FFI comparison harness
│       ├── build.rs               # cc crate builds reference C
│       ├── ffi.rs                 # Unsafe bindings to C reference
│       └── *.rs                   # Per-module comparison tests
│
└── Cargo.toml
```

### 1.2 Dependency Graph

```
                            ┌─────────────────────────┐
                            │       opus/encoder       │
                            │       opus/decoder       │
                            │     opus/multistream     │
                            │    opus/repacketizer     │
                            └──────┬──────────┬────────┘
                                   │          │
                    ┌──────────────┘          └──────────────┐
                    ▼                                        ▼
            ┌───────────────┐                       ┌───────────────┐
            │ silk/encoder  │                       │ celt/encoder  │
            │ silk/decoder  │                       │ celt/decoder  │
            └──────┬────────┘                       └──────┬────────┘
                   │                                       │
        ┌──────────┼──────────┐              ┌─────────────┼──────────────┐
        ▼          ▼          ▼              ▼             ▼              ▼
   silk/nlsf  silk/nsq  silk/vad      celt/bands    celt/quant_bands  celt/rate
   silk/pitch silk/plc  silk/stereo        │              │
   silk/cng   silk/shell               ┌───┴───┐    ┌────┴────┐
        │          │                   ▼       ▼    ▼         ▼
        └────┬─────┘              celt/vq  celt/fft  celt/pitch
             ▼                       │    celt/mdct      │
      silk/common                    ▼        │          │
      silk/resampler            celt/cwrs     ▼          ▼
             │                       │    celt/lpc ──► celt/math_ops
             ▼                       │        │              │
        silk/tables                  └────┬───┘              │
             │                            ▼                  │
             │                     celt/modes ◄──────────────┘
             │                     celt/tables
             │                            │
             └──────────┬─────────────────┘
                        ▼
                  ┌───────────┐
                  │  entropy/  │
                  │range_coder│
                  │  laplace  │
                  └─────┬─────┘
                        ▼
                  ┌───────────┐
                  │  types.rs  │
                  │  error.rs  │
                  └────────────┘
```

Arrows point from dependant → dependency. The graph is acyclic. Every module ultimately depends on `types` and `entropy/range_coder`.

### 1.3 Feature Flags

```toml
[features]
default = ["fixed-point"]
fixed-point = []          # Fixed-point arithmetic (required for bit-exact)
float-api = []            # Float input/output conversion wrappers
custom-modes = []         # Dynamic CELTMode allocation
enable-qext = []          # Opus 1.5+ extended bandwidth (96 kHz)
enable-dred = []          # Deep redundancy FEC
enable-deep-plc = []      # Neural PLC (LPCNet/FARGAN)
enable-osce = []          # Neural post-filter (LACE/NoLACE)
```

---

## 2. Type System Design

### 2.1 Primitive Type Aliases

The C reference uses typedef aliases for portability. In Rust we use explicit-width types directly, but provide aliases for readability and grep-ability:

```rust
// src/types.rs

/// Signed 16-bit integer (maps to opus_int16 / i16).
pub type OpusInt16 = i16;
/// Signed 32-bit integer (maps to opus_int32 / i32).
pub type OpusInt32 = i32;
/// Unsigned 32-bit integer (maps to opus_uint32 / u32).
pub type OpusUint32 = u32;
/// Signed 64-bit integer (maps to opus_int64 / i64).
pub type OpusInt64 = i64;
```

### 2.2 Q-Format Fixed-Point Newtypes

The C code uses `opus_val16` and `opus_val32` as conditionally-compiled types (int16/float and int32/float). We use newtypes for type safety in the fixed-point path:

```rust
/// Q15 fixed-point value stored in i16. Range [-1.0, ~1.0).
/// Used for: LPC coefficients, normalized coefficients, twiddle factors,
/// filter taps, gains.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Val16(pub i16);

/// Q-format fixed-point value stored in i32. Q-format varies by context:
/// - Q27 for celt_sig (MDCT coefficients)
/// - Q24 for celt_norm (normalized spectrum)  
/// - Q24 for celt_glog (log-domain energy, DB_SHIFT=24)
/// - Q15 for general accumulation
/// Used for: signals, accumulators, energies, intermediate products.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Val32(pub i32);
```

**Design decision**: We do NOT encode the Q-format in the type (e.g., no `Q15<i16>` generic). The C reference freely reinterprets the same `opus_val32` at different Q-formats depending on context (Q27 signal, Q24 energy, Q15 gain). Encoding Q-format in the type would require pervasive conversions at every interface boundary, adding noise without catching real bugs — the Q-format is always determined by the calling convention, not the storage type. Instead, we document Q-format in function signatures and rely on the FFI comparison harness to catch arithmetic errors.

### 2.3 Signal Type Aliases

```rust
/// CELT signal (MDCT domain). Q27 fixed-point (SIG_SHIFT=12 above Q15).
pub type CeltSig = Val32;

/// CELT normalized coefficient. Q24 fixed-point (NORM_SHIFT=24).
pub type CeltNorm = Val32;

/// CELT energy. Q0 integer (sqrt of band energy).
pub type CeltEner = Val32;

/// CELT log-energy. Q24 fixed-point (DB_SHIFT=24).
pub type CeltGlog = Val32;
```

### 2.4 Fixed-Point Arithmetic Functions

All C macros become `#[inline(always)]` functions. These are the bit-exact equivalents:

```rust
/// 16×16 → 32 multiply (no shift).
#[inline(always)]
pub fn mult16_16(a: i32, b: i32) -> i32 {
    (a as i16 as i32) * (b as i16 as i32)
}

/// 16×16 → 32 multiply, right-shift by 15 (truncate).
#[inline(always)]
pub fn mult16_16_q15(a: i32, b: i32) -> i32 {
    ((a as i16 as i32) * (b as i16 as i32)) >> 15
}

/// 16×32 → 32 multiply, right-shift by 15.
#[inline(always)]
pub fn mult16_32_q15(a: i16, b: i32) -> i32 {
    // Matches C: (opus_val32)((opus_int16)(a)) * (opus_val32)(b) >> 15
    ((a as i64) * (b as i64) >> 15) as i32
}

/// Round-to-nearest right shift: (a + (1 << (shift-1))) >> shift.
#[inline(always)]
pub fn pshr32(a: i32, shift: u32) -> i32 {
    debug_assert!(shift > 0 && shift < 32);
    (a + (1 << (shift - 1))) >> shift
}

/// Arithmetic left shift via unsigned cast (avoids signed overflow UB).
#[inline(always)]
pub fn shl32(a: i32, shift: u32) -> i32 {
    (a as u32).wrapping_shl(shift) as i32
}

/// Wrapping unsigned add (with debug overflow assertion).
#[inline(always)]
pub fn uadd32(a: u32, b: u32) -> u32 {
    debug_assert!(a.checked_add(b).is_some(), "UADD32 overflow");
    a.wrapping_add(b)
}
```

The full set (~40 functions) mirrors `fixed_generic.h`, `arch.h`, and the SILK `SigProc_FIX.h` macros. Each function includes a `debug_assert!` that replicates the C debug-mode overflow checks from `fixed_debug.h`.

### 2.5 SILK Fixed-Point Macros

SILK uses its own macro set with different naming conventions but identical semantics:

```rust
/// silk_SMULBB: bottom×bottom 16-bit multiply.
#[inline(always)]
pub fn silk_smulbb(a: i32, b: i32) -> i32 {
    (a as i16 as i32) * (b as i16 as i32)
}

/// silk_SMULWB: 32×bottom16 → 32, right-shift 16.
#[inline(always)]
pub fn silk_smulwb(a: i32, b: i32) -> i32 {
    ((a as i64 * (b as i16 as i64)) >> 16) as i32
}

/// silk_RSHIFT_ROUND: round-to-nearest right shift.
#[inline(always)]
pub fn silk_rshift_round(a: i32, shift: i32) -> i32 {
    if shift == 1 { (a >> 1) + (a & 1) }
    else { ((a >> (shift - 1)) + 1) >> 1 }
}
```

---

## 3. Error Handling

### 3.1 Error Type

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusError {
    BadArg,          // OPUS_BAD_ARG (-1)
    BufferTooSmall,  // OPUS_BUFFER_TOO_SMALL (-2)
    InternalError,   // OPUS_INTERNAL_ERROR (-3)
    InvalidPacket,   // OPUS_INVALID_PACKET (-4)
    Unimplemented,   // OPUS_UNIMPLEMENTED (-5)
    InvalidState,    // OPUS_INVALID_STATE (-6)
    AllocFail,       // OPUS_ALLOC_FAIL (-7)
}

pub type OpusResult<T> = Result<T, OpusError>;
```

### 3.2 Mapping Strategy

The C reference returns negative integers for errors and positive values for success (e.g., bytes written, samples decoded). In Rust:

- **Encoder**: `fn encode(&mut self, ...) -> OpusResult<usize>` (bytes written)
- **Decoder**: `fn decode(&mut self, ...) -> OpusResult<usize>` (samples per channel)
- **CTL**: `fn ctl(&mut self, request: CtlRequest) -> OpusResult<CtlResponse>`

Internal modules propagate errors via `OpusResult<T>`. The range coder's sticky `error` flag becomes an early-return check after each entropy operation in critical sections:

```rust
if self.ec.error() {
    return Err(OpusError::InternalError);
}
```

### 3.3 Assertion Strategy

C uses `celt_assert` (compiled out in release) and `celt_sig_assert` (weaker, sometimes compiled out even in debug). We map these to:

- `celt_assert` → `debug_assert!`
- `celt_sig_assert` → `debug_assert!` (no distinction needed in Rust)
- Precondition violations that would cause UB in C → `assert!` (always checked) at public API boundaries only

---

## 4. Memory Management

### 4.1 Guiding Principle

The C reference uses three allocation patterns:
1. **Single contiguous allocation** for encoder/decoder state (flexible array member trick)
2. **Stack allocation** via `VARDECL`/`ALLOC` macros (VLA or alloca)
3. **Static const tables** (ROM data)

In Rust, we replace these cleanly:

### 4.2 State Allocation

The C pattern of `sizeof(OpusEncoder) + trailing_bytes` with byte-offset pointers becomes **owned struct fields**:

```rust
pub struct OpusEncoder {
    // Configuration
    mode: OpusMode,
    channels: usize,
    fs: u32,
    // ...
    
    // Sub-encoders (owned, not byte-offset pointers)
    silk: SilkEncoder,
    celt: CeltEncoder,
    
    // Variable-length state (was flexible array member)
    delay_buffer: Vec<i16>,
}
```

This eliminates all `unsafe` pointer arithmetic for state access. The trade-off is slightly more allocations at init time (one Vec per variable-length buffer instead of one contiguous block), but init is not performance-critical.

### 4.3 Stack Buffers

The `VARDECL`/`ALLOC` pattern allocates variable-length arrays on the stack. In Rust:

| C Pattern | Rust Replacement | When |
|-----------|-----------------|------|
| `ALLOC(x, N, type)` where N ≤ 256 | `let x = [0i32; 256]` | Size bounded by spec |
| `ALLOC(x, N, type)` where N is large | `let mut x = vec![0i32; n]` | Size depends on frame/mode |
| Hot inner loops with small temp | Stack array | Always |

Typical maximum stack allocations per module:

| Module | Largest Stack Buffer | Size |
|--------|---------------------|------|
| CELT decoder | `freq[2 * 960]` | 7.5 KB |
| CELT encoder | `in[2 * 1080]` | 8.6 KB |
| SILK NSQ | `sLTP_Q15[640]` | 2.5 KB |
| Bands | `_norm[2 * 5184]` | 40 KB |
| FFT/MDCT | `f2[480]` complex | 3.8 KB |

For the `_norm` buffer in `quant_all_bands` (~40 KB worst case), we use `Vec<i32>` to avoid stack overflow on constrained targets.

### 4.4 Static Tables

All precomputed tables (PVQ_U_DATA, eBands, allocVectors, iCDF tables, NLSF codebooks, twiddle factors, etc.) become `static` or `const` arrays:

```rust
pub(crate) static CELT_PVQ_U_DATA: [u32; 1272] = [ /* ... */ ];
pub(crate) static EBAND5MS: [i16; 22] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100];
```

For `CELTMode`, standard modes are `const` globals. No dynamic mode allocation unless `custom-modes` feature is enabled.

### 4.5 Arena / Scratch Buffer Pattern

For modules that make many small allocations per frame (e.g., bands.rs during recursive `quant_partition`), we consider a reusable scratch buffer passed through the call chain:

```rust
pub struct FrameScratch {
    norm: Vec<CeltNorm>,
    lowband_scratch: Vec<CeltNorm>,
    // ... other per-frame temporaries
}
```

Allocated once per encoder/decoder, reused every frame. This avoids repeated `Vec` allocation in the hot path.

---

## 5. State Machines

### 5.1 Encoder/Decoder State

Both encoder and decoder are modeled as structs with `impl` blocks. No builder pattern — the C API uses a single `init()` call with all parameters, and there's no benefit to a multi-step builder for codec state.

```rust
pub struct OpusEncoder {
    // Immutable after init
    fs: u32,
    channels: usize,
    application: Application,
    
    // Mutable codec state
    mode: OpusMode,
    bandwidth: Bandwidth,
    prev_mode: OpusMode,
    // ...
    
    // Sub-codecs
    silk: SilkEncoder,
    celt: CeltEncoder,
}

impl OpusEncoder {
    pub fn new(fs: u32, channels: usize, application: Application) -> OpusResult<Self> { ... }
    pub fn encode(&mut self, pcm: &[i16], output: &mut [u8]) -> OpusResult<usize> { ... }
    pub fn reset(&mut self) { ... }
}
```

### 5.2 Mode Switching

The Opus encoder's mode selection (SILK-only / Hybrid / CELT-only) is a state machine driven by bitrate, signal analysis, and frame size:

```rust
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OpusMode {
    SilkOnly = 1000,
    Hybrid = 1001,
    CeltOnly = 1002,
}
```

Mode transitions require redundancy frames (5 ms CELT crossfade). The transition logic lives in `opus/encoder.rs` and is a direct port of the C `opus_encode_frame_native` decision tree. We do not abstract this further — the decision logic is inherently sequential and depends on ~20 state variables.

### 5.3 SILK Decoder State

SILK maintains the largest persistent state (~3 KB/channel) with complex inter-frame dependencies. We model it as a flat struct matching the C layout:

```rust
pub struct SilkDecoderChannel {
    // Synthesis state
    exc_q14: [i32; MAX_FRAME_LENGTH],
    slpc_q14_buf: [i32; MAX_LPC_ORDER],
    out_buf: [i16; MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH],
    
    // Prediction state
    prev_nlsf_q15: [i16; MAX_LPC_ORDER],
    lag_prev: i32,
    prev_gain_q16: i32,
    
    // PLC state
    plc: SilkPlcState,
    cng: SilkCngState,
    
    // Frame configuration
    fs_khz: i32,
    nb_subfr: i32,
    frame_length: i32,
    // ...
}
```

The "reset from marker" pattern (`SILK_DECODER_STATE_RESET_START`) becomes a `reset()` method that zeroes fields after the OSCE section boundary.

### 5.4 CTL Interface

The C variadic `ctl()` becomes a typed enum:

```rust
pub enum EncoderCtl {
    SetBitrate(i32),
    GetBitrate,
    SetComplexity(i32),
    SetVbr(bool),
    SetBandwidth(Bandwidth),
    ResetState,
    // ... ~30 variants
}

pub enum CtlResponse {
    Ok,
    Value(i32),
}
```

---

## 6. FFI Boundary

### 6.1 Test Harness Architecture

The comparison test harness lives in `tests/compare/` and links the C reference via `cc` crate:

```
tests/compare/
├── build.rs          # Compiles reference/ C source with cc crate
├── ffi.rs            # extern "C" declarations for C reference functions
├── helpers.rs        # Buffer allocation, comparison utilities
├── test_range_coder.rs
├── test_cwrs.rs
├── test_math_ops.rs
├── test_fft_mdct.rs
├── test_pitch.rs
├── test_bands.rs
├── test_celt_codec.rs
├── test_silk_codec.rs
├── test_opus_codec.rs
└── test_roundtrip.rs
```

### 6.2 FFI Binding Strategy

Minimal hand-written `extern "C"` blocks — no bindgen. Each tested C function gets an explicit declaration:

```rust
// tests/compare/ffi.rs
extern "C" {
    pub fn ec_enc_init(enc: *mut CEncState, buf: *mut u8, size: u32);
    pub fn ec_encode(enc: *mut CEncState, fl: u32, fh: u32, ft: u32);
    pub fn ec_enc_done(enc: *mut CEncState);
    pub fn ec_range_bytes(enc: *const CEncState) -> u32;
    pub fn ec_get_buffer(enc: *const CEncState) -> *const u8;
    // ...
}
```

The C structs needed for FFI are defined as `#[repr(C)]` Rust structs with fields matching the C layout exactly. These are test-only — the production Rust code uses its own idiomatic structs.

### 6.3 Comparison Protocol

For each module, tests follow a pattern:

```rust
#[test]
fn test_range_coder_bit_exact() {
    // 1. Prepare identical inputs
    let symbols = generate_test_symbols();
    
    // 2. Encode with C reference
    let c_output = unsafe { c_encode(&symbols) };
    
    // 3. Encode with Rust implementation
    let mut enc = RangeEncoder::new(1024);
    for &(fl, fh, ft) in &symbols {
        enc.encode(fl, fh, ft);
    }
    enc.done();
    let rust_output = enc.get_buffer();
    
    // 4. Byte-exact comparison
    assert_eq!(c_output, rust_output, "Range coder output diverges");
}
```

### 6.4 Build Configuration

The `build.rs` for tests compiles the C reference with:
- `FIXED_POINT` defined (matching our primary target)
- No SIMD (`-DOPUS_ARM_ASM=0`, etc.)
- No custom modes
- Standard Opus configuration

```rust
// tests/compare/build.rs
fn main() {
    cc::Build::new()
        .define("FIXED_POINT", "1")
        .define("OPUS_BUILD", "1")
        .include("reference/include")
        .include("reference/celt")
        .include("reference/silk")
        .files(/* C source files */)
        .compile("opus_reference");
}
```

---

## 7. Testing Strategy

### 7.1 Test Pyramid

```
                    ┌──────────────────┐
                    │  Round-trip tests │  Encode WAV → decode → compare PCM
                    │  (integration)    │  Covers full pipeline bit-exactness
                    └────────┬─────────┘
                             │
                 ┌───────────┴───────────┐
                 │  Module FFI comparison │  Per-function C↔Rust comparison
                 │  (comparison harness)  │  ~500 tests covering all modules
                 └───────────┬───────────┘
                             │
            ┌────────────────┴────────────────┐
            │  Unit tests per module           │  Edge cases, invariants,
            │  (pure Rust, no FFI)             │  property-based tests
            └─────────────────────────────────┘
```

### 7.2 Unit Tests

Each module includes `#[cfg(test)] mod tests` with:

- **Edge cases**: Zero-length inputs, maximum K, boundary pitch periods, silence, clipping
- **Invariants**: Range coder round-trip (`encode → decode` recovers input), PVQ enumeration bijectivity (`encode_pulses → decode_pulses` recovers pulse vector), NLSF stability after quantization
- **Regression**: Known-good input/output pairs extracted from C reference runs

### 7.3 FFI Comparison Tests

Per-module comparison against C reference for bit-exactness. Key test vectors:

| Module | Test Method | Coverage |
|--------|-------------|----------|
| Range coder | Encode sequence of symbols, compare byte output | All symbol types (icdf, uint, bits, bit_logp) |
| CWRS | encode_pulses/decode_pulses for all (N,K) in table | Full codebook enumeration |
| Math ops | Input sweep for each function | celt_log2, celt_exp2, celt_sqrt, celt_rcp, celt_cos_norm, isqrt32 |
| FFT/MDCT | Forward+inverse, compare coefficients | All 4 FFT sizes (60, 120, 240, 480) |
| Pitch | Synthetic signals with known pitch | Downsample, search, remove_doubling |
| Quant bands | Encode/decode energy with known residuals | Coarse, fine, finalise stages |
| VQ | alg_quant with known X,K → compare iy[] | Multiple band sizes and K values |
| Bands | quant_all_bands encode/decode full frame | Multiple bitrates and band configs |
| CELT codec | Full encode/decode frame | 10/20 ms, mono/stereo, multiple bitrates |
| SILK codec | Full encode/decode frame | 10/20/40/60 ms, NB/WB, mono/stereo |
| Opus codec | Full encode/decode frame | All modes, bandwidths, frame sizes |

### 7.4 Property-Based Tests

Using a property-testing framework (e.g., `proptest`):

- **Range coder**: For any sequence of symbols, `ec_tell()` is consistent between encoder and decoder at every step
- **CWRS**: For any pulse vector y with `sum(|y|) = K`, `decode(encode(y)) == y`
- **PVQ**: `||alg_unquant(alg_quant(X))|| == gain` (norm preservation)
- **NLSF**: After quantize→dequantize, NLSFs maintain minimum spacing
- **LPC**: `inverse_pred_gain(stabilized_coefficients) > 0`

### 7.5 Round-Trip Integration Tests

End-to-end tests that exercise the complete pipeline:

```rust
#[test]
fn test_roundtrip_celt_48khz_stereo() {
    let pcm_in: Vec<i16> = load_test_wav("assets/test_48k_stereo.wav");
    
    // Encode with C reference
    let c_encoded = unsafe { c_opus_encode(&pcm_in, 64000, 960) };
    
    // Decode with Rust
    let mut dec = OpusDecoder::new(48000, 2).unwrap();
    let rust_decoded = dec.decode(&c_encoded, 960).unwrap();
    
    // Encode with Rust
    let mut enc = OpusEncoder::new(48000, 2, Application::Audio).unwrap();
    let rust_encoded = enc.encode(&pcm_in, 960).unwrap();
    
    // Byte-exact compressed output
    assert_eq!(c_encoded, rust_encoded);
    
    // Decode C-encoded with Rust decoder → PCM-exact
    let rust_decoded_from_c = dec.decode(&c_encoded, 960).unwrap();
    assert_eq!(pcm_in_trimmed, rust_decoded_from_c);
}
```

### 7.6 Coverage Target

Use `cargo llvm-cov` (per project rules) to measure line coverage. Target: >90% for all ported modules, with 100% coverage of numerical paths (all Q-format arithmetic functions, all entropy coding paths).

---

## 8. Implementation Phases

### Phase 1: Foundation (Modules 1–3)

**Scope**: Range coder, math utilities, combinatorial coding.

**Deliverables**:
- `entropy/range_coder.rs`: Full `RangeEncoder` and `RangeDecoder` with all symbol types
- `entropy/laplace.rs`: Laplace entropy coding
- `celt/math_ops.rs`: All fixed-point math (`isqrt32`, `celt_rcp`, `celt_sqrt`, `celt_log2`, `celt_exp2`, `celt_cos_norm`, trig functions)
- `celt/lpc.rs`: `_celt_lpc`, `celt_fir`, `celt_iir`, `_celt_autocorr`
- `celt/cwrs.rs`: PVQ codebook enumeration
- `types.rs`: All type definitions and Q-format arithmetic functions
- `error.rs`: Error types

**Gate criteria**:
- All FFI comparison tests pass for range coder, math ops, CWRS
- `ec_tell` / `ec_tell_frac` match C reference at every symbol boundary
- Range coder round-trip: decode(encode(symbols)) == symbols for all test vectors
- CWRS: decode_pulses(encode_pulses(y)) == y for all (N,K) in PVQ table

### Phase 2: CELT Core (Modules 4–9)

**Scope**: Transform, pitch, bands, quantization, VQ, rate allocation, modes.

**Deliverables**:
- `celt/fft.rs`, `celt/mdct.rs`: Forward/inverse FFT and MDCT
- `celt/pitch.rs`: Pitch analysis (downsample, search, remove_doubling, xcorr_kernel)
- `celt/bands.rs`: Band processing, normalization, denormalization, quant_all_bands
- `celt/quant_bands.rs`: Energy quantization (coarse, fine, finalise)
- `celt/vq.rs`: PVQ search, exp_rotation, renormalize
- `celt/rate.rs`: Bit allocation
- `celt/modes.rs`: Static CELTMode tables

**Gate criteria**:
- FFT/MDCT: Forward then inverse recovers input (within fixed-point rounding)
- FFT/MDCT coefficients match C reference for all 4 sizes
- quant_all_bands encode/decode matches C reference for multiple bitrates
- Full CELT encode → byte-exact output matching C reference for standard test vectors

### Phase 3: CELT Codec (Modules 10–11)

**Scope**: CELT encoder and decoder, including PLC.

**Deliverables**:
- `celt/decoder.rs`: Full CELT decoder with PLC (noise + pitch)
- `celt/encoder.rs`: Full CELT encoder with pre-filter, transient detection, VBR

**Gate criteria**:
- CELT-only Opus packets: encode with C, decode with Rust → PCM-exact
- CELT-only Opus packets: encode with Rust → byte-exact with C
- PLC: packet loss produces identical concealment output as C reference
- All frame sizes (2.5/5/10/20 ms), mono and stereo, bitrates 6–510 kbps

### Phase 4: SILK (Modules 12–14)

**Scope**: SILK common utilities, decoder, encoder.

**Deliverables**:
- `silk/common.rs`, `silk/tables.rs`: All SILK tables, sorting, bwexpander, lin2log
- `silk/nlsf.rs`: NLSF codec (two-stage VQ, A2NLSF, NLSF2A)
- `silk/resampler.rs`: Polyphase resampler
- `silk/decoder.rs`: Full SILK decoder (decode_indices, decode_pulses, decode_core)
- `silk/encoder.rs`: Full SILK encoder (pitch, LPC, NSQ, LBRR)
- Supporting modules: `plc.rs`, `cng.rs`, `vad.rs`, `noise_shape.rs`, `shell_coder.rs`, `stereo.rs`, `pitch.rs`, `lpc.rs`

**Gate criteria**:
- SILK-only Opus packets: decode with Rust → PCM-exact with C decoder
- SILK-only Opus packets: encode with Rust → byte-exact with C encoder
- SILK stereo: mid/side prediction matches C reference
- SILK PLC + CNG: loss concealment matches C reference
- All SILK configurations: NB/MB/WB, 10/20/40/60 ms, mono/stereo

### Phase 5: Opus Integration (Modules 15–18)

**Scope**: Opus decoder, encoder, multistream, repacketizer.

**Deliverables**:
- `opus/decoder.rs`: Mode switching, redundancy frames, FEC, gain
- `opus/encoder.rs`: Mode/bandwidth selection, speech/music classifier, bitrate allocation
- `opus/multistream.rs`: Channel mapping families 0/1/2/3/255
- `opus/repacketizer.rs`: Packet merge/split/pad, extension iterator
- `opus/packet.rs`: TOC parsing, packet inspection

**Gate criteria**:
- All-mode round-trip: encode with Rust → decode with Rust → PCM matches C round-trip
- Mode transitions: SILK↔CELT with redundancy frames → byte-exact
- Hybrid mode: SILK+CELT combined packets → byte-exact
- Multistream: surround encoding/decoding → byte-exact
- Repacketizer: pad/unpad/merge → byte-exact
- Packet inspection utilities match C reference for all packet types

### Phase 6: DNN (Modules 19–26) — Deferred

**Scope**: Neural network inference, LPCNet, OSCE, FARGAN, FWGAN, DRED, PitchDNN, Lossgen.

This phase is deferred because:
1. DNN modules are optional enhancements (not required for RFC 6716 compliance)
2. They operate exclusively in float (no fixed-point bit-exactness concern)
3. They require weight data files and inference infrastructure
4. They can be added incrementally behind feature flags

**Gate criteria** (when implemented):
- DNN core: Dense/GRU/Conv1D output matches C reference for test inputs
- OSCE: Enhanced SILK output matches C reference
- DRED: Encoded latents → decoded features match C reference
- LPCNet PLC: Concealment audio matches C reference

---

## 9. Risk Register

### 9.1 High Risk: Fixed-Point Rounding Mismatches

**Threat**: Off-by-one errors in Q-format arithmetic producing bit-inexact output.

**Likelihood**: Very high. The C reference has ~40 arithmetic macros with subtle rounding/truncation semantics. A single mismatch propagates through the entire pipeline.

**Root causes**:
- Rust `>>` on negative numbers: arithmetic right-shift (same as C on two's complement), but the C standard doesn't guarantee this. The xiph code relies on implementation-defined behavior. Rust is well-defined here — same behavior, no risk.
- `MULT16_16(a, b)` truncates both operands to `i16` before widening. Missing this truncation produces different results for values outside [-32768, 32767].
- `FRAC_MUL16(a, b) = (16384 + a*b) >> 15` — the rounding bias of `+16384` (not `+16383`) is critical.
- `PSHR32(a, 0)` — shift of 0 means the bias is `1 << -1` = 0, which is correct but easy to get wrong.
- SILK's `silk_SMLAWB` truncates toward -∞ (not toward zero). Rust's `>>` on negative values also truncates toward -∞. Same behavior, but must not use `/` (which truncates toward zero).

**Mitigation**:
1. Port every arithmetic macro as a standalone `#[inline(always)]` function with an exhaustive unit test against the C version (test all corner cases: 0, ±1, ±32767, ±32768, MAX_INT, MIN_INT).
2. Run the FFI comparison harness at every commit gate.
3. For each module, test with at least 1000 random inputs comparing Rust and C outputs.

### 9.2 High Risk: Wrapping Arithmetic in Unsigned Domain

**Threat**: Rust panics on integer overflow in debug mode; C's unsigned overflow wraps silently.

**Likelihood**: High. The range coder, LCG PRNG, FFT butterfly, and CWRS all rely on unsigned wrapping.

**Specific locations**:
- Range coder `val` accumulation: `val += rng - IMUL32(r, ft - fl)` can wrap
- LCG: `seed = 1664525 * seed + 1013904223` wraps by design
- FFT butterfly: `ADD32_ovflw`, `SUB32_ovflw` — wrapping signed addition via unsigned cast
- CWRS: `UADD32`, `USUB32` — plain unsigned wrapping

**Mitigation**: Use `wrapping_add`, `wrapping_sub`, `wrapping_mul` explicitly at every site. Never rely on Rust's default checked arithmetic. Grep for all uses of `_ovflw` macros in C and ensure corresponding Rust uses `wrapping_*`.

### 9.3 Medium Risk: Stack Allocation Size

**Threat**: `VARDECL`/`ALLOC` stack allocations exceed Rust's default 8 MB stack, or cause performance issues with `Vec` heap allocation.

**Likelihood**: Medium. The worst case is `quant_all_bands` with ~40 KB of stack buffers.

**Mitigation**: Use `Vec<T>` for buffers > 4 KB. Pre-allocate in the encoder/decoder state and reuse per frame via `FrameScratch`. Profile to verify no performance regression from heap allocation in hot paths.

### 9.4 Medium Risk: SILK NLSF Codec Complexity

**Threat**: The NLSF codec (A2NLSF, NLSF2A, two-stage VQ, trellis, stabilization) is the most algorithmically complex module in SILK. Bit-exactness requires matching every intermediate step.

**Likelihood**: Medium. The algorithm involves polynomial root-finding with iterative refinement, bandwidth expansion with fallback logic, and trellis search with rate-distortion optimization.

**Mitigation**: Port module-by-module with FFI tests at each stage. Test NLSF quantize → dequantize → A coefficients against C reference for every NLSF codebook entry.

### 9.5 Medium Risk: FFT Twiddle Factor Generation

**Threat**: Twiddle factors must match the C reference exactly. Any rounding difference in `sin`/`cos` computation produces different FFT output.

**Likelihood**: Medium for custom modes (dynamic twiddle generation). Zero for standard modes (static tables).

**Mitigation**: For standard modes, copy the precomputed tables verbatim from the C source. For custom modes (behind feature flag), generate twiddles using the same formula: `cos(2*pi*i/nfft)` with explicit rounding to Q15: `(32768.0 * val + 0.5).floor() as i16`.

### 9.6 Low Risk: Endianness

**Threat**: The range coder reads/writes bytes in big-endian order. Platform endianness could affect multi-byte operations.

**Likelihood**: Low. All C code operates byte-by-byte. No multi-byte loads/stores.

**Mitigation**: Port byte-by-byte (matching C). No platform-specific byte-swap needed.

### 9.7 Low Risk: Signed Integer Representation

**Threat**: C code assumes two's complement (e.g., `(val + s) ^ s` for conditional negate where `s = -1`).

**Likelihood**: Zero in Rust. Rust guarantees two's complement for signed integers. The C code's assumption is also safe on all modern platforms.

---

## 10. Performance Considerations

### 10.1 SIMD in C Reference

The C reference has platform-specific SIMD implementations for these critical paths:

| Function | Platforms | Speedup | Scalar Replacement |
|----------|-----------|---------|-------------------|
| `xcorr_kernel` | SSE, SSE2, SSE4.1, NEON, MIPS | 3–8x | Process 4 correlations with 4x unrolled loop |
| `celt_pitch_xcorr` | SSE, NEON | 3–5x | Calls xcorr_kernel in groups of 4 |
| `dual_inner_prod` | SSE, NEON | 2–3x | Two dot products in single pass |
| `celt_inner_prod` | SSE, NEON | 2–3x | Standard dot product |
| `opus_fft` / `opus_ifft` | NEON, SSE | 2–4x | Mixed-radix butterfly stages |
| `clt_mdct_forward/backward` | NEON | 2–3x | Pre/post rotation + windowing |
| `exp_rotation` | NEON | 2x | Butterfly pair rotation |
| `comb_filter_const` | NEON | 2x | Pitch filter application |
| `silk_NSQ` | SSE4.1, NEON | 2–3x | Noise shaping quantizer inner loop |
| `silk_noise_shape_quantizer_short_prediction` | SSE4.1, NEON | 2–3x | LPC prediction accumulation |
| `celt_fir` / `celt_iir` | via xcorr_kernel | 2–3x | Filter inner loop |
| DNN layers | SSE2, AVX2, NEON, DOTPROD | 4–16x | Matrix-vector multiply, GRU |

### 10.2 Initial Port Strategy

**Phase 1 (correctness)**: Scalar Rust only. No SIMD. The `arch` parameter is accepted but ignored. All SIMD-dispatched functions use the `_c` (portable C) reference implementation as the porting target.

**Phase 2 (performance)**: After bit-exact correctness is established:
1. Profile to identify actual bottlenecks (don't guess)
2. Use `std::arch` intrinsics for SSE2/AVX2/NEON
3. Use `#[target_feature(enable = "...")]` for runtime dispatch
4. Consider `packed_simd2` or `std::simd` (nightly) for portable SIMD

### 10.3 Performance-Critical Paths (Ranked)

Based on C profiling data, these consume the most CPU time:

1. **PVQ search** (`op_pvq_search_c` in `vq.rs`): O(K×N) greedy loop. ~30% of encode time. The inner loop is 4 multiplies + 2 compares per dimension per pulse. Scalar Rust should be ~1.2x slower than scalar C (bounds checking overhead); SIMD gives 3–8x.

2. **xcorr_kernel** (in `pitch.rs`, also used by `celt_fir`/`celt_iir`): 4-way unrolled correlation. ~20% of encode time. Critical for pitch analysis and LPC filtering.

3. **FFT butterfly stages** (in `fft.rs`): Mixed-radix (2,3,4,5) butterflies. ~15% of codec time. The radix-4 degenerate case (all-ones twiddles) is the hottest path.

4. **NSQ inner loop** (in `silk/noise_shape.rs`): Per-sample LPC + LTP + shaping prediction. ~15% of SILK encode time.

5. **MDCT pre/post rotation** (in `mdct.rs`): Twiddle multiplies in tight loops. ~10% of codec time.

### 10.4 Rust-Specific Optimizations

- **Bounds check elimination**: Use `get_unchecked` in proven-safe inner loops (behind `unsafe` with documented invariants) only if profiling shows bounds checks as a bottleneck. Start with safe indexing.
- **Autovectorization**: Structure inner loops to be autovectorization-friendly (contiguous access, no loop-carried dependencies). The PVQ search loop has a loop-carried dependency (best_num/best_den) that defeats autovectorization — same as C.
- **Branch prediction hints**: Use `#[cold]` on error paths. Use `likely`/`unlikely` from `std::intrinsics` (nightly) or restructure branches for the common case.
- **Allocation avoidance**: Pre-allocate all per-frame scratch buffers in encoder/decoder state. Zero allocation per `encode()`/`decode()` call in the steady state.

### 10.5 Performance Budget

Target: within 1.5x of the C reference for scalar code. This is achievable because:
- Rust's codegen (LLVM) produces comparable machine code to GCC/Clang for arithmetic loops
- The main overhead is bounds checking, which can be eliminated in proven-safe inner loops
- Memory layout is identical (flat arrays, no indirection)
- No GC, no runtime overhead

The DNN modules (Phase 6) will be the most performance-sensitive due to matrix-vector multiply. These benefit enormously from SIMD (4–16x speedup). For the initial port, DNN inference will be ~4x slower than optimized C. This is acceptable because DNN features are optional and only used for enhancement (OSCE, PLC).

---

*Document version: 0.1.0*
*Generated: 2026-03-27*
*Target: bit-exact output matching xiph/opus C reference (fixed-point mode)*
