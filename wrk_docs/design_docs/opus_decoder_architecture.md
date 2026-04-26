Now I have a thorough understanding of all the source files. Here is the complete architecture document:

---

# Opus Decoder Module — Architecture Documentation

## 1. Purpose

The `opus_decoder` module is the top-level decoding entry point for the Opus codec. It sits above the two sub-codecs (SILK for speech, CELT for music/audio) and orchestrates their use based on information encoded in each packet's Table of Contents (TOC) byte. Its responsibilities:

- **Packet parsing**: Extract mode, bandwidth, frame count, frame sizes, and extension data from the Opus bitstream.
- **Sub-codec dispatch**: Route decoded frames to SILK, CELT, or both (hybrid mode).
- **Mode transition handling**: Crossfade between SILK and CELT when the encoder switches modes mid-stream.
- **Redundancy processing**: Decode 5 ms CELT redundancy frames that protect SILK↔CELT transitions.
- **Packet Loss Concealment (PLC)**: Generate plausible audio when packets are lost (data=NULL).
- **Forward Error Correction (FEC)**: Decode LBRR (Low Bit-Rate Redundancy) data from the next packet to recover a lost frame.
- **Gain application**: Apply configurable decode gain.
- **Soft clipping** (float API only): Prevent clipping artifacts via smooth non-linearity.
- **Output format conversion**: Convert internal representation to int16, int32 (24-bit), or float output.
- **CTL interface**: Get/set decoder parameters (bandwidth, gain, complexity, etc.).

## 2. Public API

### 2.1 Lifecycle

```c
// Returns byte size needed for decoder state (includes SILK + CELT sub-decoders)
int opus_decoder_get_size(int channels);

// Allocate + init. Returns NULL on failure, sets *error.
OpusDecoder *opus_decoder_create(opus_int32 Fs, int channels, int *error);

// Init pre-allocated memory. Returns OPUS_OK or error code.
int opus_decoder_init(OpusDecoder *st, opus_int32 Fs, int channels);

// Free a decoder allocated by opus_decoder_create().
void opus_decoder_destroy(OpusDecoder *st);
```

**Parameters**:
- `Fs`: Output sampling rate. Must be one of: 8000, 12000, 16000, 24000, 48000 (96000 with `ENABLE_QEXT`).
- `channels`: 1 (mono) or 2 (stereo).

### 2.2 Decoding Functions

```c
// Decode to 16-bit PCM
int opus_decode(OpusDecoder *st, const unsigned char *data,
    opus_int32 len, opus_int16 *pcm, int frame_size, int decode_fec);

// Decode to 24-bit PCM (stored in opus_int32)
int opus_decode24(OpusDecoder *st, const unsigned char *data,
    opus_int32 len, opus_int32 *pcm, int frame_size, int decode_fec);

// Decode to floating-point PCM
int opus_decode_float(OpusDecoder *st, const unsigned char *data,
    opus_int32 len, float *pcm, int frame_size, int decode_fec);
```

**Parameters**:
- `data`: Compressed Opus packet. `NULL` triggers PLC.
- `len`: Byte length of `data`. 0 or negative with `data==NULL` triggers PLC.
- `pcm`: Output buffer. Interleaved if stereo. Length must be ≥ `frame_size * channels`.
- `frame_size`: Maximum samples per channel the output buffer can hold. For PLC/FEC, must be the exact duration of missing audio and a multiple of 2.5 ms.
- `decode_fec`: 0 = normal decode, 1 = decode FEC from this packet to recover *previous* frame.

**Return value**: Number of decoded samples per channel (positive), or a negative error code (`OPUS_BAD_ARG`, `OPUS_BUFFER_TOO_SMALL`, `OPUS_INVALID_PACKET`, `OPUS_INTERNAL_ERROR`).

### 2.3 Internal Decode Entry Point

```c
// Not exported — called by the public wrappers and by multistream decoder
int opus_decode_native(OpusDecoder *st, const unsigned char *data,
    opus_int32 len, opus_res *pcm, int frame_size, int decode_fec,
    int self_delimited, opus_int32 *packet_offset, int soft_clip,
    const OpusDRED *dred, opus_int32 dred_offset);
```

Extra parameters vs public API:
- `self_delimited`: 1 when decoding from a multistream packet (each sub-frame carries its own length).
- `packet_offset`: Receives total bytes consumed (including padding).
- `soft_clip`: 1 to apply soft clipping (float builds only).
- `dred` / `dred_offset`: Deep Redundancy state for neural PLC (Opus 1.5+).

### 2.4 CTL Interface

```c
int opus_decoder_ctl(OpusDecoder *st, int request, ...);
```

Supported requests:

| Request | Type | Description |
|---------|------|-------------|
| `OPUS_GET_BANDWIDTH` | get `opus_int32*` | Last decoded packet bandwidth |
| `OPUS_SET_COMPLEXITY` / `GET` | set/get `opus_int32` | 0–10, controls DNN PLC/OSCE |
| `OPUS_GET_FINAL_RANGE` | get `opus_uint32*` | Range coder final state (for testing) |
| `OPUS_RESET_STATE` | — | Reset decoder to initial state |
| `OPUS_GET_SAMPLE_RATE` | get `opus_int32*` | Output sample rate |
| `OPUS_GET_PITCH` | get `opus_int32*` | Last pitch lag (samples at 48 kHz) |
| `OPUS_SET_GAIN` / `GET` | set/get `opus_int32` | Decode gain in Q8 dB (±32767) |
| `OPUS_GET_LAST_PACKET_DURATION` | get `opus_int32*` | Duration of last decoded packet (samples) |
| `OPUS_SET_PHASE_INVERSION_DISABLED` / `GET` | set/get `opus_int32` | 0/1, forwarded to CELT |
| `OPUS_SET_IGNORE_EXTENSIONS` / `GET` | set/get `opus_int32` | 0/1, ignore padding extensions |
| `OPUS_SET_DNN_BLOB` | set `const unsigned char*, opus_int32` | Load DNN weights (deep PLC + OSCE) |

### 2.5 Packet Inspection (Stateless)

```c
int opus_packet_get_bandwidth(const unsigned char *data);       // OPUS_BANDWIDTH_*
int opus_packet_get_nb_channels(const unsigned char *data);     // 1 or 2
int opus_packet_get_nb_frames(const unsigned char packet[], opus_int32 len);
int opus_packet_get_nb_samples(const unsigned char packet[], opus_int32 len, opus_int32 Fs);
int opus_packet_get_samples_per_frame(const unsigned char *data, opus_int32 Fs);
int opus_packet_has_lbrr(const unsigned char packet[], opus_int32 len);
int opus_decoder_get_nb_samples(const OpusDecoder *dec, const unsigned char packet[], opus_int32 len);
```

## 3. Internal State

### 3.1 `struct OpusDecoder`

The struct is defined privately in `opus_decoder.c` (not in any public header). The SILK and CELT sub-decoder states are allocated *inline* at the end of the same allocation — accessed via byte offsets.

```c
struct OpusDecoder {
    int          celt_dec_offset;    // Byte offset from struct start to CELTDecoder
    int          silk_dec_offset;    // Byte offset from struct start to SILK decoder
    int          channels;           // 1 or 2 (immutable after init)
    opus_int32   Fs;                 // Output sample rate (immutable after init)
    silk_DecControlStruct DecControl; // SILK decoder control/status block
    int          decode_gain;        // Q8 dB gain applied to output
    int          complexity;         // 0–10 (controls DNN features)
    int          ignore_extensions;  // If 1, ignore padding-carried extensions
    int          arch;               // CPU architecture flags (SIMD dispatch)

    // === Everything below cleared on OPUS_RESET_STATE ===
    int          stream_channels;    // Channels in current stream (from TOC)
    int          bandwidth;          // Current bandwidth (OPUS_BANDWIDTH_*)
    int          mode;               // Current mode (MODE_SILK_ONLY/HYBRID/CELT_ONLY)
    int          prev_mode;          // Previous frame's mode (for transitions)
    int          frame_size;         // Current frame size in samples at output Fs
    int          prev_redundancy;    // Whether previous frame had SILK→CELT redundancy
    int          last_packet_duration; // Duration of last decoded packet (samples)
    opus_val16   softclip_mem[2];    // Per-channel soft clipping state (float only)
    opus_uint32  rangeFinal;         // Range coder final state
};
```

**Memory layout** (contiguous allocation):

```
[OpusDecoder struct | padding | SILK decoder state | CELT decoder state]
 ^                            ^                      ^
 st                           st + silk_dec_offset    st + celt_dec_offset
```

All offsets are computed by `align()` which pads to the natural alignment of `{void*, opus_int32, opus_val32}`.

### 3.2 `silk_DecControlStruct`

Embedded in `OpusDecoder`, carries per-frame configuration to/from the SILK decoder:

```c
typedef struct {
    opus_int32 nChannelsAPI;         // Output channels (1 or 2)
    opus_int32 nChannelsInternal;    // Internal channels (1 or 2)
    opus_int32 API_sampleRate;       // Output sample rate
    opus_int32 internalSampleRate;   // SILK's internal rate (8000/12000/16000)
    opus_int   payloadSize_ms;       // Frame duration (10/20/40/60 ms)
    opus_int   prevPitchLag;         // Output: previous pitch lag (samples at 48 kHz)
    opus_int   enable_deep_plc;      // Enable neural PLC (complexity ≥ 5)
    // OSCE fields (conditional compilation)
} silk_DecControlStruct;
```

### 3.3 State Lifecycle

1. **Create**: `opus_decoder_create()` → `opus_alloc()` → `opus_decoder_init()`
2. **Init**: Zero entire allocation, compute offsets, init SILK decoder (`silk_InitDecoder`), init CELT decoder (`celt_decoder_init`), set CELT signalling off, set `frame_size = Fs/400` (2.5 ms), detect CPU arch
3. **Reset** (`OPUS_RESET_STATE`): Clear from `stream_channels` onward, reset SILK and CELT, restore `stream_channels = channels`, `frame_size = Fs/400`
4. **Destroy**: `opus_free(st)`

## 4. Algorithm

### 4.1 TOC Byte Parsing

The first byte of every Opus packet is the Table of Contents (TOC):

```
Bits 7-5: Configuration (mode + bandwidth)
Bit    4: (mode-dependent)
Bit    3: (mode-dependent)
Bit    2: Stereo flag (0=mono, 1=stereo)
Bits 1-0: Frame count code
```

**Mode detection** (`opus_packet_get_mode`):
- Bit 7 = 1 → CELT-only
- Bits 7-6 = 01 → Hybrid (SILK+CELT)
- Bits 7-6 = 00 → SILK-only

**Bandwidth** (`opus_packet_get_bandwidth`):
- CELT-only: bits 6-5 → mediumband(=NB), wideband, superwideband, fullband
- Hybrid: bit 4 → superwideband or fullband
- SILK-only: bits 6-5 → NB, MB, WB

**Frame count code** (bits 1-0):
- 0: 1 frame
- 1: 2 frames, CBR (equal size)
- 2: 2 frames, VBR (sizes encoded)
- 3: arbitrary count (0–48 frames), with sub-header byte

**Samples per frame** (`opus_packet_get_samples_per_frame`):
- CELT: bits 4-3 → {Fs/400, Fs/200, Fs/100, Fs/50} → 2.5, 5, 10, 20 ms
- Hybrid: bit 3 → {Fs/100, Fs/50} → 10 or 20 ms
- SILK: bits 4-3 → {Fs/100, Fs/50, Fs/25, Fs*60/1000} → 10, 20, 40, 60 ms

### 4.2 Packet Parsing (`opus_packet_parse_impl`)

Extracts individual frame pointers and sizes from a multi-frame packet:

1. Parse TOC byte → `toc`, advance pointer
2. Based on frame count code:
   - **Code 0**: 1 frame, `size[0] = remaining bytes`
   - **Code 1**: 2 CBR frames, `size[0] = size[1] = len/2` (must be even)
   - **Code 2**: 2 VBR frames, parse `size[0]` with `parse_size()`, `size[1] = remaining`
   - **Code 3**: Read sub-header byte: count (bits 0-5), padding flag (bit 6), VBR flag (bit 7). Parse optional padding bytes (255 = 254 + continue). Parse per-frame sizes for VBR, or compute for CBR.
3. For self-delimited framing: parse additional length for last frame
4. Validate no frame exceeds 1275 bytes
5. Set up frame pointer array

**Size encoding** (`parse_size` / `encode_size`):
- Values 0–251: 1 byte (literal)
- Values 252–65535: 2 bytes, `value = 4*data[1] + data[0]` where `data[0] = 252 + (size & 3)`

### 4.3 Core Decode Flow (`opus_decode_native`)

```
opus_decode() / opus_decode_float() / opus_decode24()
    └── opus_decode_native()
            ├── [PLC path: data==NULL] → loop opus_decode_frame(NULL) until frame_size filled
            ├── [FEC path: decode_fec==1] → PLC for leading frames, then opus_decode_frame(data, fec=1)
            └── [Normal path] →
                    1. Parse packet → count frames, sizes, extensions
                    2. Update state (mode, bandwidth, frame_size, stream_channels)
                    3. Loop: opus_decode_frame() for each frame in packet
                    4. Apply soft clipping (float only)
                    5. Return total samples
```

### 4.4 Single Frame Decode (`opus_decode_frame`)

This is the core algorithm. For each frame:

```
1. Compute duration constants: F20, F10, F5, F2_5 = Fs/{50,100,200,400}
2. If data==NULL → PLC mode:
   a. Use prev_mode (or MODE_CELT_ONLY if prev_redundancy)
   b. If no previous mode → output zeros
   c. If audiosize > F20 → recursively decode in 20ms chunks
   d. Snap audiosize to valid PLC sizes (2.5, 5, 10, or 20 ms)
3. If data!=NULL → normal decode:
   a. Init range decoder (ec_dec_init)
   b. Set audiosize = frame_size from state

4. Detect mode transitions (SILK↔CELT):
   - If transitioning without redundancy → set transition flag
   - Allocate transition buffer, pre-decode PLC into it for crossfade

5. SILK processing (if mode != CELT_ONLY):
   a. If prev_mode was CELT → reset SILK decoder
   b. Configure DecControl (internalSampleRate, payloadSize_ms, channels)
   c. Set lost_flag: 0=normal, 1=PLC, 2=FEC(LBRR)
   d. Call silk_Decode() in a loop until decoded_samples ≥ frame_size
      (SILK may produce 10ms sub-frames that need accumulation)
   e. If PLC failure → zero-fill (non-fatal)

6. Parse redundancy info (if present in bitstream):
   a. Hybrid: redundancy flag from ec_dec_bit_logp(12)
   b. SILK-only: redundancy always present (if enough bits)
   c. Read celt_to_silk direction flag
   d. Read redundancy_bytes (hybrid: ec_dec_uint(256)+2; SILK: remaining bytes)
   e. Shrink range decoder window by redundancy_bytes

7. Set CELT start_band = 17 when SILK is active (CELT covers 8-20 kHz only)

8. Configure CELT end_band by bandwidth:
   - NB=13, MB/WB=17, SWB=19, FB=21

9. Decode CELT→SILK redundancy (if celt_to_silk):
   - Decode 5ms (F5) redundancy from data+len

10. CELT processing (if mode != SILK_ONLY):
    a. If mode changed and no redundancy → reset CELT
    b. Decode CELT frame (possibly accumulating on top of SILK PCM)
    c. If SILK-only but prev was hybrid → decode 2.5ms silence for MDCT fade-out

11. Decode SILK→CELT redundancy (if !celt_to_silk):
    a. Reset CELT state
    b. Decode 5ms redundancy
    c. Crossfade: smooth_fade last 2.5ms of main output with first 2.5ms of redundancy

12. Apply CELT→SILK redundancy to output:
    a. Copy first 2.5ms from redundancy
    b. Crossfade next 2.5ms

13. Apply transition crossfade (if transition flag set):
    a. If audiosize ≥ F5: copy first 2.5ms from transition buffer, crossfade next 2.5ms
    b. Else: short crossfade over F2_5 samples

14. Apply decode gain (if non-zero):
    gain = celt_exp2(6.48814081e-4 * decode_gain)  [in Q25 × Q8 = Q-format product]
    Each sample: pcm[i] = SATURATE(pcm[i] * gain, 32767)

15. Compute final range:
    - If PLC (len ≤ 1): rangeFinal = 0
    - Else: rangeFinal ^= redundant_rng

16. Update state: prev_mode, prev_redundancy
17. Return audiosize (or celt_ret if negative)
```

### 4.5 FEC Decode Path

When `decode_fec=1` in `opus_decode_native`:

1. If frame_size < packet_frame_size, or mode is CELT-only → fall back to plain PLC
2. If frame_size > packet_frame_size → PLC for `(frame_size - packet_frame_size)` samples first
3. Set state to the incoming packet's mode/bandwidth/frame_size/stream_channels
4. Call `opus_decode_frame()` with `decode_fec=1` → SILK decodes LBRR data
5. Return `frame_size`

### 4.6 Soft Clipping (`opus_pcm_soft_clip`)

Float API only. Applies a smooth non-linearity to prevent hard clipping:

1. Clamp all samples to [-2, +2] range
2. For each channel, continue applying the non-linearity from previous frame (stored in `declip_mem`)
3. Find regions where |sample| > 1.0
4. Compute `a = (maxval - 1) / (maxval²)` such that `maxval + a*maxval² = 1`
5. Boost `a` by 2.4e-7 to guard against `-ffast-math` rounding
6. Apply `x = x + a*x²` for the affected region
7. Handle edge case at frame start with linear ramp to avoid discontinuity
8. Store `a` in `declip_mem` for continuity

### 4.7 Smooth Fade (`smooth_fade`)

Used for all mode transitions and redundancy crossfades:

```c
// For each sample in the overlap region:
w = window[i * inc]² // squared MDCT window
out = w * in2 + (1 - w) * in1
```

Where `inc = 48000/Fs` (step through the window at the output sample rate), and `window` is the CELT MDCT analysis/synthesis window from `CELTMode`.

Fixed-point version uses `MULT16_16_Q15(w, w)` for the squaring and `MAC16_16` / `MULT16_16` for the blend.

## 5. Data Flow

### 5.1 Input

An Opus packet byte buffer: `[TOC][frame_count_header?][size_fields?][frame_0][frame_1]...[padding]`

Maximum packet: 48 frames × 2.5 ms = 120 ms of audio. Max frame size: 1275 bytes.

### 5.2 Internal Buffers

| Buffer | Size | Purpose |
|--------|------|---------|
| `pcm` (output) | `frame_size × channels` | Main output, SILK writes directly here |
| `pcm_transition_silk`/`celt` | `F5 × channels` | PLC output for crossfade at mode transitions |
| `redundant_audio` | `F5 × channels` | 5 ms CELT redundancy frame |
| `pcm_silk` | `F10 × channels` | Temp buffer when frame_size < 10ms (SILK min) |

All declared via `VARDECL`/`ALLOC` (stack allocation macros).

### 5.3 Output

Interleaved PCM samples: `[L₀, R₀, L₁, R₁, ...]` for stereo, or `[S₀, S₁, ...]` for mono.

Output formats via wrapper functions:
- `opus_decode()` → `opus_int16` (16-bit signed, saturated)
- `opus_decode24()` → `opus_int32` (24-bit in 32-bit container)
- `opus_decode_float()` → `float` (±1.0 nominal range)

### 5.4 CELT Accumulation

In fixed-point mode when SILK is active (`mode != MODE_CELT_ONLY`), `celt_accum=1` tells the CELT decoder to *add* its output to the existing SILK PCM buffer rather than overwrite. This avoids a separate CELT buffer + summation pass. In floating-point mode, the SILK output is the native format and CELT accumulates on top.

## 6. Numerical Details

### 6.1 Fixed-Point Formats

| Symbol | Format | Range | Used for |
|--------|--------|-------|----------|
| `opus_val16` | Q15 (int16) | [-1.0, +1.0) | PCM samples (16-bit mode) |
| `opus_val32` | Q15 or Q16 (int32) | varies | Intermediate computation |
| `opus_res` | Depends on build: `opus_val16` (default), `opus_val32` (RES24), or `float` | — | Internal decode resolution |
| `decode_gain` | Q8 dB | [-32768, +32767] → ≈ ±128 dB | User-set gain |
| `celt_coef` | — | — | Window coefficients |

### 6.2 Gain Computation

```c
// decode_gain is in Q8 dB (1/256 dB per unit)
// 6.48814081e-4 ≈ ln(10)/(20*256*ln(2)) in Q25
gain = celt_exp2(MULT16_16_P15(QCONST16(6.48814081e-4f, 25), st->decode_gain));
```

In fixed-point:
- `QCONST16(6.48814081e-4f, 25)` → integer representation in Q25
- `MULT16_16_P15(a, b)` → `(a * b + 16384) >> 15` (Q25 × Q8 → Q18)
- `celt_exp2()` converts from Q18 log₂ domain to linear Q16
- Sample multiply: `MULT16_32_P16(pcm, gain)` or `MULT32_32_Q16(pcm, gain)` (RES24 mode)
- `SATURATE(x, 32767)` prevents overflow

In floating-point: all these are identity/passthrough operations; gain is just a float multiply.

### 6.3 Output Conversion

From internal `opus_res` to output format:

| Build | `opus_res` | → int16 | → int24 | → float |
|-------|-----------|---------|---------|---------|
| Fixed-point, no RES24 | `opus_val16` (Q15) | identity | `SHL32(EXTEND32(a), 8)` | `a/32768.0` |
| Fixed-point, RES24 | `opus_val32` (Q23) | `SAT16(PSHR32(a, 8))` | identity | `a/(32768*256)` |
| Floating-point | `float` | `FLOAT2INT16(a)` (clamp+round) | `float2int(a*8388608)` | identity |

### 6.4 Overflow Guards

- All PCM output is saturated to `[-32767, +32767]` (16-bit) or clamped accordingly
- Gain application uses `SATURATE(x, 32767)` after multiply
- Soft clipping boosts coefficient `a` by `2.4e-7` to guard against `-ffast-math` exceeding ±1.0
- Packet validation: frame sizes capped at 1275 bytes, total duration at 120 ms, `frame_size ≤ Fs/25*3` (120 ms)

## 7. Dependencies

### 7.1 What This Module Calls

| Module | Functions Called |
|--------|----------------|
| **SILK decoder** | `silk_Get_Decoder_Size()`, `silk_InitDecoder()`, `silk_ResetDecoder()`, `silk_Decode()` |
| **CELT decoder** | `celt_decoder_get_size()`, `celt_decoder_init()`, `celt_decode_with_ec()`, `celt_decode_with_ec_dred()` |
| **CELT decoder CTLs** | `CELT_SET_SIGNALLING`, `CELT_SET_START_BAND`, `CELT_SET_END_BAND`, `CELT_SET_CHANNELS`, `CELT_GET_MODE`, `OPUS_RESET_STATE`, `OPUS_GET_FINAL_RANGE`, `OPUS_GET_PITCH`, `OPUS_SET_COMPLEXITY`, `OPUS_SET_PHASE_INVERSION_DISABLED` |
| **Range coder** | `ec_dec_init()`, `ec_dec_bit_logp()`, `ec_dec_uint()`, `ec_tell()` |
| **Math** | `celt_exp2()` |
| **Deep PLC** (optional) | `lpcnet_plc_init()`, `lpcnet_plc_reset()`, `lpcnet_plc_fec_clear()`, `lpcnet_plc_fec_add()`, `lpcnet_plc_load_model()` |
| **OSCE** (optional) | `silk_LoadOSCEModels()` |
| **Extension iterator** | `opus_extension_iterator_init()`, `opus_extension_iterator_find()` |
| **Arch support** | `opus_select_arch()` |
| **Memory** | `opus_alloc()`, `opus_free()`, `OPUS_CLEAR()`, `OPUS_COPY()` |

### 7.2 What Calls This Module

| Caller | Function |
|--------|----------|
| **Application code** | `opus_decoder_create/init/destroy`, `opus_decode/decode_float/decode24`, `opus_decoder_ctl` |
| **Multistream decoder** (`opus_multistream_decoder.c`) | `opus_decode_native()` (with `self_delimited=1`) |
| **Packet inspection utilities** | `opus_packet_get_*()`, `opus_packet_parse()` (stateless, no decoder needed) |
| **DRED decoder** (`opus_decoder.c` itself) | `opus_decoder_dred_decode*()` → `opus_decode_native()` |

## 8. Constants and Tables

### 8.1 Mode Constants (from `opus_private.h`)

```c
#define MODE_SILK_ONLY   1000
#define MODE_HYBRID      1001
#define MODE_CELT_ONLY   1002
```

### 8.2 Bandwidth Constants (from `opus_defines.h`)

```c
#define OPUS_BANDWIDTH_NARROWBAND      1101  // 4 kHz
#define OPUS_BANDWIDTH_MEDIUMBAND      1102  // 6 kHz
#define OPUS_BANDWIDTH_WIDEBAND        1103  // 8 kHz
#define OPUS_BANDWIDTH_SUPERWIDEBAND   1104  // 12 kHz
#define OPUS_BANDWIDTH_FULLBAND        1105  // 20 kHz
```

### 8.3 SILK Decoder Flags (from `silk/control.h`)

```c
#define FLAG_DECODE_NORMAL  0
#define FLAG_PACKET_LOST    1
#define FLAG_DECODE_LBRR    2
```

### 8.4 CELT Band Limits

| Bandwidth | CELT end_band |
|-----------|---------------|
| Narrowband | 13 |
| Mediumband / Wideband | 17 |
| Superwideband | 19 |
| Fullband | 21 |

When SILK is active, `start_band = 17` (CELT only covers bands above 8 kHz).

### 8.5 Frame Duration Constants

Computed from `Fs`:
- `F20 = Fs/50` (20 ms)
- `F10 = Fs/100` (10 ms)
- `F5 = Fs/200` (5 ms)
- `F2_5 = Fs/400` (2.5 ms)

### 8.6 Key Numerical Constants

- Maximum frame size: `Fs/25 * 3` = 120 ms worth of samples
- Maximum frame byte size: 1275
- Maximum frames per packet: 48
- Maximum packet duration: 120 ms (5760 samples at 48 kHz, checked as `samples*25 > Fs*3`)
- Gain conversion factor: `6.48814081e-4f` in Q25 = `ln(10)/(20·256·ln(2))`
- Soft clip safety margin: `2.4e-7f` ≈ `2^{-22}`

### 8.7 Window

The crossfade window is not a table in this module — it's obtained from the CELT decoder's `CELTMode` struct via `CELT_GET_MODE`. It's the MDCT analysis/synthesis window, stepped through at rate `inc = 48000/Fs`.

## 9. Edge Cases

### 9.1 Error Conditions

| Condition | Error Code |
|-----------|-----------|
| `channels` not 1 or 2 | `OPUS_BAD_ARG` |
| `Fs` not in {8000,12000,16000,24000,48000} | `OPUS_BAD_ARG` |
| `frame_size ≤ 0` | `OPUS_BAD_ARG` |
| `decode_fec` not 0 or 1 | `OPUS_BAD_ARG` |
| PLC/FEC `frame_size` not multiple of `Fs/400` | `OPUS_BAD_ARG` |
| `len < 0` (with non-NULL data) | `OPUS_BAD_ARG` |
| Output buffer too small for decoded frames | `OPUS_BUFFER_TOO_SMALL` |
| Packet parsing fails (truncated, corrupt) | `OPUS_INVALID_PACKET` |
| SILK init fails | `OPUS_INTERNAL_ERROR` |
| SILK decode fails (non-PLC) | `OPUS_INTERNAL_ERROR` |
| Packet with `nb_samples*25 > Fs*3` (> 120ms) | `OPUS_INVALID_PACKET` |
| CBR packet with `len` not divisible by `count` | `OPUS_INVALID_PACKET` |

### 9.2 Special Input Handling

- **Payload ≤ 1 byte** (including 0): Treated as packet loss — `data` set to NULL, audiosize clamped to `frame_size`
- **No previous mode** (`prev_mode == 0`): First packet loss → output silence (zeros)
- **PLC audiosize > 20ms**: Recursively decoded in 20ms chunks
- **PLC audiosize between valid sizes**: Snapped down to 10ms, 5ms, or 2.5ms boundaries
- **SILK PLC failure**: Non-fatal — zero-fill the frame and continue
- **Mode transition without redundancy**: Crossfade using PLC from previous mode
- **Hybrid→SILK transition**: CELT decodes a 2.5ms silence frame to allow MDCT fade-out
- **Redundancy with incomplete CELT state** (prev_mode was SILK-only): Redundancy is decoded for range testing but audio is discarded
- **Frame too small for clean transition** (< 5ms): A "not-great-but-best-we-can-do" short crossfade

### 9.3 Redundancy Edge Cases

The redundancy frame handling has careful ordering:
- **CELT→SILK**: Redundancy decoded *before* main CELT frame; first 2.5ms comes from redundancy, next 2.5ms crossfaded
- **SILK→CELT**: Redundancy decoded *after* main CELT frame; last 2.5ms of main output crossfaded with redundancy
- Guard condition for CELT→SILK: redundancy audio only used if `prev_mode != MODE_SILK_ONLY || prev_redundancy` (ensures CELT state isn't stale)

## 10. Porting Notes

### 10.1 Inline Sub-Decoder Allocation

The C code allocates SILK and CELT decoder states at computed byte offsets within a single allocation:

```c
silk_dec = (char*)st + st->silk_dec_offset;
celt_dec = (CELTDecoder*)((char*)st + st->celt_dec_offset);
```

**Rust approach**: Use a struct with owned `SilkDecoder` and `CeltDecoder` fields. No need to replicate the pointer arithmetic pattern. The `opus_decoder_get_size()` function becomes unnecessary — Rust's type system handles sizing.

### 10.2 Stack Allocation Macros

The C code uses `VARDECL` / `ALLOC` / `ALLOC_STACK` / `RESTORE_STACK` for dynamically-sized stack arrays. These may expand to `alloca()` or VLAs.

**Rust approach**: Use `Vec<T>` with pre-computed capacity, or `SmallVec` for small sizes. The `ALLOC_NONE` sentinel (size 0) maps to not allocating at all — use `Option<Vec<T>>`.

### 10.3 Conditional Compilation Complexity

The module has extensive `#ifdef` gates:
- `FIXED_POINT` vs floating-point → different `opus_res` types, different arithmetic
- `ENABLE_RES24` → 24-bit internal resolution
- `ENABLE_DEEP_PLC` → LPCNet neural PLC
- `ENABLE_OSCE` / `ENABLE_OSCE_BWE` → neural speech enhancement
- `ENABLE_DRED` → deep redundancy
- `ENABLE_QEXT` → 96 kHz support
- `DISABLE_FLOAT_API` → no float decode path

**Rust approach**: Use Cargo features. For the initial port, target floating-point with no DNN features, matching the most common build configuration. Add fixed-point as a separate feature gate later.

### 10.4 Variadic CTL Interface

`opus_decoder_ctl()` uses C varargs to accept different parameter types per request code.

**Rust approach**: Use an enum with typed variants:
```rust
enum DecoderCtl {
    SetGain(i32),
    GetGain,
    SetComplexity(i32),
    ResetState,
    // ...
}
```
Or provide individual getter/setter methods on the decoder struct.

### 10.5 In-Place Mutation

The `smooth_fade` function and redundancy application modify the output buffer in-place with overlapping reads/writes:

```c
// Output overwrites part of input
smooth_fade(pcm+offset, redundant_audio+offset, pcm+offset, F2_5, ...);
```

**Rust approach**: Since `in1`/`out` can alias (both point into `pcm`), you'll need to either use `unsafe` with raw pointers, or copy to a temporary buffer first. The overlap is always at most F2_5 samples, so a small temp buffer on the stack is fine.

### 10.6 Recursive Self-Calls

`opus_decode_frame` calls itself recursively for:
- PLC with audiosize > 20ms (chunks of 20ms)
- Mode transition crossfade (decode PLC into transition buffer)

**Rust approach**: The recursion depth is bounded: PLC recursion is at most `ceil(audiosize/F20)` ≈ 6 at 48 kHz (120ms/20ms). Transition decode is always exactly 1 level deep. Stack overflow is not a concern, but iterative approaches are also fine.

### 10.7 The `celt_accum` Pattern

In fixed-point mode, when both SILK and CELT are active, SILK writes to the PCM buffer first, then CELT accumulates (adds) on top via the `celt_accum` flag.

**Rust approach**: Pass a flag to the CELT decoder that controls whether it overwrites or adds to the output buffer.

### 10.8 Range Coder State Manipulation

The redundancy logic directly manipulates the range decoder's `storage` field to exclude redundancy bytes from the main bitstream:

```c
dec.storage -= redundancy_bytes;
```

**Rust approach**: The range decoder struct will need a public method or field access pattern that allows shrinking the available storage.

### 10.9 Error Handling Patterns

The C code mixes:
- Negative return values as error codes
- `goto bad_arg` in the CTL function
- Fatal asserts (`celt_assert`) vs recoverable errors
- Non-fatal PLC errors (SILK PLC failure → zero-fill and continue)

**Rust approach**: Use `Result<T, OpusError>` for all fallible operations. The PLC-failure-continues pattern maps naturally to matching on the SILK decode result and falling through to zero-fill.

### 10.10 Macro-Generated Arithmetic

The fixed-point path uses many macros (`MULT16_16_Q15`, `MULT16_32_P16`, `QCONST16`, `SATURATE`, `SHR32`, `MAC16_16`) that expand to different code in fixed vs float builds. In floating-point builds they are identity or simple float operations.

**Rust approach**: Define these as generic functions or trait methods that dispatch at compile time via feature flags, or use separate fixed-point and float implementations.

### 10.11 The `align()` Function

```c
static OPUS_INLINE int align(int i) {
    struct foo {char c; union { void* p; opus_int32 i; opus_val32 v; } u;};
    unsigned int alignment = offsetof(struct foo, u);
    return ((i + alignment - 1) / alignment) * alignment;
}
```

This computes the natural alignment of the most-aligned type in the union. Used only for the sub-decoder byte offset computation.

**Rust approach**: Not needed if sub-decoders are owned struct fields rather than inline byte offsets. If replicating the memory layout for FFI test harness compatibility, use `std::mem::align_of::<T>()`.

### 10.12 Thread Safety

The C code documents that a single decoder state must not be accessed from multiple threads. The Rust `OpusDecoder` should be `Send` but not `Sync` (or wrap in `Mutex` at a higher layer). No interior mutability is needed.

### 10.13 `MUST_SUCCEED` Macro

Several CELT CTL calls are wrapped in `MUST_SUCCEED()`, which presumably asserts the return is `OPUS_OK`. These are internal calls that should never fail with valid state — they represent programming errors if they do.

**Rust approach**: Use `.expect("internal CELT CTL must succeed")` or `debug_assert!` on the result. These invariants should hold by construction in a correct implementation.
