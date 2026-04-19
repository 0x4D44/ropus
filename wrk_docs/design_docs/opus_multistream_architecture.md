Now I have all the information needed. Let me compose the document.

---

# Opus Multistream Module Architecture

## 1. Purpose

The multistream module is the **top-level multiplexing layer** of the Opus codec. It sits above the core Opus encoder/decoder and manages encoding/decoding of **multi-channel audio** (more than 2 channels, up to 255). Its responsibilities:

- **Decomposing** N input channels into a set of independent Opus streams (mono or coupled stereo pairs) according to a channel mapping
- **Bitrate allocation** across those streams (surround-aware or ambisonics-equal)
- **Surround analysis** to exploit inter-channel masking (for mapping family 1)
- **Concatenating** per-stream encoded packets into a single self-delimiting multistream packet
- **Parsing** multistream packets back into per-stream sub-packets for decoding
- **Reconstructing** multi-channel PCM output by routing decoded stream outputs to the correct output channels

The **projection** sub-module extends multistream with **matrix-based channel mixing/demixing** for Ambisonics support (mapping family 3). It wraps a multistream encoder/decoder and applies pre-computed mixing/demixing matrices to transform between Ambisonics channel ordering (ACN) and the internal stream representation.

### Mapping Families

| Family | Use Case | Max Channels | Stream Layout |
|--------|----------|-------------|---------------|
| 0 | Mono/Stereo | 2 | 1 stream, 0-1 coupled |
| 1 | Surround (Vorbis) | 8 | Vorbis channel mappings, LFE handling |
| 2 | Ambisonics (discrete) | 227 | `(order+1)^2` ACN + optional 2 non-diegetic |
| 3 | Ambisonics (projection) | 227 | Matrix-mixed, paired into coupled streams |
| 255 | Discrete | 255 | One mono stream per channel, no coupling |

## 2. Public API

### 2.1 Multistream Encoder

```c
// Size query (for pre-allocated memory)
opus_int32 opus_multistream_encoder_get_size(int nb_streams, int nb_coupled_streams);
opus_int32 opus_multistream_surround_encoder_get_size(int channels, int mapping_family);

// Create/Init
OpusMSEncoder *opus_multistream_encoder_create(
    opus_int32 Fs, int channels, int streams, int coupled_streams,
    const unsigned char *mapping, int application, int *error);

int opus_multistream_encoder_init(
    OpusMSEncoder *st, opus_int32 Fs, int channels, int streams,
    int coupled_streams, const unsigned char *mapping, int application);

OpusMSEncoder *opus_multistream_surround_encoder_create(
    opus_int32 Fs, int channels, int mapping_family,
    int *streams, int *coupled_streams, unsigned char *mapping,
    int application, int *error);

int opus_multistream_surround_encoder_init(
    OpusMSEncoder *st, opus_int32 Fs, int channels, int mapping_family,
    int *streams, int *coupled_streams, unsigned char *mapping, int application);

// Encode
int opus_multistream_encode(OpusMSEncoder *st, const opus_int16 *pcm,
    int frame_size, unsigned char *data, opus_int32 max_data_bytes);
int opus_multistream_encode_float(OpusMSEncoder *st, const float *pcm,
    int frame_size, unsigned char *data, opus_int32 max_data_bytes);
int opus_multistream_encode24(OpusMSEncoder *st, const opus_int32 *pcm,
    int frame_size, unsigned char *data, opus_int32 max_data_bytes);

// Control/Destroy
int opus_multistream_encoder_ctl(OpusMSEncoder *st, int request, ...);
void opus_multistream_encoder_destroy(OpusMSEncoder *st);
```

**Internal (not public but critical):**
```c
int opus_multistream_encode_native(
    OpusMSEncoder *st,
    opus_copy_channel_in_func copy_channel_in,
    const void *pcm, int analysis_frame_size,
    unsigned char *data, opus_int32 max_data_bytes,
    int lsb_depth, downmix_func downmix, int float_api, void *user_data);
```

### 2.2 Multistream Decoder

```c
opus_int32 opus_multistream_decoder_get_size(int nb_streams, int nb_coupled_streams);

OpusMSDecoder *opus_multistream_decoder_create(
    opus_int32 Fs, int channels, int streams, int coupled_streams,
    const unsigned char *mapping, int *error);

int opus_multistream_decoder_init(
    OpusMSDecoder *st, opus_int32 Fs, int channels, int streams,
    int coupled_streams, const unsigned char *mapping);

int opus_multistream_decode(OpusMSDecoder *st, const unsigned char *data,
    opus_int32 len, opus_int16 *pcm, int frame_size, int decode_fec);
int opus_multistream_decode_float(OpusMSDecoder *st, const unsigned char *data,
    opus_int32 len, float *pcm, int frame_size, int decode_fec);
int opus_multistream_decode24(OpusMSDecoder *st, const unsigned char *data,
    opus_int32 len, opus_int32 *pcm, int frame_size, int decode_fec);

int opus_multistream_decoder_ctl(OpusMSDecoder *st, int request, ...);
void opus_multistream_decoder_destroy(OpusMSDecoder *st);
```

**Internal:**
```c
int opus_multistream_decode_native(
    OpusMSDecoder *st, const unsigned char *data, opus_int32 len,
    void *pcm, opus_copy_channel_out_func copy_channel_out,
    int frame_size, int decode_fec, int soft_clip, void *user_data);

int opus_multistream_decoder_ctl_va_list(OpusMSDecoder *st, int request, va_list ap);
int opus_multistream_encoder_ctl_va_list(OpusMSEncoder *st, int request, va_list ap);
```

### 2.3 Projection Encoder (Ambisonics)

```c
opus_int32 opus_projection_ambisonics_encoder_get_size(int channels, int mapping_family);

OpusProjectionEncoder *opus_projection_ambisonics_encoder_create(
    opus_int32 Fs, int channels, int mapping_family,
    int *streams, int *coupled_streams, int application, int *error);

int opus_projection_ambisonics_encoder_init(OpusProjectionEncoder *st,
    opus_int32 Fs, int channels, int mapping_family,
    int *streams, int *coupled_streams, int application);

int opus_projection_encode(OpusProjectionEncoder *st, const opus_int16 *pcm,
    int frame_size, unsigned char *data, opus_int32 max_data_bytes);
int opus_projection_encode_float(OpusProjectionEncoder *st, const float *pcm,
    int frame_size, unsigned char *data, opus_int32 max_data_bytes);
int opus_projection_encode24(OpusProjectionEncoder *st, const opus_int32 *pcm,
    int frame_size, unsigned char *data, opus_int32 max_data_bytes);

int opus_projection_encoder_ctl(OpusProjectionEncoder *st, int request, ...);
void opus_projection_encoder_destroy(OpusProjectionEncoder *st);
```

### 2.4 Projection Decoder

```c
opus_int32 opus_projection_decoder_get_size(int channels, int streams, int coupled_streams);

OpusProjectionDecoder *opus_projection_decoder_create(
    opus_int32 Fs, int channels, int streams, int coupled_streams,
    unsigned char *demixing_matrix, opus_int32 demixing_matrix_size, int *error);

int opus_projection_decoder_init(OpusProjectionDecoder *st, opus_int32 Fs,
    int channels, int streams, int coupled_streams,
    unsigned char *demixing_matrix, opus_int32 demixing_matrix_size);

int opus_projection_decode(OpusProjectionDecoder *st, const unsigned char *data,
    opus_int32 len, opus_int16 *pcm, int frame_size, int decode_fec);
int opus_projection_decode_float(OpusProjectionDecoder *st, const unsigned char *data,
    opus_int32 len, float *pcm, int frame_size, int decode_fec);
int opus_projection_decode24(OpusProjectionDecoder *st, const unsigned char *data,
    opus_int32 len, opus_int32 *pcm, int frame_size, int decode_fec);

int opus_projection_decoder_ctl(OpusProjectionDecoder *st, int request, ...);
void opus_projection_decoder_destroy(OpusProjectionDecoder *st);
```

### 2.5 Mapping Matrix

```c
opus_int32 mapping_matrix_get_size(int rows, int cols);
opus_int16 *mapping_matrix_get_data(const MappingMatrix *matrix);
void mapping_matrix_init(MappingMatrix *matrix, int rows, int cols,
    int gain, const opus_int16 *data, opus_int32 data_size);

// Channel multiply operations (6 variants: in/out x float/short/int24)
void mapping_matrix_multiply_channel_in_float(...);
void mapping_matrix_multiply_channel_out_float(...);
void mapping_matrix_multiply_channel_in_short(...);
void mapping_matrix_multiply_channel_out_short(...);
void mapping_matrix_multiply_channel_in_int24(...);
void mapping_matrix_multiply_channel_out_int24(...);
```

### 2.6 Shared Utilities (opus_multistream.c)

```c
int validate_layout(const ChannelLayout *layout);
int get_left_channel(const ChannelLayout *layout, int stream_id, int prev);
int get_right_channel(const ChannelLayout *layout, int stream_id, int prev);
int get_mono_channel(const ChannelLayout *layout, int stream_id, int prev);
```

## 3. Internal State

### 3.1 ChannelLayout

```c
typedef struct ChannelLayout {
   int nb_channels;           // Total output/input channels (1..255)
   int nb_streams;            // Number of Opus streams
   int nb_coupled_streams;    // Number of stereo (coupled) streams
   unsigned char mapping[256]; // Channel-to-stream mapping table
} ChannelLayout;
```

The `mapping[]` array maps each output channel index to a stream index:
- Values `0` to `2*nb_coupled_streams - 1`: Left (even) or right (odd) of a coupled stream. Stream `s` has left=`2s`, right=`2s+1`.
- Values `2*nb_coupled_streams` to `nb_streams + nb_coupled_streams - 1`: Mono streams, offset by `nb_coupled_streams`.
- Value `255`: Silent/muted channel.

### 3.2 MappingType Enum

```c
typedef enum {
  MAPPING_TYPE_NONE,       // No surround processing (family 0, 255)
  MAPPING_TYPE_SURROUND,   // Surround analysis enabled (family 1, channels > 2)
  MAPPING_TYPE_AMBISONICS  // Ambisonics mode (family 2)
} MappingType;
```

### 3.3 OpusMSEncoder

```c
struct OpusMSEncoder {
   ChannelLayout layout;
   int arch;                  // CPU arch for SIMD dispatch
   int lfe_stream;            // Stream index of LFE channel (-1 if none)
   int application;           // OPUS_APPLICATION_* constant
   opus_int32 Fs;             // Sample rate
   int variable_duration;     // OPUS_FRAMESIZE_ARG or explicit frame size
   MappingType mapping_type;  // See above
   opus_int32 bitrate_bps;    // Target bitrate (or OPUS_AUTO/OPUS_BITRATE_MAX)
   /* --- Variable-length data follows in memory --- */
   // OpusEncoder[0..nb_coupled_streams-1]  (stereo encoders, align'd)
   // OpusEncoder[nb_coupled_streams..nb_streams-1]  (mono encoders, align'd)
   // opus_val32 window_mem[nb_channels * MAX_OVERLAP]  (surround only)
   // opus_val32 preemph_mem[nb_channels]               (surround only)
};
```

**Memory layout** (contiguous allocation):

```
[OpusMSEncoder header | pad]
[coupled encoder 0 | pad] ... [coupled encoder N-1 | pad]
[mono encoder 0 | pad] ... [mono encoder M-1 | pad]
[window_mem: channels*MAX_OVERLAP opus_val32]   // surround only
[preemph_mem: channels opus_val32]              // surround only
```

All sub-objects are `align()`-padded. MAX_OVERLAP is 120 (or 240 with ENABLE_QEXT).

### 3.4 OpusMSDecoder

```c
struct OpusMSDecoder {
   ChannelLayout layout;
   /* --- Variable-length data follows --- */
   // OpusDecoder[0..nb_coupled_streams-1]  (stereo decoders, align'd)
   // OpusDecoder[nb_coupled_streams..nb_streams-1]  (mono decoders, align'd)
};
```

### 3.5 OpusProjectionEncoder

```c
struct OpusProjectionEncoder {
  opus_int32 mixing_matrix_size_in_bytes;
  opus_int32 demixing_matrix_size_in_bytes;
  /* --- Variable-length data follows --- */
  // MappingMatrix mixing_matrix (with data)
  // MappingMatrix demixing_matrix (with data)
  // OpusMSEncoder (the underlying multistream encoder)
};
```

### 3.6 OpusProjectionDecoder

```c
struct OpusProjectionDecoder {
  opus_int32 demixing_matrix_size_in_bytes;
  /* --- Variable-length data follows --- */
  // MappingMatrix demixing_matrix (with data)
  // OpusMSDecoder (the underlying multistream decoder)
};
```

### 3.7 MappingMatrix

```c
typedef struct MappingMatrix {
    int rows;   // Output channels of the matrix
    int cols;   // Input channels of the matrix
    int gain;   // Global gain in dB, S7.8 fixed-point format
} MappingMatrix;
// Followed immediately by: opus_int16 data[rows * cols] (column-major order)
```

The matrix data is stored **column-wise** (column-major). Access pattern:
```c
#define MATRIX_INDEX(nb_rows, row, col) (nb_rows * col + row)
```

### 3.8 Function Pointer Types

```c
typedef void (*opus_copy_channel_in_func)(
    opus_res *dst, int dst_stride,
    const void *src, int src_stride,
    int src_channel, int frame_size, void *user_data);

typedef void (*opus_copy_channel_out_func)(
    void *dst, int dst_stride, int dst_channel,
    const opus_res *src, int src_stride,
    int frame_size, void *user_data);
```

These provide the **format abstraction layer** -- the encode/decode native functions are format-agnostic, with the actual float/int16/int24 conversion handled by these callbacks.

### 3.9 Lifecycle

1. **get_size()** -- compute total allocation size
2. **create()** -- `opus_alloc()` + `init()`
3. **init()** -- populate header fields, copy mapping, initialize all sub-encoders/decoders
4. **encode()/decode()** -- stateful, called per frame
5. **ctl()** -- query/set parameters, forwarded to sub-encoders/decoders
6. **destroy()** -- single `opus_free()` (everything is one contiguous allocation)

## 4. Algorithm

### 4.1 Encoding (`opus_multistream_encode_native`)

**Step 1: Setup**
- Get sample rate, VBR mode, CELT mode from the first sub-encoder
- Select frame size from `variable_duration` setting
- Compute minimum packet size: `2*nb_streams - 1` bytes (plus 1 extra per stream for 100ms frames)

**Step 2: Surround Analysis (mapping_type == SURROUND only)**
- Call `surround_analysis()` which:
  - For each channel, applies pre-emphasis and computes MDCT
  - Computes per-band energies (21 CELT bands)
  - Applies spreading function: +6 dB/band up, +12 dB/band down
  - Accumulates left/right/center masks using `logSum()` (approximate log2(2^a + 2^b))
  - Subtracts the spatial mask to produce the Signal-to-Mask Ratio (SMR) per band per channel
  - SMR is stored in `bandSMR[21*channel + band]`

**Step 3: Bitrate Allocation**
- `rate_allocation()` dispatches to:
  - `surround_rate_allocation()` for SURROUND mode: LFE gets 1/8 mono rate, coupled streams get ~2x mono rate, per-stream offset models coupling savings
  - `ambisonics_rate_allocation()` for AMBISONICS: equal rate per stream
- Minimum rate per stream: 500 bps

**Step 4: Configure Sub-Encoders**
- Set per-stream bitrate
- For SURROUND: force bandwidth based on equivalent rate, force coupled streams to MODE_CELT_ONLY with stereo
- For AMBISONICS: force all streams to MODE_CELT_ONLY

**Step 5: Encode Each Stream**
For each stream `s` in order:
1. **Extract channels**: Use `copy_channel_in` callback to deinterleave:
   - Coupled stream: extract left and right channels into interleaved stereo buffer `buf[2*frame_size]`
   - Mono stream: extract single channel into `buf[frame_size]`
2. **Set energy mask** (SURROUND only): copy per-band SMR to `bandLogE[]`
3. **Compute max bytes**: Reserve space for remaining streams (2 bytes each for self-delimiting, 1 for the last)
4. **Encode**: Call `opus_encode_native()` into `tmp_data[]`
5. **Repacketize**: Use `OpusRepacketizer` to add self-delimiting length prefixes. All streams except the last use self-delimiting framing; the last stream's length is implicit (remainder of packet).
6. **Append** repacketized data to output, advance data pointer

**Return**: Total encoded size in bytes.

### 4.2 Decoding (`opus_multistream_decode_native`)

**Step 1: Validate**
- Limit frame_size to `Fs/25*3` (120ms max) to bound stack allocation
- If `len == 0`, enter PLC (packet loss concealment) mode
- If not PLC, validate the multistream packet (parse all sub-packets, verify consistent frame count)

**Step 2: Decode Each Stream**
For each stream `s`:
1. Get the sub-decoder pointer by walking the memory layout
2. Call `opus_decode_native()` with the current data pointer
   - Non-last streams use self-delimiting parsing (`s != nb_streams-1`)
   - `packet_offset` tells how many bytes were consumed
3. Advance `data` and `len` by `packet_offset`
4. Route decoded audio using `copy_channel_out` callback:
   - **Coupled stream**: Route left channel (`buf[0], stride 2`) and right channel (`buf[1], stride 2`) to all output channels mapped to this stream via `get_left_channel()` / `get_right_channel()`
   - **Mono stream**: Route mono to all output channels mapped via `get_mono_channel()`

**Step 3: Handle Muted Channels**
- For any channel with `mapping[c] == 255`, call `copy_channel_out` with `src=NULL` to write silence

**Return**: Number of decoded samples per channel.

### 4.3 Projection Encode/Decode

The projection layer wraps multistream and adds matrix operations:

**Encode**: The `copy_channel_in` callback is replaced with `opus_projection_copy_channel_in_{short,float,int24}`, which calls `mapping_matrix_multiply_channel_in_*()` with the mixing matrix. This applies the mixing transform before the data reaches the multistream encoder. The `user_data` parameter carries a pointer to the mixing matrix.

**Decode**: The `copy_channel_out` callback is replaced with `opus_projection_copy_channel_out_{short,float,int24}`, which calls `mapping_matrix_multiply_channel_out_*()` with the demixing matrix. This applies the demixing transform to the decoded streams. On the first output channel (`dst_channel == 0`), the entire output buffer is zeroed; then each stream's contribution is accumulated additively.

### 4.4 Mapping Matrix Multiply

The matrix multiply is decomposed into per-channel operations (one row or one column at a time), called once per stream during encode/decode:

**`multiply_channel_in` (encode-side):**
```
For each sample i in frame:
    output[output_rows * i] = sum over col:
        matrix[output_row, col] * input[col, i]
```
One output row is computed per call. The matrix is applied to all input channels to produce one output stream sample.

**`multiply_channel_out` (decode-side):**
```
For each sample i in frame:
    input_sample = input[input_rows * i]
    For each output row:
        output[row, i] += matrix[row, input_row] * input_sample
```
One input column is scattered to all output rows. Output is **accumulated** (+=), not overwritten.

## 5. Data Flow

### 5.1 Encoder Data Flow

```
Multi-channel PCM (interleaved, channels * frame_size samples)
    |
    v
[copy_channel_in / matrix_multiply_in]  -- deinterleave + optional mixing
    |
    v
Per-stream mono/stereo PCM buffer (1 or 2 * frame_size)
    |
    v
[opus_encode_native]  -- per-stream encoding
    |
    v
Per-stream compressed bytes (in tmp_data[])
    |
    v
[opus_repacketizer]  -- add self-delimiting framing
    |
    v
Concatenated multistream packet
```

### 5.2 Decoder Data Flow

```
Multistream packet (concatenated self-delimiting sub-packets)
    |
    v
[opus_packet_parse_impl]  -- find sub-packet boundaries
    |
    v
Per-stream compressed bytes
    |
    v
[opus_decode_native]  -- per-stream decoding
    |
    v
Per-stream mono/stereo PCM (in buf[2*frame_size])
    |
    v
[copy_channel_out / matrix_multiply_out]  -- route/mix to output channels
    |
    v
Multi-channel PCM (interleaved, channels * frame_size)
```

### 5.3 Buffer Layouts

**Input/Output PCM**: Interleaved. Sample `[frame][channel]` is at `pcm[frame * nb_channels + channel]`.

**Internal per-stream buffer** (`buf`): Allocated as `2 * frame_size` elements of `opus_res`.
- Coupled (stereo): interleaved L/R, stride 2. `buf[0]` = L sample 0, `buf[1]` = R sample 0, `buf[2]` = L sample 1, etc.
- Mono: stride 1. `buf[0]` = sample 0, `buf[1]` = sample 1, etc.

**Matrix data**: Column-major `opus_int16[rows * cols]`. Element at (row, col) is `data[rows * col + row]`.

**Multistream packet format**:
```
[stream 0: self-delimiting] [stream 1: self-delimiting] ... [stream N-1: standard]
```
Each self-delimiting sub-packet has its byte length encoded at the start. The last stream's packet runs to the end of the buffer.

### 5.4 Mapping Table Encoding

The `mapping[]` array encodes this relationship:
- Total "stream channels" = `nb_streams + nb_coupled_streams` (each coupled stream contributes 2 channels)
- For a coupled stream `s` (where `s < nb_coupled_streams`): left = `2*s`, right = `2*s + 1`
- For a mono stream `s` (where `s >= nb_coupled_streams`): `s + nb_coupled_streams`
- `mapping[output_channel] == 255` means silence

## 6. Numerical Details

### 6.1 Matrix Coefficients

Matrix data is stored as `opus_int16` values representing **Q15 fixed-point** numbers (range approximately -1.0 to +0.99997):
- `32767` = ~1.0
- `16384` = ~0.5
- `23170` = ~sqrt(2)/2 ≈ 0.7071

### 6.2 Matrix Multiply -- Fixed-Point Path

**`multiply_channel_in_short` (Q15 matrix x Q15 input = Q30, then shift):**
```c
// Fixed-point: 16x16 -> 32-bit accumulator, shifted right by 8
tmp += ((opus_int32)matrix[...] * (opus_int32)input[...]) >> 8;
// Final: (tmp + 64) >> 7  -- rounds to nearest with bias of 0.5 LSB
// Result is Q15 (opus_int16) or Q(RES_SHIFT-7) for 24-bit res
```

The total right-shift is `8 + 7 = 15`, converting Q30 back to Q15. The `+ 64` before the final `>> 7` provides **rounding** (64 = 2^6 = half of 2^7).

With ENABLE_RES24: `SHL32(tmp, RES_SHIFT-7)` instead of the saturate+shift.

**`multiply_channel_out_short` (Q15 matrix x Q15 decoded):**
```c
input_sample = RES2INT16(input[...]);  // Convert internal to Q15
tmp = (opus_int32)matrix[...] * input_sample;  // Q15 * Q15 = Q30
output[...] += (tmp + 16384) >> 15;  // Round and shift Q30 -> Q15
```

**`multiply_channel_in_int24` / `out_int24`:**
Uses `opus_val64` accumulator for 24-bit precision. Shift by 15 with `+ 16384` rounding.

### 6.3 Matrix Multiply -- Float Path

```c
// In: matrix coefficients scaled by 1/32768.0
tmp += matrix[...] * input[...];
output[...] = (1/32768.f) * tmp;  // or (1/(32768*32768)) for short input
```

### 6.4 Matrix Gain Field

`MappingMatrix.gain` is an **S7.8 fixed-point** value representing gain in dB. The pre-computed matrices for FOA/SOA/TOA have gain=0 except SOA demixing which has gain=3050 (approximately 11.9 dB). This gain is metadata for the Ogg encapsulation; it is **not applied** in the multiply functions -- it must be applied externally.

### 6.5 Surround Analysis Numerical Details

- Band energies computed using MDCT (21 CELT bands)
- `logSum(a, b)` approximates `log2(2^a + 2^b)` using a 17-entry lookup table (`diff_table`), linear interpolation between entries
- Fixed-point: uses `celt_glog` format (DB_SHIFT-based), `GCONST()` macro for constants
- Spreading function: -6 dB/band upward, -12 dB/band downward
- Center mask = min(left mask, right mask)
- Channel offset: `0.5 * log2(2/(channels-1))`

### 6.6 Bitrate Allocation Arithmetic

`surround_rate_allocation()` uses Q8 ratios:
- `coupled_ratio = 512` (2.0x in Q8)
- `lfe_ratio = 32` (0.125x in Q8 = 1/8)
- `channel_rate` computed in Q8: `256 * (bitrate - offsets) / total`
- Per-stream rate: `offset + (channel_rate * ratio >> 8)`

### 6.7 Overflow Guards

- Matrix size capped: `rows <= 255, cols <= 255, rows*cols*2 <= 65004`
- Frame size capped at `Fs/25*3` in decoder to limit stack allocation
- Bitrate clamped to `[500*channels, 750000*channels]`
- Minimum per-stream rate: 500 bps
- `SATURATE16()` used in fixed-point matrix output to clamp to `[-32768, 32767]`

## 7. Dependencies

### 7.1 What This Module Calls

| Module | Function | Purpose |
|--------|----------|---------|
| `opus_encoder.c` | `opus_encoder_init`, `opus_encode_native`, `opus_encoder_ctl`, `opus_encoder_get_size` | Per-stream encoding |
| `opus_decoder.c` | `opus_decoder_init`, `opus_decode_native`, `opus_decoder_ctl`, `opus_decoder_get_size` | Per-stream decoding |
| `opus.c` | `opus_packet_parse_impl`, `opus_packet_get_nb_samples` | Packet validation |
| `repacketizer.c` | `opus_repacketizer_init`, `opus_repacketizer_cat`, `opus_repacketizer_out_range_impl`, `opus_repacketizer_get_nb_frames` | Self-delimiting framing |
| `bands.c` | `compute_band_energies` | Surround analysis |
| `mdct.c` | `clt_mdct_forward` | Surround analysis MDCT |
| `modes.c` | `resampling_factor`, CELTMode fields | Surround analysis |
| `quant_bands.c` | `amp2Log2` | Energy-to-log conversion |
| `pitch.c` | `celt_inner_prod` | NaN detection |
| `mathops.c` | `celt_log2`, `isqrt32`, `celt_preemphasis` | Math utilities |

### 7.2 What Calls This Module

- Application code (top-level API entry point)
- Projection encoder/decoder (wraps multistream)
- Ogg encapsulation layer (not part of libopus core)

## 8. Constants and Tables

### 8.1 Vorbis Channel Mappings

```c
static const VorbisLayout vorbis_mappings[8] = {
    {1, 0, {0}},                      // 1ch: mono
    {1, 1, {0, 1}},                   // 2ch: stereo
    {2, 1, {0, 2, 1}},                // 3ch: L, C, R -> 1 coupled + 1 mono
    {2, 2, {0, 1, 2, 3}},             // 4ch: FL, FR, RL, RR -> 2 coupled
    {3, 2, {0, 4, 1, 2, 3}},          // 5ch: FL, C, FR, RL, RR
    {4, 2, {0, 4, 1, 2, 3, 5}},       // 6ch: 5.1
    {4, 3, {0, 4, 1, 2, 3, 5, 6}},    // 7ch: 6.1
    {5, 3, {0, 6, 1, 2, 3, 4, 5, 7}}, // 8ch: 7.1
};
```

### 8.2 Channel Positions for Surround Analysis

`channel_pos()` assigns spatial positions for surround mixing analysis:
- 0 = don't mix (LFE)
- 1 = left
- 2 = center
- 3 = right

Only defined for 4-8 channels.

### 8.3 logSum Lookup Table

17-entry table for approximating `log2(2^a + 2^b)`, values from `GCONST(0.5)` down to `GCONST(0.0028123)`, indexed by `floor(2*diff)` where `diff = |a - b|`.

### 8.4 Ambisonics Mixing/Demixing Matrices

Pre-computed matrices for orders 1-5:

| Order | Name | Rows x Cols | Data Size | Gain (S7.8 dB) |
|-------|------|-------------|-----------|------|
| 1 (FOA) | `mapping_matrix_foa_mixing` | 6x6 | 36 | 0 |
| 2 (SOA) | `mapping_matrix_soa_mixing` | 11x11 | 121 | 0 |
| 3 (TOA) | `mapping_matrix_toa_mixing` | 18x18 | 324 | 0 |
| 4th | `mapping_matrix_fourthoa_mixing` | 27x27 | 729 | 0 |
| 5th | `mapping_matrix_fifthoa_mixing` | 38x38 | 1444 | 0 |

Demixing matrices have the same dimensions. SOA demixing has gain=3050 (≈11.9 dB in S7.8).

The matrix dimensions follow the pattern: for ambisonics order N, the number of ACN channels is `(N+1)^2`. The mixing matrix maps from `(N+1)^2` ACN channels (+ optional 2 non-diegetic) to the same number of streams. The last 2 rows/columns in each matrix handle the non-diegetic stereo pair as pass-through (coefficient = 32767 ≈ 1.0).

### 8.5 Encoder Constants

```c
#define MAX_OVERLAP 120       // (240 with ENABLE_QEXT)
#define MS_FRAME_TMP (6*1275+12)  // Max repacketizer temp buffer (6*QEXT_PACKET_SIZE_CAP+12 with QEXT)
```

## 9. Edge Cases

### 9.1 Error Conditions

- `channels < 1` or `channels > 255`: `OPUS_BAD_ARG`
- `streams < 1` or `streams > 255 - coupled_streams`: `OPUS_BAD_ARG`
- `coupled_streams > streams` or `coupled_streams < 0`: `OPUS_BAD_ARG`
- `streams + coupled_streams > channels` (encoder only): `OPUS_BAD_ARG`
- Invalid mapping (mapping entry >= `nb_streams + nb_coupled_streams` and != 255): `OPUS_BAD_ARG`
- Encoder layout validation failure (every stream must have its expected channels): `OPUS_BAD_ARG`
- `len < 2*nb_streams - 1` for non-PLC decode: `OPUS_INVALID_PACKET`
- Inconsistent frame counts across sub-packets: `OPUS_INVALID_PACKET`
- `max_data_bytes < smallest_packet`: `OPUS_BUFFER_TOO_SMALL`
- Decoded frame size > requested frame_size: `OPUS_BUFFER_TOO_SMALL`
- `frame_size <= 0`: `OPUS_BAD_ARG`
- Matrix dimensions too large (`rows > 255 || cols > 255 || rows*cols*2 > 65004`): returns 0 from `get_size`
- Demixing matrix size mismatch: `OPUS_BAD_ARG`

### 9.2 PLC (Packet Loss Concealment)

When `len == 0`, the decoder enters PLC mode:
- `do_plc = 1` is set
- Packet validation is skipped
- Each sub-decoder is called with the data pointer unchanged (relies on each sub-decoder's own PLC logic)
- `data` and `len` are not advanced between streams

### 9.3 Muted Channels

Channels mapped to 255 receive silence. The `copy_channel_out` callback is called with `src = NULL`, and each callback implementation writes zeros for the corresponding channel.

### 9.4 Ambisonics Channel Count Validation

Valid channel counts for ambisonics: `(N+1)^2` or `(N+1)^2 + 2` where `N = 0..14`. The `+2` accounts for optional non-diegetic stereo. Maximum: 15^2 + 2 = 227 channels.

### 9.5 CBR Last-Stream Bitrate Adjustment

In CBR mode, the last stream's bitrate is recalculated to fill the remaining bytes exactly:
```c
if (!vbr && s == st->layout.nb_streams-1)
    opus_encoder_ctl(enc, OPUS_SET_BITRATE(bits_to_bitrate(curr_max*8, Fs, frame_size)));
```

### 9.6 Final Range XOR

`OPUS_GET_FINAL_RANGE` returns the XOR of all sub-stream final ranges, providing a combined checksum for the entire multistream encode/decode.

### 9.7 NaN Protection (Float Surround Analysis)

```c
sum = celt_inner_prod(in, in, frame_size+overlap, 0);
if (!(sum < 1e18f) || celt_isnan(sum)) {
    OPUS_CLEAR(in, frame_size+overlap);
    preemph_mem[c] = 0;
}
```
The inverted comparison `!(sum < 1e18f)` catches both NaN and very large values.

## 10. Porting Notes

### 10.1 Variable-Length Structs (Critical)

The C implementation uses a single `opus_alloc()` with pointer arithmetic to lay out the header struct, sub-encoder/decoder states, and auxiliary buffers contiguously in memory. The `align()` macro ensures proper alignment.

**Rust approach**: Use a `Vec<u8>` backing buffer with manual offset tracking, or (preferred) use separate `Vec<OpusEncoder>` / `Vec<OpusDecoder>` fields in the Rust struct. The contiguous-allocation pattern is an optimization that doesn't need to be preserved -- Rust can use struct fields with proper types instead.

### 10.2 Function Pointer Callbacks

The `copy_channel_in` / `copy_channel_out` callbacks with `void*` user_data are a C pattern for generic programming. In Rust, use:
- A trait with methods for float/int16/int24 variants, or
- Generic functions with `impl Fn` parameters, or
- An enum dispatch

The projection layer passes the mixing/demixing matrix through `user_data` -- in Rust this becomes a typed parameter.

### 10.3 opus_encoder_init(NULL, ...) for Size Query

The C code calls `opus_encoder_init(NULL, Fs, channels, application)` to get the required allocation size without actually initializing. This dual-purpose function is unusual. In Rust, separate this into an explicit `get_size()` function.

### 10.4 Variadic ctl Functions

The `_ctl()` functions use C variadic arguments (`va_list`). In Rust:
- Use an enum for request types with typed payloads
- Or use separate typed getter/setter methods
- The `_ctl_va_list` variants exist to enable forwarding from projection to multistream without re-extracting varargs

### 10.5 ALLOC_STACK / VARDECL Macros

Stack allocation via `alloca()` (or `malloc()` fallback). In Rust, use `Vec` for dynamic-sized buffers. For performance-critical paths, consider `smallvec` or stack-allocated arrays with const generic size bounds.

### 10.6 Column-Major Matrix Storage

The `MATRIX_INDEX(nb_rows, row, col) = nb_rows * col + row` macro implements **column-major** ordering (Fortran-style). Ensure the Rust implementation consistently uses column-major indexing. This is the opposite of Rust's typical row-major convention.

### 10.7 Demixing Matrix Byte-Swap

The projection decoder receives the demixing matrix as raw bytes in little-endian format and converts to `opus_int16`:
```c
int s = demixing_matrix[2*i + 1] << 8 | demixing_matrix[2*i];
s = ((s & 0xFFFF) ^ 0x8000) - 0x8000;  // Sign-extend from 16-bit
```
In Rust, use `i16::from_le_bytes()`.

### 10.8 Output Accumulation in Decode-Side Matrix Multiply

`multiply_channel_out` uses `+=` to accumulate into the output buffer. The caller must zero the buffer first. The projection decoder does this on `dst_channel == 0`. This accumulation pattern requires mutable aliased access to the output buffer -- straightforward in C but needs care in Rust (use index-based access, not overlapping slices).

### 10.9 Conditional Compilation

Key `#ifdef` variants:
- `FIXED_POINT` / float: Affects all arithmetic in matrix multiply and surround analysis
- `ENABLE_RES24`: Changes internal sample resolution to 24-bit
- `ENABLE_QEXT`: Increases MAX_OVERLAP (240 vs 120) and packet size cap
- `DISABLE_FLOAT_API`: Omits float encode/decode variants

Use Rust `cfg` features for these. The `opus_res` type varies between `opus_int16` (fixed-point), `float` (float), or `opus_int32` (24-bit res).

### 10.10 Surround Analysis Dependencies

`surround_analysis()` pulls in substantial CELT machinery (MDCT, band energies, pre-emphasis, pitch inner product). This is the **only place** the multistream encoder directly calls CELT internals -- all other stream encoding goes through `opus_encode_native()`. Plan to have this function available before implementing surround mode.

### 10.11 Self-Delimiting Packet Format

The multistream packet concatenates sub-packets where all but the last use Opus self-delimiting framing. The repacketizer handles this. The length prefix is 1 byte if `<= 253`, or 2 bytes (`0xFC + overflow`) otherwise. This is handled by `opus_repacketizer_out_range_impl()`.

### 10.12 Integer Overflow in Rate Allocation

```c
channel_rate = 256*(opus_int64)(bitrate - ...) / total;
```
The cast to `opus_int64` prevents overflow. In Rust, use `i64` arithmetic for the intermediate calculation.

### 10.13 `align()` Macro

Used pervasively for struct/buffer alignment. Defined elsewhere in the codebase (likely as rounding up to pointer-size alignment). Rust's allocator handles alignment automatically for typed data, but the manual layout computation in `get_size()` functions must match.
