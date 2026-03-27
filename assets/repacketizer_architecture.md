Now I have all the necessary context. Here is the complete architecture document:

---

# Repacketizer & Extensions Module — Architecture Document

## 1. Purpose

The Repacketizer and Extensions modules provide Opus packet manipulation at the transport layer, **after encoding and before decoding**. They operate on opaque byte-level packet framing and never touch compressed audio data.

### Repacketizer (`repacketizer.c`)
- **Merge** multiple Opus packets (sharing the same configuration) into a single packet containing multiple frames.
- **Split** a multi-frame Opus packet into individual single-frame packets.
- **Pad** a packet to a target size (useful for constant-bitrate transport channels).
- **Unpad** a packet, removing all padding bytes to minimize size.
- **Multistream variants** of pad/unpad that operate on concatenated self-delimited streams.

### Extensions (`extensions.c`)
- **Parse** extension data embedded in the padding region of Code 3 Opus packets.
- **Generate** extension payloads for embedding into packet padding.
- **Iterate** over extensions with a stateful iterator that handles the "Repeat These Extensions" mechanism.
- Extensions carry per-frame metadata (e.g., DRED redundancy data, LBRR frames) in-band, introduced in Opus 1.4+.

---

## 2. Public API

### 2.1 Repacketizer Core

| Function | Signature | Description |
|----------|-----------|-------------|
| `opus_repacketizer_get_size` | `int (void)` | Returns `sizeof(OpusRepacketizer)`. For caller-allocated instances. |
| `opus_repacketizer_create` | `OpusRepacketizer* (void)` | Heap-allocates and initializes a repacketizer. Returns `NULL` on OOM. |
| `opus_repacketizer_init` | `OpusRepacketizer* (OpusRepacketizer *rp)` | (Re)initializes state: sets `nb_frames = 0`. Returns the same pointer. |
| `opus_repacketizer_destroy` | `void (OpusRepacketizer *rp)` | Frees heap-allocated repacketizer. |
| `opus_repacketizer_cat` | `int (OpusRepacketizer *rp, const unsigned char *data, opus_int32 len)` | Adds frames from a packet to the repacketizer. Returns `OPUS_OK` or `OPUS_INVALID_PACKET`. |
| `opus_repacketizer_get_nb_frames` | `int (OpusRepacketizer *rp)` | Returns number of frames currently held. |
| `opus_repacketizer_out_range` | `opus_int32 (OpusRepacketizer *rp, int begin, int end, unsigned char *data, opus_int32 maxlen)` | Emits frames `[begin, end)` as a new packet. Returns byte count or error. |
| `opus_repacketizer_out` | `opus_int32 (OpusRepacketizer *rp, unsigned char *data, opus_int32 maxlen)` | Convenience: emits all frames. Equivalent to `out_range(rp, 0, nb_frames, ...)`. |

### 2.2 Pad/Unpad (Single Stream)

| Function | Signature | Description |
|----------|-----------|-------------|
| `opus_packet_pad` | `int (unsigned char *data, opus_int32 len, opus_int32 new_len)` | Pads in-place to `new_len`. Returns `OPUS_OK` or error. |
| `opus_packet_unpad` | `opus_int32 (unsigned char *data, opus_int32 len)` | Strips all padding in-place. Returns new length or error. |

### 2.3 Pad/Unpad (Multistream)

| Function | Signature | Description |
|----------|-----------|-------------|
| `opus_multistream_packet_pad` | `int (unsigned char *data, opus_int32 len, opus_int32 new_len, int nb_streams)` | Pads the *last* stream in a multistream packet. |
| `opus_multistream_packet_unpad` | `opus_int32 (unsigned char *data, opus_int32 len, int nb_streams)` | Unpads *all* streams in a multistream packet. |

### 2.4 Internal Repacketizer Functions (not public API, but called across modules)

| Function | Signature | Description |
|----------|-----------|-------------|
| `opus_repacketizer_cat_impl` | `int (..., int self_delimited)` | Internal cat with self-delimited flag. |
| `opus_repacketizer_out_range_impl` | `opus_int32 (..., int self_delimited, int pad, const opus_extension_data *extensions, int nb_extensions)` | Full-featured output: self-delimited framing, optional padding, extension injection. |
| `opus_packet_pad_impl` | `opus_int32 (..., int pad, const opus_extension_data *extensions, int nb_extensions)` | Internal pad with extension support. |

### 2.5 Extension Iterator API

| Function | Signature | Description |
|----------|-----------|-------------|
| `opus_extension_iterator_init` | `void (OpusExtensionIterator *iter, const unsigned char *data, opus_int32 len, opus_int32 nb_frames)` | Initialize iterator over extension payload bytes. |
| `opus_extension_iterator_reset` | `void (OpusExtensionIterator *iter)` | Reset to beginning without reallocating. |
| `opus_extension_iterator_set_frame_max` | `void (OpusExtensionIterator *iter, int frame_max)` | Limit iteration to frames `[0, frame_max)`. Allows early termination. |
| `opus_extension_iterator_next` | `int (OpusExtensionIterator *iter, opus_extension_data *ext)` | Returns 1 and fills `ext` if an extension is found. Returns 0 at end. Returns `OPUS_INVALID_PACKET` on error. `ext` may be `NULL` (for counting). |
| `opus_extension_iterator_find` | `int (OpusExtensionIterator *iter, opus_extension_data *ext, int id)` | Seeks forward to the next extension with `ext->id == id`. |

### 2.6 Extension Convenience Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `opus_packet_extensions_count` | `opus_int32 (const unsigned char *data, opus_int32 len, int nb_frames)` | Count total extensions (excluding padding/separators/repeat markers). |
| `opus_packet_extensions_count_ext` | `opus_int32 (..., opus_int32 *nb_frame_exts, int nb_frames)` | Count extensions per-frame into array. |
| `opus_packet_extensions_parse` | `opus_int32 (..., opus_extension_data *extensions, opus_int32 *nb_extensions, int nb_frames)` | Parse all extensions into array in **bitstream order**. |
| `opus_packet_extensions_parse_ext` | `opus_int32 (..., const opus_int32 *nb_frame_exts, int nb_frames)` | Parse extensions into array in **frame order** (requires pre-computed counts from `_count_ext`). |
| `opus_packet_extensions_generate` | `opus_int32 (unsigned char *data, opus_int32 len, const opus_extension_data *extensions, opus_int32 nb_extensions, int nb_frames, int pad)` | Serialize extensions into bytes. `data` may be `NULL` for dry-run (size query). |

---

## 3. Internal State

### 3.1 `OpusRepacketizer` (defined in `opus_private.h:39-48`)

```c
struct OpusRepacketizer {
   unsigned char toc;                    // TOC byte of the first packet (bits 7..2 = config)
   int nb_frames;                        // Number of frames currently stored (0..48)
   const unsigned char *frames[48];      // Pointers into original packet data (borrowed)
   opus_int16 len[48];                   // Length of each frame in bytes
   int framesize;                        // Samples per frame at 8 kHz (from first packet's TOC)
   const unsigned char *paddings[48];    // Pointer to padding data for each "cat" call
   opus_int32 padding_len[48];           // Length of padding for each "cat" call
   unsigned char padding_nb_frames[48];  // Number of frames in the packet that contributed this padding
};
```

**Key design notes:**
- `frames[]` and `paddings[]` are **borrowed pointers** into caller-owned buffers. The caller must keep the original packet data alive until `init()` is called.
- Maximum 48 frames, enforced by the Opus spec (120 ms at 2.5 ms frame size = 48 frames).
- `framesize` is computed at 8 kHz sample rate (so 120 ms = 960 samples is the hard limit, checked at line 81).
- Padding info is stored per-cat-call, not per-frame. When a multi-frame packet is cat'd, only the first frame's slot gets the padding pointer; subsequent frames get `padding_len = 0` and `paddings = NULL`.

### 3.2 `OpusExtensionIterator` (defined in `opus_private.h:50-66`)

```c
typedef struct OpusExtensionIterator {
   const unsigned char *data;        // Start of all extension data (immutable)
   const unsigned char *curr_data;   // Current read position
   const unsigned char *repeat_data; // Start of extensions being repeated
   const unsigned char *last_long;   // Points past the last long (id>=32) extension payload
   const unsigned char *src_data;    // Source pointer for repeat iteration
   opus_int32 len;                   // Total length of extension data
   opus_int32 curr_len;              // Remaining length from curr_data
   opus_int32 repeat_len;            // Length of repeated extension region
   opus_int32 src_len;               // Remaining source length during repeat
   opus_int32 trailing_short_len;    // Bytes of short extension payloads after last long ext
   int nb_frames;                    // Total frames in the packet
   int frame_max;                    // Early termination frame limit
   int curr_frame;                   // Current frame index being iterated
   int repeat_frame;                 // Current frame in repeat iteration (0 = not repeating)
   unsigned char repeat_l;           // L flag of the repeat indicator
} OpusExtensionIterator;
```

**State machine:** The iterator has two modes:
1. **Normal mode** (`repeat_frame == 0`): scans forward through the byte stream, yielding extensions with `id > 2`.
2. **Repeat mode** (`repeat_frame > 0`): replays a saved region of extension bytes for each subsequent frame, advancing `curr_data` through the repeated payload bytes.

### 3.3 `opus_extension_data` (defined in `opus_private.h:68-73`)

```c
typedef struct {
   int id;                       // Extension ID (3..127)
   int frame;                    // Frame index this extension applies to
   const unsigned char *data;    // Pointer to payload bytes (borrowed)
   opus_int32 len;               // Payload length in bytes
} opus_extension_data;
```

---

## 4. Algorithm

### 4.1 Opus Packet Framing (Background)

Every Opus packet starts with a 1-byte **TOC** (Table of Contents):
- Bits 7..2: Configuration (mode, bandwidth, frame size) — the **top 6 bits** must match across merged packets.
- Bits 1..0: Frame count code:
  - `0`: 1 frame
  - `1`: 2 frames, CBR (equal sizes)
  - `2`: 2 frames, VBR (first size encoded explicitly)
  - `3`: Arbitrary number of frames (1..48), with sub-header

Code 3 sub-header byte:
- Bits 5..0: Frame count (M)
- Bit 6: Padding flag (P)
- Bit 7: VBR flag (V)

If P=1, padding length follows as a sequence of bytes: repeated `255` bytes (each contributing 254 to the total) followed by a terminating byte `< 255` (contributing its value).

Frame sizes use `encode_size`/`parse_size`: values < 252 take 1 byte; values >= 252 take 2 bytes: `byte0 = 252 + (size & 3)`, `byte1 = (size - byte0) >> 2`.

### 4.2 `opus_repacketizer_cat` (Adding Packets)

```
1. Validate len >= 1
2. If nb_frames == 0:
     Save TOC byte, compute framesize at 8 kHz
   Else:
     Check top 6 bits of TOC match (error if mismatch)
3. Get frame count of input packet via opus_packet_get_nb_frames()
4. Check (existing_frames + new_frames) * framesize <= 960 (120 ms limit)
5. Parse packet via opus_packet_parse_impl():
     - Extracts frame pointers, lengths into rp->frames[nb_frames..], rp->len[nb_frames..]
     - Extracts padding pointer/length into rp->paddings[nb_frames], rp->padding_len[nb_frames]
6. Store padding_nb_frames[nb_frames] = number of frames parsed
7. For each additional frame beyond the first:
     Clear padding info (only the first slot carries padding)
     Increment nb_frames
8. Increment nb_frames for the last frame
```

**Critical detail**: Frame pointers in `rp->frames[]` point into the **original packet buffer**. The repacketizer does not copy frame data.

### 4.3 `opus_repacketizer_out_range_impl` (Emitting Packets)

This is the core output function. It re-encodes the TOC and frame structure, selecting the most compact representation.

```
INPUT:  frames[begin..end), output buffer data[0..maxlen)
        optional: self_delimited, pad-to-maxlen, extension injection

1. Validate begin/end range
2. count = end - begin
3. Collect all extensions:
   a. Count extensions in padding of each cat'd packet via opus_packet_extensions_count()
   b. Allocate array for all_extensions (stack-allocated via VARDECL/ALLOC)
   c. Copy caller-provided extensions
   d. Parse padding extensions via opus_packet_extensions_parse(), renumbering frame indices

4. Choose packet code:
   - count == 1 → Code 0 (single frame)
   - count == 2 && equal sizes → Code 1 (2 CBR frames)
   - count == 2 && different sizes → Code 2 (2 VBR frames)
   - count > 2, OR padding requested, OR extensions present → Code 3

5. For Code 0:
   tot_size = 1 (TOC) + len[0] + self_delimited_overhead
   Write TOC with bottom 2 bits = 0: toc & 0xFC

6. For Code 1:
   tot_size = 1 + 2*len[0]
   Write TOC | 0x01

7. For Code 2:
   tot_size = 2 + len[0] + len[1] + (len[0]>=252 ? 1 : 0)
   Write TOC | 0x02, then encode_size(len[0])

8. For Code 3 (also used when pad or extensions force upgrade):
   a. Restart from beginning of buffer (ptr = data)
   b. Determine VBR vs CBR
   c. Calculate tot_size with frame count byte + per-frame sizes
   d. Write TOC | 0x03
   e. Write count byte: count | (vbr ? 0x80 : 0)
   f. If extensions present:
      - Dry-run opus_packet_extensions_generate(NULL, ...) to compute ext_len
      - Calculate pad_amount to hold extensions + lacing
   g. If padding needed:
      - Set bit 6 in count byte: data[1] |= 0x40
      - Encode padding length as sequence of 255s + remainder
      - Track ones_begin/ones_end for 0x01 fill region
   h. Write VBR frame sizes (encode_size for each frame except last)

9. Write self-delimited last-frame size if applicable

10. Copy frame data with OPUS_MOVE (memmove — supports in-place pad/unpad)

11. Generate extension bytes at ext_begin position

12. Fill 0x01 padding bytes in [ones_begin, ones_end)

13. Fill trailing zeros if pad && no extensions
```

**Buffer layout for Code 3 with padding and extensions:**
```
[TOC][count|flags][255...][remainder][VBR sizes...][frame data...][0x01 fill...][extension data...]
```

### 4.4 `opus_packet_pad` / `opus_packet_unpad`

**Pad** works by:
1. Copying the original packet to a temporary stack buffer
2. Cat'ing it into a fresh repacketizer
3. Outputting with `pad=1` and `maxlen=new_len`

The output will be Code 3 with the P bit set and padding bytes filling to `new_len`.

**Unpad** works by:
1. Cat'ing the packet into a fresh repacketizer
2. Zeroing all `padding_len` and `paddings` entries (discards all extensions too)
3. Outputting with `pad=0`, which selects the most compact code

### 4.5 Multistream Pad/Unpad

Multistream packets are concatenations of self-delimited streams, except the last stream uses normal framing.

**Multistream pad**: Seeks to the last stream by parsing (and skipping) `nb_streams-1` self-delimited packets, then pads only the last stream.

**Multistream unpad**: Iterates all streams, parsing and re-emitting each with padding stripped, writing output compactly from the front of the buffer.

### 4.6 Extension Wire Format

Extensions live inside the padding region of a Code 3 packet. Each extension is encoded as:

```
[ID byte][payload...]
```

The **ID byte** encodes:
- Bits 7..1: Extension ID (0..127)
- Bit 0: L flag (length encoding mode)

**Extension categories by ID:**

| ID | Name | Behavior |
|----|------|----------|
| 0, L=0 | Padding byte | Ignored (1 byte of padding) |
| 0, L=1 | Padding end | Ignored |
| 1, L=0 | Frame separator (+1) | Advances frame index by 1 |
| 1, L=1 | Frame separator (+N) | Advances frame index by `data[1]` (next byte) |
| 2 | Repeat These Extensions | Repeats all preceding extensions for remaining frames |
| 3..31 | Short extensions | Payload is 0 bytes (L=0) or 1 byte (L=1) |
| 32..127 | Long extensions | Payload length encoded with lacing (L=1) or extends to end (L=0) |

**Long extension length encoding (L=1):**
Sequence of bytes: each `255` adds 255 to the total; first non-255 byte adds its value. The payload immediately follows.

**Long extension with L=0:**
The payload extends to the end of the remaining data (minus any trailing short extension bytes). This is used for the last extension in the stream to avoid encoding the length.

**Repeat mechanism (ID=2):**
- L=1: Repeats extensions for frames `curr_frame+1` through `nb_frames-1`, consuming one set of payload bytes per frame.
- L=0: Same but forces the last long extension in the last frame to use L=0 encoding (no explicit length).

### 4.7 Extension Generation (`opus_packet_extensions_generate`)

```
1. Pre-scan: for each frame, find min/max extension indices (frame_min_idx, frame_max_idx)
2. For each frame f:
   a. Determine which extensions can use the repeat mechanism:
      - An extension at frame f can be repeated if every subsequent frame
        has the same extension ID in the same position
   b. For each extension in frame f:
      - If frame changed: emit separator (ID=1, L=0 for +1, or ID=1 L=1 with increment byte)
      - Emit extension: write_extension(id, L, payload)
      - If this is the trigger point for repeats: emit repeat indicator (ID=2)
        and then emit all repeated payloads for frames f+1..nb_frames-1
3. If pad requested: prepend 0x01 padding bytes via OPUS_MOVE
```

The generation is O(nb_extensions) when extensions are in frame order, due to the `frame_min_idx`/`frame_max_idx` pre-scan.

**Dry-run mode**: When `data == NULL`, all payload-writing is skipped but positions are tracked, yielding the exact byte count needed. Used by `out_range_impl` to pre-compute `ext_len`.

---

## 5. Data Flow

### 5.1 Repacketizer Data Flow

```
Caller's packet buffers (must remain valid)
        │
        ▼
  opus_repacketizer_cat()  ──►  OpusRepacketizer
        │                         │ .frames[] ──► points into caller buffers
        │                         │ .len[]
        │                         │ .paddings[] ──► points into caller buffers
        │                         │ .padding_len[]
        ▼                         │
  opus_repacketizer_out_range()   │
        │                         │
        ▼                         ▼
  Caller's output buffer ◄── frame data copied (OPUS_MOVE)
                          ◄── new TOC + framing headers written
                          ◄── extensions serialized
                          ◄── padding bytes filled
```

### 5.2 Extension Round-Trip

```
opus_extension_data[]          extensions.c              padding bytes
   (in-memory structs)    ────────────────────►     (in packet padding)
        ▲                  extensions_generate()          │
        │                                                 │
        │                  extensions_parse()              │
        └──────────────── ◄───────────────────────────────┘
```

### 5.3 In-Place Pad/Unpad Buffer Strategy

`opus_packet_pad` and `opus_packet_unpad` modify the packet **in-place**. The pad function allocates a stack copy of the original, then writes the padded version back to the original buffer. The `OPUS_MOVE` (memmove) in `out_range_impl` handles the overlap. `opus_packet_unpad` writes directly back, and since unpadded output is always shorter, this is safe.

---

## 6. Numerical Details

This module operates entirely on **byte-level packet framing**. There are no floating-point or fixed-point audio computations.

### 6.1 Size Encoding (`encode_size` / `parse_size`)

The Opus frame size encoding uses a compact variable-length format:

```c
// Encode: size → 1 or 2 bytes
int encode_size(int size, unsigned char *data) {
    if (size < 252) {
        data[0] = size;               // 1 byte: [0, 251]
        return 1;
    } else {
        data[0] = 252 + (size & 0x3); // low 2 bits in [252, 255]
        data[1] = (size - data[0]) >> 2;  // remaining bits
        return 2;
    }
}

// Decode: 1 or 2 bytes → size
// data[0] < 252: size = data[0]
// data[0] >= 252: size = 4*data[1] + data[0]
// Range: [0, 1275] (max frame = 1275 bytes per Opus spec)
```

Maximum representable value: `255 + 4*255 = 1275`, which matches the Opus maximum frame size.

### 6.2 Padding Length Encoding

Code 3 padding length uses 255-based lacing:
```
total = sum of all bytes, but each 255 byte contributes only 254
final byte < 255 contributes its full value
minimum padding = 0 (single byte with value 0)
padding_bytes = (pad_amount - 1) / 255 full 255-bytes + 1 terminator
terminator value = pad_amount - 255 * nb_255s - 1
```

### 6.3 Extension Lacing (Long Extensions)

Long extension payload lengths use 255-based lacing:
```
length = sum of lacing bytes
each 255 byte adds 255 to length (unlike packet padding!)
final byte < 255 terminates
```

Note the asymmetry: padding lacing has each 255 contribute 254 (so the terminator contributes up to 254). Extension lacing has each 255 contribute exactly 255.

### 6.4 Frame Duration Limit

The maximum packet duration is 120 ms. At 8 kHz, that is 960 samples. The check `(curr_nb_frames + rp->nb_frames) * rp->framesize > 960` enforces this.

`rp->framesize` is obtained by calling `opus_packet_get_samples_per_frame(data, 8000)` — note the hard-coded 8 kHz reference rate.

### 6.5 Integer Widths

| Value | C Type | Range | Notes |
|-------|--------|-------|-------|
| Frame size | `opus_int16` | 0..1275 | Fits in i16 |
| Padding length | `opus_int32` | 0..~2^31 | Can be very large |
| Extension payload length | `opus_int32` | 0..~2^31 | Lacing allows large payloads |
| Frame count | `int` | 0..48 | |
| Extension ID | `int` | 0..127 | 7 bits |

---

## 7. Dependencies

### 7.1 What This Module Calls

| Module | Functions Used | Purpose |
|--------|---------------|---------|
| `opus.c` | `opus_packet_parse_impl`, `opus_packet_get_samples_per_frame`, `opus_packet_get_nb_frames`, `encode_size` | Packet parsing and frame size encoding |
| `os_support.h` | `OPUS_COPY`, `OPUS_MOVE`, `OPUS_CLEAR` | memcpy/memmove/memset wrappers |
| `stack_alloc.h` | `VARDECL`, `ALLOC`, `SAVE_STACK`, `RESTORE_STACK`, `ALLOC_STACK` | Stack allocation for VLAs |
| `arch.h` | `IMIN`, `IMAX`, `celt_assert` | Integer min/max, assertions |
| `opus_defines.h` | `OPUS_OK`, `OPUS_BAD_ARG`, etc. | Error codes |

### 7.2 What Calls This Module

| Caller | Functions Called | Purpose |
|--------|----------------|---------|
| Opus encoder (`opus_encoder.c`) | `opus_repacketizer_*`, `opus_packet_pad_impl` | Assembling multi-frame output packets, injecting extensions |
| Opus decoder (`opus_decoder.c`) | Extension iterator | Extracting DRED, OSCE metadata from packets |
| Multistream encoder/decoder | `opus_multistream_packet_pad/unpad` | Transport-layer padding |
| Application code | All public API | Packet manipulation |

---

## 8. Constants and Tables

### 8.1 Hard Constants

| Constant | Value | Derivation |
|----------|-------|-----------|
| Max frames per packet | **48** | 120 ms / 2.5 ms (smallest Opus frame) |
| Max frame size (bytes) | **1275** | Opus spec maximum |
| Max packet duration (samples at 8 kHz) | **960** | 120 ms × 8000 Hz |
| Size encoding threshold | **252** | First 4 values ≥ 252 encode {252,253,254,255} as the low byte of a 2-byte size |
| TOC config mask | **0xFC** | Top 6 bits: coding mode + bandwidth + frame size |
| Code 3 VBR flag | **0x80** | Bit 7 of the count byte |
| Code 3 padding flag | **0x40** | Bit 6 of the count byte |
| Code 3 count mask | **0x3F** | Bits 5..0 of the count byte |
| Padding fill byte | **0x01** | Extension ID 0 with L=1 (padding end marker) — safe filler |
| Extension ID: padding (L=0) | **0** | `id_byte = 0x00` — 1 byte of padding |
| Extension ID: padding end (L=1) | **0** | `id_byte = 0x01` — used as fill |
| Extension ID: separator +1 (L=0) | **1** | `id_byte = 0x02` — advance frame by 1 |
| Extension ID: separator +N (L=1) | **1** | `id_byte = 0x03` — advance frame by next byte |
| Extension ID: repeat (L=0) | **2** | `id_byte = 0x04` |
| Extension ID: repeat (L=1) | **2** | `id_byte = 0x05` |
| Min valid extension ID | **3** | IDs 0..2 are reserved for padding/separator/repeat |
| Max valid extension ID | **127** | 7-bit field (bits 7..1 of ID byte) |
| Short extension threshold | **32** | IDs 3..31 are "short" (0 or 1 byte payload) |

### 8.2 No Static Tables

Neither module uses any static lookup tables, weight arrays, or precomputed data. All constants are inline.

---

## 9. Edge Cases

### 9.1 Repacketizer

1. **Empty packet** (`len < 1`): Returns `OPUS_INVALID_PACKET` from `cat`, `OPUS_BAD_ARG` from pad/unpad.

2. **TOC mismatch**: The top 6 bits (`toc & 0xFC`) must match between all cat'd packets. Mismatch returns `OPUS_INVALID_PACKET`. The bottom 2 bits (frame count code) are ignored since they vary.

3. **120 ms limit exceeded**: `(existing + new) * framesize > 960` returns `OPUS_INVALID_PACKET`. Must `init()` and re-cat.

4. **Single frame output**: `out_range(rp, i, i+1, ...)` produces a Code 0 packet. The most compact representation.

5. **Code upgrade to 3**: Even if count ≤ 2, the output is forced to Code 3 when `pad=1` or `ext_count > 0`. The buffer is restarted from scratch (line 221: `ptr = data`).

6. **In-place operation**: `opus_packet_pad` and `opus_packet_unpad` operate on the same buffer. The pad path copies to a temp buffer first. The unpad path relies on output always being ≤ input. `OPUS_MOVE` (memmove) handles overlapping regions.

7. **Multistream unpad with invalid sub-packets**: Returns the error from `opus_packet_parse_impl` propagated upward.

8. **Empty output range**: `begin >= end` returns `OPUS_BAD_ARG`.

9. **All frames CBR in Code 3**: When all frames have equal size, the VBR flag is not set and no per-frame sizes are written.

10. **Extension parsing errors during output**: If `opus_packet_extensions_parse` fails on padding data, returns `OPUS_INTERNAL_ERROR` (this should not happen with valid packets that were successfully cat'd).

### 9.2 Extensions

1. **Zero-length data**: `opus_extension_iterator_init` with `len=0` is valid; iteration immediately returns 0.

2. **Frame index overflow**: If a separator would advance `curr_frame >= nb_frames`, returns `OPUS_INVALID_PACKET` and sets `curr_len = -1` (permanently errors the iterator).

3. **Zero-increment separator** (ID=1, L=1, value=0): This is a no-op — the `continue` at line 262 skips it.

4. **Repeat with no prior extensions**: If `repeat_data == curr_data` (no extensions to repeat), the repeat loop completes immediately.

5. **Last long extension L=0 trick**: When repeat has L=0, the last long extension in the last frame is forced to L=0 encoding. This is handled at line 175-178 by clearing bit 0 of the repeat ID byte.

6. **Extension IDs 0..2 in repeat**: Padding (0) and separator (1) IDs are skipped during repeat (line 171: `if (repeat_id_byte <= 3) continue`).

7. **`ext` parameter is NULL**: Both `opus_extension_iterator_next` and the parse functions handle `ext == NULL` — used for counting only.

8. **Extension generate with `data == NULL`**: Dry-run mode. All `if (data)` guards skip writes but position tracking is accurate. Returns the total byte count.

9. **Frame ordering in parse**: `opus_packet_extensions_parse` returns extensions in **bitstream order** (not frame order). Due to the repeat mechanism, frame 3's extensions may appear between frame 0 and frame 1 extensions. Use `opus_packet_extensions_parse_ext` for frame order.

---

## 10. Porting Notes

### 10.1 Borrowed Pointers / Lifetimes

The `OpusRepacketizer` stores raw pointers (`frames[]`, `paddings[]`) into the **caller's packet buffers**. In Rust, this requires either:
- Lifetime annotations: `OpusRepacketizer<'a>` where `'a` ties to the input packet slices. This is the idiomatic approach.
- Alternatively, copy frame data on `cat()` to avoid lifetime coupling (at a performance cost).

The `opus_extension_data` struct similarly borrows into the packet buffer. The `OpusExtensionIterator` has multiple borrowed pointers (`data`, `curr_data`, `repeat_data`, `last_long`, `src_data`) — all pointing into the same buffer with different offsets.

**Recommendation**: Use `&'a [u8]` slices throughout. The iterator and extension data structs should all carry the same lifetime parameter.

### 10.2 Stack Allocation (VARDECL / ALLOC)

The C code uses `VARDECL`/`ALLOC` macros for VLAs on the stack. In `opus_repacketizer_out_range_impl`, `all_extensions` is stack-allocated:
```c
ALLOC(all_extensions, total_ext_count ? total_ext_count : ALLOC_NONE, opus_extension_data);
```

Rust options:
- Use `Vec<OpusExtensionData>` (heap). Acceptable — these allocations are small and infrequent.
- Use `smallvec` or `tinyvec` with a reasonable inline capacity (e.g., 48).
- Use a fixed-size array `[OpusExtensionData; 48]` if the max count is bounded.

Similarly, `opus_packet_pad_impl` stack-allocates `copy`:
```c
ALLOC(copy, len, unsigned char);
```
In Rust, use `Vec<u8>` for the temporary copy.

### 10.3 In-Place Mutation with `OPUS_MOVE`

`opus_repacketizer_out_range_impl` uses `OPUS_MOVE` (memmove) to copy frame data because the source and destination buffers may overlap during in-place pad/unpad operations (line 305):
```c
OPUS_MOVE(ptr, frames[i], len[i]);
```

In Rust, `slice::copy_within` handles overlapping copies within a single slice. The challenge is that the source pointers (`frames[i]`) point into the **same buffer** as the destination (`data`). This requires careful management:
- For the non-in-place case (normal `out_range`), `copy_from_slice` suffices.
- For the in-place case (`pad`/`unpad`), you need `ptr::copy` (equivalent to memmove) or work with a single mutable slice and use `copy_within`.

### 10.4 Multi-Return via Output Parameters

Many C functions return results through pointer parameters:
```c
opus_packet_parse_impl(data, len, ..., &rp->frames[nb_frames], &rp->len[nb_frames],
    NULL, NULL, &rp->paddings[nb_frames], &rp->padding_len[nb_frames]);
```

In Rust, use a result struct:
```rust
struct ParsedPacket<'a> {
    toc: u8,
    frames: &'a [&'a [u8]],  // or SmallVec
    padding: Option<&'a [u8]>,
}
```

### 10.5 Error Handling Pattern

The C code uses early returns with `RESTORE_STACK` (critical for the custom stack allocator). In Rust, this maps naturally to `Result<T, OpusError>` with `?` propagation. No explicit cleanup is needed thanks to RAII.

### 10.6 Extension Iterator State Machine

The `OpusExtensionIterator` is a complex state machine with two modes (normal / repeat). The `repeat_frame > 0` check determines the mode. Porting options:
- **Direct translation**: A Rust struct with the same fields. Most straightforward for bit-exact behavior.
- **Enum-based state**: More idiomatic Rust, but the C code's mode is just a single field check, so the direct translation is simpler.
- Consider implementing `Iterator` trait for ergonomic use.

### 10.7 The `data == NULL` Dry-Run Pattern

`opus_packet_extensions_generate` accepts `data == NULL` for a size-only query. Every write is guarded with `if (data)`. In Rust, this maps to:
- Accept `Option<&mut [u8]>` and guard writes with `if let Some(buf) = data`.
- Or have two functions: `extensions_generate_size()` and `extensions_generate()`.

### 10.8 Pointer Arithmetic as Indices

The C code frequently uses pointer subtraction for position tracking:
```c
celt_assert(iter->curr_data - iter->data == iter->len - iter->curr_len);
```

In Rust, replace pointer pairs with a single slice and an index:
```rust
struct ExtensionIterator<'a> {
    data: &'a [u8],
    pos: usize,        // replaces curr_data - data
    // ...
}
```

### 10.9 `ALLOC_NONE` Sentinel

When `total_ext_count == 0`, the C code uses `ALLOC_NONE` (which is 0 or 1 depending on the allocator) to avoid zero-size VLAs. In Rust, an empty `Vec` or zero-length slice handles this naturally.

### 10.10 `opus_multistream_packet_unpad` In-Place Compaction

This function reads from `data` and writes to `dst`, where both start at the same address but `dst` advances more slowly (unpadded output is smaller). This is safe because the read pointer is always ahead of or equal to the write pointer. In Rust, this requires working with raw indices into a single `&mut [u8]` slice, using `copy_within` for each stream's output.

### 10.11 Conditional Compilation

There is no conditional compilation in these files (no `#ifdef FIXED_POINT`, no SIMD). The code is identical for fixed-point and floating-point builds. This simplifies porting.

### 10.12 Function Visibility

| Function | C Visibility | Rust Recommendation |
|----------|-------------|---------------------|
| `opus_repacketizer_*` | `OPUS_EXPORT` (public) | `pub` |
| `opus_packet_pad/unpad` | `OPUS_EXPORT` (public) | `pub` |
| `opus_multistream_packet_pad/unpad` | `OPUS_EXPORT` (public) | `pub` |
| `opus_repacketizer_cat_impl` | `static` (file-private) | `pub(crate)` |
| `opus_repacketizer_out_range_impl` | declared in `opus_private.h` | `pub(crate)` |
| `opus_packet_pad_impl` | declared in `opus_private.h` | `pub(crate)` |
| `opus_extension_iterator_*` | declared in `opus_private.h` | `pub(crate)` |
| `opus_packet_extensions_*` | declared in `opus_private.h` | `pub(crate)` |
| `skip_extension`, `skip_extension_payload` | `static` | private (module-level) |
| `write_extension`, `write_extension_payload` | `static` | private (module-level) |
| `encode_size` | declared in `opus_private.h` | `pub(crate)` (shared with `opus.c`) |

### 10.13 Suggested Module Layout

```
src/
  repacketizer.rs       // OpusRepacketizer, cat, out_range, pad, unpad
  extensions.rs         // OpusExtensionIterator, parse, generate, extension_data
  packet.rs             // encode_size, parse_size, opus_packet_parse_impl (shared)
```
