Now I have everything needed. Here is the complete architecture document.

---

# Range Coder Module — Architecture Documentation

## Table of Contents

1. [Purpose](#1-purpose)
2. [Public API](#2-public-api)
3. [Internal State](#3-internal-state)
4. [Algorithm](#4-algorithm)
5. [Data Flow](#5-data-flow)
6. [Numerical Details](#6-numerical-details)
7. [Dependencies](#7-dependencies)
8. [Constants and Tables](#8-constants-and-tables)
9. [Edge Cases](#9-edge-cases)
10. [Porting Notes](#10-porting-notes)

---

## 1. Purpose

The range coder is the **entropy coding engine** at the heart of the Opus codec. Every coded decision in both the CELT and SILK layers—band energies, pitch parameters, codebook indices, sign bits, flag bits—passes through this module. It implements a variant of Martin's range coding algorithm (Martin 1979), which is functionally equivalent to arithmetic coding but operates on whole bytes rather than individual bits, making it faster in practice.

The module provides:

- **Encoder** (`ec_enc`): Accepts a stream of symbols with associated probability distributions and produces a compressed byte stream.
- **Decoder** (`ec_dec`): Reads the compressed byte stream and recovers the original symbol sequence.
- **Bit usage accounting** (`ec_tell`, `ec_tell_frac`): Reports how many bits have been consumed so far, which the codec uses for rate control decisions.
- **Raw bit I/O** (`ec_enc_bits`, `ec_dec_bits`): A secondary channel for uncoded bits that are packed from the **end** of the buffer, sharing the same byte buffer as the range-coded data.

The design is notable for its **bidirectional buffer**: range-coded bytes grow forward from the start of the buffer, while raw (uncoded) bits grow backward from the end. This eliminates the need for a separate side channel and enables single-pass encoding into a fixed-size packet.

### Academic references

- Pasco, R.C. "Source coding algorithms for fast data compression." PhD thesis, Stanford, 1976.
- Martin, G.N.N. "Range encoding: an algorithm for removing redundancy from a digitised message." Video & Data Recording Conference, 1979.
- Moffat, A., Neal, R., Witten, I.H. "Arithmetic Coding Revisited." ACM TOIS 16(3), 1998.

---

## 2. Public API

### 2.1 Shared (entcode.h)

| Function | Signature | Description |
|---|---|---|
| `ec_range_bytes` | `opus_uint32 ec_range_bytes(ec_ctx *_this)` | Returns `offs` — the number of range-coded bytes written/read so far. |
| `ec_get_buffer` | `unsigned char *ec_get_buffer(ec_ctx *_this)` | Returns the raw buffer pointer. |
| `ec_get_error` | `int ec_get_error(ec_ctx *_this)` | Returns nonzero if an error (overflow/underflow) occurred. |
| `ec_tell` | `int ec_tell(ec_ctx *_this)` | Returns bits used so far: `nbits_total - EC_ILOG(rng)`. Rounds up (conservative). Works identically on encoder and decoder. |
| `ec_tell_frac` | `opus_uint32 ec_tell_frac(ec_ctx *_this)` | Returns bits used, scaled by `2^BITRES` (i.e., in 1/8-bit units). Uses a lookup-table-accelerated approximation. |
| `celt_udiv` | `opus_uint32 celt_udiv(opus_uint32 n, opus_uint32 d)` | Unsigned integer division. On ARM, uses `SMALL_DIV_TABLE` to avoid hardware divide. On other platforms, compiles to `n/d`. |
| `celt_sudiv` | `opus_int32 celt_sudiv(opus_int32 n, opus_int32 d)` | Signed version of `celt_udiv`. |

### 2.2 Encoder (entenc.h)

| Function | Signature | Description |
|---|---|---|
| `ec_enc_init` | `void ec_enc_init(ec_enc *_this, unsigned char *_buf, opus_uint32 _size)` | Initializes encoder with external buffer `_buf` of `_size` bytes. |
| `ec_encode` | `void ec_encode(ec_enc *_this, unsigned _fl, unsigned _fh, unsigned _ft)` | Encodes a symbol occupying CDF range `[_fl, _fh)` out of total `_ft`. General-purpose; uses integer division. |
| `ec_encode_bin` | `void ec_encode_bin(ec_enc *_this, unsigned _fl, unsigned _fh, unsigned _bits)` | Like `ec_encode` but with `_ft = 1 << _bits`. Replaces division with a shift. |
| `ec_enc_bit_logp` | `void ec_enc_bit_logp(ec_enc *_this, int _val, unsigned _logp)` | Encodes a binary symbol where P(1) = 1/(1 << _logp). No division needed. |
| `ec_enc_icdf` | `void ec_enc_icdf(ec_enc *_this, int _s, const unsigned char *_icdf, unsigned _ftb)` | Encodes symbol `_s` using an 8-bit inverse CDF table. `_ft = 1 << _ftb`. |
| `ec_enc_icdf16` | `void ec_enc_icdf16(ec_enc *_this, int _s, const opus_uint16 *_icdf, unsigned _ftb)` | Same as `ec_enc_icdf` but with 16-bit iCDF entries. |
| `ec_enc_uint` | `void ec_enc_uint(ec_enc *_this, opus_uint32 _fl, opus_uint32 _ft)` | Encodes a uniform integer in `[0, _ft)`. Splits into range-coded MSBs + raw LSBs when `_ft` needs more than `EC_UINT_BITS` (8) bits. |
| `ec_enc_bits` | `void ec_enc_bits(ec_enc *_this, opus_uint32 _fl, unsigned _ftb)` | Writes `_ftb` raw (uncoded) bits to the **end** of the buffer. 1 <= _ftb <= 25. |
| `ec_enc_patch_initial_bits` | `void ec_enc_patch_initial_bits(ec_enc *_this, unsigned _val, unsigned _nbits)` | Overwrites the first `_nbits` (max 8) of the stream after encoding. Used for packet header flags. |
| `ec_enc_shrink` | `void ec_enc_shrink(ec_enc *_this, opus_uint32 _size)` | Compacts the buffer to `_size` bytes by relocating end-of-buffer raw bits. |
| `ec_enc_done` | `void ec_enc_done(ec_enc *_this)` | Finalizes the stream: flushes remaining range state, carry buffer, and raw bits. Clears unused bytes to zero. |

### 2.3 Decoder (entdec.h)

| Function | Signature | Description |
|---|---|---|
| `ec_dec_init` | `void ec_dec_init(ec_dec *_this, unsigned char *_buf, opus_uint32 _storage)` | Initializes decoder from buffer. Reads initial bytes and normalizes. |
| `ec_decode` | `unsigned ec_decode(ec_dec *_this, unsigned _ft)` | Returns the cumulative frequency for the next symbol (in `[0, _ft)`). Must be followed by `ec_dec_update`. |
| `ec_decode_bin` | `unsigned ec_decode_bin(ec_dec *_this, unsigned _bits)` | Like `ec_decode` with `_ft = 1 << _bits`. Must be followed by `ec_dec_update`. |
| `ec_dec_update` | `void ec_dec_update(ec_dec *_this, unsigned _fl, unsigned _fh, unsigned _ft)` | Advances the decoder state after the caller has identified the symbol from the cumulative frequency returned by `ec_decode`. |
| `ec_dec_bit_logp` | `int ec_dec_bit_logp(ec_dec *_this, unsigned _logp)` | Decodes a binary symbol where P(1) = 1/(1 << _logp). Self-contained (no update call needed). |
| `ec_dec_icdf` | `int ec_dec_icdf(ec_dec *_this, const unsigned char *_icdf, unsigned _ftb)` | Decodes a symbol using an 8-bit inverse CDF table. Self-contained. |
| `ec_dec_icdf16` | `int ec_dec_icdf16(ec_dec *_this, const opus_uint16 *_icdf, unsigned _ftb)` | Same as `ec_dec_icdf` but with 16-bit iCDF entries. Self-contained. |
| `ec_dec_uint` | `opus_uint32 ec_dec_uint(ec_dec *_this, opus_uint32 _ft)` | Decodes a uniform integer in `[0, _ft)`. Self-contained. |
| `ec_dec_bits` | `opus_uint32 ec_dec_bits(ec_dec *_this, unsigned _ftb)` | Reads `_ftb` raw bits from the **end** of the buffer. 0 <= _ftb <= 25. |

### 2.4 Two-phase vs. self-contained decode operations

The API has two patterns:

1. **Two-phase** (`ec_decode` + `ec_dec_update`): The caller gets a cumulative frequency, looks up the symbol in its probability model, then calls `ec_dec_update` with the symbol's exact CDF bounds. Used when the probability model is external (e.g., Laplace coding in `laplace.c`).

2. **Self-contained** (`ec_dec_icdf`, `ec_dec_bit_logp`, `ec_dec_uint`, `ec_dec_bits`): The function performs both the decode and the state update internally. Used when the probability model is known at call time.

---

## 3. Internal State

### 3.1 The `ec_ctx` struct

Encoder and decoder share the same struct. The fields have **dual-purpose semantics** depending on the role:

```c
struct ec_ctx {
    unsigned char *buf;      // Shared byte buffer
    opus_uint32    storage;  // Total buffer capacity in bytes
    opus_uint32    end_offs; // Bytes consumed from the END of the buffer (raw bits)
    ec_window      end_window; // Bit accumulator for raw bits at buffer end
    int            nend_bits;  // Number of valid bits in end_window
    int            nbits_total; // Total bits consumed (range-coded + raw)
    opus_uint32    offs;     // Byte offset for FRONT of buffer (range-coded data)
    opus_uint32    rng;      // Current interval width (range)
    opus_uint32    val;      // ENC: low end of interval; DEC: top - coded_value - 1
    opus_uint32    ext;      // ENC: outstanding carry count; DEC: saved divisor from ec_decode()
    int            rem;      // ENC: buffered byte awaiting carry; DEC: last byte read
    int            error;    // Nonzero if buffer overflow/underflow occurred
};
```

### 3.2 Field semantics by role

| Field | Encoder | Decoder |
|---|---|---|
| `val` | The low end of the current coding interval. Accumulates symbol contributions. | `top_of_range - coded_value - 1`. This inversion simplifies the decode comparison to `val < threshold`. |
| `ext` | Count of buffered "carry-propagating" symbols (bytes equal to `EC_SYM_MAX = 0xFF`). | The saved quotient `rng / ft` from `ec_decode()`, reused in `ec_dec_update()`. |
| `rem` | The most recently produced byte, held back until carry propagation is resolved. Initialized to `-1` (no byte yet). | The most recently read input byte, used across normalization iterations where a byte boundary falls mid-symbol. |

### 3.3 Lifecycle

**Encoder:**
1. `ec_enc_init()` — sets `rng = EC_CODE_TOP` (2^31), `val = 0`, `rem = -1`, `nbits_total = EC_CODE_BITS + 1 = 33`
2. Series of `ec_encode*` / `ec_enc_icdf*` / `ec_enc_bits` calls
3. `ec_enc_done()` — flushes remaining state, writes final bytes
4. `ec_get_buffer()` / `ec_range_bytes()` to retrieve the result

**Decoder:**
1. `ec_dec_init()` — reads initial bytes, sets `rng = 1 << EC_CODE_EXTRA` (2^7 = 128), normalizes to fill the register
2. Series of `ec_decode*` / `ec_dec_icdf*` / `ec_dec_bits` calls (must match encoder sequence exactly)
3. Check `ec_get_error()` and `ec_tell()` for validity

---

## 4. Algorithm

### 4.1 Core range coding principle

Range coding maintains an interval `[low, low + range)` within a notional code space of `[0, 2^EC_CODE_BITS)`. For each symbol with CDF range `[fl, fh)` out of total `ft`:

```
r = range / ft                  // quantized step size
low   += r * fl                 // (or equivalent, depending on the subinterval)
range  = r * (fh - fl)          // narrow the interval
```

When `range` drops below `EC_CODE_BOT` (2^23), normalization shifts out the most-significant byte and refills from the bottom.

### 4.2 Encoder: `ec_encode()`

```c
void ec_encode(ec_enc *_this, unsigned _fl, unsigned _fh, unsigned _ft) {
    opus_uint32 r;
    r = celt_udiv(_this->rng, _ft);        // Step 1: quantized step
    if (_fl > 0) {
        _this->val += _this->rng - IMUL32(r, (_ft - _fl));  // Step 2a: adjust low
        _this->rng = IMUL32(r, (_fh - _fl));                // Step 3a: new range
    } else {
        _this->rng -= IMUL32(r, (_ft - _fh));  // Step 2b/3b: combined (first symbol optimization)
    }
    ec_enc_normalize(_this);               // Step 4: emit bytes if range too small
}
```

**Why the special case for `_fl == 0`?** When encoding the first symbol in the alphabet, the low end doesn't move, so `val` is unchanged. The range computation is also slightly different to maximize precision: instead of `r * fh` (which might lose bits), it computes `rng - r * (ft - fh)`, keeping the residual from the division in the range rather than discarding it.

### 4.3 Encoder: `ec_encode_bin()`

Identical logic to `ec_encode`, but replaces `celt_udiv(_this->rng, _ft)` with `_this->rng >> _bits`. This is the hot path for power-of-two total frequencies.

### 4.4 Encoder: `ec_enc_bit_logp()`

A further specialization for binary symbols with power-of-two probabilities:

```c
void ec_enc_bit_logp(ec_enc *_this, int _val, unsigned _logp) {
    r = _this->rng;
    s = r >> _logp;          // P(1) = s/r ≈ 1/(1<<_logp)
    r -= s;                  // P(0) portion of range
    if (_val) _this->val += r;  // symbol=1: skip past the 0-region
    _this->rng = _val ? s : r;
    ec_enc_normalize(_this);
}
```

No division or multiplication at all — just a shift.

### 4.5 Encoder: Inverse CDF encoding (`ec_enc_icdf`)

The "inverse CDF" is a table where `icdf[s]` gives `ft - CDF(s+1)`, i.e., the survival function. The last entry is always 0. Values are monotonically non-increasing.

For symbol `s` with `ft = 1 << ftb`:
- If `s > 0`: The symbol's CDF range is `[ft - icdf[s-1], ft - icdf[s])`
- If `s == 0`: The symbol's CDF range is `[0, ft - icdf[0])`

```c
void ec_enc_icdf(ec_enc *_this, int _s, const unsigned char *_icdf, unsigned _ftb) {
    r = _this->rng >> _ftb;              // quantized step (shift, no division)
    if (_s > 0) {
        _this->val += _this->rng - IMUL32(r, _icdf[_s-1]);
        _this->rng = IMUL32(r, _icdf[_s-1] - _icdf[_s]);
    } else {
        _this->rng -= IMUL32(r, _icdf[_s]);
    }
    ec_enc_normalize(_this);
}
```

The inverse CDF form is used because it allows the decoder to search the table with a simple forward scan (see 4.8 below).

### 4.6 Encoder normalization and carry propagation

```c
static void ec_enc_normalize(ec_enc *_this) {
    while (_this->rng <= EC_CODE_BOT) {                     // range < 2^23
        ec_enc_carry_out(_this, (int)(_this->val >> EC_CODE_SHIFT)); // emit top byte
        _this->val = (_this->val << EC_SYM_BITS) & (EC_CODE_TOP - 1); // shift up
        _this->rng <<= EC_SYM_BITS;                          // widen range
        _this->nbits_total += EC_SYM_BITS;
    }
}
```

**Carry propagation** is the critical subtlety. When `val` overflows (a carry propagates from lower bits), it can ripple through previously output bytes. The encoder buffers the most recent byte in `rem` and counts consecutive `0xFF` bytes in `ext`. When a non-`0xFF` byte arrives:

```c
static void ec_enc_carry_out(ec_enc *_this, int _c) {
    if (_c != EC_SYM_MAX) {
        int carry = _c >> EC_SYM_BITS;      // carry bit (0 or 1)
        if (_this->rem >= 0)
            ec_write_byte(_this, _this->rem + carry);  // flush buffered byte + carry
        if (_this->ext > 0) {
            unsigned sym = (EC_SYM_MAX + carry) & EC_SYM_MAX;  // 0xFF+carry → 0x00, or 0xFF
            do ec_write_byte(_this, sym);
            while (--_this->ext > 0);
        }
        _this->rem = _c & EC_SYM_MAX;       // buffer this new byte
    } else {
        _this->ext++;                         // another 0xFF — just count it
    }
}
```

When carry = 1: the buffered `rem` byte increments by 1, and all the buffered `0xFF` bytes wrap to `0x00`. When carry = 0: everything flushes as-is.

### 4.7 Decoder: `ec_decode()` + `ec_dec_update()`

```c
unsigned ec_decode(ec_dec *_this, unsigned _ft) {
    _this->ext = celt_udiv(_this->rng, _ft);     // Save step size for ec_dec_update
    unsigned s = _this->val / _this->ext;         // Which step are we in?
    return _ft - EC_MINI(s + 1, _ft);             // Convert to cumulative frequency
}
```

The return value is the cumulative frequency `fl` such that the decoded symbol `s` satisfies `fl <= returned_value < fh`. The caller looks up the symbol, then:

```c
void ec_dec_update(ec_dec *_this, unsigned _fl, unsigned _fh, unsigned _ft) {
    opus_uint32 s = IMUL32(_this->ext, _ft - _fh);
    _this->val -= s;
    _this->rng = _fl > 0 ? IMUL32(_this->ext, _fh - _fl) : _this->rng - s;
    ec_dec_normalize(_this);
}
```

### 4.8 Decoder: Inverse CDF decoding (`ec_dec_icdf`)

```c
int ec_dec_icdf(ec_dec *_this, const unsigned char *_icdf, unsigned _ftb) {
    opus_uint32 s, d, r, t;
    int ret;
    s = _this->rng;                  // current range = initial "threshold"
    d = _this->val;                  // current coded value
    r = s >> _ftb;                   // quantized step
    ret = -1;
    do {
        t = s;                       // save previous threshold
        s = IMUL32(r, _icdf[++ret]); // next threshold = step * icdf[ret]
    } while (d < s);                 // coded value below threshold? keep searching
    _this->val = d - s;              // remove decoded contribution
    _this->rng = t - s;              // new range = gap between thresholds
    ec_dec_normalize(_this);
    return ret;
}
```

This is a **linear search** through the iCDF table. It works because:
- `icdf[]` is monotonically non-increasing, so the thresholds `r * icdf[i]` are monotonically non-increasing.
- The search finds the first `i` where `val >= r * icdf[i]`, meaning `val` falls in the interval `[r * icdf[i], r * icdf[i-1])` (or `[r * icdf[i], rng)` for `i == 0`).
- Because Opus alphabets are small (typically 2–16 symbols), linear search is faster than binary search.

### 4.9 Decoder normalization

```c
static void ec_dec_normalize(ec_dec *_this) {
    while (_this->rng <= EC_CODE_BOT) {
        int sym;
        _this->nbits_total += EC_SYM_BITS;
        _this->rng <<= EC_SYM_BITS;
        sym = _this->rem;                           // leftover bits from last byte
        _this->rem = ec_read_byte(_this);            // read next byte
        sym = (sym << EC_SYM_BITS | _this->rem) >> (EC_SYM_BITS - EC_CODE_EXTRA);
        _this->val = ((_this->val << EC_SYM_BITS) + (EC_SYM_MAX & ~sym)) & (EC_CODE_TOP - 1);
    }
}
```

The decoder's `val` is maintained as `top - coded_value - 1`. The `EC_SYM_MAX & ~sym` term is the complement of the input bits, which maintains this inverted representation. The `EC_CODE_EXTRA` shift handles the fact that the code register doesn't align exactly to byte boundaries (there are 7 "extra" bits — see Section 8).

### 4.10 Raw bit I/O

Raw bits bypass the range coder entirely and are packed from the **end** of the shared buffer, growing backward toward the front.

**Encoder** (`ec_enc_bits`): Accumulates bits in `end_window`. When the window fills up (more than `EC_WINDOW_SIZE` bits), bytes are flushed to the end of the buffer via `ec_write_byte_at_end()`.

**Decoder** (`ec_dec_bits`): Reads bytes from the end of the buffer via `ec_read_byte_from_end()` into `end_window`, then extracts the requested number of LSBs.

This mechanism is used for:
- The low-order bits in `ec_enc_uint` / `ec_dec_uint` (uniform integer coding)
- Spread bits, sign bits, and other uncoded data throughout the CELT layer

### 4.11 Encoder finalization (`ec_enc_done`)

1. Compute the minimum number of bits `l` needed so the interval `[val, val + rng)` is decodable regardless of trailing bits: `l = EC_CODE_BITS - EC_ILOG(rng)`.
2. Round `val` up to the nearest multiple of `2^(EC_CODE_BITS - 1 - l)`, producing `end`.
3. Verify that `[end, end | mask]` doesn't exceed `val + rng`; if it does, increment `l` (need one more bit).
4. Emit the top bytes of `end` through the carry propagation mechanism.
5. Flush any remaining byte in `rem` / `ext`.
6. Flush raw bits from `end_window` to the end of the buffer.
7. Zero out unused bytes between the range-coded data and the raw-bit data.
8. If there are leftover raw bits that didn't fill a complete byte, OR them into the last byte of the raw-bit region.

---

## 5. Data Flow

### 5.1 Buffer layout

```
+------------------------------------------------------------------+
| Range-coded bytes (forward) →  | ... unused ... | ← Raw bits     |
+------------------------------------------------------------------+
^                                ^                ^                ^
buf[0]                      buf[offs]    buf[storage-end_offs] buf[storage-1]
```

- `offs` grows forward from 0: each range-coded byte is written at `buf[offs++]`.
- `end_offs` grows backward from 0: each raw-bit byte is written at `buf[storage - (++end_offs)]`.
- The buffer overflows when `offs + end_offs >= storage`.

### 5.2 Encoding data flow

```
Symbol + probability model
        │
        ▼
 ┌──────────────┐      ┌──────────────────┐
 │  ec_encode*   │─────→│ ec_enc_normalize │
 │  ec_enc_icdf  │      │  (emit bytes)    │
 │  ec_enc_bit_  │      └────────┬─────────┘
 │    logp       │               │
 └──────────────┘               ▼
                          ┌─────────────────┐
                          │ ec_enc_carry_out │
                          │  (carry buffer)  │
                          └────────┬────────┘
                                   │
                                   ▼
                            ec_write_byte()
                            → buf[offs++]

Raw bits (uncoded)
        │
        ▼
 ┌──────────────┐
 │ ec_enc_bits  │───→ ec_write_byte_at_end()
 └──────────────┘     → buf[storage - (++end_offs)]
```

### 5.3 Decoding data flow

```
buf[offs++]  ──→  ec_read_byte()
                        │
                        ▼
                  ┌───────────────┐
                  │ec_dec_normalize│
                  └───────┬───────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
    ┌──────────────┐          ┌──────────────┐
    │  ec_decode   │          │ ec_dec_icdf  │
    │  (get CDF)   │          │ ec_dec_bit_  │
    └──────┬───────┘          │   logp       │
           │                  └──────┬───────┘
           ▼                         │
    ┌──────────────┐                 │
    │ec_dec_update │                 │
    └──────────────┘                 │
           │                         │
           ▼                         ▼
        Decoded symbol            Decoded symbol


buf[storage-(++end_offs)] ──→ ec_read_byte_from_end()
                                      │
                                      ▼
                               ┌─────────────┐
                               │ ec_dec_bits  │──→ Raw bits
                               └─────────────┘
```

---

## 6. Numerical Details

### 6.1 Integer widths and ranges

| Quantity | Type | Bit width | Range | Notes |
|---|---|---|---|---|
| `rng` | `opus_uint32` | 32 | `[EC_CODE_BOT+1, EC_CODE_TOP]` after normalization, i.e., `[2^23 + 1, 2^31]` | Invariant maintained by normalization |
| `val` | `opus_uint32` | 32 | `[0, EC_CODE_TOP - 1]` i.e., `[0, 2^31 - 1]` | Masked to 31 bits |
| `ext` | `opus_uint32` | 32 | ENC: unbounded count; DEC: `rng/ft` | |
| `ec_window` | `opus_uint32` | 32 | Full 32-bit range | `end_window` accumulator |
| Frequency params `fl, fh, ft` | `unsigned` (32-bit) | 32 | `ft` up to `2^32 - 1` for `ec_enc_uint`; typically `<= 2^15` for iCDF | |

### 6.2 Critical multiplication: `IMUL32`

```c
#define IMUL32(a,b) ((a)*(b))
```

This is a simple 32×32→32 multiply (lower 32 bits). It is safe because:
- `r = rng >> ftb` or `rng / ft`, so `r < rng / ft <= rng`
- The second operand is a frequency difference, bounded by `ft`
- Since `rng <= 2^31` and `ft <= 2^15` typically, the product fits in 32 bits

**Overflow risk**: If `ft` approaches `2^16` and `rng` is near `2^31`, the product `r * ft` could approach `2^31 * 2^16 / 2^16 * 2^16 = 2^31`, which still fits. The design ensures `r * ft <= rng` by construction.

For `ec_enc_uint` with large `_ft` (up to `2^32 - 1`), the function splits the value into an upper portion coded with the range coder and a lower portion coded as raw bits, keeping `ft` for the range coder at most `2^EC_UINT_BITS + 1 = 257`.

### 6.3 Division in the range coder

The only division in the hot path is `rng / ft` in `ec_encode` and `ec_decode`. `ec_encode_bin` and `ec_enc_icdf` (the most common paths) replace this with `rng >> bits`.

For the general case, `celt_udiv` on ARM uses a lookup table:

```c
// For d in [1, 256]:
t = EC_ILOG(d & -d);   // position of lowest set bit + 1
q = (uint64)SMALL_DIV_TABLE[d >> t] * (n >> (t-1)) >> 32;
return q + (n - q*d >= d);
```

This avoids hardware division instructions, which are slow on some ARM cores.

### 6.4 Bit usage measurement: `ec_tell` and `ec_tell_frac`

**`ec_tell`** returns `nbits_total - EC_ILOG(rng)`, which is the number of bits consumed so far. The `EC_ILOG(rng)` term accounts for the bits still "available" in the current range. This always rounds up — it reports the worst-case bit cost.

**`ec_tell_frac`** refines this to 1/8-bit precision using a lookup table:

```c
opus_uint32 ec_tell_frac(ec_ctx *_this) {
    static const unsigned correction[8] =
        {35733, 38967, 42495, 46340, 50535, 55109, 60097, 65535};
    opus_uint32 nbits = _this->nbits_total << BITRES;  // scale to 1/8 bits
    int l = EC_ILOG(_this->rng);
    opus_uint32 r = _this->rng >> (l - 16);    // normalize to [2^15, 2^16)
    unsigned b = (r >> 12) - 8;                 // initial estimate: top 4 bits
    b += r > correction[b];                     // one correction step
    l = (l << 3) + b;                           // combine integer and fractional parts
    return nbits - l;
}
```

The `correction[]` table contains threshold values for `r` at each 1/8-bit boundary. The entries are essentially `floor(2^16 * 2^(-k/8))` for `k = 0..7`, approximating `log2(rng)` to 1/8-bit precision.

The alternative (disabled) implementation uses iterative squaring:
```c
for (i = BITRES; i-- > 0;) {
    r = r * r >> 15;    // square and renormalize
    b = (int)(r >> 16); // extract one bit of log2
    l = l << 1 | b;
    r >>= b;
}
```
This computes `log2(rng)` one bit at a time through repeated squaring, producing `BITRES` fractional bits. The lookup-table version is faster.

### 6.5 Precision invariant

After normalization, the range is always in `(EC_CODE_BOT, EC_CODE_TOP]` = `(2^23, 2^31]`. This means there are always at least 23 bits of precision available for subdividing the range. Since frequency totals `ft` in the iCDF path are at most `2^15` (15-bit precision), the quantized step `r = rng >> ftb` is always at least `2^(23-15) = 256`, ensuring no symbol gets a zero-width subinterval.

### 6.6 Initial `nbits_total`

Both encoder and decoder initialize `nbits_total` to `EC_CODE_BITS + 1 = 33`. This accounts for the 32 "virtual" bits of state in the range register plus one extra bit of overhead. After initialization, `ec_tell()` returns `33 - EC_ILOG(EC_CODE_TOP) = 33 - 32 = 1`, reflecting that a freshly initialized coder has used "1 bit" — this worst-case accounting is intentional and documented.

The decoder adjusts for the bits consumed during initialization:
```c
_this->nbits_total = EC_CODE_BITS + 1
    - ((EC_CODE_BITS - EC_CODE_EXTRA) / EC_SYM_BITS) * EC_SYM_BITS;
// = 33 - ((32 - 7) / 8) * 8 = 33 - 24 = 9
```
After `ec_dec_normalize` adds `3 * EC_SYM_BITS = 24` (reading ~3 bytes), `nbits_total` reaches 33, matching the encoder.

---

## 7. Dependencies

### 7.1 What this module calls

| Dependency | Source | Used for |
|---|---|---|
| `opus_types.h` | `include/` | `opus_uint32`, `opus_int32`, `opus_uint16` type definitions |
| `opus_defines.h` | `include/` | `OPUS_INLINE` macro |
| `arch.h` | `celt/` | `IMUL32` macro, `celt_assert`, `celt_sig_assert` |
| `os_support.h` | `celt/` | `OPUS_MOVE` (memmove wrapper), `OPUS_CLEAR` (memset wrapper) |
| Standard C | — | `<limits.h>` (CHAR_BIT), `<stddef.h>`, `<math.h>` |

### 7.2 What calls this module

Essentially **everything** in the Opus codec that reads or writes coded bits:

| Caller | Uses |
|---|---|
| `celt/bands.c` | `ec_enc_bit_logp`, `ec_dec_bit_logp`, `ec_enc_icdf`, `ec_dec_icdf`, `ec_enc_bits`, `ec_dec_bits`, `ec_tell`, `ec_tell_frac` |
| `celt/quant_bands.c` | `ec_enc_icdf`, `ec_dec_icdf`, `ec_tell`, `ec_tell_frac`, `ec_encode`, `ec_decode`, `ec_dec_update` |
| `celt/cwrs.c` | `ec_encode`, `ec_decode`, `ec_dec_update`, `ec_enc_bits`, `ec_dec_bits`, `ec_tell` |
| `celt/celt_encoder.c` | `ec_enc_init`, `ec_enc_done`, `ec_enc_shrink`, `ec_enc_bit_logp`, `ec_enc_icdf`, `ec_enc_uint`, `ec_tell`, `ec_tell_frac` |
| `celt/celt_decoder.c` | `ec_dec_init`, `ec_dec_bit_logp`, `ec_dec_icdf`, `ec_dec_uint`, `ec_tell`, `ec_tell_frac` |
| `celt/laplace.c` | `ec_encode`, `ec_decode`, `ec_dec_update`, `ec_encode_bin`, `ec_decode_bin` |
| `celt/vq.c` | `ec_enc_icdf`, `ec_dec_icdf`, `ec_enc_bits`, `ec_dec_bits` |
| `silk/*.c` | All encoder/decoder functions extensively |
| `src/opus_encoder.c` | `ec_enc_init`, `ec_enc_done`, `ec_enc_shrink`, `ec_enc_patch_initial_bits`, `ec_tell` |
| `src/opus_decoder.c` | `ec_dec_init`, `ec_tell` |
| `src/repacketizer.c` | Direct buffer manipulation (not range coder calls) |
| `dnn/dred_encoder.c`, `dnn/dred_decoder.c` | `ec_enc_icdf`, `ec_dec_icdf` and related |

---

## 8. Constants and Tables

### 8.1 Constants from `mfrngcod.h`

| Constant | Value | Formula | Purpose |
|---|---|---|---|
| `EC_SYM_BITS` | 8 | — | Bits per output symbol (byte-oriented I/O) |
| `EC_CODE_BITS` | 32 | — | Total bits in the state registers (`val`, `rng`) |
| `EC_SYM_MAX` | 255 | `(1 << 8) - 1` | Maximum value of an output symbol |
| `EC_CODE_SHIFT` | 23 | `32 - 8 - 1` | Shift to extract top symbol from `val` |
| `EC_CODE_TOP` | 0x80000000 | `1 << 31` | Upper bound of the code range (bit 31) |
| `EC_CODE_BOT` | 0x00800000 | `EC_CODE_TOP >> 8` = `1 << 23` | Normalization threshold |
| `EC_CODE_EXTRA` | 7 | `(32 - 2) % 8 + 1` | Extra bits in the last partial symbol |

**Derivation of `EC_CODE_EXTRA`**: The 32-bit code space holds 31 usable bits (bit 31 is `EC_CODE_TOP` which acts as the carry sentinel). These 31 bits hold `floor(31/8) = 3` full bytes plus `31 - 24 = 7` extra bits. These 7 extra bits are why the decoder must handle a partial-byte boundary when reading the initial state.

**Relationship between constants:**
- `EC_CODE_TOP = 2^(EC_CODE_BITS - 1)` — one less than a full 32-bit range, leaving room for the carry bit
- `EC_CODE_BOT = EC_CODE_TOP >> EC_SYM_BITS` — normalization fires when `rng` drops below this
- `EC_CODE_SHIFT = EC_CODE_BITS - EC_SYM_BITS - 1` — extracts the top byte from `val` (below bit 31)

### 8.2 Constants from `entcode.h`

| Constant | Value | Purpose |
|---|---|---|
| `EC_UINT_BITS` | 8 | Max bits for the range-coded portion of `ec_enc_uint` |
| `BITRES` | 3 | Fractional bit resolution: 2^3 = 8, so 1/8-bit precision |
| `EC_WINDOW_SIZE` | 32 | `sizeof(ec_window) * CHAR_BIT` — width of the raw-bit accumulator |

### 8.3 `ec_tell_frac` correction table

```c
static const unsigned correction[8] =
    {35733, 38967, 42495, 46340, 50535, 55109, 60097, 65535};
```

These are the threshold values for the normalized range `r` (in `[2^15, 2^16)`) at each 1/8-bit boundary. Conceptually, `correction[k]` ≈ `floor(2^16 * 2^(-(k+1)/8))` (adjusted for the exact rounding behavior needed). The values enable a single comparison to refine the integer log2 estimate by one fractional bit.

Exact derivation: `correction[k] = round(2^16 * 2^(-(k+1)/8))` for k = 0..6, and `correction[7] = 65535 = 2^16 - 1`.

### 8.4 `SMALL_DIV_TABLE`

A 129-entry table used only on ARM (`USE_SMALL_DIV_TABLE`):

```c
const opus_uint32 SMALL_DIV_TABLE[129];  // SMALL_DIV_TABLE[i] = floor(2^32 / (2*i + 1))
```

Exception: `SMALL_DIV_TABLE[0] = 0xFFFFFFFF` (special case, since `2^32 / 1` overflows 32 bits).

Used by `celt_udiv()` to replace hardware division with multiply-and-shift for divisors up to 256. The technique: for divisor `d`, factor out the power-of-two component, look up the reciprocal of the odd part, multiply, and shift.

### 8.5 `EC_MINI` macro

```c
#define EC_MINI(_a, _b) ((_a) + (((_b) - (_a)) & -((_b) < (_a))))
```

Branchless minimum of two unsigned values. Returns `_b` if `_b < _a`, else `_a`. Used in `ec_decode` and `ec_decode_bin` to clamp the decoded frequency.

### 8.6 `EC_ILOG` macro / `ec_ilog` function

Returns `floor(log2(x)) + 1` for `x > 0`, i.e., the position of the highest set bit (1-indexed). Equivalent to `32 - clz(x)`. **Undefined for x = 0.**

Platform implementations:
- **MSVC**: `_BitScanReverse` intrinsic
- **GCC/Clang**: `__builtin_clz`
- **TI DSP**: `_lnorm`
- **Fallback**: Branchless binary search in `ec_ilog()` (entcode.c)

---

## 9. Edge Cases

### 9.1 Buffer overflow

Both `ec_write_byte` and `ec_write_byte_at_end` check `offs + end_offs >= storage` before writing. On failure, they return -1 and the error is OR'd into `_this->error`. The coder continues operating (it doesn't abort), but the output is corrupted. Callers should check `ec_get_error()` after encoding is done.

### 9.2 Decoder reading past the end

`ec_read_byte` returns **0** when `offs >= storage` (not an error signal). This means the decoder silently reads zero bytes when the stream is exhausted. The `error` flag is not set by reading past the end — the decoder relies on `ec_tell()` comparisons to detect overread.

`ec_read_byte_from_end` similarly returns 0 when `end_offs >= storage`.

### 9.3 `ec_dec_uint` out-of-range

If the decoded value exceeds `_ft - 1`, the function sets `error = 1` and returns the clamped value `_ft - 1`:

```c
if (t <= _ft) return t;
_this->error = 1;
return _ft;  // _ft here is already decremented (_ft - 1 of the original)
```

### 9.4 `ec_enc_patch_initial_bits` edge cases

Three states depending on how far encoding has progressed:
1. `offs > 0`: First byte already written to buffer — patch `buf[0]` directly.
2. `rem >= 0`: First byte buffered but not written — patch `rem`.
3. `rng <= EC_CODE_TOP >> _nbits`: Normalization hasn't run yet — patch bits inside `val`.
4. Otherwise: Fewer than `_nbits` bits have been encoded — set `error = -1`.

### 9.5 Empty stream / minimal encoding

A freshly initialized encoder has `ec_tell() = 1`. Even encoding zero symbols, `ec_enc_done()` will produce at least one byte (the finalization flush). This is intentional — the range coder always costs at least 1 bit of overhead.

### 9.6 The carry propagation counter overflow

The `ext` field counts consecutive 0xFF bytes. On 32-bit systems, this limits a single packet to ~4 billion carry-propagating symbols. The code does not check for this overflow — it's documented as a theoretical limit that is never reached in practice (Opus packets are at most ~1275 bytes).

### 9.7 `ec_enc_done` residual raw bits

When `ec_enc_done` is called, if there are leftover raw bits in `end_window` that don't fill a complete byte, they are OR'd into the last byte of the raw-bit region (`buf[storage - end_offs - 1] |= window`). This works because the unused bits in that byte are guaranteed to be zero (either from initialization or from `OPUS_CLEAR`).

If the range-coded data and raw-bit data collide (insufficient space), the raw bits are truncated and `error` is set to -1.

---

## 10. Porting Notes

### 10.1 Unified struct with dual semantics

The C code uses a single `ec_ctx` for both encoder and decoder, with `typedef ec_ctx ec_enc` and `typedef ec_ctx ec_dec`. Fields like `val`, `ext`, and `rem` have different meanings depending on the role.

**Rust approach**: Use a common inner struct for shared state and two wrapper types:

```rust
struct RangeCoderState {
    buf: Vec<u8>,  // or &mut [u8]
    storage: u32,
    end_offs: u32,
    end_window: u32,
    nend_bits: i32,
    nbits_total: i32,
    offs: u32,
    rng: u32,
    error: i32,
}

struct RangeEncoder {
    state: RangeCoderState,
    val: u32,   // low end of interval
    rem: i32,   // buffered carry byte (-1 = none)
    ext: u32,   // carry propagation count
}

struct RangeDecoder {
    state: RangeCoderState,
    val: u32,   // top - coded_value - 1
    rem: i32,   // last byte read
    ext: u32,   // saved divisor from ec_decode()
}
```

Alternatively, keep a single struct with clear documentation, since `ec_tell()` and `ec_tell_frac()` operate on both — having a shared trait or common method is useful.

### 10.2 Buffer ownership and lifetime

The C code takes an externally-owned `unsigned char *buf`. The encoder writes into it; the decoder reads from it.

**Rust options**:
- Encoder: take `&mut [u8]` (borrowed mutable slice) — no allocation, matches C semantics.
- Decoder: take `&[u8]` (borrowed immutable slice) — the decoder never writes to `buf`.
- Or use `Vec<u8>` for owned buffers if the Rust API prefers it.

The key constraint: the buffer is bidirectional (forward writes + backward writes), so it cannot be split into two independent slices.

### 10.3 The `rem = -1` sentinel

The encoder uses `rem = -1` (as a signed `int`) to indicate "no byte buffered yet." The first call to `ec_enc_carry_out` checks `rem >= 0` before writing.

**Rust approach**: Use `Option<u8>` instead of the sentinel value:
```rust
rem: Option<u8>,  // None = no byte buffered
```

### 10.4 Pointer arithmetic in `ec_write_byte_at_end`

```c
_this->buf[_this->storage - ++(_this->end_offs)] = (unsigned char)_value;
```

This pre-increments `end_offs` and indexes from the end. In Rust, the equivalent is straightforward:

```rust
self.state.end_offs += 1;
self.state.buf[(self.state.storage - self.state.end_offs) as usize] = value;
```

### 10.5 `IMUL32` — wrapping multiplication

`IMUL32(a, b)` is defined as `(a) * (b)`. In C, unsigned overflow wraps silently. In Rust, unsigned multiplication panics on overflow in debug mode.

**Rust approach**: Use `u32::wrapping_mul()` or verify that overflow cannot occur (it shouldn't, given the invariants, but wrapping_mul is defensive). Given that bit-exact matching is required, using `wrapping_mul` is safest.

### 10.6 `EC_ILOG` — count leading zeros

The C code uses compiler intrinsics (`__builtin_clz`, `_BitScanReverse`) with a fallback.

**Rust**: Use `u32::leading_zeros()` which maps to the appropriate hardware instruction. Then:
```rust
fn ec_ilog(v: u32) -> u32 {
    32 - v.leading_zeros()  // equivalent to EC_CLZ0 - EC_CLZ(v)
}
```

Note: `u32::leading_zeros()` is defined for 0 (returns 32), but `EC_ILOG(0)` is documented as undefined. The Rust implementation should maintain this precondition via debug assertions.

### 10.7 `ec_tell_frac` — the disabled alternative implementation

The `#if 1` / `#else` block in `entcode.c` means only the lookup-table version is compiled. The iterative version is dead code. Port only the lookup-table version.

### 10.8 Conditional compilation: `USE_SMALL_DIV_TABLE`

This is only enabled on ARM (`OPUS_ARM_ASM`). For the Rust port:
- The initial port should just use native division (the `#else` branch of `celt_udiv`).
- The `SMALL_DIV_TABLE` and reciprocal-multiply path can be added later as an ARM-specific optimization behind a `cfg(target_arch = "arm")` or similar.

### 10.9 `ec_enc_shrink` — `OPUS_MOVE`

```c
OPUS_MOVE(_this->buf + _size - _this->end_offs,
          _this->buf + _this->storage - _this->end_offs,
          _this->end_offs);
```

This is a `memmove` (overlapping allowed) that relocates the raw-bit bytes from the old end position to the new end position.

**Rust**: Use `slice::copy_within()` for an in-place move within the same buffer:
```rust
let src_start = (self.storage - self.end_offs) as usize;
let src_end = self.storage as usize;
let dst_start = (new_size - self.end_offs) as usize;
buf.copy_within(src_start..src_end, dst_start);
```

### 10.10 `OPUS_CLEAR` in `ec_enc_done`

```c
OPUS_CLEAR(_this->buf + _this->offs,
           _this->storage - _this->offs - _this->end_offs);
```

Zeros the gap between range-coded data and raw-bit data. In Rust:
```rust
let start = self.offs as usize;
let end = (self.storage - self.end_offs) as usize;
buf[start..end].fill(0);
```

### 10.11 Bit manipulation patterns

Several patterns need care in Rust:

1. **`d & -d`** (lowest set bit): Rust unsigned types don't support unary negation. Use `d & d.wrapping_neg()` or `1 << d.trailing_zeros()`.

2. **`!!v`** (boolean coercion to 0/1): Use `(v != 0) as u32` or `u32::from(v != 0)`.

3. **`(sym << EC_SYM_BITS | _this->rem) >> (EC_SYM_BITS - EC_CODE_EXTRA)`**: This combines two bytes with a non-byte-aligned shift. The types and widths must match exactly. Since `EC_CODE_EXTRA = 7` and `EC_SYM_BITS = 8`, this shifts a 16-bit quantity right by 1, extracting 15 bits. In Rust, use explicit `as u32` casts to match the C promotion rules.

4. **Signed/unsigned mixing**: `rem` is `int` but holds byte values (0–255) or `-1`. The comparison `rem >= 0` and the arithmetic `rem + carry` mix signed semantics. In Rust, use `Option<u8>` as described above.

### 10.12 Bit-exactness requirements

The range coder is the most critical module for bit-exact reproduction. Every division, multiplication, shift, and rounding decision must match the C reference **exactly**. Specifically:

- `celt_udiv(rng, ft)` must produce the identical quotient as C's `unsigned / unsigned` truncating division.
- `IMUL32(a, b)` must produce the identical lower-32-bit product.
- `EC_ILOG(x)` must return the identical value (position of highest set bit, 1-indexed).
- The byte emission order, carry propagation, and normalization loop counts must all match.

Rust's native unsigned arithmetic already satisfies these requirements (truncating division, wrapping or non-overflowing multiplication). The primary risk is accidental use of signed arithmetic where unsigned is needed, or debug-mode overflow panics.

### 10.13 Testing strategy

The range coder is self-contained enough to test in isolation:

1. **Round-trip test**: Initialize encoder, encode a sequence of symbols with known CDF tables, call `ec_enc_done`, then initialize a decoder on the output buffer and verify the decoded symbols match.
2. **Bit-exact test via FFI**: Link the C reference range coder and the Rust implementation, feed identical inputs to both, compare the output byte streams bit-for-bit.
3. **`ec_tell` / `ec_tell_frac` consistency**: After each symbol, verify that encoder and decoder report the same bit count.
4. **Boundary tests**: Encode until the buffer is nearly full; verify error handling when the range-coded and raw-bit regions collide.
5. **`ec_enc_patch_initial_bits`**: Test all three code paths (byte written, byte buffered, normalization not yet run).
