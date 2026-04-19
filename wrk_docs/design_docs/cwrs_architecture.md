I now have everything I need. Here is the complete architecture document:

---

# CWRS Module Architecture — `celt/cwrs.c`, `celt/cwrs.h`

## 1. Purpose

The CWRS (Combinatorial Waveform Representation Sequence) module implements **Pyramid Vector Quantization (PVQ) codebook enumeration** for the CELT layer of the Opus codec. It provides the bijective mapping between:

- **Pulse vectors** — integer vectors `y[0..N-1]` where `sum(|y[i]|) == K`, each element having a sign — and
- **Codebook indices** — a single unsigned 32-bit integer in `[0, V(N,K))`.

This is the core entropy coding primitive for CELT band quantization. The encoder converts a pulse vector (output of PVQ search in `vq.c`) into an integer index, which is then range-coded. The decoder reverses this: it reads an index from the bitstream and reconstructs the pulse vector.

The module also provides `get_required_bits()`, which computes the number of bits needed to code `V(N,K)` codewords — used by the bit allocation logic in `rate.c`.

### Position in the Opus pipeline

```
Encoder:  vq.c:alg_quant() → encode_pulses() → ec_enc_uint()
Decoder:  ec_dec_uint() → decode_pulses() → vq.c:alg_unquant()
Bit alloc: rate.c → get_required_bits()
```

## 2. Public API

### `encode_pulses`
```c
void encode_pulses(const int *_y, int _n, int _k, ec_enc *_enc);
```
| Parameter | Type | Description |
|-----------|------|-------------|
| `_y` | `const int *` | Pulse vector of length `_n`. `sum(abs(_y[i])) == _k`. |
| `_n` | `int` | Number of dimensions (band size). Must be ≥ 2. |
| `_k` | `int` | Number of pulses (sum of absolute values). Must be > 0. |
| `_enc` | `ec_enc *` | Range encoder state. |

**Behavior**: Computes the combinatorial index of `_y` within the PVQ codebook of size `V(_n, _k)`, then encodes it as a uniform integer via `ec_enc_uint`.

### `decode_pulses`
```c
opus_val32 decode_pulses(int *_y, int _n, int _k, ec_dec *_dec);
```
| Parameter | Type | Description |
|-----------|------|-------------|
| `_y` | `int *` | Output pulse vector of length `_n`. |
| `_n` | `int` | Number of dimensions. Must be > 1. |
| `_k` | `int` | Number of pulses. Must be > 0. |
| `_dec` | `ec_dec *` | Range decoder state. |

**Returns**: `opus_val32` — the squared norm `sum(y[i]^2)`, computed as a side-effect during decoding. In fixed-point mode this is `opus_int32`; in float mode it is `float`.

### `get_required_bits` (CUSTOM_MODES only)
```c
void get_required_bits(opus_int16 *_bits, int _n, int _maxk, int _frac);
```
| Parameter | Type | Description |
|-----------|------|-------------|
| `_bits` | `opus_int16 *` | Output array of size `_maxk + 1`. |
| `_n` | `int` | Band size (dimensions). |
| `_maxk` | `int` | Maximum number of pulses. Must be > 0. |
| `_frac` | `int` | Fractional precision bits for the log2 result. |

**Behavior**: Sets `_bits[k] = ceil(log2(V(_n, k)))` in Q`_frac` format for `k` in `[0, _maxk]`. `_bits[0]` is always 0.

### `log2_frac` (CUSTOM_MODES only)
```c
int log2_frac(opus_uint32 val, int frac);
```
Conservative (always rounds up) binary logarithm with `frac` fractional bits. Maximum overestimation tested at 0.06254 bits for `frac=4`.

## 3. Internal State

**This module is stateless.** There are no persistent structs. All working memory is either:
- Stack-allocated via `ALLOC()` / `VARDECL()` (in SMALL_FOOTPRINT mode), or
- Looked up from the static `CELT_PVQ_U_DATA` table (in normal mode).

The only "state" is the scratch buffer `_u[]` of size `_k + 2`, allocated on the stack for the SMALL_FOOTPRINT path, used to hold a row of the U(N,K) matrix during encode/decode.

## 4. Algorithm

### 4.1 Core Combinatorial Functions

Two functions govern the codebook:

- **V(N, K)** = number of N-dimensional signed pulse vectors with exactly K pulses.
  ```
  V(N,K) = K>0 ? sum(k=1..K, 2^k * C(N,k) * C(K-1,k-1)) : 1
  ```

- **U(N, K)** = number of such vectors where the first N-1 dimensions use at most K-1 pulses.
  ```
  U(N,K) = sum(k=0..K-1, V(N-1,k))
         = K>0 ? (V(N-1,K-1) + V(N,K-1)) / 2 : 0
  ```

Key identity: **`V(N,K) = U(N,K) + U(N,K+1)`**

- `U(N, K+1)` counts vectors where dimension 0 is non-negative.
- `U(N, K)` counts vectors where dimension 0 is negative.

Both obey the same recurrence:
```
U(N,K) = U(N-1,K) + U(N,K-1) + U(N-1,K-1)    for N>0, K>0
```

### 4.2 Encoding Algorithm (`icwrs`)

Given a pulse vector `_y[0..N-1]`, compute its codebook index.

**Normal (table-lookup) path — `!SMALL_FOOTPRINT`:**
```
Process dimensions from last to first (j = N-1 down to 0):
  1. i = (_y[N-1] < 0) ? 1 : 0     // sign of last element
  2. k = |_y[N-1]|                   // running pulse count
  3. For j = N-2 down to 0:
     a. i += U(N-j, k)              // skip all vectors with fewer pulses in remaining dims
     b. k += |_y[j]|                // add this dimension's pulses to running total
     c. If _y[j] < 0: i += U(N-j, k+1)  // account for sign
  4. Return i
```

**SMALL_FOOTPRINT path:**
Same logic, but instead of table lookup, the U row is maintained in a buffer `_u[]` and advanced via `unext()` at each step.

### 4.3 Decoding Algorithm (`cwrsi`)

Given index `_i` in `[0, V(N,K))`, reconstruct the pulse vector.

**Normal (table-lookup) path — `!SMALL_FOOTPRINT`:**
```
For each dimension (first to last):
  If K >= N ("lots of pulses"):
    1. p = U(N, K+1)                    // threshold for negative sign
    2. s = -(_i >= p); _i -= p & s      // extract sign bit
    3. Search for k in row N: find largest U(N,k) ≤ _i
       (linear scan downward from K, or from N if U(N,N) > _i)
    4. _i -= U(N, k)                    // subtract offset
    5. y[j] = (K - k + s) ^ s           // reconstruct value with sign
    6. K = k
  Else ("lots of dimensions"):
    1. p = U(K, N); q = U(K+1, N)
    2. If p ≤ _i < q: y[j] = 0, _i -= p     // zero pulses here
    3. Else: extract sign, search for k, reconstruct as above
  N--
Final two dimensions (N==2, N==1) handled with closed-form formulas.
```

**SMALL_FOOTPRINT path:**
Same logic, but uses `_u[]` buffer and `uprev()` to go backwards through rows.

### 4.4 Row Construction (`ncwrs_urow`, SMALL_FOOTPRINT only)

Builds the `U(N, 0..K+1)` row from scratch:
```
1. Initialize with U(2, k) = 2k - 1 for k >= 2, U(2,0) = 0, U(2,1) = 1
2. For n = 2 to N-1: apply unext() to advance the row
3. Return V(N,K) = U[K] + U[K+1]
```

### 4.5 Row Advancement/Regression

**`unext(_ui, _len, _ui0)`** — compute next row:
```
For j = 1 to _len-1:
  ui1 = _ui[j] + _ui[j-1] + _ui0
  _ui[j-1] = _ui0
  _ui0 = ui1
_ui[_len-1] = _ui0
```
This implements `U(N+1, K) = U(N, K) + U(N+1, K-1) + U(N, K-1)` in-place.

**`uprev(_ui, _n, _ui0)`** — compute previous row:
```
For j = 1 to _n-1:
  ui1 = _ui[j] - _ui[j-1] - _ui0
  _ui[j-1] = _ui0
  _ui0 = ui1
_ui[_n-1] = _ui0
```

Both mutate the buffer in-place. The `_ui0` parameter is the boundary condition (always 0 for `U` rows where N ≥ 2).

## 5. Data Flow

### Encoding data flow
```
int _y[N]  (pulse vector, |y|₁ = K)
     │
     ▼
  icwrs(_n, _y)  →  opus_uint32 index ∈ [0, V(N,K))
     │
     ▼
  ec_enc_uint(enc, index, V(N,K))  →  bits written to bitstream
```

### Decoding data flow
```
  ec_dec_uint(dec, V(N,K))  →  opus_uint32 index
     │
     ▼
  cwrsi(_n, _k, index, _y)  →  int _y[N]  (pulse vector)
                                opus_val32 yy = Σ(y[i]²)
```

### Buffer layouts

- **`_y[N]`**: Pulse vector. Each element is a signed integer. `sum(|_y[i]|) == K`. Written left-to-right during decode.
- **`_u[K+2]`** (SMALL_FOOTPRINT): Scratch buffer holding one row of U. `_u[k]` = U(n, k) for current row n. Size is `K + 2` because we need `U(n, K+1)` for the sign decision.

## 6. Numerical Details

### Integer types
| Type | Width | Usage |
|------|-------|-------|
| `opus_uint32` | 32-bit unsigned | Codebook indices, U/V values, all combinatorial arithmetic |
| `int` | platform int (≥32 bits) | Pulse vector elements, dimensions, pulse counts |
| `opus_int16` | 16-bit signed | Intermediate value `val` (pulse magnitude with sign), bit counts |
| `opus_val32` | 32-bit signed int (fixed) or float | Squared norm accumulator `yy` |

### Overflow constraints

All values of `V(N, K)` and `U(N, K)` **must fit in 32 bits unsigned**. This is the fundamental constraint limiting the codebook size. The precomputed table `CELT_PVQ_U_DATA` is designed so that every entry fits in `opus_uint32`.

The maximum K for each N is determined by when `V(N, K)` would overflow `2^32`:
- N=14: K max = 14 (V = 1,409,933,619 × 2 + ... — barely fits)
- N=6: K max = 96 (or 109 with CWRS_EXTRA_ROWS)
- N=2: K max = 176 (or 208 with CWRS_EXTRA_ROWS); V(2,K) = 4K

### Wrapping arithmetic
- `UADD32(a, b)` and `USUB32(a, b)` are simple `(a)+(b)` / `(a)-(b)` in release mode. In debug mode (`fixed_debug.h`) they include overflow/underflow assertions.
- The code relies on unsigned wrapping semantics being well-defined (C standard guarantees this for unsigned types).

### Sign encoding trick
Signs are encoded using the `(val + s) ^ s` pattern where `s = 0` or `s = -1`:
```c
s = -(_i >= p);     // s = 0 or s = -1 (all bits set)
_i -= p & s;        // conditional subtract without branching
val = (yj + s) ^ s; // if s=0: val=yj; if s=-1: val = ~yj + 1 = -yj (two's complement negate)
```
This is a branchless conditional negate. **Porting note**: Relies on two's complement representation.

### MAC16_16
```c
yy = MAC16_16(yy, val, val);  // yy += val * val
```
In fixed-point: 16×16→32 multiply-accumulate. In float: standard FMA. The `val` is declared `opus_int16`, so the multiply is always 16-bit × 16-bit. The result accumulates into `opus_val32`.

### Squared norm return value
`decode_pulses` returns `yy = sum(y[i]^2)`. In fixed-point this is an `opus_int32`. The caller (`alg_unquant` in `vq.c`) uses this to normalize the decoded vector. For typical N ≤ 208 and moderate K, this fits comfortably in 32 bits.

## 7. Dependencies

### This module calls:
| Module | Functions used |
|--------|---------------|
| `entenc.h` | `ec_enc_uint(enc, fl, ft)` — encode uniform integer |
| `entdec.h` | `ec_dec_uint(dec, ft)` — decode uniform integer |
| `mathops.h` | `EC_ILOG()` (via `ecintrin.h`, used by `log2_frac`) |
| `arch.h` | `IMIN`, `IMAX`, `UADD32`, `USUB32`, `MAC16_16`, type definitions |
| `stack_alloc.h` | `VARDECL`, `ALLOC`, `SAVE_STACK`, `RESTORE_STACK` |
| `os_support.h` | `celt_assert`, `celt_sig_assert` |

### What calls this module:
| Module | Functions called |
|--------|----------------|
| `vq.c` → `alg_quant()` | `encode_pulses()` — after PVQ search finds pulse vector |
| `vq.c` → `alg_unquant()` | `decode_pulses()` — to reconstruct pulse vector from bitstream |
| `rate.c` → `compute_pulse_cache()` | `get_required_bits()` — for bit allocation tables |

## 8. Constants and Tables

### `CELT_PVQ_U_DATA` (normal path only)

A flat `const opus_uint32` array containing precomputed values of `U(N, K)` for `N` in `[0, 14]` and K up to the maximum that fits in 32 bits.

**Size**: 1272 entries (standard) or 1488 entries (with `CWRS_EXTRA_ROWS` for CUSTOM_MODES/ENABLE_QEXT).

**Layout**: Rows are stored contiguously. Row N contains U(N, K) for K from N to the maximum K for that row:

| Row (N) | K range (standard) | K range (extra) | Offset (standard) |
|---------|--------------------|-----------------|--------------------|
| 0 | 0..176 | 0..208 | 0 |
| 1 | 1..176 | 1..208 | 176 |
| 2 | 2..176 | 2..208 | 351 |
| 3 | 3..176 | 3..208 | 525 |
| 4 | 4..176 | 4..208 | 698 |
| 5 | 5..176 | 5..208 | 870 |
| 6 | 6..96 | 6..109 | 1041 |
| 7 | 7..54 | 7..60 | 1131 |
| 8 | 8..37 | 8..40 | 1178 |
| 9 | 9..28 | 9..29 | 1207 |
| 10 | 10..24 | same | 1226 |
| 11 | 11..19 | 11..20 | 1240 |
| 12 | 12..18 | same | 1248 |
| 13 | 13..16 | same | 1254 |
| 14 | 14..14 | same | 1257 |

### `CELT_PVQ_U_ROW`

An array of 15 `const opus_uint32 *` pointers, one per row. Each pointer is offset so that `CELT_PVQ_U_ROW[n][k]` gives `U(n, k)` directly (note: the pointer is offset by `-n` implicitly by the data layout, so indexing by absolute K works).

### `CELT_PVQ_U` and `CELT_PVQ_V` macros

```c
#define CELT_PVQ_U(_n, _k) (CELT_PVQ_U_ROW[IMIN(_n,_k)][IMAX(_n,_k)])
#define CELT_PVQ_V(_n, _k) (CELT_PVQ_U(_n,_k) + CELT_PVQ_U(_n,(_k)+1))
```

The `IMIN/IMAX` exploits the symmetry `U(N,K) = U(K,N)` to always index into the smaller row dimension. This is critical: the table only stores rows for N up to 14, so if N > 14 and K ≤ 14, we access `U_ROW[K][N]` instead.

### Derived formulas for small N

For `N == 2`:
```
U(2, K) = 2K - 1
V(2, K) = 4K
```
The decode path exploits this:
```c
p = 2*_k + 1;           // U(2, K+1) = 2(K+1)-1 = 2K+1
_k = (_i + 1) >> 1;     // closed-form inverse of U(2, k) = 2k-1
```

For `N == 1`:
```
U(1, K) = 1 for K > 0
V(1, K) = 2 for K > 0
```
The final element just has a sign bit: `s = -(int)_i; val = (_k+s)^s`.

## 9. Edge Cases

### K == 0
Asserted against. `encode_pulses` and `decode_pulses` both `celt_assert(_k > 0)`. The caller must not call with zero pulses.

### N == 1
In the normal path, the loop exits at N==2 and handles N==1 inline. In the SMALL_FOOTPRINT path, the loop runs `j = 0..N-1`, so N==1 produces exactly one iteration with a single `uprev`.

### N == 0
Not supported. `celt_assert(_n >= 2)` in the normal-path `icwrs`.

### Very large K relative to N
The decode loop has two branches:
- `K >= N` ("lots of pulses"): searches within the row using `CELT_PVQ_U_ROW[_n]`
- `K < N` ("lots of dimensions"): searches using `CELT_PVQ_U_ROW[_k]` and `CELT_PVQ_U_ROW[_k+1]`

This avoids indexing beyond the precomputed table bounds.

### Maximum codebook size
`V(N, K)` must fit in `opus_uint32` (< 2^32). The table is sized exactly to this constraint. For standard Opus modes, band sizes go up to 176; for custom modes, up to 208. The K values are limited accordingly.

### Zero-valued dimensions
When a dimension has zero pulses, the decode path handles it with a special fast case (`_y++ = 0`, no sign extraction needed).

### `celt_sig_assert` vs `celt_assert`
At line 489, `celt_sig_assert(p > q)` is a weaker assertion that may be compiled out even when regular assertions are enabled. It guards a condition that should be mathematically guaranteed but is expensive to verify.

## 10. Porting Notes for Rust

### Two compilation paths — choose one

The file has two complete implementations gated by `#if !defined(SMALL_FOOTPRINT)`:

1. **Normal path** (lines 197–545): Uses the precomputed `CELT_PVQ_U_DATA` table for O(1) lookups. Faster, larger binary.
2. **SMALL_FOOTPRINT path** (lines 547–719): Computes U values on the fly using `unext()`/`uprev()` with an O(K) scratch buffer. Slower, smaller binary.

**Recommendation**: Port the normal (table) path as the default. The SMALL_FOOTPRINT path can be a feature flag. Both must produce identical output.

### Pointer arithmetic in `CELT_PVQ_U_ROW`

The C code uses an array of pointers into a flat data array:
```c
static const opus_uint32 *const CELT_PVQ_U_ROW[15] = {
    CELT_PVQ_U_DATA + 0, CELT_PVQ_U_DATA + 176, ...
};
```
And indexes as `CELT_PVQ_U_ROW[n][k]` where k may be large.

**Rust approach**: Use a flat `[u32; 1272]` array with a `[usize; 15]` offset table, then index as `CELT_PVQ_U_DATA[CELT_PVQ_U_ROW[n] + k]`. Or use slices:
```rust
const CELT_PVQ_U_ROW: [&[u32]; 15] = [
    &CELT_PVQ_U_DATA[0..177],
    &CELT_PVQ_U_DATA[176..352],
    // ...
];
```

### In-place buffer mutation (`unext`, `uprev`)

Both functions mutate a `u32` slice in-place with a sliding window pattern. Straightforward to port as `fn unext(ui: &mut [u32], ui0: u32)`. The do-while loop becomes:
```rust
let mut j = 1;
loop {
    let ui1 = ui[j].wrapping_add(ui[j-1]).wrapping_add(ui0);
    ui[j-1] = ui0;
    ui0 = ui1;
    j += 1;
    if j >= len { break; }
}
ui[j-1] = ui0;
```

### Wrapping arithmetic

`UADD32` and `USUB32` are plain addition/subtraction that **may wrap** in the unsigned domain. Use Rust's `u32::wrapping_add()` and `u32::wrapping_sub()`, or use the `Wrapping<u32>` type. The debug variants in the C code assert on overflow — consider adding debug assertions in Rust too.

### Branchless sign trick

```c
s = -(_i >= p);           // 0 or 0xFFFFFFFF
_i -= p & s;              // conditional subtract
val = (yj + s) ^ s;       // conditional negate
```

In Rust:
```rust
let s = if i >= p { u32::MAX } else { 0 };  // or: -(i >= p) as i32 as u32
i = i.wrapping_sub(p & s);
let val = ((yj as i32).wrapping_add(s as i32)) ^ (s as i32);
```

Be careful: the C code casts between signed and unsigned freely. In Rust, use explicit `as` casts and wrapping operations. The `val` result is `opus_int16` — ensure no truncation issues.

### `VARDECL` / `ALLOC` stack allocation

The C code uses stack allocation macros (`alloca`-based). In Rust:
- For small fixed-size buffers: use stack arrays `[u32; MAX_K + 2]` if K has a known bound
- For variable-size: use `Vec<u32>` or `SmallVec` from the `smallvec` crate
- The maximum K is bounded by the table (176 standard, 208 custom), so `K + 2 ≤ 210` — small enough for stack allocation via a fixed array

### `MAC16_16` — fixed-point multiply-accumulate

In fixed-point mode: `MAC16_16(c, a, b) = c + (a as i32) * (b as i32)`. The operands `a` and `b` are `opus_int16` (i16 in Rust), result is `i32`. No overflow risk: `i16 × i16` fits in `i32`, and the accumulation over N ≤ 208 with max K ≈ 176 still fits.

In float mode: `MAC16_16(c, a, b) = c + (a as f32) * (b as f32)`.

### Return value of `decode_pulses`

Returns `opus_val32` — this is `i32` in fixed-point, `f32` in float. Since the module only does integer combinatorics, the float path is technically computing integer squared norm as float. The Rust port should preserve this type distinction via a generic or feature flag.

### Conditional compilation summary

| Preprocessor guard | Effect |
|---------------------|--------|
| `SMALL_FOOTPRINT` | Selects table-free computation path |
| `CUSTOM_MODES` | Enables `log2_frac`, `get_required_bits` |
| `CWRS_EXTRA_ROWS` / `ENABLE_QEXT` | Extends tables to N=208 (larger bands) |

### Pointer advancing in `cwrsi` (normal path)

The decode loop uses `*_y++ = val` to write output and advance the pointer simultaneously. In Rust, use an iterator or index variable:
```rust
for j in 0..n {
    y[j] = val;
    // ...
}
```

### Symmetry exploitation

`CELT_PVQ_U(_n, _k)` uses `IMIN(_n, _k)` / `IMAX(_n, _k)` to exploit U(N,K) = U(K,N). This means the row index is always ≤ 14 (the table has 15 rows). The Rust port must preserve this: always index with `min(n, k)` as the row.

### No floating-point in the combinatorial logic

Despite `opus_val32` potentially being `float`, all the combinatorial functions (`icwrs`, `cwrsi`, `ncwrs_urow`, `unext`, `uprev`) operate exclusively on `opus_uint32`. The only floating-point interaction is the `MAC16_16` accumulation of `yy`. This means the core enumeration logic can be a pure `u32`/`i32` module with a thin wrapper that adapts the `yy` return type.

### `log2_frac` precision

The `log2_frac` function uses an iterative squaring method with 16-bit precision. It:
1. Normalizes the input to [0x8000, 0xFFFF] range
2. Iteratively squares and extracts the integer part
3. Guarantees conservative (upward) rounding

Maximum overestimation: 0.06254 bits at `frac=4`. This is only used for CUSTOM_MODES bit allocation and is not in the critical path.
