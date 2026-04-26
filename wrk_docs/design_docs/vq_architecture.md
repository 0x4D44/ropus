I now have all the information needed. Here is the complete architecture document:

---

# VQ & Rate Modules — Architecture Documentation

## Module: `celt/vq.c` / `celt/vq.h`
## Module: `celt/rate.c` / `celt/rate.h`

---

## 1. Purpose

These two modules handle **vector quantization** and **bit allocation** in the CELT layer of Opus.

**VQ module** (`vq.c`/`vq.h`): Implements Pyramid Vector Quantization (PVQ), the core mechanism that encodes spectral band coefficients as integer pulse vectors on an L1 unit sphere. The signal in each band is represented as a direction (encoded via PVQ) and a magnitude (encoded separately as "fine energy"). This is the most CPU-intensive part of the codec.

**Rate module** (`rate.c`/`rate.h`): Determines how many pulses (and thus bits) each frequency band receives, given a total bit budget. It performs bit allocation across bands using precomputed allocation tables, interpolation, and a band-skipping mechanism. It also precomputes the pulse cache — a table mapping pulse counts to bit costs for each band size.

Together they form the quantization pipeline: rate decides *how many pulses* each band gets, then VQ *encodes/decodes* those pulses.

---

## 2. Public API

### 2.1 VQ Functions (`vq.h`)

#### `alg_quant` — PVQ Encoder
```c
unsigned alg_quant(celt_norm *X, int N, int K, int spread, int B,
      ec_enc *enc, opus_val32 gain, int resynth
      ARG_QEXT(ec_enc *ext_enc) ARG_QEXT(int extra_bits), int arch);
```
- **X**: Input/output band coefficients (normalized spectrum). Modified in-place — on return contains the quantized, re-normalized, inverse-rotated version if `resynth` is set.
- **N**: Number of coefficients in the band.
- **K**: Number of pulses (controls quantization resolution).
- **spread**: Spreading mode (`SPREAD_NONE=0`, `SPREAD_LIGHT=1`, `SPREAD_NORMAL=2`, `SPREAD_AGGRESSIVE=3`).
- **B**: Number of short blocks (1 for long blocks, >1 for transients). Used for collapse mask and stride in rotation.
- **enc**: Entropy coder state (writes the encoded pulse vector).
- **gain**: Desired output norm for resynthesis (Q16 in fixed-point).
- **resynth**: If nonzero, reconstruct the quantized signal back into X.
- **ext_enc / extra_bits** (QEXT only): Extension encoder and extra precision bits.
- **arch**: CPU feature flags (for SIMD dispatch, unused in C fallback).
- **Returns**: Collapse mask — a bitmask where bit `i` is set if short block `i` received at least one pulse.

#### `alg_unquant` — PVQ Decoder
```c
unsigned alg_unquant(celt_norm *X, int N, int K, int spread, int B,
      ec_dec *dec, opus_val32 gain
      ARG_QEXT(ec_enc *ext_dec) ARG_QEXT(int extra_bits));
```
- Parameters mirror `alg_quant`. Reads pulse vector from `dec`, reconstructs X.
- **Returns**: Collapse mask (same semantics).

#### `exp_rotation` — Spreading Rotation
```c
void exp_rotation(celt_norm *X, int len, int dir, int stride, int K, int spread);
```
- Applies or removes the "experience rotation" that spreads spectral energy.
- **dir**: `+1` = forward (before quantization), `-1` = inverse (after).
- **stride**: Number of short blocks (`B`).
- In-place modification of X.

#### `op_pvq_search_c` — PVQ Search (Core Inner Loop)
```c
opus_val16 op_pvq_search_c(celt_norm *X, int *iy, int K, int N, int arch);
```
- Finds the integer pulse vector `iy[]` (K pulses in N dimensions) closest to `X`.
- **X**: Input signal (absolute values — sign stripped beforehand). Modified: absolute values stored in-place.
- **iy**: Output integer pulse vector. Signs restored afterward by caller.
- **Returns**: `yy` — the squared norm of the pulse vector (as `opus_val16`).

#### `renormalise_vector`
```c
void renormalise_vector(celt_norm *X, int N, opus_val32 gain, int arch);
```
- Rescales X to have the specified `gain` as its L2 norm. Used during band processing when the norm drifts.

#### `stereo_itheta`
```c
opus_int32 stereo_itheta(const celt_norm *X, const celt_norm *Y,
      int stereo, int N, int arch);
```
- Computes the quantized angle `itheta` between the mid (X+Y) and side (X-Y) components for stereo coding.
- Returns an integer in Q14 range [0, 16384] representing `atan2(side_energy, mid_energy)`.

#### Fixed-Point Helpers (FIXED_POINT only)
```c
void norm_scaleup(celt_norm *X, int N, int shift);
void norm_scaledown(celt_norm *X, int N, int shift);
opus_val32 celt_inner_prod_norm(const celt_norm *x, const celt_norm *y, int len, int arch);
opus_val32 celt_inner_prod_norm_shift(const celt_norm *x, const celt_norm *y, int len, int arch);
```
- `norm_scaledown`: Right-shifts each element by `shift` bits (with rounding via `PSHR32`).
- `norm_scaleup`: Left-shifts each element by `shift` bits.
- `celt_inner_prod_norm`: Standard dot product of two norm vectors.
- `celt_inner_prod_norm_shift`: Dot product with extra precision (64-bit accumulator), result right-shifted by `2*(NORM_SHIFT-14)`.

In float mode, these are macros: `celt_inner_prod_norm` → `celt_inner_prod`, scale operations are no-ops.

#### QEXT Functions (Opus 1.5+)
```c
unsigned cubic_quant(celt_norm *X, int N, int K, int B, ec_enc *enc, opus_val32 gain, int resynth);
unsigned cubic_unquant(celt_norm *X, int N, int K, int B, ec_dec *dec, opus_val32 gain);
```
- Alternative quantizer for QEXT mode using a cubic lattice instead of PVQ.

### 2.2 Rate Functions (`rate.h`)

#### `clt_compute_allocation` — Main Bit Allocation
```c
int clt_compute_allocation(const CELTMode *m, int start, int end,
      const int *offsets, const int *cap, int alloc_trim,
      int *intensity, int *dual_stereo,
      opus_int32 total, opus_int32 *balance,
      int *pulses, int *ebits, int *fine_priority,
      int C, int LM, ec_ctx *ec, int encode, int prev, int signalBandwidth);
```
- **m**: Mode definition (contains band edges, allocation vectors, pulse cache).
- **start/end**: Range of bands to allocate.
- **offsets**: Per-band dynamic allocation adjustments (from dynalloc).
- **cap**: Maximum bits each band can use.
- **alloc_trim**: Tilt parameter (0–10, nominal 5) controlling high-vs-low frequency emphasis.
- **intensity/dual_stereo**: Stereo coding parameters (in/out).
- **total**: Total bit budget (in 1/8th-bit resolution, i.e., scaled by `BITRES=3`).
- **balance**: Leftover bits passed to `quant_all_bands` for rebalancing (out).
- **pulses**: Output — bits allocated to PVQ per band (in 1/8th bits).
- **ebits**: Output — fine energy bits per band.
- **fine_priority**: Output — whether each band's fine bits are low or high priority.
- **C**: Number of channels (1 or 2).
- **LM**: Log2 of block size multiplier (0=2.5ms, 1=5ms, 2=10ms, 3=20ms).
- **ec**: Entropy coder (for encoding/decoding skip decisions).
- **encode**: 1 if encoding, 0 if decoding.
- **prev**: Previous frame's `codedBands` (for hysteresis in skip decisions).
- **signalBandwidth**: Highest useful band (encoder hint).
- **Returns**: `codedBands` — number of bands that received PVQ bits.

#### `compute_pulse_cache` (CUSTOM_MODES only)
```c
void compute_pulse_cache(CELTMode *m, int LM);
```
Precomputes the bits-per-pulse lookup table for all unique band sizes.

#### Inline Helpers

```c
static OPUS_INLINE int get_pulses(int i);
```
Decodes a compact pulse index: for `i < 8`, returns `i`; otherwise `(8 + (i & 7)) << ((i >> 3) - 1)`. This encodes pulse counts pseudo-logarithmically up to `CELT_MAX_PULSES` (128).

```c
static OPUS_INLINE int bits2pulses(const CELTMode *m, int band, int LM, int bits);
```
Binary searches the pulse cache to find the maximum number of pulses that fit within `bits` (1/8th-bit units).

```c
static OPUS_INLINE int pulses2bits(const CELTMode *m, int band, int LM, int pulses);
```
Looks up the bit cost for a given pulse count from the cache.

---

## 3. Internal State / Structs

### 3.1 `PulseCache` (defined in `modes.h`)
```c
typedef struct {
   int size;                    // Total number of cache entries
   const opus_int16 *index;    // index[LM * nbEBands + band] → offset into bits[]
   const unsigned char *bits;  // bits[offset + K] = cost of K pulses (minus 1), bits[offset + 0] = max K
   const unsigned char *caps;  // caps[(LM * 2 + C-1) * nbEBands + band] = max useful bits/coeff
} PulseCache;
```
**Lifecycle**: Built once at mode initialization by `compute_pulse_cache()`. Read-only thereafter. Indexed by `(LM+1)` (note the `LM++` in `bits2pulses`/`pulses2bits`).

### 3.2 No Other Persistent State

Both modules are **stateless** — they operate purely on their arguments plus the mode's precomputed tables. There are no module-level globals or per-frame persistent state. All working memory is stack-allocated via `ALLOC()` / `VARDECL()`.

---

## 4. Algorithms

### 4.1 PVQ Search (`op_pvq_search_c`)

The core algorithm finds the best integer approximation to a unit-norm signal using K pulses in N dimensions. The result lies on the surface of an L1 sphere of radius K.

**Step-by-step:**

1. **Fixed-point scaling** (FIXED_POINT only): Compute `shift = (ilog2(1 + ||X||²) + 1) / 2` and scale X down by `max(0, shift + NORM_SHIFT - 14 - 14)` to prevent overflow during subsequent 16×16 multiplications.

2. **Strip signs**: Store `signx[j] = (X[j] < 0)`, replace `X[j]` with `|X[j]|`. Initialize `iy[j] = 0`, `y[j] = 0`.

3. **Projection pre-search** (when `K > N/2`): Compute `sum = Σ|X[j]|`, then `rcp = K / sum`. Project: `iy[j] = floor(X[j] * rcp)` (rounding toward zero is critical in fixed-point). Compute initial `xy = Σ X[j]*y[j]` and `yy = Σ y[j]²`. Set `pulsesLeft = K - Σiy[j]`.

4. **Safety check**: If `pulsesLeft > N+3` (degenerate input), dump all remaining pulses into bin 0.

5. **Greedy refinement**: For each remaining pulse:
   - For each dimension j, compute what `Rxy` and `Ryy` would be if a pulse were added there.
   - Score = `Rxy² / Ryy` (division-free comparison: `best_den * Rxy² > Ryy * best_num`).
   - Place the pulse in the best dimension.
   - Key trick: `y[j]` stores `2 * iy[j]` to avoid multiplying by 2 in the inner loop.

6. **Restore signs**: `iy[j] = (iy[j] ^ -signx[j]) + signx[j]` (branchless sign flip).

**Complexity**: O(K·N) for the greedy search, O(N) for the projection.

### 4.2 Exp Rotation (`exp_rotation`)

Spreads spectral energy before quantization to prevent sparse pulse distributions from causing audible "birdies". The rotation angle depends on `K/len` and the spread mode.

**Step-by-step:**

1. **Skip condition**: If `2*K >= len` or `spread == SPREAD_NONE`, no rotation needed (enough pulses to cover all bins).

2. **Compute angle**: `factor = SPREAD_FACTOR[spread-1]` (15, 10, or 5). `gain = len / (len + factor*K)`. `theta = gain²/2`. Compute `c = cos(theta)`, `s = sin(theta)` via `celt_cos_norm`.

3. **Compute stride2**: Secondary stride `≈ √(len/stride)` for additional rotation pass when `len >= 8*stride`.

4. **Apply rotation** (`exp_rotation1`): For each sub-block of size `len/stride`:
   - Forward direction (`dir=1`): First apply 1-stride rotation with `(c, s)`, then optionally `stride2` rotation with `(s, c)`.
   - Inverse direction (`dir=-1`): Reverse order with negated sine.
   - Each rotation pass is a butterfly-like operation: forward sweep then backward sweep across adjacent pairs at the given stride.

5. **Scaling**: In fixed-point, vectors are scaled down by `NORM_SHIFT - 14` before rotation (to fit in Q14 for 16-bit multiplications) and scaled back up afterward.

### 4.3 Normalize Residual (`normalise_residual`)

Converts integer pulse vector to unit-norm floating-point coefficients.

1. Compute `k = ilog2(Ryy) >> 1` (half the bit-width of the squared norm).
2. Normalize: `t = Ryy >> (2*(k-7) - 15)`.
3. Compute reciprocal sqrt: `g = rsqrt_norm(t) * gain`.
4. Scale: `X[i] = (iy[i] * g) >> (k + 15 - NORM_SHIFT)`.

### 4.4 Bit Allocation (`clt_compute_allocation`)

Distributes a total bit budget across frequency bands.

**Step-by-step:**

1. **Reserve overhead bits**: 1 bit for skip flag, `LOG2_FRAC_TABLE[end-start]` bits for intensity stereo index, 1 bit for dual-stereo flag.

2. **Compute per-band allocation curves**: For each band j:
   - `thresh[j]` = minimum bits to code any PVQ pulses.
   - `trim_offset[j]` = tilt adjustment based on `alloc_trim`.

3. **Binary search for allocation vector**: Find adjacent precomputed allocation vectors (`lo`, `hi`) that bracket the total budget. Each vector gives bits-per-band at a specific overall quality level.

4. **Interpolate** (`interp_bits2pulses`): Binary search on an interpolation parameter (6-bit, `ALLOC_STEPS=6`) between the two vectors to hit the target total.

5. **Band skipping**: Working backward from the highest band, decide which bands to skip entirely. Skipped bands get only fine energy bits, no PVQ. Skip decisions use hysteresis (`prev` frame) and signal bandwidth hints. Each skip decision costs 1 bit (entropy coded).

6. **Fine energy split**: For each coded band, split the allocated bits between PVQ pulses and fine energy refinement:
   - `ebits[j]` = fine energy bits per coefficient (max `MAX_FINE_BITS = 8`).
   - `fine_priority[j]` = whether this band's fine bits should be allocated in the first or second pass.
   - `bits[j]` = remaining bits for PVQ.

7. **Re-balancing**: Excess bits that can't be used by one band are carried forward as `balance` for use by `quant_all_bands`.

### 4.5 Pulse Cache (`compute_pulse_cache`)

Precomputes bit cost tables at mode init time:

1. Enumerate all unique `(N, K_max)` pairs across all band sizes and LM values.
2. For each unique size, call `get_required_bits()` (from CWRS) to compute the exact bit cost of encoding K pulses in N dimensions.
3. Store results in a flat byte array indexed by `cache.index[LM * nbEBands + band]`.
4. Also compute `cache.caps[]` — the maximum useful bit rate per band considering all split levels.

---

## 5. Data Flow

### 5.1 Encoding Path
```
Band coefficients X[N] (normalized, unit-norm)
    │
    ├── exp_rotation(X, N, +1, B, K, spread)    // Spread energy
    │
    ├── op_pvq_search(X, iy, K, N)              // Find integer pulse vector
    │       X is modified: absolute values stored
    │       iy[N] output: signed integer pulses, Σ|iy[i]| = K
    │
    ├── encode_pulses(iy, N, K, enc)             // Entropy-code via CWRS
    │
    ├── normalise_residual(iy, X, N, yy, gain)   // Reconstruct for prediction
    │
    └── exp_rotation(X, N, -1, B, K, spread)     // Undo rotation for reconstruction
```

### 5.2 Decoding Path
```
Entropy decoder state
    │
    ├── decode_pulses(iy, N, K, dec)             // Decode integer pulses, returns Ryy
    │
    ├── normalise_residual(iy, X, N, Ryy, gain)  // Convert to normalized coefficients
    │
    └── exp_rotation(X, N, -1, B, K, spread)     // Remove rotation
```

### 5.3 Buffer Layouts

- **X[N]**: Band coefficients. In fixed-point mode, these are `opus_val32` (Q`NORM_SHIFT` = Q24). In float mode, `float` near unit scale. Modified in-place throughout.
- **iy[N]** (or `iy[N+3]` in `alg_quant` for vectorization headroom): Integer pulse vector. Each `iy[i] ∈ [-K, K]`, with constraint `Σ|iy[i]| = K`.
- **y[N]**: Temporary during PVQ search, stores `2 * |iy[i]|` (doubled to avoid multiply in inner loop).
- **signx[N]**: Sign array (0 or 1) during PVQ search.

### 5.4 Rate Module Data Flow
```
CELTMode tables + total bit budget + offsets + caps
    │
    ├── Binary search over allocation vectors
    ├── Interpolation between adjacent vectors
    ├── Band skip decisions (written to / read from bitstream)
    │
    └── Output arrays:
        pulses[nbEBands]       — PVQ bits per band (in 1/8 bits)
        ebits[nbEBands]        — fine energy bits per band
        fine_priority[nbEBands] — priority flag per band
        balance                — leftover bits for rebalancing
        codedBands (return)    — number of coded bands
```

---

## 6. Numerical Details

### 6.1 Fixed-Point Formats

| Quantity | Type | Q-format | Range | Notes |
|----------|------|----------|-------|-------|
| `celt_norm` (fixed) | `opus_val32` (int32) | Q24 (`NORM_SHIFT=24`) | ±128.0 | Unit-norm vectors: elements ≈ [-1, 1] |
| `celt_norm` (float) | `float` | n/a | ≈ [-1, 1] | |
| `X[j]` during PVQ search | | Q14 | | Scaled down by `NORM_SHIFT-14` |
| `y[j]` in PVQ search | `celt_norm` | Q14 (same as X) | [0, 2K] | Stores `2*iy[j]`, not `iy[j]` |
| `xy` accumulator | `opus_val32` | Q28 | | Sum of Q14 × Q14 products |
| `yy` accumulator | `opus_val16` | Q14 | | Kept small via integer values |
| `gain` | `opus_val32` | Q16 | | Band gain for normalization |
| `theta` (rotation) | `opus_val16` | Q15 | [0, 0.5) | Half of gain² |
| `c`, `s` (rotation) | `opus_val16` | Q15 | | cos/sin of rotation angle |
| `itheta` (stereo) | `opus_int32` | Q14 | [0, 16384] | Stereo angle quantization |
| Bit allocations | `int` | Q3 (`BITRES=3`) | | 1/8th-bit resolution |

### 6.2 Overflow Guards

- **PVQ search pre-scaling**: Before the greedy loop, X is scaled down so that all 16×16 multiply-accumulate operations fit in 32 bits. The shift is computed as `max(0, (ilog2(||X||²)+1)/2 + NORM_SHIFT - 14 - 14)`.

- **PVQ inner loop right-shift**: `rshift = 1 + celt_ilog2(K - pulsesLeft + i + 1)` ensures `Rxy` fits in 16 bits after the shift. The `+1` accounts for the worst case.

- **Projection rounding**: In fixed-point, `iy[j] = MULT16_16_Q15(X[j], rcp)` rounds toward zero (critical — rounding up could give `Σiy > K`).

- **Float degenerate check**: `if (!(sum > EPSILON && sum < 64))` guards against NaN, infinity, and near-zero inputs.

- **normalise_residual**: Uses `ilog2(Ryy)>>1` to find the right shift for the reciprocal sqrt, keeping intermediate values in a safe range.

- **renormalise_vector**: Same pattern — scale down to Q14, compute in safe range, scale back to Q`NORM_SHIFT`.

### 6.3 Precision Requirements

- PVQ encoding/decoding must be **bit-exact** between encoder and decoder. The CWRS `encode_pulses`/`decode_pulses` achieve this through exact combinatorial indexing.
- `normalise_residual` is computed from `Ryy` (which equals `Σiy[i]²` and is exact), so the normalization is deterministic given the same pulse vector.
- `exp_rotation` is applied identically on encoder (forward+inverse for resynth) and decoder (inverse only), guaranteeing identical reconstructed signals.

### 6.4 Rounding Behavior

- `PSHR32(x, shift)` = `(x + (1 << (shift-1))) >> shift` — round-to-nearest.
- `MULT16_16_Q15(a, b)` = `(a*b) >> 15` — truncation toward zero.
- PVQ projection (`floor` in float, truncation in fixed) — always rounds down to ensure `Σiy ≤ K`.
- `bits2pulses` resolves ties by preferring the lower pulse count (compares distance to both neighbors).

---

## 7. Dependencies

### 7.1 What VQ Calls

| Module | Functions Used |
|--------|---------------|
| `cwrs.c` | `encode_pulses()`, `decode_pulses()` — combinatorial pulse coding |
| `mathops.c` | `celt_rsqrt_norm()`, `celt_rsqrt_norm32()`, `celt_rcp()`, `celt_rcp_norm32()`, `celt_cos_norm()`, `celt_sqrt32()`, `celt_atan2p_norm()`, `celt_div()`, `celt_ilog2()`, `celt_udiv()` |
| `pitch.c` | `celt_inner_prod()` (float mode, via macro alias) |
| `arch.h` | All fixed-point macros: `MULT16_16`, `MAC16_16`, `MULT16_32_Q15`, `MULT32_32_Q31`, `SHL32`, `SHR32`, `VSHR32`, `PSHR32`, `EXTRACT16`, `EXTEND32`, `ABS16`, `NEG16`, etc. |
| `entenc.c/entdec.c` | `ec_enc_bits()`, `ec_dec_bits()`, `ec_enc_uint()`, `ec_dec_uint()`, `ec_enc_bit_logp()`, `ec_dec_bit_logp()`, `ec_tell()` |
| `bands.h` | `SPREAD_*` constants |
| `rate.h` | (included but not directly called from vq.c) |
| `SigProc_FIX.h` | SILK fixed-point macros (used in some QEXT paths) |

### 7.2 What Rate Calls

| Module | Functions Used |
|--------|---------------|
| `cwrs.c` | `get_required_bits()` — bit cost computation |
| `modes.h` | `CELTMode` struct access (eBands, allocVectors, logN, cache) |
| `entcode.h` | `BITRES` constant (= 3) |
| `entenc.c/entdec.c` | `ec_enc_bit_logp()`, `ec_dec_bit_logp()`, `ec_enc_uint()`, `ec_dec_uint()` |
| `quant_bands.h` | `eMeans[]` table (QEXT path only) |

### 7.3 What Calls VQ

- `bands.c` → `quant_all_bands()` calls `alg_quant()` / `alg_unquant()` for each band.
- `bands.c` calls `stereo_itheta()` for stereo angle computation.
- `bands.c` calls `renormalise_vector()` during folding/copying.

### 7.4 What Calls Rate

- `celt_encoder.c` / `celt_decoder.c` → `clt_compute_allocation()` once per frame.
- Mode init → `compute_pulse_cache()` once at startup.

---

## 8. Constants and Tables

### 8.1 VQ Constants

| Constant | Value | Derivation |
|----------|-------|------------|
| `SPREAD_FACTOR[3]` | `{15, 10, 5}` | Indexed by `spread - 1`. Controls rotation aggressiveness. Higher = less spreading. Empirically tuned. |
| `NORM_SHIFT` | 24 | Q-format for `celt_norm` in fixed-point. Gives ~7 bits of integer range. |
| `Q15_ONE` / `Q15ONE` | 32767 | Maximum Q15 value (≈1.0). |

### 8.2 Rate Constants

| Constant | Value | Derivation |
|----------|-------|------------|
| `BITRES` | 3 | Bit allocation resolution: all bit counts are in 1/8th bits. |
| `MAX_PSEUDO` | 40 | Maximum compact pulse index in the cache. |
| `LOG_MAX_PSEUDO` | 6 | log2(MAX_PSEUDO) — iterations for binary search in `bits2pulses`. |
| `CELT_MAX_PULSES` | 128 | Maximum pulse count. |
| `MAX_FINE_BITS` | 8 | Maximum fine energy bits per coefficient. |
| `FINE_OFFSET` | 21 | Offset for fine energy bit allocation curve (in 1/8th bits). |
| `QTHETA_OFFSET` | 4 | Offset for theta (stereo angle) bit allocation. |
| `QTHETA_OFFSET_TWOPHASE` | 16 | Offset for N=2 bands (different theta distribution). |
| `ALLOC_STEPS` | 6 | Binary search precision for interpolation (64 steps). |
| `LOG2_FRAC_TABLE[24]` | `{0, 8,13, 16,19,21,23, ...}` | Approximate `ceil(log2(n)) * 8` for n=0..23. Used for intensity stereo reservation: `LOG2_FRAC_TABLE[end-start]` bits to code the intensity boundary index. |

### 8.3 Mode Tables (from `CELTMode`)

| Field | Description |
|-------|-------------|
| `m->eBands[]` | Band edge frequencies (in FFT bins per short block). |
| `m->logN[]` | `log2(band_width)` per band, in Q`BITRES` = Q3 format. |
| `m->allocVectors[]` | 2D table: `[vector_index * nbEBands + band]` → bits per coefficient (in 1/4 bits). |
| `m->nbAllocVectors` | Number of allocation vectors (quality levels). |
| `m->cache.bits[]` | Precomputed bits-per-pulse table. |
| `m->cache.index[]` | Index into `cache.bits[]` by `(LM+1) * nbEBands + band`. |
| `m->cache.caps[]` | Maximum useful bits per band per LM per channel count. |

---

## 9. Edge Cases

### 9.1 VQ Edge Cases

- **K = 0**: Not valid — `alg_quant` and `alg_unquant` assert `K > 0`.
- **N = 1**: Not valid — assert `N > 1`. (N=1 bands are handled by fine energy only in rate allocation.)
- **Near-zero input** (float): If `sum` is not `> EPSILON` and `< 64`, all energy is placed in X[0]. This prevents NaN/infinity propagation.
- **Near-zero input** (fixed): If `sum <= K`, same fallback.
- **pulsesLeft > N+3**: Safety valve — dumps all excess pulses into bin 0. Should not happen with well-formed input but guards against silence or extreme quantization noise.
- **2*K >= len**: Rotation is skipped (enough pulses already provide good coverage).
- **B = 1**: `extract_collapse_mask` returns 1 (single block, always "collapsed").

### 9.2 Rate Edge Cases

- **total <= 0**: All bands get 0 bits. Returns early in QEXT path.
- **N = 1 bands**: Get only a sign bit + fine energy, no PVQ allocation.
- **Single-coefficient bands** (`N<<LM == 1`): Special cap computation: `C * (1 + MAX_FINE_BITS) << BITRES`.
- **skip_start**: Bands boosted by dynalloc (positive offset) are never skipped.
- **codedBands ≤ start + 2**: Never skip below this (would waste the skip flag bit).
- **stereo with intensity**: Extra DoF compensation: `den = C*N + 1` when `C==2 && N>2 && !dual_stereo && j < intensity`.

---

## 10. Porting Notes

### 10.1 In-Place Mutation

Both `alg_quant` and `exp_rotation` modify X in-place. The PVQ search also modifies X (takes absolute values). In Rust, this means the caller must pass `&mut [celt_norm]`. The pattern of "strip signs into separate array, modify X, then restore" needs careful borrow management.

**Strategy**: Accept `&mut [i32]` (or appropriate type) for X, allocate `signx` and `y` on the stack (Vec or array).

### 10.2 Stack Allocation Macros

`VARDECL` + `ALLOC` is the C pattern for variable-length stack allocation (`alloca`-style). In Rust:
- For small, bounded sizes: use fixed-size arrays or `[T; MAX_SIZE]` with a length parameter.
- For larger/variable sizes: use `Vec<T>` (or a reusable scratch buffer).
- `SAVE_STACK` / `RESTORE_STACK` are debugging macros — no Rust equivalent needed.

### 10.3 Fixed-Point Macro Soup

The entire numerical core is expressed through macros (`MULT16_16`, `MAC16_16`, `SHL32`, `PSHR32`, etc.) that have different definitions in fixed-point vs. float. 

**Strategy**: Define a trait (e.g., `trait CeltArith`) with methods for each operation, and implement it for `i32` (fixed) and `f32` (float). Alternatively, use generics with a `Fixed` vs `Float` type parameter. Or — more pragmatically — port only fixed-point first (the reference implementation) and define the macros as `#[inline]` Rust functions.

Key macros to port:
- `MULT16_16(a,b)` → `(a as i32) * (b as i32)` (16×16→32)
- `MAC16_16(c,a,b)` → `c + (a as i32) * (b as i32)` (multiply-accumulate)
- `MULT16_32_Q15(a,b)` → `((a as i64) * (b as i64)) >> 15` (careful: 48-bit intermediate)
- `MULT32_32_Q31(a,b)` → `((a as i64) * (b as i64)) >> 31`
- `PSHR32(a,shift)` → `(a + (1 << (shift-1))) >> shift` (rounding right-shift)
- `VSHR32(a,shift)` → variable shift (handles negative shifts via left-shift)
- `SHL32(a,shift)` → `a << shift` (with debug overflow checks)

### 10.4 Conditional Compilation

The code has three axes of conditional compilation:
1. **`FIXED_POINT`** vs float: Affects types, macro definitions, and some algorithm paths.
2. **`ENABLE_QEXT`**: Opus 1.5+ quality extension. Adds QEXT refinement paths, cubic quantizer, and extra allocation.
3. **`CUSTOM_MODES`**: Gates `compute_pulse_cache` (standard modes use static tables).
4. **`OVERRIDE_*`**: Platform-specific optimizations (SSE2 for PVQ search, ARM NEON for rotation).

**Strategy**: Start with fixed-point, no QEXT, no CUSTOM_MODES, no SIMD overrides. Use Rust `cfg` attributes or feature flags for the rest.

### 10.5 Pointer Arithmetic

`exp_rotation1` uses advancing/retreating pointers:
```c
*Xptr++ = ...;    // forward sweep
*Xptr-- = ...;    // backward sweep
```
Replace with indexed access: `X[i]`, `X[i + stride]`.

`normalise_residual` uses a `do...while` loop starting from `i=0`. This is a common C idiom throughout — all `do { ... } while (++j < N)` loops should become `for j in 0..N`.

### 10.6 Branchless Tricks

```c
iy[j] = (iy[j] ^ -signx[j]) + signx[j];
```
This is a branchless conditional negate. In Rust: `if signx[j] != 0 { -iy[j] } else { iy[j] }` — the compiler will typically generate the same branchless code, or use `.wrapping_neg()` / bitwise ops explicitly if needed for bit-exactness.

### 10.7 The `arch` Parameter

The `arch` parameter threads through many functions but is unused in the C reference implementation (it's for runtime SIMD dispatch). In Rust, either omit it or use a phantom parameter / feature flag.

### 10.8 Integer Division

`celt_udiv(a, b)` is an unsigned integer division that may use platform-specific optimization. In Rust, just use `/` on `u32`.

### 10.9 The `y[j] *= 2` Trick

In `op_pvq_search_c`, `y[j]` stores twice the actual value to avoid a multiply in the inner loop comparison `Ryy = yy + y[j]` (instead of `yy + 2*y[j]`). This is a micro-optimization that must be preserved for bit-exactness in the scoring comparisons.

### 10.10 Bit Allocation is Bitstream-Coupled

`clt_compute_allocation` reads/writes the bitstream (skip decisions, intensity, dual-stereo). The encoder and decoder must execute the **identical** allocation logic to stay synchronized. Any divergence = corrupted decode. This means the Rust port must replicate every branch exactly, including the order of entropy coder operations.

### 10.11 Overflow Sensitivity in PVQ Search

The scoring comparison:
```c
if (MULT16_16(best_den, Rxy) > MULT16_16(Ryy, best_num))
```
Both sides must fit in 32 bits. The `rshift` computation ensures this, but the Rust port must verify the same shift logic or risk overflow with different integer promotion rules.

### 10.12 `opus_unlikely` Hint

```c
if (opus_unlikely(MULT16_16(best_den, Rxy) > MULT16_16(Ryy, best_num)))
```
This is a branch prediction hint (`__builtin_expect`). Rust equivalent: `#[cold]` on a helper function or `std::intrinsics::unlikely` (nightly). Not required for correctness; omit for initial port.

### 10.13 Rate Module's Static Tables for Standard Modes

For standard (non-custom) modes, the pulse cache is precomputed at build time and baked into the binary as static data. The `compute_pulse_cache` function is only compiled for `CUSTOM_MODES`. For the Rust port, use `const` or `lazy_static` tables matching the C reference's precomputed values.
