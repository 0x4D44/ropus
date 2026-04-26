Now I have all the information needed. Here is the complete architecture document:

---

# CELT Bands Module — Architecture Documentation

## 1. Purpose

The `bands` module is the heart of CELT's spectral coding pipeline. It sits between the MDCT transform and the entropy coder, handling the critical path of:

1. **Energy computation** — measuring the amplitude (sqrt of energy) in each critical band of the MDCT spectrum
2. **Normalization / denormalization** — scaling each band to unit energy before quantization, and reversing the scaling on the decoder side
3. **Band quantization** — the recursive binary split algorithm that encodes/decodes normalized spectral coefficients using PVQ (Pyramid Vector Quantization), with stereo coupling, time-frequency resolution tradeoffs, and spectral folding for unquantized bands
4. **Anti-collapse** — injecting shaped noise into bands that received zero pulses to prevent audible energy collapse in transient signals
5. **Spreading decision** — analyzing the spectral shape to decide how much inter-band spreading to apply during PVQ search

In the Opus pipeline, bands is called by `celt_encoder.c` and `celt_decoder.c`. It is *the* module that converts between the MDCT-domain signal and the bitstream, via the range coder.

---

## 2. Public API

### 2.1 Bit-Exact Math Utilities

```c
opus_int16 bitexact_cos(opus_int16 x);
```
- **Purpose**: Polynomial cosine approximation designed for bit-exact cross-platform results. Used for mid/side energy splitting during band quantization.
- **Input**: `x` in range [0, 16384], representing angle in units of π/2 (0 = cos(0) = 1, 16384 = cos(π/2) = 0)
- **Output**: cos(x·π/32768) scaled to Q15 range [0, 32767]. Always returns ≥ 1 (never exactly 0).
- **Precision**: Approximation via degree-6 minimax polynomial in Q13/Q15 arithmetic.

```c
int bitexact_log2tan(int isin, int icos);
```
- **Purpose**: Computes `log2(tan(θ))` = `log2(sin/cos)` bit-exactly for mid/side bit allocation delta.
- **Input**: `isin`, `icos` — integer sine/cosine values (from `bitexact_cos`)
- **Output**: Q11 fixed-point `log2(isin/icos)`. Uses `EC_ILOG` for integer part and quadratic polynomial for fractional part.

### 2.2 Energy Computation

```c
void compute_band_energies(const CELTMode *m, const celt_sig *X,
    celt_ener *bandE, int end, int C, int LM, int arch);
```
- **Purpose**: Compute sqrt(energy) for each band of the MDCT spectrum.
- **Parameters**:
  - `m` — Mode configuration (provides `eBands`, `shortMdctSize`, `nbEBands`, `logN`)
  - `X` — MDCT coefficients, interleaved by channel: channel 0 at `X[0..N-1]`, channel 1 at `X[N..2N-1]` where `N = shortMdctSize << LM`
  - `bandE` — Output: sqrt energy per band, layout `bandE[band + channel*nbEBands]`
  - `end` — Number of bands to process (exclusive upper bound)
  - `C` — Number of channels (1 or 2)
  - `LM` — Log2 of number of short MDCT blocks in the frame (0–3)
  - `arch` — Architecture index for SIMD dispatch

### 2.3 Normalization

```c
void normalise_bands(const CELTMode *m, const celt_sig * OPUS_RESTRICT freq,
    celt_norm * OPUS_RESTRICT X, const celt_ener *bandE, int end, int C, int M);
```
- **Purpose**: Divide each band by its energy so the result has unit norm. This converts `celt_sig` (Q27) to `celt_norm` (Q`NORM_SHIFT` = Q24).
- **Parameters**: `freq` = input MDCT coefficients, `X` = output normalized coefficients, `M` = `1 << LM`.
- **Buffer layout**: Same interleaved-by-channel layout as `compute_band_energies`. Band `i` spans `[M*eBands[i], M*eBands[i+1])` within each channel.

```c
void denormalise_bands(const CELTMode *m, const celt_norm * OPUS_RESTRICT X,
    celt_sig * OPUS_RESTRICT freq, const celt_glog *bandLogE,
    int start, int end, int M, int downsample, int silence);
```
- **Purpose**: Multiply normalized coefficients by their energy (in log domain) to reconstruct the MDCT spectrum. Decoder-side inverse of `normalise_bands`.
- **Parameters**:
  - `bandLogE` — Log-domain energy per band (Q`DB_SHIFT` = Q24 in fixed-point). Combined with `eMeans[i]` to get actual gain.
  - `start`, `end` — Band range to process
  - `downsample` — Downsampling factor; bins above `N/downsample` are zeroed
  - `silence` — If nonzero, zero the entire output
- **Key detail**: Uses `celt_exp2_db` to convert log energy to linear gain. Fixed-point path splits into integer shift and fractional `celt_exp2_db_frac`, with overflow capping at `INT32_MAX`.

### 2.4 Spreading Decision

```c
int spreading_decision(const CELTMode *m, const celt_norm *X,
    int *average, int last_decision, int *hf_average,
    int *tapset_decision, int update_hf, int end, int C, int M,
    const int *spread_weight);
```
- **Purpose**: Analyze spectral flatness to choose PVQ pulse spreading mode.
- **Returns**: One of `SPREAD_NONE` (0), `SPREAD_LIGHT` (1), `SPREAD_NORMAL` (2), `SPREAD_AGGRESSIVE` (3).
- **Algorithm**: Computes rough CDF of |x[j]|²·N at thresholds 0.25, 0.0625, 0.015625 (Q13). Accumulates weighted vote per band. Uses recursive averaging with hysteresis against `last_decision`. Also tracks HF energy distribution for tapset selection.
- **State**: `average` and `hf_average` are exponentially-averaged accumulators (updated in-place); `tapset_decision` is output.

### 2.5 Haar Transform

```c
void haar1(celt_norm *X, int N0, int stride);
```
- **Purpose**: In-place length-2 Haar wavelet transform on strided data. Used for time-frequency resolution changes.
- **Operation**: For each pair: `(a,b) → ((a+b)/√2, (a-b)/√2)`. The `1/√2` factor is applied as `MULT32_32_Q31(QCONST32(0.70710678, 31), ·)`.
- **Parameters**: `N0` = number of pairs × 2, `stride` = distance between elements within a pair.

### 2.6 Band Quantization (Main Entry Point)

```c
void quant_all_bands(int encode, const CELTMode *m, int start, int end,
    celt_norm *X, celt_norm *Y, unsigned char *collapse_masks,
    const celt_ener *bandE, int *pulses, int shortBlocks, int spread,
    int dual_stereo, int intensity, int *tf_res, opus_int32 total_bits,
    opus_int32 balance, ec_ctx *ec, int LM, int codedBands,
    opus_uint32 *seed, int complexity, int arch, int disable_inv
    ARG_QEXT(ec_ctx *ext_ec) ARG_QEXT(int *extra_pulses)
    ARG_QEXT(opus_int32 total_ext_bits) ARG_QEXT(const int *cap));
```
- **Purpose**: Top-level function that iterates over all bands, dispatching to mono (`quant_band`) or stereo (`quant_band_stereo`) quantization. Both encode and decode paths share this function.
- **Parameters**:
  - `encode` — 1 for encoder, 0 for decoder
  - `X`, `Y` — Normalized spectra for channels 0, 1 (Y=NULL for mono)
  - `collapse_masks` — Per-band bitmask tracking which sub-blocks received pulses (for anti-collapse)
  - `pulses` — Bit allocation per band (from rate control, in 1/8-bit units)
  - `shortBlocks` — Nonzero if using short MDCT blocks (transient mode)
  - `spread` — PVQ spreading mode
  - `dual_stereo` — Nonzero: independent L/R coding. Zero: mid/side coding
  - `intensity` — Band index at which to switch from dual/MS stereo to intensity stereo
  - `tf_res` — Time-frequency resolution change per band (-1, 0, or 1)
  - `total_bits`, `balance` — Total and residual bit budget (1/8-bit units)
  - `codedBands` — Number of bands that received nonzero bit allocation
  - `seed` — LCG random seed (updated in-place), used for noise fill
  - `disable_inv` — If set, disable side inversion in stereo (needed for downmix compatibility)

### 2.7 Anti-Collapse

```c
void anti_collapse(const CELTMode *m, celt_norm *X_,
    unsigned char *collapse_masks, int LM, int C, int size,
    int start, int end, const celt_glog *logE,
    const celt_glog *prev1logE, const celt_glog *prev2logE,
    const int *pulses, opus_uint32 seed, int encode, int arch);
```
- **Purpose**: Prevent energy collapse in bands that received no pulses during transient frames (multiple short MDCTs). Injects pseudorandom noise scaled by inter-frame energy difference.
- **Parameters**:
  - `collapse_masks` — Per sub-block mask from `quant_all_bands`; bit `k`=0 means sub-block `k` collapsed
  - `logE`, `prev1logE`, `prev2logE` — Current and two previous frames' log energies
  - `size` — Total spectrum size per channel (`M * eBands[nbEBands]`)

### 2.8 Utility

```c
opus_uint32 celt_lcg_rand(opus_uint32 seed);
```
- **Purpose**: Linear congruential generator. `seed = 1664525 * seed + 1013904223`. Knuth's constants.

```c
int hysteresis_decision(opus_val16 val, const opus_val16 *thresholds,
    const opus_val16 *hysteresis, int N, int prev);
```
- **Purpose**: Generic hysteresis-based quantizer. Finds which interval `val` falls into among `N` thresholds, with hysteresis bands that resist change from `prev`.

---

## 3. Internal State

### 3.1 `struct band_ctx`

Per-frame context passed through the recursive band quantization. Allocated on the stack in `quant_all_bands`, populated once, then updated per-band (`i`, `tf_change`, `remaining_bits`).

```c
struct band_ctx {
    int encode;              // 1 = encode, 0 = decode
    int resynth;             // 1 if decoder or encoder with theta_rdo
    const CELTMode *m;       // Mode configuration
    int i;                   // Current band index
    int intensity;           // Band index where intensity stereo begins
    int spread;              // PVQ spreading mode (SPREAD_NONE..AGGRESSIVE)
    int tf_change;           // Time-freq resolution change for current band
    ec_ctx *ec;              // Range coder state
    opus_int32 remaining_bits; // Bits remaining in budget (1/8-bit units)
    const celt_ener *bandE;  // Per-band sqrt energy
    opus_uint32 seed;        // LCG random state
    int arch;                // SIMD architecture selector
    int theta_round;         // Theta RDO direction: -1=round down, 0=normal, 1=round up
    int disable_inv;         // Disable side inversion for downmix safety
    int avoid_split_noise;   // Avoid noise injection on first band of transients
#ifdef ENABLE_QEXT
    ec_ctx *ext_ec;          // Extended bitstream entropy coder
    int extra_bits;          // Extra bits for QEXT
    opus_int32 ext_total_bits;
    int extra_bands;         // Using QEXT extended bands
#endif
};
```

### 3.2 `struct split_ctx`

Output from `compute_theta` — the result of encoding/decoding the split angle parameter.

```c
struct split_ctx {
    int inv;       // Side inversion flag (stereo only)
    int imid;      // Q15 cosine of theta (mid weighting factor)
    int iside;     // Q15 cosine of (π/2 - theta) (side weighting factor)
    int delta;     // Bit allocation delta between mid and side (1/8-bit units)
    int itheta;    // Quantized theta in [0, 16384] (Q14 angle)
#ifdef ENABLE_QEXT
    int itheta_q30; // Higher-precision theta (Q30)
#endif
    int qalloc;    // Bits spent encoding theta (1/8-bit units)
};
```

### 3.3 Stack-Allocated Buffers in `quant_all_bands`

| Buffer | Type | Size | Purpose |
|--------|------|------|---------|
| `_norm` | `celt_norm[]` | `C * (M*eBands[nbEBands-1] - norm_offset)` | Folding memory — stores quantized normalized coefficients for spectral folding of later bands |
| `_lowband_scratch` | `celt_norm[]` | `M * (eBands[nbEBands] - eBands[nbEBands-1])` | Scratch space to avoid clobbering lowband during Haar transforms |
| `X_save`, `Y_save`, `X_save2`, `Y_save2`, `norm_save2` | `celt_norm[]` | Same as scratch | Theta RDO: save/restore buffers for comparing round-up vs round-down |
| `bytes_save` | `unsigned char[]` | 1275 (max Opus packet) | Theta RDO: save/restore range coder buffer bytes |

---

## 4. Algorithm

### 4.1 Energy Computation (`compute_band_energies`)

**Fixed-point path:**
1. For each band `i`, find `maxval = max(|X[j]|)` over the band.
2. Compute a dynamic `shift` to prevent overflow: `shift = max(0, 30 - ilog2(maxval) - sqrt(band_size)/2)`.
3. Accumulate `sum = Σ (X[j] << shift)² >> 31` using `MULT32_32_Q31`.
4. `bandE[i] = max(maxval, sqrt(sum/2) >> shift)`. The `max` guard ensures tiny numerical errors don't underestimate energy.
5. Zero-energy bands get `EPSILON` (= 1).

**Float path:** Simply `bandE[i] = sqrt(1e-27 + inner_prod(X, X))`.

### 4.2 Normalization (`normalise_bands`)

**Fixed-point:** For each band, compute reciprocal of energy via `celt_rcp_norm32`, then multiply each coefficient:
```
X[j] = (freq[j] << shift) * (1/E) >> (30 - NORM_SHIFT)
```
This converts from `celt_sig` (Q27) to `celt_norm` (Q24 = Q`NORM_SHIFT`).

**Float:** Simply `X[j] = freq[j] / bandE[i]`.

### 4.3 Denormalization (`denormalise_bands`)

Converts log-domain energy back to linear gain and multiplies:
1. `lg = bandLogE[i] + eMeans[i] << (DB_SHIFT - 4)` — add per-band mean (eMeans is Q4, result is Q`DB_SHIFT`)
2. Fixed-point: split `lg` into integer part (`shift = 17 - lg >> DB_SHIFT`) and fractional part (via `celt_exp2_db_frac`). Overflow capped at `INT32_MAX`.
3. `freq[j] = (X[j] << (30-NORM_SHIFT)) * g >> shift`
4. Bins beyond `bound = min(M*eBands[end], N/downsample)` are zeroed.

### 4.4 Band Quantization — Recursive Split

The core algorithm is a binary tree of splits, starting from `quant_all_bands` → `quant_band` (mono) or `quant_band_stereo` (stereo) → `quant_partition` (recursive).

#### 4.4.1 `quant_all_bands` — Outer Loop

For each band `i` from `start` to `end`:
1. Compute per-band bit budget `b` from `pulses[i] + balance_adjustment`, clamped to `remaining_bits`.
2. Determine `effective_lowband` — the offset into the `norm` buffer for spectral folding. Uses collapse masks from previously-quantized bands.
3. Apply time-frequency resolution change (`tf_change`).
4. Dispatch to:
   - `quant_band` (mono or post-intensity-stereo)
   - `quant_band_stereo` (stereo with mid/side or dual stereo)
5. With **theta RDO** (complexity ≥ 8, stereo, non-dual, bands below intensity): encode twice with `theta_round = -1` and `+1`, pick the one with better distortion.
6. Store collapse mask, advance balance tracker.
7. `update_lowband` flag: only fold from bands with ≥ 1 bit/sample depth.

#### 4.4.2 `quant_band` — Mono Band Processing

1. **N=1 special case**: Just encode/decode one sign bit (`quant_band_n1`).
2. **Time-frequency transforms**:
   - `tf_change > 0`: Haar recombine (increase frequency resolution). Apply `haar1` to merge pairs.
   - `tf_change < 0`: Haar split (increase time resolution). Apply `haar1` to split into sub-blocks.
3. **Hadamard reordering**: If `B0 > 1` (multiple short blocks), `deinterleave_hadamard` rearranges from frequency order to time order for coding efficiency. On resynth, `interleave_hadamard` reverses this.
4. Call `quant_partition` for the actual PVQ quantization.
5. On resynth, undo transforms in reverse order. Compute `lowband_out = sqrt(N) · X` for later folding.

#### 4.4.3 `quant_partition` — Recursive Quantization

**Split decision**: If bits `b` exceed the maximum useful for this band size (from pulse cache), and `N > 2`:
1. Split band in half: `N >>= 1; Y = X + N; LM -= 1`.
2. Call `compute_theta` to encode/decode the energy split angle between halves.
3. Compute `mid`/`side` scaling from theta.
4. Apply forward-masking-aware bit rebalancing (`delta` adjustment for `B0 > 1`).
5. Split bits: `mbits = max(0, min(b, (b-delta)/2))`, `sbits = b - mbits`.
6. Recurse on both halves. Larger-allocation half goes first; leftover bits from first half rebalance into second (up to 3-bit threshold).

**Leaf case** (no split):
1. Convert bits to pulse count via `bits2pulses`.
2. If `q > 0`: call `alg_quant` (encoder) or `alg_unquant` (decoder) — PVQ coding.
3. If `q == 0` and `resynth`:
   - If `lowband` exists: fold from previously-quantized band + small random perturbation (~-48 dB).
   - If no `lowband`: fill with LCG pseudorandom noise.
   - Renormalize to maintain unit energy.

#### 4.4.4 `compute_theta` — Split Angle Encoding

1. Compute `qn` (quantization levels for theta) from available bits and band size via `compute_qn`.
2. For encoding: compute `itheta_q30 = stereo_itheta(X, Y)` — the angle in Q30.
3. Quantize `itheta` to `qn` levels. With `theta_round != 0` (RDO), bias quantization toward 0 or 16384.
4. **Noise avoidance** (`avoid_split_noise`): if quantized theta would cause one side to get zero bits and inject noise, snap to 0 or `qn` instead.
5. **Entropy coding** — three PDF shapes depending on context:
   - **Stereo, N>2**: Step PDF — probability `p0=3` for itheta ≤ qn/2 (favoring mid-heavy), probability 1 above.
   - **Multiple blocks or stereo**: Uniform PDF over `[0, qn]`.
   - **Single block, mono**: Triangular PDF — peak at edges (0 and qn), minimum at center.
6. Convert quantized `itheta` back to 14-bit range: `itheta = itheta * 16384 / qn`.
7. Compute `imid = bitexact_cos(itheta)`, `iside = bitexact_cos(16384 - itheta)`.
8. Compute bit allocation delta: `delta = (N-1)*128/16384 * bitexact_log2tan(iside, imid)`.
9. For `itheta == 0`: all bits to mid, clear side fill mask. For `itheta == 16384`: all bits to side.

#### 4.4.5 `quant_band_stereo` — Stereo Band Processing

1. **N=1**: Use `quant_band_n1` for both channels.
2. **Low-energy guard**: If either channel's energy < `MIN_STEREO_ENERGY`, copy the stronger to the weaker to avoid numerical issues.
3. Call `compute_theta` with `stereo=1`.
4. **N=2 special case**: Only one bit needed for side orientation (sign of cross-product `x0·y1 - x1·y0`). All other bits go to mid. Side is constructed as orthogonal rotation of mid.
5. **General case**: Split bits between mid and side via delta. Mid is coded first (if it gets more bits) or side first, with bit rebalancing. Side uses no folding (high fill bits cleared).
6. On resynth: apply `stereo_merge` to reconstruct L/R from mid/side. If `inv` flag set, negate side channel.

### 4.5 Anti-Collapse

For each band `i`:
1. Compute `depth = (1 + pulses[i]) / N0 >> LM` — quantization depth in 1/8-bit units.
2. `thresh = 0.5 * exp2(-depth/8)` — threshold below which energy is considered collapsed.
3. `Ediff = logE[i] - min(prev1logE[i], prev2logE[i])` — energy increase from previous frames.
4. `r = 2 * exp2_db(-Ediff)` — noise amplitude, inversely proportional to energy jump.
5. For `LM == 3`: scale `r` by `sqrt(2)` (energy compensation for 8 short blocks).
6. `r = min(thresh, r) * rsqrt(N0 << LM)` — normalize by band size.
7. For each sub-block `k` where `collapse_masks[i] & (1<<k) == 0`:
   - Fill with `±r` using LCG random signs.
8. Renormalize the band after noise injection.

### 4.6 Stereo Processing

**`intensity_stereo`**: Combines two channels into one weighted sum:
```
X[j] = (left/norm) * X[j] + (right/norm) * Y[j]
```
where `left = bandE[i]`, `right = bandE[i + nbEBands]`, `norm = sqrt(left² + right²)`.

**`stereo_split`**: Mid/side transform:
```
(X[j], Y[j]) → ((X+Y)/√2, (Y-X)/√2)
```

**`stereo_merge`**: Inverse of split, with energy correction:
1. Compute `El = mid²/8 + side² - 2*xp` and `Er = mid²/8 + side² + 2*xp` where `xp = inner_prod(Y, X)`.
2. `lgain = rsqrt(El)`, `rgain = rsqrt(Er)`.
3. `X[j] = lgain * (mid*X[j] - Y[j])`, `Y[j] = rgain * (mid*X[j] + Y[j])`.

---

## 5. Data Flow

### 5.1 Encoder Path

```
MDCT coefficients (celt_sig, Q27)
        │
        ▼
compute_band_energies() ──→ bandE[] (celt_ener, sqrt of energy)
        │
        ▼
normalise_bands() ──→ X[] (celt_norm, Q24, unit-energy bands)
        │
        ▼
spreading_decision() ──→ spread mode (0–3)
        │
        ▼
quant_all_bands(encode=1) ──→ bits written to ec_ctx
        │                  ──→ collapse_masks[] updated
        │                  ──→ X[],Y[] resynthesized (if resynth)
        │                  ──→ seed updated
        ▼
anti_collapse() ──→ X[] patched (decoder side only, typically)
```

### 5.2 Decoder Path

```
ec_ctx (bitstream)
        │
        ▼
quant_all_bands(encode=0) ──→ X[],Y[] (celt_norm, Q24)
        │                  ──→ collapse_masks[]
        │                  ──→ seed updated
        ▼
anti_collapse() ──→ X[] patched
        │
        ▼
denormalise_bands() ──→ freq[] (celt_sig, Q27, full-amplitude MDCT)
        │
        ▼
inverse MDCT
```

### 5.3 Buffer Layouts

**Spectrum buffers** (`X`, `Y`, `freq`):
- Size per channel: `N = M * shortMdctSize` where `M = 1 << LM`
- Channel 0: indices `[0, N)`. Channel 1: indices `[N, 2N)`.
- Band `i` within channel `c`: indices `[c*N + M*eBands[i], c*N + M*eBands[i+1])`.

**Energy buffers** (`bandE`, `bandLogE`):
- Size: `C * nbEBands`.
- Band `i`, channel `c`: index `i + c * nbEBands`.

**Collapse masks**:
- Size: `end * C`.
- Band `i`, channel `c`: index `i * C + c`. Each is an unsigned byte bitmask with `B` significant bits (one per short MDCT block).

**Norm buffer** (internal to `quant_all_bands`):
- Two halves: `norm` for channel 0 (or mid), `norm2` for channel 1 (or side).
- Indexed relative to `norm_offset = M * eBands[start]`.
- Band `i` writes to `norm[M*eBands[i] - norm_offset]` through `norm[M*eBands[i+1] - norm_offset - 1]`.

---

## 6. Numerical Details

### 6.1 Fixed-Point Formats

| Type / Value | Q Format | Range | Notes |
|---|---|---|---|
| `celt_sig` | Q27 | ±2³¹ | MDCT coefficients, ~4 bits headroom above ±1.0 |
| `celt_norm` | Q24 (`NORM_SHIFT`) | Nominally unit-energy | Normalized spectrum |
| `celt_ener` | Q0 (integer) | ≥ 1 (`EPSILON`) | Sqrt of band energy |
| `celt_glog` / `bandLogE` | Q24 (`DB_SHIFT`) | Log-domain dB | 1.0 in Q24 = 1/16777216 dB |
| `eMeans` | Q4 | Signed byte | Mean energy per band; divide by 16 for dB |
| `itheta` | Q14 | [0, 16384] | 0 = all-mid, 16384 = all-side |
| `imid`, `iside` | Q15 | [0, 32767] | cos(θ), cos(π/2−θ) |
| `delta` | 1/8-bit (`BITRES=3`) | Signed | Bit allocation shift mid→side |
| Bit budgets (`b`, `pulses`, etc.) | 1/8-bit | ≥ 0 | `BITRES=3`, so `b=8` = 1 bit |
| `gain` (in `quant_partition`) | Q31 | [0, `Q31ONE`] | Accumulated gain through split tree |

### 6.2 Overflow Guards

- **`compute_band_energies` (fixed)**: Dynamic shift before squaring prevents overflow. `shift` derived from `celt_ilog2(maxval)` ensures squared values fit in 32 bits after `MULT32_32_Q31`.
- **`normalise_bands` (fixed)**: Uses `celt_zlog2` (safe for zero) and `celt_rcp_norm32` (normalized reciprocal).
- **`denormalise_bands` (fixed)**: Clamps `shift` to `[0, 30]`. For `shift >= 31`, forces `g = 0`. For `shift < 0` (extreme gain), caps at `INT32_MAX`.
- **`compute_theta`**: Bit budget `b` is clamped to `remaining_bits` per call. After theta coding, `qalloc` is subtracted from `b`.
- **`quant_partition`**: Loop decrements `q` while `remaining_bits < 0` to guarantee budget compliance.
- **`stereo_merge`**: Guards `El` and `Er` against near-zero with threshold `6e-4` (Q28); falls back to copying mid to side.

### 6.3 Precision-Critical Operations

- **`bitexact_cos`**: Must be bit-identical across all platforms. The polynomial:
  ```
  x2 = (4096 + x*x) >> 13
  x2 = (32767 - x2) + FRAC_MUL16(x2, (-7651 + FRAC_MUL16(x2, (8277 + FRAC_MUL16(-626, x2)))))
  return 1 + x2
  ```
  Uses `FRAC_MUL16` (which rounds via `(16384 + a*b) >> 15`) for deterministic rounding.

- **`bitexact_log2tan`**: Normalizes inputs to 15 bits via `EC_ILOG`, then applies quadratic polynomial. The integer part comes from `ls - lc` (log2 of magnitude ratio).

- **`FRAC_MUL16(a, b)`**: `(16384 + (int32)(int16)a * (int16)b) >> 15`. The `+16384` provides round-to-nearest. Critical for bit-exact cos/log2tan.

### 6.4 Rounding Behavior

- `PSHR32(a, shift)` = `(a + (1 << (shift-1))) >> shift` — round-to-nearest (biased toward positive).
- `FRAC_MUL16` — round-to-nearest via midpoint bias.
- `itheta` quantization: `(itheta * qn + 8192) >> 14` — round-to-nearest Q14→integer.
- Bit allocation: `mbits = max(0, min(b, (b-delta)/2))` — integer division truncates toward zero.

---

## 7. Dependencies

### 7.1 Modules Called by `bands`

| Module | Functions Used | Purpose |
|---|---|---|
| `vq` | `alg_quant`, `alg_unquant`, `renormalise_vector`, `stereo_itheta` | PVQ quantization core |
| `cwrs` | (via `alg_quant`/`alg_unquant`) | Combinatorial pulse coding |
| `rate` | `bits2pulses`, `pulses2bits`, `get_pulses` | Pulse ↔ bits conversion |
| `entenc` | `ec_encode`, `ec_enc_uint`, `ec_enc_bits`, `ec_enc_bit_logp` | Range encoder |
| `entdec` | `ec_decode`, `ec_dec_uint`, `ec_dec_bits`, `ec_dec_bit_logp`, `ec_dec_update` | Range decoder |
| `entcode` | `ec_tell_frac`, `EC_ILOG`, `isqrt32` | Entropy coder utilities |
| `mathops` | `celt_ilog2`, `celt_zlog2`, `celt_sqrt`, `celt_sqrt32`, `celt_rsqrt`, `celt_rsqrt_norm32`, `celt_rcp_norm32`, `celt_exp2`, `celt_exp2_db`, `celt_exp2_db_frac`, `celt_udiv`, `celt_sudiv`, `celt_maxabs32`, `celt_inner_prod`, `celt_inner_prod_norm_shift`, `celt_cos_norm32` | Fixed-point math |
| `pitch` | `celt_inner_prod` (via `celt_inner_prod_norm_shift`) | Dot product |
| `quant_bands` | `eMeans` (extern const table) | Mean energy per band |
| `modes` | `CELTMode` struct | Configuration |

### 7.2 Modules That Call `bands`

| Caller | Functions Called |
|---|---|
| `celt_encoder.c` | `compute_band_energies`, `normalise_bands`, `spreading_decision`, `quant_all_bands`, `anti_collapse` |
| `celt_decoder.c` | `denormalise_bands`, `quant_all_bands`, `anti_collapse` |

---

## 8. Constants and Tables

### 8.1 Spread Modes
```c
#define SPREAD_NONE       0  // No pulse spreading (tonal signals)
#define SPREAD_LIGHT      1
#define SPREAD_NORMAL     2  // Default for most signals
#define SPREAD_AGGRESSIVE 3  // Maximum spreading (noise-like signals)
```

### 8.2 `ordery_table` (Hadamard Reordering)
```c
static const int ordery_table[] = {
     1,  0,                                         // N=2
     3,  0,  2,  1,                                 // N=4
     7,  0,  4,  3,  6,  1,  5,  2,                 // N=8
    15,  0,  8,  7, 12,  3, 11,  4,
    14,  1,  9,  6, 13,  2, 10,  5,                 // N=16
};
```
Bit-reversed Gray code with DC moved to end. Indexed as `ordery_table + stride - 2`. Used for Hadamard transform ordering to decorrelate short MDCT blocks.

### 8.3 `bit_interleave_table` / `bit_deinterleave_table`
```c
// Recombine: maps 4-bit fill mask through interleaving
static const unsigned char bit_interleave_table[16] = {
    0,1,1,1,2,3,3,3,2,3,3,3,2,3,3,3
};
// Undo: expands 4-bit mask back to 8-bit
static const unsigned char bit_deinterleave_table[16] = {
    0x00,0x03,0x0C,0x0F,0x30,0x33,0x3C,0x3F,
    0xC0,0xC3,0xCC,0xCF,0xF0,0xF3,0xFC,0xFF
};
```

### 8.4 `exp2_table8` (in `compute_qn`)
```c
static const opus_int16 exp2_table8[8] =
    {16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048};
```
Q14 values of `2^(k/8)` for `k = 0..7`. Used to compute the number of quantization levels `qn` from fractional-bit budget.

### 8.5 LCG Constants
```c
seed = 1664525 * seed + 1013904223;
```
Knuth's multiplicative LCG. Full 32-bit period (2³²).

### 8.6 `eMeans[25]`
Mean log energy per band (Q4 fixed-point, 25 entries for maximum 25 bands). Added back during denormalization to convert differential energy to absolute.

### 8.7 Theta Coding Constants
- `QTHETA_OFFSET = 4` — default offset for theta bit budget computation
- `QTHETA_OFFSET_TWOPHASE = 16` — offset for stereo N=2 two-phase case
- `p0 = 3` — probability weight for stereo theta step PDF

### 8.8 Key Magic Numbers
- `0.70710678f` = `1/√2` — Haar/stereo split scaling factor
- `23170` (Q14) = `√2` in Q14 — anti-collapse LM=3 scaling
- `-2597`, `7932` — polynomial coefficients in `bitexact_log2tan`
- `-7651`, `8277`, `-626` — polynomial coefficients in `bitexact_cos`
- `48 dB` below folding: `QCONST16(1.0f/256, NORM_SHIFT-4)` — noise floor for spectral folding

---

## 9. Edge Cases

### 9.1 Error/Boundary Conditions

- **Empty band (N=0)**: Assertion `N > 0` in `quant_all_bands`. Cannot happen with valid mode tables.
- **Zero energy**: `compute_band_energies` returns `EPSILON` (=1). `normalise_bands` adds `EPSILON` if `E < 10` to prevent divide-by-zero / overflow.
- **Silence**: `denormalise_bands` with `silence=1` zeros the entire output and skips all processing.
- **Zero bit budget**: `quant_partition` with `q=0` falls through to spectral folding or noise fill. Never produces silence — always fills with something.
- **Budget overrun**: While loop in `quant_partition` decrements `q` until `remaining_bits >= 0`. This is a hard guarantee.
- **Negative shift in denormalize**: Caps gain at `INT32_MAX` and sets `shift=0`. Only triggers on corrupted bitstreams.
- **Single-sample band (N=1)**: `quant_band_n1` encodes one sign bit. No recursion.
- **Near-zero stereo energy**: `stereo_merge` copies mid to side when `El` or `Er` < `6e-4` (Q28).
- **Low stereo energy**: When either channel's energy < `MIN_STEREO_ENERGY` (2 fixed, 1e-10 float), the stronger channel is copied to the weaker before MS transform.
- **Downsampling**: `denormalise_bands` zeroes bins above `N/downsample`.
- **QEXT guard**: `itheta_q30` clamped to `[0, 1073741824]` (= `[0, 2^30]`) to handle corrupted extended bitstreams.

### 9.2 Conditional Compilation Guards

- `FIXED_POINT` — Selects between fixed-point and float implementations of `compute_band_energies`, `normalise_bands`, `denormalise_bands`, `anti_collapse`, `intensity_stereo`, `stereo_merge`, `compute_theta`.
- `ENABLE_QEXT` — Opus 1.5+ extended quantization: higher-precision theta (`itheta_q30`), cubic quantization paths, extra bits, extended entropy coder.
- `RESYNTH` — Forces `resynth=1` even in encoder (for analysis).
- `DISABLE_UPDATE_DRAFT` — Legacy compatibility: changes folding offset logic and `special_hybrid_folding`.
- `MEASURE_NORM_MSE` — Debug: `measure_norm_mse` function (declared in header, not in main source).
- `FUZZING` — Randomizes spreading decision for fuzz testing.

---

## 10. Porting Notes for Rust

### 10.1 Dual Fixed-Point / Float Code Paths

Nearly every function has `#ifdef FIXED_POINT` blocks with completely different implementations. Strategy options:
- **Two separate implementations** behind a trait or generic parameter.
- **Compile-time feature flag** (`cfg(feature = "fixed_point")`) — simpler, matches C model.
- The fixed-point path is required for bit-exact compliance. The float path is useful for testing and optional.

### 10.2 Macro-Heavy Arithmetic

All arithmetic uses macros (`SHL32`, `MULT32_32_Q31`, `FRAC_MUL16`, etc.) that expand differently for fixed vs float. In Rust:
- Define a `FixedPoint` trait with methods for each operation, or
- Use inline functions / const generics, or
- Write two concrete implementations.
- **Critical**: `FRAC_MUL16` rounding behavior (`+16384 >> 15`) must be exactly reproduced.

### 10.3 Stack Allocation (`VARDECL` / `ALLOC`)

The C code uses `VARDECL`/`ALLOC` macros that expand to either VLAs, `alloca`, or malloc/free. In Rust:
- Use `Vec` with pre-allocated capacity, or
- Use `SmallVec` / `arrayvec` for bounded sizes, or
- Stack-allocate with known upper bounds (the band sizes are bounded by mode tables).
- Maximum allocation: `2 * M * eBands[nbEBands-1]` elements of `celt_norm` (e.g., 2 × 8 × 648 = 10368 i32s = ~40 KB).

### 10.4 In-Place Mutation and Pointer Arithmetic

- `quant_partition` recurses with `Y = X + N` — splitting a buffer in half. In Rust, use `split_at_mut(N)`.
- `haar1` operates on strided data — express as explicit index math rather than pointer arithmetic.
- `deinterleave_hadamard` and `interleave_hadamard` allocate temporary buffers and copy back — straightforward in Rust.
- `norm` buffer indexing uses `norm_offset` subtraction — consider representing as a slice with adjusted base.

### 10.5 Encode/Decode Unification

Both encode and decode share the same functions, branching on `ctx->encode`. This pattern works well in Rust too. Consider:
- An `enum Direction { Encode, Decode }` and match on it, or
- A `const ENCODE: bool` generic parameter for compile-time specialization.

### 10.6 `OPUS_RESTRICT`

The `OPUS_RESTRICT` qualifier (`__restrict__`) tells the C compiler that pointers don't alias. In Rust, `&mut` references already guarantee non-aliasing. No special handling needed, but be careful with the `X`/`Y` pattern where both are mutable sub-slices of the same buffer.

### 10.7 `band_ctx` Mutable State

`band_ctx` has fields modified during recursion (`remaining_bits`, `seed`, `avoid_split_noise`, `tf_change`). In Rust, pass as `&mut BandCtx`. The theta RDO save/restore pattern (copying and restoring the entire struct) maps to `Clone` + reassignment.

### 10.8 Theta RDO Save/Restore

The encoder saves the range coder state (`ec_save = *ec`), encodes with two different theta roundings, and restores the better one. This involves:
- Copying the `ec_ctx` struct (including its buffer pointer)
- Copying byte ranges from the range coder buffer
- Restoring everything

In Rust, the `ec_ctx` will need `Clone`. The byte buffer save/restore is a `copy_from_slice` pattern.

### 10.9 LCG State Threading

The LCG `seed` is threaded through `band_ctx`, mutated during noise fill and anti-collapse, and written back at the end of `quant_all_bands`. In Rust, this is simply a `&mut u32` or a field on the context struct.

### 10.10 Integer Overflow Behavior

C relies on unsigned overflow wrapping (defined) and signed overflow being UB (but practically wrapping on most platforms). The LCG uses unsigned wrapping multiplication. In Rust:
- Use `u32::wrapping_mul` and `u32::wrapping_add` for the LCG.
- For `MULT32_32_Q31` and similar, ensure the Rust implementation handles the full range without panicking in debug mode. Consider `wrapping_*` or explicit casts to `i64` for intermediate products.

### 10.11 Bit-Exact Requirements

- `bitexact_cos` and `bitexact_log2tan` must produce identical output on all platforms. The C code achieves this through careful use of only 16/32-bit integer arithmetic with explicit rounding. The Rust port must reproduce every intermediate value exactly.
- `FRAC_MUL16(a, b)` = `((16384 + (a as i32) * (b as i32)) >> 15) as i16` — pay attention to sign extension and the cast from `i32` to `i16`.

### 10.12 `ENABLE_QEXT` Conditional Compilation

The QEXT extension adds ~15% more code to several functions. Recommend using `#[cfg(feature = "qext")]` in Rust. The `ARG_QEXT(arg)` macro pattern becomes optional function parameters or separate function signatures behind the feature flag.

### 10.13 `isqrt32` Usage

In `compute_theta` (triangular PDF decoding), `isqrt32` is called with `8*fm+1`. This is defined in `entcode.h`/`entcode.c` as an integer square root. Will need to port or use a Rust integer sqrt crate.

### 10.14 Division Operations

`celt_udiv` and `celt_sudiv` are optimized unsigned/signed integer divisions (may use lookup tables or compiler intrinsics). In Rust, standard `/` and `%` operators suffice, but verify that the C implementation's floor-toward-zero behavior matches Rust's.

### 10.15 Function Call Graph Summary

```
quant_all_bands
├── quant_band (mono)
│   ├── quant_band_n1 (N=1)
│   ├── haar1
│   ├── deinterleave_hadamard / interleave_hadamard
│   └── quant_partition (recursive)
│       ├── compute_theta
│       │   ├── compute_qn
│       │   ├── stereo_itheta (vq.c)
│       │   ├── bitexact_cos
│       │   ├── bitexact_log2tan
│       │   ├── intensity_stereo
│       │   ├── stereo_split
│       │   └── ec_encode / ec_decode / ec_enc_uint / ec_dec_uint
│       ├── bits2pulses / pulses2bits / get_pulses (rate.h)
│       ├── alg_quant / alg_unquant (vq.c)
│       ├── renormalise_vector (vq.c)
│       └── celt_lcg_rand
├── quant_band_stereo
│   ├── quant_band_n1 (N=1)
│   ├── compute_theta
│   ├── compute_channel_weights
│   ├── quant_band (for each of mid/side)
│   └── stereo_merge
└── special_hybrid_folding
```
