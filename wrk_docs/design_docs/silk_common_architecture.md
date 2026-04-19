Now I have everything needed. Here is the complete documentation.

---

# SILK Common Module — Architecture Documentation

## 1. Purpose

The **silk_common** module provides the foundational lookup tables, utility functions, and signal processing primitives shared by the SILK encoder and decoder within the Opus codec. SILK handles speech signals at narrowband (NB, 8 kHz), medium-band (MB, 12 kHz), and wideband (WB, 16 kHz) sample rates.

This module contains no codec state machines; rather, it provides the **stateless building blocks** that every other SILK module depends on:

- **Entropy coding tables**: iCDF (inverse cumulative distribution function) tables for the range coder, covering gains, pitch lags, pulses, NLSF coefficients, signal types, and signs.
- **Codebook tables**: Vector-quantization codebooks for LTP (Long-Term Prediction) filter taps and NLSF (Normalized Line Spectral Frequency) coefficients.
- **Resampler filter coefficients**: Polyphase IIR/FIR filter coefficients for sample-rate conversion.
- **Mathematical utilities**: Fixed-point logarithm/exponentiation, sigmoid, sorting, inner products, energy computation, bandwidth expansion, LPC stability analysis, and interpolation.

These are the lowest-dependency components in the SILK hierarchy and must be ported first.

---

## 2. Public API

### 2.1 Sorting Functions (`sort.c`)

```c
void silk_insertion_sort_increasing(
    opus_int32  *a,     // I/O  Values to sort (modified in-place)
    opus_int    *idx,   // O    Original indices of the sorted elements
    const opus_int L,   // I    Total vector length
    const opus_int K    // I    Number of top-K positions to guarantee sorted
);
```
Partial insertion sort: guarantees the first K elements of `a` are the K smallest values from the original array, in ascending order. Remaining elements L-K are unsorted. `idx` tracks original positions. Used heavily in pitch search and VQ codebook search.

```c
void silk_insertion_sort_decreasing_int16(
    opus_int16  *a,     // I/O  Values to sort (modified in-place)
    opus_int    *idx,   // O    Original indices of sorted elements
    const opus_int L,   // I    Total vector length
    const opus_int K    // I    Number of correctly sorted positions
);
```
Same as above but **descending** order, operating on `int16` values. **Fixed-point build only** (`#ifdef FIXED_POINT`).

```c
void silk_insertion_sort_increasing_all_values_int16(
    opus_int16  *a,     // I/O  Vector to sort in-place
    const opus_int L    // I    Vector length
);
```
Full in-place insertion sort, ascending, no index tracking. Used for sorting NLSF coefficients.

### 2.2 Sigmoid Approximation (`sigm_Q15.c`)

```c
opus_int silk_sigm_Q15(
    opus_int in_Q5      // I  Input in Q5 fixed-point
);
```
Returns sigmoid(x) in Q15 format (range [0, 32767] representing [0.0, 1.0)). Uses piecewise-linear interpolation over three 6-entry lookup tables (positive values, negative values, slopes). Input is clipped at ±6.0 (±192 in Q5).

### 2.3 Linear-to-Log Conversion (`lin2log.c`)

```c
opus_int32 silk_lin2log(
    const opus_int32 inLin  // I  Input in linear scale (must be > 0)
);
```
Computes `128 * log2(inLin)` using a piecewise-parabolic approximation. The output is in a Q7-like log domain: the integer part encodes the bit position of the MSB, and the fractional part provides 7 bits of sub-bit precision. The magic constant `179` in the parabolic correction term is derived from a least-squares fit to `log2(1+x)` over `x ∈ [0,1]`.

### 2.4 Log-to-Linear Conversion (`log2lin.c`)

```c
opus_int32 silk_log2lin(
    const opus_int32 inLog_Q7  // I  Input on log scale (Q7)
);
```
Computes `2^(inLog_Q7 / 128)`, the near-exact inverse of `silk_lin2log`. Returns 0 for negative inputs, `silk_int32_MAX` for inputs ≥ 3967 (to avoid overflow). Uses a branch at `inLog_Q7 < 2048` to switch between two computation paths that differ only in overflow-safe intermediate arithmetic. The magic constant `-174` is the inverse-function counterpart of `179`.

### 2.5 Bandwidth Expander — 16-bit (`bwexpander.c`)

```c
void silk_bwexpander(
    opus_int16  *ar,            // I/O  AR filter coefficients (Q12, no leading 1)
    const opus_int d,           // I    Filter order
    opus_int32 chirp_Q16        // I    Chirp factor in Q16 (typically 0.0–1.0)
);
```
Applies bandwidth expansion (chirp) to an LP AR filter: `ar[i] *= chirp^(i+1)`. The chirp factor decays geometrically each iteration. **Critical note**: uses `silk_RSHIFT_ROUND(silk_MUL(...), 16)` instead of `silk_SMULWB` to avoid bias that can cause filter instability.

### 2.6 Bandwidth Expander — 32-bit (`bwexpander_32.c`)

```c
void silk_bwexpander_32(
    opus_int32  *ar,            // I/O  AR filter coefficients (Q domain varies)
    const opus_int d,           // I    Filter order
    opus_int32 chirp_Q16        // I    Chirp factor in Q16
);
```
Same algorithm as `silk_bwexpander` but operates on `opus_int32` coefficients using `silk_SMULWW` (which does a 32×32→upper-32-bit multiply). Used by `silk_LPC_fit`.

### 2.7 Inner Product with Scaling (`inner_prod_aligned.c`)

```c
opus_int32 silk_inner_prod_aligned_scale(
    const opus_int16 *const inVec1, // I  First input vector
    const opus_int16 *const inVec2, // I  Second input vector
    const opus_int scale,           // I  Right-shift applied per product
    const opus_int len              // I  Vector length
);
```
Computes `Σ (inVec1[i] * inVec2[i]) >> scale`, accumulated into a 32-bit result. The per-element right-shift prevents overflow when summing many 16×16 products.

### 2.8 Sum of Squares with Adaptive Shift (`sum_sqr_shift.c`)

```c
void silk_sum_sqr_shift(
    opus_int32  *energy,    // O  Σ(x[i]²) >> shift
    opus_int    *shift,     // O  Number of right-shift bits applied
    const opus_int16 *x,   // I  Input signal
    opus_int len            // I  Length
);
```
Computes the energy (sum of squared samples) of a signal vector, automatically determining the minimum right-shift needed to fit the result in a signed 32-bit integer with 2 bits of headroom. Uses a two-pass approach:
1. First pass with a conservative shift to estimate the dynamic range.
2. Second pass with the refined shift for the exact result.

Processes samples in pairs using `silk_SMLABB_ovflw` for efficiency.

### 2.9 LPC Inverse Prediction Gain (`LPC_inv_pred_gain.c`)

```c
opus_int32 silk_LPC_inverse_pred_gain_c(
    const opus_int16 *A_Q12,    // I  LPC coefficients in Q12
    const opus_int order        // I  Filter order (≤ SILK_MAX_ORDER_LPC)
);
```
Returns the inverse prediction gain in Q30, or **0** if the filter is unstable (any pole outside the unit circle). This is the primary stability check for LPC filters throughout SILK. Internally converts Q12 → Q24 and runs the Levinson-Durbin recursion in reverse, peeling off reflection coefficients from highest to lowest order.

The internal function:
```c
static opus_int32 LPC_inverse_pred_gain_QA_c(
    opus_int32 A_QA[SILK_MAX_ORDER_LPC],  // I  Coefficients in Q24 (modified in-place!)
    const opus_int order
);
```

### 2.10 LPC Coefficient Fitting (`LPC_fit.c`)

```c
void silk_LPC_fit(
    opus_int16  *a_QOUT,    // O  Output coefficients in QOUT
    opus_int32  *a_QIN,     // I/O Input coefficients in QIN (modified!)
    const opus_int QOUT,    // I  Output Q-domain
    const opus_int QIN,     // I  Input Q-domain
    const opus_int d        // I  Filter order
);
```
Converts 32-bit LPC coefficients to 16-bit with overflow protection. Iteratively applies bandwidth expansion (up to 10 rounds) to shrink coefficients that exceed `int16` range. If 10 iterations don't suffice, hard-clips via `silk_SAT16`.

### 2.11 Vector Interpolation (`interpolate.c`)

```c
void silk_interpolate(
    opus_int16  xi[MAX_LPC_ORDER],      // O  Result
    const opus_int16 x0[MAX_LPC_ORDER], // I  First vector
    const opus_int16 x1[MAX_LPC_ORDER], // I  Second vector
    const opus_int ifact_Q2,            // I  Weight on x1 in Q2 [0..4]
    const opus_int d                    // I  Number of elements
);
```
Linear interpolation: `xi[i] = x0[i] + ((x1[i] - x0[i]) * ifact_Q2) >> 2`. Used to interpolate between LPC coefficient sets for sub-frame boundaries. `ifact_Q2` ranges from 0 (pure x0) to 4 (pure x1).

---

## 3. Internal State / Structs

### 3.1 `silk_NLSF_CB_struct`

The core codebook structure for Normalized Line Spectral Frequencies, used for both NB/MB and WB:

```c
typedef struct {
    const opus_int16 nVectors;           // Number of first-stage codebook vectors (32)
    const opus_int16 order;              // LPC order (10 for NB/MB, 16 for WB)
    const opus_int16 quantStepSize_Q16;  // Quantization step size in Q16
    const opus_int16 invQuantStepSize_Q6;// Inverse of above in Q6
    const opus_uint8 *CB1_NLSF_Q8;      // Stage-1 codebook vectors [nVectors * order], Q8
    const opus_int16 *CB1_Wght_Q9;      // Stage-1 weights [nVectors * order], Q9
    const opus_uint8 *CB1_iCDF;         // Stage-1 iCDF [2 * nVectors]
    const opus_uint8 *pred_Q8;          // Predictor coefficients for stage-2, Q8
    const opus_uint8 *ec_sel;           // Entropy coding selector [nVectors * (order/2)]
    const opus_uint8 *ec_iCDF;          // Stage-2 entropy coding iCDF
    const opus_uint8 *ec_Rates_Q5;      // Stage-2 bit rates in Q5
    const opus_int16 *deltaMin_Q15;     // Minimum spacing between NLSFs, Q15
} silk_NLSF_CB_struct;
```

Two global instances are defined:
- `silk_NLSF_CB_NB_MB` — order=10, quantStepSize=0.18 (Q16 = 11796), 320-byte codebook
- `silk_NLSF_CB_WB` — order=16, quantStepSize=0.15 (Q16 = 9830), 512-byte codebook

### 3.2 `silk_resampler_state_struct`

```c
typedef struct {
    opus_int32 sIIR[SILK_RESAMPLER_MAX_IIR_ORDER];  // IIR filter state (6 max)
    union {
        opus_int32 i32[SILK_RESAMPLER_MAX_FIR_ORDER]; // FIR state as int32
        opus_int16 i16[SILK_RESAMPLER_MAX_FIR_ORDER]; // FIR state as int16
    } sFIR;
    opus_int16 delayBuf[96];          // Input delay buffer
    opus_int resampler_function;      // Selects resampling path
    opus_int batchSize;               // Samples per batch
    opus_int32 invRatio_Q16;          // Inverse of rate ratio, Q16
    opus_int FIR_Order;               // Active FIR length
    opus_int FIR_Fracs;               // Number of FIR phases
    opus_int Fs_in_kHz;               // Input rate
    opus_int Fs_out_kHz;              // Output rate
    opus_int inputDelay;              // Fractional delay compensation
    const opus_int16 *Coefs;          // Pointer to filter coefficients
} silk_resampler_state_struct;
```

The `Coefs` pointer points into the `resampler_rom.c` tables at runtime.

---

## 4. Algorithms

### 4.1 Sigmoid Approximation (`silk_sigm_Q15`)

1. Take absolute value of input; determine sign.
2. Clip to range [0, 192) (i.e., [0.0, 6.0) in Q5).
3. Compute table index `ind = in_Q5 >> 5` (integer part).
4. Compute fractional part `frac = in_Q5 & 0x1F`.
5. Interpolate: `LUT[ind] ± slope[ind] * frac`.
   - Positive input: `sigm_LUT_pos_Q15[ind] + sigm_LUT_slope_Q10[ind] * frac`
   - Negative input: `sigm_LUT_neg_Q15[ind] - sigm_LUT_slope_Q10[ind] * frac`

The slope table is the first difference of the LUT values scaled to Q10. The LUT values themselves are `round(32767 * sigmoid(k))` for k=0..5.

### 4.2 Lin2Log / Log2Lin Pair

**`silk_lin2log`** computes `128 * log2(x)`:
1. `silk_CLZ_FRAC(inLin, &lz, &frac_Q7)` — counts leading zeros; `frac_Q7` = 7 MSBs below the leading 1.
2. Integer part: `(31 - lz) << 7`
3. Fractional correction (piecewise parabola): `frac_Q7 + ((frac_Q7 * (128 - frac_Q7) * 179) >> 16)`
4. Sum them: `((31 - lz) << 7) + corrected_frac`

**`silk_log2lin`** computes `2^(x/128)`:
1. Integer part: `1 << (inLog_Q7 >> 7)`
2. Fractional part: `frac_Q7 = inLog_Q7 & 0x7F`
3. Parabolic correction with constant `-174` (inverse of 179)
4. Two code paths for `inLog_Q7 < 2048` vs `≥ 2048` to prevent intermediate overflow

These are near-exact inverses; the parabolic coefficients 179 and -174 are chosen so that `log2lin(lin2log(x)) ≈ x` to within ±1 LSB for the full int32 range.

### 4.3 Bandwidth Expansion (Chirp)

Both `silk_bwexpander` and `silk_bwexpander_32` implement:
```
for i in 0..d-1:
    ar[i] = ar[i] * chirp^(i+1)
```
Rather than computing `chirp^(i+1)` via repeated exponentiation, they maintain a running `chirp_Q16` value and decay it each iteration:
```c
chirp_Q16 += silk_RSHIFT_ROUND(silk_MUL(chirp_Q16, chirp_minus_one_Q16), 16);
```
where `chirp_minus_one_Q16 = chirp_Q16 - 65536` (i.e., `chirp - 1.0`). This is equivalent to `chirp_Q16 *= chirp` because `chirp + chirp*(chirp-1) = chirp^2`.

### 4.4 Sum of Squares with Adaptive Shift

Two-pass algorithm:
1. **Pass 1 (estimation):** Compute with a conservative initial shift of `31 - CLZ32(len)` (the minimum shift that prevents overflow from adding `len` maximum-magnitude terms). Start with `nrg = len` as a rounding bias.
2. **Refine shift:** After pass 1, compute the actual required shift as `max(0, shft + 3 - CLZ32(nrg))` where the `+3` ensures 2 bits of headroom plus a guard bit.
3. **Pass 2 (exact):** Recompute with the refined shift.

Samples are processed in pairs using `silk_SMLABB_ovflw` which computes `a + b*c` with potential unsigned overflow (safe because intermediate sum of two squared int16 values fits in uint32).

### 4.5 LPC Inverse Prediction Gain

Reverse Levinson-Durbin recursion to extract reflection coefficients from AR coefficients:

```
for k = order-1 downto 1:
    1. Check |A[k]| < A_LIMIT (0.99975 in Q24)
    2. rc = -A[k] << (31 - QA)           // reflection coefficient Q31
    3. rc_mult1 = 1.0 - rc²              // Q30
    4. invGain *= rc_mult1                // accumulate inverse gain
    5. If invGain < 1/MAX_PREDICTION_POWER_GAIN: return 0 (unstable)
    6. rc_mult2 = 1/rc_mult1             // variable-Q inverse
    7. For each symmetric pair (n, k-n-1):
       A[n]     = (A[n]     - A[k-n-1] * rc) / rc_mult1
       A[k-n-1] = (A[k-n-1] - A[n]     * rc) / rc_mult1
```

The function also short-circuits if the DC response `Σ A_Q12[k] ≥ 4096` (i.e., ≥ 1.0 in Q12), which means the filter has a pole at DC.

### 4.6 LPC Coefficient Fitting

Iterative clamping loop (max 10 iterations):
1. Find `maxabs = max(|a_QIN[k]|)` and its index.
2. Convert to output domain: `maxabs >>= (QIN - QOUT)`.
3. If `maxabs > 32767`:
   - Compute a chirp factor that would bring `maxabs` just under the limit.
   - Apply `silk_bwexpander_32(a_QIN, d, chirp_Q16)`.
   - Repeat.
4. If 10 iterations exhausted, hard-clip all coefficients via `silk_SAT16`.
5. Otherwise, round-shift to output Q-domain.

### 4.7 Partial Insertion Sort

The `silk_insertion_sort_increasing` function maintains only the first K positions in sorted order:
1. Standard insertion sort for elements 0..K-1.
2. For elements K..L-1: only insert if `value < a[K-1]` (smaller than current K-th smallest). This gives O(K*L) worst case instead of O(L²), which matters when K << L.

---

## 5. Data Flow

### 5.1 Table Data Flow

```
Bitstream
   │
   ▼
Range Decoder ◄── iCDF tables (gain, pitch, pulses, NLSF, LTP, type, sign, ...)
   │
   ├─► Gain indices      ─► silk_gain_iCDF, silk_delta_gain_iCDF
   ├─► Pitch lag indices  ─► silk_pitch_lag_iCDF, silk_pitch_contour_iCDF
   ├─► Pulse counts       ─► silk_pulses_per_block_iCDF, silk_shell_code_table*
   ├─► NLSF indices       ─► silk_NLSF_CB_NB_MB / silk_NLSF_CB_WB
   ├─► LTP indices        ─► silk_LTP_gain_iCDF_ptrs, silk_LTP_vq_ptrs_Q7
   └─► Signal type/offset ─► silk_type_offset_VAD_iCDF, silk_type_offset_no_VAD_iCDF
```

### 5.2 Utility Function Data Flow

```
LPC coefficients (Q12) ──► silk_LPC_inverse_pred_gain_c ──► inverse gain (Q30) or 0 (unstable)
                        ──► silk_bwexpander              ──► chirped coefficients (Q12)
                        ──► silk_LPC_fit                 ──► clamped coefficients (Q_OUT)
                        ──► silk_interpolate              ──► interpolated coefficients

Signal samples (int16)  ──► silk_sum_sqr_shift           ──► (energy, shift) pair
                        ──► silk_inner_prod_aligned_scale ──► scaled inner product (int32)

Scalar values           ──► silk_lin2log / silk_log2lin   ──► log/linear domain conversion
                        ──► silk_sigm_Q15                 ──► sigmoid output (Q15)
```

### 5.3 Buffer Layouts

**NLSF Codebook Stage-1** (`CB1_NLSF_Q8`):
- Layout: `[nVectors][order]` stored as flat array
- NB/MB: 32 × 10 = 320 bytes; WB: 32 × 16 = 512 bytes
- Each row is a codebook vector in Q8 (values 0–255 representing 0.0–1.0 in normalized frequency)

**NLSF Codebook Stage-1 Weights** (`CB1_Wght_Q9`):
- Same layout as CB1, stored as `int16`
- Perceptual weighting for distortion measure

**Shell Code Tables** (`silk_shell_code_table0..3`):
- 152 bytes each, indexed via `silk_shell_code_table_offsets`
- Offsets array gives start position for each split level (0..16)

**LTP VQ Tables** (`silk_LTP_gain_vq_0/1/2`):
- 3 codebook tiers with 8/16/32 entries, each entry is 5 taps (`opus_int8`)
- Taps are in Q7 format
- Pointer-to-pointer arrays (`silk_LTP_vq_ptrs_Q7`) index into these

**Resampler Coefficients**:
- Format: `[2 IIR coefficients, N × (FIR_ORDER/2) FIR coefficients]` per ratio
- First 2 entries are IIR section numerator/denominator parameters
- Remaining entries are half-band FIR coefficients (exploit symmetry)

---

## 6. Numerical Details

### 6.1 Q-Format Reference

| Symbol / Table | Q-Format | Range | Notes |
|---|---|---|---|
| `silk_stereo_pred_quant_Q13` | Q13 | [-13732, 13732] | ≈ [-1.68, 1.68] |
| iCDF tables (all `_iCDF`) | Q0 uint8 | [0, 255] | Inverse CDF for range coder |
| `silk_Quantization_Offsets_Q10` | Q10 | Defined by OFFSET_*_Q10 | Quantization offsets |
| `silk_LTPScales_table_Q14` | Q14 | [8192, 15565] | ≈ [0.5, 0.95] |
| `silk_Transition_LP_B_Q28` | Q28 | ~[35M, 251M] | IIR numerator coefficients |
| `silk_Transition_LP_A_Q28` | Q28 | ~[35M, 506M] | IIR denominator coefficients |
| `silk_NLSF_CB1_*_Q8` | Q8 | [0, 255] | NLSF codebook values (normalized freq) |
| `silk_NLSF_CB1_Wght_Q9` | Q9 int16 | [1838, 5227] | Perceptual weights |
| `silk_NLSF_PRED_*_Q8` | Q8 | [59, 198] | Inter-coefficient prediction |
| `silk_NLSF_DELTA_MIN_*_Q15` | Q15 | [3, 461] | Min NLSF spacing |
| `silk_LSFCosTab_FIX_Q12` | Q12 | [-8192, 8192] | cos(ω) lookup table |
| LTP VQ taps | Q7 int8 | [-36, 124] | 5-tap FIR filter coefficients |
| LTP VQ gains | Q7 uint8 | [2, 173] | max(|H(ω)|) per codebook entry |
| Sigmoid LUT values | Q15 | [219, 32548] | 1/(1+e^(-x)) for x=0..5 |
| Sigmoid slopes | Q10 | [7, 237] | First differences of sigmoid |
| `silk_lin2log` output | Q7-ish | [0, ~3967] | 128 * log2(x) |
| `silk_bwexpander` chirp | Q16 | [0, 65536] | 0.0 to 1.0 |
| `silk_LPC_inverse_pred_gain_c` | Q30 | [0, 2^30] | Inverse gain in energy domain |
| Resampler FIR coefficients | Q15 (approx) | Various | Signed int16 |
| `silk_resampler_frac_FIR_12` | Q15 | [-600, 30567] | 12-phase FIR bank |

### 6.2 Overflow Guards

**`silk_bwexpander`**: Deliberately uses `silk_RSHIFT_ROUND(silk_MUL(), 16)` instead of `silk_SMULWB()` because the rounding bias in `SMULWB` (which truncates rather than rounding) can accumulate and push the filter to instability. This is explicitly noted in the source comments.

**`silk_sum_sqr_shift`**: Two-pass design ensures the final result fits in int32 with 2 bits of headroom. The first pass uses unsigned arithmetic (`silk_ADD_RSHIFT_uint`) because the intermediate sum of two squared int16 values can reach 2 × (32767² + 32768²) ≈ 2.15 × 10⁹ which exceeds int32_MAX but fits in uint32.

**`LPC_inverse_pred_gain_QA_c`**: Uses 64-bit intermediate arithmetic (`silk_SMULL`, `silk_RSHIFT_ROUND64`) for the coefficient update step, then checks for int32 overflow before truncating. Returns 0 (unstable) if overflow occurs.

**`silk_LPC_fit`**: Caps `maxabs` at 163838 = `(silk_int32_MAX >> 14) + silk_int16_MAX` to prevent overflow in the chirp factor computation.

**`silk_log2lin`**: Two code paths:
- `inLog_Q7 < 2048`: Uses `silk_ADD_RSHIFT32(out, silk_MUL(...), 7)` — safe because `out < 2^16`.
- `inLog_Q7 ≥ 2048`: Uses `silk_MLA(out, silk_RSHIFT(out, 7), ...)` — avoids the intermediate left-shift that would overflow.

### 6.3 Rounding Behavior

`silk_RSHIFT_ROUND(a, shift)` implements:
- For shift=1: `(a >> 1) + (a & 1)` — round-half-up
- For shift>1: `((a >> (shift-1)) + 1) >> 1` — round-half-up (biased toward positive infinity)

This rounding is used consistently throughout the module. Bit-exactness requires matching this behavior precisely.

### 6.4 `SILK_FIX_CONST` Macro

```c
#define SILK_FIX_CONST(C, Q) ((opus_int32)((C) * ((opus_int64)1 << (Q)) + 0.5))
```
Converts a floating-point constant to Q-format at compile time. The `+ 0.5` provides round-to-nearest. In Rust, these must be computed as literal constants (e.g., `const` expressions).

---

## 7. Dependencies

### 7.1 What This Module Depends On

| Dependency | Source |
|---|---|
| `opus_int8/16/32`, `opus_uint8/32` | Opus type definitions |
| `silk_CLZ32`, `silk_CLZ_FRAC` | Inline functions from `Inlines.h` |
| `silk_INVERSE32_varQ` | Inline function from `Inlines.h` |
| Fixed-point macros (`silk_MUL`, `silk_SMULBB`, etc.) | `SigProc_FIX.h` / `MacroCount.h` |
| `celt_assert` | CELT assertion macro |
| Constants (`MAX_LPC_ORDER`, `SILK_MAX_ORDER_LPC`, etc.) | `define.h` |
| `MAX_PREDICTION_POWER_GAIN` | `define.h` |

### 7.2 What Depends On This Module

| Consumer | Tables Used | Functions Used |
|---|---|---|
| Range coder (encode/decode) | All `_iCDF` tables | — |
| NLSF encode/decode | `silk_NLSF_CB_NB_MB`, `silk_NLSF_CB_WB` | `silk_insertion_sort_*`, `silk_interpolate` |
| Gain quantization | `silk_gain_iCDF`, `silk_delta_gain_iCDF` | `silk_lin2log`, `silk_log2lin` |
| LTP analysis/synthesis | `silk_LTP_vq_ptrs_Q7`, `silk_LTP_gain_iCDF_ptrs` | — |
| Pitch estimation | `silk_CB_lags_stage*`, `silk_Lag_range_stage3` | `silk_insertion_sort_increasing` |
| Pulse coding (shell coder) | `silk_shell_code_table*`, `silk_sign_iCDF` | — |
| LPC analysis | — | `silk_LPC_inverse_pred_gain_c`, `silk_bwexpander`, `silk_LPC_fit` |
| NSQ (Noise Shaping Quantizer) | — | `silk_sigm_Q15`, `silk_inner_prod_aligned_scale` |
| Resampler | Resampler coefficient tables | — |
| Energy computation (various) | — | `silk_sum_sqr_shift` |
| Sub-frame interpolation | — | `silk_interpolate` |
| Bandwidth transition smoother | `silk_Transition_LP_B/A_Q28` | — |

---

## 8. Constants and Tables

### 8.1 Entropy Coding Tables (iCDF Format)

All entropy coding tables use the **inverse CDF (iCDF)** format expected by the Opus range coder. An iCDF table of length N encodes a distribution over N symbols where:
- `iCDF[0]` = probability of symbol > 0 (scaled to 0..255)
- `iCDF[N-1]` = 0 (always, terminal sentinel)
- Probabilities are implicitly `(iCDF[k-1] - iCDF[k]) / 256`

| Table | Symbols | Distribution Shape |
|---|---|---|
| `silk_gain_iCDF[3][8]` | 64 gain levels (8 per row, 3 coding contexts) | Skewed toward low gains |
| `silk_delta_gain_iCDF[41]` | 41 delta-gain levels (MIN_DELTA_GAIN_QUANT to MAX_DELTA_GAIN_QUANT) | Sharp peak at 0 |
| `silk_pulses_per_block_iCDF[10][18]` | 18 pulse counts, 10 rate levels | Rate-dependent |
| `silk_rate_levels_iCDF[2][9]` | 9 rate levels, 2 signal types | — |
| `silk_pitch_lag_iCDF[32]` | 32 lag values | Uniform-ish |
| `silk_pitch_delta_iCDF[21]` | 21 delta values | Peaked |
| `silk_pitch_contour_iCDF[34]` | 34 contour codes (20ms) | Near-uniform |
| `silk_pitch_contour_NB_iCDF[11]` | 11 contour codes (20ms NB) | — |
| `silk_pitch_contour_10_ms_iCDF[12]` | 12 contour codes (10ms) | — |
| `silk_pitch_contour_10_ms_NB_iCDF[3]` | 3 contour codes (10ms NB) | — |
| `silk_LTP_per_index_iCDF[3]` | 3 LTP periodicity classes | Peaked at class 0 |
| `silk_LTP_gain_iCDF_0/1/2` | 8/16/32 entries per codebook tier | — |
| `silk_stereo_pred_joint_iCDF[25]` | 25 joint stereo codes | — |
| `silk_type_offset_VAD_iCDF[4]` | 4 type+offset combos (with VAD) | — |
| `silk_type_offset_no_VAD_iCDF[2]` | 2 type+offset combos (no VAD) | — |
| `silk_NLSF_interpolation_factor_iCDF[5]` | 5 interpolation factors | — |
| `silk_NLSF_EXT_iCDF[7]` | 7 extension values | Geometric decay |
| `silk_uniform3/4/5/6/8_iCDF` | 3/4/5/6/8 uniform symbols | Exactly uniform |
| `silk_lsb_iCDF[2]` | 2 values (bit) | ~53%/47% split |
| `silk_LTPscale_iCDF[3]` | 3 LTP scale values | — |
| `silk_LBRR_flags_2/3_iCDF` | 2–3 frame LBRR flags | — |
| `silk_sign_iCDF[42]` | 6×7 sign coding (rate × context) | — |

### 8.2 Shell Coding Tables

Four shell code tables (`silk_shell_code_table0` through `3`) of 152 bytes each provide iCDFs for the recursive binary splitting used in shell coding of pulse positions. The tables differ in probability distribution shape (from peaky/unbalanced to near-uniform). The offset table `silk_shell_code_table_offsets[17]` provides the byte offset into each shell code table for a given split level (0–16).

### 8.3 NLSF Codebook Tables

Two-stage vector quantization:
- **Stage 1**: 32 centroid vectors (`CB1_NLSF_Q8`), with perceptual weights (`CB1_Wght_Q9`).
- **Stage 2**: Residual coding with prediction (`pred_Q8`), variable-rate entropy coding (`ec_sel`, `ec_iCDF`, `ec_Rates_Q5`), and minimum spacing constraints (`deltaMin_Q15`).

The `quantStepSize_Q16` values (0.18 for NB/MB, 0.15 for WB) set the residual quantizer step size. Their inverses (`invQuantStepSize_Q6`) are precomputed for the decoder.

### 8.4 LTP Codebook

Three codebook tiers of increasing size and resolution:
- Tier 0: 8 entries × 5 taps (low complexity)
- Tier 1: 16 entries × 5 taps (medium)
- Tier 2: 32 entries × 5 taps (high quality)

Each entry is a 5-tap FIR filter in Q7 applied to the pitch-predicted signal. The companion `silk_LTP_gain_vq_*_gain` tables store `max(|H(e^jω)|)` for each codebook entry, used for gain normalization. The `silk_LTP_vq_sizes[3] = {8, 16, 32}` array gives codebook sizes.

### 8.5 LSF Cosine Table

`silk_LSFCosTab_FIX_Q12[129]` provides `cos(π * i / 128)` in Q12 for i = 0..128:
- `[0]` = 8192 = cos(0) = +1.0
- `[64]` = 0 = cos(π/2) = 0.0
- `[128]` = -8192 = cos(π) = -1.0

Used in the LSF-to-LPC conversion (line spectral frequency evaluation). The table step corresponds to a frequency resolution of `π/128` radians.

### 8.6 Pitch Estimation Tables

Multi-stage pitch search lag offset tables:
- `silk_CB_lags_stage2[4][11]` / `silk_CB_lags_stage3[4][34]` — lag offsets for 20ms frames (4 subframes)
- `silk_CB_lags_stage2_10_ms[2][3]` / `silk_CB_lags_stage3_10_ms[2][12]` — lag offsets for 10ms frames (2 subframes)
- `silk_Lag_range_stage3[3][4][2]` — search range [min, max] per complexity level per subframe
- `silk_nb_cbk_searchs_stage3[3]` — number of codebook searches per complexity level

### 8.7 Resampler Coefficients

Polyphase filter coefficients for various integer-ratio sample rate conversions:

| Table | Ratio | Format |
|---|---|---|
| `silk_Resampler_3_4_COEFS` | 3/4 (e.g., 16→12 kHz) | 2 IIR + 3×(ORDER/2) FIR |
| `silk_Resampler_2_3_COEFS` | 2/3 (e.g., 24→16 kHz) | 2 IIR + 2×(ORDER/2) FIR |
| `silk_Resampler_1_2_COEFS` | 1/2 (e.g., 16→8 kHz) | 2 IIR + ORDER/2 FIR |
| `silk_Resampler_1_3_COEFS` | 1/3 | 2 IIR + ORDER/2 FIR |
| `silk_Resampler_1_4_COEFS` | 1/4 | 2 IIR + ORDER/2 FIR |
| `silk_Resampler_1_6_COEFS` | 1/6 | 2 IIR + ORDER/2 FIR |
| `silk_Resampler_2_3_COEFS_LQ` | 2/3 low quality | 2 IIR + 2×2 FIR |
| `silk_resampler_frac_FIR_12[12][4]` | 12-phase fractional | 4 taps per phase |

All coefficients stored as `opus_int16` with `silk_DWORD_ALIGN` (4-byte alignment). The design specification is: Elliptic/Cauer, 0.1 dB passband ripple, 80 dB stopband attenuation.

### 8.8 Bandwidth Transition Filter Coefficients

`silk_Transition_LP_B_Q28[5][3]` (numerator) and `silk_Transition_LP_A_Q28[5][2]` (denominator) provide 5 interpolation points for a 2nd-order IIR lowpass filter used in bandwidth transition smoothing. Cut-off frequencies range from 0.95 to 0.35 of Nyquist at equal steps of 0.15.

---

## 9. Edge Cases

### 9.1 `silk_sigm_Q15`
- **Input ≥ 192 (6.0 in Q5)**: Returns 32767 (saturated positive).
- **Input ≤ -192**: Returns 0 (saturated negative).
- **Input = 0**: Returns 16384 = sigmoid(0) = 0.5.

### 9.2 `silk_lin2log`
- **Input ≤ 0**: Undefined behavior (CLZ of 0 is architecture-dependent). Callers must ensure positive input.
- **Input = 1**: Returns 0 (log2(1) = 0).

### 9.3 `silk_log2lin`
- **Input < 0**: Returns 0.
- **Input ≥ 3967**: Returns `silk_int32_MAX` (clamped, prevents shift overflow since `3967 >> 7 = 30` and `1 << 31` would be UB).
- **Input = 0**: Returns 1 (2^0 = 1).

### 9.4 `silk_LPC_inverse_pred_gain_c`
- **DC response ≥ 4096**: Early return 0 (DC-unstable filter, all coefficients sum to ≥ 1.0).
- **Any `|A_QA[k]| > A_LIMIT`**: Return 0 (coefficient too large, pole near unit circle).
- **Intermediate overflow**: 64-bit check; returns 0 if result doesn't fit int32.
- **order = 0**: Untested, assert would fire.

### 9.5 `silk_LPC_fit`
- **Pathological coefficients**: After 10 iterations of bandwidth expansion, falls back to hard clipping via `silk_SAT16`. The clipped values are written back to `a_QIN` for consistency.
- **QOUT = QIN**: Degenerate but valid; shift amount = 0.

### 9.6 `silk_insertion_sort_increasing`
- **K = L**: Full sort of entire array.
- **K = 1**: Simply finds the minimum element and its index.
- **L = 0 or K = 0**: Assertion failure.

### 9.7 `silk_interpolate`
- **ifact_Q2 = 0**: Output = x0 (pure copy).
- **ifact_Q2 = 4**: Output = x1 (pure copy).
- **ifact_Q2 outside [0,4]**: Assertion failure. Caller must ensure valid range.

---

## 10. Porting Notes

### 10.1 In-Place Mutation

Several functions modify their inputs in-place:
- **`silk_insertion_sort_*`**: Reorders `a[]` and writes `idx[]`. The partial-sort variant leaves elements beyond K in undefined order.
- **`silk_bwexpander` / `silk_bwexpander_32`**: Modifies `ar[]` in-place.
- **`silk_LPC_fit`**: Modifies both `a_QIN[]` (input) and writes `a_QOUT[]` (output). The input array is mutated by repeated bandwidth expansion.
- **`LPC_inverse_pred_gain_QA_c`**: Destroys its input `A_QA[]` array during recursion.

In Rust, use `&mut [i32]` slices. The `LPC_inverse_pred_gain_QA_c` pattern of receiving a mutable array that gets destroyed is natural with stack-allocated arrays in Rust (`let mut atmp: [i32; SILK_MAX_ORDER_LPC]`).

### 10.2 Pointer-to-Pointer Tables

Several tables use `const T * const table_ptrs[N]` to index into static sub-tables:
```c
const opus_uint8 * const silk_LTP_gain_iCDF_ptrs[NB_LTP_CBKS] = {
    silk_LTP_gain_iCDF_0, silk_LTP_gain_iCDF_1, silk_LTP_gain_iCDF_2
};
const opus_int8 * const silk_LTP_vq_ptrs_Q7[NB_LTP_CBKS] = {
    (opus_int8 *)&silk_LTP_gain_vq_0[0][0], ...
};
const opus_uint8 * const silk_LBRR_flags_iCDF_ptr[2] = { ... };
```

In Rust, replace with `&[&[T]]` slices or a flat array with a size descriptor. The LTP VQ tables cast a 2D `[N][5]` array to a flat `*int8`, so in Rust, flatten to `&[i8]` and access via `[idx * 5 .. idx * 5 + 5]`.

### 10.3 Fixed-Point Macro Translation

All `silk_*` arithmetic macros must be translated to Rust functions or inline operations. Key translations:

| C Macro | Rust Equivalent |
|---|---|
| `silk_MUL(a, b)` | `a.wrapping_mul(b)` or `a * b` (with overflow check) |
| `silk_SMULBB(a, b)` | `(a as i16 as i32) * (b as i16 as i32)` |
| `silk_SMULWB(a, b)` | `((a as i64 * (b as i16 as i64)) >> 16) as i32` |
| `silk_SMULWW(a, b)` | `((a as i64 * b as i64) >> 16) as i32` |
| `silk_SMMUL(a, b)` | `((a as i64 * b as i64) >> 32) as i32` |
| `silk_RSHIFT_ROUND(a, s)` | `if s == 1 { (a >> 1) + (a & 1) } else { ((a >> (s-1)) + 1) >> 1 }` |
| `silk_ADD_RSHIFT32(a, b, s)` | `a + (b >> s)` |
| `silk_SMLABB(a, b, c)` | `a + (b as i16 as i32) * (c as i16 as i32)` |
| `silk_SMLAWB(a, b, c)` | `(a as i64 + ((b as i64 * (c as i16 as i64)) >> 16)) as i32` |
| `silk_CLZ32(x)` | `x.leading_zeros() as i32` (Rust's built-in) |
| `silk_LSHIFT(a, s)` | `(a as u32).wrapping_shl(s as u32) as i32` |
| `SILK_FIX_CONST(C, Q)` | `((C) * (1i64 << (Q)) + 0.5) as i32` — evaluate at compile time |
| `silk_SAT16(a)` | `a.clamp(i16::MIN as i32, i16::MAX as i32) as i16` |

**Critical**: `silk_SMULBB` truncates both operands to `int16` before multiplying. This is not a simple `i32 * i32` — it masks to the bottom 16 bits. The Rust port must replicate this truncation exactly.

### 10.4 Conditional Compilation

- `silk_insertion_sort_decreasing_int16` is gated on `#ifdef FIXED_POINT`. Since this port targets fixed-point, include it.
- The `silk_SMLABB_ovflw` macro used in `sum_sqr_shift.c` deliberately allows unsigned overflow on the intermediate multiply-accumulate. In Rust, use `wrapping_add` / `wrapping_mul` or cast to `u32` for that computation.

### 10.5 `silk_DWORD_ALIGN`

Resampler tables are annotated with `silk_DWORD_ALIGN` for 4-byte alignment. In Rust, static arrays of `i16` are naturally 2-byte aligned. If alignment matters for SIMD later, use `#[repr(align(4))]` on a wrapper struct. For the initial port this can be ignored.

### 10.6 Two-Dimensional Array Flattening

C tables declared as `const T table[ROWS][COLS]` are laid out in row-major order. In Rust, either:
- Use `[[T; COLS]; ROWS]` (idiomatic, matches C layout).
- Use `[T; ROWS * COLS]` with manual indexing `[row * COLS + col]`.

The first approach is preferred for readability when the dimensions are known constants.

### 10.7 Signed/Unsigned Mixing

`silk_sum_sqr_shift` mixes `opus_int32` and `opus_uint32` in the accumulation loop, using `silk_ADD_RSHIFT_uint` which treats the shifted operand as unsigned. In Rust, this requires explicit casts between `i32` and `u32` for the intermediate sum.

### 10.8 `silk_CLZ_FRAC` Inline

This critical inline function decomposes an int32 into leading zeros + 7-bit fraction:
```c
static inline void silk_CLZ_FRAC(opus_int32 in, opus_int32 *lz, opus_int32 *frac_Q7) {
    *lz = silk_CLZ32(in);
    *frac_Q7 = silk_ROR32(in, 24 - *lz) & 0x7F;  // 7 bits after leading 1
}
```
In Rust, implement as returning a tuple `(lz: i32, frac_q7: i32)` to avoid out-pointer pattern.

### 10.9 Module Organization

Recommended Rust module structure:
```
silk/
  mod.rs              // Re-exports
  tables/
    mod.rs            // Re-exports all table sub-modules
    gain.rs           // silk_gain_iCDF, silk_delta_gain_iCDF
    pulses.rs         // Pulses per block, shell code, rate levels, sign
    pitch_lag.rs      // Pitch lag, delta, contour tables
    nlsf_cb_nb_mb.rs  // NLSF codebook for NB/MB
    nlsf_cb_wb.rs     // NLSF codebook for WB
    ltp.rs            // LTP periodicity, gain, VQ codebooks
    lsf_cos.rs        // LSF cosine table
    pitch_est.rs      // Pitch estimation stage tables
    resampler.rs      // Resampler filter coefficients
    other.rs          // Stereo, LBRR, LSB, type/offset, interpolation, transition
  sort.rs             // Insertion sort variants
  sigm.rs             // silk_sigm_Q15
  lin2log.rs          // silk_lin2log
  log2lin.rs          // silk_log2lin
  bwexpander.rs       // silk_bwexpander + silk_bwexpander_32
  inner_prod.rs       // silk_inner_prod_aligned_scale
  sum_sqr_shift.rs    // silk_sum_sqr_shift
  lpc_inv_pred_gain.rs // silk_LPC_inverse_pred_gain
  lpc_fit.rs          // silk_LPC_fit
  interpolate.rs      // silk_interpolate
  macros.rs           // Fixed-point arithmetic (silk_SMULBB, etc.)
```

### 10.10 Testing Strategy

Each function has deterministic, pure-functional behavior (no global state, no I/O). Testing approach:
1. **Differential testing via FFI**: Call both the C reference and Rust implementation with identical inputs; compare outputs byte-for-byte.
2. **Table verification**: Hash all static tables and compare against known-good values from the C reference.
3. **Edge case sweeps**: For scalar functions (`sigm_Q15`, `lin2log`, `log2lin`), exhaustively test all possible inputs (feasible since input domains are bounded).
4. **Stability oracle for `LPC_inverse_pred_gain`**: Generate random LPC coefficient sets, verify Rust agrees with C on stable/unstable classification.
