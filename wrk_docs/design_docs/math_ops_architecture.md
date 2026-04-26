# Math Operations Module — Architecture Documentation

## Module: `celt/mathops` + `celt/celt_lpc` + supporting headers

---

## 1. Purpose

The math operations module provides the fundamental arithmetic primitives used throughout the CELT layer of the Opus codec. It serves three roles:

1. **Fixed-point arithmetic library** (`arch.h`, `fixed_generic.h`): Defines Q-format multiply, shift, saturate, and rounding macros that abstract the difference between fixed-point and floating-point builds. Every arithmetic operation in CELT goes through these macros.

2. **Transcendental function approximations** (`mathops.h`, `mathops.c`): Provides bit-exact polynomial approximations of sqrt, rsqrt, cos, log2, exp2, atan2, and reciprocal — all in fixed-point Q-formats suitable for the codec's dynamic range.

3. **Linear prediction** (`celt_lpc.c`, `celt_lpc.h`): Implements Levinson-Durbin LPC analysis, FIR/IIR filtering, and autocorrelation — the signal-processing workhorses used by pitch detection, pre-emphasis, and the CELT encoder/decoder.

4. **Float-to-integer conversion** (`float_cast.h`): Platform-optimized `float2int` (round-to-nearest) and `FLOAT2INT16`/`FLOAT2INT24` conversions used at API boundaries.

These primitives are on the critical path of every encode and decode operation. Bit-exactness of every function is a hard requirement — the codec's entropy coding will diverge if any arithmetic differs by even 1 LSB.

---

## 2. Public API

### 2.1 Integer Square Root

```c
unsigned isqrt32(opus_uint32 _val);
```
- **Input**: `_val` > 0, unsigned 32-bit integer
- **Output**: `floor(sqrt(_val))`, exact for all valid 32-bit inputs
- **Used by**: Combinatorial coding (`cwrs.c`), normalization

### 2.2 Fixed-Point Reciprocal

```c
opus_val32 celt_rcp(opus_val32 x);         // Q15 in → Q16 out
opus_val16 celt_rcp_norm16(opus_val16 x);   // normalized Q15 in → Q15 out
opus_val32 celt_rcp_norm32(opus_val32 x);   // normalized Q31 in → Q30 out
```
- `celt_rcp`: General reciprocal. Normalizes input via `celt_ilog2`, calls `celt_rcp_norm16`, denormalizes result. Max relative error: 7.05e-5.
- `celt_rcp_norm16`: Core 16-bit reciprocal for input in `[0.5, 1.0)` (Q15). Uses linear approximation + 2 Newton iterations.
- `celt_rcp_norm32`: 32-bit reciprocal for input in `[0.5, 1.0)` (Q31), output `[1.0, 2.0)` (Q30). Extends 16-bit seed via Newton refinement.

### 2.3 Fixed-Point Division

```c
opus_val32 frac_div32_q29(opus_val32 a, opus_val32 b);  // → Q29 result
opus_val32 frac_div32(opus_val32 a, opus_val32 b);       // → Q31 result (saturated)
```
- `frac_div32_q29`: Computes `a/b` in Q29 using reciprocal + one refinement step.
- `frac_div32`: Wraps `frac_div32_q29`, shifts result to Q31, saturates to `[-2^31, 2^31-1]`.

### 2.4 Square Root

```c
opus_val32 celt_sqrt(opus_val32 x);    // QX in → QX/2 out (16-bit precision)
opus_val32 celt_sqrt32(opus_val32 x);  // QX in → Q(X/2+16) out (32-bit precision)
```
- `celt_sqrt`: 5th-order polynomial approximation over `[0.25, 1)`. RMS error 3.4e-5, max error 8.2e-5.
- `celt_sqrt32`: Uses `celt_rsqrt_norm32` for higher precision. Output in Q(X/2+16).

### 2.5 Reciprocal Square Root

```c
opus_val16 celt_rsqrt_norm(opus_val32 x);    // Q16 in → Q14 out
opus_val32 celt_rsqrt_norm32(opus_val32 x);  // Q31 in → Q29 out
```
- `celt_rsqrt_norm`: Minimax quadratic initial guess + 2nd-order Householder iteration. Max relative error 1.05e-4.
- `celt_rsqrt_norm32`: Extends 16-bit seed to 32 bits via Newton-Raphson refinement.

### 2.6 Cosine

```c
opus_val16 celt_cos_norm(opus_val32 x);    // Q16 phase → Q15 output
opus_val32 celt_cos_norm32(opus_val32 x);  // Q30 phase → Q31 output
```
- Computes `cos(π/2 · x)`.
- `celt_cos_norm`: Uses 4th-order Chebyshev-like polynomial, handles quadrant folding via bit manipulation.
- `celt_cos_norm32`: 4th-order polynomial with higher-precision Q27-Q35 coefficients.

### 2.7 Logarithm and Exponential

```c
// Fixed-point:
opus_val16 celt_log2(opus_val32 x);    // Q14 in → Q10 out
opus_val32 celt_exp2(opus_val16 x);    // Q10 in → Q16 out

// With ENABLE_QEXT:
opus_val32 celt_log2_db(opus_val32 x); // Q14 in → Q(DB_SHIFT=24) out
opus_val32 celt_exp2_db(opus_val32 x); // Q(DB_SHIFT) in → Q16 out

// Float (FLOAT_APPROX):
float celt_log2(float x);   // IEEE 754 bit manipulation + polynomial
float celt_exp2(float x);   // integer/fraction split + polynomial
```

### 2.8 Arctangent

```c
// Fixed-point:
opus_val32 celt_atan_norm(opus_val32 x);                       // Q30 in → Q30 out, computes atan(x)·2/π
opus_val32 celt_atan2p_norm(opus_val32 y, opus_val32 x);      // Q30 in → Q30 out, atan2(y,x)·2/π, positive quadrant
opus_val16 celt_atan01(opus_val16 x);                          // Q15 in → Q15 out, atan(x)·4/π for [0,1]
opus_val16 celt_atan2p(opus_val16 y, opus_val16 x);           // Q15 in → Q15 out

// Float:
float fast_atan2f(float y, float x);         // rational approximation
float celt_atan_norm(float x);               // Remez order-15 polynomial
float celt_atan2p_norm(float y, float x);    // wraps celt_atan_norm
```

### 2.9 Utility Functions

```c
// Max absolute value scanning:
opus_val32 celt_maxabs16(const opus_val16 *x, int len);
opus_val32 celt_maxabs32(const opus_val32 *x, int len);  // fixed-point only

// Float conversion:
void celt_float2int16_c(const float *in, short *out, int cnt);
int opus_limit2_checkwithin1_c(float *samples, int cnt);  // clamp to [-2,2], returns 0

// Integer log:
opus_int16 celt_ilog2(opus_int32 x);  // = EC_ILOG(x) - 1, undefined for x<=0
opus_int16 celt_zlog2(opus_val32 x);  // = 0 for x<=0, else celt_ilog2(x)
```

### 2.10 LPC Functions

```c
#define CELT_LPC_ORDER 24

void _celt_lpc(opus_val16 *_lpc, const opus_val32 *ac, int p);
void celt_fir(const opus_val16 *x, const opus_val16 *num, opus_val16 *y,
              int N, int ord, int arch);
void celt_iir(const opus_val32 *x, const opus_val16 *den, opus_val32 *y,
              int N, int ord, opus_val16 *mem, int arch);
int _celt_autocorr(const opus_val16 *x, opus_val32 *ac,
                   const celt_coef *window, int overlap,
                   int lag, int n, int arch);
```

---

## 3. Internal State / Structs

This module is **stateless** — all functions are pure (no global/static mutable state). The only state is:

- **`celt_iir` memory buffer** (`opus_val16 *mem`): Caller-owned IIR filter state of length `ord`. Stores the `ord` most recent output samples (as Q12 values in fixed-point). Updated in-place after each call; caller must persist across frames.

- **Static coefficient tables** (compile-time constants, no mutable state):
  - `celt_sqrt` coefficients: `C[6]` — minimax polynomial for `[0.25, 1.0)`
  - `celt_log2` coefficients: `C[5]` — for fixed-point log2
  - `celt_exp2_frac` coefficients: `D0..D3` — for fixed-point exp2
  - `celt_cos_norm` coefficients: `L1..L4` — for fixed-point cosine
  - Float `FLOAT_APPROX` tables: `log2_x_norm_coeff[8]`, `log2_y_norm_coeff[8]` — range-reduction LUTs

---

## 4. Algorithms

### 4.1 Integer Square Root (`isqrt32`)

Binary search from MSB to LSB. At each step, tests whether `(g + b)² ≤ val` where `b = 2^bshift`. Starting bit position is `(EC_ILOG(val) - 1) / 2`.

```
g = 0
bshift = (EC_ILOG(val)-1) >> 1
b = 1 << bshift
loop:
    t = (2*g + b) << bshift
    if t <= val: g += b; val -= t
    b >>= 1; bshift--
until bshift < 0
return g
```

This is exact for all 32-bit inputs > 0.

### 4.2 Reciprocal (`celt_rcp_norm16`)

1. **Linear seed**: `r = 30840 - 15420·x` (Q14, x in normalized `[0.5, 1.0)`)
2. **Newton iteration 1**: `r -= r · (r·x + r - 1.0)`
3. **Newton iteration 2**: Same form, subtracts extra 1 to compensate truncation and avoid overflow.

The general `celt_rcp` normalizes via `celt_ilog2`, calls `celt_rcp_norm16` on the normalized mantissa, then denormalizes the result.

### 4.3 Fractional Division (`frac_div32_q29`)

1. Normalize both `a` and `b` by the same shift (based on `celt_ilog2(b) - 29`)
2. Compute 16-bit reciprocal of `b`: `rcp = ROUND16(celt_rcp(ROUND16(b, 16)), 3)`
3. First estimate: `result = rcp × a` (Q15 multiply)
4. Compute remainder: `rem = a/4 - result·b` (via `MULT32_32_Q31`)
5. Refine: `result += 4 · (rcp × rem)`

### 4.4 Reciprocal Square Root (`celt_rsqrt_norm`)

Input `x` in Q16, range `[0.25, 1.0)` (i.e., integer values `[16384, 65535]`).

1. **Normalize**: `n = x - 32768` → range `[-16384, 32767]` (Q15 `[-0.5, 1.0)`)
2. **Minimax quadratic seed** (Q14): `r = 23557 + n·(-13490 + n·6713)`
3. **Compute residual**: `r2 = r·r` (Q15), then `y = 2·(r2·n + r2 - 16384)` — this is `x·r² - 1` in Q15
4. **Householder correction**: `r += r · y · (0.375·y - 0.5)` — 2nd-order, max relative error 1.05e-4

### 4.5 Square Root (`celt_sqrt`)

1. Handle edge cases: `x == 0 → 0`, `x >= 2^30 → 32767`
2. Extract exponent: `k = celt_ilog2(x) / 2 - 7`
3. Normalize to `[0.25, 1.0)` in Q15: `x = VSHR32(x, 2*k)`, `n = x - 32768`
4. 5th-order polynomial: `rt = C[0] + n·(C[1] + n·(C[2] + ...))` with coefficients `{23171, 11574, -2901, 1592, -1002, 336}`
5. Denormalize: `rt = VSHR32(rt, 7 - k)`

### 4.6 Cosine (`celt_cos_norm`)

Input is Q16 phase where `1.0 = 2^16 = π/2`.

1. **Fold to `[0, 2π)`**: `x &= 0x1FFFF` (mod 2^17)
2. **Mirror to `[0, π]`**: if `x > 2^16`, then `x = 2^17 - x`
3. **Quadrant dispatch**:
   - If fractional bits present (`x & 0x7FFF`):
     - `[0, π/2)`: return `_celt_cos_pi_2(x)`
     - `[π/2, π]`: return `-_celt_cos_pi_2(65536 - x)`
   - Exact boundary values: 0 → 32767, π/2 → 0, π → -32767

The inner `_celt_cos_pi_2` uses coefficients `{32767, -7651, 8277, -626}` in a 4th-order Chebyshev approximation.

### 4.7 Log2 (fixed-point, `celt_log2`)

1. `i = celt_ilog2(x)` — integer part of log2
2. Normalize: `n = VSHR32(x, i-15) - 32768 - 16384` — mantissa in `[-16384, 16383]` (Q14 `[-0.5, 0.5)`)
3. 4th-order polynomial with coefficients `{-6801+8, 15746, -5217, 2545, -1401}` → Q14 fractional part
4. Combine: `result = (i - 13) << 10 + frac >> 4` → Q10 output

### 4.8 Exp2 (fixed-point, `celt_exp2`)

1. Split: `integer = x >> 10`, `frac = x - integer << 10`
2. Saturate: integer > 14 → 0x7F000000, integer < -15 → 0
3. Fractional part via `celt_exp2_frac`: 3rd-order polynomial with coefficients derived from `{1, log(2), 3-4·log(2), 3·log(2)-2}` scaled to Q15 `{16383, 22804, 14819, 10204}`
4. Denormalize: `VSHR32(frac, -integer - 2)` → Q16

### 4.9 LPC Analysis (`_celt_lpc`)

Levinson-Durbin recursion computing LPC coefficients from autocorrelation:

1. Initialize `error = ac[0]`, `lpc[] = {0}`
2. For each order `i = 0..p-1`:
   a. Compute reflection coefficient: `rr = Σ lpc[j]·ac[i-j] + ac[i+1]/64`
   b. Normalize: `r = -frac_div32(rr·64, error)` — Q31 reflection coefficient
   c. Store: `lpc[i] = r / 64` (Q25 internal)
   d. Symmetric update: `lpc[j] += r·lpc[i-1-j]` (both ends toward center)
   e. Update error: `error -= r²·error`
   f. Early exit if error ≤ ac[0]/1024 (30 dB gain limit)
3. **Fixed-point coefficient fitting** (up to 10 iterations):
   - Find max |lpc[i]|, check if > 32767 in Q12
   - If so, apply bandwidth expansion (chirp factor decay)
   - If still won't fit after 10 iterations, fall back to `A(z) = 1` (lpc[0] = 4096 Q12)
4. Final conversion: Q25 → Q12 via `PSHR32(lpc[i], 13)`

### 4.10 FIR Filter (`celt_fir_c`)

```
y[i] = Σ(j=0..ord-1) num[ord-1-j] · x[i+j-ord]
```

- Coefficients reversed into `rnum[]` for correlation-style access
- Inner loop unrolled 4× using `xcorr_kernel` (platform-optimized)
- Accumulation in Q(SIG_SHIFT=12) internally, output rounded to Q0 via `SROUND16`
- Constraint: `x != y` (no in-place operation)

### 4.11 IIR Filter (`celt_iir`)

```
y[i] = x[i] - Σ(j=0..ord-1) den[j] · mem[j]
```

Two implementations:
- **SMALL_FOOTPRINT**: Direct form, `O(N·ord)`, straightforward
- **Default**: Unrolled 4×, treats as FIR with `xcorr_kernel` + patch-up for recursive feedback within each 4-sample block. Filter memory stored as most recent `ord` outputs.

### 4.12 Autocorrelation (`_celt_autocorr`)

1. **Windowing**: If `overlap > 0`, apply symmetric window to edges of signal
2. **Dynamic range management** (fixed-point only):
   - Estimate energy `ac0` with right-shift to prevent overflow
   - Compute normalization shift: `shift = (celt_ilog2(ac0) - 30 + ac0_shift + 1) / 2`
   - Scale signal down by `shift` if needed
3. **Core correlation**: `celt_pitch_xcorr(xptr, xptr, ac, fastN, lag+1, arch)` for bulk, then scalar tail
4. **Post-normalization** (fixed-point): Ensure `ac[0]` is in `[2^28, 2^30)` by additional shifting
5. **Returns**: Total shift applied (for caller to interpret magnitudes)

---

## 5. Data Flow

### Signal Chain Position

```
Input PCM → [pre-emphasis] → [LPC analysis via autocorr + _celt_lpc]
         → [MDCT via celt_sqrt, celt_cos_norm]
         → [band energy via celt_log2, celt_exp2]
         → [quantization via celt_rcp, frac_div32]
         → [entropy coding via isqrt32]
         → Bitstream
```

### Buffer Layouts

| Function | Input Buffer | Output Buffer | Scratch |
|----------|-------------|---------------|---------|
| `_celt_lpc` | `ac[0..p]` (Q? autocorrelation) | `_lpc[0..p-1]` (Q12 coefficients) | `lpc[CELT_LPC_ORDER]` (Q25 internal, stack) |
| `celt_fir_c` | `x[0..N-1]` (Q0, needs `ord` history before `x[0]`) | `y[0..N-1]` (Q0) | `rnum[ord]` (reversed coefficients, stack) |
| `celt_iir` | `_x[0..N-1]` (Q(SIG_SHIFT)) | `_y[0..N-1]` (Q(SIG_SHIFT)) | `rden[ord]`, `y[N+ord]` (stack) |
| `_celt_autocorr` | `x[0..n-1]` (Q0 samples) | `ac[0..lag]` (autocorrelation) | `xx[n]` (windowed/scaled copy, stack) |

### Key Data Flow Constraint

`celt_fir_c` requires `x != y`. The input buffer must have `ord` valid samples before `x[0]` (i.e., the caller provides `x[-ord..-1]` as filter history). The IIR filter stores its state in `mem[0..ord-1]` which the caller must persist across calls.

---

## 6. Numerical Details

### 6.1 Type System

The entire fixed-point system is built on conditional typedefs in `arch.h`:

| Type | Fixed-Point | Float |
|------|-------------|-------|
| `opus_val16` | `opus_int16` (16-bit signed) | `float` |
| `opus_val32` | `opus_int32` (32-bit signed) | `float` |
| `opus_val64` | `opus_int64` (64-bit signed) | `float` |
| `celt_sig` | `opus_int32` | `float` |
| `celt_norm` | `opus_int32` | `float` |
| `celt_glog` | `opus_int32` | `float` |

### 6.2 Q-Format Summary

| Value | Q-Format | Range | Notes |
|-------|----------|-------|-------|
| Signal samples | Q(SIG_SHIFT=12) | ±SIG_SAT (2^29-1) | Internal CELT resolution |
| LPC coefficients | Q12 | must fit int16 | Stored as `opus_val16` |
| LPC internal | Q25 | int32 | Before conversion to Q12 |
| Autocorrelation | Q(variable) | post-normalized to `[2^28, 2^30)` | Shift returned to caller |
| `celt_cos_norm` input | Q16 | `[0, 2^17)` phase | 1.0 = π/2 radians |
| `celt_cos_norm` output | Q15 | `[-32767, 32767]` | ≈ `[-1.0, 1.0]` |
| `celt_cos_norm32` input | Q30 | `[-2^30, 2^30]` | |
| `celt_cos_norm32` output | Q31 | | |
| `celt_log2` input | Q14 | | |
| `celt_log2` output | Q10 | -32767 for x=0 | |
| `celt_exp2` input | Q10 | | |
| `celt_exp2` output | Q16 | 0x7F000000 for overflow | |
| `celt_rcp` input | Q15 (positive) | | |
| `celt_rcp` output | Q16 | | |
| `celt_rsqrt_norm` input | Q16 | `[0.25, 1.0)` i.e. `[16384, 65535]` | |
| `celt_rsqrt_norm` output | Q14 | | |
| `celt_sqrt` input | QX | | |
| `celt_sqrt` output | QX/2 | | |
| `celt_sqrt32` input | QX | | |
| `celt_sqrt32` output | Q(X/2+16) | | |
| `frac_div32` output | Q31 | saturated to `[-(2^31), 2^31-1]` | |
| `frac_div32_q29` output | Q29 | | |
| `celt_atan_norm` I/O | Q30 | `[-1.0, 1.0]` | Returns `atan(x)·2/π` |
| DB_SHIFT | 24 | | Used for `celt_log2_db` / `celt_exp2_db` |

### 6.3 Key Fixed-Point Arithmetic Macros

```c
// Truncating multiplies:
MULT16_16(a,b)         // 16×16 → 32, no shift
MULT16_16_Q15(a,b)     // 16×16 → 32, >> 15 (truncate)
MULT16_16_Q14(a,b)     // 16×16 → 32, >> 14
MULT16_32_Q15(a,b)     // 16×32 → 32, >> 15
MULT16_32_Q16(a,b)     // 16×32 → 32, >> 16
MULT32_32_Q31(a,b)     // 32×32 → 32, >> 31 (requires 64-bit intermediate)
MULT32_32_Q16(a,b)     // 32×32 → 32, >> 16

// Rounding multiplies:
MULT16_16_P15(a,b)     // 16×16 → 32, + 16384 then >> 15 (round-to-nearest)
MULT16_16_P13(a,b)     // 16×16 → 32, + 4096 then >> 13
MULT32_32_P31(a,b)     // 32×32 → 32, + 2^30 then >> 31

// Shifts:
SHL32(a, shift)        // left shift via unsigned cast (avoids UB)
SHR32(a, shift)        // arithmetic right shift
PSHR32(a, shift)       // round-to-nearest right shift: (a + (1 << (shift-1))) >> shift
VSHR32(a, shift)       // variable-direction shift: positive=right, negative=left

// Saturation:
SAT16(x)               // clamp int32 to [-32768, 32767]
SATURATE(x, a)         // clamp to [-a, a]
SROUND16(x, shift)     // PSHR32 + saturate to int16
```

### 6.4 Overflow Guards

- **`MULT32_32_Q31` without 64-bit**: Decomposes into four 16×16 multiplies to stay within 32-bit arithmetic. The `OPUS_FAST_INT64` flag selects the direct 64-bit path.
- **`SHL32`**: Cast to `opus_uint32` before shifting to avoid signed overflow UB.
- **`ADD32_ovflw` / `SUB32_ovflw`**: Explicitly cast to unsigned to allow wrapping without UB (used in places where overflow is intentional).
- **`SIG_SAT = 2^29 - 1`**: Signal saturation limit chosen so that `2 × SIG_SAT` doesn't overflow 32 bits, and typical MDCT/comb-filter processing stays within range.
- **`frac_div32` saturation**: Q29 result ≥ 2^29 → clamp to 2^31-1; ≤ -2^29 → clamp to -(2^31).
- **LPC coefficient fitting**: Iterative bandwidth expansion (up to 10 rounds) ensures Q25→Q12 conversion fits in int16. Fallback to identity filter prevents divergence.

### 6.5 Rounding Behavior

All truncating multiplies (`MULT16_16_Q15`, `MULT32_32_Q31`, etc.) truncate toward negative infinity (arithmetic right shift). Rounding variants (`P15`, `P13`, `P31`) add half the divisor before shifting — round-to-nearest with ties going up.

`PSHR32(a, shift)` implements: `(a + (1 << (shift - 1))) >> shift`

**Critical**: The rounding bias in `PSHR32` is `1 << (shift - 1)` which for shift=0 is 0 (no rounding) and for shift=1 is 1. This matches the C reference exactly and must be preserved.

---

## 7. Dependencies

### What This Module Calls

| Dependency | Used For |
|-----------|---------|
| `entcode.h` → `EC_ILOG()` | Integer log2 (count leading zeros) — used by `celt_ilog2`, `isqrt32` |
| `os_support.h` | `OPUS_CLEAR`, memory macros |
| `stack_alloc.h` | `VARDECL`, `ALLOC`, `SAVE_STACK`, `RESTORE_STACK` — stack allocation for temp buffers |
| `pitch.h` → `celt_pitch_xcorr()`, `xcorr_kernel()` | Optimized correlation inner loops used by `_celt_autocorr` and `celt_fir`/`celt_iir` |
| `cpu_support.h` | `arch` parameter dispatch for SIMD overrides |

### What Calls This Module

| Caller | Functions Used |
|--------|---------------|
| `bands.c` | `celt_sqrt`, `celt_rsqrt_norm`, `celt_log2`, `celt_exp2`, `celt_cos_norm`, `celt_rcp`, `frac_div32` |
| `celt_encoder.c` / `celt_decoder.c` | `_celt_lpc`, `celt_iir`, `celt_fir`, `_celt_autocorr`, `celt_sqrt`, `celt_maxabs16` |
| `pitch.c` | `_celt_autocorr`, `celt_sqrt`, `celt_rsqrt_norm`, `isqrt32` |
| `quant_bands.c` | `celt_log2`, `celt_exp2`, `celt_cos_norm` |
| `vq.c` | `isqrt32`, `celt_rcp`, `celt_rsqrt_norm` |
| `kiss_fft.c` / `mdct.c` | `celt_cos_norm` (for twiddle factors) |
| `modes.c` | `celt_log2`, `celt_exp2` (mode initialization) |
| `silk/*.c` | `celt_log2`, `celt_exp2`, `frac_div32` |
| `opus_encoder.c` / `opus_decoder.c` | `celt_float2int16`, `FLOAT2INT16`, `FLOAT2SIG` |

---

## 8. Constants and Tables

### 8.1 `celt_sqrt` Polynomial Coefficients

```c
static const opus_val16 C[6] = {23171, 11574, -2901, 1592, -1002, 336};
```
Minimax polynomial for `sqrt(x)` over `[0.25, 1.0)` in Q15. Optimized to keep all coefficients ≤ 32767. The Q15 representation means `C[0] = 23171 ≈ 0.7071 ≈ 1/sqrt(2)`.

### 8.2 `celt_cos_norm` Chebyshev Coefficients

```c
L1 = 32767   // ≈ 1.0 in Q15
L2 = -7651   // ≈ -π²/8 (first correction term)
L3 = 8277
L4 = -626
```

### 8.3 `celt_rsqrt_norm` Initial Approximation

```
r ≈ 1.4378 - 0.8234·n + 0.4096·n²    (Q14)
```
Coefficients `{23557, -13490, 6713}` — optimal minimax quadratic for relative error of `1/sqrt(x)` over `[0.25, 1.0)`.

### 8.4 `celt_log2` Fixed-Point Coefficients

```c
C[5] = {-6801+8, 15746, -5217, 2545, -1401};  // Q15, note +8 rounding offset
```
Polynomial for `log2(1 + n/32768)` where `n ∈ [-16384, 16383]`.

### 8.5 `celt_exp2_frac` Coefficients

```c
D0 = 16383  // ≈ 0.5 in Q15 (represents 1.0 in this context)
D1 = 22804  // ≈ ln(2) scaled
D2 = 14819  // ≈ 3 - 4·ln(2)
D3 = 10204  // ≈ 3·ln(2) - 2
```

### 8.6 `celt_rcp_norm16` Linear Seed

```c
r = 30840 - 15420·x    // Q14
```
This approximates `1/(0.5 + x/32768)` where `x ∈ [16384, 32767]`, giving initial values in `[15420, 30840]`.

### 8.7 Special Constants

```c
Q15ONE    = 32767          // 1.0 in Q15
Q31ONE    = 2147483647     // 1.0 in Q31
SIG_SHIFT = 12             // Signal resolution shift
SIG_SAT   = 536870911      // 2^29 - 1, signal saturation
DB_SHIFT  = 24             // dB-domain Q-format
NORM_SHIFT = 24            // Normalization shift
CELT_SIG_SCALE = 32768.0f  // Float signal scaling
PI = 3.1415926535897931
CELT_LPC_ORDER = 24        // Maximum LPC order
```

---

## 9. Edge Cases

### 9.1 Zero / Degenerate Inputs

| Function | Edge Case | Behavior |
|----------|-----------|----------|
| `isqrt32` | `_val = 0` | **Undefined** — precondition requires `_val > 0` |
| `celt_rcp` | `x ≤ 0` | `celt_sig_assert(x > 0)` — assertion in debug, undefined in release |
| `celt_rcp_norm32` | `x < 2^30` | `celt_sig_assert(x >= 1073741824)` |
| `celt_sqrt` | `x = 0` | Returns 0 |
| `celt_sqrt` | `x ≥ 2^30` | Returns 32767 (saturated) |
| `celt_sqrt32` | `x = 0` | Returns 0 |
| `celt_sqrt32` | `x ≥ 2^30` | Returns 2^31-1 (saturated) |
| `celt_log2` | `x = 0` | Returns -32767 (negative infinity representation) |
| `celt_cos_norm` | Phase at exact quadrant boundaries | Returns exact `{32767, 0, -32767}` via special-case branch |
| `celt_cos_norm32` | `|x| = 2^30` | Returns exactly 0 (special case before polynomial) |
| `frac_div32` | Result ≥ 2^29 | Saturates to 2^31-1 |
| `frac_div32` | Result ≤ -2^29 | Saturates to -(2^31) |
| `_celt_lpc` | `ac[0] = 0` | Skips entire Levinson recursion, returns all-zero coefficients |
| `_celt_lpc` | Coefficients don't fit int16 after 10 iterations | Falls back to `A(z) = 1`: `_lpc = {4096, 0, 0, ...}` |
| `_celt_autocorr` | `overlap = 0` | Skips windowing, uses input directly |
| `opus_limit2_checkwithin1_c` | `cnt ≤ 0` | Returns 1 immediately |
| `celt_maxabs16` | `len = 0` | Returns 0 (tracks both max and min starting from 0) |

### 9.2 Precision Boundaries

- `celt_exp2`: integer > 14 → 0x7F000000 (avoids shift overflow); integer < -15 → 0
- `celt_atan_norm`: Exact return for `x = ±1.0` (Q30) — `±536870912` — avoids polynomial evaluation at boundary
- `fast_atan2f`: Returns 0 when `x² + y² < 1e-18` — prevents division by zero

---

## 10. Porting Notes

### 10.1 Macro-Heavy Arithmetic Layer

The entire fixed-point arithmetic system (`MULT16_16_Q15`, `SHL32`, `PSHR32`, etc.) is implemented as C preprocessor macros. In Rust:

- **Approach**: Implement as `#[inline(always)]` functions on newtype wrappers (e.g., `Q15(i16)`, `Q31(i32)`). This provides type safety that the C code lacks while maintaining zero overhead.
- **Critical**: The macro `MULT32_32_Q31` has two implementations — one using 64-bit intermediates (`OPUS_FAST_INT64`) and one decomposing into 16-bit multiplies. Port the 64-bit version only (Rust has native `i64`). Verify bit-exactness against both paths.
- **`SHL32` uses unsigned cast**: `(opus_int32)((opus_uint32)(a) << (shift))` — this prevents signed overflow UB. In Rust, use wrapping arithmetic: `(a as u32).wrapping_shl(shift) as i32`.

### 10.2 Conditional Compilation

The module has extensive `#ifdef` branching:

| `#ifdef` | What it controls | Rust strategy |
|----------|-----------------|---------------|
| `FIXED_POINT` | Entire arithmetic layer, all function bodies | Feature flag `fixed_point`. Two separate implementations. |
| `FLOAT_APPROX` | Bit-manipulation log2/exp2 vs. libm | Feature flag, or always use the FLOAT_APPROX versions for determinism |
| `OPUS_FAST_INT64` | 64-bit multiply decomposition | Always use 64-bit path in Rust |
| `ENABLE_QEXT` | High-precision log2_db/exp2_db | Feature flag |
| `ENABLE_RES24` | 24-bit internal resolution | Feature flag |
| `SMALL_FOOTPRINT` | Simple IIR vs. unrolled | Feature flag (cfg) |
| Platform SIMD overrides | ARM NEON, SSE, etc. | Skip for initial port per project rules |

### 10.3 In-Place Mutation Patterns

- **`celt_iir` memory update**: `mem[i] = _y[N-i-1]` at end — writes back into caller's buffer. In Rust, pass `&mut [opus_val16]` for mem.
- **`_celt_lpc` coefficient conversion**: The fixed-point path uses a temporary `lpc[CELT_LPC_ORDER]` on the stack, then copies to `_lpc`. The float path aliases `lpc = _lpc` (pointer aliasing). In Rust, use a local array for fixed-point, direct slice for float.
- **`_celt_autocorr` signal scaling**: May copy and scale `x` into `xx`, then reassign `xptr = xx`. In Rust, use a local `Vec` or stack array, with a reference that may or may not point to the original.

### 10.4 Stack Allocation Macros

`VARDECL`, `ALLOC`, `SAVE_STACK`, `RESTORE_STACK` implement C-style variable-length arrays on the stack (via `alloca` or explicit stack management). In Rust:

- For small, bounded sizes (≤ `CELT_LPC_ORDER = 24`): use fixed-size arrays on the stack.
- For sizes depending on `N` or `lag`: use `Vec<T>` or a reusable scratch buffer passed by the caller. Consider a `StackAlloc` wrapper for consistency.

### 10.5 `xcorr_kernel` Dispatch

`celt_fir`, `celt_iir`, and `_celt_autocorr` all call `xcorr_kernel(rnum, x+ptr, sum, ord, arch)` where `arch` is a runtime CPU capability flag. For the initial port:

- Implement `xcorr_kernel` as a plain Rust loop (matches `xcorr_kernel_c`)
- The `arch` parameter can be ignored initially (no SIMD)
- The function processes 4 outputs simultaneously — it accumulates `sum[0..3]` against shifted windows

### 10.6 Pointer Arithmetic in Filters

```c
xcorr_kernel(rnum, x+i-ord, sum, ord, arch);
```

The `x+i-ord` accesses samples *before* the current pointer — this requires the caller to have valid memory at negative offsets. In Rust, this means the input slice must start `ord` elements before the logical beginning. Use slice indexing: `&x[i..i+ord]` where `x` starts at the appropriate offset.

### 10.7 `float2int` Platform Sensitivity

`float_cast.h` selects among SSE `cvtss2si`, AArch64 `vcvtns_s32_f32`, `lrintf`, or `floor(0.5 + x)`. All perform round-to-nearest. In Rust:
- Use `f32::round() as i32` or the `_mm_cvtss_si32` intrinsic on x86
- **Banker's rounding vs. round-half-up**: `lrintf` uses the current FPU rounding mode (typically round-to-nearest-even), while `floor(0.5 + x)` rounds 0.5 up. The SSE path uses round-to-nearest-even. For bit-exactness, match the reference platform's behavior.

### 10.8 Union Type Punning

`celt_log2` and `celt_exp2` (FLOAT_APPROX) use `union { float f; opus_uint32 i; }` for IEEE 754 bit manipulation. In Rust: use `f32::to_bits()` / `f32::from_bits()`.

### 10.9 FRAC_MUL16 Exactness

```c
#define FRAC_MUL16(a,b) ((16384+((opus_int32)(opus_int16)(a)*(opus_int16)(b)))>>15)
```

This is a rounding 16×16→16 multiply with a +0.5 bias. It's used in critical paths and its bit-exactness is explicitly called out in the source. Ensure the Rust version uses exactly `((16384i32 + (a as i32 * b as i32)) >> 15)` with no intermediate widening surprises.

### 10.10 Negative Shift in VSHR32

```c
#define VSHR32(a, shift) (((shift)>0) ? SHR32(a, shift) : SHL32(a, -(shift)))
```

The shift amount can be negative (meaning left shift). In Rust, match arms or `if shift > 0 { a >> shift } else { (a as u32).wrapping_shl((-shift) as u32) as i32 }`.

### 10.11 Overflow-Tolerant Operations

Several macros (`ADD32_ovflw`, `SUB32_ovflw`, `NEG32_ovflw`, `SHL32_ovflw`, `PSHR32_ovflw`) explicitly use unsigned arithmetic to allow wrapping. These are used in the MDCT butterfly and a few other places. In Rust, use `.wrapping_add()`, `.wrapping_sub()`, `.wrapping_neg()`, etc.

### 10.12 LPC Coefficient Stability Check

The bandwidth expansion loop in `_celt_lpc` (lines 103-141 of `celt_lpc.c`) reuses logic from `silk_LPC_fit()` and `silk_bwexpander_32()`. The comment notes "Any bug fixes should also be applied there." When porting, consider sharing this logic or at minimum documenting the relationship.
