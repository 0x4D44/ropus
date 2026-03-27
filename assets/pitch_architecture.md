Now I have everything needed. Here is the complete architecture documentation.

---

# Pitch Module Architecture Document

**Module**: `celt/pitch.c`, `celt/pitch.h`
**Reference**: xiph/opus (BSD-licensed)
**Author**: Jean-Marc Valin (CSIRO / Xiph.Org Foundation)

---

## 1. Purpose

The pitch module implements **pitch period estimation** for the CELT layer of the Opus codec. It serves three primary functions in the codec pipeline:

1. **Pitch period detection** — estimates the fundamental frequency (F0) of the input signal, expressed as a lag in samples. Used by the CELT encoder to enable long-term prediction (pitch pre-filtering), which dramatically improves coding efficiency for voiced/tonal signals.

2. **Pitch gain computation** — estimates how periodic the signal is at the detected pitch. A high pitch gain means strong periodicity; the encoder uses this to decide whether to enable pitch prediction.

3. **Comb filtering** — applies a pitch-period comb filter for long-term prediction in both the encoder (pre-filter) and decoder (post-filter/PLC). The `comb_filter_const_c` function is declared here but defined in `celt.c`.

Additionally, the module provides **general-purpose correlation kernels** (`xcorr_kernel`, `dual_inner_prod`, `celt_inner_prod`) that are the most performance-critical inner loops in the codec, used extensively by the prefilter and PLC.

---

## 2. Public API

### 2.1. `pitch_downsample`

```c
void pitch_downsample(celt_sig * OPUS_RESTRICT x[],
                      opus_val16 * OPUS_RESTRICT x_lp,
                      int len, int C, int factor, int arch);
```

**Purpose**: Downsamples and whitens the input signal for pitch analysis. Converts from `celt_sig` (Q27 fixed-point / float) to `opus_val16` (Q15 / float) at reduced sample rate.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `celt_sig*[]` | Array of `C` channel pointers, each pointing to `len*factor` samples |
| `x_lp` | `opus_val16*` | Output buffer, `len` samples (downsampled) |
| `len` | `int` | Output length (number of downsampled samples) |
| `C` | `int` | Number of channels (1 or 2) |
| `factor` | `int` | Downsampling factor (typically 2) |
| `arch` | `int` | CPU architecture flag for SIMD dispatch |

**Algorithm**:
1. Downsample by `factor` with a 3-tap FIR (weights 0.25, 0.5, 0.25)
2. If stereo, sum both channels
3. Compute 4th-order autocorrelation of the downsampled signal
4. Add noise floor and lag windowing to the autocorrelation
5. Derive 4th-order LPC coefficients via Levinson-Durbin
6. Apply bandwidth expansion (0.9^i decay) to LPC coefficients
7. Add a zero at z = -0.8 to create a 5-tap FIR whitening filter
8. Apply the 5-tap FIR filter in-place via `celt_fir5`

### 2.2. `pitch_search`

```c
void pitch_search(const opus_val16 * OPUS_RESTRICT x_lp,
                  opus_val16 * OPUS_RESTRICT y,
                  int len, int max_pitch, int *pitch, int arch);
```

**Purpose**: Finds the best pitch period using a multi-resolution search strategy.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x_lp` | `const opus_val16*` | Whitened signal from `pitch_downsample`, `len` samples |
| `y` | `opus_val16*` | Past signal buffer for correlation, `len + max_pitch` samples |
| `len` | `int` | Analysis window length |
| `max_pitch` | `int` | Maximum pitch period to search (in samples at 2x decimation) |
| `pitch` | `int*` | Output: best pitch period (at 2x decimated rate) |
| `arch` | `int` | CPU architecture flag |

**Returns**: Best pitch period written to `*pitch`.

### 2.3. `remove_doubling`

```c
opus_val16 remove_doubling(opus_val16 *x, int maxperiod, int minperiod,
                           int N, int *T0, int prev_period,
                           opus_val16 prev_gain, int arch);
```

**Purpose**: Corrects pitch octave errors (doubling/halving) by checking subharmonics of the detected pitch. Returns the refined pitch gain.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `opus_val16*` | Signal buffer, at least `maxperiod + N` samples |
| `maxperiod` | `int` | Maximum allowed pitch period |
| `minperiod` | `int` | Minimum allowed pitch period |
| `N` | `int` | Analysis window length |
| `T0` | `int*` | In/out: initial pitch estimate → refined pitch period |
| `prev_period` | `int` | Pitch period from previous frame (for continuity) |
| `prev_gain` | `opus_val16` | Pitch gain from previous frame (Q15) |
| `arch` | `int` | CPU architecture flag |

**Returns**: `opus_val16` — pitch gain in Q15 format (0.0 to 1.0), clamped to `[0, g]` where `g` is the best subharmonic gain.

### 2.4. `celt_pitch_xcorr_c` / `celt_pitch_xcorr`

```c
#ifdef FIXED_POINT
opus_val32
#else
void
#endif
celt_pitch_xcorr_c(const opus_val16 *_x, const opus_val16 *_y,
                   opus_val32 *xcorr, int len, int max_pitch, int arch);
```

**Purpose**: Computes cross-correlation of `_x` against `max_pitch` shifted copies of `_y`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `_x` | `const opus_val16*` | Reference signal, `len` samples (must be 32-bit aligned) |
| `_y` | `const opus_val16*` | Signal to correlate against, `len + max_pitch` samples |
| `xcorr` | `opus_val32*` | Output: `max_pitch` correlation values |
| `len` | `int` | Correlation length |
| `max_pitch` | `int` | Number of lags to compute (must be > 0) |
| `arch` | `int` | CPU architecture flag |

**Returns** (fixed-point only): `opus_val32` — maximum correlation value across all lags (used for shift normalization).

### 2.5. Inline Correlation Kernels (in `pitch.h`)

#### `xcorr_kernel_c`

```c
static OPUS_INLINE void xcorr_kernel_c(const opus_val16 *x,
    const opus_val16 *y, opus_val32 sum[4], int len);
```

Computes 4 cross-correlations simultaneously: `sum[i] += Σ x[j] * y[j+i]` for `i=0..3`. The inner loop is unrolled 4x for pipeline efficiency. **This is the single most performance-critical function in the entire codec.**

Precondition: `len >= 3`.

#### `dual_inner_prod_c`

```c
static OPUS_INLINE void dual_inner_prod_c(const opus_val16 *x,
    const opus_val16 *y01, const opus_val16 *y02,
    int N, opus_val32 *xy1, opus_val32 *xy2);
```

Computes two inner products in a single pass: `*xy1 = <x, y01>` and `*xy2 = <x, y02>`.

#### `celt_inner_prod_c`

```c
static OPUS_INLINE opus_val32 celt_inner_prod_c(const opus_val16 *x,
    const opus_val16 *y, int N);
```

Single inner product: returns `Σ x[i] * y[i]`.

### 2.6. `comb_filter_const_c` (declared in `pitch.h`, defined in `celt.c`)

```c
void comb_filter_const_c(opus_val32 *y, opus_val32 *x, int T, int N,
                         opus_val16 g10, opus_val16 g11, opus_val16 g12);
```

Applies a pitch-period comb filter with constant gains. Used when the gains don't change within a subframe.

### 2.7. Arch-Dispatch Macros

All inline kernels have `#ifndef OVERRIDE_*` guards. Platform-specific implementations (SSE, SSE2, SSE4.1, NEON, MIPS) can override them. The `arch` parameter is unused in the C fallback (cast to `void`).

```c
#define xcorr_kernel(x, y, sum, len, arch)          ((void)(arch), xcorr_kernel_c(x, y, sum, len))
#define dual_inner_prod(x, y01, y02, N, xy1, xy2, arch) ((void)(arch), dual_inner_prod_c(...))
#define celt_inner_prod(x, y, N, arch)              ((void)(arch), celt_inner_prod_c(x, y, N))
#define celt_pitch_xcorr                            celt_pitch_xcorr_c
#define comb_filter_const(y, x, T, N, g10, g11, g12, arch) ((void)(arch), comb_filter_const_c(...))
```

---

## 3. Internal State

The pitch module is **stateless** — it uses no persistent structs. All state is passed via parameters and stack-allocated temporaries. This is a deliberate design choice: pitch state is maintained by the caller (CELT encoder/decoder) in `CELTEncoder` and `CELTDecoder` structs.

### Stack-Allocated Temporaries

| Function | Variable | Type | Size | Purpose |
|----------|----------|------|------|---------|
| `pitch_search` | `x_lp4` | `opus_val16[]` | `len >> 2` | 4x decimated reference |
| `pitch_search` | `y_lp4` | `opus_val16[]` | `lag >> 2` | 4x decimated past signal |
| `pitch_search` | `xcorr` | `opus_val32[]` | `max_pitch >> 1` | Cross-correlation buffer |
| `remove_doubling` | `yy_lookup` | `opus_val32[]` | `maxperiod + 1` | Precomputed energy table |

These use `VARDECL`/`ALLOC` macros (VLA or `alloca`, depending on build config).

---

## 4. Algorithm

### 4.1. Overall Pitch Detection Pipeline

The pitch detection pipeline is called from the CELT encoder and consists of three stages:

```
Input signal (celt_sig, full rate)
    │
    ▼
pitch_downsample()  ─── Downsample by factor + LPC whitening
    │
    ▼
pitch_search()  ─── Multi-resolution correlation search
    │                   ├── 4x decimated coarse search
    │                   ├── 2x decimated fine search
    │                   └── Pseudo-interpolation refinement
    ▼
remove_doubling()  ─── Octave error correction
    │
    ▼
Output: pitch period T0, pitch gain pg
```

### 4.2. `pitch_downsample` — Detailed Algorithm

**Step 1: Decimation with anti-alias filter**

A simple 3-tap FIR filter [0.25, 0.5, 0.25] is applied during downsampling. For sample index `i` at the output rate:

```
x_lp[i] = 0.25 * x[factor*i - offset] + 0.5 * x[factor*i] + 0.25 * x[factor*i + offset]
```

where `offset = factor / 2`. The boundary case `i=0` omits the `factor*i - offset` term.

If stereo (`C == 2`), the second channel is added to the first.

In fixed-point mode, a dynamic shift is computed first:
```
maxabs = max(celt_maxabs32(x[0]), celt_maxabs32(x[1]))
shift = celt_ilog2(maxabs) - 10    // Target ~10-bit headroom
if (C == 2) shift++                // Extra bit for summation
```

**Step 2: Autocorrelation and LPC analysis**

```c
_celt_autocorr(x_lp, ac, NULL, 0, 4, len, arch);  // 4th-order autocorrelation
```

Noise floor: `ac[0] *= 1.0001` (float) or `ac[0] += ac[0] >> 13` (fixed-point, ~-40 dB).

Lag windowing: `ac[i] -= ac[i] * (0.008*i)^2` — a Gaussian-shaped window that prevents ill-conditioned autocorrelation matrices.

**Step 3: LPC coefficient derivation**

```c
_celt_lpc(lpc, ac, 4);  // Levinson-Durbin, order 4
```

**Step 4: Bandwidth expansion**

Each LPC coefficient is scaled by `0.9^i`:
```c
tmp = 1.0;
for (i = 0; i < 4; i++) {
    tmp *= 0.9;          // Decaying envelope
    lpc[i] *= tmp;       // Move poles toward origin
}
```

This stabilizes the filter and ensures the whitening isn't too aggressive.

**Step 5: Add a zero and form 5-tap FIR**

A zero at z = -0.8 is convolved with the 4th-order LPC:
```c
lpc2[0] = lpc[0] + 0.8
lpc2[1] = lpc[1] + 0.8 * lpc[0]
lpc2[2] = lpc[2] + 0.8 * lpc[1]
lpc2[3] = lpc[3] + 0.8 * lpc[2]
lpc2[4] = 0.8 * lpc[3]
```

This is the convolution of `[1, lpc[0], lpc[1], lpc[2], lpc[3]]` with `[1, 0.8]`, yielding a 5-tap FIR filter with spectral-whitening properties plus a high-pass tilt (the zero at -0.8 suppresses low frequencies, which improves pitch detection robustness).

**Step 6: In-place FIR filtering via `celt_fir5`**

```c
celt_fir5(x_lp, lpc2, len);
```

The `celt_fir5` function applies the 5-tap FIR with direct-form structure, operating in-place. It uses `SIG_SHIFT` (= 12) for internal precision:
```
sum = x[i] << SIG_SHIFT
sum += num[0]*mem[0] + num[1]*mem[1] + ... + num[4]*mem[4]
x[i] = ROUND16(sum, SIG_SHIFT)
```

### 4.3. `pitch_search` — Multi-Resolution Search

**Stage 1: Decimate to 4x**

Both `x_lp` and `y` are decimated by an additional factor of 2 (simple subsampling, taking every other sample):
```c
x_lp4[j] = x_lp[2*j];   // len>>2 samples
y_lp4[j] = y[2*j];       // lag>>2 samples, where lag = len + max_pitch
```

In fixed-point, both are normalized to prevent overflow:
```c
shift = celt_ilog2(MAX(xmax, ymax)) - 14 + celt_ilog2(len)/2;
```

**Stage 2: Coarse correlation at 4x decimation**

```c
celt_pitch_xcorr(x_lp4, y_lp4, xcorr, len>>2, max_pitch>>2, arch);
find_best_pitch(xcorr, y_lp4, len>>2, max_pitch>>2, best_pitch, ...);
```

This finds the top-2 pitch candidates at 4x decimated resolution.

**Stage 3: Fine correlation at 2x decimation**

Only lags within ±2 of the two coarse candidates (scaled to 2x resolution) are evaluated:
```c
for (i = 0; i < max_pitch>>1; i++) {
    if (abs(i - 2*best_pitch[0]) > 2 && abs(i - 2*best_pitch[1]) > 2)
        continue;  // Skip lags far from coarse candidates
    xcorr[i] = inner_prod(x_lp, y+i, len>>1);
}
find_best_pitch(xcorr, y, len>>1, max_pitch>>1, best_pitch, ...);
```

This narrows the search to ~10 lags total (5 around each candidate) at 2x resolution.

**Stage 4: Pseudo-interpolation**

The final pitch is refined by examining the 3-point correlation around the best lag:
```c
a = xcorr[best - 1];
b = xcorr[best];
c = xcorr[best + 1];
if ((c - a) > 0.7 * (b - a))      offset = 1;
else if ((a - c) > 0.7 * (b - c)) offset = -1;
else                                offset = 0;
*pitch = 2 * best_pitch[0] - offset;
```

This is not true parabolic interpolation — it's a cheaper heuristic that shifts the estimate by ±1 sample at the non-decimated rate if one neighbor is significantly stronger.

### 4.4. `find_best_pitch` — Normalized Correlation Peak Finder

Finds the two lags with highest **normalized** correlation, using cross-comparison to avoid computing square roots:

```
Score(i) = xcorr[i]^2 / Syy(i)
```

Instead of comparing `xcorr[i]^2 / Syy(i) > xcorr[best]^2 / Syy(best)`, it cross-multiplies:
```c
if (num * best_den > best_num * Syy)  // where num = xcorr16^2, no division needed
```

The energy `Syy` is maintained incrementally as a sliding window:
```c
Syy += y[i+len]^2 - y[i]^2;  // Slide window by 1
Syy = MAX(1, Syy);            // Prevent division by zero
```

In fixed-point mode, `xcorr[i]` is shifted to 16-bit range before squaring to prevent overflow:
```c
xshift = celt_ilog2(maxcorr) - 14;
xcorr16 = EXTRACT16(VSHR32(xcorr[i], xshift));
num = MULT16_16_Q15(xcorr16, xcorr16);
```

In float mode, `xcorr16` is scaled by `1e-12f` to avoid underflow/overflow when squaring.

### 4.5. `remove_doubling` — Octave Error Correction

This is the most complex algorithm in the module. It checks whether the detected pitch is actually a harmonic (octave, fifth, etc.) of the true fundamental.

**Step 1: Work at half resolution**

All periods and lengths are halved at entry:
```c
maxperiod /= 2; minperiod /= 2; *T0_ /= 2; prev_period /= 2; N /= 2;
x += maxperiod;  // Advance pointer so x[0] is the start of the analysis window
```

**Step 2: Precompute energy lookup table**

```c
yy_lookup[0] = xx;  // Energy of x[0..N-1]
for (i = 1; i <= maxperiod; i++)
    yy_lookup[i] = yy + x[-i]^2 - x[N-i]^2;  // Sliding window energy
```

**Step 3: Check subharmonics**

For each divisor `k` from 2 to 15, compute the candidate period `T1 = round(T0/k)`:
```c
T1 = (2*T0 + k) / (2*k);  // Rounded division
```

A secondary check period `T1b` is also computed:
- For `k == 2`: `T1b = T0 + T1` (or `T0` if that exceeds `maxperiod`)
- For `k > 2`: `T1b = round(second_check[k] * T0 / k)` where `second_check[k]` provides pre-selected harmonics

The correlation at `T1` and `T1b` are averaged:
```c
dual_inner_prod(x, &x[-T1], &x[-T1b], N, &xy, &xy2, arch);
xy = (xy + xy2) / 2;
yy = (yy_lookup[T1] + yy_lookup[T1b]) / 2;
g1 = compute_pitch_gain(xy, xx, yy);
```

**Step 4: Apply thresholds with continuity bias**

A subharmonic is accepted if its gain exceeds a threshold that depends on:
- The original pitch gain `g0` (higher g0 → harder to beat)
- Continuity with the previous frame's pitch (`cont`)
- Period length (very short periods require higher thresholds to avoid false positives)

```c
cont = (|T1 - prev_period| <= 1) ? prev_gain :
       (|T1 - prev_period| <= 2 && 5*k*k < T0) ? prev_gain/2 : 0;

thresh = max(0.3, 0.7*g0 - cont);
if (T1 < 3*minperiod)  thresh = max(0.4, 0.85*g0 - cont);
if (T1 < 2*minperiod)  thresh = max(0.5, 0.9*g0 - cont);
```

**Step 5: Compute final pitch gain**

```c
pg = best_xy / (best_yy + 1);  // Using frac_div32 in fixed-point
```

The gain is clamped: `pg = min(pg, g)` — the normalized gain cannot exceed the raw gain.

**Step 6: Sub-sample refinement**

A 3-point correlation around the best period is computed, and the same 0.7-threshold pseudo-interpolation as `pitch_search` is applied. The final period is:
```c
*T0_ = 2*T + offset;  // Scale back to original resolution
if (*T0_ < minperiod0) *T0_ = minperiod0;
```

### 4.6. `xcorr_kernel_c` — Vectorized 4-Lag Correlation

The core kernel computes 4 correlations simultaneously by maintaining a sliding window of `y` values:

```
Iteration layout (4x unrolled main loop):

    y_0  y_1  y_2  y_3
     │    │    │    │
x[0]─┼────┼────┼────┤   sum[0] += x[0]*y_0  sum[1] += x[0]*y_1  ...
x[1]─┼────┼────┼────┤   sum[0] += x[1]*y_1  sum[1] += x[1]*y_2  ...
x[2]─┼────┼────┼────┤   sum[0] += x[2]*y_2  sum[1] += x[2]*y_3  ...
x[3]─┼────┼────┼────┤   sum[0] += x[3]*y_3  sum[1] += x[3]*y_0' ...
     │    │    │    │
     ▼    ▼    ▼    ▼
    y_0'  y_1' y_2' y_3' (shifted by 4)
```

Each iteration of the main loop processes 4 `x` values and advances `y` by 4. The 4 `y` registers rotate through positions, so each new `y` value is loaded only once. After the main loop, up to 3 remainder iterations handle `len % 4`.

The `sum[4]` array is an in-out parameter — the caller initializes it (typically to zero) and the kernel accumulates into it.

### 4.7. `celt_pitch_xcorr_c` — Full Cross-Correlation

Computes `xcorr[i] = <_x, _y + i>` for `i = 0 .. max_pitch - 1`.

The main loop processes 4 lags at a time via `xcorr_kernel`:
```c
for (i = 0; i < max_pitch - 3; i += 4) {
    sum[4] = {0, 0, 0, 0};
    xcorr_kernel(_x, _y + i, sum, len, arch);
    xcorr[i..i+3] = sum[0..3];
}
```

Remaining lags (0–3) use `celt_inner_prod` individually.

In fixed-point mode, the function also tracks and returns `maxcorr`, the maximum correlation value across all lags. This is used by `find_best_pitch` to compute the normalization shift.

### 4.8. `compute_pitch_gain` — Correlation Normalization

Computes `gain = xy / sqrt(xx * yy)`, returning a Q15 value in [-1.0, 1.0].

**Float version**: Trivial: `xy / sqrt(1 + xx*yy)`.

**Fixed-point version**: Elaborate shift management to avoid overflow:
```
1. sx = ilog2(xx) - 14        // Shift xx to ~14-bit
2. sy = ilog2(yy) - 14        // Shift yy to ~14-bit
3. x2y2 = (xx >> sx) * (yy >> sy) >> 14   // ~14-bit product
4. Adjust x2y2 to have even total shift (needed for integer sqrt)
5. den = celt_rsqrt_norm(x2y2)  // Reciprocal sqrt
6. g = den * xy                  // Multiply correlation by 1/sqrt(xx*yy)
7. Shift result to Q15, clamp to [-Q15ONE, Q15ONE]
```

---

## 5. Data Flow

### 5.1. Buffer Layout for `pitch_search`

```
x_lp:  [0 .............. len-1]           (whitened, 2x decimated signal)

y:     [0 .............. len+max_pitch-1] (past signal at 2x decimated rate)
       ╰── max_pitch ──╯╰──── len ────╯
       older ──────────────────────── newer

x_lp4: [0 ........ len/4-1]              (4x decimated reference)
y_lp4: [0 ........ lag/4-1]              (4x decimated past, lag = len + max_pitch)
xcorr: [0 ........ max_pitch/2-1]        (correlation output, reused between stages)
```

### 5.2. Buffer Layout for `remove_doubling`

```
x (input):  [0 ............... maxperiod+N-1]
            ╰── maxperiod ──╯╰──── N ────╯

After x += maxperiod:
x:          [-maxperiod .......... 0 ............. N-1]
             ╰── past signal ──╯  ╰── analysis window ──╯

yy_lookup:  [0 ................. maxperiod]
             Energy of x[-i .. -i+N-1] for lag i
```

### 5.3. Signal Type Transitions

```
celt_sig (Q27 / float)
    │
    ▼  pitch_downsample
opus_val16 (Q15 / float), downsampled by factor
    │
    ▼  pitch_search internal decimation
opus_val16 (Q15 / float), 4x and 2x decimated
    │
    ▼  pitch_search output
int (pitch period in 2x-decimated samples)
    │
    ▼  remove_doubling
int (pitch period in original samples), opus_val16 (pitch gain, Q15)
```

---

## 6. Numerical Details

### 6.1. Q-Format Summary

| Context | Type | Q-Format (fixed-point) | Range |
|---------|------|----------------------|-------|
| Input signal | `celt_sig` / `opus_val32` | Q27 (`SIG_SHIFT = 12` above Q15) | ±16.0 |
| Downsampled signal | `opus_val16` | Q15 | ±1.0 |
| Correlation values | `opus_val32` | Variable (depends on shift) | — |
| Pitch gain | `opus_val16` | Q15 | [0, 32767] = [0.0, 1.0] |
| LPC coefficients | `opus_val16` | Q15 | ±1.0 |
| FIR coefficients (`lpc2`) | `opus_val16` | Q12 (SIG_SHIFT) | ±8.0 |
| `Q15ONE` | constant | — | 32767 |

### 6.2. Overflow Prevention in `pitch_downsample`

Fixed-point downsampling computes a dynamic shift to keep values in 16-bit range:
```c
maxabs = celt_maxabs32(x[0], len*factor);
shift = celt_ilog2(maxabs) - 10;   // Target 10-bit values after shift
if (C == 2) shift++;                // Extra bit headroom for channel addition
```

The `SHR32(x[0][...], shift+2)` and `SHR32(x[0][...], shift+1)` calls ensure the 0.25/0.5 FIR weights combined with the dynamic shift keep the output within `opus_val16` range.

### 6.3. Overflow Prevention in `pitch_search`

At 4x decimation, an additional shift is computed:
```c
shift = celt_ilog2(MAX(xmax, ymax)) - 14 + celt_ilog2(len)/2;
```

The `celt_ilog2(len)/2` term accounts for accumulation of `len/4` multiply-accumulate operations. After shifting, `shift` is doubled ("use double the shift for a MAC") because the MAC product involves two shifted operands.

### 6.4. Overflow Prevention in `find_best_pitch`

The `yshift` parameter controls energy accumulation precision:
```c
Syy += SHR32(MULT16_16(y[i+len], y[i+len]), yshift);
```

The `xshift` variable normalizes correlation values to 14-bit before squaring:
```c
xshift = celt_ilog2(maxcorr) - 14;
xcorr16 = EXTRACT16(VSHR32(xcorr[i], xshift));
num = MULT16_16_Q15(xcorr16, xcorr16);  // 14-bit * 14-bit >> 15 = ~13-bit
```

The cross-multiplication comparison `num * best_den > best_num * Syy` uses `MULT16_32_Q15`, which is 16×32→32 bit — safe because `num` is ~13-bit and the denominators are 32-bit.

### 6.5. Precision in `compute_pitch_gain` (Fixed-Point)

This function performs `xy / sqrt(xx * yy)` entirely in integer arithmetic:

1. `xx` and `yy` are shifted to ~14-bit values
2. Their product is computed as 16×16→32 with a >>14 normalization
3. If the total shift `sx + sy` is odd, `x2y2` is adjusted (doubled or halved) to make it even — required because `celt_rsqrt_norm` expects its input in a specific Q format
4. `celt_rsqrt_norm(x2y2)` returns the reciprocal sqrt in Q14
5. The result is shifted by `(shift >> 1) - 1` to produce Q15 output
6. Final clamping to `[-Q15ONE, Q15ONE]`

### 6.6. Rounding Behavior

- `SHR32(a, shift)` — truncation toward negative infinity (arithmetic right shift)
- `PSHR32(a, shift)` — round-half-up: `(a + (1 << (shift-1))) >> shift`
- `ROUND16(x, a)` — uses `PSHR32`, so it rounds
- `celt_fir5` uses `ROUND16(sum, SIG_SHIFT)` — rounded conversion from Q27 to Q15
- `HALF32`/`HALF16` — truncating division by 2 (right shift by 1)

### 6.7. Float-Mode Simplifications

In float mode, all Q-format macros become identity operations or simple multiplications:
- `SHR32(a, shift)` → `a`
- `MULT16_16_Q15(a, b)` → `a * b`
- `QCONST16(x, bits)` → `x`
- `Q15ONE` → `1.0f`

The only float-specific code is in `find_best_pitch`:
```c
xcorr16 *= 1e-12f;  // Prevent underflow/overflow when squaring
```

---

## 7. Dependencies

### 7.1. Functions Called by This Module

| Function | Source | Purpose |
|----------|--------|---------|
| `_celt_autocorr` | `celt_lpc.c` | 4th-order autocorrelation for LPC analysis |
| `_celt_lpc` | `celt_lpc.c` | Levinson-Durbin LPC coefficient computation |
| `celt_ilog2` | `mathops.h` | Integer log2 (inline, uses `EC_ILOG`) |
| `celt_maxabs32` | `mathops.h` | Maximum absolute value of 32-bit array (fixed-point only) |
| `celt_maxabs16` | `mathops.h` | Maximum absolute value of 16-bit array (fixed-point only) |
| `celt_rsqrt_norm` | `mathops.c` | Reciprocal square root, normalized (fixed-point only) |
| `frac_div32` | `mathops.c` | Fractional division, Q31 result (fixed-point only) |
| `celt_udiv` | `entcode.h` | Unsigned integer division with lookup optimization |
| `celt_sqrt` | `mathops.h` | Square root (float mode only) |

### 7.2. Callers of This Module

| Caller | Function Called | Purpose |
|--------|---------------|---------|
| CELT encoder (`celt_encoder.c`) | `pitch_downsample` | Prepare signal for pitch analysis |
| CELT encoder | `pitch_search` | Find pitch period |
| CELT encoder | `remove_doubling` | Refine pitch / correct octave errors |
| CELT encoder | `xcorr_kernel` / `celt_inner_prod` | Pre-filter pitch analysis |
| CELT decoder (`celt_decoder.c`) | `comb_filter_const` | PLC and post-filter |
| CELT decoder | `celt_inner_prod` | PLC pitch search |
| SILK encoder/decoder | `celt_inner_prod` | Various correlation computations |

---

## 8. Constants and Tables

### 8.1. `second_check` Table

```c
static const int second_check[16] = {0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2};
```

Used in `remove_doubling` to select which harmonic relationship to verify when checking subharmonic `k`. For a candidate period `T0` and divisor `k`, the secondary check period is at `second_check[k] * T0 / k`.

| k | second_check[k] | Meaning |
|---|-----------------|---------|
| 2 | 3 | Check at 3/2 × T1 (special-cased to T0+T1) |
| 3 | 2 | Check at 2/3 × T0 |
| 4 | 3 | Check at 3/4 × T0 |
| 5 | 2 | Check at 2/5 × T0 |
| 6 | 5 | Check at 5/6 × T0 |
| 7 | 2 | Check at 2/7 × T0 |
| ... | ... | Pattern: small coprime numerators |

The rationale: when checking if period T0/k is a subharmonic, also verify correlation at a nearby harmonic that isn't simply T0 itself. This increases confidence that the detected subharmonic is real, not an artifact.

### 8.2. Hardcoded Constants

| Constant | Value | Context | Derivation |
|----------|-------|---------|-----------|
| `0.8` | `QCONST16(.8f, 15)` = 26214 | High-pass zero in `pitch_downsample` | Empirical; suppresses DC/low-freq for robust pitch detection |
| `0.9` | `QCONST16(.9f, 15)` = 29491 | LPC bandwidth expansion | Standard bandwidth expansion factor |
| `1.0001` | `ac[0] *= 1.0001f` | Noise floor | -40 dB, prevents ill-conditioned LPC |
| `0.008*i` | Lag window | Autocorrelation windowing | Approximation of `exp(-0.5*(2π*0.002*i)^2)` |
| `0.7` | Interpolation threshold | Pseudo-interpolation | Empirical threshold for sub-sample refinement |
| `0.3, 0.4, 0.5` | Minimum thresholds | `remove_doubling` | Floor gains for subharmonic acceptance |
| `0.7, 0.85, 0.9` | Gain multipliers | `remove_doubling` | Scaling of g0 for different period ranges |
| `1e-12f` | Float scaling | `find_best_pitch` | Prevents underflow/overflow in xcorr² |

### 8.3. Key Divisor: Factor 2

The entire pipeline operates at successive powers-of-2 decimation:
- `pitch_downsample`: `factor`× decimation (typically 2)
- `pitch_search`: Additional 2× (total 4×) for coarse search
- `remove_doubling`: Internal 2× decimation

This means pitch periods are always even at the output of `remove_doubling` (before the ±1 interpolation offset). The minimum resolution is 1 sample at the 2× decimated rate = 2 samples at full rate.

---

## 9. Edge Cases

### 9.1. Zero/Silent Input

- `pitch_downsample`: If `maxabs < 1`, it's clamped to 1 (fixed-point), preventing shift = -∞.
- `find_best_pitch`: `Syy` is initialized to 1 (not 0), preventing division by zero. The `MAX32(1, Syy)` call after each sliding update enforces this invariant.
- `compute_pitch_gain`: Returns 0 if any of `xy`, `xx`, `yy` is zero.
- `remove_doubling`: `best_yy + 1` in `frac_div32` prevents division by zero.

### 9.2. Very Short Pitch Periods

`remove_doubling` applies progressively stricter thresholds for short periods:
- `T1 < 3*minperiod`: threshold raised to max(0.4, 0.85*g0 - cont)
- `T1 < 2*minperiod`: threshold raised to max(0.5, 0.9*g0 - cont)
- `T1 < minperiod`: loop breaks (`break`)

### 9.3. Boundary Conditions

- `pitch_search` pseudo-interpolation: guarded by `best_pitch[0] > 0 && best_pitch[0] < (max_pitch>>1)-1` to prevent out-of-bounds array access.
- `remove_doubling`: `if (*T0_ >= maxperiod) *T0_ = maxperiod - 1` at entry.
- `remove_doubling`: `if (*T0_ < minperiod0) *T0_ = minperiod0` at exit.
- `xcorr_kernel_c`: `celt_assert(len >= 3)` — the unrolled structure requires at least 3 samples.
- `celt_pitch_xcorr_c`: `celt_assert(max_pitch > 0)` and alignment assertion `((size_t)_x & 3) == 0`.

### 9.4. Negative Correlations

- `find_best_pitch`: Only considers `xcorr[i] > 0` — negative correlations are ignored (a negative correlation means anti-correlated, not a useful pitch candidate).
- `remove_doubling`: `best_xy = MAX32(0, best_xy)` before gain computation.
- `remove_doubling`: `xcorr[i] = MAX32(-1, sum)` in the fine search — clamps at -1 to avoid large negative values distorting `find_best_pitch`.

### 9.5. Stereo Handling

`pitch_downsample` is the only function that handles stereo. It sums channels after downsampling. All subsequent processing operates on the mono sum. The extra `shift++` for stereo prevents overflow from the channel addition.

---

## 10. Porting Notes for Rust

### 10.1. Pointer Arithmetic and Negative Indexing

The most significant porting challenge. `remove_doubling` advances a pointer and then uses negative indices:
```c
x += maxperiod;
// Then: x[-T1], x[-T1b], x[-i], x[N-i], etc.
```

**Rust approach**: Use a base index offset rather than pointer arithmetic. Store the original slice and compute `base + offset` for all accesses:
```rust
let base = maxperiod;  // x[base] corresponds to C's x[0] after advance
// x[-T1] → x[base - T1]
// x[N-i] → x[base + N - i]
```

Similarly, `xcorr_kernel_c` advances both `x` and `y` pointers within the loop body using `*x++` and `*y++`. Convert to explicit index tracking.

### 10.2. Dual Fixed-Point / Float Compilation

The C code uses `#ifdef FIXED_POINT` pervasively — sometimes for entirely different function signatures (e.g., `compute_pitch_gain`, `celt_pitch_xcorr_c` return type).

**Rust approach**: Use a trait-based abstraction or generic type parameter. The `opus_val16`/`opus_val32` types and all arithmetic macros (`MAC16_16`, `MULT16_16_Q15`, etc.) should be methods on a trait that has both `i16`/`i32` and `f32` implementations.

Key differences between modes:
- `find_best_pitch` has extra parameters `yshift` and `maxcorr` in fixed-point
- `celt_pitch_xcorr_c` returns `opus_val32` in fixed-point, `void` in float
- `compute_pitch_gain` has completely different implementations
- `pitch_downsample` and `pitch_search` have different normalization code paths

### 10.3. `VARDECL` / `ALLOC` Stack Allocation

The C code uses VLAs or `alloca` for stack-allocated arrays whose sizes depend on runtime values:
```c
ALLOC(x_lp4, len>>2, opus_val16);
ALLOC(yy_lookup, maxperiod+1, opus_val32);
```

**Rust approach**: Use `Vec` allocations. For hot paths, consider a bump allocator or a pre-allocated scratch buffer passed as a parameter. The `SAVE_STACK`/`RESTORE_STACK` pairs are no-ops in most configurations and can be ignored.

### 10.4. In-Place Mutation

Several functions modify their input buffers:
- `celt_fir5` modifies `x` in-place (reads `x[i]`, writes `x[i]` in the same loop)
- `pitch_search` receives `y` as mutable (though it only reads it; the mutability may be for the stack allocation macro system)
- `remove_doubling` modifies `*T0_` in-place (divides by 2, then writes result)

**Rust approach**: `celt_fir5` is safe for in-place because it reads `x[i]` before writing — but document this carefully. Use `&mut` parameters where the C code uses pointers that are written.

### 10.5. The `x[]` Array-of-Pointers Parameter

`pitch_downsample` takes `celt_sig * OPUS_RESTRICT x[]` — an array of channel pointers:
```c
x[0]  → pointer to channel 0 samples
x[1]  → pointer to channel 1 samples (if C==2)
```

**Rust approach**: `&[&[opus_val32]]` or `&[&[celt_sig]; 2]`. The `OPUS_RESTRICT` qualifier is irrelevant in safe Rust (the borrow checker enforces non-aliasing).

### 10.6. `OPUS_INLINE` Functions in Header

`xcorr_kernel_c`, `dual_inner_prod_c`, and `celt_inner_prod_c` are `static OPUS_INLINE` in `pitch.h`. They are included in every translation unit that uses them.

**Rust approach**: Mark these as `#[inline]` or `#[inline(always)]`. They should live in the pitch module and be `pub(crate)` since they're used by other modules (prefilter, PLC, SILK).

### 10.7. Architecture Dispatch

The `arch` parameter enables SIMD dispatch via function pointers or `#ifdef` overrides. For the initial port, all SIMD is skipped.

**Rust approach**: The `arch` parameter can be omitted entirely in the initial port. Later, Rust's `#[cfg(target_feature = "...")]` with `#[target_feature(enable = "...")]` can replace the C dispatch mechanism.

### 10.8. Macro-Generated Arithmetic

The fixed-point macros like `MAC16_16`, `MULT16_16_Q15`, etc. are the backbone of all computation. They must be ported as either:
1. **Inline functions** (preferred for type safety)
2. **Macro-like const fn** where possible

Critical: `MULT16_16` casts both operands to `opus_val16` before widening to `opus_val32`:
```c
#define MULT16_16(a,b) (((opus_val32)(opus_val16)(a))*((opus_val32)(opus_val16)(b)))
```
This truncation is intentional and **must be preserved** for bit-exactness. In Rust:
```rust
fn mult16_16(a: i32, b: i32) -> i32 {
    (a as i16 as i32) * (b as i16 as i32)
}
```

### 10.9. Integer Overflow Semantics

C's signed integer overflow is undefined behavior, but Opus assumes two's complement throughout. Key overflow-sensitive spots:
- `SHL32` uses unsigned cast to avoid UB: `((opus_int32)((opus_uint32)(a) << (shift)))`
- `ADD32_ovflw`, `SUB32_ovflw` use unsigned casts for intentional wrapping

**Rust approach**: Use `wrapping_shl`, `wrapping_add`, `wrapping_sub` where the C code uses unsigned casts. Regular arithmetic should use checked operations in debug mode.

### 10.10. The `celt_udiv` Optimization

`celt_udiv` in `entcode.h` uses a lookup table for divisors ≤ 256, avoiding hardware division. It's called in `remove_doubling`:
```c
T1 = celt_udiv(2*T0+k, 2*k);
```

**Rust approach**: Implement as a regular division initially. The lookup table optimization can be added later if profiling shows it matters. Since `k` ranges 2–15, the divisor is always ≤ 30 and the lookup would apply.

### 10.11. `celt_assert` vs Rust Panics

The C code uses `celt_assert` for preconditions (e.g., `len >= 3`, `max_pitch > 0`). These should become `debug_assert!` in Rust, which are elided in release builds — matching the C behavior where `celt_assert` is typically a no-op.

### 10.12. `comb_filter_const_c` Separation

This function is declared in `pitch.h` but defined in `celt.c`, guarded by `NON_STATIC_COMB_FILTER_CONST_C`. It's also declared twice in the header (lines 175–177 and 192–194) — this duplication is harmless in C but should be a single declaration in Rust.

**Rust approach**: Define it in whichever module makes sense architecturally (likely `celt.rs` or a shared `comb_filter.rs`), and re-export from the pitch module if needed.

---

## Appendix: Function Call Graph

```
pitch_downsample
  ├── celt_maxabs32          (fixed-point: find dynamic range)
  ├── _celt_autocorr         (LPC analysis)
  ├── _celt_lpc              (Levinson-Durbin)
  └── celt_fir5              (whitening filter)

pitch_search
  ├── celt_maxabs16          (fixed-point: normalization)
  ├── celt_pitch_xcorr       (coarse correlation)
  │   └── xcorr_kernel       (4-at-a-time inner loop)
  ├── find_best_pitch        (top-2 peaks, ×2)
  └── celt_inner_prod        (fine search, individual lags)

remove_doubling
  ├── dual_inner_prod        (paired correlations)
  ├── compute_pitch_gain     (normalized gain)
  │   ├── celt_ilog2         (fixed-point: shift computation)
  │   ├── celt_rsqrt_norm    (fixed-point: reciprocal sqrt)
  │   └── celt_sqrt          (float: standard sqrt)
  ├── frac_div32             (fixed-point: fractional division)
  ├── celt_inner_prod        (sub-sample refinement)
  └── celt_udiv              (integer division with optimization)
```
