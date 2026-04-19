Now I have all the information needed. Here is the complete architecture document:

---

# FFT/MDCT Module Architecture — `kiss_fft.c` / `mdct.c`

## 1. Purpose

The FFT/MDCT module provides the time-frequency transform engine for the CELT layer of Opus. It implements:

- **Complex FFT/IFFT** — a mixed-radix Cooley–Tukey FFT supporting radices 2, 3, 4, and 5, derived from Mark Borgerding's KISS-FFT library and heavily modified by Jean-Marc Valin.
- **MDCT (Modified Discrete Cosine Transform)** — forward and inverse MDCT built on top of the N/4-point complex FFT, using pre/post-rotation twiddle factors.

The MDCT is the core analysis/synthesis transform in CELT: `clt_mdct_forward` maps overlapping time-domain windows to frequency-domain coefficients (encoder), and `clt_mdct_backward` performs the inverse plus windowed overlap-add (decoder).

A third file, `mini_kfft.c`, provides a standalone minimalist KISS-FFT with real-signal FFT support. It is used only in auxiliary tools (e.g., DNN training utilities, analysis scripts) — not in the codec's hot path. It is **not** needed for the Rust port of the core codec.

## 2. Public API

### 2.1 FFT Allocation / Free

```c
// Allocate FFT state for an nfft-point complex FFT.
// For non-custom modes, these are never called at runtime — pre-computed
// static tables are used instead. Only called under #ifdef CUSTOM_MODES.
kiss_fft_state *opus_fft_alloc(int nfft, void *mem, size_t *lenmem, int arch);

kiss_fft_state *opus_fft_alloc_twiddles(int nfft, void *mem, size_t *lenmem,
                                         const kiss_fft_state *base, int arch);

void opus_fft_free(const kiss_fft_state *cfg, int arch);
```

**Parameters:**
- `nfft` — FFT length (must factor into {2,3,4,5} only; or {2,4} only if `RADIX_TWO_ONLY`)
- `mem` / `lenmem` — optional pre-allocated buffer; if `lenmem == NULL`, uses `malloc`
- `base` — if non-NULL, shares twiddle table from `base` (for sub-sampled FFT sizes). The `shift` field records `log2(base->nfft / nfft)`.
- `arch` — CPU feature flags for arch-specific FFT (ARM NEON etc.)

### 2.2 FFT Execution

```c
// Forward FFT (out-of-place, fin != fout required)
void opus_fft_c(const kiss_fft_state *st,
                const kiss_fft_cpx *fin, kiss_fft_cpx *fout);

// Inverse FFT (out-of-place, fin != fout required)
void opus_ifft_c(const kiss_fft_state *st,
                 const kiss_fft_cpx *fin, kiss_fft_cpx *fout);

// Internal: operates on already bit-reversed data in fout (in-place).
// downshift: number of remaining right-shift bits (fixed-point only).
void opus_fft_impl(const kiss_fft_state *st, kiss_fft_cpx *fout
                   ARG_FIXED(int downshift));
```

The public-facing macros `opus_fft()` / `opus_ifft()` dispatch to `opus_fft_c()` / `opus_ifft_c()` (or to ARM NEON variants via RTCD function tables).

**Key constraint:** In-place FFT (`fin == fout`) is **not** supported. The code `celt_assert2(fin != fout)`.

### 2.3 MDCT Allocation / Free

```c
// Only under #ifdef CUSTOM_MODES — non-custom modes use static tables.
int clt_mdct_init(mdct_lookup *l, int N, int maxshift, int arch);
void clt_mdct_clear(mdct_lookup *l, int arch);
```

**Parameters:**
- `N` — MDCT size (full window length; FFT size = N/4)
- `maxshift` — maximum supported downshift (0..3). Each shift halves N, providing FFT states for N/4, N/8, N/16, N/32.

### 2.4 MDCT Forward / Backward

```c
// Forward MDCT: time → frequency. Scales output by 4/N. TRASHES `in`.
void clt_mdct_forward_c(const mdct_lookup *l,
    kiss_fft_scalar *in,                    // input (MODIFIED in-place)
    kiss_fft_scalar * OPUS_RESTRICT out,    // output spectrum
    const celt_coef *window,                // analysis window
    int overlap,                            // overlap size in samples
    int shift,                              // which sub-FFT to use (0..maxshift)
    int stride,                             // output stride
    int arch);

// Backward MDCT: frequency → time with windowed overlap-add.
// Implicit scale by 1/2 (via TDAC). Does NOT scale by 2/N.
void clt_mdct_backward_c(const mdct_lookup *l,
    kiss_fft_scalar *in,                     // input spectrum
    kiss_fft_scalar * OPUS_RESTRICT out,     // output time-domain buffer
    const celt_coef * OPUS_RESTRICT window,  // synthesis window
    int overlap,                             // overlap size in samples
    int shift,                               // which sub-FFT to use
    int stride,                              // input stride
    int arch);
```

**Parameters:**
- `shift` — selects a sub-sampled FFT size. `shift=0` → full size (N/4); `shift=1` → half (N/8); etc. Corresponds to `l->kfft[shift]`.
- `stride` — interleaving stride for input (backward) or output (forward). Used when processing multiple short MDCTs from a single long buffer.
- `overlap` — the overlap region in samples (typically `shortMdctSize`, e.g. 120 for 48kHz).
- `window` — pointer to the analysis/synthesis window coefficients (length = `overlap`).

## 3. Internal State

### 3.1 `kiss_fft_state`

```c
typedef struct kiss_fft_state {
    int nfft;                         // FFT length
    celt_coef scale;                  // 1/nfft (float) or Q15/Q31 approx (fixed)
    #ifdef FIXED_POINT
    int scale_shift;                  // celt_ilog2(nfft) — shift for MULT16_32_Q16 scaling
    #endif
    int shift;                        // twiddle table sub-sampling: -1 = owns twiddles,
                                      // >=0 = shared from base (twiddles[k<<shift])
    opus_int16 factors[2*MAXFACTORS]; // factorization: [p0,m0, p1,m1, ...] pairs
    const opus_int16 *bitrev;         // bit-reversal permutation table [nfft]
    const kiss_twiddle_cpx *twiddles; // twiddle factors [nfft] (shared if shift>=0)
    arch_fft_state *arch_fft;         // platform-specific state (NEON, etc.)
} kiss_fft_state;
```

**Lifecycle:** In non-custom modes, these are `static const` tables compiled from `static_modes_float.h` / `static_modes_fixed.h`. For custom modes, dynamically allocated via `opus_fft_alloc()` and freed via `opus_fft_free()`.

**Twiddle sharing:** When `shift >= 0`, the twiddles pointer is borrowed from the base (largest) FFT state. Sub-sampled FFTs access twiddles at stride `1 << shift`. Only the base state (`shift == -1`) owns and frees the twiddle memory.

### 3.2 `kiss_fft_cpx` / `kiss_twiddle_cpx`

```c
typedef struct { kiss_fft_scalar r; kiss_fft_scalar i; } kiss_fft_cpx;
typedef struct { kiss_twiddle_scalar r; kiss_twiddle_scalar i; } kiss_twiddle_cpx;
```

- **Float mode:** both are `{float, float}`
- **Fixed-point mode:** data is `{opus_int32, opus_int32}` (`kiss_fft_cpx`), twiddles are `{opus_int16, opus_int16}` (`kiss_twiddle_cpx`) — unless `ENABLE_QEXT`, in which case twiddles are also `opus_int32`.

### 3.3 `mdct_lookup`

```c
typedef struct {
    int n;                              // full MDCT size N
    int maxshift;                       // max allowed shift
    const kiss_fft_state *kfft[4];      // FFT states for N/4, N/8, N/16, N/32
    const kiss_twiddle_scalar *trig;    // pre/post-rotation twiddle table
} mdct_lookup;
```

**Lifecycle:** Same as `kiss_fft_state` — statically initialized for non-custom modes, dynamically allocated under `CUSTOM_MODES`.

**Twiddle layout:** The `trig` array is concatenated sub-tables for each shift level:
- `trig[0..N/2-1]` — twiddles for shift=0
- `trig[N/2..N/2+N/4-1]` — twiddles for shift=1
- etc.

At each shift level, for the effective N2 = N >> (shift+1):
- `trig[0..N4-1]` = `cos(2π(i+0.125)/N_effective)` — the "real" twiddle
- `trig[N4..N2-1]` = `cos(2π((i+N4)+0.125)/N_effective)` = effectively `sin(2π(i+0.125)/N_effective)` — the "imaginary" twiddle

### 3.4 `factors` Array Encoding

The factorization is stored as `[p0, m0, p1, m1, ...]` pairs where:
- `p[i]` = radix at stage i
- `m[i]` = sub-FFT length after stage i = m[i-1] / p[i]
- The product `p[i] * m[i]` = m[i-1], with m[-1] = nfft
- Terminated when m[i] == 1

Example for 480-point FFT: `[5, 96, 3, 32, 4, 8, 2, 4, 4, 1]`
- Stage 0: radix-5, sub-length 96 (480/5)
- Stage 1: radix-3, sub-length 32 (96/3)
- Stage 2: radix-4, sub-length 8 (32/4)
- Stage 3: radix-2, sub-length 4 (8/2)
- Stage 4: radix-4, sub-length 1 (4/4) — terminal

The order is reversed from natural factorization so the radix-4 ends up last, enabling the "degenerate case" optimization (`m==1` in `kf_bfly4`).

## 4. Algorithm

### 4.1 FFT Algorithm: Iterative Mixed-Radix Decimation-in-Time

The Opus FFT is a **decimation-in-time** (DIT) FFT that uses an iterative (non-recursive) approach:

1. **Bit-reversal permutation** — Input samples are scattered into output buffer in bit-reversed order, with optional scaling:
   ```c
   // Forward FFT: scale by 1/N during permutation
   fout[st->bitrev[i]].r = S_MUL2(fin[i].r, scale);
   fout[st->bitrev[i]].i = S_MUL2(fin[i].i, scale);
   ```
   ```c
   // Inverse FFT: no scaling, but conjugate input
   fout[st->bitrev[i]] = fin[i];
   fout[i].i = -fout[i].i;    // conjugate
   ```

2. **Butterfly stages** (`opus_fft_impl`) — Iterates from the innermost (smallest) factor to the outermost (largest):
   ```c
   // Iterate stages from L-1 down to 0
   for (i = L-1; i >= 0; i--) {
       switch (factors[2*i]) {
           case 2: kf_bfly2(fout, m, N_blocks); break;
           case 4: kf_bfly4(fout, fstride<<shift, st, m, N_blocks, m2); break;
           case 3: kf_bfly3(fout, fstride<<shift, st, m, N_blocks, m2); break;
           case 5: kf_bfly5(fout, fstride<<shift, st, m, N_blocks, m2); break;
       }
   }
   ```

3. **Inverse FFT** — Implemented as: conjugate → forward FFT (no scaling) → conjugate:
   ```c
   void opus_ifft_c(...) {
       fout[bitrev[i]] = fin[i];
       fout[i].i = -fout[i].i;      // conjugate input
       opus_fft_impl(st, fout, 0);   // forward FFT, no downshift
       fout[i].i = -fout[i].i;      // conjugate output
   }
   ```

### 4.2 Butterfly Functions

#### `kf_bfly4` — Radix-4 butterfly
The workhorse. Two code paths:
- **Degenerate case (`m==1`):** All twiddles are 1. Pure radix-4 DIT butterfly with no multiplications:
  ```
  scratch0 = F[0] - F[2];  F[0] += F[2];
  scratch1 = F[1] + F[3];  F[2] = F[0] - scratch1;  F[0] += scratch1;
  scratch1 = F[1] - F[3];
  F[1] = scratch0 + j*scratch1;
  F[3] = scratch0 - j*scratch1;
  ```
- **General case:** Applies twiddle multiplications at stride `fstride` from the shared twiddle table.

#### `kf_bfly2` — Radix-2 butterfly
Special-cased for `m==4` (always follows a radix-4 stage in standard modes). Uses a hardcoded `W8 = cos(π/4) ≈ 0.7071067812` twiddle factor for the intermediate elements rather than table lookups:
```c
tw = QCONST32(0.7071067812f, COEF_SHIFT-1);
// Element 0: trivial
// Element 1: multiply by W8 = (1+j)/√2 rotated
t.r = S_MUL(Fout2[1].r + Fout2[1].i, tw);
t.i = S_MUL(Fout2[1].i - Fout2[1].r, tw);
// Element 2: multiply by -j
t.r = Fout2[2].i;  t.i = -Fout2[2].r;
// Element 3: multiply by W8 conjugate
```

#### `kf_bfly3` — Radix-3 butterfly
Uses the constant `epi3.i = -sin(2π/3) ≈ -0.86602540`. In fixed-point, this is hardcoded as `QCONST32(0.86602540f, COEF_SHIFT-1)`.

#### `kf_bfly5` — Radix-5 butterfly
Uses two constants: `ya = e^{-j2π/5}` and `yb = e^{-j4π/5}`:
```
ya.r =  cos(2π/5) ≈  0.30901699     ya.i = -sin(2π/5) ≈ -0.95105652
yb.r = -cos(π/5)  ≈ -0.80901699     yb.i = -sin(4π/5) ≈ -0.58778525
```

### 4.3 MDCT Algorithm

The MDCT uses the "N/4 complex FFT" technique. For an MDCT of length N, we perform an N/4-point complex FFT.

#### Forward MDCT (`clt_mdct_forward_c`)

Given input `in[0..N-1]` and window `window[0..overlap-1]`:

**Step 1 — Window, shuffle, and fold** into N/2 real values `f[0..N/2-1]`:

The input is conceptually four blocks `[a, b, c, d]` each of length N/4. The windowed, folded output interleaves real/imaginary pairs:
```
f[2i]   = windowed combination of d-reversed and c-reversed  (real part)
f[2i+1] = windowed combination of b and a-reversed           (imag part)
```

Three loop regions handle the overlap and non-overlap portions:
1. `i < (overlap+3)>>2` — overlapped leading edge (windowed)
2. `(overlap+3)>>2 <= i < N4-(overlap+3)>>2` — non-overlapped middle (no window, just copy)
3. `i >= N4-(overlap+3)>>2` — overlapped trailing edge (windowed)

**Step 2 — Pre-rotation** (twiddle multiply before FFT):
```
for i in 0..N/4:
    re = f[2i], im = f[2i+1]
    yr = re*trig[i]    - im*trig[N/4+i]
    yi = im*trig[i]    + re*trig[N/4+i]
    f2[bitrev[i]] = (yr*scale, yi*scale)   // scale + bit-reversal in one pass
```

**Step 3 — N/4 complex FFT** (in-place on `f2`):
```c
opus_fft_impl(st, f2, scale_shift - headroom);
```

**Step 4 — Post-rotation** (deinterleave to output):
```
for i in 0..N/4:
    yr = fp[i].i*trig[N/4+i] - fp[i].r*trig[i]     // sin/cos rotation
    yi = fp[i].r*trig[N/4+i] + fp[i].i*trig[i]
    out[2*i*stride]          = yr                     // even-indexed outputs
    out[(N/2-1-2*i)*stride]  = yi                     // odd-indexed outputs (reversed)
```

The output is ordered so that even indices go forward and odd indices go backward, producing the standard MDCT output.

#### Inverse MDCT (`clt_mdct_backward_c`)

Given spectrum `in[0..N/2-1]`, output buffer `out[0..N-1]`:

**Step 1 — Pre-rotation** (with bit-reversal, direct into output buffer):
```
for i in 0..N/4:
    x1 = in[2*i*stride]                    // even spectral sample
    x2 = in[(N/2-1-2*i)*stride]            // odd spectral sample (reversed)
    yr =  x2*trig[i]    + x1*trig[N/4+i]
    yi =  x1*trig[i]    - x2*trig[N/4+i]
    // Swap real/imag because we use FFT instead of IFFT
    out[overlap/2 + 2*bitrev[i] + 1] = yr
    out[overlap/2 + 2*bitrev[i]]     = yi
```

**Step 2 — N/4 complex FFT** (in-place, on `out+overlap/2`):
```c
opus_fft_impl(l->kfft[shift], (kiss_fft_cpx*)(out+(overlap>>1)), fft_shift);
```
Note: uses forward FFT, not IFFT. The real/imaginary swap in pre-rotation compensates.

**Step 3 — Post-rotation** (in-place, from both ends simultaneously):
```
for i in 0..(N/4+1)/2:
    // Process pair from start (yp0) and end (yp1) simultaneously
    re = yp0[1], im = yp0[0]    // swapped real/imag
    yp0[0] = re*t[i] + im*t[N/4+i]
    yp1[1] = re*t[N/4+i] - im*t[i]
    // Mirror pair from other end
    re = yp1[1], im = yp1[0]
    yp1[0] = re*t[N/4-i-1] + im*t[N/2-i-1]
    yp0[1] = re*t[N/2-i-1] - im*t[N/4-i-1]
```

**Step 4 — Window and TDAC (Time-Domain Aliasing Cancellation):**
```
for i in 0..overlap/2:
    x1 = out[overlap-1-i]     // from "mirror" side
    x2 = out[i]               // from "direct" side
    out[i]          = x2*window[overlap-1-i] - x1*window[i]
    out[overlap-1-i] = x2*window[i] + x1*window[overlap-1-i]
```
This performs the windowed overlap-add with an implicit factor of 1/2 from the TDAC symmetry.

## 5. Data Flow

### Forward MDCT
```
in[N] (time-domain, windowed frame)
  │
  ├── Window + shuffle + fold ──► f[N/2] (interleaved re/im pairs)
  │
  ├── Pre-rotation (twiddle) ──► f2[N/4] (complex, bit-reversed)
  │
  ├── opus_fft_impl() ──► f2[N/4] (complex, frequency-domain)
  │
  └── Post-rotation ──► out[N/2] (real MDCT coefficients, strided)
```

### Inverse MDCT
```
in[N/2] (MDCT coefficients, strided)
  │
  ├── Pre-rotation + bitrev ──► out[overlap/2 .. overlap/2+N/2] (complex, bitrev'd)
  │
  ├── opus_fft_impl() ──► in-place complex FFT result
  │
  ├── Post-rotation (in-place, both ends) ──► time-domain samples
  │
  └── TDAC windowing ──► out[0..overlap-1] (overlap region windowed)
                          out[overlap..N-overlap-1] (pass-through)
```

### Buffer layout for inverse MDCT output

```
out[0 .. overlap-1]           → TDAC-windowed overlap region
out[overlap/2 .. overlap/2+N/2-1] → FFT working area (overlaps with above)
out[overlap .. N-1]           → pass-through (post-rotation fills, TDAC doesn't touch)
```

The overlap/2 offset is critical: the FFT works in the middle of the output buffer, and the TDAC step blends the overlap edges.

## 6. Numerical Details

### 6.1 Fixed-Point Formats

| Entity | Type | Q-format | Range |
|--------|------|----------|-------|
| FFT data (`kiss_fft_cpx`) | `opus_int32` | varies dynamically | ±2^31 |
| Twiddle factors (standard) | `opus_int16` | Q15 (±1.0 → ±32767) | `[-32767, 32767]` |
| Twiddle factors (QEXT) | `opus_int32` | Q31 | `[-2^31, 2^31-1]` |
| MDCT twiddles (standard) | `opus_int16` | Q15 | same as FFT twiddles |
| Scale factor (standard) | `opus_int16` | Q15 | `scale ≈ Q15ONE/nfft` |
| Scale factor (QEXT) | `opus_int32` | Q30 | `scale ≈ 2^30/nfft` |

### 6.2 Scaling Strategy

**Forward FFT (`opus_fft_c`):**
- Scales input by `1/nfft` during the bit-reversal step using `S_MUL2(x, scale)`.
- `S_MUL2` maps to `MULT16_32_Q16` in fixed-point, which is `(scale * x) >> 16`.
- Additional right-shifting (`fft_downshift`) happens during butterfly stages in fixed-point to prevent overflow.
- `scale_shift = celt_ilog2(nfft) - 1` tracks remaining shift budget.

**Inverse FFT (`opus_ifft_c`):**
- No scaling. Uses the conjugate-FFT-conjugate trick.
- `downshift = 0` passed to `opus_fft_impl`.

**Forward MDCT:**
- Scaling is `4/N` total. The 1/N factor comes from the FFT's own scaling; the factor of 4 comes from the MDCT's use of N/4-point FFT on an N-point signal.
- In fixed-point, scaling is split: `S_MUL2` in pre-rotation + `fft_downshift` in FFT stages.

**Inverse MDCT:**
- No explicit `1/N` scaling. The TDAC overlap-add provides an implicit factor of 1/2.
- A factor of 2 that would normally be needed is deferred to "when mixing the windows" (comment in code).

### 6.3 Fixed-Point Overflow Protection

The code uses two strategies:

1. **Overflow-wrapping arithmetic** — `ADD32_ovflw`, `SUB32_ovflw`, `NEG32_ovflw` use unsigned casts to get well-defined wrapping behavior (C standard guarantees unsigned overflow wraps). This is intentional — the butterfly stages rely on modular arithmetic correctness.

2. **Controlled downshifting** — `fft_downshift()` reduces magnitude between butterfly stages:
   ```c
   // Downshift budget allocated per stage:
   case 2: fft_downshift(fout, nfft, &downshift, 1);   // 1 bit
   case 4: fft_downshift(fout, nfft, &downshift, 2);   // 2 bits
   case 3: fft_downshift(fout, nfft, &downshift, 2);   // 2 bits
   case 5: fft_downshift(fout, nfft, &downshift, 3);   // 3 bits
   ```
   The `downshift` counter tracks remaining bits. Any leftover is applied after the final stage.

3. **Adaptive headroom in forward MDCT** — After pre-rotation, the code measures `maxval` and computes:
   ```c
   headroom = IMAX(0, IMIN(scale_shift, 28 - celt_ilog2(maxval)));
   ```
   This headroom is subtracted from the FFT's downshift budget and applied during post-rotation via `PSHR32(..., headroom)`.

4. **Inverse MDCT pre_shift/post_shift** — Scans input magnitudes to compute safe shift values:
   ```c
   pre_shift = IMAX(0, 29 - celt_zlog2(1 + maxval));
   post_shift = IMAX(0, 19 - celt_ilog2(ABS32(sumval)));
   post_shift = IMIN(post_shift, pre_shift);
   fft_shift = pre_shift - post_shift;
   ```
   - `pre_shift`: left-shift applied to input before FFT (maximizes precision)
   - `post_shift`: right-shift applied after post-rotation (prevents output overflow)
   - `fft_shift`: remaining shift budget for the FFT's internal downshifting

### 6.4 Rounding

- `PSHR32(a, shift)` = round-to-nearest: `(a + (1 << (shift-1))) >> shift`
- `SHR32(a, shift)` = truncation (round toward negative infinity)
- `fft_downshift` uses `SHR32` for shift==1, `PSHR32` for shift>1
- Post-rotation in both MDCT directions uses `PSHR32` for the final output

### 6.5 Key Multiplication Macros

| Macro | Fixed-point operation | Notes |
|-------|----------------------|-------|
| `S_MUL(a,b)` | `MULT16_32_Q15(b,a)` = `(int16)b * (int32)a >> 15` | Twiddle × data; note argument swap |
| `S_MUL2(a,b)` | `MULT16_32_Q16(b,a)` = `(int16)b * (int32)a >> 16` | Used only in FFT input scaling |
| `C_MUL(m,a,b)` | Complex multiply using `S_MUL` + `ADD32_ovflw`/`SUB32_ovflw` | Standard complex product |
| `C_MULBYSCALAR(c,s)` | `c.r = S_MUL(c.r, s); c.i = S_MUL(c.i, s)` | Scale by real scalar |
| `QCONST32(x,bits)` | `(int32)(0.5 + x * (1LL << bits))` | Compile-time float→fixed |

## 7. Dependencies

### What this module calls:
- `arch.h` — type definitions, fixed-point macros, `ARG_FIXED`
- `mathops.h` / `mathops.c` — `celt_ilog2()`, `celt_zlog2()`, `celt_cos_norm()` (for twiddle generation in custom modes)
- `os_support.h` — `opus_alloc()`, `opus_free()`
- `stack_alloc.h` — `VARDECL`, `ALLOC`, `SAVE_STACK`, `RESTORE_STACK` (stack-based temp allocation)
- `_kiss_fft_guts.h` — internal macros (`C_MUL`, `S_MUL`, `HALF_OF`, etc.)
- `fixed_generic.h` — generic fixed-point arithmetic primitives

### What calls this module:
- `celt_encoder.c` — calls `clt_mdct_forward()` to transform windowed audio frames
- `celt_decoder.c` — calls `clt_mdct_backward()` to reconstruct time-domain audio
- `modes.c` — statically references the MDCT/FFT tables from `static_modes_*.h`
- `bands.c` — does not call FFT/MDCT directly, but processes MDCT output

## 8. Constants and Tables

### 8.1 Static FFT Tables (48kHz mode)

For the standard 48kHz/960-sample mode, the MDCT has N=1920 and uses four FFT sizes:

| shift | FFT size (N/4) | Factorization | Use case |
|-------|---------------|---------------|----------|
| 0 | 480 | 5×3×4×2×4 | 20ms frame (LM=3) |
| 1 | 240 | 5×3×4×4 | 10ms frame (LM=2) |
| 2 | 120 | 5×3×2×4 | 5ms frame (LM=1) |
| 3 | 60 | 5×3×4 | 2.5ms frame (LM=0) |

All four share a single twiddle table (`fft_twiddles48000_960[480]`) — smaller FFTs access it at stride `1 << shift`.

### 8.2 MDCT Twiddle Table

`mdct_twiddles960[1800]` — concatenated cosine values:
- `[0..959]` — cos(2π(i+0.125)/1920) for i=0..959, shift=0
- `[960..1439]` — cos(2π(i+0.125)/960) for i=0..479, shift=1
- `[1440..1679]` — cos(2π(i+0.125)/480) for i=0..239, shift=2
- `[1680..1799]` — cos(2π(i+0.125)/240) for i=0..119, shift=3

Total: 960 + 480 + 240 + 120 = 1800 entries.

### 8.3 Bit-Reversal Tables

Pre-computed permutation tables: `fft_bitrev480[480]`, `fft_bitrev240[240]`, `fft_bitrev120[120]`, `fft_bitrev60[60]`.

### 8.4 Hardcoded Constants

| Constant | Value | Where used |
|----------|-------|-----------|
| W8 (cos(π/4)) | 0.7071067812 | `kf_bfly2`, hardcoded as `QCONST32(0.7071067812f, COEF_SHIFT-1)` |
| sin(2π/3) | 0.86602540 | `kf_bfly3`, as `epi3.i` |
| cos(2π/5) | 0.30901699 | `kf_bfly5`, `ya.r` |
| sin(2π/5) | 0.95105652 | `kf_bfly5`, `ya.i` |
| cos(4π/5) | 0.80901699 | `kf_bfly5`, `yb.r` (negated) |
| sin(4π/5) | 0.58778525 | `kf_bfly5`, `yb.i` (negated) |
| TRIG_UPSCALE | 1 | Fixed-point twiddle scaling (no-op) |
| TWID_MAX | 32767 | Maximum twiddle value (Q15) |
| SAMP_MAX | 2147483647 | Maximum sample value (int32) |

## 9. Edge Cases

1. **In-place FFT rejected:** `celt_assert2(fin != fout, "In-place FFT not supported")` in both `opus_fft_c` and `opus_ifft_c`.

2. **Unsupported FFT sizes:** `kf_factor()` returns 0 (failure) if `n` has prime factors > 5 (or > 4 if `RADIX_TWO_ONLY`). In practice, Opus only uses sizes that factor into {2,3,4,5}.

3. **Forward MDCT trashes input:** Explicitly documented and relied upon. The `in` buffer is used as scratch space during windowing/folding.

4. **Odd N/4 in inverse MDCT:** The post-rotation loop runs to `(N4+1)>>1`, meaning the middle element is computed twice when N/4 is odd. The code comments: "When N4 is odd, the middle pair will be computed twice."

5. **shift bounds:** `shift` must be ≤ `maxshift` (typically 3). The trig table navigation `trig += N` would overrun otherwise.

6. **`CUSTOM_MODES` gating:** `clt_mdct_init`, `clt_mdct_clear`, `opus_fft_alloc`, `opus_fft_free`, `kf_factor`, `compute_bitrev_table`, `compute_twiddles` are **only compiled** under `#ifdef CUSTOM_MODES`. Non-custom builds use static tables exclusively.

7. **Fixed-point scale computation edge case:** For power-of-2 FFT sizes, `scale = Q15ONE` exactly. For non-power-of-2 sizes, it uses integer division with rounding: `(1073741824+nfft/2)/nfft >> (15-scale_shift)`.

## 10. Porting Notes

### 10.1 Pointer Arithmetic and In-Place Mutation

The C code makes extensive use of pointer arithmetic to walk through arrays. Key patterns:

- **Forward/backward pointer pairs** (e.g., `xp1` walking forward, `xp2` walking backward in MDCT windowing). In Rust, use index-based iteration or split the slice.
- **`fout += 2*stride`** style increments. Use explicit index variables.
- **Casting `(kiss_fft_cpx*)(out+(overlap>>1))`** — reinterprets a `kiss_fft_scalar*` buffer as a complex array. In Rust, use a dedicated `[Complex; N/4]` allocation or `unsafe` transmute with alignment guarantees. Alternatively, work with interleaved `[i32; N/2]` and access `.r`/`.i` via index arithmetic.

### 10.2 Macro-Generated Arithmetic

All fixed-point arithmetic goes through macros (`S_MUL`, `C_MUL`, `ADD32_ovflw`, etc.). These must be faithfully ported as Rust functions/methods. Critical details:

- **`ADD32_ovflw`/`SUB32_ovflw`** use unsigned wrapping arithmetic. In Rust: `i32::wrapping_add()`, `i32::wrapping_sub()`.
- **`NEG32_ovflw(a)`** = `0u32.wrapping_sub(a as u32) as i32`. Not plain negation (which panics on `i32::MIN` in debug).
- **`S_MUL(a,b)` = `MULT16_32_Q15(b,a)`** — note the argument swap! The twiddle (16-bit) is the first argument to `MULT16_32_Q15`.
- **`S_MUL2(a,b)` = `MULT16_32_Q16(b,a)`** — uses Q16 shift instead of Q15. Only used for initial scaling.

### 10.3 Conditional Compilation Matrix

The code has four major compile-time configurations:

| Configuration | Data type | Twiddle type | Scale type | COEF_SHIFT |
|--------------|-----------|--------------|------------|------------|
| Float (default) | `f32` | `f32` | `f32` | N/A |
| Fixed-point (standard) | `i32` | `i16` | `i16` | 16 |
| Fixed-point + ENABLE_QEXT | `i32` | `i32` | `i32` | 32 |
| Float + USE_SIMD | `__m128` | `__m128` | `__m128` | N/A |

For the Rust port, implement float first, then fixed-point. SIMD can be deferred. Use generics or feature flags for the float/fixed split.

Key `#ifdef` blocks to track:
- `FIXED_POINT` — enables all fixed-point paths
- `ENABLE_QEXT` — 32-bit twiddles, 32-bit coefficients
- `CUSTOM_MODES` — dynamic allocation (vs. static tables)
- `RADIX_TWO_ONLY` — disables radix-3 and radix-5 (not used in standard Opus)
- `OVERRIDE_clt_mdct_forward` / `OVERRIDE_clt_mdct_backward` — platform-specific MDCT replacements

### 10.4 Stack Allocation

The MDCT uses `VARDECL`/`ALLOC` for temporary buffers:
```c
VARDECL(kiss_fft_scalar, f);    // N/2 elements
VARDECL(kiss_fft_cpx, f2);      // N/4 elements
ALLOC(f, N2, kiss_fft_scalar);
ALLOC(f2, N4, kiss_fft_cpx);
```

These map to either `alloca()` or a manually-managed stack. In Rust, use `Vec` with pre-allocated capacity, or fixed-size arrays if sizes are known at compile time. For the standard modes, N/4 max is 480, so stack allocation is safe.

### 10.5 Static Table Strategy

For non-custom modes (the default), all tables are `static const` generated by `dump_modes.c`. The Rust port should:

1. Generate equivalent `const` arrays (or `lazy_static` / `once_cell` if needed).
2. The `kiss_fft_state` structs reference their twiddle and bitrev tables by pointer; in Rust, use slice references with appropriate lifetimes, or embed the tables directly.
3. The `mdct_lookup.kfft[4]` array holds pointers to FFT states; use `&'static KissFftState` references.

### 10.6 Specific Tricky Patterns

1. **The `factors` array** uses paired `(radix, sub_length)` encoding read via pointer increment:
   ```c
   const int p = *factors++;  // radix
   const int m = *factors++;  // sub-length
   ```
   In Rust, use a slice of tuples `[(u16, u16); MAXFACTORS]` or iterate with `.chunks(2)`.

2. **The `shift` field semantics:** `-1` means "owns twiddles" (base FFT), `>= 0` means "shares twiddles at stride `1 << shift`". This affects both FFT execution (twiddle stride) and memory management (only base frees twiddles).

3. **Type punning in inverse MDCT:**
   ```c
   opus_fft_impl(l->kfft[shift], (kiss_fft_cpx*)(out+(overlap>>1)), fft_shift);
   ```
   This casts a `kiss_fft_scalar*` to `kiss_fft_cpx*`, treating consecutive scalar pairs as complex numbers. In Rust, either work with a separate `Vec<Complex>` or use `bytemuck`/`unsafe` transmute.

4. **The radix-2 butterfly in `kf_bfly2`** is not a standard radix-2. When `m==4`, it processes 4 elements at once with hardcoded twiddles for a merged radix-2×radix-4 optimization. It's really an 8-point DFT decomposition.

5. **Window pointer arithmetic in forward MDCT:** `wp1` and `wp2` walk in opposite directions through the window array. The three loop regions have different combinations of windowed vs. unwindowed samples.

### 10.7 `mini_kfft.c` — Separate, Optional

This is a self-contained, float-only, recursive KISS-FFT implementation used only by DNN tools. It differs from the main FFT in:
- Uses recursive `kf_work()` instead of iterative `opus_fft_impl()`
- Has an `inverse` flag (main FFT uses conjugation trick instead)
- Includes `mini_kiss_fftr` for real-to-complex FFT
- No fixed-point support, no bit-reversal tables, no twiddle sharing

**Recommendation:** Port this separately and only if DNN features are needed. It is not part of the core codec path.

### 10.8 Bit-Exactness Requirements

For bit-exact output matching the C reference:
- All fixed-point operations must produce identical results (wrapping behavior, rounding direction, shift amounts)
- The `MULT16_32_Q15` and `MULT16_32_Q16` implementations must match exactly — these are the most common source of off-by-one rounding errors
- The `PSHR32` rounding must be `(a + (1 << (shift-1))) >> shift`, not any other rounding mode
- Twiddle table values must be generated with the same formula and rounding
- For float builds, use `f32` (not `f64`) throughout to match C `float` precision
