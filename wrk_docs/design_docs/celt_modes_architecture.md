Now I have a thorough understanding of all the relevant source. Here is the complete architecture document.

---

# CELT Modes Module — Architecture Documentation

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

The modes module (`modes.c` / `modes.h`) is the **configuration factory** for the CELT codec layer within Opus. A `CELTMode` (aliased as `OpusCustomMode`) encapsulates every parameter the encoder and decoder need to process audio at a given sample rate and frame size: frequency band definitions, MDCT windowing, bit allocation tables, pulse caching, and pre-emphasis filter coefficients.

### Role in the Opus pipeline

```
┌──────────────────────────────────────────────────────┐
│                    Opus Encoder/Decoder                │
│  ┌──────────────┐   ┌──────────────┐                  │
│  │  SILK layer   │   │  CELT layer  │                  │
│  └──────────────┘   │              │                  │
│                     │  ┌──────────┐│                  │
│                     │  │CELTMode  ││ ← THIS MODULE    │
│                     │  │(config)  ││                  │
│                     │  └──────────┘│                  │
│                     │  bands.c     │                  │
│                     │  quant_bands │                  │
│                     │  rate.c      │                  │
│                     │  vq.c        │                  │
│                     │  mdct.c      │                  │
│                     └──────────────┘                  │
└──────────────────────────────────────────────────────┘
```

Every CELT operation — band splitting, bit allocation, transform, quantization — reads its parameters from the `CELTMode`. The mode is created once and shared (read-only) by all encoder/decoder instances at the same sample rate.

### Two operating paths

| Path | Condition | Behavior |
|------|-----------|----------|
| **Static modes** | `CUSTOM_MODES` undefined (standard Opus) | Returns a pointer to a compile-time `const` struct from `static_modes_fixed.h` or `static_modes_float.h`. Zero allocation. |
| **Custom modes** | `CUSTOM_MODES` defined (Opus Custom API) | Dynamically allocates and computes all mode tables at runtime. |

Standard Opus only uses 48 kHz / 960 samples (plus 96 kHz / 1920 with `ENABLE_QEXT`). The frame-size flexibility (120, 240, 480, 960 samples) is handled by the `maxLM` / `nbShortMdcts` mechanism — a single mode supports all four frame sizes for one sample rate.

---

## 2. Public API

### 2.1 `opus_custom_mode_create`

```c
CELTMode *opus_custom_mode_create(opus_int32 Fs, int frame_size, int *error);
```

**Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `Fs` | `opus_int32` | Sample rate in Hz. Standard: 8000, 12000, 16000, 24000, 48000. Custom: 8000–96000. |
| `frame_size` | `int` | Frame size in samples per channel at rate `Fs`. Must be even, >= 40, <= 1024 (or 2048 with QEXT). |
| `error` | `int *` | Output error code (nullable). `OPUS_OK` on success, `OPUS_BAD_ARG` or `OPUS_ALLOC_FAIL` on failure. |

**Returns:** Pointer to `CELTMode`. For static modes, this is a `const` pointer cast to mutable. For custom modes, caller-owned heap allocation. `NULL` on error.

**Behavior (static path):**
1. Iterates `static_mode_list[0..TOTAL_MODES-1]`
2. For each static mode, checks 4 possible LM values (j=0..3): `Fs == mode->Fs && (frame_size << j) == mode->shortMdctSize * mode->nbShortMdcts`
3. Returns the first match

The `j` loop means a single static mode at 48 kHz / 960 serves frame sizes 960, 480, 240, and 120. The caller just uses a different `LM` value derived from the shift.

**Behavior (custom path):** See [Section 4, Algorithm](#4-algorithm).

### 2.2 `opus_custom_mode_destroy`

```c
void opus_custom_mode_destroy(CELTMode *mode);
```

**Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `mode` | `CELTMode *` | Mode to destroy. NULL is a no-op. Static modes are silently ignored. |

**Behavior:**
1. `NULL` check → return
2. Checks if `mode` is in `static_mode_list` → return (static modes must not be freed)
3. Frees all dynamic allocations: `eBands`, `allocVectors`, `window`, `logN`, `cache.index`, `cache.bits`, `cache.caps`, MDCT twiddles, and the QEXT cache if applicable
4. Frees the mode struct itself

### 2.3 `compute_qext_mode` (ENABLE_QEXT only)

```c
void compute_qext_mode(CELTMode *qext, const CELTMode *m);
```

Populates a `CELTMode` for extended-bandwidth (>20 kHz) bands used by the Opus quality extension. Copies the base mode, then overwrites `eBands`, `logN`, `nbEBands`, `effEBands` with QEXT-specific tables, and copies the pre-computed QEXT pulse cache.

### 2.4 Supporting functions in `celt.c`

```c
int resampling_factor(opus_int32 rate);
```

Returns the integer downsampling factor from the mode's internal rate:

| Rate | Factor |
|------|--------|
| 96000 (QEXT) | 1 |
| 48000 | 1 |
| 24000 | 2 |
| 16000 | 3 |
| 12000 | 4 |
| 8000 | 6 |

```c
void init_caps(const CELTMode *m, int *cap, int LM, int C);
```

Computes per-band bit capacity limits from the mode's `cache.caps` table:
```c
N = (m->eBands[i+1] - m->eBands[i]) << LM;
cap[i] = (m->cache.caps[m->nbEBands*(2*LM+C-1)+i] + 64) * C * N >> 2;
```

---

## 3. Internal State

### 3.1 `OpusCustomMode` (a.k.a. `CELTMode`)

```c
struct OpusCustomMode {
    opus_int32  Fs;             // Sample rate (Hz)
    int         overlap;        // MDCT overlap in samples (window length)

    int         nbEBands;       // Total number of "pseudo-critical" bands
    int         effEBands;      // Effective bands (those below Nyquist for short MDCT)
    opus_val16  preemph[4];     // Pre/de-emphasis filter coefficients

    const opus_int16  *eBands;  // Band boundary table [nbEBands+1]

    int         maxLM;          // Maximum LM (log2 of max short MDCTs per frame)
    int         nbShortMdcts;   // Number of short MDCTs in a frame = 1 << maxLM
    int         shortMdctSize;  // Length of one short MDCT in samples

    int         nbAllocVectors; // Number of rows in allocation matrix
    const unsigned char *allocVectors;  // Bit allocation matrix [nbAllocVectors * nbEBands]
    const opus_int16 *logN;     // log2(band_width) per band in Q3 [nbEBands]

    const celt_coef *window;    // MDCT analysis/synthesis window [overlap]
    mdct_lookup mdct;           // Pre-computed MDCT twiddle factors
    PulseCache  cache;          // Pulse vector quantization cache
#ifdef ENABLE_QEXT
    PulseCache  qext_cache;     // Extended-bandwidth pulse cache
#endif
};
```

### 3.2 `PulseCache`

```c
typedef struct {
    int            size;    // Total number of entries in bits[]
    const opus_int16    *index;  // Index table [nbEBands * (maxLM+2)]
    const unsigned char *bits;   // Cached bit costs per pulse count
    const unsigned char *caps;   // Maximum reliable bit capacity per band
} PulseCache;
```

**`index`** maps `(LM, band)` → offset into `bits[]`. The index is structured as `(LM+1) * nbEBands` entries where `LM` is incremented by 1 (i.e., `cache.index[(LM+1)*nbEBands + band]`).

**`bits`** stores the pre-computed bit cost for each pulse count for each unique `(N, K)` pair, where `N` is the band dimension (number of MDCT bins in band) and `K` is the pulse count. The first byte at each offset is the maximum number of pseudo-pulses for that band size.

**`caps`** provides the maximum bit budget that can be reliably allocated to each band. Indexed as `nbEBands * (2*LM + C - 1) + band` where `C` is the channel count (1 or 2).

### 3.3 Lifecycle

| Phase | Static modes | Custom modes |
|-------|-------------|--------------|
| Creation | `opus_custom_mode_create()` returns `const` pointer | `opus_custom_mode_create()` allocates and initializes |
| Usage | Shared read-only across all encoders/decoders | Shared read-only (caller must ensure lifetime) |
| Destruction | `opus_custom_mode_destroy()` is a no-op | Frees all heap allocations |

Static modes live for the entire process lifetime. Custom mode pointers must outlive all encoder/decoder instances that reference them.

---

## 4. Algorithm

### 4.1 Static mode lookup (standard path)

```
opus_custom_mode_create(Fs, frame_size):
    for each mode in static_mode_list:
        for j in 0..3:
            if Fs == mode.Fs AND (frame_size << j) == mode.shortMdctSize * mode.nbShortMdcts:
                return mode
    return OPUS_BAD_ARG
```

The shift by `j` handles the four supported frame sizes per mode. For 48 kHz with `shortMdctSize=120` and `nbShortMdcts=8`:
- j=0: frame_size=960 (20 ms)
- j=1: frame_size=480 (10 ms)
- j=2: frame_size=240 (5 ms)
- j=3: frame_size=120 (2.5 ms)

### 4.2 Custom mode construction (dynamic path)

When `CUSTOM_MODES` is defined:

**Step 1: Parameter validation**
```
Reject if:
  Fs < 8000 or Fs > 96000
  frame_size < 40 or > 1024 (2048 with QEXT) or odd
  frame_size * 1000 < Fs  (sub-1ms frames)
```

**Step 2: Compute LM (short MDCT count exponent)**
```
if frame_size*75 >= Fs AND frame_size%16 == 0:   LM = 3   (8 short MDCTs)
elif frame_size*150 >= Fs AND frame_size%8 == 0:  LM = 2   (4 short MDCTs)
elif frame_size*300 >= Fs AND frame_size%4 == 0:  LM = 1   (2 short MDCTs)
else:                                             LM = 0   (1 short MDCT)
```

The thresholds ensure short blocks are <= 3.3 ms:
```
Reject if (frame_size >> LM) * 300 > Fs
```

**Step 3: Compute frame structure**
```
nbShortMdcts = 1 << LM
shortMdctSize = frame_size / nbShortMdcts
res = (Fs + shortMdctSize) / (2 * shortMdctSize)   // frequency resolution (Hz per bin)
```

**Step 4: Pre-emphasis filter coefficients**

Selected by sample rate range:

| Rate range | preemph[0] (a₁) | preemph[1] (a₂) | preemph[2] (1/gain) | preemph[3] (gain) |
|-----------|---------|---------|---------|---------|
| < 12 kHz (8 kHz) | 0.350 | -0.180 | 0.272 (Q`SIG_SHIFT`) | 3.677 (Q13) |
| 12–24 kHz (16 kHz) | 0.600 | -0.180 | 0.442 (Q`SIG_SHIFT`) | 2.260 (Q13) |
| 24–40 kHz (32 kHz) | 0.780 | -0.100 | 0.750 (Q`SIG_SHIFT`) | 1.333 (Q13) |
| >= 40 kHz (48 kHz) | 0.850 | 0.000 | 1.000 (Q`SIG_SHIFT`) | 1.000 (Q13) |
| 96 kHz (QEXT) | 0.923 | 0.220 | 1.513 (Q`SIG_SHIFT`) | 0.661 (Q13) |

The reference design is `A(z) = 1 - 0.85z⁻¹` at 48 kHz. Lower rates approximate this with a second-order filter to compensate for the narrower bandwidth. `preemph[2]` and `preemph[3]` are exact reciprocals.

**Step 5: Compute band edges — `compute_ebands()`**

For the standard 48 kHz rate where `Fs == 400 * frame_size`:
- Returns a copy of the static `eband5ms[]` table (22 entries, 21 bands)

For custom rates:
1. Find `nBark`: highest Bark band whose center < Nyquist
2. Find `lin`: boundary where Bark band spacing exceeds `res` (frequency resolution)
3. Below `lin`: linearly spaced bands at 1-bin intervals
4. Above `lin`: Bark-scale spacing, rounded to even bin counts
5. Smooth spacing at the transition boundary
6. Remove empty bands
7. Validate: each band <= last band width; each band <= 2× the previous

**Step 6: Compute effective bands**
```
effEBands = nbEBands
while eBands[effEBands] > shortMdctSize:
    effEBands--
```

Bands beyond Nyquist for the short MDCT are excluded from processing.

**Step 7: Compute overlap**
```
overlap = (shortMdctSize / 4) * 4   // round down to multiple of 4
```

**Step 8: Compute allocation table — `compute_allocation_table()`**

For standard rates: copies `band_allocation[]` directly.

For non-standard rates: interpolates from the standard `band_allocation[]` matrix by mapping each custom band's center frequency to the two nearest standard bands, then linearly interpolating the allocation values.

**Step 9: Compute window**

The MDCT window function is a "sine-of-squared-sine" window:

```
window[i] = sin(π/2 · sin²(π/2 · (i + 0.5) / overlap))
```

This is the Vorbis window (also known as the MDCT sine window with a squared-sine envelope). It satisfies the perfect-reconstruction constraint for MDCT overlap-add.

In fixed-point:
- Without QEXT: scaled to Q15 (`floor(0.5 + 32768.0 * w)`, clamped to 32767)
- With QEXT: scaled to Q31 (`floor(2147483648.0 * w)`, clamped to 2147483647)

**Step 10: Compute logN table**
```
for each band i:
    logN[i] = log2_frac(eBands[i+1] - eBands[i], BITRES)
```

`log2_frac(val, frac)` returns `log2(val)` in Q`frac` (= Q3 since `BITRES=3`), i.e., 1/8th-bit precision. This table is used by the rate allocation to scale bit budgets by band size.

**Step 11: Compute pulse cache — `compute_pulse_cache()`**

See [Section 4.3](#43-pulse-cache-computation).

**Step 12: Initialize MDCT**
```
clt_mdct_init(&mode->mdct, 2 * shortMdctSize * nbShortMdcts, maxLM, arch)
```

The MDCT size is `2 * frame_size` (the full MDCT length for the largest frame size).

### 4.3 Pulse cache computation

`compute_pulse_cache()` (in `rate.c`) pre-computes the bit cost for PVQ coding each band at each possible pulse count, for all `(LM, band)` combinations.

**Algorithm:**

1. **Scan unique band sizes**: For each `(LM_index, band)` pair, compute `N = (eBands[j+1] - eBands[j]) << LM_index >> 1`. If this `N` has been seen before (at a lower LM or earlier band), reuse its cache entry. Otherwise, allocate a new entry.

2. **Compute max pulses**: For each unique `N`, find maximum `K` such that `V(N, K)` fits in 32 bits, up to `MAX_PSEUDO=40`.

3. **Fill bit costs**: For each unique `(N, K_max)` entry, call `get_required_bits(tmp, N, get_pulses(K_max), BITRES)` which computes `log2(V(N, K))` in Q`BITRES` for each pulse count. Store `cache_bits[K] = required_bits[get_pulses(K)] - 1`.

4. **Compute caps**: For each `(LM, C, band)` triple, compute the maximum bit budget the band can reliably consume considering the PVQ encoding structure, theta coding, and band splitting.

The `get_pulses()` function maps pseudo-pulse indices to actual pulse counts:
```c
int get_pulses(int i) {
    return i < 8 ? i : (8 + (i & 7)) << ((i >> 3) - 1);
}
```

This provides linear spacing for K < 8, then exponential spacing for larger K, which keeps the cache compact.

### 4.4 QEXT mode computation

`compute_qext_mode()` creates a mode for the 20–48 kHz extended bands:

```c
void compute_qext_mode(CELTMode *qext, const CELTMode *m) {
    OPUS_COPY(qext, m, 1);           // shallow copy base mode
    // Select band table based on short MDCT size
    if (shortMdctSize * 48000 == 120 * Fs):
        qext->eBands = qext_eBands_240    // 15 bands for 240-sample MDCTs
    else if (shortMdctSize * 48000 == 90 * Fs):
        qext->eBands = qext_eBands_180    // 15 bands for 180-sample MDCTs
    qext->nbEBands = qext->effEBands = NB_QEXT_BANDS  // 14
    // Trim effEBands to Nyquist
    while (qext->eBands[qext->effEBands] > qext->shortMdctSize):
        qext->effEBands--
    qext->nbAllocVectors = 0   // no standard allocation for QEXT
    qext->allocVectors = NULL
    OPUS_COPY(&qext->cache, &m->qext_cache, 1)  // use pre-computed cache
}
```

---

## 5. Data Flow

### 5.1 Mode creation flow

```
                     Fs, frame_size
                          │
                          ▼
              ┌─────────────────────┐
              │ opus_custom_mode_   │
              │     create()        │
              └──────┬──────────────┘
                     │
         ┌───────────┼──────────────┐
         ▼           ▼              ▼
   static_mode   validate     custom path
   lookup        params       ─────────────────────────┐
   (fast path)   ─────┐                                │
         │            │                                 ▼
         │            ▼                     ┌──────────────────┐
         │      compute LM,                │ compute_ebands() │
         │      preemph                    └────────┬─────────┘
         │            │                             │
         │            ▼                             ▼
         │      ┌──────────────────┐    ┌───────────────────────┐
         │      │ allocate mode    │    │ compute_allocation_   │
         │      │ struct           │    │     table()            │
         │      └──────────────────┘    └───────────┬───────────┘
         │                                          │
         │            ┌─────────────────────────────┘
         │            ▼
         │      compute window, logN
         │            │
         │            ▼
         │      ┌──────────────────┐
         │      │ compute_pulse_   │
         │      │     cache()      │
         │      └────────┬─────────┘
         │               │
         │               ▼
         │      ┌──────────────────┐
         │      │ clt_mdct_init()  │
         │      └────────┬─────────┘
         │               │
         └───────┬───────┘
                 ▼
           CELTMode*
```

### 5.2 Mode consumption (read-only)

| Consumer | Fields read | Purpose |
|----------|------------|---------|
| `celt_encoder.c` | All fields | Frame encoding configuration |
| `celt_decoder.c` | All fields | Frame decoding configuration |
| `bands.c` | `eBands`, `nbEBands`, `effEBands`, `cache` | Band processing, spreading, folding |
| `quant_bands.c` | `eBands`, `nbEBands` | Energy quantization |
| `rate.c` | `eBands`, `nbEBands`, `cache`, `allocVectors`, `logN` | Bit allocation |
| `vq.c` | (via rate.c) | Vector quantization |
| `mdct.c` | `mdct`, `window`, `overlap` | MDCT transform |
| `pitch.c` | `eBands`, `nbEBands` | Pitch analysis |

### 5.3 Buffer layouts

**`eBands[]`**: Length `nbEBands + 1`. Values are MDCT bin indices for one short block. `eBands[i]` is the start bin of band `i`; `eBands[i+1] - eBands[i]` is the band width in bins. Indexed `[0..nbEBands]`.

For 48 kHz standard mode (`eband5ms[]`):
```
Band:  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20
Start: 0  1  2  3  4  5  6  7  8  10 12 14 16 20 24 28 34 40 48 60 78
End:   1  2  3  4  5  6  7  8  10 12 14 16 20 24 28 34 40 48 60 78 100
Width: 1  1  1  1  1  1  1  1  2  2  2  2  4  4  4  6  6  8  12 18 22
```

The actual number of MDCT coefficients per band in a frame is `(eBands[i+1] - eBands[i]) << LM` where `LM` is the frame size exponent.

**`allocVectors[]`**: Flat 2D matrix `[nbAllocVectors × nbEBands]`, row-major. Each row is one bit allocation "vector" corresponding to a quality level. Values are in units of 1/32 bit/sample. There are 11 rows (`BITALLOC_SIZE`).

**`window[]`**: Length `overlap`. Symmetric MDCT window, used for both analysis and synthesis. For overlap-add reconstruction, the window is applied as: `w[i]` for the rising edge and `w[overlap-1-i]` for the falling edge.

**`logN[]`**: Length `nbEBands`. `log2(band_width_in_bins)` in Q3 (1/8th bit precision).

**`cache.index[]`**: Length `nbEBands * (maxLM + 2)`. Maps `[(LM+1) * nbEBands + band]` → offset into `cache.bits[]`.

**`cache.bits[]`**: Variable length. At each offset, `bits[0]` = max pseudo-pulse count for this band size, then `bits[k]` for `k = 1..bits[0]` gives the bit cost (in Q`BITRES` minus 1) for `k` pseudo-pulses.

**`cache.caps[]`**: Length `(maxLM + 1) * 2 * nbEBands`. Indexed as `nbEBands * (2*LM + C - 1) + band`.

---

## 6. Numerical Details

### 6.1 Q-format conventions

| Quantity | Float build | Fixed-point build | Notes |
|----------|-------------|-------------------|-------|
| `preemph[0]` (a₁) | `float` | Q15 (`opus_val16`) | Pre-emphasis numerator |
| `preemph[1]` (a₂) | `float` | Q15 (`opus_val16`) | Pre-emphasis numerator (2nd tap) |
| `preemph[2]` | `float` | Q`SIG_SHIFT` = Q12 (`opus_val16`) | Inverse of de-emphasis gain |
| `preemph[3]` | `float` | Q13 (`opus_val16`) | De-emphasis gain |
| `window[]` | `float` (0..1) | Q15 (`opus_int16`) or Q31 (`opus_int32` with QEXT) | MDCT window |
| `logN[]` | `opus_int16` | `opus_int16` | Q3 (BITRES) |
| `allocVectors[]` | `unsigned char` | `unsigned char` | Units of 1/32 bit/sample |
| `cache.bits[]` | `unsigned char` | `unsigned char` | Q`BITRES` minus 1 |
| `eBands[]` | `opus_int16` | `opus_int16` | Bin indices (integer) |

### 6.2 BITRES precision

`BITRES = 3` means all bit counts and allocation decisions are carried out in 1/8th-bit precision. This is the fundamental fractional precision unit throughout the rate allocation subsystem.

### 6.3 Window function precision

The window function `w(i) = sin(π/2 · sin²(π/2 · (i+0.5)/overlap))` is computed in `double` during mode creation, then quantized:

**Float build:**
```c
window[i] = Q15ONE * sin(0.5*M_PI * sin(0.5*M_PI*(i+0.5)/overlap)
                                   * sin(0.5*M_PI*(i+0.5)/overlap));
```
Since `Q15ONE = 1.0f` in float builds, the window values are in [0, 1].

**Fixed-point build (standard):**
```c
window[i] = MIN32(32767, floor(0.5 + 32768.0 * sin(...)));
```
Q15, clamped to prevent overflow. The `floor(0.5 + ...)` is rounding-to-nearest.

**Fixed-point build (QEXT):**
```c
window[i] = MIN32(2147483647, 2147483648.0 * sin(...));
```
Q31 for higher precision in the 96 kHz mode.

### 6.4 Pre-emphasis coefficient precision

The `QCONST16(x, bits)` macro converts a float constant to fixed-point:
- Fixed-point: `(opus_val16)(0.5 + x * (1 << bits))`
- Float: identity (`x`)

Coefficients are given as explicit fixed-point-friendly values (e.g., `0.8500061035f` rather than `0.85f`) to ensure exact representation in Q15.

### 6.5 Overflow considerations

- `eBands[i]` values are `opus_int16`, max 100 (for 48 kHz) or 240 (for 96 kHz QEXT), well within range.
- Band width `(eBands[i+1]-eBands[i]) << LM` can reach `22 << 3 = 176` for the widest band at LM=3. The PVQ table guard (`208` max) prevents overflow in the CWRS combinatorial encoder.
- `allocVectors` values are `unsigned char` (max 200), which represents `200/32 = 6.25` bits/sample.
- Cache computations use `opus_int32` intermediates to avoid overflow during the `V(N,K)` calculation in `fits_in32()`.

---

## 7. Dependencies

### 7.1 Modules called by modes.c

| Module | Functions called | Purpose |
|--------|-----------------|---------|
| `rate.c` | `compute_pulse_cache()` | Build PVQ bit-cost cache |
| `cwrs.c` | `log2_frac()`, `get_required_bits()` | Compute PVQ bit costs (via rate.c) |
| `mdct.c` | `clt_mdct_init()`, `clt_mdct_clear()` | MDCT twiddle factor setup/teardown |
| `cpu_support.h` | `opus_select_arch()` | CPU feature detection for MDCT init |
| `os_support.h` | `opus_alloc()`, `opus_free()` | Heap allocation |
| `quant_bands.c` | (linked, not directly called by modes.c) | |

### 7.2 Modules that depend on modes

| Module | How it uses the mode |
|--------|---------------------|
| `celt_encoder.c` | Reads all fields during encoding; stores `CELTMode*` in encoder state |
| `celt_decoder.c` | Reads all fields during decoding; stores `CELTMode*` in decoder state |
| `bands.c` | `eBands`, `effEBands`, `nbEBands`, `cache` |
| `rate.c` | `eBands`, `nbEBands`, `allocVectors`, `nbAllocVectors`, `logN`, `cache` |
| `quant_bands.c` | `eBands`, `nbEBands` |
| `celt.c` | `eBands`, `nbEBands`, `cache.caps`, `window`, `overlap` |
| `pitch.c` | `eBands`, `nbEBands` |
| `opus_encoder.c` | Creates and stores mode; passes to CELT init |
| `opus_decoder.c` | Creates and stores mode; passes to CELT init |

### 7.3 Header dependencies

```
modes.h
├── opus_types.h
├── celt.h
│   ├── opus_types.h
│   ├── opus_defines.h
│   ├── opus_custom.h
│   ├── entenc.h
│   ├── entdec.h
│   ├── arch.h
│   └── kiss_fft.h
├── arch.h
├── mdct.h
└── entenc.h / entdec.h
```

---

## 8. Constants and Tables

### 8.1 `eband5ms[]` — Standard band edges

```c
static const opus_int16 eband5ms[] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  10, 12, 14, 16,
    20, 24, 28, 34, 40, 48, 60, 78, 100
};
```

22 entries defining 21 bands. Each value is a bin index for a 5 ms short MDCT (120 samples at 48 kHz → 120 bins per short block, but the table goes to 100 because only positive frequencies are stored). The bands transition from 1-bin width (200 Hz) in the low frequencies to 22-bin width (4.4 kHz) at the top, roughly following the Bark frequency scale.

**Derivation**: The first 8 bands are linear at 200 Hz spacing (one bin each at 200 Hz/bin resolution). Above 1.6 kHz, bandwidth increases following perceptual critical bands, with widths constrained to be even (for the band-splitting structure of PVQ) and no wider than twice the previous band.

### 8.2 `band_allocation[]` — Bit allocation matrix

```c
#define BITALLOC_SIZE 11
static const unsigned char band_allocation[BITALLOC_SIZE * 21] = { ... };
```

11 rows × 21 bands. Each row represents a bit allocation "recipe" for a different bitrate tier, from row 0 (all zeros, minimum quality) to row 10 (all 200, maximum). Values are in units of 1/32 bit per sample, where "sample" means per MDCT coefficient.

The 11 rows are interpolated during encoding based on the actual bit budget. The interpolation happens in `rate.c:clt_compute_allocation()`.

**Row semantics:**
- Row 0: silence (all zeros)
- Rows 1–4: low bitrate, heavy high-frequency rolloff
- Rows 5–8: medium bitrate, more balanced
- Row 9: high bitrate, still has high-frequency taper
- Row 10: maximum (all 200), used for lossless-ish encoding

### 8.3 `bark_freq[]` — Bark scale boundaries (CUSTOM_MODES only)

```c
static const opus_int16 bark_freq[26] = {
    0, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
    1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300,
    6400, 7700, 9500, 12000, 15500, 20000
};
```

25 critical bands covering 0–20 kHz based on the Bark frequency scale (from CCRMA). Used only in `compute_ebands()` for non-standard sample rates.

### 8.4 QEXT band tables

```c
// For 240-sample short MDCTs (96 kHz or 48 kHz with larger frames)
static const opus_int16 qext_eBands_240[] = {
    100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240
};

// For 180-sample short MDCTs
static const opus_int16 qext_eBands_180[] = {
    74, 82, 90, 98, 106, 114, 122, 130, 138, 146, 154, 162, 168, 174, 180
};
```

14 bands each (NB_QEXT_BANDS=14), covering the 20–48 kHz extended range with 10-bin uniform spacing (240-sample table) or 8-bin mostly-uniform spacing (180-sample table).

### 8.5 Static mode constants (48 kHz)

From `static_modes_float.h` / `static_modes_fixed.h`:

```c
static const CELTMode mode48000_960_120 = {
    .Fs             = 48000,
    .overlap        = 120,
    .nbEBands       = 21,
    .effEBands      = 21,
    .preemph        = {0.85f, 0.0f, 1.0f, 1.0f},
    .eBands         = eband5ms,        // shared static table
    .maxLM          = 3,
    .nbShortMdcts   = 8,
    .shortMdctSize  = 120,
    .nbAllocVectors = 11,
    .allocVectors   = band_allocation, // shared static table
    .logN           = logN400,
    .window         = window120,
    .mdct           = { /* pre-computed FFT twiddles */ },
    .cache          = {392, cache_index50, cache_bits50, cache_caps50},
};
```

The naming convention `mode{Fs}_{framesize}_{overlap}` encodes the mode parameters. `TOTAL_MODES` is 1 normally, or 2 with QEXT (adding `mode96000_1920_240`).

### 8.6 Key derived constants

| Constant | Value | Derivation |
|----------|-------|------------|
| `MAX_PERIOD` | 1024 | Maximum pitch period in samples |
| `DEC_PITCH_BUF_SIZE` | 2048 | Decoder pitch buffer (2 × MAX_PERIOD) |
| `BITALLOC_SIZE` | 11 | Number of allocation vectors |
| `COMBFILTER_MAXPERIOD` | 1024 | Maximum comb filter period |
| `COMBFILTER_MINPERIOD` | 15 | Minimum comb filter period |
| `NB_QEXT_BANDS` | 14 | Number of extended-bandwidth bands |
| `QEXT_PACKET_SIZE_CAP` | 3825 | Maximum QEXT packet size in bytes |
| `MAX_PSEUDO` | 40 | Maximum pseudo-pulse index in cache |
| `CELT_MAX_PULSES` | 128 | Maximum pulse count |
| `MAX_FINE_BITS` | 8 | Maximum fine energy quantization bits |

### 8.7 `LOG2_FRAC_TABLE` (in rate.c)

```c
static const unsigned char LOG2_FRAC_TABLE[24] = {
    0,
    8, 13,
    16, 19, 21, 23,
    24, 26, 27, 28, 29, 30, 31, 32,
    32, 33, 34, 34, 35, 36, 36, 37, 37
};
```

Lookup table for `log2_frac()`. Provides the fractional part of `log2(val)` at the required precision, indexed by the top bits of `val` after extracting the integer part of the logarithm.

---

## 9. Edge Cases

### 9.1 Error conditions

| Condition | Error code | Behavior |
|-----------|------------|----------|
| No matching static mode (no CUSTOM_MODES) | `OPUS_BAD_ARG` | Returns NULL |
| `Fs < 8000` or `Fs > 96000` | `OPUS_BAD_ARG` | Returns NULL |
| `frame_size < 40` | `OPUS_BAD_ARG` | Returns NULL |
| `frame_size > 1024` (or 2048 with QEXT) | `OPUS_BAD_ARG` | Returns NULL |
| `frame_size` is odd | `OPUS_BAD_ARG` | Returns NULL |
| Sub-1ms frames (`frame_size*1000 < Fs`) | `OPUS_BAD_ARG` | Returns NULL |
| Short blocks > 3.3ms | `OPUS_BAD_ARG` | Returns NULL |
| Widest band too large for PVQ table (> 208) | `OPUS_ALLOC_FAIL` | Goto failure |
| Any allocation failure | `OPUS_ALLOC_FAIL` | Goto failure, cleanup via `opus_custom_mode_destroy()` |

### 9.2 Destroying static modes

`opus_custom_mode_destroy()` silently no-ops when given a pointer that matches any entry in `static_mode_list[]`. This is critical because encoder/decoder initialization may acquire a static mode pointer and later pass it to destroy during cleanup.

### 9.3 NULL error pointer

The `error` output pointer is always checked for NULL before writing. Callers can pass NULL if they don't care about the error code.

### 9.4 Permuted arguments

The comment "The good thing here is that permutation of the arguments will automatically be invalid" refers to the fact that `Fs` (8000–96000) and `frame_size` (40–1024) have non-overlapping valid ranges, so swapping them always fails validation.

### 9.5 `effEBands` vs `nbEBands`

For standard 48 kHz with `shortMdctSize=120`, all 21 bands have `eBands[21]=100 <= 120`, so `effEBands == nbEBands == 21`. For lower sample rates or smaller MDCT sizes, some high-frequency bands may extend beyond Nyquist and are excluded: the codec uses `effEBands` for processing and `nbEBands` for table indexing.

### 9.6 Band edge at 0

`eBands[0]` is always 0 (DC). The first band always starts at bin 0.

---

## 10. Porting Notes

### 10.1 Type aliasing: `CELTMode` = `OpusCustomMode`

```c
#define CELTMode OpusCustomMode
```

In Rust, define a single struct `CeltMode` (or `OpusCustomMode`) and use a type alias for the other name. The typedef chain `CELTMode → OpusCustomMode` exists only for API compatibility.

### 10.2 Static vs dynamic modes — the two-path design

The most architecturally significant decision is how to handle the static/dynamic split:

**Option A (recommended):** Define `CeltMode` as an owned struct with `Arc`-based sharing. Provide `const` static instances via `lazy_static!` or `const` construction (if all fields can be const-initialized). The `mode_create` function returns a reference to the static instance for standard rates, or a newly-allocated `Arc<CeltMode>` for custom rates.

**Option B:** Always dynamically construct. Simpler code but wastes a small amount of startup time.

The static modes contain pointers to other static data (e.g., `eBands` points to `eband5ms`, `allocVectors` points to `band_allocation`). In Rust, these would be `&'static [i16]` slices. For custom modes, these would be `Vec<i16>` or `Box<[i16]>`. Consider using an enum:

```rust
enum ModeData<T> {
    Static(&'static [T]),
    Owned(Box<[T]>),
}
```

Or use `Cow<'static, [T]>` from the standard library.

### 10.3 Conditional compilation maze

The C code uses extensive `#ifdef` chains:

| Define | Effect |
|--------|--------|
| `CUSTOM_MODES` | Enables dynamic mode creation |
| `CUSTOM_MODES_ONLY` | Disables static mode lookup |
| `ENABLE_OPUS_CUSTOM_API` | Enables the Opus Custom encoder/decoder API |
| `FIXED_POINT` | Fixed-point arithmetic (vs float) |
| `ENABLE_QEXT` | Extended bandwidth (96 kHz, wider bands) |
| `ENABLE_RES24` | 24-bit internal resolution |

**Rust approach:** Use Cargo features. The most important split is `fixed-point` vs `float`. Consider:

```toml
[features]
default = ["float"]
float = []
fixed-point = []
qext = []
custom-modes = []
```

Use `cfg` attributes and possibly separate type definitions behind feature gates. The `celt_coef` type (Q15 vs Q31 vs f32) can be handled with a type alias gated on features.

### 10.4 The `failure` goto pattern

```c
mode = opus_alloc(sizeof(CELTMode));
if (mode==NULL) goto failure;
// ... more allocations, each guarded by goto failure ...
failure:
    if (error) *error = OPUS_ALLOC_FAIL;
    if (mode!=NULL) opus_custom_mode_destroy(mode);
    return NULL;
```

**Rust equivalent:** This maps naturally to `Result<CeltMode, OpusError>` with the `?` operator. Since Rust's `Drop` trait handles cleanup, you can use a builder pattern:

```rust
fn mode_create(fs: i32, frame_size: i32) -> Result<CeltMode, OpusError> {
    // validation...
    let ebands = compute_ebands(fs, short_mdct_size, res)?;
    let alloc_vectors = compute_allocation_table(&ebands, ...)?;
    let window = compute_window(overlap)?;
    // ... each step returns Result, ? propagates errors
    Ok(CeltMode { fs, ebands, alloc_vectors, window, ... })
}
```

No need for manual cleanup — partial construction failures drop already-constructed fields automatically.

### 10.5 In-place pointer arithmetic in `compute_pulse_cache`

The pulse cache construction writes into `cindex` (which is also `cache->index`) while reading from it to check for previously-seen band sizes:

```c
cindex[i*m->nbEBands+j] = -1;
// ... search earlier entries in cindex[] ...
if (cache->index[i*m->nbEBands+j] == -1 && N!=0) {
    cindex[i*m->nbEBands+j] = curr;
```

`cindex` and `cache->index` are aliased (same pointer). In Rust, this requires careful handling — you're reading from and writing to the same `Vec<i16>`. Since the reads are always to earlier indices than the current write position, this is safe but must be expressed without violating Rust's borrow rules. Use index-based access on the same `Vec`, or split the algorithm into a scan phase and a fill phase.

### 10.6 `const`-casting for destroy

```c
opus_free((opus_int16*)mode->eBands);
opus_free((unsigned char*)mode->allocVectors);
```

The fields are declared `const` (to prevent modification during use) but cast away for freeing. In Rust, this is unnecessary — use `Box<[i16]>` for owned data and `&'static [i16]` for static data. The `ModeData` / `Cow` approach in 10.2 handles this cleanly.

### 10.7 `mdct_lookup` by value

The `mdct` field is stored by value (not pointer) inside `OpusCustomMode`. In Rust, the MDCT lookup struct should be an owned field or an `Option<Box<MdctLookup>>` if it's large. The static modes pre-compute all twiddle factors as `const` arrays.

### 10.8 Pre-emphasis array as inline field

```c
opus_val16 preemph[4];
```

This is a fixed-size array inside the struct, not a pointer. Direct mapping in Rust: `preemph: [Val16; 4]` (or `[f32; 4]` in float mode).

### 10.9 Window function computation

The window uses `sin()` from `<math.h>` during construction, even in fixed-point builds. This is acceptable because mode construction happens once. In Rust, use `f64::sin()` for the computation, then quantize to the target format.

**Precision warning:** The fixed-point quantization uses `floor(0.5 + 32768.0 * ...)` which is "round half up" (not IEEE round-half-to-even). For bit-exact matching, Rust code must use the same rounding:

```rust
let w = (32768.0 * val + 0.5).floor() as i16;
let w = w.min(32767);  // Q15 clamp
```

### 10.10 `celt_coef` type switching

With `ENABLE_QEXT`, `celt_coef` changes from `opus_val16` (Q15) to `opus_val32` (Q31), affecting the window type and all coefficient multiplication macros. In Rust, this could be:

```rust
#[cfg(feature = "qext")]
type CeltCoef = i32;  // Q31

#[cfg(not(feature = "qext"))]
type CeltCoef = i16;  // Q15
```

Or use a trait-based approach for the arithmetic operations.

### 10.11 Thread safety

Static modes are inherently thread-safe (immutable shared state). Custom modes are also effectively immutable after creation. In Rust, `CeltMode` should implement `Send + Sync`. Use `Arc<CeltMode>` for shared ownership across encoder/decoder instances.

### 10.12 The `arch` parameter

`opus_select_arch()` returns a CPU feature detection bitmask used for runtime dispatch of SIMD-optimized paths. The initial Rust port should return 0 (generic C path). The `arch` parameter flows through `clt_mdct_init()` and into the FFT setup for selecting optimized twiddle factor layouts.

```c
static OPUS_INLINE int opus_select_arch(void) { return 0; }
```

### 10.13 `compute_ebands` — heap-allocated return

The custom-mode `compute_ebands()` returns a heap-allocated `opus_int16*` that the caller must free. In Rust, return `Vec<i16>` (or `Box<[i16]>`).

### 10.14 `fits_in32` — PVQ overflow guard

```c
static int fits_in32(int _n, int _k) {
    static const opus_int16 maxN[15] = { ... };
    static const opus_int16 maxK[15] = { ... };
    // ...
}
```

This function checks whether the combinatorial number `V(N, K)` fits in a `u32`. The lookup tables encode the maximum safe N for each K (and vice versa) up to index 14. Values >= 14 require the cross-check. Port these tables exactly — they are derived from the mathematical properties of the PVQ codebook and must match for bit-exact compatibility.

### 10.15 Allocation table interpolation

`compute_allocation_table()` performs frequency-domain interpolation of the standard `band_allocation` matrix onto non-standard band edges. The interpolation uses integer arithmetic:

```c
a1 = mode->eBands[j] * Fs / shortMdctSize - 400 * eband5ms[k-1];
a0 = 400 * eband5ms[k] - mode->eBands[j] * Fs / shortMdctSize;
result = (a0 * alloc[k-1] + a1 * alloc[k]) / (a0 + a1);
```

All intermediate values are `opus_int32`. Watch for overflow: `eBands[j] * Fs` can reach `240 * 96000 = 23,040,000`, well within `i32` range.

### 10.16 Summary of required Rust types

```rust
pub struct CeltMode {
    pub fs: i32,
    pub overlap: usize,
    pub nb_ebands: usize,
    pub eff_ebands: usize,
    pub preemph: [Val16; 4],           // Q15 or f32
    pub ebands: ModeSlice<i16>,        // Cow<'static, [i16]> or enum
    pub max_lm: usize,
    pub nb_short_mdcts: usize,
    pub short_mdct_size: usize,
    pub nb_alloc_vectors: usize,
    pub alloc_vectors: ModeSlice<u8>,
    pub log_n: ModeSlice<i16>,
    pub window: ModeSlice<CeltCoef>,   // Q15/Q31/f32
    pub mdct: MdctLookup,
    pub cache: PulseCache,
    #[cfg(feature = "qext")]
    pub qext_cache: PulseCache,
}

pub struct PulseCache {
    pub size: usize,
    pub index: ModeSlice<i16>,
    pub bits: ModeSlice<u8>,
    pub caps: ModeSlice<u8>,
}
```

Where `ModeSlice<T>` is `Cow<'static, [T]>` or a custom enum allowing either static or owned data.
