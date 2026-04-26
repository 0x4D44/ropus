Now I have all the information needed. Here is the complete architecture documentation.

---

# Module: quant_bands — Band Energy Quantization

## Reference Files

| File | Lines | Role |
|---|---|---|
| `celt/quant_bands.h` | 66 | Public API declarations |
| `celt/quant_bands.c` | 572 | Energy quantization/dequantization |
| `celt/laplace.h` | 57 | Laplace entropy coding API |
| `celt/laplace.c` | 235 | Laplace distribution codec |

---

## 1. Purpose

The `quant_bands` module quantizes and dequantizes **per-band log-domain energy envelopes** in the CELT codec. Every CELT frame encodes the spectral energy of each critical band; this module converts those energies between amplitude and log2 domains, entropy-codes them into the bitstream, and reconstructs them on the decoder side.

Energy quantization happens in **three hierarchical passes**, each refining the previous:

1. **Coarse quantization** — integer-resolution energy deltas coded with a Laplace distribution via the range coder. Uses inter-frame prediction or intra-frame coding, selected by a two-pass rate-distortion decision on the encoder.
2. **Fine quantization** — sub-integer refinement bits allocated by the bit allocator, coded as raw bits.
3. **Finalise** — any leftover bits after all other coding are used to add one more bit of energy precision per band, prioritized by the bit allocator.

The `laplace` module provides the entropy coding primitive: a discrete Laplace (double-sided geometric) distribution optimized for the range coder's 15-bit precision.

### Position in the Opus Pipeline

```
Encoder:                                        Decoder:
  bandE (amplitude)                               bitstream
      │                                               │
      ▼                                               ▼
  amp2Log2()              ┌──────────────┐     unquant_coarse_energy()
      │                   │              │            │
      ▼                   │  Laplace     │            ▼
  quant_coarse_energy() ──┤  entropy     ├── unquant_fine_energy()
      │                   │  coding      │            │
      ▼                   │              │            ▼
  quant_fine_energy() ────┤              ├── unquant_energy_finalise()
      │                   └──────────────┘            │
      ▼                                               ▼
  quant_energy_finalise()                     oldEBands (log2 energy)
      │                                               │
      ▼                                               ▼
  error[] (residual)                          denormalise_bands() [bands.c]
```

---

## 2. Public API

### 2.1 Domain Conversion

#### `amp2Log2`

```c
void amp2Log2(const CELTMode *m, int effEnd, int end,
              celt_ener *bandE, celt_glog *bandLogE, int C);
```

Converts per-band amplitude energies to mean-removed log2 domain.

| Parameter | Type | Description |
|---|---|---|
| `m` | `const CELTMode *` | Mode configuration (provides `nbEBands`) |
| `effEnd` | `int` | Last band with real energy data (exclusive) |
| `end` | `int` | Total bands to fill (bands `[effEnd, end)` get `-14.0`) |
| `bandE` | `celt_ener *` | Input amplitude energies, layout `[C][nbEBands]` |
| `bandLogE` | `celt_glog *` | Output log2 energies, layout `[C][nbEBands]` |
| `C` | `int` | Number of channels (1 or 2) |

**Formula (fixed-point):**
```
bandLogE[i + c*nbEBands] = celt_log2_db(bandE[i + c*nbEBands])
                         - (eMeans[i] << (DB_SHIFT - 4))
                         + GCONST(2.0)    // Q12→Q14 compensation
```

**Formula (float):**
```
bandLogE[i + c*nbEBands] = celt_log2(bandE[i + c*nbEBands]) - eMeans[i]
```

Bands `[effEnd, end)` are set to `-14.0` (silence floor).

#### `log2Amp` (STALE — inlined into `denormalise_bands`)

```c
void log2Amp(const CELTMode *m, int start, int end,
             celt_ener *eBands, const celt_glog *oldEBands, int C);
```

Declared in header but **no longer has a standalone definition**. Commit `ee2506b2` moved the logic into `denormalise_bands()` in `bands.c`. The header declaration is vestigial. **The Rust port should NOT implement this as a separate function** — the inverse conversion is handled entirely within `denormalise_bands`.

---

### 2.2 Encoder Functions

#### `quant_coarse_energy`

```c
void quant_coarse_energy(const CELTMode *m, int start, int end, int effEnd,
      const celt_glog *eBands, celt_glog *oldEBands, opus_uint32 budget,
      celt_glog *error, ec_enc *enc, int C, int LM,
      int nbAvailableBytes, int force_intra, opus_val32 *delayedIntra,
      int two_pass, int loss_rate, int lfe);
```

Top-level coarse energy quantization. Decides between intra/inter coding, optionally runs both and picks the better one.

| Parameter | Type | Direction | Description |
|---|---|---|---|
| `m` | `const CELTMode *` | in | Mode config |
| `start` | `int` | in | First band to encode |
| `end` | `int` | in | Last band to encode (exclusive) |
| `effEnd` | `int` | in | Effective last band (for distortion calc) |
| `eBands` | `const celt_glog *` | in | Current frame's log2 energies |
| `oldEBands` | `celt_glog *` | in/out | Previous frame's quantized energies; **updated in-place** |
| `budget` | `opus_uint32` | in | Total bit budget (bits) |
| `error` | `celt_glog *` | out | Quantization error per band |
| `enc` | `ec_enc *` | in/out | Range encoder state |
| `C` | `int` | in | Channels (1 or 2) |
| `LM` | `int` | in | Log2 of block overlap ratio (0–3, indexes frame size) |
| `nbAvailableBytes` | `int` | in | Available bytes for entire frame |
| `force_intra` | `int` | in | Force intra mode (1) or allow decision (0) |
| `delayedIntra` | `opus_val32 *` | in/out | Accumulated distortion metric for intra decision |
| `two_pass` | `int` | in | Try both intra and inter, pick best |
| `loss_rate` | `int` | in | Expected packet loss rate (affects intra bias) |
| `lfe` | `int` | in | Low-frequency effects mode (limits decay/energy) |

#### `quant_fine_energy`

```c
void quant_fine_energy(const CELTMode *m, int start, int end,
      celt_glog *oldEBands, celt_glog *error,
      int *fine_quant, int *extra_quant, ec_enc *enc, int C);
```

| Parameter | Type | Direction | Description |
|---|---|---|---|
| `fine_quant` | `int *` | in | (named `prev_quant` in implementation) Previous quantization bits per band |
| `extra_quant` | `int *` | in | Additional bits allocated per band |
| `oldEBands` | `celt_glog *` | in/out | Updated with fine correction |
| `error` | `celt_glog *` | in/out | Residual error, reduced by fine quant |

#### `quant_energy_finalise`

```c
void quant_energy_finalise(const CELTMode *m, int start, int end,
      celt_glog *oldEBands, celt_glog *error,
      int *fine_quant, int *fine_priority, int bits_left,
      ec_enc *enc, int C);
```

| Parameter | Type | Direction | Description |
|---|---|---|---|
| `fine_quant` | `int *` | in | Current total fine bits per band |
| `fine_priority` | `int *` | in | Priority class (0 or 1) per band |
| `bits_left` | `int` | in | Remaining bits to distribute |

---

### 2.3 Decoder Functions

Mirror the encoder with `ec_dec` instead of `ec_enc`. No `error` or `eBands` — the decoder reconstructs `oldEBands` from the bitstream alone.

#### `unquant_coarse_energy`

```c
void unquant_coarse_energy(const CELTMode *m, int start, int end,
      celt_glog *oldEBands, int intra, ec_dec *dec, int C, int LM);
```

The `intra` flag is read from the bitstream (1 bit with logp=3) inside this function when budget allows.

#### `unquant_fine_energy`

```c
void unquant_fine_energy(const CELTMode *m, int start, int end,
      celt_glog *oldEBands, int *fine_quant, int *extra_quant,
      ec_dec *dec, int C);
```

Note: parameter named `fine_quant` in header, `prev_quant` in implementation. These are the same array passed by the caller.

#### `unquant_energy_finalise`

```c
void unquant_energy_finalise(const CELTMode *m, int start, int end,
      celt_glog *oldEBands, int *fine_quant, int *fine_priority,
      int bits_left, ec_dec *dec, int C);
```

---

### 2.4 Laplace Coding API

#### `ec_laplace_encode`

```c
void ec_laplace_encode(ec_enc *enc, int *value, unsigned fs, int decay);
```

| Parameter | Type | Direction | Description |
|---|---|---|---|
| `value` | `int *` | in/out | Value to encode; **may be clamped** if tail runs out |
| `fs` | `unsigned` | in | Probability of zero × 32768 |
| `decay` | `int` | in | Decay rate for |val|±1, × 16384. Max 11456. |

**Critical**: `value` is a pointer because the encoder may reduce `|*value|` if the tail probability mass is exhausted. The caller must use the modified value for reconstruction.

#### `ec_laplace_decode`

```c
int ec_laplace_decode(ec_dec *dec, unsigned fs, int decay);
```

Returns the decoded integer value.

#### `ec_laplace_encode_p0` / `ec_laplace_decode_p0`

```c
void ec_laplace_encode_p0(ec_enc *enc, int value, opus_uint16 p0, opus_uint16 decay);
int ec_laplace_decode_p0(ec_dec *dec, opus_uint16 p0, opus_uint16 decay);
```

Alternative Laplace codec using ICDF tables. Used by DRED/DNN modules (not by `quant_bands`). Encodes sign separately, then magnitude in chunks of 7 using a decaying ICDF.

---

## 3. Internal State

### 3.1 No Persistent Structs

This module is **stateless** — it has no module-specific structs. All state is passed via parameters:

- `oldEBands[C * nbEBands]` — the quantized log2 energies from the previous frame, stored in the `CELTEncoder`/`CELTDecoder` state. This is the primary inter-frame state.
- `delayedIntra` — a single `opus_val32` tracking accumulated prediction distortion, stored in the encoder state.
- `prev[2]` — per-channel running prediction residual, local to each `quant_coarse_energy_impl` / `unquant_coarse_energy` call.

### 3.2 Buffer Layouts

All per-band arrays use the layout `[channel][band]`, accessed as `array[i + c * nbEBands]`:

```
Index:   0  1  2  ...  nbEBands-1  |  nbEBands  nbEBands+1  ...  2*nbEBands-1
         └── channel 0 ───────────┘  └── channel 1 ─────────────────────────┘
```

`nbEBands` is from `CELTMode` (typically 21 for 48kHz). Maximum is 25 (size of `eMeans`).

---

## 4. Algorithm

### 4.1 Coarse Energy Quantization (Encoder)

#### Step 1: Mode Decision

```
if force_intra OR (no two_pass AND delayedIntra > 2*C*(end-start) AND enough bytes):
    intra = true
```

If budget allows (tell + 3 ≤ budget), a 1-bit flag with logp=3 signals intra mode.

#### Step 2: Max Decay Calculation

```
max_decay = 16.0  (in log2 domain)
if (end - start > 10):
    max_decay = min(max_decay, 0.125 * nbAvailableBytes)
if lfe:
    max_decay = 3.0
```

This limits how fast energy can drop between frames — prevents single-bin bands from creating sharp energy drops.

#### Step 3: Two-Pass Trial (if `two_pass`)

1. Save encoder state
2. Run `quant_coarse_energy_impl` with **intra=1**, measuring `badness1`
3. Save intra bitstream bytes
4. Restore encoder state
5. Run `quant_coarse_energy_impl` with **intra=0**, measuring `badness2`
6. Pick intra if `badness1 < badness2` or (equal badness and intra uses fewer fractional bits after bias)

**Intra bias formula:**
```
intra_bias = (budget * delayedIntra * loss_rate) / (C * 512)
```

Higher loss rates bias toward intra to improve error resilience.

#### Step 4: Core Quantization (`quant_coarse_energy_impl`)

For each band `i` in `[start, end)`, for each channel `c`:

```python
# 1. Compute prediction residual
oldE = max(-9.0, oldEBands[i + c*nbEBands])
f = eBands[i + c*nbEBands] - coef * oldE - prev[c]

# 2. Round to nearest integer
qi = round(f)        # floor(f + 0.5) in fixed-point

# 3. Apply decay bound
decay_bound = max(-28.0, oldEBands[i + c*nbEBands]) - max_decay
if qi < 0 and x < decay_bound:
    qi += (decay_bound - x) >> DB_SHIFT    # prevent too-fast decay
    qi = min(qi, 0)

# 4. Bit-budget clipping
bits_left = budget - tell - 3*C*(end-i)
if i != start:
    if bits_left < 24: qi = min(1, qi)
    if bits_left < 16: qi = max(-1, qi)
if lfe and i >= 2: qi = min(qi, 0)

# 5. Entropy code
if budget - tell >= 15:
    ec_laplace_encode(enc, &qi, prob_model[2*min(i,20)] << 7,
                                prob_model[2*min(i,20)+1] << 6)
elif budget - tell >= 2:
    qi = clamp(qi, -1, 1)
    ec_enc_icdf(enc, 2*qi ^ -(qi<0), small_energy_icdf, 2)
elif budget - tell >= 1:
    qi = min(0, qi)
    ec_enc_bit_logp(enc, -qi, 1)
else:
    qi = -1

# 6. Update state
error[i + c*nbEBands] = f - (qi << DB_SHIFT)
q = qi << DB_SHIFT
oldEBands[i + c*nbEBands] = coef * oldE + prev[c] + q
prev[c] = prev[c] + q - beta * q
```

**Prediction coefficients** (indexed by `LM`):

| LM | Frame samples | `pred_coef` (Q15) | `beta_coef` (Q15) | Float equiv |
|----|---------------|--------------------|--------------------|-------------|
| 0 | 120 | 29440 | 30147 | 0.899, 0.920 |
| 1 | 240 | 26112 | 22282 | 0.797, 0.680 |
| 2 | 480 | 21248 | 12124 | 0.649, 0.370 |
| 3 | 960 | 16384 | 6554 | 0.500, 0.200 |
| intra | — | 0 | 4915 | 0.000, 0.150 |

The `prev[c]` accumulator provides inter-band smoothing: `beta` controls how quickly the prediction residual decays across bands.

#### Step 5: Delayed Intra Update

```python
if intra:
    delayedIntra = new_distortion
else:
    delayedIntra = pred_coef[LM]^2 * delayedIntra + new_distortion
```

### 4.2 Fine Energy Quantization

For each band `i` with `extra_quant[i] > 0`:

```python
extra = 1 << extra_quant[i]
prev = prev_quant[i] if prev_quant else 0

for each channel c:
    # Fixed-point: no rounding (floor toward negative infinity)
    q2 = (error[i+c*nbEBands] + (0.5 >> prev)) >> (DB_SHIFT - extra_quant[i] - prev)
    q2 = clamp(q2, 0, extra - 1)

    ec_enc_bits(enc, q2, extra_quant[i])

    # Compute offset: center of quantization bin
    offset = ((2*q2 + 1) >> (extra_quant[i] - DB_SHIFT + 1)) - 0.5
    offset >>= prev

    oldEBands[i+c*nbEBands] += offset
    error[i+c*nbEBands] -= offset
```

The `prev_quant[i]` parameter accounts for previously allocated fine bits — the offset is scaled down by `2^prev` to add precision within the existing quantization step.

### 4.3 Energy Finalise

Two priority passes (prio=0 then prio=1), spending one bit per band per channel:

```python
for prio in [0, 1]:
    for each band i (while bits_left >= C):
        if fine_quant[i] >= MAX_FINE_BITS(8) or fine_priority[i] != prio:
            continue
        for each channel c:
            q2 = 1 if error[i+c*nbEBands] >= 0 else 0
            ec_enc_bits(enc, q2, 1)
            offset = (q2 - 0.5) * 2^(14 - fine_quant[i] - 1) / 16384
            oldEBands[i+c*nbEBands] += offset
            error[i+c*nbEBands] -= offset
            bits_left -= 1
```

### 4.4 Coarse Energy Dequantization (Decoder)

Mirrors the encoder with symmetric `ec_laplace_decode` / `ec_dec_icdf` / `ec_dec_bit_logp` calls. Key differences from encoder:

- No rate-distortion decision — `intra` flag is read from bitstream
- No `error[]` tracking
- `prev[c]` uses `opus_val64` (64-bit on fixed-point to prevent accumulation drift)
- Fixed-point clamps `tmp` to `[-28.0, +28.0]`; float has no explicit clamp

---

## 5. Laplace Entropy Coding — Detailed Algorithm

### 5.1 Distribution Model

The Laplace codec models a discrete distribution over integers with:
- A peak probability `fs` at value 0
- Geometrically decaying probabilities for |val| ≥ 1
- A minimum probability `LAPLACE_MINP` (= 1) for tail values
- At least `LAPLACE_NMIN` (= 16) values guaranteed representable per side

The total probability space is 2^15 = 32768.

### 5.2 `ec_laplace_get_freq1` — First Non-Zero Probability

```c
static unsigned ec_laplace_get_freq1(unsigned fs0, int decay) {
    unsigned ft = 32768 - LAPLACE_MINP * (2 * LAPLACE_NMIN) - fs0;
    return ft * (opus_int32)(16384 - decay) >> 15;
}
```

Computes the probability of value ±1:
- `ft` = total probability mass available after reserving `fs0` for zero and `LAPLACE_MINP` for each of the 2×16 guaranteed tail symbols
- Returns `ft * (16384 - decay) / 32768`

The decay parameter is in Q14 (range 0–16384), where 16384 means maximum decay (probability drops to zero immediately).

### 5.3 Encoding Algorithm

```
if value == 0:
    encode interval [0, fs)
else:
    s = sign of value (0 for positive, -1 for negative)
    val = |value|
    fl = fs                          # cumulative lower bound
    fs = get_freq1(fs, decay) + MINP # frequency for val=1

    for i = 2 to val-1:             # walk up the geometric decay
        fs = 2 * (fs - MINP)        # undo the MINP addition
        fl += fs + 2*MINP            # accumulate (both signs)
        fs = fs * decay >> 15        # apply decay
        fs += MINP

    if geometric part exhausted (fs was 0 before MINP):
        # flat tail: each remaining value has probability MINP
        clamp val to available range
        update fl for flat-tail position
    else:
        # still in geometric part
        fl += fs if positive, fl unchanged if negative

    encode interval [fl, fl + fs) with 15-bit precision
```

**Critical detail**: The encoder may **modify `*value`** via clamping when the requested value falls in the flat tail and exceeds the representable range. The caller must use the updated value.

### 5.4 Decoding Algorithm

```
fm = ec_decode_bin(dec, 15)         # read 15-bit symbol
fl = 0; val = 0

if fm >= fs:                        # not zero
    val = 1
    fl = fs
    fs = get_freq1(fs, decay) + MINP

    while fs > MINP and fm >= fl + 2*fs:   # geometric part
        fs = 2 * (fs - MINP)
        fl += fs + 2*MINP
        fs = (fs - 2*MINP) * decay >> 15 + MINP
        val++

    if fs <= MINP:                  # flat tail
        di = (fm - fl) / (2 * MINP)
        val += di
        fl += 2 * di * MINP

    if fm < fl + fs:
        val = -val                  # negative side
    else:
        fl += fs                    # positive side

ec_dec_update(dec, fl, min(fl+fs, 32768), 32768)
```

### 5.5 The `_p0` Variant

Used by DRED/DNN modules. Different structure:
1. Encode sign as a 3-symbol ICDF: `{zero, positive, negative}`
2. Encode magnitude minus 1 in chunks of 7 using a decaying 8-entry ICDF
3. Stop when chunk value < 7

---

## 6. Numerical Details

### 6.1 Q-Formats

| Quantity | Fixed-Point Format | Range | Notes |
|----------|-------------------|-------|-------|
| `celt_glog` (log energy) | Q24 (`DB_SHIFT=24`) | ~±128.0 | Main energy representation |
| `eMeans[i]` | Q4 (stored as `signed char`) | 60–103 | Shifted left by `DB_SHIFT-4` = 20 bits |
| `pred_coef`, `beta_coef` | Q15 | 0.0–1.0 | Fixed 16-bit coefficients |
| `e_prob_model` values | Q8 (unsigned char) | 0–255 | Shifted to Q15 for Laplace: `p0 << 7`, `decay << 6` |
| Laplace `fs` | Q15 (15-bit) | 0–32768 | Total probability space |
| Laplace `decay` | Q14 | 0–11456 (max) | Geometric decay rate |
| `error[]` | Q24 (same as `celt_glog`) | — | Fractional residual |
| `bandE` (amplitude) | Q12 (`SIG_SHIFT`) | — | Input to `amp2Log2` |

### 6.2 Overflow Guards

1. **`oldE` floor**: Clamped to `max(-9.0, oldEBands[i])` in encoder to prevent prediction from diverging on very low energies.

2. **Decoder range clamp** (fixed-point only):
   ```c
   tmp = MIN32(GCONST(28.f), MAX32(-GCONST(28.f), tmp));
   ```
   Prevents `oldEBands` from exceeding `±28.0` in Q24 = `±469,762,048`.

3. **Encoder lower bound** (fixed-point only):
   ```c
   tmp = MAX32(-GCONST(28.f), tmp);
   ```

4. **`amp2Log2` compensation**: In fixed-point, `bandE` is Q12 but `celt_log2()` expects Q14, so `+2.0` in log2 domain compensates the 4× scale difference.

5. **Decay bound**: Prevents energy from dropping more than `max_decay` per frame:
   ```c
   decay_bound = max(-28.0, oldEBands[i]) - max_decay
   ```

6. **Laplace tail clamping**: When encoding, if `|val|` exceeds the representable range in the flat tail, the value is clamped:
   ```c
   ndi_max = (32768 - fl + LAPLACE_MINP - 1) >> LAPLACE_LOG_MINP;
   ndi_max = (ndi_max - s) >> 1;
   di = IMIN(val - i, ndi_max - 1);
   ```

### 6.3 Rounding Behavior

- **Coarse quantization rounding**: Uses `floor(f + 0.5)` — round half up. In fixed-point: `(f + (1 << (DB_SHIFT-1))) >> DB_SHIFT`. The comment emphasizes this is "really important" — using different rounding creates bit-inexact output.

- **Fine quantization**: Uses floor (no rounding) via `VSHR32` which is a signed right shift. The `0.5 >> prev` bias shifts the quantization region but does NOT round to nearest.

- **Finalise**: Binary decision (error < 0 → bit 0, error ≥ 0 → bit 1). No rounding needed.

### 6.4 Critical Fixed-Point Expressions

**Coarse energy prediction residual:**
```c
f = x - MULT16_32_Q15(coef, oldE) - prev[c];
// x: Q24, coef: Q15, oldE: Q24
// MULT16_32_Q15: (i16 × i32) >> 15 → Q24
// Result: Q24
```

**Fine energy offset:**
```c
offset = SUB32(VSHR32(2*q2+1, extra_quant[i]-DB_SHIFT+1), GCONST(.5f));
offset = SHR32(offset, prev);
// VSHR32: variable shift — shifts left if (extra_quant[i]-DB_SHIFT+1) < 0
// Result: Q24
```

**Finalise offset:**
```c
offset = SHR32(SHL32(q2, DB_SHIFT) - GCONST(.5f), fine_quant[i]+1);
// SHL32(q2, 24): either 0 or 2^24
// GCONST(.5f) in Q24 = 8,388,608
// Difference: -8,388,608 or +8,388,608
// Right-shifted by fine_quant[i]+1
```

---

## 7. Dependencies

### 7.1 What `quant_bands` Calls

| Module | Functions Used | Purpose |
|--------|---------------|---------|
| `laplace` | `ec_laplace_encode`, `ec_laplace_decode` | Coarse energy entropy coding |
| `entenc` | `ec_enc_bit_logp`, `ec_enc_icdf`, `ec_enc_bits`, `ec_encode_bin`, `ec_tell`, `ec_tell_frac`, `ec_range_bytes`, `ec_get_buffer` | Range encoder primitives |
| `entdec` | `ec_dec_bit_logp`, `ec_dec_icdf`, `ec_dec_bits`, `ec_decode_bin`, `ec_dec_update`, `ec_tell` | Range decoder primitives |
| `mathops` | `celt_log2_db` | Log2 computation in `amp2Log2` |
| `modes` | `CELTMode.nbEBands` | Band count for array indexing |
| `rate` | `MAX_FINE_BITS` (= 8) | Cap on fine quantization bits |
| `arch` | All Q-format macros | Fixed-point arithmetic |
| `os_support` | `OPUS_COPY` | Memory copy |
| `stack_alloc` | `VARDECL`, `ALLOC`, `SAVE_STACK`, `RESTORE_STACK` | Stack allocation |

### 7.2 What Calls `quant_bands`

| Caller | Functions Called |
|--------|----------------|
| `celt_encoder.c` | `amp2Log2`, `quant_coarse_energy`, `quant_fine_energy`, `quant_energy_finalise` |
| `celt_decoder.c` | `unquant_coarse_energy`, `unquant_fine_energy`, `unquant_energy_finalise` |
| `bands.c` | References `eMeans[]` (via extern) in `denormalise_bands` |

---

## 8. Constants and Tables

### 8.1 `eMeans[25]` — Mean Band Energies

```c
// Fixed-point: Q4 signed char
const signed char eMeans[25] = {
    103, 100, 92, 85, 81,  77, 72, 70, 78, 75,
     73,  71, 78, 74, 69,  72, 70, 74, 76, 71,
     60,  60, 60, 60, 60
};
// Float: Q4 values / 16.0
const float eMeans[25] = {
    6.4375, 6.25, 5.75, 5.3125, ...
};
```

These represent the average log2-energy for each band, derived from training data. Subtracting them before quantization centers the residual around zero, improving Laplace coding efficiency. The last 5 entries (60 = 3.75 in float) are a default for high bands.

### 8.2 `e_prob_model[4][2][42]` — Laplace Parameters

Indexed as `e_prob_model[LM][intra][2*i]` (probability of 0) and `[2*i+1]` (decay rate), both in Q8.

- Dimension 0: Frame size (LM 0–3 → 120, 240, 480, 960 samples)
- Dimension 1: Coding mode (0=inter, 1=intra)
- Dimension 2: Band index pairs (21 bands × 2 = 42 entries, band index capped at 20)

The values were trained on representative audio data. Inter-frame models have higher P(0) (more energy stays the same), while intra models have lower P(0) but faster decay (larger jumps are common when not predicting).

**Conversion to Laplace codec parameters:**
```c
fs = prob_model[2*min(i,20)] << 7;    // Q8 → Q15 (× 128)
decay = prob_model[2*min(i,20)+1] << 6; // Q8 → Q14 (× 64)
```

### 8.3 `small_energy_icdf[3]`

```c
static const unsigned char small_energy_icdf[3] = {2, 1, 0};
```

ICDF for the 3-symbol fallback when only 2 bits are available. Encodes qi ∈ {-1, 0, 1} as:
```
symbol = 2*qi ^ -(qi<0)   →  qi=-1 → 1,  qi=0 → 0,  qi=1 → 2
```

Decoded via reverse mapping: `qi = (symbol >> 1) ^ -(symbol & 1)`.

### 8.4 Prediction and Smoothing Coefficients

```c
// Q15 fixed-point values
static const opus_val16 pred_coef[4] = {29440, 26112, 21248, 16384};
static const opus_val16 beta_coef[4] = {30147, 22282, 12124, 6554};
static const opus_val16 beta_intra = 4915;
```

- `pred_coef`: AR(1) prediction weight for inter-frame energy prediction. Shorter frames → higher prediction (more correlation). Intra mode uses `coef=0` (no prediction).
- `beta_coef`: Controls how much of the quantized energy feeds back into the inter-band prediction residual `prev[c]`. `prev[c] += q * (1 - beta)`. Shorter frames → higher beta → less inter-band leakage.
- `beta_intra`: 4915/32768 ≈ 0.15 — minimal inter-band coupling in intra mode.

---

## 9. Edge Cases

### 9.1 Budget Exhaustion

When the bit budget runs low, coarse quantization degrades gracefully through four tiers:

| Available bits | Coding method | Value range |
|---------------|---------------|-------------|
| ≥ 15 | Full Laplace coding | Unbounded (tail-clamped) |
| ≥ 2, < 15 | 3-symbol ICDF | {-1, 0, 1} |
| ≥ 1, < 2 | 1-bit logp(1) | {-1, 0} |
| 0 | No coding, assume -1 | {-1} |

The `-1` default when no bits remain causes a gradual energy decrease, which produces a fade-to-silence rather than artifacts.

### 9.2 LFE Mode

When `lfe=1`:
- `max_decay` is set to 3.0 (instead of 16.0) — very slow decay for subwoofer
- For bands ≥ 2: `qi = min(qi, 0)` — energy can only decrease, preventing subwoofer pops
- `quant_coarse_energy_impl` returns 0 badness — no rate-distortion preference

### 9.3 Intra Decision Bypass

If `tell + 3 > budget` (can't even signal the intra flag):
```c
two_pass = intra = 0;
```
Forces inter mode without writing the flag.

### 9.4 Fine Quantization with Insufficient Bits

```c
if (ec_tell(enc) + C * extra_quant[i] > (opus_int32)enc->storage * 8)
    continue;
```
Silently skips bands where encoding the fine bits would exceed storage. The decoder performs the same check and skips in sync.

### 9.5 NULL `oldEBands` in Finalise

```c
if (oldEBands != NULL) oldEBands[i+c*m->nbEBands] += offset;
```

Both encoder and decoder finalise functions guard against `NULL` `oldEBands`. This allows the caller to skip energy reconstruction when only error tracking matters.

### 9.6 `prev_quant == NULL`

Both `quant_fine_energy` and `unquant_fine_energy` check:
```c
prev = (prev_quant != NULL) ? prev_quant[i] : 0;
```
When NULL, acts as if no previous fine quantization was done (zero shift).

### 9.7 Decoder `prev[c]` Type

The decoder uses `opus_val64` for `prev[c]` vs `opus_val32` in the encoder. This prevents 32-bit accumulation drift on the decoder side where exact reconstruction is critical.

---

## 10. Porting Notes

### 10.1 In-Place Mutation of `*value` in `ec_laplace_encode`

```c
void ec_laplace_encode(ec_enc *enc, int *value, unsigned fs, int decay);
```

The `value` parameter is `int *` because the encoder may clamp it. In Rust, this should be `&mut i32`. The caller (coarse energy impl) passes `&qi` and uses the potentially-modified `qi` afterward. This is a subtle semantic contract that must be preserved.

### 10.2 `do { ... } while (++c < C)` Pattern

Every per-channel loop in the module uses this C idiom:
```c
c=0;
do {
    // ... work with c ...
} while (++c < C);
```

This always executes at least once (C is always 1 or 2). In Rust, use `for c in 0..channels` — the zero-iteration case never occurs but Rust's for loop handles it safely.

### 10.3 Two's Complement Sign Manipulation

```c
s = -(val < 0);        // s = 0 for positive, -1 for negative
val = (val + s) ^ s;   // absolute value without branch
fl += fs & ~s;         // conditional add: adds fs if positive (s=0), 0 if negative (s=-1)
```

And the ICDF sign encoding:
```c
// Encode: symbol = 2*qi ^ -(qi<0)
// Decode: qi = (symbol >> 1) ^ -(symbol & 1)
```

These rely on two's complement representation. In Rust, signed integer arithmetic is two's complement by specification, but the bitwise operations on mixed signed/unsigned types need care. The `^` with `-1` is a bitwise NOT. Consider using explicit `abs()`, `signum()`, and conditional expressions for clarity.

### 10.4 Encoder State Copy and Restore

```c
ec_enc enc_start_state = *enc;
// ... trial encoding ...
*enc = enc_start_state;
```

The two-pass coarse energy encoder saves the entire `ec_enc` state by value, runs a trial encoding, then restores it. The `ec_enc` struct must implement `Clone` in Rust. This also requires saving and restoring the actual bitstream bytes:

```c
intra_buf = ec_get_buffer(&enc_intra_state) + nstart_bytes;
save_bytes = nintra_bytes - nstart_bytes;
OPUS_COPY(intra_bits, intra_buf, save_bytes);
// ... inter encoding ...
OPUS_COPY(intra_buf, intra_bits, save_bytes); // restore if intra wins
```

The buffer pointer manipulation (`ec_get_buffer` + byte offset) needs to become slice operations in Rust.

### 10.5 Stack Allocation via `ALLOC` Macro

```c
VARDECL(celt_glog, oldEBands_intra);
ALLOC(oldEBands_intra, C * m->nbEBands, celt_glog);
```

Maps to `alloca` or VLA semantics. In Rust, use `Vec<i32>` or a fixed-size array. Since `C * nbEBands ≤ 2 * 25 = 50`, a stack array `[i32; 50]` is practical. Alternatively, use `SmallVec` or pass a scratch buffer.

### 10.6 `ALLOC_NONE` Edge Case

```c
if (save_bytes == 0)
    save_bytes = ALLOC_NONE;  // platform-dependent: 0 or 1
ALLOC(intra_bits, save_bytes, unsigned char);
```

When `save_bytes` is 0, C's VLA or alloca with size 0 is undefined behavior. The code sets it to `ALLOC_NONE` (1 on VLA platforms, 0 on custom allocator). In Rust, a zero-length Vec is fine — no special case needed.

### 10.7 `#ifdef FIXED_POINT` Conditional Compilation

The module has pervasive `#ifdef FIXED_POINT` branches. The Rust port should use a generic numeric type or separate modules. Key differences:

| Operation | Fixed-Point | Float |
|-----------|------------|-------|
| Rounding `qi` | `(f + half) >> DB_SHIFT` | `floor(f + 0.5)` |
| Prediction | `MULT16_32_Q15(coef, oldE)` | `coef * oldE` |
| Max decay limit | `SHL32(MIN32(SHR32(...),...),...)`  | `MIN32(max_decay, 0.125*nbAvailableBytes)` |
| Energy clamp | `MAX32(-GCONST(28), tmp)` | None |
| Fine offset | Bit-shift chains | Float multiply chains |

For the initial port, implement **fixed-point only** since bit-exactness against the C reference is the hard requirement.

### 10.8 Integer Types in Laplace Codec

The Laplace codec mixes `unsigned`, `int`, `opus_int32`, and `opus_uint16`. The expression:
```c
return ft * (opus_int32)(16384 - decay) >> 15;
```
Multiplies `unsigned` by `opus_int32`. In C, the unsigned is converted to signed for the multiplication. The result type and right-shift behavior depend on this. In Rust, explicit casts are needed: `(ft as i32) * (16384 - decay) >> 15` or similar.

### 10.9 Assertion Discipline

```c
celt_assert(fl + fs <= 32768);
celt_assert(fs > 0);
celt_assert(fl <= fm);
celt_assert(fm < IMIN(fl + fs, 32768));
```

These are critical invariants for the range coder. In Rust, use `debug_assert!` to match the C behavior (assertions active in debug builds, removed in release).

### 10.10 The `prev[c]` Encoder/Decoder Asymmetry

```c
// Encoder:
opus_val32 prev[2] = {0, 0};     // 32-bit

// Decoder:
opus_val64 prev[2] = {0, 0};     // 64-bit in fixed-point
```

The decoder uses 64-bit to prevent accumulation errors. In Rust fixed-point, use `i64` for the decoder's `prev` and `i32` for the encoder's. Both must produce identical quantized results.

### 10.11 Macro-Generated ICDF Encoding

The sign encoding for small_energy uses a clever XOR:
```c
ec_enc_icdf(enc, 2*qi^-(qi<0), small_energy_icdf, 2);
```

| qi | `2*qi` | `-(qi<0)` | `2*qi ^ -(qi<0)` | symbol |
|----|--------|-----------|-------------------|--------|
| -1 | -2 | -1 (0xFF..FF) | 1 | 1 |
| 0 | 0 | 0 | 0 | 0 |
| 1 | 2 | 0 | 2 | 2 |

The decoder reverses: `qi = (symbol >> 1) ^ -(symbol & 1)`:

| symbol | `>>1` | `&1` | `-(...)` | `^` | qi |
|--------|-------|------|----------|-----|----|
| 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 1 | -1 | -1 | -1 |
| 2 | 1 | 0 | 0 | 1 | 1 |

### 10.12 `ec_enc_icdf16` vs `ec_enc_icdf`

The `_p0` Laplace variant uses `ec_enc_icdf16` / `ec_dec_icdf16` (16-bit ICDF tables), while the main quant_bands code uses `ec_enc_icdf` (8-bit ICDF tables). Ensure both variants are available in the range coder port.

### 10.13 Module Boundary for Rust

Suggested Rust module structure:
```
celt/
  quant_bands.rs      // All quant/unquant functions, amp2Log2, tables
  laplace.rs          // ec_laplace_{encode,decode}[_p0], ec_laplace_get_freq1
```

The Laplace codec is logically independent and only coupled through the range coder types (`ec_enc`/`ec_dec`). Keep it as a separate module.
