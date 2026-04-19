Now I have all the information needed. Here is the complete architecture document:

---

# CELT Decoder Module — Architecture Documentation

**Source file:** `reference/celt/celt_decoder.c`
**Module:** CELT Decoder (the frequency-domain codec layer of Opus)

---

## 1. Purpose

The CELT decoder is the frequency-domain audio decoder within the Opus codec. It receives entropy-coded bitstreams and produces PCM audio output. In the Opus pipeline, it handles:

- **CELT-only mode**: Full-bandwidth audio decoding (typically music)
- **Hybrid mode**: Decoding the high-frequency bands (above ~8 kHz) when combined with SILK for the low bands
- **Packet Loss Concealment (PLC)**: Generating plausible audio when packets are lost, using three strategies: periodic (pitch-based), noise-based, and neural (LPCNet)

The decoder operates on 2.5/5/10/20 ms frames at 48 kHz (internally), with optional downsampling to 8/12/16/24 kHz for output. It supports mono and stereo.

---

## 2. Public API

### 2.1 Size Query

```c
int celt_decoder_get_size(int channels);
```
- **Parameters:** `channels` — 1 (mono) or 2 (stereo)
- **Returns:** Total byte count needed for decoder allocation (struct + trailing buffers)
- **Notes:** Internally calls `opus_custom_decoder_get_size()` with the default 48 kHz/960-sample mode

```c
OPUS_CUSTOM_NOSTATIC int opus_custom_decoder_get_size(const CELTMode *mode, int channels);
```
- **Parameters:** `mode` — codec mode (sample rate, frame size, band structure); `channels` — 1 or 2
- **Returns:** `sizeof(CELTDecoder) + decode_mem + 4×2×nbEBands (energy arrays) + channels×CELT_LPC_ORDER (LPC coefficients)`
- **Notes:** `OPUS_CUSTOM_NOSTATIC` is `static inline` unless custom modes are enabled

### 2.2 Initialization

```c
int celt_decoder_init(CELTDecoder *st, opus_int32 sampling_rate, int channels);
```
- **Parameters:** `st` — pre-allocated decoder; `sampling_rate` — output rate (8000/12000/16000/24000/48000); `channels` — 1 or 2
- **Returns:** `OPUS_OK` on success, `OPUS_BAD_ARG` if rate is unsupported
- **Notes:** Creates default mode via `opus_custom_mode_create(48000, 960, NULL)`, then sets `downsample` factor. Always decodes internally at 48 kHz; downsampling happens at output.

```c
OPUS_CUSTOM_NOSTATIC int opus_custom_decoder_init(CELTDecoder *st, const CELTMode *mode, int channels);
```
- **Parameters:** `st` — decoder memory; `mode` — codec mode; `channels` — 1 or 2
- **Returns:** `OPUS_OK`, `OPUS_BAD_ARG` (channels out of range), or `OPUS_ALLOC_FAIL`
- **Behavior:** Zeros entire struct, sets `mode`, `overlap`, `channels`, `stream_channels`, `downsample=1`, `start=0`, `end=effEBands`, `signalling=1`, `disable_inv` (mono only unless `DISABLE_UPDATE_DRAFT`), `arch`, then calls `OPUS_RESET_STATE`

### 2.3 Decoding

```c
int celt_decode_with_ec(CELTDecoder *st, const unsigned char *data,
      int len, opus_res *pcm, int frame_size, ec_dec *dec, int accum);
```
- **Parameters:**
  - `st` — initialized decoder state
  - `data` — compressed bitstream bytes (NULL triggers PLC)
  - `len` — byte length of `data` (0 or ≤1 also triggers PLC); max 1275
  - `pcm` — output PCM buffer, interleaved for stereo
  - `frame_size` — number of samples per channel to decode (must match a valid LM)
  - `dec` — optional pre-initialized entropy decoder (NULL → create from `data`)
  - `accum` — if nonzero, *add* to `pcm` instead of overwriting (used in hybrid mode)
- **Returns:** Number of decoded samples per channel, or negative error code
- **Notes:** This is a thin wrapper around `celt_decode_with_ec_dred()` with `lpcnet=NULL` and no QEXT payload.

```c
int celt_decode_with_ec_dred(CELTDecoder *st, const unsigned char *data,
      int len, opus_res *pcm, int frame_size, ec_dec *dec, int accum,
      LPCNetPLCState *lpcnet,                    // ifdef ENABLE_DEEP_PLC
      const unsigned char *qext_payload, int qext_payload_len);  // ifdef ENABLE_QEXT
```
- The full-featured decode entry point. Additional parameters:
  - `lpcnet` — neural PLC state (may be NULL)
  - `qext_payload` / `qext_payload_len` — extended quality payload for higher fidelity

### 2.4 CTL (Control) Interface

```c
int opus_custom_decoder_ctl(CELTDecoder *st, int request, ...);
```
Variadic control interface. Supported requests:

| Request | Direction | Type | Notes |
|---|---|---|---|
| `OPUS_SET_COMPLEXITY_REQUEST` | Set | `opus_int32` (0–10) | Controls PLC quality |
| `OPUS_GET_COMPLEXITY_REQUEST` | Get | `opus_int32*` | |
| `CELT_SET_START_BAND_REQUEST` | Set | `opus_int32` | 0 (full) or 17 (hybrid high-band) |
| `CELT_SET_END_BAND_REQUEST` | Set | `opus_int32` | 1..nbEBands |
| `CELT_SET_CHANNELS_REQUEST` | Set | `opus_int32` | stream_channels (1 or 2) |
| `CELT_GET_AND_CLEAR_ERROR_REQUEST` | Get | `opus_int32*` | Reads and clears error flag |
| `OPUS_GET_LOOKAHEAD_REQUEST` | Get | `opus_int32*` | `overlap / downsample` |
| `OPUS_RESET_STATE` | Action | — | Clears all state from `rng` onward; sets oldLogE to −28 dB |
| `OPUS_GET_PITCH_REQUEST` | Get | `opus_int32*` | Last postfilter pitch period |
| `CELT_GET_MODE_REQUEST` | Get | `const CELTMode**` | |
| `CELT_SET_SIGNALLING_REQUEST` | Set | `opus_int32` | Custom mode signalling flag |
| `OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST` | Set | `opus_int32` | 0 or 1 |
| `OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST` | Get | `opus_int32*` | |

---

## 3. Internal State (`struct OpusCustomDecoder`)

The struct uses a C "flexible array member" pattern: `_decode_mem[1]` is the anchor for a variable-length trailing buffer.

### 3.1 Configuration Fields (persist across reset)

| Field | Type | Description |
|---|---|---|
| `mode` | `const OpusCustomMode*` | Band structure, MDCT config, window, preemph coefficients |
| `overlap` | `int` | MDCT overlap size (120 samples at 48 kHz) |
| `channels` | `int` | Output channels (1 or 2) |
| `stream_channels` | `int` | Channels encoded in the stream (may differ from output) |
| `downsample` | `int` | Output downsampling factor: 1/2/3/4/6 for 48k/24k/16k/12k/8k |
| `start` | `int` | First band to decode: 0 (full) or 17 (hybrid high-band only) |
| `end` | `int` | Last band (exclusive) to decode; up to `mode->effEBands` |
| `signalling` | `int` | Whether first byte contains signalling info (custom modes) |
| `disable_inv` | `int` | Disable phase inversion (stereo anti-phase) |
| `complexity` | `int` | 0–10, affects PLC strategy selection |
| `arch` | `int` | CPU architecture flags for SIMD dispatch |

### 3.2 Dynamic State Fields (cleared on reset at `DECODER_RESET_START`)

| Field | Type | Description |
|---|---|---|
| `rng` | `opus_uint32` | Range coder RNG state (for validation / anti-collapse) |
| `error` | `int` | Sticky error flag from range coder |
| `last_pitch_index` | `int` | Pitch lag from last PLC search (100–720 or 0) |
| `loss_duration` | `int` | Cumulative lost frames in sub-frame units, capped at 10000 |
| `plc_duration` | `int` | Continuous PLC duration (resets on good packet or DRED) |
| `last_frame_type` | `int` | FRAME_NONE/NORMAL/PLC_NOISE/PLC_PERIODIC/PLC_NEURAL/DRED |
| `skip_plc` | `int` | If set, force noise PLC (skip pitch-based); set after noise PLC |
| `postfilter_period` | `int` | Current comb filter pitch period |
| `postfilter_period_old` | `int` | Previous frame's comb filter pitch period |
| `postfilter_gain` | `opus_val16` | Current comb filter gain (Q15 fixed-point) |
| `postfilter_gain_old` | `opus_val16` | Previous frame's comb filter gain |
| `postfilter_tapset` | `int` | Current comb filter tap set (0, 1, or 2) |
| `postfilter_tapset_old` | `int` | Previous frame's tap set |
| `prefilter_and_fold` | `int` | Flag: prefilter overlap needs applying before next frame |
| `preemph_memD[2]` | `celt_sig` | De-emphasis IIR filter memory, one per channel |

### 3.3 Trailing Variable-Length Arrays (accessed via pointer arithmetic)

Starting at `_decode_mem[0]`:

| Array | Size | Description |
|---|---|---|
| `decode_mem` | `channels × (DECODE_BUFFER_SIZE + overlap)` × `sizeof(celt_sig)` | Circular decode buffer + overlap region per channel |
| `oldBandE` | `2 × nbEBands` × `sizeof(celt_glog)` | Current band energies (log domain) |
| `oldLogE` | `2 × nbEBands` × `sizeof(celt_glog)` | Previous frame's band energies |
| `oldLogE2` | `2 × nbEBands` × `sizeof(celt_glog)` | Two frames ago band energies |
| `backgroundLogE` | `2 × nbEBands` × `sizeof(celt_glog)` | Noise floor estimate per band |
| `lpc` | `channels × CELT_LPC_ORDER` × `sizeof(opus_val16)` | LPC coefficients for PLC |

All energy arrays are doubled (2×nbEBands) to hold both channels even for mono (mono copies ch0 to ch1 slot).

### 3.4 Lifecycle

1. **Allocate** `celt_decoder_get_size(channels)` bytes
2. **Init** via `celt_decoder_init()` → zeros everything, sets config, resets energy to −28 dB
3. **Decode** repeatedly via `celt_decode_with_ec()` — each call decodes one frame
4. **Reset** via `OPUS_RESET_STATE` ctl — clears from `rng` onward, preserves config
5. **Destroy** via `opus_free()` (custom modes) or stack unwinding

---

## 4. Algorithm — Core Decode Path (`celt_decode_with_ec_dred`)

### 4.1 Frame Validation and Setup (lines 1100–1270)

1. Validate CELT decoder state (`VALIDATE_CELT_DECODER`)
2. Extract mode parameters: `nbEBands`, `overlap`, `eBands`, `start`, `end`
3. Scale `frame_size` by `downsample` factor (internal decode always at native rate)
4. Compute pointers into trailing arrays (`oldBandE`, `oldLogE`, `oldLogE2`, `backgroundLogE`)
5. **Custom mode signalling** (if `signalling` is set): parse first byte for `end`, `LM`, `C`, padding
6. **LM determination**: Find log2 of frame-size-to-short-MDCT ratio. LM ∈ {0,1,2,3} corresponding to 2.5/5/10/20 ms. `M = 1 << LM`
7. Compute `N = M * mode->shortMdctSize` (total MDCT bins per channel)
8. Set up `decode_mem[]` and `out_syn[]` pointers per channel

### 4.2 Packet Loss Detection (lines 1277–1293)

If `data == NULL` or `len <= 1`:
- Call `celt_decode_lost()` for PLC
- Apply de-emphasis
- Return

### 4.3 Silence Detection (lines 1314–1325)

- If all bits consumed → silence
- If at bit position 1 → decode silence flag with 15-bit logp
- On silence: pretend all bits have been read

### 4.4 Postfilter Parameters (lines 1327–1343)

If `start == 0` and enough bits remain (16):
- Decode boolean: postfilter present?
- If yes: decode `octave` (uint, 6 values), compute `pitch = (16<<octave) + bits(4+octave) - 1`
- Decode `qg` (3 bits), compute `gain = 0.09375 × (qg+1)` (Q15)
- Decode `tapset` (icdf, 3 values: 0/1/2)

### 4.5 Transient Flag (lines 1345–1356)

If `LM > 0` and enough bits: decode `isTransient` with logp=3. Transient frames use short MDCT blocks (`shortBlocks = M`).

### 4.6 Energy Decoding (lines 1358–1393)

1. Decode `intra_ener` flag (logp=3) — whether energies are coded independently
2. **Loss recovery safety**: If not intra and recovering from loss, clamp `oldBandE` using energy trend analysis (slope-based decay or min of last 2 frames) with per-LM safety margins
3. Call `unquant_coarse_energy()` — Laplace-coded coarse energy per band
4. (Fine energy comes later, after bit allocation)

### 4.7 Temporal Frequency Resolution (lines 1395–1549)

1. Call `tf_decode()` — decode time-frequency resolution flags per band using `tf_select_table[LM][...]`

### 4.8 Spread Decision (lines 1399–1401)

Decode spread mode from `spread_icdf`: NONE(0), LIGHT(1), NORMAL(2), AGGRESSIVE(3).

### 4.9 Dynamic Bit Allocation (lines 1403–1453)

1. `init_caps()` — compute per-band bit caps
2. Per-band dynamic boost loop: decode boost flags with decreasing logp
3. Decode `alloc_trim` from `trim_icdf` (11 values)
4. Reserve bits for anti-collapse flag if transient with LM≥2
5. Call `clt_compute_allocation()` — compute `pulses[]`, `fine_quant[]`, `fine_priority[]`, `intensity`, `dual_stereo`, `balance`

### 4.10 Fine Energy + Coefficient Decoding (lines 1455–1524)

1. `unquant_fine_energy()` — decode fine energy refinement bits
2. Shift decode memory left by N samples: `OPUS_MOVE(decode_mem, decode_mem+N, ...)`
3. `quant_all_bands()` — decode PVQ (pyramid vector quantization) shape coefficients into `X[]`, producing normalized MDCT coefficients
4. QEXT extension handling (if enabled): additional energy + PVQ decoding
5. Decode anti-collapse bit; apply `anti_collapse()` if set
6. `unquant_energy_finalise()` — use remaining bits for energy refinement

### 4.11 Synthesis (lines 1531–1560)

1. **Prefilter and fold** (if `prefilter_and_fold` flag set from previous PLC frame)
2. `celt_synthesis()`:
   - `denormalise_bands()` — scale normalized coefficients by band energies
   - `clt_mdct_backward()` — inverse MDCT, producing time-domain signal with overlap-add
   - Saturate output to `SIG_SAT`
3. **Comb filter (postfilter)**: applied in two segments:
   - First `shortMdctSize` samples: crossfade from old to new postfilter params
   - Remaining `N - shortMdctSize` samples (if LM>0): crossfade from current to new params

### 4.12 State Update (lines 1549–1612)

1. Update postfilter state (old ← current, current ← decoded)
2. For mono: copy band energies to second channel slot
3. Update energy history: `oldLogE2 ← oldLogE ← oldBandE` (non-transient) or `oldLogE = min(oldLogE, oldBandE)` (transient)
4. Update `backgroundLogE` with max increase of 2.4 dB/second
5. Zero unused bands (before `start`, after `end`)
6. Copy `dec->rng` to `st->rng`
7. Apply de-emphasis filter to `out_syn[]` → `pcm[]`
8. Reset loss counters, set `last_frame_type = FRAME_NORMAL`
9. Verify bitstream consistency: `ec_tell(dec) <= 8*len`

---

## 5. Data Flow

### 5.1 Input

```
compressed bytes  →  ec_dec (range decoder)
                      ↓
                 [silence flag]
                 [postfilter params]
                 [transient flag]
                 [coarse energies]
                 [tf resolution flags]
                 [spread mode]
                 [dynamic allocation boosts]
                 [alloc trim]
                 [fine energies]
                 [PVQ coefficients]
                 [anti-collapse flag]
                 [energy finalization bits]
```

### 5.2 Internal Buffers

```
oldBandE[2*nbEBands]  ← band energies (log domain, Q-format celt_glog)

X[C*N]                ← normalized MDCT coefficients (celt_norm)
                         Layout: [ch0: N coefficients][ch1: N coefficients]
                         Within each channel: bands are contiguous,
                         band i spans eBands[i]<<LM .. eBands[i+1]<<LM

decode_mem[c]         ← per-channel circular buffer
                         Size: DECODE_BUFFER_SIZE + overlap
                         Layout: [......history......][current frame N][overlap]
                         out_syn[c] points to decode_mem[c] + DECODE_BUFFER_SIZE - N
```

### 5.3 Output

```
pcm[frame_size/downsample * CC]  — interleaved PCM
   Type: opus_res (float in float builds, int16 or int32 in fixed-point)
   Interleaving: [L0, R0, L1, R1, ...] for stereo
```

### 5.4 Channel Handling in `celt_synthesis()`

Three cases:
- **CC==2, C==1** (mono stream → stereo output): decode once, IMDCT to both channels
- **CC==1, C==2** (stereo stream → mono output): decode both, average, IMDCT once
- **CC==C** (normal): per-channel denormalize + IMDCT

---

## 6. Numerical Details

### 6.1 Type System

| Type | Float build | Fixed-point build | Description |
|---|---|---|---|
| `opus_val16` | `float` | `int16_t` | Narrow values (gains, LPC, window) |
| `opus_val32` | `float` | `int32_t` | Wide values (accumulators) |
| `celt_sig` | `float` | `int32_t` | Internal signal representation |
| `celt_norm` | `float` | `int32_t` | Normalized MDCT coefficients |
| `celt_glog` | `float` | `int32_t` | Log-domain energy (Q24 in fixed-point, raw float) |
| `opus_res` | `float` | `int16_t` (or `int32_t` w/ RES24) | Output resolution type |
| `celt_coef` | `float` | `int16_t` (or `int32_t` w/ QEXT) | Window/filter coefficients |

### 6.2 Q-Formats (Fixed-Point)

- **SIG_SHIFT = 12**: `celt_sig` = Q12 (int16 sample shifted left by 12). Range: ±SIG_SAT (536870911 ≈ 2^29−1)
- **NORM_SHIFT = 24**: `celt_norm` = Q24. `NORM_SCALING = 1<<24`
- **DB_SHIFT = 24**: `celt_glog` (log energies) are Q24 fixed-point
- **Q15**: gains, window coefficients, LPC coefficients → Q15 (divide by 32767)
- **BITRES = 3**: Bit allocation uses 1/8th-bit precision (3 fractional bits)
- **RES_SHIFT = 0** (16-bit) or **8** (24-bit): `opus_res` scaling

### 6.3 Signal Scaling Pipeline

```
PCM int16 (±32768) 
  → SIG domain: SHL(sample, SIG_SHIFT=12) → Q12 (±2^27)
  → Preemphasis in SIG domain
  → MDCT → frequency domain (celt_norm, Q24)
  → Band normalization
  → [encoding/decoding]
  → Denormalization (scale by band energy)
  → IMDCT → SIG domain
  → De-emphasis
  → SIG2RES: SIG_SHIFT right-shift → output
```

### 6.4 Overflow Guards

- **SIG_SAT = 536870911 (2^29−1)**: All IMDCT output is saturated to this value (line 507–509). Required because ARM comb filter shifts left by 1 and adds two values.
- **De-emphasis saturation**: `SATURATE(x[j] + VERY_SMALL + m, SIG_SAT)` on every sample
- **LPC bandwidth expansion** (fixed-point PLC only): iteratively multiply LPC coefficients by 0.99^i until `sum(|coefs|) < 65535`, preventing IIR overflow (lines 882–893)
- **Excitation energy check**: After LPC synthesis in PLC, compare pre/post energy. If synthesis energy exceeds 5× (fixed) or 5× (float) the excitation energy → zero the output. Otherwise scale down smoothly using the ratio (lines 989–1015).

### 6.5 De-emphasis Filter

First-order IIR: `y[n] = x[n] + coef0 * y[n-1]`

For the standard mode, `coef0 = mode->preemph[0]`. The de-emphasis has a special fast path for stereo without downsampling (`deemphasis_stereo_simple`).

For custom modes with `coef[1] != 0`, a more complex 2nd-order filter is used:
```
tmp = x[j] + m
m = coef0*tmp - coef1*x[j]
out = coef3*tmp << 2
```

### 6.6 Downsampling

When `downsample > 1`, de-emphasis writes to a scratch buffer, then every `downsample`-th sample is picked: `y[j*C] = scratch[j*downsample]`. This is a simple decimation (the preemphasis filter provides anti-aliasing).

---

## 7. Dependencies

### 7.1 Modules Called by celt_decoder

| Module | Functions Used | Purpose |
|---|---|---|
| `entdec` | `ec_dec_init`, `ec_dec_bit_logp`, `ec_dec_bits`, `ec_dec_uint`, `ec_dec_icdf`, `ec_tell`, `ec_tell_frac`, `ec_get_error` | Range decoder |
| `quant_bands` | `unquant_coarse_energy`, `unquant_fine_energy`, `unquant_energy_finalise` | Band energy decoding |
| `bands` | `quant_all_bands` (decode mode), `denormalise_bands`, `anti_collapse`, `celt_lcg_rand`, `renormalise_vector` | Coefficient decoding, denormalization |
| `rate` | `clt_compute_allocation`, `init_caps` | Bit allocation |
| `mdct` | `clt_mdct_backward` | Inverse MDCT |
| `pitch` | `pitch_downsample`, `pitch_search` | PLC pitch detection |
| `celt_lpc` | `_celt_autocorr`, `_celt_lpc`, `celt_fir`, `celt_iir` | PLC LPC analysis/synthesis |
| `mathops` | `celt_sqrt`, `frac_div32`, `celt_zlog2`, `celt_maxabs16` | Fixed-point math |
| `modes` | `opus_custom_mode_create` | Mode construction |
| `vq` | (linked via `quant_all_bands`) | Vector quantization |
| `lpcnet` | `lpcnet_plc_update`, `lpcnet_plc_conceal` | Neural PLC (optional) |

### 7.2 What Calls This Module

- `opus_decoder.c` → `celt_decode_with_ec()` / `celt_decode_with_ec_dred()`
- `opus_decoder.c` → `celt_decoder_init()`, `celt_decoder_get_size()`
- `opus_decoder.c` → `celt_decoder_ctl()` (via macro alias)

---

## 8. Constants and Tables

### 8.1 Key Constants

| Constant | Value | Source | Meaning |
|---|---|---|---|
| `DECODE_BUFFER_SIZE` | 2048 (`DEC_PITCH_BUF_SIZE`) | `modes.h:42` | Decode memory per channel in samples |
| `MAX_PERIOD` | 1024 | `modes.h:40` | Maximum pitch period for comb filter |
| `PLC_PITCH_LAG_MAX` | 720 | Local | Max pitch lag for PLC (66.67 Hz) |
| `PLC_PITCH_LAG_MIN` | 100 | Local | Min pitch lag for PLC (480 Hz) |
| `COMBFILTER_MAXPERIOD` | 1024 | `celt.h:236` | Comb filter max period |
| `COMBFILTER_MINPERIOD` | 15 | `celt.h:237` | Comb filter min period |
| `CELT_LPC_ORDER` | 24 | `celt_lpc.h:38` | LPC order for PLC |
| `BITRES` | 3 | `entcode.h:56` | Fractional bit precision (1/8 bit) |
| `SIG_SHIFT` | 12 | `arch.h:207` | Signal Q-format shift |
| `SIG_SAT` | 536870911 | `arch.h:215` | Signal saturation limit (2^29−1) |
| `NORM_SHIFT` | 24 | `arch.h:183` | Normalized coefficient Q-format |
| `DB_SHIFT` | 24 | `arch.h:219` | Log-energy Q-format |
| `PLC_UPDATE_FRAMES` | 4 | Local | Frames per neural PLC update |
| `PLC_UPDATE_SAMPLES` | `4 * FRAME_SIZE` | Local | Samples per neural PLC update (640 at 16 kHz) |
| `SINC_ORDER` | 48 | Local | Sinc resampling filter order (16k↔48k) |
| `FRAME_SIZE` | 160 | `freq.h` | LPCNet frame size (10 ms at 16 kHz) |
| `PREEMPHASIS` | 0.85 | `freq.h` | Neural PLC preemphasis coefficient |

### 8.2 Static Tables

| Table | Size | Source | Purpose |
|---|---|---|---|
| `trim_icdf[11]` | 11 bytes | `celt.h:194` | iCDF for alloc_trim (0–10) |
| `spread_icdf[4]` | 4 bytes | `celt.h:196` | iCDF for spread decision |
| `tapset_icdf[3]` | 3 bytes | `celt.h:198` | iCDF for postfilter tapset |
| `tf_select_table[4][8]` | 32 bytes | `celt.c:321–326` | Time-frequency resolution mapping |
| `sinc_filter[49]` | 49 floats | Local (line 629) | 3:1 sinc resampling kernel for neural PLC |

### 8.3 Frame Type Constants

```c
#define FRAME_NONE         0   // No previous frame (after reset)
#define FRAME_NORMAL       1   // Successfully decoded frame
#define FRAME_PLC_NOISE    2   // Noise-based concealment
#define FRAME_PLC_PERIODIC 3   // Pitch-based concealment
#define FRAME_PLC_NEURAL   4   // LPCNet neural concealment
#define FRAME_DRED         5   // Deep redundancy decoding
```

---

## 9. Packet Loss Concealment (`celt_decode_lost`)

### 9.1 PLC Strategy Selection (lines 723–736)

Priority chain:
1. **FRAME_PLC_NOISE** if: `plc_duration >= 40` OR `start != 0` (hybrid) OR `skip_plc` flag
2. **FRAME_PLC_NEURAL** if: LPCNet available, loaded, `complexity >= 5`, `plc_duration < 80`, and `!skip_plc`
3. **FRAME_DRED** if: neural available AND FEC data in buffer
4. **FRAME_PLC_PERIODIC** (default): pitch-based extrapolation

### 9.2 Noise PLC (lines 738–807)

1. Shift decode memory left by N
2. Apply prefilter_and_fold if pending
3. Decay band energies: `oldBandE[i] = max(backgroundLogE[i], oldBandE[i] - decay)` where decay = 1.5 dB (first loss) or 0.5 dB (subsequent)
4. Fill `X[]` with LCG pseudo-random values, renormalize each band
5. Synthesize via `celt_synthesis()`
6. Apply comb postfilter with last known parameters
7. Set `skip_plc = 1` (require 2 good packets before returning to pitch PLC)

### 9.3 Pitch PLC (lines 808–1077)

Per channel:
1. **Pitch search** (if not continuing from previous PLC): `celt_plc_pitch_search()` → `pitch_downsample()` + `pitch_search()`
2. **LPC analysis**: autocorrelation of last `MAX_PERIOD` samples, noise floor (−40 dB), lag windowing, Levinson-Durbin. Fixed-point bandwidth expansion ensures `sum(|coefs|) < 65535`.
3. **Excitation extraction**: `celt_fir()` on recent samples with LPC filter
4. **Decay estimation**: compare energy of two consecutive pitch periods; `decay = sqrt(E1/E2)`
5. **Extrapolation**: copy excitation periodically with `pitch_index` period, scaling by `attenuation *= decay` each period
6. **LPC synthesis**: `celt_iir()` converts excitation back to signal domain
7. **Energy guard**: if synthesis energy > pre-loss energy, attenuate (window-smoothed for overlap region, flat for remainder)
8. Set `prefilter_and_fold = 1` for next frame

### 9.4 Neural PLC (lines 1020–1074, `#ifdef ENABLE_DEEP_PLC`)

1. If transitioning to neural: call `update_plc_state()` to feed LPCNet with recent decoded audio (downsampled 48k→16k via sinc filter)
2. Generate 16 kHz concealment frames via `lpcnet_plc_conceal()`
3. Upsample 16k→48k via polyphase sinc interpolation
4. Remove preemphasis
5. For stereo: duplicate mono PLC to both channels
6. Cross-fade with pitch PLC output over overlap region for smooth transition

---

## 10. Edge Cases and Error Conditions

### 10.1 Error Returns

| Condition | Return Code |
|---|---|
| `channels < 0 || channels > 2` | `OPUS_BAD_ARG` |
| `st == NULL` | `OPUS_ALLOC_FAIL` |
| Invalid sampling rate | `OPUS_BAD_ARG` |
| `len < 0 || len > 1275 || pcm == NULL` | `OPUS_BAD_ARG` |
| `LM > mode->maxLM` | `OPUS_BAD_ARG` or `OPUS_INVALID_PACKET` |
| `frame_size < shortMdctSize << LM` | `OPUS_BUFFER_TOO_SMALL` |
| `ec_tell(dec) > 8*len` after decode | `OPUS_INTERNAL_ERROR` |
| Range coder error (`ec_get_error`) | Sets `st->error = 1` |

### 10.2 Special Input Handling

- **data==NULL or len≤1**: Triggers PLC (not an error)
- **Silence frame**: Detected via silence bit; sets all band energies to −28 dB, but still runs full synthesis pipeline with those energies
- **Mono stream to stereo output (C==1, CC==2)**: `celt_synthesis` copies to both channels
- **Stereo stream to mono output (C==2, CC==1)**: averages before IMDCT
- **Hybrid mode (start==17)**: Skips postfilter decode (no low-band data); forces noise PLC on loss
- **Loss recovery**: Energy prediction safety clamps prevent loud artifacts on first good frame after loss

### 10.3 State Consistency

- `loss_duration` capped at 10000 to prevent overflow
- `plc_duration` capped at 10000
- Postfilter period clamped to `[COMBFILTER_MINPERIOD, MAX_PERIOD)`
- Background noise estimate limited to max +2.4 dB/second increase
- Unused bands (before `start`, after `end`) zeroed in energy arrays

---

## 11. Porting Notes for Rust

### 11.1 Variable-Length Trailing Struct (`_decode_mem[1]`)

The C struct uses a classic "struct hack": `_decode_mem[1]` is a flexible array member, and the allocation size is computed by `opus_custom_decoder_get_size()`. In Rust, this must be modeled as either:
- A `Vec<celt_sig>` member with manual offset arithmetic
- A single `Box<[u8]>` with typed accessors and pointer arithmetic
- Separate `Vec` fields for each logical array (cleanest but changes layout)

The pointer arithmetic to reach `oldBandE`, `oldLogE`, etc. is:
```c
oldBandE    = _decode_mem + (decode_buffer_size + overlap) * CC
oldLogE     = oldBandE + 2 * nbEBands
oldLogE2    = oldLogE + 2 * nbEBands
backgroundLogE = oldLogE2 + 2 * nbEBands
lpc         = (opus_val16*)(backgroundLogE + 2 * nbEBands)
```
**Recommendation:** Use a flat `Vec<u8>` with accessor methods that return typed slices, or break into separate `Vec`s if bit-exact layout compatibility isn't required for the Rust-only path.

### 11.2 `VARDECL` / `ALLOC` Stack Allocation

The C code uses `VARDECL`/`ALLOC`/`SAVE_STACK`/`RESTORE_STACK` macros for variable-length stack arrays (VLAs or `alloca`). These appear extensively:
- `X[C*N]`, `tf_res[nbEBands]`, `cap[nbEBands]`, `offsets[nbEBands]`, `fine_quant[nbEBands]`, `pulses[nbEBands]`, `fine_priority[nbEBands]`, `collapse_masks[C*nbEBands]`, `freq[N]`, `scratch[N]`, `_exc[MAX_PERIOD+LPC_ORDER]`, `fir_tmp[exc_length]`, etc.

In Rust, use `Vec` for heap allocation or `SmallVec` with a stack-size estimate. For bit-exact performance matching, consider a bump allocator or arena.

### 11.3 In-Place Mutation and Aliased Pointers

Several functions operate in-place or with aliased input/output:
- `comb_filter(out_syn[c], out_syn[c], ...)` — same buffer for input and output
- `celt_iir(buf+offset, ..., buf+offset, ...)` — IIR filter in-place
- `OPUS_MOVE(buf, buf+N, ...)` — overlapping memmove within same buffer
- `decode_mem[c]` and `out_syn[c]` alias (out_syn points into decode_mem)

In Rust, these require careful handling of mutable borrow rules. Options:
- Use `unsafe` blocks with raw pointers for in-place operations
- Split buffers at known offsets using `split_at_mut()`
- Use `copy_within()` for overlapping moves

### 11.4 `c=0; do { ... } while (++c<C)` Loop Pattern

This idiom appears throughout for iterating over channels. In Rust, use `for c in 0..C { ... }`. Note that the C version always executes at least once — but C is always ≥ 1 in valid state, so `for` is equivalent.

### 11.5 Conditional Compilation

Major `#ifdef` blocks that affect the decoder:

| Flag | Purpose | Porting strategy |
|---|---|---|
| `FIXED_POINT` | Fixed-point vs float arithmetic | Use Rust generics or feature flags |
| `ENABLE_DEEP_PLC` | Neural PLC support | Feature flag; separate module |
| `ENABLE_QEXT` | Extended quality mode (96 kHz) | Feature flag |
| `ENABLE_DRED` | Deep redundancy | Feature flag (subset of DEEP_PLC) |
| `ENABLE_RES24` | 24-bit output resolution | Feature flag |
| `CUSTOM_MODES` / `ENABLE_OPUS_CUSTOM_API` | Custom mode support | Feature flag |
| `ENABLE_HARDENING` / `ENABLE_ASSERTIONS` | Validation checks | Always enable in Rust (use `debug_assert!`) |
| `DISABLE_UPDATE_DRAFT` | Disables `disable_inv` default for mono | Likely not needed |
| `RESYNTH` | Makes `deemphasis`/`celt_synthesis` non-static | Not needed in Rust |

### 11.6 Variadic CTL Function

`opus_custom_decoder_ctl(st, request, ...)` uses C variadic arguments. In Rust, replace with:
- An enum for request types: `enum CeltDecoderCtl { SetComplexity(i32), GetComplexity, ... }`
- Individual getter/setter methods
- Or a trait-based approach matching the Opus API pattern

### 11.7 Macro-Heavy Arithmetic

The fixed-point path uses macros (`MULT16_32_Q15`, `SHL32`, `PSHR32`, `SATURATE`, etc.) that expand to different code per platform. In Rust:
- Define these as `#[inline]` functions or trait methods
- For bit-exact matching, the fixed-point versions must use identical shift/multiply/round semantics
- `QCONST16(x, bits)` → `((x) * (1 << bits))` at compile time; use `const fn` in Rust
- `GCONST(x)` → `x` (float) or `(x * (1 << DB_SHIFT))` (fixed); use a trait or const fn

### 11.8 Global/Static Mutable State

- `opus_custom_mode_create()` returns a pointer to a statically-allocated mode — effectively a global singleton. In Rust, use `lazy_static!` or `OnceLock`.
- The decoder struct itself has no global mutable state — all state is per-instance.

### 11.9 NaN Handling

Float path explicitly checks for NaN in the synthesis energy guard:
```c
if (!(S1 > 0.2f*S2))  // catches NaN
```
The negated comparison is intentional — `NaN > x` is false, so `!(NaN > x)` is true, triggering the safety path. Preserve this pattern in Rust (it works identically with IEEE 754 floats).

### 11.10 Critical Bit-Exactness Points

The following operations must be numerically identical to the C reference:
1. Range decoder bit consumption (every `ec_dec_*` call)
2. Energy quantization/dequantization (Laplace coding, fine energy bits)
3. PVQ coefficient decoding (combinatorial number system)
4. Band denormalization (multiply norm coefficients by energy)
5. IMDCT computation
6. Comb filter
7. De-emphasis IIR filter
8. LPC analysis in PLC (autocorrelation, Levinson-Durbin, FIR/IIR)

Any deviation in these will produce different output and fail bit-exact validation.
