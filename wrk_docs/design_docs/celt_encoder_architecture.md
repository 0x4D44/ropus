Now I have all the information needed. Here is the complete architecture document:

---

# CELT Encoder Module — Architecture Documentation

**Source**: `reference/celt/celt_encoder.c` (3161 lines)
**Module ID**: 11 in implementation order (depends on modules 1–10)

---

## 1. Purpose

The CELT encoder is the transform-domain codec within Opus. It converts PCM audio into a compressed bitstream using an MDCT-based (Modified Discrete Cosine Transform) approach with perceptual optimization. CELT handles the full-bandwidth audio path in Opus and the high-frequency extension in hybrid mode (where SILK handles low frequencies and CELT handles the upper bands).

Key responsibilities:
- **Pre-emphasis filtering** to shape the spectral tilt before analysis
- **Pitch pre-filtering** (comb filter) to remove periodic energy before the MDCT
- **MDCT computation** with transient detection for adaptive time/frequency resolution
- **Band energy quantization** using a coarse/fine two-pass scheme
- **Spectral shape quantization** via pyramid vector quantization (PVQ)
- **Bit allocation** using a psychoacoustically-informed rate-distortion framework
- **VBR rate control** with constrained and unconstrained modes

The encoder is called either standalone (Opus CELT-only mode) or by the Opus encoder as part of hybrid encoding.

---

## 2. Public API

### 2.1 Size Query

```c
int celt_encoder_get_size(int channels);
```
- **Parameters**: `channels` — 1 (mono) or 2 (stereo)
- **Returns**: Total byte size needed for the encoder struct (including trailing variable-length arrays)
- **Notes**: Creates a temporary mode internally (48 kHz / 960 samples, or 96 kHz / 1920 for QEXT) to compute the size. The size depends on `mode->overlap`, `mode->nbEBands`, and `COMBFILTER_MAXPERIOD`.

```c
OPUS_CUSTOM_NOSTATIC int opus_custom_encoder_get_size(const CELTMode *mode, int channels);
```
- Computes exact size for a given mode. Formula:
  ```
  sizeof(CELTEncoder)
  + (channels * mode->overlap - 1) * sizeof(celt_sig)      // in_mem
  + channels * COMBFILTER_MAXPERIOD * sizeof(celt_sig)      // prefilter_mem
  + 4 * channels * mode->nbEBands * sizeof(celt_glog)       // oldBandE + oldLogE + oldLogE2 + energyError
  ```

### 2.2 Initialization

```c
int celt_encoder_init(CELTEncoder *st, opus_int32 sampling_rate, int channels, int arch);
```
- **Parameters**:
  - `st` — Pre-allocated memory of size `celt_encoder_get_size(channels)`
  - `sampling_rate` — One of: 8000, 12000, 16000, 24000, 48000 (or 96000 with QEXT)
  - `channels` — 1 or 2
  - `arch` — CPU feature flags (for SIMD dispatch)
- **Returns**: `OPUS_OK` or error code
- **Notes**: Calls `opus_custom_encoder_init_arch()` internally. For rates below 48 kHz, sets `st->upsample` to an appropriate resampling factor (e.g., 6 for 8 kHz). Always creates the mode for 48 kHz / 960 samples internally.

### 2.3 Encoding

```c
int celt_encode_with_ec(CELTEncoder * OPUS_RESTRICT st, const opus_res * pcm,
    int frame_size, unsigned char *compressed, int nbCompressedBytes, ec_enc *enc);
```
- **Parameters**:
  - `st` — Encoder state
  - `pcm` — Input PCM, interleaved if stereo. Type is `opus_res` (float or int16/int32 depending on build)
  - `frame_size` — Samples per channel (before upsampling): 120, 240, 480, 960 (2.5/5/10/20 ms at 48 kHz)
  - `compressed` — Output buffer
  - `nbCompressedBytes` — Max output size (hard-capped to 1275 bytes, or 3825 with QEXT)
  - `enc` — Optional external entropy coder (non-NULL in hybrid mode where Opus encoder pre-writes header bits; NULL for standalone CELT)
- **Returns**: Number of bytes written, or negative error code
- **Key error codes**: `OPUS_BAD_ARG` (bad frame size or null input), `OPUS_INTERNAL_ERROR` (entropy coder error)

### 2.4 Pre-emphasis (Public Helper)

```c
void celt_preemphasis(const opus_res * OPUS_RESTRICT pcmp, celt_sig * OPUS_RESTRICT inp,
    int N, int CC, int upsample, const opus_val16 *coef, celt_sig *mem, int clip);
```
Exposed publicly because the Opus encoder calls it from outside `celt_encode_with_ec` in some configurations.

### 2.5 Control Interface

```c
int opus_custom_encoder_ctl(CELTEncoder * OPUS_RESTRICT st, int request, ...);
```
Variadic control interface. Supported requests:

| Request Constant | Direction | Type | Range | Description |
|---|---|---|---|---|
| `OPUS_SET_COMPLEXITY_REQUEST` | Set | `int` | 0–10 | Encoder complexity |
| `CELT_SET_START_BAND_REQUEST` | Set | `int` | 0–nbEBands-1 | Start band (>0 in hybrid mode) |
| `CELT_SET_END_BAND_REQUEST` | Set | `int` | 1–nbEBands | End band |
| `CELT_SET_PREDICTION_REQUEST` | Set | `int` | 0–2 | 0=force intra, 1=disable PF, 2=normal |
| `OPUS_SET_PACKET_LOSS_PERC_REQUEST` | Set | `int` | 0–100 | Expected packet loss % |
| `OPUS_SET_VBR_CONSTRAINT_REQUEST` | Set | `int` | bool | Constrained VBR |
| `OPUS_SET_VBR_REQUEST` | Set | `int` | bool | VBR enable |
| `OPUS_SET_BITRATE_REQUEST` | Set | `int32` | >500 or MAX | Target bitrate (bps) |
| `CELT_SET_CHANNELS_REQUEST` | Set | `int` | 1–2 | Stream channels (may differ from init channels) |
| `OPUS_SET_LSB_DEPTH_REQUEST` | Set | `int` | 8–24 | Input PCM bit depth |
| `OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST` | Set | `int` | bool | Disable phase inversion in stereo |
| `OPUS_RESET_STATE` | Action | — | — | Reset all encoder state |
| `CELT_SET_ANALYSIS_REQUEST` | Set | `AnalysisInfo*` | ptr | Set external audio analysis |
| `CELT_SET_SILK_INFO_REQUEST` | Set | `SILKInfo*` | ptr | Set SILK codec info (hybrid mode) |
| `CELT_SET_SIGNALLING_REQUEST` | Set | `int` | bool | Enable signalling byte |
| `OPUS_SET_LFE_REQUEST` | Set | `int` | bool | LFE channel mode |
| `OPUS_SET_ENERGY_MASK_REQUEST` | Set | `celt_glog*` | ptr | Surround masking data |
| `OPUS_GET_FINAL_RANGE_REQUEST` | Get | `uint32*` | — | Final entropy coder range |
| `CELT_GET_MODE_REQUEST` | Get | `CELTMode**` | — | Current mode pointer |

---

## 3. Internal State — `struct OpusCustomEncoder`

The struct uses a **variable-length trailing array** pattern. The fixed fields are followed by dynamically-sized arrays whose offsets are computed at runtime.

### 3.1 Configuration Fields (Persist Across Reset)

| Field | Type | Description |
|---|---|---|
| `mode` | `const OpusCustomMode*` | Pointer to mode config (band tables, MDCT lookup, window) |
| `channels` | `int` | Physical channel count (1 or 2) |
| `stream_channels` | `int` | Logical stream channels (can differ from `channels` for mid/side) |
| `force_intra` | `int` | Force intra (no prediction) mode |
| `clip` | `int` | Enable input clipping to ±65536 |
| `disable_pf` | `int` | Disable pitch pre-filter |
| `complexity` | `int` | 0–10, controls algorithm selection |
| `upsample` | `int` | Resampling factor (1 for 48 kHz, 2 for 24 kHz, etc.) |
| `start` | `int` | Start band index (>0 in hybrid mode, typically 17) |
| `end` | `int` | End band index (typically `mode->effEBands`) |
| `bitrate` | `opus_int32` | Target bitrate (bps) or `OPUS_BITRATE_MAX` |
| `vbr` | `int` | VBR enabled flag |
| `signalling` | `int` | Write signalling byte in custom modes |
| `constrained_vbr` | `int` | Constrained VBR mode |
| `loss_rate` | `int` | Expected packet loss percentage |
| `lsb_depth` | `int` | Input bit depth (8–24, default 24) |
| `lfe` | `int` | Low-frequency effects channel mode |
| `disable_inv` | `int` | Disable phase inversion for stereo |
| `arch` | `int` | CPU architecture flags |

### 3.2 Running State Fields (Cleared on Reset)

Reset boundary is marked by `#define ENCODER_RESET_START rng`.

| Field | Type | Description |
|---|---|---|
| `rng` | `opus_uint32` | Entropy coder final range (for verification) |
| `spread_decision` | `int` | Current spreading mode (SPREAD_NONE/LIGHT/NORMAL/AGGRESSIVE) |
| `delayedIntra` | `opus_val32` | Delayed intra energy prediction state |
| `tonal_average` | `int` | Running average for spread decision |
| `lastCodedBands` | `int` | Number of bands coded in previous frame |
| `hf_average` | `int` | High-frequency average for spread decision |
| `tapset_decision` | `int` | Current tapset for comb filter (0, 1, or 2) |
| `prefilter_period` | `int` | Previous frame's pitch period |
| `prefilter_gain` | `opus_val16` | Previous frame's pitch gain |
| `prefilter_tapset` | `int` | Previous frame's comb filter tapset |
| `consec_transient` | `int` | Count of consecutive transient frames |
| `analysis` | `AnalysisInfo` | External audio analysis data (from Opus layer) |
| `silk_info` | `SILKInfo` | SILK codec info for hybrid mode |
| `preemph_memE[2]` | `opus_val32` | Pre-emphasis filter memory (one per channel) |
| `preemph_memD[2]` | `opus_val32` | De-emphasis filter memory (RESYNTH only) |
| `vbr_reservoir` | `opus_int32` | VBR bit reservoir (bits available above/below target) |
| `vbr_drift` | `opus_int32` | Long-term VBR drift correction |
| `vbr_offset` | `opus_int32` | VBR offset to reach target rate |
| `vbr_count` | `opus_int32` | Frame count for VBR averaging (up to 970) |
| `overlap_max` | `opus_val32` | Max sample value in overlap region |
| `stereo_saving` | `opus_val16` | Estimated stereo bit savings |
| `intensity` | `int` | Intensity stereo band threshold |
| `energy_mask` | `celt_glog*` | External surround masking energy (pointer, not owned) |
| `spec_avg` | `celt_glog` | Spectral average for temporal VBR |

### 3.3 Variable-Length Trailing Arrays

These are accessed via pointer arithmetic from `st->in_mem`:

```
in_mem[channels * overlap]                    // Input overlap memory
prefilter_mem[channels * COMBFILTER_MAXPERIOD] // Comb filter history
oldBandE[channels * nbEBands]                  // Previous frame band energies (quantized)
oldLogE[channels * nbEBands]                   // Log energy from previous non-transient frame
oldLogE2[channels * nbEBands]                  // Log energy from two frames ago
energyError[channels * nbEBands]               // Quantization error for energy stabilization
```

**Memory layout** (computed in `celt_encode_with_ec`, lines 1854–1858):
```c
prefilter_mem = st->in_mem + CC * overlap;
oldBandE      = (celt_glog*)(st->in_mem + CC * (overlap + COMBFILTER_MAXPERIOD));
oldLogE       = oldBandE  + CC * nbEBands;
oldLogE2      = oldLogE   + CC * nbEBands;
energyError   = oldLogE2  + CC * nbEBands;
```

### 3.4 State Lifecycle

1. **Allocate**: `celt_encoder_get_size(channels)` bytes
2. **Initialize**: `celt_encoder_init(st, Fs, channels, arch)` — zeros all memory, sets defaults, calls `OPUS_RESET_STATE`
3. **Reset**: `OPUS_RESET_STATE` via CTL — zeros everything from `rng` onward, initializes `oldLogE`/`oldLogE2` to `-28.0` (in dB), sets `delayedIntra=1`, `spread_decision=SPREAD_NORMAL`, `tonal_average=256`
4. **Encode**: `celt_encode_with_ec()` — main per-frame call
5. **Destroy**: `opus_free(st)` (custom modes only)

---

## 4. Algorithm — Main Encode Pipeline

The core encoding function `celt_encode_with_ec()` (line 1725, ~1100 lines) executes the following pipeline:

### Phase 1: Setup and Validation (lines 1725–1966)

1. **Extract mode parameters**: `nbEBands`, `overlap`, `eBands`, `start`, `end`
2. **Determine frame size**: Convert `frame_size` to LM (log2 of the number of short blocks). LM=0 → 2.5ms, LM=1 → 5ms, LM=2 → 10ms, LM=3 → 20ms. `M = 1 << LM`, `N = M * shortMdctSize`
3. **Compute pointers** to trailing arrays (`prefilter_mem`, `oldBandE`, etc.)
4. **Initialize entropy coder** if `enc == NULL` (standalone mode)
5. **Signalling byte** (custom modes only): Write end-band, LM, stereo flag
6. **Compute available bytes**: Handle VBR rate computation and constrained VBR reservoir logic
7. **Detect silence**: `sample_max == 0` (fixed-point) or below LSB depth threshold (float)

### Phase 2: Input Processing (lines 2008–2062)

8. **Pre-emphasis**: For each channel, apply the pre-emphasis filter via `celt_preemphasis()`. Copy overlap from prefilter memory. The pre-emphasis is a first-order high-pass: `y[n] = x[n] - coef0 * x[n-1]` (simple case) or a second-order IIR (custom modes with `coef[1] != 0`).

9. **Tone detection** (`tone_detect()`, line 2021): Fits a 2nd-order LPC model to detect pure tones. If poles are complex and the squared radius (toneishness) is high (>0.98), the detected frequency is used to override pitch estimation and prevent transient false alarms.

10. **Transient analysis** (`transient_analysis()`, line 2030): High-pass filters the input, computes forward/backward masking envelopes, and evaluates a noise-to-mask ratio (`mask_metric`). Returns `isTransient` if `mask_metric > 200`. Weak transients (metric 200–600) are flagged separately for hybrid low-bitrate handling. Suppresses false transients on pure tones.

11. **Pitch pre-filter** (`run_prefilter()`, line 2041): Finds pitch period via `pitch_downsample()` → `pitch_search()` → `remove_doubling()`. Applies inverse comb filter to remove pitch periodicity before MDCT. Encodes pitch parameters (period, gain, tapset) into the bitstream. For pure tones, bypasses pitch search and computes period directly from tone frequency.

### Phase 3: Transform and Energy (lines 2063–2232)

12. **MDCT computation** (`compute_mdcts()`, line 2090): Computes the MDCT of the pre-filtered input. If `isTransient`, uses `M` short blocks (interleaved output). Otherwise, one long block. For stereo with `stream_channels==1`, sums L+R into mono. For upsampled modes, zeroes upper-frequency bins.

13. **Second MDCT** (line 2076–2088): If transient and `complexity >= 8`, also computes a long-block MDCT for `bandLogE2` (used in dynalloc analysis for more reliable energy estimates).

14. **Band energy computation**: `compute_band_energies()` → `amp2Log2()` converts from linear to log2 domain.

15. **Patch transient decision** (line 2215–2232): At `complexity >= 5`, checks for sudden energy increases between frames. If detected, forces transient mode and recomputes MDCT + energies.

16. **Encode transient flag**: 1 bit with logp=3.

### Phase 4: Analysis and Bit Allocation (lines 2237–2628)

17. **Band normalization** (`normalise_bands()`): Divides each band's MDCT coefficients by the band energy, producing unit-norm vectors per band.

18. **Dynamic allocation analysis** (`dynalloc_analysis()`, line 2248): Computes per-band boost offsets by analyzing the spectral envelope, masking model, and noise floor. Key sub-steps:
    - Compute noise floor per band (accounting for preemphasis, band width, eMeans)
    - Compute signal-to-mask ratio (SMR) for spread weights
    - Use follower envelope with median filtering to detect spectral peaks needing extra bits
    - Tone compensation: boost bands containing detected tone frequency
    - Cap dynalloc at 2/3 of total bits for CBR/non-transient CVBR

19. **TF analysis** (`tf_analysis()`, line 2258): For each band, evaluates which time-frequency resolution (controlled by Haar wavelets) minimizes the L1 norm. Uses Viterbi dynamic programming to find the optimal per-band TF resolution, balancing cost of switching between bands. Encodes TF flags with adaptive log-probability.

20. **Energy quantization**: 
    - **Coarse** (`quant_coarse_energy()`, line 2295): Encodes band energies with inter-frame prediction. Uses Laplace coding. May use intra (no prediction) or inter mode.
    - Energy is bias-corrected using `energyError` from the previous frame (line 2289–2293).

21. **TF encoding** (`tf_encode()`, line 2300): Writes per-band TF flags and the tf_select bit.

22. **Spread decision** (lines 2302–2348): Determines noise spreading mode based on tonality analysis. At `complexity >= 3` with enough bits, calls `spreading_decision()` which analyzes spectral flatness. Otherwise uses heuristic defaults.

23. **Alloc trim** (`alloc_trim_analysis()`, line 2416): Computes a trim value (0–10) that adjusts how bits are distributed between low and high frequencies. Factors in: inter-channel correlation, spectral tilt, surround masking, TF estimate, and tonality slope.

24. **Dynalloc encoding** (lines 2356–2389): Encodes per-band boost values using adaptive binary coding with decreasing log-probability.

25. **Stereo decisions** (lines 2391–2406): Intensity stereo threshold via hysteresis. Dual stereo decision via `stereo_analysis()` (L1 norm comparison of L/R vs M/S).

26. **Bit allocation** (`clt_compute_allocation()`, line 2626): The central rate-distortion allocator. Takes all the analysis outputs (offsets, caps, alloc_trim, intensity, dual_stereo) and distributes available bits across bands and between fine energy / PVQ pulses. Returns `codedBands` (number of bands that received bits).

### Phase 5: Quantization (lines 2634–2716)

27. **Fine energy quantization** (`quant_fine_energy()`, line 2634): Refines band energies using the bits allocated by `clt_compute_allocation`.

28. **Residual quantization** (`quant_all_bands()`, line 2670): Quantizes the normalized MDCT coefficients using PVQ (pyramid vector quantization) with the allocated pulse counts. Handles stereo with intensity/dual-stereo switching. Produces `collapse_masks` indicating which bands got any pulses.

29. **Anti-collapse** (lines 2697–2704): If transient with LM≥2, signals whether to apply anti-collapse processing on the decoder side (prevents silence artifacts when a band receives zero pulses in a short block that previously had energy).

30. **Energy finalization** (`quant_energy_finalise()`, line 2706): Uses any remaining bits to refine band energies further, in priority order.

### Phase 6: State Update and Finalization (lines 2718–2831)

31. **Store quantization error** for next frame's energy bias correction (line 2711).
32. **Update oldBandE/oldLogE/oldLogE2**: For non-transient frames, shift the history. For transient frames, take the minimum (conservative prediction).
33. **Clear out-of-range bands** to `-28 dB`.
34. **Update persistent state**: `consec_transient`, `rng`, `prefilter_period/gain/tapset`, `lastCodedBands`.
35. **Finalize bitstream** (`ec_enc_done()`).
36. **Re-synthesis** (RESYNTH mode only): Full decode path for analysis/debugging.

### VBR Rate Control (lines 2435–2533)

VBR is computed in parallel with Phase 4:

1. **Base target** = `vbr_rate - overhead` (per-frame bit budget from target bitrate)
2. **`compute_vbr()`** adjusts target based on: activity, stereo savings, dynalloc boost, transient boost, tonality, surround masking, temporal VBR, floor depth
3. **Constrained VBR**: Maintains a reservoir (`vbr_reservoir`), drift correction (`vbr_drift`), and offset (`vbr_offset`). Uses exponential averaging with `alpha = 1/(count+20)` converging to 0.001.
4. **Rate clamping**: Never exceeds 2× base target. Minimum ensures no entropy coder desync.

---

## 5. Data Flow

### Input
```
pcm[CC * frame_size/upsample]   (interleaved: L0,R0,L1,R1,...)
    CC = physical channels, frame_size = N/upsample
    Type: opus_res (float or int16 depending on build)
```

### Internal Buffers (stack-allocated per frame)

| Buffer | Size | Description |
|---|---|---|
| `in` | `CC * (N + overlap)` | Pre-emphasized, pre-filtered input with overlap prepended |
| `freq` | `CC * N` | MDCT coefficients (interleaved sub-frames for transient) |
| `X` | `C * N` | Normalized MDCT coefficients (unit-norm per band) |
| `bandE` | `CC * nbEBands` | Band energies (linear) |
| `bandLogE` | `CC * nbEBands` | Band energies (log2 domain) |
| `bandLogE2` | `C * nbEBands` | Long-block band energies (for second MDCT) |
| `error` | `C * nbEBands` | Energy quantization error |
| `offsets` | `nbEBands` | Dynamic allocation boost per band |
| `tf_res` | `nbEBands` | TF resolution per band |
| `fine_quant` | `nbEBands` | Fine energy bits per band |
| `pulses` | `nbEBands` | PVQ pulse counts per band |
| `fine_priority` | `nbEBands` | Priority order for energy finalization |
| `cap` | `nbEBands` | Max bits per band |
| `collapse_masks` | `C * nbEBands` | Anti-collapse tracking |
| `importance` | `nbEBands` | Band importance weights for TF/dynalloc |
| `spread_weight` | `nbEBands` | Spread decision weights |

### Output
```
compressed[nbCompressedBytes]   (bitstream, max 1275 bytes standard, 3825 with QEXT)
Returns: actual compressed bytes written (positive) or error code (negative)
```

### Bitstream Layout (approximate order)

1. Silence flag (1 bit, logp=15) — only if at start of packet
2. Pitch pre-filter on/off (1 bit, logp=1) — only if not hybrid and enough bits
3. Pitch parameters: octave (uint, max 6), period (4+octave bits), gain (3 bits), tapset (ICDF)
4. Transient flag (1 bit, logp=3)
5. Coarse energy (Laplace-coded, intra or inter prediction)
6. TF flags (per-band binary, logp=2/4 first, then 4/5)
7. TF select (1 bit, logp=1)
8. Spread mode (ICDF, 4 symbols)
9. Dynamic allocation boosts (adaptive binary per band)
10. Alloc trim (ICDF, 11 symbols)
11. Stereo intensity + dual_stereo (from allocator)
12. Fine energy bits
13. PVQ-coded spectral coefficients
14. Anti-collapse flag (1 bit) — only if transient with LM≥2
15. Energy finalization bits (remaining bits)
16. QEXT extension payload (if enabled): padding, extension header, extra energy + spectral data

---

## 6. Numerical Details

### 6.1 Fixed-Point Q Formats

| Type / Variable | Q Format | Range | Notes |
|---|---|---|---|
| `opus_val16` | Varies | int16 | General 16-bit fixed-point |
| `opus_val32` | Varies | int32 | General 32-bit fixed-point |
| `celt_sig` | Q`SIG_SHIFT` = Q12 | int32 | Signal domain (PCM after pre-emphasis) |
| `celt_norm` | Q`NORM_SHIFT` = Q24 | int32 | Normalized MDCT coefficients |
| `celt_glog` | Q`DB_SHIFT` = Q24 | int32 | Log-domain energies (log2 scale, ~6.02 dB per unit) |
| `celt_ener` | — | int32 | Linear band energy (unsigned) |
| `prefilter_gain` | Q15 | 0–32767 | Pitch gain, quantized to 8 levels (0.09375 step) |
| `tf_estimate` | Q14 | 0–16384 | Transient strength metric |
| `stereo_saving` | Q8 | — | Estimated stereo bit savings |
| `alloc_trim` | Q8 (intermediate) | 0–10 (integer output) | Spectral tilt trim |
| `toneishness` | Q29 | 0–2^29 | Pole radius squared from LPC analysis |
| `tone_freq` | Q13 | 0–π | Detected tone frequency (radians) |
| `delayedIntra` | Q? | — | Intra prediction delay state |
| `temporal_vbr` | Q`DB_SHIFT` | ±1.5–3.0 dB | Temporal VBR adjustment |

### 6.2 Key Arithmetic Constants

```c
#define SIG_SHIFT   12      // celt_sig fractional bits
#define NORM_SHIFT  24      // celt_norm fractional bits
#define DB_SHIFT    24      // celt_glog fractional bits (but used as log2, so 1.0 ≈ 6.02 dB)
#define BITRES      3       // Sub-bit resolution for bit allocation (1/8 bit precision)
#define EPSILON     1       // Fixed-point epsilon (prevents division by zero)
```

### 6.3 Overflow Guards

- **Transient analysis** (line 298): Input shift `in_shift = max(0, ilog2(1 + maxabs(in)) - 14)` to prevent overflow in the high-pass filter.
- **Normalization** after high-pass (line 354–363): Re-shifts `tmp` to use full 14-bit range.
- **Forward/backward masking** (lines 369–399): Uses `PSHR32` (pseudo-shift-right with rounding) to maintain precision.
- **`mask_metric` normalization** (line 437): `64 * unmask * 4 / (6 * (len2 - 17))` — integer division, potential for overflow with large unmask values.
- **`tone_lpc()`** (line 1343–1356): Division guard: `den <= SHR32(MULT32_32_Q31(r00,r11), 10)` prevents near-singular systems. Output LPC coefficients clamped to ±1.0 (Q29) and ±2.0 (Q29).
- **`acos_approx()`** (lines 1290–1301): Fixed-point polynomial approximation of acos, operates in Q14 intermediate.

### 6.4 Rounding Behavior

- `PSHR32(a, shift)` = `(a + (1 << (shift-1))) >> shift` — round-to-nearest for positive values, rounds toward +∞ for negative midpoints.
- `SROUND16(y, 2)` in transient analysis — shift with rounding to Q14.
- `MULT16_16_Q15(a, b)` = `(a * b) >> 15` — truncation (floor), not rounding.
- Quantized gain: `qg = ((gain1 + 1536) >> 10) / 3 - 1` — biased rounding to map continuous gain to 8 discrete levels.
- Alloc trim: `PSHR32(trim, 8)` (fixed-point) vs `floor(0.5 + trim)` (float) — round-to-nearest.

### 6.5 Log-Domain Energy Conventions

Band energies are stored in a `celt_glog` format where the scale is **log2**: a value of `1.0` (i.e., `1 << DB_SHIFT` in fixed-point) represents a **factor of 2** (≈6.02 dB). The `GCONST(x)` macro converts a floating-point dB-like value to this format. The silence threshold is `-28.0` in this domain (≈ -168 dB).

---

## 7. Dependencies

### 7.1 Modules Called by celt_encoder

| Module | Functions Used | Purpose |
|---|---|---|
| `mdct.c` | `clt_mdct_forward()` | Forward MDCT transform |
| `pitch.c` | `pitch_downsample()`, `pitch_search()`, `remove_doubling()` | Pitch estimation |
| `bands.c` | `compute_band_energies()`, `normalise_bands()`, `quant_all_bands()`, `spreading_decision()`, `haar1()`, `anti_collapse()` | Band processing and spectral quantization |
| `quant_bands.c` | `quant_coarse_energy()`, `quant_fine_energy()`, `quant_energy_finalise()`, `amp2Log2()` | Energy quantization |
| `rate.c` | `clt_compute_allocation()`, `init_caps()` | Bit allocation |
| `entenc.c` | `ec_enc_init()`, `ec_enc_bit_logp()`, `ec_enc_uint()`, `ec_enc_bits()`, `ec_enc_icdf()`, `ec_enc_shrink()`, `ec_enc_done()` | Entropy coding |
| `entcode.c` | `ec_tell()`, `ec_tell_frac()`, `ec_get_error()` | Bitstream bookkeeping |
| `celt_lpc.c` | `celt_fir_comb()` (via `comb_filter`) | Comb filter implementation |
| `mathops.c` | `celt_sqrt()`, `celt_log2()`, `celt_exp2_db()`, `celt_rcp()`, `celt_maxabs32()`, `celt_ilog2()`, `frac_div32_q29()` | Fixed-point math |
| `modes.c` | `opus_custom_mode_create()` | Mode creation (for init) |
| `vq.c` | (via `quant_all_bands`) | PVQ vector quantization |

### 7.2 What Calls This Module

| Caller | Function Called | Context |
|---|---|---|
| `opus_encoder.c` | `celt_encoder_init()`, `celt_encoder_get_size()`, `celt_encode_with_ec()`, `opus_custom_encoder_ctl()`, `celt_preemphasis()` | Opus top-level encoder |
| Custom mode API | `opus_custom_encoder_create()`, `opus_custom_encode()`, `opus_custom_encode_float()` | Standalone CELT usage |

---

## 8. Constants and Tables

### 8.1 Static Tables in celt_encoder.c

**`inv_table[128]`** (line 286): Precomputed `6*64/x` values, trained on real data to minimize average error. Used in transient analysis to compute the harmonic mean of the temporal masking function. Index is derived from normalized energy values.

**`intensity_thresholds[21]` / `intensity_histeresis[21]`** (lines 2393–2397): Bitrate-dependent thresholds (in kbps) for intensity stereo band selection. Higher bitrates allow intensity stereo at higher frequencies. Hysteresis prevents toggling.

### 8.2 External Tables Referenced

**`tf_select_table[4][8]`** (from `rate.c`): Maps LM × isTransient × tf_select × tf_res to actual TF resolution values.

**`spread_icdf`**, **`trim_icdf`**, **`tapset_icdf`** (from `rate.c`): ICDF tables for entropy coding of spread mode, alloc trim, and prefilter tapset.

**`eMeans[25]`** (from `quant_bands.c`): Mean log energy per band, used as offset in noise floor computation.

**`mode->eBands[]`** (from mode definition): Band boundary table mapping band indices to frequency bins. For 48 kHz / 960 samples: 21 bands, typically with boundaries like {0,1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,34,40,48,60,72,84}.

**`mode->logN[]`**: Log of band width, used for noise floor calculation.

**`mode->window[]`**: MDCT window coefficients (overlap-add window).

**`mode->preemph[4]`**: Pre-emphasis filter coefficients. Index 0 is the primary coefficient (~0.85 for standard mode).

### 8.3 Magic Numbers

| Value | Location | Meaning |
|---|---|---|
| `200` | line 444 | Transient detection threshold (mask_metric) |
| `600` | line 454 | Weak transient upper threshold |
| `27` | line 459 | Scaling for VBR tf_max metric |
| `42` | line 459 | Offset for VBR tf_max metric |
| `-28.0` | line 2721, 2796–2801, 3085 | Silence/reset energy level in log2 dB (~-168 dB) |
| `256` | line 3089 | Initial tonal_average (midpoint of 0–512 range) |
| `970` | line 2501 | VBR count limit (after which alpha becomes constant 0.001) |
| `1275` | line 1801 | Max standard Opus packet bytes |
| `3825` | QEXT constant | Max QEXT packet bytes (3 × 1275) |
| `15` | COMBFILTER_MINPERIOD | Minimum pitch period (~3.2 kHz at 48 kHz) |
| `1024` | COMBFILTER_MAXPERIOD | Maximum pitch period (~46.9 Hz at 48 kHz) |

---

## 9. Edge Cases

### 9.1 Error Conditions

- **`nbCompressedBytes < 2`** or **`pcm == NULL`**: Returns `OPUS_BAD_ARG` immediately.
- **Invalid frame size** (no matching LM for `shortMdctSize << LM == frame_size`): Returns `OPUS_BAD_ARG`.
- **Entropy coder error** (`ec_get_error(enc)`): Returns `OPUS_INTERNAL_ERROR`. This can happen if the encoder writes more bits than available.

### 9.2 Silence Handling

When silence is detected (line 1972–2007):
- In VBR mode, packet is shrunk to minimum (2 bytes)
- A silence flag is encoded (1 bit, logp=15)
- The remaining bits are treated as filled with zeros
- `oldBandE` is set to -28.0 for all bands
- VBR delta is zeroed (no drift adjustment during silence)

### 9.3 Hybrid Mode

When `start > 0` (hybrid mode):
- CELT only encodes bands from `start` to `end` (typically bands 17–21, covering ~8 kHz and above)
- Pitch pre-filter is disabled
- Stereo saving and alloc trim are forced to defaults
- VBR uses a simplified target computation based on SILK info
- Minimum allowed bytes must account for 37 bits needed for redundancy signalling

### 9.4 LFE Mode

When `st->lfe == 1`:
- Band energies above band 2 are clamped to -40 dB relative to band 0
- Dynamic allocation is simplified (fixed offset for band 0)
- Signal bandwidth is set to 1 band
- Transient analysis is disabled
- Spread decision forced to `SPREAD_NORMAL`

### 9.5 Tone Protection

Tones with `toneishness > 0.98` (near-pure sinusoids):
- Suppress transient detection to prevent false alarms from partial cycles
- Override pitch pre-filter period to tone-derived value
- Boost dynalloc for bands containing the tone
- TF analysis is disabled when `toneishness >= 0.98`

### 9.6 Weak Transients (Hybrid Low Bitrate)

When `allow_weak_transients` is true and `mask_metric` is 200–600:
- Transient flag is not set (saves bits)
- Instead, TF resolution is forced to 1 for all bands (coarser temporal resolution)
- Prevents energy collapse artifacts at very low bitrate

### 9.7 Pre-filter Cancellation

The comb filter gain-check (lines 1559–1583) can revert the pre-filter if it makes the signal worse:
- **Stereo**: Cancelled if either channel's energy increases by more than `25% * gain * before + 1% * other_channel`, or if neither channel improves significantly
- **Mono**: Cancelled if `after > before`

---

## 10. Porting Notes

### 10.1 Variable-Length Struct with Trailing Arrays

The `OpusCustomEncoder` struct uses C's **flexible array member** pattern with `celt_sig in_mem[1]` at the end. The actual allocation is much larger, and pointer arithmetic is used to access the trailing arrays. In Rust:
- Use a `Vec<u8>` or custom allocator for the raw buffer, or
- **Preferred**: Break the trailing arrays into explicit `Vec` fields in the Rust struct:
  ```rust
  struct CeltEncoder {
      // ... fixed fields ...
      in_mem: Vec<f32>,          // [channels * overlap]
      prefilter_mem: Vec<f32>,   // [channels * COMBFILTER_MAXPERIOD]
      old_band_e: Vec<f32>,      // [channels * nb_ebands]
      old_log_e: Vec<f32>,       // [channels * nb_ebands]
      old_log_e2: Vec<f32>,      // [channels * nb_ebands]
      energy_error: Vec<f32>,    // [channels * nb_ebands]
  }
  ```

### 10.2 Stack Allocation Macros (VARDECL/ALLOC)

Every function in this module uses `VARDECL`/`ALLOC` for temporary buffers:
```c
VARDECL(celt_sig, in);
ALLOC(in, CC*(N+overlap), celt_sig);
```
These expand to either VLAs, `alloca`, or a global pseudo-stack. In Rust:
- Use `Vec::with_capacity()` for large allocations
- Consider a reusable scratch buffer pool to avoid per-frame allocation
- Some allocations are large: `in` can be up to `2 * (960 + 120) = 2160` celt_sig values, `freq` up to `2 * 960 = 1920`, `X` up to `2 * 960 = 1920` celt_norm values

### 10.3 Conditional Compilation

The file has extensive conditional compilation:

| Macro | Purpose | Porting Strategy |
|---|---|---|
| `FIXED_POINT` | Fixed-point vs float build | Choose one path (likely float for initial port), but must support fixed-point for bit-exactness |
| `ENABLE_QEXT` | Quality extension (Opus 1.5+) | Feature flag or separate module |
| `RESYNTH` | Re-synthesis for analysis | Debug-only feature, can defer |
| `CUSTOM_MODES` / `ENABLE_OPUS_CUSTOM_API` | Custom mode support | Feature flag |
| `DISABLE_FLOAT_API` | Disable float API | Not relevant for Rust |
| `FUZZING` | Random decisions for fuzz testing | Feature flag |
| `ENABLE_RES24` | 24-bit resolution input | Feature flag |

### 10.4 In-Place Mutation and Aliasing

Several patterns require care:
- **`in` buffer reuse**: The input buffer `in` is modified in-place by the pre-filter, then the overlap region is overwritten with memory from the previous frame.
- **`prefilter_mem` update** (lines 1585–1595): Uses `OPUS_MOVE` (memmove) for overlapping regions when `N <= max_period`.
- **`tmp` array reuse** in `transient_analysis()`: The same array is first used for the high-pass filtered signal, then reused for forward/backward masking envelopes (at half size).
- **`oldBandE` writes**: The encoder writes to `oldBandE` which is the quantized output of `quant_coarse_energy` + `quant_fine_energy`.

### 10.5 CC vs C (Channels)

Two channel count variables are used throughout:
- **`CC`** = `st->channels` = physical channels (always matches init)
- **`C`** = `st->stream_channels` = logical stream channels (can be set to 1 for mono downmix of stereo input)
- When `CC==2 && C==1`: stereo input is downmixed to mono in the MDCT output (`compute_mdcts`, line 541)

### 10.6 Entropy Coder Integration

The encoder can receive a **pre-initialized** entropy coder from the Opus layer (hybrid mode). In this case:
- `enc != NULL` on entry, with some bits already written
- `tell = ec_tell(enc)` may be non-zero
- `nbFilledBytes` accounts for pre-written bytes
- The `ec_enc_shrink()` calls resize the output buffer

For standalone CELT, `enc == NULL` and a local `ec_enc _enc` is initialized. The Rust port should model this as an `Option<&mut RangeEncoder>` or similar.

### 10.7 Variadic CTL Interface

`opus_custom_encoder_ctl` uses C variadic arguments (`va_list`). In Rust, this should become an enum-based approach:
```rust
enum CeltEncoderCtl {
    SetComplexity(i32),
    SetBitrate(i32),
    ResetState,
    // ...
}
fn encoder_ctl(&mut self, request: CeltEncoderCtl) -> Result<(), OpusError>;
```

### 10.8 The `do { ... } while (++c<CC)` Pattern

Channel loops use this C idiom extensively:
```c
c=0; do {
    // ... process channel c ...
} while (++c<CC);
```
This always executes at least once (assumes CC ≥ 1). In Rust, use `for c in 0..cc { ... }`.

### 10.9 Bit-Exact Concerns

The following are critical for bit-exactness:
1. **Integer division rounding**: C truncates toward zero. Rust's `/` also truncates toward zero for integers — this matches.
2. **Shift behavior**: C's right shift of signed negative values is implementation-defined but universally arithmetic-shift on all targets. Rust's `>>` on signed types is arithmetic — this matches.
3. **`MULT16_16_Q15` truncation**: This is `(a*b) >> 15` (truncation, not rounding). The float equivalent `(a)*(b)` will differ.
4. **`PSHR32` rounding**: Adds `1 << (shift-1)` before shifting. Must be reproduced exactly.
5. **`ec_tell_frac`**: Returns position in 1/8-bit units. The VBR computation and bit allocation depend on exact bit accounting.
6. **Integer overflow**: The C code relies on unsigned overflow being well-defined (wrapping). Signed overflow is technically UB in C but the code avoids it. In Rust, use wrapping arithmetic for `opus_uint32` operations.

### 10.10 QEXT Extension

The QEXT (Quality Extension) code at lines 2534–2594 and 2636–2695 is substantial and involves:
- A secondary entropy encoder (`ext_enc`) writing to a separate payload region
- Packet restructuring: inserting padding bytes, converting to Code 3 format
- `OPUS_MOVE` of the compressed buffer to make room for the extension header
- A secondary mode (`qext_mode_struct`) for the extended frequency bands

This should likely be ported as a separate optional module/feature.

### 10.11 Functions to Port from This File

| Function | Lines | Complexity | Notes |
|---|---|---|---|
| `celt_encoder_get_size` | 144–152 | Low | Size computation |
| `opus_custom_encoder_get_size` | 154–174 | Low | Exact size formula |
| `opus_custom_encoder_init_arch` | 194–231 | Low | Initialization |
| `celt_encoder_init` | 240–257 | Low | Public init wrapper |
| `transient_analysis` | 267–469 | High | Complex fixed-point processing, inv_table |
| `patch_transient_decision` | 473–507 | Medium | Spreading function on old energies |
| `compute_mdcts` | 511–554 | Medium | Calls clt_mdct_forward, interleaving |
| `celt_preemphasis` | 557–646 | Medium | Multiple code paths (standard, QEXT, clip) |
| `l1_metric` | 650–661 | Low | L1 norm with bias |
| `tf_analysis` | 663–822 | High | Viterbi DP, Haar wavelets |
| `tf_encode` | 824–862 | Medium | Adaptive entropy coding |
| `alloc_trim_analysis` | 865–955 | High | Many analysis factors |
| `stereo_analysis` | 957–987 | Medium | L/R vs M/S comparison |
| `median_of_5` / `median_of_3` | 990–1047 | Low | Sorting network |
| `dynalloc_analysis` | 1049–1273 | High | Complex masking model, many sub-steps |
| `normalize_tone_input` | 1276–1289 | Low | Fixed-point only |
| `acos_approx` | 1290–1301 | Low | Fixed-point only, polynomial |
| `tone_lpc` | 1305–1359 | Medium | 2nd-order LPC via least squares |
| `tone_detect` | 1362–1402 | Medium | Tone detector, LPC + root analysis |
| `run_prefilter` | 1404–1602 | High | Pitch search + comb filter application |
| `compute_vbr` | 1604–1716 | High | Many VBR adjustments |
| `celt_encode_with_ec` | 1725–2832 | Very High | Main encode pipeline (~1100 lines) |
| `opus_custom_encoder_ctl` | 2941–3161 | Medium | Variadic dispatch |

Total: ~3161 lines of C, approximately 2500 lines of substantive logic (excluding custom mode wrappers and boilerplate).
