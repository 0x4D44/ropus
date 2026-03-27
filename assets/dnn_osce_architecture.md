I now have a thorough understanding of all source files. Here is the complete architecture document:

---

# OSCE Module Architecture Documentation

## Module: `dnn/osce` — Opus Speech Coding Enhancement

**Files:** `osce.c`, `osce.h`, `osce_config.h`, `osce_structs.h`, `osce_features.c`, `osce_features.h`
**Author:** Jan Buethe (Amazon, 2023)
**Reference commit:** xiph/opus (Opus 1.4+ DNN branch)

---

## 1. Purpose

OSCE (Opus Speech Coding Enhancement) is a neural-network-based post-filter that enhances the output of the SILK decoder at decode time. It sits between the SILK synthesis stage and the final output, applying learned adaptive filtering to improve perceptual quality of decoded speech without changing the bitstream format.

OSCE provides three distinct capabilities:

1. **LACE** (Lightweight Adaptive Comb Enhancement): A lightweight model using two adaptive comb filters and one adaptive convolution to enhance SILK decoded speech.
2. **NoLACE** (No-LACE / enhanced LACE): A more complex model with deeper filtering pipelines — two adaptive comb filters, four adaptive convolutions, and three adaptive time-domain shaping stages.
3. **BBWENet** (Broadband Bandwidth Extension Network): A neural bandwidth extension module that upsamples 16 kHz decoded SILK speech to 48 kHz, generating high-frequency content that was not transmitted.

**Position in the Opus pipeline:**
```
SILK Decoder → osce_enhance_frame() → PCM output (16 kHz, in-band enhancement)
                                     ↓
SILK Decoder → osce_bwe()           → PCM output (48 kHz, bandwidth-extended)
```

`osce_enhance_frame` is called from `silk/decode_frame.c` immediately after SILK synthesis, before PLC state update. `osce_bwe` is called from `opus_decoder.c` when bandwidth extension is enabled and the output sample rate is 48 kHz.

---

## 2. Public API

### 2.1 `osce_enhance_frame`
```c
void osce_enhance_frame(
    OSCEModel              *model,      /* I    OSCE model struct (weights)              */
    silk_decoder_state     *psDec,      /* I/O  SILK decoder state (contains OSCE state) */
    silk_decoder_control   *psDecCtrl,  /* I    Decoder control (LPC, LTP, gains, pitch) */
    opus_int16              xq[],       /* I/O  Decoded speech, 320 samples (20ms@16kHz) */
    opus_int32              num_bits,   /* I    Size of SILK payload in bits              */
    int                     arch        /* I    Run-time CPU architecture                */
);
```
**Behavior:**
- Only processes 20 ms frames at 16 kHz (320 samples, 4 subframes of 80 samples). Returns immediately (no-op) for other configurations.
- Reads `xq` as input, writes enhanced signal back into `xq` (in-place).
- Calculates features from decoder state, then dispatches to LACE or NoLACE based on `psDec->osce.method`.
- If model is not loaded, copies input to output unchanged.
- On reset (first 2 frames after method change), suppresses or cross-fades enhancement to avoid transient artifacts.

### 2.2 `osce_load_models`
```c
int osce_load_models(OSCEModel *hModel, const void *data, int len);
```
**Returns:** 0 on success, -1 on failure.
**Behavior:**
- If `data != NULL && len > 0`: parses weight blob via `parse_weights()`, initializes LACE, NoLACE, and BBWENet models from parsed `WeightArray` list.
- If `data == NULL`: uses compiled-in weights (`lacelayers_arrays`, `nolacelayers_arrays`, `bbwenetlayers_arrays`). Returns -1 if `USE_WEIGHTS_FILE` is defined (weights must be loaded from file).
- Initializes overlap windows for each model via `compute_overlap_window()`.

### 2.3 `osce_reset`
```c
void osce_reset(silk_OSCE_struct *hOSCE, int method);
```
**Behavior:**
- Clears feature state (`OSCEFeatureState`) to zero.
- Resets the selected method's processing state (LACEState or NoLACEState), including reinitializing AdaComb, AdaConv, and AdaShape sub-states.
- Sets `features.reset = 2`, meaning the next two frames will suppress/cross-fade enhancement output.
- Stores `method` in `hOSCE->method`.

### 2.4 `osce_bwe` (conditional: `ENABLE_OSCE_BWE`)
```c
void osce_bwe(
    OSCEModel              *model,      /* I    OSCE model struct                        */
    silk_OSCE_BWE_struct   *psOSCEBWE,  /* I/O  OSCE BWE state                           */
    opus_int16              xq48[],     /* O    Bandwidth-extended speech (48 kHz)        */
    opus_int16              xq16[],     /* I    Decoded speech (16 kHz)                   */
    opus_int32              xq16_len,   /* I    Length of xq16 (160 or 320 samples)       */
    int                     arch        /* I    Run-time CPU architecture                 */
);
```
**Behavior:**
- Accepts 10 ms or 20 ms frames (160 or 320 samples at 16 kHz).
- Produces 3x as many samples at 48 kHz.
- Applies a 21-sample output delay (`OSCE_BWE_OUTPUT_DELAY`) buffered in `psOSCEBWE->state.bbwenet.outbut_buffer`.

### 2.5 `osce_bwe_reset` (conditional: `ENABLE_OSCE_BWE`)
```c
void osce_bwe_reset(silk_OSCE_BWE_struct *hOSCEBWE);
```
Clears BWE feature state and resets BBWENet processing state. Notably initializes `last_spec[]` real parts to `1e-9` (matching Python reference behavior).

### 2.6 Feature Calculation
```c
void osce_calculate_features(
    silk_decoder_state     *psDec,
    silk_decoder_control   *psDecCtrl,
    float                  *features,    /* O  [4 * 93] features for 4 subframes    */
    float                  *numbits,     /* O  [2] raw and smoothed bit count       */
    int                    *periods,     /* O  [4] pitch lags per subframe          */
    const opus_int16        xq[],        /* I  320 decoded samples                  */
    opus_int32              num_bits     /* I  SILK payload size in bits            */
);
```

```c
void osce_bwe_calculate_features(
    OSCEBWEFeatureState    *psFeatures,
    float                  *features,    /* O  [num_frames * 114] BWE features      */
    const opus_int16        xq[],        /* I  Decoded speech at 16 kHz             */
    int                     num_samples  /* I  Number of samples                    */
);
```

### 2.7 Cross-fade Utilities
```c
void osce_cross_fade_10ms(float *x_enhanced, float *x_in, int length);
void osce_bwe_cross_fade_10ms(opus_int16 *x_fadein, opus_int16 *x_fadeout, int length);
```
Apply 10 ms (160-sample / 480-sample at 48 kHz) cross-fade using `osce_window[]`.

---

## 3. Internal State

### 3.1 Feature State

```c
typedef struct {
    float  numbits_smooth;                              // IIR-smoothed bit count
    int    pitch_hangover_count;                         // Hangover counter for pitch
    int    last_lag;                                     // Previous pitch lag
    int    last_type;                                    // Previous frame type (voiced/unvoiced)
    float  signal_history[OSCE_FEATURES_MAX_HISTORY];   // 350 samples of history
    int    reset;                                        // Reset counter (2→1→0)
} OSCEFeatureState;
```
- `signal_history`: ring buffer of past decoded samples (as float, normalized to [-1, 1]). 350 samples = 21.875 ms at 16 kHz, sufficient for pitch analysis lookback.
- `reset`: countdown from 2. At 2, enhancement output is entirely suppressed (input passed through). At 1, a 10 ms cross-fade blends enhanced with input. At 0, full enhancement.
- `numbits_smooth`: exponential moving average with alpha=0.1: `0.9 * old + 0.1 * new_bits`.

### 3.2 BWE Feature State

```c
typedef struct {
    float signal_history[OSCE_BWE_HALF_WINDOW_SIZE];    // 160 samples
    float last_spec[2 * OSCE_BWE_MAX_INSTAFREQ_BIN + 2]; // 82 floats (complex spectrum)
} OSCEBWEFeatureState;
```

### 3.3 LACE Processing State

```c
typedef struct {
    float feature_net_conv2_state[LACE_FNET_CONV2_STATE_SIZE];
    float feature_net_gru_state[LACE_COND_DIM];
    AdaCombState cf1_state;     // 1st adaptive comb filter
    AdaCombState cf2_state;     // 2nd adaptive comb filter
    AdaConvState af1_state;     // Adaptive convolution filter
    float preemph_mem;          // Pre-emphasis filter memory (1 sample)
    float deemph_mem;           // De-emphasis filter memory (1 sample)
} LACEState;
```

### 3.4 NoLACE Processing State

```c
typedef struct {
    float feature_net_conv2_state[NOLACE_FNET_CONV2_STATE_SIZE];
    float feature_net_gru_state[NOLACE_COND_DIM];
    float post_cf1_state[NOLACE_COND_DIM];  // Feature transform after cf1
    float post_cf2_state[NOLACE_COND_DIM];  // Feature transform after cf2
    float post_af1_state[NOLACE_COND_DIM];  // Feature transform after af1
    float post_af2_state[NOLACE_COND_DIM];  // Feature transform after af2
    float post_af3_state[NOLACE_COND_DIM];  // Feature transform after af3
    AdaCombState cf1_state, cf2_state;
    AdaConvState af1_state, af2_state, af3_state, af4_state;
    AdaShapeState tdshape1_state, tdshape2_state, tdshape3_state;
    float preemph_mem;
    float deemph_mem;
} NoLACEState;
```

### 3.5 BBWENet State

```c
typedef struct {
    float feature_net_conv1_state[BBWENET_FNET_CONV1_STATE_SIZE];
    float feature_net_conv2_state[BBWENET_FNET_CONV2_STATE_SIZE];
    float feature_net_gru_state[BBWENET_FNET_GRU_STATE_SIZE];
    opus_int16 outbut_buffer[OSCE_BWE_OUTPUT_DELAY];  // 21-sample delay line (note: "outbut" is a typo in C source)
    AdaConvState af1_state, af2_state, af3_state;
    AdaShapeState tdshape1_state, tdshape2_state;
    resamp_state resampler_state[3];                   // 3 channels of resampler state
} BBWENetState;
```

### 3.6 Model Containers

```c
typedef struct {
    int loaded;
    LACE lace;         // LACE model weights + overlap window
    NoLACE nolace;     // NoLACE model weights + overlap window
    BBWENet bbwenet;   // BBWENet model weights + overlap windows (16/32/48 kHz)
} OSCEModel;

typedef union {        // Union — only one method active at a time
    LACEState lace;
    NoLACEState nolace;
} OSCEState;
```

### 3.7 Embedding in SILK Decoder

```c
typedef struct {
    OSCEFeatureState features;
    OSCEState state;           // Union of LACE/NoLACE states
    int method;
} silk_OSCE_struct;            // Field: psDec->osce

typedef struct {
    OSCEBWEFeatureState features;
    OSCEBWEState state;
} silk_OSCE_BWE_struct;        // Field: psDec->osce_bwe
```

**Lifecycle:**
- `osce_load_models()` called once during decoder init.
- `osce_reset()` called on method change or codec mode change.
- `osce_enhance_frame()` called every 20 ms frame.
- State persists across frames in `psDec->osce`.

---

## 4. Algorithm

### 4.1 Feature Extraction (`osce_calculate_features`)

For each 20 ms frame (4 subframes of 80 samples):

**Step 1: Smooth bit count**
```
numbits_smooth = 0.9 * numbits_smooth + 0.1 * num_bits
output: numbits[0] = num_bits, numbits[1] = numbits_smooth
```

**Step 2: Prepare signal buffer**
- Normalize decoded samples: `buffer[k] = xq[k] / 32768.0`
- Prepend 350 samples of signal history.

**Step 3: Per-subframe features (93 dimensions)**

For each subframe `k` (0..3), with `frame` pointing to start of subframe in buffer:

| Feature | Indices | Dim | Computation |
|---------|---------|-----|-------------|
| Clean log-spectrum | 0..63 | 64 | From LPC coefficients (every other subframe) |
| Noisy cepstrum | 64..81 | 18 | DCT of log-magnitude spectrum of signal (every other subframe) |
| Auto-correlation | 82..86 | 5 | Normalized cross-correlation at pitch lag ±2 |
| LTP coefficients | 87..91 | 5 | `LTPCoef_Q14[k*5+i] / 16384.0` |
| Log gain | 92 | 1 | `log(Gains_Q16[k] / 65536.0)` |

**Clean spectrum calculation** (`calculate_log_spectrum_from_lpc`):
1. Create impulse response: `[1, -a_q12[0]/4096, -a_q12[1]/4096, ..., 0, 0, ...]` (320 samples)
2. Compute 320-point FFT, take one-sided magnitude spectrum (161 bins), scale by 320
3. Invert: `spec[i] = 1.0 / (mag[i] + 1e-9)`
4. Apply 64-band triangular filterbank (`center_bins_clean`, `band_weights_clean`)
5. Log and scale: `spec[i] = 0.3 * log(spec[i] + 1e-9)`

**Noisy cepstrum calculation** (`calculate_cepstrum`):
1. Window 320 samples centered at subframe start using `osce_window[]`
2. Compute 320-point FFT, one-sided magnitude spectrum, scale by 320
3. Apply 18-band triangular filterbank (`center_bins_noisy`, `band_weights_noisy`)
4. Log: `spec[i] = log(spec[i] + 1e-9)`
5. Apply DCT-II (orthonormal, 18-point) to get cepstral coefficients

**Auto-correlation** (`calculate_acorr`):
For offsets `k = -2, -1, 0, +1, +2` relative to the pitch lag:
```
acorr[k+2] = Σ(signal[n] * signal[n-lag+k]) / sqrt(Σ(signal[n]²) * Σ(signal[n-lag+k]²) + 1e-9)
```
Computed over 80 samples. This is normalized cross-correlation (a correlation coefficient in [-1, 1]).

**Pitch post-processing** (`pitch_postprocessing`):
- For voiced frames: use actual pitch lag, update `last_lag`.
- For unvoiced frames: use sentinel value 7 (`OSCE_NO_PITCH_VALUE`).
- Hangover mechanism exists but is currently disabled (`OSCE_PITCH_HANGOVER = 0`).

**Step 4: Buffer update**
Copy last 350 samples into `signal_history` for next frame.

### 4.2 LACE Processing (`lace_process_20ms_frame`)

```
Input: 320 float samples (normalized [-1, 1])
Output: 320 float samples (enhanced)
Processing unit: 4 subframes of 80 samples (LACE_FRAME_SIZE)
```

**Pipeline:**
1. **Pre-emphasis**: `x'[n] = x[n] - 0.85 * x[n-1]` (first-order high-pass)
2. **Feature network** (`lace_feature_net`):
   - **Numbits embedding**: sinusoidal positional encoding of log(numbits), 8 dimensions each for raw and smoothed → 16 total
   - **Per-subframe Conv1D**: `[features(93) | pitch_embedding | numbits_embedding(16)]` → Conv1D → tanh → hidden features
   - **Subframe accumulation**: Conv1D across 4 subframes → tanh (uses conv2 state for temporal context)
   - **Transposed convolution**: Dense layer upsampling back to 4 subframes
   - **GRU**: Sequential GRU over 4 subframes → 4 × LACE_COND_DIM conditioning vectors
3. **Adaptive comb filter 1** (`adacomb_process_frame`): Pitch-period-aware comb filtering, conditioned on feature vector
4. **Adaptive comb filter 2**: Same structure, different weights
5. **Adaptive convolution** (`adaconv_process_frame`): General learned convolution, conditioned on features
6. **De-emphasis**: `y[n] = x'[n] + 0.85 * y[n-1]` (inverse of pre-emphasis)

All adaptive stages use overlap-add with a learned overlap window for smooth transitions between subframes.

### 4.3 NoLACE Processing (`nolace_process_20ms_frame`)

NoLACE has the same basic structure as LACE but with a much deeper signal processing pipeline:

**Pipeline:**
1. **Pre-emphasis** (identical to LACE)
2. **Feature network** (`nolace_feature_net`): Same architecture as LACE's feature net with NoLACE-specific dimensions
3. **Adaptive comb filter 1** + **post_cf1** feature transform (Conv1D → tanh on conditioning features)
4. **Adaptive comb filter 2** + **post_cf2** feature transform
5. **Adaptive convolution 1** (af1): 1-channel in, 2-channel out + **post_af1** feature transform
6. **Shape-mix round 1**: AdaShape on channel 2, then AdaConv (af2) mixing 2 channels → 2 channels + **post_af2** feature transform
7. **Shape-mix round 2**: AdaShape on channel 2, then AdaConv (af3) mixing 2 channels → 2 channels + **post_af3** feature transform
8. **Shape-mix round 3**: AdaShape on channel 2, then AdaConv (af4) mixing 2 channels → 1 channel
9. **De-emphasis**

Key difference from LACE: between each processing stage, the conditioning features are updated via a Conv1D (the `post_*` layers), giving the network ability to evolve its conditioning across the processing pipeline.

### 4.4 BBWENet Processing (`bbwenet_process_frames`)

**Pipeline** (operates at 16 kHz input, produces 48 kHz output):

1. **Feature network** (`bbwe_feature_net`):
   - Conv1D → Conv1D → Dense (transposed conv) → GRU
   - Operates per-frame (10 ms frames at 16 kHz), upsamples 1→2 via transposed conv
   
2. **Adaptive convolution 1** (af1): 1 channel in → 3 channels out (at 16 kHz subframe rate)

3. **First upsampling round** (16 kHz → 32 kHz):
   - 2x upsampling on each of 3 channels via `upsamp_2x()` (allpass-based half-band filter)
   - AdaShape on channel 2 (nonlinear time-domain shaping)
   - Valin activation on channel 3: `x *= sin(log(|x| + 1e-6))` (learned nonlinear harmonic generation)

4. **Adaptive convolution 2** (af2): 3 channels in → 3 channels out (at 32 kHz rate)

5. **Second upsampling round** (32 kHz → 48 kHz):
   - 3/2 interpolation on each channel via `interpol_3_2()` (8-tap polyphase FIR)
   - AdaShape on channel 2
   - Valin activation on channel 3

6. **Adaptive convolution 3** (af3): 3 channels in → 1 channel out (final 48 kHz output)

7. **Output delay**: 21-sample delay buffer to align output with causality constraints.

### 4.5 BWE Feature Extraction (`osce_bwe_calculate_features`)

Per 10 ms frame (160 samples at 16 kHz):

1. Build 320-sample buffer: 160 history + 160 new samples
2. Apply `osce_window[]` (320-point analysis window)
3. Compute FFT → complex spectrum
4. **Instantaneous frequency** for bins 0..40:
   - Compute phase difference between current and previous spectrum frame
   - `aux = conj(prev_spec) * curr_spec`, normalize: `instafreq_real = re(aux)/|aux|`, `instafreq_imag = im(aux)/|aux|`
   - Output: 82 dimensions (41 real + 41 imaginary)
5. **Log-mel spectrogram**: magnitude spectrum → 32-band filterbank (`center_bins_bwe`) → log
   - Output: 32 dimensions
6. **Total feature dimension**: 32 + 82 = 114 (`OSCE_BWE_FEATURE_DIM`)

---

## 5. Data Flow

### 5.1 `osce_enhance_frame` Data Flow

```
Inputs:
  xq[320]         opus_int16    Decoded SILK speech (16 kHz, 20 ms)
  psDec            decoder state  LPC_order, nb_subfr, fs_kHz, indices.signalType
  psDecCtrl        decoder ctrl   PredCoef_Q12[][16], LTPCoef_Q14[20], Gains_Q16[4], pitchL[4]
  num_bits         opus_int32     SILK payload size in bits
  model            OSCEModel      Neural network weights

Internal buffers:
  features[4*93]   float         Per-subframe features
  numbits[2]       float         Raw + smoothed bit count
  periods[4]       int           Post-processed pitch lags
  in_buffer[320]   float         Normalized input (÷32768)
  out_buffer[320]  float         Enhanced output

Output:
  xq[320]         opus_int16    Enhanced speech (written back in-place, ×32768, clipped to [-32767, 32767])
```

### 5.2 Buffer Layouts

**Feature vector layout per subframe** (93 = OSCE_FEATURE_DIM):
```
Offset  Length  Content
  0      64     Clean log-spectrum (from LPC, 64 bands)
 64      18     Noisy cepstrum (DCT of log-mel, 18 coefficients)
 82       5     Normalized auto-correlation at pitch lag ±2
 87       5     LTP coefficients (Q14 → float)
 92       1     Log gain
```

**Feature vector layout for 4 subframes**: `features[k * 93 + offset]`

**BWE feature layout per frame** (114 = OSCE_BWE_FEATURE_DIM):
```
Offset  Length  Content
  0      32     Log-mel spectrogram (32 bands)
 32      41     Instantaneous frequency (real part, bins 0..40)
 73      41     Instantaneous frequency (imaginary part, bins 0..40)
```

---

## 6. Numerical Details

### 6.1 Fixed-Point to Float Conversions

| Source | Q-format | Conversion |
|--------|----------|------------|
| `xq[]` (speech samples) | Q15 (opus_int16) | `/ (1U << 15)` = `/ 32768.0` |
| `PredCoef_Q12[]` (LPC) | Q12 | `/ (1U << 12)` = `/ 4096.0` |
| `LTPCoef_Q14[]` (LTP) | Q14 | `/ (1U << 14)` = `/ 16384.0` |
| `Gains_Q16[]` (gain) | Q16 | `/ (1UL << 16)` = `/ 65536.0` |

### 6.2 Float Output to Int16 Conversion

```c
float tmp = 32768.f * out_buffer[i];
if (tmp > 32767.f) tmp = 32767.f;
if (tmp < -32767.f) tmp = -32767.f;
xq[i] = float2int(tmp);
```
- Asymmetric clipping: [-32767, 32767] (not [-32768, 32767]).
- `float2int()` is from `float_cast.h` — uses `lrintf()` or equivalent for round-to-nearest.

### 6.3 Numerical Stability Guards

- `1e-9f` added throughout to prevent division by zero and log(0):
  - `1.f / (mag + 1e-9f)` in `calculate_log_spectrum_from_lpc`
  - `log(spec + 1e-9f)` in spectral calculations
  - `xy / sqrt(xx * yy + 1e-9f)` in auto-correlation
  - `1e-9` added to real part of FFT in BWE `spec[2*k] = ... + 1e-9` (noted as bug in comment)
  - `fabs(x[i]) + 1e-6f` in Valin activation

### 6.4 Pre/De-emphasis

First-order IIR with coefficient `LACE_PREEMPH` / `NOLACE_PREEMPH` (expected to be ~0.85, defined in auto-generated data headers):
```
Pre-emphasis:   y[n] = x[n] - α * x[n-1]     (state: preemph_mem = x[n-1])
De-emphasis:    y[n] = x[n] + α * y[n-1]      (state: deemph_mem = y[n-1])
```
These are exact inverses. The pre-emphasis whitens the signal before neural processing; de-emphasis restores the spectral tilt.

### 6.5 Numbits Embedding

Sinusoidal positional encoding, 8 dimensions:
```c
x = CLIP(log(numbits), log(RANGE_LOW), log(RANGE_HIGH)) - (log(RANGE_HIGH) + log(RANGE_LOW)) / 2
emb[i] = sin(x * SCALE_i - 0.5)
```
The `SCALE_i` constants are model-specific (from auto-generated data headers). This provides a smooth, continuous representation of bit rate that the network can decode.

### 6.6 Spectral Analysis Parameters

- **FFT size**: 320 (`OSCE_SPEC_WINDOW_SIZE`)
- **One-sided spectrum**: 161 bins (`OSCE_SPEC_NUM_FREQS = 320/2 + 1`)
- **FFT scaling**: magnitude spectrum scaled by `OSCE_SPEC_WINDOW_SIZE` (320)
- **Analysis window**: `osce_window[320]` — symmetric raised-cosine-like window (not exactly Hann; appears to be a Vorbis/MDCT window shape). First half rises from ~0.005 to ~1.0, second half mirrors.

### 6.7 Filterbank

Triangular filterbank implemented via `apply_filterbank()`:
```c
for each bin i in [center_bins[b], center_bins[b+1]):
    frac = (center_bins[b+1] - i) / (center_bins[b+1] - center_bins[b])
    output[b]   += weight[b]   * frac       * input[i]
    output[b+1] += weight[b+1] * (1 - frac) * input[i]
```
The `band_weights` provide per-band normalization (reciprocal of effective bandwidth).

Three filterbank configurations:
- **Clean spectrum**: 64 bands, roughly 2.5-bin spacing (high resolution)
- **Noisy spectrum**: 18 bands, Bark-like spacing (wider at high frequencies)
- **BWE spectrum**: 32 bands, 5-bin uniform spacing

### 6.8 Resampling (BBWENet)

**2x upsampling** (`upsamp_2x`): 3-pass allpass structure (half-band filter):
```
Even samples: coefficients hq_2x_even = [0.02664, 0.22867, -0.40364]
Odd samples:  coefficients hq_2x_odd  = [0.10458, 0.39320, -0.15250]
```
Each pass is a first-order allpass: `Y = x - S; X = Y * coeff; tmp = S + X; S = x + X`
Note: third pass uses `1 + coeff` (coefficient > -0.5, so `1 + coeff > 0.5`).

**3/2 interpolation** (`interpol_3_2`): Produces 3 output samples for every 2 input samples using 8-tap polyphase FIR filters at fractional phases 1/24, 17/24, and 9/24. Uses 8-sample delay line (`DELAY_SAMPLES`).

---

## 7. Dependencies

### 7.1 Modules OSCE Calls

| Module | Usage |
|--------|-------|
| `nndsp.h` / `nndsp.c` | `adaconv_process_frame`, `adacomb_process_frame`, `adashape_process_frame`, `compute_overlap_window`, `init_ada*_state` |
| `nnet.h` / `nnet.c` | `compute_generic_conv1d`, `compute_generic_dense`, `compute_generic_gru`, `parse_weights`, `WeightArray`, `LinearLayer` |
| `freq.h` / `freq.c` | `forward_transform` (320-point FFT), `dct` (DCT-II), `NB_BANDS` |
| `kiss_fft.h` | `kiss_fft_cpx` complex type |
| `lace_data.h` | LACE model constants and `LACELayers` struct (auto-generated) |
| `nolace_data.h` | NoLACE model constants and `NOLACELayers` struct (auto-generated) |
| `bbwenet_data.h` | BBWENet model constants and `BBWENETLayers` struct (auto-generated) |
| `silk/structs.h` | `silk_decoder_state`, `silk_decoder_control` |
| `silk/define.h` | `LTP_ORDER` (= 5), `TYPE_VOICED` |
| `float_cast.h` | `float2int()` |
| `mathops.h` | `celt_log()`, `celt_sin()` (used in Valin activation only) |
| `os_support.h` | `OPUS_COPY`, `OPUS_CLEAR` |
| `stack_alloc.h` | Stack allocation macros |

### 7.2 What Calls OSCE

| Caller | Function Called |
|--------|----------------|
| `silk/decode_frame.c` | `osce_enhance_frame()` — after SILK synthesis, before PLC update |
| `src/opus_decoder.c` | `osce_bwe()` — when bandwidth extension is active |
| `silk/dec_API.c` | `osce_reset()` — on decoder reset or mode change |
| Decoder initialization | `osce_load_models()` — once at startup |

---

## 8. Constants and Tables

### 8.1 Configuration Constants (`osce_config.h`)

| Constant | Value | Meaning |
|----------|-------|---------|
| `OSCE_FEATURES_MAX_HISTORY` | 350 | Signal history buffer size (samples) |
| `OSCE_FEATURE_DIM` | 93 | Features per subframe |
| `OSCE_MAX_FEATURE_FRAMES` | 4 | Max subframes per frame |
| `OSCE_CLEAN_SPEC_NUM_BANDS` | 64 | Bands in clean spectrum filterbank |
| `OSCE_NOISY_SPEC_NUM_BANDS` | 18 | Bands in noisy spectrum filterbank |
| `OSCE_NO_PITCH_VALUE` | 7 | Sentinel for unvoiced frames |
| `OSCE_PREEMPH` | 0.85f | Pre-emphasis coefficient (feature extraction) |
| `OSCE_PITCH_HANGOVER` | 0 | Pitch hangover frames (disabled) |
| `OSCE_BWE_MAX_INSTAFREQ_BIN` | 40 | Max FFT bin for instantaneous frequency |
| `OSCE_BWE_HALF_WINDOW_SIZE` | 160 | BWE analysis hop size |
| `OSCE_BWE_WINDOW_SIZE` | 320 | BWE analysis window size |
| `OSCE_BWE_NUM_BANDS` | 32 | BWE filterbank bands |
| `OSCE_BWE_FEATURE_DIM` | 114 | BWE feature dimension |
| `OSCE_BWE_OUTPUT_DELAY` | 21 | BWE output delay in samples (48 kHz) |

### 8.2 Feature Layout Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `OSCE_CLEAN_SPEC_START` | 0 | Offset of clean spectrum in feature vector |
| `OSCE_CLEAN_SPEC_LENGTH` | 64 | Length of clean spectrum |
| `OSCE_NOISY_CEPSTRUM_START` | 64 | Offset of noisy cepstrum |
| `OSCE_NOISY_CEPSTRUM_LENGTH` | 18 | Length of noisy cepstrum |
| `OSCE_ACORR_START` | 82 | Offset of auto-correlation |
| `OSCE_ACORR_LENGTH` | 5 | Length of auto-correlation |
| `OSCE_LTP_START` | 87 | Offset of LTP coefficients |
| `OSCE_LTP_LENGTH` | 5 | Length of LTP coefficients |
| `OSCE_LOG_GAIN_START` | 92 | Offset of log gain |
| `OSCE_LOG_GAIN_LENGTH` | 1 | Length of log gain |

### 8.3 Mode Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `OSCE_MODE_SILK_ONLY` | 1000 | Opus SILK-only mode |
| `OSCE_MODE_HYBRID` | 1001 | Opus Hybrid mode |
| `OSCE_MODE_CELT_ONLY` | 1002 | Opus CELT-only mode |
| `OSCE_MODE_SILK_BBWE` | 1003 | SILK with broadband bandwidth extension |
| `OSCE_METHOD_NONE` | 0 | Enhancement disabled |
| `OSCE_METHOD_LACE` | 1 | LACE enhancement |
| `OSCE_METHOD_NOLACE` | 2 | NoLACE enhancement |

### 8.4 Static Tables (`osce_features.c`)

- **`center_bins_clean[64]`**: FFT bin centers for 64-band clean spectrum filterbank. Range [0, 160]. Roughly 2.5-bin spacing.
- **`center_bins_noisy[18]`**: FFT bin centers for 18-band noisy spectrum filterbank. Range [0, 160]. Bark-like: 4-bin spacing below bin 32, widening to 24-bin at high frequencies.
- **`center_bins_bwe[32]`**: FFT bin centers for 32-band BWE filterbank. Range [0, 160]. Uniform 5-bin spacing.
- **`band_weights_clean[64]`**: Normalization weights for clean filterbank. Values 0.25–0.667.
- **`band_weights_noisy[18]`**: Normalization weights for noisy filterbank. Values 0.042–0.4.
- **`band_weights_bwe[32]`**: Normalization weights for BWE filterbank. Mostly 0.2.
- **`osce_window[320]`**: Symmetric analysis/synthesis window. Rises smoothly from ~0.005 to ~1.0 over 160 samples, mirrors in second half. Used for spectral analysis and cross-fading.

### 8.5 Resampling Filter Coefficients (`osce.c`)

- **`hq_2x_even[3]`**, **`hq_2x_odd[3]`**: Allpass coefficients for 2x upsampling half-band filter.
- **`frac_01_24[8]`**, **`frac_17_24[8]`**, **`frac_09_24[8]`**: 8-tap polyphase FIR coefficients for 3/2 rational resampling at fractional phases 1/24, 17/24, and 9/24.

---

## 9. Edge Cases

### 9.1 Non-20ms / Non-16kHz Frames
```c
if (psDec->fs_kHz != 16 || psDec->nb_subfr != 4) {
    osce_reset(&psDec->osce, psDec->osce.method);
    return;
}
```
OSCE only processes 20 ms frames at 16 kHz. Other configurations trigger a reset and pass-through.

### 9.2 Reset Behavior
After `osce_reset()`, `features.reset = 2`:
- **Frame with reset=2**: output entirely replaced by unenhanced input.
- **Frame with reset=1**: 10 ms cross-fade from unenhanced to enhanced (using first half of `osce_window`).
- **Frame with reset=0**: full enhancement.

This prevents transient artifacts when the enhancement model cold-starts.

### 9.3 Model Not Loaded
If `model->loaded == 0`, method is forced to `OSCE_METHOD_NONE` regardless of `psDec->osce.method`. Input is copied to output unchanged.

### 9.4 BWE Frame Size
`osce_bwe` asserts `xq16_len == 160 || xq16_len == 320` (10 ms or 20 ms only).

### 9.5 BWE Output Delay
The 21-sample delay (`OSCE_BWE_OUTPUT_DELAY`) means the first 21 samples of the first frame come from the delay buffer (initialized to zeros on first call). The last 21 samples of each frame's neural output are saved for the next frame's prefix.

### 9.6 Pitch Hangover Bug
The pitch hangover mechanism is intentionally disabled (`OSCE_PITCH_HANGOVER = 0`) with a comment: "hangover is currently disabled to reflect a bug in the python code." The `OSCE_HANGOVER_BUGFIX` define exists but the logic is effectively a no-op with `TESTBIT = 0`.

### 9.7 BBWENet Output Buffer Typo
The field `outbut_buffer` (line 73 of `osce_structs.h`) is a typo for `output_buffer`. Must be preserved in port for FFI compatibility or renamed if breaking from C API.

### 9.8 BWE Feature State Initialization
`osce_bwe_reset()` initializes `last_spec[2*k]` to `1e-9` for real parts (not zero), matching a quirk in the Python reference.

---

## 10. Porting Notes

### 10.1 Conditional Compilation

The module is heavily conditional:
- `ENABLE_OSCE` — master gate for all OSCE functionality
- `DISABLE_LACE` — exclude LACE model
- `DISABLE_NOLACE` — exclude NoLACE model
- `ENABLE_OSCE_BWE` — enable bandwidth extension
- `DISABLE_BBWENET` — exclude BBWENet model
- `USE_WEIGHTS_FILE` — require external weight file (no compiled-in weights)
- `ENABLE_OSCE_TRAINING_DATA` — debug: dump training data to files
- `OSCE_NUMBITS_BUGFIX` — defined but not visibly used in these files
- `OSCE_HANGOVER_BUGFIX` — enables pitch hangover (currently unused)

**Rust approach:** Use Cargo features for these. The `OSCEState` union and `OSCEModel` struct should be modeled as an enum:
```rust
enum OSCEMethod {
    None,
    Lace(LaceState),
    NoLace(NoLaceState),
}
```

### 10.2 Union Type (`OSCEState`)

```c
typedef union {
    LACEState lace;
    NoLACEState nolace;
} OSCEState;
```
Only one method is active at a time. In Rust, use an `enum` rather than unsafe union. This changes the memory layout — if FFI is needed for the test harness, an intermediate adapter will be required.

### 10.3 In-Place Processing

Several functions operate in-place (same pointer for input and output):
- `adacomb_process_frame` and `adaconv_process_frame` in the signal pipeline
- `adashape_process_frame` on specific channels
- `osce_enhance_frame` overwrites `xq[]`

In Rust, this requires mutable slice references. The `adacomb/adaconv/adashape` functions accept separate `x_out` and `x_in` pointers that may alias — the C code explicitly passes the same pointer. In Rust, these must either use a single `&mut [f32]` parameter or use split-at-mut patterns.

### 10.4 Pointer Arithmetic for Multi-Channel Buffers

NoLACE and BBWENet use multi-channel buffers with manual offset calculation:
```c
x_buffer2 + i_subframe * NOLACE_AF1_OUT_CHANNELS * NOLACE_FRAME_SIZE + NOLACE_FRAME_SIZE
```
This indexes channel 1 of subframe `i_subframe` in an interleaved-by-subframe layout. In Rust, consider a helper type:
```rust
struct MultiChannelBuffer {
    data: Vec<f32>,
    frame_size: usize,
    num_channels: usize,
}
```

### 10.5 Auto-Generated Data Headers

The `lace_data.h`, `nolace_data.h`, and `bbwenet_data.h` files are generated by the training pipeline and define:
- All model-specific constants (`*_FRAME_SIZE`, `*_COND_DIM`, `*_KERNEL_SIZE`, etc.)
- `*Layers` structs containing `LinearLayer` pointers
- `init_*layers()` functions
- Compiled-in weight arrays

These must either be:
1. Ported to Rust `const` arrays and structs, or
2. Loaded at runtime from a weight file (the `USE_WEIGHTS_FILE` path)

### 10.6 Stack Allocation

Large buffers are stack-allocated in C:
- `float x_buffer1[3 * 3 * 4 * 3 * BBWENET_FRAME_SIZE16]` in `bbwenet_process_frames`
- `float buffer[OSCE_FEATURES_MAX_HISTORY + OSCE_MAX_FEATURE_FRAMES * 80]` in feature calculation
- `kiss_fft_cpx buffer[320]` in FFT functions

In Rust, consider heap allocation for the larger buffers (BBWENet buffers can be several KB) or use the processing state struct to hold reusable scratch space.

### 10.7 OPUS_COPY / OPUS_CLEAR

- `OPUS_COPY(dst, src, n)` → `dst[..n].copy_from_slice(&src[..n])` (or `ptr::copy_nonoverlapping` equivalent)
- `OPUS_CLEAR(ptr, 1)` → zero-initialize the struct (in Rust: `*state = Default::default()` or `mem::zeroed()`)

### 10.8 CLIP Macro

```c
#define CLIP(a, min, max) (((a) < (min) ? (min) : (a)) > (max) ? (max) : (a))
```
In Rust: `a.clamp(min, max)` (available on `f32`).

### 10.9 IMAX Macro

Used for compile-time buffer sizing:
```c
float input_buffer[IMAX(4 * IMAX(LACE_COND_DIM, LACE_HIDDEN_FEATURE_DIM), ...)];
```
In Rust, use `const fn max(a: usize, b: usize) -> usize` for const-context max.

### 10.10 `float2int`

`float2int()` from `float_cast.h` performs round-to-nearest-even (banker's rounding) via `lrintf()`. In Rust: `f32::round()` uses round-half-away-from-zero. For bit-exact matching, use `libm::rintf()` or manually implement round-to-nearest-even:
```rust
fn float2int(x: f32) -> i32 {
    // Must match lrintf() behavior exactly
    x.round_ties_even() as i32  // nightly, or use libm
}
```
**This is a critical bit-exactness concern.**

### 10.11 Log/Sin/Sqrt Precision

The C code uses:
- `log()` (double precision in `calculate_cepstrum`, `calculate_log_spectrum_from_lpc`)
- `logf()` implied by `log()` on float arguments in some places
- `sqrt()` (double precision)
- `celt_log()` and `celt_sin()` (CELT fixed-point approximations, used only in Valin activation)
- `sin()` for numbits embedding

For bit-exactness, the Rust port must use the same precision. Pay attention to implicit float→double promotion in the C code.

### 10.12 `calculate_log_spectrum_from_lpc` Buffer Aliasing

```c
mag_spec_320_onesided(buffer, buffer);  // in-place FFT
```
The function writes 161 floats into the same 320-element buffer it reads from. The FFT implementation (`forward_transform`) must support this. In the Rust port, verify that the FFT library handles in-place correctly or use separate buffers.

### 10.13 `calculate_cepstrum` Internal Aliasing

```c
float buffer[320];
float *spec = &buffer[161 + 3];  // spec starts at buffer[164]
```
The filterbank output (`spec`, 18 floats at offset 164) overlaps the buffer tail. After the FFT writes 161 floats, positions 161..163 are unused padding, and 164..181 hold the filterbank output. This doesn't alias the FFT output because `apply_filterbank` asserts `x_in != x_out`. In Rust, this can be modeled with split borrows.

### 10.14 The `reset` Counter Pattern

The reset counter (`features.reset`) counts down: `2 → 1 → 0`. In `osce_enhance_frame`:
- At `reset > 1`: replace output with input
- At `reset == 1`: cross-fade, then set to 0
- At `reset == 0`: normal operation

Note that the neural network still runs even during reset (only the output is suppressed/blended). This warms up the internal state.

### 10.15 Method Selection

`psDec->osce.method` is set by `osce_reset()`. The decoder selects method based on complexity:
- complexity >= 6: LACE (if available)
- complexity >= 7: NoLACE (if available, overrides LACE)

The default method (`OSCE_DEFAULT_METHOD`) prefers NoLACE > LACE > None.

### 10.16 External Weight Arrays

```c
extern const WeightArray lacelayers_arrays[];
extern const WeightArray nolacelayers_arrays[];
extern const WeightArray bbwenetlayers_arrays[];
```
These are defined in auto-generated `.c` files compiled into the binary. In Rust, these would be `static` arrays or loaded from embedded binary data via `include_bytes!`.
