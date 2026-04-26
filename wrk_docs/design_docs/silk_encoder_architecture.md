I now have all the source material needed. Here is the complete architecture documentation.

---

# SILK Encoder Module — Architecture Documentation

## 1. Purpose

The SILK encoder is the speech coding component of the Opus codec. It encodes narrowband (8 kHz), mediumband (12 kHz), and wideband (16 kHz) speech signals into a compressed bitstream using Linear Predictive Coding (LPC) with long-term prediction (LTP), noise shaping quantization (NSQ), and entropy coding via the range coder. The encoder supports mono and stereo operation, variable bitrate (VBR) and constant bitrate (CBR) modes, in-band forward error correction (FEC/LBRR), and discontinuous transmission (DTX).

Within the Opus pipeline, SILK handles the lower-frequency speech content. The Opus encoder selects SILK-only mode for bitrates below ~12 kbps, or SILK+CELT hybrid mode for bitrates up to ~64 kbps, where SILK encodes the lower band and CELT the upper band.

---

## 2. Public API

### 2.1 `silk_Get_Encoder_Size`

```c
opus_int silk_Get_Encoder_Size(
    opus_int *encSizeBytes,   // O  Required buffer size in bytes
    opus_int  channels        // I  Number of channels (1 or 2)
);
```

Returns the byte size of the `silk_encoder` struct. For mono, subtracts one `silk_encoder_state_Fxx` (the second channel state is not needed). Used by the Opus layer to allocate the encoder.

### 2.2 `silk_InitEncoder`

```c
opus_int silk_InitEncoder(
    void                  *encState,   // I/O  Opaque encoder state
    int                    channels,   // I    Number of channels
    int                    arch,       // I    CPU architecture flags
    silk_EncControlStruct *encStatus   // O    Initial status readback
);
```

Zeros the entire encoder, calls `silk_init_encoder()` for each channel (which initializes the VAD and sets HP filter smoothing to `VARIABLE_HP_MIN_CUTOFF_HZ`), sets `first_frame_after_reset = 1`, then calls `silk_QueryEncoder` to populate `encStatus`.

### 2.3 `silk_Encode`

```c
opus_int silk_Encode(
    void                  *encState,    // I/O  Encoder state
    silk_EncControlStruct *encControl,  // I    Control parameters
    const opus_res        *samplesIn,   // I    Input PCM (interleaved if stereo)
    opus_int               nSamplesIn,  // I    Number of input samples
    ec_enc                *psRangeEnc,  // I/O  Range coder state
    opus_int32            *nBytesOut,   // I/O  Max bytes in / actual bytes out
    const opus_int         prefillFlag, // I    Prefill mode (0=normal, 1=reset, 2=save LP)
    opus_int               activity     // I    Opus-level VAD decision
);
```

This is the main encoding entry point. It:

1. Validates control input via `check_control_input()`
2. Handles mono↔stereo transitions
3. Calls `silk_control_encoder()` per channel to configure sample rate, complexity, LBRR
4. Resamples input from API rate to internal rate, buffers into `inputBuf`
5. When a full frame is buffered:
   - Encodes LBRR data from the *previous* packet at the head of the bitstream
   - Applies `silk_HP_variable_cutoff()` adaptive high-pass filter
   - Computes per-frame target bitrate with bit reservoir management
   - For stereo: calls `silk_stereo_LR_to_MS()`, encodes stereo prediction indices
   - Calls `silk_encode_do_VAD_Fxx()` (VAD + noise shaping analysis)
   - Calls `silk_encode_frame_Fxx()` per channel (the actual frame encoder)
   - Patches VAD/FEC flags into the bitstream header
6. Updates bandwidth-switch readiness and returns

**Return value**: `SILK_NO_ERROR` (0) on success, or a `SILK_ENC_*` error code.

### 2.4 `silk_EncControlStruct` (input to `silk_Encode`)

| Field | Type | Description |
|-------|------|-------------|
| `API_sampleRate` | `opus_int32` | Input sample rate (8000–48000 Hz) |
| `maxInternalSampleRate` | `opus_int32` | Max internal rate (8000/12000/16000) |
| `minInternalSampleRate` | `opus_int32` | Min internal rate |
| `desiredInternalSampleRate` | `opus_int32` | Desired internal rate |
| `payloadSize_ms` | `opus_int` | Frame size: 10, 20, 40, or 60 ms |
| `bitRate` | `opus_int32` | Target bitrate (bps) |
| `packetLossPercentage` | `opus_int` | Expected loss rate (0–100) |
| `complexity` | `opus_int` | Encoding complexity (0–10) |
| `useInBandFEC` | `opus_int` | Enable LBRR FEC (0/1) |
| `useDTX` | `opus_int` | Enable DTX (0/1) |
| `useCBR` | `opus_int` | Enable CBR (0/1) |
| `nChannelsAPI` | `opus_int32` | API channel count (1–2) |
| `nChannelsInternal` | `opus_int32` | Internal channel count (1–2) |
| `maxBits` | `opus_int` | Max bits for this packet |
| `toMono` | `opus_int` | Transition to mono flag |
| `opusCanSwitch` | `opus_int` | Opus-level bandwidth switch permission |
| `reducedDependency` | `opus_int` | Force independent coding |

---

## 3. Internal State

### 3.1 `silk_encoder` (Top-level Super Struct)

```c
typedef struct {
    stereo_enc_state        sStereo;                     // Stereo encoding state
    opus_int32              nBitsUsedLBRR;                // EMA of LBRR bit usage
    opus_int32              nBitsExceeded;                // Bit reservoir (excess bits)
    opus_int                nChannelsAPI;                 // API channel count
    opus_int                nChannelsInternal;            // Internal channel count
    opus_int                nPrevChannelsInternal;        // Previous internal channels
    opus_int                timeSinceSwitchAllowed_ms;    // BW switch cooldown timer
    opus_int                allowBandwidthSwitch;         // BW switch allowed flag
    opus_int                prev_decode_only_middle;      // Previous mid-only flag
    silk_encoder_state_FIX  state_Fxx[ENCODER_NUM_CHANNELS]; // Per-channel states
} silk_encoder;
```

For mono encoding, only `state_Fxx[0]` is used; `silk_Get_Encoder_Size` subtracts the second element.

### 3.2 `silk_encoder_state_FIX` (Per-Channel State, Fixed-Point)

```c
typedef struct {
    silk_encoder_state      sCmn;       // Common encoder state (~40+ fields)
    silk_shape_state_FIX    sShape;     // Noise shaping analysis smoothers
    opus_int16              x_buf[2 * MAX_FRAME_LENGTH + LA_SHAPE_MAX]; // Pitch/shape analysis buffer
    opus_int                LTPCorr_Q15; // Normalized LTP correlation
    opus_int32              resNrgSmth;  // Smoothed residual energy
} silk_encoder_state_FIX;
```

### 3.3 `silk_encoder_state` (Common Encoder State — `sCmn`)

This is the largest struct (~80 fields). Key groups:

**Filter states:**
- `In_HP_State[2]` — Input high-pass filter state
- `variable_HP_smth1_Q15`, `variable_HP_smth2_Q15` — Adaptive HP cutoff smoothers
- `sLP` (`silk_LP_state`) — Variable low-pass for bandwidth transitions
- `sNSQ` (`silk_nsq_state`) — Noise shaping quantizer state
- `sVAD` (`silk_VAD_state`) — Voice activity detector state
- `resampler_state` — Input resampler state

**Configuration:**
- `fs_kHz` — Internal sampling rate (8/12/16)
- `nb_subfr` — Subframes per frame (2 for 10ms, 4 for 20ms)
- `frame_length`, `subfr_length` — In samples at internal rate
- `predictLPCOrder` — LPC order (10 for NB/MB, 16 for WB)
- `Complexity` — 0–10
- `nStatesDelayedDecision` — NSQ delayed-decision states (1–4)
- `shapingLPCOrder` — Noise shaping filter order (12–24)
- `warping_Q16` — Frequency warping for noise shaping

**Codec parameters (per-frame results):**
- `indices` (`SideInfoIndices`) — All quantization indices for current frame
- `pulses[MAX_FRAME_LENGTH]` — Quantized excitation pulses
- `prev_NLSFq_Q15[MAX_LPC_ORDER]` — Previous quantized NLSFs

**LBRR (Forward Error Correction):**
- `LBRR_enabled`, `LBRR_GainIncreases`
- `indices_LBRR[MAX_FRAMES_PER_PACKET]` — LBRR indices per frame
- `pulses_LBRR[MAX_FRAMES_PER_PACKET][MAX_FRAME_LENGTH]`
- `LBRR_flags[MAX_FRAMES_PER_PACKET]`, `LBRR_flag`

**Buffering:**
- `inputBuf[MAX_FRAME_LENGTH + 2]` — Input buffer (extra 2 for stereo overlap)
- `inputBufIx` — Current write position in input buffer

### 3.4 `silk_nsq_state` (Noise Shaping Quantizer State)

```c
typedef struct {
    opus_int16  xq[2 * MAX_FRAME_LENGTH];           // Reconstructed signal history
    opus_int32  sLTP_shp_Q14[2 * MAX_FRAME_LENGTH]; // LTP noise shaping state
    opus_int32  sLPC_Q14[MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH]; // Short-term predictor state
    opus_int32  sAR2_Q14[MAX_SHAPE_LPC_ORDER];      // Noise shaping AR filter state
    opus_int32  sLF_AR_shp_Q14;                      // Low-frequency AR shaping state
    opus_int32  sDiff_shp_Q14;                       // Difference shaping state
    opus_int    lagPrev;                              // Previous pitch lag
    opus_int    sLTP_buf_idx;                         // Write index for LTP prediction buffer
    opus_int    sLTP_shp_buf_idx;                     // Write index for LTP shaping buffer
    opus_int32  rand_seed;                            // Dither PRNG state
    opus_int32  prev_gain_Q16;                        // Previous subframe gain
    opus_int    rewhite_flag;                         // Signals LTP state needs rewhitening
} silk_nsq_state;
```

**Lifecycle**: The `xq` and `sLTP_shp_Q14` arrays are circular buffers of length `2 * MAX_FRAME_LENGTH`. After each frame, the tail is shifted to the head via `silk_memmove`. The `sLPC_Q14` buffer is similarly shifted after each subframe.

### 3.5 `SideInfoIndices` (Per-Frame Quantization Indices)

```c
typedef struct {
    opus_int8   GainsIndices[MAX_NB_SUBFR];    // Gain indices per subframe
    opus_int8   LTPIndex[MAX_NB_SUBFR];        // LTP codebook indices per subframe
    opus_int8   NLSFIndices[MAX_LPC_ORDER+1];  // NLSF VQ indices [stage1, residuals...]
    opus_int16  lagIndex;                       // Pitch lag index
    opus_int8   contourIndex;                   // Pitch contour (fine lag adjustment)
    opus_int8   signalType;                     // 0=inactive, 1=unvoiced, 2=voiced
    opus_int8   quantOffsetType;                // Quantization offset (0 or 1)
    opus_int8   NLSFInterpCoef_Q2;              // NLSF interpolation factor (0..4)
    opus_int8   PERIndex;                       // LTP periodicity codebook (0/1/2)
    opus_int8   LTP_scaleIndex;                 // LTP state scaling (0/1/2)
    opus_int8   Seed;                           // Dither seed (0–3)
} SideInfoIndices;
```

### 3.6 `silk_VAD_state`

```c
typedef struct {
    opus_int32  AnaState[2];           // Analysis filterbank state: 0-8 kHz
    opus_int32  AnaState1[2];          // Analysis filterbank state: 0-4 kHz
    opus_int32  AnaState2[2];          // Analysis filterbank state: 0-2 kHz
    opus_int32  XnrgSubfr[VAD_N_BANDS]; // Subframe energies (look-ahead)
    opus_int32  NrgRatioSmth_Q8[VAD_N_BANDS]; // Smoothed energy-to-noise ratio
    opus_int16  HPstate;               // HP differentiator state
    opus_int32  NL[VAD_N_BANDS];       // Noise levels per band
    opus_int32  inv_NL[VAD_N_BANDS];   // Inverse noise levels (for stable smoothing)
    opus_int32  NoiseLevelBias[VAD_N_BANDS]; // Noise floor bias (pink noise model)
    opus_int32  counter;               // Frame counter (for initial fast adaptation)
} silk_VAD_state;
```

### 3.7 `stereo_enc_state`

```c
typedef struct {
    opus_int16  pred_prev_Q13[2];      // Previous stereo predictors (LP, HP)
    opus_int16  sMid[2];               // Mid signal overlap buffer
    opus_int16  sSide[2];              // Side signal overlap buffer
    opus_int32  mid_side_amp_Q0[4];    // Smoothed amplitudes [mid_LP, res_LP, mid_HP, res_HP]
    opus_int16  smth_width_Q14;        // Smoothed stereo width
    opus_int16  width_prev_Q14;        // Previous frame width
    opus_int16  silent_side_len;       // Samples since last side channel coded
    opus_int8   predIx[MAX_FRAMES_PER_PACKET][2][3]; // Stereo prediction quantization indices
    opus_int8   mid_only_flags[MAX_FRAMES_PER_PACKET]; // Mid-only flags per frame
} stereo_enc_state;
```

---

## 4. Algorithm

### 4.1 Top-Level Encoding Flow (`silk_Encode`)

```
Input PCM (API rate) ──► Resample to internal rate ──► Buffer in inputBuf
                                                              │
                          ┌───────────────────────────────────┘
                          ▼ (when full frame buffered)
                   ┌──────────────┐
                   │ Encode LBRR  │ (previous packet's FEC data)
                   │ from prev pkt│
                   └──────┬───────┘
                          ▼
                   ┌──────────────┐
                   │HP var cutoff │ silk_HP_variable_cutoff()
                   └──────┬───────┘
                          ▼
                   ┌──────────────┐
                   │ Compute      │ target bitrate with reservoir
                   │ TargetRate   │
                   └──────┬───────┘
                          ▼
                   ┌──────────────┐
           stereo? │ L/R → M/S    │ silk_stereo_LR_to_MS()
                   │ encode pred  │ silk_stereo_encode_pred()
                   └──────┬───────┘
                          ▼
                   ┌──────────────┐
                   │ VAD + Noise  │ silk_encode_do_VAD_Fxx()
                   │ Shape Anal.  │   → silk_VAD_GetSA_Q8()
                   └──────┬───────┘   → silk_noise_shape_analysis_FIX()
                          ▼
                   ┌──────────────┐
                   │ SNR Control  │ silk_control_SNR()
                   └──────┬───────┘
                          ▼
                   ┌──────────────┐
                   │ Encode Frame │ silk_encode_frame_Fxx()
                   │  per channel │   → find_pitch_lags
                   └──────┬───────┘   → find_pred_coefs
                          │            → process_NLSFs
                          │            → NSQ or NSQ_del_dec
                          │            → encode_indices
                          │            → encode_pulses
                          ▼
                   ┌──────────────┐
                   │ Patch flags  │ VAD + LBRR flags at bitstream start
                   │ DTX check    │
                   └──────────────┘
```

### 4.2 Per-Frame Encoding (`silk_encode_frame_Fxx`)

The per-frame encoder (in `fixed/encode_frame_FIX.c`, not in scope but orchestrates the files we're documenting) calls:

1. **Pitch analysis** → `silk_find_pitch_lags_FIX()` — estimates pitch lag via autocorrelation
2. **Prediction coefficients** → `silk_find_pred_coefs_FIX()`:
   - LPC analysis via Burg's method
   - LPC → NLSF → quantize → NLSF → LPC (via `silk_process_NLSFs`)
   - LTP gain quantization (via `silk_quant_LTP_gains`)
3. **Noise shape quantization** → `silk_NSQ()` or `silk_NSQ_del_dec()`:
   - Quantizes the excitation signal using noise shaping feedback
   - Produces `pulses[]` (quantized excitation) and `xq[]` (reconstructed signal)
4. **Bitstream encoding**:
   - `silk_encode_indices()` — entropy-codes all side information
   - `silk_encode_pulses()` — shell-codes the excitation pulses

### 4.3 Voice Activity Detection (`silk_VAD_GetSA_Q8`)

1. **Band splitting**: Three cascaded calls to `silk_ana_filt_bank_1()` split the input into 4 subbands: 0–1 kHz, 1–2 kHz, 2–4 kHz, 4–8 kHz
2. **HP filtering**: A first-order differentiator removes DC from the lowest band
3. **Energy computation**: Per-band energy is accumulated across sub-frames, with the last sub-frame weighted by 0.5 (look-ahead)
4. **Noise estimation** (`silk_VAD_GetNoiseLevels`): Adaptive smoothing of inverse noise levels with faster initial convergence (first 20 seconds)
5. **SNR estimation**: Energy-to-noise ratios converted to dB via `silk_lin2log`
6. **Speech probability**: Root-mean-square SNR → sigmoid function → `speech_activity_Q8`
7. **Frequency tilt**: Weighted combination of per-band SNR gives `input_tilt_Q15`
8. **Input quality**: Per-band smoothed SNR → sigmoid → `input_quality_bands_Q15[4]`

### 4.4 Analysis Filter Bank (`silk_ana_filt_bank_1`)

Two-band decimating filter using first-order allpass sections:

```
Coefficients (Q15-like, stored as int16):
  A_fb1_20 = 5394 << 1  = 10788
  A_fb1_21 = -24290      (= 20623 << 1, overflowed to negative)

Internal state in Q10.
For each pair of input samples:
  Even sample → allpass with A_fb1_21
  Odd sample  → allpass with A_fb1_20
  outL = (out_even + out_odd + rounding) >> 11   (low band, decimated)
  outH = (out_odd  - out_even + rounding) >> 11  (high band, decimated)
```

### 4.5 NLSF Processing (`silk_process_NLSFs`)

1. **Compute mu** (rate-distortion tradeoff):
   ```
   NLSF_mu_Q20 = 0.003 - 0.001 * speech_activity_Q8
   // Scaled by 1.5 for 10ms frames
   ```
2. **NLSF weights**: Laroia method (`silk_NLSF_VQ_weights_laroia`)
3. **Interpolation**: If enabled and `NLSFInterpCoef_Q2 < 4`, interpolates between `prev_NLSFq_Q15` and current NLSFs. The first half uses interpolated coefficients, the second half uses the new coefficients.
4. **Encoding**: `silk_NLSF_encode()` performs two-stage VQ:
   - Stage 1: Find best codebook vectors, keep `nSurvivors` candidates
   - Stage 2: `silk_NLSF_del_dec_quant()` trellis-quantizes the residual
   - Selects winner by minimum rate-distortion (Q25)
5. **Reconstruction**: `silk_NLSF2A()` converts quantized NLSFs back to LPC coefficients

### 4.6 A2NLSF Conversion (`silk_A2NLSF`)

Converts LPC coefficients `a_Q16[d]` to Normalized Line Spectral Frequencies `NLSF[d]` (Q15, range 0..32767):

1. **Initialization** (`silk_A2NLSF_init`):
   - Splits the polynomial into even (P) and odd (Q) parts
   - Divides out the known roots at z=1 and z=-1
   - Transforms from cos(nf) domain to cos(f)^n domain
2. **Root finding**: Scans `silk_LSFCosTab_FIX_Q12` table (129 entries spanning 0..π), detecting sign changes
3. **Refinement**: Binary bisection (`BIN_DIV_STEPS_A2NLSF_FIX = 3` iterations) plus linear interpolation for fractional precision
4. **Bandwidth expansion**: If not all roots are found, applies `silk_bwexpander_32()` and retries (up to `MAX_ITERATIONS_A2NLSF_FIX = 16`)

### 4.7 NLSF Delayed-Decision Quantization (`silk_NLSF_del_dec_quant`)

Trellis quantizer for NLSF residuals after first-stage VQ:

1. Processes coefficients in **reverse order** (from `order-1` down to 0)
2. Maintains up to `NLSF_QUANT_DEL_DEC_STATES` (typically 32) parallel states
3. For each coefficient: computes two candidates (floor, ceil), adds rate and distortion costs
4. State doubling: While `nStates < NLSF_QUANT_DEL_DEC_STATES/2`, doubles states by forking
5. State pruning: When full, pairwise compares winners/losers and swaps if a loser beats a winner
6. Final: selects minimum RD across all 2×nStates candidates, backtracks to get indices

### 4.8 Noise Shaping Quantization — Standard (`silk_NSQ_c`)

Per-subframe processing:

1. **State scaling** (`silk_nsq_scale_states`):
   - Scales input by `1/Gain_Q16` to get `x_sc_Q10`
   - Re-whitens LTP state when prediction coefficients change
   - Adjusts all internal states when gain changes between subframes
2. **Sample-by-sample quantization** (`silk_noise_shape_quantizer`):
   ```
   For each sample i:
     Generate dither from rand_seed (PRNG)
     Short-term prediction: LPC_pred_Q10 = Σ a_Q12[k] * sLPC_Q14[i-1-k]
     Long-term prediction:  LTP_pred_Q13 = Σ b_Q14[k] * sLTP_Q15[i-lag+2-k]  (voiced only)
     Noise shaping feedback:
       n_AR_Q12 = Σ AR_shp_Q13[k] * sAR2_Q14[k] + Tilt * sLF_AR_shp
       n_LF_Q12 = LF_shp * sLTP_shp[-1] + LF_shp_upper * sLF_AR_shp
       n_LTP_Q13 = HarmShapeFIR * (shp[-1] + shp[-3]) + shp[-2]  (if lag > 0)
     Combine: residual = x_sc - (LPC_pred + LTP_pred - noise_shape)
     Flip sign based on dither
     Two-candidate RD: q1 = floor, q2 = ceil; pick lower rate-distortion
     Update reconstructed signal and all filter states
   ```

### 4.9 Noise Shaping Quantization — Delayed Decision (`silk_NSQ_del_dec_c`)

Enhanced version maintaining `nStatesDelayedDecision` (1–4) parallel paths:

```c
typedef struct {
    opus_int32 sLPC_Q14[MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH];
    opus_int32 RandState[DECISION_DELAY];    // Dither per delayed sample
    opus_int32 Q_Q10[DECISION_DELAY];        // Quantized values (ring buffer)
    opus_int32 Xq_Q14[DECISION_DELAY];       // Reconstructed (ring buffer)
    opus_int32 Pred_Q15[DECISION_DELAY];     // Predictions
    opus_int32 Shape_Q14[DECISION_DELAY];    // Shaping state
    opus_int32 sAR2_Q14[MAX_SHAPE_LPC_ORDER];
    opus_int32 LF_AR_Q14, Diff_Q14;
    opus_int32 Seed, SeedInit;
    opus_int32 RD_Q10;                       // Cumulative rate-distortion
} NSQ_del_dec_struct;
```

Decision delay is `min(DECISION_DELAY, subfr_length)`, further limited to `pitch_lag - LTP_ORDER/2 - 1` for voiced frames. At the end of each subframe, the winning state is selected and its delayed outputs are committed.

### 4.10 LTP Gain Quantization (`silk_quant_LTP_gains`)

Iterates over 3 codebooks (`silk_LTP_vq_ptrs_Q7[0..2]`) with increasing size/precision:
- Codebook 0: 8 vectors (low rate)
- Codebook 1: 16 vectors (medium)
- Codebook 2: 32 vectors (high rate)

For each codebook and each subframe:
1. Computes `max_gain_Q7` from cumulative log gain budget (`MAX_SUM_LOG_GAIN_DB`)
2. Calls `silk_VQ_WMat_EC()` for weighted matrix VQ
3. Accumulates total rate-distortion across subframes
4. Selects codebook with minimum total RD

Output: `B_Q14[nb_subfr * LTP_ORDER]` (LTP coefficients shifted from Q7 to Q14), codebook indices, periodicity index.

### 4.11 Matrix-Weighted Vector Quantization (`silk_VQ_WMat_EC`)

Hard-coded for 5-element vectors (LTP_ORDER = 5):

```
For each codebook vector cb[]:
  Quantization error = 1 - 2*xX*cb + cb'*XX*cb   (Q15)
  Rate = cl_Q5[k]  (code length)
  Penalty for gain exceeding max_gain_Q7
  Total RD = subfr_len * lin2log(error + penalty) + rate/2
  Select minimum
```

Uses the upper-triangular correlation matrix `XX_Q17[5×5]` laid out as flat array (25 elements).

### 4.12 Stereo Encoding (`silk_stereo_LR_to_MS`)

1. **L/R to M/S**: `mid = (L+R)/2`, `side = (L-R)/2` (with saturation)
2. **2-sample overlap buffering**: `sMid[2]`, `sSide[2]` maintain continuity
3. **LP/HP splitting**: Simple 3-tap FIR: `LP = (x[n] + 2*x[n+1] + x[n+2]) / 4`, `HP = x[n+1] - LP`
4. **Predictor estimation**: `silk_stereo_find_predictor()` computes least-squares predictor for LP and HP bands separately
5. **Bitrate allocation**: Mid gets `8/(13+3*frac)` of total rate; side gets the rest. If mid rate falls below `min_mid_rate_bps`, stereo width is reduced.
6. **Panned-mono detection**: If bitrate is very low or side energy is negligible, `mid_only_flag = 1`
7. **Predictor interpolation**: During `STEREO_INTERP_LEN_MS * fs_kHz` samples, interpolates from previous to current predictors and width
8. **Side signal reconstruction**: `side_out = width * side - pred0 * LP(mid) - pred1 * HP(mid)`

### 4.13 Bitstream Encoding

#### `silk_encode_indices`

Entropy-codes all side information using `ec_enc_icdf()`:

| Parameter | Coding method |
|-----------|---------------|
| Signal type + quantizer offset | Joint 6-symbol iCDF (VAD or no-VAD variant) |
| Gain, first subframe | Independent: MSB (8 levels by signal type) + 3 LSBs; or conditional delta |
| Gain, subsequent subframes | Delta coding |
| NLSF, first stage | Per-signal-type iCDF |
| NLSF, residuals | Per-coefficient iCDF with overflow extension |
| NLSF interpolation factor | 5-level uniform (only for 4-subframe) |
| Pitch lag | Absolute (high bits + low bits) or delta from previous |
| Pitch contour | Contour iCDF (varies by rate/subframes) |
| LTP periodicity index | 3-level iCDF |
| LTP codebook indices | Per-periodicity iCDF |
| LTP scale | 3-level (independent coding only) |
| Seed | 4-level uniform |

#### `silk_encode_pulses`

Encodes quantized excitation pulses using shell coding:

1. **Shell block preparation**: Frame split into `SHELL_CODEC_FRAME_LENGTH` (16-sample) blocks
2. **Magnitude reduction**: If pulse magnitudes exceed `silk_max_pulses_table` limits, right-shift and record shift count
3. **Rate level selection**: Tests `N_RATE_LEVELS - 1` rate levels, picks minimum total bits
4. **Hierarchical shell encoding**: Binary tree decomposition of pulse counts (16→8→4→2→1)
5. **LSB encoding**: For right-shifted blocks, encode the stripped LSBs
6. **Sign encoding**: `silk_encode_signs()` entropy-codes pulse signs, conditioned on signal type and quantizer offset

---

## 5. Data Flow

### Input Path

```
API PCM (int16, API rate, interleaved stereo)
  │
  ├─► De-interleave (stereo)
  │
  ├─► Resample (silk_resampler) → internal rate (8/12/16 kHz)
  │
  └─► Buffer in inputBuf[MAX_FRAME_LENGTH + 2]
       (offset by 2 for stereo overlap with previous frame)
```

### Internal Processing Buffers

| Buffer | Q-format | Size | Purpose |
|--------|----------|------|---------|
| `inputBuf` | Q0 (int16) | `frame_length + 2` | Input + 2-sample stereo overlap |
| `x_buf` | Q0 (int16) | `2*MAX_FRAME_LENGTH + LA_SHAPE_MAX` | Pitch/shape analysis window |
| `x_sc_Q10` | Q10 | `subfr_length` | Input scaled by 1/gain |
| `sLTP` | Q0 (int16) | `ltp_mem_length + frame_length` | Re-whitened LTP state |
| `sLTP_Q15` | Q15 | `ltp_mem_length + frame_length` | Scaled LTP prediction state |
| `NSQ.xq` | Q0 (int16) | `2*MAX_FRAME_LENGTH` | Reconstructed signal ring buffer |
| `NSQ.sLTP_shp_Q14` | Q14 | `2*MAX_FRAME_LENGTH` | LTP noise shaping ring buffer |
| `pulses` | Q0 (int8) | `frame_length` | Quantized excitation pulses |

### Output

The encoder writes directly to the range coder (`ec_enc`). The bitstream structure per packet is:

```
[VAD+LBRR flags] [LBRR data...] [frame0 indices] [frame0 pulses] [frame1...] ...
```

---

## 6. Numerical Details

### 6.1 Q-Format Summary

| Signal | Format | Range/Precision |
|--------|--------|-----------------|
| Input PCM | Q0 int16 | -32768..32767 |
| LPC coefficients | Q12 int16 | ±8.0 |
| LTP coefficients | Q14 int16 | ±2.0 |
| NLSFs | Q15 int16 | 0..32767 (0..π) |
| Noise shaping AR | Q13 int16 | ±4.0 |
| Gains | Q16 int32 | ±32768.0 |
| Inverse gain | Q31 int32 | full precision |
| NSQ internal state | Q14 int32 | ±131072.0 |
| Excitation (internal) | Q10 int32 | |
| Quantized pulses | Q0 int8 | -128..127 (typically -31..30) |
| SNR table entries | Q7 (via ×21) | 0..5355 |
| Stereo predictors | Q13 int32 | ±4.0 |
| Stereo width | Q14 int16 | 0..1.0 |
| Speech activity | Q8 int | 0..255 |
| NLSF weights | Q2/Q5 int16 | |
| Rate-distortion | Q25 int32 | |

### 6.2 Overflow Handling

The C code intentionally uses wrapping arithmetic in several places:

- **`silk_SMLABB_ovflw`** in `silk_LPC_analysis_filter`: Allows intermediate overflow during LPC filtering, relying on two wraps canceling. Comment: "The rare cases where the result wraps around can only be triggered by invalid streams."
- **`silk_SUB32_ovflw`**, **`silk_ADD32_ovflw`** in NSQ: The noise shaping feedback path allows intermediate overflow during the combination of prediction and shaping signals.
- **`silk_LSHIFT32`** / **`silk_RSHIFT_ROUND`**: Used extensively with explicit Q-format tracking.

### 6.3 Saturation Points

- **PCM output**: `silk_SAT16()` applied to all int16 outputs (`xq`, `outL`, `outH`, stereo output)
- **Residual error in NSQ**: Clamped to `±(31 << 10)` before quantization
- **Quantized pulses**: `silk_RSHIFT_ROUND(q1_Q10, 10)` → cast to `opus_int8`
- **Stereo width**: Clamped to `[0, 1<<14]`
- **Noise levels**: Limited to `0x00FFFFFF` (7 bits headroom)
- **NLSF quantizer indices**: Clamped to `±NLSF_QUANT_MAX_AMPLITUDE_EXT`

### 6.4 Rounding

- **`silk_RSHIFT_ROUND`**: Adds `(1 << (shift-1))` before right-shifting (round-half-up)
- **`silk_SMLAWB`**: Multiply-accumulate with 16×32→32 rounding (fractional multiply + accumulate)
- **Quantization offset**: `silk_Quantization_Offsets_Q10[signalType>>1][quantOffsetType]` — asymmetric dead zone
- **QUANT_LEVEL_ADJUST_Q10**: Applied to move quantization levels slightly toward zero

---

## 7. Dependencies

### 7.1 Modules Called by the Encoder

| Module | Called From | Purpose |
|--------|------------|---------|
| `silk_resampler` | `enc_API.c` | Resample API rate → internal rate |
| `silk_biquad_alt_stride1` | `LP_variable_cutoff.c` | ARMA filtering for bandwidth transition |
| `silk_lin2log` / `silk_log2lin` | Throughout | Fixed-point log/exp conversions |
| `silk_sigm_Q15` | `VAD.c` | Sigmoid function for speech probability |
| `silk_SQRT_APPROX` | `VAD.c`, `stereo_find_predictor.c` | Fixed-point square root |
| `silk_NLSF_stabilize` | `NLSF_encode.c` | Enforce minimum NLSF spacing |
| `silk_NLSF_VQ` | `NLSF_encode.c` | First-stage NLSF VQ error computation |
| `silk_NLSF_VQ_weights_laroia` | `process_NLSFs.c` | NLSF weight computation |
| `silk_NLSF2A` | `process_NLSFs.c` | NLSF → LPC coefficient conversion |
| `silk_NLSF_decode` | `NLSF_encode.c` | Decode quantized NLSFs |
| `silk_NLSF_unpack` | `NLSF_encode.c`, `encode_indices.c` | Unpack entropy table indices |
| `silk_interpolate` | `process_NLSFs.c` | Linear interpolation of NLSF vectors |
| `silk_insertion_sort_increasing` | `NLSF_encode.c` | Sort for survivor selection |
| `silk_inner_prod_aligned_scale` | `stereo_find_predictor.c` | Scaled inner product |
| `silk_sum_sqr_shift` | `stereo_find_predictor.c` | Norm computation with auto-scaling |
| `silk_shell_encoder` | `encode_pulses.c` | Shell coding of pulse distributions |
| `silk_encode_signs` | `encode_pulses.c` | Sign encoding of pulses |
| `ec_enc_icdf` | `encode_indices.c`, `encode_pulses.c`, `stereo_encode_pred.c` | Range coder symbol encoding |
| `ec_enc_patch_initial_bits` | `enc_API.c` | Retroactive flag insertion |
| `silk_DIV32_varQ` | Throughout | Variable-Q division |
| `silk_INVERSE32_varQ` | `NSQ.c` | High-precision inverse |
| `silk_bwexpander_32` | `A2NLSF.c` | Bandwidth expansion for stability |
| `silk_NSQ_noise_shape_feedback_loop` | `NSQ.c` | Arch-dispatched shaping filter |
| `silk_noise_shape_quantizer_short_prediction` | `NSQ.c` | Arch-dispatched LPC prediction |

### 7.2 What Calls the Encoder

- **Opus encoder** (`src/opus_encoder.c`) → calls `silk_Encode()`
- `silk_Encode` calls `silk_encode_frame_Fxx()` (in `fixed/encode_frame_FIX.c`) which orchestrates the per-frame encoding

---

## 8. Constants and Tables

### 8.1 Key Constants

| Constant | Value | Source |
|----------|-------|--------|
| `MAX_FRAME_LENGTH` | 960 (= 60ms × 16kHz) | `define.h` |
| `MAX_NB_SUBFR` | 4 | `define.h` |
| `MAX_LPC_ORDER` | 16 | `define.h` |
| `LTP_ORDER` | 5 | `define.h` |
| `MAX_SHAPE_LPC_ORDER` | 24 | `define.h` |
| `NSQ_LPC_BUF_LENGTH` | 16 | `define.h` |
| `SHELL_CODEC_FRAME_LENGTH` | 16 | `define.h` |
| `DECISION_DELAY` | 40 | `NSQ.h` |
| `MAX_DEL_DEC_STATES` | 4 | `define.h` |
| `NLSF_QUANT_DEL_DEC_STATES` | 32 | `define.h` |
| `NLSF_QUANT_MAX_AMPLITUDE` | 4 | `define.h` |
| `NLSF_QUANT_MAX_AMPLITUDE_EXT` | 10 | `define.h` |
| `VAD_N_BANDS` | 4 | `define.h` |
| `TRANSITION_FRAMES` | 80 | `define.h` |
| `TRANSITION_NB` | 3 | (biquad numerator coeffs) |
| `TRANSITION_NA` | 2 | (biquad denominator coeffs) |
| `ENCODER_NUM_CHANNELS` | 2 | `define.h` |
| `MAX_FRAMES_PER_PACKET` | 3 | `define.h` |
| `STEREO_QUANT_TAB_SIZE` | 16 | `define.h` |
| `STEREO_QUANT_SUB_STEPS` | 5 | `define.h` |
| `VARIABLE_HP_MIN_CUTOFF_HZ` | 60 | `tuning_parameters.h` |
| `VARIABLE_HP_MAX_CUTOFF_HZ` | 100 | `tuning_parameters.h` |
| `MAX_SUM_LOG_GAIN_DB` | 250 | `tuning_parameters.h` |
| `QUANT_LEVEL_ADJUST_Q10` | 80 | `define.h` |
| `NLSF_QUANT_LEVEL_ADJ` | 0.1 | `define.h` |

### 8.2 Key Tables

| Table | Size | Q | Purpose |
|-------|------|---|---------|
| `silk_TargetRate_NB_21[107]` | uint8 | SNR/21 | NB bitrate→SNR mapping |
| `silk_TargetRate_MB_21[155]` | uint8 | SNR/21 | MB bitrate→SNR mapping |
| `silk_TargetRate_WB_21[191]` | uint8 | SNR/21 | WB bitrate→SNR mapping |
| `silk_Transition_LP_B_Q28[][]` | int32 | Q28 | LP transition filter numerators |
| `silk_Transition_LP_A_Q28[][]` | int32 | Q28 | LP transition filter denominators |
| `silk_LSFCosTab_FIX_Q12[129]` | int32 | Q12 | cos(k*π/128) for NLSF root finding |
| `silk_Quantization_Offsets_Q10[2][2]` | int16 | Q10 | Dead-zone offsets by signal/quant type |
| `silk_LTP_vq_ptrs_Q7[3]` | int8 | Q7 | LTP codebooks (8, 16, 32 vectors × 5) |
| `silk_LTP_vq_gain_ptrs_Q7[3]` | uint8 | Q7 | LTP codebook gains |
| `silk_LTP_gain_BITS_Q5_ptrs[3]` | uint8 | Q5 | LTP codebook lengths |
| `silk_pulses_per_block_iCDF[]` | uint8 | iCDF | Shell coding rate tables |
| `silk_stereo_pred_quant_Q13[]` | int32 | Q13 | Stereo predictor quantization levels |
| `silk_stereo_pred_joint_iCDF[]` | uint8 | iCDF | Joint stereo prediction entropy |
| Various `*_iCDF` tables | uint8 | 8-bit iCDF | Entropy coding distributions |

### 8.3 SNR Table Derivation

The `silk_TargetRate_*_21` tables map target bitrate (in 400 bps increments, offset by 10 entries = 4000 bps) to SNR values. The stored values are `SNR_dB / 21`, packed into uint8. The actual SNR is recovered as `table[id] * 21` (Q7). For 10ms frames, 2000 bps is subtracted from the target rate before lookup.

### 8.4 Analysis Filter Bank Coefficients

The allpass filter coefficients for `silk_ana_filt_bank_1` are derived from a bilinear transform of a halfband filter:
- `A_fb1_20 = 10788` (= 5394 × 2)
- `A_fb1_21 = -24290` (= 20623 × 2 as uint16, wraps to negative in int16)

---

## 9. Edge Cases

### 9.1 Error Conditions (from `check_control_input`)

| Condition | Error Code |
|-----------|------------|
| API sample rate not in {8k,12k,16k,24k,32k,44.1k,48k} | `SILK_ENC_FS_NOT_SUPPORTED` |
| Internal rate not in {8k,12k,16k} | `SILK_ENC_FS_NOT_SUPPORTED` |
| `min > desired` or `max < desired` or `min > max` | `SILK_ENC_FS_NOT_SUPPORTED` |
| Packet size not in {10,20,40,60} ms | `SILK_ENC_PACKET_SIZE_NOT_SUPPORTED` |
| Loss percentage outside [0,100] | `SILK_ENC_INVALID_LOSS_RATE` |
| DTX/CBR/FEC not 0 or 1 | Various `SILK_ENC_INVALID_*` |
| Channels < 1 or > `ENCODER_NUM_CHANNELS` | `SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR` |
| Internal channels > API channels | `SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR` |
| Complexity outside [0,10] | `SILK_ENC_INVALID_COMPLEXITY_SETTING` |
| Input not multiple of 10ms | `SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES` |
| Input exceeds one packet | `SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES` |

### 9.2 Special Input Handling

- **Prefill mode** (`prefillFlag`): Accepts exactly 10ms of input, resets encoder (optionally preserving LP state), runs at complexity 0 with no output. Used to warm up filter states.
- **First frame after reset**: Forces `CODE_INDEPENDENTLY` (no delta coding), disables NLSF interpolation.
- **DTX**: When both channels are in DTX (`inDTX` and VAD detects silence), `*nBytesOut = 0`.
- **Mono↔stereo transition**: On `toMono`, stereo width collapses to zero; on mono→stereo, second channel state is freshly initialized and resampler state is copied.
- **Bandwidth switch**: State machine in `silk_control_audio_bandwidth` manages gradual transitions via the LP filter. When `opusCanSwitch` is set, switches instantly; otherwise uses a transition period with double-speed filter ramping.
- **12 kHz / 10ms special case**: `frame_length = 120` is not a multiple of `SHELL_CODEC_FRAME_LENGTH = 16`, so `silk_encode_pulses` pads with zeros and adds an extra shell block (`iter++`).
- **LBRR (FEC)**: LBRR data from the *previous* packet is encoded at the *start* of the current packet's bitstream, providing one-packet delay redundancy.

### 9.3 A2NLSF Convergence

If root finding fails to find all `d` NLSFs (can happen with ill-conditioned filters), the function applies bandwidth expansion (`silk_bwexpander_32` with factor 65536 - 1 = multiply by ~0.99998) and retries, up to 16 iterations. If the first NLSF evaluates to a negative value, it is set to zero.

---

## 10. Porting Notes

### 10.1 Pointer Arithmetic and In-Place Mutation

- **`silk_stereo_LR_to_MS`**: `mid = &x1[-2]` — the mid signal pointer is set 2 samples before the input array. This relies on the `inputBuf` having 2 extra samples at the front. In Rust, use slice indexing with an explicit offset.
- **NSQ `pxq` pointer**: `pxq = &NSQ->xq[ltp_mem_length]` then advanced by `subfr_length` each subframe. The `xq` array serves as both history and output.
- **`psLPC_Q14` pointer**: Starts at `NSQ->sLPC_Q14[NSQ_LPC_BUF_LENGTH - 1]`, then incremented per sample. After the subframe, the buffer is shifted via `silk_memcpy`. In Rust, use a ring buffer or explicit index management.
- **`pred_lag_ptr` / `shp_lag_ptr`**: Negative indexing into `sLTP_Q15` and `sLTP_shp_Q14` arrays. Will require careful index arithmetic in Rust.
- **In-place filtering**: `silk_LP_variable_cutoff`, `silk_LPC_analysis_filter`, and `silk_ana_filt_bank_1` all operate in-place on their input arrays.

### 10.2 Stack Allocation (`VARDECL` / `ALLOC`)

The C code uses `VARDECL`/`ALLOC` macros for variable-length stack arrays (backed by `alloca` or VLAs). Key occurrences:

| Function | Allocation | Max Size |
|----------|-----------|----------|
| `silk_Encode` | `buf[nSamplesFromInputMax]` | ~960 × int16 |
| `silk_NSQ_c` | `sLTP_Q15`, `sLTP`, `x_sc_Q10` | `ltp_mem_length + frame_length` |
| `silk_NSQ_del_dec_c` | `psDelDec[nStatesDelayedDecision]` | 4 × `NSQ_del_dec_struct` (~4KB each) |
| `silk_NLSF_encode` | `err_Q24[nVectors]`, `RD_Q25[nSurvivors]`, `tempIndices2[nSurvivors * MAX_LPC_ORDER]` | Variable |
| `silk_VAD_GetSA_Q8` | `X[X_offset[3] + decimated_framelength1]` | ~frame_length |
| `silk_stereo_LR_to_MS` | `side`, `LP_mid`, `HP_mid`, `LP_side`, `HP_side` | `frame_length + 2` each |

In Rust, replace with `Vec` or fixed-size arrays (since maximum sizes are bounded by constants).

### 10.3 Conditional Compilation

- **`FIXED_POINT` vs float**: The encoder uses `silk_encoder_state_FIX` (fixed-point) or `silk_encoder_state_FLP` (floating-point). For bit-exact Rust port, use fixed-point only.
- **`ENABLE_DRED`**: DRED (Deep Redundancy) encoder support — can be ignored for initial port.
- **`ENABLE_QEXT`**: Extended quality mode with 96 kHz support — can be ignored.
- **`OPUS_X86_MAY_HAVE_SSE4_1`**: SSE4.1 optimized paths for `silk_noise_shape_quantizer`, `silk_VAD_GetNoiseLevels`, `silk_NSQ_noise_shape_feedback_loop`. The `_c` suffix functions are the portable C implementations.
- **`silk_short_prediction_create_arch_coef`**: Architecture-specific coefficient packing for SIMD LPC prediction.
- **`USE_CELT_FIR`**: Alternative LPC analysis path (disabled by default, `#define USE_CELT_FIR 0`).

### 10.4 Macro-Generated Fixed-Point Operations

The following macros encode specific fixed-point operation semantics and must be faithfully reproduced:

| Macro | Semantics |
|-------|-----------|
| `silk_SMULBB(a,b)` | `(int32)(int16)a * (int32)(int16)b` — bottom × bottom |
| `silk_SMLABB(a,b,c)` | `a + (int16)b * (int16)c` |
| `silk_SMULWB(a,b)` | `(a * (int16)b) >> 16` — word × bottom-half |
| `silk_SMLAWB(a,b,c)` | `a + (b * (int16)c) >> 16` |
| `silk_SMULWW(a,b)` | `(a >> 16) * b + ((a & 0xFFFF) * b >> 16)` |
| `silk_SMLAWT(a,b,c)` | `a + (b * (c >> 16)) >> 16` — word × top-half |
| `silk_ADD_LSHIFT32(a,b,s)` | `a + (b << s)` |
| `silk_SUB32_ovflw(a,b)` | `(uint32)a - (uint32)b` — wrapping subtraction |
| `silk_ADD32_ovflw(a,b)` | `(uint32)a + (uint32)b` — wrapping addition |
| `silk_SMLABB_ovflw(a,b,c)` | wrapping multiply-accumulate |
| `silk_RSHIFT_ROUND(a,s)` | `(a + (1 << (s-1))) >> s` |
| `silk_FIX_CONST(c,q)` | `(int32)(c * (1 << q) + 0.5)` — compile-time constant |
| `silk_RAND(seed)` | `seed = seed * 196314165 + 907633515` — LCG |
| `silk_LIMIT_32(v,lo,hi)` | Clamp to `[lo, hi]` |
| `silk_SAT16(v)` | Saturate to int16 range |

**Porting approach**: Implement each as a Rust `#[inline]` function with explicit types. Pay special attention to the wrapping operations — use Rust's `wrapping_add`, `wrapping_sub`, `wrapping_mul`. The "overflow" variants (`_ovflw`) are intentional and must not panic.

### 10.5 `goto` in Stereo Quantization

`silk_stereo_quant_pred` uses `goto done` to break out of a nested loop when quantization error starts increasing. In Rust, use labeled loops:

```rust
'outer: for i in 0..STEREO_QUANT_TAB_SIZE - 1 {
    for j in 0..STEREO_QUANT_SUB_STEPS {
        // ...
        if err_Q13 >= err_min_Q13 {
            break 'outer;
        }
    }
}
```

### 10.6 Union-Typed Arrays

- **`silk_LP_variable_cutoff`**: `fac_Q16` is used both as a Q16 fraction and split into integer index + fractional part. No actual union, but the dual interpretation needs clear documentation.
- **`HarmShapeFIRPacked_Q14`**: Packs two 16-bit values into one 32-bit value — low 16 bits are `HarmShapeGain >> 2`, high 16 bits are `HarmShapeGain >> 1`. Used with `silk_SMULWB` and `silk_SMLAWT` to extract the two halves. In Rust, use explicit pack/unpack.
- **`LF_shp_Q14`**: Similarly packs two int16 coefficients per int32 — accessed via `silk_SMULWB` (low) and `silk_SMLAWT` (high).

### 10.7 `silk_memmove` for Ring Buffer Maintenance

After each frame, the NSQ shifts its ring buffers:
```c
silk_memmove(NSQ->xq, &NSQ->xq[frame_length], ltp_mem_length * sizeof(opus_int16));
silk_memmove(NSQ->sLTP_shp_Q14, &NSQ->sLTP_shp_Q14[frame_length], ltp_mem_length * sizeof(opus_int32));
```

This is overlapping copy (source and dest overlap). In Rust, use `copy_within` on slices.

### 10.8 Architecture Dispatch

Functions like `silk_NSQ`, `silk_NSQ_del_dec`, `silk_VQ_WMat_EC`, `silk_VAD_GetSA_Q8` have `_c` suffix versions (portable C) and may have SIMD variants dispatched via function pointers. For the initial port, implement only the `_c` versions. The `arch` parameter can be preserved in the Rust API for future SIMD.

### 10.9 `silk_DWORD_ALIGN`

Used on `x_buf` and `PredCoef_Q12` arrays in `silk_encoder_state_FIX` and `silk_encoder_control_FIX`. Ensures 4-byte alignment for SIMD. In Rust, use `#[repr(align(4))]` wrapper or rely on natural alignment of `i16`/`i32` arrays.

### 10.10 Opaque `void*` Encoder Handle

The public API uses `void *encState` which is cast to `silk_encoder *` internally. In Rust, use a properly typed struct — the opacity is only for C API compatibility and is not needed in the Rust implementation.
