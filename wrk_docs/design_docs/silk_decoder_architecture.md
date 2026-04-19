Now I have everything needed. Here is the complete architecture document:

---

# SILK Decoder Module — Architecture Documentation

## 1. Purpose

The SILK decoder is one of two codec engines inside Opus (the other being CELT). It handles narrowband (8 kHz), medium-band (12 kHz), and wideband (16 kHz) speech decoding using a Code-Excited Linear Prediction (CELP) model. Within the Opus pipeline, `silk_Decode()` is called by the top-level `opus_decode()` when the Opus TOC byte indicates a SILK-only or Hybrid frame.

The decoder's job is:
1. Entropy-decode quantization indices from the range coder bitstream
2. Dequantize parameters (gains, NLSFs, pitch lags, LTP coefficients, excitation pulses)
3. Reconstruct the speech signal via inverse NSQ (Noise Shaping Quantization): LTP synthesis followed by LPC synthesis
4. Handle packet loss via PLC (Packet Loss Concealment) and CNG (Comfort Noise Generation)
5. Convert stereo Mid/Side to Left/Right when applicable
6. Resample from the internal rate (8/12/16 kHz) to the API output rate (8–48 kHz)

---

## 2. Public API

### 2.1 Top-Level Decoder API (`dec_API.c`)

```c
opus_int silk_Get_Decoder_Size(opus_int *decSizeBytes);
```
Returns the byte size of the `silk_decoder` super-struct. The caller allocates the memory.

```c
opus_int silk_InitDecoder(void *decState);
```
Zero-fills the entire state, then calls `silk_init_decoder()` for each channel and resets stereo state. Optionally loads OSCE models.

```c
opus_int silk_ResetDecoder(void *decState);
```
Partial reset: clears from `SILK_DECODER_STATE_RESET_START` onward (preserves OSCE/resampler state before that marker), resets CNG and PLC, sets `first_frame_after_reset = 1`, `prev_gain_Q16 = 65536` (Q16 1.0).

```c
opus_int silk_Decode(
    void *decState,
    silk_DecControlStruct *decControl,
    opus_int lostFlag,          // 0=normal, 1=lost, 2=decode FEC (LBRR)
    opus_int newPacketFlag,     // 1 for first frame in packet
    ec_dec *psRangeDec,
    opus_res *samplesOut,       // interleaved if stereo
    opus_int32 *nSamplesOut,
    int arch
);
```
Main entry point. Decodes one frame per call. Returns `SILK_NO_ERROR` (0) on success; error codes are accumulated via `ret +=`.

**Parameters:**
- `lostFlag`: `FLAG_DECODE_NORMAL` (0), `FLAG_PACKET_LOST` (1), or `FLAG_DECODE_LBRR` (2)
- `newPacketFlag`: Set on first decoder call for a new Opus packet; resets per-channel `nFramesDecoded` to 0
- `decControl`: Contains `nChannelsAPI`, `nChannelsInternal`, `API_sampleRate`, `internalSampleRate`, `payloadSize_ms`, plus OSCE/deep-PLC enable flags

### 2.2 Per-Channel Frame Decoder (`decode_frame.c`)

```c
opus_int silk_decode_frame(
    silk_decoder_state *psDec,
    ec_dec *psRangeDec,
    opus_int16 pOut[],          // output speech frame
    opus_int32 *pN,             // output: number of samples
    opus_int lostFlag,
    opus_int condCoding,        // CODE_INDEPENDENTLY, CODE_CONDITIONALLY, or CODE_INDEPENDENTLY_NO_LTP_SCALING
    int arch
);
```
Decodes a single channel's single frame. Returns 0 on success. On normal decode, calls the full pipeline: `silk_decode_indices` → `silk_decode_pulses` → `silk_decode_parameters` → `silk_decode_core` → output buffer update → OSCE → PLC update. On loss, calls `silk_PLC` in conceal mode. Always finishes with CNG and PLC glue.

### 2.3 Bitstream Parsing

```c
void silk_decode_indices(
    silk_decoder_state *psDec,
    ec_dec *psRangeDec,
    opus_int FrameIndex,
    opus_int decode_LBRR,       // nonzero when decoding FEC data
    opus_int condCoding
);
```
Entropy-decodes all side information: signal type, quantizer offset, gain indices, NLSF indices, NLSF interpolation factor, pitch lag (absolute or delta), contour index, LTP gain indices (PERIndex + per-subframe), LTP scaling index, and seed.

```c
void silk_decode_pulses(
    ec_dec *psRangeDec,
    opus_int16 pulses[],
    const opus_int signalType,
    const opus_int quantOffsetType,
    const opus_int frame_length
);
```
Decodes the excitation pulse signal in three stages: (1) rate level and sum-weighted-pulses per shell block, (2) shell decoding to distribute pulses across 16-sample blocks, (3) LSB decoding for large amplitudes, (4) sign attachment.

### 2.4 Parameter Dequantization

```c
void silk_decode_parameters(
    silk_decoder_state *psDec,
    silk_decoder_control *psDecCtrl,
    opus_int condCoding
);
```
Converts indices to actual filter parameters: dequantizes gains via `silk_gains_dequant`, decodes NLSFs via `silk_NLSF_decode`, converts NLSFs to LPC coefficients via `silk_NLSF2A` (with optional interpolation for first-half subframes), decodes pitch lags via `silk_decode_pitch`, and unpacks LTP codebook vectors.

```c
void silk_decode_pitch(
    opus_int16 lagIndex,
    opus_int8 contourIndex,
    opus_int pitch_lags[],      // output: per-subframe lags
    const opus_int Fs_kHz,
    const opus_int nb_subfr
);
```
Reconstructs per-subframe pitch lags from a base lag index plus contour codebook offsets. Clamps to `[PE_MIN_LAG_MS * Fs_kHz, PE_MAX_LAG_MS * Fs_kHz]`.

### 2.5 Synthesis Core

```c
void silk_decode_core(
    silk_decoder_state *psDec,
    silk_decoder_control *psDecCtrl,
    opus_int16 xq[],            // output: decoded speech
    const opus_int16 pulses[MAX_FRAME_LENGTH],
    int arch
);
```
The inverse NSQ engine. Converts quantized pulses to excitation, then for each subframe: LTP synthesis (voiced) followed by LPC synthesis, producing the final output speech. This is the most numerically critical function.

---

## 3. Internal State

### 3.1 `silk_decoder` (Super-Struct, defined in `dec_API.c`)

```c
typedef struct {
    silk_decoder_state  channel_state[DECODER_NUM_CHANNELS]; // DECODER_NUM_CHANNELS = 2
    stereo_dec_state    sStereo;
    opus_int            nChannelsAPI;
    opus_int            nChannelsInternal;
    opus_int            prev_decode_only_middle;
} silk_decoder;
```
Opaque to callers (cast from `void *`). Contains both channel decoders plus shared stereo state.

### 3.2 `silk_decoder_state` (Per-Channel, `structs.h`)

```c
typedef struct {
    // --- OSCE fields at top (not reset by silk_reset_decoder) ---
    // ... OSCE / BWE structs if enabled ...

#define SILK_DECODER_STATE_RESET_START prev_gain_Q16
    opus_int32  prev_gain_Q16;                              // Previous subframe gain, Q16 (init: 65536)
    opus_int32  exc_Q14[MAX_FRAME_LENGTH];                  // Excitation buffer, Q14, 320 samples max
    opus_int32  sLPC_Q14_buf[MAX_LPC_ORDER];                // LPC filter state, Q14, 16 elements
    opus_int16  outBuf[MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH]; // Output history, 480 samples
    opus_int    lagPrev;                                    // Previous frame pitch lag
    opus_int8   LastGainIndex;                              // Previous absolute gain index
    opus_int    fs_kHz;                                     // Internal sample rate (8, 12, or 16)
    opus_int32  fs_API_hz;                                  // API output rate in Hz
    opus_int    nb_subfr;                                   // 2 (10ms) or 4 (20ms)
    opus_int    frame_length;                               // nb_subfr * subfr_length
    opus_int    subfr_length;                               // SUB_FRAME_LENGTH_MS * fs_kHz
    opus_int    ltp_mem_length;                             // LTP_MEM_LENGTH_MS * fs_kHz
    opus_int    LPC_order;                                  // 10 (NB/MB) or 16 (WB)
    opus_int16  prevNLSF_Q15[MAX_LPC_ORDER];                // Previous frame's NLSFs
    opus_int    first_frame_after_reset;                    // Disables NLSF interpolation
    const opus_uint8 *pitch_lag_low_bits_iCDF;             // Rate-dependent iCDF for lag LSBs
    const opus_uint8 *pitch_contour_iCDF;                  // Rate/subfr-dependent contour iCDF
    opus_int    nFramesDecoded;                             // Frame counter within packet
    opus_int    nFramesPerPacket;                           // 1, 2, or 3
    opus_int    ec_prevSignalType;                          // For conditional pitch lag coding
    opus_int16  ec_prevLagIndex;                            // For conditional pitch lag coding
    opus_int    VAD_flags[MAX_FRAMES_PER_PACKET];           // Per-frame VAD flags from header
    opus_int    LBRR_flag;                                  // Packet-level LBRR presence
    opus_int    LBRR_flags[MAX_FRAMES_PER_PACKET];          // Per-frame LBRR presence
    silk_resampler_state_struct resampler_state;            // Internal→API resampler
    const silk_NLSF_CB_struct *psNLSF_CB;                  // Points to NB/MB or WB codebook
    SideInfoIndices indices;                                // All decoded indices for current frame
    silk_CNG_struct sCNG;                                   // Comfort noise state
    opus_int    lossCnt;                                    // Consecutive lost frame counter
    opus_int    prevSignalType;                             // TYPE_VOICED, TYPE_UNVOICED, TYPE_NO_VOICE_ACTIVITY
    int         arch;                                       // CPU feature flags
    silk_PLC_struct sPLC;                                   // Packet loss concealment state
} silk_decoder_state;
```

**Lifecycle:**
- `silk_init_decoder`: zero-fills entire struct, then calls `silk_reset_decoder`
- `silk_reset_decoder`: zero-fills from `SILK_DECODER_STATE_RESET_START` onward, preserving OSCE state at the top; sets `prev_gain_Q16 = 65536`, `first_frame_after_reset = 1`; resets CNG and PLC
- `silk_decoder_set_fs`: called when sample rate changes; reconfigures `subfr_length`, `frame_length`, `ltp_mem_length`, `LPC_order`, NLSF codebook pointer, pitch iCDF pointers; clears `outBuf` and `sLPC_Q14_buf`

### 3.3 `silk_decoder_control` (Per-Frame Transient, `structs.h`)

```c
typedef struct {
    opus_int    pitchL[MAX_NB_SUBFR];                   // Per-subframe pitch lags
    opus_int32  Gains_Q16[MAX_NB_SUBFR];                // Per-subframe gains, Q16
    opus_int16  PredCoef_Q12[2][MAX_LPC_ORDER];         // LPC coefficients: [0]=first half, [1]=second half
    opus_int16  LTPCoef_Q14[LTP_ORDER * MAX_NB_SUBFR];  // 5-tap LTP filter per subframe
    opus_int    LTP_scale_Q14;                          // LTP downscaling for inter-packet dependency
} silk_decoder_control;
```
Stack-allocated in `silk_decode_frame`. Not persisted across frames.

### 3.4 `SideInfoIndices` (Bitstream Indices)

```c
typedef struct {
    opus_int8   GainsIndices[MAX_NB_SUBFR];     // Gain indices (absolute first, delta rest)
    opus_int8   LTPIndex[MAX_NB_SUBFR];         // LTP codebook index per subframe
    opus_int8   NLSFIndices[MAX_LPC_ORDER + 1]; // [0]=CB1 index, [1..]=residual indices
    opus_int16  lagIndex;                       // Pitch lag index
    opus_int8   contourIndex;                   // Pitch contour codebook index
    opus_int8   signalType;                     // 0=inactive, 1=unvoiced, 2=voiced
    opus_int8   quantOffsetType;                // 0 or 1 — selects quantization offset
    opus_int8   NLSFInterpCoef_Q2;              // NLSF interpolation factor [0..4] in Q2
    opus_int8   PERIndex;                       // Periodicity codebook index (LTP table select)
    opus_int8   LTP_scaleIndex;                 // LTP scaling table index
    opus_int8   Seed;                           // 2-bit random seed for excitation sign randomization
} SideInfoIndices;
```

### 3.5 `silk_PLC_struct`

```c
typedef struct {
    opus_int32  pitchL_Q8;              // Pitch period in Q8 samples
    opus_int16  LTPCoef_Q14[LTP_ORDER]; // Saved LTP coefficients
    opus_int16  prevLPC_Q12[MAX_LPC_ORDER]; // Saved LPC coefficients
    opus_int    last_frame_lost;        // Flag for glue-frame energy matching
    opus_int32  rand_seed;              // PRNG state
    opus_int16  randScale_Q14;          // Random noise mixing level
    opus_int32  conc_energy;            // Concealed frame energy for fade-in
    opus_int    conc_energy_shift;      // Shift for conc_energy
    opus_int16  prevLTP_scale_Q14;      // Previous LTP scaling
    opus_int32  prevGain_Q16[2];        // Last two subframe gains
    opus_int    fs_kHz;                 // Cached sample rate
    opus_int    nb_subfr;               // Cached subframe count
    opus_int    subfr_length;           // Cached subframe length
    opus_int    enable_deep_plc;        // Deep PLC enable flag
} silk_PLC_struct;
```

### 3.6 `silk_CNG_struct`

```c
typedef struct {
    opus_int32  CNG_exc_buf_Q14[MAX_FRAME_LENGTH]; // Excitation buffer for CNG synthesis
    opus_int16  CNG_smth_NLSF_Q15[MAX_LPC_ORDER];  // Smoothed NLSF for CNG spectral shape
    opus_int32  CNG_synth_state[MAX_LPC_ORDER];     // LPC synthesis filter state
    opus_int32  CNG_smth_Gain_Q16;                  // Smoothed gain level
    opus_int32  rand_seed;                          // PRNG state (init: 3176576)
    opus_int    fs_kHz;                             // Cached sample rate
} silk_CNG_struct;
```

### 3.7 `stereo_dec_state`

```c
typedef struct {
    opus_int16  pred_prev_Q13[2];   // Previous M/S predictor values
    opus_int16  sMid[2];            // 2-sample buffer for mid channel
    opus_int16  sSide[2];           // 2-sample buffer for side channel
} stereo_dec_state;
```

### 3.8 `silk_NLSF_CB_struct` (Codebook Definition, const)

```c
typedef struct {
    const opus_int16  nVectors;           // Number of first-stage vectors
    const opus_int16  order;              // LPC order (10 or 16)
    const opus_int16  quantStepSize_Q16;  // Residual quantization step
    const opus_int16  invQuantStepSize_Q6;
    const opus_uint8  *CB1_NLSF_Q8;      // First-stage codebook entries [nVectors * order]
    const opus_int16  *CB1_Wght_Q9;      // First-stage weights [nVectors * order]
    const opus_uint8  *CB1_iCDF;          // First-stage iCDF [signalType/2 * nVectors]
    const opus_uint8  *pred_Q8;           // Backward predictor coefficients
    const opus_uint8  *ec_sel;            // Entropy coding table selector
    const opus_uint8  *ec_iCDF;           // Per-coefficient entropy iCDF tables
    const opus_uint8  *ec_Rates_Q5;       // Rate tables
    const opus_int16  *deltaMin_Q15;      // Minimum spacing [order + 1]
} silk_NLSF_CB_struct;
```
Two static instances: `silk_NLSF_CB_NB_MB` (order 10, 8/12 kHz) and `silk_NLSF_CB_WB` (order 16, 16 kHz).

### 3.9 `silk_resampler_state_struct`

```c
typedef struct {
    opus_int32   sIIR[SILK_RESAMPLER_MAX_IIR_ORDER]; // IIR filter state
    union {
        opus_int32 i32[SILK_RESAMPLER_MAX_FIR_ORDER]; // FIR state (32-bit for down_FIR)
        opus_int16 i16[SILK_RESAMPLER_MAX_FIR_ORDER]; // FIR state (16-bit for IIR_FIR)
    } sFIR;
    opus_int16       delayBuf[96];          // Input delay compensation buffer
    opus_int         resampler_function;    // Method selector (0–3)
    opus_int         batchSize;             // Fs_in_kHz * RESAMPLER_MAX_BATCH_SIZE_MS
    opus_int32       invRatio_Q16;          // Fs_in / Fs_out in Q16 (for FIR interpolation)
    opus_int         FIR_Order;             // FIR filter order (18, 24, or 36)
    opus_int         FIR_Fracs;             // Number of fractional phases
    opus_int         Fs_in_kHz;
    opus_int         Fs_out_kHz;
    opus_int         inputDelay;            // Samples of delay compensation
    const opus_int16 *Coefs;                // Points to static coefficient table
} silk_resampler_state_struct;
```

---

## 4. Algorithm

### 4.1 Top-Level Decode Flow (`silk_Decode`)

```
silk_Decode() called once per frame within a packet:
│
├─ First call (newPacketFlag): reset nFramesDecoded=0 for each channel
├─ Handle mono↔stereo transitions (init second channel if expanding)
│
├─ First frame (nFramesDecoded==0):
│  ├─ Configure nb_subfr and nFramesPerPacket from payloadSize_ms
│  ├─ Call silk_decoder_set_fs() per channel
│  ├─ Decode VAD flags and LBRR flags from bitstream header
│  └─ If normal decode: skip all LBRR data (advance range coder)
│
├─ Decode stereo predictor (MS_pred_Q13) if stereo
│
├─ For each internal channel:
│  ├─ Determine condCoding (independent vs conditional)
│  └─ Call silk_decode_frame()
│
├─ If stereo: silk_stereo_MS_to_LR() to convert Mid/Side → Left/Right
│
├─ For each output channel:
│  ├─ silk_resampler() to convert internal rate → API rate
│  └─ Interleave into output buffer if stereo
│
├─ Handle mono→stereo upmix (duplicate mono to both channels)
├─ Export pitch lag at 48 kHz for CELT
└─ Update prev_decode_only_middle, LastGainIndex
```

### 4.2 Per-Frame Decode Flow (`silk_decode_frame`)

```
silk_decode_frame():
│
├─ If NOT lost:
│  ├─ silk_decode_indices()      — parse all quantization indices
│  ├─ silk_decode_pulses()       — decode excitation pulse signal
│  ├─ silk_decode_parameters()   — dequantize gains, NLSFs→LPC, pitch, LTP
│  ├─ silk_decode_core()         — inverse NSQ: LTP + LPC synthesis
│  ├─ Update outBuf (shift left by frame_length, append new output)
│  ├─ silk_PLC(lost=0)           — update PLC state with good frame params
│  └─ Clear lossCnt, update prevSignalType
│
├─ If LOST:
│  ├─ silk_PLC(lost=1)           — generate concealment signal
│  └─ Update outBuf
│
├─ silk_CNG()                    — add comfort noise if in DTX/loss
├─ silk_PLC_glue_frames()        — smooth transition from concealed→good
└─ Update lagPrev, set output length
```

### 4.3 Inverse NSQ (`silk_decode_core`) — The Core Algorithm

This is a two-filter cascade: Long-Term Prediction (LTP) followed by Short-Term Prediction (LPC).

**Step 1: Excitation generation**
```c
// For each sample i in frame:
rand_seed = silk_RAND(rand_seed);       // 32-bit LCG PRNG
exc_Q14[i] = pulses[i] << 14;          // Scale pulse to Q14
// Dead-zone adjustment:
if (exc_Q14[i] > 0) exc_Q14[i] -= QUANT_LEVEL_ADJUST_Q10 << 4;
if (exc_Q14[i] < 0) exc_Q14[i] += QUANT_LEVEL_ADJUST_Q10 << 4;
// Add quantization offset:
exc_Q14[i] += offset_Q10 << 4;         // offset from Quantization_Offsets_Q10 table
// Random sign flip:
if (rand_seed < 0) exc_Q14[i] = -exc_Q14[i];
// PRNG feedback:
rand_seed += pulses[i];
```

**Step 2: Per-subframe processing (k = 0..nb_subfr-1)**

```
For each subframe k:
  A_Q12 = PredCoef_Q12[k >> 1]     // First or second half LPC set
  B_Q14 = LTPCoef_Q14[k * 5]       // 5-tap LTP filter
  Gain_Q10 = Gains_Q16[k] >> 6
  inv_gain_Q31 = 1 / Gains_Q16[k]  (in Q47→Q31)

  // Gain adjustment if gain changed from previous subframe
  if Gains_Q16[k] != prev_gain_Q16:
    gain_adj_Q16 = prev_gain / current_gain (Q16)
    Scale sLPC_Q14 state by gain_adj_Q16

  // LTP (voiced frames only):
  if signalType == TYPE_VOICED:
    // Re-whitening at subframe 0 or 2 (if NLSF interpolated):
    //   Apply LPC analysis filter to outBuf → sLTP
    //   Scale sLTP → sLTP_Q15 using inv_gain_Q31
    // LTP synthesis:
    for each sample i:
      LTP_pred_Q13 = 2  // rounding bias
      LTP_pred_Q13 += Σ(sLTP_Q15[lag_ptr-j] * B_Q14[j]) for j=0..4
      res_Q14[i] = exc_Q14[i] + (LTP_pred_Q13 << 1)
      sLTP_Q15[buf_idx] = res_Q14[i] << 1
  else:
    res_Q14 = exc_Q14  // No LTP for unvoiced

  // LPC synthesis:
  for each sample i:
    LPC_pred_Q10 = LPC_order/2  // rounding bias
    LPC_pred_Q10 += Σ(sLPC_Q14[i-j-1] * A_Q12[j]) for j=0..LPC_order-1
    sLPC_Q14[i] = SAT32(res_Q14[i] + SAT32(LPC_pred_Q10 << 4))
    xq[i] = SAT16(ROUND(sLPC_Q14[i] * Gain_Q10, 8))
```

### 4.4 NLSF Decoding Pipeline

```
silk_NLSF_decode():
├─ silk_NLSF_unpack()          — extract entropy table indices and backward predictor from CB1_index
├─ silk_NLSF_residual_dequant() — predictive dequantization of residual indices
│    Iterates in reverse order (i = order-1 downto 0):
│    pred_Q10 = (out_Q10 * pred_coef_Q8[i]) >> 8
│    out_Q10 = (indices[i] << 10) ± NLSF_QUANT_LEVEL_ADJ
│    out_Q10 = pred_Q10 + (out_Q10 * quantStepSize_Q16) >> 16
├─ Combine: NLSF = CB1_NLSF_Q8[i] << 7 + res_Q10[i] * (1<<14) / CB1_Wght_Q9[i]
├─ Clamp to [0, 32767]
└─ silk_NLSF_stabilize()       — enforce minimum spacing between NLSFs
```

### 4.5 NLSF → LPC Conversion (`silk_NLSF2A`)

```
1. Convert NLSF_Q15 → cos_LSF_QA via piecewise-linear table lookup (silk_LSFCosTab_FIX_Q12)
   with custom ordering (ordering16 or ordering10) for numerical accuracy
2. Build even/odd polynomials P, Q via silk_NLSF2A_find_poly()
3. Combine: a[k] = -(Q[k+1]-Q[k]) - (P[k+1]+P[k])  for k=0..d/2-1
            a[d-k-1] = (Q[k+1]-Q[k]) - (P[k+1]+P[k])
4. Convert from QA+1 to Q12 via silk_LPC_fit()
5. Verify stability via silk_LPC_inverse_pred_gain(); if unstable,
   apply bandwidth expansion (silk_bwexpander_32) and retry up to MAX_LPC_STABILIZE_ITERATIONS
```

### 4.6 Shell Coding (`shell_coder.c`)

The shell coder distributes pulse amplitudes across 16-sample blocks using a binary tree structure:

```
Decode: total → split into halves recursively (4 levels deep)
Level 3: total → [pulses3[0], pulses3[1]]     using silk_shell_code_table3
Level 2: each → [pulses2[*], pulses2[*]]       using silk_shell_code_table2
Level 1: each → [pulses1[*], pulses1[*]]       using silk_shell_code_table1
Level 0: each → [pulses0[*], pulses0[*]]       using silk_shell_code_table0
```

Each `decode_split(p_child1, p_child2, p)` decodes `p_child1` from the range coder using `shell_table[offsets[p]]` as iCDF, then `p_child2 = p - p_child1`.

### 4.7 Gain Dequantization (`gain_quant.c`)

Gains are coded on a log scale, uniformly quantized to `N_LEVELS_QGAIN` levels over `[MIN_QGAIN_DB, MAX_QGAIN_DB]` dB:

```c
// Dequantization:
#define OFFSET  ((MIN_QGAIN_DB * 128) / 6 + 16 * 128)
#define INV_SCALE_Q16  (65536 * (((MAX_QGAIN_DB - MIN_QGAIN_DB) * 128) / 6) / (N_LEVELS_QGAIN - 1))

// First subframe if independent: absolute index, clamped to prev_ind - 16
// Otherwise: delta index with double-step-size threshold for large jumps
gain_Q16[k] = silk_log2lin(min(INV_SCALE_Q16 * prev_ind / 65536 + OFFSET, 3967));
```

The double-step-size mechanism allows reaching the maximum gain level even with the delta coding constraint.

### 4.8 Packet Loss Concealment (`PLC.c`)

**On good frames (`silk_PLC_update`):**
- Save LTP coefficients from the subframe with the strongest pitch pulse
- Collapse LTP to single-tap centered at `LTP_ORDER/2`, clamped to `[V_PITCH_GAIN_START_MIN, V_PITCH_GAIN_START_MAX]`
- Save last two subframe gains, LPC coefficients, LTP scale

**On lost frames (`silk_PLC_conceal`):**
1. Find lowest-energy subframe of last two → use as random noise source
2. Apply bandwidth expansion to saved LPC coefficients (BWE_COEF)
3. LTP synthesis with saved pitch + random noise injection:
   ```
   sLTP_Q14[i] = (LTP_pred + rand_ptr[random_idx] * rand_scale) << 2
   ```
4. Gradually attenuate: LTP gains by `HARM_ATT_Q15`, noise by `PLC_RAND_ATTENUATE_*_Q15`
5. Slowly drift pitch upward by `PITCH_DRIFT_FAC_Q16`
6. LPC synthesis identical to `silk_decode_core`

**Glue frames (`silk_PLC_glue_frames`):**
On first good frame after loss, compute energy ratio between concealed and real frames. If real frame is louder, fade in with a gain ramp (slope made 4x steeper for DTX onset).

### 4.9 Comfort Noise Generation (`CNG.c`)

**Updates (on good, non-active frames):**
- Smooth NLSF with `CNG_NLSF_SMTH_Q16` towards current NLSF
- Smooth gain towards current gain (with 3dB fast-adapt threshold)
- Save highest-gain subframe's excitation to ring buffer

**Generation (on lost frames):**
1. Compute CNG gain: `sqrt(CNG_smth_Gain² - PLC_gain²)` — i.e., CNG fills the gap between PLC energy and expected background noise
2. Generate random excitation from saved buffer
3. Convert smoothed NLSF → LPC via `silk_NLSF2A`
4. LPC synthesis of CNG signal
5. Add to PLC output: `frame[i] += SAT16(CNG_sample * gain)`

### 4.10 Stereo Processing

**Predictor decoding (`silk_stereo_decode_pred`):**
Three-level hierarchical decoding: joint index → per-channel fine + mid indices. Produces two predictors in Q13. `pred_Q13[0] -= pred_Q13[1]` for efficient application.

**Mid/Side → Left/Right (`silk_stereo_MS_to_LR`):**
1. Buffer 2 samples of history for mid and side channels
2. Interpolate predictors over `STEREO_INTERP_LEN_MS * fs_kHz` samples
3. Apply prediction to side channel:
   ```
   side'[n] = side[n] + pred0 * lowpass3(mid) + pred1 * mid[n]
   ```
   where `lowpass3(mid) = (mid[n-1] + 2*mid[n] + mid[n+1]) << 9`
4. Convert: `left = mid + side'`, `right = mid - side'`

### 4.11 Resampler (`resampler.c`)

Selection matrix based on `(Fs_in, Fs_out)`:
- **Copy** (Fs_in == Fs_out): direct memcpy
- **Up2_HQ** (Fs_out == 2*Fs_in): 3rd-order allpass half-band filter per polyphase branch
- **IIR_FIR** (general upsample): allpass 2x upsample + 8-tap polyphase FIR interpolation (12 fractional phases)
- **Down_FIR** (downsample): AR2 prefilter + polyphase FIR interpolation (order 18/24/36, 1–3 fractional phases)

Processing is batched (`RESAMPLER_MAX_BATCH_SIZE_MS` milliseconds at a time) with 1ms input delay compensation. The first `inputDelay` samples come from the delay buffer; the remainder from the input directly.

---

## 5. Data Flow

### 5.1 Frame-Level Pipeline

```
Range coder bitstream (ec_dec)
    │
    ▼
silk_decode_indices() → SideInfoIndices (signalType, gains, NLSFs, pitch, LTP, seed)
    │
    ▼
silk_decode_pulses()  → opus_int16 pulses[MAX_FRAME_LENGTH]  (signed excitation amplitudes)
    │
    ▼
silk_decode_parameters() → silk_decoder_control:
    │  Gains_Q16[4], PredCoef_Q12[2][16], LTPCoef_Q14[20], pitchL[4], LTP_scale_Q14
    │
    ▼
silk_decode_core() → opus_int16 xq[frame_length]  (reconstructed speech at internal rate)
    │
    ▼
outBuf update:  shift left by frame_length, append xq
    │
    ▼
silk_PLC() / silk_CNG() / silk_PLC_glue_frames()  → xq modified in-place
    │
    ▼
silk_stereo_MS_to_LR() → convert mid/side channels to left/right (if stereo)
    │
    ▼
silk_resampler() → opus_int16 out[nSamplesOut]  (at API rate, interleaved if stereo)
```

### 5.2 Buffer Layouts

**`outBuf[MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH]` (= 480 samples at 16 kHz):**
- Acts as a circular history buffer for the LPC analysis filter in LTP re-whitening
- Size is `ltp_mem_length` = `LTP_MEM_LENGTH_MS * fs_kHz` (160/240/320 samples)
- On each frame: `memmove(outBuf, outBuf+frame_length, mv_len)` then `memcpy(outBuf+mv_len, pOut, frame_length)`

**`exc_Q14[MAX_FRAME_LENGTH]` (= 320 elements):**
- Full-frame excitation in Q14
- Written by `silk_decode_core` (pulse → excitation conversion)
- Read by CNG (ring buffer update) and PLC (energy computation, noise source)

**`sLPC_Q14_buf[MAX_LPC_ORDER]` (= 16 elements):**
- Persists the last `LPC_order` values of the LPC synthesis filter state between frames
- Copied to stack-local `sLPC_Q14[]` at frame start, saved back at frame end

**`samplesOut1_tmp[2][frame_length + 2]`:**
- Temporary output buffers per channel in `silk_Decode`
- The `+2` provides headroom for the stereo MS→LR filter which uses `[n-1]` to `[n+1]` indexing

**Resampler `delayBuf[96]`:**
- Stores up to `inputDelay` samples (max 44 at 96 kHz output)
- Provides delay equalization across different resampling paths

---

## 6. Numerical Details

### 6.1 Q-Format Summary

| Signal / Parameter | Q-Format | Range / Notes |
|---|---|---|
| `exc_Q14[]` | Q14 | Excitation signal; `pulse << 14 ± adjustments` |
| `sLPC_Q14[]`, `sLPC_Q14_buf[]` | Q14 | LPC filter state |
| `sLTP_Q15[]` | Q15 | LTP filter state (re-whitened, gain-normalized) |
| `Gains_Q16[]` | Q16 | Subframe gains; `log2lin(min(val, 3967))` where 3967 = 31 in Q7 |
| `prev_gain_Q16` | Q16 | Init: 65536 (= 1.0) |
| `inv_gain_Q31` | Q31 | `silk_INVERSE32_varQ(gain, 47)` → 47-31 = Q31 |
| `gain_adj_Q16` | Q16 | `prev_gain / current_gain` |
| `PredCoef_Q12[]` (A_Q12) | Q12 | LPC prediction coefficients |
| `LTPCoef_Q14[]` (B_Q14) | Q14 | 5-tap LTP filter; from codebook Q7 << 7 |
| `LTP_pred_Q13` | Q13 | LTP prediction output (init: 2 for rounding) |
| `LPC_pred_Q10` | Q10 | LPC prediction output (init: LPC_order/2 for rounding) |
| `NLSF_Q15[]` | Q15 | Normalized LSFs in [0, 32767] ≈ [0, π) |
| `prevNLSF_Q15[]` | Q15 | Previous frame NLSFs for interpolation |
| `NLSFInterpCoef_Q2` | Q2 | Interpolation factor [0..4]; 4 = no interpolation |
| `cos_LSF_QA[]` | QA=Q16 | `2*cos(LSF)` for polynomial evaluation |
| `MS_pred_Q13[]` | Q13 | Mid/Side stereo predictors |
| `res_Q10[]` | Q10 | NLSF residual after dequantization |
| `CB1_NLSF_Q8[]` | Q8 | First-stage NLSF codebook |
| `CB1_Wght_Q9[]` | Q9 | Codebook weights for weighted residual |
| `quantStepSize_Q16` | Q16 | Residual quantization step size |
| `pitchL_Q8` (PLC) | Q8 | Fractional pitch lag for PLC |
| `rand_scale_Q14` (PLC) | Q14 | Random noise level [0, 1] |
| `CNG_smth_Gain_Q16` | Q16 | Smoothed CNG background noise level |
| `resampler invRatio_Q16` | Q16 | `Fs_in / Fs_out` ratio |
| Biquad `B_Q28`, `A_Q28` | Q28 | Biquad filter coefficients |
| Biquad state `S[]` | Q12 | Direct-form II transposed state |

### 6.2 Rounding Bias in Prediction Loops

Both LTP and LPC predictions use a rounding bias to counter the truncation bias of `silk_SMLAWB()` (which always rounds toward -∞):

```c
// LTP: LTP_pred_Q13 = 2  (adds +0.5 LSB at Q13→Q12 conversion)
// LPC: LPC_pred_Q10 = LPC_order / 2  (adds ~0.5 LSB per tap, rounding toward zero on average)
```

This is critical for bit-exact matching. The Rust port must replicate this initialization exactly.

### 6.3 Overflow Protection

- `sLPC_Q14[i] = silk_ADD_SAT32(res_Q14[i], silk_LSHIFT_SAT32(LPC_pred_Q10, 4))` — saturating add and shift
- `xq[i] = SAT16(RSHIFT_ROUND(SMULWW(sLPC_Q14[i], Gain_Q10), 8))` — double saturation (32→16)
- `inv_gain_Q31` is clamped: `silk_min(inv_gain_Q31, silk_int32_MAX >> 1)` in PLC
- Gain dequant output clamped: `silk_min_32(val, 3967)` (31 in Q7, prevents log2lin overflow)

### 6.4 Fixed-Point Multiplication Patterns

- `silk_SMULWB(a32, b16)`: `(a * (int32_t)b) >> 16` — 32×16 fractional multiply
- `silk_SMLAWB(a32, b32, c16)`: `a + (b * (int32_t)c) >> 16` — multiply-accumulate
- `silk_SMULWW(a32, b32)`: `(int32_t)((int64_t)a * b >> 32) << 1` — 32×32→32 (keeping high 32)
- `silk_SMULBB(a16, b16)`: `(int32_t)a * (int32_t)b` — 16×16→32
- `silk_RSHIFT_ROUND(a, shift)`: `(a + (1 << (shift-1))) >> shift` — rounding right shift

---

## 7. Dependencies

### 7.1 What This Module Calls

| Dependency | Used By | Purpose |
|---|---|---|
| **Range coder** (`ec_dec_icdf`, `ec_dec_bit_logp`) | `decode_indices`, `decode_pulses`, `code_signs`, `stereo_decode_pred`, `silk_Decode` | All bitstream entropy decoding |
| **`silk_LPC_analysis_filter`** | `decode_core`, `PLC_conceal` | Re-whitening LTP state |
| **`silk_LPC_inverse_pred_gain`** | `NLSF2A`, `PLC_conceal` | Stability check / inverse gain |
| **`silk_bwexpander`** / **`silk_bwexpander_32`** | `decode_parameters`, `PLC_conceal`, `NLSF2A` | Bandwidth expansion for stability |
| **`silk_LPC_fit`** | `NLSF2A` | Convert QA+1 coefficients to Q12 with clamping |
| **`silk_INVERSE32_varQ`** | `decode_core` | Variable-Q integer reciprocal |
| **`silk_DIV32_varQ`** | `decode_core` | Variable-Q integer division |
| **`silk_log2lin`** / **`silk_lin2log`** | `gain_quant` | Log-domain ↔ linear gain conversion |
| **`silk_SQRT_APPROX`** | `PLC_glue_frames`, `CNG` | Integer square root approximation |
| **`silk_sum_sqr_shift`** | `PLC_energy`, `PLC_glue_frames` | Energy computation with normalization shift |
| **`silk_CLZ32`** | `PLC_glue_frames` | Count leading zeros |
| **`silk_insertion_sort_increasing_all_values_int16`** | `NLSF_stabilize` | Fallback sort for NLSF ordering |
| **`silk_RAND`** | `decode_core`, `CNG_exc`, `PLC_conceal` | 32-bit LCG PRNG |
| **Static tables** | Throughout | iCDF tables, codebooks, resampler coefficients |

### 7.2 What Calls This Module

- **`opus_decode` / `opus_decode_float`** (in `src/opus_decoder.c`) calls `silk_Decode()`
- The Opus decoder wrapper handles TOC parsing, determines SILK vs CELT vs Hybrid mode, and invokes the appropriate codec

---

## 8. Constants and Tables

### 8.1 Frame Geometry

| Constant | Value | Derivation |
|---|---|---|
| `MAX_NB_SUBFR` | 4 | 20ms frame = 4 × 5ms subframes |
| `SUB_FRAME_LENGTH_MS` | 5 | Fixed subframe duration |
| `MAX_FRAME_LENGTH_MS` | 20 | 4 × 5ms |
| `MAX_FRAME_LENGTH` | 320 | 20ms × 16 kHz |
| `MAX_SUB_FRAME_LENGTH` | 80 | 5ms × 16 kHz |
| `LTP_MEM_LENGTH_MS` | 20 | LTP history window |
| `MAX_LPC_ORDER` | 16 | Wideband LPC order |
| `MIN_LPC_ORDER` | 10 | NB/MB LPC order |
| `LTP_ORDER` | 5 | 5-tap LTP filter |
| `SHELL_CODEC_FRAME_LENGTH` | 16 | Shell coding block size |
| `DECODER_NUM_CHANNELS` | 2 | Max stereo channels |
| `MAX_API_FS_KHZ` | 48 (or 96 w/ QEXT) | Maximum output rate |

### 8.2 Signal Type Constants

| Constant | Value |
|---|---|
| `TYPE_NO_VOICE_ACTIVITY` | 0 |
| `TYPE_UNVOICED` | 1 |
| `TYPE_VOICED` | 2 |

### 8.3 Conditional Coding Modes

| Constant | Value | Meaning |
|---|---|---|
| `CODE_INDEPENDENTLY` | 0 | Full encoding, no inter-frame prediction |
| `CODE_CONDITIONALLY` | 1 | Delta coding relative to previous frame |
| `CODE_INDEPENDENTLY_NO_LTP_SCALING` | 2 | Independent but skip LTP downscaling |

### 8.4 Loss Flags

| Constant | Value |
|---|---|
| `FLAG_DECODE_NORMAL` | 0 |
| `FLAG_PACKET_LOST` | 1 |
| `FLAG_DECODE_LBRR` | 2 |

### 8.5 Key Static Tables

| Table | Size | Purpose |
|---|---|---|
| `silk_Quantization_Offsets_Q10[2][2]` | 4 entries | Excitation offset indexed by `[signalType>>1][quantOffsetType]` |
| `silk_LSFCosTab_FIX_Q12[129]` | 129 entries (Q12) | `2*cos(π*k/128)` for NLSF→LPC conversion |
| `silk_NLSF_CB_NB_MB`, `silk_NLSF_CB_WB` | Complex struct | Two-stage VQ codebooks for NB/MB and WB |
| `silk_LTP_vq_ptrs_Q7[3]` | 3 codebook pointers | LTP coefficient codebooks indexed by PERIndex |
| `silk_LTPScales_table_Q14[3]` | 3 entries | LTP downscaling factors |
| `silk_gain_iCDF[3]` | 3 × N entries | Gain MSB iCDFs per signal type |
| `silk_delta_gain_iCDF` | N entries | Delta gain iCDF |
| `silk_LBRR_flags_iCDF_ptr[2]` | 2 pointers | LBRR flag pattern iCDFs |
| `silk_shell_code_table{0,1,2,3}` | Variable | Shell coding iCDF tables per tree level |
| `silk_sign_iCDF[7*4]` | 28 entries | Sign coding iCDFs indexed by `(signalType, quantOffsetType, pulseCount)` |
| `silk_pulses_per_block_iCDF[N_RATE_LEVELS]` | 9 or 10 tables | Per-block pulse count iCDFs |
| `silk_rate_levels_iCDF[2]` | 2 tables | Rate level iCDFs (voiced/unvoiced) |
| `silk_pitch_lag_iCDF` | Single table | Pitch lag MSB iCDF |
| `silk_pitch_delta_iCDF` | Single table | Pitch lag delta iCDF |
| `silk_stereo_pred_quant_Q13` | ~16 entries | Stereo predictor quantization levels |
| `silk_resampler_frac_FIR_12[12][4]` | 48 coefficients | 12-phase FIR interpolation table |
| `silk_Resampler_*_COEFS` | Various | Polyphase FIR coefficients per rate ratio |
| `silk_resampler_up2_hq_{0,1}[3]` | 6 entries | 3rd-order allpass coefficients for 2x upsample |

### 8.6 PLC Constants

| Constant | Typical Value | Purpose |
|---|---|---|
| `HARM_ATT_Q15[2]` | {32440, 31130} | LTP attenuation: 0.99, 0.95 per consecutive lost frame |
| `PLC_RAND_ATTENUATE_V_Q15[2]` | {31130, 26214} | Voiced noise attenuation: 0.95, 0.8 |
| `PLC_RAND_ATTENUATE_UV_Q15[2]` | {32440, 29491} | Unvoiced noise attenuation: 0.99, 0.9 |
| `V_PITCH_GAIN_START_MIN_Q14` | ~8192 | Min LTP gain for PLC (0.5 in Q14) |
| `V_PITCH_GAIN_START_MAX_Q14` | ~14746 | Max LTP gain for PLC (0.9 in Q14) |
| `BWE_COEF` | ~0.99 | LPC bandwidth expansion during concealment |
| `PITCH_DRIFT_FAC_Q16` | Small positive | Gradual pitch elongation during loss |
| `RAND_BUF_SIZE` | 128 | Excitation noise source buffer size |
| `CNG_NLSF_SMTH_Q16` | ~6554 | CNG NLSF smoothing (≈0.1) |
| `CNG_GAIN_SMTH_Q16` | ~1638 | CNG gain smoothing (≈0.025) |
| `CNG_GAIN_SMTH_THRESHOLD_Q16` | ~92682 | 3dB fast-adapt threshold (≈√2) |

---

## 9. Edge Cases

### 9.1 Error Conditions
- **Invalid frame size**: `payloadSize_ms` not in {0, 10, 20, 40, 60} → returns `SILK_DEC_INVALID_FRAME_SIZE`
- **Invalid sample rate**: `fs_kHz_dec` not in {8, 12, 16} → returns `SILK_DEC_INVALID_SAMPLING_FREQUENCY`
- **Invalid API rate**: `API_sampleRate > MAX_API_FS_KHZ * 1000 || < 8000` → returns `SILK_DEC_INVALID_SAMPLING_FREQUENCY`

### 9.2 Channel Transitions
- **Mono→Stereo**: `silk_init_decoder(&channel_state[1])` called; stereo predictor state cleared
- **Stereo→Mono**: `stereo_to_mono` flag triggers resampling of channel 1 for smooth collapse
- **First stereo after mono**: Clears `pred_prev_Q13`, `sSide`, copies resampler state from channel 0 to channel 1

### 9.3 Rate Changes
- `silk_decoder_set_fs` detects `fs_kHz` or `frame_length` changes
- Reinitializes resampler, sets `first_frame_after_reset = 1` (disables NLSF interpolation for one frame)
- Clears `outBuf` and `sLPC_Q14_buf` to prevent artifacts
- Updates NLSF codebook pointer and pitch iCDFs

### 9.4 First Frame After Reset
- `NLSFInterpCoef_Q2` forced to 4 (no interpolation) since `prevNLSF_Q15` is invalid
- `lagPrev` set to 100 (safe default)
- `LastGainIndex` set to 10 (mid-range)
- `prevSignalType` set to `TYPE_NO_VOICE_ACTIVITY`

### 9.5 Voiced→Unvoiced Transition After PLC
When `lossCnt > 0` and `prevSignalType == TYPE_VOICED` but current frame is unvoiced, the first `MAX_NB_SUBFR/2` subframes use a smooth transition: LTP coefficients are zeroed except center tap set to 0.25, and pitch is preserved from previous frame. This avoids abrupt energy changes.

### 9.6 Mid-Only Stereo
When `decode_only_middle == 1`, side channel is zeroed. On transition back to stereo coding, channel 1's `outBuf`, `sLPC_Q14_buf` are cleared and parameters reset to safe defaults.

### 9.7 LBRR (Low Bit-Rate Redundancy) Skipping
For normal decode when LBRR data is present, the decoder must skip all LBRR frames by calling `silk_decode_indices` + `silk_decode_pulses` without using the results. This advances the range coder to the correct position.

### 9.8 10ms Frames at 12 kHz
`frame_length = 120` which is not a multiple of `SHELL_CODEC_FRAME_LENGTH = 16`. The shell coder handles this by rounding up `iter = ceil(120/16) = 8`, with the last block being only partially filled.

### 9.9 Packet Loss on First Frame
When `payloadSize_ms == 0` (loss signaled), assumes 10ms frame geometry (1 frame, 2 subframes).

---

## 10. Porting Notes

### 10.1 Pointer Arithmetic / In-Place Mutation

**`decode_core.c` — sliding buffer pointers:**
```c
pexc_Q14 += psDec->subfr_length;  // Advance excitation pointer per subframe
pxq += psDec->subfr_length;       // Advance output pointer per subframe
```
In Rust: use slice splitting or index offsets. The `sLPC_Q14` array uses negative-offset indexing relative to `MAX_LPC_ORDER`:
```c
sLPC_Q14[MAX_LPC_ORDER + i - 1]  // accesses filter history
```
Rust approach: store `sLPC_Q14` as a `Vec<i32>` or array of size `MAX_LPC_ORDER + subfr_length`; index arithmetic is straightforward but needs careful bounds checking.

**`PLC.c` — aliased buffer trick:**
```c
// SMALL_FOOTPRINT: sLTP overlaps end of sLTP_Q14 for stack savings
sLTP = ((opus_int16*)&sLTP_Q14[...]) - psDec->ltp_mem_length;
```
Rust: allocate separate buffers. The aliasing trick is a stack optimization not needed in Rust.

### 10.2 VARDECL / ALLOC Stack Allocation

The C code uses `VARDECL` + `ALLOC` macros for variable-length stack arrays (VLAs or `alloca`). Rust equivalents:
- Small fixed-size: `[0i32; MAX_FRAME_LENGTH]` on stack
- Dynamic size: `Vec<i32>` (or `SmallVec` to avoid heap for common cases)
- Critical for `decode_core`: `sLTP` (up to 320 elements), `sLTP_Q15` (up to 640), `res_Q14` (up to 80), `sLPC_Q14` (up to 96)

### 10.3 Macro-Generated Arithmetic

All `silk_SMULWB`, `silk_SMLAWB`, `silk_RSHIFT_ROUND`, etc. are macros wrapping fixed-point arithmetic. They must be implemented as `#[inline]` Rust functions with **identical** overflow/rounding semantics:

```rust
// silk_SMLAWB(a32, b32, c16) = a + ((b * (c as i32)) >> 16)
#[inline(always)]
fn silk_smlawb(a: i32, b: i32, c: i16) -> i32 {
    a.wrapping_add(((b as i64 * c as i64) >> 16) as i32)
}
```

**Critical**: `silk_SMULWB` and `silk_SMLAWB` truncate toward -∞ (arithmetic right shift), NOT round-to-nearest. The rounding biases in the prediction loops (`LTP_pred_Q13 = 2`, `LPC_pred_Q10 = LPC_order/2`) compensate for this.

### 10.4 Wrapping vs Saturating Arithmetic

- Most intermediate calculations use **wrapping** arithmetic (`silk_ADD32`, `silk_SUB32`, `silk_MUL`)
- Explicit saturation only at specific points: `silk_ADD_SAT32`, `silk_LSHIFT_SAT32`, `silk_SAT16`
- The `silk_ADD32_ovflw` in excitation generation explicitly allows overflow:
  ```c
  rand_seed = silk_ADD32_ovflw(rand_seed, pulses[i]);  // intentional wrap
  ```
  Rust: use `i32::wrapping_add()`

### 10.5 The `silk_RAND` PRNG

```c
#define silk_RAND(seed) (silk_MLA_ovflw(907633515, (seed), 196314165))
// = 907633515 + seed * 196314165  (wrapping)
```
This is a linear congruential generator. Must use wrapping multiplication and addition in Rust.

### 10.6 Struct Reset Pattern

```c
silk_memset(&psDec->SILK_DECODER_STATE_RESET_START, 0,
    sizeof(silk_decoder_state) - offsetof(silk_decoder_state, SILK_DECODER_STATE_RESET_START));
```
This uses `#define SILK_DECODER_STATE_RESET_START prev_gain_Q16` as a field marker to zero everything from that field onward. In Rust, implement as a method that explicitly zeroes all fields after the OSCE section (or use a nested struct with `Default::default()`).

### 10.7 Conditional Compilation

Key `#ifdef` blocks to handle in the Rust port:
- `ENABLE_OSCE` — OSCE neural enhancement (skip for initial port)
- `ENABLE_OSCE_BWE` — Bandwidth extension (skip for initial port)
- `ENABLE_DEEP_PLC` — LPCNet deep PLC (skip for initial port)
- `ENABLE_QEXT` — Extended quality mode with 96kHz support (skip for initial port)
- `SMALL_FOOTPRINT` — Stack optimization hacks (ignore in Rust)

For the initial port, target the baseline codec without DNN features. The OSCE/deep-PLC paths can be added later as feature flags.

### 10.8 Function Pointer Dispatch (Resampler)

```c
S->resampler_function = USE_silk_resampler_private_up2_HQ_wrapper; // integer enum
// Dispatched via switch in silk_resampler()
```
Rust: use an enum with match dispatch, or trait objects. The match approach is cleaner and zero-cost:
```rust
enum ResamplerMethod { Copy, Up2HQ, IirFir, DownFir }
```

### 10.9 Union in Resampler State

```c
union { opus_int32 i32[...]; opus_int16 i16[...]; } sFIR;
```
`i32` is used by `down_FIR`, `i16` by `IIR_FIR`. Never both simultaneously. Rust: use an enum or two separate arrays (small cost, much safer).

### 10.10 Negative Array Indexing

In the LTP prediction loop:
```c
pred_lag_ptr = &sLTP_Q15[sLTP_buf_idx - lag + LTP_ORDER/2];
// Then: pred_lag_ptr[-1], pred_lag_ptr[-2], etc.
```
Rust: compute the base index and use `array[base - 1]`, `array[base - 2]`, etc. Ensure the base index is always ≥ `LTP_ORDER/2` (guaranteed by `start_idx > 0` assertion).

### 10.11 Biquad Filter Coefficient Splitting

```c
A0_L_Q28 = (-A_Q28[0]) & 0x00003FFF;     // lower 14 bits
A0_U_Q28 = silk_RSHIFT(-A_Q28[0], 14);   // upper 18 bits
```
This splitting avoids 32×32→64 multiplication by using two 32×16 multiplies. In Rust, the same pattern works with `& 0x3FFF` and `>> 14`. Note the negation happens *before* splitting.

### 10.12 The `silk_LPC_fit` Function

Called in `NLSF2A` to convert `a32_QA1` (Q17) to `a_Q12` (Q12). This involves right-shifting with rounding and clamping to `i16` range. The function is not in the files listed but is a dependency — search for it in `silk/LPC_fit.c` or similar.

### 10.13 Interleaved Stereo Output

```c
samplesOut[n + 2 * i] = resample_out_ptr[i];  // channel n at position n + 2*i
```
Standard interleaving: `[L0, R0, L1, R1, ...]`. Rust: straightforward with index arithmetic or `chunks_exact_mut`.

### 10.14 SILK_DWORD_ALIGN

```c
silk_DWORD_ALIGN opus_int16 PredCoef_Q12[2][MAX_LPC_ORDER];
```
This is `__attribute__((aligned(4)))` or similar. In Rust, `repr(align(4))` on a wrapper type, or just rely on the allocator (array of `i16` in a struct will typically be aligned fine for performance).
