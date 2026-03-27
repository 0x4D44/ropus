Now I have all the information I need. Let me produce the document.

# Opus Encoder Module — Architecture Documentation

## 1. Purpose

The **opus_encoder** module is the top-level encoding entry point of the Opus codec. It sits above both the SILK and CELT sub-codecs and orchestrates the complete encoding pipeline:

- Accepts raw PCM audio (int16, int32/24-bit, or float)
- Selects the encoding mode (SILK-only, CELT-only, or Hybrid) per frame based on signal analysis, bitrate, and application hints
- Decides bandwidth (narrowband through fullband), channel count (mono/stereo), frame size, DTX, FEC, and redundancy
- Runs an MLP/GRU-based **tonality analysis** neural network to classify speech vs. music and estimate bandwidth
- Invokes the SILK encoder for the low-frequency band (≤8 kHz) and CELT encoder for the high-frequency band
- Produces a standards-compliant Opus packet with TOC byte, optional redundancy frames, optional DRED extension, and optional padding

### Sub-modules

| File | Role |
|------|------|
| `opus_encoder.c` | Top-level encoder: mode selection, bit allocation, SILK/CELT dispatch, packet assembly |
| `analysis.c` | Tonality/bandwidth analysis: FFT-based feature extraction, speech/music classification |
| `analysis.h` | `TonalityAnalysisState` struct and analysis API |
| `mlp.c` | Dense and GRU layer inference for the analysis neural network |
| `mlp.h` | Layer struct definitions (`AnalysisDenseLayer`, `AnalysisGRULayer`) |
| `mlp_data.c` | Quantized int8 weights and biases for the 3-layer network |
| `tansig_table.h` | 201-entry lookup table for tanh approximation (used only in `#ifdef` training path) |

---

## 2. Public API

### 2.1 `opus_encoder_get_size`

```c
int opus_encoder_get_size(int channels);
```

Returns the total byte size needed for an `OpusEncoder` allocation (including embedded SILK and CELT encoders). Internally calls `opus_encoder_init(NULL, ...)` which returns the size when `st == NULL`.

- **channels**: 1 or 2
- **Returns**: Size in bytes, or 0 on invalid input

### 2.2 `opus_encoder_create`

```c
OpusEncoder *opus_encoder_create(opus_int32 Fs, int channels, int application, int *error);
```

Heap-allocates and initializes an encoder.

- **Fs**: Sample rate — 8000, 12000, 16000, 24000, 48000 (96000 with QEXT)
- **channels**: 1 or 2
- **application**: `OPUS_APPLICATION_VOIP`, `OPUS_APPLICATION_AUDIO`, `OPUS_APPLICATION_RESTRICTED_LOWDELAY`, `OPUS_APPLICATION_RESTRICTED_SILK`, `OPUS_APPLICATION_RESTRICTED_CELT`
- **error**: Optional output error code
- **Returns**: Pointer to encoder, or NULL on failure

### 2.3 `opus_encoder_init`

```c
int opus_encoder_init(OpusEncoder* st, opus_int32 Fs, int channels, int application);
```

Initializes a pre-allocated encoder (or computes required size when `st == NULL`).

**Dual-purpose pattern** (critical for porting): when `st == NULL`, the function computes and returns the total memory layout size. When `st != NULL`, it initializes the struct and returns `OPUS_OK`.

### 2.4 `opus_encode` / `opus_encode_float` / `opus_encode24`

```c
opus_int32 opus_encode(OpusEncoder *st, const opus_int16 *pcm, int frame_size,
                       unsigned char *data, opus_int32 max_data_bytes);

opus_int32 opus_encode_float(OpusEncoder *st, const float *pcm, int frame_size,
                             unsigned char *data, opus_int32 max_data_bytes);

opus_int32 opus_encode24(OpusEncoder *st, const opus_int32 *pcm, int frame_size,
                         unsigned char *data, opus_int32 max_data_bytes);
```

Public encoding functions. All three convert input to `opus_res` format, then call the shared `opus_encode_native()`. The `frame_size` is in samples per channel.

- **Returns**: Number of bytes written to `data`, or a negative error code

### 2.5 `opus_encoder_ctl`

```c
int opus_encoder_ctl(OpusEncoder *st, int request, ...);
```

Variadic control interface. Supports ~30 get/set requests for bitrate, bandwidth, complexity, VBR, FEC, DTX, signal type, frame duration, phase inversion, DRED, QEXT, reset, etc. Uses `va_arg` dispatch.

### 2.6 `opus_encoder_destroy`

```c
void opus_encoder_destroy(OpusEncoder *st);
```

Frees a heap-allocated encoder. Just calls `opus_free(st)`.

---

## 3. Internal State

### 3.1 `OpusEncoder` (main struct)

The struct is a **single contiguous allocation** that contains the Opus-level state, followed by the SILK encoder state, followed by the CELT encoder state. Sub-encoders are accessed via byte offsets:

```
[OpusEncoder fields][padding][SILK encoder][CELT encoder]
```

```c
struct OpusEncoder {
    // --- Sub-encoder offsets (byte offsets from start of struct) ---
    int celt_enc_offset;
    int silk_enc_offset;

    // --- SILK control ---
    silk_EncControlStruct silk_mode;  // Full SILK parameter block
    // (DRED encoder embedded here when ENABLE_DRED)

    // --- User-configurable parameters ---
    int application;          // VOIP, AUDIO, RESTRICTED_*
    int channels;             // API-level channel count (1 or 2)
    int delay_compensation;   // Lookahead compensation in samples (Fs/250 = 4ms)
    int force_channels;       // OPUS_AUTO or forced 1/2
    int signal_type;          // OPUS_AUTO, OPUS_SIGNAL_VOICE, OPUS_SIGNAL_MUSIC
    int user_bandwidth;       // OPUS_AUTO or explicit OPUS_BANDWIDTH_*
    int max_bandwidth;        // Upper limit on bandwidth
    int user_forced_mode;     // OPUS_AUTO or forced MODE_*
    int voice_ratio;          // -1 = unknown, 0..100 = speech probability
    opus_int32 Fs;            // Sample rate (API level)
    int use_vbr;              // VBR enabled (default 1)
    int vbr_constraint;       // Constrained VBR (default 1)
    int variable_duration;    // Frame duration control
    opus_int32 bitrate_bps;   // Effective bitrate (after capping)
    opus_int32 user_bitrate_bps; // User-requested bitrate
    int lsb_depth;            // Input bit depth (8-24, default 24)
    int encoder_buffer;       // Samples in delay buffer (Fs/100 = 10ms)
    int lfe;                  // LFE channel mode
    int arch;                 // CPU architecture flags
    int use_dtx;              // DTX enabled
    int fec_config;           // FEC mode (0=off, 1=on, 2=on but don't switch mode)

    // --- Tonality analysis (float API only) ---
    TonalityAnalysisState analysis;

    // ===== Fields below here are cleared on OPUS_RESET_STATE =====
    // (marked by OPUS_ENCODER_RESET_START)

    int stream_channels;      // Actual channels to encode (after mono decision)
    opus_int16 hybrid_stereo_width_Q14;  // Stereo width parameter
    opus_int32 variable_HP_smth2_Q15;    // HP filter cutoff smoother
    opus_val16 prev_HB_gain;  // Previous high-band gain
    opus_val32 hp_mem[4];     // HP filter state (2 per channel)
    int mode;                 // Current mode (SILK_ONLY, CELT_ONLY, HYBRID)
    int prev_mode;            // Previous frame's mode
    int prev_channels;        // Previous frame's stream channels
    int prev_framesize;       // Previous frame size
    int bandwidth;            // Current bandwidth setting
    int auto_bandwidth;       // Auto-selected bandwidth (before user override)
    int silk_bw_switch;       // SILK bandwidth switch pending
    int first;                // First frame flag
    celt_glog *energy_masking; // Surround masking pointer (external)
    StereoWidthState width_mem; // Stereo width estimation state
    int detected_bandwidth;   // Analysis-detected bandwidth
    int nb_no_activity_ms_Q1; // DTX silence counter (Q1 = half-ms units)
    opus_val32 peak_signal_energy;  // Peak energy tracker for DTX
    int nonfinal_frame;       // Multi-frame: not the last sub-frame
    opus_uint32 rangeFinal;   // Range coder final state
    opus_res delay_buffer[MAX_ENCODER_BUFFER*2]; // Delay compensation buffer (MUST be last)
};
```

**Layout optimization**: The `delay_buffer` is the last field and may be partially or completely omitted depending on channel count and application mode. The `base_size` computation in `opus_encoder_init` adjusts:

- `RESTRICTED_SILK` or `RESTRICTED_CELT`: no delay buffer at all
- Mono: half the delay buffer
- Stereo: full delay buffer

### 3.2 `StereoWidthState`

```c
typedef struct {
    opus_val32 XX, XY, YY;       // Smoothed correlation accumulators
    opus_val16 smoothed_width;   // Low-pass filtered width estimate
    opus_val16 max_follower;     // Peak follower on smoothed width
} StereoWidthState;
```

### 3.3 `TonalityAnalysisState`

```c
typedef struct {
    int arch;
    int application;
    opus_int32 Fs;
    // --- Fields below TONALITY_ANALYSIS_RESET_START are cleared on reset ---
    float angle[240];          // Phase angles from previous FFT frame
    float d_angle[240];        // Phase angle derivatives
    float d2_angle[240];       // Second derivatives of phase angles
    opus_val32 inmem[720];     // 30 ms analysis buffer at 24 kHz
    int mem_fill;              // Samples currently in inmem
    float prev_band_tonality[18];
    float prev_tonality;
    int prev_bandwidth;
    float E[8][18];            // Band energies for last 8 frames
    float logE[8][18];         // Log band energies
    float lowE[18];            // Running minimum log energy per band
    float highE[18];           // Running maximum log energy per band
    float meanE[19];           // Mean energy per band (with decay)
    float mem[32];             // BFCC history (4 frames × 8 coefficients)
    float cmean[8];            // Running mean of BFCCs
    float std[9];              // Running variance of features
    float Etracker;            // Loudness tracker
    float lowECount;           // Low-energy frame counter
    int E_count;               // Circular index into E[][] (mod 8)
    int count;                 // Total frames analyzed
    int analysis_offset;       // Offset into analysis PCM
    int write_pos, read_pos;   // Circular buffer indices into info[]
    int read_subframe;         // Sub-frame position for reading
    float hp_ener_accum;       // High-pass energy accumulator
    int initialized;
    float rnn_state[32];       // GRU hidden state (MAX_NEURONS)
    opus_val32 downmix_state[3]; // Resampler state
    AnalysisInfo info[100];    // Circular buffer of analysis results (DETECT_SIZE)
} TonalityAnalysisState;
```

### 3.4 `AnalysisInfo`

```c
typedef struct {
    int valid;
    float tonality;
    float tonality_slope;
    float noisiness;
    float activity;
    float music_prob;          // Music probability (current frame)
    float music_prob_min;      // Min threshold for switching to music
    float music_prob_max;      // Max threshold for switching from music
    int   bandwidth;           // Detected bandwidth (in band index, 0-20)
    float activity_probability; // VAD probability
    float max_pitch_ratio;     // Below-pitch / above-pitch energy ratio
    unsigned char leak_boost[19]; // Q6 leakage boost per band
} AnalysisInfo;
```

### 3.5 MLP Layer Structs

```c
typedef struct {
    const opus_int8 *bias;
    const opus_int8 *input_weights;
    int nb_inputs;
    int nb_neurons;
    int sigmoid;               // 0 = tansig, 1 = sigmoid activation
} AnalysisDenseLayer;

typedef struct {
    const opus_int8 *bias;
    const opus_int8 *input_weights;
    const opus_int8 *recurrent_weights;
    int nb_inputs;
    int nb_neurons;
} AnalysisGRULayer;
```

---

## 4. Algorithm

### 4.1 Top-Level Encode Flow (`opus_encode_native`)

```
1. Input validation (frame size, buffer size)
2. Retrieve sub-encoder pointers via byte offset arithmetic
3. Digital silence detection (celt_maxabs on PCM)
4. Tonality analysis (if complexity >= 7/10 and Fs >= 16 kHz):
   a. run_analysis() → tonality_analysis() → tonality_get_info()
   b. Produces AnalysisInfo with music_prob, bandwidth, activity
5. Voice/music estimation:
   - signal_type forced → use directly
   - Auto → use analysis_info.music_prob (with hysteresis)
   - Default: VOIP=115 (high voice), AUDIO=48 (neutral)
6. Peak signal energy tracking (for DTX pseudo-SNR)
7. Stereo width computation (if stereo)
8. Bitrate computation (user → effective, capping to max_data_bytes)
9. DRED bitrate allocation (if enabled)
10. Low-bitrate PLC-frame shortcut (< ~6 kbps)
11. Equivalent rate normalization (for decision thresholds)
12. Mono/stereo decision (rate vs. threshold with hysteresis)
13. Mode selection (SILK_ONLY / CELT_ONLY / HYBRID):
    - Based on equiv_rate vs. mode_thresholds, interpolated by voice_est
    - VOIP bias (+8000), FEC bias, DTX bias
    - Frame size override (< 10ms → force CELT)
14. Redundancy decision (mode transitions need overlap)
15. Bandwidth selection:
    - Rate-dependent thresholds with hysteresis
    - Clamp to Nyquist, max_bandwidth, user_bandwidth
    - Analysis-detected bandwidth (conservative)
    - SILK bandwidth switch coordination
16. FEC decision (decide_fec)
17. Multi-frame handling (>20ms non-SILK, >60ms):
    - Split into sub-frames, encode individually
    - Repacketize with opus_repacketizer
18. Delegate to opus_encode_frame_native() for single-frame encoding
```

### 4.2 Single-Frame Encode (`opus_encode_frame_native`)

```
1. Activity decision (silence, analysis VAD, or energy-based)
2. SILK bandwidth switch handling (prefill)
3. Redundancy byte computation
4. Bit target = min(max_data_bytes*8, bitrate_to_bits) - redundancy - 8
5. Skip TOC byte: data += 1; ec_enc_init on remaining space
6. Copy delay buffer into pcm_buf, prepending lookahead samples
7. HP/DC filtering:
   - VOIP: variable HP cutoff (silk smoother → cutoff_Hz)
   - Other: DC reject (6.3*cutoff/Fs first-order IIR)
8. NaN protection (float API)
9. DRED latent computation (if enabled)
10. SILK processing (if mode != CELT_ONLY):
    a. Bit allocation between SILK and CELT (via rate table interpolation)
    b. HB_gain attenuation for hybrid
    c. Surround masking rate offset
    d. Configure SILK parameters (sample rate, payload_ms, max bits)
    e. Prefill with smooth onset (on mode transition)
    f. silk_Encode() → compressed SILK data into ec_enc
    g. Extract internal SILK bandwidth
11. CELT processing (if mode != SILK_ONLY):
    a. Configure end band (13-21 based on bandwidth)
    b. Set VBR, bitrate, prediction
    c. Prefill on mode transition
    d. celt_encode_with_ec() → compressed CELT data sharing the ec_enc
12. Stereo width reduction (fade at low bitrates)
13. Redundancy frame encoding (5ms CELT frame):
    - CELT→SILK: encode before main CELT frame
    - SILK→CELT: encode after (with prefill)
14. Write TOC byte: gen_toc(mode, framerate, bandwidth, channels)
15. DTX decision (silence counter thresholds)
16. DRED extension data (if enabled, appended via padding)
17. CBR padding (if !VBR)
18. Update state (prev_mode, prev_channels, prev_framesize, first=0)
19. Return total packet bytes
```

### 4.3 Tonality Analysis Algorithm (`tonality_analysis`)

The analysis runs on 20ms frames at 24 kHz (480 samples). Input at other sample rates is resampled via `downmix_and_resample()`.

```
1. Downmix to mono, resample to 24 kHz, fill 720-sample buffer
2. Apply analysis window (240 samples × 2 halves)
3. 480-point FFT (reusing CELT's kiss_fft)
4. For each bin i (1..239):
   a. Extract phase angle via atan2
   b. Compute 1st and 2nd phase derivatives
   c. Tonality = 1/(1 + 40·16·π⁴·avg_mod) - 0.015
   d. Noisiness = |mod| (closeness to integer phase derivative)
5. Per-band analysis (18 bands, defined by tbands[]):
   a. Band energy E, tonal energy tE, noise energy nE
   b. Log energy tracking (lowE, highE, meanE with decay)
   c. Stationarity = (L1/√(N·L2))⁴  (ratio of norms)
   d. Band tonality = max(tE/E, stationarity × prev_tonality)
   e. Leakage compensation (forward+backward propagation)
6. Bandwidth detection:
   - Band active if energy > 90dB below peak AND above noise floor
   - Masked-band pruning
7. Spectral variability (minimum pairwise distance across 8 frames)
8. BFCC computation (16-point DCT on log energies → 8 coefficients)
9. Feature vector (25 dimensions):
   - BFCC deltas (temporal derivatives via FIR)
   - BFCC second derivatives
   - Feature standard deviations
   - Spectral variability, tonality, activity, stationarity, slope, lowECount
10. MLP/GRU inference:
    - layer0: Dense(25→32, tansig)
    - layer1: GRU(32→24) — recurrent, hidden state persists
    - layer2: Dense(24→2, sigmoid) → [music_prob, activity_probability]
11. Store result in circular AnalysisInfo buffer
```

### 4.4 Speech/Music Transition Algorithm (`tonality_get_info`)

Uses a **badness function minimization** with look-ahead to find optimal switching points. The key formula for the switching threshold:

```
T = (Σ v_i·p_i + S·(v_k - v_0)) / (Σ v_i)
```

Where `v_i` = VAD probability, `p_i` = music probability, `S` = transition penalty (10). Computes `music_prob_min` (speech→music threshold) and `music_prob_max` (music→speech threshold) by searching over all look-ahead positions.

---

## 5. Data Flow

### 5.1 Input → Output Pipeline

```
PCM input (int16/float/int32)
  ↓ Convert to opus_res (in-place or via temp buffer)
  ↓ frame_size_select() — apply variable_duration
  ↓ opus_encode_native()
      ├─→ Analysis path: downmix → resample → FFT → features → MLP/GRU → AnalysisInfo
      │
      ├─→ Mode/bandwidth/channel decisions
      │
      ├─→ Copy delay_buffer[lookahead] + new PCM → pcm_buf
      │
      ├─→ HP/DC filter (pcm_buf, in-place)
      │
      ├─→ SILK path: pcm_buf → silk_Encode() → ec_enc bitstream
      │
      ├─→ CELT path: pcm_buf → celt_encode_with_ec() → ec_enc bitstream
      │
      ├─→ Redundancy: pcm_buf tail → celt_encode_with_ec() → appended bytes
      │
      └─→ TOC byte + encoded data + (DRED extension) + (padding) → output packet
```

### 5.2 Buffer Layouts

**`delay_buffer`** (persistent in OpusEncoder):
- Size: `encoder_buffer * channels` samples
- Contains: past `encoder_buffer` samples for lookahead and prefill
- Updated every frame: slide left by `frame_size`, append new PCM

**`pcm_buf`** (stack allocated):
- Size: `(total_buffer + frame_size) * channels` samples
- Layout: `[delay_compensation samples from delay_buffer][frame_size new samples]`
- HP/DC filter applied in-place to the new-sample portion

**Analysis `inmem`** (in TonalityAnalysisState):
- Size: 720 samples (30 ms at 24 kHz)
- Circular-ish: after processing, slides left by 480, keeps 240 overlap samples

---

## 6. Numerical Details

### 6.1 Fixed-Point Formats

| Variable | Format | Notes |
|----------|--------|-------|
| `hybrid_stereo_width_Q14` | Q14 | 0 to 16384 (0.0 to 1.0) |
| `variable_HP_smth2_Q15` | Q15 | Log-scale HP cutoff frequency |
| `hp_mem[4]` | Q12 (fixed), float (float) | Biquad filter state |
| `silk_mode.stereoWidth_Q14` | Q14 | Stereo width for SILK |
| `nb_no_activity_ms_Q1` | Q1 | Half-millisecond resolution |
| `B_Q28, A_Q28` | Q28 | Biquad filter coefficients |
| `Fc_Q19` | Q19 | Normalized cutoff frequency |
| `r_Q28, r_Q22` | Q28, Q22 | Filter pole radius |
| `voice_est` | Q7 | 0-127 voice probability |

### 6.2 HP/Biquad Filter (Fixed-Point)

The `silk_biquad_res` in fixed-point uses a Direct Form II Transposed structure with coefficient splitting to avoid overflow:

```c
A0_L_Q28 = (-A_Q28[0]) & 0x00003FFF;   // lower 14 bits
A0_U_Q28 = silk_RSHIFT(-A_Q28[0], 14); // upper bits
out32_Q14 = silk_LSHIFT(silk_SMLAWB(S[0], B_Q28[0], inval), 2);
```

The coefficient is split into upper and lower parts because `silk_SMULWB` operates on the upper 16 bits, so splitting avoids precision loss in 32-bit intermediate products.

### 6.3 DC Reject Filter (Fixed-Point)

```c
shift = celt_ilog2(Fs / (cutoff_Hz * 4));
x = SHL32(x, 14 - RES_SHIFT);         // Scale to Q14
y = x - hp_mem[2*c];                   // High-pass
hp_mem[2*c] += PSHR32(x - hp_mem[2*c], shift);  // LP update
out = SATURATE(PSHR32(y, 14 - RES_SHIFT), 32767);
```

### 6.4 MLP Weight Scaling

All neural network weights are stored as `opus_int8` (range -128..127). After the GEMM accumulation, outputs are scaled by `WEIGHTS_SCALE = 1/128`:

```c
output[i] *= WEIGHTS_SCALE;  // = 1.0f/128
```

This means the effective weight range is -1.0 to ~0.992.

### 6.5 Activation Functions

**tansig_approx** (rational polynomial, not lookup-based):
```c
num = ((N2*X² + N1)*X² + N0)
den = ((D2*X² + D1)*X² + D0)
return clamp(num*x/den, -1, 1)
```

**sigmoid_approx**:
```c
return 0.5 + 0.5 * tansig_approx(0.5 * x)
```

The `tansig_table[201]` in `tansig_table.h` is NOT used at runtime — it exists for training/verification. The runtime uses the polynomial approximation.

### 6.6 Stereo Width Computation

Uses unrolled-by-4 correlation accumulation:
```c
xx += Σ x²,  xy += Σ x·y,  yy += Σ y²   (with right-shift by frame_size_log2-2)
```

Then exponentially smoothed, with inter-channel correlation and loudness difference:
```c
corr = XY / (√XX · √YY)
ldiff = |⁴√XX - ⁴√YY| / (⁴√XX + ⁴√YY)
width = √(1 - corr²) · ldiff
```

### 6.7 Equivalent Rate Normalization

The `compute_equiv_rate()` function adjusts the raw bitrate to account for overhead, creating a comparable "20ms complexity-10 VBR" equivalent:

- Frame overhead: subtract `(40*channels+20) * (frame_rate - 50)` for rates > 50 fps
- CBR penalty: ~8% reduction
- Complexity penalty: scale by `(90+complexity)/100`
- SILK loss penalty: `equiv -= equiv*loss/(6*loss+10)`
- CELT no-pitch penalty: 10% for complexity < 5

---

## 7. Dependencies

### 7.1 What opus_encoder calls

| Module | Functions Called |
|--------|----------------|
| SILK | `silk_Get_Encoder_Size`, `silk_InitEncoder`, `silk_Encode`, `silk_lin2log`, `silk_log2lin`, `silk_SMLAWB`, `silk_SMULWB`, etc. |
| CELT | `celt_encoder_init`, `celt_encode_with_ec`, `celt_encoder_ctl`, `celt_encoder_get_size`, `celt_inner_prod`, `celt_maxabs_res`, `celt_sqrt`, `celt_exp2`, `frac_div32` |
| Range coder | `ec_enc_init`, `ec_enc_bit_logp`, `ec_enc_uint`, `ec_enc_shrink`, `ec_enc_done`, `ec_tell` |
| FFT | `opus_fft` (via analysis path) |
| Packet | `opus_repacketizer_init`, `opus_repacketizer_cat`, `opus_repacketizer_out_range_impl`, `opus_packet_pad`, `opus_packet_pad_impl` |
| Math | `fast_atan2f`, `celt_ilog2`, `float2int` |
| DRED (optional) | `dred_encoder_init`, `dred_compute_latents`, `dred_encode_silk_frame`, `compute_quantizer` |

### 7.2 What calls opus_encoder

- Application code via `opus_encode()`, `opus_encode_float()`, `opus_encode24()`
- Multistream encoder (`opus_multistream_encoder.c`) — calls the same API plus sets `energy_masking` and `lfe` via ctl

---

## 8. Constants and Tables

### 8.1 Bandwidth Thresholds (bitrate in bps)

```c
// [threshold, hysteresis] pairs for NB↔MB, MB↔WB, WB↔SWB, SWB↔FB
mono_voice_bandwidth_thresholds  = {9000,700, 9000,700, 13500,1000, 14000,2000}
mono_music_bandwidth_thresholds  = {9000,700, 9000,700, 11000,1000, 12000,2000}
stereo_voice_bandwidth_thresholds = {9000,700, 9000,700, 13500,1000, 14000,2000}
stereo_music_bandwidth_thresholds = {9000,700, 9000,700, 11000,1000, 12000,2000}
```

Interpolated at runtime by `voice_est²` between voice and music thresholds.

### 8.2 Mode Thresholds

```c
mode_thresholds[2][2] = {
    {64000, 10000},  // mono:   [voice, music]
    {44000, 10000},  // stereo: [voice, music]
};
```

Below threshold → SILK; above → CELT. Hysteresis of ±4000. VOIP bias of +8000.

### 8.3 Stereo Thresholds

```c
stereo_voice_threshold = 19000;
stereo_music_threshold = 17000;
```

### 8.4 FEC Thresholds

```c
fec_thresholds[] = {12000,1000, 14000,1000, 16000,1000, 20000,1000, 22000,1000};
```

Indexed by `bandwidth - OPUS_BANDWIDTH_NARROWBAND`, scaled by `(125 - min(loss, 25)) * 0.01`.

### 8.5 Hybrid SILK Rate Table

```c
rate_table[][5] = {
    // total_rate, SILK_no_FEC_10ms, SILK_no_FEC_20ms, SILK_FEC_10ms, SILK_FEC_20ms
    {0,     0,     0,     0,     0},
    {12000, 10000, 10000, 11000, 11000},
    {16000, 13500, 13500, 15000, 15000},
    {20000, 16000, 16000, 18000, 18000},
    {24000, 18000, 18000, 21000, 21000},
    {32000, 22000, 22000, 28000, 28000},
    {64000, 38000, 38000, 50000, 50000}
};
```

Piecewise linear interpolation between table entries.

### 8.6 Analysis Constants

```c
NB_FRAMES = 8           // Frames of energy history
NB_TBANDS = 18           // Tonal analysis bands
ANALYSIS_BUF_SIZE = 720  // 30 ms at 24 kHz
DETECT_SIZE = 100        // Circular buffer for AnalysisInfo
ANALYSIS_COUNT_MAX = 10000
NB_TONAL_SKIP_BANDS = 9  // Skip first 9 bands for frame tonality

tbands[19] = {4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 136, 160, 192, 240}
```

### 8.7 MLP Network Topology

```
layer0: Dense  (25 inputs → 32 neurons, tansig activation)
         800 weights (25×32), 32 biases
layer1: GRU    (32 inputs → 24 neurons)
         2304 input weights (32×3×24), 1728 recurrent weights (24×3×24), 72 biases (3×24)
layer2: Dense  (24 inputs → 2 neurons, sigmoid activation)
         48 weights (24×2), 2 biases
```

Output: `[music_probability, activity_probability]`, both in [0, 1].

### 8.8 Analysis Window and DCT Table

- `analysis_window[240]`: Asymmetric raised-cosine window for the 480-point FFT (240 unique values, symmetric halves use same weights)
- `dct_table[128]`: 8×16 DCT-II matrix for BFCC computation from 16 log-energy bands

---

## 9. Edge Cases

### 9.1 Error Conditions

| Condition | Response |
|-----------|----------|
| Invalid Fs (not 8/12/16/24/48k) | `OPUS_BAD_ARG` |
| Invalid channels (not 1 or 2) | `OPUS_BAD_ARG` |
| Invalid application | `OPUS_BAD_ARG` |
| `frame_size <= 0` or `max_data_bytes <= 0` | `OPUS_BAD_ARG` |
| 100ms frame in 1 byte | `OPUS_BUFFER_TOO_SMALL` |
| SILK busts target (ec_tell > max*8) | Returns 1-byte PLC packet |
| Internal encoder errors | `OPUS_INTERNAL_ERROR` |
| SILK returns 0 bytes | Returns 1-byte DTX packet |

### 9.2 Special Input Handling

- **Digital silence**: Detected via `celt_maxabs_res`. On silence, `voice_ratio` is preserved from last non-silent frame. Analysis copies previous results.
- **NaN/Inf input (float API)**: Entire PCM buffer cleared, HP filter state reset. Analysis returns `valid=0`.
- **2.5 ms at 12 kHz**: Frame size (30 samples) not divisible by 4; stereo width discards last 2 samples.
- **First frame**: `st->first = 1` disables hysteresis on bandwidth selection, forces `MODE_HYBRID` initial state.
- **Very low bitrate (< 3 bytes or < 6 kbps)**: Emits PLC frame (TOC only, no coded data).
- **Multi-frame (> 20ms CELT/Hybrid, > 60ms SILK)**: Split into sub-frames, encoded individually, then repacketized.

### 9.3 Mode Transitions

Mode transitions (SILK↔CELT) require a **redundancy frame** — a 5ms CELT frame that provides seamless overlap:
- **CELT→SILK**: Redundancy frame encoded first (before SILK prefill)
- **SILK→CELT**: Redundancy frame encoded after main frame, from the tail of pcm_buf
- SILK is re-initialized on entering SILK mode from CELT (`prefill=1`)
- CELT is reset and prefilled (2.5ms) on entering CELT mode from SILK

### 9.4 Stereo→Mono Transition

Delayed by 2 frames (`silk_mode.toMono`) to allow SILK to perform a smooth downmix internally.

---

## 10. Porting Notes

### 10.1 Single-Allocation Memory Layout

The `OpusEncoder` struct uses C pointer arithmetic to embed SILK and CELT encoders at computed byte offsets within the same allocation:

```c
silk_enc = (char*)st + st->silk_enc_offset;
celt_enc = (CELTEncoder*)((char*)st + st->celt_enc_offset);
```

**Rust approach**: Use a `Vec<u8>` or boxed byte buffer for the raw allocation, with typed accessors. Alternatively, refactor to hold `SilkEncoder` and `CeltEncoder` as owned fields (preferred — eliminates unsafe pointer arithmetic at the cost of differing from C's allocation pattern). The C design exists purely for single-`malloc` allocation.

### 10.2 OPUS_ENCODER_RESET_START Macro

The reset logic uses a C trick to clear everything from a named field to the end of the sub-encoder region:

```c
#define OPUS_ENCODER_RESET_START stream_channels
start = (char*)&st->OPUS_ENCODER_RESET_START;
OPUS_CLEAR(start, st->silk_enc_offset - (start - (char*)st));
```

**Rust approach**: Group the resettable fields into a sub-struct and implement `Default` for it.

### 10.3 TONALITY_ANALYSIS_RESET_START

Same pattern in `TonalityAnalysisState`:
```c
#define TONALITY_ANALYSIS_RESET_START angle
char *start = (char*)&tonal->TONALITY_ANALYSIS_RESET_START;
OPUS_CLEAR(start, sizeof(TonalityAnalysisState) - (start - (char*)tonal));
```

**Rust approach**: Same sub-struct pattern.

### 10.4 Conditional Compilation

The source uses extensive `#ifdef` blocks:

| Define | Effect |
|--------|--------|
| `FIXED_POINT` | Fixed-point arithmetic (Q formats, `silk_SMLAWB`, etc.) |
| `DISABLE_FLOAT_API` | Removes `opus_encode_float`, tonality analysis |
| `ENABLE_DRED` | Deep Redundancy Encoder |
| `ENABLE_QEXT` | Quality Extension (96 kHz, larger packets) |
| `ENABLE_RES24` | 24-bit residual support |
| `ENABLE_OSCE_TRAINING_DATA` | Debug file output |
| `MLP_TRAINING` | Print MLP features to stdout |
| `FUZZING` | Random mode/channel decisions |

**Rust approach**: Use Cargo features. The primary port should support both fixed-point and float via generics or feature flags. DRED/QEXT can be gated behind features.

### 10.5 `opus_res` Type

`opus_res` is a typedef that changes based on build configuration:
- `FIXED_POINT` without `ENABLE_RES24`: `opus_int16`
- `FIXED_POINT` with `ENABLE_RES24`: `opus_int32`
- Float: `float`

This affects buffer sizes, conversion macros (`INT16TORES`, `FLOAT2RES`, `RES2VAL16`, `RES2INT16`), and arithmetic throughout.

**Rust approach**: Parameterize with a trait or newtype. For the initial port, pick one representation (likely float for simplicity, matching the float build).

### 10.6 `VARDECL` / `ALLOC` / Stack Allocation

The C code uses `VARDECL`/`ALLOC` macros for variable-length stack arrays (VLAs or `alloca`):

```c
VARDECL(opus_res, pcm_buf);
ALLOC(pcm_buf, (total_buffer+frame_size)*st->channels, opus_res);
```

**Rust approach**: Use `Vec` with pre-computed capacity. For hot paths, consider a scratch buffer pool or `SmallVec`.

### 10.7 Variadic `opus_encoder_ctl`

The C API uses `va_list` for the control interface:

```c
int opus_encoder_ctl(OpusEncoder *st, int request, ...);
```

**Rust approach**: Use an enum for request types:
```rust
enum EncoderCtl {
    SetBitrate(i32),
    GetBitrate,
    SetComplexity(i32),
    // ...
}
```

Or provide individual typed methods (`set_bitrate()`, `get_bitrate()`).

### 10.8 In-Place Buffer Mutation

Several operations modify buffers in-place:

- `hp_cutoff()` and `dc_reject()` can write to the same buffer they read from
- `stereo_fade()` and `gain_fade()` explicitly operate with `out == in`
- `OPUS_MOVE` (memmove) used when source and destination overlap in `delay_buffer`

**Rust approach**: These are fine with `&mut [T]` slices. For the overlap case, use `copy_within()`.

### 10.9 GRU Layer Weight Layout

The GRU interleaves weights for update (z), reset (r), and output (h) gates with stride `3*N`:

```c
stride = 3*N;
// Update gate: bias[0..N], weights[0..], recurrent[0..]
// Reset gate:  bias[N..2N], weights[N..], recurrent[N..]
// Output gate: bias[2N..3N], weights[2N..], recurrent[2N..]
```

The `gemm_accum` function indexes into the interleaved layout via the stride parameter and offset. This is column-major with the 3 gates interleaved per neuron.

### 10.10 `VERY_SMALL` Anti-Denormal

In the float DC reject filter:
```c
m0 = coef*x0 + VERY_SMALL + coef2*m0;
```

`VERY_SMALL` (typically `1e-30f` or similar) prevents denormalized floats which cause massive performance drops on some x86 CPUs. In Rust, the FTZ (flush-to-zero) flag on the MXCSR register is the proper solution, but adding the constant is simpler for bit-exactness.

### 10.11 Delay Buffer Tail Optimization

The `delay_buffer` is declared as the **last field** of `OpusEncoder` and its size is reduced based on channel count and application mode. The `base_size` calculation subtracts unused buffer space:

```c
if (channels == 1)
   base_size = align(base_size - MAX_ENCODER_BUFFER*sizeof(opus_res));
```

**Rust approach**: This is essentially a flexible array member optimization. In Rust, either always allocate the full size (simpler, wastes ~1.9 KB for mono) or use a separately-allocated `Vec<opus_res>`.

### 10.12 Bit-Exactness Critical Paths

The following require exact numerical matching with the C reference:

1. **HP/DC filter coefficients**: Q28 arithmetic with specific rounding (`silk_RSHIFT_ROUND`, `PSHR32`)
2. **TOC byte generation**: Bit-field layout is standardized
3. **Range coder interaction**: `ec_tell()` determines bit allocation decisions
4. **MLP inference**: int8 weights × float inputs with specific scaling — the `tansig_approx` polynomial must match exactly
5. **Stereo width**: The `frac_div32` and `celt_sqrt` chain must match
6. **Bandwidth/mode thresholds**: Integer arithmetic on `voice_est²`

The analysis and decision logic (mode selection, bandwidth, FEC, DTX) doesn't need bit-exact audio output, but the decisions must be identical for the same input to produce identical packets.
