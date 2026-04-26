Now I have all the information needed. Here is the complete architecture documentation.

---

# DNN LPCNet Module — Architecture Documentation

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

LPCNet is a neural speech codec integrated into Opus (1.4+). It combines classical linear predictive coding (LPC) with recurrent neural networks to achieve high-quality speech synthesis at very low bitrates. Within the Opus pipeline, LPCNet serves three roles:

1. **Feature Extraction (Encoder)**: Analyzes PCM audio to produce compact feature vectors (cepstral coefficients, pitch period, pitch correlation) that can be quantized and transmitted.

2. **Neural Synthesis (Synthesizer)**: Generates speech waveforms sample-by-sample from feature vectors using an autoregressive neural network with GRU (Gated Recurrent Unit) layers operating in the LPC excitation domain.

3. **Packet Loss Concealment (PLC)**: When packets are lost, a separate prediction network forecasts the next feature vectors, and FARGAN (a neural waveform generator) synthesizes the concealed audio.

The architecture follows a source-filter model: LPC coefficients define the filter (spectral envelope), while the neural network generates the excitation signal. Samples are processed in the mu-law domain (8-bit, 256 levels) to reduce the dynamic range the network must model.

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sample rate | 16 kHz | Fixed |
| Frame size | 160 samples | 10 ms per feature frame |
| Packet size | 640 samples | 4 frames = 40 ms |
| Compressed size | 8 bytes | Per packet |
| Feature vector | 20 floats (`NB_FEATURES`) | Core features per frame |
| Total features | 36 floats (`NB_TOTAL_FEATURES`) | Including LPC coefficients |
| LPC order | 16 (`LPC_ORDER`) | |
| Frequency bands | 18 (`NB_BANDS`) | Bark-scale bands |

---

## 2. Public API

### 2.1 Synthesis State (`LPCNetState`)

Low-level synthesis from feature vectors. This is the core neural vocoder.

```c
int  lpcnet_get_size(void);
// Returns sizeof(LPCNetState).

int  lpcnet_init(LPCNetState *st);
// Zero-initializes, builds sampling_logit_table, loads model weights
// (from compiled-in lpcnet_arrays unless USE_WEIGHTS_FILE defined),
// calls lpcnet_reset(). Returns 0 on success.

LPCNetState *lpcnet_create(void);
// opus_alloc + lpcnet_init. Returns pointer (caller frees via lpcnet_destroy).

void lpcnet_destroy(LPCNetState *lpcnet);
// opus_free.

void lpcnet_reset(LPCNetState *lpcnet);
// Clears all fields from LPCNET_RESET_START onward (preserves model weights
// and sampling_logit_table). Sets last_exc = lin2ulaw(0). Seeds PRNG with "LPCNet".

int  lpcnet_load_model(LPCNetState *st, const void *data, int len);
// Loads model weights from external binary blob. Returns 0 on success, -1 on error.

void lpcnet_synthesize(LPCNetState *st, const float *features,
                       opus_int16 *output, int N);
// Synthesizes N PCM samples from a feature vector (NB_FEATURES floats).
// Wrapper around lpcnet_synthesize_impl with preload=0.
```

### 2.2 Decoder State (`LPCNetDecState`)

Packet-level decoder wrapping the synthesizer. Decompresses 8-byte packets into 640 PCM samples.

```c
int  lpcnet_decoder_get_size(void);
int  lpcnet_decoder_init(LPCNetDecState *st);
LPCNetDecState *lpcnet_decoder_create(void);
void lpcnet_decoder_destroy(LPCNetDecState *st);

int  lpcnet_decode(LPCNetDecState *st, const unsigned char *buf,
                   opus_int16 *pcm);
// Decodes buf[LPCNET_COMPRESSED_SIZE=8] -> pcm[LPCNET_PACKET_SAMPLES=640].
// Returns 0 on success.
```

### 2.3 Encoder State (`LPCNetEncState`)

Feature extraction from PCM. Computes per-frame features and optionally produces compressed packets.

```c
int  lpcnet_encoder_get_size(void);
// Returns sizeof(LPCNetEncState).

int  lpcnet_encoder_init(LPCNetEncState *st);
// memset to zero, initializes PitchDNN sub-state. Returns 0.

int  lpcnet_encoder_load_model(LPCNetEncState *st, const void *data, int len);
// Loads PitchDNN model weights. Returns 0 on success.

LPCNetEncState *lpcnet_encoder_create(void);
void lpcnet_encoder_destroy(LPCNetEncState *st);

int  lpcnet_encode(LPCNetEncState *st, const opus_int16 *pcm,
                   unsigned char *buf);
// Encodes pcm[640] -> buf[8]. Returns 0.

int  lpcnet_compute_single_frame_features(
        LPCNetEncState *st, const opus_int16 *pcm,
        float features[NB_TOTAL_FEATURES], int arch);
// Computes features for one 160-sample frame from int16 PCM. Returns 0.

int  lpcnet_compute_single_frame_features_float(
        LPCNetEncState *st, const float *pcm,
        float features[NB_TOTAL_FEATURES], int arch);
// Same but from float PCM input. Returns 0.
```

### 2.4 PLC State (`LPCNetPLCState`)

Packet loss concealment. Maintains a buffer of recent audio and uses a prediction network + FARGAN to generate concealment frames.

```c
int  lpcnet_plc_init(LPCNetPLCState *st);
// Initializes FARGAN, encoder, loads PLC model weights if compiled in.
// Calls lpcnet_plc_reset(). Returns 0 on success.

void lpcnet_plc_reset(LPCNetPLCState *st);
// Clears state from LPCNET_PLC_RESET_START onward. Resets encoder.
// Sets analysis_gap=1, analysis_pos=PLC_BUF_SIZE, predict_pos=PLC_BUF_SIZE.

int  lpcnet_plc_update(LPCNetPLCState *st, opus_int16 *pcm);
// Called on each GOOD packet. Shifts PCM buffer, stores new audio (scaled to
// [-1,1] float). Resets loss_count and blend. Returns 0.

int  lpcnet_plc_conceal(LPCNetPLCState *st, opus_int16 *pcm);
// Called on each LOST packet. Generates one frame of concealment audio.
// On first loss (blend==0): runs analysis catchup, initializes FARGAN
// continuation from buffered audio. Uses get_fec_or_pred() for features.
// Applies energy attenuation based on loss_count. Returns 0.

void lpcnet_plc_fec_add(LPCNetPLCState *st, const float *features);
// Adds a forward error correction feature vector. NULL increments fec_skip.

void lpcnet_plc_fec_clear(LPCNetPLCState *st);
// Resets FEC read/fill positions and skip counter.

int  lpcnet_plc_load_model(LPCNetPLCState *st, const void *data, int len);
// Loads PLC model, encoder model (PitchDNN), and FARGAN model from blob.
```

---

## 3. Internal State

### 3.1 `LPCNetEncState` — Encoder/Feature Extractor

Defined in `lpcnet_private.h:25-42`:

```c
struct LPCNetEncState {
  PitchDNNState pitchdnn;          // Neural pitch estimator state
  float analysis_mem[OVERLAP_SIZE]; // 160 samples: overlap-add analysis window memory
  float mem_preemph;               // Single-sample preemphasis filter memory
  kiss_fft_cpx prev_if[PITCH_IF_MAX_FREQ]; // Previous FFT bins for instantaneous freq (30 bins)
  float if_features[PITCH_IF_FEATURES];    // Instantaneous frequency features (88 = 3*30-2)
  float xcorr_features[PITCH_MAX_PERIOD - PITCH_MIN_PERIOD]; // 224 cross-correlation values
  float dnn_pitch;                 // Neural pitch estimate (continuous, log-scale)
  float pitch_mem[LPC_ORDER];     // 16 samples: LPC analysis memory for pitch residual
  float pitch_filt;               // Single-sample pitch pre-filter memory
  float exc_buf[PITCH_BUF_SIZE];  // 576 (256+320): excitation ring buffer
  float lp_buf[PITCH_BUF_SIZE];   // 576: low-passed residual ring buffer
  float lp_mem[4];                // Biquad LP filter memory (2 biquad state + 2 unused)
  float lpc[LPC_ORDER];           // 16: current LPC coefficients
  float features[NB_TOTAL_FEATURES]; // 36: computed feature vector
  float sig_mem[LPC_ORDER];       // (unused in current code path)
  float burg_cepstrum[2*NB_BANDS]; // 36: Burg cepstral analysis output
};
```

**Lifecycle**: Created via `lpcnet_encoder_init()` (memset to zero). Accumulates state frame-by-frame via `compute_frame_features()`. No explicit teardown beyond `opus_free()`.

### 3.2 `LPCNetState` — Synthesis Core

The full struct definition is **generated** by `training_tf2/dump_lpcnet.py` into `nnet_data.h`. From usage in `lpcnet.c`, the key fields are:

```c
struct LPCNetState {
  LPCNetModel model;             // All NN layer weights (generated)
  NNetState nnet;                // Runtime state for conv/GRU layers (generated)

  // Fields before LPCNET_RESET_START are preserved across reset:
  float sampling_logit_table[256]; // Precomputed logit LUT for mu-law sampling

#define LPCNET_RESET_START ...     // Marker: everything from here is cleared on reset

  int frame_count;               // Frames processed (capped at 1000)
  int feature_buffer_fill;       // Deferred frame buffer fill level
  float feature_buffer[MAX_FEATURE_BUFFER_SIZE * NB_FEATURES]; // Deferred features

  float gru_a_condition[3*GRU_A_STATE_SIZE]; // Frame-level GRU-A conditioning
  float gru_b_condition[3*GRU_B_STATE_SIZE]; // Frame-level GRU-B conditioning
  float lpc[LPC_ORDER];         // Current LPC coefficients for synthesis
  float last_sig[LPC_ORDER];    // Last 16 output samples (pre-deemphasis)
  int   last_exc;               // Last excitation sample (mu-law index 0-255)
  float deemph_mem;             // De-emphasis filter memory
  kiss99_ctx rng;               // PRNG state for sampling

  float old_lpc[FEATURES_DELAY][LPC_ORDER]; // LPC delay line (when FEATURES_DELAY > 0)
};
```

**`NNetState`** (generated) holds per-layer runtime state:
```c
typedef struct {
  float feature_conv1_state[FEATURE_CONV1_STATE_SIZE];
  float feature_conv2_state[FEATURE_CONV2_STATE_SIZE];
  float gru_a_state[GRU_A_STATE_SIZE];
  float gru_b_state[GRU_B_STATE_SIZE];
} NNetState;
```

### 3.3 `LPCNetPLCState` — PLC Engine

Defined in `lpcnet_private.h:50-72`:

```c
struct LPCNetPLCState {
  PLCModel model;               // PLC-specific NN weights
  FARGANState fargan;           // FARGAN synthesis state
  LPCNetEncState enc;           // Embedded encoder for analysis
  int loaded;                   // Whether model weights are loaded
  int arch;                     // CPU architecture flags (SIMD selection)

  // --- LPCNET_PLC_RESET_START: everything below is cleared on reset ---
  float fec[PLC_MAX_FEC][NB_FEATURES]; // 104 × 20: FEC feature ring buffer
  int   analysis_gap;           // 1 if there's a gap in analysis history
  int   fec_read_pos;           // Read cursor into fec[]
  int   fec_fill_pos;           // Write cursor into fec[]
  int   fec_skip;               // FEC frames to skip (NULL features added)
  int   analysis_pos;           // Current analysis position in pcm buffer
  int   predict_pos;            // Position from which PLC predictions begin
  float pcm[PLC_BUF_SIZE];     // (CONT_VECTORS+10)*FRAME_SIZE = 2400 float samples
  int   blend;                  // 0 = first conceal call, 1 = continuation
  float features[NB_TOTAL_FEATURES]; // 36: working feature buffer
  float cont_features[CONT_VECTORS*NB_FEATURES]; // 5×20: continuation feature history
  int   loss_count;             // Consecutive lost frames
  PLCNetState plc_net;          // PLC prediction GRU state (current)
  PLCNetState plc_bak[2];       // PLC prediction GRU state (backup, 2 deep)
};
```

**`PLCNetState`** holds the PLC prediction network's recurrent state:
```c
typedef struct {
  float gru1_state[PLC_GRU1_STATE_SIZE];
  float gru2_state[PLC_GRU2_STATE_SIZE];
} PLCNetState;
```

### 3.4 KISS99 PRNG State

```c
struct kiss99_ctx {
  uint32_t z;       // MWC generator state (high half)
  uint32_t w;       // MWC generator state (low half)
  uint32_t jsr;     // XorShift generator state
  uint32_t jcong;   // Linear congruential generator state
};
```

---

## 4. Algorithm

### 4.1 Feature Extraction (`compute_frame_features`)

This is the encoder's core analysis function, called once per 10 ms frame (160 samples). The pipeline:

**Step 1 — Preemphasis**
```c
void preemphasis(float *y, float *mem, const float *x, float coef, int N) {
  for (i=0;i<N;i++) {
    yi = x[i] + *mem;      // Note: this is DE-emphasis style (additive)
    *mem = -coef*x[i];     // coef = PREEMPHASIS = 0.85
    y[i] = yi;
  }
}
```
This is a first-order high-pass filter: `y[n] = x[n] - 0.85 * x[n-1]`, implemented as `y[n] = x[n] + mem`, `mem = -0.85 * x[n]`.

**Step 2 — Windowed FFT Analysis (`frame_analysis`)**
- Copy `OVERLAP_SIZE` (160) samples from analysis_mem, append `FRAME_SIZE` (160) new samples → 320-sample window
- Save last `OVERLAP_SIZE` samples of input to `analysis_mem` for next frame overlap
- Apply `half_window[]` (raised-cosine half-window applied symmetrically to both ends)
- 320-point real FFT via `forward_transform()` → 161 complex bins (`FREQ_SIZE = WINDOW_SIZE/2 + 1`)
- Compute band energies across 18 Bark-scale bands using triangular overlap

**Step 3 — Instantaneous Frequency Features**
For bins 1..29 (`PITCH_IF_MAX_FREQ`):
- Compute cross-frame phase difference: `prod = X[i] * conj(prev_if[i])`
- Normalize: `prod /= |prod|`
- Store `{prod.r, prod.i, log_energy}` → 88 features (`3*30-2`)
- Bin 0 stores only log energy → `if_features[0]`

**Step 4 — Cepstral Coefficients**
- Compute log band energies: `Ly[i] = log10(1e-2 + Ex[i])`
- Apply spectral floor: each band ≥ max-8 dB and ≥ running follower - 2.5 dB
- DCT of `Ly[]` → 18 cepstral coefficients stored in `features[0..17]`
- Subtract 4 from DC term: `features[0] -= 4`

**Step 5 — LPC from Cepstrum**
- Inverse DCT of cepstrum (with DC+4 restored) → log band energies
- Convert to linear: `Ex[i] = 10^(Ex[i]) * compensation[i]`
- Interpolate band energies to per-bin gains
- IFFT to get autocorrelation
- Levinson-Durbin recursion → 16 LPC coefficients
- Store LPC in `features[NB_BANDS+2 .. NB_BANDS+2+LPC_ORDER-1]` (indices 20..35)

**Step 6 — Pitch Analysis**
- Compute LPC residual via `celt_fir()` → `lp_buf[]`
- Apply pitch pre-filter: `exc_buf[i] = lp_buf[i] + 0.7 * lp_buf[i-1]`
- Apply elliptic low-pass biquad (cutoff ~1200 Hz) to `lp_buf[]`
- Cross-correlate excitation with past excitation → 224 values
- Normalize cross-correlations by energy: `xcorr[i] /= (1 + ener0 + ener1)`
- Run PitchDNN on `{if_features, xcorr_features}` → continuous pitch estimate `dnn_pitch`
- Convert to integer period: `pitch = floor(0.5 + 256 / 2^((dnn_pitch+1.5)))`

**Step 7 — Pitch Correlation**
- Compute normalized correlation at detected pitch on low-passed signal
- `frame_corr = xy / sqrt(1 + xx*yy)`
- Apply soft-plus scaling: `frame_corr = log(1 + exp(5*frame_corr)) / log(1 + exp(5))`
- Store: `features[NB_BANDS] = dnn_pitch`, `features[NB_BANDS+1] = frame_corr - 0.5`

### 4.2 Frame-Level Conditioning (`run_frame_network`)

Converts a 20-element feature vector into conditioning signals for the sample-level network. Called once per frame.

```
Input: features[NB_FEATURES=20]

1. Extract pitch period:
   pitch = floor(0.1 + 50*features[NB_BANDS] + 100)
   pitch = clamp(pitch, 33, 255)

2. Build frame input:
   in[0..19]  = features[0..NB_FEATURES-1]
   in[20..xx] = embed_pitch(pitch)    // pitch embedding lookup

3. Feature conditioning network (all causal conv1d + dense):
   conv1_out = conv1d(feature_conv1, in)     // with state persistence
   if frame_count < FEATURE_CONV1_DELAY: zero conv1_out
   conv2_out = conv1d(feature_conv2, conv1_out)
   if frame_count < FEATURES_DELAY: zero conv2_out
   dense1_out = dense(feature_dense1, conv2_out)   // tanh activation
   condition  = dense(feature_dense2, dense1_out)   // tanh activation

4. Extract LPC from condition:
   rc[0..15] = condition[0..15]    // (END2END mode: reflection coeffs → LPC)
   // Normal mode (FEATURES_DELAY>0): use delayed lpc_from_cepstrum

5. Compute GRU conditioning:
   gru_a_condition = dense(gru_a_dense_feature, condition)
   gru_b_condition = dense(gru_b_dense_feature, condition)
```

The convolutional layers introduce a startup delay (`FEATURE_CONV1_DELAY`, `FEATURES_DELAY`), during which outputs are zeroed.

### 4.3 Sample-Level Synthesis (`run_sample_network`)

Generates one sample at a time, autoregessively. Called 160 times per frame.

```
Inputs:
  gru_a_condition[3*GRU_A_STATE_SIZE]  — from frame network
  gru_b_condition[3*GRU_B_STATE_SIZE]  — from frame network
  last_exc  — previous excitation (mu-law index, 0-255)
  last_sig  — previous output signal (mu-law index)
  pred      — LPC prediction (mu-law index)

1. GRU-A input assembly:
   gru_a_input = gru_a_condition
              + embed_sig(last_sig)
              + embed_pred(pred)
              + embed_exc(last_exc)
   // Done via compute_gru_a_input() — optimized fused function

2. Sparse GRU-A:
   gru_a_state = sparse_gru(sparse_gru_a, gru_a_state, gru_a_input)
   // Block-sparse weight matrix for efficiency

3. GRU-B:
   in_b = gru_a_state
   gru_b_input = gru_b_condition
   gru_b_state = gruB(gru_b, gru_b_input, gru_b_state, in_b)

4. Sampling:
   return sample_mdense(dual_fc, gru_b_state, sampling_logit_table, rng)
   // Dual fully-connected layer → probability over 256 mu-law levels
   // Stochastic sampling using KISS99 PRNG
```

### 4.4 Full Synthesis Loop (`lpcnet_synthesize_tail_impl`)

Per-sample loop integrating LPC filtering with neural excitation:

```c
for (i = 0; i < N; i++) {
    // 1. LPC prediction (FIR filter on last_sig)
    pred = 0;
    for (j = 0; j < LPC_ORDER; j++)
        pred -= last_sig[j] * lpc[j];

    // 2. Convert previous signal and prediction to mu-law
    last_sig_ulaw = lin2ulaw(last_sig[0]);
    pred_ulaw = lin2ulaw(pred);

    // 3. Neural network generates excitation sample (mu-law index)
    exc = run_sample_network(..., last_exc, last_sig_ulaw, pred_ulaw, ...);

    // 4. Reconstruct PCM: prediction + excitation
    pcm = pred + ulaw2lin(exc);

    // 5. Update signal history (shift register)
    memmove(last_sig+1, last_sig, (LPC_ORDER-1)*sizeof(float));
    last_sig[0] = pcm;
    last_exc = exc;

    // 6. De-emphasis: pcm += 0.85 * deemph_mem
    pcm += PREEMPH * deemph_mem;
    deemph_mem = pcm;

    // 7. Clip and output
    output[i] = clamp(floor(0.5 + pcm), -32767, 32767);
}
```

### 4.5 Deferred Frame Processing

For latency optimization, frames can be buffered and processed in batch:

- **`run_frame_network_deferred()`**: Stores features in `feature_buffer[]` (FIFO, up to `max_buffer_size = conv1.kernel_size + conv2.kernel_size - 2`). Does not run the network.
- **`run_frame_network_flush()`**: Processes all buffered frames through `run_frame_network()` in order, then clears the buffer.

### 4.6 PLC Concealment Algorithm (`lpcnet_plc_conceal`)

**First lost frame (`blend == 0`)**:
1. Restore PLC network state from backup: `plc_net = plc_bak[0]`
2. **Analysis catchup**: Process all unanalyzed audio in the PCM buffer:
   - Scale float PCM back to int16 range
   - Compute Burg cepstral features (split-frame: two half-frames → mean + difference)
   - Run encoder feature extraction
   - Feed `{burg_cepstrum, features, flag=1}` into PLC prediction network
   - Queue features into `cont_features[]` ring buffer
3. Run prediction network twice to generate 2 lookahead frames
4. Initialize FARGAN continuation from the last `FARGAN_CONT_SAMPLES` of PCM buffer

**Each lost frame (including first)**:
1. Back up PLC network state: `plc_bak[0] = plc_bak[1]; plc_bak[1] = plc_net`
2. Get features from FEC buffer (if available) or from prediction network
3. Apply energy attenuation based on `loss_count` (see `att_table[]`)
4. Synthesize one frame via `fargan_synthesize_int()`
5. Queue features, shift PCM buffer, store synthesized audio

**PLC Prediction Network (`compute_plc_pred`)**:
```
input:  [2*NB_BANDS burg cepstrum | NB_FEATURES features | 1 flag]
        flag = 1 for real features, -1 for FEC, 0 for prediction
  → dense_in (tanh)
  → GRU1 (recurrent)
  → GRU2 (recurrent)
  → dense_out (linear)
output: NB_FEATURES predicted features
```

### 4.7 Burg Cepstral Analysis

Used exclusively by PLC for short-segment spectral analysis:

1. Apply preemphasis to input PCM
2. Run Silk Burg analysis (Burg's method for AR coefficients) with order 16
3. Compute frequency response of AR model via FFT of LPC polynomial
4. Compute **inverse** band energy (1/(|H(f)|² + ε))
5. Scale by residual energy and normalization factors
6. Apply spectral floor (same as encoder)
7. DCT → cepstrum, subtract 4 from DC

Split analysis: first half and second half of frame analyzed separately, then:
- Mean: `ceps[i] = 0.5*(c0+c1)`
- Difference: `ceps[NB_BANDS+i] = c0 - c1`

---

## 5. Data Flow

### 5.1 Feature Vector Layout

**`features[NB_TOTAL_FEATURES=36]`**:

| Index | Count | Content |
|-------|-------|---------|
| 0..17 | 18 (`NB_BANDS`) | Cepstral coefficients (DCT of log band energy). `[0]` has DC-4 offset. |
| 18 | 1 | Pitch period (continuous, log-scale: `dnn_pitch`) |
| 19 | 1 | Pitch correlation - 0.5 |
| 20..35 | 16 (`LPC_ORDER`) | LPC coefficients |

**`NB_FEATURES=20`**: Only indices 0..19 (cepstrum + pitch + correlation). This is what gets transmitted/predicted.

**`NB_TOTAL_FEATURES=36`**: Includes LPC coefficients appended by the encoder. The synthesizer derives its own LPC from the cepstrum, so indices 20..35 are informational.

### 5.2 Buffer Layouts

**Excitation / LP buffers** (`exc_buf[]`, `lp_buf[]`, size `PITCH_BUF_SIZE=576`):
```
[--- PITCH_MAX_PERIOD=256 history ---|--- FRAME_SIZE=160 current ---]
```
The history portion shifts left by `FRAME_SIZE` each frame via `OPUS_MOVE`.

**PLC PCM buffer** (`pcm[]`, size `PLC_BUF_SIZE = (5+10)*160 = 2400`):
```
[--- old audio (scrolls left by FRAME_SIZE each update) ---|--- newest FRAME_SIZE ---]
```
Stored as float in range [-1, 1] (divided by 32768).

**FEC buffer** (`fec[PLC_MAX_FEC=104][NB_FEATURES=20]`):
Simple FIFO with `fec_read_pos` and `fec_fill_pos` cursors. `fec_skip` counts NULL entries to skip.

**Feature continuation buffer** (`cont_features[CONT_VECTORS*NB_FEATURES = 100]`):
Rolling window of last 5 feature vectors, shifted left by `NB_FEATURES` on each `queue_features()` call. Used by FARGAN for waveform continuation.

### 5.3 Encoder Data Flow

```
PCM (int16, 160 samples)
  │
  ├─→ preemphasis (0.85) ─→ float[160]
  │
  ├─→ frame_analysis:
  │     overlap-add window ─→ 320-pt FFT ─→ band energy (18 bands)
  │                                     └─→ instantaneous freq features (88)
  │
  ├─→ log band energy ─→ spectral floor ─→ DCT ─→ cepstrum[0..17]
  │
  ├─→ lpc_from_cepstrum ─→ LPC[16] ─→ features[20..35]
  │
  ├─→ LPC residual ─→ pitch pre-filter ─→ xcorrelation (224 values)
  │                                   └─→ biquad LP filter
  │
  ├─→ PitchDNN(if_features, xcorr) ─→ dnn_pitch ─→ features[18]
  │
  └─→ normalized correlation at pitch ─→ soft-plus ─→ features[19]
```

### 5.4 Synthesis Data Flow

```
features[20]
  │
  ├─→ run_frame_network:
  │     pitch embedding ─→ conv1d ─→ conv1d ─→ dense1 ─→ dense2 ─→ condition
  │     condition ─→ gru_a_condition, gru_b_condition, lpc[16]
  │
  └─→ for each sample (160×):
        │
        ├─→ LPC prediction: pred = -Σ(last_sig[j] * lpc[j])
        │
        ├─→ run_sample_network:
        │     embeddings(last_exc, last_sig_ulaw, pred_ulaw)
        │       ─→ sparse GRU-A ─→ GRU-B ─→ dual_fc ─→ sample_mdense
        │       ─→ excitation index (0-255, mu-law)
        │
        ├─→ pcm = pred + ulaw2lin(exc)
        │
        └─→ de-emphasis ─→ clip ─→ output[i] (int16)
```

---

## 6. Numerical Details

### 6.1 Mu-Law Companding

The codec operates in an 8-bit mu-law domain (256 levels, indices 0-255). This compresses the dynamic range so the neural network only needs to model 256 discrete output levels.

**Linear to mu-law** (`lin2ulaw`):
```c
// Input: float x (linear PCM, range roughly ±32768)
// Output: int (0-255)
scale = 255.0 / 32768.0;
s = sign(x);
u = s * 128 * log_approx(1 + scale*|x|) / LOG256;
return floor(0.5 + clamp(128 + u, 0, 255));
```
Where `LOG256 = 5.5451774445 = ln(256)` and `log_approx` is a fast `ln()` approximation.

**Mu-law to linear** (`ulaw2lin`):
```c
// Input: float u (mu-law index, will be converted from 0-255 to ±128)
// Output: float (linear PCM)
scale_1 = 32768.0 / 255.0;
u = u - 128;
s = sign(u);
return s * scale_1 * (exp(|u|/128 * LOG256) - 1);
```

**Precision note**: `log_approx` uses a polynomial approximation of `log2`:
```c
float log2_approx(float x) {
    // Extract exponent and mantissa via IEEE-754 bit manipulation
    integer = (float_bits >> 23) - 127;
    frac = reconstruct_mantissa - 1.5;
    frac = -0.41445418 + frac*(0.95909232 + frac*(-0.33951290 + frac*0.16541097));
    return 1 + integer + frac;
}
```
This is accurate to ~20 bits. The `ulaw2lin` direction uses standard `exp()`, so the mapping is not perfectly invertible. **For bit-exact porting, this polynomial and the IEEE-754 bit manipulation must be reproduced exactly.**

### 6.2 Sampling Logit Table

Precomputed at init time for the sample-level network's probabilistic output:

```c
for (i = 0; i < 256; i++) {
    float prob = 0.025 + 0.95 * i / 255.0;
    sampling_logit_table[i] = -log((1-prob) / prob);
}
```

This maps uniform indices to logit space, used by `sample_mdense()` to convert network logits to sampling probabilities. The floor of 0.025 and ceiling of 0.975 prevent infinite logits.

### 6.3 Preemphasis / De-emphasis

- **Coefficient**: `PREEMPH = PREEMPHASIS = 0.85f`
- **Preemphasis** (encoder): `y[n] = x[n] - 0.85*x[n-1]` (high-pass, boosts high frequencies)
- **De-emphasis** (synthesizer): `out[n] = pcm[n] + 0.85*deemph_mem` (low-pass, restores spectrum)

The preemphasis in `lpcnet_enc.c:preemphasis()` is implemented with inverted sign convention:
```c
yi = x[i] + *mem;        // mem starts at 0
*mem = -coef * x[i];     // mem = -0.85 * x[i]
```
Expanding: `y[i] = x[i] + (-0.85 * x[i-1]) = x[i] - 0.85*x[i-1]`. Correct.

### 6.4 Band Energy Computation

Band boundaries follow the Bark scale via `eband5ms[]`:
```c
{0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40}
```
These are indices in 5ms units; multiplied by `WINDOW_SIZE_5MS=4` (= 80 samples / 20 bins per 5ms at 320-point FFT) to get bin indices.

Energy is computed with **triangular overlap** between adjacent bands:
```c
for (j = 0; j < band_size; j++) {
    frac = j / band_size;
    tmp = |X[bin]|²;
    sum[i]   += (1-frac) * tmp;
    sum[i+1] += frac * tmp;
}
```
First and last bands are doubled (`sum[0] *= 2; sum[NB_BANDS-1] *= 2`) to compensate for missing overlap partner.

A **compensation** table adjusts for varying band widths:
```c
{0.8, 1, 1, 1, 1, 1, 1, 1, 0.667, 0.5, 0.5, 0.5, 0.333, 0.25, 0.25, 0.2, 0.167, 0.174}
```

### 6.5 DCT

Type-II DCT with orthogonal normalization:
```c
out[i] = sqrt(2/NB_BANDS) * Σ_j in[j] * cos(π*(2j+1)*i / (2*NB_BANDS))
```
Implemented via precomputed `dct_table[NB_BANDS × NB_BANDS]`. The IDCT uses the same table transposed (exploiting DCT-II / DCT-III duality with the same normalization factor).

### 6.6 LPC Computation

Two paths:

1. **From cepstrum** (`lpc_from_cepstrum`): IDCT → exponentiate → interpolate to frequency bins → IFFT → autocorrelation → Levinson-Durbin. Includes:
   - Noise floor: `ac[0] += ac[0]*1e-4 + 320/12/38` (~-40 dB)
   - Lag windowing: `ac[i] *= (1 - 6e-5 * i²)`
   - Bail out at 30 dB gain: `if (error < 0.001 * ac[0]) break`

2. **Burg analysis** (`silk_burg_analysis`): Direct AR parameter estimation using Burg's method. All internal computation in **double precision** for stability. Key features:
   - Minimum inverse gain parameter (`minInvGain = 1e-3` in PLC usage)
   - Reflection coefficient clamping when max prediction gain is exceeded
   - Output coefficients are negated: `A[k] = -Af[k]`

### 6.7 Pitch Period Encoding/Decoding

**Encoder** (continuous → integer):
```c
pitch = floor(0.5 + 256 / pow(2, (dnn_pitch + 1.5)));
```

**Frame network** (features → integer for embedding):
```c
pitch = floor(0.1 + 50*features[NB_BANDS] + 100);
pitch = clamp(pitch, 33, 255);
```

These are two different conversions. The encoder stores the continuous `dnn_pitch` value directly in the feature vector. The frame network re-derives an integer pitch for the embedding lookup.

### 6.8 Pitch Correlation Normalization

```c
frame_corr = xy / sqrt(1 + xx*yy);         // Normalized, +1 prevents /0
frame_corr = log(1+exp(5*corr)) / log(1+exp(5));  // Soft-plus scaling to ~[0,1]
features[NB_BANDS+1] = frame_corr - 0.5;   // Center around 0
```

---

## 7. Dependencies

### 7.1 Modules Called by LPCNet

| Module | Functions Used | Purpose |
|--------|---------------|---------|
| **nnet** (`nnet.h`, `nnet.c`) | `compute_conv1d`, `compute_embedding`, `accum_embedding`, `compute_sparse_gru`, `compute_gruB`, `compute_gru_a_input`, `_lpcnet_compute_dense`, `sample_mdense`, `compute_generic_dense`, `compute_generic_gru`, `init_lpcnet_model`, `parse_weights` | Neural network layer implementations |
| **nnet_data** (`nnet_data.h`) | `LPCNetModel`, `NNetState`, `lpcnet_arrays`, all `*_SIZE` macros | Generated model definitions and weights |
| **plc_data** (`plc_data.h`) | `PLCModel`, `PLC_*_STATE_SIZE`, `plcmodel_arrays`, `init_plcmodel` | PLC model definitions and weights |
| **pitchdnn** (`pitchdnn.h`) | `PitchDNNState`, `pitchdnn_init`, `compute_pitchdnn`, `pitchdnn_load_model` | Neural pitch estimator |
| **fargan** (`fargan.h`) | `FARGANState`, `fargan_init`, `fargan_cont`, `fargan_synthesize_int`, `fargan_load_model` | Neural waveform generator (PLC) |
| **kiss_fft** (`kiss_fft.h`) | `opus_fft`, `kiss_fft_cpx`, `kiss_fft_state` | FFT implementation |
| **celt_lpc** (`celt_lpc.h`) | `celt_fir` | FIR filtering for LPC residual |
| **pitch** (`pitch.h`) | `celt_pitch_xcorr`, `celt_inner_prod`, `PITCH_MIN_PERIOD`, `PITCH_MAX_PERIOD` | Pitch cross-correlation primitives |
| **mathops** (`mathops.h`) | `celt_log2`, `celt_log10` | Fast log approximations |
| **common** (`common.h`) | `lin2ulaw`, `ulaw2lin`, `log2_approx`, `log_approx` | Mu-law conversion, fast math |
| **kiss99** (`kiss99.h`) | `kiss99_srand`, `kiss99_rand` | PRNG for stochastic sampling |
| **freq** (`freq.h`, `freq.c`) | `lpcn_compute_band_energy`, `dct`, `forward_transform`, `lpc_from_cepstrum`, `apply_window`, `lpc_weighting`, `burg_cepstral_analysis` | Frequency-domain analysis utilities |
| **burg** (`burg.h`, `burg.c`) | `silk_burg_analysis` | Burg AR coefficient estimation |

### 7.2 What Calls LPCNet

- **Opus decoder** (`opus_decoder.c`): Uses `LPCNetDecState` for low-bitrate speech decoding
- **Opus encoder** (`opus_encoder.c`): Uses `LPCNetEncState` for feature extraction
- **OSCE** (`osce.c`): Uses `LPCNetPLCState` for speech coding enhancement
- **DRED** (`dred_*.c`): Uses `LPCNetEncState` for deep redundancy features

---

## 8. Constants and Tables

### 8.1 Compile-Time Constants

```c
// Frame geometry (freq.h)
#define FRAME_SIZE_5MS     2
#define OVERLAP_SIZE_5MS   2
#define TRAINING_OFFSET_5MS 1
#define WINDOW_SIZE_5MS    (FRAME_SIZE_5MS + OVERLAP_SIZE_5MS)  // = 4
#define FRAME_SIZE         (80 * FRAME_SIZE_5MS)     // = 160 samples (10 ms)
#define OVERLAP_SIZE       (80 * OVERLAP_SIZE_5MS)   // = 160 samples
#define TRAINING_OFFSET    (80 * TRAINING_OFFSET_5MS) // = 80 samples
#define WINDOW_SIZE        (FRAME_SIZE + OVERLAP_SIZE) // = 320 samples
#define FREQ_SIZE          (WINDOW_SIZE/2 + 1)        // = 161 bins

// Feature dimensions (lpcnet.h)
#define NB_FEATURES         20
#define NB_TOTAL_FEATURES   36
#define LPCNET_FRAME_SIZE   160

// Spectral (freq.h)
#define NB_BANDS            18
#define LPC_ORDER           16
#define PREEMPHASIS         0.85f

// Pitch (lpcnet_private.h, pitchdnn.h)
#define PITCH_MIN_PERIOD    32     // ~500 Hz at 16 kHz
#define PITCH_MAX_PERIOD    256    // ~62.5 Hz at 16 kHz
#define PITCH_FRAME_SIZE    320
#define PITCH_BUF_SIZE      (PITCH_MAX_PERIOD + PITCH_FRAME_SIZE) // = 576
#define PITCH_IF_MAX_FREQ   30
#define PITCH_IF_FEATURES   (3*PITCH_IF_MAX_FREQ - 2)  // = 88

// Synthesis (lpcnet.c)
#define PREEMPH             0.85f
#define PDF_FLOOR           0.002
#define FRAME_INPUT_SIZE    (NB_FEATURES + EMBED_PITCH_OUT_SIZE)

// PLC (lpcnet_private.h)
#define PLC_MAX_FEC         104
#define CONT_VECTORS        5
#define PLC_BUF_SIZE        ((CONT_VECTORS + 10) * FRAME_SIZE)  // = 2400
#define FEATURES_DELAY      1
#define MAX_FEATURE_BUFFER_SIZE 4

// Burg analysis (burg.c)
#define MAX_FRAME_SIZE      384
#define SILK_MAX_ORDER_LPC  16
#define FIND_LPC_COND_FAC   1e-5f

// common.h
#define LOG256              5.5451774445f
```

### 8.2 Static Tables in `lpcnet_tables.c`

| Table | Size | Purpose |
|-------|------|---------|
| `fft_bitrev[320]` | 320 × int16 | Bit-reversal permutation for 320-point FFT |
| `fft_twiddles[320]` | 320 × complex | FFT twiddle factors (unit roots) |
| `kfft` | struct | Complete FFT state: `{nfft=320, scale=1/320, factors=[5,64,4,16,4,4,4,1,...]}` |
| `half_window[160]` | 160 × float | Analysis window (raised cosine half-window, applied to both ends) |
| `dct_table[324]` | 18×18 float | Precomputed `cos(π(2j+1)i/(2×18))` values for DCT/IDCT |

### 8.3 Static Tables in `freq.c`

| Table | Values | Purpose |
|-------|--------|---------|
| `eband5ms[18]` | `{0,1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,34,40}` | Bark-scale band boundaries in 5ms units |
| `compensation[18]` | `{0.8,1,1,1,1,1,1,1,0.667,...,0.174}` | Band width compensation factors |

### 8.4 Static Tables in `lpcnet_plc.c`

| Table | Values | Purpose |
|-------|--------|---------|
| `att_table[10]` | `{0, 0, -0.2, -0.2, -0.4, -0.4, -0.8, -0.8, -1.6, -1.6}` | Energy attenuation (dB) per consecutive lost frame |

### 8.5 Biquad Filter Coefficients

Elliptic low-pass filter at 1200 Hz (for 16 kHz sample rate), used in pitch analysis:
```c
// [b,a] = ellip(2, 2, 20, 1200/8000)
static const float lp_b[2] = {-0.84946, 1.0};
static const float lp_a[2] = {-1.54220, 0.70781};
```

### 8.6 Generated Tables and Model Weights

The `nnet_data.h` / `nnet_data.c` files (generated by `dump_lpcnet.py`) contain:
- All neural network weight matrices and bias vectors
- `lpcnet_arrays[]`: weight array for `init_lpcnet_model()`
- Size macros: `GRU_A_STATE_SIZE`, `GRU_B_STATE_SIZE`, `EMBED_PITCH_OUT_SIZE`, `FEATURE_CONV1_OUT_SIZE`, `FEATURE_CONV2_OUT_SIZE`, `FEATURE_DENSE1_OUT_SIZE`, `FEATURE_DENSE2_OUT_SIZE`, `FEATURE_CONV1_DELAY`, etc.

These depend on the trained model and cannot be hardcoded until the specific model architecture is known.

---

## 9. Edge Cases

### 9.1 Startup / Warmup

- **`frame_count`**: Incremented each frame, capped at 1000. Used to zero out conv outputs during warmup:
  - If `frame_count < FEATURE_CONV1_DELAY`: zero `conv1_out`
  - If `frame_count < FEATURES_DELAY`: zero `conv2_out`
  - If `frame_count <= FEATURES_DELAY`: `lpcnet_synthesize_tail_impl` outputs silence
- **LPC delay line**: When `FEATURES_DELAY > 0`, LPC coefficients are delayed by `FEATURES_DELAY` frames via `old_lpc[][]` shift register. The first frame(s) use whatever is in the zero-initialized delay line.

### 9.2 PLC State Management

- **`analysis_gap`**: Set to 1 on reset and when `analysis_pos` underflows. Causes the first analysis frame to be skipped in the PLC catchup loop (prevents using stale state).
- **`blend`**: 0 on first conceal call (full catchup + FARGAN init), 1 on subsequent calls (just predict + synthesize). Reset to 0 on every good packet.
- **`plc_bak[2]`**: Two-deep backup of PLC GRU state. Allows rewinding state when transitioning from good to lost. State is shifted: `bak[0] = bak[1]; bak[1] = current` before each prediction.
- **FEC skip**: When `lpcnet_plc_fec_add(NULL)` is called, `fec_skip` increments. On `get_fec_or_pred`, if `fec_skip > 0`, the FEC buffer is bypassed and the prediction network runs instead, decrementing `fec_skip`.

### 9.3 Signal Clipping

Output is hard-clipped to `[-32767, 32767]` with rounding:
```c
if (pcm < -32767) pcm = -32767;
if (pcm > 32767) pcm = 32767;
output[i] = (int)floor(0.5 + pcm);
```
Note: asymmetric clip (not -32768). This is intentional for consistency with the training pipeline.

### 9.4 Pitch Clamping

Pitch period is clamped to `[33, 255]` in `run_frame_network` (the embedding table has 256 entries indexed 0-255, but only values ≥ 33 are valid pitch periods). The `floor(.1 + ...)` adds 0.1 to avoid rounding issues at integer boundaries.

### 9.5 Energy Floor in Band Computation

All energy computations use additive floors:
- Band energy: `log10(1e-2 + Ex[i])` — prevents log(0)
- Spectral floor: `max(logMax-8, max(follow-2.5, Ly[i]))` — prevents extreme valleys
- Burg inverse energy: `1/(|X|² + 1e-9)` — prevents division by zero
- IF normalization: `1/sqrt(1e-15 + ...)` — prevents division by zero
- Pitch correlation: `xy / sqrt(1 + xx*yy)` — the +1 prevents division by zero

### 9.6 Preload Mode

`lpcnet_synthesize_tail_impl` supports a `preload` parameter: for the first `preload` samples, instead of using the neural network's excitation, it derives excitation from provided output PCM:
```c
if (i < preload) {
    exc = lin2ulaw(output[i] - PREEMPH*deemph_mem - pred);
    pcm = output[i] - PREEMPH*deemph_mem;
}
```
This allows "priming" the synthesis state with known audio before switching to neural generation. Used internally by `lpcnet_synthesize_blend_impl`.

### 9.7 PLC Attenuation Ramp

After 10 consecutive lost frames, attenuation increases aggressively:
```c
if (loss_count >= 10)
    features[0] = MAX(-15, features[0] + att_table[9] - 2*(loss_count-9));
```
The DC cepstral coefficient (overall energy) is driven toward -15, which corresponds to near-silence after de-normalization.

---

## 10. Porting Notes

### 10.1 Generated Code Dependency

**Critical**: The `LPCNetState`, `NNetState`, `LPCNetModel`, and all `*_SIZE` macros are **generated** by `training_tf2/dump_lpcnet.py` from the trained Keras model. The Rust port must either:

1. Port the code generator to produce Rust structs and const arrays, or
2. Define the structs manually based on a specific model snapshot, or
3. Use a runtime model loading path (`parse_weights` + `init_lpcnet_model`) and define structs with dynamic sizing.

Option 2 is likely simplest for initial porting. The size macros can be extracted from a generated `nnet_data.h` for a specific model.

### 10.2 `OPUS_COPY` / `OPUS_MOVE` / `OPUS_CLEAR` Macros

These map to:
- `OPUS_COPY(dst, src, n)` → `memcpy` (non-overlapping) → `dst[..n].copy_from_slice(&src[..n])`
- `OPUS_MOVE(dst, src, n)` → `memmove` (overlapping allowed) → `slice.copy_within(..)`
- `OPUS_CLEAR(dst, n)` → `memset(0)` → `dst[..n].fill(0)` or `[0.0; N]`

**Porting hazard**: `OPUS_CLEAR(lpcnet, 1)` zeros the entire struct (1 element of `LPCNetState` size). The reset functions use pointer arithmetic to clear only from a marker field onward:
```c
OPUS_CLEAR((char*)&lpcnet->LPCNET_RESET_START,
    sizeof(LPCNetState) - ((char*)&lpcnet->LPCNET_RESET_START - (char*)lpcnet));
```
In Rust, implement this with a dedicated `reset()` method that explicitly zeros the relevant fields, or use `unsafe` pointer arithmetic with `std::ptr::write_bytes` if matching exact layout.

### 10.3 In-Place Buffer Mutation

Several patterns mutate buffers in place:

1. **Shift register** (`OPUS_MOVE` with overlapping src/dst):
   ```c
   OPUS_MOVE(st->exc_buf, &st->exc_buf[FRAME_SIZE], PITCH_MAX_PERIOD);
   ```
   In Rust: `buf.copy_within(FRAME_SIZE.., 0)` or split into temp + copy.

2. **Signal history shift**:
   ```c
   OPUS_MOVE(&lpcnet->last_sig[1], &lpcnet->last_sig[0], LPC_ORDER-1);
   lpcnet->last_sig[0] = pcm;
   ```
   This is a right-shift by 1. In Rust: `last_sig.copy_within(..LPC_ORDER-1, 1); last_sig[0] = pcm;`

3. **Preemphasis operates in-place** when src == dst (as called from `lpcnet_compute_single_frame_features_impl`).

### 10.4 IEEE-754 Bit Manipulation

`log2_approx` in `common.h` uses a union to reinterpret `float` as `int`:
```c
union { float f; int i; } in;
in.f = x;
integer = (in.i >> 23) - 127;
in.i -= integer << 23;
frac = in.f - 1.5f;
```
In Rust: use `f32::to_bits()` and `f32::from_bits()`:
```rust
let bits = x.to_bits() as i32;
let integer = (bits >> 23) - 127;
let mantissa = f32::from_bits((bits - (integer << 23)) as u32);
let frac = mantissa - 1.5;
```

### 10.5 Conditional Compilation

Several `#ifdef` gates affect behavior:

| Define | Effect | Port Strategy |
|--------|--------|---------------|
| `END2END` | Use reflection coefficients from network directly instead of cepstrum-derived LPC | Likely skip (not standard path) |
| `FEATURES_DELAY` | Controls LPC delay line depth; set to 1 in header, overridable | Hardcode to 1 (default) |
| `LPC_GAMMA` | Apply LPC bandwidth expansion (`lpc_weighting`) | Likely not defined (skip) |
| `USE_WEIGHTS_FILE` | Skip compiled-in weights, require runtime loading | Support both paths via feature flag |
| `PLC_SKIP_UPDATES` | Skip running LPCNet state update on good packets (PLC optimization) | Include (it's the default) |

### 10.6 Biquad Filter Implementation

The biquad in `lpcnet_enc.c:84-105` uses a non-standard form optimized to reduce dependency chains:
```c
mem0 = (b[0]-a[0])*xi + mem1 - a[0]*mem0;
mem1 = (b[1]-a[1])*xi + 1e-30f - a[1]*mem00;
```
The `1e-30f` forces evaluation ordering without affecting the result. In Rust, this might not be necessary (floating-point evaluation order is well-defined), but include it for bit-exactness.

### 10.7 Double Precision in Burg Analysis

`silk_burg_analysis` performs all internal computation in `double` for numerical stability, only converting to `float` at the end. The helper functions `silk_energy_FLP` and `silk_inner_product_FLP` use `double` accumulators with 4x loop unrolling. The Rust port must use `f64` internally for these functions.

### 10.8 KISS99 PRNG

The PRNG must produce **identical sequences** for bit-exact output. The implementation combines three generators:
1. **MWC** (Multiply-With-Carry): two 16-bit half-generators with constants 36969 and 18000
2. **XorShift**: with shifts 13, 17, 5 (note: swapped 13/17 from original 1999 KISS to produce maximal cycle)
3. **Linear congruential**: `69069 * jcong + 1234567`

Final output: `(mwc ^ cong) + shr3`

Seed function processes input bytes in groups of 4, XORing into state fields, with short-cycle fixes applied after seeding.

**Porting hazard**: All arithmetic is `u32` wrapping. Rust's default arithmetic panics on overflow in debug mode; use `Wrapping<u32>` or `.wrapping_mul()` / `.wrapping_add()`.

### 10.9 FFT State as Global Constant

The `kfft` FFT state is a file-scope `const` in `lpcnet_tables.c`, initialized with the full twiddle and bitrev tables. In the C code it's used as a global:
```c
opus_fft(&kfft, x, y, 0);
```
In Rust, this should be a `static` or `const` (or `lazy_static`). The struct contains a pointer (`arch_fft_state *arch_fft`) that will need special handling — likely replace with an owned value or `Option`.

### 10.10 Architecture-Specific Dispatch (`arch` parameter)

Several functions take an `int arch` parameter for SIMD dispatch (e.g., `celt_pitch_xcorr`, `celt_fir`, `celt_inner_prod`, `compute_generic_dense`). The initial Rust port should use scalar implementations only, matching the `OPUS_ARCH_DEFAULT` path. The `arch` parameter can be carried through as a phantom type or enum for future SIMD optimization.

### 10.11 PLC PCM Buffer: Float Scaling

The PLC stores audio as float in [-1, 1] (divided by 32768), but the encoder expects audio in the float-as-int16 range ([-32768, 32768]). The PLC code manually scales back and forth:
```c
// Store: float = int16 / 32768
st->pcm[...] = (1.f/32768.f) * pcm[i];
// Retrieve: float_for_encoder = float * 32768
for (i=0;i<FRAME_SIZE;i++) x[i] = 32768.f * st->pcm[st->analysis_pos+i];
```
This scaling must be exact in the Rust port.

### 10.12 Struct Embedding and Partial Initialization

`LPCNetPLCState` embeds `LPCNetEncState` and `FARGANState` directly (not as pointers). The PLC init function initializes sub-components individually. In Rust, use nested structs with `Default` implementations or builder patterns.

### 10.13 Memory Aliasing in PLC Backup

The PLC state management performs struct copies for backup/restore:
```c
st->plc_bak[0] = st->plc_bak[1];
st->plc_bak[1] = st->plc_net;
// and
st->plc_net = st->plc_bak[0];
```
These are full struct copies of `PLCNetState` (containing GRU state arrays). In Rust, derive `Clone` on `PLCNetState` and use `clone()` or direct assignment if `Copy`.
