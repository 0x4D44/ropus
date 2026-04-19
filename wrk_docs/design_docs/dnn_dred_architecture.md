I now have all the information needed. Here is the complete architecture documentation.

---

# DRED Module Architecture Documentation

## Module: `dnn_dred` — Deep REDundancy Coding

---

## 1. Purpose

DRED (Deep REDundancy) provides neural-network-based forward error correction (FEC) for the Opus codec. It is an Opus extension (extension ID 126, experimental) introduced in Opus 1.4+ that embeds redundant representations of past audio frames inside the Opus bitstream. When packets are lost, the decoder uses these redundant representations to reconstruct lost frames with significantly better quality than traditional PLC (Packet Loss Concealment).

### Position in the Opus pipeline

- **Encoder side**: After the main SILK/CELT encoder produces its payload, DRED compresses recent audio into a compact latent representation using an RDOVAE (Rate-Distortion Optimized Variational AutoEncoder) and appends it as an Opus extension.
- **Decoder side**: When packet loss is detected, the decoder extracts DRED data from the last successfully received packet, entropy-decodes the latents, then runs the RDOVAE decoder to reconstruct feature frames that can be fed to FARGAN/LPCNet for waveform synthesis.

### Key design properties

- Covers up to ~1.04 seconds of audio history (26 latent "dframes", each covering 20ms = 4 × 5ms sub-frames)
- Uses progressive coding: newest frames are coded at highest quality, oldest at coarsest quantization
- The quality degrades gracefully with bitrate budget via adaptive quantizer stepping (`q0`, `dQ`, `qmax`)
- Voice activity detection suppresses encoding of silence periods

---

## 2. Public API

### 2.1 Encoder API (`dred_encoder.h`)

```c
int  dred_encoder_load_model(DREDEnc* enc, const void *data, int len);
void dred_encoder_init(DREDEnc* enc, opus_int32 Fs, int channels);
void dred_encoder_reset(DREDEnc* enc);
void dred_deinit_encoder(DREDEnc *enc);
void dred_compute_latents(DREDEnc *enc, const float *pcm, int frame_size,
                          int extra_delay, int arch);
int  dred_encode_silk_frame(DREDEnc *enc, unsigned char *buf, int max_chunks,
                            int max_bytes, int q0, int dQ, int qmax,
                            unsigned char *activity_mem, int arch);
```

| Function | Purpose | Parameters | Return |
|----------|---------|------------|--------|
| `dred_encoder_load_model` | Load RDOVAE + LPCNet weights from binary blob | `data`: weight blob, `len`: byte length | `OPUS_OK` on success, `OPUS_BAD_ARG` on failure |
| `dred_encoder_init` | Initialize encoder state, optionally load built-in weights | `Fs`: sample rate (8k/12k/16k/24k/48k/96k), `channels`: 1 or 2 | void |
| `dred_encoder_reset` | Zero all working buffers, reset LPCNet and RDOVAE state | — | void |
| `dred_deinit_encoder` | Release resources (currently a no-op in the C code) | — | void |
| `dred_compute_latents` | Feed PCM, compute RDOVAE latents for all complete dframes | `pcm`: float PCM input, `frame_size`: samples per channel, `extra_delay`: additional encoder delay in samples, `arch`: SIMD arch ID | void |
| `dred_encode_silk_frame` | Entropy-encode latents into DRED extension bytes | `buf`: output buffer, `max_chunks`: max latent pairs, `max_bytes`: byte budget, `q0`/`dQ`/`qmax`: quantizer params, `activity_mem`: VAD history | Number of bytes written (0 = no DRED emitted) |

### 2.2 Decoder API (`dred_decoder.h`)

```c
int dred_ec_decode(OpusDRED *dec, const opus_uint8 *bytes, int num_bytes,
                   int min_feature_frames, int dred_frame_offset);
```

| Function | Purpose | Parameters | Return |
|----------|---------|------------|--------|
| `dred_ec_decode` | Entropy-decode DRED extension bytes into latent vectors | `bytes`: DRED payload, `num_bytes`: payload length, `min_feature_frames`: max feature frames to decode, `dred_frame_offset`: position offset for multiframe packets | Number of latent pairs decoded |

### 2.3 RDOVAE Decoder API (`dred_rdovae_dec.h`)

```c
void dred_rdovae_dec_init_states(RDOVAEDecState *h, const RDOVAEDec *model,
                                  const float *initial_state, int arch);
void dred_rdovae_decode_qframe(RDOVAEDecState *h, const RDOVAEDec *model,
                                float *qframe, const float *z, int arch);
void DRED_rdovae_decode_all(const RDOVAEDec *model, float *features,
                             const float *state, const float *latents,
                             int nb_latents, int arch);
```

| Function | Purpose | Parameters | Return |
|----------|---------|------------|--------|
| `dred_rdovae_dec_init_states` | Initialize GRU states from the decoded initial state vector | `initial_state`: state vector of `DRED_STATE_DIM` floats | void |
| `dred_rdovae_decode_qframe` | Decode one latent vector into a quad-frame of features | `qframe`: output `4*DRED_NUM_FEATURES` floats, `z`: latent vector of `DRED_LATENT_DIM+1` floats | void |
| `DRED_rdovae_decode_all` | Convenience: init states + decode all latents sequentially | `features`: output buffer, `nb_latents`: count of latent pairs | void |

### 2.4 RDOVAE Encoder API (`dred_rdovae_enc.h`)

```c
void dred_rdovae_encode_dframe(RDOVAEEncState *enc_state, const RDOVAEEnc *model,
                                float *latents, float *initial_state,
                                const float *input, int arch);
```

| Function | Purpose | Parameters | Return |
|----------|---------|------------|--------|
| `dred_rdovae_encode_dframe` | Encode a double-frame of features into a latent vector + state | `latents`: output buffer (writes at index 0), `initial_state`: output state buffer (writes at index 0), `input`: `2*DRED_NUM_FEATURES` floats | void |

### 2.5 Quantizer Computation (`dred_coding.h`)

```c
int compute_quantizer(int q0, int dQ, int qmax, int i);
```

Computes the quantization level for the `i`-th latent pair:
```
quant = q0 + (dQ_table[dQ] * i + 8) / 16
return min(quant, qmax)
```

---

## 3. Internal State

### 3.1 `DREDEnc` (Encoder State)

```c
typedef struct {
    RDOVAEEnc model;                          // RDOVAE encoder neural network weights
    LPCNetEncState lpcnet_enc_state;          // LPCNet feature extraction state
    RDOVAEEncState rdovae_enc;                // RDOVAE encoder RNN hidden states
    int loaded;                                // 1 if model weights loaded successfully
    opus_int32 Fs;                            // Input sample rate
    int channels;                              // 1 (mono) or 2 (stereo)

    // --- Fields below are zeroed by dred_encoder_reset() ---
    float input_buffer[2*DRED_DFRAME_SIZE];   // 640 samples of 16kHz PCM (2 × 160-sample frames)
    int input_buffer_fill;                     // Current fill level (starts at DRED_SILK_ENCODER_DELAY = 11)
    int dred_offset;                           // Timing offset of newest latent relative to Opus frame
    int latent_offset;                         // Number of latents to skip (timing alignment)
    int last_extra_dred_offset;               // Saved offset for delayed DRED after silence
    float latents_buffer[DRED_MAX_FRAMES * DRED_LATENT_DIM];   // Ring buffer of RDOVAE latents
    int latents_buffer_fill;                   // How many latent vectors are valid
    float state_buffer[DRED_MAX_FRAMES * DRED_STATE_DIM];      // Ring buffer of RDOVAE initial states
    float resample_mem[RESAMPLING_ORDER + 1]; // IIR resampler filter memory (order 8)
} DREDEnc;
```

**Lifecycle**:
1. `dred_encoder_init()` — sets `Fs`, `channels`, loads built-in weights (if `USE_WEIGHTS_FILE` not defined), calls `reset`
2. `dred_encoder_load_model()` — (optional) loads external weight blob
3. Per-frame: `dred_compute_latents()` → `dred_encode_silk_frame()`
4. `dred_encoder_reset()` — clears everything from `input_buffer` onward, re-inits LPCNet and RDOVAE states

**Reset mechanism**: The `DREDENC_RESET_START` macro is defined as `input_buffer`, and reset uses pointer arithmetic to zero from that field to the end of the struct:
```c
OPUS_CLEAR((char*)&enc->DREDENC_RESET_START,
           sizeof(DREDEnc) - ((char*)&enc->DREDENC_RESET_START - (char*)enc));
```

### 3.2 `OpusDRED` (Decoder State)

```c
struct OpusDRED {
    float fec_features[2*DRED_NUM_REDUNDANCY_FRAMES*DRED_NUM_FEATURES]; // Decoded feature frames
    float state[DRED_STATE_DIM];                                         // Decoded initial state
    float latents[(DRED_NUM_REDUNDANCY_FRAMES/2)*(DRED_LATENT_DIM+1)];  // Decoded latent vectors + q_level
    int   nb_latents;     // Number of successfully decoded latent pairs
    int   process_stage;  // 0 = not decoded, 1 = latents decoded (needs RDOVAE decode)
    int   dred_offset;    // Temporal offset of DRED data relative to current packet
};
```

**Lifecycle**:
1. `dred_ec_decode()` fills `state`, `latents`, `nb_latents`, `dred_offset`, sets `process_stage = 1`
2. Caller invokes `DRED_rdovae_decode_all()` to fill `fec_features` from `state` + `latents`
3. Feature frames are then fed to FARGAN/LPCNet for waveform synthesis

### 3.3 `RDOVAEEncState` (RDOVAE Encoder Hidden State)

```c
struct RDOVAEEncStruct {
    int initialized;                           // Lazy conv init flag
    float gru1_state[ENC_GRU1_STATE_SIZE];    // 5 GRU hidden state vectors
    float gru2_state[ENC_GRU2_STATE_SIZE];
    float gru3_state[ENC_GRU3_STATE_SIZE];
    float gru4_state[ENC_GRU4_STATE_SIZE];
    float gru5_state[ENC_GRU5_STATE_SIZE];
    float conv1_state[ENC_CONV1_STATE_SIZE];  // 5 conv1d state buffers (conv1: dilation 1, conv2-5: dilation 2)
    float conv2_state[2*ENC_CONV2_STATE_SIZE];
    float conv3_state[2*ENC_CONV3_STATE_SIZE];
    float conv4_state[2*ENC_CONV4_STATE_SIZE];
    float conv5_state[2*ENC_CONV5_STATE_SIZE];
};
```

Note the `2*` multiplier on conv2–conv5: these use dilation=2 and thus need two slots of history.

### 3.4 `RDOVAEDecState` (RDOVAE Decoder Hidden State)

```c
struct RDOVAEDecStruct {
    int initialized;                           // Lazy conv init flag
    float gru1_state[DEC_GRU1_STATE_SIZE];    // 5 GRU hidden state vectors
    float gru2_state[DEC_GRU2_STATE_SIZE];
    float gru3_state[DEC_GRU3_STATE_SIZE];
    float gru4_state[DEC_GRU4_STATE_SIZE];
    float gru5_state[DEC_GRU5_STATE_SIZE];
    float conv1_state[DEC_CONV1_STATE_SIZE];  // 5 conv1d state buffers (all dilation 1)
    float conv2_state[DEC_CONV2_STATE_SIZE];
    float conv3_state[DEC_CONV3_STATE_SIZE];
    float conv4_state[DEC_CONV4_STATE_SIZE];
    float conv5_state[DEC_CONV5_STATE_SIZE];
};
```

Unlike the encoder, the decoder convolutions all use dilation=1.

---

## 4. Algorithm

### 4.1 Encoding Pipeline

#### Step 1: PCM → 16 kHz conversion (`dred_compute_latents` → `dred_convert_to_16k`)

1. Stereo → mono downmix: `0.5 * (left + right)`
2. Float → int16 conversion via `FLOAT2INT16()`
3. Upsample to intermediate rate (e.g., 8 kHz → 16 kHz with up=2)
4. Apply elliptic IIR anti-aliasing filter (7th order, DF2T structure)
5. Decimate to 16 kHz

The resampling filter coefficients are designed with MATLAB's `ellip(7, .2, 70, cutoff)`:
- 48kHz/24kHz: cutoff = 7750/24000
- 12kHz: cutoff = 5800/24000
- 8kHz: cutoff = 3900/8000
- 96kHz (QEXT only): cutoff = 7750/48000

#### Step 2: Feature extraction (`dred_process_frame`)

1. Accumulate 16 kHz PCM into `input_buffer` until 320 samples (2 × DRED_FRAME_SIZE = 2 × 160)
2. Compute LPCNet features for each 10 ms frame (160 samples), producing 36 features per frame
3. Discard LPC coefficients, keeping first `DRED_NUM_FEATURES` from each frame
4. Concatenate into a double-frame input: `[features_frame1 | features_frame2]` = `2 * DRED_NUM_FEATURES` floats

#### Step 3: RDOVAE encoding (`dred_rdovae_encode_dframe`)

The encoder neural network is a stack of 5 {Dense → GRU → Conv1D} blocks:

```
input (2*DRED_NUM_FEATURES floats)
  → Dense1 (tanh)
  → GRU1 → GLU output copy → Conv_dense1 (tanh) → Conv1d_1 (dilation=1, tanh)
  → GRU2 → GLU output copy → Conv_dense2 (tanh) → Conv1d_2 (dilation=2, tanh)
  → GRU3 → GLU output copy → Conv_dense3 (tanh) → Conv1d_3 (dilation=2, tanh)
  → GRU4 → GLU output copy → Conv_dense4 (tanh) → Conv1d_4 (dilation=2, tanh)
  → GRU5 → GLU output copy → Conv_dense5 (tanh) → Conv1d_5 (dilation=2, tanh)

concatenated buffer
  → enc_zdense (linear) → padded_latents → COPY first DRED_LATENT_DIM → latents[0]
  → gdense1 (tanh) → gdense2 (linear) → padded_state → COPY first DRED_STATE_DIM → state[0]
```

Key details:
- All intermediate outputs are concatenated into a growing `buffer[]`
- Each GRU/Conv block reads the full buffer up to that point (ever-growing context)
- Encoder Conv1d layers 2–5 use **dilation=2**; only Conv1d_1 uses dilation=1
- Latent and state outputs are padded to 8-byte alignment (`DRED_PADDED_LATENT_DIM`, `DRED_PADDED_STATE_DIM`)
- Only the first `DRED_LATENT_DIM` / `DRED_STATE_DIM` values are kept (the padding is discarded)

#### Step 4: Buffer management

Latents and states are stored newest-first in ring buffers:
```c
OPUS_MOVE(enc->latents_buffer + DRED_LATENT_DIM, enc->latents_buffer, (DRED_MAX_FRAMES-1)*DRED_LATENT_DIM);
OPUS_MOVE(enc->state_buffer + DRED_STATE_DIM, enc->state_buffer, (DRED_MAX_FRAMES-1)*DRED_STATE_DIM);
```
New data is written at index 0. This means `latents_buffer[0]` is always the most recent frame.

#### Step 5: Entropy coding (`dred_encode_silk_frame`)

**Header encoding** (bitstream format):
1. `q0` — base quantizer level, coded as `ec_enc_uint(q0, 16)` — 4 bits
2. `dQ` — quantizer increment index, coded as `ec_enc_uint(dQ, 8)` — 3 bits
3. Offset encoding — branched:
   - If `total_offset > 31`: code `1` (1 bit), then `total_offset>>5` as uint(256), then `total_offset&31` as uint(32)
   - If `total_offset <= 31`: code `0` (1 bit), then `total_offset` as uint(32)
4. `qmax` encoding — only if `q0 < 14 && dQ > 0`:
   - Combined symbol: probability split evenly between "qmax≥15" (coded as CDF range `[0, nvals)`) and specific qmax values (coded as CDF range `[nvals + qmax - (q0+1), nvals + qmax - q0)` out of `2*nvals`)

**State encoding**:
- The initial state vector is quantized and entropy-coded using Laplace coding with per-dimension parameters from `dred_state_quant_scales_q8`, `dred_state_dead_zone_q8`, `dred_state_r_q8`, `dred_state_p0_q8`, indexed by `q0 * DRED_STATE_DIM`

**Latent encoding** (progressive, newest → oldest):
- For each latent pair `i` (stepping by 2):
  1. Compute `q_level = compute_quantizer(q0, dQ, qmax, i/2)`
  2. Entropy-code the latent vector using `dred_encode_latents()` with per-dimension Laplace parameters from `dred_latent_*_q8` tables indexed by `q_level * DRED_LATENT_DIM`
  3. After each pair, check if byte budget exceeded; if so, roll back to last successful `ec_bak`
  4. Track voice activity; only keep coded data through the last active region

**Latent quantization** (`dred_encode_latents`):
```c
delta[i] = dzone[i] * (1.f/256.f);             // dead zone width
xq[i] = x[i] * scale[i] * (1.f/256.f);        // scaled input
deadzone[i] = xq[i] / (delta[i] + eps);        // soft dead zone ratio (eps=0.1)
deadzone[i] = tanh(deadzone[i]);                // smooth dead zone via tanh
xq[i] = xq[i] - delta[i] * deadzone[i];        // apply dead zone
q[i] = floor(0.5 + xq[i]);                      // round to nearest integer
```

Then each `q[i]` is coded with `ec_laplace_encode_p0(enc, q[i], p0[i]<<7, r[i]<<7)` unless `r[i]==0` or `p0[i]==255`, in which case `q[i]` is forced to 0 (deterministic symbol, not coded).

### 4.2 Decoding Pipeline

#### Step 1: Entropy decoding (`dred_ec_decode`)

Mirror of the encoder:
1. Decode header: `q0`, `dQ`, extra_offset, `dred_offset`, `qmax`
2. Decode initial state using `dred_decode_latents()` with state parameters
3. For each latent pair `i` (newest to oldest):
   - Compute `q_level = compute_quantizer(q0, dQ, qmax, i/2)`
   - Decode latent vector using `dred_decode_latents()` with latent parameters
   - Store decoded q_level as normalized float: `latents[(i/2)*(DRED_LATENT_DIM+1) + DRED_LATENT_DIM] = q_level * 0.125 - 1`
   - Stop if fewer than 8 bits remain

**Latent dequantization** (`dred_decode_latents`):
```c
if (r[i] == 0 || p0[i] == 255) q = 0;
else q = ec_laplace_decode_p0(dec, p0[i]<<7, r[i]<<7);
x[i] = q * 256.f / (scale[i] == 0 ? 1 : scale[i]);
```

Note the asymmetry: the encoder uses a tanh dead zone before quantization, but the decoder simply dequantizes by dividing by scale. The dead zone is a training-time rate-distortion optimization that doesn't need to be inverted.

#### Step 2: RDOVAE decoding

**State initialization** (`dred_rdovae_dec_init_states`):
```
initial_state (DRED_STATE_DIM floats)
  → dec_hidden_init (Dense, tanh) → hidden
  → dec_gru_init (Dense, tanh) → state_init
  → split into gru1..gru5 states (by cumulative offset)
```

**Quad-frame decoding** (`dred_rdovae_decode_qframe`):

The decoder network mirrors the encoder structure with 5 {Dense → GRU → GLU → Conv1D} blocks:

```
latent input (DRED_LATENT_DIM+1 floats, including q_level)
  → Dense1 (tanh)
  → GRU1 → GLU1 → Conv_dense1 (tanh) → Conv1d_1 (dilation=1, tanh)
  → GRU2 → GLU2 → Conv_dense2 (tanh) → Conv1d_2 (dilation=1, tanh)
  → GRU3 → GLU3 → Conv_dense3 (tanh) → Conv1d_3 (dilation=1, tanh)
  → GRU4 → GLU4 → Conv_dense4 (tanh) → Conv1d_4 (dilation=1, tanh)
  → GRU5 → GLU5 → Conv_dense5 (tanh) → Conv1d_5 (dilation=1, tanh)

concatenated buffer → dec_output (Dense, linear) → qframe (4*DRED_NUM_FEATURES floats)
```

Key differences from encoder:
- Decoder uses **GLU** (Gated Linear Unit) instead of raw GRU output copy
- All decoder Conv1d layers use **dilation=1** (encoder uses dilation=2 for layers 2-5)
- Output is 4 feature frames (one "quad-frame") per latent vector
- Conv states are lazily initialized on first call via `conv1_cond_init()`

**Bulk decode** (`DRED_rdovae_decode_all`):
```c
dred_rdovae_dec_init_states(&dec, model, state, arch);
for (i = 0; i < 2*nb_latents; i += 2) {
    dred_rdovae_decode_qframe(&dec, model,
        &features[2*i*DRED_NUM_FEATURES],
        &latents[(i/2)*(DRED_LATENT_DIM+1)], arch);
}
```
This iterates over latent pairs, producing `4*DRED_NUM_FEATURES` floats per pair, filling the features buffer sequentially.

---

## 5. Data Flow

### 5.1 Encoder Data Flow

```
PCM (float, any supported rate, mono/stereo)
  │
  ▼
dred_compute_latents()
  ├── dred_convert_to_16k() ──→ 16 kHz mono float PCM
  │     ├── Stereo downmix
  │     ├── FLOAT2INT16 + upsample
  │     ├── IIR anti-aliasing filter (DF2T)
  │     └── Decimation
  │
  ├── Accumulate into input_buffer[640]
  │
  └── When 320 samples ready → dred_process_frame()
        ├── LPCNet feature extraction (2 × 160 samples → 2 × 36 features)
        ├── Select first DRED_NUM_FEATURES from each frame
        ├── dred_rdovae_encode_dframe()
        │     ├── 5-layer GRU+Conv1d encoder stack
        │     ├── → latent vector (DRED_LATENT_DIM floats)
        │     └── → initial state (DRED_STATE_DIM floats)
        └── Push to front of latents_buffer[] and state_buffer[]

dred_encode_silk_frame()
  ├── VAD check: skip silent frames
  ├── Entropy-code header (q0, dQ, offset, qmax)
  ├── Entropy-code initial state (Laplace coding)
  ├── For each latent pair (newest → oldest):
  │     ├── compute_quantizer() → q_level
  │     ├── dred_encode_latents() (dead zone + Laplace coding)
  │     └── Budget check; stop or rollback if exceeded
  └── → buf[] (DRED extension bytes)
```

### 5.2 Decoder Data Flow

```
DRED extension bytes
  │
  ▼
dred_ec_decode()
  ├── Decode header (q0, dQ, offset, qmax)
  ├── Decode initial state (Laplace decoding → DRED_STATE_DIM floats)
  ├── For each latent pair (newest → oldest):
  │     ├── compute_quantizer() → q_level
  │     ├── dred_decode_latents() (Laplace decoding)
  │     └── Store latent + normalized q_level
  └── → OpusDRED.state[], OpusDRED.latents[], nb_latents

DRED_rdovae_decode_all()
  ├── dred_rdovae_dec_init_states() (state → 5 GRU hidden states)
  └── For each latent pair:
        └── dred_rdovae_decode_qframe()
              ├── 5-layer GRU+GLU+Conv1d decoder stack
              └── → 4×DRED_NUM_FEATURES floats per latent

  → fec_features[] (up to 2*DRED_NUM_REDUNDANCY_FRAMES*DRED_NUM_FEATURES floats)
     ↓
  Fed to FARGAN/LPCNet for waveform synthesis
```

### 5.3 Buffer Layouts

**Latents buffer** (encoder, `latents_buffer[DRED_MAX_FRAMES * DRED_LATENT_DIM]`):
```
[newest_latent (DRED_LATENT_DIM)] [second_newest] ... [oldest]
 ↑ index 0                                              ↑ index (fill-1)*DRED_LATENT_DIM
```
Newest is always at index 0; shifted right by `OPUS_MOVE` on each new frame.

**Decoded latents** (decoder, `latents[(DRED_NUM_REDUNDANCY_FRAMES/2) * (DRED_LATENT_DIM+1)]`):
```
For latent pair i:
  latents[i*(DRED_LATENT_DIM+1) .. i*(DRED_LATENT_DIM+1)+DRED_LATENT_DIM-1] = latent vector
  latents[i*(DRED_LATENT_DIM+1)+DRED_LATENT_DIM] = q_level * 0.125 - 1  (normalized quantizer info)
```
The extra float at position `DRED_LATENT_DIM` in each slot carries the quantization level as a side-channel to the decoder network.

**Feature output** (decoder, `fec_features[2*DRED_NUM_REDUNDANCY_FRAMES*DRED_NUM_FEATURES]`):
```
For latent pair i, the quad-frame output occupies:
  features[2*i*2*DRED_NUM_FEATURES .. 2*i*2*DRED_NUM_FEATURES + 4*DRED_NUM_FEATURES - 1]
```
That is, each decoded latent produces 4 consecutive feature frames of `DRED_NUM_FEATURES` floats each. Since we step `i` by 2, the stride is `4*DRED_NUM_FEATURES` per decoded latent.

---

## 6. Numerical Details

### 6.1 Floating-Point Arithmetic

The entire DRED module operates in **single-precision float** (32-bit IEEE 754). There is no fixed-point path. All neural network computations, feature extraction, and quantization arithmetic use `float`.

### 6.2 Q-Formats in Statistical Tables

The per-dimension quantization parameters are stored as **Q8 unsigned 8-bit integers** (`opus_uint8`), derived during weight export:

| Table | Derivation | Interpretation |
|-------|-----------|----------------|
| `quant_scales_q8` | `round(quant_scales * 2^8)` where `quant_scales` is normalized so max × 256/255 = 1 | Multiply by `1/256` to get float scale |
| `dead_zone_q8` | `clip(round(dead_zone * 2^8), 0, 255)` | Multiply by `1/256` to get float dead zone |
| `r_q8` | `clip(round(sigmoid(w) * 2^8), 0, 255)` | Laplace decay parameter; shift left by 7 before passing to `ec_laplace_*_p0` |
| `p0_q8` | `clip(round((1 - r^(0.5+0.5*sigmoid(w))) * 2^8), 0, 255)` | Probability of zero; shift left by 7 before passing to `ec_laplace_*_p0` |

**Laplace coding parameter scaling**: The `ec_laplace_encode_p0`/`ec_laplace_decode_p0` functions expect `p0` and `decay` in Q15 (i.e., scaled by 32768). The DRED code stores them as Q8 and shifts left by 7 at call sites:
```c
ec_laplace_encode_p0(enc, q[i], p0[i]<<7, r[i]<<7);  // Q8 << 7 = Q15
```

### 6.3 Quantizer Progression

The `dQ_table` maps the 3-bit `dQ` index to a fractional step size:
```c
static const int dQ_table[8] = {0, 2, 3, 4, 6, 8, 12, 16};
```

The quantizer for the `i`-th latent pair is:
```
q_level = min(q0 + (dQ_table[dQ] * i + 8) / 16, qmax)
```
This is integer arithmetic with rounding: `+8` provides rounding towards nearest for the `/16` division.

### 6.4 Quantizer Normalization in Latent Slot

The decoder stores a normalized quantizer value as the `(DRED_LATENT_DIM+1)`-th element:
```c
dec->latents[(i/2)*(DRED_LATENT_DIM+1)+DRED_LATENT_DIM] = q_level * 0.125f - 1.0f;
```
This maps the range [0, 15] to [-1.0, 0.875] and is passed as an extra input to the RDOVAE decoder, allowing it to adapt its reconstruction based on quantization coarseness.

### 6.5 Dead Zone Quantization (Encoder Only)

The encoder applies a soft dead zone using `tanh`:
```c
eps = 0.1f;
delta = dzone * (1/256);
xq = x * scale * (1/256);
deadzone_ratio = xq / (delta + eps);
deadzone_ratio = tanh(deadzone_ratio);     // Smooth soft dead zone
xq = xq - delta * deadzone_ratio;
q = floor(0.5 + xq);                       // Round to nearest
```

The `tanh` acts as a smooth dead zone: for small `|xq|` relative to `delta`, the `deadzone_ratio` is small and `xq` is reduced toward zero; for large `|xq|`, `tanh ≈ ±1` and `xq` is shifted by ±`delta`. The `eps=0.1` prevents division by zero when `delta=0`.

### 6.6 Dequantization (Decoder)

The decoder dequantization is simpler — no dead zone inversion:
```c
x[i] = q * 256.0f / (scale[i] == 0 ? 1 : scale[i]);
```
Division-by-zero guard: `scale[i] == 0` is replaced with 1.

### 6.7 Precision Concerns for Rust Port

- `floor(0.5f + xq)` — must match C `floorf()` behavior exactly. In Rust, use `(xq + 0.5f32).floor()` or `f32::round()` noting that Rust's `round()` uses "round half away from zero" while `floor(0.5 + x)` rounds half up. For negative values these differ. The C code uses `floor(.5f+xq)` which rounds toward negative infinity after adding 0.5.
- `tanh` — the `compute_activation()` function is called vectorized; this must produce identical results. The C implementation may use fast approximations (check `nnet.c`).
- The resampling filter coefficients are specified as `float` literals; ensure the Rust port uses the same values (not `f64` intermediates that get rounded differently).

---

## 7. Dependencies

### 7.1 What DRED Calls

| Module | Functions Used | Purpose |
|--------|---------------|---------|
| **celt/entenc** | `ec_enc_init`, `ec_enc_uint`, `ec_encode`, `ec_enc_shrink`, `ec_enc_done`, `ec_tell`, `ec_enc_icdf16` | Range coder encoder |
| **celt/entdec** | `ec_dec_init`, `ec_dec_uint`, `ec_decode`, `ec_dec_update`, `ec_tell`, `ec_dec_icdf16` | Range coder decoder |
| **celt/laplace** | `ec_laplace_encode_p0`, `ec_laplace_decode_p0` | Laplace-distributed entropy coding |
| **dnn/nnet** | `compute_generic_dense`, `compute_generic_gru`, `compute_generic_conv1d`, `compute_generic_conv1d_dilation`, `compute_glu`, `compute_activation`, `parse_weights` | Neural network primitives |
| **dnn/lpcnet** | `lpcnet_encoder_init`, `lpcnet_encoder_load_model`, `lpcnet_compute_single_frame_features_float` | LPCNet feature extraction |
| **os_support** | `OPUS_COPY`, `OPUS_MOVE`, `OPUS_CLEAR`, `opus_free` | Memory operations |

### 7.2 What Calls DRED

| Caller | Functions Called | Context |
|--------|----------------|---------|
| **Opus encoder** (`opus_encoder.c`) | `dred_compute_latents`, `dred_encode_silk_frame` | Appends DRED extension data to Opus packets |
| **Opus decoder** (`opus_decoder.c`) | `dred_ec_decode`, `DRED_rdovae_decode_all` | Decodes DRED on packet loss |
| **FARGAN demo** (`fargan_demo.c`) | `dred_decode_latents`, `DRED_rdovae_decode_all` | Standalone testing tool |

---

## 8. Constants and Tables

### 8.1 Compile-Time Constants (`dred_config.h`)

| Constant | Value | Meaning |
|----------|-------|---------|
| `DRED_EXTENSION_ID` | 126 | Opus extension number |
| `DRED_EXPERIMENTAL_VERSION` | 12 | Experimental protocol version |
| `DRED_EXPERIMENTAL_BYTES` | 2 | Extra header bytes for experimental mode |
| `DRED_MIN_BYTES` | 8 | Minimum DRED payload size |
| `DRED_SILK_ENCODER_DELAY` | `79+12-80` = **11** | Samples of SILK encoder lookahead at 16 kHz |
| `DRED_FRAME_SIZE` | 160 | 10 ms at 16 kHz |
| `DRED_DFRAME_SIZE` | 320 | Double-frame = 20 ms |
| `DRED_MAX_DATA_SIZE` | 1000 | Maximum DRED extension payload bytes |
| `DRED_ENC_Q0` | 6 | Default initial quantizer |
| `DRED_ENC_Q1` | 15 | Default max quantizer |
| `DRED_MAX_LATENTS` | 26 | Maximum latent pairs (covers ~1.04s) |
| `DRED_NUM_REDUNDANCY_FRAMES` | 52 | = `2 * DRED_MAX_LATENTS` |
| `DRED_MAX_FRAMES` | 104 | = `4 * DRED_MAX_LATENTS` |
| `RESAMPLING_ORDER` | 8 | Order of the anti-aliasing IIR filter |

### 8.2 Generated Constants (`dred_rdovae_constants.h`, auto-generated)

These are generated by `export_rdovae_weights.py` and depend on the trained model:

| Constant | Typical Value | Meaning |
|----------|--------------|---------|
| `DRED_NUM_FEATURES` | 20 | LPCNet features per frame (36 total minus 16 LPC coefficients) |
| `DRED_LATENT_DIM` | Model-dependent | Latent vector dimension (after pruning dead dimensions) |
| `DRED_STATE_DIM` | Model-dependent | Initial state vector dimension (after pruning) |
| `DRED_PADDED_LATENT_DIM` | `(DRED_LATENT_DIM+7)&~7` | 8-byte aligned latent dim |
| `DRED_PADDED_STATE_DIM` | `(DRED_STATE_DIM+7)&~7` | 8-byte aligned state dim |
| `DRED_NUM_QUANTIZATION_LEVELS` | 16 | Number of quantization levels (= `qembedding.shape[0]`) |
| `DRED_MAX_CONV_INPUTS` | Model-dependent | Max Conv1d input buffer size |
| `ENC_*_STATE_SIZE`, `DEC_*_STATE_SIZE` | Model-dependent | GRU/Conv state dimensions |

### 8.3 Quantizer Step Table

```c
static const int dQ_table[8] = {0, 2, 3, 4, 6, 8, 12, 16};
```

Maps the 3-bit `dQ` index to the per-latent-pair quantizer increment (in units of 1/16). The effective step per latent pair is `dQ_table[dQ] / 16`:

| dQ index | dQ_table | Effective step/pair |
|----------|----------|-------------------|
| 0 | 0 | 0.000 (constant quality) |
| 1 | 2 | 0.125 |
| 2 | 3 | 0.1875 |
| 3 | 4 | 0.250 |
| 4 | 6 | 0.375 |
| 5 | 8 | 0.500 |
| 6 | 12 | 0.750 |
| 7 | 16 | 1.000 |

### 8.4 Resampling Filter Coefficients

Stored as `static const float` arrays within `dred_convert_to_16k()`. Each variant is an 8th-order IIR filter in Direct Form II Transposed with separate numerator coefficients `b[8]`, denominator coefficients `a[8]`, and `b0` gain. The filters are elliptic designs with 0.2 dB passband ripple and 70 dB stopband attenuation.

### 8.5 Statistical Model Tables (Auto-Generated)

Generated by `dump_statistical_model()` in `export_rdovae_weights.py`:

| Array | Shape | Type | Indexing |
|-------|-------|------|----------|
| `dred_state_quant_scales_q8` | `[16 × DRED_STATE_DIM]` | `opus_uint8` | `[q_level * DRED_STATE_DIM + dim]` |
| `dred_state_dead_zone_q8` | `[16 × DRED_STATE_DIM]` | `opus_uint8` | Same |
| `dred_state_r_q8` | `[16 × DRED_STATE_DIM]` | `opus_uint8` | Same |
| `dred_state_p0_q8` | `[16 × DRED_STATE_DIM]` | `opus_uint8` | Same |
| `dred_latent_quant_scales_q8` | `[16 × DRED_LATENT_DIM]` | `opus_uint8` | `[q_level * DRED_LATENT_DIM + dim]` |
| `dred_latent_dead_zone_q8` | `[16 × DRED_LATENT_DIM]` | `opus_uint8` | Same |
| `dred_latent_r_q8` | `[16 × DRED_LATENT_DIM]` | `opus_uint8` | Same |
| `dred_latent_p0_q8` | `[16 × DRED_LATENT_DIM]` | `opus_uint8` | Same |

16 quantization levels × dimension, laid out row-major by quantization level.

The pruning step during export removes dimensions where `max(r_q8) == 0` or `min(p0_q8) == 255` (i.e., always-zero dimensions). The corresponding columns in the latent/state weight matrices are removed and the remaining columns are rescaled by `1/scale_norm`.

---

## 9. Edge Cases

### 9.1 Insufficient Bitrate Budget

- `dred_encode_silk_frame()` returns 0 if the initial state alone exceeds `max_bytes`
- If the first latent pair exceeds budget (`i==0`), returns 0 (no DRED at all)
- Otherwise, rolls back to the last successful `ec_bak` snapshot
- Empty DRED packets are suppressed: returns 0 if `dred_encoded==0` or `(dred_encoded<=2 && extra_dred_offset)`

### 9.2 Silence Suppression

- `dred_voice_active()` checks 16 consecutive activity bytes; returns 1 if any is active
- The encoder skips latents over silent regions by incrementing `latent_offset`
- After silence → voice transition, DRED encoding is delayed by one frame (`delayed_dred` mechanism) because the main Opus payload already covers the transition frame

### 9.3 Decoder Bit Exhaustion

```c
if (8*num_bytes - ec_tell(&ec) <= 7)
    break;
```
Decoding stops when fewer than 8 bits remain, even if more latent pairs are expected.

### 9.4 Division by Zero Guards

- Dequantization: `scale[i] == 0 ? 1 : scale[i]` — prevents division by zero
- Dead zone computation: `delta[i] + eps` where `eps = 0.1f` — prevents division by zero even when `dzone[i] == 0`

### 9.5 Deterministic Dimensions

When `r[i] == 0` or `p0[i] == 255`, the quantized value is forced to 0 without entropy coding. This represents dimensions that the model has learned are uninformative and should always be zero.

### 9.6 Model Not Loaded

- `dred_process_frame()` asserts `enc->loaded`
- `dred_compute_latents()` asserts `enc->loaded`
- If `USE_WEIGHTS_FILE` is defined, built-in weights are not available; `loaded` remains 0 until `dred_encoder_load_model()` succeeds

### 9.7 Multiframe Offset

The `dred_frame_offset` parameter in `dred_ec_decode()` accounts for DRED's position within multiframe Opus packets. The total offset computation:
```c
dec->dred_offset = 16 - ec_dec_uint(&ec, 32) - extra_offset + dred_frame_offset;
```

---

## 10. Porting Notes

### 10.1 Struct Reset via Pointer Arithmetic

The encoder reset uses C pointer arithmetic to zero a portion of the struct:
```c
#define DREDENC_RESET_START input_buffer
OPUS_CLEAR((char*)&enc->DREDENC_RESET_START,
           sizeof(DREDEnc) - ((char*)&enc->DREDENC_RESET_START - (char*)enc));
```

**Rust approach**: In Rust, this cannot be done directly. Options:
1. Split `DREDEnc` into a "persistent" sub-struct (model, lpcnet_state, loaded, Fs, channels) and a "resettable" sub-struct that derives `Default`. Reset = assign `Default::default()`.
2. Implement a custom `reset()` method that zeroes each field individually.

Option 1 is cleaner and the struct layout naturally separates into these two halves.

### 10.2 In-Place Buffer Shifts

```c
OPUS_MOVE(enc->latents_buffer + DRED_LATENT_DIM, enc->latents_buffer,
          (DRED_MAX_FRAMES - 1) * DRED_LATENT_DIM);
```

This is a `memmove` (overlapping regions). In Rust, use `slice.copy_within()`:
```rust
let len = (DRED_MAX_FRAMES - 1) * DRED_LATENT_DIM;
self.latents_buffer.copy_within(0..len, DRED_LATENT_DIM);
```

### 10.3 Growing Concatenation Buffer

Both encoder and decoder use a stack-allocated `buffer[]` that accumulates the outputs of each layer, with an `output_index` that advances:
```c
float buffer[ENC_DENSE1_OUT_SIZE + ENC_GRU1_OUT_SIZE + ... + ENC_CONV5_OUT_SIZE];
int output_index = 0;
compute_generic_dense(&model->enc_dense1, &buffer[output_index], input, ACTIVATION_TANH, arch);
output_index += ENC_DENSE1_OUT_SIZE;
// Each subsequent layer reads buffer[0..output_index] and writes at buffer[output_index..]
```

**Rust approach**: This is a large stack allocation of a fixed-size array. Rust can handle this directly. The tricky part is that each `compute_*` call needs a mutable slice at `output_index` while passing the entire buffer (including already-written portions) as the input. This requires either:
- Using split_at_mut() to get non-overlapping mutable and immutable slices
- Using indices with unsafe
- Restructuring to use separate input/output buffers

The cleanest Rust approach: since all the `compute_*` functions read from `buffer[0..output_index]` and write to `buffer[output_index..output_index+N]`, use `split_at_mut(output_index)` to get `(&[input_region], &mut [output_region])`.

### 10.4 Encoder `ec_bak` Snapshot/Rollback

```c
ec_enc ec_bak;
ec_bak = ec_encoder;
// ... try encoding ...
if (over_budget) {
    ec_encoder = ec_bak;  // Roll back
}
```

The `ec_enc` struct is copied by value for snapshotting. In Rust, `ec_enc` must implement `Clone` (or equivalent). Since `ec_enc` contains an internal pointer to the output buffer, care must be taken that the clone doesn't own the buffer. The C code works because `ec_enc` stores a raw pointer to the external `buf[]` — both original and copy point to the same buffer. In Rust, this would need a reference or raw pointer approach.

### 10.5 The `arch` Parameter

Most functions take an `int arch` parameter used for SIMD dispatch (SSE2/SSE4.1/AVX2/NEON). Since the initial Rust port skips platform-specific SIMD, this parameter can be a unit type or a const. However, the code must still call through the same function signatures (e.g., `compute_activation` dispatches based on `arch`).

### 10.6 Conditional Compilation

- `HAVE_CONFIG_H` — build system configuration
- `USE_WEIGHTS_FILE` — controls whether built-in weights are compiled in
- `ENABLE_QEXT` — enables 96 kHz support in resampler
- SIMD arch selection in `nnet.h` / `dnn_x86.h` / `dnn_arm.h`

For the Rust port, these map to Cargo features.

### 10.7 Auto-Generated Weight/Stats Files

The files `dred_rdovae_enc_data.{c,h}`, `dred_rdovae_dec_data.{c,h}`, `dred_rdovae_stats_data.{c,h}`, and `dred_rdovae_constants.h` are **generated** by `dnn/torch/rdovae/export_rdovae_weights.py` from a trained PyTorch checkpoint. The Rust port must either:
1. Run the same export script to generate Rust code (preferred)
2. Parse the binary weight blob at runtime (supported via `parse_weights()`)
3. Include the C-generated data and access via FFI (test harness only)

### 10.8 `FLOAT2INT16` Macro

Used in `dred_convert_to_16k()`:
```c
downmix[up*i] = FLOAT2INT16(up*in[i]) + VERY_SMALL;
```
This converts float to int16 range with clamping. `VERY_SMALL` (likely ~1e-15) is added as a dither-like bias to avoid exact zeros. The Rust port needs the exact same conversion and the same `VERY_SMALL` constant.

### 10.9 `IMAX`/`IMIN` Macros

Used for array sizing:
```c
int q[IMAX(DRED_LATENT_DIM, DRED_STATE_DIM)];
```
In Rust, use `const` generics or compute the max at compile time.

### 10.10 Neural Network Weight Structure

The `RDOVAEEnc` and `RDOVAEDec` model structs are auto-generated and contain `LinearLayer` fields for each layer. These reference weight data either compiled in (static arrays) or loaded from a binary blob. In Rust, the model struct should hold references (`&[f32]`, `&[i8]`, etc.) to weight data, with lifetime tied to either static data or a loaded weight buffer.

### 10.11 Laplace Coding p0 Function Interface

The `ec_laplace_encode_p0` / `ec_laplace_decode_p0` functions use an escape-coded geometric distribution with a 3-symbol sign coding step. The decode function can produce unbounded integers in theory (the `do { v = decode; value += v; } while (v == 7)` loop). In practice, the range coder constrains this, but the Rust port should handle arbitrarily large decoded values safely.

### 10.12 Filter State Persistence

The DF2T resampling filter state (`resample_mem[9]`) persists across calls to `dred_convert_to_16k()`. It is zeroed by `dred_encoder_reset()` as part of the bulk clear. The filter is computed in-place on the `downmix[]` buffer — the same buffer is both input and output for `filter_df2t()` when `in == out`.

### 10.13 GRU State Input/Output Aliasing

In the encoder, GRU state is both read and updated in place:
```c
compute_generic_gru(&model->enc_gru1_input, &model->enc_gru1_recurrent,
                    enc_state->gru1_state, buffer, arch);
OPUS_COPY(&buffer[output_index], enc_state->gru1_state, ENC_GRU1_OUT_SIZE);
```

In the decoder, a GLU layer transforms the GRU output before concatenation:
```c
compute_generic_gru(&model->dec_gru1_input, &model->dec_gru1_recurrent,
                    dec_state->gru1_state, buffer, arch);
compute_glu(&model->dec_glu1, &buffer[output_index], dec_state->gru1_state, arch);
```

The `compute_generic_gru` modifies state in-place. The Rust port must ensure mutable access to the GRU state while passing the buffer as read-only input.
