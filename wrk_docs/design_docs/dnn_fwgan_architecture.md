Now I have all the information needed. Here is the complete architecture document.

---

# FWGAN Module Architecture Document

## 1. Purpose

FWGAN (**F**rame**W**ise **G**enerative **A**dversarial **N**etwork) is a neural waveform synthesizer in the Opus 1.4+ DNN subsystem. It generates speech audio frame-by-frame from compact feature representations, operating as an alternative synthesis backend within the OSCE (Opus Speech Coding Enhancement) pipeline.

The module synthesizes 160-sample PCM frames (10 ms at 16 kHz) by:
1. Converting cepstral features into per-subframe conditioning vectors
2. Generating pitch-synchronous phase embeddings
3. Running a cascade of neural network layers (GRU + 7 framewise convolutions) to produce excitation signals
4. Applying LPC synthesis filtering with pre/de-emphasis to shape the output spectrum

FWGAN is trained as a GAN (the generator half ships in the codec; the discriminator is training-only). The C code implements only forward inference of the generator. The default model variant is **FWGAN400ContLarge** (~400k parameters).

## 2. Public API

### `fwgan_init`

```c
void fwgan_init(FWGANState *st);
```

Initializes an `FWGANState` from the compiled-in default weights (`fwgan_arrays`).

- **`st`**: Pointer to caller-allocated `FWGANState`. Zero-cleared and populated.
- **Side effects**: Calls `init_fwgan(&st->model, fwgan_arrays)`. Asserts on failure.
- **Note**: Does **not** perform CPU architecture detection (`st->arch` remains 0). The code has a `FIXME` comment about this.

### `fwgan_load_model`

```c
int fwgan_load_model(FWGANState *st, const unsigned char *data, int len);
```

Loads model weights from a binary blob (for runtime model switching).

- **`data`**: Pointer to serialized weight data.
- **`len`**: Length in bytes.
- **Returns**: `0` on success, `-1` on failure.
- **Mechanism**: Calls `parse_weights()` to deserialize into a `WeightArray` list, then `init_fwgan()` to populate `st->model`. Frees the weight list after loading.

### `fwgan_cont`

```c
void fwgan_cont(FWGANState *st, const float *pcm0, const float *features0);
```

Initializes FWGAN state from a continuation context (previous PCM history), enabling seamless splicing into an ongoing audio stream.

- **`pcm0`**: 320 samples of preceding PCM (float, in [-1, 1] range).
- **`features0`**: Feature vector for the continuation frame (20 floats, standard LPCNet feature layout).
- **Side effects**: Populates all neural network hidden states via a 6-layer continuation network, sets filter memories, runs one warmup frame (discards first subframe, saves remaining 3 to `pcm_buf`).
- **Must be called before** `fwgan_synthesize()`.

### `fwgan_synthesize`

```c
void fwgan_synthesize(FWGANState *st, float *pcm, const float *features);
```

Synthesizes one 160-sample frame of float PCM.

- **`pcm`**: Output buffer, 160 floats. Values are in approximately [-1, 1] range (not hard-clamped).
- **`features`**: 20-element LPCNet feature vector for this frame.
- **Buffering**: Output is delayed by 3 subframes (120 samples). See §5 for layout.
- **Precondition**: `fwgan_cont()` must have been called (asserts `cont_initialized != 0`).

### `fwgan_synthesize_int`

```c
void fwgan_synthesize_int(FWGANState *st, opus_int16 *pcm, const float *features);
```

Same as `fwgan_synthesize` but outputs 16-bit integer PCM.

- **Conversion**: `floor(0.5 + clamp(32768.0 * fpcm[i], -32767, 32767))` — note the **asymmetric** clamp (min is -32767, not -32768).
- **Loop bound**: Uses `LPCNET_FRAME_SIZE` (= 160, same as `FWGAN_FRAME_SIZE`).

## 3. Internal State

### `FWGANState` structure

```c
typedef struct {
  FWGAN model;                                    // All neural network layer weights
  int arch;                                       // CPU architecture ID (for SIMD dispatch)
  int cont_initialized;                           // 0=uninitialized, 1=first-subframe, 2=running
  float embed_phase[2];                           // Phase oscillator state [cos, sin]
  float last_gain;                                // Gain from previous subframe
  float last_lpc[LPC_ORDER];                      // LPC coefficients from previous subframe
  float syn_mem[LPC_ORDER];                       // LPC synthesis filter memory (16 samples)
  float preemph_mem;                              // Pre-emphasis filter memory (1 sample)
  float deemph_mem;                               // De-emphasis filter memory (1 sample)
  float pcm_buf[FWGAN_FRAME_SIZE];                // Output buffering (160 samples)
  float cont[CONT_NET_10_OUT_SIZE];               // Continuation latent vector (64 floats)
  float cont_conv1_mem[FEAT_IN_CONV1_CONV_STATE_SIZE]; // feat_in conv1d state
  float rnn_state[RNN_GRU_STATE_SIZE];            // GRU hidden state (256 floats)
  float fwc1_state[FWC1_STATE_SIZE];              // Framewise conv layer 1 state (512)
  float fwc2_state[FWC2_STATE_SIZE];              // Framewise conv layer 2 state (512)
  float fwc3_state[FWC3_STATE_SIZE];              // Framewise conv layer 3 state (256)
  float fwc4_state[FWC4_STATE_SIZE];              // Framewise conv layer 4 state (256)
  float fwc5_state[FWC5_STATE_SIZE];              // Framewise conv layer 5 state (128)
  float fwc6_state[FWC6_STATE_SIZE];              // Framewise conv layer 6 state (128)
  float fwc7_state[FWC7_STATE_SIZE];              // Framewise conv layer 7 state (80)
} FWGANState;
```

### Lifecycle

```
fwgan_init() or fwgan_load_model()   →   cont_initialized = 0
         ↓
fwgan_cont(pcm0, features0)         →   cont_initialized = 1 → 2
         ↓                                (runs 1 warmup frame internally)
fwgan_synthesize() (repeated)        →   cont_initialized stays 2
```

### `cont_initialized` state machine

| Value | Meaning |
|-------|---------|
| `0` | Not initialized. `fwgan_synthesize` will assert-fail. |
| `1` | Set at start of `fwgan_cont`. During the first subframe of the warmup frame, `run_fwgan_subframe` skips the GRU+FWC cascade, outputs zeros, and transitions to `2`. |
| `2` | Fully running. All subsequent subframes execute the full neural network pipeline. |

### `FWGAN` model struct

The `FWGAN` struct (defined in generated `fwgan_data.h`) contains `LinearLayer` fields for every neural network layer. For the FWGAN400 model, these are:

| Field | Type | Description |
|-------|------|-------------|
| `bfcc_with_corr_upsampler_fc` | LinearLayer | Feature upsampler: 19→320 |
| `cont_net_0` through `cont_net_10` | LinearLayer (×6) | Continuation network layers |
| `feat_in_conv1_conv` | LinearLayer | Feature input convolution |
| `feat_in_nl1_gate` | LinearLayer | Feature input gated activation |
| `rnn_gru_input`, `rnn_gru_recurrent` | LinearLayer (×2) | GRU input and recurrent weights |
| `rnn_nl_gate` | LinearLayer | GRU output gated activation |
| `fwcN_fc_0` | LinearLayer (×7) | FWC layer N linear transform |
| `fwcN_fc_1_gate` | LinearLayer (×7) | FWC layer N gated activation |
| `rnn_cont_fc_0` | LinearLayer | Continuation→GRU state mapping |
| `fwcN_cont_fc_0` | LinearLayer (×7) | Continuation→FWC-N state mapping |

## 4. Algorithm

### 4.1 Continuation (`fwgan_cont`)

Initializes all hidden states from PCM history so synthesis can splice seamlessly into an ongoing audio stream.

**Step 1 — LPC analysis of continuation PCM:**
```c
compute_wlpc(lpc, features0);     // Cepstrum → LPC with bandwidth expansion (γ=0.92)
```

**Step 2 — Filter memory initialization:**
```c
st->deemph_mem = pcm0[319];       // De-emphasis memory = last continuation sample
st->preemph_mem = wpcm0[319];     // Pre-emphasis memory = last weighted sample
st->syn_mem[i] = pcm0[319-i] - 0.85*pcm0[318-i];  // Pre-emphasized PCM, reversed
```

**Step 3 — LPC analysis filtering:**
```c
// Apply analysis filter A(z) = 1 + Σ lpc[j]·z^(-j-1) to get weighted/residual signal
for i in [LPC_ORDER..320):
    wpcm0[i] = pcm0[i] + Σ(j=0..15) lpc[j]*pcm0[i-j-1]
// First LPC_ORDER samples: copy value at index LPC_ORDER (crude initialization)
```

**Step 4 — Normalize and build continuation input (321 values):**
```c
norm2 = inner_product(wpcm0, wpcm0)
norm_1 = 1/sqrt(1e-8 + norm2)
cont_inputs[0]   = log(sqrt(norm2) + 1e-7)    // Log-energy
cont_inputs[1..] = norm_1 * wpcm0[0..319]     // Unit-normalized weighted PCM
```

**Step 5 — Run 6-layer continuation MLP:**
```
321 → Dense+Tanh → 160 → Dense+Tanh → 160 → Dense+Tanh → 80
  → Dense+Tanh → 80 → Dense+Tanh → 64 → Dense+Tanh → 64
```
Output stored in `st->cont` (64 floats).

**Step 6 — Initialize all layer states from continuation vector:**
Each layer has a dedicated dense projection from the 64-dim continuation vector:
```c
cont(64) → rnn_cont_fc_0   → rnn_state (256)
cont(64) → fwc1_cont_fc_0  → fwc1_state (512)
cont(64) → fwc2_cont_fc_0  → fwc2_state (512)
// ... through fwc7
```
All use `ACTIVATION_TANH`.

**Step 7 — Warmup frame:**
Run `fwgan_synthesize_impl` for one frame. The first subframe outputs zeros (sets up gain/LPC state only). Save subframes 1–3 (120 samples) to `pcm_buf` for the output delay buffer.

### 4.2 Frame synthesis (`fwgan_synthesize_impl`)

**Step 1 — Feature preparation:**
```c
// Copy 18 BFCC features, skip pitch period (features[18]),
// remap features[19] with +0.5 bias
fwgan_features[0..17]  = features[0..17]
fwgan_features[18]     = features[19] + 0.5

// Extract pitch period and angular frequency
period = floor(0.1 + 50*features[18] + 100)    // Typically 32–256 samples
w0 = 2π / period
```

**Step 2 — Feature upsampling:**
```c
// Dense layer: 19 features → 320 conditioning values (4 subframes × 80)
run_fwgan_upsampler(st, cond, fwgan_features)   // Uses ACTIVATION_TANH
```

**Step 3 — Process each of 4 subframes:**
For subframe `s` in `[0, 4)`, call `run_fwgan_subframe` with `sub_cond = &cond[s*80]`.

### 4.3 Subframe synthesis (`run_fwgan_subframe`)

This is the core neural network inference loop, producing 40 PCM samples per call.

**Step 1 — Pitch embeddings (80 values: 40 sin + 40 cos):**
```c
// Rotate phase oscillator by w0 per sample
for i in [0..40):
    [cos, sin] = rotate(phase, w0)   // Complex multiply by exp(-iw0)
    pembed[i]    = sin               // Imaginary part
    pembed[40+i] = cos               // Real part
// Renormalize phase to unit magnitude every subframe
```

**Step 2 — Assemble feature input (160 values):**
```c
feat_in[0..79]   = pembed[0..79]     // Pitch embeddings (sin + cos)
feat_in[80..159] = sub_cond[0..79]   // Upsampled BFCC conditioning
```

**Step 3 — Feature input convolution + gated activation:**
```c
conv1d(feat_in_conv1_conv, feat_in, cont_conv1_mem) → rnn_in (256)
gated_activation(feat_in_nl1_gate, rnn_in) → rnn_in
// Gated activation: tanh(x) ⊙ σ(W·x)
```

**Step 4 — Skip on first subframe (cont_initialized == 1):**
```c
if first_subframe:
    output zeros, update last_gain and last_lpc
    set cont_initialized = 2
    return
```

**Step 5 — GRU update:**
```c
GRU(rnn_gru_input, rnn_gru_recurrent, rnn_state, rnn_in)
gated_activation(rnn_nl_gate, rnn_state) → tmp2 (256)
```

**Step 6 — Framewise convolution cascade (7 layers):**
Each FWC layer is a 1D convolution (kernel_size=3) followed by gated activation:
```
tmp2(256) → fwc1_fc_0 → glu → tmp1(256)
tmp1(256) → fwc2_fc_0 → glu → tmp2(128)
tmp2(128) → fwc3_fc_0 → glu → tmp1(128)
tmp1(128) → fwc4_fc_0 → glu → tmp2(64)
tmp2(64)  → fwc5_fc_0 → glu → tmp1(64)
tmp1(64)  → fwc6_fc_0 → glu → tmp2(40)
tmp2(40)  → fwc7_fc_0 → glu → pcm(40)
```
The convolutions use the `fwcN_state` arrays as sliding-window memories. Each state holds 2 previous input frames (kernel_size - 1 = 2).

**Step 7 — Post-processing chain (in-place on 40 PCM samples):**
```
pcm → apply_gain → fwgan_preemphasis → fwgan_lpc_syn → fwgan_deemphasis → output
```

### 4.4 Post-processing details

**Gain application:**
```c
gain = 10^(0.5 * c0 / √18)          // c0 = features[0] (first BFCC = log-energy)
pcm[i] *= last_gain                  // Apply PREVIOUS subframe's gain (smooth transition)
last_gain = gain                     // Save new gain for next subframe
```

**Pre-emphasis** — high-pass filter `H(z) = 1 - 0.85·z⁻¹`:
```c
for each sample i:
    tmp = pcm[i]
    pcm[i] -= 0.85 * preemph_mem
    preemph_mem = tmp
```

**LPC synthesis** — all-pole filter `H(z) = 1 / A(z)` where `A(z) = 1 + Σ lpc[j]·z^(-j-1)`:
```c
for each sample i:
    pcm[i] -= Σ(j=0..15) syn_mem[j] * last_lpc[j]   // Uses PREVIOUS subframe's LPC
    shift syn_mem right by 1, insert pcm[i] at [0]
// After subframe: last_lpc = lpc (update for next subframe)
```

**De-emphasis** — inverse of pre-emphasis `H(z) = 1 / (1 - 0.85·z⁻¹)`:
```c
for each sample i:
    pcm[i] += 0.85 * deemph_mem
    deemph_mem = pcm[i]
```

### 4.5 Output buffering (`fwgan_synthesize`)

The public `fwgan_synthesize` introduces a 3-subframe (120-sample) output delay:

```c
// Synthesize 160 new samples into new_pcm[0..159]
fwgan_synthesize_impl(st, new_pcm, lpc, features);

// Output: 120 from previous frame's buffer + 40 from current frame's first subframe
pcm[0..119]   = st->pcm_buf[0..119]    // Subframes 1-3 from previous call
pcm[120..159] = new_pcm[0..39]         // Subframe 0 from this call

// Save remaining 120 for next call
st->pcm_buf[0..119] = new_pcm[40..159] // Subframes 1-3 from this call
```

## 5. Data Flow

### Frame-level flow

```
features[20]
    │
    ├─ features[0..17] + features[19]+0.5 ──→ upsampler(19→320) ──→ cond[320]
    │                                                                  │
    │                                                          split into 4 × 80
    │                                                                  │
    ├─ features[18] ──→ period ──→ w0 ──→ pitch_embeddings ──→ pembed[80/subframe]
    │                                                                  │
    │                                                         ┌────────┘
    │                                                         ▼
    │                                              ┌─────────────────────┐
    │                                              │  per-subframe loop  │
    │                                              │  (4 iterations)     │
    │                                              │                     │
    │   pembed(80) + sub_cond(80) ──→ feat_in(160) │                     │
    │                                    │         │                     │
    │                              conv1d+glu(256) │                     │
    │                                    │         │                     │
    │                               GRU(256)       │                     │
    │                                    │         │                     │
    │                            fwc1..fwc7 cascade│                     │
    │                                    │         │                     │
    │                               pcm(40)        │                     │
    │                                    │         │                     │
    │           gain → preemph → lpc_syn → deemph  │                     │
    │                                    │         │                     │
    └─ features[0] (c0 for gain) ────────┘         │                     │
                                                   └─────────────────────┘
                                                              │
                                                     new_pcm[160]
                                                              │
                                                    output buffering
                                                              │
                                                      pcm[160] output
```

### Buffer layouts

| Buffer | Size | Layout |
|--------|------|--------|
| `features` input | 20 | `[bfcc0..bfcc17, pitch_period_encoded, pitch_corr_or_aux]` |
| `cond` (upsampled) | 320 | `[sub0_cond(80) | sub1_cond(80) | sub2_cond(80) | sub3_cond(80)]` |
| `feat_in` | 160 | `[pembed_sin(40) | pembed_cos(40) | sub_cond(80)]` |
| `pembed` | 80 | `[sin_phase(40) | cos_phase(40)]` |
| `pcm_buf` | 160 | `[subframe1(40) | subframe2(40) | subframe3(40) | unused(40)]` — only 120 used |
| `cont_inputs` | 321 | `[log_energy(1) | normalized_weighted_pcm(320)]` |

### Feature remapping (from LPCNet 20-feature format)

```c
fwgan_features[0..17]  = features[0..17]          // 18 BFCCs (band energies)
fwgan_features[18]     = features[19] + 0.5        // Auxiliary feature, shifted
// features[18] (pitch period) used separately for phase embeddings
// features[19] remapped to index 18 in fwgan_features
```

## 6. Numerical Details

### Floating-point format

All computation is **32-bit float** (no fixed-point). The neural network layers use float weights and float arithmetic throughout.

### Pitch embedding — Taylor expansion for sin/cos

Instead of calling `sin()`/`cos()`, the code uses a 4th-order Taylor expansion:

```c
float w2 = w0*w0;
wreal = 1 - 0.5*w2*(1.0 - 0.083333333*w2);       // cos(w0) ≈ 1 - w²/2 + w⁴/24
wimag = w0*(1 - 0.166666667*w2*(1.0 - 0.05*w2));  // sin(w0) ≈ w - w³/6 + w⁵/120
```

This is valid because `w0 = 2π/period` where period ∈ [~32, ~256+], so `w0` ∈ [~0.025, ~0.196], making higher-order terms negligible. The Taylor series coefficients are:
- `0.083333333 ≈ 1/12` (for cos)
- `0.166666667 ≈ 1/6` (for sin)
- `0.05 = 1/20` (for sin)

### Phase oscillator renormalization

The complex phase `[cos θ, sin θ]` is renormalized to unit magnitude once per subframe to prevent drift:
```c
float r = 1.0 / sqrt(phase[0]² + phase[1]²);
phase[0] *= r;
phase[1] *= r;
```

### Gain formula

```c
gain = 10^(0.5 * c0 / √18)
```
where `c0 = features[0]` is the first BFCC coefficient (related to log-energy). The `√18` normalization factor relates to the BFCC band count (`NB_BANDS = 18`).

### Pitch period computation

```c
period = (int)floor(0.1 + 50*features[NB_BANDS] + 100)
```
The `0.1` bias ensures correct rounding. `features[18]` encodes pitch as `(period - 100) / 50`.

### LPC bandwidth expansion

```c
lpc[i] *= γ^(i+1)    where γ = FWGAN_GAMMA = 0.92
```
This moves LPC poles toward the origin by factor γ per order, smoothing the spectral envelope and preventing instability. The weighting is cumulative: `lpc[0] *= 0.92`, `lpc[1] *= 0.8464`, etc.

### Float-to-int16 conversion

```c
pcm_int = (int)floor(0.5 + MIN32(32767, MAX32(-32767, 32768.0 * fpcm[i])))
```
- Scale factor: 32768 (not 32767) — standard for audio
- Clamp range: [-32767, 32767] — **asymmetric**, avoids INT16_MIN
- Rounding: `floor(x + 0.5)` = round-half-up

### Continuation normalization

```c
norm2 = dot(wpcm0, wpcm0, 320)         // Squared L2 norm
norm_1 = 1/sqrt(1e-8 + norm2)           // ε = 1e-8 prevents division by zero
log_energy = log(sqrt(norm2) + 1e-7)    // ε = 1e-7 prevents log(0)
```

### Gated activation (GLU variant)

```c
output = tanh(x) ⊙ σ(W·x)
```
Element-wise product of `tanh` activation and learned sigmoid gate. This is a **modified GLU** (standard GLU uses identity instead of tanh).

## 7. Dependencies

### Modules called by FWGAN

| Module | Functions used | Purpose |
|--------|---------------|---------|
| `nnet.h` / `nnet.c` | `compute_generic_dense`, `compute_generic_conv1d`, `compute_generic_gru`, `compute_gated_activation`, `parse_weights`, `init_fwgan` | All neural network layer inference |
| `freq.h` / `freq.c` | `lpc_from_cepstrum` | Convert BFCC features to LPC coefficients |
| `pitch.h` | (included but no direct calls visible) | Pitch constants (PITCH_MIN/MAX_PERIOD) |
| `lpcnet.h` | `NB_FEATURES`, `LPCNET_FRAME_SIZE` constants | Feature format definitions |
| `lpcnet_private.h` | (included for shared constants) | Internal LPCNet definitions |
| `os_support.h` | `OPUS_COPY`, `OPUS_MOVE`, `OPUS_CLEAR` | Memory operations |
| `fwgan_data.h` (generated) | `FWGAN` struct, `fwgan_arrays`, all `*_OUT_SIZE` constants | Model weights and architecture |
| CELT | `celt_assert`, `celt_inner_prod`, `IMAX` | Assertions, dot product, max macro |

### Modules that call FWGAN

FWGAN is called from the OSCE module (`osce.c`) and the LPCNet PLC (packet loss concealment) system. It serves as a waveform generator when the decoder needs to synthesize audio from features rather than decode from the bitstream directly.

## 8. Constants and Tables

### Compile-time constants (from `fwgan.h`)

| Constant | Value | Derivation |
|----------|-------|------------|
| `FWGAN_CONT_SAMPLES` | 320 | Continuation PCM window size (20 ms at 16 kHz) |
| `NB_SUBFRAMES` | 4 | Subframes per frame |
| `SUBFRAME_SIZE` | 40 | Samples per subframe (2.5 ms) |
| `FWGAN_FRAME_SIZE` | 160 | = 4×40, total frame size (10 ms) |
| `CONT_PCM_INPUTS` | 320 | Same as FWGAN_CONT_SAMPLES |
| `FWGAN_GAMMA` | 0.92 | LPC bandwidth expansion factor |
| `FWGAN_DEEMPHASIS` | 0.85 | Pre/de-emphasis coefficient (= PREEMPHASIS from freq.h) |

### Hardcoded state sizes (from `fwgan.h`)

These are **upper bounds** for the FWGAN400 model. The code has a `FIXME` about deriving them from the model.

| Constant | Value | Derivation: `(kernel-1) × input_frame_len` |
|----------|-------|-----|
| `FWC1_STATE_SIZE` | 512 | 2 × 256 (fwc1 input dim) |
| `FWC2_STATE_SIZE` | 512 | 2 × 256 (fwc2 input dim) |
| `FWC3_STATE_SIZE` | 256 | 2 × 128 |
| `FWC4_STATE_SIZE` | 256 | 2 × 128 |
| `FWC5_STATE_SIZE` | 128 | 2 × 64 |
| `FWC6_STATE_SIZE` | 128 | 2 × 64 |
| `FWC7_STATE_SIZE` | 80 | 2 × 40 |

### Generated constants (from `fwgan_data.h`, FWGAN400 model)

| Constant | Value | Source |
|----------|-------|--------|
| `BFCC_WITH_CORR_UPSAMPLER_FC_OUT_SIZE` | 320 | `UpsampleFC(19, 80, 4)`: 80×4 |
| `CONT_NET_0_OUT_SIZE` = `MAX_CONT_SIZE` | 160 | First continuation layer output |
| `CONT_NET_10_OUT_SIZE` | 64 | Final continuation layer output |
| `FEAT_IN_CONV1_CONV_IN_SIZE` | 160 | = 80 pitch + 80 conditioning |
| `FEAT_IN_CONV1_CONV_OUT_SIZE` | 256 | `ConvLookahead(160, 256)` |
| `FEAT_IN_CONV1_CONV_STATE_SIZE` | ≈640 | `(kernel-1) × input_size = 4×160` |
| `FEAT_IN_NL1_GATE_OUT_SIZE` | 256 | `GLU(256)` |
| `RNN_GRU_STATE_SIZE` | 256 | GRU hidden_size |
| `FWC1_FC_0_OUT_SIZE` | 256 | fwc1 out_dim |
| `FWC2_FC_0_OUT_SIZE` | 128 | fwc2 out_dim |
| `FWC3_FC_0_OUT_SIZE` | 128 | fwc3 out_dim |
| `FWC4_FC_0_OUT_SIZE` | 64 | fwc4 out_dim |
| `FWC5_FC_0_OUT_SIZE` | 64 | fwc5 out_dim |
| `FWC6_FC_0_OUT_SIZE` | 40 | fwc6 out_dim |
| `FWC7_FC_0_OUT_SIZE` | 40 | fwc7 out_dim |

### Derived constants (computed in `fwgan.c`)

```c
#define FEAT_IN_SIZE  (BFCC_WITH_CORR_UPSAMPLER_FC_OUT_SIZE/4 + FWGAN_FRAME_SIZE/2)
                    // = 320/4 + 160/2 = 80 + 80 = 160

#define FWGAN_FEATURES  (NB_FEATURES-1)  // = 19 (skip pitch period feature)
```

### External constants used

| Constant | Value | Source |
|----------|-------|--------|
| `NB_FEATURES` | 20 | `lpcnet.h` |
| `NB_BANDS` | 18 | `freq.h` |
| `LPC_ORDER` | 16 | `freq.h` |
| `LPCNET_FRAME_SIZE` | 160 | `lpcnet.h` |

### FWGAN400 model architecture (from PyTorch)

```
Continuation network:
  321 →[Dense+Tanh]→ 160 →[Dense+Tanh]→ 160 →[Dense+Tanh]→ 80
    →[Dense+Tanh]→ 80 →[Dense+Tanh]→ 64 →[Dense+Tanh]→ 64

Feature upsampler:
  19 →[Dense+Tanh]→ 320   (= 80 × 4 subframes)

Feature input:
  160 →[Conv1d(k=5)+GLU]→ 256

Recurrent:
  256 →[GRU]→ 256 →[GLU]→ 256

FWC cascade (all kernel_size=3, all GLU activation):
  256 →[fwc1]→ 256 →[fwc2]→ 128 →[fwc3]→ 128 →[fwc4]→ 64
    →[fwc5]→ 64 →[fwc6]→ 40 →[fwc7]→ 40
```

### FWGAN500 variant differences (for reference)

The FWGAN500Cont model differs from FWGAN400:
- Upsampler: `ConvTranspose1d(19, 64, kernel_size=5, stride=5)` instead of FC
- Final layers: fwc6→32, fwc7→32 (not 40)
- Phase signals use 32-sample chunks (not 40)
- No separate `cont_net`; continuation goes directly from raw PCM (320 values) into layer states
- The C code's constants (SUBFRAME_SIZE=40, etc.) are hardcoded for FWGAN400

## 9. Edge Cases

### Missing `arch` parameter

**Critical bug in reference code**: All calls to `compute_generic_dense`, `compute_generic_conv1d`, `compute_generic_gru`, and `compute_gated_activation` in `fwgan.c` are missing the final `int arch` parameter required by their declarations in `nnet.h`. Other modules (e.g., `dred_rdovae_dec.c`) correctly pass `arch`. This means `fwgan.c` either:
- Doesn't compile against the current `nnet.h`, or
- Relies on an implicit value (undefined behavior in C)

The `FWGANState.arch` field exists but is never set (the `fwgan_init` code has `FIXME: perform arch detection`). For the Rust port, always pass `arch` explicitly.

### Continuation first-sample hack

```c
for (i=0;i<LPC_ORDER;i++) wpcm0[i] = wpcm0[LPC_ORDER];
```
The first `LPC_ORDER` (16) samples of the weighted PCM can't be properly computed because there's no PCM history before `pcm0[0]`. The code fills them with the value at index 16. This is acknowledged with a `/* FIXME: Make this less stupid. */` comment.

### Zero-energy continuation

When `pcm0` is all zeros or near-zero:
- `norm2 ≈ 0`, `norm_1 = 1/sqrt(1e-8) = 10000` — the epsilon prevents infinity
- `cont_inputs[0] = log(1e-7) ≈ -16.1` — the epsilon prevents `-∞`
- Normalized PCM values will be near zero (multiplied by large norm_1 but inputs are near zero)

### First subframe zero output

When `cont_initialized == 1`, the first subframe outputs zeros (after gain application, which multiplies zeros by `last_gain`). This is intentional — it lets the feat_in conv1d prime its state before the full pipeline runs.

### Phase drift

The phase oscillator accumulates error from the Taylor approximation and finite-precision arithmetic. Renormalization every 40 samples (once per subframe) prevents magnitude drift but not phase angle drift. For typical pitch periods (32–256 samples), the error per subframe is negligible.

### LPC coefficient update timing

Both gain and LPC use a **one-subframe delay**: the current subframe uses the *previous* subframe's gain/LPC, and saves the new values for next time. This creates smooth transitions at subframe boundaries. The first subframe after continuation uses the gain/LPC set during `fwgan_cont`.

## 10. Porting Notes

### 10.1 Generated code dependency

`fwgan_data.h` and `fwgan_data.c` are **generated** by `dnn/torch/fwgan/dump_model_weights.py` using the `wexchange` library. They define:
- The `FWGAN` struct with all `LinearLayer` fields
- All `*_OUT_SIZE` and `*_STATE_SIZE` constants
- The `init_fwgan()` function
- The `fwgan_arrays` weight data

For the Rust port, you need to either:
1. Generate equivalent Rust code from the Python export script, or
2. Define the `FWGAN` struct and constant derivation manually, matching the architecture
3. Load weights at runtime via `fwgan_load_model` (parse the binary blob format)

### 10.2 Missing `arch` parameter

Every nnet function call in `fwgan.c` omits the `arch` parameter. The Rust port must include it. Since `FWGANState.arch` is never initialized, pass `0` (generic/scalar implementation) unless CPU feature detection is added.

### 10.3 Alternating temp buffers

The subframe function uses two temporary buffers that alternate through the FWC cascade:

```c
float tmp1[FWC1_FC_0_OUT_SIZE];                        // 256
float tmp2[IMAX(RNN_GRU_STATE_SIZE, FWC2_FC_0_OUT_SIZE)]; // max(256, 128) = 256
```

`tmp1` holds outputs of fwc1, fwc3, fwc5, fwc7. `tmp2` holds GRU output, fwc2, fwc4, fwc6. The sizes are the maximum across all values each buffer holds:
- `tmp1` max: FWC1_FC_0_OUT_SIZE = 256
- `tmp2` max: max(RNN_GRU_STATE_SIZE, FWC2_FC_0_OUT_SIZE) = max(256, 128) = 256

In Rust, use `[f32; 256]` for both, or use the constant expressions.

### 10.4 In-place mutations

Several operations mutate buffers in-place:
- `compute_gated_activation` reads and writes the same buffer: `compute_gated_activation(gate, tmp1, tmp1, ...)`
- `fwgan_preemphasis`, `fwgan_deemphasis`, `apply_gain`, `fwgan_lpc_syn` all modify `pcm` in-place
- `OPUS_MOVE(&mem[1], &mem[0], LPC_ORDER-1)` overlapping memmove in LPC synthesis

In Rust, in-place mutation is fine with `&mut [f32]`. The `OPUS_MOVE` corresponds to `slice.copy_within(0..LPC_ORDER-1, 1)` (source and destination overlap, copy direction matters).

### 10.5 `OPUS_COPY` / `OPUS_MOVE` / `OPUS_CLEAR`

| C macro | Rust equivalent |
|---------|-----------------|
| `OPUS_COPY(dst, src, n)` | `dst[..n].copy_from_slice(&src[..n])` |
| `OPUS_MOVE(dst, src, n)` | `buf.copy_within(src_range, dst_start)` (overlapping) |
| `OPUS_CLEAR(p, n)` | `p[..n].fill(0.0)` or for structs: `*st = Default::default()` |

### 10.6 `celt_inner_prod` with arch dispatch

```c
norm2 = celt_inner_prod(wpcm0, wpcm0, CONT_PCM_INPUTS, st->arch);
```
This is a dot product with optional SIMD. In Rust, use a simple loop; add SIMD later if needed.

### 10.7 Double precision for `w0`

The pitch angular frequency `w0` is computed and passed as `double`, not `float`:

```c
double w0;
w0 = 2*M_PI/period;
```

The `pitch_embeddings` function receives `double w0` but casts to float for the Taylor expansion:
```c
float w2 = w0*w0;   // Implicit double→float conversion
```

The Rust port should preserve this mixed precision to ensure bit-exact output.

### 10.8 `floor` and `pow` precision

```c
period = (int)floor(.1 + 50*features[NB_BANDS]+100);   // float arithmetic, then floor
gain = pow(10.f, (0.5f*c0/sqrt(18.f)));                // powf
```

Use `f32::floor()` and `f32::powf()` in Rust. Note `sqrt(18.f)` is `f32::sqrt(18.0)`.

### 10.9 Static function visibility

All functions except the 5 public API functions are `static`. In Rust, make these module-private (`fn`, not `pub fn`).

### 10.10 Model struct layout

The `FWGAN` struct contains ~30 `LinearLayer` fields. Each `LinearLayer` has:
```c
typedef struct {
  const float *bias;
  const float *subias;
  const opus_int8 *weights;        // Quantized int8 weights
  const float *float_weights;      // Full-precision float weights
  const int *weights_idx;
  const float *diag;
  const float *scale;
  int nb_inputs;
  int nb_outputs;
} LinearLayer;
```

The pointers reference into the weight data blob. In Rust, consider:
- Using indices into a `Vec<f32>` / `Vec<i8>` weight buffer instead of raw pointers
- Or using `&[f32]` / `&[i8]` slices with appropriate lifetimes tied to the weight data

### 10.11 Quantized vs unquantized layers

From `dump_model_weights.py`, these layers are **not quantized** (use `float_weights`):
- `bfcc_with_corr_upsampler.fc`
- `cont_net.0`
- `fwc6.cont_fc.0`, `fwc6.fc.0`, `fwc6.fc.1.gate`
- `fwc7.cont_fc.0`, `fwc7.fc.0`, `fwc7.fc.1.gate`

All other layers use **int8 quantized weights** with per-output scaling. The `compute_generic_dense` function dispatches based on whether `weights` (quantized) or `float_weights` is non-NULL.

### 10.12 Conditional compilation

The only conditional compilation is `#ifdef HAVE_CONFIG_H` for the config include. The `#if 1` / `#else` block in `pitch_embeddings` selects Taylor vs. `cos`/`sin` — the Taylor path is always active. No SIMD intrinsic guards exist in fwgan.c.

### 10.13 Two model variants

The C code is architecturally generic (uses LinearLayer abstraction), but the constants in `fwgan.h` are hardcoded for FWGAN400. Supporting FWGAN500 would require:
- Different SUBFRAME_SIZE (32 vs 40)
- Different state sizes
- Different upsampler type (ConvTranspose1d vs FC)

Consider making these configurable in the Rust port, or explicitly supporting only FWGAN400 initially.

### 10.14 `M_PI` usage

```c
w0 = 2*M_PI/period;
```

`M_PI` is not in the C standard but is available on most platforms. In Rust, use `std::f64::consts::PI`.
