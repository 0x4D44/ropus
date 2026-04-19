Now I have a complete picture. Let me generate the architecture document.

# FARGAN Module Architecture Documentation

## 1. Purpose

FARGAN (**F**requency-domain **A**uto**R**egressive **G**enerative **A**dversarial **N**etwork) is a neural waveform generator within the Opus codec's DNN subsystem (Opus 1.4+). It synthesizes time-domain PCM audio from spectral feature vectors, one 10 ms frame (160 samples at 16 kHz) at a time.

FARGAN is the successor/alternative to FWGAN in the Opus pipeline. Its primary consumer is the **LPCNet PLC (Packet Loss Concealment)** module (`lpcnet_plc.c`), where it generates replacement audio when packets are lost. It can also be used standalone for neural speech synthesis from pre-computed features (see `fargan_demo.c`).

The architecture is a **conditional autoregressive model**: a conditioning network processes per-frame spectral features into a dense conditioning vector, and a signal network generates audio sample-by-sample (in subframes of 40 samples), conditioned on the features and its own previous output via a pitch-based feedback loop.

## 2. Public API

### `void fargan_init(FARGANState *st)`

Zeros the entire state structure, selects the CPU architecture for optimized kernels (`opus_select_arch()`), and initializes the neural network weights from the compiled-in `fargan_arrays[]` weight table (unless `USE_WEIGHTS_FILE` is defined, in which case weights must be loaded separately).

- **Parameters**: `st` — pointer to caller-allocated `FARGANState`
- **Returns**: void (asserts on failure)
- **Postcondition**: `st->cont_initialized == 0` — the state is not yet ready for synthesis; `fargan_cont()` must be called first.

### `int fargan_load_model(FARGANState *st, const void *data, int len)`

Loads model weights from a binary blob (the "weights file" format). Parses the blob into a `WeightArray` list, then calls `init_fargan()` to populate `st->model`.

- **Parameters**:
  - `st` — initialized state
  - `data` — pointer to weight blob
  - `len` — blob size in bytes
- **Returns**: `0` on success, `-1` on failure
- **Note**: Used only when `USE_WEIGHTS_FILE` is defined. The parsed `WeightArray` list is freed after initialization.

### `void fargan_cont(FARGANState *st, const float *pcm0, const float *features0)`

**Continuation/warm-up** function. Primes the internal state (pitch buffer, GRU states, convolution memories, de-emphasis memory) from known PCM history and feature vectors. Must be called once before `fargan_synthesize()` to establish temporal continuity.

- **Parameters**:
  - `st` — initialized state (from `fargan_init`)
  - `pcm0` — `float[FARGAN_CONT_SAMPLES]` (320 samples = 20 ms) of previous PCM audio, in the range approximately [-1, 1] (not pre-emphasized)
  - `features0` — `float[5 * NB_FEATURES]` (5 frames × 20 features) of conditioning features. The first 3 frames are "look-back" for the conv1d, and the last 2 frames correspond to the 2 frames covering the 320 PCM samples.
- **Returns**: void
- **Postcondition**: `st->cont_initialized == 1`

### `void fargan_synthesize(FARGANState *st, float *pcm, const float *features)`

Synthesizes one frame (160 samples) of float PCM audio from a feature vector.

- **Parameters**:
  - `st` — state with `cont_initialized == 1`
  - `pcm` — output buffer `float[FARGAN_FRAME_SIZE]` (160 samples). Output is de-emphasized PCM in approximately [-1, 1] range.
  - `features` — `float[NB_FEATURES]` (20 features for one 10 ms frame)
- **Returns**: void

### `void fargan_synthesize_int(FARGANState *st, opus_int16 *pcm, const float *features)`

Same as `fargan_synthesize` but converts output to 16-bit signed integer PCM.

- **Parameters**:
  - `st` — state with `cont_initialized == 1`
  - `pcm` — output buffer `opus_int16[LPCNET_FRAME_SIZE]` (160 samples)
  - `features` — `float[NB_FEATURES]` (20 features)
- **Returns**: void
- **Conversion**: `floor(0.5 + clamp(fpcm[i] * 32768, -32767, 32767))`

## 3. Internal State

### `FARGANState` (main state struct)

```c
typedef struct {
  FARGAN model;                                    // Neural network weights (all LinearLayer structs)
  int arch;                                        // CPU architecture ID (for SIMD dispatch)
  int cont_initialized;                            // 0 until fargan_cont() called
  float deemph_mem;                                // De-emphasis filter state (1 sample)
  float pitch_buf[PITCH_MAX_PERIOD];               // Circular pitch buffer [256 floats]
  float cond_conv1_state[COND_NET_FCONV1_STATE_SIZE]; // Conv1d state for conditioning network
  float fwc0_mem[SIG_NET_FWC0_STATE_SIZE];         // FWConv state memory
  float gru1_state[SIG_NET_GRU1_STATE_SIZE];       // GRU layer 1 hidden state [160 floats]
  float gru2_state[SIG_NET_GRU2_STATE_SIZE];       // GRU layer 2 hidden state [128 floats]
  float gru3_state[SIG_NET_GRU3_STATE_SIZE];       // GRU layer 3 hidden state [128 floats]
  int last_period;                                  // Pitch period from previous frame
} FARGANState;
```

**Lifecycle**:
1. Allocate (caller-managed, stack or heap)
2. `fargan_init()` — zeros everything, loads weights
3. `fargan_load_model()` — (optional, only with `USE_WEIGHTS_FILE`)
4. `fargan_cont()` — primes state from known history; sets `cont_initialized = 1`
5. `fargan_synthesize()` / `fargan_synthesize_int()` — call repeatedly for each frame
6. No explicit destroy/free needed (no dynamic allocation within the struct)

### `FARGAN` (model weights struct, auto-generated in `fargan_data.h`)

Contains `LinearLayer` members for every neural network layer. Based on the Python model definition and the weight dump script, the struct contains:

```c
typedef struct {
    LinearLayer cond_net_pembed;           // Pitch embedding lookup [224 × 12]
    LinearLayer cond_net_fdense1;          // Dense: (20+12) → 64, no bias
    LinearLayer cond_net_fconv1;           // Conv1d: 64 → 128, kernel_size=3, no bias
    LinearLayer cond_net_fdense2;          // Dense: 128 → 320 (80×4), no bias

    LinearLayer sig_net_cond_gain_dense;   // Dense: 80 → 1 (gain prediction, with bias)
    LinearLayer sig_net_fwc0_conv;         // FWConv linear: input_size×2 → 192, no bias
    LinearLayer sig_net_fwc0_glu_gate;     // GLU gate: 192 → 192, no bias
    LinearLayer sig_net_gain_dense_out;    // Dense: 192 → 4 (pitch gates, with bias)

    LinearLayer sig_net_gru1_input;        // GRU1 input weights: (192+2×40) → 3×160
    LinearLayer sig_net_gru1_recurrent;    // GRU1 recurrent weights: 160 → 3×160
    LinearLayer sig_net_gru1_glu_gate;     // GLU gate: 160 → 160

    LinearLayer sig_net_gru2_input;        // GRU2 input weights: (160+2×40) → 3×128
    LinearLayer sig_net_gru2_recurrent;    // GRU2 recurrent weights: 128 → 3×128
    LinearLayer sig_net_gru2_glu_gate;     // GLU gate: 128 → 128

    LinearLayer sig_net_gru3_input;        // GRU3 input weights: (128+2×40) → 3×128
    LinearLayer sig_net_gru3_recurrent;    // GRU3 recurrent weights: 128 → 3×128
    LinearLayer sig_net_gru3_glu_gate;     // GLU gate: 128 → 128

    LinearLayer sig_net_skip_dense;        // Dense: (160+128+128+192+40+40) → 128
    LinearLayer sig_net_skip_glu_gate;     // GLU gate: 128 → 128
    LinearLayer sig_net_sig_dense_out;     // Dense: 128 → 40 (subframe output)
} FARGAN;
```

### `LinearLayer` (from `nnet.h`)

```c
typedef struct {
  const float *bias;          // Bias vector (NULL if no bias)
  const float *subias;        // Sub-bias (quantization support)
  const opus_int8 *weights;   // Quantized int8 weights
  const float *float_weights; // Float weights (used for embeddings, unquantized layers)
  const int *weights_idx;     // Sparse weight indices
  const float *diag;          // Diagonal weights
  const float *scale;         // Quantization scale factors
  int nb_inputs;              // Input dimension
  int nb_outputs;             // Output dimension
} LinearLayer;
```

## 4. Algorithm

### 4.1 High-level Flow

Each call to `fargan_synthesize()` processes one 10 ms frame in two stages:

1. **Conditioning** (`compute_fargan_cond`): Features → dense conditioning vector (320 floats, split into 4 × 80 for subframes)
2. **Signal generation** (`run_fargan_subframe` × 4): For each of 4 subframes (40 samples each), generate PCM autoregressively using GRU-based signal network

### 4.2 Pitch Period Extraction

```c
period = (int)floor(.5 + 256.0 / pow(2.0, (1.0/60.0) * ((features[NB_BANDS] + 1.5) * 60)));
```

The pitch period is decoded from feature index `NB_BANDS` (index 18, the first feature after the 18 bark-scale band energies). The formula converts from a log-frequency feature to a sample-domain period in the range [32, 256] (`PITCH_MIN_PERIOD` to `PITCH_MAX_PERIOD`).

**Critical detail**: The pitch period used for signal generation is `st->last_period` (the *previous* frame's period), not the current frame's period. The current period is stored for the next frame. This introduces a one-frame lag.

### 4.3 Conditioning Network (`compute_fargan_cond`)

```
Input: features[NB_FEATURES=20] + period (integer)

1. Pitch embedding lookup:
   - Clamp period to [32, 255]: index = IMAX(0, IMIN(period-32, 223))
   - Look up float_weights[index * PEMBED_OUT_SIZE .. (index+1) * PEMBED_OUT_SIZE]
   - Result: pembed[12]

2. Concatenate: dense_in = [features[20], pembed[12]]  → 32 floats

3. fdense1: Linear(32 → 64, no bias) + tanh → conv1_in[64]

4. fconv1: Conv1d(64 → 128, kernel_size=3, causal) + tanh → fdense2_in[128]
   - Uses st->cond_conv1_state to store the previous 2 input frames (128 floats)
   - Implemented as generic_conv1d: concatenates [state, current_input], applies linear, updates state

5. fdense2: Linear(128 → 320, no bias) + tanh → cond[320]
```

The 320-float output is split into 4 chunks of 80 (`FARGAN_COND_SIZE`) for the 4 subframes.

### 4.4 Signal Network (`run_fargan_subframe`)

Called 4 times per frame. Each invocation produces 40 PCM samples.

```
Inputs:
  - cond[80]: conditioning vector for this subframe
  - period: pitch period (from previous frame)
  - st->pitch_buf[256]: history of generated (pre-emphasis) samples

Step 1: Gain computation
  - gain = exp(Linear(cond[80] → 1, LINEAR activation))
  - gain_1 = 1.0 / (1e-5 + gain)

Step 2: Pitch prediction (pred) and previous samples (prev)
  - pos = PITCH_MAX_PERIOD - period - 2 = 256 - period - 2
  - For i in 0..43 (FARGAN_SUBFRAME_SIZE + 4):
      pred[i] = clamp(gain_1 * pitch_buf[pos], -1, 1)
      pos++; if pos == 256: pos -= period    // wrap-around
  - prev[i] = clamp(gain_1 * pitch_buf[256 - 40 + i], -1, 1)  for i in 0..39
  (pred and prev are gain-normalized)

Step 3: FWConv (frame-wise convolution)
  - fwc0_in = [cond[80], pred[44], prev[40]]  → total SIG_NET_INPUT_SIZE
  - fwc0_out = GLU(tanh(Conv1d(fwc0_in, kernel_size=2)))
    Uses st->fwc0_mem as state (previous frame's input)

Step 4: Pitch gate
  - pitch_gate[4] = sigmoid(Linear(fwc0_out[192] → 4))
  (4 gates, one per GRU layer + one for skip connection)

Step 5: GRU cascade (3 layers)
  For each GRU layer k ∈ {1, 2, 3}:
    - gru_in = [prev_layer_glu_out, pitch_gate[k-1] * pred[2:42], prev[40]]
    - GRU update: state_k = GRU(gru_in, state_k)
    - glu_out_k = GLU(state_k)
  
  Layer dimensions:
    - GRU1: input=(192+40+40)=272, hidden=160, GLU out=160
    - GRU2: input=(160+40+40)=240, hidden=128, GLU out=128
    - GRU3: input=(128+40+40)=208, hidden=128, GLU out=128

Step 6: Skip connections
  - skip_cat = [gru1_glu_out[160], gru2_glu_out[128], gru3_glu_out[128],
                fwc0_out[192], pitch_gate[3]*pred[2:42], prev[40]]
    Total: 160 + 128 + 128 + 192 + 40 + 40 = 688 floats
  - skip_out = GLU(tanh(Linear(skip_cat → 128)))

Step 7: Output
  - pcm = tanh(Linear(skip_out[128] → 40)) * gain

Step 8: State update
  - Shift pitch_buf left by 40, append new pcm[40] at the end
  - Apply de-emphasis: pcm[i] += 0.85 * pcm[i-1]  (first-order IIR)
```

### 4.5 GRU Implementation Detail

The GRU in `compute_generic_gru` follows standard GRU equations:

```
z, r = sigmoid(W_input * in + W_recurrent[:2N] * state)
h = tanh(W_input_h * in + r ⊙ (W_recurrent_h * state))
state = z ⊙ state + (1 - z) ⊙ h
```

- Input and recurrent contributions are computed separately then summed for z and r gates
- The h (candidate) gate uses the reset gate `r` to modulate the recurrent contribution
- `recurrent_weights.nb_inputs = N`, `recurrent_weights.nb_outputs = 3*N`

### 4.6 GLU (Gated Linear Unit)

```
gate = sigmoid(Linear(input))
output = input ⊙ gate
```

The GLU layer's `LinearLayer` has `nb_inputs == nb_outputs`. It supports in-place operation (`input == output`).

### 4.7 De-emphasis Filter

A simple first-order IIR filter applied to each subframe's output:

```
y[i] = x[i] + 0.85 * y[i-1]
```

This inverts the pre-emphasis (`x_preemph[i] = x[i] - 0.85 * x[i-1]`) that was applied during feature extraction.

### 4.8 Continuation (`fargan_cont`)

```
1. Pre-load conditioning: iterate over 5 feature frames, calling compute_fargan_cond
   for each. This fills the conv1d state with valid history.

2. Apply pre-emphasis to PCM input:
   x0[0] = 0
   x0[i] = pcm0[i] - 0.85 * pcm0[i-1]   for i in 1..319

3. Copy last FARGAN_FRAME_SIZE (160) pre-emphasized samples into pitch_buf tail

4. Set cont_initialized = 1

5. Run 4 subframes (one full frame) using dummy output buffers to warm up
   GRU states and FWConv memory. After each subframe, override pitch_buf
   tail with the actual known pre-emphasized samples (from x0[160..319]).

6. Set deemph_mem = pcm0[FARGAN_CONT_SAMPLES-1] (last original PCM sample)
```

This ensures smooth continuation: the signal network's recurrent states are warmed up with real data, while the pitch buffer contains the actual known history.

## 5. Data Flow

### Input Features

`features[NB_FEATURES]` where `NB_FEATURES = 20`:
- `features[0..17]` — 18 bark-scale band energies (NB_BANDS = 18)
- `features[18]` — pitch correlation / log-frequency pitch feature (used for period extraction)
- `features[19]` — additional feature

### Buffer Layouts

**Pitch buffer** (`pitch_buf[PITCH_MAX_PERIOD]`, 256 floats):
- Contains **pre-emphasized** (gain-normalized) generated samples
- Most recent samples at the **end** (index 255)
- Shifted left by `FARGAN_SUBFRAME_SIZE` (40) after each subframe
- Pitch prediction reads backwards from `pos = 256 - period - 2` with wrap-around

**Conditioning vector** (`cond[COND_NET_FDENSE2_OUT_SIZE]`, 320 floats):
- Organized as 4 chunks of `FARGAN_COND_SIZE` (80 floats each)
- `cond[0..79]` → subframe 0, `cond[80..159]` → subframe 1, etc.

**pred buffer** (44 floats = `FARGAN_SUBFRAME_SIZE + 4`):
- Pitch-predicted samples, gain-normalized and clamped to [-1, 1]
- Extra 4 samples (±2 around center) originally intended for tap filtering
- Only `pred[2..41]` (the center 40 samples) are used in the signal path

**FWConv input** (`fwc0_in[SIG_NET_INPUT_SIZE]`):
- Layout: `[cond[80], pred[44], prev[40]]`
- Total: `80 + 44 + 40 = 164` floats (= `FARGAN_COND_SIZE + 2*FARGAN_SUBFRAME_SIZE + 4`)

**Skip concatenation** (`skip_cat[]`, up to 688 floats):
- Layout: `[gru1_glu[160], gru2_glu[128], gru3_glu[128], fwc0_glu[192], gated_pred[40], prev[40]]`
- Note: declared as `float skip_cat[10000]` — massively over-allocated on the stack

**GRU input concatenation patterns**:
- GRU1: `[fwc0_glu_out[192], pitch_gate[0]*pred[40], prev[40]]` = 272
- GRU2: `[gru1_glu_out[160], pitch_gate[1]*pred[40], prev[40]]` = 240
- GRU3: `[gru2_glu_out[128], pitch_gate[2]*pred[40], prev[40]]` = 208

### PCM Output

- `fargan_synthesize`: `float[160]`, approximately in [-1, 1] after de-emphasis
- `fargan_synthesize_int`: `opus_int16[160]`, converted via `floor(0.5 + clamp(x * 32768, -32767, 32767))`

## 6. Numerical Details

### Floating-Point Arithmetic

FARGAN operates entirely in **single-precision float** (f32). There is no fixed-point (Q-format) arithmetic in this module.

### Gain Normalization

The gain mechanism is central to numerical stability:
- `gain = exp(linear_output)` — always positive, can be very large or very small
- `gain_1 = 1.0 / (1e-5 + gain)` — inverse gain with epsilon guard against division by zero
- Pitch predictions and previous samples are normalized by `gain_1` and clamped to [-1, 1]
- Final output is multiplied by `gain`, restoring the original scale

This normalization ensures the neural network operates on bounded [-1, 1] signals regardless of the actual signal amplitude.

### Clamping

- `MIN32` / `MAX32` macros clamp pitch prediction and previous samples to [-1, 1]
- Float-to-int16 conversion clamps to [-32767, 32767] (note: not -32768, asymmetric)

### Activation Functions

- `tanh` — used in conditioning network dense layers and signal output
- `sigmoid` — used in GLU gates and pitch gates
- `LINEAR` — used only for gain prediction (raw output)

### Rounding

- Period computation: `floor(0.5 + ...)` = round-half-up
- Float-to-int16: `floor(0.5 + ...)` = round-half-up

### Precision Requirements

For bit-exact reproduction:
- The `exp()` and `pow()` calls in period computation and gain calculation must match the C library's behavior exactly
- `floor(0.5 + x)` rounding must match (watch for ties at 0.5 boundaries)
- All `tanh`/`sigmoid` activations go through `compute_activation_c` which uses standard C math

## 7. Dependencies

### Modules Called by FARGAN

| Module | Functions Used | Purpose |
|--------|---------------|---------|
| `nnet.h/nnet.c` | `compute_generic_dense`, `compute_generic_conv1d`, `compute_generic_gru`, `compute_glu`, `compute_linear`, `compute_activation`, `parse_weights`, `linear_init` | All neural network layer computations |
| `fargan_data.h/c` | `init_fargan`, `fargan_arrays[]` | Auto-generated model weight initialization |
| `freq.h` | `NB_BANDS` constant (18) | Feature dimension reference |
| `pitchdnn.h` | `PITCH_MAX_PERIOD` (256) | Pitch buffer size constant |
| `lpcnet.h` | `NB_FEATURES` (20), `LPCNET_FRAME_SIZE` (160) | Feature/frame size constants |
| `os_support.h` | `OPUS_COPY`, `OPUS_MOVE`, `OPUS_CLEAR` | Memory operations (memcpy/memmove/memset wrappers) |
| `cpu_support.h` | `opus_select_arch()` | SIMD architecture detection |
| `pitch.h` | `PITCH_MAX_PERIOD` | (indirect, via pitchdnn.h) |
| `arch.h` | `IMAX`, `IMIN`, `MIN32`, `MAX32`, `MIN16` | Clamping/min/max macros |

### Modules That Call FARGAN

| Module | Usage |
|--------|-------|
| `lpcnet_plc.c` | Primary consumer — PLC concealment uses FARGAN for waveform generation |
| `fargan_demo.c` | Standalone demo/test application |

## 8. Constants and Tables

### Compile-Time Constants

| Constant | Value | Derivation |
|----------|-------|------------|
| `FARGAN_CONT_SAMPLES` | 320 | 20 ms at 16 kHz (2 frames for continuation) |
| `FARGAN_NB_SUBFRAMES` | 4 | Frame split into 4 subframes |
| `FARGAN_SUBFRAME_SIZE` | 40 | 2.5 ms at 16 kHz |
| `FARGAN_FRAME_SIZE` | 160 | 4 × 40 = 10 ms at 16 kHz |
| `FARGAN_DEEMPHASIS` | 0.85f | De-emphasis coefficient (matches `PREEMPHASIS` in freq.h) |
| `NB_FEATURES` | 20 | Feature vector dimension |
| `NB_BANDS` | 18 | Number of bark-scale frequency bands |
| `PITCH_MAX_PERIOD` | 256 | Maximum pitch period in samples (~62.5 Hz at 16 kHz) |
| `PITCH_MIN_PERIOD` | 32 | Minimum pitch period (implicit, via embedding size 224 = 256-32) |
| `LPCNET_FRAME_SIZE` | 160 | Frame size (same as `FARGAN_FRAME_SIZE`) |

### Derived Constants (from auto-generated `fargan_data.h`)

Based on the Python model definition:

| Constant | Value | Derivation |
|----------|-------|------------|
| `COND_NET_PEMBED_OUT_SIZE` | 12 | Embedding dimension |
| `COND_NET_FCONV1_IN_SIZE` | 64 | = fdense1 output size |
| `COND_NET_FCONV1_OUT_SIZE` | 128 | Conv1d output channels |
| `COND_NET_FCONV1_STATE_SIZE` | 128 | = 64 × (kernel_size-1) = 64 × 2 |
| `COND_NET_FDENSE2_OUT_SIZE` | 320 | = 80 × 4 subframes |
| `FARGAN_COND_SIZE` | 80 | = 320 / 4 |
| `SIG_NET_INPUT_SIZE` | 164 | = 80 + 2×40 + 4 |
| `SIG_NET_FWC0_STATE_SIZE` | 328 | = 2 × 164 |
| `SIG_NET_FWC0_CONV_OUT_SIZE` | 192 | FWConv output before GLU |
| `SIG_NET_FWC0_GLU_GATE_OUT_SIZE` | 192 | Same as conv out (GLU preserves size) |
| `SIG_NET_GRU1_OUT_SIZE` | 160 | GRU1 hidden size |
| `SIG_NET_GRU2_OUT_SIZE` | 128 | GRU2 hidden size |
| `SIG_NET_GRU3_OUT_SIZE` | 128 | GRU3 hidden size |
| `SIG_NET_SKIP_DENSE_OUT_SIZE` | 128 | Skip connection dense output |
| `FARGAN_MAX_RNN_NEURONS` | 160 | = max(GRU1, GRU2, GRU3) hidden sizes |

### Magic Numbers

| Value | Location | Meaning |
|-------|----------|---------|
| `224` | Embedding size / period clamp | = `PITCH_MAX_PERIOD - PITCH_MIN_PERIOD` = 256 - 32 |
| `32` | Period offset | = `PITCH_MIN_PERIOD` |
| `1e-5f` | Gain inverse computation | Epsilon to prevent division by zero |
| `0.85f` | De-emphasis coefficient | Standard pre-/de-emphasis for speech at 16 kHz |
| `1./60.` | Period formula | Scaling factor for log-frequency feature |
| `1.5` | Period formula | Offset in feature-to-frequency conversion |
| `60` | Period formula | Feature range scaling |
| `256.` | Period formula | = `PITCH_MAX_PERIOD`, base period for conversion |
| `32768.f` | Float-to-int16 | Full-scale factor |
| `32767` | Float-to-int16 clamp | Max int16 magnitude |
| `10000` | `skip_cat[]` declaration | Overly large stack allocation (only ~688 used) |
| `5` | Number of feature frames in `fargan_cont` | 3 look-back + 2 current frames |

## 9. Edge Cases

### Period Clamping

```c
IMAX(0, IMIN(period-32, 223))
```

The embedding index is clamped to [0, 223]. Periods below 32 or above 255 are clamped to the nearest valid value.

### Pitch Buffer Wrap-Around

```c
pos++;
if (pos == PITCH_MAX_PERIOD) pos -= period;
```

When the read position reaches the end of the pitch buffer, it wraps back by one pitch period. This creates the periodic repetition needed for pitch prediction. If `period` is very small relative to the buffer, this can wrap multiple times during the 44-sample read.

### Zero Gain

The `1e-5f` epsilon in `gain_1 = 1.f/(1e-5f + gain)` prevents division by zero when the gain prediction outputs a very large negative number (since `gain = exp(x)` approaches 0 as x → -∞).

### Continuation with Zero PCM

The demo code calls `fargan_cont` with `zeros[320]`, meaning the pitch buffer is populated with zeros. The first `run_fargan_subframe` call after `fargan_cont` will have only pre-emphasis difference values in the pitch buffer (which are also zero in this case). This is a valid cold-start pattern.

### Pre-emphasis Edge at Index 0

In `fargan_cont`:
```c
x0[0] = 0;  // Not pcm0[0], because there's no pcm0[-1]
```

The first pre-emphasized sample is explicitly zeroed rather than attempting to use a nonexistent predecessor sample.

### `cont_initialized` Guard

Both `run_fargan_subframe` and `fargan_synthesize_impl` assert `st->cont_initialized`. Calling `fargan_synthesize` before `fargan_cont` is undefined behavior in release builds (no assertion) — the GRU states, pitch buffer, and conv states would all be zero.

### Asymmetric Int16 Clamp

The float-to-int16 conversion clamps to [-32767, 32767], not [-32768, 32767]. This is intentional — it ensures symmetric clipping and avoids the problematic -32768 value.

## 10. Porting Notes

### Stack Allocation

`run_fargan_subframe` allocates large buffers on the stack:
- `float skip_cat[10000]` — 40 KB on the stack, massively over-allocated
- Multiple other arrays totaling several KB

**Rust approach**: Use `Vec` or sized arrays. The `skip_cat` buffer only needs ~688 floats. Consider computing the exact required size from the constants. For performance-critical paths, consider stack-allocated arrays with const generics or `arrayvec`.

### In-Place Mutation Patterns

Several patterns where output aliases input:
- `compute_glu(&model->sig_net_fwc0_glu_gate, gru1_in, gru1_in, ...)` — in-place GLU
- `compute_glu(&model->sig_net_skip_glu_gate, skip_out, skip_out, ...)` — in-place GLU
- The `compute_glu` function explicitly handles `input == output` with a separate code path

**Rust approach**: The borrow checker will reject aliased mutable references. Options:
1. Always use separate input/output buffers and copy
2. Use a single mutable slice and document that `compute_glu` is safe for in-place operation
3. Use `unsafe` with careful documentation (not recommended for this project)

### Pointer Arithmetic and Buffer Indexing

The pitch buffer read loop uses raw index manipulation with wrap-around:
```c
pos = PITCH_MAX_PERIOD - period - 2;
for (i=0; i<FARGAN_SUBFRAME_SIZE+4; i++) {
    pred[i] = ... st->pitch_buf[IMAX(0, pos)] ...;
    pos++;
    if (pos == PITCH_MAX_PERIOD) pos -= period;
}
```

**Rust approach**: Use checked indexing or assert bounds. The `IMAX(0, pos)` guard suggests `pos` can be negative initially (when `period > 254`). In Rust, use `usize` carefully or `isize` with explicit bounds checking.

### `OPUS_COPY` / `OPUS_MOVE` Semantics

- `OPUS_COPY` = `memcpy` (no overlap allowed)
- `OPUS_MOVE` = `memmove` (overlap safe)

**Rust approach**: Use `slice::copy_from_slice` for OPUS_COPY and `slice::copy_within` for OPUS_MOVE.

### `compute_generic_conv1d` State Management

The conv1d implementation concatenates `[state, current_input]`, computes the linear layer, then copies the new tail into state:
```c
if (layer->nb_inputs != input_size)
    OPUS_COPY(tmp, mem, layer->nb_inputs - input_size);
OPUS_COPY(&tmp[layer->nb_inputs - input_size], input, input_size);
compute_linear(layer, output, tmp, arch);
if (layer->nb_inputs != input_size)
    OPUS_COPY(mem, &tmp[input_size], layer->nb_inputs - input_size);
```

**Rust approach**: This is a standard causal convolution with explicit state. Model the state as a fixed-size ring buffer or shift register.

### FWConv (Frame-Wise Convolution)

The FWConv in the C code is implemented as a `compute_generic_conv1d` with `kernel_size=2`, followed by a GLU gate. The state size is `2 * SIG_NET_INPUT_SIZE` (one previous input frame). The Python `FWConv.forward` returns `(output, new_state)` — in C this is implicit via the `st->fwc0_mem` buffer.

### Auto-Generated Code (`fargan_data.h/c`)

The `fargan_data.h` and `fargan_data.c` files are auto-generated by `dump_fargan_weights.py` using the `wexchange` library. They are **not checked into the repo** and must be generated from a trained model checkpoint.

**Rust approach**: Either:
1. Port the weight-exchange format parser (`parse_weights`) and load at runtime
2. Generate Rust source from the weight dump (build script)
3. Use `include_bytes!` with the binary weight blob and parse at initialization

### Conditional Compilation

```c
#ifndef USE_WEIGHTS_FILE
  ret = init_fargan(&st->model, fargan_arrays);
#else
  ret = 0;
#endif
```

When `USE_WEIGHTS_FILE` is defined, compiled-in weights are not available and must be loaded via `fargan_load_model`. The Rust port should use a feature flag for this.

### Architecture Dispatch (`st->arch`)

The `arch` field is passed to every `compute_*` function for SIMD dispatch. In the base C implementation (no SIMD), `arch` is ignored via `(void)(arch)` casts. The Rust port can initially ignore this and add SIMD dispatch later.

### `FARGAN_FEATURES` Unused Macro

```c
#define FARGAN_FEATURES (NB_FEATURES)
```

Defined but never referenced in `fargan.c`. Can be omitted in the Rust port.

### Period Timing: `last_period` vs Current Period

A subtle but critical detail: `run_fargan_subframe` receives `st->last_period` (not the current frame's period). The current period is only stored at the end of `fargan_synthesize_impl`:
```c
st->last_period = period;
```

In `fargan_cont`, this pattern is established by computing `st->last_period = period` in the feature pre-loading loop, so by the time `run_fargan_subframe` is called, `last_period` holds the period from the penultimate feature frame.

### `MIN16` vs `MIN32` Inconsistency

Line 98 uses `MIN16` where `MIN32` is used everywhere else for the same clamping pattern:
```c
for (i=0;i<FARGAN_SUBFRAME_SIZE;i++)
    prev[i] = MAX32(-1.f, MIN16(1.f, gain_1*st->pitch_buf[...]));
```

Both macros likely expand to the same thing for `float` arguments, but this inconsistency should be normalized in the Rust port (just use `f32::clamp`).

### GRU Weight Layout

The GRU's recurrent weights have `nb_inputs = N` (hidden size) and `nb_outputs = 3*N` (z, r, h gates concatenated). The input weights have `nb_inputs = input_size` and `nb_outputs = 3*N`. Both are stored as flat matrices. The z/r/h split happens at indices `[0..N]`, `[N..2N]`, `[2N..3N]` in the output.

### No Dynamic Memory

FARGAN allocates no heap memory internally. All buffers are either stack-local or inside `FARGANState`. The weight data is either statically linked (`fargan_arrays`) or loaded from a caller-provided blob. This is ideal for Rust's ownership model — `FARGANState` can be a simple struct with no `Drop` impl needed.
