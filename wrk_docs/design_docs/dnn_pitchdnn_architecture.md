Now I have everything needed. Here is the complete architecture document.

---

# PitchDNN Module — Architecture Documentation

## 1. Purpose

PitchDNN is a neural-network-based pitch estimator within the Opus codec's DNN subsystem. It replaces (or augments) classical autocorrelation pitch detection with a small recurrent neural network that jointly processes two input streams:

- **Instantaneous Frequency (IF) features** — spectral phase-difference features extracted per frame
- **Cross-correlation (xcorr) features** — normalized autocorrelation values at candidate pitch lags

The network outputs a continuous log-frequency pitch estimate used by the LPCNet encoder for pitch-period selection. It is called once per frame from `compute_frame_features()` in `lpcnet_enc.c`.

**Location in pipeline:** Audio input → frame analysis → IF features + xcorr features → **PitchDNN** → pitch period → LPCNet encoder features.

---

## 2. Public API

### `void pitchdnn_init(PitchDNNState *st)`

Initializes the state struct to zero and loads compiled-in model weights.

| Parameter | Type | Description |
|-----------|------|-------------|
| `st` | `PitchDNNState *` | State struct to initialize |

**Behavior:**
- Zeros the entire struct via `OPUS_CLEAR(st, 1)` (equivalent to `memset(st, 0, sizeof(*st))`)
- If `USE_WEIGHTS_FILE` is **not** defined: calls `init_pitchdnn(&st->model, pitchdnn_arrays)` to load statically-compiled weights
- Asserts the initialization succeeded (`ret == 0`)

### `int pitchdnn_load_model(PitchDNNState *st, const void *data, int len)`

Loads model weights from a binary blob at runtime (alternative to compiled-in weights).

| Parameter | Type | Description |
|-----------|------|-------------|
| `st` | `PitchDNNState *` | State struct |
| `data` | `const void *` | Serialized weight data |
| `len` | `int` | Length of data in bytes |
| **Returns** | `int` | `0` on success, `-1` on failure |

**Behavior:**
- Calls `parse_weights(&list, data, len)` to deserialize into a `WeightArray` list
- Calls `init_pitchdnn(&st->model, list)` to populate the model layers
- Frees the parsed weight list

### `float compute_pitchdnn(PitchDNNState *st, const float *if_features, const float *xcorr_features, int arch)`

Runs one frame of pitch estimation through the neural network.

| Parameter | Type | Size | Description |
|-----------|------|------|-------------|
| `st` | `PitchDNNState *` | — | Persistent state (GRU + conv memory) |
| `if_features` | `const float *` | 88 floats | Instantaneous frequency features |
| `xcorr_features` | `const float *` | 224 floats | Normalized cross-correlation values |
| `arch` | `int` | — | CPU feature flags (for SIMD dispatch) |
| **Returns** | `float` | — | Log-frequency pitch estimate (see §6) |

---

## 3. Internal State

### `PitchDNNState` (defined in `pitchdnn.h`)

```c
typedef struct {
  PitchDNN model;                                    // Neural network weights
  float gru_state[GRU_1_STATE_SIZE];                 // GRU hidden state (64 floats)
  float xcorr_mem1[(NB_XCORR_FEATURES + 2)*2];       // Conv2d_1 temporal memory (452 floats)
  float xcorr_mem2[(NB_XCORR_FEATURES + 2)*2*8];     // Conv2d_2 temporal memory (3616 floats)
  float xcorr_mem3[(NB_XCORR_FEATURES + 2)*2*8];     // Unused (legacy from 3-conv variant)
} PitchDNNState;
```

| Field | Size (floats) | Lifecycle | Purpose |
|-------|---------------|-----------|---------|
| `model` | N/A | Initialized once, read-only thereafter | Weight tensors for all layers |
| `gru_state` | 64 | Zeroed at init, updated every frame | GRU recurrent hidden state |
| `xcorr_mem1` | 452 | Zeroed at init, updated every frame | Causal temporal buffer for conv2d_1 (stores 2 previous time frames) |
| `xcorr_mem2` | 3616 | Zeroed at init, updated every frame | Causal temporal buffer for conv2d_2 (stores 2 previous time frames) |
| `xcorr_mem3` | 3616 | Zeroed at init, **never used** | Vestigial buffer from `PitchDNNXcorr` variant (had 3 conv layers) |

### `PitchDNN` (auto-generated struct in `pitchdnn_data.h`)

Contains the model weights. Fields inferred from the export script:

```c
typedef struct PitchDNN {
  LinearLayer dense_if_upsampler_1;    // 88 → 64, quantized int8
  LinearLayer dense_if_upsampler_2;    // 64 → 64, quantized int8
  Conv2dLayer conv2d_1;                // 1→4 channels, 3×3 kernel, float weights
  Conv2dLayer conv2d_2;                // 4→1 channels, 3×3 kernel, float weights
  LinearLayer dense_downsampler;       // 288 → 64, quantized int8
  LinearLayer gru_1_input;             // GRU input weights (64→192 logically)
  LinearLayer gru_1_recurrent;         // GRU recurrent weights (64→192 logically)
  LinearLayer dense_final_upsampler;   // 64 → 192, quantized int8
} PitchDNN;
```

### Supporting types (from `nnet.h`)

```c
typedef struct {
  const float *bias;
  const float *subias;
  const opus_int8 *weights;       // Quantized int8 weights (may be NULL)
  const float *float_weights;     // Float weights (may be NULL)
  const int *weights_idx;
  const float *diag;
  const float *scale;             // Per-output dequantization scale
  int nb_inputs;
  int nb_outputs;
} LinearLayer;

typedef struct {
  const float *bias;
  const float *float_weights;
  int in_channels;
  int out_channels;
  int ktime;
  int kheight;
} Conv2dLayer;
```

---

## 4. Algorithm

`compute_pitchdnn` executes a 7-step neural inference pipeline:

### Step 1 — IF Feature Upsampling

Two dense layers project the 88-dim IF features up to 64 dimensions:

```
if_features[88] → dense_if_upsampler_1 → tanh → if1_out[64]
if1_out[64]     → dense_if_upsampler_2 → tanh → downsampler_in[224..287]
```

The result is written directly into the upper portion of the `downsampler_in` buffer, preparing for concatenation.

### Step 2 — Xcorr Convolution (Conv2d_1)

The 224-dim xcorr features are treated as a 1-channel 2D signal (1 channel × 224 spatial positions × time) and processed by a `3×3` convolution with 4 output channels:

```c
// Pad xcorr_features with one zero on each spatial side
OPUS_COPY(&conv1_tmp1[1], xcorr_features, NB_XCORR_FEATURES);  // [0, xcorr[0..223], 0, ...]

// Conv2d: 1 in → 4 out channels, kernel 3×3, hstride=226
compute_conv2d(&model->conv2d_1, &conv1_tmp2[1], st->xcorr_mem1,
               conv1_tmp1, 224, 226, ACTIVATION_TANH, arch);
```

- **Input layout:** `conv1_tmp1` has `(NB_XCORR_FEATURES + 2) * 8` floats, zero-initialized. Only positions `[1..224]` contain data. The leading zero and trailing zeros provide spatial zero-padding (matching PyTorch `ZeroPad2d((2,0,1,1))`).
- **Output layout:** Written to `&conv1_tmp2[1]`, 4 channels × 226 stride. The `[1]` offset provides zero-padding for the next conv layer's spatial kernel.
- **Temporal state:** `xcorr_mem1` stores 2 previous time frames (for the `ktime=3` causal convolution).

### Step 3 — Xcorr Convolution (Conv2d_2)

Second 2D convolution reduces back to 1 output channel:

```c
compute_conv2d(&model->conv2d_2, downsampler_in, st->xcorr_mem2,
               conv1_tmp2, 224, 224, ACTIVATION_TANH, arch);
```

- **Input:** `conv1_tmp2`, 4 channels of 224-element spatial vectors
- **Output:** Written to `downsampler_in[0..223]`, completing the lower portion of the concatenated feature vector
- **Temporal state:** `xcorr_mem2` stores 2 previous time frames

### Step 4 — Feature Concatenation (Implicit)

At this point, `downsampler_in` contains:

| Index range | Content | Size |
|------------|---------|------|
| `[0..223]` | Conv2d output from xcorr path | 224 floats |
| `[224..287]` | IF upsampled features from Step 1 | 64 floats |

Total: 288 floats.

### Step 5 — Downsampling Dense Layer

```c
compute_generic_dense(&model->dense_downsampler, downsampler_out,
                      downsampler_in, ACTIVATION_TANH, arch);
```

Projects the 288-dim concatenated features down to 64 dimensions with tanh activation.

### Step 6 — GRU Update

```c
compute_generic_gru(&model->gru_1_input, &model->gru_1_recurrent,
                    st->gru_state, downsampler_out, arch);
```

Updates the persistent 64-dim GRU hidden state. The GRU equations (from `nnet.c`):

```
z, r, h = split(W_in × input + W_rec × state, 3)
z, r = sigmoid(z), sigmoid(r)
h = tanh(h_input + (W_rec_h × state) * r)
state = z * state + (1 - z) * h
```

Note: This is the standard GRU formulation with reset gate applied **after** the recurrent projection.

### Step 7 — Final Upsampling + Soft Argmax

```c
compute_generic_dense(&model->dense_final_upsampler, output,
                      st->gru_state, ACTIVATION_LINEAR, arch);
```

Projects GRU state to 192 output logits (**linear activation** — no nonlinearity). Then applies a soft argmax over bins 0–179:

```c
// Hard argmax to find peak bin
for (i=0; i<180; i++) {
    if (output[i] > maxval) { pos = i; maxval = output[i]; }
}
// Soft argmax: exp-weighted average over [pos-2, pos+2]
for (i = IMAX(0, pos-2); i <= IMIN(179, pos+2); i++) {
    float p = exp(output[i]);
    sum += p * i;
    count += p;
}
return (1.f/60.f) * (sum/count) - 1.5f;
```

This yields sub-bin resolution pitch estimation.

---

## 5. Data Flow

### Input Features

**IF features** (`if_features`, 88 floats):
- `if_features[0]` = clipped log-power of DC bin: `MAX(-1, MIN(1, (1/64) * (10*log10(|X[0]|²) - 6)))`
- For bins `i = 1..29`:
  - `if_features[3*i-2]` = real part of normalized inter-frame phase difference
  - `if_features[3*i-1]` = imaginary part of normalized inter-frame phase difference
  - `if_features[3*i]` = clipped log-power of bin `i`
- Total: `1 + 29*3 = 88` = `PITCH_IF_FEATURES = 3*PITCH_IF_MAX_FREQ - 2`
- Value range: approximately `[-1, 1]` (phase diffs are unit-normalized; powers are clipped)

**Xcorr features** (`xcorr_features`, 224 floats):
- `xcorr_features[i]` = `2 * xcorr[i] / (1 + ener0 + ener1)` for lag offsets `i = 0..223`
- Lag offset `i` corresponds to pitch period `PITCH_MIN_PERIOD + i` = `32 + i` samples
- Value range: approximately `[-2, 2]` (normalized but scaled by factor 2)

### Buffer Layouts

```
conv1_tmp1[(NB_XCORR_FEATURES + 2) * 8]:
  Index:    0    1..224    225   226..451   ...
  Content: [0] [xcorr] [0..0] [0..0..0] [...]
  Only first 226 elements used; rest zero-initialized.

conv1_tmp2[(NB_XCORR_FEATURES + 2) * 8]:
  After conv2d_1 (written at offset 1):
  Channel 0: [0] [out_0[0..223]] [0]  (stride 226)
  Channel 1: [0] [out_1[0..223]] [0]  (stride 226)
  Channel 2: [0] [out_2[0..223]] [0]
  Channel 3: [0] [out_3[0..223]] [0]
  Leading zeros from zero-init; position [0] per channel is spatial padding.

downsampler_in[224 + 64 = 288]:
  [0..223]  = xcorr conv output (from conv2d_2)
  [224..287] = IF upsampled features (from dense_if_upsampler_2)
```

### Output

Return value is a **log-frequency pitch index** in a custom encoding:

```
return_value = (1/60) * weighted_bin_index - 1.5
```

The caller in `lpcnet_enc.c` converts to a pitch period in samples:

```c
pitch = (int)floor(0.5 + 256.0 / pow(2.0, (1.0/60.0) * ((dnn_pitch + 1.5) * 60)));
// Simplifies to: pitch = round(256 / 2^(weighted_bin_index / 60))
```

Where: `freq = 62.5 × 2^(bin/60)`, so `period = 16000/freq = 256 / 2^(bin/60)`.

The 192 output bins span pitch frequencies from 62.5 Hz (bin 0, period 256) up to approximately 4000 Hz (bin 191). Only bins 0–179 are used in the soft argmax; bins 180–191 are computed but ignored.

---

## 6. Numerical Details

### Floating-Point Precision

All computation is **single-precision float (`float`)** throughout. There is no fixed-point path in this module.

### Weight Quantization

Dense layers (`LinearLayer`) use **int8 quantized weights** with per-output float scales, dequantized during `compute_linear`. Conv2d layers use **float weights** directly (`float_weights` field). This is set during the export: dense layers pass `quantize=True`, conv layers do not.

### Activation Functions

| Layer | Activation | Implementation |
|-------|-----------|----------------|
| dense_if_upsampler_1 | tanh | `vec_tanh` (polynomial approximation) |
| dense_if_upsampler_2 | tanh | `vec_tanh` |
| conv2d_1 | tanh | `vec_tanh` per channel |
| conv2d_2 | tanh | `vec_tanh` per channel |
| dense_downsampler | tanh | `vec_tanh` |
| GRU gates (z, r) | sigmoid | `vec_sigmoid` |
| GRU candidate (h) | tanh | `vec_tanh` |
| dense_final_upsampler | **linear** | Identity (no-op) |

`vec_tanh` and `vec_sigmoid` are fast polynomial approximations (not `libm`), so outputs will not be bit-exact with standard `tanh()`/`sigmoid()`. The `#ifdef HIGH_ACCURACY` path uses `libm` but is off by default.

### Soft Argmax Precision

The soft argmax uses `exp()` from `<math.h>` (not an approximation). Since the output logits have **linear activation**, they can be arbitrarily large. The `exp()` values could overflow if logits are extreme, but in practice the trained model produces bounded logits.

The division `sum/count` has no explicit zero-division guard. `count` is always positive because `exp(x) > 0` for all finite `x`, and at least one element (at `pos`) is in the window.

### Rounding Behavior

The soft argmax provides sub-bin resolution. The caller applies `floor(0.5 + ...)` for final rounding to integer period. The pitch DNN output itself is continuous.

---

## 7. Dependencies

### Modules Called by PitchDNN

| Function | Source | Purpose |
|----------|--------|---------|
| `compute_generic_dense` | `nnet.c` | Dense (fully-connected) layer forward pass |
| `compute_conv2d` | `nnet_arch.h` (macro → `compute_conv2d_c`) | 2D convolution forward pass |
| `compute_generic_gru` | `nnet.c` | GRU recurrent layer update |
| `init_pitchdnn` | `pitchdnn_data.c` (auto-generated) | Load weight arrays into `PitchDNN` struct |
| `parse_weights` | `parse_lpcnet_weights.c` | Deserialize binary weight blob into `WeightArray` list |
| `OPUS_COPY` | `os_support.h` | `memcpy` with type checking |
| `OPUS_CLEAR` | `os_support.h` | `memset` to zero |
| `IMAX`, `IMIN` | `arch.h` | Integer min/max |
| `exp()` | `<math.h>` | Used in soft argmax |

### Modules That Call PitchDNN

| Caller | Source | Context |
|--------|--------|---------|
| `compute_frame_features` | `lpcnet_enc.c:187` | Per-frame pitch estimation |
| `lpcnet_encoder_init` | `lpcnet_enc.c:55` | State initialization |
| `lpcnet_encoder_load_model` | `lpcnet_enc.c:60` | Runtime weight loading |

The `PitchDNNState` is embedded directly within `LPCNetEncState`:

```c
// In lpcnet_private.h (LPCNetEncState struct)
PitchDNNState pitchdnn;
```

---

## 8. Constants and Tables

### Compile-Time Constants

| Constant | Value | Derivation |
|----------|-------|------------|
| `PITCH_MIN_PERIOD` | 32 | Minimum pitch period in samples (500 Hz at 16 kHz) |
| `PITCH_MAX_PERIOD` | 256 | Maximum pitch period in samples (62.5 Hz at 16 kHz) |
| `NB_XCORR_FEATURES` | 224 | `PITCH_MAX_PERIOD - PITCH_MIN_PERIOD` = 256 − 32 |
| `PITCH_IF_MAX_FREQ` | 30 | Number of spectral bins used for IF features |
| `PITCH_IF_FEATURES` | 88 | `3 * PITCH_IF_MAX_FREQ - 2` = 3 × 30 − 2 |

### Auto-Generated Constants (from `pitchdnn_data.h`)

These values come from the trained PyTorch model (`PitchDNN(input_IF_dim=88, input_xcorr_dim=224, gru_dim=64, output_dim=192)`):

| Constant | Value | Source |
|----------|-------|--------|
| `GRU_1_STATE_SIZE` | 64 | `gru_dim` |
| `DENSE_IF_UPSAMPLER_1_OUT_SIZE` | 64 | First linear: 88 → 64 |
| `DENSE_IF_UPSAMPLER_2_OUT_SIZE` | 64 | Second linear: 64 → 64 |
| `DENSE_DOWNSAMPLER_OUT_SIZE` | 64 | 288 → 64 |
| `DENSE_FINAL_UPSAMPLER_OUT_SIZE` | 192 | 64 → 192 |
| `PITCH_DNN_MAX_RNN_UNITS` | 64 | Max GRU dimension (for buffer sizing) |

### Magic Numbers in Soft Argmax

| Value | Meaning |
|-------|---------|
| `180` | Number of valid output bins (of 192 total); bins 180–191 are ignored |
| `pos-2` / `pos+2` | 5-bin soft argmax window centered on the hard argmax |
| `1.f/60.f` | Converts bin index to log-frequency units (60 bins per octave = 20 cents per bin) |
| `1.5` | Offset: bin 0 maps to pitch period 256 (62.5 Hz); the `−1.5` shifts the origin |

### Pitch Bin Encoding

The model is trained with ground truth:
```python
cents = round(60 * log2(frequency / 62.5))
cents = clip(cents, 0, 179)
```

So bin `b` represents frequency `62.5 × 2^(b/60)` Hz, or period `256 / 2^(b/60)` samples at 16 kHz.

---

## 9. Edge Cases

### Boundary Handling in Soft Argmax

```c
for (i = IMAX(0, pos-2); i <= IMIN(179, pos+2); i++)
```

- If the hard argmax is at bin 0 or 1, the left side of the window is clamped to 0
- If the hard argmax is at bin 178 or 179, the right side is clamped to 179
- The window always contains at least 1 element (the peak itself), so `count > 0` is guaranteed

### Unused Output Bins

Only 180 of the 192 output bins participate in the argmax. Bins 180–191 exist because `output_dim=192` in the PyTorch model but are never read. This means the final upsampler computes 12 wasted outputs per frame.

### Unused Memory Buffer

`xcorr_mem3` is declared in `PitchDNNState` but never referenced in `compute_pitchdnn`. It was likely used by the `PitchDNNXcorr` variant (which has 3 conv layers with 8 channels) and was not removed when the `PitchDNN` joint model (2 conv layers with 4 channels) became the default. The Rust port should **omit this field**.

### Oversized Memory Buffers

`xcorr_mem2` is declared as `(NB_XCORR_FEATURES + 2) * 2 * 8 = 3616` floats, but `conv2d_2` only needs `(ktime-1) × in_channels × (height + kheight - 1) = 2 × 4 × 226 = 1808` floats. The buffer is 2× oversized, likely from the 8-channel variant. The Rust port should use the **exact required size**.

### Zero-Initialization of Temporaries

```c
float conv1_tmp1[(NB_XCORR_FEATURES + 2)*8] = {0};
float conv1_tmp2[(NB_XCORR_FEATURES + 2)*8] = {0};
```

These are zero-initialized on every call. This is essential because:
- `conv1_tmp1[0]` is the spatial zero-pad before the xcorr data
- Unused elements beyond `NB_XCORR_FEATURES + 1` in `conv1_tmp1` must be zero
- `conv1_tmp2` zero-initialization provides zero-padding at `[0]` per channel for conv2d_2

### No Error Handling in `compute_pitchdnn`

The function has no return code, no bounds checking on inputs, and no NaN/Inf guards. It assumes:
- `st` is properly initialized
- `if_features` points to at least 88 valid floats
- `xcorr_features` points to at least 224 valid floats
- Model weights are valid

---

## 10. Porting Notes

### Conv2d Memory Management

The `compute_conv2d` function implements **causal temporal convolution** by maintaining a ring-buffer-like memory:

```c
// In compute_conv2d_c:
time_stride = conv->in_channels * (height + conv->kheight - 1);
OPUS_COPY(in_buf, mem, (conv->ktime-1) * time_stride);           // Restore previous frames
OPUS_COPY(&in_buf[(conv->ktime-1)*time_stride], in, time_stride); // Append current frame
OPUS_COPY(mem, &in_buf[time_stride], (conv->ktime-1)*time_stride); // Shift memory
```

In Rust, this becomes a slice copy pattern. The `mem` buffer acts as a FIFO for the last `ktime-1` time frames. Key pitfall: the copy from `in_buf` back to `mem` overlaps logically (shifting data left) but uses separate source/dest regions — `memcpy` semantics are sufficient (no `memmove` needed).

### In-Place GRU State Update

The GRU function updates `state` in-place. The assertion `celt_assert(in != state)` ensures the input and state buffers don't alias. In Rust, this naturally maps to passing `&mut state` and `&input` as separate borrows.

### Implicit Feature Concatenation via Buffer Layout

The concatenation of xcorr conv output and IF features is done by writing into different regions of the same buffer:

```c
// IF path writes to upper portion:
compute_generic_dense(..., &downsampler_in[NB_XCORR_FEATURES], ...);
// Conv2d path writes to lower portion:
compute_conv2d(..., downsampler_in, ...);
```

In Rust, this requires either:
- A single `Vec<f32>` with careful slice indexing
- Or computing into separate buffers and concatenating before the downsampler

The second approach is cleaner Rust and avoids borrow conflicts.

### Zero-Initialized Stack Arrays

```c
float conv1_tmp1[(NB_XCORR_FEATURES + 2)*8] = {0};  // 1808 floats on stack
float conv1_tmp2[(NB_XCORR_FEATURES + 2)*8] = {0};  // 1808 floats on stack
```

These are ~14 KB of stack allocation per call, zero-initialized. In Rust, use `[0.0f32; SIZE]` for fixed-size arrays or `vec![0.0; SIZE]` if stack size is a concern. For bit-exactness, the zero-initialization is mandatory.

### RTCD (Runtime CPU Detection) and `arch` Parameter

The `arch` parameter is threaded through all compute functions for SIMD dispatch. In the default (non-SIMD) path, it is discarded via the macro:

```c
#define compute_conv2d(conv, out, mem, in, height, hstride, activation, arch) \
    ((void)(arch), compute_conv2d_c(conv, out, mem, in, height, hstride, activation))
```

The Rust port should ignore `arch` initially and implement only the scalar C path. SIMD can be added later via `std::arch` or `packed_simd`.

### Auto-Generated Weight Initialization

`pitchdnn_data.h` and `pitchdnn_data.c` are **generated files** produced by `export_neuralpitch_weights.py` using the `CWriter` tool. The Rust port needs either:
1. A Rust equivalent of the weight deserializer that reads the same binary format as `parse_weights`
2. A build script that converts the C weight arrays to Rust `const` arrays
3. Runtime loading via `pitchdnn_load_model` only

Option 1 is recommended for compatibility with the existing Opus weight file format.

### Activation Function Approximations

`vec_tanh` and `vec_sigmoid` use polynomial approximations, **not** `libm`. For bit-exact matching with the C reference, the Rust port must replicate these same approximations, not use `f32::tanh()` from the standard library. These are defined in `vec.h` and are architecture-specific.

### Conv2d Weight Layout

Weights are stored in `[out_channels][in_channels][ktime][kheight]` order (row-major). The inner loop accesses:

```c
weights[i*in_channels*ktime*kheight + m*ktime*kheight + t*kheight + h]
```

where `i` = output channel, `m` = input channel, `t` = time offset, `h` = height offset. This is a standard 4D tensor in OIHW (or equivalently OITK) layout.

### Conv2d Spatial Padding Strategy

The PyTorch model uses `ZeroPad2d((2,0,1,1))` which pads:
- Left=2 in time dimension (handled by `xcorr_mem` temporal buffer + ktime=3)
- Right=0 in time dimension (causal — no future frames)
- Top=1 in height dimension (handled by the `[0]` padding in `conv1_tmp1/tmp2`)
- Bottom=1 in height dimension (handled by zero-initialized trailing elements)

The C code manually manages this padding through buffer offsets and zero-initialization rather than an explicit padding operation. The Rust port should replicate this exactly.

### `hstride` vs `height` Distinction

In `compute_conv2d`, `height` is the number of valid spatial output elements, while `hstride` is the memory stride between channels in the output buffer. When `hstride > height`, the extra elements serve as spatial padding for the next layer. The first conv uses `hstride=226` (= `height+2`) to leave room for the second conv's kernel; the second conv uses `hstride=224` (= `height`) with no extra padding.

### Conditional Compilation

```c
#ifndef USE_WEIGHTS_FILE
  ret = init_pitchdnn(&st->model, pitchdnn_arrays);
#else
  ret = 0;
#endif
```

When `USE_WEIGHTS_FILE` is defined, `pitchdnn_init` skips loading compiled-in weights (expecting `pitchdnn_load_model` to be called later). In Rust, this could be a compile-time feature flag or simply two constructors (`PitchDNNState::new()` vs `PitchDNNState::new_empty()`).
