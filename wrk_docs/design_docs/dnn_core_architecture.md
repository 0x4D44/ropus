Now I have a comprehensive understanding of all the files. Let me compile the architecture document.

---

# DNN Core Module Architecture Documentation

**Module**: `dnn/nnet.c`, `dnn/nnet_arch.h`, `dnn/nndsp.c` and supporting headers
**C Reference**: xiph/opus `dnn/` directory
**Porting Target**: Rust, safe code only

---

## 1. Purpose

The DNN core module is the **neural network inference engine** for Opus 1.4+. It provides a small, self-contained library of layer primitives used by all neural enhancement subsystems in Opus:

- **LPCNet** — neural speech codec
- **FARGAN / FWGAN** — neural waveform generators
- **OSCE (LACE/NoLACE)** — opus speech coding enhancement
- **DRED** — deep redundancy coding
- **PitchDNN** — neural pitch detection
- **Lossgen** — packet loss simulator

The module implements these layer types:
1. **Dense (fully-connected)** — matrix-vector multiply + bias + activation
2. **GRU** — gated recurrent unit (recurrent layer)
3. **Conv1D** — causal 1D convolution (with optional dilation)
4. **Conv2D** — 2D convolution (one output frame at a time)
5. **GLU** — gated linear unit
6. **Gated Activation** — generalized gated nonlinearity

The DSP extension module (`nndsp.c`) provides three **adaptive signal processing** primitives:
1. **AdaConv** — adaptive convolution (feature-conditioned FIR filter)
2. **AdaComb** — adaptive comb filter (pitch-conditioned)
3. **AdaShape** — adaptive waveform shaping (temporal envelope control)

All computation is **floating-point** (`float`). This is not fixed-point code — the DNN subsystem operates entirely in IEEE 754 single-precision, unlike the CELT/SILK core which uses fixed-point.

---

## 2. Public API

### 2.1 Layer Computation Functions (`nnet.h` / `nnet.c`)

```c
void compute_generic_dense(
    const LinearLayer *layer,   // Weight matrix + bias
    float *output,              // [nb_outputs]
    const float *input,         // [nb_inputs]
    int activation,             // ACTIVATION_* enum
    int arch                    // RTCD CPU feature flags
);
```
Dense layer: `output = activation(W * input + bias)`.

---

```c
void compute_generic_gru(
    const LinearLayer *input_weights,      // Input-to-hidden weights [3N x M]
    const LinearLayer *recurrent_weights,  // Hidden-to-hidden weights [3N x N]
    float *state,                          // [N] — modified in-place
    const float *in,                       // [M] — current input
    int arch
);
```
GRU recurrent cell. Updates `state` in-place. The `recurrent_weights` layer has `nb_inputs = N` and `nb_outputs = 3*N` (for z, r, h gates).

---

```c
void compute_generic_conv1d(
    const LinearLayer *layer,  // Weights for flattened kernel [nb_outputs x (kernel_size * input_size)]
    float *output,             // [nb_outputs]
    float *mem,                // [nb_inputs - input_size] — delay line, modified in-place
    const float *input,        // [input_size] — current frame's features
    int input_size,            // Width of one time step
    int activation,
    int arch
);
```
Causal 1D convolution. The kernel is flattened into `layer->nb_inputs` = `kernel_size * input_size`. History is maintained in `mem`.

---

```c
void compute_generic_conv1d_dilation(
    const LinearLayer *layer,
    float *output,
    float *mem,                // Delay line for dilated access
    const float *input,
    int input_size,
    int dilation,              // Dilation factor (1 = standard conv)
    int activation,
    int arch
);
```
Dilated causal 1D convolution. When `dilation > 1`, samples from `mem` are gathered at stride `dilation`.

---

```c
void compute_glu(
    const LinearLayer *layer,  // [nb_inputs x nb_outputs], nb_inputs == nb_outputs
    float *output,             // [nb_outputs] — can alias input
    const float *input,        // [nb_inputs]
    int arch
);
```
Gated Linear Unit: `output = input * sigmoid(W * input + bias)`. Supports in-place (`output == input`).

---

```c
void compute_gated_activation(
    const LinearLayer *layer,
    float *output,
    const float *input,
    int activation,            // Activation for the gate (e.g. ACTIVATION_TANH)
    int arch
);
```
Generalized gated nonlinearity: `output = input * activation(W * input + bias)`. Identical to `compute_glu` but with a caller-specified activation instead of hardcoded sigmoid. **Note**: This function is declared in `nnet.h` but its implementation is not present in the current source tree — it may be in an unreleased file or generated. Functionally equivalent to `compute_glu` with a variable activation.

---

### 2.2 Low-Level Primitives (`nnet_arch.h`)

These are the RTCD-dispatched inner kernels. Each architecture variant (C, SSE2, SSE4.1, AVX2, NEON, DOTPROD) provides its own implementation.

```c
void compute_linear_c(const LinearLayer *linear, float *out, const float *in);
void compute_activation_c(float *output, const float *input, int N, int activation);
void compute_conv2d_c(const Conv2dLayer *conv, float *out, float *mem, const float *in,
                       int height, int hstride, int activation);
```

The `_c` suffix denotes the portable C fallback. RTCD macros expand `compute_linear(...)` to the best available implementation.

### 2.3 Weight Parsing & Layer Initialization (`parse_lpcnet_weights.c`)

```c
int parse_weights(WeightArray **list, const void *data, int len);
```
Parses a binary weight blob into an array of `WeightArray` entries. Returns count, or -1 on error. Caller must `opus_free(*list)`.

```c
int linear_init(LinearLayer *layer, const WeightArray *arrays,
    const char *bias, const char *subias, const char *weights,
    const char *float_weights, const char *weights_idx,
    const char *diag, const char *scale,
    int nb_inputs, int nb_outputs);
```
Initializes a `LinearLayer` by looking up named weight arrays. Returns 0 on success, 1 on failure. All pointer fields are set to point into the `WeightArray` data (zero-copy).

```c
int conv2d_init(Conv2dLayer *layer, const WeightArray *arrays,
    const char *bias, const char *float_weights,
    int in_channels, int out_channels, int ktime, int kheight);
```
Initializes a `Conv2dLayer`. Same semantics as `linear_init`.

### 2.4 Adaptive DSP Functions (`nndsp.h` / `nndsp.c`)

```c
void init_adaconv_state(AdaConvState *hAdaConv);
void init_adacomb_state(AdaCombState *hAdaComb);
void init_adashape_state(AdaShapeState *hAdaShape);

void compute_overlap_window(float *window, int overlap_size);

void adaconv_process_frame(
    AdaConvState *hAdaConv, float *x_out, const float *x_in,
    const float *features,
    const LinearLayer *kernel_layer, const LinearLayer *gain_layer,
    int feature_dim, int frame_size, int overlap_size,
    int in_channels, int out_channels, int kernel_size, int left_padding,
    float filter_gain_a, float filter_gain_b, float shape_gain,
    float *window, int arch);

void adacomb_process_frame(
    AdaCombState *hAdaComb, float *x_out, const float *x_in,
    const float *features,
    const LinearLayer *kernel_layer, const LinearLayer *gain_layer,
    const LinearLayer *global_gain_layer,
    int pitch_lag, int feature_dim, int frame_size, int overlap_size,
    int kernel_size, int left_padding,
    float filter_gain_a, float filter_gain_b, float log_gain_limit,
    float *window, int arch);

void adashape_process_frame(
    AdaShapeState *hAdaShape, float *x_out, const float *x_in,
    const float *features,
    const LinearLayer *alpha1f, const LinearLayer *alpha1t,
    const LinearLayer *alpha2,
    int feature_dim, int frame_size, int avg_pool_k, int interpolate_k,
    int arch);
```

---

## 3. Internal State — Structs and Lifecycle

### 3.1 `LinearLayer`

```c
typedef struct {
    const float *bias;          // [nb_outputs] — additive bias
    const float *subias;        // [nb_outputs] — SU-bias (unsigned-quantized variant)
    const opus_int8 *weights;   // [nb_inputs * nb_outputs] or sparse — int8 quantized
    const float *float_weights; // [nb_inputs * nb_outputs] or sparse — float32
    const int *weights_idx;     // Sparse index structure (NULL = dense)
    const float *diag;          // [nb_outputs] — diagonal weights (GRU only)
    const float *scale;         // [nb_outputs] — per-output scale for int8 dequant
    int nb_inputs;
    int nb_outputs;
} LinearLayer;
```

**Lifecycle**: Initialized once via `linear_init()`, then immutable. All pointers reference external weight blob memory (zero-copy). No allocation, no cleanup.

**Weight storage modes**:
- `float_weights != NULL, weights_idx == NULL` → dense float32 matrix
- `float_weights != NULL, weights_idx != NULL` → sparse float32 (8x4 block sparse)
- `weights != NULL, weights_idx == NULL` → dense int8 quantized
- `weights != NULL, weights_idx != NULL` → sparse int8 quantized (8x4 block sparse)
- Both NULL → layer outputs zero (bias-only)

**`diag` field**: Only used for GRU recurrent weights. Adds element-wise `diag[i] * in[i]` for each of the three gate sections (z, r, h). Assert: `3 * nb_inputs == nb_outputs`.

### 3.2 `Conv2dLayer`

```c
typedef struct {
    const float *bias;           // [out_channels]
    const float *float_weights;  // [out_channels * in_channels * ktime * kheight]
    int in_channels;
    int out_channels;
    int ktime;                   // Temporal kernel size
    int kheight;                 // Frequency/spatial kernel size
} Conv2dLayer;
```

**Lifecycle**: Same as `LinearLayer` — initialized once, immutable, zero-copy pointers.

### 3.3 `WeightArray`

```c
typedef struct {
    const char *name;  // NULL-terminated, from blob header
    int type;          // WEIGHT_TYPE_float, _int, _qweight, _int8
    int size;          // Size in bytes
    const void *data;  // Pointer into blob memory
} WeightArray;
```

### 3.4 `WeightHead` (Binary blob header)

```c
typedef struct {
    char head[4];       // Magic bytes
    int version;        // WEIGHT_BLOB_VERSION = 0
    int type;           // WEIGHT_TYPE_*
    int size;           // Payload size in bytes
    int block_size;     // Padded block size (>= size)
    char name[44];      // NULL-terminated name string
} WeightHead;
```

Total header size: 64 bytes (`WEIGHT_BLOCK_SIZE`). Payload follows immediately after.

### 3.5 `AdaConvState`

```c
typedef struct {
    float history[ADACONV_MAX_KERNEL_SIZE * ADACONV_MAX_INPUT_CHANNELS];  // [32 * 3 = 96]
    float last_kernel[ADACONV_MAX_KERNEL_SIZE * ADACONV_MAX_INPUT_CHANNELS
                      * ADACONV_MAX_OUTPUT_CHANNELS];                      // [32 * 3 * 3 = 288]
    float last_gain;
} AdaConvState;
```

Stores the convolution tail from the previous frame and the previous kernel for overlap-add crossfading.

### 3.6 `AdaCombState`

```c
typedef struct {
    float history[ADACOMB_MAX_KERNEL_SIZE + ADACOMB_MAX_LAG];  // [16 + 300 = 316]
    float last_kernel[ADACOMB_MAX_KERNEL_SIZE];                 // [16]
    float last_global_gain;
    int last_pitch_lag;
} AdaCombState;
```

Stores the comb filter history (up to 300 samples of pitch lag) and previous kernel/gain for crossfading.

### 3.7 `AdaShapeState`

```c
typedef struct {
    float conv_alpha1f_state[ADASHAPE_MAX_INPUT_DIM];    // [512]
    float conv_alpha1t_state[ADASHAPE_MAX_INPUT_DIM];    // [512]
    float conv_alpha2_state[ADASHAPE_MAX_FRAME_SIZE];    // [240]
    float interpolate_state[1];                           // Last interpolation sample
} AdaShapeState;
```

Conv1D delay lines for the three linear layers, plus one sample for linear interpolation state.

**Lifecycle for all Ada* states**: Zero-initialized via `init_*()` (which calls `OPUS_CLEAR`). Modified frame-by-frame. Caller manages allocation.

---

## 4. Algorithms

### 4.1 Dense Layer (`compute_generic_dense`)

```
output = activation(W * input + bias)
```

1. Call `compute_linear()` to compute `W * input + bias`
2. Call `compute_activation()` to apply the nonlinearity in-place

### 4.2 Linear Transform (`compute_linear` in `nnet_arch.h`)

The core matrix-vector multiply with four code paths selected at runtime:

```
M = nb_inputs, N = nb_outputs

if float_weights:
    if weights_idx: sparse_sgemv8x4(out, float_weights, idx, N, in)
    else:           sgemv(out, float_weights, N, M, N, in)
elif int8 weights:
    if weights_idx: sparse_cgemv8x4(out, weights, idx, scale, N, M, in)
    else:           cgemv8x4(out, weights, scale, N, M, in)
else:
    OPUS_CLEAR(out, N)   // no weights, output zeros

if bias:  out[i] += bias[i]
if diag:  out[i] += diag[i] * in[i]       // for each of 3 GRU gate sections
```

**Key detail**: `sgemv` dispatches to `sgemv16x1` (rows multiple of 16), `sgemv8x1` (rows multiple of 8), or a generic scalar loop. Weight matrix is stored **column-major** — weights are accessed as `weights[j*col_stride + i]` where `j` is the input (column) index and `i` is the output (row) index.

### 4.3 Int8 Quantized GEMV (`cgemv8x4`)

The quantized path:

1. **Quantize input**: `x[i] = round(127 * _x[i])` (signed int8, range [-127, 127])
   - With `USE_SU_BIAS`: `x[i] = 127 + round(127 * _x[i])` (unsigned, range [0, 254])
2. **Accumulate** in blocks of 8 outputs × 4 inputs: inner products in integer arithmetic
3. **Dequantize output**: `out[i] *= scale[i]` — per-output float scale factor

The `USE_SU_BIAS` variant uses unsigned quantization with a different bias (`subias`) to avoid asymmetric rounding. Both variants produce the same effective result after scale.

### 4.4 Sparse GEMV (`sparse_sgemv8x4` / `sparse_cgemv8x4`)

Block-sparse format with **8×4 blocks**:

```
For each group of 8 output rows:
    colblocks = *idx++          // Number of non-zero 4-column blocks
    for each block:
        pos = *idx++            // Starting column index (must be 4-aligned)
        accumulate 8×4 block of weights × 4 input values
```

The sparse index stores: `[count, pos0, pos1, ..., count, pos0, pos1, ...]` repeating for each group of 8 output rows.

### 4.5 GRU Cell (`compute_generic_gru`)

Standard GRU with coupled input/recurrent computation:

```
N = hidden size
zrh = W_input * in              // [3N] — input contribution
recur = W_recurrent * state     // [3N] — recurrent contribution (includes diag)

// Update and reset gates
z = sigmoid(zrh[0..N] + recur[0..N])
r = sigmoid(zrh[N..2N] + recur[N..2N])

// Candidate hidden state
h = tanh(zrh[2N..3N] + recur[2N..3N] * r)   // reset gate applied to recurrent part only

// Final state
state[i] = z[i] * state[i] + (1 - z[i]) * h[i]
```

**Critical detail**: The `recur` contribution to the candidate `h` is gated by `r` *after* the linear transform, not before. This is the "type 1" GRU variant (not the "type 3" where reset is applied pre-multiplication).

**`diag` optimization**: For the recurrent weights, `diag[i] * in[i]` is added to each of the three gate sections. This captures the dominant diagonal of the recurrent matrix efficiently, allowing the off-diagonal to be sparser.

### 4.6 Conv1D (`compute_generic_conv1d`)

Causal 1D convolution using the LinearLayer as a flattened kernel:

```
kernel_size = nb_inputs / input_size

1. Concatenate: tmp = [mem | input]     // mem holds (kernel_size-1) previous frames
2. output = activation(W * tmp + bias)  // W is the flattened convolution kernel
3. Update mem: shift left by input_size
```

`mem` size = `nb_inputs - input_size` = `(kernel_size - 1) * input_size`.

### 4.7 Conv1D Dilation (`compute_generic_conv1d_dilation`)

For `dilation > 1`:

```
1. Gather: for each of (ksize-1) history taps, copy input_size elements
   from mem at stride (dilation * input_size)
2. Append current input as the last tap
3. output = activation(W * tmp + bias)
4. Update mem: shift and append current input at the end
```

For `dilation == 1`, falls back to the same logic as `compute_generic_conv1d`.

### 4.8 Conv2D (`compute_conv2d` in `nnet_arch.h`)

Processes one time frame at a time with temporal history in `mem`:

```
time_stride = in_channels * (height + kheight - 1)

1. Build input buffer: in_buf = [mem (ktime-1 frames) | current_input (1 frame)]
2. Update mem = in_buf[time_stride:] for next call
3. Compute convolution:
   For each output channel i:
     For each input channel m:
       For each temporal tap t (0..ktime):
         For each kernel height h:
           out[i, j] += weight[i,m,t,h] * in[t, m, j+h]
4. Add per-channel bias
5. Apply per-channel activation
```

Input layout: `[ktime][in_channels][height + kheight - 1]`
Weight layout: `[out_channels][in_channels][ktime][kheight]`
Output layout: `[out_channels][hstride]` (only first `height` positions used per channel)

A specialized 3×3 path (`conv2d_3x3_float`) fully unrolls the inner loops.

### 4.9 Activation Functions (`compute_activation` in `nnet_arch.h`)

| ID | Name | Formula | Implementation |
|----|------|---------|----------------|
| 0 | `LINEAR` | `y = x` | Copy (or no-op if in-place) |
| 1 | `SIGMOID` | `y = σ(x)` | `sigmoid_approx(x) = 0.5 + 0.5 * tanh_approx(0.5 * x)` |
| 2 | `TANH` | `y = tanh(x)` | `tanh_approx(x)` — rational Padé approximant |
| 3 | `RELU` | `y = max(0, x)` | Scalar comparison |
| 4 | `SOFTMAX` | `y = exp(x) / Σexp(x)` | `lpcnet_exp` then normalize (or identity with `SOFTMAX_HACK`) |
| 5 | `SWISH` | `y = x * σ(x)` | `vec_sigmoid` then element-wise multiply |
| 6 | `EXP` | `y = exp(x)` | `lpcnet_exp` (element-wise, no normalization) |

**`SOFTMAX_HACK`**: Defined in `nnet.c`. When active, softmax becomes identity (pass-through). This is used in production Opus — the normalization is handled downstream or is unnecessary for the specific use case.

### 4.10 AdaConv (`adaconv_process_frame`)

Adaptive convolution: a neural network predicts FIR filter kernels per-frame, applied to waveform data with overlap-add crossfading.

```
1. Prepare input: prepend history (kernel_size samples per channel) to x_in
2. Predict kernel: kernel = dense(features, kernel_layer)    // linear activation
3. Predict gain:   gain = exp(a * tanh(dense(features, gain_layer)) + b)
4. Normalize kernel: kernel *= gain / ||kernel||₂            // per output channel
5. Crossfade overlap zone:
   For first overlap_size samples:
     out += window * xcorr(last_kernel, input)
     out += (1-window) * xcorr(new_kernel, input)
   For remaining samples:
     out += xcorr(new_kernel, input)
6. Save kernel and history tail for next frame
```

The convolution is implemented via `celt_pitch_xcorr` (cross-correlation), treating the kernel as a short pattern matched against sliding input. Overlap window: `w[n] = 0.5 + 0.5 * cos(π(n+0.5)/overlap_size)`.

### 4.11 AdaComb (`adacomb_process_frame`)

Adaptive comb filter: similar to AdaConv but filters at pitch-lag delay.

```
1. Prepare input: prepend history (kernel_size + max_lag samples)
2. Predict kernel:       k = dense(features)          // linear
3. Predict gain:         g = exp(log_limit - relu(dense(features)))
4. Predict global gain: gg = exp(a * tanh(dense(features)) + b)
5. Normalize kernel
6. Crossfade at overlap:
   For overlap zone:
     comb_out = last_gg * w * xcorr(last_k, input[-last_lag]) +
                gg * (1-w) * xcorr(k, input[-lag])
     out = comb_out + (w*last_gg + (1-w)*gg) * input
   For remaining:
     out = gg * (xcorr(k, input[-lag]) + input)
7. Save state
```

The comb filter accesses input at `pitch_lag` samples back — this is the "comb" part that reinforces periodic (pitched) content.

### 4.12 AdaShape (`adashape_process_frame`)

Adaptive waveform shaping: computes a time-varying gain envelope and applies it sample-by-sample.

```
1. Compute temporal envelope:
   tenv[i] = log(avg_pool(|x_in|, k) + ε)     // ε = 2^{-16}
   Subtract mean, append mean as extra feature
2. Compute alpha weights:
   hidden = LeakyReLU(conv1d_f(features) + conv1d_t(tenv))   // slope=0.2
   weights = conv1d_2(hidden)
3. Upsample weights via linear interpolation:
   For each hidden_dim sample, interpolate over interpolate_k sub-samples
4. Apply exponential gain:
   x_out[i] = exp(weights[i]) * x_in[i]
```

---

## 5. Data Flow

### 5.1 Overall Data Flow

```
Weight blob (binary)
    │
    ▼
parse_weights() ──► WeightArray[]
    │
    ▼
linear_init() / conv2d_init() ──► LinearLayer / Conv2dLayer (pointers into blob)
    │
    ▼
Caller passes features/audio to:
  compute_generic_dense()     ──► Feature vectors
  compute_generic_gru()       ──► Updated hidden state
  compute_generic_conv1d()    ──► Filtered features
  compute_conv2d()            ──► 2D-convolved features
  adaconv_process_frame()     ──► Filtered audio
  adacomb_process_frame()     ──► Pitch-enhanced audio
  adashape_process_frame()    ──► Shaped audio
```

### 5.2 Buffer Layouts

**Dense layer**: Input `[nb_inputs]`, output `[nb_outputs]`. Input and output must not alias.

**GRU**: Input `[M]`, state `[N]` (in-place). `input_weights` is `[M] → [3N]`, `recurrent_weights` is `[N] → [3N]`. Temporary `zrh[3N]` and `recur[3N]` on stack.

**Conv1D**: Input `[input_size]`, output `[nb_outputs]`, mem `[nb_inputs - input_size]`. Temporary `tmp[nb_inputs]` on stack (max `MAX_CONV_INPUTS_ALL`).

**Conv2D**: Input `[in_channels × (height + kheight - 1)]`, output `[out_channels × hstride]`, mem `[(ktime-1) × in_channels × (height + kheight - 1)]`. Temporary `in_buf[MAX_CONV2D_INPUTS]` on stack.

**AdaConv audio**: Input `[in_channels × frame_size]`, output `[out_channels × frame_size]`. Channels are stored contiguously per channel (not interleaved).

**Weight matrix (column-major)**: Element `W[i][j]` (output i, input j) is at `float_weights[j * col_stride + i]` where `col_stride = nb_outputs`.

---

## 6. Numerical Details

### 6.1 Floating-Point Precision

All computation uses IEEE 754 `float` (single precision, 32-bit). There is **no fixed-point** (Q-format) arithmetic in the DNN module. This is a deliberate design — neural network weights are trained in float and the codec specifies float inference.

### 6.2 Approximation Functions

The module avoids `libm` transcendentals in favor of polynomial approximations for reproducibility and speed:

**`tanh_approx(x)`** — Rational Padé approximant:
```c
num = N0 + x² * (N1 + x² * N2)    // N0=952.528, N1=96.392, N2=0.609
den = D0 + x² * (D1 + x² * D2)    // D0=952.724, D1=413.368, D2=11.886
result = clamp(x * num / den, -1, 1)
```
Maximum error: ~1e-5 in [-8, 8]. Clamped to [-1, 1] for safety.

**`sigmoid_approx(x)`** — Derived from tanh:
```c
sigmoid(x) = 0.5 + 0.5 * tanh_approx(0.5 * x)
```

**`lpcnet_exp2(x)`** — Fast 2^x approximation:
```c
integer = floor(x)
frac = x - integer
result.f = 0.99992522 + frac*(0.69583354 + frac*(0.22606716 + 0.078024523*frac))
result.i = (result.i + (integer << 23)) & 0x7fffffff
```
Uses IEEE 754 bit manipulation: adds the integer part directly to the exponent field. Masks sign bit to ensure non-negative result. Returns 0 for `x < -50`.

**`lpcnet_exp(x)`** = `lpcnet_exp2(x * 1.44269504)` (log2(e) conversion).

**`log2_approx(x)`** (in `common.h`):
```c
integer = (bits >> 23) - 127       // Extract exponent
frac = mantissa_as_float - 1.5
result = 1 + integer + polynomial(frac)
```
Cubic polynomial approximation after extracting the IEEE 754 exponent.

### 6.3 Quantization Details

Int8 quantized weights use a symmetric scheme:
- Input quantization: `x_q = round(127 * x_float)` — clips inputs assumed in [-1, 1]
- Weight storage: `int8` values, already quantized during training
- Accumulation: integer multiply-add (no overflow concern — max accumulation is 4 int8×int8 products per step = max 4×127×127 = 64,516, well within int32)
- Dequantization: `output_float = accumulated_int * scale[output_idx]`

The per-output `scale` factor absorbs both the weight quantization scale and any bias correction.

### 6.4 Precision-Critical Points

1. **GRU gate computation**: Sigmoid/tanh approximation error compounds across time steps. The Padé approximant is chosen for accuracy near zero where gates are most sensitive.

2. **Kernel normalization** in AdaConv/AdaComb: Division by L2 norm with `1e-6` epsilon guard against divide-by-zero.

3. **Gain transform**: `exp(a * tanh(x) + b)` — the tanh clamps the argument to prevent exp overflow.

4. **Softmax epsilon**: `sum = 1/(sum + 1e-30)` prevents division by zero in softmax normalization.

### 6.5 `tansig_table.h`

A precomputed 201-entry lookup table for `tanh(x)` sampled at points `0.0, 0.05, 0.10, ..., 10.0`. Used by some SIMD paths (AVX/NEON) for faster approximation via table lookup + interpolation. The C fallback path uses `tanh_approx()` instead.

---

## 7. Dependencies

### 7.1 What This Module Calls

| Module | Functions Used | Purpose |
|--------|---------------|---------|
| `celt/os_support.h` | `OPUS_COPY`, `OPUS_CLEAR` | memcpy/memset wrappers |
| `celt/arch.h` | `celt_assert`, `MAX32`, `MIN32`, `IMAX` | Assertions, min/max macros |
| `celt/pitch.h` | `celt_pitch_xcorr` | Cross-correlation (used in AdaConv/AdaComb) |
| `celt/mathops.h` | `celt_log` | Log approximation (AdaShape temporal envelope) |
| `opus_types.h` | `opus_int8`, `opus_uint32` | Integer types |
| `opus_defines.h` | `OPUS_INLINE` | Inline qualifier |

### 7.2 What Calls This Module

| Consumer | Usage |
|----------|-------|
| `dnn/fargan.c` | FARGAN waveform generator — dense, GRU, conv1d, gated_activation |
| `dnn/fwgan.c` | FWGAN waveform generator — dense, GRU, conv1d, gated_activation |
| `dnn/lpcnet.c` | LPCNet speech codec — dense, GRU, conv1d |
| `dnn/osce.c` | OSCE (LACE/NoLACE) — dense, GRU, conv1d, conv2d, glu, adaconv, adacomb, adashape |
| `dnn/dred_rdovae_enc.c` | DRED encoder — dense, GRU, conv1d |
| `dnn/dred_rdovae_dec.c` | DRED decoder — dense, GRU, conv1d |
| `dnn/pitchdnn.c` | Pitch DNN — dense, GRU, conv1d, conv2d |
| `dnn/lossgen.c` | Loss generator — dense, GRU |

---

## 8. Constants and Tables

### 8.1 Activation IDs

```c
#define ACTIVATION_LINEAR  0
#define ACTIVATION_SIGMOID 1
#define ACTIVATION_TANH    2
#define ACTIVATION_RELU    3
#define ACTIVATION_SOFTMAX 4
#define ACTIVATION_SWISH   5
#define ACTIVATION_EXP     6
```

### 8.2 Weight Types

```c
#define WEIGHT_TYPE_float   0
#define WEIGHT_TYPE_int     1
#define WEIGHT_TYPE_qweight 2
#define WEIGHT_TYPE_int8    3
```

### 8.3 Buffer Size Limits

| Constant | Value | Used In |
|----------|-------|---------|
| `MAX_ACTIVATIONS` | 4096 | `vec_swish` temporary buffer |
| `MAX_INPUTS` | 2048 | `cgemv8x4` input quantization, `compute_glu` |
| `MAX_CONV2D_INPUTS` | 8192 | `compute_conv2d` input buffer |
| `MAX_RNN_NEURONS_ALL` | `max(FARGAN, PLC, DRED, [OSCE])` | GRU temporary buffers |
| `MAX_CONV_INPUTS_ALL` | `max(DRED_MAX_CONV_INPUTS, 1024)` | Conv1D temporary buffers |
| `WEIGHT_BLOCK_SIZE` | 64 | Binary blob header alignment |
| `SPARSE_BLOCK_SIZE` | 32 | 8×4 sparse block = 32 elements |
| `SCALE` | `128 * 127 = 16256` | Int8 quantization scale constant |
| `SCALE_1` | `1/(128*127)` | Int8 dequantization |

### 8.4 NNDSP Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `ADACONV_MAX_KERNEL_SIZE` | 32 | Max adaptive conv kernel length |
| `ADACONV_MAX_INPUT_CHANNELS` | 3 | Max input channels |
| `ADACONV_MAX_OUTPUT_CHANNELS` | 3 | Max output channels |
| `ADACONV_MAX_FRAME_SIZE` | 240 | Max samples per frame |
| `ADACONV_MAX_OVERLAP_SIZE` | 120 | Max crossfade overlap |
| `ADACOMB_MAX_LAG` | 300 | Max pitch lag (samples) |
| `ADACOMB_MAX_KERNEL_SIZE` | 16 | Max comb kernel length |
| `ADACOMB_MAX_FRAME_SIZE` | 80 | Max samples per frame |
| `ADACOMB_MAX_OVERLAP_SIZE` | 40 | Max crossfade overlap |
| `ADASHAPE_MAX_INPUT_DIM` | 512 | Max feature + envelope dimension |
| `ADASHAPE_MAX_FRAME_SIZE` | 240 | Max samples per frame |

### 8.5 Approximation Coefficients

**tanh Padé**: `N = {952.528, 96.392, 0.609}`, `D = {952.724, 413.368, 11.886}`
**exp2 polynomial**: `K = {0.99993, 0.69583, 0.22607, 0.07802}`
**log2 polynomial**: `{-0.41445, 0.95909, -0.33951, 0.16541}`
**Overlap window**: `w[n] = 0.5 + 0.5 * cos(π(n+0.5)/N)` (raised cosine)

---

## 9. Edge Cases and Error Conditions

### 9.1 Assertions (Fatal)

| Location | Assertion | Meaning |
|----------|-----------|---------|
| `compute_linear` | `in != out` | Input and output must not alias |
| `compute_generic_gru` | `3*rec.nb_inputs == rec.nb_outputs` | Recurrent weights must be 3× square |
| `compute_generic_gru` | `in != state` | Input and state must not alias |
| `compute_generic_conv1d` | `input != output` | No aliasing |
| `compute_conv2d` | `in != out` | No aliasing |
| `compute_conv2d` | `ktime*time_stride <= MAX_CONV2D_INPUTS` | Buffer overflow guard |
| `compute_glu` | `nb_inputs == nb_outputs` | GLU requires square layer |
| `vec_swish` | `N <= MAX_ACTIVATIONS` | Stack buffer guard |
| `adaconv_process_frame` | `shape_gain == 1` | Only shape_gain=1 currently supported |
| `adaconv_process_frame` | `left_padding == kernel_size - 1` | Causal-only |
| `adaconv_process_frame` | `kernel_size < frame_size` | Kernel must fit in frame |
| `adashape_process_frame` | `frame_size % avg_pool_k == 0` | Exact divisibility |

### 9.2 Graceful Handling

- **Null weights**: If both `weights` and `float_weights` are NULL, `compute_linear` outputs zeros (`OPUS_CLEAR`)
- **Null bias**: Skip bias addition
- **Null diag**: Skip diagonal addition
- **`lpcnet_exp2(x)` underflow**: Returns exactly 0.0 for x < -50
- **`tanh_approx` overflow**: Clamped to [-1, 1]
- **Kernel norm zero**: `1/(1e-6 + norm)` prevents division by zero in `scale_kernel`

### 9.3 Weight Parsing Errors

`parse_weights` returns -1 and frees `*list` on:
- Truncated header (`len < WEIGHT_BLOCK_SIZE`)
- `block_size < size`
- Block extends past data (`block_size > len - WEIGHT_BLOCK_SIZE`)
- Unterminated name string
- Negative size

`linear_init` / `conv2d_init` return 1 on:
- Named array not found in weight list
- Array size mismatch (wrong number of bytes for declared dimensions)
- Sparse index validation failure (out-of-bounds position, non-4-aligned position, row count mismatch)

---

## 10. Porting Notes for Rust

### 10.1 RTCD (Runtime CPU Detection) Dispatch

The C code uses a macro-based RTCD system:

```c
// nnet_default.c
#define RTCD_ARCH c
#include "nnet_arch.h"    // Instantiates compute_linear_c, compute_activation_c, compute_conv2d_c

// nnet_sse2.c
#define RTCD_ARCH sse2
#include "nnet_arch.h"    // Instantiates compute_linear_sse2, etc.
```

`nnet_arch.h` is **included multiple times** with different `RTCD_ARCH` values, generating multiple copies of the same functions with different suffixes. The `RTCD_SUF()` macro concatenates the function name with the architecture suffix.

**Rust approach**: Use a trait with associated functions, or `#[cfg(target_feature)]` for compile-time dispatch, or function pointers for runtime dispatch. The portable C implementation should be the starting point.

### 10.2 Stack-Allocated Temporary Buffers

The C code allocates large temporary arrays on the stack:

```c
float zrh[3*MAX_RNN_NEURONS_ALL];    // Could be 3*384 = 1152 floats = 4.5 KB
float in_buf[MAX_CONV2D_INPUTS];     // 8192 floats = 32 KB
float tmp[MAX_CONV_INPUTS_ALL];      // ~4 KB
```

AdaConv alone uses ~6 stack-allocated buffers totaling ~5 KB.

**Rust approach**: These are fine as local arrays in Rust (stack allocation). However, the `MAX_*` constants make them larger than necessary for any given call. Consider using actual dimensions where sizes are known, or `Vec` with reuse. For bit-exact behavior, the buffer sizes don't matter — only the written portion affects output.

### 10.3 Pointer Arithmetic in Sparse GEMV

The sparse index pointer `idx` is consumed via post-increment:

```c
colblocks = *idx++;
pos = (*idx++);
```

**Rust approach**: Use a slice with an advancing index, or an iterator over the index array. Example:
```rust
let mut idx_pos = 0;
let colblocks = idx[idx_pos]; idx_pos += 1;
let pos = idx[idx_pos] as usize; idx_pos += 1;
```

### 10.4 Union-Based Bit Manipulation

`lpcnet_exp2` and `log2_approx` use `union { float f; int/uint32 i; }` for IEEE 754 bit access:

```c
res.i = (res.i + (integer<<23)) & 0x7fffffff;
```

**Rust approach**: Use `f32::to_bits()` / `f32::from_bits()`:
```rust
let bits = res.to_bits();
let bits = (bits.wrapping_add((integer as u32) << 23)) & 0x7FFFFFFF;
let res = f32::from_bits(bits);
```

### 10.5 In-Place Mutations

Several patterns require attention:

1. **GRU state update**: `state` is read and written in the same call
2. **`compute_glu` in-place**: `output == input` is explicitly handled
3. **Conv1D `mem` update**: `mem` is both read and written
4. **AdaConv/AdaComb state**: Multiple fields updated at end of frame

**Rust approach**: These will require `&mut` references. The aliasing constraints in `compute_linear` (`in != out`) align well with Rust's borrow checker. The in-place `compute_glu` case will need either:
- A temporary copy of input before mutation, or
- Unsafe code with `as_ptr()` / `as_mut_ptr()` (not recommended), or
- Restructure to always use separate buffers

### 10.6 `restrict` Qualifier

The GEMV kernels use `float * restrict` for vectorization hints:

```c
const float * restrict w;
float * restrict y;
```

**Rust approach**: Rust references inherently guarantee no aliasing. No special action needed — `&[f32]` and `&mut [f32]` already convey the `restrict` semantics to LLVM.

### 10.7 `SOFTMAX_HACK` and Conditional Compilation

```c
#define SOFTMAX_HACK  // In nnet.c — makes softmax a no-op (pass-through)
```

This is always defined in production builds. The Rust port should implement the same behavior: `ACTIVATION_SOFTMAX` is effectively `ACTIVATION_LINEAR`.

Also watch for:
- `#ifdef ENABLE_OSCE` — controls OSCE-specific buffer sizing
- `#ifdef ENABLE_OSCE_BWE` — BWE extension adds to max neuron count
- `#ifdef USE_SU_BIAS` — switches between signed/unsigned int8 quantization
- `#ifdef HIGH_ACCURACY` — uses `libm` exp/tanh instead of approximations (never enabled in production)

### 10.8 Weight Blob Zero-Copy Pattern

`LinearLayer` and `Conv2dLayer` contain raw pointers into the parsed weight blob. The blob must outlive all layers.

**Rust approach**: Use lifetime annotations:
```rust
struct LinearLayer<'a> {
    bias: Option<&'a [f32]>,
    weights: Option<&'a [i8]>,
    float_weights: Option<&'a [f32]>,
    // ...
}
```
Or use `Arc<WeightBlob>` with index-based access if layers need to be `'static`.

### 10.9 Cross-Module Dependency: `celt_pitch_xcorr`

AdaConv and AdaComb use `celt_pitch_xcorr` from the CELT pitch module. This is a cross-correlation function that also has RTCD dispatch (NEON, SSE, etc.).

**Signature**:
```c
void celt_pitch_xcorr(const float *_x, const float *_y, float *xcorr,
                       int len, int max_pitch, int arch);
```

This computes `max_pitch` correlation values, each being the dot product of `_x[0..len]` with `_y[-i..len-i]` for `i = 0..max_pitch`. The CELT module must be ported (or stubbed) before AdaConv/AdaComb can work.

### 10.10 Column-Major Weight Storage

The `sgemv` function accesses weights as `weights[j*col_stride + i]` — this is column-major (Fortran-style). Each column corresponds to one input, each row to one output. The `col_stride` parameter (set to `nb_outputs` for dense matrices) allows for potential padding.

**Rust approach**: Use a 1D slice with explicit indexing. Consider a newtype wrapper:
```rust
struct ColMajorMatrix<'a> {
    data: &'a [f32],
    rows: usize,    // nb_outputs
    cols: usize,    // nb_inputs
    col_stride: usize,
}
```

### 10.11 GCC Vectorization Pragma

```c
#pragma GCC push_options
#pragma GCC optimize("tree-vectorize")
```

This forces auto-vectorization for the DNN kernels even at `-O1`. The comment in `conv2d_float` says "no intrinsics because gcc auto-vectorizer is smart enough."

**Rust approach**: Rust with `opt-level = 2` or higher will auto-vectorize. For explicit control, use `#[target_feature(enable = "...")]` or write manual SIMD later. The initial port should rely on auto-vectorization and verify performance.

### 10.12 `OPUS_COPY` / `OPUS_CLEAR` Patterns

These expand to `memcpy` and `memset` respectively. The `OPUS_COPY` macro includes a type-checking expression `0*((dst)-(src))` that ensures both pointers have compatible types.

**Rust approach**: `slice.copy_from_slice()` for OPUS_COPY, `slice.fill(0.0)` for OPUS_CLEAR.

### 10.13 `compute_gated_activation` — Missing Implementation

This function is declared in `nnet.h` but has no visible implementation in the current source tree. It's used only by `fwgan.c`. Based on the signature and usage pattern, it is equivalent to:

```c
void compute_gated_activation(const LinearLayer *layer, float *output, const float *input, int activation, int arch)
{
    float gate[MAX_INPUTS];
    compute_linear(layer, gate, input, arch);
    compute_activation(gate, gate, layer->nb_outputs, activation, arch);
    // apply sigmoid gate
    compute_activation(gate, gate, layer->nb_outputs, ACTIVATION_SIGMOID, arch);  // or just use activation directly
    for (int i = 0; i < layer->nb_outputs; i++) output[i] = input[i] * gate[i];
}
```

The exact semantics should be verified when porting FWGAN. It may live in a file not yet cloned or be generated. For the DNN core port, implement the same pattern as `compute_glu` with a variable activation parameter.

### 10.14 Integer Overflow in Quantized GEMV

In `cgemv8x4` / `sparse_cgemv8x4`, the inner accumulation:
```c
y[0] += (w[0]*xj0 + w[1]*xj1 + w[2]*xj2 + w[3]*xj3);
```

Where `w` is `int8` and `xj` is either `int8` or `unsigned char`. The multiplication produces `int` (C promotion rules), and accumulation is in `float`. The implicit int-to-float conversion happens at the `+=` to `float y[0]`.

**Rust approach**: Be explicit about the promotion chain:
```rust
let val = w[0] as i32 * xj0 as i32 + w[1] as i32 * xj1 as i32 + ...;
y[0] += val as f32;
```
