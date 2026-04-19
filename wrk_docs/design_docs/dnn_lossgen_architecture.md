Now I have all the information needed. Here's the complete architecture documentation:

---

# DNN Lossgen Module — Architecture Document

## 1. Purpose

The `lossgen` module is a **neural network–based packet loss pattern generator** used for testing and evaluating Opus's error concealment mechanisms (PLC, DRED, OSCE). Rather than using simple i.i.d. random loss (which is unrealistic), it generates temporally-correlated loss patterns that model real-world network behavior — burst losses, recovery patterns, and loss rate drift.

The module sits **outside the codec pipeline proper**. It is a testing/simulation utility. Its output is a binary sequence (0 = packet received, 1 = packet lost) that can be fed into an Opus decoder's PLC path.

The module is explicitly designed to be **self-contained** — compilable independently of the Opus library. The comment block at `reference/dnn/lossgen.c:27-48` enumerates the exact files needed for standalone compilation.

## 2. Public API

### `lossgen_init`

```c
void lossgen_init(LossGenState *st);
```

Initializes a `LossGenState` with the **compiled-in default weights** (`lossgen_arrays`).

- Zeros the entire struct via `OPUS_CLEAR(st, 1)` (equivalent to `memset(st, 0, sizeof(*st))`)
- Calls `init_lossgen(&st->model, lossgen_arrays)` to populate `LinearLayer` fields from the static weight arrays
- Asserts that `init_lossgen` returns 0 (success)
- After this call, both GRU states are zero-initialized, `last_loss = 0`, `used = 0`

### `lossgen_load_model`

```c
int lossgen_load_model(LossGenState *st, const void *data, int len);
```

Loads model weights from a **binary weight blob** (runtime alternative to compiled-in weights).

- **Parameters**:
  - `data`: Pointer to serialized weight blob (sequence of `WeightHead` + data blocks)
  - `len`: Length of the blob in bytes
- **Returns**: `0` on success, `-1` on failure
- Calls `parse_weights()` to deserialize the blob into a `WeightArray` list, then `init_lossgen()` to bind weights into the `LossGen` struct
- Frees the parsed list after initialization
- **Note**: Does NOT zero the GRU states or reset `used`/`last_loss`. The caller should call `lossgen_init` first or manually zero the state.

### `sample_loss`

```c
int sample_loss(LossGenState *st, float percent_loss);
```

Generates one packet loss decision.

- **Parameters**:
  - `st`: Initialized state (must have been passed to `lossgen_init` or `lossgen_load_model`)
  - `percent_loss`: Target loss rate as a **fraction** (0.0 to 1.0), NOT a percentage. Despite the parameter name, the demo program (`lossgen_demo.c:19`) multiplies by `0.01f` before calling, confirming this is a fraction.
- **Returns**: `1` if the packet is lost, `0` if received
- On first call (`st->used == 0`), runs 1000 warm-up iterations to flush GRU zero-initialization bias, then sets `st->used = 1`
- Updates `st->last_loss` with the result
- The loss decision is **stochastic**: `loss = (rand() / RAND_MAX) < sigmoid_output`

## 3. Internal State

### `LossGenState`

```c
typedef struct {
  LossGen model;                               // Neural network weights/layers
  float gru1_state[LOSSGEN_GRU1_STATE_SIZE];   // GRU1 hidden state (16 floats)
  float gru2_state[LOSSGEN_GRU2_STATE_SIZE];   // GRU2 hidden state (32 floats)
  int last_loss;                                // Previous loss decision (0 or 1)
  int used;                                     // Whether warm-up has been done
} LossGenState;
```

**Fields**:

| Field | Type | Size | Description |
|-------|------|------|-------------|
| `model` | `LossGen` | ~6 `LinearLayer`s | Pointers to weight arrays for all layers |
| `gru1_state` | `float[16]` | 64 bytes | Hidden state of first GRU (persists across calls) |
| `gru2_state` | `float[32]` | 128 bytes | Hidden state of second GRU (persists across calls) |
| `last_loss` | `int` | 4 bytes | Binary: was last packet lost? (fed back as input) |
| `used` | `int` | 4 bytes | Flag: has warm-up been performed? |

**Lifecycle**:
1. Allocate (stack or heap)
2. `lossgen_init()` — zeros everything, loads weights
3. First `sample_loss()` call — triggers 1000 warm-up iterations (modifies GRU states)
4. Subsequent `sample_loss()` calls — normal operation
5. No explicit destructor needed (all weight data is statically allocated or externally owned)

### `LossGen` (generated struct, from `lossgen_data.h`)

Based on the export script and usage in `lossgen.c`, the `LossGen` struct contains six `LinearLayer` fields:

```c
typedef struct {
  LinearLayer lossgen_dense_in;        // Input dense: 2 → 8
  LinearLayer lossgen_gru1_input;      // GRU1 input weights: 8 → 3*16=48
  LinearLayer lossgen_gru1_recurrent;  // GRU1 recurrent weights: 16 → 3*16=48
  LinearLayer lossgen_gru2_input;      // GRU2 input weights: 16 → 3*32=96
  LinearLayer lossgen_gru2_recurrent;  // GRU2 recurrent weights: 32 → 3*32=96
  LinearLayer lossgen_dense_out;       // Output dense: 32 → 1
} LossGen;
```

### Size Constants (from `lossgen_data.h`, inferred from PyTorch model)

From the training script (`train_lossgen.py:51`): `gru1_size=16, gru2_size=32`

From the PyTorch model (`lossgen.py`):
- `dense_in = nn.Linear(2, 8)`
- `gru1 = nn.GRU(8, 16)`
- `gru2 = nn.GRU(16, 32)`
- `dense_out = nn.Linear(32, 1)`

Therefore:
```c
#define LOSSGEN_DENSE_IN_OUT_SIZE   8    // dense_in output size
#define LOSSGEN_GRU1_STATE_SIZE    16    // gru1 hidden size
#define LOSSGEN_GRU2_STATE_SIZE    32    // gru2 hidden size
```

### `LinearLayer` (from `nnet.h`)

```c
typedef struct {
  const float *bias;          // Bias vector [nb_outputs]
  const float *subias;        // SU-bias (alternative bias for int8 arch)
  const opus_int8 *weights;   // Quantized int8 weights [nb_inputs * nb_outputs]
  const float *float_weights; // Float32 weights (used when non-NULL)
  const int *weights_idx;     // Sparse index structure (NULL = dense)
  const float *diag;          // Diagonal weights for GRU recurrent [3*N]
  const float *scale;         // Per-output scale for int8 weights [nb_outputs]
  int nb_inputs;              // Input dimension
  int nb_outputs;             // Output dimension (for GRU layers: 3*hidden_size)
} LinearLayer;
```

## 4. Algorithm

### Neural Network Architecture

The model is a sequential RNN that generates loss probabilities autoregressively:

```
Input: [last_loss (0/1), percent_loss (0.0–1.0)]  — 2 floats
  │
  ▼
Dense_in: Linear(2 → 8) + tanh activation
  │
  ▼
GRU1: GRU(input=8, hidden=16)  — state persists across calls
  │
  ▼
GRU2: GRU(input=16, hidden=32) — state persists across calls
  │
  ▼
Dense_out: Linear(32 → 1) + sigmoid activation
  │
  ▼
Output: loss_probability ∈ (0, 1)
  │
  ▼
Sampling: loss = (rand()/RAND_MAX < probability) ? 1 : 0
```

### Step-by-Step Execution of `sample_loss_impl`

```c
static int sample_loss_impl(LossGenState *st, float percent_loss)
```

**Step 1**: Construct input vector
```c
float input[2];
input[0] = st->last_loss;    // 0.0 or 1.0 (int → float promotion)
input[1] = percent_loss;     // target loss rate as fraction
```

**Step 2**: Input dense layer — `compute_generic_dense_lossgen`
```c
compute_generic_dense_lossgen(&model->lossgen_dense_in, tmp, input, ACTIVATION_TANH, 0);
// tmp[8] = tanh(W_in * input[2] + b_in)
```

**Step 3**: GRU layer 1 — `compute_generic_gru_lossgen`
```c
compute_generic_gru_lossgen(
    &model->lossgen_gru1_input,       // input weights
    &model->lossgen_gru1_recurrent,   // recurrent weights
    st->gru1_state,                   // hidden state [16], modified in-place
    tmp,                              // input from Step 2 [8]
    0                                 // arch (unused, forced to C)
);
```

**Step 4**: GRU layer 2 — uses GRU1's state as input
```c
compute_generic_gru_lossgen(
    &model->lossgen_gru2_input,
    &model->lossgen_gru2_recurrent,
    st->gru2_state,                   // hidden state [32], modified in-place
    st->gru1_state,                   // input = GRU1 output [16]
    0
);
```

**Step 5**: Output dense layer
```c
compute_generic_dense_lossgen(&model->lossgen_dense_out, &out, st->gru2_state, ACTIVATION_SIGMOID, 0);
// out = sigmoid(W_out * gru2_state[32] + b_out)   — scalar float ∈ (0,1)
```

**Step 6**: Stochastic sampling
```c
loss = (float)rand() / (float)RAND_MAX < out;
st->last_loss = loss;
return loss;
```

### GRU Algorithm Detail

`compute_generic_gru_lossgen` implements a standard GRU cell. Given input weights `W`, recurrent weights `U`, and hidden state `h[N]`:

```
zrh = W * input                     // [3*N] = input_weights * in
recur = U * state                   // [3*N] = recurrent_weights * state

// Split into z (update gate), r (reset gate), h (candidate)
z = zrh[0..N]
r = zrh[N..2N]
h_candidate = zrh[2N..3N]

// Add recurrent contributions to gates
z[i] += recur[i]         for i in 0..2N
r[i] += recur[i]         (combined with z in the loop: zrh[i] += recur[i] for i in 0..2N)

// Apply sigmoid to gates
z = sigmoid(z)
r = sigmoid(r)

// Apply reset gate to recurrent candidate
h_candidate[i] += recur[2N+i] * r[i]

// Apply tanh to candidate
h_candidate = tanh(h_candidate)

// Update gate interpolation
state[i] = z[i] * state[i] + (1 - z[i]) * h_candidate[i]
```

**Important GRU variant note**: This GRU applies the reset gate `r` to the _recurrent_ part of the candidate computation _after_ the linear transform (not before, as in some implementations). Specifically:
```c
h[i] += recur[2*N+i] * r[i];   // reset gate applied to recurrent candidate term
```
This is the "type 1" or "modified" GRU variant from the original paper, where `r` gates the recurrent contribution after the linear projection rather than gating the hidden state before it.

## 5. Data Flow

### Buffer Sizes and Layouts

| Buffer | Size | Contents |
|--------|------|----------|
| `input[2]` | 2 floats | `[last_loss, percent_loss]` |
| `tmp[LOSSGEN_DENSE_IN_OUT_SIZE]` | 8 floats | Dense_in output |
| `gru1_state[LOSSGEN_GRU1_STATE_SIZE]` | 16 floats | GRU1 hidden; also serves as GRU2 input |
| `gru2_state[LOSSGEN_GRU2_STATE_SIZE]` | 32 floats | GRU2 hidden; fed to dense_out |
| `out` | 1 float | Scalar loss probability |
| `zrh[3*MAX_RNN_NEURONS_ALL]` | 96 floats | Scratch for GRU gate computation (stack-allocated) |
| `recur[3*MAX_RNN_NEURONS_ALL]` | 96 floats | Scratch for recurrent projection (stack-allocated) |

Where `MAX_RNN_NEURONS_ALL = IMAX(16, 32) = 32`, so GRU scratch buffers are `3*32 = 96` floats each.

### Data Flow Diagram

```
                     ┌─────────────┐
  percent_loss ─────►│             │
                     │ input[2]    │
  last_loss ────────►│             │
                     └──────┬──────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ dense_in        │
                   │ W[8×2] + b[8]  │
                   │ tanh activation │
                   └────────┬────────┘
                            │ tmp[8]
                            ▼
                   ┌─────────────────┐
                   │ GRU1            │
                   │ in: 8 → h: 16  │◄──── gru1_state[16] (in/out)
                   └────────┬────────┘
                            │ gru1_state[16]
                            ▼
                   ┌─────────────────┐
                   │ GRU2            │
                   │ in: 16 → h: 32 │◄──── gru2_state[32] (in/out)
                   └────────┬────────┘
                            │ gru2_state[32]
                            ▼
                   ┌─────────────────┐
                   │ dense_out       │
                   │ W[1×32] + b[1]  │
                   │ sigmoid activ.  │
                   └────────┬────────┘
                            │ out (scalar)
                            ▼
                   ┌─────────────────┐
                   │ Bernoulli       │
                   │ rand() < out    │
                   └────────┬────────┘
                            │
                            ▼
                     loss ∈ {0, 1}
                            │
                            └──────► st->last_loss (feedback)
```

## 6. Numerical Details

### Floating-Point Format

All computation is **single-precision float (f32)**. No fixed-point arithmetic is used in this module.

### Activation Function Approximations

The module forces the C (non-SIMD) code path via:
```c
#define RTCD_ARCH c
#define compute_activation(output, input, N, activation, arch) \
    ((void)(arch), compute_activation_c(output, input, N, activation))
```

This routes to `nnet_arch.h:compute_activation_c`, which calls `vec_sigmoid` and `vec_tanh` from `vec.h`. These use **polynomial approximations**, not `<math.h>` functions:

#### `tanh_approx` (from `vec.h:338-352`)
```c
static OPUS_INLINE float tanh_approx(float x) {
    const float N0 = 952.52801514f;
    const float N1 = 96.39235687f;
    const float N2 = 0.60863042f;
    const float D0 = 952.72399902f;
    const float D1 = 413.36801147f;
    const float D2 = 11.88600922f;
    float X2, num, den;
    X2 = x * x;
    num = fmadd(fmadd(N2, X2, N1), X2, N0);  // N0 + N1*x² + N2*x⁴
    den = fmadd(fmadd(D2, X2, D1), X2, D0);  // D0 + D1*x² + D2*x⁴
    num = num * x / den;
    return MAX32(-1.f, MIN32(1.f, num));       // clamp to [-1, 1]
}
```

This is a rational (Padé-like) approximation: `tanh(x) ≈ x · P(x²) / Q(x²)`, clamped to `[-1, 1]`.

Where `fmadd(a, b, c) = a*b + c` (scalar FMA, not hardware FMA instruction).

#### `sigmoid_approx` (from `vec.h:354-357`)
```c
static inline float sigmoid_approx(float x) {
    return .5f + .5f * tanh_approx(.5f * x);
}
```

Derived from the identity: `σ(x) = (1 + tanh(x/2)) / 2`.

### Precision Requirements

- **NOT bit-exact critical**: This module generates stochastic test patterns. The neural network's purpose is to produce _statistically_ realistic loss patterns, not numerically exact outputs. Small variations in activation function approximations will produce different random sequences but statistically equivalent loss behavior.
- However, for **test harness reproducibility**, matching the C approximations exactly (with the same PRNG seed) will produce identical sequences, which is valuable for differential testing.

### Overflow Considerations

- Input `percent_loss` is expected in `[0.0, 1.0]`. No clamping is applied.
- `last_loss` is always 0 or 1 (integer promoted to float).
- `tanh_approx` clamps output to `[-1, 1]` explicitly.
- `sigmoid_approx` therefore returns values in `[0, 1]`.
- GRU gate activations (sigmoid on z, r; tanh on h) are inherently bounded.
- The output sigmoid guarantees `out ∈ (0, 1)` approximately (subject to approximation clamping at exact 0 and 1).

### Random Number Generation

```c
loss = (float)rand() / (float)RAND_MAX < out;
```

Uses C standard library `rand()`, which is:
- Platform-dependent (different sequences on different implementations)
- Not cryptographically secure (not relevant here)
- **Global state** — calls to `rand()` elsewhere affect this module's output

## 7. Dependencies

### What Lossgen Calls

| Dependency | Source | What's Used |
|------------|--------|-------------|
| `nnet.h` | `dnn/nnet.h` | `LinearLayer` struct, `WeightArray`, activation constants, `compute_linear`/`compute_activation` macros |
| `nnet_arch.h` | `dnn/nnet_arch.h` | `compute_linear_c`, `compute_activation_c` — **directly included** as a C file |
| `parse_lpcnet_weights.c` | `dnn/parse_lpcnet_weights.c` | `parse_weights()`, `linear_init()`, `init_lossgen()` — **directly included** as a C file |
| `vec.h` | `dnn/vec.h` | `tanh_approx`, `sigmoid_approx`, `vec_tanh`, `vec_sigmoid` |
| `lossgen_data.h` | `dnn/lossgen_data.h` (generated) | `LossGen` struct typedef, size constants, `init_lossgen()` function declaration |
| `lossgen_data.c` | `dnn/lossgen_data.c` (generated) | `lossgen_arrays[]` — compiled-in default weights, `init_lossgen()` implementation |
| `arch.h` | `celt/arch.h` | `IMAX`, `MAX32`, `MIN32` macros |
| `os_support.h` | `celt/os_support.h` | `OPUS_CLEAR`, `OPUS_COPY` macros |
| `<stdlib.h>` | C standard | `rand()`, `RAND_MAX` |

### Direct Inclusion Pattern

Critically, `lossgen.c` uses `#include` to directly embed C source files:
```c
#include "parse_lpcnet_weights.c"   // line 70
#include "nnet_arch.h"              // line 71 (contains function definitions)
```

This is done deliberately so lossgen can be compiled as a **standalone unit** without linking against libopus. The included functions get private copies with the `_c` suffix via the `RTCD_ARCH` mechanism.

After inclusion, lossgen **redefines** the compute macros to force the C path:
```c
#undef compute_linear
#undef compute_activation
#define compute_linear(linear, out, in, arch) ((void)(arch),compute_linear_c(linear, out, in))
#define compute_activation(output, input, N, activation, arch) ((void)(arch),compute_activation_c(output, input, N, activation))
```

### What Calls Lossgen

- `lossgen_demo.c` — standalone command-line tool
- Opus test infrastructure — for generating realistic loss patterns during PLC/DRED testing
- The module is called **externally** to the codec; it is not part of the encode/decode pipeline

## 8. Constants and Tables

### Defined Constants

| Constant | Value | Location | Purpose |
|----------|-------|----------|---------|
| `PITCH_MIN_PERIOD` | 32 | `lossgen.h:34` | **Unused** in lossgen — appears to be a leftover from copy-paste |
| `PITCH_MAX_PERIOD` | 256 | `lossgen.h:35` | **Unused** in lossgen |
| `NB_XCORR_FEATURES` | 224 | `lossgen.h:37` | **Unused** in lossgen |
| `LOSSGEN_GRU1_STATE_SIZE` | 16 | `lossgen_data.h` (generated) | GRU1 hidden dimension |
| `LOSSGEN_GRU2_STATE_SIZE` | 32 | `lossgen_data.h` (generated) | GRU2 hidden dimension |
| `LOSSGEN_DENSE_IN_OUT_SIZE` | 8 | `lossgen_data.h` (generated) | Dense input layer output size |
| `MAX_RNN_NEURONS_ALL` | 32 | `lossgen.c:80` | Max of GRU sizes, for scratch buffer sizing |
| `ACTIVATION_TANH` | 2 | `nnet.h:36` | Tanh activation enum |
| `ACTIVATION_SIGMOID` | 1 | `nnet.h:35` | Sigmoid activation enum |

### Warm-up Constant

```c
for (i=0;i<1000;i++) sample_loss_impl(st, percent_loss);
```

The magic number `1000` is the warm-up iteration count. No formal derivation is documented; it's empirically sufficient to flush zero-initialized GRU states into a realistic regime.

### Weight Arrays

`lossgen_arrays[]` (from `lossgen_data.c`, generated by `export_lossgen.py`) contains the following named weight blobs:

| Name (inferred from export script) | Type | Shape |
|-------------------------------------|------|-------|
| `lossgen_dense_in_bias` | float | [8] |
| `lossgen_dense_in_weights` | float | [8 × 2] = [16] |
| `lossgen_gru1_input_bias` | float | [48] |
| `lossgen_gru1_input_weights` | int8 | [48 × 8] = [384] |
| `lossgen_gru1_input_weights_scale` | float | [48] |
| `lossgen_gru1_recurrent_bias` | float | [48] |
| `lossgen_gru1_recurrent_weights` | int8 | [48 × 16] = [768] |
| `lossgen_gru1_recurrent_weights_scale` | float | [48] |
| `lossgen_gru1_recurrent_diag` | float | [48] |
| `lossgen_gru2_input_bias` | float | [96] |
| `lossgen_gru2_input_weights` | int8 | [96 × 16] = [1536] |
| `lossgen_gru2_input_weights_scale` | float | [96] |
| `lossgen_gru2_recurrent_bias` | float | [96] |
| `lossgen_gru2_recurrent_weights` | int8 | [96 × 32] = [3072] |
| `lossgen_gru2_recurrent_weights_scale` | float | [96] |
| `lossgen_gru2_recurrent_diag` | float | [96] |
| `lossgen_dense_out_bias` | float | [1] |
| `lossgen_dense_out_weights` | float | [1 × 32] = [32] |

Note: Dense layers use `quantize=False` in the export script, so they have float weights. GRU layers use `quantize=True`, so they have int8 weights + scale factors + diagonal weights.

## 9. Edge Cases

### First Call Warm-up

When `st->used == 0`, the first call to `sample_loss` runs 1000 iterations of `sample_loss_impl` before returning a result. This:
- Modifies `st->gru1_state` and `st->gru2_state` extensively
- Calls `rand()` 1000 times (consuming PRNG state)
- All use the same `percent_loss` value — if the loss rate changes dramatically after init, the warm-up was done at a different operating point

### `percent_loss` Naming Confusion

The parameter is named `percent_loss` but expects a **fraction** (0.0–1.0), not a percentage (0–100). The demo program (`lossgen_demo.c:19`) converts:
```c
printf("%d\n", sample_loss(&st, percent*0.01f));
```

No clamping or validation is performed on this input.

### NULL Weight Handling

`compute_linear_c` handles the case where both `float_weights` and `weights` are NULL by zeroing the output:
```c
else OPUS_CLEAR(out, N);
```
Then adds bias if present. This is a degenerate case that shouldn't occur with valid model weights.

### `rand()` Global State

`rand()` is not thread-safe on many platforms. Concurrent calls to `sample_loss` from multiple threads will produce undefined behavior with the random number generator. No mutex protection is provided.

### Assertion on Input/State Aliasing

```c
celt_assert(in != state);
```

The GRU function asserts that the input buffer is not the same pointer as the state buffer. In the GRU2 call, `in = gru1_state` and `state = gru2_state`, so this is always satisfied.

## 10. Porting Notes

### A. Self-Contained Module — Simplifies Porting

This module is an excellent candidate for early/independent porting:
- Small, self-contained neural network with clear boundaries
- No interaction with the codec pipeline
- No fixed-point arithmetic
- Simple data flow (sequential, no branching logic)

### B. Direct C File Inclusion

The C code `#include`s `.c` files directly:
```c
#include "parse_lpcnet_weights.c"
#include "nnet_arch.h"
```

In Rust, these become normal module dependencies. The weight parsing and linear algebra functions should be in separate Rust modules that lossgen imports.

### C. Generated Code (`lossgen_data.h/c`)

The `LossGen` struct and `init_lossgen()` function are generated by `export_lossgen.py`. For the Rust port:
- Define the `LossGen` struct manually based on the known architecture
- The `init_lossgen` function maps named weight arrays to `LinearLayer` fields — implement as a Rust constructor
- Static weight data can be included via `include_bytes!` or as const arrays

### D. Macro-Generated Function Names

The `RTCD_SUF` macro concatenates function names:
```c
#define RTCD_SUF(name) CAT_SUFFIX(name, RTCD_ARCH)
// With RTCD_ARCH = c, produces: compute_linear_c, compute_activation_c
```

In Rust, simply implement `compute_linear` and `compute_activation` as regular functions — no RTCD dispatch needed for the initial port.

### E. Stack-Allocated Scratch Buffers

```c
float zrh[3*MAX_RNN_NEURONS_ALL];   // 96 floats = 384 bytes on stack
float recur[3*MAX_RNN_NEURONS_ALL]; // 96 floats = 384 bytes on stack
```

These are small enough for stack allocation in Rust. Use fixed-size arrays: `[f32; 96]`.

### F. In-Place State Mutation

The GRU function mutates `state` in-place:
```c
for (i=0;i<N;i++)
    state[i] = h[i];
```

In Rust, pass `&mut [f32]` for the state. The function signature would be:
```rust
fn compute_gru(
    input_weights: &LinearLayer,
    recurrent_weights: &LinearLayer,
    state: &mut [f32],  // modified in-place
    input: &[f32],
)
```

### G. Overlapping Pointer Semantics in Activations

`compute_activation` is called with `output == input` (in-place):
```c
compute_activation(zrh, zrh, 2*N, ACTIVATION_SIGMOID, arch);
compute_activation(h, h, N, ACTIVATION_TANH, arch);
```

In Rust, use a single `&mut [f32]` slice or process via temporary. The element-wise nature of sigmoid/tanh means in-place is safe (no read-after-write hazard across elements).

### H. `rand()` Replacement

C's `rand()` is platform-dependent and non-reproducible. For the Rust port:
- Use a deterministic PRNG (e.g., `rand::rngs::SmallRng` or a custom LCG) for reproducibility
- If bit-exact matching with the C reference is needed for testing, implement the same LCG that the platform's `rand()` uses (typically MSVC's or glibc's)
- Consider making the RNG a field of `LossGenState` rather than relying on global state

### I. `init_lossgen` — Weight Binding

`init_lossgen` (generated) calls `linear_init` for each layer, which does string-based lookup of weight arrays by name. In Rust:
- The lookup-by-name pattern maps naturally to a `HashMap<String, WeightArray>` or match statements
- For compiled-in weights, consider using const static arrays with direct references (no lookup)

### J. Unused Constants

`PITCH_MIN_PERIOD`, `PITCH_MAX_PERIOD`, and `NB_XCORR_FEATURES` are defined in `lossgen.h` but never used in lossgen code. Do not port these — they appear to be copy-paste artifacts from another module.

### K. Activation Function Reproducibility

For differential testing against the C reference, the Rust port must use the **same** polynomial approximations for tanh and sigmoid:
```rust
fn tanh_approx(x: f32) -> f32 {
    let x2 = x * x;
    let num = ((0.60863042f32 * x2) + 96.39235687) * x2 + 952.52801514;
    let den = ((11.88600922f32 * x2) + 413.36801147) * x2 + 952.72399902;
    (num * x / den).clamp(-1.0, 1.0)
}

fn sigmoid_approx(x: f32) -> f32 {
    0.5 + 0.5 * tanh_approx(0.5 * x)
}
```

Do NOT use `f32::tanh()` or `f32::exp()` — they will produce different results.

### L. Weight Layout for Linear Algebra

`compute_linear_c` uses column-major layout for `sgemv`:
```c
sgemv(out, linear->float_weights, N, M, N, in);
// weights[j * col_stride + i] where j=col, i=row
```

The Rust port needs to match this layout. The weight matrix is `[nb_outputs × nb_inputs]` stored column-major (or equivalently, row-major with transposed semantics — `weights[col * nb_outputs + row]`).

### M. `OPUS_CLEAR` Initialization

`OPUS_CLEAR(st, 1)` zeros the entire `LossGenState` struct. In Rust, initialize with `Default::default()` or zero-fill. Note that this zeros the `LossGen` model pointers too — they are populated by `init_lossgen` immediately after. In Rust, use `Option<LossGen>` or a builder pattern to make the uninitialized state impossible.
