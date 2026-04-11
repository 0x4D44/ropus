//! DNN Core — Neural network inference engine for Opus 1.4+.
//!
//! Matches the C reference: `nnet.c`, `nnet_arch.h`, `nndsp.c`, `vec.h`,
//! `common.h`, `parse_lpcnet_weights.c`, `tansig_table.h`.
//!
//! All computation is IEEE 754 single-precision float. No fixed-point
//! arithmetic in this module (unlike CELT/SILK core).

use std::f32::consts::PI;

// ===========================================================================
// Constants
// ===========================================================================

// Activation function IDs (matches nnet.h)
pub const ACTIVATION_LINEAR: i32 = 0;
pub const ACTIVATION_SIGMOID: i32 = 1;
pub const ACTIVATION_TANH: i32 = 2;
pub const ACTIVATION_RELU: i32 = 3;
pub const ACTIVATION_SOFTMAX: i32 = 4;
pub const ACTIVATION_SWISH: i32 = 5;
pub const ACTIVATION_EXP: i32 = 6;

// Weight types (matches nnet.h)
pub const WEIGHT_TYPE_FLOAT: i32 = 0;
pub const WEIGHT_TYPE_INT: i32 = 1;
pub const WEIGHT_TYPE_QWEIGHT: i32 = 2;
pub const WEIGHT_TYPE_INT8: i32 = 3;

// Weight blob constants
pub const WEIGHT_BLOB_VERSION: i32 = 0;
pub const WEIGHT_BLOCK_SIZE: usize = 64;
const SPARSE_BLOCK_SIZE: usize = 32;

// Buffer size limits — upper bounds matching C reference.
// Actual sizes are computed dynamically; these only guard debug assertions.
const MAX_ACTIVATIONS: usize = 4096;
#[allow(dead_code)]
const MAX_INPUTS: usize = 2048;

// NNDSP constants (matches nndsp.h)
pub const ADACONV_MAX_KERNEL_SIZE: usize = 32;
pub const ADACONV_MAX_INPUT_CHANNELS: usize = 3;
pub const ADACONV_MAX_OUTPUT_CHANNELS: usize = 3;
pub const ADACONV_MAX_FRAME_SIZE: usize = 240;
pub const ADACONV_MAX_OVERLAP_SIZE: usize = 120;

pub const ADACOMB_MAX_LAG: usize = 300;
pub const ADACOMB_MAX_KERNEL_SIZE: usize = 16;
pub const ADACOMB_MAX_FRAME_SIZE: usize = 80;

pub const ADASHAPE_MAX_INPUT_DIM: usize = 512;
pub const ADASHAPE_MAX_FRAME_SIZE: usize = 240;

const LOG256: f32 = 5.5451774445;

// ===========================================================================
// Tansig lookup table (matches tansig_table.h)
// ===========================================================================

/// Precomputed tanh(x) for x = 0.0, 0.05, 0.10, ..., 10.0 (201 entries).
/// Used by SIMD paths; the scalar fallback uses `tanh_approx` instead.
#[allow(dead_code)]
pub static TANSIG_TABLE: [f32; 201] = [
    0.000000, 0.039979, 0.079830, 0.119427, 0.158649, 0.197375, 0.235496, 0.272905, 0.309507,
    0.345214, 0.379949, 0.413644, 0.446244, 0.477700, 0.507977, 0.537050, 0.564900, 0.591519,
    0.616909, 0.641077, 0.664037, 0.685809, 0.706419, 0.725897, 0.744277, 0.761594, 0.777888,
    0.793199, 0.807569, 0.821040, 0.833655, 0.845456, 0.856485, 0.866784, 0.876393, 0.885352,
    0.893698, 0.901468, 0.908698, 0.915420, 0.921669, 0.927473, 0.932862, 0.937863, 0.942503,
    0.946806, 0.950795, 0.954492, 0.957917, 0.961090, 0.964028, 0.966747, 0.969265, 0.971594,
    0.973749, 0.975743, 0.977587, 0.979293, 0.980869, 0.982327, 0.983675, 0.984921, 0.986072,
    0.987136, 0.988119, 0.989027, 0.989867, 0.990642, 0.991359, 0.992020, 0.992631, 0.993196,
    0.993718, 0.994199, 0.994644, 0.995055, 0.995434, 0.995784, 0.996108, 0.996407, 0.996682,
    0.996937, 0.997172, 0.997389, 0.997590, 0.997775, 0.997946, 0.998104, 0.998249, 0.998384,
    0.998508, 0.998623, 0.998728, 0.998826, 0.998916, 0.999000, 0.999076, 0.999147, 0.999213,
    0.999273, 0.999329, 0.999381, 0.999428, 0.999472, 0.999513, 0.999550, 0.999585, 0.999617,
    0.999646, 0.999673, 0.999699, 0.999722, 0.999743, 0.999763, 0.999781, 0.999798, 0.999813,
    0.999828, 0.999841, 0.999853, 0.999865, 0.999875, 0.999885, 0.999893, 0.999902, 0.999909,
    0.999916, 0.999923, 0.999929, 0.999934, 0.999939, 0.999944, 0.999948, 0.999952, 0.999956,
    0.999959, 0.999962, 0.999965, 0.999968, 0.999970, 0.999973, 0.999975, 0.999977, 0.999978,
    0.999980, 0.999982, 0.999983, 0.999984, 0.999986, 0.999987, 0.999988, 0.999989, 0.999990,
    0.999990, 0.999991, 0.999992, 0.999992, 0.999993, 0.999994, 0.999994, 0.999994, 0.999995,
    0.999995, 0.999996, 0.999996, 0.999996, 0.999997, 0.999997, 0.999997, 0.999997, 0.999997,
    0.999998, 0.999998, 0.999998, 0.999998, 0.999998, 0.999998, 0.999999, 0.999999, 0.999999,
    0.999999, 0.999999, 0.999999, 0.999999, 0.999999, 0.999999, 0.999999, 0.999999, 0.999999,
    0.999999, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
    1.000000, 1.000000, 1.000000,
];

// ===========================================================================
// Types
// ===========================================================================

/// Generic sparse affine transformation layer.
/// Matches C `LinearLayer` from nnet.h.
///
/// Weight data is owned (copied from weight blob during init).
/// The C reference uses zero-copy pointers into the blob, but safe Rust
/// requires either lifetimes or owned data. We choose owned for API simplicity.
#[derive(Clone, Debug, Default)]
pub struct LinearLayer {
    pub bias: Option<Vec<f32>>,
    pub subias: Option<Vec<f32>>,
    pub weights: Option<Vec<i8>>,
    pub float_weights: Option<Vec<f32>>,
    pub weights_idx: Option<Vec<i32>>,
    pub diag: Option<Vec<f32>>,
    pub scale: Option<Vec<f32>>,
    pub nb_inputs: usize,
    pub nb_outputs: usize,
}

/// 2D convolution layer.
/// Matches C `Conv2dLayer` from nnet.h.
#[derive(Clone, Debug, Default)]
pub struct Conv2dLayer {
    pub bias: Option<Vec<f32>>,
    pub float_weights: Option<Vec<f32>>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub ktime: usize,
    pub kheight: usize,
}

/// Named weight array parsed from a binary weight blob.
/// Matches C `WeightArray` from nnet.h.
#[derive(Clone, Debug)]
pub struct WeightArray {
    pub name: String,
    pub weight_type: i32,
    pub size: usize,
    pub data: Vec<u8>,
}

/// Adaptive convolution state. Matches C `AdaConvState`.
#[derive(Clone, Debug)]
pub struct AdaConvState {
    pub history: Vec<f32>,
    pub last_kernel: Vec<f32>,
    pub last_gain: f32,
}

impl AdaConvState {
    pub fn new() -> Self {
        Self {
            history: vec![0.0; ADACONV_MAX_KERNEL_SIZE * ADACONV_MAX_INPUT_CHANNELS],
            last_kernel: vec![
                0.0;
                ADACONV_MAX_KERNEL_SIZE
                    * ADACONV_MAX_INPUT_CHANNELS
                    * ADACONV_MAX_OUTPUT_CHANNELS
            ],
            last_gain: 0.0,
        }
    }
}

/// Adaptive comb filter state. Matches C `AdaCombState`.
#[derive(Clone, Debug)]
pub struct AdaCombState {
    pub history: Vec<f32>,
    pub last_kernel: Vec<f32>,
    pub last_global_gain: f32,
    pub last_pitch_lag: usize,
}

impl AdaCombState {
    pub fn new() -> Self {
        Self {
            history: vec![0.0; ADACOMB_MAX_KERNEL_SIZE + ADACOMB_MAX_LAG],
            last_kernel: vec![0.0; ADACOMB_MAX_KERNEL_SIZE],
            last_global_gain: 0.0,
            last_pitch_lag: 0,
        }
    }
}

/// Adaptive waveform shaping state. Matches C `AdaShapeState`.
#[derive(Clone, Debug)]
pub struct AdaShapeState {
    pub conv_alpha1f_state: Vec<f32>,
    pub conv_alpha1t_state: Vec<f32>,
    pub conv_alpha2_state: Vec<f32>,
    pub interpolate_state: [f32; 1],
}

impl AdaShapeState {
    pub fn new() -> Self {
        Self {
            conv_alpha1f_state: vec![0.0; ADASHAPE_MAX_INPUT_DIM],
            conv_alpha1t_state: vec![0.0; ADASHAPE_MAX_INPUT_DIM],
            conv_alpha2_state: vec![0.0; ADASHAPE_MAX_FRAME_SIZE],
            interpolate_state: [0.0],
        }
    }
}

// ===========================================================================
// C math compatibility helpers (private)
// ===========================================================================

/// Match C `exp()` — double precision with float argument/result.
#[inline(always)]
fn exp_dp(x: f32) -> f32 {
    (x as f64).exp() as f32
}

/// Match C `cos()` — double precision.
#[inline(always)]
fn cos_dp(x: f64) -> f64 {
    x.cos()
}

/// Match C `sqrt()` — double precision with float argument/result.
#[inline(always)]
fn sqrt_dp(x: f32) -> f32 {
    (x as f64).sqrt() as f32
}

/// Match C `(float)log(x)` — double precision log, result truncated to float.
/// Used for `celt_log` in the float API path.
#[inline(always)]
fn celt_log_f(x: f32) -> f32 {
    (x as f64).ln() as f32
}

// ===========================================================================
// Approximation functions (matches vec.h, common.h)
// ===========================================================================

/// Rational Padé approximation of tanh(x).
/// Matches C `tanh_approx` from vec.h. Clamped to [-1, 1].
/// Max error ~1e-5 in [-8, 8].
#[inline]
pub fn tanh_approx(x: f32) -> f32 {
    const N0: f32 = 952.52801514;
    const N1: f32 = 96.39235687;
    const N2: f32 = 0.60863042;
    const D0: f32 = 952.72399902;
    const D1: f32 = 413.36801147;
    const D2: f32 = 11.88600922;

    let x2 = x * x;
    // fmadd(a,b,c) = a*b+c in C — standard float multiply-add
    let num = (N2 * x2 + N1) * x2 + N0;
    let den = (D2 * x2 + D1) * x2 + D0;
    let result = num * x / den;
    // Clamp to [-1, 1]
    if result > 1.0 {
        1.0
    } else if result < -1.0 {
        -1.0
    } else {
        result
    }
}

/// Sigmoid approximation derived from tanh.
/// Matches C `sigmoid_approx` from vec.h.
#[inline]
pub fn sigmoid_approx(x: f32) -> f32 {
    0.5 + 0.5 * tanh_approx(0.5 * x)
}

/// Fast 2^x approximation using IEEE 754 bit manipulation.
/// Matches C `lpcnet_exp2` from vec.h.
/// Returns 0.0 for x < -50.
#[inline]
pub fn lpcnet_exp2(x: f32) -> f32 {
    let integer = x.floor() as i32;
    if integer < -50 {
        return 0.0;
    }
    let frac = x - integer as f32;
    // Polynomial: K0 + K1*frac + K2*frac^2 + K3*frac^3
    let res = 0.99992522_f32 + frac * (0.69583354 + frac * (0.22606716 + 0.078024523 * frac));
    let mut bits = res.to_bits();
    // Add integer to IEEE 754 exponent field, clear sign bit
    bits = bits.wrapping_add((integer as u32).wrapping_shl(23)) & 0x7FFFFFFF;
    f32::from_bits(bits)
}

/// Fast exp(x) approximation: exp(x) = 2^(x * log2(e)).
/// Matches C `lpcnet_exp` macro from vec.h.
#[inline]
pub fn lpcnet_exp(x: f32) -> f32 {
    lpcnet_exp2(x * 1.44269504)
}

/// Fast log2(x) approximation using IEEE 754 bit extraction.
/// Matches C `log2_approx` from common.h.
#[inline]
pub fn log2_approx(x: f32) -> f32 {
    let bits = x.to_bits() as i32;
    let integer = (bits >> 23) - 127;
    // Remove exponent, leaving mantissa in [1.0, 2.0)
    let mantissa_bits = (bits - (integer << 23)) as u32;
    let mantissa = f32::from_bits(mantissa_bits);
    let frac = mantissa - 1.5;
    let frac = -0.41445418 + frac * (0.95909232 + frac * (-0.33951290 + frac * 0.16541097));
    1.0 + integer as f32 + frac
}

/// Fast ln(x) approximation. Matches C `log_approx` macro from common.h.
#[inline]
pub fn log_approx(x: f32) -> f32 {
    0.69315 * log2_approx(x)
}

/// Convert mu-law encoded value to linear PCM.
/// Matches C `ulaw2lin` from common.h.
pub fn ulaw2lin(u: f32) -> f32 {
    let scale_1: f32 = 32768.0 / 255.0;
    let u = u - 128.0;
    let s = if u >= 0.0 { 1.0f32 } else { -1.0f32 };
    let u = u.abs();
    // C uses double-precision exp: exp((double)(u/128.) * LOG256)
    // u/128. promotes to double (128. is double literal in C)
    let arg = (u as f64 / 128.0) * LOG256 as f64;
    s * scale_1 * (arg.exp() as f32 - 1.0)
}

/// Convert linear PCM to mu-law encoded value.
/// Matches C `lin2ulaw` from common.h.
pub fn lin2ulaw(x: f32) -> i32 {
    let scale: f32 = 255.0 / 32768.0;
    let s = if x >= 0.0 { 1 } else { -1 };
    let x = x.abs();
    // All float arithmetic (log_approx returns float)
    let u = (s as f32) * (128.0 * log_approx(1.0 + scale * x) / LOG256);
    let u = 128.0 + u;
    let u = if u < 0.0 { 0.0 } else { u };
    let u = if u > 255.0 { 255.0 } else { u };
    (0.5 + u).floor() as i32
}

// ===========================================================================
// Vector operations (private, in-place)
// ===========================================================================

/// Element-wise tanh approximation in-place.
fn vec_tanh(data: &mut [f32], n: usize) {
    for i in 0..n {
        data[i] = tanh_approx(data[i]);
    }
}

/// Element-wise sigmoid approximation in-place.
fn vec_sigmoid(data: &mut [f32], n: usize) {
    for i in 0..n {
        data[i] = sigmoid_approx(data[i]);
    }
}

/// Element-wise exp via lpcnet_exp, in-place.
/// Named `softmax` in C (vec.h) but it's just element-wise exp, not normalized.
fn softmax_exp(data: &mut [f32], n: usize) {
    for i in 0..n {
        data[i] = lpcnet_exp(data[i]);
    }
}

/// Swish activation: y = x * sigmoid(x). Uses temporary buffer.
fn vec_swish(data: &mut [f32], n: usize) {
    debug_assert!(n <= MAX_ACTIVATIONS);
    let mut tmp = vec![0.0f32; n];
    for i in 0..n {
        tmp[i] = sigmoid_approx(data[i]);
    }
    for i in 0..n {
        data[i] *= tmp[i];
    }
}

// ===========================================================================
// GEMV primitives (private, matches vec.h)
// ===========================================================================

/// Dense float SGEMV. Column-major weights: W(i,j) = weights[j*col_stride + i].
/// Matches C `sgemv` from vec.h (scalar fallback path).
fn sgemv(out: &mut [f32], weights: &[f32], rows: usize, cols: usize, col_stride: usize, x: &[f32]) {
    for i in 0..rows {
        out[i] = 0.0;
        for j in 0..cols {
            out[i] += weights[j * col_stride + i] * x[j];
        }
    }
}

/// Sparse float SGEMV with 8×4 block structure.
/// Matches C `sparse_sgemv8x4` from vec.h.
///
/// Weight block layout (input-major): for output k ∈ [0,8), input s ∈ [0,4):
///   w[s*8 + k] is the weight for (output k, input s).
///
/// Index format: for each group of 8 output rows:
///   [count, pos0, pos1, ...] where pos is the starting column (4-aligned).
fn sparse_sgemv8x4(out: &mut [f32], weights: &[f32], idx: &[i32], rows: usize, x: &[f32]) {
    for v in out[..rows].iter_mut() {
        *v = 0.0;
    }
    let mut w_pos: usize = 0;
    let mut idx_pos: usize = 0;
    let mut i: usize = 0;
    while i < rows {
        let col_blocks = idx[idx_pos] as usize;
        idx_pos += 1;
        for _ in 0..col_blocks {
            let pos = idx[idx_pos] as usize;
            idx_pos += 1;
            let xj0 = x[pos];
            let xj1 = x[pos + 1];
            let xj2 = x[pos + 2];
            let xj3 = x[pos + 3];
            // Separate += per input group to match C accumulation order
            for k in 0..8 {
                out[i + k] += weights[w_pos + k] * xj0;
            }
            for k in 0..8 {
                out[i + k] += weights[w_pos + 8 + k] * xj1;
            }
            for k in 0..8 {
                out[i + k] += weights[w_pos + 16 + k] * xj2;
            }
            for k in 0..8 {
                out[i + k] += weights[w_pos + 24 + k] * xj3;
            }
            w_pos += 32;
        }
        i += 8;
    }
}

/// Dense int8 quantized GEMV with 8×4 blocking.
/// Matches C `cgemv8x4` from vec.h (non-USE_SU_BIAS path).
///
/// Weight block layout (output-major): for output k ∈ [0,8), input s ∈ [0,4):
///   w[k*4 + s] is the weight for (output k, input s).
fn cgemv8x4(out: &mut [f32], w: &[i8], scale: &[f32], rows: usize, cols: usize, x: &[f32]) {
    // Quantize input to signed int8: round(127 * x)
    // C: x[i] = (int)floor(.5+127*_x[i]) — 0.5 is double, so promotes to double
    let mut qx = vec![0i8; cols];
    for i in 0..cols {
        qx[i] = (0.5_f64 + 127.0 * x[i] as f64).floor() as i32 as i8;
    }
    for v in out[..rows].iter_mut() {
        *v = 0.0;
    }
    let mut w_pos: usize = 0;
    let mut i: usize = 0;
    while i < rows {
        let mut j: usize = 0;
        while j < cols {
            let xj0 = qx[j] as f32;
            let xj1 = qx[j + 1] as f32;
            let xj2 = qx[j + 2] as f32;
            let xj3 = qx[j + 3] as f32;
            // Parenthesized sum matches C: y[k] += (w[k*4]*xj0 + w[k*4+1]*xj1 + ...)
            for k in 0..8 {
                out[i + k] += w[w_pos + k * 4] as f32 * xj0
                    + w[w_pos + k * 4 + 1] as f32 * xj1
                    + w[w_pos + k * 4 + 2] as f32 * xj2
                    + w[w_pos + k * 4 + 3] as f32 * xj3;
            }
            w_pos += 32;
            j += 4;
        }
        i += 8;
    }
    // Dequantize: per-output scale factor
    for i in 0..rows {
        out[i] *= scale[i];
    }
}

/// Sparse int8 quantized GEMV with 8×4 block structure.
/// Matches C `sparse_cgemv8x4` from vec.h (non-USE_SU_BIAS path).
fn sparse_cgemv8x4(
    out: &mut [f32],
    w: &[i8],
    idx: &[i32],
    scale: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
) {
    let mut qx = vec![0i8; cols];
    for i in 0..cols {
        qx[i] = (0.5_f64 + 127.0 * x[i] as f64).floor() as i32 as i8;
    }
    for v in out[..rows].iter_mut() {
        *v = 0.0;
    }
    let mut w_pos: usize = 0;
    let mut idx_pos: usize = 0;
    let mut i: usize = 0;
    while i < rows {
        let col_blocks = idx[idx_pos] as usize;
        idx_pos += 1;
        for _ in 0..col_blocks {
            let pos = idx[idx_pos] as usize;
            idx_pos += 1;
            let xj0 = qx[pos] as i32;
            let xj1 = qx[pos + 1] as i32;
            let xj2 = qx[pos + 2] as i32;
            let xj3 = qx[pos + 3] as i32;
            for k in 0..8 {
                out[i + k] += (w[w_pos + k * 4] as i32 * xj0
                    + w[w_pos + k * 4 + 1] as i32 * xj1
                    + w[w_pos + k * 4 + 2] as i32 * xj2
                    + w[w_pos + k * 4 + 3] as i32 * xj3) as f32;
            }
            w_pos += 32;
        }
        i += 8;
    }
    for i in 0..rows {
        out[i] *= scale[i];
    }
}

// ===========================================================================
// Core compute functions
// ===========================================================================

/// Linear transform: out = W * in + bias [+ diag].
/// Matches C `compute_linear_c` from nnet_arch.h.
///
/// Dispatches to the appropriate GEMV based on weight storage format.
/// `out` and `input` must not alias.
pub fn compute_linear(linear: &LinearLayer, out: &mut [f32], input: &[f32]) {
    let m = linear.nb_inputs;
    let n = linear.nb_outputs;

    if let Some(ref fw) = linear.float_weights {
        if let Some(ref idx) = linear.weights_idx {
            sparse_sgemv8x4(out, fw, idx, n, input);
        } else {
            sgemv(out, fw, n, m, n, input);
        }
    } else if let Some(ref w) = linear.weights {
        let scale = linear.scale.as_ref().expect("int8 weights require scale");
        if let Some(ref idx) = linear.weights_idx {
            sparse_cgemv8x4(out, w, idx, scale, n, m, input);
        } else {
            cgemv8x4(out, w, scale, n, m, input);
        }
    } else {
        // No weights — output zeros
        for v in out[..n].iter_mut() {
            *v = 0.0;
        }
    }

    // Add bias
    if let Some(ref bias) = linear.bias {
        for i in 0..n {
            out[i] += bias[i];
        }
    }

    // Add diagonal (GRU recurrent weights only)
    if let Some(ref diag) = linear.diag {
        debug_assert_eq!(3 * m, n, "diag requires 3*nb_inputs == nb_outputs");
        for i in 0..m {
            out[i] += diag[i] * input[i];
            out[i + m] += diag[i + m] * input[i];
            out[i + 2 * m] += diag[i + 2 * m] * input[i];
        }
    }
}

/// Apply activation function in-place on the first `n` elements of `output`.
/// Matches C `compute_activation_c` from nnet_arch.h.
///
/// SOFTMAX_HACK is active (production Opus behavior): ACTIVATION_SOFTMAX
/// is treated as identity.
pub fn compute_activation(output: &mut [f32], n: usize, activation: i32) {
    match activation {
        ACTIVATION_SIGMOID => vec_sigmoid(output, n),
        ACTIVATION_TANH => vec_tanh(output, n),
        ACTIVATION_RELU => {
            for i in 0..n {
                if output[i] < 0.0 {
                    output[i] = 0.0;
                }
            }
        }
        ACTIVATION_SOFTMAX => {
            // SOFTMAX_HACK: identity pass-through (production Opus behavior).
            // The normalization is handled downstream.
        }
        ACTIVATION_SWISH => vec_swish(output, n),
        ACTIVATION_EXP => softmax_exp(output, n),
        _ => {
            // ACTIVATION_LINEAR: no-op (already in-place)
            debug_assert_eq!(activation, ACTIVATION_LINEAR);
        }
    }
}

/// Dense layer: output = activation(W * input + bias).
/// Matches C `compute_generic_dense` from nnet.c.
pub fn compute_generic_dense(
    layer: &LinearLayer,
    output: &mut [f32],
    input: &[f32],
    activation: i32,
) {
    compute_linear(layer, output, input);
    compute_activation(output, layer.nb_outputs, activation);
}

/// GRU recurrent cell. Updates `state` in-place.
/// Matches C `compute_generic_gru` from nnet.c.
///
/// `input_weights`: [M] → [3N], `recurrent_weights`: [N] → [3N].
/// `state` has length N, `input` has length M.
pub fn compute_generic_gru(
    input_weights: &LinearLayer,
    recurrent_weights: &LinearLayer,
    state: &mut [f32],
    input: &[f32],
) {
    let n = recurrent_weights.nb_inputs;
    let three_n = recurrent_weights.nb_outputs;
    debug_assert_eq!(3 * n, three_n);
    debug_assert_eq!(input_weights.nb_outputs, three_n);

    let mut zrh = vec![0.0f32; three_n];
    let mut recur = vec![0.0f32; three_n];

    // Input and recurrent linear transforms
    compute_linear(input_weights, &mut zrh, input);
    compute_linear(recurrent_weights, &mut recur, state);

    // Add recurrent contribution to z and r gates
    for i in 0..2 * n {
        zrh[i] += recur[i];
    }

    // Sigmoid on z and r gates
    compute_activation(&mut zrh, 2 * n, ACTIVATION_SIGMOID);

    // Gated recurrent contribution to candidate h:
    // h[i] += recur_h[i] * r[i]
    for i in 0..n {
        zrh[2 * n + i] += recur[2 * n + i] * zrh[n + i];
    }

    // Tanh on candidate h
    compute_activation(&mut zrh[2 * n..], n, ACTIVATION_TANH);

    // Final state update: state = z * old_state + (1-z) * h
    for i in 0..n {
        let z_i = zrh[i];
        let h_i = zrh[2 * n + i];
        zrh[2 * n + i] = z_i * state[i] + (1.0 - z_i) * h_i;
    }
    state[..n].copy_from_slice(&zrh[2 * n..3 * n]);
}

/// Gated Linear Unit: output = input * sigmoid(W * input + bias).
/// Matches C `compute_glu` from nnet.c (output != input case).
pub fn compute_glu(layer: &LinearLayer, output: &mut [f32], input: &[f32]) {
    let n = layer.nb_outputs;
    debug_assert_eq!(layer.nb_inputs, n);
    let mut act2 = vec![0.0f32; n];
    compute_linear(layer, &mut act2, input);
    compute_activation(&mut act2, n, ACTIVATION_SIGMOID);
    for i in 0..n {
        output[i] = input[i] * act2[i];
    }
}

/// In-place GLU: data = data * sigmoid(W * data + bias).
/// Matches C `compute_glu` from nnet.c (output == input case).
pub fn compute_glu_inplace(layer: &LinearLayer, data: &mut [f32]) {
    let n = layer.nb_outputs;
    debug_assert_eq!(layer.nb_inputs, n);
    let mut act2 = vec![0.0f32; n];
    compute_linear(layer, &mut act2, &data[..n]);
    compute_activation(&mut act2, n, ACTIVATION_SIGMOID);
    for i in 0..n {
        data[i] *= act2[i];
    }
}

/// Gated activation: output = input * activation(W * input + bias).
/// Like GLU but with caller-specified activation instead of hardcoded sigmoid.
/// Matches C `compute_gated_activation` declaration in nnet.h.
pub fn compute_gated_activation(
    layer: &LinearLayer,
    output: &mut [f32],
    input: &[f32],
    activation: i32,
) {
    let n = layer.nb_outputs;
    debug_assert_eq!(layer.nb_inputs, n);
    let mut act2 = vec![0.0f32; n];
    compute_linear(layer, &mut act2, input);
    compute_activation(&mut act2, n, activation);
    for i in 0..n {
        output[i] = input[i] * act2[i];
    }
}

/// In-place gated activation.
pub fn compute_gated_activation_inplace(layer: &LinearLayer, data: &mut [f32], activation: i32) {
    let n = layer.nb_outputs;
    debug_assert_eq!(layer.nb_inputs, n);
    let mut act2 = vec![0.0f32; n];
    compute_linear(layer, &mut act2, &data[..n]);
    compute_activation(&mut act2, n, activation);
    for i in 0..n {
        data[i] *= act2[i];
    }
}

/// Causal 1D convolution.
/// Matches C `compute_generic_conv1d` from nnet.c.
///
/// The kernel is flattened into `layer.nb_inputs` = kernel_size * input_size.
/// `mem` holds (kernel_size - 1) previous frames, size = nb_inputs - input_size.
pub fn compute_generic_conv1d(
    layer: &LinearLayer,
    output: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    input_size: usize,
    activation: i32,
) {
    let mem_size = layer.nb_inputs - input_size;
    let mut tmp = vec![0.0f32; layer.nb_inputs];

    // Concatenate history and current input
    if layer.nb_inputs != input_size {
        tmp[..mem_size].copy_from_slice(&mem[..mem_size]);
    }
    tmp[mem_size..].copy_from_slice(&input[..input_size]);

    compute_linear(layer, output, &tmp);
    compute_activation(output, layer.nb_outputs, activation);

    // Update memory: shift out oldest frame, shift in current
    if layer.nb_inputs != input_size {
        mem[..mem_size].copy_from_slice(&tmp[input_size..layer.nb_inputs]);
    }
}

/// Dilated causal 1D convolution.
/// Matches C `compute_generic_conv1d_dilation` from nnet.c.
///
/// For dilation > 1, samples are gathered from `mem` at stride `dilation * input_size`.
pub fn compute_generic_conv1d_dilation(
    layer: &LinearLayer,
    output: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    input_size: usize,
    dilation: usize,
    activation: i32,
) {
    let ksize = layer.nb_inputs / input_size;
    let mut tmp = vec![0.0f32; layer.nb_inputs];

    if dilation == 1 {
        let mem_size = layer.nb_inputs - input_size;
        tmp[..mem_size].copy_from_slice(&mem[..mem_size]);
    } else {
        // Gather history taps at dilated stride
        for i in 0..ksize - 1 {
            let dst = i * input_size;
            let src = i * input_size * dilation;
            tmp[dst..dst + input_size].copy_from_slice(&mem[src..src + input_size]);
        }
    }
    // Current input is the last tap
    tmp[layer.nb_inputs - input_size..].copy_from_slice(&input[..input_size]);

    compute_linear(layer, output, &tmp);
    compute_activation(output, layer.nb_outputs, activation);

    if dilation == 1 {
        let mem_size = layer.nb_inputs - input_size;
        mem[..mem_size].copy_from_slice(&tmp[input_size..layer.nb_inputs]);
    } else {
        // Shift mem left by input_size, append current input at end
        let total_mem = input_size * dilation * (ksize - 1);
        mem.copy_within(input_size..total_mem, 0);
        mem[total_mem - input_size..total_mem].copy_from_slice(&input[..input_size]);
    }
}

// ===========================================================================
// Conv2D (matches nnet_arch.h)
// ===========================================================================

/// General 2D convolution kernel.
/// Weight layout: [out_channels][in_channels][ktime][kheight]
/// Input layout:  [ktime][in_channels][height + kheight - 1]
fn conv2d_float(
    out: &mut [f32],
    weights: &[f32],
    in_channels: usize,
    out_channels: usize,
    ktime: usize,
    kheight: usize,
    input: &[f32],
    height: usize,
    hstride: usize,
) {
    let in_stride = height + kheight - 1;
    for i in 0..out_channels {
        for j in 0..height {
            out[i * hstride + j] = 0.0;
        }
        for m in 0..in_channels {
            for t in 0..ktime {
                for h in 0..kheight {
                    for j in 0..height {
                        out[i * hstride + j] += weights[i * in_channels * ktime * kheight
                            + m * ktime * kheight
                            + t * kheight
                            + h]
                            * input[t * in_channels * in_stride + m * in_stride + j + h];
                    }
                }
            }
        }
    }
}

/// Specialized 3×3 convolution with fully unrolled inner loops.
/// Matches C `conv2d_3x3_float` from nnet_arch.h.
fn conv2d_3x3_float(
    out: &mut [f32],
    weights: &[f32],
    in_channels: usize,
    out_channels: usize,
    input: &[f32],
    height: usize,
    hstride: usize,
) {
    let kheight = 3;
    let ktime = 3;
    let in_stride = height + kheight - 1;
    for i in 0..out_channels {
        for j in 0..height {
            out[i * hstride + j] = 0.0;
        }
        for m in 0..in_channels {
            let wb = i * in_channels * ktime * kheight + m * ktime * kheight;
            for j in 0..height {
                out[i * hstride + j] += weights[wb + 0 * kheight + 0]
                    * input[0 * in_channels * in_stride + m * in_stride + j + 0]
                    + weights[wb + 0 * kheight + 1]
                        * input[0 * in_channels * in_stride + m * in_stride + j + 1]
                    + weights[wb + 0 * kheight + 2]
                        * input[0 * in_channels * in_stride + m * in_stride + j + 2]
                    + weights[wb + 1 * kheight + 0]
                        * input[1 * in_channels * in_stride + m * in_stride + j + 0]
                    + weights[wb + 1 * kheight + 1]
                        * input[1 * in_channels * in_stride + m * in_stride + j + 1]
                    + weights[wb + 1 * kheight + 2]
                        * input[1 * in_channels * in_stride + m * in_stride + j + 2]
                    + weights[wb + 2 * kheight + 0]
                        * input[2 * in_channels * in_stride + m * in_stride + j + 0]
                    + weights[wb + 2 * kheight + 1]
                        * input[2 * in_channels * in_stride + m * in_stride + j + 1]
                    + weights[wb + 2 * kheight + 2]
                        * input[2 * in_channels * in_stride + m * in_stride + j + 2];
            }
        }
    }
}

/// 2D convolution processing one time frame. Maintains temporal history in `mem`.
/// Matches C `compute_conv2d_c` from nnet_arch.h.
pub fn compute_conv2d(
    conv: &Conv2dLayer,
    out: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    height: usize,
    hstride: usize,
    activation: i32,
) {
    let time_stride = conv.in_channels * (height + conv.kheight - 1);
    let in_buf_size = conv.ktime * time_stride;
    let mut in_buf = vec![0.0f32; in_buf_size];

    // Build input buffer: [mem (ktime-1 frames) | current input (1 frame)]
    let mem_frames = (conv.ktime - 1) * time_stride;
    in_buf[..mem_frames].copy_from_slice(&mem[..mem_frames]);
    in_buf[mem_frames..mem_frames + time_stride].copy_from_slice(&input[..time_stride]);

    // Update mem for next call: drop first frame, keep the rest
    mem[..mem_frames].copy_from_slice(&in_buf[time_stride..in_buf_size]);

    let weights = conv
        .float_weights
        .as_ref()
        .expect("conv2d requires float weights");
    if conv.kheight == 3 && conv.ktime == 3 {
        conv2d_3x3_float(
            out,
            weights,
            conv.in_channels,
            conv.out_channels,
            &in_buf,
            height,
            hstride,
        );
    } else {
        conv2d_float(
            out,
            weights,
            conv.in_channels,
            conv.out_channels,
            conv.ktime,
            conv.kheight,
            &in_buf,
            height,
            hstride,
        );
    }

    // Add per-channel bias
    if let Some(ref bias) = conv.bias {
        for i in 0..conv.out_channels {
            for j in 0..height {
                out[i * hstride + j] += bias[i];
            }
        }
    }

    // Per-channel activation
    for i in 0..conv.out_channels {
        let start = i * hstride;
        compute_activation(&mut out[start..], height, activation);
    }
}

// ===========================================================================
// Weight parsing (matches parse_lpcnet_weights.c)
// ===========================================================================

/// Read a native-endian i32 from a byte slice at the given offset.
fn read_i32_ne(data: &[u8], offset: usize) -> i32 {
    i32::from_ne_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

fn bytes_to_f32_vec(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bytes_to_i8_vec(data: &[u8]) -> Vec<i8> {
    data.iter().map(|&b| b as i8).collect()
}

fn bytes_to_i32_vec(data: &[u8]) -> Vec<i32> {
    data.chunks_exact(4)
        .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Parse one weight record from the blob. Advances `offset`.
/// Matches C `parse_record` from parse_lpcnet_weights.c.
fn parse_record(data: &[u8], offset: &mut usize) -> Result<WeightArray, i32> {
    let remaining = data.len() - *offset;
    if remaining < WEIGHT_BLOCK_SIZE {
        return Err(-1);
    }
    let base = *offset;
    let _version = read_i32_ne(data, base + 4);
    let weight_type = read_i32_ne(data, base + 8);
    let size = read_i32_ne(data, base + 12);
    let block_size = read_i32_ne(data, base + 16);

    if block_size < size {
        return Err(-1);
    }
    if block_size as usize > remaining - WEIGHT_BLOCK_SIZE {
        return Err(-1);
    }
    // Name must be null-terminated within the 44-byte field
    if data[base + 63] != 0 {
        return Err(-1);
    }
    if size < 0 {
        return Err(-1);
    }

    // Parse null-terminated name from bytes 20..64
    let name_bytes = &data[base + 20..base + 64];
    let nul_pos = name_bytes.iter().position(|&b| b == 0).unwrap_or(44);
    let name = String::from_utf8_lossy(&name_bytes[..nul_pos]).into_owned();

    let data_start = base + WEIGHT_BLOCK_SIZE;
    let data_end = data_start + size as usize;
    let array_data = data[data_start..data_end].to_vec();

    *offset = base + WEIGHT_BLOCK_SIZE + block_size as usize;

    Ok(WeightArray {
        name,
        weight_type,
        size: size as usize,
        data: array_data,
    })
}

/// Parse a binary weight blob into an array of named weight arrays.
/// Matches C `parse_weights` from parse_lpcnet_weights.c.
/// Returns the array on success, or Err(-1) on parse failure.
pub fn parse_weights(data: &[u8]) -> Result<Vec<WeightArray>, i32> {
    let mut list = Vec::with_capacity(20);
    let mut offset = 0;
    while offset < data.len() {
        let array = parse_record(data, &mut offset)?;
        list.push(array);
    }
    Ok(list)
}

/// Find a named array with exact size match.
fn find_array_check<'a>(
    arrays: &'a [WeightArray],
    name: &str,
    expected_size: usize,
) -> Option<&'a WeightArray> {
    arrays
        .iter()
        .find(|a| a.name == name && a.size == expected_size)
}

/// Find a named array — returns None if name absent, Err if name present but size wrong.
fn opt_array_check<'a>(
    arrays: &'a [WeightArray],
    name: &str,
    expected_size: usize,
) -> Result<Option<&'a WeightArray>, ()> {
    match arrays.iter().find(|a| a.name == name) {
        Some(a) if a.size == expected_size => Ok(Some(a)),
        Some(_) => Err(()), // Size mismatch
        None => Ok(None),
    }
}

/// Validate a sparse index structure. Returns total number of 8×4 blocks.
/// Matches C `find_idx_check` from parse_lpcnet_weights.c.
fn find_idx_check(
    arrays: &[WeightArray],
    name: &str,
    nb_in: usize,
    nb_out: usize,
) -> Result<(Vec<i32>, usize), ()> {
    let arr = arrays.iter().find(|a| a.name == name).ok_or(())?;
    let idx = bytes_to_i32_vec(&arr.data);
    let mut total_blocks: usize = 0;
    let mut remaining_outs = nb_out as i32;
    let mut pos: usize = 0;

    while pos < idx.len() {
        let nb_blocks = idx[pos] as usize;
        pos += 1;
        if pos + nb_blocks > idx.len() {
            return Err(());
        }
        for _ in 0..nb_blocks {
            let col_pos = idx[pos] as usize;
            pos += 1;
            if col_pos + 3 >= nb_in || (col_pos & 0x3) != 0 {
                return Err(());
            }
        }
        remaining_outs -= 8;
        total_blocks += nb_blocks;
    }
    if remaining_outs != 0 {
        return Err(());
    }
    Ok((idx, total_blocks))
}

/// Initialize a LinearLayer from named weight arrays.
/// Matches C `linear_init` from parse_lpcnet_weights.c.
/// Returns Ok(layer) on success, Err(()) if a required array is missing or wrong size.
pub fn linear_init(
    arrays: &[WeightArray],
    bias_name: Option<&str>,
    subias_name: Option<&str>,
    weights_name: Option<&str>,
    float_weights_name: Option<&str>,
    weights_idx_name: Option<&str>,
    diag_name: Option<&str>,
    scale_name: Option<&str>,
    nb_inputs: usize,
    nb_outputs: usize,
) -> Result<LinearLayer, ()> {
    let mut layer = LinearLayer {
        nb_inputs,
        nb_outputs,
        ..Default::default()
    };

    if let Some(name) = bias_name {
        let arr = find_array_check(arrays, name, nb_outputs * 4).ok_or(())?;
        layer.bias = Some(bytes_to_f32_vec(&arr.data));
    }
    if let Some(name) = subias_name {
        let arr = find_array_check(arrays, name, nb_outputs * 4).ok_or(())?;
        layer.subias = Some(bytes_to_f32_vec(&arr.data));
    }

    if let Some(idx_name) = weights_idx_name {
        let (idx_data, total_blocks) = find_idx_check(arrays, idx_name, nb_inputs, nb_outputs)?;
        layer.weights_idx = Some(idx_data);
        if let Some(name) = weights_name {
            let arr = find_array_check(arrays, name, SPARSE_BLOCK_SIZE * total_blocks).ok_or(())?;
            layer.weights = Some(bytes_to_i8_vec(&arr.data));
        }
        if let Some(name) = float_weights_name {
            if let Some(arr) = opt_array_check(arrays, name, SPARSE_BLOCK_SIZE * total_blocks * 4)?
            {
                layer.float_weights = Some(bytes_to_f32_vec(&arr.data));
            }
        }
    } else {
        if let Some(name) = weights_name {
            let arr = find_array_check(arrays, name, nb_inputs * nb_outputs).ok_or(())?;
            layer.weights = Some(bytes_to_i8_vec(&arr.data));
        }
        if let Some(name) = float_weights_name {
            if let Some(arr) = opt_array_check(arrays, name, nb_inputs * nb_outputs * 4)? {
                layer.float_weights = Some(bytes_to_f32_vec(&arr.data));
            }
        }
    }

    if let Some(name) = diag_name {
        let arr = find_array_check(arrays, name, nb_outputs * 4).ok_or(())?;
        layer.diag = Some(bytes_to_f32_vec(&arr.data));
    }
    if weights_name.is_some() {
        let sname = scale_name.ok_or(())?;
        let arr = find_array_check(arrays, sname, nb_outputs * 4).ok_or(())?;
        layer.scale = Some(bytes_to_f32_vec(&arr.data));
    }

    Ok(layer)
}

/// Initialize a Conv2dLayer from named weight arrays.
/// Matches C `conv2d_init` from parse_lpcnet_weights.c.
pub fn conv2d_init(
    arrays: &[WeightArray],
    bias_name: Option<&str>,
    float_weights_name: Option<&str>,
    in_channels: usize,
    out_channels: usize,
    ktime: usize,
    kheight: usize,
) -> Result<Conv2dLayer, ()> {
    let mut layer = Conv2dLayer {
        in_channels,
        out_channels,
        ktime,
        kheight,
        ..Default::default()
    };

    if let Some(name) = bias_name {
        let arr = find_array_check(arrays, name, out_channels * 4).ok_or(())?;
        layer.bias = Some(bytes_to_f32_vec(&arr.data));
    }
    if let Some(name) = float_weights_name {
        let expected = in_channels * out_channels * ktime * kheight * 4;
        if let Some(arr) = opt_array_check(arrays, name, expected)? {
            layer.float_weights = Some(bytes_to_f32_vec(&arr.data));
        }
    }

    Ok(layer)
}

// ===========================================================================
// NNDSP helpers (matches nndsp.c)
// ===========================================================================

/// Cross-correlation: xcorr[i] = sum_{j=0}^{len-1} x[j] * y[i+j].
/// Matches the float-API C `celt_pitch_xcorr_c` from pitch.c.
///
/// This is a local implementation used by AdaConv/AdaComb until the CELT
/// pitch module is ported.
fn pitch_xcorr(x: &[f32], y: &[f32], xcorr: &mut [f32], len: usize, max_pitch: usize) {
    for i in 0..max_pitch {
        let mut sum = 0.0f32;
        for j in 0..len {
            sum += x[j] * y[i + j];
        }
        xcorr[i] = sum;
    }
}

/// Compute raised-cosine overlap window: w[n] = 0.5 + 0.5 * cos(π(n+0.5)/N).
/// Matches C `compute_overlap_window` from nndsp.c.
pub fn compute_overlap_window(window: &mut [f32], overlap_size: usize) {
    for i in 0..overlap_size {
        // M_PI is defined as 3.141592653589793f (float) in C nndsp.c.
        // The float multiplication then promotes to double for cos().
        let arg = PI * (i as f32 + 0.5) / overlap_size as f32;
        let cos_val = cos_dp(arg as f64);
        window[i] = (0.5_f64 + 0.5_f64 * cos_val) as f32;
    }
}

/// Normalize kernel per output channel: kernel *= gain / ||kernel||_2.
/// Matches C `scale_kernel` from nndsp.c.
fn scale_kernel(
    kernel: &mut [f32],
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    gain: &[f32],
) {
    for i_out in 0..out_channels {
        let mut norm = 0.0f32;
        for i_in in 0..in_channels {
            for i_k in 0..kernel_size {
                let idx = (i_out * in_channels + i_in) * kernel_size + i_k;
                norm += kernel[idx] * kernel[idx];
            }
        }
        // C: norm = 1.f / (1e-6f + sqrt(norm))
        // sqrt promotes to double in C
        norm = 1.0 / (1e-6_f32 + sqrt_dp(norm));
        for i_in in 0..in_channels {
            for i_k in 0..kernel_size {
                let idx = (i_out * in_channels + i_in) * kernel_size + i_k;
                kernel[idx] *= norm * gain[i_out];
            }
        }
    }
}

/// Transform raw gains: gain = exp(a * gain + b).
/// Matches C `transform_gains` from nndsp.c.
fn transform_gains(gains: &mut [f32], num_gains: usize, filter_gain_a: f32, filter_gain_b: f32) {
    for i in 0..num_gains {
        // C exp() promotes float arg to double
        gains[i] = exp_dp(filter_gain_a * gains[i] + filter_gain_b);
    }
}

// ===========================================================================
// NNDSP — Adaptive convolution (matches nndsp.c)
// ===========================================================================

/// Adaptive convolution: feature-conditioned FIR filter with overlap-add crossfading.
/// Matches C `adaconv_process_frame` from nndsp.c.
#[allow(clippy::too_many_arguments)]
pub fn adaconv_process_frame(
    state: &mut AdaConvState,
    x_out: &mut [f32],
    x_in: &[f32],
    features: &[f32],
    kernel_layer: &LinearLayer,
    gain_layer: &LinearLayer,
    _feature_dim: usize,
    frame_size: usize,
    overlap_size: usize,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    left_padding: usize,
    filter_gain_a: f32,
    filter_gain_b: f32,
    _shape_gain: f32,
    window: &[f32],
) {
    debug_assert_eq!(left_padding, kernel_size - 1);
    debug_assert!(kernel_size < frame_size);

    let mut output_buffer = vec![0.0f32; ADACONV_MAX_FRAME_SIZE * ADACONV_MAX_OUTPUT_CHANNELS];
    let total_kernel = kernel_size * in_channels * out_channels;
    let mut kernel_buffer = vec![0.0f32; total_kernel];
    let mut input_buffer = vec![
        0.0f32;
        ADACONV_MAX_INPUT_CHANNELS
            * (ADACONV_MAX_FRAME_SIZE + ADACONV_MAX_KERNEL_SIZE)
    ];
    let mut gain_buffer = vec![0.0f32; out_channels];

    // Prepare input: prepend history per channel
    for ic in 0..in_channels {
        let ib_off = ic * (kernel_size + frame_size);
        let hist_off = ic * kernel_size;
        input_buffer[ib_off..ib_off + kernel_size]
            .copy_from_slice(&state.history[hist_off..hist_off + kernel_size]);
        let xin_off = frame_size * ic;
        input_buffer[ib_off + kernel_size..ib_off + kernel_size + frame_size]
            .copy_from_slice(&x_in[xin_off..xin_off + frame_size]);
    }

    // Predict kernel and gain from features
    compute_generic_dense(
        kernel_layer,
        &mut kernel_buffer,
        features,
        ACTIVATION_LINEAR,
    );
    compute_generic_dense(gain_layer, &mut gain_buffer, features, ACTIVATION_TANH);
    transform_gains(&mut gain_buffer, out_channels, filter_gain_a, filter_gain_b);
    scale_kernel(
        &mut kernel_buffer,
        in_channels,
        out_channels,
        kernel_size,
        &gain_buffer,
    );

    // Overlap-add crossfading between last frame's kernel and new kernel
    let mut kernel0 = vec![0.0f32; ADACONV_MAX_KERNEL_SIZE];
    let mut kernel1 = vec![0.0f32; ADACONV_MAX_KERNEL_SIZE];
    let mut channel_buffer0 = vec![0.0f32; ADACONV_MAX_OVERLAP_SIZE];
    let mut channel_buffer1 = vec![0.0f32; ADACONV_MAX_FRAME_SIZE];

    for i_out in 0..out_channels {
        for i_in in 0..in_channels {
            // Zero and copy kernels
            for v in kernel0.iter_mut() {
                *v = 0.0;
            }
            for v in kernel1.iter_mut() {
                *v = 0.0;
            }
            let ki = (i_out * in_channels + i_in) * kernel_size;
            kernel0[..kernel_size].copy_from_slice(&state.last_kernel[ki..ki + kernel_size]);
            kernel1[..kernel_size].copy_from_slice(&kernel_buffer[ki..ki + kernel_size]);

            // p_input for this channel, offset by -left_padding
            let p_base = kernel_size + i_in * (frame_size + kernel_size);
            let y_start = p_base - left_padding;

            // Cross-correlate last kernel with input (overlap zone)
            pitch_xcorr(
                &kernel0,
                &input_buffer[y_start..],
                &mut channel_buffer0,
                ADACONV_MAX_KERNEL_SIZE,
                overlap_size,
            );
            // Cross-correlate new kernel with input (full frame)
            pitch_xcorr(
                &kernel1,
                &input_buffer[y_start..],
                &mut channel_buffer1,
                ADACONV_MAX_KERNEL_SIZE,
                frame_size,
            );

            // Overlap zone: crossfade
            for s in 0..overlap_size {
                let ob = s + i_out * frame_size;
                output_buffer[ob] += window[s] * channel_buffer0[s];
                output_buffer[ob] += (1.0 - window[s]) * channel_buffer1[s];
            }
            // Remaining: new kernel only
            for s in overlap_size..frame_size {
                output_buffer[s + i_out * frame_size] += channel_buffer1[s];
            }
        }
    }

    x_out[..out_channels * frame_size].copy_from_slice(&output_buffer[..out_channels * frame_size]);

    // Update state: save history tail and current kernel
    for ic in 0..in_channels {
        let p_base = kernel_size + ic * (frame_size + kernel_size);
        let hist_src = p_base + frame_size - kernel_size;
        let hist_dst = ic * kernel_size;
        state.history[hist_dst..hist_dst + kernel_size]
            .copy_from_slice(&input_buffer[hist_src..hist_src + kernel_size]);
    }
    state.last_kernel[..total_kernel].copy_from_slice(&kernel_buffer[..total_kernel]);
}

// ===========================================================================
// NNDSP — Adaptive comb filter (matches nndsp.c)
// ===========================================================================

/// Adaptive comb filter: pitch-conditioned filtering with overlap-add crossfading.
/// Matches C `adacomb_process_frame` from nndsp.c.
#[allow(clippy::too_many_arguments)]
pub fn adacomb_process_frame(
    state: &mut AdaCombState,
    x_out: &mut [f32],
    x_in: &[f32],
    features: &[f32],
    kernel_layer: &LinearLayer,
    gain_layer: &LinearLayer,
    global_gain_layer: &LinearLayer,
    pitch_lag: usize,
    _feature_dim: usize,
    frame_size: usize,
    overlap_size: usize,
    kernel_size: usize,
    left_padding: usize,
    filter_gain_a: f32,
    filter_gain_b: f32,
    log_gain_limit: f32,
    window: &[f32],
) {
    let mut output_buffer = vec![0.0f32; ADACOMB_MAX_FRAME_SIZE];
    let mut output_buffer_last = vec![0.0f32; ADACOMB_MAX_FRAME_SIZE];
    let mut kernel_buffer = [0.0f32; ADACOMB_MAX_KERNEL_SIZE];
    let hist_len = kernel_size + ADACOMB_MAX_LAG;
    let input_len = ADACOMB_MAX_FRAME_SIZE + ADACOMB_MAX_LAG + ADACOMB_MAX_KERNEL_SIZE;
    let mut input_buffer = vec![0.0f32; input_len];
    let mut gain = 0.0f32;
    let mut global_gain = 0.0f32;

    // Prepare input
    input_buffer[..hist_len].copy_from_slice(&state.history[..hist_len]);
    input_buffer[hist_len..hist_len + frame_size].copy_from_slice(&x_in[..frame_size]);
    // p_input points to start of current frame
    let p_input = kernel_size + ADACOMB_MAX_LAG;

    // Predict kernel, gain, global gain
    compute_generic_dense(
        kernel_layer,
        &mut kernel_buffer[..kernel_size],
        features,
        ACTIVATION_LINEAR,
    );
    compute_generic_dense(
        gain_layer,
        std::slice::from_mut(&mut gain),
        features,
        ACTIVATION_RELU,
    );
    compute_generic_dense(
        global_gain_layer,
        std::slice::from_mut(&mut global_gain),
        features,
        ACTIVATION_TANH,
    );

    // C: gain = exp(log_gain_limit - gain)
    gain = exp_dp(log_gain_limit - gain);
    // C: global_gain = exp(filter_gain_a * global_gain + filter_gain_b)
    global_gain = exp_dp(filter_gain_a * global_gain + filter_gain_b);
    scale_kernel(
        &mut kernel_buffer[..kernel_size],
        1,
        1,
        kernel_size,
        &[gain],
    );

    // Padded kernels for xcorr (ADACOMB_MAX_KERNEL_SIZE elements)
    let mut kernel = vec![0.0f32; ADACOMB_MAX_KERNEL_SIZE];
    let mut last_kernel = vec![0.0f32; ADACOMB_MAX_KERNEL_SIZE];
    kernel[..kernel_size].copy_from_slice(&kernel_buffer[..kernel_size]);
    last_kernel[..kernel_size].copy_from_slice(&state.last_kernel[..kernel_size]);

    // Cross-correlate last kernel at last pitch lag (overlap zone)
    let y_last = p_input - left_padding - state.last_pitch_lag;
    pitch_xcorr(
        &last_kernel,
        &input_buffer[y_last..],
        &mut output_buffer_last,
        ADACOMB_MAX_KERNEL_SIZE,
        overlap_size,
    );

    // Cross-correlate new kernel at current pitch lag (full frame)
    let y_new = p_input - left_padding - pitch_lag;
    pitch_xcorr(
        &kernel,
        &input_buffer[y_new..],
        &mut output_buffer,
        ADACOMB_MAX_KERNEL_SIZE,
        frame_size,
    );

    // Overlap zone: crossfade comb output + input
    for s in 0..overlap_size {
        output_buffer[s] = state.last_global_gain * window[s] * output_buffer_last[s]
            + global_gain * (1.0 - window[s]) * output_buffer[s];
    }
    for s in 0..overlap_size {
        output_buffer[s] += (window[s] * state.last_global_gain + (1.0 - window[s]) * global_gain)
            * input_buffer[p_input + s];
    }

    // Remaining: comb + input scaled by global gain
    for s in overlap_size..frame_size {
        output_buffer[s] = global_gain * (output_buffer[s] + input_buffer[p_input + s]);
    }

    x_out[..frame_size].copy_from_slice(&output_buffer[..frame_size]);

    // Update state
    state.last_kernel[..kernel_size].copy_from_slice(&kernel_buffer[..kernel_size]);
    let hist_src = p_input + frame_size - kernel_size - ADACOMB_MAX_LAG;
    state.history[..hist_len].copy_from_slice(&input_buffer[hist_src..hist_src + hist_len]);
    state.last_pitch_lag = pitch_lag;
    state.last_global_gain = global_gain;
}

// ===========================================================================
// NNDSP — Adaptive waveform shaping (matches nndsp.c)
// ===========================================================================

/// Adaptive waveform shaping: time-varying gain envelope from features.
/// Matches C `adashape_process_frame` from nndsp.c.
#[allow(clippy::too_many_arguments)]
pub fn adashape_process_frame(
    state: &mut AdaShapeState,
    x_out: &mut [f32],
    x_in: &[f32],
    features: &[f32],
    alpha1f: &LinearLayer,
    alpha1t: &LinearLayer,
    alpha2: &LinearLayer,
    feature_dim: usize,
    frame_size: usize,
    avg_pool_k: usize,
    interpolate_k: usize,
) {
    debug_assert_eq!(frame_size % avg_pool_k, 0);
    debug_assert_eq!(frame_size % interpolate_k, 0);

    let hidden_dim = frame_size / interpolate_k;
    let tenv_size = frame_size / avg_pool_k;
    let f = 1.0f32 / avg_pool_k as f32;

    debug_assert!(feature_dim + tenv_size + 1 < ADASHAPE_MAX_INPUT_DIM);

    // Build in_buffer = [features | tenv | mean]
    let in_buf_len = feature_dim + tenv_size + 1;
    let mut in_buffer = vec![0.0f32; in_buf_len.max(hidden_dim)];
    in_buffer[..feature_dim].copy_from_slice(&features[..feature_dim]);

    // Compute temporal envelope
    let tenv_start = feature_dim;
    let mut mean = 0.0f32;
    for i in 0..tenv_size {
        let mut acc = 0.0f32;
        for k in 0..avg_pool_k {
            acc += x_in[i * avg_pool_k + k].abs();
        }
        // celt_log = (float)log(x) — double precision
        in_buffer[tenv_start + i] = celt_log_f(acc * f + 1.52587890625e-05);
        mean += in_buffer[tenv_start + i];
    }
    mean /= tenv_size as f32;
    for i in 0..tenv_size {
        in_buffer[tenv_start + i] -= mean;
    }
    in_buffer[tenv_start + tenv_size] = mean;

    // Temporal weights: two parallel conv1d paths
    let mut out_buffer = vec![0.0f32; frame_size];
    let mut tmp_buffer = vec![0.0f32; frame_size];

    compute_generic_conv1d(
        alpha1f,
        &mut out_buffer,
        &mut state.conv_alpha1f_state,
        &in_buffer[..feature_dim],
        feature_dim,
        ACTIVATION_LINEAR,
    );
    compute_generic_conv1d(
        alpha1t,
        &mut tmp_buffer,
        &mut state.conv_alpha1t_state,
        &in_buffer[tenv_start..tenv_start + tenv_size + 1],
        tenv_size + 1,
        ACTIVATION_LINEAR,
    );

    // Leaky ReLU (slope = 0.2) combining the two paths
    for i in 0..hidden_dim {
        let tmp = out_buffer[i] + tmp_buffer[i];
        // C: 0.2 is double — promotes tmp to double, result truncated to float
        in_buffer[i] = if tmp >= 0.0 {
            tmp
        } else {
            (0.2_f64 * tmp as f64) as f32
        };
    }

    compute_generic_conv1d(
        alpha2,
        &mut tmp_buffer,
        &mut state.conv_alpha2_state,
        &in_buffer[..hidden_dim],
        hidden_dim,
        ACTIVATION_LINEAR,
    );

    // Upsample by linear interpolation
    for i in 0..hidden_dim {
        for k in 0..interpolate_k {
            let alpha = (k + 1) as f32 / interpolate_k as f32;
            out_buffer[i * interpolate_k + k] =
                alpha * tmp_buffer[i] + (1.0 - alpha) * state.interpolate_state[0];
        }
        state.interpolate_state[0] = tmp_buffer[i];
    }

    // Apply exponential gain
    compute_activation(&mut out_buffer, frame_size, ACTIVATION_EXP);
    for i in 0..frame_size {
        x_out[i] = out_buffer[i] * x_in[i];
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for &value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        bytes
    }

    fn i8_bytes(values: &[i8]) -> Vec<u8> {
        values.iter().map(|&value| value as u8).collect()
    }

    fn i32_bytes(values: &[i32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for &value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        bytes
    }

    fn make_weight_record(
        name: &str,
        weight_type: i32,
        payload: &[u8],
        block_size: usize,
        nul_terminated: bool,
    ) -> Vec<u8> {
        let mut record = vec![0u8; WEIGHT_BLOCK_SIZE + block_size];
        record[4..8].copy_from_slice(&WEIGHT_BLOB_VERSION.to_ne_bytes());
        record[8..12].copy_from_slice(&weight_type.to_ne_bytes());
        record[12..16].copy_from_slice(&(payload.len() as i32).to_ne_bytes());
        record[16..20].copy_from_slice(&(block_size as i32).to_ne_bytes());

        let name_bytes = name.as_bytes();
        record[20..20 + name_bytes.len()].copy_from_slice(name_bytes);
        if nul_terminated {
            record[20 + name_bytes.len()] = 0;
        } else {
            record[63] = b'X';
        }

        record[WEIGHT_BLOCK_SIZE..WEIGHT_BLOCK_SIZE + payload.len()].copy_from_slice(payload);
        record
    }

    // --- Approximation functions ---

    #[test]
    fn test_tanh_approx_zero() {
        assert!(approx_eq(tanh_approx(0.0), 0.0, EPS));
    }

    #[test]
    fn test_tanh_approx_one() {
        // tanh(1.0) ≈ 0.7616
        assert!(approx_eq(tanh_approx(1.0), 0.7616, 0.001));
    }

    #[test]
    fn test_tanh_approx_large() {
        assert!(approx_eq(tanh_approx(8.0), 1.0, 0.001));
        assert!(approx_eq(tanh_approx(-8.0), -1.0, 0.001));
    }

    #[test]
    fn test_tanh_approx_clamp() {
        // Should never exceed [-1, 1]
        assert!(tanh_approx(100.0) <= 1.0);
        assert!(tanh_approx(-100.0) >= -1.0);
    }

    #[test]
    fn test_sigmoid_approx_zero() {
        assert!(approx_eq(sigmoid_approx(0.0), 0.5, EPS));
    }

    #[test]
    fn test_sigmoid_approx_large() {
        assert!(sigmoid_approx(10.0) > 0.99);
        assert!(sigmoid_approx(-10.0) < 0.01);
    }

    #[test]
    fn test_lpcnet_exp2_identity() {
        // 2^0 = 1.0
        assert!(approx_eq(lpcnet_exp2(0.0), 1.0, 0.001));
    }

    #[test]
    fn test_lpcnet_exp2_powers() {
        assert!(approx_eq(lpcnet_exp2(1.0), 2.0, 0.01));
        assert!(approx_eq(lpcnet_exp2(2.0), 4.0, 0.02));
        assert!(approx_eq(lpcnet_exp2(-1.0), 0.5, 0.001));
    }

    #[test]
    fn test_lpcnet_exp2_underflow() {
        assert_eq!(lpcnet_exp2(-51.0), 0.0);
        assert_eq!(lpcnet_exp2(-100.0), 0.0);
    }

    #[test]
    fn test_lpcnet_exp_identity() {
        // e^0 = 1.0
        assert!(approx_eq(lpcnet_exp(0.0), 1.0, 0.001));
    }

    #[test]
    fn test_log2_approx() {
        assert!(approx_eq(log2_approx(1.0), 0.0, 0.01));
        assert!(approx_eq(log2_approx(2.0), 1.0, 0.01));
        assert!(approx_eq(log2_approx(4.0), 2.0, 0.01));
        assert!(approx_eq(log2_approx(0.5), -1.0, 0.01));
    }

    #[test]
    fn test_log_approx() {
        // ln(e) ≈ 1.0
        assert!(approx_eq(log_approx(std::f32::consts::E), 1.0, 0.01));
    }

    // --- GEMV ---

    #[test]
    fn test_sgemv_identity() {
        // 2×2 identity matrix (column-major)
        let weights = vec![1.0, 0.0, 0.0, 1.0]; // col0=[1,0], col1=[0,1]
        let x = vec![3.0, 7.0];
        let mut out = vec![0.0; 2];
        sgemv(&mut out, &weights, 2, 2, 2, &x);
        assert!(approx_eq(out[0], 3.0, EPS));
        assert!(approx_eq(out[1], 7.0, EPS));
    }

    #[test]
    fn test_sgemv_2x3() {
        // 2×3 matrix (col-major, col_stride=2):
        // W = [[1, 2, 3], [4, 5, 6]]
        // weights stored column-major: col0=[1,4], col1=[2,5], col2=[3,6]
        let weights = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let x = vec![1.0, 1.0, 1.0];
        let mut out = vec![0.0; 2];
        sgemv(&mut out, &weights, 2, 3, 2, &x);
        // row0: 1+2+3 = 6, row1: 4+5+6 = 15
        assert!(approx_eq(out[0], 6.0, EPS));
        assert!(approx_eq(out[1], 15.0, EPS));
    }

    // --- compute_linear ---

    #[test]
    fn test_compute_linear_dense_float() {
        // 2×2 layer: W = [[1, 2], [3, 4]], bias = [10, 20]
        // Column-major: col0=[1,3], col1=[2,4]
        let layer = LinearLayer {
            bias: Some(vec![10.0, 20.0]),
            float_weights: Some(vec![1.0, 3.0, 2.0, 4.0]),
            nb_inputs: 2,
            nb_outputs: 2,
            ..Default::default()
        };
        let input = vec![1.0, 1.0];
        let mut out = vec![0.0; 2];
        compute_linear(&layer, &mut out, &input);
        // out = W*x + bias = [1+2+10, 3+4+20] = [13, 27]
        assert!(approx_eq(out[0], 13.0, EPS));
        assert!(approx_eq(out[1], 27.0, EPS));
    }

    #[test]
    fn test_compute_linear_no_weights() {
        // No weights → output zeros, then add bias
        let layer = LinearLayer {
            bias: Some(vec![5.0, 10.0]),
            nb_inputs: 2,
            nb_outputs: 2,
            ..Default::default()
        };
        let input = vec![99.0, 99.0];
        let mut out = vec![0.0; 2];
        compute_linear(&layer, &mut out, &input);
        assert!(approx_eq(out[0], 5.0, EPS));
        assert!(approx_eq(out[1], 10.0, EPS));
    }

    // --- compute_activation ---

    #[test]
    fn test_activation_relu() {
        let mut data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        compute_activation(&mut data, 5, ACTIVATION_RELU);
        assert_eq!(data, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_activation_linear() {
        let mut data = vec![1.0, 2.0, 3.0];
        compute_activation(&mut data, 3, ACTIVATION_LINEAR);
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_activation_sigmoid() {
        let mut data = vec![0.0];
        compute_activation(&mut data, 1, ACTIVATION_SIGMOID);
        assert!(approx_eq(data[0], 0.5, 0.001));
    }

    #[test]
    fn test_activation_tanh() {
        let mut data = vec![0.0];
        compute_activation(&mut data, 1, ACTIVATION_TANH);
        assert!(approx_eq(data[0], 0.0, EPS));
    }

    #[test]
    fn test_activation_softmax_hack() {
        // SOFTMAX_HACK: identity
        let mut data = vec![1.0, 2.0, 3.0];
        compute_activation(&mut data, 3, ACTIVATION_SOFTMAX);
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    // --- compute_generic_dense ---

    #[test]
    fn test_generic_dense_relu() {
        // W = [[1, -1], [-1, 1]], bias = [0, 0], input = [1, 0]
        // output before activation = [1, -1], after ReLU = [1, 0]
        let layer = LinearLayer {
            bias: Some(vec![0.0, 0.0]),
            float_weights: Some(vec![1.0, -1.0, -1.0, 1.0]),
            nb_inputs: 2,
            nb_outputs: 2,
            ..Default::default()
        };
        let input = vec![1.0, 0.0];
        let mut out = vec![0.0; 2];
        compute_generic_dense(&layer, &mut out, &input, ACTIVATION_RELU);
        assert!(approx_eq(out[0], 1.0, EPS));
        assert!(approx_eq(out[1], 0.0, EPS));
    }

    // --- compute_generic_gru ---

    #[test]
    fn test_gru_zero_input() {
        // With zero weights and zero state, output should stay near zero
        let n = 2;
        let input_w = LinearLayer {
            bias: Some(vec![0.0; 6]),
            float_weights: Some(vec![0.0; 12]), // 2 inputs × 6 outputs
            nb_inputs: 2,
            nb_outputs: 6,
            ..Default::default()
        };
        let recur_w = LinearLayer {
            bias: Some(vec![0.0; 6]),
            float_weights: Some(vec![0.0; 12]), // 2 inputs × 6 outputs
            nb_inputs: n,
            nb_outputs: 3 * n,
            ..Default::default()
        };
        let mut state = vec![0.0f32; n];
        let input = vec![0.0f32; 2];
        compute_generic_gru(&input_w, &recur_w, &mut state, &input);
        // With all zeros: z = sigmoid(0) = 0.5, h = tanh(0) = 0
        // state = 0.5*0 + 0.5*0 = 0
        assert!(approx_eq(state[0], 0.0, 0.01));
        assert!(approx_eq(state[1], 0.0, 0.01));
    }

    // --- compute_generic_conv1d ---

    #[test]
    fn test_conv1d_single_frame() {
        // kernel_size=1, input_size=2 → nb_inputs=2, no memory needed
        let layer = LinearLayer {
            bias: Some(vec![1.0]),
            float_weights: Some(vec![1.0, 1.0]), // col-major: col0=[1], col1=[1]
            nb_inputs: 2,
            nb_outputs: 1,
            ..Default::default()
        };
        let input = vec![2.0, 3.0];
        let mut output = vec![0.0];
        let mut mem = vec![];
        compute_generic_conv1d(&layer, &mut output, &mut mem, &input, 2, ACTIVATION_LINEAR);
        // output = 1*2 + 1*3 + 1 = 6
        assert!(approx_eq(output[0], 6.0, EPS));
    }

    #[test]
    fn test_conv1d_with_memory() {
        // kernel_size=2, input_size=1 → nb_inputs=2, mem_size=1
        let layer = LinearLayer {
            bias: Some(vec![0.0]),
            float_weights: Some(vec![1.0, 2.0]), // col-major: col0=[1], col1=[2]
            nb_inputs: 2,
            nb_outputs: 1,
            ..Default::default()
        };
        let mut mem = vec![5.0]; // previous frame
        let input = vec![3.0]; // current frame

        let mut output = vec![0.0];
        compute_generic_conv1d(&layer, &mut output, &mut mem, &input, 1, ACTIVATION_LINEAR);
        // tmp = [5.0, 3.0], output = 1*5 + 2*3 = 11
        assert!(approx_eq(output[0], 11.0, EPS));
        // mem should now be [3.0] (current input becomes history)
        assert!(approx_eq(mem[0], 3.0, EPS));
    }

    // --- compute_glu ---

    #[test]
    fn test_glu_basic() {
        // GLU: output = input * sigmoid(W*input + bias)
        // With W=0, bias=0: sigmoid(0)=0.5, so output = input * 0.5
        let layer = LinearLayer {
            bias: Some(vec![0.0, 0.0]),
            float_weights: Some(vec![0.0; 4]),
            nb_inputs: 2,
            nb_outputs: 2,
            ..Default::default()
        };
        let input = vec![4.0, 6.0];
        let mut output = vec![0.0; 2];
        compute_glu(&layer, &mut output, &input);
        assert!(approx_eq(output[0], 2.0, 0.01));
        assert!(approx_eq(output[1], 3.0, 0.01));
    }

    // --- compute_overlap_window ---

    #[test]
    fn test_overlap_window() {
        let mut window = vec![0.0; 4];
        compute_overlap_window(&mut window, 4);
        // w[0] = 0.5 + 0.5*cos(π*0.5/4) ≈ 0.5 + 0.5*0.9239 ≈ 0.9619
        assert!(window[0] > 0.9);
        // w[3] = 0.5 + 0.5*cos(π*3.5/4) ≈ 0.5 + 0.5*(-0.3827) ≈ 0.3087
        assert!(window[3] < 0.35);
        // Window should be monotonically decreasing
        for i in 1..4 {
            assert!(window[i] < window[i - 1]);
        }
    }

    // --- Weight parsing ---

    #[test]
    fn test_bytes_to_f32_vec() {
        let val: f32 = 1.5;
        let bytes = val.to_ne_bytes();
        let mut data = Vec::new();
        data.extend_from_slice(&bytes);
        data.extend_from_slice(&(2.5f32).to_ne_bytes());
        let result = bytes_to_f32_vec(&data);
        assert_eq!(result.len(), 2);
        assert!(approx_eq(result[0], 1.5, EPS));
        assert!(approx_eq(result[1], 2.5, EPS));
    }

    // --- conv2d ---

    #[test]
    fn test_conv2d_float_1x1() {
        // 1 in_channel, 1 out_channel, ktime=1, kheight=1, height=2
        // This is just a scalar multiply per position
        let weights = vec![2.0f32]; // single weight
        let input = vec![3.0, 4.0]; // height=2, in_stride = 2+1-1 = 2
        let mut out = vec![0.0; 4]; // hstride=4
        conv2d_float(&mut out, &weights, 1, 1, 1, 1, &input, 2, 4);
        assert!(approx_eq(out[0], 6.0, EPS));
        assert!(approx_eq(out[1], 8.0, EPS));
    }

    // --- Integration: ulaw roundtrip ---

    #[test]
    fn test_ulaw_roundtrip() {
        // Encode then decode should be approximately identity for moderate values
        let original = 1000.0f32;
        let encoded = lin2ulaw(original);
        let decoded = ulaw2lin(encoded as f32);
        // Should be within ~5% for moderate values
        assert!((decoded - original).abs() / original < 0.1);
    }

    #[test]
    fn test_activation_swish_and_exp_paths() {
        let mut swish = vec![-1.0, 0.0, 1.0];
        let swish_len = swish.len();
        compute_activation(&mut swish, swish_len, ACTIVATION_SWISH);
        assert!(swish[0] < 0.0);
        assert!(approx_eq(swish[1], 0.0, EPS));
        assert!(swish[2] > 0.5);

        let mut exp = vec![0.0, 1.0];
        let exp_len = exp.len();
        compute_activation(&mut exp, exp_len, ACTIVATION_EXP);
        assert!(exp[0] > 0.99);
        assert!(exp[1] > 2.5);
    }

    #[test]
    fn test_compute_gated_activation_inplace_sigmoid() {
        let layer = LinearLayer {
            bias: Some(vec![0.0, 0.0]),
            float_weights: Some(vec![0.0; 4]),
            nb_inputs: 2,
            nb_outputs: 2,
            ..Default::default()
        };
        let mut data = vec![4.0, -2.0];
        compute_gated_activation_inplace(&layer, &mut data, ACTIVATION_SIGMOID);
        assert!(approx_eq(data[0], 2.0, 0.01));
        assert!(approx_eq(data[1], -1.0, 0.01));
    }

    #[test]
    fn test_compute_linear_dense_diag_and_int8_paths() {
        let dense = LinearLayer {
            bias: Some(vec![0.0; 6]),
            subias: Some(vec![1.0; 6]),
            float_weights: Some(vec![0.0; 12]),
            diag: Some(vec![1.0; 6]),
            nb_inputs: 2,
            nb_outputs: 6,
            ..Default::default()
        };
        let input = vec![2.0, 3.0];
        let mut dense_out = vec![0.0; 6];
        compute_linear(&dense, &mut dense_out, &input);
        assert_eq!(dense.subias.as_ref().unwrap().len(), 6);
        assert_eq!(dense_out, vec![2.0, 3.0, 2.0, 3.0, 2.0, 3.0]);

        let sparse = LinearLayer {
            float_weights: None,
            weights: Some(vec![1; 32]),
            scale: Some(vec![1.0; 8]),
            weights_idx: Some(vec![1, 0]),
            nb_inputs: 4,
            nb_outputs: 8,
            ..Default::default()
        };
        let mut sparse_out = vec![0.0; 8];
        let sparse_input = vec![1.0 / 127.0; 4];
        compute_linear(&sparse, &mut sparse_out, &sparse_input);
        assert!(sparse_out.iter().all(|&v| approx_eq(v, 4.0, EPS)));
    }

    #[test]
    fn test_compute_generic_conv1d_dilation_updates_memory() {
        let layer = LinearLayer {
            bias: Some(vec![0.0]),
            float_weights: Some(vec![0.0; 4]),
            nb_inputs: 4,
            nb_outputs: 1,
            ..Default::default()
        };
        let input = vec![1.0, 2.0];

        let mut output1 = vec![9.0];
        let mut mem1 = vec![10.0, 20.0];
        compute_generic_conv1d_dilation(
            &layer,
            &mut output1,
            &mut mem1,
            &input,
            2,
            1,
            ACTIVATION_LINEAR,
        );
        assert!(approx_eq(output1[0], 0.0, EPS));
        assert_eq!(mem1, vec![1.0, 2.0]);

        let mut output2 = vec![9.0];
        let mut mem2 = vec![10.0, 11.0, 20.0, 21.0];
        compute_generic_conv1d_dilation(
            &layer,
            &mut output2,
            &mut mem2,
            &input,
            2,
            2,
            ACTIVATION_LINEAR,
        );
        assert!(approx_eq(output2[0], 0.0, EPS));
        assert_eq!(mem2, vec![20.0, 21.0, 1.0, 2.0]);
    }

    #[test]
    fn test_compute_conv2d_generic_and_specialized_paths() {
        let generic = Conv2dLayer {
            bias: Some(vec![1.0]),
            float_weights: Some(vec![2.0]),
            in_channels: 1,
            out_channels: 1,
            ktime: 1,
            kheight: 1,
        };
        let mut out = vec![0.0; 2];
        let mut mem = vec![];
        let input = vec![3.0, 4.0];
        compute_conv2d(
            &generic,
            &mut out,
            &mut mem,
            &input,
            2,
            2,
            ACTIVATION_LINEAR,
        );
        assert_eq!(out, vec![7.0, 9.0]);
        assert!(mem.is_empty());

        let specialized = Conv2dLayer {
            bias: Some(vec![0.0]),
            float_weights: Some(vec![
                0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 1.0, 0.0,
            ]),
            in_channels: 1,
            out_channels: 1,
            ktime: 3,
            kheight: 3,
        };
        let mut out2 = vec![0.0; 1];
        let mut mem2 = vec![0.0; 6];
        let input2 = vec![1.0, 2.0, 3.0];
        compute_conv2d(
            &specialized,
            &mut out2,
            &mut mem2,
            &input2,
            1,
            1,
            ACTIVATION_LINEAR,
        );
        assert!(approx_eq(out2[0], 2.0, EPS));
        assert_eq!(mem2, vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_parse_weights_and_linear_init_paths() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&make_weight_record(
            "bias",
            WEIGHT_TYPE_FLOAT,
            &f32_bytes(&[1.25, -2.5]),
            8,
            true,
        ));
        blob.extend_from_slice(&make_weight_record(
            "weights",
            WEIGHT_TYPE_INT8,
            &i8_bytes(&[1, -2, 3, 4]),
            4,
            true,
        ));
        let arrays = parse_weights(&blob).expect("valid weight blob");
        assert_eq!(arrays.len(), 2);
        assert_eq!(arrays[0].name, "bias");
        assert_eq!(arrays[0].weight_type, WEIGHT_TYPE_FLOAT);
        assert_eq!(arrays[1].name, "weights");
        assert_eq!(arrays[1].weight_type, WEIGHT_TYPE_INT8);

        let dense_arrays = vec![
            WeightArray {
                name: "bias".into(),
                weight_type: WEIGHT_TYPE_FLOAT,
                size: 24,
                data: f32_bytes(&[0.0; 6]),
            },
            WeightArray {
                name: "subias".into(),
                weight_type: WEIGHT_TYPE_FLOAT,
                size: 24,
                data: f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            },
            WeightArray {
                name: "fw".into(),
                weight_type: WEIGHT_TYPE_FLOAT,
                size: 48,
                data: f32_bytes(&[0.0; 12]),
            },
            WeightArray {
                name: "diag".into(),
                weight_type: WEIGHT_TYPE_FLOAT,
                size: 24,
                data: f32_bytes(&[1.0; 6]),
            },
        ];
        let dense = linear_init(
            &dense_arrays,
            Some("bias"),
            Some("subias"),
            None,
            Some("fw"),
            None,
            Some("diag"),
            None,
            2,
            6,
        )
        .expect("dense linear init");
        assert_eq!(dense.subias.as_ref().unwrap()[1], 2.0);
        let mut dense_out = vec![0.0; 6];
        compute_linear(&dense, &mut dense_out, &[2.0, 3.0]);
        assert_eq!(dense_out, vec![2.0, 3.0, 2.0, 3.0, 2.0, 3.0]);

        let sparse_arrays = vec![
            WeightArray {
                name: "idx".into(),
                weight_type: WEIGHT_TYPE_INT,
                size: 8,
                data: i32_bytes(&[1, 0]),
            },
            WeightArray {
                name: "weights".into(),
                weight_type: WEIGHT_TYPE_INT8,
                size: 32,
                data: i8_bytes(&[1; 32]),
            },
            WeightArray {
                name: "scale".into(),
                weight_type: WEIGHT_TYPE_FLOAT,
                size: 32,
                data: f32_bytes(&[1.0; 8]),
            },
        ];
        let sparse = linear_init(
            &sparse_arrays,
            None,
            None,
            Some("weights"),
            None,
            Some("idx"),
            None,
            Some("scale"),
            4,
            8,
        )
        .expect("sparse linear init");
        let mut sparse_out = vec![0.0; 8];
        compute_linear(&sparse, &mut sparse_out, &[1.0 / 127.0; 4]);
        assert!(sparse_out.iter().all(|&v| approx_eq(v, 4.0, EPS)));

        assert!(
            linear_init(
                &sparse_arrays,
                None,
                None,
                Some("weights"),
                None,
                Some("idx"),
                None,
                None,
                4,
                8,
            )
            .is_err()
        );

        let bad_idx_arrays = vec![WeightArray {
            name: "bad_idx".into(),
            weight_type: WEIGHT_TYPE_INT,
            size: 8,
            data: i32_bytes(&[1, 1]),
        }];
        assert!(
            linear_init(
                &bad_idx_arrays,
                None,
                None,
                Some("weights"),
                None,
                Some("bad_idx"),
                None,
                Some("scale"),
                4,
                8,
            )
            .is_err()
        );

        let mut negative_size =
            make_weight_record("bad", WEIGHT_TYPE_FLOAT, &f32_bytes(&[0.0]), 4, true);
        negative_size[12..16].copy_from_slice(&(-1i32).to_ne_bytes());
        assert!(parse_weights(&negative_size).is_err());

        let mut bad_nul =
            make_weight_record("bad", WEIGHT_TYPE_FLOAT, &f32_bytes(&[0.0]), 4, false);
        bad_nul[16..20].copy_from_slice(&(4i32).to_ne_bytes());
        assert!(parse_weights(&bad_nul).is_err());

        let mut small_block =
            make_weight_record("bad", WEIGHT_TYPE_FLOAT, &f32_bytes(&[0.0]), 4, true);
        small_block[12..16].copy_from_slice(&(8i32).to_ne_bytes());
        assert!(parse_weights(&small_block).is_err());
    }

    #[test]
    fn test_nndsp_smoke_paths_with_zero_weights() {
        let zero_dense = |nb_inputs: usize, nb_outputs: usize| LinearLayer {
            float_weights: Some(vec![0.0; nb_inputs * nb_outputs]),
            nb_inputs,
            nb_outputs,
            ..Default::default()
        };

        let mut adaconv_state = AdaConvState::new();
        let adaconv_kernel = zero_dense(2, 2);
        let adaconv_gain = zero_dense(2, 1);
        let mut adaconv_out = vec![1.0; 4];
        adaconv_process_frame(
            &mut adaconv_state,
            &mut adaconv_out,
            &[1.0, 2.0, 3.0, 4.0],
            &[0.25, -0.5],
            &adaconv_kernel,
            &adaconv_gain,
            2,
            4,
            2,
            1,
            1,
            2,
            1,
            0.0,
            0.0,
            0.0,
            &[1.0, 0.0],
        );
        assert!(adaconv_out.iter().all(|&v| approx_eq(v, 0.0, EPS)));
        assert_eq!(&adaconv_state.history[..2], &[3.0, 4.0]);
        assert!(
            adaconv_state.last_kernel[..2]
                .iter()
                .all(|&v| approx_eq(v, 0.0, EPS))
        );

        let mut adacomb_state = AdaCombState::new();
        let adacomb_kernel = zero_dense(2, 2);
        let adacomb_gain = zero_dense(2, 1);
        let adacomb_global = zero_dense(2, 1);
        let mut adacomb_out = vec![0.0; 4];
        adacomb_process_frame(
            &mut adacomb_state,
            &mut adacomb_out,
            &[1.0, 2.0, 3.0, 4.0],
            &[0.1, -0.2],
            &adacomb_kernel,
            &adacomb_gain,
            &adacomb_global,
            1,
            2,
            4,
            2,
            2,
            1,
            0.0,
            0.0,
            0.0,
            &[0.0, 0.0],
        );
        assert!(
            adacomb_out
                .iter()
                .zip([1.0, 2.0, 3.0, 4.0])
                .all(|(a, b)| approx_eq(*a, b, 0.001))
        );
        assert_eq!(adacomb_state.last_pitch_lag, 1);
        assert!(approx_eq(adacomb_state.last_global_gain, 1.0, EPS));

        let mut adashape_state = AdaShapeState::new();
        let alpha1f = zero_dense(2, 4);
        let alpha1t = zero_dense(3, 4);
        let alpha2 = zero_dense(2, 4);
        let mut adashape_out = vec![0.0; 4];
        adashape_process_frame(
            &mut adashape_state,
            &mut adashape_out,
            &[1.0, 2.0, 3.0, 4.0],
            &[0.25, -0.5],
            &alpha1f,
            &alpha1t,
            &alpha2,
            2,
            4,
            2,
            2,
        );
        assert!(
            adashape_out
                .iter()
                .zip([1.0, 2.0, 3.0, 4.0])
                .all(|(a, b)| approx_eq(*a, b, 0.001))
        );
        assert!(
            adashape_state
                .conv_alpha1f_state
                .iter()
                .all(|&v| approx_eq(v, 0.0, EPS))
        );
        assert!(
            adashape_state
                .conv_alpha1t_state
                .iter()
                .all(|&v| approx_eq(v, 0.0, EPS))
        );
        assert!(
            adashape_state
                .conv_alpha2_state
                .iter()
                .all(|&v| approx_eq(v, 0.0, EPS))
        );
        assert!(approx_eq(adashape_state.interpolate_state[0], 0.0, EPS));
    }

    // --- Coverage additions: bias-only, boundary activations, conv2d, GRU ---

    #[test]
    fn test_compute_linear_bias_only_no_weights() {
        let layer = LinearLayer {
            bias: Some(vec![7.0, -3.0, 1.5]),
            nb_inputs: 4,
            nb_outputs: 3,
            ..Default::default()
        };
        let input = vec![100.0, 200.0, 300.0, 400.0];
        let mut out = vec![0.0; 3];
        compute_linear(&layer, &mut out, &input);
        assert!(approx_eq(out[0], 7.0, EPS));
        assert!(approx_eq(out[1], -3.0, EPS));
        assert!(approx_eq(out[2], 1.5, EPS));
    }

    #[test]
    fn test_compute_linear_no_bias_no_weights() {
        let layer = LinearLayer {
            nb_inputs: 3,
            nb_outputs: 2,
            ..Default::default()
        };
        let input = vec![1.0, 2.0, 3.0];
        let mut out = vec![999.0; 2];
        compute_linear(&layer, &mut out, &input);
        assert!(approx_eq(out[0], 0.0, EPS));
        assert!(approx_eq(out[1], 0.0, EPS));
    }

    #[test]
    fn test_activation_relu_boundary_values() {
        let mut data = vec![-1e-10, 0.0, 1e-10, -100.0, 100.0];
        let n = data.len();
        compute_activation(&mut data, n, ACTIVATION_RELU);
        assert!(data[0] == 0.0);
        assert!(data[1] == 0.0);
        assert!(data[2] > 0.0);
        assert!(data[3] == 0.0);
        assert!(approx_eq(data[4], 100.0, EPS));
    }

    #[test]
    fn test_activation_tanh_boundary_values() {
        let mut data = vec![-50.0, -1.0, 0.0, 1.0, 50.0];
        let n = data.len();
        compute_activation(&mut data, n, ACTIVATION_TANH);
        assert!(data[0] >= -1.0 && data[0] <= -0.99);
        assert!(data[2].abs() < 0.001);
        assert!(data[4] >= 0.99 && data[4] <= 1.0);
    }

    #[test]
    fn test_activation_sigmoid_boundary_saturation() {
        let mut data = vec![-100.0, 100.0];
        let n = data.len();
        compute_activation(&mut data, n, ACTIVATION_SIGMOID);
        assert!(data[0] < 0.01);
        assert!(data[1] > 0.99);
    }

    #[test]
    fn test_conv2d_ktime1_kheight1_multiple_channels() {
        let weights = vec![1.0f32, 0.0, 0.0, 1.0];
        let input = vec![3.0, 7.0];
        let mut out = vec![0.0; 2];
        conv2d_float(&mut out, &weights, 2, 2, 1, 1, &input, 1, 1);
        assert!(approx_eq(out[0], 3.0, EPS));
        assert!(approx_eq(out[1], 7.0, EPS));
    }

    #[test]
    fn test_gru_nonzero_bias_drives_state() {
        let n = 2;
        let mut input_bias = vec![0.0f32; 6];
        input_bias[0] = 10.0;
        input_bias[1] = 10.0;
        let input_w = LinearLayer {
            bias: Some(input_bias),
            float_weights: Some(vec![0.0; 12]),
            nb_inputs: 2,
            nb_outputs: 6,
            ..Default::default()
        };
        let recur_w = LinearLayer {
            bias: Some(vec![0.0; 6]),
            float_weights: Some(vec![0.0; 12]),
            nb_inputs: n,
            nb_outputs: 3 * n,
            ..Default::default()
        };
        let mut state = vec![5.0f32; n];
        let input = vec![0.0f32; 2];
        compute_generic_gru(&input_w, &recur_w, &mut state, &input);
        assert!(state[0] > 4.0);
    }

    #[test]
    fn test_gru_saturated_initial_state() {
        let n = 2;
        let input_w = LinearLayer {
            bias: Some(vec![0.0; 6]),
            float_weights: Some(vec![0.0; 12]),
            nb_inputs: 2,
            nb_outputs: 6,
            ..Default::default()
        };
        let recur_w = LinearLayer {
            bias: Some(vec![0.0; 6]),
            float_weights: Some(vec![0.0; 12]),
            nb_inputs: n,
            nb_outputs: 3 * n,
            ..Default::default()
        };
        let mut state = vec![1000.0f32; n];
        let input = vec![0.0f32; 2];
        compute_generic_gru(&input_w, &recur_w, &mut state, &input);
        assert!(state[0].is_finite());
        assert!(state[0].abs() < 1001.0);
    }

    #[test]
    fn test_conv1d_zero_memory_single_input() {
        let layer = LinearLayer {
            bias: Some(vec![0.0]),
            float_weights: Some(vec![3.0]),
            nb_inputs: 1,
            nb_outputs: 1,
            ..Default::default()
        };
        let input = vec![2.0];
        let mut output = vec![0.0];
        let mut mem = vec![];
        compute_generic_conv1d(&layer, &mut output, &mut mem, &input, 1, ACTIVATION_LINEAR);
        assert!(approx_eq(output[0], 6.0, EPS));
    }

    // --- sparse_sgemv8x4 via compute_linear ---

    #[test]
    fn test_sparse_sgemv8x4_identity_via_compute_linear() {
        // 8 outputs, 4 inputs, 1 sparse block at position 0.
        // Weight layout (input-major): w[s*8 + k] for input s, output k.
        // Build an "identity-ish" matrix: output[k] = x[k] for k < 4, output[k] = 0 for k >= 4.
        let mut weights = vec![0.0f32; 32]; // 4 inputs * 8 outputs
        // For input s=0: w[0*8+0] = 1.0 (output 0 gets x[0])
        weights[0 * 8 + 0] = 1.0;
        // For input s=1: w[1*8+1] = 1.0 (output 1 gets x[1])
        weights[1 * 8 + 1] = 1.0;
        // For input s=2: w[2*8+2] = 1.0 (output 2 gets x[2])
        weights[2 * 8 + 2] = 1.0;
        // For input s=3: w[3*8+3] = 1.0 (output 3 gets x[3])
        weights[3 * 8 + 3] = 1.0;

        let idx = vec![1i32, 0]; // 1 block at column position 0
        let layer = LinearLayer {
            float_weights: Some(weights),
            weights_idx: Some(idx),
            nb_inputs: 4,
            nb_outputs: 8,
            ..Default::default()
        };
        let x = vec![10.0, 20.0, 30.0, 40.0];
        let mut out = vec![0.0; 8];
        compute_linear(&layer, &mut out, &x);
        assert!(approx_eq(out[0], 10.0, EPS));
        assert!(approx_eq(out[1], 20.0, EPS));
        assert!(approx_eq(out[2], 30.0, EPS));
        assert!(approx_eq(out[3], 40.0, EPS));
        for k in 4..8 {
            assert!(approx_eq(out[k], 0.0, EPS));
        }
    }

    #[test]
    fn test_sparse_sgemv8x4_uniform_weights_with_bias() {
        // 8 outputs, 8 inputs, 2 sparse blocks per row-group (at columns 0 and 4).
        // All weights = 1.0, so each output = sum of all 8 inputs = 36.0
        let weights = vec![1.0f32; 64]; // 2 blocks * 32 weights each
        let idx = vec![2i32, 0, 4]; // 2 blocks: at position 0 and position 4
        let bias = vec![100.0f32; 8];
        let layer = LinearLayer {
            float_weights: Some(weights),
            weights_idx: Some(idx),
            bias: Some(bias),
            nb_inputs: 8,
            nb_outputs: 8,
            ..Default::default()
        };
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out = vec![0.0; 8];
        compute_linear(&layer, &mut out, &x);
        // Each output = sum(1..8) + 100 = 36 + 100 = 136
        for k in 0..8 {
            assert!(approx_eq(out[k], 136.0, EPS));
        }
    }

    #[test]
    fn test_sparse_sgemv8x4_16_outputs_two_row_groups() {
        // 16 outputs, 4 inputs. Two row groups of 8 each, each with 1 block.
        // First group: all weights = 2.0, second group: all weights = -1.0
        let mut weights = vec![0.0f32; 64];
        weights[..32].fill(2.0);
        weights[32..64].fill(-1.0);
        // idx: [1, 0] for first group, [1, 0] for second group
        let idx = vec![1i32, 0, 1, 0];
        let layer = LinearLayer {
            float_weights: Some(weights),
            weights_idx: Some(idx),
            nb_inputs: 4,
            nb_outputs: 16,
            ..Default::default()
        };
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; 16];
        compute_linear(&layer, &mut out, &x);
        // First 8 outputs: 2*1 + 2*1 + 2*1 + 2*1 = 8.0
        for k in 0..8 {
            assert!(approx_eq(out[k], 8.0, EPS));
        }
        // Last 8 outputs: -1*1 + -1*1 + -1*1 + -1*1 = -4.0
        for k in 8..16 {
            assert!(approx_eq(out[k], -4.0, EPS));
        }
    }

    // --- compute_glu_inplace ---

    #[test]
    fn test_compute_glu_inplace_zero_weights() {
        // With W=0, bias=0: sigmoid(0) = 0.5, so data *= 0.5
        let layer = LinearLayer {
            bias: Some(vec![0.0, 0.0, 0.0]),
            float_weights: Some(vec![0.0; 9]),
            nb_inputs: 3,
            nb_outputs: 3,
            ..Default::default()
        };
        let mut data = vec![10.0, -6.0, 0.0];
        compute_glu_inplace(&layer, &mut data);
        assert!(approx_eq(data[0], 5.0, 0.01));
        assert!(approx_eq(data[1], -3.0, 0.01));
        assert!(approx_eq(data[2], 0.0, EPS));
    }

    #[test]
    fn test_compute_glu_inplace_large_positive_bias() {
        // With W=0, bias=+100: sigmoid(100) ~ 1.0, so data is nearly unchanged
        let layer = LinearLayer {
            bias: Some(vec![100.0, 100.0]),
            float_weights: Some(vec![0.0; 4]),
            nb_inputs: 2,
            nb_outputs: 2,
            ..Default::default()
        };
        let mut data = vec![7.0, -3.0];
        compute_glu_inplace(&layer, &mut data);
        assert!(approx_eq(data[0], 7.0, 0.01));
        assert!(approx_eq(data[1], -3.0, 0.01));
    }

    // --- compute_gated_activation ---

    #[test]
    fn test_compute_gated_activation_with_relu() {
        // output = input * relu(W*input + bias)
        // W=0, bias=[-1, 2]: relu(-1)=0, relu(2)=2
        // output[0] = input[0] * 0 = 0, output[1] = input[1] * 2
        let layer = LinearLayer {
            bias: Some(vec![-1.0, 2.0]),
            float_weights: Some(vec![0.0; 4]),
            nb_inputs: 2,
            nb_outputs: 2,
            ..Default::default()
        };
        let input = vec![5.0, 3.0];
        let mut output = vec![0.0; 2];
        compute_gated_activation(&layer, &mut output, &input, ACTIVATION_RELU);
        assert!(approx_eq(output[0], 0.0, EPS));
        assert!(approx_eq(output[1], 6.0, EPS));
    }

    #[test]
    fn test_compute_gated_activation_with_tanh() {
        // output = input * tanh(W*input + bias)
        // W=0, bias=0: tanh(0) = 0, so output should be ~0
        let layer = LinearLayer {
            bias: Some(vec![0.0, 0.0]),
            float_weights: Some(vec![0.0; 4]),
            nb_inputs: 2,
            nb_outputs: 2,
            ..Default::default()
        };
        let input = vec![100.0, -50.0];
        let mut output = vec![0.0; 2];
        compute_gated_activation(&layer, &mut output, &input, ACTIVATION_TANH);
        assert!(approx_eq(output[0], 0.0, 0.01));
        assert!(approx_eq(output[1], 0.0, 0.01));
    }

    // --- conv2d_init ---

    #[test]
    fn test_conv2d_init_with_bias_and_weights() {
        let in_ch = 2;
        let out_ch = 3;
        let kt = 2;
        let kh = 2;
        let weight_count = in_ch * out_ch * kt * kh; // 24
        let weight_values: Vec<f32> = (0..weight_count).map(|i| i as f32 * 0.1).collect();
        let bias_values = vec![1.0f32, 2.0, 3.0];
        let arrays = vec![
            WeightArray {
                name: "conv_bias".into(),
                weight_type: WEIGHT_TYPE_FLOAT,
                size: out_ch * 4, // 12
                data: f32_bytes(&bias_values),
            },
            WeightArray {
                name: "conv_weights".into(),
                weight_type: WEIGHT_TYPE_FLOAT,
                size: weight_count * 4, // 96
                data: f32_bytes(&weight_values),
            },
        ];
        let layer = conv2d_init(
            &arrays,
            Some("conv_bias"),
            Some("conv_weights"),
            in_ch,
            out_ch,
            kt,
            kh,
        )
        .expect("conv2d_init should succeed");
        assert_eq!(layer.in_channels, 2);
        assert_eq!(layer.out_channels, 3);
        assert_eq!(layer.ktime, 2);
        assert_eq!(layer.kheight, 2);
        assert_eq!(layer.bias.as_ref().unwrap(), &bias_values);
        assert_eq!(layer.float_weights.as_ref().unwrap().len(), weight_count);
        assert!(approx_eq(
            layer.float_weights.as_ref().unwrap()[5],
            0.5,
            EPS,
        ));
    }

    #[test]
    fn test_conv2d_init_no_bias_no_weights() {
        let arrays: Vec<WeightArray> = vec![];
        let layer = conv2d_init(&arrays, None, None, 1, 1, 1, 1)
            .expect("conv2d_init with no arrays should succeed");
        assert!(layer.bias.is_none());
        assert!(layer.float_weights.is_none());
        assert_eq!(layer.in_channels, 1);
    }

    // --- linear_init sparse float path ---

    #[test]
    fn test_linear_init_sparse_float_weights() {
        // 8 outputs, 4 inputs, 1 sparse block at position 0.
        // idx = [1, 0] → 1 block at col 0, total_blocks = 1
        // float_weights size = SPARSE_BLOCK_SIZE * total_blocks * 4 = 32 * 1 * 4 = 128 bytes
        let float_weight_values = vec![1.0f32; SPARSE_BLOCK_SIZE]; // 32 floats
        let bias_values = vec![0.5f32; 8];
        let arrays = vec![
            WeightArray {
                name: "sp_idx".into(),
                weight_type: WEIGHT_TYPE_INT,
                size: 8, // 2 i32s = 8 bytes
                data: i32_bytes(&[1, 0]),
            },
            WeightArray {
                name: "sp_fw".into(),
                weight_type: WEIGHT_TYPE_FLOAT,
                size: SPARSE_BLOCK_SIZE * 1 * 4,
                data: f32_bytes(&float_weight_values),
            },
            WeightArray {
                name: "sp_bias".into(),
                weight_type: WEIGHT_TYPE_FLOAT,
                size: 8 * 4,
                data: f32_bytes(&bias_values),
            },
        ];
        let layer = linear_init(
            &arrays,
            Some("sp_bias"),
            None,
            None,           // no int8 weights
            Some("sp_fw"),  // float weights
            Some("sp_idx"), // sparse index
            None,
            None,
            4,
            8,
        )
        .expect("sparse float linear_init should succeed");
        assert!(layer.weights_idx.is_some());
        assert!(layer.float_weights.is_some());
        assert!(layer.weights.is_none());
        assert_eq!(
            layer.float_weights.as_ref().unwrap().len(),
            SPARSE_BLOCK_SIZE
        );
        assert_eq!(layer.bias.as_ref().unwrap().len(), 8);

        // Verify it works through compute_linear (exercises sparse_sgemv8x4)
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 8];
        compute_linear(&layer, &mut out, &x);
        // All weights = 1.0, each output = 1+2+3+4 = 10, plus bias 0.5 = 10.5
        for k in 0..8 {
            assert!(approx_eq(out[k], 10.5, EPS));
        }
    }

    #[test]
    fn test_linear_init_sparse_int8_and_float_together() {
        // Test the path where both int8 weights AND float weights are provided
        // with a sparse index. Int8 should be loaded, float should be loaded too.
        let arrays = vec![
            WeightArray {
                name: "sp_idx".into(),
                weight_type: WEIGHT_TYPE_INT,
                size: 8,
                data: i32_bytes(&[1, 0]),
            },
            WeightArray {
                name: "sp_w".into(),
                weight_type: WEIGHT_TYPE_INT8,
                size: SPARSE_BLOCK_SIZE * 1, // 32 bytes of i8
                data: i8_bytes(&[1i8; SPARSE_BLOCK_SIZE]),
            },
            WeightArray {
                name: "sp_fw".into(),
                weight_type: WEIGHT_TYPE_FLOAT,
                size: SPARSE_BLOCK_SIZE * 1 * 4,
                data: f32_bytes(&vec![2.0f32; SPARSE_BLOCK_SIZE]),
            },
            WeightArray {
                name: "sp_scale".into(),
                weight_type: WEIGHT_TYPE_FLOAT,
                size: 8 * 4,
                data: f32_bytes(&[1.0f32; 8]),
            },
        ];
        let layer = linear_init(
            &arrays,
            None,
            None,
            Some("sp_w"),   // int8 weights
            Some("sp_fw"),  // float weights
            Some("sp_idx"), // sparse index
            None,
            Some("sp_scale"),
            4,
            8,
        )
        .expect("sparse int8+float linear_init should succeed");
        assert!(layer.weights_idx.is_some());
        assert!(layer.weights.is_some());
        assert!(layer.float_weights.is_some());
        assert_eq!(layer.weights.as_ref().unwrap().len(), SPARSE_BLOCK_SIZE);
        assert_eq!(
            layer.float_weights.as_ref().unwrap().len(),
            SPARSE_BLOCK_SIZE
        );
    }
}
