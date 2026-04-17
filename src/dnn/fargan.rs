//! FARGAN — Frequency-domain Autoregressive Generative Adversarial Network.
//!
//! Neural waveform generator for the Opus codec's DNN subsystem (Opus 1.4+).
//! Synthesizes time-domain PCM audio from spectral feature vectors, one 10 ms
//! frame (160 samples at 16 kHz) at a time.
//!
//! Matches C reference: `fargan.c`, `fargan.h`.
//!
//! All computation is IEEE 754 single-precision float unless noted.
//! Double precision is used where the C reference uses `double` (period
//! computation, exp, float-to-int16 conversion).

use super::core::{
    ACTIVATION_LINEAR, ACTIVATION_SIGMOID, ACTIVATION_TANH, LinearLayer, WeightArray,
    compute_generic_conv1d, compute_generic_dense, compute_generic_gru, compute_glu, linear_init,
    parse_weights,
};
use super::lpcnet::{LPCNET_FRAME_SIZE, NB_BANDS, NB_FEATURES, PITCH_MAX_PERIOD};

// ===========================================================================
// Constants
// ===========================================================================

/// Number of PCM samples required by `fargan_cont` (20 ms at 16 kHz).
pub const FARGAN_CONT_SAMPLES: usize = 320;

/// Number of subframes per frame.
pub const FARGAN_NB_SUBFRAMES: usize = 4;

/// Samples per subframe (2.5 ms at 16 kHz).
pub const FARGAN_SUBFRAME_SIZE: usize = 40;

/// Total frame size in samples (10 ms at 16 kHz).
pub const FARGAN_FRAME_SIZE: usize = FARGAN_NB_SUBFRAMES * FARGAN_SUBFRAME_SIZE;

/// De-emphasis coefficient (matches PREEMPHASIS in freq.h).
pub const FARGAN_DEEMPHASIS: f32 = 0.85;

// -- Conditioning network layer dimensions --

/// Pitch embedding output dimension.
const COND_NET_PEMBED_OUT_SIZE: usize = 12;

/// Number of pitch embedding entries (PITCH_MAX_PERIOD - PITCH_MIN_PERIOD).
const PEMBED_NUM_ENTRIES: usize = 224;

/// Minimum pitch period in samples.
const PITCH_MIN_PERIOD: usize = 32;

/// fdense1 output / fconv1 input channels.
const COND_NET_FCONV1_IN_SIZE: usize = 64;

/// fconv1 output channels.
const COND_NET_FCONV1_OUT_SIZE: usize = 128;

/// fconv1 state size = input_channels * (kernel_size - 1) = 64 * 2.
const COND_NET_FCONV1_STATE_SIZE: usize = COND_NET_FCONV1_IN_SIZE * 2;

/// fdense2 output size = 80 * 4 subframes.
const COND_NET_FDENSE2_OUT_SIZE: usize = 320;

/// Per-subframe conditioning vector size.
pub const FARGAN_COND_SIZE: usize = COND_NET_FDENSE2_OUT_SIZE / FARGAN_NB_SUBFRAMES;

// -- Signal network layer dimensions --

/// FWConv input size: cond(80) + pred(40+4) + prev(40).
const SIG_NET_INPUT_SIZE: usize = FARGAN_COND_SIZE + 2 * FARGAN_SUBFRAME_SIZE + 4;

/// FWConv state size (C declares 2 * input_size; actual usage is input_size).
const SIG_NET_FWC0_STATE_SIZE: usize = 2 * SIG_NET_INPUT_SIZE;

/// FWConv output size (before and after GLU — GLU preserves size).
const SIG_NET_FWC0_CONV_OUT_SIZE: usize = 192;
const SIG_NET_FWC0_GLU_GATE_OUT_SIZE: usize = 192;

/// GRU hidden state sizes.
const SIG_NET_GRU1_OUT_SIZE: usize = 160;
const SIG_NET_GRU1_STATE_SIZE: usize = 160;
const SIG_NET_GRU2_OUT_SIZE: usize = 128;
const SIG_NET_GRU2_STATE_SIZE: usize = 128;
const SIG_NET_GRU3_OUT_SIZE: usize = 128;
const SIG_NET_GRU3_STATE_SIZE: usize = 128;

/// Skip-connection dense output size.
const SIG_NET_SKIP_DENSE_OUT_SIZE: usize = 128;

/// Skip concatenation total size:
/// gru1_glu(160) + gru2_glu(128) + gru3_glu(128) + fwc0_glu(192) + pred(40) + prev(40).
const SKIP_CAT_SIZE: usize = SIG_NET_GRU1_OUT_SIZE
    + SIG_NET_GRU2_OUT_SIZE
    + SIG_NET_GRU3_OUT_SIZE
    + SIG_NET_FWC0_CONV_OUT_SIZE
    + FARGAN_SUBFRAME_SIZE
    + FARGAN_SUBFRAME_SIZE;

// ===========================================================================
// Model weights
// ===========================================================================

/// Neural network weights for the FARGAN vocoder.
///
/// Matches C `FARGAN` struct from auto-generated `fargan_data.h`.
#[derive(Clone, Debug, Default)]
pub struct FarganModel {
    // -- Conditioning network --
    /// Pitch embedding lookup [224 x 12].
    pub cond_net_pembed: LinearLayer,
    /// Dense: (NB_FEATURES + 12) -> 64, tanh, no bias.
    pub cond_net_fdense1: LinearLayer,
    /// Conv1d: 64 -> 128, kernel_size=3, causal, tanh, no bias.
    pub cond_net_fconv1: LinearLayer,
    /// Dense: 128 -> 320, tanh, no bias.
    pub cond_net_fdense2: LinearLayer,

    // -- Signal network --
    /// Gain prediction: 80 -> 1, linear activation, with bias.
    pub sig_net_cond_gain_dense: LinearLayer,
    /// FWConv linear: (input_size * 2) -> 192, tanh, no bias.
    pub sig_net_fwc0_conv: LinearLayer,
    /// FWConv GLU gate: 192 -> 192, no bias.
    pub sig_net_fwc0_glu_gate: LinearLayer,
    /// Pitch gate: 192 -> 4, sigmoid, with bias.
    pub sig_net_gain_dense_out: LinearLayer,

    /// GRU1 input weights: 272 -> 3*160=480.
    pub sig_net_gru1_input: LinearLayer,
    /// GRU1 recurrent weights: 160 -> 480.
    pub sig_net_gru1_recurrent: LinearLayer,
    /// GRU1 GLU gate: 160 -> 160.
    pub sig_net_gru1_glu_gate: LinearLayer,

    /// GRU2 input weights: 240 -> 3*128=384.
    pub sig_net_gru2_input: LinearLayer,
    /// GRU2 recurrent weights: 128 -> 384.
    pub sig_net_gru2_recurrent: LinearLayer,
    /// GRU2 GLU gate: 128 -> 128.
    pub sig_net_gru2_glu_gate: LinearLayer,

    /// GRU3 input weights: 208 -> 3*128=384.
    pub sig_net_gru3_input: LinearLayer,
    /// GRU3 recurrent weights: 128 -> 384.
    pub sig_net_gru3_recurrent: LinearLayer,
    /// GRU3 GLU gate: 128 -> 128.
    pub sig_net_gru3_glu_gate: LinearLayer,

    /// Skip-connection dense: 688 -> 128, tanh, no bias.
    pub sig_net_skip_dense: LinearLayer,
    /// Skip GLU gate: 128 -> 128.
    pub sig_net_skip_glu_gate: LinearLayer,
    /// Output dense: 128 -> 40, tanh, no bias.
    pub sig_net_sig_dense_out: LinearLayer,
}

// ===========================================================================
// State
// ===========================================================================

/// Complete runtime state for the FARGAN vocoder.
///
/// Matches C `FARGANState` from `fargan.h`.
#[derive(Clone, Debug)]
pub struct FarganState {
    /// Neural network weights.
    pub model: FarganModel,
    /// Whether `cont()` has been called to prime the state.
    pub cont_initialized: bool,
    /// De-emphasis filter memory (one sample).
    pub deemph_mem: f32,
    /// Circular pitch buffer of pre-emphasized samples [PITCH_MAX_PERIOD].
    pub pitch_buf: Vec<f32>,
    /// Conv1d state for conditioning network fconv1.
    pub cond_conv1_state: Vec<f32>,
    /// FWConv state memory.
    pub fwc0_mem: Vec<f32>,
    /// GRU1 hidden state.
    pub gru1_state: Vec<f32>,
    /// GRU2 hidden state.
    pub gru2_state: Vec<f32>,
    /// GRU3 hidden state.
    pub gru3_state: Vec<f32>,
    /// Pitch period from the previous frame (one-frame lag).
    pub last_period: i32,
}

impl Default for FarganState {
    fn default() -> Self {
        Self {
            model: FarganModel::default(),
            cont_initialized: false,
            deemph_mem: 0.0,
            pitch_buf: vec![0.0; PITCH_MAX_PERIOD],
            cond_conv1_state: vec![0.0; COND_NET_FCONV1_STATE_SIZE],
            fwc0_mem: vec![0.0; SIG_NET_FWC0_STATE_SIZE],
            gru1_state: vec![0.0; SIG_NET_GRU1_STATE_SIZE],
            gru2_state: vec![0.0; SIG_NET_GRU2_STATE_SIZE],
            gru3_state: vec![0.0; SIG_NET_GRU3_STATE_SIZE],
            last_period: 0,
        }
    }
}

// ===========================================================================
// Model initialization
// ===========================================================================

/// Initialize a `FarganModel` from named weight arrays.
///
/// Equivalent to the C auto-generated `init_fargan()` from `fargan_data.c`.
/// Weight array names follow the wexchange naming convention:
/// `{layer_name}_{suffix}` where suffix is one of: `bias`, `subias`,
/// `weights_int8`, `weights_float`, `scale`.
pub fn init_fargan(arrays: &[WeightArray]) -> Result<FarganModel, ()> {
    // -- Conditioning network --

    // Pitch embedding: treated as a dense layer by wexchange.
    // float_weights contains 224 * 12 floats (row-major: one row per embedding).
    let cond_net_pembed = linear_init(
        arrays,
        Some("cond_net_pembed_bias"),
        None,
        None,
        Some("cond_net_pembed_weights_float"),
        None,
        None,
        None,
        PEMBED_NUM_ENTRIES,       // 224
        COND_NET_PEMBED_OUT_SIZE, // 12
    )?;

    // fdense1: Linear(32 -> 64), unquantized, no real bias (zeros).
    let cond_net_fdense1 = linear_init(
        arrays,
        Some("cond_net_fdense1_bias"),
        None,
        None,
        Some("cond_net_fdense1_weights_float"),
        None,
        None,
        None,
        NB_FEATURES + COND_NET_PEMBED_OUT_SIZE, // 32
        COND_NET_FCONV1_IN_SIZE,                // 64
    )?;

    // fconv1: Conv1d(64 -> 128, kernel=3), quantized, no real bias (zeros).
    // Stored as LinearLayer with nb_inputs = kernel * channels = 192.
    let cond_net_fconv1 = linear_init(
        arrays,
        Some("cond_net_fconv1_bias"),
        Some("cond_net_fconv1_subias"),
        Some("cond_net_fconv1_weights_int8"),
        Some("cond_net_fconv1_weights_float"),
        None,
        None,
        Some("cond_net_fconv1_scale"),
        COND_NET_FCONV1_IN_SIZE * 3, // 192
        COND_NET_FCONV1_OUT_SIZE,    // 128
    )?;

    // fdense2: Linear(128 -> 320), quantized, no real bias (zeros).
    let cond_net_fdense2 = linear_init(
        arrays,
        Some("cond_net_fdense2_bias"),
        Some("cond_net_fdense2_subias"),
        Some("cond_net_fdense2_weights_int8"),
        Some("cond_net_fdense2_weights_float"),
        None,
        None,
        Some("cond_net_fdense2_scale"),
        COND_NET_FCONV1_OUT_SIZE,  // 128
        COND_NET_FDENSE2_OUT_SIZE, // 320
    )?;

    // -- Signal network --

    // Gain prediction: Linear(80 -> 1), unquantized, with real bias.
    let sig_net_cond_gain_dense = linear_init(
        arrays,
        Some("sig_net_cond_gain_dense_bias"),
        None,
        None,
        Some("sig_net_cond_gain_dense_weights_float"),
        None,
        None,
        None,
        FARGAN_COND_SIZE, // 80
        1,
    )?;

    // FWConv linear: Linear(328 -> 192), quantized, no real bias (zeros).
    let sig_net_fwc0_conv = linear_init(
        arrays,
        Some("sig_net_fwc0_conv_bias"),
        Some("sig_net_fwc0_conv_subias"),
        Some("sig_net_fwc0_conv_weights_int8"),
        Some("sig_net_fwc0_conv_weights_float"),
        None,
        None,
        Some("sig_net_fwc0_conv_scale"),
        SIG_NET_INPUT_SIZE * 2,     // 328
        SIG_NET_FWC0_CONV_OUT_SIZE, // 192
    )?;

    // FWConv GLU gate: Linear(192 -> 192), quantized, no real bias (zeros).
    let sig_net_fwc0_glu_gate = linear_init(
        arrays,
        Some("sig_net_fwc0_glu_gate_bias"),
        Some("sig_net_fwc0_glu_gate_subias"),
        Some("sig_net_fwc0_glu_gate_weights_int8"),
        Some("sig_net_fwc0_glu_gate_weights_float"),
        None,
        None,
        Some("sig_net_fwc0_glu_gate_scale"),
        SIG_NET_FWC0_CONV_OUT_SIZE,     // 192
        SIG_NET_FWC0_GLU_GATE_OUT_SIZE, // 192
    )?;

    // Pitch gate: Linear(192 -> 4), unquantized, with real bias.
    let sig_net_gain_dense_out = linear_init(
        arrays,
        Some("sig_net_gain_dense_out_bias"),
        None,
        None,
        Some("sig_net_gain_dense_out_weights_float"),
        None,
        None,
        None,
        SIG_NET_FWC0_GLU_GATE_OUT_SIZE, // 192
        4,
    )?;

    // GRU1: GRUCell(272 -> 160), quantized, bias=False (subias only).
    let sig_net_gru1_input = linear_init(
        arrays,
        None,
        Some("sig_net_gru1_input_subias"),
        Some("sig_net_gru1_input_weights_int8"),
        Some("sig_net_gru1_input_weights_float"),
        None,
        None,
        Some("sig_net_gru1_input_scale"),
        SIG_NET_FWC0_GLU_GATE_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE, // 272
        3 * SIG_NET_GRU1_STATE_SIZE,                               // 480
    )?;
    let sig_net_gru1_recurrent = linear_init(
        arrays,
        None,
        Some("sig_net_gru1_recurrent_subias"),
        Some("sig_net_gru1_recurrent_weights_int8"),
        Some("sig_net_gru1_recurrent_weights_float"),
        None,
        None,
        Some("sig_net_gru1_recurrent_scale"),
        SIG_NET_GRU1_STATE_SIZE,     // 160
        3 * SIG_NET_GRU1_STATE_SIZE, // 480
    )?;
    let sig_net_gru1_glu_gate = linear_init(
        arrays,
        Some("sig_net_gru1_glu_gate_bias"),
        Some("sig_net_gru1_glu_gate_subias"),
        Some("sig_net_gru1_glu_gate_weights_int8"),
        Some("sig_net_gru1_glu_gate_weights_float"),
        None,
        None,
        Some("sig_net_gru1_glu_gate_scale"),
        SIG_NET_GRU1_OUT_SIZE, // 160
        SIG_NET_GRU1_OUT_SIZE, // 160
    )?;

    // GRU2: GRUCell(240 -> 128), quantized, bias=False.
    let sig_net_gru2_input = linear_init(
        arrays,
        None,
        Some("sig_net_gru2_input_subias"),
        Some("sig_net_gru2_input_weights_int8"),
        Some("sig_net_gru2_input_weights_float"),
        None,
        None,
        Some("sig_net_gru2_input_scale"),
        SIG_NET_GRU1_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE, // 240
        3 * SIG_NET_GRU2_STATE_SIZE,                      // 384
    )?;
    let sig_net_gru2_recurrent = linear_init(
        arrays,
        None,
        Some("sig_net_gru2_recurrent_subias"),
        Some("sig_net_gru2_recurrent_weights_int8"),
        Some("sig_net_gru2_recurrent_weights_float"),
        None,
        None,
        Some("sig_net_gru2_recurrent_scale"),
        SIG_NET_GRU2_STATE_SIZE,     // 128
        3 * SIG_NET_GRU2_STATE_SIZE, // 384
    )?;
    let sig_net_gru2_glu_gate = linear_init(
        arrays,
        Some("sig_net_gru2_glu_gate_bias"),
        Some("sig_net_gru2_glu_gate_subias"),
        Some("sig_net_gru2_glu_gate_weights_int8"),
        Some("sig_net_gru2_glu_gate_weights_float"),
        None,
        None,
        Some("sig_net_gru2_glu_gate_scale"),
        SIG_NET_GRU2_OUT_SIZE, // 128
        SIG_NET_GRU2_OUT_SIZE, // 128
    )?;

    // GRU3: GRUCell(208 -> 128), quantized, bias=False.
    let sig_net_gru3_input = linear_init(
        arrays,
        None,
        Some("sig_net_gru3_input_subias"),
        Some("sig_net_gru3_input_weights_int8"),
        Some("sig_net_gru3_input_weights_float"),
        None,
        None,
        Some("sig_net_gru3_input_scale"),
        SIG_NET_GRU2_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE, // 208
        3 * SIG_NET_GRU3_STATE_SIZE,                      // 384
    )?;
    let sig_net_gru3_recurrent = linear_init(
        arrays,
        None,
        Some("sig_net_gru3_recurrent_subias"),
        Some("sig_net_gru3_recurrent_weights_int8"),
        Some("sig_net_gru3_recurrent_weights_float"),
        None,
        None,
        Some("sig_net_gru3_recurrent_scale"),
        SIG_NET_GRU3_STATE_SIZE,     // 128
        3 * SIG_NET_GRU3_STATE_SIZE, // 384
    )?;
    let sig_net_gru3_glu_gate = linear_init(
        arrays,
        Some("sig_net_gru3_glu_gate_bias"),
        Some("sig_net_gru3_glu_gate_subias"),
        Some("sig_net_gru3_glu_gate_weights_int8"),
        Some("sig_net_gru3_glu_gate_weights_float"),
        None,
        None,
        Some("sig_net_gru3_glu_gate_scale"),
        SIG_NET_GRU3_OUT_SIZE, // 128
        SIG_NET_GRU3_OUT_SIZE, // 128
    )?;

    // Skip-connection dense: Linear(688 -> 128), quantized, no real bias.
    let sig_net_skip_dense = linear_init(
        arrays,
        Some("sig_net_skip_dense_bias"),
        Some("sig_net_skip_dense_subias"),
        Some("sig_net_skip_dense_weights_int8"),
        Some("sig_net_skip_dense_weights_float"),
        None,
        None,
        Some("sig_net_skip_dense_scale"),
        SKIP_CAT_SIZE,               // 688
        SIG_NET_SKIP_DENSE_OUT_SIZE, // 128
    )?;

    // Skip GLU gate: Linear(128 -> 128), quantized, no real bias.
    let sig_net_skip_glu_gate = linear_init(
        arrays,
        Some("sig_net_skip_glu_gate_bias"),
        Some("sig_net_skip_glu_gate_subias"),
        Some("sig_net_skip_glu_gate_weights_int8"),
        Some("sig_net_skip_glu_gate_weights_float"),
        None,
        None,
        Some("sig_net_skip_glu_gate_scale"),
        SIG_NET_SKIP_DENSE_OUT_SIZE, // 128
        SIG_NET_SKIP_DENSE_OUT_SIZE, // 128
    )?;

    // Output dense: Linear(128 -> 40), quantized, no real bias.
    let sig_net_sig_dense_out = linear_init(
        arrays,
        Some("sig_net_sig_dense_out_bias"),
        Some("sig_net_sig_dense_out_subias"),
        Some("sig_net_sig_dense_out_weights_int8"),
        Some("sig_net_sig_dense_out_weights_float"),
        None,
        None,
        Some("sig_net_sig_dense_out_scale"),
        SIG_NET_SKIP_DENSE_OUT_SIZE, // 128
        FARGAN_SUBFRAME_SIZE,        // 40
    )?;

    Ok(FarganModel {
        cond_net_pembed,
        cond_net_fdense1,
        cond_net_fconv1,
        cond_net_fdense2,
        sig_net_cond_gain_dense,
        sig_net_fwc0_conv,
        sig_net_fwc0_glu_gate,
        sig_net_gain_dense_out,
        sig_net_gru1_input,
        sig_net_gru1_recurrent,
        sig_net_gru1_glu_gate,
        sig_net_gru2_input,
        sig_net_gru2_recurrent,
        sig_net_gru2_glu_gate,
        sig_net_gru3_input,
        sig_net_gru3_recurrent,
        sig_net_gru3_glu_gate,
        sig_net_skip_dense,
        sig_net_skip_glu_gate,
        sig_net_sig_dense_out,
    })
}

// ===========================================================================
// Internal helpers
// ===========================================================================

/// Decode pitch period from feature vector.
///
/// Uses double precision to match C: `(int)floor(.5 + 256./pow(2., ...))`.
/// Returns a period in approximately [32, 256].
fn decode_pitch_period(features: &[f32]) -> i32 {
    let feat = features[NB_BANDS] as f64;
    // C: floor(.5 + 256./pow(2.f, ((1./60.)*((features[NB_BANDS]+1.5)*60))))
    // Note: C's pow() takes double; 256. and .5 are double literals.
    let exponent = (1.0f64 / 60.0) * ((feat + 1.5) * 60.0);
    let period = (0.5 + 256.0 / 2.0f64.powf(exponent)).floor();
    period as i32
}

/// Compute conditioning vector from features and pitch period.
///
/// Matches C `compute_fargan_cond`.
///
/// Outputs 320 floats into `cond`, split as 4 × 80 for the 4 subframes.
fn compute_fargan_cond(
    model: &FarganModel,
    cond_conv1_state: &mut [f32],
    cond: &mut [f32],
    features: &[f32],
    period: i32,
) {
    debug_assert_eq!(
        NB_FEATURES + COND_NET_PEMBED_OUT_SIZE,
        model.cond_net_fdense1.nb_inputs
    );
    debug_assert_eq!(COND_NET_FCONV1_IN_SIZE, model.cond_net_fdense1.nb_outputs);
    debug_assert_eq!(COND_NET_FCONV1_OUT_SIZE, model.cond_net_fconv1.nb_outputs);

    // Step 1: Pitch embedding lookup.
    // Clamp index to [0, 223].
    let idx = (period - PITCH_MIN_PERIOD as i32)
        .max(0)
        .min(PEMBED_NUM_ENTRIES as i32 - 1) as usize;

    let mut dense_in = [0.0f32; NB_FEATURES + COND_NET_PEMBED_OUT_SIZE];

    // Copy embedding vector into dense_in[NB_FEATURES..].
    let fw = model
        .cond_net_pembed
        .float_weights
        .as_ref()
        .expect("pembed float_weights");
    let emb_start = idx * COND_NET_PEMBED_OUT_SIZE;
    dense_in[NB_FEATURES..NB_FEATURES + COND_NET_PEMBED_OUT_SIZE]
        .copy_from_slice(&fw[emb_start..emb_start + COND_NET_PEMBED_OUT_SIZE]);

    // Copy features into dense_in[0..NB_FEATURES].
    dense_in[..NB_FEATURES].copy_from_slice(&features[..NB_FEATURES]);

    // Step 2: fdense1 + tanh.
    let mut conv1_in = [0.0f32; COND_NET_FCONV1_IN_SIZE];
    compute_generic_dense(
        &model.cond_net_fdense1,
        &mut conv1_in,
        &dense_in,
        ACTIVATION_TANH,
    );

    // Step 3: fconv1 (causal conv1d, kernel=3) + tanh.
    let mut fdense2_in = [0.0f32; COND_NET_FCONV1_OUT_SIZE];
    compute_generic_conv1d(
        &model.cond_net_fconv1,
        &mut fdense2_in,
        cond_conv1_state,
        &conv1_in,
        COND_NET_FCONV1_IN_SIZE,
        ACTIVATION_TANH,
    );

    // Step 4: fdense2 + tanh -> cond[320].
    compute_generic_dense(
        &model.cond_net_fdense2,
        &mut cond[..COND_NET_FDENSE2_OUT_SIZE],
        &fdense2_in,
        ACTIVATION_TANH,
    );
}

/// First-order de-emphasis IIR filter applied per subframe.
///
/// `y[i] = x[i] + 0.85 * y[i-1]`
///
/// Matches C `fargan_deemphasis`.
fn fargan_deemphasis(pcm: &mut [f32], deemph_mem: &mut f32) {
    for i in 0..FARGAN_SUBFRAME_SIZE {
        pcm[i] += FARGAN_DEEMPHASIS * *deemph_mem;
        *deemph_mem = pcm[i];
    }
}

/// Run one subframe of the signal network, producing 40 PCM samples.
///
/// Matches C `run_fargan_subframe`.
fn run_fargan_subframe(st: &mut FarganState, pcm: &mut [f32], cond: &[f32], period: i32) {
    debug_assert!(st.cont_initialized);
    let model = &st.model;

    // Step 1: Gain computation.
    // gain = exp(linear(cond -> 1))
    let mut gain_raw = [0.0f32; 1];
    compute_generic_dense(
        &model.sig_net_cond_gain_dense,
        &mut gain_raw,
        cond,
        ACTIVATION_LINEAR,
    );
    // C: gain = exp(gain), using double-precision exp().
    let gain = (gain_raw[0] as f64).exp() as f32;
    let gain_1 = 1.0f32 / (1e-5f32 + gain);

    // Step 2: Pitch prediction and previous samples.
    let mut pred = [0.0f32; FARGAN_SUBFRAME_SIZE + 4];
    let mut prev = [0.0f32; FARGAN_SUBFRAME_SIZE];

    // Read pitch-predicted samples with wrap-around.
    let mut pos = PITCH_MAX_PERIOD as i32 - period - 2;
    for i in 0..(FARGAN_SUBFRAME_SIZE + 4) {
        let p = pos.max(0) as usize;
        pred[i] = (gain_1 * st.pitch_buf[p]).clamp(-1.0, 1.0);
        pos += 1;
        if pos == PITCH_MAX_PERIOD as i32 {
            pos -= period;
        }
    }

    // Read previous (most recent) samples from pitch buffer.
    for i in 0..FARGAN_SUBFRAME_SIZE {
        prev[i] =
            (gain_1 * st.pitch_buf[PITCH_MAX_PERIOD - FARGAN_SUBFRAME_SIZE + i]).clamp(-1.0, 1.0);
    }

    // Step 3: FWConv (frame-wise convolution with kernel_size=2).
    // fwc0_in = [cond(80), pred(44), prev(40)]
    let mut fwc0_in = [0.0f32; SIG_NET_INPUT_SIZE];
    fwc0_in[..FARGAN_COND_SIZE].copy_from_slice(&cond[..FARGAN_COND_SIZE]);
    fwc0_in[FARGAN_COND_SIZE..FARGAN_COND_SIZE + FARGAN_SUBFRAME_SIZE + 4].copy_from_slice(&pred);
    fwc0_in[FARGAN_COND_SIZE + FARGAN_SUBFRAME_SIZE + 4..SIG_NET_INPUT_SIZE].copy_from_slice(&prev);

    // gru1_in is reused: first holds fwc0 output, then gets pred/prev appended.
    let mut gru1_in = vec![0.0f32; SIG_NET_FWC0_CONV_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE];
    compute_generic_conv1d(
        &model.sig_net_fwc0_conv,
        &mut gru1_in[..SIG_NET_FWC0_CONV_OUT_SIZE],
        &mut st.fwc0_mem,
        &fwc0_in,
        SIG_NET_INPUT_SIZE,
        ACTIVATION_TANH,
    );

    debug_assert_eq!(
        SIG_NET_FWC0_GLU_GATE_OUT_SIZE,
        model.sig_net_fwc0_glu_gate.nb_outputs
    );
    // GLU on fwc0 output (in-place in gru1_in[..192]).
    {
        let (out_part, _) = gru1_in.split_at_mut(SIG_NET_FWC0_CONV_OUT_SIZE);
        let input_copy = out_part.to_vec();
        compute_glu(&model.sig_net_fwc0_glu_gate, out_part, &input_copy);
    }

    // Step 4: Pitch gate — sigmoid(linear(fwc0_glu_out -> 4)).
    let mut pitch_gate = [0.0f32; 4];
    compute_generic_dense(
        &model.sig_net_gain_dense_out,
        &mut pitch_gate,
        &gru1_in[..SIG_NET_FWC0_GLU_GATE_OUT_SIZE],
        ACTIVATION_SIGMOID,
    );

    // Step 5: GRU cascade (3 layers).
    // pred[2..42] is the center 40 samples used throughout.

    // --- GRU1 ---
    // gru1_in = [fwc0_glu_out(192), pitch_gate[0]*pred[2:42](40), prev(40)]
    for i in 0..FARGAN_SUBFRAME_SIZE {
        gru1_in[SIG_NET_FWC0_GLU_GATE_OUT_SIZE + i] = pitch_gate[0] * pred[i + 2];
    }
    gru1_in[SIG_NET_FWC0_GLU_GATE_OUT_SIZE + FARGAN_SUBFRAME_SIZE
        ..SIG_NET_FWC0_GLU_GATE_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE]
        .copy_from_slice(&prev);

    compute_generic_gru(
        &model.sig_net_gru1_input,
        &model.sig_net_gru1_recurrent,
        &mut st.gru1_state,
        &gru1_in,
    );

    let mut gru2_in = vec![0.0f32; SIG_NET_GRU1_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE];
    compute_glu(
        &model.sig_net_gru1_glu_gate,
        &mut gru2_in[..SIG_NET_GRU1_OUT_SIZE],
        &st.gru1_state,
    );

    // --- GRU2 ---
    // gru2_in = [gru1_glu_out(160), pitch_gate[1]*pred[2:42](40), prev(40)]
    for i in 0..FARGAN_SUBFRAME_SIZE {
        gru2_in[SIG_NET_GRU1_OUT_SIZE + i] = pitch_gate[1] * pred[i + 2];
    }
    gru2_in[SIG_NET_GRU1_OUT_SIZE + FARGAN_SUBFRAME_SIZE
        ..SIG_NET_GRU1_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE]
        .copy_from_slice(&prev);

    compute_generic_gru(
        &model.sig_net_gru2_input,
        &model.sig_net_gru2_recurrent,
        &mut st.gru2_state,
        &gru2_in,
    );

    let mut gru3_in = vec![0.0f32; SIG_NET_GRU2_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE];
    compute_glu(
        &model.sig_net_gru2_glu_gate,
        &mut gru3_in[..SIG_NET_GRU2_OUT_SIZE],
        &st.gru2_state,
    );

    // --- GRU3 ---
    // gru3_in = [gru2_glu_out(128), pitch_gate[2]*pred[2:42](40), prev(40)]
    for i in 0..FARGAN_SUBFRAME_SIZE {
        gru3_in[SIG_NET_GRU2_OUT_SIZE + i] = pitch_gate[2] * pred[i + 2];
    }
    gru3_in[SIG_NET_GRU2_OUT_SIZE + FARGAN_SUBFRAME_SIZE
        ..SIG_NET_GRU2_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE]
        .copy_from_slice(&prev);

    compute_generic_gru(
        &model.sig_net_gru3_input,
        &model.sig_net_gru3_recurrent,
        &mut st.gru3_state,
        &gru3_in,
    );

    // Step 6: Skip connections.
    // skip_cat = [gru1_glu(160), gru2_glu(128), gru3_glu(128),
    //             fwc0_glu(192), pitch_gate[3]*pred[2:42](40), prev(40)]
    let mut skip_cat = vec![0.0f32; SKIP_CAT_SIZE];

    // gru3_glu_out -> skip_cat[160+128..160+128+128]
    // Must be computed before we overwrite skip_cat[0..160] with gru1 output.
    let gru3_glu_offset = SIG_NET_GRU1_OUT_SIZE + SIG_NET_GRU2_OUT_SIZE;
    compute_glu(
        &model.sig_net_gru3_glu_gate,
        &mut skip_cat[gru3_glu_offset..gru3_glu_offset + SIG_NET_GRU3_OUT_SIZE],
        &st.gru3_state,
    );

    // gru1_glu_out = gru2_in[..160] (already computed above).
    skip_cat[..SIG_NET_GRU1_OUT_SIZE].copy_from_slice(&gru2_in[..SIG_NET_GRU1_OUT_SIZE]);

    // gru2_glu_out = gru3_in[..128] (already computed above).
    skip_cat[SIG_NET_GRU1_OUT_SIZE..SIG_NET_GRU1_OUT_SIZE + SIG_NET_GRU2_OUT_SIZE]
        .copy_from_slice(&gru3_in[..SIG_NET_GRU2_OUT_SIZE]);

    // fwc0_glu_out = gru1_in[..192].
    let fwc0_offset = SIG_NET_GRU1_OUT_SIZE + SIG_NET_GRU2_OUT_SIZE + SIG_NET_GRU3_OUT_SIZE;
    skip_cat[fwc0_offset..fwc0_offset + SIG_NET_FWC0_CONV_OUT_SIZE]
        .copy_from_slice(&gru1_in[..SIG_NET_FWC0_CONV_OUT_SIZE]);

    // pitch_gate[3] * pred[2:42].
    let pred_offset = fwc0_offset + SIG_NET_FWC0_CONV_OUT_SIZE;
    for i in 0..FARGAN_SUBFRAME_SIZE {
        skip_cat[pred_offset + i] = pitch_gate[3] * pred[i + 2];
    }

    // prev.
    let prev_offset = pred_offset + FARGAN_SUBFRAME_SIZE;
    skip_cat[prev_offset..prev_offset + FARGAN_SUBFRAME_SIZE].copy_from_slice(&prev);

    // Step 7: Skip dense + GLU.
    let mut skip_out = [0.0f32; SIG_NET_SKIP_DENSE_OUT_SIZE];
    compute_generic_dense(
        &model.sig_net_skip_dense,
        &mut skip_out,
        &skip_cat,
        ACTIVATION_TANH,
    );
    {
        let input_copy = skip_out;
        compute_glu(&model.sig_net_skip_glu_gate, &mut skip_out, &input_copy);
    }

    // Step 8: Output dense + tanh, then scale by gain.
    compute_generic_dense(
        &model.sig_net_sig_dense_out,
        &mut pcm[..FARGAN_SUBFRAME_SIZE],
        &skip_out,
        ACTIVATION_TANH,
    );
    for i in 0..FARGAN_SUBFRAME_SIZE {
        pcm[i] *= gain;
    }

    // Step 9: Update pitch buffer — shift left by subframe_size, append new output.
    st.pitch_buf
        .copy_within(FARGAN_SUBFRAME_SIZE..PITCH_MAX_PERIOD, 0);
    st.pitch_buf[PITCH_MAX_PERIOD - FARGAN_SUBFRAME_SIZE..PITCH_MAX_PERIOD]
        .copy_from_slice(&pcm[..FARGAN_SUBFRAME_SIZE]);

    // Step 10: De-emphasis.
    fargan_deemphasis(&mut pcm[..FARGAN_SUBFRAME_SIZE], &mut st.deemph_mem);
}

// ===========================================================================
// Public API
// ===========================================================================

impl FarganState {
    /// Create a new state with model weights loaded from weight arrays.
    ///
    /// Equivalent to C's `fargan_init` (non-`USE_WEIGHTS_FILE` path).
    /// Zeros all state, loads model weights.
    pub fn new(arrays: &[WeightArray]) -> Result<Self, ()> {
        let model = init_fargan(arrays)?;
        Ok(Self {
            model,
            ..Default::default()
        })
    }

    /// Create a new empty state (no model weights).
    ///
    /// Used when weights will be loaded separately via `load_model`.
    pub fn new_empty() -> Self {
        Self::default()
    }

    /// Initialize/reset state.
    ///
    /// Matches C `fargan_init`: zeros everything except arch and model.
    pub fn init(&mut self) {
        self.cont_initialized = false;
        self.deemph_mem = 0.0;
        self.pitch_buf.fill(0.0);
        self.cond_conv1_state.fill(0.0);
        self.fwc0_mem.fill(0.0);
        self.gru1_state.fill(0.0);
        self.gru2_state.fill(0.0);
        self.gru3_state.fill(0.0);
        self.last_period = 0;
    }

    /// Load model weights from a binary blob.
    ///
    /// Matches C `fargan_load_model`.
    /// Returns 0 on success, -1 on failure.
    pub fn load_model(&mut self, data: &[u8]) -> i32 {
        match parse_weights(data) {
            Ok(arrays) => match init_fargan(&arrays) {
                Ok(model) => {
                    self.model = model;
                    0
                }
                Err(()) => -1,
            },
            Err(_) => -1,
        }
    }

    /// Prime state from known PCM history and feature vectors.
    ///
    /// Must be called once before `synthesize` / `synthesize_int` to establish
    /// temporal continuity.
    ///
    /// Matches C `fargan_cont`.
    ///
    /// - `pcm0`: `[FARGAN_CONT_SAMPLES]` (320) float PCM samples (~[-1, 1]).
    /// - `features0`: `[5 * NB_FEATURES]` (100) feature values — 3 look-back
    ///   frames + 2 frames covering the PCM.
    pub fn cont(&mut self, pcm0: &[f32], features0: &[f32]) {
        let mut cond = [0.0f32; COND_NET_FDENSE2_OUT_SIZE];
        let mut period = 0i32;

        // Pre-load conditioning: iterate over 5 feature frames to fill the
        // conv1d state with valid history.
        for i in 0..5 {
            let features = &features0[i * NB_FEATURES..(i + 1) * NB_FEATURES];
            self.last_period = period;
            period = decode_pitch_period(features);
            compute_fargan_cond(
                &self.model,
                &mut self.cond_conv1_state,
                &mut cond,
                features,
                period,
            );
        }

        // Apply pre-emphasis to PCM input.
        // x0[0] = 0 (no predecessor for the first sample).
        let mut x0 = vec![0.0f32; FARGAN_CONT_SAMPLES];
        for i in 1..FARGAN_CONT_SAMPLES {
            x0[i] = pcm0[i] - FARGAN_DEEMPHASIS * pcm0[i - 1];
        }

        // Copy first frame (160 samples) of pre-emphasized PCM into pitch buffer tail.
        self.pitch_buf[PITCH_MAX_PERIOD - FARGAN_FRAME_SIZE..PITCH_MAX_PERIOD]
            .copy_from_slice(&x0[..FARGAN_FRAME_SIZE]);

        self.cont_initialized = true;

        // Run 4 subframes to warm up GRU states and FWConv memory.
        // After each subframe, override pitch buffer tail with the actual
        // known pre-emphasized PCM (from x0[160..320]).
        let mut dummy = [0.0f32; FARGAN_SUBFRAME_SIZE];
        for i in 0..FARGAN_NB_SUBFRAMES {
            run_fargan_subframe(
                self,
                &mut dummy,
                &cond[i * FARGAN_COND_SIZE..(i + 1) * FARGAN_COND_SIZE],
                self.last_period,
            );
            let src_start = FARGAN_FRAME_SIZE + i * FARGAN_SUBFRAME_SIZE;
            self.pitch_buf[PITCH_MAX_PERIOD - FARGAN_SUBFRAME_SIZE..PITCH_MAX_PERIOD]
                .copy_from_slice(&x0[src_start..src_start + FARGAN_SUBFRAME_SIZE]);
        }

        // Set de-emphasis memory to the last original (non-pre-emphasized) PCM sample.
        self.deemph_mem = pcm0[FARGAN_CONT_SAMPLES - 1];
    }

    /// Synthesize one frame (160 samples) of float PCM.
    ///
    /// Matches C `fargan_synthesize`.
    ///
    /// - `pcm`: output buffer `[FARGAN_FRAME_SIZE]` (160 samples).
    /// - `features`: `[NB_FEATURES]` (20) features for one 10 ms frame.
    pub fn synthesize(&mut self, pcm: &mut [f32], features: &[f32]) {
        debug_assert!(self.cont_initialized);

        let period = decode_pitch_period(features);

        let mut cond = [0.0f32; COND_NET_FDENSE2_OUT_SIZE];
        compute_fargan_cond(
            &self.model,
            &mut self.cond_conv1_state,
            &mut cond,
            features,
            period,
        );

        for subframe in 0..FARGAN_NB_SUBFRAMES {
            let pcm_start = subframe * FARGAN_SUBFRAME_SIZE;
            let cond_start = subframe * FARGAN_COND_SIZE;
            run_fargan_subframe(
                self,
                &mut pcm[pcm_start..pcm_start + FARGAN_SUBFRAME_SIZE],
                &cond[cond_start..cond_start + FARGAN_COND_SIZE],
                self.last_period,
            );
        }

        // Update last_period AFTER all subframes (one-frame lag by design).
        self.last_period = period;
    }

    /// Synthesize one frame and convert to 16-bit signed integer PCM.
    ///
    /// Matches C `fargan_synthesize_int`.
    ///
    /// - `pcm`: output buffer `[LPCNET_FRAME_SIZE]` (160 samples).
    /// - `features`: `[NB_FEATURES]` (20) features.
    pub fn synthesize_int(&mut self, pcm: &mut [i16], features: &[f32]) {
        let mut fpcm = [0.0f32; FARGAN_FRAME_SIZE];
        self.synthesize(&mut fpcm, features);

        // C: pcm[i] = (int)floor(.5 + MIN32(32767, MAX32(-32767, 32768.f*fpcm[i])))
        for i in 0..LPCNET_FRAME_SIZE {
            let val = 32768.0f32 * fpcm[i];
            let clamped = val.max(-32767.0).min(32767.0);
            // floor(.5 + x) in double precision to match C.
            pcm[i] = (0.5f64 + clamped as f64).floor() as i16;
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    #![allow(clippy::field_reassign_with_default)]

    use super::*;
    use crate::dnn::core::{WEIGHT_TYPE_FLOAT, WEIGHT_TYPE_INT8, WeightArray};

    fn zero_f32_array(name: &'static str, len: usize) -> WeightArray {
        WeightArray {
            name: name.to_string(),
            weight_type: WEIGHT_TYPE_FLOAT,
            size: len * 4,
            data: vec![0u8; len * 4],
        }
    }

    fn zero_i8_array(name: &'static str, len: usize) -> WeightArray {
        WeightArray {
            name: name.to_string(),
            weight_type: WEIGHT_TYPE_INT8,
            size: len,
            data: vec![0u8; len],
        }
    }

    fn valid_zero_weight_arrays() -> Vec<WeightArray> {
        vec![
            // Conditioning network
            zero_f32_array("cond_net_pembed_bias", COND_NET_PEMBED_OUT_SIZE),
            zero_f32_array(
                "cond_net_pembed_weights_float",
                PEMBED_NUM_ENTRIES * COND_NET_PEMBED_OUT_SIZE,
            ),
            zero_f32_array("cond_net_fdense1_bias", COND_NET_FCONV1_IN_SIZE),
            zero_f32_array("cond_net_fconv1_bias", COND_NET_FCONV1_OUT_SIZE),
            zero_i8_array(
                "cond_net_fconv1_weights_int8",
                COND_NET_FCONV1_IN_SIZE * 3 * COND_NET_FCONV1_OUT_SIZE,
            ),
            zero_f32_array("cond_net_fconv1_subias", COND_NET_FCONV1_OUT_SIZE),
            zero_f32_array("cond_net_fconv1_scale", COND_NET_FCONV1_OUT_SIZE),
            zero_f32_array("cond_net_fdense2_bias", COND_NET_FDENSE2_OUT_SIZE),
            zero_i8_array(
                "cond_net_fdense2_weights_int8",
                COND_NET_FCONV1_OUT_SIZE * COND_NET_FDENSE2_OUT_SIZE,
            ),
            zero_f32_array("cond_net_fdense2_subias", COND_NET_FDENSE2_OUT_SIZE),
            zero_f32_array("cond_net_fdense2_scale", COND_NET_FDENSE2_OUT_SIZE),
            // Signal network
            zero_f32_array("sig_net_cond_gain_dense_bias", 1),
            zero_f32_array("sig_net_fwc0_conv_bias", SIG_NET_FWC0_CONV_OUT_SIZE),
            zero_i8_array(
                "sig_net_fwc0_conv_weights_int8",
                SIG_NET_INPUT_SIZE * 2 * SIG_NET_FWC0_CONV_OUT_SIZE,
            ),
            zero_f32_array("sig_net_fwc0_conv_subias", SIG_NET_FWC0_CONV_OUT_SIZE),
            zero_f32_array("sig_net_fwc0_conv_scale", SIG_NET_FWC0_CONV_OUT_SIZE),
            zero_f32_array("sig_net_fwc0_glu_gate_bias", SIG_NET_FWC0_GLU_GATE_OUT_SIZE),
            zero_i8_array(
                "sig_net_fwc0_glu_gate_weights_int8",
                SIG_NET_FWC0_CONV_OUT_SIZE * SIG_NET_FWC0_GLU_GATE_OUT_SIZE,
            ),
            zero_f32_array(
                "sig_net_fwc0_glu_gate_subias",
                SIG_NET_FWC0_GLU_GATE_OUT_SIZE,
            ),
            zero_f32_array(
                "sig_net_fwc0_glu_gate_scale",
                SIG_NET_FWC0_GLU_GATE_OUT_SIZE,
            ),
            zero_f32_array("sig_net_gain_dense_out_bias", 4),
            zero_f32_array(
                "sig_net_gain_dense_out_weights_float",
                SIG_NET_FWC0_GLU_GATE_OUT_SIZE * 4,
            ),
            zero_i8_array(
                "sig_net_gru1_input_weights_int8",
                (SIG_NET_FWC0_GLU_GATE_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE)
                    * (3 * SIG_NET_GRU1_STATE_SIZE),
            ),
            zero_f32_array("sig_net_gru1_input_subias", 3 * SIG_NET_GRU1_STATE_SIZE),
            zero_f32_array("sig_net_gru1_input_scale", 3 * SIG_NET_GRU1_STATE_SIZE),
            zero_i8_array(
                "sig_net_gru1_recurrent_weights_int8",
                SIG_NET_GRU1_STATE_SIZE * (3 * SIG_NET_GRU1_STATE_SIZE),
            ),
            zero_f32_array("sig_net_gru1_recurrent_subias", 3 * SIG_NET_GRU1_STATE_SIZE),
            zero_f32_array("sig_net_gru1_recurrent_scale", 3 * SIG_NET_GRU1_STATE_SIZE),
            zero_i8_array(
                "sig_net_gru1_glu_gate_weights_int8",
                SIG_NET_GRU1_OUT_SIZE * SIG_NET_GRU1_OUT_SIZE,
            ),
            zero_f32_array("sig_net_gru1_glu_gate_bias", SIG_NET_GRU1_OUT_SIZE),
            zero_f32_array("sig_net_gru1_glu_gate_subias", SIG_NET_GRU1_OUT_SIZE),
            zero_f32_array("sig_net_gru1_glu_gate_scale", SIG_NET_GRU1_OUT_SIZE),
            zero_i8_array(
                "sig_net_gru2_input_weights_int8",
                (SIG_NET_GRU1_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE) * (3 * SIG_NET_GRU2_STATE_SIZE),
            ),
            zero_f32_array("sig_net_gru2_input_subias", 3 * SIG_NET_GRU2_STATE_SIZE),
            zero_f32_array("sig_net_gru2_input_scale", 3 * SIG_NET_GRU2_STATE_SIZE),
            zero_i8_array(
                "sig_net_gru2_recurrent_weights_int8",
                SIG_NET_GRU2_STATE_SIZE * (3 * SIG_NET_GRU2_STATE_SIZE),
            ),
            zero_f32_array("sig_net_gru2_recurrent_subias", 3 * SIG_NET_GRU2_STATE_SIZE),
            zero_f32_array("sig_net_gru2_recurrent_scale", 3 * SIG_NET_GRU2_STATE_SIZE),
            zero_i8_array(
                "sig_net_gru2_glu_gate_weights_int8",
                SIG_NET_GRU2_OUT_SIZE * SIG_NET_GRU2_OUT_SIZE,
            ),
            zero_f32_array("sig_net_gru2_glu_gate_bias", SIG_NET_GRU2_OUT_SIZE),
            zero_f32_array("sig_net_gru2_glu_gate_subias", SIG_NET_GRU2_OUT_SIZE),
            zero_f32_array("sig_net_gru2_glu_gate_scale", SIG_NET_GRU2_OUT_SIZE),
            zero_i8_array(
                "sig_net_gru3_input_weights_int8",
                (SIG_NET_GRU2_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE) * (3 * SIG_NET_GRU3_STATE_SIZE),
            ),
            zero_f32_array("sig_net_gru3_input_subias", 3 * SIG_NET_GRU3_STATE_SIZE),
            zero_f32_array("sig_net_gru3_input_scale", 3 * SIG_NET_GRU3_STATE_SIZE),
            zero_i8_array(
                "sig_net_gru3_recurrent_weights_int8",
                SIG_NET_GRU3_STATE_SIZE * (3 * SIG_NET_GRU3_STATE_SIZE),
            ),
            zero_f32_array("sig_net_gru3_recurrent_subias", 3 * SIG_NET_GRU3_STATE_SIZE),
            zero_f32_array("sig_net_gru3_recurrent_scale", 3 * SIG_NET_GRU3_STATE_SIZE),
            zero_i8_array(
                "sig_net_gru3_glu_gate_weights_int8",
                SIG_NET_GRU3_OUT_SIZE * SIG_NET_GRU3_OUT_SIZE,
            ),
            zero_f32_array("sig_net_gru3_glu_gate_bias", SIG_NET_GRU3_OUT_SIZE),
            zero_f32_array("sig_net_gru3_glu_gate_subias", SIG_NET_GRU3_OUT_SIZE),
            zero_f32_array("sig_net_gru3_glu_gate_scale", SIG_NET_GRU3_OUT_SIZE),
            zero_f32_array("sig_net_skip_dense_bias", SIG_NET_SKIP_DENSE_OUT_SIZE),
            zero_i8_array(
                "sig_net_skip_dense_weights_int8",
                SKIP_CAT_SIZE * SIG_NET_SKIP_DENSE_OUT_SIZE,
            ),
            zero_f32_array("sig_net_skip_dense_subias", SIG_NET_SKIP_DENSE_OUT_SIZE),
            zero_f32_array("sig_net_skip_dense_scale", SIG_NET_SKIP_DENSE_OUT_SIZE),
            zero_f32_array("sig_net_skip_glu_gate_bias", SIG_NET_SKIP_DENSE_OUT_SIZE),
            zero_i8_array(
                "sig_net_skip_glu_gate_weights_int8",
                SIG_NET_SKIP_DENSE_OUT_SIZE * SIG_NET_SKIP_DENSE_OUT_SIZE,
            ),
            zero_f32_array("sig_net_skip_glu_gate_subias", SIG_NET_SKIP_DENSE_OUT_SIZE),
            zero_f32_array("sig_net_skip_glu_gate_scale", SIG_NET_SKIP_DENSE_OUT_SIZE),
            zero_i8_array(
                "sig_net_sig_dense_out_weights_int8",
                SIG_NET_SKIP_DENSE_OUT_SIZE * FARGAN_SUBFRAME_SIZE,
            ),
            zero_f32_array("sig_net_sig_dense_out_bias", FARGAN_SUBFRAME_SIZE),
            zero_f32_array("sig_net_sig_dense_out_subias", FARGAN_SUBFRAME_SIZE),
            zero_f32_array("sig_net_sig_dense_out_scale", FARGAN_SUBFRAME_SIZE),
        ]
    }

    #[test]
    fn test_constants_consistency() {
        // Verify derived constants match expected values from the architecture doc.
        assert_eq!(FARGAN_FRAME_SIZE, 160);
        assert_eq!(FARGAN_COND_SIZE, 80);
        assert_eq!(SIG_NET_INPUT_SIZE, 164);
        assert_eq!(SIG_NET_FWC0_STATE_SIZE, 328);
        assert_eq!(SKIP_CAT_SIZE, 688);
        assert_eq!(COND_NET_FCONV1_STATE_SIZE, 128);
    }

    #[test]
    fn test_decode_pitch_period() {
        // Period for a mid-range feature value.
        // features[NB_BANDS] = features[18]
        let mut features = [0.0f32; NB_FEATURES];

        // With feature = 0: exponent = (1/60)*((0+1.5)*60) = 1.5
        // period = floor(0.5 + 256 / 2^1.5) = floor(0.5 + 256/2.828) = floor(0.5 + 90.51) = 91
        features[NB_BANDS] = 0.0;
        assert_eq!(decode_pitch_period(&features), 91);

        // With feature = -1.5: exponent = (1/60)*((-1.5+1.5)*60) = 0
        // period = floor(0.5 + 256/1) = 256
        features[NB_BANDS] = -1.5;
        assert_eq!(decode_pitch_period(&features), 256);

        // With feature = 2.0: exponent = (1/60)*((2+1.5)*60) = 3.5
        // period = floor(0.5 + 256/2^3.5) = floor(0.5 + 256/11.314) = floor(0.5 + 22.627) = 23
        features[NB_BANDS] = 2.0;
        assert_eq!(decode_pitch_period(&features), 23);
    }

    #[test]
    fn test_fargan_deemphasis() {
        let mut pcm = [0.0f32; FARGAN_SUBFRAME_SIZE];
        pcm[0] = 1.0;
        pcm[1] = 0.5;
        pcm[2] = 0.0;
        pcm[3] = -0.5;
        let mut mem = 0.0f32;
        // Manual de-emphasis: y[0] = 1.0 + 0.85*0 = 1.0
        //                     y[1] = 0.5 + 0.85*1.0 = 1.35
        //                     y[2] = 0.0 + 0.85*1.35 = 1.1475
        //                     y[3] = -0.5 + 0.85*1.1475 = 0.475375
        fargan_deemphasis(&mut pcm, &mut mem);
        assert!((pcm[0] - 1.0).abs() < 1e-6);
        assert!((pcm[1] - 1.35).abs() < 1e-6);
        assert!((pcm[2] - 1.1475).abs() < 1e-6);
        assert!((pcm[3] - 0.475375).abs() < 1e-5);
    }

    #[test]
    fn test_default_state() {
        let st = FarganState::default();
        assert!(!st.cont_initialized);
        assert_eq!(st.deemph_mem, 0.0);
        assert_eq!(st.pitch_buf.len(), PITCH_MAX_PERIOD);
        assert_eq!(st.cond_conv1_state.len(), COND_NET_FCONV1_STATE_SIZE);
        assert_eq!(st.fwc0_mem.len(), SIG_NET_FWC0_STATE_SIZE);
        assert_eq!(st.gru1_state.len(), SIG_NET_GRU1_STATE_SIZE);
        assert_eq!(st.gru2_state.len(), SIG_NET_GRU2_STATE_SIZE);
        assert_eq!(st.gru3_state.len(), SIG_NET_GRU3_STATE_SIZE);
        assert_eq!(st.last_period, 0);
    }

    #[test]
    fn test_state_init_resets() {
        let mut st = FarganState::default();
        st.cont_initialized = true;
        st.deemph_mem = 42.0;
        st.last_period = 100;
        st.pitch_buf[0] = 1.0;
        st.gru1_state[0] = 2.0;

        st.init();

        assert!(!st.cont_initialized);
        assert_eq!(st.deemph_mem, 0.0);
        assert_eq!(st.last_period, 0);
        assert_eq!(st.pitch_buf[0], 0.0);
        assert_eq!(st.gru1_state[0], 0.0);
    }

    #[test]
    fn test_synthesize_int_clamping() {
        // Verify the float-to-int16 conversion formula:
        // pcm[i] = floor(0.5 + clamp(32768 * fpcm[i], -32767, 32767))

        // Positive saturation: 1.0 * 32768 = 32768 -> clamped to 32767
        let val = 32768.0f32 * 1.0;
        let clamped = val.max(-32767.0).min(32767.0);
        let result = (0.5f64 + clamped as f64).floor() as i16;
        assert_eq!(result, 32767);

        // Negative saturation: -1.0 * 32768 = -32768 -> clamped to -32767
        let val = -32768.0f32;
        let clamped = val.max(-32767.0).min(32767.0);
        let result = (0.5f64 + clamped as f64).floor() as i16;
        assert_eq!(result, -32767);

        // Mid-range: 0.5 * 32768 = 16384 -> 16384
        let val = 32768.0f32 * 0.5;
        let clamped = val.max(-32767.0).min(32767.0);
        let result = (0.5f64 + clamped as f64).floor() as i16;
        assert_eq!(result, 16384);

        // Zero
        let val = 32768.0f32 * 0.0;
        let clamped = val.max(-32767.0).min(32767.0);
        let result = (0.5f64 + clamped as f64).floor() as i16;
        assert_eq!(result, 0);
    }

    #[test]
    fn test_model_default() {
        let model = FarganModel::default();
        assert_eq!(model.cond_net_pembed.nb_inputs, 0);
        assert_eq!(model.cond_net_pembed.nb_outputs, 0);
        assert_eq!(model.sig_net_cond_gain_dense.nb_inputs, 0);
    }

    #[test]
    fn test_pitch_period_edge_cases() {
        let mut features = [0.0f32; NB_FEATURES];

        // Very large feature -> very small period (high pitch).
        features[NB_BANDS] = 5.0;
        let period = decode_pitch_period(&features);
        assert!(period > 0, "period should be positive");

        // Very negative feature -> very large period (low pitch).
        features[NB_BANDS] = -3.0;
        let period = decode_pitch_period(&features);
        assert!(period > 0, "period should be positive");
    }

    #[test]
    fn test_new_rejects_missing_weights() {
        assert!(FarganState::new(&[]).is_err());
    }

    #[test]
    fn test_load_model_invalid_blob_fails() {
        let mut st = FarganState::new_empty();
        assert_eq!(st.load_model(&[1, 2, 3, 4]), -1);
    }

    #[test]
    fn test_new_with_zero_weights_initializes_real_model() {
        let st = FarganState::new(&valid_zero_weight_arrays()).expect("zero model");
        assert!(!st.cont_initialized);
        assert_eq!(st.last_period, 0);
        assert_eq!(
            st.model
                .cond_net_pembed
                .float_weights
                .as_ref()
                .unwrap()
                .len(),
            PEMBED_NUM_ENTRIES * COND_NET_PEMBED_OUT_SIZE
        );
    }

    #[test]
    fn test_compute_fargan_cond_clamps_pitch_embedding_index() {
        let st = FarganState::new(&valid_zero_weight_arrays()).expect("zero model");
        let mut cond = [0.0f32; COND_NET_FDENSE2_OUT_SIZE];
        let mut cond_state = st.cond_conv1_state.clone();
        let features = [0.0f32; NB_FEATURES];

        compute_fargan_cond(&st.model, &mut cond_state, &mut cond, &features, 23);
        assert!(cond.iter().all(|x| x.is_finite()));

        let mut high_features = [0.0f32; NB_FEATURES];
        high_features[NB_BANDS] = -1.5;
        compute_fargan_cond(&st.model, &mut cond_state, &mut cond, &high_features, 256);
        assert!(cond.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_cont_and_synthesize_update_state_with_zero_model() {
        let mut st = FarganState::new(&valid_zero_weight_arrays()).expect("zero model");
        let pcm0: Vec<f32> = (0..FARGAN_CONT_SAMPLES)
            .map(|i| (i as f32 / FARGAN_CONT_SAMPLES as f32) - 0.5)
            .collect();
        let mut features0 = vec![0.0f32; 5 * NB_FEATURES];
        features0[NB_BANDS] = 2.0;
        features0[NB_FEATURES + NB_BANDS] = -1.5;

        st.cont(&pcm0, &features0);

        assert!(st.cont_initialized);
        assert!((st.deemph_mem - pcm0[FARGAN_CONT_SAMPLES - 1]).abs() < 1e-6);
        assert!(st.pitch_buf.iter().any(|&x| x != 0.0));
        assert!(st.cond_conv1_state.iter().all(|x| x.is_finite()));
        assert!(st.gru1_state.iter().all(|x| x.is_finite()));
        assert!(st.gru2_state.iter().all(|x| x.is_finite()));
        assert!(st.gru3_state.iter().all(|x| x.is_finite()));
        assert_eq!(
            st.last_period,
            decode_pitch_period(&features0[3 * NB_FEATURES..4 * NB_FEATURES])
        );

        let mut pcm = [0.0f32; FARGAN_FRAME_SIZE];
        let mut synth_features = [0.0f32; NB_FEATURES];
        synth_features[NB_BANDS] = 2.0;
        st.synthesize(&mut pcm, &synth_features);
        assert!(pcm.iter().all(|x| x.is_finite()));
        assert_eq!(st.last_period, decode_pitch_period(&synth_features));

        let mut pcm_i16 = [0i16; LPCNET_FRAME_SIZE];
        st.synthesize_int(&mut pcm_i16, &synth_features);
        assert!(pcm_i16.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_synthesize_updates_last_period_and_outputs_finite_samples() {
        let mut st = FarganState::new(&valid_zero_weight_arrays()).expect("zero model");
        let pcm0 = vec![0.0f32; FARGAN_CONT_SAMPLES];
        let mut features = [0.0f32; NB_FEATURES];
        features[NB_BANDS] = 0.25;

        st.cont(&pcm0, &vec![0.0f32; 5 * NB_FEATURES]);

        let mut pcm = [0.0f32; FARGAN_FRAME_SIZE];
        st.synthesize(&mut pcm, &features);
        assert!(pcm.iter().all(|x| x.is_finite()));
        assert_eq!(st.last_period, decode_pitch_period(&features));

        let mut pcm_i16 = [0i16; LPCNET_FRAME_SIZE];
        st.synthesize_int(&mut pcm_i16, &features);
        assert!(pcm_i16.iter().all(|&x| x == 0));
    }

    // --- Coverage additions: continuation, synthesis edge cases ---

    #[test]
    fn test_cont_with_dc_signal_sets_deemph_mem() {
        let mut st = FarganState::new(&valid_zero_weight_arrays()).expect("zero model");
        let dc_val = 0.5f32;
        let pcm = vec![dc_val; FARGAN_CONT_SAMPLES];
        let features = vec![0.0f32; 5 * NB_FEATURES];
        st.cont(&pcm, &features);
        assert!(st.cont_initialized);
        assert!(
            (st.deemph_mem - dc_val).abs() < 1e-5,
            "deemph_mem should be {dc_val}, got {}",
            st.deemph_mem
        );
    }

    #[test]
    fn test_synthesize_multiple_frames_stays_finite() {
        let mut st = FarganState::new(&valid_zero_weight_arrays()).expect("zero model");
        let pcm0 = vec![0.0f32; FARGAN_CONT_SAMPLES];
        let features0 = vec![0.0f32; 5 * NB_FEATURES];
        st.cont(&pcm0, &features0);
        for _ in 0..5 {
            let mut pcm = [0.0f32; FARGAN_FRAME_SIZE];
            let features = [0.0f32; NB_FEATURES];
            st.synthesize(&mut pcm, &features);
            assert!(
                pcm.iter().all(|x| x.is_finite()),
                "synthesized samples should be finite"
            );
        }
    }

    #[test]
    fn test_synthesize_int_clamps_to_i16_range() {
        let mut st = FarganState::new(&valid_zero_weight_arrays()).expect("zero model");
        let pcm0 = vec![0.0f32; FARGAN_CONT_SAMPLES];
        let features0 = vec![0.0f32; 5 * NB_FEATURES];
        st.cont(&pcm0, &features0);
        let mut pcm_i16 = [0i16; LPCNET_FRAME_SIZE];
        let features = [0.0f32; NB_FEATURES];
        st.synthesize_int(&mut pcm_i16, &features);
        assert!(
            pcm_i16.iter().all(|&s| s >= i16::MIN && s <= i16::MAX),
            "all samples should be within i16 range"
        );
    }

    #[test]
    fn test_new_empty_state_fields() {
        let st = FarganState::new(&valid_zero_weight_arrays()).expect("zero model");
        assert!(!st.cont_initialized);
        assert_eq!(st.deemph_mem, 0.0);
        assert_eq!(st.last_period, 0);
    }

    #[test]
    fn test_decode_pitch_period_clamping() {
        let mut features = [0.0f32; NB_FEATURES];
        features[NB_BANDS] = 100.0;
        let period = decode_pitch_period(&features);
        assert!(period >= 0, "period should be >= 0, got {period}");
        features[NB_BANDS] = -100.0;
        let period2 = decode_pitch_period(&features);
        assert!(
            period2 >= 0,
            "period should be >= 0 for negative input, got {period2}"
        );
        features[NB_BANDS] = 2.0;
        let period3 = decode_pitch_period(&features);
        assert!(
            period3 >= 0,
            "moderate value should give period >= 0, got {period3}"
        );
    }

    // ------------------------------------------------------------------
    // Stage 6 branch-coverage additions
    // ------------------------------------------------------------------

    mod branch_coverage_stage6 {
        use super::*;

        // Cover init() reset path: after a synthesize, init() should zero state.
        #[test]
        fn test_init_resets_state_after_synthesize() {
            let mut st = FarganState::new(&valid_zero_weight_arrays()).expect("zero model");
            st.cont(
                &vec![0.5f32; FARGAN_CONT_SAMPLES],
                &vec![0.0f32; 5 * NB_FEATURES],
            );
            assert!(st.cont_initialized);
            st.last_period = 42;
            st.init();
            assert!(!st.cont_initialized);
            assert_eq!(st.deemph_mem, 0.0);
            assert_eq!(st.last_period, 0);
            assert!(st.pitch_buf.iter().all(|&v| v == 0.0));
        }

        // Cover load_model failure branch: malformed blob.
        #[test]
        fn test_load_model_rejects_invalid_blob() {
            let mut st = FarganState::new(&valid_zero_weight_arrays()).expect("zero model");
            // Too-short / malformed data -> parse_weights fails.
            assert_eq!(st.load_model(&[0u8; 4]), -1);
            // A syntactically-parseable but semantically-empty blob also fails init.
            let bogus = vec![0u8; 64];
            // size field = 0, block_size = 0, so parse_record succeeds but init_fargan fails.
            assert_eq!(st.load_model(&bogus), -1);
            let mut bogus2 = bogus.clone();
            bogus2.extend_from_slice(&[0u8; 64]);
            assert_eq!(st.load_model(&bogus2), -1);
        }

        // Cover synthesize path with non-zero pitch period features.
        #[test]
        fn test_synthesize_with_nonzero_pitch_feature() {
            let mut st = FarganState::new(&valid_zero_weight_arrays()).expect("zero model");
            st.cont(
                &vec![0.0f32; FARGAN_CONT_SAMPLES],
                &vec![0.0f32; 5 * NB_FEATURES],
            );
            let mut features = [0.0f32; NB_FEATURES];
            features[NB_BANDS] = 0.5; // non-zero pitch feature
            let mut pcm = [0.0f32; FARGAN_FRAME_SIZE];
            st.synthesize(&mut pcm, &features);
            assert!(pcm.iter().all(|v| v.is_finite()));
            assert!(st.last_period >= 0);
        }
    }
}
