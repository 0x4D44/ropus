//! DRED (Deep REDundancy) — skeleton.
//!
//! Mirrors the C reference:
//! - `reference/dnn/dred_encoder.{c,h}`
//! - `reference/dnn/dred_decoder.{c,h}`
//! - `reference/dnn/dred_rdovae_enc.{c,h}` (+ `_data.{c,h}`)
//! - `reference/dnn/dred_rdovae_dec.{c,h}` (+ `_data.{c,h}`)
//!
//! This stage (8.3 in the DRED-port HLD) lands only the data model:
//! layer structs, state structs, and the blob-driven init paths. No
//! forward-pass logic yet — `dred_rdovae_encode_dframe` lands in 8.4
//! and `dred_rdovae_decode_qframe` in 8.5.

use super::core::{
    ACTIVATION_LINEAR, ACTIVATION_TANH, LinearLayer, WeightArray, compute_activation,
    compute_generic_conv1d, compute_generic_conv1d_dilation, compute_generic_dense,
    compute_generic_gru, compute_glu, linear_init, parse_weights,
};
use super::embedded_weights::WEIGHTS_BLOB;
use super::lpcnet::{LPCNetEncState, NB_TOTAL_FEATURES};
use crate::celt::range_coder::RangeEncoder;
use crate::dnn::dred_stats::{
    dred_latent_dead_zone_q8, dred_latent_p0_q8, dred_latent_quant_scales_q8, dred_latent_r_q8,
    dred_state_dead_zone_q8, dred_state_p0_q8, dred_state_quant_scales_q8, dred_state_r_q8,
};
use crate::types::float2int16;

// ===========================================================================
// Constants (match reference/dnn/dred_config.h + dred_rdovae_constants.h)
// ===========================================================================

pub const DRED_EXTENSION_ID: u32 = 126;
pub const DRED_EXPERIMENTAL_VERSION: u32 = 12;
pub const DRED_EXPERIMENTAL_BYTES: usize = 2;

pub const DRED_MIN_BYTES: usize = 8;
pub const DRED_SILK_ENCODER_DELAY: i32 = 79 + 12 - 80;
pub const DRED_FRAME_SIZE: usize = 160;
pub const DRED_DFRAME_SIZE: usize = 2 * DRED_FRAME_SIZE;
pub const DRED_MAX_DATA_SIZE: usize = 1000;
pub const DRED_ENC_Q0: i32 = 6;
pub const DRED_ENC_Q1: i32 = 15;
pub const DRED_MAX_LATENTS: usize = 26;
pub const DRED_NUM_REDUNDANCY_FRAMES: usize = 2 * DRED_MAX_LATENTS;
pub const DRED_MAX_FRAMES: usize = 4 * DRED_MAX_LATENTS;

pub const DRED_NUM_FEATURES: usize = 20;
pub const DRED_LATENT_DIM: usize = 25;
pub const DRED_STATE_DIM: usize = 50;
pub const DRED_PADDED_LATENT_DIM: usize = 32;
pub const DRED_PADDED_STATE_DIM: usize = 56;
pub const DRED_NUM_QUANTIZATION_LEVELS: usize = 16;

pub const RESAMPLING_ORDER: usize = 8;

/// Stack-buffer ceiling for conv1d `tmp[]` in the RDOVAE encoder / decoder.
/// Matches C `DRED_MAX_CONV_INPUTS` in `dred_rdovae_constants.h`.
pub const DRED_MAX_CONV_INPUTS: usize = 128;

// Encoder dimensions (dred_rdovae_enc_data.h).
const ENC_DENSE1_OUT_SIZE: usize = 64;
const ENC_ZDENSE_OUT_SIZE: usize = 32;
const GDENSE2_OUT_SIZE: usize = 56;
const GDENSE1_OUT_SIZE: usize = 128;
const ENC_CONV_DENSE1_OUT_SIZE: usize = 64;
const ENC_CONV_DENSE2_OUT_SIZE: usize = 64;
const ENC_CONV_DENSE3_OUT_SIZE: usize = 64;
const ENC_CONV_DENSE4_OUT_SIZE: usize = 64;
const ENC_CONV_DENSE5_OUT_SIZE: usize = 64;

const ENC_GRU1_OUT_SIZE: usize = 32;
const ENC_GRU2_OUT_SIZE: usize = 32;
const ENC_GRU3_OUT_SIZE: usize = 32;
const ENC_GRU4_OUT_SIZE: usize = 32;
const ENC_GRU5_OUT_SIZE: usize = 32;

const ENC_GRU1_STATE_SIZE: usize = 32;
const ENC_GRU2_STATE_SIZE: usize = 32;
const ENC_GRU3_STATE_SIZE: usize = 32;
const ENC_GRU4_STATE_SIZE: usize = 32;
const ENC_GRU5_STATE_SIZE: usize = 32;

const ENC_CONV1_OUT_SIZE: usize = 64;
const ENC_CONV2_OUT_SIZE: usize = 64;
const ENC_CONV3_OUT_SIZE: usize = 64;
const ENC_CONV4_OUT_SIZE: usize = 64;
const ENC_CONV5_OUT_SIZE: usize = 64;

const ENC_CONV1_IN_SIZE: usize = 64;
const ENC_CONV2_IN_SIZE: usize = 64;
const ENC_CONV3_IN_SIZE: usize = 64;
const ENC_CONV4_IN_SIZE: usize = 64;
const ENC_CONV5_IN_SIZE: usize = 64;

const ENC_CONV1_STATE_SIZE: usize = 64;
const ENC_CONV2_STATE_SIZE: usize = 64;
const ENC_CONV3_STATE_SIZE: usize = 64;
const ENC_CONV4_STATE_SIZE: usize = 64;
const ENC_CONV5_STATE_SIZE: usize = 64;

// Decoder dimensions (dred_rdovae_dec_data.h).
const DEC_DENSE1_OUT_SIZE: usize = 96;
const DEC_GLU_OUT_SIZE: usize = 64;
const DEC_HIDDEN_INIT_OUT_SIZE: usize = 128;
const DEC_OUTPUT_OUT_SIZE: usize = 80;
const DEC_CONV_DENSE_OUT_SIZE: usize = 32;
const DEC_GRU_INIT_OUT_SIZE: usize = 320;
const DEC_GRU_STATE_SIZE: usize = 64;

// Per-layer decoder GRU/conv dimensions. Uniform sizes in the current
// upstream checkpoint; named explicitly so the forward pass body mirrors
// the C source one-to-one.
const DEC_GRU1_OUT_SIZE: usize = 64;
const DEC_GRU2_OUT_SIZE: usize = 64;
const DEC_GRU3_OUT_SIZE: usize = 64;
const DEC_GRU4_OUT_SIZE: usize = 64;
const DEC_GRU5_OUT_SIZE: usize = 64;

const DEC_GRU1_STATE_SIZE: usize = 64;
const DEC_GRU2_STATE_SIZE: usize = 64;
const DEC_GRU3_STATE_SIZE: usize = 64;
const DEC_GRU4_STATE_SIZE: usize = 64;
const DEC_GRU5_STATE_SIZE: usize = 64;

const DEC_CONV1_OUT_SIZE: usize = 32;
const DEC_CONV2_OUT_SIZE: usize = 32;
const DEC_CONV3_OUT_SIZE: usize = 32;
const DEC_CONV4_OUT_SIZE: usize = 32;
const DEC_CONV5_OUT_SIZE: usize = 32;

const DEC_CONV1_IN_SIZE: usize = 32;
const DEC_CONV2_IN_SIZE: usize = 32;
const DEC_CONV3_IN_SIZE: usize = 32;
const DEC_CONV4_IN_SIZE: usize = 32;
const DEC_CONV5_IN_SIZE: usize = 32;

const DEC_CONV_DENSE1_OUT_SIZE: usize = 32;
const DEC_CONV_DENSE2_OUT_SIZE: usize = 32;
const DEC_CONV_DENSE3_OUT_SIZE: usize = 32;
const DEC_CONV_DENSE4_OUT_SIZE: usize = 32;
const DEC_CONV_DENSE5_OUT_SIZE: usize = 32;

const DEC_CONV_STATE_SIZE: usize = 32;

// ===========================================================================
// RDOVAE encoder model (struct-of-LinearLayer)
// ===========================================================================

/// RDOVAE encoder weights. Matches C `struct RDOVAEEnc`
/// (`reference/dnn/dred_rdovae_enc_data.h`).
#[derive(Clone, Debug, Default)]
pub struct RDOVAEEnc {
    pub enc_dense1: LinearLayer,
    pub enc_zdense: LinearLayer,
    pub gdense2: LinearLayer,
    pub gdense1: LinearLayer,
    pub enc_conv_dense1: LinearLayer,
    pub enc_conv_dense2: LinearLayer,
    pub enc_conv_dense3: LinearLayer,
    pub enc_conv_dense4: LinearLayer,
    pub enc_conv_dense5: LinearLayer,
    pub enc_gru1_input: LinearLayer,
    pub enc_gru1_recurrent: LinearLayer,
    pub enc_gru2_input: LinearLayer,
    pub enc_gru2_recurrent: LinearLayer,
    pub enc_gru3_input: LinearLayer,
    pub enc_gru3_recurrent: LinearLayer,
    pub enc_gru4_input: LinearLayer,
    pub enc_gru4_recurrent: LinearLayer,
    pub enc_gru5_input: LinearLayer,
    pub enc_gru5_recurrent: LinearLayer,
    pub enc_conv1: LinearLayer,
    pub enc_conv2: LinearLayer,
    pub enc_conv3: LinearLayer,
    pub enc_conv4: LinearLayer,
    pub enc_conv5: LinearLayer,
}

/// Initialise the RDOVAE encoder from a parsed weight blob.
/// Matches C `init_rdovaeenc` in `dred_rdovae_enc_data.c`.
pub fn init_rdovaeenc(arrays: &[WeightArray]) -> Result<RDOVAEEnc, ()> {
    // Plain float dense.
    let enc_dense1 = linear_init(
        arrays,
        Some("enc_dense1_bias"),
        None,
        None,
        Some("enc_dense1_weights_float"),
        None,
        None,
        None,
        40,
        ENC_DENSE1_OUT_SIZE,
    )?;

    // Dense int8 (no sparse idx).
    let enc_zdense = linear_init(
        arrays,
        Some("enc_zdense_bias"),
        Some("enc_zdense_subias"),
        Some("enc_zdense_weights_int8"),
        Some("enc_zdense_weights_float"),
        None,
        None,
        Some("enc_zdense_scale"),
        544,
        ENC_ZDENSE_OUT_SIZE,
    )?;
    let gdense2 = linear_init(
        arrays,
        Some("gdense2_bias"),
        Some("gdense2_subias"),
        Some("gdense2_weights_int8"),
        Some("gdense2_weights_float"),
        None,
        None,
        Some("gdense2_scale"),
        128,
        GDENSE2_OUT_SIZE,
    )?;

    // Sparse int8 (weights_idx).
    let gdense1 = linear_init(
        arrays,
        Some("gdense1_bias"),
        Some("gdense1_subias"),
        Some("gdense1_weights_int8"),
        Some("gdense1_weights_float"),
        Some("gdense1_weights_idx"),
        None,
        Some("gdense1_scale"),
        544,
        GDENSE1_OUT_SIZE,
    )?;

    let enc_conv_dense1 = linear_init(
        arrays,
        Some("enc_conv_dense1_bias"),
        Some("enc_conv_dense1_subias"),
        Some("enc_conv_dense1_weights_int8"),
        Some("enc_conv_dense1_weights_float"),
        Some("enc_conv_dense1_weights_idx"),
        None,
        Some("enc_conv_dense1_scale"),
        96,
        ENC_CONV_DENSE1_OUT_SIZE,
    )?;
    let enc_conv_dense2 = linear_init(
        arrays,
        Some("enc_conv_dense2_bias"),
        Some("enc_conv_dense2_subias"),
        Some("enc_conv_dense2_weights_int8"),
        Some("enc_conv_dense2_weights_float"),
        Some("enc_conv_dense2_weights_idx"),
        None,
        Some("enc_conv_dense2_scale"),
        192,
        ENC_CONV_DENSE2_OUT_SIZE,
    )?;
    let enc_conv_dense3 = linear_init(
        arrays,
        Some("enc_conv_dense3_bias"),
        Some("enc_conv_dense3_subias"),
        Some("enc_conv_dense3_weights_int8"),
        Some("enc_conv_dense3_weights_float"),
        Some("enc_conv_dense3_weights_idx"),
        None,
        Some("enc_conv_dense3_scale"),
        288,
        ENC_CONV_DENSE3_OUT_SIZE,
    )?;
    let enc_conv_dense4 = linear_init(
        arrays,
        Some("enc_conv_dense4_bias"),
        Some("enc_conv_dense4_subias"),
        Some("enc_conv_dense4_weights_int8"),
        Some("enc_conv_dense4_weights_float"),
        Some("enc_conv_dense4_weights_idx"),
        None,
        Some("enc_conv_dense4_scale"),
        384,
        ENC_CONV_DENSE4_OUT_SIZE,
    )?;
    let enc_conv_dense5 = linear_init(
        arrays,
        Some("enc_conv_dense5_bias"),
        Some("enc_conv_dense5_subias"),
        Some("enc_conv_dense5_weights_int8"),
        Some("enc_conv_dense5_weights_float"),
        Some("enc_conv_dense5_weights_idx"),
        None,
        Some("enc_conv_dense5_scale"),
        480,
        ENC_CONV_DENSE5_OUT_SIZE,
    )?;

    // GRU input (sparse idx) + recurrent (dense int8, no idx).
    let enc_gru1_input = linear_init(
        arrays,
        Some("enc_gru1_input_bias"),
        Some("enc_gru1_input_subias"),
        Some("enc_gru1_input_weights_int8"),
        Some("enc_gru1_input_weights_float"),
        Some("enc_gru1_input_weights_idx"),
        None,
        Some("enc_gru1_input_scale"),
        64,
        96,
    )?;
    let enc_gru1_recurrent = linear_init(
        arrays,
        Some("enc_gru1_recurrent_bias"),
        Some("enc_gru1_recurrent_subias"),
        Some("enc_gru1_recurrent_weights_int8"),
        Some("enc_gru1_recurrent_weights_float"),
        None,
        None,
        Some("enc_gru1_recurrent_scale"),
        32,
        96,
    )?;
    let enc_gru2_input = linear_init(
        arrays,
        Some("enc_gru2_input_bias"),
        Some("enc_gru2_input_subias"),
        Some("enc_gru2_input_weights_int8"),
        Some("enc_gru2_input_weights_float"),
        Some("enc_gru2_input_weights_idx"),
        None,
        Some("enc_gru2_input_scale"),
        160,
        96,
    )?;
    let enc_gru2_recurrent = linear_init(
        arrays,
        Some("enc_gru2_recurrent_bias"),
        Some("enc_gru2_recurrent_subias"),
        Some("enc_gru2_recurrent_weights_int8"),
        Some("enc_gru2_recurrent_weights_float"),
        None,
        None,
        Some("enc_gru2_recurrent_scale"),
        32,
        96,
    )?;
    let enc_gru3_input = linear_init(
        arrays,
        Some("enc_gru3_input_bias"),
        Some("enc_gru3_input_subias"),
        Some("enc_gru3_input_weights_int8"),
        Some("enc_gru3_input_weights_float"),
        Some("enc_gru3_input_weights_idx"),
        None,
        Some("enc_gru3_input_scale"),
        256,
        96,
    )?;
    let enc_gru3_recurrent = linear_init(
        arrays,
        Some("enc_gru3_recurrent_bias"),
        Some("enc_gru3_recurrent_subias"),
        Some("enc_gru3_recurrent_weights_int8"),
        Some("enc_gru3_recurrent_weights_float"),
        None,
        None,
        Some("enc_gru3_recurrent_scale"),
        32,
        96,
    )?;
    let enc_gru4_input = linear_init(
        arrays,
        Some("enc_gru4_input_bias"),
        Some("enc_gru4_input_subias"),
        Some("enc_gru4_input_weights_int8"),
        Some("enc_gru4_input_weights_float"),
        Some("enc_gru4_input_weights_idx"),
        None,
        Some("enc_gru4_input_scale"),
        352,
        96,
    )?;
    let enc_gru4_recurrent = linear_init(
        arrays,
        Some("enc_gru4_recurrent_bias"),
        Some("enc_gru4_recurrent_subias"),
        Some("enc_gru4_recurrent_weights_int8"),
        Some("enc_gru4_recurrent_weights_float"),
        None,
        None,
        Some("enc_gru4_recurrent_scale"),
        32,
        96,
    )?;
    let enc_gru5_input = linear_init(
        arrays,
        Some("enc_gru5_input_bias"),
        Some("enc_gru5_input_subias"),
        Some("enc_gru5_input_weights_int8"),
        Some("enc_gru5_input_weights_float"),
        Some("enc_gru5_input_weights_idx"),
        None,
        Some("enc_gru5_input_scale"),
        448,
        96,
    )?;
    let enc_gru5_recurrent = linear_init(
        arrays,
        Some("enc_gru5_recurrent_bias"),
        Some("enc_gru5_recurrent_subias"),
        Some("enc_gru5_recurrent_weights_int8"),
        Some("enc_gru5_recurrent_weights_float"),
        None,
        None,
        Some("enc_gru5_recurrent_scale"),
        32,
        96,
    )?;

    // Dense-int8 "conv" layers (no sparse idx).
    let enc_conv1 = linear_init(
        arrays,
        Some("enc_conv1_bias"),
        Some("enc_conv1_subias"),
        Some("enc_conv1_weights_int8"),
        Some("enc_conv1_weights_float"),
        None,
        None,
        Some("enc_conv1_scale"),
        128,
        64,
    )?;
    let enc_conv2 = linear_init(
        arrays,
        Some("enc_conv2_bias"),
        Some("enc_conv2_subias"),
        Some("enc_conv2_weights_int8"),
        Some("enc_conv2_weights_float"),
        None,
        None,
        Some("enc_conv2_scale"),
        128,
        64,
    )?;
    let enc_conv3 = linear_init(
        arrays,
        Some("enc_conv3_bias"),
        Some("enc_conv3_subias"),
        Some("enc_conv3_weights_int8"),
        Some("enc_conv3_weights_float"),
        None,
        None,
        Some("enc_conv3_scale"),
        128,
        64,
    )?;
    let enc_conv4 = linear_init(
        arrays,
        Some("enc_conv4_bias"),
        Some("enc_conv4_subias"),
        Some("enc_conv4_weights_int8"),
        Some("enc_conv4_weights_float"),
        None,
        None,
        Some("enc_conv4_scale"),
        128,
        64,
    )?;
    let enc_conv5 = linear_init(
        arrays,
        Some("enc_conv5_bias"),
        Some("enc_conv5_subias"),
        Some("enc_conv5_weights_int8"),
        Some("enc_conv5_weights_float"),
        None,
        None,
        Some("enc_conv5_scale"),
        128,
        64,
    )?;

    Ok(RDOVAEEnc {
        enc_dense1,
        enc_zdense,
        gdense2,
        gdense1,
        enc_conv_dense1,
        enc_conv_dense2,
        enc_conv_dense3,
        enc_conv_dense4,
        enc_conv_dense5,
        enc_gru1_input,
        enc_gru1_recurrent,
        enc_gru2_input,
        enc_gru2_recurrent,
        enc_gru3_input,
        enc_gru3_recurrent,
        enc_gru4_input,
        enc_gru4_recurrent,
        enc_gru5_input,
        enc_gru5_recurrent,
        enc_conv1,
        enc_conv2,
        enc_conv3,
        enc_conv4,
        enc_conv5,
    })
}

/// RDOVAE encoder running state. Matches C `struct RDOVAEEncStruct`
/// (`reference/dnn/dred_rdovae_enc.h`). All state arrays start zeroed
/// and `initialized` is set true once `dred_rdovae_encode_dframe`
/// bootstraps the network (in stage 8.4).
#[derive(Clone, Debug)]
pub struct RDOVAEEncState {
    pub initialized: bool,
    pub gru1_state: [f32; ENC_GRU1_STATE_SIZE],
    pub gru2_state: [f32; ENC_GRU2_STATE_SIZE],
    pub gru3_state: [f32; ENC_GRU3_STATE_SIZE],
    pub gru4_state: [f32; ENC_GRU4_STATE_SIZE],
    pub gru5_state: [f32; ENC_GRU5_STATE_SIZE],
    pub conv1_state: [f32; ENC_CONV1_STATE_SIZE],
    pub conv2_state: [f32; 2 * ENC_CONV2_STATE_SIZE],
    pub conv3_state: [f32; 2 * ENC_CONV3_STATE_SIZE],
    pub conv4_state: [f32; 2 * ENC_CONV4_STATE_SIZE],
    pub conv5_state: [f32; 2 * ENC_CONV5_STATE_SIZE],
}

impl Default for RDOVAEEncState {
    fn default() -> Self {
        Self {
            initialized: false,
            gru1_state: [0.0; ENC_GRU1_STATE_SIZE],
            gru2_state: [0.0; ENC_GRU2_STATE_SIZE],
            gru3_state: [0.0; ENC_GRU3_STATE_SIZE],
            gru4_state: [0.0; ENC_GRU4_STATE_SIZE],
            gru5_state: [0.0; ENC_GRU5_STATE_SIZE],
            conv1_state: [0.0; ENC_CONV1_STATE_SIZE],
            conv2_state: [0.0; 2 * ENC_CONV2_STATE_SIZE],
            conv3_state: [0.0; 2 * ENC_CONV3_STATE_SIZE],
            conv4_state: [0.0; 2 * ENC_CONV4_STATE_SIZE],
            conv5_state: [0.0; 2 * ENC_CONV5_STATE_SIZE],
        }
    }
}

// Buffer concatenates the per-stage outputs the encoder feeds into
// `gdense1` / `enc_zdense`. Order must exactly match the C `output_index`
// sequence in `dred_rdovae_encode_dframe`:
//
//   dense1 [64] + gru1 [32] + conv1 [64] + gru2 [32] + conv2 [64]
//        + gru3 [32] + conv3 [64] + gru4 [32] + conv4 [64]
//        + gru5 [32] + conv5 [64]                                      = 544
//
// This equals `gdense1.nb_inputs` in the weight init, so the concatenated
// buffer is precisely what the final dense layers want.
const ENC_FORWARD_BUFFER_SIZE: usize = ENC_DENSE1_OUT_SIZE
    + ENC_GRU1_OUT_SIZE
    + ENC_CONV1_OUT_SIZE
    + ENC_GRU2_OUT_SIZE
    + ENC_CONV2_OUT_SIZE
    + ENC_GRU3_OUT_SIZE
    + ENC_CONV3_OUT_SIZE
    + ENC_GRU4_OUT_SIZE
    + ENC_CONV4_OUT_SIZE
    + ENC_GRU5_OUT_SIZE
    + ENC_CONV5_OUT_SIZE;
const _: () = assert!(ENC_FORWARD_BUFFER_SIZE == 544);

/// Zero the first `dilation` taps of a dilated-conv memory buffer the very
/// first time the encoder runs. Matches C `conv1_cond_init` (private helper
/// in `dred_rdovae_enc.c`).
///
/// Stride note: each tap is `len` floats wide, so we zero the first
/// `dilation * len` elements. The C code writes `OPUS_CLEAR(&mem[i*len], len)`
/// for `i = 0..dilation-1`, which covers exactly that prefix.
#[inline]
fn conv1_cond_init(mem: &mut [f32], len: usize, dilation: usize, init: &mut bool) {
    if !*init {
        let zero_len = dilation * len;
        for x in &mut mem[..zero_len] {
            *x = 0.0;
        }
    }
    *init = true;
}

impl RDOVAEEncState {
    /// Run the RDOVAE encoder forward pass on a single `DRED_DFRAME_SIZE`
    /// / 2 = 40-wide feature frame (two concatenated LPCNet feature
    /// vectors). Updates `self` (all GRU / conv hidden state) in place and
    /// writes:
    /// - `latents[0..DRED_LATENT_DIM]` — 25 compressed latents.
    /// - `initial_state[0..DRED_STATE_DIM]` — 50-wide initial decoder state.
    ///
    /// Mirrors C `dred_rdovae_encode_dframe` in `dred_rdovae_enc.c`. The
    /// `arch` parameter is absent: `dnn::core` primitives pick the scalar
    /// reference path at compile time (no RTCD).
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the output slice lengths don't satisfy
    /// `latents.len() >= DRED_LATENT_DIM` and
    /// `initial_state.len() >= DRED_STATE_DIM`, or if the input has fewer
    /// than `2 * DRED_NUM_FEATURES` elements. In release builds the slice
    /// indexing panics on its own if the caller violates the contract.
    pub fn encode_dframe(
        &mut self,
        model: &RDOVAEEnc,
        latents: &mut [f32],
        initial_state: &mut [f32],
        input: &[f32],
    ) {
        debug_assert!(input.len() >= 2 * DRED_NUM_FEATURES);
        debug_assert!(latents.len() >= DRED_LATENT_DIM);
        debug_assert!(initial_state.len() >= DRED_STATE_DIM);

        let mut padded_latents = [0.0f32; DRED_PADDED_LATENT_DIM];
        let mut padded_state = [0.0f32; DRED_PADDED_STATE_DIM];
        let mut buffer = [0.0f32; ENC_FORWARD_BUFFER_SIZE];
        let mut state_hidden = [0.0f32; GDENSE1_OUT_SIZE];
        let mut conv_tmp = [0.0f32; DRED_MAX_CONV_INPUTS];

        // Stage 1: dense1 -> GRU1 -> conv1 (dilation = 1).
        let mut output_index = 0;
        compute_generic_dense(
            &model.enc_dense1,
            &mut buffer[output_index..output_index + ENC_DENSE1_OUT_SIZE],
            input,
            ACTIVATION_TANH,
        );
        output_index += ENC_DENSE1_OUT_SIZE;

        // The C reference passes `buffer` (fixed-length stack array) to the
        // GRU. The GRU only reads its first `nb_inputs` = ENC_DENSE1_OUT_SIZE
        // elements, so slicing to the dense1 output alone is equivalent.
        compute_generic_gru(
            &model.enc_gru1_input,
            &model.enc_gru1_recurrent,
            &mut self.gru1_state,
            &buffer[..ENC_DENSE1_OUT_SIZE],
        );
        buffer[output_index..output_index + ENC_GRU1_OUT_SIZE]
            .copy_from_slice(&self.gru1_state);
        output_index += ENC_GRU1_OUT_SIZE;
        conv1_cond_init(&mut self.conv1_state, ENC_CONV1_IN_SIZE, 1, &mut self.initialized);
        compute_generic_dense(
            &model.enc_conv_dense1,
            &mut conv_tmp[..ENC_CONV_DENSE1_OUT_SIZE],
            &buffer[..model.enc_conv_dense1.nb_inputs],
            ACTIVATION_TANH,
        );
        let mut conv_out = [0.0f32; ENC_CONV1_OUT_SIZE];
        compute_generic_conv1d(
            &model.enc_conv1,
            &mut conv_out,
            &mut self.conv1_state,
            &conv_tmp[..ENC_CONV1_IN_SIZE],
            ENC_CONV1_OUT_SIZE,
            ACTIVATION_TANH,
        );
        buffer[output_index..output_index + ENC_CONV1_OUT_SIZE].copy_from_slice(&conv_out);
        output_index += ENC_CONV1_OUT_SIZE;

        // Stage 2: GRU2 -> conv2 (dilation = 2).
        compute_generic_gru(
            &model.enc_gru2_input,
            &model.enc_gru2_recurrent,
            &mut self.gru2_state,
            &buffer[..model.enc_gru2_input.nb_inputs],
        );
        buffer[output_index..output_index + ENC_GRU2_OUT_SIZE]
            .copy_from_slice(&self.gru2_state);
        output_index += ENC_GRU2_OUT_SIZE;
        conv1_cond_init(&mut self.conv2_state, ENC_CONV2_IN_SIZE, 2, &mut self.initialized);
        compute_generic_dense(
            &model.enc_conv_dense2,
            &mut conv_tmp[..ENC_CONV_DENSE2_OUT_SIZE],
            &buffer[..model.enc_conv_dense2.nb_inputs],
            ACTIVATION_TANH,
        );
        let mut conv_out = [0.0f32; ENC_CONV2_OUT_SIZE];
        compute_generic_conv1d_dilation(
            &model.enc_conv2,
            &mut conv_out,
            &mut self.conv2_state,
            &conv_tmp[..ENC_CONV2_IN_SIZE],
            ENC_CONV2_OUT_SIZE,
            2,
            ACTIVATION_TANH,
        );
        buffer[output_index..output_index + ENC_CONV2_OUT_SIZE].copy_from_slice(&conv_out);
        output_index += ENC_CONV2_OUT_SIZE;

        // Stage 3: GRU3 -> conv3 (dilation = 2).
        compute_generic_gru(
            &model.enc_gru3_input,
            &model.enc_gru3_recurrent,
            &mut self.gru3_state,
            &buffer[..model.enc_gru3_input.nb_inputs],
        );
        buffer[output_index..output_index + ENC_GRU3_OUT_SIZE]
            .copy_from_slice(&self.gru3_state);
        output_index += ENC_GRU3_OUT_SIZE;
        conv1_cond_init(&mut self.conv3_state, ENC_CONV3_IN_SIZE, 2, &mut self.initialized);
        compute_generic_dense(
            &model.enc_conv_dense3,
            &mut conv_tmp[..ENC_CONV_DENSE3_OUT_SIZE],
            &buffer[..model.enc_conv_dense3.nb_inputs],
            ACTIVATION_TANH,
        );
        let mut conv_out = [0.0f32; ENC_CONV3_OUT_SIZE];
        compute_generic_conv1d_dilation(
            &model.enc_conv3,
            &mut conv_out,
            &mut self.conv3_state,
            &conv_tmp[..ENC_CONV3_IN_SIZE],
            ENC_CONV3_OUT_SIZE,
            2,
            ACTIVATION_TANH,
        );
        buffer[output_index..output_index + ENC_CONV3_OUT_SIZE].copy_from_slice(&conv_out);
        output_index += ENC_CONV3_OUT_SIZE;

        // Stage 4: GRU4 -> conv4 (dilation = 2).
        compute_generic_gru(
            &model.enc_gru4_input,
            &model.enc_gru4_recurrent,
            &mut self.gru4_state,
            &buffer[..model.enc_gru4_input.nb_inputs],
        );
        buffer[output_index..output_index + ENC_GRU4_OUT_SIZE]
            .copy_from_slice(&self.gru4_state);
        output_index += ENC_GRU4_OUT_SIZE;
        conv1_cond_init(&mut self.conv4_state, ENC_CONV4_IN_SIZE, 2, &mut self.initialized);
        compute_generic_dense(
            &model.enc_conv_dense4,
            &mut conv_tmp[..ENC_CONV_DENSE4_OUT_SIZE],
            &buffer[..model.enc_conv_dense4.nb_inputs],
            ACTIVATION_TANH,
        );
        let mut conv_out = [0.0f32; ENC_CONV4_OUT_SIZE];
        compute_generic_conv1d_dilation(
            &model.enc_conv4,
            &mut conv_out,
            &mut self.conv4_state,
            &conv_tmp[..ENC_CONV4_IN_SIZE],
            ENC_CONV4_OUT_SIZE,
            2,
            ACTIVATION_TANH,
        );
        buffer[output_index..output_index + ENC_CONV4_OUT_SIZE].copy_from_slice(&conv_out);
        output_index += ENC_CONV4_OUT_SIZE;

        // Stage 5: GRU5 -> conv5 (dilation = 2).
        compute_generic_gru(
            &model.enc_gru5_input,
            &model.enc_gru5_recurrent,
            &mut self.gru5_state,
            &buffer[..model.enc_gru5_input.nb_inputs],
        );
        buffer[output_index..output_index + ENC_GRU5_OUT_SIZE]
            .copy_from_slice(&self.gru5_state);
        output_index += ENC_GRU5_OUT_SIZE;
        conv1_cond_init(&mut self.conv5_state, ENC_CONV5_IN_SIZE, 2, &mut self.initialized);
        compute_generic_dense(
            &model.enc_conv_dense5,
            &mut conv_tmp[..ENC_CONV_DENSE5_OUT_SIZE],
            &buffer[..model.enc_conv_dense5.nb_inputs],
            ACTIVATION_TANH,
        );
        let mut conv_out = [0.0f32; ENC_CONV5_OUT_SIZE];
        compute_generic_conv1d_dilation(
            &model.enc_conv5,
            &mut conv_out,
            &mut self.conv5_state,
            &conv_tmp[..ENC_CONV5_IN_SIZE],
            ENC_CONV5_OUT_SIZE,
            2,
            ACTIVATION_TANH,
        );
        buffer[output_index..output_index + ENC_CONV5_OUT_SIZE].copy_from_slice(&conv_out);
        output_index += ENC_CONV5_OUT_SIZE;
        debug_assert_eq!(output_index, ENC_FORWARD_BUFFER_SIZE);

        // Final latent projection: enc_zdense (linear), then truncate the
        // 32-wide padded output down to DRED_LATENT_DIM = 25.
        compute_generic_dense(
            &model.enc_zdense,
            &mut padded_latents,
            &buffer,
            ACTIVATION_LINEAR,
        );
        latents[..DRED_LATENT_DIM].copy_from_slice(&padded_latents[..DRED_LATENT_DIM]);

        // Initial-state projection: gdense1 (tanh) -> gdense2 (linear).
        compute_generic_dense(
            &model.gdense1,
            &mut state_hidden,
            &buffer,
            ACTIVATION_TANH,
        );
        compute_generic_dense(
            &model.gdense2,
            &mut padded_state,
            &state_hidden,
            ACTIVATION_LINEAR,
        );
        initial_state[..DRED_STATE_DIM].copy_from_slice(&padded_state[..DRED_STATE_DIM]);
    }
}

// ===========================================================================
// RDOVAE decoder model (struct-of-LinearLayer)
// ===========================================================================

/// RDOVAE decoder weights. Matches C `struct RDOVAEDec`
/// (`reference/dnn/dred_rdovae_dec_data.h`).
#[derive(Clone, Debug, Default)]
pub struct RDOVAEDec {
    pub dec_dense1: LinearLayer,
    pub dec_glu1: LinearLayer,
    pub dec_glu2: LinearLayer,
    pub dec_glu3: LinearLayer,
    pub dec_glu4: LinearLayer,
    pub dec_glu5: LinearLayer,
    pub dec_hidden_init: LinearLayer,
    pub dec_output: LinearLayer,
    pub dec_conv_dense1: LinearLayer,
    pub dec_conv_dense2: LinearLayer,
    pub dec_conv_dense3: LinearLayer,
    pub dec_conv_dense4: LinearLayer,
    pub dec_conv_dense5: LinearLayer,
    pub dec_gru_init: LinearLayer,
    pub dec_gru1_input: LinearLayer,
    pub dec_gru1_recurrent: LinearLayer,
    pub dec_gru2_input: LinearLayer,
    pub dec_gru2_recurrent: LinearLayer,
    pub dec_gru3_input: LinearLayer,
    pub dec_gru3_recurrent: LinearLayer,
    pub dec_gru4_input: LinearLayer,
    pub dec_gru4_recurrent: LinearLayer,
    pub dec_gru5_input: LinearLayer,
    pub dec_gru5_recurrent: LinearLayer,
    pub dec_conv1: LinearLayer,
    pub dec_conv2: LinearLayer,
    pub dec_conv3: LinearLayer,
    pub dec_conv4: LinearLayer,
    pub dec_conv5: LinearLayer,
}

/// Initialise the RDOVAE decoder from a parsed weight blob.
/// Matches C `init_rdovaedec` in `dred_rdovae_dec_data.c`.
pub fn init_rdovaedec(arrays: &[WeightArray]) -> Result<RDOVAEDec, ()> {
    let dec_dense1 = linear_init(
        arrays,
        Some("dec_dense1_bias"),
        None,
        None,
        Some("dec_dense1_weights_float"),
        None,
        None,
        None,
        26,
        DEC_DENSE1_OUT_SIZE,
    )?;

    let dec_glu1 = linear_init(
        arrays,
        Some("dec_glu1_bias"),
        Some("dec_glu1_subias"),
        Some("dec_glu1_weights_int8"),
        Some("dec_glu1_weights_float"),
        None,
        None,
        Some("dec_glu1_scale"),
        64,
        DEC_GLU_OUT_SIZE,
    )?;
    let dec_glu2 = linear_init(
        arrays,
        Some("dec_glu2_bias"),
        Some("dec_glu2_subias"),
        Some("dec_glu2_weights_int8"),
        Some("dec_glu2_weights_float"),
        None,
        None,
        Some("dec_glu2_scale"),
        64,
        DEC_GLU_OUT_SIZE,
    )?;
    let dec_glu3 = linear_init(
        arrays,
        Some("dec_glu3_bias"),
        Some("dec_glu3_subias"),
        Some("dec_glu3_weights_int8"),
        Some("dec_glu3_weights_float"),
        None,
        None,
        Some("dec_glu3_scale"),
        64,
        DEC_GLU_OUT_SIZE,
    )?;
    let dec_glu4 = linear_init(
        arrays,
        Some("dec_glu4_bias"),
        Some("dec_glu4_subias"),
        Some("dec_glu4_weights_int8"),
        Some("dec_glu4_weights_float"),
        None,
        None,
        Some("dec_glu4_scale"),
        64,
        DEC_GLU_OUT_SIZE,
    )?;
    let dec_glu5 = linear_init(
        arrays,
        Some("dec_glu5_bias"),
        Some("dec_glu5_subias"),
        Some("dec_glu5_weights_int8"),
        Some("dec_glu5_weights_float"),
        None,
        None,
        Some("dec_glu5_scale"),
        64,
        DEC_GLU_OUT_SIZE,
    )?;

    let dec_hidden_init = linear_init(
        arrays,
        Some("dec_hidden_init_bias"),
        None,
        None,
        Some("dec_hidden_init_weights_float"),
        None,
        None,
        None,
        50,
        DEC_HIDDEN_INIT_OUT_SIZE,
    )?;

    let dec_output = linear_init(
        arrays,
        Some("dec_output_bias"),
        Some("dec_output_subias"),
        Some("dec_output_weights_int8"),
        Some("dec_output_weights_float"),
        Some("dec_output_weights_idx"),
        None,
        Some("dec_output_scale"),
        576,
        DEC_OUTPUT_OUT_SIZE,
    )?;

    let dec_conv_dense1 = linear_init(
        arrays,
        Some("dec_conv_dense1_bias"),
        Some("dec_conv_dense1_subias"),
        Some("dec_conv_dense1_weights_int8"),
        Some("dec_conv_dense1_weights_float"),
        Some("dec_conv_dense1_weights_idx"),
        None,
        Some("dec_conv_dense1_scale"),
        160,
        DEC_CONV_DENSE_OUT_SIZE,
    )?;
    let dec_conv_dense2 = linear_init(
        arrays,
        Some("dec_conv_dense2_bias"),
        Some("dec_conv_dense2_subias"),
        Some("dec_conv_dense2_weights_int8"),
        Some("dec_conv_dense2_weights_float"),
        Some("dec_conv_dense2_weights_idx"),
        None,
        Some("dec_conv_dense2_scale"),
        256,
        DEC_CONV_DENSE_OUT_SIZE,
    )?;
    let dec_conv_dense3 = linear_init(
        arrays,
        Some("dec_conv_dense3_bias"),
        Some("dec_conv_dense3_subias"),
        Some("dec_conv_dense3_weights_int8"),
        Some("dec_conv_dense3_weights_float"),
        Some("dec_conv_dense3_weights_idx"),
        None,
        Some("dec_conv_dense3_scale"),
        352,
        DEC_CONV_DENSE_OUT_SIZE,
    )?;
    let dec_conv_dense4 = linear_init(
        arrays,
        Some("dec_conv_dense4_bias"),
        Some("dec_conv_dense4_subias"),
        Some("dec_conv_dense4_weights_int8"),
        Some("dec_conv_dense4_weights_float"),
        Some("dec_conv_dense4_weights_idx"),
        None,
        Some("dec_conv_dense4_scale"),
        448,
        DEC_CONV_DENSE_OUT_SIZE,
    )?;
    let dec_conv_dense5 = linear_init(
        arrays,
        Some("dec_conv_dense5_bias"),
        Some("dec_conv_dense5_subias"),
        Some("dec_conv_dense5_weights_int8"),
        Some("dec_conv_dense5_weights_float"),
        Some("dec_conv_dense5_weights_idx"),
        None,
        Some("dec_conv_dense5_scale"),
        544,
        DEC_CONV_DENSE_OUT_SIZE,
    )?;

    let dec_gru_init = linear_init(
        arrays,
        Some("dec_gru_init_bias"),
        Some("dec_gru_init_subias"),
        Some("dec_gru_init_weights_int8"),
        Some("dec_gru_init_weights_float"),
        Some("dec_gru_init_weights_idx"),
        None,
        Some("dec_gru_init_scale"),
        128,
        DEC_GRU_INIT_OUT_SIZE,
    )?;

    let dec_gru1_input = linear_init(
        arrays,
        Some("dec_gru1_input_bias"),
        Some("dec_gru1_input_subias"),
        Some("dec_gru1_input_weights_int8"),
        Some("dec_gru1_input_weights_float"),
        Some("dec_gru1_input_weights_idx"),
        None,
        Some("dec_gru1_input_scale"),
        96,
        192,
    )?;
    let dec_gru1_recurrent = linear_init(
        arrays,
        Some("dec_gru1_recurrent_bias"),
        Some("dec_gru1_recurrent_subias"),
        Some("dec_gru1_recurrent_weights_int8"),
        Some("dec_gru1_recurrent_weights_float"),
        None,
        None,
        Some("dec_gru1_recurrent_scale"),
        64,
        192,
    )?;
    let dec_gru2_input = linear_init(
        arrays,
        Some("dec_gru2_input_bias"),
        Some("dec_gru2_input_subias"),
        Some("dec_gru2_input_weights_int8"),
        Some("dec_gru2_input_weights_float"),
        Some("dec_gru2_input_weights_idx"),
        None,
        Some("dec_gru2_input_scale"),
        192,
        192,
    )?;
    let dec_gru2_recurrent = linear_init(
        arrays,
        Some("dec_gru2_recurrent_bias"),
        Some("dec_gru2_recurrent_subias"),
        Some("dec_gru2_recurrent_weights_int8"),
        Some("dec_gru2_recurrent_weights_float"),
        None,
        None,
        Some("dec_gru2_recurrent_scale"),
        64,
        192,
    )?;
    let dec_gru3_input = linear_init(
        arrays,
        Some("dec_gru3_input_bias"),
        Some("dec_gru3_input_subias"),
        Some("dec_gru3_input_weights_int8"),
        Some("dec_gru3_input_weights_float"),
        Some("dec_gru3_input_weights_idx"),
        None,
        Some("dec_gru3_input_scale"),
        288,
        192,
    )?;
    let dec_gru3_recurrent = linear_init(
        arrays,
        Some("dec_gru3_recurrent_bias"),
        Some("dec_gru3_recurrent_subias"),
        Some("dec_gru3_recurrent_weights_int8"),
        Some("dec_gru3_recurrent_weights_float"),
        None,
        None,
        Some("dec_gru3_recurrent_scale"),
        64,
        192,
    )?;
    let dec_gru4_input = linear_init(
        arrays,
        Some("dec_gru4_input_bias"),
        Some("dec_gru4_input_subias"),
        Some("dec_gru4_input_weights_int8"),
        Some("dec_gru4_input_weights_float"),
        Some("dec_gru4_input_weights_idx"),
        None,
        Some("dec_gru4_input_scale"),
        384,
        192,
    )?;
    let dec_gru4_recurrent = linear_init(
        arrays,
        Some("dec_gru4_recurrent_bias"),
        Some("dec_gru4_recurrent_subias"),
        Some("dec_gru4_recurrent_weights_int8"),
        Some("dec_gru4_recurrent_weights_float"),
        None,
        None,
        Some("dec_gru4_recurrent_scale"),
        64,
        192,
    )?;
    let dec_gru5_input = linear_init(
        arrays,
        Some("dec_gru5_input_bias"),
        Some("dec_gru5_input_subias"),
        Some("dec_gru5_input_weights_int8"),
        Some("dec_gru5_input_weights_float"),
        Some("dec_gru5_input_weights_idx"),
        None,
        Some("dec_gru5_input_scale"),
        480,
        192,
    )?;
    let dec_gru5_recurrent = linear_init(
        arrays,
        Some("dec_gru5_recurrent_bias"),
        Some("dec_gru5_recurrent_subias"),
        Some("dec_gru5_recurrent_weights_int8"),
        Some("dec_gru5_recurrent_weights_float"),
        None,
        None,
        Some("dec_gru5_recurrent_scale"),
        64,
        192,
    )?;

    let dec_conv1 = linear_init(
        arrays,
        Some("dec_conv1_bias"),
        Some("dec_conv1_subias"),
        Some("dec_conv1_weights_int8"),
        Some("dec_conv1_weights_float"),
        None,
        None,
        Some("dec_conv1_scale"),
        64,
        DEC_CONV_DENSE_OUT_SIZE,
    )?;
    let dec_conv2 = linear_init(
        arrays,
        Some("dec_conv2_bias"),
        Some("dec_conv2_subias"),
        Some("dec_conv2_weights_int8"),
        Some("dec_conv2_weights_float"),
        None,
        None,
        Some("dec_conv2_scale"),
        64,
        DEC_CONV_DENSE_OUT_SIZE,
    )?;
    let dec_conv3 = linear_init(
        arrays,
        Some("dec_conv3_bias"),
        Some("dec_conv3_subias"),
        Some("dec_conv3_weights_int8"),
        Some("dec_conv3_weights_float"),
        None,
        None,
        Some("dec_conv3_scale"),
        64,
        DEC_CONV_DENSE_OUT_SIZE,
    )?;
    let dec_conv4 = linear_init(
        arrays,
        Some("dec_conv4_bias"),
        Some("dec_conv4_subias"),
        Some("dec_conv4_weights_int8"),
        Some("dec_conv4_weights_float"),
        None,
        None,
        Some("dec_conv4_scale"),
        64,
        DEC_CONV_DENSE_OUT_SIZE,
    )?;
    let dec_conv5 = linear_init(
        arrays,
        Some("dec_conv5_bias"),
        Some("dec_conv5_subias"),
        Some("dec_conv5_weights_int8"),
        Some("dec_conv5_weights_float"),
        None,
        None,
        Some("dec_conv5_scale"),
        64,
        DEC_CONV_DENSE_OUT_SIZE,
    )?;

    Ok(RDOVAEDec {
        dec_dense1,
        dec_glu1,
        dec_glu2,
        dec_glu3,
        dec_glu4,
        dec_glu5,
        dec_hidden_init,
        dec_output,
        dec_conv_dense1,
        dec_conv_dense2,
        dec_conv_dense3,
        dec_conv_dense4,
        dec_conv_dense5,
        dec_gru_init,
        dec_gru1_input,
        dec_gru1_recurrent,
        dec_gru2_input,
        dec_gru2_recurrent,
        dec_gru3_input,
        dec_gru3_recurrent,
        dec_gru4_input,
        dec_gru4_recurrent,
        dec_gru5_input,
        dec_gru5_recurrent,
        dec_conv1,
        dec_conv2,
        dec_conv3,
        dec_conv4,
        dec_conv5,
    })
}

/// RDOVAE decoder running state. Matches C `struct RDOVAEDecStruct`
/// (`reference/dnn/dred_rdovae_dec.h`).
#[derive(Clone, Debug)]
pub struct RDOVAEDecState {
    pub initialized: bool,
    pub gru1_state: [f32; DEC_GRU_STATE_SIZE],
    pub gru2_state: [f32; DEC_GRU_STATE_SIZE],
    pub gru3_state: [f32; DEC_GRU_STATE_SIZE],
    pub gru4_state: [f32; DEC_GRU_STATE_SIZE],
    pub gru5_state: [f32; DEC_GRU_STATE_SIZE],
    pub conv1_state: [f32; DEC_CONV_STATE_SIZE],
    pub conv2_state: [f32; DEC_CONV_STATE_SIZE],
    pub conv3_state: [f32; DEC_CONV_STATE_SIZE],
    pub conv4_state: [f32; DEC_CONV_STATE_SIZE],
    pub conv5_state: [f32; DEC_CONV_STATE_SIZE],
}

impl Default for RDOVAEDecState {
    fn default() -> Self {
        Self {
            initialized: false,
            gru1_state: [0.0; DEC_GRU_STATE_SIZE],
            gru2_state: [0.0; DEC_GRU_STATE_SIZE],
            gru3_state: [0.0; DEC_GRU_STATE_SIZE],
            gru4_state: [0.0; DEC_GRU_STATE_SIZE],
            gru5_state: [0.0; DEC_GRU_STATE_SIZE],
            conv1_state: [0.0; DEC_CONV_STATE_SIZE],
            conv2_state: [0.0; DEC_CONV_STATE_SIZE],
            conv3_state: [0.0; DEC_CONV_STATE_SIZE],
            conv4_state: [0.0; DEC_CONV_STATE_SIZE],
            conv5_state: [0.0; DEC_CONV_STATE_SIZE],
        }
    }
}

// Buffer concatenates the per-stage outputs the decoder feeds into
// `dec_output`. Order must exactly match the C `output_index` sequence in
// `dred_rdovae_decode_qframe`:
//
//   dense1 [96] + gru1 [64] + conv1 [32] + gru2 [64] + conv2 [32]
//        + gru3 [64] + conv3 [32] + gru4 [64] + conv4 [32]
//        + gru5 [64] + conv5 [32]                                      = 576
//
// This equals `dec_output.nb_inputs` in the weight init, so the concatenated
// buffer is precisely what the final dense layer wants.
const DEC_FORWARD_BUFFER_SIZE: usize = DEC_DENSE1_OUT_SIZE
    + DEC_GRU1_OUT_SIZE
    + DEC_CONV1_OUT_SIZE
    + DEC_GRU2_OUT_SIZE
    + DEC_CONV2_OUT_SIZE
    + DEC_GRU3_OUT_SIZE
    + DEC_CONV3_OUT_SIZE
    + DEC_GRU4_OUT_SIZE
    + DEC_CONV4_OUT_SIZE
    + DEC_GRU5_OUT_SIZE
    + DEC_CONV5_OUT_SIZE;
const _: () = assert!(DEC_FORWARD_BUFFER_SIZE == 576);

/// Width of the latent vector passed to `decode_qframe`. Matches the
/// C `dec_dense1.nb_inputs` (26 = DRED_LATENT_DIM + 1, the trailing byte
/// being the quantiser index).
const DEC_DENSE1_IN_SIZE: usize = DRED_LATENT_DIM + 1;

/// Total size of the GRU-init vector produced by `dec_gru_init`. Must equal
/// the sum of the five per-GRU state sizes so the `OPUS_COPY` chain in
/// `dred_rdovae_dec_init_states` covers exactly those outputs.
const _: () = assert!(
    DEC_GRU_INIT_OUT_SIZE
        == DEC_GRU1_STATE_SIZE
            + DEC_GRU2_STATE_SIZE
            + DEC_GRU3_STATE_SIZE
            + DEC_GRU4_STATE_SIZE
            + DEC_GRU5_STATE_SIZE
);

impl RDOVAEDecState {
    /// Seed the five GRU hidden states from a reconstructed `initial_state`
    /// (the 50-wide bank the encoder emitted via `gdense2`). Matches C
    /// `dred_rdovae_dec_init_states` in `dred_rdovae_dec.c`. Calls two
    /// dense layers, then splits the 320-wide output into the five GRU
    /// banks. Also clears `initialized` so the next `decode_qframe` call
    /// performs the one-shot conv-memory zeroing.
    ///
    /// # Panics
    ///
    /// Debug-only: panics if `initial_state.len() < DRED_STATE_DIM`.
    pub fn init_states(&mut self, model: &RDOVAEDec, initial_state: &[f32]) {
        debug_assert!(initial_state.len() >= DRED_STATE_DIM);

        let mut hidden = [0.0f32; DEC_HIDDEN_INIT_OUT_SIZE];
        let mut state_init = [0.0f32; DEC_GRU_INIT_OUT_SIZE];

        compute_generic_dense(
            &model.dec_hidden_init,
            &mut hidden,
            &initial_state[..DRED_STATE_DIM],
            ACTIVATION_TANH,
        );
        compute_generic_dense(
            &model.dec_gru_init,
            &mut state_init,
            &hidden,
            ACTIVATION_TANH,
        );

        let mut counter = 0;
        self.gru1_state
            .copy_from_slice(&state_init[counter..counter + DEC_GRU1_STATE_SIZE]);
        counter += DEC_GRU1_STATE_SIZE;
        self.gru2_state
            .copy_from_slice(&state_init[counter..counter + DEC_GRU2_STATE_SIZE]);
        counter += DEC_GRU2_STATE_SIZE;
        self.gru3_state
            .copy_from_slice(&state_init[counter..counter + DEC_GRU3_STATE_SIZE]);
        counter += DEC_GRU3_STATE_SIZE;
        self.gru4_state
            .copy_from_slice(&state_init[counter..counter + DEC_GRU4_STATE_SIZE]);
        counter += DEC_GRU4_STATE_SIZE;
        self.gru5_state
            .copy_from_slice(&state_init[counter..counter + DEC_GRU5_STATE_SIZE]);
        debug_assert_eq!(counter + DEC_GRU5_STATE_SIZE, DEC_GRU_INIT_OUT_SIZE);

        self.initialized = false;
    }

    /// Run the RDOVAE decoder forward pass on a single `DRED_LATENT_DIM+1`
    /// = 26-wide latent. Updates `self` (all GRU / conv hidden state) in
    /// place and writes:
    /// - `qframe[0..DEC_OUTPUT_OUT_SIZE]` — 80 floats, four concatenated
    ///   LPCNet feature vectors in reverse time order.
    ///
    /// Mirrors C `dred_rdovae_decode_qframe` in `dred_rdovae_dec.c`. The
    /// `arch` parameter is absent: `dnn::core` primitives pick the scalar
    /// reference path at compile time (no RTCD).
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the output slice is shorter than
    /// `DEC_OUTPUT_OUT_SIZE`, or if the input has fewer than
    /// `DEC_DENSE1_IN_SIZE` elements. In release builds the slice indexing
    /// panics on its own if the caller violates the contract.
    pub fn decode_qframe(
        &mut self,
        model: &RDOVAEDec,
        qframe: &mut [f32],
        input: &[f32],
    ) {
        debug_assert!(input.len() >= DEC_DENSE1_IN_SIZE);
        debug_assert!(qframe.len() >= DEC_OUTPUT_OUT_SIZE);

        let mut buffer = [0.0f32; DEC_FORWARD_BUFFER_SIZE];
        let mut conv_tmp = [0.0f32; DRED_MAX_CONV_INPUTS];

        // Stage 1: dense1 -> GRU1 -> GLU1 -> conv1 (dilation = 1).
        let mut output_index = 0;
        compute_generic_dense(
            &model.dec_dense1,
            &mut buffer[output_index..output_index + DEC_DENSE1_OUT_SIZE],
            &input[..DEC_DENSE1_IN_SIZE],
            ACTIVATION_TANH,
        );
        output_index += DEC_DENSE1_OUT_SIZE;

        // The C reference passes `buffer` (fixed-length stack array) to the
        // GRU. The GRU only reads its first `nb_inputs` elements (96 =
        // DEC_DENSE1_OUT_SIZE for GRU1), so slicing to that prefix is
        // equivalent.
        compute_generic_gru(
            &model.dec_gru1_input,
            &model.dec_gru1_recurrent,
            &mut self.gru1_state,
            &buffer[..model.dec_gru1_input.nb_inputs],
        );
        compute_glu(
            &model.dec_glu1,
            &mut buffer[output_index..output_index + DEC_GLU_OUT_SIZE],
            &self.gru1_state,
        );
        output_index += DEC_GRU1_OUT_SIZE;
        conv1_cond_init(
            &mut self.conv1_state,
            DEC_CONV1_IN_SIZE,
            1,
            &mut self.initialized,
        );
        compute_generic_dense(
            &model.dec_conv_dense1,
            &mut conv_tmp[..DEC_CONV_DENSE1_OUT_SIZE],
            &buffer[..model.dec_conv_dense1.nb_inputs],
            ACTIVATION_TANH,
        );
        let mut conv_out = [0.0f32; DEC_CONV1_OUT_SIZE];
        compute_generic_conv1d(
            &model.dec_conv1,
            &mut conv_out,
            &mut self.conv1_state,
            &conv_tmp[..DEC_CONV1_IN_SIZE],
            DEC_CONV1_OUT_SIZE,
            ACTIVATION_TANH,
        );
        buffer[output_index..output_index + DEC_CONV1_OUT_SIZE].copy_from_slice(&conv_out);
        output_index += DEC_CONV1_OUT_SIZE;

        // Stage 2: GRU2 -> GLU2 -> conv2 (dilation = 1).
        compute_generic_gru(
            &model.dec_gru2_input,
            &model.dec_gru2_recurrent,
            &mut self.gru2_state,
            &buffer[..model.dec_gru2_input.nb_inputs],
        );
        compute_glu(
            &model.dec_glu2,
            &mut buffer[output_index..output_index + DEC_GLU_OUT_SIZE],
            &self.gru2_state,
        );
        output_index += DEC_GRU2_OUT_SIZE;
        conv1_cond_init(
            &mut self.conv2_state,
            DEC_CONV2_IN_SIZE,
            1,
            &mut self.initialized,
        );
        compute_generic_dense(
            &model.dec_conv_dense2,
            &mut conv_tmp[..DEC_CONV_DENSE2_OUT_SIZE],
            &buffer[..model.dec_conv_dense2.nb_inputs],
            ACTIVATION_TANH,
        );
        let mut conv_out = [0.0f32; DEC_CONV2_OUT_SIZE];
        compute_generic_conv1d(
            &model.dec_conv2,
            &mut conv_out,
            &mut self.conv2_state,
            &conv_tmp[..DEC_CONV2_IN_SIZE],
            DEC_CONV2_OUT_SIZE,
            ACTIVATION_TANH,
        );
        buffer[output_index..output_index + DEC_CONV2_OUT_SIZE].copy_from_slice(&conv_out);
        output_index += DEC_CONV2_OUT_SIZE;

        // Stage 3: GRU3 -> GLU3 -> conv3 (dilation = 1).
        compute_generic_gru(
            &model.dec_gru3_input,
            &model.dec_gru3_recurrent,
            &mut self.gru3_state,
            &buffer[..model.dec_gru3_input.nb_inputs],
        );
        compute_glu(
            &model.dec_glu3,
            &mut buffer[output_index..output_index + DEC_GLU_OUT_SIZE],
            &self.gru3_state,
        );
        output_index += DEC_GRU3_OUT_SIZE;
        conv1_cond_init(
            &mut self.conv3_state,
            DEC_CONV3_IN_SIZE,
            1,
            &mut self.initialized,
        );
        compute_generic_dense(
            &model.dec_conv_dense3,
            &mut conv_tmp[..DEC_CONV_DENSE3_OUT_SIZE],
            &buffer[..model.dec_conv_dense3.nb_inputs],
            ACTIVATION_TANH,
        );
        let mut conv_out = [0.0f32; DEC_CONV3_OUT_SIZE];
        compute_generic_conv1d(
            &model.dec_conv3,
            &mut conv_out,
            &mut self.conv3_state,
            &conv_tmp[..DEC_CONV3_IN_SIZE],
            DEC_CONV3_OUT_SIZE,
            ACTIVATION_TANH,
        );
        buffer[output_index..output_index + DEC_CONV3_OUT_SIZE].copy_from_slice(&conv_out);
        output_index += DEC_CONV3_OUT_SIZE;

        // Stage 4: GRU4 -> GLU4 -> conv4 (dilation = 1).
        compute_generic_gru(
            &model.dec_gru4_input,
            &model.dec_gru4_recurrent,
            &mut self.gru4_state,
            &buffer[..model.dec_gru4_input.nb_inputs],
        );
        compute_glu(
            &model.dec_glu4,
            &mut buffer[output_index..output_index + DEC_GLU_OUT_SIZE],
            &self.gru4_state,
        );
        output_index += DEC_GRU4_OUT_SIZE;
        conv1_cond_init(
            &mut self.conv4_state,
            DEC_CONV4_IN_SIZE,
            1,
            &mut self.initialized,
        );
        compute_generic_dense(
            &model.dec_conv_dense4,
            &mut conv_tmp[..DEC_CONV_DENSE4_OUT_SIZE],
            &buffer[..model.dec_conv_dense4.nb_inputs],
            ACTIVATION_TANH,
        );
        let mut conv_out = [0.0f32; DEC_CONV4_OUT_SIZE];
        compute_generic_conv1d(
            &model.dec_conv4,
            &mut conv_out,
            &mut self.conv4_state,
            &conv_tmp[..DEC_CONV4_IN_SIZE],
            DEC_CONV4_OUT_SIZE,
            ACTIVATION_TANH,
        );
        buffer[output_index..output_index + DEC_CONV4_OUT_SIZE].copy_from_slice(&conv_out);
        output_index += DEC_CONV4_OUT_SIZE;

        // Stage 5: GRU5 -> GLU5 -> conv5 (dilation = 1).
        compute_generic_gru(
            &model.dec_gru5_input,
            &model.dec_gru5_recurrent,
            &mut self.gru5_state,
            &buffer[..model.dec_gru5_input.nb_inputs],
        );
        compute_glu(
            &model.dec_glu5,
            &mut buffer[output_index..output_index + DEC_GLU_OUT_SIZE],
            &self.gru5_state,
        );
        output_index += DEC_GRU5_OUT_SIZE;
        conv1_cond_init(
            &mut self.conv5_state,
            DEC_CONV5_IN_SIZE,
            1,
            &mut self.initialized,
        );
        compute_generic_dense(
            &model.dec_conv_dense5,
            &mut conv_tmp[..DEC_CONV_DENSE5_OUT_SIZE],
            &buffer[..model.dec_conv_dense5.nb_inputs],
            ACTIVATION_TANH,
        );
        let mut conv_out = [0.0f32; DEC_CONV5_OUT_SIZE];
        compute_generic_conv1d(
            &model.dec_conv5,
            &mut conv_out,
            &mut self.conv5_state,
            &conv_tmp[..DEC_CONV5_IN_SIZE],
            DEC_CONV5_OUT_SIZE,
            ACTIVATION_TANH,
        );
        buffer[output_index..output_index + DEC_CONV5_OUT_SIZE].copy_from_slice(&conv_out);
        output_index += DEC_CONV5_OUT_SIZE;
        debug_assert_eq!(output_index, DEC_FORWARD_BUFFER_SIZE);

        // Final projection: dec_output (linear) -> qframe[0..80].
        compute_generic_dense(
            &model.dec_output,
            &mut qframe[..DEC_OUTPUT_OUT_SIZE],
            &buffer,
            ACTIVATION_LINEAR,
        );
    }
}

// ===========================================================================
// DREDEnc — full encoder-side DRED state
// ===========================================================================

/// DRED encoder state. Matches C `DREDEnc`
/// (`reference/dnn/dred_encoder.h`).
///
/// Holds the RDOVAE model + running state, an LPCNet feature extractor,
/// a 48→16 kHz resample memory, and the sliding buffers that feed the
/// RDOVAE at its native 16 kHz / 20 ms cadence. Populated as of stage
/// 8.3; the encode forward pass and payload emission land in 8.6.
#[derive(Clone, Debug)]
pub struct DREDEnc {
    pub model: RDOVAEEnc,
    pub lpcnet_enc_state: LPCNetEncState,
    pub rdovae_enc: RDOVAEEncState,
    pub loaded: bool,
    pub fs: i32,
    pub channels: i32,

    pub input_buffer: [f32; 2 * DRED_DFRAME_SIZE],
    pub input_buffer_fill: i32,
    pub dred_offset: i32,
    pub latent_offset: i32,
    pub last_extra_dred_offset: i32,
    pub latents_buffer: [f32; DRED_MAX_FRAMES * DRED_LATENT_DIM],
    pub latents_buffer_fill: i32,
    pub state_buffer: [f32; DRED_MAX_FRAMES * DRED_STATE_DIM],
    pub resample_mem: [f32; RESAMPLING_ORDER + 1],
}

impl Default for DREDEnc {
    fn default() -> Self {
        Self {
            model: RDOVAEEnc::default(),
            lpcnet_enc_state: LPCNetEncState::default(),
            rdovae_enc: RDOVAEEncState::default(),
            loaded: false,
            fs: 0,
            channels: 0,
            input_buffer: [0.0; 2 * DRED_DFRAME_SIZE],
            input_buffer_fill: 0,
            dred_offset: 0,
            latent_offset: 0,
            last_extra_dred_offset: 0,
            latents_buffer: [0.0; DRED_MAX_FRAMES * DRED_LATENT_DIM],
            latents_buffer_fill: 0,
            state_buffer: [0.0; DRED_MAX_FRAMES * DRED_STATE_DIM],
            resample_mem: [0.0; RESAMPLING_ORDER + 1],
        }
    }
}

impl DREDEnc {
    /// Construct a fresh DRED encoder with all runtime state zeroed.
    ///
    /// If the crate-embedded `WEIGHTS_BLOB` contains the RDOVAE encoder
    /// layers (built from `reference/dnn/dred_rdovae_enc_data.c` via
    /// `ropus/build/gen_weights_blob.c`), the model is populated and
    /// `loaded` is set `true`. Otherwise the encoder is returned
    /// unloaded — `loaded` stays `false` and the caller must invoke
    /// `load_model` with a user-provided blob before DRED payloads can
    /// be emitted. Matches the C `dred_encoder_init` + `opus_decoder`
    /// auto-load pattern established by stage 7b.1.5.
    pub fn new(fs: i32, channels: i32) -> Self {
        let mut enc = Self {
            fs,
            channels,
            ..Self::default()
        };
        if !WEIGHTS_BLOB.is_empty() {
            let _ = enc.load_model(WEIGHTS_BLOB);
        }
        enc
    }

    /// Construct a DREDEnc without attempting an embedded-blob load.
    /// Useful for tests and for call-sites that will always set the
    /// blob explicitly via a later `load_model`.
    pub fn new_unloaded(fs: i32, channels: i32) -> Self {
        Self {
            fs,
            channels,
            ..Self::default()
        }
    }

    /// Load model weights from a serialised binary weight blob.
    /// Matches C `dred_encoder_load_model` in `dred_encoder.c`:
    /// first `init_rdovaeenc` from the parsed arrays, then
    /// `lpcnet_encoder_load_model` (which re-parses internally and
    /// populates the encoder's `pitchdnn` sub-model). Only when both
    /// succeed is `loaded` flipped to `true`.
    /// Returns `Ok(())` on success; `Err(-1)` on parse failure or a
    /// missing RDOVAE/LPCNet weight array.
    pub fn load_model(&mut self, data: &[u8]) -> Result<(), i32> {
        let arrays = parse_weights(data)?;
        self.model = init_rdovaeenc(&arrays).map_err(|_| -1)?;
        // Mirror C's second composite step — load the LPCNet encoder-state
        // weights (currently the `pitchdnn` sub-model). `LPCNetEncState::
        // load_model` returns 0 on success, non-zero on any parse or init
        // failure.
        if self.lpcnet_enc_state.load_model(data) != 0 {
            self.loaded = false;
            return Err(-1);
        }
        self.loaded = true;
        self.reset();
        Ok(())
    }

    /// Zero the running state (not the model weights).
    /// Matches C `dred_encoder_reset`'s memset-from-`DREDENC_RESET_START`,
    /// followed by `input_buffer_fill = DRED_SILK_ENCODER_DELAY` and the
    /// sub-state re-inits (`lpcnet_encoder_init` + `DRED_rdovae_init_encoder`).
    ///
    /// Note: in the C reference `lpcnet_encoder_init` auto-loads the
    /// compile-time pitchdnn weights when `USE_WEIGHTS_FILE` is undefined.
    /// We don't replicate that here — the Rust composite flow relies on a
    /// subsequent `load_model(blob)` to populate the LPCNet sub-model, and
    /// `new()` chains them so external callers see the same end state.
    pub fn reset(&mut self) {
        self.input_buffer = [0.0; 2 * DRED_DFRAME_SIZE];
        self.input_buffer_fill = DRED_SILK_ENCODER_DELAY;
        self.dred_offset = 0;
        self.latent_offset = 0;
        self.last_extra_dred_offset = 0;
        self.latents_buffer = [0.0; DRED_MAX_FRAMES * DRED_LATENT_DIM];
        self.latents_buffer_fill = 0;
        self.state_buffer = [0.0; DRED_MAX_FRAMES * DRED_STATE_DIM];
        self.resample_mem = [0.0; RESAMPLING_ORDER + 1];
        self.rdovae_enc = RDOVAEEncState::default();
    }

    /// Resample a block of PCM from the encoder's native sample rate to the
    /// internal 16 kHz DRED rate and append to `input_buffer`. Matches C
    /// `dred_convert_to_16k` in `dred_encoder.c`:
    ///
    /// - The input is optionally zero-stuffed by `up` (to produce a polyphase
    ///   pre-filter input), a signed-int16 quantisation ± a tiny dither, then
    ///   run through one of four fixed ellip(7,.2,70) biquad cascades (one
    ///   per supported source rate), and finally decimated by 3 on the 48 kHz
    ///   path (or 3 after a ×4 upsample on 12 kHz, or 1:1 on 8/16 kHz).
    ///
    /// Only the 48 kHz (→ 16 kHz, decimate by 3 after bypass-upsample),
    /// 24 kHz (→ 16 kHz, decimate by 3 after ×2 upsample), 16 kHz (passthrough),
    /// 12 kHz (→ 16 kHz, decimate by 3 after ×4 upsample), and 8 kHz (→ 16 kHz,
    /// direct 8→16 filter path) code paths are ported. `ENABLE_QEXT` / 96 kHz
    /// is not ported (matches `harness-deep-plc`'s build defines).
    ///
    /// # Panics
    ///
    /// Debug-only: panics if `out.len() * self.fs != in_len * 16000` or if
    /// `self.channels * in_len > MAX_DOWNMIX_BUFFER`. Matches the two
    /// `celt_assert` guards in the C reference.
    fn convert_to_16k(&mut self, pcm: &[f32], in_len: usize, out: &mut [f32], out_len: usize) {
        // Mirror of C `MAX_DOWNMIX_BUFFER` with `ENABLE_QEXT` off.
        const MAX_DOWNMIX_BUFFER: usize = 960 * 2;
        let mut downmix = [0.0f32; MAX_DOWNMIX_BUFFER];

        debug_assert!((self.channels as usize) * in_len <= MAX_DOWNMIX_BUFFER);
        debug_assert_eq!(
            in_len as i32 * 16000,
            out_len as i32 * self.fs,
            "in_len * 16000 must equal out_len * Fs"
        );
        let up: usize = match self.fs {
            8000 => 2,
            12000 => 4,
            16000 => 1,
            24000 => 2,
            48000 => 1,
            _ => unreachable!("unsupported Fs {} — Opus only allows 8/12/16/24/48 kHz", self.fs),
        };
        debug_assert!(up * in_len <= MAX_DOWNMIX_BUFFER);
        // Zero-stuff at stride `up` so odd entries are 0 and even entries
        // carry the ×up-scaled quantised mono mix. Matches C memset +
        // strided overwrite.
        for x in &mut downmix[..up * in_len] {
            *x = 0.0;
        }
        if self.channels == 1 {
            for i in 0..in_len {
                // VERY_SMALL = 1e-30f in float mode. The tiny dither prevents
                // denormals when the input is exactly zero; it's +ve-biased
                // so it survives rounding.
                downmix[up * i] = float2int16(up as f32 * pcm[i]) as f32 + 1.0e-30f32;
            }
        } else {
            for i in 0..in_len {
                let mono = 0.5 * up as f32 * (pcm[2 * i] + pcm[2 * i + 1]);
                downmix[up * i] = float2int16(mono) as f32 + 1.0e-30f32;
            }
        }
        match self.fs {
            16000 => {
                out[..out_len].copy_from_slice(&downmix[..out_len]);
            }
            48000 | 24000 => {
                // ellip(7, .2, 70, 7750/24000)
                const FILTER_B: [f32; 8] = [
                    0.005873358047,
                    0.012980854831,
                    0.014531340042,
                    0.014531340042,
                    0.012980854831,
                    0.005873358047,
                    0.004523418224,
                    0.0,
                ];
                const FILTER_A: [f32; 8] = [
                    -3.878718597768,
                    7.748834257468,
                    -9.653651699533,
                    8.007342726666,
                    -4.379450178552,
                    1.463182111810,
                    -0.231720677804,
                    0.0,
                ];
                const B0: f32 = 0.004523418224;
                // In-place filter (in_ptr == out_ptr in the C source).
                filter_df2t_inplace(
                    &mut downmix[..up * in_len],
                    B0,
                    &FILTER_B,
                    &FILTER_A,
                    RESAMPLING_ORDER,
                    &mut self.resample_mem,
                );
                // Decimate by 3.
                for i in 0..out_len {
                    out[i] = downmix[3 * i];
                }
            }
            12000 => {
                // ellip(7, .2, 70, 5800/24000)
                const FILTER_B: [f32; 8] = [
                    -0.001017101081,
                    0.003673127243,
                    0.001009165267,
                    0.001009165267,
                    0.003673127243,
                    -0.001017101081,
                    0.002033596776,
                    0.0,
                ];
                const FILTER_A: [f32; 8] = [
                    -4.930414411612,
                    11.291643096504,
                    -15.322037343815,
                    13.216403930898,
                    -7.220409219553,
                    2.310550142771,
                    -0.334338618782,
                    0.0,
                ];
                const B0: f32 = 0.002033596776;
                filter_df2t_inplace(
                    &mut downmix[..up * in_len],
                    B0,
                    &FILTER_B,
                    &FILTER_A,
                    RESAMPLING_ORDER,
                    &mut self.resample_mem,
                );
                for i in 0..out_len {
                    out[i] = downmix[3 * i];
                }
            }
            8000 => {
                // ellip(7, .2, 70, 3900/8000)
                const FILTER_B: [f32; 8] = [
                    0.081670120929,
                    0.180401598565,
                    0.259391051971,
                    0.259391051971,
                    0.180401598565,
                    0.081670120929,
                    0.020109185709,
                    0.0,
                ];
                const FILTER_A: [f32; 8] = [
                    -1.393651933659,
                    2.609789872676,
                    -2.403541968806,
                    2.056814957331,
                    -1.148908574570,
                    0.473001413788,
                    -0.110359852412,
                    0.0,
                ];
                const B0: f32 = 0.020109185709;
                // Unlike the other paths, the 8 kHz filter writes directly
                // into `out` (no decimation stage).
                filter_df2t(
                    &mut out[..out_len],
                    &downmix[..up * in_len],
                    B0,
                    &FILTER_B,
                    &FILTER_A,
                    RESAMPLING_ORDER,
                    &mut self.resample_mem,
                );
            }
            _ => unreachable!(),
        }
    }

    /// Process one 2*DRED_FRAME_SIZE = 320-sample (20 ms at 16 kHz) chunk
    /// of input buffer: shift the latent/state ring buffers down one slot,
    /// run two LPCNet feature-extraction frames, and push the 2×20-wide
    /// feature concat through the RDOVAE encoder. Matches C
    /// `dred_process_frame` in `dred_encoder.c`.
    fn process_frame(&mut self) {
        debug_assert!(self.loaded);

        // Shift latents buffer: latents_buffer[DRED_LATENT_DIM..] <- latents_buffer[..-DRED_LATENT_DIM]
        self.latents_buffer.copy_within(
            0..(DRED_MAX_FRAMES - 1) * DRED_LATENT_DIM,
            DRED_LATENT_DIM,
        );
        self.state_buffer.copy_within(
            0..(DRED_MAX_FRAMES - 1) * DRED_STATE_DIM,
            DRED_STATE_DIM,
        );

        // Two back-to-back LPCNet feature frames.
        let mut feature_buffer = [0.0f32; 2 * NB_TOTAL_FEATURES];
        let mut f0 = [0.0f32; NB_TOTAL_FEATURES];
        let mut f1 = [0.0f32; NB_TOTAL_FEATURES];
        self.lpcnet_enc_state
            .compute_single_frame_features_float(&self.input_buffer[..DRED_FRAME_SIZE], &mut f0);
        self.lpcnet_enc_state.compute_single_frame_features_float(
            &self.input_buffer[DRED_FRAME_SIZE..2 * DRED_FRAME_SIZE],
            &mut f1,
        );
        feature_buffer[..NB_TOTAL_FEATURES].copy_from_slice(&f0);
        feature_buffer[NB_TOTAL_FEATURES..2 * NB_TOTAL_FEATURES].copy_from_slice(&f1);

        // RDOVAE input is two concatenated DRED_NUM_FEATURES (=20) vectors
        // — LPCNet emits 36 features per frame but the last 16 are LPC
        // coefficients the RDOVAE encoder doesn't consume, so they're
        // discarded here. Matches C:
        //   OPUS_COPY(input_buffer, feature_buffer, DRED_NUM_FEATURES);
        //   OPUS_COPY(input_buffer + DRED_NUM_FEATURES, feature_buffer + 36, DRED_NUM_FEATURES);
        let mut rdovae_input = [0.0f32; 2 * DRED_NUM_FEATURES];
        rdovae_input[..DRED_NUM_FEATURES]
            .copy_from_slice(&feature_buffer[..DRED_NUM_FEATURES]);
        rdovae_input[DRED_NUM_FEATURES..]
            .copy_from_slice(&feature_buffer[NB_TOTAL_FEATURES..NB_TOTAL_FEATURES + DRED_NUM_FEATURES]);

        // Write latent + state into the head of the ring buffers.
        let (latent_head, _) = self.latents_buffer.split_at_mut(DRED_LATENT_DIM);
        let (state_head, _) = self.state_buffer.split_at_mut(DRED_STATE_DIM);
        self.rdovae_enc
            .encode_dframe(&self.model, latent_head, state_head, &rdovae_input);

        self.latents_buffer_fill =
            (self.latents_buffer_fill + 1).min(DRED_NUM_REDUNDANCY_FRAMES as i32);
    }

    /// Resample a block of 48/24/16/12/8 kHz PCM to 16 kHz, buffer up in
    /// `input_buffer`, and whenever we accumulate 2*DRED_FRAME_SIZE = 320
    /// samples (20 ms at 16 kHz) run `process_frame` to emit one latent +
    /// state pair. Matches C `dred_compute_latents`.
    ///
    /// `frame_size` is in samples at `self.fs` (not at 16 kHz).
    /// `extra_delay` is the total SILK+CELT lookahead in samples at
    /// `self.fs`, contributed by the Opus encoder's frame layout.
    pub fn compute_latents(&mut self, pcm: &[f32], frame_size: i32, extra_delay: i32) {
        debug_assert!(self.loaded);
        // TODO(stage-8.x): the channel-stride pointer arithmetic below
        // matches the C reference's pcm+1 / pcm+2 stride pattern in
        // `convert_to_16k`, but bit-exactness has only been validated for
        // mono input. Revisit against the C reference before enabling
        // stereo DRED.
        debug_assert!(
            self.channels == 1,
            "TODO(stage-8.x): verify DRED stereo against C ref"
        );
        // The C reference maintains `curr_offset16k` across the loop but
        // never reads it after the initial `dred_offset` computation.
        // Rust tracks that explicitly to avoid a dead-store warning;
        // `#[allow]` on the read would still flag the write.
        let curr_offset_16k =
            40 + extra_delay * 16000 / self.fs - self.input_buffer_fill;
        // `(a + 20) / 40` with floor semantics for positive and negative a.
        // Matches C `(int)floor((curr_offset16k + 20.f) / 40.f)`.
        self.dred_offset = ((curr_offset_16k as f32 + 20.0) / 40.0).floor() as i32;
        self.latent_offset = 0;

        let mut pcm_offset: usize = 0;
        let mut frame_size_16k = (frame_size * 16000) / self.fs;
        while frame_size_16k > 0 {
            let process_size_16k = (2 * DRED_FRAME_SIZE as i32).min(frame_size_16k);
            let process_size = process_size_16k * self.fs / 16000;
            let channel_stride = self.channels as usize;
            let pcm_slice = &pcm[pcm_offset * channel_stride
                ..pcm_offset * channel_stride + process_size as usize * channel_stride];
            let buf_start = self.input_buffer_fill as usize;
            // Scratch copy: `convert_to_16k` takes an immutable `pcm` and a
            // mutable `out` that cannot alias each other.
            let mut scratch = [0.0f32; 2 * DRED_FRAME_SIZE];
            self.convert_to_16k(
                pcm_slice,
                process_size as usize,
                &mut scratch[..process_size_16k as usize],
                process_size_16k as usize,
            );
            self.input_buffer[buf_start..buf_start + process_size_16k as usize]
                .copy_from_slice(&scratch[..process_size_16k as usize]);
            self.input_buffer_fill += process_size_16k;
            if self.input_buffer_fill >= 2 * DRED_FRAME_SIZE as i32 {
                // C does `curr_offset16k += 320` here but the variable is
                // dead after this block; the Rust-side mutation is omitted.
                self.process_frame();
                self.input_buffer_fill -= 2 * DRED_FRAME_SIZE as i32;
                // Shift remainder to the front. `copy_within` handles overlap.
                if self.input_buffer_fill > 0 {
                    self.input_buffer.copy_within(
                        2 * DRED_FRAME_SIZE..2 * DRED_FRAME_SIZE + self.input_buffer_fill as usize,
                        0,
                    );
                }
                // 15 ms = 6 * 2.5 ms = ideal offset (vocoder look-ahead).
                if self.dred_offset < 6 {
                    self.dred_offset += 8;
                } else {
                    self.latent_offset += 1;
                }
            }
            pcm_offset += process_size as usize;
            frame_size_16k -= process_size_16k;
        }
    }

    /// Range-code the accumulated latents + initial state into a DRED
    /// extension payload. Matches C `dred_encode_silk_frame` in
    /// `dred_encoder.c`.
    ///
    /// - `buf` is the output buffer, sized `max_bytes`. The raw DRED
    ///   payload (no 2-byte `'D' + version` experimental prefix) is written
    ///   starting at `buf[0]`.
    /// - `max_chunks` caps the number of 40 ms latent chunks encoded
    ///   (1 chunk = 2 consecutive 20 ms frames).
    /// - `q0` is the base quantiser (0..15). `dQ` is an 8-value step
    ///   selector; `qmax` clips the quantiser escalation.
    /// - `activity_mem` is the 2.5 ms-resolution voice-activity ring buffer
    ///   from the enclosing `OpusEncoder` (`4*DRED_MAX_FRAMES = 416` bytes).
    ///   Entries come from the SILK VAD; we just read them.
    ///
    /// Returns the number of payload bytes actually written, or 0 when
    /// budget was exhausted before a single chunk could be coded (or when
    /// the only codeable region was silence).
    pub fn encode_silk_frame(
        &mut self,
        buf: &mut [u8],
        max_chunks: i32,
        max_bytes: usize,
        q0: i32,
        d_q: i32,
        qmax: i32,
        activity_mem: &[u8],
    ) -> i32 {
        let buf_slice = &mut buf[..max_bytes];
        let mut ec_encoder = RangeEncoder::new(buf_slice);

        let mut latent_offset = self.latent_offset;
        let mut extra_dred_offset: i32 = 0;
        let mut delayed_dred = false;

        // Delaying new DRED data when just out of silence because we
        // already have the main Opus payload for that frame.
        if activity_mem[0] != 0 && self.last_extra_dred_offset > 0 {
            latent_offset = self.last_extra_dred_offset;
            delayed_dred = true;
            self.last_extra_dred_offset = 0;
        }
        while latent_offset < self.latents_buffer_fill
            && !dred_voice_active(activity_mem, latent_offset)
        {
            latent_offset += 1;
            extra_dred_offset += 1;
        }
        if !delayed_dred {
            self.last_extra_dred_offset = extra_dred_offset;
        }

        // Entropy-coded header: quantiser base + step selector + offset.
        ec_encoder.encode_uint(q0 as u32, 16);
        ec_encoder.encode_uint(d_q as u32, 8);
        let total_offset = 16 - (self.dred_offset - extra_dred_offset * 8);
        debug_assert!(total_offset >= 0);
        if total_offset > 31 {
            ec_encoder.encode_uint(1, 2);
            ec_encoder.encode_uint((total_offset >> 5) as u32, 256);
            ec_encoder.encode_uint((total_offset & 31) as u32, 32);
        } else {
            ec_encoder.encode_uint(0, 2);
            ec_encoder.encode_uint(total_offset as u32, 32);
        }
        debug_assert!(qmax >= q0);
        if q0 < 14 && d_q > 0 {
            // If you want to use qmax == q0, you should have set dQ = 0.
            debug_assert!(qmax > q0);
            let nvals = 15 - (q0 + 1);
            let (fl, fh) = if qmax >= 15 {
                (0, nvals)
            } else {
                (nvals + qmax - (q0 + 1), nvals + qmax - q0)
            };
            ec_encoder.encode(fl as u32, fh as u32, (2 * nvals) as u32);
        }

        let state_qoffset = (q0 as usize) * DRED_STATE_DIM;
        let state_slice_start = (latent_offset as usize) * DRED_STATE_DIM;
        // Snapshot state buffer read so we can pass it alongside a &mut
        // encoder without a borrow conflict. State stays in self so the
        // reference is live across the call.
        let state_buf: Vec<f32> = self.state_buffer
            [state_slice_start..state_slice_start + DRED_STATE_DIM]
            .to_vec();
        dred_encode_latents(
            &mut ec_encoder,
            &state_buf,
            &dred_state_quant_scales_q8[state_qoffset..state_qoffset + DRED_STATE_DIM],
            &dred_state_dead_zone_q8[state_qoffset..state_qoffset + DRED_STATE_DIM],
            &dred_state_r_q8[state_qoffset..state_qoffset + DRED_STATE_DIM],
            &dred_state_p0_q8[state_qoffset..state_qoffset + DRED_STATE_DIM],
        );
        if ec_encoder.tell() > 8 * max_bytes as i32 {
            return 0;
        }

        // Snapshot encoder state (ec_bak) so we can roll back to the last
        // voice-active chunk when we finish the loop. Matches C
        // `ec_enc ec_bak = ec_encoder;`.
        let mut ec_bak = ec_encoder.save_snapshot();

        let mut prev_active = false;
        let mut dred_encoded: i32 = 0;
        let chunk_count_cap = 2 * max_chunks.min(self.latents_buffer_fill - latent_offset - 1);
        let mut i = 0i32;
        while i < chunk_count_cap {
            let q_level = compute_quantizer(q0, d_q, qmax, i / 2);
            let offset = (q_level as usize) * DRED_LATENT_DIM;
            let latent_idx = (i + latent_offset) as usize;
            let latent_slice_start = latent_idx * DRED_LATENT_DIM;
            let latent_buf: Vec<f32> = self.latents_buffer
                [latent_slice_start..latent_slice_start + DRED_LATENT_DIM]
                .to_vec();
            dred_encode_latents(
                &mut ec_encoder,
                &latent_buf,
                &dred_latent_quant_scales_q8[offset..offset + DRED_LATENT_DIM],
                &dred_latent_dead_zone_q8[offset..offset + DRED_LATENT_DIM],
                &dred_latent_r_q8[offset..offset + DRED_LATENT_DIM],
                &dred_latent_p0_q8[offset..offset + DRED_LATENT_DIM],
            );
            if ec_encoder.tell() > 8 * max_bytes as i32 {
                // If we haven't been able to code one chunk, give up on
                // DRED completely.
                if i == 0 {
                    return 0;
                }
                break;
            }
            let active = dred_voice_active(activity_mem, i + latent_offset);
            if active || prev_active {
                ec_bak = ec_encoder.save_snapshot();
                dred_encoded = i + 2;
            }
            prev_active = active;
            i += 2;
        }

        // Avoid sending empty DRED packets.
        if dred_encoded == 0 || (dred_encoded <= 2 && extra_dred_offset != 0) {
            return 0;
        }
        ec_encoder.restore_snapshot(&ec_bak);

        let ec_buffer_fill = (ec_encoder.tell() + 7) / 8;
        ec_encoder.shrink(ec_buffer_fill as u32);
        ec_encoder.done();
        ec_buffer_fill
    }
}

// ===========================================================================
// DRED coding helpers (free functions)
// ===========================================================================

/// Direct-form-II transposed 8-tap IIR filter with stride-1 input and
/// stride-1 output. Matches C `filter_df2t` in `dred_encoder.c`.
///
/// `b0` is the leading feed-forward coefficient (x[i] → y[i] direct path),
/// `filter_b[0..order]` are the remaining feed-forward coefficients mixed
/// into the delay line, `filter_a[0..order]` are the (negated) feedback
/// coefficients. `mem[0..order+1]` is the transposed-form state memory;
/// `mem[order]` is always 0 (unused write slot).
fn filter_df2t(
    out: &mut [f32],
    input: &[f32],
    b0: f32,
    filter_b: &[f32; 8],
    filter_a: &[f32; 8],
    order: usize,
    mem: &mut [f32],
) {
    let len = out.len();
    debug_assert!(input.len() >= len);
    debug_assert!(mem.len() > order);
    for i in 0..len {
        let xi = input[i];
        let yi = xi * b0 + mem[0];
        let nyi = -yi;
        for j in 0..order {
            mem[j] = mem[j + 1] + filter_b[j] * xi + filter_a[j] * nyi;
        }
        out[i] = yi;
    }
}

/// In-place variant of `filter_df2t`. The C source uses `in == out` at
/// the 48/24/12 kHz call sites, which Rust's aliasing rules disallow via
/// separate slices; this variant reads and writes the same buffer.
fn filter_df2t_inplace(
    io: &mut [f32],
    b0: f32,
    filter_b: &[f32; 8],
    filter_a: &[f32; 8],
    order: usize,
    mem: &mut [f32],
) {
    let len = io.len();
    debug_assert!(mem.len() > order);
    for i in 0..len {
        let xi = io[i];
        let yi = xi * b0 + mem[0];
        let nyi = -yi;
        for j in 0..order {
            mem[j] = mem[j + 1] + filter_b[j] * xi + filter_a[j] * nyi;
        }
        io[i] = yi;
    }
}

/// Compute the per-chunk quantiser level. Matches C `compute_quantizer` in
/// `dred_coding.c`.
pub fn compute_quantizer(q0: i32, d_q: i32, qmax: i32, i: i32) -> i32 {
    const D_Q_TABLE: [i32; 8] = [0, 2, 3, 4, 6, 8, 12, 16];
    let quant = q0 + (D_Q_TABLE[d_q as usize] * i + 8) / 16;
    if quant > qmax { qmax } else { quant }
}

/// VAD probe into the 2.5 ms-resolution `activity_mem` ring buffer. Matches
/// C `dred_voice_active` in `dred_encoder.c`: a 20 ms chunk is "active" if
/// any of its 8 2.5 ms sub-frames crossed the SILK VAD threshold.
fn dred_voice_active(activity_mem: &[u8], offset: i32) -> bool {
    let base = 8 * offset as usize;
    for i in 0..16 {
        if activity_mem[base + i] == 1 {
            return true;
        }
    }
    false
}

/// Encode `dim` floats through the asymmetric deadzone quantiser + Laplace
/// range coder. Matches C `dred_encode_latents` in `dred_encoder.c`.
///
/// `scale`, `dzone`, `r`, `p0` all point at `dim`-length u8 slices (per-
/// dimension quantiser parameters from `dred_stats`). When `r[i] == 0` or
/// `p0[i] == 255` the output bit is forced to zero (impossible dim), so
/// no range-coder call is made for that dim.
fn dred_encode_latents(
    ec: &mut RangeEncoder,
    x: &[f32],
    scale: &[u8],
    dzone: &[u8],
    r: &[u8],
    p0: &[u8],
) {
    let dim = x.len();
    // DRED_LATENT_DIM = 25, DRED_STATE_DIM = 50 — allocate for the larger.
    debug_assert!(dim <= DRED_STATE_DIM);
    let mut q_arr = [0i32; DRED_STATE_DIM];
    let mut xq_arr = [0.0f32; DRED_STATE_DIM];
    let mut delta_arr = [0.0f32; DRED_STATE_DIM];
    let mut deadzone_arr = [0.0f32; DRED_STATE_DIM];
    const EPS: f32 = 0.1;

    for i in 0..dim {
        delta_arr[i] = dzone[i] as f32 * (1.0 / 256.0);
        xq_arr[i] = x[i] * scale[i] as f32 * (1.0 / 256.0);
        deadzone_arr[i] = xq_arr[i] / (delta_arr[i] + EPS);
    }
    compute_activation(&mut deadzone_arr[..dim], dim, ACTIVATION_TANH);
    for i in 0..dim {
        xq_arr[i] -= delta_arr[i] * deadzone_arr[i];
        // `floor(0.5 + x)` — C uses this exact form; Rust `.floor()`
        // matches the IEEE-754 round-towards-negative-infinity the C
        // compilers emit. For negative half-integers (e.g. -2.5), C
        // `floor(-2.0)` = -2, which matches Rust's `(-2.0_f32).floor()`.
        q_arr[i] = (0.5 + xq_arr[i]).floor() as i32;
    }
    for i in 0..dim {
        // Skip dims the stats say can't produce a nonzero output.
        if r[i] == 0 || p0[i] == 255 {
            q_arr[i] = 0;
        } else {
            ec.encode_laplace_p0(
                q_arr[i],
                (p0[i] as u16) << 7,
                (r[i] as u16) << 7,
            );
        }
    }
}

// ===========================================================================
// OpusDred — decoded payload state
// ===========================================================================

/// Decoded DRED payload state. Matches C `struct OpusDRED`
/// (`reference/dnn/dred_decoder.h`). Stores the decoded latents /
/// state that the RDOVAE decoder turns into reconstructed LPCNet
/// features, ready for the FARGAN last-mile (stage 7b seam).
#[derive(Clone, Debug)]
pub struct OpusDred {
    pub fec_features: [f32; 2 * DRED_NUM_REDUNDANCY_FRAMES * DRED_NUM_FEATURES],
    pub state: [f32; DRED_STATE_DIM],
    pub latents: [f32; (DRED_NUM_REDUNDANCY_FRAMES / 2) * (DRED_LATENT_DIM + 1)],
    pub nb_latents: i32,
    pub process_stage: i32,
    pub dred_offset: i32,
}

impl Default for OpusDred {
    fn default() -> Self {
        Self {
            fec_features: [0.0; 2 * DRED_NUM_REDUNDANCY_FRAMES * DRED_NUM_FEATURES],
            state: [0.0; DRED_STATE_DIM],
            latents: [0.0; (DRED_NUM_REDUNDANCY_FRAMES / 2) * (DRED_LATENT_DIM + 1)],
            nb_latents: 0,
            process_stage: 0,
            dred_offset: 0,
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Confirm the blob format — when the crate ships with DRED weights
    /// embedded — parses through both RDOVAE init paths without panic.
    /// No-op (skipped) when the blob is empty, which happens on bare
    /// checkouts that haven't fetched the reference DNN data files.
    #[test]
    fn embedded_blob_initialises_rdovae_enc_and_dec() {
        if WEIGHTS_BLOB.is_empty() {
            // Empty blob path — nothing to validate. Explicit so CI
            // doesn't silently pass on a misconfigured tree.
            eprintln!(
                "dred::tests: WEIGHTS_BLOB empty — skipping RDOVAE init check \
                 (run `cargo run -p fetch-assets -- weights` to populate)."
            );
            return;
        }

        let arrays = parse_weights(WEIGHTS_BLOB).expect("parse_weights blob");
        assert!(!arrays.is_empty(), "blob parsed but empty");

        init_rdovaeenc(&arrays).expect(
            "init_rdovaeenc: blob missing a required RDOVAE encoder weight array — \
             check that gen_weights_blob.c emits rdovaeenc_arrays",
        );
        init_rdovaedec(&arrays).expect(
            "init_rdovaedec: blob missing a required RDOVAE decoder weight array — \
             check that gen_weights_blob.c emits rdovaedec_arrays",
        );
    }

    #[test]
    fn dred_enc_default_state_is_zeroed() {
        let enc = DREDEnc::new_unloaded(48000, 1);
        assert!(!enc.loaded);
        assert_eq!(enc.fs, 48000);
        assert_eq!(enc.channels, 1);
        assert_eq!(enc.input_buffer_fill, 0);
        assert_eq!(enc.dred_offset, 0);
        assert!(enc.input_buffer.iter().all(|&x| x == 0.0));
        assert!(enc.latents_buffer.iter().all(|&x| x == 0.0));
        assert!(enc.state_buffer.iter().all(|&x| x == 0.0));
        assert!(!enc.rdovae_enc.initialized);
    }

    #[test]
    fn dred_new_attempts_embedded_load_and_reports_status() {
        // Stable regardless of whether the embedded blob is present —
        // new() never panics, and `loaded` accurately reflects whether
        // weights were found.
        let enc = DREDEnc::new(48000, 2);
        if WEIGHTS_BLOB.is_empty() {
            assert!(!enc.loaded);
        } else {
            assert!(
                enc.loaded,
                "embedded blob is present but RDOVAE enc init didn't populate the model"
            );
        }
    }

    #[test]
    fn load_model_rejects_empty_blob() {
        let mut enc = DREDEnc::new_unloaded(48000, 1);
        assert_eq!(enc.load_model(&[]), Err(-1));
        assert!(!enc.loaded);
    }

    #[test]
    fn opus_dred_default_is_zeroed() {
        let d = OpusDred::default();
        assert_eq!(d.nb_latents, 0);
        assert_eq!(d.process_stage, 0);
        assert_eq!(d.dred_offset, 0);
        assert!(d.fec_features.iter().all(|&x| x == 0.0));
        assert!(d.state.iter().all(|&x| x == 0.0));
        assert!(d.latents.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn constants_match_c_reference() {
        assert_eq!(DRED_EXTENSION_ID, 126);
        assert_eq!(DRED_EXPERIMENTAL_VERSION, 12);
        assert_eq!(DRED_FRAME_SIZE, 160);
        assert_eq!(DRED_DFRAME_SIZE, 320);
        assert_eq!(DRED_MAX_LATENTS, 26);
        assert_eq!(DRED_NUM_REDUNDANCY_FRAMES, 52);
        assert_eq!(DRED_MAX_FRAMES, 104);
        assert_eq!(DRED_NUM_FEATURES, 20);
        assert_eq!(DRED_LATENT_DIM, 25);
        assert_eq!(DRED_STATE_DIM, 50);
        assert_eq!(DRED_PADDED_LATENT_DIM, 32);
        assert_eq!(DRED_PADDED_STATE_DIM, 56);
        assert_eq!(DRED_MAX_CONV_INPUTS, 128);
    }

    /// Smoke test for `RDOVAEEncState::encode_dframe` — confirm the forward
    /// pass runs without panic on a tiny deterministic input, returns the
    /// expected output shapes, and promotes `initialized` from `false` to
    /// `true` on the first call. Skipped when the embedded DRED weight
    /// blob is empty (bare checkout without fetched DNN data files).
    ///
    /// This is a shape / liveness check only; the C-differential tier-1
    /// (or tier-2 SNR) oracle lives in `harness-deep-plc/tests/`.
    #[test]
    fn encode_dframe_liveness_check() {
        if WEIGHTS_BLOB.is_empty() {
            eprintln!(
                "dred::tests: WEIGHTS_BLOB empty — skipping encode_dframe smoke test."
            );
            return;
        }
        let arrays = parse_weights(WEIGHTS_BLOB).expect("parse_weights blob");
        let model = init_rdovaeenc(&arrays).expect("init_rdovaeenc");

        // Deterministic 40-wide feature input (two concatenated LPCNet
        // feature vectors). Small magnitudes keep the tanh stack out of
        // saturation so the output isn't all ±1.
        let mut input = [0.0f32; 2 * DRED_NUM_FEATURES];
        for (i, x) in input.iter_mut().enumerate() {
            *x = ((i as f32) * 0.017 - 0.3).sin() * 0.2;
        }

        let mut state = RDOVAEEncState::default();
        assert!(!state.initialized);

        let mut latents = [0.0f32; DRED_LATENT_DIM];
        let mut initial = [0.0f32; DRED_STATE_DIM];
        state.encode_dframe(&model, &mut latents, &mut initial, &input);

        assert!(state.initialized, "conv1_cond_init must flip initialized");

        // At least one output in each bank is nonzero and all are finite.
        assert!(
            latents.iter().all(|v| v.is_finite()),
            "latents contain non-finite values: {latents:?}"
        );
        assert!(
            initial.iter().all(|v| v.is_finite()),
            "initial_state contains non-finite values: {initial:?}"
        );
        assert!(
            latents.iter().any(|&v| v != 0.0),
            "latents all zero — forward pass looks inert"
        );
        assert!(
            initial.iter().any(|&v| v != 0.0),
            "initial_state all zero — forward pass looks inert"
        );
    }

    /// Smoke test for `RDOVAEDecState::init_states` +
    /// `RDOVAEDecState::decode_qframe` — confirm both run without panic on
    /// deterministic input, produce finite and nontrivial qframe output,
    /// and promote `initialized` from `false` to `true` on the first
    /// decode call. Skipped when the embedded DRED weight blob is empty
    /// (bare checkout without fetched DNN data files).
    ///
    /// This is a shape / liveness check only; the C-differential tier-1
    /// (or tier-2 SNR) oracle lives in `harness-deep-plc/tests/`.
    #[test]
    fn decode_qframe_liveness_check() {
        if WEIGHTS_BLOB.is_empty() {
            eprintln!(
                "dred::tests: WEIGHTS_BLOB empty — skipping decode_qframe smoke test."
            );
            return;
        }
        let arrays = parse_weights(WEIGHTS_BLOB).expect("parse_weights blob");
        let model = init_rdovaedec(&arrays).expect("init_rdovaedec");

        // 50-wide initial state (encoder-side `gdense2` output, simulated
        // here with a small deterministic ramp so the init path has
        // something nontrivial to project).
        let mut initial_state = [0.0f32; DRED_STATE_DIM];
        for (i, x) in initial_state.iter_mut().enumerate() {
            *x = ((i as f32) * 0.019 - 0.5).sin() * 0.1;
        }

        // 26-wide latent (25 quantised latents + 1 auxiliary byte slot).
        let mut input = [0.0f32; DRED_LATENT_DIM + 1];
        for (i, x) in input.iter_mut().enumerate() {
            *x = ((i as f32) * 0.023 + 0.1).cos() * 0.15;
        }

        let mut state = RDOVAEDecState::default();
        state.init_states(&model, &initial_state);
        // After init_states the flag must be `false` so the first decode
        // call performs the one-shot conv-memory zeroing.
        assert!(!state.initialized);

        let mut qframe = [0.0f32; DEC_OUTPUT_OUT_SIZE];
        state.decode_qframe(&model, &mut qframe, &input);

        assert!(state.initialized, "conv1_cond_init must flip initialized");
        assert!(
            qframe.iter().all(|v| v.is_finite()),
            "qframe contains non-finite values: {qframe:?}"
        );
        assert!(
            qframe.iter().any(|&v| v != 0.0),
            "qframe all zero — forward pass looks inert"
        );
    }
}
