//! PitchDNN — Neural pitch estimator for the Opus DNN subsystem.
//!
//! Matches C reference: `pitchdnn.c`, `pitchdnn.h`, `pitchdnn_data.h`.
//!
//! A small recurrent neural network that jointly processes instantaneous
//! frequency (IF) features and cross-correlation (xcorr) features to produce
//! a continuous log-frequency pitch estimate. Called once per frame from
//! the LPCNet encoder's `compute_frame_features`.
//!
//! Pipeline: IF features + xcorr features → conv2d → dense → GRU → soft argmax → pitch.

use super::core::{
    ACTIVATION_LINEAR, ACTIVATION_TANH, Conv2dLayer, LinearLayer, WeightArray, compute_conv2d,
    compute_generic_dense, compute_generic_gru, conv2d_init, linear_init, parse_weights,
};

// ===========================================================================
// Constants
// ===========================================================================

/// Minimum pitch period in samples (500 Hz at 16 kHz).
pub const PITCH_MIN_PERIOD: usize = 32;

/// Maximum pitch period in samples (62.5 Hz at 16 kHz).
pub const PITCH_MAX_PERIOD: usize = 256;

/// Number of cross-correlation features (lag offsets): 256 − 32 = 224.
pub const NB_XCORR_FEATURES: usize = PITCH_MAX_PERIOD - PITCH_MIN_PERIOD;

/// Number of instantaneous frequency features: 3 × 30 − 2 = 88.
pub const PITCH_IF_FEATURES: usize = 88;

// Model dimension constants (match pitchdnn_data.h, auto-generated from PyTorch model).
const GRU_1_STATE_SIZE: usize = 64;
const DENSE_IF_UPSAMPLER_1_OUT_SIZE: usize = 64;
const DENSE_IF_UPSAMPLER_2_OUT_SIZE: usize = 64;
const DENSE_DOWNSAMPLER_OUT_SIZE: usize = 64;
const DENSE_FINAL_UPSAMPLER_OUT_SIZE: usize = 192;

/// Downsampler input: xcorr conv output (224) + IF upsampled features (64) = 288.
const DOWNSAMPLER_IN_SIZE: usize = NB_XCORR_FEATURES + DENSE_IF_UPSAMPLER_2_OUT_SIZE;

/// Number of valid output bins for the soft argmax (bins 180–191 are unused).
const NUM_PITCH_BINS: usize = 180;

/// Half-width of the soft argmax window around the hard argmax peak.
const SOFT_ARGMAX_RADIUS: usize = 2;

// Conv2d memory sizing — exact required sizes (C uses oversized buffers).
//
// Conv2d_1: 1 in → 4 out channels, ktime=3, kheight=3
//   time_stride = in_channels × (height + kheight − 1) = 1 × 226 = 226
//   mem = (ktime − 1) × time_stride = 2 × 226 = 452
const XCORR_MEM1_SIZE: usize = 2 * (NB_XCORR_FEATURES + 2);

// Conv2d_2: 4 in → 1 out channels, ktime=3, kheight=3
//   time_stride = 4 × (224 + 2) = 904
//   mem = 2 × 904 = 1808
const XCORR_MEM2_SIZE: usize = 2 * 4 * (NB_XCORR_FEATURES + 2);

/// Temporary buffer size for conv2d intermediates.
/// Matches C: `(NB_XCORR_FEATURES + 2) * 8` = 1808 floats.
const CONV_TMP_SIZE: usize = (NB_XCORR_FEATURES + 2) * 8;

// ===========================================================================
// Types
// ===========================================================================

/// Neural network weights for pitch estimation.
/// Matches C `PitchDNN` struct from pitchdnn_data.h.
#[derive(Clone, Debug, Default)]
pub struct PitchDnn {
    /// IF feature projection layer 1: 88 → 64, quantized int8.
    pub dense_if_upsampler_1: LinearLayer,
    /// IF feature projection layer 2: 64 → 64, quantized int8.
    pub dense_if_upsampler_2: LinearLayer,
    /// Xcorr spatial convolution 1: 1→4 channels, 3×3 kernel, float weights.
    pub conv2d_1: Conv2dLayer,
    /// Xcorr spatial convolution 2: 4→1 channels, 3×3 kernel, float weights.
    pub conv2d_2: Conv2dLayer,
    /// Feature downsampler: 288 → 64, quantized int8.
    pub dense_downsampler: LinearLayer,
    /// GRU input weights: 64 → 192 (3×64), quantized int8.
    pub gru_1_input: LinearLayer,
    /// GRU recurrent weights: 64 → 192 (3×64), quantized int8.
    pub gru_1_recurrent: LinearLayer,
    /// Final projection to pitch logits: 64 → 192, quantized int8.
    pub dense_final_upsampler: LinearLayer,
}

/// Persistent state for the pitch DNN estimator.
/// Matches C `PitchDNNState` from pitchdnn.h.
///
/// The C struct also contains `xcorr_mem3` (3616 floats) which is vestigial
/// and never used — omitted here. `xcorr_mem2` is sized to the exact required
/// amount (1808 floats) rather than the C's oversized 3616.
#[derive(Clone, Debug)]
pub struct PitchDnnState {
    /// Neural network weights (initialized once, read-only during inference).
    pub model: PitchDnn,
    /// GRU hidden state, updated every frame. 64 floats.
    pub gru_state: Vec<f32>,
    /// Conv2d_1 temporal memory (2 previous time frames). 452 floats.
    pub xcorr_mem1: Vec<f32>,
    /// Conv2d_2 temporal memory (2 previous time frames). 1808 floats.
    pub xcorr_mem2: Vec<f32>,
}

// ===========================================================================
// Weight initialization
// ===========================================================================

/// Initialize PitchDnn model weights from parsed weight arrays.
/// Matches the auto-generated C `init_pitchdnn` from pitchdnn_data.c.
///
/// Weight array names follow the CWriter export convention:
///   dense layers: `{name}_bias`, `{name}_subias`, `{name}_weights_int8`,
///                 `{name}_weights_float`, `{name}_scale`
///   conv2d layers: `{name}_bias`, `{name}_weight_float`
///   GRU layers:    `{name}_input_*` and `{name}_recurrent_*`
pub fn init_pitchdnn(arrays: &[WeightArray]) -> Result<PitchDnn, ()> {
    // IF upsampler: 88 → 64 → 64
    let dense_if_upsampler_1 = linear_init(
        arrays,
        Some("dense_if_upsampler_1_bias"),
        Some("dense_if_upsampler_1_subias"),
        Some("dense_if_upsampler_1_weights_int8"),
        Some("dense_if_upsampler_1_weights_float"),
        None, // no sparse index
        None, // no diagonal
        Some("dense_if_upsampler_1_scale"),
        PITCH_IF_FEATURES,             // 88
        DENSE_IF_UPSAMPLER_1_OUT_SIZE, // 64
    )?;

    let dense_if_upsampler_2 = linear_init(
        arrays,
        Some("dense_if_upsampler_2_bias"),
        Some("dense_if_upsampler_2_subias"),
        Some("dense_if_upsampler_2_weights_int8"),
        Some("dense_if_upsampler_2_weights_float"),
        None,
        None,
        Some("dense_if_upsampler_2_scale"),
        DENSE_IF_UPSAMPLER_1_OUT_SIZE, // 64
        DENSE_IF_UPSAMPLER_2_OUT_SIZE, // 64
    )?;

    // Xcorr convolutions: 1→4→1 channels, 3×3 kernels
    let conv2d_1 = conv2d_init(
        arrays,
        Some("conv2d_1_bias"),
        Some("conv2d_1_weight_float"),
        1, // in_channels
        4, // out_channels
        3, // ktime
        3, // kheight
    )?;

    let conv2d_2 = conv2d_init(
        arrays,
        Some("conv2d_2_bias"),
        Some("conv2d_2_weight_float"),
        4, // in_channels
        1, // out_channels
        3, // ktime
        3, // kheight
    )?;

    // Downsampler: 288 → 64
    let dense_downsampler = linear_init(
        arrays,
        Some("dense_downsampler_bias"),
        Some("dense_downsampler_subias"),
        Some("dense_downsampler_weights_int8"),
        Some("dense_downsampler_weights_float"),
        None,
        None,
        Some("dense_downsampler_scale"),
        DOWNSAMPLER_IN_SIZE,        // 288
        DENSE_DOWNSAMPLER_OUT_SIZE, // 64
    )?;

    // GRU: input 64→192, recurrent 64→192
    let gru_1_input = linear_init(
        arrays,
        Some("gru_1_input_bias"),
        Some("gru_1_input_subias"),
        Some("gru_1_input_weights_int8"),
        Some("gru_1_input_weights_float"),
        None,
        None,
        Some("gru_1_input_scale"),
        DENSE_DOWNSAMPLER_OUT_SIZE, // 64
        3 * GRU_1_STATE_SIZE,       // 192
    )?;

    let gru_1_recurrent = linear_init(
        arrays,
        Some("gru_1_recurrent_bias"),
        Some("gru_1_recurrent_subias"),
        Some("gru_1_recurrent_weights_int8"),
        Some("gru_1_recurrent_weights_float"),
        None,
        None,
        Some("gru_1_recurrent_scale"),
        GRU_1_STATE_SIZE,     // 64
        3 * GRU_1_STATE_SIZE, // 192
    )?;

    // Final upsampler: 64 → 192 pitch logits
    let dense_final_upsampler = linear_init(
        arrays,
        Some("dense_final_upsampler_bias"),
        Some("dense_final_upsampler_subias"),
        Some("dense_final_upsampler_weights_int8"),
        Some("dense_final_upsampler_weights_float"),
        None,
        None,
        Some("dense_final_upsampler_scale"),
        GRU_1_STATE_SIZE,               // 64
        DENSE_FINAL_UPSAMPLER_OUT_SIZE, // 192
    )?;

    Ok(PitchDnn {
        dense_if_upsampler_1,
        dense_if_upsampler_2,
        conv2d_1,
        conv2d_2,
        dense_downsampler,
        gru_1_input,
        gru_1_recurrent,
        dense_final_upsampler,
    })
}

// ===========================================================================
// State management
// ===========================================================================

impl PitchDnnState {
    /// Create a new state with model weights loaded from weight arrays.
    /// Matches C `pitchdnn_init` (non-`USE_WEIGHTS_FILE` path).
    pub fn new(arrays: &[WeightArray]) -> Result<Self, ()> {
        Ok(Self {
            model: init_pitchdnn(arrays)?,
            gru_state: vec![0.0; GRU_1_STATE_SIZE],
            xcorr_mem1: vec![0.0; XCORR_MEM1_SIZE],
            xcorr_mem2: vec![0.0; XCORR_MEM2_SIZE],
        })
    }

    /// Create a new state without loading weights (for later `load_model()`).
    /// Matches C `pitchdnn_init` (`USE_WEIGHTS_FILE` path).
    pub fn new_empty() -> Self {
        Self {
            model: PitchDnn::default(),
            gru_state: vec![0.0; GRU_1_STATE_SIZE],
            xcorr_mem1: vec![0.0; XCORR_MEM1_SIZE],
            xcorr_mem2: vec![0.0; XCORR_MEM2_SIZE],
        }
    }

    /// Load model weights from a serialized binary weight blob.
    /// Matches C `pitchdnn_load_model`. Returns `Ok(())` on success, `Err(-1)` on failure.
    pub fn load_model(&mut self, data: &[u8]) -> Result<(), i32> {
        let arrays = parse_weights(data)?;
        self.model = init_pitchdnn(&arrays).map_err(|_| -1)?;
        Ok(())
    }

    // =======================================================================
    // Inference
    // =======================================================================

    /// Run one frame of pitch estimation through the neural network.
    /// Matches C `compute_pitchdnn` from pitchdnn.c.
    ///
    /// # Arguments
    /// * `if_features` — Instantaneous frequency features, at least 88 floats.
    /// * `xcorr_features` — Normalized cross-correlation values, at least 224 floats.
    ///
    /// # Returns
    /// Log-frequency pitch estimate. The caller converts to a pitch period via:
    /// `period = round(256 / 2^((result + 1.5) * 60 / 60))`.
    pub fn compute(&mut self, if_features: &[f32], xcorr_features: &[f32]) -> f32 {
        debug_assert!(if_features.len() >= PITCH_IF_FEATURES);
        debug_assert!(xcorr_features.len() >= NB_XCORR_FEATURES);

        // ── Step 1: IF Feature Upsampling ──
        // if_features[88] → dense_1 → tanh → if1_out[64]
        // if1_out[64]     → dense_2 → tanh → downsampler_in[224..288]
        let mut if1_out = [0.0f32; DENSE_IF_UPSAMPLER_1_OUT_SIZE];
        compute_generic_dense(
            &self.model.dense_if_upsampler_1,
            &mut if1_out,
            if_features,
            ACTIVATION_TANH,
        );

        let mut downsampler_in = [0.0f32; DOWNSAMPLER_IN_SIZE];
        compute_generic_dense(
            &self.model.dense_if_upsampler_2,
            &mut downsampler_in[NB_XCORR_FEATURES..],
            &if1_out,
            ACTIVATION_TANH,
        );

        // ── Step 2: Xcorr Convolution (Conv2d_1): 1→4 channels ──
        // Zero-initialized temporaries provide spatial zero-padding.
        // conv1_tmp1[0] = 0 (left pad), xcorr at [1..225], rest = 0.
        let mut conv1_tmp1 = [0.0f32; CONV_TMP_SIZE];
        let mut conv1_tmp2 = [0.0f32; CONV_TMP_SIZE];

        conv1_tmp1[1..1 + NB_XCORR_FEATURES].copy_from_slice(&xcorr_features[..NB_XCORR_FEATURES]);

        // Output at offset 1 with hstride=226 leaves [0]=0 per channel for conv2d_2's padding.
        compute_conv2d(
            &self.model.conv2d_1,
            &mut conv1_tmp2[1..],
            &mut self.xcorr_mem1,
            &conv1_tmp1,
            NB_XCORR_FEATURES,
            NB_XCORR_FEATURES + 2, // hstride: 226 (extra 2 for spatial padding)
            ACTIVATION_TANH,
        );

        // ── Step 3: Xcorr Convolution (Conv2d_2): 4→1 channels ──
        // Output directly to downsampler_in[0..224], preserving IF features at [224..288].
        compute_conv2d(
            &self.model.conv2d_2,
            &mut downsampler_in[..],
            &mut self.xcorr_mem2,
            &conv1_tmp2,
            NB_XCORR_FEATURES,
            NB_XCORR_FEATURES, // hstride: 224 (no extra padding needed)
            ACTIVATION_TANH,
        );

        // ── Step 4: Feature concatenation is implicit ──
        // downsampler_in = [conv2d_2 output (224) | IF upsampled (64)] = 288 floats

        // ── Step 5: Downsampling Dense Layer: 288 → 64 ──
        let mut downsampler_out = [0.0f32; DENSE_DOWNSAMPLER_OUT_SIZE];
        compute_generic_dense(
            &self.model.dense_downsampler,
            &mut downsampler_out,
            &downsampler_in,
            ACTIVATION_TANH,
        );

        // ── Step 6: GRU Update ──
        compute_generic_gru(
            &self.model.gru_1_input,
            &self.model.gru_1_recurrent,
            &mut self.gru_state,
            &downsampler_out,
        );

        // ── Step 7: Final Upsampling + Soft Argmax ──
        let mut output = [0.0f32; DENSE_FINAL_UPSAMPLER_OUT_SIZE];
        compute_generic_dense(
            &self.model.dense_final_upsampler,
            &mut output,
            &self.gru_state,
            ACTIVATION_LINEAR, // No nonlinearity on output logits
        );

        // Hard argmax over bins 0..180 to find peak bin
        let mut pos: usize = 0;
        let mut maxval: f32 = -1.0;
        for i in 0..NUM_PITCH_BINS {
            if output[i] > maxval {
                pos = i;
                maxval = output[i];
            }
        }

        // Soft argmax: exp-weighted average over [pos-2, pos+2], clamped to [0, 179].
        // Uses std exp() (double precision) matching C <math.h> exp().
        let mut sum: f32 = 0.0;
        let mut count: f32 = 0.0;
        let lo = pos.saturating_sub(SOFT_ARGMAX_RADIUS);
        let hi = (pos + SOFT_ARGMAX_RADIUS).min(NUM_PITCH_BINS - 1);
        for i in lo..=hi {
            // C: float p = exp(output[i]) — promotes float→double, exp in double, truncate to float
            let p = (output[i] as f64).exp() as f32;
            sum += p * i as f32;
            count += p;
        }

        // Convert weighted bin index to log-frequency pitch.
        // C: return (1.f/60.f)*(sum/count) - 1.5;
        // The C literal 1.5 is double, causing float→double promotion for the subtraction.
        let ratio = (1.0f32 / 60.0f32) * (sum / count);
        (ratio as f64 - 1.5) as f32
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(NB_XCORR_FEATURES, 224);
        assert_eq!(PITCH_IF_FEATURES, 88);
        assert_eq!(DOWNSAMPLER_IN_SIZE, 288);
        assert_eq!(DENSE_FINAL_UPSAMPLER_OUT_SIZE, 192);
        assert_eq!(XCORR_MEM1_SIZE, 452);
        assert_eq!(XCORR_MEM2_SIZE, 1808);
        assert_eq!(CONV_TMP_SIZE, 1808);
    }

    #[test]
    fn test_new_empty_state() {
        let state = PitchDnnState::new_empty();
        assert_eq!(state.gru_state.len(), GRU_1_STATE_SIZE);
        assert_eq!(state.xcorr_mem1.len(), XCORR_MEM1_SIZE);
        assert_eq!(state.xcorr_mem2.len(), XCORR_MEM2_SIZE);
        assert!(state.gru_state.iter().all(|&x| x == 0.0));
        assert!(state.xcorr_mem1.iter().all(|&x| x == 0.0));
        assert!(state.xcorr_mem2.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_soft_argmax_center() {
        // Strong peak at bin 90: weighted average should converge to ~90.
        let mut output = [0.0f32; DENSE_FINAL_UPSAMPLER_OUT_SIZE];
        output[90] = 10.0;

        let mut pos: usize = 0;
        let mut maxval: f32 = -1.0;
        for i in 0..NUM_PITCH_BINS {
            if output[i] > maxval {
                pos = i;
                maxval = output[i];
            }
        }
        assert_eq!(pos, 90);

        let mut sum: f32 = 0.0;
        let mut count: f32 = 0.0;
        let lo = pos.saturating_sub(SOFT_ARGMAX_RADIUS);
        let hi = (pos + SOFT_ARGMAX_RADIUS).min(NUM_PITCH_BINS - 1);
        for i in lo..=hi {
            let p = (output[i] as f64).exp() as f32;
            sum += p * i as f32;
            count += p;
        }
        let weighted = sum / count;
        assert!(
            (weighted - 90.0).abs() < 0.01,
            "Expected ~90.0, got {weighted}"
        );
    }

    #[test]
    fn test_soft_argmax_symmetric_shift() {
        // Symmetric pair at bins 89 and 91 (equal values), rest 0.
        // Hard argmax picks bin 89 (first max encountered).
        // Window [87, 91]: bins 87,88,90 have exp(0)=1, bins 89,91 have exp(5)≈148.4.
        // The window is centered on 89 not 90, so the weighted average is ~89.98.
        let mut output = [0.0f32; DENSE_FINAL_UPSAMPLER_OUT_SIZE];
        output[89] = 5.0;
        output[91] = 5.0;

        let mut pos: usize = 0;
        let mut maxval: f32 = -1.0;
        for i in 0..NUM_PITCH_BINS {
            if output[i] > maxval {
                pos = i;
                maxval = output[i];
            }
        }
        assert_eq!(pos, 89);

        let mut sum: f32 = 0.0;
        let mut count: f32 = 0.0;
        let lo = pos.saturating_sub(SOFT_ARGMAX_RADIUS);
        let hi = (pos + SOFT_ARGMAX_RADIUS).min(NUM_PITCH_BINS - 1);
        for i in lo..=hi {
            let p = (output[i] as f64).exp() as f32;
            sum += p * i as f32;
            count += p;
        }
        let weighted = sum / count;
        // Weighted average biased toward 90 but window centered on 89
        assert!(
            (weighted - 90.0).abs() < 0.05,
            "Expected ~90.0, got {weighted}"
        );
    }

    #[test]
    fn test_soft_argmax_boundary_low() {
        // Peak at bin 0: window should clamp to [0, 2].
        let mut output = [-10.0f32; DENSE_FINAL_UPSAMPLER_OUT_SIZE];
        output[0] = 5.0;

        let mut pos: usize = 0;
        let mut maxval: f32 = -1.0;
        for i in 0..NUM_PITCH_BINS {
            if output[i] > maxval {
                pos = i;
                maxval = output[i];
            }
        }
        assert_eq!(pos, 0);

        let lo = pos.saturating_sub(SOFT_ARGMAX_RADIUS);
        let hi = (pos + SOFT_ARGMAX_RADIUS).min(NUM_PITCH_BINS - 1);
        assert_eq!(lo, 0);
        assert_eq!(hi, 2);
    }

    #[test]
    fn test_soft_argmax_boundary_high() {
        // Peak at bin 179: window should clamp to [177, 179].
        let mut output = [-10.0f32; DENSE_FINAL_UPSAMPLER_OUT_SIZE];
        output[179] = 5.0;

        let mut pos: usize = 0;
        let mut maxval: f32 = -1.0;
        for i in 0..NUM_PITCH_BINS {
            if output[i] > maxval {
                pos = i;
                maxval = output[i];
            }
        }
        assert_eq!(pos, 179);

        let lo = pos.saturating_sub(SOFT_ARGMAX_RADIUS);
        let hi = (pos + SOFT_ARGMAX_RADIUS).min(NUM_PITCH_BINS - 1);
        assert_eq!(lo, 177);
        assert_eq!(hi, 179);
    }

    #[test]
    fn test_soft_argmax_boundary_one() {
        // Peak at bin 1: window should be [0, 3].
        let lo = 1_usize.saturating_sub(SOFT_ARGMAX_RADIUS);
        let hi = (1 + SOFT_ARGMAX_RADIUS).min(NUM_PITCH_BINS - 1);
        assert_eq!(lo, 0);
        assert_eq!(hi, 3);
    }

    #[test]
    fn test_return_value_range() {
        // Bin 0 → (1/60)*0 - 1.5 = -1.5
        let low = (1.0f32 / 60.0f32) * 0.0;
        let low = (low as f64 - 1.5) as f32;
        assert!((low - (-1.5f32)).abs() < 1e-6);

        // Bin 179 → (1/60)*179 - 1.5 ≈ 1.4833
        let high = (1.0f32 / 60.0f32) * 179.0;
        let high = (high as f64 - 1.5) as f32;
        assert!((high - 1.4833).abs() < 0.01, "got {high}");
    }

    #[test]
    fn test_exp_double_precision() {
        // Verify exp uses double precision matching C: float p = exp(output[i])
        let x: f32 = 2.5;
        let p = (x as f64).exp() as f32;
        // exp(2.5) ≈ 12.18249396
        assert!((p - 12.18249).abs() < 0.001, "got {p}");
    }

    #[test]
    fn test_return_matches_c_formula() {
        // Reproduce the full soft argmax with a known output vector.
        // Place a peak at bin 100 with neighbors slightly lower.
        let mut output = [-100.0f32; DENSE_FINAL_UPSAMPLER_OUT_SIZE];
        output[98] = 1.0;
        output[99] = 3.0;
        output[100] = 5.0;
        output[101] = 3.0;
        output[102] = 1.0;

        // Hard argmax
        let mut pos: usize = 0;
        let mut maxval: f32 = -1.0;
        for i in 0..NUM_PITCH_BINS {
            if output[i] > maxval {
                pos = i;
                maxval = output[i];
            }
        }
        assert_eq!(pos, 100);

        // Soft argmax
        let mut sum: f32 = 0.0;
        let mut count: f32 = 0.0;
        let lo = pos.saturating_sub(SOFT_ARGMAX_RADIUS);
        let hi = (pos + SOFT_ARGMAX_RADIUS).min(NUM_PITCH_BINS - 1);
        for i in lo..=hi {
            let p = (output[i] as f64).exp() as f32;
            sum += p * i as f32;
            count += p;
        }

        let ratio = (1.0f32 / 60.0f32) * (sum / count);
        let result = (ratio as f64 - 1.5) as f32;

        // Weighted average should be very close to 100 (symmetric neighbors)
        let weighted = sum / count;
        assert!((weighted - 100.0).abs() < 0.01, "weighted avg: {weighted}");

        // Result should be (1/60)*100 - 1.5 ≈ 0.1667
        let expected = (1.0f32 / 60.0) * 100.0;
        let expected = (expected as f64 - 1.5) as f32;
        assert!(
            (result - expected).abs() < 0.001,
            "result={result}, expected={expected}"
        );
    }

    #[test]
    fn test_new_empty_cannot_compute() {
        // new_empty() creates a state with default (zero-dimensioned) conv layers.
        // Calling compute() would panic due to ktime=0 underflow in compute_conv2d.
        // This is expected: the C code also asserts weights are loaded before inference.
        // Verify the state is correctly initialized for later load_model().
        let state = PitchDnnState::new_empty();
        assert_eq!(state.model.conv2d_1.ktime, 0); // not yet initialized
        assert_eq!(state.model.dense_if_upsampler_1.nb_inputs, 0);
    }

    #[test]
    fn test_soft_argmax_all_equal_logits() {
        // When all output logits within the window are equal, the soft argmax
        // reduces to a simple average of the bin indices.
        let output = [0.0f32; DENSE_FINAL_UPSAMPLER_OUT_SIZE];

        // Hard argmax: first bin (0) since all are equal
        let mut pos: usize = 0;
        let mut maxval: f32 = -1.0;
        for i in 0..NUM_PITCH_BINS {
            if output[i] > maxval {
                pos = i;
                maxval = output[i];
            }
        }
        assert_eq!(pos, 0);

        // Window [0, 2]: all exp(0) = 1.0
        // sum = 1*0 + 1*1 + 1*2 = 3, count = 3, weighted = 1.0
        let mut sum: f32 = 0.0;
        let mut count: f32 = 0.0;
        let lo = pos.saturating_sub(SOFT_ARGMAX_RADIUS);
        let hi = (pos + SOFT_ARGMAX_RADIUS).min(NUM_PITCH_BINS - 1);
        for i in lo..=hi {
            let p = (output[i] as f64).exp() as f32;
            sum += p * i as f32;
            count += p;
        }
        let weighted = sum / count;
        assert!(
            (weighted - 1.0).abs() < 1e-5,
            "Expected 1.0, got {weighted}"
        );

        // Final result: (1/60)*1.0 - 1.5 ≈ -1.4833
        let ratio = (1.0f32 / 60.0f32) * weighted;
        let result = (ratio as f64 - 1.5) as f32;
        let expected = (1.0f64 / 60.0 - 1.5) as f32;
        assert!(
            (result - expected).abs() < 1e-5,
            "result={result}, expected={expected}"
        );
    }
}
