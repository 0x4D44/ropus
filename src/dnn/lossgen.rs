//! Neural network–based packet loss pattern generator.
//!
//! Generates temporally-correlated loss patterns using a small recurrent
//! neural network, providing more realistic test scenarios than i.i.d. random
//! loss. The model architecture is:
//!
//! ```text
//! [last_loss, percent_loss] → Dense(2→8,tanh) → GRU(8→16) → GRU(16→32)
//!     → Dense(32→1,sigmoid) → Bernoulli sampling → loss ∈ {0,1}
//! ```
//!
//! This module sits outside the codec pipeline — it is a testing/simulation
//! utility for evaluating error concealment (PLC, DRED, OSCE).
//!
//! Matches C reference: `dnn/lossgen.c`

use super::core::{
    ACTIVATION_SIGMOID, ACTIVATION_TANH, LinearLayer, WeightArray, compute_generic_dense,
    compute_generic_gru, linear_init, parse_weights,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Dense input layer output size (matches PyTorch `nn.Linear(2, 8)`).
pub const LOSSGEN_DENSE_IN_OUT_SIZE: usize = 8;

/// GRU1 hidden state size (matches PyTorch `nn.GRU(8, 16)`).
pub const LOSSGEN_GRU1_STATE_SIZE: usize = 16;

/// GRU2 hidden state size (matches PyTorch `nn.GRU(16, 32)`).
pub const LOSSGEN_GRU2_STATE_SIZE: usize = 32;

/// Number of warm-up iterations on first call to flush zero-initialized GRU states.
const WARMUP_COUNT: usize = 1000;

/// Default seed for the deterministic PRNG.
const DEFAULT_RNG_SEED: u32 = 42;

// ---------------------------------------------------------------------------
// PRNG — deterministic replacement for C's global rand()
// ---------------------------------------------------------------------------

/// 32-bit xorshift PRNG for deterministic loss pattern generation.
///
/// Replaces C's `rand()` which uses global state and varies across platforms.
/// Stored in `LossGenState` so each generator instance is independent and
/// reproducible.
#[derive(Clone, Debug)]
struct Xorshift32 {
    state: u32,
}

impl Xorshift32 {
    fn new(seed: u32) -> Self {
        // Xorshift cannot recover from a zero state, so force nonzero.
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    /// Produce a uniform float in [0, 1).
    /// Uses the upper 24 bits for uniform distribution across f32 mantissa range.
    /// Mimics C's `(float)rand() / (float)RAND_MAX`.
    fn next_f32(&mut self) -> f32 {
        // f32 has a 23-bit mantissa, so 24 bits of integer give full coverage.
        (self.next_u32() >> 8) as f32 / 16_777_216.0 // 2^24
    }
}

// ---------------------------------------------------------------------------
// Model struct
// ---------------------------------------------------------------------------

/// Neural network weights for the loss pattern generator.
///
/// Contains six layers forming a sequential model:
/// `Dense(2→8) → GRU(8→16) → GRU(16→32) → Dense(32→1)`
///
/// Matches C `LossGen` struct from generated `lossgen_data.h`.
#[derive(Clone, Debug)]
pub struct LossGen {
    pub lossgen_dense_in: LinearLayer,
    pub lossgen_gru1_input: LinearLayer,
    pub lossgen_gru1_recurrent: LinearLayer,
    pub lossgen_gru2_input: LinearLayer,
    pub lossgen_gru2_recurrent: LinearLayer,
    pub lossgen_dense_out: LinearLayer,
}

// ---------------------------------------------------------------------------
// State struct
// ---------------------------------------------------------------------------

/// Complete state for the loss pattern generator.
///
/// Holds model weights, GRU hidden states, previous loss decision,
/// warm-up flag, and PRNG state.
///
/// Matches C `LossGenState` from `lossgen.h`.
#[derive(Clone, Debug)]
pub struct LossGenState {
    model: LossGen,
    gru1_state: [f32; LOSSGEN_GRU1_STATE_SIZE],
    gru2_state: [f32; LOSSGEN_GRU2_STATE_SIZE],
    last_loss: i32,
    used: bool,
    rng: Xorshift32,
}

// ---------------------------------------------------------------------------
// Model initialization
// ---------------------------------------------------------------------------

/// Initialize a `LossGen` model from named weight arrays.
///
/// Equivalent to the C generated `init_lossgen()` function from `lossgen_data.c`.
/// Maps weight array names to `LinearLayer` fields using `linear_init`.
///
/// # Weight array names expected
///
/// Dense layers (float weights):
/// - `lossgen_dense_in_bias` [8], `lossgen_dense_in_weights` [16 floats]
/// - `lossgen_dense_out_bias` [1], `lossgen_dense_out_weights` [32 floats]
///
/// GRU layers (int8 quantized weights):
/// - `lossgen_gru{1,2}_input_bias`, `_weights`, `_weights_scale`
/// - `lossgen_gru{1,2}_recurrent_bias`, `_weights`, `_weights_scale`, `_diag`
pub fn init_lossgen(arrays: &[WeightArray]) -> Result<LossGen, ()> {
    // Dense input: Linear(2 → 8), float weights, tanh activation
    let lossgen_dense_in = linear_init(
        arrays,
        Some("lossgen_dense_in_bias"),
        None, // no subias
        None, // no int8 weights
        Some("lossgen_dense_in_weights"),
        None, // no sparse index
        None, // no diag
        None, // no scale (float weights don't need scale)
        2,    // nb_inputs: [last_loss, percent_loss]
        LOSSGEN_DENSE_IN_OUT_SIZE,
    )?;

    // GRU1 input weights: Linear(8 → 3×16=48), int8 quantized
    let lossgen_gru1_input = linear_init(
        arrays,
        Some("lossgen_gru1_input_bias"),
        None,
        Some("lossgen_gru1_input_weights"),
        None,
        None,
        None, // input layer has no diag
        Some("lossgen_gru1_input_weights_scale"),
        LOSSGEN_DENSE_IN_OUT_SIZE,   // 8
        3 * LOSSGEN_GRU1_STATE_SIZE, // 48
    )?;

    // GRU1 recurrent weights: Linear(16 → 3×16=48), int8 quantized + diagonal
    let lossgen_gru1_recurrent = linear_init(
        arrays,
        Some("lossgen_gru1_recurrent_bias"),
        None,
        Some("lossgen_gru1_recurrent_weights"),
        None,
        None,
        Some("lossgen_gru1_recurrent_diag"),
        Some("lossgen_gru1_recurrent_weights_scale"),
        LOSSGEN_GRU1_STATE_SIZE,     // 16
        3 * LOSSGEN_GRU1_STATE_SIZE, // 48
    )?;

    // GRU2 input weights: Linear(16 → 3×32=96), int8 quantized
    let lossgen_gru2_input = linear_init(
        arrays,
        Some("lossgen_gru2_input_bias"),
        None,
        Some("lossgen_gru2_input_weights"),
        None,
        None,
        None,
        Some("lossgen_gru2_input_weights_scale"),
        LOSSGEN_GRU1_STATE_SIZE,     // 16 (GRU1 output feeds GRU2 input)
        3 * LOSSGEN_GRU2_STATE_SIZE, // 96
    )?;

    // GRU2 recurrent weights: Linear(32 → 3×32=96), int8 quantized + diagonal
    let lossgen_gru2_recurrent = linear_init(
        arrays,
        Some("lossgen_gru2_recurrent_bias"),
        None,
        Some("lossgen_gru2_recurrent_weights"),
        None,
        None,
        Some("lossgen_gru2_recurrent_diag"),
        Some("lossgen_gru2_recurrent_weights_scale"),
        LOSSGEN_GRU2_STATE_SIZE,     // 32
        3 * LOSSGEN_GRU2_STATE_SIZE, // 96
    )?;

    // Dense output: Linear(32 → 1), float weights, sigmoid activation
    let lossgen_dense_out = linear_init(
        arrays,
        Some("lossgen_dense_out_bias"),
        None,
        None,
        Some("lossgen_dense_out_weights"),
        None,
        None,
        None,
        LOSSGEN_GRU2_STATE_SIZE, // 32
        1,
    )?;

    Ok(LossGen {
        lossgen_dense_in,
        lossgen_gru1_input,
        lossgen_gru1_recurrent,
        lossgen_gru2_input,
        lossgen_gru2_recurrent,
        lossgen_dense_out,
    })
}

// ---------------------------------------------------------------------------
// LossGenState implementation
// ---------------------------------------------------------------------------

impl LossGenState {
    /// Create a new `LossGenState` from weight arrays with the default RNG seed.
    ///
    /// Equivalent to C's `lossgen_init` with explicit weight data (since Rust
    /// doesn't use compiled-in default weights).
    ///
    /// The state is zero-initialized: GRU states are all zeros, `last_loss = 0`,
    /// and warm-up has not been performed. The first call to `sample_loss` will
    /// trigger 1000 warm-up iterations.
    pub fn new(arrays: &[WeightArray]) -> Result<Self, ()> {
        Self::new_with_seed(arrays, DEFAULT_RNG_SEED)
    }

    /// Create a new `LossGenState` with a specific RNG seed for reproducibility.
    pub fn new_with_seed(arrays: &[WeightArray], seed: u32) -> Result<Self, ()> {
        let model = init_lossgen(arrays)?;
        Ok(Self {
            model,
            gru1_state: [0.0; LOSSGEN_GRU1_STATE_SIZE],
            gru2_state: [0.0; LOSSGEN_GRU2_STATE_SIZE],
            last_loss: 0,
            used: false,
            rng: Xorshift32::new(seed),
        })
    }

    /// Load model weights from a binary weight blob.
    ///
    /// Equivalent to C's `lossgen_load_model`. Does NOT reset GRU states,
    /// `last_loss`, `used` flag, or RNG state — only the model weights are
    /// replaced. Call `reset()` first if you want a clean state.
    pub fn load_model(&mut self, data: &[u8]) -> Result<(), i32> {
        let arrays = parse_weights(data)?;
        let model = init_lossgen(&arrays).map_err(|()| -1)?;
        self.model = model;
        Ok(())
    }

    /// Set the RNG seed for reproducible sequences.
    pub fn seed_rng(&mut self, seed: u32) {
        self.rng = Xorshift32::new(seed);
    }

    /// Reset GRU states, loss history, and warm-up flag to initial conditions.
    ///
    /// Does not change model weights or RNG state. The next `sample_loss` call
    /// will trigger warm-up again.
    pub fn reset(&mut self) {
        self.gru1_state = [0.0; LOSSGEN_GRU1_STATE_SIZE];
        self.gru2_state = [0.0; LOSSGEN_GRU2_STATE_SIZE];
        self.last_loss = 0;
        self.used = false;
    }

    /// Generate one packet loss decision.
    ///
    /// Returns `1` if the packet is lost, `0` if received.
    ///
    /// `percent_loss` is the target loss rate as a **fraction** (0.0–1.0),
    /// not a percentage. Despite the parameter name (preserved from the C API),
    /// this is a fraction — pass `0.05` for 5% loss, not `5.0`.
    ///
    /// On the first call after initialization or `reset()`, runs 1000 warm-up
    /// iterations to flush zero-initialized GRU states into a realistic regime.
    pub fn sample_loss(&mut self, percent_loss: f32) -> i32 {
        // Warm-up: flush GRU zero-initialization bias.
        // Matches C: `for (i=0;i<1000;i++) sample_loss_impl(st, percent_loss);`
        if !self.used {
            for _ in 0..WARMUP_COUNT {
                self.sample_loss_impl(percent_loss);
            }
            self.used = true;
        }
        self.sample_loss_impl(percent_loss)
    }

    /// Core inference step: run one forward pass through the neural network
    /// and make a stochastic loss decision.
    ///
    /// Matches C `sample_loss_impl` from `lossgen.c:122–140`.
    fn sample_loss_impl(&mut self, percent_loss: f32) -> i32 {
        // Step 1: Construct input vector [last_loss, percent_loss]
        // C: input[0] = st->last_loss; input[1] = percent_loss;
        let input = [self.last_loss as f32, percent_loss];
        let mut tmp = [0.0f32; LOSSGEN_DENSE_IN_OUT_SIZE];

        // Step 2: Dense input layer — tanh(W_in * input + b_in)
        compute_generic_dense(
            &self.model.lossgen_dense_in,
            &mut tmp,
            &input,
            ACTIVATION_TANH,
        );

        // Step 3: GRU1 — updates gru1_state in-place
        compute_generic_gru(
            &self.model.lossgen_gru1_input,
            &self.model.lossgen_gru1_recurrent,
            &mut self.gru1_state,
            &tmp,
        );

        // Step 4: GRU2 — uses GRU1 state as input, updates gru2_state in-place.
        // Copy gru1_state to a local to avoid aliasing (C asserts in != state,
        // and here gru1_state is input while gru2_state is the mutated state).
        let gru1_out = self.gru1_state;
        compute_generic_gru(
            &self.model.lossgen_gru2_input,
            &self.model.lossgen_gru2_recurrent,
            &mut self.gru2_state,
            &gru1_out,
        );

        // Step 5: Dense output layer — sigmoid(W_out * gru2_state + b_out)
        let mut out = [0.0f32; 1];
        compute_generic_dense(
            &self.model.lossgen_dense_out,
            &mut out,
            &self.gru2_state,
            ACTIVATION_SIGMOID,
        );

        // Step 6: Stochastic Bernoulli sampling
        // C: loss = (float)rand()/(float)RAND_MAX < out;
        let loss = if self.rng.next_f32() < out[0] { 1 } else { 0 };
        self.last_loss = loss;
        loss
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dnn::core::{WEIGHT_TYPE_FLOAT, WEIGHT_TYPE_INT8, sigmoid_approx};

    const EPS: f32 = 1e-5;

    // --- Test helpers ---

    /// Create a float weight array from f32 values.
    fn make_f32_weight(name: &str, values: &[f32]) -> WeightArray {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        WeightArray {
            name: name.to_string(),
            weight_type: WEIGHT_TYPE_FLOAT,
            size: data.len(),
            data,
        }
    }

    /// Create an int8 weight array from i8 values.
    fn make_i8_weight(name: &str, values: &[i8]) -> WeightArray {
        let data: Vec<u8> = values.iter().map(|&v| v as u8).collect();
        WeightArray {
            name: name.to_string(),
            weight_type: WEIGHT_TYPE_INT8,
            size: data.len(),
            data,
        }
    }

    /// Build a complete set of zero/neutral weight arrays for LossGen.
    /// All biases and weights are zero, scales are 1.0.
    /// With these weights, the network output is sigmoid(dense_out_bias).
    fn make_zero_weights() -> Vec<WeightArray> {
        make_zero_weights_with_out_bias(0.0)
    }

    /// Build zero weights with a specific dense_out bias.
    /// The network output will be approximately sigmoid(bias).
    fn make_zero_weights_with_out_bias(out_bias: f32) -> Vec<WeightArray> {
        vec![
            // Dense input: 2 → 8, float
            make_f32_weight("lossgen_dense_in_bias", &[0.0; LOSSGEN_DENSE_IN_OUT_SIZE]),
            make_f32_weight(
                "lossgen_dense_in_weights",
                &[0.0; 2 * LOSSGEN_DENSE_IN_OUT_SIZE],
            ),
            // GRU1 input: 8 → 48, int8
            make_f32_weight(
                "lossgen_gru1_input_bias",
                &[0.0; 3 * LOSSGEN_GRU1_STATE_SIZE],
            ),
            make_i8_weight(
                "lossgen_gru1_input_weights",
                &[0i8; LOSSGEN_DENSE_IN_OUT_SIZE * 3 * LOSSGEN_GRU1_STATE_SIZE],
            ),
            make_f32_weight(
                "lossgen_gru1_input_weights_scale",
                &[1.0; 3 * LOSSGEN_GRU1_STATE_SIZE],
            ),
            // GRU1 recurrent: 16 → 48, int8 + diag
            make_f32_weight(
                "lossgen_gru1_recurrent_bias",
                &[0.0; 3 * LOSSGEN_GRU1_STATE_SIZE],
            ),
            make_i8_weight(
                "lossgen_gru1_recurrent_weights",
                &[0i8; LOSSGEN_GRU1_STATE_SIZE * 3 * LOSSGEN_GRU1_STATE_SIZE],
            ),
            make_f32_weight(
                "lossgen_gru1_recurrent_weights_scale",
                &[1.0; 3 * LOSSGEN_GRU1_STATE_SIZE],
            ),
            make_f32_weight(
                "lossgen_gru1_recurrent_diag",
                &[0.0; 3 * LOSSGEN_GRU1_STATE_SIZE],
            ),
            // GRU2 input: 16 → 96, int8
            make_f32_weight(
                "lossgen_gru2_input_bias",
                &[0.0; 3 * LOSSGEN_GRU2_STATE_SIZE],
            ),
            make_i8_weight(
                "lossgen_gru2_input_weights",
                &[0i8; LOSSGEN_GRU1_STATE_SIZE * 3 * LOSSGEN_GRU2_STATE_SIZE],
            ),
            make_f32_weight(
                "lossgen_gru2_input_weights_scale",
                &[1.0; 3 * LOSSGEN_GRU2_STATE_SIZE],
            ),
            // GRU2 recurrent: 32 → 96, int8 + diag
            make_f32_weight(
                "lossgen_gru2_recurrent_bias",
                &[0.0; 3 * LOSSGEN_GRU2_STATE_SIZE],
            ),
            make_i8_weight(
                "lossgen_gru2_recurrent_weights",
                &[0i8; LOSSGEN_GRU2_STATE_SIZE * 3 * LOSSGEN_GRU2_STATE_SIZE],
            ),
            make_f32_weight(
                "lossgen_gru2_recurrent_weights_scale",
                &[1.0; 3 * LOSSGEN_GRU2_STATE_SIZE],
            ),
            make_f32_weight(
                "lossgen_gru2_recurrent_diag",
                &[0.0; 3 * LOSSGEN_GRU2_STATE_SIZE],
            ),
            // Dense output: 32 → 1, float
            make_f32_weight("lossgen_dense_out_bias", &[out_bias]),
            make_f32_weight("lossgen_dense_out_weights", &[0.0; LOSSGEN_GRU2_STATE_SIZE]),
        ]
    }

    // --- PRNG tests ---

    #[test]
    fn test_xorshift_deterministic() {
        let mut rng1 = Xorshift32::new(12345);
        let mut rng2 = Xorshift32::new(12345);
        for _ in 0..100 {
            assert_eq!(rng1.next_u32(), rng2.next_u32());
        }
    }

    #[test]
    fn test_xorshift_different_seeds() {
        let mut rng1 = Xorshift32::new(1);
        let mut rng2 = Xorshift32::new(2);
        // Different seeds should produce different sequences
        let seq1: Vec<u32> = (0..10).map(|_| rng1.next_u32()).collect();
        let seq2: Vec<u32> = (0..10).map(|_| rng2.next_u32()).collect();
        assert_ne!(seq1, seq2);
    }

    #[test]
    fn test_xorshift_zero_seed_forced_nonzero() {
        let mut rng = Xorshift32::new(0);
        // Should not get stuck at zero
        assert_ne!(rng.next_u32(), 0);
    }

    #[test]
    fn test_xorshift_f32_range() {
        let mut rng = Xorshift32::new(42);
        for _ in 0..10_000 {
            let v = rng.next_f32();
            assert!(v >= 0.0, "value {v} below 0.0");
            assert!(v < 1.0, "value {v} at or above 1.0");
        }
    }

    // --- Model initialization tests ---

    #[test]
    fn test_init_lossgen_success() {
        let arrays = make_zero_weights();
        let model = init_lossgen(&arrays);
        assert!(model.is_ok());
        let model = model.unwrap();
        assert_eq!(model.lossgen_dense_in.nb_inputs, 2);
        assert_eq!(model.lossgen_dense_in.nb_outputs, LOSSGEN_DENSE_IN_OUT_SIZE);
        assert_eq!(
            model.lossgen_gru1_input.nb_inputs,
            LOSSGEN_DENSE_IN_OUT_SIZE
        );
        assert_eq!(
            model.lossgen_gru1_input.nb_outputs,
            3 * LOSSGEN_GRU1_STATE_SIZE
        );
        assert_eq!(
            model.lossgen_gru1_recurrent.nb_inputs,
            LOSSGEN_GRU1_STATE_SIZE
        );
        assert_eq!(
            model.lossgen_gru1_recurrent.nb_outputs,
            3 * LOSSGEN_GRU1_STATE_SIZE
        );
        assert_eq!(model.lossgen_gru2_input.nb_inputs, LOSSGEN_GRU1_STATE_SIZE);
        assert_eq!(
            model.lossgen_gru2_input.nb_outputs,
            3 * LOSSGEN_GRU2_STATE_SIZE
        );
        assert_eq!(
            model.lossgen_gru2_recurrent.nb_inputs,
            LOSSGEN_GRU2_STATE_SIZE
        );
        assert_eq!(
            model.lossgen_gru2_recurrent.nb_outputs,
            3 * LOSSGEN_GRU2_STATE_SIZE
        );
        assert_eq!(model.lossgen_dense_out.nb_inputs, LOSSGEN_GRU2_STATE_SIZE);
        assert_eq!(model.lossgen_dense_out.nb_outputs, 1);
    }

    #[test]
    fn test_init_lossgen_missing_weights_fails() {
        // Empty weight array list should fail
        let arrays: Vec<WeightArray> = vec![];
        assert!(init_lossgen(&arrays).is_err());
    }

    #[test]
    fn test_init_lossgen_has_float_weights_for_dense() {
        let arrays = make_zero_weights();
        let model = init_lossgen(&arrays).unwrap();
        // Dense layers should have float_weights
        assert!(model.lossgen_dense_in.float_weights.is_some());
        assert!(model.lossgen_dense_out.float_weights.is_some());
        // Dense layers should NOT have int8 weights
        assert!(model.lossgen_dense_in.weights.is_none());
        assert!(model.lossgen_dense_out.weights.is_none());
    }

    #[test]
    fn test_init_lossgen_has_int8_weights_for_gru() {
        let arrays = make_zero_weights();
        let model = init_lossgen(&arrays).unwrap();
        // GRU layers should have int8 weights and scale
        assert!(model.lossgen_gru1_input.weights.is_some());
        assert!(model.lossgen_gru1_input.scale.is_some());
        assert!(model.lossgen_gru1_recurrent.weights.is_some());
        assert!(model.lossgen_gru1_recurrent.scale.is_some());
        assert!(model.lossgen_gru2_input.weights.is_some());
        assert!(model.lossgen_gru2_input.scale.is_some());
        assert!(model.lossgen_gru2_recurrent.weights.is_some());
        assert!(model.lossgen_gru2_recurrent.scale.is_some());
        // Recurrent layers should have diag
        assert!(model.lossgen_gru1_recurrent.diag.is_some());
        assert!(model.lossgen_gru2_recurrent.diag.is_some());
    }

    // --- State tests ---

    #[test]
    fn test_state_new() {
        let arrays = make_zero_weights();
        let st = LossGenState::new(&arrays);
        assert!(st.is_ok());
        let st = st.unwrap();
        assert_eq!(st.last_loss, 0);
        assert!(!st.used);
        assert_eq!(st.gru1_state, [0.0; LOSSGEN_GRU1_STATE_SIZE]);
        assert_eq!(st.gru2_state, [0.0; LOSSGEN_GRU2_STATE_SIZE]);
    }

    #[test]
    fn test_state_reset() {
        let arrays = make_zero_weights();
        let mut st = LossGenState::new(&arrays).unwrap();
        // Run some iterations to modify state
        st.sample_loss(0.5);
        assert!(st.used);
        // Reset
        st.reset();
        assert!(!st.used);
        assert_eq!(st.last_loss, 0);
        assert_eq!(st.gru1_state, [0.0; LOSSGEN_GRU1_STATE_SIZE]);
        assert_eq!(st.gru2_state, [0.0; LOSSGEN_GRU2_STATE_SIZE]);
    }

    // --- Inference tests ---

    #[test]
    fn test_sample_loss_returns_binary() {
        let arrays = make_zero_weights();
        let mut st = LossGenState::new(&arrays).unwrap();
        for _ in 0..100 {
            let loss = st.sample_loss(0.5);
            assert!(loss == 0 || loss == 1, "loss must be 0 or 1, got {loss}");
        }
    }

    #[test]
    fn test_warmup_triggers_on_first_call() {
        let arrays = make_zero_weights();
        let mut st = LossGenState::new_with_seed(&arrays, 42).unwrap();
        assert!(!st.used);
        st.sample_loss(0.5);
        assert!(st.used);
    }

    #[test]
    fn test_warmup_not_repeated() {
        let arrays = make_zero_weights();
        let mut st = LossGenState::new_with_seed(&arrays, 42).unwrap();
        st.sample_loss(0.5);
        assert!(st.used);

        // Save RNG state after first call
        let rng_state_after_first = st.rng.state;

        // Second call should only advance RNG by 1 (not 1001)
        st.sample_loss(0.5);
        let rng_state_after_second = st.rng.state;
        assert_ne!(rng_state_after_first, rng_state_after_second);
    }

    #[test]
    fn test_zero_weights_loss_near_half() {
        // With all-zero weights and biases, network output is sigmoid(0) = 0.5,
        // so about 50% of samples should be 1.
        let arrays = make_zero_weights();
        let mut st = LossGenState::new_with_seed(&arrays, 42).unwrap();
        let n = 10_000;
        let losses: i32 = (0..n).map(|_| st.sample_loss(0.5)).sum();
        let loss_rate = losses as f32 / n as f32;
        assert!(
            (0.4..0.6).contains(&loss_rate),
            "expected ~50% loss rate with zero weights, got {:.1}%",
            loss_rate * 100.0
        );
    }

    #[test]
    fn test_high_bias_loss_near_one() {
        // With dense_out bias = 5.0, sigmoid(5.0) ≈ 0.993, so almost all
        // samples should be 1 (lost).
        let arrays = make_zero_weights_with_out_bias(5.0);
        let mut st = LossGenState::new_with_seed(&arrays, 42).unwrap();
        let n = 1_000;
        let losses: i32 = (0..n).map(|_| st.sample_loss(0.5)).sum();
        let loss_rate = losses as f32 / n as f32;
        let expected = sigmoid_approx(5.0);
        assert!(
            loss_rate > 0.95,
            "expected >95% loss rate (sigmoid(5.0)={expected:.3}), got {:.1}%",
            loss_rate * 100.0
        );
    }

    #[test]
    fn test_low_bias_loss_near_zero() {
        // With dense_out bias = -5.0, sigmoid(-5.0) ≈ 0.007, so almost no
        // samples should be 1 (lost).
        let arrays = make_zero_weights_with_out_bias(-5.0);
        let mut st = LossGenState::new_with_seed(&arrays, 42).unwrap();
        let n = 1_000;
        let losses: i32 = (0..n).map(|_| st.sample_loss(0.5)).sum();
        let loss_rate = losses as f32 / n as f32;
        let expected = sigmoid_approx(-5.0);
        assert!(
            loss_rate < 0.05,
            "expected <5% loss rate (sigmoid(-5.0)={expected:.3}), got {:.1}%",
            loss_rate * 100.0
        );
    }

    // --- Determinism tests ---

    #[test]
    fn test_deterministic_with_same_seed() {
        let arrays = make_zero_weights();
        let mut st1 = LossGenState::new_with_seed(&arrays, 12345).unwrap();
        let mut st2 = LossGenState::new_with_seed(&arrays, 12345).unwrap();
        let seq1: Vec<i32> = (0..100).map(|_| st1.sample_loss(0.3)).collect();
        let seq2: Vec<i32> = (0..100).map(|_| st2.sample_loss(0.3)).collect();
        assert_eq!(seq1, seq2, "same seed should produce identical sequences");
    }

    #[test]
    fn test_different_seeds_differ() {
        let arrays = make_zero_weights();
        let mut st1 = LossGenState::new_with_seed(&arrays, 1).unwrap();
        let mut st2 = LossGenState::new_with_seed(&arrays, 2).unwrap();
        let seq1: Vec<i32> = (0..100).map(|_| st1.sample_loss(0.3)).collect();
        let seq2: Vec<i32> = (0..100).map(|_| st2.sample_loss(0.3)).collect();
        assert_ne!(
            seq1, seq2,
            "different seeds should produce different sequences"
        );
    }

    #[test]
    fn test_seed_rng_resets_sequence() {
        let arrays = make_zero_weights();
        let mut st = LossGenState::new_with_seed(&arrays, 42).unwrap();
        // Generate some values
        let _: Vec<i32> = (0..50).map(|_| st.sample_loss(0.5)).collect();
        // Re-seed and reset state
        st.seed_rng(42);
        st.reset();
        let seq_after_reseed: Vec<i32> = (0..50).map(|_| st.sample_loss(0.5)).collect();

        // Fresh state with same seed
        let mut st_fresh = LossGenState::new_with_seed(&arrays, 42).unwrap();
        let seq_fresh: Vec<i32> = (0..50).map(|_| st_fresh.sample_loss(0.5)).collect();

        assert_eq!(seq_after_reseed, seq_fresh);
    }

    // --- GRU state evolution tests ---

    #[test]
    fn test_gru_states_zero_with_zero_weights() {
        // With all-zero weights and biases, GRU states should remain zero:
        // z = sigmoid(0) = 0.5, h = tanh(0) = 0
        // state = 0.5 * old_state + 0.5 * 0 = 0.5 * old_state
        // Starting from 0, stays at 0.
        let arrays = make_zero_weights();
        let mut st = LossGenState::new_with_seed(&arrays, 42).unwrap();
        // Run impl directly (no warm-up overhead) to check state
        st.sample_loss_impl(0.5);
        for &v in st.gru1_state.iter() {
            assert!(
                v.abs() < EPS,
                "gru1_state should stay zero with zero weights"
            );
        }
        for &v in st.gru2_state.iter() {
            assert!(
                v.abs() < EPS,
                "gru2_state should stay zero with zero weights"
            );
        }
    }

    #[test]
    fn test_last_loss_feedback() {
        // Verify that last_loss is fed back as input to the next step.
        let arrays = make_zero_weights();
        let mut st = LossGenState::new_with_seed(&arrays, 42).unwrap();
        st.used = true; // Skip warm-up for controlled testing
        let loss = st.sample_loss(0.5);
        assert_eq!(st.last_loss, loss);
    }

    // --- Constants validation ---

    #[test]
    fn test_constants() {
        assert_eq!(LOSSGEN_DENSE_IN_OUT_SIZE, 8);
        assert_eq!(LOSSGEN_GRU1_STATE_SIZE, 16);
        assert_eq!(LOSSGEN_GRU2_STATE_SIZE, 32);
        assert_eq!(WARMUP_COUNT, 1000);
    }
}
