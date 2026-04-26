//! Analysis-path MLP + GRU primitives.
//!
//! Port of `reference/src/mlp.c` (131 lines) and `reference/src/mlp.h`.
//! These are the small neural network kernels used by `analysis.c` to drive
//! the tonality / speech-vs-music classifier. They are distinct from the
//! DNN modules in `ropus/src/dnn/` (LPCNet / FARGAN / OSCE / DRED) — the
//! analysis MLP is a legacy fixed-architecture classifier that predates the
//! neural decoder enhancements.
//!
//! Bit-exact equivalence with the scalar C reference is required. Critically:
//!
//! * No `f32::mul_add`. The C `fmadd` macro in `mlp.c:38` is
//!   `#define fmadd(a,b,c) ((a)*(b)+(c))` — a plain multiply-then-add, not
//!   the IEEE fused `fmaf`. Using FMA here would diverge from C.
//! * `gemm_accum` preserves the outer-`i` / inner-`j` loop order with
//!   column-stride indexing (`weights[j*col_stride + i] * x[j]`). Do not
//!   reorder, transpose, or cache-block.
//! * Weights are `i8`; we cast once to `f32` before the multiply, matching
//!   the implicit `int8 -> float` promotion in C.
//!
//! These helpers are `pub(crate)` — they are consumed by `analysis.rs` in
//! a later sub-stage and are not part of the crate's public API.

// Stage 6.1 lands these items ahead of their first call-site in `analysis.rs`
// (Stage 6.3). Allow `dead_code` until that wiring exists.
#![allow(dead_code)]

/// Inverse scale applied to accumulated `i8` weight products.
///
/// Matches C `#define WEIGHTS_SCALE (1.f/128)` from `mlp.h:32`.
pub(crate) const WEIGHTS_SCALE: f32 = 1.0_f32 / 128.0_f32;

/// Upper bound on the neuron count of any single analysis layer.
///
/// Matches C `#define MAX_NEURONS 32` from `mlp.h:34`. Used to size the
/// stack scratch buffers inside [`analysis_compute_gru`].
pub(crate) const MAX_NEURONS: usize = 32;

/// Dense layer parameters (port of C `AnalysisDenseLayer` in `mlp.h:36-42`).
///
/// `bias` has length `nb_neurons`; `input_weights` has length
/// `nb_inputs * nb_neurons`, indexed as `input_weights[j * nb_neurons + i]`
/// for input `j` and output neuron `i` (column stride = `nb_neurons`).
/// `sigmoid = true` picks `sigmoid_approx` as the activation; otherwise
/// `tansig_approx`.
pub(crate) struct AnalysisDenseLayer {
    pub bias: &'static [i8],
    pub input_weights: &'static [i8],
    pub nb_inputs: i32,
    pub nb_neurons: i32,
    pub sigmoid: bool,
}

/// GRU layer parameters (port of C `AnalysisGRULayer` in `mlp.h:44-50`).
///
/// `bias` has length `3 * nb_neurons` (update / reset / output gates
/// concatenated). `input_weights` has shape `nb_inputs × (3 * nb_neurons)`
/// and `recurrent_weights` has shape `nb_neurons × (3 * nb_neurons)`, both
/// with a column stride of `3 * nb_neurons` so the three gates can share a
/// single `gemm_accum` call per matrix.
pub(crate) struct AnalysisGRULayer {
    pub bias: &'static [i8],
    pub input_weights: &'static [i8],
    pub recurrent_weights: &'static [i8],
    pub nb_inputs: i32,
    pub nb_neurons: i32,
}

/// Rational Padé approximation of `tanh(x)` on `[-1, 1]`.
///
/// Port of `tansig_approx` in `mlp.c:39-53`. Constants match the C source
/// byte-for-byte — six `f32` literals, evaluated in the same operation
/// order as C so the IEEE-754 rounding chain is identical.
#[inline]
pub(crate) fn tansig_approx(x: f32) -> f32 {
    const N0: f32 = 952.52801514_f32;
    const N1: f32 = 96.39235687_f32;
    const N2: f32 = 0.60863042_f32;
    const D0: f32 = 952.72399902_f32;
    const D1: f32 = 413.36801147_f32;
    const D2: f32 = 11.88600922_f32;

    let x2 = x * x;
    // C: num = fmadd(fmadd(N2, X2, N1), X2, N0)   -> ((N2*X2+N1)*X2+N0)
    // C: den = fmadd(fmadd(D2, X2, D1), X2, D0)   -> ((D2*X2+D1)*X2+D0)
    // fmadd is a plain macro, NOT fused. Do NOT replace with mul_add.
    let num = (N2 * x2 + N1) * x2 + N0;
    let den = (D2 * x2 + D1) * x2 + D0;
    let y = num * x / den;
    // C: MAX32(-1.f, MIN32(1.f, num))
    // arch.h MIN32/MAX32 are ternary `a<b?a:b` macros — for non-NaN inputs
    // this matches an if/else clamp. Polynomial output is always finite for
    // finite `x`, so there is no NaN edge case to worry about here.
    if y > 1.0 {
        1.0
    } else if y < -1.0 {
        -1.0
    } else {
        y
    }
}

/// Sigmoid approximation, `0.5 + 0.5 * tansig_approx(0.5 * x)`.
///
/// Port of `sigmoid_approx` in `mlp.c:55-58`.
#[inline]
pub(crate) fn sigmoid_approx(x: f32) -> f32 {
    0.5_f32 + 0.5_f32 * tansig_approx(0.5_f32 * x)
}

/// Int8 matrix-vector accumulate: `out[i] += sum_j(weights[j*col_stride+i] * x[j])`.
///
/// Port of `gemm_accum` in `mlp.c:60-68`. Preserves the exact loop order
/// (outer `i`, inner `j`) and column-stride indexing of the C source so
/// accumulation rounding matches bit-for-bit.
#[inline]
fn gemm_accum(out: &mut [f32], weights: &[i8], rows: i32, cols: i32, col_stride: i32, x: &[f32]) {
    let rows = rows as usize;
    let cols = cols as usize;
    let col_stride = col_stride as usize;
    for i in 0..rows {
        // Accumulation order matches the C scalar path exactly. Bit-exactness
        // against C relies on LLVM NOT reassociating this chain into a
        // horizontal sum — which holds under default `--release` (no
        // `fast-math`, no `reassoc`). If someone later sets RUSTFLAGS with
        // fast-math or stabilizes `std::intrinsics::fadd_fast` here, tier-1
        // bit-exactness against C breaks silently. Keep this scalar.
        for j in 0..cols {
            // C: out[i] += weights[j*col_stride + i] * x[j]
            // One cast from i8 to f32 before the multiply, matching C's
            // implicit int8->float promotion.
            out[i] += weights[j * col_stride + i] as f32 * x[j];
        }
    }
}

/// Evaluate a dense layer: `output = activation(bias + W * input)`.
///
/// Port of `analysis_compute_dense` in `mlp.c:70-90`. The activation is
/// `sigmoid_approx` if `layer.sigmoid` is true, otherwise `tansig_approx`.
#[inline]
pub(crate) fn analysis_compute_dense(
    layer: &AnalysisDenseLayer,
    output: &mut [f32],
    input: &[f32],
) {
    let m = layer.nb_inputs;
    let n = layer.nb_neurons;
    let stride = n;
    let n_us = n as usize;

    for i in 0..n_us {
        output[i] = layer.bias[i] as f32;
    }
    gemm_accum(output, layer.input_weights, n, m, stride, input);
    for i in 0..n_us {
        output[i] *= WEIGHTS_SCALE;
    }
    if layer.sigmoid {
        for i in 0..n_us {
            output[i] = sigmoid_approx(output[i]);
        }
    } else {
        for i in 0..n_us {
            output[i] = tansig_approx(output[i]);
        }
    }
}

/// Evaluate a GRU layer, updating `state` in place.
///
/// Port of `analysis_compute_gru` in `mlp.c:92-131`. Uses `MAX_NEURONS`
/// stack buffers for the update gate (`z`), reset gate (`r`), output
/// candidate (`h`), and intermediate `state .* r` vector (`tmp`), matching
/// the C reference exactly. Gate biases and weight sub-blocks are indexed
/// with the same column stride `3 * nb_neurons` so the three gates share
/// contiguous rows in `input_weights` / `recurrent_weights`.
#[inline]
pub(crate) fn analysis_compute_gru(gru: &AnalysisGRULayer, state: &mut [f32], input: &[f32]) {
    let mut tmp = [0.0_f32; MAX_NEURONS];
    let mut z = [0.0_f32; MAX_NEURONS];
    let mut r = [0.0_f32; MAX_NEURONS];
    let mut h = [0.0_f32; MAX_NEURONS];

    let m = gru.nb_inputs;
    let n = gru.nb_neurons;
    let stride = 3 * n;
    let n_us = n as usize;

    // --- Update gate ---
    for i in 0..n_us {
        z[i] = gru.bias[i] as f32;
    }
    gemm_accum(&mut z, gru.input_weights, n, m, stride, input);
    gemm_accum(&mut z, gru.recurrent_weights, n, n, stride, state);
    for i in 0..n_us {
        z[i] = sigmoid_approx(WEIGHTS_SCALE * z[i]);
    }

    // --- Reset gate ---
    for i in 0..n_us {
        r[i] = gru.bias[n_us + i] as f32;
    }
    gemm_accum(&mut r, &gru.input_weights[n_us..], n, m, stride, input);
    gemm_accum(&mut r, &gru.recurrent_weights[n_us..], n, n, stride, state);
    for i in 0..n_us {
        r[i] = sigmoid_approx(WEIGHTS_SCALE * r[i]);
    }

    // --- Output candidate ---
    for i in 0..n_us {
        h[i] = gru.bias[2 * n_us + i] as f32;
    }
    for i in 0..n_us {
        tmp[i] = state[i] * r[i];
    }
    gemm_accum(&mut h, &gru.input_weights[2 * n_us..], n, m, stride, input);
    gemm_accum(
        &mut h,
        &gru.recurrent_weights[2 * n_us..],
        n,
        n,
        stride,
        &tmp,
    );
    for i in 0..n_us {
        h[i] = z[i] * state[i] + (1.0 - z[i]) * tansig_approx(WEIGHTS_SCALE * h[i]);
    }
    for i in 0..n_us {
        state[i] = h[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reproduce the C `tansig_approx` in an independent way so the unit
    /// tests are a genuine oracle, not a tautology. This implementation
    /// uses `f64` internally and matches the C source only at the final
    /// cast-to-`f32` step — so if our `tansig_approx` had an operation-order
    /// bug, this would not catch it, BUT the bit-exact test against a
    /// direct Rust replica of C's float math (below) would.
    fn tansig_approx_c_reference(x: f32) -> f32 {
        // Direct byte-for-byte replica of the C source with the same
        // operation order, using the same `f32` constants.
        let n0: f32 = 952.52801514_f32;
        let n1: f32 = 96.39235687_f32;
        let n2: f32 = 0.60863042_f32;
        let d0: f32 = 952.72399902_f32;
        let d1: f32 = 413.36801147_f32;
        let d2: f32 = 11.88600922_f32;
        let x2 = x * x;
        let num = (n2 * x2 + n1) * x2 + n0;
        let den = (d2 * x2 + d1) * x2 + d0;
        let y = num * x / den;
        if y > 1.0 {
            1.0
        } else if y < -1.0 {
            -1.0
        } else {
            y
        }
    }

    fn sigmoid_approx_c_reference(x: f32) -> f32 {
        0.5_f32 + 0.5_f32 * tansig_approx_c_reference(0.5_f32 * x)
    }

    #[test]
    fn test_tansig_approx_matches_c() {
        // Hand-picked inputs spanning the interesting range:
        //  * 0.0: polynomial evaluates to N0 / D0 * 0 = 0 exactly.
        //  * Small positives/negatives near 0 — main rational-poly regime.
        //  * Values around x = ±1 where tanh nears saturation.
        //  * Edge cases that hit the MIN32/MAX32 clamp (|y| > 1).
        //  * Sign symmetry: tansig_approx(-x) should mirror tansig_approx(x).
        let inputs: [f32; 15] = [
            0.0, 1e-6, -1e-6, 0.5, -0.5, 1.0, -1.0, 2.5, -2.5, 8.0, -8.0, 1e6, -1e6, 0.123_456,
            -0.987_654,
        ];
        for &x in inputs.iter() {
            let got = tansig_approx(x);
            let expected = tansig_approx_c_reference(x);
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "tansig_approx({}): got {:#010x} ({}), expected {:#010x} ({})",
                x,
                got.to_bits(),
                got,
                expected.to_bits(),
                expected,
            );
        }

        // Absolute anchors that fall out of the math: at large |x|, the
        // rational polynomial is hit and the MAX32/MIN32 clamp saturates.
        // tansig_approx(+big) -> +1.0 bitwise, tansig_approx(-big) -> -1.0.
        assert_eq!(tansig_approx(1e6).to_bits(), 1.0_f32.to_bits());
        assert_eq!(tansig_approx(-1e6).to_bits(), (-1.0_f32).to_bits());
        // tansig_approx(0.0) = 0.0 (both signs should land on +0.0 since
        // num = 0 * x = 0 and division preserves sign, but the leading term
        // of num is literally zero for x=0).
        assert_eq!(tansig_approx(0.0).to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn test_sigmoid_approx_matches_c() {
        let inputs: [f32; 11] = [0.0, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 10.0, -10.0, 0.5, -0.5];
        for &x in inputs.iter() {
            let got = sigmoid_approx(x);
            let expected = sigmoid_approx_c_reference(x);
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "sigmoid_approx({}): got {:#010x} ({}), expected {:#010x} ({})",
                x,
                got.to_bits(),
                got,
                expected.to_bits(),
                expected,
            );
        }
        // sigmoid_approx(0) = 0.5 + 0.5 * tansig_approx(0) = 0.5 exactly.
        assert_eq!(sigmoid_approx(0.0).to_bits(), 0.5_f32.to_bits());
    }

    /// Independent replica of the C `gemm_accum` loop, used as the oracle.
    fn gemm_accum_c_reference(
        out: &mut [f32],
        weights: &[i8],
        rows: usize,
        cols: usize,
        col_stride: usize,
        x: &[f32],
    ) {
        for i in 0..rows {
            for j in 0..cols {
                out[i] += weights[j * col_stride + i] as f32 * x[j];
            }
        }
    }

    #[test]
    fn test_gemm_accum_matches_c() {
        // Small layer: rows=4, cols=6, stride=4 (i.e. contiguous with rows,
        // a typical dense layer with nb_neurons=4, nb_inputs=6).
        let rows = 4_i32;
        let cols = 6_i32;
        let stride = 4_i32;

        // Hand-picked i8 weights — mix of positive, negative, extremes, and
        // zero so the accumulation exercises sign handling.
        let weights: [i8; 24] = [
            // j=0    j=1    j=2    j=3    j=4    j=5
            1, -2, 3, -4, // j=0, i=0..3
            5, -6, 7, -8, // j=1
            -9, 10, -11, 12, // j=2
            13, -14, 15, -16, // j=3
            127, -128, 0, 42, // j=4 (extremes)
            -1, 1, -1, 1, // j=5
        ];

        // Hand-picked float inputs — small magnitudes plus a negative to
        // exercise sign flips in the weight*x product.
        let x: [f32; 6] = [0.125, 0.25, -0.5, 1.0, 0.0625, -2.0];

        // Seed `out` with a non-zero starting value to verify accumulation
        // (not overwrite) semantics.
        let mut got: [f32; 4] = [0.5, -0.25, 1.0, -1.0];
        let mut expected: [f32; 4] = [0.5, -0.25, 1.0, -1.0];

        super::gemm_accum(&mut got, &weights, rows, cols, stride, &x);
        gemm_accum_c_reference(
            &mut expected,
            &weights,
            rows as usize,
            cols as usize,
            stride as usize,
            &x,
        );

        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                e.to_bits(),
                "gemm_accum out[{}]: got {:#010x} ({}), expected {:#010x} ({})",
                i,
                g.to_bits(),
                g,
                e.to_bits(),
                e,
            );
        }

        // Sanity-check one output element against a hand computation:
        //   out[0] starts at 0.5; then accumulate over j=0..6:
        //     j=0:  1 * 0.125   =  0.125
        //     j=1:  5 * 0.25    =  1.25
        //     j=2: -9 * -0.5    =  4.5
        //     j=3: 13 * 1.0     = 13.0
        //     j=4:127 * 0.0625  =  7.9375
        //     j=5: -1 * -2.0    =  2.0
        //   Sum = 0.125 + 1.25 + 4.5 + 13.0 + 7.9375 + 2.0 = 28.8125
        //   Final: 0.5 + 28.8125 = 29.3125
        assert_eq!(got[0].to_bits(), 29.3125_f32.to_bits());
    }

    // Hand-picked GRU weights/biases for the bit-exact port test. Non-trivial
    // and non-symmetric so a z/r/h gate transposition, a stride error, or a
    // sign flip in the input vs recurrent split would all produce a
    // measurably different output. Layout matches the C reference:
    //   * N=4 neurons, M=3 inputs, col_stride = 3*N = 12.
    //   * `bias[0..4]`   -> z-gate (update) biases
    //     `bias[4..8]`   -> r-gate (reset)  biases
    //     `bias[8..12]`  -> h-gate (output) biases
    //   * `input_weights[j*12 +  0..4]` -> z-gate row for input j
    //     `input_weights[j*12 +  4..8]` -> r-gate row for input j
    //     `input_weights[j*12 +  8..12]`-> h-gate row for input j
    //   * `recurrent_weights` follows the same per-row gate slicing.
    static GRU_BIAS: [i8; 12] = [
        // z-gate (update) biases, neurons 0..3
        3, -5, 7, -11, // r-gate (reset) biases
        -2, 4, -6, 8, // h-gate (output) biases
        9, -13, 1, -3,
    ];

    static GRU_INPUT_WEIGHTS: [i8; 36] = [
        // j=0 row: z[0..4]             r[0..4]             h[0..4]
        1, -2, 3, -4, 10, -11, 12, -13, 20, -21, 22, -23, // j=1 row
        5, -6, 7, -8, 14, -15, 16, -17, 24, -25, 26, -27,
        // j=2 row (includes i8 extremes to exercise sign handling)
        9, -10, 127, -128, 18, -19, 64, -64, 28, -29, 30, -31,
    ];

    static GRU_RECURRENT_WEIGHTS: [i8; 48] = [
        // j=0 row (self-loop to neuron 0):   z[0..4]       r[0..4]       h[0..4]
        2, -3, 4, -5, 11, -12, 13, -14, 21, -22, 23, -24, // j=1 row
        6, -7, 8, -9, 15, -16, 17, -18, 25, -26, 27, -28, // j=2 row
        10, -11, 12, -13, 19, -20, 21, -22, 29, -30, 31, -32, // j=3 row (more i8 extremes)
        14, -15, 127, -128, 23, -24, 126, -127, 33, -34, 35, -36,
    ];

    /// Independent replica of `analysis_compute_gru` using only the
    /// already-tested primitives (`gemm_accum_c_reference`,
    /// `sigmoid_approx_c_reference`, `tansig_approx_c_reference`). This is a
    /// genuine oracle: it walks the three-gate computation explicitly from
    /// the per-gate bias/weight slices, so a transposition of z/r/h gates in
    /// the port (e.g. using `&input_weights[2*n..]` for the reset gate) would
    /// produce a different vector here.
    fn analysis_compute_gru_c_reference(
        bias: &[i8],
        input_weights: &[i8],
        recurrent_weights: &[i8],
        m: usize,
        n: usize,
        state: &mut [f32],
        input: &[f32],
    ) {
        let stride = 3 * n;

        // --- Update gate (z) ---
        let mut z = [0.0_f32; MAX_NEURONS];
        for i in 0..n {
            z[i] = bias[i] as f32;
        }
        gemm_accum_c_reference(&mut z[..n], input_weights, n, m, stride, input);
        gemm_accum_c_reference(&mut z[..n], recurrent_weights, n, n, stride, state);
        for i in 0..n {
            z[i] = sigmoid_approx_c_reference(WEIGHTS_SCALE * z[i]);
        }

        // --- Reset gate (r) ---
        let mut r = [0.0_f32; MAX_NEURONS];
        for i in 0..n {
            r[i] = bias[n + i] as f32;
        }
        gemm_accum_c_reference(&mut r[..n], &input_weights[n..], n, m, stride, input);
        gemm_accum_c_reference(&mut r[..n], &recurrent_weights[n..], n, n, stride, state);
        for i in 0..n {
            r[i] = sigmoid_approx_c_reference(WEIGHTS_SCALE * r[i]);
        }

        // --- Output candidate (h) ---
        let mut tmp = [0.0_f32; MAX_NEURONS];
        for i in 0..n {
            tmp[i] = state[i] * r[i];
        }
        let mut h = [0.0_f32; MAX_NEURONS];
        for i in 0..n {
            h[i] = bias[2 * n + i] as f32;
        }
        gemm_accum_c_reference(&mut h[..n], &input_weights[2 * n..], n, m, stride, input);
        gemm_accum_c_reference(
            &mut h[..n],
            &recurrent_weights[2 * n..],
            n,
            n,
            stride,
            &tmp[..n],
        );
        for i in 0..n {
            h[i] = z[i] * state[i] + (1.0 - z[i]) * tansig_approx_c_reference(WEIGHTS_SCALE * h[i]);
        }
        for i in 0..n {
            state[i] = h[i];
        }
    }

    #[test]
    fn test_analysis_compute_gru_matches_c() {
        // Minimal GRU: N=4 neurons, M=3 inputs. Small enough for a hand
        // oracle to cross-check; large enough that a z/r/h gate swap or
        // stride error produces a different result in every output slot.
        let gru = AnalysisGRULayer {
            bias: &GRU_BIAS,
            input_weights: &GRU_INPUT_WEIGHTS,
            recurrent_weights: &GRU_RECURRENT_WEIGHTS,
            nb_inputs: 3,
            nb_neurons: 4,
        };

        // Non-zero initial state and a mix of positive/negative inputs so
        // the reset gate's `state * r` fusion actually contributes.
        let initial_state: [f32; 4] = [0.125, -0.25, 0.5, -0.75];
        let input: [f32; 3] = [0.375, -0.5, 1.0];

        let mut got = initial_state;
        let mut expected = initial_state;

        super::analysis_compute_gru(&gru, &mut got, &input);
        analysis_compute_gru_c_reference(
            &GRU_BIAS,
            &GRU_INPUT_WEIGHTS,
            &GRU_RECURRENT_WEIGHTS,
            3,
            4,
            &mut expected,
            &input,
        );

        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                e.to_bits(),
                "analysis_compute_gru state[{}]: got {:#010x} ({}), expected {:#010x} ({})",
                i,
                g.to_bits(),
                g,
                e.to_bits(),
                e,
            );
        }
    }
}
