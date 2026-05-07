#![cfg(not(no_reference))]
//! Phase 3.1 diagnostic: per-frame LPCNet feature drift between the ropus
//! Rust DRED encoder and the xiph C reference, on the existing 440 Hz
//! deterministic fixture. Active oracle. Replaces the bare comment at
//! dred_encode_payload_diff.rs:151 ("Stage 7 LPCNet drift") with a
//! measured, locked-bound regression guard.
//!
//! See HLD: wrk_docs/2026.05.06 - HLD - dred lpcnet feature drift diagnostic.md.

use std::fs;
use std::os::raw::c_void;
use std::path::PathBuf;

use ropus::dnn::dred::DREDEnc;
use ropus::dnn::embedded_weights::WEIGHTS_BLOB;
use ropus::dnn::lpcnet::NB_TOTAL_FEATURES;

use ropus_harness_deep_plc::{
    ropus_test_dred_compute_latents, ropus_test_dredenc_copy_lpcnet_features,
    ropus_test_dredenc_free, ropus_test_dredenc_latents_buffer_fill, ropus_test_dredenc_new,
};

const FIXTURE_NAME: &str = "48000hz_mono_sine440.wav";
const FIXTURE_SR: i32 = 48_000;
const FIXTURE_CH: i32 = 1;
const FIXTURE_FRAME_MS: i32 = 20;
const FIXTURE_EXTRA_DELAY: i32 = 312;

// Locked bounds. Observed-and-locked with no slack (HLD §"Measured-And-
// Locked Bounds"); reproducibility confirmed across consecutive runs on
// the lock host. Re-locking requires explicit Arthur signoff per the PLC
// drift policy.
//
// 2026-05-07 re-lock after kf_bfly5 associativity fix
// (ropus/src/dnn/lpcnet.rs:432-450 grouped to match
// reference/celt/kiss_fft.c:289-302) and lpc_from_cepstrum f64 pow
// (ropus/src/dnn/lpcnet.rs:821 matching reference/dnn/freq.c:313):
//   - cepstral features 0-17: bit-exact (was ~1e-6 noise floor).
//   - LPC features 22-35: bit-exact.
//   - Residual drift confined to features 18-21 (pitch/frame_corr/
//     LPC[0]/LPC[1]); max abs 3.0994415e-5, max RMS 5.4790185e-6.
//
// 2026-05-07 re-lock after lpc_from_bands f64 lag-windowing fix
// (ropus/src/dnn/lpcnet.rs:793-808 matching reference/dnn/freq.c:287-289)
// — the C `1 - 6e-5*i*i` expression is `double` because `6e-5` is a double
// literal; previous Rust f32 pipeline drifted ~1 ULP per coefficient. Also
// applied the noise-floor int-division fix `320/12=26` then `/38.0`
// (correctness bug, no measurable effect on this fixture because
// `ac[0]*1e-4` dominates the constant for loud signals):
//   - LPC[0]/LPC[1] (features 20, 21): now bit-exact (was 7.6293945e-6 each).
//   - pitch/frame_corr (features 18, 19): residual 1.001358e-5 / 8.34465e-7
//     (was 9.775162e-6 / 3.0994415e-5). Feature 19 dropped ~37x; feature 18
//     unchanged (~1e-5) — that path is pitch detection, not lpc_from_bands.
// Previous bounds (pre-fix): MAX_ABS_FEATURE_DRIFT=3.0994415e-5,
// MAX_RMS_PER_FRAME_DRIFT=5.4790185e-6.
//
// 2026-05-07 re-lock after if_features norm_1 f64 divide+sqrt fix
// (ropus/src/dnn/lpcnet.rs:1632 matching reference/dnn/lpcnet_enc.c:134) —
// C's `1.f / sqrt(sumsq)` promotes to double at the libm sqrt() call site
// while keeping the sumsq accumulation in f32 (both operands are float).
// Previous Rust f32 path drifted 1 ULP at the rsqrt and propagated into
// `prod.r*norm_1` / `prod.i*norm_1`, which feed the IF feature buffer used
// by pitch detection (feat 18 dnn_pitch):
//   - feat 18 (dnn_pitch): 9.894371e-6 (was 1.001358e-5; ~1.2% reduction).
//     Residual is the xcorr_features cascade, below the propagation
//     threshold to bit-exact but not dominated by the rsqrt anymore.
//   - feat 19 (frame_corr): 8.34465e-7 (unchanged from previous re-lock).
// Previous bounds (pre-fix): MAX_ABS_FEATURE_DRIFT=1.001358e-5,
// MAX_RMS_PER_FRAME_DRIFT=1.6713232e-6.
//
// 2026-05-07 re-lock after celt_fir_f32 summation-order fix
// (ropus/src/dnn/lpcnet.rs:1052-1066 matching
// reference/celt/celt_lpc.c:159-189). The C FIR accumulates
// `num[ord-1]*x[i]` first, then `num[ord-2]*x[i+1]`, ..., then
// `num[0]*x[i+ord-1]` last; float addition is non-associative so this
// summation order is part of the bit-exact contract. Previous Rust loop
// summed in the opposite direction.
//
// After the fix, 34 of 36 features are bit-exact; only features 18
// (dnn_pitch) and 19 (frame_corr) retain residual drift, both bounded at
// values much smaller than before:
//   - feats 0-17 (cepstral) and 20-35 (LPC + pitch period + remaining):
//     bit-exact (per_feature_max_abs_drift == 0.0). Bit-exact lock
//     asserted via `BIT_EXACT_FEATURE_INDICES` below.
//   - feat 18 (dnn_pitch): 1.013279e-5 (was 9.894371e-6 — slightly looser
//     because the FIR fix re-balances drift between feats 18 and 19).
//     Drift now lives entirely inside the PitchDNN neural inference;
//     PitchDNN sees bit-exact inputs but produces a marginally different
//     output due to its own f32 internal arithmetic noise — a separate
//     surface for a future HLD. Locked via `FEAT18_MAX_ABS_DRIFT`.
//   - feat 19 (frame_corr): 1.7881393e-7 (was 8.34465e-7; ~4.7x
//     reduction). Locked via `FEAT19_MAX_ABS_DRIFT`.
// Previous bounds (pre-fix): MAX_ABS_FEATURE_DRIFT=9.894371e-6,
// MAX_RMS_PER_FRAME_DRIFT=1.6514838e-6.
//
// Cross-reference: wrk_docs/2026.05.07 - HLD - xcorr-features-cascade-fix.md.
const MAX_ABS_FEATURE_DRIFT: f32 = 1.013279e-5;
const MAX_RMS_PER_FRAME_DRIFT: f32 = 1.6888275e-6;
const MAX_DRIFTING_FRAME_COUNT: usize = 50;
const FIRST_DIVERGENT_FRAME_AT_LEAST: usize = 0;

// Feature-18 (dnn_pitch) residual drift after the celt_fir_f32 fix. The
// drift is now contained inside the PitchDNN neural inference (bit-exact
// inputs, f32-noise output). Locking at the observed value pins the
// residual; future PitchDNN-side fixes tighten this bound.
const FEAT18_MAX_ABS_DRIFT: f32 = 1.013279e-5;

// Feature-19 (frame_corr) residual drift after the celt_fir_f32 fix.
// 4.7x smaller than the pre-fix bound; remaining drift is the float
// arithmetic noise inside the frame correlation computation.
const FEAT19_MAX_ABS_DRIFT: f32 = 1.7881393e-7;

// All features except 18 (dnn_pitch) and 19 (frame_corr) are bit-exact
// against the C reference. Any future change drifting any of these indices
// fails the test.
const BIT_EXACT_FEATURE_INDICES: &[usize] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35,
];

#[derive(Debug, Clone)]
struct LpcnetFeatureDriftReport {
    n_compared_frames: usize,
    first_divergent_frame: Option<usize>,
    max_abs_drift: f32,
    max_abs_drift_frame: Option<usize>,
    max_abs_drift_feature_idx: Option<usize>,
    max_rms_per_frame: f32,
    max_rms_per_frame_idx: Option<usize>,
    drifting_frame_count: usize,
    first_mismatches: Vec<(usize, usize, f32, f32, f32)>,
    per_feature_max_abs_drift: [f32; NB_TOTAL_FEATURES],
}

fn lpcnet_feature_drift_report(per_frame: &[(Vec<f32>, Vec<f32>)]) -> LpcnetFeatureDriftReport {
    let mut first_divergent_frame: Option<usize> = None;
    let mut max_abs_drift = 0.0f32;
    let mut max_abs_drift_frame: Option<usize> = None;
    let mut max_abs_drift_feature_idx: Option<usize> = None;
    let mut max_rms_per_frame = 0.0f32;
    let mut max_rms_per_frame_idx: Option<usize> = None;
    let mut drifting_frame_count = 0usize;
    let mut first_mismatches: Vec<(usize, usize, f32, f32, f32)> = Vec::new();
    let mut per_feature_max_abs_drift = [0.0f32; NB_TOTAL_FEATURES];

    for (frame, (c_feat, r_feat)) in per_frame.iter().enumerate() {
        assert_eq!(c_feat.len(), NB_TOTAL_FEATURES);
        assert_eq!(r_feat.len(), NB_TOTAL_FEATURES);

        let mut sum_sq = 0.0f64;
        let mut frame_has_drift = false;
        for j in 0..NB_TOTAL_FEATURES {
            let diff = c_feat[j] - r_feat[j];
            let abs_diff = diff.abs();
            sum_sq += (diff as f64) * (diff as f64);

            if abs_diff > per_feature_max_abs_drift[j] {
                per_feature_max_abs_drift[j] = abs_diff;
            }

            if abs_diff > 0.0 {
                frame_has_drift = true;
                if first_divergent_frame.is_none() {
                    first_divergent_frame = Some(frame);
                }
                if abs_diff > max_abs_drift {
                    max_abs_drift = abs_diff;
                    max_abs_drift_frame = Some(frame);
                    max_abs_drift_feature_idx = Some(j);
                }
                if first_mismatches.len() < 8 {
                    first_mismatches.push((frame, j, c_feat[j], r_feat[j], abs_diff));
                }
            }
        }

        if frame_has_drift {
            drifting_frame_count += 1;
        }

        let rms = (sum_sq / NB_TOTAL_FEATURES as f64).sqrt() as f32;
        if rms > max_rms_per_frame {
            max_rms_per_frame = rms;
            max_rms_per_frame_idx = Some(frame);
        }
    }

    LpcnetFeatureDriftReport {
        n_compared_frames: per_frame.len(),
        first_divergent_frame,
        max_abs_drift,
        max_abs_drift_frame,
        max_abs_drift_feature_idx,
        max_rms_per_frame,
        max_rms_per_frame_idx,
        drifting_frame_count,
        first_mismatches,
        per_feature_max_abs_drift,
    }
}

fn lpcnet_feature_drift_failure_context(report: &LpcnetFeatureDriftReport) -> String {
    format!(
        "fixture sr={}, ch={}, frame_ms={}, extra_delay={}, file={}; \
         locked bounds: MAX_ABS_FEATURE_DRIFT={:?}, MAX_RMS_PER_FRAME_DRIFT={:?}, \
         MAX_DRIFTING_FRAME_COUNT={}, FIRST_DIVERGENT_FRAME_AT_LEAST={}; \
         observed: max_abs_drift={:?} (frame={:?}, feat_idx={:?}), \
         max_rms_per_frame={:?} (frame={:?}), \
         drifting_frame_count = {} / {}  (frames where any of {} features differs by > 0.0 absolute; \
         bit-exact under -ffp-contract=off), \
         first_divergent_frame={:?}, n_compared_frames={}; \
         per_feature_max_abs_drift={:?}; \
         first_mismatches=(frame, feat_idx, c, r, |c-r|): {:?}",
        FIXTURE_SR,
        FIXTURE_CH,
        FIXTURE_FRAME_MS,
        FIXTURE_EXTRA_DELAY,
        FIXTURE_NAME,
        MAX_ABS_FEATURE_DRIFT,
        MAX_RMS_PER_FRAME_DRIFT,
        MAX_DRIFTING_FRAME_COUNT,
        FIRST_DIVERGENT_FRAME_AT_LEAST,
        report.max_abs_drift,
        report.max_abs_drift_frame,
        report.max_abs_drift_feature_idx,
        report.max_rms_per_frame,
        report.max_rms_per_frame_idx,
        report.drifting_frame_count,
        report.n_compared_frames,
        NB_TOTAL_FEATURES,
        report.first_divergent_frame,
        report.n_compared_frames,
        report.per_feature_max_abs_drift,
        report.first_mismatches,
    )
}

#[test]
fn test_lpcnet_feature_drift_report_all_equal() {
    let frame = vec![0.5f32; NB_TOTAL_FEATURES];
    let pairs = vec![
        (frame.clone(), frame.clone()),
        (frame.clone(), frame.clone()),
        (frame.clone(), frame.clone()),
    ];
    let report = lpcnet_feature_drift_report(&pairs);

    assert_eq!(report.n_compared_frames, 3);
    assert_eq!(report.first_divergent_frame, None);
    assert_eq!(report.max_abs_drift, 0.0);
    assert_eq!(report.max_abs_drift_frame, None);
    assert_eq!(report.max_abs_drift_feature_idx, None);
    assert_eq!(report.max_rms_per_frame, 0.0);
    assert_eq!(report.drifting_frame_count, 0);
    assert!(report.first_mismatches.is_empty());
    assert!(report.per_feature_max_abs_drift.iter().all(|&v| v == 0.0));
}

#[test]
fn test_lpcnet_feature_drift_report_tracks_first_divergent_frame() {
    let zero = vec![0.0f32; NB_TOTAL_FEATURES];
    let mut second_r = zero.clone();
    second_r[7] = 0.25;
    let pairs = vec![
        (zero.clone(), zero.clone()),
        (zero.clone(), second_r.clone()),
    ];
    let report = lpcnet_feature_drift_report(&pairs);

    assert_eq!(report.first_divergent_frame, Some(1));
    assert_eq!(report.drifting_frame_count, 1);
    assert_eq!(report.max_abs_drift, 0.25);
    assert_eq!(report.max_abs_drift_frame, Some(1));
    assert_eq!(report.max_abs_drift_feature_idx, Some(7));
    assert_eq!(report.first_mismatches.len(), 1);
    assert_eq!(report.first_mismatches[0].0, 1);
    assert_eq!(report.first_mismatches[0].1, 7);
}

#[test]
fn test_lpcnet_feature_drift_report_rms_matches_hand_calc() {
    // diff = [1.0, -2.0, 0.0, ..., 0.0] (length 36).
    // mean(diff^2) = (1 + 4) / 36 = 5/36 -> sqrt = 0.372677996...
    let mut c = vec![0.0f32; NB_TOTAL_FEATURES];
    let mut r = vec![0.0f32; NB_TOTAL_FEATURES];
    c[0] = 1.0;
    c[1] = -2.0;
    r[0] = 0.0;
    r[1] = 0.0;
    let pairs = vec![(c, r)];
    let report = lpcnet_feature_drift_report(&pairs);

    let expected = (5.0_f64 / NB_TOTAL_FEATURES as f64).sqrt() as f32;
    assert!(
        (report.max_rms_per_frame - expected).abs() <= f32::EPSILON * 4.0,
        "rms {} vs expected {}",
        report.max_rms_per_frame,
        expected
    );
    assert_eq!(report.max_rms_per_frame_idx, Some(0));
    assert_eq!(report.max_abs_drift, 2.0);
    assert_eq!(report.max_abs_drift_feature_idx, Some(1));
}

#[test]
fn test_lpcnet_feature_drift_report_per_feature_max_drift_array() {
    // Two synthetic frames with different drifts at different feature
    // indices; the per-feature array should record the max abs drift at
    // each index across all frames.
    let zero = vec![0.0f32; NB_TOTAL_FEATURES];
    let mut frame_a = zero.clone();
    frame_a[3] = 0.10; // feature 3: drift 0.10 in frame 0
    frame_a[5] = -0.05; // feature 5: drift 0.05 in frame 0
    let mut frame_b = zero.clone();
    frame_b[3] = 0.07; // feature 3: drift 0.07 in frame 1 (smaller)
    frame_b[5] = 0.20; // feature 5: drift 0.20 in frame 1 (larger)
    let pairs = vec![(zero.clone(), frame_a), (zero.clone(), frame_b)];
    let report = lpcnet_feature_drift_report(&pairs);

    assert_eq!(report.per_feature_max_abs_drift.len(), 36);
    assert!((report.per_feature_max_abs_drift[3] - 0.10).abs() <= f32::EPSILON * 4.0);
    assert!((report.per_feature_max_abs_drift[5] - 0.20).abs() <= f32::EPSILON * 4.0);
    for (j, &v) in report.per_feature_max_abs_drift.iter().enumerate() {
        if j != 3 && j != 5 {
            assert_eq!(v, 0.0, "feature {} should be zero drift", j);
        }
    }
}

fn pcm_i16_to_f32(s: i16) -> f32 {
    s as f32 * (1.0 / 32768.0)
}

struct Wav {
    sample_rate: u32,
    channels: u16,
    samples: Vec<i16>,
}

fn read_wav(path: &PathBuf) -> Wav {
    let data = fs::read(path).unwrap_or_else(|e| {
        panic!("cannot read {}: {}", path.display(), e);
    });
    assert!(data.len() >= 44, "WAV too small");
    assert_eq!(&data[0..4], b"RIFF");
    assert_eq!(&data[8..12], b"WAVE");

    let mut pos = 12;
    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut bits_per_sample = 0u16;
    let mut pcm: Vec<i16> = Vec::new();

    while pos + 8 <= data.len() {
        let id = &data[pos..pos + 4];
        let sz = u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]])
            as usize;
        if id == b"fmt " {
            channels = u16::from_le_bytes([data[pos + 10], data[pos + 11]]);
            sample_rate = u32::from_le_bytes([
                data[pos + 12],
                data[pos + 13],
                data[pos + 14],
                data[pos + 15],
            ]);
            bits_per_sample = u16::from_le_bytes([data[pos + 22], data[pos + 23]]);
        } else if id == b"data" {
            assert_eq!(bits_per_sample, 16, "only 16-bit PCM supported");
            let bytes = &data[pos + 8..pos + 8 + sz];
            pcm = bytes
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect();
        }
        pos += 8 + sz;
        if !sz.is_multiple_of(2) {
            pos += 1;
        }
    }
    assert!(sample_rate > 0, "no fmt chunk");
    assert!(!pcm.is_empty(), "no data chunk");
    Wav {
        sample_rate,
        channels,
        samples: pcm,
    }
}

fn weights_or_skip() -> bool {
    if WEIGHTS_BLOB.is_empty() {
        eprintln!(
            "dred_lpcnet_feature_drift: WEIGHTS_BLOB empty -- skipping. \
             Run `cargo run -p fetch-assets -- all` to populate."
        );
        return false;
    }
    true
}

fn vectors_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("tests")
        .join("vectors")
        .join(name)
}

#[test]
fn test_dred_lpcnet_feature_drift_is_bounded_against_c_reference() {
    if !weights_or_skip() {
        return;
    }

    let path = vectors_path(FIXTURE_NAME);
    let wav = read_wav(&path);
    assert_eq!(wav.sample_rate, FIXTURE_SR as u32);
    assert_eq!(wav.channels, FIXTURE_CH as u16);

    let fs_rate = wav.sample_rate as i32;
    let channels = wav.channels as i32;

    let c_enc = unsafe { ropus_test_dredenc_new(fs_rate, channels) };
    assert!(
        !c_enc.is_null(),
        "C dred_encoder_init failed (weights not loaded?)"
    );
    let mut r_enc = DREDEnc::new(fs_rate, channels);
    assert!(
        r_enc.loaded,
        "Rust DREDEnc embedded-blob load failed -- check gen_weights_blob.c"
    );

    let frame_size = (fs_rate * FIXTURE_FRAME_MS / 1000) as usize;
    let total_frames = wav.samples.len() / (frame_size * channels as usize);
    let pcm_f: Vec<f32> = wav.samples.iter().map(|&s| pcm_i16_to_f32(s)).collect();

    let mut per_frame: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();

    for fi in 0..total_frames {
        let sample_start = fi * frame_size * channels as usize;
        let sample_end = sample_start + frame_size * channels as usize;
        let frame_pcm = &pcm_f[sample_start..sample_end];

        unsafe {
            ropus_test_dred_compute_latents(
                c_enc,
                frame_pcm.as_ptr(),
                frame_size as i32,
                FIXTURE_EXTRA_DELAY,
            );
        }
        r_enc.compute_latents(frame_pcm, frame_size as i32, FIXTURE_EXTRA_DELAY);

        let c_fill = unsafe { ropus_test_dredenc_latents_buffer_fill(c_enc as *const c_void) };
        if c_fill > 0 {
            let mut c_feat = vec![0.0f32; NB_TOTAL_FEATURES];
            unsafe {
                ropus_test_dredenc_copy_lpcnet_features(
                    c_enc as *const c_void,
                    c_feat.as_mut_ptr(),
                    NB_TOTAL_FEATURES as i32,
                );
            }
            let r_feat = r_enc.lpcnet_enc_state.features[..NB_TOTAL_FEATURES].to_vec();
            per_frame.push((c_feat, r_feat));
        }
    }

    unsafe {
        ropus_test_dredenc_free(c_enc);
    }

    let report = lpcnet_feature_drift_report(&per_frame);
    let context = lpcnet_feature_drift_failure_context(&report);

    eprintln!("dred lpcnet feature drift report: {context}");

    assert!(report.n_compared_frames >= 4, "{context}");
    // FIRST_DIVERGENT_FRAME_AT_LEAST is locked at 0 because drift starts at
    // the first feature-valid frame today. Re-locking to a positive value
    // (after remediation pushes first divergence later) will activate this
    // guard; until then `i >= 0` is trivially true on usize.
    #[allow(clippy::absurd_extreme_comparisons)]
    {
        assert!(
            report
                .first_divergent_frame
                .is_none_or(|i| i >= FIRST_DIVERGENT_FRAME_AT_LEAST),
            "{context}"
        );
    }
    assert!(report.max_abs_drift <= MAX_ABS_FEATURE_DRIFT, "{context}");
    assert!(
        report.max_rms_per_frame <= MAX_RMS_PER_FRAME_DRIFT,
        "{context}"
    );
    assert!(
        report.drifting_frame_count <= MAX_DRIFTING_FRAME_COUNT,
        "{context}"
    );

    // Per-feature bit-exact lock: every feature except feats 18 (dnn_pitch)
    // and 19 (frame_corr) is bit-exact against the C reference after the
    // celt_fir_f32 summation-order fix. Catches future regressions on the
    // 34 bit-exact features even if aggregate bounds remain satisfied.
    for &j in BIT_EXACT_FEATURE_INDICES {
        assert_eq!(
            report.per_feature_max_abs_drift[j], 0.0,
            "feature {j} regressed from bit-exact: per_feature_max_abs_drift[{j}] = {} ({context})",
            report.per_feature_max_abs_drift[j],
        );
    }

    // Feature-18-specific bound (PitchDNN neural inference residual).
    assert!(
        report.per_feature_max_abs_drift[18] <= FEAT18_MAX_ABS_DRIFT,
        "feat 18 (dnn_pitch) drift exceeded FEAT18_MAX_ABS_DRIFT={FEAT18_MAX_ABS_DRIFT:?}: \
         observed {} ({context})",
        report.per_feature_max_abs_drift[18],
    );

    // Feature-19-specific bound (frame_corr float-arithmetic residual).
    assert!(
        report.per_feature_max_abs_drift[19] <= FEAT19_MAX_ABS_DRIFT,
        "feat 19 (frame_corr) drift exceeded FEAT19_MAX_ABS_DRIFT={FEAT19_MAX_ABS_DRIFT:?}: \
         observed {} ({context})",
        report.per_feature_max_abs_drift[19],
    );
}
