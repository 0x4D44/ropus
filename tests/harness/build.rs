//! Build script: compiles the xiph/opus C reference via the `cc` crate.
//!
//! We build fixed-point, platform-independent (no SIMD), no DNN.
//! The resulting static library is linked as `opus_ref` so FFI bindings
//! can call into it.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=MDOPUS_SKIP_REFERENCE");

    let ref_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("reference");
    let harness_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/harness");
    let skip_reference = std::env::var_os("MDOPUS_SKIP_REFERENCE").is_some();

    if skip_reference {
        println!(
            "cargo:warning=Skipping C reference harness build because MDOPUS_SKIP_REFERENCE is set"
        );
        return;
    }

    if !ref_dir.join("celt/bands.c").exists() {
        panic!(
            "Reference opus source not found under {}. \
             Clone https://github.com/xiph/opus into reference/ or set MDOPUS_SKIP_REFERENCE=1 \
             for library-only test/coverage runs.",
            ref_dir.display()
        );
    }

    // --- CELT sources (platform-independent only) ---
    let celt_sources = [
        "celt/bands.c",
        "celt/celt.c",
        "celt/celt_encoder.c",
        "celt/celt_decoder.c",
        "celt/cwrs.c",
        "celt/entcode.c",
        "celt/entdec.c",
        "celt/entenc.c",
        "celt/kiss_fft.c",
        "celt/laplace.c",
        "celt/mathops.c",
        "celt/mdct.c",
        "celt/modes.c",
        "celt/pitch.c",
        "celt/celt_lpc.c",
        "celt/quant_bands.c",
        "celt/rate.c",
        "celt/vq.c",
    ];

    // --- SILK common sources ---
    let silk_sources = [
        "silk/CNG.c",
        "silk/code_signs.c",
        "silk/init_decoder.c",
        "silk/decode_core.c",
        "silk/decode_frame.c",
        "silk/decode_parameters.c",
        "silk/decode_indices.c",
        "silk/decode_pulses.c",
        "silk/decoder_set_fs.c",
        "silk/dec_API.c",
        "silk/enc_API.c",
        "silk/encode_indices.c",
        "silk/encode_pulses.c",
        "silk/gain_quant.c",
        "silk/interpolate.c",
        "silk/LP_variable_cutoff.c",
        "silk/NLSF_decode.c",
        "silk/NSQ.c",
        "silk/NSQ_del_dec.c",
        "silk/PLC.c",
        "silk/shell_coder.c",
        "silk/tables_gain.c",
        "silk/tables_LTP.c",
        "silk/tables_NLSF_CB_NB_MB.c",
        "silk/tables_NLSF_CB_WB.c",
        "silk/tables_other.c",
        "silk/tables_pitch_lag.c",
        "silk/tables_pulses_per_block.c",
        "silk/VAD.c",
        "silk/control_audio_bandwidth.c",
        "silk/quant_LTP_gains.c",
        "silk/VQ_WMat_EC.c",
        "silk/HP_variable_cutoff.c",
        "silk/NLSF_encode.c",
        "silk/NLSF_VQ.c",
        "silk/NLSF_unpack.c",
        "silk/NLSF_del_dec_quant.c",
        "silk/process_NLSFs.c",
        "silk/stereo_LR_to_MS.c",
        "silk/stereo_MS_to_LR.c",
        "silk/check_control_input.c",
        "silk/control_SNR.c",
        "silk/init_encoder.c",
        "silk/control_codec.c",
        "silk/A2NLSF.c",
        "silk/ana_filt_bank_1.c",
        "silk/biquad_alt.c",
        "silk/bwexpander_32.c",
        "silk/bwexpander.c",
        "silk/debug.c",
        "silk/decode_pitch.c",
        "silk/inner_prod_aligned.c",
        "silk/lin2log.c",
        "silk/log2lin.c",
        "silk/LPC_analysis_filter.c",
        "silk/LPC_inv_pred_gain.c",
        "silk/table_LSF_cos.c",
        "silk/NLSF2A.c",
        "silk/NLSF_stabilize.c",
        "silk/NLSF_VQ_weights_laroia.c",
        "silk/pitch_est_tables.c",
        "silk/resampler.c",
        "silk/resampler_down2_3.c",
        "silk/resampler_down2.c",
        "silk/resampler_private_AR2.c",
        "silk/resampler_private_down_FIR.c",
        "silk/resampler_private_IIR_FIR.c",
        "silk/resampler_private_up2_HQ.c",
        "silk/resampler_rom.c",
        "silk/sigm_Q15.c",
        "silk/sort.c",
        "silk/sum_sqr_shift.c",
        "silk/stereo_decode_pred.c",
        "silk/stereo_encode_pred.c",
        "silk/stereo_find_predictor.c",
        "silk/stereo_quant_pred.c",
        "silk/LPC_fit.c",
    ];

    // --- SILK fixed-point sources ---
    let silk_fixed_sources = [
        "silk/fixed/LTP_analysis_filter_FIX.c",
        "silk/fixed/LTP_scale_ctrl_FIX.c",
        "silk/fixed/corrMatrix_FIX.c",
        "silk/fixed/encode_frame_FIX.c",
        "silk/fixed/find_LPC_FIX.c",
        "silk/fixed/find_LTP_FIX.c",
        "silk/fixed/find_pitch_lags_FIX.c",
        "silk/fixed/find_pred_coefs_FIX.c",
        "silk/fixed/noise_shape_analysis_FIX.c",
        "silk/fixed/process_gains_FIX.c",
        "silk/fixed/regularize_correlations_FIX.c",
        "silk/fixed/residual_energy16_FIX.c",
        "silk/fixed/residual_energy_FIX.c",
        "silk/fixed/warped_autocorrelation_FIX.c",
        "silk/fixed/apply_sine_window_FIX.c",
        "silk/fixed/autocorr_FIX.c",
        "silk/fixed/burg_modified_FIX.c",
        "silk/fixed/k2a_FIX.c",
        "silk/fixed/k2a_Q16_FIX.c",
        "silk/fixed/pitch_analysis_core_FIX.c",
        "silk/fixed/vector_ops_FIX.c",
        "silk/fixed/schur64_FIX.c",
        "silk/fixed/schur_FIX.c",
    ];

    // --- CELT x86 SIMD sources ---
    let celt_x86_sources = [
        "celt/x86/x86cpu.c",
        "celt/x86/x86_celt_map.c",
        "celt/x86/pitch_sse.c",
        "celt/x86/pitch_sse2.c",
        "celt/x86/pitch_sse4_1.c",
        "celt/x86/vq_sse2.c",
        "celt/x86/celt_lpc_sse4_1.c",
    ];

    // --- SILK x86 SIMD sources ---
    let silk_x86_sources = [
        "silk/x86/x86_silk_map.c",
        "silk/x86/NSQ_sse4_1.c",
        "silk/x86/NSQ_del_dec_sse4_1.c",
        "silk/x86/VAD_sse4_1.c",
        "silk/x86/VQ_WMat_EC_sse4_1.c",
        "silk/fixed/x86/burg_modified_FIX_sse4_1.c",
        "silk/fixed/x86/vector_ops_FIX_sse4_1.c",
    ];

    // --- Opus top-level sources ---
    let opus_sources = [
        "src/opus.c",
        "src/opus_decoder.c",
        "src/opus_encoder.c",
        "src/extensions.c",
        "src/opus_multistream.c",
        "src/opus_multistream_encoder.c",
        "src/opus_multistream_decoder.c",
        "src/repacketizer.c",
        "src/opus_projection_encoder.c",
        "src/opus_projection_decoder.c",
        "src/mapping_matrix.c",
        // Float analysis (needed by encoder even in fixed-point mode)
        "src/analysis.c",
        "src/mlp.c",
        "src/mlp_data.c",
    ];

    let mut build = cc::Build::new();

    build
        .warnings(false)
        // Include paths
        .include(&harness_dir) // for our config.h
        .include(ref_dir.join("include"))
        .include(ref_dir.join("celt"))
        .include(ref_dir.join("silk"))
        .include(ref_dir.join("silk/fixed"))
        .include(&ref_dir) // for src/ internal headers
        .include(ref_dir.join("src"))
        // Defines
        .define("HAVE_CONFIG_H", "1")
        .define("OPUS_BUILD", None);

    // --- Fuzzing context detection ---
    //
    // When built under cargo-fuzz, the Rust code is compiled with
    // AddressSanitizer + libFuzzer coverage instrumentation. The C reference
    // should match so that (a) C-side memory errors become clean ASAN reports
    // instead of raw segfaults, and (b) libFuzzer's coverage guidance includes
    // code paths through the C library. Without this, C memory bugs would be
    // indistinguishable from Rust bugs in differential fuzzing.
    //
    // cargo-fuzz sets `--cfg fuzzing` for all crates in the build, which
    // surfaces to build.rs as CARGO_CFG_FUZZING. We additionally check
    // CARGO_ENCODED_RUSTFLAGS to confirm ASAN is actually requested (cargo-fuzz
    // disables sanitizers on platforms where they're unsupported, e.g. Windows).
    let fuzzing = std::env::var("CARGO_CFG_FUZZING").is_ok();
    let rustflags = std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    let asan_requested = rustflags.contains("sanitizer=address");

    if fuzzing && asan_requested {
        // ASAN instrumentation for the C code. flag_if_supported is a no-op on
        // compilers that don't understand -fsanitize (e.g. MSVC).
        build.flag_if_supported("-fsanitize=address");
        build.flag_if_supported("-fno-omit-frame-pointer");
        // libFuzzer coverage instrumentation matches what cargo-fuzz asks for
        // on the Rust side, so the fuzzer can learn from C code paths too.
        build.flag_if_supported("-fsanitize-coverage=inline-8bit-counters,pc-table,trace-cmp");
    }
    println!("cargo:rerun-if-env-changed=CARGO_CFG_FUZZING");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");

    // Add all source files
    for src in &celt_sources {
        build.file(ref_dir.join(src));
    }
    for src in &celt_x86_sources {
        build.file(ref_dir.join(src));
    }
    for src in &silk_sources {
        build.file(ref_dir.join(src));
    }
    for src in &silk_fixed_sources {
        build.file(ref_dir.join(src));
    }
    for src in &silk_x86_sources {
        build.file(ref_dir.join(src));
    }
    for src in &opus_sources {
        build.file(ref_dir.join(src));
    }

    // Debug helper for direct function comparisons
    build.file(harness_dir.join("debug_helper.c"));

    build.compile("opus_ref");

    // Tell cargo to link the static library
    println!("cargo:rustc-link-lib=static=opus_ref");
    println!("cargo:rerun-if-changed=tests/harness/config.h");
    println!("cargo:rerun-if-changed=tests/harness/build.rs");
    println!("cargo:rerun-if-changed=tests/harness/debug_helper.c");
    println!("cargo:rerun-if-changed=reference/celt/celt_decoder.c");
}
// force rebuild
// force rebuild 2
