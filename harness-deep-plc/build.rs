//! Build script for the sibling DEEP_PLC float-mode harness.
//!
//! Compiles the xiph/opus C reference in FLOAT mode (no `FIXED_POINT` define)
//! with `ENABLE_DEEP_PLC=1` and the full DNN C source set linked. The resulting
//! static library is `opus_ref_float` — distinct from the fixed-point harness's
//! `opus_ref` — so both harnesses can coexist in the same workspace without
//! symbol collisions.
//!
//! See `wrk_journals/2026.04.19 - JRN - stage7-dnn-wiring-supervisor.md`
//! Stage 7b.2 for the motivation: xiph's `configure.ac:973` forbids
//! FIXED_POINT + DEEP_PLC, so we need a second build flavour for the PLC
//! comparison harness.

use std::path::PathBuf;

fn main() {
    let ref_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../reference");
    let harness_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".");

    if !ref_dir.join("celt/bands.c").exists() {
        panic!(
            "\n\n\
             === Reference opus C source not found ===\n\
             Expected under: {}\n\n\
             Fetch it (pinned commit, idempotent):\n\
             \n\
             \x20   cargo run -p fetch-assets -- all\n\
             \n\
             (Must be `all`, not `reference` — this harness needs the DNN\n\
              weights tarball too.)\n\n",
            ref_dir.display()
        );
    }
    // DEEP_PLC requires the weights tarball, not just the source tree.
    if !ref_dir.join("dnn/fargan_data.c").exists() || !ref_dir.join("dnn/plc_data.c").exists() {
        panic!(
            "\n\n\
             === DNN weight source files not found ===\n\
             Expected under: {}/dnn/\n\n\
             The DEEP_PLC harness needs the xiph weights tarball unpacked\n\
             (generates `dnn/*_data.c` files):\n\
             \n\
             \x20   cargo run -p fetch-assets -- all\n\
             \n",
            ref_dir.display()
        );
    }

    // --- CELT sources (platform-independent only) ---
    // Mirrors harness/build.rs; SIMD excluded.
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

    // --- SILK float-point sources (this is the float-mode harness) ---
    // Derived from xiph's `silk_sources_float` in silk_sources.mk / Makefile.am.
    let silk_float_sources = [
        "silk/float/apply_sine_window_FLP.c",
        "silk/float/corrMatrix_FLP.c",
        "silk/float/encode_frame_FLP.c",
        "silk/float/find_LPC_FLP.c",
        "silk/float/find_LTP_FLP.c",
        "silk/float/find_pitch_lags_FLP.c",
        "silk/float/find_pred_coefs_FLP.c",
        "silk/float/LPC_analysis_filter_FLP.c",
        "silk/float/LTP_analysis_filter_FLP.c",
        "silk/float/LTP_scale_ctrl_FLP.c",
        "silk/float/noise_shape_analysis_FLP.c",
        "silk/float/process_gains_FLP.c",
        "silk/float/regularize_correlations_FLP.c",
        "silk/float/residual_energy_FLP.c",
        "silk/float/warped_autocorrelation_FLP.c",
        "silk/float/wrappers_FLP.c",
        "silk/float/autocorrelation_FLP.c",
        "silk/float/burg_modified_FLP.c",
        "silk/float/bwexpander_FLP.c",
        "silk/float/energy_FLP.c",
        "silk/float/inner_product_FLP.c",
        "silk/float/k2a_FLP.c",
        "silk/float/LPC_inv_pred_gain_FLP.c",
        "silk/float/pitch_analysis_core_FLP.c",
        "silk/float/scale_copy_vector_FLP.c",
        "silk/float/scale_vector_FLP.c",
        "silk/float/schur_FLP.c",
        "silk/float/sort_FLP.c",
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
        "src/analysis.c",
        "src/mlp.c",
        "src/mlp_data.c",
    ];

    // --- DNN / DEEP_PLC sources ---
    //
    // Authoritative list: `reference/lpcnet_sources.mk` `DEEP_PLC_SOURCES`.
    // Compile-time weights are provided by the `*_data.c` files from the
    // xiph weights tarball. Platform-specific SIMD under `dnn/x86/` and
    // `dnn/arm/` is excluded — scalar only, same choice as the sibling
    // fixed-point harness.
    let dnn_sources = [
        "dnn/burg.c",
        "dnn/freq.c",
        "dnn/fargan.c",
        "dnn/fargan_data.c",
        "dnn/lpcnet_enc.c",
        "dnn/lpcnet_plc.c",
        "dnn/lpcnet_tables.c",
        "dnn/nnet.c",
        "dnn/nnet_default.c",
        "dnn/plc_data.c",
        "dnn/parse_lpcnet_weights.c",
        "dnn/pitchdnn.c",
        "dnn/pitchdnn_data.c",
    ];

    // --- DRED sources ---
    //
    // Stage 8.4 adds the RDOVAE encoder forward pass for C-vs-Rust
    // differential testing. These files need no ENABLE_DRED gate at the
    // call level (the top-level DRED features in opus_encoder.c do, but
    // `dred_rdovae_enc.c` itself is just a pure forward pass the test
    // invokes directly). The weight tables arrive via
    // `dred_rdovae_enc_data.c`. Matches xiph's `DRED_SOURCES` list in
    // `reference/lpcnet_sources.mk`.
    //
    // Stage 8.6 adds `dred_encoder.c` / `dred_coding.c` — the full
    // encoder-side pipeline (`dred_compute_latents`,
    // `dred_encode_silk_frame`, `compute_quantizer`). These reference
    // the LPCNet feature extractor, which is already linked via
    // `lpcnet_enc.c` above.
    let dred_sources = [
        "dnn/dred_rdovae_enc.c",
        "dnn/dred_rdovae_enc_data.c",
        "dnn/dred_rdovae_dec.c",
        "dnn/dred_rdovae_dec_data.c",
        "dnn/dred_rdovae_stats_data.c",
        "dnn/dred_encoder.c",
        "dnn/dred_coding.c",
        // Stage 8.7: need `dred_ec_decode` symbol for the C-diff test
        // that parses encoder-emitted payloads back with the C decoder.
        "dnn/dred_decoder.c",
    ];

    let mut build = cc::Build::new();

    build
        .warnings(false)
        // Include paths
        .include(&harness_dir) // for our config.h
        .include(ref_dir.join("include"))
        .include(ref_dir.join("celt"))
        .include(ref_dir.join("silk"))
        .include(ref_dir.join("silk/float"))
        .include(&ref_dir) // for src/ internal headers
        .include(ref_dir.join("src"))
        .include(ref_dir.join("dnn"))
        // Defines
        .define("HAVE_CONFIG_H", "1")
        .define("OPUS_BUILD", None);

    // Disable compiler-driven multiply-add fusion so the C reference and the
    // Rust side agree bit-for-bit on scalar float paths. MSVC `/fp:precise`
    // already prohibits contraction; this flag matches gcc/clang behaviour.
    // Silent no-op on MSVC.
    build.flag_if_supported("-ffp-contract=off");

    // Add all source files
    for src in &celt_sources {
        build.file(ref_dir.join(src));
    }
    for src in &silk_sources {
        build.file(ref_dir.join(src));
    }
    for src in &silk_float_sources {
        build.file(ref_dir.join(src));
    }
    for src in &opus_sources {
        build.file(ref_dir.join(src));
    }
    for src in &dnn_sources {
        build.file(ref_dir.join(src));
    }
    for src in &dred_sources {
        build.file(ref_dir.join(src));
    }

    // Local shim that exposes `dred_rdovae_encode_dframe` via opaque
    // pointers so the Rust differential test can call it without
    // replicating the C struct layouts.
    build.file(harness_dir.join("dred_enc_shim.c"));
    // Sibling shim for stage 8.5: RDOVAE decoder forward pass + init.
    build.file(harness_dir.join("dred_dec_shim.c"));
    // Stage 8.6 shim: full encoder-side pipeline (DREDEnc init, compute
    // latents, encode silk frame) behind opaque handles.
    build.file(harness_dir.join("dred_encode_shim.c"));

    build.compile("opus_ref_float");

    println!("cargo:rustc-link-lib=static=opus_ref_float");
    println!("cargo:rerun-if-changed=config.h");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=dred_enc_shim.c");
    println!("cargo:rerun-if-changed=dred_dec_shim.c");
    println!("cargo:rerun-if-changed=dred_encode_shim.c");
}
