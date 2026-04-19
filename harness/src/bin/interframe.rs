//! ropus-interframe: Compare SILK inter-frame state between Rust and C
//! reference encoders to pinpoint multi-frame divergence.

#![allow(
    clippy::needless_range_loop,
    clippy::manual_range_contains,
    clippy::unnecessary_cast,
    clippy::collapsible_if,
    clippy::identity_op,
    clippy::items_after_test_module,
    clippy::single_match,
    clippy::unnecessary_unwrap
)]

#[path = "../bindings.rs"]
mod bindings;

use std::os::raw::c_int;

use ropus::opus::encoder::{
    OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_APPLICATION_VOIP,
    OpusEncoder as RustOpusEncoder,
};

const SAMPLE_RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];
const APPLICATIONS: [i32; 3] = [
    OPUS_APPLICATION_VOIP,
    OPUS_APPLICATION_AUDIO,
    OPUS_APPLICATION_RESTRICTED_LOWDELAY,
];

fn byte_to_bitrate(b0: u8, b1: u8) -> i32 {
    let raw = u16::from_le_bytes([b0, b1]) as i32;
    6000 + (raw % 504001)
}

/// Holds C-side inter-frame state extracted via FFI.
#[derive(Debug)]
struct CInterframeState {
    last_gain_index: i32,
    prev_gain_q16: i32,
    variable_hp_smth1_q15: i32,
    variable_hp_smth2_q15: i32,
    harm_shape_gain_smth: i32,
    tilt_smth: i32,
    prev_signal_type: i32,
    prev_lag: i32,
    ec_prev_lag_index: i32,
    ec_prev_signal_type: i32,
    prev_nlsfq_q15: [i16; 16],
    stereo_width_prev_q14: i32,
    stereo_smth_width_q14: i32,
    stereo_pred_prev_q13_0: i32,
    stereo_pred_prev_q13_1: i32,
    n_bits_exceeded: i32,
}

fn get_c_interframe_state(enc: *mut bindings::OpusEncoder, channel: i32) -> CInterframeState {
    let mut st = CInterframeState {
        last_gain_index: 0,
        prev_gain_q16: 0,
        variable_hp_smth1_q15: 0,
        variable_hp_smth2_q15: 0,
        harm_shape_gain_smth: 0,
        tilt_smth: 0,
        prev_signal_type: 0,
        prev_lag: 0,
        ec_prev_lag_index: 0,
        ec_prev_signal_type: 0,
        prev_nlsfq_q15: [0i16; 16],
        stereo_width_prev_q14: 0,
        stereo_smth_width_q14: 0,
        stereo_pred_prev_q13_0: 0,
        stereo_pred_prev_q13_1: 0,
        n_bits_exceeded: 0,
    };
    unsafe {
        bindings::debug_get_silk_interframe_state(
            enc,
            channel as c_int,
            &mut st.last_gain_index,
            &mut st.prev_gain_q16,
            &mut st.variable_hp_smth1_q15,
            &mut st.variable_hp_smth2_q15,
            &mut st.harm_shape_gain_smth,
            &mut st.tilt_smth,
            &mut st.prev_signal_type,
            &mut st.prev_lag,
            &mut st.ec_prev_lag_index,
            &mut st.ec_prev_signal_type,
            st.prev_nlsfq_q15.as_mut_ptr(),
            &mut st.stereo_width_prev_q14,
            &mut st.stereo_smth_width_q14,
            &mut st.stereo_pred_prev_q13_0,
            &mut st.stereo_pred_prev_q13_1,
            &mut st.n_bits_exceeded,
        );
    }
    st
}

/// Holds Rust-side inter-frame state read directly from the encoder.
#[derive(Debug)]
struct RustInterframeState {
    last_gain_index: i32,
    prev_gain_q16: i32,
    variable_hp_smth1_q15: i32,
    variable_hp_smth2_q15: i32,
    harm_shape_gain_smth: i32,
    tilt_smth: i32,
    prev_signal_type: i32,
    prev_lag: i32,
    ec_prev_lag_index: i32,
    ec_prev_signal_type: i32,
    prev_nlsfq_q15: [i16; 16],
    stereo_width_prev_q14: i32,
    stereo_smth_width_q14: i32,
    stereo_pred_prev_q13_0: i32,
    stereo_pred_prev_q13_1: i32,
    n_bits_exceeded: i32,
}

/// Extended SILK encoder state for bug E/F/G/H investigation.
#[derive(Debug, Default)]
struct ExtendedState {
    // NSQ
    nsq_rand_seed: i32,
    nsq_slf_ar_shp_q14: i32,
    nsq_lag_prev: i32,
    nsq_sdiff_shp_q14: i32,
    nsq_sltp_buf_idx: i32,
    nsq_sltp_shp_buf_idx: i32,
    nsq_rewhite_flag: i32,
    nsq_slpc_q14: [i32; 16],
    nsq_sar2_q14: [i32; 16],
    nsq_sltp_shp_q14: [i32; 32],
    // VAD
    vad_hp_state: i32,
    vad_counter: i32,
    vad_noise_level_bias: [i32; 4],
    vad_ana_state: [i32; 2],
    vad_ana_state1: [i32; 2],
    vad_ana_state2: [i32; 2],
    vad_nrg_ratio_smth_q8: [i32; 4],
    vad_nl: [i32; 4],
    vad_inv_nl: [i32; 4],
    // Common
    sum_log_gain_q7: i32,
    in_hp_state: [i32; 2],
    input_tilt_q15: i32,
    input_quality_bands_q15: [i32; 4],
    frame_counter: i32,
    no_speech_counter: i32,
    // LP state
    lp_in_lp_state: [i32; 2],
    lp_transition_frame_no: i32,
    // Shape
    shape_harm_boost_smth: i32,
    // x_buf
    x_buf_hash: i32,
    ltp_corr_q15: i32,
    res_nrg_smth: i32,
}

fn get_c_extended_state(enc: *mut bindings::OpusEncoder, channel: i32) -> ExtendedState {
    let mut s = ExtendedState::default();
    unsafe {
        bindings::debug_get_silk_extended_state(
            enc,
            channel as c_int,
            &mut s.nsq_rand_seed,
            &mut s.nsq_slf_ar_shp_q14,
            &mut s.nsq_lag_prev,
            &mut s.nsq_sdiff_shp_q14,
            &mut s.nsq_sltp_buf_idx,
            &mut s.nsq_sltp_shp_buf_idx,
            &mut s.nsq_rewhite_flag,
            s.nsq_slpc_q14.as_mut_ptr(),
            s.nsq_sar2_q14.as_mut_ptr(),
            s.nsq_sltp_shp_q14.as_mut_ptr(),
            &mut s.vad_hp_state,
            &mut s.vad_counter,
            s.vad_noise_level_bias.as_mut_ptr(),
            s.vad_ana_state.as_mut_ptr(),
            s.vad_ana_state1.as_mut_ptr(),
            s.vad_ana_state2.as_mut_ptr(),
            s.vad_nrg_ratio_smth_q8.as_mut_ptr(),
            s.vad_nl.as_mut_ptr(),
            s.vad_inv_nl.as_mut_ptr(),
            &mut s.sum_log_gain_q7,
            s.in_hp_state.as_mut_ptr(),
            &mut s.input_tilt_q15,
            s.input_quality_bands_q15.as_mut_ptr(),
            &mut s.frame_counter,
            &mut s.no_speech_counter,
            s.lp_in_lp_state.as_mut_ptr(),
            &mut s.lp_transition_frame_no,
            &mut s.shape_harm_boost_smth,
            &mut s.x_buf_hash,
            &mut s.ltp_corr_q15,
            &mut s.res_nrg_smth,
        );
    }
    s
}

fn get_rust_extended_state(enc: &RustOpusEncoder, channel: usize) -> ExtendedState {
    let silk = enc.silk_encoder().expect("SILK encoder not allocated");
    let ch = &silk.state_fxx[channel];
    let st = &ch.s_cmn;

    let mut s = ExtendedState::default();
    s.nsq_rand_seed = st.s_nsq.rand_seed;
    s.nsq_slf_ar_shp_q14 = st.s_nsq.s_lf_ar_shp_q14;
    s.nsq_lag_prev = st.s_nsq.lag_prev;
    s.nsq_sdiff_shp_q14 = st.s_nsq.s_diff_shp_q14;
    s.nsq_sltp_buf_idx = st.s_nsq.s_ltp_buf_idx;
    s.nsq_sltp_shp_buf_idx = st.s_nsq.s_ltp_shp_buf_idx;
    s.nsq_rewhite_flag = st.s_nsq.rewhite_flag;
    for i in 0..16 {
        s.nsq_slpc_q14[i] = st.s_nsq.s_lpc_q14[i];
        s.nsq_sar2_q14[i] = st.s_nsq.s_ar2_q14[i];
    }
    for i in 0..32 {
        s.nsq_sltp_shp_q14[i] = st.s_nsq.s_ltp_shp_q14[i];
    }

    s.vad_hp_state = st.s_vad.hp_state as i32;
    s.vad_counter = st.s_vad.counter;
    for i in 0..4 {
        s.vad_noise_level_bias[i] = st.s_vad.noise_level_bias[i];
        s.vad_nrg_ratio_smth_q8[i] = st.s_vad.nrg_ratio_smth_q8[i];
        s.vad_nl[i] = st.s_vad.nl[i];
        s.vad_inv_nl[i] = st.s_vad.inv_nl[i];
    }
    for i in 0..2 {
        s.vad_ana_state[i] = st.s_vad.ana_state[i];
        s.vad_ana_state1[i] = st.s_vad.ana_state1[i];
        s.vad_ana_state2[i] = st.s_vad.ana_state2[i];
    }

    s.sum_log_gain_q7 = st.sum_log_gain_q7;
    s.in_hp_state[0] = st.in_hp_state[0];
    s.in_hp_state[1] = st.in_hp_state[1];
    s.input_tilt_q15 = st.input_tilt_q15;
    for i in 0..4 {
        s.input_quality_bands_q15[i] = st.input_quality_bands_q15[i];
    }
    s.frame_counter = st.frame_counter;
    s.no_speech_counter = st.no_speech_counter;

    s.lp_in_lp_state[0] = st.s_lp.in_lp_state[0];
    s.lp_in_lp_state[1] = st.s_lp.in_lp_state[1];
    s.lp_transition_frame_no = st.s_lp.transition_frame_no;

    // Rust doesn't have HarmBoost_smth_Q16 — C has it but it's a dead field.
    // Report 0 to match an uninitialized/zero C value if C also zeros it.
    s.shape_harm_boost_smth = 0;

    // x_buf hash
    let mut h: i32 = 0;
    for b in ch.x_buf.iter() {
        h = h.wrapping_mul(31).wrapping_add(*b as i32);
    }
    s.x_buf_hash = h;
    s.ltp_corr_q15 = ch.ltp_corr_q15;
    s.res_nrg_smth = ch.res_nrg_smth;
    s
}

/// Compare extended state — returns the index of the first-mismatched scalar field
/// (or a descriptive name). Prints a MISMATCH line for each divergence.
fn compare_extended(
    frame_idx: usize,
    channel: usize,
    c: &ExtendedState,
    r: &ExtendedState,
) -> usize {
    let mut mismatches = 0usize;

    macro_rules! scalar {
        ($name:ident) => {
            if c.$name != r.$name {
                println!(
                    "  EXT MISMATCH ch{} frame{} {}: C={} Rust={}",
                    channel,
                    frame_idx,
                    stringify!($name),
                    c.$name,
                    r.$name
                );
                mismatches += 1;
            }
        };
    }
    macro_rules! array {
        ($name:ident) => {
            for i in 0..c.$name.len() {
                if c.$name[i] != r.$name[i] {
                    println!(
                        "  EXT MISMATCH ch{} frame{} {}[{}]: C={} Rust={}",
                        channel,
                        frame_idx,
                        stringify!($name),
                        i,
                        c.$name[i],
                        r.$name[i]
                    );
                    mismatches += 1;
                }
            }
        };
    }
    // NSQ
    scalar!(nsq_rand_seed);
    scalar!(nsq_slf_ar_shp_q14);
    scalar!(nsq_lag_prev);
    scalar!(nsq_sdiff_shp_q14);
    scalar!(nsq_sltp_buf_idx);
    scalar!(nsq_sltp_shp_buf_idx);
    scalar!(nsq_rewhite_flag);
    array!(nsq_slpc_q14);
    array!(nsq_sar2_q14);
    array!(nsq_sltp_shp_q14);
    // VAD
    scalar!(vad_hp_state);
    scalar!(vad_counter);
    array!(vad_noise_level_bias);
    array!(vad_ana_state);
    array!(vad_ana_state1);
    array!(vad_ana_state2);
    array!(vad_nrg_ratio_smth_q8);
    array!(vad_nl);
    array!(vad_inv_nl);
    // Common
    scalar!(sum_log_gain_q7);
    array!(in_hp_state);
    scalar!(input_tilt_q15);
    array!(input_quality_bands_q15);
    scalar!(frame_counter);
    scalar!(no_speech_counter);
    // LP
    array!(lp_in_lp_state);
    scalar!(lp_transition_frame_no);
    // x_buf
    scalar!(x_buf_hash);
    scalar!(ltp_corr_q15);
    scalar!(res_nrg_smth);
    mismatches
}

fn get_rust_interframe_state(enc: &RustOpusEncoder, channel: usize) -> RustInterframeState {
    let silk = enc.silk_encoder().expect("SILK encoder not allocated");
    let ch = &silk.state_fxx[channel];
    let st = &ch.s_cmn;

    let mut prev_nlsfq_q15 = [0i16; 16];
    for i in 0..16 {
        prev_nlsfq_q15[i] = st.prev_nlsfq_q15[i];
    }

    RustInterframeState {
        last_gain_index: ch.s_shape.last_gain_index as i32,
        prev_gain_q16: st.s_nsq.prev_gain_q16,
        variable_hp_smth1_q15: st.variable_hp_smth1_q15,
        variable_hp_smth2_q15: st.variable_hp_smth2_q15,
        harm_shape_gain_smth: ch.s_shape.harm_shape_gain_smth,
        tilt_smth: ch.s_shape.tilt_smth,
        prev_signal_type: st.prev_signal_type,
        prev_lag: st.prev_lag,
        ec_prev_lag_index: st.ec_prev_lag_index as i32,
        ec_prev_signal_type: st.ec_prev_signal_type,
        prev_nlsfq_q15,
        stereo_width_prev_q14: silk.s_stereo.width_prev_q14 as i32,
        stereo_smth_width_q14: silk.s_stereo.smth_width_q14 as i32,
        stereo_pred_prev_q13_0: silk.s_stereo.pred_prev_q13[0] as i32,
        stereo_pred_prev_q13_1: silk.s_stereo.pred_prev_q13[1] as i32,
        n_bits_exceeded: silk.n_bits_exceeded,
    }
}

fn compare_and_print(
    frame_idx: usize,
    channel: usize,
    c: &CInterframeState,
    r: &RustInterframeState,
) -> usize {
    let mut mismatches = 0usize;

    macro_rules! cmp_field {
        ($name:ident) => {
            if c.$name != r.$name {
                println!(
                    "  MISMATCH ch{} frame{} {}: C={} Rust={}",
                    channel,
                    frame_idx,
                    stringify!($name),
                    c.$name,
                    r.$name
                );
                mismatches += 1;
            } else {
                println!(
                    "       OK  ch{} frame{} {}: {}",
                    channel,
                    frame_idx,
                    stringify!($name),
                    c.$name
                );
            }
        };
    }

    cmp_field!(last_gain_index);
    cmp_field!(prev_gain_q16);
    cmp_field!(variable_hp_smth1_q15);
    cmp_field!(variable_hp_smth2_q15);
    cmp_field!(harm_shape_gain_smth);
    cmp_field!(tilt_smth);
    cmp_field!(prev_signal_type);
    cmp_field!(prev_lag);
    cmp_field!(ec_prev_lag_index);
    cmp_field!(ec_prev_signal_type);
    cmp_field!(n_bits_exceeded);

    // Compare prev_nlsfq_q15 array
    for i in 0..16 {
        if c.prev_nlsfq_q15[i] != r.prev_nlsfq_q15[i] {
            println!(
                "  MISMATCH ch{} frame{} prev_nlsfq_q15[{}]: C={} Rust={}",
                channel, frame_idx, i, c.prev_nlsfq_q15[i], r.prev_nlsfq_q15[i]
            );
            mismatches += 1;
        }
    }
    let nlsf_match = (0..16).all(|i| c.prev_nlsfq_q15[i] == r.prev_nlsfq_q15[i]);
    if nlsf_match {
        println!(
            "       OK  ch{} frame{} prev_nlsfq_q15: all 16 match",
            channel, frame_idx
        );
    }

    // Stereo state (only print once, for channel 0)
    if channel == 0 {
        cmp_field!(stereo_width_prev_q14);
        cmp_field!(stereo_smth_width_q14);
        cmp_field!(stereo_pred_prev_q13_0);
        cmp_field!(stereo_pred_prev_q13_1);
    }

    mismatches
}

fn compare_nlsf_indices(
    frame_idx: usize,
    channel: usize,
    c_enc: *mut bindings::OpusEncoder,
    rust_enc: &RustOpusEncoder,
) {
    // Get C NLSF indices
    let mut c_indices = [0i8; 17];
    let mut c_order: i32 = 0;
    let mut c_signal_type: i32 = 0;
    let mut c_interp_coef: i32 = 0;
    unsafe {
        bindings::debug_get_silk_nlsf_indices(
            c_enc,
            channel as c_int,
            c_indices.as_mut_ptr(),
            &mut c_order,
            &mut c_signal_type,
            &mut c_interp_coef,
        );
    }

    // Get Rust NLSF indices
    let silk = rust_enc.silk_encoder().expect("SILK encoder");
    let r_ch = &silk.state_fxx[channel];
    let r_order = r_ch.s_cmn.predict_lpc_order as i32;
    let r_signal_type = r_ch.s_cmn.indices.signal_type as i32;
    let r_indices = &r_ch.s_cmn.indices.nlsf_indices;
    let r_interp_coef = r_ch.s_cmn.indices.nlsf_interp_coef_q2 as i32;

    println!(
        "  NLSF diagnostic ch{} frame{}: C order={} sigtype={} interp={}, Rust order={} sigtype={} interp={}",
        channel,
        frame_idx,
        c_order,
        c_signal_type,
        c_interp_coef,
        r_order,
        r_signal_type,
        r_interp_coef
    );

    let order = c_order.max(r_order) as usize;
    let idx0_match = c_indices[0] == r_indices[0];
    println!(
        "    CB1 index: C={} Rust={} {}",
        c_indices[0],
        r_indices[0],
        if idx0_match { "MATCH" } else { "MISMATCH" }
    );

    let mut idx2_mismatches = 0;
    for i in 0..order {
        if c_indices[1 + i] != r_indices[1 + i] {
            println!(
                "    CB2 index[{}]: C={} Rust={}",
                i,
                c_indices[1 + i],
                r_indices[1 + i]
            );
            idx2_mismatches += 1;
        }
    }
    if idx2_mismatches == 0 {
        println!("    CB2 indices: all {} match", order);
    } else {
        println!("    CB2 indices: {} of {} differ", idx2_mismatches, order);
    }

    // Also compare xbuf hash
    let mut c_hash: i32 = 0;
    let mut c_buflen: i32 = 0;
    unsafe {
        bindings::debug_get_silk_xbuf_hash(c_enc, channel as c_int, &mut c_hash, &mut c_buflen);
    }
    // Rust input buffer hash
    let r_buf = &r_ch.s_cmn.input_buf;
    let r_buflen = r_buf.len();
    let mut r_hash: i32 = 0;
    for j in 0..r_buflen.min(2000) {
        r_hash = r_hash.wrapping_mul(31).wrapping_add(r_buf[j] as i32);
    }
    println!(
        "    xbuf hash: C=0x{:08x} (len={}) Rust=0x{:08x} (len={}) {}",
        c_hash as u32,
        c_buflen,
        r_hash as u32,
        r_buflen,
        if c_hash == r_hash {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );
}

fn main() {
    // Load the seed file
    let seed_dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fuzz/corpus/fuzz_encode_multiframe"
    );
    // Try a specific seed if provided as arg, otherwise scan all mf5_24k_stereo seeds
    let args: Vec<String> = std::env::args().collect();
    let seeds: Vec<std::path::PathBuf> = if args.len() > 1 {
        vec![std::path::PathBuf::from(&args[1])]
    } else {
        std::fs::read_dir(seed_dir)
            .expect("cannot read seed dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .map(|n| {
                        n.to_string_lossy().contains("mf5_24k_stereo")
                            || n.to_string_lossy().contains("mf10_24k_stereo")
                    })
                    .unwrap_or(false)
            })
            .collect()
    };
    println!("Scanning {} seed files...\n", seeds.len());

    let mut seeds_with_mismatches = 0usize;

    for seed_path in &seeds {
        let data = std::fs::read(seed_path).unwrap_or_else(|e| {
            eprintln!("SKIP: cannot read seed file {:?}: {}", seed_path, e);
            Vec::new()
        });
        if data.is_empty() {
            continue;
        }

        // Minimal viable size: 7 header bytes + at least one frame's PCM.
        // For single-file mode we relax the lower bound.
        if data.len() < 7 + 320 {
            eprintln!(
                "SKIP: seed {:?} too small ({} bytes)",
                seed_path.file_name().unwrap_or_default(),
                data.len()
            );
            continue;
        }

        // Parse config bytes (same logic as fuzz_encode_multiframe.rs)
        let sample_rate = SAMPLE_RATES[(data[0] as usize) % SAMPLE_RATES.len()];
        let channels = if data[1] & 1 == 0 { 1 } else { 2 };
        let application = APPLICATIONS[(data[2] as usize) % APPLICATIONS.len()];
        let bitrate = byte_to_bitrate(data[3], data[4]);
        let complexity = (data[5] as i32) % 11;
        let num_frames = 5 + ((data[6] as usize) % 6);
        let pcm_bytes = &data[7..];

        let frame_size = sample_rate / 50; // 20ms
        let samples_per_frame = frame_size as usize * channels as usize;
        let bytes_per_frame = samples_per_frame * 2;
        let total_bytes_needed = bytes_per_frame * num_frames;

        if pcm_bytes.len() < total_bytes_needed {
            eprintln!(
                "SKIP: seed {:?} not enough PCM data ({} < {})",
                seed_path.file_name().unwrap_or_default(),
                pcm_bytes.len(),
                total_bytes_needed
            );
            continue;
        }

        // Split input into frames of i16 PCM
        let mut pcm_frames: Vec<Vec<i16>> = Vec::with_capacity(num_frames);
        for i in 0..num_frames {
            let start = i * bytes_per_frame;
            let frame_bytes = &pcm_bytes[start..start + bytes_per_frame];
            let pcm: Vec<i16> = frame_bytes
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect();
            pcm_frames.push(pcm);
        }

        // --- Create Rust encoder ---
        let mut rust_enc = RustOpusEncoder::new(sample_rate, channels, application)
            .expect("Failed to create Rust encoder");
        rust_enc.set_bitrate(bitrate);
        rust_enc.set_vbr(0);
        rust_enc.set_complexity(complexity);

        // --- Create C encoder ---
        let c_enc = unsafe {
            let mut error: c_int = 0;
            let enc = bindings::opus_encoder_create(sample_rate, channels, application, &mut error);
            if enc.is_null() || error != bindings::OPUS_OK {
                eprintln!("SKIP: C opus_encoder_create failed with error {}", error);
                continue;
            }
            bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_BITRATE_REQUEST, bitrate);
            bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_VBR_REQUEST, 0 as c_int);
            bindings::opus_encoder_ctl(enc, bindings::OPUS_SET_COMPLEXITY_REQUEST, complexity);
            enc
        };

        // Encode frames and compare state after each.
        // When a specific seed file is given as arg, check every frame;
        // otherwise cap at 5 for batch speed.
        let single_file_mode = args.len() > 1;
        let frames_to_check = if single_file_mode {
            num_frames
        } else {
            std::cmp::min(num_frames, 5)
        };
        let num_channels = if channels == 2 { 2 } else { 1 };
        let mut seed_has_mismatch = false;

        for frame_idx in 0..frames_to_check {
            let pcm = &pcm_frames[frame_idx];

            // Encode with Rust
            let mut rust_out = vec![0u8; 4000];
            let rust_len = match rust_enc.encode(pcm, frame_size, &mut rust_out, 4000) {
                Ok(n) => n,
                Err(e) => {
                    eprintln!("ERROR: Rust encode frame {} failed: {}", frame_idx, e);
                    break;
                }
            };
            rust_out.truncate(rust_len as usize);

            // Encode with C
            let mut c_out = vec![0u8; 4000];
            let c_len = unsafe {
                bindings::opus_encode(
                    c_enc,
                    pcm.as_ptr() as *const bindings::opus_int16,
                    frame_size,
                    c_out.as_mut_ptr(),
                    4000,
                )
            };
            if c_len < 0 {
                eprintln!("ERROR: C encode frame {} failed: {}", frame_idx, c_len);
                break;
            }
            c_out.truncate(c_len as usize);

            // Compare encoded output
            let bytes_match = rust_out == c_out;

            if !bytes_match && !seed_has_mismatch {
                // First mismatch for this seed — print header
                seed_has_mismatch = true;
                seeds_with_mismatches += 1;
                println!(
                    "\n=== MISMATCH in {:?} ===",
                    seed_path.file_name().unwrap_or_default()
                );
                println!(
                    "Config: sr={}, ch={}, app={}, br={}, cx={}, frames={}",
                    sample_rate, channels, application, bitrate, complexity, num_frames
                );
            }

            if !bytes_match {
                println!(
                    "\n--- Frame {} ---  Rust={} bytes, C={} bytes",
                    frame_idx,
                    rust_out.len(),
                    c_out.len(),
                );

                // Find first byte difference
                let min_len = std::cmp::min(rust_out.len(), c_out.len());
                for b in 0..min_len {
                    if rust_out[b] != c_out[b] {
                        println!(
                            "  First byte diff at offset {}: Rust=0x{:02x} C=0x{:02x}",
                            b, rust_out[b], c_out[b]
                        );
                        break;
                    }
                }

                // Compare inter-frame state for each channel
                for ch in 0..num_channels {
                    println!(
                        "\n  [Channel {}] Inter-frame state after frame {}:",
                        ch, frame_idx
                    );
                    let c_state = get_c_interframe_state(c_enc, ch as i32);
                    let r_state = get_rust_interframe_state(&rust_enc, ch);
                    let mismatches = compare_and_print(frame_idx, ch, &c_state, &r_state);
                    if mismatches > 0 {
                        println!(
                            "  *** {} state mismatches ch{} frame{} ***",
                            mismatches, ch, frame_idx
                        );
                        // NLSF indices diagnostic
                        compare_nlsf_indices(frame_idx, ch, c_enc, &rust_enc);
                    }
                }
            }

            // Always print NLSF indices/xbuf for frames before the first mismatch
            if bytes_match && seed_has_mismatch {
                // State after a matching frame that follows a mismatch
            } else if bytes_match && frame_idx == 0 {
                // Print abbreviated state after frame 0 for context
                for ch in 0..num_channels {
                    compare_nlsf_indices(frame_idx, ch, c_enc, &rust_enc);
                }
            }

            // Extended state comparison (bug E/F/G/H investigation):
            // always dump after every frame when in single-file mode so we
            // can find the first-diverging "untracked" field.
            if single_file_mode {
                let match_tag = if bytes_match {
                    "BYTES-OK"
                } else {
                    "BYTES-DIFF"
                };
                println!(
                    "[ext frame {} {}] Rust={}B C={}B",
                    frame_idx,
                    match_tag,
                    rust_out.len(),
                    c_out.len()
                );
                for ch in 0..num_channels {
                    let c_ext = get_c_extended_state(c_enc, ch as i32);
                    let r_ext = get_rust_extended_state(&rust_enc, ch);
                    let ext_mm = compare_extended(frame_idx, ch, &c_ext, &r_ext);
                    if ext_mm > 0 {
                        println!(
                            "[ext frame {} ch{}] {} extended-state mismatches",
                            frame_idx, ch, ext_mm
                        );
                    } else {
                        println!("[ext frame {} ch{}] extended state OK", frame_idx, ch);
                    }
                }
            }
        }

        // Cleanup
        unsafe {
            bindings::opus_encoder_destroy(c_enc);
        }
    }

    println!(
        "\n=== DONE: scanned {} seeds, {} with mismatches ===",
        seeds.len(),
        seeds_with_mismatches
    );
}
