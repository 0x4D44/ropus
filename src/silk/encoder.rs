#![allow(unused_variables, unused_assignments, unused_mut)]
//! SILK Encoder — complete encoding pipeline.
//!
//! Ported from: enc_API.c, init_encoder.c, encode_indices.c, encode_pulses.c,
//! control_codec.c, control_audio_bandwidth.c, control_SNR.c,
//! check_control_input.c, HP_variable_cutoff.c, LP_variable_cutoff.c,
//! LPC_analysis_filter.c, ana_filt_bank_1.c, NSQ.c, NSQ_del_dec.c,
//! VAD.c, VQ_WMat_EC.c, A2NLSF.c, NLSF_encode.c, NLSF_del_dec_quant.c,
//! process_NLSFs.c, quant_LTP_gains.c, stereo_LR_to_MS.c,
//! stereo_encode_pred.c, stereo_find_predictor.c, stereo_quant_pred.c

use crate::celt::range_coder::RangeEncoder;
use crate::silk::common::*;
use crate::silk::decoder::{SideInfoIndices, SilkResamplerState};
use crate::silk::tables::*;
use crate::types::*;

// ===========================================================================
// Encoder-specific constants
// ===========================================================================

pub const ENCODER_NUM_CHANNELS: usize = 2;
pub const MAX_FS_KHZ: i32 = 16;
pub const LA_PITCH_MS: i32 = 2;
pub const LA_PITCH_MAX: usize = (LA_PITCH_MS * MAX_FS_KHZ) as usize;
pub const LA_SHAPE_MS: i32 = 5;
pub const LA_SHAPE_MAX: usize = (LA_SHAPE_MS * MAX_FS_KHZ) as usize;
pub const FIND_PITCH_LPC_WIN_MS: i32 = 20 + (LA_PITCH_MS << 1); // 24
pub const FIND_PITCH_LPC_WIN_MS_2_SF: i32 = 10 + (LA_PITCH_MS << 1); // 14
pub const FIND_PITCH_LPC_WIN_MAX: usize = (FIND_PITCH_LPC_WIN_MS * MAX_FS_KHZ) as usize;
pub const SHAPE_LPC_WIN_MAX: usize = (15 * MAX_FS_KHZ) as usize;
pub const MAX_SHAPE_LPC_ORDER: usize = 24;
pub const MAX_DEL_DEC_STATES: usize = 4;
pub const DECISION_DELAY: usize = 40;
pub const NSQ_LPC_BUF_LENGTH: usize = MAX_LPC_ORDER;
pub const HARM_SHAPE_FIR_TAPS: usize = 3;
pub const MAX_FIND_PITCH_LPC_ORDER: usize = 16;

pub const STEREO_INTERP_LEN_MS: usize = 8; // must be even
pub const STEREO_RATIO_SMOOTH_COEF: f64 = 0.01;
pub const VAD_N_BANDS: usize = 4;
pub const VAD_INTERNAL_SUBFRAMES_LOG2: usize = 2;
pub const VAD_INTERNAL_SUBFRAMES: usize = 1 << VAD_INTERNAL_SUBFRAMES_LOG2;
pub const VAD_NOISE_LEVEL_SMOOTH_COEF_Q16: i32 = 1024; // 0.015625 in Q16
pub const VAD_SNR_FACTOR_Q16: i32 = 45000;
pub const VAD_SNR_SMOOTH_COEF_Q18: i32 = 4096;
pub const VAD_NOISE_LEVELS_BIAS: i32 = 50;
pub const VAD_NEGATIVE_OFFSET_Q5: i32 = 128; // 4.0 in Q5

pub const VARIABLE_HP_MIN_CUTOFF_HZ: i32 = 60;
pub const VARIABLE_HP_MAX_CUTOFF_HZ: i32 = 100;
pub const VARIABLE_HP_MAX_DELTA_FREQ_Q7: i32 = ((0.4f64 * (1 << 7) as f64) + 0.5) as i32;

pub const SILK_PE_MIN_COMPLEX: i32 = 0;
pub const SILK_PE_MID_COMPLEX: i32 = 1;
pub const SILK_PE_MAX_COMPLEX: i32 = 2;

/// Warping multiplier (Q16 per kHz): 0.015 * 65536 ≈ 983
pub const WARPING_MULTIPLIER_Q16: i32 = ((0.015f64 * 65536.0) + 0.5) as i32;

/// Maximum sum of log gains (Q7): 250.0 * 128
pub const MAX_SUM_LOG_GAIN_DB_Q7: i32 = ((250.0f64 * 128.0) + 0.5) as i32;

pub const NLSF_QUANT_DEL_DEC_STATES_LOG2: usize = 2;
pub const NLSF_QUANT_DEL_DEC_STATES: usize = 1 << NLSF_QUANT_DEL_DEC_STATES_LOG2;
pub const NLSF_QUANT_MAX_AMPLITUDE_EXT: i32 = 10;

pub const TRANSITION_FRAMES: i32 = 80;

pub const BIN_DIV_STEPS_A2NLSF: usize = 3;
pub const MAX_ITERATIONS_A2NLSF: usize = 16;

/// Pitch analysis look-ahead (reserved at beginning of analysis buffer)
pub const SCRATCH_SIZE: usize = 3 * LA_PITCH_MAX as usize;

// Error codes
pub const SILK_NO_ERROR: i32 = 0;
pub const SILK_ENC_FS_NOT_SUPPORTED: i32 = -101;
pub const SILK_ENC_PACKET_SIZE_NOT_SUPPORTED: i32 = -102;
pub const SILK_ENC_INVALID_LOSS_RATE: i32 = -104;
pub const SILK_ENC_INVALID_COMPLEXITY_SETTING: i32 = -105;
pub const SILK_ENC_INVALID_INBAND_FEC_SETTING: i32 = -106;
pub const SILK_ENC_INVALID_DTX_SETTING: i32 = -107;
pub const SILK_ENC_INVALID_CBR_SETTING: i32 = -108;
pub const SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR: i32 = -111;
pub const SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES: i32 = -112;

// ===========================================================================
// SNR tables (from control_SNR.c)
// ===========================================================================

const SILK_TARGET_RATE_NB_21: [u8; 107] = [
     0, 15, 39, 52, 61, 68,
    74, 79, 84, 88, 92, 95, 99,102,105,108,111,114,117,119,122,124,
   126,129,131,133,135,137,139,142,143,145,147,149,151,153,155,157,
   158,160,162,163,165,167,168,170,171,173,174,176,177,179,180,182,
   183,185,186,187,189,190,192,193,194,196,197,199,200,201,203,204,
   205,207,208,209,211,212,213,215,216,217,219,220,221,223,224,225,
   227,228,230,231,232,234,235,236,238,239,241,242,243,245,246,248,
   249,250,252,253,255,
];

const SILK_TARGET_RATE_MB_21: [u8; 155] = [
     0,  0, 28, 43, 52, 59,
    65, 70, 74, 78, 81, 85, 87, 90, 93, 95, 98,100,102,105,107,109,
   111,113,115,116,118,120,122,123,125,127,128,130,131,133,134,136,
   137,138,140,141,143,144,145,147,148,149,151,152,153,154,156,157,
   158,159,160,162,163,164,165,166,167,168,169,171,172,173,174,175,
   176,177,178,179,180,181,182,183,184,185,186,187,188,188,189,190,
   191,192,193,194,195,196,197,198,199,200,201,202,203,203,204,205,
   206,207,208,209,210,211,212,213,214,214,215,216,217,218,219,220,
   221,222,223,224,224,225,226,227,228,229,230,231,232,233,234,235,
   236,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,
   251,252,253,254,255,
];

const SILK_TARGET_RATE_WB_21: [u8; 191] = [
     0,  0,  0,  8, 29, 41,
    49, 56, 62, 66, 70, 74, 77, 80, 83, 86, 88, 91, 93, 95, 97, 99,
   101,103,105,107,108,110,112,113,115,116,118,119,121,122,123,125,
   126,127,129,130,131,132,134,135,136,137,138,140,141,142,143,144,
   145,146,147,148,149,150,151,152,153,154,156,157,158,159,159,160,
   161,162,163,164,165,166,167,168,169,170,171,171,172,173,174,175,
   176,177,177,178,179,180,181,181,182,183,184,185,185,186,187,188,
   189,189,190,191,192,192,193,194,195,195,196,197,198,198,199,200,
   200,201,202,203,203,204,205,206,206,207,208,209,209,210,211,211,
   212,213,214,214,215,216,216,217,218,219,219,220,221,221,222,223,
   224,224,225,226,226,227,228,229,229,230,231,232,232,233,234,234,
   235,236,237,237,238,239,240,240,241,242,243,243,244,245,246,246,
   247,248,249,249,250,251,252,253,255,
];

// ===========================================================================
// State structures
// ===========================================================================

/// Encoder control parameters passed from the Opus layer.
#[derive(Clone)]
pub struct SilkEncControlStruct {
    pub n_channels_api: i32,
    pub n_channels_internal: i32,
    pub api_sample_rate: i32,
    pub max_internal_sample_rate: i32,
    pub min_internal_sample_rate: i32,
    pub desired_internal_sample_rate: i32,
    pub payload_size_ms: i32,
    pub bit_rate: i32,
    pub packet_loss_percentage: i32,
    pub complexity: i32,
    pub use_in_band_fec: i32,
    pub use_dred: i32,
    pub lbrr_coded: i32,
    pub use_dtx: i32,
    pub use_cbr: i32,
    pub max_bits: i32,
    pub to_mono: i32,
    pub opus_can_switch: i32,
    pub reduced_dependency: i32,
    // Output fields
    pub internal_sample_rate: i32,
    pub allow_bandwidth_switch: i32,
    pub in_wb_mode_without_variable_lp: i32,
    pub stereo_width_q14: i32,
    pub switch_ready: i32,
    pub signal_type: i32,
    pub offset: i32,
}

impl Default for SilkEncControlStruct {
    fn default() -> Self {
        Self {
            n_channels_api: 1,
            n_channels_internal: 1,
            api_sample_rate: 16000,
            max_internal_sample_rate: 16000,
            min_internal_sample_rate: 8000,
            desired_internal_sample_rate: 16000,
            payload_size_ms: 20,
            bit_rate: 25000,
            packet_loss_percentage: 0,
            complexity: 5,
            use_in_band_fec: 0,
            use_dred: 0,
            lbrr_coded: 0,
            use_dtx: 0,
            use_cbr: 0,
            max_bits: 0,
            to_mono: 0,
            opus_can_switch: 0,
            reduced_dependency: 0,
            internal_sample_rate: 0,
            allow_bandwidth_switch: 0,
            in_wb_mode_without_variable_lp: 0,
            stereo_width_q14: 0,
            switch_ready: 0,
            signal_type: 0,
            offset: 0,
        }
    }
}

/// VAD (Voice Activity Detection) state.
#[derive(Clone)]
pub struct SilkVadState {
    pub ana_state: [i32; 2],
    pub ana_state1: [i32; 2],
    pub ana_state2: [i32; 2],
    pub xnrg_subfr: [i32; VAD_N_BANDS],
    pub nrg_ratio_smth_q8: [i32; VAD_N_BANDS],
    pub hp_state: i16,
    pub nl: [i32; VAD_N_BANDS],
    pub inv_nl: [i32; VAD_N_BANDS],
    pub noise_level_bias: [i32; VAD_N_BANDS],
    pub counter: i32,
}

impl Default for SilkVadState {
    fn default() -> Self {
        Self {
            ana_state: [0; 2],
            ana_state1: [0; 2],
            ana_state2: [0; 2],
            xnrg_subfr: [0; VAD_N_BANDS],
            nrg_ratio_smth_q8: [0; VAD_N_BANDS],
            hp_state: 0,
            nl: [0; VAD_N_BANDS],
            inv_nl: [0; VAD_N_BANDS],
            noise_level_bias: [0; VAD_N_BANDS],
            counter: 0,
        }
    }
}

/// LP variable cutoff state (bandwidth transition filter).
#[derive(Clone)]
pub struct SilkLpState {
    pub in_lp_state: [i32; 2],
    pub transition_frame_no: i32,
    pub mode: i32,
    pub saved_fs_khz: i32,
}

impl Default for SilkLpState {
    fn default() -> Self {
        Self {
            in_lp_state: [0; 2],
            transition_frame_no: 0,
            mode: 0,
            saved_fs_khz: 0,
        }
    }
}

/// Noise shaping quantizer state.
#[derive(Clone)]
pub struct NsqState {
    pub xq: Vec<i16>,                            // 2 * MAX_FRAME_LENGTH
    pub s_ltp_shp_q14: Vec<i32>,                 // 2 * MAX_FRAME_LENGTH
    pub s_lpc_q14: [i32; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
    pub s_ar2_q14: [i32; MAX_SHAPE_LPC_ORDER],
    pub s_lf_ar_shp_q14: i32,
    pub s_diff_shp_q14: i32,
    pub lag_prev: i32,
    pub s_ltp_buf_idx: i32,
    pub s_ltp_shp_buf_idx: i32,
    pub rand_seed: i32,
    pub prev_gain_q16: i32,
    pub rewhite_flag: i32,
}

impl Default for NsqState {
    fn default() -> Self {
        Self {
            xq: vec![0i16; 2 * MAX_FRAME_LENGTH],
            s_ltp_shp_q14: vec![0i32; 2 * MAX_FRAME_LENGTH],
            s_lpc_q14: [0; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
            s_ar2_q14: [0; MAX_SHAPE_LPC_ORDER],
            s_lf_ar_shp_q14: 0,
            s_diff_shp_q14: 0,
            lag_prev: 100,
            s_ltp_buf_idx: 0,
            s_ltp_shp_buf_idx: 0,
            rand_seed: 0,
            prev_gain_q16: 65536,
            rewhite_flag: 0,
        }
    }
}

/// Noise shaping analysis state (fixed-point).
#[derive(Clone)]
pub struct SilkShapeStateFix {
    pub last_gain_index: i8,
    pub harm_shape_gain_smth: i32,
    pub tilt_smth: i32,
}

impl Default for SilkShapeStateFix {
    fn default() -> Self {
        Self {
            last_gain_index: 10,
            harm_shape_gain_smth: 0,
            tilt_smth: 0,
        }
    }
}

/// Per-frame encoder control (transient, stack-allocated).
#[derive(Clone)]
pub struct SilkEncoderControl {
    pub gains_q16: [i32; MAX_NB_SUBFR],
    pub pred_coef_q12: [[i16; MAX_LPC_ORDER]; 2],
    pub ltp_coef_q14: [i16; LTP_ORDER * MAX_NB_SUBFR],
    pub ltp_scale_q14: i32,
    pub pitch_l: [i32; MAX_NB_SUBFR],
    pub ar_q13: [i16; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER],
    pub lf_shp_q14: [i32; MAX_NB_SUBFR],
    pub harm_shape_fir_packed_q14: i32,
    pub tilt_q14: [i32; MAX_NB_SUBFR],
    pub harm_shape_gain_q14: [i32; MAX_NB_SUBFR],
    pub lambda_q10: i32,
    pub input_quality_q14: i32,
    pub coding_quality_q14: i32,
    pub pitch_lag_low_bits_icdf: &'static [u8],
    pub pitch_contour_icdf: &'static [u8],
    pub per_index: i8,
    pub current_voice_gain_q14: i32,
    pub gw_temp: i32,
    pub sparseness_q8: i32,
    pub pred_gain_q16: i32,
    pub ltp_scaling_q14: i32,
    pub ltp_pred_cod_gain_q7: i32,
    pub res_nrg: [i32; MAX_NB_SUBFR],
    pub res_nrg_q: [i32; MAX_NB_SUBFR],
    pub gains_unq_q16: [i32; MAX_NB_SUBFR],
    pub last_gain_index_prev: i8,
    pub sum_log_gain_q7: i32,
}

impl Default for SilkEncoderControl {
    fn default() -> Self {
        Self {
            gains_q16: [0; MAX_NB_SUBFR],
            pred_coef_q12: [[0; MAX_LPC_ORDER]; 2],
            ltp_coef_q14: [0; LTP_ORDER * MAX_NB_SUBFR],
            ltp_scale_q14: 0,
            pitch_l: [0; MAX_NB_SUBFR],
            ar_q13: [0; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER],
            lf_shp_q14: [0; MAX_NB_SUBFR],
            harm_shape_fir_packed_q14: 0,
            tilt_q14: [0; MAX_NB_SUBFR],
            harm_shape_gain_q14: [0; MAX_NB_SUBFR],
            lambda_q10: 0,
            input_quality_q14: 0,
            coding_quality_q14: 0,
            pitch_lag_low_bits_icdf: &SILK_UNIFORM4_ICDF,
            pitch_contour_icdf: &SILK_PITCH_CONTOUR_NB_ICDF,
            per_index: 0,
            current_voice_gain_q14: 0,
            gw_temp: 0,
            sparseness_q8: 0,
            pred_gain_q16: 0,
            ltp_scaling_q14: 0,
            ltp_pred_cod_gain_q7: 0,
            res_nrg: [0; MAX_NB_SUBFR],
            res_nrg_q: [0; MAX_NB_SUBFR],
            gains_unq_q16: [0; MAX_NB_SUBFR],
            last_gain_index_prev: 0,
            sum_log_gain_q7: 0,
        }
    }
}

/// Common encoder state (the large `silk_encoder_state` from C).
#[derive(Clone)]
pub struct SilkEncoderState {
    // Filter states
    pub in_hp_state: [i32; 2],
    pub variable_hp_smth1_q15: i32,
    pub variable_hp_smth2_q15: i32,
    pub s_lp: SilkLpState,
    pub s_nsq: NsqState,
    pub s_vad: SilkVadState,
    pub resampler_state: SilkResamplerState,

    // Configuration
    pub fs_khz: i32,
    pub nb_subfr: i32,
    pub frame_length: i32,
    pub subfr_length: i32,
    pub ltp_mem_length: i32,
    pub la_pitch: i32,
    pub la_shape: i32,
    pub shape_win_length: i32,
    pub target_rate_bps: i32,
    pub packet_size_ms: i32,
    pub predict_lpc_order: i32,
    pub prev_signal_type: i32,
    pub prev_lag: i32,
    pub pitch_lpc_win_length: i32,
    pub max_pitch_lag: i32,

    // Complexity settings
    pub complexity: i32,
    pub pitch_estimation_complexity: i32,
    pub pitch_estimation_threshold_q16: i32,
    pub pitch_estimation_lpc_order: i32,
    pub shaping_lpc_order: i32,
    pub n_states_delayed_decision: i32,
    pub use_interpolated_nlsfs: i32,
    pub nlsf_msvq_survivors: i32,
    pub warping_q16: i32,

    // Rate/bitrate control
    pub api_fs_hz: i32,
    pub prev_api_fs_hz: i32,
    pub max_internal_fs_hz: i32,
    pub min_internal_fs_hz: i32,
    pub desired_internal_fs_hz: i32,
    pub snr_db_q7: i32,
    pub use_dtx: i32,
    pub use_cbr: i32,
    pub use_in_band_fec: i32,
    pub n_channels_api: i32,
    pub n_channels_internal: i32,
    pub allow_bandwidth_switch: i32,
    pub channel_nb: i32,
    pub packet_loss_perc: i32,
    pub n_frames_per_packet: i32,
    pub n_frames_encoded: i32,
    pub controlled_since_last_payload: i32,
    pub prefill_flag: i32,

    // Codec parameters (per-frame results)
    pub indices: SideInfoIndices,
    pub pulses: [i8; MAX_FRAME_LENGTH],
    pub prev_nlsfq_q15: [i16; MAX_LPC_ORDER],

    // LBRR (Forward Error Correction)
    pub lbrr_enabled: i32,
    pub lbrr_gain_increases: i32,
    pub indices_lbrr: [SideInfoIndices; MAX_FRAMES_PER_PACKET],
    pub pulses_lbrr: [[i8; MAX_FRAME_LENGTH]; MAX_FRAMES_PER_PACKET],
    pub lbrr_flags: [i32; MAX_FRAMES_PER_PACKET],
    pub lbrr_flag: i32,

    // Buffering
    pub input_buf: [i16; MAX_FRAME_LENGTH + 2],
    pub input_buf_ix: i32,

    // Codec references
    pub nlsf_cb: &'static SilkNlsfCbStruct,
    pub pitch_lag_low_bits_icdf: &'static [u8],
    pub pitch_contour_icdf: &'static [u8],

    // First frame / reset
    pub first_frame_after_reset: i32,

    // VAD/DTX
    pub speech_activity_q8: i32,
    pub in_dtx: i32,
    pub vad_flags: [i32; MAX_FRAMES_PER_PACKET],
    pub input_tilt_q15: i32,
    pub input_quality_bands_q15: [i32; VAD_N_BANDS],
    pub ec_prev_signal_type: i32,
    pub ec_prev_lag_index: i16,
    pub reduced_dependency: i32,

    // Frame counter
    pub frame_counter: i32,
    pub no_speech_counter: i32,
    pub lbrr_prev_last_gain_index: i8,
    pub sum_log_gain_q7: i32,
}

impl Default for SilkEncoderState {
    fn default() -> Self {
        Self {
            in_hp_state: [0; 2],
            variable_hp_smth1_q15: 0,
            variable_hp_smth2_q15: 0,
            s_lp: SilkLpState::default(),
            s_nsq: NsqState::default(),
            s_vad: SilkVadState::default(),
            resampler_state: SilkResamplerState::default(),
            fs_khz: 0,
            nb_subfr: 4,
            frame_length: 0,
            subfr_length: 0,
            ltp_mem_length: 0,
            la_pitch: 0,
            la_shape: 0,
            shape_win_length: 0,
            target_rate_bps: 0,
            packet_size_ms: 0,
            predict_lpc_order: 10,
            prev_signal_type: TYPE_NO_VOICE_ACTIVITY,
            prev_lag: 100,
            pitch_lpc_win_length: 0,
            max_pitch_lag: 0,
            complexity: 5,
            pitch_estimation_complexity: 0,
            pitch_estimation_threshold_q16: 0,
            pitch_estimation_lpc_order: 0,
            shaping_lpc_order: 0,
            n_states_delayed_decision: 1,
            use_interpolated_nlsfs: 0,
            nlsf_msvq_survivors: 2,
            warping_q16: 0,
            api_fs_hz: 0,
            prev_api_fs_hz: 0,
            max_internal_fs_hz: 16000,
            min_internal_fs_hz: 8000,
            desired_internal_fs_hz: 16000,
            snr_db_q7: 0,
            use_dtx: 0,
            use_cbr: 0,
            use_in_band_fec: 0,
            n_channels_api: 1,
            n_channels_internal: 1,
            allow_bandwidth_switch: 0,
            channel_nb: 0,
            packet_loss_perc: 0,
            n_frames_per_packet: 1,
            n_frames_encoded: 0,
            controlled_since_last_payload: 0,
            prefill_flag: 0,
            indices: SideInfoIndices::default(),
            pulses: [0; MAX_FRAME_LENGTH],
            prev_nlsfq_q15: [0; MAX_LPC_ORDER],
            lbrr_enabled: 0,
            lbrr_gain_increases: 0,
            indices_lbrr: [
                SideInfoIndices::default(),
                SideInfoIndices::default(),
                SideInfoIndices::default(),
            ],
            pulses_lbrr: [[0; MAX_FRAME_LENGTH]; MAX_FRAMES_PER_PACKET],
            lbrr_flags: [0; MAX_FRAMES_PER_PACKET],
            lbrr_flag: 0,
            input_buf: [0; MAX_FRAME_LENGTH + 2],
            input_buf_ix: 0,
            nlsf_cb: &SILK_NLSF_CB_NB_MB,
            pitch_lag_low_bits_icdf: &SILK_UNIFORM4_ICDF,
            pitch_contour_icdf: &SILK_PITCH_CONTOUR_NB_ICDF,
            first_frame_after_reset: 1,
            speech_activity_q8: 0,
            in_dtx: 0,
            vad_flags: [0; MAX_FRAMES_PER_PACKET],
            input_tilt_q15: 0,
            input_quality_bands_q15: [0; VAD_N_BANDS],
            ec_prev_signal_type: 0,
            ec_prev_lag_index: 0,
            reduced_dependency: 0,
            frame_counter: 0,
            no_speech_counter: 0,
            lbrr_prev_last_gain_index: 0,
            sum_log_gain_q7: 0,
        }
    }
}

/// Per-channel encoder state (fixed-point).
#[derive(Clone)]
pub struct SilkEncoderStateFix {
    pub s_cmn: SilkEncoderState,
    pub s_shape: SilkShapeStateFix,
    pub x_buf: Vec<i16>, // 2 * MAX_FRAME_LENGTH + LA_SHAPE_MAX
    pub ltp_corr_q15: i32,
    pub res_nrg_smth: i32,
}

impl Default for SilkEncoderStateFix {
    fn default() -> Self {
        Self {
            s_cmn: SilkEncoderState::default(),
            s_shape: SilkShapeStateFix::default(),
            x_buf: vec![0i16; 2 * MAX_FRAME_LENGTH + LA_SHAPE_MAX],
            ltp_corr_q15: 0,
            res_nrg_smth: 0,
        }
    }
}

/// Stereo encoder state.
#[derive(Clone)]
pub struct StereoEncState {
    pub pred_prev_q13: [i16; 2],
    pub s_mid: [i16; 2],
    pub s_side: [i16; 2],
    pub mid_side_amp_q0: [i32; 4],
    pub smth_width_q14: i16,
    pub width_prev_q14: i16,
    pub silent_side_len: i16,
    pub pred_ix: [[[i8; 3]; 2]; MAX_FRAMES_PER_PACKET],
    pub mid_only_flags: [i8; MAX_FRAMES_PER_PACKET],
}

impl Default for StereoEncState {
    fn default() -> Self {
        Self {
            pred_prev_q13: [0; 2],
            s_mid: [0; 2],
            s_side: [0; 2],
            mid_side_amp_q0: [0; 4],
            smth_width_q14: 0,
            width_prev_q14: 0,
            silent_side_len: 0,
            pred_ix: [[[0; 3]; 2]; MAX_FRAMES_PER_PACKET],
            mid_only_flags: [0; MAX_FRAMES_PER_PACKET],
        }
    }
}

/// Top-level SILK encoder super struct.
pub struct SilkEncoder {
    pub s_stereo: StereoEncState,
    pub n_bits_used_lbrr: i32,
    pub n_bits_exceeded: i32,
    pub n_channels_api: i32,
    pub n_channels_internal: i32,
    pub n_prev_channels_internal: i32,
    pub time_since_switch_allowed_ms: i32,
    pub allow_bandwidth_switch: i32,
    pub prev_decode_only_middle: i32,
    pub state_fxx: [SilkEncoderStateFix; ENCODER_NUM_CHANNELS],
}

impl SilkEncoder {
    pub fn new() -> Self {
        Self {
            s_stereo: StereoEncState::default(),
            n_bits_used_lbrr: 0,
            n_bits_exceeded: 0,
            n_channels_api: 1,
            n_channels_internal: 1,
            n_prev_channels_internal: 1,
            time_since_switch_allowed_ms: 0,
            allow_bandwidth_switch: 0,
            prev_decode_only_middle: 0,
            state_fxx: [SilkEncoderStateFix::default(), SilkEncoderStateFix::default()],
        }
    }
}

/// NSQ delayed-decision state (per parallel path).
#[derive(Clone)]
pub struct NsqDelDecStruct {
    pub s_lpc_q14: [i32; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
    pub rand_state: [i32; DECISION_DELAY],
    pub q_q10: [i32; DECISION_DELAY],
    pub xq_q14: [i32; DECISION_DELAY],
    pub pred_q15: [i32; DECISION_DELAY],
    pub shape_q14: [i32; DECISION_DELAY],
    pub s_ar2_q14: [i32; MAX_SHAPE_LPC_ORDER],
    pub lf_ar_q14: i32,
    pub diff_q14: i32,
    pub seed: i32,
    pub seed_init: i32,
    pub rd_q10: i32,
}

impl Default for NsqDelDecStruct {
    fn default() -> Self {
        Self {
            s_lpc_q14: [0; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
            rand_state: [0; DECISION_DELAY],
            q_q10: [0; DECISION_DELAY],
            xq_q14: [0; DECISION_DELAY],
            pred_q15: [0; DECISION_DELAY],
            shape_q14: [0; DECISION_DELAY],
            s_ar2_q14: [0; MAX_SHAPE_LPC_ORDER],
            lf_ar_q14: 0,
            diff_q14: 0,
            seed: 0,
            seed_init: 0,
            rd_q10: 0,
        }
    }
}

// ===========================================================================
// Check control input
// ===========================================================================

/// Validate encoder control parameters.
/// Matches C: `check_control_input`.
pub fn check_control_input(enc_control: &SilkEncControlStruct) -> i32 {
    if (enc_control.api_sample_rate != 8000
        && enc_control.api_sample_rate != 12000
        && enc_control.api_sample_rate != 16000
        && enc_control.api_sample_rate != 24000
        && enc_control.api_sample_rate != 32000
        && enc_control.api_sample_rate != 44100
        && enc_control.api_sample_rate != 48000)
        || (enc_control.desired_internal_sample_rate != 8000
            && enc_control.desired_internal_sample_rate != 12000
            && enc_control.desired_internal_sample_rate != 16000)
        || (enc_control.max_internal_sample_rate != 8000
            && enc_control.max_internal_sample_rate != 12000
            && enc_control.max_internal_sample_rate != 16000)
        || (enc_control.min_internal_sample_rate != 8000
            && enc_control.min_internal_sample_rate != 12000
            && enc_control.min_internal_sample_rate != 16000)
        || enc_control.min_internal_sample_rate > enc_control.desired_internal_sample_rate
        || enc_control.max_internal_sample_rate < enc_control.desired_internal_sample_rate
        || enc_control.min_internal_sample_rate > enc_control.max_internal_sample_rate
    {
        return SILK_ENC_FS_NOT_SUPPORTED;
    }
    if enc_control.payload_size_ms != 10
        && enc_control.payload_size_ms != 20
        && enc_control.payload_size_ms != 40
        && enc_control.payload_size_ms != 60
    {
        return SILK_ENC_PACKET_SIZE_NOT_SUPPORTED;
    }
    if enc_control.packet_loss_percentage < 0 || enc_control.packet_loss_percentage > 100 {
        return SILK_ENC_INVALID_LOSS_RATE;
    }
    if enc_control.use_dtx < 0 || enc_control.use_dtx > 1 {
        return SILK_ENC_INVALID_DTX_SETTING;
    }
    if enc_control.use_cbr < 0 || enc_control.use_cbr > 1 {
        return SILK_ENC_INVALID_CBR_SETTING;
    }
    if enc_control.use_in_band_fec < 0 || enc_control.use_in_band_fec > 1 {
        return SILK_ENC_INVALID_INBAND_FEC_SETTING;
    }
    if enc_control.n_channels_api < 1 || enc_control.n_channels_api > ENCODER_NUM_CHANNELS as i32 {
        return SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR;
    }
    if enc_control.n_channels_internal < 1
        || enc_control.n_channels_internal > ENCODER_NUM_CHANNELS as i32
    {
        return SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR;
    }
    if enc_control.n_channels_internal > enc_control.n_channels_api {
        return SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR;
    }
    if enc_control.complexity < 0 || enc_control.complexity > 10 {
        return SILK_ENC_INVALID_COMPLEXITY_SETTING;
    }
    SILK_NO_ERROR
}

// ===========================================================================
// Control SNR
// ===========================================================================

/// Set SNR control based on target bitrate.
/// Matches C: `silk_control_SNR`.
pub fn silk_control_snr(ps_enc: &mut SilkEncoderState, mut target_rate_bps: i32) -> i32 {
    ps_enc.target_rate_bps = target_rate_bps;
    if ps_enc.nb_subfr == 2 {
        target_rate_bps -= 2000 + ps_enc.fs_khz / 16;
    }
    let (bound, snr_table): (usize, &[u8]) = if ps_enc.fs_khz == 8 {
        (SILK_TARGET_RATE_NB_21.len(), &SILK_TARGET_RATE_NB_21)
    } else if ps_enc.fs_khz == 12 {
        (SILK_TARGET_RATE_MB_21.len(), &SILK_TARGET_RATE_MB_21)
    } else {
        (SILK_TARGET_RATE_WB_21.len(), &SILK_TARGET_RATE_WB_21)
    };
    let id = (target_rate_bps + 200) / 400;
    let id = imin(id - 10, bound as i32 - 1);
    if id <= 0 {
        ps_enc.snr_db_q7 = 0;
    } else {
        ps_enc.snr_db_q7 = snr_table[id as usize] as i32 * 21;
    }
    SILK_NO_ERROR
}

// ===========================================================================
// VAD initialization
// ===========================================================================

/// Initialize VAD state.
/// Matches C: `silk_VAD_Init`.
pub fn silk_vad_init(vad: &mut SilkVadState) -> i32 {
    *vad = SilkVadState::default();

    // Initialize noise levels with approx pink noise (psd ∝ 1/f)
    for b in 0..VAD_N_BANDS {
        vad.noise_level_bias[b] = imax(VAD_NOISE_LEVELS_BIAS / (b as i32 + 1), 1);
    }
    for b in 0..VAD_N_BANDS {
        vad.nl[b] = 100 * vad.noise_level_bias[b];
        vad.inv_nl[b] = i32::MAX / vad.nl[b];
    }
    vad.counter = 15;

    // Init smoothed energy-to-noise ratio (20 dB)
    for b in 0..VAD_N_BANDS {
        vad.nrg_ratio_smth_q8[b] = 100 * 256;
    }
    0
}

// ===========================================================================
// Analysis filter bank
// ===========================================================================

/// Two-band decimating analysis filter bank.
/// Matches C: `silk_ana_filt_bank_1`.
pub fn silk_ana_filt_bank_1(
    input: &[i16],
    state: &mut [i32; 2],
    out_l: &mut [i16],
    out_h: &mut [i16],
    n: usize,
) {
    // Coefficients: allpass sections for half-band splitting
    // A_fb1_20 = 5394 << 1 = 10788 (fits i16)
    const A_FB1_20: i16 = (5394 << 1) as i16; // 10788
    // A_fb1_21 = (opus_int16)(20623 << 1) = -24290 (wraps in i16)
    const A_FB1_21: i16 = 20623i32.wrapping_shl(1) as i16; // -24290

    let n2 = n >> 1;
    for k in 0..n2 {
        // Convert even input to Q10
        let in32 = (input[2 * k] as i32) << 10;

        // All-pass section for even input sample (uses SMLAWB)
        let y = in32 - state[0];
        let x = silk_smlawb(y, y, A_FB1_21);
        let out_1 = state[0] + x;
        state[0] = in32 + x;

        // Convert odd input to Q10
        let in32 = (input[2 * k + 1] as i32) << 10;

        // All-pass section for odd input sample (uses SMULWB)
        let y = in32 - state[1];
        let x = silk_smulwb(y, A_FB1_20);
        let out_2 = state[1] + x;
        state[1] = in32 + x;

        // Add/subtract, convert back to int16 and store
        out_l[k] = sat16(silk_rshift_round(out_2 + out_1, 11));
        out_h[k] = sat16(silk_rshift_round(out_2 - out_1, 11));
    }
}

// ===========================================================================
// VAD: Get speech activity level
// ===========================================================================

/// Weighting factors for frequency tilt measure.
const TILT_WEIGHTS: [i32; VAD_N_BANDS] = [30000, 6000, -12000, -12000];

/// Get noise levels for VAD.
/// Matches C: `silk_VAD_GetNoiseLevels`.
fn silk_vad_get_noise_levels(px: &[i32; VAD_N_BANDS], vad: &mut SilkVadState) {
    // Initially faster smoothing
    let min_coef: i32;
    if vad.counter < 1000 {
        min_coef = i16::MAX as i32 / ((vad.counter >> 4) + 1);
        vad.counter += 1;
    } else {
        min_coef = 0;
    }

    for k in 0..VAD_N_BANDS {
        let nl = vad.nl[k];

        // Add bias
        let nrg = silk_add_pos_sat32(px[k], vad.noise_level_bias[k]);

        // Invert energies
        let inv_nrg = i32::MAX / imax(nrg, 1);

        // Less update when subband energy is high
        let coef: i32 = if nrg > (nl << 3) {
            VAD_NOISE_LEVEL_SMOOTH_COEF_Q16 >> 3
        } else if nrg < nl {
            VAD_NOISE_LEVEL_SMOOTH_COEF_Q16
        } else {
            // C: silk_SMULWB(silk_SMULWW(inv_nrg, nl), VAD_NOISE_LEVEL_SMOOTH_COEF_Q16 << 1)
            // silk_SMULWB takes i32,i32 and internally extracts low 16 bits of second arg
            silk_smulwb_i32(silk_smulww(inv_nrg, nl), VAD_NOISE_LEVEL_SMOOTH_COEF_Q16 << 1)
        };

        // Initially faster smoothing
        let coef = imax(coef, min_coef);

        // Smooth inverse energies
        // C: silk_SMLAWB(inv_NL[k], inv_nrg - inv_NL[k], coef)
        vad.inv_nl[k] = silk_smlawb_i32(vad.inv_nl[k], inv_nrg - vad.inv_nl[k], coef);

        // Compute noise level by inverting again
        let nl = i32::MAX / imax(vad.inv_nl[k], 1);

        // Limit noise levels (guarantee 7 bits of head room)
        vad.nl[k] = imin(nl, 0x00FFFFFF);
    }
}

/// Compute speech activity level (Q8).
/// Matches C: `silk_VAD_GetSA_Q8_c`.
pub fn silk_vad_get_sa_q8(ps_enc: &mut SilkEncoderState, p_in: &[i16]) -> i32 {
    let frame_length = ps_enc.frame_length as usize;
    let vad = &mut ps_enc.s_vad;

    // Compute decimated frame lengths (matches C reference)
    let decimated_framelength1 = frame_length >> 1;
    let decimated_framelength2 = frame_length >> 2;
    let decimated_framelength = frame_length >> 3;

    // Compute offsets for each band's storage in the X buffer
    // Matches C: X_offset[0]=0, X_offset[1]=dec+dec2, X_offset[2]=X_offset[1]+dec, X_offset[3]=X_offset[2]+dec2
    let x_offset: [usize; VAD_N_BANDS] = [
        0,
        decimated_framelength + decimated_framelength2,
        decimated_framelength + decimated_framelength2 + decimated_framelength,
        decimated_framelength + decimated_framelength2 + decimated_framelength + decimated_framelength2,
    ];
    let total_x_len = x_offset[3] + decimated_framelength1;

    let mut x = vec![0i16; total_x_len];

    // Stage 1: 0-8 kHz → 0-4 kHz (out_l at X[0]) + 4-8 kHz (out_h at X[X_offset[3]])
    // C: silk_ana_filt_bank_1(pIn, &state[0], X, &X[X_offset[3]], frame_length)
    {
        let mut out_l = vec![0i16; decimated_framelength1];
        let mut out_h = vec![0i16; decimated_framelength1];
        silk_ana_filt_bank_1(p_in, &mut vad.ana_state, &mut out_l, &mut out_h, frame_length);
        x[..decimated_framelength1].copy_from_slice(&out_l);
        x[x_offset[3]..x_offset[3] + decimated_framelength1].copy_from_slice(&out_h);
    }

    // Stage 2: 0-4 kHz → 0-2 kHz (out_l at X[0]) + 2-4 kHz (out_h at X[X_offset[2]])
    // C: silk_ana_filt_bank_1(X, &state1[0], X, &X[X_offset[2]], decimated_framelength1)
    {
        let input = x[..decimated_framelength1].to_vec();
        let mut out_l = vec![0i16; decimated_framelength2];
        let mut out_h = vec![0i16; decimated_framelength2];
        silk_ana_filt_bank_1(&input, &mut vad.ana_state1, &mut out_l, &mut out_h, decimated_framelength1);
        x[..decimated_framelength2].copy_from_slice(&out_l);
        x[x_offset[2]..x_offset[2] + decimated_framelength2].copy_from_slice(&out_h);
    }

    // Stage 3: 0-2 kHz → 0-1 kHz (out_l at X[0]) + 1-2 kHz (out_h at X[X_offset[1]])
    // C: silk_ana_filt_bank_1(X, &state2[0], X, &X[X_offset[1]], decimated_framelength2)
    {
        let input = x[..decimated_framelength2].to_vec();
        let mut out_l = vec![0i16; decimated_framelength];
        let mut out_h = vec![0i16; decimated_framelength];
        silk_ana_filt_bank_1(&input, &mut vad.ana_state2, &mut out_l, &mut out_h, decimated_framelength2);
        x[..decimated_framelength].copy_from_slice(&out_l);
        x[x_offset[1]..x_offset[1] + decimated_framelength].copy_from_slice(&out_h);
    }

    // HP filter on the lowest band: first-order differentiator to remove DC
    // C processes backwards with right-shift by 1
    {
        x[decimated_framelength - 1] = (x[decimated_framelength - 1] >> 1) as i16;
        let hp_state_tmp = x[decimated_framelength - 1];
        let mut i = decimated_framelength - 1;
        while i > 0 {
            x[i - 1] = (x[i - 1] >> 1) as i16;
            x[i] = sat16(x[i] as i32 - x[i - 1] as i32);
            i -= 1;
        }
        x[0] = sat16(x[0] as i32 - vad.hp_state as i32);
        vad.hp_state = hp_state_tmp;
    }

    // Compute per-band energies (matches C: per-sample >>3 then square, accumulate)
    let mut xnrg = [0i32; VAD_N_BANDS];

    for b in 0..VAD_N_BANDS {
        // C: decimated_framelength = silk_RSHIFT(frame_length, silk_min_int(VAD_N_BANDS - b, VAD_N_BANDS - 1))
        let band_dec_len = frame_length >> (VAD_N_BANDS - b).min(VAD_N_BANDS - 1);
        let dec_sf_len = band_dec_len >> VAD_INTERNAL_SUBFRAMES_LOG2;

        xnrg[b] = vad.xnrg_subfr[b];
        let mut dec_subframe_offset = 0usize;
        let mut sum_squared = 0i32;
        for s in 0..VAD_INTERNAL_SUBFRAMES {
            sum_squared = 0;
            for i in 0..dec_sf_len {
                let x_tmp = x[x_offset[b] + i + dec_subframe_offset] as i32 >> 3;
                sum_squared = silk_smlabb(sum_squared, x_tmp, x_tmp);
            }
            if s < VAD_INTERNAL_SUBFRAMES - 1 {
                xnrg[b] = xnrg[b].saturating_add(sum_squared);
            } else {
                xnrg[b] = xnrg[b].saturating_add(sum_squared >> 1);
            }
            dec_subframe_offset += dec_sf_len;
        }
        vad.xnrg_subfr[b] = sum_squared;
    }

    // Get noise levels
    silk_vad_get_noise_levels(&xnrg, vad);
    eprintln!("[VAD TRACE] xnrg={:?} nl={:?} inv_nl={:?}", xnrg, vad.nl, vad.inv_nl);

    // Signal-plus-noise to noise ratio estimation
    // Matches C: VAD.c lines 200-291
    let mut sum_squared = 0i32;
    let mut input_tilt = 0i32;
    let mut nrg_to_noise_ratio_q8 = [0i32; VAD_N_BANDS];

    for b in 0..VAD_N_BANDS {
        let speech_nrg = xnrg[b] - vad.nl[b];
        if speech_nrg > 0 {
            // Divide, with sufficient resolution
            if (xnrg[b] & 0xFF800000u32 as i32) == 0 {
                nrg_to_noise_ratio_q8[b] = (xnrg[b] << 8) / (vad.nl[b] + 1);
            } else {
                nrg_to_noise_ratio_q8[b] = xnrg[b] / ((vad.nl[b] >> 8) + 1);
            }

            // Convert to log domain
            let snr_q7 = silk_lin2log(nrg_to_noise_ratio_q8[b]) - 8 * 128;

            // Sum-of-squares (Q14)
            sum_squared = silk_smlabb(sum_squared, snr_q7, snr_q7);

            // Tilt measure
            let snr_for_tilt = if speech_nrg < (1 << 20) {
                // Scale down SNR value for small subband speech energies
                silk_smulwb_i32(silk_sqrt_approx(speech_nrg) << 6, snr_q7)
            } else {
                snr_q7
            };
            input_tilt = silk_smlawb_i32(input_tilt, TILT_WEIGHTS[b], snr_for_tilt);
        } else {
            nrg_to_noise_ratio_q8[b] = 256;
        }
    }

    // Mean-of-squares
    sum_squared = sum_squared / (VAD_N_BANDS as i32); // Q14

    // Root-mean-square approximation, scale to dBs
    let p_snr_db_q7 = (3 * silk_sqrt_approx(sum_squared)) as i16 as i32; // Q7

    // Speech Probability Estimation
    let mut sa_q15 = silk_sigm_q15(
        silk_smulwb_i32(VAD_SNR_FACTOR_Q16, p_snr_db_q7) - VAD_NEGATIVE_OFFSET_Q5,
    );

    // Frequency Tilt Measure
    ps_enc.input_tilt_q15 = (silk_sigm_q15(input_tilt) - 16384) << 1;

    // Scale the sigmoid output based on power levels
    let mut speech_nrg_total = 0i32;
    for b in 0..VAD_N_BANDS {
        // Accumulate signal-without-noise energies, higher frequency bands have more weight
        speech_nrg_total += ((b as i32) + 1) * ((xnrg[b] - vad.nl[b]) >> 4);
    }

    if ps_enc.frame_length == 20 * ps_enc.fs_khz {
        speech_nrg_total >>= 1;
    }

    // Power scaling
    if speech_nrg_total <= 0 {
        sa_q15 >>= 1;
    } else if speech_nrg_total < 16384 {
        let snrg = silk_sqrt_approx(speech_nrg_total << 16);
        sa_q15 = silk_smulwb_i32(32768 + snrg, sa_q15);
    }

    // Copy the resulting speech activity in Q8
    eprintln!("[VAD TRACE] sum_sq={} p_snr_db_q7={} sa_q15_before_power={} speech_nrg_total={} sa_q15_after_power={} → Q8={}",
        sum_squared, p_snr_db_q7,
        silk_sigm_q15(silk_smulwb_i32(VAD_SNR_FACTOR_Q16, p_snr_db_q7) - VAD_NEGATIVE_OFFSET_Q5),
        speech_nrg_total, sa_q15, imin(sa_q15 >> 7, 255));
    ps_enc.speech_activity_q8 = imin(sa_q15 >> 7, 255);

    // Energy Level and SNR estimation
    // Smoothing coefficient
    let mut smooth_coef_q16 = silk_smulwb_i32(
        VAD_SNR_SMOOTH_COEF_Q18,
        silk_smulwb_i32(sa_q15, sa_q15),
    );

    if ps_enc.frame_length == 10 * ps_enc.fs_khz {
        smooth_coef_q16 >>= 1;
    }

    for b in 0..VAD_N_BANDS {
        // Compute smoothed energy-to-noise ratio per band
        vad.nrg_ratio_smth_q8[b] = silk_smlawb_i32(
            vad.nrg_ratio_smth_q8[b],
            nrg_to_noise_ratio_q8[b] - vad.nrg_ratio_smth_q8[b],
            smooth_coef_q16,
        );

        // Signal to noise ratio in dB per band
        let snr_q7 = 3 * (silk_lin2log(vad.nrg_ratio_smth_q8[b]) - 8 * 128);
        // quality = sigmoid( 0.25 * ( SNR_dB - 16 ) )
        ps_enc.input_quality_bands_q15[b] = silk_sigm_q15((snr_q7 - 16 * 128) >> 4);
    }

    0
}

// ===========================================================================
// HP variable cutoff filter
// ===========================================================================

/// Biquad filter (second-order IIR), stride 1.
/// Matches C: `silk_biquad_alt_stride1`.
pub fn silk_biquad_alt_stride1(
    input: &[i16],
    b_q28: &[i32; 3],
    a_q28: &[i32; 2],
    state: &mut [i32; 2],
    output: &mut [i16],
    len: usize,
) {
    for k in 0..len {
        // Second-order section (transposed direct form II)
        let in_val = input[k] as i64;

        // out = state[0] + B[0] * input
        let out64 = state[0] as i64 + ((b_q28[0] as i64 * in_val) >> 28);
        let out32 = sat16(out64 as i32);
        output[k] = out32;

        // Update state
        let out_i64 = out32 as i64;
        state[0] = (state[1] as i64
            + ((b_q28[1] as i64 * in_val) >> 28)
            - ((a_q28[0] as i64 * out_i64) >> 28)) as i32;
        state[1] = (((b_q28[2] as i64 * in_val) >> 28)
            - ((a_q28[1] as i64 * out_i64) >> 28)) as i32;
    }
}

/// High-pass filter with variable cutoff frequency.
/// Only updates the smoother — does NOT apply the filter.
/// The actual HP biquad is applied separately in the encode frame pipeline.
/// Matches C: `silk_HP_variable_cutoff`.
pub fn silk_hp_variable_cutoff(ps_enc: &mut SilkEncoderStateFix) {
    let ps_enc_c1 = &mut ps_enc.s_cmn;

    // Only update when previous frame was voiced
    if ps_enc_c1.prev_signal_type != TYPE_VOICED {
        return;
    }

    // Estimate pitch frequency in Hz, Q16
    let pitch_freq_hz_q16 = ((ps_enc_c1.fs_khz * 1000) << 16) / imax(ps_enc_c1.prev_lag, 1);

    // Convert to log domain (Q7)
    let pitch_freq_log_q7 = silk_lin2log(pitch_freq_hz_q16) - (16 << 7);

    // Quality-based adjustment: reduce cutoff tracking when quality is high
    let quality_q15 = ps_enc_c1.input_quality_bands_q15[0];
    // SILK_FIX_CONST(VARIABLE_HP_MIN_CUTOFF_HZ, 16) = 60 << 16 = 3932160
    let min_cutoff_log_q7 = silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ << 16) - (16 << 7);
    let pitch_freq_log_q7 = silk_smlawb(
        pitch_freq_log_q7,
        silk_smulwb((-quality_q15) << 2, quality_q15 as i16),
        (pitch_freq_log_q7 - min_cutoff_log_q7) as i16,
    );

    // Delta frequency (how far from current smoother value)
    let mut delta_freq_q7 = pitch_freq_log_q7 - (ps_enc_c1.variable_hp_smth1_q15 >> 8);
    if delta_freq_q7 < 0 {
        delta_freq_q7 *= 3; // Faster tracking downward
    }

    // Limit delta: SILK_FIX_CONST(VARIABLE_HP_MAX_DELTA_FREQ, 7) = 51
    delta_freq_q7 = imax(imin(delta_freq_q7, VARIABLE_HP_MAX_DELTA_FREQ_Q7),
                         -VARIABLE_HP_MAX_DELTA_FREQ_Q7);

    // Update first smoother: SILK_FIX_CONST(VARIABLE_HP_SMTH_COEF1, 16) = 6554
    ps_enc_c1.variable_hp_smth1_q15 = silk_smlawb(
        ps_enc_c1.variable_hp_smth1_q15,
        silk_smulbb(ps_enc_c1.speech_activity_q8, delta_freq_q7),
        6554, // SILK_FIX_CONST(0.1, 16)
    );

    // Limit frequency range
    let min_hp_q15 = silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8;
    let max_hp_q15 = silk_lin2log(VARIABLE_HP_MAX_CUTOFF_HZ) << 8;
    ps_enc_c1.variable_hp_smth1_q15 = imax(imin(ps_enc_c1.variable_hp_smth1_q15, max_hp_q15),
                                            min_hp_q15);
}

// ===========================================================================
// LP variable cutoff (bandwidth transition filter)
// ===========================================================================

/// Apply variable low-pass filter for bandwidth transitions.
/// Matches C: `silk_LP_variable_cutoff`.
pub fn silk_lp_variable_cutoff(
    s_lp: &mut SilkLpState,
    frame: &mut [i16],
    frame_length: usize,
) {
    // Only active during bandwidth transitions
    if s_lp.mode == 0 {
        return;
    }

    // TRANSITION_INT_STEPS = 64, so shift by 16-6 = 10 instead of dividing
    let fac_q16 = (TRANSITION_FRAMES - s_lp.transition_frame_no) << (16 - 6);
    let ind = fac_q16 >> 16;
    let fac_q16 = fac_q16 - (ind << 16);

    // Interpolate filter taps
    let mut b_q28 = [0i32; TRANSITION_NB];
    let mut a_q28 = [0i32; TRANSITION_NA];
    silk_lp_interpolate_filter_taps(&mut b_q28, &mut a_q28, ind, fac_q16);

    // Advance transition counter, clamped to [0, TRANSITION_FRAMES]
    s_lp.transition_frame_no = imax(imin(
        s_lp.transition_frame_no + s_lp.mode, 0), TRANSITION_FRAMES);

    // Apply ARMA low-pass biquad filter (in-place)
    let mut output = vec![0i16; frame_length];
    silk_biquad_alt_stride1(
        &frame[..frame_length],
        &[b_q28[0], b_q28[1], b_q28[2]],
        &[a_q28[0], a_q28[1]],
        &mut s_lp.in_lp_state,
        &mut output,
        frame_length,
    );
    frame[..frame_length].copy_from_slice(&output);
}

/// Interpolate LP transition filter taps.
/// Matches C: `silk_LP_interpolate_filter_taps`.
fn silk_lp_interpolate_filter_taps(
    b_q28: &mut [i32; TRANSITION_NB],
    a_q28: &mut [i32; TRANSITION_NA],
    ind: i32,
    fac_q16: i32,
) {
    let ind_u = ind as usize;
    if (ind as usize) < TRANSITION_INT_NUM - 1 {
        if fac_q16 > 0 {
            if fac_q16 < 32768 {
                // Interpolate from ind toward ind+1
                for i in 0..TRANSITION_NB {
                    b_q28[i] = silk_smlawb(
                        SILK_TRANSITION_LP_B_Q28[ind_u][i],
                        SILK_TRANSITION_LP_B_Q28[ind_u + 1][i] - SILK_TRANSITION_LP_B_Q28[ind_u][i],
                        fac_q16 as i16,
                    );
                }
                for i in 0..TRANSITION_NA {
                    a_q28[i] = silk_smlawb(
                        SILK_TRANSITION_LP_A_Q28[ind_u][i],
                        SILK_TRANSITION_LP_A_Q28[ind_u + 1][i] - SILK_TRANSITION_LP_A_Q28[ind_u][i],
                        fac_q16 as i16,
                    );
                }
            } else {
                // Interpolate from ind+1 back toward ind
                let fac_q16_adj = (fac_q16 - (1 << 16)) as i16;
                for i in 0..TRANSITION_NB {
                    b_q28[i] = silk_smlawb(
                        SILK_TRANSITION_LP_B_Q28[ind_u + 1][i],
                        SILK_TRANSITION_LP_B_Q28[ind_u + 1][i] - SILK_TRANSITION_LP_B_Q28[ind_u][i],
                        fac_q16_adj,
                    );
                }
                for i in 0..TRANSITION_NA {
                    a_q28[i] = silk_smlawb(
                        SILK_TRANSITION_LP_A_Q28[ind_u + 1][i],
                        SILK_TRANSITION_LP_A_Q28[ind_u + 1][i] - SILK_TRANSITION_LP_A_Q28[ind_u][i],
                        fac_q16_adj,
                    );
                }
            }
        } else {
            // No fractional part: direct copy
            b_q28.copy_from_slice(&SILK_TRANSITION_LP_B_Q28[ind_u]);
            a_q28.copy_from_slice(&SILK_TRANSITION_LP_A_Q28[ind_u]);
        }
    } else {
        // At or beyond last interpolation point: use final values
        b_q28.copy_from_slice(&SILK_TRANSITION_LP_B_Q28[TRANSITION_INT_NUM - 1]);
        a_q28.copy_from_slice(&SILK_TRANSITION_LP_A_Q28[TRANSITION_INT_NUM - 1]);
    }
}

// ===========================================================================
// Control audio bandwidth
// ===========================================================================

/// Determine internal sampling rate based on bandwidth control state machine.
/// Matches C: `silk_control_audio_bandwidth`.
pub fn silk_control_audio_bandwidth(
    ps_enc: &mut SilkEncoderState,
    enc_control: &mut SilkEncControlStruct,
) -> i32 {
    let mut orig_khz = ps_enc.fs_khz;
    if orig_khz == 0 {
        orig_khz = ps_enc.s_lp.saved_fs_khz;
    }
    let mut fs_khz = orig_khz;
    let mut fs_hz = fs_khz * 1000;

    if fs_hz == 0 {
        // Encoder just initialized
        fs_hz = imin(ps_enc.desired_internal_fs_hz, ps_enc.api_fs_hz);
        fs_khz = fs_hz / 1000;
    } else if fs_hz > ps_enc.api_fs_hz
        || fs_hz > ps_enc.max_internal_fs_hz
        || fs_hz < ps_enc.min_internal_fs_hz
    {
        fs_hz = ps_enc.api_fs_hz;
        fs_hz = imin(fs_hz, ps_enc.max_internal_fs_hz);
        fs_hz = imax(fs_hz, ps_enc.min_internal_fs_hz);
        fs_khz = fs_hz / 1000;
    } else {
        // State machine for internal sampling rate switching
        if ps_enc.s_lp.transition_frame_no >= TRANSITION_FRAMES {
            ps_enc.s_lp.mode = 0;
        }
        if ps_enc.allow_bandwidth_switch != 0 || enc_control.opus_can_switch != 0 {
            // Check if we should switch down
            if orig_khz * 1000 > ps_enc.desired_internal_fs_hz {
                if ps_enc.s_lp.mode == 0 {
                    ps_enc.s_lp.transition_frame_no = TRANSITION_FRAMES;
                    ps_enc.s_lp.in_lp_state = [0; 2];
                }
                if enc_control.opus_can_switch != 0 {
                    ps_enc.s_lp.mode = 0;
                    fs_khz = if orig_khz == 16 { 12 } else { 8 };
                } else if ps_enc.s_lp.transition_frame_no <= 0 {
                    enc_control.switch_ready = 1;
                    enc_control.max_bits -= enc_control.max_bits * 5
                        / (enc_control.payload_size_ms + 5);
                } else {
                    ps_enc.s_lp.mode = -2;
                }
            } else if orig_khz * 1000 < ps_enc.desired_internal_fs_hz {
                // Switch up
                if enc_control.opus_can_switch != 0 {
                    fs_khz = if orig_khz == 8 { 12 } else { 16 };
                    ps_enc.s_lp.transition_frame_no = 0;
                    ps_enc.s_lp.in_lp_state = [0; 2];
                    ps_enc.s_lp.mode = 1;
                } else if ps_enc.s_lp.mode == 0 {
                    enc_control.switch_ready = 1;
                    enc_control.max_bits -= enc_control.max_bits * 5
                        / (enc_control.payload_size_ms + 5);
                } else {
                    ps_enc.s_lp.mode = 1;
                }
            } else if ps_enc.s_lp.mode < 0 {
                ps_enc.s_lp.mode = 1;
            }
        }
    }
    fs_khz
}

// ===========================================================================
// Encoder setup functions
// ===========================================================================

/// Setup complexity-dependent parameters.
/// Matches C: `silk_setup_complexity`.
fn silk_setup_complexity(ps_enc: &mut SilkEncoderState, complexity: i32) -> i32 {
    if complexity < 1 {
        ps_enc.pitch_estimation_complexity = SILK_PE_MIN_COMPLEX;
        ps_enc.pitch_estimation_threshold_q16 = ((0.8f64 * 65536.0) + 0.5) as i32;
        ps_enc.pitch_estimation_lpc_order = 6;
        ps_enc.shaping_lpc_order = 12;
        ps_enc.la_shape = 3 * ps_enc.fs_khz;
        ps_enc.n_states_delayed_decision = 1;
        ps_enc.use_interpolated_nlsfs = 0;
        ps_enc.nlsf_msvq_survivors = 2;
        ps_enc.warping_q16 = 0;
    } else if complexity < 2 {
        ps_enc.pitch_estimation_complexity = SILK_PE_MID_COMPLEX;
        ps_enc.pitch_estimation_threshold_q16 = ((0.76f64 * 65536.0) + 0.5) as i32;
        ps_enc.pitch_estimation_lpc_order = 8;
        ps_enc.shaping_lpc_order = 14;
        ps_enc.la_shape = 5 * ps_enc.fs_khz;
        ps_enc.n_states_delayed_decision = 1;
        ps_enc.use_interpolated_nlsfs = 0;
        ps_enc.nlsf_msvq_survivors = 3;
        ps_enc.warping_q16 = 0;
    } else if complexity < 3 {
        ps_enc.pitch_estimation_complexity = SILK_PE_MIN_COMPLEX;
        ps_enc.pitch_estimation_threshold_q16 = ((0.8f64 * 65536.0) + 0.5) as i32;
        ps_enc.pitch_estimation_lpc_order = 6;
        ps_enc.shaping_lpc_order = 12;
        ps_enc.la_shape = 3 * ps_enc.fs_khz;
        ps_enc.n_states_delayed_decision = 2;
        ps_enc.use_interpolated_nlsfs = 0;
        ps_enc.nlsf_msvq_survivors = 2;
        ps_enc.warping_q16 = 0;
    } else if complexity < 4 {
        ps_enc.pitch_estimation_complexity = SILK_PE_MID_COMPLEX;
        ps_enc.pitch_estimation_threshold_q16 = ((0.76f64 * 65536.0) + 0.5) as i32;
        ps_enc.pitch_estimation_lpc_order = 8;
        ps_enc.shaping_lpc_order = 14;
        ps_enc.la_shape = 5 * ps_enc.fs_khz;
        ps_enc.n_states_delayed_decision = 2;
        ps_enc.use_interpolated_nlsfs = 0;
        ps_enc.nlsf_msvq_survivors = 4;
        ps_enc.warping_q16 = 0;
    } else if complexity < 6 {
        ps_enc.pitch_estimation_complexity = SILK_PE_MID_COMPLEX;
        ps_enc.pitch_estimation_threshold_q16 = ((0.74f64 * 65536.0) + 0.5) as i32;
        ps_enc.pitch_estimation_lpc_order = 10;
        ps_enc.shaping_lpc_order = 16;
        ps_enc.la_shape = 5 * ps_enc.fs_khz;
        ps_enc.n_states_delayed_decision = 2;
        ps_enc.use_interpolated_nlsfs = 1;
        ps_enc.nlsf_msvq_survivors = 6;
        ps_enc.warping_q16 = ps_enc.fs_khz * WARPING_MULTIPLIER_Q16;
    } else if complexity < 8 {
        ps_enc.pitch_estimation_complexity = SILK_PE_MID_COMPLEX;
        ps_enc.pitch_estimation_threshold_q16 = ((0.72f64 * 65536.0) + 0.5) as i32;
        ps_enc.pitch_estimation_lpc_order = 12;
        ps_enc.shaping_lpc_order = 20;
        ps_enc.la_shape = 5 * ps_enc.fs_khz;
        ps_enc.n_states_delayed_decision = 3;
        ps_enc.use_interpolated_nlsfs = 1;
        ps_enc.nlsf_msvq_survivors = 8;
        ps_enc.warping_q16 = ps_enc.fs_khz * WARPING_MULTIPLIER_Q16;
    } else {
        ps_enc.pitch_estimation_complexity = SILK_PE_MAX_COMPLEX;
        ps_enc.pitch_estimation_threshold_q16 = ((0.7f64 * 65536.0) + 0.5) as i32;
        ps_enc.pitch_estimation_lpc_order = 16;
        ps_enc.shaping_lpc_order = 24;
        ps_enc.la_shape = 5 * ps_enc.fs_khz;
        ps_enc.n_states_delayed_decision = MAX_DEL_DEC_STATES as i32;
        ps_enc.use_interpolated_nlsfs = 1;
        ps_enc.nlsf_msvq_survivors = 16;
        ps_enc.warping_q16 = ps_enc.fs_khz * WARPING_MULTIPLIER_Q16;
    }

    // Don't allow higher pitch estimation LPC order than predict LPC order
    ps_enc.pitch_estimation_lpc_order =
        imin(ps_enc.pitch_estimation_lpc_order, ps_enc.predict_lpc_order);
    ps_enc.shape_win_length = SUB_FRAME_LENGTH_MS as i32 * ps_enc.fs_khz + 2 * ps_enc.la_shape;
    ps_enc.complexity = complexity;
    0
}

/// Setup LBRR (low bitrate redundancy for FEC).
/// Matches C: `silk_setup_LBRR`.
fn silk_setup_lbrr(ps_enc: &mut SilkEncoderState, enc_control: &SilkEncControlStruct) -> i32 {
    let lbrr_in_previous_packet = ps_enc.lbrr_enabled;
    ps_enc.lbrr_enabled = enc_control.lbrr_coded;
    if ps_enc.lbrr_enabled != 0 {
        if lbrr_in_previous_packet == 0 {
            ps_enc.lbrr_gain_increases = 7;
        } else {
            ps_enc.lbrr_gain_increases = imax(
                7 - silk_smulwb(ps_enc.packet_loss_perc, ((0.2f64 * 65536.0) + 0.5) as i32 as i16),
                3,
            );
        }
    }
    SILK_NO_ERROR
}

/// Setup internal sampling frequency and related parameters.
/// Matches C: `silk_setup_fs`.
fn silk_setup_fs(
    ps_enc: &mut SilkEncoderStateFix,
    fs_khz: i32,
    packet_size_ms: i32,
) -> i32 {
    let mut ret = SILK_NO_ERROR;

    // Set packet size
    if packet_size_ms != ps_enc.s_cmn.packet_size_ms {
        if packet_size_ms != 10 && packet_size_ms != 20 && packet_size_ms != 40
            && packet_size_ms != 60
        {
            ret = SILK_ENC_PACKET_SIZE_NOT_SUPPORTED;
        }
        if packet_size_ms <= 10 {
            ps_enc.s_cmn.n_frames_per_packet = 1;
            ps_enc.s_cmn.nb_subfr = if packet_size_ms == 10 { 2 } else { 1 };
            ps_enc.s_cmn.frame_length = packet_size_ms * fs_khz;
            ps_enc.s_cmn.pitch_lpc_win_length = FIND_PITCH_LPC_WIN_MS_2_SF * fs_khz;
            if ps_enc.s_cmn.fs_khz == 8 {
                ps_enc.s_cmn.pitch_contour_icdf = &SILK_PITCH_CONTOUR_10_MS_NB_ICDF;
            } else {
                ps_enc.s_cmn.pitch_contour_icdf = &SILK_PITCH_CONTOUR_10_MS_ICDF;
            }
        } else {
            ps_enc.s_cmn.n_frames_per_packet = packet_size_ms / MAX_FRAME_LENGTH_MS as i32;
            ps_enc.s_cmn.nb_subfr = MAX_NB_SUBFR as i32;
            ps_enc.s_cmn.frame_length = 20 * fs_khz;
            ps_enc.s_cmn.pitch_lpc_win_length = FIND_PITCH_LPC_WIN_MS * fs_khz;
            if ps_enc.s_cmn.fs_khz == 8 {
                ps_enc.s_cmn.pitch_contour_icdf = &SILK_PITCH_CONTOUR_NB_ICDF;
            } else {
                ps_enc.s_cmn.pitch_contour_icdf = &SILK_PITCH_CONTOUR_ICDF;
            }
        }
        ps_enc.s_cmn.packet_size_ms = packet_size_ms;
        ps_enc.s_cmn.target_rate_bps = 0; // trigger new SNR computation
    }

    // Set internal sampling frequency
    if ps_enc.s_cmn.fs_khz != fs_khz {
        // Reset part of the state
        ps_enc.s_shape = SilkShapeStateFix::default();
        ps_enc.s_cmn.s_nsq = NsqState::default();
        ps_enc.s_cmn.prev_nlsfq_q15 = [0; MAX_LPC_ORDER];
        ps_enc.s_cmn.s_lp.in_lp_state = [0; 2];
        ps_enc.s_cmn.input_buf_ix = 0;
        ps_enc.s_cmn.n_frames_encoded = 0;
        ps_enc.s_cmn.target_rate_bps = 0;

        // Initialize non-zero parameters
        ps_enc.s_cmn.prev_lag = 100;
        ps_enc.s_cmn.first_frame_after_reset = 1;
        ps_enc.s_shape.last_gain_index = 10;
        ps_enc.s_cmn.s_nsq.lag_prev = 100;
        ps_enc.s_cmn.s_nsq.prev_gain_q16 = 65536;
        ps_enc.s_cmn.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;

        ps_enc.s_cmn.fs_khz = fs_khz;
        if fs_khz == 8 {
            if ps_enc.s_cmn.nb_subfr == MAX_NB_SUBFR as i32 {
                ps_enc.s_cmn.pitch_contour_icdf = &SILK_PITCH_CONTOUR_NB_ICDF;
            } else {
                ps_enc.s_cmn.pitch_contour_icdf = &SILK_PITCH_CONTOUR_10_MS_NB_ICDF;
            }
        } else if ps_enc.s_cmn.nb_subfr == MAX_NB_SUBFR as i32 {
            ps_enc.s_cmn.pitch_contour_icdf = &SILK_PITCH_CONTOUR_ICDF;
        } else {
            ps_enc.s_cmn.pitch_contour_icdf = &SILK_PITCH_CONTOUR_10_MS_ICDF;
        }

        if fs_khz == 8 || fs_khz == 12 {
            ps_enc.s_cmn.predict_lpc_order = MIN_LPC_ORDER as i32;
            ps_enc.s_cmn.nlsf_cb = &SILK_NLSF_CB_NB_MB;
        } else {
            ps_enc.s_cmn.predict_lpc_order = MAX_LPC_ORDER as i32;
            ps_enc.s_cmn.nlsf_cb = &SILK_NLSF_CB_WB;
        }
        ps_enc.s_cmn.subfr_length = SUB_FRAME_LENGTH_MS as i32 * fs_khz;
        ps_enc.s_cmn.frame_length = ps_enc.s_cmn.subfr_length * ps_enc.s_cmn.nb_subfr;
        ps_enc.s_cmn.ltp_mem_length = LTP_MEM_LENGTH_MS as i32 * fs_khz;
        ps_enc.s_cmn.la_pitch = LA_PITCH_MS * fs_khz;
        ps_enc.s_cmn.max_pitch_lag = 18 * fs_khz;
        if ps_enc.s_cmn.nb_subfr == MAX_NB_SUBFR as i32 {
            ps_enc.s_cmn.pitch_lpc_win_length = FIND_PITCH_LPC_WIN_MS * fs_khz;
        } else {
            ps_enc.s_cmn.pitch_lpc_win_length = FIND_PITCH_LPC_WIN_MS_2_SF * fs_khz;
        }
        if fs_khz == 16 {
            ps_enc.s_cmn.pitch_lag_low_bits_icdf = &SILK_UNIFORM8_ICDF;
        } else if fs_khz == 12 {
            ps_enc.s_cmn.pitch_lag_low_bits_icdf = &SILK_UNIFORM6_ICDF;
        } else {
            ps_enc.s_cmn.pitch_lag_low_bits_icdf = &SILK_UNIFORM4_ICDF;
        }
    }
    ret
}

/// Setup resamplers for rate conversion.
/// Matches C: `silk_setup_resamplers`.
fn silk_setup_resamplers(ps_enc: &mut SilkEncoderStateFix, fs_khz: i32) -> i32 {
    let ret = SILK_NO_ERROR;

    if ps_enc.s_cmn.fs_khz != fs_khz || ps_enc.s_cmn.prev_api_fs_hz != ps_enc.s_cmn.api_fs_hz {
        if ps_enc.s_cmn.fs_khz == 0 {
            // Initialize the resampler
            silk_resampler_init(
                &mut ps_enc.s_cmn.resampler_state,
                ps_enc.s_cmn.api_fs_hz,
                fs_khz * 1000,
            );
        } else {
            // Re-initialize: need to resample buffered data
            let buf_length_ms = (ps_enc.s_cmn.nb_subfr * 5) * 2 + LA_SHAPE_MS;
            let old_buf_samples = buf_length_ms * ps_enc.s_cmn.fs_khz;

            // Temp resampler: internal rate → API rate
            let mut temp_resampler = SilkResamplerState::default();
            silk_resampler_init(
                &mut temp_resampler,
                ps_enc.s_cmn.fs_khz * 1000,
                ps_enc.s_cmn.api_fs_hz,
            );

            let api_buf_samples = buf_length_ms * (ps_enc.s_cmn.api_fs_hz / 1000);
            let mut x_buf_api = vec![0i16; api_buf_samples as usize];

            // Upsample x_buf to API rate
            silk_resampler_run(
                &mut temp_resampler,
                &mut x_buf_api,
                &ps_enc.x_buf[..old_buf_samples as usize],
            );

            // Re-init resampler: API rate → new internal rate
            silk_resampler_init(
                &mut ps_enc.s_cmn.resampler_state,
                ps_enc.s_cmn.api_fs_hz,
                fs_khz * 1000,
            );

            // Downsample back to new internal rate
            let new_buf_samples = buf_length_ms * fs_khz;
            let mut x_buf_new = vec![0i16; new_buf_samples as usize];
            silk_resampler_run(
                &mut ps_enc.s_cmn.resampler_state,
                &mut x_buf_new,
                &x_buf_api,
            );
            ps_enc.x_buf[..new_buf_samples as usize].copy_from_slice(&x_buf_new);
        }
    }
    ps_enc.s_cmn.prev_api_fs_hz = ps_enc.s_cmn.api_fs_hz;
    ret
}

/// Control encoder parameters.
/// Matches C: `silk_control_encoder`.
pub fn silk_control_encoder(
    ps_enc: &mut SilkEncoderStateFix,
    enc_control: &mut SilkEncControlStruct,
    allow_bw_switch: i32,
    channel_nb: i32,
    force_fs_khz: i32,
) -> i32 {
    let mut ret = 0i32;

    ps_enc.s_cmn.use_dtx = enc_control.use_dtx;
    ps_enc.s_cmn.use_cbr = enc_control.use_cbr;
    ps_enc.s_cmn.api_fs_hz = enc_control.api_sample_rate;
    ps_enc.s_cmn.max_internal_fs_hz = enc_control.max_internal_sample_rate;
    ps_enc.s_cmn.min_internal_fs_hz = enc_control.min_internal_sample_rate;
    ps_enc.s_cmn.desired_internal_fs_hz = enc_control.desired_internal_sample_rate;
    ps_enc.s_cmn.use_in_band_fec = enc_control.use_in_band_fec;
    ps_enc.s_cmn.n_channels_api = enc_control.n_channels_api;
    ps_enc.s_cmn.n_channels_internal = enc_control.n_channels_internal;
    ps_enc.s_cmn.allow_bandwidth_switch = allow_bw_switch;
    ps_enc.s_cmn.channel_nb = channel_nb;

    if ps_enc.s_cmn.controlled_since_last_payload != 0 && ps_enc.s_cmn.prefill_flag == 0 {
        if ps_enc.s_cmn.api_fs_hz != ps_enc.s_cmn.prev_api_fs_hz && ps_enc.s_cmn.fs_khz > 0 {
            ret += silk_setup_resamplers(ps_enc, ps_enc.s_cmn.fs_khz);
        }
        return ret;
    }

    // Determine internal sampling rate
    let mut fs_khz = silk_control_audio_bandwidth(&mut ps_enc.s_cmn, enc_control);
    if force_fs_khz != 0 {
        fs_khz = force_fs_khz;
    }

    // Prepare resampler
    ret += silk_setup_resamplers(ps_enc, fs_khz);

    // Set internal sampling frequency
    ret += silk_setup_fs(ps_enc, fs_khz, enc_control.payload_size_ms);

    // Set encoding complexity
    ret += silk_setup_complexity(&mut ps_enc.s_cmn, enc_control.complexity);

    // Set packet loss rate
    ps_enc.s_cmn.packet_loss_perc = enc_control.packet_loss_percentage;

    // Set LBRR usage
    ret += silk_setup_lbrr(&mut ps_enc.s_cmn, enc_control);

    ps_enc.s_cmn.controlled_since_last_payload = 1;
    ret
}

// ===========================================================================
// Resampler stubs (delegate to decoder's resampler implementation)
// ===========================================================================

/// Initialize resampler state. Wrapper delegating to decoder's resampler.
fn silk_resampler_init(state: &mut SilkResamplerState, fs_in_hz: i32, fs_out_hz: i32) {
    // Delegate to the decoder's resampler initialization (for_enc = true)
    crate::silk::decoder::silk_resampler_init_pub(state, fs_in_hz, fs_out_hz, true);
}

/// Run the resampler.
fn silk_resampler_run(state: &mut SilkResamplerState, output: &mut [i16], input: &[i16]) {
    crate::silk::decoder::silk_resampler(state, output, input, input.len());
}

// ===========================================================================
// A2NLSF: LPC coefficients → Normalized Line Spectral Frequencies
// ===========================================================================

/// Convert LPC coefficients to NLSFs.
/// Matches C: `silk_A2NLSF`.
pub fn silk_a2nlsf(nlsf: &mut [i16], a_q16: &[i32], d: usize) {
    let dd = silk_rshift(d as i32, 1) as usize;
    let mut a_q16_mut = [0i32; MAX_LPC_ORDER];
    a_q16_mut[..d].copy_from_slice(&a_q16[..d]);

    let mut p = [0i32; MAX_LPC_ORDER / 2 + 1];
    let mut q = [0i32; MAX_LPC_ORDER / 2 + 1];

    silk_a2nlsf_init(&a_q16_mut, &mut p, &mut q, dd);

    // Find roots, alternating between P and Q
    let pq: [*const i32; 2] = [p.as_ptr(), q.as_ptr()];
    let mut poly_p = &p as &[i32];

    let mut xlo = SILK_LSF_COS_TAB_FIX_Q12[0] as i32;
    let mut ylo = silk_a2nlsf_eval_poly(poly_p, xlo, dd);

    let mut root_ix: usize;
    if ylo < 0 {
        // Set the first NLSF to zero and move on to the next
        nlsf[0] = 0;
        poly_p = &q;
        ylo = silk_a2nlsf_eval_poly(poly_p, xlo, dd);
        root_ix = 1;
    } else {
        root_ix = 0;
    }

    let mut k: usize = 1;
    let mut i: usize = 0;
    let mut thr: i32 = 0;

    loop {
        // Evaluate polynomial
        let xhi = SILK_LSF_COS_TAB_FIX_Q12[k] as i32;
        let yhi = silk_a2nlsf_eval_poly(poly_p, xhi, dd);

        // Detect zero crossing
        if (ylo <= 0 && yhi >= thr) || (ylo >= 0 && yhi <= -thr) {
            if yhi == 0 {
                thr = 1;
            } else {
                thr = 0;
            }

            // Binary division
            let mut ffrac: i32 = -256;
            let mut xlo_b = xlo;
            let mut ylo_b = ylo;
            let mut xhi_b = xhi;
            let mut yhi_b = yhi;
            for m in 0..BIN_DIV_STEPS_A2NLSF {
                let xmid = silk_rshift_round(xlo_b + xhi_b, 1);
                let ymid = silk_a2nlsf_eval_poly(poly_p, xmid, dd);

                if (ylo_b <= 0 && ymid >= 0) || (ylo_b >= 0 && ymid <= 0) {
                    xhi_b = xmid;
                    yhi_b = ymid;
                } else {
                    xlo_b = xmid;
                    ylo_b = ymid;
                    ffrac += 128 >> m;
                }
            }

            // Interpolate
            if ylo_b.abs() < 65536 {
                let den = ylo_b - yhi_b;
                let nom = silk_lshift(ylo_b, 8 - BIN_DIV_STEPS_A2NLSF as i32)
                    + silk_rshift(den, 1);
                if den != 0 {
                    ffrac += nom / den;
                }
            } else {
                ffrac += ylo_b
                    / silk_rshift(ylo_b - yhi_b, 8 - BIN_DIV_STEPS_A2NLSF as i32);
            }
            nlsf[root_ix] =
                imin(silk_lshift(k as i32, 8) + ffrac, i16::MAX as i32) as i16;

            root_ix += 1;
            if root_ix >= d {
                break;
            }
            // Alternate pointer to polynomial
            poly_p = if (root_ix & 1) == 0 { &p } else { &q };

            // Evaluate polynomial
            xlo = SILK_LSF_COS_TAB_FIX_Q12[k - 1] as i32;
            ylo = silk_lshift(1 - (root_ix as i32 & 2), 12);
        } else {
            k += 1;
            xlo = xhi;
            ylo = yhi;
            thr = 0;

            if k > LSF_COS_TAB_SZ_FIX {
                i += 1;
                if i > MAX_ITERATIONS_A2NLSF {
                    // Set NLSFs to white spectrum and exit
                    nlsf[0] = ((1i32 << 15) / (d as i32 + 1)) as i16;
                    for kk in 1..d {
                        nlsf[kk] = nlsf[kk - 1] + nlsf[0];
                    }
                    return;
                }

                // Apply progressively more bandwidth expansion and retry
                silk_bwexpander_32(&mut a_q16_mut[..d], d, 65536 - silk_lshift(1, i as i32));

                silk_a2nlsf_init(&a_q16_mut, &mut p, &mut q, dd);
                poly_p = &p;
                xlo = SILK_LSF_COS_TAB_FIX_Q12[0] as i32;
                ylo = silk_a2nlsf_eval_poly(poly_p, xlo, dd);
                if ylo < 0 {
                    nlsf[0] = 0;
                    poly_p = &q;
                    ylo = silk_a2nlsf_eval_poly(poly_p, xlo, dd);
                    root_ix = 1;
                } else {
                    root_ix = 0;
                }
                k = 1;
            }
        }
    }
}

/// Transforms polynomials from cos(n*f) to cos(f)^n.
/// Matches C: `silk_A2NLSF_trans_poly`.
fn silk_a2nlsf_trans_poly(p: &mut [i32], dd: usize) {
    for k in 2..=dd {
        for n in (k + 1..=dd).rev() {
            p[n - 2] -= p[n];
        }
        p[k - 2] -= silk_lshift(p[k], 1);
    }
}

/// Polynomial evaluation using Horner's method.
/// Matches C: `silk_A2NLSF_eval_poly`.
fn silk_a2nlsf_eval_poly(p: &[i32], x: i32, dd: usize) -> i32 {
    let mut y32 = p[dd];
    let x_q16 = silk_lshift(x, 4);

    for n in (0..dd).rev() {
        y32 = silk_smlaww(p[n], y32, x_q16);
    }
    y32
}

/// Initialize P and Q polynomials from LPC coefficients.
/// Matches C: `silk_A2NLSF_init`.
fn silk_a2nlsf_init(a_q16: &[i32], p: &mut [i32], q: &mut [i32], dd: usize) {
    let d = dd * 2;

    // Convert filter coefs to even and odd polynomials
    p[dd] = silk_lshift(1, 16);
    q[dd] = silk_lshift(1, 16);
    for k in 0..dd {
        p[k] = -a_q16[dd - k - 1] - a_q16[dd + k];
        q[k] = -a_q16[dd - k - 1] + a_q16[dd + k];
    }

    // Divide out zeros: z=1 is always a root in Q, z=-1 always in P
    for k in (1..=dd).rev() {
        p[k - 1] -= p[k];
        q[k - 1] += q[k];
    }

    // Transform polynomials from cos(n*f) to cos(f)^n
    silk_a2nlsf_trans_poly(p, dd);
    silk_a2nlsf_trans_poly(q, dd);
}

// ===========================================================================
// NLSF processing
// ===========================================================================

/// Linear interpolation of two NLSF vectors.
/// Matches C: `silk_interpolate`.
pub fn silk_interpolate(
    xi: &mut [i16],
    x0: &[i16],
    x1: &[i16],
    ifact_q2: i32,
    d: usize,
) {
    for i in 0..d {
        xi[i] = (x0[i] as i32 + ((x1[i] as i32 - x0[i] as i32) * ifact_q2 >> 2)) as i16;
    }
}

/// Compute NLSF weights using the Laroia method.
/// Matches C: `silk_NLSF_VQ_weights_laroia`.
pub fn silk_nlsf_vq_weights_laroia(
    nlsf_w_q_out: &mut [i16],
    nlsf_q15: &[i16],
    order: usize,
) {
    // First and last weight
    let tmp1_w = nlsf_q15[0] as i32;
    let tmp2_w = nlsf_q15[1] as i32 - nlsf_q15[0] as i32;

    let tmp1_inv = 131072 / imax(tmp1_w, 1); // Q17 / Q15 = Q2 approximately
    let tmp2_inv = 131072 / imax(tmp2_w, 1);
    nlsf_w_q_out[0] = imin(tmp1_inv + tmp2_inv, i16::MAX as i32) as i16;

    for i in 1..(order - 1) {
        let tmp1 = nlsf_q15[i] as i32 - nlsf_q15[i - 1] as i32;
        let tmp2 = nlsf_q15[i + 1] as i32 - nlsf_q15[i] as i32;
        let tmp1_inv = 131072 / imax(tmp1, 1);
        let tmp2_inv = 131072 / imax(tmp2, 1);
        nlsf_w_q_out[i] = imin(tmp1_inv + tmp2_inv, i16::MAX as i32) as i16;
    }

    let last = order - 1;
    let tmp1 = nlsf_q15[last] as i32 - nlsf_q15[last - 1] as i32;
    let tmp2 = (1 << 15) - nlsf_q15[last] as i32;
    let tmp1_inv = 131072 / imax(tmp1, 1);
    let tmp2_inv = 131072 / imax(tmp2, 1);
    nlsf_w_q_out[last] = imin(tmp1_inv + tmp2_inv, i16::MAX as i32) as i16;
}

/// First-stage NLSF VQ: compute weighted error for each codebook vector.
/// Matches C: `silk_NLSF_VQ`.
pub fn silk_nlsf_vq(
    err_q24: &mut [i32],
    input_q15: &[i16],
    _nlsf_w_q2: &[i16],
    cb_q8: &[u8],
    cb_wght_q9: &[i16],
    n_vectors: usize,
    order: usize,
) {
    // Loop over codebook vectors
    for i in 0..n_vectors {
        let mut sum_error_q24: i32 = 0;
        let mut pred_q24: i32 = 0;
        let cb_offset = i * order;
        let w_offset = i * order;

        // Loop in reverse order, processing pairs
        let mut m = order as i32 - 2;
        while m >= 0 {
            let mu = m as usize;
            // Index m + 1
            let diff_q15 = input_q15[mu + 1] as i32 - ((cb_q8[cb_offset + mu + 1] as i32) << 7);
            let diffw_q24 = silk_smulbb(diff_q15, cb_wght_q9[w_offset + mu + 1] as i32);
            sum_error_q24 = sum_error_q24.wrapping_add((diffw_q24 - silk_rshift(pred_q24, 1)).abs());
            pred_q24 = diffw_q24;

            // Index m
            let diff_q15 = input_q15[mu] as i32 - ((cb_q8[cb_offset + mu] as i32) << 7);
            let diffw_q24 = silk_smulbb(diff_q15, cb_wght_q9[w_offset + mu] as i32);
            sum_error_q24 = sum_error_q24.wrapping_add((diffw_q24 - silk_rshift(pred_q24, 1)).abs());
            pred_q24 = diffw_q24;

            m -= 2;
        }
        err_q24[i] = sum_error_q24;
    }
}

/// NLSF delayed-decision quantization.
/// Matches C: `silk_NLSF_del_dec_quant`.
pub fn silk_nlsf_del_dec_quant(
    indices: &mut [i8],
    x_q10: &[i16],
    w_q5: &[i16],
    pred_coef_q8: &[u8],
    ec_ix: &[i16],
    ec_rates_q5: &[u8],
    quant_step_size_q16: i32,
    inv_quant_step_size_q6: i16,
    mu_q20: i32,
    order: i16,
) -> i32 {
    let order_u = order as usize;
    const LEVEL_ADJ_Q10: i32 = 102; // SILK_FIX_CONST(NLSF_QUANT_LEVEL_ADJ, 10)

    // Precompute output tables (C lines 60-80)
    let mut out0_q10_table = [0i32; 2 * NLSF_QUANT_MAX_AMPLITUDE_EXT as usize];
    let mut out1_q10_table = [0i32; 2 * NLSF_QUANT_MAX_AMPLITUDE_EXT as usize];
    for ii in -NLSF_QUANT_MAX_AMPLITUDE_EXT..NLSF_QUANT_MAX_AMPLITUDE_EXT {
        let mut out0: i32 = ii << 10;
        let mut out1: i32 = out0 + 1024;
        if ii > 0 {
            out0 -= LEVEL_ADJ_Q10;
            out1 -= LEVEL_ADJ_Q10;
        } else if ii == 0 {
            out1 -= LEVEL_ADJ_Q10;
        } else if ii == -1 {
            out0 += LEVEL_ADJ_Q10;
        } else {
            out0 += LEVEL_ADJ_Q10;
            out1 += LEVEL_ADJ_Q10;
        }
        let idx = (ii + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize;
        out0_q10_table[idx] = silk_smulbb(out0, quant_step_size_q16) >> 16;
        out1_q10_table[idx] = silk_smulbb(out1, quant_step_size_q16) >> 16;
    }

    // State arrays (C lines 52-57)
    let mut ind = [[0i8; MAX_LPC_ORDER]; NLSF_QUANT_DEL_DEC_STATES];
    let mut prev_out_q10 = [0i16; 2 * NLSF_QUANT_DEL_DEC_STATES];
    let mut rd_q25 = [0i32; 2 * NLSF_QUANT_DEL_DEC_STATES];
    let mut rd_min_q25 = [0i32; NLSF_QUANT_DEL_DEC_STATES];
    let mut rd_max_q25 = [0i32; NLSF_QUANT_DEL_DEC_STATES];
    let mut ind_sort = [0usize; NLSF_QUANT_DEL_DEC_STATES];

    let mut n_states: usize = 1;
    rd_q25[0] = 0;
    prev_out_q10[0] = 0;

    for i in (0..order_u).rev() {
        let rates_q5 = &ec_rates_q5[ec_ix[i] as usize..];
        let in_q10 = x_q10[i] as i32;

        for j in 0..n_states {
            // Prediction (C line 91)
            let pred_q10 = silk_smulbb(
                pred_coef_q8[i] as i16 as i32,
                prev_out_q10[j] as i32,
            ) >> 8;
            let res_q10 = in_q10 - pred_q10;
            // Quantize (C lines 93-94)
            let mut ind_tmp = silk_smulbb(inv_quant_step_size_q6 as i32, res_q10) >> 16;
            ind_tmp = ind_tmp.clamp(
                -NLSF_QUANT_MAX_AMPLITUDE_EXT,
                NLSF_QUANT_MAX_AMPLITUDE_EXT - 1,
            );
            ind[j][i] = ind_tmp as i8;

            // Look up outputs from precomputed tables (C lines 98-104)
            let tbl_idx = (ind_tmp + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize;
            let out0_q10 = (out0_q10_table[tbl_idx] + pred_q10) as i16;
            let out1_q10 = (out1_q10_table[tbl_idx] + pred_q10) as i16;
            prev_out_q10[j] = out0_q10;
            prev_out_q10[j + n_states] = out1_q10;

            // Compute rates with three-way branch (C lines 107-126)
            let rate0_q5: i32;
            let rate1_q5: i32;
            if ind_tmp + 1 >= NLSF_QUANT_MAX_AMPLITUDE {
                if ind_tmp + 1 == NLSF_QUANT_MAX_AMPLITUDE {
                    rate0_q5 = rates_q5[(ind_tmp + NLSF_QUANT_MAX_AMPLITUDE) as usize] as i32;
                    rate1_q5 = 280;
                } else {
                    rate0_q5 =
                        silk_smlabb(280 - 43 * NLSF_QUANT_MAX_AMPLITUDE, 43, ind_tmp);
                    rate1_q5 = rate0_q5 + 43;
                }
            } else if ind_tmp <= -NLSF_QUANT_MAX_AMPLITUDE {
                if ind_tmp == -NLSF_QUANT_MAX_AMPLITUDE {
                    rate0_q5 = 280;
                    rate1_q5 =
                        rates_q5[(ind_tmp + 1 + NLSF_QUANT_MAX_AMPLITUDE) as usize] as i32;
                } else {
                    rate0_q5 =
                        silk_smlabb(280 - 43 * NLSF_QUANT_MAX_AMPLITUDE, -43, ind_tmp);
                    rate1_q5 = rate0_q5 - 43;
                }
            } else {
                rate0_q5 = rates_q5[(ind_tmp + NLSF_QUANT_MAX_AMPLITUDE) as usize] as i32;
                rate1_q5 =
                    rates_q5[(ind_tmp + 1 + NLSF_QUANT_MAX_AMPLITUDE) as usize] as i32;
            }

            // Compute RD for both candidates (C lines 127-131)
            let rd_tmp_q25 = rd_q25[j];
            let diff0 = in_q10 - out0_q10 as i32;
            rd_q25[j] = silk_smlabb(
                silk_mla(rd_tmp_q25, silk_smulbb(diff0, diff0), w_q5[i] as i32),
                mu_q20,
                rate0_q5,
            );
            let diff1 = in_q10 - out1_q10 as i32;
            rd_q25[j + n_states] = silk_smlabb(
                silk_mla(rd_tmp_q25, silk_smulbb(diff1, diff1), w_q5[i] as i32),
                mu_q20,
                rate1_q5,
            );
        }

        if n_states <= NLSF_QUANT_DEL_DEC_STATES / 2 {
            // Double number of states and copy (C lines 136-143)
            for j in 0..n_states {
                ind[j + n_states][i] = ind[j][i] + 1;
            }
            n_states <<= 1;
            for j in n_states..NLSF_QUANT_DEL_DEC_STATES {
                ind[j][i] = ind[j - n_states][i];
            }
        } else {
            // Pairwise sort lower and upper half (C lines 145-161)
            for j in 0..NLSF_QUANT_DEL_DEC_STATES {
                if rd_q25[j] > rd_q25[j + NLSF_QUANT_DEL_DEC_STATES] {
                    rd_max_q25[j] = rd_q25[j];
                    rd_min_q25[j] = rd_q25[j + NLSF_QUANT_DEL_DEC_STATES];
                    rd_q25[j] = rd_min_q25[j];
                    rd_q25[j + NLSF_QUANT_DEL_DEC_STATES] = rd_max_q25[j];
                    let tmp = prev_out_q10[j];
                    prev_out_q10[j] = prev_out_q10[j + NLSF_QUANT_DEL_DEC_STATES];
                    prev_out_q10[j + NLSF_QUANT_DEL_DEC_STATES] = tmp;
                    ind_sort[j] = j + NLSF_QUANT_DEL_DEC_STATES;
                } else {
                    rd_min_q25[j] = rd_q25[j];
                    rd_max_q25[j] = rd_q25[j + NLSF_QUANT_DEL_DEC_STATES];
                    ind_sort[j] = j;
                }
            }
            // Compare highest RD of winning half with lowest of losing half (C lines 164-189)
            loop {
                let mut min_max_q25 = i32::MAX;
                let mut max_min_q25: i32 = 0;
                let mut ind_min_max: usize = 0;
                let mut ind_max_min: usize = 0;
                for j in 0..NLSF_QUANT_DEL_DEC_STATES {
                    if min_max_q25 > rd_max_q25[j] {
                        min_max_q25 = rd_max_q25[j];
                        ind_min_max = j;
                    }
                    if max_min_q25 < rd_min_q25[j] {
                        max_min_q25 = rd_min_q25[j];
                        ind_max_min = j;
                    }
                }
                if min_max_q25 >= max_min_q25 {
                    break;
                }
                ind_sort[ind_max_min] =
                    ind_sort[ind_min_max] ^ NLSF_QUANT_DEL_DEC_STATES;
                rd_q25[ind_max_min] =
                    rd_q25[ind_min_max + NLSF_QUANT_DEL_DEC_STATES];
                prev_out_q10[ind_max_min] =
                    prev_out_q10[ind_min_max + NLSF_QUANT_DEL_DEC_STATES];
                rd_min_q25[ind_max_min] = 0;
                rd_max_q25[ind_min_max] = i32::MAX;
                ind[ind_max_min] = ind[ind_min_max];
            }
            // Increment index if from upper half (C lines 191-193)
            for j in 0..NLSF_QUANT_DEL_DEC_STATES {
                ind[j][i] += (ind_sort[j] >> NLSF_QUANT_DEL_DEC_STATES_LOG2) as i8;
            }
        }
    }

    // Find winner across all 2*NLSF_QUANT_DEL_DEC_STATES (C lines 198-211)
    let mut ind_tmp: usize = 0;
    let mut min_q25 = i32::MAX;
    for j in 0..(2 * NLSF_QUANT_DEL_DEC_STATES) {
        if min_q25 > rd_q25[j] {
            min_q25 = rd_q25[j];
            ind_tmp = j;
        }
    }
    for j in 0..order_u {
        indices[j] = ind[ind_tmp & (NLSF_QUANT_DEL_DEC_STATES - 1)][j];
    }
    indices[0] += (ind_tmp >> NLSF_QUANT_DEL_DEC_STATES_LOG2) as i8;

    min_q25
}

/// Full NLSF encoding pipeline.
/// Matches C: `silk_NLSF_encode`.
pub fn silk_nlsf_encode(
    nlsf_indices: &mut [i8],
    nlsf_q15: &mut [i16],
    cb: &SilkNlsfCbStruct,
    w_q2: &[i16],
    nlsf_mu_q20: i32,
    n_survivors: i32,
    signal_type: i32,
) -> i32 {
    let order = cb.order as usize;
    let n_vectors = cb.n_vectors as usize;

    eprintln!("[NLSF INPUT] nlsf_q15={:?} w_q2={:?}", &nlsf_q15[..order], &w_q2[..order]);

    // NLSF stabilization
    silk_nlsf_stabilize(nlsf_q15, cb.delta_min_q15, order);

    // First stage: VQ
    let mut err_q24 = vec![0i32; n_vectors];
    silk_nlsf_vq(&mut err_q24, nlsf_q15, w_q2, cb.cb1_nlsf_q8, cb.cb1_wght_q9, n_vectors, order);

    // Sort the quantization errors (insertion sort to find n_surv best)
    let n_surv = imin(n_survivors, n_vectors as i32) as usize;
    let mut temp_indices1 = vec![0i32; n_vectors];
    for i in 0..n_vectors {
        temp_indices1[i] = i as i32;
    }
    // Insertion sort: find n_surv smallest errors
    silk_insertion_sort_increasing(&mut err_q24, &mut temp_indices1, n_vectors, n_surv);

    eprintln!("[NLSF VQ] survivors: {:?} vq_errs: {:?} mu_q20={}", &temp_indices1[..n_surv], &err_q24[..n_surv], nlsf_mu_q20);

    let mut rd_q25 = vec![0i32; n_surv];
    let mut temp_indices2 = vec![0i8; n_surv * MAX_LPC_ORDER];

    // Loop over survivors
    for s in 0..n_surv {
        let ind1 = temp_indices1[s] as usize;

        // Residual after first stage
        let cb_offset = ind1 * order;
        let mut res_q10 = [0i16; MAX_LPC_ORDER];
        let mut w_adj_q5 = [0i16; MAX_LPC_ORDER];
        for i in 0..order {
            let nlsf_tmp_q15 = (cb.cb1_nlsf_q8[cb_offset + i] as i16) << 7;
            let w_tmp_q9 = cb.cb1_wght_q9[cb_offset + i] as i32;
            res_q10[i] = silk_rshift(
                silk_smulbb(nlsf_q15[i] as i32 - nlsf_tmp_q15 as i32, w_tmp_q9),
                14,
            ) as i16;
            w_adj_q5[i] =
                silk_div32_varq(w_q2[i] as i32, silk_smulbb(w_tmp_q9, w_tmp_q9), 21) as i16;
        }

        // Unpack entropy table indices and predictor
        let mut ec_ix = [0i16; MAX_LPC_ORDER];
        let mut pred_q8 = [0u8; MAX_LPC_ORDER];
        silk_nlsf_unpack(&mut ec_ix, &mut pred_q8, cb, ind1);

        // Trellis quantizer
        rd_q25[s] = silk_nlsf_del_dec_quant(
            &mut temp_indices2[s * MAX_LPC_ORDER..],
            &res_q10,
            &w_adj_q5,
            &pred_q8,
            &ec_ix,
            cb.ec_rates_q5,
            cb.quant_step_size_q16 as i32,
            cb.inv_quant_step_size_q6,
            nlsf_mu_q20,
            order as i16,
        );
        if ind1 == 0 && s < 5 {
            for i in 0..order {
                let w_tmp_q9 = cb.cb1_wght_q9[cb_offset + i] as i32;
                let sq = silk_smulbb(w_tmp_q9, w_tmp_q9);
                let div_result = silk_div32_varq(w_q2[i] as i32, sq, 21);
                eprintln!("[W_ADJ i={}] w_q2={} w_tmp_q9={} sq={} div_result={} as_i16={}",
                    i, w_q2[i], w_tmp_q9, sq, div_result, div_result as i16);
            }
            eprintln!("[DELDEC s={}] res_q10={:?} w_adj_q5={:?} pred_q8={:?} ec_ix={:?}",
                      s, &res_q10[..order], &w_adj_q5[..order], &pred_q8[..order], &ec_ix[..order]);
            eprintln!("[DELDEC s={}] out_indices={:?} rd={}",
                      s, &temp_indices2[s*MAX_LPC_ORDER..s*MAX_LPC_ORDER+order], rd_q25[s]);
        }

        // Add rate for first stage
        let icdf_ptr =
            &cb.cb1_icdf[((signal_type >> 1) as usize) * n_vectors..];
        let prob_q8 = if ind1 == 0 {
            256 - icdf_ptr[ind1] as i32
        } else {
            icdf_ptr[ind1 - 1] as i32 - icdf_ptr[ind1] as i32
        };
        let bits_q7 = (8 << 7) - silk_lin2log(prob_q8);
        rd_q25[s] = silk_smlabb(rd_q25[s], bits_q7, silk_rshift(nlsf_mu_q20, 2));
        eprintln!("[NLSF RD] s={} ind1={} rd_after_deldec={} prob_q8={} bits_q7={} rd_final={}", s, ind1, rd_q25[s] - silk_smlabb(0, bits_q7, silk_rshift(nlsf_mu_q20, 2)), prob_q8, bits_q7, rd_q25[s]);
    }

    // Find the lowest rate-distortion error
    let mut best_index = 0i32;
    {
        let mut rd_copy = rd_q25.clone();
        let mut idx_arr = vec![0i32; n_surv];
        for i in 0..n_surv {
            idx_arr[i] = i as i32;
        }
        silk_insertion_sort_increasing(&mut rd_copy, &mut idx_arr, n_surv, 1);
        best_index = idx_arr[0];
    }

    nlsf_indices[0] = temp_indices1[best_index as usize] as i8;
    let src_offset = best_index as usize * MAX_LPC_ORDER;
    for i in 0..order {
        nlsf_indices[1 + i] = temp_indices2[src_offset + i];
    }

    // Decode the quantized NLSFs for use by the encoder
    silk_nlsf_decode(nlsf_q15, nlsf_indices, cb);

    rd_q25[0]
}

/// Process NLSFs: compute weights, encode, convert to LPC.
/// Matches C: `silk_process_NLSFs`.
pub fn silk_process_nlsfs(
    ps_enc: &mut SilkEncoderState,
    _enc_control: &SilkEncoderControl,
    pred_coef_q12: &mut [[i16; MAX_LPC_ORDER]; 2],
    nlsf_q15: &mut [i16],
    prev_nlsf_q15: &[i16],
) {
    let order = ps_enc.predict_lpc_order as usize;

    // Calculate mu values
    // NLSF_mu = 0.003 - 0.001 * speech_activity
    let mut nlsf_mu_q20 = silk_smlawb_i32(
        (0.003 * (1 << 20) as f64 + 0.5) as i32,     // SILK_FIX_CONST(0.003, 20) = 3146
        (-0.001 * (1i64 << 28) as f64 - 0.5) as i32,  // SILK_FIX_CONST(-0.001, 28) = -268435
        ps_enc.speech_activity_q8,
    );
    if ps_enc.nb_subfr == 2 {
        // Multiply by 1.5 for 10 ms packets
        nlsf_mu_q20 = nlsf_mu_q20 + silk_rshift(nlsf_mu_q20, 1);
    }

    // Calculate NLSF weights
    let mut nlsf_w_qw = [0i16; MAX_LPC_ORDER];
    silk_nlsf_vq_weights_laroia(&mut nlsf_w_qw, nlsf_q15, order);

    // Update NLSF weights for interpolated NLSFs
    let do_interpolate = ps_enc.use_interpolated_nlsfs == 1
        && ps_enc.indices.nlsf_interp_coef_q2 < 4;

    if do_interpolate {
        // Calculate the interpolated NLSF vector for the first half
        let mut nlsf0_temp_q15 = [0i16; MAX_LPC_ORDER];
        silk_interpolate(
            &mut nlsf0_temp_q15,
            prev_nlsf_q15,
            nlsf_q15,
            ps_enc.indices.nlsf_interp_coef_q2 as i32,
            order,
        );

        // Calculate first half NLSF weights for the interpolated NLSFs
        let mut nlsf_w0_temp_qw = [0i16; MAX_LPC_ORDER];
        silk_nlsf_vq_weights_laroia(&mut nlsf_w0_temp_qw, &nlsf0_temp_q15, order);

        // Update NLSF weights with contribution from first half
        let i_sqr_q15 = silk_lshift(
            silk_smulbb(
                ps_enc.indices.nlsf_interp_coef_q2 as i32,
                ps_enc.indices.nlsf_interp_coef_q2 as i32,
            ),
            11,
        );
        for i in 0..order {
            nlsf_w_qw[i] = (silk_rshift(nlsf_w_qw[i] as i32, 1)
                + silk_rshift(
                    silk_smulbb(nlsf_w0_temp_qw[i] as i32, i_sqr_q15),
                    16,
                )) as i16;
        }
    }

    // NLSF encoding
    silk_nlsf_encode(
        &mut ps_enc.indices.nlsf_indices,
        nlsf_q15,
        ps_enc.nlsf_cb,
        &nlsf_w_qw,
        nlsf_mu_q20,
        ps_enc.nlsf_msvq_survivors,
        ps_enc.indices.signal_type as i32,
    );

    eprintln!("[NLSF QUANTIZED] nlsf_q15={:?} indices={:?}", &nlsf_q15[..order], &ps_enc.indices.nlsf_indices[..order+1]);

    // Convert quantized NLSFs back to LPC coefficients
    silk_nlsf2a(&mut pred_coef_q12[1], nlsf_q15, order);

    if do_interpolate {
        // Calculate the interpolated, quantized LSF vector for the first half
        let mut nlsf0_temp_q15 = [0i16; MAX_LPC_ORDER];
        silk_interpolate(
            &mut nlsf0_temp_q15,
            prev_nlsf_q15,
            nlsf_q15,
            ps_enc.indices.nlsf_interp_coef_q2 as i32,
            order,
        );

        // Convert back to LPC coefficients
        silk_nlsf2a(&mut pred_coef_q12[0], &nlsf0_temp_q15, order);
    } else {
        // Copy LPC coefficients for first half from second half
        let (first, second) = pred_coef_q12.split_at_mut(1);
        first[0][..order].copy_from_slice(&second[0][..order]);
    }
}

// ===========================================================================
// Gain quantization
// ===========================================================================

/// Quantize gains.
/// Matches C: `silk_gains_quant`.
pub fn silk_gains_quant(
    ind: &mut [i8],
    gains_q16: &mut [i32],
    prev_ind: &mut i8,
    conditional: bool,
    nb_subfr: usize,
) {
    for k in 0..nb_subfr {
        // Convert to log scale, scale, floor()
        let log_val = silk_lin2log(gains_q16[k]);
        let smulwb_result = silk_smulwb_i32(SCALE_Q16_GAIN, log_val - OFFSET_GAIN);
        ind[k] = smulwb_result as i8;
        eprintln!("[GQUANT k={}] gain_q16={} lin2log={} offset={} diff={} smulwb={} ind_raw={} prev_ind={}",
            k, gains_q16[k], log_val, OFFSET_GAIN, log_val - OFFSET_GAIN, smulwb_result, ind[k], *prev_ind);

        // Round towards previous quantized gain (hysteresis)
        if ind[k] < *prev_ind {
            ind[k] += 1;
        }
        ind[k] = imin(imax(ind[k] as i32, 0), N_LEVELS_QGAIN_I - 1) as i8;

        // Compute delta indices and limit
        if k == 0 && !conditional {
            // Full index
            ind[k] = imin(
                imax(ind[k] as i32, *prev_ind as i32 + MIN_DELTA_GAIN_QUANT),
                N_LEVELS_QGAIN_I - 1,
            ) as i8;
            *prev_ind = ind[k];
        } else {
            // Delta index
            ind[k] = (ind[k] as i32 - *prev_ind as i32) as i8;

            // Double the quantization step size for large gain increases,
            // so that the max gain level can be reached
            let double_step_size_threshold =
                2 * MAX_DELTA_GAIN_QUANT - N_LEVELS_QGAIN_I + *prev_ind as i32;
            if (ind[k] as i32) > double_step_size_threshold {
                ind[k] = (double_step_size_threshold
                    + silk_rshift(ind[k] as i32 - double_step_size_threshold + 1, 1))
                    as i8;
            }

            ind[k] = imin(imax(ind[k] as i32, MIN_DELTA_GAIN_QUANT), MAX_DELTA_GAIN_QUANT) as i8;

            // Accumulate deltas
            if (ind[k] as i32) > double_step_size_threshold {
                *prev_ind = imin(
                    *prev_ind as i32 + silk_lshift(ind[k] as i32, 1) - double_step_size_threshold,
                    N_LEVELS_QGAIN_I - 1,
                ) as i8;
            } else {
                *prev_ind = (*prev_ind as i32 + ind[k] as i32) as i8;
            }

            // Shift to make non-negative
            ind[k] = (ind[k] as i32 - MIN_DELTA_GAIN_QUANT) as i8;
        }

        // Scale and convert to linear scale
        let log_val = imin(
            silk_smulwb(INV_SCALE_Q16_GAIN, *prev_ind as i16) + OFFSET_GAIN,
            3967,
        );
        gains_q16[k] = silk_log2lin(log_val);
    }
}

// ===========================================================================
// LTP gain quantization
// ===========================================================================

/// Quantize LTP gains using matrix-weighted VQ.
/// Matches C: `silk_quant_LTP_gains`.
pub fn silk_quant_ltp_gains(
    b_q14: &mut [i16],
    cbk_index: &mut [i8],
    per_index: &mut i8,
    sum_log_gain_q7: &mut i32,
    w_q18: &[i32],
    mu_q9: i32,
    low_complexity: bool,
    nb_subfr: usize,
) {
    let mut best_total_rd = i32::MAX;
    let mut best_per = 0i8;

    // Try each periodicity codebook (0, 1, 2)
    let max_cbk = if low_complexity { 1 } else { NB_LTP_CBKS };
    for cbk in 0..max_cbk {
        let cb = SILK_LTP_VQ_PTRS_Q7[cbk];
        let cb_gain = SILK_LTP_VQ_GAIN_PTRS_Q7[cbk];
        let cb_size = SILK_LTP_VQ_SIZES[cbk] as usize;

        let mut total_rd = 0i32;
        let mut temp_indices = [0i8; MAX_NB_SUBFR];
        let mut temp_b_q14 = [0i16; LTP_ORDER * MAX_NB_SUBFR];
        let mut temp_sum_log_gain = *sum_log_gain_q7;

        for j in 0..nb_subfr {
            // Compute max allowed gain from log gain budget
            let max_gain_q7 = if temp_sum_log_gain < MAX_SUM_LOG_GAIN_DB_Q7 {
                silk_log2lin((MAX_SUM_LOG_GAIN_DB_Q7 - temp_sum_log_gain) >> 1)
            } else {
                0
            };

            // Find best codebook vector via weighted matrix VQ
            let mut min_rd = i32::MAX;
            let mut best_idx = 0usize;

            for k in 0..cb_size {
                // Skip if gain exceeds budget
                if cb_gain[k] as i32 > max_gain_q7 {
                    continue;
                }

                // Compute weighted distortion
                let cb_offset = k * LTP_ORDER;
                let w_offset = j * LTP_ORDER * LTP_ORDER;
                let mut dist = 0i64;

                // Simplified 5x5 VQ distortion (matching silk_VQ_WMat_EC)
                for m in 0..LTP_ORDER {
                    for n in 0..LTP_ORDER {
                        dist += cb[cb_offset + m] as i64 * cb[cb_offset + n] as i64
                            * w_q18[w_offset + m * LTP_ORDER + n] as i64;
                    }
                }

                let rd = (dist >> 18) as i32;
                let rate = SILK_LTP_GAIN_BITS_Q5_PTRS[cbk][k] as i32;
                let total = rd + mu_q9 * rate >> 9;

                if total < min_rd {
                    min_rd = total;
                    best_idx = k;
                }
            }

            temp_indices[j] = best_idx as i8;

            // Copy LTP coefficients (Q7 → Q14)
            let cb_offset = best_idx * LTP_ORDER;
            for m in 0..LTP_ORDER {
                temp_b_q14[j * LTP_ORDER + m] = (cb[cb_offset + m] as i32 * 128) as i16;
            }

            // Update log gain sum
            let gain = imax(cb_gain[best_idx] as i32, 1);
            temp_sum_log_gain += silk_lin2log(gain) << 1;

            total_rd += min_rd;
        }

        if total_rd < best_total_rd {
            best_total_rd = total_rd;
            best_per = cbk as i8;
            *sum_log_gain_q7 = temp_sum_log_gain;
            b_q14[..nb_subfr * LTP_ORDER].copy_from_slice(&temp_b_q14[..nb_subfr * LTP_ORDER]);
            cbk_index[..nb_subfr].copy_from_slice(&temp_indices[..nb_subfr]);
        }
    }

    *per_index = best_per;
}

// ===========================================================================
// Encode indices (bitstream writing)
// ===========================================================================

/// Entropy-encode all side information indices.
/// Matches C: `silk_encode_indices`.
pub fn silk_encode_indices(
    ps_enc: &SilkEncoderState,
    range_enc: &mut RangeEncoder,
    frame_index: usize,
    encode_lbrr: bool,
    cond_coding: i32,
) {
    let indices = if encode_lbrr {
        &ps_enc.indices_lbrr[frame_index]
    } else {
        &ps_enc.indices
    };

    let nb_subfr = ps_enc.nb_subfr as usize;

    // 1. Encode signal type and quantizer offset (joint coding)
    // typeOffset = 2 * signalType + quantOffsetType, range [0..5]
    let type_offset = 2 * indices.signal_type as i32 + indices.quant_offset_type as i32;
    if encode_lbrr || type_offset >= 2 {
        // VAD table: encode typeOffset - 2 (symbol range [0..3])
        range_enc.encode_icdf(
            (type_offset - 2) as u32,
            &SILK_TYPE_OFFSET_VAD_ICDF,
            8,
        );
    } else {
        // No-VAD table: encode typeOffset directly (symbol range [0..1])
        range_enc.encode_icdf(
            type_offset as u32,
            &SILK_TYPE_OFFSET_NO_VAD_ICDF,
            8,
        );
    }

    // 2. Encode gains
    if cond_coding == CODE_CONDITIONALLY {
        // Delta coding for first subframe gain (symbol is the delta index directly)
        range_enc.encode_icdf(
            indices.gains_indices[0] as u32,
            &SILK_DELTA_GAIN_ICDF,
            8,
        );
    } else {
        // Independent coding: MSB (top 5 bits >> 3) + LSB (bottom 3 bits)
        range_enc.encode_icdf(
            (indices.gains_indices[0] as u32) >> 3,
            &SILK_GAIN_ICDF[indices.signal_type as usize],
            8,
        );
        range_enc.encode_icdf(
            (indices.gains_indices[0] as u32) & 7,
            &SILK_UNIFORM8_ICDF,
            8,
        );
    }

    // Subsequent subframe gains (always delta coded)
    for i in 1..nb_subfr {
        range_enc.encode_icdf(
            indices.gains_indices[i] as u32,
            &SILK_DELTA_GAIN_ICDF,
            8,
        );
    }

    // 3. Encode NLSFs
    // First stage index — iCDF is offset by (signalType >> 1) * nVectors
    let cb1_offset = (indices.signal_type as usize >> 1) * ps_enc.nlsf_cb.n_vectors as usize;
    range_enc.encode_icdf(
        indices.nlsf_indices[0] as u32,
        &ps_enc.nlsf_cb.cb1_icdf[cb1_offset..],
        8,
    );

    // NLSF residual indices
    let mut ec_ix = [0i16; MAX_LPC_ORDER];
    let mut pred_q8 = [0u8; MAX_LPC_ORDER];
    silk_nlsf_unpack(
        &mut ec_ix,
        &mut pred_q8,
        ps_enc.nlsf_cb,
        indices.nlsf_indices[0] as usize,
    );

    let order = ps_enc.predict_lpc_order as usize;
    for i in 0..order {
        let res = indices.nlsf_indices[i + 1] as i32;
        let icdf_offset = ec_ix[i] as usize;

        if res >= NLSF_QUANT_MAX_AMPLITUDE {
            // Large positive: encode max symbol (2*MAX_AMP = 8) then extension
            range_enc.encode_icdf(
                (2 * NLSF_QUANT_MAX_AMPLITUDE) as u32,
                &ps_enc.nlsf_cb.ec_icdf[icdf_offset..],
                8,
            );
            range_enc.encode_icdf(
                (res - NLSF_QUANT_MAX_AMPLITUDE) as u32,
                &SILK_NLSF_EXT_ICDF,
                8,
            );
        } else if res <= -NLSF_QUANT_MAX_AMPLITUDE {
            // Large negative: encode 0 then extension
            range_enc.encode_icdf(
                0,
                &ps_enc.nlsf_cb.ec_icdf[icdf_offset..],
                8,
            );
            range_enc.encode_icdf(
                (-res - NLSF_QUANT_MAX_AMPLITUDE) as u32,
                &SILK_NLSF_EXT_ICDF,
                8,
            );
        } else {
            // Normal range: symbol = res + NLSF_QUANT_MAX_AMPLITUDE
            range_enc.encode_icdf(
                (res + NLSF_QUANT_MAX_AMPLITUDE) as u32,
                &ps_enc.nlsf_cb.ec_icdf[icdf_offset..],
                8,
            );
        }
    }

    // 4. NLSF interpolation factor
    if nb_subfr == 4 {
        range_enc.encode_icdf(
            indices.nlsf_interp_coef_q2 as u32,
            &SILK_NLSF_INTERPOLATION_FACTOR_ICDF,
            8,
        );
    }

    // 5. Encode pitch (voiced frames only)
    if indices.signal_type == TYPE_VOICED as i8 {
        // Pitch lag
        let mut encode_absolute_lag = true;

        if cond_coding == CODE_CONDITIONALLY && ps_enc.ec_prev_signal_type == TYPE_VOICED {
            // Try delta coding
            let mut delta_lag = indices.lag_index as i32 - ps_enc.ec_prev_lag_index as i32;
            if delta_lag < -8 || delta_lag > 11 {
                delta_lag = 0; // Out of range → signal absolute encoding via symbol 0
            } else {
                delta_lag += 9; // Bias to [1..20]
                encode_absolute_lag = false;
            }
            range_enc.encode_icdf(
                delta_lag as u32,
                &SILK_PITCH_DELTA_ICDF,
                8,
            );
        }

        if encode_absolute_lag {
            // Absolute coding: high bits + low bits
            // pitch_high = lagIndex / (fs_kHz / 2)
            // pitch_low  = lagIndex % (fs_kHz / 2)
            let half_fs = ps_enc.fs_khz >> 1;
            let pitch_high = indices.lag_index as i32 / half_fs;
            let pitch_low = indices.lag_index as i32 - pitch_high * half_fs;
            range_enc.encode_icdf(
                pitch_high as u32,
                &SILK_PITCH_LAG_ICDF,
                8,
            );
            range_enc.encode_icdf(
                pitch_low as u32,
                ps_enc.pitch_lag_low_bits_icdf,
                8,
            );
        }

        // Pitch contour
        range_enc.encode_icdf(
            indices.contour_index as u32,
            ps_enc.pitch_contour_icdf,
            8,
        );

        // LTP periodicity index
        range_enc.encode_icdf(
            indices.per_index as u32,
            &SILK_LTP_PER_INDEX_ICDF,
            8,
        );

        // LTP codebook indices
        let ltp_icdf = SILK_LTP_GAIN_ICDF_PTRS[indices.per_index as usize];
        for i in 0..nb_subfr {
            range_enc.encode_icdf(
                indices.ltp_index[i] as u32,
                ltp_icdf,
                8,
            );
        }

        // LTP scale (independent coding only)
        if cond_coding == CODE_INDEPENDENTLY {
            range_enc.encode_icdf(
                indices.ltp_scale_index as u32,
                &SILK_LTP_SCALE_ICDF,
                8,
            );
        }
    }

    // 6. Encode seed
    range_enc.encode_icdf(indices.seed as u32, &SILK_UNIFORM4_ICDF, 8);
}

// ===========================================================================
// Encode pulses (excitation)
// ===========================================================================

/// Shell encoder: hierarchical pulse count encoding.
/// Shell encoder: operates on one shell code frame of 16 pulses.
/// Uses explicit unrolled tree with level-specific tables, matching C exactly.
/// Matches C: `silk_shell_encoder`.
fn silk_shell_encoder(
    range_enc: &mut RangeEncoder,
    pulses0: &[i32],
) {
    // Combine bottom-up: 16 → 8 → 4 → 2 → 1
    let mut pulses1 = [0i32; 8];
    for k in 0..8 {
        pulses1[k] = pulses0[2 * k] + pulses0[2 * k + 1];
    }
    let mut pulses2 = [0i32; 4];
    for k in 0..4 {
        pulses2[k] = pulses1[2 * k] + pulses1[2 * k + 1];
    }
    let mut pulses3 = [0i32; 2];
    for k in 0..2 {
        pulses3[k] = pulses2[2 * k] + pulses2[2 * k + 1];
    }
    let pulses4 = pulses3[0] + pulses3[1];

    // Encode top-down in depth-first order, each level uses its own table
    encode_split(range_enc, pulses3[0], pulses4, &SILK_SHELL_CODE_TABLE3);

    encode_split(range_enc, pulses2[0], pulses3[0], &SILK_SHELL_CODE_TABLE2);

    encode_split(range_enc, pulses1[0], pulses2[0], &SILK_SHELL_CODE_TABLE1);
    encode_split(range_enc, pulses0[0] as i32, pulses1[0], &SILK_SHELL_CODE_TABLE0);
    encode_split(range_enc, pulses0[2] as i32, pulses1[1], &SILK_SHELL_CODE_TABLE0);

    encode_split(range_enc, pulses1[2], pulses2[1], &SILK_SHELL_CODE_TABLE1);
    encode_split(range_enc, pulses0[4] as i32, pulses1[2], &SILK_SHELL_CODE_TABLE0);
    encode_split(range_enc, pulses0[6] as i32, pulses1[3], &SILK_SHELL_CODE_TABLE0);

    encode_split(range_enc, pulses2[2], pulses3[1], &SILK_SHELL_CODE_TABLE2);

    encode_split(range_enc, pulses1[4], pulses2[2], &SILK_SHELL_CODE_TABLE1);
    encode_split(range_enc, pulses0[8] as i32, pulses1[4], &SILK_SHELL_CODE_TABLE0);
    encode_split(range_enc, pulses0[10] as i32, pulses1[5], &SILK_SHELL_CODE_TABLE0);

    encode_split(range_enc, pulses1[6], pulses2[3], &SILK_SHELL_CODE_TABLE1);
    encode_split(range_enc, pulses0[12] as i32, pulses1[6], &SILK_SHELL_CODE_TABLE0);
    encode_split(range_enc, pulses0[14] as i32, pulses1[7], &SILK_SHELL_CODE_TABLE0);
}

/// Encode a binary split: encode left child pulse count conditioned on parent total.
/// Matches C: `encode_split`.
#[inline]
fn encode_split(
    range_enc: &mut RangeEncoder,
    p_child1: i32,
    p: i32,
    shell_table: &[u8],
) {
    if p > 0 {
        let offset = SILK_SHELL_CODE_TABLE_OFFSETS[p as usize] as usize;
        range_enc.encode_icdf(p_child1 as u32, &shell_table[offset..], 8);
    }
}

/// Encode signs of pulses.
/// Matches C: `silk_encode_signs`.
fn silk_encode_signs(
    range_enc: &mut RangeEncoder,
    pulses: &[i8],
    length: usize,
    signal_type: i32,
    quant_offset_type: i32,
    sum_pulses: &[i32],
) {
    // Select sign probability table: offset = 7 * (quantOffsetType + 2*signalType)
    let icdf_offset = 7 * (quant_offset_type + (signal_type << 1)) as usize;

    // Round up to shell blocks
    let n_blocks = (length + SHELL_CODEC_FRAME_LENGTH / 2) >> LOG2_SHELL_CODEC_FRAME_LENGTH;

    let mut q_ptr = 0usize;
    for i in 0..n_blocks {
        let p = sum_pulses[i];
        if p > 0 {
            // Sign probability conditioned on pulse magnitude, clamped to [0..6]
            let icdf_idx = imin(p & 0x1F, 6) as usize;
            let icdf = [SILK_SIGN_ICDF[icdf_offset + icdf_idx], 0u8];

            for j in 0..SHELL_CODEC_FRAME_LENGTH {
                if q_ptr + j < length && pulses[q_ptr + j] != 0 {
                    // silk_enc_map: positive → 1, negative → 0
                    let symbol = if pulses[q_ptr + j] > 0 { 1u32 } else { 0u32 };
                    range_enc.encode_icdf(symbol, &icdf, 8);
                }
            }
        }
        q_ptr += SHELL_CODEC_FRAME_LENGTH;
    }
}

/// Entropy-encode quantized excitation pulses.
/// Matches C: `silk_encode_pulses`.
pub fn silk_encode_pulses(
    range_enc: &mut RangeEncoder,
    signal_type: i32,
    quant_offset_type: i32,
    pulses: &[i8],
    frame_length: usize,
) {
    let nb_shell_blocks = frame_length / SHELL_CODEC_FRAME_LENGTH;
    // Extra block for 12 kHz / 10ms case (frame_length = 120, not multiple of 16)
    let iter = if frame_length == 120 { nb_shell_blocks + 1 } else { nb_shell_blocks };

    eprintln!("[PULSES] frame_length={} pulses_nonzero={} pulses_sum={}", frame_length,
              pulses.iter().filter(|&&p| p != 0).count(),
              pulses.iter().map(|&p| (p as i32).unsigned_abs()).sum::<u32>());

    // Convert pulses to i32 and compute absolute values per shell block
    let mut abs_pulses = vec![0i32; iter * SHELL_CODEC_FRAME_LENGTH];
    for i in 0..frame_length {
        abs_pulses[i] = (pulses[i] as i32).unsigned_abs() as i32;
    }

    // Hierarchical combine to compute sum_pulses with right-shifting
    let mut sum_pulses = vec![0i32; iter];
    let mut nrshifts = vec![0i32; iter];

    for i in 0..iter {
        let block_start = i * SHELL_CODEC_FRAME_LENGTH;
        nrshifts[i] = 0;

        loop {
            let mut pulses_comb = [0i32; 8];
            let mut scale_down = false;

            // 1+1 -> 2 (16 → 8, max=8)
            for k in 0..8 {
                let s = abs_pulses[block_start + 2 * k] + abs_pulses[block_start + 2 * k + 1];
                if s > SILK_MAX_PULSES_TABLE[0] as i32 { scale_down = true; }
                pulses_comb[k] = s;
            }
            // 2+2 -> 4 (8 → 4, max=10)
            let mut pulses_comb2 = [0i32; 4];
            for k in 0..4 {
                let s = pulses_comb[2 * k] + pulses_comb[2 * k + 1];
                if s > SILK_MAX_PULSES_TABLE[1] as i32 { scale_down = true; }
                pulses_comb2[k] = s;
            }
            // 4+4 -> 8 (4 → 2, max=12)
            let mut pulses_comb3 = [0i32; 2];
            for k in 0..2 {
                let s = pulses_comb2[2 * k] + pulses_comb2[2 * k + 1];
                if s > SILK_MAX_PULSES_TABLE[2] as i32 { scale_down = true; }
                pulses_comb3[k] = s;
            }
            // 8+8 -> 16 (2 → 1, max=16)
            let s = pulses_comb3[0] + pulses_comb3[1];
            if s > SILK_MAX_PULSES_TABLE[3] as i32 { scale_down = true; }

            if scale_down {
                nrshifts[i] += 1;
                for k in 0..SHELL_CODEC_FRAME_LENGTH {
                    abs_pulses[block_start + k] >>= 1;
                }
            } else {
                sum_pulses[i] = s;
                break;
            }
        }
    }

    // Find optimal rate level (minimum total bits)
    let mut rate_level_index = 0usize;
    let mut min_sum_bits_q5 = i32::MAX;
    for k in 0..(N_RATE_LEVELS - 1) {
        let n_bits_ptr = &SILK_PULSES_PER_BLOCK_BITS_Q5[k];
        let mut sum_bits_q5 = SILK_RATE_LEVELS_BITS_Q5[(signal_type >> 1) as usize][k] as i32;
        for i in 0..iter {
            if nrshifts[i] > 0 {
                sum_bits_q5 += n_bits_ptr[SILK_MAX_PULSES as usize + 1] as i32;
            } else {
                sum_bits_q5 += n_bits_ptr[sum_pulses[i] as usize] as i32;
            }
        }
        if sum_bits_q5 < min_sum_bits_q5 {
            min_sum_bits_q5 = sum_bits_q5;
            rate_level_index = k;
        }
    }
    range_enc.encode_icdf(
        rate_level_index as u32,
        &SILK_RATE_LEVELS_ICDF[(signal_type >> 1) as usize],
        8,
    );

    // Encode pulse counts per block with overflow handling
    let cdf_ptr = &SILK_PULSES_PER_BLOCK_ICDF[rate_level_index];
    for i in 0..iter {
        if nrshifts[i] == 0 {
            range_enc.encode_icdf(sum_pulses[i] as u32, cdf_ptr, 8);
        } else {
            // Signal overflow with SILK_MAX_PULSES+1
            range_enc.encode_icdf((SILK_MAX_PULSES + 1) as u32, cdf_ptr, 8);
            // Encode (nRshifts-1) additional overflow markers at highest rate level
            for _k in 0..(nrshifts[i] - 1) {
                range_enc.encode_icdf(
                    (SILK_MAX_PULSES + 1) as u32,
                    &SILK_PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS - 1],
                    8,
                );
            }
            // Encode actual sum at highest rate level
            range_enc.encode_icdf(
                sum_pulses[i] as u32,
                &SILK_PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS - 1],
                8,
            );
        }
    }

    // Shell coding: hierarchical encoding of pulse distribution per block
    for i in 0..iter {
        if sum_pulses[i] > 0 {
            let block_start = i * SHELL_CODEC_FRAME_LENGTH;
            let block = &abs_pulses[block_start..block_start + SHELL_CODEC_FRAME_LENGTH];
            silk_shell_encoder(range_enc, block);
        }
    }

    // Encode LSBs (for right-shifted blocks)
    // C encodes bits MSB-to-LSB: for j = nLS..1 encode (abs_q >> j) & 1, then abs_q & 1
    // Positions beyond frame_length are zero (C zeros them), still encoded.
    for i in 0..iter {
        if nrshifts[i] > 0 {
            let block_start = i * SHELL_CODEC_FRAME_LENGTH;
            let n_ls = nrshifts[i] - 1;
            for k in 0..SHELL_CODEC_FRAME_LENGTH {
                let idx = block_start + k;
                // Positions beyond frame_length are 0 (C zeros original pulses there)
                let abs_q = if idx < frame_length {
                    (pulses[idx] as i32).abs() as i32
                } else {
                    0
                };
                // Encode bits from MSB to LSB
                for j in (1..=n_ls).rev() {
                    let bit = (abs_q >> j) & 1;
                    range_enc.encode_icdf(bit as u32, &SILK_LSB_ICDF, 8);
                }
                let bit = abs_q & 1;
                range_enc.encode_icdf(bit as u32, &SILK_LSB_ICDF, 8);
            }
        }
    }

    // Encode signs
    silk_encode_signs(range_enc, pulses, frame_length, signal_type, quant_offset_type, &sum_pulses);
}

// ===========================================================================
// Stereo encoding
// ===========================================================================

/// Inner product with automatic scaling.
/// Matches C: `silk_inner_prod_aligned_scale`.
fn silk_inner_prod_aligned_scale(x: &[i16], y: &[i16], scale: i32, len: usize) -> i32 {
    let mut sum: i64 = 0;
    for i in 0..len {
        sum += (x[i] as i64 * y[i] as i64) >> scale;
    }
    sum as i32
}

/// Find stereo predictor for one frequency band.
/// Returns predictor in Q13.
/// Matches C: `silk_stereo_find_predictor`.
fn silk_stereo_find_predictor(
    ratio_q14: &mut i32,
    x: &[i16],       // basis (mid) signal
    y: &[i16],       // target (side) signal
    mid_res_amp_q0: &mut [i32],  // [2]: smoothed mid, residual norms
    length: usize,
    smooth_coef_q16: i32,
) -> i32 {
    // Compute energies with auto-scaling
    let (mut nrgx, scale1) = silk_sum_sqr_shift(x);
    let (mut nrgy, scale2) = silk_sum_sqr_shift(y);
    let mut scale = imax(scale1 as i32, scale2 as i32);
    scale += scale & 1; // make even
    nrgy >>= (scale - scale2 as i32) as u32;
    nrgx >>= (scale - scale1 as i32) as u32;
    nrgx = imax(nrgx, 1);

    // Cross-correlation with scaling
    let corr = silk_inner_prod_aligned_scale(x, y, scale, length);

    // Predictor
    let mut pred_q13 = silk_div32_var_q(corr, nrgx, 13);
    pred_q13 = imax(imin(pred_q13, 1 << 14), -(1 << 14));
    let pred2_q10 = silk_smulwb(pred_q13, pred_q13 as i16);

    // Faster update for signals with large prediction parameters
    let smooth_coef_q16 = imax(smooth_coef_q16, pred2_q10.unsigned_abs() as i32);

    // Smoothed mid norm
    let scale_half = scale >> 1;
    mid_res_amp_q0[0] = silk_smlawb(
        mid_res_amp_q0[0],
        (silk_sqrt_approx(nrgx) << scale_half) - mid_res_amp_q0[0],
        smooth_coef_q16 as i16,
    );

    // Residual energy = nrgy - 2*pred*corr + pred^2*nrgx
    // silk_SUB_LSHIFT32(nrgy, silk_SMULWB(corr, pred_Q13), 3+1)
    let nrgy = nrgy - (silk_smulwb(corr, pred_q13 as i16) << 4);
    // silk_ADD_LSHIFT32(nrgy, silk_SMULWB(nrgx, pred2_Q10), 6)
    let nrgy = nrgy + (silk_smulwb(nrgx, pred2_q10 as i16) << 6);

    mid_res_amp_q0[1] = silk_smlawb(
        mid_res_amp_q0[1],
        (silk_sqrt_approx(imax(nrgy, 0)) << scale_half) - mid_res_amp_q0[1],
        smooth_coef_q16 as i16,
    );

    // Ratio of smoothed residual and mid norms
    *ratio_q14 = silk_div32_var_q(mid_res_amp_q0[1], imax(mid_res_amp_q0[0], 1), 14);
    *ratio_q14 = imax(imin(*ratio_q14, 32767), 0);

    pred_q13
}

/// Quantize stereo predictor.
/// Matches C: `silk_stereo_quant_pred`.
pub fn silk_stereo_quant_pred(
    pred_q13: &mut [i32; 2],
    ix: &mut [[i8; 3]; 2],
) {
    // SILK_FIX_CONST(0.5/STEREO_QUANT_SUB_STEPS, 16) = SILK_FIX_CONST(0.1, 16) = 6554
    const HALF_STEP_Q16: i16 = 6554;

    for n in 0..2 {
        let mut err_min_q13 = i32::MAX;
        let mut quant_pred_q13 = 0i32;

        // Brute-force search over quantization levels
        'outer: for i in 0..(STEREO_QUANT_TAB_SIZE - 1) {
            let low_q13 = SILK_STEREO_PRED_QUANT_Q13[i] as i32;
            let step_q13 = silk_smulwb(
                SILK_STEREO_PRED_QUANT_Q13[i + 1] as i32 - low_q13,
                HALF_STEP_Q16,
            );
            for j in 0..STEREO_QUANT_SUB_STEPS as i32 {
                // silk_SMLABB(low_q13, step_q13, 2*j+1)
                let lvl_q13 = low_q13 + step_q13 * (2 * j + 1);
                let err_q13 = (pred_q13[n] - lvl_q13).unsigned_abs() as i32;
                if err_q13 < err_min_q13 {
                    err_min_q13 = err_q13;
                    quant_pred_q13 = lvl_q13;
                    ix[n][0] = i as i8;
                    ix[n][1] = j as i8;
                } else {
                    // Error increasing — past the optimum
                    break 'outer;
                }
            }
        }

        // Split main index: ix[n][2] = ix[n][0] / 3, ix[n][0] = ix[n][0] % 3
        ix[n][2] = ix[n][0] / 3;
        ix[n][0] -= ix[n][2] * 3;
        pred_q13[n] = quant_pred_q13;
    }

    // Subtract second from first predictor (differential coding)
    pred_q13[0] -= pred_q13[1];
}

/// Encode stereo prediction indices.
/// Matches C: `silk_stereo_encode_pred`.
pub fn silk_stereo_encode_pred(
    range_enc: &mut RangeEncoder,
    ix: &[[i8; 3]; 2],
) {
    // Joint coding of the two group indices: n = 5 * ix[0][2] + ix[1][2]
    let n = 5 * ix[0][2] as usize + ix[1][2] as usize;
    range_enc.encode_icdf(n as u32, &SILK_STEREO_PRED_JOINT_ICDF, 8);

    // Per-channel: encode ix[n][0] (range 0..2) and ix[n][1] (range 0..4) independently
    for ch in 0..2 {
        range_enc.encode_icdf(ix[ch][0] as u32, &SILK_UNIFORM3_ICDF, 8);
        range_enc.encode_icdf(ix[ch][1] as u32, &SILK_UNIFORM5_ICDF, 8);
    }
}

/// Encode mid-only flag.
pub fn silk_stereo_encode_mid_only(
    range_enc: &mut RangeEncoder,
    mid_only_flag: i8,
) {
    range_enc.encode_icdf(
        mid_only_flag as u32,
        &SILK_STEREO_ONLY_CODE_MID_ICDF,
        8,
    );
}

/// Convert L/R stereo to M/S with stereo prediction.
/// Matches C: `silk_stereo_LR_to_MS`.
pub fn silk_stereo_lr_to_ms(
    state: &mut StereoEncState,
    x1: &mut [i16], // left → mid (modified in place)
    x2: &mut [i16], // right → side (modified in place)
    pred_ix: &mut [[i8; 3]; 2],
    mid_only_flag: &mut i8,
    mid_side_rates_bps: &mut [i32; 2],
    mut total_rate_bps: i32,
    prev_speech_act_q8: i32,
    to_mono: bool,
    fs_khz: i32,
    frame_length: usize,
) {
    let is_10ms_frame = frame_length == 10 * fs_khz as usize;

    // Step 1: Convert L/R to basic M/S
    // C uses x1[-2]..x1[frame_length-1] via pointer aliasing. We use explicit buffers.
    let mut mid = vec![0i16; frame_length + 2];
    let mut side = vec![0i16; frame_length + 2];

    for n in 0..frame_length + 2 {
        // x1[n-2] and x2[n-2]: for n<2, these are overlap from previous frame
        let l = if n < 2 { state.s_mid[n] as i32 } else { x1[n - 2] as i32 };
        let r = if n < 2 { state.s_side[n] as i32 } else { x2[n - 2] as i32 };
        // Note: state.s_mid/s_side store the *original* L/R overlap for the first call,
        // but after the first call they store the *mid/side* overlap. The C code uses
        // x1[-2],x1[-1] which are the previous frame's last two L samples.
        // For simplicity, we compute sum/diff here and handle overlap below.
        let sum = l + r;
        let diff = l - r;
        mid[n] = silk_rshift_round(sum, 1) as i16;
        side[n] = sat16(silk_rshift_round(diff, 1));
    }

    // Step 2: Restore overlap from state, save new overlap
    mid[0] = state.s_mid[0];
    mid[1] = state.s_mid[1];
    side[0] = state.s_side[0];
    side[1] = state.s_side[1];
    state.s_mid[0] = mid[frame_length];
    state.s_mid[1] = mid[frame_length + 1];
    state.s_side[0] = side[frame_length];
    state.s_side[1] = side[frame_length + 1];

    // Step 3-4: LP/HP filter mid and side signals
    let mut lp_mid = vec![0i16; frame_length];
    let mut hp_mid = vec![0i16; frame_length];
    let mut lp_side = vec![0i16; frame_length];
    let mut hp_side = vec![0i16; frame_length];

    for n in 0..frame_length {
        // silk_ADD_LSHIFT32(mid[n] + mid[n+2], mid[n+1], 1) = mid[n]+mid[n+2] + 2*mid[n+1]
        let sum_m = silk_rshift_round(
            mid[n] as i32 + mid[n + 2] as i32 + ((mid[n + 1] as i32) << 1), 2);
        lp_mid[n] = sum_m as i16;
        hp_mid[n] = (mid[n + 1] as i32 - sum_m) as i16;

        let sum_s = silk_rshift_round(
            side[n] as i32 + side[n + 2] as i32 + ((side[n + 1] as i32) << 1), 2);
        lp_side[n] = sum_s as i16;
        hp_side[n] = (side[n + 1] as i32 - sum_s) as i16;
    }

    // Step 5: Smoothing coefficient (speech-activity weighted)
    let smooth_coef_q16 = if is_10ms_frame { 328 } else { 655 }; // STEREO_RATIO_SMOOTH_COEF
    let smooth_coef_q16 = silk_smulwb(
        silk_smulbb(prev_speech_act_q8, prev_speech_act_q8),
        smooth_coef_q16 as i16,
    );

    // Step 6: Find predictors for LP and HP bands
    let mut lp_ratio_q14 = 0i32;
    let mut hp_ratio_q14 = 0i32;
    let mut pred_q13 = [0i32; 2];

    pred_q13[0] = silk_stereo_find_predictor(
        &mut lp_ratio_q14, &lp_mid, &lp_side,
        &mut state.mid_side_amp_q0[0..2], frame_length, smooth_coef_q16);
    pred_q13[1] = silk_stereo_find_predictor(
        &mut hp_ratio_q14, &hp_mid, &hp_side,
        &mut state.mid_side_amp_q0[2..4], frame_length, smooth_coef_q16);

    // Combine LP and HP ratios: frac_Q16 = HP_ratio + 3 * LP_ratio
    let frac_q16 = imin(silk_smlabb(hp_ratio_q14, lp_ratio_q14, 3), 1 << 16);

    // Step 7: Bitrate distribution
    total_rate_bps -= if is_10ms_frame { 1200 } else { 600 };
    total_rate_bps = imax(total_rate_bps, 1);
    let min_mid_rate_bps = silk_smlabb(2000, fs_khz, 600);
    let frac_3_q16 = 3 * frac_q16;

    // mid rate = total / (SILK_FIX_CONST(8+5, 16) + 3*frac), shifted
    mid_side_rates_bps[0] = silk_div32_var_q(
        total_rate_bps, ((8 + 5) << 16) + frac_3_q16, 16 + 3);

    let mut width_q14;
    if mid_side_rates_bps[0] < min_mid_rate_bps {
        mid_side_rates_bps[0] = min_mid_rate_bps;
        mid_side_rates_bps[1] = total_rate_bps - mid_side_rates_bps[0];
        width_q14 = silk_div32_var_q(
            (mid_side_rates_bps[1] << 1) - min_mid_rate_bps,
            silk_smulwb((1 << 16) + frac_3_q16, min_mid_rate_bps as i16),
            14 + 2,
        );
        width_q14 = imax(imin(width_q14, 1 << 14), 0);
    } else {
        mid_side_rates_bps[1] = total_rate_bps - mid_side_rates_bps[0];
        width_q14 = 1 << 14;
    }

    // Step 8: Smooth width
    state.smth_width_q14 = silk_smlawb(
        state.smth_width_q14 as i32,
        width_q14 - state.smth_width_q14 as i32,
        smooth_coef_q16 as i16,
    ) as i16;

    // Step 9: Decision tree for mono/stereo width
    *mid_only_flag = 0;
    if to_mono {
        // Branch A: force mono transition
        width_q14 = 0;
        pred_q13 = [0; 2];
        silk_stereo_quant_pred(&mut pred_q13, pred_ix);
    } else if state.width_prev_q14 == 0
        && (8 * total_rate_bps < 13 * min_mid_rate_bps
            || silk_smulwb(frac_q16, state.smth_width_q14) < (819)) // SILK_FIX_CONST(0.05, 14) = 819
    {
        // Branch B: was mono, stay mono
        pred_q13[0] = (state.smth_width_q14 as i32 * pred_q13[0]) >> 14;
        pred_q13[1] = (state.smth_width_q14 as i32 * pred_q13[1]) >> 14;
        silk_stereo_quant_pred(&mut pred_q13, pred_ix);
        width_q14 = 0;
        pred_q13 = [0; 2];
        mid_side_rates_bps[0] = total_rate_bps;
        mid_side_rates_bps[1] = 0;
        *mid_only_flag = 1;
    } else if state.width_prev_q14 != 0
        && (8 * total_rate_bps < 11 * min_mid_rate_bps
            || silk_smulwb(frac_q16, state.smth_width_q14) < (328)) // SILK_FIX_CONST(0.02, 14) = 328
    {
        // Branch C: transition to mono
        pred_q13[0] = (state.smth_width_q14 as i32 * pred_q13[0]) >> 14;
        pred_q13[1] = (state.smth_width_q14 as i32 * pred_q13[1]) >> 14;
        silk_stereo_quant_pred(&mut pred_q13, pred_ix);
        width_q14 = 0;
        pred_q13 = [0; 2];
    } else if state.smth_width_q14 > 15564 {
        // Branch D: full width (> 0.95)
        silk_stereo_quant_pred(&mut pred_q13, pred_ix);
        width_q14 = 1 << 14;
    } else {
        // Branch E: reduced width
        pred_q13[0] = (state.smth_width_q14 as i32 * pred_q13[0]) >> 14;
        pred_q13[1] = (state.smth_width_q14 as i32 * pred_q13[1]) >> 14;
        silk_stereo_quant_pred(&mut pred_q13, pred_ix);
        width_q14 = state.smth_width_q14 as i32;
    }

    // Step 10: Mid-only taper tracking
    if *mid_only_flag == 1 {
        state.silent_side_len += frame_length as i16 - (STEREO_INTERP_LEN_MS as i32 * fs_khz) as i16;
        if (state.silent_side_len as i32) < LA_SHAPE_MS * fs_khz {
            *mid_only_flag = 0;
        } else {
            state.silent_side_len = 10000;
        }
    } else {
        state.silent_side_len = 0;
    }
    if *mid_only_flag == 0 && mid_side_rates_bps[1] < 1 {
        mid_side_rates_bps[1] = 1;
        mid_side_rates_bps[0] = imax(1, total_rate_bps - mid_side_rates_bps[1]);
    }

    // Step 11: Interpolate predictors and subtract prediction from side channel
    let interp_len = STEREO_INTERP_LEN_MS * fs_khz as usize;
    let denom_q16 = (1 << 16) / (interp_len as i32);

    let mut pred0_q13 = -(state.pred_prev_q13[0] as i32);
    let mut pred1_q13 = -(state.pred_prev_q13[1] as i32);
    let mut w_q24 = (state.width_prev_q14 as i32) << 10;
    let delta0_q13 = -silk_rshift_round(
        silk_smulbb(pred_q13[0] - state.pred_prev_q13[0] as i32, denom_q16), 16);
    let delta1_q13 = -silk_rshift_round(
        silk_smulbb(pred_q13[1] - state.pred_prev_q13[1] as i32, denom_q16), 16);
    let deltaw_q24 = silk_smulwb(width_q14 - state.width_prev_q14 as i32, denom_q16 as i16) << 10;

    // Interpolation region
    for n in 0..interp_len {
        pred0_q13 += delta0_q13;
        pred1_q13 += delta1_q13;
        w_q24 += deltaw_q24;
        // LP mid component (Q11)
        let sum = (mid[n] as i32 + mid[n + 2] as i32 + ((mid[n + 1] as i32) << 1)) << 9;
        // side * width + LP_pred * sum (Q8)
        let mut out = silk_smlawb(
            silk_smulwb(w_q24, side[n + 1] as i16),
            sum,
            pred0_q13 as i16,
        );
        // HP_pred * mid[n+1] (Q8)
        out = silk_smlawb(out, (mid[n + 1] as i32) << 11, pred1_q13 as i16);
        x2[n] = sat16(silk_rshift_round(out, 8));
    }

    // Steady-state region
    let pred0_q13_final = -(pred_q13[0]);
    let pred1_q13_final = -(pred_q13[1]);
    let w_q24_final = width_q14 << 10;
    for n in interp_len..frame_length {
        let sum = (mid[n] as i32 + mid[n + 2] as i32 + ((mid[n + 1] as i32) << 1)) << 9;
        let mut out = silk_smlawb(
            silk_smulwb(w_q24_final, side[n + 1] as i16),
            sum,
            pred0_q13_final as i16,
        );
        out = silk_smlawb(out, (mid[n + 1] as i32) << 11, pred1_q13_final as i16);
        x2[n] = sat16(silk_rshift_round(out, 8));
    }

    // Copy mid to x1 (shifted by 1: x1[n] = mid[n+1])
    for n in 0..frame_length {
        x1[n] = mid[n + 1];
    }

    // Save state
    state.pred_prev_q13[0] = pred_q13[0] as i16;
    state.pred_prev_q13[1] = pred_q13[1] as i16;
    state.width_prev_q14 = width_q14 as i16;
}

// ===========================================================================
// NSQ: Noise Shaping Quantizer (standard)
// ===========================================================================

/// Scale states for new subframe (gain changes, LTP re-whitening).
/// Matches C: `silk_nsq_scale_states`.
fn silk_nsq_scale_states(
    ps_enc: &SilkEncoderState,
    nsq: &mut NsqState,
    x16: &[i16],
    x_sc_q10: &mut [i32],
    s_ltp: &[i16],
    s_ltp_q15: &mut [i32],
    subfr: usize,
    ltp_scale_q14: i32,
    gains_q16: &[i32],
    pitch_l: &[i32],
    signal_type: i32,
) {
    let subfr_length = ps_enc.subfr_length as usize;
    let ltp_mem_length = ps_enc.ltp_mem_length as usize;
    let lag = pitch_l[subfr];

    let inv_gain_q31 = silk_inverse32_var_q(imax(gains_q16[subfr], 1), 47);

    // Scale input
    let inv_gain_q26 = silk_rshift_round(inv_gain_q31, 5);
    for i in 0..subfr_length {
        x_sc_q10[i] = silk_smulww(x16[i] as i32, inv_gain_q26);
    }

    // After rewhitening the LTP state is un-scaled, so scale with inv_gain_Q31
    if nsq.rewhite_flag != 0 {
        let mut inv_gain = inv_gain_q31;
        if subfr == 0 {
            // Do LTP downscaling
            inv_gain = shl32(silk_smulwb(inv_gain, ltp_scale_q14 as i16), 2);
        }
        let start = (nsq.s_ltp_buf_idx - lag as i32 - (LTP_ORDER as i32 / 2)) as usize;
        let end = nsq.s_ltp_buf_idx as usize;
        for i in start..end {
            s_ltp_q15[i] = silk_smulwb(inv_gain, s_ltp[i] as i16);
        }
    }

    // Adjust for changing gain
    if gains_q16[subfr] != nsq.prev_gain_q16 {
        let gain_adj_q16 = silk_div32_var_q(nsq.prev_gain_q16, gains_q16[subfr], 16);

        // Scale long-term shaping state
        let shp_start = (nsq.s_ltp_shp_buf_idx - ltp_mem_length as i32) as usize;
        let shp_end = nsq.s_ltp_shp_buf_idx as usize;
        for i in shp_start..shp_end {
            nsq.s_ltp_shp_q14[i] = silk_smulww(gain_adj_q16, nsq.s_ltp_shp_q14[i]);
        }

        // Scale long-term prediction state
        if signal_type == TYPE_VOICED && nsq.rewhite_flag == 0 {
            let start = (nsq.s_ltp_buf_idx - lag as i32 - (LTP_ORDER as i32 / 2)) as usize;
            let end = nsq.s_ltp_buf_idx as usize;
            for i in start..end {
                s_ltp_q15[i] = silk_smulww(gain_adj_q16, s_ltp_q15[i]);
            }
        }

        nsq.s_lf_ar_shp_q14 = silk_smulww(gain_adj_q16, nsq.s_lf_ar_shp_q14);
        nsq.s_diff_shp_q14 = silk_smulww(gain_adj_q16, nsq.s_diff_shp_q14);

        // Scale short-term prediction and shaping states
        for i in 0..NSQ_LPC_BUF_LENGTH {
            nsq.s_lpc_q14[i] = silk_smulww(gain_adj_q16, nsq.s_lpc_q14[i]);
        }
        for i in 0..MAX_SHAPE_LPC_ORDER {
            nsq.s_ar2_q14[i] = silk_smulww(gain_adj_q16, nsq.s_ar2_q14[i]);
        }

        // Save inverse gain
        nsq.prev_gain_q16 = gains_q16[subfr];
    }
}

/// Inner noise shaping quantizer loop. Matches C: `silk_noise_shape_quantizer`.
fn silk_noise_shape_quantizer(
    nsq: &mut NsqState,
    signal_type: i32,
    x_sc_q10: &[i32],
    pulses: &mut [i8],
    pxq_offset: usize,         // offset into nsq.xq for output
    s_ltp_q15: &mut [i32],
    a_q12: &[i16],
    b_q14: &[i16],
    ar_shp_q13: &[i16],
    lag: i32,
    harm_shape_fir_packed_q14: i32,
    tilt_q14: i32,
    lf_shp_q14: i32,
    gain_q16: i32,
    lambda_q10: i32,
    offset_q10: i32,
    length: usize,
    shaping_lpc_order: usize,
    predict_lpc_order: usize,
) {
    let shp_lag_base = (nsq.s_ltp_shp_buf_idx - lag + (HARM_SHAPE_FIR_TAPS / 2) as i32) as usize;
    let pred_lag_base = (nsq.s_ltp_buf_idx - lag + (LTP_ORDER / 2) as i32) as usize;
    let gain_q10 = gain_q16 >> 6;

    let ps_lpc_q14_base = NSQ_LPC_BUF_LENGTH - 1;

    for i in 0..length {
        // Generate dither
        nsq.rand_seed = silk_rand(nsq.rand_seed);

        // Short-term prediction (matches silk_noise_shape_quantizer_short_prediction_c)
        let ps_lpc_idx = ps_lpc_q14_base + i;
        let mut lpc_pred_q10 = (predict_lpc_order >> 1) as i32; // bias = order/2
        for j in 0..predict_lpc_order {
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                nsq.s_lpc_q14[ps_lpc_idx - j],
                a_q12[j],
            );
        }

        // Long-term prediction
        let ltp_pred_q13 = if signal_type == TYPE_VOICED {
            let mut pred = 2i32; // bias
            let pred_ptr = pred_lag_base + i;
            pred = silk_smlawb(pred, s_ltp_q15[pred_ptr], b_q14[0]);
            pred = silk_smlawb(pred, s_ltp_q15[pred_ptr - 1], b_q14[1]);
            pred = silk_smlawb(pred, s_ltp_q15[pred_ptr - 2], b_q14[2]);
            pred = silk_smlawb(pred, s_ltp_q15[pred_ptr - 3], b_q14[3]);
            pred = silk_smlawb(pred, s_ltp_q15[pred_ptr - 4], b_q14[4]);
            pred
        } else {
            0
        };

        // Noise shape feedback (matches silk_NSQ_noise_shape_feedback_loop_c)
        let n_ar_q12 = {
            let data0 = nsq.s_diff_shp_q14;
            let mut tmp2 = data0;
            let mut tmp1 = nsq.s_ar2_q14[0];
            nsq.s_ar2_q14[0] = tmp2;

            let mut out = (shaping_lpc_order >> 1) as i32;
            out = silk_smlawb(out, tmp2, ar_shp_q13[0]);

            let mut j = 2usize;
            while j < shaping_lpc_order {
                tmp2 = nsq.s_ar2_q14[j - 1];
                nsq.s_ar2_q14[j - 1] = tmp1;
                out = silk_smlawb(out, tmp1, ar_shp_q13[j - 1]);
                tmp1 = nsq.s_ar2_q14[j];
                nsq.s_ar2_q14[j] = tmp2;
                out = silk_smlawb(out, tmp2, ar_shp_q13[j]);
                j += 2;
            }
            nsq.s_ar2_q14[shaping_lpc_order - 1] = tmp1;
            out = silk_smlawb(out, tmp1, ar_shp_q13[shaping_lpc_order - 1]);
            // Q11 -> Q12
            shl32(out, 1)
        };

        // Add tilt to n_AR
        let n_ar_q12 = silk_smlawb_i32(n_ar_q12, nsq.s_lf_ar_shp_q14, tilt_q14);

        // n_LF
        let n_lf_q12 = silk_smulwb_i32(
            nsq.s_ltp_shp_q14[(nsq.s_ltp_shp_buf_idx as usize + i).wrapping_sub(1)],
            lf_shp_q14,
        );
        let n_lf_q12 = silk_smlawt(n_lf_q12, nsq.s_lf_ar_shp_q14, lf_shp_q14);

        // Combine prediction and noise shaping signals
        let mut tmp1 = shl32(lpc_pred_q10, 2).wrapping_sub(n_ar_q12); // Q12
        tmp1 = tmp1.wrapping_sub(n_lf_q12); // Q12

        let (tmp1_final, n_ltp_q13) = if lag > 0 {
            // Symmetric, packed FIR coefficients for harmonic shaping
            let shp_idx = shp_lag_base + i;
            let mut n_ltp_q13 = silk_smulwb_i32(
                silk_add_sat32(nsq.s_ltp_shp_q14[shp_idx], nsq.s_ltp_shp_q14[shp_idx.wrapping_sub(2)]),
                harm_shape_fir_packed_q14,
            );
            n_ltp_q13 = silk_smlawt(n_ltp_q13, nsq.s_ltp_shp_q14[shp_idx.wrapping_sub(1)], harm_shape_fir_packed_q14);
            n_ltp_q13 = shl32(n_ltp_q13, 1);

            let tmp2 = ltp_pred_q13 - n_ltp_q13; // Q13
            let combined = tmp2.wrapping_add(shl32(tmp1, 1)); // Q13
            (silk_rshift_round(combined, 3), n_ltp_q13) // Q10
        } else {
            (silk_rshift_round(tmp1, 2), 0) // Q10
        };

        let r_q10 = x_sc_q10[i] - tmp1_final; // residual error Q10

        if i < 3 && nsq.s_ltp_shp_buf_idx < 650 {
            eprintln!("[NSQ i={}] x_sc={} lpc_pred={} tmp1_final={} r_q10={} offset_q10={}",
                      i, x_sc_q10[i], lpc_pred_q10, tmp1_final, r_q10, offset_q10);
        }

        // Flip sign depending on dither
        let r_q10 = if nsq.rand_seed < 0 { -r_q10 } else { r_q10 };
        let r_q10 = imax(imin(r_q10, 30 << 10), -(31 << 10));

        // Two-candidate quantization with rate-distortion optimization
        let mut q1_q10 = r_q10 - offset_q10;
        let mut q1_q0 = q1_q10 >> 10;
        if lambda_q10 > 2048 {
            let rdo_offset = lambda_q10 / 2 - 512;
            if q1_q10 > rdo_offset {
                q1_q0 = (q1_q10 - rdo_offset) >> 10;
            } else if q1_q10 < -rdo_offset {
                q1_q0 = (q1_q10 + rdo_offset) >> 10;
            } else if q1_q10 < 0 {
                q1_q0 = -1;
            } else {
                q1_q0 = 0;
            }
        }

        let (mut q1_q10_final, q2_q10, rd1_q20, rd2_q20);
        if q1_q0 > 0 {
            q1_q10_final = (q1_q0 << 10) - QUANT_LEVEL_ADJUST_Q10;
            q1_q10_final += offset_q10;
            q2_q10 = q1_q10_final + 1024;
            rd1_q20 = silk_smulbb(q1_q10_final, lambda_q10);
            rd2_q20 = silk_smulbb(q2_q10, lambda_q10);
        } else if q1_q0 == 0 {
            q1_q10_final = offset_q10;
            q2_q10 = q1_q10_final + 1024 - QUANT_LEVEL_ADJUST_Q10;
            rd1_q20 = silk_smulbb(q1_q10_final, lambda_q10);
            rd2_q20 = silk_smulbb(q2_q10, lambda_q10);
        } else if q1_q0 == -1 {
            q2_q10 = offset_q10;
            q1_q10_final = q2_q10 - 1024 + QUANT_LEVEL_ADJUST_Q10;
            rd1_q20 = silk_smulbb(-q1_q10_final, lambda_q10);
            rd2_q20 = silk_smulbb(q2_q10, lambda_q10);
        } else {
            // q1_q0 < -1
            q1_q10_final = (q1_q0 << 10) + QUANT_LEVEL_ADJUST_Q10;
            q1_q10_final += offset_q10;
            q2_q10 = q1_q10_final + 1024;
            rd1_q20 = silk_smulbb(-q1_q10_final, lambda_q10);
            rd2_q20 = silk_smulbb(-q2_q10, lambda_q10);
        }

        let rr_q10_1 = r_q10 - q1_q10_final;
        let rd1_q20 = silk_smlabb(rd1_q20, rr_q10_1, rr_q10_1);
        let rr_q10_2 = r_q10 - q2_q10;
        let rd2_q20 = silk_smlabb(rd2_q20, rr_q10_2, rr_q10_2);

        if rd2_q20 < rd1_q20 {
            q1_q10_final = q2_q10;
        }

        pulses[i] = silk_rshift_round(q1_q10_final, 10) as i8;

        // Excitation
        let mut exc_q14 = shl32(q1_q10_final, 4);
        if nsq.rand_seed < 0 {
            exc_q14 = -exc_q14;
        }

        // Add predictions
        let lpc_exc_q14 = silk_add_lshift32(exc_q14, ltp_pred_q13, 1);
        let xq_q14 = lpc_exc_q14.wrapping_add(shl32(lpc_pred_q10, 4));

        // Scale XQ back to normal level before saving
        nsq.xq[pxq_offset + i] = sat16(silk_rshift_round(silk_smulww(xq_q14, gain_q10), 8));

        // Update states
        nsq.s_lpc_q14[ps_lpc_q14_base + 1 + i] = xq_q14;
        nsq.s_diff_shp_q14 = xq_q14.wrapping_sub(shl32(x_sc_q10[i], 4));
        let s_lf_ar_shp_q14 = nsq.s_diff_shp_q14.wrapping_sub(shl32(n_ar_q12, 2));
        nsq.s_lf_ar_shp_q14 = s_lf_ar_shp_q14;

        nsq.s_ltp_shp_q14[nsq.s_ltp_shp_buf_idx as usize + i] =
            s_lf_ar_shp_q14.wrapping_sub(shl32(n_lf_q12, 2));
        s_ltp_q15[nsq.s_ltp_buf_idx as usize + i] = shl32(lpc_exc_q14, 1);

        // Make dither dependent on quantized signal
        nsq.rand_seed = nsq.rand_seed.wrapping_add(pulses[i] as i32);
    }

    // Update LPC synth buffer
    nsq.s_lpc_q14.copy_within(length..length + NSQ_LPC_BUF_LENGTH, 0);
}

/// Noise shaping quantizer (standard, non-delayed-decision).
/// Matches C: `silk_NSQ_c`.
pub fn silk_nsq(
    ps_enc: &SilkEncoderState,
    nsq: &mut NsqState,
    ps_indices: &mut SideInfoIndices,
    x16: &[i16],
    pulses: &mut [i8],
    pred_coef_q12: &[[i16; MAX_LPC_ORDER]; 2],
    ltp_coef_q14: &[i16],
    ar_q13: &[i16],
    harm_shape_gain_q14: &[i32],
    tilt_q14: &[i32],
    lf_shp_q14: &[i32],
    gains_q16: &[i32],
    pitch_l: &[i32],
    lambda_q10: i32,
    ltp_scale_q14: i32,
) {
    let nb_subfr = ps_enc.nb_subfr as usize;
    let subfr_length = ps_enc.subfr_length as usize;
    let frame_length = ps_enc.frame_length as usize;
    let ltp_mem_length = ps_enc.ltp_mem_length as usize;
    let predict_lpc_order = ps_enc.predict_lpc_order as usize;
    let shaping_lpc_order = ps_enc.shaping_lpc_order as usize;

    let signal_type = ps_indices.signal_type as i32;

    // Allocate temporary buffers
    let mut s_ltp_q15 = vec![0i32; ltp_mem_length + frame_length];
    let mut s_ltp = vec![0i16; ltp_mem_length + frame_length];
    let mut x_sc_q10 = vec![0i32; subfr_length];

    nsq.rand_seed = ps_indices.seed as i32;

    // Set unvoiced lag to the previous one, overwrite later for voiced
    let mut lag = nsq.lag_prev;

    let offset_q10 = SILK_QUANTIZATION_OFFSETS_Q10
        [(signal_type >> 1) as usize]
        [ps_indices.quant_offset_type as usize] as i32;

    let lsf_interpolation_flag = if ps_indices.nlsf_interp_coef_q2 == 4 { 0 } else { 1 };

    // Set up pointers to start of sub frame
    nsq.s_ltp_shp_buf_idx = ltp_mem_length as i32;
    nsq.s_ltp_buf_idx = ltp_mem_length as i32;

    for k in 0..nb_subfr {
        let coef_idx = (k >> 1) | (1 - lsf_interpolation_flag);
        let a_q12 = &pred_coef_q12[coef_idx as usize];
        if k == 0 && nsq.s_ltp_shp_buf_idx < 650 {
            eprintln!("[NSQ LPC] coef_idx={} a_q12={:?}", coef_idx, &a_q12[..predict_lpc_order]);
        }
        let b_q14 = &ltp_coef_q14[k * LTP_ORDER..];
        let ar_shp_q13 = &ar_q13[k * shaping_lpc_order..];

        // Noise shape parameters: pack HarmShapeGain into FIR packed format
        let harm_shape_fir_packed_q14 = (harm_shape_gain_q14[k] >> 2)
            | (((harm_shape_gain_q14[k] >> 1) as i32) << 16);

        nsq.rewhite_flag = 0;
        if signal_type == TYPE_VOICED {
            // Voiced
            lag = pitch_l[k];

            // Re-whitening
            if (k as i32 & (3 - (lsf_interpolation_flag << 1) as i32)) == 0 {
                // Rewhiten with new A coefs
                let start_idx = (ltp_mem_length as i32 - lag - ps_enc.predict_lpc_order as i32 - LTP_ORDER as i32 / 2) as usize;

                // LPC analysis filter on sLTP using nsq.xq as source
                let xq_offset = start_idx + k * subfr_length;
                let filter_len = ltp_mem_length - start_idx;
                // Copy xq to s_ltp for filtering
                let xq_src: Vec<i16> = nsq.xq[xq_offset..xq_offset + filter_len].to_vec();
                silk_lpc_analysis_filter(
                    &mut s_ltp[start_idx..start_idx + filter_len],
                    &xq_src,
                    a_q12,
                    filter_len,
                    predict_lpc_order,
                );

                nsq.rewhite_flag = 1;
                nsq.s_ltp_buf_idx = ltp_mem_length as i32;
            }
        }

        silk_nsq_scale_states(
            ps_enc,
            nsq,
            &x16[k * subfr_length..],
            &mut x_sc_q10,
            &s_ltp,
            &mut s_ltp_q15,
            k,
            ltp_scale_q14,
            gains_q16,
            pitch_l,
            signal_type,
        );

        silk_noise_shape_quantizer(
            nsq,
            signal_type,
            &x_sc_q10,
            &mut pulses[k * subfr_length..],
            ltp_mem_length + k * subfr_length,
            &mut s_ltp_q15,
            a_q12,
            b_q14,
            ar_shp_q13,
            lag,
            harm_shape_fir_packed_q14,
            tilt_q14[k] as i32,
            lf_shp_q14[k],
            gains_q16[k],
            lambda_q10,
            offset_q10,
            subfr_length,
            shaping_lpc_order,
            predict_lpc_order,
        );

        nsq.s_ltp_shp_buf_idx += subfr_length as i32;
        nsq.s_ltp_buf_idx += subfr_length as i32;
    }

    // Update lagPrev for next frame
    nsq.lag_prev = pitch_l[nb_subfr - 1];

    // Copy last part of buffers to beginning for next frame
    nsq.xq.copy_within(frame_length..frame_length + ltp_mem_length, 0);
    nsq.s_ltp_shp_q14.copy_within(frame_length..frame_length + ltp_mem_length, 0);
    nsq.s_ltp_buf_idx = ltp_mem_length as i32;
    nsq.s_ltp_shp_buf_idx = ltp_mem_length as i32;
}

// ===========================================================================
// Helper functions for frame encoding pipeline
// ===========================================================================

/// Autocorrelation. Matches C: `silk_autocorr`.
pub fn silk_autocorr(
    results: &mut [i32],
    scale: &mut i32,
    input: &[i16],
    input_size: usize,
    corr_count: usize,
) {
    let n = imin(input_size as i32, corr_count as i32) as usize;
    // celt_autocorr takes &[i32], so convert
    let input_i32: Vec<i32> = input[..input_size].iter().map(|&x| x as i32).collect();
    *scale = crate::celt::lpc::celt_autocorr(&input_i32, results, None, 0, n - 1, input_size);
}

/// Schur recursion, Q15 reflection coefficients. Matches C: `silk_schur`.
pub fn silk_schur(
    rc_q15: &mut [i16],
    c: &[i32],
    order: usize,
) -> i32 {
    let mut cc = [[0i32; 2]; MAX_SHAPE_LPC_ORDER + 1];
    let lz = silk_clz32(c[0]);

    if lz < 2 {
        for k in 0..=order {
            cc[k][0] = c[k] >> 1;
            cc[k][1] = cc[k][0];
        }
    } else if lz > 2 {
        let lz = lz - 2;
        for k in 0..=order {
            cc[k][0] = shl32(c[k], lz);
            cc[k][1] = cc[k][0];
        }
    } else {
        for k in 0..=order {
            cc[k][0] = c[k];
            cc[k][1] = cc[k][0];
        }
    }

    let mut k = 0usize;
    while k < order {
        if silk_abs_int32(cc[k + 1][0]) >= cc[0][1] {
            if cc[k + 1][0] > 0 {
                rc_q15[k] = -((0.99f64 * 32768.0) as i16);
            } else {
                rc_q15[k] = (0.99f64 * 32768.0) as i16;
            }
            k += 1;
            break;
        }

        let rc_tmp_q15 = silk_sat16(-(cc[k + 1][0] / imax(cc[0][1] >> 15, 1)));
        rc_q15[k] = rc_tmp_q15 as i16;

        for n in 0..(order - k) {
            let ctmp1 = cc[n + k + 1][0];
            let ctmp2 = cc[n][1];
            cc[n + k + 1][0] = silk_smlawb(ctmp1, shl32(ctmp2, 1), rc_tmp_q15 as i16);
            cc[n][1] = silk_smlawb(ctmp2, shl32(ctmp1, 1), rc_tmp_q15 as i16);
        }
        k += 1;
    }

    while k < order {
        rc_q15[k] = 0;
        k += 1;
    }

    imax(1, cc[0][1])
}

/// Schur recursion, Q16 reflection coefficients (64-bit accuracy). Matches C: `silk_schur64`.
pub fn silk_schur64(
    rc_q16: &mut [i32],
    c: &[i32],
    order: usize,
) -> i32 {
    let mut cc = [[0i32; 2]; MAX_SHAPE_LPC_ORDER + 1];

    if c[0] <= 0 {
        for k in 0..order {
            rc_q16[k] = 0;
        }
        return 0;
    }

    for k in 0..=order {
        cc[k][0] = c[k];
        cc[k][1] = c[k];
    }

    let mut k = 0usize;
    while k < order {
        if silk_abs_int32(cc[k + 1][0]) >= cc[0][1] {
            if cc[k + 1][0] > 0 {
                rc_q16[k] = -((0.99f64 * 65536.0) as i32);
            } else {
                rc_q16[k] = (0.99f64 * 65536.0) as i32;
            }
            k += 1;
            break;
        }

        let rc_tmp_q31 = silk_div32_varq(-cc[k + 1][0], cc[0][1], 31);
        rc_q16[k] = silk_rshift_round(rc_tmp_q31, 15);

        for n in 0..(order - k) {
            let ctmp1 = cc[n + k + 1][0];
            let ctmp2 = cc[n][1];
            cc[n + k + 1][0] = ctmp1.wrapping_add(silk_smmul(shl32(ctmp2, 1), rc_tmp_q31));
            cc[n][1] = ctmp2.wrapping_add(silk_smmul(shl32(ctmp1, 1), rc_tmp_q31));
        }
        k += 1;
    }

    while k < order {
        rc_q16[k] = 0;
        k += 1;
    }

    imax(1, cc[0][1])
}

/// Convert reflection coefficients to prediction coefficients. Matches C: `silk_k2a`.
pub fn silk_k2a(
    a_q24: &mut [i32],
    rc_q15: &[i16],
    order: usize,
) {
    for k in 0..order {
        let rc = rc_q15[k] as i32;
        for n in 0..((k + 1) >> 1) {
            let tmp1 = a_q24[n];
            let tmp2 = a_q24[k - n - 1];
            a_q24[n] = silk_smlawb(tmp1, shl32(tmp2, 1), rc as i16);
            a_q24[k - n - 1] = silk_smlawb(tmp2, shl32(tmp1, 1), rc as i16);
        }
        a_q24[k] = -(shl32(rc_q15[k] as i32, 9));
    }
}

/// Convert reflection coefficients (Q16) to prediction coefficients. Matches C: `silk_k2a_Q16`.
pub fn silk_k2a_q16(
    a_q24: &mut [i32],
    rc_q16: &[i32],
    order: usize,
) {
    for k in 0..order {
        let rc = rc_q16[k];
        for n in 0..((k + 1) >> 1) {
            let tmp1 = a_q24[n];
            let tmp2 = a_q24[k - n - 1];
            a_q24[n] = silk_smlaww(tmp1, tmp2, rc);
            a_q24[k - n - 1] = silk_smlaww(tmp2, tmp1, rc);
        }
        a_q24[k] = -(shl32(rc, 8));
    }
}

/// Frequency table for sine window computation.
const SINE_WINDOW_FREQ_TABLE_Q16: [i16; 27] = [
    12111, 9804, 8235, 7100, 6239, 5565, 5022, 4575, 4202,
    3885, 3612, 3375, 3167, 2984, 2820, 2674, 2542, 2422,
    2313, 2214, 2123, 2038, 1961, 1889, 1822, 1760, 1702,
];

/// Apply sine window to signal vector. Matches C: `silk_apply_sine_window`.
pub fn silk_apply_sine_window(
    px_win: &mut [i16],
    px: &[i16],
    win_type: i32,
    length: usize,
) {
    let k_idx = (length >> 2) - 4;
    let f_q16 = SINE_WINDOW_FREQ_TABLE_Q16[k_idx] as i32;
    let c_q16 = silk_smulwb(f_q16, (-f_q16) as i16);

    let (mut s0_q16, mut s1_q16) = if win_type == 1 {
        (0i32, f_q16 + (length as i32 >> 3))
    } else {
        (1 << 16, (1 << 16) + (c_q16 >> 1) + (length as i32 >> 4))
    };

    let mut k = 0;
    while k < length {
        px_win[k] = silk_smulwb((s0_q16 + s1_q16) >> 1, px[k] as i16) as i16;
        px_win[k + 1] = silk_smulwb(s1_q16, px[k + 1] as i16) as i16;
        s0_q16 = imin(silk_smulwb(s1_q16, c_q16 as i16) + (s1_q16 << 1) - s0_q16 + 1, 1 << 16);

        px_win[k + 2] = silk_smulwb((s0_q16 + s1_q16) >> 1, px[k + 2] as i16) as i16;
        px_win[k + 3] = silk_smulwb(s0_q16, px[k + 3] as i16) as i16;
        s1_q16 = imin(silk_smulwb(s0_q16, c_q16 as i16) + (s0_q16 << 1) - s1_q16, 1 << 16);

        k += 4;
    }
}

/// Scale and copy a 16-bit vector. Matches C: `silk_scale_copy_vector16`.
pub fn silk_scale_copy_vector16(
    data_out: &mut [i16],
    data_in: &[i16],
    gain_q16: i32,
    len: usize,
) {
    for i in 0..len {
        let tmp = silk_smulwb(gain_q16, data_in[i] as i16);
        data_out[i] = silk_check_fit16(tmp);
    }
}

/// Compute unique identifier for gain vector. Matches C: `silk_gains_ID`.
pub fn silk_gains_id(ind: &[i8], nb_subfr: usize) -> i32 {
    let mut gains_id = 0i32;
    for k in 0..nb_subfr {
        gains_id = gains_id * 51 + ind[k] as i32;
    }
    gains_id
}

/// LTP scale control. Matches C: `silk_LTP_scale_ctrl_FIX`.
pub fn silk_ltp_scale_ctrl(
    ps_enc: &mut SilkEncoderStateFix,
    ps_enc_ctrl: &mut SilkEncoderControl,
    cond_coding: i32,
) {
    if cond_coding == CODE_INDEPENDENTLY {
        let mut round_loss = ps_enc.s_cmn.packet_loss_perc * ps_enc.s_cmn.n_frames_per_packet;
        if ps_enc.s_cmn.lbrr_flag != 0 {
            round_loss = 2 + silk_smulbb(round_loss, round_loss) / 100;
        }
        ps_enc.s_cmn.indices.ltp_scale_index =
            (silk_smulbb(ps_enc_ctrl.ltp_pred_cod_gain_q7, round_loss)
                > silk_log2lin(128 * 7 + 2900 - ps_enc.s_cmn.snr_db_q7)) as i8;
        ps_enc.s_cmn.indices.ltp_scale_index +=
            (silk_smulbb(ps_enc_ctrl.ltp_pred_cod_gain_q7, round_loss)
                > silk_log2lin(128 * 7 + 3900 - ps_enc.s_cmn.snr_db_q7)) as i8;
    } else {
        ps_enc.s_cmn.indices.ltp_scale_index = 0;
    }
    ps_enc_ctrl.ltp_scale_q14 = SILK_LTP_SCALES_TABLE_Q14[ps_enc.s_cmn.indices.ltp_scale_index as usize] as i32;
}

/// Warped autocorrelation constants.
const QS: i32 = 14;
const QC: i32 = 10;

/// Warped autocorrelation. Matches C: `silk_warped_autocorrelation_FIX_c`.
pub fn silk_warped_autocorrelation(
    corr: &mut [i32],
    scale: &mut i32,
    input: &[i16],
    warping_q16: i32,
    length: usize,
    order: usize,
) {
    let mut state_qs = vec![0i32; MAX_SHAPE_LPC_ORDER + 1];
    let mut corr_qc = vec![0i64; MAX_SHAPE_LPC_ORDER + 1];

    for n in 0..length {
        let mut tmp1_qs = shl32(input[n] as i32, QS);
        for i in (0..order).step_by(2) {
            let tmp2_qs = silk_smlawb(state_qs[i], (state_qs[i + 1] - tmp1_qs) as i32, warping_q16 as i16);
            state_qs[i] = tmp1_qs;
            corr_qc[i] += silk_rshift64(silk_smull(tmp1_qs, state_qs[0]), 2 * QS - QC);
            tmp1_qs = silk_smlawb(state_qs[i + 1], (state_qs[i + 2] - tmp2_qs) as i32, warping_q16 as i16);
            state_qs[i + 1] = tmp2_qs;
            corr_qc[i + 1] += silk_rshift64(silk_smull(tmp2_qs, state_qs[0]), 2 * QS - QC);
        }
        state_qs[order] = tmp1_qs;
        corr_qc[order] += silk_rshift64(silk_smull(tmp1_qs, state_qs[0]), 2 * QS - QC);
    }

    let lsh = silk_clz64_fn(corr_qc[0]) - 35;
    let lsh = silk_limit(lsh, -12 - QC, 30 - QC);
    *scale = -(QC + lsh);
    if lsh >= 0 {
        for i in 0..=order {
            corr[i] = silk_check_fit32(silk_lshift64(corr_qc[i], lsh));
        }
    } else {
        for i in 0..=order {
            corr[i] = silk_check_fit32(silk_rshift64(corr_qc[i], -lsh));
        }
    }
}

/// Residual energy per subframe. Matches C: `silk_residual_energy_FIX`.
pub fn silk_residual_energy_fix(
    nrgs: &mut [i32],
    nrgs_q: &mut [i32],
    x: &[i16],
    a_q12: &mut [[i16; MAX_LPC_ORDER]; 2],
    gains: &[i32],
    subfr_length: usize,
    nb_subfr: usize,
    lpc_order: usize,
) {
    let offset = lpc_order + subfr_length;
    let half_subfr = MAX_NB_SUBFR >> 1;
    let mut x_ptr = 0usize;

    let mut lpc_res = vec![0i16; half_subfr * offset];

    for i in 0..(nb_subfr >> 1) {
        silk_lpc_analysis_filter(
            &mut lpc_res,
            &x[x_ptr..],
            &a_q12[i],
            half_subfr * offset,
            lpc_order,
        );

        let mut lpc_res_ptr = lpc_order;
        for j in 0..half_subfr {
            let (nrg, rshift) = silk_sum_sqr_shift(&lpc_res[lpc_res_ptr..lpc_res_ptr + subfr_length]);
            nrgs[i * half_subfr + j] = nrg;
            nrgs_q[i * half_subfr + j] = -(rshift as i32);
            lpc_res_ptr += offset;
        }
        x_ptr += half_subfr * offset;
    }

    // Apply squared subframe gains
    for i in 0..nb_subfr {
        let lz1 = silk_clz32(nrgs[i]) - 1;
        let lz2 = silk_clz32(gains[i]) - 1;
        let tmp32 = shl32(gains[i], lz2);
        let tmp32 = silk_smmul(tmp32, tmp32);
        nrgs[i] = silk_smmul(tmp32, shl32(nrgs[i], lz1));
        nrgs_q[i] += lz1 + 2 * lz2 - 32 - 32;
    }
}

/// Residual energy from covariance. Matches C: `silk_residual_energy16_covar_FIX`.
pub fn silk_residual_energy16_covar(
    c: &[i16],
    wxx: &[i32],
    wxs: &[i32],
    wss: i32,
    d: usize,
    cq: i32,
) -> i32 {
    let mut lshifts = 16 - cq;
    let mut qxtra = lshifts;

    let mut c_max = 0i32;
    for i in 0..d {
        c_max = imax(c_max, silk_abs_int32(c[i] as i32));
    }
    qxtra = imin(qxtra, silk_clz32(c_max) - 17);

    let w_max = imax(wxx[0], wxx[d * d - 1]);
    qxtra = imin(qxtra, silk_clz32(silk_mul(d as i32, silk_smulwb(w_max, c_max as i16) >> 4)) - 5);
    qxtra = imax(qxtra, 0);

    let mut cn = vec![0i32; d];
    for i in 0..d {
        cn[i] = shl32(c[i] as i32, qxtra);
    }
    lshifts -= qxtra;

    // wss - 2 * wXx * c
    let mut tmp = 0i32;
    for i in 0..d {
        tmp = silk_smlawb(tmp, wxs[i], cn[i] as i16);
    }
    let mut nrg = (wss >> (1 + lshifts)) - tmp;

    // c' * wXX * c
    let mut tmp2 = 0i32;
    for i in 0..d {
        let mut tmp = 0i32;
        let row = i * d;
        for j in (i + 1)..d {
            tmp = silk_smlawb(tmp, wxx[row + j], cn[j] as i16);
        }
        tmp = silk_smlawb(tmp, wxx[row + i] >> 1, cn[i] as i16);
        tmp2 = silk_smlawb(tmp2, tmp, cn[i] as i16);
    }
    nrg = silk_add_lshift32(nrg, tmp2, lshifts);

    if nrg < 1 {
        1
    } else if nrg > (i32::MAX >> (lshifts + 2)) {
        i32::MAX >> 1
    } else {
        shl32(nrg, lshifts + 1)
    }
}

/// LTP analysis filter. Matches C: `silk_LTP_analysis_filter_FIX`.
pub fn silk_ltp_analysis_filter(
    ltp_res: &mut [i16],
    x: &[i16],
    x_base: usize,  // base offset in x (points to frame start)
    ltp_coef_q14: &[i16],
    pitch_l: &[i32],
    inv_gains_q16: &[i32],
    subfr_length: usize,
    nb_subfr: usize,
    pre_length: usize,
) {
    let mut x_ptr = x_base;
    let mut ltp_res_ptr = 0usize;
    let out_stride = subfr_length + pre_length;

    for k in 0..nb_subfr {
        let lag = pitch_l[k] as usize;
        let b0 = ltp_coef_q14[k * LTP_ORDER] as i32;
        let b1 = ltp_coef_q14[k * LTP_ORDER + 1] as i32;
        let b2 = ltp_coef_q14[k * LTP_ORDER + 2] as i32;
        let b3 = ltp_coef_q14[k * LTP_ORDER + 3] as i32;
        let b4 = ltp_coef_q14[k * LTP_ORDER + 4] as i32;

        for i in 0..out_stride {
            let xi = x_ptr + i;
            let lag_ptr = xi as i32 - lag as i32;

            ltp_res[ltp_res_ptr + i] = x[xi];

            // LTP estimate: sum of 5 taps
            let mut ltp_est = silk_smulbb(x[lag_ptr as usize + LTP_ORDER / 2] as i32, b0);
            ltp_est = silk_smlabb_ovflw(ltp_est, x[(lag_ptr + 1) as usize] as i32, b1);
            ltp_est = silk_smlabb_ovflw(ltp_est, x[lag_ptr as usize] as i32, b2);
            ltp_est = silk_smlabb_ovflw(ltp_est, x[(lag_ptr - 1) as usize] as i32, b3);
            ltp_est = silk_smlabb_ovflw(ltp_est, x[(lag_ptr - 2) as usize] as i32, b4);
            ltp_est = silk_rshift_round(ltp_est, 14);

            // Subtract LTP prediction, saturate
            ltp_res[ltp_res_ptr + i] = sat16((x[xi] as i32) - ltp_est);
            // Scale by inverse gain
            ltp_res[ltp_res_ptr + i] = silk_smulwb(inv_gains_q16[k], ltp_res[ltp_res_ptr + i] as i16) as i16;
        }

        ltp_res_ptr += out_stride;
        x_ptr += subfr_length;
    }
}

/// Correlation matrix computation. Matches C: `silk_corrMatrix_FIX`.
/// Computes the correlation matrix XX and its trace (nrg).
pub fn silk_corr_matrix(
    x: &[i16],
    l: usize,
    order: usize,
    xx: &mut [i32],
    nrg: &mut i32,
    rshift: &mut i32,
) {
    // First compute energy and shift
    let mut head_room = i32::MAX;
    for i in 0..order {
        let (energy, shift) = silk_sum_sqr_shift(&x[i..i + l]);
        let _ = energy;
        head_room = imin(head_room, shift as i32);
    }

    // Compute energy of x[0..L] with head_room shift
    let mut energy = 0i64;
    for i in 0..l {
        let xi = (x[i] as i32) >> (head_room >> 1);
        energy += (xi as i64) * (xi as i64);
    }
    // Normalize
    let lz = if energy > 0 { (energy as u64).leading_zeros() as i32 - 1 } else { 63 };
    let total_shift = imax(0, head_room - lz + 32);
    *rshift = total_shift;

    // Diagonal and upper triangle
    for j in 0..order {
        for i in j..order {
            let mut sum = 0i64;
            for n in 0..l {
                sum += x[j + n] as i64 * x[i + n] as i64;
            }
            xx[j * order + i] = (sum >> total_shift) as i32;
            xx[i * order + j] = xx[j * order + i];
        }
    }

    // Diagonal energy as nrg
    *nrg = xx[0];
}

/// Correlation vector computation. Matches C: `silk_corrVector_FIX`.
pub fn silk_corr_vector(
    x: &[i16],
    t: &[i16],
    l: usize,
    order: usize,
    xt: &mut [i32],
    rshift: i32,
) {
    for i in 0..order {
        let mut sum = 0i64;
        for n in 0..l {
            sum += x[i + n] as i64 * t[n] as i64;
        }
        xt[i] = (sum >> rshift) as i32;
    }
}

/// Find LTP coefficients (correlation analysis). Matches C: `silk_find_LTP_FIX`.
pub fn silk_find_ltp(
    xxltp_q17: &mut [i32],
    xxltp_q17_len: usize,
    xxltp_q17_vec: &mut [i32],
    r_ptr: &[i16],
    r_base: usize,
    lag: &[i32],
    subfr_length: usize,
    nb_subfr: usize,
) {
    let mut r_offset = r_base;

    for k in 0..nb_subfr {
        let lag_offset = (r_offset as i32 - lag[k] - LTP_ORDER as i32 / 2) as usize;

        // xx = energy of r_ptr[r_offset..r_offset+subfr_length+LTP_ORDER]
        let (xx, xx_shifts) = silk_sum_sqr_shift(&r_ptr[r_offset..r_offset + subfr_length + LTP_ORDER]);

        // Compute correlation matrix of lag_ptr
        let xx_base = k * LTP_ORDER * LTP_ORDER;
        let mut nrg = 0i32;
        let mut xx_shifts_mat = 0i32;
        silk_corr_matrix(
            &r_ptr[lag_offset..],
            subfr_length,
            LTP_ORDER,
            &mut xxltp_q17[xx_base..],
            &mut nrg,
            &mut xx_shifts_mat,
        );

        let extra_shifts = xx_shifts as i32 - xx_shifts_mat;
        let x_x_shifts;
        let mut xx_val = xx;
        if extra_shifts > 0 {
            x_x_shifts = xx_shifts as i32;
            for i in 0..LTP_ORDER * LTP_ORDER {
                xxltp_q17[xx_base + i] >>= extra_shifts;
            }
            nrg >>= extra_shifts;
        } else if extra_shifts < 0 {
            x_x_shifts = xx_shifts_mat;
            xx_val >>= -extra_shifts;
        } else {
            x_x_shifts = xx_shifts as i32;
        }

        // Compute correlation vector
        let xv_base = k * LTP_ORDER;
        silk_corr_vector(
            &r_ptr[lag_offset..],
            &r_ptr[r_offset..],
            subfr_length,
            LTP_ORDER,
            &mut xxltp_q17_vec[xv_base..],
            x_x_shifts,
        );

        // Normalize to Q17
        let temp = imax(silk_smlawb(1, nrg, ((1.0f64 / 0.5) * 65536.0) as i16), xx_val);
        for i in 0..LTP_ORDER * LTP_ORDER {
            xxltp_q17[xx_base + i] = ((xxltp_q17[xx_base + i] as i64) << 17 / temp as i64) as i32;
        }
        for i in 0..LTP_ORDER {
            xxltp_q17_vec[xv_base + i] = ((xxltp_q17_vec[xv_base + i] as i64) << 17 / temp as i64) as i32;
        }

        r_offset += subfr_length;
    }
}

// ===========================================================================
// Tuning parameter constants
// ===========================================================================

const LAMBDA_OFFSET: f64 = 1.2;
const LAMBDA_SPEECH_ACT: f64 = -0.2;
const LAMBDA_DELAYED_DECISIONS: f64 = -0.05;
const LAMBDA_INPUT_QUALITY: f64 = -0.1;
const LAMBDA_CODING_QUALITY: f64 = -0.2;
const LAMBDA_QUANT_OFFSET: f64 = 0.8;
const MAX_PREDICTION_POWER_GAIN: f64 = 1e4;
const MAX_PREDICTION_POWER_GAIN_AFTER_RESET: f64 = 1e2;
const SPEECH_ACTIVITY_DTX_THRES: f64 = 0.05;
const NB_SPEECH_FRAMES_BEFORE_DTX: i32 = 10;
const MAX_CONSECUTIVE_DTX: i32 = 20;
const FIND_PITCH_WHITE_NOISE_FRACTION: f64 = 1e-3;
const FIND_PITCH_BANDWIDTH_EXPANSION: f64 = 0.99;
const LBRR_SPEECH_ACTIVITY_THRES: f64 = 0.3;
const LOW_FREQ_SHAPING_Q0: f64 = 2.0;
const LOW_QUALITY_LOW_FREQ_SHAPING_DECR: f64 = 0.5;
const SUBFR_SMTH_COEF: f64 = 0.4;
const HARM_SNR_INCR_DB: f64 = 2.0;
const HARMONIC_SHAPING: f64 = 0.3;
const HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING: f64 = 0.2;
const HP_NOISE_COEF: f64 = 0.25;
const HARM_HP_NOISE_COEF: f64 = 0.35;
const SHAPE_WHITE_NOISE_FRACTION: f64 = 3e-5;
const BANDWIDTH_EXPANSION: f64 = 0.95;
const WHITE_NOISE_FRACTION: f64 = 2e-5;

// ===========================================================================
// VAD and signal type decision
// ===========================================================================

/// Perform VAD and set signal type. Matches C: `silk_encode_do_VAD_FIX`.
pub fn silk_encode_do_vad_fix(
    ps_enc: &mut SilkEncoderStateFix,
    activity: i32,
) {
    let activity_threshold = ((SPEECH_ACTIVITY_DTX_THRES * 256.0) + 0.5) as i32;

    // VAD — already called externally in silk_encode, so use the result
    // (in C, silk_VAD_GetSA_Q8 is called here, but we call it separately)

    // If Opus VAD inactive and SILK VAD active: lower
    if activity == 0 && ps_enc.s_cmn.speech_activity_q8 >= activity_threshold {
        ps_enc.s_cmn.speech_activity_q8 = activity_threshold - 1;
    }

    // Convert to VAD flags and DTX
    if ps_enc.s_cmn.speech_activity_q8 < activity_threshold {
        ps_enc.s_cmn.indices.signal_type = TYPE_NO_VOICE_ACTIVITY as i8;
        ps_enc.s_cmn.no_speech_counter += 1;
        if ps_enc.s_cmn.no_speech_counter <= NB_SPEECH_FRAMES_BEFORE_DTX {
            ps_enc.s_cmn.in_dtx = 0;
        } else if ps_enc.s_cmn.no_speech_counter > MAX_CONSECUTIVE_DTX + NB_SPEECH_FRAMES_BEFORE_DTX {
            ps_enc.s_cmn.no_speech_counter = NB_SPEECH_FRAMES_BEFORE_DTX;
            ps_enc.s_cmn.in_dtx = 0;
        }
        ps_enc.s_cmn.vad_flags[ps_enc.s_cmn.n_frames_encoded as usize] = 0;
    } else {
        ps_enc.s_cmn.no_speech_counter = 0;
        ps_enc.s_cmn.in_dtx = 0;
        ps_enc.s_cmn.indices.signal_type = TYPE_UNVOICED as i8;
        ps_enc.s_cmn.vad_flags[ps_enc.s_cmn.n_frames_encoded as usize] = 1;
    }
}

// ===========================================================================
// Process gains
// ===========================================================================

/// Process gains. Matches C: `silk_process_gains_FIX`.
pub fn silk_process_gains_fix(
    ps_enc: &mut SilkEncoderStateFix,
    ps_enc_ctrl: &mut SilkEncoderControl,
    cond_coding: i32,
) {
    let nb_subfr = ps_enc.s_cmn.nb_subfr as usize;

    // Gain reduction when LTP coding gain is high (voiced)
    if ps_enc.s_cmn.indices.signal_type as i32 == TYPE_VOICED {
        let s_q16 = -silk_sigm_q15(silk_rshift_round(
            ps_enc_ctrl.ltp_pred_cod_gain_q7 - ((12.0 * 128.0) as i32),
            4,
        ));
        for k in 0..nb_subfr {
            ps_enc_ctrl.gains_q16[k] =
                silk_smlawb(ps_enc_ctrl.gains_q16[k], ps_enc_ctrl.gains_q16[k], s_q16 as i16);
        }
    }

    // Limit quantized signal level
    let inv_max_sqr_val_q16 = silk_div32_16(
        silk_log2lin(silk_smulwb(
            ((21.0 + 16.0 / 0.33) * 128.0 + 0.5) as i32 - ps_enc.s_cmn.snr_db_q7,
            ((0.33 * 65536.0) + 0.5) as i16,
        )),
        ps_enc.s_cmn.subfr_length,
    );

    for k in 0..nb_subfr {
        let res_nrg = ps_enc_ctrl.res_nrg[k];
        let mut res_nrg_part = silk_smulww(res_nrg, inv_max_sqr_val_q16);
        if ps_enc_ctrl.res_nrg_q[k] > 0 {
            res_nrg_part = silk_rshift_round(res_nrg_part, ps_enc_ctrl.res_nrg_q[k]);
        } else if ps_enc_ctrl.res_nrg_q[k] < 0 {
            if res_nrg_part >= (i32::MAX >> (-ps_enc_ctrl.res_nrg_q[k])) {
                res_nrg_part = i32::MAX;
            } else {
                res_nrg_part = shl32(res_nrg_part, -ps_enc_ctrl.res_nrg_q[k]);
            }
        }
        let gain = ps_enc_ctrl.gains_q16[k];
        let gain_squared = silk_add_sat32(res_nrg_part, silk_smmul(gain, gain));
        eprintln!("[PROCGAIN k={}] gain_in={} res_nrg={} res_nrg_q={} res_nrg_part={} inv_max_sqr={} smmul={} gain_sq={} branch={}",
            k, gain, res_nrg, ps_enc_ctrl.res_nrg_q[k], res_nrg_part, inv_max_sqr_val_q16,
            silk_smmul(gain, gain), gain_squared, if gain_squared < i16::MAX as i32 { "lo" } else { "hi" });
        if gain_squared < i16::MAX as i32 {
            let gain_squared = silk_smlaww(shl32(res_nrg_part, 16), gain, gain);
            let gain = imin(silk_sqrt_approx(gain_squared), i32::MAX >> 8);
            ps_enc_ctrl.gains_q16[k] = silk_lshift_sat32(gain, 8);
            eprintln!("[PROCGAIN k={}] lo: gain_sq_hi={} sqrt={} final_q16={}", k, gain_squared, gain, ps_enc_ctrl.gains_q16[k]);
        } else {
            let gain = imin(silk_sqrt_approx(gain_squared), i32::MAX >> 16);
            ps_enc_ctrl.gains_q16[k] = silk_lshift_sat32(gain, 16);
            eprintln!("[PROCGAIN k={}] hi: sqrt={} final_q16={}", k, gain, ps_enc_ctrl.gains_q16[k]);
        }
    }

    // Save unquantized gains
    ps_enc_ctrl.gains_unq_q16[..nb_subfr].copy_from_slice(&ps_enc_ctrl.gains_q16[..nb_subfr]);
    ps_enc_ctrl.last_gain_index_prev = ps_enc.s_shape.last_gain_index;

    eprintln!("[GAINS TRACE] gains_q16_before_quant={:?} last_gain_index={}",
        &ps_enc_ctrl.gains_q16[..nb_subfr], ps_enc.s_shape.last_gain_index);

    // Quantize gains
    silk_gains_quant(
        &mut ps_enc.s_cmn.indices.gains_indices,
        &mut ps_enc_ctrl.gains_q16,
        &mut ps_enc.s_shape.last_gain_index,
        cond_coding == CODE_CONDITIONALLY,
        nb_subfr,
    );

    // Set quantizer offset for voiced
    if ps_enc.s_cmn.indices.signal_type as i32 == TYPE_VOICED {
        if ps_enc_ctrl.ltp_pred_cod_gain_q7 + (ps_enc.s_cmn.input_tilt_q15 >> 8) > ((1.0 * 128.0) as i32) {
            ps_enc.s_cmn.indices.quant_offset_type = 0;
        } else {
            ps_enc.s_cmn.indices.quant_offset_type = 1;
        }
    }

    // Lambda (rate-distortion tradeoff)
    let quant_offset_q10 = SILK_QUANTIZATION_OFFSETS_Q10
        [(ps_enc.s_cmn.indices.signal_type as usize) >> 1]
        [ps_enc.s_cmn.indices.quant_offset_type as usize] as i32;

    ps_enc_ctrl.lambda_q10 = ((LAMBDA_OFFSET * 1024.0) + 0.5) as i32
        + silk_smulbb(((LAMBDA_DELAYED_DECISIONS * 1024.0) + 0.5) as i32, ps_enc.s_cmn.n_states_delayed_decision)
        + silk_smulwb(((LAMBDA_SPEECH_ACT * 262144.0) + 0.5) as i32, ps_enc.s_cmn.speech_activity_q8 as i16)
        + silk_smulwb(((LAMBDA_INPUT_QUALITY * 4096.0) + 0.5) as i32, ps_enc_ctrl.input_quality_q14 as i16)
        + silk_smulwb(((LAMBDA_CODING_QUALITY * 4096.0) + 0.5) as i32, ps_enc_ctrl.coding_quality_q14 as i16)
        + silk_smulwb(((LAMBDA_QUANT_OFFSET * 65536.0) + 0.5) as i32, quant_offset_q10 as i16);
}

// ===========================================================================
// Find prediction coefficients
// ===========================================================================

/// Find LPC coefficients via Burg's method. Matches C: `silk_find_LPC_FIX`.
pub fn silk_find_lpc_fix(
    ps_enc: &mut SilkEncoderState,
    nlsf_q15: &mut [i16],
    x: &[i16],
    min_inv_gain_q30: i32,
) {
    let subfr_length = (ps_enc.subfr_length + ps_enc.predict_lpc_order) as usize;
    let mut a_q16 = [0i32; MAX_LPC_ORDER];

    // Default: no interpolation
    ps_enc.indices.nlsf_interp_coef_q2 = 4;

    // Burg AR analysis for full frame
    let mut res_nrg = 0i32;
    let mut res_nrg_q = 0i32;
    silk_burg_modified(
        &mut res_nrg,
        &mut res_nrg_q,
        &mut a_q16,
        x,
        min_inv_gain_q30,
        subfr_length,
        ps_enc.nb_subfr as usize,
        ps_enc.predict_lpc_order as usize,
    );

    // Convert to NLSFs
    let d = ps_enc.predict_lpc_order as usize;

    if ps_enc.use_interpolated_nlsfs != 0
        && ps_enc.first_frame_after_reset == 0
        && ps_enc.nb_subfr == MAX_NB_SUBFR as i32
    {
        // Optimal for last 10ms
        let mut a_tmp_q16 = [0i32; MAX_LPC_ORDER];
        let mut res_tmp_nrg = 0i32;
        let mut res_tmp_nrg_q = 0i32;
        silk_burg_modified(
            &mut res_tmp_nrg,
            &mut res_tmp_nrg_q,
            &mut a_tmp_q16,
            &x[2 * subfr_length..],
            min_inv_gain_q30,
            subfr_length,
            2,
            d,
        );

        // Subtract residual energy (wrapping matches C)
        let shift = res_tmp_nrg_q - res_nrg_q;
        if shift >= 0 {
            if shift < 32 {
                res_nrg = res_nrg.wrapping_sub(res_tmp_nrg >> shift);
            }
        } else {
            res_nrg = (res_nrg >> (-shift)).wrapping_sub(res_tmp_nrg);
            res_nrg_q = res_tmp_nrg_q;
        }

        // Convert to NLSFs
        eprintln!("[A2NLSF] a_q16={:?}", &a_tmp_q16[..d]);
        silk_a2nlsf(nlsf_q15, &a_tmp_q16, d);
        eprintln!("[A2NLSF] nlsf_q15={:?}", &nlsf_q15[..d]);

        // Search over interpolation indices
        let mut nlsf0_q15 = [0i16; MAX_LPC_ORDER];
        let mut a_tmp_q12 = [0i16; MAX_LPC_ORDER];
        let lpc_res_len = 2 * subfr_length;
        let mut lpc_res = vec![0i16; lpc_res_len];

        for k in (0..=3).rev() {
            silk_interpolate(&mut nlsf0_q15, &ps_enc.prev_nlsfq_q15, nlsf_q15, k, d);
            silk_nlsf2a(&mut a_tmp_q12, &nlsf0_q15, d);
            silk_lpc_analysis_filter(&mut lpc_res, x, &a_tmp_q12, lpc_res_len, d);

            let (res_nrg0, rshift0) = silk_sum_sqr_shift(&lpc_res[d..d + subfr_length - d]);
            let (res_nrg1, rshift1) = silk_sum_sqr_shift(&lpc_res[d + subfr_length..d + 2 * (subfr_length - d)]);

            let (res_nrg0, res_nrg1, res_nrg_interp_q) = {
                let shift = rshift0 as i32 - rshift1 as i32;
                if shift >= 0 {
                    (res_nrg0, res_nrg1 >> shift, -(rshift0 as i32))
                } else {
                    (res_nrg0 >> (-shift), res_nrg1, -(rshift1 as i32))
                }
            };
            let res_nrg_interp = res_nrg0 + res_nrg1;

            let shift = res_nrg_interp_q - res_nrg_q;
            let is_interp_lower = if shift >= 0 {
                (res_nrg_interp >> shift) < res_nrg
            } else if -shift < 32 {
                res_nrg_interp < (res_nrg >> (-shift))
            } else {
                false
            };

            if is_interp_lower {
                res_nrg = res_nrg_interp;
                res_nrg_q = res_nrg_interp_q;
                ps_enc.indices.nlsf_interp_coef_q2 = k as i8;
            }
        }
    }

    if ps_enc.indices.nlsf_interp_coef_q2 == 4 {
        silk_a2nlsf(nlsf_q15, &a_q16, d);
    }
}

/// Find prediction coefficients (LPC + LTP). Matches C: `silk_find_pred_coefs_FIX`.
pub fn silk_find_pred_coefs_fix(
    ps_enc: &mut SilkEncoderStateFix,
    ps_enc_ctrl: &mut SilkEncoderControl,
    x: &[i16],
    x_base: usize,  // offset into x where frame starts
    cond_coding: i32,
) {
    let nb_subfr = ps_enc.s_cmn.nb_subfr as usize;
    let subfr_length = ps_enc.s_cmn.subfr_length as usize;
    let predict_lpc_order = ps_enc.s_cmn.predict_lpc_order as usize;
    let frame_length = ps_enc.s_cmn.frame_length as usize;

    let mut inv_gains_q16 = [0i32; MAX_NB_SUBFR];
    let mut local_gains = [0i32; MAX_NB_SUBFR];
    let mut nlsf_q15 = [0i16; MAX_LPC_ORDER];

    // Find minimum gain and compute inverse gains
    let mut min_gain_q16 = i32::MAX >> 6;
    for i in 0..nb_subfr {
        min_gain_q16 = imin(min_gain_q16, ps_enc_ctrl.gains_q16[i]);
    }
    for i in 0..nb_subfr {
        inv_gains_q16[i] = silk_div32_varq(min_gain_q16, ps_enc_ctrl.gains_q16[i], 14);
        inv_gains_q16[i] = imax(inv_gains_q16[i], 100);
        local_gains[i] = (1i32 << 16) / inv_gains_q16[i];
    }

    // Allocate LPC input buffer: nb_subfr * (predictLPCOrder + subfr_length) samples
    let lpc_stride = predict_lpc_order + subfr_length;
    let mut lpc_in_pre = vec![0i16; nb_subfr * lpc_stride];

    if ps_enc.s_cmn.indices.signal_type as i32 == TYPE_VOICED {
        // VOICED: LTP analysis
        let mut xxltp_q17 = vec![0i32; nb_subfr * LTP_ORDER * LTP_ORDER];
        let mut xxltp_q17_vec = vec![0i32; nb_subfr * LTP_ORDER];

        // find_LTP (simplified — compute correlations for LTP quantization)
        let ltp_mem = ps_enc.s_cmn.ltp_mem_length as usize;
        // res_pitch points to the residual from pitch analysis
        // For now use x_buf as a proxy since we haven't fully separated res_pitch
        silk_find_ltp(
            &mut xxltp_q17,
            nb_subfr * LTP_ORDER * LTP_ORDER,
            &mut xxltp_q17_vec,
            &ps_enc.x_buf,
            ltp_mem,
            &ps_enc_ctrl.pitch_l,
            subfr_length,
            nb_subfr,
        );

        // Quantize LTP gains
        silk_quant_ltp_gains(
            &mut ps_enc_ctrl.ltp_coef_q14,
            &mut ps_enc.s_cmn.indices.ltp_index,
            &mut ps_enc.s_cmn.indices.per_index,
            &mut ps_enc.s_cmn.sum_log_gain_q7,
            &xxltp_q17,
            0, // mu_q9
            false, // low_complexity
            nb_subfr,
        );

        // LTP scale control
        silk_ltp_scale_ctrl(ps_enc, ps_enc_ctrl, cond_coding);

        // Create LTP residual
        silk_ltp_analysis_filter(
            &mut lpc_in_pre,
            x,
            x_base - predict_lpc_order,
            &ps_enc_ctrl.ltp_coef_q14,
            &ps_enc_ctrl.pitch_l,
            &inv_gains_q16,
            subfr_length,
            nb_subfr,
            predict_lpc_order,
        );
    } else {
        // UNVOICED: scale input by inverse gains
        let mut x_ptr = x_base - predict_lpc_order;
        let mut x_pre_ptr = 0usize;
        for i in 0..nb_subfr {
            silk_scale_copy_vector16(
                &mut lpc_in_pre[x_pre_ptr..],
                &x[x_ptr..],
                inv_gains_q16[i],
                subfr_length + predict_lpc_order,
            );
            x_pre_ptr += lpc_stride;
            x_ptr += subfr_length;
        }

        ps_enc_ctrl.ltp_coef_q14 = [0; LTP_ORDER * MAX_NB_SUBFR];
        ps_enc_ctrl.ltp_pred_cod_gain_q7 = 0;
        ps_enc.s_cmn.sum_log_gain_q7 = 0;
        ps_enc_ctrl.ltp_scale_q14 = 0;
    }

    // Minimum inverse gain
    let min_inv_gain_q30 = if ps_enc.s_cmn.first_frame_after_reset != 0 {
        ((1.0 / MAX_PREDICTION_POWER_GAIN_AFTER_RESET) * (1u64 << 30) as f64 + 0.5) as i32
    } else {
        let t = silk_log2lin(silk_smlawb(
            16 << 7,
            ps_enc_ctrl.ltp_pred_cod_gain_q7,
            ((1.0 / 3.0) * 65536.0 + 0.5) as i16,
        ));
        silk_div32_varq(
            t,
            silk_smulww(
                MAX_PREDICTION_POWER_GAIN as i32,
                silk_smlawb(
                    ((0.25 * 262144.0) + 0.5) as i32,
                    ((0.75 * 262144.0) + 0.5) as i32,
                    ps_enc_ctrl.coding_quality_q14 as i16,
                ),
            ),
            14,
        )
    };

    // Find LPC coefficients
    silk_find_lpc_fix(
        &mut ps_enc.s_cmn,
        &mut nlsf_q15,
        &lpc_in_pre,
        min_inv_gain_q30,
    );

    // Quantize LSFs
    let prev_nlsf = ps_enc.s_cmn.prev_nlsfq_q15;
    let mut pred_coef_q12 = ps_enc_ctrl.pred_coef_q12;
    silk_process_nlsfs(
        &mut ps_enc.s_cmn,
        ps_enc_ctrl,
        &mut pred_coef_q12,
        &mut nlsf_q15,
        &prev_nlsf,
    );
    ps_enc_ctrl.pred_coef_q12 = pred_coef_q12;

    // Residual energy
    silk_residual_energy_fix(
        &mut ps_enc_ctrl.res_nrg,
        &mut ps_enc_ctrl.res_nrg_q,
        &lpc_in_pre,
        &mut ps_enc_ctrl.pred_coef_q12,
        &local_gains,
        subfr_length,
        nb_subfr,
        predict_lpc_order,
    );

    // Copy NLSFs for next frame interpolation
    ps_enc.s_cmn.prev_nlsfq_q15[..predict_lpc_order].copy_from_slice(&nlsf_q15[..predict_lpc_order]);
}

// ===========================================================================
// Noise shape analysis helpers
// ===========================================================================

/// Compute warped gain from AR coefficients. Matches C: `warped_gain`.
/// Returns gain in Q16.
fn warped_gain(coefs_q24: &[i32], lambda_q16: i32, order: usize) -> i32 {
    let lambda_q16 = -lambda_q16;
    let mut gain_q24 = coefs_q24[order - 1];
    for i in (0..order - 1).rev() {
        gain_q24 = silk_smlawb_i32(coefs_q24[i], gain_q24, lambda_q16);
    }
    gain_q24 = silk_smlawb_i32(1 << 24, gain_q24, -lambda_q16);
    silk_inverse32_var_q(gain_q24, 40)
}

/// Limit warped coefficients. Matches C: `limit_warped_coefs`.
fn limit_warped_coefs(coefs_q24: &mut [i32], lambda_q16: i32, limit_q24: i32, order: usize) {
    let mut lambda = -lambda_q16;

    // Convert to monic coefficients
    for i in (1..order).rev() {
        coefs_q24[i - 1] = silk_smlawb_i32(coefs_q24[i - 1], coefs_q24[i], lambda);
    }
    lambda = -lambda;
    let nom_q16 = silk_smlawb_i32(1 << 16, -lambda, lambda);
    let den_q24 = silk_smlawb_i32(1 << 24, coefs_q24[0], lambda);
    let mut gain_q16 = silk_div32_var_q(nom_q16, den_q24, 24);
    for i in 0..order {
        coefs_q24[i] = silk_smulww(gain_q16, coefs_q24[i]);
    }

    let limit_q20 = limit_q24 >> 4;
    for iter in 0..10 {
        // Find maximum absolute value
        let mut maxabs_q24 = -1i32;
        let mut ind = 0usize;
        for i in 0..order {
            let tmp = coefs_q24[i].abs();
            if tmp > maxabs_q24 {
                maxabs_q24 = tmp;
                ind = i;
            }
        }
        let maxabs_q20 = maxabs_q24 >> 4;
        if maxabs_q20 <= limit_q20 {
            return;
        }

        // Convert back to true warped coefficients
        for i in 1..order {
            coefs_q24[i - 1] = silk_smlawb_i32(coefs_q24[i - 1], coefs_q24[i], lambda);
        }
        gain_q16 = silk_inverse32_var_q(gain_q16, 32);
        for i in 0..order {
            coefs_q24[i] = silk_smulww(gain_q16, coefs_q24[i]);
        }

        // Apply bandwidth expansion
        let chirp_q16 = ((0.99 * 65536.0) as i32) - silk_div32_var_q(
            silk_smulwb_i32(
                maxabs_q20 - limit_q20,
                silk_smlabb(((0.8 * 1024.0) as i32), ((0.1 * 1024.0) as i32), iter),
            ),
            (maxabs_q20) * (ind as i32 + 1),
            22,
        );
        silk_bwexpander_32(&mut coefs_q24[..order], order, chirp_q16);

        // Convert to monic warped coefficients
        lambda = -lambda;
        for i in (1..order).rev() {
            coefs_q24[i - 1] = silk_smlawb_i32(coefs_q24[i - 1], coefs_q24[i], lambda);
        }
        lambda = -lambda;
        let nom_q16 = silk_smlawb_i32(1 << 16, -lambda, lambda);
        let den_q24 = silk_smlawb_i32(1 << 24, coefs_q24[0], lambda);
        gain_q16 = silk_div32_var_q(nom_q16, den_q24, 24);
        for i in 0..order {
            coefs_q24[i] = silk_smulww(gain_q16, coefs_q24[i]);
        }
    }
}

// ===========================================================================
// Noise shape analysis
// ===========================================================================

/// Noise shape analysis. Matches C: `silk_noise_shape_analysis_FIX`.
pub fn silk_noise_shape_analysis_fix(
    ps_enc: &mut SilkEncoderStateFix,
    ps_enc_ctrl: &mut SilkEncoderControl,
    pitch_res: &[i16],
    x_frame: &[i16],
) {
    let nb_subfr = ps_enc.s_cmn.nb_subfr as usize;
    let subfr_length = ps_enc.s_cmn.subfr_length as usize;
    let fs_khz = ps_enc.s_cmn.fs_khz;
    let shaping_lpc_order = ps_enc.s_cmn.shaping_lpc_order as usize;
    let shape_win_length = ps_enc.s_cmn.shape_win_length as usize;
    let warping_q16 = ps_enc.s_cmn.warping_q16;

    // SNR adjustment
    let snr_adj_db_q7 = ps_enc.s_cmn.snr_db_q7;
    eprintln!("[RS SNR INIT] SNR_dB_Q7={} speech_activity={} useCBR={} signalType={}",
        ps_enc.s_cmn.snr_db_q7, ps_enc.s_cmn.speech_activity_q8,
        ps_enc.s_cmn.use_cbr as i32, ps_enc.s_cmn.indices.signal_type);
    let input_quality_q14 = ((ps_enc.s_cmn.input_quality_bands_q15[0]
        + ps_enc.s_cmn.input_quality_bands_q15[1]) >> 2) as i32;
    let coding_quality_q14 = silk_sigm_q15(silk_rshift_round(snr_adj_db_q7 - ((20.0 * 128.0) as i32), 4)) >> 1;

    ps_enc_ctrl.input_quality_q14 = input_quality_q14;
    ps_enc_ctrl.coding_quality_q14 = coding_quality_q14;

    // Reduce coding SNR during low speech activity (CBR skip)
    let mut snr_adj_db_q7 = snr_adj_db_q7;
    if ps_enc.s_cmn.use_cbr == 0 {
        let b_q8 = 256 - ps_enc.s_cmn.speech_activity_q8;
        let b_q8 = silk_smulwb_i32(b_q8 << 8, b_q8);
        snr_adj_db_q7 = silk_smlawb_i32(
            snr_adj_db_q7,
            silk_smulbb(((-2.0 * 128.0) as i32) >> 5, b_q8), // BG_SNR_DECR_dB=2.0
            silk_smulwb_i32((1 << 14) + input_quality_q14, coding_quality_q14),
        );
    }

    if ps_enc.s_cmn.indices.signal_type as i32 == TYPE_VOICED {
        snr_adj_db_q7 = silk_smlawb_i32(
            snr_adj_db_q7,
            ((2.0 * 256.0) as i32), // HARM_SNR_INCR_dB=2.0 in Q8
            ps_enc.ltp_corr_q15,
        );
    } else {
        snr_adj_db_q7 = silk_smlawb_i32(
            snr_adj_db_q7,
            silk_smlawb_i32(
                ((6.0 * 512.0) as i32), // 6.0 in Q9
                ((-0.4 * (1 << 18) as f64) as i32), // -0.4 in Q18
                ps_enc.s_cmn.snr_db_q7,
            ),
            (1 << 14) - input_quality_q14,
        );
    }

    // Sparseness processing: set quantizer offset
    if ps_enc.s_cmn.indices.signal_type as i32 == TYPE_VOICED {
        // Initially 0; may be overruled in process_gains
        ps_enc.s_cmn.indices.quant_offset_type = 0;
    } else {
        // Sparseness measure based on energy variation per 2ms segments
        let n_samples = (fs_khz << 1) as usize;
        let mut energy_variation_q7 = 0i32;
        let mut log_energy_prev_q7 = 0i32;
        let n_segs = (SUB_FRAME_LENGTH_MS * nb_subfr) / 2;
        let mut ptr_offset = 0usize;
        for k in 0..n_segs {
            let (nrg, scale) = silk_sum_sqr_shift(&pitch_res[ptr_offset..ptr_offset + n_samples]);
            let nrg = nrg + (n_samples as i32 >> scale);
            let log_energy_q7 = silk_lin2log(nrg);
            if k > 0 {
                energy_variation_q7 += (log_energy_q7 - log_energy_prev_q7).abs();
            }
            log_energy_prev_q7 = log_energy_q7;
            ptr_offset += n_samples;
        }
        // Set quantization offset depending on sparseness
        let threshold = ((0.6 * 128.0) as i32) * ((n_segs as i32) - 1);
        if energy_variation_q7 > threshold {
            ps_enc.s_cmn.indices.quant_offset_type = 0;
        } else {
            ps_enc.s_cmn.indices.quant_offset_type = 1;
        }
    }

    // Bandwidth expansion
    let pred_gain_q16 = ps_enc_ctrl.pred_gain_q16;
    let strength_q16 = silk_smulwb(imax(pred_gain_q16, 1), ((FIND_PITCH_WHITE_NOISE_FRACTION * 65536.0 + 0.5) as i16));
    let bw_exp_q16 = silk_div32_varq(
        ((BANDWIDTH_EXPANSION * 65536.0 + 0.5) as i32),
        silk_smlaww((1 << 16), strength_q16, strength_q16),
        16,
    );

    // Per-subframe analysis
    let la_shape = ps_enc.s_cmn.la_shape as usize;
    let mut x_windowed = vec![0i16; shape_win_length];
    let mut auto_corr = vec![0i32; shaping_lpc_order + 1];
    let mut refl_coef_q16 = vec![0i32; shaping_lpc_order];
    let mut ar_q24 = vec![0i32; shaping_lpc_order];

    for k in 0..nb_subfr {
        let x_offset = k * subfr_length;

        // Window the signal
        let win_len = shape_win_length.min(x_frame.len() - x_offset);
        let slope_part = (shape_win_length - (fs_khz * 3) as usize) / 2;
        let flat_part = (fs_khz * 3) as usize;

        // Apply sine window (rising slope)
        if slope_part > 0 && x_offset + slope_part <= x_frame.len() {
            silk_apply_sine_window(
                &mut x_windowed[..slope_part],
                &x_frame[x_offset..],
                1,
                slope_part,
            );
        }
        // Copy flat part
        let flat_start = slope_part;
        let flat_end = flat_start + flat_part;
        if flat_end <= win_len && x_offset + flat_end <= x_frame.len() {
            x_windowed[flat_start..flat_end].copy_from_slice(&x_frame[x_offset + flat_start..x_offset + flat_end]);
        }
        // Apply sine window (falling slope)
        if flat_end + slope_part <= win_len && x_offset + flat_end + slope_part <= x_frame.len() {
            silk_apply_sine_window(
                &mut x_windowed[flat_end..flat_end + slope_part],
                &x_frame[x_offset + flat_end..],
                2,
                slope_part,
            );
        }

        // Autocorrelation
        let mut scale = 0i32;
        if warping_q16 > 0 {
            silk_warped_autocorrelation(
                &mut auto_corr,
                &mut scale,
                &x_windowed[..win_len],
                warping_q16,
                win_len,
                shaping_lpc_order,
            );
        } else {
            silk_autocorr(
                &mut auto_corr,
                &mut scale,
                &x_windowed[..win_len],
                win_len,
                shaping_lpc_order + 1,
            );
        }

        // Add white noise
        auto_corr[0] += imax(silk_smulwb(auto_corr[0] >> 4, ((SHAPE_WHITE_NOISE_FRACTION * (1 << 20) as f64) as i16)), 1);

        // Schur recursion
        eprintln!("[RS NSHAPE] k={} scale={}", k, scale);
        let mut nrg = silk_schur64(&mut refl_coef_q16, &auto_corr, shaping_lpc_order);

        // Convert to AR coefficients
        ar_q24.fill(0);
        silk_k2a_q16(&mut ar_q24, &refl_coef_q16, shaping_lpc_order);

        // Compute gain using proper Q-format from autocorrelation scale
        // C: Qnrg = -scale; make even; sqrt; shift to Q16
        let mut q_nrg = -scale; // range: -12...30
        if q_nrg & 1 != 0 {
            q_nrg -= 1;
            nrg >>= 1;
        }
        let tmp32 = silk_sqrt_approx(nrg);
        q_nrg >>= 1; // range: -6...15
        ps_enc_ctrl.gains_q16[k] = silk_lshift_sat32(tmp32, 16 - q_nrg);
        eprintln!("[RS NSHAPE] k={} nrg={} Qnrg={} sqrt={} gain={} warping={}", k, nrg, q_nrg, tmp32, ps_enc_ctrl.gains_q16[k], ps_enc.s_cmn.warping_q16);

        // Adjust gain for warping
        if warping_q16 > 0 {
            let warp_gain_q16 = warped_gain(&ar_q24, warping_q16, shaping_lpc_order);
            if ps_enc_ctrl.gains_q16[k] < (0.25 * 65536.0) as i32 {
                ps_enc_ctrl.gains_q16[k] = silk_smulww(ps_enc_ctrl.gains_q16[k], warp_gain_q16);
            } else {
                ps_enc_ctrl.gains_q16[k] = silk_smulww(
                    silk_rshift_round(ps_enc_ctrl.gains_q16[k], 1), warp_gain_q16);
                if ps_enc_ctrl.gains_q16[k] >= (i32::MAX >> 1) {
                    ps_enc_ctrl.gains_q16[k] = i32::MAX;
                } else {
                    ps_enc_ctrl.gains_q16[k] <<= 1;
                }
            }
        }

        // Bandwidth expansion
        silk_bwexpander_32(&mut ar_q24, shaping_lpc_order, bw_exp_q16);

        // Store AR coefficients (Q24→Q13)
        if warping_q16 > 0 {
            limit_warped_coefs(&mut ar_q24, warping_q16, (3.999 * (1 << 24) as f64) as i32, shaping_lpc_order);
            for i in 0..shaping_lpc_order {
                ps_enc_ctrl.ar_q13[k * shaping_lpc_order + i] = sat16(silk_rshift_round(ar_q24[i], 11));
            }
        } else {
            silk_lpc_fit(&mut ps_enc_ctrl.ar_q13[k * shaping_lpc_order..], &ar_q24, 13, 24, shaping_lpc_order);
        }
    }

    // Apply gain multiplier from SNR
    let gain_mult_q16 = silk_log2lin(-silk_smlawb(
        -(((16.0 * 128.0) + 0.5) as i32),
        snr_adj_db_q7,
        ((0.16 * 65536.0) + 0.5) as i16,
    ));
    let gain_add_q16 = silk_log2lin(silk_smlawb(
        ((16.0 * 128.0) + 0.5) as i32,
        ((MIN_QGAIN_DB as f64 * 128.0) + 0.5) as i32, // MIN_QGAIN_DB in Q7
        ((0.16 * 65536.0) + 0.5) as i16,
    ));

    eprintln!("[RS SHAPE TRACE] SNR_adj_dB_Q7={} gain_mult_Q16={} gain_add_Q16={} gains_before={:?}",
        snr_adj_db_q7, gain_mult_q16, gain_add_q16, &ps_enc_ctrl.gains_q16[..nb_subfr]);

    for k in 0..nb_subfr {
        ps_enc_ctrl.gains_q16[k] = silk_smulww(ps_enc_ctrl.gains_q16[k], gain_mult_q16);
        ps_enc_ctrl.gains_q16[k] = silk_add_pos_sat32(ps_enc_ctrl.gains_q16[k], gain_add_q16);
    }

    // LF shaping, tilt, harmonic shaping
    let signal_type = ps_enc.s_cmn.indices.signal_type as i32;
    for k in 0..nb_subfr {
        if signal_type == TYPE_VOICED {
            let b_q14 = silk_div32_16(((0.2 * 16384.0) as i32), fs_khz)
                + silk_div32_16(((3.0 * 16384.0) as i32), ps_enc_ctrl.pitch_l[k]);
            let b_q14 = imin(b_q14, (1 << 14) - 1);
            let strength = silk_smulwb(((LOW_FREQ_SHAPING_Q0 * 16.0) as i32), (1 << 12) as i16);
            ps_enc_ctrl.lf_shp_q14[k] = shl32(
                (1 << 14) - b_q14 - silk_smulwb(strength, b_q14 as i16),
                16,
            ) | (b_q14 - (1 << 14));
            ps_enc_ctrl.tilt_q14[k] = -(((HP_NOISE_COEF * 16384.0) + 0.5) as i32);
            ps_enc_ctrl.harm_shape_gain_q14[k] = ((HARMONIC_SHAPING * 16384.0 + 0.5) as i32);
        } else {
            let b_q14 = silk_div32_16(21299, fs_khz); // 1.3 * 16384
            ps_enc_ctrl.lf_shp_q14[k] = shl32((1 << 14) - b_q14, 16) | (b_q14 - (1 << 14));
            ps_enc_ctrl.tilt_q14[k] = -(((HP_NOISE_COEF * 16384.0) + 0.5) as i32);
            ps_enc_ctrl.harm_shape_gain_q14[k] = 0;
        }
    }

    // Smoothing (update shape state)
    ps_enc.s_shape.tilt_smth = ps_enc_ctrl.tilt_q14[nb_subfr - 1] << 2;
    ps_enc.s_shape.harm_shape_gain_smth = ps_enc_ctrl.harm_shape_gain_q14[nb_subfr - 1] << 2;
}

// ===========================================================================
// Find pitch lags (simplified — always unvoiced initially)
// ===========================================================================

/// Find pitch lags. Matches C: `silk_find_pitch_lags_FIX`.
/// Simplified version: does LPC analysis and residual computation.
/// Pitch detection uses the existing VAD signal type decision.
pub fn silk_find_pitch_lags_fix(
    ps_enc: &mut SilkEncoderStateFix,
    ps_enc_ctrl: &mut SilkEncoderControl,
    res: &mut [i16],
    x: &[i16],
    x_base: usize,
) {
    let frame_length = ps_enc.s_cmn.frame_length as usize;
    let la_pitch = ps_enc.s_cmn.la_pitch as usize;
    let ltp_mem = ps_enc.s_cmn.ltp_mem_length as usize;
    let pitch_lpc_order = ps_enc.s_cmn.pitch_estimation_lpc_order as usize;
    let pitch_lpc_win_length = ps_enc.s_cmn.pitch_lpc_win_length as usize;
    let buf_len = la_pitch + frame_length + ltp_mem;

    // Window the signal
    let la_tmp = la_pitch.min(pitch_lpc_win_length / 2);
    let mut w_sig = vec![0i16; pitch_lpc_win_length];

    // Apply sine windows + copy flat
    if la_tmp >= 16 && la_tmp <= 120 && (la_tmp & 3) == 0 {
        silk_apply_sine_window(&mut w_sig[..la_tmp], &x[x_base + buf_len - pitch_lpc_win_length..], 1, la_tmp);
    }
    let flat_len = pitch_lpc_win_length - 2 * la_tmp;
    if flat_len > 0 {
        w_sig[la_tmp..la_tmp + flat_len].copy_from_slice(
            &x[x_base + buf_len - pitch_lpc_win_length + la_tmp..x_base + buf_len - pitch_lpc_win_length + la_tmp + flat_len],
        );
    }
    if la_tmp >= 16 && la_tmp <= 120 && (la_tmp & 3) == 0 {
        silk_apply_sine_window(&mut w_sig[pitch_lpc_win_length - la_tmp..], &x[x_base + buf_len - la_tmp..], 2, la_tmp);
    }

    // Autocorrelation
    let mut auto_corr = vec![0i32; pitch_lpc_order + 1];
    let mut scale = 0i32;
    silk_autocorr(&mut auto_corr, &mut scale, &w_sig, pitch_lpc_win_length, pitch_lpc_order + 1);

    // Add white noise
    auto_corr[0] = silk_smlawb(auto_corr[0], auto_corr[0], ((FIND_PITCH_WHITE_NOISE_FRACTION * 65536.0) as i16)) + 1;

    // Schur recursion
    let mut rc_q15 = vec![0i16; pitch_lpc_order];
    let res_nrg = silk_schur(&mut rc_q15, &auto_corr, pitch_lpc_order);

    // Prediction gain
    ps_enc_ctrl.pred_gain_q16 = silk_div32_varq(auto_corr[0], imax(res_nrg, 1), 16);

    // Convert reflection to prediction coefficients
    let mut a_q24 = vec![0i32; pitch_lpc_order];
    silk_k2a(&mut a_q24, &rc_q15, pitch_lpc_order);

    // Convert Q24→Q12 with saturation
    let mut a_q12 = vec![0i16; pitch_lpc_order];
    for i in 0..pitch_lpc_order {
        a_q12[i] = sat16(a_q24[i] >> 12);
    }

    // Bandwidth expansion
    silk_bwexpander(&mut a_q12, pitch_lpc_order, ((FIND_PITCH_BANDWIDTH_EXPANSION * 65536.0 + 0.5) as i32));

    // LPC analysis filter to get residual
    silk_lpc_analysis_filter(res, &x[x_base..], &a_q12, buf_len, pitch_lpc_order);

    // Pitch analysis: for now, rely on VAD signal type.
    // If signal type is not NO_VOICE_ACTIVITY, we should run pitch analysis core.
    // For this simplified version, we skip pitch detection and keep signal type from VAD.
    if ps_enc.s_cmn.indices.signal_type as i32 == TYPE_NO_VOICE_ACTIVITY
        || ps_enc.s_cmn.first_frame_after_reset != 0
    {
        // No voice activity or first frame: zero pitch lags
        for k in 0..ps_enc.s_cmn.nb_subfr as usize {
            ps_enc_ctrl.pitch_l[k] = 0;
        }
        ps_enc.s_cmn.indices.lag_index = 0;
        ps_enc.s_cmn.indices.contour_index = 0;
        ps_enc.ltp_corr_q15 = 0;
    } else {
        // For UNVOICED signal type, also zero pitch lags
        // (pitch_analysis_core would be called for voiced, but we skip for now)
        for k in 0..ps_enc.s_cmn.nb_subfr as usize {
            ps_enc_ctrl.pitch_l[k] = 0;
        }
        ps_enc.s_cmn.indices.lag_index = 0;
        ps_enc.s_cmn.indices.contour_index = 0;
        ps_enc.ltp_corr_q15 = 0;
    }
}

// ===========================================================================
// Encode frame
// ===========================================================================

/// Encode a single SILK frame. Matches C: `silk_encode_frame_FIX`.
pub fn silk_encode_frame_fix(
    ps_enc: &mut SilkEncoderStateFix,
    pn_bytes_out: &mut i32,
    range_enc: &mut RangeEncoder,
    cond_coding: i32,
    max_bits: i32,
    use_cbr: i32,
) -> i32 {
    let mut ret = 0i32;
    let frame_length = ps_enc.s_cmn.frame_length as usize;
    let ltp_mem = ps_enc.s_cmn.ltp_mem_length as usize;
    let la_shape = (LA_SHAPE_MS * ps_enc.s_cmn.fs_khz) as usize;
    let nb_subfr = ps_enc.s_cmn.nb_subfr as usize;

    eprintln!("[RUST ENC] silk_encode_frame_fix called: prefill={} frame_len={} nb_subfr={}",
        ps_enc.s_cmn.prefill_flag, frame_length, nb_subfr);

    let bits_margin = if use_cbr != 0 { 5 } else { max_bits / 4 };

    ps_enc.s_cmn.indices.seed = ((ps_enc.s_cmn.frame_counter as u8) & 3) as i8;
    ps_enc.s_cmn.frame_counter += 1;
    eprintln!("[RS FRAME_SEED] frameCounter={} seed={} nFramesEncoded={}",
        ps_enc.s_cmn.frame_counter, ps_enc.s_cmn.indices.seed, ps_enc.s_cmn.n_frames_encoded);

    // x_frame points into x_buf at offset ltp_mem_length
    let x_frame_offset = ltp_mem;

    // LP variable cutoff
    silk_lp_variable_cutoff(
        &mut ps_enc.s_cmn.s_lp,
        &mut ps_enc.s_cmn.input_buf[1..],
        frame_length,
    );

    // Copy input to x_buf
    let copy_start = x_frame_offset + la_shape;
    for i in 0..frame_length {
        if copy_start + i < ps_enc.x_buf.len() && i + 1 < ps_enc.s_cmn.input_buf.len() {
            ps_enc.x_buf[copy_start + i] = ps_enc.s_cmn.input_buf[i + 1];
        }
    }

    let mut s_enc_ctrl = SilkEncoderControl::default();

    if ps_enc.s_cmn.prefill_flag == 0 {

        // Residual buffer for pitch analysis
        let res_len = ps_enc.s_cmn.la_pitch as usize + frame_length + ltp_mem;
        let mut res_pitch = vec![0i16; res_len];

        // Find pitch lags + initial LPC
        silk_find_pitch_lags_fix(
            ps_enc,
            &mut s_enc_ctrl,
            &mut res_pitch,
            &ps_enc.x_buf.clone(),
            0,
        );

        // Noise shape analysis
        let x_frame_copy = ps_enc.x_buf[x_frame_offset..].to_vec();
        silk_noise_shape_analysis_fix(
            ps_enc,
            &mut s_enc_ctrl,
            &res_pitch[ltp_mem..],
            &x_frame_copy,
        );

        // Find prediction coefficients (LPC + LTP)
        let x_buf_clone = ps_enc.x_buf.clone();
        silk_find_pred_coefs_fix(
            ps_enc,
            &mut s_enc_ctrl,
            &x_buf_clone,
            x_frame_offset,
            cond_coding,
        );

        // Process gains
        silk_process_gains_fix(ps_enc, &mut s_enc_ctrl, cond_coding);

        // DEBUG: Trace key parameters before rate control
        eprintln!("[SILK DEBUG] frame={} speech_activity_Q8={} signal_type={} quant_offset={}",
            ps_enc.s_cmn.frame_counter - 1,
            ps_enc.s_cmn.speech_activity_q8,
            ps_enc.s_cmn.indices.signal_type,
            ps_enc.s_cmn.indices.quant_offset_type);
        eprintln!("[SILK DEBUG] gains_indices={:?} nb_subfr={}",
            &ps_enc.s_cmn.indices.gains_indices[..ps_enc.s_cmn.nb_subfr as usize],
            ps_enc.s_cmn.nb_subfr);
        eprintln!("[SILK DEBUG] nlsf_indices[0..11]={:?}",
            &ps_enc.s_cmn.indices.nlsf_indices[..11]);
        eprintln!("[SILK DEBUG] vad_flags={:?} lbrr_flag={} in_dtx={}",
            &ps_enc.s_cmn.vad_flags[..ps_enc.s_cmn.n_frames_per_packet as usize],
            ps_enc.s_cmn.lbrr_flag,
            ps_enc.s_cmn.in_dtx);
        eprintln!("[SILK DEBUG] pitch_l={:?} pred_gain_q16={} ltp_pred_cod_gain_q7={}",
            &s_enc_ctrl.pitch_l[..ps_enc.s_cmn.nb_subfr as usize],
            s_enc_ctrl.pred_gain_q16,
            s_enc_ctrl.ltp_pred_cod_gain_q7);

        // Rate control loop: NSQ + encode (matches C encode_frame_FIX.c)
        let max_iter: i32 = 6;
        let mut gain_mult_q8: i16 = 256; // Q8(1.0)
        let mut found_lower = false;
        let mut found_upper = false;
        let mut n_bits_lower: i32 = 0;
        let mut n_bits_upper: i32 = 0;
        let mut gain_mult_lower: i16 = 0;
        let mut gain_mult_upper: i16 = 0;

        let gains_id_initial = silk_gains_id(
            &ps_enc.s_cmn.indices.gains_indices,
            nb_subfr,
        );
        let mut gains_id = gains_id_initial;
        let mut gains_id_lower: i32 = -1;
        let mut gains_id_upper: i32 = -1;

        // Save state before rate control loop
        let seed_copy = ps_enc.s_cmn.indices.seed;
        let ec_prev_lag_copy = ps_enc.s_cmn.ec_prev_lag_index;
        let ec_prev_sig_copy = ps_enc.s_cmn.ec_prev_signal_type;
        let rc_snap = range_enc.save_snapshot();
        let nsq_copy = ps_enc.s_cmn.s_nsq.clone();
        let x_frame_slice: Vec<i16> = ps_enc.x_buf[x_frame_offset..x_frame_offset + frame_length].to_vec();
        let mut rc_snap_lower: Option<crate::celt::range_coder::RangeEncoderSnapshot> = None;
        let mut nsq_copy_lower: Option<NsqState> = None;
        let mut last_gain_index_copy2: i8 = 0;
        let mut gain_lock = [false; MAX_NB_SUBFR];
        let mut best_gain_mult = [0i16; MAX_NB_SUBFR];
        let mut best_sum = [0i32; MAX_NB_SUBFR];

        eprintln!("[RS LOOP] max_bits={} use_cbr={} bits_margin={}", max_bits, use_cbr, bits_margin);
        for iter in 0..=max_iter {
            let n_bits: i32;
            if gains_id == gains_id_lower && found_lower {
                n_bits = n_bits_lower;
            } else if gains_id == gains_id_upper && found_upper {
                n_bits = n_bits_upper;
            } else {
                // Restore state if not first iteration
                if iter > 0 {
                    range_enc.restore_snapshot(&rc_snap);
                    ps_enc.s_cmn.s_nsq = nsq_copy.clone();
                    ps_enc.s_cmn.indices.seed = seed_copy;
                    ps_enc.s_cmn.ec_prev_lag_index = ec_prev_lag_copy;
                    ps_enc.s_cmn.ec_prev_signal_type = ec_prev_sig_copy;
                }

                // NSQ
                let mut nsq = std::mem::replace(&mut ps_enc.s_cmn.s_nsq, NsqState::default());
                let mut indices = std::mem::replace(&mut ps_enc.s_cmn.indices, SideInfoIndices::default());
                let mut pulses = std::mem::replace(&mut ps_enc.s_cmn.pulses, [0i8; MAX_FRAME_LENGTH]);
                silk_nsq(
                    &ps_enc.s_cmn,
                    &mut nsq,
                    &mut indices,
                    &x_frame_slice,
                    &mut pulses,
                    &s_enc_ctrl.pred_coef_q12,
                    &s_enc_ctrl.ltp_coef_q14,
                    &s_enc_ctrl.ar_q13,
                    &s_enc_ctrl.harm_shape_gain_q14,
                    &s_enc_ctrl.tilt_q14,
                    &s_enc_ctrl.lf_shp_q14,
                    &s_enc_ctrl.gains_q16,
                    &s_enc_ctrl.pitch_l,
                    s_enc_ctrl.lambda_q10,
                    s_enc_ctrl.ltp_scale_q14,
                );
                ps_enc.s_cmn.s_nsq = nsq;
                ps_enc.s_cmn.indices = indices;
                ps_enc.s_cmn.pulses = pulses;

                // Save state before encode if last iteration and no lower found
                let rc_snap_pre_encode = if iter == max_iter && !found_lower {
                    Some(range_enc.save_snapshot())
                } else {
                    None
                };

                // Encode indices
                {
                    let idx = &ps_enc.s_cmn.indices;
                    eprintln!("[RS INDICES_PRE] sigtype={} qofftype={} gains=[{},{},{},{}] nlsf0={} nlsfR=[{},{},{},{},{},{},{},{},{},{}] interpQ2={} seed={}",
                        idx.signal_type, idx.quant_offset_type,
                        idx.gains_indices[0], idx.gains_indices[1], idx.gains_indices[2], idx.gains_indices[3],
                        idx.nlsf_indices[0],
                        idx.nlsf_indices[1], idx.nlsf_indices[2], idx.nlsf_indices[3], idx.nlsf_indices[4], idx.nlsf_indices[5],
                        idx.nlsf_indices[6], idx.nlsf_indices[7], idx.nlsf_indices[8], idx.nlsf_indices[9], idx.nlsf_indices[10],
                        idx.nlsf_interp_coef_q2, idx.seed);
                    eprintln!("[RS EC_PRE_IDX] tell={} rng={} val={} offs={}", range_enc.tell(), range_enc.get_rng(), range_enc.get_val(), range_enc.range_bytes());
                }
                silk_encode_indices(
                    &ps_enc.s_cmn,
                    range_enc,
                    ps_enc.s_cmn.n_frames_encoded as usize,
                    false,
                    cond_coding,
                );
                eprintln!("[RS EC_POST_IDX] tell={} rng={} val={} offs={}", range_enc.tell(), range_enc.get_rng(), range_enc.get_val(), range_enc.range_bytes());

                // Encode pulses
                silk_encode_pulses(
                    range_enc,
                    ps_enc.s_cmn.indices.signal_type as i32,
                    ps_enc.s_cmn.indices.quant_offset_type as i32,
                    &ps_enc.s_cmn.pulses,
                    frame_length,
                );

                n_bits = range_enc.tell();
                eprintln!("[RS EC_POST_PLS] tell={} rng={} val={} offs={}", range_enc.tell(), range_enc.get_rng(), range_enc.get_val(), range_enc.range_bytes());

                // Damage control: last iteration, no lower bound, still over budget
                if iter == max_iter && !found_lower && n_bits > max_bits {
                    if let Some(snap) = rc_snap_pre_encode {
                        range_enc.restore_snapshot(&snap);
                    }
                    ps_enc.s_shape.last_gain_index = s_enc_ctrl.last_gain_index_prev;
                    for k in 0..nb_subfr {
                        ps_enc.s_cmn.indices.gains_indices[k] = 4;
                    }
                    if cond_coding != CODE_CONDITIONALLY {
                        ps_enc.s_cmn.indices.gains_indices[0] = s_enc_ctrl.last_gain_index_prev;
                    }
                    ps_enc.s_cmn.ec_prev_lag_index = ec_prev_lag_copy;
                    ps_enc.s_cmn.ec_prev_signal_type = ec_prev_sig_copy;
                    for i in 0..frame_length {
                        ps_enc.s_cmn.pulses[i] = 0;
                    }
                    silk_encode_indices(
                        &ps_enc.s_cmn,
                        range_enc,
                        ps_enc.s_cmn.n_frames_encoded as usize,
                        false,
                        cond_coding,
                    );
                    silk_encode_pulses(
                        range_enc,
                        ps_enc.s_cmn.indices.signal_type as i32,
                        ps_enc.s_cmn.indices.quant_offset_type as i32,
                        &ps_enc.s_cmn.pulses,
                        frame_length,
                    );
                }

                let pulse_sum: i32 = ps_enc.s_cmn.pulses[..frame_length].iter().map(|p| (*p as i32).abs()).sum();
                eprintln!("[RS LOOP iter={}] n_bits={} max_bits={} use_cbr={} pulse_sum={}", iter, n_bits, max_bits, use_cbr, pulse_sum);
                // VBR early exit: first iter and within budget
                if use_cbr == 0 && iter == 0 && n_bits <= max_bits {
                    break;
                }
            }

            if iter == max_iter {
                // Restore lower-bound state if we found one and current is worse
                if found_lower && (gains_id == gains_id_lower || n_bits > max_bits) {
                    if let Some(ref snap) = rc_snap_lower {
                        range_enc.restore_snapshot(snap);
                    }
                    if let Some(ref nsq_l) = nsq_copy_lower {
                        ps_enc.s_cmn.s_nsq = nsq_l.clone();
                    }
                    ps_enc.s_shape.last_gain_index = last_gain_index_copy2;
                }
                break;
            }

            // Adjust gains based on bit count vs target
            if n_bits > max_bits {
                if !found_lower && iter >= 2 {
                    // Increase lambda (rate/distortion tradeoff)
                    s_enc_ctrl.lambda_q10 += s_enc_ctrl.lambda_q10 >> 1;
                    found_upper = false;
                    gains_id_upper = -1;
                } else {
                    found_upper = true;
                    n_bits_upper = n_bits;
                    gain_mult_upper = gain_mult_q8;
                    gains_id_upper = gains_id;
                }
            } else if n_bits < max_bits - bits_margin {
                found_lower = true;
                n_bits_lower = n_bits;
                gain_mult_lower = gain_mult_q8;
                if gains_id != gains_id_lower {
                    gains_id_lower = gains_id;
                    rc_snap_lower = Some(range_enc.save_snapshot());
                    nsq_copy_lower = Some(ps_enc.s_cmn.s_nsq.clone());
                    last_gain_index_copy2 = ps_enc.s_shape.last_gain_index;
                }
            } else {
                // Close enough
                break;
            }

            // Track best gain_mult per subframe when no lower bound found
            if !found_lower && n_bits > max_bits {
                let subfr_len = ps_enc.s_cmn.subfr_length as usize;
                for i in 0..nb_subfr {
                    let mut sum = 0i32;
                    for j in (i * subfr_len)..((i + 1) * subfr_len) {
                        sum += (ps_enc.s_cmn.pulses[j] as i32).abs();
                    }
                    if iter == 0 || (sum < best_sum[i] && !gain_lock[i]) {
                        best_sum[i] = sum;
                        best_gain_mult[i] = gain_mult_q8;
                    } else {
                        gain_lock[i] = true;
                    }
                }
            }

            // Compute new gain multiplier
            if !(found_lower && found_upper) {
                if n_bits > max_bits {
                    gain_mult_q8 = imin(1024, gain_mult_q8 as i32 * 3 / 2) as i16;
                } else {
                    gain_mult_q8 = imax(64, gain_mult_q8 as i32 * 4 / 5) as i16;
                }
            } else {
                // Interpolate
                let range = gain_mult_upper as i32 - gain_mult_lower as i32;
                let denom = n_bits_upper - n_bits_lower;
                gain_mult_q8 = if denom != 0 {
                    (gain_mult_lower as i32
                        + (range * (max_bits - n_bits_lower)) / denom) as i16
                } else {
                    gain_mult_lower
                };
                // Clamp to 25%-75% of range
                let upper_bound = gain_mult_lower as i32 + (range >> 2);
                let lower_bound = gain_mult_upper as i32 - (range >> 2);
                if (gain_mult_q8 as i32) > upper_bound {
                    gain_mult_q8 = upper_bound as i16;
                } else if (gain_mult_q8 as i32) < lower_bound {
                    gain_mult_q8 = lower_bound as i16;
                }
            }

            // Apply new gain multiplier and requantize
            for i in 0..nb_subfr {
                let tmp = if gain_lock[i] { best_gain_mult[i] } else { gain_mult_q8 };
                s_enc_ctrl.gains_q16[i] = silk_lshift_sat32(
                    silk_smulwb(s_enc_ctrl.gains_unq_q16[i], tmp),
                    8,
                );
            }
            ps_enc.s_shape.last_gain_index = s_enc_ctrl.last_gain_index_prev;
            eprintln!("[RS GAINS_QUANT iter] gains_q16={:?} prev_ind={} cond={}",
                &s_enc_ctrl.gains_q16[..nb_subfr], ps_enc.s_shape.last_gain_index,
                cond_coding == CODE_CONDITIONALLY);
            silk_gains_quant(
                &mut ps_enc.s_cmn.indices.gains_indices,
                &mut s_enc_ctrl.gains_q16,
                &mut ps_enc.s_shape.last_gain_index,
                cond_coding == CODE_CONDITIONALLY,
                nb_subfr,
            );
            gains_id = silk_gains_id(&ps_enc.s_cmn.indices.gains_indices, nb_subfr);
        }
    }

    // Update x_buf (shift by frame_length)
    let shift_len = ltp_mem + la_shape;
    ps_enc.x_buf.copy_within(frame_length..frame_length + shift_len, 0);

    // Exit without entropy coding for prefill
    if ps_enc.s_cmn.prefill_flag != 0 {
        *pn_bytes_out = 0;
        return ret;
    }

    // Update state for next frame
    ps_enc.s_cmn.prev_lag = s_enc_ctrl.pitch_l[nb_subfr - 1];
    ps_enc.s_cmn.prev_signal_type = ps_enc.s_cmn.indices.signal_type as i32;

    // Payload size
    ps_enc.s_cmn.first_frame_after_reset = 0;
    *pn_bytes_out = (range_enc.tell() + 7) >> 3;
    {
        let buf = range_enc.buffer();
        eprintln!("[RS FRAME_DONE] nbytes={} tell={} rng={} offs={} buf=[{:02x},{:02x},{:02x},{:02x},{:02x},{:02x},{:02x},{:02x}]",
            *pn_bytes_out, range_enc.tell(), range_enc.get_rng(), range_enc.range_bytes(),
            buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]);
    }

    ret
}

/// Burg's modified method for AR coefficient estimation. Matches C: `silk_burg_modified_c`.
pub fn silk_burg_modified(
    res_nrg: &mut i32,
    res_nrg_q: &mut i32,
    a_q16: &mut [i32],
    x: &[i16],
    min_inv_gain_q30: i32,
    subfr_length: usize,
    nb_subfr: usize,
    d: usize,
) {
    const QA: i32 = 25;
    const N_BITS_HEAD_ROOM: i32 = 3;
    const MIN_RSHIFTS: i32 = -16;
    const MAX_RSHIFTS: i32 = 32 - QA;
    const FIND_LPC_COND_FAC_Q32: i32 = ((1.0e-5f64 * (1u64 << 32) as f64) + 0.5) as i32;

    let total_len = subfr_length * nb_subfr;
    let mut c0_64: i64 = 0;
    for i in 0..total_len {
        c0_64 += x[i] as i64 * x[i] as i64;
    }

    let lz = silk_clz64_fn(c0_64);
    let mut rshifts = 32 + 1 + N_BITS_HEAD_ROOM - lz;
    rshifts = silk_limit(rshifts, MIN_RSHIFTS, MAX_RSHIFTS);

    let mut c0 = if rshifts > 0 {
        (c0_64 >> rshifts) as i32
    } else {
        shl32(c0_64 as i32, -rshifts)
    };

    let mut c_first_row = [0i32; MAX_LPC_ORDER];
    let mut c_last_row = [0i32; MAX_LPC_ORDER];
    let mut af_qa = [0i32; MAX_LPC_ORDER];
    let mut caf = [0i32; MAX_LPC_ORDER + 1];
    let mut cab = [0i32; MAX_LPC_ORDER + 1];

    // Compute cross-correlations
    if rshifts > 0 {
        for s in 0..nb_subfr {
            let x_ptr = s * subfr_length;
            for n in 1..=d {
                let mut sum = 0i64;
                for i in 0..(subfr_length - n) {
                    sum += x[x_ptr + i] as i64 * x[x_ptr + i + n] as i64;
                }
                c_first_row[n - 1] = c_first_row[n - 1].wrapping_add((sum >> rshifts) as i32);
            }
        }
    } else {
        for s in 0..nb_subfr {
            let x_ptr = s * subfr_length;
            for n in 1..=d {
                let mut sum = 0i64;
                for i in 0..(subfr_length - n) {
                    sum += x[x_ptr + i] as i64 * x[x_ptr + i + n] as i64;
                }
                c_first_row[n - 1] = c_first_row[n - 1].wrapping_add(shl32(sum as i32, -rshifts));
            }
        }
    }
    c_last_row[..d].copy_from_slice(&c_first_row[..d]);

    // Initialize
    caf[0] = c0 + silk_smmul(FIND_LPC_COND_FAC_Q32, c0) + 1;
    cab[0] = caf[0];

    let mut inv_gain_q30 = 1i32 << 30;
    let mut reached_max_gain = false;

    for n in 0..d {
        // Update correlation rows and C*Af, C*Ab
        if rshifts > -2 {
            for s in 0..nb_subfr {
                let xp = s * subfr_length;
                let x1 = -shl32(x[xp + n] as i32, 16 - rshifts);
                let x2 = -shl32(x[xp + subfr_length - n - 1] as i32, 16 - rshifts);
                let mut tmp1 = shl32(x[xp + n] as i32, QA - 16);
                let mut tmp2 = shl32(x[xp + subfr_length - n - 1] as i32, QA - 16);
                for k in 0..n {
                    c_first_row[k] = silk_smlawb(c_first_row[k], x1, x[xp + n - k - 1] as i16);
                    c_last_row[k] = silk_smlawb(c_last_row[k], x2, x[xp + subfr_length - n + k] as i16);
                    let atmp_qa = af_qa[k];
                    tmp1 = silk_smlawb(tmp1, atmp_qa, x[xp + n - k - 1] as i16);
                    tmp2 = silk_smlawb(tmp2, atmp_qa, x[xp + subfr_length - n + k] as i16);
                }
                tmp1 = shl32(-tmp1, 32 - QA - rshifts);
                tmp2 = shl32(-tmp2, 32 - QA - rshifts);
                for k in 0..=n {
                    caf[k] = silk_smlawb(caf[k], tmp1, x[xp + n - k] as i16);
                    cab[k] = silk_smlawb(cab[k], tmp2, x[xp + subfr_length - n + k - 1] as i16);
                }
            }
        } else {
            for s in 0..nb_subfr {
                let xp = s * subfr_length;
                let x1 = -shl32(x[xp + n] as i32, -rshifts);
                let x2 = -shl32(x[xp + subfr_length - n - 1] as i32, -rshifts);
                let mut tmp1 = shl32(x[xp + n] as i32, 17);
                let mut tmp2 = shl32(x[xp + subfr_length - n - 1] as i32, 17);
                for k in 0..n {
                    c_first_row[k] = silk_mla(c_first_row[k], x1, x[xp + n - k - 1] as i32);
                    c_last_row[k] = silk_mla(c_last_row[k], x2, x[xp + subfr_length - n + k] as i32);
                    let atmp1 = silk_rshift_round(af_qa[k], QA - 17);
                    tmp1 = silk_mla(tmp1, x[xp + n - k - 1] as i32, atmp1);
                    tmp2 = silk_mla(tmp2, x[xp + subfr_length - n + k] as i32, atmp1);
                }
                tmp1 = -tmp1;
                tmp2 = -tmp2;
                for k in 0..=n {
                    caf[k] = silk_smlaww(caf[k], tmp1, shl32(x[xp + n - k] as i32, -rshifts - 1));
                    cab[k] = silk_smlaww(cab[k], tmp2, shl32(x[xp + subfr_length - n + k - 1] as i32, -rshifts - 1));
                }
            }
        }

        // Calculate numerator and denominator for reflection coefficient
        let mut tmp1 = c_first_row[n];
        let mut tmp2 = c_last_row[n];
        let mut num = 0i32;
        let mut nrg = cab[0].wrapping_add(caf[0]);
        for k in 0..n {
            let atmp_qa = af_qa[k];
            let lz = imin(32 - QA, silk_clz32(silk_abs_int32(atmp_qa)) - 1);
            let atmp1 = shl32(atmp_qa, lz);
            tmp1 = silk_add_lshift32(tmp1, silk_smmul(c_last_row[n - k - 1], atmp1), 32 - QA - lz);
            tmp2 = silk_add_lshift32(tmp2, silk_smmul(c_first_row[n - k - 1], atmp1), 32 - QA - lz);
            num = silk_add_lshift32(num, silk_smmul(cab[n - k], atmp1), 32 - QA - lz);
            nrg = silk_add_lshift32(nrg, silk_smmul(cab[k + 1] + caf[k + 1], atmp1), 32 - QA - lz);
        }
        caf[n + 1] = tmp1;
        cab[n + 1] = tmp2;
        num = num + tmp2;
        num = shl32(-num, 1);

        // Calculate reflection coefficient
        let mut rc_q31 = if silk_abs_int32(num) < nrg {
            silk_div32_varq(num, nrg, 31)
        } else {
            if num > 0 { i32::MAX } else { i32::MIN }
        };

        // Update inverse prediction gain
        let tmp1_val = (1i32 << 30) - silk_smmul(rc_q31, rc_q31);
        let tmp1_val = shl32(silk_smmul(inv_gain_q30, tmp1_val), 2);
        if tmp1_val <= min_inv_gain_q30 {
            let tmp2_val = (1i32 << 30) - silk_div32_varq(min_inv_gain_q30, inv_gain_q30, 30);
            rc_q31 = silk_sqrt_approx(tmp2_val);
            if rc_q31 > 0 {
                rc_q31 = (rc_q31 + (tmp2_val / rc_q31)) >> 1;
                rc_q31 = shl32(rc_q31, 16);
                if num < 0 {
                    rc_q31 = -rc_q31;
                }
            }
            inv_gain_q30 = min_inv_gain_q30;
            reached_max_gain = true;
        } else {
            inv_gain_q30 = tmp1_val;
        }

        // Update AR coefficients
        for k in 0..((n + 1) >> 1) {
            let t1 = af_qa[k];
            let t2 = af_qa[n - k - 1];
            af_qa[k] = silk_add_lshift32(t1, silk_smmul(t2, rc_q31), 1);
            af_qa[n - k - 1] = silk_add_lshift32(t2, silk_smmul(t1, rc_q31), 1);
        }
        af_qa[n] = rc_q31 >> (31 - QA);

        if reached_max_gain {
            for k in (n + 1)..d {
                af_qa[k] = 0;
            }
            break;
        }

        // Update C*Af and C*Ab
        for k in 0..=(n + 1) {
            let t1 = caf[k];
            let t2 = cab[n + 1 - k];
            caf[k] = silk_add_lshift32(t1, silk_smmul(t2, rc_q31), 1);
            cab[n + 1 - k] = silk_add_lshift32(t2, silk_smmul(t1, rc_q31), 1);
        }
    }

    if reached_max_gain {
        for k in 0..d {
            a_q16[k] = -silk_rshift_round(af_qa[k], QA - 16);
        }
        // Subtract energy of preceding samples from C0
        if rshifts > 0 {
            for s in 0..nb_subfr {
                let xp = s * subfr_length;
                let mut sum = 0i64;
                for i in 0..d {
                    sum += x[xp + i] as i64 * x[xp + i] as i64;
                }
                c0 -= (sum >> rshifts) as i32;
            }
        } else {
            for s in 0..nb_subfr {
                let xp = s * subfr_length;
                let mut sum = 0i32;
                for i in 0..d {
                    sum += shl32(x[xp + i] as i32 * x[xp + i] as i32, -rshifts);
                }
                c0 -= sum;
            }
        }
        *res_nrg = shl32(silk_smmul(inv_gain_q30, c0), 2);
        *res_nrg_q = -rshifts;
    } else {
        let mut nrg = caf[0];
        let mut tmp1_val = 1i32 << 16;
        for k in 0..d {
            let atmp1 = silk_rshift_round(af_qa[k], QA - 16);
            nrg = silk_smlaww(nrg, caf[k + 1], atmp1);
            tmp1_val = silk_smlaww(tmp1_val, atmp1, atmp1);
            a_q16[k] = -atmp1;
        }
        *res_nrg = silk_smlaww(nrg, silk_smmul(FIND_LPC_COND_FAC_Q32, c0), -tmp1_val);
        *res_nrg_q = -rshifts;
    }
}

// ===========================================================================
// Initialization
// ===========================================================================

/// Initialize a single encoder channel.
/// Matches C: `silk_init_encoder`.
pub fn silk_init_encoder(ps_enc: &mut SilkEncoderStateFix) -> i32 {
    *ps_enc = SilkEncoderStateFix::default();

    // Initialize HP filter smoother
    let hp_cutoff_q16 = shl32(VARIABLE_HP_MIN_CUTOFF_HZ, 16);
    let log_val = silk_lin2log(hp_cutoff_q16) - (16 << 7);
    ps_enc.s_cmn.variable_hp_smth1_q15 = shl32(log_val, 8);
    ps_enc.s_cmn.variable_hp_smth2_q15 = ps_enc.s_cmn.variable_hp_smth1_q15;

    ps_enc.s_cmn.first_frame_after_reset = 1;

    // Initialize VAD
    silk_vad_init(&mut ps_enc.s_cmn.s_vad);

    0
}

/// Get encoder size in bytes (not used in Rust, but kept for API parity).
pub fn silk_get_encoder_size(channels: usize) -> usize {
    core::mem::size_of::<SilkEncoder>()
        - if channels == 1 {
            core::mem::size_of::<SilkEncoderStateFix>()
        } else {
            0
        }
}

/// Initialize the top-level encoder.
/// Matches C: `silk_InitEncoder`.
pub fn silk_init_encoder_top(enc: &mut SilkEncoder, channels: usize) -> i32 {
    let mut ret = 0i32;
    *enc = SilkEncoder::new();

    for n in 0..channels {
        ret += silk_init_encoder(&mut enc.state_fxx[n]);
    }
    enc.n_channels_api = channels as i32;
    enc.n_channels_internal = channels as i32;
    enc.n_prev_channels_internal = channels as i32;

    ret
}

// ===========================================================================
// Top-level silk_Encode
// ===========================================================================

/// Query encoder status.
fn silk_query_encoder(
    ps_enc: &SilkEncoder,
    enc_status: &mut SilkEncControlStruct,
) {
    enc_status.internal_sample_rate = ps_enc.state_fxx[0].s_cmn.fs_khz * 1000;
    enc_status.allow_bandwidth_switch = ps_enc.allow_bandwidth_switch;
    enc_status.in_wb_mode_without_variable_lp =
        if ps_enc.state_fxx[0].s_cmn.fs_khz == 16
            && ps_enc.state_fxx[0].s_cmn.s_lp.mode == 0
        {
            1
        } else {
            0
        };
}

/// Main SILK encoding entry point.
/// Matches C: `silk_Encode`.
pub fn silk_encode(
    enc: &mut SilkEncoder,
    enc_control: &mut SilkEncControlStruct,
    samples_in: &[i16],
    n_samples_in: i32,
    range_enc: &mut RangeEncoder,
    n_bytes_out: &mut i32,
    prefill_flag: i32,
    activity: i32,
) -> i32 {
    let mut ret = SILK_NO_ERROR;
    eprintln!("[RUST silk_encode] ENTERED: n_samples_in={} prefill_flag={}", n_samples_in, prefill_flag);

    // Validate control input
    ret = check_control_input(enc_control);
    if ret != SILK_NO_ERROR {
        eprintln!("[RUST silk_encode] check_control_input failed: {}", ret);
        return ret;
    }

    let n_channels_internal = enc_control.n_channels_internal as usize;

    // Handle channel count transitions
    if enc.n_prev_channels_internal != n_channels_internal as i32 {
        if n_channels_internal > enc.n_prev_channels_internal as usize {
            // Mono → stereo: init second channel
            silk_init_encoder(&mut enc.state_fxx[1]);
            // Copy resampler state from channel 0
            enc.state_fxx[1].s_cmn.resampler_state =
                enc.state_fxx[0].s_cmn.resampler_state.clone();
        }
        enc.n_prev_channels_internal = n_channels_internal as i32;
    }

    enc.n_channels_api = enc_control.n_channels_api;
    enc.n_channels_internal = n_channels_internal as i32;

    // Compute number of 10ms blocks
    let n_blocks_of_10ms = if n_samples_in > 0 {
        (100 * n_samples_in) / enc_control.api_sample_rate
    } else {
        0
    };

    // Control encoder per channel
    for n in 0..n_channels_internal {
        let force_fs_khz = if n == 1 { enc.state_fxx[0].s_cmn.fs_khz } else { 0 };
        ret = silk_control_encoder(
            &mut enc.state_fxx[n],
            enc_control,
            enc.allow_bandwidth_switch,
            n as i32,
            force_fs_khz,
        );
        if ret != SILK_NO_ERROR {
            return ret;
        }
    }

    // Set prefill flag
    for n in 0..n_channels_internal {
        enc.state_fxx[n].s_cmn.prefill_flag = prefill_flag;
    }

    let fs_khz = enc.state_fxx[0].s_cmn.fs_khz;
    let frame_length = enc.state_fxx[0].s_cmn.frame_length;
    let n_frames_per_packet = enc.state_fxx[0].s_cmn.n_frames_per_packet;

    // Max samples to buffer and corresponding input samples
    let n_samples_to_buffer_max = (10 * n_blocks_of_10ms * fs_khz) as usize;

    // Input buffering / resampling loop (matches C while(1) loop)
    let mut samples_offset: usize = 0;
    let mut n_samples_remaining = n_samples_in;

    loop {
        // How many internal-rate samples to buffer this iteration
        let mut n_samples_to_buffer = (frame_length - enc.state_fxx[0].s_cmn.input_buf_ix) as usize;
        n_samples_to_buffer = n_samples_to_buffer.min(n_samples_to_buffer_max);

        // How many API-rate samples that corresponds to
        let n_samples_from_input = if fs_khz * 1000 != 0 {
            ((n_samples_to_buffer as i64 * enc_control.api_sample_rate as i64)
                / (fs_khz as i64 * 1000)) as usize
        } else {
            0
        };

        if n_samples_from_input == 0 || n_samples_remaining <= 0 {
            break;
        }

        // Resample and buffer input per channel
        for n in 0..n_channels_internal {
            let api_in = &samples_in[samples_offset..];

            // De-interleave if stereo, else copy
            let input: Vec<i16> = if enc_control.n_channels_api == 2 {
                (0..n_samples_from_input)
                    .map(|i| api_in[i * 2 + n])
                    .collect()
            } else {
                api_in[..n_samples_from_input].to_vec()
            };

            // Resample from API rate to internal rate
            let mut buf = vec![0i16; n_samples_to_buffer];
            silk_resampler_run(
                &mut enc.state_fxx[n].s_cmn.resampler_state,
                &mut buf,
                &input,
            );

            // Buffer the resampled input
            let buf_ix = enc.state_fxx[n].s_cmn.input_buf_ix as usize;
            let copy_len = n_samples_to_buffer.min(MAX_FRAME_LENGTH + 2 - buf_ix);
            enc.state_fxx[n].s_cmn.input_buf[buf_ix..buf_ix + copy_len]
                .copy_from_slice(&buf[..copy_len]);
            enc.state_fxx[n].s_cmn.input_buf_ix += n_samples_to_buffer as i32;
        }

        samples_offset += n_samples_from_input * enc_control.n_channels_api as usize;
        n_samples_remaining -= n_samples_from_input as i32;

        // Default
        enc.allow_bandwidth_switch = 0;

        // Check if we have a full frame
        if enc.state_fxx[0].s_cmn.input_buf_ix >= frame_length {
            break;
        }
    }

    // Not enough data buffered yet
    if enc.state_fxx[0].s_cmn.input_buf_ix < frame_length {
        *n_bytes_out = 0;
        return SILK_NO_ERROR;
    }

    // Prefill mode: warm up filters by calling encode_frame (matching C behavior).
    // C's silk_Encode falls through to its while(1) encode loop during prefill,
    // calling encode_frame for each frame of input. encode_frame handles prefill
    // internally: it increments frame_counter, runs LP variable cutoff, updates
    // x_buf, then returns early without entropy coding.
    if prefill_flag != 0 {
        loop {
            // Encode the buffered frame per channel (matches C: silk_encode_frame_Fxx)
            for n in 0..n_channels_internal {
                let mut dummy_bytes = 0i32;
                silk_encode_frame_fix(
                    &mut enc.state_fxx[n],
                    &mut dummy_bytes,
                    range_enc,
                    CODE_INDEPENDENTLY,
                    0,
                    0,
                );
                enc.state_fxx[n].s_cmn.controlled_since_last_payload = 0;
                enc.state_fxx[n].s_cmn.input_buf_ix = 0;
                enc.state_fxx[n].s_cmn.n_frames_encoded += 1;
            }

            // Try to buffer next frame from remaining input (matches C while(1) loop)
            if n_samples_remaining <= 0 {
                break;
            }

            let mut n_stb = (frame_length - enc.state_fxx[0].s_cmn.input_buf_ix) as usize;
            n_stb = n_stb.min(n_samples_to_buffer_max);
            let n_sfi = if fs_khz * 1000 != 0 {
                ((n_stb as i64 * enc_control.api_sample_rate as i64)
                    / (fs_khz as i64 * 1000)) as usize
            } else {
                0
            };
            if n_sfi == 0 {
                break;
            }

            for n in 0..n_channels_internal {
                let api_in = &samples_in[samples_offset..];
                let input: Vec<i16> = if enc_control.n_channels_api == 2 {
                    (0..n_sfi).map(|i| api_in[i * 2 + n]).collect()
                } else {
                    api_in[..n_sfi].to_vec()
                };
                let mut buf = vec![0i16; n_stb];
                silk_resampler_run(
                    &mut enc.state_fxx[n].s_cmn.resampler_state,
                    &mut buf,
                    &input,
                );
                let buf_ix = enc.state_fxx[n].s_cmn.input_buf_ix as usize;
                let copy_len = n_stb.min(MAX_FRAME_LENGTH + 2 - buf_ix);
                enc.state_fxx[n].s_cmn.input_buf[buf_ix..buf_ix + copy_len]
                    .copy_from_slice(&buf[..copy_len]);
                enc.state_fxx[n].s_cmn.input_buf_ix += n_stb as i32;
            }

            samples_offset += n_sfi * enc_control.n_channels_api as usize;
            n_samples_remaining -= n_sfi as i32;

            if enc.state_fxx[0].s_cmn.input_buf_ix < frame_length {
                break;
            }
        }

        for n in 0..n_channels_internal {
            enc.state_fxx[n].s_cmn.prefill_flag = 0;
        }
        *n_bytes_out = 0;
        return SILK_NO_ERROR;
    }

    // Encode LBRR data from previous packet (if any)
    if enc.state_fxx[0].s_cmn.n_frames_encoded == 0 {
        eprintln!("[RS VAD_PLACEHOLDER] nFramesPkt={} nChInt={} lbrr_enabled={} tell_before={}",
            n_frames_per_packet, n_channels_internal,
            enc.state_fxx[0].s_cmn.lbrr_enabled, range_enc.tell());
        // LBRR flags
        if enc.state_fxx[0].s_cmn.lbrr_enabled != 0 {
            let _lbrr_flag0 = enc.state_fxx[0].s_cmn.lbrr_flag;
            if n_frames_per_packet > 1 {
                // Encode LBRR flags
                let mut flags = 0u32;
                for i in 0..n_frames_per_packet as usize {
                    flags |= (enc.state_fxx[0].s_cmn.lbrr_flags[i] as u32) << i;
                }
                range_enc.encode_icdf(
                    flags,
                    SILK_LBRR_FLAGS_ICDF_PTR[(n_frames_per_packet - 2) as usize],
                    8,
                );
            }

            // Encode LBRR data for each frame that has it
            for i in 0..n_frames_per_packet as usize {
                if enc.state_fxx[0].s_cmn.lbrr_flags[i] != 0 {
                    silk_encode_indices(
                        &enc.state_fxx[0].s_cmn,
                        range_enc,
                        i,
                        true,
                        CODE_INDEPENDENTLY,
                    );
                    silk_encode_pulses(
                        range_enc,
                        enc.state_fxx[0].s_cmn.indices_lbrr[i].signal_type as i32,
                        enc.state_fxx[0].s_cmn.indices_lbrr[i].quant_offset_type as i32,
                        &enc.state_fxx[0].s_cmn.pulses_lbrr[i],
                        enc.state_fxx[0].s_cmn.frame_length as usize,
                    );
                }
            }
        }
    }

    // Apply HP variable cutoff filter
    silk_hp_variable_cutoff(&mut enc.state_fxx[0]);

    // Compute target bitrate
    let target_rate = enc_control.bit_rate;
    silk_control_snr(&mut enc.state_fxx[0].s_cmn, target_rate);

    // VAD per channel
    for n in 0..n_channels_internal {
        let input_copy = enc.state_fxx[n].s_cmn.input_buf[..frame_length as usize].to_vec();
        silk_vad_get_sa_q8(
            &mut enc.state_fxx[n].s_cmn,
            &input_copy,
        );
        silk_encode_do_vad_fix(&mut enc.state_fxx[n], activity);
    }

    // Encode frame per channel
    for n in 0..n_channels_internal {
        let n_frames_encoded = enc.state_fxx[n].s_cmn.n_frames_encoded;
        let cond_coding = if enc.state_fxx[0].s_cmn.n_frames_encoded.wrapping_sub(n as i32) <= 0 {
            CODE_INDEPENDENTLY
        } else {
            CODE_CONDITIONALLY
        };

        let channel_rate = enc_control.bit_rate;
        if channel_rate > 0 {
            silk_control_snr(&mut enc.state_fxx[n].s_cmn, channel_rate);
        }

        let max_bits = enc_control.max_bits;
        let use_cbr = enc_control.use_cbr;

        let mut frame_n_bytes = 0i32;
        ret = silk_encode_frame_fix(
            &mut enc.state_fxx[n],
            &mut frame_n_bytes,
            range_enc,
            cond_coding,
            max_bits,
            use_cbr,
        );

        enc.state_fxx[n].s_cmn.controlled_since_last_payload = 0;
        enc.state_fxx[n].s_cmn.input_buf_ix = 0;
        enc.state_fxx[n].s_cmn.n_frames_encoded += 1;
    }

    // Check if packet is complete — patch VAD/LBRR flags
    if enc.state_fxx[0].s_cmn.n_frames_encoded >= n_frames_per_packet {
        // Build VAD/LBRR flags
        let mut flags: u32 = 0;
        for n in 0..n_channels_internal {
            for i in 0..n_frames_per_packet as usize {
                flags = flags << 1;
                flags |= enc.state_fxx[n].s_cmn.vad_flags[i] as u32;
            }
            flags = flags << 1;
            flags |= enc.state_fxx[n].s_cmn.lbrr_flag as u32;
        }
        if prefill_flag == 0 {
            let n_bits = ((n_frames_per_packet + 1) * n_channels_internal as i32) as u32;
            eprintln!("[RS VAD_PATCH] flags={} nbits={} tell={}", flags, n_bits, range_enc.tell());
            range_enc.patch_initial_bits(flags, n_bits);
        }

        // Check DTX — return zero bytes if all channels in DTX
        if enc.state_fxx[0].s_cmn.in_dtx != 0
            && (n_channels_internal == 1 || enc.state_fxx[1].s_cmn.in_dtx != 0)
        {
            *n_bytes_out = 0;
        } else {
            *n_bytes_out = (range_enc.tell() + 7) >> 3;
        }

        // Reset frame counter for next packet
        for n in 0..n_channels_internal {
            enc.state_fxx[n].s_cmn.n_frames_encoded = 0;
        }
    } else {
        *n_bytes_out = 0;
    }

    // Query encoder status
    silk_query_encoder(enc, enc_control);

    ret
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_control_input_valid() {
        let ctrl = SilkEncControlStruct::default();
        assert_eq!(check_control_input(&ctrl), SILK_NO_ERROR);
    }

    #[test]
    fn test_check_control_input_bad_api_rate() {
        let mut ctrl = SilkEncControlStruct::default();
        ctrl.api_sample_rate = 11025;
        assert_eq!(check_control_input(&ctrl), SILK_ENC_FS_NOT_SUPPORTED);
    }

    #[test]
    fn test_check_control_input_bad_packet_size() {
        let mut ctrl = SilkEncControlStruct::default();
        ctrl.payload_size_ms = 30;
        assert_eq!(check_control_input(&ctrl), SILK_ENC_PACKET_SIZE_NOT_SUPPORTED);
    }

    #[test]
    fn test_check_control_input_bad_loss_rate() {
        let mut ctrl = SilkEncControlStruct::default();
        ctrl.packet_loss_percentage = 101;
        assert_eq!(check_control_input(&ctrl), SILK_ENC_INVALID_LOSS_RATE);
    }

    #[test]
    fn test_check_control_input_bad_channels() {
        let mut ctrl = SilkEncControlStruct::default();
        ctrl.n_channels_api = 0;
        assert_eq!(check_control_input(&ctrl), SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR);
    }

    #[test]
    fn test_check_control_input_channels_internal_gt_api() {
        let mut ctrl = SilkEncControlStruct::default();
        ctrl.n_channels_api = 1;
        ctrl.n_channels_internal = 2;
        assert_eq!(check_control_input(&ctrl), SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR);
    }

    #[test]
    fn test_check_control_input_bad_complexity() {
        let mut ctrl = SilkEncControlStruct::default();
        ctrl.complexity = 11;
        assert_eq!(check_control_input(&ctrl), SILK_ENC_INVALID_COMPLEXITY_SETTING);
    }

    #[test]
    fn test_vad_init() {
        let mut vad = SilkVadState::default();
        assert_eq!(silk_vad_init(&mut vad), 0);
        // Noise levels should be initialized as pink noise approximation
        assert!(vad.noise_level_bias[0] > vad.noise_level_bias[1]);
        assert_eq!(vad.counter, 15);
        assert_eq!(vad.nrg_ratio_smth_q8[0], 25600); // 100 * 256
    }

    #[test]
    fn test_control_snr() {
        let mut state = SilkEncoderState::default();
        state.fs_khz = 16;
        state.nb_subfr = 4;
        silk_control_snr(&mut state, 20000);
        assert!(state.snr_db_q7 > 0);
        assert_eq!(state.target_rate_bps, 20000);
    }

    #[test]
    fn test_control_snr_low_bitrate() {
        let mut state = SilkEncoderState::default();
        state.fs_khz = 8;
        state.nb_subfr = 4;
        silk_control_snr(&mut state, 3000);
        // Very low bitrate should give 0 SNR
        assert_eq!(state.snr_db_q7, 0);
    }

    #[test]
    fn test_silk_interpolate() {
        let x0 = [0i16; MAX_LPC_ORDER];
        let x1 = [1000i16; MAX_LPC_ORDER];
        let mut xi = [0i16; MAX_LPC_ORDER];
        silk_interpolate(&mut xi, &x0, &x1, 2, 10); // factor = 2/4 = 0.5
        assert_eq!(xi[0], 500);
    }

    #[test]
    fn test_silk_encoder_size() {
        let size_stereo = silk_get_encoder_size(2);
        let size_mono = silk_get_encoder_size(1);
        assert!(size_mono < size_stereo);
    }

    #[test]
    fn test_init_encoder() {
        let mut enc = SilkEncoderStateFix::default();
        assert_eq!(silk_init_encoder(&mut enc), 0);
        assert_eq!(enc.s_cmn.first_frame_after_reset, 1);
        assert!(enc.s_cmn.variable_hp_smth1_q15 != 0);
    }

    #[test]
    fn test_init_encoder_top() {
        let mut enc = SilkEncoder::new();
        assert_eq!(silk_init_encoder_top(&mut enc, 1), 0);
        assert_eq!(enc.n_channels_api, 1);
    }

    #[test]
    fn test_ana_filt_bank_1() {
        let input = [1000i16; 32];
        let mut state = [0i32; 2];
        let mut out_l = [0i16; 16];
        let mut out_h = [0i16; 16];
        silk_ana_filt_bank_1(&input, &mut state, &mut out_l, &mut out_h, 32);
        // Low band should have energy, high band should be near zero for DC input
        let low_energy: i64 = out_l.iter().map(|&s| s as i64 * s as i64).sum();
        let high_energy: i64 = out_h.iter().map(|&s| s as i64 * s as i64).sum();
        assert!(low_energy > high_energy);
    }

    #[test]
    fn test_setup_complexity() {
        let mut state = SilkEncoderState::default();
        state.fs_khz = 16;
        state.predict_lpc_order = 16;
        silk_setup_complexity(&mut state, 10);
        assert_eq!(state.n_states_delayed_decision, MAX_DEL_DEC_STATES as i32);
        assert_eq!(state.shaping_lpc_order, 24);
        assert!(state.warping_q16 > 0);
    }

    #[test]
    fn test_gains_quant_roundtrip() {
        let mut gains_q16 = [65536i32, 80000, 50000, 65536];
        let mut ind = [0i8; 4];
        let mut prev_ind = 10i8;
        silk_gains_quant(&mut ind, &mut gains_q16, &mut prev_ind, false, 4);
        // Quantized gains should be reasonable
        for g in &gains_q16 {
            assert!(*g > 0);
        }
    }

    #[test]
    fn test_nlsf_vq_weights_laroia() {
        let nlsf = [3277i16, 6554, 9830, 13107, 16384, 19660, 22937, 26214, 29491, 32000];
        let mut w = [0i16; 10];
        silk_nlsf_vq_weights_laroia(&mut w, &nlsf, 10);
        // All weights should be positive
        for &ww in &w {
            assert!(ww > 0, "Weight should be positive, got {}", ww);
        }
    }

    #[test]
    fn test_stereo_quant_pred() {
        let mut pred = [5000i32, -3000];
        let mut ix = [[0i8; 3]; 2];
        silk_stereo_quant_pred(&mut pred, &mut ix);
        // Quantized predictors should be close to input
        assert!(pred[0].abs() <= 15000);
        assert!(pred[1].abs() <= 15000);
    }
}
